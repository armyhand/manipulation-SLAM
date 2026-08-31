"""接触式轮廓映射离线仿真实验。

本脚本用于验证一个简化的 contact-SLAM / contact mapping 任务：
已知可移动物体的精确轮廓、障碍物参考轮廓 ``obstacle_ref`` 以及一个
包含真实障碍物的探索边界，机器人只能沿 ``+X/-X/+Y/-Y`` 四个方向离散
运动。运动过程中如果可移动物体即将与真实障碍物产生面积重叠，则把
第一处边界接触视为一次触觉观测，并用该观测逐步确定真实障碍物每条边
的位置。

注意：``OBSTACLE_TRUE`` 在这里只作为仿真 oracle 和误差评估使用。算法
本身不使用 ``obstacle_true = scale * obstacle_ref`` 这样的整体缩放关系；
每条参考边都有独立的位置不确定区间。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
from scipy.optimize import linprog


OBSTACLE_TRUE = [np.array(
    [
        [0.0, 0.0],
        [0.0, -80.0],
        [20.0, -80.0],
        [20.0, -20.0],
        [100.0, -20.0],
        [110.0, -80.0],
        [120.0, -80.0],
        [120.0, -20.0],
        [200.0, -20.0],
        [200.0, -80.0],
        [220.0, -80.0],
        [220.0, -20.0],
        [300.0, -20.0],
        [300.0, -80.0],
        [320.0, -80.0],
        [320.0, 0.0]
    ],
    dtype=float,
)]
# OBSTACLE_TRUE = [
#             np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 100.0], [0.0, 100.0]]),
#             np.array([[100.0, 0.0], [120.0, 0.0], [120.0, 100.0], [100.0, 100.0]]),
#             np.array([[200.0, 0.0], [220.0, 0.0], [220.0, 100.0], [200.0, 100.0]]),
#             np.array([[300.0, 0.0], [320.0, 0.0], [320.0, 100.0], [300.0, 100.0]]),
#         ]

# 参考轮廓：算法只使用它的拓扑顺序和边方向，不使用“0.7 倍缩放”这一真值关系。
OBSTACLE_REF = [0.1 * np.array(
    [
        [0.0, 0.0],
        [0.0, -100.0],
        [20.0, -100.0],
        [20.0, -20.0],
        [100.0, -20.0],
        [110.0, -80.0],
        [120.0, -80.0],
        [120.0, -20.0],
        [200.0, -20.0],
        [200.0, -80.0],
        [220.0, -80.0],
        [220.0, -20.0],
        [300.0, -20.0],
        [300.0, -80.0],
        [320.0, -80.0],
        [320.0, 0.0]
    ],
    dtype=float,
) + np.array([100.0, 100.0], dtype=float)]

# 可移动物体轮廓，坐标系原点是物体参考点。这里继承源代码中的方形块设定。
MOVABLE_OBJECTS = [
    np.array(
        [[-10.0, 10.0], [-10.0, -10.0], [10.0, -10.0], [10.0, 10.0]],
        dtype=float,
    )
]

# 源代码风格的四方向离散动作。后续所有执行轨迹都必须由这些方向的线段组成。
ACTIONS = [
    np.array([1.0, 0.0]),
    np.array([-1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([0.0, -1.0]),
]


@dataclass(frozen=True)
class Edge:  # 每条障碍物边的统一描述，后续接触判定和支撑线估计都会使用。
    """单条多边形边的几何描述。

    对任意方向的边，都统一写成支撑线形式：
    ``dot(point, outward) = const``。其中 ``outward`` 是单位外法线，
    ``tangent`` 是沿边方向的单位向量，``span`` 是边端点在切向方向上的
    投影范围，用于判断接触侧是否与该边段真正重叠。
    """

    index: int
    contour_index: int
    local_index: int
    p0: np.ndarray
    p1: np.ndarray
    tangent: np.ndarray
    outward: np.ndarray  # 外法线方向；探测动作取 -outward，表示从外侧推向该边。
    axis: str
    const: float
    span: tuple[float, float]


@dataclass
class SupportGroup:
    """待估计的一条障碍边支撑线参数。

    早期版本曾把共线边合并成一组；当前版本为了满足“某条边可以独立变长
    或变短”的约束，实际上每条参考边单独对应一个 ``SupportGroup``。

    ``ref_const`` 是已经放入探索边界内的形状先验支撑线常数 ``dot(p, normal)``；
    ``min_const`` / ``max_const`` 是由探索边界给出的初始可行范围；
    ``observed_const`` 是接触后确定的真实支撑线位置。
    """

    key: str
    axis: str
    ref_const: float
    edge_indices: list[int]
    min_const: float
    max_const: float
    observed_const: float | None = None


@dataclass
class Probe:
    """一次“计划探测”的几何目标。

    Probe 表示算法希望从 ``start`` 沿 ``action`` 去接触 ``edge_index``。
    但实际执行时，机器人会沿四方向折线运动；如果途中先碰到其它边，
    那么途中第一处接触会成为真实观测。
    """

    group_key: str
    edge_index: int
    action: np.ndarray
    start: np.ndarray
    expected_contact: np.ndarray


@dataclass
class ContactObservation:
    """一次实际接触观测。

    ``axis_const`` 是由接触反推出的障碍边支撑线位置；例如接触到竖直边时
    它就是真实的 x 坐标。``candidate_edge_indices`` 是算法根据当前不确定
    性判断出的可能接触边集合，可能包含多条边。
    """

    ok: bool
    action: np.ndarray
    start: np.ndarray
    contact_ref: np.ndarray | None = None
    distance: float = math.inf
    plane_const: float | None = None
    axis_const: float | None = None
    overlap_area: float = 0.0
    max_path_overlap_area: float = 0.0
    true_edge_indices: list[int] = field(default_factory=list)
    candidate_edge_indices: list[int] = field(default_factory=list)
    message: str = ""


@dataclass
class StepRecord:
    """单步探索的完整日志。

    ``trajectory_points`` 保存本步实际执行的连续四方向折线：
    ``move_start -> ... -> contact_ref``。这能用于检查当前步起点是否等于
    上一步终点，以及路径中是否存在对角线运动。
    """

    step: int
    strategy: str
    group_key: str
    target_edge: int
    action: list[float]
    move_start: list[float]
    start: list[float]
    probe_start: list[float]
    contact_ref: list[float]
    trajectory_points: list[list[float]]
    continuity_gap: float
    distance: float
    transit_steps: int
    approach_steps: int
    total_steps: int
    true_edges: list[int]
    candidate_edges: list[int]
    hypothesis_count: int
    contact_pair_converged: bool
    overlap_area: float
    max_path_overlap_area: float
    max_vertex_error: float
    mean_vertex_error: float
    figure: str


@dataclass(frozen=True)
class ExplorationBoundary:
    """已知探索边界。

    机器人参考点运动到该边界以外时，可认为本次探索方向结束。仿真中该边界
    由真实障碍物外接矩形扩张 50 mm 得到；真实实验里可以替换为外部给定边界。
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                [self.x_min, self.y_min],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                [self.x_min, self.y_max],
            ],
            dtype=float,
        )


@dataclass
class Hypothesis:
    """一个接触对解释假设。

    ``observed`` 保存该假设认为已经被确定的边支撑线常数。真实系统无法在一次
    接触中直接知道是哪条边，因此同一次观测可能派生出多个 Hypothesis。
    后续观测会不断筛除与新接触不一致的假设。
    """

    observed: dict[int, float] = field(default_factory=dict)
    contact_pairs: list[tuple[int, int]] = field(default_factory=list)

    def copy(self) -> "Hypothesis":
        return Hypothesis(
            observed=dict(self.observed),
            contact_pairs=list(self.contact_pairs),
        )


def unit(v: np.ndarray) -> np.ndarray:
    """返回向量单位化结果，避免后续法线/切线计算重复写归一化逻辑。"""

    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("zero-length vector")
    return v / norm


def signed_area(poly: np.ndarray) -> float:
    """计算多边形有向面积。

    本实验要求障碍物顶点逆时针排列，因此面积应为正值。这个检查可以避免
    外法线方向被反过来。
    """

    shifted = np.roll(poly, -1, axis=0)
    return 0.5 * float(np.sum(poly[:, 0] * shifted[:, 1] - poly[:, 1] * shifted[:, 0]))


def as_contours(obstacle: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    """把单轮廓或多轮廓障碍统一转换为轮廓列表。"""

    if isinstance(obstacle, np.ndarray):
        return [np.asarray(obstacle, dtype=float)]
    return [np.asarray(poly, dtype=float) for poly in obstacle]


def flatten_contours(obstacle: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """把多个轮廓顶点按轮廓顺序拼成一个扁平数组，便于误差统计。"""

    contours = as_contours(obstacle)
    if not contours:
        return np.empty((0, 2), dtype=float)
    return np.vstack(contours)


def split_flat_vertices(vertices: np.ndarray, counts: list[int]) -> list[np.ndarray]:
    """按照每个轮廓的顶点数拆分扁平顶点数组。"""

    offsets = np.cumsum([0] + counts)
    return [
        np.asarray(vertices[start:end], dtype=float)
        for start, end in zip(offsets[:-1], offsets[1:])
    ]


def edge_records(obstacle: np.ndarray | list[np.ndarray]) -> list[Edge]:
    """把一个或多个多边形轮廓转换为 ``Edge`` 列表。

    对每条边记录端点、单位切向量、外法线、法线支撑线常数和切向跨度。
    由于任务中的约束是“对应边方向不变”，这些边方向会作为后续未知支撑线
    估计的基础。这里不再要求边必须水平或竖直。
    """

    edges: list[Edge] = []
    for contour_i, poly in enumerate(as_contours(obstacle)):
        area = signed_area(poly)
        if area <= 0.0:
            raise ValueError("Every obstacle contour must be counter-clockwise.")

        for local_i, p0 in enumerate(poly):
            p1 = poly[(local_i + 1) % len(poly)]
            tangent = unit(p1 - p0)
            outward = np.array([tangent[1], -tangent[0]], dtype=float)
            const = float(np.dot(p0, outward))
            span_values = (float(np.dot(p0, tangent)), float(np.dot(p1, tangent)))
            span = (min(span_values), max(span_values))
            if abs(tangent[0]) < 1e-9:
                axis = "vertical"
            elif abs(tangent[1]) < 1e-9:
                axis = "horizontal"
            else:
                axis = "oblique"

            edges.append(
                Edge(
                    index=len(edges),
                    contour_index=contour_i,
                    local_index=local_i,
                    p0=p0.copy(),
                    p1=p1.copy(),
                    tangent=tangent,
                    outward=outward,
                    axis=axis,
                    const=const,
                    span=span,
                )
            )
    return edges


def make_exploration_boundary(
    obstacle_true: np.ndarray | list[np.ndarray],
    margin: float = 50.0,
) -> ExplorationBoundary:
    """构造包含真实障碍物的矩形探索边界。

    这里为了仿真方便用 ``obstacle_true`` 生成边界。算法后续只知道这个边界，
    不直接使用真实顶点来推断障碍物轮廓。
    """

    points = flatten_contours(obstacle_true)
    x_min, y_min = np.min(points, axis=0) - margin
    x_max, y_max = np.max(points, axis=0) + margin
    return ExplorationBoundary(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
    )


def normalize_ref_to_boundary(
    obstacle_ref: np.ndarray | list[np.ndarray],
    boundary: ExplorationBoundary,
    fill_ratio: float = 0.55,
) -> np.ndarray | list[np.ndarray]:
    """把 ``obstacle_ref`` 归一化到探索边界内部，作为初始形状先验。

    真实探索时 ``obstacle_ref`` 只表示“边的拓扑顺序、方向和大致相对形状”，
    不应该携带世界坐标中的绝对位置或绝对尺度。因此无论参考轮廓被放在哪里、
    被整体缩放多少倍，都先做统一尺度缩放和平移，使它落在已知探索边界内。

    这里采用等比例缩放，避免改变各条边的方向矢量；再把参考轮廓中心移动到
    探索边界中心。``fill_ratio`` 小于 1，可给后续支撑线调整留出余量。
    """

    ref_contours = as_contours(obstacle_ref)
    flat_ref = flatten_contours(ref_contours)
    ref_min = np.min(flat_ref, axis=0)
    ref_max = np.max(flat_ref, axis=0)
    ref_center = 0.5 * (ref_min + ref_max)
    ref_size = ref_max - ref_min
    if np.any(ref_size <= 1e-9):
        raise ValueError("obstacle_ref must have non-zero width and height.")

    boundary_size = np.array(
        [
            boundary.x_max - boundary.x_min,
            boundary.y_max - boundary.y_min,
        ],
        dtype=float,
    )
    scale = fill_ratio * float(np.min(boundary_size / ref_size))
    boundary_center = np.array(
        [
            0.5 * (boundary.x_min + boundary.x_max),
            0.5 * (boundary.y_min + boundary.y_max),
        ],
        dtype=float,
    )
    prior_contours = [(poly - ref_center) * scale + boundary_center for poly in ref_contours]
    prior_flat = flatten_contours(prior_contours)

    # 数值上再做一次极小裁剪式平移，确保所有顶点严格位于探索边界内。
    prior_min = np.min(prior_flat, axis=0)
    prior_max = np.max(prior_flat, axis=0)
    shift = np.zeros(2, dtype=float)
    margin = 1e-6
    if prior_min[0] < boundary.x_min + margin:
        shift[0] += boundary.x_min + margin - prior_min[0]
    if prior_max[0] > boundary.x_max - margin:
        shift[0] -= prior_max[0] - (boundary.x_max - margin)
    if prior_min[1] < boundary.y_min + margin:
        shift[1] += boundary.y_min + margin - prior_min[1]
    if prior_max[1] > boundary.y_max - margin:
        shift[1] -= prior_max[1] - (boundary.y_max - margin)
    shifted = [poly + shift for poly in prior_contours]
    return shifted if isinstance(obstacle_ref, list) else shifted[0]


def group_support_lines(
    ref_edges: list[Edge],
    boundary: ExplorationBoundary,
    objects: list[np.ndarray] | None = None,
) -> dict[str, SupportGroup]:
    """为每一条参考边建立独立的不确定支撑线。

    支撑线常数采用 ``dot(point, outward)``。探索边界是一个已知矩形，因此
    某条边的支撑线常数初始范围可以通过把边界四个角点投影到该法线方向得到。
    同时还要保证这条边存在可接触位姿：接触时 movable object 的参考点不能
    已经越过探索边界，否则该边虽然作为顶点坐标可能在边界内，但实际不可碰撞。

    注意这里没有把同一条直线上的多段边合并，因为用户要求单条边可以独立
    相对 ``obstacle_ref`` 变化。
    """

    result: dict[str, SupportGroup] = {}
    boundary_vertices = boundary.as_array()
    if objects is None:
        objects = MOVABLE_OBJECTS
    for edge in ref_edges:
        boundary_supports = boundary_vertices @ edge.outward ##计算边界四个顶点在该边外法线方向上的投影，这是该线段端点的范围约束之一。
        vertex_min_const = float(np.min(boundary_supports))
        vertex_max_const = float(np.max(boundary_supports))

        # 接触位姿满足 dot(ref_point, outward) = const - support_min(object, outward)。
        # 因此 const 的范围还必须让 ref_point 的法向投影落在探索边界投影范围内。
        object_support = support_min_value(objects, edge.outward)
        contact_min_const = vertex_min_const + object_support
        contact_max_const = vertex_max_const + object_support
        min_const = max(vertex_min_const, contact_min_const)
        max_const = min(vertex_max_const, contact_max_const)
        if min_const > max_const:
            raise ValueError(
                f"Edge {edge.index} has no support-line position that is both "
                "inside the exploration boundary and contactable by movable_objects."
            )
        key = f"e{edge.index:02d}_{edge.axis}_{edge.const:.3f}"
        result[key] = SupportGroup(
            key=key,
            axis=edge.axis,
            ref_const=edge.const,
            edge_indices=[edge.index],
            min_const=min_const,
            max_const=max_const,
        )
    return result


def clone_groups(groups: dict[str, SupportGroup]) -> dict[str, SupportGroup]:
    """复制支撑线组，避免多假设评估时互相污染。"""

    return {
        key: SupportGroup(
            key=group.key,
            axis=group.axis,
            ref_const=group.ref_const,
            edge_indices=list(group.edge_indices),
            min_const=group.min_const,
            max_const=group.max_const,
            observed_const=group.observed_const,
        )
        for key, group in groups.items()
    }


def groups_from_hypothesis(
    base_groups: dict[str, SupportGroup],
    ref_edges: list[Edge],
    hypothesis: Hypothesis,
) -> dict[str, SupportGroup]:
    """把一个假设转换成当前估计函数可用的 ``SupportGroup`` 集合。"""

    groups = clone_groups(base_groups)
    edge_to_group = edge_group_map(groups)
    for edge_i, const in hypothesis.observed.items():
        edge_to_group[edge_i].observed_const = const
    return groups


def consensus_observed_edges(hypotheses: list[Hypothesis], tol: float = 1e-6) -> dict[int, float]:
    """提取所有存活假设都一致确认的边位置。

    只有当每个假设都包含某条边，且支撑线常数在容差内一致时，该边才被视作
    真正收敛。这样不会因为单次多候选接触而过早固定错误边。
    """

    if not hypotheses:
        return {}
    common_edges = set(hypotheses[0].observed)
    for hyp in hypotheses[1:]:
        common_edges &= set(hyp.observed)

    result: dict[int, float] = {}
    for edge_i in common_edges:
        values = [hyp.observed[edge_i] for hyp in hypotheses]
        if max(values) - min(values) <= tol:
            result[edge_i] = float(np.mean(values))
    return result


def consensus_groups(
    base_groups: dict[str, SupportGroup],
    hypotheses: list[Hypothesis],
) -> dict[str, SupportGroup]:
    """生成只包含“所有假设一致确认边”的支撑线集合。"""

    groups = clone_groups(base_groups)
    edge_to_group = edge_group_map(groups)
    for edge_i, const in consensus_observed_edges(hypotheses).items():
        edge_to_group[edge_i].observed_const = const
    return groups


def hypothesis_signature(hypothesis: Hypothesis, digits: int = 6) -> tuple[tuple[int, float], ...]:
    """用于合并重复假设的稳定签名。"""

    return tuple(sorted((edge_i, round(float(const), digits)) for edge_i, const in hypothesis.observed.items()))


def hypothesis_constants(
    ref_edges: list[Edge],
    base_groups: dict[str, SupportGroup],
    hypothesis: Hypothesis,
) -> list[float]:
    """返回某个假设下每条边的支撑线常数。"""

    hyp_groups = groups_from_hypothesis(base_groups, ref_edges, hypothesis)
    return current_support_constants(ref_edges, hyp_groups)


def adjacent_edge_indices(ref_edges: list[Edge]) -> dict[int, tuple[int, int]]:
    """返回每条边在同一轮廓内的前一条/后一条边索引。"""

    by_contour: dict[int, list[Edge]] = {}
    for edge in ref_edges:
        by_contour.setdefault(edge.contour_index, []).append(edge)

    result: dict[int, tuple[int, int]] = {}
    for contour_edges in by_contour.values():
        ordered = sorted(contour_edges, key=lambda e: e.local_index)
        n = len(ordered)
        for i, edge in enumerate(ordered):
            result[edge.index] = (ordered[(i - 1) % n].index, ordered[(i + 1) % n].index)
    return result


def edge_span_from_polygon(
    poly: np.ndarray,
    edge: Edge,
    ref_edges: list[Edge],
) -> tuple[float, float]:
    """由当前估计多边形计算某条边段的实际切向跨度。"""

    p0 = poly[edge.index]
    next_i = adjacent_edge_indices(ref_edges)[edge.index][1]
    p1 = poly[next_i]
    vals = (float(np.dot(p0, edge.tangent)), float(np.dot(p1, edge.tangent)))
    return min(vals), max(vals)


def edge_segment_from_polygon(poly: np.ndarray, edge: Edge, ref_edges: list[Edge]) -> tuple[np.ndarray, np.ndarray]:
    """从当前估计顶点中取出某条参考边对应的实际线段。"""

    next_i = adjacent_edge_indices(ref_edges)[edge.index][1]
    return poly[edge.index], poly[next_i]


def hypothesis_is_consistent(
    hypothesis: Hypothesis,
    observations: list[ContactObservation],
    ref_edges: list[Edge],
    base_groups: dict[str, SupportGroup],
    objects: list[np.ndarray],
    boundary: ExplorationBoundary,
    line_tol: float,
) -> bool:
    """检查一个多接触对假设是否满足全局几何约束。

    这里不能把未观测边直接固定在参考位置后再检查。未观测边仍然有一整段
    支撑线可行区间，若只检查单个代表轮廓，会把“真实可行但代表轮廓没碰到”
    的候选接触边提前剔除。因此一致性检查采用存在性 LP：只要存在一组支撑线
    常数同时满足边界、边长、非自交割平面和所有接触切向重叠约束，该假设就
    暂时保留，等待后续观测继续消歧。
    """

    hyp_groups = groups_from_hypothesis(base_groups, ref_edges, hypothesis)
    extra_rows: list[np.ndarray] = []
    extra_bs: list[float] = []
    try:
        vertex_coeffs = vertex_linear_coefficients(ref_edges)
    except ValueError:
        return False
    for step_i, edge_i in hypothesis.contact_pairs:
        if step_i >= len(observations):
            return False
        obs = observations[step_i]
        if obs.contact_ref is None:
            return False
        edge = ref_edges[edge_i]
        if not action_matches_edge(obs.action, edge):
            return False
        observed_const = observed_const_for_edge(obs, edge, objects)
        if abs(observed_const - hypothesis.observed.get(edge_i, observed_const)) > line_tol:
            return False
        obj_interval = support_points_lateral_interval(
            objects,
            obs.contact_ref,
            edge.outward,
            edge.tangent,
        )
        start_coeff, end_coeff = edge_endpoint_coeffs(edge_i, ref_edges, vertex_coeffs)
        start_t = edge.tangent @ start_coeff
        end_t = edge.tangent @ end_coeff
        # 接触点只要求“存在”落在该边段切向跨度内：
        # start_t <= object_t_max 且 end_t >= object_t_min。
        # 这比用当前蓝色代表轮廓检查更合理，因为未观测相邻边还可移动。
        extra_rows.append(start_t)
        extra_bs.append(obj_interval[1] + line_tol)
        extra_rows.append(-end_t)
        extra_bs.append(-obj_interval[0] + line_tol)

    constants = solve_feasible_support_constants(
        ref_edges,
        hyp_groups,
        boundary,
        objects,
        extra_rows=extra_rows,
        extra_bs=extra_bs,
    )
    if constants is None:
        return False

    # 候选接触对不仅要能解释“最终接触点”，还必须解释“为什么之前没有接触”。
    # 例如 step=0 的真实接触在 [-10,-40]，若把它误解释成较后面的 edge 4，
    # 可行轮廓会把 vertex00/01 放到 x<0；此时 movable object 从起点推进到
    # contact_ref 的途中已经穿入该轮廓，违反“第一次接触”的观测语义。
    poly = polygon_from_supports(ref_edges, constants)
    obstacle_contours = contours_from_flat_vertices(poly, ref_edges)
    for step_i, _ in hypothesis.contact_pairs:
        obs = observations[step_i]
        if obs.contact_ref is None:
            return False
        if max_overlap_along_path(objects, [obs.start, obs.contact_ref], obstacle_contours) > 1e-8:
            return False
    return True


def prune_duplicate_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """合并观测边集合完全相同的重复假设。"""

    unique: dict[tuple[tuple[int, float], ...], Hypothesis] = {}
    for hyp in hypotheses:
        unique.setdefault(hypothesis_signature(hyp), hyp)
    return list(unique.values())


def hypothesis_signature_set(hypotheses: list[Hypothesis]) -> set[tuple[tuple[int, float], ...]]:
    """返回一组假设的签名集合，用于判断某次观测是否提供了新信息。"""

    return {hypothesis_signature(hyp) for hyp in hypotheses}


def update_hypotheses_with_observation(
    hypotheses: list[Hypothesis],
    obs: ContactObservation,
    observations: list[ContactObservation],
    ref_edges: list[Edge],
    base_groups: dict[str, SupportGroup],
    objects: list[np.ndarray],
    boundary: ExplorationBoundary,
    line_tol: float,
    span_pad: float,
    step_index: int,
) -> tuple[list[Hypothesis], list[int]]:
    """用一次接触观测扩展并剪枝多假设集合。

    对每个存活假设，先在该假设对应的支撑线约束下计算候选接触边；然后对每条
    候选边生成一个新假设，并写入该边的支撑线常数。如果候选边与该假设中已有
    常数冲突，则该分支被丢弃。
    """

    expanded: list[Hypothesis] = []
    all_candidates: set[int] = set()
    for hyp in hypotheses:
        hyp_groups = groups_from_hypothesis(base_groups, ref_edges, hyp)
        candidates = candidate_edges_for_observation(
            obs,
            ref_edges,
            hyp_groups,
            objects,
            line_tol=line_tol,
            span_pad=span_pad,
        )
        all_candidates.update(candidates)
        for edge_i in candidates:
            edge = ref_edges[edge_i]
            const = observed_const_for_edge(obs, edge, objects)
            group = edge_group_map(hyp_groups)[edge_i]
            if not (group.min_const - line_tol <= const <= group.max_const + line_tol):
                continue
            if edge_i in hyp.observed and abs(hyp.observed[edge_i] - const) > line_tol:
                continue
            child = hyp.copy()
            # 同一条边已经被该假设观测过时，容差内的新观测只说明重复接触，
            # 不应覆盖原支撑线常数；否则斜边/近似候选会因微小差异反复改变
            # hypothesis_signature，造成策略误判为“获得了新信息”。
            if edge_i not in child.observed:
                child.observed[edge_i] = const
            child.contact_pairs.append((step_index, edge_i))
            if not hypothesis_is_consistent(
                child,
                observations,
                ref_edges,
                base_groups,
                objects,
                boundary,
                line_tol,
            ):
                continue
            expanded.append(child)

    return prune_duplicate_hypotheses(expanded), sorted(all_candidates)


def estimate_from_hypotheses(
    ref_edges: list[Edge],
    base_groups: dict[str, SupportGroup],
    hypotheses: list[Hypothesis],
    boundary: ExplorationBoundary | None = None,
    objects: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, list[float], float]:
    """由多假设集合生成当前用于绘图/误差统计的平均轮廓。"""

    if not hypotheses:
        groups = clone_groups(base_groups)
        if boundary is not None and objects is not None:
            return representative_feasible_polygon(ref_edges, groups, boundary, objects)
        return constrained_estimated_polygon(ref_edges, groups)
    # 直接展开唯一一次使用的假设轮廓列表推导，减少中间包装函数。
    polygons = [
        representative_feasible_polygon(
            ref_edges,
            groups_from_hypothesis(base_groups, ref_edges, hyp),
            boundary,
            objects,
        )[0]
        for hyp in hypotheses
    ]
    estimated = np.mean(np.stack(polygons, axis=0), axis=0)
    constants = [
        float(np.mean([np.dot(poly[i], ref_edges[i].outward) for poly in polygons]))
        for i in range(len(ref_edges))
    ]
    return estimated, constants, 1.0 if is_simple_obstacle(estimated, ref_edges) else 0.0


def vertex_uncertainty_ellipses_from_hypotheses(
    ref_edges: list[Edge],
    base_groups: dict[str, SupportGroup],
    hypotheses: list[Hypothesis],
    boundary: ExplorationBoundary | None = None,
    objects: list[np.ndarray] | None = None,
) -> list[tuple[np.ndarray, float, float]]:
    """根据所有存活假设实时计算顶点范围椭圆。"""

    if not hypotheses:
        return vertex_uncertainty_ellipses(ref_edges, base_groups, boundary, objects)

    # 每个假设内部仍有未观测边，这些边应保留探索边界给出的支撑线区间。
    # 因此不能只把每个假设变成一个预测多边形单点；要先计算该假设下的顶点
    # 区间，再把所有假设的区间取并集。
    per_hyp_ranges: list[list[tuple[float, float, float, float]]] = []
    for hyp in hypotheses:
        hyp_groups = groups_from_hypothesis(base_groups, ref_edges, hyp)
        ranges: list[tuple[float, float, float, float]] = []
        for center, radius_x, radius_y in vertex_uncertainty_ellipses(
            ref_edges,
            hyp_groups,
            boundary,
            objects,
        ):
            ranges.append(
                (
                    float(center[0] - radius_x),
                    float(center[0] + radius_x),
                    float(center[1] - radius_y),
                    float(center[1] + radius_y),
                )
            )
        per_hyp_ranges.append(ranges)

    ellipses: list[tuple[np.ndarray, float, float]] = []
    for vertex_i in range(len(ref_edges)):
        x_min = min(ranges[vertex_i][0] for ranges in per_hyp_ranges)
        x_max = max(ranges[vertex_i][1] for ranges in per_hyp_ranges)
        y_min = min(ranges[vertex_i][2] for ranges in per_hyp_ranges)
        y_max = max(ranges[vertex_i][3] for ranges in per_hyp_ranges)
        center = np.array([0.5 * (x_min + x_max), 0.5 * (y_min + y_max)], dtype=float)
        ellipses.append(
            (
                center,
                max(0.6, 0.5 * float(x_max - x_min)),
                max(0.6, 0.5 * float(y_max - y_min)),
            )
        )
    return ellipses


def vertex_ranges_from_hypotheses(
    ref_edges: list[Edge],
    base_groups: dict[str, SupportGroup],
    hypotheses: list[Hypothesis],
    boundary: ExplorationBoundary | None = None,
    objects: list[np.ndarray] | None = None,
) -> list[tuple[float, float, float, float]]:
    """返回每个顶点真实可行 x/y 范围，不加入绘图用最小椭圆半径。"""

    source_hypotheses = hypotheses or [Hypothesis()]
    per_hyp_ranges: list[list[tuple[float, float, float, float]]] = []
    for hyp in source_hypotheses:
        groups = groups_from_hypothesis(base_groups, ref_edges, hyp)
        ranges = (
            global_feasible_vertex_bounds(ref_edges, groups, boundary, objects=objects)
            if boundary is not None
            else None
        )
        if ranges is None:
            ranges = []
            for center, radius_x, radius_y in vertex_uncertainty_ellipses(
                ref_edges,
                groups,
                boundary,
                objects,
            ):
                ranges.append(
                    (
                        float(center[0] - radius_x),
                        float(center[0] + radius_x),
                        float(center[1] - radius_y),
                        float(center[1] + radius_y),
                    )
                )
        per_hyp_ranges.append(ranges)

    result: list[tuple[float, float, float, float]] = []
    for vertex_i in range(len(ref_edges)):
        result.append(
            (
                min(ranges[vertex_i][0] for ranges in per_hyp_ranges),
                max(ranges[vertex_i][1] for ranges in per_hyp_ranges),
                min(ranges[vertex_i][2] for ranges in per_hyp_ranges),
                max(ranges[vertex_i][3] for ranges in per_hyp_ranges),
            )
        )
    return result


def vertex_uncertainty_summary(
    ref_edges: list[Edge],
    base_groups: dict[str, SupportGroup],
    hypotheses: list[Hypothesis],
    boundary: ExplorationBoundary | None = None,
    objects: list[np.ndarray] | None = None,
) -> dict:
    """输出每个顶点范围椭圆的半径和整体最大直径，便于数值验证收敛程度。"""

    vertices = []
    for x_min, x_max, y_min, y_max in vertex_ranges_from_hypotheses(
        ref_edges,
        base_groups,
        hypotheses,
        boundary,
        objects,
    ):
        vertices.append(
            {
                "center": [0.5 * (x_min + x_max), 0.5 * (y_min + y_max)],
                "radius_x": float(0.5 * (x_max - x_min)),
                "radius_y": float(0.5 * (y_max - y_min)),
                "diameter_x": float(x_max - x_min),
                "diameter_y": float(y_max - y_min),
            }
        )
    return {
        "max_vertex_range_diameter_mm": float(
            max((max(item["diameter_x"], item["diameter_y"]) for item in vertices), default=0.0)
        ),
        "vertices": vertices,
    }


def print_step_diagnostics(
    ref_edges: list[Edge],
    base_groups: dict[str, SupportGroup],
    hypotheses: list[Hypothesis],
    objects: list[np.ndarray],
    boundary: ExplorationBoundary,
    record: StepRecord,
) -> None:
    """每次探索后打印候选接触对和所有顶点当前范围。"""

    estimated_poly, _, _ = estimate_from_hypotheses(
        ref_edges,
        base_groups,
        hypotheses,
        boundary,
        objects,
    )
    candidate_pairs: list[tuple[int, int | None, int | None, int]] = []
    for edge_i in record.candidate_edges:
        seg0, seg1 = edge_segment_from_polygon(estimated_poly, ref_edges[edge_i], ref_edges)
        movable_pairs = movable_edge_contact_pairs(
            objects,
            np.array(record.contact_ref, dtype=float),
            seg0,
            seg1,
        )
        if movable_pairs:
            candidate_pairs.extend((record.step, obj_i, obj_edge_i, edge_i) for obj_i, obj_edge_i in movable_pairs)
        else:
            candidate_pairs.append((record.step, None, None, edge_i))
    print(
        f"\n[{record.strategy}] step={record.step} "
        f"contact_ref={np.round(record.contact_ref, 3).tolist()} "
        f"oracle_true_edges={record.true_edges} "
        f"algorithm_candidate_edges={record.candidate_edges} "
        f"algorithm_candidate_contact_pairs=(step, movable_obj, movable_edge, obstacle_edge) {candidate_pairs} "
        f"hypotheses={record.hypothesis_count}"
    )
    for vertex_i, (edge, (x_min, x_max, y_min, y_max)) in enumerate(
        zip(
            ref_edges,
            vertex_ranges_from_hypotheses(
                ref_edges,
                base_groups,
                hypotheses,
                boundary,
                objects,
            ),
        )
    ):
        print(
            f"  vertex {vertex_i:02d} "
            f"(contour={edge.contour_index}, local={edge.local_index}): "
            f"x=[{x_min:.3f}, {x_max:.3f}], "
            f"y=[{y_min:.3f}, {y_max:.3f}]"
        )


def print_initial_vertex_ranges(
    strategy: str,
    ref_edges: list[Edge],
    base_groups: dict[str, SupportGroup],
    hypotheses: list[Hypothesis],
    objects: list[np.ndarray],
    boundary: ExplorationBoundary,
) -> None:
    """在第一次探索前打印初始顶点范围。"""

    print(f"\n[{strategy}] initial vertex ranges")
    for vertex_i, (edge, (x_min, x_max, y_min, y_max)) in enumerate(
        zip(
            ref_edges,
            vertex_ranges_from_hypotheses(
                ref_edges,
                base_groups,
                hypotheses,
                boundary,
                objects,
            ),
        )
    ):
        print(
            f"  vertex {vertex_i:02d} "
            f"(contour={edge.contour_index}, local={edge.local_index}): "
            f"x=[{x_min:.3f}, {x_max:.3f}], "
            f"y=[{y_min:.3f}, {y_max:.3f}]"
        )


def support_min_value(objects: list[np.ndarray], direction: np.ndarray) -> float:
    """计算物体在某个方向上的最小投影。

    用外法线表示障碍边时，物体从外侧向内运动，最先碰到边界的是外法线投影
    最小的那一侧。例如从下方向上接触水平底边时，是物体上侧先碰到底边。
    """

    return min(float(np.min(poly @ direction)) for poly in objects)


def support_points_lateral_interval(
    objects: list[np.ndarray],
    ref_point: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    tol: float = 1e-7,
) -> tuple[float, float]:
    """计算物体在给定法线方向上的支撑点切向范围。

    对斜边而言，最先接触的可能是方块的一个角点，也可能是一条边。先找出
    ``dot(local_point, normal)`` 最小的局部支撑点，再把这些点平移到世界坐标
    后投影到障碍边切向方向。
    """

    support = support_min_value(objects, normal)
    vals: list[float] = []
    for poly in objects:
        dots = poly @ normal
        support_pts = poly[np.isclose(dots, support, atol=tol)]
        vals.extend((support_pts + ref_point) @ tangent)
    return float(min(vals)), float(max(vals))


def intervals_overlap(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-9) -> bool:
    """判断两个一维闭区间是否重叠。"""

    return max(a[0], b[0]) <= min(a[1], b[1]) + tol


def movable_edge_contact_pairs(
    objects: list[np.ndarray],
    ref_point: np.ndarray,
    obstacle_p0: np.ndarray,
    obstacle_p1: np.ndarray,
    tol: float = 1e-7,
) -> list[tuple[int, int]]:
    """返回与某条障碍边发生边界接触的 movable object 边。

    这个检查显式落实“movable_objects 与 obstacle_true 的所有边均可发生碰撞，
    且两者无相交”的约束：接触必须发生在两个多边形的边界线段之间，而不是
    只满足法线投影相等。
    """

    pairs: list[tuple[int, int]] = []
    for obj_i, poly in enumerate(objects):
        world_poly = poly + ref_point
        for edge_i, (p0, p1) in enumerate(zip(world_poly, np.roll(world_poly, -1, axis=0))):
            if segments_intersect(p0, p1, obstacle_p0, obstacle_p1, tol=tol):
                pairs.append((obj_i, edge_i))
    return pairs


def polygon_y_intervals(poly: np.ndarray, x_value: float) -> list[tuple[float, float]]:
    """求多边形在竖直扫描线 ``x = x_value`` 上的内部 y 区间。

    面积求交采用扫描线思想：在相邻 x 坐标之间，多边形的竖直截面形状不变，
    只需要取中点 x，计算该竖线与多边形的交点，再两两配对得到内部区间。
    这里跳过竖直边，避免顶点被重复计数。
    """

    crossings: list[float] = []
    for p0, p1 in zip(poly, np.roll(poly, -1, axis=0)):
        x0, y0 = p0
        x1, y1 = p1
        if abs(x0 - x1) < 1e-12:
            continue
        x_min, x_max = min(x0, x1), max(x0, x1)
        if x_min < x_value < x_max:
            ratio = (x_value - x0) / (x1 - x0)
            crossings.append(float(y0 + ratio * (y1 - y0)))

    crossings.sort()
    intervals: list[tuple[float, float]] = []
    for i in range(0, len(crossings) - 1, 2):
        if crossings[i + 1] > crossings[i]:
            intervals.append((crossings[i], crossings[i + 1]))
    return intervals


def polygon_intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """计算两个任意直线边多边形的重叠面积。

    本实验需要严格保证 ``movable_objects`` 与 ``obstacle_true`` 不发生面积重叠。
    为了避免引入额外几何库，这里仍采用竖直扫描线：收集两多边形顶点的 x 坐标
    和两多边形边-边交点的 x 坐标作为分段边界。在每个小区间内，多边形截面的
    拓扑不会变化，因此取中点 x 比较 y 区间重叠即可。这个方法适用于水平、
    竖直和斜边组成的简单多边形。
    """

    x_values = polygon_scan_x_values(poly_a, poly_b)
    if len(x_values) < 2:
        return 0.0

    area = 0.0
    for x0, x1 in zip(x_values[:-1], x_values[1:]):
        width = float(x1 - x0)
        if width <= 1e-12:
            continue
        x_mid = 0.5 * (x0 + x1)
        intervals_a = polygon_y_intervals(poly_a, x_mid)
        intervals_b = polygon_y_intervals(poly_b, x_mid)
        for a0, a1 in intervals_a:
            for b0, b1 in intervals_b:
                overlap = min(a1, b1) - max(a0, b0)
                if overlap > 0.0:
                    area += width * overlap
    return float(area)


def segment_intersection_x_values(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    tol: float = 1e-9,
) -> list[float]:
    """返回两条线段交点/重叠端点的 x 坐标，用于扫描线分段。

    如果两线段严格相交，返回交点 x；如果共线重叠，返回重叠区间端点 x。
    只需要 x 坐标，因为面积积分沿 x 方向分段。
    """

    r = a1 - a0
    s = b1 - b0
    denom = float(r[0] * s[1] - r[1] * s[0])
    q_minus_p = b0 - a0
    values: list[float] = []
    if abs(denom) <= tol:
        cross = float(q_minus_p[0] * r[1] - q_minus_p[1] * r[0])
        if abs(cross) > tol:
            return values
        axis = 0 if abs(r[0]) >= abs(r[1]) else 1
        if abs(r[axis]) <= tol:
            return values
        t0 = float((b0[axis] - a0[axis]) / r[axis])
        t1 = float((b1[axis] - a0[axis]) / r[axis])
        lo = max(0.0, min(t0, t1))
        hi = min(1.0, max(t0, t1))
        if lo <= hi + tol:
            values.extend([float((a0 + lo * r)[0]), float((a0 + hi * r)[0])])
        return values

    t = float((q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / denom)
    u = float((q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / denom)
    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        point = a0 + np.clip(t, 0.0, 1.0) * r
        values.append(float(point[0]))
    return values


def polygon_scan_x_values(poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
    """收集扫描线分段所需的所有 x 坐标。"""

    values = [float(v) for v in np.concatenate([poly_a[:, 0], poly_b[:, 0]])]
    for a0, a1 in zip(poly_a, np.roll(poly_a, -1, axis=0)):
        for b0, b1 in zip(poly_b, np.roll(poly_b, -1, axis=0)):
            values.extend(segment_intersection_x_values(a0, a1, b0, b1))
    return np.unique(np.round(np.array(values, dtype=float), 10))


def total_overlap_area(
    objects: list[np.ndarray],
    ref_point: np.ndarray,
    obstacle: np.ndarray | list[np.ndarray],
) -> float:
    """计算所有可移动物体放在 ``ref_point`` 后与障碍物的总重叠面积。

    ``objects`` 中每个多边形都是相对物体参考点的局部坐标，先整体平移到
    世界坐标，再分别与障碍物求交。返回值为 0 表示只接触边界或完全分离。
    """

    return float(
        sum(
            polygon_intersection_area(poly + ref_point, contour)
            for poly in objects
            for contour in as_contours(obstacle)
        )
    )


def max_reference_distance_inside_boundary(
    ref_point: np.ndarray,
    direction: np.ndarray,
    boundary: ExplorationBoundary,
) -> float:
    """计算参考点沿某方向运动到探索边界前的最大距离。

    探索任务的终止条件是 movable object 的参考点运动到探索边界以外，而不是
    要求物体所有顶点始终位于边界内。旧逻辑按物体外接盒扣掉半宽，会把探测
    起点限制在障碍物附近，导致可探索运动范围偏小。
    """

    max_distance = math.inf
    for axis, lo, hi in (
        (0, boundary.x_min, boundary.x_max),
        (1, boundary.y_min, boundary.y_max),
    ):
        d_axis = float(direction[axis])
        value = float(ref_point[axis])
        if abs(d_axis) < 1e-12:
            if value < lo - 1e-9 or value > hi + 1e-9:
                return 0.0
            continue
        if d_axis > 0.0:
            max_distance = min(max_distance, (hi - value) / d_axis)
        else:
            max_distance = min(max_distance, (lo - value) / d_axis)

    if not math.isfinite(max_distance):
        return 0.0
    return max(0.0, float(max_distance))


def cleaned_axis_path(points: list[np.ndarray]) -> list[np.ndarray]:
    """删除连续重复的路径点。

    轴对齐路径在生成候选折线时可能出现零长度段，例如起点和转折点相同。
    删除这些点可以让后续长度统计、连续性检查和绘图更稳定。
    """

    cleaned: list[np.ndarray] = []
    for point in points:
        p = np.asarray(point, dtype=float)
        if cleaned and np.linalg.norm(p - cleaned[-1]) < 1e-9:
            continue
        cleaned.append(p)
    return cleaned


def is_axis_aligned_path(points: list[np.ndarray]) -> bool:
    """检查路径是否只由水平/竖直线段组成。

    用户要求运动方向只能是 ``+X/-X/+Y/-Y``，因此任何同时改变 x 和 y 的
    对角线段都应被拒绝。
    """

    for p0, p1 in zip(points[:-1], points[1:]):
        delta = p1 - p0
        if abs(delta[0]) > 1e-9 and abs(delta[1]) > 1e-9:
            return False
    return True


def path_length_l1(points: list[np.ndarray]) -> float:
    """计算四方向路径的曼哈顿长度。"""

    return float(sum(np.sum(np.abs(p1 - p0)) for p0, p1 in zip(points[:-1], points[1:])))


def max_overlap_along_path(
    objects: list[np.ndarray],
    points: list[np.ndarray],
    obstacle: np.ndarray | list[np.ndarray],
) -> float:
    """计算整条折线路径上的最大面积重叠。

    如果路径还没形成线段，就只检查起始位置；否则逐段调用
    ``max_overlap_along_segment``。
    """

    if len(points) < 2:
        return total_overlap_area(objects, points[0], obstacle) if points else 0.0
    return float(
        max(
            # 直接展开唯一一次使用的单段采样重叠检查，保留原来的 81 点离散分辨率。
            max(
                total_overlap_area(objects, p0 + (p1 - p0) * alpha, obstacle)
                for alpha in np.linspace(0.0, 1.0, 81)
            )
            for p0, p1 in zip(points[:-1], points[1:])
        )
    )


def object_half_extents(objects: list[np.ndarray]) -> np.ndarray:
    """返回物体集合相对参考点的 x/y 最大半宽。

    候选路径会优先考虑贴近探索边界的“走廊”。半宽用于构造让物体整体仍在
    边界内的通道坐标。
    """

    all_points = np.vstack(objects)
    return np.max(np.abs(all_points), axis=0)


def axis_aligned_path_candidates(
    start: np.ndarray,
    end: np.ndarray,
    boundary: ExplorationBoundary,
    objects: list[np.ndarray],
) -> list[list[np.ndarray]]:
    """生成从 ``start`` 到 ``end`` 的若干四方向候选折线。

    候选包括两种最短 L 形路径，以及沿探索边界内侧/边界线绕行的路径。
    后续会根据是否碰撞、路径长度等条件选择可执行路径。
    """

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    half = object_half_extents(objects)
    x_corridors = [
        boundary.x_min,
        boundary.x_min + half[0],
        boundary.x_max - half[0],
        boundary.x_max,
    ]
    y_corridors = [
        boundary.y_min,
        boundary.y_min + half[1],
        boundary.y_max - half[1],
        boundary.y_max,
    ]

    # 两条直接 L 形路径：先横后竖、先竖后横。
    paths: list[list[np.ndarray]] = [
        [start, np.array([end[0], start[1]], dtype=float), end],
        [start, np.array([start[0], end[1]], dtype=float), end],
    ]
    # 贴近上下边界的横向通道，用于绕开中间障碍区域。
    for y in y_corridors:
        paths.append(
            [
                start,
                np.array([start[0], y], dtype=float),
                np.array([end[0], y], dtype=float),
                end,
            ]
        )
    # 贴近左右边界的纵向通道。
    for x in x_corridors:
        paths.append(
            [
                start,
                np.array([x, start[1]], dtype=float),
                np.array([x, end[1]], dtype=float),
                end,
            ]
        )
    # 双走廊绕行路径：先到某条纵向边界走廊，再转到横向边界走廊，最后接近目标。
    # 这类路径可表达“沿探索边界外侧绕过凹形障碍”的动作，对内凹底边探测很关键。
    for x in x_corridors:
        for y in y_corridors:
            paths.append(
                [
                    start,
                    np.array([x, start[1]], dtype=float),
                    np.array([x, y], dtype=float),
                    np.array([end[0], y], dtype=float),
                    end,
                ]
            )
            paths.append(
                [
                    start,
                    np.array([start[0], y], dtype=float),
                    np.array([x, y], dtype=float),
                    np.array([x, end[1]], dtype=float),
                    end,
                ]
            )

    # 删除重复路径，避免相同候选被反复评估。
    unique: list[list[np.ndarray]] = []
    signatures: set[tuple[tuple[float, float], ...]] = set()
    for path in paths:
        cleaned = cleaned_axis_path(path)
        signature = tuple((round(float(p[0]), 9), round(float(p[1]), 9)) for p in cleaned)
        if signature not in signatures:
            signatures.add(signature)
            unique.append(cleaned)
    return unique


def segment_action_and_distance(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    """把一段轴对齐运动转换为单位动作方向和距离。

    返回的动作只能是 ``[1,0]``、``[-1,0]``、``[0,1]``、``[0,-1]`` 或零向量。
    若线段不是轴对齐，则直接报错，防止隐式产生不符合约束的运动。
    """

    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    if np.linalg.norm(delta) < 1e-12:
        return np.zeros(2, dtype=float), 0.0
    if abs(delta[0]) > 1e-9 and abs(delta[1]) > 1e-9:
        raise ValueError(f"Segment is not 4-direction axis aligned: {start} -> {end}")
    distance = float(np.sum(np.abs(delta)))
    return delta / distance, distance


def first_contact_on_segment(
    start: np.ndarray,
    end: np.ndarray,
    true_edges: list[Edge],
    objects: list[np.ndarray],
) -> ContactObservation | None:
    """求一条四方向线段上最早发生的接触。

    逻辑是沿运动方向投影每条真实障碍边，计算物体支撑边到该障碍边的距离。
    只接受满足两个条件的边：动作方向确实是从外侧推向该边，并且接触侧在
    切向方向上有重叠。若多条边同一时刻接触，则全部记录。
    """

    action, segment_distance = segment_action_and_distance(start, end)
    if segment_distance <= 1e-12:
        return None

    best_t = math.inf
    hit_edges: list[int] = []
    contact_eps = 1e-7

    for edge in true_edges:
        normal_speed = float(np.dot(action, edge.outward))
        if normal_speed >= -1e-9:
            continue
        # t_contact 是物体参考点从 start 出发需要走多远，才让支撑边贴上障碍边。
        support = support_min_value(objects, edge.outward)
        edge_plane = edge.const
        t_contact = (edge_plane - support - float(np.dot(start, edge.outward))) / normal_speed
        if t_contact < -1e-8 or t_contact > segment_distance + 1e-8:
            continue
        if t_contact <= contact_eps:
            # 如果起点就在接触面附近，向前微小探测：只有马上产生面积重叠时，
            # 才把它视为真实接触，避免把“刚好擦边但会离开”的情况误判为碰撞。
            probe_distance = min(segment_distance, 1e-3)
            probe_ref = start + action * probe_distance
            if total_overlap_area(objects, probe_ref, OBSTACLE_TRUE) <= 1e-8:
                continue

        contact_ref = start + action * max(0.0, t_contact)
        obj_interval = support_points_lateral_interval(
            objects,
            contact_ref,
            edge.outward,
            edge.tangent,
        )
        if not intervals_overlap(obj_interval, edge.span, tol=1e-8):
            continue
        if not movable_edge_contact_pairs(objects, contact_ref, edge.p0, edge.p1):
            continue

        if t_contact < best_t - 1e-8:
            best_t = max(0.0, float(t_contact))
            hit_edges = [edge.index]
        elif abs(t_contact - best_t) <= 1e-8:
            hit_edges.append(edge.index)

    if not hit_edges:
        return None

    contact_ref = start + action * best_t
    overlap_area = total_overlap_area(objects, contact_ref, OBSTACLE_TRUE)
    observed_consts = []
    for edge_i in hit_edges:
        edge = true_edges[edge_i]
        observed_consts.append(
            float(np.dot(contact_ref, edge.outward) + support_min_value(objects, edge.outward))
        )
    axis_const = float(np.mean(observed_consts))
    plane_const = axis_const

    return ContactObservation(
        ok=True,
        action=action.copy(),
        start=np.asarray(start, dtype=float).copy(),
        contact_ref=contact_ref,
        distance=best_t,
        plane_const=plane_const,
        axis_const=float(axis_const),
        overlap_area=overlap_area,
        max_path_overlap_area=0.0,
        true_edge_indices=hit_edges,
    )


def first_informative_contact_on_path(
    points: list[np.ndarray],
    true_edges: list[Edge],
    objects: list[np.ndarray],
    hypotheses: list[Hypothesis],
    observations: list[ContactObservation],
    ref_edges: list[Edge],
    base_groups: dict[str, SupportGroup],
    boundary: ExplorationBoundary,
    line_tol: float,
    step_index: int,
) -> tuple[ContactObservation | None, list[np.ndarray], list[Hypothesis], list[int]]:
    """沿路径寻找第一个会改变多假设集合的接触。

    如果某段首先碰到的是已经由所有假设解释过的边，该接触不会带来新信息。
    这种情况下允许路径继续向后检查，直到找到真正能扩展/剪枝假设的接触。
    """

    traversed: list[np.ndarray] = [np.asarray(points[0], dtype=float)]
    current_signatures = hypothesis_signature_set(hypotheses)
    current_consensus = len(consensus_observed_edges(hypotheses))

    for p0, p1 in zip(points[:-1], points[1:]):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        contact = first_contact_on_segment(p0, p1, true_edges, objects)
        if contact is None:
            traversed.append(p1)
            continue

        next_hypotheses: list[Hypothesis] = []
        candidate_edges: list[int] = []
        for tol_scale in (1.0, 2.0, 4.0):
            next_hypotheses, candidate_edges = update_hypotheses_with_observation(
                hypotheses,
                contact,
                observations + [contact],
                ref_edges,
                base_groups,
                objects,
                boundary,
                line_tol=line_tol * tol_scale,
                span_pad=8.0 + 6.0 * (tol_scale - 1.0),
                step_index=step_index,
            )
            if next_hypotheses:
                break

        next_signatures = hypothesis_signature_set(next_hypotheses)
        next_consensus = len(consensus_observed_edges(next_hypotheses)) if next_hypotheses else 0
        if next_hypotheses and (
            next_signatures != current_signatures or next_consensus > current_consensus
        ):
            traversed.append(contact.contact_ref.copy())
            return contact, cleaned_axis_path(traversed), next_hypotheses, candidate_edges

        traversed.append(contact.contact_ref.copy())
        if np.linalg.norm(contact.contact_ref - p1) > 1e-8:
            # 路径在该段中途被无信息接触阻断；为保持“不穿透”，不继续同一线段。
            return None, cleaned_axis_path(traversed), [], []
    return None, cleaned_axis_path(traversed), [], []


def candidate_order(
    ordering: str,
    current_pos: np.ndarray,
    groups: dict[str, SupportGroup],
    ref_edges: list[Edge],
    objects: list[np.ndarray],
    remaining: set[str],
    boundary: ExplorationBoundary,
    fixed_order: list[str],
) -> list[str]:
    """根据策略给出下一批待尝试支撑线的顺序。

    ``reference`` / ``axis_anchors`` 使用固定顺序；``nearest`` 优先选择离当前
    位置最近的探测起点；``information`` 在不确定区间宽度和移动代价之间折中。
    """

    if ordering in ("reference", "axis_anchors"):
        return [key for key in fixed_order if key in remaining]
    if ordering == "nearest":
        scored: list[tuple[float, str]] = []
        for key in remaining:
            group = groups[key]
            probe = make_probe(
                group,
                ref_edges,
                objects,
                predicted_const(group, groups),
                approach_margin=90.0,
                groups=groups,
                boundary=boundary,
            )
            scored.append((float(np.sum(np.abs(probe.start - current_pos))), key))
        return [key for _, key in sorted(scored)]
    if ordering == "information":
        scored = []
        for key in remaining:
            group = groups[key]
            width = group.max_const - group.min_const if group.observed_const is None else 0.0
            probe = make_probe(
                group,
                ref_edges,
                objects,
                predicted_const(group, groups),
                approach_margin=90.0,
                groups=groups,
                boundary=boundary,
            )
            transit = float(np.sum(np.abs(probe.start - current_pos)))
            scored.append((-(width - 0.015 * transit), key))
        return [key for _, key in sorted(scored)]
    raise ValueError(f"unknown ordering: {ordering}")


def plan_next_disambiguating_probe(
    ordering: str,
    current_pos: np.ndarray,
    groups: dict[str, SupportGroup],
    ref_edges: list[Edge],
    true_edges: list[Edge],
    objects: list[np.ndarray],
    boundary: ExplorationBoundary,
    fixed_order: list[str],
    hypotheses: list[Hypothesis],
    observations: list[ContactObservation],
    base_groups: dict[str, SupportGroup],
    line_tol: float,
    step_index: int,
) -> tuple[str, Probe, ContactObservation, list[np.ndarray], list[Hypothesis], list[int]] | None:
    """选择最能区分当前多接触对假设的下一次探测。

    每个候选探测都会先在仿真环境中得到一次可能观测，再用这次观测更新当前
    hypothesis 集合。评分时优先选择能让假设数量变少的动作；若假设数相同，
    则选择能让更多边在所有假设中达成一致、候选接触对更少、移动距离更短的动作。
    """

    remaining_keys = {
        key for key, group in groups.items() if group.observed_const is None
    }
    current_signatures = hypothesis_signature_set(hypotheses)
    current_consensus = consensus_observed_edges(hypotheses)
    current_consensus_count = len(current_consensus)
    best: tuple[
        tuple[int, int, int, float],
        str,
        Probe,
        ContactObservation,
        list[np.ndarray],
        list[Hypothesis],
        list[int],
    ] | None = None

    if len(hypotheses) == 1:
        # 接触对已经唯一后，剩余任务主要是把各边支撑线补齐。此时继续按
        # information/nearest 只看局部距离，可能把凹槽深处的边拖到最后，
        # 导致后续路径被已知边阻断。按参考拓扑顺序收尾更稳定。
        ordered_group_keys = [
            key for key in order_reference(groups, ref_edges) if key in remaining_keys
        ]
    else:
        ordered_group_keys = candidate_order(
            ordering,
            current_pos,
            groups,
            ref_edges,
            objects,
            remaining_keys,
            boundary,
            fixed_order,
        )

    for group_key in ordered_group_keys:
        group = groups[group_key]
        probe_variants = probe_variants_for_group(
            group,
            ref_edges,
            objects,
            predicted_const(group, groups),
            groups=groups,
            boundary=boundary,
        )
        for probe in probe_variants:
            planned_obs = simulate_contact(
                probe,
                true_edges,
                objects,
                max_distance=220.0,
                boundary=boundary,
            )
            if not planned_obs.ok:
                continue

            path_options = sorted(
                axis_aligned_path_candidates(current_pos, planned_obs.start, boundary, objects),
                key=path_length_l1,
            )
            for transit_path in path_options:
                planned_path = cleaned_axis_path(transit_path + [planned_obs.contact_ref])
                if not is_axis_aligned_path(planned_path):
                    continue

                path_contact, full_path, next_hypotheses, candidate_edges = first_informative_contact_on_path(
                    planned_path,
                    true_edges,
                    objects,
                    hypotheses,
                    observations,
                    ref_edges,
                    base_groups,
                    boundary,
                    line_tol,
                    step_index,
                )
                if path_contact is None:
                    continue
                if not next_hypotheses:
                    continue
                if path_length_l1(full_path) <= 1e-8:
                    # 当前参考点已经处在同一个接触位姿时，不把它记录为一次新的
                    # 探索。每次探索必须包含实际四方向运动，然后才以接触/出界终止。
                    continue
                if candidate_edges and all(edge_i in current_consensus for edge_i in candidate_edges):
                    # 只碰到已经共识确认的边，不会缩小接触对或顶点范围；继续尝试
                    # 同一目标边的其它起点/路径，避免已知边把未观测边拖到最后。
                    continue

                consensus_count = len(consensus_observed_edges(next_hypotheses))
                next_signatures = hypothesis_signature_set(next_hypotheses)
                is_noop = (
                    next_signatures == current_signatures
                    and consensus_count <= current_consensus_count
                )
                if is_noop:
                    continue
                score = (
                    len(next_hypotheses),
                    -consensus_count,
                    len(candidate_edges),
                    path_length_l1(full_path),
                )
                if best is None or score < best[0]:
                    best = (
                        score,
                        group_key,
                        probe,
                        path_contact,
                        full_path,
                        next_hypotheses,
                        candidate_edges,
                    )
                break
        if len(hypotheses) == 1 and best is not None:
            break

    if best is None:
        return None
    _, group_key, probe, obs, full_path, next_hypotheses, candidate_edges = best
    return group_key, probe, obs, full_path, next_hypotheses, candidate_edges


def action_matches_edge(action: np.ndarray, edge: Edge) -> bool:
    """判断四方向动作是否会从外侧向内推向该边。

    斜边的法线通常不等于 ``+X/-X/+Y/-Y``，因此不能再要求二者平行。只要动作
    会让物体参考点在外法线方向上的投影变小，就可能从外侧压向这条边。
    """

    return float(np.dot(action, edge.outward)) < -1e-9


def non_overlapping_contact_ref(
    contact_ref: np.ndarray,
    action: np.ndarray,
    target: Edge,
    objects: list[np.ndarray],
    obstacle: np.ndarray | list[np.ndarray],
    samples: int = 161,
) -> tuple[np.ndarray | None, float]:
    """在目标边切向方向上寻找一个“不重叠”的接触位姿。

    纯粹按边中点构造的接触点可能让方形物体的一部分压进凹形障碍物内部。
    因此这里固定法向接触位置，只沿边的切向方向搜索可行参考点，优先返回
    面积重叠为 0 且离原始位置最近的方案。
    """

    normal = target.outward
    tangent = target.tangent
    contact_dot = float(np.dot(contact_ref, normal))
    current_lat = float(np.dot(contact_ref, tangent))
    local_interval = support_points_lateral_interval(
        objects,
        np.zeros(2),
        normal,
        tangent,
    )
    edge_interval = target.span

    lat_min = edge_interval[0] - local_interval[1]
    lat_max = edge_interval[1] - local_interval[0]
    if lat_min > lat_max:
        return None, math.inf

    # 候选包含当前横向位置、均匀采样点，以及区间端点组合；这样能覆盖边界接触。
    candidates = [float(np.clip(current_lat, lat_min, lat_max))]
    candidates.extend(np.linspace(lat_min, lat_max, samples).tolist())
    candidates.extend(
        [
            edge_interval[0] - local_interval[0],
            edge_interval[0] - local_interval[1],
            edge_interval[1] - local_interval[0],
            edge_interval[1] - local_interval[1],
        ]
    )

    best_ref: np.ndarray | None = None
    best_area = math.inf
    best_shift = math.inf
    for lat in candidates:
        if lat < lat_min - 1e-9 or lat > lat_max + 1e-9:
            continue
        candidate_ref = normal * contact_dot + tangent * lat
        area = total_overlap_area(objects, candidate_ref, obstacle)
        shift = abs(lat - current_lat)
        if area < best_area - 1e-9 or (abs(area - best_area) <= 1e-9 and shift < best_shift):
            best_area = area
            best_shift = shift
            best_ref = candidate_ref
        if area <= 1e-9:
            return candidate_ref, 0.0

    return best_ref, best_area


def make_probe(
    group: SupportGroup,
    ref_edges: list[Edge],
    objects: list[np.ndarray],
    predicted_const: float,
    approach_margin: float,
    groups: dict[str, SupportGroup] | None = None,
    boundary: ExplorationBoundary | None = None,
    start_at_boundary: bool = True,
) -> Probe:
    """根据当前估计支撑线构造一次计划探测。

    ``predicted_const`` 是该边目前认为最可能的支撑线坐标。函数先计算预期接触
    参考点，再沿接触动作反方向退让到探索边界侧作为探测起点。这样每次直线
    探测都覆盖“从边界进入、直到接触或出界”的完整运动范围。
    """

    edge = ref_edges[group.edge_indices[0]]
    # 直接展开唯一一次使用的动作选择：从四方向动作里挑选最朝向 -outward 的那个。
    action = max(ACTIONS, key=lambda a: float(np.dot(a, -edge.outward))).copy()
    ref_mid = 0.5 * (edge.p0 + edge.p1)
    contact_dot = predicted_const - support_min_value(objects, edge.outward)
    expected_contact = edge.outward * contact_dot + edge.tangent * float(np.dot(ref_mid, edge.tangent))
    if boundary is not None and start_at_boundary:
        # 以参考点探索边界为准，而不是以 movable object 外接盒为准；否则探测
        # 起点会被物体半宽收缩到边界内侧，运动范围明显变小。
        approach_margin = max_reference_distance_inside_boundary(
            expected_contact,
            -action,
            boundary,
        )
    elif boundary is not None:
        approach_margin = min(
            approach_margin,
            max_reference_distance_inside_boundary(expected_contact, -action, boundary),
        )
    start = expected_contact - action * approach_margin
    return Probe(
        group_key=group.key,
        edge_index=edge.index,
        action=action,
        start=start,
        expected_contact=expected_contact,
    )


def probe_variants_for_group(
    group: SupportGroup,
    ref_edges: list[Edge],
    objects: list[np.ndarray],
    predicted_const: float,
    groups: dict[str, SupportGroup],
    boundary: ExplorationBoundary,
) -> list[Probe]:
    """为同一候选边生成边界探测和局部探测两类起点。

    外侧边适合从探索边界进入；凹槽内部边若也从全局边界进入，射线会先撞到
    其它已知边。因此保留多个局部退让距离，让策略可以选择真正可达的探测段。
    """

    probes = [
        make_probe(
            group,
            ref_edges,
            objects,
            predicted_const,
            approach_margin=90.0,
            groups=groups,
            boundary=boundary,
            start_at_boundary=True,
        )
    ]
    for margin in (20.0, 45.0, 90.0, 140.0):
        probes.append(
            make_probe(
                group,
                ref_edges,
                objects,
                predicted_const,
                approach_margin=margin,
                groups=groups,
                boundary=boundary,
                start_at_boundary=False,
            )
        )

    unique: list[Probe] = []
    signatures: set[tuple[float, float, float, float]] = set()
    for probe in probes:
        signature = (
            round(float(probe.start[0]), 6),
            round(float(probe.start[1]), 6),
            round(float(probe.action[0]), 6),
            round(float(probe.action[1]), 6),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(probe)
    return unique


def simulate_contact(
    probe: Probe,
    true_edges: list[Edge],
    objects: list[np.ndarray],
    max_distance: float,
    boundary: ExplorationBoundary | None = None,
) -> ContactObservation:
    """仿真一次针对目标边的直线接触探测。

    这里允许使用 ``true_edges``，因为它是离线仿真的环境 oracle；实际估计算法
    不从中读取边界常数，只把仿真返回的第一接触结果当作触觉观测。
    """

    action = probe.action
    start = probe.start
    if boundary is not None:
        # 单次探索沿四方向一直执行到接触；若没有接触，则参考点到达/越过探索
        # 边界即终止，因此最大探测距离由参考点到边界的距离决定。
        max_distance = max_reference_distance_inside_boundary(start, action, boundary)
    target = true_edges[probe.edge_index]
    if not action_matches_edge(action, target):
        return ContactObservation(
            ok=False,
            action=action.copy(),
            start=start.copy(),
            message=f"target edge {target.index} cannot be reached by action {action.tolist()}",
        )

    support = support_min_value(objects, target.outward)
    normal_speed = float(np.dot(action, target.outward))
    edge_plane = target.const
    t_contact = (edge_plane - support - float(np.dot(start, target.outward))) / normal_speed
    if t_contact < -1e-8 or t_contact > max_distance:
        return ContactObservation(
            ok=False,
            action=action.copy(),
            start=start.copy(),
            message=f"target edge {target.index} is outside the probe distance",
        )

    contact_ref = start + action * t_contact
    # 调整切向位置，保证接触位姿本身没有面积重叠；面积重叠代表已经侵入障碍。
    contact_ref, overlap_area = non_overlapping_contact_ref(
        contact_ref,
        action,
        target,
        objects,
        OBSTACLE_TRUE,
    )
    if contact_ref is None or overlap_area > 1e-8:
        return ContactObservation(
            ok=False,
            action=action.copy(),
            start=start.copy(),
            message=(
                f"target edge {target.index} has no non-overlapping contact pose "
                f"(minimum overlap area={overlap_area:.6f})"
            ),
        )
    # 切向调整后，保持同样的法向接触距离，重新得到实际探测起点。
    start = contact_ref - action * t_contact
    hit_edges: list[int] = []
    for edge in true_edges:
        if not action_matches_edge(action, edge):
            continue
        candidate_support = support_min_value(objects, edge.outward)
        candidate_plane = float(np.dot(contact_ref, edge.outward) + candidate_support)
        if abs(candidate_plane - edge.const) > 1e-8:
            continue
        support_interval = support_points_lateral_interval(
            objects,
            contact_ref,
            edge.outward,
            edge.tangent,
        )
        if intervals_overlap(support_interval, edge.span, tol=1e-8):
            if movable_edge_contact_pairs(objects, contact_ref, edge.p0, edge.p1):
                hit_edges.append(edge.index)

    if not hit_edges:
        return ContactObservation(
            ok=False,
            action=action.copy(),
            start=start.copy(),
            message=f"target edge {target.index} has no lateral overlap with the movable object",
        )

    plane_const = float(np.dot(contact_ref, target.outward) + support)
    axis_const = plane_const

    return ContactObservation(
        ok=True,
        action=action.copy(),
        start=start.copy(),
        contact_ref=contact_ref,
        distance=float(t_contact),
        plane_const=plane_const,
        axis_const=float(axis_const),
        overlap_area=overlap_area,
        max_path_overlap_area=0.0,
        true_edge_indices=hit_edges,
    )


def const_interval(group: SupportGroup) -> tuple[float, float]:
    """返回某条支撑线当前仍可能取值的区间。

    一旦该边已经被接触观测确定，区间退化为单点；否则使用探索边界给出的
    初始范围。
    """

    if group.observed_const is not None:
        return group.observed_const, group.observed_const
    return group.min_const, group.max_const


def predicted_const(group: SupportGroup, groups: dict[str, SupportGroup]) -> float:
    """给出当前用于绘图/规划的支撑线预测值。

    未观测边默认使用“已放入探索边界内部”的形状先验常数，而不是原始
    ``obstacle_ref`` 的绝对坐标；已观测边直接使用触觉反推出的真实常数。
    """

    if group.observed_const is not None:
        return group.observed_const
    return float(np.clip(group.ref_const, group.min_const, group.max_const))


def current_support_constants(
    ref_edges: list[Edge], groups: dict[str, SupportGroup]
) -> list[float]:
    """按参考边顺序输出当前每条边的支撑线常数。"""

    edge_to_group = {
        edge_i: group for group in groups.values() for edge_i in group.edge_indices
    }
    constants: list[float] = []
    for edge in ref_edges:
        group = edge_to_group[edge.index]
        constants.append(predicted_const(group, groups))
    return constants


def polygon_from_supports(ref_edges: list[Edge], constants: list[float]) -> np.ndarray:
    """由相邻支撑线交点重建当前估计多边形。

    每个顶点都是前一条边和当前边两条支撑线的交点。支撑线形式统一为
    ``dot(point, outward)=const``，因此水平、竖直和斜边都可用同一套 2x2
    线性方程求解。
    """

    vertices: list[np.ndarray | None] = [None] * len(ref_edges)
    neighbors = adjacent_edge_indices(ref_edges)
    for curr_edge in ref_edges:
        prev_i, _ = neighbors[curr_edge.index]
        prev_edge = ref_edges[prev_i]
        prev_const = constants[prev_i]
        curr_const = constants[curr_edge.index]

        matrix = np.vstack([prev_edge.outward, curr_edge.outward])
        rhs = np.array([prev_const, curr_const], dtype=float)
        det = float(np.linalg.det(matrix))
        if abs(det) < 1e-9:
            raise ValueError(
                f"Adjacent support lines are nearly parallel: "
                f"edge {prev_edge.index} and edge {curr_edge.index}"
            )
        vertices[curr_edge.index] = np.linalg.solve(matrix, rhs)
    return np.vstack([v for v in vertices if v is not None])


def point_on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, tol: float = 1e-9) -> bool:
    """判断点 ``p`` 是否位于线段 ``a-b`` 上。"""

    cross = float(np.cross(b - a, p - a))
    if abs(cross) > tol:
        return False
    return (
        min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
        and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
    )


def segment_orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """返回三点有向面积的两倍，用于线段相交判断。"""

    return float(np.cross(b - a, c - a))


def segments_intersect(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    tol: float = 1e-9,
) -> bool:
    """判断两条闭线段是否相交。"""

    o1 = segment_orientation(a0, a1, b0)
    o2 = segment_orientation(a0, a1, b1)
    o3 = segment_orientation(b0, b1, a0)
    o4 = segment_orientation(b0, b1, a1)

    if abs(o1) <= tol and point_on_segment(a0, a1, b0, tol):
        return True
    if abs(o2) <= tol and point_on_segment(a0, a1, b1, tol):
        return True
    if abs(o3) <= tol and point_on_segment(b0, b1, a0, tol):
        return True
    if abs(o4) <= tol and point_on_segment(b0, b1, a1, tol):
        return True
    return o1 * o2 < -tol and o3 * o4 < -tol


def polygon_self_intersections(poly: np.ndarray) -> list[tuple[int, int]]:
    """找出多边形中所有非相邻边之间的交叉。

    相邻边共享一个顶点是合法的；非相邻边一旦相交，说明当前估计轮廓不再是
    简单多边形，违反“闭合多边形且不自交”的约束。
    """

    intersections: list[tuple[int, int]] = []
    n = len(poly)
    for i in range(n):
        a0 = poly[i]
        a1 = poly[(i + 1) % n]
        for j in range(i + 1, n):
            # 跳过同一条边、首尾相邻边以及普通相邻边。
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            b0 = poly[j]
            b1 = poly[(j + 1) % n]
            if segments_intersect(a0, a1, b0, b1):
                intersections.append((i, j))
    return intersections


def is_simple_polygon(poly: np.ndarray, tol: float = 1e-8) -> bool:
    """判断多边形是否为逆时针、面积非零且无自交的简单多边形。"""

    if len(poly) < 3:
        return False
    if signed_area(poly) <= tol:
        return False
    return not polygon_self_intersections(poly)


def contour_counts_from_edges(ref_edges: list[Edge]) -> list[int]:
    """按轮廓顺序统计每个轮廓的边/顶点数量。"""

    counts: dict[int, int] = {}
    for edge in ref_edges:
        counts[edge.contour_index] = counts.get(edge.contour_index, 0) + 1
    return [counts[i] for i in sorted(counts)]


def contours_from_flat_vertices(vertices: np.ndarray, ref_edges: list[Edge]) -> list[np.ndarray]:
    """把估计得到的扁平顶点数组恢复成多个闭合轮廓。"""

    return split_flat_vertices(vertices, contour_counts_from_edges(ref_edges))


def is_simple_obstacle(vertices: np.ndarray, ref_edges: list[Edge], tol: float = 1e-8) -> bool:
    """检查多轮廓估计是否由若干简单、互不重叠的逆时针多边形组成。"""

    contours = contours_from_flat_vertices(vertices, ref_edges)
    if not all(is_simple_polygon(poly, tol=tol) for poly in contours):
        return False
    for i, poly_a in enumerate(contours):
        for poly_b in contours[i + 1 :]:
            if polygon_intersection_area(poly_a, poly_b) > tol:
                return False
    return True


def obstacle_self_intersections(vertices: np.ndarray, ref_edges: list[Edge]) -> list[tuple[int, int, int]]:
    """返回每个轮廓内部的自交边对，轮廓之间不当作同一个闭环检查。"""

    result: list[tuple[int, int, int]] = []
    offset = 0
    for contour_i, poly in enumerate(contours_from_flat_vertices(vertices, ref_edges)):
        for a, b in polygon_self_intersections(poly):
            result.append((contour_i, offset + a, offset + b))
        offset += len(poly)
    return result


def constrained_estimated_polygon(
    ref_edges: list[Edge],
    groups: dict[str, SupportGroup],
) -> tuple[np.ndarray, list[float], float]:
    """生成满足简单多边形约束的估计轮廓。

    当前观测尚不完整时，如果直接把“已观测真实支撑线”和“未观测参考支撑线”
    混在一起，相邻支撑线交点仍能形成闭合点列，但非相邻边可能交叉。这里先
    尝试使用原始估计；若它自交，则在参考轮廓和原始估计之间做二分插值，取
    最大的可行插值比例。这样蓝色绘图轮廓始终是闭合且无自交的简单多边形。

    返回值中的 alpha=1 表示完全采用当前支撑线估计；alpha<1 表示为了满足
    闭合简单多边形约束，对中间态显示做了保守回退。
    """

    raw_constants = current_support_constants(ref_edges, groups)
    raw_poly = polygon_from_supports(ref_edges, raw_constants)
    if is_simple_obstacle(raw_poly, ref_edges):
        return raw_poly, raw_constants, 1.0

    ref_constants = [edge.const for edge in ref_edges]
    low = 0.0
    high = 1.0
    best_constants = ref_constants
    best_poly = polygon_from_supports(ref_edges, best_constants)
    for _ in range(60):
        alpha = 0.5 * (low + high)
        constants = [
            ref_c + alpha * (raw_c - ref_c)
            for ref_c, raw_c in zip(ref_constants, raw_constants)
        ]
        poly = polygon_from_supports(ref_edges, constants)
        if is_simple_obstacle(poly, ref_edges):
            low = alpha
            best_constants = constants
            best_poly = poly
        else:
            high = alpha
    return best_poly, best_constants, low


def representative_feasible_polygon(
    ref_edges: list[Edge],
    groups: dict[str, SupportGroup],
    boundary: ExplorationBoundary | None,
    objects: list[np.ndarray] | None,
) -> tuple[np.ndarray, list[float], float]:
    """从全局可行域中取一个代表轮廓用于绘图和误差统计。

    蓝色估计轮廓必须和打印的顶点范围来自同一套约束。若仍用
    ``predicted_const`` 把未观测边放回参考位置，图像可能显示到 LP 顶点范围
    之外。这里优先解一个全局可行 LP；若数值失败，再退回原来的保守显示。
    """

    if boundary is None or objects is None:
        return constrained_estimated_polygon(ref_edges, groups)
    constants = solve_feasible_support_constants(ref_edges, groups, boundary, objects)
    if constants is None:
        return constrained_estimated_polygon(ref_edges, groups)
    poly = polygon_from_supports(ref_edges, constants)
    return poly, constants, 1.0


def observed_const_for_edge(
    obs: ContactObservation,
    edge: Edge,
    objects: list[np.ndarray],
) -> float:
    """根据接触位姿计算某条候选边对应的支撑线常数。"""

    if obs.contact_ref is None:
        raise ValueError("Cannot compute support constant without contact_ref.")
    return float(np.dot(obs.contact_ref, edge.outward) + support_min_value(objects, edge.outward))


def candidate_edges_for_observation(
    obs: ContactObservation,
    ref_edges: list[Edge],
    groups: dict[str, SupportGroup],
    objects: list[np.ndarray],
    line_tol: float,
    span_pad: float,
) -> list[int]:
    """根据一次接触观测推断可能被接触的参考边集合。

    实验日志中同时保存真实边和候选边。真实边来自仿真 oracle，候选边则模拟
    算法在只知道参考轮廓、探索边界和已有观测时能推断出的集合。判断标准包括：
    运动方向是否匹配、接触平面常数是否落在当前不确定区间内、切向跨度是否重叠。
    """

    if not obs.ok or obs.contact_ref is None:
        return []

    candidates: list[int] = []
    edge_to_group = edge_group_map(groups)
    for edge in ref_edges:
        if not action_matches_edge(obs.action, edge):
            continue

        group = edge_to_group[edge.index]
        min_const, max_const = const_interval(group)
        observed_const = observed_const_for_edge(obs, edge, objects)
        if not (min_const - line_tol <= observed_const <= max_const + line_tol):
            continue

        span_guess = edge_lateral_span_interval(
            edge.index,
            ref_edges,
            groups,
            obs.action,
            span_pad,
        )
        obj_interval = support_points_lateral_interval(
            objects,
            obs.contact_ref,
            edge.outward,
            edge.tangent,
        )
        if intervals_overlap(obj_interval, span_guess, tol=0.0):
            candidates.append(edge.index)
    return candidates


def edge_group_map(groups: dict[str, SupportGroup]) -> dict[int, SupportGroup]:
    """建立 ``edge_index -> SupportGroup`` 的快速查询表。"""

    return {edge_i: group for group in groups.values() for edge_i in group.edge_indices}


def edge_lateral_span_interval(
    edge_index: int,
    ref_edges: list[Edge],
    groups: dict[str, SupportGroup],
    action: np.ndarray,
    pad: float = 0.0,
) -> tuple[float, float]:
    """估计某条边在接触切向方向上的可能跨度。

    一条边的两个端点由相邻两条支撑线决定。当相邻边尚未观测时，端点范围会
    扩大为一个区间；该区间用于判断当前物体接触侧边是否可能与该障碍边重叠。
    """

    edge_to_group = edge_group_map(groups)
    prev_i, next_i = adjacent_edge_indices(ref_edges)[edge_index]
    prev_group = edge_to_group[prev_i]
    next_group = edge_to_group[next_i]
    prev_min, prev_max = const_interval(prev_group)
    next_min, next_max = const_interval(next_group)

    edge = ref_edges[edge_index]
    curr_group = edge_to_group[edge_index]
    curr_const = predicted_const(curr_group, groups)
    values: list[float] = []
    for neighbor_i, neighbor_const in (
        (prev_i, prev_min),
        (prev_i, prev_max),
        (next_i, next_min),
        (next_i, next_max),
    ):
        neighbor = ref_edges[neighbor_i]
        matrix = np.vstack([edge.outward, neighbor.outward])
        if abs(float(np.linalg.det(matrix))) < 1e-9:
            continue
        point = np.linalg.solve(matrix, np.array([curr_const, neighbor_const], dtype=float))
        values.append(float(np.dot(point, edge.tangent)))
    if not values:
        values = [edge.span[0], edge.span[1]]
    return min(values) - pad, max(values) + pad


def clip_parameter_polygon(
    polygon: list[np.ndarray],
    normal: np.ndarray,
    limit: float,
    tol: float = 1e-9,
) -> list[np.ndarray]:
    """在二维支撑线参数空间中用半平面 ``dot(normal, p) <= limit`` 裁剪多边形。"""

    if not polygon:
        return []
    clipped: list[np.ndarray] = []
    for current, nxt in zip(polygon, polygon[1:] + polygon[:1]):
        current_value = float(np.dot(normal, current) - limit)
        next_value = float(np.dot(normal, nxt) - limit)
        current_inside = current_value <= tol
        next_inside = next_value <= tol
        if current_inside and next_inside:
            clipped.append(nxt)
        elif current_inside and not next_inside:
            denom = current_value - next_value
            if abs(denom) > tol:
                clipped.append(current + current_value / denom * (nxt - current))
        elif not current_inside and next_inside:
            denom = current_value - next_value
            if abs(denom) > tol:
                clipped.append(current + current_value / denom * (nxt - current))
            clipped.append(nxt)
    return clipped


def feasible_pair_polygon(
    prev_edge: Edge,
    curr_edge: Edge,
    prev_interval: tuple[float, float],
    curr_interval: tuple[float, float],
    boundary: ExplorationBoundary,
    extra_halfplanes: tuple[tuple[np.ndarray, float], ...] = (),
) -> list[np.ndarray]:
    """计算相邻两条支撑线在边界和额外线性约束下的可行参数多边形。"""

    matrix = np.vstack([prev_edge.outward, curr_edge.outward])
    if abs(float(np.linalg.det(matrix))) < 1e-9:
        return []
    transform = np.linalg.inv(matrix)
    polygon = [
        np.array([prev_interval[0], curr_interval[0]], dtype=float),
        np.array([prev_interval[1], curr_interval[0]], dtype=float),
        np.array([prev_interval[1], curr_interval[1]], dtype=float),
        np.array([prev_interval[0], curr_interval[1]], dtype=float),
    ]
    # vertex = transform @ [prev_const, curr_const]，边界矩形约束是参数空间中的线性半平面。
    for normal, limit in (
        (transform[0], boundary.x_max),
        (-transform[0], -boundary.x_min),
        (transform[1], boundary.y_max),
        (-transform[1], -boundary.y_min),
        *extra_halfplanes,
    ):
        polygon = clip_parameter_polygon(polygon, normal, limit)
        if not polygon:
            return []
    return polygon


def feasible_pair_interval(
    prev_edge: Edge,
    curr_edge: Edge,
    prev_interval: tuple[float, float],
    curr_interval: tuple[float, float],
    boundary: ExplorationBoundary,
    extra_halfplanes: tuple[tuple[np.ndarray, float], ...] = (),
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """计算相邻两条支撑线在全局约束下的可行区间。"""

    polygon = feasible_pair_polygon(
        prev_edge,
        curr_edge,
        prev_interval,
        curr_interval,
        boundary,
        extra_halfplanes,
    )
    if not polygon:
        return None
    values = np.array(polygon, dtype=float)
    return (
        (float(np.min(values[:, 0])), float(np.max(values[:, 0]))),
        (float(np.min(values[:, 1])), float(np.max(values[:, 1]))),
    )


def boundary_tightened_groups(
    ref_edges: list[Edge],
    groups: dict[str, SupportGroup],
    boundary: ExplorationBoundary,
    iterations: int = 12,
    tol: float = 1e-7,
) -> dict[str, SupportGroup]:
    """根据“所有顶点必须位于探索边界内”的全局约束收缩支撑线范围。"""

    tightened = clone_groups(groups)
    edge_to_group = edge_group_map(tightened)
    neighbors = adjacent_edge_indices(ref_edges)

    def apply_pair_interval(
        prev_i: int,
        curr_i: int,
        feasible: tuple[tuple[float, float], tuple[float, float]] | None,
    ) -> bool:
        if feasible is None:
            return False
        changed = False
        for group, (new_min, new_max) in (
            (edge_to_group[prev_i], feasible[0]),
            (edge_to_group[curr_i], feasible[1]),
        ):
            old_min, old_max = group.min_const, group.max_const
            group.min_const = max(group.min_const, new_min)
            group.max_const = min(group.max_const, new_max)
            group.ref_const = float(np.clip(group.ref_const, group.min_const, group.max_const))
            changed = changed or abs(group.min_const - old_min) > tol or abs(group.max_const - old_max) > tol
        return changed

    for _ in range(iterations):
        changed = False
        for curr_edge in ref_edges:
            prev_i, _ = neighbors[curr_edge.index]
            prev_edge = ref_edges[prev_i]
            changed = apply_pair_interval(
                prev_i,
                curr_edge.index,
                feasible_pair_interval(
                    prev_edge,
                    curr_edge,
                    const_interval(edge_to_group[prev_i]),
                    const_interval(edge_to_group[curr_edge.index]),
                    boundary,
                ),
            ) or changed

        # 方向矢量完全相同意味着每条边只能沿参考 tangent 正方向伸缩，长度不能变成负数。
        for edge in ref_edges:
            prev_i, next_i = neighbors[edge.index]
            prev_edge = ref_edges[prev_i]
            next_edge = ref_edges[next_i]
            start_poly = feasible_pair_polygon(
                prev_edge,
                edge,
                const_interval(edge_to_group[prev_i]),
                const_interval(edge_to_group[edge.index]),
                boundary,
            )
            end_poly = feasible_pair_polygon(
                edge,
                next_edge,
                const_interval(edge_to_group[edge.index]),
                const_interval(edge_to_group[next_i]),
                boundary,
            )
            if not start_poly or not end_poly:
                continue

            start_matrix = np.vstack([prev_edge.outward, edge.outward])
            end_matrix = np.vstack([edge.outward, next_edge.outward])
            start_transform = np.linalg.inv(start_matrix)
            end_transform = np.linalg.inv(end_matrix)
            start_coeff = edge.tangent @ start_transform
            end_coeff = edge.tangent @ end_transform
            start_values = [float(start_coeff @ params) for params in start_poly]
            end_values = [float(end_coeff @ params) for params in end_poly]
            start_min = min(start_values)
            end_max = max(end_values)

            # start_t <= end_max
            changed = apply_pair_interval(
                prev_i,
                edge.index,
                feasible_pair_interval(
                    prev_edge,
                    edge,
                    const_interval(edge_to_group[prev_i]),
                    const_interval(edge_to_group[edge.index]),
                    boundary,
                    ((start_coeff, end_max),),
                ),
            ) or changed
            # end_t >= start_min  等价于  -end_t <= -start_min
            changed = apply_pair_interval(
                edge.index,
                next_i,
                feasible_pair_interval(
                    edge,
                    next_edge,
                    const_interval(edge_to_group[edge.index]),
                    const_interval(edge_to_group[next_i]),
                    boundary,
                    ((-end_coeff, -start_min),),
                ),
            ) or changed
        if not changed:
            break
    return tightened


def vertex_linear_coefficients(ref_edges: list[Edge]) -> list[np.ndarray]:
    """把每个顶点写成所有边支撑线常数的线性函数。

    第 ``i`` 个顶点是前一条边和第 ``i`` 条边两条支撑线的交点，因此
    ``vertex_i = coeffs[i] @ constants``。后续用这个线性表达式建立完整轮廓
    的全局约束，而不是只看相邻两条边的局部区间。
    """

    n = len(ref_edges)
    neighbors = adjacent_edge_indices(ref_edges)
    coeffs: list[np.ndarray] = []
    for curr_edge in ref_edges:
        prev_i, _ = neighbors[curr_edge.index]
        prev_edge = ref_edges[prev_i]
        matrix = np.vstack([prev_edge.outward, curr_edge.outward])
        if abs(float(np.linalg.det(matrix))) < 1e-9:
            raise ValueError(
                f"Adjacent support lines are nearly parallel: {prev_i}, {curr_edge.index}"
            )
        transform = np.linalg.inv(matrix)
        coeff = np.zeros((2, n), dtype=float)
        coeff[:, prev_i] = transform[:, 0]
        coeff[:, curr_edge.index] = transform[:, 1]
        coeffs.append(coeff)
    return coeffs


def global_support_linear_constraints(
    ref_edges: list[Edge],
    boundary: ExplorationBoundary,
    objects: list[np.ndarray] | None = None,
    min_edge_length: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """生成完整障碍轮廓的线性可行域约束。

    这些约束对应科研需求中的全局几何条件：
    1. 所有顶点必须落在已知探索边界内；
    2. 每条边的方向矢量与 ``obstacle_ref`` 对应边完全相同，只允许沿该方向
       伸缩，不能反向翻折或退化成负长度。

    “每条边都能与 movable object 发生边界接触”的顶点范围前提，不只是边长
    为正。边段至少要能容纳 movable object 在该法向接触时的支撑跨度；否则即便
    某个接触参考点存在，物体也只能同时撞到相邻边，无法满足“与整条 obstacle
    零面积相交地接触这条边”的要求。
    """

    if objects is None:
        objects = MOVABLE_OBJECTS
    vertex_coeffs = vertex_linear_coefficients(ref_edges)
    neighbors = adjacent_edge_indices(ref_edges)
    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    for coeff in vertex_coeffs:
        # x_min <= x <= x_max, y_min <= y <= y_max
        a_rows.append(coeff[0])
        b_rows.append(boundary.x_max)
        a_rows.append(-coeff[0])
        b_rows.append(-boundary.x_min)
        a_rows.append(coeff[1])
        b_rows.append(boundary.y_max)
        a_rows.append(-coeff[1])
        b_rows.append(-boundary.y_min)

    for edge in ref_edges:
        _, next_i = neighbors[edge.index]
        start_coeff = vertex_coeffs[edge.index]
        end_coeff = vertex_coeffs[next_i]
        length_coeff = edge.tangent @ (end_coeff - start_coeff)
        local_support = support_points_lateral_interval(
            objects,
            np.zeros(2, dtype=float),
            edge.outward,
            edge.tangent,
        )
        contact_span = max(min_edge_length, float(local_support[1] - local_support[0]))
        # dot(end - start, tangent) >= contact_span
        a_rows.append(-length_coeff)
        b_rows.append(-contact_span)

    return np.vstack(a_rows), np.array(b_rows, dtype=float), vertex_coeffs


def edge_endpoint_coeffs(
    edge_index: int,
    ref_edges: list[Edge],
    vertex_coeffs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """返回某条边两个端点的线性系数。"""

    _, next_i = adjacent_edge_indices(ref_edges)[edge_index]
    return vertex_coeffs[edge_index], vertex_coeffs[next_i]


def edge_reference_endpoints(edge: Edge, ref_edges: list[Edge]) -> tuple[np.ndarray, np.ndarray]:
    """返回参考轮廓中某条边的两个端点。"""

    _, next_i = adjacent_edge_indices(ref_edges)[edge.index]
    return ref_edges[edge.index].p0, ref_edges[next_i].p0


def interval_on_axis(points: tuple[np.ndarray, np.ndarray], axis: np.ndarray) -> tuple[float, float]:
    """计算线段两个端点在某个轴上的投影区间。"""

    vals = [float(np.dot(p, axis)) for p in points]
    return min(vals), max(vals)


def separation_axis_from_reference(
    edge_i: int,
    edge_j: int,
    ref_edges: list[Edge],
    tol: float = 1e-9,
) -> tuple[np.ndarray, int] | None:
    """根据参考轮廓为两条非相邻边选择一个保持拓扑顺序的分离轴。

    非相邻边不相交是非线性约束。这里采用保守线性化：在参考轮廓中找一个能
    分开这两条边的投影轴，并在 LP 中保持相同的投影先后关系。返回的 order=1
    表示 edge_i 应位于 edge_j 的低投影侧，order=-1 表示相反。
    """

    edge_a = ref_edges[edge_i]
    edge_b = ref_edges[edge_j]
    a_points = edge_reference_endpoints(edge_a, ref_edges)
    b_points = edge_reference_endpoints(edge_b, ref_edges)
    mid_axis = 0.5 * (b_points[0] + b_points[1]) - 0.5 * (a_points[0] + a_points[1])
    candidates = [
        edge_a.outward,
        edge_a.tangent,
        edge_b.outward,
        edge_b.tangent,
    ]
    if np.linalg.norm(mid_axis) > tol:
        candidates.append(unit(mid_axis))

    best: tuple[float, np.ndarray, int] | None = None
    for axis in candidates:
        axis = unit(axis)
        a_min, a_max = interval_on_axis(a_points, axis)
        b_min, b_max = interval_on_axis(b_points, axis)
        if a_max <= b_min - tol:
            gap = b_min - a_max
            order = 1
        elif b_max <= a_min - tol:
            gap = a_min - b_max
            order = -1
        else:
            continue
        if best is None or gap > best[0]:
            best = (gap, axis, order)
    if best is None:
        return None
    return best[1], best[2]


def nonintersection_cut_rows(
    edge_i: int,
    edge_j: int,
    ref_edges: list[Edge],
    vertex_coeffs: list[np.ndarray],
    sep_tol: float = 1e-7,
) -> list[tuple[np.ndarray, float]]:
    """为一对自交非相邻边生成线性分离约束。

    如果参考轮廓中 edge_i 在 edge_j 的低投影侧，则要求 edge_i 两个端点都不
    超过 edge_j 两个端点；反之亦然。这样能排除当前自交解，并收紧后续顶点
    范围。该约束是保守近似，但符合“保持参考拓扑顺序”的假设。
    """

    axis_order = separation_axis_from_reference(edge_i, edge_j, ref_edges)
    if axis_order is None:
        return []
    axis, order = axis_order
    a0_coeff, a1_coeff = edge_endpoint_coeffs(edge_i, ref_edges, vertex_coeffs)
    b0_coeff, b1_coeff = edge_endpoint_coeffs(edge_j, ref_edges, vertex_coeffs)
    if order == 1:
        low_coeffs = (a0_coeff, a1_coeff)
        high_coeffs = (b0_coeff, b1_coeff)
    else:
        low_coeffs = (b0_coeff, b1_coeff)
        high_coeffs = (a0_coeff, a1_coeff)

    rows: list[tuple[np.ndarray, float]] = []
    for low_coeff in low_coeffs:
        for high_coeff in high_coeffs:
            # dot(low, axis) <= dot(high, axis) - sep_tol
            rows.append((axis @ (low_coeff - high_coeff), -sep_tol))
    return rows


def solve_simple_vertex_bound(
    objective: np.ndarray,
    ref_edges: list[Edge],
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: list[tuple[float, float]],
    vertex_coeffs: list[np.ndarray],
    maximize: bool = False,
    max_cuts: int = 24,
) -> float | None:
    """求一个顶点坐标极值，并用自交割平面排除无效 LP 解。"""

    local_a = np.array(a_ub, dtype=float, copy=True)
    local_b = np.array(b_ub, dtype=float, copy=True)
    cost = -objective if maximize else objective
    added_cuts: set[tuple[int, int]] = set()

    for _ in range(max_cuts + 1):
        result = linprog(
            cost,
            A_ub=local_a,
            b_ub=local_b,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return None
        constants = np.asarray(result.x, dtype=float)
        poly = polygon_from_supports(ref_edges, constants.tolist())
        if is_simple_obstacle(poly, ref_edges):
            value = float(objective @ constants)
            return value

        new_rows: list[np.ndarray] = []
        new_bs: list[float] = []
        for _, edge_i, edge_j in obstacle_self_intersections(poly, ref_edges):
            key = tuple(sorted((edge_i, edge_j)))
            if key in added_cuts:
                continue
            for row, b in nonintersection_cut_rows(edge_i, edge_j, ref_edges, vertex_coeffs):
                new_rows.append(row)
                new_bs.append(b)
            added_cuts.add(key)

        if not new_rows:
            # 找不到可靠分离轴时退回原 LP 极值，避免让范围计算整体失效。
            value = float(objective @ constants)
            return value
        local_a = np.vstack([local_a, np.vstack(new_rows)])
        local_b = np.concatenate([local_b, np.array(new_bs, dtype=float)])
    return None


def solve_feasible_support_constants(
    ref_edges: list[Edge],
    groups: dict[str, SupportGroup],
    boundary: ExplorationBoundary,
    objects: list[np.ndarray] | None = None,
    extra_rows: list[np.ndarray] | None = None,
    extra_bs: list[float] | None = None,
    max_cuts: int = 24,
) -> list[float] | None:
    """求一组满足全局约束的支撑线常数。

    该函数服务于两处：一是判断候选接触对是否“存在可行轮廓”，二是给蓝色
    估计轮廓取一个与顶点范围一致的代表解。``extra_rows``/``extra_bs`` 用来
    加入某次接触的切向重叠约束。
    """

    if objects is None:
        objects = MOVABLE_OBJECTS
    edge_to_group = edge_group_map(groups)
    bounds = [const_interval(edge_to_group[edge.index]) for edge in ref_edges]
    try:
        a_ub, b_ub, vertex_coeffs = global_support_linear_constraints(
            ref_edges,
            boundary,
            objects=objects,
        )
    except ValueError:
        return None

    local_a = np.array(a_ub, dtype=float, copy=True)
    local_b = np.array(b_ub, dtype=float, copy=True)
    if extra_rows:
        local_a = np.vstack([local_a, np.vstack(extra_rows)])
        local_b = np.concatenate([local_b, np.array(extra_bs or [], dtype=float)])

    objective = np.zeros(len(ref_edges), dtype=float)
    added_cuts: set[tuple[int, int]] = set()
    for _ in range(max_cuts + 1):
        result = linprog(
            objective,
            A_ub=local_a,
            b_ub=local_b,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return None
        constants = np.asarray(result.x, dtype=float)
        poly = polygon_from_supports(ref_edges, constants.tolist())
        if is_simple_obstacle(poly, ref_edges):
            return constants.tolist()

        new_rows: list[np.ndarray] = []
        new_bs: list[float] = []
        for _, edge_i, edge_j in obstacle_self_intersections(poly, ref_edges):
            key = tuple(sorted((edge_i, edge_j)))
            if key in added_cuts:
                continue
            for row, b in nonintersection_cut_rows(edge_i, edge_j, ref_edges, vertex_coeffs):
                new_rows.append(row)
                new_bs.append(b)
            added_cuts.add(key)

        if not new_rows:
            return None
        local_a = np.vstack([local_a, np.vstack(new_rows)])
        local_b = np.concatenate([local_b, np.array(new_bs, dtype=float)])
    return None


def global_feasible_vertex_bounds(
    ref_edges: list[Edge],
    groups: dict[str, SupportGroup],
    boundary: ExplorationBoundary,
    objects: list[np.ndarray] | None = None,
) -> list[tuple[float, float, float, float]] | None:
    """在完整轮廓可行域内求每个顶点 x/y 的最小和最大值。

    这里使用线性规划而不是枚举支撑线区间端点。原因是顶点范围必须来自同一个
    全局可行 obstacle：如果某个顶点取极值会导致其它边反向、顶点出界或轮廓
    无法保持闭合方向，则该极值不能出现在显示范围中。
    """

    edge_to_group = edge_group_map(groups)
    bounds = [const_interval(edge_to_group[edge.index]) for edge in ref_edges]
    try:
        a_ub, b_ub, vertex_coeffs = global_support_linear_constraints(
            ref_edges,
            boundary,
            objects=objects,
        )
    except ValueError:
        return None

    ranges: list[tuple[float, float, float, float]] = []
    for coeff in vertex_coeffs:
        values: list[tuple[float, float]] = []
        for dim in range(2):
            objective = coeff[dim]
            min_value = solve_simple_vertex_bound(
                objective,
                ref_edges,
                a_ub,
                b_ub,
                bounds=bounds,
                vertex_coeffs=vertex_coeffs,
                maximize=False,
            )
            max_value = solve_simple_vertex_bound(
                objective,
                ref_edges,
                a_ub,
                b_ub,
                bounds=bounds,
                vertex_coeffs=vertex_coeffs,
                maximize=True,
            )
            if min_value is None or max_value is None:
                return None
            values.append((float(min_value), float(max_value)))
        ranges.append((values[0][0], values[0][1], values[1][0], values[1][1]))
    return ranges


def vertex_errors(
    estimated: np.ndarray,
    true_poly: np.ndarray | list[np.ndarray],
) -> tuple[float, float]:
    """计算估计顶点相对真实顶点的最大误差和平均误差。"""

    true_vertices = flatten_contours(true_poly)
    if estimated.shape != true_vertices.shape:
        raise ValueError(
            f"Estimated/true vertex shapes differ: {estimated.shape} vs {true_vertices.shape}"
        )
    d = np.linalg.norm(estimated - true_vertices, axis=1)
    return float(np.max(d)), float(np.mean(d))


def vertex_uncertainty_ellipses(
    ref_edges: list[Edge],
    groups: dict[str, SupportGroup],
    boundary: ExplorationBoundary | None = None,
    objects: list[np.ndarray] | None = None,
) -> list[tuple[np.ndarray, float, float]]:
    """把每个顶点的可行坐标范围转成绘图用椭圆。

    顶点的 x/y 范围来自相邻两条支撑线的当前区间。这里用椭圆表达不确定性，
    形式上类似 SLAM 图中常见的位姿/特征点协方差椭圆；注意它是范围可视化，
    不是统计意义上的真实高斯协方差。
    """

    if boundary is not None:
        global_ranges = global_feasible_vertex_bounds(
            ref_edges,
            groups,
            boundary,
            objects=objects,
        )
        if global_ranges is not None:
            ellipses: list[tuple[np.ndarray, float, float]] = []
            for x_min, x_max, y_min, y_max in global_ranges:
                center = np.array(
                    [
                        0.5 * (x_min + x_max),
                        0.5 * (y_min + y_max),
                    ],
                    dtype=float,
                )
                ellipses.append(
                    (
                        center,
                        max(0.6, 0.5 * (x_max - x_min)),
                        max(0.6, 0.5 * (y_max - y_min)),
                    )
                )
            return ellipses
        groups = boundary_tightened_groups(ref_edges, groups, boundary)
    edge_to_group = edge_group_map(groups)
    neighbors = adjacent_edge_indices(ref_edges)
    ellipses: list[tuple[np.ndarray, float, float]] = []
    for curr_edge in ref_edges:
        prev_i, _ = neighbors[curr_edge.index]
        prev_edge = ref_edges[prev_i]
        prev_group = edge_to_group[prev_edge.index]
        curr_group = edge_to_group[curr_edge.index]
        prev_min, prev_max = const_interval(prev_group)
        curr_min, curr_max = const_interval(curr_group)

        matrix = np.vstack([prev_edge.outward, curr_edge.outward])
        if abs(float(np.linalg.det(matrix))) < 1e-9:
            continue
        vertex_candidates = [
            np.linalg.solve(matrix, np.array([prev_c, curr_c], dtype=float))
            for prev_c in (prev_min, prev_max)
            for curr_c in (curr_min, curr_max)
        ]
        xs = [float(p[0]) for p in vertex_candidates]
        ys = [float(p[1]) for p in vertex_candidates]
        x_range = (min(xs), max(xs))
        y_range = (min(ys), max(ys))

        center = np.array(
            [
                0.5 * (x_range[0] + x_range[1]),
                0.5 * (y_range[0] + y_range[1]),
            ],
            dtype=float,
        )
        radius_x = max(0.6, 0.5 * (x_range[1] - x_range[0]))
        radius_y = max(0.6, 0.5 * (y_range[1] - y_range[0]))
        ellipses.append((center, radius_x, radius_y))
    return ellipses


def line_group_observed_count(groups: dict[str, SupportGroup]) -> int:
    """统计已经被接触观测确定的支撑线数量。"""

    return sum(g.observed_const is not None for g in groups.values())


def order_reference(groups: dict[str, SupportGroup], ref_edges: list[Edge]) -> list[str]:
    """按参考多边形边序生成探测顺序。"""

    edge_to_group = {
        edge_i: group.key for group in groups.values() for edge_i in group.edge_indices
    }
    keys: list[str] = []
    for edge in ref_edges:
        key = edge_to_group[edge.index]
        if key not in keys:
            keys.append(key)
    return keys


def order_axis_anchors(groups: dict[str, SupportGroup], ref_edges: list[Edge]) -> list[str]:
    """把坐标为 0 的锚定边提前，再按参考顺序探测其它边。"""

    keys = order_reference(groups, ref_edges)
    nonzero = [k for k in keys if abs(groups[k].ref_const) > 1e-9]
    zero = [k for k in keys if abs(groups[k].ref_const) <= 1e-9]
    return zero[:1] + nonzero + zero[1:]


def plot_step(
    path: Path,
    true_poly: np.ndarray | list[np.ndarray],
    ref_poly: np.ndarray | list[np.ndarray],
    estimated_poly: np.ndarray,
    objects: list[np.ndarray],
    obs: ContactObservation,
    ref_edges: list[Edge],
    groups: dict[str, SupportGroup],
    hypotheses: list[Hypothesis],
    boundary: ExplorationBoundary,
    step_record: StepRecord,
) -> None:
    """绘制单次接触后的状态图。

    图中包含真实障碍物、参考障碍物、当前估计轮廓、探索边界、顶点不确定性
    椭圆、可移动物体接触位姿、实际四方向轨迹以及候选接触边。
    """

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # 三条轮廓线：黑色真值仅用于仿真评估，灰色参考轮廓是算法初始先验，蓝色为当前估计。
    for i, poly in enumerate(as_contours(true_poly)):
        closed = np.vstack([poly, poly[0]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="black",
            lw=2.0,
            label="obstacle_true" if i == 0 else None,
        )
    for i, poly in enumerate(contours_from_flat_vertices(estimated_poly, ref_edges)):
        closed = np.vstack([poly, poly[0]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="#2563eb",
            lw=2.0,
            label="estimated" if i == 0 else None,
        )
    ax.add_patch(
        Rectangle(
            (boundary.x_min, boundary.y_min),
            boundary.x_max - boundary.x_min,
            boundary.y_max - boundary.y_min,
            fill=False,
            edgecolor="#7c3aed",
            linewidth=1.4,
            linestyle=":",
            label="exploration boundary",
        )
    )

    # 绿色椭圆展示每个顶点当前可能落入的坐标范围。
    for center, radius_x, radius_y in vertex_uncertainty_ellipses_from_hypotheses(
        ref_edges,
        groups,
        hypotheses,
        boundary,
        objects,
    ):
        ax.add_patch(
            Ellipse(
                center,
                width=2.0 * radius_x,
                height=2.0 * radius_y,
                angle=0.0,
                facecolor="#22c55e",
                edgecolor="#15803d",
                alpha=0.12,
                linewidth=1.0,
            )
        )

    if obs.contact_ref is not None:
        # 红色方块为接触瞬间的 movable_object；绿色折线为本步真实执行轨迹。
        for poly in objects:
            obj = poly + obs.contact_ref
            obj_closed = np.vstack([obj, obj[0]])
            ax.fill(obj_closed[:, 0], obj_closed[:, 1], color="#ef4444", alpha=0.18)
            ax.plot(obj_closed[:, 0], obj_closed[:, 1], color="#ef4444", lw=1.5)
        traj = np.array(step_record.trajectory_points, dtype=float)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color="#16a34a",
            lw=1.8,
            marker="o",
            ms=3,
            label="probe path",
        )

    # 橙色粗线标出算法认为可能被本次观测命中的参考边。
    for edge_i in step_record.candidate_edges:
        e = ref_edges[edge_i]
        ax.plot(
            [e.p0[0], e.p1[0]],
            [e.p0[1], e.p1[1]],
            color="#f59e0b",
            lw=4,
            alpha=0.35,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.9", lw=0.6)
    ax.set_title(
        f"{step_record.strategy} step {step_record.step}: "
        f"candidates={len(step_record.candidate_edges)}, "
        f"max_err={step_record.max_vertex_error:.3f} mm"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(boundary.x_min - 20, boundary.x_max + 20)
    ax.set_ylim(boundary.y_min - 20, boundary.y_max + 20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_strategy(
    strategy: str,
    ordering: str,
    out_dir: Path,
    step_size: float,
    line_tol: float,
) -> dict:
    """运行一种探测策略，并返回完整实验摘要。

    每个策略都会从同一个探索边界和参考轮廓开始。循环中反复选择下一条可执行
    探测路径，记录第一接触观测，更新对应支撑线，直到所有边都被确定或策略失败。
    """

    boundary = make_exploration_boundary(OBSTACLE_TRUE, margin=50.0) ##返回boundary的四个边界值
    prior_poly = normalize_ref_to_boundary(OBSTACLE_REF, boundary) ##返回在边界内的多边形轮廓初值list
    true_edges = edge_records(OBSTACLE_TRUE)
    ref_edges = edge_records(prior_poly) ## 返回各个边的Edge对象列表，包含边的索引、端点、法向量、切向量等信息
    base_groups = group_support_lines(ref_edges, boundary) ## 返回轮廓每条边的顶点可行范围
    hypotheses: list[Hypothesis] = [Hypothesis()]
    groups = consensus_groups(base_groups, hypotheses)
    initial_pos = np.array([boundary.x_min, boundary.y_min], dtype=float)
    current_pos = initial_pos.copy()
    records: list[StepRecord] = []
    observations: list[ContactObservation] = []

    # 固定顺序类策略提前算好顺序；动态策略会在每一步重新打分。
    if ordering == "reference":
        fixed_order = order_reference(groups, ref_edges)
    elif ordering == "axis_anchors":
        fixed_order = order_axis_anchors(groups, ref_edges)
    else:
        fixed_order = []

    remaining = set(groups)
    step_idx = 0
    total_steps = 0
    success = True
    failure_reason = ""

    print_initial_vertex_ranges(
        strategy,
        ref_edges,
        base_groups,
        hypotheses,
        MOVABLE_OBJECTS,
        boundary,
    )

    while remaining:
        # 防止由于候选集或几何约束错误造成死循环；正常情况下最多每条边观测一次。
        if step_idx > len(groups) * 3:
            success = False
            failure_reason = "strategy did not converge before the loop guard"
            break

        groups = consensus_groups(base_groups, hypotheses)
        remaining = {
            key
            for key, group in groups.items()
            if group.observed_const is None
        }
        if not remaining:
            break

        move_start = current_pos.copy()
        next_plan = plan_next_disambiguating_probe(
            ordering,
            move_start,
            groups,
            ref_edges,
            true_edges,
            MOVABLE_OBJECTS,
            boundary,
            fixed_order,
            hypotheses,
            observations,
            base_groups,
            line_tol,
            step_idx,
        )
        if next_plan is None:
            success = False
            failure_reason = "No executable 4-direction path can reach a new contact."
            break
        group_key, probe, obs, full_path, new_hypotheses, candidate_edges = next_plan
        obs.candidate_edge_indices = candidate_edges

        if not new_hypotheses:
            success = False
            failure_reason = (
                f"{group_key}: no candidate contact-pair hypothesis survived "
                f"(true_edges={obs.true_edge_indices}, candidates={sorted(obs.candidate_edge_indices)})"
            )
            break
        hypotheses = new_hypotheses
        observations.append(obs)
        groups = consensus_groups(base_groups, hypotheses)
        consensus_edges = consensus_observed_edges(hypotheses)
        converged_edges = sorted(consensus_edges)
        observed_group_key = (
            # 直接展开唯一一次使用的 edge->group 映射，避免单独保留小包装函数。
            {edge_i: key for key, group in groups.items() for edge_i in group.edge_indices}.get(
                converged_edges[-1], group_key
            )
            if converged_edges
            else group_key
        )

        # 轨迹连续性检查：当前 move_start 必须等于上一轮 contact_ref。
        continuity_gap = 0.0
        if records:
            prev_end = np.array(records[-1].contact_ref, dtype=float)
            continuity_gap = float(np.linalg.norm(move_start - prev_end))
        # 将整条路径拆成“转移段”和“最终接触段”，分别统计离散动作步数。
        contact_segment_start = obs.start.copy()
        for p0, p1 in zip(full_path[:-1], full_path[1:]):
            if np.linalg.norm(np.asarray(p1) - obs.contact_ref) < 1e-9:
                contact_segment_start = np.asarray(p0, dtype=float)
                break
        transit_path_executed = cleaned_axis_path(full_path[:-1] + [contact_segment_start])
        approach_path = cleaned_axis_path([contact_segment_start, obs.contact_ref])
        # 只检查接触前路径的面积重叠；最终 contact_ref 允许边界接触但不允许面积重叠。
        max_path_overlap = max_overlap_along_path(MOVABLE_OBJECTS, full_path[:-1], OBSTACLE_TRUE) if len(full_path) > 2 else 0.0
        if max_path_overlap > 1e-8:
            success = False
            failure_reason = (
                f"{group_key}: pre-contact 4-direction path overlaps obstacle_true "
                f"(max overlap={max_path_overlap:.6f})"
            )
            break
        transit_steps = int(math.ceil(path_length_l1(transit_path_executed) / step_size))
        approach_steps = int(math.ceil(path_length_l1(approach_path) / step_size))
        total_steps += transit_steps + approach_steps
        current_pos = obs.contact_ref.copy()

        # 更新估计多边形并记录当前误差，用于报告和逐步可视化。
        # constrained_estimated_polygon 会保证蓝色估计轮廓闭合且无自交。
        estimated_poly, _, _ = estimate_from_hypotheses(
            ref_edges,
            base_groups,
            hypotheses,
            boundary,
            MOVABLE_OBJECTS,
        )
        max_err, mean_err = vertex_errors(estimated_poly, OBSTACLE_TRUE)
        fig_name = f"step_{step_idx:02d}_{observed_group_key}.png"
        record = StepRecord(
            step=step_idx,
            strategy=strategy,
            group_key=observed_group_key,
            target_edge=probe.edge_index,
            action=obs.action.tolist(),
            move_start=move_start.tolist(),
            start=move_start.tolist(),
            probe_start=obs.start.tolist(),
            contact_ref=obs.contact_ref.tolist(),
            trajectory_points=[p.tolist() for p in full_path],
            continuity_gap=continuity_gap,
            distance=obs.distance,
            transit_steps=transit_steps,
            approach_steps=approach_steps,
            total_steps=total_steps,
            true_edges=obs.true_edge_indices,
            candidate_edges=obs.candidate_edge_indices,
            hypothesis_count=len(hypotheses),
            contact_pair_converged=len(hypotheses) == 1,
            overlap_area=obs.overlap_area,
            max_path_overlap_area=max_path_overlap,
            max_vertex_error=max_err,
            mean_vertex_error=mean_err,
            figure=fig_name,
        )
        records.append(record)
        print_step_diagnostics(ref_edges, base_groups, hypotheses, MOVABLE_OBJECTS, boundary, record)
        plot_step(
            out_dir / fig_name,
            OBSTACLE_TRUE,
            prior_poly,
            estimated_poly,
            MOVABLE_OBJECTS,
            obs,
            ref_edges,
            groups,
            hypotheses,
            boundary,
            record,
        )
        step_idx += 1

    # 循环结束后再计算一次最终估计结果，即使策略失败也能输出当前最好状态。
    groups = consensus_groups(base_groups, hypotheses)
    estimated_poly, _, final_polygon_alpha = estimate_from_hypotheses(
        ref_edges,
        base_groups,
        hypotheses,
        boundary,
        MOVABLE_OBJECTS,
    )
    max_err, mean_err = vertex_errors(estimated_poly, OBSTACLE_TRUE)
    uncertainty = vertex_uncertainty_summary(
        ref_edges,
        base_groups,
        hypotheses,
        boundary,
        MOVABLE_OBJECTS,
    )

    summary = {
        "strategy": strategy,
        "ordering": ordering,
        "success": success and line_group_observed_count(groups) == len(groups) and len(hypotheses) == 1,
        "failure_reason": failure_reason,
        "support_groups_total": len(groups),
        "support_groups_observed": line_group_observed_count(groups),
        "hypothesis_count": len(hypotheses),
        "contact_pair_converged": len(hypotheses) == 1,
        "probe_count": len(records),
        "discrete_action_steps": total_steps,
        "max_vertex_error_mm": max_err,
        "mean_vertex_error_mm": mean_err,
        "max_contact_overlap_area": max(
            (record.overlap_area for record in records),
            default=0.0,
        ),
        "max_probe_path_overlap_area": max(
            (record.max_path_overlap_area for record in records),
            default=0.0,
        ),
        "max_continuity_gap": max(
            (record.continuity_gap for record in records),
            default=0.0,
        ),
        "simple_polygon_alpha": final_polygon_alpha,
        "max_vertex_range_diameter_mm": uncertainty["max_vertex_range_diameter_mm"],
        "vertex_uncertainty": uncertainty["vertices"],
        "estimated_polygon_self_intersections": obstacle_self_intersections(estimated_poly, ref_edges),
        "estimated_vertices": estimated_poly.tolist(),
        "estimated_contours": [
            poly.tolist() for poly in contours_from_flat_vertices(estimated_poly, ref_edges)
        ],
        "hypotheses": [
            {
                "observed": {
                    str(edge_i): const for edge_i, const in sorted(hyp.observed.items())
                },
                "contact_pairs": hyp.contact_pairs,
            }
            for hyp in hypotheses
        ],
        "prior_vertices": flatten_contours(prior_poly).tolist(),
        "prior_contours": [poly.tolist() for poly in as_contours(prior_poly)],
        "raw_obstacle_ref_vertices": flatten_contours(OBSTACLE_REF).tolist(),
        "raw_obstacle_ref_contours": [poly.tolist() for poly in as_contours(OBSTACLE_REF)],
        "observed_supports": {
            key: {
                "axis": group.axis,
                "ref_const": group.ref_const,
                "min_const": group.min_const,
                "max_const": group.max_const,
                "observed_const": group.observed_const,
                "normal": ref_edges[group.edge_indices[0]].outward.tolist(),
                "tangent": ref_edges[group.edge_indices[0]].tangent.tolist(),
                "edge_indices": group.edge_indices,
            }
            for key, group in groups.items()
        },
        "records": [record.__dict__ for record in records],
    }
    return summary


def write_strategy_outputs(summary: dict, strategy_dir: Path) -> None:
    """把单个策略的结果写成 JSON、CSV 和 npz。

    ``summary.json`` 便于程序读取，``trace.csv`` 便于人工检查每一步轨迹，
    ``estimated_obstacle.npz`` 便于后续 Python/NumPy 分析。
    """

    with (strategy_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (strategy_dir / "trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "group_key",
                "target_edge",
                "action",
                "move_start",
                "start",
                "probe_start",
                "contact_ref",
                "trajectory_points",
                "continuity_gap",
                "distance",
                "transit_steps",
                "approach_steps",
                "total_steps",
                "true_edges",
                "candidate_edges",
                "hypothesis_count",
                "contact_pair_converged",
                "overlap_area",
                "max_path_overlap_area",
                "max_vertex_error",
                "mean_vertex_error",
                "figure",
            ],
        )
        writer.writeheader()
        for r in summary["records"]:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    np.savez(
        strategy_dir / "estimated_obstacle.npz",
        obstacle_true=np.array([poly for poly in as_contours(OBSTACLE_TRUE)], dtype=object),
        obstacle_ref_raw=np.array([poly for poly in as_contours(OBSTACLE_REF)], dtype=object),
        obstacle_prior=np.array(summary["prior_vertices"], dtype=float),
        estimated_vertices=np.array(summary["estimated_vertices"], dtype=float),
        estimated_contours=np.array(summary["estimated_contours"], dtype=object),
    )


def write_report(best: dict, all_summaries: list[dict], out_dir: Path) -> None:
    """生成 Markdown 实验报告。"""

    lines = [
        "# Contact Mapping Experiment Report",
        "",
        "This folder is generated by `mapping_contact_experiment.py`.",
        "",
        "Model: each obstacle edge has an independent support-line constant bounded by",
        "the known exploration rectangle. `obstacle_ref` is used only as a shape/topology",
        "prior and is normalized into the exploration boundary before planning.",
        "No global scale or absolute-pose relation between `obstacle_ref` and `obstacle_true` is used.",
        "",
        "## Best Strategy",
        "",
        f"- Strategy: `{best['strategy']}`",
        f"- Success: `{best['success']}`",
        f"- Probe count: `{best['probe_count']}`",
        f"- Hypothesis count: `{best['hypothesis_count']}`",
        f"- Contact pair converged: `{best['contact_pair_converged']}`",
        f"- Discrete 4-direction action steps: `{best['discrete_action_steps']}`",
        f"- Max vertex error: `{best['max_vertex_error_mm']:.6f}` mm",
        f"- Mean vertex error: `{best['mean_vertex_error_mm']:.6f}` mm",
        f"- Max vertex range diameter: `{best['max_vertex_range_diameter_mm']:.6f}` mm",
        f"- Max contact overlap area: `{best['max_contact_overlap_area']:.12f}` mm^2",
        f"- Max pre-contact path overlap area: `{best['max_probe_path_overlap_area']:.12f}` mm^2",
        f"- Max continuity gap: `{best['max_continuity_gap']:.12f}` mm",
        "",
        "## Strategy Comparison",
        "",
        "| strategy | success | probes | action steps | max error (mm) | contact overlap (mm^2) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in all_summaries:
        lines.append(
            f"| {s['strategy']} | {s['success']} | {s['probe_count']} | "
            f"{s['discrete_action_steps']} | {s['max_vertex_error_mm']:.6f} | "
            f"{s['max_contact_overlap_area']:.12f} |"
        )
    lines.extend(
        [
            "",
            "## Recovered Vertices",
            "",
            "```text",
            np.array2string(np.array(best["estimated_vertices"]), precision=3, suppress_small=True),
            "```",
            "",
            "The figures in the best strategy folder show each contact step, the candidate contact pair set,",
            "and the current optimized obstacle contour.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """命令行入口：运行全部策略，选择最佳结果并生成报告。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="mapping_contact_experiment/results")
    parser.add_argument("--step-size", type=float, default=1.5)
    parser.add_argument("--line-tol", type=float, default=12.0)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["reference_order", "axis_anchor_order", "nearest_next", "information_gain"],
        default=["reference_order"],
        help="Subset of strategies to run.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 四种策略共享同一套几何约束，只改变下一条待探测边的排序方式。
    all_strategies: list[tuple[str, str]] = [
        ("reference_order", "reference"),
        ("axis_anchor_order", "axis_anchors"),
        ("nearest_next", "nearest"),
        ("information_gain", "information"),
    ]
    selected = set(args.strategies)
    strategies = [item for item in all_strategies if item[0] in selected]
    if not strategies:
        raise ValueError("No strategies selected.")
    summaries: list[dict] = []
    for strategy_name, ordering in strategies:
        strategy_dir = out_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        summary = run_strategy(
            strategy=strategy_name,
            ordering=ordering,
            out_dir=strategy_dir,
            step_size=args.step_size,
            line_tol=args.line_tol,
        )
        write_strategy_outputs(summary, strategy_dir)
        summaries.append(summary)

    # 优先在成功策略中选动作步数最少的；若全失败，则选最终误差更小且观测更多的。
    successful = [s for s in summaries if s["success"]]
    if successful:
        best = min(
            successful,
            key=lambda s: (s["discrete_action_steps"], s["probe_count"], s["max_vertex_error_mm"]),
        )
    else:
        best = min(
            summaries,
            key=lambda s: (s["max_vertex_error_mm"], -s["support_groups_observed"]),
        )

    with (out_dir / "best_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    with (out_dir / "all_strategy_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    write_report(best, summaries, out_dir)

    print(
        "Best strategy:",
        best["strategy"],
        "success=",
        best["success"],
        "probes=",
        best["probe_count"],
        "action_steps=",
        best["discrete_action_steps"],
        "max_error_mm=",
        f"{best['max_vertex_error_mm']:.6f}",
    )


if __name__ == "__main__":
    main()
