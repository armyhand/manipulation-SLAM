"""
触觉 marker 几何特征提取。

接触类型的几何判别依据：物体与环境的接触载荷在触觉传感器上形成
一个“活跃接触区”。通过分析位移/受力显著的 marker 点云的空间形状，
可以区分：

    点接触  ：活跃区为紧凑团簇（三维范围都小）；
    线接触  ：活跃区沿一个方向拉长（一维延展，二维紧凑）；
    面接触  ：活跃区在二维平面内展开（二维延展，法向紧凑）。

本模块使用 PCA 奇异值描述活跃区形状，输出可供 contact_classifier 使用
的特征字典。只依赖 numpy。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# 常量：接触区形状判据的默认阈值（mm 与比值）
# --------------------------------------------------------------------------

DEFAULT_POINT_EXTENT_MM = 8.0        # 活跃区主轴半径小于该值 -> 点接触
DEFAULT_LINE_ELONGATION = 0.45       # 次轴/主轴比值小于该值 -> 线接触
DEFAULT_SURFACE_PLANARITY = 0.55     # 第三轴/次轴比值小于该值 -> 面接触
DEFAULT_TOP_N = 60                   # 每侧传感器取位移最大的 marker 数量


def select_top_n_indices(matrix: np.ndarray, n: int = 50) -> np.ndarray:
    """按位移向量范数返回最显著的 marker 索引。"""
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    return np.argpartition(norms, -n)[-n:]


def contact_patch_geometry(
    positions: np.ndarray,
    displacements: np.ndarray,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, np.ndarray | float | int]:
    """分析单个传感器上的活跃接触区几何。

    参数
    ----
    positions      : (M, 3) marker 位置（传感器坐标系，使用世界对齐坐标更稳妥）
    displacements  : (M, 3) marker 位移

    返回特征
    -------
    active_indices : 活跃 marker 索引
    centroid       : 活跃区重心
    singular_values: 去中心化点云协方差的奇异值（主轴、次轴、第三轴）
    extent         : 主轴半径（mm）
    elongation     : 次轴/主轴比值
    planarity      : 第三轴/次轴比值
    n_active       : 活跃 marker 数量
    """
    positions = np.asarray(positions, dtype=float)
    displacements = np.asarray(displacements, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != displacements.shape[0]:
        raise ValueError(
            "positions and displacements must share the first (marker) dimension"
        )
    top_n = max(4, min(int(top_n), positions.shape[0]))
    indices = select_top_n_indices(displacements, n=top_n)
    active = positions[indices]

    centroid = active.mean(axis=0)
    centered = active - centroid
    _, singular_values, _ = np.linalg.svd(centered, compute_uv=True)
    singular_values = singular_values / np.sqrt(max(len(active) - 1, 1))

    extent = float(singular_values[0])
    elongation = (
        float(singular_values[1] / singular_values[0])
        if singular_values[0] > 1e-12
        else 1.0
    )
    planarity = (
        float(singular_values[2] / singular_values[1])
        if singular_values[1] > 1e-12
        else 1.0
    )

    return {
        "active_indices": indices,
        "centroid": centroid,
        "singular_values": singular_values,
        "extent": extent,
        "elongation": elongation,
        "planarity": planarity,
        "n_active": int(len(indices)),
    }


@dataclass
class ContactPatchFeatures:
    """左右两侧传感器活跃接触区的联合几何特征。"""

    left: dict[str, np.ndarray | float | int] = field(default_factory=dict)
    right: dict[str, np.ndarray | float | int] = field(default_factory=dict)
    combined_extent: float = 0.0          # 联合活跃区主轴半径（mm）
    combined_elongation: float = 1.0      # 联合活跃区次轴/主轴比值
    combined_planarity: float = 1.0       # 联合活跃区第三轴/次轴比值
    n_active_total: int = 0               # 两侧活跃 marker 总数
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 联合重心

    def as_dict(self) -> dict[str, float | np.ndarray]:
        return {
            "left_extent": float(self.left.get("extent", 0.0)),
            "left_elongation": float(self.left.get("elongation", 1.0)),
            "right_extent": float(self.right.get("extent", 0.0)),
            "right_elongation": float(self.right.get("elongation", 1.0)),
            "combined_extent": self.combined_extent,
            "combined_elongation": self.combined_elongation,
            "combined_planarity": self.combined_planarity,
            "n_active_total": self.n_active_total,
            "centroid": self.centroid,
        }


def combine_patch_features(
    left: dict[str, np.ndarray | float | int],
    right: dict[str, np.ndarray | float | int],
) -> ContactPatchFeatures:
    """合并左右两侧几何特征，并用联合点云重新计算整体形状。"""
    left_active = np.asarray(left["active_indices"])
    right_active = np.asarray(right["active_indices"])
    # 为联合形状提供原始点云：这里仅用重心+奇异值近似联合点云形状。
    # 精确做法是外部传入两侧点云，参见 compute_patch_features_from_snapshot。
    combined_extent = max(float(left["extent"]), float(right["extent"]))
    combined_elongation = max(float(left["elongation"]), float(right["elongation"]))
    combined_planarity = max(float(left["planarity"]), float(right["planarity"]))
    centroid = (
        np.asarray(left["centroid"]) * len(left_active)
        + np.asarray(right["centroid"]) * len(right_active)
    ) / max(len(left_active) + len(right_active), 1)

    return ContactPatchFeatures(
        left=left,
        right=right,
        combined_extent=combined_extent,
        combined_elongation=combined_elongation,
        combined_planarity=combined_planarity,
        n_active_total=int(len(left_active) + len(right_active)),
        centroid=centroid,
    )


def compute_patch_features_from_snapshot(
    position_left: np.ndarray,
    position_right: np.ndarray,
    displacement_left: np.ndarray,
    displacement_right: np.ndarray,
    top_n: int = DEFAULT_TOP_N,
    rotation_left: np.ndarray | None = None,
    rotation_right: np.ndarray | None = None,
) -> ContactPatchFeatures:
    """从单帧触觉数据计算左右联合接触区几何特征。

    建议传入旋转矩阵（传感器坐标系 -> 世界坐标系），使两侧点云在
    同一坐标系下做联合 PCA；不传时默认传感器坐标系即为公共系。
    """
    position_left = np.asarray(position_left, dtype=float)
    position_right = np.asarray(position_right, dtype=float)
    displacement_left = np.asarray(displacement_left, dtype=float)
    displacement_right = np.asarray(displacement_right, dtype=float)

    if rotation_left is None:
        rotation_left = np.eye(3)
    if rotation_right is None:
        rotation_right = np.eye(3)

    world_left = (rotation_left @ position_left.T).T
    world_right = (rotation_right @ position_right.T).T

    left = contact_patch_geometry(world_left, displacement_left, top_n=top_n)
    right = contact_patch_geometry(world_right, displacement_right, top_n=top_n)

    left_indices = np.asarray(left["active_indices"])
    right_indices = np.asarray(right["active_indices"])
    combined_points = np.vstack(
        [world_left[left_indices], world_right[right_indices]]
    )
    centroid = combined_points.mean(axis=0)
    centered = combined_points - centroid
    _, singular_values, _ = np.linalg.svd(centered, compute_uv=True)
    singular_values = singular_values / np.sqrt(max(len(combined_points) - 1, 1))

    extent = float(singular_values[0])
    elongation = (
        float(singular_values[1] / singular_values[0])
        if singular_values[0] > 1e-12
        else 1.0
    )
    planarity = (
        float(singular_values[2] / singular_values[1])
        if singular_values[1] > 1e-12
        else 1.0
    )

    return ContactPatchFeatures(
        left=left,
        right=right,
        combined_extent=extent,
        combined_elongation=elongation,
        combined_planarity=planarity,
        n_active_total=int(len(left_indices) + len(right_indices)),
        centroid=centroid,
    )
