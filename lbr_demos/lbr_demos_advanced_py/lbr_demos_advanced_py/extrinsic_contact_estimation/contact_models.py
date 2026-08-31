"""
接触几何模型与残差函数。

本模块实现三类外部接触（extrinsic contact）的几何表示、SE(3) 位姿代数
以及用于因子图优化和力矩分类的残差函数。因子设计与以下文献对应：

    "Simultaneous Tactile Estimation and Control of Extrinsic Contact"
        (Y. Luo, S. Wang, K. Swaminathan, C. K. Liu, D. Rus, A. Rodriguez; RSS 2021)

其中各因子含义（符号沿用现有 robot_pivoting_estimate 代码）:

    Ftac   : 触觉位移测量，约束 g_i^{-1} o_i 等于由 marker 位移解算的相对变换；
    Foc    : 接触几何（点/线/面）在物体坐标系中保持不变（相邻帧）；
    Fcc    : 接触几何在世界坐标系中缓慢变化；
    Fsurf  : 接触点/线落在已知环境平面上；
    Ftorq  : 力矩约束 —— 点接触在接触点处无合外力矩；
             线接触绕接触线无摩擦扭矩；面接触绕面法向无扭矩；
    Fwr    : 由触觉位移回归力/力矩（可选，实时管线直接读合力矩）；
    Ffric  : 摩擦力锥约束（可选，用于分类与旋转角度约束）。

单位约定（与现有管线一致）：位置/点在 mm，旋转角在 rad，力在 N，力矩在 N*mm。

本模块只依赖 numpy，不依赖 gtsam，便于单独测试。
"""

from __future__ import annotations

import numpy as np
from enum import Enum


# --------------------------------------------------------------------------
# SE(3) 位姿代数
# --------------------------------------------------------------------------

def skew(vector: np.ndarray) -> np.ndarray:
    """三维向量对应的反对称矩阵。"""
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues 公式，旋转向量 -> 旋转矩阵。"""
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3) + skew(rotvec)
    axis = rotvec / theta
    axis_hat = skew(axis)
    return (
        np.eye(3)
        + np.sin(theta) * axis_hat
        + (1.0 - np.cos(theta)) * (axis_hat @ axis_hat)
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 旋转向量。"""
    cos_theta = (np.trace(rotation) - 1.0) / 2.0
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    if theta < 1e-12:
        return np.array(
            [
                0.5 * (rotation[2, 1] - rotation[1, 2]),
                0.5 * (rotation[0, 2] - rotation[2, 0]),
                0.5 * (rotation[1, 0] - rotation[0, 1]),
            ]
        )
    return (
        theta
        / (2.0 * np.sin(theta))
        * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
    )


def make_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """由旋转矩阵和平移向量构造 4x4 齐次位姿矩阵。"""
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def pose_inverse(pose: np.ndarray) -> np.ndarray:
    """求 4x4 齐次位姿矩阵的逆。"""
    inverse = np.eye(4)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inverse


def pose_between(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """left 到 right 的相对位姿，即 right 在 left 坐标系下的表示。"""
    return pose_inverse(left) @ right


def pose_transform(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    """把三维点经位姿变换到目标坐标系。"""
    return pose[:3, :3] @ np.asarray(point, dtype=float) + pose[:3, 3]


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """SE(3) 李代数向量 (rotvec3, trans3) -> 位姿矩阵。"""
    xi = np.asarray(xi, dtype=float).reshape(6)
    return make_pose(so3_exp(xi[:3]), xi[3:])


def se3_log(pose: np.ndarray) -> np.ndarray:
    """位姿矩阵 -> SE(3) 李代数向量 (rotvec3, trans3)。"""
    return np.r_[so3_log(pose[:3, :3]), pose[:3, 3]]


def perturb_pose_right(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """对位姿做右扰动：pose 被右乘 exp(delta)。"""
    return pose @ se3_exp(delta)


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """向量归一化，范数过小时返回备用方向（默认 +x），避免除零。"""
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0])
        return np.asarray(fallback, dtype=float).copy()
    return vector / norm


def line_point_near_reference(
    point: np.ndarray, direction: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """把直线上的代表点移动到最靠近参考点的位置，便于稳定显示与命令。"""
    direction = normalize(direction)
    return point + np.dot(reference - point, direction) * direction


def point_on_line(p: np.ndarray, u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """把直线 {p + t u} 上离点 x 最近的点求出。"""
    u = normalize(u)
    return p + np.dot(x - p, u) * u


# --------------------------------------------------------------------------
# 接触类型
# --------------------------------------------------------------------------

class ContactType(Enum):
    """接触类型：点接触 / 线接触 / 面接触。"""

    POINT = "point"
    LINE = "line"
    SURFACE = "surface"

    def __str__(self) -> str:
        return self.value


# --------------------------------------------------------------------------
# 接触几何在世界/物体坐标系之间的转换
# --------------------------------------------------------------------------

def point_in_pose_frame(
    pose: np.ndarray, point_world: np.ndarray
) -> np.ndarray:
    """把世界坐标系下的点变换到位姿 pose 的局部坐标系。"""
    return pose_transform(pose_inverse(pose), point_world)


def line_in_pose_frame(
    pose: np.ndarray, line_point_world: np.ndarray, line_direction_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """把世界坐标系下的接触线转换到给定位姿的局部坐标系。"""
    inverse = pose_inverse(pose)
    local_point = inverse[:3, :3] @ line_point_world + inverse[:3, 3]
    local_direction = normalize(pose[:3, :3].T @ normalize(line_direction_world))
    return local_point, local_direction


def plane_in_pose_frame(
    pose: np.ndarray, plane_point_world: np.ndarray, plane_normal_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """把世界坐标系下的平面转换到给定位姿的局部坐标系。"""
    inverse = pose_inverse(pose)
    local_point = inverse[:3, :3] @ plane_point_world + inverse[:3, 3]
    local_normal = normalize(pose[:3, :3].T @ normalize(plane_normal_world))
    return local_point, local_normal


# --------------------------------------------------------------------------
# 因子残差函数
# --------------------------------------------------------------------------

# ----- Ftac：触觉位移测量 ------------------------------------------------

def tactile_residual(
    object_pose: np.ndarray,
    gripper_pose: np.ndarray,
    measurement: np.ndarray,
) -> np.ndarray:
    """由触觉位移得到的相对物体位姿与测量值之差（6 维）。"""
    predicted = se3_log(pose_between(gripper_pose, object_pose))
    return predicted - np.asarray(measurement, dtype=float).reshape(6)


# ----- Foc：接触几何在物体坐标系中保持不变（相邻帧） --------------------

def point_object_contact_residual(
    previous_pose: np.ndarray,
    previous_contact_world: np.ndarray,
    current_pose: np.ndarray,
    current_contact_world: np.ndarray,
) -> np.ndarray:
    """点接触：接触点在物体坐标系中的坐标逐帧保持不变（3 维）。"""
    previous_local = point_in_pose_frame(previous_pose, previous_contact_world)
    current_local = point_in_pose_frame(current_pose, current_contact_world)
    return current_local - previous_local


def line_object_contact_residual(
    previous_pose: np.ndarray,
    previous_point: np.ndarray,
    previous_direction: np.ndarray,
    current_pose: np.ndarray,
    current_point: np.ndarray,
    current_direction: np.ndarray,
) -> np.ndarray:
    """线接触：接触线在物体坐标系中的表示逐帧保持不变（6 维）。

    线上点的残差取“垂直于平均方向”的分量，以去除沿直线方向的冗余自由度。
    """
    previous_local_point, previous_local_direction = line_in_pose_frame(
        previous_pose, previous_point, previous_direction
    )
    current_local_point, current_local_direction = line_in_pose_frame(
        current_pose, current_point, current_direction
    )
    if np.dot(previous_local_direction, current_local_direction) < 0.0:
        current_local_direction = -current_local_direction

    average_direction = normalize(previous_local_direction + current_local_direction)
    point_delta = current_local_point - previous_local_point
    point_residual = point_delta - average_direction * np.dot(
        point_delta, average_direction
    )
    direction_residual = current_local_direction - previous_local_direction
    return np.r_[point_residual, direction_residual]


def plane_object_contact_residual(
    previous_pose: np.ndarray,
    previous_point: np.ndarray,
    previous_normal: np.ndarray,
    current_pose: np.ndarray,
    current_point: np.ndarray,
    current_normal: np.ndarray,
) -> np.ndarray:
    """面接触：接触平面在物体坐标系中的表示逐帧保持不变（6 维）。"""
    previous_local_point, previous_local_normal = plane_in_pose_frame(
        previous_pose, previous_point, previous_normal
    )
    current_local_point, current_local_normal = plane_in_pose_frame(
        current_pose, current_point, current_normal
    )
    if np.dot(previous_local_normal, current_local_normal) < 0.0:
        current_local_normal = -current_local_normal

    average_normal = normalize(previous_local_normal + current_local_normal)
    point_delta = current_local_point - previous_local_point
    point_residual = point_delta - average_normal * np.dot(point_delta, average_normal)
    normal_residual = current_local_normal - previous_local_normal
    return np.r_[point_residual, normal_residual]


# ----- Fcc：接触几何在世界坐标系中缓慢变化（相邻帧） ---------------------

def world_point_residual(
    previous_contact_world: np.ndarray, current_contact_world: np.ndarray
) -> np.ndarray:
    """点接触：接触点在世界坐标系中逐帧缓慢变化（3 维）。"""
    return current_contact_world - previous_contact_world


def world_line_residual(
    previous_point: np.ndarray,
    previous_direction: np.ndarray,
    current_point: np.ndarray,
    current_direction: np.ndarray,
) -> np.ndarray:
    """线接触：接触线在世界坐标系中逐帧缓慢变化（6 维）。"""
    previous_direction = normalize(previous_direction)
    current_direction = normalize(current_direction)
    if np.dot(previous_direction, current_direction) < 0.0:
        current_direction = -current_direction

    average_direction = normalize(previous_direction + current_direction)
    point_delta = current_point - previous_point
    point_residual = point_delta - average_direction * np.dot(
        point_delta, average_direction
    )
    direction_residual = current_direction - previous_direction
    return np.r_[point_residual, direction_residual]


def world_plane_residual(
    previous_point: np.ndarray,
    previous_normal: np.ndarray,
    current_point: np.ndarray,
    current_normal: np.ndarray,
) -> np.ndarray:
    """面接触：接触平面在世界坐标系中逐帧缓慢变化（6 维）。"""
    previous_normal = normalize(previous_normal)
    current_normal = normalize(current_normal)
    if np.dot(previous_normal, current_normal) < 0.0:
        current_normal = -current_normal

    average_normal = normalize(previous_normal + current_normal)
    point_delta = current_point - previous_point
    point_residual = point_delta - average_normal * np.dot(point_delta, average_normal)
    normal_residual = current_normal - previous_normal
    return np.r_[point_residual, normal_residual]


def direction_unit_residual(direction: np.ndarray) -> np.ndarray:
    """单位长度约束（1 维），用于接触线方向/面法向。"""
    return np.array([np.linalg.norm(direction) - 1.0])


# ----- Fsurf：接触几何落在已知环境平面上 ---------------------------------

def point_surface_residual(
    contact_world: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """点接触：接触点到环境平面的带符号距离（1 维）。"""
    n = normalize(plane_normal)
    return np.array([np.dot(n, contact_world - plane_point)])


def line_surface_residual(
    line_point_world: np.ndarray,
    line_direction_world: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """线接触：接触线整体位于环境平面内（线上两点都在平面内，2 维）。"""
    n = normalize(plane_normal)
    p = np.asarray(line_point_world, dtype=float)
    u = normalize(line_direction_world)
    return np.array([np.dot(n, p - plane_point), np.dot(n, p + u - plane_point)])


# ----- Ftorq：接触处的力矩约束 -------------------------------------------

def moment_at_contact_gripper(
    moment_gripper: np.ndarray,
    force_gripper: np.ndarray,
    contact_gripper: np.ndarray,
    wrench_origin_gripper: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> np.ndarray:
    """把夹爪坐标系合力/合力矩折算到接触点处（夹爪坐标系）。

    杠杆臂为 contact - wrench_origin，折算后的力矩：
        m_c = m - (contact - origin) × f
    """
    lever = contact_gripper - wrench_origin_gripper
    return moment_gripper - np.cross(lever, force_gripper)


def point_torque_residual(
    contact_world: np.ndarray,
    moment_gripper: np.ndarray,
    force_gripper: np.ndarray,
    gripper_pose: np.ndarray,
    wrench_origin_gripper: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> np.ndarray:
    """点接触：接触点处的合力矩应为零（3 维）。

    对应文献中 Ftorq 的点接触形式：
        m - (c_g - eta) × f = 0
    """
    contact_gripper = point_in_pose_frame(gripper_pose, contact_world)
    return moment_at_contact_gripper(
        moment_gripper, force_gripper, contact_gripper, wrench_origin_gripper
    )


def line_torque_residual(
    line_point_world: np.ndarray,
    line_direction_world: np.ndarray,
    moment_gripper: np.ndarray,
    force_gripper: np.ndarray,
    gripper_pose: np.ndarray,
    wrench_origin_gripper: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> np.ndarray:
    """线接触：接触线处绕接触线方向无摩擦扭矩（1 维）。

    线接触可传递沿线的合力和绕线的摩擦扭矩；若忽略滚动/扭转摩擦，
    则接触处力矩在接触线方向的分量应为零，即 r = m_c · u_g。
    该式与现有 robot_pivoting_estimate 的 line_contact_torque_residual 一致。
    """
    inverse = pose_inverse(gripper_pose)
    point_gripper = inverse[:3, :3] @ line_point_world + inverse[:3, 3]
    direction_gripper = normalize(inverse[:3, :3] @ line_direction_world)
    moment_at_contact = moment_at_contact_gripper(
        moment_gripper, force_gripper, point_gripper, wrench_origin_gripper
    )
    return np.array([np.dot(moment_at_contact, direction_gripper)])


def line_perpendicular_torque_residual(
    line_point_world: np.ndarray,
    line_direction_world: np.ndarray,
    moment_gripper: np.ndarray,
    force_gripper: np.ndarray,
    gripper_pose: np.ndarray,
    wrench_origin_gripper: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> np.ndarray:
    """线接触：接触处力矩垂直于接触线方向的分量（3 维）。

    若接触线能传递任意绕线扭矩（存在扭转摩擦），则该分量应为零而
    沿线的分量自由；本残差用于诊断或作为可选的松约束。
    """
    inverse = pose_inverse(gripper_pose)
    point_gripper = inverse[:3, :3] @ line_point_world + inverse[:3, 3]
    direction_gripper = normalize(inverse[:3, :3] @ line_direction_world)
    moment_at_contact = moment_at_contact_gripper(
        moment_gripper, force_gripper, point_gripper, wrench_origin_gripper
    )
    return moment_at_contact - np.dot(moment_at_contact, direction_gripper) * direction_gripper


def plane_torque_residual(
    plane_point_world: np.ndarray,
    plane_normal_world: np.ndarray,
    moment_gripper: np.ndarray,
    force_gripper: np.ndarray,
    gripper_pose: np.ndarray,
    wrench_origin_gripper: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> np.ndarray:
    """面接触：接触处绕面法向的扭矩应为零（1 维）。

    平面摩擦接触可传递面内的合力和绕面内两轴（垂直于法向）的力矩，
    但理想情况下不能传递绕法向的扭转力矩。
    """
    inverse = pose_inverse(gripper_pose)
    point_gripper = inverse[:3, :3] @ plane_point_world + inverse[:3, 3]
    normal_gripper = normalize(inverse[:3, :3] @ plane_normal_world)
    moment_at_contact = moment_at_contact_gripper(
        moment_gripper, force_gripper, point_gripper, wrench_origin_gripper
    )
    return np.array([np.dot(moment_at_contact, normal_gripper)])


# ----- Fwr：触觉位移回归力/力矩（可选） -----------------------------------

def wrench_regression_residual(
    object_pose: np.ndarray,
    moment: np.ndarray,
    force: np.ndarray,
    gripper_pose: np.ndarray,
    stiffness: np.ndarray,
) -> np.ndarray:
    """由触觉位移预测的夹爪系力/力矩与测量之差（6 维）。

        [m, f] = K · se3_log(g_i^{-1} o_i)
    K 为触觉刚度对角矩阵（旋转部分与平移部分量纲不同）。
    """
    predicted_delta = se3_log(pose_between(gripper_pose, object_pose))
    return np.r_[moment, force] - stiffness * predicted_delta


# ----- Ffric：摩擦力锥约束（可选，用于分类与角度选择） ---------------------

def friction_cone_residual(
    normal_force: float,
    tangential_force: float,
    mu: float,
) -> np.ndarray:
    """摩擦力锥软约束：|f_t| <= mu * |f_n|（1 维）。

    仅当接触力滑出锥面时产生残差（max 形式），用于分类校验和
    旋转角度约束；不参与因子图主优化以免引入非平滑项。
    """
    return np.array([max(0.0, abs(tangential_force) - mu * abs(normal_force))])


# --------------------------------------------------------------------------
# 接触力矩的模型拟合分数（用于接触类型分类）
# --------------------------------------------------------------------------

def wrench_model_scores(
    force_gripper: np.ndarray,
    moment_gripper: np.ndarray,
    gripper_pose: np.ndarray,
    candidate_point: np.ndarray,
    candidate_direction: np.ndarray | None = None,
    candidate_normal: np.ndarray | None = None,
    wrench_origin_gripper: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> dict[str, float]:
    """用夹爪合力/力矩评估三类接触模型的拟合程度。

    返回：
        point_score    ：点接触残差 ||m_c||（越小越接近纯力点接触）；
        line_score     ：线接触残差 |m_c·u|（越小越接近无扭转线接触）；
        surface_score  ：面接触残差 |m_c·n|（越小越接近理想平面接触）。
    """
    point_score = np.linalg.norm(
        point_torque_residual(
            candidate_point, moment_gripper, force_gripper, gripper_pose, wrench_origin_gripper
        )
    )
    line_score = np.inf
    if candidate_direction is not None:
        line_score = float(
            np.linalg.norm(
                line_torque_residual(
                    candidate_point,
                    candidate_direction,
                    moment_gripper,
                    force_gripper,
                    gripper_pose,
                    wrench_origin_gripper,
                )
            )
        )
    surface_score = np.inf
    if candidate_normal is not None:
        surface_score = float(
            np.linalg.norm(
                plane_torque_residual(
                    candidate_point,
                    candidate_normal,
                    moment_gripper,
                    force_gripper,
                    gripper_pose,
                    wrench_origin_gripper,
                )
            )
        )
    return {
        "point": float(point_score),
        "line": line_score,
        "surface": surface_score,
    }
