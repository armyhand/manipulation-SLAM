"""
数据输入输出与预处理。

提供与 pose_planning_node_realtime_contact_line.py 的数据快照结构
一致的数据容器，以及从 npz / 内存缓存计算夹爪位姿、合力矩、
触觉相对变换、接触线初值等预处理函数。本模块自包含实现，
不依赖现有 robot_pivoting_estimate 仓库（几何常量保持一致）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_models import make_pose, normalize, pose_inverse

# 夹爪 <-> 触觉传感器标定常量（mm），与现有代码一致。
ROTATION_LEFT_GT = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=float)
ROTATION_RIGHT_GT = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)
TRANS_LEFT_GT = np.array([[0.0, 4.0, 0.0]], dtype=float)
TRANS_RIGHT_GT = np.array([[0.0, -4.0, 0.0]], dtype=float)
# 夹爪坐标系 <-> 世界坐标系（传感器安装位置）的常值变换。
TRANSFORMS_GS = np.array(
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 223], [0, 0, 0, 1]], dtype=float
)
WRENCH_ORIGIN_GRIPPER_MM = np.array([0.0, 0.0, 223.0])
MARKER_COUNT = 400


@dataclass
class TactileSnapshot:
    """一帧或多帧触觉+位姿数据的快照（与节点内存缓存结构一致）。"""

    contact_start_frame: int = 0
    Pose: list = field(default_factory=list)          # 位置，m
    Quat: list = field(default_factory=list)          # 四元数 xyzw
    Position_left: list = field(default_factory=list)  # (T, 400, 3)
    Position_right: list = field(default_factory=list)
    Displacement_left: list = field(default_factory=list)
    Displacement_right: list = field(default_factory=list)
    Force_dis_left: list = field(default_factory=list)  # (T, 400, 3)
    Force_dis_right: list = field(default_factory=list)
    F_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    F_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    F_z: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def frame_count(self) -> int:
        return len(self.Pose)

    def frame(self, index: int) -> "TactileSnapshot":
        """返回只含第 index 帧数据的快照。"""
        return TactileSnapshot(
            contact_start_frame=self.contact_start_frame,
            Pose=[self.Pose[index]],
            Quat=[self.Quat[index]],
            Position_left=[self.Position_left[index]],
            Position_right=[self.Position_right[index]],
            Displacement_left=[self.Displacement_left[index]],
            Displacement_right=[self.Displacement_right[index]],
            Force_dis_left=[self.Force_dis_left[index]],
            Force_dis_right=[self.Force_dis_right[index]],
            F_x=np.asarray([self.F_x[index]]),
            F_y=np.asarray([self.F_y[index]]),
            F_z=np.asarray([self.F_z[index]]),
        )


def load_snapshot_from_npz(path: Path | str, contact_start_frame: int | None = None) -> TactileSnapshot:
    """从 npz 数据文件构造快照（与节点保存的 npz 结构一致）。"""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        n = len(data["Pose"])
        contact_start = (
            contact_start_frame
            if contact_start_frame is not None
            else max(0, n - 10)
        )
        return TactileSnapshot(
            contact_start_frame=contact_start,
            Pose=[np.asarray(row, dtype=float) for row in data["Pose"][:n]],
            Quat=[np.asarray(row, dtype=float) for row in data["Quat"][:n]],
            Position_left=[
                np.asarray(row, dtype=float).reshape(MARKER_COUNT, 3)
                for row in data["Position_left"][:n]
            ],
            Position_right=[
                np.asarray(row, dtype=float).reshape(MARKER_COUNT, 3)
                for row in data["Position_right"][:n]
            ],
            Displacement_left=[
                np.asarray(row, dtype=float).reshape(MARKER_COUNT, 3)
                for row in data["Displacement_left"][:n]
            ],
            Displacement_right=[
                np.asarray(row, dtype=float).reshape(MARKER_COUNT, 3)
                for row in data["Displacement_right"][:n]
            ],
            Force_dis_left=[
                np.asarray(row, dtype=float).reshape(MARKER_COUNT, 3)
                for row in data["Fordis_left"][:n]
            ],
            Force_dis_right=[
                np.asarray(row, dtype=float).reshape(MARKER_COUNT, 3)
                for row in data["Fordis_right"][:n]
            ],
            F_x=np.asarray(data["F_x"][:n], dtype=float),
            F_y=np.asarray(data["F_y"][:n], dtype=float),
            F_z=np.asarray(data["F_z"][:n], dtype=float),
        )


def select_top_n_indices(matrix: np.ndarray, n: int = 50, component: int = 2) -> np.ndarray:
    """按指定分量的绝对值选出位移最大的 marker 索引。"""
    values = np.abs(matrix[:, component])
    return np.argpartition(values, -n)[-n:]


def estimate_rigid_transform_kabsch(
    source_points: np.ndarray, target_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch 方法估计 source 到 target 的刚体旋转与平移。"""
    source_points = np.asarray(source_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    centered_source = source_points - source_centroid
    centered_target = target_points - target_centroid
    u, _, vt = np.linalg.svd(centered_source.T @ centered_target)
    v = vt.T
    rotation = v @ u.T
    if np.linalg.det(rotation) < 0:
        v[:, -1] *= -1.0
        rotation = v @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def poses_from_snapshot(
    snapshot: TactileSnapshot, frames: np.ndarray
) -> np.ndarray:
    """由快照中指定帧的夹爪位姿构造 4x4 齐次矩阵（单位 mm）。"""
    frames = np.asarray(frames, dtype=int).reshape(-1)
    positions_mm = np.asarray(snapshot.Pose, dtype=float).reshape(-1, 3)[frames] * 1000.0
    quaternions_xyzw = np.asarray(snapshot.Quat, dtype=float).reshape(-1, 4)[frames]
    poses = []
    for position, quaternion in zip(positions_mm, quaternions_xyzw):
        rotation = Rotation.from_quat(quaternion).as_matrix()
        poses.append(make_pose(rotation, position))
    return np.asarray(poses)


def wrenches_from_snapshot(
    snapshot: TactileSnapshot,
    frames: np.ndarray,
    baseline_frames: int = 10,
    top_n: int = 50,
    wrench_origin_gripper: np.ndarray = WRENCH_ORIGIN_GRIPPER_MM,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算夹爪坐标系合力和合力矩（对指定帧），并进行基线校正。

    与现有 wrenches_from_realtime 一致的算法：
        1. F_x/F_y/F_z 做基线校正；
        2. 由 marker 位置与分布力计算传感器坐标系下的力矩；
        3. 转换到世界坐标系并做旋转中心补偿。
    """
    frames = np.asarray(frames, dtype=int).reshape(-1)
    frame_count = snapshot.frame_count
    if np.any(frames < 0) or np.any(frames >= frame_count):
        raise IndexError("Wrench frame index is outside the snapshot buffer.")

    forces = np.column_stack(
        [np.asarray(snapshot.F_x), np.asarray(snapshot.F_y), np.asarray(snapshot.F_z)]
    ).astype(float, copy=False)
    baseline_count = min(max(int(baseline_frames), 1), len(forces))
    forces = forces - np.mean(forces[:baseline_count], axis=0)

    position_left = np.asarray(snapshot.Position_left, dtype=float)
    position_right = np.asarray(snapshot.Position_right, dtype=float)
    displacement_left = np.asarray(snapshot.Displacement_left, dtype=float)
    displacement_right = np.asarray(snapshot.Displacement_right, dtype=float)
    fordis_left = np.asarray(snapshot.Force_dis_left, dtype=float)
    fordis_right = np.asarray(snapshot.Force_dis_right, dtype=float)
    fordis_left = fordis_left - np.mean(fordis_left[:baseline_count], axis=0)
    fordis_right = fordis_right - np.mean(fordis_right[:baseline_count], axis=0)

    # 传感器坐标系下的力矩，转换到世界坐标系。
    torque_left_sensor = np.cross(position_left, fordis_left)  # (T, 400, 3)
    torque_right_sensor = np.cross(position_right, fordis_right)
    torque_left_world = np.array(
        [(ROTATION_LEFT_GT @ torque_left_sensor[t].T).T for t in range(len(torque_left_sensor))]
    )
    torque_right_world = np.array(
        [(ROTATION_RIGHT_GT @ torque_right_sensor[t].T).T for t in range(len(torque_right_sensor))]
    )

    # 旋转中心补偿项（与现有实现一致，简化取传感器区域均值附近的杠杆）。
    left_initial_world = (ROTATION_LEFT_GT @ position_left[0].T).T
    right_initial_world = (ROTATION_RIGHT_GT @ position_right[0].T).T
    d_left = -(left_initial_world.mean(axis=0) + np.array([0.0, -4.0, 0.0]))
    d_right = -(right_initial_world.mean(axis=0) + np.array([0.0, 4.0, 0.0]))

    fordis_left_W = np.array([(ROTATION_LEFT_GT @ fordis_left[t].T).T for t in range(len(fordis_left))])
    fordis_right_W = np.array([(ROTATION_RIGHT_GT @ fordis_right[t].T).T for t in range(len(fordis_right))])
    torque_left_add_world = np.array(
        [np.cross(d_left, fordis_left_W[t]) for t in range(len(fordis_left_W))]
    )
    torque_right_add_world = np.array(
        [np.cross(d_right, fordis_right_W[t]) for t in range(len(fordis_right_W))]
    )
    torque_additional = np.sum(torque_left_add_world, axis=1) + np.sum(
        torque_right_add_world, axis=1
    )

    moments = (
        np.sum(torque_left_world, axis=1)
        + np.sum(torque_right_world, axis=1)
        + torque_additional
    )
    return forces[frames], moments[frames]


def tactile_transforms_from_snapshot(
    snapshot: TactileSnapshot,
    frames: np.ndarray,
    top_n: int = 50,
) -> np.ndarray:
    """由触觉 marker 位移估计每个分析帧的触觉相对物体变换（4x4, mm）。

    算法：在接触参考帧选取位移最大的 marker 子集，用 Kabsch 配准
    解出该子集从初始到当前的刚体变换，再乘上夹爪->世界常值变换。
    """
    frames = np.asarray(frames, dtype=int).reshape(-1)
    contact_start_frame = int(snapshot.contact_start_frame)
    position_left = np.asarray(snapshot.Position_left, dtype=float)
    position_right = np.asarray(snapshot.Position_right, dtype=float)
    displacement_left = np.asarray(snapshot.Displacement_left, dtype=float)
    displacement_right = np.asarray(snapshot.Displacement_right, dtype=float)

    selection_frame = max(0, contact_start_frame - 10)
    left_indices = select_top_n_indices(displacement_left[selection_frame], n=top_n)
    right_indices = select_top_n_indices(displacement_right[selection_frame], n=top_n)

    left_initial = (ROTATION_LEFT_GT @ position_left[contact_start_frame].T).T + TRANS_LEFT_GT
    right_initial = (ROTATION_RIGHT_GT @ position_right[contact_start_frame].T).T + TRANS_RIGHT_GT
    initial_points = np.vstack([left_initial[left_indices], right_initial[right_indices]])

    transforms_tactile = []
    for frame in frames:
        left_current = (ROTATION_LEFT_GT @ position_left[frame].T).T + TRANS_LEFT_GT
        right_current = (ROTATION_RIGHT_GT @ position_right[frame].T).T + TRANS_RIGHT_GT
        current_points = np.vstack(
            [left_current[left_indices], right_current[right_indices]]
        )
        rotation, translation = estimate_rigid_transform_kabsch(initial_points, current_points)
        transforms_tactile.append(make_pose(rotation, translation))
    transforms = [TRANSFORMS_GS @ transform for transform in transforms_tactile]
    return np.asarray(transforms)


def initial_line_from_object_poses(
    object_poses: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """根据初始物体位姿序列，用固定轴约束估计接触线初值（点 + 方向）。"""
    object_poses = np.asarray(object_poses, dtype=float)
    reference_pose = object_poses[0]
    rotations = []
    translations = []
    for pose in object_poses[1:]:
        relative = pose @ pose_inverse(reference_pose)
        rotations.append(relative[:3, :3])
        translations.append(relative[:3, 3])
    if len(rotations) == 0:
        return reference_pose[:3, 3].copy(), np.array([1.0, 0.0, 0.0])

    rotations = np.asarray(rotations)
    translations = np.asarray(translations)
    system_matrix = np.vstack([np.eye(3) - rotation for rotation in rotations])
    system_rhs = translations.reshape(-1)
    point = np.linalg.lstsq(system_matrix, system_rhs, rcond=None)[0]

    direction_matrix = np.vstack([rotation - np.eye(3) for rotation in rotations])
    _, _, vt = np.linalg.svd(direction_matrix)
    direction = normalize(vt[-1])
    return point, direction


def initial_point_from_object_poses(
    object_poses: np.ndarray,
) -> np.ndarray:
    """用 Kabsch 解共享不动点作为点接触初值（适用于纯点接触绕点转动）。"""
    object_poses = np.asarray(object_poses, dtype=float)
    reference_pose = object_poses[0]
    rotations = []
    translations = []
    for pose in object_poses[1:]:
        relative = pose @ pose_inverse(reference_pose)
        rotations.append(relative[:3, :3])
        translations.append(relative[:3, 3])
    if len(rotations) == 0:
        return reference_pose[:3, 3].copy()

    system_matrix = np.vstack([np.eye(3) - rotation for rotation in rotations])
    system_rhs = np.concatenate(translations)
    point = np.linalg.lstsq(system_matrix, system_rhs, rcond=None)[0]
    return point


def detect_contact_start(snapshot: TactileSnapshot, baseline_window: int = 80) -> int:
    """用合力幅值的鲁棒阈值自动检测接触开始帧。"""
    force = np.column_stack(
        [np.asarray(snapshot.F_x), np.asarray(snapshot.F_y), np.asarray(snapshot.F_z)]
    )
    load = np.linalg.norm(force, axis=1)
    baseline_window = min(baseline_window, len(load))
    baseline = load[:baseline_window]
    center = np.median(baseline)
    mad = np.median(np.abs(baseline - center))
    scale = max(1.4826 * mad, np.std(baseline), 1e-12)
    indices = np.flatnonzero(load > center + 6.0 * scale)
    if len(indices) == 0:
        return 0
    return int(indices[0])
