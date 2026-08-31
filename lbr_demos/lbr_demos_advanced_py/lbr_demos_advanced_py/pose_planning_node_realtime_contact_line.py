import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Vector3
import numpy as np
from .motion_planning import Motion_planning
# import ros2_numpy as rnp
from tutorial_interfaces.msg import Cloud, Array3
from .tac3d_test import gripper_force_direct, gripper_force_only
from scipy.spatial.transform import Rotation as R

import csv
import json
import math
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple
from .contact_analysis import estimate_rotation_center

CONTACT_GRAPH_DIR = Path(
    "/home/armyhand/contact_point_estimation_and_scene_reconstruct/robot_pivoting_estimate"
)
if CONTACT_GRAPH_DIR.exists() and str(CONTACT_GRAPH_DIR) not in sys.path:
    sys.path.append(str(CONTACT_GRAPH_DIR))
try:
    from contact_line_factor_graph import (  # type: ignore
        GtsamContactLineISAM2,
        ContactLineResult,
        ROTATION_LEFT_GT,
        ROTATION_RIGHT_GT,
        TRANS_LEFT_GT,
        TRANS_RIGHT_GT,
        select_top_n_indices,
        estimate_rigid_transform_kabsch,
        initial_line_from_object_poses,
        line_point_near_reference,
        evaluate_line_result,
        normalize,
    )
    from gtsam_tactile_factor_graph import make_pose, se3_log  # type: ignore
    from slipping_detection import moving_average, robust_center_scale, robust_zscore, baseline_correct, \
        compute_side_features, detect_contact_frames, group_events

except Exception as exc:
    raise ImportError(
        f"Failed to import contact-line estimation dependencies from {CONTACT_GRAPH_DIR}"
    ) from exc

"""
绕定轴旋转，旋转的角度不应为定值，而是根据物体是否出现滑移来判定。桌面距离原点竖直距离为14.5mm。
1. 接触后的稳态接触指标应该用是否出现滑移来判定，或者用指尖处的力矩来判定。旋转过程应该判定是否平稳。力矩需具有自校正功能。
2. 接触线或接触点的检测需要确保接触不变的情况下进行偏转，如何实现需要仔细考虑。
260707:
3. 被动接触的情况下已经可以较为准确的（误差5mm内）估计接触线了，下一步需要主动偏转来进一步提高估计精度，拟先采用固定接触线反馈控制。(基本实现)
4. 需要自适应地估计滑动因子，以取代固定的阈值。（已完成）
5. 以上完成后，需递归估计力和力矩信息，并加入到因子图中。(已完成，但是误差太大，需要调整因子权重)
6. 还需进一步加入摩擦因子，并思考如何区分线接触、点接触和面接触。最终的目标是实现接触后继续稳态偏转，然后根据估计的结果修正旋转中心和旋转轴，继续偏转，最终实现稳定接触和鲁棒估计。
7. 需引入旋转的力矩补偿，避免绕不同轴旋转时力矩差距过大。
"""


class MinimumJerkPosePlanner(Node):
    def __init__(self):
        super().__init__('min_jerk_pose_planner_py')
        self.Q = None
        self.cmd = Pose()
        self.v_vec = None
        self.initial = None
        self.current = None
        self.target = None
        self.total_time = 0.0
        self.sampling_time = 0.01
        self.freq = 100
        self.start_time = None
        self.X, self.Y, self.Z = None, None, None
        self.world_rotation_line_point = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.world_rotation_line_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.world_rotation_angle = np.deg2rad(-20.0)
        self.world_rotation_position_step = 0.006

        self.pose_pub = self.create_publisher(Pose, '/lbr/command/pose', 1)
        self.pose_sub = self.create_subscription(Pose, '/lbr/state/pose',
                                                 self.on_pose, 1)

        self.P_l = None  # 左手Tac3D marker点位置分布数据
        self.P_r = None  # 右手Tac3D marker点位置分布数据
        self.D_l = None  # 左手marker点位移分布数据
        self.D_r = None  # 右手marker点位移分布数据
        self.F_l = None  # 左手力分布
        self.F_r = None  # 右手力分布
        self.Fr_l = None  # 左手合力数据
        self.Fr_r = None  # 右手合力数据
        self.Mr_l = None  # 左手合力数据
        self.Mr_r = None  # 右手合力数据

        self.m = 0
        self.F_x_1 = np.zeros(0)
        self.F_y_1 = np.zeros(0)
        self.F_z_1 = np.zeros(0)
        self.R_x_1 = np.zeros(0)
        self.R_y_1 = np.zeros(0)
        self.R_z_1 = np.zeros(0)
        self.f_x_mean = None
        self.f_y_mean = None
        self.f_z_mean = None
        self.r_x_mean = None
        self.r_y_mean = None
        self.r_z_mean = None

        self.Force_left = []  # 左手合力集合
        self.Force_right = []  # 右手合力集合
        self.Matrix_left = []
        self.Matrix_right = []
        self.Force_dis_left = []  # 左手力分布集合
        self.Force_dis_right = []  # 右手力分布集合
        self.Position_left = []
        self.Position_right = []
        self.Displacement_left = []
        self.Displacement_right = []
        self.F_x, self.F_y, self.F_z, self.R_x, self.R_y, self.R_z = [], [], [], [], [], []
        self.Pose, self.Time, self.Quat = [], [], []
        self.Pose_tool_all = np.zeros(0)

        self.force_inside_flag = False
        self.force_control_flag = False
        self.move_hori_flag = False
        self.i = 0
        self.N = 0
        self.n_now = 0

        self.window_size = 20
        self.buffer = np.zeros((self.window_size, 3), dtype=np.float64)
        self.sum = np.zeros(3, dtype=np.float64)
        self.count = 0
        self.pos = 0

        self.transforms_GS = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 223],
            [0, 0, 0, 1]
        ], dtype=float)
        self.contact_start_frame = None
        self.contact_active = False
        self.contact_force_threshold = 0.2
        self.contact_threshold = None
        self.slip_threshold = None
        self.slip_score = None
        self.Slip_score = []
        self.feature_series_base = None
        self.contact_line_frame_step = 5
        self.contact_line_min_frames = self.contact_line_frame_step + 1
        self.contact_line_top_n = 50
        self.contact_line_result = None
        self.contact_line_incremental_result = None
        self.contact_line_rebuild_result = None
        self.contact_line_estimator = None
        self.contact_line_tactile_transforms = np.zeros((0, 4, 4), dtype=float)
        self.contact_line_processed_frames = []
        self.contact_line_incremental_object_poses = []
        self.contact_line_incremental_line_points = []
        self.contact_line_incremental_line_directions = []
        self.contact_line_incremental_wrench_origins = []
        self.contact_line_estimation_started = False
        self.contact_line_estimation_running = False
        self.contact_line_estimation_thread = None
        self.contact_line_lock = threading.Lock()

        self.P_l_sub = self.create_subscription(
            Cloud, '/positions_l', self.on_P_l, 10)
        self.P_r_sub = self.create_subscription(
            Cloud, '/positions_r', self.on_P_r, 10)
        self.D_l_sub = self.create_subscription(
            Cloud, '/displacements_l', self.on_D_l, 10)
        self.D_r_sub = self.create_subscription(
            Cloud, '/displacements_r', self.on_D_r, 10)
        self.F_l_sub = self.create_subscription(
            Cloud, '/forces_l', self.on_F_l, 10)
        self.F_r_sub = self.create_subscription(
            Cloud, '/forces_r', self.on_F_r, 10)
        self.Fr_l_sub = self.create_subscription(
            Vector3, '/resultant_force_l', self.on_Fr_l, 10)
        self.Fr_r_sub = self.create_subscription(
            Vector3, '/resultant_force_r', self.on_Fr_r, 10)
        self.Mr_l_sub = self.create_subscription(
            Vector3, '/resultant_moment_l', self.on_Mr_l, 10)
        self.Mr_r_sub = self.create_subscription(
            Vector3, '/resultant_moment_r', self.on_Mr_r, 10)

        # # 可选：从话题接收目标 Pose
        # self.target_sub = self.create_subscription(Pose, 'target/pose',
        #                                            self.on_target, 1

    def on_P_l(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.P_l = xyz
        # self.get_logger().info(f'P_l array shape: {xyz.shape}')

    def on_P_r(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.P_r = xyz
        # self.get_logger().info(f'P_r array shape: {xyz.shape}')

    def on_D_l(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.D_l = xyz
        # self.get_logger().info(f'D_l array shape: {xyz.shape}')

    def on_D_r(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.D_r = xyz
        # self.get_logger().info(f'D_r array shape: {xyz.shape}')

    def on_F_l(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.F_l = xyz
        # self.get_logger().info(f'F_l array shape: {xyz.shape}')

    def on_F_r(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.F_r = xyz
        # self.get_logger().info(f'F_r array shape: {xyz.shape}')

    def on_Fr_l(self, msg: Vector3):
        # 提取并清理XYZ数据
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            x_val = self.clean_single_float(msg.x)
            y_val = self.clean_single_float(msg.y)
            z_val = self.clean_single_float(msg.z)

            self.Fr_l = np.array([x_val, y_val, z_val]).reshape(1, 3)
            # self.get_logger().info(f'Fr_l: {self.Fr_l}')
        else:
            self.get_logger().info(f'Don\'t Received left positions array shape')

    def on_Fr_r(self, msg: Vector3):
        # 提取并清理XYZ数据
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            x_val = self.clean_single_float(msg.x)
            y_val = self.clean_single_float(msg.y)
            z_val = self.clean_single_float(msg.z)

            self.Fr_r = np.array([x_val, y_val, z_val]).reshape(1, 3)
            # self.get_logger().info(f'Fr_r: {self.Fr_r}')
        else:
            self.get_logger().info(f'Don\'t Received left positions array shape')

    def on_Mr_l(self, msg: Vector3):
        # 提取并清理XYZ数据
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            x_val = self.clean_single_float(msg.x)
            y_val = self.clean_single_float(msg.y)
            z_val = self.clean_single_float(msg.z)

            self.Mr_l = np.array([x_val, y_val, z_val]).reshape(1, 3)
            self.Matrix_left.append(self.Mr_l)
            # self.get_logger().info(f'Mr_r: {self.Mr_l}')
        else:
            self.get_logger().info(f'Don\'t Received left positions array shape')

    def on_Mr_r(self, msg: Vector3):
        # 提取并清理XYZ数据
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            x_val = self.clean_single_float(msg.x)
            y_val = self.clean_single_float(msg.y)
            z_val = self.clean_single_float(msg.z)

            self.Mr_r = np.array([x_val, y_val, z_val]).reshape(1, 3)
            self.Matrix_right.append(self.Mr_r)
            # self.get_logger().info(f'Mr_r: {self.Mr_r}')
        else:
            self.get_logger().info(f'Don\'t Received left positions array shape')

    def poses_from_realtime(
            self,
            pose_data: list | np.ndarray | None = None,
            quat_data: list | np.ndarray | None = None,
            frames: np.ndarray | None = None,
    ) -> np.ndarray:
        poses = []
        if pose_data is None:
            pose_data = self.Pose
        if quat_data is None:
            quat_data = self.Quat

        positions_mm = np.array(pose_data).reshape(-1, 3) * 1000
        quaternions_xyzw = np.array(quat_data).reshape(-1, 4)
        if frames is not None:
            positions_mm = positions_mm[frames]
            quaternions_xyzw = quaternions_xyzw[frames]
        for position, quaternion in zip(positions_mm, quaternions_xyzw):
            rotation = R.from_quat(quaternion).as_matrix()
            poses.append(make_pose(rotation, position))

        return np.asarray(poses)

    def wrenches_from_realtime(
            self,
            frames: np.ndarray,
            baseline_frames: int = 10,
            top_n: int = 50,
            position_left: list | np.ndarray | None = None,
            position_right: list | np.ndarray | None = None,
            displacement_left: list | np.ndarray | None = None,
            displacement_right: list | np.ndarray | None = None,
            fordis_left: list | np.ndarray | None = None,
            fordis_right: list | np.ndarray | None = None,
            F_x: list | np.ndarray | None = None,
            F_y: list | np.ndarray | None = None,
            F_z: list | np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """由实时缓存计算夹爪坐标系合力和合力矩。"""
        if position_left is None:
            position_left = self.Position_left
        if position_right is None:
            position_right = self.Position_right
        if displacement_left is None:
            displacement_left = self.Displacement_left
        if displacement_right is None:
            displacement_right = self.Displacement_right
        if fordis_left is None:
            fordis_left = self.Force_dis_left
        if fordis_right is None:
            fordis_right = self.Force_dis_right
        if F_x is None:
            F_x = self.F_x
        if F_y is None:
            F_y = self.F_y
        if F_z is None:
            F_z = self.F_z

        frames = np.asarray(frames, dtype=int).reshape(-1)
        forces = np.column_stack([F_x, F_y, F_z]).astype(float, copy=False)
        if len(forces) == 0:
            raise ValueError("No realtime wrench samples are available.")
        if np.any(frames < 0) or np.any(frames >= len(forces)):
            raise IndexError("Wrench frame index is outside the realtime buffer.")
        baseline_count = min(max(int(baseline_frames), 1), len(forces))
        forces = forces - np.mean(forces[:baseline_count], axis=0)

        position_left = np.asarray(position_left, dtype=float)
        position_right = np.asarray(position_right, dtype=float)
        displacement_left = np.asarray(displacement_left, dtype=float)
        displacement_right = np.asarray(displacement_right, dtype=float)
        fordis_left = np.asarray(fordis_left, dtype=float)
        fordis_right = np.asarray(fordis_right, dtype=float)
        tactile_histories = {
            "position_left": position_left,
            "position_right": position_right,
            "displacement_left": displacement_left,
            "displacement_right": displacement_right,
            "force_distribution_left": fordis_left,
            "force_distribution_right": fordis_right,
        }
        for name, values in tactile_histories.items():
            if values.ndim != 3 or values.shape[2] != 3:
                raise ValueError(f"{name} must have shape (frame_count, marker_count, 3).")
            if len(values) != len(forces):
                raise ValueError(
                    f"{name} has {len(values)} frames, expected {len(forces)}."
                )
        marker_counts = {values.shape[1] for values in tactile_histories.values()}
        if len(marker_counts) != 1:
            raise ValueError("Realtime tactile buffers use inconsistent marker counts.")
        if top_n <= 0 or top_n > marker_counts.pop():
            raise ValueError("top_n must be within the available tactile marker count.")

        # Tac3D 的分布力也需要使用接触前样本消除零偏。
        fordis_left = fordis_left - np.mean(fordis_left[:baseline_count], axis=0)
        fordis_right = fordis_right - np.mean(fordis_right[:baseline_count], axis=0)

        # 计算传感器坐标系下的力矩。
        torque_left_sensor = np.cross(position_left, fordis_left)  # (T, 400, 3)
        torque_right_sensor = np.cross(position_right, fordis_right)  # (T, 400, 3)
        # 转化到世界坐标系
        torque_left_world = np.array(
            [
                (ROTATION_LEFT_GT @ torque_left_sensor[t].T).T
                for t in range(len(torque_left_sensor))
            ]
        )
        torque_right_world = np.array(
            [
                (ROTATION_RIGHT_GT @ torque_right_sensor[t].T).T
                for t in range(len(torque_right_sensor))
            ]
        )
        rotation_center_world_left, rotation_center_world_right = estimate_rotation_center(
            position_left,
            position_right,
            displacement_left,
            displacement_right,
            ROTATION_LEFT_GT,
            ROTATION_RIGHT_GT,
            top_n=top_n,
            min_rotation_deg=0.5,
            min_motion_mm=0.02,
            regularization=1e-3, )
        d_right = -(rotation_center_world_right + np.array([0.0, 4.0, 0.0]))
        d_left = -(rotation_center_world_left + np.array([0.0, -4.0, 0.0]))
        ##计算转换到世界坐标系下的力矩
        fordis_left_W = np.array(
            [(ROTATION_LEFT_GT @ fordis_left[t].T).T for t in range(len(fordis_left))]
        )
        fordis_right_W = np.array(
            [(ROTATION_RIGHT_GT @ fordis_right[t].T).T for t in range(len(fordis_right))]
        )
        torque_left_add_world = np.array(
            [np.cross(d_left[t], fordis_left_W[t]) for t in range(len(fordis_left_W))]
        )  # (T, 400, 3)
        torque_right_add_world = np.array(
            [np.cross(d_right[t], fordis_right_W[t]) for t in range(len(fordis_right_W))]
        )  # (T, 400, 3)
        torque_additional = np.sum(torque_left_add_world, axis=1) + np.sum(
            torque_right_add_world, axis=1
        )  # (T, 3)
        # 计算总力矩
        total_torque_left = np.sum(torque_left_world, axis=1)  # (T, 3)
        total_torque_right = np.sum(torque_right_world, axis=1)  # (T, 3)
        moments = total_torque_left + total_torque_right + torque_additional

        return forces[frames], moments[frames]

    def tactile_transforms_from_realtime(self,
                                         frames: np.ndarray,
                                         top_n: int = 50,
                                         position_left_data: list | np.ndarray | None = None,
                                         position_right_data: list | np.ndarray | None = None,
                                         displacement_left_data: list | np.ndarray | None = None,
                                         displacement_right_data: list | np.ndarray | None = None,
                                         contact_start_frame: int | None = None,
                                         ) -> np.ndarray:
        """从触觉 marker 位移估计每个分析帧的触觉相对物体变换。"""
        if position_left_data is None:
            position_left_data = self.Position_left
        if position_right_data is None:
            position_right_data = self.Position_right
        if displacement_left_data is None:
            displacement_left_data = self.Displacement_left
        if displacement_right_data is None:
            displacement_right_data = self.Displacement_right
        if contact_start_frame is None:
            contact_start_frame = self.contact_start_frame

        frames = np.asarray(frames, dtype=int).reshape(-1)
        position_left = np.asarray(position_left_data, dtype=float)
        position_right = np.asarray(position_right_data, dtype=float)
        displacement_left = np.asarray(displacement_left_data, dtype=float)
        displacement_right = np.asarray(displacement_right_data, dtype=float)
        for name, values in (
                ("position_left", position_left),
                ("position_right", position_right),
                ("displacement_left", displacement_left),
                ("displacement_right", displacement_right),
        ):
            if values.ndim != 3 or values.shape[2] != 3:
                raise ValueError(f"{name} must have shape (frame_count, marker_count, 3).")
        frame_count = len(position_left)
        if not all(
                len(values) == frame_count
                for values in (position_right, displacement_left, displacement_right)
        ):
            raise ValueError("Realtime tactile pose buffers have inconsistent lengths.")
        if not 0 <= contact_start_frame < frame_count:
            raise IndexError("contact_start_frame is outside the realtime tactile buffer.")
        if np.any(frames < contact_start_frame) or np.any(frames >= frame_count):
            raise IndexError("Tactile transform frame is outside the contact interval.")
        marker_count = position_left.shape[1]
        if not all(
                values.shape[1] == marker_count
                for values in (position_right, displacement_left, displacement_right)
        ):
            raise ValueError("Realtime tactile pose buffers use inconsistent marker counts.")
        if top_n <= 0 or top_n > marker_count:
            raise ValueError("top_n must be within the available tactile marker count.")

        selection_frame = max(0, contact_start_frame - 10)
        left_indices = select_top_n_indices(displacement_left[selection_frame], n=top_n)

        # 将触觉marker点转换成与世界坐标系对齐
        left_initial = (
                               ROTATION_LEFT_GT @ position_left[contact_start_frame].T
                       ).T + TRANS_LEFT_GT
        # initial_points = np.vstack(
        #     [left_initial[left_indices], right_initial[right_indices]]
        # )
        initial_points = left_initial[left_indices]

        transforms_tactile = []  ## tactile【0】到tactile【i】的变换
        for frame in frames:
            left_current = (ROTATION_LEFT_GT @ position_left[frame].T).T + TRANS_LEFT_GT
            right_current = (ROTATION_RIGHT_GT @ position_right[frame].T).T + TRANS_RIGHT_GT
            # current_points = np.vstack(
            #     [left_current[left_indices], right_current[right_indices]]
            # )
            current_points = left_current[left_indices]
            rotation, translation = estimate_rigid_transform_kabsch(
                initial_points, current_points
            )
            transforms_tactile.append(make_pose(rotation, translation))
        transforms = [self.transforms_GS @ transform for transform in transforms_tactile]

        return np.asarray(transforms)

    def estimate_contact_line_realtime(self,
                                       frame_step: int = 5,
                                       top_n: int = 50,
                                       report_interval_frames: int = 10,
                                       print_progress: bool = True,
                                       snapshot: dict[str, Any] | None = None,
                                       ) -> ContactLineResult:
        """当接触后，执行增量因子图优化并返回最终与过程结果。"""
        if snapshot is None:
            snapshot = self._make_contact_line_snapshot()

        contact_start_frame = int(snapshot["contact_start_frame"])
        pose_data = snapshot["Pose"]
        quat_data = snapshot["Quat"]
        frame_count = len(pose_data)
        frames = np.arange(contact_start_frame, frame_count, frame_step, dtype=int)
        if len(frames) < 2:
            raise ValueError(
                f"Not enough frames for contact line estimation: "
                f"contact_start_frame={contact_start_frame}, frame_count={frame_count}"
            )

        gripper_poses = self.poses_from_realtime(pose_data, quat_data, frames)
        forces, moments = self.wrenches_from_realtime(
            frames,
            baseline_frames=10,
            top_n=top_n,
            position_left=snapshot["Position_left"],
            position_right=snapshot["Position_right"],
            displacement_left=snapshot["Displacement_left"],
            displacement_right=snapshot["Displacement_right"],
            fordis_left=snapshot["Force_dis_left"],
            fordis_right=snapshot["Force_dis_right"],
            F_x=snapshot["F_x"],
            F_y=snapshot["F_y"],
            F_z=snapshot["F_z"],
        )
        tactile_transforms = self.tactile_transforms_from_realtime(
            frames,
            top_n=top_n,
            position_left_data=snapshot["Position_left"],
            position_right_data=snapshot["Position_right"],
            displacement_left_data=snapshot["Displacement_left"],
            displacement_right_data=snapshot["Displacement_right"],
            contact_start_frame=contact_start_frame,
        )

        initial_object_poses = np.asarray(
            [gripper @ tactile for gripper, tactile in zip(gripper_poses, tactile_transforms)]
        )
        initial_point, initial_direction = initial_line_from_object_poses(
            initial_object_poses
        )

        estimator = GtsamContactLineISAM2(
            gripper_poses=gripper_poses,
            tactile_transforms=tactile_transforms,
            initial_point=initial_point,
            initial_direction=initial_direction,
            forces=forces,
            moments=moments,
        )
        estimate = estimator.run(
            frames=frames,
            contact_start_frame=contact_start_frame,
            report_interval_frames=report_interval_frames,
            print_progress=print_progress,
        )

        representative_direction = normalize(np.mean(estimate["line_directions"], axis=0))
        if np.dot(representative_direction, estimate["line_directions"][0]) < 0.0:
            representative_direction = -representative_direction
        reference = np.mean(gripper_poses[:, :3, 3], axis=0)
        representative_point = line_point_near_reference(
            np.mean(estimate["line_points"], axis=0),
            representative_direction,
            reference,
        )

        (
            object_line_point_errors,
            object_line_direction_errors_deg,
            world_line_point_drift,
            world_line_direction_drift_deg,
        ) = evaluate_line_result(
            estimate["object_poses"], estimate["line_points"], estimate["line_directions"]
        )

        return ContactLineResult(
            data_path=Path("realtime"),
            contact_start_frame=contact_start_frame,
            analysis_frames=frames,
            gripper_poses=gripper_poses,
            tactile_transforms=tactile_transforms,
            object_poses=estimate["object_poses"],
            line_points=estimate["line_points"],
            line_directions=estimate["line_directions"],
            wrench_origin_gripper=estimate["wrench_origin_gripper"],
            representative_point=representative_point,
            representative_direction=representative_direction,
            object_line_point_errors=object_line_point_errors,
            object_line_direction_errors_deg=object_line_direction_errors_deg,
            world_line_point_drift=world_line_point_drift,
            world_line_direction_drift_deg=world_line_direction_drift_deg,
            initial_point=initial_point,
            initial_direction=initial_direction,
            incremental_frames=estimate["incremental_frames"],
            incremental_object_poses=estimate["incremental_object_poses"],
            incremental_line_points=estimate["incremental_line_points"],
            incremental_line_directions=estimate["incremental_line_directions"],
            incremental_wrench_origins_gripper=estimate[
                "incremental_wrench_origins_gripper"
            ],
        )

    def _copy_array_list(self, values: list) -> list:
        return [np.array(value, copy=True) for value in values]

    def _make_contact_line_snapshot(self) -> dict[str, Any]:
        frame_count = len(self.Pose)
        return {
            "contact_start_frame": int(self.contact_start_frame),
            "Pose": self._copy_array_list(self.Pose[:frame_count]),
            "Quat": self._copy_array_list(self.Quat[:frame_count]),
            "Position_left": self._copy_array_list(self.Position_left[:frame_count]),
            "Position_right": self._copy_array_list(self.Position_right[:frame_count]),
            "Displacement_left": self._copy_array_list(self.Displacement_left[:frame_count]),
            "Displacement_right": self._copy_array_list(self.Displacement_right[:frame_count]),
            "Force_dis_left": self._copy_array_list(self.Force_dis_left[:frame_count]),
            "Force_dis_right": self._copy_array_list(self.Force_dis_right[:frame_count]),
            "F_x": np.asarray(self.F_x[:frame_count], dtype=float).copy(),
            "F_y": np.asarray(self.F_y[:frame_count], dtype=float).copy(),
            "F_z": np.asarray(self.F_z[:frame_count], dtype=float).copy(),
        }

    def _contact_line_snapshot_ready(self) -> bool:
        frame_count = len(self.Pose)
        if frame_count - self.contact_start_frame < self.contact_line_min_frames:
            return False
        required_histories = (
            self.Quat,
            self.Position_left,
            self.Position_right,
            self.Displacement_left,
            self.Displacement_right,
            self.Force_dis_left,
            self.Force_dis_right,
        )
        if not all(len(history) >= frame_count for history in required_histories):
            return False
        if not all(
                len(history) >= frame_count for history in (self.F_x, self.F_y, self.F_z)
        ):
            return False
        return all(
            value is not None
            for history in required_histories
            for value in history[:frame_count]
        )

    def _start_contact_line_estimation_async(self):
        if self.contact_line_estimation_running:
            return
        if not self._contact_line_snapshot_ready():
            return

        snapshot = self._make_contact_line_snapshot()
        self.contact_line_estimation_started = True
        self.contact_line_estimation_running = True
        self.get_logger().info(
            f"Starting contact line estimation from frame {snapshot['contact_start_frame']} "
            f"with {len(snapshot['Pose'])} frames."
        )

        self.contact_line_estimation_thread = threading.Thread(
            target=self._run_contact_line_estimation,
            args=(snapshot,),
            daemon=True,
        )
        self.contact_line_estimation_thread.start()

    def _run_contact_line_estimation(self, snapshot: dict[str, Any]):
        try:
            rebuild_result = self.estimate_contact_line_realtime(
                frame_step=self.contact_line_frame_step,
                top_n=self.contact_line_top_n,
                print_progress=False,
                snapshot=snapshot,
            )
            with self.contact_line_lock:
                self.contact_line_rebuild_result = rebuild_result
            # self.get_logger().info(
            #     f"rebuild_point(mm)={rebuild_result.representative_point}, "
            #     f"rebuild_direction={rebuild_result.representative_direction}"
            # )
        except Exception as exc:
            self.get_logger().error(f"Contact line estimation failed: {exc}")
        finally:
            self.contact_line_estimation_running = False

    # def _update_contact_line_incremental_estimator(self, snapshot: dict[str, Any]) -> ContactLineResult:
    #     contact_start_frame = int(snapshot["contact_start_frame"])
    #     frame_count = len(snapshot["Pose"])
    #     frames = np.arange(contact_start_frame, frame_count, self.contact_line_frame_step, dtype=int)
    #     processed_count = len(self.contact_line_processed_frames)
    #     if len(frames) <= processed_count:
    #         if self.contact_line_incremental_result is None:
    #             raise ValueError("No new contact line frame and no previous incremental result.")
    #         return self.contact_line_incremental_result

    #     new_frames = frames[processed_count:]
    #     new_gripper_poses = self.poses_from_realtime(
    #         snapshot["Pose"], snapshot["Quat"], new_frames
    #     )
    #     new_forces, new_moments = self.wrenches_from_realtime(
    #         new_frames,
    #         baseline_frames=10,
    #         top_n=self.contact_line_top_n,
    #         position_left=snapshot["Position_left"],
    #         position_right=snapshot["Position_right"],
    #         displacement_left=snapshot["Displacement_left"],
    #         displacement_right=snapshot["Displacement_right"],
    #         fordis_left=snapshot["Force_dis_left"],
    #         fordis_right=snapshot["Force_dis_right"],
    #         F_x=snapshot["F_x"],
    #         F_y=snapshot["F_y"],
    #         F_z=snapshot["F_z"],
    #     )
    #     new_tactile_transforms = self.tactile_transforms_from_realtime(
    #         new_frames,
    #         top_n=self.contact_line_top_n,
    #         position_left_data=snapshot["Position_left"],
    #         position_right_data=snapshot["Position_right"],
    #         displacement_left_data=snapshot["Displacement_left"],
    #         displacement_right_data=snapshot["Displacement_right"],
    #         contact_start_frame=contact_start_frame,
    #     )

    #     if self.contact_line_estimator is None:
    #         initial_object_poses = np.asarray(
    #             [gripper @ tactile for gripper, tactile in zip(new_gripper_poses, new_tactile_transforms)]
    #         )
    #         initial_point, initial_direction = initial_line_from_object_poses(initial_object_poses)
    #         self.contact_line_estimator = GtsamContactLineISAM2(
    #             gripper_poses=new_gripper_poses,
    #             tactile_transforms=new_tactile_transforms,
    #             initial_point=initial_point,
    #             initial_direction=initial_direction,
    #             forces=new_forces,
    #             moments=new_moments,
    #         )
    #         self.contact_line_tactile_transforms = np.asarray(new_tactile_transforms)
    #     else:
    #         self.contact_line_estimator.gripper_poses = np.concatenate(
    #             [self.contact_line_estimator.gripper_poses, new_gripper_poses],
    #             axis=0,
    #         )
    #         self.contact_line_estimator.forces = np.concatenate(
    #             [self.contact_line_estimator.forces, new_forces], axis=0
    #         )
    #         self.contact_line_estimator.moments = np.concatenate(
    #             [self.contact_line_estimator.moments, new_moments], axis=0
    #         )
    #         self.contact_line_tactile_transforms = np.concatenate(
    #             [self.contact_line_tactile_transforms, new_tactile_transforms],
    #             axis=0,
    #         )
    #         self.contact_line_estimator.tactile_measurements = np.asarray(
    #             [se3_log(transform) for transform in self.contact_line_tactile_transforms]
    #         )
    #         new_initial_object_poses = np.asarray(
    #             [gripper @ tactile for gripper, tactile in zip(new_gripper_poses, new_tactile_transforms)]
    #         )
    #         self.contact_line_estimator.initial_object_poses = np.concatenate(
    #             [self.contact_line_estimator.initial_object_poses, new_initial_object_poses],
    #             axis=0,
    #         )

    #     for index in range(processed_count, len(frames)):
    #         self.contact_line_estimator.add_step(index)
    #         step_estimate = self.contact_line_estimator.current_estimate(index + 1)
    #         self.contact_line_processed_frames.append(int(frames[index]))
    #         self.contact_line_incremental_object_poses.append(step_estimate["object_poses"][-1])
    #         self.contact_line_incremental_line_points.append(step_estimate["line_points"][-1])
    #         self.contact_line_incremental_line_directions.append(step_estimate["line_directions"][-1])
    #         self.contact_line_incremental_wrench_origins.append(
    #             step_estimate["wrench_origin_gripper"]
    #         )

    #     estimate = self.contact_line_estimator.current_estimate()
    #     representative_direction = normalize(np.mean(estimate["line_directions"], axis=0))
    #     if np.dot(representative_direction, estimate["line_directions"][0]) < 0.0:
    #         representative_direction = -representative_direction
    #     reference = np.mean(self.contact_line_estimator.gripper_poses[:, :3, 3], axis=0)
    #     representative_point = line_point_near_reference(
    #         np.mean(estimate["line_points"], axis=0),
    #         representative_direction,
    #         reference,
    #     )

    #     (
    #         object_line_point_errors,
    #         object_line_direction_errors_deg,
    #         world_line_point_drift,
    #         world_line_direction_drift_deg,
    #     ) = evaluate_line_result(
    #         estimate["object_poses"], estimate["line_points"], estimate["line_directions"]
    #     )

    #     return ContactLineResult(
    #         data_path=Path("realtime_incremental"),
    #         contact_start_frame=contact_start_frame,
    #         analysis_frames=np.asarray(self.contact_line_processed_frames),
    #         gripper_poses=self.contact_line_estimator.gripper_poses,
    #         tactile_transforms=self.contact_line_tactile_transforms,
    #         object_poses=estimate["object_poses"],
    #         line_points=estimate["line_points"],
    #         line_directions=estimate["line_directions"],
    #         wrench_origin_gripper=estimate["wrench_origin_gripper"],
    #         representative_point=representative_point,
    #         representative_direction=representative_direction,
    #         object_line_point_errors=object_line_point_errors,
    #         object_line_direction_errors_deg=object_line_direction_errors_deg,
    #         world_line_point_drift=world_line_point_drift,
    #         world_line_direction_drift_deg=world_line_direction_drift_deg,
    #         initial_point=self.contact_line_estimator.initial_point,
    #         initial_direction=self.contact_line_estimator.initial_direction,
    #         incremental_frames=np.asarray(self.contact_line_processed_frames),
    #         incremental_object_poses=np.asarray(self.contact_line_incremental_object_poses),
    #         incremental_line_points=np.asarray(self.contact_line_incremental_line_points),
    #         incremental_line_directions=np.asarray(self.contact_line_incremental_line_directions),
    #         incremental_wrench_origins_gripper=np.asarray(
    #             self.contact_line_incremental_wrench_origins
    #         ),
    #     )

    def _get_rotation_line_or_default(self) -> tuple[np.ndarray, np.ndarray]:
        with self.contact_line_lock:
            if self.contact_line_result is not None:
                return (
                    np.array(self.contact_line_rebuild_result.representative_point, dtype=np.float64,
                             copy=True) / 1000.0,
                    normalize(np.array(self.contact_line_rebuild_result.representative_direction, dtype=np.float64,
                                       copy=True)),
                )
        return (
            np.array(
                [self.current.position.x, self.current.position.y, self.current.position.z - 0.5],
                dtype=np.float64,
            ),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )

    def _reset_contact_detection_state(self):
        self.contact_active = False
        self.contact_line_result = None
        self.contact_line_incremental_result = None
        self.contact_line_rebuild_result = None
        self.contact_line_estimator = None
        self.contact_line_tactile_transforms = np.zeros((0, 4, 4), dtype=float)
        self.contact_line_processed_frames = []
        self.contact_line_incremental_object_poses = []
        self.contact_line_incremental_line_points = []
        self.contact_line_incremental_line_directions = []
        self.contact_line_incremental_wrench_origins = []
        self.contact_line_estimation_started = False
        if not self.contact_line_estimation_running:
            self.contact_line_estimation_thread = None

    def on_pose(self, msg):
        if self.initial is None:  ##确定任务规划的起点
            self.initial = msg
            self.cmd.position.x = self.initial.position.x
            self.cmd.position.y = self.initial.position.y
            self.cmd.position.z = self.initial.position.z
            self.cmd.orientation = self.initial.orientation
            self.get_logger().info('Initial pose received.')
        else:
            # 如果已有初始与目标 且 未启动轨迹
            if self.start_time is None:
                if any(v is None for v in (self.Fr_r, self.Fr_l, self.Mr_r, self.Mr_l)):
                    self.get_logger().info('Waiting for tac3d information.')
                else:
                    if -10 < self.Fr_r[0, 2] < -8 and -10 < self.Fr_l[0, 2] < -8 and self.force_inside_flag == False:
                        # self.force_inside_flag = True
                        self.start_time = self.get_clock().now().to_msg().sec + \
                                          self.get_clock().now().to_msg().nanosec * 1e-9
                        self.get_logger().info('Starting minimum jerk trajectory.')
                    else:
                        if self.i % 100 == 0:
                            self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
            # 如果轨迹已经启动
            if self.start_time is not None:
                if -15 < self.Fr_r[0, 2] < -5 and -15 < self.Fr_l[0, 2] < -5:
                    self.force_inside_flag = True
                else:
                    self.force_inside_flag = False
                    self.get_logger().info('The force is out of range!.')
                    self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')

                if self.force_inside_flag:
                    self.current = msg
                    if self.m < 20:
                        f_x, f_y, f_z, r_x, r_y, r_z = gripper_force_direct(self.Fr_l, self.Fr_r, self.Mr_l, self.Mr_r)
                        self.F_x_1 = np.append(self.F_x_1, f_x)
                        self.F_y_1 = np.append(self.F_y_1, f_y)
                        self.F_z_1 = np.append(self.F_z_1, f_z)
                        # self.R_x_1 = np.append(self.R_x_1, r_x)
                        # self.R_y_1 = np.append(self.R_y_1, r_y)
                        # self.R_z_1 = np.append(self.R_z_1, r_z)
                        self.m += 1
                    elif self.m == 20:
                        self.f_x_mean = np.mean(self.F_x_1)
                        self.f_y_mean = np.mean(self.F_y_1)
                        self.f_z_mean = np.mean(self.F_z_1)
                        # self.r_x_mean = np.mean(self.R_x_1)
                        # self.r_y_mean = np.mean(self.R_y_1)
                        # self.r_z_mean = np.mean(self.R_z_1)
                        self.get_logger().info(
                            f'The mean initial force is f_x={self.f_x_mean}, f_y={self.f_y_mean}, f_z={self.f_z_mean}')
                        self.m += 1
                    else:
                        self.publish_at_time()

        self.i += 1

    def clean_single_float(self, value):
        """清理单个浮点数，处理NaN和无穷大等奇异值"""
        import math

        try:
            # 转换为浮点数
            float_val = float(value)

            # 检查是否为NaN或无穷大
            if math.isnan(float_val) or math.isinf(float_val):
                return -99.0
            # 检查是否为异常大的数值（绝对值超过1e10）
            elif abs(float_val) > 1e10:
                return -99.0
            else:
                return float_val
        except (ValueError, TypeError):
            return -99.0  # 转换失败时返回-99

    def _realtime_tactile_sample_ready(self) -> bool:
        """检查当前时刻所需的全部 Tac3D 话题是否已提供有效样本。"""
        samples = (
            self.P_l,
            self.P_r,
            self.D_l,
            self.D_r,
            self.F_l,
            self.F_r,
            self.Fr_l,
            self.Fr_r,
            self.Mr_l,
            self.Mr_r,
        )
        if any(sample is None for sample in samples):
            return False
        marker_samples = [np.asarray(sample) for sample in samples[:6]]
        if any(
                sample.ndim != 2 or sample.shape[1] != 3
                for sample in marker_samples
        ):
            return False
        if len({sample.shape[0] for sample in marker_samples}) != 1:
            return False
        return all(np.all(np.isfinite(np.asarray(sample))) for sample in samples)

    def rotate_pose_about_world_line(self, position, quat, line_point, line_direction, angle):
        line_point = np.asarray(line_point, dtype=np.float64)
        line_direction = np.asarray(line_direction, dtype=np.float64)
        direction_norm = np.linalg.norm(line_direction)
        if direction_norm < 1e-12:
            return np.asarray(position, dtype=np.float64), quat
        line_direction = line_direction / direction_norm
        delta_rot = R.from_rotvec(line_direction * angle)
        relative_position = np.asarray(position, dtype=np.float64) - line_point
        rotated_position = delta_rot.apply(relative_position) + line_point
        current_rot = R.from_quat(quat)
        rotated = delta_rot * current_rot
        return rotated_position, rotated.as_quat()

    def generate_rotation_path_points(self, start_position, start_quat, line_point, line_direction, total_angle,
                                      position_step):
        start_position = np.asarray(start_position, dtype=np.float64)
        start_quat = np.asarray(start_quat, dtype=np.float64)
        line_point = np.asarray(line_point, dtype=np.float64)
        line_direction = np.asarray(line_direction, dtype=np.float64)
        direction_norm = np.linalg.norm(line_direction)
        if direction_norm < 1e-12:
            return [start_position.copy()], [start_quat.copy()]

        line_direction = line_direction / direction_norm
        relative_position = start_position - line_point
        radial_component = relative_position - np.dot(relative_position, line_direction) * line_direction
        rotation_radius = np.linalg.norm(radial_component)
        arc_length = rotation_radius * abs(total_angle)

        if arc_length < 1e-12 or position_step <= 0.0:
            num_segments = 1
        else:
            num_segments = max(1, int(np.ceil(arc_length / position_step)))

        path_points = []
        path_quats = []
        for step_index in range(1, num_segments + 1):
            angle = total_angle * step_index / num_segments
            rotated_position, rotated_quat = self.rotate_pose_about_world_line(
                position=start_position,
                quat=start_quat,
                line_point=line_point,
                line_direction=line_direction,
                angle=angle,
            )
            path_points.append(rotated_position)
            path_quats.append(rotated_quat)

        return path_points, path_quats

    def As_planner(self):
        if self.initial is not None and self.target is not None:
            M = Motion_planning(dx=0.001, dy=0.001, dz=0.001)
            start = np.array([self.initial.position.x, self.initial.position.y, self.initial.position.z])
            end = np.array([self.target.position.x, self.target.position.y, self.target.position.z])
            # Points_recover = M.path_searching(start=start, end=end)
            self.total_time = 0.0
            Points_recover = []
            Quats_recover = []
            Points_recover.append(start)
            q0 = np.array([self.initial.orientation.x,
                           self.initial.orientation.y,
                           self.initial.orientation.z,
                           self.initial.orientation.w])
            Quats_recover.append(q0)
            if self.N == 0:
                Points_recover.append(start.copy() + np.array([0.0, 0.0, 0.02]))
                Points_recover.append(start.copy() + np.array([0.0, 0.0, 0.0]))
                Quats_recover.append(np.array([0.0,
                                               math.sin(89.9 * math.pi / 180),
                                               0.0,
                                               math.cos(89.9 * math.pi / 180)]))
                Quats_recover.append(np.array([0.0,
                                               1.0,
                                               0.0,
                                               0.0]))

            if self.N == 1:  ##初次接触，一定要有足够的运动幅度
                Points_recover.append(start.copy() + np.array([0.05, 0.0, 0.0]))
                Points_recover.append(start.copy() + np.array([0.1, 0., 0.0]))
                Quats_recover.append(np.array([0.0,
                                               math.sin(89.9 * math.pi / 180),
                                               0.0,
                                               math.cos(89.9 * math.pi / 180)]))
                Quats_recover.append(np.array([0.0,
                                               1.0,
                                               0.0,
                                               0.0]))
            if self.N == 2:
                self.world_rotation_angle = np.deg2rad(10.0)
                self.world_rotation_line_point, self.world_rotation_line_direction = self._get_rotation_line_or_default()
                self.world_rotation_line_point = (self.world_rotation_line_point + np.array(
                    [self.current.position.x, self.current.position.y, self.current.position.z - 0.5],
                    dtype=np.float64)) / 2.0
                self.world_rotation_line_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                self.get_logger().info(
                    f'The rotation center is {self.world_rotation_line_point}, the direction is {self.world_rotation_line_direction}')
                rotation_path_points, rotation_path_quats = self.generate_rotation_path_points(
                    start_position=start,
                    start_quat=q0,
                    line_point=self.world_rotation_line_point,
                    line_direction=self.world_rotation_line_direction,
                    total_angle=self.world_rotation_angle,
                    position_step=self.world_rotation_position_step,
                )
                Points_recover.extend(rotation_path_points)
                Quats_recover.extend(rotation_path_quats)
            if self.N == 3:
                self.world_rotation_angle = np.deg2rad(-7.5)
                self.world_rotation_line_point, self.world_rotation_line_direction = self._get_rotation_line_or_default()
                self.world_rotation_line_point = (self.world_rotation_line_point + np.array(
                    [self.current.position.x, self.current.position.y, self.current.position.z - 0.5],
                    dtype=np.float64)) / 2.0
                self.world_rotation_line_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                self.get_logger().info(
                    f'The rotation center is {self.world_rotation_line_point}, the direction is {self.world_rotation_line_direction}')
                rotation_path_points, rotation_path_quats = self.generate_rotation_path_points(
                    start_position=start,
                    start_quat=q0,
                    line_point=self.world_rotation_line_point,
                    line_direction=self.world_rotation_line_direction,
                    total_angle=self.world_rotation_angle,
                    position_step=self.world_rotation_position_step,
                )
                Points_recover.extend(rotation_path_points)
                Quats_recover.extend(rotation_path_quats)
            if self.N > 3:
                self.get_logger().info('End the trajectory!')
            for j in range(len(Points_recover) - 1):
                quat_angle = abs(np.arccos(np.clip(np.dot(Quats_recover[j], Quats_recover[j + 1]), -1.0, 1.0)) * 2.0)
                self.total_time += max(np.linalg.norm(Points_recover[j + 1] - Points_recover[j]) / 0.006,
                                       quat_angle / 0.055)

            if Points_recover is not None:
                print(f'current_pose = {self.current}')
                # ===== 新增: 等待键盘输入 'q' 才继续 =====
                self.get_logger().info("Press 'q' and Enter to continue...")
                while True:
                    user_in = input()
                    if user_in == 'q':
                        break
                    else:
                        print("无效输入，请输入 'q' 或 'w'")
                self.X, self.Y, self.Z, self.Q = M.path_smoothing(Path_points=Points_recover, Path_quats=Quats_recover,
                                                                  t_final=self.total_time,
                                                                  freq=self.freq)  ##轨迹使用二次B样条曲线进行平滑处理
                self.get_logger().info(f"Q.shape={self.Q.shape}")
                self.get_logger().info("path_points get !")
            else:
                self.get_logger().info("the path is not found!!")

    def publish_at_time(self):
        self.cmd = self.current  ##发布量的原始值为当前位置
        current_time = self.get_clock().now().to_msg().sec + \
                       self.get_clock().now().to_msg().nanosec * 1e-9
        if not self._realtime_tactile_sample_ready():
            if self.i % 100 == 0:
                self.get_logger().info('Waiting for complete realtime Tac3D data.')
            return
        ##进行触觉反馈探索
        if all(v is not None for v in (self.Fr_r, self.Fr_l, self.Mr_r, self.Mr_l)):
            f_x, f_y, f_z = gripper_force_only(self.Fr_l, self.Fr_r)
            f_x = f_x - self.f_x_mean
            f_y = f_y - self.f_y_mean
            f_z = f_z - self.f_z_mean
            
            self.F_x.append(f_x)
            self.F_y.append(f_y)
            self.F_z.append(f_z)
            self.Force_dis_left.append(self.F_l)
            self.Force_dis_right.append(self.F_r)
            self.Force_left.append(self.Fr_l)
            self.Force_right.append(self.Fr_r)
            self.Position_left.append(self.P_l)
            self.Position_right.append(self.P_r)
            self.Displacement_left.append(self.D_l)
            self.Displacement_right.append(self.D_r)
            self.Pose.append(np.array(
                [self.current.position.x, self.current.position.y, self.current.position.z]))
            self.Quat.append(np.array(
                [self.current.orientation.x, self.current.orientation.y, self.current.orientation.z,
                 self.current.orientation.w]))
            self.Time.append(current_time)

            ##计算偏移量(刚度越大，偏移量应该越小)
            delta_z = np.sign(f_z - 2) * min(0.0035, 0.0035 * abs(f_z - 2)) if abs(f_z) > 0.3 else 0.0
            delta_x = np.sign(f_x - 0.05) * min(0.005, 0.005 * abs(f_x - 0.05)) if abs(f_x) > 0.1 else 0.0
            delta_y = np.sign(f_y - 0.1) * min(0.005, 0.005 * abs(f_y - 0.1)) if abs(f_y) > 0.2 else 0.0

            ##检测contact是否开始，然后进行估计
            baseline_frames = 30
            top_n = 50
            slip_sigma = 2.0
            threshold_sigma = 8.0
            if len(self.Force_dis_left) < baseline_frames:
                print('continue collect frames')
                self.R_x.append(0.0)
                self.R_y.append(0.0)
                self.R_z.append(0.0)
            else:
                """由实时缓存计算夹爪坐标系合力和合力矩。"""
                forces = np.column_stack([f_x, f_y, f_z]).astype(float, copy=False)
                position_left = np.asarray(self.Position_left, dtype=float)
                position_right = np.asarray(self.Position_right, dtype=float)
                displacement_left = np.asarray(self.Displacement_left, dtype=float)
                displacement_right = np.asarray(self.Displacement_right, dtype=float)
                fordis_left = np.asarray(self.Force_dis_left, dtype=float)
                fordis_right = np.asarray(self.Force_dis_right, dtype=float)
        
                # Tac3D 的分布力也需要使用接触前样本消除零偏。
                fordis_left = fordis_left - np.mean(fordis_left[:baseline_frames], axis=0)
                fordis_right = fordis_right - np.mean(fordis_right[:baseline_frames], axis=0)
        
                # 计算传感器坐标系下的力矩。
                torque_left_sensor = np.cross(position_left, fordis_left)  # (T, 400, 3)
                torque_right_sensor = np.cross(position_right, fordis_right)  # (T, 400, 3)
                # 转化到世界坐标系
                torque_left_world = np.array(
                    [
                        (ROTATION_LEFT_GT @ torque_left_sensor[t].T).T
                        for t in range(len(torque_left_sensor))
                    ]
                )
                torque_right_world = np.array(
                    [
                        (ROTATION_RIGHT_GT @ torque_right_sensor[t].T).T
                        for t in range(len(torque_right_sensor))
                    ]
                )
                rotation_center_world_left, rotation_center_world_right = estimate_rotation_center(
                    position_left,
                    position_right,
                    displacement_left,
                    displacement_right,
                    ROTATION_LEFT_GT,
                    ROTATION_RIGHT_GT,
                    top_n=top_n,
                    min_rotation_deg=0.5,
                    min_motion_mm=0.02,
                    regularization=1e-3, )
                d_right = -(rotation_center_world_right + np.array([0.0, 4.0, 0.0]))
                d_left = -(rotation_center_world_left + np.array([0.0, -4.0, 0.0]))
                ##计算转换到世界坐标系下的力矩
                fordis_left_W = np.array(
                    [(ROTATION_LEFT_GT @ fordis_left[t].T).T for t in range(len(fordis_left))]
                )
                fordis_right_W = np.array(
                    [(ROTATION_RIGHT_GT @ fordis_right[t].T).T for t in range(len(fordis_right))]
                )
                torque_left_add_world = np.array(
                    [np.cross(d_left[t], fordis_left_W[t]) for t in range(len(fordis_left_W))]
                )  # (T, 400, 3)
                torque_right_add_world = np.array(
                    [np.cross(d_right[t], fordis_right_W[t]) for t in range(len(fordis_right_W))]
                )  # (T, 400, 3)
                torque_additional = np.sum(torque_left_add_world, axis=1) + np.sum(
                    torque_right_add_world, axis=1
                )  # (T, 3)
                # 计算总力矩
                total_torque_left = np.sum(torque_left_world, axis=1)  # (T, 3)
                total_torque_right = np.sum(torque_right_world, axis=1)  # (T, 3)
                moment = total_torque_left + total_torque_right + torque_additional

                ##计算合力矢量
                f_vec = np.array([f_x, f_y, f_z])
                r_vec = moment
                self.R_x.append(r_vec[0])
                self.R_y.append(r_vec[1])
                self.R_z.append(r_vec[2])

                if self.force_control_flag:  ##接触力调整阶段，保持一个恒定的接触力（目前为z方向恒定，后续扩展到其他方向）
                    self.cmd.position.x += delta_x / self.freq
                    self.cmd.position.y += delta_y / self.freq
                    self.cmd.position.z += delta_z / self.freq

                    ##合力在运动方向上的分量
                    f_projec = (f_vec @ self.v_vec) / (self.v_vec @ self.v_vec) * self.v_vec
                    if self.count < self.window_size:  ##计算最近若干个值的均值
                        self.sum += r_vec
                        self.buffer[self.pos] = r_vec
                        self.count += 1
                        self.pos = (self.pos + 1) % self.window_size
                    else:
                        self.sum += r_vec - self.buffer[self.pos]
                        self.buffer[self.pos] = r_vec
                        self.pos = (self.pos + 1) % self.window_size
                        r_projec_mean = self.sum / self.count

                        if 8 < np.linalg.norm(r_projec_mean) < 10:  ##分量的大小最近若干个值均在范围内，则认为接触力调整完成。
                            self.force_control_flag = False
                            self.get_logger().info('Contact force within threshold!')
                            self.initial = None  ##此时再接收初始位置
                            self.buffer = np.zeros((self.window_size, 3), dtype=np.float64)
                            self.sum = np.zeros(3, dtype=np.float64)
                            self.count = 0
                            self.pos = 0  ##这些计算平均力的值也要归零

                            np.savez(f'Force_data_{self.N}.npz', Fordis_left=self.Force_dis_left,
                                     Fordis_right=self.Force_dis_right,
                                     Matrix_left=self.Matrix_left, Matrix_right=self.Matrix_right,
                                     Position_left=self.Position_left,
                                     Position_right=self.Position_right, Displacement_left=self.Displacement_left,
                                     Displacement_right=self.Displacement_right,
                                     F_x=self.F_x, F_y=self.F_y, F_z=self.F_z, R_x=self.R_x, R_y=self.R_y, R_z=self.R_z,
                                     Pose=self.Pose, Time=self.Time, Quat=self.Quat, slip_score=self.Slip_score)
                            print(f'save {self.N} Data!')
                            self.N += 1
                            # self._reset_contact_detection_state()
                else:
                    ##发布A_star得到的期望目标点，执行轨迹规划的结果
                    if self.X is not None:
                        if self.n_now < self.X.shape[0]:  ##轨迹追踪还没有进行完
                            current_pose = np.array(
                                [self.current.position.x, self.current.position.y, self.current.position.z])
                            last_pose = np.array([self.X[self.n_now], self.Y[self.n_now], self.Z[self.n_now]])
                            ## 姿态误差的计算
                            current_quat = np.array([self.current.orientation.x, self.current.orientation.y,
                                                     self.current.orientation.z, self.current.orientation.w])
                            last_quat = self.Q[self.n_now]
                            if (np.linalg.norm(current_pose - last_pose) < 0.002 and
                                    abs(np.arccos(np.clip(np.dot(current_quat, last_quat), -1.0,
                                                          1.0)) * 2.0) < 0.002):  ##与上一个目标点的误差足够小，开始发布下一个目标点
                                self.n_now += 1
                                if self.n_now < self.X.shape[0]:
                                    next_pose = np.array([self.X[self.n_now], self.Y[self.n_now], self.Z[self.n_now]])
                                    self.v_vec = next_pose - last_pose  ##运动方向矢量
                            else:
                                self.v_vec = last_pose - current_pose

                            ##合力在运动方向上的分量
                            f_projec = (f_vec @ self.v_vec) / (self.v_vec @ self.v_vec) * self.v_vec
                            if (
                                    # self.slip_threshold is not None
                                    # and self.slip_score[-1] > self.slip_threshold
                                    # and (f_vec @ self.v_vec) < 0
                                np.linalg.norm(r_vec) > 45
                            ):
                                self.force_control_flag = True  ##接触力超出阈值，进入接触力调整的阶段（调整为与滑动阈值的比例）
                                ## 重置以接收新目标（但先不接收初始位置）
                                self.get_logger().info('Terminate and change contact force!')
                                self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                                self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                                self.total_time = 0.0
                                self.n_now = 0
                                self.X, self.Y, self.Z, self.Q = None, None, None, None
                                self.force_inside_flag = False
                            else:
                                ##如果脱离接触，则增强相应方向的力
                                if self.N != 0:
                                    delta_x = 0.005 * (f_x - np.sign(f_x) * 0.2) if abs(
                                        f_x) < 0.2 else 0.0
                                    delta_y = 0.005 * (f_y - np.sign(f_y) * 0.2) if abs(
                                        f_y) < 0.2 else 0.0
                                if self.n_now < self.X.shape[0]:  ##轨迹还未执行完，继续执行轨迹
                                    self.cmd.position.x = self.X[self.n_now] + delta_x / self.freq
                                    self.cmd.position.y = self.Y[self.n_now] + delta_y / self.freq
                                    self.cmd.position.z = self.Z[self.n_now] + delta_z / self.freq
                                    # orientation 可插值或保持初始
                                    self.cmd.orientation.x = self.Q[self.n_now][0]
                                    self.cmd.orientation.y = self.Q[self.n_now][1]
                                    self.cmd.orientation.z = self.Q[self.n_now][2]
                                    self.cmd.orientation.w = self.Q[self.n_now][3]

                                else:
                                    self.cmd.position.x += delta_x / self.freq
                                    self.cmd.position.y += delta_y / self.freq
                                    self.cmd.position.z += delta_z / self.freq
                        else:
                            ## 重置以接收新目标
                            self.get_logger().info('Trajectory complete.')
                            self.initial = None
                            self.total_time = 0.0
                            self.n_now = 0
                            self.X, self.Y, self.Z, self.Q = None, None, None, None
                            self.force_inside_flag = False
                            if self.N == 0: ## 估计接触阈值
                                baseline = total_load[:baseline_frames]
                                center, scale = robust_center_scale(baseline)
                                self.contact_threshold = center + threshold_sigma * scale
                                self.get_logger().info(f"The contact_threshold is: {self.contact_threshold}")
                            self.N += 1

                    else:
                        if self.N < 4:
                            self.As_planner()
                        else:
                            self.get_logger().info('The Planning Process has ended!')
                self.pose_pub.publish(self.cmd)

                if self.i % 75 == 0:
                    # self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                    if self.contact_active:
                        self._start_contact_line_estimation_async()
                    self.get_logger().info(f'N={self.N}')
                    self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                    if self.X is not None:
                        self.get_logger().info(
                            f'delta_x= {delta_x}, delta_y= {delta_y}, delta_z= {delta_z}, '
                            f'v_vec= {self.v_vec}, r_vec= {r_vec}, f_vec= {f_vec}, n_now= {self.n_now}/{self.X.shape[0]}')
                    if self.force_control_flag:
                        self.get_logger().info(
                            f'delta_x= {delta_x}, delta_y= {delta_y}, delta_z= {delta_z}, '
                            f'v_vec= {self.v_vec}, r_vec= {r_vec}, f_vec= {f_vec}')
                    with self.contact_line_lock:
                        contact_line_rebuild_result = self.contact_line_rebuild_result
                    if contact_line_rebuild_result is not None:
                        self.get_logger().info(
                            f'rebuild_contact_point(mm)= '
                            f'{contact_line_rebuild_result.representative_point}, '
                            f'rebuild_contact_line= '
                            f'{contact_line_rebuild_result.representative_direction}'
                        )
                    elif self.contact_active:
                        self.get_logger().info('contact_line= estimating...')


def main(args=None):
    rclpy.init(args=args)
    node = MinimumJerkPosePlanner()

    # 设置拉杆：假定先设置 total_time，可通过 launch 参数或 Node 参数改动
    # node.total_time = 8.0  # 例如 5 秒完成轨迹
    node.target = Pose()
    node.target.position.x = 0.0
    node.target.position.y = 0.0
    node.target.position.z = 0.0

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        np.savez(r'Force_data_2508301147.npz', Fordis_left=node.Force_dis_left, Fordis_right=node.Force_dis_right,
                 Matrix_left=node.Matrix_left, Matrix_right=node.Matrix_right, Position_left=node.Position_left,
                 Position_right=node.Position_right, Displacement_left=node.Displacement_left,
                 Displacement_right=node.Displacement_right,
                 F_x=node.F_x, F_y=node.F_y, F_z=node.F_z, R_x=node.R_x, R_y=node.R_y, R_z=node.R_z,
                 Pose=node.Pose, Time=node.Time, Quat=node.Quat, slip_score=node.Slip_score)
        print('save Data!')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
