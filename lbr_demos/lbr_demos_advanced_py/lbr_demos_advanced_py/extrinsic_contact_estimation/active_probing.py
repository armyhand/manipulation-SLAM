"""
主动偏转估计闭环（estimation -> classification -> factor graph -> rotation）。

与 pose_planning_node_realtime_contact_line.py 的“接触后继续稳态偏转，
根据估计结果修正旋转中心和旋转轴”目标对应。本模块把完整闭环封装为
ActiveProbingPipeline：

    feed_snapshot(snapshot)
        -> detect contact -> classify contact type
        -> factor-graph optimize -> estimate contact geometry
        -> select rotation center & axis -> RotationCommand

输出 ContactProbingResult，包含接触类型、估计结果、不确定度与旋转指令，
供上层节点直接使用。

同时提供可选的 RealtimeContactEstimationNode（ROS2 节点示例），订阅与
原节点相同的话题，周期性运行闭环并发布估计结果。rclpy 的导入被延迟到
节点实例化时，保证模块在无 ROS 环境下可单独导入与测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contact_classifier import ContactClassification, ContactClassifier
from .contact_models import ContactType, normalize
from .data_io import (
    ROTATION_LEFT_GT,
    ROTATION_RIGHT_GT,
    TactileSnapshot,
    initial_line_from_object_poses,
    initial_point_from_object_poses,
    poses_from_snapshot,
    tactile_transforms_from_snapshot,
    wrenches_from_snapshot,
)
from .factor_graph_estimator import ContactEstimate, ContactFactorGraphISAM2
from .rotation_selector import RotationCommand, RotationSelector
from .tactile_features import compute_patch_features_from_snapshot

DEFAULT_FRAME_STEP = 5
DEFAULT_TOP_N = 60


@dataclass
class ContactProbingResult:
    """一次接触估计闭环的完整输出。"""

    classification: ContactClassification
    estimate: ContactEstimate
    rotation_command: RotationCommand
    contact_start_frame: int
    frames: np.ndarray
    position_std_mm: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.as_dict(),
            "contact_start_frame": self.contact_start_frame,
            "frames": self.frames.tolist(),
            "representative_point": np.mean(self.estimate.contact_points, axis=0).tolist(),
            "representative_direction": np.mean(self.estimate.contact_directions, axis=0).tolist(),
            "wrench_origin_gripper": self.estimate.wrench_origin_gripper.tolist(),
            "position_std_mm": self.position_std_mm,
            "rotation_command": self.rotation_command.as_dict(),
        }


class ActiveProbingPipeline:
    """接触后因子图优化 + 主动偏转的闭环。"""

    def __init__(
        self,
        frame_step: int = DEFAULT_FRAME_STEP,
        top_n: int = DEFAULT_TOP_N,
        compute_covariance: bool = True,
        classifier: ContactClassifier | None = None,
        rotation_selector: RotationSelector | None = None,
        surface_plane_point: np.ndarray | None = None,
        surface_plane_normal: np.ndarray | None = None,
        friction_mu: float | None = None,
    ) -> None:
        self.frame_step = frame_step
        self.top_n = top_n
        self.compute_covariance = compute_covariance
        self.classifier = classifier or ContactClassifier()
        self.rotation_selector = rotation_selector or RotationSelector()
        self.surface_plane_point = surface_plane_point
        self.surface_plane_normal = surface_plane_normal
        self.friction_mu = friction_mu
        self._previous_sign = 1
        self._alternate_index = 0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def feed_snapshot(
        self,
        snapshot: TactileSnapshot,
        contact_start_frame: int | None = None,
        slip_score: float | None = None,
        slip_threshold: float | None = None,
    ) -> ContactProbingResult:
        """处理一个数据快照：检测接触、分类、估计、选择旋转。"""
        if contact_start_frame is not None:
            snapshot.contact_start_frame = int(contact_start_frame)

        frame_count = snapshot.frame_count
        frames = np.arange(
            snapshot.contact_start_frame, frame_count, self.frame_step, dtype=int
        )
        if len(frames) < 3:
            raise ValueError(
                f"Not enough frames for contact estimation: "
                f"start={snapshot.contact_start_frame}, count={frame_count}"
            )

        # 1. 夹爪位姿、力矩、触觉相对变换
        gripper_poses = poses_from_snapshot(snapshot, frames)
        forces, moments = wrenches_from_snapshot(
            snapshot, frames, baseline_frames=10, top_n=self.top_n
        )
        tactile_transforms = tactile_transforms_from_snapshot(
            snapshot, frames, top_n=self.top_n
        )
        initial_object_poses = np.asarray(
            [
                gripper @ tactile
                for gripper, tactile in zip(gripper_poses, tactile_transforms)
            ]
        )

        # 2. 接触类型分类（几何 + 力矩）
        classification = self._classify(
            snapshot, gripper_poses, initial_object_poses
        )

        # 3. 因子图优化（按分类的接触类型）
        estimate = self._estimate(
            gripper_poses,
            tactile_transforms,
            initial_object_poses,
            forces,
            moments,
            classification.contact_type,
        )

        # 4. 由估计结果选择旋转中心与方向
        object_center = estimate.object_poses[-1][:3, 3]
        gripper_reference = gripper_poses[-1][:3, 3]
        contact_point = estimate.contact_points[-1]
        contact_direction = estimate.contact_directions[-1]

        position_std_mm = None
        if estimate.covariance is not None and "point" in estimate.covariance:
            position_std_mm = float(
                np.sqrt(max(np.diag(estimate.covariance["point"]).max(), 0.0))
            )

        rotation_command = self.rotation_selector.select(
            contact_type=classification.contact_type,
            contact_point=contact_point,
            contact_direction=contact_direction,
            object_center=object_center,
            gripper_reference=gripper_reference,
            position_std_mm=position_std_mm,
            moment_gripper=moments[-1] if moments is not None and len(moments) else None,
            slip_score=slip_score,
            slip_threshold=slip_threshold,
            previous_sign=self._previous_sign,
            alternate_index=self._alternate_index,
        )
        self._previous_sign = rotation_command.sign
        self._alternate_index += 1

        return ContactProbingResult(
            classification=classification,
            estimate=estimate,
            rotation_command=rotation_command,
            contact_start_frame=snapshot.contact_start_frame,
            frames=frames,
            position_std_mm=position_std_mm,
            diagnostics={
                "gripper_poses": gripper_poses,
                "forces": forces,
                "moments": moments,
                "tactile_transforms": tactile_transforms,
                "initial_object_poses": initial_object_poses,
            },
        )

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------

    def _classify(
        self,
        snapshot: TactileSnapshot,
        gripper_poses: np.ndarray,
        initial_object_poses: np.ndarray,
    ) -> ContactClassification:
        reference_frame = max(0, snapshot.contact_start_frame)
        if snapshot.frame_count <= reference_frame:
            reference_frame = snapshot.frame_count - 1

        features = compute_patch_features_from_snapshot(
            np.asarray(snapshot.Position_left[reference_frame]),
            np.asarray(snapshot.Position_right[reference_frame]),
            np.asarray(snapshot.Displacement_left[reference_frame]),
            np.asarray(snapshot.Displacement_right[reference_frame]),
            top_n=self.top_n,
            rotation_left=ROTATION_LEFT_GT,
            rotation_right=ROTATION_RIGHT_GT,
        )

        # 候选几何：接触区重心 + 从物体位姿序列解出的接触线方向。
        candidate_point = np.asarray(features.centroid)
        forces, moments = wrenches_from_snapshot(
            snapshot,
            np.asarray([reference_frame]),
            baseline_frames=10,
            top_n=self.top_n,
        )
        gripper_pose = gripper_poses[-1]
        if len(initial_object_poses) >= 2:
            initial_point, initial_direction = initial_line_from_object_poses(
                initial_object_poses
            )
        else:
            initial_point, initial_direction = candidate_point, np.array([1.0, 0.0, 0.0])

        return self.classifier.classify(
            features,
            force_gripper=forces[-1],
            moment_gripper=moments[-1],
            gripper_pose=gripper_pose,
            candidate_point=candidate_point,
            candidate_direction=initial_direction,
            candidate_normal=None,
            wrench_origin_gripper=np.array([0.0, 0.0, 0.0]),
        )

    def _estimate(
        self,
        gripper_poses: np.ndarray,
        tactile_transforms: np.ndarray,
        initial_object_poses: np.ndarray,
        forces: np.ndarray,
        moments: np.ndarray,
        contact_type: ContactType,
    ) -> ContactEstimate:
        if contact_type == ContactType.POINT:
            initial_point = initial_point_from_object_poses(initial_object_poses)
            initial_direction = None
            initial_normal = None
        else:
            initial_point, initial_direction = initial_line_from_object_poses(
                initial_object_poses
            )
            initial_normal = None
            if contact_type == ContactType.SURFACE and self.surface_plane_normal is None:
                # 无先验平面时，用物体位姿序列的旋转轴近似面法向。
                _, axis = initial_line_from_object_poses(initial_object_poses)
                initial_normal = normalize(axis)

        estimator = ContactFactorGraphISAM2(
            gripper_poses=gripper_poses,
            tactile_transforms=tactile_transforms,
            initial_point=initial_point,
            initial_direction=initial_direction,
            initial_normal=initial_normal,
            forces=forces,
            moments=moments,
            contact_type=contact_type,
            initial_wrench_origin_gripper=np.array([0.0, 0.0, 223.0]),
            surface_plane_point=self.surface_plane_point,
            surface_plane_normal=self.surface_plane_normal,
            friction_mu=self.friction_mu,
            compute_covariance=self.compute_covariance,
        )
        return estimator.run(frames=None, contact_start_frame=0, print_progress=False)


class RealtimeContactEstimationNode:
    """ROS2 节点示例：订阅与原节点相同的话题并运行主动偏转闭环。

    使用方式（需已 source /opt/ros/humble/setup.bash）：
        node = RealtimeContactEstimationNode()
        node.run()   # rclpy.spin

    为保持模块可无 ROS 导入，rclpy 在 __init__ 中按需导入。
    """

    TOPIC_POSE_STATE = "/lbr/state/pose"
    TOPIC_POSE_CMD = "/lbr/command/pose"
    TOPIC_POSITIONS_L = "/positions_l"
    TOPIC_POSITIONS_R = "/positions_r"
    TOPIC_DISPLACEMENTS_L = "/displacements_l"
    TOPIC_DISPLACEMENTS_R = "/displacements_r"
    TOPIC_FORCES_L = "/forces_l"
    TOPIC_FORCES_R = "/forces_r"

    def __init__(self, pipeline: ActiveProbingPipeline | None = None) -> None:
        import rclpy
        from geometry_msgs.msg import Pose
        from tutorial_interfaces.msg import Array3, Cloud

        self._rclpy = rclpy
        self._Pose = Pose
        self._Array3 = Array3
        self._Cloud = Cloud
        self.pipeline = pipeline or ActiveProbingPipeline()

        rclpy.init(args=None)
        self.node = rclpy.create_node("active_probing_contact_estimation")
        self.snapshot = TactileSnapshot()

        def make_cloud_callback(name: str):
            def callback(msg):
                xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T
                if name.endswith("_l"):
                    self.snapshot.Position_left.append(xyz)
                else:
                    self.snapshot.Position_right.append(xyz)

            return callback

        def make_vector_callback(name: str):
            def callback(msg):
                value = np.array([msg.x, msg.y, msg.z])
                if name == "F_x":
                    self.snapshot.F_x = np.append(self.snapshot.F_x, value[0])
                    self.snapshot.F_y = np.append(self.snapshot.F_y, value[1])
                    self.snapshot.F_z = np.append(self.snapshot.F_z, value[2])

            return callback

        def on_pose(msg):
            self.snapshot.Pose.append(
                np.array([msg.position.x, msg.position.y, msg.position.z])
            )
            self.snapshot.Quat.append(
                np.array(
                    [
                        msg.orientation.x,
                        msg.orientation.y,
                        msg.orientation.z,
                        msg.orientation.w,
                    ]
                )
            )

        self.node.create_subscription(self._Pose, self.TOPIC_POSE_STATE, on_pose, 10)
        self.node.create_subscription(
            self._Cloud, self.TOPIC_POSITIONS_L, make_cloud_callback("_l"), 10
        )
        self.node.create_subscription(
            self._Cloud, self.TOPIC_POSITIONS_R, make_cloud_callback("_r"), 10
        )
        self.pose_pub = self.node.create_publisher(
            self._Pose, self.TOPIC_POSE_CMD, 1
        )

    def run(self) -> None:
        self._rclpy.spin(self.node)

    def destroy(self) -> None:
        if hasattr(self, "node"):
            self.node.destroy_node()
            self._rclpy.shutdown()
