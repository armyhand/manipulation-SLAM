"""
统一接触因子图估计器（GTSAM ISAM2）。

在现有 robot_pivoting_estimate/contact_line_factor_graph.py 的线接触
实现基础上，扩展为同时支持点接触 / 线接触 / 面接触的增量因子图。

变量（世界/机器人坐标系，单位 mm、rad、N、N*mm）：

    o_i : 物体位姿（Pose3）
    接触几何（按接触类型不同）：
        点接触  c_i : 接触点（Point3）
        线接触  p_i : 接触线上一点（Point3），u_i : 接触线方向（单位向量）
        面接触  p_i : 接触平面上一点（Point3），n_i : 接触平面法向（单位向量）
    共享    w   : 夹爪坐标系力矩折算原点（Point3，仅当使用力矩因子时）

因子：

    Ftac   ：g_i^-1 o_i 等于触觉相对位移（逐帧）
    Foc    ：接触几何在物体坐标系中保持不变（相邻帧）
    Fcc    ：接触几何在世界坐标系中缓慢变化（相邻帧）
    Fsurf  ：接触点/线落在已知环境平面上（可选用）
    Ftorq  ：接触处力矩约束（点/线/面，按类型不同）
    Fwr    ：触觉位移回归力/力矩（可选用，实时管线直接读取合力矩）
    Ffric  ：摩擦力锥软约束（默认关闭，用于诊断）
    先验   ：物体位姿、接触几何初值、共享力矩原点

估计不确定性通过 gtsam.Marginals 提取接触几何的边际协方差，用于
rotation_selector 的主动偏转角度决策。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

from .contact_models import (
    ContactType,
    normalize,
    perturb_pose_right,
    pose_between,
    se3_log,
    direction_unit_residual,
    line_object_contact_residual,
    line_perpendicular_torque_residual,
    line_surface_residual,
    line_torque_residual,
    plane_object_contact_residual,
    plane_torque_residual,
    point_object_contact_residual,
    point_surface_residual,
    point_torque_residual,
    world_line_residual,
    world_plane_residual,
    world_point_residual,
    wrench_regression_residual,
)

WRENCH_ORIGIN_GRIPPER_MM = np.array([0.0, 0.0, 223.0])


@dataclass
class ContactEstimate:
    """因子图估计结果。"""

    object_poses: np.ndarray            # (T, 4, 4)
    contact_points: np.ndarray          # (T, 3)  点/线上点/面上点
    contact_directions: np.ndarray      # (T, 3)  线方向或面法向（点接触时为零向量）
    contact_normals: np.ndarray | None  # (T, 3)  面法向（非面接触时为 None）
    wrench_origin_gripper: np.ndarray   # (3,)
    contact_type: ContactType
    covariance: dict[str, np.ndarray] | None = None


class ContactFactorGraphISAM2:
    """支持点/线/面三类接触的增量 GTSAM ISAM2 接触估计器。"""

    def __init__(
        self,
        gripper_poses: np.ndarray,
        tactile_transforms: np.ndarray,
        initial_point: np.ndarray,
        initial_direction: np.ndarray | None = None,
        initial_normal: np.ndarray | None = None,
        forces: np.ndarray | None = None,
        moments: np.ndarray | None = None,
        contact_type: ContactType = ContactType.LINE,
        initial_wrench_origin_gripper: np.ndarray = WRENCH_ORIGIN_GRIPPER_MM,
        surface_plane_point: np.ndarray | None = None,
        surface_plane_normal: np.ndarray | None = None,
        friction_mu: float | None = None,
        use_wrench_regression: bool = False,
        stiffness: np.ndarray | None = None,
        compute_covariance: bool = False,
        line_perp_torque_enabled: bool = False,
    ) -> None:
        """初始化因子图、噪声模型与变量初值。

        参数
        ----
        gripper_poses          : (T, 4, 4) 夹爪在世界坐标系位姿；
        tactile_transforms     : (T, 4, 4) 由触觉 marker 解算的物体相对位移；
        initial_point          : 接触几何初值（点/线上点/面上点），世界系；
        initial_direction      : 线方向（线接触）；
        initial_normal         : 面法向（面接触）；
        forces / moments       : 夹爪系合力/合力矩（Ftorq 使用）；
        contact_type           : 接触类型；
        surface_plane_*        : 已知环境平面（Fsurf 使用）；
        friction_mu            : 摩擦系数（Ffric 诊断使用，默认关闭）；
        use_wrench_regression  : 是否使用 Fwr；
        stiffness              : Fwr 使用的触觉刚度对角阵；
        compute_covariance     : 是否在 current_estimate 中计算边际协方差。
        """
        import gtsam

        self.gtsam = gtsam
        self.contact_type = contact_type
        self.gripper_poses = np.asarray(gripper_poses, dtype=float)
        self.tactile_measurements = np.asarray(
            [se3_log(transform) for transform in tactile_transforms], dtype=float
        )
        self.initial_object_poses = np.asarray(
            [
                gripper @ tactile
                for gripper, tactile in zip(self.gripper_poses, tactile_transforms)
            ],
            dtype=float,
        )
        self.initial_point = np.asarray(initial_point, dtype=float).reshape(3)
        self.initial_direction = (
            normalize(initial_direction) if initial_direction is not None else None
        )
        self.initial_normal = (
            normalize(initial_normal) if initial_normal is not None else None
        )
        self.initial_wrench_origin_gripper = np.asarray(
            initial_wrench_origin_gripper, dtype=float
        ).reshape(3)
        self.surface_plane_point = (
            None
            if surface_plane_point is None
            else np.asarray(surface_plane_point, dtype=float).reshape(3)
        )
        self.surface_plane_normal = (
            None
            if surface_plane_normal is None
            else normalize(surface_plane_normal)
        )
        self.friction_mu = friction_mu
        self.use_wrench_regression = use_wrench_regression
        self.stiffness = (
            np.asarray(stiffness, dtype=float)
            if stiffness is not None
            else np.array([1100.0, 1150.0, 1000.0, 0.18, 0.16, 0.22])
        )
        self.compute_covariance = compute_covariance
        self.line_perp_torque_enabled = line_perp_torque_enabled

        self.forces = None if forces is None else np.asarray(forces, dtype=float)
        self.moments = None if moments is None else np.asarray(moments, dtype=float)
        if (self.forces is None) != (self.moments is None):
            raise ValueError("forces and moments must be provided together")
        if self.forces is not None and (
            self.forces.shape != (len(self.gripper_poses), 3)
            or self.moments.shape != (len(self.gripper_poses), 3)
        ):
            raise ValueError("forces and moments must have shape (frame_count, 3)")

        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)
        self._graph = gtsam.NonlinearFactorGraph()

        # 噪声模型（数值与现有 contact_line_factor_graph 保持一致）
        self.noise_tactile = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.004, 0.004, 0.004, 0.35, 0.35, 0.35])
        )
        self.noise_foc = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.8, 0.8, 0.8, 0.015, 0.015, 0.015])
        )
        self.noise_fcc = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.35, 0.35, 0.35, 0.006, 0.006, 0.006])
        )
        # 点接触的 Foc/Fcc 残差为 3 维（线/面为 6 维），需要匹配维度的噪声。
        self.noise_foc_point = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.8, 0.8, 0.8])
        )
        self.noise_fcc_point = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.35, 0.35, 0.35])
        )
        self.noise_unit = gtsam.noiseModel.Isotropic.Sigma(1, 0.003)
        self.noise_torque = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            gtsam.noiseModel.Isotropic.Sigma(1, 10.0),
        )
        # 点接触力矩残差为 3 维，需要匹配维度的鲁棒噪声模型。
        self.noise_torque_3d = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            gtsam.noiseModel.Isotropic.Sigma(3, 10.0),
        )
        self.noise_pose_prior = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.01, 0.01, 0.6, 0.6, 0.6])
        )
        self.noise_line_point_prior = gtsam.noiseModel.Isotropic.Sigma(3, 20.0)
        self.noise_line_direction_prior = gtsam.noiseModel.Isotropic.Sigma(3, 0.15)
        self.noise_wrench_origin_prior = gtsam.noiseModel.Isotropic.Sigma(3, 30.0)
        # 面接触的平面点/法向先验更弱，依赖 Fcc 平滑。
        self.noise_plane_point_prior = gtsam.noiseModel.Isotropic.Sigma(3, 25.0)
        self.noise_plane_normal_prior = gtsam.noiseModel.Isotropic.Sigma(3, 0.25)
        self.noise_surface = gtsam.noiseModel.Isotropic.Sigma(1, 0.5)
        self.noise_wrench_regression = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.030, 0.030, 0.030, 0.012, 0.012, 0.012])
        )
        self.noise_friction = gtsam.noiseModel.Isotropic.Sigma(1, 0.05)

    # ------------------------------------------------------------------
    # key 与 GTSAM 类型转换
    # ------------------------------------------------------------------

    def key(self, prefix: str, index: int) -> int:
        try:
            return self.gtsam.symbol(prefix, index)
        except TypeError:
            return self.gtsam.symbol(ord(prefix), index)

    def object_key(self, index: int) -> int:
        return self.key("o", index)

    def point_key(self, index: int) -> int:
        return self.key("p", index)

    def direction_key(self, index: int) -> int:
        return self.key("u", index)

    def normal_key(self, index: int) -> int:
        return self.key("n", index)

    def wrench_origin_key(self) -> int:
        return self.key("w", 0)

    def pose3(self, pose: np.ndarray):
        return self.gtsam.Pose3(
            self.gtsam.Rot3(pose[:3, :3]), self.gtsam.Point3(*pose[:3, 3])
        )

    def pose3_matrix(self, pose3) -> np.ndarray:
        return np.asarray(pose3.matrix(), dtype=float)

    def point_array(self, point) -> np.ndarray:
        return np.asarray(point, dtype=float).reshape(3)

    def as_point3(self, point: np.ndarray):
        return self.gtsam.Point3(float(point[0]), float(point[1]), float(point[2]))

    def add_custom_factor(
        self,
        graph,
        keys: Sequence[int],
        noise_model,
        value_getters: Sequence[Callable],
        local_error: Callable[[List[np.ndarray]], np.ndarray],
        dimensions: Sequence[int],
    ) -> None:
        """添加带数值雅可比的 CustomFactor，复用 numpy 形式的残差函数。"""
        gtsam = self.gtsam

        def error_func(this, values, jacobians):
            parts = [getter(values) for getter in value_getters]
            base_error = np.asarray(local_error(parts), dtype=float)
            if jacobians is not None:
                for part_index, dim in enumerate(dimensions):
                    jacobian = np.zeros((len(base_error), dim))
                    epsilon = 1e-6
                    for axis in range(dim):
                        delta = np.zeros(dim)
                        delta[axis] = epsilon
                        perturbed_parts = [np.array(part, copy=True) for part in parts]
                        if dim == 6:
                            perturbed_parts[part_index] = perturb_pose_right(
                                perturbed_parts[part_index], delta
                            )
                        else:
                            perturbed_parts[part_index] = (
                                perturbed_parts[part_index] + delta
                            )
                        jacobian[:, axis] = (
                            np.asarray(local_error(perturbed_parts), dtype=float)
                            - base_error
                        ) / epsilon
                    jacobians[part_index] = np.asfortranarray(jacobian)
            return base_error

        graph.add(gtsam.CustomFactor(noise_model, list(keys), error_func))

    # ------------------------------------------------------------------
    # 增量加入一帧
    # ------------------------------------------------------------------

    def add_step(self, index: int) -> None:
        """向 ISAM2 中加入第 index 帧的变量、先验和相邻帧约束并立即更新。"""
        gtsam = self.gtsam
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        object_key = self.object_key(index)
        point_key = self.point_key(index)

        object_initial = self.initial_object_poses[index]
        point_initial = self.initial_point
        direction_initial = self.initial_direction
        normal_initial = self.initial_normal
        if index > 0:
            estimate = self.isam.calculateEstimate()
            point_initial = self.point_array(
                estimate.atPoint3(self.point_key(index - 1))
            )
            if self.contact_type == ContactType.LINE:
                direction_initial = normalize(
                    self.point_array(estimate.atPoint3(self.direction_key(index - 1)))
                )
            if self.contact_type == ContactType.SURFACE:
                normal_initial = normalize(
                    self.point_array(estimate.atPoint3(self.normal_key(index - 1)))
                )

        values.insert(object_key, self.pose3(object_initial))
        values.insert(point_key, self.as_point3(point_initial))
        if self.contact_type == ContactType.LINE:
            values.insert(
                self.direction_key(index), self.as_point3(direction_initial)
            )
        if self.contact_type == ContactType.SURFACE:
            # 面接触只使用 点 + 法向 两个几何量（方向冗余，不使用）。
            values.insert(self.normal_key(index), self.as_point3(normal_initial))
        if index == 0 and self.forces is not None:
            values.insert(
                self.wrench_origin_key(),
                self.as_point3(self.initial_wrench_origin_gripper),
            )

        gripper_pose = self.gripper_poses[index]
        tactile_measurement = self.tactile_measurements[index]

        # ----- Ftac：触觉位移测量 -----
        self.add_custom_factor(
            graph,
            [object_key],
            self.noise_tactile,
            [lambda vals, k=object_key: self.pose3_matrix(vals.atPose3(k))],
            lambda parts, gp=gripper_pose, tm=tactile_measurement: (
                se3_log(pose_between(gp, parts[0])) - tm
            ),
            [6],
        )

        # ----- Fwr：触觉位移回归力/力矩（可选） -----
        if self.use_wrench_regression and self.forces is not None:
            self.add_custom_factor(
                graph,
                [object_key],
                self.noise_wrench_regression,
                [lambda vals, k=object_key: self.pose3_matrix(vals.atPose3(k))],
                lambda parts, gp=gripper_pose, f=self.forces[index], m=self.moments[index], k=self.stiffness: (
                    wrench_regression_residual(parts[0], m, f, gp, k)
                ),
                [6],
            )

        # ----- 单位长度约束 -----
        if self.contact_type == ContactType.LINE:
            self.add_custom_factor(
                graph,
                [self.direction_key(index)],
                self.noise_unit,
                [
                    lambda vals, k=self.direction_key(index): self.point_array(
                        vals.atPoint3(k)
                    )
                ],
                lambda parts: direction_unit_residual(parts[0]),
                [3],
            )
        if self.contact_type == ContactType.SURFACE:
            self.add_custom_factor(
                graph,
                [self.normal_key(index)],
                self.noise_unit,
                [
                    lambda vals, k=self.normal_key(index): self.point_array(
                        vals.atPoint3(k)
                    )
                ],
                lambda parts: direction_unit_residual(parts[0]),
                [3],
            )

        # ----- Ftorq：接触处力矩约束 -----
        if self.forces is not None:
            if self.contact_type == ContactType.POINT:
                self.add_custom_factor(
                    graph,
                    [point_key],
                    self.noise_torque_3d,
                    [
                        lambda vals, k=point_key: self.point_array(
                            vals.atPoint3(k)
                        )
                    ],
                    lambda parts, gp=gripper_pose, f=self.forces[index], m=self.moments[index]: (
                        point_torque_residual(
                            parts[0], m, f, gp, self.initial_wrench_origin_gripper
                        )
                    ),
                    [3],
                )
            elif self.contact_type == ContactType.LINE:
                self.add_custom_factor(
                    graph,
                    [point_key, self.direction_key(index)],
                    self.noise_torque,
                    [
                        lambda vals, k=point_key: self.point_array(
                            vals.atPoint3(k)
                        ),
                        lambda vals, k=self.direction_key(index): self.point_array(
                            vals.atPoint3(k)
                        ),
                    ],
                    lambda parts, gp=gripper_pose, f=self.forces[index], m=self.moments[index]: (
                        line_torque_residual(
                            parts[0],
                            parts[1],
                            m,
                            f,
                            gp,
                            self.initial_wrench_origin_gripper,
                        )
                    ),
                    [3, 3],
                )
                if self.line_perp_torque_enabled:
                    self.add_custom_factor(
                        graph,
                        [point_key, self.direction_key(index)],
                        self.noise_torque,
                        [
                            lambda vals, k=point_key: self.point_array(
                                vals.atPoint3(k)
                            ),
                            lambda vals, k=self.direction_key(index): self.point_array(
                                vals.atPoint3(k)
                            ),
                        ],
                        lambda parts, gp=gripper_pose, f=self.forces[index], m=self.moments[index]: (
                            line_perpendicular_torque_residual(
                                parts[0],
                                parts[1],
                                m,
                                f,
                                gp,
                                self.initial_wrench_origin_gripper,
                            )
                        ),
                        [3, 3],
                    )
            elif self.contact_type == ContactType.SURFACE:
                self.add_custom_factor(
                    graph,
                    [point_key, self.normal_key(index)],
                    self.noise_torque,
                    [
                        lambda vals, k=point_key: self.point_array(
                            vals.atPoint3(k)
                        ),
                        lambda vals, k=self.normal_key(index): self.point_array(
                            vals.atPoint3(k)
                        ),
                    ],
                    lambda parts, gp=gripper_pose, f=self.forces[index], m=self.moments[index]: (
                        plane_torque_residual(
                            parts[0],
                            parts[1],
                            m,
                            f,
                            gp,
                            self.initial_wrench_origin_gripper,
                        )
                    ),
                    [3, 3],
                )

        # ----- Fsurf：接触点/线落在已知环境平面上 -----
        if self.surface_plane_point is not None and self.surface_plane_normal is not None:
            if self.contact_type == ContactType.POINT:
                self.add_custom_factor(
                    graph,
                    [point_key],
                    self.noise_surface,
                    [
                        lambda vals, k=point_key: self.point_array(
                            vals.atPoint3(k)
                        )
                    ],
                    lambda parts: point_surface_residual(
                        parts[0], self.surface_plane_point, self.surface_plane_normal
                    ),
                    [3],
                )
            elif self.contact_type == ContactType.LINE:
                self.add_custom_factor(
                    graph,
                    [point_key, self.direction_key(index)],
                    self.noise_surface,
                    [
                        lambda vals, k=point_key: self.point_array(
                            vals.atPoint3(k)
                        ),
                        lambda vals, k=self.direction_key(index): self.point_array(
                            vals.atPoint3(k)
                        ),
                    ],
                    lambda parts: line_surface_residual(
                        parts[0],
                        parts[1],
                        self.surface_plane_point,
                        self.surface_plane_normal,
                    ),
                    [3, 3],
                )

        # ----- 先验 -----
        graph.add(
            gtsam.PriorFactorPose3(
                object_key, self.pose3(object_initial), self.noise_pose_prior
            )
        )
        if self.contact_type == ContactType.POINT:
            graph.add(
                gtsam.PriorFactorPoint3(
                    point_key, self.as_point3(point_initial), self.noise_line_point_prior
                )
            )
        else:
            graph.add(
                gtsam.PriorFactorPoint3(
                    point_key, self.as_point3(point_initial), self.noise_line_point_prior
                )
            )
        if self.contact_type == ContactType.LINE:
            graph.add(
                gtsam.PriorFactorPoint3(
                    self.direction_key(index),
                    self.as_point3(direction_initial),
                    self.noise_line_direction_prior,
                )
            )
        if self.contact_type == ContactType.SURFACE:
            graph.add(
                gtsam.PriorFactorPoint3(
                    self.normal_key(index),
                    self.as_point3(normal_initial),
                    self.noise_plane_normal_prior,
                )
            )

        if index == 0 and self.forces is not None:
            graph.add(
                gtsam.PriorFactorPoint3(
                    self.wrench_origin_key(),
                    self.as_point3(self.initial_wrench_origin_gripper),
                    self.noise_wrench_origin_prior,
                )
            )

        # ----- 相邻帧约束：Foc + Fcc -----
        if index > 0:
            previous_object_key = self.object_key(index - 1)
            previous_point_key = self.point_key(index - 1)

            if self.contact_type == ContactType.POINT:
                self.add_custom_factor(
                    graph,
                    [previous_object_key, previous_point_key, object_key, point_key],
                    self.noise_foc_point,
                    [
                        lambda vals, k=previous_object_key: self.pose3_matrix(vals.atPose3(k)),
                        lambda vals, k=previous_point_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=object_key: self.pose3_matrix(vals.atPose3(k)),
                        lambda vals, k=point_key: self.point_array(vals.atPoint3(k)),
                    ],
                    lambda parts: point_object_contact_residual(
                        parts[0], parts[1], parts[2], parts[3]
                    ),
                    [6, 3, 6, 3],
                )
                self.add_custom_factor(
                    graph,
                    [previous_point_key, point_key],
                    self.noise_fcc_point,
                    [
                        lambda vals, k=previous_point_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=point_key: self.point_array(vals.atPoint3(k)),
                    ],
                    lambda parts: world_point_residual(parts[0], parts[1]),
                    [3, 3],
                )

            elif self.contact_type == ContactType.LINE:
                previous_direction_key = self.direction_key(index - 1)
                self.add_custom_factor(
                    graph,
                    [
                        previous_object_key,
                        previous_point_key,
                        previous_direction_key,
                        object_key,
                        point_key,
                        self.direction_key(index),
                    ],
                    self.noise_foc,
                    [
                        lambda vals, k=previous_object_key: self.pose3_matrix(vals.atPose3(k)),
                        lambda vals, k=previous_point_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=previous_direction_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=object_key: self.pose3_matrix(vals.atPose3(k)),
                        lambda vals, k=point_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=self.direction_key(index): self.point_array(vals.atPoint3(k)),
                    ],
                    lambda parts: line_object_contact_residual(
                        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                    ),
                    [6, 3, 3, 6, 3, 3],
                )
                self.add_custom_factor(
                    graph,
                    [
                        previous_point_key,
                        previous_direction_key,
                        point_key,
                        self.direction_key(index),
                    ],
                    self.noise_fcc,
                    [
                        lambda vals, k=previous_point_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=previous_direction_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=point_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=self.direction_key(index): self.point_array(vals.atPoint3(k)),
                    ],
                    lambda parts: world_line_residual(parts[0], parts[1], parts[2], parts[3]),
                    [3, 3, 3, 3],
                )

            elif self.contact_type == ContactType.SURFACE:
                previous_normal_key = self.normal_key(index - 1)
                self.add_custom_factor(
                    graph,
                    [
                        previous_object_key,
                        previous_point_key,
                        previous_normal_key,
                        object_key,
                        point_key,
                        self.normal_key(index),
                    ],
                    self.noise_foc,
                    [
                        lambda vals, k=previous_object_key: self.pose3_matrix(vals.atPose3(k)),
                        lambda vals, k=previous_point_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=previous_normal_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=object_key: self.pose3_matrix(vals.atPose3(k)),
                        lambda vals, k=point_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=self.normal_key(index): self.point_array(vals.atPoint3(k)),
                    ],
                    lambda parts: plane_object_contact_residual(
                        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                    ),
                    [6, 3, 3, 6, 3, 3],
                )
                self.add_custom_factor(
                    graph,
                    [
                        previous_point_key,
                        previous_normal_key,
                        point_key,
                        self.normal_key(index),
                    ],
                    self.noise_fcc,
                    [
                        lambda vals, k=previous_point_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=previous_normal_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=point_key: self.point_array(vals.atPoint3(k)),
                        lambda vals, k=self.normal_key(index): self.point_array(vals.atPoint3(k)),
                    ],
                    lambda parts: world_plane_residual(parts[0], parts[1], parts[2], parts[3]),
                    [3, 3, 3, 3],
                )

        self.isam.update(graph, values)
        self.isam.update()
        # 累积因子图用于事后边际协方差计算。
        self._graph.push_back(graph)

    # ------------------------------------------------------------------
    # 估计读取
    # ------------------------------------------------------------------

    def current_estimate(self, count: int | None = None) -> ContactEstimate:
        """读取前 count 帧的估计结果，可选计算边际协方差。"""
        if count is None:
            count = len(self.gripper_poses)
        values = self.isam.calculateEstimate()

        object_poses = []
        contact_points = []
        contact_directions = []
        contact_normals = []
        for index in range(count):
            object_poses.append(self.pose3_matrix(values.atPose3(self.object_key(index))))
            contact_points.append(self.point_array(values.atPoint3(self.point_key(index))))
            if self.contact_type == ContactType.LINE:
                contact_directions.append(
                    normalize(self.point_array(values.atPoint3(self.direction_key(index))))
                )
                contact_normals.append(np.zeros(3))
            elif self.contact_type == ContactType.SURFACE:
                # 面接触的“方向向量”即面法向（点接触时为零向量）。
                normal = normalize(
                    self.point_array(values.atPoint3(self.normal_key(index)))
                )
                contact_directions.append(normal)
                contact_normals.append(normal)
            else:
                contact_directions.append(np.zeros(3))
                contact_normals.append(np.zeros(3))

        covariance = None
        if self.compute_covariance and count > 0:
            covariance = self.marginal_covariance(values, count)

        return ContactEstimate(
            object_poses=np.asarray(object_poses),
            contact_points=np.asarray(contact_points),
            contact_directions=np.asarray(contact_directions),
            contact_normals=np.asarray(contact_normals)
            if self.contact_type == ContactType.SURFACE
            else None,
            wrench_origin_gripper=(
                self.point_array(values.atPoint3(self.wrench_origin_key()))
                if self.forces is not None
                else self.initial_wrench_origin_gripper.copy()
            ),
            contact_type=self.contact_type,
            covariance=covariance,
        )

    def marginal_covariance(self, values, count: int | None = None) -> dict[str, np.ndarray]:
        """用累积因子图与当前估计计算接触几何的边际协方差。

        返回每个关键量最近一帧的 3x3 协方差矩阵；对物体位姿取平移部分。
        注意：累积图包含全部历史，协方差计算代价随帧数增长，
        实时使用时建议仅在触发偏转决策时调用。
        """
        if count is None:
            count = len(self.gripper_poses)
        gtsam = self.gtsam
        # 用累积图计算边际协方差时，values 必须包含全部变量；
        # 若传入的是局部 estimate，则回退到 isam 的全局估计。
        if not hasattr(values, "exists") or not values.exists(self.point_key(0)):
            values = self.isam.calculateEstimate()
        try:
            marginals = gtsam.Marginals(self._graph, values)
        except Exception:
            return {}

        last = max(0, count - 1)
        covariance: dict[str, np.ndarray] = {}
        try:
            covariance["point"] = np.asarray(
                marginals.marginalCovariance(self.point_key(last)), dtype=float
            )
        except Exception:
            pass
        if self.contact_type == ContactType.LINE:
            try:
                covariance["direction"] = np.asarray(
                    marginals.marginalCovariance(self.direction_key(last)), dtype=float
                )
            except Exception:
                pass
        if self.contact_type == ContactType.SURFACE:
            try:
                covariance["normal"] = np.asarray(
                    marginals.marginalCovariance(self.normal_key(last)), dtype=float
                )
            except Exception:
                pass
        try:
            covariance["object_translation"] = np.asarray(
                marginals.marginalCovariance(self.object_key(last)), dtype=float
            )[:3, :3]
        except Exception:
            pass
        return covariance

    def run(
        self,
        frames: np.ndarray | None = None,
        contact_start_frame: int = 0,
        report_interval_frames: int = 10,
        print_progress: bool = True,
    ) -> ContactEstimate:
        """逐帧增量更新因子图，返回最终估计。"""
        if frames is None:
            frames = np.arange(len(self.gripper_poses), dtype=int)

        for index in range(len(self.gripper_poses)):
            self.add_step(index)
            estimate = self.current_estimate(index + 1)
            current_object_pose = estimate.object_poses[-1]
            current_point = estimate.contact_points[-1]
            current_direction = estimate.contact_directions[-1]

            should_report = (
                report_interval_frames > 0
                and (int(frames[index]) - contact_start_frame) % report_interval_frames == 0
            )
            if print_progress and (should_report or index == len(self.gripper_poses) - 1):
                translation = current_object_pose[:3, 3]
                print(
                    f"[{self.contact_type.value}] "
                    f"frame={int(frames[index])} "
                    f"object_t=[{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}] "
                    f"contact_p=[{current_point[0]:.3f}, {current_point[1]:.3f}, {current_point[2]:.3f}] "
                    f"dir=[{current_direction[0]:.5f}, {current_direction[1]:.5f}, {current_direction[2]:.5f}] "
                    f"wrench_origin_g=[{estimate.wrench_origin_gripper[0]:.3f}, "
                    f"{estimate.wrench_origin_gripper[1]:.3f}, {estimate.wrench_origin_gripper[2]:.3f}]"
                )

        return self.current_estimate()
