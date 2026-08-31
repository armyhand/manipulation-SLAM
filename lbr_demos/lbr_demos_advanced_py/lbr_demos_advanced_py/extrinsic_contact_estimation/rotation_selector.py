"""
主动偏转：根据当前因子图估计结果选择旋转中心点与旋转方向。

设计目标（对应源代码注释第 3/6/7 条）：
    1. 接触后继续稳态偏转；
    2. 根据估计结果修正旋转中心和旋转轴，继续偏转；
    3. 最终实现稳定接触与鲁棒估计。

旋转轴选择策略（保持接触不变的前提下激励触觉信号）：

    线接触（本任务主工况）：
        旋转轴 = 估计接触线（方向 u，线上点 p）。
        绕接触线旋转时接触线保持不变，物体绕线摆动，与现有
        pose_planning_node_realtime_contact_line.py 的定轴旋转一致；
        为了抑制沿线的扭矩过大，旋转方向交替取反（dither）。

    点接触：
        旋转轴 = 过接触点 c 的直线，方向取“垂直于点 c 到物体中心
        杠杆臂”的方向 u1 = normalize(lever × a)，并在平行于接触面的
        两个正交方向 u1、u2 = u1 × lever_hat 之间交替，最大化 marker
        位移，同时保持接触点固定。

    面接触：
        旋转轴 = 过接触区重心 p、位于接触面内的方向（u = n × lever），
        交替方向滚动，保持平面与环境的贴合。

旋转角度选择：
    角度与估计不确定性正相关（不确定越大，偏转越大以获取更多信息），
    并被滑移分数（slip_score）与最大步长约束；旋转方向由力矩补偿决定
    （选择使后续接触力矩减小的符号，避免绕不同轴力矩差距过大）。

本模块只依赖 numpy 与 contact_models，可直接单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contact_models import ContactType, normalize, point_on_line

# 默认参数
DEFAULT_MIN_ANGLE_DEG = 1.5
DEFAULT_MAX_ANGLE_DEG = 10.0
DEFAULT_UNCERTAINTY_GAIN = 1.5          # 不确定度(mm) -> 角度(rad) 的增益
DEFAULT_MAX_SLIP_RATIO = 0.5            # slip_score 超过阈值时角度缩减系数


@dataclass
class RotationCommand:
    """主动偏转指令：绕世界坐标系下的直线（中心点 + 方向）旋转一定角度。"""

    center_point: np.ndarray            # 旋转轴上的一个点（mm）
    axis_direction: np.ndarray          # 旋转轴方向（单位向量）
    angle: float                        # 旋转角大小（rad，>0）
    sign: int                           # 旋转方向（+1 / -1）
    reason: str                         # 决策说明

    def effective_angle(self) -> float:
        """带符号的旋转角。"""
        return self.sign * self.angle

    def as_dict(self) -> dict[str, Any]:
        return {
            "center_point": np.asarray(self.center_point).tolist(),
            "axis_direction": np.asarray(self.axis_direction).tolist(),
            "angle": self.angle,
            "sign": self.sign,
            "reason": self.reason,
        }


def rotation_angle_from_uncertainty(
    position_std_mm: float,
    min_angle_rad: float,
    max_angle_rad: float,
    gain: float = DEFAULT_UNCERTAINTY_GAIN,
) -> float:
    """由接触点位置不确定度（mm）决定偏转角（rad）。

    不确定度越大 -> 偏转角越大（获取更多信息量），但被 max_angle 限制。
    """
    angle = float(gain * np.deg2rad(position_std_mm))
    return float(np.clip(angle, min_angle_rad, max_angle_rad))


def clamp_angle_by_slip(
    angle: float,
    slip_score: float | None,
    slip_threshold: float | None,
    max_slip_ratio: float = DEFAULT_MAX_SLIP_RATIO,
) -> float:
    """滑移分数接近/超过阈值时缩减偏转角，避免失稳。"""
    if slip_score is None or slip_threshold is None:
        return float(angle)
    excess = max(0.0, float(slip_score) - float(slip_threshold))
    ratio = min(1.0, excess / (abs(float(slip_threshold)) + 1e-12))
    return float(angle * (1.0 - max_slip_ratio * ratio))


def compensated_rotation_sign(
    moment_gripper: np.ndarray,
    axis_gripper: np.ndarray,
    previous_sign: int,
) -> int:
    """力矩补偿：选择使后续接触力矩减小的旋转方向。

    若绕旋转轴方向的力矩分量很大，继续同向旋转会使力矩进一步增大，
    因此取反号；否则沿用上一符号（连续偏转）。
    """
    moment_gripper = np.asarray(moment_gripper, dtype=float)
    axis_gripper = normalize(axis_gripper)
    component = float(np.dot(moment_gripper, axis_gripper))
    threshold = 8.0  # N*mm，超过该值认为绕轴力矩显著
    if abs(component) > threshold and component * previous_sign > 0:
        return -previous_sign
    return previous_sign


def select_line_rotation(
    line_point: np.ndarray,
    line_direction: np.ndarray,
    gripper_reference: np.ndarray,
    moment_gripper: np.ndarray | None,
    previous_sign: int,
    angle: float,
) -> RotationCommand:
    """线接触：绕估计接触线旋转。"""
    direction = normalize(line_direction)
    center = point_on_line(line_point, direction, gripper_reference)
    sign = previous_sign
    if moment_gripper is not None:
        # 力矩方向与旋转方向的关系按杠杆臂近似：这里用补偿函数
        sign = compensated_rotation_sign(moment_gripper, direction, previous_sign)
    return RotationCommand(
        center_point=center,
        axis_direction=direction,
        angle=angle,
        sign=sign,
        reason="line contact: rotate about the estimated contact line",
    )


def select_point_rotation(
    contact_point: np.ndarray,
    object_center: np.ndarray,
    world_reference: np.ndarray,
    previous_sign: int,
    angle: float,
    alternate_index: int = 0,
) -> RotationCommand:
    """点接触：绕通过接触点的两个正交轴之一旋转（交替）。

    杠杆臂 lever = object_center - contact_point，旋转轴取杠杆臂的
    垂直方向；alternate_index=0 用 lever × world_reference，
    =1 用 (lever × world_reference) × lever_hat，交替激励两个方向。
    """
    lever = np.asarray(object_center, dtype=float) - np.asarray(contact_point, dtype=float)
    lever_hat = normalize(lever)
    first = normalize(np.cross(lever_hat, np.asarray(world_reference, dtype=float)))
    if np.linalg.norm(first) < 1e-6:
        first = np.array([1.0, 0.0, 0.0])
    if alternate_index % 2 == 0:
        axis = first
    else:
        axis = normalize(np.cross(first, lever_hat))
    sign = previous_sign if previous_sign != 0 else 1
    return RotationCommand(
        center_point=np.asarray(contact_point, dtype=float).copy(),
        axis_direction=axis,
        angle=angle,
        sign=sign,
        reason="point contact: rotate about an axis through the contact point",
    )


def select_surface_rotation(
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    object_center: np.ndarray,
    previous_sign: int,
    angle: float,
    alternate_index: int = 0,
) -> RotationCommand:
    """面接触：绕接触面内（垂直于法向）的轴旋转，交替滚动。"""
    normal = normalize(plane_normal)
    lever = np.asarray(object_center, dtype=float) - np.asarray(plane_point, dtype=float)
    in_plane = lever - np.dot(lever, normal) * normal
    if np.linalg.norm(in_plane) < 1e-6:
        in_plane = np.array([1.0, 0.0, 0.0])
    in_plane = normalize(in_plane)
    if alternate_index % 2 == 0:
        axis = in_plane
    else:
        axis = normalize(np.cross(normal, in_plane))
    sign = previous_sign if previous_sign != 0 else 1
    return RotationCommand(
        center_point=np.asarray(plane_point, dtype=float).copy(),
        axis_direction=axis,
        angle=angle,
        sign=sign,
        reason="surface contact: rotate about an in-plane axis through the patch",
    )


class RotationSelector:
    """根据当前估计结果选择旋转中心与旋转方向的决策器。"""

    def __init__(
        self,
        min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
        max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG,
        uncertainty_gain: float = DEFAULT_UNCERTAINTY_GAIN,
        max_slip_ratio: float = DEFAULT_MAX_SLIP_RATIO,
    ) -> None:
        self.min_angle_rad = float(np.deg2rad(min_angle_deg))
        self.max_angle_rad = float(np.deg2rad(max_angle_deg))
        self.uncertainty_gain = uncertainty_gain
        self.max_slip_ratio = max_slip_ratio

    def select(
        self,
        contact_type: ContactType,
        contact_point: np.ndarray,
        contact_direction: np.ndarray,
        object_center: np.ndarray,
        gripper_reference: np.ndarray,
        position_std_mm: float | None = None,
        moment_gripper: np.ndarray | None = None,
        slip_score: float | None = None,
        slip_threshold: float | None = None,
        previous_sign: int = 1,
        alternate_index: int = 0,
    ) -> RotationCommand:
        """从当前估计生成旋转指令。

        参数
        ----
        contact_type      : 接触类型；
        contact_point     : 接触点/线上点/面上点（mm，世界系）；
        contact_direction : 线方向或面法向（点接触时可为零向量）；
        object_center     : 当前物体/夹爪参考位置（mm，世界系），
                            用于点/面接触的杠杆臂计算；
        gripper_reference : 用于把线上代表点移到夹爪附近的参考点；
        position_std_mm   : 接触点位置不确定度（mm，来自边际协方差），
                            缺失时用默认角度；
        moment_gripper    : 夹爪系合力矩（用于力矩补偿与方向决策）；
        slip_score / slip_threshold : 滑移监测（可空）；
        previous_sign     : 上一次旋转方向（用于连续/交替）；
        alternate_index   : 点/面接触下选择交替轴的序号。
        """
        angle = self.min_angle_rad
        if position_std_mm is not None:
            angle = rotation_angle_from_uncertainty(
                position_std_mm,
                self.min_angle_rad,
                self.max_angle_rad,
                self.uncertainty_gain,
            )
        angle = clamp_angle_by_slip(
            angle, slip_score, slip_threshold, self.max_slip_ratio
        )

        if contact_type == ContactType.LINE:
            return select_line_rotation(
                contact_point,
                contact_direction,
                gripper_reference,
                moment_gripper,
                previous_sign,
                angle,
            )
        if contact_type == ContactType.POINT:
            return select_point_rotation(
                contact_point,
                object_center,
                np.array([0.0, 0.0, 1.0]),
                previous_sign,
                angle,
                alternate_index,
            )
        if contact_type == ContactType.SURFACE:
            return select_surface_rotation(
                contact_point,
                contact_direction,
                object_center,
                previous_sign,
                angle,
                alternate_index,
            )
        raise ValueError(f"Unsupported contact type: {contact_type}")
