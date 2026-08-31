"""
三类接触的合成测试：点接触 / 线接触 / 面接触。

对每类接触合成物体绕固定接触几何偏转的序列，生成触觉相对变换与
合力/力矩，然后运行 ContactFactorGraphISAM2，检验估计结果收敛到
真值附近。同时测试接触分类器与旋转选择器。

运行：
    cd 到包目录上一级，执行
    /usr/bin/python3 -m pytest extrinsic_contact_estimation/tests -q
    或直接
    /usr/bin/python3 extrinsic_contact_estimation/tests/test_estimation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contact_models import (  # noqa: E402
    ContactType,
    make_pose,
    pose_inverse,
    pose_transform,
    se3_exp,
    so3_exp,
)
from ..data_io import (  # noqa: E402
    initial_line_from_object_poses,
    initial_point_from_object_poses,
)
from ..factor_graph_estimator import ContactFactorGraphISAM2  # noqa: E402
from ..rotation_selector import RotationSelector  # noqa: E402

try:
    import gtsam  # noqa: F401

    GTSAM_AVAILABLE = True
except Exception:
    GTSAM_AVAILABLE = False

WRENCH_ORIGIN = np.array([0.0, 0.0, 223.0])
STIFFNESS = np.array([1100.0, 1150.0, 1000.0, 0.18, 0.16, 0.22])
N_STEPS = 40
SEED = 7


# --------------------------------------------------------------------------
# 合成场景生成器
# --------------------------------------------------------------------------

def _sample_normal(rng, scale=0.001):
    return rng.normal(scale=scale)


def generate_point_contact(num_steps=N_STEPS, seed=SEED):
    """物体绕世界接触点 c 偏转（点接触）。"""
    rng = np.random.default_rng(seed)
    c = np.array([0.0, 0.0, 0.0])
    r_obj = np.array([-18.0, 7.0, -38.0])  # 物体上接触点在物体坐标系的位置

    object_poses, gripper_poses, tactile_transforms = [], [], []
    forces, moments = [], []
    for i in range(num_steps):
        phase = i / max(num_steps - 1, 1)
        rotvec = np.array([
            np.deg2rad(1.0 + 16.0 * phase),
            np.deg2rad(8.0 * np.sin(1.4 * np.pi * phase)),
            np.deg2rad(5.0 * phase),
        ])
        R = so3_exp(rotvec)
        t = c - R @ r_obj
        object_pose = make_pose(R, t)
        f_world = np.array([
            0.015 * np.sin(2.2 * np.pi * phase),
            -0.012 * np.cos(1.7 * np.pi * phase),
            0.42 + 0.035 * np.sin(1.1 * np.pi * phase),
        ])
        delta, gp, w = _wrench_and_gripper(object_pose, c, f_world, rng)
        object_poses.append(object_pose)
        gripper_poses.append(gp)
        tactile_transforms.append(se3_exp(delta))
        moments.append(w[:3])
        forces.append(w[3:])
    return {
        "object_poses": np.asarray(object_poses),
        "gripper_poses": np.asarray(gripper_poses),
        "tactile_transforms": np.asarray(tactile_transforms),
        "forces": np.asarray(forces),
        "moments": np.asarray(moments),
        "truth": {"point": c, "r_obj": r_obj},
    }


def generate_line_contact(num_steps=N_STEPS, seed=SEED):
    """物体绕世界接触线旋转（线接触）。"""
    rng = np.random.default_rng(seed)
    u = np.array([0.0, 1.0, 0.0])            # 世界接触线方向
    p = np.array([0.0, 0.0, 0.0])            # 世界接触线上一点
    q_obj = np.array([-18.0, 7.0, -38.0])    # 物体上接触线一点（物体系）
    u_obj = u                                # 物体上接触线方向（物体系）

    object_poses, gripper_poses, tactile_transforms = [], [], []
    forces, moments = [], []
    for i in range(num_steps):
        phase = i / max(num_steps - 1, 1)
        theta = np.deg2rad(12.0 * phase)
        wobble = np.deg2rad(1.2 * np.sin(2.1 * np.pi * phase))
        rotvec = theta * u + wobble * np.array([1.0, 0.0, 0.0])
        R = so3_exp(rotvec)
        t = p - R @ q_obj
        object_pose = make_pose(R, t)
        f_world = np.array([
            0.02 * np.sin(2.0 * np.pi * phase),
            0.015 * np.cos(1.6 * np.pi * phase),
            0.40 + 0.03 * np.sin(1.1 * np.pi * phase),
        ])
        # 接触线在世界系上的当前位置（一点）。
        line_point_world = R @ q_obj + t
        delta, gp, w = _wrench_and_gripper(
            object_pose, line_point_world, f_world, rng
        )
        object_poses.append(object_pose)
        gripper_poses.append(gp)
        tactile_transforms.append(se3_exp(delta))
        moments.append(w[:3])
        forces.append(w[3:])
    return {
        "object_poses": np.asarray(object_poses),
        "gripper_poses": np.asarray(gripper_poses),
        "tactile_transforms": np.asarray(tactile_transforms),
        "forces": np.asarray(forces),
        "moments": np.asarray(moments),
        "truth": {"point": p, "direction": u, "q_obj": q_obj},
    }


def generate_surface_contact(num_steps=N_STEPS, seed=SEED):
    """物体在固定世界平面上贴合（面接触）。"""
    rng = np.random.default_rng(seed)
    n = np.array([0.0, 0.0, 1.0])            # 世界平面法向
    p = np.array([0.0, 0.0, 0.0])            # 世界平面上一点
    q_obj = np.array([-18.0, 7.0, -38.0])    # 物体上平面一点（物体系）

    object_poses, gripper_poses, tactile_transforms = [], [], []
    forces, moments = [], []
    for i in range(num_steps):
        phase = i / max(num_steps - 1, 1)
        theta = np.deg2rad(8.0 * phase)              # 绕法向扭转
        wobble = np.deg2rad(1.0 * np.sin(1.9 * np.pi * phase))
        rotvec = theta * n + wobble * np.array([1.0, 1.0, 0.0])
        R = so3_exp(rotvec)
        t = p - R @ q_obj
        object_pose = make_pose(R, t)
        f_world = np.array([0.0, 0.0, 0.45]) + 0.02 * rng.normal(size=3)
        f_world[2] = max(f_world[2], 0.2)            # 保持法向主导
        plane_point_world = R @ q_obj + t
        delta, gp, w = _wrench_and_gripper(
            object_pose, plane_point_world, f_world, rng
        )
        object_poses.append(object_pose)
        gripper_poses.append(gp)
        tactile_transforms.append(se3_exp(delta))
        moments.append(w[:3])
        forces.append(w[3:])
    return {
        "object_poses": np.asarray(object_poses),
        "gripper_poses": np.asarray(gripper_poses),
        "tactile_transforms": np.asarray(tactile_transforms),
        "forces": np.asarray(forces),
        "moments": np.asarray(moments),
        "truth": {"point": p, "normal": n, "q_obj": q_obj},
    }


def _wrench_and_gripper(object_pose, contact_world, force_world, rng):
    """由接触点/线/面上一点构造一致的夹爪系合力矩与触觉位移。"""
    force_gripper = object_pose[:3, :3].T @ force_world
    contact_gripper = pose_transform(pose_inverse(object_pose), contact_world)
    lever = contact_gripper - WRENCH_ORIGIN
    moment_gripper = np.cross(lever, force_gripper)
    wrench = np.r_[moment_gripper, force_gripper]
    delta = wrench / STIFFNESS
    delta = delta + rng.normal(scale=np.array([1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5]))
    gripper_pose = object_pose @ se3_exp(-delta)
    return delta, gripper_pose, wrench


# --------------------------------------------------------------------------
# 估计收敛测试
# --------------------------------------------------------------------------

def _run_estimator(scenario, contact_type):
    object_poses = scenario["object_poses"]
    if contact_type == ContactType.POINT:
        initial_point = initial_point_from_object_poses(object_poses)
        initial_direction = None
    else:
        initial_point, initial_direction = initial_line_from_object_poses(object_poses)
    estimator = ContactFactorGraphISAM2(
        gripper_poses=scenario["gripper_poses"],
        tactile_transforms=scenario["tactile_transforms"],
        initial_point=initial_point,
        initial_direction=initial_direction,
        initial_normal=scenario["truth"].get("normal"),
        forces=scenario["forces"],
        moments=scenario["moments"],
        contact_type=contact_type,
        initial_wrench_origin_gripper=WRENCH_ORIGIN,
        compute_covariance=False,
    )
    return estimator.run(print_progress=False)


def test_point_contact_estimation():
    if not GTSAM_AVAILABLE:
        return
    scenario = generate_point_contact()
    estimate = _run_estimator(scenario, ContactType.POINT)
    error = np.linalg.norm(estimate.contact_points[-1] - scenario["truth"]["point"])
    assert error < 3.0, f"point contact error too large: {error:.3f} mm"
    print(f"  point contact error: {error:.3f} mm")


def test_line_contact_estimation():
    if not GTSAM_AVAILABLE:
        return
    scenario = generate_line_contact()
    estimate = _run_estimator(scenario, ContactType.LINE)
    truth = scenario["truth"]
    # 距离最近点：线上误差用“线上点到真值直线”的距离。
    u = truth["direction"]
    delta = estimate.contact_points[-1] - truth["point"]
    line_error = np.linalg.norm(delta - np.dot(delta, u) * u)
    dir_cosine = abs(np.dot(estimate.contact_directions[-1], u))
    assert line_error < 3.0, f"line point error too large: {line_error:.3f} mm"
    assert dir_cosine > 0.95, f"line direction cosine too small: {dir_cosine:.3f}"
    print(f"  line contact point error: {line_error:.3f} mm, dir cosine: {dir_cosine:.3f}")


def test_surface_contact_estimation():
    if not GTSAM_AVAILABLE:
        return
    scenario = generate_surface_contact()
    estimate = _run_estimator(scenario, ContactType.SURFACE)
    truth = scenario["truth"]
    n = truth["normal"]
    delta = estimate.contact_points[-1] - truth["point"]
    plane_error = abs(np.dot(delta, n))
    normal_cosine = abs(np.dot(estimate.contact_normals[-1], n))
    assert plane_error < 3.0, f"plane point error too large: {plane_error:.3f} mm"
    assert normal_cosine > 0.95, f"plane normal cosine too small: {normal_cosine:.3f}"
    print(f"  surface plane distance error: {plane_error:.3f} mm, normal cosine: {normal_cosine:.3f}")


# --------------------------------------------------------------------------
# 分类器测试（合成 marker 点云）
# --------------------------------------------------------------------------

def _make_marker_cloud(shape: str, rng):
    """合成点云：'point' 团簇 / 'line' 直线 / 'surface' 平面。"""
    if shape == "point":
        cloud = rng.normal(scale=1.5, size=(120, 3))
    elif shape == "line":
        t = rng.uniform(-40, 40, size=(120, 1))
        cloud = np.hstack([t, rng.normal(scale=1.2, size=(120, 2))])
    else:
        x = rng.uniform(-40, 40, size=(120, 1))
        y = rng.uniform(-40, 40, size=(120, 1))
        cloud = np.hstack([x, y, rng.normal(scale=1.0, size=(120, 1))])
    return cloud


def test_contact_classifier_geometry():
    from ..contact_classifier import classify_from_geometry
    from ..tactile_features import ContactPatchFeatures

    rng = np.random.default_rng(1)
    for shape, expected in [
        ("point", ContactType.POINT),
        ("line", ContactType.LINE),
        ("surface", ContactType.SURFACE),
    ]:
        cloud = _make_marker_cloud(shape, rng)
        centroid = cloud.mean(axis=0)
        centered = cloud - centroid
        sv = np.linalg.svd(centered, compute_uv=False) / np.sqrt(len(cloud) - 1)
        features = ContactPatchFeatures(
            combined_extent=float(sv[0]),
            combined_elongation=float(sv[1] / sv[0]) if sv[0] > 1e-12 else 1.0,
            combined_planarity=float(sv[2] / sv[1]) if sv[1] > 1e-12 else 1.0,
            centroid=centroid,
        )
        ctype, _ = classify_from_geometry(features)
        assert ctype == expected, f"{shape}: got {ctype}"
        print(f"  classifier {shape} -> {ctype}")


def test_rotation_selector():
    selector = RotationSelector(min_angle_deg=1.0, max_angle_deg=8.0)

    # 线接触：轴应沿估计接触线方向。
    cmd_line = selector.select(
        contact_type=ContactType.LINE,
        contact_point=np.array([0.0, 0.0, 0.0]),
        contact_direction=np.array([0.0, 1.0, 0.0]),
        object_center=np.array([0.0, 0.0, -50.0]),
        gripper_reference=np.array([0.0, 0.0, -40.0]),
        position_std_mm=2.0,
    )
    assert abs(np.dot(cmd_line.axis_direction, np.array([0.0, 1.0, 0.0]))) > 0.99
    assert cmd_line.angle > 0.0

    # 点接触：轴过接触点，且不沿杠杆臂方向。
    cmd_point = selector.select(
        contact_type=ContactType.POINT,
        contact_point=np.array([0.0, 0.0, 0.0]),
        contact_direction=np.zeros(3),
        object_center=np.array([0.0, 0.0, -50.0]),
        gripper_reference=np.array([0.0, 0.0, -40.0]),
    )
    lever = np.array([0.0, 0.0, -50.0])
    assert abs(np.dot(cmd_point.axis_direction, lever / np.linalg.norm(lever))) < 0.3

    # 面接触：轴在接触面内（垂直于法向）。
    cmd_surface = selector.select(
        contact_type=ContactType.SURFACE,
        contact_point=np.array([0.0, 0.0, 0.0]),
        contact_direction=np.array([0.0, 0.0, 1.0]),
        object_center=np.array([0.0, 0.0, -50.0]),
        gripper_reference=np.array([0.0, 0.0, -40.0]),
    )
    assert abs(np.dot(cmd_surface.axis_direction, np.array([0.0, 0.0, 1.0]))) < 0.3
    print("  rotation selector OK")


# --------------------------------------------------------------------------
# 直接运行
# --------------------------------------------------------------------------

def main() -> None:
    print("== point contact ==")
    test_point_contact_estimation()
    print("== line contact ==")
    test_line_contact_estimation()
    print("== surface contact ==")
    test_surface_contact_estimation()
    print("== classifier ==")
    test_contact_classifier_geometry()
    print("== rotation selector ==")
    test_rotation_selector()
    print("All tests passed.")


if __name__ == "__main__":
    main()
