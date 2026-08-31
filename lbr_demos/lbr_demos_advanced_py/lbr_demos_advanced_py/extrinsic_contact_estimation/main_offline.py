"""
离线分析：在 npz 数据集上运行“接触分类 -> 因子图估计 -> 旋转选择”闭环。

用法：
    /usr/bin/python3 main_offline.py [--data PATH.npz] [--start N] [--step N] [--out DIR]

默认数据为 robot_pivoting_estimate/260806/Force_data_1.npz。
输出：结果文本、分类图、估计过程图、旋转轴可视化与 Markdown 报告。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .active_probing import ActiveProbingPipeline
from .contact_models import ContactType
from .data_io import detect_contact_start, load_snapshot_from_npz


def _default_data_path() -> Path:
    candidates = [
        Path(
            "/home/armyhand/contact_point_estimation_and_scene_reconstruct/"
            "robot_pivoting_estimate/260806/Force_data_1.npz"
        ),
        Path(
            "/home/armyhand/contact_point_estimation_and_scene_reconstruct/"
            "robot_pivoting_estimate/260807_3/Force_data_2.npz"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_offline(
    data_path: Path,
    contact_start_frame: int | None = None,
    frame_step: int = 5,
    top_n: int = 60,
    compute_covariance: bool = True,
    output_dir: Path | None = None,
) -> None:
    """运行离线闭环并输出报告与图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) if output_dir is not None else data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading {data_path}")
    snapshot = load_snapshot_from_npz(data_path)
    if contact_start_frame is None:
        contact_start_frame = detect_contact_start(snapshot)
    snapshot.contact_start_frame = contact_start_frame
    print(f"      contact_start_frame={contact_start_frame}, frames={snapshot.frame_count}")

    pipeline = ActiveProbingPipeline(
        frame_step=frame_step,
        top_n=top_n,
        compute_covariance=compute_covariance,
    )
    print(f"[2/4] Running classification + factor-graph estimation")
    result = pipeline.feed_snapshot(snapshot)

    classification = result.classification
    print(f"      contact type    : {classification.contact_type}")
    print(f"      confidence      : {classification.confidence:.3f}")
    print(f"      geometric type  : {classification.geometric_type}")
    print(f"      wrench type     : {classification.wrench_type}")

    estimate = result.estimate
    point = np.mean(estimate.contact_points, axis=0)
    direction = np.mean(estimate.contact_directions, axis=0)
    print(f"      contact point(mm): {point}")
    if classification.contact_type == ContactType.LINE:
        print(f"      line direction   : {direction}")

    cmd = result.rotation_command
    print(f"[3/4] Rotation command")
    print(f"      center_point(mm) : {cmd.center_point}")
    print(f"      axis_direction   : {cmd.axis_direction}")
    print(f"      angle(deg)       : {np.rad2deg(cmd.angle):.2f}  sign={cmd.sign}")
    print(f"      reason           : {cmd.reason}")
    if result.position_std_mm is not None:
        print(f"      position_std(mm) : {result.position_std_mm:.3f}")

    print(f"[4/4] Saving figures & report to {output_dir}")
    save_figures(result, output_dir)
    save_report(result, output_dir / "contact_probing_report.md")
    print("Done.")


def save_figures(result, output_dir: Path) -> None:
    """保存接触估计过程与旋转轴可视化图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimate = result.estimate
    frames = np.arange(len(estimate.contact_points))

    # 图1：接触点/线随帧的估计过程。
    fig1, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = ["x", "y", "z"]
    for axis_index, label in enumerate(labels):
        axes[axis_index].plot(
            frames, estimate.contact_points[:, axis_index], label=f"contact {label}"
        )
        axes[axis_index].plot(
            frames,
            estimate.object_poses[:, axis_index, 3],
            "--",
            label=f"object {label}",
        )
        axes[axis_index].set_ylabel("mm")
        axes[axis_index].legend(fontsize=8)
    axes[0].set_title(
        f"Contact estimation [{result.classification.contact_type}] "
        f"(conf={result.classification.confidence:.2f})"
    )
    axes[-1].set_xlabel("frame")
    fig1.tight_layout()
    fig1.savefig(output_dir / "contact_estimation.png", dpi=160)
    plt.close(fig1)

    # 图2：旋转轴可视化（世界坐标系）。
    fig2 = plt.figure(figsize=(7, 7))
    ax = fig2.add_subplot(111, projection="3d")
    cmd = result.rotation_command
    center = cmd.center_point
    axis = cmd.axis_direction * 40.0
    ax.plot(
        [center[0] - axis[0], center[0] + axis[0]],
        [center[1] - axis[1], center[1] + axis[1]],
        [center[2] - axis[2], center[2] + axis[2]],
        color="tab:red",
        linewidth=3,
        label="rotation axis",
    )
    ax.scatter(*center, color="black", s=60, label="center point")
    object_xyz = estimate.object_poses[:, :3, 3]
    ax.plot(
        object_xyz[:, 0], object_xyz[:, 1], object_xyz[:, 2],
        color="tab:blue", label="object path",
    )
    ax.scatter(
        estimate.contact_points[:, 0],
        estimate.contact_points[:, 1],
        estimate.contact_points[:, 2],
        color="tab:orange", s=12, label="contact point",
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend(fontsize=8)
    ax.set_title("Rotation axis from contact estimate")
    fig2.tight_layout()
    fig2.savefig(output_dir / "rotation_axis.png", dpi=160)
    plt.close(fig2)


def save_report(result, output_path: Path) -> None:
    """把结果摘要写入 Markdown 报告。"""
    classification = result.classification
    estimate = result.estimate
    cmd = result.rotation_command

    lines = [
        "# 外部接触估计与主动偏转报告",
        "",
        f"- 接触类型：`{classification.contact_type}`（置信度 {classification.confidence:.2f}）",
        f"- 几何判据：{classification.geometric_type}，力矩判据：{classification.wrench_type}",
        f"- 接触起始帧：{result.contact_start_frame}",
        f"- 接触代表点(mm)：{np.mean(estimate.contact_points, axis=0).tolist()}",
        f"- 接触代表方向：{np.mean(estimate.contact_directions, axis=0).tolist()}",
        f"- 位置不确定度(mm)："
        f"{result.position_std_mm if result.position_std_mm is not None else 'N/A'}",
        "",
        "## 旋转指令",
        "",
        f"- 中心点(mm)：{cmd.center_point.tolist()}",
        f"- 轴方向：{cmd.axis_direction.tolist()}",
        f"- 角度(deg)：{np.rad2deg(cmd.angle):.2f}，方向 sign={cmd.sign}",
        f"- 原因：{cmd.reason}",
        "",
        "## 分类细节",
        "",
        "```json",
        str(classification.as_dict()),
        "```",
        "",
        "## 附图",
        "",
        "- `contact_estimation.png`：接触几何随帧的估计过程",
        "- `rotation_axis.png`：旋转轴与接触点可视化",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="离线接触估计与主动偏转")
    parser.add_argument("--data", type=Path, default=None, help="npz 数据路径")
    parser.add_argument("--start", type=int, default=None, help="接触起始帧")
    parser.add_argument("--step", type=int, default=5, help="分析帧步长")
    parser.add_argument("--out", type=Path, default=None, help="输出目录")
    args = parser.parse_args()

    data_path = args.data if args.data is not None else _default_data_path()
    run_offline(
        data_path,
        contact_start_frame=args.start,
        frame_step=args.step,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
