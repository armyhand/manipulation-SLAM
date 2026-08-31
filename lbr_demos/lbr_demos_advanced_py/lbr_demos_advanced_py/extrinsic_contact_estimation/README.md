# 外部接触估计与主动偏转（因子图优化）

在接触发生后，对机器人夹爪所持物体与环境之间的**外部接触（extrinsic contact）**
进行因子图优化估计，实现：

1. **接触类型判别**：点接触 / 线接触 / 面接触；
2. **接触几何估计**：接触点（点接触）、接触线（线接触）、接触平面（面接触）的稳定估计；
3. **主动偏转决策**：根据当前估计结果选择旋转中心点与旋转方向，
   绕接触点/线继续稳态偏转，逐步修正旋转轴，最终实现稳定接触与鲁棒估计。

方法依据文献：

- **Simultaneous Tactile Estimation and Control of Extrinsic Contact**
  (S. Kim, D. K. Jha, D. Romeres, P. M. Patre, A. Rodriguez; ICRA 2023, arXiv:2303.03385)
- **TEXterity: Tactile Extrinsic deXterity**
  (A. Bronars, S. Kim, P. Patre, A. Rodriguez; arXiv:2401.10230 / 2403.00049)

本包是独立模块，**不修改** `pose_planning_node_realtime_contact_line.py` 等既有文件，
可在离线 npz 数据或 ROS2 实时话题上运行。

---

## 目录结构

```
extrinsic_contact_estimation/
├── README.md                 本说明
├── contact_models.py         接触几何模型与残差函数（点/线/面，只依赖 numpy）
├── tactile_features.py       触觉 marker 几何特征提取
├── contact_classifier.py     接触类型分类器（几何 + 力矩证据融合）
├── factor_graph_estimator.py GTSAM ISAM2 统一接触因子图
├── rotation_selector.py      旋转中心/方向选择与角度决策
├── active_probing.py         主动偏转估计闭环（含可选 ROS2 节点示例）
├── data_io.py                数据预处理（npz / 内存快照）
├── main_offline.py           离线分析入口
├── docs/
│   ├── 设计说明.md           算法设计与数学推导
│   └── 使用说明.md           使用与参数调优手册
└── tests/
    └── test_estimation.py    三类接触的合成测试
```

## 快速开始

### 依赖

- Python 3.10+（推荐 `/usr/bin/python3`，ROS2 Humble 自带）
- `numpy`, `scipy`
- `gtsam`（因子图求解）
- `matplotlib`（离线绘图，可选）
- `rclpy` + `tutorial_interfaces`（仅 ROS2 实时节点需要）

```bash
sudo apt install python3-gtsam    # 或 pip install gtsam
```

### 离线分析

在 `lbr_demos_advanced_py/lbr_demos_advanced_py` 目录下执行：

```bash
/usr/bin/python3 -m lbr_demos_advanced_py.extrinsic_contact_estimation.main_offline \
    --data /path/to/Force_data_1.npz \
    --out   /path/to/output_dir
```

输出接触类型、接触点/线、位置不确定度、旋转指令，并保存：
`contact_estimation.png`、`rotation_axis.png`、`contact_probing_report.md`。

### 合成测试

```bash
/usr/bin/python3 -m lbr_demos_advanced_py.extrinsic_contact_estimation.tests.test_estimation
```

验证点/线/面三类接触估计收敛（<3 mm）与分类、旋转选择。

### 集成到实时节点

`active_probing.py` 提供 `ActiveProbingPipeline`，可被现有
`pose_planning_node_realtime_contact_line.py` 直接调用（见 `docs/使用说明.md`）：

```python
from lbr_demos_advanced_py.extrinsic_contact_estimation import ActiveProbingPipeline

pipeline = ActiveProbingPipeline(frame_step=5, top_n=60, compute_covariance=True)

# 每轮接触缓冲更新后：
result = pipeline.feed_snapshot(
    snapshot,                     # TactileSnapshot，由节点缓存构造
    slip_score=node.slip_score[-1],
    slip_threshold=node.slip_threshold,
)
# result.rotation_command: 旋转中心 center_point、轴方向 axis_direction、
#                         角度 angle 与方向 sign
```

也可直接运行内置 ROS2 节点示例 `RealtimeContactEstimationNode`。

## 核心思想（一句话）

以 GTSAM/ISAM2 因子图同时估计物体位姿与外部接触几何（点/线/面），
用**接触处力矩残差**区分接触类型（点接触：力矩全零；线接触：仅沿线无扭矩；
面接触：仅绕法向无扭矩），并据此决定**绕接触几何的主动偏转轴**，
形成“估计 → 选轴 → 偏转 → 再估计”的闭环。

详见 [`docs/设计说明.md`](docs/设计说明.md) 与 [`docs/使用说明.md`](docs/使用说明.md)。
