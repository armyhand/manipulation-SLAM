"""
外部接触估计与主动偏转闭环（基于因子图优化）。

本包实现接触后的因子图优化估计：
    1. 区分点接触 / 线接触 / 面接触；
    2. 根据当前估计结果选择旋转中心点与旋转方向；
    3. 逐步修正旋转轴，实现接触点/线的稳定估计。

对应文献：
    "Simultaneous Tactile Estimation and Control of Extrinsic Contact"
        (Y. Luo, S. Wang, K. Swaminathan, C. K. Liu, D. Rus, A. Rodriguez; RSS 2021)
    "TEXterity: Tactile Extrinsic deXterity"

主要模块：
    contact_models         接触几何模型与残差函数（点/线/面）；
    tactile_features       触觉 marker 几何特征；
    contact_classifier     接触类型分类器；
    factor_graph_estimator GTSAM ISAM2 统一接触因子图；
    rotation_selector      旋转中心与方向选择；
    active_probing         主动偏转估计闭环；
    data_io                数据预处理（npz / 内存快照）。
"""

from .active_probing import ActiveProbingPipeline, ContactProbingResult
from .contact_classifier import ContactClassification, ContactClassifier
from .contact_models import ContactType
from .data_io import (
    ROTATION_LEFT_GT,
    ROTATION_RIGHT_GT,
    TactileSnapshot,
    load_snapshot_from_npz,
)
from .factor_graph_estimator import ContactEstimate, ContactFactorGraphISAM2
from .rotation_selector import RotationCommand, RotationSelector

__all__ = [
    "ActiveProbingPipeline",
    "ContactProbingResult",
    "ContactClassification",
    "ContactClassifier",
    "ContactType",
    "ContactEstimate",
    "ContactFactorGraphISAM2",
    "RotationCommand",
    "RotationSelector",
    "TactileSnapshot",
    "load_snapshot_from_npz",
    "ROTATION_LEFT_GT",
    "ROTATION_RIGHT_GT",
]

__version__ = "1.0.0"
