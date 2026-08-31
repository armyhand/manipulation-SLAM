"""
接触类型分类器：区分点接触 / 线接触 / 面接触。

分类依据两部分：

1. 几何特征（主要）：活跃接触区点云的 PCA 形状。
   - 点接触  ：主轴半径小（紧凑团簇）；
   - 线接触  ：主轴半径大、次轴/主轴比值小（沿一条线拉长）；
   - 面接触  ：两轴都大、第三轴/次轴比值小（平面内展开）。

2. 力矩模型拟合（校验）：用夹爪合力/合力矩在候选接触几何处的残差
   判断哪个接触模型最一致。
   - 点接触  ：接触点处合力矩 ||m_c|| 接近零；
   - 线接触  ：接触处绕线方向无扭矩 |m_c·u| 接近零；
   - 面接触  ：接触处绕面法向无扭矩 |m_c·n| 接近零。

两类证据通过 softmax 似然加权融合，输出接触类型、置信度与各分数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contact_models import ContactType, wrench_model_scores
from .tactile_features import (
    DEFAULT_LINE_ELONGATION,
    DEFAULT_POINT_EXTENT_MM,
    DEFAULT_SURFACE_PLANARITY,
    ContactPatchFeatures,
)

EPS = 1e-12


@dataclass
class ContactClassification:
    """接触类型分类结果。"""

    contact_type: ContactType
    confidence: float = 0.0
    geometric_type: ContactType | None = None
    wrench_type: ContactType | None = None
    scores: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contact_type": self.contact_type.value,
            "confidence": self.confidence,
            "geometric_type": (
                self.geometric_type.value if self.geometric_type else None
            ),
            "wrench_type": self.wrench_type.value if self.wrench_type else None,
            "scores": self.scores,
            "features": self.features,
        }


def _softmax(values: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax。"""
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / (np.sum(exp) + EPS)


def classify_from_geometry(
    features: ContactPatchFeatures,
    point_extent_mm: float = DEFAULT_POINT_EXTENT_MM,
    line_elongation: float = DEFAULT_LINE_ELONGATION,
    surface_planarity: float = DEFAULT_SURFACE_PLANARITY,
) -> tuple[ContactType | None, float]:
    """由几何特征做规则式分类，返回 (类型, 证据强度)。

    证据强度取“规则判定边距”的饱和函数：距离阈值越远越可信。
    """
    extent = features.combined_extent
    elongation = features.combined_elongation
    planarity = features.combined_planarity

    if extent < point_extent_mm:
        # 紧凑团簇：点接触
        margin = point_extent_mm - extent
        evidence = 1.0 - float(np.clip(margin / max(point_extent_mm, EPS), 0.0, 1.0))
        return ContactType.POINT, evidence
    if elongation < line_elongation:
        # 沿一条线拉长：线接触
        margin = line_elongation - elongation
        evidence = float(np.clip(margin / max(line_elongation, EPS), 0.0, 1.0))
        return ContactType.LINE, evidence
    if planarity < surface_planarity:
        # 平面内展开：面接触
        margin = surface_planarity - planarity
        evidence = float(np.clip(margin / max(surface_planarity, EPS), 0.0, 1.0))
        return ContactType.SURFACE, evidence
    return None, 0.0


def classify_from_wrench(
    force_gripper: np.ndarray,
    moment_gripper: np.ndarray,
    gripper_pose: np.ndarray,
    candidate_point: np.ndarray,
    candidate_direction: np.ndarray | None = None,
    candidate_normal: np.ndarray | None = None,
    wrench_origin_gripper: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> tuple[ContactType | None, float, dict[str, float]]:
    """由合力/合力矩在候选接触几何处的残差分类。

    返回 (类型, 证据强度, 各类残差分数)。残差越小拟合越好，
    转换为“负对数似然”后做 softmax 取最大者。
    """
    scores = wrench_model_scores(
        force_gripper,
        moment_gripper,
        gripper_pose,
        candidate_point,
        candidate_direction,
        candidate_normal,
        wrench_origin_gripper,
    )
    order = [ContactType.POINT, ContactType.LINE, ContactType.SURFACE]
    residual = np.array([scores[t.value] for t in order], dtype=float)

    # 候选几何缺失导致残差为 inf 时，用很大的有限值替代，
    # 使分类只在可用类型之间进行（面接触缺少法向时排除面接触）。
    unavailable = [False, False, False]
    if candidate_direction is None:
        unavailable[1] = True
    if candidate_normal is None:
        unavailable[2] = True
    residual[~np.isfinite(residual)] = 1e6
    residual[np.asarray(unavailable, dtype=bool)] = 1e6

    # 所有类型都不可用时返回 None（有可用类型时取残差最小者）。
    if np.all(unavailable):
        return None, 0.0, scores

    # 用残差倒数作为“得分”，做 softmax 取概率；即使证据较弱也返回
    # 最佳模型，由上层按几何证据加权/冲突处理决定最终类型。
    inverse = 1.0 / (residual + 1.0)
    probabilities = _softmax(inverse)
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])

    return order[best_index], confidence, scores


class ContactClassifier:
    """融合几何与力矩证据的接触类型分类器。"""

    def __init__(
        self,
        point_extent_mm: float = DEFAULT_POINT_EXTENT_MM,
        line_elongation: float = DEFAULT_LINE_ELONGATION,
        surface_planarity: float = DEFAULT_SURFACE_PLANARITY,
        geometric_weight: float = 0.7,
        wrench_weight: float = 0.3,
    ) -> None:
        self.point_extent_mm = point_extent_mm
        self.line_elongation = line_elongation
        self.surface_planarity = surface_planarity
        self.geometric_weight = geometric_weight
        self.wrench_weight = wrench_weight

    def classify(
        self,
        features: ContactPatchFeatures,
        force_gripper: np.ndarray | None = None,
        moment_gripper: np.ndarray | None = None,
        gripper_pose: np.ndarray | None = None,
        candidate_point: np.ndarray | None = None,
        candidate_direction: np.ndarray | None = None,
        candidate_normal: np.ndarray | None = None,
        wrench_origin_gripper: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ) -> ContactClassification:
        """融合几何证据与（可选的）力矩证据进行分类。"""
        geometric_type, geometric_evidence = classify_from_geometry(
            features,
            point_extent_mm=self.point_extent_mm,
            line_elongation=self.line_elongation,
            surface_planarity=self.surface_planarity,
        )

        wrench_type = None
        wrench_evidence = 0.0
        wrench_scores: dict[str, float] = {}
        if (
            force_gripper is not None
            and moment_gripper is not None
            and gripper_pose is not None
            and candidate_point is not None
        ):
            wrench_type, wrench_evidence, wrench_scores = classify_from_wrench(
                force_gripper,
                moment_gripper,
                gripper_pose,
                candidate_point,
                candidate_direction,
                candidate_normal,
                wrench_origin_gripper,
            )

        # 融合：几何证据为主，力矩证据为辅（只有几何证据时直接采用几何类型）。
        if geometric_type is None and wrench_type is not None:
            final_type = wrench_type
            confidence = wrench_evidence
        elif geometric_type is None:
            final_type = ContactType.LINE  # 默认：本任务以线接触为主
            confidence = 0.0
        elif wrench_type is None:
            final_type = geometric_type
            confidence = geometric_evidence
        else:
            weight_geo = self.geometric_weight
            weight_wrench = self.wrench_weight
            total = weight_geo + weight_wrench
            geo_score = weight_geo / total * geometric_evidence
            wrench_score = weight_wrench / total * wrench_evidence
            if geometric_type == wrench_type:
                final_type = geometric_type
                confidence = float(np.clip(geo_score + wrench_score, 0.0, 1.0))
            else:
                final_type = geometric_type
                confidence = float(
                    np.clip(geo_score, 0.0, 1.0)
                )  # 几何证据优先，冲突时降级

        features_dict = {
            "combined_extent": features.combined_extent,
            "combined_elongation": features.combined_elongation,
            "combined_planarity": features.combined_planarity,
            "n_active_total": features.n_active_total,
            "centroid": np.asarray(features.centroid).tolist(),
        }

        return ContactClassification(
            contact_type=final_type,
            confidence=confidence,
            geometric_type=geometric_type,
            wrench_type=wrench_type,
            scores={
                "geometric": {
                    t.value: (
                        1.0 if geometric_type == t else 0.0
                    )
                    for t in ContactType
                },
                "wrench": wrench_scores,
            },
            features=features_dict,
        )
