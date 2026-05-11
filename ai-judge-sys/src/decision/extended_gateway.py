"""扩展置信度网关模块

支持三层置信度路由：
- auto (>= 0.8): 直接自动通过/拒绝
- review (0.4-0.8): 三智能体复核
- manual (< 0.4): 直接到人工队列

复核结果路由：
- auto_pass (>= 0.65): 自动通过
- dlq_manual (< 0.65): 死信队列到人工
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ExtendedThresholds:
    """扩展置信度阈值配置"""

    auto_threshold: float = float(os.getenv("AUTO_THRESHOLD", "0.8"))
    review_threshold: float = float(os.getenv("REVIEW_THRESHOLD", "0.4"))
    review_confidence_pass: float = float(os.getenv("REVIEW_CONFIDENCE_PASS", "0.65"))


class ExtendedConfidenceGateway:
    """扩展置信度网关

    三层路由：
    - auto: 置信度 >= 0.8 -> 直接自动通过/拒绝
    - review: 0.4 <= 置信度 < 0.8 -> 多智能体复核
    - manual: 置信度 < 0.4 -> 直接到人工队列

    复核结果路由：
    - auto_pass: 复核后置信度 >= 0.65 -> 自动通过
    - dlq_manual: 复核后置信度 < 0.65 -> 死信队列
    """

    def __init__(self, thresholds: ExtendedThresholds | None = None) -> None:
        self.thresholds = thresholds or ExtendedThresholds()

    def route_initial(self, confidence: float) -> str:
        """基于初始置信度进行路由

        Args:
            confidence: 初始置信度分数 (0.0-1.0)

        Returns:
            路由类型: "auto", "review", 或 "manual"
        """
        if confidence >= self.thresholds.auto_threshold:
            return "auto"
        if confidence < self.thresholds.review_threshold:
            return "manual"
        return "review"

    def route_review_result(self, review_confidence: float) -> str:
        """基于多智能体复核结果进行路由

        Args:
            review_confidence: 复核后的置信度分数 (0.0-1.0)

        Returns:
            路由类型: "auto_pass" 或 "dlq_manual"
        """
        if review_confidence >= self.thresholds.review_confidence_pass:
            return "auto_pass"
        return "dlq_manual"

    def get_threshold_description(self) -> dict[str, float]:
        """获取阈值配置描述"""
        return {
            "auto_threshold": self.thresholds.auto_threshold,
            "review_threshold": self.thresholds.review_threshold,
            "review_confidence_pass": self.thresholds.review_confidence_pass,
        }
