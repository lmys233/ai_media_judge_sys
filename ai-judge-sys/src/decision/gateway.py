from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DecisionThresholds:
    auto_threshold: float = float(os.getenv("AUTO_THRESHOLD", "0.8"))
    manual_threshold: float = float(os.getenv("MANUAL_THRESHOLD", "0.2"))


class ConfidenceGateway:
    def __init__(self, thresholds: DecisionThresholds | None = None) -> None:
        self.thresholds = thresholds or DecisionThresholds()

    def route(self, confidence: float) -> str:
        if confidence >= self.thresholds.auto_threshold:
            return "auto"
        if confidence <= self.thresholds.manual_threshold:
            return "manual"
        return "review"
