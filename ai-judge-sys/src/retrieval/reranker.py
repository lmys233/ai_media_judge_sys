from __future__ import annotations

from typing import Any


class SimpleReranker:
    """
    Reranker with multi-signal scoring:
    1) violation_type match bonus (any matching type)
    2) human verified bonus
    3) vector similarity score
    4) risk score closeness
    """

    def rerank(
        self,
        records: list[dict[str, Any]],
        target_risk: float,
        target_violation: str | list[str] = "",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        # Normalize to list
        if isinstance(target_violation, str):
            target_violations = [target_violation] if target_violation else []
        else:
            target_violations = target_violation

        def _score(item: dict[str, Any]) -> float:
            item_type = item.get("violation_type", "")
            # Match if any of the target violation types match
            violation_match = 1.0 if (target_violations and item_type in target_violations) else 0.0
            human_bonus = 0.8 if item.get("human_verified", False) else 0.0
            vector_score = float(item.get("score", 0.0))
            risk = float(item.get("risk_score", 0.0))
            risk_closeness = 1.0 - min(abs(risk - target_risk), 1.0)
            return violation_match + human_bonus + vector_score * 0.6 + risk_closeness * 0.3

        return sorted(records, key=_score, reverse=True)[:top_k]
