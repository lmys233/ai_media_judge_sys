from __future__ import annotations

from src.models.contracts import ManualReviewResultMessage
from src.retrieval.service import RetrievalService


class FeedbackLearner:
    """
    Minimal online learning policy:
    - human verified samples are always written back to vector store.
    - weak labels can be appended by confidence policy in engine.
    """

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service

    def apply_manual_result(self, result: ManualReviewResultMessage) -> dict:
        # In this PoC we only return signal; engine owns actual upsert context.
        return {
            "task_id": result.task_id,
            "trace_id": result.trace_id,
            "final_label": result.final_label.value,
            "approved": result.approved,
            "reviewer": result.reviewer,
            "human_verified": True,
        }
