from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.models.contracts import ViolationType


class VideoFrameSummarizer:
    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm

    def summarize(self, frame_evidence: list[str]) -> str:
        joined = "; ".join(frame_evidence[:20]) if frame_evidence else "no frame evidence"
        if self.llm is None:
            return f"Video frame summary: {joined}"

        prompt = (
            "You are a moderation assistant. Summarize video frame content safely and briefly. "
            "Do not include explicit details. Evidence:\n"
            f"{joined}"
        )
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

    def infer_label(self, labels: list[ViolationType], scores: list[float]) -> tuple[ViolationType, float]:
        if not labels:
            return ViolationType.NORMAL, 0.0
        highest_idx = max(range(len(scores)), key=lambda i: scores[i])
        return labels[highest_idx], scores[highest_idx]
