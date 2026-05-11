from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.models.contracts import AuditDecision


@dataclass
class ReviewerResult:
    accepted: bool
    reason: str


class CoTAuditor:
    def __init__(self, llm: BaseChatModel | None = None, focus: str = "logic") -> None:
        self.llm = llm
        self.focus = focus

    def audit(self, decision: AuditDecision, metadata: dict) -> ReviewerResult:
        if self.llm is None:
            # PoC heuristic fallback
            if decision.confidence < 0.3 and decision.final_label.value == "normal":
                return ReviewerResult(False, f"{self.focus}_fallback_reject_low_confidence")
            return ReviewerResult(True, f"{self.focus}_fallback_accept")

        prompt = (
            f"你是{self.focus}审计模型。判断当前审核结论是否可靠，只返回 YES 或 NO 开头并附一句原因。\n"
            f"decision={decision.model_dump_json()}\n"
            f"metadata={metadata}\n"
        )
        resp = self.llm.invoke([HumanMessage(content=prompt)])
        content = str(getattr(resp, "content", "")).strip()
        if content.upper().startswith("YES"):
            return ReviewerResult(True, content)
        return ReviewerResult(False, content or f"{self.focus}_reject")


class ReActReviewer:
    def __init__(
        self,
        react_llm: BaseChatModel | None = None,
        logic_auditor: CoTAuditor | None = None,
        evidence_auditor: CoTAuditor | None = None,
        max_retry: int = 3,
    ) -> None:
        self.react_llm = react_llm
        self.logic_auditor = logic_auditor or CoTAuditor(focus="logic")
        self.evidence_auditor = evidence_auditor or CoTAuditor(focus="evidence")
        self.max_retry = max_retry

    def review(self, decision: AuditDecision, metadata: dict) -> AuditDecision:
        current = decision
        attempt = 0
        while attempt < self.max_retry:
            logic_result = self.logic_auditor.audit(current, metadata)
            evidence_result = self.evidence_auditor.audit(current, metadata)

            if logic_result.accepted and evidence_result.accepted:
                current.decision_path.extend(["react_review_pass", f"attempt_{attempt + 1}"])
                return current

            attempt += 1
            current.decision_path.extend(
                [
                    "react_review_reject",
                    f"logic={logic_result.reason}",
                    f"evidence={evidence_result.reason}",
                ]
            )
            current = self._rethink(current, metadata, attempt)

        current.needs_manual = True
        current.reason = f"react_review_exceeded_{self.max_retry}"
        current.decision_path.append("react_to_manual")
        return current

    def _rethink(self, decision: AuditDecision, metadata: dict, attempt: int) -> AuditDecision:
        if self.react_llm is None:
            # fallback rethink: conservative reduction and move to uncertain zone
            decision.confidence = max(0.1, min(0.7, decision.confidence * 0.9))
            decision.reason = f"fallback_rethink_attempt_{attempt}"
            return decision

        prompt = (
            "你是ReAct复核模型。请根据审核结论和审计意见重新判断置信度与理由，"
            "只输出：confidence=<0~1>; reason=<一句话>\n"
            f"decision={decision.model_dump_json()}\n"
            f"metadata={metadata}\n"
        )
        response = self.react_llm.invoke([HumanMessage(content=prompt)])
        content = str(getattr(response, "content", ""))
        confidence = decision.confidence
        reason = decision.reason
        for part in content.split(";"):
            part = part.strip()
            if part.startswith("confidence="):
                try:
                    confidence = float(part.split("=", 1)[1].strip())
                except ValueError:
                    pass
            if part.startswith("reason="):
                reason = part.split("=", 1)[1].strip()
        decision.confidence = max(0.0, min(1.0, confidence))
        decision.reason = reason
        return decision
