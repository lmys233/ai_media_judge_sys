from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.models.contracts import AuditDecision, ViolationDetail, ViolationType
from src.models.llm_factory import create_judge_llm

logger = logging.getLogger(__name__)


class JudgeLLM:
    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm

    def judge(
        self,
        task_id: str,
        trace_id: str,
        metadata: dict[str, Any],
        prelabel_summary: str,
        retrieved_cases: list[dict[str, Any]],
    ) -> AuditDecision:
        if self.llm is None:
            return self._fallback_decision(task_id, trace_id, metadata, prelabel_summary)

        evidence = [
            {
                "violation_type": row.get("violation_type"),
                "risk_score": row.get("risk_score"),
                "description": row.get("description", "")[:200],
                "human_verified": row.get("human_verified", False),
            }
            for row in retrieved_cases[:5]
        ]

        # Build retrieved cases context (include review_reason from full data if available)
        case_contexts = []
        for row in retrieved_cases[:5]:
            full_data = row.get("_full_data", {})
            review_reason = full_data.get("reviewReason") or full_data.get("review_reason") or ""
            ai_result = full_data.get("aiResult") or {}
            ai_reason = ai_result.get("reason", "") if isinstance(ai_result, dict) else ""
            desc = row.get("description", "")[:100]
            context = f"- 类型:{row.get('violation_type')}, 风险:{row.get('risk_score')}, 描述:{desc}"
            if review_reason:
                context += f", 审核理由:{review_reason[:50]}"
            if ai_reason:
                context += f", AI理由:{ai_reason[:50]}"
            case_contexts.append(context)
        retrieved_context = "\n".join(case_contexts)

        # Get NB pre-detected violation types from metadata
        nb_candidates = metadata.get("violation_candidates", [])
        nb_candidates_str = ", ".join(nb_candidates) if nb_candidates else "未知"

        prompt = (
            "你是内容审核专家。给定一段文本和检索到的相似案例，你需要：\n"
            "1. 分析原文中的所有违规内容\n"
            "2. 识别出所有违规类型（可能有多种）\n"
            "3. 为每种违规类型提供置信度和证据\n\n"
            "输出JSON格式：\n"
            "{\n"
            '  "final_label": "主违规类型（normal/abuse/violence/porn/politics/other）",\n'
            '  "confidence": 0.0-1.0的主置信度,\n'
            '  "reason": "一句话简述判定理由",\n'
            '  "violation_details": [\n'
            '    {"violation_type": "abuse/violence/porn/politics/other", "confidence": 0.0-1.0, "evidence": ["证据1", "证据2"], "reason": "该类型的判定理由"},\n'
            "  ]\n"
            "}\n\n"
            "重要：\n"
            "- 必须输出所有检测到的违规类型，禁止只输出单一类型\n"
            "- 如果NB预训练模型已检测到多种违规类型（如 " + nb_candidates_str + "），必须全部在violation_details中输出\n"
            "- violation_details数组必须包含所有检测到的违规类型，每种类型一个对象\n"
            "- evidence字段列出支持该违规判定的前3个关键证据/关键词\n"
            "- 如果文本正常，violation_details为空数组\n"
            "- 检索案例中的'审核理由'来自历史审核记录，包含了人工或AI的判定依据，请参考这些理由进行判断\n\n"
            f"原文内容：{prelabel_summary}\n\n"
            f"检索到的相似案例（含历史审核理由）：\n{retrieved_context}\n\n"
            f"元数据：{json.dumps(metadata, ensure_ascii=False)}\n"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", "")
        return self._parse_llm_json(task_id, trace_id, content, retrieved_cases[:5])

    def _parse_llm_json(
        self,
        task_id: str,
        trace_id: str,
        content: str,
        retrieved_cases: list[dict[str, Any]],
    ) -> AuditDecision:
        try:
            payload = json.loads(content)
            label = ViolationType(payload["final_label"])
            confidence = max(0.0, min(1.0, float(payload["confidence"])))

            # Parse violation details
            violation_details: list[ViolationDetail] = []
            for v in payload.get("violation_details", []):
                try:
                    vd = ViolationDetail(
                        violation_type=ViolationType(v["violation_type"]),
                        confidence=max(0.0, min(1.0, float(v["confidence"]))),
                        evidence=v.get("evidence", [])[:5],
                        reason=str(v.get("reason", ""))[:100],
                    )
                    violation_details.append(vd)
                except (ValueError, KeyError):
                    continue

            # Log retrieved cases for debugging
            logger.info(
                "向量检索匹配案例: task_id=%s, cases=%d, labels=%s",
                task_id,
                len(retrieved_cases),
                [c.get("violation_type") for c in retrieved_cases],
            )

            return AuditDecision(
                trace_id=trace_id,
                task_id=task_id,
                final_label=label,
                confidence=confidence,
                reason=str(payload.get("reason", "")),
                decision_path=["judge_llm"],
                violation_details=violation_details,
            )
        except Exception:  # noqa: BLE001
            return AuditDecision(
                trace_id=trace_id,
                task_id=task_id,
                final_label=ViolationType.OTHER,
                confidence=0.5,
                reason="judge_llm_parse_failed",
                decision_path=["judge_llm", "parse_failed"],
            )

    def _fallback_decision(
        self,
        task_id: str,
        trace_id: str,
        metadata: dict[str, Any],
        prelabel_summary: str,
    ) -> AuditDecision:
        risk = float(metadata.get("risk_score_pre", 0.0))
        candidates = metadata.get("violation_candidates", [])
        label = ViolationType(candidates[0]) if candidates else ViolationType.NORMAL

        # Build violation details from candidates
        violation_details: list[ViolationDetail] = []
        for vtype in candidates:
            try:
                v = ViolationType(vtype)
                if v != ViolationType.NORMAL:
                    violation_details.append(ViolationDetail(
                        violation_type=v,
                        confidence=risk,
                        evidence=metadata.get("evidence", [])[:3],
                        reason=f"fallback_keyword_match: {vtype}",
                    ))
            except ValueError:
                continue

        reason = f"fallback_by_prelabel: {prelabel_summary[:120]}"
        return AuditDecision(
            trace_id=trace_id,
            task_id=task_id,
            final_label=label,
            confidence=max(0.0, min(1.0, risk)),
            reason=reason,
            decision_path=["judge_fallback"],
            violation_details=violation_details,
        )


def build_default_judge_llm() -> BaseChatModel | None:
    return create_judge_llm()
