"""ReAct 决策者智能体模块

决策者智能体使用 ReAct 模式进行审核复核：
- 接收原始审核结论和案例上下文
- 输出结构化的 thinking chain
- 接收辅助者质疑后可以 rerethink
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.models.contracts import AuditDecision
from src.reasoning.multi_agent.contracts import AssistantObjection, ThinkingChain

logger = logging.getLogger(__name__)


@dataclass
class DecisionMakerOutput:
    """决策者输出"""

    thinking_chain: list[ThinkingChain]
    decision: AuditDecision
    confidence: float
    key_evidence: list[str]
    reasoning_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "thinking_chain": [t.to_dict() for t in self.thinking_chain],
            "decision": self.decision.model_dump() if hasattr(self.decision, "model_dump") else str(self.decision),
            "confidence": self.confidence,
            "key_evidence": self.key_evidence,
            "reasoning_summary": self.reasoning_summary,
        }


class ReActDecisionMaker:
    """ReAct 决策者智能体

    职责：
    - 分析原始审核结论和案例上下文
    - 输出 thinking chain 供辅助者分析
    - 根据辅助者质疑进行 rerethink
    """

    def __init__(self, llm: BaseChatModel | None = None, max_retries: int = 3) -> None:
        self.llm = llm
        self.max_retries = max_retries

    def decide(
        self,
        original_decision: AuditDecision,
        context: dict[str, Any],
        assistant_objections: list[AssistantObjection] | None = None,
        round_num: int = 1,
    ) -> DecisionMakerOutput:
        """使用 ReAct 模式进行决策

        Args:
            original_decision: 原始审核决策
            context: 案例上下文（检索到的案例、元数据等）
            assistant_objections: 辅助者之前的质疑（用于 rerethink）
            round_num: 当前复核轮次

        Returns:
            DecisionMakerOutput，包含 thinking chain 和决策
        """
        if self.llm is None:
            return self._fallback_decide(original_decision, context, round_num)

        try:
            prompt = self._build_decision_prompt(
                original_decision, context, assistant_objections, round_num
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = str(getattr(response, "content", "")).strip()
            return self._parse_decision_response(content, original_decision)
        except Exception as e:
            logger.error("决策者 LLM 调用失败: %s", e)
            return self._fallback_decide(original_decision, context, round_num)

    def _build_decision_prompt(
        self,
        original_decision: AuditDecision,
        context: dict[str, Any],
        objections: list[AssistantObjection] | None = None,
        round_num: int = 1,
    ) -> str:
        """构建决策者提示词"""
        context_json = json.dumps(context, ensure_ascii=False, indent=2)

        base_prompt = (
            "你是内容审核决策专家。基于原始审核结论和案例上下文，用 ReAct 模式进行复核决策。\n\n"
            "【输入】\n"
            f"原始审核结论: final_label={original_decision.final_label.value}, "
            f"confidence={original_decision.confidence:.2f}, reason={original_decision.reason}\n"
            f"案例上下文: {context_json}\n\n"
        )

        if objections and any(obj.has_objection for obj in objections):
            objections_text = "\n".join([
                f"- [{obj.assistant_type}] {obj.reasoning_chain}"
                for obj in objections if obj.has_objection
            ])
            base_prompt += (
                f"【辅助者提出的质疑】\n{objections_text}\n\n"
                "请根据质疑重新思考，输出更新后的结论。\n\n"
            )

        base_prompt += (
            "【输出要求】\n"
            "你必须输出包含 thinking_chain 的完整 JSON 响应：\n\n"
            "{\n"
            '  "thinking_chain": [\n'
            '    {"step": 1, "observation": "观察到什么", "reasoning": "如何推理", "action": "采取什么行动", "conclusion": "得出什么结论"},\n'
            '    {"step": 2, "observation": "...", "reasoning": "...", "action": "...", "conclusion": "..."},\n'
            '    {"step": 3, ...}\n'
            "  ],\n"
            '  "decision": {"final_label": "normal/abuse/violence/porn/politics/other", "confidence": 0.0-1.0, "reason": "一句话判定理由"},\n'
            '  "key_evidence": ["关键证据1", "关键证据2"],\n'
            '  "reasoning_summary": "一句话总结推理过程"\n'
            "}\n\n"
            "【注意事项】\n"
            "- thinking_chain 必须包含至少 2 个推理步骤\n"
            "- 每个步骤的 conclusion 应环环相扣\n"
            "- key_evidence 列出支持结论的最重要的 2-3 个证据\n"
            "- confidence 不宜超过 0.90（除非图片/内容极其明显）\n"
        )
        return base_prompt

    def _parse_decision_response(
        self,
        response_content: str,
        original_decision: AuditDecision,
    ) -> DecisionMakerOutput:
        """解析 LLM 响应"""
        try:
            # 尝试提取 JSON（可能包含在 markdown 代码块中）
            json_str = response_content

            # 检查是否在 ```json ... ``` 块中
            if "```json" in response_content:
                start = response_content.find("```json") + 7
                end = response_content.find("```", start)
                json_str = response_content[start:end].strip()
            elif "```" in response_content:
                start = response_content.find("```") + 3
                end = response_content.find("```", start)
                json_str = response_content[start:end].strip()

            payload = json.loads(json_str)

            # 解析 thinking_chain
            thinking_chain = []
            for step_data in payload.get("thinking_chain", []):
                thinking_chain.append(ThinkingChain(
                    step=int(step_data.get("step", 0)),
                    observation=str(step_data.get("observation", "")),
                    reasoning=str(step_data.get("reasoning", "")),
                    action=str(step_data.get("action", "")),
                    conclusion=str(step_data.get("conclusion", "")),
                ))

            # 解析 decision
            decision_data = payload.get("decision", {})
            from src.models.contracts import ViolationType

            try:
                final_label = ViolationType(decision_data.get("final_label", "other"))
            except ValueError:
                final_label = ViolationType.OTHER

            # 创建 AuditDecision
            new_decision = AuditDecision(
                trace_id=original_decision.trace_id,
                task_id=original_decision.task_id,
                final_label=final_label,
                confidence=float(decision_data.get("confidence", original_decision.confidence)),
                status=original_decision.status,
                reason=str(decision_data.get("reason", original_decision.reason)),
                decision_path=original_decision.decision_path.copy(),
                needs_manual=original_decision.needs_manual,
                violation_details=original_decision.violation_details.copy(),
            )

            return DecisionMakerOutput(
                thinking_chain=thinking_chain,
                decision=new_decision,
                confidence=float(decision_data.get("confidence", original_decision.confidence)),
                key_evidence=payload.get("key_evidence", []),
                reasoning_summary=payload.get("reasoning_summary", ""),
            )

        except json.JSONDecodeError as e:
            logger.error("决策者 JSON 解析失败: %s, content=%s", e, response_content[:200])
            return self._fallback_output(original_decision)

    def _fallback_decide(
        self,
        original_decision: AuditDecision,
        context: dict[str, Any],
        round_num: int,
    ) -> DecisionMakerOutput:
        """降级决策：当 LLM 不可用时"""
        # 保守地降低置信度
        new_confidence = max(0.1, min(0.7, original_decision.confidence * 0.9))

        fallback_chain = [
            ThinkingChain(
                step=1,
                observation=f"原始置信度为 {original_decision.confidence:.2f}",
                reasoning="置信度处于中等区间，需要进一步分析",
                action="进行保守估计",
                conclusion=f"置信度调整至 {new_confidence:.2f}",
            ),
            ThinkingChain(
                step=2,
                observation="LLM 不可用或调用失败",
                reasoning="使用降级策略",
                action="保持原有结论但降低置信度",
                conclusion=f"最终置信度 {new_confidence:.2f}",
            ),
        ]

        fallback_decision = AuditDecision(
            trace_id=original_decision.trace_id,
            task_id=original_decision.task_id,
            final_label=original_decision.final_label,
            confidence=new_confidence,
            status=original_decision.status,
            reason=f"fallback_rerethink_round_{round_num}",
            decision_path=original_decision.decision_path + [f"fallback_rerethink_{round_num}"],
            needs_manual=original_decision.needs_manual,
            violation_details=original_decision.violation_details,
        )

        return DecisionMakerOutput(
            thinking_chain=fallback_chain,
            decision=fallback_decision,
            confidence=new_confidence,
            key_evidence=[],
            reasoning_summary="使用降级策略，置信度保守下调",
        )

    def _fallback_output(self, original_decision: AuditDecision) -> DecisionMakerOutput:
        """解析失败时的降级输出"""
        return DecisionMakerOutput(
            thinking_chain=[
                ThinkingChain(
                    step=1,
                    observation="JSON 解析失败",
                    reasoning="使用原始决策",
                    action="保持原决策",
                    conclusion=f"保持原置信度 {original_decision.confidence:.2f}",
                ),
            ],
            decision=original_decision,
            confidence=original_decision.confidence,
            key_evidence=[],
            reasoning_summary="JSON 解析失败，保持原决策",
        )
