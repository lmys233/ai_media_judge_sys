"""CoT 证据辅助者模块

证据辅助者使用 Chain-of-Thought 模式分析决策者引用的证据：
- 检查证据是否充分支持结论
- 识别缺失或薄弱的证据
- 指出证据与结论的不匹配
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.models.contracts import AuditDecision
from src.reasoning.multi_agent.contracts import AssistantObjection, ThinkingChain

logger = logging.getLogger(__name__)


class CoTEvidenceAssistant:
    """CoT 证据辅助者

    职责：
    - 分析决策者引用的证据是否充分支持结论
    - 检查证据与结论的相关性
    - 发现证据不足时提出质疑
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm

    def analyze(
        self,
        thinking_chain: list[ThinkingChain],
        key_evidence: list[str],
        original_decision: AuditDecision,
        context: dict[str, Any],
    ) -> AssistantObjection:
        """分析决策者引用的证据是否充分

        Args:
            thinking_chain: 决策者的推理链
            key_evidence: 决策者引用的关键证据列表
            original_decision: 原始审核决策
            context: 案例上下文

        Returns:
            AssistantObjection，包含质疑点（如果有）
        """
        if self.llm is None:
            return self._fallback_analyze(original_decision, key_evidence)

        try:
            prompt = self._build_analysis_prompt(
                thinking_chain, key_evidence, original_decision, context
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = str(getattr(response, "content", "")).strip()
            return self._parse_objection_response(content)
        except Exception as e:
            logger.error("证据辅助者 LLM 调用失败: %s", e)
            return AssistantObjection(
                assistant_type="evidence",
                has_objection=False,
                objection_points=[],
                confidence_impact=0.0,
                reasoning_chain=f"LLM 调用异常: {str(e)[:50]}",
            )

    def _build_analysis_prompt(
        self,
        thinking_chain: list[ThinkingChain],
        key_evidence: list[str],
        original_decision: AuditDecision,
        context: dict[str, Any],
    ) -> str:
        """构建分析提示词"""
        chain_text = "\n".join([
            f"步骤 {step.step}: 结论={step.conclusion}"
            for step in thinking_chain
        ])
        evidence_text = "\n".join([f"- {e}" for e in key_evidence]) if key_evidence else "无明确证据"

        return (
            "你是证据审计专家。分析决策者引用的证据是否充分支持其结论。\n\n"
            f"【推理链结论】\n{chain_text}\n\n"
            f"【引用的证据】\n{evidence_text}\n\n"
            f"【原始结论】\n类型: {original_decision.final_label.value}, 置信度: {original_decision.confidence:.2f}\n\n"
            "【质疑标准】\n"
            "- 证据是否支持结论的每个关键点？\n"
            "- 是否存在关键证据缺失？\n"
            "- 证据是否与结论类型匹配？（如色情内容需要视觉证据）\n"
            "- 证据描述是否具体、明确？\n\n"
            "【输出格式】\n"
            "{\n"
            '  "has_objection": true/false,\n'
            '  "objection_points": ["质疑点1", "质疑点2"],\n'
            '  "confidence_impact": -0.1 到 0.1 之间的值,\n'
            '  "reasoning_chain": "详细说明质疑理由"\n'
            "}\n\n"
            "【注意】\n"
            "- 如果证据充分支持结论，返回 has_objection: false\n"
            "- confidence_impact 表示该质疑可能导致置信度变化的估计值\n"
            "- 即使证据列表为空，也要判断是否需要质疑（可能说明决策过于武断）\n"
        )

    def _parse_objection_response(self, content: str) -> AssistantObjection:
        """解析 LLM 响应"""
        try:
            # 提取 JSON
            json_str = content
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()

            payload = json.loads(json_str)

            return AssistantObjection(
                assistant_type="evidence",
                has_objection=bool(payload.get("has_objection", False)),
                objection_points=payload.get("objection_points", []),
                confidence_impact=float(payload.get("confidence_impact", 0.0)),
                reasoning_chain=str(payload.get("reasoning_chain", "")),
            )

        except json.JSONDecodeError as e:
            logger.error("证据辅助者 JSON 解析失败: %s", e)
            return AssistantObjection(
                assistant_type="evidence",
                has_objection=False,
                objection_points=[],
                confidence_impact=0.0,
                reasoning_chain=f"JSON 解析失败: {str(e)[:50]}",
            )

    def _fallback_analyze(
        self,
        original_decision: AuditDecision,
        key_evidence: list[str],
    ) -> AssistantObjection:
        """降级分析：当 LLM 不可用时"""
        # 简单的启发式检查
        if not key_evidence and original_decision.confidence > 0.7:
            return AssistantObjection(
                assistant_type="evidence",
                has_objection=True,
                objection_points=["高置信度但无明确证据支持"],
                confidence_impact=-0.1,
                reasoning_chain="启发式检查：高置信度+无证据组合存在风险",
            )
        return AssistantObjection(
            assistant_type="evidence",
            has_objection=False,
            objection_points=[],
            confidence_impact=0.0,
            reasoning_chain="LLM 不可用，使用启发式检查通过",
        )
