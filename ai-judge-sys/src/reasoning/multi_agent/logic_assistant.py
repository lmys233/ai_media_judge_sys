"""CoT 逻辑辅助者模块

逻辑辅助者使用 Chain-of-Thought 模式分析决策者的推理链：
- 检测逻辑漏洞
- 识别推理矛盾
- 指出未支持的结论
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


class CoTLogicAssistant:
    """CoT 逻辑辅助者

    职责：
    - 分析决策者的 thinking chain 是否存在逻辑漏洞
    - 识别推理中的矛盾或缺口
    - 发现错误时提出质疑
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm

    def analyze(
        self,
        thinking_chain: list[ThinkingChain],
        original_decision: AuditDecision,
        context: dict[str, Any],
    ) -> AssistantObjection:
        """分析决策者推理链的逻辑漏洞

        Args:
            thinking_chain: 决策者的推理链
            original_decision: 原始审核决策
            context: 案例上下文

        Returns:
            AssistantObjection，包含质疑点（如果有）
        """
        if self.llm is None:
            return self._fallback_analyze(original_decision)

        try:
            prompt = self._build_analysis_prompt(thinking_chain, original_decision, context)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = str(getattr(response, "content", "")).strip()
            return self._parse_objection_response(content)
        except Exception as e:
            logger.error("逻辑辅助者 LLM 调用失败: %s", e)
            return AssistantObjection(
                assistant_type="logic",
                has_objection=False,
                objection_points=[],
                confidence_impact=0.0,
                reasoning_chain=f"LLM 调用异常: {str(e)[:50]}",
            )

    def _build_analysis_prompt(
        self,
        thinking_chain: list[ThinkingChain],
        original_decision: AuditDecision,
        context: dict[str, Any],
    ) -> str:
        """构建分析提示词"""
        chain_text = "\n".join([
            f"步骤 {step.step}: 观察={step.observation}, 推理={step.reasoning}, "
            f"行动={step.action}, 结论={step.conclusion}"
            for step in thinking_chain
        ])

        return (
            "你是逻辑审计专家。分析决策者的推理链是否存在逻辑漏洞。\n\n"
            f"【推理链】\n{chain_text}\n\n"
            f"【原始结论】\n类型: {original_decision.final_label.value}, 置信度: {original_decision.confidence:.2f}\n"
            f"理由: {original_decision.reason}\n\n"
            "【质疑标准】\n"
            "- 推理是否存在矛盾？\n"
            "- 观察是否支持结论？\n"
            "- 是否存在逻辑跳跃？\n"
            "- 证据是否充分支持推理？\n\n"
            "【输出格式】\n"
            "{\n"
            '  "has_objection": true/false,\n'
            '  "objection_points": ["质疑点1", "质疑点2"],\n'
            '  "confidence_impact": -0.1 到 0.1 之间的值,\n'
            '  "reasoning_chain": "详细说明质疑理由"\n'
            "}\n\n"
            "【注意】\n"
            "- 如果推理链逻辑严密、无明显漏洞，返回 has_objection: false\n"
            "- confidence_impact 表示该质疑可能导致置信度变化的估计值\n"
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
                assistant_type="logic",
                has_objection=bool(payload.get("has_objection", False)),
                objection_points=payload.get("objection_points", []),
                confidence_impact=float(payload.get("confidence_impact", 0.0)),
                reasoning_chain=str(payload.get("reasoning_chain", "")),
            )

        except json.JSONDecodeError as e:
            logger.error("逻辑辅助者 JSON 解析失败: %s", e)
            return AssistantObjection(
                assistant_type="logic",
                has_objection=False,
                objection_points=[],
                confidence_impact=0.0,
                reasoning_chain=f"JSON 解析失败: {str(e)[:50]}",
            )

    def _fallback_analyze(self, original_decision: AuditDecision) -> AssistantObjection:
        """降级分析：当 LLM 不可用时"""
        # 简单的启发式检查
        if original_decision.confidence < 0.3 and original_decision.final_label.value == "normal":
            return AssistantObjection(
                assistant_type="logic",
                has_objection=True,
                objection_points=["低置信度但判定为正常，可能存在漏检风险"],
                confidence_impact=-0.1,
                reasoning_chain="启发式检查：低置信度+正常标签组合存在风险",
            )
        return AssistantObjection(
            assistant_type="logic",
            has_objection=False,
            objection_points=[],
            confidence_impact=0.0,
            reasoning_chain="LLM 不可用，使用启发式检查通过",
        )
