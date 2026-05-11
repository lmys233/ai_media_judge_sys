"""图片最终审核判断模块

使用通义千问 VL（Qwen-VL）结合召回的相似案例，对图片进行最终审核。
返回结构化的审核结果。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# VL 审核判断提示词
_IMAGE_JUDGE_PROMPT = """你是一个内容审核专家。给定一张图片和检索到的相似审核案例，你需要：

1. 分析图片中的所有内容
2. 识别出所有违规内容
3. 为每种违规类型提供置信度和证据

【输入格式】
- 图片：[Base64编码的图片]
- 相似案例：详见下方案例列表

【案例参考】
{cases_context}

【输出要求】
请以 JSON 格式输出审核结果：

{{
  "final_label": "主违规类型（normal/abuse/violence/porn/politics/other）",
  "confidence": 0.0-1.0 的置信度分数,
  "reason": "简短判定理由",
  "violation_details": [
    {{
      "violation_type": "违规类型",
      "confidence": 0.0-1.0,
      "evidence": ["证据1", "证据2"],
      "reason": "该类型的判定理由"
    }}
  ]
}}

【注意事项】
- 如果图片正常，final_label 设为 "normal"
- 重点关注案例中标注的违规类型
- 置信度结合案例相似度和图片内容判断
- 证据应基于图片实际内容
- 除非图片非常明显，否则置信度不宜超过 0.90
"""


class ImageJudgeLLM:
    """基于 VL 模型的图片审核判断器

    结合召回的相似案例，对图片进行多模态审核。
    返回结构化的审核决策。
    """

    def __init__(self, vl_llm: Any | None = None) -> None:
        """初始化图片审核判断器

        Args:
            vl_llm: 可选的 VL 大模型实例。如果不提供，将使用 LLM 工厂创建。
        """
        self._vl_llm = vl_llm

    def _get_vl_llm(self) -> Any:
        """获取 VL 大模型实例"""
        if self._vl_llm is not None:
            return self._vl_llm

        try:
            from langchain_openai import ChatOpenAI

            # 通义千问 VL 配置
            return ChatOpenAI(
                model=os.getenv("VL_MODEL", "qwen-vl-max"),
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=60,
            )
        except Exception as e:
            logger.error("VL 模型初始化失败: %s", e)
            return None

    def _build_cases_context(self, retrieved_cases: list[dict]) -> str:
        """构建案例上下文

        将召回的相似案例格式化为提示词上下文。

        Args:
            retrieved_cases: 召回的案例列表

        Returns:
            格式化的案例上下文字符串
        """
        if not retrieved_cases:
            return "无相似案例供参考"

        cases_text = []
        for i, case in enumerate(retrieved_cases[:5], 1):  # 最多使用5个案例
            violation = case.get("violation_type", "unknown")
            risk = case.get("risk_score", 0.0)
            description = case.get("description", "")
            evidence = case.get("evidence", "")
            human_verified = case.get("human_verified", False)
            source = "人工审核" if human_verified else "AI审核"

            case_text = f"""案例{i}:
  - 违规类型: {violation}
  - 风险分数: {risk:.2f}
  - 来源: {source}
  - 描述: {description[:200]}{"..." if len(description) > 200 else ""}"""

            # 如果有 evidence 风格的详细描述，添加到上下文中
            if evidence:
                case_text += f"\n  - 详细证据:\n    {evidence[:500]}{"..." if len(evidence) > 500 else ""}"

            cases_text.append(case_text)

        return "\n\n".join(cases_text)

    def judge_with_image(
        self,
        task_id: str,
        trace_id: str,
        image_base64: str,
        retrieved_cases: list[dict],
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """使用 VL 模型进行图片审核判断

        Args:
            task_id: 任务 ID
            trace_id: 追踪 ID
            image_base64: Base64 编码的图片
            retrieved_cases: 召回的相似案例
            metadata: 可选的元数据

        Returns:
            审核决策字典，包含 final_label, confidence, reason, violation_details
        """
        vl_llm = self._get_vl_llm()
        if vl_llm is None:
            logger.error("VL 模型不可用，无法进行图片审核")
            return {
                "final_label": "other",
                "confidence": 0.0,
                "reason": "VL模型不可用",
                "violation_details": [],
                "error": "vl_model_unavailable",
            }

        try:
            from langchain_core.messages import HumanMessage

            # 构建案例上下文
            cases_context = self._build_cases_context(retrieved_cases)

            # 构建完整提示词
            prompt = _IMAGE_JUDGE_PROMPT.format(cases_context=cases_context)

            # 调用 VL 模型
            response = vl_llm.invoke([
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ])
            ])

            # 解析 JSON 响应
            content = str(getattr(response, "content", "")).strip()
            return self._parse_judge_response(content, retrieved_cases)

        except Exception as e:
            logger.error("VL 图片审核失败: task_id=%s, error=%s", task_id, e)
            return {
                "final_label": "other",
                "confidence": 0.0,
                "reason": f"审核失败: {str(e)[:50]}",
                "violation_details": [],
                "error": str(e),
            }

    def _parse_judge_response(
        self,
        content: str,
        retrieved_cases: list[dict],
    ) -> dict[str, Any]:
        """解析 VL 模型的 JSON 响应

        Args:
            content: VL 模型返回的原始内容
            retrieved_cases: 召回的案例（用于置信度调整）

        Returns:
            解析后的审核决策字典
        """
        try:
            # 尝试提取 JSON（可能包含在 markdown 代码块中）
            json_str = content

            # 检查是否在 ```json ... ``` 块中
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()

            # 解析 JSON
            result = json.loads(json_str)

            # 验证必需字段
            if "final_label" not in result:
                result["final_label"] = "other"

            # 添加召回案例信息供参考
            if retrieved_cases and "retrieved_count" not in result:
                result["retrieved_count"] = len(retrieved_cases)

            logger.info(
                "VL 审核完成: final_label=%s, confidence=%.2f",
                result.get("final_label"),
                result.get("confidence", 0.0),
            )

            return result

        except json.JSONDecodeError as e:
            logger.error("JSON 解析失败: %s, content=%s", e, content[:200])
            return {
                "final_label": "other",
                "confidence": 0.0,
                "reason": "审核结果解析失败",
                "violation_details": [],
                "error": "json_parse_failed",
            }


# 全局实例
_judge_llm: ImageJudgeLLM | None = None


def get_image_judge_llm() -> ImageJudgeLLM:
    """获取全局图片审核 LLM 实例"""
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ImageJudgeLLM()
    return _judge_llm
