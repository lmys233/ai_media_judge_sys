from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM_PROMPT = """\
你是一个严肃的文本语义重写助手。你的任务是将包含违规内容的原始文本，重写为语义等价但不包含任何违禁词（脏话、暴力、色情、政治敏感）的新文本。

【核心原则】
1. 语义等价：重写后的文本必须保留原文的核心含义和语义，不得改变原文的实际意图
2. 语义去污：使用中性、正式、规范的语言描述原本的违规内容，使其不包含明显的脏话或敏感词
3. 不得新增内容：只能对原文进行改写，不得添加原文中不存在的新概念或信息
4. 多类型处理：如果文本包含多种违规内容（辱骂+暴力、暴力+色情等），需要将所有违规内容都进行语义去污

【违规类型与语义去污指导】
- 辱骂/脏话：将脏话改写为对其行为的客观描述，如"骂人"→"言语攻击"或"进行侮辱"
- 暴力威胁：将暴力描述改写为对暴力行为的客观陈述，如"我要杀了你"→"威胁要伤害他人"
- 色情内容：将性暗示改写为中性描述，如"裸聊"→"进行不适当的视频交流"
- 政治敏感：去除具体敏感表述，使用中性概括，如具体的人名/事件→"某些敏感议题"

【示例】
原文："我囚禁，虐待了他"
正确重写："我限制了他的人身自由，并对他施加了暴力行为"

原文："你这个傻逼垃圾，滚回家吃屎去吧"
正确重写："你这个人表现很差，请回家反思"

原文："约人去酒店裸聊做爱"
正确重写："约人到酒店进行不当的私密交流活动"

原文："我要杀了你，你这个垃圾"（暴力+辱骂）
正确重写："此人威胁要伤害他人，并对其进行了言语攻击"

【输出格式】
- 只输出重写后的文本
- 不得包含任何解释、注释、前缀或后缀
- 严禁以JSON或其他结构化格式输出
- 如果原文本身不包含任何违规内容，则保持原样输出

【严禁行为】
- 绝对不得生成任何脏话、骂人词语或谐音变体
- 绝对不得生成具体的暴力实施细节（如杀人方法）
- 绝对不得生成色情身体部位或性行为的具体描述
- 不得创造新的违禁词汇或表达方式
- 不得改变原文的褒贬倾向（负面还是负面）
"""


class DesensitizeProcessor:
    def __init__(self, llm: Any | None = None) -> None:
        self.overrides_path = Path(__file__).resolve().parent.parent / "policy" / "text_keyword_overrides.yaml"
        self._mtime = 0.0
        self.pattern_map = self._build_pattern_map()
        self._llm = llm  # Use shared LLM instance

    def _build_pattern_map(self) -> list[tuple[re.Pattern, str]]:
        defaults = {
            "abuse": ["操你妈", "妈的", "傻逼", "垃圾", "滚开"],
            "violence": ["杀了你", "打死你", "砍人", "爆头"],
            "porn": ["约炮", "裸体", "色情", "成人视频"],
            "politics": [],
        }
        if not self.overrides_path.exists():
            return self._patterns_from_terms(defaults)
        try:
            payload = yaml.safe_load(self.overrides_path.read_text(encoding="utf-8")) or {}
            hard = payload.get("hard_keywords", {})
            soft = payload.get("soft_keywords", {})
            legacy = payload.get("keyword_overrides", {})
            merged: dict[str, list[str]] = {k: list(v) for k, v in defaults.items()}
            for category in merged:
                values = []
                if isinstance(legacy, dict):
                    values += legacy.get(category, []) if isinstance(legacy.get(category, []), list) else []
                if isinstance(soft, dict):
                    values += soft.get(category, []) if isinstance(soft.get(category, []), list) else []
                if isinstance(hard, dict):
                    values += hard.get(category, []) if isinstance(hard.get(category, []), list) else []
                merged[category] = list(dict.fromkeys(merged[category] + [str(v) for v in values if str(v).strip()]))
        except Exception:  # noqa: BLE001
            return self._patterns_from_terms(defaults)
        return self._patterns_from_terms(merged)

    def _patterns_from_terms(self, terms_by_category: dict[str, list[str]]) -> list[tuple[re.Pattern, str]]:
        replace_map = {
            "abuse": "[违规言语]",
            "violence": "[暴力威胁]",
            "porn": "[不当性暗示]",
            "politics": "[敏感议题]",
        }
        pattern_map: list[tuple[re.Pattern, str]] = []
        for category, replacement in replace_map.items():
            terms = terms_by_category.get(category, [])
            if not terms:
                continue
            # Replace longer phrases first to avoid shorter term partial matches.
            unique_terms = sorted({str(t).strip() for t in terms if str(t).strip()}, key=len, reverse=True)
            escaped = [re.escape(t) for t in unique_terms]
            pattern_map.append((re.compile("(" + "|".join(escaped) + ")", re.IGNORECASE), replacement))
        return pattern_map

    def sanitize(self, text: str) -> str:
        """Rule-based keyword replacement (fast, no API call)."""
        try:
            mtime = self.overrides_path.stat().st_mtime if self.overrides_path.exists() else 0.0
            if mtime > self._mtime:
                self.pattern_map = self._build_pattern_map()
                self._mtime = mtime
        except Exception:  # noqa: BLE001
            pass
        sanitized = text
        for pattern, replacement in self.pattern_map:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    def sanitize_for_rag(self, text: str, violation_labels: list[str] | None = None) -> str:
        """LLM-based semantic rewrite for RAG vector storage.

        Falls back to rule-based sanitize when LLM is unavailable.
        """
        if not text.strip():
            return text

        llm = self._llm if self._llm is not None else self._build_llm()
        if llm is None:
            logger.info("去敏感化LLM不可用，降级为规则替换")
            return self.sanitize(text)

        labels_hint = ""
        if violation_labels:
            labels_hint = f"\n该文本的违规标签为: {', '.join(violation_labels)}。"

        user_prompt = f"原始文本：{text}{labels_hint}"

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            result = str(getattr(response, "content", "")).strip()
            if not result:
                logger.warning("去敏感化LLM返回空结果，降级为规则替换")
                return self.sanitize(text)
            return result
        except Exception:  # noqa: BLE001
            logger.exception("去敏感化LLM调用失败，降级为规则替换")
            return self.sanitize(text)

    @staticmethod
    def _build_llm():  # noqa: ANN205
        from src.models.llm_factory import create_llm
        return create_llm(
            provider_env="DESENSITIZE_LLM_PROVIDER",
            api_key_env="DESENSITIZE_LLM_API_KEY",
            model_env="DESENSITIZE_LLM_MODEL",
            base_url_env="DESENSITIZE_LLM_BASE_URL",
            default_temperature=0.3,
        )
