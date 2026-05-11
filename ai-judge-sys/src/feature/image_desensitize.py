"""图片描述生成与脱敏模块

使用通义千问 VL（Qwen-VL）生成图片的场景描述，并进行脱敏处理。
描述结果将存入向量数据库用于后续检索。
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# VL 模型描述提示词（纯描述，不含审核判断）
_IMAGE_DESCRIPTION_PROMPT = """你是一个图像描述助手。请用简洁的中文客观描述这张图片的内容：
1. 图片场景（室内/室外、环境、活动类型等）
2. 主要人物/物体的数量、外貌特征、姿态、动作
3. 场景中的物品、布局等细节

要求：
- 描述要客观、中性，只陈述看到的画面内容
- 不要对内容做任何判断或评价（如"正常""违规"等）
- 不要提及任何违规相关的判断
- 最多3句话，控制在100字以内
"""

# 描述脱敏提示词
_DESCRIPTION_DESENSITIZE_PROMPT = """你是一个文本语义重写助手。你的任务是将包含违规内容的图片描述，重写为语义等价但不包含任何违禁词的新文本。

【核心原则】
1. 语义等价：重写后的文本必须保留原文的核心含义
2. 语义去污：使用中性、正式的语言描述原本的违规内容
3. 不得新增内容：只能对原文进行改写

【违规类型与去污指导】
- 裸露/色情：将具体描述改为"穿着不当"等中性表述
- 暴力：将血腥细节改为"暴力行为"等中性表述
- 辱骂：将脏话改为"言语攻击"等中性表述

【输出格式】
- 只输出重写后的文本
- 不得包含任何解释或注释

示例：
原文："一个女人在镜头前裸露身体"
正确重写："一名女性穿着不当"

原文："两人在街头激烈打斗，血流满面"
正确重写："两人在街头发生肢体冲突"
"""

# Evidence 风格描述提示词（用于存入向量库，为检索提供更具体的细节）
_IMAGE_EVIDENCE_PROMPT = """你是一个图像证据描述专家。请详细分析并描述这张图片中的所有视觉元素，重点关注可能涉及违规的内容细节。

请从以下维度进行详细描述：

1. **穿着打扮**：具体描述衣着款式、覆盖程度、材质效果（如紧身、透视、暴露程度等）
2. **身体姿态**：具体描述身体姿势、暴露的部位、动作的挑逗性程度
3. **人物特征**：描述面部表情、眼神、身材曲线的强调程度
4. **光影效果**：描述打光方式（如高光、阴影、聚光）如何突出身体部位
5. **场景氛围**：描述背景、道具、整体画面的情色暗示程度

要求：
- 描述要极其具体，直接指出身体暴露程度、曲线强调、性暗示等细节
- 不要说"姿态优雅"、"服装装饰"等中性词汇，要直接说"暴露"、"挑逗"、"性感化"
- 不要做判断（如"这是色情内容"），只描述画面
- 列出 3-6 条具体的描述要点
- 输出格式：每行一条描述，以"• "开头

示例输出：
• 角色穿着极度过分暴露的紧身衣物，仅覆盖关键部位，材质轻薄透明
• 身体姿态呈现明显性感化和挑逗性，侧卧并强调身体曲线
• 皮肤表面有高光反射效果，增强视觉上的性暗示
• 整体构图强调身体曲线和私密区域，镜头角度刻意营造挑逗感
• 角色面部表情带有明显的性暗示意味
"""


class ImageDesensitizeProcessor:
    """图片描述生成与脱敏处理器

    使用 VL 模型生成图片描述，然后进行脱敏处理。
    生成的描述用于向量数据库存储和后续检索。
    """

    def __init__(self, vl_llm: Any | None = None) -> None:
        """初始化描述处理器

        Args:
            vl_llm: 可选的 VL 大模型实例。如果不提供，将使用 LLM 工厂创建。
        """
        self._vl_llm = vl_llm
        self._desensitize_llm: Any | None = None

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
                timeout=30,
            )
        except Exception as e:
            logger.error("VL 模型初始化失败: %s", e)
            return None

    def _get_desensitize_llm(self) -> Any:
        """获取脱敏用的大模型实例"""
        if self._desensitize_llm is not None:
            return self._desensitize_llm

        try:
            from src.models.llm_factory import create_llm

            self._desensitize_llm = create_llm(
                provider_env="DESENSITIZE_LLM_PROVIDER",
                api_key_env="DESENSITIZE_LLM_API_KEY",
                model_env="DESENSITIZE_LLM_MODEL",
                base_url_env="DESENSITIZE_LLM_BASE_URL",
                default_temperature=0.3,
            )
            return self._desensitize_llm
        except Exception as e:
            logger.warning("脱敏 LLM 初始化失败，使用规则替换: %s", e)
            return None

    def describe_image(self, image_base64: str) -> str:
        """使用 VL 模型生成图片描述

        Args:
            image_base64: Base64 编码的图片数据（不含 data:image 前缀）

        Returns:
            图片的中文描述文本
        """
        vl_llm = self._get_vl_llm()
        if vl_llm is None:
            logger.warning("VL 模型不可用，返回默认描述")
            return "无法识别图片内容"

        try:
            from langchain_core.messages import HumanMessage

            # 构建消息：图片 + 文本提示
            response = vl_llm.invoke([
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": _IMAGE_DESCRIPTION_PROMPT}
                ])
            ])

            description = str(getattr(response, "content", "")).strip()
            logger.info("VL 图片描述生成成功，长度=%d", len(description))
            return description

        except Exception as e:
            logger.error("VL 图片描述生成失败: %s", e)
            return "无法识别图片内容"

    def describe_for_storage(self, image_base64: str) -> dict[str, str]:
        """生成用于存储的图片描述（包含证据细节）

        生成两条描述：
        - description: 简短的中性描述，用于快速理解图片内容
        - evidence: 详细的证据风格描述，包含关键视觉细节，用于检索和参考

        Args:
            image_base64: Base64 编码的图片数据（不含 data:image 前缀）

        Returns:
            包含 description 和 evidence 的字典
        """
        vl_llm = self._get_vl_llm()
        if vl_llm is None:
            return {
                "description": "无法识别图片内容",
                "evidence": "无法识别图片内容",
            }

        try:
            from langchain_core.messages import HumanMessage

            # 生成证据风格描述
            response = vl_llm.invoke([
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": _IMAGE_EVIDENCE_PROMPT}
                ])
            ])

            evidence = str(getattr(response, "content", "")).strip()
            logger.info("VL Evidence 描述生成成功，长度=%d", len(evidence))

            # 生成简短中性描述（用于快速理解）
            desc_response = vl_llm.invoke([
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": _IMAGE_DESCRIPTION_PROMPT}
                ])
            ])
            description = str(getattr(desc_response, "content", "")).strip()

            return {
                "description": description,
                "evidence": evidence,
            }

        except Exception as e:
            logger.error("VL Evidence 描述生成失败: %s", e)
            return {
                "description": "无法识别图片内容",
                "evidence": "无法识别图片内容",
            }

    def desensitize_description(
        self,
        description: str,
        violation_labels: list[str] | None = None,
    ) -> str:
        """对图片描述进行脱敏处理

        Args:
            description: 原始图片描述
            violation_labels: 可选的违规类型标签列表，用于辅助判断

        Returns:
            脱敏后的描述文本
        """
        if not description.strip():
            return description

        # 检查是否包含明显的违规关键词（规则快速处理）
        sanitized = self._rule_desensitize(description)
        if sanitized == description:
            # 规则处理后没变化，说明没有明显违规词
            return description

        # 有违规词，使用 LLM 进行语义重写
        return self._llm_desensitize(sanitized, violation_labels)

    def _rule_desensitize(self, text: str) -> str:
        """基于规则的快速脱敏

        替换明显的违规词汇为中性表述。

        Args:
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        # 常见的需要脱敏的词汇映射（可扩展）
        replacements = {
            # 色情相关
            "裸露": "穿着不当",
            "裸体": "服装异常",
            "色情": "不当内容",
            "性行为": "不当活动",
            "做爱": "私密行为",
            "裸聊": "不当视频交流",

            # 暴力相关
            "血腥": "暴力场面",
            "爆头": "暴力伤害",
            "砍死": "暴力威胁",
            "杀人": "伤害行为",
            "殴打": "肢体冲突",

            # 辱骂相关
            "傻逼": "表现差",
            "操你妈": "言语攻击",
            "滚": "请离开",
        }

        result = text
        for old, new in replacements.items():
            if old in result:
                result = result.replace(old, new)

        return result

    def _llm_desensitize(
        self,
        text: str,
        violation_labels: list[str] | None = None,
    ) -> str:
        """基于 LLM 的语义脱敏重写

        Args:
            text: 经过规则处理的文本
            violation_labels: 违规类型标签

        Returns:
            重写后的文本
        """
        desensitize_llm = self._get_desensitize_llm()
        if desensitize_llm is None:
            # LLM 不可用，返回规则处理结果
            return text

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            labels_hint = ""
            if violation_labels:
                labels_hint = f"\n该文本的违规标签为: {', '.join(violation_labels)}"

            user_prompt = f"原始描述：{text}{labels_hint}"

            response = desensitize_llm.invoke([
                SystemMessage(content=_DESCRIPTION_DESENSITIZE_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            result = str(getattr(response, "content", "")).strip()
            if result:
                logger.info("LLM 描述脱敏成功")
                return result

            return text

        except Exception as e:
            logger.warning("LLM 脱敏失败，使用规则结果: %s", e)
            return text

    def describe_and_desensitize(
        self,
        image_base64: str,
        violation_labels: list[str] | None = None,
    ) -> str:
        """生成图片描述并进行脱敏（完整流程）

        Args:
            image_base64: Base64 编码的图片
            violation_labels: 违规类型标签

        Returns:
            脱敏后的图片描述
        """
        # 步骤1: 生成描述
        description = self.describe_image(image_base64)

        # 步骤2: 脱敏处理
        desensitized = self.desensitize_description(description, violation_labels)

        logger.info(
            "图片描述生成完成: 原始长度=%d, 脱敏后长度=%d",
            len(description),
            len(desensitized),
        )

        return desensitized


# 全局处理器实例
_processor: ImageDesensitizeProcessor | None = None


def get_image_desensitize_processor() -> ImageDesensitizeProcessor:
    """获取全局图片描述处理器实例"""
    global _processor
    if _processor is None:
        _processor = ImageDesensitizeProcessor()
    return _processor
