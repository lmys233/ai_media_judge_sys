"""基于 CLIP 的图片快速打标模块

使用 CLIP Zero-shot 能力对图片进行多标签违规分类。
替代传统 MobileNet，CLIP 对抽象概念（如暴力、色情）有更好的零样本分类能力。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

from src.models.contracts import ViolationType

logger = logging.getLogger(__name__)


@dataclass
class ImageQuickLabelResult:
    """图片快速打标结果

    Attributes:
        violation_types: 检测到的违规类型列表（可能多个）
        risk_score: 综合风险分数（0.0-1.0）
        evidence: 检测依据列表
        decision_source: 结果来源 ("clip_zero_shot" / "vl_fallback")
    """

    violation_types: list[ViolationType]
    risk_score: float
    evidence: list[str]
    decision_source: str


class CLIPViolationClassifier:
    """基于 CLIP Zero-shot 的违规分类器

    使用预定义的违规类型文本描述，计算图片与各类型的相似度，
    返回多标签分类结果。

    工作原理：
    1. 为每个违规类型定义文本描述
    2. 计算图片与各描述的 CLIP 相似度
    3. 超过阈值的类型标记为违规
    """

    # 各违规类型的文本提示词（用于 CLIP 相似度计算）
    _VIOLATION_PROMPTS = {
        ViolationType.ABUSE: [
            "a person being harassed or bullied",
            "threatening or intimidating behavior",
            "someone being verbally abused",
        ],
        ViolationType.VIOLENCE: [
            "violent scene with blood or injury",
            "weapon such as gun or knife",
            "physical fight or assault",
        ],
        ViolationType.PORN: [
            "explicit sexual content or nudity",
            "adult entertainment or pornography",
            "nude or semi-nude person",
        ],
        ViolationType.POLITICS: [
            "political protest or demonstration",
            "politically sensitive content",
            "election or voting related",
        ],
        ViolationType.OTHER: [
            "suspicious or unusual activity",
            "potentially harmful content",
        ],
    }

    # 相似度阈值：超过此值认为是该违规类型
    _THRESHOLD = 0.25

    # 各类型的风险基础分
    _BASE_SCORES = {
        ViolationType.ABUSE: 0.5,
        ViolationType.VIOLENCE: 0.6,
        ViolationType.PORN: 0.7,
        ViolationType.POLITICS: 0.5,
        ViolationType.OTHER: 0.3,
    }

    def __init__(self) -> None:
        """初始化分类器"""
        import torch
        from src.feature.image_embedder import CLIPImageEmbedder

        self._torch = torch
        self._embedder = CLIPImageEmbedder()

        # 预计算的文本嵌入（tensor 形式，避免重复计算）
        self._text_embeddings: dict[ViolationType, torch.Tensor] = {}
        self._embeddings_ready = False

    def _ensure_embeddings(self) -> None:
        """预计算各违规类型的文本嵌入向量（如果尚未计算）"""
        if self._embeddings_ready:
            return

        if not self._embedder.ready:
            logger.warning("CLIP 编码器未就绪，无法预计算文本嵌入")
            return

        try:
            import clip

            for vtype, prompts in self._VIOLATION_PROMPTS.items():
                # 合并多个提示词为一句
                combined_text = ". ".join(prompts)
                text_tokens = clip.tokenize([combined_text], truncate=True).to(
                    self._embedder.device
                )

                with self._torch.no_grad():
                    emb = self._embedder._model.encode_text(text_tokens)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    self._text_embeddings[vtype] = emb

            self._embeddings_ready = True
            logger.info("CLIP 文本嵌入预计算完成")

        except Exception as e:
            logger.error("文本嵌入预计算失败: %s", e)

    def predict(self, image_path: str) -> ImageQuickLabelResult:
        """对单张图片进行快速打标

        Args:
            image_path: 图片文件路径

        Returns:
            ImageQuickLabelResult，包含违规类型和风险分数
        """
        if not self._embedder.ready:
            logger.warning("CLIP 编码器未就绪，降级到默认结果")
            return self._default_fallback()

        try:
            # 加载图片
            image = Image.open(image_path).convert("RGB")
            image_input = self._embedder._preprocess(image).unsqueeze(0).to(
                self._embedder.device
            )

            # 确保文本嵌入已计算
            self._ensure_embeddings()

            # 计算图片嵌入
            with self._torch.no_grad():
                image_emb = self._embedder._model.encode_image(image_input)
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

            # 计算与各违规类型的相似度
            similarities: dict[ViolationType, float] = {}
            for vtype, text_emb in self._text_embeddings.items():
                sim = self._torch.cosine_similarity(image_emb, text_emb).item()
                similarities[vtype] = sim

            # 多标签分类：超过阈值即为该类型
            detected_types: list[ViolationType] = []
            evidence: list[str] = []

            for vtype, sim in similarities.items():
                if sim > self._THRESHOLD:
                    detected_types.append(vtype)
                    evidence.append(f"{vtype.value}_similarity={sim:.3f}")

            # 如果没有检测到违规类型，标记为 normal
            if not detected_types:
                max_sim = max(similarities.values()) if similarities else 0.0
                return ImageQuickLabelResult(
                    violation_types=[ViolationType.NORMAL],
                    risk_score=1.0 - max_sim,
                    evidence=[f"max_similarity={max_sim:.3f}"],
                    decision_source="clip_zero_shot",
                )

            # 计算综合风险分数（取最高风险类型的加权分数）
            max_risk = 0.0
            for vtype in detected_types:
                base = self._BASE_SCORES.get(vtype, 0.5)
                sim = similarities.get(vtype, 0.0)
                risk = base * sim
                max_risk = max(max_risk, risk)

            return ImageQuickLabelResult(
                violation_types=detected_types,
                risk_score=min(max_risk, 1.0),
                evidence=evidence,
                decision_source="clip_zero_shot",
            )

        except Exception as e:
            logger.error("CLIP 快速打标失败: %s", e)
            return self._default_fallback()

    def _default_fallback(self) -> ImageQuickLabelResult:
        """CLIP 失败时的默认降级结果

        返回需要进一步审核的标记，由调用方决定是否使用 VL。
        """
        return ImageQuickLabelResult(
            violation_types=[ViolationType.OTHER],
            risk_score=0.5,
            evidence=["clip_failed_default_fallback"],
            decision_source="default_fallback",
        )

    def predict_from_base64(self, base64_str: str) -> ImageQuickLabelResult:
        """从 Base64 字符串进行快速打标

        Args:
            base64_str: Base64 编码的图片字符串

        Returns:
            ImageQuickLabelResult
        """
        import base64 as b64
        import io

        try:
            image_data = b64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            if not self._embedder.ready:
                return self._default_fallback()

            image_input = self._embedder._preprocess(image).unsqueeze(0).to(
                self._embedder.device
            )

            self._ensure_embeddings()

            with self._torch.no_grad():
                image_emb = self._embedder._model.encode_image(image_input)
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

            similarities: dict[ViolationType, float] = {}
            for vtype, text_emb in self._text_embeddings.items():
                sim = self._torch.cosine_similarity(image_emb, text_emb).item()
                similarities[vtype] = sim

            detected_types: list[ViolationType] = []
            evidence: list[str] = []

            for vtype, sim in similarities.items():
                if sim > self._THRESHOLD:
                    detected_types.append(vtype)
                    evidence.append(f"{vtype.value}_similarity={sim:.3f}")

            if not detected_types:
                max_sim = max(similarities.values()) if similarities else 0.0
                return ImageQuickLabelResult(
                    violation_types=[ViolationType.NORMAL],
                    risk_score=1.0 - max_sim,
                    evidence=[f"max_similarity={max_sim:.3f}"],
                    decision_source="clip_zero_shot",
                )

            max_risk = 0.0
            for vtype in detected_types:
                base = self._BASE_SCORES.get(vtype, 0.5)
                sim = similarities.get(vtype, 0.0)
                risk = base * sim
                max_risk = max(max_risk, risk)

            return ImageQuickLabelResult(
                violation_types=detected_types,
                risk_score=min(max_risk, 1.0),
                evidence=evidence,
                decision_source="clip_zero_shot",
            )

        except Exception as e:
            logger.error("Base64 图片快速打标失败: %s", e)
            return self._default_fallback()


# 全局分类器实例（延迟初始化）
_classifier: CLIPViolationClassifier | None = None


def get_image_classifier() -> CLIPViolationClassifier:
    """获取全局图片分类器实例"""
    global _classifier
    if _classifier is None:
        _classifier = CLIPViolationClassifier()
    return _classifier
