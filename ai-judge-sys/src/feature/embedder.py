"""文本向量化模块

使用 CLIP Text Encoder 统一文本和图片的语义空间。
支持与环境变量 EMBEDDING_PROVIDER 切换：
- "clip": 使用 CLIP（默认，推荐，可与图片统一检索）
- "openai": 使用 OpenAI text-embedding-3-small
- "hash": 使用 HashFallback（兼容旧数据）
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class CLIPTextEmbeddings(Embeddings):
    """CLIP 文本编码器

    使用 CLIP 模型将文本编码为 768 维向量。
    与 CLIP 图像编码器共用同一个语义空间，支持跨模态检索。
    """

    def __init__(
        self,
        model_name: str = "ViT-B/16",  # ViT-B/16 = 512维
        device: str | None = None,
    ) -> None:
        """初始化 CLIP 文本编码器

        Args:
            model_name: CLIP 模型名称，默认使用 ViT-B/16 (768维)
            device: 运行设备，"cuda" 或 "cpu"，默认自动检测
        """
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            import clip
            self._model, self._preprocess = clip.load(model_name, device=self.device)
            self._model.eval()
            self.ready = True
            logger.info("CLIP 文本编码器初始化成功: model=%s, device=%s", model_name, self.device)
        except Exception as e:
            logger.warning("CLIP 模型加载失败，降级到 HashFallback: %s", e)
            self._model = None
            self._preprocess = None
            self.ready = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本列表

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量 768 维
        """
        import torch
        import clip

        if not self.ready or self._model is None:
            # 降级：使用零向量
            return [[0.0] * 768 for _ in texts]

        # CLIP 对文本长度有限制（最多 77 个 token），进行截断
        texts = [text[:77 * 4] for text in texts]  # 约 300 字符

        with torch.no_grad():
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            embeddings = self._model.encode_text(text_tokens)
            # L2 归一化，使余弦相似度等于点积
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.float().cpu().numpy().tolist()

    def embed_query(self, text: str) -> list[float]:
        """编码单个查询文本

        Args:
            text: 查询文本

        Returns:
            768 维向量
        """
        import torch
        import clip

        if not self.ready or self._model is None:
            return [0.0] * 768

        # 截断文本
        text = text[:77 * 4]

        with torch.no_grad():
            text_tokens = clip.tokenize([text], truncate=True).to(self.device)
            embedding = self._model.encode_text(text_tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.float().cpu().numpy()[0].tolist()


class HashFallbackEmbeddings(Embeddings):
    """Hash 回退编码器（兼容旧数据）

    使用 SHA256 对每个 token 进行哈希，生成伪嵌入向量。
    维度兼容但语义质量不如神经网络编码器。
    仅用于兼容旧数据或作为 CLIP 加载失败时的回退。
    """

    def __init__(self, dim: int = 768) -> None:
        import hashlib

        self.dim = dim
        self._hashlib_sha256 = hashlib.sha256

    def _hash_embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.split():
            digest = self._hashlib_sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            vector[idx] += 1.0

        # L2 归一化
        norm = sum(v * v for v in vector) ** 0.5
        if norm <= 1e-12:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)


def build_embedding_model() -> Embeddings:
    """构建文本向量化模型

    根据环境变量 EMBEDDING_PROVIDER 选择编码器：
    - "clip": CLIP 神经网络编码器（默认，推荐）
    - "openai": OpenAI text-embedding-3-small
    - "hash": HashFallbackEmbeddings（兼容旧数据）

    Returns:
        Embeddings 实例
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "clip").lower()

    if provider == "clip":
        try:
            return CLIPTextEmbeddings()
        except Exception as e:
            logger.warning("CLIP 编码器初始化失败，降级到 Hash: %s", e)
            return HashFallbackEmbeddings(dim=768)

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings

            model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            return OpenAIEmbeddings(model=model_name)
        except Exception as e:
            logger.warning("OpenAI 编码器初始化失败，降级到 Hash: %s", e)
            return HashFallbackEmbeddings(dim=768)

    # 默认或 hash
    logger.info("使用 HashFallback 文本编码器")
    return HashFallbackEmbeddings(dim=768)
