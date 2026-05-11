"""CLIP 图像编码器模块

使用 CLIP Vision Encoder 将图片编码为 768 维向量。
与 CLIPTextEmbeddings 共用同一个语义空间，支持跨模态检索。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

# CLIP 延迟导入，避免启动时强制依赖
try:
    import clip
    CLIP_AVAILABLE = True
except (ImportError, OSError):
    # OSError: torch DLL 加载失败（如 Python 3.14 不兼容）
    CLIP_AVAILABLE = False
    clip = None

logger = logging.getLogger(__name__)

# CLIP 模型名称（ViT-B/16 = 512维向量）
DEFAULT_MODEL = "ViT-B/16"


class CLIPImageEmbedder:
    """CLIP 图像编码器

    将图片编码为 768 维向量，与 CLIPTextEmbeddings 共用语义空间。
    支持单图和批量编码。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        """初始化 CLIP 图像编码器

        Args:
            model_name: CLIP 模型名称，默认使用 ViT-B/16 (768维)
            device: 运行设备，"cuda" 或 "cpu"，默认自动检测
        """
        import torch
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if not CLIP_AVAILABLE:
            logger.error("CLIP 模块不可用，请安装: pip install git+https://github.com/openai/CLIP.git")
            self._model = None
            self._preprocess = None
            self.ready = False
            return

        try:
            self._model, self._preprocess = clip.load(model_name, device=self.device)
            self._model.eval()
            self.ready = True
            logger.info("CLIP 图像编码器初始化成功: model=%s, device=%s", model_name, self.device)
        except Exception as e:
            logger.error("CLIP 图像编码器初始化失败: %s", e)
            self._model = None
            self._preprocess = None
            self.ready = False

    def encode_image(self, image_path: str) -> list[float]:
        """编码单张图片

        Args:
            image_path: 图片文件路径

        Returns:
            768 维向量
        """
        if not self.ready or self._model is None:
            return [0.0] * 768

        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self._preprocess(image).unsqueeze(0).to(self.device)

            with self._torch.no_grad():
                embedding = self._model.encode_image(image_input)
                # L2 归一化
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.float().cpu().numpy()[0].tolist()

        except Exception as e:
            logger.error("图片编码失败: path=%s, error=%s", image_path, e)
            return [0.0] * 768

    def encode_images(self, image_paths: list[str]) -> list[list[float]]:
        """批量编码多张图片

        Args:
            image_paths: 图片文件路径列表

        Returns:
            向量列表，每个向量 768 维
        """
        if not self.ready or self._model is None:
            return [[0.0] * 768 for _ in image_paths]

        images = []
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(self._preprocess(image))
            except Exception as e:
                logger.warning("图片加载失败，跳过: path=%s, error=%s", path, e)
                # 使用零图代替加载失败的图片（使用 PIL 创建黑色图片）
                images.append(self._preprocess(Image.new("RGB", (224, 224))))

        if not images:
            return [[0.0] * 768 for _ in image_paths]

        # 批量编码
        image_input = self._torch.stack(images).to(self.device)

        with self._torch.no_grad():
            embeddings = self._model.encode_image(image_input)
            # L2 归一化
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.float().cpu().numpy().tolist()

    def encode_from_pil(self, image: Image.Image) -> list[float]:
        """从 PIL 图片对象编码

        Args:
            image: PIL 图片对象

        Returns:
            768 维向量
        """
        if not self.ready or self._model is None:
            return [0.0] * 768

        try:
            image_rgb = image.convert("RGB")
            image_input = self._preprocess(image_rgb).unsqueeze(0).to(self.device)

            with self._torch.no_grad():
                embedding = self._model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.float().cpu().numpy()[0].tolist()

        except Exception as e:
            logger.error("PIL 图片编码失败: %s", e)
            return [0.0] * 768

    def encode_from_base64(self, base64_str: str) -> list[float]:
        """从 Base64 字符串编码图片

        Args:
            base64_str: Base64 编码的图片字符串

        Returns:
            768 维向量
        """
        import base64 as b64
        import io

        if not self.ready or self._model is None:
            return [0.0] * 768

        try:
            image_data = b64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return self.encode_from_pil(image)

        except Exception as e:
            logger.error("Base64 图片解码失败: %s", e)
            return [0.0] * 768

    def encode_text(self, text: str) -> list[float]:
        """编码文本（用于相似度计算）

        Args:
            text: 文本字符串

        Returns:
            768 维向量
        """
        if not self.ready or self._model is None:
            return [0.0] * 768

        try:
            # CLIP 对文本长度有限制（最多 77 个 token），进行截断
            text = text[:300]

            with self._torch.no_grad():
                text_tokens = clip.tokenize([text], truncate=True).to(self.device)
                embedding = self._model.encode_text(text_tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.float().cpu().numpy()[0].tolist()

        except Exception as e:
            logger.error("文本编码失败: %s", e)
            return [0.0] * 768


def embed_from_pil(self, image: Image.Image) -> list[float]:
    """从 PIL 图片对象编码（向后兼容别名）"""
    return self.encode_from_pil(image)


def embed_from_base64(self, base64_str: str) -> list[float]:
    """从 Base64 字符串编码图片（向后兼容别名）"""
    return self.encode_from_base64(base64_str)
