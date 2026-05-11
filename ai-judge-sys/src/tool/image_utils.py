"""图片预处理工具模块

提供图片审核前的预处理功能：
1. 图片格式校验
2. 等比缩放
3. 分辨率降低（控制token消耗）
4. Base64 编码（供 VL 模型使用）
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# 支持的图片格式
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# VL 模型输入的目标尺寸（1024x1024 平衡了细节和 token 消耗）
TARGET_SIZE = (1024, 1024)

# JPEG 压缩质量（0-100，越低文件越小）
JPEG_QUALITY = 85


def validate_image_format(image_path: str) -> bool:
    """校验图片格式是否支持

    Args:
        image_path: 图片文件路径

    Returns:
        True 如果格式支持，否则 False
    """
    suffix = Path(image_path).suffix.lower()
    return suffix in SUPPORTED_FORMATS


def validate_image_path(image_path: str) -> bool:
    """校验图片文件是否存在且格式支持

    Args:
        image_path: 图片文件路径

    Returns:
        True 如果文件存在且格式支持
    """
    path = Path(image_path)
    return path.exists() and path.is_file() and validate_image_format(image_path)


def resize_image_keep_aspect(
    image: Image.Image,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> Image.Image:
    """等比缩放图片

    根据较长边进行缩放，保持原始宽高比。
    尺寸会调整为 32 的倍数（CLIP 模型的要求）。

    Args:
        image: PIL 图片对象
        target_size: 目标尺寸 (宽, 高)

    Returns:
        缩放后的 PIL 图片对象
    """
    original_width, original_height = image.size
    target_width, target_height = target_size

    # 根据较长边计算新尺寸
    if original_width >= original_height:
        new_width = target_width
        new_height = int(original_height * (target_width / original_width))
    else:
        new_height = target_height
        new_width = int(original_width * (target_height / original_height))

    # 调整为 32 的倍数（CLIP 要求）
    new_width = (new_width // 32) * 32
    new_height = (new_height // 32) * 32

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def reduce_resolution(
    image: Image.Image,
    max_dimension: int = 1024,
) -> Image.Image:
    """降低图片分辨率

    如果图片任意边超过 max_dimension，按比例缩小。
    尺寸会调整为 32 的倍数。

    Args:
        image: PIL 图片对象
        max_dimension: 最大边长阈值

    Returns:
        调整后的 PIL 图片对象

    Raises:
        ValueError: 图片尺寸无效（宽或高为0）
    """
    width, height = image.size

    # 校验尺寸有效性，防止除零
    if width == 0 or height == 0:
        raise ValueError(f"图片尺寸无效: {width}x{height}")

    # 如果图片已经在阈值内，直接返回
    if width <= max_dimension and height <= max_dimension:
        return image

    # 按比例缩小
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))

    # 调整为 32 的倍数
    new_width = (new_width // 32) * 32
    new_height = (new_height // 32) * 32

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def encode_image_to_base64(image: Image.Image, img_format: str = "JPEG") -> str:
    """将图片编码为 Base64 字符串

    用于发送给 VL 模型（如 Qwen-VL）进行审核。

    Args:
        image: PIL 图片对象
        img_format: 输出格式 (JPEG 或 PNG)

    Returns:
        Base64 编码字符串（不含 data:image 前缀）
    """
    buffer = io.BytesIO()

    # JPEG 不支持 RGBA 模式，需要转换
    if img_format.upper() == "JPEG":
        image = image.convert("RGB")

    image.save(buffer, format=img_format.upper(), quality=JPEG_QUALITY)
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_image_path_to_base64(image_path: str) -> str:
    """将图片文件编码为 Base64

    Args:
        image_path: 图片文件路径

    Returns:
        Base64 编码字符串
    """
    with Image.open(image_path) as image:
        return encode_image_to_base64(image)


def preprocess_image(
    image_path: str,
    max_dimension: int = 1024,
) -> tuple[Image.Image, str]:
    """图片预处理完整流程

    处理步骤：
    1. 校验图片路径和格式
    2. 降低分辨率（控制 token 消耗）
    3. 等比缩放
    4. 编码为 Base64

    Args:
        image_path: 图片文件路径
        max_dimension: 最大边长阈值

    Returns:
        (处理后的 PIL 图片对象, Base64 字符串)

    Raises:
        ValueError: 文件不存在或不支持的图片格式
    """
    if not validate_image_path(image_path):
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"图片文件不存在: {image_path}")
        if not validate_image_format(image_path):
            raise ValueError(f"不支持的图片格式: {path.suffix}")

    image = Image.open(image_path).convert("RGB")

    # 步骤1: 降低分辨率
    image = reduce_resolution(image, max_dimension=max_dimension)

    # 步骤2: 等比缩放
    image = resize_image_keep_aspect(image, target_size=TARGET_SIZE)

    # 步骤3: 编码为 Base64
    base64_str = encode_image_to_base64(image)

    logger.info(
        "图片预处理完成: path=%s, 尺寸=%s",
        image_path,
        image.size,
    )

    return image, base64_str


def preprocess_image_from_bytes(
    image_bytes: bytes,
    max_dimension: int = 1024,
) -> tuple[Image.Image, str]:
    """从字节数据预处理图片

    用于从 MinIO 等存储直接获取 bytes 进行处理。

    Args:
        image_bytes: 图片原始字节数据
        max_dimension: 最大边长阈值

    Returns:
        (处理后的 PIL 图片对象, Base64 字符串)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 降低分辨率
    image = reduce_resolution(image, max_dimension=max_dimension)

    # 等比缩放
    image = resize_image_keep_aspect(image, target_size=TARGET_SIZE)

    # 编码为 Base64
    base64_str = encode_image_to_base64(image)

    return image, base64_str


def get_image_info(image_path: str) -> dict[str, Any]:
    """获取图片元信息（不进行预处理）

    Args:
        image_path: 图片文件路径

    Returns:
        包含宽、高、格式、大小等信息的字典
    """
    path = Path(image_path)
    with Image.open(image_path) as image:
        width, height = image.size
        return {
            "width": width,
            "height": height,
            "format": image.format,
            "mode": image.mode,
            "size_bytes": path.stat().st_size,
            "aspect_ratio": width / height if height > 0 else 0,
        }
