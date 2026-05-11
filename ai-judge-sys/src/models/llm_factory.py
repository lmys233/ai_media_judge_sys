"""
LLM工厂模块 - 支持多种AI模型提供商

支持的模型:
- qwen: 阿里通义千问 (默认)
- openai: OpenAI GPT系列
- anthropic: Anthropic Claude系列

配置方式:
1. 环境变量:
   - LLM_PROVIDER: 模型提供商 (qwen/openai/anthropic)
   - LLM_API_KEY: API密钥
   - LLM_MODEL: 模型名称 (可选，有默认值)
   - LLM_BASE_URL: API地址 (可选，用于代理/自定义端点)

2. 或者在各模块特定环境变量:
   - JUDGE_PROVIDER, JUDGE_API_KEY, JUDGE_MODEL
   - DESENSITIZE_LLM_PROVIDER, DESENSITIZE_LLM_API_KEY, etc.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_PROVIDER = "qwen"
DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_TEMPERATURE = 0.0


def create_llm(
    provider_env: str = "LLM_PROVIDER",
    api_key_env: str = "LLM_API_KEY",
    model_env: str = "LLM_MODEL",
    base_url_env: str = "LLM_BASE_URL",
    default_temperature: float = DEFAULT_TEMPERATURE,
    fallback_provider: str | None = None,
) -> Any | None:
    """创建LLM实例的工厂函数。

    Args:
        provider_env: 模型提供商环境变量名
        api_key_env: API密钥环境变量名
        model_env: 模型名称环境变量名
        base_url_env: API地址环境变量名 (可选)
        default_temperature: 默认温度参数
        fallback_provider: 当环境变量未设置时的默认值

    Returns:
        LLM实例，如果创建失败则返回None
    """
    # 尝试获取provider配置
    provider = os.getenv(provider_env)
    if not provider and fallback_provider:
        provider = fallback_provider

    # 如果仍然没有provider，返回None（将使用fallback逻辑）
    if not provider or provider == "none":
        logger.info("LLM provider未配置或为'none'，将使用fallback逻辑")
        return None

    provider = provider.lower().strip()

    # 获取API密钥
    api_key = os.getenv(api_key_env)
    if not api_key:
        # 尝试通用的API key环境变量
        api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        logger.warning(f"未找到API密钥 ({api_key_env} 或 LLM_API_KEY)，LLM将不可用")
        return None

    # 获取模型名称
    model = os.getenv(model_env) or os.getenv("LLM_MODEL")

    # 获取base URL（用于代理）
    base_url = os.getenv(base_url_env) or os.getenv("LLM_BASE_URL")

    try:
        if provider == "qwen":
            return _create_qwen_llm(api_key, model, base_url, default_temperature)
        elif provider == "openai":
            return _create_openai_llm(api_key, model, base_url, default_temperature)
        elif provider in ("anthropic", "claude"):
            return _create_anthropic_llm(api_key, model, base_url, default_temperature)
        else:
            logger.warning(f"不支持的LLM provider: {provider}，尝试使用qwen作为默认")
            return _create_qwen_llm(api_key, model, base_url, default_temperature)
    except Exception:  # noqa: BLE001
        logger.exception(f"创建LLM实例失败: provider={provider}")
        return None


def _create_qwen_llm(
    api_key: str,
    model: str | None,
    base_url: str | None,
    temperature: float,
) -> Any:
    """创建阿里通义千问LLM实例。"""
    from langchain_openai import ChatOpenAI

    # 通义千问的API地址
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    return ChatOpenAI(
        model=model or DEFAULT_QWEN_MODEL,
        api_key=api_key,
        base_url=base_url or default_base_url,
        temperature=temperature,
    )


def _create_openai_llm(
    api_key: str,
    model: str | None,
    base_url: str | None,
    temperature: float,
) -> Any:
    """创建OpenAI LLM实例。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or DEFAULT_OPENAI_MODEL,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def _create_anthropic_llm(
    api_key: str,
    model: str | None,
    base_url: str | None,
    temperature: float,
) -> Any:
    """创建Anthropic Claude LLM实例。"""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model or DEFAULT_ANTHROPIC_MODEL,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


# 便捷函数：创建默认的Judge LLM
def create_judge_llm() -> Any | None:
    """创建用于审核判断的LLM实例。"""
    return create_llm(
        provider_env="JUDGE_PROVIDER",
        api_key_env="JUDGE_API_KEY",
        model_env="JUDGE_MODEL",
        base_url_env="JUDGE_BASE_URL",
        default_temperature=0.0,
        fallback_provider=DEFAULT_PROVIDER,
    )


# 便捷函数：创建用于文本重写的LLM
def create_desensitize_llm() -> Any | None:
    """创建用于文本重写的LLM实例。"""
    return create_llm(
        provider_env="DESENSITIZE_LLM_PROVIDER",
        api_key_env="DESENSITIZE_LLM_API_KEY",
        model_env="DESENSITIZE_LLM_MODEL",
        base_url_env="DESENSITIZE_LLM_BASE_URL",
        default_temperature=0.3,
        fallback_provider=DEFAULT_PROVIDER,
    )
