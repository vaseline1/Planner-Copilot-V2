"""config.py —— 双轨道 API 与服务配置

职责：
    从 .env 读取所有外部服务的配置（大模型 API、数据库、Redis、规划师
    鉴权 Key），向外暴露 get_chat_config() / get_embedding_config()，
    以及各种连接字符串常量。

为什么"双轨道"：
    DeepSeek 没有 /embeddings 接口，所以"对话推理"和"向量化"必须能
    各自独立配置、独立切换供应商。完整版从第9周开始 Embedding 轨道
    真正被 rag_system.py 使用。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """配置缺失或不合法时抛出，message 里必须包含具体的修复步骤。"""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: Optional[str]
    base_url: str
    model: str
    supports_embedding: bool = False
    dimension: Optional[int] = None


def _build_chat_providers() -> dict[str, ProviderConfig]:
    return {
        "deepseek": ProviderConfig(
            name="deepseek",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            supports_embedding=False,
        ),
        "siliconflow": ProviderConfig(
            name="siliconflow",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            model="deepseek-ai/DeepSeek-V3",
            supports_embedding=True,
            dimension=1024,
        ),
    }


def _build_embedding_providers() -> dict[str, ProviderConfig]:
    return {
        "siliconflow": ProviderConfig(
            name="siliconflow",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            model="BAAI/bge-large-zh-v1.5",
            supports_embedding=True,
            dimension=1024,
        ),
    }


def get_chat_config() -> ProviderConfig:
    """获取当前 Chat LLM 配置（对话推理、工具调用都走这条轨道）。"""
    provider = os.getenv("CHAT_PROVIDER", "deepseek")
    providers = _build_chat_providers()
    cfg = providers.get(provider)
    if cfg is None:
        raise ConfigError(f"未知的 CHAT_PROVIDER='{provider}'。可选值：{list(providers)}")
    if not cfg.api_key:
        env_var = "DEEPSEEK_API_KEY" if provider == "deepseek" else "SILICONFLOW_API_KEY"
        raise ConfigError(f"{provider} 的 API Key 未设置。请在 .env 文件里填写：{env_var}=你的真实key")
    return cfg


def get_embedding_config() -> ProviderConfig:
    """获取当前 Embedding 配置（rag_system.py 用，简章向量化）。"""
    provider = os.getenv("EMBEDDING_PROVIDER", "siliconflow")
    providers = _build_embedding_providers()
    cfg = providers.get(provider)
    if cfg is None:
        raise ConfigError(f"未知的 EMBEDDING_PROVIDER='{provider}'。")
    if not cfg.api_key:
        raise ConfigError(f"{provider} 的 API Key 未设置。请在 .env 文件里填写：SILICONFLOW_API_KEY=你的真实key")
    return cfg


# ── 数据库 / Redis 连接字符串 ────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://planner:planner123@localhost:5432/planner_copilot"
)
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_TOOL_CALL_ROUNDS: int = int(os.getenv("MAX_TOOL_CALL_ROUNDS", "6"))

# ── 知识库 / RAG ─────────────────────────────────────────────
KNOWLEDGE_BASE_PATH: str = os.getenv("KNOWLEDGE_BASE_PATH", "data/tianjin_programs.json")
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5"))

# ── FastAPI 多规划师鉴权 ─────────────────────────────────────
# 格式：PLANNER_API_KEYS=key1:planner_001,key2:planner_002
def get_planner_api_keys() -> dict[str, str]:
    """解析 .env 里的 PLANNER_API_KEYS，返回 {api_key: planner_id}。"""
    raw = os.getenv("PLANNER_API_KEYS", "planner-key-001:planner_001")
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, planner_id = pair.split(":", 1)
        result[key.strip()] = planner_id.strip()
    return result


# ── 批量处理（batch_analysis.py 用）────────────────────────────
BATCH_MAX_CONCURRENT: int = int(os.getenv("BATCH_MAX_CONCURRENT", "5"))
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
