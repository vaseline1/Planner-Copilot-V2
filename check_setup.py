"""check_setup.py —— 启动前体检脚本

职责：
    在真正运行 main.py / app.py / api.py 之前，把全部外部依赖是否就绪
    挨个检查一遍：API Key、PostgreSQL、pgvector 扩展、Redis、知识库文件。
    有问题直接告诉你具体哪里有问题、该怎么修。

用法：
    uv run python check_setup.py
"""
from __future__ import annotations

import sys


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    try:
        from config import get_chat_config

        cfg = get_chat_config()
        checks.append((f"Chat API Key（当前供应商：{cfg.name}）", True, f"模型：{cfg.model}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("Chat API Key", False, str(exc)))

    try:
        from config import get_embedding_config

        emb_cfg = get_embedding_config()
        checks.append((f"Embedding API Key（RAG用，供应商：{emb_cfg.name}）", True, f"模型：{emb_cfg.model}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("Embedding API Key（RAG用）", False, str(exc)))

    try:
        from database import check_connection as db_check

        ok, msg = db_check()
        checks.append(("PostgreSQL 数据库", ok, msg))
    except Exception as exc:  # noqa: BLE001
        checks.append(("PostgreSQL 数据库", False, f"检查过程本身出错：{exc}"))

    try:
        from sqlmodel import Session, text

        from database import engine

        with Session(engine) as session:
            result = session.exec(text("SELECT extname FROM pg_extension WHERE extname='vector'")).first()
        if result:
            checks.append(("pgvector 扩展（RAG用）", True, "已启用"))
        else:
            checks.append(
                (
                    "pgvector 扩展（RAG用）",
                    False,
                    "未启用。请执行：psql \"$DATABASE_URL\" -c \"CREATE EXTENSION vector;\"",
                )
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(("pgvector 扩展（RAG用）", False, f"检查失败（可能是数据库本身连不上）：{exc}"))

    try:
        from memory import check_connection as redis_check

        ok, msg = redis_check()
        checks.append(("Redis", ok, msg))
    except Exception as exc:  # noqa: BLE001
        checks.append(("Redis", False, f"检查过程本身出错：{exc}"))

    try:
        from pathlib import Path

        from config import KNOWLEDGE_BASE_PATH

        kb_path = Path(KNOWLEDGE_BASE_PATH)
        if kb_path.exists():
            import json

            data = json.loads(kb_path.read_text(encoding="utf-8"))
            checks.append(("本地知识库文件", True, f"{kb_path}，{len(data)} 条记录（示例数据，建议补充真实数据）"))
        else:
            checks.append(("本地知识库文件", False, f"{kb_path} 不存在"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("本地知识库文件", False, str(exc)))

    try:
        from observability import is_langfuse_enabled

        if is_langfuse_enabled():
            checks.append(("Langfuse 可观测性（可选）", True, "已配置，追踪数据会发往 Langfuse"))
        else:
            checks.append(("Langfuse 可观测性（可选）", True, "未配置，Agent 正常工作，只是不追踪——不影响使用"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("Langfuse 可观测性（可选）", True, f"检查本身出错，但不影响使用：{exc}"))

    print("=" * 55)
    print("PlannerCopilot 完整版 启动前体检")
    print("=" * 55)
    all_ok = True
    for name, ok, detail in checks:
        icon = "✅" if ok else "❌"
        print(f"{icon} {name}：{detail}")
        all_ok = all_ok and ok

    print("=" * 55)
    if all_ok:
        print("全部检查通过！可以运行：")
        print("  uv run python main.py              命令行版")
        print("  uv run streamlit run app.py         网页工作台")
        print("  uv run uvicorn api:app --reload     API 服务")
        return 0

    print("有检查项没通过，请先按上面的提示修复，再重新运行本脚本。")
    print()
    print("常见修复方式（macOS）：")
    print("  PostgreSQL 没启动     -> brew services start postgresql@16")
    print("  Redis 没启动          -> brew services start redis")
    print("  pgvector 没装/没启用  -> brew install pgvector 然后重新执行 CREATE EXTENSION")
    print("  API Key 没填          -> 检查 .env 文件里的 DEEPSEEK_API_KEY / SILICONFLOW_API_KEY")
    return 1


if __name__ == "__main__":
    sys.exit(main())
