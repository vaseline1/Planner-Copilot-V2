"""memory.py —— 两层记忆：Redis 存"最近对话"，PostgreSQL 存"长期结论"

设计思路：短期记忆（Redis）按 session 过期，快但不持久；长期记忆
（PostgreSQL）永久保留关键结论，规划师隔天再打开这个学生依然能看到。
所有 Redis 操作都做了防御性异常处理——Redis 抽风时 Agent 退化成
"这轮不带记忆"照常回答，不会崩溃。

三期改造后的变化：
    Redis 里的对话历史从散装 dict 换成 ConversationTurn（models.py）
    存取——每条记录带 ISO 时间戳，排查"这段历史是什么时候聊的"时有据
    可查。对外接口不变：get_conversation_history() 依然返回
    [{"role":..., "content":...}, ...]（喂给大模型前剥掉时间戳，
    调用方 app.py / agent_core.py 不用改）。存取时逐条用模型校验，
    坏数据跳过而不是整段历史作废。

四期改造（完整 CRUD）后的变化：
    长期记忆补齐删除能力——delete_memory() 删单条（带 student_id
    双条件防误删）、delete_memories_for_student() 清空某学生的全部
    长期记忆（供"清空学生数据"场景复用）。
"""
import json
import logging
from datetime import datetime
from typing import Optional

import redis
from openai import OpenAI
from pydantic import ValidationError
from sqlmodel import Field, Session, SQLModel, select

from config import REDIS_URL, SESSION_TTL_SECONDS, get_chat_config
from database import engine
from models import ConversationTurn
from prompts import MEMORY_SUMMARY_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

_redis_client: Optional[redis.Redis] = None


def _get_redis() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client is None:
        try:
            client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
            client.ping()
            _redis_client = client
        except redis.RedisError as exc:
            logger.warning("Redis 连接失败，本轮对话将不使用短期记忆：%s", exc)
            return None
    return _redis_client


def check_connection() -> tuple[bool, str]:
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        return True, "Redis 连接正常"
    except redis.RedisError as exc:
        return False, f"Redis 连接失败：{exc}"


class ConsultationMemory(SQLModel, table=True):
    __tablename__ = "consultation_memories"

    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="students.id", index=True)
    memory_type: str
    title: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


def save_memory(student_id: int, memory_type: str, title: str, content: str) -> None:
    with Session(engine) as session:
        existing = session.exec(
            select(ConsultationMemory).where(
                ConsultationMemory.student_id == student_id,
                ConsultationMemory.title == title,
            )
        ).first()
        if existing:
            existing.content = content
            existing.updated_at = datetime.now()
            session.add(existing)
        else:
            session.add(
                ConsultationMemory(student_id=student_id, memory_type=memory_type, title=title, content=content)
            )
        session.commit()


def get_student_memories(student_id: int) -> list[ConsultationMemory]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(ConsultationMemory)
                .where(ConsultationMemory.student_id == student_id)
                .order_by(ConsultationMemory.updated_at.desc())
            )
        )


def delete_memory(memory_id: int, student_id: Optional[int] = None) -> bool:
    """删除单条长期记忆。返回是否成功。

    student_id 传入时做双条件校验——调用方（api.py / app.py）都是
    "先选中学生、再删它名下的记忆"，带上这个条件可以防止跨学生误删。
    """
    with Session(engine) as session:
        memory = session.get(ConsultationMemory, memory_id)
        if memory is None:
            return False
        if student_id is not None and memory.student_id != student_id:
            return False
        session.delete(memory)
        session.commit()
        return True


def delete_memories_for_student(student_id: int) -> int:
    """删掉一个学生的全部长期记忆，返回删除条数。

    供"清空某学生所有数据"的场景复用；database.py 的级联删除因为
    循环导入问题用的是原生 SQL，两者效果一致。
    """
    with Session(engine) as session:
        memories = session.exec(
            select(ConsultationMemory).where(ConsultationMemory.student_id == student_id)
        ).all()
        count = len(memories)
        for memory in memories:
            session.delete(memory)
        session.commit()
        return count


def format_memories_for_context(student_id: int) -> str:
    memories = get_student_memories(student_id)
    if not memories:
        return "（该学生暂无历史咨询记录，本次为首次对话）"
    lines = ["该学生的历史咨询要点："]
    for mem in memories[:5]:
        snippet = mem.content if len(mem.content) <= 120 else mem.content[:120] + "..."
        lines.append(f"- [{mem.title}] {snippet}")
    return "\n".join(lines)


MAX_TURNS_BEFORE_COMPRESS = 20
KEEP_RECENT_TURNS = 6


def _history_key(session_id: str) -> str:
    return f"history:{session_id}"


def _load_raw_turns(session_id: str) -> list[dict]:
    """从 Redis 读原始历史（含时间戳的 ConversationTurn 序列化结果）。"""
    client = _get_redis()
    if client is None:
        return []
    raw = client.get(_history_key(session_id))
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("会话 %s 的历史记录解析失败，视为空历史", session_id)
        return []
    return data if isinstance(data, list) else []


def get_conversation_history(session_id: str) -> list[dict]:
    """读取会话历史，返回 [{"role":..., "content":...}, ...]。

    存储格式是带时间戳的 ConversationTurn，但喂给大模型只需要
    role/content——在这里剥掉，调用方拿到的格式与旧版完全一致。
    每条单独校验，坏掉的条目跳过，不让一条脏数据毁掉整段历史。
    """
    turns: list[dict] = []
    for item in _load_raw_turns(session_id):
        try:
            turn = ConversationTurn.model_validate(item)
        except ValidationError:
            logger.warning("会话 %s 中有一条历史记录格式异常，已跳过", session_id)
            continue
        turns.append({"role": turn.role, "content": turn.content})
    return turns


def append_turn(session_id: str, role: str, content: str) -> None:
    client = _get_redis()
    if client is None:
        return
    history = _load_raw_turns(session_id)
    turn = ConversationTurn(role=role, content=content)
    history.append(turn.model_dump(mode="json"))

    if len(history) > MAX_TURNS_BEFORE_COMPRESS:
        older, recent = history[:-KEEP_RECENT_TURNS], history[-KEEP_RECENT_TURNS:]
        summary = _summarize(older)
        summary_turn = ConversationTurn(role="system", content=f"[历史对话摘要] {summary}")
        history = [summary_turn.model_dump(mode="json")] + recent

    try:
        client.setex(_history_key(session_id), SESSION_TTL_SECONDS, json.dumps(history, ensure_ascii=False))
    except redis.RedisError as exc:
        logger.warning("写入 Redis 失败，本轮对话未持久化：%s", exc)


def clear_history(session_id: str) -> None:
    client = _get_redis()
    if client is None:
        return
    try:
        client.delete(_history_key(session_id))
    except redis.RedisError as exc:
        logger.warning("清空 Redis 会话失败：%s", exc)


def _summarize(history: list[dict]) -> str:
    text = "\n".join(f"{h['role']}: {h['content']}" for h in history)
    try:
        cfg = get_chat_config()
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": MEMORY_SUMMARY_PROMPT_TEMPLATE.format(text=text[:3000])}],
            max_tokens=200,
        )
        return response.choices[0].message.content or text[:200]
    except Exception as exc:  # noqa: BLE001
        logger.warning("生成对话摘要失败，退化为截断：%s", exc)
        return text[:200] + "……（摘要生成失败，已截断保留）"
