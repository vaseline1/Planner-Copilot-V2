"""batch_analysis.py —— 批量学生路径分析（异步并发 + 成本统计）

职责：
    一次性并发分析多个学生的升学路径适配情况，而不是一个个排队处理。
    同时提供 Chat/Embedding 双轨道的成本统计和 Redis 结果缓存。

为什么要用 asyncio 而不是简单的 for 循环：
    调用大模型 API 大部分时间花在"等网络返回"上，不是真的在计算。
    同步写法（for 循环一个个调用）会让 CPU 大部分时间闲着干等；
    asyncio 让程序在等第一个学生的网络返回时，同时把第二个、第三个
    学生的请求也发出去——处理 N 个学生的总耗时能从"N 次网络等待的
    总和"降到"约等于 1 次网络等待"，具体取决于并发上限。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import redis
from openai import AsyncOpenAI

from config import BATCH_MAX_CONCURRENT, CACHE_TTL_SECONDS, REDIS_URL, get_chat_config
from database import StudentDB, engine
from prompts import BATCH_ANALYSIS_SYSTEM_PROMPT
from sqlmodel import Session, select

logger = logging.getLogger(__name__)


async def analyze_student_async(student: StudentDB) -> dict:
    """异步分析单个学生的升学路径适配情况。"""
    cfg = get_chat_config()
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    try:
        response = await client.chat.completions.create(
            model=cfg.model,
            messages=[
                {
                    "role": "system",
                    "content": BATCH_ANALYSIS_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "学生姓名": student.name,
                            "年级": student.grade,
                            "选科组合": student.subject_combo,
                            "预估位次": student.estimated_rank,
                            "竞赛奖项": "有" if student.has_competition_award else "无",
                            "任务": "分析该天津学生可走的升学路径，输出JSON",
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        result_json = json.loads(response.choices[0].message.content)
        return {
            "student_id": student.id,
            "student_name": student.name,
            "success": True,
            "analysis": result_json,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
    except Exception as exc:  # noqa: BLE001 —— 单个学生分析失败不能影响其他学生
        logger.warning("学生 %s 分析失败：%s", student.name, exc)
        return {"student_id": student.id, "student_name": student.name, "success": False, "error": str(exc), "tokens_used": 0}


async def batch_analyze_students(
    student_ids: Optional[list[int]] = None, max_concurrent: int = BATCH_MAX_CONCURRENT
) -> list[dict]:
    """批量分析学生路径适配（并发处理，用信号量限制最大并发数）。"""
    with Session(engine) as session:
        stmt = select(StudentDB)
        if student_ids:
            stmt = stmt.where(StudentDB.id.in_(student_ids))
        students = list(session.exec(stmt))

    if not students:
        logger.info("没有找到学生数据")
        return []

    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def analyze_with_limit(student: StudentDB) -> dict:
        async with semaphore:
            return await analyze_student_async(student)

    tasks = [analyze_with_limit(s) for s in students]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time
    clean_results = [r for r in results if isinstance(r, dict)]
    success_count = sum(1 for r in clean_results if r.get("success"))
    total_tokens = sum(r.get("tokens_used", 0) for r in clean_results)

    logger.info(
        "批量分析完成：学生数=%d 成功=%d 耗时=%.1fs 总Token=%d",
        len(students), success_count, elapsed, total_tokens,
    )
    return clean_results


class CostTracker:
    """分别追踪 Chat（DeepSeek）和 Embedding（SiliconFlow）的 API 成本。
    两条轨道计费方式和单价不同，必须分开统计，不能混在一起算总账。"""

    DEEPSEEK_PRICE_PER_1K_TOKENS = 0.001
    SILICONFLOW_EMB_PRICE_PER_1K = 0.0001

    def __init__(self) -> None:
        self.chat_tokens_total = 0
        self.embedding_chars_total = 0
        self.chat_calls = 0
        self.embedding_calls = 0

    def add_chat_usage(self, tokens: int) -> None:
        self.chat_tokens_total += tokens
        self.chat_calls += 1

    def add_embedding_usage(self, char_count: int) -> None:
        self.embedding_chars_total += char_count
        self.embedding_calls += 1

    def get_report(self) -> dict:
        chat_cost = self.chat_tokens_total / 1000 * self.DEEPSEEK_PRICE_PER_1K_TOKENS
        emb_cost = self.embedding_chars_total / 1000 * self.SILICONFLOW_EMB_PRICE_PER_1K
        return {
            "Chat (DeepSeek)": {"调用次数": self.chat_calls, "总Token数": self.chat_tokens_total, "预估成本": f"¥{chat_cost:.4f}"},
            "Embedding (SiliconFlow)": {"调用次数": self.embedding_calls, "总字符数": self.embedding_chars_total, "预估成本": f"¥{emb_cost:.4f}"},
            "合计预估成本": f"¥{(chat_cost + emb_cost):.4f}",
        }


# ── 选科适配结果缓存（Redis，按500位次分组缓存24小时）─────────────
_cache_client: Optional[redis.Redis] = None


def _get_cache_client() -> Optional[redis.Redis]:
    global _cache_client
    if _cache_client is None:
        try:
            client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
            client.ping()
            _cache_client = client
        except redis.RedisError as exc:
            logger.warning("缓存 Redis 连接失败，本次不缓存：%s", exc)
            return None
    return _cache_client


def get_cached_eligibility(subject_combo: str, rank: int) -> Optional[dict]:
    client = _get_cache_client()
    if client is None:
        return None
    cache_key = f"eligibility:{subject_combo}:{rank // 500}"
    cached = client.get(cache_key)
    return json.loads(cached) if cached else None


def set_cached_eligibility(subject_combo: str, rank: int, result: dict) -> None:
    client = _get_cache_client()
    if client is None:
        return
    cache_key = f"eligibility:{subject_combo}:{rank // 500}"
    try:
        client.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(result, ensure_ascii=False))
    except redis.RedisError as exc:
        logger.warning("写入缓存失败：%s", exc)


# ══════════════════════════════════════════════════════════════
# 运行入口：uv run python batch_analysis.py
# 并发分析数据库里全部学生，打印每份分析和成本报告。
# CostTracker 在这里接的是 analyze_student_async 返回的真实 token 消耗。
# ══════════════════════════════════════════════════════════════

async def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tracker = CostTracker()

    results = await batch_analyze_students()
    if not results:
        print("数据库里还没有学生档案——先用 main.py 输入 new 建几个，再来跑批量分析。")
        return

    for r in results:
        tracker.add_chat_usage(r.get("tokens_used", 0))
        if r.get("success"):
            print(f"\n===== {r['student_name']}（ID {r['student_id']}）=====")
            print(json.dumps(r["analysis"], ensure_ascii=False, indent=2))
        else:
            print(f"\n===== {r['student_name']}（ID {r['student_id']}）分析失败：{r.get('error')} =====")

    print("\n===== 本次批量分析成本报告 =====")
    print(json.dumps(tracker.get_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_demo())
