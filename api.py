"""api.py —— FastAPI 对外接口：多规划师版本

职责：
    把 agent_core / database 包装成 HTTP 接口，支持多个规划师各自用
    自己的 API Key 访问、数据互相隔离（规划师A看不到规划师B的学生）。
    这是"给其他规划师也能用"这一层，不影响 main.py 命令行版本继续独立工作
    ——两者共用同一套 agent_core/database/memory，互不冲突。

为什么鉴权检查放在 Depends()：
    FastAPI 的 Depends 会在真正执行接口逻辑之前先跑鉴权函数，鉴权失败
    直接 401，接口函数本体完全不用操心"这个人是谁"——鉴权逻辑改动时
    只用改 get_planner_id 这一个函数，不用挨个接口去改。

三期改造后的变化：
    - 新增 PATCH /api/students/{id}/stage：规划师手动推进咨询阶段
      （初次建档 → 规划进行中 → 报名申请中 → 已完成），阶段值用
      ConsultationStage 枚举校验，非法值直接 400；
    - StudentResponse 增加 consultation_stage 字段——阶段现在会真实
      流转（agent_core 也会自动推进），列表接口应该看得见。
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select

from agent_core import PlannerAgent
from config import get_planner_api_keys
from database import (
    ConsultationDB,
    StudentDB,
    create_student,
    delete_consultation,
    delete_student,
    engine,
    update_consultation_stage,
    update_student,
    wipe_student_data,
)
from memory import ConsultationMemory, delete_memory
from models import ConsultationStage

app = FastAPI(title="PlannerCopilot API", description="多元升学规划师 AI 助手接口（天津版）", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_agent: Optional[PlannerAgent] = None


def get_agent() -> PlannerAgent:
    """懒加载 Agent 实例——放到第一次真正处理请求时才创建，这样 API Key
    没配置时，`/health` 这类不需要大模型的接口依然能正常访问。"""
    global _agent
    if _agent is None:
        _agent = PlannerAgent()
    return _agent


def get_planner_id(x_api_key: str = Header(...)) -> str:
    valid_keys = get_planner_api_keys()
    planner_id = valid_keys.get(x_api_key)
    if not planner_id:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return planner_id


class ChatRequest(BaseModel):
    message: str
    student_id: Optional[int] = None


class ChatResponse(BaseModel):
    reply: str
    student_id: Optional[int] = None


class CreateStudentRequest(BaseModel):
    name: str
    grade: str
    subject_combo: str
    estimated_rank: Optional[int] = None


class UpdateStageRequest(BaseModel):
    stage: str


class StudentResponse(BaseModel):
    id: int
    name: str
    grade: str
    subject_combo: str
    consultation_stage: str


class UpdateStudentRequest(BaseModel):
    """档案更新请求——所有字段可选，只更新传进来的字段。
    与 database.UPDATABLE_STUDENT_FIELDS 白名单保持一致。"""

    name: Optional[str] = None
    grade: Optional[str] = None
    subject_combo: Optional[str] = None
    estimated_rank: Optional[int] = None
    estimated_score: Optional[int] = None
    gaokao_score: Optional[int] = None
    gaokao_rank: Optional[int] = None
    mock_scores: Optional[list[int]] = None
    competition_awards: Optional[list[dict]] = None
    intended_major_directions: Optional[list[str]] = None
    target_path_types: Optional[list[str]] = None
    planner_notes: Optional[str] = None


class ConsultationResponse(BaseModel):
    id: int
    student_id: int
    topic: str
    content: str
    agent_response: str


class MemoryResponse(BaseModel):
    id: int
    student_id: int
    memory_type: str
    title: str
    content: str


def _get_owned_student(student_id: int, planner_id: str) -> StudentDB:
    """取学生并校验归属；不存在或不属于当前规划师时统一 404。
    用 404 而不是 403——不暴露"这个 ID 的学生存在，只是不归你管"。"""
    with Session(engine) as session:
        student = session.get(StudentDB, student_id)
    if student is None or student.planner_id != planner_id:
        raise HTTPException(status_code=404, detail="学生不存在或不属于当前规划师")
    return student


@app.get("/api/health")
def health_check() -> dict:
    """健康检查接口（监控系统调用，不需要鉴权）。"""
    import redis as redis_lib

    from config import REDIS_URL

    db_ok = False
    try:
        with Session(engine) as session:
            session.exec(select(1))
        db_ok = True
    except Exception:  # noqa: BLE001
        pass

    redis_ok = False
    try:
        redis_lib.from_url(REDIS_URL, socket_connect_timeout=2).ping()
        redis_ok = True
    except Exception:  # noqa: BLE001
        pass

    return {
        "status": "ok" if (db_ok and redis_ok) else "degraded",
        "components": {"database": "ok" if db_ok else "error", "redis": "ok" if redis_ok else "error"},
        "version": "1.0.0",
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, planner_id: str = Depends(get_planner_id)) -> ChatResponse:
    """规划师对话接口。

    注意：这里把 planner_id 当作 session_id 传进去——一举两得：
    ① PlannerAgent.chat() 的签名是 (user_message, session_id, student_id)，
       并没有 planner_id 这个参数，之前写成 planner_id=... 会直接 TypeError 崩掉；
    ② 用 planner_id 作会话号，天然把不同规划师的短期对话记忆隔离在各自的
       Redis session 里，不会共用同一个 "default" 会话串味。
    """
    reply = get_agent().chat(
        user_message=request.message, session_id=planner_id, student_id=request.student_id
    )
    return ChatResponse(reply=reply, student_id=request.student_id)


@app.post("/api/students", response_model=StudentResponse)
def create_student_api(req: CreateStudentRequest, planner_id: str = Depends(get_planner_id)) -> StudentResponse:
    """创建学生档案（自动打上当前规划师的 planner_id 标记）。"""
    student = create_student(
        name=req.name, grade=req.grade, subject_combo=req.subject_combo,
        estimated_rank=req.estimated_rank, planner_id=planner_id,
    )
    return StudentResponse(
        id=student.id, name=student.name, grade=student.grade,
        subject_combo=student.subject_combo, consultation_stage=student.consultation_stage,
    )


@app.get("/api/students", response_model=list[StudentResponse])
def list_students_api(planner_id: str = Depends(get_planner_id)) -> list[StudentResponse]:
    """获取当前规划师名下的学生列表——只返回 planner_id 匹配的记录，
    这就是"多规划师数据隔离"的核心：查询语句里带上 WHERE planner_id = ..."""
    with Session(engine) as session:
        students = session.exec(
            select(StudentDB).where(StudentDB.planner_id == planner_id).order_by(StudentDB.updated_at.desc())
        )
        return [
            StudentResponse(
                id=s.id, name=s.name, grade=s.grade,
                subject_combo=s.subject_combo, consultation_stage=s.consultation_stage,
            )
            for s in students
        ]


@app.patch("/api/students/{student_id}/stage")
def update_stage_api(
    student_id: int, req: UpdateStageRequest, planner_id: str = Depends(get_planner_id)
) -> dict:
    """推进学生的咨询阶段（三期改造新增）。

    阶段值必须是 ConsultationStage 枚举之一：初次建档 / 规划进行中 /
    报名申请中 / 已完成。校验归属——规划师只能改自己名下的学生。
    """
    try:
        stage = ConsultationStage(req.stage)
    except ValueError:
        valid = [s.value for s in ConsultationStage]
        raise HTTPException(status_code=400, detail=f"无效的阶段值：'{req.stage}'。可选：{valid}") from None

    ok = update_consultation_stage(student_id, stage, planner_id=planner_id)
    if not ok:
        raise HTTPException(status_code=404, detail="学生不存在或不属于当前规划师")
    return {"student_id": student_id, "consultation_stage": stage.value}


# ══════════════════════════════════════════════════════════════
# 以下为四期改造新增：完整 CRUD 的 Update / Delete 端点
# ══════════════════════════════════════════════════════════════

@app.patch("/api/students/{student_id}", response_model=StudentResponse)
def update_student_api(
    student_id: int, req: UpdateStudentRequest, planner_id: str = Depends(get_planner_id)
) -> StudentResponse:
    """更新学生档案（只更新请求里出现的字段）。"""
    fields = req.model_dump(exclude_unset=True)
    student = update_student(student_id, fields, planner_id=planner_id)
    if student is None:
        raise HTTPException(status_code=404, detail="学生不存在或不属于当前规划师")
    return StudentResponse(
        id=student.id, name=student.name, grade=student.grade,
        subject_combo=student.subject_combo, consultation_stage=student.consultation_stage,
    )


@app.delete("/api/students/{student_id}")
def delete_student_api(student_id: int, planner_id: str = Depends(get_planner_id)) -> dict:
    """删除学生档案：学生行 + 咨询记录 + 长期记忆全部级联删除，不可恢复。"""
    ok = delete_student(student_id, planner_id=planner_id)
    if not ok:
        raise HTTPException(status_code=404, detail="学生不存在或不属于当前规划师")
    return {"deleted": True, "student_id": student_id}


@app.delete("/api/students/{student_id}/data")
def wipe_student_data_api(student_id: int, planner_id: str = Depends(get_planner_id)) -> dict:
    """清空某学生的所有数据：删掉咨询记录和长期记忆，咨询阶段重置回
    "初次建档"，但保留学生档案壳子——相当于重新开始服务这个学生。"""
    ok = wipe_student_data(student_id, planner_id=planner_id)
    if not ok:
        raise HTTPException(status_code=404, detail="学生不存在或不属于当前规划师")
    return {"wiped": True, "student_id": student_id, "consultation_stage": ConsultationStage.INITIAL.value}


@app.get("/api/students/{student_id}/consultations", response_model=list[ConsultationResponse])
def list_consultations_api(
    student_id: int, planner_id: str = Depends(get_planner_id)
) -> list[ConsultationResponse]:
    """查某学生的咨询记录列表（先校验学生归属）。"""
    _get_owned_student(student_id, planner_id)
    with Session(engine) as session:
        records = session.exec(
            select(ConsultationDB)
            .where(ConsultationDB.student_id == student_id)
            .order_by(ConsultationDB.created_at.desc())
        )
        return [
            ConsultationResponse(
                id=r.id, student_id=r.student_id, topic=r.topic,
                content=r.content, agent_response=r.agent_response,
            )
            for r in records
        ]


@app.delete("/api/consultations/{consultation_id}")
def delete_consultation_api(consultation_id: int, planner_id: str = Depends(get_planner_id)) -> dict:
    """删除单条咨询记录（归属校验在数据层通过 parent 学生完成）。"""
    ok = delete_consultation(consultation_id, planner_id=planner_id)
    if not ok:
        raise HTTPException(status_code=404, detail="咨询记录不存在或不属于当前规划师")
    return {"deleted": True, "consultation_id": consultation_id}


@app.get("/api/students/{student_id}/memories", response_model=list[MemoryResponse])
def list_memories_api(student_id: int, planner_id: str = Depends(get_planner_id)) -> list[MemoryResponse]:
    """查某学生的长期记忆列表（先校验学生归属）。"""
    _get_owned_student(student_id, planner_id)
    with Session(engine) as session:
        memories = session.exec(
            select(ConsultationMemory)
            .where(ConsultationMemory.student_id == student_id)
            .order_by(ConsultationMemory.updated_at.desc())
        )
        return [
            MemoryResponse(
                id=m.id, student_id=m.student_id, memory_type=m.memory_type,
                title=m.title, content=m.content,
            )
            for m in memories
        ]


@app.delete("/api/memories/{memory_id}")
def delete_memory_api(memory_id: int, planner_id: str = Depends(get_planner_id)) -> dict:
    """删除单条长期记忆。

    记忆表自己没有 planner_id，先查出它属于哪个学生，再校验该学生
    是否归当前规划师管——与 delete_consultation 的归属校验思路一致。
    """
    with Session(engine) as session:
        memory = session.get(ConsultationMemory, memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="长期记忆不存在")
        student_id = memory.student_id
    _get_owned_student(student_id, planner_id)
    delete_memory(memory_id, student_id=student_id)
    return {"deleted": True, "memory_id": memory_id}
