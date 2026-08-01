"""database.py —— PostgreSQL 持久化层（学生档案 + 咨询记录）

职责：
    定义"学生档案"和"咨询记录"两张表，提供完整的增删改查函数。
    其它文件（memory.py 的长期记忆表、rag_system.py 的向量表）各自管理
    自己的表定义，但都复用这里的 engine——整个项目只有一个数据库连接池。

⚠️ 注意：不要在这个文件顶部写 `from __future__ import annotations`——
它会让 SQLAlchemy 解析 Relationship() 里的字符串前向引用报错
（实测过，见组装说明书第一节问题 5）。

三期改造后的变化：
    StudentDB 补齐完整档案字段——预估分、高考位次、历次模考成绩、竞赛
    奖项明细、意向专业方向、目标升学路径（列表类字段用 JSON 列存储，
    不单独建奖项表：奖项永远跟随学生读写，没有独立查询需求，JSON 列
    比多一张表 + JOIN 简单得多）。

    新增四个函数，让 StudentProfile 成为业务层和数据库之间的标准形状：
      - create_student_from_profile()  以 StudentProfile 建档（enrollment_bot 用）
      - profile_from_db()              数据库行还原成 StudentProfile
      - update_consultation_stage()    推进咨询阶段（api.py 的手动推进接口用）
      - advance_stage_to_planning()    阶段自动流转（agent_core 在学生首次
                                       触发简章问答后调用，只进不退）
    旧的 create_student() 保留不动，兼容既有调用（api.py 等）。

⚠️ 老库迁移：新列不会自动加到老表上。开发期直接删表重建即可；要保留
    已有数据，在 psql 里执行：
      ALTER TABLE students ADD COLUMN IF NOT EXISTS estimated_score INTEGER;
      ALTER TABLE students ADD COLUMN IF NOT EXISTS gaokao_rank INTEGER;
      ALTER TABLE students ADD COLUMN IF NOT EXISTS mock_scores JSON DEFAULT '[]';
      ALTER TABLE students ADD COLUMN IF NOT EXISTS competition_awards JSON DEFAULT '[]';
      ALTER TABLE students ADD COLUMN IF NOT EXISTS intended_major_directions JSON DEFAULT '[]';
      ALTER TABLE students ADD COLUMN IF NOT EXISTS target_path_types JSON DEFAULT '[]';
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, text
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine, select

from config import DATABASE_URL
from models import (
    CompetitionAward,
    ConsultationStage,
    Grade,
    PathType,
    StudentProfile,
    SubjectCombo,
)

engine = create_engine(DATABASE_URL, echo=False)


class StudentDB(SQLModel, table=True):
    __tablename__ = "students"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=20, index=True)
    grade: str
    province: str = Field(default="天津")
    subject_combo: str
    estimated_rank: Optional[int] = None
    estimated_score: Optional[int] = None
    gaokao_score: Optional[int] = None
    gaokao_rank: Optional[int] = None
    # 列表类字段存 JSON 列。default_factory 保证旧代码不传这些字段也能正常建档。
    mock_scores: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    competition_awards: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    intended_major_directions: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    target_path_types: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    has_competition_award: bool = Field(default=False)
    planner_notes: str = Field(default="")
    consultation_stage: str = Field(default="初次建档")
    planner_id: str = Field(default="default", index=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    consultations: list["ConsultationDB"] = Relationship(back_populates="student")


class ConsultationDB(SQLModel, table=True):
    __tablename__ = "consultations"

    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="students.id", index=True)
    topic: str
    content: str
    agent_response: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)

    student: Optional[StudentDB] = Relationship(back_populates="consultations")


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def check_connection() -> tuple[bool, str]:
    try:
        with Session(engine) as session:
            session.exec(select(1))
        return True, "PostgreSQL 连接正常"
    except Exception as exc:  # noqa: BLE001
        return False, f"PostgreSQL 连接失败：{exc}"


# ══════════════════════════════════════════════════════════════
# StudentProfile <-> StudentDB 转换（三期改造新增）
# ══════════════════════════════════════════════════════════════

def create_student_from_profile(profile: StudentProfile, planner_id: Optional[str] = None) -> StudentDB:
    """以 StudentProfile 为标准形状建档。

    字段校验（分数范围、省份、选科合法性等）在 Pydantic 模型层已经完成，
    这里只做"模型 → 表行"的映射，不再重复校验。
    planner_id 显式传入时优先，否则用 profile 里带的。
    """
    with Session(engine) as session:
        student = StudentDB(
            name=profile.name,
            grade=profile.grade.value,
            province=profile.province,
            subject_combo=profile.subject_combo.value,
            estimated_rank=profile.estimated_rank,
            estimated_score=profile.estimated_score,
            gaokao_score=profile.gaokao_score,
            gaokao_rank=profile.gaokao_rank,
            mock_scores=profile.mock_scores,
            competition_awards=[a.model_dump() for a in profile.competition_awards],
            intended_major_directions=profile.intended_major_directions,
            target_path_types=[p.value for p in profile.target_path_types],
            has_competition_award=bool(profile.competition_awards),
            planner_notes=profile.planner_notes,
            consultation_stage=profile.consultation_stage.value,
            planner_id=planner_id or profile.planner_id,
        )
        session.add(student)
        session.commit()
        session.refresh(student)
        return student


def profile_from_db(student: StudentDB) -> StudentProfile:
    """数据库行 → StudentProfile。需要完整档案形状时用（批量复盘、详情展示）。"""
    return StudentProfile(
        name=student.name,
        grade=Grade(student.grade),
        province=student.province,
        subject_combo=SubjectCombo(student.subject_combo),
        mock_scores=student.mock_scores or [],
        estimated_rank=student.estimated_rank,
        estimated_score=student.estimated_score,
        gaokao_score=student.gaokao_score,
        gaokao_rank=student.gaokao_rank,
        competition_awards=[CompetitionAward.model_validate(a) for a in student.competition_awards or []],
        intended_major_directions=student.intended_major_directions or [],
        target_path_types=[PathType(p) for p in student.target_path_types or []],
        planner_notes=student.planner_notes,
        consultation_stage=ConsultationStage(student.consultation_stage),
        planner_id=student.planner_id,
        created_date=student.created_at.date(),
    )


# ══════════════════════════════════════════════════════════════
# 咨询阶段流转（三期改造新增）
# ══════════════════════════════════════════════════════════════

def update_consultation_stage(
    student_id: int, stage: ConsultationStage, planner_id: Optional[str] = None
) -> bool:
    """把学生的咨询阶段推进到指定阶段。返回是否成功。

    planner_id 传入时会校验归属——api.py 的多规划师场景下，规划师不能
    改别人名下学生的阶段。
    """
    with Session(engine) as session:
        student = session.get(StudentDB, student_id)
        if student is None:
            return False
        if planner_id is not None and student.planner_id != planner_id:
            return False
        student.consultation_stage = stage.value
        student.updated_at = datetime.now()
        session.add(student)
        session.commit()
        return True


def advance_stage_to_planning(student_id: int) -> None:
    """阶段自动流转：仅在"初次建档"时推进到"规划进行中"。

    刻意只进不退——如果学生已经在"报名申请中"或"已完成"，绝不能因为
    又问了一次简章问题被拉回"规划进行中"。
    agent_core.py 在学生首次触发 answer_from_brochure（RAG 简章问答）后调用。
    """
    with Session(engine) as session:
        student = session.get(StudentDB, student_id)
        if student and student.consultation_stage == ConsultationStage.INITIAL.value:
            student.consultation_stage = ConsultationStage.PLANNING.value
            student.updated_at = datetime.now()
            session.add(student)
            session.commit()


# ══════════════════════════════════════════════════════════════
# 以下为原有接口，签名保持不变，兼容既有调用方
# ══════════════════════════════════════════════════════════════

def create_student(
    name: str,
    grade: str,
    subject_combo: str,
    estimated_rank: Optional[int] = None,
    planner_id: str = "default",
) -> StudentDB:
    with Session(engine) as session:
        student = StudentDB(
            name=name,
            grade=grade,
            subject_combo=subject_combo,
            estimated_rank=estimated_rank,
            planner_id=planner_id,
        )
        session.add(student)
        session.commit()
        session.refresh(student)
        return student


def get_student_by_name(name: str, planner_id: str = "default") -> Optional[StudentDB]:
    with Session(engine) as session:
        return session.exec(
            select(StudentDB).where(StudentDB.name == name, StudentDB.planner_id == planner_id)
        ).first()


def get_student_by_id(student_id: int) -> Optional[StudentDB]:
    with Session(engine) as session:
        return session.get(StudentDB, student_id)


def list_students(planner_id: Optional[str] = None) -> list[StudentDB]:
    with Session(engine) as session:
        stmt = select(StudentDB).order_by(StudentDB.updated_at.desc())
        if planner_id is not None:
            stmt = stmt.where(StudentDB.planner_id == planner_id)
        return list(session.exec(stmt))


def add_consultation(student_id: int, topic: str, content: str, agent_response: str = "") -> ConsultationDB:
    with Session(engine) as session:
        record = ConsultationDB(student_id=student_id, topic=topic, content=content, agent_response=agent_response)
        session.add(record)
        session.commit()
        session.refresh(record)
        return record


def get_consultations_for_student(student_id: int) -> list[ConsultationDB]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(ConsultationDB)
                .where(ConsultationDB.student_id == student_id)
                .order_by(ConsultationDB.created_at.desc())
            )
        )


# ══════════════════════════════════════════════════════════════
# Update / Delete（四期改造：完整 CRUD 新增）
#
# ⚠️ 为什么这里用 text() 原生 SQL 删 consultation_memories：
#    ConsultationMemory 表定义在 memory.py 里，而 memory.py 又反过来
#    从这里 import engine——database.py 如果 import memory.py 就构成
#    循环导入。用原生 SQL 按表名删，绕开这个依赖方向问题。
# ⚠️ 删除顺序固定为"先子后父"：长期记忆 → 咨询记录 → 学生行。
#    两张子表的外键都没配 ON DELETE CASCADE，直接删学生行会被
#    PostgreSQL 外键约束拦下报错。
# ══════════════════════════════════════════════════════════════

# 允许通过 update_student() 修改的字段白名单——id / planner_id / created_at
# 不在其中：主键不可改，归属不可转移，建档时间不可伪造。
UPDATABLE_STUDENT_FIELDS = {
    "name", "grade", "province", "subject_combo",
    "estimated_rank", "estimated_score", "gaokao_score", "gaokao_rank",
    "mock_scores", "competition_awards", "intended_major_directions",
    "target_path_types", "has_competition_award", "planner_notes",
}


def update_student(
    student_id: int, fields: dict, planner_id: Optional[str] = None
) -> Optional[StudentDB]:
    """通用档案更新。fields 里只有白名单字段会被应用，其余静默忽略。
    返回更新后的学生；学生不存在或不属于该规划师时返回 None。
    """
    with Session(engine) as session:
        student = session.get(StudentDB, student_id)
        if student is None:
            return None
        if planner_id is not None and student.planner_id != planner_id:
            return None
        for key, value in fields.items():
            if key in UPDATABLE_STUDENT_FIELDS:
                setattr(student, key, value)
        student.updated_at = datetime.now()
        session.add(student)
        session.commit()
        session.refresh(student)
        return student


def delete_consultation(consultation_id: int, planner_id: Optional[str] = None) -> bool:
    """删除单条咨询记录。返回是否成功。

    planner_id 传入时会 JOIN 学生表校验归属——咨询记录自己没有
    planner_id 字段，必须通过它的 parent 学生来确认"这条记录归谁管"。
    """
    with Session(engine) as session:
        record = session.get(ConsultationDB, consultation_id)
        if record is None:
            return False
        if planner_id is not None:
            student = session.get(StudentDB, record.student_id)
            if student is None or student.planner_id != planner_id:
                return False
        session.delete(record)
        session.commit()
        return True


def _delete_child_data(session: Session, student_id: int) -> None:
    """按"先子后父"顺序删掉一个学生的全部子表数据（长期记忆 + 咨询记录）。
    只在已确认学生存在且归属无误之后调用；本函数不 commit，由调用方统一提交。
    """
    session.exec(
        text("DELETE FROM consultation_memories WHERE student_id = :sid"), params={"sid": student_id}
    )
    session.exec(
        text("DELETE FROM consultations WHERE student_id = :sid"), params={"sid": student_id}
    )


def delete_student(student_id: int, planner_id: Optional[str] = None) -> bool:
    """删除学生档案：长期记忆 + 咨询记录 + 学生行全部删掉，不可恢复。

    级联是强制的——子表外键没配 ON DELETE CASCADE，留着子数据
    直接删学生行会报外键约束错误；就算没约束，孤儿数据也不该存在。
    """
    with Session(engine) as session:
        student = session.get(StudentDB, student_id)
        if student is None:
            return False
        if planner_id is not None and student.planner_id != planner_id:
            return False
        _delete_child_data(session, student_id)
        session.delete(student)
        session.commit()
        return True


def wipe_student_data(student_id: int, planner_id: Optional[str] = None) -> bool:
    """清空某学生的所有数据，但保留档案壳子（姓名、年级、选科等基本信息）。

    只删掉咨询记录和长期记忆，咨询阶段重置回"初次建档"——
    相当于"重新开始服务这个学生"。与 delete_student() 的区别：
    那个连档案一起消失，这个学生还在。
    """
    with Session(engine) as session:
        student = session.get(StudentDB, student_id)
        if student is None:
            return False
        if planner_id is not None and student.planner_id != planner_id:
            return False
        _delete_child_data(session, student_id)
        student.consultation_stage = ConsultationStage.INITIAL.value
        student.updated_at = datetime.now()
        session.add(student)
        session.commit()
        return True
