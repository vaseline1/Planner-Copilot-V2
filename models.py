"""models.py —— 全项目唯一的数据模型定义

职责：
    定义选科组合、年级、路径类型等枚举，学生档案、院校项目等 Pydantic 模型。
    这是整个项目的"词汇表"——任何文件要用到这些概念，都从这里 import，
    不允许各自重新定义一份（原方案代码里出现过 SubjectCombo 被独立定义
    两次的问题，这一版强制只保留一份）。

三期改造后的变化（原先"预留"的模型全部正式启用）：
    - StudentProfile 增补 estimated_score / planner_id 两个字段，成为建档
      流程与数据库之间的标准数据形状（enrollment_bot.py 组装它、
      database.py 的 create_student_from_profile() 消费它）；
    - CompetitionAward 不再只是摆设：建档 bot 会收集奖项明细入库，
      tools.py 用 is_qiangji_competition_eligible 逐条核验竞赛通道资格；
    - EligibilityResult 成为 tools.py 全部判断函数的统一返回类型，新增
      path_label / status_label / notes 字段和 to_display_dict()——内部
      代码读类型化字段，喂给大模型/展示给规划师时一键转中文键 dict；
    - ConversationTurn 被 memory.py 启用，Redis 里的对话历史带时间戳；
    - ConsultationStage 真正流转：database.py 提供推进函数，agent_core.py
      和 api.py 都会触发阶段变化。
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Grade(str, Enum):
    G10 = "高一"
    G11 = "高二"
    G12 = "高三"


class SubjectCombo(str, Enum):
    """天津新高考 3+3 选科组合（6选3，共20种）。
    命名规则：物=物理 化=化学 生=生物 史=历史 地=地理 政=政治，
    tools.py 判断"含不含某科"时直接用子串匹配，不再手写重复清单。
    """

    P_C_B = "物化生"
    P_C_H = "物化史"
    P_C_G = "物化地"
    P_C_P = "物化政"
    P_B_H = "物生史"
    P_B_G = "物生地"
    P_B_P = "物生政"
    P_H_G = "物史地"
    P_H_P = "物史政"
    P_G_P = "物地政"
    C_B_H = "化生史"
    C_B_G = "化生地"
    C_B_P = "化生政"
    C_H_G = "化史地"
    C_H_P = "化史政"
    C_G_P = "化地政"
    B_H_G = "生史地"
    B_H_P = "生史政"
    B_G_P = "生地政"
    H_G_P = "史地政"


class PathType(str, Enum):
    QIANGJI = "强基计划"
    COMP_EVAL = "综合评价"
    HKMO = "港澳升学"
    BAOSONG = "保送生"
    SCHOOL_SPECIAL = "高校专项计划"
    REGULAR = "普通高考"


class ConsultationStage(str, Enum):
    INITIAL = "初次建档"
    PLANNING = "规划进行中"
    APPLYING = "报名申请中"
    COMPLETED = "已完成"


class CompetitionAward(BaseModel):
    competition_name: str = Field(description="竞赛名称，如'全国化学奥林匹克联赛'")
    level: str = Field(description="获奖等级，如'全国决赛银牌（二等奖）'")
    year: int = Field(ge=2010, le=2035, description="获奖年份")
    subject: str = Field(description="对应学科")

    @property
    def is_qiangji_competition_eligible(self) -> bool:
        qualifying_levels = ["金牌", "银牌", "铜牌", "一等奖", "二等奖"]
        qualifying_scopes = ["全国决赛", "全国联赛", "国家级"]
        has_level = any(kw in self.level for kw in qualifying_levels)
        has_scope = any(
            kw in self.level or kw in self.competition_name
            for kw in qualifying_scopes
        )
        return has_level and has_scope


class StudentProfile(BaseModel):
    """完整学生档案——enrollment_bot.py 建档流程的标准产出形状。

    database.py 的 create_student_from_profile() 负责把它写入 StudentDB，
    profile_from_db() 负责反向还原：业务层只跟这个模型打交道，不直接
    拼数据库字段。
    """

    name: str = Field(min_length=2, max_length=20, description="学生姓名")
    grade: Grade
    province: str = Field(default="天津")
    subject_combo: SubjectCombo
    mock_scores: list[int] = Field(default_factory=list, description="历次模考成绩，按时间先后排列")
    estimated_rank: Optional[int] = Field(default=None, ge=1, le=300_000)
    estimated_score: Optional[int] = Field(default=None, ge=0, le=750, description="预估高考总分（非实考成绩）")
    gaokao_score: Optional[int] = Field(default=None, ge=0, le=750)
    gaokao_rank: Optional[int] = Field(default=None, ge=1)
    competition_awards: list[CompetitionAward] = Field(default_factory=list)
    intended_major_directions: list[str] = Field(default_factory=list)
    target_path_types: list[PathType] = Field(default_factory=list)
    planner_notes: str = Field(default="")
    consultation_stage: ConsultationStage = Field(default=ConsultationStage.INITIAL)
    planner_id: str = Field(default="default")
    created_date: Optional[date] = None

    @field_validator("province")
    @classmethod
    def _province_must_be_tianjin(cls, v: str) -> str:
        if v != "天津":
            raise ValueError(f"当前版本仅支持天津市，收到：{v}")
        return v

    @field_validator("mock_scores")
    @classmethod
    def _scores_in_range(cls, v: list[int]) -> list[int]:
        for score in v:
            if not (0 <= score <= 750):
                raise ValueError(f"分数必须在 0-750 之间，收到：{score}")
        return v


class UniversityProgram(BaseModel):
    """院校升学项目数据——knowledge_base.py / rag_system.py 共用的数据形状。

    ⚠️ 注意：这个模型只定义"数据长什么样"，具体数据内容（招生名额、
    分数线、报名截止日期）需要你从院校官方最新简章核实后自己填，
    见 data/tianjin_programs.json 里的说明。
    """

    university_name: str
    path_type: PathType
    province_scope: list[str] = Field(description="招生面向省份，含'天津'才对本项目适用")
    majors: list[str] = Field(default_factory=list)
    subject_combo_required: list[SubjectCombo] = Field(default_factory=list)
    has_physics_requirement: Optional[bool] = None
    score_requirement_desc: str = Field(default="")
    competition_requirement: Optional[str] = None
    campus_test_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    enrollment_tianjin: Optional[int] = None
    application_deadline: Optional[str] = None
    source_year: int
    source_url: str = ""


class EligibilityResult(BaseModel):
    """单条升学路径的资格判断结果——tools.py 全部判断函数的统一返回类型。

    为什么要同时保留 is_eligible（布尔）和 status_label（中文结论）：
        "初步资格"的中文表述是各工具自己的业务语言（强基说"满足基本条件/
        存在缺失条件"，综评说"分数线参考不满足"，港澳说"分数仍有差距"），
        无法从布尔值机械推导。内部逻辑判断用 is_eligible，展示/喂模型用
        status_label，两个字段各管一段，互不迁就。

    喂给大模型或展示给规划师时，调用 to_display_dict() 得到中文键 dict，
    键名和旧版 tools.py 返回的散装 dict 完全一致（路径/初步资格/满足条件/
    缺失条件/建议/数据说明），下游展示代码不用改习惯。
    """

    path_type: PathType
    path_label: str = Field(
        default="",
        description="展示用路径名，如'综合评价（南开/天大）'；为空时用 path_type.value",
    )
    target_university: Optional[str] = None
    is_eligible: bool
    status_label: str = Field(description="资格结论的中文表述，如'满足基本条件'")
    conditions_met: list[str] = Field(default_factory=list)
    conditions_missing: list[str] = Field(default_factory=list)
    recommended_action: str
    notes: list[str] = Field(
        default_factory=list,
        description="补充说明（如'奖项未达强基竞赛通道认定标准'这类不属于缺失条件的提示）",
    )
    data_note: str = Field(default="基于粗略参考阈值，正式规划需核查当年最新简章")
    requires_verification: bool = True

    @property
    def display_path(self) -> str:
        return self.path_label or self.path_type.value

    def to_display_dict(self) -> dict:
        """转成中文键 dict：agent_core 序列化工具结果、bot 拼展示文案时用。"""
        result: dict = {
            "路径": self.display_path,
            "初步资格": self.status_label,
            "满足条件": self.conditions_met,
            "缺失条件": self.conditions_missing,
            "建议": self.recommended_action,
            "数据说明": self.data_note,
        }
        if self.target_university:
            result["目标院校"] = self.target_university
        if self.notes:
            result["备注"] = self.notes
        return result


class ConversationTurn(BaseModel):
    """一轮对话——memory.py 的 Redis 会话历史以此形状存储。

    timestamp 不参与 LLM 上下文（喂给大模型前 memory.py 会剥掉，只留
    role/content），它服务于排查问题："这段历史是什么时候聊的"。
    """

    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
