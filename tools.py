"""tools.py —— 天津升学路径判断的核心业务逻辑 + Function Calling schema

职责：
    选科相关判断（含不含物理/化学/生物/历史）+ 三条路径的资格初判
    （强基计划/综合评价/港澳升学）。这部分是纯规则判断，不依赖数据库、
    不依赖知识库，是整个 Agent 里唯一"不会因为外部服务挂了而出错"的模块。

    knowledge_base.py（关键词检索）和 rag_system.py（简章问答）是另外
    两类工具，各自在自己的文件里定义、在 agent_core.py 里统一注册，
    没有放进这个文件——因为它们依赖外部数据/服务，混在一起会让这个
    本该"零依赖、随时能跑"的文件也被拖下水。

三期改造后的变化：
    - 三个判断函数和 check_all_paths 的返回值从散装 dict 换成
      EligibilityResult（models.py）：内部代码读 is_eligible / path_type
      等类型化字段；要喂给大模型或展示时调 to_display_dict()，得到的
      中文键 dict 与旧版完全一致；
    - check_qiangji_eligibility 新增可选参数 competition_awards：传入
      奖项明细（CompetitionAward 列表）时逐条核验是否达到强基竞赛通道
      认定标准，比单个布尔值精确；LLM 工具调用仍走 has_competition_award
      布尔值（Function Calling 参数保持简单，调用成功率高）；
    - 新增 check_all_paths_for_student()：以 StudentProfile 为输入的
      一站式路径判断，建档、批量复盘场景直接用。
"""
from __future__ import annotations

from typing import Optional

from models import CompetitionAward, EligibilityResult, PathType, StudentProfile, SubjectCombo

PHYSICS_COMBOS = frozenset(c for c in SubjectCombo if "物" in c.value)
CHEMISTRY_COMBOS = frozenset(c for c in SubjectCombo if "化" in c.value)
BIOLOGY_COMBOS = frozenset(c for c in SubjectCombo if "生" in c.value)
HISTORY_COMBOS = frozenset(c for c in SubjectCombo if "史" in c.value)

TIANJIN_QIANGJI_ROUGH_THRESHOLD = 3000
TIANJIN_COMP_EVAL_ROUGH_THRESHOLD = 8000
HKMO_SCORE_THRESHOLD = 600


def has_physics(combo: SubjectCombo) -> bool:
    return combo in PHYSICS_COMBOS


def has_chemistry(combo: SubjectCombo) -> bool:
    return combo in CHEMISTRY_COMBOS


def has_biology(combo: SubjectCombo) -> bool:
    return combo in BIOLOGY_COMBOS


def has_history(combo: SubjectCombo) -> bool:
    return combo in HISTORY_COMBOS


def is_pure_humanities(combo: SubjectCombo) -> bool:
    return combo is SubjectCombo.H_G_P


def _parse_combo(subject_combo_str: str) -> SubjectCombo:
    try:
        return SubjectCombo(subject_combo_str)
    except ValueError as exc:
        valid = "、".join(c.value for c in SubjectCombo)
        raise ValueError(f"'{subject_combo_str}' 不是有效的天津选科组合。有效组合：{valid}") from exc


def check_qiangji_eligibility(
    subject_combo_str: str,
    estimated_rank: int,
    has_competition_award: bool = False,
    target_university: str = "综合判断",
    competition_awards: Optional[list[CompetitionAward]] = None,
) -> EligibilityResult:
    """强基计划资格初判。

    competition_awards 传入奖项明细时，用 CompetitionAward 自带的
    is_qiangji_competition_eligible 逐条核验——哪些奖够格走竞赛通道、
    哪些不够格（写进 notes 提示），比单个布尔值精确得多。
    没传明细时退化为按 has_competition_award 布尔值判断（LLM 工具调用
    和旧代码走这条路，行为与旧版一致）。
    """
    combo = _parse_combo(subject_combo_str)
    conditions_met: list[str] = []
    conditions_missing: list[str] = []
    notes: list[str] = []

    award_ok = has_competition_award
    if competition_awards:
        eligible_awards = [a for a in competition_awards if a.is_qiangji_competition_eligible]
        award_ok = bool(eligible_awards)
        for award in eligible_awards:
            conditions_met.append(
                f"竞赛奖项「{award.competition_name}·{award.level}（{award.year}年）」达到强基竞赛通道认定参考标准"
            )
        for award in competition_awards:
            if not award.is_qiangji_competition_eligible:
                notes.append(
                    f"奖项「{award.competition_name}·{award.level}」未达强基竞赛通道参考标准"
                    "（全国决赛二等奖及以上），本次按无竞赛通道评估"
                )

    if award_ok:
        if not competition_awards:
            conditions_met.append("具备竞赛奖项，可走竞赛通道（分数门槛大幅降低）")
    elif estimated_rank <= TIANJIN_QIANGJI_ROUGH_THRESHOLD:
        conditions_met.append(f"预估位次 {estimated_rank}，满足多数强基院校基本分数线参考")
    else:
        conditions_missing.append(
            f"预估位次 {estimated_rank}，超出多数强基分数线参考阈值（约前 {TIANJIN_QIANGJI_ROUGH_THRESHOLD} 名）"
        )

    physics, chem, bio = has_physics(combo), has_chemistry(combo), has_biology(combo)
    pure_hum = is_pure_humanities(combo)
    if physics and chem and bio:
        conditions_met.append("理工医类全部强基专业可报")
    elif physics:
        conditions_met.append("数学/物理/工学类强基可报")
        conditions_missing.append("不含化学或生物，医学类强基部分受限，需核查目标院校简章")
    elif pure_hum:
        conditions_met.append("人文类强基专业可报（汉语言/历史/哲学/考古）")
        conditions_missing.append("不含物理，理工医类全部强基专业不可报")
    else:
        conditions_missing.append(f"选科 {combo.value} 不含物理，多数强基专业受限，仅限部分人文类")

    is_eligible = (not conditions_missing) or award_ok
    return EligibilityResult(
        path_type=PathType.QIANGJI,
        target_university=None if target_university == "综合判断" else target_university,
        is_eligible=is_eligible,
        status_label="满足基本条件" if is_eligible else "存在缺失条件",
        conditions_met=conditions_met,
        conditions_missing=conditions_missing,
        recommended_action=(
            f"建议研究{target_university}强基简章，核实校测规则和当年天津名额"
            if is_eligible
            else "建议评估综合评价路径，或针对竞赛通道制定备赛计划"
        ),
        notes=notes,
        data_note="基于粗略参考阈值，正式规划需核查当年最新简章",
    )


def check_comprehensive_eval(subject_combo_str: str, estimated_rank: int, grade: str) -> EligibilityResult:
    """综合评价路径判断（天津重点：南开大学、天津大学在津设综评批次）。"""
    combo = _parse_combo(subject_combo_str)
    conditions_met: list[str] = []
    conditions_missing: list[str] = []

    if estimated_rank <= TIANJIN_COMP_EVAL_ROUGH_THRESHOLD:
        conditions_met.append(f"预估位次 {estimated_rank}，满足南开/天大综评基本分数线参考")
    else:
        conditions_missing.append(
            f"预估位次 {estimated_rank}，超出南开/天大综评参考阈值（约前 {TIANJIN_COMP_EVAL_ROUGH_THRESHOLD} 名）"
        )

    if has_physics(combo):
        conditions_met.append("含物理，工学/理学类综评专业无限制")
    else:
        conditions_met.append("综评对选科要求较灵活，但具体专业仍有选科限制，需查简章")

    if grade in ("高一", "高二"):
        conditions_met.append(f"当前{grade}，有时间准备综评所需的综合素质材料")

    is_eligible = estimated_rank <= TIANJIN_COMP_EVAL_ROUGH_THRESHOLD
    return EligibilityResult(
        path_type=PathType.COMP_EVAL,
        path_label="综合评价（南开/天大）",
        is_eligible=is_eligible,
        status_label="满足基本条件" if is_eligible else "分数线参考不满足",
        conditions_met=conditions_met,
        conditions_missing=conditions_missing,
        recommended_action=(
            "建议关注南开/天大综评报名时间（通常3-4月），提前准备综合素质证明材料"
            if is_eligible
            else "当前位次偏低，建议优先提升成绩，同时了解综评流程以备后续"
        ),
        data_note="南开/天大综评每年招生计划有变，需查当年招生简章",
    )


def check_hkmo_eligibility(estimated_score: int, subject_combo_str: str) -> EligibilityResult:
    """港澳升学基础判断（天津考生参加 JUPAS 联招或独立申请）。"""
    combo = _parse_combo(subject_combo_str)
    conditions_met: list[str] = []
    conditions_missing: list[str] = []

    if estimated_score >= HKMO_SCORE_THRESHOLD:
        conditions_met.append(f"预估分数 {estimated_score}，满足港澳主流院校基本分数参考")
    else:
        conditions_missing.append(f"预估分数 {estimated_score}，港澳主流院校参考线约 {HKMO_SCORE_THRESHOLD} 分以上")

    conditions_met.append("天津高考成绩可用于港澳申请（JUPAS 联招通常 3 月报名）")
    if not is_pure_humanities(combo):
        conditions_met.append("选科覆盖理/工/商类，港澳院校专业选择范围较广")

    is_eligible = estimated_score >= HKMO_SCORE_THRESHOLD
    return EligibilityResult(
        path_type=PathType.HKMO,
        is_eligible=is_eligible,
        status_label="基本满足" if is_eligible else "分数仍有差距",
        conditions_met=conditions_met,
        conditions_missing=conditions_missing,
        recommended_action=(
            "建议了解 JUPAS 联招时间表，同步准备英语成绩（港校通常要求）"
            if is_eligible
            else "可作为冲刺目标，建议重点提升英语，关注港澳分数线变化"
        ),
        data_note="港澳各院校分数参考差异较大，需查具体院校要求",
    )


def check_all_paths(
    subject_combo_str: str,
    grade: str,
    estimated_rank: int,
    estimated_score: int,
    has_competition_award: bool = False,
    competition_awards: Optional[list[CompetitionAward]] = None,
) -> list[EligibilityResult]:
    """一次输入，返回三条路径的资格判断——规划师最常用的入口函数。"""
    _parse_combo(subject_combo_str)
    if grade not in ("高一", "高二", "高三"):
        raise ValueError(f"年级必须是 高一/高二/高三，收到：{grade}")
    if estimated_rank <= 0:
        raise ValueError("位次必须是正整数")
    return [
        check_qiangji_eligibility(
            subject_combo_str, estimated_rank, has_competition_award,
            competition_awards=competition_awards,
        ),
        check_comprehensive_eval(subject_combo_str, estimated_rank, grade),
        check_hkmo_eligibility(estimated_score, subject_combo_str),
    ]


def check_all_paths_for_student(profile: StudentProfile) -> list[EligibilityResult]:
    """以 StudentProfile 为输入的一站式路径判断（三期改造新增）。

    分数/位次按"预估优先、实考其次、模考兜底"的顺序取——建档后还没
    高考时用的是预估数据，出分后档案更新了实考数据也能直接用。
    竞赛奖项明细自动逐条核验强基竞赛通道资格。
    """
    score = profile.estimated_score or profile.gaokao_score or (
        profile.mock_scores[-1] if profile.mock_scores else None
    )
    if score is None:
        raise ValueError(f"学生 {profile.name} 没有任何可用分数（预估分/高考分/模考成绩均为空）")
    rank = profile.estimated_rank or profile.gaokao_rank
    if rank is None:
        raise ValueError(f"学生 {profile.name} 没有可用位次（预估位次/高考位次均为空）")
    return check_all_paths(
        subject_combo_str=profile.subject_combo.value,
        grade=profile.grade.value,
        estimated_rank=rank,
        estimated_score=score,
        has_competition_award=bool(profile.competition_awards),
        competition_awards=profile.competition_awards or None,
    )


# ══════════════════════════════════════════════════════════════
# Function Calling Schema（这3个工具不依赖外部数据/服务）
# Schema 与旧版完全一致：参数保持扁平简单（布尔值而非奖项列表），
# LLM 的调用成功率高；奖项明细核验是建档/bot 内部路径的能力。
# ══════════════════════════════════════════════════════════════
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_qiangji_eligibility",
            "description": "判断天津学生是否满足报名强基计划的基本条件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_combo_str": {"type": "string", "description": "天津3+3选科组合，如'物化生'"},
                    "estimated_rank": {"type": "integer", "description": "天津市预估位次"},
                    "has_competition_award": {"type": "boolean", "description": "是否有五大联赛全国决赛二等奖及以上"},
                    "target_university": {"type": "string", "description": "目标院校，如'南开大学'"},
                },
                "required": ["subject_combo_str", "estimated_rank", "has_competition_award", "target_university"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_comprehensive_eval",
            "description": "判断天津学生是否满足南开大学/天津大学综合评价报名条件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_combo_str": {"type": "string"},
                    "estimated_rank": {"type": "integer"},
                    "grade": {"type": "string", "enum": ["高一", "高二", "高三"]},
                },
                "required": ["subject_combo_str", "estimated_rank", "grade"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_hkmo_eligibility",
            "description": "判断天津学生的港澳升学基础资格。",
            "parameters": {
                "type": "object",
                "properties": {
                    "estimated_score": {"type": "integer", "description": "预估高考总分"},
                    "subject_combo_str": {"type": "string"},
                },
                "required": ["estimated_score", "subject_combo_str"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "check_qiangji_eligibility": check_qiangji_eligibility,
    "check_comprehensive_eval": check_comprehensive_eval,
    "check_hkmo_eligibility": check_hkmo_eligibility,
}
