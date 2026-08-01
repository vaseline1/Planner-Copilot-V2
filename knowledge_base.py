"""knowledge_base.py —— 天津升学政策本地知识库（关键词检索版）

职责：
    从 data/tianjin_programs.json 加载院校项目数据，提供关键词搜索
    （"南开强基""天津有哪些综评"这类问题）。这是"先粗后精"里"粗"的
    那一层——不需要 PDF、不需要向量数据库，能覆盖大部分"有没有""是
    什么"类型的问题；细节问题（"校测占比具体多少"）交给 rag_system.py。

⚠️ 关于数据本身：
    data/tianjin_programs.json 里目前只有 3 条来自原始学习方案的示例
    记录，用来验证代码能跑通、格式对不对。**这不是可以直接使用的真实
    数据**——具体的招生名额、分数线、报名截止日期每年都会变，必须由你
    对照当年官方简章核实后再录入。原方案的要求是"至少10条有效记录"，
    这部分内容本身不属于代码，我不会替你编造。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import KNOWLEDGE_BASE_PATH
from models import PathType, SubjectCombo, UniversityProgram

PHYSICS_COMBO_VALUES = {c.value for c in SubjectCombo if "物" in c.value}


def load_programs(path: Optional[str] = None) -> list[UniversityProgram]:
    """从 JSON 文件加载全部院校项目数据。文件不存在时返回空列表（不报错）。"""
    kb_path = Path(path or KNOWLEDGE_BASE_PATH)
    if not kb_path.exists():
        return []
    raw = json.loads(kb_path.read_text(encoding="utf-8"))
    return [UniversityProgram.model_validate(item) for item in raw]


def search_programs(
    query: str,
    subject_combo: Optional[SubjectCombo] = None,
    path_type: Optional[PathType] = None,
    province: str = "天津",
) -> list[dict]:
    """关键词检索院校项目。

    Args:
        query: 搜索关键词，如"南开""强基""综合评价"
        subject_combo: 按学生选科过滤（可选）
        path_type: 按路径类型过滤（可选）
        province: 必须在招生省份列表里包含该省份
    """
    programs = load_programs()
    results: list[dict] = []
    query_chars = set(query.replace(" ", ""))

    for prog in programs:
        if province not in prog.province_scope:
            continue
        if path_type and prog.path_type != path_type:
            continue
        if subject_combo and prog.subject_combo_required:
            if subject_combo not in prog.subject_combo_required:
                if prog.has_physics_requirement is True and subject_combo.value not in PHYSICS_COMBO_VALUES:
                    continue

        searchable_text = (
            prog.university_name + prog.path_type.value + "".join(prog.majors) + prog.score_requirement_desc
        )
        text_chars = set(searchable_text)
        if query_chars and query_chars.issubset(text_chars):
            results.append(
                {
                    "院校": prog.university_name,
                    "路径类型": prog.path_type.value,
                    "招生专业": prog.majors,
                    "天津名额": prog.enrollment_tianjin,
                    "分数要求": prog.score_requirement_desc,
                    "竞赛要求": prog.competition_requirement,
                    "校测占比": f"{prog.campus_test_ratio * 100:.0f}%" if prog.campus_test_ratio else "待查",
                    "报名时间": prog.application_deadline,
                    "数据年份": prog.source_year,
                    "⚠ 提示": "以上数据供参考，请规划师核查最新简章",
                }
            )
    return results


def get_tianjin_summary() -> dict:
    """获取天津升学路径全局概览（规划师快速了解全局用）。"""
    programs = load_programs()
    tianjin_programs = [p for p in programs if "天津" in p.province_scope]
    qiangji = [p for p in tianjin_programs if p.path_type == PathType.QIANGJI]
    comp_eval = [p for p in tianjin_programs if p.path_type == PathType.COMP_EVAL]

    return {
        "天津强基计划": {
            "已收录院校数": len(set(p.university_name for p in qiangji)),
            "院校列表": sorted(set(p.university_name for p in qiangji)),
            "说明": "全国36所强基院校均在津招生（本地知识库目前只收录了示例数据，未覆盖全部36所）",
        },
        "天津综合评价": {
            "已收录院校数": len(set(p.university_name for p in comp_eval)),
            "院校列表": sorted(set(p.university_name for p in comp_eval)),
            "说明": "南开/天大在津设独立综评批次",
        },
        "港澳升学": {"说明": "天津考生参加 JUPAS 联招或独立申请，通常 3 月报名"},
    }
