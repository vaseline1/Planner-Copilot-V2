"""test_tools.py —— tools.py 的单元测试

这里测的全是纯逻辑，不需要数据库、Redis、API Key，运行成本最低，
建议每次改完 tools.py 都跑一遍：uv run pytest test_tools.py -v

三期改造同步：返回值从散装 dict 换成 EligibilityResult，断言改为读
类型化字段（is_eligible / status_label / path_type）；喂给大模型的
中文键格式用 to_display_dict() 单独验证，保证下游展示习惯不变。
"""
import pytest

from models import CompetitionAward, Grade, PathType, StudentProfile, SubjectCombo
from tools import (
    check_all_paths,
    check_all_paths_for_student,
    check_comprehensive_eval,
    check_hkmo_eligibility,
    check_qiangji_eligibility,
    has_biology,
    has_chemistry,
    has_history,
    has_physics,
    is_pure_humanities,
)

EXPECTED_FLAGS = {
    combo: ("物" in combo.value, "化" in combo.value, "生" in combo.value, "史" in combo.value)
    for combo in SubjectCombo
}


class TestSubjectComboFlags:
    @pytest.mark.parametrize("combo", list(SubjectCombo))
    def test_all_combos_against_ground_truth(self, combo: SubjectCombo) -> None:
        expected_physics, expected_chem, expected_bio, expected_hist = EXPECTED_FLAGS[combo]
        assert has_physics(combo) == expected_physics, combo.value
        assert has_chemistry(combo) == expected_chem, combo.value
        assert has_biology(combo) == expected_bio, combo.value
        assert has_history(combo) == expected_hist, combo.value

    def test_physics_combo_count_is_ten(self) -> None:
        assert sum(1 for c in SubjectCombo if has_physics(c)) == 10

    def test_pure_humanities_is_only_shidizheng(self) -> None:
        assert [c for c in SubjectCombo if is_pure_humanities(c)] == [SubjectCombo.H_G_P]


class TestQiangjiEligibility:
    def test_high_rank_no_award_eligible(self) -> None:
        r = check_qiangji_eligibility("物化生", 1500, False, "清华大学")
        assert r.is_eligible is True
        assert r.status_label == "满足基本条件"
        assert r.path_type == PathType.QIANGJI
        assert r.target_university == "清华大学"

    def test_low_rank_no_physics_not_eligible(self) -> None:
        r = check_qiangji_eligibility("史地政", 50000, False)
        assert r.is_eligible is False
        assert r.conditions_missing

    def test_competition_award_overrides_low_rank(self) -> None:
        assert check_qiangji_eligibility("物化生", 100000, True).is_eligible is True

    def test_invalid_combo_raises(self) -> None:
        with pytest.raises(ValueError):
            check_qiangji_eligibility("语数外", 1000, False)

    def test_qualifying_award_detail_enables_competition_channel(self) -> None:
        """奖项明细达到认定标准 → 即使位次不够也能走竞赛通道。"""
        awards = [
            CompetitionAward(
                competition_name="全国化学奥林匹克联赛", level="全国决赛二等奖", year=2024, subject="化学"
            )
        ]
        r = check_qiangji_eligibility("物化生", 100000, False, competition_awards=awards)
        assert r.is_eligible is True
        assert any("竞赛通道" in line for line in r.conditions_met)

    def test_non_qualifying_award_falls_back_to_rank_with_note(self) -> None:
        """奖项不够格 → 按无竞赛通道评估，且 notes 里有明确提示。"""
        awards = [
            CompetitionAward(competition_name="校级数学竞赛", level="校级一等奖", year=2024, subject="数学")
        ]
        r = check_qiangji_eligibility("物化生", 100000, False, competition_awards=awards)
        assert r.is_eligible is False
        assert any("未达强基竞赛通道" in note for note in r.notes)
        assert "备注" in r.to_display_dict()


class TestComprehensiveEval:
    def test_within_threshold_eligible(self) -> None:
        r = check_comprehensive_eval("化生史", 5000, "高二")
        assert r.is_eligible is True
        assert r.status_label == "满足基本条件"
        assert r.path_type == PathType.COMP_EVAL
        assert r.display_path == "综合评价（南开/天大）"

    def test_beyond_threshold_not_eligible(self) -> None:
        r = check_comprehensive_eval("化生史", 50000, "高二")
        assert r.is_eligible is False
        assert r.status_label == "分数线参考不满足"


class TestHkmoEligibility:
    def test_high_score_eligible(self) -> None:
        r = check_hkmo_eligibility(650, "物化生")
        assert r.is_eligible is True
        assert r.status_label == "基本满足"
        assert r.path_type == PathType.HKMO

    def test_low_score_not_eligible(self) -> None:
        r = check_hkmo_eligibility(400, "物化生")
        assert r.is_eligible is False
        assert r.status_label == "分数仍有差距"


class TestCheckAllPaths:
    def test_returns_three_paths(self) -> None:
        results = check_all_paths("物化生", "高二", 1500, 620, False)
        assert len(results) == 3
        assert {r.path_type for r in results} == {PathType.QIANGJI, PathType.COMP_EVAL, PathType.HKMO}

    def test_display_dict_keeps_chinese_keys_for_llm(self) -> None:
        """喂给大模型/展示给规划师的 dict，键名与旧版散装 dict 完全一致。"""
        r = check_all_paths("物化生", "高二", 1500, 620, False)[0]
        d = r.to_display_dict()
        assert {"路径", "初步资格", "满足条件", "缺失条件", "建议", "数据说明"}.issubset(d.keys())

    def test_invalid_grade_raises(self) -> None:
        with pytest.raises(ValueError):
            check_all_paths("物化生", "大一", 1000, 620, False)

    def test_invalid_rank_raises(self) -> None:
        with pytest.raises(ValueError):
            check_all_paths("物化生", "高三", -1, 620, False)


class TestCheckAllPathsForStudent:
    """三期改造新增：以 StudentProfile 为输入的一站式判断。"""

    def test_profile_input_returns_three_paths(self) -> None:
        profile = StudentProfile(
            name="测试学生",
            grade=Grade.G11,
            subject_combo=SubjectCombo.P_C_B,
            estimated_rank=1500,
            estimated_score=620,
        )
        results = check_all_paths_for_student(profile)
        assert len(results) == 3
        assert all(r.is_eligible for r in results)

    def test_profile_awards_verified_one_by_one(self) -> None:
        """档案里的奖项明细会被逐条核验，够格的才给竞赛通道。"""
        profile = StudentProfile(
            name="测试学生",
            grade=Grade.G12,
            subject_combo=SubjectCombo.P_C_B,
            estimated_rank=100000,
            estimated_score=580,
            competition_awards=[
                CompetitionAward(
                    competition_name="全国中学生物理竞赛", level="全国决赛铜牌", year=2024, subject="物理"
                ),
                CompetitionAward(competition_name="校级英语演讲比赛", level="校级一等奖", year=2024, subject="英语"),
            ],
        )
        qiangji = check_all_paths_for_student(profile)[0]
        assert qiangji.is_eligible is True  # 物理铜牌够格，竞赛通道成立
        assert any("物理竞赛" in line for line in qiangji.conditions_met)
        assert any("英语演讲" in note for note in qiangji.notes)  # 校级奖写进备注

    def test_missing_score_and_rank_raises(self) -> None:
        profile = StudentProfile(name="测试学生", grade=Grade.G11, subject_combo=SubjectCombo.P_C_B)
        with pytest.raises(ValueError):
            check_all_paths_for_student(profile)
