"""test_enrollment_bot.py —— enrollment_bot.py 状态机的单元测试

需要能连上 PostgreSQL（建档最后一步会真的写数据库）。
本地用 SQLite 冒烟也行：DATABASE_URL=sqlite:///./test.db uv run pytest test_enrollment_bot.py -v

三期改造同步：流程从 8 步扩为 9 步 + 奖项明细子流程，测试输入序列
已按新流程更新，并新增"完整档案字段真的落库"的断言。
"""
import database
from enrollment_bot import ConversationState, EnrollmentBot

database.init_db()


class TestHappyPath:
    def test_full_flow_reaches_complete(self) -> None:
        bot = EnrollmentBot(planner_id="test_planner")
        bot.process("")
        for inp in ["测试学生甲", "高二", "物化生", "2500", "620", "无", "没有", "计算机", "强基计划、综合评价", "确认"]:
            bot.process(inp)
        assert bot.state == ConversationState.COMPLETE

    def test_award_detail_flow_persists_full_profile(self) -> None:
        """含奖项明细的完整流程：模考/奖项/意向/目标路径全部落库。"""
        bot = EnrollmentBot(planner_id="test_planner")
        bot.process("")
        inputs = [
            "测试学生乙", "高一", "物化生", "3000", "610", "600、615",
            "有", "全国化学奥林匹克联赛，全国决赛二等奖，2024，化学", "完成",
            "待定", "待定", "确认",
        ]
        reply = ""
        for inp in inputs:
            reply = bot.process(inp)
        assert bot.state == ConversationState.COMPLETE
        assert "ID" in reply

        student = database.get_student_by_name("测试学生乙", planner_id="test_planner")
        assert student is not None
        assert student.mock_scores == [600, 615]
        assert len(student.competition_awards) == 1
        assert student.competition_awards[0]["competition_name"] == "全国化学奥林匹克联赛"
        assert student.has_competition_award is True
        assert student.estimated_score == 610

    def test_multiple_awards_all_saved(self) -> None:
        """奖项明细子流程支持逐条录入多条。"""
        bot = EnrollmentBot(planner_id="test_planner")
        bot.process("")
        inputs = [
            "测试学生丙", "高三", "物化生", "1500", "650", "无",
            "有",
            "全国中学生物理竞赛，全国决赛铜牌，2024，物理",
            "全国数学奥林匹克联赛，全国联赛一等奖，2023，数学",
            "完成",
            "数学、物理", "强基计划", "确认",
        ]
        for inp in inputs:
            bot.process(inp)
        assert bot.state == ConversationState.COMPLETE
        student = database.get_student_by_name("测试学生丙", planner_id="test_planner")
        assert len(student.competition_awards) == 2
        assert student.intended_major_directions == ["数学", "物理"]
        assert student.target_path_types == ["强基计划"]

    def test_majors_and_paths_actually_persisted(self) -> None:
        """旧版遗漏修复验证：意向专业、目标路径不再被丢弃。"""
        bot = EnrollmentBot(planner_id="test_planner")
        bot.process("")
        for inp in ["测试学生丁", "高二", "史地政", "8000", "580", "无", "没有", "汉语言文学、历史", "港澳升学", "确认"]:
            bot.process(inp)
        assert bot.state == ConversationState.COMPLETE
        student = database.get_student_by_name("测试学生丁", planner_id="test_planner")
        assert student.intended_major_directions == ["汉语言文学", "历史"]
        assert student.target_path_types == ["港澳升学"]


class TestValidation:
    def test_name_too_short_rejected(self) -> None:
        bot = EnrollmentBot()
        bot.process("")
        reply = bot.process("张")
        assert bot.state == ConversationState.ASK_NAME
        assert "❌" in reply

    def test_invalid_grade_rejected(self) -> None:
        bot = EnrollmentBot()
        bot.process("")
        bot.process("合法姓名")
        reply = bot.process("大四")
        assert bot.state == ConversationState.ASK_GRADE
        assert "❌" in reply

    def test_invalid_combo_rejected(self) -> None:
        bot = EnrollmentBot()
        bot.process("")
        bot.process("合法姓名")
        bot.process("高三")
        reply = bot.process("语数外")
        assert bot.state == ConversationState.ASK_COMBO
        assert "❌" in reply

    def test_rank_out_of_range_rejected(self) -> None:
        bot = EnrollmentBot()
        bot.process("")
        bot.process("合法姓名")
        bot.process("高三")
        bot.process("物化生")
        reply = bot.process("-5")
        assert bot.state == ConversationState.ASK_RANK
        assert "❌" in reply

    def test_mock_scores_bad_format_rejected(self) -> None:
        bot = EnrollmentBot()
        bot.process("")
        for inp in ["合法姓名", "高三", "物化生", "1000", "650"]:
            bot.process(inp)
        reply = bot.process("abc")
        assert bot.state == ConversationState.ASK_MOCK_SCORES
        assert "❌" in reply

    def test_award_detail_bad_format_rejected(self) -> None:
        bot = EnrollmentBot()
        bot.process("")
        for inp in ["合法姓名", "高三", "物化生", "1000", "650", "无", "有"]:
            bot.process(inp)
        reply = bot.process("格式不对的一条")
        assert bot.state == ConversationState.ASK_AWARD_DETAIL
        assert "❌" in reply

    def test_award_detail_finish_without_any_award_rejected(self) -> None:
        """答了'有'却一条都没录就说'完成'——不放行。"""
        bot = EnrollmentBot()
        bot.process("")
        for inp in ["合法姓名", "高三", "物化生", "1000", "650", "无", "有"]:
            bot.process(inp)
        reply = bot.process("完成")
        assert bot.state == ConversationState.ASK_AWARD_DETAIL
        assert "❌" in reply

    def test_invalid_path_type_rejected(self) -> None:
        bot = EnrollmentBot()
        bot.process("")
        for inp in ["合法姓名", "高三", "物化生", "1000", "650", "无", "没有", "计算机"]:
            bot.process(inp)
        reply = bot.process("出国留学")
        assert bot.state == ConversationState.ASK_PATHS
        assert "❌" in reply


class TestRetryLimit:
    def test_max_retry_terminates(self) -> None:
        bot = EnrollmentBot()
        bot.process("")
        for _ in range(bot.MAX_RETRY):
            bot.process("x")  # 一直是无效的单字姓名
        assert bot.state == ConversationState.ERROR


class TestExitCommand:
    def test_exit_mid_flow(self) -> None:
        bot = EnrollmentBot()
        bot.process("")
        bot.process("合法姓名")
        reply = bot.process("退出")
        assert bot.state == ConversationState.ERROR
        assert "退出" in reply
