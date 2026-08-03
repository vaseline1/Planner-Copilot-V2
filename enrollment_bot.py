"""enrollment_bot.py —— 学生建档 Bot（状态机实现）

职责：
    用状态机（而不是直接把一堆问题丢给大模型）实现"规划师一步步回答，
    系统逐步收集学生信息"的建档流程。建完档自动触发一次三条路径的
    资格初判，并把结果存进数据库。

为什么建档要用状态机，而不是让大模型自由对话去问：
    建档信息（姓名/年级/选科/位次……）每一项都有明确的格式要求，
    状态机能保证"没填对之前不放你去下一步"这种确定性的流程控制；
    连续填错3次不会把用户丢进大模型，而是自动回到建档开头重新循环。这类"表单类"交互用状态机比用大模型自由发挥更
    可靠、更省 Token、也更容易测试——这也是为什么下面每个状态的处理
    函数都能被单独写单元测试，不需要真的联网调用大模型。

三期改造后的变化（收集完整档案，一个字段都不丢）：
    流程从 8 步扩为 9 步 + 一个奖项明细子流程：
      1姓名 → 2年级 → 3选科 → 4预估位次 → 5预估总分 → 6模考成绩（可跳过）
      → 7竞赛奖项（答"有"则逐条录入明细：名称/等级/年份/学科）
      → 8意向专业 → 9目标升学路径 → 确认
    收尾时先用收集到的数据组装 StudentProfile——Pydantic 在这里做最终
    校验（分数范围、选科合法性等），再通过 create_student_from_profile()
    落库。旧版的两个遗漏被修掉：意向专业收集了却不入库、竞赛信息只存
    一个布尔值丢掉全部明细。
"""
from __future__ import annotations

import logging
import re
from datetime import date
from enum import Enum, auto

from database import add_consultation, create_student_from_profile
from memory import save_memory
from models import CompetitionAward, Grade, PathType, StudentProfile, SubjectCombo
from tools import check_all_paths_for_student

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    START = auto()
    ASK_NAME = auto()
    ASK_GRADE = auto()
    ASK_COMBO = auto()
    ASK_RANK = auto()
    ASK_SCORE = auto()
    ASK_MOCK_SCORES = auto()
    ASK_AWARD = auto()
    ASK_AWARD_DETAIL = auto()
    ASK_MAJOR = auto()
    ASK_PATHS = auto()
    CONFIRM = auto()
    COMPLETE = auto()
    ERROR = auto()


VALID_GRADES = {g.value for g in Grade}
VALID_COMBOS = {c.value for c in SubjectCombo}
COMBO_LIST_STR = "、".join(list(VALID_COMBOS)[:5]) + " 等20种"
VALID_PATH_TYPES = {p.value for p in PathType}
PATH_LIST_STR = "、".join(p.value for p in PathType)
SKIP_WORDS = {"无", "没有", "暂无", "跳过"}
# 奖项子流程的"结束录入"和"反悔跳回"口令
AWARD_DONE_WORDS = {"完成", "好了", "录完了", "结束"}
AWARD_ESCAPE_WORDS = {"没有", "无", "算了", "记错了", "跳过", "暂无"}
# 兼容中英文逗号、顿号、空格作为字段分隔符
_AWARD_SPLIT_RE = re.compile(r"[,，、;；\s]+")
# 年份识别：2010-2039 的四位数，可带"年"字
_YEAR_RE = re.compile(r"^(20[0-3]\d)年?$")


class EnrollmentBot:
    """学生建档对话机器人（状态机实现）。

    用法：
        bot = EnrollmentBot()
        reply = bot.process("")       # 触发开场白
        reply = bot.process("张明")    # 逐步推进状态机
        ...
    """

    MAX_RETRY = 3

    def __init__(self, planner_id: str = "default") -> None:
        self.state = ConversationState.START
        self.data: dict = {}
        self.retry_count = 0
        self.planner_id = planner_id

    def process(self, user_input: str) -> str:
        """处理规划师输入，返回 Bot 回复；每次调用推进状态机一步。"""
        user_input = user_input.strip()

        if user_input.lower() in ("退出", "exit", "quit", "q"):
            self.state = ConversationState.ERROR
            return "已退出建档流程，数据未保存。"

        handler = self._get_handler()
        return handler(user_input)

    def _get_handler(self):
        return {
            ConversationState.START: self._handle_start,
            ConversationState.ASK_NAME: self._handle_name,
            ConversationState.ASK_GRADE: self._handle_grade,
            ConversationState.ASK_COMBO: self._handle_combo,
            ConversationState.ASK_RANK: self._handle_rank,
            ConversationState.ASK_SCORE: self._handle_score,
            ConversationState.ASK_MOCK_SCORES: self._handle_mock_scores,
            ConversationState.ASK_AWARD: self._handle_award,
            ConversationState.ASK_AWARD_DETAIL: self._handle_award_detail,
            ConversationState.ASK_MAJOR: self._handle_major,
            ConversationState.ASK_PATHS: self._handle_paths,
            ConversationState.CONFIRM: self._handle_confirm,
        }.get(self.state, lambda _: "当前状态异常，请重新开始建档。")

    # ── 第1步：姓名 ────────────────────────────────────────────
    def _handle_start(self, _: str) -> str:
        self.state = ConversationState.ASK_NAME
        return "开始为新学生建档。输入'退出'可随时中止。\n\n第1步：请输入学生姓名："

    def _handle_name(self, user_input: str) -> str:
        if len(user_input) < 2 or len(user_input) > 20:
            return self._retry("姓名长度应在2-20字之间，请重新输入：")
        self.data["name"] = user_input
        self.retry_count = 0
        self.state = ConversationState.ASK_GRADE
        return f"✅ 姓名：{user_input}\n\n第2步：请输入年级（高一 / 高二 / 高三）："

    # ── 第2步：年级 ────────────────────────────────────────────
    def _handle_grade(self, user_input: str) -> str:
        if user_input not in VALID_GRADES:
            return self._retry("请输入：高一、高二 或 高三")
        self.data["grade"] = user_input
        self.retry_count = 0
        self.state = ConversationState.ASK_COMBO
        return f"✅ 年级：{user_input}\n\n第3步：请输入选科组合（天津3+3，如：物化生）\n常见组合：{COMBO_LIST_STR}"

    # ── 第3步：选科 ────────────────────────────────────────────
    def _handle_combo(self, user_input: str) -> str:
        if user_input not in VALID_COMBOS:
            return self._retry(f"'{user_input}' 不是有效的天津3+3选科组合\n请重新输入，如：物化生、物化史、史地政 等")
        self.data["subject_combo"] = user_input
        self.retry_count = 0
        self.state = ConversationState.ASK_RANK
        return f"✅ 选科：{user_input}\n\n第4步：请输入天津市预估位次（如：2500，纯数字）："

    # ── 第4步：预估位次 ────────────────────────────────────────
    def _handle_rank(self, user_input: str) -> str:
        try:
            rank = int(user_input)
            if not (1 <= rank <= 300000):
                raise ValueError
        except ValueError:
            return self._retry("请输入合理的位次（1-300000之间的整数）：")
        self.data["estimated_rank"] = rank
        self.retry_count = 0
        self.state = ConversationState.ASK_SCORE
        return f"✅ 预估位次：{rank}\n\n第5步：请输入预估高考总分（如：620）："

    # ── 第5步：预估总分 ────────────────────────────────────────
    def _handle_score(self, user_input: str) -> str:
        try:
            score = int(user_input)
            if not (0 <= score <= 750):
                raise ValueError
        except ValueError:
            return self._retry("请输入0-750之间的整数：")
        self.data["estimated_score"] = score
        self.retry_count = 0
        self.state = ConversationState.ASK_MOCK_SCORES
        return (
            f"✅ 预估分数：{score}\n\n第6步：最近的模考成绩（可多个，用顿号分隔，如：610、625、630）\n"
            "没有可输入'无'："
        )

    # ── 第6步：模考成绩（可跳过）────────────────────────────────
    def _handle_mock_scores(self, user_input: str) -> str:
        if user_input in SKIP_WORDS:
            scores: list[int] = []
        else:
            try:
                scores = [
                    int(s)
                    for s in user_input.replace("，", "、").replace(",", "、").split("、")
                    if s.strip()
                ]
            except ValueError:
                return self._retry("模考成绩必须是数字，多个用顿号分隔，如：610、625、630\n没有可输入'无'：")
            if not scores or any(not (0 <= s <= 750) for s in scores):
                return self._retry("每个模考成绩需在0-750之间，多个用顿号分隔\n没有可输入'无'：")
        self.data["mock_scores"] = scores
        self.retry_count = 0
        self.state = ConversationState.ASK_AWARD
        shown = "、".join(str(s) for s in scores) if scores else "无"
        return f"✅ 模考成绩：{shown}\n\n第7步：是否有五大学科竞赛奖项？\n（输入：有 / 没有）"

    # ── 第7步：竞赛奖项（有 → 逐条录入明细）─────────────────────
    def _handle_award(self, user_input: str) -> str:
        if user_input in ("有", "是", "yes", "y"):
            self.data["awards"] = []
            self.retry_count = 0
            self.state = ConversationState.ASK_AWARD_DETAIL
            return (
                "✅ 竞赛奖项：有\n\n"
                "请一条一条录入奖项，每条包含 4 项信息，用逗号或顿号隔开：\n"
                "　① 竞赛名称　② 获奖等级　③ 获奖年份　④ 学科\n"
                "👉 照着例子写就行：全国化学奥林匹克联赛，全国决赛二等奖，2024，化学\n\n"
                "· 输完一条回车，我会逐条确认\n"
                "· 全部录完输入「完成」\n"
                "· 如果其实没有奖项，输入「没有」即可跳过这一步："
            )
        if user_input in ("没有", "无", "否", "no", "n"):
            self.data["awards"] = []
            self.retry_count = 0
            self.state = ConversationState.ASK_MAJOR
            return (
                "✅ 竞赛奖项：无\n\n第8步：意向专业方向是什么？（可多个，用顿号分隔，如：计算机、金融）\n"
                "不确定可输入'待定'："
            )
        return self._retry("请回答'有'或'没有'：")

    def _handle_award_detail(self, user_input: str) -> str:
        """奖项明细录入子流程：每条一行，「完成」结束。

        容错设计（真实测试者踩过的坑都在这里）：
        · 答了"有"却直接说"完成"/"没有"——不卡死，按无奖项继续往下走；
        · 分隔符兼容逗号、顿号、空格；年份写"2024年"也能认；
        · 只写了 3 项（常见是漏了获奖等级）——不整条打回，
          记住已收到的部分，只追问缺的那一项。
        录入的每条都立即用 CompetitionAward 模型校验（年份范围等），不合格不放行。"""
        awards: list[CompetitionAward] = self.data["awards"]

        # ── 上一条缺"获奖等级"，本轮输入直接当作等级补录 ──
        pending = self.data.get("_pending_award")
        if (
            pending is not None
            and user_input not in AWARD_DONE_WORDS
            and user_input not in AWARD_ESCAPE_WORDS
        ):
            return self._complete_pending_award(pending, level=user_input)

        self.data.pop("_pending_award", None)

        # ── 反悔通道：答了"有"又改主意，按无奖项继续 ──
        if user_input in AWARD_ESCAPE_WORDS or (user_input in AWARD_DONE_WORDS and not awards):
            self.retry_count = 0
            self.state = ConversationState.ASK_MAJOR
            return (
                "✅ 好的，本步按「无竞赛奖项」继续。\n\n"
                "第8步：意向专业方向是什么？（可多个，用顿号分隔，如：计算机、金融）\n不确定可输入'待定'："
            )

        # ── 正常结束录入（已至少录入一条）──
        if user_input in AWARD_DONE_WORDS:
            self.retry_count = 0
            self.state = ConversationState.ASK_MAJOR
            lines = "；".join(f"{a.competition_name}·{a.level}（{a.year}年）" for a in awards)
            return (
                f"✅ 已录入 {len(awards)} 条奖项：{lines}\n\n"
                "第8步：意向专业方向是什么？（可多个，用顿号分隔，如：计算机、金融）\n不确定可输入'待定'："
            )

        parts = [p for p in (s.strip() for s in _AWARD_SPLIT_RE.split(user_input)) if p]

        # ── 常见场景：只写了 3 项且能认出年份 → 缺的是"获奖等级"，只追问这一项 ──
        if len(parts) == 3:
            year_idx = next((i for i, p in enumerate(parts) if _YEAR_RE.match(p)), None)
            if year_idx is not None:
                year = int(_YEAR_RE.match(parts[year_idx]).group(1))
                rest = [p for i, p in enumerate(parts) if i != year_idx]
                # 剩下两项按"名称、学科"理解（名称一般更长、在前）
                name, subject = rest[0], rest[1]
                self.data["_pending_award"] = {"name": name, "year": year, "subject": subject}
                self.retry_count = 0
                return (
                    f"📝 已收到：{name}（{year}年，{subject}），还差一项「获奖等级」。\n"
                    "请直接回复等级即可，例如：全国决赛二等奖 / 省级一等奖："
                )

        if len(parts) != 4:
            return self._retry(
                "这一条需要 4 项信息：① 竞赛名称 ② 获奖等级 ③ 年份 ④ 学科\n"
                "👉 例如：全国化学奥林匹克联赛，全国决赛二等奖，2024，化学\n"
                "请重新输入本条；没有奖项可输入「没有」，录完了输入「完成」："
            )
        name, level, year_raw, subject = parts
        m = _YEAR_RE.match(year_raw)
        if not m:
            return self._retry(
                f"年份「{year_raw}」看不懂，请写成 4 位数字（如 2024）。\n"
                "请重新输入本条；没有奖项可输入「没有」，录完了输入「完成」："
            )
        try:
            award = CompetitionAward(competition_name=name, level=level, year=int(m.group(1)), subject=subject)
        except Exception:  # noqa: BLE001 —— Pydantic 校验失败（年份越界等）给同一条提示
            return self._retry(
                "这条没通过校验（年份需在2010-2035之间）。请重新输入本条，或输入「完成」结束录入："
            )
        awards.append(award)
        self.retry_count = 0
        return f"✅ 已记录：{award.competition_name}·{award.level}（{award.year}年）\n继续录入下一条，或输入「完成」："

    def _complete_pending_award(self, pending: dict, level: str) -> str:
        """补录流程：pending 里已有名称/年份/学科，本轮输入作为获奖等级。"""
        level = level.strip()
        if not level or len(level) > 30:
            return self._retry("获奖等级请简短填写，例如：全国决赛二等奖 / 省级一等奖：")
        try:
            award = CompetitionAward(
                competition_name=pending["name"], level=level,
                year=pending["year"], subject=pending["subject"],
            )
        except Exception:  # noqa: BLE001
            self.data.pop("_pending_award", None)
            return self._retry("这条没通过校验，请按完整格式重新录入本条（名称，等级，年份，学科）：")
        self.data.pop("_pending_award", None)
        self.data["awards"].append(award)
        self.retry_count = 0
        return f"✅ 已记录：{award.competition_name}·{award.level}（{award.year}年）\n继续录入下一条，或输入「完成」："

    # ── 第8步：意向专业 ────────────────────────────────────────
    def _handle_major(self, user_input: str) -> str:
        if user_input in ("待定", "不确定", "暂无"):
            majors: list[str] = []
        else:
            majors = [m.strip() for m in user_input.replace("，", "、").split("、") if m.strip()]
        self.data["majors"] = majors
        self.retry_count = 0
        self.state = ConversationState.ASK_PATHS
        return (
            f"✅ 意向方向：{'、'.join(majors) if majors else '待定'}\n\n"
            f"第9步：目标升学路径（可多选，用顿号分隔）\n可选：{PATH_LIST_STR}\n不确定可输入'待定'："
        )

    # ── 第9步：目标升学路径 ────────────────────────────────────
    def _handle_paths(self, user_input: str) -> str:
        if user_input in ("待定", "不确定", "暂无"):
            paths: list[PathType] = []
        else:
            raw = [p.strip() for p in user_input.replace("，", "、").split("、") if p.strip()]
            invalid = [p for p in raw if p not in VALID_PATH_TYPES]
            if invalid:
                return self._retry(
                    f"'{'、'.join(invalid)}' 不是有效的路径类型。可选：{PATH_LIST_STR}\n请重新输入："
                )
            paths = [PathType(p) for p in raw]
        self.data["target_paths"] = paths
        self.retry_count = 0
        self.state = ConversationState.CONFIRM

        awards: list[CompetitionAward] = self.data["awards"]
        award_shown = "；".join(f"{a.competition_name}·{a.level}（{a.year}年）" for a in awards) if awards else "无"
        mock_shown = "、".join(str(s) for s in self.data["mock_scores"]) if self.data["mock_scores"] else "无"
        path_shown = "、".join(p.value for p in paths) if paths else "待定"
        return (
            "📋 请确认以下学生信息：\n"
            f"  姓名：{self.data['name']}\n"
            f"  年级：{self.data['grade']}\n"
            f"  选科：{self.data['subject_combo']}\n"
            f"  预估位次：{self.data['estimated_rank']}\n"
            f"  预估分数：{self.data['estimated_score']}\n"
            f"  模考成绩：{mock_shown}\n"
            f"  竞赛奖项：{award_shown}\n"
            f"  意向方向：{'、'.join(self.data['majors']) if self.data['majors'] else '待定'}\n"
            f"  目标路径：{path_shown}\n\n"
            "信息正确请输入'确认'，需要修改请输入'重新开始'："
        )

    # ── 确认并保存 ─────────────────────────────────────────────
    def _handle_confirm(self, user_input: str) -> str:
        if user_input in ("确认", "是", "对", "yes", "y", "ok"):
            return self._save_and_complete()
        if user_input in ("重新开始", "重新", "no", "否"):
            planner_id = self.planner_id
            self.__init__(planner_id=planner_id)
            return "已重置，重新开始建档。\n请输入任意内容开始："
        return "请输入'确认'或'重新开始'："

    def _save_and_complete(self) -> str:
        """组装 StudentProfile（Pydantic 最终校验）→ 落库 → 触发路径初判。"""
        try:
            profile = StudentProfile(
                name=self.data["name"],
                grade=Grade(self.data["grade"]),
                subject_combo=SubjectCombo(self.data["subject_combo"]),
                mock_scores=self.data.get("mock_scores", []),
                estimated_rank=self.data["estimated_rank"],
                estimated_score=self.data["estimated_score"],
                competition_awards=self.data.get("awards", []),
                intended_major_directions=self.data.get("majors", []),
                target_path_types=self.data.get("target_paths", []),
                planner_id=self.planner_id,
                created_date=date.today(),
            )
            student = create_student_from_profile(profile, planner_id=self.planner_id)

            # 一站式判断：预估分/位次优先，实考/模考兜底的逻辑收在 tools 里；
            # 建档流程两项都已收集，这里不会触发 ValueError。
            results = check_all_paths_for_student(profile)
            summary = "建档时路径初判：" + " | ".join(f"{r.display_path}{r.status_label}" for r in results)
            add_consultation(student.id, "建档初判", "自动路径适配分析", summary)
            # 路径初判是最值得"长期记住"的结论，同步写入长期记忆表，
            # 规划师隔天重开这个学生时，Agent 能在上下文里看到它。
            # best-effort：学生此刻已成功建档，长期记忆写入失败绝不能让这一步整体回滚，
            # 所以单独 try 包起来，失败只记日志。
            try:
                save_memory(student.id, "路径初判", "建档时路径初判", summary)
            except Exception as exc:  # noqa: BLE001
                logger.warning("长期记忆写入失败（学生已建档成功，不影响）：%s", exc)

            self.state = ConversationState.COMPLETE
            path_summary = "\n".join(
                f"  {'✅' if r.is_eligible else '❌'} {r.display_path}：{r.status_label}" for r in results
            )

            return (
                f"✅ 学生档案已建立（ID：{student.id}）\n\n"
                f"📊 升学路径初判结果：\n{path_summary}\n\n"
                "💡 提示：以上为基础规则判断，需核查最新简章确认。"
            )
        except Exception as exc:  # noqa: BLE001 —— 建档失败要给出可读提示，不能整个崩掉
            return f"❌ 保存失败：{exc}\n请联系技术支持或重试。"

    def _retry(self, prompt: str) -> str:
        """输入无效的重试控制。

        连续失败 MAX_RETRY 次后不再"终止建档、把用户丢进大模型"——
        那对规划师来说流程莫名其妙就断了。改为自动清空已填内容、
        回到第1步重新建档，让流程始终留在状态机里循环。
        想真正退出仍可随时输入"退出"。"""
        self.retry_count += 1
        if self.retry_count >= self.MAX_RETRY:
            planner_id = self.planner_id
            self.__init__(planner_id=planner_id)
            self.state = ConversationState.ASK_NAME
            return (
                f"❌ 连续{self.MAX_RETRY}次输入无效，已自动回到建档开头。\n"
                "别担心，我们重新来一遍（输入'退出'可中止建档）。\n\n"
                "第1步：请输入学生姓名："
            )
        return f"❌ 输入无效（第{self.retry_count}次）。{prompt}"


def create_gradio_app():
    """Gradio 快速预览界面——建档流程的可视化调试工具，不是给规划师日常用的正式界面
    （正式界面是 app.py 的 Streamlit 三栏工作台）。"""
    import gradio as gr

    bot = EnrollmentBot()

    def respond(message: str, history: list) -> str:
        if bot.state == ConversationState.START or not history:
            bot_reply = bot.process("")
            if message:
                bot_reply += "\n" + bot.process(message)
        else:
            bot_reply = bot.process(message)
        return bot_reply

    return gr.ChatInterface(
        fn=respond,
        title="🎓 PlannerCopilot —— 学生建档 Bot（天津版）",
        description="通过对话方式完成学生档案录入，自动判断升学路径适配情况",
        examples=[
            "开始建档", "张明", "高二", "物化生", "2500", "620", "610、625",
            "有", "全国化学奥林匹克联赛，全国决赛二等奖，2024，化学", "完成",
            "计算机、金融", "强基计划、综合评价",
        ],
    )


if __name__ == "__main__":
    create_gradio_app().launch()
