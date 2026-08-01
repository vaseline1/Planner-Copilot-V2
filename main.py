"""main.py —— 命令行入口：PyCharm 里点 Run 实际执行的就是这个文件

职责：
    启动前体检 -> 初始化数据库 -> 命令行对话循环（支持完整建档
    流程、6个工具的对话式调用）。

这个项目现在有三个可选的启动入口，共用同一套 agent_core / database /
memory / tools / knowledge_base / rag_system，选哪个看场景：
    - main.py   命令行，最简单，适合本地开发调试
    - app.py    Streamlit 网页工作台，适合规划师日常使用
    - api.py    FastAPI 服务，适合对外提供给多个规划师、或接入其它系统

为什么 import 顺序有讲究：
    下面特意先 import 了 memory 和 rag_system（它们各自定义了数据库表），
    再调用 database.init_db()——这样建表时才能看到全部表结构
    （ConsultationMemory 来自 memory.py，document_chunks 由 rag_system
    单独管理，不受 init_db() 影响，需要额外调用 init_rag_db()）。

三期改造同步：建档流程从 8 步扩为 9 步 + 奖项明细子流程（模考成绩/
竞赛奖项明细/意向专业/目标升学路径全部收集入库），下方 WELCOME 文案
已同步更新，主循环逻辑不变。
"""
from __future__ import annotations

import sys

import memory  # noqa: F401 —— import 是为了触发 ConsultationMemory 表注册
import database
from agent_core import PlannerAgent
from config import ConfigError
from enrollment_bot import ConversationState, EnrollmentBot

WELCOME = """
========================================================
 PlannerCopilot 完整版 —— 天津升学规划师 AI 助手
========================================================
输入你的问题，比如：
  "帮我判断一个物化生、天津市预估位次1500的高二学生，能否报南开强基"
  "南开大学强基计划校测占比多少"（需要先用 index_pdf() 索引过简章）
  "天津有哪些院校做综合评价"

命令：
  exit / quit   退出程序
  new           开始建档（完整流程：姓名/年级/选科/位次/预估分/模考/竞赛奖项
                （含明细）/意向专业/目标路径/确认）
  who           查看当前正在跟进的学生
--------------------------------------------------------
"""


def run() -> None:
    try:
        agent = PlannerAgent()
    except ConfigError as exc:
        print(f"❌ 启动失败：{exc}")
        print("提示：先运行 `uv run python check_setup.py` 做一次完整体检。")
        sys.exit(1)

    try:
        database.init_db()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 数据库初始化失败：{exc}")
        print("提示：确认 PostgreSQL 已启动，且 .env 里的 DATABASE_URL 正确。")
        sys.exit(1)

    print(WELCOME)

    session_id = "cli-session"
    current_student_id: int | None = None
    enrollment_bot: EnrollmentBot | None = None

    while True:
        try:
            user_input = input("规划师> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("👋 再见！")
            break

        if user_input.lower() == "new":
            enrollment_bot = EnrollmentBot(planner_id="default")
            print(f"\nPlannerCopilot> {enrollment_bot.process('')}\n")
            continue

        if user_input.lower() == "who":
            if current_student_id is None:
                print("当前没有跟进中的学生，输入 new 建一个。\n")
            else:
                student = database.get_student_by_id(current_student_id)
                print(f"当前跟进：{student.name}（{student.grade}，{student.subject_combo}）\n")
            continue

        # 建档流程进行中：这一轮输入交给状态机，不交给大模型
        if enrollment_bot is not None and enrollment_bot.state not in (
            ConversationState.COMPLETE,
            ConversationState.ERROR,
        ):
            reply = enrollment_bot.process(user_input)
            print(f"\nPlannerCopilot> {reply}\n")
            if enrollment_bot.state == ConversationState.COMPLETE:
                student = database.get_student_by_name(enrollment_bot.data["name"], planner_id="default")
                if student:
                    current_student_id = student.id
                enrollment_bot = None
            continue

        reply = agent.chat(user_input, session_id=session_id, student_id=current_student_id)
        print(f"\nPlannerCopilot> {reply}\n")


if __name__ == "__main__":
    run()
