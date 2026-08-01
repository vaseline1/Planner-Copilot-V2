"""app.py —— PlannerCopilot 规划师工作台（Streamlit 三栏界面）

职责：
    给规划师一个不用敲命令行的网页界面：左栏学生列表（点击切换）、
    中栏 AI 对话、右栏快速工具箱（选科速查/政策搜索）。

运行方式：
    uv run streamlit run app.py

为什么用 st.session_state 存当前学生/对话历史：
    Streamlit 的运行模型是"每次交互整个脚本从头重跑一遍"——如果不把
    "当前选中哪个学生""聊了什么"存进 session_state，每次点一下按钮
    页面都会"失忆"。session_state 是跨重跑保留状态的唯一方式。

四期改造（完整 CRUD）后的变化：
    右栏新增三个管理区块——咨询记录和长期记忆支持逐条删除；
    "危险操作区"提供"清空该学生所有数据"（保留档案、阶段重置）和
    "删除学生档案"（级联全删）两个不可逆操作，均需勾选确认后才出现按钮。
"""
import streamlit as st
from sqlmodel import Session, select

import database
import memory as memory_module
from agent_core import PlannerAgent
from config import ConfigError
from database import StudentDB, engine
from enrollment_bot import ConversationState, EnrollmentBot
from knowledge_base import search_programs

st.set_page_config(page_title="PlannerCopilot 升学规划师工作台（天津版）", page_icon="🎓", layout="wide")

PLANNER_ID = "planner_001"  # 单机版先固定一个规划师身份；多规划师场景走 api.py


@st.cache_resource
def get_agent() -> PlannerAgent:
    """Streamlit 每次重跑脚本都会执行到这一行，用 cache_resource 保证 Agent
    只真正初始化一次（避免每次点击都重新创建 OpenAI 客户端）。"""
    return PlannerAgent()


def _init_session_state() -> None:
    defaults = {
        "current_student_id": None,
        "current_student_name": "（未选择学生）",
        "messages": [],
        "enrollment_bot": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()

try:
    database.init_db()
except Exception as exc:  # noqa: BLE001
    st.error(f"数据库连接失败：{exc}\n请确认 PostgreSQL 已启动，且 .env 里 DATABASE_URL 正确。")
    st.stop()

try:
    agent = get_agent()
except ConfigError as exc:
    st.error(f"配置错误：{exc}")
    st.stop()

left_col, mid_col, right_col = st.columns([1, 2, 1])

# ════════════════════════════════════════════════════════════
# 左栏：学生管理
# ════════════════════════════════════════════════════════════
with left_col:
    st.markdown("### 📋 学生列表")

    with Session(engine) as session:
        students = list(
            session.exec(
                select(StudentDB).where(StudentDB.planner_id == PLANNER_ID).order_by(StudentDB.updated_at.desc())
            )
        )

    if not students:
        st.info("暂无学生档案，点击下方按钮建档")
    else:
        search_term = st.text_input("搜索学生", placeholder="输入姓名...", label_visibility="collapsed")
        filtered = [s for s in students if not search_term or search_term.lower() in s.name.lower()]
        for student in filtered:
            is_current = student.id == st.session_state.current_student_id
            label = f"{'▶ ' if is_current else ''}{student.name} · {student.grade} · {student.subject_combo}"
            if st.button(label, key=f"student_{student.id}", use_container_width=True):
                st.session_state.current_student_id = student.id
                st.session_state.current_student_name = student.name
                st.session_state.messages = memory_module.get_conversation_history(f"web:{student.id}")
                st.session_state.enrollment_bot = None
                st.rerun()

    st.divider()
    if st.button("➕ 新建学生档案", use_container_width=True):
        st.session_state.enrollment_bot = EnrollmentBot(planner_id=PLANNER_ID)
        st.session_state.current_student_id = None
        st.session_state.current_student_name = "（建档中）"
        opening = st.session_state.enrollment_bot.process("")
        st.session_state.messages = [{"role": "assistant", "content": opening}]
        st.rerun()

# ════════════════════════════════════════════════════════════
# 中栏：AI 对话 / 建档
# ════════════════════════════════════════════════════════════
with mid_col:
    st.markdown(f"### 💬 对话 —— {st.session_state.current_student_name}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("输入问题，如：该学生能报南开强基吗？")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            bot = st.session_state.enrollment_bot
            if bot is not None and bot.state != ConversationState.COMPLETE and bot.state != ConversationState.ERROR:
                # 建档流程进行中：这一轮输入交给状态机，不交给大模型
                with st.spinner("处理中..."):
                    reply = bot.process(prompt)
                if bot.state == ConversationState.COMPLETE:
                    st.session_state.current_student_id = bot.data.get("name")  # 占位，下面会用真实ID覆盖
                    with Session(engine) as session:
                        just_created = session.exec(
                            select(StudentDB).where(StudentDB.planner_id == PLANNER_ID).order_by(StudentDB.id.desc())
                        ).first()
                        if just_created:
                            st.session_state.current_student_id = just_created.id
                            st.session_state.current_student_name = just_created.name
            else:
                with st.spinner("🔍 分析中..."):
                    reply = agent.chat(
                        user_message=prompt,
                        session_id=f"web:{st.session_state.current_student_id or 'general'}",
                        student_id=st.session_state.current_student_id,
                    )
            st.write(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ════════════════════════════════════════════════════════════
# 右栏：快速工具箱
# ════════════════════════════════════════════════════════════
with right_col:
    st.markdown("### 🛠️ 快速工具箱")

    with st.expander("🔍 政策信息查询", expanded=True):
        search_query = st.text_input("关键词", placeholder="如：南开 综评", label_visibility="collapsed", key="policy_q")
        if st.button("搜索", key="policy_search") and search_query:
            results = search_programs(search_query)
            if results:
                for r in results[:3]:
                    st.write(f"**{r['院校']}** {r['路径类型']}")
                    st.caption(f"{r['分数要求']}")
            else:
                st.info("未找到相关信息，可能需要先扩充本地知识库数据")

    if st.session_state.current_student_id:
        _sid = st.session_state.current_student_id
        with st.expander("👤 当前学生档案", expanded=True):
            student = database.get_student_by_id(_sid)
            if student:
                st.write(f"**{student.name}**")
                st.write(f"年级：{student.grade}　选科：{student.subject_combo}")
                if student.estimated_rank:
                    st.write(f"预估位次：{student.estimated_rank}")
                st.caption(f"咨询阶段：{student.consultation_stage}")

        # ── 咨询记录：逐条查看 / 删除 ──────────────────────────
        with st.expander("📝 咨询记录"):
            consultations = database.get_consultations_for_student(_sid)
            if not consultations:
                st.caption("暂无咨询记录")
            for record in consultations:
                r_col, del_col = st.columns([5, 1])
                r_col.write(f"**{record.topic}**")
                r_col.caption(record.created_at.strftime("%Y-%m-%d %H:%M"))
                if del_col.button("🗑️", key=f"del_consult_{record.id}", help="删除这条咨询记录"):
                    database.delete_consultation(record.id, planner_id=PLANNER_ID)
                    st.rerun()

        # ── 长期记忆：逐条查看 / 删除 ──────────────────────────
        with st.expander("🧠 长期记忆"):
            memories = memory_module.get_student_memories(_sid)
            if not memories:
                st.caption("暂无长期记忆")
            for mem in memories:
                m_col, del_col = st.columns([5, 1])
                m_col.write(f"**[{mem.title}]** {mem.content[:60]}{'...' if len(mem.content) > 60 else ''}")
                m_col.caption(f"更新于 {mem.updated_at.strftime('%Y-%m-%d %H:%M')}")
                if del_col.button("🗑️", key=f"del_mem_{mem.id}", help="删除这条长期记忆"):
                    memory_module.delete_memory(mem.id, student_id=_sid)
                    st.rerun()

        # ── 危险操作区：清空数据 / 删除档案（勾选确认后才可点击）──
        with st.expander("⚠️ 危险操作区"):
            st.caption("以下操作不可恢复，勾选确认后按钮才会出现。")

            if st.checkbox("确认清空：删除该学生的全部咨询记录和长期记忆，档案保留、阶段重置", key="confirm_wipe"):
                if st.button("🧹 清空该学生所有数据", key="btn_wipe"):
                    database.wipe_student_data(_sid, planner_id=PLANNER_ID)
                    st.success("已清空：咨询记录和长期记忆已删除，咨询阶段已重置为「初次建档」。")
                    # ⚠️ 不要在这里写 st.session_state.confirm_wipe = False——
                    # 复选框 widget 已在本轮实例化，再改它的 key 会直接抛
                    # StreamlitAPIException。清空是幂等操作，复选框保持勾选无害。
                    st.rerun()

            if st.checkbox("确认删除：该学生档案连同所有数据一起删除，不可恢复", key="confirm_delete"):
                if st.button("🗑️ 删除学生档案", key="btn_delete"):
                    database.delete_student(_sid, planner_id=PLANNER_ID)
                    # 删除后当前学生置空，危险操作区整个不再渲染，
                    # confirm_delete 的勾选状态随之失效，同样不能手动改它的 key。
                    st.session_state.current_student_id = None
                    st.session_state.current_student_name = "（未选择学生）"
                    st.session_state.messages = []
                    st.session_state.enrollment_bot = None
                    st.success("学生档案已删除。")
                    st.rerun()

    with st.expander("📌 天津政策提示"):
        st.caption("⚠️ 请核查最新简章")
        st.write("· 综合评价：通常3-4月　· 强基计划：通常4月下旬　· 港澳JUPAS：通常3月")

st.divider()
st.caption("PlannerCopilot 天津版 · 使用前请核查最新简章")
