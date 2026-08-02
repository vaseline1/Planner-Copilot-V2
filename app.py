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

五期改造（数据管理入口）后的变化：
    侧边栏新增"数据管理"页——规划师在网页上直接维护两类资料库，
    不用再 SSH 上服务器敲命令：
    ① 院校项目卡片（data/tianjin_programs.json 的增删改）；
    ② 招生简章 PDF 上传入库（向量库）与已入库文档管理。
"""
import os
import tempfile

import streamlit as st
from sqlmodel import Session, select

import database
import memory as memory_module
from agent_core import PlannerAgent
from config import ConfigError
from database import StudentDB, engine
from enrollment_bot import ConversationState, EnrollmentBot
from knowledge_base import add_program, delete_program, load_programs, search_programs, update_program
from models import PathType, SubjectCombo, UniversityProgram
from rag_system import delete_document, index_pdf, list_documents

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
        "kb_editing_index": None,
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


# ════════════════════════════════════════════════════════════
# 数据管理页：两类资料库的网页录入入口
# ════════════════════════════════════════════════════════════

def _split_cn_list(raw: str) -> list[str]:
    """把"天津，北京，河北"这类中英文逗号混用的输入拆成干净列表。"""
    return [item.strip() for item in raw.replace("，", ",").split(",") if item.strip()]


def _program_form(form_suffix: str, default: UniversityProgram | None) -> None:
    """院校项目卡片的"新增/编辑"表单。form_suffix 区分编辑目标，
    避免切换卡片时 widget 记住上一张卡的旧值。"""
    k = lambda name: f"{name}_{form_suffix}"  # noqa: E731

    with st.form(f"kb_form_{form_suffix}"):
        st.text_input("大学名称 *", value=default.university_name if default else "",
                      placeholder="如：南开大学", key=k("university_name"))
        col1, col2 = st.columns(2)
        path_options = [e.value for e in PathType]
        col1.selectbox("路径类型 *", path_options,
                       index=path_options.index(default.path_type.value) if default else 0, key=k("path_type"))
        col2.number_input("数据年份 *", min_value=2020, max_value=2035,
                          value=default.source_year if default else 2026, key=k("source_year"))

        st.text_input("招生面向省份（逗号分隔，须含天津）",
                      value="，".join(default.province_scope) if default else "天津", key=k("province_scope"))
        st.text_input("招生专业（逗号分隔）", value="，".join(default.majors) if default else "",
                      placeholder="如：数学与应用数学，物理学", key=k("majors"))

        combo_options = [e.value for e in SubjectCombo]
        st.multiselect("选科要求（不选 = 不限选科）", combo_options,
                       default=[c.value for c in default.subject_combo_required] if default else [],
                       key=k("subject_combo"))

        col3, col4 = st.columns(2)
        physics_choice = ["（未填）", "是", "否"]
        physics_default = 0 if not default or default.has_physics_requirement is None else (1 if default.has_physics_requirement else 2)
        col3.selectbox("是否必须选物理", physics_choice, index=physics_default, key=k("physics"))
        col4.text_input("校测占比（如 0.15，留空=待查）",
                        value=str(default.campus_test_ratio) if default and default.campus_test_ratio is not None else "",
                        key=k("campus_ratio"))

        st.text_input("分数要求描述", value=default.score_requirement_desc if default else "",
                      placeholder="如：高考成绩占85%，校测占15%", key=k("score_desc"))
        st.text_input("竞赛要求", value=default.competition_requirement or "" if default else "",
                      placeholder="如：数学/物理竞赛省一以上优先", key=k("competition"))

        col5, col6 = st.columns(2)
        col5.text_input("天津招生名额（留空=未知）",
                        value=str(default.enrollment_tianjin) if default and default.enrollment_tianjin is not None else "",
                        key=k("enrollment"))
        col6.text_input("报名截止时间", value=default.application_deadline or "" if default else "",
                        placeholder="如：4月20日", key=k("deadline"))
        st.text_input("来源链接（官方简章网址，备查）", value=default.source_url if default else "", key=k("source_url"))

        submitted = st.form_submit_button("💾 保存卡片", use_container_width=True)

    if not submitted:
        return

    # ── 收集并校验 ──
    vals = {name: st.session_state[k(name)] for name in [
        "university_name", "path_type", "source_year", "province_scope", "majors",
        "subject_combo", "physics", "campus_ratio", "score_desc", "competition",
        "enrollment", "deadline", "source_url",
    ]}
    if not vals["university_name"].strip():
        st.error("大学名称不能为空")
        return
    try:
        campus_ratio = float(vals["campus_ratio"]) if vals["campus_ratio"].strip() else None
        if campus_ratio is not None and not (0.0 <= campus_ratio <= 1.0):
            raise ValueError("校测占比必须在 0~1 之间")
        enrollment = int(vals["enrollment"]) if vals["enrollment"].strip() else None
        program = UniversityProgram(
            university_name=vals["university_name"].strip(),
            path_type=PathType(vals["path_type"]),
            province_scope=_split_cn_list(vals["province_scope"]),
            majors=_split_cn_list(vals["majors"]),
            subject_combo_required=[SubjectCombo(v) for v in vals["subject_combo"]],
            has_physics_requirement={"（未填）": None, "是": True, "否": False}[vals["physics"]],
            score_requirement_desc=vals["score_desc"].strip(),
            competition_requirement=vals["competition"].strip() or None,
            campus_test_ratio=campus_ratio,
            enrollment_tianjin=enrollment,
            application_deadline=vals["deadline"].strip() or None,
            source_year=int(vals["source_year"]),
            source_url=vals["source_url"].strip(),
        )
    except (ValueError, TypeError) as exc:
        st.error(f"填写有误：{exc}")
        return

    if form_suffix == "new":
        add_program(program)
        st.success(f"✅ 已新增卡片：{program.university_name} · {program.path_type.value}")
    else:
        update_program(int(form_suffix), program)
        st.session_state.kb_editing_index = None
        st.success(f"✅ 已更新卡片：{program.university_name} · {program.path_type.value}")
    st.rerun()


def render_data_admin() -> None:
    """数据管理页：卡片库（JSON）+ 简章库（向量库）。"""
    st.markdown("### 📚 数据管理")
    st.caption(
        "两类资料库的录入入口：**卡片库**存结构化关键信息（回答“有没有/是什么”）；"
        "**简章库**存官方简章原文切片（回答细节问题）。数据请对照当年官方简章核实后录入。"
    )

    tab_cards, tab_pdfs = st.tabs(["📇 院校项目卡片库", "📄 招生简章库（RAG）"])

    # ── Tab 1：卡片管理 ─────────────────────────────────────
    with tab_cards:
        programs = load_programs()
        st.caption(f"共 {len(programs)} 张卡片 · 存储于 data/tianjin_programs.json")

        if not programs:
            st.info("卡片库是空的，先在下方新增第一张")
        for idx, p in enumerate(programs):
            info_col, edit_col, del_col = st.columns([7, 1, 1])
            info_col.write(f"**{idx}｜{p.university_name}** · {p.path_type.value} · {p.source_year}年")
            info_col.caption(
                f"专业：{'、'.join(p.majors) if p.majors else '未填'}　"
                f"天津名额：{p.enrollment_tianjin if p.enrollment_tianjin is not None else '未知'}　"
                f"报名截止：{p.application_deadline or '未填'}"
            )
            if edit_col.button("✏️", key=f"kb_edit_{idx}", help="编辑这张卡片"):
                st.session_state.kb_editing_index = idx
                st.rerun()
            if del_col.button("🗑️", key=f"kb_del_{idx}", help="删除这张卡片"):
                delete_program(idx)
                if st.session_state.kb_editing_index == idx:
                    st.session_state.kb_editing_index = None
                st.rerun()

        st.divider()
        editing = st.session_state.kb_editing_index
        if editing is not None and 0 <= editing < len(programs):
            st.markdown(f"#### ✏️ 编辑卡片 {editing}｜{programs[editing].university_name}")
            if st.button("取消编辑", key="kb_cancel_edit"):
                st.session_state.kb_editing_index = None
                st.rerun()
            _program_form(str(editing), programs[editing])
        else:
            st.markdown("#### ➕ 新增卡片")
            _program_form("new", None)

    # ── Tab 2：简章 PDF 库 ──────────────────────────────────
    with tab_pdfs:
        st.markdown("#### 📤 上传简章 PDF")
        st.caption("只支持**文字版** PDF（扫描件请先 OCR）；同学校+同路径+同年份重复上传会覆盖旧数据，不会重复。")
        up1, up2, up3 = st.columns(3)
        rag_university = up1.text_input("大学名称", placeholder="如：南开大学", key="rag_university")
        rag_path_type = up2.selectbox("路径类型", [e.value for e in PathType], key="rag_path_type")
        rag_year = up3.number_input("简章年份", min_value=2020, max_value=2035, value=2026, key="rag_year")
        uploaded = st.file_uploader("选择 PDF 文件", type=["pdf"], key="rag_file")

        if st.button("📥 上传并索引", key="rag_upload_btn"):
            if not rag_university.strip():
                st.error("请填写大学名称")
            elif uploaded is None:
                st.error("请选择 PDF 文件")
            else:
                tmp_path = ""
                try:
                    with st.spinner("解析 + 向量化中（几百个文本块约需一两分钟）…"):
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            tmp.write(uploaded.getvalue())
                            tmp_path = tmp.name
                        chunks = index_pdf(tmp_path, rag_university.strip(), rag_path_type, int(rag_year))
                    if chunks > 0:
                        st.success(f"✅ 已索引 {chunks} 个文本块")
                        st.rerun()
                    else:
                        st.warning("PDF 解析出 0 个文本块，可能是纯扫描件，请先 OCR 再上传")
                except Exception as exc:  # noqa: BLE001 —— embedding API 故障等，直接展示原因
                    st.error(f"索引失败：{exc}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        st.divider()
        st.markdown("#### 📚 已入库的简章")
        try:
            docs = list_documents()
        except Exception as exc:  # noqa: BLE001 —— 向量表未初始化时给出可操作提示
            st.warning(f"读取文档列表失败：{exc}（服务器上是否执行过 init_rag_db()？）")
            docs = []
        if not docs:
            st.info("简章库还是空的，先上传几份当年官方简章")
        for doc in docs:
            d_col, del_col = st.columns([7, 1])
            d_col.write(
                f"**{doc['university']}** · {doc['path_type']} · {doc['source_year']}年　"
                f"共 `{doc['chunk_count']}` 个文本块"
            )
            if del_col.button("🗑️", key=f"rag_del_{doc['doc_id']}", help="删除这份简章的所有文本块"):
                delete_document(doc["doc_id"])
                st.rerun()


# ════════════════════════════════════════════════════════════
# 侧边栏：功能入口切换
# ════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🎓 PlannerCopilot")
page = st.sidebar.radio("功能入口", ["💬 学生咨询", "📚 数据管理"], label_visibility="collapsed")
st.sidebar.caption("数据管理页用于录入院校卡片和招生简章")

if page == "💬 学生咨询":
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
else:
    render_data_admin()

st.divider()
st.caption("PlannerCopilot 天津版 · 使用前请核查最新简章")
