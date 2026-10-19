# PlannerCopilot —— 升学规划师 AI 助手

> 面向升学规划师工作场景的全栈 AI 应用：对话式资格初判、招生简章 RAG 问答、
> 学生建档与三层记忆、多租户 API，已部署上云并对外提供服务。
>
> **在线体验**：点击左栏「示例学生」即可直接提问；「新建学生档案」可体验状态机引导式建档流程——先建档、再咨询，对应规划师真实工作流。

---

## 这是什么

升学规划师的日常工作中，大量时间花在重复性判断上：这个选科组合能不能报强基？
南开综评的位次门槛是多少？某校简章里校测成绩占比多少？

PlannerCopilot 把这些场景做进了一个 Agent：

- **报考资格规则判断**：强基计划 / 综合评价（南开·天大）/ 港澳升学三条路径，
  输入选科组合、预估位次、竞赛奖项即可得到结构化初判（满足条件 / 缺失条件 / 建议动作）
- **招生简章 RAG 问答**：上传院校简章 PDF 建库，提问时检索原文、基于原文回答并标注来源页码
- **学生建档与跟进**：状态机引导的建档流程，咨询阶段自动流转（初次建档 → 规划进行中 →
  报名申请中 → 已完成，只进不退）
- **三层记忆**：Redis 短期会话（TTL + 超量摘要压缩）、PostgreSQL 长期要点、完整咨询记录留痕
- **多租户 API**：FastAPI + API Key 鉴权，不同规划师的数据互相隔离

## 技术架构

```
规划师输入
  └─ 三个入口：main.py（命令行）/ app.py（Streamlit 工作台）/ api.py（FastAPI）
       └─ agent_core.PlannerAgent.chat()                ← 编排中枢
            ├─ 拼消息：SYSTEM_PROMPT + 长期记忆(PG) + 短期历史(Redis) + 用户输入
            ├─ 工具调用循环（带轮次上限保护，手搓，未用 Agent 框架）
            │    └─ TOOL_REGISTRY 分发 6 个工具：
            │         ├─ tools.py          规则判断 ×3（零外部依赖）
            │         ├─ knowledge_base.py 院校卡片关键词检索（本地 JSON）
            │         ├─ rag_system.py     简章 RAG（pgvector + 二次生成）
            │         └─ 全局概览
            ├─ append_turn() → Redis（满 20 轮触发摘要压缩，保留最近 6 轮）
            └─ 落咨询记录 + 阶段自动流转 + 长期记忆（全部 best-effort 降级）

RAG 链路：PDF → PyMuPDF 逐页取文字 → 段落→句子→硬切三级分块（重叠窗口 + 噪声过滤）
        → bge-large-zh 向量化（API）→ pgvector 余弦检索（相似度阈值 + 同组最新年份优先）
        → 拼 context 二次生成，标注来源页码
```

| 层        | 选型                                                         |
| --------- | ------------------------------------------------------------ |
| LLM       | DeepSeek（OpenAI 兼容 SDK），可切换 SiliconFlow              |
| Embedding | bge-large-zh-v1.5（SiliconFlow API，1024 维）                |
| 向量存储  | PostgreSQL 16 + pgvector                                     |
| 短期记忆  | Redis 7（TTL + 摘要压缩 + 全链路降级）                       |
| 后端      | FastAPI + SQLModel + Pydantic v2                             |
| 前端      | Streamlit 工作台（另有命令行入口）                           |
| 可观测性  | Langfuse v4（span / generation / tool 三级追踪，可选开启）   |
| 部署      | Dockerfile + docker-compose（应用/后端/PG/Redis 四服务）+ 宿主机 Nginx 反代 |

## 设计决策摘录

这个项目里没有"因为流行所以用"的技术，每个选型都有明确理由：

- **为什么不用 LangChain/LangGraph**：以掌握 Agent 底层机制为首要目标。手搓的工具调用循环
  让轮次上限保护、异常降级、消息回填这些细节完全可控；理解原理之后再对齐框架思想成本很低。
- **为什么 Embedding 走 API 而不是本地部署**：个人项目文本块量级在几百到几千条，API 按量计费
  成本接近零、免运维；且 DeepSeek 无 embeddings 接口，对话与向量化本就需要两条独立配置轨道。
- **为什么没给向量建索引**：实测过小数据量 + ivfflat 近似索引会静默漏召回（不报错、只返回
  错误结果）；当前量级下 PostgreSQL 顺序扫描精确且够快，数据量涨到数万条再按官方建议建索引。
- **为什么 RAG 是"注册成工具"而不是"每轮必检"**：每轮强制检索会让闲聊白跑 Embedding、
  无关片段污染上下文；交给模型按需调用，是从 naive RAG 到 agentic RAG 的关键一步。
- **为什么查别人的学生返回 404 而不是 403**：403 会暴露"这个 ID 存在"。统一 404 不泄露存在性。
- **为什么工具返回统一收口到单一序列化函数**：业务层以 Pydantic 模型为唯一数据词汇表，
  LLM 交互边界只在一个函数里转 JSON，新增工具零改动接入。

## 演进历程

项目从极简的对话循环起步，按真实踩坑顺序迭代了 6 轮：项目详细的演进记录见 [docs/演进记录.md](docs/演进记录.md)

1. **对话骨架**：手搓 Function Calling 双循环（外层多轮对话、内层工具调用）+ 轮次上限保护
2. **记忆系统**：上下文撑爆 → 滑动窗口 → 摘要压缩；跨对话需求 → 长期记忆（早期本地 JSON）
3. **业务落地**：三个资格判断工具 → 院校卡片关键词检索 → RAG 简章问答
   （RAG 内部又经历：内存 numpy 索引 → pgvector；空行分块 → 三级分块引擎；每轮必检 → 按需调用）
4. **数据层重构**：PostgreSQL 持久化 + 三层记忆分工 + Pydantic 统一全部数据模型
5. **工程化**：建档状态机、三入口、API Key 多租户隔离、完整 CRUD、Langfuse 可观测性
6. **部署上线**：Docker 层缓存优化、compose 多服务编排、Nginx 反代、云服务器 + 域名

## 快速开始

### 环境要求

- Python ≥ 3.11
- PostgreSQL 16（启用 pgvector 扩展）
- Redis 7
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

### 安装与配置

```bash
git clone https://github.com/vaseline1/Planner-Copilot-V2.git
cd planner-copilot
uv sync                      # 安装依赖；需要 Langfuse 追踪时：uv sync --extra observability
cp .env.example .env         # 按下方说明填写
```

`.env` 关键配置：

```ini
DEEPSEEK_API_KEY=...                  # 对话模型（必需）
SILICONFLOW_API_KEY=...               # Embedding（RAG 必需）
DATABASE_URL=postgresql://planner:planner123@localhost:5432/planner_copilot
REDIS_URL=redis://localhost:6379/0
PLANNER_API_KEYS=key1:planner_001     # API 鉴权，格式 key:planner_id，逗号分隔多组
# LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY   # 可选，配置后自动开启链路追踪
```

### 运行（三种入口，共用同一套核心）

```bash
python check_setup.py                                   # 环境自检（数据库/Redis/pgvector）
uv run python main.py                                   # 命令行对话
uv run streamlit run app.py                             # Streamlit 工作台（规划师日常用）
uv run uvicorn api:app --host 0.0.0.0 --port 8000       # FastAPI 对外服务
```

### Docker 部署

```bash
docker compose -f deploy/docker-compose.prod.yml --project-directory . up -d --build
```

数据库与 Redis 不映射公网端口，应用端口只绑 127.0.0.1，由宿主机 Nginx 统一反代入口。

## 项目结构

```
├── agent_core.py       # Agent 编排中枢：工具调用循环（全项目唯一认识 OpenAI SDK 的文件）
├── tools.py            # 三个报考资格规则判断工具（零外部依赖）+ Function Calling Schema
├── models.py           # 全项目唯一数据模型定义（Pydantic）
├── rag_system.py       # 简章 RAG：三级分块引擎 + 向量化 + pgvector 检索 + 二次生成
├── memory.py           # 三层记忆：Redis 短期（TTL+摘要压缩）/ PG 长期要点 / 降级保护
├── database.py         # PostgreSQL 持久化：学生档案 + 咨询记录 + 阶段流转
├── knowledge_base.py   # 院校卡片关键词检索（本地 JSON，"先粗后精"的粗层）
├── enrollment_bot.py   # 学生建档状态机
├── batch_analysis.py   # 批量复盘（异步并发 + 结果缓存）
├── api.py              # FastAPI 多租户接口（Depends 注入鉴权）
├── app.py / main.py    # Streamlit 工作台 / 命令行入口
├── config.py           # 双轨道配置（对话 / Embedding 独立切换供应商）
├── prompts.py          # 全部系统级提示词收口
├── observability.py    # Langfuse 薄封装（未配置 Key 时静默降级）
└── test_*.py           # pytest：规则判断 / 知识库检索 / 建档状态机
```

## 关于这个项目

本项目为独立开发的个人作品，从 2025 年 11 月起步，采用对话式 AI 辅助的开发方式——
与 AI 讨论方案、追问机制、逐轮推演，架构设计、技术选型与全部代码决策由本人主导。
项目留有从极简双循环到当前架构的完整演进记录，每个模块为什么存在、当时踩过什么坑、
做过哪些取舍，均可对照演进文档与源码逐一复核。

仓库中的代码注释刻意保留了大量"为什么这样写"的设计动机与实测踩坑记录
（如 ivfflat 索引在小数据量下的静默漏召回、Langfuse v2→v4 接口变迁、
SQLAlchemy 前向引用与 `__future__` 注解的冲突等），作为演进过程的一部分予以保留。
