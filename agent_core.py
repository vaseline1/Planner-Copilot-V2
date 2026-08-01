"""agent_core.py —— Agent 大脑：编排"对话 → 工具调用 → 生成回答"这条主线

职责：
    定义 PlannerAgent 类，整合全部 6 个工具（3 个规则判断 + 关键词检索
    + RAG 问答 + 概览），编排"拼消息 → 发请求 → 按需调用工具 → 拿到最终
    回答 → 写记忆 → 落库"这条完整链路。

为什么这是整个项目里唯一"认识"OpenAI SDK 的文件：
    "关注点分离"——config.py 只管配置，tools.py/knowledge_base.py/
    rag_system.py 各自管好自己的工具逻辑，agent_core.py 负责把它们
    编排起来。以后想加第7个工具，只用在 TOOLS 列表和 TOOL_REGISTRY
    里各加一条，这个文件的整体结构不用动。

三期改造后的变化：
    - 新增 _serialize_tool_result()：tools.py 的判断函数现在返回
      EligibilityResult 模型，不能直接 json.dumps。统一在这里转成
      to_display_dict() 的中文键 dict 再序列化——喂给大模型的内容
      与旧版完全一致，Langfuse 里的工具输出也可读；
    - 咨询阶段自动流转：学生首次触发 answer_from_brochure（说明规划师
      开始研究简章细节了），建档学生的阶段自动从"初次建档"推进到
      "规划进行中"（只进不退，见 database.advance_stage_to_planning）。
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from openai import APIError, OpenAI

from config import MAX_TOOL_CALL_ROUNDS, get_chat_config
from database import add_consultation, advance_stage_to_planning
from knowledge_base import get_tianjin_summary, search_programs
from memory import append_turn, format_memories_for_context, get_conversation_history, save_memory
from models import PathType
from observability import flush, observe, update_current_generation, update_current_span
from prompts import SYSTEM_PROMPT
from rag_system import answer_question
from tools import TOOL_REGISTRY as CORE_TOOL_REGISTRY
from tools import TOOLS as CORE_TOOLS

logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """Agent 对话过程中发生不可恢复错误时抛出，message 面向终端用户可读。"""


# ── 额外的3个工具的 Schema（规则判断3个工具的 Schema 在 tools.py 里）──
EXTRA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_programs",
            "description": "关键词搜索天津院校招生项目（适合'天津有哪些院校有综评'类问题）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，如'南开''强基'"},
                    "path_type_str": {"type": "string", "description": "路径类型过滤（可选），如'强基计划'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_from_brochure",
            "description": "从招生简章原文中检索回答（适合'校测占比''专业录取规则'等需要简章原文细节的问题）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "具体问题"},
                    "university": {"type": "string", "description": "目标院校名称（提高检索精度，可选）"},
                    "path_type": {"type": "string", "description": "路径类型（可选）"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tianjin_overview",
            "description": "获取天津市升学路径全局概览（强基/综评院校数量和基本情况）。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOLS = CORE_TOOLS + EXTRA_TOOLS


def _serialize_tool_result(result) -> str:
    """工具执行结果 → 喂回给大模型的 JSON 字符串。

    三期改造后 tools.py 返回 EligibilityResult 模型，不能 json.dumps。
    约定：带 to_display_dict() 方法的模型转成中文键 dict（与旧版散装
    dict 完全一致），其余（dict / list / 基本类型）原样序列化。
    以后新增工具只要遵守这个约定，这里不用改。
    """
    if hasattr(result, "to_display_dict"):
        return json.dumps(result.to_display_dict(), ensure_ascii=False)
    if isinstance(result, (list, tuple)):
        items = [item.to_display_dict() if hasattr(item, "to_display_dict") else item for item in result]
        return json.dumps(items, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


def _dispatch_search_programs(query: str, path_type_str: Optional[str] = None) -> dict:
    path_type = PathType(path_type_str) if path_type_str else None
    results = search_programs(query, path_type=path_type)
    return {"results": results, "count": len(results)}


def _dispatch_answer_from_brochure(
    question: str, university: Optional[str] = None, path_type: Optional[str] = None
) -> dict:
    return {"answer": answer_question(question, university, path_type)}


def _dispatch_get_tianjin_overview() -> dict:
    return get_tianjin_summary()


TOOL_REGISTRY = {
    **CORE_TOOL_REGISTRY,
    "search_programs": _dispatch_search_programs,
    "answer_from_brochure": _dispatch_answer_from_brochure,
    "get_tianjin_overview": _dispatch_get_tianjin_overview,
}


class PlannerAgent:
    """PlannerCopilot 完整版 Agent，整合全部 6 个工具。"""

    def __init__(self) -> None:
        cfg = get_chat_config()
        self._model = cfg.model
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    @observe(name="planner_agent_chat")
    def chat(self, user_message: str, session_id: str = "default", student_id: Optional[int] = None) -> str:
        """处理规划师的一句话，返回 Agent 的最终文字回答。

        （用 @observe 包起来：没配置 Langfuse 时这行装饰器等于没写，行为
        完全不变；配置了 Langfuse 之后，每一次调用都会在 Langfuse 后台
        生成一条完整的追踪记录——用了哪些工具、每次调用大模型花了多少
        Token、总共耗时多久，见 observability.py 顶部注释。）
        """
        if not user_message.strip():
            return "（没有收到任何内容，请重新输入。）"

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if student_id is not None:
            messages.append({"role": "system", "content": format_memories_for_context(student_id)})
        messages.extend(get_conversation_history(session_id))
        messages.append({"role": "user", "content": user_message})

        try:
            final_reply, tools_called = self._run_tool_loop(messages)
        except APIError as exc:
            logger.error("调用大模型 API 失败：%s", exc)
            return f"❌ 调用大模型接口失败，暂时无法回答。\n错误详情：{exc}\n常见原因：API Key 无效 / 账户余额不足 / 网络无法访问。"
        finally:
            flush()  # 命令行这类短生命周期脚本，进程退出前把追踪数据发出去

        append_turn(session_id, "user", user_message)
        append_turn(session_id, "assistant", final_reply)

        if student_id is not None:
            # 咨询阶段自动流转：首次触发简章问答 → "初次建档"升级为"规划进行中"
            #（只进不退；best-effort，失败不影响本轮回答）
            if "answer_from_brochure" in tools_called:
                try:
                    advance_stage_to_planning(student_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("咨询阶段自动流转失败（不影响本轮回答）：%s", exc)
            try:
                add_consultation(student_id=student_id, topic="Agent 对话", content=user_message, agent_response=final_reply)
            except Exception as exc:  # noqa: BLE001
                logger.warning("咨询记录写入数据库失败（不影响本轮回答）：%s", exc)
            # 长期记忆：把本轮问答的精炼要点写入（save_memory 按 student_id+title
            # 去重覆盖，所以这个学生永远只保留"最近一轮"的要点，不会越攒越多）。
            # 隔天重开这个学生时，format_memories_for_context 就能读到最新进展。
            # best-effort：回答已经生成，记忆写入失败只记日志，绝不影响返回。
            try:
                save_memory(
                    student_id,
                    "对话要点",
                    "最近一轮对话要点",
                    f"规划师问：{user_message[:120]}\n助手回复：{final_reply[:300]}",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("长期记忆写入失败（不影响本轮回答）：%s", exc)

        return final_reply

    def _run_tool_loop(self, messages: list[dict]) -> tuple[str, list[str]]:
        """工具调用循环。返回（最终回复, 本轮调用过的工具名列表）——
        工具名列表给 chat() 做咨询阶段自动流转判断用。"""
        tools_called: list[str] = []
        for _round in range(MAX_TOOL_CALL_ROUNDS):
            response = self._call_llm(messages)
            choice = response.choices[0]
            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                update_current_span(metadata={"tools_called": tools_called, "rounds_used": _round + 1})
                return (choice.message.content or "（模型没有返回任何内容）", tools_called)

            messages.append(choice.message.model_dump(exclude_none=True))
            for tool_call in choice.message.tool_calls:
                tools_called.append(tool_call.function.name)
                result = self._execute_tool(tool_call.function.name, tool_call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": _serialize_tool_result(result)})

        update_current_span(metadata={"tools_called": tools_called, "terminated_reason": "max_rounds_reached"})
        return (
            "⚠️ 已达到单轮最大工具调用次数，为避免无限循环提前终止。请尝试把问题拆得更具体一些再试一次。",
            tools_called,
        )

    @observe(as_type="generation", name="llm_call")
    def _call_llm(self, messages: list[dict]):
        """单独抽出这一步（而不是直接把 create() 调用写在 _run_tool_loop 里），
        是为了能单独标记成 Langfuse 的 "generation" 类型观测——只有标记成
        generation，Langfuse 才会把 usage_details 换算成 Token 用量和费用
        展示在界面上，标记成普通 span 是不会显示这些指标的。"""
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, tools=TOOLS, tool_choice="auto", max_tokens=1500
        )
        if response.usage:
            update_current_generation(model=response.model, usage_details=response.usage.model_dump(exclude_none=True))
        return response

    @staticmethod
    @observe(as_type="tool")
    def _execute_tool(name: str, raw_arguments: str):
        """执行单个工具调用。

        `@observe(as_type="tool")` 不带固定 name，是因为这个函数被全部
        6 个工具共用——如果写死 name="execute_tool"，Langfuse 后台看到
        的全是一样的标签，分不清哪次调用的是 check_qiangji_eligibility、
        哪次是 answer_from_brochure。进函数体第一行就用
        update_current_span(name=...) 把当前这一次的 span 动态改名成
        实际工具名，这样每个工具在 Langfuse 里各自是一条独立、可筛选、
        能看到单独耗时的记录。
        """
        update_current_span(name=name)
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {"error": f"工具参数解析失败：{exc}"}

        func = TOOL_REGISTRY.get(name)
        if func is None:
            return {"error": f"未知工具：{name}"}

        try:
            result = func(**arguments)
            # 结果可能是 EligibilityResult 模型，序列化成可读的 dict 再上报 Langfuse
            update_current_span(output=json.loads(_serialize_tool_result(result)))
            return result
        except TypeError as exc:
            return {"error": f"工具 {name} 参数不匹配：{exc}"}
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("工具 %s 执行时发生未预期的错误", name)
            return {"error": f"工具 {name} 执行失败：{exc}", "fallback": "请规划师直接查阅相关简章"}
