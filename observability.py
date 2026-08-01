"""observability.py —— Langfuse 可观测性开关封装

职责：
    只有当 .env 里同时配置了 LANGFUSE_PUBLIC_KEY 和 LANGFUSE_SECRET_KEY 时，
    才真正启用 Langfuse 追踪；没配置时，这里导出的 observe/更新函数全部
    安全地什么都不做——既不报错，也不会往终端刷"未配置"的警告。

为什么不直接在业务代码里 `from langfuse import observe`：
    Langfuse 官方装饰器在没配置 Key 时虽然也不报错（我实测过），但每次
    调用都会往 stderr 打一条认证警告——普通规划师每问一句话终端就滚一屏
    警告，会以为程序坏了。这里做一层薄封装，没配置就彻底安静；配置好了，
    行为和直接用官方 API 完全一样，不打折扣。

    另外，这样封装还有个好处：以后 Langfuse 的 SDK 接口再变（它已经从
    v2 的 `langfuse.decorators.observe` 换成了现在 v4 这种基于
    OpenTelemetry 的 `langfuse.observe`），只用改这一个文件，业务代码
    （agent_core.py 等）永远只 import 这里的 observe，不直接依赖 Langfuse
    包的具体版本。

用法：
    from observability import observe, update_current_span, update_current_generation, flush

    @observe(name="my_function")
    def my_function(): ...
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def is_langfuse_enabled() -> bool:
    """两个 Key 都配置了才算启用——只配一个通常是漏填，同样按未启用处理。"""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


_langfuse_active = False

if is_langfuse_enabled():
    try:
        from langfuse import get_client
        from langfuse import observe as observe  # noqa: PLC0414 —— 显式重导出，保持接口稳定

        _langfuse_active = True
    except ImportError:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "检测到 LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY 已配置，但 langfuse 包未安装——"
            "追踪功能不会生效（Agent 其余功能不受影响）。"
            "请执行 `uv sync --extra observability` 补装依赖。"
        )

if _langfuse_active:

    def update_current_span(**kwargs: Any) -> None:
        try:
            get_client().update_current_span(**kwargs)
        except Exception:  # noqa: BLE001 —— 追踪失败不该影响主业务逻辑
            pass

    def update_current_generation(**kwargs: Any) -> None:
        try:
            get_client().update_current_generation(**kwargs)
        except Exception:  # noqa: BLE001
            pass

    def get_trace_url() -> Optional[str]:
        try:
            return get_client().get_trace_url()
        except Exception:  # noqa: BLE001
            return None

    def flush() -> None:
        """把缓冲区里的追踪数据立刻发出去。命令行这种短生命周期脚本运行结束前
        建议调用一次，否则进程退出太快，数据可能还没来得及发送。"""
        try:
            get_client().flush()
        except Exception:  # noqa: BLE001
            pass

else:

    def observe(*args: Any, **kwargs: Any):  # noqa: ANN201, F811 —— 与 try 分支里的 observe 互斥，运行时只存在一个
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func: F) -> F:
            return func

        return decorator

    def update_current_span(**kwargs: Any) -> None:
        pass

    def update_current_generation(**kwargs: Any) -> None:
        pass

    def get_trace_url() -> Optional[str]:
        return None

    def flush() -> None:
        pass
