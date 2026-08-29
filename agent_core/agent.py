"""Agent：编程智能体的核心交互循环（ReAct 风格）。

设计要点（对应题目要求）：
- 对话历史与上下文管理：借助 ConversationMemory 维护 messages 并按窗口裁剪；
- 工具的定义与本地执行：借助 tools.execute_tool 在本地沙箱执行；
- 模型输出的解析：由 LLMClient.generate 返回的结构化 LLMResult 驱动；
- 循环终止条件：① 模型给出不含工具调用的最终答案；② 达到最大迭代次数；
- 错误处理：工具异常被捕获并回传给模型；模型调用异常有重试与安全退出。

单轮流程：
  user(prompt)
      ↓
  调用模型 → 解析出 content + tool_calls
      ↓
  有 tool_calls？ ──是──▶ 逐个本地执行工具 → 结果 append 回对话 → 继续下一轮
      ↓
     否（content 为最终答案）
      ↓
  结束，返回最终答案
"""

from __future__ import annotations

import time
from typing import Any

from .config import AgentConfig, validate_config
from .llm import LLMError, LLMResult, MockLLM, OpenAICompatLLM
from .memory import ConversationMemory
from .prompts import SYSTEM_PROMPT
from .tools import execute_tool, tool_schemas, ToolError

# 模型调用失败的最大重试次数（错误处理要点）
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.0


class Agent:
    def __init__(self, config: AgentConfig) -> None:
        validate_config(config)
        self.config = config
        self.memory = ConversationMemory(
            system_prompt=SYSTEM_PROMPT, max_messages=config.max_history_messages
        )
        self.registry = None  # lazy：run 时按 workdir 绑定工具
        self.llm = self._build_llm()

    def _build_llm(self):
        if self.config.offline or not self.config.api_key:
            return MockLLM(self.config)
        return OpenAICompatLLM(self.config)

    def run(self, task: str, max_iterations: int | None = None) -> dict[str, Any]:
        """执行一次完整任务，返回结果统计与最终答案。"""
        from .tools import bind_tools

        self.registry = bind_tools(self.config.workdir)
        self.memory.add_user(task)
        limit = max_iterations or self.config.max_iterations

        trace: list[dict[str, Any]] = []
        final_answer: str | None = None
        start = time.time()

        for step in range(1, limit + 1):
            tool_calls = self._call_llm()
            trace.append({"step": step, "tool_calls": [t.name for t in tool_calls]})
            if not tool_calls:
                final_answer = self._final_content()
                trace[-1]["answer"] = final_answer
                break
            self._execute_and_append(tool_calls)
            trace[-1]["results"] = [m.content for m in self.memory.messages if m.role == "tool"][-len(tool_calls):]

        elapsed = time.time() - start
        if final_answer is None:
            final_answer = f"已达到最大迭代次数（{limit}），任务未能完成。请检查任务复杂度或增加 max_iterations。"

        return {
            "task": task,
            "answer": final_answer,
            "steps": step if "step" in locals() else 0,
            "trace": trace,
            "elapsed_seconds": round(elapsed, 2),
        }

    # --------------------- 私有辅助 --------------------- #

    def _call_llm(self) -> list[Any]:
        """调用模型并返回本次解析出的工具调用列表（可能为空）。

        内置重试与退避，避免偶发网络/限流导致循环中断。
        """
        payload = self.memory.to_payload()
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result: LLMResult = self.llm.generate(payload, tools=tool_schemas())
                return self._handle_result(result)
            except (LLMError, ToolError) as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
        raise AgentError(f"模型调用连续失败 {_MAX_RETRIES} 次：{last_error}")

    def _handle_result(self, result: LLMResult) -> list[Any]:
        """把模型结果写入对话历史并返回工具调用列表。"""
        if result.has_tool_calls:
            self.memory.add_assistant(result.content, [t.to_dict() for t in result.tool_calls])
            return result.tool_calls
        self.memory.add_assistant(result.content)
        return []

    def _final_content(self) -> str | None:
        """取最近一条 assistant 的 content 作为最终答案。"""
        for m in reversed(self.memory.messages):
            if m.role == "assistant":
                return m.content
        return None

    def _execute_and_append(self, tool_calls: list[Any]) -> None:
        """本地逐个执行工具调用，并把结果以 role=tool 追加回对话历史。"""
        for tc in tool_calls:
            result = execute_tool(self.registry, tc.name, tc.arguments)
            self.memory.add_tool(tc.id, tc.name, result)


class AgentError(Exception):
    """agent 运行层的异常。"""
