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

class Agent:
    def __init__(self, config: AgentConfig) -> None:
        validate_config(config)
        self.config = config
        self.llm = self._build_llm()
        self.memory = ConversationMemory(
            system_prompt=SYSTEM_PROMPT,
            token_budget=config.token_budget,
            recent_rounds_keep=config.recent_rounds_keep,
            max_summaries_in_context=config.max_summaries_in_context,
            summary_max_chars=config.summary_max_chars,
            summarizer=self._make_summarizer(),
        )
        self.registry = None  # lazy：run 时按 workdir 绑定工具

    def _build_llm(self):
        if self.config.offline or not self.config.api_key:
            return MockLLM(self.config)
        return OpenAICompatLLM(self.config)

    def _make_summarizer(self):
        """在线模型用 LLM 压缩摘要；Mock 用内置简易摘要器。"""
        if isinstance(self.llm, MockLLM):
            return None  # 交给 memory 使用内置 simple_summarize
        return self._llm_summarize

    def _llm_summarize(self, text: str) -> str:
        prompt = (
            "请把下面这段对话记录压缩成不超过 80 字的中文摘要，突出关键动作、结论和已完成事项，"
            "作为历史参考（不要写成指令）：\n" + text
        )
        try:
            result = self.llm.generate([{"role": "user", "content": prompt}], tools=None)
            s = (result.content or "").strip()
            return s or text[:200]
        except (LLMError, ToolError):
            return text[:200]

    def reset(self) -> None:
        """开启新会话：清空对话历史与所有摘要（对应交互命令 reset）。"""
        self.memory.reset()

    def run(self, task: str, max_iterations: int | None = None) -> dict[str, Any]:
        """执行一次完整任务，返回结果统计与最终答案。"""
        from .tools import bind_tools

        # 新任务开始：先压缩上一轮——既控住长对话(token 预算)，也避免旧答案污染新任务
        self.memory.compress_previous_round()

        self.registry = bind_tools(
            self.config.workdir,
            output_limit=self.config.max_tool_output_chars,
            command_timeout=self.config.command_timeout,
        )
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
            trace[-1]["results"] = self._execute_and_append(tool_calls)

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
        max_retries = self.config.max_retries
        backoff = self.config.retry_backoff_seconds
        for attempt in range(1, max_retries + 1):
            try:
                result: LLMResult = self.llm.generate(payload, tools=tool_schemas())
                return self._handle_result(result)
            except (LLMError, ToolError) as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
        raise AgentError(f"模型调用连续失败 {max_retries} 次：{last_error}")

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

    def _execute_and_append(self, tool_calls: list[Any]) -> list[str]:
        """本地逐个执行工具调用，并把结果以 role=tool 追加回对话历史。

        返回“本轮”执行的结果列表，供 trace 记录，避免回溯整个累积历史导致跨轮不精确。
        """
        results: list[str] = []
        for tc in tool_calls:
            result = execute_tool(self.registry, tc.name, tc.arguments)
            self.memory.add_tool(tc.id, tc.name, result)
            results.append(result)
        return results


class AgentError(Exception):
    """agent 运行层的异常。"""
