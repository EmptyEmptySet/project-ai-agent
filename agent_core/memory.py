"""对话历史与上下文管理。

设计要点（对应题目要求“对话历史与上下文管理”）：
- 维护一份消息列表，兼容 OpenAI 的 role / tool_call / tool_role 结构；
- 提供 token 近似估算；
- 按“token 预算”裁剪（而非仅按条数），默认给出推荐值；
- 支持“新会话重置”（reset），与交互入口的 quit/reset 命令配套；
- 长时间对话/超预算时，把“上一轮已完成对话”压缩成摘要，并记录所有摘要的机制
  （同时解决“旧答案污染新任务”的问题）；
- 摘要以历史上下文形式在 to_payload 时注入到系统提示之后，保证单条 system 消息兼容性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# 推荐的默认 token 预算：留足上下文窗口给“模型响应 + 当前轮工具结果”。
# 大多数 OpenAI 兼容模型上下文 >= 32k，取 24000 给响应与工具留缓冲。
DEFAULT_TOKEN_BUDGET = 24000

# 保留“最近 N 个完整轮次”不参与压缩裁剪，保证进行中/最新上下文完整。
_RECENT_ROUNDS_KEEP = 1

# 注入上下文的历史摘要条数上限（完整记录保存在 self.summaries，避免无限膨胀）。
_MAX_SUMMARIES_IN_CONTEXT = 4

# 单条摘要的最大字符数（内置简易摘要器用）。
_SUMMARY_MAX_CHARS = 500


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：英文约 4 字符/token，中文约 1.5 字/token。

    仅用于近似判断上下文大小，无需精确。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4) + 1


def simple_summarize(round_text: str, max_chars: int = _SUMMARY_MAX_CHARS) -> str:
    """内置简易摘要器（用于离线/无 LLM 的兜底）：归一化空白后做首部截取。

    接入真实模型时可用 LLM 摘要器替换（见 agent 层）。返回一行摘要文本。
    """
    text = " ".join((round_text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…（已压缩）"


@dataclass
class ConversationMessage:
    """一条对话消息（兼容 OpenAI 协议）。"""

    role: str  # system / user / assistant / tool
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  # assistant 发起的工具调用

    def to_dict(self) -> dict[str, Any]:
        """转成发送给模型的字典；空字段不输出。"""
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d

    def token_size(self) -> int:
        size = estimate_tokens(self.content or "")
        if self.tool_calls:
            size += estimate_tokens(str(self.tool_calls))
        return size


class ConversationMemory:
    """管理对话历史，提供追加、按 token 预算裁剪、轮次压缩与摘要记录能力。"""

    def __init__(
        self,
        system_prompt: str,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        recent_rounds_keep: int = _RECENT_ROUNDS_KEEP,
        max_summaries_in_context: int = _MAX_SUMMARIES_IN_CONTEXT,
        summary_max_chars: int = _SUMMARY_MAX_CHARS,
        summarizer: Callable[[str], str] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.token_budget = token_budget
        self.recent_rounds_keep = recent_rounds_keep
        self.max_summaries_in_context = max_summaries_in_context
        self.summary_max_chars = summary_max_chars
        if summarizer is not None:
            self.summarizer = summarizer
        else:
            self.summarizer = lambda text: simple_summarize(text, self.summary_max_chars)

        # 保存“激活窗口”内的摘要（机制：#3）：≤ max_summaries_in_context 条，
        # 超出时会做“分层压缩”（telescoping，把最旧一组再合成 1 条），见 _condense_summaries。
        self.summaries: list[str] = []
        # 触发“4 条合成 1 条”的次数（用于观察分层深度）
        self.condense_count = 0
        self.messages: list[ConversationMessage] = []
        self._tool_call_seq = 0

        self.reset()

    # ------------------------------------------------------------------ #
    # 新会话 / 追加 / 查询
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """重置为新会话：清空对话历史与所有摘要（机制：#1 reset/新会话）。"""
        self.messages = [ConversationMessage(role="system", content=self.system_prompt)]
        self.summaries = []
        self.condense_count = 0
        self._tool_call_seq = 0

    def add_user(self, content: str) -> None:
        self.messages.append(ConversationMessage(role="user", content=content))

    def add_assistant(self, content: str | None, tool_calls: list[dict[str, Any]] | None = None) -> None:
        self.messages.append(
            ConversationMessage(role="assistant", content=content, tool_calls=tool_calls)
        )

    def add_tool(self, tool_call_id: str, name: str, output: str) -> None:
        self.messages.append(
            ConversationMessage(role="tool", tool_call_id=tool_call_id, name=name, content=output)
        )

    def next_tool_call_id(self) -> str:
        self._tool_call_seq += 1
        return f"call_{self._tool_call_seq}"

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def summary_count(self) -> int:
        return len(self.summaries)

    # ------------------------------------------------------------------ #
    # 摘要：写入 / 分层压缩 / 注入
    # ------------------------------------------------------------------ #

    def _add_summary(self, summary: str) -> None:
        """新增一条摘要，并在超过注入窗口上限时触发分层压缩。"""
        text = (summary or "").strip()
        if not text:
            return
        self.summaries.append(text)
        self._condense_summaries()

    def _condense_summaries(self) -> None:
        """分层压缩（telescoping：4 条合成 1 条）。

        当摘要条数超过 max_summaries_in_context 时，把“最旧的一组”（前 N 条）
        再次压缩成 1 条更高层的摘要，使激活窗口始终 <= max_summaries_in_context。
        这样既避免摘要无限膨胀，又能把更早的历史逐层凝结到更高层的摘要里。
        """
        n = max(1, self.max_summaries_in_context)
        while len(self.summaries) > n:
            group = self.summaries[:n]  # 最旧的一组
            if len(group) < 2:
                break
            combined = "\n".join(f"[摘要] {s}" for s in group)
            condensed = (self.summarizer(combined) or "").strip()
            if not condensed:
                break
            self.summaries = [condensed] + self.summaries[n:]
            self.condense_count += 1

    def _summaries_context(self) -> str:
        """返回将要注入系统提示的摘要上下文（激活窗口内 <= max_summaries_in_context 条）。"""
        if not self.summaries:
            return ""
        recent = self.summaries[-self.max_summaries_in_context:]
        block = "# 历史对话摘要（此为之前已压缩轮的记录，仅供参考，非当前指令）：\n"
        block += "\n".join(f"- {s}" for s in recent)
        return block

    def total_tokens(self) -> int:
        """对话总量 token：消息 + 将要注入的摘要。"""
        base = sum(m.token_size() for m in self.messages)
        base += estimate_tokens(self._summaries_context())
        return base

    def is_over_budget(self) -> bool:
        return self.total_tokens() > self.token_budget

    def _round_ranges(self) -> list[tuple[int, int]]:
        """返回每个“用户消息开始的轮次”范围 [(start, end)]，不含 system。"""
        ranges: list[tuple[int, int]] = []
        start = None
        for i, m in enumerate(self.messages):
            if m.role == "user":
                if start is not None:
                    ranges.append((start, i))
                start = i
        if start is not None:
            ranges.append((start, len(self.messages)))
        return ranges

    # ------------------------------------------------------------------ #
    # 压缩 / 裁剪
    # ------------------------------------------------------------------ #

    def _summarize_round(self, start: int, end: int) -> str:
        round_msgs = self.messages[start:end]
        text = "\n".join(self._msg_to_text(m) for m in round_msgs)
        return self.summarizer(text)

    @staticmethod
    def _msg_to_text(m: ConversationMessage) -> str:
        if m.role == "user":
            return f"用户：{m.content}"
        if m.role == "assistant":
            if m.tool_calls:
                names = ", ".join(tc.get("function", {}).get("name", "?") for tc in m.tool_calls)
                return f"助手：调用工具 [{names}]"
            return f"助手：{m.content}"
        if m.role == "tool":
            return f"工具结果 [{m.name}]：{m.content}"
        return m.content or ""

    def compress_previous_round(self) -> str | None:
        """把“最近一个已完成的轮次”压缩成摘要并记录（机制：#3、#4）。

        若最近轮次为主对话消息而无后续内容（如仅有一个 user 消息），则不压缩。
        返回生成的摘要文本；无内容可压缩时返回 None。
        """
        ranges = self._round_ranges()
        if not ranges:
            return None
        start, end = ranges[-1]
        if end - start < 2:  # 只有 user，无 assistant/tool 结果，不值得压缩
            return None
        summary = self._summarize_round(start, end)
        if summary and summary.strip():
            self._add_summary(summary)
            del self.messages[start:end]
            return summary
        return None

    def enforce_token_budget(self) -> bool:
        """按 token 预算裁剪（机制：#2）。

        优先“整体压缩最旧的完整轮次”，绝不拆散一轮（避免破坏 user/assistant/tool
        的配对关系）；最近 _RECENT_ROUNDS_KEEP 轮次保留下不压缩。
        极端兜底：仅剩单个超大轮次时，截断最长的工具输出（无害）。
        返回是否有任何调整发生。
        """
        if not self.is_over_budget():
            return False
        changed = False

        # 1) 压缩最旧的可压缩轮次，直到不超预算或没有更多可压缩轮次
        while self.is_over_budget():
            ranges = self._round_ranges()
            compressible = ranges[:max(0, len(ranges) - self.recent_rounds_keep)]
            if not compressible:
                break
            start, end = compressible[0]  # 最旧
            summary = self._summarize_round(start, end)
            if summary and summary.strip():
                self._add_summary(summary)
                del self.messages[start:end]
                changed = True
            else:
                break

        # 2) 极端兜底：无可压缩轮次但仍超预算时，截断最长的工具输出（不破坏配对）
        guard = 0
        while self.is_over_budget() and guard < len(self.messages) + 1:
            guard += 1
            heavy = max(
                (m for m in self.messages if m.role == "tool" and m.content),
                default=None,
                key=lambda m: estimate_tokens(m.content or ""),
            )
            if heavy is None:
                break
            new_len = int(len(heavy.content) * 0.8)
            if new_len < 50:
                break
            heavy.content = heavy.content[:new_len]
            changed = True

        return changed

    # 兼容旧名：trim 现在等价于按 token 预算裁剪
    def trim(self) -> None:
        self.enforce_token_budget()

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #

    def to_payload(self) -> list[dict[str, Any]]:
        """先按 token 预算裁剪，再注入摘要上下文，返回发送给模型的字典列表。"""
        self.enforce_token_budget()
        msgs = [m.to_dict() for m in self.messages]
        context = self._summaries_context()
        if context and msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = (msgs[0].get("content") or "") + "\n\n" + context
        return msgs

    def compact_summary_placeholder(self) -> str:
        """历史压缩摘要机制已实现；此处保留占位以向后兼容。"""
        return ""
