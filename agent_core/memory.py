"""对话历史与上下文管理。

设计要点：
- 这是题目要求的“对话历史与上下文管理”核心逻辑，必须由项目手写实现；
- 维护一份消息列表，兼容 OpenAI 的 role / tool_call / tool_role 结构；
- 提供 token 近似估算，用于触摸上下文窗口时的自动裁剪；
- 裁剪策略：优先丢弃最早的非系统消息，保证系统提示词与最近历史始终保留；
- 同时限制单条工具结果的最大长度（在 tools 层已截断），两层防护上下文爆炸。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：英文约 4 字符/token，中文约 1.5 字/token。

    仅用于近似判断上下文大小，无需精确。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4) + 1


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
    """管理对话历史，提供追加与裁剪能力。"""

    def __init__(self, system_prompt: str, max_messages: int = 40) -> None:
        self.max_messages = max_messages
        self.messages: list[ConversationMessage] = [
            ConversationMessage(role="system", content=system_prompt)
        ]
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

    def total_tokens(self) -> int:
        return sum(m.token_size() for m in self.messages)

    def trim(self) -> None:
        """按 max_messages 裁剪历史：保留 system 与最近的若干条消息。

        防御性处理：绝不连续裁剪时把 system 也丢掉。
        """
        if len(self.messages) <= self.max_messages:
            return
        keep_tail = self.max_messages - 1  # 扣除 system 占用的 1 条
        head = [self.messages[0]]  # system
        tail = self.messages[-(keep_tail):]
        self.messages = head + tail

    def to_payload(self) -> list[dict[str, Any]]:
        self.trim()
        return [m.to_dict() for m in self.messages]

    def compact_summary_placeholder(self) -> str:
        """供扩展：若未来做历史压缩摘要，可在此实现。当前返回空。"""
        return ""
