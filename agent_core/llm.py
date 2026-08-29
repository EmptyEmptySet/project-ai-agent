"""大模型接入层（LLM Provider）。

设计要点：
- 这是题目要求的“模型输出的解析”与对外模型调用逻辑，由项目自行实现；
- 使用原始 HTTP（requests）直连 OpenAI 兼容网关的 /chat/completions 接口，
  因而“模型原生 tool calling”的授权即可，但无需依赖任何 agent 框架 / SDK；
- 由于不借助厂商托管工具，模型的服务端只负责“生成”，所有文件/命令动作都在本地完成；
- 提供统一接口 generate()，返回解析后的 LLMResult；
- 另提供 MockLLM（脱机演示 / 自动化测试用），不消耗真实 API。

凭据说明：api_key 仅从 config 注入，绝不写入任何文件或日志。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import AgentConfig
from .memory import estimate_tokens


@dataclass
class ParsedToolCall:
    """从模型输出中解析出的单个工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class LLMResult:
    """模型一次生成的解析结果。"""

    content: str | None = None
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def final_text(self) -> str | None:
        return (self.content or "").strip() or None


class OpenAICompatLLM:
    """通过原始 HTTP 调用 OpenAI 兼容网关（含原生 tool calling）。"""

    def __init__(self, config: AgentConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model
        self.timeout = config.timeout
        self.provider_params = dict(config.provider_params)

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        """调用模型并解析输出（这是核心的“模型输出解析”）。"""
        if not self.api_key:
            raise ValueError("缺少 API key，无法发起在线请求。")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            **self.provider_params,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(
                self._endpoint(), headers=headers, json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise LLMError(f"网络请求失败：{exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"模型接口返回状态码 {resp.status_code}：{resp.text[:500]}")

        data = resp.json()
        return self._parse_response(data)

    def _parse_response(self, data: dict[str, Any]) -> LLMResult:
        """解析模型原始 JSON 响应，提取 content 与 tool_calls。"""
        try:
            choices = data.get("choices") or []
            message = choices[0].get("message") or {} if choices else {}
        except (IndexError, TypeError, AttributeError) as exc:
            raise LLMError(f"响应结构异常：{data}") from exc

        content = message.get("content")

        tool_calls: list[ParsedToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                # 某些模型返回的 arguments 不是合法 JSON 时，当成字符串处理
                arguments = {"_raw_arguments": raw_args}
            if not isinstance(arguments, dict):
                arguments = {"_raw_arguments": arguments}
            tool_calls.append(
                ParsedToolCall(
                    id=tc.get("id") or "",
                    name=fn.get("name") or "",
                    arguments=arguments,
                )
            )

        return LLMResult(content=content, tool_calls=tool_calls, raw=data)

    def usage_estimate(self, messages: list[dict[str, Any]]) -> int:
        """可选：估算本次请求的输入 token（用于上下文管理）。"""
        return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


class MockLLM(OpenAICompatLLM):
    """脱机模拟模型：不消耗真实 API，用于演示与自动化测试。

    script 驱动：每次 generate 依次返回下一个脚本项；
    脚本项形如：
        {"content": "..."}                              # 最终答案（结束循环）
        {"tool_calls": [{"name": "read_file", "arguments": {...}}, ...]}  # 发起工具调用
    脚本用尽后，默认返回一条最终答案，避免死循环。
    """

    def __init__(self, config: AgentConfig, script: list[dict[str, Any]] | None = None) -> None:
        super().__init__(config)
        self.script = list(script or [])
        self._index = 0
        self._seq = 0

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        if self._index < len(self.script):
            step = self.script[self._index]
            self._index += 1
        else:
            step = {"content": "最终答案：演示完成。"}

        content = step.get("content")
        tool_calls: list[ParsedToolCall] = []
        for tc in step.get("tool_calls") or []:
            self._seq += 1
            tool_calls.append(
                ParsedToolCall(
                    id=tc.get("id", f"mock_call_{self._seq}"),
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                )
            )
        return LLMResult(content=content, tool_calls=tool_calls, raw={})


class LLMError(Exception):
    """模型调用层的异常。"""
