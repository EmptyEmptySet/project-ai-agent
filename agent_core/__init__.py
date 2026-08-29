"""agent_core —— 编程智能体（coding agent）核心包。

本包暴露 get_agent() 工厂函数，用于构建并运行 agent。
"""

from .agent import Agent
from .config import AgentConfig
from .llm import MockLLM, OpenAICompatLLM
from .tools import TOOL_REGISTRY


def get_agent(config: AgentConfig):
    """根据配置构建 Agent 实例。"""
    return Agent(config=config)


__all__ = [
    "Agent",
    "AgentConfig",
    "MockLLM",
    "OpenAICompatLLM",
    "TOOL_REGISTRY",
    "get_agent",
]
