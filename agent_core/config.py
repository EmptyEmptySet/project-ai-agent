"""配置文件与凭据管理。

设计要点：
- 所有凭据（API key 等）只从环境变量读取，绝不写死在代码或仓库中；
- 也可通过未入库的 config.json 提供（.gitignore 已排除）；
- 优先级：环境变量 > config.json > 默认值。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# 环境变量名（统一前缀 AGENT_，避免与其它程序冲突）
ENV_API_KEY = "AGENT_API_KEY"
ENV_BASE_URL = "AGENT_BASE_URL"
ENV_MODEL = "AGENT_MODEL"
ENV_OFFLINE = "AGENT_OFFLINE"

# 未入库的配置文件路径（.gitignore 中需排除 config.json 以及 .env）
CONFIG_FILE = "config.json"

# 默认工作区域：agent 读写文件、执行命令的根目录（沙箱边界），不再默认项目根目录
DEFAULT_WORKDIR = "workdir"

# 常用 OpenAI 兼容网关的默认地址
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class AgentConfig:
    """Agent 运行所需的全部配置。"""

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    # 工作目录：agent 读写文件、执行命令的根目录（沙箱边界），默认指向 ./workdir
    workdir: str = DEFAULT_WORKDIR
    # 循环终止条件之一：最大交互轮数
    max_iterations: int = 30
    # 单次请求超时（秒）
    timeout: int = 120
    # 上下文裁剪后保留的最近消息条数上限
    max_history_messages: int = 40
    # 捕获到的工具器输出（命令 / 文件）最大字符数，防止上下文爆炸
    max_tool_output_chars: int = 8000
    # 是否开启 mock（脱机演示）模式
    offline: bool = False
    # 额外传给模型实例的参数
    provider_params: dict = field(default_factory=dict)


def _read_config_file(path: str | Path) -> dict:
    """读取未入库的 config.json（若存在）。"""
    p = Path(path)
    if p.exists() and p.is_file():
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _project_root() -> Path:
    """config.json 固定放在项目根目录（即 agent_core 包的上一级），与 workdir 无关。"""
    return Path(__file__).resolve().parent.parent


def load_config(workdir_hint: str | None = None) -> AgentConfig:
    """按优先级加载配置：环境变量 > config.json > 默认值。

    - config.json 一律从项目根目录读取（连接设置），与执行沙箱 workdir 解耦；
    - workdir 解析顺序：workdir_hint(命令行 --workdir) > config.json 的 workdir 字段
      > 默认 DEFAULT_WORKDIR（即 ./workdir）。
    """
    file_cfg = _read_config_file(_project_root() / CONFIG_FILE)

    raw_base = os.environ.get(ENV_BASE_URL) or file_cfg.get("base_url") or DEFAULT_BASE_URL
    raw_model = os.environ.get(ENV_MODEL) or file_cfg.get("model") or DEFAULT_MODEL
    raw_key = os.environ.get(ENV_API_KEY) or file_cfg.get("api_key")
    raw_offline = os.environ.get(ENV_OFFLINE)
    offline = raw_offline.lower() in {"1", "true", "yes"} if raw_offline else bool(
        file_cfg.get("offline", False)
    )

    workdir = workdir_hint or file_cfg.get("workdir") or DEFAULT_WORKDIR

    return AgentConfig(
        api_key=raw_key,
        base_url=raw_base,
        model=raw_model,
        workdir=str(workdir),
        max_iterations=int(file_cfg.get("max_iterations", 30)),
        timeout=int(file_cfg.get("timeout", 120)),
        max_history_messages=int(file_cfg.get("max_history_messages", 40)),
        max_tool_output_chars=int(file_cfg.get("max_tool_output_chars", 8000)),
        offline=offline,
        provider_params=dict(file_cfg.get("provider_params", {})),
    )


def validate_config(config: AgentConfig) -> None:
    """校验配置：在线模式必须提供 API key。"""
    if not config.offline and not config.api_key:
        raise ValueError(
            f"未检测到 API key：请通过环境变量 {ENV_API_KEY} 或未入库的 {CONFIG_FILE} 提供。"
            "如需脱机演示，请设置 offline = true 或环境变量 AGENT_OFFLINE=1。"
        )
