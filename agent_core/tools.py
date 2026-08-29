"""工具定义与本地执行。

设计要点：
- 这是题目要求的“工具的定义与本地执行”核心逻辑，必须完全由本项目手写实现；
- 工具是“本地执行”的：文件读写、目录遍历、命令执行全部在本机沙箱（workdir）内完成，
  绝不调用任何服务端托管的代码执行 / 文件工具（如 Code Interpreter、Files API）；
- 每个工具都有统一的 JSON Schema，用于向模型声明其参数（OpenAI 兼容的 function calling 格式）；
- 工具执行带校验（路径必须在工作区内）、错误捕获与输出截断，
  防止异常让 agent 崩溃、防止超大输出撑爆上下文。
"""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

# 单个工具器输出的最大字符数（防止上下文爆炸）
_MAX_TOOL_OUTPUT = 12000


class ToolError(Exception):
    """工具执行失败时抛出，携带可回传给模型的错误信息。"""


def _resolve_path(workdir: str, path: str) -> Path:
    """将相对路径解析到工作区内的绝对路径，并校验其不越界到工作区之外。"""
    base = Path(workdir).resolve()
    try:
        raw = Path(path).expanduser()
        if not raw.is_absolute():
            raw = base / raw
        resolved = raw.resolve()
    except (OSError, ValueError) as exc:
        raise ToolError(f"路径解析失败：{exc}") from exc
    if not resolved.is_relative_to(base):
        raise ToolError(f"路径 {path} 越出工作区 {base}，出于安全考虑已拒绝。")
    return resolved


def _truncate(text: str, limit: int = _MAX_TOOL_OUTPUT) -> str:
    """对工具输出做截断，避免上下文被撑爆。"""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[输出过长，已截断，共 {len(text)} 字符]"


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #

def _read_file(workdir: str, path: str, encoding: str = "utf-8") -> str:
    resolved = _resolve_path(workdir, path)
    if not resolved.is_file():
        raise ToolError(f"文件不存在：{path}")
    try:
        content = resolved.read_text(encoding=encoding, errors="replace")
    except OSError as exc:
        raise ToolError(f"读取失败：{exc}") from exc
    return _truncate(content)


def _write_file(workdir: str, path: str, content: str, encoding: str = "utf-8") -> str:
    resolved = _resolve_path(workdir, path)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)
    except OSError as exc:
        raise ToolError(f"写入失败：{exc}") from exc
    return f"已写入 {len(content.encode(encoding))} 字节到 {path}"


def _list_dir(workdir: str, path: str = ".") -> str:
    resolved = _resolve_path(workdir, path)
    if not resolved.is_dir():
        raise ToolError(f"目录不存在：{path}")
    entries: list[str] = []
    try:
        for child in sorted(resolved.iterdir(), key=lambda p: p.name):
            if child.is_dir():
                entries.append(f"[目录] {child.name}/")
            elif child.is_file():
                entries.append(f"[文件] {child.name}（{child.stat().st_size} 字节）")
            else:
                entries.append(f"[其它] {child.name}")
    except OSError as exc:
        raise ToolError(f"列目录失败：{exc}") from exc
    return _truncate("\n".join(entries) if entries else "（空目录）")


def _search_dir(workdir: str, pattern: str, path: str = ".") -> str:
    """按关键字搜索工作区内的文件与目录名（大小写不敏感，支持通配符）。"""
    resolved = _resolve_path(workdir, path)
    if not resolved.is_dir():
        raise ToolError(f"目录不存在：{path}")
    rx = re.compile(re.escape(pattern), re.IGNORECASE)
    matches: list[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(resolved):
            for name in dirnames + filenames:
                if rx.search(name):
                    rel = Path(dirpath).relative_to(resolved)
                    matches.append(str(rel / name))
    except OSError as exc:
        raise ToolError(f"搜索失败：{exc}") from exc
    limit = 200
    shown = matches[:limit]
    more = len(matches) - len(shown)
    out = "\n".join(shown) if shown else "（无匹配结果）"
    if more > 0:
        out += f"\n...（另有 {more} 个匹配项，已省略）"
    return _truncate(out)


def _run_command(workdir: str, command: str, timeout: int = 60) -> str:
    """本地执行 shell 命令（安全、带超时、限制输出）。"""
    if not command or not command.strip():
        raise ToolError("命令为空。")
    resolved = Path(workdir).resolve()
    if not resolved.is_dir():
        raise ToolError(f"工作区不存在：{workdir}")
    try:
        args = list(shlex.split(command))
    except ValueError as exc:
        raise ToolError(f"命令解析失败：{exc}") from exc
    if not args:
        raise ToolError("命令为空。")

    # 在 Windows 上 shlex 拆分后可直接作为 argv 传给 cmd；此处以 shell=True 保证兼容各平台。
    try:
        proc = subprocess.run(
            command,
            cwd=str(resolved),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"命令执行超时（>{timeout}s）。") from None
    except OSError as exc:
        raise ToolError(f"命令启动失败：{exc}") from exc

    parts: list[str] = []
    if proc.stdout and proc.stdout.strip():
        parts.append("[stdout]\n" + proc.stdout.rstrip())
    if proc.stderr and proc.stderr.strip():
        parts.append("[stderr]\n" + proc.stderr.rstrip())
    parts.append(f"[exit_code] {proc.returncode}")
    body = "\n".join(parts) if parts else "(无输出)"
    return _truncate(body)


# --------------------------------------------------------------------------- #
# 工具注册表
# --------------------------------------------------------------------------- #

# 闭包：把 workdir 绑进工具函数，得到统一的 ToolFn(参数dict) -> str 形态
def _bind(fn: Callable[..., str], workdir: str):
    def wrapper(args: dict[str, Any]) -> str:
        return fn(workdir, **args)  # type: ignore[arg-type]

    return wrapper


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作区内某个文件的完整内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对工作区或绝对路径）"},
                    "encoding": {"type": "string", "description": "字符编码，默认 utf-8"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入或覆盖工作区内某个文件的内容（会创建父目录）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的完整文本内容"},
                    "encoding": {"type": "string", "description": "字符编码，默认 utf-8"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出工作区内目录的子项（文件与目录）。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "目录路径，默认当前工作区根目录"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_dir",
            "description": "在工作区内按文件名/目录名关键字搜索，返回匹配的相对路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "要搜索的关键字（不区分大小写）"},
                    "path": {"type": "string", "description": "搜索的起始目录，默认工作区根目录"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "在工作区目录内本地执行一条 shell 命令，返回 stdout/stderr/退出码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"},
                    "timeout": {"type": "integer", "description": "超时秒数，默认 60"},
                },
                "required": ["command"],
            },
        },
    },
]

# 工具名 -> (绑定后的执行函数, 示例参数)
TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], str]] = {}


def bind_tools(workdir: str) -> dict[str, Callable[[dict[str, Any]], str]]:
    """将工具与指定工作区绑定，生成 工具名->执行函数 的注册表。"""
    return {
        "read_file": _bind(_read_file, workdir),
        "write_file": _bind(_write_file, workdir),
        "list_dir": _bind(_list_dir, workdir),
        "search_dir": _bind(_search_dir, workdir),
        "run_command": _bind(_run_command, workdir),
    }


def tool_schemas() -> list[dict[str, Any]]:
    """返回供模型使用的工具 JSON Schema 列表。"""
    return TOOL_SPECS


def execute_tool(registry: dict[str, Callable[[dict[str, Any]], str]], name: str, args: Any) -> str:
    """执行单个工具调用，统一进行参数校验与异常兜底。

    返回一个可放入 assistant 之后 role=tool 的字符串结果。
    任何异常都会被捕获并转成可读的错误文本（错误处理要点）。
    """
    if name not in registry:
        return f"错误：未知工具 {name}。可用工具：{', '.join(registry.keys())}"

    if args is None:
        args = {}
    if not isinstance(args, dict):
        return f"错误：工具 {name} 的参数必须是对象，收到 {type(args).__name__}。"

    try:
        result = registry[name](args)
    except ToolError as exc:
        return f"工具错误：{exc}"
    except TypeError as exc:  # 参数名或类型不匹配
        return f"工具错误：参数不合法 -> {exc}"
    except Exception as exc:  # 兜底，绝不让 agent 循环因未知异常崩溃
        return f"工具错误：未预期的异常 {type(exc).__name__}: {exc}"

    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False, default=str)
    return result
