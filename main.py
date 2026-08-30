"""命令行入口。

用法：
    python main.py "你的编程任务"
    echo "读取当前目录结构并写一个 demo.txt" | python main.py

支持 --offline 以无 API key 的脱机演示方式运行（使用 MockLLM）。
"""

from __future__ import annotations

import argparse
import json
import sys

from agent_core import get_agent
from agent_core.agent import AgentError
from agent_core.config import load_config


def _ensure_utf8_streams() -> None:
    """Windows 控制台常见编码为 GBK/OEM，强制 stdout/stderrstdin 用 UTF-8 显示中文。

    在支持 UTF-8 的终端（Windows Terminal、设定 PYTHONUTF8=1）下可避免中文乱码；
    若终端不支持 UTF-8，这里仅尝试，失败则静默忽略，不影响运行。
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="编程智能体（coding agent）命令行入口")
    p.add_argument("task", nargs="*", help="要让 agent 完成的编程任务；不传则从 stdin 读取")
    p.add_argument("--offline", action="store_true", help="离线演示模式（不调用真实 API）")
    p.add_argument("--workdir", default=None,
                   help="agent 的工作区域（默认取 config.json 的 workdir，缺省为 ./workdir）")
    p.add_argument("--max-iterations", type=int, default=None, help="最大迭代轮数，覆盖配置文件")
    p.add_argument("--json", action="store_true", help="以 JSON 形式输出结果（便于集成/测试）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _ensure_utf8_streams()

    config = load_config(workdir_hint=args.workdir)
    if args.offline:
        config.offline = True
    if args.max_iterations:
        config.max_iterations = args.max_iterations

    task = " ".join(args.task).strip()

    try:
        agent = get_agent(config)
    except (ValueError, AgentError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1

    if sys.stdin.isatty() and not task:
        print(
            "Mini Coding Agent 交互模式：直接输入任务并回车即可开始；"
            "输入 -reset 开启新会话；输入 -quit 退出。",
            file=sys.stderr,
        )

    first = bool(task)
    while True:
        if first:
            first = False  # 使用命令行传入的首条任务
        else:
            task = _read_prompt()
        if not task:
            return 0  # 无更多输入（EOF）→ 正常退出
        if task == "-quit":
            return 0
        if task == "-reset":
            agent.reset()
            print("（已开启新会话，历史与摘要已清空）", file=sys.stderr)
            continue

        try:
            result = agent.run(task)
        except (ValueError, AgentError) as exc:
            print(f"运行失败：{exc}", file=sys.stderr)
            continue

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_human(result)

    return 0


def _read_prompt() -> str:
    """从 stdin 读取一行交互输入；EOF 时返回空字符串。"""
    line = sys.stdin.readline()
    return line.strip()


def _print_human(result: dict) -> None:
    print("=" * 60)
    print(f"任务：{result['task']}")
    print(f"轮数：{result['steps']}    用时：{result['elapsed_seconds']} 秒")
    print("=" * 60)
    for t in result["trace"]:
        if t.get("tool_calls"):
            print(f"[步 {t['step']}] 调用工具：{', '.join(t['tool_calls'])}")
        if t.get("answer"):
            print(f"[完成的] {t['answer']}")
    print("-" * 60)
    print("最终答案：")
    print(result["answer"])
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
