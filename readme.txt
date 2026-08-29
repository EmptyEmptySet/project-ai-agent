项目名称：Mini Coding Agent（简版编程智能体）

Git 仓库地址：（请在 git init 后补充你的实际仓库地址，例如 https://github.com/your-name/project-ai-agent）

一、项目整体框架含义
本项目是一个“简化的 Claude Code / Codex / OpenCode”式的编程智能体。它借助大语言模型
（LLM）的思维与工具调用能力，在一个交互循环中自主地“读文件 -> 改文件 -> 执行命令 ->
查看结果 -> 继续推理”，最终完成用户交给的编程任务。整个循环、上下文管理、工具执行、
输出解析全部由本仓库自行编写，不依赖任何 agent 框架 / SDK，也不借助服务端托管的代码或
文件工具。
核心循环为 ReAct 风格：
  user(task) → 模型 → 解析出 content+tool_calls →（有工具调用？）→ 本地执行工具
  → 结果回填对话 → 回到“模型”……直至模型给出无工具调用的最终答案，或达到最大迭代轮数。

二、目录结构与每个文件的作用
project-ai-agent/
├─ main.py                命令行入口：读取任务、加载配置、运行 agent、打印/输出 JSON。
├─ requirements.txt       依赖清单（仅 requests，用于直连 OpenAI 兼容网关）。
├─ .gitignore             排除 config.json/.env 等可能含密钥的未入库文件。
├─ work.txt               从题目 PDF 解析出的任务目标文字稿。
├─ readme.txt             本文件：框架说明与使用指南。
├─ introduce.txt          项目简要介绍（供评分方快速了解项目）。
└─ agent_core/            agent 核心实现
   ├─ __init__.py         包入口，暴露 get_agent() 等便于外部调用。
   ├─ config.py           配置与凭据管理：connect 设置(api/base_url/model)从环境变量/项目根 config.json 读取；
│                      执行沙箱 workdir 独立解析，默认指向 ./workdir。
   ├─ prompts.py          系统提示词，定义 agent 角色、能力边界与工具使用规范。
   ├─ memory.py           对话历史与上下文管理：消息列表、token 估算、窗口自动裁剪。
   ├─ tools.py            工具定义与本地执行：文件读写、列目录、搜索、shell 命令；
   │                      含路径越界校验、错误捕获、输出截断。
   ├─ llm.py              大模型接入层：OpenAICompatLLM 直连兼容网关并手写解析 output；
   │                      MockLLM 用于离线演示 / 自动化测试。
   └─ agent.py            核心循环：每轮调用模型、执行工具、回填结果、判定终止、错误重试。

三、如何运行
1. 安装依赖：pip install -r requirements.txt
2. 提供凭据（二选一）：
   · 设置环境变量：AGENT_API_KEY=你的key  AGENT_BASE_URL=网关地址  AGENT_MODEL=模型名
   · 或配置项目根的 config.json（已被 .gitignore 排除，不会进仓库）
3. 运行任务：
   python main.py "列出当前目录结构，并写一个 hello.txt"
   （默认工作区域为 ./workdir；如需指定别的任务目录，用 --workdir "目录"）
   或 echo "你的任务" | python main.py
4. 无 API key 的演示方式：
   python main.py --offline "模拟一次简单的写文件任务"
   （离线模式使用 MockLLM，不调用真实模型，也不产生真实任务结果，仅用于演示循环如何运转）

四、特色功能说明
· 本地沙箱隔离：所有文件读写与命令执行都限制在 workdir 内，越界路径会被拒绝，保证安全。
· 完整手写闭环：上下文裁剪、工具注册与执行、模型输出解析、终止条件、错误重试均自主实现。
· 双层防上下文爆炸：工具输出截断 + 对话历史窗口裁剪。
· 离线可测：内置 MockLLM，无需密钥即可验证循环与工具链路。

五、其它说明
本项目严格遵守题目规则：不使用任何 agent 框架 / SDK；不使用服务端托管的代码执行或文件工具；
API key 一律通过环境变量或未入库配置文件提供，且已配置 .gitignore 防止误提交。
