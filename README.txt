Mini Coding Agent —— 简版编程智能体

一、Git 仓库地址
https://github.com/EmptyEmptySet/project-ai-agent

二、如何运行
（本程序运行于 Windows 环境，命令基于 Windows 命令行/PowerShell；run_command 工具与提示词均已按 Windows 适配。）
1. 安装依赖：pip install -r requirements.txt
2. 提供凭据（二选一）：
   · 环境变量：AGENT_API_KEY / AGENT_BASE_URL / AGENT_MODEL
   · 或编辑项目根 config.json，现提供config-example.json作为格式参考
3. 运行任务：python main.py "具体任务"
   · --workdir 指定任务区（默认 ./workdir 沙箱）
   · --offline 无密钥的离线演示
   · --json 机器可读输出
   · 交互模式：直接 python main.py，逐行输入任务；输入 reset 开新会话，quit 退出。

三、特色功能说明
· 本地沙箱工具：read_file / write_file / list_dir / search_dir / run_command，全部在本机执行，越界访问会被拒绝。
· 上下文记忆：按 token 预算裁剪；每个任务完成时把上一轮压缩成摘要；摘要满 4 条会把最旧 4 条再合成 1 条（分层凝固），避免旧答案污染新任务。
· 错误处理：模型调用失败自动重试退避；工具异常被捕获并回传；工具输出超长自动截断。

四、其它说明
· 修改 config.json 的 base_url / model 即可切换模型；token_budget、max_tool_output_chars 等均可调。
· 离线模式使用内置 MockLLM 演示循环，不调用真实模型，也不会改动任务区文件。
