# Mini Coding Agent：从零实现的编程智能体

Mini Coding Agent 是一个轻量级命令行编程智能体。用户给出任务后，它把对话历史和
本地工具定义发送给支持原生 Tool Calling 的大语言模型；模型选择下一步操作；程序
在指定工作区执行工具，再把结构化结果返回模型。这个过程持续循环，直到模型确认
任务完成或程序触发终止条件。

项目没有使用 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI 等
Agent 框架，也不依赖服务端托管的代码执行或文件工具。对话循环、上下文管理、工具
派发、本地执行、模型输出解析、API重试、终止条件和错误处理均由本项目自行实现。

## 主要功能

- 五个文件工具：列出、读取、搜索、创建和精确修改文件。
- 一个命令工具：非 Shell 执行构建、测试和工作区内程序，支持超时及标准输入。
- Windows 下可直接运行 `./program.exe`，同时兼容普通 PATH 命令。
- 文件路径限制在一个明确工作区内，并屏蔽凭据及内部目录。
- 子进程不会继承 API Key、Token、Secret 或 Password 类环境变量。
- 上下文过长时按完整 assistant-tool 轮次压缩，保持消息结构合法。
- 最大步骤数、连续工具错误、上下文异常和用户中断均可终止循环。
- 终端展示执行进度，本地 JSONL 日志记录每次决策、操作和结果。
- 模型客户端可以完全 Mock，核心测试不需要消耗 API 额度。

## 架构

```text
CLI -> Settings -> CodingAgent -> OpenAIChatClient -> 大语言模型
                     |
                     v
                ToolRegistry
                /          \
           文件工具      命令工具
                \          /
                  本地工作区
```

模型不能直接访问本地文件或终端，只能请求预定义的 JSON Schema 工具。详细流程和
设计取舍见 [`docs/架构设计.md`](docs/架构设计.md)。

## 环境要求

- Python 3.11 或更高版本；
- 支持 OpenAI Chat Completions Function Calling 的模型接口；
- 对应模型服务的 API Key；
- 目标项目需要的编译器或测试工具，例如 `g++`、`pytest`。

## 安装

建议在独立虚拟环境中进入仓库根目录执行：

```powershell
python -m pip install -e ".[dev]"
```

## 配置

真实凭据只能放在环境变量中。程序不会自动读取 `.env` 文件。

```powershell
$env:AGENT_API_KEY = "你的API-Key"
$env:AGENT_MODEL = "你的模型名称"
$env:AGENT_BASE_URL = "兼容接口地址"
$env:AGENT_THINKING_MODE = "disabled"
```

只有接口需要显式控制思考模式时才设置 `AGENT_THINKING_MODE`。其他可选配置见
[`.env.example`](.env.example)。

## 运行

始终把 `--workspace` 指向具体目标项目，不要指向包含无关项目的上级目录：

```powershell
python -m mini_agent --workspace D:\code\target-project "检查项目，修复失败测试并验证结果。"
```

也可以使用安装后的控制台命令：

```powershell
mini-agent --workspace D:\code\target-project "增加输入校验和对应测试。"
```

常用选项：

```text
--max-steps N   覆盖本次运行的最大模型步骤数
--no-log        不生成 .mini-agent/runs/*.jsonl 本地记录
--version       显示版本
```

## 本地工具

| 工具 | 作用 | 关键限制 |
| --- | --- | --- |
| `list_files` | 查看项目结构 | 跳过内部、缓存和凭据目录 |
| `read_file` | 按行读取 UTF-8 文本 | 限制文件大小和读取行数 |
| `search_text` | 搜索字面字符串 | 限制结果数量 |
| `write_file` | 创建文件 | 覆盖已有文件必须明确声明 |
| `replace_in_file` | 精确替换文本 | 匹配次数必须符合预期 |
| `run_command` | 构建、测试、运行程序 | 非 Shell、超时、输出限制、凭据隔离 |

## 测试

```powershell
python -m pytest
python -m pytest -q examples\todo_demo_template
```

测试覆盖配置校验、模型响应解析、路径逃逸、凭据文件保护、文件修改、命令标准输入、
Windows 本地程序解析、命令超时、子进程凭据隔离、上下文压缩、终止条件、日志和示例
生成器。

## 可复现示例

生成一个全新的 Todo CLI 目标项目：

```powershell
python -m scripts.create_demo_workspace D:\code\todo-agent-demo
python -m pytest -q D:\code\todo-agent-demo
```

具体任务示例见 [`docs/使用示例.md`](docs/使用示例.md)。逐文件说明见
[`docs/代码导读.md`](docs/代码导读.md)。

## 安全边界与局限

路径限制、凭据屏蔽、非 Shell 命令、危险命令检查、超时和输出截断用于减少误操作，
但不构成操作系统沙箱。被允许执行的项目程序仍可能访问当前用户拥有权限的资源。
处理不可信代码时，应使用一次性副本、容器或虚拟机。

