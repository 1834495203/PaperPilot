# PaperPilot

PaperPilot 是一个可观测的同步多 Agent 文献研究骨架：Supervisor 理解用户目标，
在 Search、Reader、Analyst 之间动态选择下一步，最后由 Writer 生成回答。FastAPI
通过 SSE 返回 LangGraph 执行事件，对话消息、运行、工具调用和关键事件持久化到 SQLite。

## 当前范围

- Next.js + React + strict TypeScript
- FastAPI 异步 API 和 `text/event-stream`
- LangGraph `Supervisor → specialized agent → Supervisor → Writer` 状态循环
- DeepSeek（OpenAI-compatible API）和结构化 Tool Calling
- arXiv Atom API 检索
- Search / Reader / Analyst / Writer 职责隔离
- SQLite + SQLAlchemy 2.0 异步持久化
- token 流、阶段摘要、工具参数、论文结果、耗时和 token usage 展示

“思考过程”采用可审计的执行摘要和状态事件表示，不展示或保存模型隐藏的 chain-of-thought。

## 架构边界

```text
frontend
  └─ typed API client + SSE parser + presentation components

backend/app
  ├─ domain           实体、枚举、抽象端口
  ├─ application      用例编排和任务生命周期
  ├─ infrastructure
  │  ├─ agent         LangGraph 图和事件发布
  │  ├─ tools         arXiv API 适配器
  │  └─ db            SQLAlchemy 持久化适配器
  └─ api              FastAPI DTO、依赖和路由
```

当前顶层工作流：

```text
START → Supervisor ─┬→ Search ───┐
                    ├→ Reader ───┤
                    ├→ Analyst ──┤
                    └→ Writer → END
                                  │
             Search/Reader/Analyst┘ 回到 Supervisor
```

Supervisor 只产生经过 Pydantic 校验的结构化决策。中间 Agent 返回内存中的
结构化 Artifact，只有 Writer 的最终回答作为 assistant message 持久化并通过 SSE
流式展示。`SUPERVISOR_MAX_STEPS` 限制同步调度轮数，达到预算后强制进入 Writer。

执行轨迹用来源标签区分 `AI 原话`、`工作流`、`策略规则`、`工具执行` 和`外部结果`。
Supervisor 会输出自己的观察、缺失信息、选择理由、执行目标和完成标准；如果确定性
策略覆盖模型决定，轨迹会同时保留模型原始选择与策略改写，避免把代码行为伪装成
模型判断。这些内容是模型生成的可审计说明，不是隐藏的逐 Token Chain-of-Thought。

Reader MVP 可以分析用户粘贴的论文内容和 arXiv 元数据/摘要，还没有实现 PDF 二进制
下载、版面解析和逐页证据定位；遇到这类输入时最终回答必须明确证据范围。

Agent 节点不直接请求 arXiv，应用服务不直接操作 SQLAlchemy，前端不依赖 LangGraph 的内部事件格式。接口使用 ABC、Pydantic、dataclass、TypedDict 和 TypeScript interface 明确表达。

## 本地启动

### 1. 配置

```powershell
Copy-Item .env.example .env
```

填写 `DEEPSEEK_API_KEY`。默认模型为 `deepseek-v4-flash`，默认服务地址为
`https://api.deepseek.com`。也支持通用的 `LLM_API_KEY`、`LLM_BASE_URL`、
`LLM_MODEL`，并兼容原有的 `OPENAI_*` 变量名。

### 2. 后端

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -e "backend[dev]"
backend/.venv/Scripts/python.exe -m uvicorn app.main:app --reload --app-dir backend
```

API 文档位于 `http://localhost:8000/docs`。

### 3. 前端

```powershell
Set-Location frontend
npm install
npm run dev
```

打开 `http://localhost:3000`。

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/conversations` | 创建对话 |
| `GET` | `/api/v1/conversations` | 获取持久化对话列表 |
| `GET` | `/api/v1/conversations/{id}/messages` | 获取消息历史 |
| `GET` | `/api/v1/conversations/{id}/metrics` | 获取会话累计指标 |
| `GET` | `/api/v1/conversations/{id}/events` | 获取持久化执行轨迹 |
| `POST` | `/api/v1/conversations/{id}/messages/stream` | 发送消息并读取 SSE |

SSE 事件包括：

- `run.started` / `run.completed` / `run.failed`
- `stage.started` / `decision.recorded`
- `message.token` / `message.completed`
- `tool.started` / `tool.completed` / `tool.failed`
- `metrics.updated`

数据库额外保存 `agent_runs`、`tool_calls` 和精简后的 `agent_events`。Token chunk
只通过 SSE 实时传输，不写入数据库；`agent_events` 只保留运行生命周期、决策摘要、
工具轨迹和聚合指标，后续可以直接增加运行回放与评估。

`conversation_metrics` 以 `conversation_id` 为主键，同一会话每结束一轮都会根据
`agent_runs` 明细重新汇总并覆盖同一行，累计 Token、总耗时、LLM/Tool 调用次数和
运行次数；读取指标时也会自动重建，因此升级前已有的 Run 可以自动回填。

## 质量检查

```powershell
backend/.venv/Scripts/python.exe -m ruff check backend
backend/.venv/Scripts/python.exe -m mypy backend/app
backend/.venv/Scripts/python.exe -m pytest backend/tests

Set-Location frontend
npm run lint
npm run build
```

## 下一步适合扩展

1. 将 SQLite 换成 PostgreSQL，并为 LangGraph 增加数据库 checkpointer，实现断点恢复。
2. 增加 PDF 获取、解析和 pgvector 检索，但保持论文搜索工具与索引管道独立。
3. 将同步的多篇 Reader 调度升级为 Orchestrator–Worker 并行执行。
4. 增加固定 benchmark，评估路由、工具成功率、引用准确率和端到端延迟。
