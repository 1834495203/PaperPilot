# PaperPilot

PaperPilot 是一个可观测的单 Agent 文献检索骨架：用户从 Next.js 页面提问，FastAPI 通过 SSE 返回 LangGraph 执行事件，Agent 自主决定是否调用 arXiv，所有对话消息、运行、工具调用和事件都持久化到 SQLite。

## 当前范围

- Next.js + React + strict TypeScript
- FastAPI 异步 API 和 `text/event-stream`
- LangGraph `agent → tools → agent` 状态循环
- DeepSeek（OpenAI-compatible API）和结构化 Tool Calling
- arXiv Atom API 检索
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
| `POST` | `/api/v1/conversations/{id}/messages/stream` | 发送消息并读取 SSE |

SSE 事件包括：

- `run.started` / `run.completed` / `run.failed`
- `stage.started`
- `message.token` / `message.completed`
- `tool.started` / `tool.completed` / `tool.failed`
- `metrics.updated`

数据库额外保存 `agent_runs`、`tool_calls` 和 `agent_events`，后续可以直接增加运行回放与评估。

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
3. 增加固定 benchmark，评估工具成功率、引用准确率、token 和端到端延迟。
