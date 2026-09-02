# PaperPilot

PaperPilot 是一个可观测的同步多 Agent 文献研究骨架：Supervisor 理解用户目标，
在 Search、Reader、Analyst 之间动态选择下一步，最后由 Writer 生成回答。FastAPI
通过 SSE 返回 LangGraph 执行事件，对话消息、运行、工具调用和关键事件持久化到 SQLite。

## 当前范围

- Next.js + React + strict TypeScript
- FastAPI 异步 API 和 `text/event-stream`
- LangGraph `Supervisor → specialized agent → Supervisor → Writer` 状态循环
- DeepSeek（OpenAI-compatible API）和结构化 Tool Calling
- arXiv Atom API 检索、标题/摘要相关性筛选、查询改写和结果去重
- arXiv PDF 安全下载、受限全文提取和带页码证据的快速阅读
- TreeRAG 风格的科学论文解析、层级 Chunk、祖先标题前缀和 Chroma 向量索引
- Chroma 向量召回、按任务模式双向树扩展和轻量混合 rerank
- PDF 上传、本地论文选择和 Reader 驱动的 RAG 证据阅读
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
流式展示。每个中间 Agent 同时产出一份完整报告和一份强类型的 `supervisor_summary`：
Supervisor 只读取后者，不读取、预览或截断完整报告；Reader、Analyst 和 Writer 则按
显式 Artifact ID 获得完整报告，Agent 之间的产物传递不采用字符截断。Supervisor 自己
判断何时进入 Writer；完整回答、有限证据下的部分回答以及外部工具不可用说明都可以是
合理终态，代码不会再因“目标未完全满足”把 Writer 强制改回 Search。
`SUPERVISOR_MAX_STEPS` 只作为防止无限调度的最终安全预算。

Supervisor 的下一步任务使用按 `agent` 区分的联合类型：Search 只接收查询和历史 Search
Artifact；Reader 只接收阅读目标和一个论文 ID，arXiv 论文额外引用一个 Search
Artifact；Analyst/Writer 接收明确的来源
Artifact ID 列表。不同 Agent 的字段不能混用，空列表也不再隐式代表“全部产物”。

执行轨迹用来源标签区分 `AI 原话`、`工作流`、`策略规则`、`工具执行` 和`外部结果`。
Supervisor 会输出自己的观察、缺失信息、选择理由和执行目标；如果确定性
策略覆盖模型决定，轨迹会同时保留模型原始选择与策略改写，避免把代码行为伪装成
模型判断。这些内容是模型生成的可审计说明，不是隐藏的逐 Token Chain-of-Thought。

每次用户请求都会创建独立 `AgentRun`，用户消息、AI 消息、指标和事件通过 `run_id`
关联。前端按一轮问答显示本轮耗时、Token、LLM/工具调用以及可展开的执行轨迹；会话级
累计指标仍作为辅助概览保留。

Search 在单个节点内最多执行 `SEARCH_MAX_ITERATIONS` 轮“检索 → 结构化相关性筛选 →
按缺口改写查询”，将论文区分为直接相关、相邻和无关，并把计数、理由和已尝试查询
交给 Supervisor；是否满足整个用户任务仍由 Supervisor 决定。arXiv 客户端会对请求做
串行限速，尊重数值型 `Retry-After` 并对 429/5xx/网络错误执行有限退避重试。最终失败会
以 `rate_limited`、`provider_error` 等结构化状态进入 Search 摘要，不能再伪装成空结果。
查询中出现 arXiv ID 时直接使用精确 `id_list`，普通关键词则逐项生成 `all:` 字段查询。

Reader 是独立的 LangGraph 子图，在本地论文上执行“规划 → 检索 → 证据检查 → 按缺口
改写查询 → 总结”；检索词、检索模式和补检决定均由 Reader 自己负责，最多执行
`READER_MAX_RETRIEVAL_ROUNDS` 轮。Reader 也可以按 Supervisor 指定的
`paper_id` 下载一篇 arXiv PDF，用 `pypdf` 提取带 `PAGE` 标记的受限全文并快速总结
方法、组件、实验、结果和局限。下载大小、页数和输入字符均有配置上限；扫描版 PDF、
复杂版面/公式视觉理解和 OCR 尚不支持，截断及提取警告会进入证据范围。
Reader 的内部计划和证据状态保持结构化，但不再强制生成包含方法、实验和局限的完整
论文报告；元数据足以回答时会跳过 RAG/PDF 下载，最终由 Writer 根据问题范围自然组织
语言，小问题默认直接回答。

独立的论文建库流程不经过在线 Supervisor：`PypdfScientificPaperParser` 优先复用论文
原生章节编号构建父子层级，`TreeRagChunker` 按段落语义边界生成叶节点，并按照 TreeRAG
公式将论文标题与全部祖先章节标题放在正文之前生成 embedding 输入。根、章节和内容
Chunk 都写入同一 Chroma collection；原文、父节点、子节点、章节路径、页码和内容哈希
作为记录或 metadata 保存。`TreeRagRetriever` 对事实问题仅召回内容叶节点；对总结、方法、
比较和综合任务，允许根/章节/叶节点参与初始召回，再执行 leaf-to-root-to-leaves 扩展，
将候选还原为可引用的内容 Chunk。最终结果先按 embedding 相似度与查询词覆盖率进行轻量
混合 rerank，并限制单篇论文的返回数量。当前不使用 LLM 重建已有论文标题。上传的 PDF
和 manifest 保存在 `PAPER_LIBRARY_PATH`，索引保存在 Chroma；前端选择的 `paper_ids`
会随本轮问题传给 Supervisor。Reader 根据结构化任务调用检索器，完整 Chunk 证据只进入
Reader Artifact，Supervisor 仍只读取 Reader 摘要。

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

### 4. 将论文写入 TreeRAG 向量索引

建库使用独立的 OpenAI-compatible embedding 配置，不继承聊天模型的 `OPENAI_*` 或
`DEEPSEEK_*`。默认连接本机 Ollama `http://localhost:11434/v1`，并使用
`qwen3-embedding:latest`；Ollama 的 API key 字段必须存在但会被本地服务忽略：

```dotenv
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=qwen3-embedding:latest
```

运行建库：

```powershell
Set-Location backend
.venv/Scripts/python.exe -m app.cli.ingest_paper `
  paper/TreeRAG.pdf `
  --paper-id treerag
```

命令会输出页数、章节数、树节点数、内容 Chunk 数和目标 collection。重复使用相同
`paper-id` 会 upsert 新节点，并在成功写入后清理该论文已经失效的旧节点。

检索已入库论文：

```powershell
.venv/Scripts/python.exe -m app.cli.retrieve_paper `
  "How does TreeRAG construct its tree index?" `
  --paper-id treerag `
  --mode method
```

`--paper-id` 可以重复传入以检索多篇论文；`--mode` 支持 `fact`、`summary`、`method`、
`compare` 和 `synthesis`。JSON 结果保留章节路径、页码、原始向量分数、rerank 分数以及
候选来自直接向量召回还是树扩展。当前 rerank 是零额外模型依赖的轻量实现，不等同于
cross-encoder；接入专用 reranker 后可以替换排序策略而不改变检索结果类型。

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | 健康检查 |
| `POST` | `/api/v1/conversations` | 创建对话 |
| `DELETE` | `/api/v1/conversations/{id}` | 删除会话及其运行数据 |
| `GET` | `/api/v1/conversations` | 获取持久化对话列表 |
| `GET` | `/api/v1/conversations/{id}/messages` | 获取消息历史 |
| `GET` | `/api/v1/conversations/{id}/metrics` | 获取会话累计指标 |
| `GET` | `/api/v1/conversations/{id}/events` | 获取持久化执行轨迹 |
| `POST` | `/api/v1/conversations/{id}/runs/{run_id}/cancel` | 停止正在执行的任务 |
| `POST` | `/api/v1/conversations/{id}/messages/stream` | 发送消息并读取 SSE |
| `POST` | `/api/v1/papers` | 上传 PDF、解析并写入 TreeRAG 索引 |
| `GET` | `/api/v1/papers` | 获取本地已建库论文 |
| `GET` | `/api/v1/papers/{paper_id}` | 获取论文树索引与 Chunk 预览 |
| `POST` | `/api/v1/papers/retrieve` | 直接调试 TreeRAG 检索结果 |

SSE 事件包括：

- `run.started` / `run.completed` / `run.failed` / `run.cancelled`
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
2. 为 Reader 增加 query decomposition、证据充分性判断和二次检索闭环。
3. 将同步的多篇 Reader 调度升级为 Orchestrator–Worker 并行执行。
4. 增加固定 benchmark，评估路由、工具成功率、引用准确率和端到端延迟。
