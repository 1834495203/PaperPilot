import type { AgentEvent, JsonValue, PaperResult } from "@/lib/types";

interface EventTimelineProps {
  events: AgentEvent[];
}

function readString(payload: Record<string, JsonValue>, key: string): string | undefined {
  const value = payload[key];
  return typeof value === "string" ? value : undefined;
}

function readNumber(payload: Record<string, JsonValue>, key: string): number | undefined {
  const value = payload[key];
  return typeof value === "number" ? value : undefined;
}

function readBoolean(payload: Record<string, JsonValue>, key: string): boolean | undefined {
  const value = payload[key];
  return typeof value === "boolean" ? value : undefined;
}

function readStringArray(payload: Record<string, JsonValue>, key: string): string[] {
  const value = payload[key];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function displayJsonValue(value: JsonValue | undefined): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function sourceLabel(event: AgentEvent): string {
  switch (readString(event.payload, "source")) {
    case "model":
      return "AI 原话";
    case "policy":
      return "策略规则";
    case "tool":
      return "工具执行";
    case "external":
      return "外部结果";
    case "workflow":
      return "工作流";
    default:
      return event.type.startsWith("run.") ? "工作流" : "系统";
  }
}

function isPaper(value: JsonValue): value is PaperResult {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    typeof value.title === "string" &&
    typeof value.paper_id === "string" &&
    typeof value.source === "string" &&
    Array.isArray(value.authors)
  );
}

function papersFromEvent(event: AgentEvent): PaperResult[] {
  const papers = event.payload.papers;
  return Array.isArray(papers) ? papers.filter(isPaper) : [];
}

function eventLabel(event: AgentEvent): string {
  switch (event.type) {
    case "run.started":
      return "任务开始";
    case "stage.started":
      return readString(event.payload, "summary") ?? "Agent 阶段开始";
    case "decision.recorded":
      if (readString(event.payload, "source") === "policy") {
        return `策略调整 · ${readString(event.payload, "policy_rule") ?? "安全规则"}`;
      }
      if (readString(event.payload, "actor") === "supervisor") {
        return `Supervisor 选择 ${readString(event.payload, "next_agent") ?? "下一步"}`;
      }
      if (readString(event.payload, "actor") === "search") {
        return readString(event.payload, "stage") === "screening"
          ? "Search 完成候选筛选"
          : "Search 生成工具调用";
      }
      if (readString(event.payload, "actor") === "reader") {
        if (readString(event.payload, "stage") === "reader.plan") return "Reader 制定检索计划";
        if (readString(event.payload, "stage") === "reader.assess") return "Reader 评估证据覆盖";
        return "Reader 形成阅读结论";
      }
      if (readString(event.payload, "actor") === "analyst") return "Analyst 形成分析结论";
      return "Agent 已记录决策";
    case "tool.started":
      return `调用 ${readString(event.payload, "tool_name") ?? "工具"}`;
    case "tool.completed":
      if (readString(event.payload, "tool_name") === "fetch_arxiv_pdf") {
        return `PDF 解析完成 · ${readNumber(event.payload, "extracted_pages") ?? 0} 页`;
      }
      if (readString(event.payload, "tool_name") === "retrieve_indexed_paper") {
        return [
          `向量召回 ${readNumber(event.payload, "initial_hit_count") ?? 0}`,
          `树扩展 ${readNumber(event.payload, "expanded_candidate_count") ?? 0}`,
          `rerank 返回 ${readNumber(event.payload, "hit_count") ?? 0}`,
        ].join(" · ");
      }
      return `工具返回 ${readNumber(event.payload, "result_count") ?? 0} 篇论文`;
    case "tool.failed":
      return `工具失败：${readString(event.payload, "error") ?? "未知错误"}`;
    case "message.completed":
      return event.payload.has_tool_calls === true ? "Agent 已生成工具调用" : "回答生成完成";
    case "metrics.updated":
      return `本轮累计 ${readNumber(event.payload, "total_tokens") ?? 0} tokens`;
    case "run.completed":
      return "任务完成";
    case "run.failed":
      return `任务失败：${readString(event.payload, "error") ?? "未知错误"}`;
    case "run.cancelled":
      return readString(event.payload, "summary") ?? "任务已停止";
    case "message.token":
      return "";
  }
}

interface DetailGroupProps {
  label: string;
  values: string[];
}

function DetailGroup({ label, values }: DetailGroupProps) {
  if (values.length === 0) return null;
  return (
    <div className="event-detail-group">
      <span>{label}</span>
      {values.length === 1 ? <p>{values[0]}</p> : (
        <ul>{values.map((value, index) => <li key={`${label}-${index}`}>{value}</li>)}</ul>
      )}
    </div>
  );
}

function EventDetails({ event }: { event: AgentEvent }) {
  if (event.type === "tool.failed") {
    const category = readString(event.payload, "error_category");
    const statusCode = readNumber(event.payload, "status_code");
    const retryAfter = readNumber(event.payload, "retry_after_seconds");
    const retryable = readBoolean(event.payload, "retryable");
    return (
      <div className="event-details">
        <DetailGroup label="失败类型" values={category ? [category] : []} />
        <DetailGroup
          label="恢复信息"
          values={[
            ...(statusCode === undefined ? [] : [`HTTP ${statusCode}`]),
            ...(retryable === undefined ? [] : [retryable ? "可重试" : "不可重试"]),
            ...(retryAfter === undefined ? [] : [`建议等待 ${retryAfter} 秒`]),
          ]}
        />
      </div>
    );
  }
  if (
    event.type === "tool.completed" &&
    readString(event.payload, "tool_name") === "retrieve_indexed_paper"
  ) {
    return (
      <div className="event-details">
        <DetailGroup label="RAG 流程" values={[
          `向量初始召回：${readNumber(event.payload, "initial_hit_count") ?? 0}`,
          `树结构扩展候选：${readNumber(event.payload, "expanded_candidate_count") ?? 0}`,
          `混合 rerank 最终证据：${readNumber(event.payload, "hit_count") ?? 0}`,
        ]} />
      </div>
    );
  }
  if (event.type !== "decision.recorded") return null;
  const summary = readString(event.payload, "summary");
  const objective = readString(event.payload, "objective");
  const query = readString(event.payload, "query");
  const domain = readString(event.payload, "domain");
  const researchProblem = readString(event.payload, "research_problem");
  const evidenceScope = readString(event.payload, "evidence_scope");
  const noveltyAssessment = readString(event.payload, "novelty_assessment");
  const originalValue = displayJsonValue(event.payload.original_value);
  const effectiveValue = displayJsonValue(event.payload.effective_value);
  const adjustment = originalValue !== undefined || effectiveValue !== undefined
    ? [`${originalValue ?? "未提供"} → ${effectiveValue ?? "未提供"}`]
    : [];

  return (
    <div className="event-details">
      <DetailGroup label={readString(event.payload, "source") === "model" ? "AI 的说明" : "规则说明"} values={summary ? [summary] : []} />
      <DetailGroup label="观察到" values={readStringArray(event.payload, "observations")} />
      <DetailGroup label="仍缺少" values={readStringArray(event.payload, "missing_information")} />
      <DetailGroup label="证据缺口" values={readStringArray(event.payload, "missing_requirements")} />
      <DetailGroup label="需要证据" values={readStringArray(event.payload, "evidence_requirements")} />
      <DetailGroup label="执行目标" values={objective ? [objective] : []} />
      <DetailGroup label="检索词" values={query ? [query] : []} />
      <DetailGroup label="策略改写" values={adjustment} />
      <DetailGroup label="领域" values={domain ? [domain] : []} />
      <DetailGroup label="研究问题" values={researchProblem ? [researchProblem] : []} />
      <DetailGroup label="证据范围" values={evidenceScope ? [evidenceScope] : []} />
      <DetailGroup label="局限" values={readStringArray(event.payload, "limitations")} />
      <DetailGroup label="Novelty 判断" values={noveltyAssessment ? [noveltyAssessment] : []} />
      <DetailGroup label="Research gaps" values={readStringArray(event.payload, "research_gaps")} />
      <DetailGroup label="未解决问题" values={readStringArray(event.payload, "unresolved_questions")} />
    </div>
  );
}

export function EventTimeline({ events }: EventTimelineProps) {
  const visibleEvents = events.filter((event) => event.type !== "message.token");
  return (
    <section className="timeline" aria-label="Agent 执行轨迹">
      <div className="section-heading">
        <h2>执行轨迹</h2>
        <span>{visibleEvents.length} events</span>
      </div>
      {visibleEvents.length === 0 ? (
        <p className="empty">发送问题后，这里会展示阶段、工具调用、耗时与用量。</p>
      ) : (
        <ol>
          {visibleEvents.map((event) => {
            const papers = papersFromEvent(event);
            return (
              <li key={event.id} className={`event event-${event.type.replace(".", "-")}`}>
                <div className="event-line">
                  <span className="event-dot" />
                  <div className="event-heading">
                    <span className={`event-source source-${readString(event.payload, "source") ?? "system"}`}>
                      {sourceLabel(event)}
                    </span>
                    <strong>{eventLabel(event)}</strong>
                  </div>
                  <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
                </div>
                <EventDetails event={event} />
                {event.type === "tool.started" ? (
                  <pre>{JSON.stringify(event.payload.arguments, null, 2)}</pre>
                ) : null}
                {papers.length > 0 ? (
                  <div className="paper-list">
                    {papers.map((paper) => (
                      <a href={paper.landing_page_url} target="_blank" rel="noreferrer" key={paper.paper_id}>
                        <strong>{paper.title}</strong>
                        <span>{paper.authors.slice(0, 3).join(", ")} · {paper.source}</span>
                      </a>
                    ))}
                  </div>
                ) : null}
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
