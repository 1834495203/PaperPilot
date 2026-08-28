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

function isPaper(value: JsonValue): value is PaperResult {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    typeof value.title === "string" &&
    typeof value.arxiv_id === "string" &&
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
    case "tool.started":
      return `调用 ${readString(event.payload, "tool_name") ?? "工具"}`;
    case "tool.completed":
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
    case "message.token":
      return "";
  }
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
                  <strong>{eventLabel(event)}</strong>
                  <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
                </div>
                {event.type === "tool.started" ? (
                  <pre>{JSON.stringify(event.payload.arguments, null, 2)}</pre>
                ) : null}
                {papers.length > 0 ? (
                  <div className="paper-list">
                    {papers.map((paper) => (
                      <a href={paper.abstract_url} target="_blank" rel="noreferrer" key={paper.arxiv_id}>
                        <strong>{paper.title}</strong>
                        <span>{paper.authors.slice(0, 3).join(", ")}</span>
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

