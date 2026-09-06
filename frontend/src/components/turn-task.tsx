import { EventTimeline } from "@/components/event-timeline";
import { readCitations } from "@/lib/api";
import type { AgentEvent, AgentRun, Citation, Message } from "@/lib/types";
import type { ReactNode } from "react";

interface TurnTaskProps {
  run: AgentRun;
  userMessage: Message | undefined;
  assistantMessage: Message | undefined;
  events: AgentEvent[];
  onCitationClick: (citation: Citation) => void;
  onRegenerate?: () => void;
  regenerateDisabled?: boolean;
}

function durationLabel(durationMs: number): string {
  return durationMs > 0 ? `${(durationMs / 1000).toFixed(2)}s` : "执行中";
}

function renderContentWithCitations(
  content: string,
  citations: Citation[],
  onCitationClick: (citation: Citation) => void,
) {
  const parts = content.split(/\[E-[0-9a-f]{12}\]/g);
  const markers = content.match(/\[E-[0-9a-f]{12}\]/g) ?? [];
  const citationIndex = new Map(citations.map((citation, index) => [
    citation.evidence_id,
    index + 1,
  ]));
  const nodes: ReactNode[] = [];
  parts.forEach((part, index) => {
    if (index > 0) {
      const marker = markers[index - 1];
      const evidenceId = marker?.slice(1, -1);
      const citation = evidenceId
        ? citations.find((item) => item.evidence_id === evidenceId)
        : undefined;
      if (citation) {
        nodes.push(
          <button
            className="citation-chip"
            key={`${marker}-${index}`}
            onClick={() => onCitationClick(citation)}
            title={`${citation.paper_title} · P${citation.page_start ?? "?"}`}
            type="button"
          >
            [{citationIndex.get(citation.evidence_id) ?? "?"}]
          </button>,
        );
      }
    }
    if (part) nodes.push(<span key={`text-${index}`}>{part}</span>);
  });
  return nodes;
}

export function TurnTask({
  run,
  userMessage,
  assistantMessage,
  events,
  onCitationClick,
  onRegenerate,
  regenerateDisabled,
}: TurnTaskProps) {
  const citations = assistantMessage ? readCitations(assistantMessage) : [];
  return (
    <section className="turn-task">
      {userMessage ? (
        <article className="bubble user">
          <span>YOU</span><p>{userMessage.content}</p>
        </article>
      ) : null}
      {assistantMessage ? (
        <article className="bubble assistant">
          <span>PAPERPILOT</span>
          <p>{renderContentWithCitations(assistantMessage.content, citations, onCitationClick)}</p>
        </article>
      ) : null}
      {onRegenerate ? (
        <button
          className="regenerate-button"
          disabled={regenerateDisabled}
          onClick={onRegenerate}
          type="button"
        >重新生成</button>
      ) : null}
      <div className="turn-metrics" aria-label="本次任务指标">
        <span>{durationLabel(run.duration_ms)}</span>
        <span>{run.total_tokens.toLocaleString()} tokens</span>
        <span>{run.llm_calls} LLM</span>
        <span>{run.tool_calls} tools</span>
        <span className={`run-status status-${run.status}`}>{run.status}</span>
      </div>
      <details className="turn-trace">
        <summary>执行轨迹 · {events.length} events</summary>
        <EventTimeline events={events} />
      </details>
    </section>
  );
}
