import { EventTimeline } from "@/components/event-timeline";
import type { AgentEvent, AgentRun, Message } from "@/lib/types";

interface TurnTaskProps {
  run: AgentRun;
  userMessage: Message | undefined;
  assistantMessage: Message | undefined;
  events: AgentEvent[];
}

function durationLabel(durationMs: number): string {
  return durationMs > 0 ? `${(durationMs / 1000).toFixed(2)}s` : "执行中";
}

export function TurnTask({ run, userMessage, assistantMessage, events }: TurnTaskProps) {
  return (
    <section className="turn-task">
      {userMessage ? (
        <article className="bubble user">
          <span>YOU</span><p>{userMessage.content}</p>
        </article>
      ) : null}
      {assistantMessage ? (
        <article className="bubble assistant">
          <span>PAPERPILOT</span><p>{assistantMessage.content}</p>
        </article>
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
