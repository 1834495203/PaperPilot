"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import { EventTimeline } from "@/components/event-timeline";
import { MetricsPanel } from "@/components/metrics-panel";
import {
  createConversation,
  listConversations,
  listMessages,
  streamMessage,
} from "@/lib/api";
import type { AgentEvent, Conversation, JsonValue, Message, RunMetrics } from "@/lib/types";

const EMPTY_METRICS: RunMetrics = {
  inputTokens: 0,
  outputTokens: 0,
  totalTokens: 0,
  llmCalls: 0,
  toolCalls: 0,
  durationMs: 0,
};

function numberValue(payload: Record<string, JsonValue>, key: string): number {
  const value = payload[key];
  return typeof value === "number" ? value : 0;
}

export function ChatWorkspace() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [draft, setDraft] = useState("");
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [metrics, setMetrics] = useState<RunMetrics>(EMPTY_METRICS);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadConversation = useCallback(async (conversationId: string) => {
    setActiveId(conversationId);
    setMessages(await listMessages(conversationId));
    setEvents([]);
    setStreamedAnswer("");
    setMetrics(EMPTY_METRICS);
  }, []);

  useEffect(() => {
    const initialize = async () => {
      try {
        const existing = await listConversations();
        if (existing.length > 0 && existing[0] !== undefined) {
          setConversations(existing);
          await loadConversation(existing[0].id);
          return;
        }
        const created = await createConversation("New research");
        setConversations([created]);
        await loadConversation(created.id);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "初始化失败");
      }
    };
    void initialize();
  }, [loadConversation]);

  const renderedMessages = useMemo(
    () => messages.filter((message) => message.role === "user" || message.role === "assistant"),
    [messages],
  );

  const handleEvent = useCallback((event: AgentEvent) => {
    setEvents((current) => [...current, event]);
    if (event.type === "message.token") {
      const text = event.payload.text;
      if (typeof text === "string") setStreamedAnswer((current) => current + text);
    }
    if (event.type === "message.completed" && event.payload.has_tool_calls === true) {
      setStreamedAnswer("");
    }
    if (event.type === "metrics.updated" || event.type === "run.completed") {
      setMetrics({
        inputTokens: numberValue(event.payload, "input_tokens"),
        outputTokens: numberValue(event.payload, "output_tokens"),
        totalTokens: numberValue(event.payload, "total_tokens"),
        llmCalls: numberValue(event.payload, "llm_calls"),
        toolCalls: numberValue(event.payload, "tool_calls"),
        durationMs: numberValue(event.payload, "duration_ms"),
      });
    }
    if (event.type === "run.failed") {
      const message = event.payload.error;
      setError(typeof message === "string" ? message : "Agent 运行失败");
    }
  }, []);

  const handleSubmit = async (submitEvent: FormEvent<HTMLFormElement>) => {
    submitEvent.preventDefault();
    const content = draft.trim();
    if (content.length === 0 || activeId === null || isRunning) return;

    setDraft("");
    setError(null);
    setEvents([]);
    setStreamedAnswer("");
    setMetrics(EMPTY_METRICS);
    setIsRunning(true);
    setMessages((current) => [
      ...current,
      {
        id: `optimistic-${Date.now()}`,
        conversation_id: activeId,
        role: "user",
        content,
        sequence: current.length + 1,
        created_at: new Date().toISOString(),
        metadata: {},
      },
    ]);
    try {
      await streamMessage(activeId, content, handleEvent);
      setMessages(await listMessages(activeId));
      setStreamedAnswer("");
      setConversations(await listConversations());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "请求失败");
    } finally {
      setIsRunning(false);
    }
  };

  const handleNewConversation = async () => {
    const created = await createConversation("New research");
    setConversations((current) => [created, ...current]);
    await loadConversation(created.id);
  };

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">P</span>
          <div><strong>PaperPilot</strong><small>Research agent</small></div>
        </div>
        <button className="new-button" type="button" onClick={() => void handleNewConversation()}>
          + 新对话
        </button>
        <nav aria-label="对话列表">
          {conversations.map((conversation) => (
            <button
              className={conversation.id === activeId ? "conversation active" : "conversation"}
              key={conversation.id}
              onClick={() => void loadConversation(conversation.id)}
              type="button"
            >
              {conversation.title}
            </button>
          ))}
        </nav>
      </aside>

      <section className="chat-column">
        <header>
          <div><span className={isRunning ? "status running" : "status"} /> Single Research Agent</div>
          <span>{isRunning ? "执行中" : "Ready"}</span>
        </header>
        <div className="messages">
          {renderedMessages.length === 0 ? (
            <div className="hero">
              <span>ARXIV RESEARCH WORKSPACE</span>
              <h1>从问题，到可追溯的论文线索。</h1>
              <p>试试：帮我找 5 篇关于 RAG hallucination evaluation 的论文，并比较研究重点。</p>
            </div>
          ) : null}
          {renderedMessages.map((message) => (
            <article className={`bubble ${message.role}`} key={message.id}>
              <span>{message.role === "user" ? "YOU" : "PAPERPILOT"}</span>
              <p>{message.content}</p>
            </article>
          ))}
          {streamedAnswer ? (
            <article className="bubble assistant streaming">
              <span>PAPERPILOT · STREAMING</span><p>{streamedAnswer}</p>
            </article>
          ) : null}
          {error ? <div className="error">{error}</div> : null}
        </div>
        <form className="composer" onSubmit={(event) => void handleSubmit(event)}>
          <textarea
            aria-label="研究问题"
            onChange={(event) => setDraft(event.target.value)}
            placeholder="输入一个需要检索论文的研究问题…"
            rows={3}
            value={draft}
          />
          <button disabled={isRunning || activeId === null} type="submit">
            {isRunning ? "研究中…" : "发送"}
          </button>
        </form>
      </section>

      <aside className="trace-column">
        <MetricsPanel metrics={metrics} />
        <EventTimeline events={events} />
      </aside>
    </main>
  );
}

