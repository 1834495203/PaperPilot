"use client";

import {
  ChangeEvent,
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { EventTimeline } from "@/components/event-timeline";
import { MetricsPanel } from "@/components/metrics-panel";
import { PaperDetailPanel } from "@/components/paper-detail-panel";
import { TurnTask } from "@/components/turn-task";
import {
  cancelRun,
  createConversation,
  deleteConversation,
  getIndexedPaperDetail,
  getConversationMetrics,
  listConversationEvents,
  listConversationRuns,
  listConversations,
  listMessages,
  listIndexedPapers,
  streamMessage,
  uploadPaper,
} from "@/lib/api";
import type {
  AgentEvent,
  AgentRun,
  Conversation,
  ConversationMetricsResponse,
  IndexedPaperDetail,
  JsonValue,
  IndexedPaper,
  Message,
  RunMetrics,
} from "@/lib/types";

const EMPTY_METRICS: RunMetrics = {
  inputTokens: 0,
  outputTokens: 0,
  totalTokens: 0,
  llmCalls: 0,
  toolCalls: 0,
  durationMs: 0,
  runCount: 0,
};

function fromConversationMetrics(metrics: ConversationMetricsResponse): RunMetrics {
  return {
    inputTokens: metrics.input_tokens,
    outputTokens: metrics.output_tokens,
    totalTokens: metrics.total_tokens,
    llmCalls: metrics.llm_calls,
    toolCalls: metrics.tool_calls,
    durationMs: metrics.total_duration_ms,
    runCount: metrics.run_count,
  };
}

function numberValue(payload: Record<string, JsonValue>, key: string): number {
  const value = payload[key];
  return typeof value === "number" ? value : 0;
}

export function ChatWorkspace() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [draft, setDraft] = useState("");
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [metrics, setMetrics] = useState<RunMetrics>(EMPTY_METRICS);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [papers, setPapers] = useState<IndexedPaper[]>([]);
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [paperDetail, setPaperDetail] = useState<IndexedPaperDetail | null>(null);
  const [loadingPaperId, setLoadingPaperId] = useState<string | null>(null);
  const metricsBaseline = useRef<RunMetrics>(EMPTY_METRICS);
  const activeRunId = useRef<string | null>(null);
  const streamController = useRef<AbortController | null>(null);

  useEffect(() => () => streamController.current?.abort(), []);

  const loadConversation = useCallback(async (conversationId: string) => {
    setActiveId(conversationId);
    const [storedMessages, storedMetrics, storedEvents, storedRuns] = await Promise.all([
      listMessages(conversationId),
      getConversationMetrics(conversationId),
      listConversationEvents(conversationId),
      listConversationRuns(conversationId),
    ]);
    const restoredMetrics = fromConversationMetrics(storedMetrics);
    setMessages(storedMessages);
    setEvents(storedEvents);
    setRuns(storedRuns);
    setStreamedAnswer("");
    setMetrics(restoredMetrics);
    metricsBaseline.current = restoredMetrics;
  }, []);

  useEffect(() => {
    const initialize = async () => {
      try {
        const [existing, indexedPapers] = await Promise.all([
          listConversations(),
          listIndexedPapers(),
        ]);
        setPapers(indexedPapers);
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

  const taskView = useMemo(() => {
    const messagesByRun = new Map<string, Message[]>();
    const legacy = renderedMessages.filter(
      (message) => typeof message.metadata.run_id !== "string",
    );
    for (const message of renderedMessages) {
      const runId = message.metadata.run_id;
      if (typeof runId !== "string") continue;
      messagesByRun.set(runId, [...(messagesByRun.get(runId) ?? []), message]);
    }
    const usedMessageIds = new Set<string>();
    let legacyIndex = 0;
    const groupedTasks = runs.map((run) => {
      let runMessages = messagesByRun.get(run.id) ?? [];
      if (runMessages.length === 0) {
        const fallback: Message[] = [];
        while (legacyIndex < legacy.length && fallback.length < 2) {
          const message = legacy[legacyIndex];
          legacyIndex += 1;
          if (message !== undefined) fallback.push(message);
          if (message?.role === "assistant") break;
        }
        runMessages = fallback;
      }
      runMessages.forEach((message) => usedMessageIds.add(message.id));
      return {
        run,
        userMessage: runMessages.find((message) => message.role === "user"),
        assistantMessage: runMessages.find((message) => message.role === "assistant"),
        events: events.filter((event) => event.run_id === run.id),
      };
    });
    return {
      tasks: groupedTasks,
      ungroupedMessages: renderedMessages.filter(
        (message) => !usedMessageIds.has(message.id),
      ),
    };
  }, [events, renderedMessages, runs]);

  const latestRunEvents = useMemo(() => {
    const runId = events.at(-1)?.run_id;
    return runId === undefined ? [] : events.filter((event) => event.run_id === runId);
  }, [events]);

  const handleEvent = useCallback((event: AgentEvent) => {
    setEvents((current) => [...current, event]);
    if (event.type === "run.started") {
      activeRunId.current = event.run_id;
      setMessages((current) => current.map((message) =>
        message.id.startsWith("optimistic-") && typeof message.metadata.run_id !== "string"
          ? { ...message, metadata: { ...message.metadata, run_id: event.run_id } }
          : message
      ));
      setRuns((current) => current.some((run) => run.id === event.run_id) ? current : [
        ...current,
        {
          id: event.run_id,
          conversation_id: event.conversation_id,
          status: "running",
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          tool_calls: 0,
          duration_ms: 0,
          error: null,
          started_at: event.timestamp,
          completed_at: null,
        },
      ]);
    }
    if (event.type === "message.token") {
      const text = event.payload.text;
      if (typeof text === "string") setStreamedAnswer((current) => current + text);
    }
    if (event.type === "message.completed" && event.payload.has_tool_calls === true) {
      setStreamedAnswer("");
    }
    if (event.type === "metrics.updated") {
      setRuns((current) => current.map((run) => run.id === event.run_id ? {
        ...run,
        input_tokens: numberValue(event.payload, "input_tokens"),
        output_tokens: numberValue(event.payload, "output_tokens"),
        total_tokens: numberValue(event.payload, "total_tokens"),
        llm_calls: numberValue(event.payload, "llm_calls"),
        tool_calls: numberValue(event.payload, "tool_calls"),
      } : run));
      const baseline = metricsBaseline.current;
      setMetrics({
        inputTokens: baseline.inputTokens + numberValue(event.payload, "input_tokens"),
        outputTokens: baseline.outputTokens + numberValue(event.payload, "output_tokens"),
        totalTokens: baseline.totalTokens + numberValue(event.payload, "total_tokens"),
        llmCalls: baseline.llmCalls + numberValue(event.payload, "llm_calls"),
        toolCalls: baseline.toolCalls + numberValue(event.payload, "tool_calls"),
        durationMs: baseline.durationMs,
        runCount: baseline.runCount,
      });
    }
    if (
      event.type === "run.completed" ||
      event.type === "run.failed" ||
      event.type === "run.cancelled"
    ) {
      activeRunId.current = null;
      setRuns((current) => current.map((run) => run.id === event.run_id ? {
        ...run,
        status: event.type === "run.completed"
          ? "completed"
          : event.type === "run.cancelled"
            ? "cancelled"
            : "failed",
        input_tokens: numberValue(event.payload, "input_tokens"),
        output_tokens: numberValue(event.payload, "output_tokens"),
        total_tokens: numberValue(event.payload, "total_tokens"),
        llm_calls: numberValue(event.payload, "llm_calls"),
        tool_calls: numberValue(event.payload, "tool_calls"),
        duration_ms: numberValue(event.payload, "duration_ms"),
        error: typeof event.payload.error === "string" ? event.payload.error : null,
        completed_at: event.timestamp,
      } : run));
      const cumulativeMetrics: RunMetrics = {
        inputTokens: numberValue(event.payload, "conversation_input_tokens"),
        outputTokens: numberValue(event.payload, "conversation_output_tokens"),
        totalTokens: numberValue(event.payload, "conversation_total_tokens"),
        llmCalls: numberValue(event.payload, "conversation_llm_calls"),
        toolCalls: numberValue(event.payload, "conversation_tool_calls"),
        durationMs: numberValue(event.payload, "conversation_total_duration_ms"),
        runCount: numberValue(event.payload, "conversation_run_count"),
      };
      metricsBaseline.current = cumulativeMetrics;
      setMetrics(cumulativeMetrics);
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
    setStreamedAnswer("");
    metricsBaseline.current = metrics;
    setIsRunning(true);
    activeRunId.current = null;
    const controller = new AbortController();
    streamController.current = controller;
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
      await streamMessage(
        activeId,
        content,
        selectedPaperIds,
        handleEvent,
        controller.signal,
      );
      const [storedMessages, storedRuns] = await Promise.all([
        listMessages(activeId),
        listConversationRuns(activeId),
      ]);
      setMessages(storedMessages);
      setRuns(storedRuns);
      setStreamedAnswer("");
      setConversations(await listConversations());
    } catch (caught) {
      if (!(caught instanceof DOMException && caught.name === "AbortError")) {
        setError(caught instanceof Error ? caught.message : "请求失败");
      }
    } finally {
      if (streamController.current === controller) streamController.current = null;
      activeRunId.current = null;
      setIsRunning(false);
    }
  };

  const handleStop = async () => {
    const controller = streamController.current;
    const runId = activeRunId.current;
    if (activeId !== null && runId !== null) {
      try {
        await cancelRun(activeId, runId);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "停止任务失败");
      }
    }
    controller?.abort();
  };

  const handleNewConversation = async () => {
    const created = await createConversation("New research");
    setConversations((current) => [created, ...current]);
    await loadConversation(created.id);
  };

  const handleDeleteConversation = async (conversation: Conversation) => {
    if (isRunning || !window.confirm(`删除会话“${conversation.title}”？此操作不可撤销。`)) {
      return;
    }
    setError(null);
    try {
      await deleteConversation(conversation.id);
      const remaining = conversations.filter((item) => item.id !== conversation.id);
      if (conversation.id !== activeId) {
        setConversations(remaining);
      } else if (remaining[0] !== undefined) {
        setConversations(remaining);
        await loadConversation(remaining[0].id);
      } else {
        const created = await createConversation("New research");
        setConversations([created]);
        await loadConversation(created.id);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "删除会话失败");
    }
  };

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file === undefined || isUploading) return;
    setError(null);
    setIsUploading(true);
    try {
      const paper = await uploadPaper(file);
      setPapers((current) => [paper, ...current.filter((item) => item.paper_id !== paper.paper_id)]);
      setSelectedPaperIds((current) =>
        current.includes(paper.paper_id) ? current : [...current, paper.paper_id],
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "论文上传失败");
    } finally {
      setIsUploading(false);
    }
  };

  const togglePaper = (paperId: string) => {
    setSelectedPaperIds((current) =>
      current.includes(paperId)
        ? current.filter((item) => item !== paperId)
        : [...current, paperId],
    );
  };

  const viewPaper = async (paperId: string) => {
    setError(null);
    setLoadingPaperId(paperId);
    try {
      setPaperDetail(await getIndexedPaperDetail(paperId));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "读取论文索引失败");
    } finally {
      setLoadingPaperId(null);
    }
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
        <section className="paper-library">
          <div className="paper-library-heading">
            <strong>本地论文库</strong><span>{papers.length}</span>
          </div>
          <label className={isUploading ? "upload-button disabled" : "upload-button"}>
            {isUploading ? "解析与建库中…" : "+ 上传 PDF"}
            <input
              accept="application/pdf,.pdf"
              disabled={isUploading || isRunning}
              onChange={(event) => void handleUpload(event)}
              type="file"
            />
          </label>
          <div className="indexed-papers">
            {papers.map((paper) => (
              <div className="indexed-paper" key={paper.paper_id}>
                <input
                  aria-label={`选择 ${paper.title} 用于问答`}
                  checked={selectedPaperIds.includes(paper.paper_id)}
                  onChange={() => togglePaper(paper.paper_id)}
                  type="checkbox"
                />
                <button onClick={() => void viewPaper(paper.paper_id)} type="button">
                  <strong>{paper.title}</strong>
                  <small>
                    {loadingPaperId === paper.paper_id
                      ? "读取索引中…"
                      : paper.authors.length > 0
                        ? `${paper.authors.slice(0, 2).join(", ")} · ${paper.page_count} 页`
                        : `${paper.page_count} 页 · ${paper.chunk_count} chunks`}
                  </small>
                </button>
              </div>
            ))}
          </div>
        </section>
        <nav aria-label="对话列表">
          {conversations.map((conversation) => (
            <div className="conversation-row" key={conversation.id}>
              <button
                className={conversation.id === activeId ? "conversation active" : "conversation"}
                onClick={() => void loadConversation(conversation.id)}
                type="button"
              >
                {conversation.title}
              </button>
              <button
                aria-label={`删除会话 ${conversation.title}`}
                className="delete-conversation"
                disabled={isRunning}
                onClick={() => void handleDeleteConversation(conversation)}
                title="删除会话"
                type="button"
              >×</button>
            </div>
          ))}
        </nav>
      </aside>

      <section className="chat-column">
        <header>
          <div><span className={isRunning ? "status running" : "status"} /> Supervisor Research Team</div>
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
          {taskView.ungroupedMessages.map((message) => (
            <article className={`bubble ${message.role}`} key={message.id}>
              <span>{message.role === "user" ? "YOU" : "PAPERPILOT"}</span>
              <p>{message.content}</p>
            </article>
          ))}
          {taskView.tasks.map((task) => (
            <TurnTask
              assistantMessage={task.assistantMessage}
              events={task.events}
              key={task.run.id}
              run={task.run}
              userMessage={task.userMessage}
            />
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
          {isRunning ? (
            <button className="stop-button" onClick={() => void handleStop()} type="button">
              停止
            </button>
          ) : (
            <button disabled={activeId === null} type="submit">发送</button>
          )}
        </form>
      </section>

      <aside className="trace-column">
        <div className="section-heading"><h2>会话累计</h2></div>
        <MetricsPanel metrics={metrics} />
        <EventTimeline events={latestRunEvents} />
      </aside>
      {paperDetail !== null ? (
        <PaperDetailPanel detail={paperDetail} onClose={() => setPaperDetail(null)} />
      ) : null}
    </main>
  );
}
