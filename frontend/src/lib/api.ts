import type {
  AgentEvent,
  AgentRun,
  Citation,
  CitationIssue,
  CitationVerification,
  Conversation,
  ConversationMetricsResponse,
  JsonValue,
  Message,
  IndexedPaper,
  IndexedPaperDetail,
} from "@/lib/types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

export class ApiError extends Error {
  public constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    throw new ApiError(await response.text(), response.status);
  }
  return (await response.json()) as T;
}

export function listConversations(): Promise<Conversation[]> {
  return requestJson<Conversation[]>("/conversations", { cache: "no-store" });
}

export function createConversation(title: string): Promise<Conversation> {
  return requestJson<Conversation>("/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/conversations/${conversationId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new ApiError(await response.text(), response.status);
  }
}

export function listMessages(
  conversationId: string,
  includeSuperseded = false,
): Promise<Message[]> {
  const suffix = includeSuperseded ? "?include_superseded=true" : "";
  return requestJson<Message[]>(
    `/conversations/${conversationId}/messages${suffix}`,
    { cache: "no-store" },
  );
}

export function listIndexedPapers(): Promise<IndexedPaper[]> {
  return requestJson<IndexedPaper[]>("/papers", { cache: "no-store" });
}

export function getIndexedPaperDetail(paperId: string): Promise<IndexedPaperDetail> {
  return requestJson<IndexedPaperDetail>(`/papers/${encodeURIComponent(paperId)}`, {
    cache: "no-store",
  });
}

export function assetUrl(paperId: string, filename: string): string {
  return `${API_BASE_URL}/papers/${encodeURIComponent(paperId)}/assets/${encodeURIComponent(filename)}`;
}

export function pdfUrl(paperId: string): string {
  return `${API_BASE_URL}/papers/${encodeURIComponent(paperId)}/pdf`;
}

function toCitation(value: JsonValue): Citation | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const record = value as Record<string, JsonValue>;
  if (typeof record.evidence_id !== "string" || typeof record.paper_id !== "string") return null;
  return {
    evidence_id: record.evidence_id,
    paper_id: record.paper_id,
    paper_title: typeof record.paper_title === "string" ? record.paper_title : "",
    page_start: typeof record.page_start === "number" ? record.page_start : null,
    page_end: typeof record.page_end === "number" ? record.page_end : null,
    excerpt: typeof record.excerpt === "string" ? record.excerpt : "",
    spans: [],
  };
}

export function readCitations(message: Message): Citation[] {
  const raw = message.metadata?.citations;
  if (!Array.isArray(raw)) return [];
  return raw.map(toCitation).filter((item): item is Citation => item !== null);
}

/**
 * Read the citation-support verdict of an answer.
 *
 * Older answers predate verification, so a missing record means "not checked"
 * rather than "clean".
 */
export function readCitationVerification(message: Message): CitationVerification | null {
  const raw = message.metadata?.citation_verification;
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
  const record = raw as Record<string, JsonValue>;
  const issues: CitationIssue[] = [];
  const rawIssues = record.issues;
  if (Array.isArray(rawIssues)) {
    for (const item of rawIssues) {
      if (typeof item !== "object" || item === null || Array.isArray(item)) continue;
      const entry = item as Record<string, JsonValue>;
      const evidenceId = entry.evidence_id;
      const kind = entry.kind;
      const detail = entry.detail;
      if (typeof evidenceId !== "string" || typeof detail !== "string") continue;
      if (kind !== "unknown_evidence_id" && kind !== "unsupported_claim") continue;
      issues.push({ evidence_id: evidenceId, kind, detail });
    }
  }
  const supported = Array.isArray(record.supported)
    ? record.supported.filter((item): item is string => typeof item === "string")
    : [];
  const verificationError = record.verification_error;
  return {
    checked: typeof record.checked === "number" ? record.checked : 0,
    supported,
    issues,
    verification_error: typeof verificationError === "string" ? verificationError : null,
    has_problems:
      issues.length > 0 ||
      (typeof verificationError === "string" && verificationError.length > 0),
  };
}

export function uploadPaper(file: File): Promise<IndexedPaper> {
  const body = new FormData();
  body.append("file", file);
  return requestJson<IndexedPaper>("/papers", { method: "POST", body });
}

export async function deleteIndexedPaper(paperId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE_URL}/papers/${encodeURIComponent(paperId)}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new ApiError(await response.text(), response.status);
  }
}

export function getConversationMetrics(
  conversationId: string,
): Promise<ConversationMetricsResponse> {
  return requestJson<ConversationMetricsResponse>(
    `/conversations/${conversationId}/metrics`,
    { cache: "no-store" },
  );
}

export function listConversationRuns(conversationId: string): Promise<AgentRun[]> {
  return requestJson<AgentRun[]>(`/conversations/${conversationId}/runs`, {
    cache: "no-store",
  });
}

export async function cancelRun(conversationId: string, runId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE_URL}/conversations/${conversationId}/runs/${runId}/cancel`,
    { method: "POST" },
  );
  if (!response.ok && response.status !== 409) {
    throw new ApiError(await response.text(), response.status);
  }
}

export function listConversationEvents(
  conversationId: string,
  limit = 500,
): Promise<AgentEvent[]> {
  return requestJson<AgentEvent[]>(
    `/conversations/${conversationId}/events?limit=${limit}`,
    { cache: "no-store" },
  );
}

async function streamEvents(
  path: string,
  init: RequestInit,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    ...(signal === undefined ? {} : { signal }),
  });
  if (!response.ok || response.body === null) {
    throw new ApiError(await response.text(), response.status);
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const dataLines = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart());
      if (dataLines.length > 0) {
        onEvent(JSON.parse(dataLines.join("\n")) as AgentEvent);
      }
    }
  }
}

export async function streamMessage(
  conversationId: string,
  content: string,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  await streamEvents(
    `/conversations/${conversationId}/messages/stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ content }),
    },
    onEvent,
    signal,
  );
}

/**
 * Re-attach to a run after a dropped connection, replaying events after the last
 * sequence the client saw. The run itself keeps executing on the server.
 */
export async function resumeRunStream(
  conversationId: string,
  runId: string,
  after: number,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  await streamEvents(
    `/conversations/${conversationId}/runs/${runId}/stream?after=${after}`,
    { method: "GET", headers: { Accept: "text/event-stream" } },
    onEvent,
    signal,
  );
}

export async function regenerateMessage(
  conversationId: string,
  messageId: string,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  await streamEvents(
    `/conversations/${conversationId}/messages/${messageId}/regenerate`,
    { method: "POST", headers: { Accept: "text/event-stream" } },
    onEvent,
    signal,
  );
}
