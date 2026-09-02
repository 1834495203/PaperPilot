import type {
  AgentEvent,
  AgentRun,
  Conversation,
  ConversationMetricsResponse,
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

export function listMessages(conversationId: string): Promise<Message[]> {
  return requestJson<Message[]>(`/conversations/${conversationId}/messages`, {
    cache: "no-store",
  });
}

export function listIndexedPapers(): Promise<IndexedPaper[]> {
  return requestJson<IndexedPaper[]>("/papers", { cache: "no-store" });
}

export function getIndexedPaperDetail(paperId: string): Promise<IndexedPaperDetail> {
  return requestJson<IndexedPaperDetail>(`/papers/${encodeURIComponent(paperId)}`, {
    cache: "no-store",
  });
}

export function uploadPaper(file: File): Promise<IndexedPaper> {
  const body = new FormData();
  body.append("file", file);
  return requestJson<IndexedPaper>("/papers", { method: "POST", body });
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

export async function streamMessage(
  conversationId: string,
  content: string,
  paperIds: string[],
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(
    `${API_BASE_URL}/conversations/${conversationId}/messages/stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ content, paper_ids: paperIds }),
      ...(signal === undefined ? {} : { signal }),
    },
  );
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
