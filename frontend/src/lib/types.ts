export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type MessageRole = "user" | "assistant" | "tool" | "system";

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: MessageRole;
  content: string;
  sequence: number;
  created_at: string;
  metadata: Record<string, JsonValue>;
}

export type AgentEventType =
  | "run.started"
  | "stage.started"
  | "message.token"
  | "message.completed"
  | "tool.started"
  | "tool.completed"
  | "tool.failed"
  | "metrics.updated"
  | "run.completed"
  | "run.failed";

export interface AgentEvent {
  id: string;
  run_id: string;
  conversation_id: string;
  sequence: number;
  type: AgentEventType;
  timestamp: string;
  payload: Record<string, JsonValue>;
}

export interface RunMetrics {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  llmCalls: number;
  toolCalls: number;
  durationMs: number;
}

export interface PaperResult {
  [key: string]: JsonValue;
  arxiv_id: string;
  title: string;
  summary: string;
  authors: string[];
  published_at: string;
  updated_at: string;
  abstract_url: string;
  pdf_url: string | null;
}
