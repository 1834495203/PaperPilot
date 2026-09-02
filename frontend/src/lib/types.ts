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
  | "decision.recorded"
  | "message.token"
  | "message.completed"
  | "tool.started"
  | "tool.completed"
  | "tool.failed"
  | "metrics.updated"
  | "run.completed"
  | "run.failed"
  | "run.cancelled";

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
  runCount: number;
}

export interface ConversationMetricsResponse {
  conversation_id: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  llm_calls: number;
  tool_calls: number;
  total_duration_ms: number;
  run_count: number;
  updated_at: string | null;
}

export interface AgentRun {
  id: string;
  conversation_id: string;
  status: "running" | "completed" | "failed" | "cancelled";
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  llm_calls: number;
  tool_calls: number;
  duration_ms: number;
  error: string | null;
  started_at: string;
  completed_at: string | null;
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

export interface IndexedPaper {
  paper_id: string;
  title: string;
  authors: string[];
  abstract: string | null;
  keywords: string[];
  doi: string | null;
  arxiv_id: string | null;
  original_filename: string;
  page_count: number;
  section_count: number;
  node_count: number;
  chunk_count: number;
  created_at: string;
}

export type PaperTreeNodeType = "root" | "section" | "chunk";

export interface PaperTreeNode {
  node_id: string;
  node_type: PaperTreeNodeType;
  title: string;
  parent_id: string | null;
  children_ids: string[];
  level: number;
  section_path: string[];
  page_start: number | null;
  page_end: number | null;
  text_preview: string;
}

export interface IndexedPaperDetail {
  paper: IndexedPaper;
  nodes: PaperTreeNode[];
}
