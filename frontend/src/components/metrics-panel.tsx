import type { RunMetrics } from "@/lib/types";

interface MetricsPanelProps {
  metrics: RunMetrics;
}

const EMPTY_VALUE = "—";

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  const items: ReadonlyArray<readonly [string, string]> = [
    ["总耗时", metrics.durationMs > 0 ? `${(metrics.durationMs / 1000).toFixed(2)}s` : EMPTY_VALUE],
    ["Token", metrics.totalTokens > 0 ? metrics.totalTokens.toLocaleString() : EMPTY_VALUE],
    ["输入 / 输出", `${metrics.inputTokens} / ${metrics.outputTokens}`],
    ["LLM 调用", metrics.llmCalls.toString()],
    ["工具调用", metrics.toolCalls.toString()],
  ];

  return (
    <section className="metrics" aria-label="运行指标">
      {items.map(([label, value]) => (
        <div className="metric" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </section>
  );
}

