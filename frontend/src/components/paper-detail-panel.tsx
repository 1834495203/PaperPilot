import type { IndexedPaperDetail, PaperTreeNode } from "@/lib/types";

interface PaperDetailPanelProps {
  detail: IndexedPaperDetail;
  onClose: () => void;
}

function TreeNode({ node, byId }: { node: PaperTreeNode; byId: Map<string, PaperTreeNode> }) {
  const children = node.children_ids
    .map((childId) => byId.get(childId))
    .filter((child): child is PaperTreeNode => child !== undefined);
  const pageLabel =
    node.page_start === null
      ? ""
      : node.page_end !== null && node.page_end !== node.page_start
        ? `P${node.page_start}–${node.page_end}`
        : `P${node.page_start}`;
  const summary = (
    <span className="tree-node-summary">
      <span className={`node-kind kind-${node.node_type}`}>{node.node_type}</span>
      <strong>{node.title}</strong>
      {pageLabel ? <small>{pageLabel}</small> : null}
    </span>
  );

  if (children.length === 0) {
    return <li className="tree-leaf">{summary}<p>{node.text_preview}</p></li>;
  }
  return (
    <li>
      <details open={node.level < 2}>
        <summary>{summary}</summary>
        {node.node_type === "section" ? <p>{node.text_preview}</p> : null}
        <ul>{children.map((child) => <TreeNode byId={byId} key={child.node_id} node={child} />)}</ul>
      </details>
    </li>
  );
}

export function PaperDetailPanel({ detail, onClose }: PaperDetailPanelProps) {
  const byId = new Map(detail.nodes.map((node) => [node.node_id, node]));
  const roots = detail.nodes.filter((node) => node.parent_id === null);

  return (
    <div className="paper-detail-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        aria-label="论文索引详情"
        aria-modal="true"
        className="paper-detail-panel"
        role="dialog"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div><span>LOCAL TREE INDEX</span><h2>{detail.paper.title}</h2></div>
          <button aria-label="关闭论文详情" onClick={onClose} type="button">×</button>
        </header>
        <div className="paper-stats">
          <span><strong>{detail.paper.page_count}</strong>页</span>
          <span><strong>{detail.paper.section_count}</strong>章节</span>
          <span><strong>{detail.paper.chunk_count}</strong>Chunks</span>
          <span><strong>{detail.paper.node_count}</strong>节点</span>
        </div>
        <p className="paper-id">{detail.paper.paper_id}</p>
        {detail.paper.authors.length > 0 ? (
          <p className="paper-id">作者：{detail.paper.authors.join(", ")}</p>
        ) : null}
        {detail.paper.abstract ? <p>{detail.paper.abstract}</p> : null}
        {detail.paper.keywords.length > 0 ? (
          <p className="paper-id">关键词：{detail.paper.keywords.join(" · ")}</p>
        ) : null}
        <div className="paper-tree">
          <ul>{roots.map((root) => <TreeNode byId={byId} key={root.node_id} node={root} />)}</ul>
        </div>
      </section>
    </div>
  );
}
