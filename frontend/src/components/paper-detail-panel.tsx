"use client";

import { useMemo, useState } from "react";

import { assetUrl, pdfUrl } from "@/lib/api";
import type { IndexedPaperDetail, PaperTreeNode } from "@/lib/types";

interface PaperDetailPanelProps {
  deleteDisabled: boolean;
  detail: IndexedPaperDetail;
  focusPage?: number | null;
  isDeleting: boolean;
  onClose: () => void;
  onDelete: () => void;
}

type NodeFilter = "all" | "section" | "chunk";
type PanelTab = "index" | "about" | "pdf";

function pageLabel(node: PaperTreeNode): string {
  if (node.page_start === null) return "";
  return node.page_end !== null && node.page_end !== node.page_start
    ? `P${node.page_start}–${node.page_end}`
    : `P${node.page_start}`;
}

function NodeLabel({ node, selected }: { node: PaperTreeNode; selected: boolean }) {
  const pages = pageLabel(node);
  return (
    <span className={selected ? "tree-node-summary selected" : "tree-node-summary"}>
      <span className={`node-kind kind-${node.node_type}`}>{node.node_type}</span>
      <strong>{node.title}</strong>
      {pages ? <small>{pages}</small> : null}
    </span>
  );
}

function TreeNode({
  node,
  byId,
  selectedId,
  onSelect,
}: {
  node: PaperTreeNode;
  byId: Map<string, PaperTreeNode>;
  selectedId: string;
  onSelect: (node: PaperTreeNode) => void;
}) {
  const children = node.children_ids
    .map((childId) => byId.get(childId))
    .filter((child): child is PaperTreeNode => child !== undefined);
  const label = <NodeLabel node={node} selected={selectedId === node.node_id} />;

  if (children.length === 0) {
    return (
      <li className="tree-leaf">
        <button onClick={() => onSelect(node)} type="button">{label}</button>
      </li>
    );
  }
  return (
    <li>
      <details open={node.level < 2}>
        <summary onClick={() => onSelect(node)}>{label}</summary>
        <ul>
          {children.map((child) => (
            <TreeNode
              byId={byId}
              key={child.node_id}
              node={child}
              onSelect={onSelect}
              selectedId={selectedId}
            />
          ))}
        </ul>
      </details>
    </li>
  );
}

function matchesNode(node: PaperTreeNode, filter: NodeFilter, query: string): boolean {
  if (filter !== "all" && node.node_type !== filter) return false;
  if (!query) return true;
  const haystack = [node.title, node.text_preview, ...node.section_path]
    .join(" ")
    .toLocaleLowerCase();
  return haystack.includes(query.toLocaleLowerCase());
}

function TableView({ rows }: { rows: string[][] }) {
  if (rows.length === 0) return null;
  const width = Math.max(...rows.map((row) => row.length));
  const header = rows[0] ?? [];
  const body = rows.slice(1);
  return (
    <div className="table-scroll">
      <table className="structured-table">
        <thead>
          <tr>
            {Array.from({ length: width }, (_, index) => (
              <th key={index}>{header[index] ?? ""}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {Array.from({ length: width }, (_, index) => (
                <td key={index}>{row[index] ?? ""}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function NodeContent({ node, paperId }: { node: PaperTreeNode; paperId: string }) {
  const figureSrc = node.figure_asset ?? node.raw_asset_ref ?? null;
  if (node.block_types.includes("figure") && figureSrc) {
    return (
      <figure className="figure-block">
        <img
          alt={node.figure_caption ?? node.title}
          loading="lazy"
          src={assetUrl(paperId, figureSrc)}
        />
        <figcaption>{node.figure_caption ?? node.text}</figcaption>
      </figure>
    );
  }
  if (node.block_types.includes("table") && node.table_rows && node.table_rows.length > 0) {
    return <TableView rows={node.table_rows} />;
  }
  return <div className="node-content">{node.text || node.text_preview}</div>;
}

export function PaperDetailPanel({
  deleteDisabled,
  detail,
  focusPage,
  isDeleting,
  onClose,
  onDelete,
}: PaperDetailPanelProps) {
  const initialNode =
    detail.nodes.find((node) => node.node_type === "chunk") ?? detail.nodes[0] ?? null;
  const [tab, setTab] = useState<PanelTab>(focusPage != null ? "pdf" : "index");
  const [filter, setFilter] = useState<NodeFilter>("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState(initialNode?.node_id ?? "");
  const byId = useMemo(
    () => new Map(detail.nodes.map((node) => [node.node_id, node])),
    [detail.nodes],
  );
  const roots = useMemo(
    () => detail.nodes.filter((node) => node.parent_id === null),
    [detail.nodes],
  );
  const filteredNodes = useMemo(
    () => detail.nodes.filter((node) => matchesNode(node, filter, query.trim())),
    [detail.nodes, filter, query],
  );
  const selectedNode = byId.get(selectedId) ?? initialNode;
  const showHierarchy = filter === "all" && query.trim() === "";

  const selectFilter = (nextFilter: NodeFilter) => {
    setFilter(nextFilter);
    const nextNode = detail.nodes.find((node) => matchesNode(node, nextFilter, query.trim()));
    if (nextNode !== undefined) setSelectedId(nextNode.node_id);
  };

  return (
    <div className="paper-detail-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        aria-label="论文索引详情"
        aria-modal="true"
        className="paper-detail-panel"
        role="dialog"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="paper-detail-header">
          <div className="paper-detail-title">
            <span>LOCAL TREE INDEX</span>
            <h2>{detail.paper.title}</h2>
            <small>{detail.paper.page_count} 页 · {detail.paper.chunk_count} Chunks</small>
          </div>
          <div className="paper-detail-actions">
            <div className="paper-tabs" role="tablist" aria-label="论文详情视图">
              <button aria-selected={tab === "index"} className={tab === "index" ? "active" : ""} onClick={() => setTab("index")} role="tab" type="button">索引与 Chunk</button>
              <button aria-selected={tab === "pdf"} className={tab === "pdf" ? "active" : ""} onClick={() => setTab("pdf")} role="tab" type="button">原文 PDF</button>
              <button aria-selected={tab === "about"} className={tab === "about" ? "active" : ""} onClick={() => setTab("about")} role="tab" type="button">论文信息</button>
            </div>
            <button
              className="paper-delete-button"
              disabled={deleteDisabled}
              onClick={onDelete}
              type="button"
            >{isDeleting ? "删除中…" : "删除论文"}</button>
            <button className="paper-close" aria-label="关闭论文详情" onClick={onClose} type="button">×</button>
          </div>
        </header>

        {tab === "index" ? (
          <div className="paper-index-workbench">
            <aside className="tree-browser">
              <div className="tree-browser-toolbar">
                <div className="node-filters" aria-label="节点类型筛选">
                  {(["all", "section", "chunk"] as NodeFilter[]).map((value) => (
                    <button className={filter === value ? "active" : ""} key={value} onClick={() => selectFilter(value)} type="button">
                      {value === "all" ? "树结构" : value === "section" ? "章节" : "Chunks"}
                    </button>
                  ))}
                </div>
                <input aria-label="搜索索引节点" onChange={(event) => setQuery(event.target.value)} placeholder="搜索标题或 Chunk 内容" type="search" value={query} />
                <span>{filteredNodes.length} / {detail.nodes.length} 节点</span>
              </div>
              <div className="paper-tree hidden-scrollbar">
                {showHierarchy ? (
                  <ul>
                    {roots.map((root) => (
                      <TreeNode byId={byId} key={root.node_id} node={root} onSelect={(node) => setSelectedId(node.node_id)} selectedId={selectedId} />
                    ))}
                  </ul>
                ) : (
                  <div className="flat-node-list">
                    {filteredNodes.map((node) => (
                      <button className={node.node_id === selectedId ? "active" : ""} key={node.node_id} onClick={() => setSelectedId(node.node_id)} type="button">
                        <NodeLabel node={node} selected={node.node_id === selectedId} />
                        <span>{node.section_path.join(" / ") || detail.paper.title}</span>
                      </button>
                    ))}
                    {filteredNodes.length === 0 ? <p className="empty">没有匹配的节点。</p> : null}
                  </div>
                )}
              </div>
            </aside>

            <article className="node-inspector hidden-scrollbar">
              {selectedNode !== null ? (
                <>
                  <header>
                    <div><NodeLabel node={selectedNode} selected={false} /><h3>{selectedNode.title}</h3></div>
                    <span>{pageLabel(selectedNode)}</span>
                  </header>
                  <p className="node-path">{selectedNode.section_path.join(" / ") || detail.paper.title}</p>
                  <div className="node-metadata">
                    {selectedNode.semantic_role ? <span>{selectedNode.semantic_role}</span> : null}
                    {selectedNode.block_types.map((type) => <span key={type}>{type}</span>)}
                    {selectedNode.object_labels.map((label) => <span key={label}>#{label}</span>)}
                  </div>
                  <NodeContent node={selectedNode} paperId={detail.paper.paper_id} />
                </>
              ) : <p className="empty">该论文没有可显示的索引节点。</p>}
            </article>
          </div>
        ) : tab === "pdf" ? (
          <div className="paper-pdf-viewer hidden-scrollbar">
            <iframe
              className="paper-pdf-frame"
              src={pdfUrl(detail.paper.paper_id) + (focusPage != null ? `#page=${focusPage}` : "")}
              title={`${detail.paper.title} PDF`}
            />
          </div>
        ) : (
          <div className="paper-overview hidden-scrollbar">
            <div className="paper-stats">
              <span><strong>{detail.paper.page_count}</strong>页</span>
              <span><strong>{detail.paper.section_count}</strong>章节</span>
              <span><strong>{detail.paper.chunk_count}</strong>Chunks</span>
              <span><strong>{detail.paper.node_count}</strong>节点</span>
            </div>
            <section><span>作者</span><p>{detail.paper.authors.join(", ") || "未提取到作者信息"}</p></section>
            {detail.paper.abstract ? <section><span>摘要</span><p>{detail.paper.abstract}</p></section> : null}
            {detail.paper.keywords.length > 0 ? <section><span>关键词</span><p>{detail.paper.keywords.join(" · ")}</p></section> : null}
            <section><span>Paper ID</span><p className="paper-id">{detail.paper.paper_id}</p></section>
          </div>
        )}
      </section>
    </div>
  );
}
