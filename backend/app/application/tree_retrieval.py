import math
import re
from dataclasses import dataclass

from app.domain.ports import TextEmbeddingGateway, TreeVectorStore
from app.domain.rag import (
    IndexedTreeNode,
    RetrievalHit,
    RetrievalMode,
    RetrievalSource,
    TreeIndexNode,
    TreeNodeType,
    TreeRetrievalReport,
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    node: TreeIndexNode
    vector_score: float
    source: RetrievalSource
    expanded_from: str | None = None


class TreeRagRetriever:
    """Retrieve content chunks with TreeRAG leaf-to-root-to-leaves expansion."""

    def __init__(
        self,
        *,
        embedder: TextEmbeddingGateway,
        vector_store: TreeVectorStore,
        initial_top_k: int = 12,
        final_top_k: int = 8,
        max_expanded_per_hit: int = 8,
        max_candidates: int = 40,
        max_chunks_per_paper: int = 8,
    ) -> None:
        limits = {
            "initial_top_k": initial_top_k,
            "final_top_k": final_top_k,
            "max_expanded_per_hit": max_expanded_per_hit,
            "max_candidates": max_candidates,
            "max_chunks_per_paper": max_chunks_per_paper,
        }
        if any(value < 1 for value in limits.values()):
            raise ValueError("All retrieval limits must be positive")
        self._embedder = embedder
        self._vector_store = vector_store
        self._initial_top_k = initial_top_k
        self._final_top_k = final_top_k
        self._max_expanded_per_hit = max_expanded_per_hit
        self._max_candidates = max_candidates
        self._max_chunks_per_paper = max_chunks_per_paper

    async def retrieve(
        self,
        query: str,
        *,
        paper_ids: list[str],
        mode: RetrievalMode,
    ) -> TreeRetrievalReport:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        if not paper_ids:
            raise ValueError("At least one paper_id is required")
        query_embedding = await self._embedder.embed_query(query)
        initial = await self._vector_store.similarity_search(
            query_embedding,
            paper_ids=paper_ids,
            top_k=self._initial_top_k,
            chunks_only=not mode.expands_tree,
        )
        candidates: dict[str, _Candidate] = {}
        for match in initial:
            if match.node.node_type is TreeNodeType.CHUNK:
                candidates[match.node.node_id] = _Candidate(
                    node=match.node,
                    vector_score=match.vector_score,
                    source=RetrievalSource.VECTOR,
                )

        if mode.expands_tree and initial:
            indexed_nodes = await self._vector_store.load_paper_nodes(paper_ids)
            self._expand_candidates(
                candidates=candidates,
                initial_nodes=[match.node for match in initial],
                indexed_nodes=indexed_nodes,
                query_embedding=query_embedding,
            )

        expanded_count = sum(
            candidate.source is RetrievalSource.TREE_EXPANSION
            for candidate in candidates.values()
        )
        ranked = sorted(
            (
                (candidate, self._ranking_score(query, candidate))
                for candidate in candidates.values()
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: self._max_candidates]
        diversified = self._enforce_paper_cap(ranked)[: self._final_top_k]
        hits = [
            RetrievalHit(
                rank=rank,
                node_id=candidate.node.node_id,
                paper_id=candidate.node.paper_id,
                section_path=candidate.node.section_path,
                page_start=candidate.node.page_start,
                page_end=candidate.node.page_end,
                text=candidate.node.text,
                vector_score=candidate.vector_score,
                ranking_score=ranking_score,
                source=candidate.source,
                expanded_from=candidate.expanded_from,
            )
            for rank, (candidate, ranking_score) in enumerate(diversified, start=1)
        ]
        return TreeRetrievalReport(
            query=query,
            mode=mode,
            paper_ids=paper_ids,
            initial_hit_count=len(initial),
            expanded_candidate_count=expanded_count,
            hits=hits,
        )

    def _expand_candidates(
        self,
        *,
        candidates: dict[str, _Candidate],
        initial_nodes: list[TreeIndexNode],
        indexed_nodes: list[IndexedTreeNode],
        query_embedding: list[float],
    ) -> None:
        by_id = {item.node.node_id: item for item in indexed_nodes}
        for initial_node in initial_nodes:
            anchor_id = (
                initial_node.parent_id
                if initial_node.node_type is TreeNodeType.CHUNK
                else initial_node.node_id
            )
            if anchor_id is None or anchor_id not in by_id:
                continue
            descendants = [
                item
                for item in self._descendants(anchor_id, by_id)
                if item.node.node_type is TreeNodeType.CHUNK
            ]
            scored = sorted(
                (
                    (item, self._cosine_similarity(query_embedding, item.embedding))
                    for item in descendants
                ),
                key=lambda item: item[1],
                reverse=True,
            )[: self._max_expanded_per_hit]
            for indexed_node, score in scored:
                if indexed_node.node.node_id in candidates:
                    continue
                candidates[indexed_node.node.node_id] = _Candidate(
                    node=indexed_node.node,
                    vector_score=score,
                    source=RetrievalSource.TREE_EXPANSION,
                    expanded_from=initial_node.node_id,
                )

    @staticmethod
    def _descendants(
        anchor_id: str,
        by_id: dict[str, IndexedTreeNode],
    ) -> list[IndexedTreeNode]:
        descendants: list[IndexedTreeNode] = []
        visited: set[str] = set()
        pending = list(by_id[anchor_id].node.children_ids)
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            indexed = by_id.get(node_id)
            if indexed is None:
                continue
            descendants.append(indexed)
            pending.extend(indexed.node.children_ids)
        return descendants

    def _enforce_paper_cap(
        self,
        candidates: list[tuple[_Candidate, float]],
    ) -> list[tuple[_Candidate, float]]:
        counts: dict[str, int] = {}
        selected: list[tuple[_Candidate, float]] = []
        for candidate, ranking_score in candidates:
            count = counts.get(candidate.node.paper_id, 0)
            if count >= self._max_chunks_per_paper:
                continue
            selected.append((candidate, ranking_score))
            counts[candidate.node.paper_id] = count + 1
        return selected

    @classmethod
    def _ranking_score(cls, query: str, candidate: _Candidate) -> float:
        """Lightweight hybrid rerank before a dedicated cross-encoder is configured."""
        query_terms = cls._terms(query)
        searchable = " ".join([*candidate.node.section_path, candidate.node.text])
        candidate_terms = cls._terms(searchable)
        lexical_score = (
            len(query_terms & candidate_terms) / len(query_terms)
            if query_terms
            else 0.0
        )
        return 0.85 * candidate.vector_score + 0.15 * lexical_score

    @staticmethod
    def _terms(text: str) -> set[str]:
        latin_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+", text)
            if len(token) > 1
        }
        chinese_characters = set(re.findall(r"[\u4e00-\u9fff]", text))
        return latin_terms | chinese_characters

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Query and stored embeddings use different dimensions")
        left_magnitude = math.sqrt(sum(value * value for value in left))
        right_magnitude = math.sqrt(sum(value * value for value in right))
        if left_magnitude == 0 or right_magnitude == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (
            left_magnitude * right_magnitude
        )
