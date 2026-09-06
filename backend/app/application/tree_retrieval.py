import asyncio
import math
import re
import unicodedata
from dataclasses import dataclass, replace

from app.domain.ports import TextEmbeddingGateway, TextRerankerGateway, TreeVectorStore
from app.domain.rag import (
    IndexedTreeNode,
    RetrievalHit,
    RetrievalMode,
    RetrievalSource,
    TreeIndexNode,
    TreeNodeType,
    TreeRetrievalReport,
    TreeVectorMatch,
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    node: TreeIndexNode
    vector_score: float
    source: RetrievalSource
    expanded_from: str | None = None
    paper_score: float = 0.0
    section_score: float = 0.0
    paper_title: str | None = None
    embedding: list[float] | None = None
    rerank_score: float | None = None


class TreeRagRetriever:
    """Retrieve globally with paper -> section -> chunk routing and tree expansion."""

    def __init__(
        self,
        *,
        embedder: TextEmbeddingGateway,
        vector_store: TreeVectorStore,
        reranker: TextRerankerGateway | None = None,
        initial_top_k: int = 12,
        final_top_k: int = 8,
        max_expanded_per_hit: int = 8,
        max_candidates: int = 40,
        max_chunks_per_paper: int = 3,
        paper_top_k: int = 5,
        sections_per_paper: int = 3,
        global_fallback_top_k: int = 6,
        min_ranking_score: float = 0.20,
        score_window: float = 0.18,
        mmr_top_k: int = 30,
        mmr_lambda: float = 0.70,
    ) -> None:
        limits = {
            "initial_top_k": initial_top_k,
            "final_top_k": final_top_k,
            "max_expanded_per_hit": max_expanded_per_hit,
            "max_candidates": max_candidates,
            "max_chunks_per_paper": max_chunks_per_paper,
            "paper_top_k": paper_top_k,
            "sections_per_paper": sections_per_paper,
            "global_fallback_top_k": global_fallback_top_k,
            "mmr_top_k": mmr_top_k,
        }
        if any(value < 1 for value in limits.values()):
            raise ValueError("All retrieval limits must be positive")
        if not -1.0 <= min_ranking_score <= 1.0:
            raise ValueError("min_ranking_score must be between -1 and 1")
        if score_window < 0:
            raise ValueError("score_window cannot be negative")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._initial_top_k = initial_top_k
        self._final_top_k = final_top_k
        self._max_expanded_per_hit = max_expanded_per_hit
        self._max_candidates = max_candidates
        self._max_chunks_per_paper = max_chunks_per_paper
        self._paper_top_k = paper_top_k
        self._sections_per_paper = sections_per_paper
        self._global_fallback_top_k = global_fallback_top_k
        self._min_ranking_score = min_ranking_score
        self._score_window = score_window
        self._mmr_top_k = mmr_top_k
        self._mmr_lambda = mmr_lambda

    async def retrieve(
        self,
        query: str,
        *,
        paper_ids: list[str] | None = None,
        mode: RetrievalMode,
    ) -> TreeRetrievalReport:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        requested_paper_ids = list(dict.fromkeys(paper_ids)) if paper_ids else []
        query_embedding = await self._embedder.embed_query(query)
        root_matches = await self._vector_store.similarity_search(
            query_embedding,
            paper_ids=requested_paper_ids or None,
            top_k=max(self._paper_top_k, len(requested_paper_ids)),
            chunks_only=False,
            node_types=[TreeNodeType.ROOT],
        )
        routed_paper_ids = (
            requested_paper_ids
            if requested_paper_ids
            else list(
                dict.fromkeys(match.node.paper_id for match in root_matches)
            )[: self._paper_top_k]
        )

        section_matches = []
        selected_sections: list[TreeIndexNode] = []
        if routed_paper_ids:
            per_paper_matches = await asyncio.gather(
                *(
                    self._vector_store.similarity_search(
                        query_embedding,
                        paper_ids=[paper_id],
                        top_k=self._sections_per_paper,
                        chunks_only=False,
                        node_types=[TreeNodeType.SECTION],
                    )
                    for paper_id in routed_paper_ids
                )
            )
            section_matches = [
                match for paper_matches in per_paper_matches for match in paper_matches
            ]
            selected_sections = self._select_sections(section_matches)

        hierarchical_matches = []
        if selected_sections:
            hierarchical_matches = await self._vector_store.similarity_search(
                query_embedding,
                paper_ids=routed_paper_ids,
                top_k=self._initial_top_k,
                chunks_only=True,
                parent_ids=[node.node_id for node in selected_sections],
            )
        fallback_matches = await self._vector_store.similarity_search(
            query_embedding,
            paper_ids=requested_paper_ids or None,
            top_k=self._global_fallback_top_k,
            chunks_only=True,
        )

        candidates: dict[str, _Candidate] = {}
        for match in [*hierarchical_matches, *fallback_matches]:
            if match.node.node_type is TreeNodeType.CHUNK:
                existing = candidates.get(match.node.node_id)
                if existing is not None and existing.vector_score >= match.vector_score:
                    continue
                candidates[match.node.node_id] = _Candidate(
                    node=match.node,
                    vector_score=match.vector_score,
                    source=RetrievalSource.VECTOR,
                )

        candidate_paper_ids = list(
            dict.fromkeys(
                [
                    *routed_paper_ids,
                    *(candidate.node.paper_id for candidate in candidates.values()),
                ]
            )
        )
        indexed_nodes = (
            await self._vector_store.load_paper_nodes(candidate_paper_ids)
            if candidate_paper_ids
            else []
        )
        if mode.expands_tree and indexed_nodes:
            self._expand_candidates(
                candidates=candidates,
                initial_nodes=selected_sections,
                indexed_nodes=indexed_nodes,
                query_embedding=query_embedding,
            )
        self._add_hierarchy_scores(
            candidates=candidates,
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
        ranked = self._filter_weak_candidates(ranked)
        deduplicated = self._deduplicate_normalized_text(ranked)
        capped = self._enforce_paper_cap(deduplicated)
        mmr_selected = self._mmr_select(capped)[: self._mmr_top_k]
        reranked, reranker_error = await self._rerank_candidates(query, mmr_selected)
        diversified = reranked[: self._final_top_k]
        hits = [
            RetrievalHit(
                rank=rank,
                node_id=candidate.node.node_id,
                paper_id=candidate.node.paper_id,
                paper_title=candidate.paper_title,
                section_path=candidate.node.section_path,
                semantic_role=candidate.node.semantic_role,
                block_types=candidate.node.block_types,
                object_labels=candidate.node.object_labels,
                page_start=candidate.node.page_start,
                page_end=candidate.node.page_end,
                text=candidate.node.text,
                vector_score=candidate.vector_score,
                ranking_score=ranking_score,
                rerank_score=candidate.rerank_score,
                source=candidate.source,
                expanded_from=candidate.expanded_from,
                figure_asset=candidate.node.figure_asset,
                figure_caption=candidate.node.figure_caption,
                raw_asset_ref=candidate.node.raw_asset_ref,
                table_rows=candidate.node.table_rows,
                spans=candidate.node.spans,
            )
            for rank, (candidate, ranking_score) in enumerate(diversified, start=1)
        ]
        return TreeRetrievalReport(
            query=query,
            mode=mode,
            paper_ids=requested_paper_ids,
            searched_globally=not requested_paper_ids,
            candidate_paper_ids=candidate_paper_ids,
            initial_hit_count=len(hierarchical_matches) + len(fallback_matches),
            expanded_candidate_count=expanded_count,
            deduplicated_candidate_count=len(deduplicated),
            mmr_candidate_count=len(mmr_selected),
            reranker_name=self._reranker.name if self._reranker is not None else None,
            reranker_applied=(
                self._reranker is not None and reranker_error is None and bool(mmr_selected)
            ),
            reranker_error=reranker_error,
            hits=hits,
        )

    def _select_sections(
        self,
        matches: list[TreeVectorMatch],
    ) -> list[TreeIndexNode]:
        counts: dict[str, int] = {}
        selected: list[TreeIndexNode] = []
        for match in matches:
            node = match.node
            if node.node_type is not TreeNodeType.SECTION:
                continue
            count = counts.get(node.paper_id, 0)
            if count >= self._sections_per_paper:
                continue
            selected.append(node)
            counts[node.paper_id] = count + 1
        return selected

    def _add_hierarchy_scores(
        self,
        *,
        candidates: dict[str, _Candidate],
        indexed_nodes: list[IndexedTreeNode],
        query_embedding: list[float],
    ) -> None:
        by_id = {item.node.node_id: item for item in indexed_nodes}
        roots = {
            item.node.paper_id: item
            for item in indexed_nodes
            if item.node.node_type is TreeNodeType.ROOT
        }
        for node_id, candidate in list(candidates.items()):
            parent = by_id.get(candidate.node.parent_id or "")
            root = roots.get(candidate.node.paper_id)
            indexed_candidate = by_id.get(candidate.node.node_id)
            candidates[node_id] = _Candidate(
                node=candidate.node,
                vector_score=candidate.vector_score,
                source=candidate.source,
                expanded_from=candidate.expanded_from,
                paper_score=(
                    self._cosine_similarity(query_embedding, root.embedding)
                    if root is not None
                    else 0.0
                ),
                section_score=(
                    self._cosine_similarity(query_embedding, parent.embedding)
                    if parent is not None
                    else 0.0
                ),
                paper_title=root.node.title if root is not None else None,
                embedding=(
                    list(indexed_candidate.embedding)
                    if indexed_candidate is not None
                    else None
                ),
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
    def _deduplicate_normalized_text(
        cls,
        candidates: list[tuple[_Candidate, float]],
    ) -> list[tuple[_Candidate, float]]:
        seen: set[str] = set()
        selected: list[tuple[_Candidate, float]] = []
        for item in candidates:
            normalized = cls._normalize_text(item[0].node.text)
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append(item)
        return selected

    def _mmr_select(
        self,
        candidates: list[tuple[_Candidate, float]],
    ) -> list[tuple[_Candidate, float]]:
        if len(candidates) < 2:
            return list(candidates)
        scores = [score for _, score in candidates]
        low = min(scores)
        high = max(scores)

        def normalized_relevance(score: float) -> float:
            return 1.0 if math.isclose(high, low) else (score - low) / (high - low)

        remaining = list(candidates)
        selected: list[tuple[_Candidate, float]] = []
        while remaining and len(selected) < self._mmr_top_k:
            best = max(
                remaining,
                key=lambda item: (
                    self._mmr_lambda * normalized_relevance(item[1])
                    - (1.0 - self._mmr_lambda)
                    * max(
                        (
                            self._candidate_similarity(item[0], chosen[0])
                            for chosen in selected
                        ),
                        default=0.0,
                    )
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    async def _rerank_candidates(
        self,
        query: str,
        candidates: list[tuple[_Candidate, float]],
    ) -> tuple[list[tuple[_Candidate, float]], str | None]:
        if self._reranker is None or not candidates:
            return candidates, None
        documents = [
            "\n".join(
                [
                    candidate.paper_title or candidate.node.paper_id,
                    *candidate.node.section_path,
                    candidate.node.text,
                ]
            )
            for candidate, _ in candidates
        ]
        try:
            scores = await self._reranker.rerank(query, documents)
            if len(scores) != len(candidates):
                raise ValueError("Reranker score count does not match candidate count")
        except (RuntimeError, ValueError, TypeError) as error:
            return candidates, str(error)
        reranked = [
            (replace(candidate, rerank_score=score), score)
            for (candidate, _), score in zip(candidates, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: item[1], reverse=True), None

    @classmethod
    def _candidate_similarity(cls, left: _Candidate, right: _Candidate) -> float:
        if left.embedding is not None and right.embedding is not None:
            return max(0.0, cls._cosine_similarity(left.embedding, right.embedding))
        left_terms = cls._terms(left.node.text)
        right_terms = cls._terms(right.node.text)
        union = left_terms | right_terms
        return len(left_terms & right_terms) / len(union) if union else 0.0

    def _filter_weak_candidates(
        self,
        candidates: list[tuple[_Candidate, float]],
    ) -> list[tuple[_Candidate, float]]:
        if not candidates:
            return []
        best_score = candidates[0][1]
        threshold = max(self._min_ranking_score, best_score - self._score_window)
        return [item for item in candidates if item[1] >= threshold]

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
        return (
            0.50 * candidate.vector_score
            + 0.25 * candidate.section_score
            + 0.15 * candidate.paper_score
            + 0.10 * lexical_score
        )

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
    def _normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        normalized = re.sub(r"-\s*\n\s*", "", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

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
