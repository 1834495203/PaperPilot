import asyncio
import math
import re
import unicodedata
from dataclasses import dataclass, field, replace

from app.domain.ports import TextEmbeddingGateway, TextRerankerGateway, TreeVectorStore
from app.domain.rag import (
    CoverageCell,
    CoverageMatrix,
    CoverageStatus,
    IndexedTreeNode,
    KeywordSearchResult,
    RetrievalBudget,
    RetrievalHit,
    RetrievalMode,
    RetrievalQuery,
    RetrievalSource,
    RetrievalStrategy,
    TreeIndexNode,
    TreeKeywordMatch,
    TreeNodeType,
    TreeRetrievalReport,
    TreeVectorMatch,
)
from app.domain.text import keyword_terms, lexical_terms


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
    rerank_query: str | None = None
    keyword_score: float | None = None
    fused_score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    matched_queries: list[str] = field(default_factory=list)
    matched_dimensions: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Recall:
    """Raw channel output for one retrieval sub-question, before fusion."""

    query: RetrievalQuery
    embedding: list[float]
    routed_paper_ids: tuple[str, ...]
    roots: tuple[TreeIndexNode, ...]
    sections: tuple[TreeIndexNode, ...]
    vector_matches: tuple[TreeVectorMatch, ...]
    keyword_matches: tuple[TreeKeywordMatch, ...]
    keyword_pool_size: int = 0
    keyword_truncated_terms: tuple[str, ...] = ()


class TreeRagRetriever:
    """Retrieve globally with paper -> section -> chunk routing and tree expansion.

    The pipeline is: route papers -> recall through two independent channels
    (vector and lexical) -> fuse with reciprocal rank fusion -> expand the tree ->
    deduplicate and diversify -> rerank a deliberately wide pool -> select
    evidence by paper coverage, coverage dimension and budget.
    """

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
        keyword_top_k: int = 10,
        keyword_enabled: bool = True,
        rerank_candidate_limit: int = 30,
        multi_paper_chunks_per_paper: int = 2,
        survey_paper_top_k: int = 20,
        survey_final_top_k: int = 16,
        max_final_top_k: int = 24,
        max_expansion_levels: int = 2,
        rrf_constant: int = 60,
        rrf_ranking_weight: float = 0.20,
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
            "keyword_top_k": keyword_top_k,
            "rerank_candidate_limit": rerank_candidate_limit,
            "multi_paper_chunks_per_paper": multi_paper_chunks_per_paper,
            "survey_paper_top_k": survey_paper_top_k,
            "survey_final_top_k": survey_final_top_k,
            "max_final_top_k": max_final_top_k,
            "max_expansion_levels": max_expansion_levels,
            "rrf_constant": rrf_constant,
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
        if not 0.0 <= rrf_ranking_weight <= 1.0:
            raise ValueError("rrf_ranking_weight must be between 0 and 1")
        if final_top_k > max_final_top_k:
            raise ValueError("final_top_k cannot exceed max_final_top_k")
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
        self._keyword_top_k = keyword_top_k
        self._keyword_enabled = keyword_enabled
        self._rerank_candidate_limit = rerank_candidate_limit
        self._multi_paper_chunks_per_paper = multi_paper_chunks_per_paper
        self._survey_paper_top_k = survey_paper_top_k
        self._survey_final_top_k = survey_final_top_k
        self._max_final_top_k = max_final_top_k
        self._max_expansion_levels = max_expansion_levels
        self._rrf_constant = rrf_constant
        self._rrf_ranking_weight = rrf_ranking_weight
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
        strategy: RetrievalStrategy | None = None,
        queries: list[RetrievalQuery] | None = None,
        budget: RetrievalBudget | None = None,
    ) -> TreeRetrievalReport:
        """Retrieve evidence for one question or for an explicit sub-question list.

        ``queries`` turns the call into a per-paper, per-dimension retrieval plan:
        every sub-question is recalled with its own scope and mode, and the
        coverage matrix reports which paper x dimension cells are still empty.
        """

        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        requested_paper_ids = list(dict.fromkeys(paper_ids)) if paper_ids else []
        plan_queries = self._plan_queries(
            query=query,
            mode=mode,
            requested_paper_ids=requested_paper_ids,
            queries=queries,
        )
        resolved_strategy = strategy or self._infer_strategy(
            requested_paper_ids, plan_queries
        )
        resolved_budget = budget or self._budget_for(
            resolved_strategy, requested_paper_ids, plan_queries
        )
        coverage_dimensions = list(
            dict.fromkeys(item.dimension for item in plan_queries if item.dimension)
        )
        target_paper_ids = list(
            dict.fromkeys(
                [
                    *requested_paper_ids,
                    *(paper_id for item in plan_queries for paper_id in item.paper_ids),
                ]
            )
        )

        recalls = await asyncio.gather(
            *(self._recall(item, resolved_budget) for item in plan_queries)
        )
        routed_paper_ids = list(
            dict.fromkeys(
                paper_id for recall in recalls for paper_id in recall.routed_paper_ids
            )
        )
        roots: dict[str, TreeIndexNode] = {}
        sections: list[TreeIndexNode] = []
        keyword_candidate_count = 0
        keyword_pool_size = 0
        keyword_truncated_terms: list[str] = []
        for recall in recalls:
            keyword_candidate_count += len(recall.keyword_matches)
            keyword_pool_size += recall.keyword_pool_size
            keyword_truncated_terms.extend(recall.keyword_truncated_terms)
            for root in recall.roots:
                roots.setdefault(root.paper_id, root)
            sections.extend(recall.sections)

        candidates = self._merge_channels(recalls)
        candidate_paper_ids = list(
            dict.fromkeys(
                [
                    *routed_paper_ids,
                    *(candidate.node.paper_id for candidate in candidates.values()),
                ]
            )
        )
        query_embeddings = [recall.embedding for recall in recalls]
        if self._should_expand(mode, plan_queries):
            await self._expand_candidates(
                candidates=candidates,
                initial_nodes=sections,
                query_embeddings=query_embeddings,
                budget=resolved_budget,
            )
        await self._add_hierarchy_scores(
            candidates=candidates,
            roots=roots,
            query_embeddings=query_embeddings,
        )

        expanded_count = sum(
            candidate.source is RetrievalSource.TREE_EXPANSION
            for candidate in candidates.values()
        )
        fused_total = sum(1 for candidate in candidates.values() if candidate.fused_score > 0)
        ranked = sorted(
            (
                (candidate, self._ranking_score(candidate, fallback_query=query))
                for candidate in candidates.values()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        reserved, reserved_node_ids = self._reserve_candidates(
            ranked,
            target_paper_ids=target_paper_ids,
            coverage_dimensions=coverage_dimensions,
            budget=resolved_budget,
        )
        ranked = self._filter_weak_candidates(
            reserved,
            relative_window=(
                resolved_strategy is RetrievalStrategy.SINGLE_PAPER
            ),
            protected_node_ids=reserved_node_ids,
        )
        deduplicated = self._deduplicate_normalized_text(ranked)
        mmr_selected = self._mmr_select(
            deduplicated,
            limit=resolved_budget.rerank_candidate_limit,
            required_node_ids=reserved_node_ids,
        )
        reranked, reranker_error = await self._rerank_candidates(
            mmr_selected,
            fallback_query=query,
        )
        selected, missing_paper_ids = self._select_final(
            ranked=reranked,
            budget=resolved_budget,
            target_paper_ids=target_paper_ids,
            coverage_dimensions=coverage_dimensions,
        )
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
                rerank_query=candidate.rerank_query,
                keyword_score=candidate.keyword_score,
                fused_score=candidate.fused_score,
                matched_terms=candidate.matched_terms,
                matched_queries=candidate.matched_queries,
                matched_dimensions=candidate.matched_dimensions,
                source=candidate.source,
                expanded_from=candidate.expanded_from,
                figure_asset=candidate.node.figure_asset,
                figure_caption=candidate.node.figure_caption,
                raw_asset_ref=candidate.node.raw_asset_ref,
                table_rows=candidate.node.table_rows,
                spans=candidate.node.spans,
            )
            for rank, (candidate, ranking_score) in enumerate(selected, start=1)
        ]
        return TreeRetrievalReport(
            query=query,
            mode=mode,
            paper_ids=requested_paper_ids,
            strategy=resolved_strategy,
            budget=resolved_budget,
            queries=plan_queries,
            searched_globally=not requested_paper_ids,
            candidate_paper_ids=candidate_paper_ids,
            missing_paper_ids=missing_paper_ids,
            coverage=self._build_coverage(
                hits=hits,
                target_paper_ids=target_paper_ids,
                coverage_dimensions=coverage_dimensions,
            ),
            initial_hit_count=sum(
                len(recall.vector_matches) + len(recall.keyword_matches)
                for recall in recalls
            ),
            expanded_candidate_count=expanded_count,
            deduplicated_candidate_count=len(deduplicated),
            mmr_candidate_count=len(mmr_selected),
            keyword_candidate_count=keyword_candidate_count,
            keyword_pool_size=keyword_pool_size,
            keyword_truncated_terms=list(dict.fromkeys(keyword_truncated_terms)),
            fused_candidate_count=fused_total,
            reranker_name=self._reranker.name if self._reranker is not None else None,
            reranker_applied=(
                self._reranker is not None and reranker_error is None and bool(mmr_selected)
            ),
            reranker_error=reranker_error,
            hits=hits,
        )

    @staticmethod
    def _plan_queries(
        *,
        query: str,
        mode: RetrievalMode,
        requested_paper_ids: list[str],
        queries: list[RetrievalQuery] | None,
    ) -> list[RetrievalQuery]:
        if not queries:
            return [
                RetrievalQuery(
                    query=query,
                    mode=mode,
                    paper_ids=requested_paper_ids,
                )
            ]
        scoped = [
            item
            if item.paper_ids or not requested_paper_ids
            else item.model_copy(update={"paper_ids": requested_paper_ids})
            for item in queries
        ]
        unique: dict[tuple[str, str, tuple[str, ...], str | None], RetrievalQuery] = {}
        for item in scoped:
            unique.setdefault(
                (item.query, item.mode.value, tuple(item.paper_ids), item.dimension),
                item,
            )
        return list(unique.values())

    @staticmethod
    def _infer_strategy(
        requested_paper_ids: list[str],
        plan_queries: list[RetrievalQuery],
    ) -> RetrievalStrategy:
        scoped_paper_ids = {
            paper_id for item in plan_queries for paper_id in item.paper_ids
        }
        targets = set(requested_paper_ids) | scoped_paper_ids
        if len(targets) > 1:
            return RetrievalStrategy.MULTI_PAPER
        if len(plan_queries) > 1 or len(
            {item.dimension for item in plan_queries if item.dimension}
        ) > 1:
            return RetrievalStrategy.CORPUS_SURVEY
        return RetrievalStrategy.SINGLE_PAPER

    def _budget_for(
        self,
        strategy: RetrievalStrategy,
        requested_paper_ids: list[str],
        plan_queries: list[RetrievalQuery],
    ) -> RetrievalBudget:
        base = RetrievalBudget(
            max_papers=self._paper_top_k,
            sections_per_paper=self._sections_per_paper,
            initial_top_k=self._initial_top_k,
            global_fallback_top_k=self._global_fallback_top_k,
            keyword_top_k=self._keyword_top_k,
            max_expanded_per_hit=self._max_expanded_per_hit,
            rerank_candidate_limit=self._rerank_candidate_limit,
            max_chunks_per_paper=self._max_chunks_per_paper,
            final_top_k=self._final_top_k,
        )
        if strategy is RetrievalStrategy.MULTI_PAPER:
            # A plan scopes each sub-question to one paper, so the budget has to
            # count the union of those scopes, not just the top-level argument.
            scoped_papers = list(
                dict.fromkeys(
                    [
                        *requested_paper_ids,
                        *(paper_id for item in plan_queries for paper_id in item.paper_ids),
                    ]
                )
            )
            targets = max(len(scoped_papers), 1)
            dimension_count = max(
                len({item.dimension for item in plan_queries if item.dimension}), 1
            )
            per_paper = max(self._multi_paper_chunks_per_paper, dimension_count)
            return base.model_copy(
                update={
                    "max_papers": max(base.max_papers, targets),
                    "final_top_k": self._clamp_final(
                        max(
                            base.final_top_k,
                            targets * per_paper,
                            len(plan_queries) * 2,
                        )
                    ),
                    "max_chunks_per_paper": per_paper,
                    "sections_per_paper": max(base.sections_per_paper, 2),
                    "initial_top_k": max(base.initial_top_k, targets * 3),
                    "global_fallback_top_k": max(base.global_fallback_top_k, targets * 2),
                    "keyword_top_k": max(base.keyword_top_k, targets * 2),
                    "rerank_candidate_limit": max(
                        base.rerank_candidate_limit,
                        targets * per_paper * 3,
                    ),
                }
            )
        if strategy is RetrievalStrategy.CORPUS_SURVEY:
            return base.model_copy(
                update={
                    "max_papers": max(base.max_papers, self._survey_paper_top_k),
                    "final_top_k": self._clamp_final(
                        max(base.final_top_k, self._survey_final_top_k)
                    ),
                    "max_chunks_per_paper": min(base.max_chunks_per_paper, 2),
                    "initial_top_k": max(base.initial_top_k, 20),
                    "global_fallback_top_k": max(base.global_fallback_top_k, 12),
                    "keyword_top_k": max(base.keyword_top_k, 12),
                    "rerank_candidate_limit": max(base.rerank_candidate_limit, 40),
                }
            )
        return base.model_copy(
            update={
                "final_top_k": self._clamp_final(
                    max(base.final_top_k, len(plan_queries) * 2)
                ),
                "sections_per_paper": max(
                    base.sections_per_paper, min(len(plan_queries), 6)
                ),
                "initial_top_k": max(base.initial_top_k, len(plan_queries) * 3),
            }
        )

    def _clamp_final(self, value: int) -> int:
        return min(value, self._max_final_top_k)

    async def _recall(
        self,
        item: RetrievalQuery,
        budget: RetrievalBudget,
    ) -> _Recall:
        query_embedding = await self._embedder.embed_query(item.query)
        scoped_paper_ids = list(dict.fromkeys(item.paper_ids))
        root_matches = await self._vector_store.similarity_search(
            query_embedding,
            paper_ids=scoped_paper_ids or None,
            top_k=max(budget.max_papers, len(scoped_paper_ids)),
            chunks_only=False,
            node_types=[TreeNodeType.ROOT],
        )
        routed_paper_ids = (
            scoped_paper_ids
            if scoped_paper_ids
            else list(
                dict.fromkeys(match.node.paper_id for match in root_matches)
            )[: budget.max_papers]
        )
        section_matches: list[TreeVectorMatch] = []
        if routed_paper_ids:
            per_paper_matches = await asyncio.gather(
                *(
                    self._vector_store.similarity_search(
                        query_embedding,
                        paper_ids=[paper_id],
                        top_k=budget.sections_per_paper,
                        chunks_only=False,
                        node_types=[TreeNodeType.SECTION],
                    )
                    for paper_id in routed_paper_ids
                )
            )
            section_matches = [
                match for paper_matches in per_paper_matches for match in paper_matches
            ]
        sections = self._select_sections(
            section_matches,
            sections_per_paper=budget.sections_per_paper,
        )
        hierarchical_matches: list[TreeVectorMatch] = []
        if sections:
            hierarchical_matches = await self._vector_store.similarity_search(
                query_embedding,
                paper_ids=routed_paper_ids,
                top_k=budget.initial_top_k,
                chunks_only=True,
                parent_ids=[node.node_id for node in sections],
            )
        fallback_matches = await self._vector_store.similarity_search(
            query_embedding,
            paper_ids=scoped_paper_ids or None,
            top_k=budget.global_fallback_top_k,
            chunks_only=True,
        )
        keyword_result = (
            await self._keyword_recall(
                item.query,
                paper_ids=scoped_paper_ids,
                top_k=budget.keyword_top_k,
            )
            if self._keyword_enabled
            else KeywordSearchResult()
        )
        return _Recall(
            query=item,
            embedding=query_embedding,
            routed_paper_ids=tuple(routed_paper_ids),
            roots=tuple(match.node for match in root_matches),
            sections=tuple(sections),
            vector_matches=tuple(
                self._deduplicate_matches([*hierarchical_matches, *fallback_matches])
            ),
            keyword_matches=tuple(keyword_result.matches),
            keyword_pool_size=keyword_result.pool_size,
            keyword_truncated_terms=tuple(keyword_result.truncated_terms),
        )

    async def _keyword_recall(
        self,
        query: str,
        *,
        paper_ids: list[str],
        top_k: int,
    ) -> KeywordSearchResult:
        terms = keyword_terms(query, max_terms=12)
        if not terms:
            return KeywordSearchResult()
        return await self._vector_store.keyword_search(
            terms,
            paper_ids=paper_ids or None,
            top_k=top_k,
        )

    @staticmethod
    def _deduplicate_matches(
        matches: list[TreeVectorMatch],
    ) -> list[TreeVectorMatch]:
        best: dict[str, TreeVectorMatch] = {}
        for match in matches:
            if match.node.node_type is not TreeNodeType.CHUNK:
                continue
            existing = best.get(match.node.node_id)
            if existing is None or existing.vector_score < match.vector_score:
                best[match.node.node_id] = match
        return sorted(
            best.values(),
            key=lambda match: match.vector_score,
            reverse=True,
        )

    def _merge_channels(self, recalls: list[_Recall]) -> dict[str, _Candidate]:
        """Fuse the vector and lexical rankings of every sub-question with RRF.

        Each channel contributes ``1 / (k + rank)`` once per sub-question, so a
        chunk recalled by both channels of one sub-question is credited twice for
        that sub-question and a chunk recalled by several sub-questions outranks a
        chunk recalled by one. The accumulated value is normalized to (0, 1].
        """

        candidates: dict[str, _Candidate] = {}
        fused_totals: dict[str, float] = {}
        for recall in recalls:
            for rank, match in enumerate(recall.vector_matches, start=1):
                node_id = match.node.node_id
                fused_totals[node_id] = (
                    fused_totals.get(node_id, 0.0) + self._rrf_weight(rank)
                )
            for rank, keyword_match in enumerate(recall.keyword_matches, start=1):
                node_id = keyword_match.node.node_id
                fused_totals[node_id] = (
                    fused_totals.get(node_id, 0.0) + self._rrf_weight(rank)
                )
            for match in recall.vector_matches:
                self._merge_vector_match(candidates, match, recall)
            for keyword_match in recall.keyword_matches:
                self._merge_keyword_match(candidates, keyword_match, recall)
        if not candidates:
            return candidates
        best_fused = max(fused_totals.values(), default=0.0)
        if best_fused <= 0:
            return candidates
        return {
            node_id: replace(
                candidate,
                fused_score=fused_totals.get(node_id, 0.0) / best_fused,
            )
            for node_id, candidate in candidates.items()
        }

    def _rrf_weight(self, rank: int) -> float:
        return 1.0 / (self._rrf_constant + rank)

    @staticmethod
    def _merge_vector_match(
        candidates: dict[str, _Candidate],
        match: TreeVectorMatch,
        recall: _Recall,
    ) -> None:
        node_id = match.node.node_id
        existing = candidates.get(node_id)
        contributions = _query_contributions(recall)
        if existing is None:
            candidates[node_id] = _Candidate(
                node=match.node,
                vector_score=match.vector_score,
                source=RetrievalSource.VECTOR,
                matched_queries=[recall.query.query],
                matched_dimensions=contributions,
            )
            return
        candidates[node_id] = replace(
            existing,
            vector_score=max(existing.vector_score, match.vector_score),
            matched_queries=_append(existing.matched_queries, recall.query.query),
            matched_dimensions=_extend(existing.matched_dimensions, contributions),
        )

    @staticmethod
    def _merge_keyword_match(
        candidates: dict[str, _Candidate],
        match: TreeKeywordMatch,
        recall: _Recall,
    ) -> None:
        node_id = match.node.node_id
        existing = candidates.get(node_id)
        contributions = _query_contributions(recall)
        if existing is None:
            candidates[node_id] = _Candidate(
                node=match.node,
                vector_score=0.0,
                source=RetrievalSource.KEYWORD,
                keyword_score=match.keyword_score,
                matched_terms=list(match.matched_terms),
                matched_queries=[recall.query.query],
                matched_dimensions=contributions,
            )
            return
        candidates[node_id] = replace(
            existing,
            keyword_score=max(existing.keyword_score or 0.0, match.keyword_score),
            matched_terms=_extend(existing.matched_terms, match.matched_terms),
            matched_queries=_append(existing.matched_queries, recall.query.query),
            matched_dimensions=_extend(existing.matched_dimensions, contributions),
        )

    @staticmethod
    def _select_sections(
        matches: list[TreeVectorMatch],
        *,
        sections_per_paper: int,
    ) -> list[TreeIndexNode]:
        counts: dict[str, int] = {}
        selected: list[TreeIndexNode] = []
        for match in matches:
            node = match.node
            if node.node_type is not TreeNodeType.SECTION:
                continue
            count = counts.get(node.paper_id, 0)
            if count >= sections_per_paper:
                continue
            selected.append(node)
            counts[node.paper_id] = count + 1
        return selected

    async def _add_hierarchy_scores(
        self,
        *,
        candidates: dict[str, _Candidate],
        roots: dict[str, TreeIndexNode],
        query_embeddings: list[list[float]],
    ) -> None:
        """Attach parent and paper context by loading only the nodes in play."""

        if not candidates:
            return
        parent_ids = list(
            dict.fromkeys(
                candidate.node.parent_id
                for candidate in candidates.values()
                if candidate.node.parent_id
            )
        )
        loaded = await self._vector_store.load_nodes(
            node_ids=[*candidates.keys(), *parent_ids, *[root.node_id for root in roots.values()]]
        )
        by_id = {item.node.node_id: item for item in loaded}
        root_by_paper: dict[str, TreeIndexNode] = {
            item.node.paper_id: item.node
            for item in loaded
            if item.node.node_type is TreeNodeType.ROOT
        }
        root_by_paper.update(
            {
                paper_id: node
                for paper_id, node in roots.items()
                if paper_id not in root_by_paper
            }
        )
        missing_root_papers = [
            paper_id
            for paper_id in dict.fromkeys(
                candidate.node.paper_id for candidate in candidates.values()
            )
            if paper_id not in root_by_paper
        ]
        if missing_root_papers:
            extra = await self._vector_store.load_nodes(
                paper_ids=missing_root_papers,
                node_types=[TreeNodeType.ROOT],
            )
            for item in extra:
                if item.node.node_type is TreeNodeType.ROOT:
                    root_by_paper.setdefault(item.node.paper_id, item.node)

        for node_id, candidate in list(candidates.items()):
            parent = by_id.get(candidate.node.parent_id or "")
            root = root_by_paper.get(candidate.node.paper_id)
            indexed_root = by_id.get(root.node_id) if root is not None else None
            indexed_candidate = by_id.get(node_id)
            candidates[node_id] = replace(
                candidate,
                paper_score=(
                    max(
                        self._cosine_similarity(embedding, indexed_root.embedding)
                        for embedding in query_embeddings
                    )
                    if indexed_root is not None
                    else 0.0
                ),
                section_score=(
                    max(
                        self._cosine_similarity(embedding, parent.embedding)
                        for embedding in query_embeddings
                    )
                    if parent is not None
                    else 0.0
                ),
                paper_title=root.title if root is not None else candidate.paper_title,
                embedding=(
                    list(indexed_candidate.embedding)
                    if indexed_candidate is not None
                    else candidate.embedding
                ),
            )

    @staticmethod
    def _should_expand(
        mode: RetrievalMode,
        plan_queries: list[RetrievalQuery],
    ) -> bool:
        return mode.expands_tree or any(item.mode.expands_tree for item in plan_queries)

    async def _expand_candidates(
        self,
        *,
        candidates: dict[str, _Candidate],
        initial_nodes: list[TreeIndexNode],
        query_embeddings: list[list[float]],
        budget: RetrievalBudget,
    ) -> None:
        """Walk child levels on demand instead of loading whole paper trees."""

        frontier = list(
            dict.fromkeys(
                anchor.parent_id
                if anchor.node_type is TreeNodeType.CHUNK and anchor.parent_id
                else anchor.node_id
                for anchor in initial_nodes
            )
        )
        if not frontier:
            return
        inherited = self._section_provenance(candidates)
        visited: set[str] = set()
        for _ in range(self._max_expansion_levels):
            if not frontier:
                return
            visited.update(frontier)
            children = await self._vector_store.load_nodes(parent_ids=frontier)
            grouped: dict[str, list[tuple[IndexedTreeNode, float]]] = {}
            deeper: list[str] = []
            for item in children:
                if item.node.node_type is not TreeNodeType.CHUNK:
                    if item.node.node_id not in visited:
                        deeper.append(item.node.node_id)
                    continue
                if item.node.node_id in candidates:
                    continue
                score = max(
                    self._cosine_similarity(embedding, item.embedding)
                    for embedding in query_embeddings
                )
                grouped.setdefault(item.node.parent_id or "", []).append((item, score))
            for anchor_id, items in grouped.items():
                items.sort(key=lambda entry: entry[1], reverse=True)
                queries, dimensions = inherited.get(anchor_id, ([], []))
                for indexed_node, score in items[: budget.max_expanded_per_hit]:
                    candidates[indexed_node.node.node_id] = _Candidate(
                        node=indexed_node.node,
                        vector_score=score,
                        source=RetrievalSource.TREE_EXPANSION,
                        expanded_from=anchor_id,
                        matched_queries=list(queries),
                        matched_dimensions=list(dimensions),
                    )
            frontier = list(dict.fromkeys(deeper))

    @staticmethod
    def _section_provenance(
        candidates: dict[str, _Candidate],
    ) -> dict[str, tuple[list[str], list[str]]]:
        """Map a parent section to the sub-questions and dimensions that reached it."""

        provenance: dict[str, tuple[list[str], list[str]]] = {}
        for candidate in candidates.values():
            parent_id = candidate.node.parent_id
            if not parent_id:
                continue
            queries, dimensions = provenance.get(parent_id, ([], []))
            provenance[parent_id] = (
                _extend(queries, candidate.matched_queries),
                _extend(dimensions, candidate.matched_dimensions),
            )
        return provenance

    @classmethod
    def _deduplicate_normalized_text(
        cls,
        candidates: list[tuple[_Candidate, float]],
    ) -> list[tuple[_Candidate, float]]:
        """Drop repeated text within one paper, never across papers.

        Two papers can carry identical boilerplate, and dropping one of them would
        silently remove a paper's only evidence and break paper coverage, so the
        key is the paper and the normalized text together.
        """

        seen: set[tuple[str, str]] = set()
        selected: list[tuple[_Candidate, float]] = []
        for item in candidates:
            key = (item[0].node.paper_id, cls._normalize_text(item[0].node.text))
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
        return selected

    def _reserve_candidates(
        self,
        ranked: list[tuple[_Candidate, float]],
        *,
        target_paper_ids: list[str],
        coverage_dimensions: list[str],
        budget: RetrievalBudget,
    ) -> tuple[list[tuple[_Candidate, float]], set[str]]:
        """Keep per-paper and per-cell representation before the global cut.

        The candidate pool is capped before thresholding, deduplication and
        reranking, so a named paper whose evidence ranks just below the global cut
        would otherwise be unrecoverable later. Each target paper and each
        paper x dimension cell therefore keeps its best candidates first, then the
        remaining budget is filled by score. The reserved node IDs are returned so
        later stages can keep protecting them.
        """

        limit = self._max_candidates
        reserved: list[tuple[_Candidate, float]] = []
        taken: set[str] = set()
        protected: set[str] = set()

        def take(item: tuple[_Candidate, float], *, protect: bool = False) -> bool:
            node_id = item[0].node.node_id
            if node_id in taken:
                return False
            taken.add(node_id)
            reserved.append(item)
            if protect:
                protected.add(node_id)
            return True

        def take_next_for(
            paper_id: str,
            *,
            dimension: str | None = None,
        ) -> bool:
            """Reserve the best not-yet-reserved candidate of one paper.

            Already reserved candidates are skipped rather than ending the search,
            so a paper whose best candidate was reserved by an earlier pass can
            still contribute its next one.
            """

            for item in ranked:
                node_id = item[0].node.node_id
                if node_id in taken or item[0].node.paper_id != paper_id:
                    continue
                if dimension is not None and dimension not in item[0].matched_dimensions:
                    continue
                return take(item, protect=True)
            return False

        per_paper_reserve = max(1, budget.max_chunks_per_paper)
        if target_paper_ids:
            reserve_dimensions: list[str | None] = list(coverage_dimensions) or [None]
            for dimension in reserve_dimensions:
                for paper_id in target_paper_ids:
                    take_next_for(paper_id, dimension=dimension)
            for _ in range(per_paper_reserve):
                for paper_id in target_paper_ids:
                    take_next_for(paper_id)
            limit = max(limit, len(reserved))
        for item in ranked:
            if len(reserved) >= limit:
                break
            take(item)
        return sorted(reserved, key=lambda item: item[1], reverse=True), protected

    def _select_final(
        self,
        *,
        ranked: list[tuple[_Candidate, float]],
        budget: RetrievalBudget,
        target_paper_ids: list[str],
        coverage_dimensions: list[str],
    ) -> tuple[list[tuple[_Candidate, float]], list[str]]:
        """Choose evidence after reranking: coverage first, then quality order.

        Every target paper with evidence gets representation before the budget is
        filled by score, and a paper x dimension cell is filled from the best
        candidate that actually matched that dimension.
        """

        if not ranked:
            return [], list(target_paper_ids)
        counts: dict[str, int] = {}
        selected: list[tuple[_Candidate, float]] = []
        taken: set[str] = set()

        def take(item: tuple[_Candidate, float]) -> bool:
            node_id = item[0].node.node_id
            paper_id = item[0].node.paper_id
            if node_id in taken or len(selected) >= budget.final_top_k:
                return False
            if counts.get(paper_id, 0) >= budget.max_chunks_per_paper:
                return False
            taken.add(node_id)
            counts[paper_id] = counts.get(paper_id, 0) + 1
            selected.append(item)
            return True

        if coverage_dimensions:
            # Dimensions iterate outermost so a paper that appears late in the
            # comparison still gets its first cell before any paper gets a second.
            for dimension in coverage_dimensions:
                for paper_id in target_paper_ids:
                    match = next(
                        (
                            item
                            for item in ranked
                            if item[0].node.paper_id == paper_id
                            and item[0].node.node_id not in taken
                            and dimension in item[0].matched_dimensions
                        ),
                        None,
                    )
                    if match is not None:
                        take(match)
        for paper_id in target_paper_ids:
            match = next(
                (
                    item
                    for item in ranked
                    if item[0].node.paper_id == paper_id and item[0].node.node_id not in taken
                ),
                None,
            )
            if match is not None:
                take(match)
        for _ in range(budget.max_chunks_per_paper):
            for paper_id in target_paper_ids:
                match = next(
                    (
                        item
                        for item in ranked
                        if item[0].node.paper_id == paper_id
                        and item[0].node.node_id not in taken
                    ),
                    None,
                )
                if match is not None:
                    take(match)
        for item in ranked:
            take(item)
        covered_paper_ids = {item[0].node.paper_id for item in selected}
        missing_paper_ids = [
            paper_id for paper_id in target_paper_ids if paper_id not in covered_paper_ids
        ]
        return selected, missing_paper_ids

    @staticmethod
    def _build_coverage(
        *,
        hits: list[RetrievalHit],
        target_paper_ids: list[str],
        coverage_dimensions: list[str],
    ) -> CoverageMatrix:
        """Report cells that received a candidate, not cells that are answered.

        A delivered chunk only proves that retrieval reached the cell. Whether the
        chunk actually answers the dimension is an evidence judgement, so the cell
        is marked CANDIDATE and an evidence reader must verify or reject it.
        """

        paper_ids = target_paper_ids or list(
            dict.fromkeys(hit.paper_id for hit in hits)
        )
        matrix = CoverageMatrix.build(
            paper_ids=paper_ids,
            dimensions=coverage_dimensions,
        )
        for hit in hits:
            dimensions = (
                hit.matched_dimensions
                or [CoverageMatrix.DEFAULT_DIMENSION]
            )
            for dimension in dimensions:
                cell = matrix.cell_for(hit.paper_id, dimension)
                if cell is None or hit.node_id in cell.evidence_ids:
                    continue
                matrix = matrix.with_cell(
                    CoverageCell(
                        paper_id=cell.paper_id,
                        dimension=cell.dimension,
                        status=CoverageStatus.CANDIDATE,
                        evidence_ids=[*cell.evidence_ids, hit.node_id],
                    )
                )
        return matrix

    def _mmr_select(
        self,
        candidates: list[tuple[_Candidate, float]],
        *,
        limit: int | None = None,
        required_node_ids: set[str] | None = None,
    ) -> list[tuple[_Candidate, float]]:
        """Diversify the pool while keeping reserved coverage candidates.

        Reserved candidates seed the selection instead of competing for it, so a
        named paper's protected evidence cannot be dropped by the diversity pass.
        They still influence the redundancy penalty of everything selected after
        them.
        """

        if len(candidates) < 2:
            return list(candidates)
        pool_limit = limit if limit is not None else self._mmr_top_k
        required = required_node_ids or set()
        required_items = [
            item for item in candidates if item[0].node.node_id in required
        ]
        remaining = [
            item for item in candidates if item[0].node.node_id not in required
        ]
        scores = [score for _, score in candidates]
        low = min(scores)
        high = max(scores)

        def normalized_relevance(score: float) -> float:
            return 1.0 if math.isclose(high, low) else (score - low) / (high - low)

        selected: list[tuple[_Candidate, float]] = []
        for item in required_items:
            if len(selected) >= pool_limit:
                return selected
            selected.append(item)
        while remaining and len(selected) < pool_limit:
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
        candidates: list[tuple[_Candidate, float]],
        *,
        fallback_query: str,
    ) -> tuple[list[tuple[_Candidate, float]], str | None]:
        """Rerank each sub-question's candidates with that sub-question's query.

        A comparison plans one sub-question per paper and dimension. Scoring every
        candidate against only the first sub-question would depress the other
        papers' correct evidence, so candidates are grouped by the sub-question
        that retrieved them, reranked per group, and merged afterwards. Scores stay
        raw so they remain comparable across groups and auditable per hit.
        """

        if self._reranker is None or not candidates:
            return candidates, None
        grouped: dict[str, list[tuple[_Candidate, float]]] = {}
        for item in candidates:
            grouped.setdefault(
                self._candidate_query(item[0], fallback_query),
                [],
            ).append(item)
        scored: list[tuple[_Candidate, float]] = []
        for group_query, items in grouped.items():
            documents = [
                "\n".join(
                    [
                        candidate.paper_title or candidate.node.paper_id,
                        *candidate.node.section_path,
                        candidate.node.text,
                    ]
                )
                for candidate, _ in items
            ]
            try:
                scores = await self._reranker.rerank(group_query, documents)
                if len(scores) != len(items):
                    raise ValueError(
                        "Reranker score count does not match candidate count"
                    )
            except (RuntimeError, ValueError, TypeError) as error:
                return candidates, str(error)
            scored.extend(
                (replace(candidate, rerank_score=score, rerank_query=group_query), score)
                for (candidate, _), score in zip(items, scores, strict=True)
            )
        return sorted(scored, key=lambda item: item[1], reverse=True), None

    @staticmethod
    def _candidate_query(candidate: _Candidate, fallback_query: str) -> str:
        """The sub-question a candidate should be scored against."""

        return candidate.matched_queries[0] if candidate.matched_queries else fallback_query

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
        *,
        relative_window: bool = True,
        protected_node_ids: set[str] | None = None,
    ) -> list[tuple[_Candidate, float]]:
        """Drop weak candidates by an absolute floor and, for narrow tasks, by gap.

        A comparison must not lose its weakest named paper merely because another
        paper scored higher, so multi-paper and survey tasks keep the absolute
        floor only. Reserved coverage candidates bypass the thresholds entirely:
        the point of reserving them is that the named paper still gets to report
        what it does, or does not, contain.
        """

        if not candidates:
            return []
        protected = protected_node_ids or set()
        best_score = candidates[0][1]
        threshold = (
            max(self._min_ranking_score, best_score - self._score_window)
            if relative_window
            else self._min_ranking_score
        )
        return [
            item
            for item in candidates
            if item[1] >= threshold or item[0].node.node_id in protected
        ]

    def _ranking_score(
        self,
        candidate: _Candidate,
        *,
        fallback_query: str,
    ) -> float:
        """Channel-aware pre-rank, scored against the candidate's own sub-question.

        Reciprocal rank fusion contributes as a bounded gain on top of the raw
        recall score rather than replacing it: pure rank fusion normalizes the best
        candidate of any query to 1.0, which would let an unrelated query clear the
        absolute quality floor. Lexical overlap uses the sub-question that retrieved
        the candidate, so evidence planned for one paper is not judged by another
        paper's question.
        """

        query_terms = self._terms(self._candidate_query(candidate, fallback_query))
        searchable = " ".join([*candidate.node.section_path, candidate.node.text])
        candidate_terms = self._terms(searchable)
        lexical_score = (
            len(query_terms & candidate_terms) / len(query_terms)
            if query_terms
            else 0.0
        )
        recall_score = (
            candidate.vector_score
            if candidate.keyword_score is None
            else max(candidate.vector_score, candidate.keyword_score)
        )
        fused_score = min(
            1.0,
            max(
                0.0,
                recall_score + self._rrf_ranking_weight * candidate.fused_score,
            ),
        )
        return (
            0.50 * fused_score
            + 0.25 * candidate.section_score
            + 0.15 * candidate.paper_score
            + 0.10 * lexical_score
        )

    @staticmethod
    def _terms(text: str) -> set[str]:
        return lexical_terms(text)

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


def _query_contributions(recall: _Recall) -> list[str]:
    dimension = recall.query.dimension
    return [dimension] if dimension else []


def _extend(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    for item in additions:
        if item not in merged:
            merged.append(item)
    return merged


def _append(existing: list[str], value: str) -> list[str]:
    return _extend(existing, [value])
