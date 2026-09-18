from collections.abc import Sequence

import pytest

from app.application.tree_retrieval import TreeRagRetriever, _Candidate
from app.domain.ports import TextEmbeddingGateway, TextRerankerGateway, TreeVectorStore
from app.domain.rag import (
    IndexedTreeNode,
    KeywordSearchResult,
    ParsedPaperDocument,
    RetrievalMode,
    RetrievalQuery,
    RetrievalSource,
    RetrievalStrategy,
    TreeIndexNode,
    TreeKeywordMatch,
    TreeNodeType,
    TreeVectorMatch,
)


class _Embedder(TextEmbeddingGateway):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _RecordingReranker(TextRerankerGateway):
    def __init__(self) -> None:
        self.document_count = 0

    @property
    def name(self) -> str:
        return "recording-cross-encoder"

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        self.document_count = len(documents)
        return [1.0 - index * 0.01 for index, _ in enumerate(documents)]


def _chunk(
    paper_id: str,
    name: str,
    text: str,
    *,
    parent_id: str,
    role: str = "method",
) -> TreeIndexNode:
    return TreeIndexNode(
        node_id=f"{paper_id}:{name}",
        paper_id=paper_id,
        node_type=TreeNodeType.CHUNK,
        title=name,
        parent_id=parent_id,
        children_ids=[],
        level=2,
        section_path=[f"3 {role.title()}"],
        semantic_role=role,
        text=text,
        embedding_text=text,
        page_start=3,
        page_end=3,
    )


class _MultiPaperStore(TreeVectorStore):
    """Three routed papers plus one paper only the lexical channel can reach."""

    LEXICAL_ONLY_PAPER = "paper-d"

    def __init__(self) -> None:
        self.paper_ids = ("paper-a", "paper-b", "paper-c")
        self.sections = {
            paper_id: TreeIndexNode(
                node_id=f"{paper_id}:method",
                paper_id=paper_id,
                node_type=TreeNodeType.SECTION,
                title="3 Method",
                parent_id=f"{paper_id}:root",
                children_ids=[f"{paper_id}:chunk"],
                level=1,
                section_path=["3 Method"],
                text=f"{paper_id} method section",
                embedding_text=f"{paper_id} method section",
                page_start=2,
                page_end=5,
            )
            for paper_id in self.paper_ids
        }
        self.roots = {
            paper_id: TreeIndexNode(
                node_id=f"{paper_id}:root",
                paper_id=paper_id,
                node_type=TreeNodeType.ROOT,
                title=f"{paper_id} title",
                parent_id=None,
                children_ids=[f"{paper_id}:method"],
                level=0,
                section_path=[],
                text=f"{paper_id} title",
                embedding_text=f"{paper_id} title",
            )
            for paper_id in self.paper_ids
        }
        self.chunks = {
            "paper-a": _chunk(
                "paper-a",
                "chunk",
                "Paper A constructs a hierarchical index with ancestor titles.",
                parent_id="paper-a:method",
            ),
            "paper-b": _chunk(
                "paper-b",
                "chunk",
                "Paper B reports dataset scale and latency numbers.",
                parent_id="paper-b:method",
            ),
            "paper-c": _chunk(
                "paper-c",
                "chunk",
                "Paper C ablates chunk overlap and section roles.",
                parent_id="paper-c:method",
                role="results",
            ),
        }
        lexical_paper = self.LEXICAL_ONLY_PAPER
        self.roots[lexical_paper] = TreeIndexNode(
            node_id=f"{lexical_paper}:root",
            paper_id=lexical_paper,
            node_type=TreeNodeType.ROOT,
            title=f"{lexical_paper} title",
            parent_id=None,
            children_ids=[f"{lexical_paper}:method"],
            level=0,
            section_path=[],
            text=f"{lexical_paper} title",
            embedding_text=f"{lexical_paper} title",
        )
        self.sections[lexical_paper] = TreeIndexNode(
            node_id=f"{lexical_paper}:method",
            paper_id=lexical_paper,
            node_type=TreeNodeType.SECTION,
            title="4 Results",
            parent_id=f"{lexical_paper}:root",
            children_ids=[f"{lexical_paper}:chunk"],
            level=1,
            section_path=["4 Results"],
            text=f"{lexical_paper} results section",
            embedding_text=f"{lexical_paper} results section",
            page_start=2,
            page_end=5,
        )
        self.chunks[lexical_paper] = _chunk(
            lexical_paper,
            "chunk",
            "Paper D swaps in the Qwen3-Embedding-0613 model for retrieval.",
            parent_id=f"{lexical_paper}:method",
            role="results",
        )
        self._embeddings = {
            "paper-a:root": [0.9, 0.1],
            "paper-a:method": [0.85, 0.15],
            "paper-a:chunk": [0.95, 0.05],
            "paper-b:root": [0.6, 0.4],
            "paper-b:method": [0.55, 0.45],
            "paper-b:chunk": [0.5, 0.5],
            "paper-c:root": [0.35, 0.65],
            "paper-c:method": [0.3, 0.7],
            "paper-c:chunk": [0.45, 0.55],
            f"{lexical_paper}:root": [0.88, 0.12],
            f"{lexical_paper}:method": [0.86, 0.14],
            f"{lexical_paper}:chunk": [0.9, 0.1],
        }
        self.similarity_queries: list[tuple[Sequence[str] | None, int]] = []
        self.keyword_queries: list[list[str]] = []
        self.truncated_terms: list[str] = []

    @property
    def collection_name(self) -> str:
        return "multi_paper"

    async def replace_paper(
        self,
        document: ParsedPaperDocument,
        nodes: Sequence[TreeIndexNode],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        return None

    async def delete_paper(self, paper_id: str) -> None:
        return None

    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        paper_ids: Sequence[str] | None,
        top_k: int,
        chunks_only: bool,
        node_types: Sequence[TreeNodeType] | None = None,
        parent_ids: Sequence[str] | None = None,
    ) -> list[TreeVectorMatch]:
        self.similarity_queries.append((paper_ids or (), top_k))
        if node_types == [TreeNodeType.ROOT]:
            return [
                TreeVectorMatch(node=self.roots[paper_id], vector_score=score)
                for paper_id, score in (("paper-a", 0.8), ("paper-b", 0.6), ("paper-c", 0.2))
                if not paper_ids or paper_id in paper_ids
            ][:top_k]
        if node_types == [TreeNodeType.SECTION]:
            return [
                TreeVectorMatch(node=self.sections[paper_id], vector_score=0.7)
                for paper_id in (paper_ids or ())
                if paper_id in self.sections
            ][:top_k]
        if parent_ids:
            return [
                TreeVectorMatch(node=self.chunks[paper_id], vector_score=0.75)
                for paper_id in (paper_ids or ())
                if paper_id in self.chunks
            ][:top_k]
        return [
            TreeVectorMatch(node=self.chunks[paper_id], vector_score=score)
            for paper_id, score in (
                ("paper-a", 0.95),
                ("paper-b", 0.50),
                ("paper-c", 0.30),
            )
            if not paper_ids or paper_id in paper_ids
        ][:top_k]

    async def keyword_search(
        self,
        terms: Sequence[str],
        *,
        paper_ids: Sequence[str] | None,
        top_k: int,
        chunks_only: bool = True,
        node_types: Sequence[TreeNodeType] | None = None,
        parent_ids: Sequence[str] | None = None,
    ) -> KeywordSearchResult:
        self.keyword_queries.append(list(terms))
        if "qwen3" not in terms:
            return KeywordSearchResult()
        return KeywordSearchResult(
            matches=[
                TreeKeywordMatch(
                    node=self.chunks[self.LEXICAL_ONLY_PAPER],
                    keyword_score=1.0,
                    matched_terms=["qwen3"],
                )
            ][:top_k],
            pool_size=1,
            truncated_terms=self.truncated_terms,
        )

    async def load_paper_nodes(
        self,
        paper_ids: Sequence[str],
    ) -> list[IndexedTreeNode]:
        return [
            IndexedTreeNode(node=node, embedding=self._embeddings[node.node_id])
            for paper_id in paper_ids
            for node in (self.roots[paper_id], self.sections[paper_id], self.chunks[paper_id])
            if paper_id in self.roots
        ]

    async def load_nodes(
        self,
        *,
        paper_ids: Sequence[str] | None = None,
        node_ids: Sequence[str] | None = None,
        parent_ids: Sequence[str] | None = None,
        node_types: Sequence[TreeNodeType] | None = None,
    ) -> list[IndexedTreeNode]:
        candidates = [
            node
            for paper_id in [*self.paper_ids, self.LEXICAL_ONLY_PAPER]
            for node in (self.roots[paper_id], self.sections[paper_id], self.chunks[paper_id])
        ]
        wanted_ids = set(node_ids or ())
        wanted_parents = set(parent_ids or ())
        wanted_papers = set(paper_ids or ())
        selected = [
            node
            for node in candidates
            if (wanted_ids and node.node_id in wanted_ids)
            or (wanted_parents and (node.parent_id or "") in wanted_parents)
            or (
                wanted_papers
                and node.paper_id in wanted_papers
                and (node_types is None or node.node_type in node_types)
            )
        ]
        return [
            IndexedTreeNode(node=node, embedding=self._embeddings[node.node_id])
            for node in selected
        ]


@pytest.mark.asyncio
async def test_multi_paper_retrieval_returns_every_requested_paper() -> None:
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=_MultiPaperStore())

    report = await retriever.retrieve(
        "Compare the retrieval methods of the indexed papers",
        paper_ids=["paper-a", "paper-b", "paper-c"],
        mode=RetrievalMode.COMPARE,
    )

    assert report.strategy is RetrievalStrategy.MULTI_PAPER
    assert {hit.paper_id for hit in report.hits} == {"paper-a", "paper-b", "paper-c"}
    assert report.missing_paper_ids == []
    assert report.budget is not None
    assert report.budget.final_top_k >= 2 * len(report.paper_ids)
    assert report.budget.max_chunks_per_paper == 2
    assert report.budget.max_papers >= len(report.paper_ids)


@pytest.mark.asyncio
async def test_multi_paper_budget_scales_with_papers_and_dimensions() -> None:
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=_MultiPaperStore())
    papers = ["paper-a", "paper-b", "paper-c"]
    dimensions = ["method", "results", "limitations"]

    report = await retriever.retrieve(
        "Compare method, results and limitations across the indexed papers",
        paper_ids=papers,
        mode=RetrievalMode.COMPARE,
        queries=[
            RetrievalQuery(
                query=f"{dimension} of {paper_id}",
                mode=RetrievalMode.COMPARE,
                paper_ids=[paper_id],
                dimension=dimension,
            )
            for paper_id in papers
            for dimension in dimensions
        ],
    )

    assert report.budget is not None
    assert report.budget.max_chunks_per_paper == len(dimensions)
    assert report.budget.final_top_k >= len(papers) * len(dimensions)
    assert report.coverage is not None
    assert len(report.coverage.cells) == len(papers) * len(dimensions)


@pytest.mark.asyncio
async def test_lexical_channel_recovers_a_chunk_vector_recall_missed() -> None:
    store = _MultiPaperStore()
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=store)

    report = await retriever.retrieve(
        "Which model does the Qwen3-Embedding-0613 swap use?",
        paper_ids=["paper-a", "paper-b", "paper-c"],
        mode=RetrievalMode.COMPARE,
    )

    lexical_paper = _MultiPaperStore.LEXICAL_ONLY_PAPER
    assert lexical_paper not in report.paper_ids
    lexical_hit = next(
        hit for hit in report.hits if hit.paper_id == lexical_paper
    )
    assert lexical_hit.source is RetrievalSource.KEYWORD
    assert lexical_hit.keyword_score == 1.0
    assert lexical_hit.matched_terms == ["qwen3"]
    assert report.keyword_candidate_count == 1
    assert lexical_hit.paper_id in report.candidate_paper_ids
    assert store.keyword_queries


@pytest.mark.asyncio
async def test_per_paper_cap_is_applied_after_reranking_sees_a_wide_pool() -> None:
    reranker = _RecordingReranker()
    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_MultiPaperStore(),
        reranker=reranker,
        multi_paper_chunks_per_paper=2,
    )

    report = await retriever.retrieve(
        "Compare the indexed papers",
        paper_ids=["paper-a", "paper-b", "paper-c"],
        mode=RetrievalMode.COMPARE,
    )

    assert report.reranker_applied is True
    assert report.mmr_candidate_count == 3
    assert reranker.document_count == 3
    assert len(report.hits) == 3
    assert all(hit.rerank_score is not None for hit in report.hits)


@pytest.mark.asyncio
async def test_dimension_sub_questions_produce_a_coverage_matrix() -> None:
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=_MultiPaperStore())

    report = await retriever.retrieve(
        "Compare method and results of paper A and paper C",
        paper_ids=["paper-a", "paper-c"],
        mode=RetrievalMode.COMPARE,
        queries=[
            RetrievalQuery(
                query="hierarchical index construction method",
                mode=RetrievalMode.METHOD,
                paper_ids=["paper-a"],
                dimension="method",
            ),
            RetrievalQuery(
                query="qwen3 embedding model results",
                mode=RetrievalMode.SUMMARY,
                paper_ids=["paper-c"],
                dimension="results",
            ),
        ],
    )

    matrix = report.coverage
    assert matrix is not None
    assert matrix.dimensions == ["method", "results"]
    assert matrix.cell_for("paper-a", "method") is not None
    assert matrix.coverage_ratio == 0.0
    assert ("paper-a", "method") in {
        (cell.paper_id, cell.dimension) for cell in matrix.candidate_cells()
    }
    assert ("paper-c", "results") in {
        (cell.paper_id, cell.dimension) for cell in matrix.candidate_cells()
    }
    assert ("paper-a", "results") in {
        (cell.paper_id, cell.dimension) for cell in matrix.unresolved_cells()
    }
    assert report.queries[0].dimension == "method"
    assert report.missing_paper_ids == []


@pytest.mark.asyncio
async def test_survey_budget_widens_paper_routing_and_final_evidence() -> None:
    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_MultiPaperStore(),
        survey_paper_top_k=8,
        survey_final_top_k=12,
        max_final_top_k=12,
    )

    report = await retriever.retrieve(
        "Survey which indexed papers use a hierarchical index",
        mode=RetrievalMode.SYNTHESIS,
        queries=[
            RetrievalQuery(query="hierarchical index", mode=RetrievalMode.SYNTHESIS),
            RetrievalQuery(query="dataset and metrics", mode=RetrievalMode.SYNTHESIS),
        ],
    )

    assert report.strategy is RetrievalStrategy.CORPUS_SURVEY
    assert report.budget is not None
    assert report.budget.max_papers == 8
    assert report.budget.final_top_k == 12
    assert report.searched_globally is True
    assert set(report.candidate_paper_ids) == {"paper-a", "paper-b", "paper-c"}


@pytest.mark.asyncio
async def test_tight_budget_and_weak_paper_still_gets_evidence() -> None:
    """A named paper whose evidence scores far below the rest must still appear."""

    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_MultiPaperStore(),
        max_candidates=2,
        min_ranking_score=0.75,
        score_window=0.05,
    )

    report = await retriever.retrieve(
        "Compare every indexed paper",
        paper_ids=["paper-a", "paper-b", "paper-c"],
        mode=RetrievalMode.COMPARE,
    )

    assert {hit.paper_id for hit in report.hits} == {"paper-a", "paper-b", "paper-c"}
    assert report.missing_paper_ids == []
    assert report.coverage is not None
    assert report.coverage.candidate_ratio == 1.0


def test_reservation_reaches_every_papers_second_candidate() -> None:
    """The reservation pass must skip taken candidates instead of stopping."""

    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_MultiPaperStore(),
        max_candidates=20,
    )
    ranked: list[tuple[_Candidate, float]] = []
    for paper_id, base in (("paper-a", 0.9), ("paper-b", 0.4)):
        for index in range(3):
            ranked.append(
                (
                    _Candidate(
                        node=_chunk(
                            paper_id,
                            f"chunk:{index}",
                            f"{paper_id} evidence {index}",
                            parent_id=f"{paper_id}:method",
                        ),
                        vector_score=base,
                        source=RetrievalSource.VECTOR,
                    ),
                    base - index * 0.01,
                )
            )
    ranked.sort(key=lambda item: item[1], reverse=True)
    budget = retriever._budget_for(  # noqa: SLF001
        RetrievalStrategy.MULTI_PAPER,
        ["paper-a", "paper-b"],
        [RetrievalQuery(query="compare both", mode=RetrievalMode.COMPARE)],
    )
    budget = budget.model_copy(update={"max_chunks_per_paper": 2})

    reserved, protected = retriever._reserve_candidates(  # noqa: SLF001
        ranked,
        target_paper_ids=["paper-a", "paper-b"],
        coverage_dimensions=[],
        budget=budget,
    )

    node_ids = [item[0].node.node_id for item in reserved]
    assert "paper-a:chunk:0" in node_ids
    assert "paper-a:chunk:1" in node_ids
    assert "paper-b:chunk:0" in node_ids
    assert "paper-b:chunk:1" in node_ids
    assert {"paper-a:chunk:1", "paper-b:chunk:1"} <= protected


@pytest.mark.asyncio
async def test_identical_text_in_two_papers_keeps_both_sources() -> None:
    """Cross-paper deduplication must not delete a paper's only evidence."""

    store = _MultiPaperStore()
    shared = "This shared boilerplate sentence appears in two different papers."
    for paper_id in ("paper-a", "paper-b"):
        original = store.chunks[paper_id]
        store.chunks[paper_id] = original.model_copy(update={"text": shared})
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=store)

    report = await retriever.retrieve(
        "Compare the indexed papers",
        paper_ids=["paper-a", "paper-b"],
        mode=RetrievalMode.COMPARE,
    )

    returned = {hit.paper_id for hit in report.hits}
    assert returned == {"paper-a", "paper-b"}
    assert report.missing_paper_ids == []
    texts = {hit.text for hit in report.hits}
    assert texts == {shared}


class _QueryAwareReranker(TextRerankerGateway):
    """Scores a document high only when the query names its paper."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return "query-aware-cross-encoder"

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, len(documents)))
        return [
            0.9 if query.split()[-1] in document else 0.1 for document in documents
        ]


@pytest.mark.asyncio
async def test_each_sub_question_reranks_its_own_candidates() -> None:
    """Evidence planned for paper-b must not be scored against paper-a's question."""

    reranker = _QueryAwareReranker()
    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_MultiPaperStore(),
        reranker=reranker,
    )

    report = await retriever.retrieve(
        "Compare both papers",
        paper_ids=["paper-a", "paper-b"],
        mode=RetrievalMode.COMPARE,
        queries=[
            RetrievalQuery(
                query="method of paper-a",
                mode=RetrievalMode.METHOD,
                paper_ids=["paper-a"],
                dimension="method",
            ),
            RetrievalQuery(
                query="results of paper-b",
                mode=RetrievalMode.SUMMARY,
                paper_ids=["paper-b"],
                dimension="results",
            ),
        ],
    )

    reranked_queries = {query for query, _ in reranker.calls}
    assert reranked_queries == {"method of paper-a", "results of paper-b"}
    assert len(reranker.calls) == 2
    paper_a_hit = next(hit for hit in report.hits if hit.paper_id == "paper-a")
    paper_b_hit = next(hit for hit in report.hits if hit.paper_id == "paper-b")
    assert paper_a_hit.rerank_query == "method of paper-a"
    assert paper_b_hit.rerank_query == "results of paper-b"
    assert all(hit.rerank_score == 0.9 for hit in report.hits)


@pytest.mark.asyncio
async def test_budget_counts_the_union_of_sub_question_scopes() -> None:
    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_MultiPaperStore(),
        max_final_top_k=64,
    )

    report = await retriever.retrieve(
        "Compare three papers on one dimension",
        mode=RetrievalMode.COMPARE,
        queries=[
            RetrievalQuery(
                query=f"method of {paper_id}",
                mode=RetrievalMode.METHOD,
                paper_ids=[paper_id],
                dimension="method",
            )
            for paper_id in ("paper-a", "paper-b", "paper-c")
        ],
    )

    assert report.strategy is RetrievalStrategy.MULTI_PAPER
    assert report.budget is not None
    # Three scoped papers, although the top-level call named none of them.
    assert report.budget.max_papers >= 3
    assert report.budget.final_top_k >= 6
    assert set(report.coverage.paper_ids if report.coverage else []) == {
        "paper-a",
        "paper-b",
        "paper-c",
    }
