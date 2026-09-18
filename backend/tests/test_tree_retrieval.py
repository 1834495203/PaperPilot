from collections.abc import Sequence

import pytest

from app.application.tree_retrieval import TreeRagRetriever, _Candidate, _Recall
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


def _node(
    node_id: str,
    node_type: TreeNodeType,
    *,
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
    text: str = "",
    paper_id: str = "paper",
) -> TreeIndexNode:
    return TreeIndexNode(
        node_id=node_id,
        paper_id=paper_id,
        node_type=node_type,
        title=node_id,
        parent_id=parent_id,
        children_ids=children_ids or [],
        level=1,
        section_path=["3 Method"],
        text=text,
        embedding_text=text or node_id,
        page_start=3,
        page_end=3,
    )


class _Embedder(TextEmbeddingGateway):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _Reranker(TextRerankerGateway):
    def __init__(self, scores: list[float], *, error: str | None = None) -> None:
        self.scores = scores
        self.error = error

    @property
    def name(self) -> str:
        return "test-cross-encoder"

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if self.error is not None:
            raise RuntimeError(self.error)
        assert query
        assert len(documents) == len(self.scores)
        return self.scores


class _Store(TreeVectorStore):
    def __init__(self) -> None:
        self.chunk_a = _node(
            "chunk-a",
            TreeNodeType.CHUNK,
            parent_id="method",
            text="TreeRAG builds a hierarchical index.",
        )
        self.chunk_b = _node(
            "chunk-b",
            TreeNodeType.CHUNK,
            parent_id="method",
            text="Sibling nodes restore related context.",
        )
        self.method = _node(
            "method",
            TreeNodeType.SECTION,
            parent_id="root",
            children_ids=["chunk-a", "chunk-b"],
        )
        self.root = _node(
            "root",
            TreeNodeType.ROOT,
            children_ids=["method"],
        )
        self.searches: list[
            tuple[Sequence[str] | None, bool, Sequence[TreeNodeType] | None, Sequence[str] | None]
        ] = []
        self.loaded_paper_ids: list[str] = []
        self.loaded_node_ids: list[str] = []
        self.loaded_parent_ids: list[str] = []
        self.keyword_queries: list[list[str]] = []
        self.load_count = 0

    @property
    def collection_name(self) -> str:
        return "test_tree"

    def _nodes(self) -> list[TreeIndexNode]:
        return [self.root, self.method, self.chunk_a, self.chunk_b]

    def _embeddings(self) -> dict[str, list[float]]:
        return {
            "root": [0.1, 0.9],
            "method": [0.8, 0.2],
            "chunk-a": [1.0, 0.0],
            "chunk-b": [0.9, 0.1],
        }

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
        self.searches.append((paper_ids, chunks_only, node_types, parent_ids))
        if node_types == [TreeNodeType.ROOT]:
            return [TreeVectorMatch(node=self.root, vector_score=0.6)]
        if node_types == [TreeNodeType.SECTION]:
            return [TreeVectorMatch(node=self.method, vector_score=0.8)]
        return [TreeVectorMatch(node=self.chunk_a, vector_score=1.0)]

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
        return KeywordSearchResult()

    async def load_paper_nodes(
        self,
        paper_ids: Sequence[str],
    ) -> list[IndexedTreeNode]:
        self.load_count += 1
        self.loaded_paper_ids = list(paper_ids)
        return [
            IndexedTreeNode(node=node, embedding=self._embeddings()[node.node_id])
            for node in self._nodes()
        ]

    async def load_nodes(
        self,
        *,
        paper_ids: Sequence[str] | None = None,
        node_ids: Sequence[str] | None = None,
        parent_ids: Sequence[str] | None = None,
        node_types: Sequence[TreeNodeType] | None = None,
    ) -> list[IndexedTreeNode]:
        self.load_count += 1
        if paper_ids:
            self.loaded_paper_ids = list(paper_ids)
        if node_ids:
            self.loaded_node_ids = list(node_ids)
        if parent_ids:
            self.loaded_parent_ids = list(parent_ids)
        wanted_ids = set(node_ids or ())
        wanted_parents = set(parent_ids or ())
        wanted_papers = set(paper_ids or ())
        selected = [
            node
            for node in self._nodes()
            if (wanted_ids and node.node_id in wanted_ids)
            or (
                wanted_parents
                and (node.parent_id or "") in wanted_parents
                and (node_types is None or node.node_type in node_types)
            )
            or (
                wanted_papers
                and node.paper_id in wanted_papers
                and (node_types is None or node.node_type in node_types)
            )
        ]
        return [
            IndexedTreeNode(node=node, embedding=self._embeddings()[node.node_id])
            for node in selected
        ]


@pytest.mark.asyncio
async def test_fact_mode_returns_direct_chunk_hits_without_tree_expansion() -> None:
    store = _Store()
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=store)

    report = await retriever.retrieve(
        "How is the index built?",
        paper_ids=["paper"],
        mode=RetrievalMode.FACT,
    )

    assert [search[2] for search in store.searches[:2]] == [
        [TreeNodeType.ROOT],
        [TreeNodeType.SECTION],
    ]
    assert store.searches[2][3] == ["method"]
    assert store.load_count == 1
    assert [hit.node_id for hit in report.hits] == ["chunk-a"]
    assert report.hits[0].source is RetrievalSource.VECTOR


@pytest.mark.asyncio
async def test_method_mode_expands_a_leaf_to_related_sibling_chunks() -> None:
    store = _Store()
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=store)

    report = await retriever.retrieve(
        "Explain the retrieval method",
        paper_ids=["paper"],
        mode=RetrievalMode.METHOD,
    )

    assert store.load_count == 2
    assert store.loaded_parent_ids == ["method"]
    assert [hit.node_id for hit in report.hits] == ["chunk-a", "chunk-b"]
    sibling = report.hits[1]
    assert sibling.source is RetrievalSource.TREE_EXPANSION
    assert sibling.expanded_from == "method"
    assert report.expanded_candidate_count == 1
    assert sibling.ranking_score != sibling.vector_score
    assert report.mmr_candidate_count == 2


@pytest.mark.asyncio
async def test_cross_encoder_controls_final_order() -> None:
    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_Store(),
        reranker=_Reranker([0.1, 0.9]),
    )

    report = await retriever.retrieve(
        "Explain the retrieval method",
        paper_ids=["paper"],
        mode=RetrievalMode.METHOD,
    )

    assert [hit.node_id for hit in report.hits] == ["chunk-b", "chunk-a"]
    assert [hit.rerank_score for hit in report.hits] == [0.9, 0.1]
    assert report.reranker_applied is True
    assert report.reranker_name == "test-cross-encoder"


@pytest.mark.asyncio
async def test_reranker_failure_falls_back_and_is_observable() -> None:
    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_Store(),
        reranker=_Reranker([], error="model unavailable"),
    )

    report = await retriever.retrieve(
        "Explain the retrieval method",
        paper_ids=["paper"],
        mode=RetrievalMode.METHOD,
    )

    assert [hit.node_id for hit in report.hits] == ["chunk-a", "chunk-b"]
    assert report.reranker_applied is False
    assert report.reranker_error == "model unavailable"


@pytest.mark.asyncio
async def test_normalized_duplicate_chunks_are_removed_before_mmr() -> None:
    store = _Store()
    store.chunk_b = _node(
        "chunk-b",
        TreeNodeType.CHUNK,
        parent_id="method",
        text="  treerag BUILDS a hierarchical index.  ",
    )
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=store)

    report = await retriever.retrieve(
        "Explain the retrieval method",
        paper_ids=["paper"],
        mode=RetrievalMode.METHOD,
    )

    assert [hit.node_id for hit in report.hits] == ["chunk-a"]
    assert report.deduplicated_candidate_count == 1


def test_mmr_prefers_a_diverse_chunk_over_a_near_duplicate() -> None:
    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_Store(),
        mmr_top_k=2,
        mmr_lambda=0.5,
    )
    candidates = [
        (
            _Candidate(
                node=_node("a", TreeNodeType.CHUNK, text="top result"),
                vector_score=1.0,
                source=RetrievalSource.VECTOR,
                embedding=[1.0, 0.0],
            ),
            1.0,
        ),
        (
            _Candidate(
                node=_node("b", TreeNodeType.CHUNK, text="near duplicate"),
                vector_score=0.99,
                source=RetrievalSource.VECTOR,
                embedding=[0.999, 0.001],
            ),
            0.99,
        ),
        (
            _Candidate(
                node=_node("c", TreeNodeType.CHUNK, text="different evidence"),
                vector_score=0.8,
                source=RetrievalSource.VECTOR,
                embedding=[0.0, 1.0],
            ),
            0.8,
        ),
    ]

    selected = retriever._mmr_select(candidates)  # noqa: SLF001

    assert [item[0].node.node_id for item in selected] == ["a", "c"]


@pytest.mark.asyncio
async def test_global_retrieval_expands_only_papers_found_by_initial_recall() -> None:
    store = _Store()
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=store)

    report = await retriever.retrieve(
        "How does hierarchical retrieval work?",
        mode=RetrievalMode.METHOD,
    )

    assert store.searches[0][0] is None
    assert store.loaded_paper_ids == []
    assert store.loaded_node_ids == ["chunk-a", "chunk-b", "method", "root"]
    assert store.loaded_parent_ids == ["method"]
    assert report.searched_globally is True
    assert report.paper_ids == []
    assert report.candidate_paper_ids == ["paper"]


class _ParentAwareStore(TreeVectorStore):
    def __init__(self) -> None:
        self.good_root = _node(
            "good-root",
            TreeNodeType.ROOT,
            children_ids=["good-section"],
            paper_id="good-paper",
        )
        self.good_section = _node(
            "good-section",
            TreeNodeType.SECTION,
            parent_id="good-root",
            children_ids=["good-chunk"],
            paper_id="good-paper",
        )
        self.good_chunk = _node(
            "good-chunk",
            TreeNodeType.CHUNK,
            parent_id="good-section",
            text="PaperQA handles noisy PDF parsing with quality-aware evidence.",
            paper_id="good-paper",
        )
        self.bad_root = _node(
            "bad-root",
            TreeNodeType.ROOT,
            children_ids=["bad-section"],
            paper_id="bad-paper",
        )
        self.bad_section = _node(
            "bad-section",
            TreeNodeType.SECTION,
            parent_id="bad-root",
            children_ids=["bad-chunk"],
            paper_id="bad-paper",
        )
        self.bad_chunk = _node(
            "bad-chunk",
            TreeNodeType.CHUNK,
            parent_id="bad-section",
            text="The underlying model version is GPT-4-0613.",
            paper_id="bad-paper",
        )

    @property
    def collection_name(self) -> str:
        return "parent_aware"

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
        if node_types == [TreeNodeType.ROOT]:
            return [TreeVectorMatch(node=self.good_root, vector_score=0.90)]
        if node_types == [TreeNodeType.SECTION]:
            return [TreeVectorMatch(node=self.good_section, vector_score=0.92)]
        if parent_ids:
            return [TreeVectorMatch(node=self.good_chunk, vector_score=0.78)]
        return [TreeVectorMatch(node=self.bad_chunk, vector_score=0.98)]

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
        return KeywordSearchResult()

    async def load_paper_nodes(
        self,
        paper_ids: Sequence[str],
    ) -> list[IndexedTreeNode]:
        return [
            IndexedTreeNode(node=self.good_root, embedding=[0.9, 0.1]),
            IndexedTreeNode(node=self.good_section, embedding=[0.95, 0.05]),
            IndexedTreeNode(node=self.good_chunk, embedding=[0.78, 0.22]),
            IndexedTreeNode(node=self.bad_root, embedding=[0.0, 1.0]),
            IndexedTreeNode(node=self.bad_section, embedding=[0.0, 1.0]),
            IndexedTreeNode(node=self.bad_chunk, embedding=[0.98, 0.02]),
        ]

    async def load_nodes(
        self,
        *,
        paper_ids: Sequence[str] | None = None,
        node_ids: Sequence[str] | None = None,
        parent_ids: Sequence[str] | None = None,
        node_types: Sequence[TreeNodeType] | None = None,
    ) -> list[IndexedTreeNode]:
        embeddings = {
            "good-root": [0.9, 0.1],
            "good-section": [0.95, 0.05],
            "good-chunk": [0.78, 0.22],
            "bad-root": [0.0, 1.0],
            "bad-section": [0.0, 1.0],
            "bad-chunk": [0.98, 0.02],
        }
        nodes = [
            self.good_root,
            self.good_section,
            self.good_chunk,
            self.bad_root,
            self.bad_section,
            self.bad_chunk,
        ]
        wanted_ids = set(node_ids or ())
        wanted_parents = set(parent_ids or ())
        wanted_papers = set(paper_ids or ())
        selected = [
            node
            for node in nodes
            if (wanted_ids and node.node_id in wanted_ids)
            or (wanted_parents and (node.parent_id or "") in wanted_parents)
            or (
                wanted_papers
                and node.paper_id in wanted_papers
                and (node_types is None or node.node_type in node_types)
            )
        ]
        return [
            IndexedTreeNode(node=node, embedding=embeddings[node.node_id])
            for node in selected
        ]


@pytest.mark.asyncio
async def test_parent_context_demotes_a_generic_global_chunk() -> None:
    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_ParentAwareStore(),
        score_window=1.0,
    )

    report = await retriever.retrieve(
        "How does PaperQA handle PDF parsing quality?",
        mode=RetrievalMode.FACT,
    )

    assert [hit.node_id for hit in report.hits] == ["good-chunk", "bad-chunk"]
    assert report.hits[0].ranking_score > report.hits[1].ranking_score


@pytest.mark.asyncio
async def test_retrieval_does_not_fill_top_k_below_quality_threshold() -> None:
    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_ParentAwareStore(),
        min_ranking_score=0.95,
    )

    report = await retriever.retrieve("unrelated question", mode=RetrievalMode.FACT)

    assert report.hits == []


def test_fused_score_changes_the_pre_rank() -> None:
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=_Store())
    node = _node("chunk-a", TreeNodeType.CHUNK, text="TreeRAG builds a hierarchical index.")

    weak = _Candidate(
        node=node,
        vector_score=0.60,
        source=RetrievalSource.VECTOR,
        fused_score=0.01,
    )
    strong = _Candidate(
        node=node,
        vector_score=0.60,
        source=RetrievalSource.VECTOR,
        fused_score=1.0,
    )

    weak_score = retriever._ranking_score(  # noqa: SLF001
        weak, fallback_query="How is the index built?"
    )
    strong_score = retriever._ranking_score(  # noqa: SLF001
        strong, fallback_query="How is the index built?"
    )

    assert strong_score > weak_score
    assert strong_score - weak_score == pytest.approx(0.5 * 0.20 * 0.99, abs=1e-6)


def test_fused_score_is_not_double_counted_within_one_sub_question() -> None:
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=_Store())
    both = _node("chunk-both", TreeNodeType.CHUNK, text="shared chunk")
    vector_only = _node("chunk-vector", TreeNodeType.CHUNK, text="vector only")
    query = RetrievalQuery(query="shared query", mode=RetrievalMode.METHOD)

    candidates = retriever._merge_channels(  # noqa: SLF001
        [
            _Recall(
                query=query,
                embedding=[1.0, 0.0],
                routed_paper_ids=("paper",),
                roots=(),
                sections=(),
                vector_matches=(
                    TreeVectorMatch(node=both, vector_score=0.9),
                    TreeVectorMatch(node=vector_only, vector_score=0.8),
                ),
                keyword_matches=(
                    TreeKeywordMatch(node=both, keyword_score=0.9, matched_terms=["shared"]),
                ),
            )
        ]
    )

    assert candidates["chunk-both"].fused_score == pytest.approx(1.0)
    assert candidates["chunk-both"].source is RetrievalSource.VECTOR
    assert candidates["chunk-both"].keyword_score == pytest.approx(0.9)
    expected = (1 / 62) / (2 / 61)
    assert candidates["chunk-vector"].fused_score == pytest.approx(expected)


def test_rrf_weighted_candidates_accumulate_across_sub_questions() -> None:
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=_Store())
    repeated = _node("chunk-repeated", TreeNodeType.CHUNK, text="found by both questions")
    single = _node("chunk-single", TreeNodeType.CHUNK, text="found by one question")

    def recall(query: str, matches: tuple[TreeVectorMatch, ...]) -> _Recall:
        return _Recall(
            query=RetrievalQuery(query=query, mode=RetrievalMode.COMPARE),
            embedding=[1.0, 0.0],
            routed_paper_ids=("paper",),
            roots=(),
            sections=(),
            vector_matches=matches,
            keyword_matches=(),
        )

    candidates = retriever._merge_channels(  # noqa: SLF001
        [
            recall(
                "first question",
                (
                    TreeVectorMatch(node=repeated, vector_score=0.9),
                    TreeVectorMatch(node=single, vector_score=0.9),
                ),
            ),
            recall(
                "second question",
                (TreeVectorMatch(node=repeated, vector_score=0.9),),
            ),
        ]
    )

    assert candidates["chunk-repeated"].fused_score > candidates["chunk-single"].fused_score


def test_named_paper_survives_the_global_candidate_cut() -> None:
    retriever = TreeRagRetriever(
        embedder=_Embedder(),
        vector_store=_Store(),
        max_candidates=5,
    )
    ranked: list[tuple[_Candidate, float]] = [
        (
            _Candidate(
                node=_node(f"paper-a:chunk:{index}", TreeNodeType.CHUNK, paper_id="paper-a"),
                vector_score=0.9,
                source=RetrievalSource.VECTOR,
            ),
            0.9 - index * 0.01,
        )
        for index in range(20)
    ]
    ranked.append(
        (
            _Candidate(
                node=_node("paper-b:chunk:1", TreeNodeType.CHUNK, paper_id="paper-b"),
                vector_score=0.3,
                source=RetrievalSource.VECTOR,
            ),
            0.30,
        )
    )
    budget = retriever._budget_for(  # noqa: SLF001
        RetrievalStrategy.MULTI_PAPER,
        ["paper-a", "paper-b"],
        [RetrievalQuery(query="compare both", mode=RetrievalMode.COMPARE)],
    )

    reserved, protected = retriever._reserve_candidates(  # noqa: SLF001
        ranked,
        target_paper_ids=["paper-a", "paper-b"],
        coverage_dimensions=[],
        budget=budget,
    )

    assert "paper-b:chunk:1" in protected

    node_ids = [item[0].node.node_id for item in reserved]
    assert node_ids[0] == "paper-a:chunk:0"
    assert len(reserved) >= 5
    assert "paper-b:chunk:1" in node_ids
    assert node_ids[-1] == "paper-b:chunk:1"
    assert all(
        reserved[index][1] >= reserved[index + 1][1] for index in range(len(reserved) - 1)
    )
