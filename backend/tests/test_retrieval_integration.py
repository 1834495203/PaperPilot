"""End-to-end retrieval against a real Chroma collection with a deterministic embedder.

These tests exercise the fused pipeline (vector channel, keyword channel, tree
expansion, coverage-aware selection) and the evaluation harness on a real store, so
they catch store-level regressions that mocked stores cannot see.
"""

import hashlib
from pathlib import Path

import pytest

from app.application.tree_retrieval import TreeRagRetriever
from app.domain.ports import TextEmbeddingGateway
from app.domain.rag import (
    CoverageStatus,
    PageTextBlock,
    PaperMetadata,
    PaperSection,
    ParsedPaperDocument,
    RetrievalMode,
    RetrievalStrategy,
)
from app.eval.dataset import RetrievalEvalDataset
from app.eval.runner import RetrievalEvaluator
from app.infrastructure.rag.chroma_store import ChromaTreeVectorStore
from app.infrastructure.rag.tree_chunker import TreeRagChunker

VOCABULARY_DIMENSIONS = 64


class _HashingEmbedder(TextEmbeddingGateway):
    """Deterministic term-frequency hashing so equal terms imply equal vectors."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * VOCABULARY_DIMENSIONS
        for token in text.lower().replace(":", " ").replace("-", " ").split():
            token = token.strip(".,()[]")
            if not token:
                continue
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vector[bucket % VOCABULARY_DIMENSIONS] += 1.0
        return vector


def _document(paper_id: str, title: str, paragraphs: list[str]) -> ParsedPaperDocument:
    return ParsedPaperDocument(
        paper_id=paper_id,
        metadata=PaperMetadata(title=title),
        source_path=Path(f"{paper_id}.pdf"),
        page_count=1,
        sections=[
            PaperSection(
                section_id=f"{paper_id}:section:{index + 1:04d}",
                index=str(index + 1),
                title=heading,
                semantic_role="method",
                level=1,
                blocks=[
                    PageTextBlock(
                        block_id=f"{paper_id}:block:{index}",
                        page_number=1,
                        text=text,
                    )
                ],
                page_start=1,
                page_end=1,
            )
            for index, (heading, text) in enumerate(
                zip(
                    ["1 Method", "2 Results"],
                    paragraphs,
                    strict=False,
                )
            )
        ],
    )


async def _seeded_store(tmp_path: Path) -> ChromaTreeVectorStore:
    store = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="integration_tree",
    )
    chunker = TreeRagChunker(max_chunk_chars=400)
    embedder = _HashingEmbedder()
    papers = [
        _document(
            "paper-a",
            "Hierarchical Tree Retrieval",
            [
                "We construct a hierarchical tree index and store ancestor titles "
                "with every chunk for retrieval.",
                "The method improves passage recall on a long-document benchmark.",
            ],
        ),
        _document(
            "paper-b",
            "Dense Retrieval Baselines",
            [
                "Our baseline compares dense retrieval and measures latency on the "
                "BEIR dataset.",
                "We swap the embedding model to Qwen3-Embedding-0613 for the final run.",
            ],
        ),
    ]
    for paper in papers:
        nodes = chunker.chunk(paper)
        embeddings = await embedder.embed_documents([node.embedding_text for node in nodes])
        await store.replace_paper(paper, nodes, embeddings)
    return store


@pytest.mark.asyncio
async def test_multi_paper_comparison_covers_every_requested_paper(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    store = await _seeded_store(tmp_path)
    retriever = TreeRagRetriever(embedder=_HashingEmbedder(), vector_store=store)

    report = await retriever.retrieve(
        "Compare hierarchical tree retrieval with dense retrieval baselines",
        paper_ids=["paper-a", "paper-b"],
        mode=RetrievalMode.COMPARE,
    )

    assert report.strategy is RetrievalStrategy.MULTI_PAPER
    assert {hit.paper_id for hit in report.hits} == {"paper-a", "paper-b"}
    assert report.missing_paper_ids == []
    assert report.coverage is not None
    assert report.coverage.paper_ids == ["paper-a", "paper-b"]
    # Retrieval only produces candidates; verification belongs to an evidence reader.
    assert all(
        cell.status is CoverageStatus.CANDIDATE for cell in report.coverage.cells
    )
    assert report.coverage.coverage_ratio == 0.0
    assert report.coverage.candidate_ratio == 1.0


@pytest.mark.asyncio
async def test_search_result_reports_a_requested_paper_with_no_evidence(
    tmp_path: Path,
) -> None:
    pytest.importorskip("chromadb")
    store = await _seeded_store(tmp_path)
    retriever = TreeRagRetriever(embedder=_HashingEmbedder(), vector_store=store)

    report = await retriever.retrieve(
        "Which papers report a latency measurement on the BEIR dataset?",
        paper_ids=["paper-a", "paper-b", "paper-c"],
        mode=RetrievalMode.COMPARE,
    )

    returned = {hit.paper_id for hit in report.hits}
    assert "paper-b" in returned
    assert "paper-c" in report.missing_paper_ids
    assert report.coverage is not None
    missing_cell = report.coverage.cell_for("paper-c", "content")
    assert missing_cell is not None
    assert missing_cell.status is CoverageStatus.MISSING
    assert missing_cell.evidence_ids == []


@pytest.mark.asyncio
async def test_evaluator_scores_a_real_corpus_comparison(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    store = await _seeded_store(tmp_path)
    retriever = TreeRagRetriever(embedder=_HashingEmbedder(), vector_store=store)
    dataset = RetrievalEvalDataset.model_validate(
        {
            "name": "integration",
            "cases": [
                {
                    "case_id": "compare-two-papers",
                    "question": "Compare the retrieval approach and reported results",
                    "mode": "compare",
                    "paper_ids": ["paper-a", "paper-b"],
                    "expected_strategy": "multi_paper",
                    "expected_paper_ids": ["paper-a", "paper-b"],
                    "expected_dimensions": ["method", "results"],
                    "sub_questions": [
                        {
                            "query": "hierarchical tree index ancestor titles",
                            "dimension": "method",
                            "paper_ids": ["paper-a"],
                        },
                        {
                            "query": "dense retrieval baseline latency BEIR dataset",
                            "dimension": "method",
                            "paper_ids": ["paper-b"],
                        },
                        {
                            "query": "passage recall long document benchmark",
                            "dimension": "results",
                            "paper_ids": ["paper-a"],
                        },
                        {
                            "query": "Qwen3-Embedding-0613 final run",
                            "dimension": "results",
                            "paper_ids": ["paper-b"],
                        },
                    ],
                    "expected_evidence": [
                        {"paper_id": "paper-a", "anchor": "hierarchical tree index"},
                        {"paper_id": "paper-b", "anchor": "BEIR"},
                        {"paper_id": "paper-b", "anchor": "qwen3-embedding-0613"},
                    ],
                }
            ],
        }
    )

    report = await RetrievalEvaluator(retriever).evaluate(dataset)

    assert report.target_paper_recall == 1.0
    assert report.evidence_recall == 1.0
    assert report.strategy_accuracy == 1.0
    assert report.candidate_dimension_coverage == 1.0
    case = report.cases[0]
    assert case.missing_paper_ids == []
    assert case.unmatched_anchors == []
    assert case.unretrieved_cells == []


@pytest.mark.asyncio
async def test_keyword_channel_matches_an_exact_model_version_through_chroma(
    tmp_path: Path,
) -> None:
    pytest.importorskip("chromadb")
    store = await _seeded_store(tmp_path)

    result = await store.keyword_search(
        ["qwen3", "0613"],
        paper_ids=None,
        top_k=5,
    )

    assert result.matches
    assert all(term in result.matches[0].matched_terms for term in ("qwen3", "0613"))
    assert "Qwen3-Embedding-0613" in result.matches[0].node.text
    assert result.matches[0].keyword_score == 1.0
    assert result.truncated_terms == []
