from app.domain.rag import Evidence
from app.infrastructure.agent.supervisor.writer_agent import WriterAgentNode


def test_extract_citations_maps_markers_to_evidence() -> None:
    evidence = Evidence(
        evidence_id="E-1234567890ab",
        paper_id="paper-x",
        paper_title="A Test Paper",
        chunk_id="paper-x:chunk:1",
        section_path=["1 Method"],
        page_start=4,
        page_end=4,
        raw_text="self-attention improves parallelization",
        evidence_text="self-attention improves parallelization",
        retrieval_score=0.9,
        spans=[],
    )

    citations = WriterAgentNode._extract_citations(  # noqa: SLF001
        "Attention improves parallelization. [E-1234567890ab]",
        {"E-1234567890ab": evidence},
    )

    assert len(citations) == 1
    assert citations[0]["paper_id"] == "paper-x"
    assert citations[0]["paper_title"] == "A Test Paper"
    assert citations[0]["page_start"] == 4
    assert citations[0]["excerpt"] == "self-attention improves parallelization"


def test_extract_citations_ignores_unknown_ids_and_deduplicates() -> None:
    evidence = Evidence(
        evidence_id="E-1234567890ab",
        paper_id="paper-x",
        paper_title="A Test Paper",
        chunk_id="paper-x:chunk:1",
        section_path=[],
        raw_text="x",
        evidence_text="x",
        retrieval_score=0.9,
        spans=[],
    )
    text = "A [E-1234567890ab] B [E-1234567890ab] C [E-deadbeef0000]"
    citations = WriterAgentNode._extract_citations(  # noqa: SLF001
        text,
        {"E-1234567890ab": evidence},
    )

    assert len(citations) == 1
    assert citations[0]["evidence_id"] == "E-1234567890ab"
