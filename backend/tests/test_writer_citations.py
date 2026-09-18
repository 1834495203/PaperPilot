from app.domain.rag import Evidence
from app.infrastructure.agent.supervisor.writer_agent import WriterAgentNode


def _evidence() -> Evidence:
    return Evidence(
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


def test_extract_citations_maps_markers_to_evidence() -> None:
    citations, unknown = WriterAgentNode._extract_citations(  # noqa: SLF001
        "Attention improves parallelization. [E-1234567890ab]",
        {"E-1234567890ab": _evidence()},
    )

    assert unknown == []
    assert len(citations) == 1
    assert citations[0]["paper_id"] == "paper-x"
    assert citations[0]["paper_title"] == "A Test Paper"
    assert citations[0]["page_start"] == 4
    assert citations[0]["excerpt"] == "self-attention improves parallelization"


def test_extract_citations_reports_unknown_ids_instead_of_dropping_them() -> None:
    """A citation that cannot be resolved must stay visible to the reader."""

    text = "A [E-1234567890ab] B [E-1234567890ab] C [E-deadbeef0000]"
    citations, unknown = WriterAgentNode._extract_citations(  # noqa: SLF001
        text,
        {"E-1234567890ab": _evidence()},
    )

    assert [item["evidence_id"] for item in citations] == ["E-1234567890ab"]
    assert unknown == ["E-deadbeef0000"]


def test_unknown_citation_issues_describe_the_problem() -> None:
    issues = WriterAgentNode._unknown_citation_issues(["E-deadbeef0000"])  # noqa: SLF001

    assert len(issues) == 1
    assert issues[0].evidence_id == "E-deadbeef0000"
    assert issues[0].kind == "unknown_evidence_id"
    assert "no supplied artifact" in issues[0].detail
