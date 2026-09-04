import pytest
from pydantic import ValidationError

from app.api.schemas import RetrievePaperRequest, SendMessageRequest


def test_chat_request_uses_the_indexed_corpus_without_paper_selection() -> None:
    request = SendMessageRequest.model_validate({"content": "How does PaperQA parse PDFs?"})

    assert request.content == "How does PaperQA parse PDFs?"
    with pytest.raises(ValidationError):
        SendMessageRequest.model_validate(
            {"content": "legacy request", "paper_ids": ["paper-1"]}
        )


def test_retrieval_request_defaults_to_global_scope() -> None:
    request = RetrievePaperRequest.model_validate({"query": "hierarchical retrieval"})

    assert request.paper_ids is None
