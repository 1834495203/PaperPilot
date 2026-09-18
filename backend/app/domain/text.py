"""Shared lexical tokenization for the vector and keyword retrieval channels.

Two tokenizers exist on purpose. ``lexical_terms`` feeds overlap scoring and keeps
single CJK characters, while ``keyword_terms`` produces substring search terms
used as Chroma ``$contains`` filters, where a single CJK character would match
almost every document.
"""

import re

_LATIN_TOKEN = re.compile(r"[A-Za-z0-9]+")
_CJK_CHARACTER = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_MINIMUM_TERM_LENGTH = 2

_ASCII_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "all",
        "also",
        "and",
        "any",
        "are",
        "because",
        "been",
        "before",
        "between",
        "both",
        "but",
        "can",
        "could",
        "did",
        "does",
        "doing",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "into",
        "its",
        "more",
        "most",
        "not",
        "only",
        "other",
        "our",
        "out",
        "over",
        "same",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "use",
        "used",
        "using",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
    }
)


def lexical_terms(text: str) -> set[str]:
    """Case-folded latin terms and single CJK characters used for overlap scoring."""

    latin = {token.lower() for token in _LATIN_TOKEN.findall(text) if len(token) > 1}
    return latin | set(_CJK_CHARACTER.findall(text))


def keyword_terms(text: str, *, max_terms: int | None = None) -> list[str]:
    """Ordered unique substring terms for an independent lexical recall channel.

    Latin tokens keep model names, dataset names and metric names such as
    ``qwen3`` or ``0613``. CJK runs become character bigrams so that a
    ``$contains`` filter stays selective. ``max_terms`` bounds query terms; use
    :func:`keyword_tokens` when scoring documents, because term frequencies need
    every occurrence rather than a unique list.
    """

    return _unique(keyword_tokens(text), max_terms=max_terms)


def keyword_tokens(text: str) -> list[str]:
    """Document tokens with repetition, so BM25 can see real term frequencies."""

    ordered: list[str] = []
    for token in _LATIN_TOKEN.findall(text):
        term = token.lower()
        if len(term) < _MINIMUM_TERM_LENGTH or term in _ASCII_STOPWORDS:
            continue
        ordered.append(term)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            ordered.append(run)
            continue
        ordered.extend(run[index : index + 2] for index in range(len(run) - 1))
    return ordered


def _unique(terms: list[str], *, max_terms: int | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        ordered.append(term)
        if max_terms is not None and len(ordered) >= max_terms:
            break
    return ordered
