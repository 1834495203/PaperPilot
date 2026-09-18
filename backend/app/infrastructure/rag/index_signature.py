"""Build the index fingerprint that ties stored vectors to their producers."""

from app.domain.rag import IndexSignature
from app.infrastructure.rag.pdf_parser import PypdfScientificPaperParser
from app.infrastructure.rag.tree_chunker import TreeRagChunker


def build_index_signature(
    *,
    embedding_model: str,
    embedding_dimensions: int | None,
) -> IndexSignature:
    """Identify the embedding model and code versions used to build an index.

    Stored manifests keep this fingerprint, so changing the embedding model,
    parser or chunker marks existing papers as needing a rebuild instead of
    silently mixing incompatible vectors.
    """

    return IndexSignature(
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        chunker_version=TreeRagChunker.VERSION,
        parser_version=PypdfScientificPaperParser.VERSION,
    )
