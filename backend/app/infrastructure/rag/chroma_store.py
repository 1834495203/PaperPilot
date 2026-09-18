import asyncio
import hashlib
import importlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from app.domain.ports import TreeVectorStore
from app.domain.rag import (
    EvidenceSpan,
    IndexedTreeNode,
    IndexSignature,
    KeywordSearchResult,
    PaperBlockType,
    ParsedPaperDocument,
    TreeIndexNode,
    TreeKeywordMatch,
    TreeNodeType,
    TreeVectorMatch,
)
from app.domain.text import keyword_tokens


class IndexSignatureMismatchError(RuntimeError):
    """Raised when the configured models disagree with the stored collection."""


class ChromaTreeVectorStore(TreeVectorStore):
    """Persist TreeRAG nodes and caller-supplied embeddings in a Chroma collection.

    Nodes keep the model and code versions that produced their embedding, and the
    collection records the index fingerprint. A mismatch is enforced, not just
    reported: reading or writing through a collection built by another embedding
    model would mix incompatible vector spaces, so it fails with a clear message
    until the paper is re-ingested into a fresh collection.
    """

    SCHEMA_VERSION = 4

    def __init__(
        self,
        *,
        persist_directory: Path,
        collection_name: str,
        keyword_filter_limit: int = 400,
        index_signature: IndexSignature | None = None,
        enforce_index_signature: bool = True,
    ) -> None:
        if len(collection_name) < 3:
            raise ValueError("Chroma collection_name must contain at least three characters")
        if keyword_filter_limit < 1:
            raise ValueError("keyword_filter_limit must be positive")
        try:
            chromadb = importlib.import_module("chromadb")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Chroma is not installed; install backend dependencies before ingestion"
            ) from error
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata=self._collection_metadata(index_signature),
        )
        self._collection_name = collection_name
        self._keyword_filter_limit = keyword_filter_limit
        self._index_signature = index_signature
        self._enforce_index_signature = enforce_index_signature
        stored_signature = str((self._collection.metadata or {}).get("index_signature", ""))
        self._stored_index_signature = stored_signature or None

    @staticmethod
    def _collection_metadata(
        index_signature: IndexSignature | None,
    ) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "description": "PaperPilot TreeRAG paper and chunk index",
            "hnsw:space": "cosine",
            "schema_version": ChromaTreeVectorStore.SCHEMA_VERSION,
        }
        if index_signature is not None:
            metadata["index_signature"] = index_signature.fingerprint
            metadata["embedding_model"] = index_signature.embedding_model
        return metadata

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def index_signature(self) -> IndexSignature | None:
        return self._index_signature

    @property
    def stored_index_signature(self) -> str | None:
        return self._stored_index_signature

    @property
    def index_signature_matches(self) -> bool:
        """False when the current models disagree with the stored collection."""

        if self._index_signature is None or self._stored_index_signature is None:
            return True
        return self._stored_index_signature == self._index_signature.fingerprint

    def signature_fingerprint(self, *, action: str) -> str:
        """Return the stored fingerprint or refuse the operation on a mismatch."""

        if not self._enforce_index_signature or self.index_signature_matches:
            return self._stored_index_signature or ""
        current = (
            self._index_signature.fingerprint if self._index_signature is not None else "unknown"
        )
        raise IndexSignatureMismatchError(
            f"{action} refused: collection '{self._collection_name}' was built with index "
            f"signature {self._stored_index_signature}, but the configured embedding model "
            f"and code versions produce {current}. Re-ingest the papers into a new "
            "VECTOR_COLLECTION, or set VECTOR_ENFORCE_INDEX_SIGNATURE=false to bypass this "
            "check."
        )

    async def replace_paper(
        self,
        document: ParsedPaperDocument,
        nodes: Sequence[TreeIndexNode],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.signature_fingerprint(action="Index write")
        await asyncio.to_thread(self._replace_paper_sync, document, nodes, embeddings)

    async def delete_paper(self, paper_id: str) -> None:
        await asyncio.to_thread(self._collection.delete, where={"paper_id": paper_id})

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
        self.signature_fingerprint(action="Vector retrieval")
        return await asyncio.to_thread(
            self._similarity_search_sync,
            query_embedding,
            paper_ids,
            top_k,
            chunks_only,
            node_types,
            parent_ids,
        )

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
        return await asyncio.to_thread(
            self._keyword_search_sync,
            list(terms),
            paper_ids,
            top_k,
            chunks_only,
            node_types,
            parent_ids,
        )

    async def load_paper_nodes(
        self,
        paper_ids: Sequence[str],
    ) -> list[IndexedTreeNode]:
        if not paper_ids:
            raise ValueError("load_paper_nodes requires at least one paper id")
        self.signature_fingerprint(action="Stored embedding read")
        return await asyncio.to_thread(self._load_paper_nodes_sync, paper_ids)

    async def load_nodes(
        self,
        *,
        paper_ids: Sequence[str] | None = None,
        node_ids: Sequence[str] | None = None,
        parent_ids: Sequence[str] | None = None,
        node_types: Sequence[TreeNodeType] | None = None,
    ) -> list[IndexedTreeNode]:
        self.signature_fingerprint(action="Stored embedding read")
        return await asyncio.to_thread(
            self._load_nodes_sync,
            paper_ids,
            node_ids,
            parent_ids,
            node_types,
        )

    def _replace_paper_sync(
        self,
        document: ParsedPaperDocument,
        nodes: Sequence[TreeIndexNode],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not nodes:
            raise ValueError("Cannot index a paper without tree nodes")
        if len(nodes) != len(embeddings):
            raise ValueError("Each tree node requires exactly one embedding")
        dimensions = {len(vector) for vector in embeddings}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("All embeddings must have one consistent non-zero dimension")

        existing = cast(
            dict[str, Any],
            self._collection.get(
                where={"paper_id": document.paper_id},
                include=[],
            ),
        )
        existing_ids = {str(item) for item in existing.get("ids", [])}
        new_ids = {node.node_id for node in nodes}
        max_batch_size = min(int(self._client.get_max_batch_size()), 500)
        for start in range(0, len(nodes), max_batch_size):
            node_batch = list(nodes[start : start + max_batch_size])
            embedding_batch = embeddings[start : start + max_batch_size]
            self._collection.upsert(
                ids=[node.node_id for node in node_batch],
                embeddings=[list(vector) for vector in embedding_batch],
                documents=[node.text for node in node_batch],
                metadatas=[self._metadata(document, node) for node in node_batch],
            )
        stale_ids = sorted(existing_ids - new_ids)
        if stale_ids:
            self._collection.delete(ids=stale_ids)

    def _similarity_search_sync(
        self,
        query_embedding: Sequence[float],
        paper_ids: Sequence[str] | None,
        top_k: int,
        chunks_only: bool,
        node_types: Sequence[TreeNodeType] | None,
        parent_ids: Sequence[str] | None,
    ) -> list[TreeVectorMatch]:
        if top_k < 1:
            return []
        where = self._where_filter(
            paper_ids,
            chunks_only=chunks_only,
            node_types=node_types,
            parent_ids=parent_ids,
        )
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [list(query_embedding)],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where
        raw = cast(
            dict[str, Any],
            self._collection.query(**query_kwargs),
        )
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        return [
            TreeVectorMatch(
                node=self._node_from_record(
                    node_id=str(node_id),
                    document=str(document or ""),
                    metadata=cast(dict[str, Any], metadata),
                ),
                vector_score=1.0 - float(distance),
            )
            for node_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]

    def _keyword_search_sync(
        self,
        terms: list[str],
        paper_ids: Sequence[str] | None,
        top_k: int,
        chunks_only: bool,
        node_types: Sequence[TreeNodeType] | None,
        parent_ids: Sequence[str] | None,
    ) -> KeywordSearchResult:
        """Recall by substring terms, then rank the pool with BM25.

        The channel is independent of vector recall, so a chunk that the
        embedding model never surfaced can still enter the candidate pool. Two
        Two bounds apply and both are reported instead of silently widening: each
        term fetches at most ``keyword_filter_limit`` records, and term statistics
        are computed over the resulting pool rather than the whole corpus. A term
        common enough to exhaust its fetch budget is listed in ``truncated_terms``,
        so a library that outgrows the limit is visible instead of quietly losing
        recall. That bound is why a real full-text index remains the next step for
        large libraries.
        """

        if top_k < 1 or not terms:
            return KeywordSearchResult()
        where = self._where_filter(
            paper_ids,
            chunks_only=chunks_only,
            node_types=node_types,
            parent_ids=parent_ids,
        )
        fetch_limit = self._keyword_filter_limit
        pool: dict[str, tuple[str, dict[str, Any]]] = {}
        truncated_terms: list[str] = []
        for term in terms:
            get_kwargs: dict[str, Any] = {
                "where_document": {"$contains": term},
                "include": ["documents", "metadatas"],
                "limit": fetch_limit + 1,
            }
            if where is not None:
                get_kwargs["where"] = where
            raw = cast(dict[str, Any], self._collection.get(**get_kwargs))
            ids = raw.get("ids", [])
            documents = raw.get("documents", [])
            metadatas = raw.get("metadatas", [])
            if len(ids) > fetch_limit:
                ids = ids[:fetch_limit]
                documents = documents[:fetch_limit]
                metadatas = metadatas[:fetch_limit]
                truncated_terms.append(term)
            for node_id, document, metadata in zip(
                ids, documents, metadatas, strict=True
            ):
                pool.setdefault(
                    str(node_id),
                    (str(document or ""), cast(dict[str, Any], metadata)),
                )
        scored = self._rank_pool(terms, pool)
        return KeywordSearchResult(
            matches=[
                TreeKeywordMatch(
                    node=self._node_from_record(
                        node_id=node_id,
                        document=pool[node_id][0],
                        metadata=pool[node_id][1],
                    ),
                    keyword_score=score,
                    matched_terms=matched,
                )
                for node_id, score, matched in scored[:top_k]
            ],
            pool_size=len(pool),
            truncated_terms=truncated_terms,
        )

    @staticmethod
    def _rank_pool(
        terms: list[str],
        pool: dict[str, tuple[str, dict[str, Any]]],
    ) -> list[tuple[str, float, list[str]]]:
        """BM25 over the fetched pool, with real term frequencies per document."""

        if not pool:
            return []
        tokens = {node_id: keyword_tokens(text) for node_id, (text, _) in pool.items()}
        document_count = len(tokens)
        average_length = (
            sum(len(values) for values in tokens.values()) / document_count
        ) or 1.0
        document_frequency: Counter[str] = Counter()
        for values in tokens.values():
            document_frequency.update(set(values))
        k1 = 1.2
        b = 0.75
        raw_scores: list[tuple[str, float, list[str]]] = []
        for node_id, values in tokens.items():
            frequencies = Counter(values)
            length = len(values) or 1
            score = 0.0
            matched: list[str] = []
            for term in terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                matched.append(term)
                inverse_frequency = math.log(
                    1
                    + (document_count - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                score += (
                    inverse_frequency
                    * frequency
                    * (k1 + 1)
                    / (frequency + k1 * (1 - b + b * length / average_length))
                )
            if score > 0:
                raw_scores.append((node_id, score, matched))
        if not raw_scores:
            return []
        best = max(score for _, score, _ in raw_scores)
        normalized = [
            (node_id, score / best if best > 0 else 0.0, matched)
            for node_id, score, matched in raw_scores
        ]
        normalized.sort(key=lambda item: item[1], reverse=True)
        return normalized

    def _load_paper_nodes_sync(
        self,
        paper_ids: Sequence[str],
    ) -> list[IndexedTreeNode]:
        return self._load_nodes_sync(paper_ids, None, None, None)

    def _load_nodes_sync(
        self,
        paper_ids: Sequence[str] | None,
        node_ids: Sequence[str] | None,
        parent_ids: Sequence[str] | None,
        node_types: Sequence[TreeNodeType] | None,
    ) -> list[IndexedTreeNode]:
        where = self._where_filter(
            paper_ids,
            chunks_only=False,
            node_types=node_types,
            parent_ids=parent_ids,
        )
        if not node_ids and where is None:
            raise ValueError("load_nodes requires a paper, node, parent, or node-type filter")
        get_kwargs: dict[str, Any] = {
            "include": ["documents", "metadatas", "embeddings"]
        }
        if node_ids:
            get_kwargs["ids"] = [str(node_id) for node_id in node_ids]
        if where is not None:
            get_kwargs["where"] = where
        raw = cast(
            dict[str, Any],
            self._collection.get(**get_kwargs),
        )
        ids = raw.get("ids", [])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])
        embeddings = raw.get("embeddings")
        if embeddings is None:
            raise RuntimeError("Chroma did not return stored node embeddings")
        return [
            IndexedTreeNode(
                node=self._node_from_record(
                    node_id=str(node_id),
                    document=str(document or ""),
                    metadata=cast(dict[str, Any], metadata),
                ),
                embedding=[float(value) for value in embedding],
            )
            for node_id, document, metadata, embedding in zip(
                ids,
                documents,
                metadatas,
                embeddings,
                strict=True,
            )
        ]

    @staticmethod
    def _where_filter(
        paper_ids: Sequence[str] | None,
        *,
        chunks_only: bool,
        node_types: Sequence[TreeNodeType] | None = None,
        parent_ids: Sequence[str] | None = None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if paper_ids:
            clauses.append(
                {"paper_id": str(paper_ids[0])}
                if len(paper_ids) == 1
                else {"paper_id": {"$in": [str(item) for item in paper_ids]}}
            )
        effective_types = [TreeNodeType.CHUNK] if chunks_only else list(node_types or [])
        if effective_types:
            clauses.append(
                {"node_type": effective_types[0].value}
                if len(effective_types) == 1
                else {"node_type": {"$in": [item.value for item in effective_types]}}
            )
        if parent_ids:
            clauses.append(
                {"parent_id": str(parent_ids[0])}
                if len(parent_ids) == 1
                else {"parent_id": {"$in": [str(item) for item in parent_ids]}}
            )
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _node_from_record(
        *,
        node_id: str,
        document: str,
        metadata: dict[str, Any],
    ) -> TreeIndexNode:
        page_start = int(metadata.get("page_start", 0)) or None
        page_end = int(metadata.get("page_end", 0)) or None
        section_path = json.loads(str(metadata.get("section_path", "[]")))
        children_ids = json.loads(str(metadata.get("children_ids", "[]")))
        block_types = json.loads(str(metadata.get("block_types", "[]")))
        object_labels = json.loads(str(metadata.get("object_labels", "[]")))
        prefix = str(metadata.get("embedding_prefix", ""))
        figure_asset = str(metadata.get("figure_asset", "")) or None
        figure_caption = str(metadata.get("figure_caption", "")) or None
        raw_asset_ref = str(metadata.get("raw_asset_ref", "")) or None
        raw_table_rows = metadata.get("table_rows", "")
        table_rows = (
            json.loads(str(raw_table_rows)) if raw_table_rows else None
        )
        raw_spans = metadata.get("spans", "")
        spans = (
            [EvidenceSpan(**item) for item in json.loads(str(raw_spans))]
            if raw_spans
            else []
        )
        return TreeIndexNode(
            node_id=node_id,
            paper_id=str(metadata["paper_id"]),
            node_type=TreeNodeType(str(metadata["node_type"])),
            title=str(metadata.get("section_title", "")),
            parent_id=str(metadata.get("parent_id", "")) or None,
            children_ids=[str(item) for item in children_ids],
            level=int(metadata.get("level", 0)),
            section_path=[str(item) for item in section_path],
            semantic_role=str(metadata.get("semantic_role", "")) or None,
            block_types=[PaperBlockType(str(item)) for item in block_types],
            object_labels=[str(item) for item in object_labels],
            text=document,
            embedding_text="\n".join([prefix, document]).strip(),
            page_start=page_start,
            page_end=page_end,
            figure_asset=figure_asset,
            figure_caption=figure_caption,
            raw_asset_ref=raw_asset_ref,
            table_rows=table_rows,
            spans=spans,
        )

    def _metadata(
        self,
        document: ParsedPaperDocument,
        node: TreeIndexNode,
    ) -> dict[str, str | int | bool]:
        prefix = "\n".join([document.title, *node.section_path])
        metadata: dict[str, str | int | bool] = {
            "schema_version": self.SCHEMA_VERSION,
            "paper_id": document.paper_id,
            "paper_title": document.title,
            "source_path": str(document.source_path),
            "node_type": node.node_type.value,
            "parent_id": node.parent_id or "",
            "children_ids": json.dumps(node.children_ids, ensure_ascii=False),
            "level": node.level,
            "section_path": json.dumps(node.section_path, ensure_ascii=False),
            "semantic_role": node.semantic_role or "",
            "block_types": json.dumps(
                [item.value for item in node.block_types], ensure_ascii=False
            ),
            "object_labels": json.dumps(node.object_labels, ensure_ascii=False),
            "section_title": node.title,
            "page_start": node.page_start or 0,
            "page_end": node.page_end or 0,
            "is_leaf": node.is_leaf,
            "embedding_prefix": prefix,
            "content_sha256": hashlib.sha256(node.text.encode("utf-8")).hexdigest(),
            "figure_asset": node.figure_asset or "",
            "figure_caption": node.figure_caption or "",
            "raw_asset_ref": node.raw_asset_ref or "",
            "table_rows": (
                json.dumps(node.table_rows, ensure_ascii=False)
                if node.table_rows is not None
                else ""
            ),
            "spans": json.dumps(
                [span.model_dump(mode="json") for span in node.spans],
                ensure_ascii=False,
            ),
        }
        if self._index_signature is not None:
            metadata["index_signature"] = self._index_signature.fingerprint
            metadata["embedding_model"] = self._index_signature.embedding_model
            metadata["chunker_version"] = self._index_signature.chunker_version
        return metadata
