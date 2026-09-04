import asyncio
import hashlib
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from app.domain.ports import TreeVectorStore
from app.domain.rag import (
    IndexedTreeNode,
    PaperBlockType,
    ParsedPaperDocument,
    TreeIndexNode,
    TreeNodeType,
    TreeVectorMatch,
)


class ChromaTreeVectorStore(TreeVectorStore):
    """Persist TreeRAG nodes and caller-supplied embeddings in a Chroma collection."""

    def __init__(self, *, persist_directory: Path, collection_name: str) -> None:
        if len(collection_name) < 3:
            raise ValueError("Chroma collection_name must contain at least three characters")
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
            metadata={
                "description": "PaperPilot TreeRAG paper and chunk index",
                "hnsw:space": "cosine",
                "schema_version": 3,
            },
        )
        self._collection_name = collection_name

    @property
    def collection_name(self) -> str:
        return self._collection_name

    async def replace_paper(
        self,
        document: ParsedPaperDocument,
        nodes: Sequence[TreeIndexNode],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
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
        return await asyncio.to_thread(
            self._similarity_search_sync,
            query_embedding,
            paper_ids,
            top_k,
            chunks_only,
            node_types,
            parent_ids,
        )

    async def load_paper_nodes(
        self,
        paper_ids: Sequence[str] | None,
    ) -> list[IndexedTreeNode]:
        return await asyncio.to_thread(self._load_paper_nodes_sync, paper_ids)

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

    def _load_paper_nodes_sync(
        self,
        paper_ids: Sequence[str] | None,
    ) -> list[IndexedTreeNode]:
        get_kwargs: dict[str, Any] = {
            "include": ["documents", "metadatas", "embeddings"]
        }
        where = self._where_filter(paper_ids, chunks_only=False)
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
        )

    @staticmethod
    def _metadata(
        document: ParsedPaperDocument,
        node: TreeIndexNode,
    ) -> dict[str, str | int | bool]:
        prefix = "\n".join([document.title, *node.section_path])
        return {
            "schema_version": 3,
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
        }
