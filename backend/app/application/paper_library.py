import asyncio
import hashlib
import json
import re
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from app.application.paper_ingestion import PaperIngestionService
from app.application.tree_retrieval import TreeRagRetriever
from app.domain.ports import ScientificPaperParser, TreeVectorStore
from app.domain.rag import (
    IndexedPaper,
    IndexedPaperDetail,
    PaperTreeNodeView,
    RetrievalMode,
    TreeRetrievalReport,
)


class PaperUploadError(ValueError):
    pass


class PaperNotFoundError(LookupError):
    pass


class PaperLibraryService:
    """Own uploaded PDFs and expose ingestion/retrieval as one application boundary."""

    def __init__(
        self,
        *,
        library_path: Path,
        max_upload_bytes: int,
        ingestion: PaperIngestionService,
        retriever: TreeRagRetriever,
        vector_store: TreeVectorStore | None = None,
        metadata_parser: ScientificPaperParser | None = None,
    ) -> None:
        self._library_path = library_path.resolve()
        self._max_upload_bytes = max_upload_bytes
        self._ingestion = ingestion
        self._retriever = retriever
        self._vector_store = vector_store
        self._metadata_parser = metadata_parser
        self._lock = asyncio.Lock()

    @property
    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes

    async def initialize(self) -> None:
        await asyncio.to_thread(self._library_path.mkdir, parents=True, exist_ok=True)
        await self._backfill_legacy_metadata()

    async def upload_pdf(
        self,
        *,
        filename: str,
        content: bytes,
        title: str | None = None,
    ) -> IndexedPaper:
        self._validate_upload(filename, content)
        digest = hashlib.sha256(content).hexdigest()
        paper_id = f"{self._slug(Path(filename).stem)}-{digest[:12]}"
        pdf_path = self._library_path / f"{paper_id}.pdf"
        manifest_path = self._manifest_path(paper_id)
        existed = pdf_path.exists()
        async with self._lock:
            await asyncio.to_thread(pdf_path.write_bytes, content)
            try:
                result = await self._ingestion.ingest_pdf(
                    str(pdf_path),
                    paper_id=paper_id,
                    title=title.strip() if title and title.strip() else None,
                )
            except Exception:
                if not existed:
                    await asyncio.to_thread(pdf_path.unlink, missing_ok=True)
                raise
            record = IndexedPaper(
                paper_id=paper_id,
                metadata=result.metadata,
                original_filename=Path(filename).name,
                content_sha256=digest,
                page_count=result.page_count,
                section_count=result.section_count,
                node_count=result.node_count,
                chunk_count=result.chunk_count,
            )
            await asyncio.to_thread(
                manifest_path.write_text,
                record.model_dump_json(indent=2),
                encoding="utf-8",
            )
            return record

    async def list_papers(self) -> list[IndexedPaper]:
        return await asyncio.to_thread(self._list_papers_sync)

    async def get_paper(self, paper_id: str) -> IndexedPaper:
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", paper_id) is None:
            raise PaperNotFoundError(paper_id)
        try:
            content = await asyncio.to_thread(
                self._manifest_path(paper_id).read_text,
                encoding="utf-8",
            )
            return IndexedPaper.model_validate_json(content)
        except (OSError, ValueError) as error:
            raise PaperNotFoundError(paper_id) from error

    async def get_paper_detail(self, paper_id: str) -> IndexedPaperDetail:
        paper = await self.get_paper(paper_id)
        if self._vector_store is None:
            raise RuntimeError("Paper tree index reader is not configured")
        indexed_nodes = await self._vector_store.load_paper_nodes([paper_id])
        nodes = sorted(
            (
                PaperTreeNodeView(
                    node_id=item.node.node_id,
                    node_type=item.node.node_type,
                    title=item.node.title,
                    parent_id=item.node.parent_id,
                    children_ids=item.node.children_ids,
                    level=item.node.level,
                    section_path=item.node.section_path,
                    semantic_role=item.node.semantic_role,
                    block_types=item.node.block_types,
                    object_labels=item.node.object_labels,
                    page_start=item.node.page_start,
                    page_end=item.node.page_end,
                    text_preview=self._preview(item.node.text),
                    text=item.node.text,
                )
                for item in indexed_nodes
            ),
            key=lambda item: (
                item.page_start or 0,
                item.level,
                item.node_id,
            ),
        )
        return IndexedPaperDetail(paper=paper, nodes=nodes)

    async def delete_paper(self, paper_id: str) -> IndexedPaper:
        """Delete one paper's vector nodes, manifest, and managed PDF as one operation."""
        if self._vector_store is None:
            raise RuntimeError("Paper tree index writer is not configured")
        async with self._lock:
            paper = await self.get_paper(paper_id)
            sources = [
                self._manifest_path(paper.paper_id),
                self._library_path / f"{paper.paper_id}.pdf",
            ]
            token = uuid4().hex
            staged: list[tuple[Path, Path]] = []
            try:
                for source in sources:
                    if not source.exists():
                        continue
                    temporary = source.with_name(f".{source.name}.{token}.deleting")
                    await asyncio.to_thread(source.replace, temporary)
                    staged.append((source, temporary))
                await self._vector_store.delete_paper(paper.paper_id)
            except Exception:
                for source, temporary in reversed(staged):
                    if temporary.exists():
                        await asyncio.to_thread(temporary.replace, source)
                raise
            await asyncio.to_thread(self._remove_staged_files, staged)
            return paper

    async def retrieve(
        self,
        query: str,
        *,
        paper_ids: list[str] | None = None,
        mode: RetrievalMode,
    ) -> TreeRetrievalReport:
        known_ids = {paper.paper_id for paper in await self.list_papers()}
        missing = [paper_id for paper_id in (paper_ids or []) if paper_id not in known_ids]
        if missing:
            raise PaperNotFoundError(", ".join(missing))
        return await self._retriever.retrieve(query, paper_ids=paper_ids, mode=mode)

    def _list_papers_sync(self) -> list[IndexedPaper]:
        records: list[IndexedPaper] = []
        for path in self._library_path.glob("*.json"):
            try:
                records.append(IndexedPaper.model_validate_json(path.read_text("utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def _manifest_path(self, paper_id: str) -> Path:
        return self._library_path / f"{paper_id}.json"

    @staticmethod
    def _remove_staged_files(staged: list[tuple[Path, Path]]) -> None:
        for _, temporary in staged:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    async def _backfill_legacy_metadata(self) -> None:
        """Upgrade pre-metadata manifests without rebuilding vector embeddings."""
        if self._metadata_parser is None:
            return
        for manifest_path in self._library_path.glob("*.json"):
            try:
                raw = json.loads(await asyncio.to_thread(manifest_path.read_text, "utf-8"))
                if not isinstance(raw, dict) or "metadata" in raw:
                    continue
                paper_id = str(raw["paper_id"])
                pdf_path = self._library_path / f"{paper_id}.pdf"
                document = await self._metadata_parser.parse(
                    str(pdf_path),
                    paper_id=paper_id,
                )
                raw.pop("title", None)
                raw["metadata"] = document.metadata.model_dump(mode="json")
                record = IndexedPaper.model_validate(raw)
                await asyncio.to_thread(
                    manifest_path.write_text,
                    record.model_dump_json(indent=2),
                    encoding="utf-8",
                )
            except (KeyError, OSError, ValueError, RuntimeError):
                continue

    @staticmethod
    def _preview(text: str, max_characters: int = 240) -> str:
        compact = " ".join(text.split())
        return (
            compact
            if len(compact) <= max_characters
            else f"{compact[:max_characters].rstrip()}…"
        )

    def _validate_upload(self, filename: str, content: bytes) -> None:
        if not filename.lower().endswith(".pdf"):
            raise PaperUploadError("Only PDF files are supported")
        if not content:
            raise PaperUploadError("Uploaded PDF is empty")
        if len(content) > self._max_upload_bytes:
            raise PaperUploadError(
                f"PDF exceeds the {self._max_upload_bytes}-byte upload limit"
            )
        if not content.startswith(b"%PDF-"):
            raise PaperUploadError("Uploaded file does not have a valid PDF signature")

    @staticmethod
    def _slug(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return normalized[:48] or "paper"
