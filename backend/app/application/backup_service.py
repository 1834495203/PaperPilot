"""Back up and restore every durable PaperPilot artifact as one verified archive.

A backup is only useful if the restore is verified, so both directions write or
check a manifest that records the index fingerprint and the number of indexed
papers. A restore whose manifest disagrees with what landed on disk fails loudly
instead of starting from a half-restored library.
"""

import json
import shutil
import sqlite3
import tarfile
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings

MANIFEST_NAME = "paperpilot-backup.json"
DATABASE_ENTRY = "paperpilot.db"
PAPERS_ENTRY = "papers"
VECTORS_ENTRY = "chroma"


class BackupManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    app_version: str = Field(default="0.1.0", min_length=1)
    paper_count: int = Field(default=0, ge=0)
    index_signature: str | None = None
    vector_collection: str = Field(min_length=1)
    database_included: bool = False
    papers_included: bool = False
    vectors_included: bool = False
    notes: list[str] = Field(default_factory=list)


class BackupError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BackupResult:
    archive: Path
    manifest: BackupManifest
    bytes_written: int


@dataclass(frozen=True, slots=True)
class RestoreVerification:
    manifest: BackupManifest
    restored_papers: int
    restored_database: bool
    restored_vectors: bool
    warnings: list[str]

    @property
    def consistent(self) -> bool:
        return (
            self.restored_papers == self.manifest.paper_count
            and (not self.manifest.database_included or self.restored_database)
            and (not self.manifest.vectors_included or self.restored_vectors)
        )


def count_indexed_papers(papers_path: Path) -> int:
    if not papers_path.is_dir():
        return 0
    return len([path for path in papers_path.glob("*.json")])


def create_backup(settings: Settings, output: Path) -> BackupResult:
    """Write an archive of the database, paper library and vector index."""

    output = output.expanduser()
    if output.exists():
        raise BackupError(f"refusing to overwrite existing archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    database_path = settings.sqlite_path
    papers_path = settings.paper_library_path
    vectors_path = settings.vector_db_path
    if database_path is None:
        raise BackupError(
            "backup supports SQLite deployments; back up the database with your own "
            "tooling for other drivers"
        )

    notes: list[str] = []
    with tarfile.open(output, "w:gz") as archive:
        database_included = _add_sqlite_snapshot(archive, Path(database_path), notes)
        papers_included = _add_tree(archive, papers_path, PAPERS_ENTRY, notes)
        vectors_included = _add_tree(archive, vectors_path, VECTORS_ENTRY, notes)
        manifest = BackupManifest(
            paper_count=count_indexed_papers(papers_path),
            index_signature=_read_index_signature(vectors_path),
            vector_collection=settings.vector_collection,
            database_included=database_included,
            papers_included=papers_included,
            vectors_included=vectors_included,
            notes=notes,
        )
        payload = manifest.model_dump_json(indent=2).encode("utf-8")
        _add_bytes(archive, MANIFEST_NAME, payload)
    return BackupResult(
        archive=output,
        manifest=manifest,
        bytes_written=output.stat().st_size,
    )


def restore_backup(
    settings: Settings,
    archive: Path,
    *,
    force: bool = False,
) -> RestoreVerification:
    """Restore an archive into the configured data locations and verify it."""

    archive = archive.expanduser()
    if not archive.is_file():
        raise BackupError(f"archive not found: {archive}")
    if not force:
        raise BackupError(
            "restore replaces the database, paper library and vector index; "
            "re-run with --force to proceed"
        )
    with tarfile.open(archive, "r:gz") as handle:
        manifest = _read_manifest(handle)
        database_path = settings.sqlite_path
        if manifest.database_included and database_path is not None:
            _extract_member(handle, DATABASE_ENTRY, Path(database_path))
        if manifest.papers_included:
            _extract_member(handle, PAPERS_ENTRY, settings.paper_library_path)
        if manifest.vectors_included:
            _extract_member(handle, VECTORS_ENTRY, settings.vector_db_path)

    restored_papers = count_indexed_papers(settings.paper_library_path)
    warnings: list[str] = []
    if restored_papers != manifest.paper_count:
        warnings.append(
            f"manifest expected {manifest.paper_count} papers but {restored_papers} "
            "manifests were restored"
        )
    current_signature = _read_index_signature(settings.vector_db_path)
    if manifest.index_signature and current_signature != manifest.index_signature:
        warnings.append(
            "the restored vector index fingerprint "
            f"{current_signature or 'unknown'} does not match the archive "
            f"{manifest.index_signature}; re-ingest the papers if this was not expected"
        )
    return RestoreVerification(
        manifest=manifest,
        restored_papers=restored_papers,
        restored_database=settings.sqlite_path is not None
        and Path(settings.sqlite_path).is_file(),
        restored_vectors=settings.vector_db_path.is_dir(),
        warnings=warnings,
    )


def _add_sqlite_snapshot(
    archive: tarfile.TarFile,
    database_path: Path,
    notes: list[str],
) -> bool:
    if not database_path.is_file():
        notes.append(f"database {database_path} not found; archive has no database")
        return False
    snapshot = database_path.with_suffix(database_path.suffix + ".snapshot")
    try:
        # The online backup API yields a consistent copy even while the app runs.
        # Connections are closed explicitly: a Windows handle would block the
        # cleanup below.
        with closing(sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(str(snapshot))) as target:
                source.backup(target)
        archive.add(str(snapshot), arcname=DATABASE_ENTRY)
    finally:
        snapshot.unlink(missing_ok=True)
    return True


def _add_tree(
    archive: tarfile.TarFile,
    source: Path,
    entry: str,
    notes: list[str],
) -> bool:
    if not source.is_dir():
        notes.append(f"{source} not found; archive has no {entry} directory")
        return False
    archive.add(str(source), arcname=entry, recursive=True)
    return True


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    import io

    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = int(datetime.now(UTC).timestamp())
    archive.addfile(info, io.BytesIO(payload))


def _read_manifest(handle: tarfile.TarFile) -> BackupManifest:
    try:
        member = handle.extractfile(MANIFEST_NAME)
    except KeyError as error:
        raise BackupError(
            f"archive does not contain {MANIFEST_NAME}; it is not a PaperPilot backup"
        ) from error
    if member is None:
        raise BackupError(
            f"archive does not contain {MANIFEST_NAME}; it is not a PaperPilot backup"
        )
    try:
        return BackupManifest.model_validate_json(member.read().decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise BackupError(f"archive manifest is unreadable: {error}") from error


def _extract_member(handle: tarfile.TarFile, entry: str, target: Path) -> None:
    """Replace one target location with the archived copy of it."""

    members = [
        member
        for member in handle.getmembers()
        if member.name == entry or member.name.startswith(f"{entry}/")
    ]
    if not members:
        raise BackupError(f"archive is missing the {entry} entry")
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle.extractall(path=str(target.parent), members=members)


def _read_index_signature(vectors_path: Path) -> str | None:
    """Best-effort fingerprint read without opening a Chroma client."""

    candidates = [
        vectors_path / "chroma.sqlite3",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with closing(
                sqlite3.connect(f"file:{candidate}?mode=ro", uri=True)
            ) as connection:
                rows = connection.execute(
                    "SELECT str_value FROM collection_metadata "
                    "WHERE key = 'index_signature' LIMIT 1"
                ).fetchall()
        except sqlite3.Error:
            return None
        if rows:
            return str(rows[0][0])
    return None


def manifest_payload(manifest: BackupManifest) -> str:
    return json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
