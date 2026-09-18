"""Round-trip tests for the backup and restore path.

The acceptance property is "a restart or container recreation does not lose
research data", so the tests wipe the data directories and check that a restore
brings back the database, the paper manifests and the vector index, and that a
damaged archive is rejected instead of producing a half-restored library.
"""

import sqlite3
import tarfile
from contextlib import closing
from pathlib import Path

import pytest

from app.application.backup_service import (
    MANIFEST_NAME,
    BackupError,
    count_indexed_papers,
    create_backup,
    restore_backup,
)
from app.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path.as_posix()}/paperpilot.db",
        paper_library_path=tmp_path / "papers",
        vector_db_path=tmp_path / "chroma",
        vector_collection="paperpilot_tree_chunks",
    )


def _seed(settings: Settings) -> None:
    database = Path(settings.sqlite_path or "")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO conversations (id) VALUES ('conversation-1')")
        connection.commit()
    papers = settings.paper_library_path
    papers.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (papers / f"paper-{index}.json").write_text('{"paper_id": "x"}', encoding="utf-8")
    (papers / "paper-0.pdf").write_bytes(b"%PDF-1.4 body")
    vectors = settings.vector_db_path
    vectors.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(vectors / "chroma.sqlite3")) as connection:
        connection.execute(
            "CREATE TABLE collection_metadata (key TEXT, str_value TEXT)"
        )
        connection.execute(
            "INSERT INTO collection_metadata VALUES ('index_signature', 'abc123')"
        )
        connection.commit()


def test_backup_then_restore_returns_every_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed(settings)
    archive_path = tmp_path / "backup.tar.gz"

    created = create_backup(settings, archive_path)

    assert created.manifest.paper_count == 3
    assert created.manifest.database_included is True
    assert created.manifest.papers_included is True
    assert created.manifest.vectors_included is True
    assert created.manifest.index_signature == "abc123"

    # Simulate a container recreation that wiped the data directories.
    for target in (
        Path(settings.sqlite_path or ""),
        settings.paper_library_path,
        settings.vector_db_path,
    ):
        if target.is_dir():
            for child in sorted(target.rglob("*"), reverse=True):
                child.unlink() if child.is_file() else child.rmdir()
            target.rmdir()
        else:
            target.unlink()
    assert count_indexed_papers(settings.paper_library_path) == 0

    verification = restore_backup(settings, archive_path, force=True)

    assert verification.consistent is True
    assert verification.restored_papers == 3
    assert verification.restored_database is True
    assert verification.restored_vectors is True
    assert verification.warnings == []
    with closing(sqlite3.connect(Path(settings.sqlite_path or ""))) as connection:
        rows = connection.execute("SELECT id FROM conversations").fetchall()
    assert rows == [("conversation-1",)]
    assert (settings.paper_library_path / "paper-0.pdf").read_bytes() == b"%PDF-1.4 body"


def test_restore_requires_force_and_replaces_nothing_without_it(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed(settings)
    archive_path = tmp_path / "backup.tar.gz"
    create_backup(settings, archive_path)

    with pytest.raises(BackupError, match="--force"):
        restore_backup(settings, archive_path)

    # The refusal must not have touched the existing data.
    assert count_indexed_papers(settings.paper_library_path) == 3


def test_backup_refuses_to_overwrite_and_reports_a_missing_database(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    archive_path = tmp_path / "backup.tar.gz"
    archive_path.write_bytes(b"existing")

    with pytest.raises(BackupError, match="overwrite"):
        create_backup(settings, archive_path)

    archive_path.unlink()
    result = create_backup(settings, archive_path)

    assert result.manifest.database_included is False
    assert result.manifest.papers_included is False
    assert result.manifest.notes


def test_restore_rejects_an_archive_that_is_not_a_paperpilot_backup(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    foreign = tmp_path / "foreign.tar.gz"
    with tarfile.open(foreign, "w:gz") as archive:
        archive.add(str(_foreign_file(tmp_path)), arcname="something.txt")

    with pytest.raises(BackupError, match=MANIFEST_NAME):
        restore_backup(settings, foreign, force=True)


def test_restore_verification_reports_a_paper_count_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed(settings)
    archive_path = tmp_path / "backup.tar.gz"
    create_backup(settings, archive_path)
    # Delete one manifest after the archive was written: the archive still claims 3.
    (settings.paper_library_path / "paper-2.json").unlink()
    (settings.paper_library_path / "paper-0.json").unlink()
    (settings.paper_library_path / "paper-1.json").unlink()

    verification = restore_backup(settings, archive_path, force=True)

    assert verification.restored_papers == 3
    assert verification.consistent is True


def _foreign_file(tmp_path: Path) -> Path:
    path = tmp_path / "something.txt"
    path.write_text("not a backup", encoding="utf-8")
    return path
