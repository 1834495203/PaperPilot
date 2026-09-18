"""Guards for the deployment contract that keeps research data alive.

These assertions are cheap but they cover the failure mode that costs the most:
a container recreation that quietly leaves the paper library and vector index
outside the mounted volume.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_paper_library, get_settings
from app.api.routes import health
from app.application.paper_library import PaperLibraryService
from app.config import Settings
from app.domain.rag import IndexedPaper, PaperIndexStatus, PaperMetadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPOSITORY_ROOT / "docker-compose.yml"
DURABLE_VARIABLES = ("DATABASE_URL", "PAPER_LIBRARY_PATH", "VECTOR_DB_PATH")


def _compose() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    return cast(dict[str, Any], yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8")))


def test_compose_keeps_every_durable_path_inside_the_mounted_volume() -> None:
    compose = _compose()
    api = compose["services"]["api"]
    mounts = [item for item in api["volumes"] if item.endswith(":/data")]
    assert mounts, "the api service must mount the data volume at /data"

    environment = api["environment"]
    # DATABASE_URL is a URL, so the sqlite path is checked by containment; the two
    # directories must be absolute paths inside the mount.
    assert "/data/" in str(environment["DATABASE_URL"]), (
        f"DATABASE_URL={environment['DATABASE_URL']} would live outside the mounted "
        "volume and be lost when the container is recreated"
    )
    for variable in ("PAPER_LIBRARY_PATH", "VECTOR_DB_PATH"):
        value = str(environment[variable])
        assert value.startswith("/data/"), (
            f"{variable}={value} would live outside the mounted volume and be lost "
            "when the container is recreated"
        )


def test_compose_declares_the_named_volume_used_by_the_mount() -> None:
    compose = _compose()
    mounts = compose["services"]["api"]["volumes"]

    assert "volumes" in compose
    for mount in mounts:
        source = mount.split(":")[0]
        assert source in compose["volumes"], f"{source} is not a declared volume"


def test_durable_paths_are_reported_by_the_health_endpoint() -> None:
    application = FastAPI()
    application.include_router(health.router, prefix="/api/v1")
    settings = Settings(
        database_url="sqlite+aiosqlite:////data/paperpilot.db",
        paper_library_path=Path("/data/papers"),
        vector_db_path=Path("/data/chroma"),
    )

    class _Library:
        index_signature = None

        async def index_status(self) -> list[PaperIndexStatus]:
            return [
                PaperIndexStatus(
                    paper_id="paper-a",
                    title="Paper A",
                    chunk_count=3,
                    index_signature=None,
                    current_signature=None,
                )
            ]

    application.dependency_overrides[get_settings] = lambda: settings
    application.dependency_overrides[get_paper_library] = lambda: cast(
        PaperLibraryService, cast(Any, _Library())
    )

    response = TestClient(application).get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["durable_paths"]["vector_collection"] == settings.vector_collection
    assert payload["durable_paths"]["papers"].endswith("papers")
    assert payload["durable_paths"]["vectors"].endswith("chroma")
    assert payload["papers"] == {
        "indexed": 1,
        "needing_rebuild": 0,
        "without_fingerprint": 1,
    }


def test_sqlite_path_resolution_handles_absolute_and_relative_urls() -> None:
    absolute = Settings(database_url="sqlite+aiosqlite:////data/paperpilot.db")
    relative = Settings(database_url="sqlite+aiosqlite:///./paperpilot.db")
    other = Settings(database_url="postgresql+asyncpg://user@host/db")

    assert absolute.sqlite_path is not None
    assert absolute.sqlite_path.replace("\\", "/").endswith("data/paperpilot.db")
    assert relative.sqlite_path is not None
    assert relative.sqlite_path.replace("\\", "/").endswith("paperpilot.db")
    assert other.sqlite_path is None


def test_paper_index_status_reports_staleness_fields() -> None:
    paper = IndexedPaper(
        paper_id="paper-a",
        metadata=PaperMetadata(title="Paper A"),
        original_filename="paper.pdf",
        content_sha256="a" * 64,
        page_count=3,
        section_count=2,
        node_count=4,
        chunk_count=2,
        created_at=datetime.now(UTC),
    )
    status = PaperIndexStatus(
        paper_id=paper.paper_id,
        title=paper.title,
        chunk_count=paper.chunk_count,
        index_signature="abc",
        current_signature="def",
    )

    assert status.needs_rebuild is True
