from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_paper_library, get_settings
from app.application.paper_library import PaperLibraryService
from app.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(
    settings: Annotated[Settings, Depends(get_settings)],
    paper_library: Annotated[PaperLibraryService, Depends(get_paper_library)],
) -> dict[str, object]:
    """Report liveness plus the durable locations and paper counts.

    The durable paths make a restore verifiable: after restoring a backup the
    paper count and index fingerprint should match the archive, instead of the
    library silently coming back empty.
    """

    statuses = await paper_library.index_status()
    return {
        "status": "ok",
        "durable_paths": settings.durable_paths,
        "papers": {
            "indexed": len(statuses),
            "needing_rebuild": len([item for item in statuses if item.needs_rebuild]),
            "without_fingerprint": len(
                [item for item in statuses if not item.fingerprint_recorded]
            ),
        },
        "index_signature": (
            None
            if paper_library.index_signature is None
            else paper_library.index_signature.fingerprint
        ),
    }
