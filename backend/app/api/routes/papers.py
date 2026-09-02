from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import get_paper_library
from app.api.schemas import (
    IndexedPaperDetailResponse,
    IndexedPaperResponse,
    RetrievePaperRequest,
    TreeRetrievalResponse,
)
from app.application.paper_library import (
    PaperLibraryService,
    PaperNotFoundError,
    PaperUploadError,
)

router = APIRouter(prefix="/papers", tags=["papers"])


@router.post("", response_model=IndexedPaperResponse, status_code=status.HTTP_201_CREATED)
async def upload_paper(
    service: Annotated[PaperLibraryService, Depends(get_paper_library)],
    file: Annotated[UploadFile, File(description="Scientific paper PDF")],
    title: Annotated[str | None, Form(max_length=500)] = None,
) -> IndexedPaperResponse:
    try:
        content = await file.read(service.max_upload_bytes + 1)
        paper = await service.upload_pdf(
            filename=file.filename or "paper.pdf",
            content=content,
            title=title,
        )
    except PaperUploadError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Paper ingestion failed: {error}",
        ) from error
    finally:
        await file.close()
    return IndexedPaperResponse.from_domain(paper)


@router.get("", response_model=list[IndexedPaperResponse])
async def list_papers(
    service: Annotated[PaperLibraryService, Depends(get_paper_library)],
) -> list[IndexedPaperResponse]:
    return [IndexedPaperResponse.from_domain(item) for item in await service.list_papers()]


@router.get("/{paper_id}", response_model=IndexedPaperDetailResponse)
async def get_paper_detail(
    paper_id: str,
    service: Annotated[PaperLibraryService, Depends(get_paper_library)],
) -> IndexedPaperDetailResponse:
    try:
        detail = await service.get_paper_detail(paper_id)
    except PaperNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Indexed paper not found",
        ) from error
    return IndexedPaperDetailResponse.from_domain(detail)


@router.post("/retrieve", response_model=TreeRetrievalResponse)
async def retrieve_papers(
    request: RetrievePaperRequest,
    service: Annotated[PaperLibraryService, Depends(get_paper_library)],
) -> TreeRetrievalResponse:
    try:
        report = await service.retrieve(
            request.query,
            paper_ids=request.paper_ids,
            mode=request.mode,
        )
    except PaperNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Indexed paper not found: {error}",
        ) from error
    return TreeRetrievalResponse.from_domain(report)
