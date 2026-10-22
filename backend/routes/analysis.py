
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.git_service import (
    GitServiceError,
    get_changed_files,
    get_diff,
)

router = APIRouter()


class AnalyzeRequest(BaseModel):
    repository_path: str = Field(min_length=1)
    base_revision: str = Field(min_length=1)
    target_revision: str = Field(min_length=1)


class AnalyzeResponse(BaseModel):
    status: str
    repository_path: str
    base_revision: str
    target_revision: str
    changed_files: list[str]
    changed_file_count: int
    diff: str
    tests_discovered: int | None
    message: str


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"description": "Invalid repository path or Git revision"},
        422: {"description": "Invalid request fields"},
    },
)
def analyze_repository(request: AnalyzeRequest):
    """Analyze changes between two revisions of a local Git repository."""
    try:
        changed_files = get_changed_files(
            request.repository_path,
            request.base_revision,
            request.target_revision,
        )

        diff = get_diff(
            request.repository_path,
            request.base_revision,
            request.target_revision,
        )

        return AnalyzeResponse(
            status="success",
            repository_path=request.repository_path,
            base_revision=request.base_revision,
            target_revision=request.target_revision,
            changed_files=changed_files,
            changed_file_count=len(changed_files),
            diff=diff,
            tests_discovered=None,
            message="Git repository analysis completed",
        )

    except GitServiceError as exc:
        # Avoid exposing raw Git errors in API responses.
        raise HTTPException(
            status_code=400,
            detail="Unable to analyze the repository or revisions provided.",
        ) from exc
