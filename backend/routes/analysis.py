from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.git_service import (
    GitServiceError,
    get_changed_files,
    get_diff,
)
from services.test_discovery import discover_tests

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
    discovered_tests: list[str] | None = None


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

        try:
            discovery_result = discover_tests(request.repository_path)
            tests_discovered = discovery_result.total_tests
            discovered_test_node_ids = [t.node_id for t in discovery_result.tests]
        except Exception:
            tests_discovered = 0
            discovered_test_node_ids = []

        return AnalyzeResponse(
            status="success",
            repository_path=request.repository_path,
            base_revision=request.base_revision,
            target_revision=request.target_revision,
            changed_files=changed_files,
            changed_file_count=len(changed_files),
            diff=diff,
            tests_discovered=tests_discovered,
            discovered_tests=discovered_test_node_ids,
            message="Git repository analysis and test discovery completed",
        )

    except GitServiceError as exc:
        # Avoid exposing raw Git errors in API responses.
        raise HTTPException(
            status_code=400,
            detail="Unable to analyze the repository or revisions provided.",
        ) from exc
