from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config import get_settings

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


class ReadinessResponse(BaseModel):
    status: str  # "ok" | "degraded"
    checks: dict[str, str]


@router.get("", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Liveness probe — confirms the process is running."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness() -> JSONResponse:
    """Readiness probe — confirms the app is configured and ready to serve traffic."""
    settings = get_settings()
    checks: dict[str, str] = {}
    healthy = True

    # LLM providers — at least one key must be present
    checks["openai"] = "ok" if settings.openai_api_key else "missing key"
    checks["anthropic"] = "ok" if settings.anthropic_api_key else "missing key"

    if not settings.openai_api_key and not settings.anthropic_api_key:
        healthy = False

    status = "ok" if healthy else "degraded"
    status_code = 200 if healthy else 503

    return JSONResponse(
        status_code=status_code,
        content=ReadinessResponse(status=status, checks=checks).model_dump(mode="json"),
    )
