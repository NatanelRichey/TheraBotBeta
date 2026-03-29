from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.exceptions import BudgetExceededError, LLMProviderError
from app.core.logging import get_logger, setup_logging

STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logger = get_logger(__name__)
    logger.info(
        "therabot_startup",
        default_profile=f"{settings.primary_model} → {settings.fallback_model}",
        cheap_profile=f"{settings.deepseek_model} → {settings.kimi_model}",
    )
    yield


app = FastAPI(
    title="TheraBot",
    description="AI wellness companion",
    version="0.1.0",
    lifespan=lifespan,
)

# --- CORS ---
# allow_origins=["*"] is intentional for local dev / portfolio demos.
# TODO: restrict to your actual domain before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Exception handlers ---

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger = get_logger(__name__)
    body: bytes = await request.body()
    logger.warning(
        "request_validation_error",
        path=request.url.path,
        method=request.method,
        errors=exc.errors(),
        body=body.decode("utf-8", errors="replace"),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(LLMProviderError)
async def llm_error_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "LLM provider unavailable"})


@app.exception_handler(BudgetExceededError)
async def budget_error_handler(request: Request, exc: BudgetExceededError) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "Monthly token budget exceeded"})


# --- Routers ---

from app.api.chat import router as chat_router  # noqa: E402
from app.api.health import router as health_router  # noqa: E402

app.include_router(chat_router)
app.include_router(health_router)


# --- Chat UI ---
# Serve the single-file chat interface at /
# Mount must come after API routers so /chat and /health are not shadowed.
@app.get("/", include_in_schema=False)
async def serve_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
