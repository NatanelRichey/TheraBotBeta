from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    port: int = 8000

    # LLM Providers — direct keys
    openai_api_key: str = Field(default="", repr=False)
    anthropic_api_key: str = Field(default="", repr=False)
    deepseek_api_key: str = Field(default="", repr=False)
    kimi_api_key: str = Field(default="", repr=False)

    # LLM Providers — OpenRouter (one key, all models)
    openrouter_api_key: str = Field(default="", repr=False)

    # Primary / fallback model IDs (standard routing — direct keys)
    primary_model: str = "gpt-4o"
    fallback_model: str = "claude-sonnet-4-6"
    embedding_model: str = "text-embedding-3-small"

    # Cheap routing profile — direct keys
    deepseek_model: str = "deepseek-chat"
    kimi_model: str = "moonshot-v1-8k"

    # OpenRouter model strings (provider-prefixed)
    openrouter_default_primary_model: str = "openai/gpt-4o"
    openrouter_default_fallback_model: str = "anthropic/claude-sonnet-4-5"
    openrouter_cheap_primary_model: str = "deepseek/deepseek-chat"  # DeepSeek V3
    openrouter_cheap_fallback_model: str = "moonshotai/kimi-k2.5"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Semantic cache
    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    semantic_cache_ttl_seconds: int = 3600

    # Vector DB
    chroma_persist_dir: str = "./data/chroma"
    episodic_chroma_dir: str = "./data/episodic"
    chroma_url: str = "http://localhost:8001"
    pgvector_url: str = ""
    rag_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Agent / memory settings
    psych_profile_rag_threshold: int = Field(default=5, ge=1)
    working_memory_rebuild_interval: int = Field(default=20, ge=1)
    episodic_top_k: int = Field(default=3, ge=1)
    session_history_window: int = Field(default=30, ge=1)
    longterm_chroma_dir: str = "./data/longterm"

    # Cost tracking
    monthly_budget_usd: float = Field(default=10.0, ge=0.0)  # hard stop in dollars
    cost_alert_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # Evaluation
    eval_model: str = "gpt-4o-mini"
    eval_dataset_path: str = "./data/evals/"
    turn_trace_jsonl_path: str = "./data/evals/traces.jsonl"

    # LLM call behaviour
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3

    @field_validator("openai_api_key", "anthropic_api_key", mode="after")
    @classmethod
    def _warn_empty_keys(cls, v: str) -> str:
        # Validation happens at startup; missing keys will surface as errors
        # when the relevant provider is first used.
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def use_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def vector_store_backend(self) -> Literal["chroma", "pgvector"]:
        return "pgvector" if (self.is_production and self.pgvector_url) else "chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
