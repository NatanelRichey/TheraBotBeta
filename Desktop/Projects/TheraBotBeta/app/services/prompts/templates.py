"""Prompt template loader with in-memory caching.

Templates are stored as plain .txt files under data/prompts/.
File naming convention: {name}_{version}.txt  (e.g. identity_warm_v1.txt)
"""
import threading
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

# Resolved at import time so it's stable regardless of working directory.
# app/services/prompts/templates.py → parents[3] == project root
_PROMPTS_DIR: Path = Path(__file__).parents[3] / "data" / "prompts"

_cache: dict[str, str] = {}
_lock = threading.Lock()


def load(name: str, version: str = "v1") -> str:
    """Load a prompt template from disk, caching the result in memory.

    Args:
        name: Template base name, e.g. ``"identity_warm"``.
        version: Version suffix, e.g. ``"v1"``.

    Returns:
        The raw template text with leading/trailing whitespace stripped.

    Raises:
        FileNotFoundError: If ``data/prompts/{name}_{version}.txt`` does not exist.
    """
    key = f"{name}_{version}"

    # Fast path — no lock needed for reads after the first load.
    if key in _cache:
        return _cache[key]

    with _lock:
        # Double-checked locking: another thread may have loaded it while we waited.
        if key in _cache:
            return _cache[key]

        file_path = _PROMPTS_DIR / f"{key}.txt"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {file_path}. "
                f"Expected file: data/prompts/{key}.txt"
            )

        content = file_path.read_text(encoding="utf-8").strip()
        _cache[key] = content
        logger.info("prompt_template_loaded", name=name, version=version)
        return content


def render(name: str, version: str = "v1", **variables: str) -> str:
    """Load a template and apply optional ``{variable}`` substitution.

    Args:
        name: Template base name.
        version: Version suffix.
        **variables: Key/value pairs substituted into ``{key}`` placeholders.

    Returns:
        The rendered template string.
    """
    template = load(name, version)
    return template.format_map(variables) if variables else template
