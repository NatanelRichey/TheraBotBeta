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


RETRIEVAL_QUERY_REWRITE_PROMPT = """You are a DBT (Dialectical Behavior Therapy) clinical search assistant.

Your job is to rewrite a patient's message into a short search query that will match relevant DBT skill handouts.

The knowledge base contains handouts from these modules:
- Mindfulness (Wise Mind, What/How skills, observing, describing)
- Emotion Regulation (identifying emotions, fear, anger, shame, reducing vulnerability)
- Distress Tolerance (crisis survival, radical acceptance, paired relaxation, rethinking)
- Interpersonal Effectiveness (relationship skills, DEAR MAN, boundaries)
- General wellness skills

Available skill categories — prefer these exact terms in your query when relevant:
{vocabulary}

Rules:
- Return ONLY the rewritten query. No explanation, no preamble.
- Prefer terms from the skill categories list above when they match the patient's need.
- Keep it under 20 words.
- Focus on the skill or emotion the patient needs help with, not the story around it.

Examples:
Patient: "I feel so anxious and overwhelmed and I can't stop my thoughts"
Query: anxiety overwhelmed emotional dysregulation mindfulness grounding distress tolerance

Patient: "My partner and I keep fighting and I don't know how to talk to them"
Query: interpersonal conflict communication relationship effectiveness DEAR MAN

Patient: "I know I should feel better but I just can't stop being sad"
Query: depression emotion regulation sadness reducing emotional vulnerability

Patient: "I'm about to do something I'll regret, I need to calm down right now"
Query: crisis survival distress tolerance urge surfing radical acceptance

---
Patient: "{message}"
Query:"""


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
