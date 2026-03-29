"""
Knowledge retriever with LLM-powered query rewriting.

Flow:
    user message
        → rewrite_query()  (LLM rewrites into DBT terminology)
        → ChromaDB .query()  (embed rewritten query, cosine search)
        → list[RetrievalResult]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.chat import ChatMessage, MessageRole
from app.services.llm.router import get_cheap_router, LLMRouter
from app.services.prompts.templates import RETRIEVAL_QUERY_REWRITE_PROMPT

logger = get_logger(__name__)

COLLECTION_NAME = "wellness_knowledge"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 3


# ---------------------------------------------------------------------------
# Hand-written skill definitions — one line per category, framed to help the
# query-rewrite LLM understand what user situations map to each DBT skill.
# Covers acronym expansions (ACCEPTS, DEAR MAN, TIP, PLEASE) and clinical terms
# that are opaque from the name alone.
# ---------------------------------------------------------------------------
_SKILL_DEFINITIONS: dict[str, str] = {
    # --- Mindfulness ---
    "Wise Mind": (
        "The balanced state integrating emotional mind (pure feeling) and reasonable mind "
        "(pure logic) — intuitive knowing that is neither cold nor impulsive"
    ),
    "Wise Mind: States of Mind": (
        "Three mental states: emotional mind (feelings-driven), reasonable mind (logic-driven), "
        "and wise mind (integrated balance between the two)"
    ),
    "Wise Mind: Walking the Middle Path": (
        "Using wise mind to find the middle ground between emotional extremes and cold rationality"
    ),
    "Wise Mind from a Spiritual Perspective": (
        "Wise mind as inner knowing connected to spiritual practice, intuition, and "
        "transcendent awareness beyond rational thought"
    ),
    'Mindfulness "What" Skills': (
        "Three core mindfulness actions — Observe (notice without reacting), Describe "
        "(label without judgment), Participate (engage fully in the present moment)"
    ),
    'Mindfulness "How" Skills': (
        "How to practice any mindfulness skill — non-judgmentally, one-mindfully (one thing "
        "at a time), and effectively (do what works)"
    ),
    'Taking Hold of Your Mind: "What" Skills': (
        "Observe, Describe, Participate — the three things you do when practicing mindfulness"
    ),
    'Taking Hold of Your Mind: "How" Skills': (
        "Practice mindfulness non-judgmentally, one-mindfully, and effectively"
    ),
    "Overview: Core Mindfulness Skills": (
        "Overview of wise mind and the What and How mindfulness skills"
    ),
    "Goals of Mindfulness Practice": (
        "Reduce suffering and increase happiness by gaining control of your mind and "
        "staying in the present moment"
    ),
    "Goals of Mindfulness Practice: A Spiritual Perspective": (
        "Mindfulness as a spiritual path to reduce suffering through inner peace, "
        "awareness, and awakening"
    ),
    "Mindfulness Practice: A Spiritual Perspective": (
        "Connecting mindfulness to spiritual traditions — loving kindness, compassion, "
        "and transcendent awareness"
    ),
    "Mindfulness Definitions": (
        "Paying attention, on purpose, to the present moment, non-judgmentally — "
        "the core definition and elements of mindfulness"
    ),
    "Mindfulness Skills": (
        "DBT mindfulness skills for observing and participating in the present moment "
        "without automatic emotional reactions"
    ),
    "Other Perspectives on Mindfulness": (
        "Mindfulness viewed through Buddhist, contemplative, and Western spiritual frameworks"
    ),
    "Overview: Other Perspectives on Mindfulness": (
        "Survey of spiritual and contemplative perspectives on mindfulness beyond standard DBT"
    ),
    "Skillful Means: Balancing Doing Mind and Being Mind": (
        "Balancing goal-directed action (doing mode) with relaxed present-moment awareness "
        "(being mode) to reduce stress and burnout"
    ),
    "Ideas for Practicing Balancing Doing Mind and Being Mind": (
        "Exercises for shifting between effortful action mode and receptive awareness mode"
    ),
    "Ideas for Practicing Wise Mind": (
        "Practical exercises for accessing balanced intuitive knowing — breathing into "
        "wise mind, asking wise mind, listening for answers"
    ),
    "Ideas for Practicing Observing": (
        "Exercises for noticing thoughts, feelings, and sensations without getting "
        "swept away — watching the mind like clouds passing"
    ),
    "Ideas for Practicing Describing": (
        "Exercises for labeling inner experiences precisely and without judgment — "
        "putting words to feelings and thoughts"
    ),
    "Ideas for Practicing Participating": (
        "Exercises for fully throwing yourself into activity without self-consciousness "
        "or distraction — becoming one with what you are doing"
    ),
    "Ideas for Practicing One-Mindfulness": (
        "Practicing doing one thing at a time with complete undivided attention — "
        "antidote to multitasking and rumination"
    ),
    "Ideas for Practicing Nonjudgmentalness": (
        "Practicing observing without labeling experiences as good or bad, right or wrong — "
        "just the facts"
    ),
    "Ideas for Practicing Effectiveness": (
        "Practicing doing what works given your goals, rather than what feels fair, "
        "right, or satisfying to your ego"
    ),
    "Practicing Loving Kindness to Increase Love and Compassion": (
        "Meditation for cultivating warmth and compassion toward yourself and others — "
        "reduces shame, self-hatred, loneliness, and anger"
    ),
    "Half-Smiling and Willing Hands": (
        "Adopting a gentle half-smile and open relaxed hands to signal acceptance to "
        "your nervous system — changes emotion through body posture"
    ),
    "Practicing Half-Smiling and Willing Hands": (
        "Practice exercises for using facial expression and hand posture to cultivate "
        "acceptance and reduce emotional resistance"
    ),

    # --- Emotion Regulation ---
    "Goals of Emotion Regulation": (
        "Understand and name emotions, reduce vulnerability, decrease emotional suffering, "
        "and change unwanted emotional experiences"
    ),
    "Understanding and Naming Emotions": (
        "Identifying and accurately labeling emotions as the first and most essential "
        "step in regulating them"
    ),
    "Model for Describing Emotions": (
        "Framework showing how prompting events, interpretations, body changes, action "
        "urges, and aftereffects connect to form a full emotional experience"
    ),
    "Ways to Describe Emotions": (
        "Vocabulary and structured approaches for describing emotional experiences "
        "clearly, specifically, and without judgment"
    ),
    "Myths about Emotions": (
        "Challenging false beliefs about emotions — e.g., 'showing emotions is weakness,' "
        "'negative emotions are bad' — that block effective regulation"
    ),
    "Goals and Factors That Interfere": (
        "What gets in the way of using emotion regulation skills — emotional myths, "
        "fear of change, skill deficits, and reinforcement of problem behaviors"
    ),
    "Check the Facts": (
        "Examining whether your emotional reaction fits the actual facts of the situation "
        "or the story and interpretations you have added to those facts"
    ),
    "Opposite Action": (
        "Acting opposite to what the emotion urges you to do in order to change the "
        "emotion — approach what fear says to avoid, engage when depression says withdraw"
    ),
    "Opposite Action and Problem Solving": (
        "Deciding whether to use opposite action (change the emotion) or problem solving "
        "(change the situation) based on whether the emotion fits the facts"
    ),
    "Problem Solving": (
        "Identifying and addressing the actual problem causing emotional distress when "
        "the situation can and should be changed"
    ),
    "Reducing Vulnerability to Emotion Mind": (
        "PLEASE skills — reducing physical and lifestyle factors that lower your threshold "
        "for emotional dysregulation and reactivity"
    ),
    "PLEASE Skills": (
        "Physical self-care for emotional resilience — treat PhysicaL illness, balanced "
        "Eating, Avoid mood-Altering substances, balanced Sleep, Exercise"
    ),
    "Build Mastery": (
        "Doing activities that build competence, skill, and a sense of accomplishment "
        "to combat helplessness, hopelessness, and depression"
    ),
    "Accumulate Positive Emotions": (
        "Deliberately increasing positive emotional experiences short-term by doing "
        "pleasant, meaningful, or enjoyable activities each day"
    ),
    "Accumulating Positive Emotions: Long Term": (
        "Building a life worth living by working toward long-term values-based goals, "
        "nurturing relationships, and building positive experiences over time"
    ),
    "Cope Ahead": (
        "Mentally rehearsing a detailed coping plan for an anticipated difficult situation "
        "so you are emotionally prepared when it arrives"
    ),
    "Cope Ahead of Time with Difficult Situations": (
        "Planning and mentally rehearsing skillful responses to expected difficult "
        "emotional situations before they occur"
    ),
    "Changing Emotional Responses": (
        "Using check the facts, opposite action, and problem solving together to "
        "systematically change unwanted emotional experiences"
    ),
    "Emotion Regulation Skills": (
        "Skills for understanding emotions, reducing physiological vulnerability, "
        "and changing difficult emotional states"
    ),
    "Fear": (
        "Fear and anxiety — understanding triggers, body sensations, avoidance urges, "
        "and using approach behavior (opposite action) to reduce fear"
    ),
    "Anger": (
        "Anger and rage — understanding triggers, physiological arousal, attack urges, "
        "and using opposite action (gentle, avoid, be kind) to regulate anger"
    ),
    "Sadness": (
        "Sadness and depression — understanding grief triggers, withdrawal urges, "
        "and using behavioral activation and opposite action to reduce sadness"
    ),
    "Guilt": (
        "Guilt as a signal of actual or perceived values violations — distinguishing "
        "justified guilt (repair, apologize) from unjustified guilt (opposite action)"
    ),
    "Shame": (
        "Shame as a social emotion signaling fear of rejection or judgment — "
        "distinguishing justified from unjustified shame and using opposite action "
        "(disclose, hold head up)"
    ),
    "Disgust": (
        "Disgust and contempt — understanding its protective function, when it is "
        "unjustified, and using opposite action to reduce it"
    ),
    "Envy": (
        "Envy — wanting what others have — understanding its function, when it is "
        "justified, and how to respond skillfully rather than destructively"
    ),
    "Jealousy": (
        "Jealousy — fear of losing something you have — understanding its triggers "
        "and managing it without damaging relationships through controlling behavior"
    ),
    "Love": (
        "Love as an emotion with prompting events, action urges, and aftereffects — "
        "understanding when love needs regulation or skillful expression"
    ),
    "Happiness": (
        "Happiness, joy, and excitement — understanding what builds them, what blocks "
        "them, and how to cultivate positive emotions intentionally"
    ),
    "Mindfulness of Current Emotions: Letting Go of Emotional Suffering": (
        "Observing emotions as passing waves without amplifying (rumination) or "
        "suppressing them — radical acceptance of the emotion itself"
    ),
    "Managing Extreme Emotions": (
        "Skills for when emotional arousal is so extreme it blocks thinking and "
        "effective action — slow down, ground, reduce crisis intensity first"
    ),
    "Overview: Managing Really Difficult Emotions": (
        "Overview of skills for extreme emotional states — mindfulness of emotion, "
        "managing crisis-level arousal, and troubleshooting what isn't working"
    ),
    "Troubleshooting Emotion Regulation Skills": (
        "Diagnosing why emotion regulation skills are not working — identifying the "
        "specific block and finding solutions"
    ),
    "Review of Skills for Emotion Regulation": (
        "Summary map of all emotion regulation skills — naming emotions, checking "
        "facts, opposite action, PLEASE, accumulating positives, and building a life worth living"
    ),

    # --- Distress Tolerance ---
    "Crisis Survival Skills": (
        "Short-term skills to get through a crisis without making it worse — "
        "TIPP (body chemistry), ACCEPTS (distraction), self-soothing, improve the moment"
    ),
    "Distress Tolerance Skills": (
        "Skills for tolerating painful emotions and situations that cannot be immediately "
        "changed, without acting in ways that make things worse"
    ),
    "Distracting with Wise Mind ACCEPTS": (
        "Seven distraction strategies — Activities (engage in tasks), Contributing "
        "(help others), Comparisons, Emotions (induce different emotion), Pushing away, "
        "Thoughts, Sensations"
    ),
    "Self-Soothing": (
        "Calming and comforting yourself through the five senses — find soothing "
        "in vision, hearing, smell, taste, and touch"
    ),
    "Improve the Moment": (
        "IMPROVE skills for surviving crisis — Imagery, Meaning (find purpose), "
        "Prayer, Relaxation, One thing at a time, Vacation (brief mental break), "
        "Encouragement (self-talk)"
    ),
    "Pros and Cons of Acting on Crisis Urges": (
        "Weighing the short and long-term advantages and disadvantages of acting "
        "on vs. tolerating a crisis urge, to choose the most skillful path"
    ),
    "Changing Body Chemistry with TIP Skills": (
        "Rapidly reducing extreme emotional arousal — Temperature (cold water on face), "
        "Intense exercise, Paced breathing, Paired muscle relaxation"
    ),
    "Effective Rethinking and Paired Relaxation": (
        "Identifying stress-causing thoughts and pairing progressive muscle relaxation "
        "with calming self-statements to reduce physiological distress"
    ),
    "Reality Acceptance Skills": (
        "Skills for accepting reality as it is without fighting it — Radical Acceptance, "
        "Turning the Mind, Willingness, Half-Smiling, Willing Hands"
    ),
    "Radical Acceptance": (
        "Completely and fully accepting reality exactly as it is in this moment — "
        "not approving of it or liking it, but no longer fighting against what is true"
    ),
    "Nightmare Protocol": (
        "Step-by-step imagery rehearsal technique for reducing the frequency and "
        "distress of recurring nightmares and trauma-related sleep disturbance"
    ),
    "Sleep Hygiene Protocol": (
        "Behavioral practices for improving sleep quality — regulating sleep schedule, "
        "reducing arousal, avoiding sleep-incompatible behaviors, managing rumination"
    ),
    "Skills When the Crisis Is Addiction": (
        "Distress tolerance adapted for addiction — dialectical abstinence, urge surfing "
        "(riding out cravings), burning bridges, alternative rebellion, adaptive denial"
    ),
    "Practicing the STOP Skill": (
        "Stop, Take a step back, Observe what is happening in yourself and the situation, "
        "Proceed mindfully — a four-step pause before reacting impulsively in crisis"
    ),

    # --- Interpersonal Effectiveness ---
    "Obtaining Objectives Skillfully": (
        "DEAR MAN — Describe, Express, Assert, Reinforce, stay Mindful, Appear confident, "
        "Negotiate — skills for getting what you need from others effectively"
    ),
    "Building Relationships and Ending Destructive Ones": (
        "Skills for finding and building new relationships, deepening existing ones, "
        "and safely ending harmful or destructive relationships"
    ),
    "Walking the Middle Path": (
        "Finding balance between opposing extremes in relationships — validating vs. "
        "pushing for change, autonomy vs. closeness, emotion vs. reason"
    ),
    "Walking the Middle Path: Finding the Synthesis Between Opposites": (
        "Dialectical thinking for navigating conflict — both sides can be partly true, "
        "seek synthesis rather than winning or losing"
    ),
    "Values and Priorities List": (
        "Clarifying personal values to decide which goal to prioritize in an "
        "interpersonal situation — objectives, relationship, or self-respect"
    ),
    "Interpersonal Effectiveness Skills": (
        "Core skills for asking for what you need, saying no, maintaining relationships, "
        "and preserving self-respect in interactions with others"
    ),

    # --- General / Structural ---
    "Analyzing Behavior": (
        "Chain analysis — mapping the exact sequence of vulnerabilities, triggering "
        "events, thoughts, emotions, and behaviors that led to a problem behavior"
    ),
    "General Skills: Orientation and Analyzing Behavior": (
        "Overview of DBT skills structure and behavioral analysis tools — chain analysis "
        "and missing links analysis for understanding problem behaviors"
    ),
    "Orientation": (
        "Introduction to DBT skills training — the biosocial theory of emotional "
        "dysregulation, how to use the skills, and what to expect"
    ),
    "Orientation Handouts": (
        "Reference materials for orienting to DBT skills training structure and purpose"
    ),
    "Introduction to Handouts and Worksheets": (
        "Guide to using DBT handouts (psychoeducation) and worksheets (skill practice) "
        "effectively in treatment"
    ),
    "Introduction to This Book": (
        "Overview of the DBT skills training manual, its purpose, and how to navigate it"
    ),
    "How This Book Is Organized": (
        "Structure of the DBT skills manual — five modules, how handouts and worksheets "
        "are organized, and how to use them"
    ),
    "Numbering of Handouts and Worksheets": (
        "Reference system for navigating handout and worksheet numbers across DBT modules"
    ),
}


def _first_sentence(text: str, max_chars: int = 120) -> str:
    """Extract the first meaningful sentence from a chunk, stripped of markdown."""
    import re
    # Strip markdown bold/italic/headers
    clean = re.sub(r"[*#_`]+", "", text).strip()
    # Collapse newlines and multiple spaces
    clean = re.sub(r"\s+", " ", clean)
    # Split on sentence boundary (". " or ".\n")
    parts = re.split(r"\.\s+", clean, maxsplit=1)
    sentence = parts[0].strip()
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rsplit(" ", 1)[0] + "..."
    return sentence


@dataclass
class RetrievalResult:
    text: str
    handout_id: str
    module: str
    skill_category: str
    content_type: str
    score: float  # cosine similarity: 1 - distance (higher = more relevant)


class KnowledgeRetriever:
    """Retrieves relevant DBT knowledge chunks for a given user message.

    Args:
        collection: A ChromaDB Collection (already configured with an embedding function).
        router: LLMRouter used for query rewriting. The cheap router is appropriate
                since query rewriting is a short, low-stakes completion.
    """

    def __init__(self, collection: chromadb.Collection, router: LLMRouter) -> None:
        self._collection = collection
        self._router = router
        self._vocabulary: str | None = None  # lazy-loaded on first use

    def _get_vocabulary(self) -> str:
        """Build an annotated skill-category vocabulary string.

        Uses hand-written clinical definitions from _SKILL_DEFINITIONS for all known
        categories. For any category not in the dict (new data added later), falls back
        to extracting the first sentence from the shortest definitional chunk in the
        collection.

        Result is cached after first load.
        """
        if self._vocabulary is None:
            # All unique categories currently in the collection
            results = self._collection.get(include=["documents", "metadatas"])
            docs = results["documents"]
            metas = results["metadatas"]
            all_categories = sorted(set(m["skill_category"] for m in metas))

            # Build fallback: shortest skill_description/overview chunk per category
            DEFINITION_TYPES = {"skill_description", "overview"}
            by_category: dict[str, list[str]] = {}
            for doc, meta in zip(docs, metas):
                cat = meta["skill_category"]
                if meta["content_type"] in DEFINITION_TYPES:
                    by_category.setdefault(cat, []).append(doc)

            lines: list[str] = []
            for cat in all_categories:
                if cat in _SKILL_DEFINITIONS:
                    lines.append(f"- {cat}: {_SKILL_DEFINITIONS[cat]}")
                elif cat in by_category:
                    gloss = _first_sentence(min(by_category[cat], key=len), max_chars=120)
                    lines.append(f"- {cat}: {gloss}")
                else:
                    lines.append(f"- {cat}")

            self._vocabulary = "\n".join(lines)
            defined = sum(1 for c in all_categories if c in _SKILL_DEFINITIONS)
            logger.debug(
                "retriever_vocabulary_loaded",
                category_count=len(all_categories),
                hand_defined=defined,
                fallback=len(all_categories) - defined,
            )
        return self._vocabulary

    async def rewrite_query(self, message: str) -> str:
        """Rewrite a raw user message into a DBT-terminology search query via LLM."""
        prompt = RETRIEVAL_QUERY_REWRITE_PROMPT.format(
            vocabulary=self._get_vocabulary(),
            message=message,
        )
        response = await self._router.complete(
            [ChatMessage(role=MessageRole.user, content=prompt)],
            max_tokens=60,
            temperature=0.0,
        )
        rewritten = response.content.strip()
        logger.debug(
            "retrieval_query_rewritten",
            original=message,
            rewritten=rewritten,
        )
        return rewritten

    async def retrieve(self, message: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Rewrite the query then retrieve the top-k most relevant knowledge chunks.

        Args:
            message: Raw user message.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult ordered by descending cosine similarity.
        """
        rewritten = await self.rewrite_query(message)

        raw = self._collection.query(
            query_texts=[rewritten],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        distances = raw["distances"][0]

        results = [
            RetrievalResult(
                text=doc,
                handout_id=meta["handout_id"],
                module=meta["module"],
                skill_category=meta["skill_category"],
                content_type=meta["content_type"],
                score=round(1.0 - dist, 4),
            )
            for doc, meta, dist in zip(docs, metas, distances)
        ]

        logger.debug(
            "retrieval_complete",
            rewritten_query=rewritten,
            top_score=results[0].score if results else None,
            result_count=len(results),
        )
        return results


@lru_cache
def get_retriever() -> KnowledgeRetriever:
    """Singleton factory — wires ChromaDB + cheap LLM router."""
    settings = get_settings()
    openai_api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    ef = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=EMBEDDING_MODEL,
    )

    mode = os.environ.get("CHROMA_MODE", "local").strip().lower()
    if mode == "http":
        url = settings.chroma_url.removeprefix("https://").removeprefix("http://")
        host, port_str = url.rsplit(":", 1) if ":" in url else (url, "8001")
        ssl = settings.chroma_url.startswith("https://")
        client = chromadb.HttpClient(host=host, port=int(port_str), ssl=ssl)
    else:
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        persist_path = str(root / settings.chroma_persist_dir.lstrip("./"))
        client = chromadb.PersistentClient(path=persist_path)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    return KnowledgeRetriever(collection=collection, router=get_cheap_router())
