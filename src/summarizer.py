"""
Gemini-powered summarizer — T9.4 Sports Highlights Briefing.

Uses the google-genai SDK (replaces deprecated google-generativeai).
Model: gemini-2.0-flash (free tier, 1 M tokens/day).
Results are cached to .cache/summaries.json.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash"

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent.parent / ".cache"
SUMMARY_CACHE_FILE = CACHE_DIR / "summaries.json"

_in_memory_cache: dict[str, str] = {}
_cache_loaded = False


def _ensure_cache():
    global _cache_loaded
    if _cache_loaded:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if SUMMARY_CACHE_FILE.exists():
        try:
            with open(SUMMARY_CACHE_FILE, "r", encoding="utf-8") as f:
                _in_memory_cache.update(json.load(f))
        except Exception as exc:
            logger.warning("Could not load summary cache: %s", exc)
    _cache_loaded = True


def _save_cache():
    try:
        with open(SUMMARY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_in_memory_cache, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Could not save summary cache: %s", exc)


# ---------------------------------------------------------------------------
# Gemini client (new google-genai SDK)
# ---------------------------------------------------------------------------

_client = None


def _get_client(api_key: Optional[str] = None):
    global _client
    if _client is not None:
        return _client

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY not set.")

    from google import genai
    _client = genai.Client(api_key=key)
    return _client


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a concise sports news summarizer for an Indian sports briefing app. "
    "Given a sports news article title and body, produce EXACTLY 3 bullet points:\n"
    "• Bullet 1: The main event / result (who, what, score/outcome)\n"
    "• Bullet 2: A key highlight, statistic, or quote\n"
    "• Bullet 3: Context or what happens next\n\n"
    "Rules:\n"
    "- Each bullet must be one sentence, ≤ 25 words.\n"
    "- Use plain English. No jargon. No markdown headers.\n"
    "- Start each bullet with the • character.\n"
    "- Do NOT repeat the title verbatim.\n"
    "- If the article is about cricket, include runs/wickets if available.\n"
)


def _build_prompt(article: dict) -> str:
    title = article.get("title", "")
    body = article.get("summary_raw", "").strip()
    category = article.get("category", "Sports")
    context = (
        f"Title: {title}\nBody: {body}"
        if body
        else f"Title: {title}\n(No body text — infer from title and your knowledge)"
    )
    return f"Sport category: {category}\n{context}\n\nWrite the 3-bullet summary:"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_article(
    article: dict,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    retry_on_rate_limit: bool = True,
) -> dict:
    """Generate an AI summary for a single article (in-place update)."""
    _ensure_cache()

    art_id = article.get("id", "")
    if use_cache and art_id in _in_memory_cache:
        article["ai_summary"] = _in_memory_cache[art_id]
        return article

    if not article.get("title"):
        article["ai_summary"] = "• No content available.\n• —\n• —"
        return article

    try:
        client = _get_client(api_key)
        from google.genai import types
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_build_prompt(article),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=200,
            ),
        )
        summary_text = response.text.strip()
    except EnvironmentError as exc:
        logger.info("Gemini key missing, using fallback: %s", exc)
        summary_text = _fallback_summary(article)
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        if retry_on_rate_limit and "429" in str(exc):
            time.sleep(5)
            try:
                client = _get_client(api_key)
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=_build_prompt(article),
                )
                summary_text = response.text.strip()
            except Exception:
                summary_text = _fallback_summary(article)
        else:
            summary_text = _fallback_summary(article)

    article["ai_summary"] = summary_text
    if use_cache and art_id:
        _in_memory_cache[art_id] = summary_text
        _save_cache()

    return article


def summarize_batch(
    articles: list[dict],
    api_key: Optional[str] = None,
    delay_between_calls: float = 0.5,
    progress_callback=None,
) -> list[dict]:
    total = len(articles)
    for i, art in enumerate(articles):
        if not art.get("ai_summary"):
            summarize_article(art, api_key=api_key)
            if delay_between_calls > 0:
                time.sleep(delay_between_calls)
        if progress_callback:
            progress_callback(i + 1, total)
    return articles


# ---------------------------------------------------------------------------
# Fallback (no API key)
# ---------------------------------------------------------------------------


def _fallback_summary(article: dict) -> str:
    body = article.get("summary_raw", "").strip()
    title = article.get("title", "").strip()

    if not body:
        return (
            f"• {title}\n"
            "• Full details not available in the RSS feed.\n"
            "• Click the headline above to read the full story."
        )

    import re
    sentences = [s.strip() for s in re.split(r"[.!?]+", body) if len(s.strip()) > 20]

    if len(sentences) >= 3:
        b1, b2, b3 = sentences[0], sentences[1], sentences[2]
    elif len(sentences) == 2:
        b1, b2 = sentences[0], sentences[1]
        b3 = "Click the headline to read the full article."
    elif len(sentences) == 1:
        b1 = sentences[0]
        b2 = "Further details are available in the full article."
        b3 = "Click the headline to read the full article."
    else:
        b1 = title
        b2 = body[:120] if body else "No details available."
        b3 = "Visit the source article for more details."

    return f"• {b1}\n• {b2}\n• {b3}"
