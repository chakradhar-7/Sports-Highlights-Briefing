"""
Live RSS evaluation set — T9.4 improvement #8.

Fetches a snapshot of live sports articles, applies a strong keyword-rule
labeller as silver-standard ground truth, then evaluates all configured
models on that set. This complements the static Kaggle-headlines eval
because the live data has a body field (300+ chars), unlike the static
dataset which is title-only.

The auto-labelled test set is saved to ``models/live_eval_set.json`` so
the notebook can rerun evaluations without re-fetching feeds.

Why "silver" and not "gold"?
  Hand-labelling is costly. The keyword rules cover the obvious cases
  (~70-80 % of articles), and we drop articles where the rule is uncertain.
  This gives us a clean test set of ~25-40 confidently-labelled live articles.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from src.classifier import KEYWORD_RULES, CATEGORIES, rule_based_classify
from src.rss_fetcher import fetch_all_articles, make_fallback_articles

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "models" / "live_eval_set.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(message)s")
logger = logging.getLogger(__name__)


def _strong_keyword_label(text: str) -> str | None:
    """A *stricter* version of the rule-based classifier: require 2 distinct
    keyword hits OR one very specific keyword. Used to build a silver test
    set we can trust for evaluation."""
    text_lower = text.lower()
    hits_per_cat: dict[str, int] = {}
    for cat, kws in KEYWORD_RULES.items():
        n = sum(1 for kw in kws if kw in text_lower)
        if n > 0:
            hits_per_cat[cat] = n
    if not hits_per_cat:
        return None
    # Strongest signal wins; require ≥2 hits OR exactly one hit on the
    # category-defining noun (cricket/football/tennis/...).
    top_cat, top_n = max(hits_per_cat.items(), key=lambda kv: kv[1])
    defining = {
        "Cricket": "cricket", "Football": "football", "Tennis": "tennis",
        "Hockey": "hockey", "Kabaddi": "kabaddi", "Racing": "formula",
        "Combat Sports": "boxing", "Badminton": "badminton",
        "Golf": "golf", "Basketball": "basketball", "Chess": "chess",
    }
    if top_n >= 2:
        return top_cat
    if defining.get(top_cat, "") in text_lower:
        return top_cat
    return None


def build_eval_set(total_max: int = 60) -> list[dict]:
    logger.info("Fetching live articles (max %d) …", total_max)
    articles = fetch_all_articles(total_max=total_max)
    if not articles:
        logger.warning("No live feeds — using fallback samples")
        articles = make_fallback_articles()

    labelled = []
    for art in articles:
        text = art.get("title", "")
        if art.get("summary_raw"):
            text = f"{text}. {art['summary_raw'][:300]}"
        gt = _strong_keyword_label(text)
        if gt is not None:
            entry = {
                "id": art.get("id"),
                "title": art["title"],
                "url": art.get("url", ""),
                "summary_raw": art.get("summary_raw", ""),
                "source": art.get("source", ""),
                "gt_silver": gt,
            }
            labelled.append(entry)
    logger.info("Silver-labelled %d / %d live articles", len(labelled), len(articles))
    logger.info("Distribution: %s", Counter(a["gt_silver"] for a in labelled))

    EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_FILE.write_text(json.dumps(labelled, indent=2, ensure_ascii=False))
    logger.info("Saved to %s", EVAL_FILE)
    return labelled


if __name__ == "__main__":
    build_eval_set()
