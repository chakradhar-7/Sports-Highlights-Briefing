"""
RSS feed fetcher for sports news — T9.4 Sports Highlights Briefing.

Fetches the latest articles from multiple cricket / sports RSS feeds,
normalises them into a common dict schema, and de-duplicates by URL.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import feedparser
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feed registry
# ---------------------------------------------------------------------------

SPORTS_FEEDS: dict[str, list[dict]] = {
    "Cricket": [
        {
            "name": "ESPNcricinfo – Top Stories",
            "url": "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
        },
        {
            "name": "Times of India – Cricket",
            "url": "https://timesofindia.indiatimes.com/rssfeeds/54829575.cms",
        },
        {
            "name": "Hindustan Times – Cricket",
            "url": "https://www.hindustantimes.com/feeds/rss/cricket/rssfeed.xml",
        },
    ],
    "Football": [
        {
            "name": "Times of India – Football",
            "url": "https://timesofindia.indiatimes.com/rssfeeds/30359486.cms",
        },
    ],
    "General Sports": [
        {
            "name": "Times of India – Sports",
            "url": "https://timesofindia.indiatimes.com/rssfeeds/4719161.cms",
        },
        {
            "name": "Hindustan Times – Sports",
            "url": "https://www.hindustantimes.com/feeds/rss/sports/rssfeed.xml",
        },
        {
            "name": "The Hindu – Sport",
            "url": "https://www.thehindu.com/sport/feeder/default.rss",
        },
        {
            "name": "Indian Express – Sports",
            "url": "https://indianexpress.com/section/sports/feed/",
        },
        {
            "name": "NDTV Sports",
            "url": "https://sports.ndtv.com/rss/all",
        },
    ],
}

# Flat list for convenience
ALL_FEEDS: list[dict] = [
    {**feed, "sport_hint": sport}
    for sport, feeds in SPORTS_FEEDS.items()
    for feed in feeds
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(entry: feedparser.FeedParserDict) -> Optional[datetime]:
    """Return a timezone-aware datetime from an RSS entry, or None."""
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def _entry_to_article(entry: feedparser.FeedParserDict, source_name: str, sport_hint: str) -> dict:
    """Normalise a feedparser entry into our article schema."""
    summary_raw = getattr(entry, "summary", "") or ""
    # Strip HTML tags from summary
    import re
    summary_clean = re.sub(r"<[^>]+>", "", summary_raw).strip()

    return {
        "id": hashlib.md5((entry.get("link", "") or entry.get("title", "")).encode()).hexdigest(),
        "title": entry.get("title", "").strip(),
        "url": entry.get("link", ""),
        "summary_raw": summary_clean[:800],  # cap at 800 chars
        "source": source_name,
        "sport_hint": sport_hint,
        "published": _parse_date(entry),
        "published_str": entry.get("published", entry.get("updated", "")),
        "category": None,       # filled by classifier
        "category_score": None,
        "ai_summary": None,     # filled by summarizer
    }


def _fetch_single_feed(feed: dict, timeout: int = 10) -> list[dict]:
    """Fetch one RSS feed and return parsed articles."""
    articles: list[dict] = []
    try:
        response = requests.get(feed["url"], timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
        for entry in parsed.entries:
            article = _entry_to_article(entry, feed["name"], feed.get("sport_hint", "General Sports"))
            if article["title"]:
                articles.append(article)
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", feed["name"], exc)
    return articles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_all_articles(
    max_per_feed: int = 20,
    total_max: int = 60,
    deduplicate: bool = True,
) -> list[dict]:
    """
    Fetch articles from all registered sports feeds.

    Parameters
    ----------
    max_per_feed : int
        Maximum articles to take from each feed.
    total_max : int
        Hard cap on total articles returned.
    deduplicate : bool
        Remove duplicate articles by URL hash.

    Returns
    -------
    list[dict]
        Articles sorted newest-first (when date is available).
    """
    all_articles: list[dict] = []
    seen_ids: set[str] = set()

    for feed in ALL_FEEDS:
        raw = _fetch_single_feed(feed)
        for art in raw[:max_per_feed]:
            if deduplicate and art["id"] in seen_ids:
                continue
            seen_ids.add(art["id"])
            all_articles.append(art)
        time.sleep(0.1)  # be polite

    # Sort newest first; articles without dates go to the end
    all_articles.sort(
        key=lambda a: a["published"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return all_articles[:total_max]


def fetch_articles_by_sport(sport: str, max_articles: int = 20) -> list[dict]:
    """Fetch articles from feeds tagged with a specific sport."""
    feeds = SPORTS_FEEDS.get(sport, [])
    articles: list[dict] = []
    seen_ids: set[str] = set()
    for feed_info in feeds:
        feed = {**feed_info, "sport_hint": sport}
        for art in _fetch_single_feed(feed)[:max_articles]:
            if art["id"] not in seen_ids:
                seen_ids.add(art["id"])
                articles.append(art)
    articles.sort(
        key=lambda a: a["published"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return articles[:max_articles]


def make_fallback_articles() -> list[dict]:
    """
    Return a small set of hardcoded sample articles for offline / demo mode.
    These are used when no live feeds can be reached.
    """
    samples = [
        {
            "title": "India beat Australia by 6 wickets in 2nd ODI",
            "url": "https://www.espncricinfo.com/sample",
            "summary_raw": "India clinched the second ODI against Australia with a commanding 6-wicket victory. Virat Kohli top-scored with 82* while Shubman Gill contributed a quick 54.",
            "source": "Demo / Fallback",
            "sport_hint": "Cricket",
        },
        {
            "title": "Rohit Sharma named T20 World Cup squad captain",
            "url": "https://www.cricbuzz.com/sample",
            "summary_raw": "The BCCI selection committee announced the T20 World Cup squad with Rohit Sharma retaining captaincy. Jasprit Bumrah will lead the pace attack.",
            "source": "Demo / Fallback",
            "sport_hint": "Cricket",
        },
        {
            "title": "ISL: Mumbai City FC storm to top of table",
            "url": "https://timesofindia.com/sample",
            "summary_raw": "Mumbai City FC registered their fifth consecutive ISL win, moving to the top of the standings with a 2-0 victory over Bengaluru FC.",
            "source": "Demo / Fallback",
            "sport_hint": "Football",
        },
        {
            "title": "Saina Nehwal announces retirement from professional badminton",
            "url": "https://ndtvsports.com/sample",
            "summary_raw": "Indian badminton legend Saina Nehwal formally announced her retirement citing recurring knee injuries.",
            "source": "Demo / Fallback",
            "sport_hint": "General Sports",
        },
        {
            "title": "PV Sindhu wins silver at All England Championships",
            "url": "https://thehindu.com/sample",
            "summary_raw": "PV Sindhu put up a spirited performance at the All England Championships but had to settle for silver after a hard-fought final loss.",
            "source": "Demo / Fallback",
            "sport_hint": "General Sports",
        },
    ]
    import hashlib
    from datetime import datetime, timezone
    articles = []
    for i, s in enumerate(samples):
        art = {
            "id": hashlib.md5(s["url"].encode()).hexdigest(),
            "title": s["title"],
            "url": s["url"],
            "summary_raw": s["summary_raw"],
            "source": s["source"],
            "sport_hint": s["sport_hint"],
            "published": datetime.now(timezone.utc),
            "published_str": "Demo article",
            "category": None,
            "category_score": None,
            "ai_summary": None,
        }
        articles.append(art)
    return articles
