"""
T9.4 — Sports Highlights Briefing
===================================
Streamlit web app that:
  1. Fetches the latest sports news from live RSS feeds (ESPNcricinfo,
     Times of India, Hindustan Times, Indian Express, The Hindu, etc.).
  2. Classifies each article into a sport category (BART / MiniLM /
     DistilBERT / SBERT / ensemble).
  3. Generates three-bullet AI summaries via the Gemini 2.0 Flash API.
  4. Displays everything in a tabbed dashboard with analytics.

Usage: ``streamlit run app.py``

Environment: set ``GEMINI_API_KEY`` locally in ``.env``, or in Streamlit Cloud
Secrets (see README).
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))

import streamlit as st

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.classifier import (
    SPORT_EMOJI,
    classify_batch,
    classify_article_fast,
    group_by_category,
    get_stats,
    load_classifier,
    FINETUNED_DISTILBERT_DIR,
    SBERT_HEAD_PATH,
)
from src.rss_fetcher import fetch_all_articles, make_fallback_articles
from src.summarizer import summarize_article, summarize_batch
from src.few_shot_gemini import disambiguate_batch
from src.multilingual import is_likely_indic, classify_multilingual

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🏏 Sports Highlights Briefing",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Community Cloud: Secrets are not in `.env`; mirror them into the process env
# so `src/summarizer` and `few_shot_gemini` see GEMINI_API_KEY.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", str(st.secrets["GEMINI_API_KEY"]))
except Exception:
    pass

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Card styling */
    .article-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 18px;
        border-left: 4px solid #7c3aed;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .article-card h4 { margin: 0 0 6px 0; font-size: 1.05rem; }
    .article-card .meta { font-size: 0.78rem; color: #aaa; margin-bottom: 10px; }
    .article-card .summary { font-size: 0.9rem; line-height: 1.6; color: #ddd; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-cricket   { background: #1a4a2e; color: #4ade80; }
    .badge-football  { background: #1a3a4a; color: #60a5fa; }
    .badge-tennis    { background: #3a2a1a; color: #fb923c; }
    .badge-hockey    { background: #2a1a3a; color: #c084fc; }
    .badge-kabaddi   { background: #3a1a2a; color: #f472b6; }
    .badge-athletics { background: #1a3a3a; color: #34d399; }
    .badge-racing    { background: #3a3a1a; color: #fbbf24; }
    .badge-other     { background: #2a2a2a; color: #94a3b8; }
    .badge-positive  { background: #052e16; color: #4ade80; }
    .badge-neutral   { background: #1e3a5f; color: #7dd3fc; }
    .badge-negative  { background: #3b0b12; color: #f87171; }
    /* Score bar */
    .conf-bar-wrap { margin: 4px 0 12px 0; }
    .conf-bar { height: 4px; border-radius: 4px; background: #7c3aed; }
    /* Top bar */
    .top-stats { display: flex; gap: 24px; margin-bottom: 20px; }
    .stat-box { background: #1e1e2e; border-radius: 10px; padding: 12px 20px; text-align: center; flex: 1; }
    .stat-box .num { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
    .stat-box .lbl { font-size: 0.72rem; color: #aaa; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "articles" not in st.session_state:
    st.session_state.articles = []
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "pending_summary_id" not in st.session_state:
    st.session_state.pending_summary_id = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🏏 Sports Briefing")
    st.title("⚙️ Settings")
    st.markdown("---")

    # Gemini key is loaded from .env only — not exposed in the UI
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    st.markdown("---")
    available_models = ["facebook/bart-large-mnli",
                       "cross-encoder/nli-MiniLM2-L6-H768",
                       "typeform/distilbart-mnli-12-3"]
    if FINETUNED_DISTILBERT_DIR.exists():
        available_models.append("finetuned-distilbert")
    if SBERT_HEAD_PATH.exists():
        available_models.append("sbert+logreg")
    if FINETUNED_DISTILBERT_DIR.exists() or SBERT_HEAD_PATH.exists():
        available_models.append("ensemble")

    default_idx = available_models.index("ensemble") if "ensemble" in available_models else 0
    model_choice = st.selectbox(
        "Classifier Model",
        options=available_models,
        index=default_idx,
        help=("ensemble = rule -> fine-tuned -> BART (recommended).  "
              "finetuned-distilbert = supervised, fastest accurate option.  "
              "sbert+logreg = tiny supervised baseline.  "
              "BART/MiniLM/DistilBART = zero-shot fallbacks."),
    )

    use_fast_classify = st.toggle(
        "Fast keyword pre-filter",
        value=True,
        help="Use keyword rules for obvious articles; model only for ambiguous ones. Recommended.",
    )

    use_multilingual = st.toggle(
        "Multilingual mode (Hindi/Indic)",
        value=False,
        help="Loads xlm-roberta-large-xnli on demand for Devanagari headlines (~2 GB).",
    )

    use_few_shot = st.toggle(
        "Gemini few-shot rescue (low-confidence)",
        value=False,
        help="Re-classifies articles with confidence < 0.45 via Gemini Flash few-shot.",
    )

    st.markdown("---")
    max_articles = st.slider("Max articles to fetch", 10, 80, 40, step=5)
    summarize_all = st.toggle("Auto-summarize all articles", value=False,
                               help="Calls Gemini for every article. May be slow / use API quota.")

    st.markdown("---")
    refresh_btn = st.button("🔄 Refresh News Feed", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.caption("**T9.4 Sports Highlights Briefing**  \nSMAI Assignment 3 · IIIT Hyderabad")
    st.caption(f"Model: `{model_choice}`")
    if st.session_state.last_fetch:
        st.caption(f"Last fetch: {st.session_state.last_fetch.astimezone(IST).strftime('%H:%M:%S IST')}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🏏 Sports Highlights Briefing")
    st.markdown(
        "_Live sports news from ESPNcricinfo, Cricbuzz, NDTV Sports, Times of India & more — "
        "classified by sport, summarized by AI._"
    )
with col_h2:
    now = datetime.now(IST)
    st.metric("Date (IST)", now.strftime("%d %b %Y"))
    st.metric("Time (IST)", now.strftime("%H:%M"))

st.divider()


# ---------------------------------------------------------------------------
# Helper: render a single article card
# ---------------------------------------------------------------------------


def _badge(text: str, css_class: str) -> str:
    return f'<span class="badge {css_class}">{text}</span>'


def _sport_badge(category: str) -> str:
    cls_map = {
        "Cricket": "badge-cricket",
        "Football": "badge-football",
        "Tennis": "badge-tennis",
        "Hockey": "badge-hockey",
        "Kabaddi": "badge-kabaddi",
        "Athletics": "badge-athletics",
        "Racing": "badge-racing",
    }
    css = cls_map.get(category, "badge-other")
    emoji = SPORT_EMOJI.get(category, "🏅")
    return _badge(f"{emoji} {category}", css)


def _sent_badge(sentiment: str) -> str:
    cls_map = {"Positive": "badge-positive", "Neutral": "badge-neutral", "Negative": "badge-negative"}
    emoji_map = {"Positive": "🟢", "Neutral": "🔵", "Negative": "🔴"}
    css = cls_map.get(sentiment, "badge-neutral")
    emoji = emoji_map.get(sentiment, "⚪")
    return _badge(f"{emoji} {sentiment}", css)


def render_article_card(article: dict, show_summarize_btn: bool = True, key_prefix: str = ""):
    title = article.get("title", "No title")
    url = article.get("url", "#")
    source = article.get("source", "Unknown source")
    published = article.get("published_str", "")
    category = article.get("category") or "Other Sports"
    sentiment = article.get("sentiment") or "Neutral"
    score = article.get("category_score") or 0.0
    ai_summary = article.get("ai_summary") or ""
    is_multi = article.get("multilingual", False)
    is_fewshot = "category_few_shot" in article

    sport_b = _sport_badge(category)
    sent_b = _sent_badge(sentiment)
    extra_b = ""
    if is_multi:
        extra_b += _badge("🌐 Hindi", "badge-other")
    if is_fewshot:
        extra_b += _badge("✨ Few-shot rescue", "badge-other")
    conf_pct = int(score * 100)

    with st.container():
        st.markdown(
            f"""
            <div class="article-card">
                <h4><a href="{url}" target="_blank" style="color:#e2e8f0;text-decoration:none;">{title}</a></h4>
                <div class="meta">
                    {sport_b}{sent_b}{extra_b}
                    &nbsp;📰 {source} &nbsp;·&nbsp; 🕐 {published}
                </div>
                <div class="conf-bar-wrap">
                    <div style="font-size:0.72rem;color:#888;margin-bottom:2px;">
                        Model confidence: {conf_pct}%
                    </div>
                    <div class="conf-bar" style="width:{conf_pct}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if ai_summary:
            st.markdown("**AI Summary**")
            for line in ai_summary.split("\n"):
                if line.strip():
                    st.markdown(line)
        elif show_summarize_btn and gemini_key:
            import hashlib
            unique_key = hashlib.md5((key_prefix + article.get("id", "") + article.get("url", "")).encode()).hexdigest()
            btn_key = f"sum_{unique_key}"
            # If this article was just summarized (from a previous click), show it
            if st.session_state.pending_summary_id == article.get("id"):
                with st.spinner("Generating summary…"):
                    summarize_article(article, api_key=gemini_key)
                st.session_state.pending_summary_id = None
                st.markdown("**AI Summary**")
                for line in (article.get("ai_summary") or "").split("\n"):
                    if line.strip():
                        st.markdown(line)
            else:
                if st.button("✨ Summarize", key=btn_key):
                    st.session_state.pending_summary_id = article.get("id")
                    st.rerun()

        st.markdown("---")


# ---------------------------------------------------------------------------
# Data loading pipeline
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading AI model…")
def get_pipeline(model_name: str):
    return load_classifier(model_name)


def run_pipeline(
    model_name: str, max_art: int, fast: bool, auto_sum: bool,
    multilingual: bool = False, few_shot: bool = False,
):
    """Fetch → classify → (optional rescue) → (optional summary)."""
    progress = st.progress(0, text="Fetching RSS feeds…")

    # 1. Fetch
    articles = fetch_all_articles(total_max=max_art)
    if not articles:
        st.warning("⚠️ No live articles fetched. Using demo/fallback articles.")
        articles = make_fallback_articles()

    progress.progress(15, text=f"Fetched {len(articles)} articles. Loading classifier…")

    # 2. Load model
    pipe = get_pipeline(model_name)
    progress.progress(30, text="Classifying articles…")

    # 3. Classify
    def _classify_progress(i, total):
        pct = 30 + int(40 * i / total)
        progress.progress(pct, text=f"Classifying… {i}/{total}")

    if fast and model_name in {"facebook/bart-large-mnli",
                                "cross-encoder/nli-MiniLM2-L6-H768",
                                "typeform/distilbart-mnli-12-3"}:
        # The fast keyword pre-filter only helps zero-shot models
        for art in articles:
            classify_article_fast(art, pipe=pipe, model_name=model_name)
    else:
        classify_batch(articles, model_name=model_name, progress_callback=_classify_progress)

    progress.progress(70, text="Classification done.")

    # 3b. Multilingual override for Indic-script articles
    if multilingual:
        indic = [a for a in articles if is_likely_indic(a.get("title", ""))]
        if indic:
            progress.progress(72, text=f"Re-classifying {len(indic)} Indic articles…")
            for art in indic:
                try:
                    cat, score = classify_multilingual(art)
                    art["category"] = cat
                    art["category_score"] = round(score, 4)
                    art["multilingual"] = True
                except Exception as exc:
                    logger.warning("Multilingual classify failed: %s", exc)

    # 3c. Gemini few-shot rescue for low-confidence predictions
    if few_shot and gemini_key:
        low_conf = [a for a in articles if (a.get("category_score") or 0) < 0.45]
        if low_conf:
            progress.progress(76, text=f"Few-shot rescue on {len(low_conf)} low-confidence articles…")
            disambiguate_batch(articles, api_key=gemini_key, confidence_threshold=0.45)

    progress.progress(80, text="Classification + rescue done.")

    # 4. Summarize
    if auto_sum and gemini_key:
        progress.progress(82, text="Generating AI summaries (Gemini)…")

        def _sum_progress(i, total):
            pct = 82 + int(15 * i / total)
            progress.progress(pct, text=f"Summarizing… {i}/{total}")

        summarize_batch(articles, api_key=gemini_key, progress_callback=_sum_progress)

    progress.progress(100, text="Done!")
    time.sleep(0.3)
    progress.empty()

    st.session_state.articles = articles
    st.session_state.last_fetch = datetime.now(IST)


# ---------------------------------------------------------------------------
# Trigger fetch on first load or refresh
# ---------------------------------------------------------------------------

if refresh_btn or not st.session_state.articles:
    run_pipeline(
        model_choice, max_articles, use_fast_classify, summarize_all,
        multilingual=use_multilingual, few_shot=use_few_shot,
    )

articles = st.session_state.articles

# ---------------------------------------------------------------------------
# Stats bar
# ---------------------------------------------------------------------------

if articles:
    stats = get_stats(articles)
    groups = group_by_category(articles)
    top_sport = max(stats["categories"], key=stats["categories"].get) if stats["categories"] else "—"
    top_sent = max(stats["sentiments"], key=stats["sentiments"].get) if stats["sentiments"] else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Articles", stats["total"])
    c2.metric("Sport Categories", len(stats["categories"]))
    c3.metric("Top Sport", f"{SPORT_EMOJI.get(top_sport,'🏅')} {top_sport}")
    c4.metric("Dominant Tone", top_sent)
    c5.metric("Sources", len(stats["sources"]))

    st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

TAB_ORDER = [
    "Cricket", "Football", "Tennis", "Hockey", "Kabaddi",
    "Athletics", "Racing", "Badminton", "Golf", "Basketball",
    "Combat Sports", "Other Sports",
]

if articles:
    groups = group_by_category(articles)
    present_cats = [c for c in TAB_ORDER if c in groups]
    leftover = [c for c in groups if c not in TAB_ORDER]
    tab_names = present_cats + leftover + ["📊 All Articles", "📈 Analytics"]

    tab_objects = st.tabs(tab_names)

    # Sport tabs
    for i, cat in enumerate(present_cats + leftover):
        with tab_objects[i]:
            cat_articles = groups[cat]
            emoji = SPORT_EMOJI.get(cat, "🏅")
            st.subheader(f"{emoji} {cat} — {len(cat_articles)} article(s)")

            # Search within tab
            search = st.text_input(
                "🔍 Filter by keyword",
                key=f"search_{cat}",
                placeholder="e.g. Kohli, IPL, injury…",
            )
            filtered = (
                [a for a in cat_articles if search.lower() in (a["title"] + a.get("summary_raw", "")).lower()]
                if search
                else cat_articles
            )

            if not filtered:
                st.info("No articles match your search.")
            else:
                for art in filtered:
                    render_article_card(art, key_prefix=cat)

    # All Articles tab
    all_tab_idx = len(present_cats + leftover)
    with tab_objects[all_tab_idx]:
        st.subheader(f"📰 All {len(articles)} Articles")
        search_all = st.text_input("🔍 Search all articles", key="search_all",
                                    placeholder="Search title or summary…")
        cat_filter = st.multiselect("Filter by category", options=list(groups.keys()),
                                     default=list(groups.keys()), key="cat_filter")
        disp = [
            a for a in articles
            if a.get("category", "Other Sports") in cat_filter
            and (not search_all or search_all.lower() in (a["title"] + a.get("summary_raw", "")).lower())
        ]
        st.caption(f"Showing {len(disp)} / {len(articles)} articles")
        for art in disp:
            render_article_card(art, key_prefix="all")

    # Analytics tab
    analytics_tab_idx = len(present_cats + leftover) + 1
    with tab_objects[analytics_tab_idx]:
        st.subheader("📈 Analytics Dashboard")

        if not articles:
            st.info("No articles loaded yet.")
        else:
            import pandas as pd

            df = pd.DataFrame(articles)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Articles by Sport Category")
                cat_counts = df["category"].fillna("Other Sports").value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                st.bar_chart(cat_counts.set_index("Category"))

            with col2:
                st.markdown("#### Sentiment Distribution")
                sent_counts = df["sentiment"].fillna("Neutral").value_counts().reset_index()
                sent_counts.columns = ["Sentiment", "Count"]
                st.bar_chart(sent_counts.set_index("Sentiment"))

            st.markdown("#### Top 10 Sources")
            src_counts = df["source"].value_counts().head(10).reset_index()
            src_counts.columns = ["Source", "Articles"]
            st.dataframe(src_counts, use_container_width=True, hide_index=True)

            st.markdown("#### Confidence Score Distribution")
            conf_df = df["category_score"].dropna()
            st.line_chart(conf_df.sort_values().reset_index(drop=True))

            st.markdown("#### Raw Article Table")
            display_cols = ["title", "category", "sentiment", "category_score", "source", "published_str"]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[display_cols].rename(columns={
                    "title": "Title",
                    "category": "Sport",
                    "sentiment": "Tone",
                    "category_score": "Confidence",
                    "source": "Source",
                    "published_str": "Published",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Download button
            csv = df[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download as CSV",
                csv,
                file_name=f"sports_news_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

else:
    st.info("Click **Refresh News Feed** in the sidebar to load articles.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "**T9.4 Sports Highlights Briefing** · SMAI Assignment 3 · IIIT Hyderabad  \n"
    "Classifiers: BART-large-MNLI · MiniLM-NLI · DistilBART-MNLI · DistilBERT (fine-tuned) · "
    "SBERT+LogReg · Ensemble · XLM-RoBERTa (multilingual)  \n"
    "Summarizer: Gemini 2.0 Flash · Low-confidence rescue: Gemini few-shot  \n"
    "Sources: ESPNcricinfo · NDTV Sports · Times of India · Hindustan Times · The Hindu · Indian Express"
)
