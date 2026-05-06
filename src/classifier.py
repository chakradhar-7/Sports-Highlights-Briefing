"""
Sports news classifier — T9.4 Sports Highlights Briefing.

Five backends are supported (selectable from the Streamlit UI):

1.  ``rule``                       — keyword rules (fastest, ~0 ms)
2.  ``facebook/bart-large-mnli``   — zero-shot NLI, primary baseline (~370 ms / sample on GPU)
3.  ``cross-encoder/nli-MiniLM2-L6-H768`` — zero-shot NLI, fast ablation (~150 ms)
4.  ``sbert+logreg``               — sentence-transformer + LogReg, supervised (~3 ms)
5.  ``finetuned-distilbert``       — DistilBERT classification head fine-tuned on 137 K
                                     India-headlines labels, supervised (~6 ms / sample)
6.  ``ensemble``                   — rule -> fine-tuned -> BART zero-shot fallback

Plus an optional Gemini few-shot disambiguator for low-confidence cases (#9).

Improvements over v1:
  • Improved hypothesis-template phrasing (+ 8 % over default).
  • Cleaner label set: dropped 'Athletics' / 'Multi-sport Events'
    that had no support in the static eval data.
  • Confidence-threshold fallback to 'Other Sports' when no class
    crosses the threshold (fixes recall = 0.04 for that class).
  • Pluggable supervised backends via :func:`load_classifier`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------

# Final supported categories (display names)
CATEGORIES: list[str] = [
    "Cricket", "Football", "Tennis", "Hockey", "Kabaddi",
    "Racing", "Combat Sports", "Badminton", "Golf",
    "Basketball", "Chess", "Other Sports",
]

# Verbose, discriminative phrasings used as zero-shot candidate labels.
# Each phrasing concatenates the sport name with characteristic terms — this
# alone gives BART-large-MNLI ~3-5 % more accuracy than bare nouns.
SPORT_LABELS_VERBOSE: list[str] = [
    "cricket match, IPL, ODI, Test or T20",
    "association football, soccer, ISL or Premier League",
    "tennis, Grand Slam, ATP, WTA",
    "field hockey, FIH or Hockey India",
    "kabaddi or Pro Kabaddi League",
    "Formula One, MotoGP or motorsport racing",
    "boxing, wrestling, MMA, UFC or martial arts",
    "badminton, BWF, PV Sindhu",
    "golf, PGA Tour or Open Championship",
    "basketball or NBA",
    "chess, FIDE, grandmaster",
    "other sports not listed above",
]

# Mapping verbose label → canonical display name
LABEL_DISPLAY: dict[str, str] = dict(zip(SPORT_LABELS_VERBOSE, CATEGORIES))

# Backwards-compatibility: keep the old simple labels exported for the notebook
SPORT_LABELS: list[str] = SPORT_LABELS_VERBOSE

# Sentiment labels (also verbose for higher accuracy)
SENTIMENT_LABELS: list[str] = [
    "positive news, victory, triumph, achievement",
    "neutral, factual update, transfer, announcement",
    "negative news, defeat, loss, injury or controversy",
]

SENTIMENT_DISPLAY: dict[str, str] = {
    SENTIMENT_LABELS[0]: "Positive",
    SENTIMENT_LABELS[1]: "Neutral",
    SENTIMENT_LABELS[2]: "Negative",
}

SENTIMENT_EMOJI: dict[str, str] = {
    "Positive": "🟢",
    "Neutral": "🔵",
    "Negative": "🔴",
}

SPORT_EMOJI: dict[str, str] = {
    "Cricket": "🏏",
    "Football": "⚽",
    "Tennis": "🎾",
    "Hockey": "🏑",
    "Kabaddi": "🤼",
    "Racing": "🏎️",
    "Combat Sports": "🥊",
    "Badminton": "🏸",
    "Golf": "⛳",
    "Basketball": "🏀",
    "Chess": "♟️",
    "Other Sports": "🏅",
    # Legacy keys (kept for old serialised data):
    "Athletics": "🏃",
}

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

_PIPELINE_CACHE: dict[str, object] = {}

# Confidence threshold below which we fall back to "Other Sports"
OTHER_SPORTS_THRESHOLD: float = 0.35

# Optimal hypothesis template (winner of the ablation study, +8 % over default)
HYPOTHESIS_TEMPLATE: str = "This sports headline covers {}."

# Where the fine-tuned DistilBERT and SBERT-LogReg artefacts live
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FINETUNED_DISTILBERT_DIR = MODELS_DIR / "distilbert_sports"
SBERT_HEAD_PATH = MODELS_DIR / "sbert_logreg.joblib"

# All model identifiers that the app exposes
MODEL_IDS: list[str] = [
    "facebook/bart-large-mnli",
    "cross-encoder/nli-MiniLM2-L6-H768",
    "typeform/distilbart-mnli-12-3",
    "sbert+logreg",
    "finetuned-distilbert",
    "ensemble",
]


def load_classifier(model_name: str = "facebook/bart-large-mnli", device: Optional[int] = None):
    """Load (or fetch from cache) a classifier backend.

    Returns a callable ``predict(text) -> {"labels": [...], "scores": [...]}``
    so all backends share a unified API."""
    if model_name in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_name]

    if model_name == "sbert+logreg":
        backend = _load_sbert_logreg()
    elif model_name == "finetuned-distilbert":
        backend = _load_finetuned_distilbert(device)
    elif model_name == "ensemble":
        backend = _load_ensemble(device)
    else:
        backend = _load_zero_shot_pipeline(model_name, device)

    _PIPELINE_CACHE[model_name] = backend
    return backend


def _load_zero_shot_pipeline(model_name: str, device: Optional[int]):
    import torch
    from transformers import pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    device_label = f"GPU (cuda:{device})" if device >= 0 else "CPU"
    logger.info("Loading zero-shot classifier %s on %s …", model_name, device_label)

    t0 = time.time()
    pipe = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
        multi_label=False,
    )
    logger.info("Loaded %s in %.1fs", model_name, time.time() - t0)
    return pipe


def _load_finetuned_distilbert(device: Optional[int]):
    """Load the fine-tuned DistilBERT classifier saved by ``src/train.py``."""
    if not FINETUNED_DISTILBERT_DIR.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at {FINETUNED_DISTILBERT_DIR}. "
            "Run: python -m src.train"
        )
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    tok = AutoTokenizer.from_pretrained(str(FINETUNED_DISTILBERT_DIR))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(FINETUNED_DISTILBERT_DIR))
    mdl.eval()
    if device >= 0:
        mdl = mdl.cuda()

    id2label = {int(k): v for k, v in mdl.config.id2label.items()}

    @torch.no_grad()
    def predict(text, candidate_labels=None, hypothesis_template=None):
        if isinstance(text, list):
            inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        else:
            inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        if device >= 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # Sort labels by probability so we mirror the HF zero-shot pipeline output
        sorted_idx = probs[0].argsort(descending=True)
        labels = [id2label[int(i)] for i in sorted_idx]
        scores = [float(probs[0][i]) for i in sorted_idx]
        return {"labels": labels, "scores": scores}

    return predict


def _load_sbert_logreg():
    """Load the sentence-transformer + LogisticRegression head."""
    if not SBERT_HEAD_PATH.exists():
        raise FileNotFoundError(
            f"SBERT head not found at {SBERT_HEAD_PATH}. "
            "Run: python -m src.sbert_baseline"
        )
    import joblib
    import numpy as np
    from sentence_transformers import SentenceTransformer

    bundle = joblib.load(SBERT_HEAD_PATH)
    encoder_name = bundle["encoder_name"]
    clf = bundle["clf"]
    classes = bundle["classes"]

    encoder = SentenceTransformer(encoder_name)

    def predict(text, candidate_labels=None, hypothesis_template=None):
        emb = encoder.encode([text] if isinstance(text, str) else text,
                              normalize_embeddings=True, show_progress_bar=False)
        probs = clf.predict_proba(emb)[0]
        order = probs.argsort()[::-1]
        return {
            "labels": [classes[i] for i in order],
            "scores": [float(probs[i]) for i in order],
        }

    return predict


def _load_ensemble(device: Optional[int]):
    """Ensemble: rule → fine-tuned (or sbert) → BART zero-shot fallback."""
    bart = _load_zero_shot_pipeline("facebook/bart-large-mnli", device)
    fine = None
    sbert = None
    if FINETUNED_DISTILBERT_DIR.exists():
        try:
            fine = _load_finetuned_distilbert(device)
        except Exception as exc:
            logger.warning("Fine-tuned head unavailable: %s", exc)
    if SBERT_HEAD_PATH.exists():
        try:
            sbert = _load_sbert_logreg()
        except Exception as exc:
            logger.warning("SBERT head unavailable: %s", exc)

    def predict(text, candidate_labels=None, hypothesis_template=None):
        # 1. Rule-based first pass (fast)
        rule_cat = rule_based_classify(text if isinstance(text, str) else " ".join(text))
        if rule_cat:
            return {"labels": [rule_cat] + [c for c in CATEGORIES if c != rule_cat],
                    "scores": [0.95] + [0.05 / (len(CATEGORIES) - 1)] * (len(CATEGORIES) - 1)}
        # 2. Supervised head (preferred — far higher accuracy than zero-shot)
        if fine is not None:
            return fine(text)
        if sbert is not None:
            return sbert(text)
        # 3. BART zero-shot fallback
        return bart(text, candidate_labels=SPORT_LABELS_VERBOSE,
                    hypothesis_template=HYPOTHESIS_TEMPLATE)

    return predict


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _zero_shot_with_threshold(pipe, text: str) -> tuple[str, float, dict[str, float]]:
    """Run a zero-shot pipeline and apply the Other-Sports threshold fallback."""
    result = pipe(
        text,
        candidate_labels=SPORT_LABELS_VERBOSE,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
    )
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    display = LABEL_DISPLAY.get(top_label, top_label)
    if top_score < OTHER_SPORTS_THRESHOLD:
        display = "Other Sports"
    all_scores = {
        LABEL_DISPLAY.get(lbl, lbl): round(sc, 4)
        for lbl, sc in zip(result["labels"], result["scores"])
    }
    return display, top_score, all_scores


def _supervised_predict(pipe, text: str) -> tuple[str, float, dict[str, float]]:
    """Run a supervised backend that already returns display-name labels."""
    result = pipe(text)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    all_scores = {lbl: round(sc, 4) for lbl, sc in zip(result["labels"], result["scores"])}
    if top_score < OTHER_SPORTS_THRESHOLD and top_label != "Other Sports":
        top_label = "Other Sports"
    return top_label, top_score, all_scores


def classify_article(
    article: dict,
    pipe=None,
    model_name: str = "facebook/bart-large-mnli",
    include_sentiment: bool = True,
) -> dict:
    """Classify a single article in-place. Adds ``category`` / ``sentiment`` keys."""
    if pipe is None:
        pipe = load_classifier(model_name)

    text = _build_input_text(article)

    if model_name in {"sbert+logreg", "finetuned-distilbert", "ensemble"}:
        cat, score, all_scores = _supervised_predict(pipe, text)
    else:
        cat, score, all_scores = _zero_shot_with_threshold(pipe, text)

    article["category"] = cat
    article["category_score"] = round(score, 4)
    article["category_scores_all"] = all_scores

    # Sentiment — always via a zero-shot NLI pipeline (BART or any NLI model).
    if include_sentiment:
        try:
            nli_pipe = pipe if hasattr(pipe, "task") else load_classifier("facebook/bart-large-mnli")
            sent_result = nli_pipe(
                text,
                candidate_labels=SENTIMENT_LABELS,
                hypothesis_template="The tone of this sports news is {}.",
            )
            top_sent = sent_result["labels"][0]
            article["sentiment"] = SENTIMENT_DISPLAY.get(top_sent, top_sent)
            article["sentiment_score"] = round(sent_result["scores"][0], 4)
        except Exception:
            article["sentiment"] = "Neutral"
            article["sentiment_score"] = 0.5
    else:
        article["sentiment"] = "Neutral"
        article["sentiment_score"] = 0.0

    return article


def classify_batch(
    articles: list[dict],
    model_name: str = "facebook/bart-large-mnli",
    include_sentiment: bool = True,
    progress_callback=None,
) -> list[dict]:
    pipe = load_classifier(model_name)
    total = len(articles)
    for i, art in enumerate(articles):
        try:
            classify_article(art, pipe=pipe, model_name=model_name,
                             include_sentiment=include_sentiment)
        except Exception as exc:
            logger.warning("Classification failed for article %d: %s", i, exc)
            art["category"] = "Other Sports"
            art["category_score"] = 0.0
            art["sentiment"] = "Neutral"
            art["sentiment_score"] = 0.0
        if progress_callback:
            progress_callback(i + 1, total)
    return articles


# ---------------------------------------------------------------------------
# Rule-based fast pre-classifier
# ---------------------------------------------------------------------------

KEYWORD_RULES: dict[str, list[str]] = {
    # Football is checked BEFORE Cricket to avoid "isl" / "goal" ambiguity
    "Football": [
        " football", "soccer", " isl ", "isl:", "i-league", " fifa", "premier league",
        "la liga", "bundesliga", "serie a", "champions league", "sunil chhetri",
        "transfer window", "indian super league", "euro cup football",
        "copa america", "world cup football",
    ],
    "Cricket": [
        "cricket", " ipl ", "ipl:", "test match", " odi ", "odi:", " t20 ", "t20i", "bcci",
        "rohit sharma", "virat kohli", "ms dhoni", "shubman gill", "jasprit bumrah",
        "ravindra jadeja", "world cup cricket", " csk ", " rcb ", " kkr ", " srh ",
        "wicket", "runs scored", "century", "six sixes", "stumped", "lbw ", "drs ",
        "bowling spell", "batting average", "innings", "maiden over",
        "world test championship", "asia cup cricket", "cricbuzz", "espncricinfo",
        "test series cricket", "one-day international", "icc cricket",
    ],
    "Tennis": [
        "tennis", "wimbledon", "us open", "french open", "australian open", "atp", "wta",
        "grand slam", "sania mirza", "sumit nagal", "serve", "forehand", "backhand",
    ],
    "Hockey": [
        "hockey", "fih", "hockey world cup", "hockey india", "pr sreejesh",
        "drag flick", "penalty corner",
    ],
    "Kabaddi": ["kabaddi", "pkl", "pro kabaddi"],
    "Racing": [
        "formula one", " f1 ", "grand prix", "motogp", "racing circuit",
        "lewis hamilton", "verstappen", "ferrari f1",
    ],
    "Badminton": ["badminton", "pv sindhu", "saina nehwal", "kidambi srikanth", "bwf"],
    "Golf": ["golf", "pga tour", " masters ", "open championship"],
    "Basketball": ["basketball", " nba ", "nba finals"],
    "Chess": ["chess", "magnus carlsen", "viswanathan anand", "grandmaster", "fide"],
    "Combat Sports": ["boxing", " wrestling", " mma ", " ufc ", " wwe ", "wrestling india"],
}


def rule_based_classify(text: str) -> Optional[str]:
    text_lower = text.lower()
    for category, keywords in KEYWORD_RULES.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return None


def classify_article_fast(
    article: dict,
    pipe=None,
    model_name: str = "facebook/bart-large-mnli",
    run_sentiment: bool = True,
) -> dict:
    """Two-stage classifier: keyword rules first, transformer model if ambiguous."""
    text = _build_input_text(article)
    fast_cat = rule_based_classify(text)

    if fast_cat:
        article["category"] = fast_cat
        article["category_score"] = 0.95
        article["category_scores_all"] = {fast_cat: 0.95}

        if run_sentiment:
            if pipe is None:
                pipe = load_classifier(model_name)
            try:
                sent_result = pipe(
                    text,
                    candidate_labels=SENTIMENT_LABELS,
                    hypothesis_template="The tone of this sports news is {}.",
                )
                top_sent = sent_result["labels"][0]
                article["sentiment"] = SENTIMENT_DISPLAY.get(top_sent, top_sent)
                article["sentiment_score"] = round(sent_result["scores"][0], 4)
            except Exception:
                article["sentiment"] = "Neutral"
                article["sentiment_score"] = 0.5
        else:
            article["sentiment"] = "Neutral"
            article["sentiment_score"] = 0.5
    else:
        classify_article(article, pipe=pipe, model_name=model_name)

    return article


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _build_input_text(article: dict) -> str:
    title = article.get("title", "")
    summary = article.get("summary_raw", "")
    if summary:
        return f"{title}. {summary[:300]}"
    return title


def group_by_category(articles: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for art in articles:
        cat = art.get("category") or "Other Sports"
        groups.setdefault(cat, []).append(art)
    return groups


def get_stats(articles: list[dict]) -> dict:
    from collections import Counter
    cats = Counter(a.get("category", "Unknown") for a in articles)
    sents = Counter(a.get("sentiment", "Unknown") for a in articles)
    sources = Counter(a.get("source", "Unknown") for a in articles)
    return {
        "total": len(articles),
        "categories": dict(cats.most_common()),
        "sentiments": dict(sents.most_common()),
        "sources": dict(sources.most_common(10)),
    }
