"""
Multilingual / Hindi headline support — T9.4 improvement #10.

Indian sports RSS sometimes mixes English and Hindi (e.g. Aaj Tak, NavBharat
Times). XLM-RoBERTa-large is a strong zero-shot NLI checkpoint that handles
~100 languages, including Hindi. We expose a thin wrapper so the Streamlit
app can opt-in via the model dropdown.

Quick language detection is a script-based heuristic: any Devanagari character
flips the article into "needs multilingual model" mode.

Usage::

    from src.multilingual import is_likely_indic, classify_multilingual
    if is_likely_indic(article["title"]):
        cat, score = classify_multilingual(article)

The model is large (~2.2 GB), so we lazy-load only when an Indic article
is encountered.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.classifier import (
    SPORT_LABELS_VERBOSE,
    LABEL_DISPLAY,
    HYPOTHESIS_TEMPLATE,
    OTHER_SPORTS_THRESHOLD,
)

logger = logging.getLogger(__name__)

XLM_MODEL = "joeddav/xlm-roberta-large-xnli"

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_BENGALI_RE = re.compile(r"[\u0980-\u09FF]")

_pipeline_cache: dict[str, object] = {}


def is_likely_indic(text: str) -> bool:
    """Return True if the headline contains Devanagari/Tamil/Bengali characters."""
    if not text:
        return False
    return any(rx.search(text) for rx in (_DEVANAGARI_RE, _TAMIL_RE, _BENGALI_RE))


def _load_xlm_pipeline(device: Optional[int] = None):
    if XLM_MODEL in _pipeline_cache:
        return _pipeline_cache[XLM_MODEL]
    import torch
    from transformers import pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    logger.info("Loading multilingual model %s (this may take a while) …", XLM_MODEL)
    pipe = pipeline(
        "zero-shot-classification",
        model=XLM_MODEL,
        device=device,
        multi_label=False,
    )
    _pipeline_cache[XLM_MODEL] = pipe
    return pipe


def classify_multilingual(article: dict, device: Optional[int] = None) -> tuple[str, float]:
    pipe = _load_xlm_pipeline(device)
    text = article.get("title", "")
    if article.get("summary_raw"):
        text = f"{text}. {article['summary_raw'][:300]}"
    res = pipe(text, candidate_labels=SPORT_LABELS_VERBOSE,
               hypothesis_template=HYPOTHESIS_TEMPLATE)
    top = LABEL_DISPLAY.get(res["labels"][0], res["labels"][0])
    score = float(res["scores"][0])
    if score < OTHER_SPORTS_THRESHOLD:
        top = "Other Sports"
    return top, score
