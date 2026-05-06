"""
Sentence-Transformer + Logistic Regression baseline — T9.4.

Train a tiny LogReg classifier on top of frozen all-MiniLM-L6-v2 embeddings.
This typically beats zero-shot BART on headline-only data while running
~120 × faster (and the saved model is only ~80 MB total).

Run:
    python -m src.sbert_baseline                 # default 30 K balanced sample
    python -m src.sbert_baseline --max-per-class 5000

Saves:
    models/sbert_logreg.joblib
    models/sbert_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.train import LABEL_MAP, CATEGORIES, load_dataset

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
HEAD_PATH = MODELS_DIR / "sbert_logreg.joblib"
METRICS_PATH = MODELS_DIR / "sbert_metrics.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(message)s")
logger = logging.getLogger(__name__)


def train(
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_per_class: int = 2500,
    val_split: float = 0.1,
    seed: int = 42,
):
    from sentence_transformers import SentenceTransformer

    df = load_dataset(max_per_class=max_per_class, seed=seed)
    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["coarse"], random_state=seed
    )
    logger.info("Train: %d  Val: %d", len(train_df), len(val_df))

    logger.info("Loading encoder %s …", encoder_name)
    encoder = SentenceTransformer(encoder_name)

    t0 = time.time()
    logger.info("Encoding train set …")
    X_train = encoder.encode(
        train_df["headline_text"].tolist(),
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )
    logger.info("Encoding val set …")
    X_val = encoder.encode(
        val_df["headline_text"].tolist(),
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )
    encode_time = time.time() - t0

    y_train = train_df["coarse"].values
    y_val = val_df["coarse"].values

    logger.info("Fitting LogisticRegression …")
    t1 = time.time()
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, C=4.0)
    clf.fit(X_train, y_train)
    fit_time = time.time() - t1

    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average="macro")
    logger.info("\nVal accuracy: %.4f  macroF1: %.4f", acc, macro_f1)
    logger.info("\n%s", classification_report(y_val, preds, zero_division=0, digits=3))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "encoder_name": encoder_name,
        "clf": clf,
        "classes": list(clf.classes_),
        "categories": CATEGORIES,
    }
    joblib.dump(bundle, HEAD_PATH)
    logger.info("Saved head -> %s", HEAD_PATH)

    metrics = {
        "encoder": encoder_name,
        "max_per_class": max_per_class,
        "val_acc": round(acc, 4),
        "val_macro_f1": round(macro_f1, 4),
        "encode_time_s": round(encode_time, 1),
        "fit_time_s": round(fit_time, 1),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "categories": CATEGORIES,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max-per-class", type=int, default=2500)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(encoder_name=args.encoder, max_per_class=args.max_per_class)
