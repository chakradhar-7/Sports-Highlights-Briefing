"""
Fine-tune DistilBERT for sports news classification — T9.4.

Trains on the labelled subset of the India News Headlines dataset
(``india-news-headlines.csv``) using the same coarse-sport label map as the
notebook. Saves a HuggingFace-format model directory at
``models/distilbert_sports/`` which the Streamlit app can then load.

Run:
    python -m src.train                  # default 30 K balanced sample, 3 epochs
    python -m src.train --max-per-class 5000 --epochs 4

The label set is the canonical 12-category one defined in ``src.classifier``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "india-news-headlines.csv"
MODELS_DIR = ROOT / "models"
OUT_DIR = MODELS_DIR / "distilbert_sports"
METRICS_FILE = MODELS_DIR / "distilbert_metrics.json"

# Same label map as in the notebook (cell 17). Athletics is *intentionally
# omitted* because it had 0 support in the static eval set; "Multi-sport
# Events" is folded into "Other Sports" for cleaner taxonomy.
LABEL_MAP = {
    "Cricket": [
        "cricket", "sports.cricket", "sports.cricket.ipl",
        "sports.cricket.ind-vs-aus", "sports.cricket.ind-vs-eng",
        "sports.cricket.ind-vs-sa", "sports.cricket.ind-vs-pak",
        "sports.cricket.ind-vs-nz", "sports.cricket.ind-vs-wi",
        "sports.cricket.ind-vs-ban", "sports.cricket.ind-vs-sl",
        "sports.cricket.world-cup", "sports.cricket.t20-world-cup",
        "sports.cricket.icc",
    ],
    "Football": [
        "football", "sports.football", "sports.football.epl",
        "sports.football.la-liga", "sports.football.fifa-world-cup",
        "sports.football.indian-football", "sports.football.serie-a",
        "sports.football.bundesliga", "sports.football.champions-league",
        "sports.football.uefa", "sports.football.copa-america",
    ],
    "Tennis": ["tennis", "sports.tennis"],
    "Hockey": ["hockey", "sports.hockey"],
    "Kabaddi": ["sports.kabaddi", "kabaddi"],
    "Racing": ["racing", "sports.racing", "sports.formula-1", "sports.motogp"],
    "Combat Sports": [
        "boxing", "sports.boxing", "wrestling", "sports.wrestling",
        "sports.mixed-martial-arts", "sports.wwe",
    ],
    "Badminton": ["badminton", "sports.badminton"],
    "Golf": ["golf", "sports.golf"],
    "Basketball": ["basketball", "sports.basketball", "sports.nba"],
    "Chess": ["chess", "sports.chess"],
    "Other Sports": [
        "sports.others", "sports.olympics", "sports.commonwealth-games",
        "sports.asian-games", "sports.athletics", "sports.swimming",
        "sports.gymnastics", "sports.weight-lifting",
    ],
}

CATEGORIES = list(LABEL_MAP.keys())
LABEL2ID = {lbl: i for i, lbl in enumerate(CATEGORIES)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}

logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _coarse_label(category_str: str) -> str | None:
    """Map a fine-grained ToI category to one of our 12 coarse labels."""
    cat = (category_str or "").lower()
    for coarse, patterns in LABEL_MAP.items():
        if any(p in cat for p in patterns):
            return coarse
    return None


def load_dataset(max_per_class: int = 2500, seed: int = 42) -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {CSV_PATH}. Download from\n"
            "https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset"
        )

    logger.info("Loading dataset …")
    df = pd.read_csv(CSV_PATH, dtype={"publish_date": str})
    df.columns = [c.strip() for c in df.columns]

    df["coarse"] = df["headline_category"].map(_coarse_label)
    df = df.dropna(subset=["coarse", "headline_text"]).copy()
    df["headline_text"] = df["headline_text"].astype(str).str.strip()
    df = df[df["headline_text"].str.len() > 5]

    logger.info("Labelled rows: %d", len(df))
    logger.info("Per-class counts:\n%s", df["coarse"].value_counts().to_string())

    # Balanced sample
    samples = []
    rng = np.random.default_rng(seed)
    for cat in CATEGORIES:
        sub = df[df["coarse"] == cat]
        n = min(len(sub), max_per_class)
        if n == 0:
            continue
        idx = rng.choice(len(sub), size=n, replace=False)
        samples.append(sub.iloc[idx])
    out = pd.concat(samples, ignore_index=True).sample(frac=1.0, random_state=seed)
    logger.info("Balanced sample: %d rows", len(out))
    return out


# ---------------------------------------------------------------------------
# Torch dataset wrapper
# ---------------------------------------------------------------------------


class HeadlineDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 64):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    base_model: str = "distilbert-base-uncased",
    max_per_class: int = 2500,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 5e-5,
    val_split: float = 0.1,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = load_dataset(max_per_class=max_per_class, seed=seed)
    df["label_id"] = df["coarse"].map(LABEL2ID)

    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["label_id"], random_state=seed
    )
    logger.info("Train: %d  Val: %d", len(train_df), len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model.to(device)

    train_ds = HeadlineDataset(train_df["headline_text"].tolist(),
                                train_df["label_id"].tolist(), tokenizer)
    val_ds = HeadlineDataset(val_df["headline_text"].tolist(),
                              val_df["label_id"].tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(out.loss) * batch["labels"].size(0)
            seen += batch["labels"].size(0)
            if step % 50 == 0:
                logger.info("  epoch %d step %d/%d  loss=%.4f",
                            epoch + 1, step, len(train_loader), running / max(1, seen))
        train_loss = running / max(1, seen)

        # Validation
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
                gts.extend(batch["labels"].tolist())
        acc = accuracy_score(gts, preds)
        macro_f1 = f1_score(gts, preds, average="macro")
        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4),
                        "val_acc": round(acc, 4), "val_macro_f1": round(macro_f1, 4)})
        logger.info("Epoch %d  train_loss=%.4f  val_acc=%.4f  val_macroF1=%.4f",
                    epoch + 1, train_loss, acc, macro_f1)

    elapsed = time.time() - t0
    logger.info("Training done in %.1fs", elapsed)

    # Final classification report
    final_report = classification_report(
        gts, preds, target_names=CATEGORIES, zero_division=0, digits=3
    )
    logger.info("\n%s", final_report)

    # Save artefacts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    metrics = {
        "base_model": base_model,
        "max_per_class": max_per_class,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "history": history,
        "final_val_acc": round(acc, 4),
        "final_val_macro_f1": round(macro_f1, 4),
        "elapsed_s": round(elapsed, 1),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "categories": CATEGORIES,
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved model to %s", OUT_DIR)
    logger.info("Saved metrics to %s", METRICS_FILE)

    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="distilbert-base-uncased")
    p.add_argument("--max-per-class", type=int, default=2500)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        base_model=args.base_model,
        max_per_class=args.max_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
