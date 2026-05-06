# 🏏 T9.4 Sports Highlights Briefing

**SMAI Assignment 3 · IIIT Hyderabad · 2025–26**

A daily sports-news briefing app that pulls live articles from Indian and international sports RSS feeds, classifies them by sport using zero-shot and optional supervised models, and generates concise AI summaries using the **Gemini 2.0 Flash** API.

---

## Demo

- **Live app (Streamlit Cloud):** [https://sports-highlight-briefing.streamlit.app/](https://sports-highlight-briefing.streamlit.app/)
- **Repository (GitHub):** [https://github.com/chakradhar-7/Sports-Highlights-Briefing/tree/main](https://github.com/chakradhar-7/Sports-Highlights-Briefing/tree/main)

The sidebar **Max articles to fetch** defaults to **15** (slider up to 80) to reduce load on free hosting.

---

## Features

| Feature | Detail |
|---|---|
| **Live RSS fetching** | ESPNcricinfo · NDTV Sports · Times of India · Hindustan Times · The Hindu · Indian Express |
| **Six classifiers** | BART-MNLI · MiniLM-NLI · DistilBART-MNLI · **DistilBERT fine-tuned** · **SBERT + LogReg** · **Ensemble** |
| **Sport categories** | Cricket · Football · Tennis · Hockey · Kabaddi · Racing · Combat Sports · Badminton · Golf · Basketball · Chess · Other Sports |
| **Sentiment analysis** | Positive / Neutral / Negative tone per article |
| **Multilingual** | XLM-RoBERTa-large-XNLI loaded on demand for Hindi/Indic headlines |
| **Few-shot rescue** | Gemini Flash re-classifies low-confidence articles (< 0.45) |
| **AI summaries** | 3-bullet **Gemini 2.0 Flash** summaries |
| **Smart caching** | Summaries cached locally; re-fetches skip already-summarised articles |
| **Analytics tab** | Category/sentiment charts, source breakdown, CSV download |

---

## Project Structure

```
Smai_A3/
├── app.py                            # Main Streamlit application
├── requirements.txt
├── packages.txt                      # apt packages for HF Spaces (libgomp1)
├── README.md
├── src/
│   ├── __init__.py
│   ├── rss_fetcher.py                # RSS fetching & normalisation
│   ├── classifier.py                 # 6 classifier backends + ensemble + threshold fallback
│   ├── summarizer.py                 # Gemini 2.0 Flash summarization + JSON cache
│   ├── train.py                      # Fine-tunes DistilBERT on the 137 K labelled headlines
│   ├── sbert_baseline.py             # SBERT + LogReg supervised baseline
│   ├── few_shot_gemini.py            # Low-confidence Gemini few-shot rescue (#9)
│   ├── multilingual.py               # XLM-RoBERTa Hindi/Indic support (#10)
│   └── live_eval.py                  # Silver-labelled live RSS evaluation set (#8)
├── models/                           # Fine-tuned DistilBERT (**Git LFS**), SBERT+LogReg, metrics
│   ├── distilbert_sports/            # Fine-tuned model
│   ├── distilbert_metrics.json
│   ├── sbert_logreg.joblib
│   ├── sbert_metrics.json
│   └── live_eval_set.json
├── notebooks/
│   ├── T9_4_Analysis_Evaluation.ipynb   # 16 sections: EDA → eval → ablations → live eval
│   └── fig_*.png                        # Saved figures (model comparison, confusion, etc.)
├── app_screenshots/                  # UI screenshots for the report
├── .streamlit/
│   └── config.toml                   # Streamlit dark theme + headless port
├── runtime.txt                       # Python version hint for Streamlit Cloud
└── india-news-headlines.csv          # Static corpus (download from Kaggle, git-ignored)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/sports-highlights-briefing
cd sports-highlights-briefing
pip install -r requirements.txt
# Optional: Jupyter + notebook plots
pip install -r requirements-dev.txt
cp .env.example .env               # Windows: copy .env.example .env
# Edit `.env` and set GEMINI_API_KEY=...
```

### 2. Get a free Gemini API key

Go to [Google AI Studio](https://aistudio.google.com) → **Get API key** (free, no credit card needed).

### 3. (Optional) Train the supervised heads

The Streamlit dropdown automatically lists `finetuned-distilbert`,
`sbert+logreg`, and `ensemble` once these artefacts exist:

```bash
# Fine-tune DistilBERT on 30 K balanced headlines (~6 min on a small GPU)
python -m src.train --max-per-class 2500 --epochs 3

# Sentence-Transformer + LogReg head (~1 min, CPU-friendly)
python -m src.sbert_baseline --max-per-class 2500
```

Both write into `./models/` and the notebook & app pick them up automatically.

### 4. Run the app

```bash
streamlit run app.py
```
Ensure `GEMINI_API_KEY` is set in `.env` or your environment (`export GEMINI_API_KEY=...` on Linux/Mac, `set GEMINI_API_KEY=...` on Windows CMD).

---

## Dataset

**Static evaluation corpus:**  
[India News Headlines (therohk)](https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset)  
21 years of headlines from *The Times of India* (2001–2022), ~3.87 M rows.

Download `india-news-headlines.csv` from Kaggle and place it in the project root.  
The notebook uses it for EDA and model evaluation; it is **not required** to run the app.

---

## Models

| Role | Model | Size | Accuracy* | Latency |
|---|---|---|---|---|
| Rule-based baseline | Keyword regex | — | 0.45 | <1 ms |
| Zero-shot (primary) | `facebook/bart-large-mnli` | 406 M | 0.59 | ~370 ms |
| Zero-shot (fast) | `cross-encoder/nli-MiniLM2-L6-H768` | 35 M | 0.40 | ~150 ms |
| **SBERT + LogReg** (supervised) | `all-MiniLM-L6-v2` + 12-class LogReg | 22 M | _trained locally_ | ~3 ms |
| **DistilBERT fine-tuned** | DistilBERT-base + classification head | 66 M | _trained locally_ | ~6 ms |
| **Ensemble** | rule → DistilBERT → BART fallback | composite | _highest_ | ~6 ms |
| Multilingual (opt-in) | `joeddav/xlm-roberta-large-xnli` | 560 M | — | ~500 ms |
| Summarizer | Gemini **2.0 Flash** | API | — | ~1 s/article |
| Few-shot rescue | Gemini **2.0 Flash** + exemplars | API | — | ~1 s/article |

*Accuracy on a balanced 2 000-sample subset of the India News Headlines test split.
SBERT/DistilBERT/Ensemble numbers are filled in by re-running the notebook after `python -m src.train` and `python -m src.sbert_baseline`.

---

## Notebook

Open `notebooks/T9_4_Analysis_Evaluation.ipynb` (Jupyter / Colab).

Sections:
1. Dataset loading & overview
2. Exploratory Data Analysis (category distribution, temporal trends, word cloud)
3. Ground-truth label construction from fine-grained categories
4. Zero-shot classifier evaluation (BART vs MiniLM vs rule-based)
5. **Ablation study** — hypothesis templates, confidence thresholds, model size vs accuracy
6. Confusion matrix & error analysis
7. Live RSS feed test

---

## Push to GitHub

1. Create a **new empty public repository** on GitHub (no README/license if you will push an existing tree).
2. From the project root:

```bash
git init   # omit if `.git/` already exists
git add .
git commit -m "Initial commit: T9.4 Sports Highlights Briefing"   # omit if nothing to commit
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Train locally. You can ship weights in **either** way:

- **Git LFS (GitHub):** `models/distilbert_sports/model.safetensors` is ~256 MB → commit with [Git LFS](https://git-lfs.com/). `git push` uploads that blob (slow on home Wi‑Fi).

- **Hugging Face Model Hub (recommended for speed):** Upload the folder `models/distilbert_sports/` to a new *Model* repo, then set **`HF_DISTILBERT_SPORTS = "user/repo"`** in Streamlit (or `.env`). The app loads weights from HF on first run (CDN; no huge git/LFS push).

If you use HF for DistilBERT, you can omit the large file from Git entirely; keep small files (e.g. `sbert_logreg.joblib`) in git if you like.

Clone with LFS (only if weights are in GitHub LFS):

```bash
git clone https://github.com/chakradhar-7/Sports-Highlights-Briefing.git
cd Sports-Highlights-Briefing
git lfs pull
```

Streamlit Community Cloud can pull LFS objects, or use `HF_DISTILBERT_SPORTS` in Secrets instead.

**Do not commit** `.env` or `india-news-headlines.csv` (they are listed in `.gitignore`).

---

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (see above).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
3. **New app** → pick the repository → **Main file path:** `app.py` → branch `main` → **Deploy**.
4. **Settings → Secrets** and add (TOML format):

```toml
GEMINI_API_KEY = "your_key_from_aistudio_google_com"
# Optional — skip Git LFS: load fine-tuned DistilBERT from Hugging Face Hub (public model id)
HF_DISTILBERT_SPORTS = "your-hf-username/your-model-repo"
# Only if the HF model repo is private:
# HF_TOKEN = "hf_..."
```

5. **Reboot** the app from the Cloud dashboard so the secret is picked up.

**Notes**

- The app reads `GEMINI_API_KEY` (and optional `HF_DISTILBERT_SPORTS`, `HF_TOKEN`) from Streamlit Secrets (`app.py` after `st.set_page_config`).
- **Supervised DistilBERT:** either commit **`models/distilbert_sports`** with **Git LFS**, or set **`HF_DISTILBERT_SPORTS`** so weights download from Hugging Face — **much faster than waiting on a 256 MB LFS git push**. **SBERT** head (`sbert_logreg.joblib`, ~40 KB) can stay in plain git.
- **Memory:** BART-large-MNLI is large. If the app crashes on startup, select **MiniLM** in the sidebar or upgrade to a paid Streamlit workspace.
- `runtime.txt` pins Python **3.11.9** for reproducible builds.

---

## Deployment on Hugging Face Spaces

```bash
# Create a new Space (Streamlit SDK) on hf.co/new-space
# Add GEMINI_API_KEY as a Secret in Space settings
# Push with:
git remote add hf https://huggingface.co/spaces/<user>/<space-name>
git push hf main
```

---

## Acknowledgements

- **Classifier:** [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) — Yin et al. 2019
- **Summarizer:** Google **Gemini 2.0 Flash** via [Google AI Studio](https://aistudio.google.com)
- **Dataset:** [therohk/india-headlines-news-dataset](https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset)
- **Code assistance:** Claude (Anthropic), used for scaffolding and code review

---

## License

MIT
