# 🥊 FightMind AI — Python Model Service (`fightmind-model`)

> **6-Level AI Pipeline** for Boxing, Kickboxing & Karate chatbot  
> **Stack**: Python 3.12 · FastAPI · ChromaDB · Gemini 1.5 Flash · sentence-transformers  
> **Cost**: ₹0 / $0 per month

---

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Implementation Status](#implementation-status)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Running the Service](#running-the-service)
- [Data Setup](#data-setup-required-before-first-run)
- [API Endpoints](#api-endpoints)
- [Logging](#logging)
- [Testing](#testing)
- [Docker](#docker)

---

## Implementation Status

> Updated after every completed step. ✅ = done · 🔄 = in progress · ⬜ = not started

| Step | Description | Status |
|------|-------------|--------|
| 1.1–1.3 | Project skeleton, Dockerfile, CI | ✅ |
| 1.4 | `scraper.py` — Wikipedia + rules + YouTube | ✅ |
| 1.5 | `sports_api.py` — live events (3-source fallback) | ✅ |
| 1.6 | `preprocess.py` — clean + chunk raw data | ✅ |
| 1.7 | `fine_tune.py` — fine-tune embedding model (Colab) | ✅ |
| 1.8 | `vector_store.py` — ChromaDB ingestion | ✅ |
| 1.9 | RAG retrieval accuracy tests | ✅ |
| 1.10–1.16 | 6-level pipeline (L1 intent → L6 validate) | ⬜ |
| 1.17–1.19 | FastAPI endpoints + health check | ⬜ |
| 1.20 | Model accuracy improvements (>95% MRR) | ⬜ |

---

## Overview

This service is the AI brain of FightMind. Every user query flows through a 6-level pipeline:

```
User Input (text / image / both)
    │
    ▼
L1  Intent Detection   — (✅ Done) classify user query using Gemini 2.0 Flash
    ▼
L2  Vision Processing  — (✅ Done) Gemini Vision analyzes image (if provided)
    ▼
L3  RAG Retrieval      — (✅ Done) search ChromaDB knowledge base
    ▼
L4  Live Events        — (✅ Done) inject real match data (event queries only)
    ▼
L5  LLM Generation     — (✅ Done) Gemini 2.0 Flash generates the answer
    ▼
L6  Personalization    — (✅ Done) adapt tone to skill level, update profile
    │
    ▼
Final Response → Java Backend → React Frontend → User
```

---

## Project Structure

```
fightmind-model/
├── logging.yaml                  ← ⚙️  Per-module log level config (edit this!)
├── requirements.txt
├── Dockerfile
├── .env.example
├── docs/
│   └── COLAB_FINE_TUNE_GUIDE.md  ← step-by-step Colab fine-tuning instructions
├── .github/workflows/
│   ├── ci.yml                    ← pytest + flake8 on every push
│   └── cd.yml                    ← auto-deploy to Render on main merge
├── data/
│   ├── raw/                      ← scraped JSON files (gitignored)
│   └── processed/                ← chunked text for embedding (gitignored)
├── embeddings/vectorstore/       ← ChromaDB persistent files (gitignored)
├── models/fine_tuned/            ← saved fine-tuned model weights (gitignored)
├── src/
│   ├── core/
│   │   └── logging_config.py     ← centralized logging (loads logging.yaml)
│   ├── api/
│   │   └── main.py               ← FastAPI app entry point
│   ├── data_collection/
│   │   ├── scraper.py            ← Wikipedia + web + YouTube scraper
│   │   └── sports_api.py         ← TheSportsDB + API-Sports client (3-source fallback)
│   ├── training/
│   │   ├── preprocess.py         ← clean + chunk raw JSON into data/processed/
│   │   ├── fine_tune.py          ← fine-tune embedding model (run on Colab)
│   │   └── evaluate.py           ← benchmark retrieval accuracy
│   └── pipeline/
│       ├── level1_intent.py
│       ├── level2_vision.py
│       ├── level3_rag.py
│       ├── level4_events.py
│       ├── level5_llm.py
│       ├── level6_validate.py
│       └── pipeline_runner.py    ← wires all 6 levels together
└── tests/                        ← pytest test files
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- [Gemini API key](https://aistudio.google.com) (free)

### Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/YOUR_USERNAME/fightmind-model.git
cd fightmind-model

# 2. Create virtual environment (Python 3.12 required)
py -3.12 -m venv .venv        # Windows (use the py launcher)
# python3.12 -m venv .venv    # Mac/Linux

# 3. Activate the virtual environment
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Mac/Linux

# ⚠️  Windows only: if you see "running scripts is disabled", run this once first:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 4. Install dependencies
pip install -r requirements.txt --prefer-binary
# --prefer-binary avoids compiling C/Rust packages from source

# 5. Configure environment
copy .env.example .env          # Windows
# cp .env.example .env          # Mac/Linux
# Then edit .env and fill in your API keys
```

---

## Environment Variables

Copy `.env.example` → `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key |
| `SPORTS_DB_API_KEY` | ✅ Yes | Use `1` for free tier |
| `API_SPORTS_KEY` | Optional | api-sports.io fallback key |
| `ENV` | Optional | `development` (default) or `production` |
| `LOG_LEVEL` | Optional | `DEBUG` / `INFO` / `WARNING` (default: `INFO`) |
| `CORS_ORIGINS` | Optional | Comma-separated allowed origins |

---

## Running the Service

### Development (hot-reload)
```bash
uvicorn src.api.main:app --reload --port 8000
```
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Data Setup (Required Before First Run)

The AI pipeline depends on a local knowledge base built from scraped data.  
These files are **gitignored** (too large for Git) — every developer must generate them locally.

### Step 1 — Collect raw data (`data/raw/`)
```bash
# Takes 5–15 mins — downloads Wikipedia articles, rules pages, YouTube transcripts
python -m src.data_collection.scraper
```

Generates these files:

| File | Contents | Approx. Size |
|------|----------|-------------|
| `data/raw/wikipedia.json` | ~180 Wikipedia articles (boxing, kickboxing, karate techniques, fighters, rules) | ~25 MB |
| `data/raw/official_rules.json` | ~15 rules pages (WBC, WBA, Kumite, Kata, K-1, etc.) | ~1 MB |
| `data/raw/youtube_transcripts.json` | Up to 30 tutorial video transcripts | ~2 MB |
| `data/raw/live_events.json` | Today's scheduled events (3-source fallback) | ~50 KB |

### Step 2 — Fetch live events (`data/raw/live_events.json`)
```bash
# Run any time to refresh today's event data
python -m src.data_collection.sports_api
```

### Step 3 — Process into chunks (`data/processed/`)
```bash
# Cleans + chunks raw text for ChromaDB ingestion
python -m src.training.preprocess
```

### Step 4 — Fine-tune embedding model (Google Colab)
See **[docs/COLAB_FINE_TUNE_GUIDE.md](docs/COLAB_FINE_TUNE_GUIDE.md)** for the full Colab notebook walkthrough.

```bash
# Local quick test (CPU, 1 epoch — just to verify setup):
python -m src.training.fine_tune --epochs 1 --batch-size 8 --max-pairs 500

# Full training → run on Google Colab (free T4 GPU, ~45 min):
# See docs/COLAB_FINE_TUNE_GUIDE.md
```

After training, copy the model files to `models/fine_tuned/`.

### Step 5 — Build ChromaDB vector store (`embeddings/vectorstore/`)
```bash
# Embeds all chunks using your fine-tuned model and saves to disk
python -m src.rag.vector_store
```

> ⚠️ **Run in order:** scraper → sports_api → preprocess → fine_tune → ChromaDB

### Step 6 — Evaluate model accuracy
```bash
# Tests RAG retrieval performance on predefined queries
python -m src.training.evaluate
```


---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok"}` |
| `GET` | `/docs` | Swagger UI (interactive API docs) |
| `POST` | `/pipeline/process` | Main AI pipeline |

### `/pipeline/process` — Request body:
```json
{
  "text":        "What is a jab?",
  "image_url":  "https://...",
  "user_id":    "user-uuid",
  "skill_level": "beginner"
}
```

### `/pipeline/process` — Response:
```json
{
  "answer":       "A jab is a quick straight punch...",
  "confidence":   0.92,
  "sources":     [{"title": "Boxing", "url": "..."}],
  "level_outputs": {}
}
```

---

## Logging

Log levels for each module are controlled by **`logging.yaml`** in the project root.  
Edit that file to change verbosity — no code changes needed.

```yaml
# logging.yaml — example: enable DEBUG for a specific pipeline level
loggers:
  src.pipeline.level3_rag:
    development_level: DEBUG    # ← change this
    production_level:  INFO
```

### Viewing Logs

#### 🖥️ Local Development — human-readable, colored output
```bash
# Default (INFO and above)
uvicorn src.api.main:app --reload

# Override log level with env var
LOG_LEVEL=DEBUG uvicorn src.api.main:app --reload

# Run data scraper
python -m src.data_collection.scraper

# Sample output:
# 2026-03-08 12:30:01 | INFO     | data_collection.scraper  | Wikipedia scraper starting  [topic_count=180]
# 2026-03-08 12:30:02 | WARNING  | data_collection.scraper  | Wikipedia topic not found   [topic=Budokan]
# 2026-03-08 12:30:45 | INFO     | data_collection.scraper  | Wikipedia scraper complete  [collected=174 skipped=6]
```

#### 🐳 Docker (local)
```bash
# Stream all logs
docker-compose logs -f fightmind-model

# Filter to errors only
docker-compose logs -f fightmind-model | findstr "ERROR"    # Windows
docker-compose logs -f fightmind-model | grep ERROR         # Mac/Linux
```

#### ☁️ Production (Render.com) — structured JSON logs
```json
{
  "timestamp": "2026-03-08T06:30:00Z",
  "level": "INFO",
  "service": "fightmind-model",
  "logger": "src.api.main",
  "message": "→ Incoming request",
  "request_id": "a3f2c1b0",
  "path": "/pipeline/process"
}
```

1. **Render Dashboard** → your service → **Logs** tab
2. Filter by `"level":"ERROR"` to see only errors
3. Search by `request_id` to trace a single request across all log lines

#### GitHub Actions (CI logs)
1. GitHub → your repo → **Actions** tab → click a workflow run
2. Expand the **Run tests with pytest** step

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_api.py -v
pytest tests/test_scraper.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

All tests are **offline** — HTTP calls and APIs are mocked so tests pass without internet or API keys.

---

## Docker

```bash
# Build image
docker build -t fightmind-model .

# Run container
docker run -p 8000:8000 --env-file .env fightmind-model

# Or use docker-compose (recommended — starts all services)
cd ..                           # go to project root
docker-compose up fightmind-model
```

---

## 🛠️ Troubleshooting

### Red import underlines in VS Code (`fastapi`, `dotenv`, etc.)
This means VS Code is pointing to the wrong Python interpreter.
1. Press `Ctrl+Shift+P` → **Python: Select Interpreter**
2. Choose **Python 3.12 (`.venv`)** — the one marked *Recommended* inside your project folder
3. Press `Ctrl+Shift+P` → **Developer: Reload Window**

> If you have the **Pyre2** extension installed, uninstall it — it runs its own analysis independently of your interpreter and causes false import errors even when packages are correctly installed.

### `pip install` fails with "Could not find vswhere.exe" or "can't find Rust compiler"
This means pip is trying to **compile a package from source** because no pre-built wheel exists for your Python version. Fixes:
- **Use Python 3.12** — all packages in this project ship pre-built wheels for 3.12. Avoid Python 3.14+ for now (too new for the ML ecosystem).
- **Always use `--prefer-binary`** — tells pip to never compile from source: `pip install -r requirements.txt --prefer-binary`

### `Set-ExecutionPolicy` — PowerShell blocks `.venv\Scripts\activate`
Run this once in PowerShell (allows local scripts, safe for development):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### `python` not found in terminal, but works in PowerShell
Your Python is in your **user PATH** but not the **system PATH**. Use `py -3.12` (the Python Launcher) instead of `python`, or always work within your activated `.venv` where `python` is always defined.

### `ImportError: Using the Trainer with PyTorch requires accelerate>=0.26.0`
The `accelerate` package was not installed. Fix:
```bash
pip install "accelerate>=0.26.0"
# Or reinstall all dependencies (now includes accelerate):
pip install -r requirements.txt --prefer-binary
```

### HuggingFace warning: `machine does not support symlinks`
This is a Windows Developer Mode warning — **not an error**. The model will still download and train correctly, just using slightly more disk space. To suppress the warning permanently:
```powershell
# Option 1: Enable Windows Developer Mode
# Settings → Privacy & Security → For Developers → Developer Mode → ON

# Option 2: Suppress the warning via environment variable
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

### HuggingFace warning: `hf_xet package is not installed`
Also **not an error** — just a performance hint. Downloads fall back to regular HTTP. To install for faster downloads:
```bash
pip install hf_xet
```
