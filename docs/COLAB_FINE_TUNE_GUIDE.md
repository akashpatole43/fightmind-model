# FightMind AI — Google Colab Fine-Tuning Guide

> **Purpose:** Fine-tune the `all-MiniLM-L6-v2` embedding model on domain-specific
> martial arts text to maximise RAG retrieval accuracy.  
> **Runtime:** Google Colab — Free T4 GPU (~45 min for 10 epochs on ~10k pairs)  
> **Cost:** ₹0 / $0

---

## Why Fine-Tune?

The base `all-MiniLM-L6-v2` model was trained on general English text.
After fine-tuning on martial arts chunks:

| Query | Base model top result | Fine-tuned top result |
|-------|-----------------------|----------------------|
| "How do I throw a jab?" | Generic punch article | Jab technique tutorial ✅ |
| "What is ippon in kumite?" | General sports scoring | Karate kumite rules ✅ |
| "Explain low kick" | Generic leg strike | Kickboxing low kick technique ✅ |

---

## Prerequisites (run once locally)

Make sure `data/processed/chunks.json` exists:
```bash
# Step 1 — collect raw data (if not done already)
python -m src.data_collection.scraper

# Step 2 — preprocess into chunks
python -m src.training.preprocess

# Verify chunks were created — expected: data/processed/chunks.json (~10–30 MB)
```

---

## Step-by-Step: Running on Google Colab

### 1. Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### 2. Select T4 GPU Runtime
**Runtime → Change runtime type → T4 GPU → Save**

### 3. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Upload `chunks.json` to Google Drive

**Create this folder structure in Google Drive:**
```
My Drive/
└── fightmind_fine_tuned_model/    ← create this folder
    └── chunks.json                ← upload your local data/processed/chunks.json here
```

1. Go to [drive.google.com](https://drive.google.com)
2. Create folder **`fightmind_fine_tuned_model`** in My Drive
3. Upload `fightmind-model/data/processed/chunks.json` into that folder

> ℹ️ The model will be saved to `fightmind_fine_tuned_model/model_output/` — separate from chunks.json.

### 5. Clone repo and install fine-tuning packages
```python
# Clone the repo — provides the src/ package used in Step 7
!git clone https://github.com/YOUR_USERNAME/fightmind-model.git

# ⚠️  DO NOT run: pip install -r requirements.txt
# Colab pre-installs pandas, requests, fastapi etc. at fixed versions.
# Installing the full requirements.txt conflicts with Colab's environment.
# Only install what fine_tune.py needs:
!pip install sentence-transformers==3.4.1 "accelerate>=0.26.0" pyyaml hf_xet --quiet
```

### 6. Copy chunks.json to the repo
```python
import shutil, os

# MUST set working directory first — otherwise chunks.json lands in /content/
# instead of /content/fightmind-model/ where fine_tune.py looks for it
os.chdir('/content/fightmind-model')
os.makedirs("data/processed", exist_ok=True)

shutil.copy("/content/drive/MyDrive/fightmind_fine_tuned_model/chunks.json",
            "data/processed/chunks.json")

print(f"chunks.json ready ✅  {os.path.getsize('data/processed/chunks.json')/1024/1024:.1f} MB")
print(f"Saved at: {os.path.abspath('data/processed/chunks.json')}")
```

### 7. Run fine-tuning
```python
import sys, os
from pathlib import Path

# Make src/ importable — Colab shell doesn't inherit sys.path from %cd
os.chdir('/content/fightmind-model')
sys.path.insert(0, '/content/fightmind-model')

# Sanity check
print("Working dir:", os.getcwd())
print("fine_tune.py exists:", os.path.exists('src/training/fine_tune.py'))

from src.core.logging_config import setup_logging
from src.training.fine_tune import fine_tune

setup_logging()
fine_tune(
    epochs=10,
    batch_size=32,
    max_pairs=10_000,
    output_dir=Path('/content/drive/MyDrive/fightmind_fine_tuned_model/model_output')
)
```

**Expected output:**
```
INFO | Fine-tuning started  [pairs=..., epochs=10, batch_size=32]
Epoch 1/10: 100%|████████| 264/264 [04:12<00:00]
...
INFO | Fine-tuning complete — model saved  [output=.../model_output]
```

### 8. Verify saved model
```python
import os
model_output = "/content/drive/MyDrive/fightmind_fine_tuned_model/model_output"

if os.path.exists(model_output):
    print(f"✅ Model saved at: {model_output}")
    for f in os.listdir(model_output):
        print(f"   {f}")
else:
    print("❌ model_output not found — check Step 7 output for errors")
```

### 9. Download the model to your machine
Model is saved at: `MyDrive/fightmind_fine_tuned_model/model_output/`

Download the **`model_output`** folder and copy its contents to your local repo:
```
fightmind-model/
└── models/
    └── fine_tuned/        ← copy all ROOT files from model_output/ here (overwrite)
        ├── config.json
        ├── model.safetensors
        ├── tokenizer_config.json
        ├── tokenizer.json
        ├── vocab.txt
        ├── modules.json
        └── 1_Pooling/
```

> ℹ️ The `checkpoints/` subfolder inside `model_output/` holds intermediate epoch saves — you can safely delete it.

---

## Colab Notebook (copy-paste ready)

**Cell 1 — Setup**
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/fightmind-model.git

# Only install fine-tuning packages — don't run pip install -r requirements.txt
!pip install sentence-transformers==3.4.1 "accelerate>=0.26.0" pyyaml hf_xet --quiet
```

**Cell 2 — Copy chunks.json**
```python
import shutil, os

# MUST set working directory first — otherwise chunks.json lands in /content/
# instead of /content/fightmind-model/ where fine_tune.py looks for it
os.chdir('/content/fightmind-model')
os.makedirs("data/processed", exist_ok=True)

shutil.copy("/content/drive/MyDrive/fightmind_fine_tuned_model/chunks.json",
            "data/processed/chunks.json")

print(f"chunks.json ready ✅  {os.path.getsize('data/processed/chunks.json')/1024/1024:.1f} MB")
print(f"Saved at: {os.path.abspath('data/processed/chunks.json')}")
```

**Cell 3 — Fine-tune**
```python
import sys, os
from pathlib import Path

os.chdir('/content/fightmind-model')
sys.path.insert(0, '/content/fightmind-model')

print("Working dir:", os.getcwd())
print("fine_tune.py exists:", os.path.exists('src/training/fine_tune.py'))

from src.core.logging_config import setup_logging
from src.training.fine_tune import fine_tune

setup_logging()
fine_tune(
    epochs=10,
    batch_size=32,
    max_pairs=10_000,
    output_dir=Path('/content/drive/MyDrive/fightmind_fine_tuned_model/model_output')
)
```

**Cell 4 — Verify**
```python
import os
model_output = "/content/drive/MyDrive/fightmind_fine_tuned_model/model_output"

if os.path.exists(model_output):
    print(f"✅ Model saved:")
    for f in os.listdir(model_output):
        print(f"   {f}")
else:
    print("❌ model_output not found — check Cell 3 for errors")
```

---

## Training Configuration Reference

| Parameter | Colab T4 (recommended) | Local CPU (smoke test) |
|-----------|----------------------|------------------------|
| `epochs` | 10 | 1 |
| `batch_size` | 32 | 8 |
| `max_pairs` | 10 000 | 500 |
| Estimated time | ~45 min | ~10 min |

> Parameters are passed as Python kwargs to `fine_tune()` in Cell 3, not as CLI flags.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named src.core` | `sys.path` not set | Add `sys.path.insert(0, '/content/fightmind-model')` before imports |
| `fine_tune.py exists: False` | Old clone (repo not re-cloned after push) | Delete old clone: `shutil.rmtree('/content/fightmind-model')` then re-run Cell 1 |
| Dependency conflicts on `pip install -r requirements.txt` | Colab has pinned versions of pandas/requests/fastapi | Don't use `requirements.txt` in Colab — install only fine-tuning packages as shown |
| `accelerate>=0.26.0` not found | Not installed | Add to Cell 1: `!pip install "accelerate>=0.26.0" --quiet` |

---

## After Fine-Tuning

1. Download `model_output/` from Google Drive
2. Copy all files **except** `checkpoints/` into `fightmind-model/models/fine_tuned/` (overwrite existing)
3. Verify locally:
```bash
# Should show config.json, model.safetensors, tokenizer files etc.
dir models\fine_tuned\
```

Then proceed to **Step 1.8 — ChromaDB ingestion**, which will automatically load the model from `models/fine_tuned/`.
