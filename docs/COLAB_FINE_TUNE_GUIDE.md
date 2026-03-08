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

# Verify chunks were created
# Expected: data/processed/chunks.json (~10–30 MB)
```

---

## Step-by-Step: Running on Google Colab

### 1. Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### 2. Select T4 GPU Runtime
**Runtime → Change runtime type → T4 GPU → Save**

### 3. Mount Google Drive (to save the model)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Upload `chunks.json` to Google Drive

**Create this folder structure in Google Drive:**
```
My Drive/
└── fightmind_fine_tuned_model/    ← create this folder
    └── chunks.json                ← upload your chunks.json here
```

Steps:
1. Go to [drive.google.com](https://drive.google.com)
2. Create folder **`fightmind_fine_tuned_model`** in My Drive
3. Open that folder and upload `fightmind-model/data/processed/chunks.json` from your local machine

> The model output will be saved to `fightmind_fine_tuned_model/model_output/` (a subfolder), keeping input and output separate.

### 5. Install dependencies
```python
# accelerate is required by sentence-transformers Trainer (model.fit)
!pip install sentence-transformers==3.4.1 "accelerate>=0.26.0" hf_xet --quiet
```

### 6. Clone your repo and install only fine-tuning dependencies
```python
# Clone the repo
!git clone https://github.com/YOUR_USERNAME/fightmind-model.git
%cd fightmind-model

# ⚠️  DO NOT run: pip install -r requirements.txt
# Colab pre-installs pandas, requests, fastapi etc. at specific versions.
# Installing the full requirements.txt overwrites them and causes conflicts.
# Instead, install only the packages that fine_tune.py actually needs:
!pip install sentence-transformers==3.4.1 "accelerate>=0.26.0" pyyaml hf_xet --quiet
```

### 7. Copy chunks.json to expected location
```python
import shutil, os
os.makedirs("data/processed", exist_ok=True)

# chunks.json is in fightmind_fine_tuned_model/ on Drive
shutil.copy("/content/drive/MyDrive/fightmind_fine_tuned_model/chunks.json",
            "data/processed/chunks.json")

print(f"chunks.json ready ✅  size={(os.path.getsize('data/processed/chunks.json')/1024/1024):.1f} MB")
```

### 8. Run fine-tuning
```python
import sys, os
from pathlib import Path

# Ensure Python can find the src package (Colab shell doesn't inherit sys.path)
os.chdir('/content/fightmind-model')
sys.path.insert(0, '/content/fightmind-model')

# Verify setup
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

### 9. Download the model to your machine
After training completes, the model is saved to your Google Drive at:  
`MyDrive/fightmind_fine_tuned_model/model_output/`

Download the **`model_output`** folder and copy its contents to your local repo:
```
fightmind-model/
└── models/
    └── fine_tuned/        ← copy all files from model_output/ here
        ├── config.json
        ├── model.safetensors  (or pytorch_model.bin)
        ├── tokenizer_config.json
        └── tokenizer.json
```

---

## Colab Notebook (copy-paste ready)

Create a new Colab cell for each block below:

**Cell 1 — Setup**
```python
from google.colab import drive
drive.mount('/content/drive')

# Clone the repo
!git clone https://github.com/YOUR_USERNAME/fightmind-model.git
%cd fightmind-model

# ⚠️ Install ONLY fine-tuning packages (not full requirements.txt)
# Colab pre-installs pandas/requests/fastapi at fixed versions - don't overwrite them
!pip install sentence-transformers==3.4.1 "accelerate>=0.26.0" pyyaml hf_xet --quiet
```

**Cell 2 — Copy chunks.json from Drive**
```python
import shutil, os
os.makedirs("data/processed", exist_ok=True)
# chunks.json is in MyDrive/fightmind_fine_tuned_model/ (uploaded in Step 4)
shutil.copy("/content/drive/MyDrive/fightmind_fine_tuned_model/chunks.json",
            "data/processed/chunks.json")
print(f"chunks.json ready ✅  size={(os.path.getsize('data/processed/chunks.json')/1024/1024):.1f} MB")
```

**Cell 3 — Fine-tune**
```python
import sys, os
from pathlib import Path

# Ensure Python can find the src package (Colab shell doesn't inherit sys.path)
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

**Cell 4 — Verify saved model**
```python
import os

# sentence-transformers 3.x may save to the root dir OR a checkpoint subfolder
model_root = "/content/drive/MyDrive/fightmind_fine_tuned_model"

def find_model(base_dir):
    """Search base_dir and one level of subdirs for config.json (model marker)."""
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        print("This means training did not complete or the output path is wrong.")
        return

    print(f"📂 Contents of {base_dir}:")
    for item in os.listdir(base_dir):
        print(f"   {item}")

    # Check subdirectories for config.json (in case Trainer saved to a checkpoint folder)
    for sub in os.listdir(base_dir):
        sub_path = os.path.join(base_dir, sub)
        if os.path.isdir(sub_path) and "config.json" in os.listdir(sub_path):
            print(f"\n✅ Model found in subdirectory: {sub_path}")
            return sub_path

    if "config.json" in os.listdir(base_dir):
        print(f"\n✅ Model found at root: {base_dir}")
        return base_dir

    print("\n⚠️  config.json not found — training may be incomplete. Check Cell 3 output.")

find_model(model_root)
```

---

## Training Configuration Reference

| Parameter | Colab T4 (recommended) | Local CPU (testing) |
|-----------|----------------------|---------------------|
| `--epochs` | 10 | 1 |
| `--batch-size` | 32 | 8 |
| `--max-pairs` | 10 000 | 500 |
| Estimated time | ~45 min | ~10 min (slow) |

---

## After Fine-Tuning

Copy the model files to `fightmind-model/models/fine_tuned/` and update `logging.yaml` if needed:

```yaml
loggers:
  src.training.fine_tune:
    development_level: INFO
    production_level:  WARNING
```

Then proceed to **Step 1.8 — ChromaDB ingestion**, which will automatically use the fine-tuned model from `models/fine_tuned/`.
