"""Corn (maize) leaf disease MLOps pipeline package.

Shared paths and constants live here so the notebook, API, retraining code and
tests all agree on where artifacts are stored.
"""

import os
from pathlib import Path

# Project root = parent of this src/ folder. Override with the PROJECT_ROOT env var
# (useful inside Docker, where data/ and models/ may be mounted volumes).
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"          # newly uploaded, labelled images waiting for retraining

MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
MODEL_PATH = MODELS_DIR / "corn_disease_model.keras"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EDA_STATS_PATH = MODELS_DIR / "eda_stats.json"
ARCHIVE_DIR = MODELS_DIR / "archive"       # previous model versions are kept here

CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
IMG_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", 0.65))

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "TRAIN_DIR", "TEST_DIR", "UPLOAD_DIR",
    "MODELS_DIR", "MODEL_PATH", "METADATA_PATH", "EDA_STATS_PATH", "ARCHIVE_DIR",
    "CLASS_NAMES", "IMG_SIZE", "LOW_CONFIDENCE_THRESHOLD",
]
