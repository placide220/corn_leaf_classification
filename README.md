# 🌽 Corn Leaf Disease Detection — End-to-End MLOps Pipeline

An image classification pipeline that detects **maize leaf diseases** (`Blight`, `Common_Rust`, `Gray_Leaf_Spot`,
`Healthy`) from a photo, served through a FastAPI backend that supports **bulk data upload, automatic and manual
retraining, model versioning and hot-reload**.

| | |
|---|---|
| 🎥 **Video demo** | _add YouTube link_ |
| 🌐 **Live API** | _add deployed URL_ (`/docs` for Swagger) |
| 🖥️ **UI** | _add deployed Streamlit URL_ |

## Project description

Maize is the staple crop across East Africa and the most-traded commodity in the WFP Kenya food-price data I modelled in
my Introduction to ML summative, where the mission was helping smallholder farmers get fair prices. Leaf diseases cut
both yield and grain quality, so a farmer who catches them early protects their income. This project extends that
mission to **images**: a farmer photographs a leaf, the model names the disease, and the system keeps learning from
newly labelled photos.

**Dataset:** [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
(Kaggle), about 4,200 images in 4 classes.

**Model:** MobileNetV2 (ImageNet weights, frozen base) → GlobalAveragePooling → Dropout(0.3) → Dense(4, softmax),
trained with `tf.data` augmentation and balanced class weights. It's small (~10 MB) and fast on CPU, which suits a cheap
cloud container, load testing and retraining inside the API.

## Results (test set)

_Copy from `models/model_metadata.json` after running the notebook._

| Metric | Macro | Weighted |
|---|---|---|
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1-score | | |
| ROC-AUC (OvR) | | |

Plots: `models/eda_features.png`, `models/confusion_matrix.png`, `models/roc_curves.png`, `models/training_curves.png`.

## Repository structure

```
corn_disease_mlops/
├── README.md
├── requirements.txt
├── notebook/
│   └── corn_leaf_classification.ipynb   # acquisition → cleaning → EDA → training → evaluation → prediction
├── src/
│   ├── __init__.py                      # shared paths, class names, thresholds
│   ├── preprocessing.py                 # shared 224×224 MobileNetV2 preprocessing + tf.data pipelines
│   ├── model.py                         # build, evaluate, retrain_model(), check_retrain_trigger()
│   └── prediction.py                    # thread-safe model loader with hot-reload
├── api/
│   └── main.py                          # FastAPI endpoints
├── ui/
│   └── app.py                           # Streamlit dashboard (uptime, visualizations, predict, retrain)
├── data/
│   ├── train/<class>/                   # 80 % (created by the notebook)
│   ├── test/<class>/                    # ≈18–20 %
│   └── uploads/<class>/                 # new labelled images waiting for retraining
└── models/
    ├── corn_disease_model.keras
    ├── model_metadata.json
    ├── eda_stats.json
    └── archive/                         # previous versions, kept on every retrain
```

## Setup

### 1. Train the model in Google Colab

1. Download the Kaggle dataset and rename the zip to `raw_corn_data.zip` (no need to unzip or reorganise it).
2. Open `notebook/corn_leaf_classification.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. When asked, allow Google Drive access, then upload `raw_corn_data.zip`. The notebook:
   * finds the 4 class folders (even if nested), removes corrupt files and exact duplicates, converts everything to
     RGB JPEG, and splits **80 % train / ≈18–20 % test / 5 images per class for the demo**;
   * extracts the 3 EDA features, trains, evaluates, and saves everything to `MyDrive/corn_disease_mlops/`.
4. From Drive, download **`corn_api_artifacts.zip`** (models + train/test data) and **`demo_upload_batch.zip`** (20
   unseen images for the retraining demo).
5. Unzip `corn_api_artifacts.zip` into the repository root so you have `models/corn_disease_model.keras`,
   `data/train/…` and `data/test/…`.

> Re-running the notebook skips the upload step if `data/train` in Drive already has images. Set
> `FORCE_UPLOAD = True` to rebuild the split from scratch.

### 2. Run the API locally

```bash
git clone <your-repo-url> && cd corn_disease_mlops
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs to try every endpoint.

> **TensorFlow version:** `model_metadata.json` records the `tf_version` used in Colab. If loading the model fails,
> install that version or a newer one (`pip install tensorflow==<tf_version>`).

### 3. Run the UI

In a second terminal (with the API still running):

```bash
streamlit run ui/app.py                                   # opens http://localhost:8501
API_URL=https://your-api-url streamlit run ui/app.py      # point it at a deployed API
```

| Page | What it shows |
|---|---|
| 📈 **Model Uptime** | Live status, uptime, model version, test metrics, prediction latency (avg/p95), requests per endpoint, retraining queue. Auto-refreshes every 5 s |
| 📊 **Visualizations** | The 3 dataset features with their interpretations, plus the confusion matrix and per-class F1 of the live model |
| 🔍 **Predict** | Upload one leaf photo → disease, confidence, probability per class, and what it means for the farmer |
| 🔁 **Upload & Retrain** | Bulk upload (zip with class folders, or many images + a label), a **Retrain now** button, live training progress, before/after accuracy and the version history |

## API endpoints

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/health` | Status, uptime, model version, test metrics, prediction latency, request counts, pending uploads |
| GET | `/model-info` | Full metadata: metrics, per-class report, confusion matrix, retraining history |
| GET | `/eda-stats` | The feature statistics and interpretations from `eda_stats.json` (for the UI charts) |
| POST | `/predict` | One image (`file`) → class, confidence, all class probabilities, low-confidence flag |
| POST | `/upload-data` | Bulk upload: many images + a `label` field, **or** a `.zip` with one folder per class. Runs the retraining trigger check |
| POST | `/retrain` | Starts retraining in the background (manual trigger) |
| GET | `/retrain-status` | `idle` / `running` (with epoch progress) / `completed` / `rejected` / `failed` |

```bash
# Predict one image
curl -F "file=@leaf.jpg" http://localhost:8000/predict

# Upload several images of one class
curl -F "files=@a.jpg" -F "files=@b.jpg" -F "label=Common_Rust" http://localhost:8000/upload-data

# Upload the demo batch zip (class taken from folder names); 20 images ≥ 10, so retraining starts automatically
curl -F "files=@demo_upload_batch.zip" http://localhost:8000/upload-data

# Manual retrain and progress
curl -X POST http://localhost:8000/retrain
curl http://localhost:8000/retrain-status
```

## Retraining design

**Triggers** (`src/model.py → check_retrain_trigger`):
* **New data:** ≥ **10** new labelled images waiting in `data/uploads/`; checked after every upload.
* **Model uncertainty:** ≥ **20** predictions with confidence **< 0.65** since the last retrain. This only starts
  retraining when there is also new labelled data, because retraining on unchanged data cannot fix drift. Otherwise
  `/predict` reports that labelled images are needed.
* **Manual:** the `POST /retrain` button.

**Process** (`retrain_model`):
1. Move uploaded images into `data/train/<class>/`.
2. Load the current `.keras` model and fine-tune it on the full training set (90/10 train/validation split, augmentation,
   class weights, early stopping, learning rate 1e-4).
3. Evaluate on the **fixed** `data/test/` set so versions are compared fairly.
4. **Promote only if test accuracy didn't drop by more than 2 points.** The previous model is copied to
   `models/archive/`, the new one is swapped in atomically, and `model_version` goes v1.0 → v1.1 → v1.2 …
5. Every run, promoted or rejected, is logged in `model_metadata.json → retrain_history`.
6. The API **hot-reloads** the new model without restarting. Other containers sharing the `models/` folder pick it up
   on their next request (file-modification check), and a lock file stops two containers retraining at once.

Settings can be changed with environment variables: `RETRAIN_MIN_NEW_IMAGES`, `RETRAIN_LOW_CONF_LIMIT`,
`RETRAIN_EPOCHS`, `RETRAIN_LEARNING_RATE`, `RETRAIN_MAX_ACCURACY_DROP`, `LOW_CONFIDENCE_THRESHOLD`.

## Feature interpretations (EDA)

1. **Class distribution:** Gray Leaf Spot has about half as many images as the other classes, so the model uses
   balanced class weights and macro-averaged metrics.
2. **Chlorophyll vs. necrosis index (mean G − mean R):** healthy leaves are the greenest; rust and blight shift towards
   red/brown as tissue dies. Colour separates healthy from diseased well, but the diseases overlap.
3. **Lesion texture contrast (Laplacian variance):** pustules and small lesions add sharp edges. The overlap and long
   tails show variation in photo quality, which is why augmentation and a pretrained CNN are used.

The full, data-driven interpretations are in the notebook and in `models/eda_stats.json`.

## Flood request simulation (Locust)

_Add results after load testing._

| Containers | Users | Median (ms) | p95 (ms) | Requests/s | Failure % |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |

## Notes

* `data/` holds about 4,000 images (~150 MB). Either commit it (GitHub allows files < 100 MB each) or keep only
  `data/test` plus a download link, because retraining on the server needs `data/train` and `data/test`.
* The model file is ~10 MB, so Git LFS is not needed.
