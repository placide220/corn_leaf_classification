"""FastAPI backend for the corn leaf disease classifier.

Run from the repository root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import io
import json
import socket
import sys
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from src import CLASS_NAMES, EDA_STATS_PATH, UPLOAD_DIR, database  # noqa: E402
from src.model import (check_retrain_trigger, count_pending_uploads, get_retrain_status,  # noqa: E402
                       is_retraining, start_retraining_async)
from src.preprocessing import (is_image_file, md5_bytes, normalize_class_name,  # noqa: E402
                               save_as_jpeg, standardize_image_bytes)
from src.leaf_check import NotMaizeLeafError  # noqa: E402
from src.prediction import get_model_manager  # noqa: E402

MAX_FILE_BYTES = 10 * 1024 * 1024        # 10 MB per image
MAX_ZIP_BYTES = 200 * 1024 * 1024        # 200 MB per zip

START_TIME = time.time()
# Container/host name. With several Docker replicas behind nginx this shows WHICH replica answered.
INSTANCE_ID = socket.gethostname()
REQUEST_STATS: Dict[str, Any] = {"total_requests": 0, "errors": 0, "by_endpoint": {}}
manager = get_model_manager()
startup_error: Optional[str] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global startup_error
    import threading
    threading.Thread(target=_dataset_counter_loop, name="dataset-counter", daemon=True).start()
    try:
        manager.load()
    except Exception as exc:  # noqa: BLE001 - keep the API up so /health can report the problem
        startup_error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(
    title="Corn Leaf Disease Classifier API",
    description="Custom CNN classifier for Blight, Common Rust, Gray Leaf Spot and Healthy maize leaves, "
                "with bulk upload and retraining.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _openapi_with_file_pickers():
    """Make the /docs page show a "Choose File" button for every upload field.

    Newer FastAPI versions describe uploads as {"type": "string", "contentMediaType": ...}.
    Swagger UI turns that into a file picker for a SINGLE file, but for a LIST of files
    (/upload-data) it shows plain text boxes instead, so a zip gets pasted in as text and the
    upload fails. Adding the older "format": "binary" marker fixes the picker in Swagger.
    """
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)

    def add_binary_format(node):
        if isinstance(node, dict):
            if node.get("type") == "string" and node.get("contentMediaType") == "application/octet-stream":
                node["format"] = "binary"
            for value in node.values():
                add_binary_format(value)
        elif isinstance(node, list):
            for value in node:
                add_binary_format(value)

    add_binary_format(schema)
    app.openapi_schema = schema
    return schema


app.openapi = _openapi_with_file_pickers


@app.middleware("http")
async def count_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        REQUEST_STATS["errors"] += 1
        raise
    path = request.url.path
    ep = REQUEST_STATS["by_endpoint"].setdefault(path, {"count": 0, "total_ms": 0.0})
    ep["count"] += 1
    ep["total_ms"] += (time.perf_counter() - start) * 1000
    REQUEST_STATS["total_requests"] += 1
    if response.status_code >= 500:
        REQUEST_STATS["errors"] += 1
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    response.headers["X-Instance"] = INSTANCE_ID
    return response


def _on_retrain_complete(result: Dict[str, Any]) -> None:
    if result.get("promoted"):
        manager.reload()                       # hot-swap the new model in this process


def _auto_trigger(source: str) -> Dict[str, Any]:
    should, reason = check_retrain_trigger(manager.low_confidence_count)
    started = False
    if should:
        started = start_retraining_async(trigger=f"auto ({source}): {reason}", on_complete=_on_retrain_complete)
    return {"should_retrain": should, "started": started, "reason": reason}


_DATA_COUNTS: Dict[str, Any] = {"value": None, "updated": None}


def _count_dataset() -> Dict[str, Any]:
    """How many images the SERVER has in data/train and data/test, per class."""
    from src import TEST_DIR, TRAIN_DIR
    from src.preprocessing import list_image_files

    out = {}
    for split, folder in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        _, labels = list_image_files(folder)
        out[split] = {"total": len(labels), "per_class": {c: labels.count(i) for i, c in enumerate(CLASS_NAMES)},
                      "folder": str(folder)}
    return out


def _dataset_counter_loop() -> None:
    """Background thread: recount the dataset every 60 s.
    Listing ~4,000 files can take seconds on a Docker volume (especially on Windows). Doing that
    inside /health would freeze the whole API during the listing, so /health only reads this
    cached value and never touches the disk itself."""
    while True:
        try:
            _DATA_COUNTS["value"] = _count_dataset()
            _DATA_COUNTS["updated"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:  # noqa: BLE001
            _DATA_COUNTS["value"] = {"error": str(exc)}
        time.sleep(60)


def _data_counts() -> Dict[str, Any]:
    value = _DATA_COUNTS["value"]
    if value is None:
        return {"status": "counting..."}
    return {**value, "counted_at": _DATA_COUNTS["updated"]}


def _db_summary() -> Dict[str, Any]:
    try:
        return database.upload_summary()
    except Exception as exc:  # noqa: BLE001 - /health must answer even if the database has a problem
        return {"error": str(exc)}


def _format_uptime(seconds: float) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}d {h}h {m}m {s}s"


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
async def root():
    return {"message": "Corn Leaf Disease Classifier API", "docs": "/docs",
            "endpoints": ["/health", "/model-info", "/eda-stats", "/predict", "/upload-data", "/uploads",
                          "/retrain", "/retrain-status", "/retrain-runs"]}


# The light endpoints below are `async def`: they run directly on the event loop and only read
# small files/counters, so /health keeps answering instantly even while a flood of /predict
# requests is queued in the thread pool. (Heavy work, i.e. /predict and /upload-data, stays `def`,
# which FastAPI runs in a thread pool so it never blocks the event loop.)
@app.get("/health")
async def health():
    meta = manager.metadata
    uptime = time.time() - START_TIME
    endpoints = {
        path: {"count": v["count"], "avg_ms": round(v["total_ms"] / v["count"], 2)}
        for path, v in REQUEST_STATS["by_endpoint"].items() if v["count"]
    }
    return {
        "status": "ok" if manager.is_loaded else "degraded",
        "instance": INSTANCE_ID,
        "model_loaded": manager.is_loaded,
        "startup_error": startup_error if not manager.is_loaded else None,
        "started_at": datetime.fromtimestamp(START_TIME, timezone.utc).isoformat(),
        "uptime_seconds": round(uptime, 1),
        "uptime": _format_uptime(uptime),
        "model_version": meta.get("model_version"),
        "last_trained": meta.get("last_trained"),
        "model_metrics": {k: meta.get(k) for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc",
                                                    "train_samples", "test_samples")},
        "class_names": meta.get("class_names", CLASS_NAMES),
        "prediction_stats": manager.stats(),
        "request_stats": {"total_requests": REQUEST_STATS["total_requests"],
                          "errors": REQUEST_STATS["errors"], "endpoints": endpoints},
        "pending_uploads": count_pending_uploads(),
        "dataset_on_server": _data_counts(),
        "database": _db_summary(),
        "leaf_check": manager.leaf_checker.status(),
        "retraining": get_retrain_status()["state"],
    }


@app.get("/model-info")
async def model_info():
    """Full model metadata: metrics, per-class report, confusion matrix and the retraining history.
    Read from disk (not from memory) so it is always the latest, even right after a retrain."""
    from src.model import load_metadata as load_meta_from_disk
    return load_meta_from_disk()


@app.get("/eda-stats")
async def eda_stats():
    if not EDA_STATS_PATH.exists():
        raise HTTPException(404, "eda_stats.json not found in models/. Run the notebook first.")
    with open(EDA_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict")
def predict(file: UploadFile = File(..., description="A single maize leaf image (jpg/png)")):
    if not manager.is_loaded:
        raise HTTPException(503, f"Model not loaded: {startup_error}")
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "Image larger than 10 MB.")
    try:
        standardize_image_bytes(data)              # reject corrupt/non-image uploads with a clear 400
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        result = manager.predict(data)
    except NotMaizeLeafError as exc:
        # 422 = "I understood the request, but I can't process this content"
        raise HTTPException(422, {
            "error": "not_a_maize_leaf",
            "message": str(exc),
            **exc.info,                     # leaf_distance, max_distance, plant_pixels, min_plant_pixels, reason
            "filename": file.filename,
        }) from exc
    result["filename"] = file.filename
    if result["low_confidence"]:
        result["note"] = ("Low confidence: the leaf may be unclear, a mixed infection, or unlike the "
                          "training data. Consider uploading it with the correct label for retraining.")
        result["retrain_trigger"] = _auto_trigger("low-confidence prediction")
    return result


class _UploadBatch:
    """Processes the files of ONE upload request and records every file in the database
    (data/pipeline.db -> uploads table), whatever happens to it."""

    def __init__(self):
        self.batch_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.saved: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
        self.rejected: List[Dict[str, str]] = []
        self.duplicates = 0
        self.seen: set = set()

    def reject(self, filename: str, reason: str, label: Optional[str] = None, md5: Optional[str] = None) -> None:
        self.rejected.append({"file": filename, "reason": reason})
        database.record_upload(self.batch_id, filename, label, md5, "rejected", reason=reason)

    def add(self, filename: str, data: bytes, label: str) -> None:
        """Preprocess one image for the training set, then save it to data/uploads/<label>/:
        1. fingerprint (MD5) -> skip exact duplicates, in this upload AND in earlier ones (database)
        2. fully decode -> reject corrupt or tiny files; fix EXIF rotation; convert to 3-channel RGB
        3. leaf check -> reject pictures that are not maize leaves
        4. save as JPEG; resizing to 224x224 and scaling happen in the training pipeline"""
        digest = md5_bytes(data)
        earlier = database.is_known_image(digest)
        if digest in self.seen or earlier:
            self.duplicates += 1
            reason = ("same image twice in this upload" if digest in self.seen else
                      f"already uploaded on {earlier['uploaded_at']} as {earlier['label']} ({earlier['status']})")
            database.record_upload(self.batch_id, filename, label, digest, "duplicate", reason=reason)
            return
        try:
            img = standardize_image_bytes(data)                    # raises ValueError if corrupt
            is_leaf, leaf_info = manager.is_maize_leaf(data)       # keep non-maize pictures out of training
            if not is_leaf:
                manager.leaf_checker.rejected += 1
                raise ValueError(f"not a maize leaf: {leaf_info['reason']}")
        except ValueError as exc:
            self.reject(filename, str(exc), label, digest)
            return
        dest = UPLOAD_DIR / label / f"{digest[:12]}_{uuid.uuid4().hex[:6]}.jpg"
        save_as_jpeg(img, dest)
        database.record_upload(self.batch_id, filename, label, digest, "pending",
                               stored_path=str(dest), width=img.width, height=img.height)
        self.seen.add(digest)
        self.saved[label] += 1


@app.post("/upload-data")
def upload_data(
    files: List[UploadFile] = File(..., description="Images (all with the same label) and/or .zip files "
                                                     "with one folder per class"),
    label: Optional[str] = Form(None, description="Class for the plain images: Blight, Common_Rust, "
                                                   "Gray_Leaf_Spot or Healthy. Not needed for zip files."),
    auto_retrain: bool = Form(True, description="Start retraining automatically if the trigger fires"),
):
    # Swagger's /docs page pre-fills optional text fields with the word "string" and sends it
    # unless you clear the box. Treat that placeholder (and empty values) as "no label given".
    if label is not None and label.strip().lower() in ("", "string", "none", "null"):
        label = None
    canonical_label = normalize_class_name(label) if label else None
    if label and canonical_label is None:
        raise HTTPException(400, f"Unknown label '{label}'. Use one of {CLASS_NAMES}.")

    batch = _UploadBatch()
    for up in files:
        data = up.file.read()
        name = up.filename or "unnamed"
        if name.lower().endswith(".zip"):
            if len(data) > MAX_ZIP_BYTES:
                batch.reject(name, "zip larger than 200 MB")
                continue
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                batch.reject(name, "not a valid zip file")
                continue
            for member in zf.infolist():
                parts = Path(member.filename).parts
                if member.is_dir() or "__MACOSX" in parts or Path(member.filename).name.startswith("."):
                    continue
                if not is_image_file(member.filename):
                    continue
                # The label comes from the folder name inside the zip (e.g. Blight/leaf1.jpg)
                cls = next((normalize_class_name(p) for p in reversed(parts[:-1])
                            if normalize_class_name(p)), None) or canonical_label
                member_name = f"{name}/{member.filename}"
                if cls is None:
                    batch.reject(member_name, "no class folder in path and no label given")
                    continue
                batch.add(member_name, zf.read(member), cls)
            continue

        if canonical_label is None:
            batch.reject(name, "plain images need a 'label' form field")
            continue
        if len(data) > MAX_FILE_BYTES:
            batch.reject(name, "larger than 10 MB", canonical_label)
            continue
        batch.add(name, data, canonical_label)

    trigger = _auto_trigger("upload") if auto_retrain else {
        "should_retrain": False, "started": False, "reason": "auto_retrain disabled for this upload"}
    return {
        "batch_id": batch.batch_id,               # look it up in GET /uploads
        "saved": sum(batch.saved.values()),
        "saved_per_class": {k: v for k, v in batch.saved.items() if v},
        "duplicates_skipped": batch.duplicates,
        "rejected": batch.rejected,
        "pending_uploads": count_pending_uploads(),
        "retrain_trigger": trigger,
    }


@app.get("/uploads")
def uploads(limit: int = 100, status: Optional[str] = None):
    """Upload records from the database (newest first), plus a summary.
    status filter: pending | in_training_set | duplicate | rejected"""
    return {"summary": database.upload_summary(), "uploads": database.list_uploads(limit, status)}


@app.get("/retrain-runs")
def retrain_runs(limit: int = 50):
    """Every retraining run recorded in the database (newest first)."""
    return {"runs": database.list_runs(limit)}


@app.post("/retrain")
def retrain():
    if is_retraining():
        raise HTTPException(409, "Retraining is already running. Poll GET /retrain-status.")
    pending = sum(count_pending_uploads().values())
    started = start_retraining_async(trigger="manual", on_complete=_on_retrain_complete)
    if not started:
        raise HTTPException(409, "Retraining is already running.")
    return {"message": "Retraining started in the background.", "pending_new_images": pending,
            "status_endpoint": "/retrain-status"}


@app.get("/retrain-status")
async def retrain_status():
    status = get_retrain_status()
    status["current_model_version"] = manager.metadata.get("model_version")
    return status
