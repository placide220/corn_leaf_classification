"""Model creation, evaluation, retraining and the retraining trigger.

Retraining flow (``retrain_model``):
    1. Move newly uploaded, labelled images from data/uploads/<class>/ into data/train/<class>/
    2. Load the current .keras model (our custom CNN) and keep training it on the full (old + new)
       training set with a 10x smaller learning rate
    3. Evaluate on the untouched data/test/ set
    4. Promote the new model only if accuracy did not drop by more than
       RETRAIN_MAX_ACCURACY_DROP (default 0.02). The previous model is archived first.
    5. Update model_metadata.json and bump model_version (v1.0 -> v1.1 -> v1.2 ...)

Trigger (``check_retrain_trigger``):
    * >= RETRAIN_MIN_NEW_IMAGES (default 10) new labelled images are waiting, OR
    * low-confidence predictions (confidence < 0.65) reached RETRAIN_LOW_CONF_LIMIT
      (default 20) AND there is at least one new labelled image to learn from.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src import (ARCHIVE_DIR, CLASS_NAMES, IMG_SIZE, METADATA_PATH, MODEL_PATH, MODELS_DIR,
                 TEST_DIR, TRAIN_DIR, UPLOAD_DIR)
from src import database
from src.preprocessing import build_dataset, is_image_file, list_image_files

RETRAIN_MIN_NEW_IMAGES = int(os.getenv("RETRAIN_MIN_NEW_IMAGES", 10))
RETRAIN_LOW_CONF_LIMIT = int(os.getenv("RETRAIN_LOW_CONF_LIMIT", 20))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", 3))
RETRAIN_LEARNING_RATE = float(os.getenv("RETRAIN_LEARNING_RATE", 1e-4))
# 16 (not 32): retraining keeps two models in memory (the live one + the one being trained),
# and smaller batches roughly halve the training memory so the container is not killed.
RETRAIN_BATCH_SIZE = int(os.getenv("RETRAIN_BATCH_SIZE", 16))

RETRAIN_MAX_ACCURACY_DROP = float(os.getenv("RETRAIN_MAX_ACCURACY_DROP", 0.02))
# Safety check: refuse to retrain on a tiny training set (it would wreck the model)
RETRAIN_MIN_TRAIN_IMAGES = int(os.getenv("RETRAIN_MIN_TRAIN_IMAGES", 100))
LOCK_PATH = MODELS_DIR / ".retrain.lock"
LOCK_STALE_SECONDS = 3 * 60 * 60


# --------------------------------------------------------------------------- #
# Model creation (same architecture as the notebook)
# --------------------------------------------------------------------------- #
# BatchNorm momentum 0.9 instead of Keras' default 0.99. BatchNorm keeps running averages of the
# activations that are used at prediction time. Training from scratch changes the weights quickly,
# and with 0.99 those averages lag far behind: training accuracy was 99% while validation stayed at
# chance (27%). With 0.9 they keep up, and validation accuracy follows training from epoch 2.
BN_MOMENTUM = 0.9


def conv_block(x, filters: int, name: str, l2: float = 1e-4, dropout: float = 0.15):
    """Two 3x3 convolutions (each followed by BatchNorm + ReLU), then 2x2 max-pooling and dropout.
    - Conv2D learns local patterns (edges -> spots -> lesion shapes as the network gets deeper)
    - BatchNormalization keeps activations in a stable range, so training from scratch converges faster
    - MaxPooling halves the image size, so deeper blocks look at bigger regions of the leaf
    - Dropout + L2 weight decay are regularisation: they stop the network memorising the training images"""
    import tensorflow as tf

    reg = tf.keras.regularizers.l2(l2)
    for i in (1, 2):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=reg,
                                   name=f"{name}_conv{i}")(x)
        x = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_bn{i}")(x)
        x = tf.keras.layers.Activation("relu", name=f"{name}_relu{i}")(x)
    x = tf.keras.layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    return tf.keras.layers.Dropout(dropout, name=f"{name}_drop")(x)


def build_model(num_classes: int = len(CLASS_NAMES), learning_rate: float = 1e-3, l2: float = 1e-4):
    """Custom CNN trained FROM SCRATCH (random initial weights, no pre-trained network).

    Input 224x224x3 (scaled to [-1, 1])
      -> stem: Conv 32 (stride 2) + BN + ReLU                      112x112
      -> block 1:  2 x Conv 32  + BN + ReLU, MaxPool, Dropout       56x56
      -> block 2:  2 x Conv 64  + BN + ReLU, MaxPool, Dropout       28x28
      -> block 3:  2 x Conv 128 + BN + ReLU, MaxPool, Dropout       14x14
      -> block 4:  2 x Conv 256 + BN + ReLU, MaxPool, Dropout        7x7
      -> GlobalAveragePooling ("gap", 256 features)
      -> Dropout 0.4 -> Dense 128 + ReLU (L2) -> Dropout 0.3 -> Dense 4 softmax
    About 1.2 M parameters (~5 MB). The "gap" layer name is used by the leaf check (src/leaf_check.py).
    """
    import tensorflow as tf

    reg = tf.keras.regularizers.l2(l2)
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=reg,
                               name="stem_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, name="stem_bn")(x)
    x = tf.keras.layers.Activation("relu", name="stem_relu")(x)
    for i, filters in enumerate((32, 64, 128, 256), start=1):
        x = conv_block(x, filters, name=f"block{i}", l2=l2, dropout=0.1 + 0.05 * i)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.4, name="head_dropout1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg, name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="corn_custom_cnn")
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate: float):
    import tensorflow as tf

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def compute_class_weights(labels: List[int], num_classes: int = len(CLASS_NAMES)) -> Dict[int, float]:
    """Balanced class weights (n_samples / (n_classes * n_class_samples))."""
    from sklearn.utils.class_weight import compute_class_weight

    present = np.unique(labels)
    weights = compute_class_weight("balanced", classes=present, y=np.asarray(labels))
    cw = {int(c): float(w) for c, w in zip(present, weights)}
    return {i: cw.get(i, 1.0) for i in range(num_classes)}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_model(model, test_dir: Path = TEST_DIR, batch_size: int = 32,
                   class_names: List[str] = CLASS_NAMES) -> Dict[str, Any]:
    from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score)
    from sklearn.preprocessing import label_binarize

    paths, labels = list_image_files(test_dir, class_names)
    if not paths:
        raise FileNotFoundError(f"No test images in {test_dir}")
    ds = build_dataset(paths, labels, training=False, batch_size=batch_size)
    probs = model.predict(ds, verbose=0)
    y_true = np.asarray(labels)
    y_pred = probs.argmax(axis=1)

    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    try:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        roc_auc = float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"))
    except ValueError:                             # a class missing from the test set
        roc_auc = None

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(p_m), 4),
        "recall": round(float(r_m), 4),
        "f1_score": round(float(f_m), 4),
        "precision_weighted": round(float(p_w), 4),
        "recall_weighted": round(float(r_w), 4),
        "f1_weighted": round(float(f_w), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        "test_samples": int(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# Metadata helpers
# --------------------------------------------------------------------------- #
def load_metadata(path: Path = METADATA_PATH) -> Dict[str, Any]:
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"class_names": CLASS_NAMES, "model_version": "v1.0"}


def save_metadata(meta: Dict[str, Any], path: Path = METADATA_PATH) -> None:
    tmp = Path(path).with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)                          # atomic: readers never see half a file


def bump_version(version: str) -> str:
    """'v1.0' -> 'v1.1', 'v1.9' -> 'v1.10', anything odd -> 'v1.1'."""
    try:
        major, minor = version.lstrip("v").split(".")[:2]
        return f"v{int(major)}.{int(minor) + 1}"
    except (ValueError, AttributeError):
        return "v1.1"


# --------------------------------------------------------------------------- #
# Uploaded-data bookkeeping
# --------------------------------------------------------------------------- #
def count_pending_uploads(upload_dir: Path = UPLOAD_DIR) -> Dict[str, int]:
    counts = {}
    for cls in CLASS_NAMES:
        d = Path(upload_dir) / cls
        counts[cls] = sum(1 for f in d.iterdir() if f.is_file() and is_image_file(f)) if d.is_dir() else 0
    return counts


def merge_uploads_into_train(upload_dir: Path = UPLOAD_DIR, train_dir: Path = TRAIN_DIR):
    """Move every uploaded image into data/train/<class>/ (unique file names).
    Returns (images moved per class, [(old_path, new_path, class), ...]) so the database can be updated."""
    moved, moves = {}, []
    for cls in CLASS_NAMES:
        src_dir = Path(upload_dir) / cls
        if not src_dir.is_dir():
            continue
        dst_dir = Path(train_dir) / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in list(src_dir.iterdir()):
            if f.is_file() and is_image_file(f):
                dest = dst_dir / f"upload_{uuid.uuid4().hex[:10]}{f.suffix.lower()}"
                shutil.move(str(f), dest)
                moves.append((str(f), str(dest), cls))
                n += 1
        if n:
            moved[cls] = n
    return moved, moves


def _db_call(fn, *args, **kwargs):
    """Database bookkeeping must never break retraining itself: log the error and carry on."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[database] {fn.__name__} failed: {exc}", flush=True)
        return None


# --------------------------------------------------------------------------- #
# Retraining status + cross-process lock
# --------------------------------------------------------------------------- #
_status_lock = threading.Lock()
_retrain_status: Dict[str, Any] = {
    "state": "idle",            # idle | running | completed | rejected | failed
    "trigger": None,
    "started_at": None,
    "finished_at": None,
    "message": "No retraining has run since the API started.",
    "epoch": 0,
    "total_epochs": 0,
    "result": None,
}


def get_retrain_status() -> Dict[str, Any]:
    with _status_lock:
        status = dict(_retrain_status)
    status["lock_held_by_other_process"] = LOCK_PATH.exists() and status["state"] != "running"
    return status


def _set_status(**kwargs) -> None:
    with _status_lock:
        _retrain_status.update(kwargs)


def _acquire_lock() -> bool:
    """File lock so only one process/container retrains at a time."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists() and time.time() - LOCK_PATH.stat().st_mtime > LOCK_STALE_SECONDS:
        LOCK_PATH.unlink(missing_ok=True)          # left behind by a crashed run
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def is_retraining() -> bool:
    with _status_lock:
        running = _retrain_status["state"] == "running"
    return running or LOCK_PATH.exists()


# --------------------------------------------------------------------------- #
# Retraining
# --------------------------------------------------------------------------- #
def retrain_model(trigger: str = "manual",
                  epochs: int = RETRAIN_EPOCHS,
                  learning_rate: float = RETRAIN_LEARNING_RATE,
                  batch_size: int = RETRAIN_BATCH_SIZE,
                  on_complete: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """Merge uploads, continue training the saved model, evaluate, promote and version it.

    Blocking; the API runs it in a background thread via ``start_retraining_async``.
    ``on_complete`` is called with the result (used by the API to hot-reload the model).
    """
    import tensorflow as tf

    if not _acquire_lock():
        msg = "Retraining is already running in another process (models/.retrain.lock exists)."
        _set_status(state="failed", finished_at=datetime.now(timezone.utc).isoformat(), message=msg)
        raise RuntimeError(msg)

    _set_status(state="running", trigger=trigger, started_at=datetime.now(timezone.utc).isoformat(),
                finished_at=None, message="Step 1/5 - Loading the uploaded images recorded in the database...",
                epoch=0, total_epochs=epochs, result=None)
    run_id = None
    try:
        old_meta = load_metadata()
        old_version = old_meta.get("model_version", "v1.0")
        old_accuracy = old_meta.get("accuracy")
        run_id = _db_call(database.start_run, trigger, old_version)

        # 1. Merge new data: uploaded images (already validated, de-duplicated, converted to RGB JPEG and
        #    checked to be maize leaves when they were uploaded) move into data/train/<class>/
        merged, moves = merge_uploads_into_train()
        n_new = sum(merged.values())
        _db_call(database.mark_merged, moves, run_id)

        # 2. Build datasets: 90% of train for fitting, 10% for validation; test stays untouched
        from sklearn.model_selection import train_test_split

        paths, labels = list_image_files(TRAIN_DIR)
        counts = {c: int(n) for c, n in zip(CLASS_NAMES, np.bincount(labels, minlength=len(CLASS_NAMES)))}
        if len(paths) < RETRAIN_MIN_TRAIN_IMAGES:
            # Almost always means data/train was not copied to the server/container
            raise ValueError(
                f"Only {len(paths)} training images found in {TRAIN_DIR} {counts}. Retraining needs the full "
                f"training set (at least {RETRAIN_MIN_TRAIN_IMAGES}). Copy data/train into the project and rebuild "
                f"the Docker image.")

        # Validation set = 10% of train, but at least 1 image per class so a stratified split is possible.
        # If a class is too small to split, fall back to a plain random split.
        n_classes = len(set(labels))
        n_val = max(int(round(0.1 * len(paths))), n_classes)
        stratify = labels if min(np.bincount(labels)[np.bincount(labels) > 0]) >= 2 else None
        tr_p, va_p, tr_y, va_y = train_test_split(paths, labels, test_size=n_val,
                                                  stratify=stratify, random_state=int(time.time()) % 10_000)
        # 2. Preprocessing pipeline (same as the notebook): decode -> RGB -> resize 224x224 ->
        #    augmentation (flips, rotations, brightness/contrast/saturation; training only) -> scale to [-1, 1]
        _set_status(message=(f"Step 2/5 - Preprocessing {len(paths)} images ({n_new} new): resize to 224x224, "
                             f"augmentation, scaling to [-1, 1]; {len(tr_p)} for training, {len(va_p)} for validation"))
        train_ds = build_dataset(tr_p, tr_y, training=True, batch_size=batch_size)
        val_ds = build_dataset(va_p, va_y, training=False, batch_size=batch_size)

        # 3. Continue training OUR OWN saved model (the custom CNN trained from scratch in the notebook is
        #    the pre-trained starting point) with a 10x smaller learning rate, so it keeps what it already
        #    learned and adapts to the new images instead of starting over
        _set_status(message=f"Step 3/5 - Retraining our saved model {old_version} (custom CNN) on {len(tr_p)} images ({n_new} new)...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        compile_model(model, learning_rate)

        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                _set_status(epoch=epoch + 1,
                            message=(f"Step 3/5 - Epoch {epoch + 1}/{epochs} - loss {logs.get('loss', 0):.4f}, "
                                     f"val_accuracy {logs.get('val_accuracy', 0):.4f}"))

        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            class_weight=compute_class_weights(tr_y), verbose=0,
            callbacks=[ProgressCallback(),
                       tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                        restore_best_weights=True)],
        )

        # 4. Evaluate on the fixed test set
        _set_status(message="Step 4/5 - Evaluating the new model on the fixed test set...")
        metrics = evaluate_model(model, TEST_DIR, batch_size)

        promoted = (old_accuracy is None
                    or metrics["accuracy"] >= float(old_accuracy) - RETRAIN_MAX_ACCURACY_DROP)
        now = datetime.now(timezone.utc).isoformat()
        run = {
            "timestamp": now, "trigger": trigger, "from_version": old_version,
            "new_images": merged, "train_samples": len(paths),
            "epochs_run": len(history.history.get("loss", [])),
            "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "previous_accuracy": old_accuracy, "promoted": promoted,
        }

        if promoted:
            new_version = bump_version(old_version)
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            if MODEL_PATH.exists():
                shutil.copy2(MODEL_PATH, ARCHIVE_DIR / f"corn_disease_model_{old_version}.keras")
            tmp_path = MODEL_PATH.with_name("corn_disease_model.tmp.keras")
            model.save(tmp_path)
            os.replace(tmp_path, MODEL_PATH)       # atomic swap -> hot-reload picks it up
            run["to_version"] = new_version

            meta = dict(old_meta)
            meta.update({
                "class_names": old_meta.get("class_names", CLASS_NAMES),
                "model_version": new_version,
                "last_trained": now,
                "accuracy": metrics["accuracy"], "precision": metrics["precision"],
                "recall": metrics["recall"], "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_weighted": metrics["f1_weighted"],
                "confusion_matrix": metrics["confusion_matrix"],
                "train_samples": len(paths), "test_samples": metrics["test_samples"],
            })
        else:
            meta = dict(old_meta)
            run["to_version"] = old_version
        meta.setdefault("retrain_history", []).append(run)
        save_metadata(meta)

        state = "completed" if promoted else "rejected"
        msg = (f"Promoted {old_version} -> {run['to_version']} (test accuracy {metrics['accuracy']:.4f})."
               if promoted else
               f"New model rejected: accuracy {metrics['accuracy']:.4f} < previous {old_accuracy:.4f}. "
               f"Kept {old_version}; new images stay in the training set.")
        run["run_id"] = run_id
        _db_call(database.finish_run, run_id, status=state, to_version=run["to_version"], new_images=n_new,
                 train_samples=len(paths), accuracy_before=old_accuracy, accuracy_after=metrics["accuracy"],
                 f1_after=metrics["f1_score"], message=msg)
        if on_complete:                             # e.g. hot-reload, BEFORE reporting "completed"
            _set_status(message="Step 5/5 - " + msg + " Loading the new model into the API...")
            on_complete(run)
        _set_status(state=state, finished_at=datetime.now(timezone.utc).isoformat(), message=msg, result=run)
        return run

    except Exception as exc:  # noqa: BLE001 - report every failure to the status endpoint
        _set_status(state="failed", finished_at=datetime.now(timezone.utc).isoformat(),
                    message=f"{type(exc).__name__}: {exc}", result={"traceback": traceback.format_exc()})
        _db_call(database.finish_run, run_id, status="failed", message=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _release_lock()


def start_retraining_async(trigger: str = "manual",
                           on_complete: Optional[Callable[[Dict[str, Any]], None]] = None) -> bool:
    """Start retrain_model in a daemon thread. Returns False if one is already running."""
    if is_retraining():
        return False
    _set_status(state="running", trigger=trigger, message="Starting retraining...",
                started_at=datetime.now(timezone.utc).isoformat(), finished_at=None, epoch=0, result=None)

    def _runner():
        try:
            retrain_model(trigger=trigger, on_complete=on_complete)
        except Exception:                          # status already set to failed
            pass

    threading.Thread(target=_runner, name="retrain-worker", daemon=True).start()
    return True


# --------------------------------------------------------------------------- #
# Trigger
# --------------------------------------------------------------------------- #
def check_retrain_trigger(low_confidence_count: int = 0,
                          min_new_images: int = RETRAIN_MIN_NEW_IMAGES,
                          low_conf_limit: int = RETRAIN_LOW_CONF_LIMIT) -> Tuple[bool, str]:
    """Decide whether retraining should start automatically.

    Returns (should_retrain, reason).
    """
    if is_retraining():
        return False, "Retraining already in progress."
    pending = sum(count_pending_uploads().values())
    if pending >= min_new_images:
        return True, f"{pending} new labelled images uploaded (threshold {min_new_images})."
    if low_confidence_count >= low_conf_limit and pending > 0:
        return True, (f"{low_confidence_count} low-confidence predictions (limit {low_conf_limit}) "
                      f"and {pending} new labelled images available.")
    if low_confidence_count >= low_conf_limit:
        return False, (f"{low_confidence_count} low-confidence predictions: model may be drifting, "
                       "upload labelled images so it has something new to learn from.")
    return False, f"{pending}/{min_new_images} new images pending; no retraining needed yet."
