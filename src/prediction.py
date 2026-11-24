"""Thread-safe model loading, hot-reload and single-image prediction.

* One ModelManager per process (``get_model_manager()``).
* Requests take a local reference to the current model, so a reload never
  breaks a prediction that is already running.
* Hot-reload: after retraining, ``reload()`` swaps in the new model. Every
  prediction also checks the model file's modification time, so other API
  processes/containers sharing the same models/ folder pick up a new model
  automatically.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src import CLASS_NAMES, LOW_CONFIDENCE_THRESHOLD, METADATA_PATH, MODEL_PATH
from src.leaf_check import LeafChecker, NotMaizeLeafError
from src.preprocessing import preprocess_image_bytes


def load_metadata(path: Path = METADATA_PATH) -> Dict[str, Any]:
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"class_names": CLASS_NAMES, "model_version": "unknown"}


class ModelManager:
    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self._lock = threading.RLock()
        self._model = None
        self._metadata: Dict[str, Any] = {}
        self._loaded_mtime: Optional[float] = None
        self.loaded_at: Optional[float] = None

        # Only PREDICT_CONCURRENCY images go through the model at the same time; extra requests
        # wait their turn. On a 1-CPU container, running dozens of predictions at once makes
        # them ALL slow (they fight for the CPU and memory) and can stall the whole API.
        # Queuing them keeps each prediction fast and the API responsive under a flood.
        self._predict_slots = threading.BoundedSemaphore(int(os.getenv("PREDICT_CONCURRENCY", 2)))

        # Hot-reload check runs at most once every few seconds, not on every request
        # (checking a file on a Docker volume can be slow, especially on Windows).
        self._last_reload_check = 0.0
        self._reload_check_interval = float(os.getenv("RELOAD_CHECK_SECONDS", 5))

        # "Is this a maize leaf?" guard (see src/leaf_check.py)
        self.leaf_checker = LeafChecker()

        # Monitoring counters (per process)
        self._stats_lock = threading.Lock()
        self.total_predictions = 0
        self.low_confidence_count = 0          # since last retrain
        self._latencies_ms = deque(maxlen=1000)

    # ----------------------------- loading ------------------------------- #
    def load(self) -> None:
        import tensorflow as tf                # imported lazily so module import is cheap

        with self._lock:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            mtime = self.model_path.stat().st_mtime
            model = tf.keras.models.load_model(self.model_path, compile=False)
            # One model with TWO outputs: the class probabilities AND the 1,280 image features
            # from the "gap" layer, which the leaf check compares with real training leaves.
            # A single forward pass gives both, so the check costs almost nothing extra.
            dual = tf.keras.Model(model.inputs, [model.outputs[0], self._gap_layer(model).output])
            # Warm-up call so the first real request is not slow
            dual(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
            self._model = dual
            self._metadata = load_metadata(self.metadata_path)
            self._loaded_mtime = mtime
            self.loaded_at = time.time()

        # New model -> new features -> recalibrate the leaf check (in the background)
        self.leaf_checker.reset()
        self.leaf_checker.calibrate_async(lambda x: np.asarray(dual(x, training=False)[1]), self.model_path)

    @staticmethod
    def _gap_layer(model):
        """The GlobalAveragePooling2D layer (named 'gap' in the notebook and src/model.py)."""
        import tensorflow as tf

        try:
            return model.get_layer("gap")
        except ValueError:
            return next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))

    def reload(self) -> None:
        """Force-load the model from disk (called right after retraining)."""
        self.load()
        with self._stats_lock:
            self.low_confidence_count = 0

    def _maybe_hot_reload(self) -> None:
        now = time.time()
        if now - self._last_reload_check < self._reload_check_interval:
            return
        self._last_reload_check = now
        try:
            mtime = self.model_path.stat().st_mtime
        except FileNotFoundError:
            return
        if self._loaded_mtime is None or mtime > self._loaded_mtime:
            with self._lock:
                # re-check inside the lock: another thread may have reloaded already
                if self._loaded_mtime is None or self.model_path.stat().st_mtime > self._loaded_mtime:
                    self.load()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @property
    def class_names(self):
        return self._metadata.get("class_names", CLASS_NAMES)

    # ---------------------------- prediction ----------------------------- #
    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        start = time.perf_counter()
        self._maybe_hot_reload()
        with self._lock:
            model = self._model                 # local reference = safe across reloads
            class_names = list(self.class_names)
            version = self._metadata.get("model_version", "unknown")
        if model is None:
            raise RuntimeError("Model is not loaded")

        queued_at = time.perf_counter()
        with self._predict_slots:                 # wait for a free slot (see __init__)
            wait_ms = (time.perf_counter() - queued_at) * 1000
            # Decoding/resizing is also CPU work, so it happens INSIDE the slot too.
            # Otherwise dozens of waiting requests would all decode at once and starve the CPU.
            x = preprocess_image_bytes(image_bytes)
            probs, features = model(x, training=False)
        probs = np.asarray(probs)[0]

        # Reject images that are not maize leaves BEFORE trusting the class prediction
        is_leaf, leaf_info = self.leaf_checker.check(np.asarray(features), x)
        if not is_leaf:
            self.leaf_checker.rejected += 1
            raise NotMaizeLeafError(leaf_info)

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        low_conf = confidence < LOW_CONFIDENCE_THRESHOLD
        latency_ms = (time.perf_counter() - start) * 1000

        with self._stats_lock:
            self.total_predictions += 1
            if low_conf:
                self.low_confidence_count += 1
            self._latencies_ms.append(latency_ms)

        return {
            "predicted_class": class_names[idx],
            "confidence": round(confidence, 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(class_names, probs)},
            "low_confidence": low_conf,
            "model_version": version,
            "latency_ms": round(latency_ms, 2),
            "queue_wait_ms": round(wait_ms, 2),     # time spent waiting for a free model slot
            # Leaf-check scores for this image (None = check still calibrating / switched off)
            "leaf_check": leaf_info,
        }

    def is_maize_leaf(self, image_bytes: bytes) -> tuple:
        """(ok, details) for an uploaded training image, so pictures that are not maize leaves
        never get into the training data. Always ok while the check is still calibrating."""
        if not self.leaf_checker.ready:
            return True, None
        with self._lock:
            model = self._model
        with self._predict_slots:
            x = preprocess_image_bytes(image_bytes)
            _, features = model(x, training=False)
        return self.leaf_checker.check(np.asarray(features), x)

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            lat = list(self._latencies_ms)
        return {
            "total_predictions": self.total_predictions,
            "low_confidence_predictions": self.low_confidence_count,
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "avg_latency_ms": round(float(np.mean(lat)), 2) if lat else None,
            "p95_latency_ms": round(float(np.percentile(lat, 95)), 2) if lat else None,
            "rejected_not_maize_leaf": self.leaf_checker.rejected,
        }


_manager: Optional[ModelManager] = None
_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ModelManager()
        return _manager


def predict_image(image_bytes: bytes) -> Dict[str, Any]:
    """Convenience function: predict one image with the shared model."""
    mgr = get_model_manager()
    if not mgr.is_loaded:
        mgr.load()
    return mgr.predict(image_bytes)


if __name__ == "__main__":                       # python -m src.prediction path/to/leaf.jpg
    import sys

    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.prediction <image_path>")
    with open(sys.argv[1], "rb") as f:
        print(json.dumps(predict_image(f.read()), indent=2))
