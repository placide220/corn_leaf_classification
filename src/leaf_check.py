"""
"Is this even a maize leaf?" check (out-of-distribution detection).

The classifier only knows 4 classes, so on its own it would label a photo of a phone, a cat or a
document as "Blight" or "Healthy". This module rejects such images BEFORE the prediction is
trusted (HTTP 422), and also stops them from being uploaded as training data. Two independent
gates, both calibrated automatically from YOUR training leaves:

Gate 1 - "Does the model see a maize leaf?" (Mahalanobis distance on the model's features)
    The model's "gap" layer turns every image into 1,280 numbers describing its content.
    Calibration learns where real maize leaves sit in that space: the average features of each
    class and how much they normally vary (a covariance matrix, after reducing to the 64 most
    informative directions with PCA). A new image's score is its Mahalanobis distance to the
    closest class: "how many typical deviations away from any real leaf class is this?".
    A phone or a cat lands far outside every class.

Gate 2 - "Does the picture contain plant tissue?" (colour check)
    Share of pixels with plant colours (green/yellow/brown hues with some saturation).
    Maize leaf photos are mostly leaf; a white product photo, a document or a blue sign is not.

Calibration uses a sample of training leaves (default 250 per class): 60% to learn the statistics,
the other 40% (held back) to set the thresholds so that ~99% of REAL leaves pass both gates.
Using held-back images matters: thresholds set on the same images used for learning would be too
strict and reject real leaves.

Self-test: after calibration the check must reject at least 4 of 5 obviously-not-a-leaf images
(noise, blank page, black, solid red, gradient). If it cannot, it switches itself off instead of
blocking real leaves; /health -> leaf_check shows the state and the reason.

The calibration is cached in models/leaf_check.npz (keyed by a hash of the model file), so it is
recomputed only when the model changes, e.g. after retraining.

Settings (environment variables):
    LEAF_CHECK_ENABLED            "true" (default) / "false"
    LEAF_CHECK_SAMPLES_PER_CLASS  training images per class used for calibration (default 250)
    LEAF_CHECK_PERCENTILE         % of real held-back leaves allowed to fail each gate (default 1.0)
    LEAF_CHECK_MARGIN             tolerance on both limits, 0.25 = 25% (default)
    LEAF_CHECK_PCA_DIMS           feature directions kept for gate 1 (default 64)
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from src import CLASS_NAMES, MODELS_DIR, TRAIN_DIR

ENABLED = os.getenv("LEAF_CHECK_ENABLED", "true").lower() in ("1", "true", "yes")
SAMPLES_PER_CLASS = int(os.getenv("LEAF_CHECK_SAMPLES_PER_CLASS", 250))
PERCENTILE = float(os.getenv("LEAF_CHECK_PERCENTILE", 1.0))
# Tolerance on both limits: distance limit x (1 + MARGIN), plant-colour minimum x (1 - MARGIN).
# Real leaves sit just around the limits while non-leaf images are typically 3-8x further away,
# so a 25% tolerance lets unusual real leaves through without letting junk in.
MARGIN = float(os.getenv("LEAF_CHECK_MARGIN", 0.25))
PCA_DIMS = int(os.getenv("LEAF_CHECK_PCA_DIMS", 64))
# Images per forward pass during calibration. Small on purpose: TensorFlow keeps the working memory
# of the largest batch it has seen, and 32 images at once added ~3 GB to the API's memory.
CALIBRATION_BATCH = int(os.getenv("LEAF_CHECK_BATCH", 4))
CACHE_PATH = MODELS_DIR / "leaf_check.npz"
CACHE_VERSION = 3                      # bump when the calibration method changes


class NotMaizeLeafError(ValueError):
    """Raised when an image does not look like a maize leaf."""

    def __init__(self, info: Dict[str, Any]):
        self.info = info
        super().__init__(f"This image does not look like a maize leaf ({info['reason']}). "
                         "Please upload a clear photo of a single maize leaf.")


# ----------------------------------------------------------------------------- helpers
def file_hash(path: Path) -> str:
    """SHA-1 of the model file: identifies the model version, even after copying/cloning."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plant_pixel_ratio(x: np.ndarray) -> np.ndarray:
    """Share of pixels with plant colours, for a batch of preprocessed images (values in [-1, 1]).

    A pixel counts as plant-like when its hue is between orange-brown and green-cyan (15-170 deg),
    it has some colour (saturation >= 0.15) and is not almost black (value >= 0.10).
    That covers healthy green tissue, yellowing, and brown/tan lesions."""
    rgb = (np.asarray(x, dtype=np.float32) + 1.0) / 2.0              # back to [0, 1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax, cmin = rgb.max(axis=-1), rgb.min(axis=-1)
    delta = cmax - cmin
    sat = np.where(cmax > 0, delta / np.maximum(cmax, 1e-6), 0.0)
    d = np.maximum(delta, 1e-6)
    hue = np.where(cmax == r, ((g - b) / d) % 6, np.where(cmax == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    plant = (hue >= 15) & (hue <= 170) & (sat >= 0.15) & (cmax >= 0.10)
    return plant.reshape(plant.shape[0], -1).mean(axis=1)


def _junk_probes() -> np.ndarray:
    """Five obviously-not-a-maize-leaf images, already preprocessed like real inputs ([-1, 1])."""
    rng = np.random.default_rng(0)
    size = (224, 224, 3)
    gradient = np.repeat(np.linspace(0, 255, 224, dtype=np.float32)[None, :, None], 224, axis=0)
    images = [
        rng.uniform(0, 255, size),                              # random noise
        np.full(size, 240.0),                                   # blank white page
        np.full(size, 15.0),                                    # black image
        np.broadcast_to(np.array([200.0, 40.0, 40.0]), size),   # solid red
        np.repeat(gradient, 3, axis=2),                         # grey gradient
    ]
    return np.stack([img.astype(np.float32) / 127.5 - 1.0 for img in images])


# ----------------------------------------------------------------------------- the checker
class LeafChecker:
    def __init__(self):
        self._lock = threading.Lock()
        self._params: Optional[Dict[str, np.ndarray]] = None   # PCA + class statistics + thresholds
        self._generation = 0                                    # bumped on every model change (see reset)
        self.rejected = 0
        self.state = "disabled" if not ENABLED else "not started"
        self.message = "" if ENABLED else "LEAF_CHECK_ENABLED=false"
        self.calibration_report: Dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return ENABLED and self._params is not None

    def status(self) -> Dict[str, Any]:
        p = self._params
        return {
            "enabled": ENABLED,
            "state": self.state,                    # not started | calibrating | ready | unavailable | disabled
            "max_distance": round(float(p["dist_threshold"]), 2) if p is not None else None,
            "min_plant_pixels": round(float(p["plant_threshold"]), 3) if p is not None else None,
            "rejected_images": self.rejected,
            "calibration": self.calibration_report,
            "message": self.message,
        }

    # ------------------------------------------------------------------ calibration
    def calibrate_async(self, features_fn: Callable[[np.ndarray], np.ndarray], model_path: Path) -> None:
        """Calibrate in a background thread so the API answers immediately.
        Until calibration finishes, predictions are served without the leaf check."""
        if not ENABLED:
            return
        threading.Thread(target=self._calibrate, args=(features_fn, model_path, self._generation),
                         name="leaf-check-calibration", daemon=True).start()

    def _calibrate(self, features_fn: Callable[[np.ndarray], np.ndarray], model_path: Path, generation: int) -> None:
        from sklearn.covariance import LedoitWolf

        from src.preprocessing import build_dataset, list_image_files

        try:
            self.state, self.message = "calibrating", "Learning what real maize leaves look like..."
            key = file_hash(model_path)

            # 1) Re-use the cached calibration if it was made for exactly this model file
            if CACHE_PATH.exists():
                cached = dict(np.load(CACHE_PATH, allow_pickle=False))
                if (str(cached.get("model_key")) == key and int(cached.get("cache_version", 0)) == CACHE_VERSION
                        and int(cached.get("samples_per_class", 0)) == SAMPLES_PER_CLASS):
                    report = {"source": "cache", "reference_leaves": int(cached["n_fit"]) + int(cached["n_cal"])}
                    self._install(generation, cached, report, "ready (loaded from cache)")
                    return

            # 2) Sample training leaves per class; 60% to learn, 40% held back to set thresholds
            paths, labels = list_image_files(TRAIN_DIR)
            if len(paths) < 100:
                self.state = "unavailable"
                self.message = f"Only {len(paths)} training images on this server; leaf check needs data/train."
                return
            rng = random.Random(42)
            fit_paths, fit_y, cal_paths = [], [], []
            for i, _ in enumerate(CLASS_NAMES):
                cls_paths = [p for p, y in zip(paths, labels) if y == i]
                chosen = rng.sample(cls_paths, min(SAMPLES_PER_CLASS, len(cls_paths)))
                n_fit = int(round(0.6 * len(chosen)))
                fit_paths += chosen[:n_fit]
                fit_y += [i] * n_fit
                cal_paths += chosen[n_fit:]

            t0 = time.time()

            def features_and_colour(ps):
                feats, plant = [], []
                for x, _ in build_dataset(ps, [0] * len(ps), training=False, batch_size=CALIBRATION_BATCH):
                    x = x.numpy()
                    feats.append(features_fn(x))
                    plant.append(plant_pixel_ratio(x))
                return np.concatenate(feats).astype(np.float64), np.concatenate(plant)

            f_fit, plant_fit = features_and_colour(fit_paths)
            f_cal, plant_cal = features_and_colour(cal_paths)
            y_fit = np.asarray(fit_y)

            # 3) Gate 1: PCA (keep the most informative directions) + class means + shared covariance
            mu = f_fit.mean(axis=0)
            _, _, vt = np.linalg.svd(f_fit - mu, full_matrices=False)
            k = int(min(PCA_DIMS, vt.shape[0] - len(CLASS_NAMES) - 1))
            components = vt[:k].T                                   # (1280, k)
            z_fit = (f_fit - mu) @ components
            present = np.unique(y_fit)
            class_means = np.stack([z_fit[y_fit == c].mean(axis=0) for c in present])
            residuals = np.concatenate([z_fit[y_fit == c] - class_means[j] for j, c in enumerate(present)])
            precision = LedoitWolf().fit(residuals).precision_      # shrinkage keeps it stable

            params = {"mu": mu, "components": components, "class_means": class_means, "precision": precision}
            d_cal = self._distances(params, f_cal)
            dist_threshold = float(np.percentile(d_cal, 100 - PERCENTILE)) * (1 + MARGIN)
            plant_threshold = float(np.percentile(plant_cal, PERCENTILE)) * (1 - MARGIN)
            params.update(dist_threshold=np.float64(dist_threshold), plant_threshold=np.float64(plant_threshold))

            # 4) Self-test on obvious junk: the check must reject at least 4 of these 5
            probes = _junk_probes()
            ok_probes = self._accept(params, features_fn(probes).astype(np.float64), plant_pixel_ratio(probes))
            rejected_probes = int((~ok_probes).sum())
            accepted_cal = float(self._accept(params, f_cal, plant_cal).mean())
            report = {
                "source": "computed",
                "reference_leaves": len(fit_paths) + len(cal_paths),
                "held_back_leaves_accepted": round(accepted_cal, 4),
                "junk_test_images_rejected": f"{rejected_probes}/{len(probes)}",
                "seconds": round(time.time() - t0, 1),
            }
            if rejected_probes < len(probes) - 1:
                self.calibration_report = report
                self.state = "unavailable"
                self.message = (f"Self-test failed: only {rejected_probes}/{len(probes)} obvious non-leaf test "
                                "images would be rejected, so the check is switched off (predictions are not blocked).")
                return

            params.update(model_key=np.str_(key), cache_version=np.int64(CACHE_VERSION),
                          samples_per_class=np.int64(SAMPLES_PER_CLASS),
                          n_fit=np.int64(len(fit_paths)), n_cal=np.int64(len(cal_paths)))
            tmp = CACHE_PATH.with_name("leaf_check.tmp.npz")
            np.savez(tmp, **params)
            os.replace(tmp, CACHE_PATH)
            self._install(generation, params, report,
                          f"ready: {accepted_cal:.1%} of held-back real leaves pass, "
                          f"{rejected_probes}/{len(probes)} junk test images rejected")
        except Exception as exc:  # noqa: BLE001 - never take the API down because of the check
            self.state, self.message = "unavailable", f"{type(exc).__name__}: {exc}"

    def _install(self, generation: int, params: Dict[str, Any], report: Dict[str, Any], message: str) -> None:
        with self._lock:
            if generation != self._generation:
                return                                # the model changed while calibrating: discard
            self._params = params
        self.calibration_report = report
        self.state, self.message = "ready", message

    def reset(self) -> None:
        """Called when a new model is loaded: its features differ, so recalibrate."""
        with self._lock:
            self._params = None
            self._generation += 1
        if ENABLED:
            self.state, self.message = "not started", ""

    # ------------------------------------------------------------------ scoring
    @staticmethod
    def _distances(params: Dict[str, Any], features: np.ndarray) -> np.ndarray:
        """Mahalanobis distance from each image to the CLOSEST maize-leaf class."""
        z = (np.asarray(features, dtype=np.float64).reshape(len(features), -1) - params["mu"]) @ params["components"]
        diffs = z[:, None, :] - params["class_means"][None, :, :]          # (n, classes, k)
        d2 = np.einsum("nck,kl,ncl->nc", diffs, params["precision"], diffs)
        return np.sqrt(np.maximum(d2.min(axis=1), 0.0))

    def _accept(self, params, features, plant) -> np.ndarray:
        return ((self._distances(params, features) <= params["dist_threshold"])
                & (np.asarray(plant) >= params["plant_threshold"]))

    def check(self, features: np.ndarray, x: np.ndarray) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """features: (1, 1280) from the gap layer; x: the preprocessed image (1, 224, 224, 3).
        Returns (is_maize_leaf, details). Always accepts while the check is not ready."""
        with self._lock:
            params = self._params
        if params is None:
            return True, None
        dist = float(self._distances(params, features)[0])
        plant = float(plant_pixel_ratio(x)[0])
        dist_thr, plant_thr = float(params["dist_threshold"]), float(params["plant_threshold"])
        reasons = []
        if dist > dist_thr:
            reasons.append(f"its features are {dist:.1f} away from real maize leaves, the limit is {dist_thr:.1f}")
        if plant < plant_thr:
            reasons.append(f"only {plant:.0%} of the pixels have plant colours, at least {plant_thr:.0%} expected")
        info = {"leaf_distance": round(dist, 2), "max_distance": round(dist_thr, 2),
                "plant_pixels": round(plant, 3), "min_plant_pixels": round(plant_thr, 3),
                "reason": "; ".join(reasons) if reasons else "looks like a maize leaf"}
        return not reasons, info
