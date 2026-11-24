"""Shared image preprocessing for training, retraining and prediction.

Every image that reaches the model goes through ``preprocess_pixels``:
    decode -> RGB uint8 -> resize to 224x224 (bilinear) -> scale to [-1, 1]
Scaling to [-1, 1] (x / 127.5 - 1) centres the pixel values on zero, which helps the CNN train. The notebook uses an identical copy of this function, so the
model in production sees pixels prepared the same way as in training.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

from src import CLASS_NAMES, IMG_SIZE


def _configure_tf_threads() -> None:
    """Apply TF_NUM_INTRAOP_THREADS / TF_NUM_INTEROP_THREADS if they are set.

    In Docker each API replica is limited to 1 CPU (see docker-compose.yml). Without this,
    TensorFlow starts one thread per HOST core and replicas slow each other down.
    Must run before TensorFlow executes its first operation, so it runs at import time.
    """
    import os

    try:
        if os.getenv("TF_NUM_INTRAOP_THREADS"):
            tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_NUM_INTRAOP_THREADS"]))
        if os.getenv("TF_NUM_INTEROP_THREADS"):
            tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["TF_NUM_INTEROP_THREADS"]))
    except (RuntimeError, ValueError):
        pass  # TensorFlow already initialised (e.g. imported earlier in a notebook): keep defaults


_configure_tf_threads()

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

# Folder-name variants seen in Kaggle copies of the dataset -> canonical class name.
_CLASS_ALIASES = {
    "blight": "Blight",
    "northern_leaf_blight": "Blight",
    "leaf_blight": "Blight",
    "common_rust": "Common_Rust",
    "rust": "Common_Rust",
    "gray_leaf_spot": "Gray_Leaf_Spot",
    "grey_leaf_spot": "Gray_Leaf_Spot",
    "cercospora_leaf_spot_gray_leaf_spot": "Gray_Leaf_Spot",
    "healthy": "Healthy",
}


# --------------------------------------------------------------------------- #
# Class-name helpers
# --------------------------------------------------------------------------- #
def normalize_class_name(name: str) -> Optional[str]:
    """Map a folder name or user label to one of CLASS_NAMES (or None).

    Handles case, spaces, hyphens and prefixes such as 'Corn_(maize)___Common_rust_'.
    """
    if not name:
        return None
    key = re.sub(r"[^a-z]+", "_", name.lower()).strip("_")
    key = re.sub(r"^corn_maize_+", "", key)      # PlantVillage-style prefix
    key = re.sub(r"^(corn|maize)_+", "", key)
    if key in _CLASS_ALIASES:
        return _CLASS_ALIASES[key]
    for canonical in CLASS_NAMES:                # exact canonical match (any case)
        if key == canonical.lower():
            return canonical
    return None


def is_image_file(path: Path | str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


# --------------------------------------------------------------------------- #
# Validation / standardisation (used by the notebook cleaner and /upload-data)
# --------------------------------------------------------------------------- #
def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def standardize_image_bytes(image_bytes: bytes) -> Image.Image:
    """Open bytes, fully decode them (catches truncated files), fix EXIF rotation,
    and return a 3-channel RGB PIL image. Raises ValueError for unreadable images."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            probe.verify()                       # structural check
        img = Image.open(io.BytesIO(image_bytes))
        img.load()                               # full decode
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if min(img.size) < 32:
            raise ValueError(f"image too small: {img.size}")
        return img
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError(f"corrupt or unsupported image: {exc}") from exc


def save_as_jpeg(img: Image.Image, dest: Path, quality: int = 95) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="JPEG", quality=quality)


# --------------------------------------------------------------------------- #
# Core preprocessing (shared by training and inference)
# --------------------------------------------------------------------------- #
def preprocess_pixels(image: tf.Tensor) -> tf.Tensor:
    """uint8/float RGB tensor (H, W, 3), values 0-255 -> float32 (224, 224, 3) in [-1, 1]."""
    image = tf.image.resize(tf.cast(image, tf.float32), (IMG_SIZE, IMG_SIZE), method="bilinear")
    return image / 127.5 - 1.0


def decode_image_bytes(image_bytes: bytes) -> tf.Tensor:
    """Decode raw bytes to an RGB uint8 tensor using the same decoder as training.
    Falls back to Pillow for formats TensorFlow cannot decode (e.g. WebP, TIFF)."""
    try:
        img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    except (tf.errors.InvalidArgumentError, ValueError):
        pil = standardize_image_bytes(image_bytes)
        img = tf.convert_to_tensor(np.asarray(pil, dtype=np.uint8))
    return img


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Single uploaded image -> batch of one, shape (1, 224, 224, 3), ready for model()."""
    img = decode_image_bytes(image_bytes)
    return preprocess_pixels(img)[tf.newaxis, ...].numpy()


def preprocess_image_batch(list_of_bytes: Iterable[bytes]) -> np.ndarray:
    """Several images -> (N, 224, 224, 3)."""
    return np.stack([preprocess_pixels(decode_image_bytes(b)).numpy() for b in list_of_bytes])


# --------------------------------------------------------------------------- #
# Directory -> tf.data pipelines
# --------------------------------------------------------------------------- #
def list_image_files(directory: Path | str, class_names: List[str] = CLASS_NAMES) -> Tuple[List[str], List[int]]:
    """Return (file_paths, integer_labels) for directory/<class_name>/*.jpg ..."""
    directory = Path(directory)
    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = directory / cls
        if not cls_dir.is_dir():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.is_file() and is_image_file(f):
                paths.append(str(f))
                labels.append(idx)
    return paths, labels


def _load_from_path(path: tf.Tensor, label: tf.Tensor):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(tf.cast(img, tf.float32), (IMG_SIZE, IMG_SIZE), method="bilinear")
    return img, label                            # still 0-255 so augmentation is intuitive


def _augment(img: tf.Tensor, label: tf.Tensor):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, max_delta=25.0)       # field lighting varies
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 255.0), label


def _scale(img: tf.Tensor, label: tf.Tensor):
    return img / 127.5 - 1.0, label              # same as preprocess_pixels after resize


def build_dataset(paths: List[str], labels: List[int], training: bool,
                  batch_size: int = 32, seed: int = 42) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load_from_path, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.map(_scale, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def dataset_from_directory(directory: Path | str, training: bool, batch_size: int = 32):
    """Convenience wrapper: returns (dataset, paths, labels)."""
    paths, labels = list_image_files(directory)
    if not paths:
        raise FileNotFoundError(f"No images found under {directory}/<class_name>/")
    return build_dataset(paths, labels, training, batch_size), paths, labels
