"""
SQLite database for the retraining pipeline: data/pipeline.db

Two tables:

    uploads        one row per uploaded image (whether it was accepted or not)
                   status:  pending          saved in data/uploads/<class>/, waiting for the next retraining
                            in_training_set  moved into data/train/<class>/ by a retraining run
                            duplicate        identical image already uploaded before (same MD5)
                            rejected         corrupt file or not a maize leaf (see `reason`)

    retrain_runs   one row per retraining run: trigger, versions, accuracy before/after, outcome

Why a database and not only folders? The folders hold the image FILES; the database holds the
FACTS about them: who/when/what label, why an image was refused, and which retraining run (and
therefore which model version) learned from it. That makes the pipeline auditable and lets the
API spot duplicates across separate uploads.

SQLite is part of Python's standard library: no server to install. The file lives in data/, which
docker-compose shares between API replicas, and every write is a short transaction with a busy
timeout, so several replicas can use it safely (writes happen only on uploads and retraining, not
on every prediction).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src import DATA_DIR

DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "pipeline.db"))
_init_lock = threading.Lock()
_initialized = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_at        TEXT    NOT NULL,
    batch_id           TEXT    NOT NULL,              -- all files sent in one request share a batch id
    original_filename  TEXT    NOT NULL,
    label              TEXT,                          -- Blight | Common_Rust | Gray_Leaf_Spot | Healthy
    md5                TEXT,                          -- fingerprint of the original bytes (duplicate check)
    status             TEXT    NOT NULL,              -- pending | in_training_set | duplicate | rejected
    reason             TEXT,                          -- why it was rejected / marked duplicate
    stored_path        TEXT,                          -- data/uploads/... while pending, data/train/... after merge
    width              INTEGER,
    height             INTEGER,
    retrain_run_id     INTEGER REFERENCES retrain_runs(id)
);
CREATE INDEX IF NOT EXISTS idx_uploads_md5    ON uploads(md5);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);

CREATE TABLE IF NOT EXISTS retrain_runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at         TEXT    NOT NULL,
    finished_at        TEXT,
    trigger            TEXT,
    status             TEXT    NOT NULL,              -- running | completed | rejected | failed
    from_version       TEXT,
    to_version         TEXT,
    new_images         INTEGER,
    train_samples      INTEGER,
    accuracy_before    REAL,
    accuracy_after     REAL,
    f1_after           REAL,
    message            TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    """Short-lived connection per operation (safe across threads and processes)."""
    global _initialized
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)          # wait up to 30 s if another writer holds the lock
    conn.row_factory = sqlite3.Row
    try:
        if not _initialized:
            with _init_lock:
                if not _initialized:
                    conn.executescript(SCHEMA)
                    _initialized = True
        yield conn
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------------------- uploads
def is_known_image(md5: str) -> Optional[Dict[str, Any]]:
    """An earlier accepted upload with the same fingerprint (pending or already trained on)."""
    with connect() as db:
        row = db.execute("SELECT id, uploaded_at, label, status FROM uploads "
                         "WHERE md5 = ? AND status IN ('pending', 'in_training_set') LIMIT 1", (md5,)).fetchone()
    return dict(row) if row else None


def record_upload(batch_id: str, filename: str, label: Optional[str], md5: Optional[str], status: str,
                  reason: Optional[str] = None, stored_path: Optional[str] = None,
                  width: Optional[int] = None, height: Optional[int] = None) -> int:
    with connect() as db:
        cur = db.execute(
            "INSERT INTO uploads (uploaded_at, batch_id, original_filename, label, md5, status, reason, "
            "stored_path, width, height) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), batch_id, filename, label, md5, status, reason, stored_path, width, height))
        return int(cur.lastrowid)


def mark_merged(moves: Iterable[tuple], run_id: Optional[int]) -> None:
    """Retraining moved files from data/uploads to data/train: update their rows.
    `moves` = [(old_path, new_path, label), ...]. Files that have no row (uploaded before the database
    existed) get one, so every image in the training set added after deployment is accounted for."""
    with connect() as db:
        for old, new, label in moves:
            cur = db.execute("UPDATE uploads SET status = 'in_training_set', stored_path = ?, retrain_run_id = ? "
                             "WHERE stored_path = ? AND status = 'pending'", (str(new), run_id, str(old)))
            if cur.rowcount == 0:
                db.execute("INSERT INTO uploads (uploaded_at, batch_id, original_filename, label, status, reason, "
                           "stored_path, retrain_run_id) VALUES (?, 'pre-database', ?, ?, 'in_training_set', "
                           "'uploaded before the database existed', ?, ?)",
                           (_now(), Path(old).name, label, str(new), run_id))


def list_uploads(limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
    query = ("SELECT u.*, r.to_version AS trained_into_version, r.status AS retrain_status "
             "FROM uploads u LEFT JOIN retrain_runs r ON u.retrain_run_id = r.id")
    args: list = []
    if status:
        query += " WHERE u.status = ?"
        args.append(status)
    query += " ORDER BY u.id DESC LIMIT ?"
    args.append(limit)
    with connect() as db:
        return [dict(r) for r in db.execute(query, args).fetchall()]


def upload_summary() -> Dict[str, Any]:
    with connect() as db:
        by_status = {r["status"]: r["n"] for r in
                     db.execute("SELECT status, COUNT(*) AS n FROM uploads GROUP BY status")}
        by_label = {r["label"]: r["n"] for r in
                    db.execute("SELECT label, COUNT(*) AS n FROM uploads "
                               "WHERE status IN ('pending', 'in_training_set') GROUP BY label")}
        batches = db.execute("SELECT COUNT(DISTINCT batch_id) AS n FROM uploads").fetchone()["n"]
    return {"total": sum(by_status.values()), "by_status": by_status,
            "accepted_by_label": by_label, "upload_batches": batches, "database": str(DB_PATH)}


# ----------------------------------------------------------------------------- retraining runs
def start_run(trigger: str, from_version: str) -> int:
    with connect() as db:
        cur = db.execute("INSERT INTO retrain_runs (started_at, trigger, status, from_version) VALUES (?, ?, 'running', ?)",
                         (_now(), trigger, from_version))
        return int(cur.lastrowid)


def finish_run(run_id: Optional[int], **fields: Any) -> None:
    if run_id is None:
        return
    allowed = {"status", "to_version", "new_images", "train_samples", "accuracy_before",
               "accuracy_after", "f1_after", "message"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    fields["finished_at"] = _now()
    with connect() as db:
        db.execute(f"UPDATE retrain_runs SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                   (*fields.values(), run_id))


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    with connect() as db:
        return [dict(r) for r in db.execute(
            "SELECT r.*, (SELECT COUNT(*) FROM uploads u WHERE u.retrain_run_id = r.id) AS images_from_uploads "
            "FROM retrain_runs r ORDER BY r.id DESC LIMIT ?", (limit,)).fetchall()]
