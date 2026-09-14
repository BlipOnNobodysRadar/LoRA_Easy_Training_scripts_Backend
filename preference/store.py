"""Preference storage module (stdlib only).

SQLite-backed store for image-pair comparisons and human preference feedback.
No persistent connection is held; every operation uses a short-lived
transaction so generation workers and the UI can access the database
concurrently (WAL + busy timeout).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DB_FILENAME = "preferences.sqlite3"

PREFERENCES = ("a", "b", "tie", "skip", "unrated")
STRENGTHS = ("slight", "normal", "strong")
QUALITIES = ("excellent", "good", "acceptable", "bad", "especially_bad")
SPLITS = ("train", "validation")
STRENGTH_PREFERENCES = ("a", "b")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS comparisons (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_id TEXT NOT NULL UNIQUE,
    record TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    rev INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_id TEXT NOT NULL,
    feedback TEXT NOT NULL,
    undone INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_feedback_cid ON feedback(comparison_id);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_utc(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field} is not a valid ISO timestamp: {value!r}")
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware (UTC)")
    return value


class PreferenceStore:
    """Storage for comparison records and preference feedback."""

    def __init__(self, dataset_dir):
        self.root = Path(dataset_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / DB_FILENAME
        self._init_db()

    # -- low level -------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    # -- validation ------------------------------------------------------
    def _validate_record(self, record: dict) -> None:
        if not isinstance(record, dict):
            raise ValueError("record must be a dict")
        cid = record.get("id")
        if not isinstance(cid, str) or not cid:
            raise ValueError("record id must be a non-empty string")
        for field in ("prompt", "negative_prompt"):
            if not isinstance(record.get(field), str):
                raise ValueError(f"{field} must be a string")
        _validate_utc(record.get("created_at"), "created_at")
        for field in ("session_id", "group_id"):
            if not isinstance(record.get(field), str):
                raise ValueError(f"{field} must be a string")
        if record.get("split") not in SPLITS:
            raise ValueError("split must be 'train' or 'validation'")
        if not isinstance(record.get("synthetic"), bool):
            raise ValueError("synthetic must be a bool")
        if not isinstance(record.get("generation_settings"), dict):
            raise ValueError("generation_settings must be a dict")
        if not isinstance(record.get("model"), dict):
            raise ValueError("model must be a dict")
        images = record.get("images")
        if not isinstance(images, list) or len(images) < 2:
            raise ValueError("images must be a list with >= 2 entries")
        seen_ids = set()
        for image in images:
            self._validate_image(image, seen_ids)

    def _validate_image(self, image: object, seen_ids: set) -> None:
        if not isinstance(image, dict):
            raise ValueError("each image must be a dict")
        iid = image.get("id")
        if not isinstance(iid, str) or not iid:
            raise ValueError("image id must be a non-empty string")
        if iid in seen_ids:
            raise ValueError(f"duplicate image id: {iid!r}")
        seen_ids.add(iid)
        if not isinstance(image.get("seed"), int) or isinstance(image.get("seed"), bool):
            raise ValueError("image seed must be an int")
        rel = image.get("path")
        if not isinstance(rel, str) or not rel:
            raise ValueError("image path must be a non-empty string")
        target = self._resolve_image_path(rel)
        if not target.is_file():
            raise ValueError(f"image file does not exist: {rel!r}")
        digest = image.get("sha256")
        if not isinstance(digest, str) or not digest:
            raise ValueError("image sha256 must be a non-empty string")
        actual = _sha256_file(target)
        if actual.lower() != digest.lower():
            raise ValueError(
                f"sha256 mismatch for {rel!r}: expected {digest}, got {actual}"
            )

    def _resolve_image_path(self, rel: str) -> Path:
        candidate = Path(rel)
        if candidate.is_absolute():
            raise ValueError(f"image path must be relative: {rel!r}")
        # Resolve through any symlinks and verify containment in root.
        target = (self.root / candidate).resolve()
        root = self.root.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"image path escapes dataset root: {rel!r}")
        return target

    def _validate_feedback(self, feedback: dict) -> dict:
        if not isinstance(feedback, dict):
            raise ValueError("feedback must be a dict")
        pref = feedback.get("preference")
        if pref not in PREFERENCES:
            raise ValueError(f"invalid preference: {pref!r}")
        strength = feedback.get("strength")
        if pref in STRENGTH_PREFERENCES:
            if strength not in STRENGTHS:
                raise ValueError(
                    f"preference {pref!r} requires strength in {STRENGTHS}"
                )
        else:
            if strength is not None:
                raise ValueError(
                    f"preference {pref!r} requires null strength"
                )
        for field in ("quality_a", "quality_b"):
            value = feedback.get(field)
            if value is not None and value not in QUALITIES:
                raise ValueError(f"invalid {field}: {value!r}")
        reasons = feedback.get("reasons")
        if reasons is None:
            reasons = []
        if not isinstance(reasons, list) or not all(
            isinstance(r, str) for r in reasons
        ):
            raise ValueError("reasons must be a list of strings")
        _validate_utc(feedback.get("updated_at"), "updated_at")
        return dict(feedback)

    # -- public API ------------------------------------------------------
    def add_comparison(self, record: dict) -> str:
        self._validate_record(record)
        cid = record["id"]
        payload = json.dumps(record, sort_keys=True)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT 1 FROM comparisons WHERE comparison_id = ?", (cid,)
            ).fetchone()
            if row is not None:
                raise ValueError(f"duplicate comparison id: {cid!r}")
            conn.execute(
                "INSERT INTO comparisons (comparison_id, record) VALUES (?, ?)",
                (cid, payload),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return cid

    def _current_feedback(self, conn, cid: str):
        row = conn.execute(
            "SELECT feedback FROM feedback WHERE comparison_id = ? AND undone = 0 "
            "ORDER BY rev DESC LIMIT 1",
            (cid,),
        ).fetchone()
        return json.loads(row["feedback"]) if row else None

    def get(self, comparison_id: str) -> dict:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT record FROM comparisons WHERE comparison_id = ?",
                (comparison_id,),
            ).fetchone()
            if row is None:
                raise KeyError(comparison_id)
            record = json.loads(row["record"])
            record["feedback"] = self._current_feedback(conn, comparison_id)
        finally:
            conn.close()
        return record

    def list_comparisons(self, status: str = "all", split=None) -> list:
        if status not in ("all", "unrated", "rated", "eligible"):
            raise ValueError(f"invalid status: {status!r}")
        if split is not None and split not in SPLITS:
            raise ValueError(f"invalid split: {split!r}")
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT comparison_id, record FROM comparisons ORDER BY seq ASC"
            ).fetchall()
            results = []
            for row in rows:
                record = json.loads(row["record"])
                if split is not None and record.get("split") != split:
                    continue
                feedback = self._current_feedback(conn, row["comparison_id"])
                record["feedback"] = feedback
                if not _matches_status(feedback, status):
                    continue
                results.append(record)
        finally:
            conn.close()
        return results

    def put_feedback(self, comparison_id: str, feedback: dict) -> None:
        validated = self._validate_feedback(feedback)
        payload = json.dumps(validated, sort_keys=True)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT 1 FROM comparisons WHERE comparison_id = ?",
                (comparison_id,),
            ).fetchone()
            if row is None:
                raise KeyError(comparison_id)
            conn.execute(
                "INSERT INTO feedback (comparison_id, feedback) VALUES (?, ?)",
                (comparison_id, payload),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def undo_last_feedback(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT rev, comparison_id FROM feedback WHERE undone = 0 "
                "ORDER BY rev DESC LIMIT 1"
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            conn.execute("UPDATE feedback SET undone = 1 WHERE rev = ?", (row["rev"],))
            conn.execute("COMMIT")
            return row["comparison_id"]
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def counts(self) -> dict:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT comparison_id FROM comparisons"
            ).fetchall()
            total = len(rows)
            unrated = rated = eligible = ties = skipped = 0
            for row in rows:
                fb = self._current_feedback(conn, row["comparison_id"])
                pref = fb.get("preference") if fb else None
                if pref in ("a", "b"):
                    eligible += 1
                    rated += 1
                elif pref == "tie":
                    ties += 1
                    rated += 1
                elif pref == "skip":
                    skipped += 1
                    rated += 1
                else:
                    unrated += 1
        finally:
            conn.close()
        return {
            "total": total,
            "unrated": unrated,
            "rated": rated,
            "eligible": eligible,
            "ties": ties,
            "skipped": skipped,
        }

    def export_jsonl(self, path) -> None:
        target = Path(path).resolve()
        if target.suffix.lower() != ".jsonl":
            raise ValueError("Export destination must end in .jsonl")
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            rows = conn.execute(
                "SELECT comparison_id, record FROM comparisons ORDER BY seq ASC"
            ).fetchall()
            lines = []
            for row in rows:
                record = json.loads(row["record"])
                record["feedback"] = self._current_feedback(conn, row["comparison_id"])
                lines.append(json.dumps(record, sort_keys=True))
        finally:
            conn.close()
        tmp = target.with_name(target.name + "." + uuid4().hex + ".tmp")
        with open(tmp, "x", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)


def _matches_status(feedback, status: str) -> bool:
    if status == "all":
        return True
    pref = feedback.get("preference") if feedback else None
    if status == "eligible":
        return pref in ("a", "b")
    if status == "rated":
        return pref in ("a", "b", "tie", "skip")
    if status == "unrated":
        return pref not in ("a", "b", "tie", "skip")
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
