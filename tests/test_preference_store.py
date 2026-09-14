"""Tests for backend.preference.store (stdlib only, no Qt/torch/network)."""

import hashlib
import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

from backend.preference.store import PreferenceStore


def _now():
    return datetime.now(timezone.utc).isoformat()


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "images").mkdir()
        self.store = PreferenceStore(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    # -- helpers ---------------------------------------------------------
    def _write_image(self, name, data=b"tiny-png-bytes"):
        path = self.root / "images" / name
        path.write_bytes(data)
        return name, hashlib.sha256(data).hexdigest()

    def _record(self, cid="c1", images=None, **overrides):
        if images is None:
            images = []
            for iid in ("a", "b"):
                name, digest = self._write_image(f"{cid}-{iid}.png", b"png-" + cid.encode() + iid.encode())
                images.append(
                    {"id": iid, "path": f"images/{name}", "seed": 123, "sha256": digest}
                )
        record = {
            "id": cid,
            "prompt": "a prompt",
            "negative_prompt": "",
            "created_at": _now(),
            "session_id": "s1",
            "group_id": "g1",
            "split": "train",
            "synthetic": False,
            "generation_settings": {"steps": 25, "cfg": 7.0},
            "model": {"family": "sdxl", "checkpoint": "x.safetensors"},
            "images": images,
            "extra_unknown": {"nested": [1, 2, 3]},
        }
        record.update(overrides)
        return record

    def _feedback(self, preference="a", strength="normal", **overrides):
        fb = {
            "preference": preference,
            "strength": strength,
            "quality_a": None,
            "quality_b": None,
            "reasons": [],
            "updated_at": _now(),
        }
        fb.update(overrides)
        return fb

    # -- tests -----------------------------------------------------------
    def test_root_created_and_absolute(self):
        self.assertTrue(self.store.root.is_absolute())
        self.assertTrue((self.root / "preferences.sqlite3").exists())

    def test_add_get_roundtrip_and_insertion_order(self):
        self.store.add_comparison(self._record("c1"))
        self.store.add_comparison(self._record("c2"))
        self.store.add_comparison(self._record("c3"))
        self.assertEqual([c["id"] for c in self.store.list_comparisons()], ["c1", "c2", "c3"])
        got = self.store.get("c1")
        self.assertEqual(got["extra_unknown"], {"nested": [1, 2, 3]})
        self.assertIsNone(got["feedback"])
        with self.assertRaises(KeyError):
            self.store.get("missing")

    def test_duplicate_id_rejected(self):
        self.store.add_comparison(self._record("c1"))
        with self.assertRaises(ValueError):
            self.store.add_comparison(self._record("c1"))

    def test_reopen_durability(self):
        self.store.add_comparison(self._record("c1"))
        self.store.put_feedback("c1", self._feedback("b", "strong"))
        reopened = PreferenceStore(self.root)
        got = reopened.get("c1")
        self.assertEqual(got["feedback"]["preference"], "b")
        self.assertEqual(got["feedback"]["strength"], "strong")
        self.assertEqual(reopened.counts()["eligible"], 1)

    def test_richer_fields_preserved(self):
        rec = self._record("c1")
        rec["feedback_notes"] = {"anything": True}
        rec["images"][0]["future_field"] = [1, 2]
        rec["images"].append(
            {
                "id": "c",
                "path": f"images/{self._write_image('c1-c.png', b'third')[0]}",
                "seed": 9,
                "sha256": self._write_image("c1-c.png", b"third")[1],
            }
        )
        self.store.add_comparison(rec)
        got = self.store.get("c1")
        self.assertEqual(len(got["images"]), 3)
        self.assertEqual(got["images"][0]["future_field"], [1, 2])
        self.assertEqual(got["feedback_notes"], {"anything": True})

    def test_both_bad_and_tie_semantics(self):
        self.store.add_comparison(self._record("c1"))
        # both-bad quality with a tie preference is valid, strength null
        self.store.put_feedback(
            "c1",
            self._feedback("tie", None, quality_a="especially_bad", quality_b="especially_bad"),
        )
        fb = self.store.get("c1")["feedback"]
        self.assertEqual(fb["preference"], "tie")
        self.assertIsNone(fb["strength"])
        self.assertEqual(fb["quality_a"], "especially_bad")
        # quality-only feedback with unrated preference is valid
        self.store.put_feedback(
            "c1", self._feedback("unrated", None, quality_a="good")
        )
        counts = self.store.counts()
        self.assertEqual(counts["unrated"], 1)
        self.assertEqual(counts["ties"], 0)
        self.assertEqual(counts["eligible"], 0)

    def test_counts_and_status_filters(self):
        for cid in ("c1", "c2", "c3", "c4", "c5"):
            self.store.add_comparison(self._record(cid))
        self.store.put_feedback("c1", self._feedback("a", "slight"))
        self.store.put_feedback("c2", self._feedback("b", "strong"))
        self.store.put_feedback("c3", self._feedback("tie", None))
        self.store.put_feedback("c4", self._feedback("skip", None))
        counts = self.store.counts()
        self.assertEqual(counts, {"total": 5, "unrated": 1, "rated": 4, "eligible": 2, "ties": 1, "skipped": 1})
        self.assertEqual(len(self.store.list_comparisons("eligible")), 2)
        self.assertEqual(len(self.store.list_comparisons("rated")), 4)
        self.assertEqual([c["id"] for c in self.store.list_comparisons("unrated")], ["c5"])
        self.assertEqual(len(self.store.list_comparisons(split="train")), 5)
        self.assertEqual(self.store.list_comparisons(split="validation"), [])

    def test_undo_restores_previous_and_retains_history(self):
        self.store.add_comparison(self._record("c1"))
        self.store.put_feedback("c1", self._feedback("a", "normal"))
        self.store.put_feedback("c1", self._feedback("b", "strong"))
        self.assertEqual(self.store.get("c1")["feedback"]["preference"], "b")
        self.assertEqual(self.store.undo_last_feedback(), "c1")
        self.assertEqual(self.store.get("c1")["feedback"]["preference"], "a")
        self.assertEqual(self.store.undo_last_feedback(), "c1")
        self.assertIsNone(self.store.get("c1")["feedback"])
        self.assertIsNone(self.store.undo_last_feedback())
        # history retained in the feedback table
        import sqlite3

        conn = sqlite3.connect(self.root / "preferences.sqlite3")
        rows = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        conn.close()
        self.assertEqual(rows, 2)

    def test_invalid_records(self):
        bad = self._record("c1")
        bad["split"] = "test"
        with self.assertRaises(ValueError):
            self.store.add_comparison(bad)
        with self.assertRaises(ValueError):
            self.store.add_comparison(self._record("c2", prompt=5))
        with self.assertRaises(ValueError):
            self.store.add_comparison(self._record("c3", synthetic="yes"))
        one_image = self._record("c4")
        one_image["images"] = one_image["images"][:1]
        with self.assertRaises(ValueError):
            self.store.add_comparison(one_image)

    def test_invalid_feedback(self):
        self.store.add_comparison(self._record("c1"))
        for fb in (
            self._feedback("z", "normal"),
            self._feedback("a", None),
            self._feedback("tie", "normal"),
            self._feedback("a", "normal", quality_a="meh"),
            self._feedback("a", "normal", reasons="notalist"),
            self._feedback("a", "normal", updated_at="not-a-date"),
        ):
            with self.assertRaises(ValueError):
                self.store.put_feedback("c1", fb)
        with self.assertRaises(KeyError):
            self.store.put_feedback("missing", self._feedback("a", "normal"))

    def test_image_path_and_hash_validation(self):
        outside = self.root.parent / "outside.png"
        outside.write_bytes(b"outside")
        rec = self._record("c1")
        rec["images"][0]["path"] = "../outside.png"
        with self.assertRaises(ValueError):
            self.store.add_comparison(rec)

        rec = self._record("c2")
        rec["images"][0]["path"] = str(outside)
        with self.assertRaises(ValueError):
            self.store.add_comparison(rec)

        rec = self._record("c3")
        rec["images"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.store.add_comparison(rec)

        rec = self._record("c4")
        rec["images"][0]["path"] = "images/does-not-exist.png"
        with self.assertRaises(ValueError):
            self.store.add_comparison(rec)

        # symlink escape
        link = self.root / "images" / "escape.png"
        try:
            os.symlink(outside, link)
        except OSError:
            return
        rec = self._record("c5")
        rec["images"][0]["path"] = "images/escape.png"
        rec["images"][0]["sha256"] = hashlib.sha256(b"outside").hexdigest()
        with self.assertRaises(ValueError):
            self.store.add_comparison(rec)

    def test_export_jsonl(self):
        self.store.add_comparison(self._record("c1"))
        self.store.add_comparison(self._record("c2"))
        self.store.put_feedback("c1", self._feedback("a", "slight"))
        out = self.root / "export" / "out.jsonl"
        self.store.export_jsonl(out)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["id"], "c1")
        self.assertEqual(first["feedback"]["preference"], "a")
        second = json.loads(lines[1])
        self.assertIsNone(second["feedback"])

    def test_concurrent_independent_instances(self):
        store_b = PreferenceStore(self.root)
        for i in range(8):
            self.store.add_comparison(self._record(f"c{i}"))
        # store_b sees records written by store
        self.assertEqual(len(store_b.list_comparisons()), 8)

        errors = []

        def worker(cid):
            try:
                local = PreferenceStore(self.root)
                local.put_feedback(cid, self._feedback("a", "normal"))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"c{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(store_b.counts()["eligible"], 8)


if __name__ == "__main__":
    unittest.main()
