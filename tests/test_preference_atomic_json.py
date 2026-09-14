"""Status publication must survive transient Windows reader locks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.preference.config import atomic_json


class AtomicJsonTests(unittest.TestCase):
    def test_transient_reader_lock_preserves_old_file_until_replace(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "status.json"
            atomic_json(path, {"step": 86})
            import os
            original_replace = os.replace
            attempts = []

            def locked_then_available(source, destination):
                attempts.append(1)
                self.assertEqual(json.loads(path.read_text()), {"step": 86})
                if len(attempts) < 3:
                    raise PermissionError("simulated Windows reader lock")
                return original_replace(source, destination)

            with patch("backend.preference.config.os.replace", side_effect=locked_then_available), patch("backend.preference.config.time.sleep"):
                atomic_json(path, {"step": 87})
            self.assertEqual(json.loads(path.read_text()), {"step": 87})
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_persistent_access_denial_is_reported_without_destroying_status(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "status.json"
            atomic_json(path, {"step": 86})
            with patch("backend.preference.config.os.replace", side_effect=PermissionError("denied")), patch("backend.preference.config.time.sleep"):
                with self.assertRaises(PermissionError):
                    atomic_json(path, {"step": 87})
            self.assertEqual(json.loads(path.read_text()), {"step": 86})
            self.assertEqual(list(Path(folder).iterdir()), [path])
