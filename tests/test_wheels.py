"""Test checksum and path handling without downloading packages."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_wheels import verify_wheels


class WheelCacheTests(unittest.TestCase):
    def make_cache(self, folder, filename="example.whl"):
        folder = Path(folder)
        payload = b"checksum test fixture"
        (folder / "example.whl").write_bytes(payload)
        manifest = {
            "target": "test fixture",
            "files": [
                {"filename": filename, "sha256": hashlib.sha256(payload).hexdigest()}
            ],
        }
        (folder / "MANIFEST.json").write_text(json.dumps(manifest))
        return folder

    def test_valid_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            result = verify_wheels(self.make_cache(folder))
        self.assertEqual(result["files"][0]["filename"], "example.whl")

    def test_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_cache(folder)
            (path / "example.whl").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_wheels(path)

    def test_missing_wheel(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_cache(folder)
            (path / "example.whl").unlink()
            with self.assertRaises(OSError):
                verify_wheels(path)

    def test_rejects_path_outside_cache(self):
        for filename in ["../example.whl", "..\\example.whl", "/example.whl"]:
            with (
                self.subTest(filename=filename),
                tempfile.TemporaryDirectory() as folder,
            ):
                with self.assertRaises(ValueError):
                    verify_wheels(self.make_cache(folder, filename))


if __name__ == "__main__":
    unittest.main()
