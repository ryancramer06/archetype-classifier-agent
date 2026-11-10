"""Check readable output and interactive/noninteractive CLI behavior."""

from contextlib import redirect_stdout
import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from config import ROOT
import demo


class DemoTests(unittest.TestCase):
    def run_demo(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "demo.py"), "--backend", "portable", *args],
            capture_output=True,
            text=True,
        )

    def test_summary_covers_predictions_and_rejections(self):
        result = self.run_demo("--auto")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Status:"), 5)
        for text in [
            "Category: explorer",
            "99.85%",
            "Status: uncertain",
            "48.30%",
            "nonnumeric_feature",
            "unexpected_feature",
            "out_of_range",
            "eye_distance: +3.326",
        ]:
            self.assertIn(text, result.stdout)
        self.assertNotIn("Press Enter", result.stdout)
        self.assertNotIn("schema_version", result.stdout)

    def test_json_format_retains_full_responses(self):
        result = self.run_demo("--auto", "--format", "json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('"schema_version": "1.0.0"'), 5)
        self.assertIn(
            '"confidence_type": "uncalibrated_model_probability"', result.stdout
        )

    def test_interactive_mode_pauses_between_examples(self):
        with (
            patch.object(sys, "argv", ["demo.py", "--backend", "portable"]),
            patch("builtins.input", return_value="") as pause,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(demo.main(), 0)
        self.assertEqual(pause.call_count, 5)

    def test_end_of_input_exits_cleanly(self):
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["demo.py", "--backend", "portable"]),
            patch("builtins.input", side_effect=EOFError),
            redirect_stdout(output),
        ):
            self.assertEqual(demo.main(), 0)
        self.assertIn("Use --auto", output.getvalue())

    def test_missing_model_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_demo("--auto", "--model-dir", folder)
        self.assertEqual(result.returncode, 3)
        self.assertIn("Demo unavailable", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
