"""Regression checks for damaged artifacts and CLI failure responses."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from agent import ArchetypeAgent, predict_request
from config import ROOT


class PackageFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = ArchetypeAgent()
        cls.request = json.loads(
            (ROOT / "samples/requests/valid_explorer.json").read_text()
        )

    def assert_unavailable(
        self, model=None, metadata=None, pipeline=None, backend="portable"
    ):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "portable_model.json").write_text(
                json.dumps(model if model is not None else self.agent.model)
            )
            meta = deepcopy(metadata if metadata is not None else self.agent.metadata)
            if pipeline is not None:
                (path / "pipeline.joblib").write_bytes(pipeline)
                meta["pipeline_sha256"] = hashlib.sha256(pipeline).hexdigest()
            (path / "metadata.json").write_text(json.dumps(meta))
            result = predict_request(self.request, model_dir=path, backend=backend)
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["error"]["code"], "model_unavailable")
            self.assertIsNone(result["category"])

    def test_rejects_unsupported_probability_method(self):
        model = deepcopy(self.agent.model)
        model["probability_method"] = "unsupported"
        self.assert_unavailable(model=model)

    def test_rejects_model_and_metadata_schema_mismatches(self):
        for location in ["model", "metadata"]:
            with self.subTest(location=location):
                model, meta = deepcopy(self.agent.model), deepcopy(self.agent.metadata)
                (model if location == "model" else meta)["schema_version"] = "2.0.0"
                self.assert_unavailable(model=model, metadata=meta)

    def test_rejects_boolean_model_parameters(self):
        for field in ["mean", "scale", "intercepts", "coefficients"]:
            with self.subTest(field=field):
                model = deepcopy(self.agent.model)
                if field == "coefficients":
                    model[field][0][0] = True
                else:
                    model[field][0] = True
                self.assert_unavailable(model=model)

    def test_oversized_model_integer_returns_unavailable(self):
        model = deepcopy(self.agent.model)
        model["mean"][0] = 10**400
        self.assert_unavailable(model=model)

    def test_boolean_saved_threshold_is_rejected(self):
        metadata = deepcopy(self.agent.metadata)
        metadata["acceptance_threshold"] = True
        self.assert_unavailable(metadata=metadata)

    @unittest.skipUnless(
        importlib.util.find_spec("joblib"), "Native dependencies not installed"
    )
    def test_truncated_pipeline_returns_unavailable(self):
        self.assert_unavailable(pipeline=b"", backend="sklearn")

    def test_cli_missing_model_returns_json_and_exit_three(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "predict.py"),
                    "--backend",
                    "portable",
                    "--request",
                    str(ROOT / "samples/requests/valid_explorer.json"),
                    "--model-dir",
                    folder,
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 3)
        self.assertEqual(json.loads(result.stdout)["status"], "unavailable")
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_stdin_prediction(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "predict.py"),
                "--backend",
                "portable",
                "--stdin",
            ],
            input=json.dumps(self.request),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["category"], "explorer")

    def test_cli_output_failure_preserves_response_and_exit_four(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "predict.py"),
                    "--backend",
                    "portable",
                    "--request",
                    str(ROOT / "samples/requests/valid_explorer.json"),
                    "--output",
                    folder,
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 4)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")
        self.assertIn("Could not write response", result.stderr)


if __name__ == "__main__":
    unittest.main()
