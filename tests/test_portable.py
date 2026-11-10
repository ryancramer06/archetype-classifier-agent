"""Standard-library-only tests: these also run on the Pi without pip installs."""

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent import ArchetypeAgent, ModelError, predict_request, validate_request
from config import EXPECTED_COUNTS, FEATURES, MODEL_VERSION, RANGES, ROOT
from data_utils import load_dataset
from predict import read_request, strict_json


class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = ArchetypeAgent()
        cls.valid = json.loads(
            (ROOT / "samples/requests/valid_explorer.json").read_text()
        )

    def test_all_bundled_requests(self):
        expected = json.loads((ROOT / "samples/expected.json").read_text())
        for name, case in expected.items():
            with self.subTest(name=name):
                request, response = read_request(ROOT / "samples/requests" / name)
                response = (
                    response if response is not None else self.agent.predict(request)
                )
                self.assertEqual(response["status"], case["status"])
                if case["error_code"]:
                    self.assertEqual(response["error"]["code"], case["error_code"])
                if response["status"] != "ok":
                    self.assertIsNone(response["category"])

    def test_input_order_does_not_change_prediction(self):
        shuffled = deepcopy(self.valid)
        shuffled["features"] = dict(reversed(list(shuffled["features"].items())))
        original = self.agent.predict(self.valid)
        result = self.agent.predict(shuffled)
        self.assertEqual(original["probabilities"], result["probabilities"])
        self.assertEqual(original["explanation"], result["explanation"])

    def test_all_feature_ranges_and_missing_values(self):
        for name in FEATURES:
            for change in ["missing", RANGES[name][0] - 0.01, RANGES[name][1] + 0.01]:
                request = deepcopy(self.valid)
                if change == "missing":
                    del request["features"][name]
                else:
                    request["features"][name] = change
                self.assertEqual(self.agent.predict(request)["status"], "invalid_input")

    def test_inclusive_range_boundaries(self):
        for name in FEATURES:
            for boundary in RANGES[name]:
                request = deepcopy(self.valid)
                request["features"][name] = boundary
                self.assertIn(
                    self.agent.predict(request)["status"], ["ok", "uncertain"]
                )

    def test_boolean_strings_null_and_nonfinite_rejected(self):
        for value in [
            True,
            False,
            "8.0",
            None,
            [],
            {},
            float("nan"),
            float("inf"),
            -float("inf"),
            10**400,
        ]:
            request = deepcopy(self.valid)
            request["features"]["palm_width"] = value
            self.assertEqual(self.agent.predict(request)["status"], "invalid_input")

    def test_malformed_request_shapes(self):
        for request in [None, [], 42, "hello", {}, {1: "bad key"}]:
            self.assertEqual(predict_request(request)["status"], "invalid_input")
        request = deepcopy(self.valid)
        request["features"][1] = 0
        self.assertEqual(predict_request(request)["status"], "invalid_input")

    def test_request_id_and_schema_validation(self):
        for request_id in ["", None, False, "has spaces", "x" * 81]:
            request = deepcopy(self.valid)
            request["request_id"] = request_id
            self.assertEqual(self.agent.predict(request)["status"], "invalid_input")
        request = deepcopy(self.valid)
        request["schema_version"] = "2.0.0"
        self.assertEqual(
            self.agent.predict(request)["error"]["code"], "unsupported_schema_version"
        )
        self.assertEqual(
            self.agent.predict(self.valid)["request_id"], self.valid["request_id"]
        )

    def test_extra_fields_and_target_are_rejected(self):
        request = deepcopy(self.valid)
        request["features"]["archetype"] = "leader"
        self.assertEqual(
            self.agent.predict(request)["error"]["code"], "unexpected_feature"
        )
        request = deepcopy(self.valid)
        request["person_name"] = "extra"
        self.assertEqual(
            self.agent.predict(request)["error"]["code"], "unexpected_field"
        )

    def test_json_rejects_duplicates_and_nonstandard_numbers(self):
        for text in ['{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}', "{bad"]:
            with self.assertRaises(ValueError):
                strict_json(text)

    def test_probabilities_and_warning_contract(self):
        response = self.agent.predict(self.valid)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["model_version"], MODEL_VERSION)
        self.assertEqual(response["confidence_type"], "uncalibrated_model_probability")
        self.assertAlmostEqual(sum(response["probabilities"].values()), 1, places=10)
        self.assertEqual(set(response["probabilities"]), set(EXPECTED_COUNTS))
        self.assertIn(
            "Fictional category; not a personal assessment.", response["warnings"]
        )

    def test_actual_uncertain_input(self):
        request = json.loads((ROOT / "samples/requests/uncertain.json").read_text())
        response = self.agent.predict(request)
        self.assertEqual(response["status"], "uncertain")
        self.assertIsNone(response["category"])
        self.assertLess(response["confidence"], response["acceptance_threshold"])

    def test_acceptance_boundary_with_controlled_probabilities(self):
        request = self.valid
        for probability, status in [(0.60, "ok"), (0.599999, "uncertain")]:
            output = (
                [0.0] * 6,
                [0.0] * 4,
                [probability, 0.2, 0.1, 1 - probability - 0.3],
            )
            with patch.object(self.agent, "predict_vector", return_value=output):
                self.assertEqual(self.agent.predict(request)["status"], status)

    def test_local_contributions_reconstruct_score(self):
        vector = validate_request(self.valid)
        z, scores, probabilities = self.agent.predict_vector(vector)
        winner = max(range(4), key=lambda i: probabilities[i])
        contributions = [
            value * coefficient
            for value, coefficient in zip(z, self.agent.model["coefficients"][winner])
        ]
        score = self.agent.model["intercepts"][winner] + sum(contributions)
        self.assertAlmostEqual(score, scores[winner], places=12)
        explanation = self.agent.predict(self.valid)["explanation"]
        self.assertAlmostEqual(explanation["linear_score"], score, places=7)
        expected = sorted(range(6), key=lambda i: abs(contributions[i]), reverse=True)[
            :3
        ]
        self.assertEqual(
            [row["feature"] for row in explanation["top_features"]],
            [FEATURES[i] for i in expected],
        )
        for row, index in zip(explanation["top_features"], expected):
            self.assertAlmostEqual(row["contribution"], contributions[index], places=7)
            self.assertEqual(
                row["direction"],
                (
                    "supports"
                    if contributions[index] > 0
                    else "opposes" if contributions[index] < 0 else "neutral"
                ),
            )

    def test_zero_standardized_reference_has_neutral_contributions(self):
        agent = ArchetypeAgent(threshold_override=0)
        request = deepcopy(self.valid)
        request["features"] = dict(zip(FEATURES, agent.model["mean"]))
        response = agent.predict(request)
        for row in response["explanation"]["top_features"]:
            self.assertEqual(row["contribution"], 0)
            self.assertEqual(row["direction"], "neutral")

    def test_all_saved_golden_probability_vectors(self):
        cases = json.loads((ROOT / "models/consistency_vectors.json").read_text())
        self.assertEqual(len(cases), 1000)
        for case in cases:
            actual = self.agent.predict_vector(case["features"])[2]
            self.assertLess(
                max(abs(a - b) for a, b in zip(actual, case["probabilities"])), 1e-10
            )

    def test_provided_dataset_counts_and_no_duplicates(self):
        vectors, targets, info = load_dataset()
        self.assertEqual(len(vectors), 1000)
        self.assertEqual(dict(Counter(targets)), EXPECTED_COUNTS)
        self.assertEqual(info["duplicate_feature_rows"], 0)
        self.assertEqual(
            info["csv_sha256"], self.agent.metadata["dataset"]["csv_sha256"]
        )

    def test_split_and_each_training_fold_are_disjoint(self):
        split = self.agent.metadata["split"]
        train, test = set(split["train_indices"]), set(split["test_indices"])
        self.assertEqual((len(train), len(test)), (800, 200))
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(range(1000)))
        audits = json.loads((ROOT / "reports/fold_audit.json").read_text())
        validation = []
        for row in audits:
            fitted, held = set(row["fit_row_indices"]), set(
                row["validation_row_indices"]
            )
            self.assertEqual((len(fitted), len(held)), (640, 160))
            self.assertEqual(row["scaler_samples_seen"], 640)
            self.assertFalse(fitted & held or fitted & test or held & test)
            self.assertEqual(fitted | held, train)
            validation.extend(held)
        self.assertEqual(Counter(validation), Counter({index: 1 for index in train}))

    def test_scaler_parameters_use_training_rows_only(self):
        vectors, _, _ = load_dataset()
        train = self.agent.metadata["split"]["train_indices"]
        self.assertEqual(self.agent.metadata["scaler_samples_seen"], 800)
        for j in range(6):
            mean = statistics.fmean(vectors[i][j] for i in train)
            self.assertAlmostEqual(mean, self.agent.model["mean"][j], places=11)

    def test_sample_requests_come_from_training_rows(self):
        training = set(self.agent.metadata["split"]["train_indices"])
        origins = json.loads((ROOT / "samples/provenance.json").read_text())
        self.assertEqual(len(origins), 5)
        self.assertTrue(all(row["row_index"] in training for row in origins.values()))

    def test_missing_model_and_corrupt_export(self):
        response = predict_request(self.valid, model_dir=ROOT / "models/missing")
        self.assertEqual(response["status"], "unavailable")
        self.assertIsNone(response["category"])
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test_tmp_") as folder:
            path = Path(folder)
            (path / "metadata.json").write_text(json.dumps(self.agent.metadata))
            bad = deepcopy(self.agent.model)
            bad["scale"][0] = 0
            (path / "portable_model.json").write_text(json.dumps(bad))
            self.assertEqual(
                predict_request(self.valid, model_dir=path)["status"], "unavailable"
            )

    def test_prediction_does_not_modify_saved_models(self):
        paths = list((ROOT / "models").iterdir())
        before = {p.name: p.stat().st_mtime_ns for p in paths}
        self.agent.predict(self.valid)
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in paths})

    def test_cli_statuses_and_saved_json(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test_tmp_") as folder:
            output = Path(folder) / "response.json"
            for name, code in [
                ("valid_explorer.json", 0),
                ("uncertain.json", 1),
                ("boolean_feature.json", 2),
            ]:
                run = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "predict.py"),
                        "--backend",
                        "portable",
                        "--request",
                        str(ROOT / "samples/requests" / name),
                        "--output",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(run.returncode, code, run.stderr)
                self.assertEqual(json.loads(run.stdout), json.loads(output.read_text()))

    def test_prediction_runs_with_site_packages_disabled(self):
        run = subprocess.run(
            [
                sys.executable,
                "-S",
                str(ROOT / "predict.py"),
                "--backend",
                "portable",
                "--request",
                str(ROOT / "samples/requests/valid_explorer.json"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "ok")


if __name__ == "__main__":
    unittest.main()
