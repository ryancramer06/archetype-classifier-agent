"""Full sklearn tests; install requirements-core.txt before running this file."""

import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

try:
    import joblib
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    FULL_AVAILABLE = True
except ImportError:
    FULL_AVAILABLE = False

from agent import ArchetypeAgent
from config import ROOT
from data_utils import load_dataset


@unittest.skipUnless(
    FULL_AVAILABLE,
    "Optional sklearn dependencies are not installed; run test_portable.py on the lightweight Pi setup.",
)
class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = joblib.load(ROOT / "models/pipeline.joblib")
        vectors, targets, _ = load_dataset()
        cls.X, cls.y = np.asarray(vectors), np.asarray(targets)
        cls.meta = json.loads((ROOT / "models/metadata.json").read_text())

    def test_saved_complete_pipeline(self):
        self.assertIsInstance(self.pipeline, Pipeline)
        self.assertIsInstance(self.pipeline.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(
            self.pipeline.named_steps["classifier"], LogisticRegression
        )
        self.assertEqual(self.pipeline.named_steps["scaler"].n_samples_seen_, 800)
        self.assertEqual(
            self.pipeline.named_steps["classifier"].classes_.tolist(),
            self.meta["class_order"],
        )

    def test_split_preview_matches_saved_training_ids(self):
        from train import split_indices

        training, test = split_indices(self.y)
        self.assertEqual(training.tolist(), self.meta["split"]["train_indices"])
        self.assertEqual(test.tolist(), self.meta["split"]["test_indices"])

    def test_split_preview_cli_does_not_change_model_files(self):
        before = {p.name: p.stat().st_mtime_ns for p in (ROOT / "models").iterdir()}
        result = subprocess.run(
            [sys.executable, str(ROOT / "train.py"), "--preview-split"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(
            (summary["training_rows"], summary["test_rows"], summary["overlap"]),
            (800, 200, 0),
        )
        self.assertEqual(summary["training_counts"], self.meta["split"]["train_counts"])
        self.assertEqual(summary["test_counts"], self.meta["split"]["test_counts"])
        self.assertFalse(summary["model_fitted"])
        self.assertEqual(
            before, {p.name: p.stat().st_mtime_ns for p in (ROOT / "models").iterdir()}
        )

    def test_joblib_save_reload_prediction_consistency(self):
        before = self.pipeline.predict_proba(self.X)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test_tmp_") as folder:
            path = Path(folder) / "roundtrip.joblib"
            joblib.dump(self.pipeline, path)
            reloaded = joblib.load(path)
            self.assertTrue(np.array_equal(before, reloaded.predict_proba(self.X)))
            self.assertTrue(
                np.array_equal(self.pipeline.predict(self.X), reloaded.predict(self.X))
            )

    def test_portable_matches_sklearn_for_all_1000_rows(self):
        portable = ArchetypeAgent()
        expected = self.pipeline.predict_proba(self.X)
        actual = np.asarray([portable.predict_vector(row)[2] for row in self.X])
        self.assertLess(np.max(np.abs(expected - actual)), 1e-10)
        self.assertTrue(np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1)))

    def test_both_agent_backends_agree(self):
        portable, native = ArchetypeAgent(), ArchetypeAgent(backend="sklearn")
        for name in ["valid_explorer.json", "valid_creator.json", "uncertain.json"]:
            request = json.loads((ROOT / "samples/requests" / name).read_text())
            a, b = portable.predict(request), native.predict(request)
            self.assertEqual((a["status"], a["category"]), (b["status"], b["category"]))
            for label in self.meta["class_order"]:
                self.assertAlmostEqual(
                    a["probabilities"][label], b["probabilities"][label], places=10
                )

    def test_reported_heldout_metrics_reproduce(self):
        from evaluation_utils import evaluate_predictions

        ids = self.meta["split"]["test_indices"]
        expected = json.loads((ROOT / "reports/metrics.json").read_text())[
            "selected_model"
        ]
        actual = evaluate_predictions(
            self.y[ids],
            self.pipeline.predict(self.X[ids]),
            self.pipeline.predict_proba(self.X[ids]),
            self.meta["class_order"],
            self.meta["acceptance_threshold"],
        )
        self.assertEqual(expected, actual)

    def test_majority_baseline_uses_training_data(self):
        train, test = (
            self.meta["split"]["train_indices"],
            self.meta["split"]["test_indices"],
        )
        model = DummyClassifier(strategy="most_frequent").fit(
            self.X[train], self.y[train]
        )
        predictions = model.predict(self.X[test])
        expected = json.loads((ROOT / "reports/metrics.json").read_text())["baseline"]
        self.assertEqual(
            accuracy_score(self.y[test], predictions), expected["accuracy"]
        )
        self.assertEqual(
            f1_score(self.y[test], predictions, average="macro"), expected["macro_f1"]
        )

    def test_cv_search_space_fold_scores_and_selection(self):
        search = json.loads((ROOT / "reports/cv_results.json").read_text())
        actual = {(row["C"], row["class_weight"]) for row in search["candidates"]}
        self.assertEqual(
            actual, {(c, w) for c in [0.01, 0.1, 1.0, 10.0] for w in [None, "balanced"]}
        )
        for row in search["candidates"]:
            self.assertEqual(len(row["fold_f1_macro"]), 5)
            self.assertAlmostEqual(
                np.mean(row["fold_f1_macro"]), row["mean_f1_macro"], places=12
            )
            self.assertAlmostEqual(
                np.std(row["fold_f1_macro"]), row["std_f1_macro"], places=12
            )
        self.assertEqual(
            search["selected"],
            max(search["candidates"], key=lambda row: row["mean_f1_macro"]),
        )
        self.assertEqual(search["convergence_warning_count"], 0)

    def test_local_linear_scores_match_pipeline(self):
        agent = ArchetypeAgent()
        z, scores, _ = agent.predict_vector(self.X[0])
        self.assertTrue(
            np.allclose(
                scores,
                self.pipeline.decision_function(self.X[:1])[0],
                atol=1e-12,
                rtol=0,
            )
        )
        scaler = self.pipeline.named_steps["scaler"]
        self.assertTrue(
            np.allclose(z, scaler.transform(self.X[:1])[0], atol=1e-12, rtol=0)
        )


if __name__ == "__main__":
    unittest.main()
