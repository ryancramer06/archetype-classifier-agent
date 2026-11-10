"""Build reproducible JSON demonstrations from training rows and invalid copies.

This does not regenerate or relabel the instructor's CSV. The acceptance rule
and fitted model are already fixed before these presentation examples are chosen.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import ArchetypeAgent
from config import FEATURES, SCHEMA_VERSION


def make_requests(vectors, targets, train_ids, agent):
    folder = ROOT / "samples" / "requests"
    folder.mkdir(parents=True, exist_ok=True)
    expected = {}
    origin = {}

    def request(vector, request_id):
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "features": dict(zip(FEATURES, vector)),
        }

    def save(name, value, status, error=None, row_index=None):
        (folder / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        expected[name] = {"status": status, "error_code": error}
        if row_index is not None:
            origin[name] = {"partition": "training", "row_index": row_index}

    predictions = []
    for index in train_ids:
        _, _, probabilities = agent.predict_vector(vectors[index])
        winner = max(range(4), key=lambda k: probabilities[k])
        predictions.append((index, agent.model["classes"][winner], max(probabilities)))
    for label in agent.model["classes"]:
        eligible = [
            row
            for row in predictions
            if row[1] == label
            and targets[row[0]] == label
            and row[2] >= agent.threshold
        ]
        index, _, _ = max(eligible, key=lambda row: row[2])
        save(
            f"valid_{label}.json",
            request(vectors[index], f"demo-{label}"),
            "ok",
            row_index=index,
        )
    uncertain = min(predictions, key=lambda row: row[2])
    if uncertain[2] >= agent.threshold:
        raise RuntimeError(
            "No actual training-row uncertainty example at the fixed threshold; add a clearly labeled controlled test instead."
        )
    save(
        "uncertain.json",
        request(vectors[uncertain[0]], "demo-uncertain"),
        "uncertain",
        row_index=uncertain[0],
    )
    base = request(vectors[train_ids[0]], "demo-validation")
    cases = []
    value = deepcopy(base)
    del value["features"]["palm_width"]
    cases.append(("missing_feature.json", value, "missing_feature"))
    value = deepcopy(base)
    value["features"]["palm_width"] = True
    cases.append(("boolean_feature.json", value, "nonnumeric_feature"))
    value = deepcopy(base)
    value["features"]["palm_width"] = "8.0"
    cases.append(("numeric_string.json", value, "nonnumeric_feature"))
    value = deepcopy(base)
    value["features"]["palm_width"] = None
    cases.append(("null_feature.json", value, "nonnumeric_feature"))
    value = deepcopy(base)
    value["features"]["palm_width"] = 5.9
    cases.append(("below_range.json", value, "out_of_range"))
    value = deepcopy(base)
    value["features"]["eye_distance"] = 7.6
    cases.append(("above_range.json", value, "out_of_range"))
    value = deepcopy(base)
    value["features"]["archetype"] = "explorer"
    cases.append(("extra_feature.json", value, "unexpected_feature"))
    value = deepcopy(base)
    value["extra"] = "ignored? no"
    cases.append(("extra_field.json", value, "unexpected_field"))
    value = deepcopy(base)
    value["schema_version"] = "2.0.0"
    cases.append(("wrong_schema.json", value, "unsupported_schema_version"))
    value = deepcopy(base)
    value["request_id"] = "bad id"
    cases.append(("bad_request_id.json", value, "invalid_request_id"))
    value = deepcopy(base)
    del value["request_id"]
    cases.append(("missing_request_id.json", value, "missing_field"))
    value = deepcopy(base)
    value["features"] = []
    cases.append(("features_not_object.json", value, "invalid_features"))
    cases.append(("request_not_object.json", [], "invalid_request"))
    for name, value, error in cases:
        save(name, value, "invalid_input", error)
    malformed = {
        "malformed.json": "{ this is not JSON }\n",
        "nan.json": json.dumps(base).replace(
            str(base["features"]["palm_width"]), "NaN", 1
        ),
        "infinity.json": json.dumps(base).replace(
            str(base["features"]["palm_width"]), "Infinity", 1
        ),
        "duplicate_key.json": '{"schema_version":"1.0.0","schema_version":"2.0.0","request_id":"duplicate","features":{}}\n',
    }
    for name, text in malformed.items():
        (folder / name).write_text(text, encoding="utf-8")
        expected[name] = {"status": "invalid_input", "error_code": "malformed_json"}
    (ROOT / "samples/expected.json").write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "samples/provenance.json").write_text(
        json.dumps(origin, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Created {len(expected)} requests: four accepted categories, an actual uncertain row, and validation failures."
    )


if __name__ == "__main__":
    from data_utils import load_dataset

    vectors, targets, _ = load_dataset()
    split = json.loads((ROOT / "models/split.json").read_text())
    make_requests(vectors, targets, split["train_indices"], ArchetypeAgent())
