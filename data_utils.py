"""Read the supplied CSV unchanged and check its documented schema."""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from config import CARD_SHA256, DATASET_SHA256, EXPECTED_COUNTS, FEATURES, RANGES, ROOT


def load_dataset(data_dir=None):
    folder = Path(data_dir) if data_dir is not None else ROOT / "data"
    source = folder / "archetype_synthetic.csv"
    card_path = folder / "dataset_card.json"
    if hashlib.sha256(source.read_bytes()).hexdigest() != DATASET_SHA256 or hashlib.sha256(card_path.read_bytes()).hexdigest() != CARD_SHA256:
        raise ValueError("Restore the unchanged instructor CSV/card; file hashes differ from the supplied originals.")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card["feature_order"] != FEATURES or card["target_column"] != "archetype":
        raise ValueError("The dataset card does not match the six-feature contract.")
    for name in FEATURES:
        if (card["feature_schema"][name]["min"], card["feature_schema"][name]["max"]) != RANGES[name]:
            raise ValueError(f"Unexpected feature range: {name}")
    vectors, labels = [], []
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FEATURES + ["archetype"]:
            raise ValueError("CSV columns/order differ from the supplied feature contract.")
        for row in reader:
            vector = [float(row[name]) for name in FEATURES]
            for name, value in zip(FEATURES, vector):
                low, high = RANGES[name]
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"Invalid dataset value for {name}")
            if row["archetype"] not in EXPECTED_COUNTS:
                raise ValueError("Unexpected target label")
            vectors.append(vector)
            labels.append(row["archetype"])
    counts = dict(Counter(labels))
    if len(vectors) != 1000 or counts != EXPECTED_COUNTS or counts != card["actual_class_counts"]:
        raise ValueError("Expected the unchanged 1,000-row instructor dataset and exact class counts.")
    duplicates = len(vectors) - len({tuple(row) for row in vectors})
    if duplicates:
        raise ValueError("Duplicate feature rows need review before the required row-wise split.")
    return vectors, labels, {"records": len(vectors), "class_counts": counts, "duplicate_feature_rows": duplicates,
                             "csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                             "card_sha256": hashlib.sha256(card_path.read_bytes()).hexdigest(), "card": card}
