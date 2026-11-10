"""Shared names and rules. These stay the same during training and prediction."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES = [
    "palm_width",
    "palm_length",
    "finger_ratio",
    "face_width",
    "face_height",
    "eye_distance",
]
RANGES = {
    "palm_width": (6.0, 11.0),
    "palm_length": (14.0, 23.0),
    "finger_ratio": (0.75, 1.15),
    "face_width": (11.0, 18.0),
    "face_height": (15.0, 25.0),
    "eye_distance": (4.5, 7.5),
}
EXPECTED_COUNTS = {"explorer": 492, "guardian": 248, "creator": 150, "leader": 110}
SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "archetype-lr-1.1.0"
SEED = 317
# Declared before any test evaluation. This is a simple abstention rule, not calibration.
ACCEPTANCE_THRESHOLD = 0.60
CONFIDENCE_TYPE = "uncalibrated_model_probability"
DATASET_SHA256 = "eba5c260ea44c4aa2fc80fb2b06cfee51f9a8628bf4f5c7aa24a3d195ed65abd"
CARD_SHA256 = "7182795b9128d8f677bf4a93b4ba57e494183115e21f175b9d2afc8425f30fe4"
LIMITATIONS = [
    "Fictional category; not a personal assessment.",
    "Synthetic data do not validate relationships between measurements and personal traits.",
    "Probabilities are uncalibrated; contributions explain linear scores, not causal effects.",
]
