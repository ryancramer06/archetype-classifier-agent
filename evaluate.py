"""Recheck the fixed saved model on the reserved test split; never tune here."""

import json
import sys

from agent import ArchetypeAgent
from config import ROOT
from data_utils import load_dataset


def main():
    import numpy as np
    from evaluation_utils import evaluate_predictions
    vectors,targets,data=load_dataset()
    metadata=json.loads((ROOT/"models/metadata.json").read_text())
    if data["csv_sha256"]!=metadata["dataset"]["csv_sha256"]:
        raise ValueError("Dataset differs from the one used for this saved model.")
    ids=metadata["split"]["test_indices"]
    agent=ArchetypeAgent(backend="sklearn")
    X=np.asarray([vectors[i] for i in ids])
    y=np.asarray([targets[i] for i in ids])
    pipeline=agent.pipeline
    metrics=evaluate_predictions(y,pipeline.predict(X),pipeline.predict_proba(X),metadata["class_order"],agent.threshold)
    reference=json.loads((ROOT/"reports/metrics.json").read_text())["selected_model"]
    for name in ["accuracy","macro_f1","macro_ovr_auc","coverage","accepted_accuracy"]:
        if metrics[name] is None:
            assert reference[name] is None
        else:
            assert abs(metrics[name]-reference[name])<1e-12,(name,metrics[name],reference[name])
    assert metrics["confusion_matrix"]==reference["confusion_matrix"]
    target=ROOT/"local_results"/"heldout_recheck.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metrics,indent=2))
    print("Saved-model metrics match the original held-out evaluation. No fitting or parameter selection ran.")


if __name__=="__main__":
    main()
