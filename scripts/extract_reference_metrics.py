"""Extract per-seed reference metrics from preprint v1 prediction files.

Recomputes RMSE/R²/etc. from saved (y_true, y_pred) arrays in the
kinase repo's results/predictions/. These bit-exact values are the
reference for the integration test.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

KINASE_REPO = Path("/Users/joshuaabbott/mlproject")
PRED_DIR = KINASE_REPO / "results" / "predictions"
OUTPUT = Path("/Users/joshuaabbott/target-affinity-ml/tests/integration/kinase_v1_reference.json")

SMOKE_TEST_MODEL = "random_forest"
SMOKE_TEST_SPLIT = "random"


def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    pearson_r, _ = pearsonr(y_true, y_pred)
    return {"rmse": rmse, "r2": r2, "pearson_r": float(pearson_r)}


def extract_reference():
    pred_file = PRED_DIR / f"{SMOKE_TEST_MODEL}_{SMOKE_TEST_SPLIT}.npz"
    assert pred_file.exists(), f"Missing reference predictions: {pred_file}"

    d = np.load(pred_file)
    y_true_keys = ["y_test_true", "y_true"]
    y_pred_keys = ["y_test_pred", "y_test_mean", "y_pred"]

    y_true = next((d[k] for k in y_true_keys if k in d), None)
    y_pred = next((d[k] for k in y_pred_keys if k in d), None)

    assert y_true is not None and y_pred is not None, (
        f"Could not find y_true/y_pred in {pred_file}. Keys: {list(d.keys())}"
    )

    metrics = compute_metrics(y_true, y_pred)
    print(f"Reference metrics from {pred_file.name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    reference = {
        "model": SMOKE_TEST_MODEL,
        "split": SMOKE_TEST_SPLIT,
        "predictions_file": str(pred_file.relative_to(KINASE_REPO)),
        "n_test_samples": int(len(y_true)),
        "metrics": metrics,
        "tolerance": {
            "rmse": 0.001,
            "r2": 0.005,
            "pearson_r": 0.005,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"\nReference saved to: {OUTPUT}")
    return reference


if __name__ == "__main__":
    extract_reference()
