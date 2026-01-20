"""Evaluation step (baseline/prototype).

Expected inputs from previous steps (same Python session):
- df_ideal_address: DataFrame with columns:
    - united_addr (ground truth reference)
    - keyboard_noise (noisy query)  [created in 2_make_noise.py]
  (Optionally: random_insert)
- matcher: fitted AddressMatcher (created in 3_model.py)

Outputs:
- df_eval: dataframe with predictions + scores
- prints: Accuracy@1 + a simple threshold table

Note: because we synthesize noise from united_addr, we know the ground truth.
"""

import os
import sys

import numpy as np
import pandas as pd

# Allow running this file from within the /code folder or from notebooks
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from src.address_normalize import normalize_ru_address


def _threshold_report(df: pd.DataFrame, thresholds=(0.70, 0.75, 0.80, 0.85, 0.90)) -> pd.DataFrame:
    """Precision/Recall/Coverage for an 'auto-match' threshold.

    - auto-match if final_score >= T
    - precision: among auto-matches, how many are correct
    - recall: among all queries, how many correct auto-matches we got
    - coverage: share of queries auto-matched
    """
    out = []
    for t in thresholds:
        auto = df[df["final_score"] >= t]
        coverage = len(auto) / len(df) if len(df) else 0.0
        if len(auto) == 0:
            precision = 0.0
        else:
            precision = float((auto["is_correct"]).mean())
        recall = float(((df["final_score"] >= t) & (df["is_correct"]).astype(bool)).mean())
        out.append({"threshold": t, "precision": precision, "recall": recall, "coverage": coverage})
    return pd.DataFrame(out)


# 1) Build evaluation dataset
df_eval = pd.DataFrame(
    {
        "query": df_ideal_address["keyboard_noise"],
        "true": df_ideal_address["united_addr"],
    }
)

# 2) Predict
pred = matcher.match_batch(df_eval["query"].tolist())
df_eval = df_eval.join(pred)

# 3) Compare (with normalization to be robust to formatting)
df_eval["true_norm"] = df_eval["true"].map(normalize_ru_address)
df_eval["best_norm"] = df_eval["best"].map(normalize_ru_address)
df_eval["is_correct"] = df_eval["true_norm"] == df_eval["best_norm"]

acc1 = float(df_eval["is_correct"].mean()) if len(df_eval) else 0.0

print(f"Accuracy@1 (keyboard_noise): {acc1:.3f}  (n={len(df_eval)})")

# 4) Show a simple threshold trade-off table
report = _threshold_report(df_eval)
print("\nThreshold report (auto-match by final_score):")
print(report.to_string(index=False, formatters={
    "threshold": "{:.2f}".format,
    "precision": "{:.3f}".format,
    "recall": "{:.3f}".format,
    "coverage": "{:.3f}".format,
}))

# 5) Save results for demo
out_path = os.path.join(_ROOT, "results_keyboard_noise.csv")
df_eval.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
