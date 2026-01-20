"""Baseline model step.

This file used to build a TF-IDF + NearestNeighbors index directly.
Now we wrap it into a small reusable matcher class (see src/matcher.py).

Expected inputs from previous steps:
- df_ideal_address (DataFrame) with column 'united_addr'

Outputs (created in the current Python session):
- matcher (AddressMatcher) fitted on 'united_addr'
"""

import os
import sys

import pandas as pd

# Allow running this file from within the /code folder or from notebooks
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from src.matcher import AddressMatcher


# 1) Reference corpus
df_ref = pd.DataFrame({"united_addr": df_ideal_address["united_addr"]})


# 2) Build matcher (top_k>1 allows reranking and Recall@k evaluation)
matcher = AddressMatcher(
    ngram_range=(2, 4),
    analyzer="char_wb",
    top_k=10,
    w_cosine=0.6,
    w_fuzz=0.4,
    do_normalize=True,
).fit(df_ref["united_addr"].tolist())


if __name__ == "__main__":
    # Quick sanity-check
    bad_address = "ул труд, челяба"
    res = matcher.match_one(bad_address)
    print(f"Вход: {bad_address}")
    print(f"Найдено: {res.best}")
    print(f"cosine={res.cosine_sim:.3f} fuzz={res.fuzz_score:.3f} final={res.final_score:.3f}")
