"""Score the expert evaluation of explanations.

Input CSV (one row per explanation judged), columns:
    item_id, method, rater1, rater2
 - method : e.g. "rule_based" (NSA-EAL) and "shap" (baseline)
 - rater1, rater2 : 1 = the explanation is accurate (the stated cause really determined the recommendation), 0 = not
Usage:
    python score_explanations.py --file annotations.csv
Reports, per method: n, % accurate (accurate = BOTH raters say 1), 95% Wilson CI, Cohen's kappa between raters,
and a Fisher exact test between the two methods.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.file)
    df["accurate"] = ((df["rater1"] == 1) & (df["rater2"] == 1)).astype(int)
    stats = {}
    for m, g in df.groupby("method"):
        k, n = int(g["accurate"].sum()), len(g)
        lo, hi = wilson(k, n)
        kappa = cohen_kappa_score(g["rater1"], g["rater2"])
        stats[m] = (k, n)
        print(f"{m:12s} n={n:4d}  accurate={100 * k / n:5.1f}%  95% CI [{100 * lo:.1f}, {100 * hi:.1f}]  "
              f"kappa={kappa:.2f}  (raters disagree on {(g['rater1'] != g['rater2']).sum()} items)")
    if len(stats) == 2:
        (m1, (k1, n1)), (m2, (k2, n2)) = stats.items()
        _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
        print(f"\nFisher exact test {m1} vs {m2}: p = {p:.4g}")


if __name__ == "__main__":
    main()
