"""Infer a prerequisite graph from the TRAINING learners only (no information from validation/test learners).

    python infer_graph.py --data assist2012_slim.csv --out prereq_edges_inferred.csv

Procedure (an association-based heuristic, NOT causal evidence and NOT an expert graph). For an ordered pair of
concepts (a, b), the edge a -> b ("a is a prerequisite of b") is proposed when ALL of the following hold:
  1. Support      : at least `n_both` training learners practised both concepts, and the 'a mastered' and
                    'a not mastered' groups each contain at least `n_min` attempts on b.
  2. Precedence   : among learners who practised both, a was met before b in at least `min_prec` of the cases.
  3. Performance  : attempts on b are more successful when a is already mastered (>= `min_attempts` attempts on a
                    with accuracy >= `mastery_thr`) than when it is not. The success is measured as a residual
                    (answer - the learner's running overall accuracy), which partly controls for learner ability.
                    The difference of mean residuals must be >= `min_lift`.
  4. Acyclicity   : concepts are ordered by their mean first appearance in the training sequences, and only edges
                    going forward in this order are kept, so that the graph is a DAG by construction.
  5. Sparsity     : at most `top_k` prerequisites (largest lift) are kept for each concept.
The thresholds are fixed a priori (defaults below). Do not tune them on the test set.
"""
import argparse
import json

import numpy as np
import pandas as pd

from nsa_eal.data import load_assistments, split_users


def infer_graph(train_seqs, K, mastery_thr=0.8, min_attempts=3, n_min=200, n_both=200,
                min_prec=0.7, min_lift=0.02, top_k=3):
    N1, N0 = np.zeros((K, K)), np.zeros((K, K))      # indexed [b, a]: attempts on b with a mastered / not mastered
    R1, R0 = np.zeros((K, K)), np.zeros((K, K))      # sums of residuals
    both, pre = np.zeros((K, K)), np.zeros((K, K))   # indexed [a, b]
    first_sum, first_n = np.zeros(K), np.zeros(K)
    for skills, correct in train_seqs:
        cnt, cor = np.zeros(K), np.zeros(K)
        tot_n = tot_c = 0
        first = np.full(K, np.inf)
        for t, (b, y) in enumerate(zip(skills.tolist(), correct.tolist())):
            if first[b] == np.inf:
                first[b] = t
            m = ((cnt >= min_attempts) & (cor >= mastery_thr * np.maximum(cnt, 1))).astype(float)
            r = y - (tot_c + 1.0) / (tot_n + 2.0)
            N1[b] += m
            N0[b] += 1.0 - m
            R1[b] += m * r
            R0[b] += (1.0 - m) * r
            cnt[b] += 1
            cor[b] += y
            tot_n += 1
            tot_c += y
        seen = np.isfinite(first)
        s2 = seen[:, None] & seen[None, :]
        both += s2
        pre += s2 & (first[:, None] < first[None, :])
        first_sum[seen] += first[seen]
        first_n += seen

    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (R1 / N1 - R0 / N0).T                  # [a, b]
    support = (N1.T >= n_min) & (N0.T >= n_min) & (both >= n_both)
    prec = pre / np.maximum(both, 1)
    cand = support & (prec >= min_prec) & np.nan_to_num(lift, nan=-1.0).__ge__(min_lift)
    np.fill_diagonal(cand, False)
    mean_first = first_sum / np.maximum(first_n, 1) + 1e-9 * np.arange(K)
    pos = np.argsort(np.argsort(mean_first))
    cand &= pos[:, None] < pos[None, :]               # acyclic by construction
    edges = []
    for b in range(K):
        a_idx = np.where(cand[:, b])[0]
        if len(a_idx):
            a_idx = a_idx[np.argsort(-lift[a_idx, b])][:top_k]
            edges += [(int(a), b) for a in a_idx]
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="prereq_edges_inferred.csv")
    ap.add_argument("--mastery-thr", type=float, default=0.8)
    ap.add_argument("--min-attempts", type=int, default=3)
    ap.add_argument("--n-min", type=int, default=200)
    ap.add_argument("--n-both", type=int, default=200)
    ap.add_argument("--min-prec", type=float, default=0.7)
    ap.add_argument("--min-lift", type=float, default=0.02)
    ap.add_argument("--top-k", type=int, default=3)
    a = ap.parse_args()

    seqs, mapping = load_assistments(a.data)
    train, _, _ = split_users(seqs, seed=0)          # same split as run_experiments.py
    print(f"{len(seqs)} learners, {len(train)} used for the inference (training split only), {len(mapping)} concepts")
    params = dict(mastery_thr=a.mastery_thr, min_attempts=a.min_attempts, n_min=a.n_min, n_both=a.n_both,
                  min_prec=a.min_prec, min_lift=a.min_lift, top_k=a.top_k)
    edges = infer_graph(train, len(mapping), **params)
    inv = {v: k for k, v in mapping.items()}
    df = pd.DataFrame([(inv[p], inv[c]) for p, c in edges], columns=["prereq", "concept"])
    df.to_csv(a.out, index=False)
    with open(a.out.replace(".csv", "_params.json"), "w") as f:
        json.dump(dict(params, n_edges=len(df), n_train_learners=len(train)), f, indent=2)
    print(f"{len(df)} edges written to {a.out}; {df['concept'].nunique()} concepts have >= 1 prerequisite")
    print("parameters saved next to it (_params.json): report them in the article")


if __name__ == "__main__":
    main()
