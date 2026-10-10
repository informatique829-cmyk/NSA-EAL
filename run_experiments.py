"""Run Models A (DKT-Transformer), B (rule-based) and C (NSA-EAL) and write results.csv.

Real data:   python run_experiments.py --data 2012-2013-data-with-predictions-4-final.csv --graph prereq_edges.csv
Smoke test:  python run_experiments.py --synthetic --seeds 2 --epochs 3
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

from nsa_eal.config import Cfg
from nsa_eal.data import load_assistments, make_loader, make_synthetic, split_users
from nsa_eal.evaluate import evaluate
from nsa_eal.symbolic import KnowledgeGraph
from nsa_eal.train import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--graph", help="CSV with columns prereq,concept (original skill ids), built by domain experts")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--max-learners", type=int, default=0, help="debug: keep only the first N learners (0 = all)")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    cfg = Cfg(epochs=a.epochs, lam=a.lam, gamma=a.gamma, batch_size=a.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if a.synthetic:
        seqs, mapping, kg = make_synthetic()
    else:
        seqs, mapping = load_assistments(a.data, max_len=cfg.max_len)
        kg = KnowledgeGraph.from_csv(a.graph, mapping)
    if a.max_learners:
        seqs = seqs[:a.max_learners]
    kg.to(device)
    tr, va, te = split_users(seqs, seed=0)
    print(f"{len(seqs)} learners, {kg.n} concepts, {len(kg.c_idx)} prerequisite edges")
    ltr, lva = make_loader(tr, cfg.batch_size, True), make_loader(va, cfg.batch_size, False)
    lte = make_loader(te, cfg.batch_size, False)

    rows = [dict(model="B: rule-based", seed=-1, **evaluate(None, lte, kg, cfg, "B", device))]
    for seed in range(a.seeds):
        netA = train_model(ltr, lva, kg, cfg, device, seed, lam=0.0, gamma=0.0)
        rows.append(dict(model="A: DKT-Transformer", seed=seed, **evaluate(netA, lte, kg, cfg, "A", device)))
        netC = train_model(ltr, lva, kg, cfg, device, seed, lam=cfg.lam, gamma=cfg.gamma)
        rows.append(dict(model="C: NSA-EAL", seed=seed, **evaluate(netC, lte, kg, cfg, "C", device)))
        pd.DataFrame(rows).to_csv(a.out, index=False)      # save after every seed: nothing is lost if the PC stops
        print(f"== seed {seed} done, results saved to {a.out}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("\n", df.groupby("model")[["auc", "cold_auc", "pvr"]].agg(["mean", "std"]).round(4).to_string())
    A = df[df.model.str.startswith("A")].sort_values("seed")["auc"].to_numpy()
    C = df[df.model.str.startswith("C")].sort_values("seed")["auc"].to_numpy()
    if len(A) > 1:
        t, p = ttest_rel(C, A)
        print(f"\npaired t-test AUC (C vs A): t = {t:.3f}, p = {p:.4g}")


if __name__ == "__main__":
    main()
