"""Measure the latency of one recommendation (neural + rule-based state + fusion + symbolic filtering).
Uses a randomly initialised network: latency does not depend on the trained weights.

    python benchmark_latency.py --skills skills.csv --graph prereq_edges.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from nsa_eal.config import Cfg
from nsa_eal.model import TransformerKT
from nsa_eal.symbolic import (KnowledgeGraph, recommend, rule_based_mastery, symbolic_score, unmet_prereqs, zpd)


@torch.no_grad()
def one_pass(model, kg, cfg, sk, co, m):
    M = torch.sigmoid(model(sk, co, m))
    R = rule_based_mastery(sk, co, m, kg.n)
    alpha = min(1.0, sk.shape[1] / cfg.n0)
    st = (alpha * M + (1 - alpha) * R)[:, -1:, :]                 # state after the last interaction
    unmet = unmet_prereqs(st, kg, cfg.tau)
    score = alpha * zpd(st) + (1 - alpha) * symbolic_score(unmet, kg)
    return recommend(score, st < cfg.tau, unmet, True)[0]


def timeit(fn, reps):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return np.array(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--history", type=int, default=100)
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()

    cfg = Cfg()
    ids = sorted(pd.read_csv(a.skills)["skill_id"].tolist())
    kg = KnowledgeGraph.from_csv(a.graph, {s: i for i, s in enumerate(ids)})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kg.to(device)
    model = TransformerKT(kg.n, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout).to(device).eval()
    print(f"device = {device}, {kg.n} concepts, {len(kg.c_idx)} edges, history length = {a.history}")

    for B in (1, 64, 512):
        sk = torch.randint(0, kg.n, (B, a.history), device=device)
        co = torch.randint(0, 2, (B, a.history), device=device)
        m = torch.ones(B, a.history, dtype=torch.bool, device=device)
        ts = timeit(lambda: one_pass(model, kg, cfg, sk, co, m), a.reps if B < 512 else 20)
        print(f"batch {B:4d}: median {np.median(ts):8.2f} ms, p95 {np.percentile(ts, 95):8.2f} ms "
              f"-> {B / (np.median(ts) / 1000):9.0f} recommendations/s")


if __name__ == "__main__":
    main()
