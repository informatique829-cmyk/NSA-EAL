import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .symbolic import (rule_based_mastery, unmet_prereqs, symbolic_score, zpd, recommend)


def _auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


@torch.no_grad()
def evaluate(model, loader, kg, cfg, mode, device, recs=True):
    """mode 'A': neural only | 'B': rule-based only | 'C': NSA-EAL (fusion + symbolic filtering).
    Returns AUC, cold-start AUC and PVR (%)."""
    if model is not None:
        model.eval()
    ys, ps, ns = [], [], []
    viol = nrec = 0
    for sk, co, m in loader:
        sk, co, m = sk.to(device), co.to(device), m.to(device)
        B, L = sk.shape
        if L < 2:
            continue
        n = torch.arange(1, L + 1, device=device).float().view(1, L, 1)   # interactions seen so far
        alpha = (n / cfg.n0).clamp(max=1.0)                                 # neural weight, 0 -> 1
        if mode in ("A", "C"):
            M = torch.sigmoid(model(sk, co, m))
        if mode in ("B", "C"):
            R = rule_based_mastery(sk, co, m, kg.n)
        state = M if mode == "A" else R if mode == "B" else alpha * M + (1 - alpha) * R

        st, a = state[:, :-1], alpha[:, :-1]
        valid = m[:, 1:]
        p = st.gather(-1, sk[:, 1:].unsqueeze(-1)).squeeze(-1)
        n_seen = n[0, :-1, 0].expand(B, L - 1)
        ys.append(co[:, 1:][valid].cpu().numpy())
        ps.append(p[valid].cpu().numpy())
        ns.append(n_seen[valid].cpu().numpy())

        if recs:
            unmet = unmet_prereqs(st, kg, cfg.tau)
            unmastered = st < cfg.tau
            if mode == "C":
                score = a * zpd(st) + (1 - a) * symbolic_score(unmet, kg)   # Score(c), Eq. 2
                hard = True                                                  # a posteriori filtering (R3)
            else:
                score, hard = zpd(st), mode == "B"
            rec, has = recommend(score, unmastered, unmet, hard)
            v = unmet.gather(-1, rec.unsqueeze(-1)).squeeze(-1) > 0
            sel = valid & has
            viol += int(v[sel].sum())
            nrec += int(sel.sum())

    y, p, ns = np.concatenate(ys), np.concatenate(ps), np.concatenate(ns)
    cold = ns <= cfg.cold_n
    return {"auc": _auc(y, p), "cold_auc": _auc(y[cold], p[cold]),
            "pvr": 100.0 * viol / nrec if nrec else float("nan")}
