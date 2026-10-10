"""Symbolic layer: prerequisite graph, logical rules R1-R3, symbolic scoring, filtering, explanations."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class KnowledgeGraph:
    """Prerequisite graph G = (C, E). An edge (p, c) means 'p is a prerequisite of c'."""

    def __init__(self, n_skills, edges):
        e = torch.tensor(sorted(set(map(tuple, edges))), dtype=torch.long).reshape(-1, 2)
        self.n = n_skills
        self.p_idx = e[:, 0].contiguous()
        self.c_idx = e[:, 1].contiguous()
        self.n_prereq = torch.zeros(n_skills).index_add_(0, self.c_idx, torch.ones(len(self.c_idx)))
        if not self._is_dag():
            print("WARNING: the prerequisite graph contains a cycle.")

    def _is_dag(self):
        indeg = self.n_prereq.clone().tolist()
        out = {}
        for p, c in zip(self.p_idx.tolist(), self.c_idx.tolist()):
            out.setdefault(p, []).append(c)
        stack = [i for i in range(self.n) if indeg[i] == 0]
        seen = 0
        while stack:
            u = stack.pop()
            seen += 1
            for v in out.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
        return seen == self.n

    def to(self, device):
        self.p_idx, self.c_idx, self.n_prereq = (t.to(device) for t in (self.p_idx, self.c_idx, self.n_prereq))
        return self

    @staticmethod
    def from_csv(path, mapping, prereq_col="prereq", concept_col="concept"):
        """CSV with one expert-defined edge per row, using the ORIGINAL skill ids of the dataset."""
        df = pd.read_csv(path)
        edges = []
        for p, c in zip(df[prereq_col], df[concept_col]):
            if p in mapping and c in mapping and p != c:
                edges.append((mapping[p], mapping[c]))
        return KnowledgeGraph(len(mapping), edges)

    @staticmethod
    def random_dag(n, seed=0, avg_prereqs=1.5):
        rng = np.random.RandomState(seed)
        edges = [(i, j) for j in range(1, n) for i in range(j) if rng.rand() < avg_prereqs / max(j, 1)]
        return KnowledgeGraph(n, edges)


# ---------------------------------------------------------------- state estimation
def rule_based_mastery(skills, correct, mask, n_skills):
    """Data-free symbolic estimate: Laplace-smoothed running accuracy per concept. -> [B, L, K]"""
    oh = F.one_hot(skills, n_skills).float() * mask.unsqueeze(-1).float()
    cnt = oh.cumsum(1)
    cor = (oh * correct.unsqueeze(-1).float()).cumsum(1)
    return (cor + 1.0) / (cnt + 2.0)


def unmet_prereqs(state, kg, tau):
    """Number of non-mastered prerequisites for every concept. -> [B, L, K]"""
    unm = (state[..., kg.p_idx] < tau).float()
    out = torch.zeros_like(state)
    out.index_add_(2, kg.c_idx, unm)
    return out


def symbolic_score(unmet, kg):
    """R_symbolic(c): fraction of prerequisites already mastered (1 for concepts without prerequisites)."""
    return 1.0 - unmet / kg.n_prereq.clamp(min=1.0)


def symbolic_loss(M, kg, mask):
    """L_symbolic: penalise mastery[c] > mastery[p] for every prerequisite edge (p -> c). M: [B, T, K]"""
    if len(kg.c_idx) == 0:
        return M.new_zeros(())
    viol = F.relu(M[..., kg.c_idx] - M[..., kg.p_idx])          # [B, T, E]
    w = mask.unsqueeze(-1).float()
    return (viol * w).sum() / (w.sum() * viol.shape[-1]).clamp(min=1.0)


# ---------------------------------------------------------------- recommendation + rules
def zpd(state):
    """Neural/statistical desirability: highest when mastery is near 0.5 (zone of proximal development)."""
    return 1.0 - 2.0 * (state - 0.5).abs()


def recommend(score, unmastered, unmet, hard_filter):
    """Pick the best unmastered concept.
    hard_filter=True applies rule R3 (veto concepts with non-mastered prerequisites); if every candidate
    is vetoed, fall back to the best unfiltered candidate so that the system always recommends something.
    Returns (concept_index [B, T], has_candidate [B, T])."""
    cand = unmastered
    ok = cand & (unmet == 0) if hard_filter else cand
    use = torch.where(ok.any(-1, keepdim=True), ok, cand)
    rec = score.masked_fill(~use, -1e9).argmax(-1)
    return rec, cand.any(-1)


# ---------------------------------------------------------------- explanations
def explain(concept, state_row, kg, tau=0.8, names=None, neural_choice=None):
    """Rule-traceable explanation (evidence -> rule -> inference). state_row: tensor [K]."""
    nm = (lambda i: names[i]) if names else (lambda i: f"concept {i}")
    pre = kg.p_idx[kg.c_idx == concept].tolist()
    mastered = [p for p in pre if state_row[p] >= tau]
    evidence = "; ".join(f"{nm(p)} mastery = {state_row[p]:.2f}" for p in pre) or "no prerequisites"
    if neural_choice is not None and neural_choice != concept:
        rule = (f"R3: the neural choice {nm(neural_choice)} was vetoed because a prerequisite is not mastered; "
                f"redirected to {nm(concept)}.")
    elif pre and len(mastered) == len(pre):
        rule = f"R1: all prerequisites of {nm(concept)} are mastered, so it is unlocked."
    else:
        rule = f"{nm(concept)} has no prerequisite constraint."
    inference = f"Current estimated mastery of {nm(concept)} is {state_row[concept]:.2f}; recommending it."
    return {"evidence": evidence, "rule": rule, "inference": inference}
