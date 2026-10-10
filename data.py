import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def load_assistments(csv_path, max_len=200, min_len=5, user_col="user_id", skill_col="skill_id",
                     correct_col="correct", time_col="start_time"):
    """Load the ASSISTments 2012-2013 CSV. Check that the column names match your file.

    Returns (sequences, skill_mapping). Each sequence is (skills[int64], correct[int64]).
    """
    # The raw ASSISTments 2012-13 file is ISO-8859-15 and very heavy; the slim copy made by
    # `python graph_tools.py slim` is UTF-8 (use it: it avoids out-of-memory errors).
    slim = "slim" in str(csv_path).lower()
    enc = "utf-8" if slim else "ISO-8859-15"
    dtypes = {user_col: "int64", skill_col: "int32", correct_col: "int8"} if slim else None
    parts = []   # chunked reading + datetime conversion keeps RAM low (strings -> 8-byte timestamps)
    for ch in pd.read_csv(csv_path, usecols=[user_col, skill_col, correct_col, time_col], dtype=dtypes,
                          low_memory=False, encoding=enc, chunksize=500_000):
        ch[time_col] = pd.to_datetime(ch[time_col], errors="coerce")
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df = df.dropna(subset=[skill_col, correct_col])
    df = df[df[correct_col].isin([0, 1])].copy()
    df[skill_col] = df[skill_col].astype(int)
    df = df.sort_values([user_col, time_col], kind="stable")
    mapping = {s: i for i, s in enumerate(sorted(df[skill_col].unique()))}
    df["_s"] = df[skill_col].map(mapping)
    seqs = []
    for _, g in df.groupby(user_col, sort=False):
        s = g["_s"].to_numpy(np.int64)[:max_len]
        c = g[correct_col].to_numpy(np.int64)[:max_len]
        if len(s) >= min_len:
            seqs.append((s, c))
    return seqs, mapping


def split_users(seqs, seed=0, ratios=(0.7, 0.1, 0.2)):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(seqs))
    n_tr = int(ratios[0] * len(seqs))
    n_va = int(ratios[1] * len(seqs))
    pick = lambda ids: [seqs[i] for i in ids]
    return pick(idx[:n_tr]), pick(idx[n_tr:n_tr + n_va]), pick(idx[n_tr + n_va:])


class _Seqs(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def _collate(batch):
    L = max(len(s) for s, _ in batch)
    B = len(batch)
    sk = torch.zeros(B, L, dtype=torch.long)
    co = torch.zeros(B, L, dtype=torch.long)
    m = torch.zeros(B, L, dtype=torch.bool)
    for i, (s, c) in enumerate(batch):
        n = len(s)
        sk[i, :n] = torch.from_numpy(s)
        co[i, :n] = torch.from_numpy(c)
        m[i, :n] = True
    return sk, co, m


def make_loader(seqs, batch_size, shuffle):
    return DataLoader(_Seqs(seqs), batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)


def make_synthetic(n_users=600, n_skills=40, length=120, seed=0):
    """Tiny synthetic generator, ONLY for smoke-testing the code (not for any result in the paper)."""
    from .symbolic import KnowledgeGraph
    rng = np.random.RandomState(seed)
    kg = KnowledgeGraph.random_dag(n_skills, seed=seed)
    prereqs = {c: [] for c in range(n_skills)}
    for p, c in zip(kg.p_idx.tolist(), kg.c_idx.tolist()):
        prereqs[c].append(p)
    seqs = []
    for _ in range(n_users):
        mast = np.clip(rng.beta(2, 4, n_skills), 0, 1)
        s_list, c_list = [], []
        for _ in range(length):
            s = rng.randint(n_skills)
            cap = min([mast[p] for p in prereqs[s]] + [1.0])
            p_ok = 0.15 + 0.75 * min(mast[s], cap + 0.2)
            c = int(rng.rand() < p_ok)
            mast[s] = min(1.0, mast[s] + 0.08 * (1 - mast[s]) * (1 + c))
            s_list.append(s)
            c_list.append(c)
        seqs.append((np.array(s_list, np.int64), np.array(c_list, np.int64)))
    return seqs, {i: i for i in range(n_skills)}, kg
