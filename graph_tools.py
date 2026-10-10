"""Utilities for the ASSISTments file and the prerequisite graph.

STEP 0 (once): the raw CSV is huge (the 'actions' column is long and multi-line), so create a slim copy.
       It is read row by row, so it needs almost no RAM:
       python graph_tools.py slim --data 2012-2013-data-with-predictions-4-final.csv
   -> writes assist2012_slim.csv (user_id, skill_id, skill, correct, start_time). Use this file everywhere after.

1) List the skills so that domain experts can annotate prerequisites:
       python graph_tools.py list --data assist2012_slim.csv
   -> writes skills.csv (skill_id, skill name, number of interactions)

2) Create a RANDOM graph, ONLY to test that the pipeline runs on the real data:
       python graph_tools.py test --data assist2012_slim.csv
   -> writes prereq_edges.csv.
   !! Results obtained with this random graph must NEVER be reported in the paper !!

EXPERT GRAPH WORKFLOW (the graph used in the paper must come from this, not from `test`):
   a) python graph_tools.py list     --data assist2012_slim.csv                       -> skills.csv
   b) python graph_tools.py template --skills skills.csv                              -> expert_template.csv
      Give a COPY of expert_template.csv to each expert. They fill the column `prerequisites`
      with the skill_id values of the DIRECT prerequisites, separated by ";"  (e.g.  12;45).
   c) python graph_tools.py merge --skills skills.csv --experts expert1.csv expert2.csv [expert3.csv]
      -> prereq_edges.csv (edges kept by majority), disagreements.csv (to discuss), agreement statistics
   d) python graph_tools.py validate --skills skills.csv --graph prereq_edges.csv     -> checks (ids, cycles) + statistics

The graph file must be a CSV with the columns:  prereq,concept   (original skill_id values)
"""
import argparse
import csv
import re
from collections import Counter

import numpy as np
import pandas as pd

COLS = ["user_id", "skill_id", "skill", "correct", "start_time"]


def make_slim(src, dst="assist2012_slim.csv"):
    csv.field_size_limit(2 ** 31 - 1)
    n_in = n_out = 0
    with open(src, newline="", encoding="ISO-8859-15") as f, open(dst, "w", newline="", encoding="utf-8") as g:
        r = csv.reader(f)
        header = next(r)
        idx = {c: header.index(c) for c in COLS}
        w = csv.writer(g)
        w.writerow(COLS)
        for row in r:
            n_in += 1
            if n_in % 500000 == 0:
                print(f"  {n_in:,} rows read, {n_out:,} kept")
            try:
                sid, cor = row[idx["skill_id"]], row[idx["correct"]]
                if sid == "" or cor == "":
                    continue
                cor = int(float(cor))
                if cor not in (0, 1):
                    continue
                w.writerow([row[idx["user_id"]], int(float(sid)), row[idx["skill"]], cor, row[idx["start_time"]]])
                n_out += 1
            except (IndexError, ValueError):
                continue
    print(f"done: {n_in:,} rows read, {n_out:,} kept (rows with a skill_id and a 0/1 answer) -> {dst}")


def load_skills(path):
    df = pd.read_csv(path, usecols=["skill_id", "skill"], low_memory=False)
    df["skill_id"] = df["skill_id"].astype(int)
    return (df.groupby("skill_id")
              .agg(skill=("skill", lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
                   n_interactions=("skill", "size"))
              .reset_index()
              .sort_values("skill_id"))


def _parse_expert(path, valid):
    df = pd.read_csv(path)
    edges, bad = set(), []
    for sid, pre in zip(df["skill_id"], df["prerequisites"].fillna("")):
        sid = int(sid)
        for tok in re.split(r"[;,\s]+", str(pre).strip()):
            if not tok:
                continue
            try:
                p = int(float(tok))
            except ValueError:
                bad.append((sid, tok))
                continue
            if p not in valid or p == sid:
                bad.append((sid, tok))
                continue
            edges.add((p, sid))
    return edges, bad


def _has_cycle(n_nodes, edges):
    nodes = sorted({x for e in edges for x in e})
    indeg = {u: 0 for u in nodes}
    out = {u: [] for u in nodes}
    for p, c in edges:
        out[p].append(c)
        indeg[c] += 1
    stack = [u for u in nodes if indeg[u] == 0]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return len(order) != len(nodes), order, out


def make_template(skills_csv, out):
    sk = pd.read_csv(skills_csv)
    sk["prerequisites"] = ""
    sk.to_csv(out, index=False)
    print(f"template for experts written to {out} ({len(sk)} skills). Fill the column 'prerequisites' "
          f"with the skill_id of the DIRECT prerequisites, separated by ';'.")


def merge_experts(skills_csv, expert_files, out, min_votes=None):
    valid = set(pd.read_csv(skills_csv)["skill_id"].astype(int))
    K = len(valid)
    sets = []
    for f in expert_files:
        e, bad = _parse_expert(f, valid)
        print(f"{f}: {len(e)} edges" + (f", {len(bad)} invalid entries ignored: {bad[:5]}" if bad else ""))
        sets.append(e)
    k = len(sets)
    min_votes = min_votes or (k // 2 + 1)
    votes = Counter(e for s in sets for e in s)
    kept = sorted(e for e, v in votes.items() if v >= min_votes)
    disputed = sorted((e, v) for e, v in votes.items() if v < k)
    pd.DataFrame(kept, columns=["prereq", "concept"]).to_csv(out, index=False)
    pd.DataFrame([(p, c, v, k) for (p, c), v in disputed], columns=["prereq", "concept", "votes", "n_experts"]) \
        .to_csv("disagreements.csv", index=False)
    print(f"\n{len(kept)} edges kept (>= {min_votes} of {k} experts) -> {out}")
    print(f"{len(disputed)} edges not proposed by every expert -> disagreements.csv (to be discussed)")
    n_pairs = K * (K - 1)
    for i in range(k):
        for j in range(i + 1, k):
            A, B = sets[i], sets[j]
            inter, union = len(A & B), len(A | B)
            po = 1 - len(A ^ B) / n_pairs
            pa, pb = len(A) / n_pairs, len(B) / n_pairs
            pe = pa * pb + (1 - pa) * (1 - pb)
            kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
            print(f"experts {i + 1} & {j + 1}: Jaccard = {inter / union if union else float('nan'):.2f}, "
                  f"Cohen's kappa = {kappa:.2f} (over all {n_pairs} ordered pairs of concepts)")
    cyc, _, _ = _has_cycle(K, kept)
    if cyc:
        print("WARNING: the merged graph contains a cycle. Run `validate`, fix and merge again.")


def validate_graph(skills_csv, graph_csv):
    valid = set(pd.read_csv(skills_csv)["skill_id"].astype(int))
    g = pd.read_csv(graph_csv)
    edges = [(int(p), int(c)) for p, c in zip(g["prereq"], g["concept"])]
    unknown = [e for e in edges if e[0] not in valid or e[1] not in valid]
    selfl = [e for e in edges if e[0] == e[1]]
    dup = len(edges) - len(set(edges))
    cyc, order, out = _has_cycle(len(valid), set(edges))
    print(f"{len(edges)} edges; {len(unknown)} with unknown skill_id; {len(selfl)} self-loops; {dup} duplicates")
    print(f"concepts with >= 1 prerequisite: {g['concept'].nunique()} / {len(valid)}; "
          f"concepts that are a prerequisite of another: {g['prereq'].nunique()}")
    if cyc:
        print("ERROR: the graph contains a cycle (a concept cannot be, directly or not, its own prerequisite).")
        return
    depth = {u: 0 for u in order}
    for u in order:
        for v in out[u]:
            depth[v] = max(depth[v], depth[u] + 1)
    print(f"graph is acyclic; longest prerequisite chain = {max(depth.values()) if depth else 0} edges")
    print("OK" if not (unknown or selfl or dup) else "Fix the problems above, then validate again.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["slim", "list", "test", "template", "merge", "validate"])
    ap.add_argument("--data")
    ap.add_argument("--skills")
    ap.add_argument("--graph")
    ap.add_argument("--experts", nargs="+")
    ap.add_argument("--min-votes", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.cmd == "template":
        make_template(a.skills, a.out or "expert_template.csv")
        return
    if a.cmd == "merge":
        merge_experts(a.skills, a.experts, a.out or "prereq_edges.csv", a.min_votes)
        return
    if a.cmd == "validate":
        validate_graph(a.skills, a.graph)
        return
    if a.cmd == "slim":
        make_slim(a.data, a.out or "assist2012_slim.csv")
        return
    skills = load_skills(a.data)
    if a.cmd == "list":
        out = a.out or "skills.csv"
        skills.to_csv(out, index=False)
        print(f"{len(skills)} skills written to {out}")
    else:
        out = a.out or "prereq_edges.csv"
        rng = np.random.RandomState(a.seed)
        ids = skills["skill_id"].tolist()
        order = rng.permutation(len(ids))                       # random topological order -> always a DAG
        edges = []
        for j in range(1, len(ids)):
            for i in range(j):
                if rng.rand() < 1.5 / j:
                    edges.append((ids[order[i]], ids[order[j]]))
        pd.DataFrame(edges, columns=["prereq", "concept"]).to_csv(out, index=False)
        print(f"RANDOM test graph with {len(edges)} edges written to {out} (do not use for the paper)")


if __name__ == "__main__":
    main()
