# NSA-EAL — reference implementation (PyTorch)

Code accompanying "Neuro-Symbolic Architecture for Explainable Adaptive Learning".

This repository contains the code used for the experiments reported in the paper (Models A, B and C, the inferred
prerequisite graph, the metrics, and the scripts to score a human evaluation of explanations and to measure latency).
Parts of the code were developed with the assistance of Claude (Anthropic), as declared in the paper.

**Data.** The ASSISTments 2012-2013 data are not redistributed here. Download them from the official page
(https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) and cite Feng et al. (2009).

**Numbers.** The results of the paper come from `python run_experiments.py ... --seeds 10 --epochs 30` with the graph
produced by `infer_graph.py` with its default parameters. `results_final.csv`, the inferred graph and its
`_params.json` are provided with the paper's artefacts. Small differences are possible across hardware and library versions.

## Install and test
```bash
pip install -r requirements.txt
python run_experiments.py --synthetic --seeds 2 --epochs 3     # smoke test on synthetic data (meaningless numbers)
```

## Workflow used for the paper
```bash
python graph_tools.py slim --data 2012-2013-data-with-predictions-4-final.csv     # slim UTF-8 copy (low RAM)
python graph_tools.py list --data assist2012_slim.csv                              # skills.csv
python infer_graph.py --data assist2012_slim.csv --out prereq_edges_inferred.csv  # graph inferred from TRAINING learners only
python run_experiments.py --data assist2012_slim.csv --graph prereq_edges_inferred.csv --seeds 10 --epochs 30
```
Other tools: `graph_tools.py template/merge/validate` (expert graph workflow), `score_explanations.py`
(scoring of a human evaluation of explanations), `benchmark_latency.py`. Debug options of `run_experiments.py`:
`--max-learners N`, `--batch-size B`.

## Real experiment (ASSISTments 2012-2013)
```bash
python run_experiments.py --data 2012-2013-data-with-predictions-4-final.csv \
                          --graph prereq_edges.csv --seeds 10
```
* Check the column names in `nsa_eal/data.py::load_assistments` (`user_id, skill_id, correct, start_time`).
* `prereq_edges.csv` = the expert-built prerequisite graph (columns `prereq,concept`, original `skill_id` values;
  see `prereq_edges_example.csv`). ASSISTments does not provide prerequisite relations.
* Output: `results.csv` (per seed) and a printed mean/std table + paired t-test of AUC (C vs A).

## Map code <-> paper
| Paper | Code |
|---|---|
| Neural KT module (3.2.1) | `model.py::TransformerKT` (outputs mastery for all concepts after each interaction) |
| Knowledge graph, rules R1-R3 (3.2.2) | `symbolic.py` (`KnowledgeGraph`, `unmet_prereqs`, `recommend`, `explain`) |
| Constraint-based learning, Eq. (1) | `train.py` + `symbolic.py::symbolic_loss` |
| A posteriori symbolic filtering | `symbolic.py::recommend(hard_filter=True)` |
| Embedding alignment | `model.py::Aligner` (contrastive + graph smoothness) |
| Score(c), Eq. (2), alpha schedule | `evaluate.py` (mode `C`) |
| Models A / B / C, AUC, cold-start AUC, PVR | `evaluate.py`, `run_experiments.py` |

## Design choices NOT specified in the paper (my assumptions: review, report, or change them)
* **Hyperparameters** (`config.py`): lambda = 0.1, gamma = 0.05, tau = 0.8, d_model = 128, 2 layers, 30 epochs, etc.
* **L_symbolic**: hinge penalty `max(0, mastery(c) - mastery(p))` for each prerequisite edge `p -> c`.
* **alpha**: fixed schedule `alpha = min(1, n / n0)` (n = interactions seen), not learned; beta = 1 - alpha.
  The mastery state of Model C is `alpha * neural + (1 - alpha) * rule_based`.
* **Rule-based mastery** (Model B and the symbolic side of C): Laplace-smoothed running accuracy per concept.
* **Recommendation policy**: among non-mastered concepts (mastery < tau), pick the one closest to 0.5 mastery
  (zone of proximal development); Model C adds the symbolic score and rule R3. If every candidate is vetoed, it falls
  back to the best unfiltered one (so PVR is not zero by construction for B and C).
* **PVR** uses each model's own mastery estimates (as in the paper's definition).
* Each learner is truncated to its first 200 interactions; learners are split 70/10/20 by user.
* Not included: SHAP baseline, expert evaluation of explanations (the 94.5% / 71.2% figures), latency benchmark,
  zero-shot generalisation to new concepts, learned alpha/beta.
