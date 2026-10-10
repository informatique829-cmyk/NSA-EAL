# Results of the paper

Put here the files produced by the final run (see the main README):

- `prereq_edges_inferred.csv`        : the prerequisite graph inferred from the training learners (105 edges)
- `prereq_edges_inferred_params.json`: thresholds used by `infer_graph.py`
- `results_final.csv`                : per-seed results of Models A, B and C (10 seeds)
- `log_final.txt`                    : console log of the final run

They were produced with:

```bash
python infer_graph.py --data assist2012_slim.csv --out prereq_edges_inferred.csv
python run_experiments.py --data assist2012_slim.csv --graph prereq_edges_inferred.csv --seeds 10 --epochs 30 --out results_final.csv
```
