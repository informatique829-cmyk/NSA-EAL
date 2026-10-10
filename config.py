from dataclasses import dataclass


@dataclass
class Cfg:
    # --- data ---
    max_len: int = 200        # each learner is truncated to its first max_len interactions
    batch_size: int = 64
    # --- model ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    # --- training ---
    lr: float = 1e-3
    epochs: int = 30
    patience: int = 5
    lam: float = 0.1          # weight of the symbolic loss  (lambda in Eq. 1)
    gamma: float = 0.05       # weight of the embedding-alignment loss
    # --- symbolic / fusion ---
    tau: float = 0.8          # mastery threshold (P >= tau  <=>  mastered)
    n0: int = 20              # interactions after which alpha reaches 1 (alpha = min(1, n / n0))
    cold_n: int = 5           # cold-start = first cold_n interactions of a learner
