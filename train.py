import copy

import torch

from .evaluate import evaluate
from .model import Aligner, TransformerKT, kt_loss
from .symbolic import symbolic_loss


def train_model(train_loader, val_loader, kg, cfg, device, seed, lam, gamma):
    """lam = gamma = 0 gives the purely neural DKT-Transformer (Model A);
    lam, gamma > 0 gives the NSA-EAL network (constraint-based learning + embedding alignment)."""
    torch.manual_seed(seed)
    model = TransformerKT(kg.n, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout).to(device)
    aligner = Aligner(kg.n, cfg.d_model).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(aligner.parameters()), lr=cfg.lr)
    best, best_state, bad = -1.0, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        for sk, co, m in train_loader:
            sk, co, m = sk.to(device), co.to(device), m.to(device)
            logits = model(sk, co, m)
            loss = kt_loss(logits, sk, co, m)
            if lam > 0:
                loss = loss + lam * symbolic_loss(torch.sigmoid(logits[:, :-1]), kg, m[:, 1:])
            if gamma > 0:
                loss = loss + gamma * aligner.loss(model.skill_emb.weight, kg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_auc = evaluate(model, val_loader, kg, cfg, "A", device, recs=False)["auc"]
        print(f"  seed {seed} epoch {epoch + 1:02d}  val AUC {val_auc:.4f}")
        if val_auc > best:
            best, best_state, bad = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model
