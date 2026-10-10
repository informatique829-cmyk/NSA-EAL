import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerKT(nn.Module):
    """Causal Transformer knowledge-tracing module.
    forward() returns logits [B, L, K]: mastery logits for every concept AFTER each interaction."""

    def __init__(self, n_skills, d_model=128, n_heads=4, n_layers=2, dropout=0.1, max_len=512):
        super().__init__()
        self.skill_emb = nn.Embedding(n_skills, d_model)     # concept embeddings (used for alignment)
        self.resp_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_skills)

    def forward(self, skills, correct, mask):
        B, L = skills.shape
        pos = torch.arange(L, device=skills.device)
        x = self.skill_emb(skills) + self.resp_emb(correct) + self.pos_emb(pos)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=skills.device), 1)
        h = self.enc(x, mask=causal, src_key_padding_mask=~mask)
        return self.head(self.norm(h))


class Aligner(nn.Module):
    """Embedding alignment: contrastive loss between neural concept embeddings and ontology embeddings,
    plus a graph-smoothness term so that related concepts lie close together."""

    def __init__(self, n_skills, d_model, d_ont=64, temp=0.1):
        super().__init__()
        self.onto = nn.Embedding(n_skills, d_ont)
        self.proj = nn.Linear(d_model, d_ont)
        self.temp = temp

    def loss(self, concept_emb, kg):
        z = F.normalize(self.onto.weight, dim=-1)
        e = F.normalize(self.proj(concept_emb), dim=-1)
        labels = torch.arange(z.shape[0], device=z.device)
        contrastive = F.cross_entropy(e @ z.T / self.temp, labels)
        smooth = ((z[kg.p_idx] - z[kg.c_idx]) ** 2).sum(-1).mean() if len(kg.c_idx) else z.new_zeros(())
        return contrastive + smooth


def kt_loss(logits, skills, correct, mask):
    """BCE of the mastery of the NEXT concept: logits[:, t] predicts interaction t+1."""
    tgt_skill, tgt, m = skills[:, 1:], correct[:, 1:].float(), mask[:, 1:].float()
    lg = logits[:, :-1].gather(-1, tgt_skill.unsqueeze(-1)).squeeze(-1)
    return (F.binary_cross_entropy_with_logits(lg, tgt, reduction="none") * m).sum() / m.sum().clamp(min=1.0)
