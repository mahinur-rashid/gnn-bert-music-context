"""Task 4 (Advanced) -- contrastive GNN-BERT alignment (model + objective).

The dual encoder, the InfoNCE objective and the retrieval metrics live here;
the training/evaluation driver is :mod:`src.train_task4`.

    python -m src.graph_builder --dataset musiccaps --jobs 10
    python -m src.train_task4  --dataset musiccaps

InfoNCE over paired (graph, caption):

    L = -log exp(sim(g_i, t_i)/tau) / sum_j exp(sim(g_i, t_j)/tau)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_encoder import BertEncoder
from .gnn_model import GNNEncoder


class DualEncoder(nn.Module):
    """Graph encoder and text encoder projected into a shared embedding space."""

    def __init__(self, in_dim: int, embed_dim: int = 256, hidden: int = 256,
                 gnn_layers: int = 3, conv: str = "sage", dropout: float = 0.2,
                 model_name: str | None = None, freeze_bert: bool = False,
                 n_trainable_layers: int | None = None, temperature: float = 0.07,
                 learn_temperature: bool = True):
        super().__init__()
        self.gnn = GNNEncoder(in_dim, hidden, gnn_layers, conv, dropout, out_dim=hidden)
        self.bert = BertEncoder(model_name, freeze=freeze_bert,
                                n_trainable_layers=n_trainable_layers)
        self.graph_proj = nn.Linear(hidden, embed_dim)
        self.text_proj = nn.Linear(self.bert.hidden_size, embed_dim)
        # CLIP-style learnable temperature by default; set learn_temperature=False
        # to keep tau fixed at the configured value, exactly as written in the brief
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature).log(),
                                        requires_grad=learn_temperature)

    def encode_graph(self, data) -> torch.Tensor:
        g, _ = self.gnn(data.x, data.edge_index, data.batch,
                        getattr(data, "edge_attr", None))
        return F.normalize(self.graph_proj(g), dim=-1)

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        cls, _, _ = self.bert(input_ids, attention_mask)
        return F.normalize(self.text_proj(cls), dim=-1)

    def forward(self, data):
        g = self.encode_graph(data)
        t = self.encode_text(data.input_ids, data.attention_mask)
        return {"g": g, "t": t, "logit_scale": self.logit_scale.exp().clamp(max=100.0)}


def info_nce(g: torch.Tensor, t: torch.Tensor, logit_scale: torch.Tensor):
    """Symmetric InfoNCE (graph->caption and caption->graph)."""
    logits = logit_scale * g @ t.t()
    target = torch.arange(len(g), device=g.device)
    loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
    return loss, logits


@torch.no_grad()
def retrieval_metrics(g: torch.Tensor, t: torch.Tensor,
                      ks=(1, 5, 10)) -> dict[str, float]:
    """Caption->Audio and Audio->Caption R@K plus median rank."""
    sim = g @ t.t()
    n = len(g)
    target = torch.arange(n, device=sim.device)
    out: dict[str, float] = {}
    for name, S in (("audio_to_caption", sim), ("caption_to_audio", sim.t())):
        ranks = (S.argsort(dim=1, descending=True) == target[:, None]).float().argmax(1)
        for k in ks:
            out[f"{name}_R@{k}"] = float((ranks < k).float().mean())
        out[f"{name}_median_rank"] = float(ranks.median() + 1)
    return out


def main() -> None:  # pragma: no cover
    from .train_task4 import main as run

    run()


if __name__ == "__main__":
    main()
