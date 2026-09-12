"""Task 3 -- GNN-BERT fusion for multi-context understanding.

Cross-attention fusion (recommended variant in the brief):

    Q = g W_Q,  K = H_text W_K,  A = softmax(Q K^T / sqrt(d))
    z = CONCAT(g, A H_text),     y_hat = sigma(W z)

Multi-task objective:

    L = L_tags + alpha ||v - v_hat||^2 + beta ||a - a_hat||^2

Four fusion modes are provided so the ablation required by the brief
(BERT-only / GNN-only / early concat / cross-attention) is a single flag.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_encoder import BertEncoder
from .gnn_model import GNNEncoder

FUSION_MODES = ["cross_attn", "concat", "bert_only", "gnn_only"]


# --------------------------------------------------------------------------- #
# cross-attention block
# --------------------------------------------------------------------------- #


class CrossAttentionFusion(nn.Module):
    """Graph readout g attends over the BERT token states H_text."""

    def __init__(self, d_graph: int, d_text: int, d_model: int = 256,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_graph, d_model)
        self.kv_proj = nn.Linear(d_text, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, g: torch.Tensor, H_text: torch.Tensor,
                attention_mask: torch.Tensor | None = None):
        q = self.q_proj(g).unsqueeze(1)                      # [B, 1, d]
        kv = self.kv_proj(H_text)                            # [B, L, d]
        pad = None if attention_mask is None else (attention_mask == 0)
        ctx, weights = self.attn(q, kv, kv, key_padding_mask=pad,
                                 need_weights=True, average_attn_weights=True)
        ctx = self.norm(ctx.squeeze(1))                      # [B, d]
        return ctx, weights.squeeze(1)                       # attn: [B, L]


# --------------------------------------------------------------------------- #
# full model
# --------------------------------------------------------------------------- #


class FusionModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_tags: int,
        mode: str = "cross_attn",
        model_name: str | None = None,
        hidden: int = 256,
        gnn_layers: int = 3,
        conv: str = "sage",
        dropout: float = 0.3,
        n_heads: int = 4,
        n_va: int = 0,
        freeze_bert: bool = False,
        n_trainable_layers: int | None = None,
        edge_dim: int | None = None,
    ):
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"mode must be one of {FUSION_MODES}")
        self.mode = mode
        self.use_graph = mode != "bert_only"
        self.use_text = mode != "gnn_only"

        if self.use_graph:
            self.gnn = GNNEncoder(in_dim, hidden, gnn_layers, conv, dropout,
                                  out_dim=hidden, heads=n_heads, edge_dim=edge_dim)
        if self.use_text:
            self.bert = BertEncoder(model_name, freeze=freeze_bert,
                                    n_trainable_layers=n_trainable_layers, dropout=0.1)
            d_text = self.bert.hidden_size
            self.text_proj = nn.Sequential(nn.Linear(d_text, hidden), nn.ReLU(inplace=True))

        if mode == "cross_attn":
            self.fusion = CrossAttentionFusion(hidden, d_text, hidden, n_heads, dropout)
            z_dim = 2 * hidden
        elif mode == "concat":
            z_dim = 2 * hidden
        else:
            z_dim = hidden

        self.z_dim = z_dim
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, n_tags),
        )
        self.va_head = nn.Sequential(
            nn.Linear(z_dim, hidden // 2), nn.ReLU(inplace=True), nn.Linear(hidden // 2, n_va)
        ) if n_va else None

    # ------------------------------------------------------------------ #

    def forward(self, data, output_attentions: bool = False):
        g = h = None
        if self.use_graph:
            g, h = self.gnn(data.x, data.edge_index, data.batch,
                            getattr(data, "edge_attr", None))

        t = tokens = attn = None
        if self.use_text:
            cls, tokens, _ = self.bert(data.input_ids, data.attention_mask)
            t = self.text_proj(cls)

        if self.mode == "gnn_only":
            z = g
        elif self.mode == "bert_only":
            z = t
        elif self.mode == "concat":
            z = torch.cat([g, t], dim=-1)
        else:
            ctx, attn = self.fusion(g, tokens, data.attention_mask)
            z = torch.cat([g, ctx], dim=-1)

        z = self.dropout(z)
        out = {"logits": self.head(z), "z": z, "g": g, "t": t, "h": h}
        out["va"] = torch.tanh(self.va_head(z)) if self.va_head is not None else None
        if output_attentions:
            out["text_attention"] = attn
        return out


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #


def multitask_loss(out: dict, y: torch.Tensor, va: torch.Tensor | None = None,
                   alpha: float = 0.5, beta: float = 0.5,
                   pos_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """L_tags (BCE) + alpha * MSE(valence) + beta * MSE(arousal)."""
    logits = out["logits"].float()
    if y.dtype == torch.long and y.dim() == 1:
        tag_loss = F.cross_entropy(logits, y)
    else:
        tag_loss = F.binary_cross_entropy_with_logits(logits, y.float(),
                                                      pos_weight=pos_weight)
    total = tag_loss
    parts = {"tags": float(tag_loss.detach())}

    if va is not None and out.get("va") is not None:
        pred = out["va"].float()
        v = F.mse_loss(pred[:, 0], va[:, 0])
        a = F.mse_loss(pred[:, 1], va[:, 1])
        total = total + alpha * v + beta * a
        parts["valence"] = float(v.detach())
        parts["arousal"] = float(a.detach())
    return total, parts
