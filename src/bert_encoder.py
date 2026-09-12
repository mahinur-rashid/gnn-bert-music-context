"""BERT text encoder and the Task-1 multi-label tag classifier.

Implements

    H_text = BERT(X_text) in R^{L x d},      t = H_text[CLS]
    y_hat_k = sigmoid(w_k^T t + b_k)

with a binary-cross-entropy-per-tag objective.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .config import CFG
from .utils import LOG


# --------------------------------------------------------------------------- #
# tokenised dataset
# --------------------------------------------------------------------------- #


class TextTagDataset(Dataset):
    """Tokenised (text, multi-hot label) pairs; optional valence/arousal targets."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 192,
        va: np.ndarray | None = None,
        ids: Sequence[str] | None = None,
    ):
        self.texts = [str(t) for t in texts]
        self.labels = np.asarray(labels, dtype=np.float32)
        self.va = None if va is None else np.asarray(va, dtype=np.float32)
        self.ids = list(ids) if ids is not None else [str(i) for i in range(len(self.texts))]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "y": torch.from_numpy(self.labels[i]),
            "index": torch.tensor(i, dtype=torch.long),
        }
        if self.va is not None:
            item["va"] = torch.from_numpy(self.va[i])
        return item


def make_tokenizer(model_name: str | None = None):
    from transformers import AutoTokenizer

    model_name = model_name or CFG["text"]["model_name"]
    return AutoTokenizer.from_pretrained(model_name)


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #


class BertEncoder(nn.Module):
    """Wraps a HuggingFace encoder and exposes (CLS vector, token states)."""

    def __init__(
        self,
        model_name: str | None = None,
        freeze: bool = False,
        n_trainable_layers: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.model_name = model_name or CFG["text"]["model_name"]
        cfg = AutoConfig.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = int(getattr(cfg, "hidden_size", getattr(cfg, "dim", 768)))
        self.dropout = nn.Dropout(dropout)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
            LOG.info("BERT frozen (%s)", self.model_name)
        elif n_trainable_layers is not None:
            self._freeze_all_but_last(n_trainable_layers)

    def _encoder_layers(self) -> list[nn.Module]:
        """Transformer blocks across BERT / DistilBERT / RoBERTa naming."""
        for attr in ("encoder", "transformer"):
            enc = getattr(self.bert, attr, None)
            if enc is None:
                continue
            for name in ("layer", "layers"):
                layers = getattr(enc, name, None)
                if layers is not None:
                    return list(layers)
        return []

    def _freeze_all_but_last(self, k: int) -> None:
        for p in self.bert.parameters():
            p.requires_grad = False
        layers = self._encoder_layers()
        for layer in layers[-k:] if k > 0 else []:
            for p in layer.parameters():
                p.requires_grad = True
        LOG.info("BERT: fine-tuning the last %d of %d transformer blocks", k, len(layers))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        output_attentions=output_attentions)
        tokens = out.last_hidden_state                 # [B, L, d]
        cls = self.dropout(tokens[:, 0])               # [B, d]  -- the [CLS] vector
        attentions = getattr(out, "attentions", None) if output_attentions else None
        return cls, tokens, attentions


# --------------------------------------------------------------------------- #
# Task 1 model
# --------------------------------------------------------------------------- #


class BertTagClassifier(nn.Module):
    """BERT_CLS -> linear head -> K tag logits (Algorithm 1)."""

    def __init__(
        self,
        n_tags: int,
        model_name: str | None = None,
        freeze: bool = False,
        n_trainable_layers: int | None = None,
        dropout: float = 0.1,
        n_va: int = 0,
    ):
        super().__init__()
        self.encoder = BertEncoder(model_name, freeze, n_trainable_layers, dropout)
        d = self.encoder.hidden_size
        self.head = nn.Linear(d, n_tags)
        self.va_head = nn.Linear(d, n_va) if n_va else None

    def forward(self, input_ids, attention_mask, output_attentions: bool = False):
        cls, _, attn = self.encoder(input_ids, attention_mask, output_attentions)
        logits = self.head(cls)
        va = torch.tanh(self.va_head(cls)) if self.va_head is not None else None
        return {"logits": logits, "va": va, "cls": cls, "attentions": attn}


# --------------------------------------------------------------------------- #
# attention visualisation helper (optional Task-1 deliverable)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def cls_attention_tokens(model: BertTagClassifier, tokenizer, text: str,
                         device: torch.device, max_length: int = 192,
                         top_n: int = 10) -> list[tuple[str, float]]:
    """Tokens most attended to by [CLS] in the last transformer block."""
    model.eval()
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt").to(device)
    out = model(enc["input_ids"], enc["attention_mask"], output_attentions=True)
    if not out["attentions"]:
        return []
    last = out["attentions"][-1][0]                    # [heads, L, L]
    cls_attn = last.mean(0)[0]                         # [L] -- averaged over heads
    ids = enc["input_ids"][0]
    mask = enc["attention_mask"][0].bool()
    toks = tokenizer.convert_ids_to_tokens(ids[mask])
    scores = cls_attn[mask].float().cpu().numpy()
    order = np.argsort(-scores)[:top_n]
    return [(toks[i], float(scores[i])) for i in order
            if toks[i] not in ("[CLS]", "[SEP]", "[PAD]")]
