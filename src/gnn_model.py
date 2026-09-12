"""GNN encoders on music-structure graphs + the CNN mel-spectrogram baseline.

GraphSAGE update (Task 2):

    h_i^{l+1} = sigma( W^l . CONCAT( h_i^l, MEAN_{j in N(i)} h_j^l ) )

Graph readout: mean pooling (concatenated with max pooling, which is strictly
more informative and costs nothing):

    g = 1/|V| sum_i h_i^{L},      y_hat = sigma(W g + b)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool, global_mean_pool


# --------------------------------------------------------------------------- #
# graph encoder
# --------------------------------------------------------------------------- #


class GNNEncoder(nn.Module):
    """Stacked SAGE / GAT / GCN layers with mean+max readout."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        layers: int = 3,
        conv: str = "sage",
        dropout: float = 0.3,
        out_dim: int | None = None,
        heads: int = 4,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.conv_type = conv
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.input_norm = nn.BatchNorm1d(in_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            if conv == "sage":
                c = SAGEConv(d, hidden, aggr="mean")
            elif conv == "gat":
                c = GATConv(d, hidden // heads, heads=heads, dropout=dropout,
                            edge_dim=edge_dim)
            elif conv == "gcn":
                c = GCNConv(d, hidden)
            else:
                raise ValueError(f"unknown conv {conv!r}")
            self.convs.append(c)
            self.norms.append(nn.BatchNorm1d(hidden))
            d = hidden

        self.out_dim = out_dim or hidden
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = self.input_norm(x)
        for conv, norm in zip(self.convs, self.norms):
            if self.conv_type == "gat" and self.edge_dim is not None and edge_attr is not None:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = F.relu(norm(h), inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        return self.readout(g), h


class GNNClassifier(nn.Module):
    """Task 2 model: GNN encoder -> MLP head over graph-level features."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, layers: int = 3,
                 conv: str = "sage", dropout: float = 0.3, heads: int = 4,
                 edge_dim: int | None = None):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden, layers, conv, dropout,
                                  out_dim=hidden, heads=heads, edge_dim=edge_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )

    def forward(self, data):
        g, h = self.encoder(data.x, data.edge_index, data.batch,
                            getattr(data, "edge_attr", None))
        return {"logits": self.head(g), "g": g, "h": h}


# --------------------------------------------------------------------------- #
# B2 -- CNN on the mel-spectrogram (no graph, no text)
# --------------------------------------------------------------------------- #


class MelCNN(nn.Module):
    """Four conv blocks -> global average pooling -> linear head."""

    def __init__(self, n_classes: int, in_ch: int = 1, width: int = 32, dropout: float = 0.3):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, width), block(width, width * 2),
            block(width * 2, width * 4), block(width * 4, width * 4),
        )
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = width * 4
        self.head = nn.Linear(self.embed_dim, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.features(x)
        g = h.mean(dim=(2, 3))
        return {"logits": self.head(self.dropout(g)), "g": g}
