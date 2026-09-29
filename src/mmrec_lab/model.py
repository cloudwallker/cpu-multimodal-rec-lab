"""CPU adaptation of Xin Zhou's FREEDOM implementation in enoche/MMRec.

Source: https://github.com/enoche/MMRec/blob/master/src/models/freedom.py
Original author: Xin Zhou. Adaptation date: 2026-09-29.
SPDX-License-Identifier: GPL-3.0-only
See docs/model-implementation.md for preserved behavior and explicit deviations.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CONFIG = {
    "embedding_size": 64, "feat_embed_dim": 64, "knn_k": 10,
    "n_mm_layers": 1, "n_ui_layers": 2, "mm_image_weight": 0.1,
    "dropout": 0.8, "reg_weight": 1e-4,
}


def _sparse(indices: torch.Tensor, values: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32,
                                   check_invariants=True).coalesce()


class FreedomModel(nn.Module):
    """Fixed modality graph, trainable feature tables and pruned UI training graph.

    Graphs are derived from this constructor's inputs and excluded from parameter
    checkpoints. Restore with identical initial inputs/config and then load the
    state dict. ``scores`` always uses the complete training interaction graph.
    """

    def __init__(self, n_users: int, n_items: int, train: np.ndarray,
                 vision: np.ndarray | None, text: np.ndarray | None,
                 config: dict, seed: int = 0):
        super().__init__()
        if n_users < 1 or n_items < 1:
            raise ValueError("n_users and n_items must be positive")
        self.n_users, self.n_items = int(n_users), int(n_items)
        self.device = torch.device("cpu")
        self.config = {**DEFAULT_CONFIG, **config}
        for key in ["embedding_size", "feat_embed_dim", "knn_k", "n_mm_layers", "n_ui_layers"]:
            value = self.config[key]
            if not isinstance(value, (int, np.integer)) or value < (0 if "layers" in key else 1):
                raise ValueError(f"Invalid model parameter: {key}")
        if self.config["embedding_size"] != self.config["feat_embed_dim"]:
            raise ValueError("Auxiliary BPR requires embedding_size == feat_embed_dim")
        for key in ["mm_image_weight", "dropout", "reg_weight"]:
            value = self.config[key]
            if not np.isfinite(value) or value < 0 or (key != "reg_weight" and value > 1):
                raise ValueError(f"Invalid model parameter: {key}")
        self.embedding_dim = self.config["embedding_size"]
        self.feat_embed_dim = self.config["feat_embed_dim"]
        self.knn_k = self.config["knn_k"]
        self.n_layers = self.config["n_mm_layers"]
        self.n_ui_layers = self.config["n_ui_layers"]
        self.mm_image_weight = self.config["mm_image_weight"]
        self.dropout = self.config["dropout"]
        self.reg_weight = self.config["reg_weight"]
        train = np.asarray(train)
        if train.ndim != 2 or train.shape[1] != 2 or not np.issubdtype(train.dtype, np.integer):
            raise ValueError("train must be an Nx2 array of integer IDs")
        if len(train) and (train.min() < 0 or train[:, 0].max() >= n_users or train[:, 1].max() >= n_items):
            raise ValueError("Training IDs exceed the model vocabulary")
        self.register_buffer("norm_adj", self._norm_ui_graph(train), persistent=False)
        edges = torch.from_numpy(np.array(train.T, dtype=np.int64, copy=True))
        self.register_buffer("edge_indices", edges, persistent=False)
        self.register_buffer("edge_values", self._normalize_edges(edges), persistent=False)
        self.register_buffer("masked_adj", self.norm_adj, persistent=False)

        # A local initialization stream avoids perturbing sampling/evaluation RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
            self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_id_embedding.weight)
            graphs = {}
            for name, features in [("image", vision), ("text", text)]:
                if features is None:
                    continue
                features = np.array(features, dtype=np.float32, copy=True)
                if features.ndim != 2 or features.shape[0] != n_items or features.shape[1] < 1:
                    raise ValueError(f"{name} features must have n_items rows and nonempty columns")
                if not np.isfinite(features).all():
                    raise ValueError(f"{name} features contain nonfinite values")
                weight = torch.from_numpy(features)
                setattr(self, name + "_embedding", nn.Embedding.from_pretrained(weight, freeze=False))
                setattr(self, name + "_trs", nn.Linear(weight.shape[1], self.feat_embed_dim))
                graphs[name] = self._knn_graph(weight)
        if not graphs:
            raise ValueError("At least one modality feature matrix is required")
        if len(graphs) == 1:
            mm_adj = next(iter(graphs.values()))
        elif self.mm_image_weight == 0:
            mm_adj = graphs["text"]
        elif self.mm_image_weight == 1:
            mm_adj = graphs["image"]
        else:
            mm_adj = (self.mm_image_weight * graphs["image"] +
                      (1 - self.mm_image_weight) * graphs["text"]).coalesce()
        self.register_buffer("mm_adj", mm_adj, persistent=False)

    def _norm_ui_graph(self, train: np.ndarray) -> torch.Tensor:
        interaction = sp.coo_matrix((np.ones(len(train), dtype=np.float32),
                                    (train[:, 0], train[:, 1])),
                                   shape=(self.n_users, self.n_items)).tocsr()
        interaction.data[:] = 1  # official DOK graph binarizes repeated coordinates
        adj = sp.bmat([[None, interaction], [interaction.T, None]], format="csr")
        degree = np.asarray((adj > 0).sum(axis=1)).reshape(-1) + 1e-7
        scale = sp.diags(np.power(degree, -0.5))
        norm = (scale @ adj @ scale).tocoo()
        indices = torch.from_numpy(np.stack([norm.row, norm.col]).astype(np.int64))
        values = torch.from_numpy(norm.data.astype(np.float32))
        return _sparse(indices, values, (self.n_users + self.n_items,) * 2)

    def _normalize_edges(self, edges: torch.Tensor) -> torch.Tensor:
        row_degree = torch.bincount(edges[0], minlength=self.n_users).float() + 1e-7
        col_degree = torch.bincount(edges[1], minlength=self.n_items).float() + 1e-7
        return row_degree.pow(-0.5)[edges[0]] * col_degree.pow(-0.5)[edges[1]]

    def _knn_graph(self, weight: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            norm = torch.norm(weight, p=2, dim=-1, keepdim=True)
            valid = torch.where(norm[:, 0] > 0)[0]
            if not len(valid):
                return _sparse(torch.empty((2, 0), dtype=torch.long), torch.empty(0), (self.n_items,) * 2)
            normalized = weight[valid] / norm[valid]
            similarity = normalized @ normalized.T
            k = min(self.knn_k, len(valid))
            neighbors = torch.topk(similarity, k, dim=-1).indices
            rows = valid[:, None].expand(-1, k).reshape(-1)
            cols = valid[neighbors].reshape(-1)
            indices = torch.stack([rows, cols])
            degree = torch.bincount(rows, minlength=self.n_items).float() + 1e-7
            inv = degree.pow(-0.5)
            return _sparse(indices, inv[rows] * inv[cols], (self.n_items,) * 2)

    def sample_epoch_graph(self, generator: torch.Generator) -> None:
        """Sample degree-sensitive training edges using a caller-owned CPU RNG."""
        if self.dropout <= 0:
            self.masked_adj = self.norm_adj
            return
        count = int(self.edge_values.numel() * (1 - self.dropout))
        if count:
            selected = torch.multinomial(self.edge_values, count, replacement=False, generator=generator)
            retained = self.edge_indices[:, selected].clone()
            values = self._normalize_edges(retained)
            retained[1] += self.n_users
            indices = torch.cat([retained, retained.flip(0)], dim=1)
            values = torch.cat([values, values])
        else:
            indices, values = torch.empty((2, 0), dtype=torch.long), torch.empty(0)
        self.masked_adj = _sparse(indices, values, (self.n_users + self.n_items,) * 2)

    def forward(self, adj: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        adj = self.norm_adj if adj is None else adj
        h = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        ego = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        layers = [ego]
        for _ in range(self.n_ui_layers):
            ego = torch.sparse.mm(adj, ego)
            layers.append(ego)
        propagated = torch.stack(layers, dim=1).mean(dim=1)
        users, items = torch.split(propagated, [self.n_users, self.n_items], dim=0)
        return users, items + h

    @staticmethod
    def _bpr(users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        difference = (users * pos).sum(dim=1) - (users * neg).sum(dim=1)
        return -F.logsigmoid(difference).mean()

    def loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.forward(self.masked_adj)
        batch_users = user_emb[users]
        loss = self._bpr(batch_users, item_emb[pos], item_emb[neg])
        sampled = torch.cat([pos, neg])
        for name in ["image", "text"]:
            embedding = getattr(self, name + "_embedding", None)
            if embedding is not None:
                projected = getattr(self, name + "_trs")(embedding(sampled))
                positive, negative = projected.split(len(pos), dim=0)
                loss = loss + self.reg_weight * self._bpr(batch_users, positive, negative)
        return loss

    def scores(self, users: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.forward(self.norm_adj)
        return user_emb[users] @ item_emb.T
