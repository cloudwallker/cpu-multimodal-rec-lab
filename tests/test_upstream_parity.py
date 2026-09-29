"""Execute the frozen author's real FREEDOM class, not a mirrored formula."""
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_upstream(monkeypatch):
    path = ROOT / "external" / "MMRec" / "src" / "models" / "freedom.py"
    if not path.exists():
        path = ROOT / "external" / "FREEDOM" / "src" / "models" / "freedom.py"
    assert path.exists(), "Frozen official FREEDOM source is required for parity"

    # The actual base class loads files and imports unrelated LMDB/PIL helpers.
    # This adapter supplies only dataset counts and already-loaded features.
    class FeatureBase(nn.Module):
        def __init__(self, config, dataset):
            super().__init__()
            self.n_users, self.n_items = dataset.shape
            self.device = torch.device("cpu")
            self.v_feat = dataset.vision
            self.t_feat = dataset.text

    base = types.ModuleType("common.abstract_recommender")
    base.GeneralRecommender = FeatureBase
    loss = types.ModuleType("common.loss")
    for name in ["BPRLoss", "EmbLoss", "L2Loss"]:
        setattr(loss, name, object)  # unused imports in upstream FREEDOM
    utils = types.ModuleType("utils.utils")
    utils.build_sim = utils.compute_normalized_laplacian = None  # unused imports
    for name, module in [("common", types.ModuleType("common")),
                         ("common.abstract_recommender", base), ("common.loss", loss),
                         ("utils", types.ModuleType("utils")), ("utils.utils", utils)]:
        monkeypatch.setitem(sys.modules, name, module)
    if not hasattr(sp.dok_matrix, "_update"):
        def update_dok(matrix, entries):
            for index, value in entries.items():
                matrix[index] = value
        monkeypatch.setattr(sp.dok_matrix, "_update", update_dok, raising=False)
    spec = importlib.util.spec_from_file_location("official_freedom_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FREEDOM


@pytest.mark.filterwarnings("ignore:torch.sparse.SparseTensor:UserWarning")
@pytest.mark.parametrize("modalities", ["both", "vision", "text"])
@pytest.mark.parametrize("layers", [(1, 2), (2, 1)])
def test_official_graph_forward_loss_gradients_and_pruning(monkeypatch, tmp_path, modalities, layers):
    assert importlib.util.find_spec("mmrec_lab.model") is not None, "FreedomModel is missing"
    FreedomModel = importlib.import_module("mmrec_lab.model").FreedomModel
    Official = load_upstream(monkeypatch)
    train = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
    rng = np.random.default_rng(87)
    vision = rng.normal(size=(4, 5)).astype(np.float32) if modalities != "text" else None
    text = rng.normal(size=(4, 6)).astype(np.float32) if modalities != "vision" else None
    config = dict(embedding_size=4, feat_embed_dim=4, knn_k=2, lambda_coeff=0.9,
                  cf_model="LightGCN", n_mm_layers=layers[0], n_ui_layers=layers[1],
                  reg_weight=0.03, mm_image_weight=0.1, dropout=0.5, degree_ratio=1.0,
                  data_path=str(tmp_path) + "/", dataset="baby")
    (tmp_path / "baby").mkdir()

    class Dataset:
        shape = (3, 4)
        def inter_matrix(self, form="coo"):
            return sp.coo_matrix((np.ones(len(train)), (train[:, 0], train[:, 1])), shape=self.shape)
    dataset = Dataset()
    dataset.vision = None if vision is None else torch.from_numpy(vision.copy())
    dataset.text = None if text is None else torch.from_numpy(text.copy())
    official = Official(config, dataset)
    adapted = FreedomModel(3, 4, train, vision, text, config, seed=12)
    for name, parameter in official.named_parameters():
        with torch.no_grad():
            parameter.copy_(dict(adapted.named_parameters())[name])
    assert set(dict(official.named_parameters())) == set(dict(adapted.named_parameters()))
    torch.testing.assert_close(adapted.norm_adj.to_dense(), official.norm_adj.to_dense(), atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(adapted.mm_adj.to_dense(), official.mm_adj.to_dense(), atol=2e-7, rtol=2e-6)
    for actual, expected in zip(adapted.forward(), official.forward(official.norm_adj)):
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)
    users, pos, neg = torch.tensor([0, 1, 2, 0]), torch.tensor([0, 1, 3, 0]), torch.tensor([3, 0, 2, 2])
    torch.testing.assert_close(adapted.scores(users), official.full_sort_predict([users]), atol=3e-7, rtol=3e-6)
    torch.manual_seed(123)
    official.pre_epoch_processing()
    adapted.sample_epoch_graph(torch.Generator().manual_seed(123))
    torch.testing.assert_close(adapted.masked_adj.to_dense(), official.masked_adj.to_dense(), atol=2e-7, rtol=2e-6)
    official_loss = official.calculate_loss([users, pos, neg])
    adapted_loss = adapted.loss(users, pos, neg)
    torch.testing.assert_close(adapted_loss, official_loss, atol=3e-7, rtol=3e-6)
    official_loss.backward()
    adapted_loss.backward()
    for name, parameter in official.named_parameters():
        torch.testing.assert_close(dict(adapted.named_parameters())[name].grad, parameter.grad,
                                   atol=5e-7, rtol=5e-5, msg=lambda msg: name + ": " + msg)
