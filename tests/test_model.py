"""Model boundary tests: zero-feature safety, graph choice and training."""
import importlib

import numpy as np
import pytest


def model_class():
    global torch
    spec = importlib.util.find_spec("mmrec_lab.model")
    assert spec is not None, "FreedomModel implementation is missing"
    torch = importlib.import_module("torch")
    return importlib.import_module("mmrec_lab.model").FreedomModel


def make_model(**updates):
    config = dict(embedding_size=4, feat_embed_dim=4, knn_k=2,
                  dropout=0.5, reg_weight=0.1)
    config.update(updates)
    train = np.array([[0, 0], [0, 1], [1, 1], [1, 2]])
    vision = np.array([[1., 0., 0.], [0., 0., 0.], [0., 1., 0.]])
    text = np.array([[1., 0.], [1., 1.], [0., 1.]])
    return model_class()(2, 3, train, vision, text, config, seed=7)


def test_zero_feature_nodes_have_no_modality_edges():
    model = make_model(mm_image_weight=1.0)
    graph = model.mm_adj.to_dense()
    assert torch.isfinite(graph).all()
    assert torch.count_nonzero(graph[1]) == 0
    assert torch.count_nonzero(graph[:, 1]) == 0
    torch.testing.assert_close(graph[[0, 2]][:, [0, 2]], torch.full((2, 2), 0.5))


def test_entire_missing_modality_has_empty_graph_and_finite_loss():
    cls = model_class()
    model = cls(1, 2, np.array([[0, 0]]), np.zeros((2, 3)), None,
                dict(embedding_size=4, feat_embed_dim=4, knn_k=10), seed=1)
    assert model.mm_adj._nnz() == 0
    model.sample_epoch_graph(torch.Generator().manual_seed(3))
    assert model.masked_adj._nnz() == 0  # floor(1 * 0.2) == 0
    loss = model.loss(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    assert torch.isfinite(loss)
    loss.backward()


def test_scores_use_full_graph_after_epoch_pruning():
    model = make_model()
    users = torch.tensor([0, 1])
    before = model.scores(users).detach().clone()
    model.sample_epoch_graph(torch.Generator().manual_seed(9))
    assert model.masked_adj._nnz() == 4  # two retained bipartite edges
    torch.testing.assert_close(model.scores(users), before)
    u, i = model.forward()
    torch.testing.assert_close(model.scores(users), u @ i.T)


def test_sampling_uses_explicit_rng_and_checkpoint_restores_scores():
    model = make_model()
    expected = model.scores(torch.tensor([0, 1])).detach().clone()
    model.sample_epoch_graph(torch.Generator().manual_seed(11))
    sample = model.masked_adj.to_dense().clone()
    torch.manual_seed(987)
    model.sample_epoch_graph(torch.Generator().manual_seed(11))
    torch.testing.assert_close(model.masked_adj.to_dense(), sample)
    restored = make_model()
    with torch.no_grad():
        restored.user_embedding.weight.fill_(123)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.scores(torch.tensor([0, 1])), expected)


@pytest.mark.parametrize("modality", ["vision", "text"])
def test_single_modality_features_remain_trainable(modality):
    cls = model_class()
    feat = np.array([[1., 0.], [0., 1.], [1., 1.]])
    model = cls(2, 3, np.array([[0, 0], [1, 1]]),
                feat if modality == "vision" else None,
                feat if modality == "text" else None,
                dict(embedding_size=4, feat_embed_dim=4, knn_k=2, dropout=0), seed=2)
    loss = model.loss(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 2]))
    loss.backward()
    embedding = model.image_embedding if modality == "vision" else model.text_embedding
    assert embedding.weight.requires_grad
    assert embedding.weight.grad is not None
    assert torch.count_nonzero(embedding.weight.grad) > 0


def test_loss_projects_only_sampled_features():
    model = make_model(dropout=0)
    projected_rows = []
    hook = model.image_trs.register_forward_pre_hook(
        lambda module, args: projected_rows.append(args[0].shape[0]))
    try:
        model.loss(torch.tensor([0]), torch.tensor([0]), torch.tensor([2]))
    finally:
        hook.remove()
    assert projected_rows and sum(projected_rows) == 2


def test_small_training_loss_decreases():
    model = make_model(dropout=0, reg_weight=0.01)
    users, pos, neg = torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = model.loss(users, pos, neg).item()
    for _ in range(15):
        optimizer.zero_grad()
        loss = model.loss(users, pos, neg)
        loss.backward()
        optimizer.step()
    assert model.loss(users, pos, neg).item() < initial * 0.8


@pytest.mark.parametrize("change", ["nonfinite", "wrong_rows", "wrong_ids"])
def test_malformed_input_is_rejected(change):
    cls = model_class()
    train, feat = np.array([[0, 0]]), np.ones((2, 3))
    if change == "nonfinite":
        feat[0, 0] = np.nan
    elif change == "wrong_rows":
        feat = np.ones((3, 3))
    else:
        train[0, 1] = 2
    with pytest.raises(ValueError):
        cls(1, 2, train, feat, None, {}, seed=1)
