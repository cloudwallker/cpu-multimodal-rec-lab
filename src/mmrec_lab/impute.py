"""Train-only co-interaction neighbors and one-step visible-feature imputation."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from numbers import Real

METHODS = ('zero_fill', 'global_mean', 'paper_neighmean', 'observed_mean', 'fixed_mix', 'support_mix')


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _bool_mask(value, size, name):
    value = np.asarray(value)
    if value.dtype != np.bool_ or value.shape != (size,):
        raise ValueError(f'{name} must be a boolean vector of length {size}')
    return value


def build_neighbors(train: np.ndarray, n_users: int, n_items: int, k: int = 20) -> sparse.csr_matrix:
    """Directed binary Top-K; duplicates count once, ties use ascending item ID."""
    n_users = _positive_int(n_users, 'n_users')
    n_items = _positive_int(n_items, 'n_items')
    k = _positive_int(k, 'k')
    train = np.asarray(train)
    if train.ndim != 2 or train.shape[1] != 2 or not np.issubdtype(train.dtype, np.integer):
        raise ValueError('train must be an integer N-by-2 array')
    if train.size and (np.any(train < 0) or np.any(train[:, 0] >= n_users) or np.any(train[:, 1] >= n_items)):
        raise ValueError('training IDs out of range')
    pairs = np.unique(train, axis=0)
    b = sparse.csr_matrix((np.ones(len(pairs), dtype=np.int64), (pairs[:, 0], pairs[:, 1])), shape=(n_users, n_items))
    counts = (b.T @ b).tocsr()
    counts.setdiag(0)
    counts.eliminate_zeros()
    rows, cols = [], []
    for item in range(n_items):
        start, end = counts.indptr[item:item + 2]
        ids = counts.indices[start:end]
        weights = counts.data[start:end]
        selected = ids[np.lexsort((ids, -weights))[:k]]
        rows.extend([item] * len(selected))
        cols.extend(selected.tolist())
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_items, n_items))


def make_missing(n_items: int, tail: np.ndarray, scenario: str, seed: int) -> np.ndarray:
    n_items = _positive_int(n_items, 'n_items')
    tail = _bool_mask(tail, n_items, 'tail')
    if scenario not in ('clean', 'uniform30', 'tail30'):
        raise ValueError('unknown missing scenario')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    missing = np.zeros(n_items, dtype=bool)
    if scenario == 'clean':
        return missing
    count = 3 * n_items // 10
    candidates = np.flatnonzero(tail) if scenario == 'tail30' else np.arange(n_items)
    if len(candidates) < count:
        raise ValueError('tail has insufficient items for the requested missing count')
    missing[np.random.default_rng(seed).choice(candidates, size=count, replace=False)] = True
    return missing


def impute_features(visible: np.ndarray, missing: np.ndarray, neighbors: sparse.csr_matrix,
                    method: str, alpha: float = .5, tau: float = 4) -> tuple[np.ndarray, dict]:
    """Never accepts hidden feature values; missing rows must already be exactly zero.

    All means are in the original feature coordinates. Newly imputed rows are never
    reused as neighbor evidence. Paper NeighMean uses *all* neighbors as divisor.
    """
    visible = np.asarray(visible)
    if visible.ndim != 2 or 0 in visible.shape or not np.issubdtype(visible.dtype, np.floating):
        raise ValueError('visible must be a nonempty floating point matrix')
    if not np.isfinite(visible).all():
        raise ValueError('visible contains nonfinite values')
    n_items = visible.shape[0]
    missing = _bool_mask(missing, n_items, 'missing')
    if missing.all():
        raise ValueError('no observed image features remain')
    if np.any(visible[missing] != 0):
        raise ValueError('hidden rows must already be zero; original features are forbidden')
    if method not in METHODS:
        raise ValueError('unknown imputation method')
    if not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite and in [0, 1]')
    if not isinstance(tau, Real) or not np.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite and positive')
    if not sparse.isspmatrix_csr(neighbors) or neighbors.shape != (n_items, n_items):
        raise ValueError('neighbors must be an N-by-N CSR matrix')
    graph = neighbors.copy()
    graph.sum_duplicates()
    graph.eliminate_zeros()
    if not np.isfinite(graph.data).all() or np.any(graph.data != 1) or np.any(graph.diagonal() != 0):
        raise ValueError('neighbors must be binary with no self edges')
    degree = np.diff(graph.indptr).astype(np.int64)
    support = np.asarray(graph @ (~missing).astype(np.int64)).reshape(-1).astype(np.int64)
    mu = visible[~missing].mean(axis=0, dtype=np.float64)
    sums = np.asarray(graph @ visible.astype(np.float64))
    observed = np.broadcast_to(mu, visible.shape).copy()
    np.divide(sums, support[:, None], out=observed, where=support[:, None] > 0)
    if method == 'zero_fill':
        replacement = np.zeros_like(sums)
    elif method == 'global_mean':
        replacement = np.broadcast_to(mu, visible.shape)
    elif method == 'paper_neighmean':
        replacement = np.zeros_like(sums)
        np.divide(sums, degree[:, None], out=replacement, where=degree[:, None] > 0)
    elif method == 'observed_mean':
        replacement = observed
    elif method == 'fixed_mix':
        replacement = alpha * observed + (1 - alpha) * mu
    else:
        replacement = (sums + tau * mu) / (support[:, None] + tau)
    out = visible.astype(np.float32, copy=True)
    out[missing] = replacement[missing]
    if not np.isfinite(out).all():
        raise ValueError('imputation output contains nonfinite values')
    hidden_support = support[missing]
    summary = {
        'method': method, 'n_items': n_items, 'missing_items': int(missing.sum()),
        'observed_items': int((~missing).sum()), 'zero_support_missing': int(np.sum(hidden_support == 0)),
        'mean_support_missing': float(hidden_support.mean()) if len(hidden_support) else None,
        'support_histogram_missing': {str(int(x)): int(y) for x, y in zip(*np.unique(hidden_support, return_counts=True))},
        'alpha': float(alpha), 'tau': float(tau),
    }
    return out, {'support_count': support, 'degree': degree, 'global_mean': mu.astype(np.float32), 'summary': summary}
