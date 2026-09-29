"""Full-catalog ranking metrics, with group positives and unchanged rank positions."""
from __future__ import annotations

import numpy as np


def _int(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (1 if positive else 0):
        raise ValueError(f'{name} must be a {"positive" if positive else "nonnegative"} integer')
    return int(value)


def _groups(groups):
    if not isinstance(groups, dict) or not groups:
        raise ValueError('groups must be a nonempty mapping')
    masks = {name: np.asarray(mask) for name, mask in groups.items()}
    first = next(iter(masks.values()))
    if first.ndim != 1 or not len(first):
        raise ValueError('groups must contain nonempty one-dimensional masks')
    n_items = len(first)
    if any(not isinstance(name, str) or mask.dtype != np.bool_ or mask.shape != (n_items,) for name, mask in masks.items()):
        raise ValueError('group masks must be boolean and match the catalog length')
    return masks, n_items


def _sets(mapping, n_items, name):
    if not isinstance(mapping, dict):
        raise ValueError(f'{name} must map users to item sets')
    validated = {}
    for user, items in mapping.items():
        user = _int(user, 'user')
        if not isinstance(items, (set, frozenset)):
            raise ValueError(f'{name} values must be sets')
        targets = {_int(item, 'item') for item in items}
        if any(item >= n_items for item in targets):
            raise ValueError(f'{name} item out of catalog range')
        validated[user] = targets
    return validated


def evaluate(scores: np.ndarray, users: np.ndarray, truth: dict[int, set[int]],
             blocked: dict[int, set[int]], groups: dict[str, np.ndarray], k: int = 20) -> dict:
    k = _int(k, 'k', positive=True)
    masks, n_items = _groups(groups)
    scores = np.asarray(scores)
    users = np.asarray(users)
    if scores.ndim != 2 or scores.shape[1] != n_items or not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores):
        raise ValueError('scores must be a real U-by-I matrix matching group masks')
    if not np.isfinite(scores).all():
        raise ValueError('scores contain nonfinite values')
    scores = scores.astype(np.float64, copy=False)
    if users.shape != (scores.shape[0],) or not np.issubdtype(users.dtype, np.integer) or np.any(users < 0):
        raise ValueError('users must be a nonnegative integer vector matching score rows')
    if len(np.unique(users)) != len(users):
        raise ValueError('duplicate users')
    truth = _sets(truth, n_items, 'truth')
    blocked = _sets(blocked, n_items, 'blocked')
    for user in truth.keys() | blocked.keys():
        if truth.get(user, set()) & blocked.get(user, set()):
            raise ValueError('truth and blocked overlap')
    catalog = np.arange(n_items)
    rankings = []
    for row, user in zip(scores, users):
        user = int(user)
        excluded = blocked.get(user, set())
        candidates = catalog[~np.isin(catalog, list(excluded))]
        order = np.lexsort((candidates, -row[candidates]))
        rankings.append({'user': user, 'items': candidates[order[:k]].tolist()})
    return evaluate_rankings(rankings, truth, masks, k)


def evaluate_rankings(rankings: list[dict], truth: dict[int, set[int]],
                      groups: dict[str, np.ndarray], k: int = 20) -> dict:
    """Recompute from saved Top-K lists; caller remains responsible for blocking."""
    k = _int(k, 'k', positive=True)
    masks, n_items = _groups(groups)
    truth = _sets(truth, n_items, 'truth')
    if not isinstance(rankings, (list, tuple)):
        raise ValueError('rankings must be a list of user/item records')
    clean_rankings, seen = [], set()
    for record in rankings:
        if not isinstance(record, dict) or 'user' not in record or 'items' not in record:
            raise ValueError('invalid ranking record')
        user = _int(record['user'], 'user')
        if user in seen:
            raise ValueError('duplicate ranking users')
        seen.add(user)
        if not isinstance(record['items'], (list, tuple, np.ndarray)):
            raise ValueError('ranking items must be a sequence')
        items = [_int(x, 'item') for x in record['items']]
        if len(items) != len(set(items)) or any(item >= n_items for item in items):
            raise ValueError('ranking items duplicated or out of range')
        clean_rankings.append({'user': user, 'items': items[:k]})
    rows = []
    for record in clean_rankings:
        user, items = record['user'], record['items']
        positives = truth.get(user, set())
        for name, mask in masks.items():
            group_truth = {item for item in positives if mask[item]}
            count = len(group_truth)
            if not count:
                continue
            hits = np.array([item in group_truth for item in items], dtype=np.float64)
            discounts = 1 / np.log2(np.arange(len(items)) + 2)
            ideal = float((1 / np.log2(np.arange(min(k, count)) + 2)).sum())
            rows.append({'user': user, 'group': name, 'recall': float(hits.sum() / count),
                         'ndcg': float(np.dot(hits, discounts) / ideal), 'positives': count})
    summary = {}
    for name in masks:
        group_rows = [row for row in rows if row['group'] == name]
        summary[name] = {
            'recall': float(np.mean([r['recall'] for r in group_rows])) if group_rows else None,
            'ndcg': float(np.mean([r['ndcg'] for r in group_rows])) if group_rows else None,
            'users': len(group_rows), 'positives': sum(r['positives'] for r in group_rows),
        }
    return {'summary': summary, 'rows': rows, 'rankings': clean_rankings}
