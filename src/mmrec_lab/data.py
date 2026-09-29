"""Auditable train-only subset preparation for the authors' Baby data."""
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .io import array_hash, canonical_hash, file_hash, read_json, write_json

FILES = ['baby.inter', 'image_feat.npy', 'text_feat.npy',
         'i_id_mapping.csv', 'u_id_mapping.csv']


def _integer_column(table, column):
    numeric = pd.to_numeric(table[column], errors='raise').to_numpy()
    if not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric != np.floor(numeric)).any():
        raise ValueError(f'{column} must contain nonnegative integer IDs')
    return numeric.astype(np.int64)


def _valid_rows(features):
    if features.ndim != 2 or features.shape[1] < 1 or features.dtype.kind not in 'fi':
        raise ValueError('features must be a numeric matrix')
    valid = np.ones(len(features), dtype=bool)
    for start in range(0, len(features), 256):
        block = np.asarray(features[start:start + 256])
        if not np.isfinite(block).all():
            raise ValueError('feature matrix must be finite')
        valid[start:start + len(block)] = np.any(block != 0, axis=1)
    return valid


def prepare_dataset(raw_dir, out_dir, target_items=2000, seed=20260929):
    start = time.perf_counter()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    if isinstance(target_items, bool) or int(target_items) != target_items or target_items < 4:
        raise ValueError('target_items must be an integer >=4')
    if (out_dir / 'manifest.json').exists():
        raise FileExistsError('Prepared output already exists; use a distinct directory or load it')
    raw_hashes = {name: file_hash(raw_dir / name) for name in FILES}
    table = pd.read_csv(raw_dir / 'baby.inter', sep='\t')
    for col in ('userID', 'itemID', 'x_label'):
        if col not in table:
            raise ValueError(f'Missing interaction field: {col}')
        table[col] = _integer_column(table, col)
    if len(table) == 0 or set(table.x_label.unique()) != {0, 1, 2}:
        raise ValueError('Expected nonempty train/validation/test labels 0/1/2')
    if table.duplicated(['userID', 'itemID']).any():
        raise ValueError('duplicate user-item pairs or cross-split overlap')
    vision = np.load(raw_dir / 'image_feat.npy', mmap_mode='r', allow_pickle=False)
    text = np.load(raw_dir / 'text_feat.npy', mmap_mode='r', allow_pickle=False)
    if len(vision) != len(text) or table.itemID.max() >= len(vision):
        raise ValueError('feature row mapping does not match item IDs')
    valid_rows = _valid_rows(vision) & _valid_rows(text)
    for filename, field, expected in [('i_id_mapping.csv', 'itemID', len(vision)),
                                     ('u_id_mapping.csv', 'userID', int(table.userID.max()) + 1)]:
        mapping = pd.read_csv(raw_dir / filename, sep='\t')
        if field not in mapping:
            raise ValueError(f'Invalid mapping schema: {filename}')
        ids = _integer_column(mapping, field)
        if len(np.unique(ids)) != len(ids) or not np.array_equal(np.sort(ids), np.arange(expected)):
            raise ValueError(f'Invalid mapping coverage: {filename}')
    train_full = table.loc[table.x_label == 0, ['userID', 'itemID']].to_numpy(np.int64)
    degrees = np.bincount(train_full[:, 1], minlength=len(vision))
    pool = np.flatnonzero(valid_rows & (degrees > 0))
    # Descending popularity, ascending original ID for ties.
    ranked = pool[np.lexsort((pool, -degrees[pool]))]
    strata = np.array_split(ranked, 4)
    target_items = min(int(target_items), len(pool))
    quotas = [target_items // 4 + (i < target_items % 4) for i in range(4)]
    rng = np.random.default_rng(seed)
    selected = np.sort(np.concatenate([rng.choice(s, min(q, len(s)), replace=False)
                                       for s, q in zip(strata, quotas)]))
    initial_selected = selected.copy()
    subset_train = train_full[np.isin(train_full[:, 1], selected)]
    rounds = []
    while True:
        users, counts = np.unique(subset_train[:, 0], return_counts=True)
        keep_users = users[counts >= 2]
        next_train = subset_train[np.isin(subset_train[:, 0], keep_users)]
        rounds.append({'users': len(keep_users), 'items': len(np.unique(next_train[:, 1])),
                       'train_interactions': len(next_train)})
        if len(next_train) == len(subset_train):
            break
        subset_train = next_train
    user_ids = np.unique(subset_train[:, 0])
    item_ids = np.unique(subset_train[:, 1])
    if len(user_ids) == 0 or len(item_ids) < 4:
        raise ValueError('Subset infeasible after train-only pruning')
    splits = {}
    for name, label in [('train', 0), ('valid', 1), ('test', 2)]:
        sub = table.loc[(table.x_label == label) & table.userID.isin(user_ids) & table.itemID.isin(item_ids),
                        ['userID', 'itemID']].to_numpy(np.int64)
        mapped = np.column_stack([np.searchsorted(user_ids, sub[:, 0]),
                                  np.searchsorted(item_ids, sub[:, 1])]).astype(np.int64)
        splits[name] = mapped[np.lexsort((mapped[:, 1], mapped[:, 0]))]
        if not len(mapped):
            raise ValueError(f'No eligible {name} interactions after fixed subset')
    degree = np.bincount(splits['train'][:, 1], minlength=len(item_ids))
    popularity_order = np.lexsort((item_ids, -degree))
    tail = np.zeros(len(item_ids), dtype=bool)
    tail[popularity_order[len(item_ids) // 2:]] = True
    if not tail[splits['test'][:, 1]].any() or not tail[splits['valid'][:, 1]].any():
        raise ValueError('No eligible tail evaluation positives; do not redraw subset')
    arrays = {**splits, 'vision': np.asarray(vision[item_ids], dtype=np.float32),
              'text': np.asarray(text[item_ids], dtype=np.float32),
              'user_ids': user_ids, 'item_ids': item_ids, 'tail': tail}
    for key in ('vision', 'text'):
        if not np.isfinite(arrays[key]).all():
            raise ValueError('float32 feature conversion is not finite')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'dataset.npz', **arrays)
    count_summary = {name: {'interactions': len(splits[name]),
                            'users': len(np.unique(splits[name][:, 0])),
                            'tail_positives': int(tail[splits[name][:, 1]].sum())}
                     for name in splits}
    manifest = {
        'schema_version': 1, 'dataset': 'author-preprocessed Amazon Baby',
        'native_image_availability_verified': False,
        'native_image_note': 'Author preprocessing may already mean-fill native missing images; native missing ID list is not supplied.',
        'split_labels': {'train': 0, 'valid': 1, 'test': 2},
        'subset_seed': seed, 'target_items': target_items,
        'n_users': len(user_ids), 'n_items': len(item_ids),
        'raw_files': {f: {'sha256': raw_hashes[f], 'bytes': (raw_dir / f).stat().st_size} for f in FILES},
        'raw_shape': {'vision': list(vision.shape), 'text': list(text.shape)},
        'raw_dtype': {'vision': str(vision.dtype), 'text': str(text.dtype)},
        'eligible_pool': len(pool), 'zero_feature_rows_excluded': int((~valid_rows).sum()),
        'initial_item_ids': initial_selected.tolist(),
        'pruned_item_ids': sorted(set(initial_selected.tolist()) - set(item_ids.tolist())),
        'pruning_rounds': rounds, 'splits': count_summary,
        'array_hashes': {key: array_hash(value) for key, value in arrays.items()},
        'dataset_sha256': file_hash(out_dir / 'dataset.npz'),
        'preparation_seconds': time.perf_counter() - start,
    }
    manifest['identity'] = canonical_hash({k: v for k, v in manifest.items() if k not in ['preparation_seconds']})
    write_json(out_dir / 'manifest.json', manifest)
    pd.DataFrame({'item': range(len(item_ids)), 'original_item_id': item_ids,
                  'train_degree': degree, 'tail': tail}).to_csv(out_dir / 'item_mapping.csv', index=False)
    pd.DataFrame({'user': range(len(user_ids)), 'original_user_id': user_ids}).to_csv(out_dir / 'user_mapping.csv', index=False)
    return manifest


def load_dataset(path):
    path = Path(path)
    manifest = read_json(path / 'manifest.json')
    if canonical_hash({k:v for k,v in manifest.items() if k not in ['identity', 'preparation_seconds']}) != manifest['identity']:
        raise ValueError('Prepared manifest identity mismatch')
    if file_hash(path / 'dataset.npz') != manifest['dataset_sha256']:
        raise ValueError('Prepared dataset checksum/hash mismatch')
    with np.load(path / 'dataset.npz', allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    for name, value in data.items():
        if array_hash(value) != manifest['array_hashes'][name]:
            raise ValueError(f'Prepared array hash mismatch: {name}')
    return {**data, 'n_users': manifest['n_users'], 'n_items': manifest['n_items'],
            'manifest': manifest}
