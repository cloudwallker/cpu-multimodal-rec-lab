"""Predeclared development search and paired 45-run CPU study."""
from pathlib import Path
import gc
import time

import numpy as np
from scipy.sparse import save_npz

from .data import load_dataset
from .impute import build_neighbors, make_missing, impute_features
from .io import array_hash, canonical_hash, file_hash, read_json, write_json

METHODS = ['zero_fill', 'global_mean', 'paper_neighmean', 'observed_mean', 'fixed_mix', 'support_mix']
SCENARIOS = ['uniform30', 'tail30']
BASE_CONFIG = {'embedding_size': 64, 'feat_embed_dim': 64, 'knn_k': 10,
               'n_mm_layers': 1, 'n_ui_layers': 2, 'mm_image_weight': 0.1,
               'dropout': 0.8, 'reg_weight': 1e-4, 'epochs': 200, 'patience': 20,
               'batch_size': 2048, 'lr': .001, 'num_threads': 2, 'eval_batch_size': 256}


def experiment_matrix():
    runs = []
    for seed, mask in [(201, 301), (202, 302), (203, 303)]:
        for modality in ['both', 'image', 'text']:
            runs.append({'run_id': f'clean_{modality}_s{seed}', 'scenario': 'clean',
                         'modality': modality, 'method': 'clean', 'seed': seed, 'mask_seed': None})
        for scenario in SCENARIOS:
            for method in METHODS:
                runs.append({'run_id': f'{scenario}_{method}_s{seed}', 'scenario': scenario,
                             'modality': 'both', 'method': method, 'seed': seed, 'mask_seed': mask})
    return runs


def select_parameter(rows):
    if not rows:
        raise ValueError('Empty development search')
    candidates = []
    for value in sorted({r['value'] for r in rows}):
        pair = [r for r in rows if r['value'] == value]
        if len(pair) != 2 or {r['scenario'] for r in pair} != set(SCENARIOS):
            raise ValueError('Each candidate requires both development scenarios exactly once')
        scores = [r['recall'] for r in pair]
        if any(v is None or not np.isfinite(v) for v in scores):
            raise ValueError('Missing or invalid development score')
        candidates.append({'value': value, 'mean_validation_recall': float(np.mean(scores))})
    # Numerical differences below 1e-12 are ties, resolved by the smaller value.
    best_score = max(r['mean_validation_recall'] for r in candidates)
    selected = min((r for r in candidates if abs(r['mean_validation_recall'] - best_score) <= 1e-12),
                   key=lambda r: r['value'])
    return {**selected, 'candidates': candidates}


def source_manifest():
    root = Path(__file__).parent
    names = ['data.py', 'impute.py', 'metrics.py', 'model.py', 'train.py', 'study.py', 'io.py']
    return {name: file_hash(root / name) for name in names}


def prepare_condition(dataset, neighbors, condition, out_dir, alpha=.5, tau=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    vision, text = dataset['vision'].copy(), dataset['text']
    if condition['scenario'] == 'clean':
        missing = np.zeros(dataset['n_items'], dtype=bool)
        info = {'support_count': np.zeros(dataset['n_items'], dtype=int), 'summary': {}}
    else:
        missing = make_missing(dataset['n_items'], dataset['tail'], condition['scenario'], condition['mask_seed'])
        vision[missing] = 0
        vision, info = impute_features(vision, missing, neighbors, condition['method'], alpha=alpha, tau=tau)
    if condition['modality'] == 'image':
        text = None
    elif condition['modality'] == 'text':
        vision = None
    meta = {**condition, 'alpha': alpha if condition['method'] == 'fixed_mix' else None,
            'tau': tau if condition['method'] == 'support_mix' else None,
            'missing_hash': array_hash(missing), 'vision_hash': array_hash(vision) if vision is not None else None,
            'support_hash': array_hash(info['support_count']),
            'neighbor_hash': canonical_hash([array_hash(neighbors.indptr), array_hash(neighbors.indices), array_hash(neighbors.data)]),
            'text_hash': array_hash(text) if text is not None else None,
            'missing_rate': float(missing.mean()),
            'tail_missing_rate': float(missing[dataset['tail']].mean()),
            'imputation_seconds': time.perf_counter() - start, 'support_summary': info['summary']}
    if (out_dir / 'condition.json').exists():
        old = read_json(out_dir / 'condition.json')
        comparable = lambda x: {k:v for k,v in x.items() if k != 'imputation_seconds'}
        if comparable(old) != comparable(meta):
            raise ValueError('Saved condition mismatch; choose another run directory')
        if array_hash(np.load(out_dir / 'missing.npy', allow_pickle=False)) != meta['missing_hash'] or array_hash(np.load(out_dir / 'support.npy', allow_pickle=False)) != meta['support_hash']:
            raise ValueError('Saved condition arrays are corrupted')
    else:
        np.save(out_dir / 'missing.npy', missing)
        np.save(out_dir / 'support.npy', info['support_count'])
        write_json(out_dir / 'condition.json', meta)
    return vision, text, missing


def _neighbors(dataset, out_dir):
    start = time.perf_counter()
    neighbors = build_neighbors(dataset['train'], dataset['n_users'], dataset['n_items'], 20)
    path = Path(out_dir) / 'neighbors.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, neighbors)
    write_json(path.with_suffix('.json'), {'sha256': file_hash(path), 'train_hash': array_hash(dataset['train']),
               'n_edges': neighbors.nnz, 'k': 20, 'seconds': time.perf_counter() - start})
    return neighbors


def develop(dataset_dir, output_dir, config=None):
    from .train import train_run
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_dir)
    config = {**BASE_CONFIG, **(config or {})}
    identity = {'dataset': dataset['manifest']['identity'], 'config': config,
                'source_files': source_manifest()}
    meta_path = out / 'development_identity.json'
    if meta_path.exists() and read_json(meta_path) != identity:
        raise ValueError('Development identity mismatch; use a new output directory')
    write_json(meta_path, identity)
    backbone = []
    for dropout in [.8, .9]:
        for reg in [1e-4, 1e-3]:
            cfg = {**config, 'dropout': dropout, 'reg_weight': reg}
            run_id = f'backbone_d{dropout}_r{reg}'
            print(f'DEVELOP {run_id}', flush=True)
            result = train_run(dataset, dataset['vision'], dataset['text'], cfg, out / run_id, 101, test=False)
            backbone.append({'dropout': dropout, 'reg_weight': reg,
                             'recall': result['best_validation_recall'], 'run_id': run_id})
            gc.collect()
    best = min(backbone, key=lambda r: (-round(r['recall'], 12), r['dropout'], r['reg_weight']))
    common = {**config, 'dropout': best['dropout'], 'reg_weight': best['reg_weight']}
    write_json(out / 'backbone_selection.json', {'candidates': backbone, 'selected': best})
    neighbors = _neighbors(dataset, out)
    selections = {}
    for method, parameter, values in [('fixed_mix', 'alpha', [.25, .5, .75]), ('support_mix', 'tau', [1, 4, 16])]:
        candidates = []
        for value in values:
            for scenario in SCENARIOS:
                run_id = f'{method}_{parameter}{value}_{scenario}'
                condition = {'run_id': run_id, 'method': method, 'scenario': scenario,
                             'modality': 'both', 'seed': 101, 'mask_seed': 101}
                print(f'DEVELOP {run_id}', flush=True)
                vision, text, missing = prepare_condition(dataset, neighbors, condition, out / run_id, **{parameter: value})
                result = train_run(dataset, vision, text, common, out / run_id, 101, missing, test=False)
                candidates.append({'value': value, 'scenario': scenario,
                                   'recall': result['best_validation_recall'], 'run_id': run_id})
                gc.collect()
        selections[parameter] = {**select_parameter(candidates), 'runs': candidates}
        write_json(out / f'{parameter}_selection.json', selections[parameter])
    protocol = {'dataset_identity': dataset['manifest']['identity'], 'config': common,
                'alpha': selections['alpha']['value'], 'tau': selections['tau']['value'],
                'source_files': source_manifest(), 'runs': experiment_matrix(),
                'development_seed': 101, 'development_mask_seed': 101,
                'selection_rule': 'mean validation all Recall@20 across uniform30/tail30; ties smaller value',
                'test_opened': False}
    protocol['identity'] = canonical_hash(protocol)
    write_json(out / 'frozen_protocol.json', protocol)
    return protocol


def run_study(dataset_dir, protocol_path, output_dir):
    from .train import train_run
    dataset = load_dataset(dataset_dir)
    protocol = read_json(protocol_path)
    if protocol['identity'] != canonical_hash({k:v for k,v in protocol.items() if k != 'identity'}):
        raise ValueError('Frozen protocol checksum mismatch')
    if protocol['dataset_identity'] != dataset['manifest']['identity'] or protocol['source_files'] != source_manifest():
        raise ValueError('Data or research source changed after protocol freeze')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'frozen_protocol.json').exists() and read_json(out / 'frozen_protocol.json') != protocol:
        raise ValueError('Output directory belongs to another protocol')
    write_json(out / 'frozen_protocol.json', protocol)
    neighbors = _neighbors(dataset, out)
    results = []
    for index, condition in enumerate(protocol['runs'], 1):
        run_dir = out / condition['run_id']
        print(f'STUDY {index}/{len(protocol["runs"])} {condition["run_id"]}', flush=True)
        vision, text, missing = prepare_condition(dataset, neighbors, condition, run_dir,
                                                 protocol['alpha'], protocol['tau'])
        try:
            result = train_run(dataset, vision, text, protocol['config'], run_dir, condition['seed'], missing, test=True)
        except BaseException as exc:
            write_json(out / 'study_status.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                'completed': len(results), 'planned': len(protocol['runs']), 'runs': results,
                'failed_run': condition['run_id'], 'error_type': type(exc).__name__, 'protocol_identity': protocol['identity']})
            raise
        results.append({'run_id': condition['run_id'], 'result_identity': result['identity']})
        write_json(out / 'study_status.json', {'status': 'complete' if len(results) == len(protocol['runs']) else 'running',
                                             'completed': len(results), 'planned': len(protocol['runs']),
                                             'protocol_identity': protocol['identity'], 'runs': results})
        gc.collect()
    return results
