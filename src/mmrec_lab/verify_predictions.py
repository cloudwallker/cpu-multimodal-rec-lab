"""Independent checkpoint-to-prediction evidence, without retraining or changing selection."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.sparse import load_npz
import torch

from .data import load_dataset
from .impute import build_neighbors, impute_features, make_missing
from .io import array_hash, canonical_hash, file_hash, read_json, write_json
from .metrics import evaluate
from .model import FreedomModel
from .study import experiment_matrix, source_manifest


def _sets(pairs):
    mapping = {}
    for user, item in pairs:
        mapping.setdefault(int(user), set()).add(int(item))
    return mapping


def _compare_metrics(actual, saved, run_id):
    if set(actual) != set(saved):
        raise ValueError(f'{run_id}: prediction metric fields differ')
    if actual['rankings'] != saved['rankings']:
        raise ValueError(f'{run_id}: checkpoint prediction rankings differ')
    if set(actual['summary']) != set(saved['summary']):
        raise ValueError(f'{run_id}: prediction metric groups differ')
    for group, row in actual['summary'].items():
        stored = saved['summary'][group]
        if set(stored) != set(row):
            raise ValueError(f'{run_id}: prediction summary fields differ')
        for key, value in row.items():
            other = stored[key]
            if value is None or key in ('users', 'positives'):
                match = value == other
            else:
                match = other is not None and np.isfinite(other) and np.isclose(value, other, rtol=1e-10, atol=1e-12)
            if not match:
                raise ValueError(f'{run_id}: prediction metric mismatch {group}/{key}')
    # Same exact rankings and deterministic metric implementation also imply exact rows.
    if actual['rows'] != saved['rows']:
        raise ValueError(f'{run_id}: prediction per-user metrics differ')


def verify_predictions(dataset_dir, study_dir, output_dir) -> dict:
    """Require all 45 completed runs, reload CPU best parameters and recompute test scores.

    The output marker is pessimistic until every run succeeds. Failures overwrite any
    previous success with success=False and propagate; no run is skipped or trained.
    """
    started = time.perf_counter()
    study, output = Path(study_dir), Path(output_dir)
    marker = output / 'prediction_verification.json'
    record = dict(schema_version=1, success=False, verified_runs=0, planned_runs=45, runs=[],
                  verification_source_sha256=file_hash(__file__))
    write_json(marker, record)
    run_id = None
    old_threads = torch.get_num_threads()
    try:
        record['source_files'] = source_manifest()
        dataset = load_dataset(dataset_dir)
        protocol = read_json(study / 'frozen_protocol.json')
        if protocol['identity'] != canonical_hash({k: v for k, v in protocol.items() if k != 'identity'}):
            raise ValueError('Frozen protocol identity mismatch')
        if protocol['dataset_identity'] != dataset['manifest']['identity']:
            raise ValueError('Prediction verification data differs from frozen protocol')
        if protocol['source_files'] != record['source_files']:
            raise ValueError('Prediction verification source differs from frozen protocol')
        planned = experiment_matrix()
        if protocol['runs'] != planned or len(planned) != 45:
            raise ValueError('Prediction verification requires the complete predeclared 45-run matrix')
        status = read_json(study / 'study_status.json')
        identities = {r['run_id']: r['result_identity'] for r in status['runs']}
        if (status.get('status') != 'complete' or status['completed'] != 45 or status['planned'] != 45
                or len(status['runs']) != 45 or set(identities) != {r['run_id'] for r in planned}
                or status['protocol_identity'] != protocol['identity']):
            raise ValueError('Prediction verification requires all 45 completed runs')
        record.update(protocol_identity=protocol['identity'], dataset_identity=protocol['dataset_identity'],
                      protocol_sha256=file_hash(study / 'frozen_protocol.json'))
        n_users, n_items = dataset['n_users'], dataset['n_items']
        neighbors = build_neighbors(dataset['train'], n_users, n_items, 20)
        saved_neighbors = load_npz(study / 'neighbors.npz').tocsr()
        neighbor_meta = read_json(study / 'neighbors.json')
        if (saved_neighbors.shape != neighbors.shape or (saved_neighbors != neighbors).nnz
                or neighbor_meta['sha256'] != file_hash(study / 'neighbors.npz')
                or neighbor_meta['train_hash'] != array_hash(dataset['train'])):
            raise ValueError('Saved neighbor graph differs from train-only reconstruction')
        neighbor_hash = canonical_hash([array_hash(neighbors.indptr), array_hash(neighbors.indices), array_hash(neighbors.data)])
        train_sets, valid_sets, truth = [_sets(dataset[key]) for key in ['train', 'valid', 'test']]
        users = np.array(sorted(truth), np.int64)
        blocked = {u: train_sets.get(u, set()) | valid_sets.get(u, set()) for u in truth}
        environment = dict(python=platform.python_version(), torch=torch.__version__,
                           numpy=np.__version__, scipy=scipy.__version__)
        record['environment'] = environment
        data_hashes = {key: array_hash(dataset[key]) for key in ['train', 'valid', 'test', 'tail']}
        for condition in planned:
            run_id = condition['run_id']
            print(f'PREDICTION VERIFY {len(record["runs"])+1}/45 {run_id}', flush=True)
            run_started = time.perf_counter()
            path = study / run_id
            result = read_json(path / 'result.json')
            metadata = read_json(path / 'run_metadata.json')
            saved_condition = read_json(path / 'condition.json')
            if any(saved_condition.get(key) != value for key, value in condition.items()):
                raise ValueError(f'{run_id}: frozen condition mismatch')
            expected_alpha = protocol['alpha'] if condition['method'] == 'fixed_mix' else None
            expected_tau = protocol['tau'] if condition['method'] == 'support_mix' else None
            if saved_condition.get('alpha') != expected_alpha or saved_condition.get('tau') != expected_tau:
                raise ValueError(f'{run_id}: frozen imputation parameter mismatch')
            meta_identity = canonical_hash({k: v for k, v in metadata.items() if k not in ['identity', 'status']})
            if (metadata.get('status') != 'completed' or metadata['identity'] != meta_identity
                    or metadata['identity'] != result['identity'] or result['identity'] != identities[run_id]
                    or metadata.get('test_enabled') is not True or result.get('test') is None):
                raise ValueError(f'{run_id}: completed result identity/test setting mismatch')
            if (metadata['dataset_manifest'] != dataset['manifest'] or metadata['n_users'] != n_users
                    or metadata['n_items'] != n_items or result['config'] != protocol['config']
                    or metadata['config'] != result['config'] or metadata['seed'] != condition['seed']
                    or result['seed'] != condition['seed']):
                raise ValueError(f'{run_id}: frozen data/configuration/seed mismatch')
            for field in ['inputs', 'source_files', 'environment']:
                if metadata[field] != result[field]:
                    raise ValueError(f'{run_id}: metadata {field} mismatch')
            expected_sources = {name: record['source_files'][name] for name in ['train.py', 'model.py', 'metrics.py', 'io.py']}
            if result['source_files'] != expected_sources or result['environment'] != environment:
                raise ValueError(f'{run_id}: prediction source/environment mismatch')
            required_artifacts = {'best.pt', 'rankings.json', 'condition.json', 'missing.npy', 'support.npy'}
            if not required_artifacts.issubset(result['artifacts']):
                raise ValueError(f'{run_id}: required completed artifacts are missing from checksums')
            for name, digest in result['artifacts'].items():
                if Path(name).name != name or not (path / name).is_file() or file_hash(path / name) != digest:
                    raise ValueError(f'{run_id}: completed artifact checksum mismatch {name}')
            missing = make_missing(n_items, dataset['tail'], condition['scenario'], condition['mask_seed'] or 0)
            vision, text = dataset['vision'].copy(), dataset['text']
            support = np.zeros(n_items, np.int64)
            if condition['scenario'] != 'clean':
                vision[missing] = 0
                vision, info = impute_features(vision, missing, neighbors, condition['method'],
                                               alpha=protocol['alpha'], tau=protocol['tau'])
                support = info['support_count']
            if condition['modality'] == 'image':
                text = None
            elif condition['modality'] == 'text':
                vision = None
            inputs = {**data_hashes, 'missing': array_hash(missing),
                      'vision': None if vision is None else array_hash(vision),
                      'text': None if text is None else array_hash(text)}
            if result['inputs'] != inputs:
                raise ValueError(f'{run_id}: reconstructed model input hashes differ')
            if (saved_condition['missing_hash'] != inputs['missing'] or saved_condition['vision_hash'] != inputs['vision']
                    or saved_condition['text_hash'] != inputs['text'] or saved_condition['support_hash'] != array_hash(support)
                    or saved_condition['neighbor_hash'] != neighbor_hash
                    or not np.array_equal(np.load(path / 'missing.npy', allow_pickle=False), missing)
                    or not np.array_equal(np.load(path / 'support.npy', allow_pickle=False), support)):
                raise ValueError(f'{run_id}: reconstructed condition/mask/support differs')
            seeds = [int(value) for value in np.random.SeedSequence(condition['seed']).generate_state(4)]
            expected_streams = dict(zip(['initialization', 'negative', 'graph', 'shuffle'], seeds))
            if result['stream_seeds'] != expected_streams:
                raise ValueError(f'{run_id}: recorded random streams differ from training seed')
            best_hash = file_hash(path / 'best.pt')
            checkpoint = torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
            if checkpoint['identity'] != result['identity'] or checkpoint['epoch'] != result['best_epoch']:
                raise ValueError(f'{run_id}: best checkpoint identity/epoch mismatch')
            torch.set_num_threads(result['config']['num_threads'])
            prediction_started = time.perf_counter()
            model = FreedomModel(n_users, n_items, dataset['train'], vision, text,
                                 result['config'], seed=result['stream_seeds']['initialization'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval()
            with torch.no_grad():
                user_embedding, item_embedding = model.forward(model.norm_adj)
                blocks = []
                for offset in range(0, len(users), result['config']['eval_batch_size']):
                    batch = torch.from_numpy(users[offset:offset + result['config']['eval_batch_size']])
                    blocks.append((user_embedding[batch] @ item_embedding.T).numpy())
                scores = np.concatenate(blocks, axis=0)
            groups = dict(all=np.ones(n_items, bool), tail=dataset['tail'], missing=missing,
                          tail_missing=dataset['tail'] & missing)
            actual = evaluate(scores, users, truth, blocked, groups, k=20)
            if actual['rankings'] != read_json(path / 'rankings.json'):
                raise ValueError(f'{run_id}: checkpoint prediction rankings differ from saved Top20')
            _compare_metrics(actual, result['test'], run_id)
            record['runs'].append(dict(run_id=run_id, success=True, result_identity=result['identity'],
                best_epoch=checkpoint['epoch'], best_sha256=best_hash, inputs=inputs, source_files=result['source_files'],
                result_sha256=file_hash(path / 'result.json'), rankings_sha256=file_hash(path / 'rankings.json'),
                prediction_seconds=time.perf_counter() - prediction_started,
                verification_seconds=time.perf_counter() - run_started))
            record['verified_runs'] = len(record['runs'])
            del model, checkpoint, scores, blocks, user_embedding, item_embedding, actual, vision, text
            gc.collect()
        record.update(success=True, total_seconds=time.perf_counter() - started)
        write_json(marker, record)
        return record
    except BaseException as error:
        record.update(success=False, failed_run=run_id, error_type=type(error).__name__,
                      error=str(error), total_seconds=time.perf_counter() - started)
        write_json(marker, record)
        raise
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description='从全部45个最佳CPU检查点重新预测并核验测试Top20与指标')
    parser.add_argument('--data', required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = verify_predictions(args.data, args.study, args.output)
    print(f'Checkpoint prediction verification: success={result["success"]}, runs={result["verified_runs"]}/45')


if __name__ == '__main__':
    main()
