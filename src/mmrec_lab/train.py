"""CPU FREEDOM training with validation-only selection and epoch checkpoints."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import platform
import threading
import time

import numpy as np
import psutil
import scipy
import torch

from .io import array_hash, canonical_hash, file_hash, read_json, write_json
from .metrics import evaluate
from .model import DEFAULT_CONFIG, FreedomModel

TRAIN_DEFAULTS = {'epochs': 200, 'patience': 20, 'batch_size': 2048, 'lr': .001,
                  'num_threads': 2, 'eval_batch_size': 256}


def _pairs(data, name, n_users, n_items):
    a = np.asarray(data[name])
    if a.ndim != 2 or a.shape[1] != 2 or not np.issubdtype(a.dtype, np.integer) or not len(a):
        raise ValueError(f'{name} must be a nonempty Nx2 integer array')
    if np.any(a < 0) or np.any(a[:, 0] >= n_users) or np.any(a[:, 1] >= n_items):
        raise ValueError(f'{name} IDs out of range')
    if len(np.unique(a, axis=0)) != len(a):
        raise ValueError(f'{name} contains duplicate user/item pairs')
    return a


def _item_sets(pairs):
    result = {}
    for u, i in pairs:
        result.setdefault(int(u), set()).add(int(i))
    return result


def _save_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class _MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self.stop.wait(.05):
            self.sample()

    def sample(self):
        self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.sample()
        self.stop.set()
        self.thread.join()


class _RunStatus:
    def __init__(self, path, identity_payload):
        self.path, self.metadata = path, identity_payload

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is not None:
            # This context is entered only after the requested identity passed.
            # Avoid copying arbitrary exception text or input values to artifacts.
            status = 'interrupted' if issubclass(exception_type, KeyboardInterrupt) else 'failed'
            write_json(self.path, {**self.metadata, 'status': status,
                                  'error_type': exception_type.__name__,
                                  'reason': 'Training interrupted; resume last completed epoch' if status == 'interrupted'
                                            else 'Training failed; see exception at call site'})
        return False


def train_run(dataset: dict, vision: np.ndarray | None, text: np.ndarray | None,
              config: dict, out_dir: Path, seed: int, missing: np.ndarray | None = None,
              test: bool = True, resume: bool = True) -> dict:
    """Train/restore on CPU; test=False never constructs or evaluates test truth.

    Checkpoints are atomic at epoch boundaries. Interrupted partial epochs restart
    from the last completed epoch with restored optimizer and independent RNGs.
    Completed runs are reused only after identity and artifact checks succeed.
    """
    started = time.perf_counter()
    config = {**DEFAULT_CONFIG, **TRAIN_DEFAULTS, **config}
    for key in ('epochs', 'patience', 'batch_size', 'num_threads', 'eval_batch_size'):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, np.integer)) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not isinstance(config['lr'], (float, int, np.floating, np.integer)) or not np.isfinite(config['lr']) or config['lr'] <= 0:
        raise ValueError('lr must be finite and positive')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    n_users, n_items = int(dataset['n_users']), int(dataset['n_items'])
    if n_users < 1 or n_items < 1:
        raise ValueError('dataset vocabulary must be nonempty')
    train = _pairs(dataset, 'train', n_users, n_items)
    valid = _pairs(dataset, 'valid', n_users, n_items)
    train_sets, valid_sets = _item_sets(train), _item_sets(valid)
    if any(train_sets.get(u, set()) & items for u, items in valid_sets.items()):
        raise ValueError('training and validation interactions overlap')
    if any(len(items) == n_items for items in train_sets.values()):
        raise ValueError('training user has no available negative item')
    tail = np.asarray(dataset['tail'])
    missing = np.zeros(n_items, dtype=bool) if missing is None else np.asarray(missing)
    if tail.dtype != bool or tail.shape != (n_items,) or missing.dtype != bool or missing.shape != (n_items,):
        raise ValueError('tail and missing must be catalog-length boolean masks')
    groups = {'all': np.ones(n_items, bool), 'tail': tail, 'missing': missing, 'tail_missing': tail & missing}
    source_files = {name: file_hash(Path(__file__).with_name(name)) for name in ('train.py', 'model.py', 'metrics.py', 'io.py')}
    environment = {'python': platform.python_version(), 'torch': torch.__version__,
                   'numpy': np.__version__, 'scipy': scipy.__version__}
    inputs = {'train': array_hash(train), 'valid': array_hash(valid), 'test': array_hash(dataset['test']),
              'tail': array_hash(tail), 'missing': array_hash(missing),
              'vision': None if vision is None else array_hash(vision),
              'text': None if text is None else array_hash(text)}
    identity_payload = {'schema_version': 1, 'config': config, 'seed': seed, 'test_enabled': bool(test),
                        'n_users': n_users, 'n_items': n_items, 'inputs': inputs,
                        'dataset_manifest': dataset.get('manifest', {}),
                        'source_files': source_files, 'environment': environment}
    identity = canonical_hash(identity_payload)
    out_dir = Path(out_dir)
    metadata_path, result_path = out_dir / 'run_metadata.json', out_dir / 'result.json'
    if metadata_path.exists():
        if read_json(metadata_path)['identity'] != identity:
            raise ValueError('Run identity mismatch: use a distinct output directory')
        if not resume:
            raise FileExistsError('Run output exists and resume=False')
        if result_path.exists():
            completed = read_json(result_path)
            if completed['identity'] != identity:
                raise ValueError('Completed result identity mismatch')
            for name, digest in completed['artifacts'].items():
                if not (out_dir / name).is_file() or file_hash(out_dir / name) != digest:
                    raise ValueError(f'Completed artifact checksum mismatch: {name}')
            return completed
    elif out_dir.exists():
        allowed = {'environment.json', 'condition.json', 'missing.npy', 'support.npy'}
        if any(not p.is_file() or p.name not in allowed for p in out_dir.iterdir()):
            raise FileExistsError('Nonempty output directory has no valid run identity')
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, {'identity': identity, **identity_payload, 'status': 'running'})
    torch.set_num_threads(config['num_threads'])
    init_seed, neg_seed, graph_seed, shuffle_seed = [int(x) for x in np.random.SeedSequence(seed).generate_state(4)]
    negative_rng = np.random.default_rng(neg_seed)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    graph_rng = torch.Generator(device='cpu').manual_seed(graph_seed)
    history, best_epoch, best_score, patience_count, epoch_start = [], 0, float('-inf'), 0, 1
    best_state = None
    accumulated = {'train_seconds': 0., 'eval_seconds': 0., 'checkpoint_seconds': 0., 'total_seconds': 0.}

    with _RunStatus(metadata_path, {'identity': identity, **identity_payload}), _MemoryMonitor() as memory, torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        init_started = time.perf_counter()
        model = FreedomModel(n_users, n_items, train, vision, text, config, seed=init_seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        init_seconds = time.perf_counter() - init_started
        checkpoint_path = out_dir / 'resume.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if checkpoint['identity'] != identity:
                raise ValueError('Checkpoint identity mismatch')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            negative_rng.bit_generator.state = checkpoint['negative_rng']
            shuffle_rng.bit_generator.state = checkpoint['shuffle_rng']
            graph_rng.set_state(checkpoint['graph_rng'])
            torch.random.set_rng_state(checkpoint['torch_rng'])
            history = checkpoint['history']
            best_epoch, best_score, patience_count = checkpoint['best_epoch'], checkpoint['best_score'], checkpoint['patience_count']
            epoch_start = checkpoint['epoch'] + 1
            accumulated = checkpoint['resources']
            memory.peak = max(memory.peak, checkpoint['peak_rss_bytes'])
            best_state = checkpoint['best_state']
            # resume.pt is the authoritative atomic epoch snapshot, including
            # the best state; best.pt alone might be ahead after a write crash.
            _save_torch(out_dir / 'best.pt', {'identity': identity, 'epoch': best_epoch, 'state_dict': best_state})
            resource_path = out_dir / 'epoch_resources.json'
            if resource_path.exists():
                epoch_resources = read_json(resource_path)
                if epoch_resources.get('identity') == identity and epoch_resources.get('epoch') == checkpoint['epoch']:
                    accumulated = epoch_resources['resources']
                    memory.peak = max(memory.peak, epoch_resources['peak_rss_bytes'])
        previous_seconds = accumulated['total_seconds']

        def assess(truth, blocked):
            begin = time.perf_counter()
            users = np.array(sorted(truth), dtype=np.int64)
            model.eval()
            with torch.no_grad():
                user_embedding, item_embedding = model.forward(model.norm_adj)
                blocks = []
                for start in range(0, len(users), config['eval_batch_size']):
                    batch = torch.from_numpy(users[start:start + config['eval_batch_size']])
                    blocks.append((user_embedding[batch] @ item_embedding.T).numpy())
                scores = np.concatenate(blocks, axis=0)
            result = evaluate(scores, users, truth, blocked, groups, k=20)
            accumulated['eval_seconds'] += time.perf_counter() - begin
            return result

        def negatives(users):
            samples = negative_rng.integers(0, n_items, size=len(users), dtype=np.int64)
            for row, user in enumerate(users):
                occupied = train_sets[int(user)]
                while int(samples[row]) in occupied:
                    samples[row] = negative_rng.integers(n_items)
            return samples

        for epoch in range(epoch_start, config['epochs'] + 1):
            if patience_count >= config['patience']:
                break
            epoch_started = time.perf_counter()
            model.train()
            model.sample_epoch_graph(graph_rng)
            ordered = train[shuffle_rng.permutation(len(train))]
            loss_sum = 0.
            for start in range(0, len(ordered), config['batch_size']):
                batch = ordered[start:start + config['batch_size']]
                neg = negatives(batch[:, 0])
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(torch.from_numpy(batch[:, 0].copy()), torch.from_numpy(batch[:, 1].copy()), torch.from_numpy(neg))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite training gradients')
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch)
            training_seconds = time.perf_counter() - epoch_started
            accumulated['train_seconds'] += training_seconds
            validation = assess(valid_sets, train_sets)
            score = validation['summary']['all']['recall']
            if score is None:
                raise ValueError('Validation has no evaluable users')
            improved = score > best_score
            if improved:
                best_score, best_epoch, patience_count = score, epoch, 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                patience_count += 1
            history.append({'epoch': epoch, 'loss': loss_sum / len(train),
                            'validation_recall': score, 'validation_ndcg': validation['summary']['all']['ndcg'],
                            'train_seconds': training_seconds, 'best': improved})
            save_started = time.perf_counter()
            if improved:
                _save_torch(out_dir / 'best.pt', {'identity': identity, 'epoch': epoch, 'state_dict': best_state})
            memory.sample()
            accumulated['total_seconds'] = previous_seconds + time.perf_counter() - started
            _save_torch(checkpoint_path, {'identity': identity, 'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'negative_rng': negative_rng.bit_generator.state,
                        'shuffle_rng': shuffle_rng.bit_generator.state, 'graph_rng': graph_rng.get_state(),
                        'torch_rng': torch.random.get_rng_state(),
                        'history': history, 'best_epoch': best_epoch, 'best_score': best_score,
                        'best_state': best_state,
                        'patience_count': patience_count, 'resources': accumulated,
                        'peak_rss_bytes': memory.peak})
            _write_csv(out_dir / 'history.csv', history, list(history[0]))
            accumulated['checkpoint_seconds'] += time.perf_counter() - save_started
            accumulated['total_seconds'] = previous_seconds + time.perf_counter() - started
            write_json(out_dir / 'epoch_resources.json', {'identity': identity, 'epoch': epoch,
                       'resources': accumulated, 'peak_rss_bytes': memory.peak})
        best = torch.load(out_dir / 'best.pt', map_location='cpu', weights_only=False)
        if best['identity'] != identity:
            raise ValueError('Best checkpoint identity mismatch')
        model.load_state_dict(best['state_dict'])
        validation = assess(valid_sets, train_sets)
        testing = None
        if test:
            test_pairs = _pairs(dataset, 'test', n_users, n_items)
            test_sets = _item_sets(test_pairs)
            blocked = {u: train_sets.get(u, set()) | valid_sets.get(u, set()) for u in set(train_sets) | set(valid_sets)}
            testing = assess(test_sets, blocked)
            write_json(out_dir / 'rankings.json', testing['rankings'])
            _write_csv(out_dir / 'per_user_metrics.csv', testing['rows'], ['user', 'group', 'recall', 'ndcg', 'positives'])
        memory.sample()
        resources = {**accumulated, 'init_seconds': init_seconds, 'peak_rss_bytes': memory.peak,
                     'threads': config['num_threads'], 'device': 'cpu',
                     'total_seconds': previous_seconds + time.perf_counter() - started,
                     'rss_sampling_interval_seconds': .05}
        stopped = patience_count >= config['patience']
        result = {'identity': identity, 'config': config, 'seed': seed, 'inputs': inputs,
                  'source_files': source_files, 'environment': environment,
                  'stream_seeds': {'initialization': init_seed, 'negative': neg_seed, 'graph': graph_seed, 'shuffle': shuffle_seed},
                  'best_epoch': best_epoch, 'best_validation_recall': best_score,
                  'epochs_completed': len(history), 'stop_reason': 'validation_patience' if stopped else 'epoch_limit',
                  'is_converged': stopped, 'history': history, 'resources': resources,
                  'validation': validation, 'test': testing}
    names = ['best.pt', 'resume.pt', 'history.csv', 'epoch_resources.json'] + (['rankings.json', 'per_user_metrics.csv'] if test else [])
    names += [name for name in ('environment.json', 'condition.json', 'missing.npy', 'support.npy') if (out_dir / name).is_file()]
    result['artifacts'] = {name: file_hash(out_dir / name) for name in names}
    write_json(result_path, result)
    write_json(metadata_path, {'identity': identity, **identity_payload, 'status': 'completed'})
    return result
