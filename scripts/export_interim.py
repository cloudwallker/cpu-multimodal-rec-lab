"""Export verified development evidence only; never train or evaluate test scores."""
from pathlib import Path
import csv
import gc
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from mmrec_lab.data import load_dataset
from mmrec_lab.impute import build_neighbors, impute_features, make_missing
from mmrec_lab.io import array_hash, canonical_hash, file_hash, read_json, write_json
from mmrec_lab.metrics import evaluate
from mmrec_lab.model import FreedomModel
from mmrec_lab.study import source_manifest
from mmrec_lab.verify_predictions import _compare_metrics, _sets


def main():
    root = Path(__file__).resolve().parents[1]
    dev, output = root / 'results/development', root / 'results/interim'
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'verification.json'
    evidence = dict(success=False, scope='completed development runs only', training_performed=False,
                    test_evaluated=False, script_sha256=file_hash(__file__), runs=[])
    write_json(marker, evidence)
    started = time.perf_counter()
    try:
        dataset = load_dataset(root / 'data/processed/baby2000')
        development = read_json(dev / 'development_identity.json')
        assert development['source_files'] == source_manifest(), 'Frozen source mismatch'
        assert development['dataset'] == dataset['manifest']['identity'], 'Dataset identity mismatch'
        train = _sets(dataset['train'])
        truth = _sets(dataset['valid'])
        users = np.array(sorted(truth), dtype=np.int64)
        neighbors = build_neighbors(dataset['train'], dataset['n_users'], dataset['n_items'], 20)
        environment = dict(python=platform.python_version(), torch=torch.__version__,
                           numpy=np.__version__, scipy=scipy.__version__)
        rows, curves = [], []
        for path in sorted(dev.iterdir()):
            if not path.is_dir() or not (path / 'result.json').exists():
                continue
            result, meta = read_json(path / 'result.json'), read_json(path / 'run_metadata.json')
            assert meta['status'] == 'completed' and result['test'] is None
            assert not meta['test_enabled'] and result['environment'] == environment
            assert result['identity'] == meta['identity'] == canonical_hash(
                {k: v for k, v in meta.items() if k not in ['identity', 'status']})
            for name, digest in result['artifacts'].items():
                assert file_hash(path / name) == digest, f'Artifact mismatch: {path.name}/{name}'
            condition = read_json(path / 'condition.json') if (path / 'condition.json').exists() else None
            vision = dataset['vision'].copy()
            missing = np.zeros(dataset['n_items'], dtype=bool)
            if condition:
                missing = make_missing(dataset['n_items'], dataset['tail'], condition['scenario'], condition['mask_seed'])
                assert np.array_equal(missing, np.load(path / 'missing.npy', allow_pickle=False))
                vision[missing] = 0
                vision, info = impute_features(vision, missing, neighbors, condition['method'],
                                              alpha=condition['alpha'] or .5, tau=condition['tau'] or 4.)
                assert array_hash(info['support_count']) == condition['support_hash']
            inputs = {k: array_hash(dataset[k]) for k in ['train', 'valid', 'test', 'tail']}
            inputs.update(missing=array_hash(missing), vision=array_hash(vision), text=array_hash(dataset['text']))
            assert inputs == result['inputs'], 'Reconstructed input mismatch'
            best = torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
            assert best['identity'] == result['identity'] and best['epoch'] == result['best_epoch']
            torch.set_num_threads(result['config']['num_threads'])
            model = FreedomModel(dataset['n_users'], dataset['n_items'], dataset['train'], vision,
                                 dataset['text'], result['config'], seed=result['stream_seeds']['initialization'])
            model.load_state_dict(best['state_dict'], strict=True)
            model.eval()
            with torch.no_grad():
                user_emb, item_emb = model.forward(model.norm_adj)
                batch_size = result['config']['eval_batch_size']
                scores = np.concatenate([(user_emb[torch.from_numpy(users[start:start + batch_size])] @ item_emb.T).numpy()
                                         for start in range(0, len(users), batch_size)])
            actual = evaluate(scores, users, truth, train,
                              dict(all=np.ones(dataset['n_items'], bool), tail=dataset['tail'],
                                   missing=missing, tail_missing=dataset['tail'] & missing), 20)
            _compare_metrics(actual, result['validation'], path.name)
            assert np.isclose(actual['summary']['all']['recall'], result['best_validation_recall'], atol=1e-12)
            row = dict(run_id=path.name, scenario=condition['scenario'] if condition else 'clean',
                       method=condition['method'] if condition else 'backbone', seed=result['seed'],
                       dropout=result['config']['dropout'], reg_weight=result['config']['reg_weight'],
                       best_epoch=result['best_epoch'], epochs_completed=result['epochs_completed'],
                       valid_recall20=actual['summary']['all']['recall'],
                       valid_ndcg20=actual['summary']['all']['ndcg'],
                       total_minutes=result['resources']['total_seconds'] / 60,
                       peak_rss_mib=result['resources']['peak_rss_bytes'] / 2**20,
                       stop_reason=result['stop_reason'])
            rows.append(row)
            with (path / 'history.csv').open(encoding='utf-8', newline='') as stream:
                history = list(csv.DictReader(stream))
            curves.append((path.name, history))
            evidence['runs'].append(dict(run_id=path.name, identity=result['identity'],
                result_sha256=file_hash(path / 'result.json'), artifacts_checked=len(result['artifacts']),
                validation_predictions_match=True, validation_users=len(users)))
            print(f'VERIFIED {path.name}: validation predictions and artifact hashes', flush=True)
            del model, best, scores, actual, vision, user_emb, item_emb
            gc.collect()
        assert len(rows) == 5, 'This stopped-run snapshot expects exactly five complete development results'
        with (output / 'development_summary.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        labels = ['d=.8, reg=1e-4', 'd=.8, reg=1e-3', 'd=.9, reg=1e-4', 'd=.9, reg=1e-3', 'uniform30, alpha=.25']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        colors = ['#2874A6'] * 4 + ['#D68910']
        for ax, field, title in [(axes[0], 'total_minutes', 'Completed run time (minutes)'),
                                 (axes[1], 'peak_rss_mib', 'Peak process memory (MiB)')]:
            bars = ax.barh(labels, [row[field] for row in rows], color=colors)
            ax.invert_yaxis()
            ax.set_xlim(0, max(row[field] for row in rows) * 1.2)
            ax.bar_label(bars, fmt='%.1f', padding=3)
            ax.set_title(title)
        fig.suptitle('Development runs only - not final method comparisons')
        fig.savefig(output / 'development_resources.png', dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for index, (name, history) in enumerate(curves):
            axis = axes[0] if index < 4 else axes[1]
            # Read the stored header, without deriving scores from a selected subset of epochs.
            recall_key = next(key for key in history[0] if 'recall' in key)
            axis.plot([int(h['epoch']) for h in history],
                      [100 * float(h[recall_key]) for h in history], label=labels[index])
        for axis, title in zip(axes, ['Clean backbone search', 'One completed imputation candidate']):
            axis.set(xlabel='Epoch', ylabel='Validation Recall@20 (%)', title=title, ylim=(0, 16))
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        fig.suptitle('Validation curves - seed 101; no test evaluation')
        fig.savefig(output / 'validation_curves.png', dpi=160)
        plt.close(fig)
        evidence.update(success=True, completed_development_runs=len(rows), planned_development_runs=16,
                        completed_main_runs=0, planned_main_runs=45,
                        source_files=source_manifest(), dataset_identity=dataset['manifest']['identity'],
                        total_seconds=time.perf_counter() - started,
                        outputs={p.name: file_hash(p) for p in output.iterdir() if p.name != marker.name and p.is_file()})
        write_json(marker, evidence)
        print(f'Interim export complete: {len(rows)} validated runs; no training or test evaluation')
    except BaseException as error:
        evidence.update(success=False, error_type=type(error).__name__, error=str(error))
        write_json(marker, evidence)
        raise


if __name__ == '__main__':
    main()
