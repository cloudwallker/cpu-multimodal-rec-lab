"""Separate warm component timings; never substitutes for end-to-end run time."""
import csv
import gc
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .data import load_dataset
from .impute import build_neighbors, make_missing, impute_features
from .io import canonical_hash, file_hash, read_json, write_json
from .model import FreedomModel
from .study import METHODS, SCENARIOS, source_manifest


def benchmark_components(dataset_dir, protocol_path, output_dir, repeats=3):
    if not isinstance(repeats, int) or repeats < 2:
        raise ValueError('At least two component timing repetitions are required')
    dataset = load_dataset(dataset_dir)
    protocol = read_json(protocol_path)
    if protocol['identity'] != canonical_hash({k:v for k,v in protocol.items() if k != 'identity'}):
        raise ValueError('Protocol checksum mismatch')
    if protocol['dataset_identity'] != dataset['manifest']['identity']:
        raise ValueError('Benchmark data differs from frozen study')
    sources = source_manifest()
    if protocol.get('source_files') != sources:
        raise ValueError('Benchmark source files differ from frozen study')
    torch.set_num_threads(protocol['config']['num_threads'])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    def measure(component, scenario, method, seed, fn):
        value = fn()  # explicit unmeasured warm-up
        for repetition in range(repeats):
            begin = time.perf_counter()
            value = fn()
            rows.append({'component': component, 'scenario': scenario, 'method': method,
                         'mask_seed': seed, 'repetition': repetition + 1,
                         'seconds': time.perf_counter() - begin})
        return value

    neighbors = measure('train_cointeraction_graph', 'shared', 'shared', None,
                        lambda: build_neighbors(dataset['train'], dataset['n_users'], dataset['n_items'], 20))
    # _knn_graph reads only these two attributes, as checked in model.py. This
    # times the same pure graph routine without embeddings/optimizer/init cost.
    graph_context = SimpleNamespace(n_items=dataset['n_items'], knn_k=protocol['config']['knn_k'])
    text_tensor = torch.from_numpy(dataset['text'])
    measure('text_knn_graph', 'shared', 'shared', None,
            lambda: FreedomModel._knn_graph(graph_context, text_tensor))
    for scenario in SCENARIOS:
        for mask_seed in [301, 302, 303]:
            missing = make_missing(dataset['n_items'], dataset['tail'], scenario, mask_seed)
            visible = dataset['vision'].copy()
            visible[missing] = 0
            for method in METHODS:
                filled, _ = measure('imputation_function', scenario, method, mask_seed,
                    lambda: impute_features(visible, missing, neighbors, method,
                        alpha=protocol['alpha'], tau=protocol['tau']))
                tensor = torch.from_numpy(filled)
                measure('image_knn_graph', scenario, method, mask_seed,
                        lambda: FreedomModel._knn_graph(graph_context, tensor))
            gc.collect()
    path = output / 'component_timings.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {'dataset_identity': dataset['manifest']['identity'], 'protocol_identity': protocol['identity'],
                'repeats': repeats, 'threads': protocol['config']['num_threads'], 'device': 'cpu',
                'source_files': sources, 'model_source_sha256': sources['model.py'],
                'impute_source_sha256': sources['impute.py'],
                'benchmark_source_sha256': file_hash(__file__), 'csv_sha256': file_hash(path),
                'rows': len(rows), 'timing_scope': 'One warm-up plus recorded repetitions; imputation includes its validation and copy, excludes externally constructing the mask and zeroed visible input. kNN timings use the exact model graph routine, exclude fusion and embedding/optimizer initialization. No model training or test scoring.'}
    write_json(output / 'component_benchmark.json', manifest)
    return manifest
