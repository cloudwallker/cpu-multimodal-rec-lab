"""Pre-test structural diagnostic using only train interactions and fixed masks."""
import argparse
import csv
from pathlib import Path

import numpy as np

from mmrec_lab.data import load_dataset
from mmrec_lab.impute import build_neighbors, make_missing
from mmrec_lab.io import file_hash, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()
data = load_dataset(args.data)
neighbors = build_neighbors(data['train'], data['n_users'], data['n_items'], 20)
rows, distributions = [], []
for scenario in ['uniform30', 'tail30']:
    for seed in [301, 302, 303]:
        missing = make_missing(data['n_items'], data['tail'], scenario, seed)
        support = np.asarray(neighbors @ (~missing).astype(np.float32)).reshape(-1).astype(int)
        values = support[missing]
        hist = np.bincount(values, minlength=21)
        if len(hist) != 21 or hist.sum() != missing.sum():
            raise ValueError('Invalid support count in fixed Top20 graph')
        rows.append({'scenario': scenario, 'mask_seed': seed, 'masked_items': int(missing.sum()),
            'missing_rate': float(missing.mean()), 'tail_missing_rate': float(missing[data['tail']].mean()),
            'mean_support': float(values.mean()), 'median_support': float(np.median(values)),
            'std_support_population': float(values.std()), 'min_support': int(values.min()),
            'max_support': int(values.max()), 'zero_support_fraction': float((values == 0).mean()),
            'top20_fraction': float((values == 20).mean()), 'distinct_support_counts': len(np.unique(values))})
        distributions.append({'scenario': scenario, 'mask_seed': seed, 'histogram': hist.tolist()})
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
with (output / 'support_pretest.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
write_json(output / 'support_pretest.json', {'dataset_identity': data['manifest']['identity'],
    'source_sha256': file_hash(__file__), 'train_hash': data['manifest']['array_hashes']['train'],
    'impute_source_sha256': file_hash(Path(__file__).parents[1] / 'src/mmrec_lab/impute.py'),
    'scope': 'Training graph and predeclared masks only; no recommendation scoring or parameter selection',
    'rows': rows, 'distributions': distributions})
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True, sharey=True)
    for ax, scenario in zip(axes, ['uniform30', 'tail30']):
        for record in distributions:
            if record['scenario'] == scenario:
                counts = np.array(record['histogram'])
                ax.plot(np.arange(21), counts / counts.sum(), marker='.', label=str(record['mask_seed']))
        ax.set(title=scenario, xlabel='Observed neighbors of masked items', ylabel='Fraction of masked items')
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(bottom=0)
        ax.legend(title='Mask seed')
        ax.grid(alpha=.2)
    fig.savefig(output / 'support_pretest.png', dpi=170)
    plt.close(fig)
for row in rows:
    print(row)
