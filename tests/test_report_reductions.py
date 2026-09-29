"""Small hand-calculated reductions, without the 45-run artifact fixture."""
import numpy as np
import pytest

from mmrec_lab.metrics import evaluate_rankings
from mmrec_lab.report import _cases, _stats


def test_three_seed_summary_uses_sample_not_population_standard_deviation():
    result = _stats([.01, .02, .03])
    assert result['mean'] == pytest.approx(.02)
    assert result['std'] == pytest.approx(.01)
    assert result['std'] != pytest.approx(np.std([.01, .02, .03], ddof=0))
    assert result['n'] == 3


def test_clean_empty_groups_stay_null_and_sparse_cases_are_not_padded():
    tail = np.array([False, True, True, False])
    missing = np.zeros(4, dtype=bool)
    groups = dict(all=np.ones(4, bool), tail=tail, missing=missing, tail_missing=tail & missing)
    truth = {0: {1}, 1: {2}}
    details = {}
    empty_values = []
    # Exactly one improvement and one decline; all other pairs are ties.
    for seed, support_item, fixed_item in [(201, 1, 3), (202, 3, 1), (203, 1, 1)]:
        for method, item in [('support_mix', support_item), ('fixed_mix', fixed_item)]:
            metrics = evaluate_rankings([{'user': 0, 'items': [item]}, {'user': 1, 'items': [2]}],
                                        truth, groups, k=20)
            for group in ('missing', 'tail_missing'):
                assert metrics['summary'][group] == {'recall': None, 'ndcg': None, 'users': 0, 'positives': 0}
            if method == 'support_mix':
                empty_values.append(metrics['summary']['missing']['recall'])
            details[seed, method] = dict(metrics=metrics, missing=missing, support=np.zeros(4, dtype=np.int64))
    assert _stats(empty_values) == {'mean': None, 'std': None, 'n': 0}
    cases, improving, worsening = _cases(details, truth, tail)
    assert improving == 1 and worsening == 1 and len(cases) == 2
    assert [(case['kind'], case['seed'], case['user_index']) for case in cases] == [
        ('improving', 201, 0), ('worsening', 202, 0)]
    assert all(case['delta_recall'] != 0 for case in cases)
