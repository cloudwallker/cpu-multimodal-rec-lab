import importlib
import json
import unittest

import numpy as np


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mmrec_lab.metrics'), 'metrics module missing')
        self.m = importlib.import_module('mmrec_lab.metrics')

    def groups(self):
        return {'all': np.ones(5, bool), 'tail': np.array([False, True, False, True, False]),
                'missing': np.array([False, False, True, False, False]),
                'tail_missing': np.zeros(5, bool)}

    def test_hand_ranking_tie_blocking_and_group_denominators(self):
        result = self.m.evaluate(np.array([[9., 5., 5., 4., 3.]]), np.array([7]),
                                 {7: {1, 3}}, {7: {0}}, self.groups(), k=3)
        self.assertEqual(result['rankings'], [{'user': 7, 'items': [1, 2, 3]}])
        self.assertEqual(result['summary']['all']['recall'], 1)
        expected = 1.5 / (1 + 1 / np.log2(3))
        self.assertAlmostEqual(result['summary']['all']['ndcg'], expected)
        self.assertEqual(result['summary']['tail']['positives'], 2)
        self.assertEqual(result['summary']['missing']['users'], 0)
        self.assertIsNone(result['summary']['missing']['recall'])
        self.assertIsNone(result['summary']['tail_missing']['ndcg'])
        json.dumps(result, allow_nan=False)
        self.assertEqual(result, self.m.evaluate_rankings(result['rankings'], {7: {1, 3}}, self.groups(), k=3))

    def test_group_uses_full_catalog_rank_positions(self):
        groups = self.groups(); groups['tail'] = np.array([False, False, False, True, False])
        r = self.m.evaluate_rankings([{'user': 7, 'items': [1, 2, 3]}], {7: {1, 3}}, groups, 3)
        self.assertEqual(r['summary']['tail']['recall'], 1)
        self.assertEqual(r['summary']['tail']['ndcg'], .5)

    def test_short_candidate_list_and_users_without_positives(self):
        r = self.m.evaluate(np.ones((2, 5)), np.array([7, 8]), {7: {4}, 8: set()},
                            {7: {0, 1, 2, 3}, 8: set()}, self.groups(), k=20)
        self.assertEqual(r['rankings'][0]['items'], [4])
        self.assertEqual(r['summary']['all']['users'], 1)
        self.assertEqual(r['summary']['all']['positives'], 1)
        self.assertEqual(r['summary']['all']['recall'], 1)

    def test_empty_users_have_null_metrics(self):
        r = self.m.evaluate(np.empty((0, 5)), np.array([], dtype=np.int64), {}, {}, self.groups())
        self.assertIsNone(r['summary']['all']['recall'])

    def test_unsigned_scores_rank_numerically(self):
        r = self.m.evaluate(np.array([[0, 2, 1, 0, 0]], dtype=np.uint8), np.array([7]),
                            {7: {1}}, {}, self.groups(), k=2)
        self.assertEqual(r['rankings'][0]['items'], [1, 2])

    def test_invalid_inputs_fail_instead_of_silent_scoring(self):
        scores = np.ones((1, 5)); users = np.array([7]); groups = self.groups()
        for sc, us, truth, blocked, gs, k in [
            (scores * np.inf, users, {7: {1}}, {}, groups, 20),
            (scores, users, {7: {1}}, {7: {1}}, groups, 20),
            (scores, users, {7: {5}}, {}, groups, 20),
            (scores, users, {7: {1}}, {7: {-1}}, groups, 20),
            (np.ones((2, 5)), np.array([7, 7]), {7: {1}}, {}, groups, 20),
            (scores, users, {7: {1}}, {}, {'all': np.ones(4, bool)}, 20),
            (scores, users, {7: {1}}, {}, groups, 0),
        ]:
            with self.subTest(k=k, truth=truth):
                with self.assertRaises(ValueError):
                    self.m.evaluate(sc, us, truth, blocked, gs, k)
        with self.assertRaises(ValueError):
            self.m.evaluate_rankings([{'user': 7, 'items': [1, 1]}], {7: {1}}, groups)


if __name__ == '__main__':
    unittest.main()
