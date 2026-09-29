import importlib
import unittest
import json

import numpy as np
from scipy import sparse


class ImputationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mmrec_lab.impute'), 'impute module missing')
        self.m = importlib.import_module('mmrec_lab.impute')

    def fixture(self):
        visible = np.array([[0, 0], [2, 4], [0, 0], [6, 8], [0, 0]], dtype=np.float32)
        missing = np.array([True, False, True, False, True])
        graph = sparse.csr_matrix(([1, 1, 1], ([0, 0, 2], [1, 2, 4])), shape=(5, 5))
        return visible, missing, graph

    def test_six_formulas_and_no_recursive_evidence(self):
        v, missing, graph = self.fixture()
        expected = {
            'zero_fill': [[0, 0], [0, 0], [0, 0]],
            'global_mean': [[4, 6], [4, 6], [4, 6]],
            'paper_neighmean': [[1, 2], [0, 0], [0, 0]],
            'observed_mean': [[2, 4], [4, 6], [4, 6]],
            'fixed_mix': [[3, 5], [4, 6], [4, 6]],
            'support_mix': [[3.6, 5.6], [4, 6], [4, 6]],
        }
        for method, values in expected.items():
            with self.subTest(method=method):
                out, info = self.m.impute_features(v, missing, graph, method)
                np.testing.assert_allclose(out[missing], values, atol=1e-6)
                np.testing.assert_array_equal(out[~missing], v[~missing])
                np.testing.assert_array_equal(info['support_count'], [1, 0, 0, 0, 0])
                np.testing.assert_array_equal(info['degree'], [2, 0, 1, 0, 0])
                np.testing.assert_array_equal(info['global_mean'], [4, 6])
                json.dumps(info['summary'], allow_nan=False)
                self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(v[missing], 0)

    def test_neighbors_binary_directed_tie_and_duplicate_interactions(self):
        train = np.array([[0, 0], [0, 1], [0, 2], [0, 0], [1, 0], [1, 2]], dtype=np.int64)
        graph = self.m.build_neighbors(train, 2, 4, k=1)
        np.testing.assert_array_equal(graph.toarray(), [[0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertTrue(sparse.isspmatrix_csr(graph))

    def test_empty_interactions_and_singleton(self):
        graph = self.m.build_neighbors(np.empty((0, 2), dtype=np.int64), 1, 1)
        self.assertEqual(graph.nnz, 0)
        out, _ = self.m.impute_features(np.ones((1, 2), dtype=np.float32), np.array([False]), graph, 'support_mix')
        np.testing.assert_array_equal(out, [[1, 1]])

    def test_mask_exact_counts_determinism_and_tail_membership(self):
        tail = np.arange(11) < 6
        for scenario in ['uniform30', 'tail30', 'clean']:
            a = self.m.make_missing(11, tail, scenario, 301)
            b = self.m.make_missing(11, tail, scenario, 301)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.sum(), 0 if scenario == 'clean' else 3)
            if scenario == 'tail30':
                self.assertFalse(np.any(a & ~tail))

    def test_bad_masks_and_graph_input(self):
        v, missing, graph = self.fixture()
        bad_v = v.copy(); bad_v[0, 0] = 99
        for features, mask, a, method, alpha, tau in [
            (bad_v, missing, graph, 'support_mix', .5, 4),
            (v * np.nan, missing, graph, 'support_mix', .5, 4),
            (v, np.ones(5, dtype=bool), graph, 'support_mix', .5, 4),
            (v, missing.astype(int), graph, 'support_mix', .5, 4),
            (v, missing, graph, 'unknown', .5, 4),
            (v, missing, graph, 'fixed_mix', 1.1, 4),
            (v, missing, graph, 'support_mix', .5, 0),
            (v, missing, graph, 'support_mix', .5, np.inf),
            (v, missing, sparse.eye(5, format='csr'), 'support_mix', .5, 4),
            (v, missing, graph * 2, 'support_mix', .5, 4),
        ]:
            with self.subTest(method=method, tau=tau):
                with self.assertRaises(ValueError):
                    self.m.impute_features(features, mask, a, method, alpha, tau)
        with self.assertRaises(ValueError):
            self.m.make_missing(11, np.zeros(11, dtype=bool), 'tail30', 1)
        for train in [np.array([[0, 4]]), np.array([[2, 0]]), np.array([[0., 1.]]), np.array([1, 2])]:
            with self.assertRaises(ValueError):
                self.m.build_neighbors(train, 2, 4)

    def test_empty_shapes_and_malformed_parameters(self):
        v, missing, graph = self.fixture()
        for alpha, tau in [('bad', 4), (.5, 'bad'), (None, 4), (.5, None)]:
            with self.subTest(alpha=alpha, tau=tau):
                with self.assertRaises(ValueError):
                    self.m.impute_features(v, missing, graph, 'support_mix', alpha, tau)
        for shape in [(0, 2), (2, 0)]:
            with self.assertRaises(ValueError):
                self.m.impute_features(np.zeros(shape, dtype=np.float32), np.zeros(shape[0], bool),
                                       sparse.csr_matrix((shape[0], shape[0])), 'global_mean')
        with self.assertRaises(ValueError):
            self.m.build_neighbors(np.empty((0, 2), dtype=np.int64), 1, 0)


if __name__ == '__main__':
    unittest.main()
