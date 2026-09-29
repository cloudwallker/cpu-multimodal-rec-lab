import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class TrainTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mmrec_lab.train'), 'training module missing')
        self.module = importlib.import_module('mmrec_lab.train')
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.path = Path(self.root.name)
        self.data = {'train': np.array([[0, 0], [0, 1], [1, 1], [1, 2]], np.int64),
                     'valid': np.array([[0, 2], [1, 3]], np.int64),
                     'test': np.array([[0, 3], [1, 0]], np.int64),
                     'tail': np.array([False, False, True, True]),
                     'n_users': 2, 'n_items': 4, 'manifest': {'identity': 'tiny-unit-fixture'}}
        self.vision = np.array([[1., 0], [0, 1], [1, 1], [.5, 1]], np.float32)
        self.config = {'embedding_size': 4, 'feat_embed_dim': 4, 'knn_k': 2,
                       'dropout': .25, 'epochs': 3, 'patience': 5,
                       'batch_size': 8, 'num_threads': 1, 'eval_batch_size': 1, 'lr': .01}

    def run_training(self, path, test=True, **overrides):
        return self.module.train_run(self.data, self.vision, self.vision, {**self.config, **overrides},
                                     path, seed=201, test=test)

    def test_real_cpu_training_test_artifacts_and_completed_skip(self):
        r = self.run_training(self.path)
        self.assertEqual(len(r['history']), 3)
        self.assertTrue(np.isfinite([x['loss'] for x in r['history']]).all())
        self.assertGreater(r['resources']['peak_rss_bytes'], 0)
        self.assertEqual(r['test']['summary']['all']['recall'], 1.)
        for name in ['best.pt', 'resume.pt', 'result.json', 'rankings.json', 'per_user_metrics.csv']:
            self.assertTrue((self.path / name).exists())
        for record in r['test']['rankings']:
            banned = {0, 1, 2} if record['user'] == 0 else {1, 2, 3}
            self.assertFalse(banned.intersection(record['items']))
        with patch.object(self.module.FreedomModel, 'loss', side_effect=AssertionError('should skip')):
            self.assertEqual(self.run_training(self.path)['identity'], r['identity'])

    def test_development_never_invokes_test_evaluation(self):
        original = self.module.evaluate
        def checked(scores, users, truth, blocked, groups, k=20):
            self.assertEqual(truth, {0: {2}, 1: {3}})
            return original(scores, users, truth, blocked, groups, k)
        with patch.object(self.module, 'evaluate', side_effect=checked):
            r = self.run_training(self.path, test=False)
        self.assertIsNone(r['test'])
        self.assertFalse((self.path / 'rankings.json').exists())

    def test_changed_identity_and_corrupted_completed_artifact_fail(self):
        self.run_training(self.path)
        with self.assertRaises(ValueError):
            self.run_training(self.path, lr=.02)
        (self.path / 'rankings.json').write_text('[]', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.run_training(self.path)

    def test_interruption_epoch_checkpoint_restores_exact_training(self):
        uninterrupted = self.run_training(self.path / 'full', dropout=0.)
        original = self.module.FreedomModel.loss
        calls = [0]
        def interrupt(model, users, pos, neg):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError('simulated interruption')
            return original(model, users, pos, neg)
        with patch.object(self.module.FreedomModel, 'loss', new=interrupt):
            with self.assertRaises(RuntimeError):
                self.run_training(self.path / 'resumed', dropout=0.)
        from mmrec_lab.io import read_json
        self.assertEqual(read_json(self.path / 'resumed' / 'run_metadata.json')['status'], 'failed')
        # Simulate an ahead-of-resume best.pt after interruption between writes.
        import torch
        best_path = self.path / 'resumed' / 'best.pt'
        best = torch.load(best_path, weights_only=False)
        best['epoch'] = 99
        best['state_dict'] = {key: torch.zeros_like(value) for key, value in best['state_dict'].items()}
        torch.save(best, best_path)
        resumed = self.run_training(self.path / 'resumed', dropout=0.)
        self.assertEqual(resumed['test'], uninterrupted['test'])
        self.assertEqual([h['loss'] for h in resumed['history']], [h['loss'] for h in uninterrupted['history']])
        full_best = torch.load(self.path / 'full' / 'best.pt', weights_only=False)
        resumed_best = torch.load(best_path, weights_only=False)
        for key in full_best['state_dict']:
            self.assertTrue(torch.equal(full_best['state_dict'][key], resumed_best['state_dict'][key]))

    def test_validation_patience_and_learning_not_test_select_best(self):
        r = self.run_training(self.path, test=False, epochs=12, patience=2, dropout=0.)
        self.assertEqual(r['best_epoch'], 1)
        self.assertEqual(r['epochs_completed'], 3)
        self.assertEqual(r['stop_reason'], 'validation_patience')
        self.assertTrue(r['is_converged'])
        self.assertLess(r['history'][-1]['loss'], r['history'][0]['loss'])

    def test_training_preserves_callers_torch_rng(self):
        import torch
        before = torch.random.get_rng_state().clone()
        self.run_training(self.path, test=False)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_condition_preparation_files_are_allowed_but_unknown_files_fail(self):
        from mmrec_lab.io import write_json
        write_json(self.path / 'environment.json', {'device': 'cpu'})
        write_json(self.path / 'condition.json', {'method': 'clean'})
        np.save(self.path / 'missing.npy', np.zeros(4, bool))
        np.save(self.path / 'support.npy', np.zeros(4, np.int64))
        self.run_training(self.path, test=False)
        unrelated = self.path / 'other'
        unrelated.mkdir()
        (unrelated / 'unknown.txt').write_text('stale', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.run_training(unrelated)


if __name__ == '__main__':
    unittest.main()
