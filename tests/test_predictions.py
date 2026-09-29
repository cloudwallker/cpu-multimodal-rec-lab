"""Real tiny CPU checkpoints, exclusively temporary test data, never research results."""
import importlib
import platform

import numpy as np
import pytest
import scipy
import torch
from scipy.sparse import save_npz

from mmrec_lab.impute import build_neighbors, make_missing, impute_features
from mmrec_lab.io import array_hash, canonical_hash, file_hash, read_json, write_json
from mmrec_lab.metrics import evaluate, evaluate_rankings
from mmrec_lab.model import FreedomModel
from mmrec_lab.study import BASE_CONFIG, experiment_matrix, source_manifest


def verifier():
    assert importlib.util.find_spec('mmrec_lab.verify_predictions') is not None, 'checkpoint prediction verifier missing'
    return importlib.import_module('mmrec_lab.verify_predictions').verify_predictions


@pytest.fixture
def checkpoint_study(tmp_path):
    torch.set_num_threads(1)
    data_dir, study = tmp_path / 'data', tmp_path / 'study'
    data_dir.mkdir()
    study.mkdir()
    arrays = dict(train=np.array([[0, 0], [0, 1], [1, 1], [1, 2]], np.int64),
                  valid=np.array([[0, 2], [1, 3]], np.int64),
                  test=np.array([[0, 3], [1, 4]], np.int64),
                  vision=np.array([[1., 0], [0, 1], [1, 1], [.2, .9], [.8, .3], [.3, .4]], np.float32),
                  text=np.array([[.3, 1], [1, .2], [.5, .3], [.7, .8], [.1, .6], [.9, .5]], np.float32),
                  tail=np.array([False]*3+[True]*3), item_ids=np.arange(6), user_ids=np.arange(2))
    np.savez(data_dir / 'dataset.npz', **arrays)
    manifest = dict(dataset='checkpoint-unit-fixture', n_users=2, n_items=6,
                    dataset_sha256=file_hash(data_dir / 'dataset.npz'),
                    array_hashes={k: array_hash(v) for k, v in arrays.items()})
    manifest['identity'] = canonical_hash(manifest)
    write_json(data_dir / 'manifest.json', manifest)
    config = {**BASE_CONFIG, 'embedding_size': 2, 'feat_embed_dim': 2, 'knn_k': 2,
              'num_threads': 1, 'eval_batch_size': 1, 'epochs': 2}
    sources = source_manifest()
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       numpy=np.__version__, scipy=scipy.__version__)
    protocol = dict(dataset_identity=manifest['identity'], config=config, alpha=.5, tau=4,
                    source_files=sources, runs=experiment_matrix())
    protocol['identity'] = canonical_hash(protocol)
    write_json(study / 'frozen_protocol.json', protocol)
    neighbors = build_neighbors(arrays['train'], 2, 6)
    save_npz(study / 'neighbors.npz', neighbors)
    write_json(study / 'neighbors.json', dict(sha256=file_hash(study / 'neighbors.npz'),
                                             train_hash=array_hash(arrays['train'])))
    completed = []
    for condition in protocol['runs']:
        path = study / condition['run_id']
        path.mkdir()
        missing = make_missing(6, arrays['tail'], condition['scenario'], condition['mask_seed'] or 0)
        vision, text = arrays['vision'].copy(), arrays['text']
        support = np.zeros(6, np.int64)
        if condition['scenario'] != 'clean':
            vision[missing] = 0
            vision, info = impute_features(vision, missing, neighbors, condition['method'], alpha=.5, tau=4)
            support = info['support_count']
        if condition['modality'] == 'image':
            text = None
        elif condition['modality'] == 'text':
            vision = None
        inputs = {k: array_hash(arrays[k]) for k in ['train', 'valid', 'test', 'tail']}
        inputs.update(missing=array_hash(missing), vision=None if vision is None else array_hash(vision),
                      text=None if text is None else array_hash(text))
        np.save(path / 'missing.npy', missing)
        np.save(path / 'support.npy', support)
        meta = {**condition, 'missing_hash': inputs['missing'], 'vision_hash': inputs['vision'],
                'text_hash': inputs['text'], 'support_hash': array_hash(support),
                'neighbor_hash': canonical_hash([array_hash(neighbors.indptr), array_hash(neighbors.indices), array_hash(neighbors.data)]),
                'alpha': .5 if condition['method'] == 'fixed_mix' else None,
                'tau': 4 if condition['method'] == 'support_mix' else None}
        write_json(path / 'condition.json', meta)
        stream_values = [int(x) for x in np.random.SeedSequence(condition['seed']).generate_state(4)]
        streams = dict(zip(['initialization', 'negative', 'graph', 'shuffle'], stream_values))
        model = FreedomModel(2, 6, arrays['train'], vision, text, config, seed=streams['initialization'])
        with torch.no_grad():
            users, items = model.forward(model.norm_adj)
            scores = (users @ items.T).numpy()
        groups = dict(all=np.ones(6, bool), tail=arrays['tail'], missing=missing, tail_missing=arrays['tail'] & missing)
        prediction = evaluate(scores, np.array([0, 1]), {0: {3}, 1: {4}}, {0: {0, 1, 2}, 1: {1, 2, 3}}, groups)
        identity_payload = dict(schema_version=1, config=config, seed=condition['seed'], test_enabled=True,
            n_users=2, n_items=6, inputs=inputs, dataset_manifest=manifest,
            source_files={k: sources[k] for k in ['train.py', 'model.py', 'metrics.py', 'io.py']}, environment=environment)
        identity = canonical_hash(identity_payload)
        torch.save(dict(identity=identity, epoch=1, state_dict=model.state_dict()), path / 'best.pt')
        write_json(path / 'rankings.json', prediction['rankings'])
        artifacts = {name: file_hash(path / name) for name in ['best.pt', 'rankings.json', 'condition.json', 'missing.npy', 'support.npy']}
        write_json(path / 'run_metadata.json', {**identity_payload, 'identity': identity, 'status': 'completed'})
        write_json(path / 'result.json', dict(identity=identity, seed=condition['seed'], config=config,
             inputs=inputs, source_files=identity_payload['source_files'], environment=environment,
             stream_seeds=streams, best_epoch=1, test=prediction, artifacts=artifacts))
        completed.append(dict(run_id=condition['run_id'], result_identity=identity))
    write_json(study / 'study_status.json', dict(status='complete', completed=45, planned=45,
                                               runs=completed, protocol_identity=protocol['identity']))
    return data_dir, study, tmp_path / 'verification'


def test_all_45_checkpoints_are_loaded_and_actually_predicted(checkpoint_study, monkeypatch):
    verify = verifier()
    original = FreedomModel.forward
    actual_calls = []
    def counted(model, adjacency=None):
        actual_calls.append(adjacency is model.norm_adj)
        return original(model, adjacency)
    monkeypatch.setattr(FreedomModel, 'forward', counted)
    result = verify(*checkpoint_study)
    assert result['success'] is True and result['verified_runs'] == 45
    assert actual_calls == [True]*45
    assert len(result['runs']) == 45 and all(row['success'] for row in result['runs'])
    assert read_json(checkpoint_study[2] / 'prediction_verification.json') == result
    assert all(row['best_sha256'] and row['prediction_seconds'] >= 0 for row in result['runs'])


def test_self_consistent_forged_ranking_fails_real_prediction(checkpoint_study):
    verify = verifier()
    data_dir, study, output = checkpoint_study
    verify(data_dir, study, output)  # a stale success must become failure
    path = study / 'tail30_support_mix_s203'
    saved = read_json(path / 'result.json')
    rankings = read_json(path / 'rankings.json')
    rankings[0]['items'].reverse()  # valid candidates and internally consistent metrics
    missing = np.load(path / 'missing.npy')
    groups = dict(all=np.ones(6, bool), tail=np.array([False]*3+[True]*3), missing=missing,
                  tail_missing=np.array([False]*3+[True]*3) & missing)
    saved['test'] = evaluate_rankings(rankings, {0: {3}, 1: {4}}, groups)
    write_json(path / 'rankings.json', rankings)
    saved['artifacts']['rankings.json'] = file_hash(path / 'rankings.json')
    write_json(path / 'result.json', saved)
    with pytest.raises(ValueError, match='prediction.*rank|rank.*prediction'):
        verify(data_dir, study, output)
    failure = read_json(output / 'prediction_verification.json')
    assert failure['success'] is False and failure['failed_run'] == path.name


@pytest.mark.parametrize('corruption', ['best_hash', 'epoch', 'source', 'environment', 'missing_run'])
def test_checkpoint_identity_and_incomplete_study_are_rejected(checkpoint_study, corruption):
    verify = verifier()
    _, study, output = checkpoint_study
    path = study / 'clean_both_s201'
    if corruption == 'best_hash':
        with (path / 'best.pt').open('ab') as stream:
            stream.write(b'corruption')
    elif corruption == 'epoch':
        checkpoint = torch.load(path / 'best.pt', weights_only=False)
        checkpoint['epoch'] = 2
        torch.save(checkpoint, path / 'best.pt')
        saved = read_json(path / 'result.json')
        saved['artifacts']['best.pt'] = file_hash(path / 'best.pt')
        write_json(path / 'result.json', saved)
    elif corruption in ['source', 'environment']:
        protocol = read_json(study / 'frozen_protocol.json')
        if corruption == 'source':
            protocol['source_files']['model.py'] = 'changed'
            protocol['identity'] = canonical_hash({k: v for k, v in protocol.items() if k != 'identity'})
            write_json(study / 'frozen_protocol.json', protocol)
        else:
            saved = read_json(path / 'result.json')
            saved['environment']['torch'] = 'another-runtime'
            write_json(path / 'result.json', saved)
    else:
        (path / 'best.pt').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(*checkpoint_study)
    assert read_json(output / 'prediction_verification.json')['success'] is False
