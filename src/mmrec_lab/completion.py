"""Continue an existing live development job, gate, run and verify the full study.

Imports stay light while waiting, to avoid loading a second Torch process beside
the 16-run developer search on a memory-constrained laptop.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import psutil


def audit_development(runs, physical_ram_bytes):
    """Pre-test gate: complete search, CPU resources and early-stop evidence.

    More than half hitting the cap operationalizes '普遍未收敛' as a reason for
    review. This gate does not choose a new size/epoch cap or read test scores.
    """
    if len(runs) != 16 or len({r['run_id'] for r in runs}) != 16:
        raise ValueError('Exactly 16 distinct development candidates are required')
    capped = []
    for run in runs:
        if run['seed'] != 101 or run['test'] is not None:
            raise ValueError('Development seed must be 101 and test must remain disabled')
        if run['stop_reason'] not in ['epoch_limit', 'validation_patience']:
            raise ValueError('Development candidate has an incomplete stop state')
        if not 1 <= run['epochs_completed'] <= run['config']['epochs']:
            raise ValueError('Invalid completed epoch count')
        if run['stop_reason'] == 'epoch_limit':
            if run['epochs_completed'] != run['config']['epochs']:
                raise ValueError('Epoch-limit candidate has inconsistent history length')
            capped.append(run['run_id'])
        resource = run['resources']
        if resource['device'] != 'cpu':
            raise ValueError('Non-CPU development result')
        if not math.isfinite(resource['total_seconds']) or not 0 < resource['total_seconds'] <= 1800:
            raise ValueError('Development run exceeds 30-minute CPU target; review resources before test')
        if not 0 < resource['peak_rss_bytes'] <= physical_ram_bytes / 2:
            raise ValueError('Development peak RSS exceeds half physical RAM; review before test')
    if len(capped) > len(runs) / 2:
        raise ValueError('Majority of development candidates reached epoch limit; review before test')
    return {'decision': 'ready_for_main', 'candidates': len(runs), 'epoch_limit_runs': len(capped),
            'epoch_limit_run_ids': capped, 'max_run_seconds': max(r['resources']['total_seconds'] for r in runs),
            'total_development_seconds': sum(r['resources']['total_seconds'] for r in runs),
            'max_peak_rss_bytes': max(r['resources']['peak_rss_bytes'] for r in runs),
            'test_results_seen': False,
            'gate_rule': 'All 16 verified; test disabled; each <=1800s and RSS<=half physical RAM; <=8 epoch-limit candidates'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='等待已有开发任务，核验后顺序完成45次实验、重算报告与组件计时')
    parser.add_argument('--data', required=True)
    parser.add_argument('--development', required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--status', default='results/completion_status.json')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--wait-created', type=float)
    args = parser.parse_args(argv)
    status_path = Path(args.status)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    lock = status_path.with_suffix('.lock')
    owner = {'pid': os.getpid(), 'created': psutil.Process().create_time()}
    # A stale lock is handled explicitly after checking the exact prior process;
    # never start a second experiment loop merely because a status file is old.
    if lock.exists():
        prior = _read(lock)
        try:
            alive = abs(psutil.Process(prior['pid']).create_time() - prior['created']) < .01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError('Completion process is already live')
        lock.unlink()
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(owner, stream)
    state = {**owner, 'stage': 'waiting_development', 'status': 'running',
             'data': args.data, 'development': args.development, 'study': args.study, 'report': args.report}
    try:
        _write(status_path, state)
        if args.wait_pid is not None:
            if args.wait_created is None:
                raise ValueError('--wait-created is required with --wait-pid to prevent PID reuse confusion')
            while True:
                try:
                    process = psutil.Process(args.wait_pid)
                    if abs(process.create_time() - args.wait_created) >= .01 or not process.is_running():
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(15)
        development = Path(args.development)
        if not (development / 'frozen_protocol.json').is_file():
            raise RuntimeError('Development process ended without a frozen protocol; inspect its real failure')
        state['stage'] = 'verify_development'
        _write(status_path, state)
        print('VERIFY 16 development runs and their saved artifact hashes', flush=True)
        # Reusing develop performs its identity/parameter/artifact checks; every
        # already-completed training run is reused, not retrained.
        from .io import file_hash, write_json
        from .study import develop, run_study
        original = _read(development / 'frozen_protocol.json')
        dev_identity = _read(development / 'development_identity.json')
        protocol = develop(args.data, development, dev_identity['config'])
        if protocol != original:
            raise ValueError('Development re-verification changed the frozen protocol')
        run_ids = [r['run_id'] for r in _read(development / 'backbone_selection.json')['candidates']]
        for parameter in ['alpha', 'tau']:
            run_ids += [r['run_id'] for r in _read(development / f'{parameter}_selection.json')['runs']]
        results = [{**_read(development / run_id / 'result.json'), 'run_id': run_id} for run_id in run_ids]
        audit = audit_development(results, psutil.virtual_memory().total)
        audit.update({'protocol_identity': protocol['identity'], 'completion_source_sha256': file_hash(__file__)})
        write_json(development / 'pretest_audit.json', audit)
        state['stage'] = 'main_45_runs'
        _write(status_path, state)
        run_study(args.data, development / 'frozen_protocol.json', args.study)
        state['stage'] = 'recompute_report'
        _write(status_path, state)
        from .report import build_report
        report = build_report(args.data, args.study, args.report)
        state['stage'] = 'component_benchmarks'
        _write(status_path, state)
        from .benchmark import benchmark_components
        benchmark = benchmark_components(args.data, development / 'frozen_protocol.json', args.report)
        state.update({'stage': 'artifacts_ready_for_review', 'status': 'completed',
                      'recomputed_runs': report['recomputed_runs'], 'benchmark_rows': benchmark['rows'],
                      'requires': 'Review visual outputs, findings and acceptance evidence before drawing research conclusions.'})
        _write(status_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        state.update({'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      'error_type': type(exc).__name__, 'error': str(exc)})
        _write(status_path, state)
        raise
    finally:
        if lock.exists() and _read(lock) == owner:
            lock.unlink()


if __name__ == '__main__':
    main()
