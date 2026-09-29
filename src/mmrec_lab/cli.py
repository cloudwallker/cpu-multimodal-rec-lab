"""Command line entry points; all learning runs use CPU."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import psutil

from .io import read_json, write_json


def environment():
    cpu = platform.processor()
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return {'python': platform.python_version(), 'platform': platform.system(),
            'platform_release': platform.release(), 'machine': platform.machine(), 'processor': cpu,
            'physical_cpus': psutil.cpu_count(logical=False), 'logical_cpus': psutil.cpu_count(),
            'ram_bytes': psutil.virtual_memory().total,
            'available_ram_at_measurement': psutil.virtual_memory().available,
            'packages': {p: importlib.metadata.version(p) for p in
                         ['torch', 'numpy', 'scipy', 'pandas', 'matplotlib', 'psutil', 'gdown', 'pytest']}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='CPU FREEDOM 缺失图像研究')
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download', help='下载并核验作者Baby预提取文件')
    download.add_argument('--output', default='data/raw/baby')
    prepare = commands.add_parser('prepare', help='固定训练热度分层子集')
    prepare.add_argument('--raw', default='data/raw/baby')
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--items', type=int, default=2000)
    prepare.add_argument('--seed', type=int, default=20260929)
    env = commands.add_parser('environment', help='保存无个人身份的环境版本与硬件')
    env.add_argument('--output', default='results/environment.json')
    for name in ['profile', 'develop', 'study', 'report', 'benchmark']:
        sub = commands.add_parser(name)
        sub.add_argument('--data', required=True)
        sub.add_argument('--output', required=True)
        if name in ['profile', 'develop']:
            sub.add_argument('--threads', type=int, default=2)
            sub.add_argument('--epochs', type=int, default=3 if name == 'profile' else 200)
            sub.add_argument('--patience', type=int, default=20)
        if name in ['study', 'benchmark']:
            sub.add_argument('--protocol', required=True)
        if name == 'report':
            sub.add_argument('--study', required=True)
    args = parser.parse_args(argv)
    if args.command == 'download':
        from .download import download_baby
        result = download_baby(args.output)
    elif args.command == 'prepare':
        from .data import prepare_dataset
        manifest = prepare_dataset(args.raw, args.output, args.items, args.seed)
        result = {k: manifest[k] for k in ['n_items', 'n_users', 'splits', 'identity']}
    elif args.command == 'environment':
        result = environment()
        write_json(args.output, result)
    elif args.command == 'profile':
        from .data import load_dataset
        from .train import train_run
        from .study import BASE_CONFIG
        data = load_dataset(args.data)
        config = {**BASE_CONFIG, 'epochs': args.epochs, 'patience': args.patience, 'num_threads': args.threads}
        env_path = Path(args.output) / 'environment.json'
        if not env_path.exists():
            write_json(env_path, environment())
        result = train_run(data, data['vision'], data['text'], config, args.output, 101, test=False)
        result = {k: result[k] for k in ['best_epoch', 'resources']}
    elif args.command == 'develop':
        from .study import develop
        result = develop(args.data, args.output, {'num_threads': args.threads,
                         'epochs': args.epochs, 'patience': args.patience})
    elif args.command == 'study':
        from .study import run_study
        result = run_study(args.data, args.protocol, args.output)
    elif args.command == 'benchmark':
        from .benchmark import benchmark_components
        result = benchmark_components(args.data, args.protocol, args.output)
    else:
        from .report import build_report
        result = build_report(args.data, args.study, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
