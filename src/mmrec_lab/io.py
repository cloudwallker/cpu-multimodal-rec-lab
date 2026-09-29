"""Small, deterministic artifact utilities."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported JSON type: {type(value)}')


def canonical_hash(value):
    encoded = json.dumps(value, default=json_default, sort_keys=True,
                         separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, default=json_default, ensure_ascii=False,
                         indent=2, sort_keys=True, allow_nan=False) + '\n'
    temp = path.with_name(path.name + '.part')
    temp.write_text(encoded, encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def code_hash():
    root = Path(__file__).parent
    return canonical_hash({p.name: file_hash(p) for p in sorted(root.glob('*.py'))})
