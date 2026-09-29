"""Validate repository-local Markdown file links without downloading anything."""
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

root = Path(__file__).resolve().parents[1]
files = [root / 'README.md', root / 'README_ZH.md', *sorted((root / 'docs').rglob('*.md'))]
broken, checked = [], 0
for path in files:
    if not path.is_file():
        broken.append((str(path.relative_to(root)), 'missing document'))
        continue
    for raw in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
        target = raw.strip().strip('<>')
        if target.startswith('#') or urlparse(target).scheme:
            continue
        target = unquote(target.split('#')[0])
        if target:
            checked += 1
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(root) or not resolved.exists():
                broken.append((str(path.relative_to(root)), target))
for path, target in broken:
    print(f'BROKEN {path}: {target}')
print(f'{len(files)} Markdown files, {checked} local links, {len(broken)} broken')
raise SystemExit(bool(broken))
