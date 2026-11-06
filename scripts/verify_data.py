"""Check local snapshot hashes without materialising market observations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'docs/source_manifest.json').read_text())
    expected = [x for x in manifest['files'] if x['path'].startswith('data_current/')]
    if not expected:
        raise SystemExit('No snapshot hashes recorded')
    for item in expected:
        path = ROOT / item['path']
        if not path.is_file():
            raise SystemExit(f'Missing {item["path"]}. See docs/REPRODUCIBILITY.md.')
        if digest(path) != item['sha256']:
            raise SystemExit(f'Snapshot mismatch: {item["path"]}')
    print(f'PASS: {len(expected)} data files match; no market rows materialised.')


if __name__ == '__main__':
    main()
