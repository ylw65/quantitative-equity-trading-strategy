"""Build a source ZIP from Git-tracked files, refusing private/generated paths."""
import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.git', '.venv', '__pycache__', '.pytest_cache', 'results', 'data_current'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.is_relative_to(ROOT):
        raise SystemExit('Write the release archive outside the repository.')
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    names = [name for name in names if name]
    if not names:
        raise SystemExit('Stage the reviewed source files before packaging.')
    for name in names:
        p = Path(name)
        if (FORBIDDEN.intersection(p.parts) or name.startswith('data_pipeline/data/')
                or p.name.startswith('.env') or p.suffix in {'.parquet', '.pdf'}):
            raise SystemExit(f'Private or generated file unexpectedly tracked: {name}')
        if not (ROOT / p).is_file():
            raise SystemExit(f'Missing tracked file: {name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.write(ROOT / name, f'{ROOT.name}/{name}')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + '.sha256').write_text(
        f'{digest}  {target.name}\n', encoding='utf-8')
    print(f'Packaged {len(names)} reviewed files: {target.name}; SHA256 {digest}')


if __name__ == '__main__':
    main()
