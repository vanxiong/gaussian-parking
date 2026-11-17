"""Restore pinned upstream sources and the project's local changes.

Existing directories are left untouched. Use --check to verify local revisions,
patches and overlays without downloading or changing anything.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ROOT / 'dependencies'


def restore(check_only=False):
    for source in json.loads((DEPENDENCIES / 'sources.json').read_text()):
        repo = ROOT / source['path']
        if not repo.exists():
            if check_only:
                raise RuntimeError(f'Missing source: {repo}')
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', source['url'], str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', source['revision']], check=True)
            if source.get('patch'):
                subprocess.run(['git', '-C', str(repo), 'apply', str(DEPENDENCIES / source['patch'])], check=True)
            for relative in source['overlays']:
                target = repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(DEPENDENCIES / 'overlays' / source['name'] / relative, target)
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != source['revision']:
            raise RuntimeError(f'Unexpected revision in {repo}; existing source was preserved')
        if source.get('patch'):
            subprocess.run(['git', '-C', str(repo), 'apply', '--reverse', '--check', str(DEPENDENCIES / source['patch'])], check=True)
        for relative in source['overlays']:
            expected = DEPENDENCIES / 'overlays' / source['name'] / relative
            if (repo / relative).read_bytes() != expected.read_bytes():
                raise RuntimeError(f'Overlay differs: {repo / relative}')
        print(f"Verified {source['path']} at {revision}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    restore(parser.parse_args().check)
