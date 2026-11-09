#!/usr/bin/env python3
"""Package a completed VirtualWorld build with provenance and an audited inventory."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from validate_virtualworld import validate

REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--sources', type=Path, default=REPO / 'outputs/virtualworld-sources')
    parser.add_argument('--archive', action='store_true')
    parser.add_argument('--verification-dir', type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifests = list((root / 'tasks').glob('*/manifest.json'))
    if len(manifests) != 1000: raise ValueError('Refusing to finalize an incomplete build')
    if args.verification_dir:
        shutil.copytree(args.verification_dir, root / 'verification', dirs_exist_ok=True)
    for name in ['scienceworld', 'alfworld', 'alfred']:
        dest = root / 'licenses' / (name + '-LICENSE')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.sources / (name + '-LICENSE'), dest)
    # Synchronize exported clients after implementation validation, before hashing.
    for path in manifests:
        m = json.loads(path.read_text())
        workspace = path.parent / 'workspace'
        for name in ['vw.py', 'sw.py' if m['environment']['benchmark'] == 'scienceworld' else 'alf.py']:
            shutil.copyfile(REPO / 'scripts/virtualworld_client.py', workspace / name)
        shutil.copyfile(REPO / 'src/mixed_pilot_harness/inference/alfworld_runtime.py', workspace / 'native_alfworld.py')
    shutil.copyfile(REPO / 'docs/VIRTUALWORLD.md', root / 'README.md')
    (root / 'build_tools').mkdir(exist_ok=True)
    (root / 'build_tools/README.md').write_text('These scripts are snapshots for use in this mixed-pilot harness checkout.\n'
        'Copy them into scripts/ of the matching checkout; they depend on src/mixed_pilot_harness.\n'
        'The archive provides task payloads and a compatible runtime wheel, not a standalone harness installation.\n')
    for name in ['build_virtualworld.py', 'validate_virtualworld.py', 'virtualworld_client.py', 'finalize_virtualworld.py']:
        dest = root / 'build_tools' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / 'scripts' / name, dest)
    versions = {p: version(p) for p in ['scienceworld', 'alfworld', 'textworld', 'fast-downward-textworld',
                'tatsu', 'jericho', 'numpy', 'networkx', 'hashids', 'mementos', 'termcolor', 'more-itertools']}
    (root / 'runtime').mkdir(exist_ok=True)
    (root / 'runtime/requirements.txt').write_text('\n'.join(f'{k}=={v}' for k, v in versions.items()) + '\n')
    # Distribute the local PDDL-only TextWorld wheel; Fast Downward must match host platform.
    wheels = list((Path.home() / 'Library/Caches/pip/wheels').rglob('textworld-1.7.0-*.whl'))
    if len(wheels) != 1: raise ValueError('Expected one prepared TextWorld PDDL wheel')
    shutil.copyfile(wheels[0], root / 'runtime' / wheels[0].name)
    (root / 'runtime/BUILD.md').write_text(
        'TextWorld 1.7.0 from the PyPI source distribution. Only setup.py _pre_install was replaced with pass.\n'
        'The Inform7 compiler download/mount is irrelevant to ALFWorld PDDL and was skipped.\n'
        'All textworld/ Python sources are unchanged. Wheel platform: macOS arm64, Python 3.12.\n'
        'Install the provided wheel plus runtime/requirements.txt in a compatible environment.\n'
        'fast-downward-textworld 20.6.4 was built from the unmodified PyPI source.\n'
        'Java is required by ScienceWorld. No pretrained model, detector or AI2-THOR renderer is needed.\n')
    provenance = {'dataset': 'VirtualWorld-1000', 'schema_version': '1.0', 'seed': 42,
                  'reference_format': '/Users/liangjianfeng/Data/Router/scienceworld_task',
                  'python': sys.version, 'runtime': versions,
                  'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in args.sources.iterdir() if p.is_file() and not p.name.endswith('.partial')},
                  'upstream_revisions': {name: json.loads((args.sources / (name + '-revision.json')).read_text())['sha']
                                         for name in ['scienceworld', 'alfworld']},
                  'validation': 'All 1000 accepted instances reset and replayed to native score 100 during extraction.',
                  'observation_conditions': 'infer.py uses text observations; shell workspace additionally exposes PDDL initial state.',
                  'sampling_limitations': 'Oracle-success filtering; fixed seed orders candidates, ScienceWorld 1.2.3 can vary across JVMs.'}
    write(root / 'provenance.json', provenance)
    summary = json.loads((root / 'dataset.json').read_text())
    rejections = []
    for path in root.glob('*.rejections.json'): rejections.extend(json.loads(path.read_text()))
    summary['rejected_candidates'] = dict(Counter(r['instance']['benchmark'] for r in rejections))
    summary['native_oracle_replays'] = 1000
    write(root / 'dataset.json', summary)
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name not in {'SHA256SUMS', 'validation_report.json'}
                   and '__pycache__' not in p.parts)
    (root / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' +
                                           str(p.relative_to(root)) + '\n' for p in files))
    report = validate(root)
    write(root / 'validation_report.json', report)
    if args.archive:
        archive = root.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(root.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts: z.write(p, str(p.relative_to(root.parent)))
        write(root.parent / (root.name + '.zip.sha256.json'),
              {'file': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size})
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
