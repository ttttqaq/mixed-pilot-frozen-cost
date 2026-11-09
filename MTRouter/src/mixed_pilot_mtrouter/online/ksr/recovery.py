"""Explicit, audited compatibility for an engineering-only runtime repair."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

PACKAGE = 'MTRouter/src/mixed_pilot_mtrouter/online/ksr'
RULES_SOURCE = 'src/mixed_pilot_harness/step_collection/ksr.py'


def method_fingerprint(base, rules, project_root, *, revision=None):
    root = Path(project_root)
    h = hashlib.sha256(base.encode())
    if revision is None:
        sources = [(p.name, p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*.py'))]
        rule_source = (root/RULES_SOURCE).read_bytes()
    else:
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, '--', PACKAGE], cwd=root, text=True).splitlines()
        sources = [(Path(p).name, subprocess.check_output(['git', 'show', revision+':'+p], cwd=root))
                   for p in sorted(paths) if p.endswith('.py')]
        if not sources:
            raise ValueError('Repair ancestor has no KSR source package')
        rule_source = subprocess.check_output(['git', 'show', revision+':'+RULES_SOURCE], cwd=root)
    for name, content in sources:
        h.update(name.encode()); h.update(content)
    h.update(rule_source)
    h.update(json.dumps(rules, sort_keys=True).encode())
    return h.hexdigest()


def prepare_runtime_repair(config, manifest, base_fingerprint, rules, revision):
    """Keep historical cache identity only after reconstructing its exact source hash.

    The CLI flag explicitly opts into the reviewed engineering repair. Data,
    prices, base runtime, references and all method settings must still match.
    No saved manifest, cache, label, checkpoint or ledger is rewritten.
    """
    if not re.fullmatch(r'[0-9a-fA-F]{7,40}', revision):
        raise ValueError('Runtime repair requires an explicit ancestor commit hash')
    root = Path(config['project_root'])
    try:
        ancestor = subprocess.check_output(['git', 'rev-parse', '--verify', revision+'^{commit}'], cwd=root, text=True).strip()
        subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, 'HEAD'], cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Runtime repair source is not an ancestor of the current checkout') from exc
    out = Path(config['output_dir'])
    previous = json.loads((out/'config.resolved.json').read_text())
    saved = json.loads((out/'split_manifest.json').read_text())
    if previous != config:
        raise ValueError('Runtime repair cannot change the saved method configuration')
    for key in ('split_fingerprint', 'reference_fingerprint'):
        if saved.get(key) != manifest.get(key):
            raise ValueError('Runtime repair cannot change data or references: '+key)
    expected = method_fingerprint(base_fingerprint, rules, root, revision=ancestor)
    if saved.get('runtime_fingerprint') != expected:
        raise ValueError('Repair ancestor/base source/prices do not reproduce the saved runtime fingerprint')
    execution = manifest['runtime_fingerprint']
    manifest['execution_runtime_fingerprint'] = execution
    manifest['runtime_fingerprint'] = expected
    return {'time': time.time(), 'ancestor_commit': ancestor,
            'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'preserved_runtime_fingerprint': expected, 'execution_runtime_fingerprint': execution,
            'method_configuration_changed': False, 'artifacts_rewritten': False,
            'scope': 'Teacher error recovery and live-environment budget-stop grading'}
