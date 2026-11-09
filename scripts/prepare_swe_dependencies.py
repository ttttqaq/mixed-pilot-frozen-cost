#!/usr/bin/env python3
"""Prepare isolated dependency images and validate empty/gold baselines offline. No LLM calls.

Only unmodified base source is used during connected preparation. Reference patches
are applied in disposable, network-disabled evaluation containers, never committed.
Original image tags are never overwritten. See docs/swe-environment-preparation.md.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import re
import signal
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mixed_pilot_harness.inference import swe_runtime as native


def now():
    return datetime.datetime.now().astimezone().isoformat()


@contextlib.contextmanager
def image_manifest(path):
    old = os.environ.pop('SWE_NEO_PREPARED_IMAGES', None)
    if path:
        os.environ['SWE_NEO_PREPARED_IMAGES'] = str(path)
    try:
        yield
    finally:
        os.environ.pop('SWE_NEO_PREPARED_IMAGES', None)
        if old is not None:
            os.environ['SWE_NEO_PREPARED_IMAGES'] = old


def recipe(row):
    """Fetch dependencies using the lockfiles at base_commit and installed toolchains."""
    language = str(row.get('language') or row.get('repo_language') or '').lower()
    commands = row.get('install_config', {}).get('install', [])
    if isinstance(commands, str):
        commands = [commands]
    # Preserve upstream environment choices without reinstalling system toolchains
    # or applying upstream source edits. Those are already present in base images.
    exports = [c for c in commands if c.strip().startswith('export ') and
               not any(s in c for s in ('&&', ';', '\n'))]
    script = ['set -e', *exports, 'export COREPACK_ENABLE_DOWNLOAD_PROMPT=0']
    if language == 'go':
        if row['repo'].lower() == 'gravitational/teleport':
            # hid's cgo bindings require the Linux UAPI headers in Alpine images.
            script += ['if ! test -f /usr/include/linux/hidraw.h; then '
                       'if command -v apk >/dev/null; then apk add --no-cache linux-headers; '
                       'elif command -v apt-get >/dev/null; then apt-get update && apt-get install -y linux-libc-dev; '
                       'else echo "Missing linux/hidraw.h; unsupported package manager" >&2; exit 2; fi; fi']
        script += ['go mod download all']
        if row['repo'].lower() == 'flipt-io/flipt':
            # Required version observed in the offline build's missing-module error.
            script += ['go mod download github.com/aws/aws-sdk-go-v2/service/ecrpublic@v1.23.4']
    elif language == 'rust':
        script += ['cargo fetch']
    elif language in ('python', 'py'):
        pip = [c for c in commands if c.strip().startswith(('pip install ', 'pip3 install '))]
        if pip:
            script += pip
        elif row['repo'].lower() == 'qutebrowser/qutebrowser':
            script += ["python -m pip install $(grep -E '^adblock==' requirements.txt)"]
        else:
            # Pro images ship their installation. Do not invent new package versions.
            raise ValueError('No explicit Python dependency recipe; inspect the offline precheck logs')
    elif language in ('ts', 'js', 'typescript', 'javascript'):
        installs = [c for c in commands if c.strip().startswith(
            ('npm ci', 'npm install', 'pnpm install', 'yarn install'))]
        script += installs or [
            'if test -f pnpm-lock.yaml; then pnpm install --frozen-lockfile; '
            'elif test -f yarn.lock; then yarn install --frozen-lockfile; '
            'elif test -f package-lock.json; then npm ci --no-audit; '
            'else echo "No pinned JS dependency lockfile" >&2; exit 2; fi']
    elif language == 'java':
        script += ['mvn -B -DskipTests test-compile dependency:go-offline']
    elif language == 'kotlin':
        script += ['./gradlew testClasses --no-daemon --console=plain']
    elif language == 'php':
        script += ['composer install --no-interaction --prefer-dist']
    elif language == 'elixir':
        script += ['mix deps.get', 'mix deps.compile']
    elif language == 'julia':
        script += ["julia --project=. -e 'using Pkg; Pkg.instantiate()'",
                   # Pkg.test resolves a separate environment including test extras.
                   # Base-source assertions may fail; readiness is decided offline.
                   "julia --project=. -e 'using Pkg; Pkg.test()' || true"]
    else:
        raise ValueError(f'No dependency recipe for {language!r}; inspect offline precheck logs')
    if row['repo'].lower() == 'meilisearch/meilisearch':
        script += [
            'export MILLI_BENCH_DATASETS_PATH=/opt/swe-bench-datasets',
            'mkdir -p "$MILLI_BENCH_DATASETS_PATH"',
            "sed -nE 's/^const DATASET[^=]*= \\(\"([^\"]+)\", \"([^\"]+)\"\\);/\\1 \\2/p' benchmarks/build.rs | "
            'while read -r name extension; do '
            'curl -fL --retry 3 "https://milli-benchmarks.fra1.digitaloceanspaces.com/datasets/$name.$extension.gz" '
            '-o /tmp/swe-dataset.gz && gzip -dc /tmp/swe-dataset.gz > "$MILLI_BENCH_DATASETS_PATH/$name.$extension" || exit 1; done']
    if row['repo'].lower() == 'nodebb/nodebb':
        script += ['npm install --prefix /opt/swe-plugin-cache --no-audit --package-lock=false nodebb-plugin-imgur@1.0.16',
                   'npm install --prefix /opt/swe-plugin-cache --no-audit --package-lock=false nodebb-plugin-imgur@latest',
                   'npm config set prefer-offline true']
    script += ['env -0 > /tmp/swe-prepared-env']
    return '\n'.join(script) + '\n'


def offline_check(row, image, source, folder, args, progress=None):
    entry = dict(image=image, source_image=source, base_commit=row['base_commit'])
    manifest = folder / 'candidate.json'
    native.write_json(manifest, {'images': {row['instance_id']: entry}})
    checks = {}
    with image_manifest(manifest):
        # Gold must run first: missing dependencies should not trigger two long runs.
        for name, content in [('gold', row['patch']), ('empty', '')]:
            if progress:
                progress(name, checks)
            patch = folder / f'{name}.patch'
            patch.write_text(content)
            value = native.native_evaluate(row, patch, args.cache, folder / name,
                                          args.eval_timeout, args.cpus, args.memory)
            checks[name] = {k: value.get(k) for k in
                            ('status', 'error', 'error_category', 'phase', 'elapsed_seconds')}
            if progress:
                progress(name + '_finished', checks)
            if name == 'gold' and value['status'] != 'passed':
                break
    return checks, (checks.get('gold', {}).get('status') == 'passed' and
                    checks.get('empty', {}).get('status') == 'failed')


def recovered_checks(row, old):
    """Reclassify saved, complete baselines without rerunning tests or changing logs."""
    from mixed_pilot_harness.inference.maintenance import reparse
    if not old.get('logs'):
        return None
    for kind in ('prepared', 'original'):
        root = Path(old['logs']) / kind
        if not all((root / name / 'result.json').exists() for name in ('gold', 'empty')):
            continue
        results = {name: reparse(row, root / name) for name in ('gold', 'empty')}
        if results['gold']['status'] == 'passed' and results['empty']['status'] == 'failed':
            native.write_json(root / 'reclassified-baselines.json', results)
            image = results['gold']['image']
            if native.command(['docker', 'image', 'inspect', image], check=False).returncode == 0:
                return image, {name: {k: v.get(k) for k in ('status', 'error_category')}
                               for name, v in results.items()}
    return None


def prepare(row, source, repo, folder, args):
    commands = recipe(row)
    name = 'swe-deps-' + uuid.uuid4().hex[:16]
    key = hashlib.sha256((source + row['base_commit']).encode()).hexdigest()[:20]
    tag = 'swe-neo-prepared:' + key
    proxy_setup = ''
    if args.proxy and str(row.get('language', '')).lower() in ('java', 'kotlin'):
        proxy = urlsplit(args.proxy)
        # JVM clients do not consistently read HTTP_PROXY/HTTPS_PROXY.
        if not proxy.hostname or not proxy.port or proxy.username:
            raise ValueError('JVM preparation requires a proxy host:port without embedded credentials')
        flags = (f'-Dhttp.proxyHost={proxy.hostname} -Dhttp.proxyPort={proxy.port} '
                 f'-Dhttps.proxyHost={proxy.hostname} -Dhttps.proxyPort={proxy.port} '
                 '-Djava.net.preferIPv6Addresses=false')
        proxy_setup = 'export JAVA_TOOL_OPTIONS=' + shlex.quote(flags) + '\n'
        if row.get('language') == 'java':
            xml = ('<settings><proxies><proxy><id>swe-preparation</id><active>true</active>'
                   f'<protocol>http</protocol><host>{proxy.hostname}</host><port>{proxy.port}</port>'
                   '</proxy></proxies></settings>')
            proxy_setup += 'printf %s ' + shlex.quote(xml) + ' > /tmp/swe-maven-settings.xml\n'
            commands = commands.replace('mvn -B ', 'mvn -s /tmp/swe-maven-settings.xml -U -B ')
    script = ('set -e\nexec </dev/null\n' + proxy_setup + 'git config core.hooksPath /dev/null\ngit reset --hard ' +
              shlex.quote(row['base_commit']) + '\n' + commands +
              '\ngit reset --hard ' + shlex.quote(row['base_commit']) + '\n')
    (folder / 'prepare.sh').write_text(script)
    argv = ['docker', 'create', '--platform', 'linux/amd64', '--network', 'bridge',
            '--name', name, '--cpus', str(args.cpus), '--memory', args.memory,
            '--workdir', repo, '--entrypoint', '/bin/bash']
    for key, value in native.NONINTERACTIVE.items():
        argv += ['--env', key + '=' + value]
    if args.proxy:
        for variable in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
            argv += ['--env', variable + '=' + args.proxy]
        argv += ['--env', 'NO_PROXY=localhost,127.0.0.1,::1',
                 '--env', 'no_proxy=localhost,127.0.0.1,::1']
    argv += [source, '-c', script]
    # No host mounts, Docker socket, reference patches, or model credentials.
    native.command(argv)
    try:
        for attempt in range(1, 4):
            with (folder / 'download.log').open('a') as stream:
                stream.write(f'\n--- preparation attempt {attempt} ---\n'); stream.flush()
                proc = native.run_process(['docker', 'start', '--attach', name], stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=args.prepare_timeout)
            if proc.returncode == 0:
                break
            logs = (folder / 'download.log').read_text(errors='replace')
            if attempt == 3 or not re.search(r'EOF|SSL|TLS|resolve host|name resolution|timed out|EAI_AGAIN', logs, re.I):
                raise RuntimeError(f'Dependency preparation exited {proc.returncode}; see download.log')
            time.sleep(5 * attempt)
        native.command(['docker', 'cp', name + ':/tmp/swe-prepared-env', str(folder / 'environment.bin')])
        # Persist only build environment settings needed by the upstream installer.
        allowed = {'PATH', 'HOME', 'GOPATH', 'GOMODCACHE', 'GOCACHE', 'XDG_CACHE_HOME',
                   'CARGO_HOME', 'RUSTUP_HOME', 'GRADLE_USER_HOME', 'JAVA_HOME',
                   'COMPOSER_HOME', 'VIRTUAL_ENV', 'NODE_PATH', 'PYTHONPATH', 'MILLI_BENCH_DATASETS_PATH'}
        change = []
        for value in (folder / 'environment.bin').read_bytes().decode().split('\0'):
            key, sep, val = value.partition('=')
            if sep and key in allowed:
                change += ['--change', 'ENV ' + key + '=' + json.dumps(val)]
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy'):
            change += ['--change', 'ENV ' + key + '=']
        # Reset the image command: the preparation script must never rerun implicitly.
        change += ['--change', 'CMD ["/bin/bash"]', '--change', 'ENTRYPOINT []']
        native.command(['docker', 'commit', *change, name, tag], timeout=600)
        return tag
    finally:
        native.command(['docker', 'rm', '--force', '--volumes', name], check=False, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('run/inference-cache'))
    parser.add_argument('--proxy', help='Proxy address reachable FROM the container; e.g. http://host.docker.internal:8902')
    parser.add_argument('--only', help='Task ID substring for a targeted repair')
    parser.add_argument('--prepare-timeout', type=int, default=1800)
    parser.add_argument('--eval-timeout', type=int, default=1200)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--memory', default='8g')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--remaining-first', action='store_true', help='Process unstarted tasks before interrupted and failed tasks')
    args = parser.parse_args()
    if min(args.prepare_timeout, args.eval_timeout, args.cpus) < 1:
        parser.error('timeouts and cpus must be positive')
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = json.loads(args.plan.read_text())['tasks']
    selected = [t for t in tasks if not args.only or args.only in t['task_id']]
    if not selected:
        parser.error('No matching tasks')
    report_path = args.output / 'preparation.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'results': {}}
    report.update(runner_pid=os.getpid(), batch_started_at=now(), batch_status='running',
                  selected_tasks=len(selected))
    report.pop('batch_finished_at', None)
    results = report['results']
    def save():
        report.update(updated_at=now(), total_tasks=len(tasks), new_api_cost=0,
                      ready=sum(r['status'] == 'ready' for r in results.values()))
        native.write_json(report_path, report)
        images = {r['instance_id']: {k: r[k] for k in ('image', 'source_image', 'base_commit')}
                  for r in results.values() if r['status'] == 'ready'}
        native.write_json(args.output / 'ready-images.json', {'require_prepared': True, 'images': images})
    def finish():
        if report.get('batch_status') == 'running':
            report.update(batch_status='interrupted', batch_finished_at=now())
            save()
    atexit.register(finish)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    # Reuse real test evidence when a parser fix resolves an earlier error.
    for task in selected:
        old = results.get(task['task_id'], {})
        if old.get('status') not in ('error', 'verifying_offline'):
            continue
        row = json.loads(Path(task['record_path']).read_text())
        recovered = recovered_checks(row, old)
        if recovered:
            image, checks = recovered
            old.update(status='ready', image=image, recovered_from_logs=True, reclassified_checks=checks,
                       previous_error=old.pop('error', None), finished_at=now())
    if args.remaining_first:
        selected.sort(key=lambda t: 0 if t['task_id'] not in results else
                      2 if results[t['task_id']]['status'] == 'error' else 1)
    save()
    for index, task in enumerate(selected, 1):
        old = results.get(task['task_id'], {})
        if old.get('status') == 'ready' and native.command(
                ['docker', 'image', 'inspect', old['image']], check=False).returncode == 0:
            continue
        if old.get('status') == 'error' and not args.retry_failed:
            continue
        row = json.loads(Path(task['record_path']).read_text())
        with image_manifest(None):
            source, repo = native.image_for(row)
        item = dict(task_id=task['task_id'], instance_id=row['instance_id'], source_image=source,
                    base_commit=row['base_commit'], status='checking', started_at=now())
        results[task['task_id']] = item
        if old:
            item['previous_attempts'] = old.get('previous_attempts', []) + [
                {k: v for k, v in old.items() if k != 'previous_attempts'}]
        folder = args.output / task['task_id'] / str(time.time_ns())
        folder.mkdir(parents=True)
        item['logs'] = str(folder)
        save()
        print(f'[{index}/{len(selected)}] checking {task["task_id"]}', flush=True)
        try:
            if native.command(['docker', 'image', 'inspect', source], check=False).returncode:
                raise RuntimeError('Base image not cached yet; run pull_swe_images.py first')
            def progress(stage, checks):
                item.update(active_stage=stage, baseline_progress=checks, progress_at=now())
                save()
            checks, ready = offline_check(row, source, source, folder / 'original', args, progress)
            item['original_checks'] = checks
            if ready:
                item.update(status='ready', image=source)
            else:
                item['status'] = 'downloading_dependencies'
                save()
                image = prepare(row, source, repo, folder, args)
                item.update(image=image, status='verifying_offline')
                save()
                checks, ready = offline_check(row, image, source, folder / 'prepared', args, progress)
                item['prepared_checks'] = checks
                item['status'] = 'ready' if ready else 'error'
                if not ready:
                    item['error'] = 'Prepared image did not pass the offline empty/gold baseline checks; inspect logs'
        except Exception as exc:
            item.update(status='error', error=str(exc))
        item['finished_at'] = now()
        save()
        print(f'[{index}/{len(selected)}] {item["status"]}: {item.get("error", task["task_id"])}', flush=True)
    report.update(batch_status='finished', batch_finished_at=now())
    save()
    return int(any(r['status'] != 'ready' for r in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
