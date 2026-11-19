#!/usr/bin/env python3
"""Run the fixed-9115 read-only audit without editing source or calling providers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
FIXED_COMMIT = '9115f9adf9dad1f52f106a2ab14babe8b68243dd'
STEP_SCRIPTS = {
    'ast': 'probe_ast.py',
    'confirmation': 'probe_confirmation_training_leak.py',
    'runtime': 'probe_runtime.py',
    'receipts': 'recompute_receipts.py',
    'replay': 'replay_study.py',
    'controls': 'run_controls.py',
}


def check_repo(repo):
    manifest = json.loads((PACKAGE / 'provenance.json').read_text(encoding='utf-8'))
    mismatches = []
    for item in manifest['core_files']:
        path = repo / item['path']
        if not path.is_file():
            mismatches.append({'path': item['path'], 'reason': 'missing'})
        else:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != item['sha256']:
                mismatches.append({'path': item['path'], 'expected': item['sha256'], 'actual': actual})
    head = None
    try:
        git = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
        if git.returncode == 0:
            head = git.stdout.strip()
            if head != FIXED_COMMIT:
                mismatches.append({'git_head': head, 'expected': FIXED_COMMIT})
    except FileNotFoundError:
        pass
    if mismatches:
        raise SystemExit(json.dumps({'fixed_commit_verification_failed': mismatches}, ensure_ascii=False, indent=2))
    return {'expected_commit': FIXED_COMMIT, 'git_head': head, 'verified_core_files': len(manifest['core_files']),
            'git_note': 'No git metadata available; file hashes checked.' if head is None else 'Git commit and core file hashes checked.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=os.environ.get('HEGEL_REPO'),
                        help='Fixed checkout Hegel Machine root; alternatively set HEGEL_REPO.')
    parser.add_argument('--output', type=Path, default=PACKAGE / 'rerun',
                        help='New output directory; bundled results are never overwritten by default.')
    parser.add_argument('--steps', nargs='+', choices=[*STEP_SCRIPTS, 'all'],
                        default=['ast', 'confirmation', 'runtime', 'receipts'])
    parser.add_argument('--python-hash-seed', default='0',
                        help='Declared child-process hash seed; HExA ties in the original code depend on this.')
    parser.add_argument('--check-only', action='store_true', help='Verify fixed commit and hashes without running any scientific control.')
    args = parser.parse_args()
    if args.repo is None:
        parser.error('--repo or HEGEL_REPO is required')
    repo = Path(args.repo).expanduser().resolve()
    if not (repo / 'src/hegel_machine').is_dir() and (repo / 'Hegel Machine/src/hegel_machine').is_dir():
        repo = repo / 'Hegel Machine'
    checked = check_repo(repo)
    print(json.dumps(checked, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return 0
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    steps = list(STEP_SCRIPTS) if 'all' in args.steps else list(dict.fromkeys(args.steps))
    env = dict(os.environ)
    env['HEGEL_REPO'] = str(repo)
    env['AUDIT_OUTPUT_DIR'] = str(output)
    env['PYTHONHASHSEED'] = args.python_hash_seed
    env['PYTHONPATH'] = str(repo / 'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    metadata = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'verification': checked,
                'python': sys.version, 'python_hash_seed': args.python_hash_seed, 'steps': [],
                'scope': 'Read-only verification plus explicitly requested local diagnostics; no providers or network tools are called.'}
    for step in steps:
        script = PACKAGE / 'scripts' / STEP_SCRIPTS[step]
        started = time.perf_counter()
        log = output / f'{step}.log'
        print(f'Running {step}; log: {log}', flush=True)
        with log.open('w', encoding='utf-8') as handle:
            completed = subprocess.run([sys.executable, str(script)], cwd=output, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT, check=False)
        metadata['steps'].append({'step': step, 'returncode': completed.returncode,
                                  'wall_seconds': time.perf_counter() - started, 'log': log.name})
        (output / 'run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if completed.returncode:
            print(f'{step} failed; inspect {log}', file=sys.stderr)
            return completed.returncode
        print(f'Completed {step}', flush=True)
    metadata['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
    (output / 'run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
