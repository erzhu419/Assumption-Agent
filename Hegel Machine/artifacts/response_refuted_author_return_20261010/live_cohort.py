"""Matched continuation from one recorded failed proposal, with no new hints."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from scripts.run_online_inquiry_llm import verify_codex_subscription


def main():
    auth = verify_codex_subscription()
    directory = OUT / 'live'
    directory.mkdir(exist_ok=False)
    jobs = [{'variant': variant, 'repetition': repetition}
        for repetition in range(1, 4)
        for variant in (['before', 'after'] if repetition % 2 else ['after', 'before'])]
    prior = json.loads((ROOT / 'artifacts/response_current_public_view_live_20261010/protocol.json').read_text())
    protocol = {'jobs': jobs, 'workers': 3, 'model': prior['model'], 'auth_mode': auth,
        'limits': prior['limits'], 'recorded_prefix_calls': 3,
        'runtime': 'Common workspace code, archived at launch and held unchanged during this cohort; '
            'inquiry module loaded from saved before/after variants.',
        'scope': 'Both variants replay the same three original Hegel decisions and notes from zero observations. '
            'At the same171-observation refuted-proposal state they differ only in automatic inquiry return. '
            'Further choices come from the same model without a revised assignment or supplied hypothesis. '
            'Original24-call budget includes3recorded calls, so at most21new calls. '
            'Both variants use the historical-public-view repair. This is a development continuation diagnosis, '
            'not a new task discovery benchmark or Hegel versusCode comparison.'}
    (directory / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    for folder in ('src', 'scripts', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            destination = directory / 'frozen_runtime' / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(path.read_text())
    for name in ('replay.py', 'live_cohort.py'):
        (directory / name).write_text((OUT / name).read_text())
    rows = []

    def save():
        (directory / 'summary.json').write_text(json.dumps({'planned_runs': len(jobs),
            'finished_runs': len(rows), 'status': 'finished' if len(rows) == len(jobs) else 'running',
            'runs': rows}, indent=2))

    def launch(job):
        target = directory / f"repeat_{job['repetition']:02d}" / job['variant']
        log = directory / f"repeat_{job['repetition']:02d}_{job['variant']}.log"
        with log.open('x') as stream:
            process = subprocess.run([sys.executable, str(OUT / 'replay.py'), job['variant'],
                'hegel_failure', str(target), '--live'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if (target / 'summary.json').exists():
            result = json.loads((target / 'summary.json').read_text())
            return {**result, **job}
        return {**job, 'status': 'runner_failed', 'returncode': process.returncode, 'log': str(log)}

    save()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: (r['repetition'], r['variant']))
            save()
            print(json.dumps({k: row.get(k) for k in ('variant','repetition','status','success','new_model_calls')}), flush=True)


if __name__ == '__main__':
    main()
