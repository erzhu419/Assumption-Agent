"""Fixed six-run comparison of the saved connection repairs, with fresh decisions."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
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
    previous = ROOT / 'artifacts/response_refuted_author_return_20261010'
    prior = json.loads((ROOT / 'artifacts/response_current_public_view_live_20261010/protocol.json').read_text())
    original = json.loads((previous / 'protocol.json').read_text())
    jobs = [{'variant': variant, 'repetition': repetition}
        for repetition in range(1, 4)
        for variant in (['before', 'after'] if repetition % 2 else ['after', 'before'])]
    protocol = {
        'jobs': jobs, 'workers': 3, 'model': prior['model'], 'auth_mode': auth,
        'limits': prior['limits'], 'recorded_prefix_calls': 3,
        'original_cohort': original['original_cohort'],
        'cases': {'hegel_failure': original['cases']['hegel_failure']},
        'runtime_before': 'artifacts/response_research_connections_20261010/runtime_before',
        'runtime_after_overlay': 'artifacts/response_research_connections_20261010/runtime_after',
        'primary_endpoints': ['task_completion', 'c5_requested_setting_confirmation'],
        'scope': 'Both Hegel arms begin in the same world with the same prior declarations and '
            'three recorded decisions. The before runtime already includes the prior refuted-author '
            'and public-view repairs. The after runtime adds all three connection repairs. '
            'Those repairs can change experiments within the third recorded action, so the first '
            'fresh model request need not have the same observations or computational cost. '
            'Report full-run costs and separately the actual boundary for each run. '
            'All subsequent decisions are fresh gpt-6-luna/medium choices, at most21 new calls '
            'within the original24-call allowance. No in-run scientific hints or selective retries. '
            'This tests the combined repair on one development task, not component attribution, '
            'Hegel-versus-Code, independent task generalization, or a unique recursion advantage.'}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    shutil.copy2(previous / 'replay.py', OUT / 'replay.py')
    # The reused runner explicitly loads this module. The worker selects the
    # entire corresponding cognition path before any of its dependencies load.
    for variant in ('before', 'after'):
        target = OUT / ('runtime_' + variant)
        target.mkdir(exist_ok=False)
        shutil.copy2(ROOT / protocol['runtime_before' if variant == 'before'
                                     else 'runtime_after_overlay'] / 'world_llm_inquiry.py',
                     target / 'world_llm_inquiry.py')
    rows = []

    def save():
        payload = {'planned_runs': len(jobs), 'finished_runs': len(rows),
            'status': 'finished' if len(rows) == len(jobs) else 'running', 'runs': rows}
        (directory / 'summary.json').write_text(json.dumps(payload, indent=2))

    def launch(job):
        target = directory / f"repeat_{job['repetition']:02d}" / job['variant']
        log = directory / f"repeat_{job['repetition']:02d}_{job['variant']}.log"
        with log.open('x') as stream:
            process = subprocess.run([sys.executable, str(OUT / 'worker.py'),
                job['variant'], str(target)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if (target / 'summary.json').exists():
            return {**json.loads((target / 'summary.json').read_text()), **job}
        return {**job, 'status': 'runner_failed', 'returncode': process.returncode,
                'log': str(log.relative_to(ROOT))}

    save()
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda item: (item['repetition'], item['variant']))
            save()
            print(json.dumps({key: row.get(key) for key in
                ('variant', 'repetition', 'status', 'success', 'new_model_calls')}), flush=True)


if __name__ == '__main__':
    main()
