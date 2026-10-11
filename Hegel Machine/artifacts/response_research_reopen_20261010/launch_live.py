"""Fixed fresh-decision comparison after the recorded failed author proposal."""
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
    previous = ROOT / 'artifacts/response_research_connections_live_20261010'
    prior = json.loads((previous / 'protocol.json').read_text())
    jobs = [{'variant': variant, 'repetition': repetition}
            for repetition in range(1, 4)
            for variant in (('before', 'after') if repetition % 2 else ('after', 'before'))]
    protocol = {
        'jobs': jobs, 'workers': 3, 'model': prior['model'], 'auth_mode': auth,
        'limits': prior['limits'], 'recorded_prefix_calls': 7,
        'maximum_new_model_calls_per_run': 17,
        'source': str(previous.relative_to(ROOT) / 'live/repeat_01/after'),
        'recipes': {variant: str((OUT / f'world_reference_recalibration_{variant}.py').relative_to(ROOT))
                    for variant in ('before', 'after')},
        'common_runtime': prior['runtime_after_overlay'],
        'common_runtime_base': prior['runtime_before'],
        'primary_endpoints': ['task_completion', 'c5_requested_setting_confirmation'],
        'scope': 'Both arms are Hegel with the previously integrated connections. Seven identical '
            'recorded decisions include the failed mixed quadratic-interaction proposal. Only '
            'the parent-evidence reopening connection and its embedded method copy differ. '
            'All later decisions are fresh Luna choices within the original total allowance. '
            'Report full costs and each actual first fresh decision boundary. Repetitions are '
            'not paired random seeds; this is one opened development failure, not a Code '
            'comparison or independent task generalization. No in-run scientific hints or retries.'}
    directory = OUT / 'live'
    directory.mkdir(exist_ok=False)
    (OUT / 'live_protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = []

    def save():
        (directory / 'summary.json').write_text(json.dumps({
            'planned_runs': len(jobs), 'finished_runs': len(rows),
            'status': 'finished' if len(rows) == len(jobs) else 'running', 'runs': rows}, indent=2))

    def run(job):
        target = directory / f"repeat_{job['repetition']:02d}" / job['variant']
        target.parent.mkdir(parents=True, exist_ok=True)
        log = directory / f"repeat_{job['repetition']:02d}_{job['variant']}.log"
        with log.open('x') as stream:
            process = subprocess.run([sys.executable, str(OUT / 'replay_boundary.py'),
                job['variant'], 'failure', '--live', '--directory', str(target)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if (target / 'summary.json').exists():
            return {**json.loads((target / 'summary.json').read_text()), **job}
        return {**job, 'status': 'runner_failed', 'returncode': process.returncode,
                'log': str(log.relative_to(ROOT))}

    save()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            rows.append(future.result())
            rows.sort(key=lambda row: (row['repetition'], row['variant']))
            save()
            subprocess.run([sys.executable, str(OUT / 'analyze_live.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
