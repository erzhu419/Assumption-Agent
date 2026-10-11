from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / 'fresh_replication'
protocol = json.loads((OUT / 'protocol.json').read_text())
completion = json.loads((OUT / 'learning_completion.json').read_text())
assert completion == {'arms': list(protocol['arms']), 'roots': list(protocol['roots']), 'complete': True}
def run(job):
    arm, root = job
    path = OUT / 'runs' / arm / root / 'evaluate.log'
    with path.open('a') as log:
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_research_investment_attribution_study.py'),
            '--output', str(OUT), '--phase', 'evaluate', '--arm', arm, '--root', root],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print({'arm': arm, 'root': root, 'terminal_returncode': result.returncode}, flush=True)
    return {'arm': arm, 'root': root, 'returncode': result.returncode}
jobs = [(arm, root) for arm in protocol['arms'] for root in protocol['roots']]
with ThreadPoolExecutor(max_workers=8) as executor:
    records = list(executor.map(run, jobs))
(BASE / 'terminal_jobs.json').write_text(json.dumps(records, indent=2) + '\n')
raise SystemExit(int(any(r['returncode'] for r in records)))
