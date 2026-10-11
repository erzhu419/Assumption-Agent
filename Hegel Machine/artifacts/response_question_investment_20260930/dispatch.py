"""Run frozen study arms in separate processes; terminal scoring stays separate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--group', required=True)
parser.add_argument('--phase', required=True, choices=('learn', 'evaluate'))
parser.add_argument('--arms', nargs='+')
parser.add_argument('--workers', type=int, default=4)
args = parser.parse_args()
output = BASE / args.group
protocol = json.loads((output / 'protocol.json').read_text())
jobs = [(arm, root) for arm in args.arms or protocol['arms'] for root in protocol['roots']]

def run(job):
    arm, root = job
    command = [sys.executable, str(ROOT / 'scripts/run_nonlinear_mechanism_revision.py'),
               '--output', str(output), '--phase', args.phase, '--arm', arm, '--root', root]
    subprocess.run(command, cwd=ROOT, check=True)

with ThreadPoolExecutor(max_workers=args.workers) as executor:
    list(executor.map(run, jobs))
print({'group': args.group, 'phase': args.phase, 'finished_jobs': len(jobs)}, flush=True)
