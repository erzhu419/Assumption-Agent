"""Replay the fixed M4.1 study without overwriting committed evidence."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

COMMIT = '99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c'
parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True, help='Hegel Machine directory in the fixed checkout')
parser.add_argument('--out', type=Path, required=True, help='New evidence directory outside the checkout')
args = parser.parse_args()
repo, out = args.repo.resolve(), args.out.resolve()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
if head != COMMIT:
    raise SystemExit(f'Expected {COMMIT}, got {head}')
checkout = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=repo, text=True).strip()).resolve()
if out.is_relative_to(checkout):
    raise SystemExit('Choose --out outside the git checkout to preserve original evidence.')
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()):
    raise SystemExit('Choose an empty --out directory to preserve earlier audit results.')
sys.path.insert(0, str(repo / 'src'))
spec = importlib.util.spec_from_file_location('committed_m41_study', repo / 'scripts/run_v17_3_m4_1_unified_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
study.EVID_DIR = out
study.main()
print(json.dumps({'commit': head, 'evidence_dir': str(out)}, indent=2))
