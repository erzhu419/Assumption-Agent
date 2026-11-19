"""Audit-only launcher: run frozen entry points with independent evidence roots."""
from pathlib import Path
import importlib.util
import os
import sys

BASE = Path('/workspace/scratch/3f14ad2da0ed')
REPO = Path(os.environ.get('HEGEL_REPO', str(BASE / 'repo_m4_11961f702' / 'Hegel Machine')))
AUDIT = Path(os.environ.get('HEGEL_AUDIT_OUT', str(Path(__file__).resolve().parent)))
kind = sys.argv[1]
entry, output_name = {
    'm31': ('run_v17_1_m3_1_factorized_walk_study.py', 'EVID_DIR'),
    'm4': ('run_v17_2_m4_multiscale_open_discovery_study.py', 'EVIDENCE_DIR'),
}[kind]
spec = importlib.util.spec_from_file_location('audit_frozen_' + kind, REPO / 'scripts' / entry)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
setattr(module, output_name, AUDIT / ('replay_' + kind))
module.main()
