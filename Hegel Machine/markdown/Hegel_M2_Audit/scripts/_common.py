"""Location-only bootstrap for the read-only M2 audit scripts."""
import os
from pathlib import Path
import sys

value = os.environ.get('HEGEL_REPO')
if not value:
    raise SystemExit('Set HEGEL_REPO to the fixed checkout Hegel Machine directory, or use run_audit.py --repo.')
HEGEL_REPO = Path(value).expanduser().resolve()
if not (HEGEL_REPO / 'src/hegel_machine').is_dir():
    raise SystemExit('HEGEL_REPO must contain src/hegel_machine; use the Hegel Machine directory, not its parent.')
AUDIT_OUTPUT_DIR = Path(os.environ.get('AUDIT_OUTPUT_DIR', Path(__file__).resolve().parents[1] / 'rerun')).expanduser().resolve()
AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HEGEL_REPO / 'src'))
