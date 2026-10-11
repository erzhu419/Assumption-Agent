"""Retest the shared current-object view under the previous scientific protocol."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from scripts import run_response_matched_adoption as matched

if __name__ == '__main__':
    prior = ROOT / 'artifacts/response_parent_return_live_20261010'
    protocol = json.loads((prior / 'protocol.json').read_text())
    output = ROOT / 'artifacts/response_current_public_view_live_20261010'
    result = matched.run(output,
        cases={'changed_mechanism': protocol['cases']['changed_mechanism']},
        bundle=json.loads((prior / 'frozen_bundle.json').read_text()),
        knowledge_scope=protocol['initial_knowledge'])
    print(json.dumps(result, indent=2), flush=True)
