"""Reuse actual provider-log accounting, excluding the seven recorded choices."""
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
source = OUT.parent / 'response_research_reopen_20261010/analyze_live.py'
spec = importlib.util.spec_from_file_location('prior_live_analysis', source)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
prior.OUT = OUT


if __name__ == '__main__':
    cohort = json.loads((OUT / 'live/summary.json').read_text())
    protocol = json.loads((OUT / 'live_protocol.json').read_text())
    result = prior.analyze(cohort, protocol)
    for row in result['runs']:
        row['same_223_boundary'] = (row.get('next_decision') or {}).get('observations') == 223
    (OUT / 'live_analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ('status', 'finished_runs', 'totals')}), flush=True)
