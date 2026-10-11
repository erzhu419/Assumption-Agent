"""Known 747 development sequence; independent cohorts are not read here."""
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.unified_session import UnifiedResearchSession, ResearchConfig

protocol = json.loads((ROOT / 'artifacts/relation_interval_confirmation_20260913/protocol.json').read_text())
seed = 74713144600
session = UnifiedResearchSession(ResearchConfig(**protocol['hegel_config'], seed=seed),
    tasks=protocol['tasks_by_seed'][str(seed)], measurement_budget=protocol['measurement_budget'])
times = []
for position in range(6):
    started = monotonic()
    state = session.run(max_tasks=1)
    times.append(monotonic() - started)
    (OUT / 'full_checkpoint.json').write_text(json.dumps(state, indent=2) + '\n')
    if position == 3:
        (OUT / 'full_clean_prefix.json').write_text(json.dumps(state, indent=2) + '\n')
    (OUT / 'full_wall_times.json').write_text(json.dumps(times) + '\n')
    result = state['results'][-1]['research']
    print(json.dumps({'stage': position, 'status': result['status'], 'costs': result['costs'], 'seconds': times[-1]}), flush=True)
    session = UnifiedResearchSession(state=json.loads(json.dumps(state)))
state = session.evaluate()
(OUT / 'full_evaluation.json').write_text(json.dumps(state, indent=2) + '\n')
print(json.dumps(state['evaluation'], indent=2), flush=True)
