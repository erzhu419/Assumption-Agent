"""Fixed replay of the first two recorded native decisions; no model API."""
import json
from pathlib import Path
import sys

variant, output = sys.argv[1], Path(sys.argv[2]).resolve()
protocol = json.loads((output / 'protocol.json').read_text())
runtime = Path(protocol['runtime_roots'][variant])
sys.path[:0] = [str(runtime / 'src'), str(runtime / 'tests'), str(runtime)]
from scripts import run_response_matched_adoption as matched
from test_discoveryworld_agent import ScriptedProvider
from hegel_machine.cognition import world_response_hypotheses as response

class RecordedProvider(ScriptedProvider):
    def next_action(self, request):
        decision = super().next_action(request)
        decision['usage'] = {'input_tokens': 0, 'output_tokens': 0,
            'cached_input_tokens': 0, 'total_tokens': 0}
        return decision

provider = RecordedProvider(protocol['recorded_model_actions'])
bundle = json.loads((output / 'frozen_bundle.json').read_text())
row = matched.run_case(protocol['case'], protocol['config'], 1, 'hegel', bundle,
    output / variant, protocol['limits'], provider_factory=lambda _: provider)
row.update(variant=variant, new_llm_calls=0, recorded_decisions_replayed=len(provider.requests),
    runtime_module=response.__file__, host_model_calls_are_replays=True)
(output / variant / 'replay_summary.json').write_text(json.dumps(row, indent=2) + '\n')
assert len(provider.requests) == 2, 'The fixed diagnostic must replay exactly two decisions'
assert row['costs']['model_calls'] == 2 and row['costs']['reported_tokens'] == 0
assert Path(response.__file__).is_relative_to(runtime), 'Wrong runtime imported'
print(json.dumps({k: row[k] for k in ('variant', 'status', 'success', 'costs',
    'new_llm_calls', 'recorded_decisions_replayed')}), flush=True)
