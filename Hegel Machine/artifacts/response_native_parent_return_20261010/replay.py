"""Replay all original native decisions under the original limits, without APIs."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

variant, output = sys.argv[1], Path(sys.argv[2]).resolve()
protocol = json.loads((output / 'protocol.json').read_text())
runtime = Path(protocol['runtime_roots'][variant])
sys.path[:0] = [str(runtime / 'src'), str(runtime / 'tests'), str(runtime)]
from scripts import run_response_matched_adoption as matched
from scripts.run_response_structure_transfer import source_parts
from test_discoveryworld_agent import ScriptedProvider
from hegel_machine.cognition import world_llm_inquiry, world_reference_recalibration

class RecordedProvider(ScriptedProvider):
    def next_action(self, request):
        decision = super().next_action(request)
        decision['usage'] = {'input_tokens': 0, 'output_tokens': 0,
            'cached_input_tokens': 0, 'total_tokens': 0}
        return decision

steps = [(d['action'], d['arguments']) for d in protocol['recorded_decisions']]
provider = RecordedProvider(steps)
original = json.loads((output / 'original_frozen_bundle.json').read_text())
bundle = matched.prepare_bundle(original) if variant == 'parent_return' else deepcopy(original)
directory = output / variant
directory.mkdir(exist_ok=False)
(directory / 'frozen_bundle.json').write_text(json.dumps(bundle, indent=2) + '\n')
path = matched.study.METHOD_PATH
before, after = original['files'][path], bundle['files'][path]
other = lambda code: [ast.dump(n) for n in ast.parse(code).body if not (
    isinstance(n, ast.FunctionDef) and n.name in ('_extend_hypotheses', 'reference_research_question'))]
assert other(before) == other(after), 'Only embedded parent continuation/helper may change'
assert source_parts(before)[0] == source_parts(after)[0], 'Keep all nine original declarations'
assert (before != after) == (variant == 'parent_return'), 'Bundle preparation not connected'
row = matched.run_case(protocol['case'], protocol['config'], 1, 'hegel', bundle,
    directory, protocol['limits'], provider_factory=lambda _: provider)
row.update(variant=variant, new_llm_calls=0, recorded_decisions_replayed=len(provider.requests),
    unused_recorded_decisions=len(steps)-len(provider.requests),
    host_model_calls_are_replays=True, runtime_module=world_llm_inquiry.__file__,
    parent_runtime_module=world_reference_recalibration.__file__)
(directory / 'replay_summary.json').write_text(json.dumps(row, indent=2) + '\n')
assert row['costs']['model_calls'] == len(provider.requests) <= 24
assert row['costs']['reported_tokens'] == 0
assert Path(world_llm_inquiry.__file__).is_relative_to(runtime)
print(json.dumps({k:row[k] for k in ('variant', 'status', 'success', 'costs',
    'new_llm_calls', 'recorded_decisions_replayed', 'unused_recorded_decisions')}), flush=True)
