"""Replay a recorded proposal with the previous and repaired research scope."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from hegel_machine.cognition import world_response_consequences as consequences
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration
from scripts import consolidate_response_measurement as study
from scripts.run_online_inquiry_llm import write_json
from scripts.run_response_family_recovery import declarations


class RecordedDecisions:
    """Replay the six already-authored decisions; never request a new model."""
    def __init__(self, events, workspace, prefix):
        self.events, self.workspace, self.prefix = events, workspace, prefix
        self.calls = 0

    def next_action(self, request):
        if self.calls == 5:
            records = json.loads((self.workspace / 'laboratory_observations.json').read_text())
            if records != self.prefix:
                raise RuntimeError('The public pre-submission prefix differs from the recorded run')
            write_json(self.workspace.parent / 'prefix_state.json', json.loads(
                (self.workspace / 'shared/reference_recalibration.json').read_text()))
        index = self.calls
        self.calls += 1
        if index >= len(self.events):
            return {'action': 'finish', 'arguments': {'lesson': 'Recorded proposal replay complete.'},
                    'usage': {'input_tokens': 0, 'output_tokens': 0}}
        event = self.events[index]
        return {'action': event['action'], 'arguments': deepcopy(event['arguments']),
                'usage': {'input_tokens': 0, 'output_tokens': 0}}


if __name__ == '__main__':
    output = Path(__file__).resolve().parent
    source = ROOT / 'artifacts/response_controller_adoption_20261009'
    protocol = json.loads((source / 'protocol.json').read_text())
    previous = json.loads((source / 'task_01/live/result.json').read_text())
    events = previous['events'][:6]
    prefix = min(events[-1]['feedback']['new_observation_indices'])
    package = json.loads((source / 'initial_bundle.json').read_text())
    current = Path(recipe.__file__).read_text()
    for hypothesis in declarations(package['files'][study.METHOD_PATH]):
        current = _retain_declaration(current, hypothesis)
    package['files'][study.METHOD_PATH] = current
    spec = importlib.util.spec_from_file_location('hegel_machine.cognition._previous_consequences',
                                                 output / 'previous_consequences.py')
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    repaired = consequences.assess_response_consequences
    study.CASES = {'task_01': protocol['cases']['task_01']['config']}
    write_json(output / 'protocol.json', {'source': str(source), 'shared_limits': protocol['limits'],
        'recorded_decisions': len(events), 'expected_common_prefix': prefix,
        'new_external_model_calls': 0,
        'scope': 'Replay the same five recorded setpoints and same model-authored hypothesis. '
                 'Verify exact public observations before submission in both arms. Compare only '
                 'old versus repaired family/measurement scope handling, preserving candidates, '
                 'paid observations and resource ceilings. A scripted finish returns after the '
                 'proposal; later original model decisions are not replayed on changed evidence.'})
    rows = []
    try:
        for arm, assessment in [('previous', historical.assess_response_consequences), ('revised', repaired)]:
            consequences.assess_response_consequences = assessment
            factory = lambda workspace: RecordedDecisions(events, workspace, previous['observations'][:prefix])
            row = study.run_transfer_case('task_01', arm, package, output, protocol['limits'],
                                          provider_factory=factory)
            result = json.loads((output / 'task_01' / arm / 'result.json').read_text())
            state = json.loads((output / 'task_01' / arm / 'agent/shared/reference_recalibration.json').read_text())
            row.update(new_external_model_calls=0, scripted_decisions=row['costs']['model_calls'],
                post_proposal_actions=row['costs']['environment_steps'] - prefix,
                proposal_result=next(e['feedback'] for e in result['events']
                    if e.get('arguments', {}).get('operator') == 'reference_hypothesis'),
                history=[{'status': h['status'], 'response_count': h['response_count'],
                    'selected_experiment': h.get('selected_experiment'),
                    'family_research': h.get('family_research')} for h in state['response_research_history']])
            rows.append(row)
            write_json(output / 'results.json', rows)
            print(json.dumps({k: row[k] for k in ['arm','status','success','post_proposal_actions',
                                                 'scripted_decisions','new_external_model_calls']}), flush=True)
    finally:
        consequences.assess_response_consequences = repaired
