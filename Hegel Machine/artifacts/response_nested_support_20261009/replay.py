"""Paired replay of the recorded composite proposal from its paid public state."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition import world_reference_response as execution
from hegel_machine.cognition import world_response_hypotheses as hypotheses
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration
from scripts import consolidate_response_measurement as study
from scripts.run_discoveryworld_prerequisite_check import replay_prefix
from scripts.run_online_inquiry_llm import write_json
from scripts.run_response_family_recovery import declarations


def summarize_analysis(result):
    """Retain scientific decisions without repeating the paid observation log."""
    family = result.get('family_research', {})
    selected = (result.get('selection') or {}).get('selected_experiment')
    support = {key: deepcopy(value) for key, value in result.items()
               if any(word in key for word in ('support', 'submodel', 'nested', 'reduction', 'simplification'))}
    return {
        'response_count': result['response_count'], 'status': result['status'],
        'compute_cost': result['compute_cost'],
        'candidate_families': dict(Counter(c['family'] for c in result['candidates'])),
        'compatible_families': dict(Counter(c['family'] for c in result['candidates']
                                           if c['status'] == 'compatible')),
        'member_families': dict(Counter(h['family'] for h in result['hypotheses'])),
        'selected': ({key: selected.get(key) for key in
                      ('number', 'prediction', 'family', 'candidate_id')}
                     if selected is not None else None),
        'family_statuses': {f['id']: f['status'] for f in family.get('families', [])},
        'panel': family.get('panel'),
        'admitted_signatures': family.get('admitted_signatures'),
        'selected_role': family.get('selected_role'),
        'no_progress_cycles': family.get('no_progress_cycles'),
        'nonlocal_progress': family.get('nonlocal_progress'),
        'experiment_values': family.get('experiment_values'),
        'support_analysis': support,
    }


if __name__ == '__main__':
    output = Path(__file__).resolve().parent
    source = ROOT / 'artifacts/response_controller_adoption_20261009'
    scope = ROOT / 'artifacts/response_revised_scope_20261009'
    protocol = json.loads((source / 'protocol.json').read_text())
    recorded = json.loads((source / 'task_01/live/result.json').read_text())
    proposal = recorded['events'][5]
    prefix = min(proposal['feedback']['new_observation_indices'])
    records = recorded['observations'][:prefix]
    state_path = scope / 'task_01/revised/prefix_state.json'
    state = json.loads(state_path.read_text())
    bundle = json.loads((source / 'initial_bundle.json').read_text())
    initial_declarations = declarations(bundle['files'][study.METHOD_PATH])
    code = Path(recipe.__file__).read_text()
    for declaration in initial_declarations:
        code = _retain_declaration(code, declaration)

    spec = importlib.util.spec_from_file_location(
        'hegel_machine.cognition._before_nested_hypotheses', output / 'before_hypotheses.py')
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    original_proposal = hypotheses.propose_response_research
    original_execution = execution.propose_response_research
    study.CASES = {'task_01': protocol['cases']['task_01']['config']}
    limits = {**protocol['limits'], 'max_model_calls': 0}
    write_json(output / 'protocol.json', {
        'source': str(source), 'case': 'task_01', 'prefix': prefix,
        'prefix_state': str(state_path), 'new_external_model_calls': 0,
        'initial_declaration_ids': [d['id'] for d in initial_declarations],
        'hypothesis': proposal['arguments']['hypothesis'], 'limits': limits,
        'scope': 'Same 160 paid public observations, saved controller state, original declarations, '
                 'and already authored composite proposal. Only the response hypothesis constructor '
                 'differs; physical control and consequence assessment are shared. '
                 'Costs start at proposal submission. No model call or future response is supplied.'})
    rows = []
    try:
        for arm, implementation in [('before', before.propose_response_research),
                                    ('nested_support', original_proposal)]:
            analyses = []

            def observe(state, estimators, path):
                result = implementation(state, estimators, path)
                analyses.append(summarize_analysis(result))
                return result

            hypotheses.propose_response_research = observe
            execution.propose_response_research = observe
            agent = DiscoveryWorldAgent(None, output / arm / 'agent', arm='hegel', limits=limits)
            lab = study.TransferSignalLab('task_01')
            agent.lab = lab
            replay = replay_prefix(lab, {'records': records, 'environment_steps': prefix})
            agent.observations = deepcopy(records)
            agent.costs['environment_steps'] = prefix
            agent._execute('write', {'path': study.METHOD_PATH, 'code': code}, lab)
            write_json(agent.workspace / 'shared/reference_recalibration.json', deepcopy(state))
            feedback = agent._execute('run', deepcopy(proposal['arguments']), lab)
            agent._save_state()
            final = json.loads((agent.workspace / 'shared/reference_recalibration.json').read_text())
            new_tests = final.get('tests', [])[len(state.get('tests', [])):]
            new_responses = [r for r in final.get('response_settings', [])
                             if any(i >= prefix for i in r.get('evidence', []))]
            write_json(output / arm / 'feedback.json', feedback)
            write_json(output / arm / 'analysis.json', analyses)
            write_json(output / arm / 'new_observations.json', agent.observations[prefix:])
            row = {
                'arm': arm, 'status': feedback['status'],
                'success': lab.score()['completedSuccessfully'],
                'new_actions': lab.environment_steps - prefix,
                'new_costs': {**agent.result()['costs'],
                              'environment_steps': lab.environment_steps - prefix},
                'replay': replay, 'completed_targets': final.get('completed', []),
                'new_tests': [{key: t.get(key) for key in
                              ('number', 'prediction', 'family', 'status', 'question')}
                             for t in new_tests],
                'new_responses': new_responses,
                'retained_declaration_ids': [d['id'] for d in declarations(
                    (agent.workspace / study.METHOD_PATH).read_text())],
                'analyses': analyses,
            }
            rows.append(row)
            write_json(output / 'results.json', rows)
            print(json.dumps({key: row[key] for key in
                              ('arm', 'status', 'success', 'new_actions')}), flush=True)
    finally:
        hypotheses.propose_response_research = original_proposal
        execution.propose_response_research = original_execution
