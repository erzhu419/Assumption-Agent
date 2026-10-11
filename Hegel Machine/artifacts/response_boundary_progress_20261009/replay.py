"""Paired current/frozen research from two already-paid proposal prefixes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition import world_reference_response as execution
from hegel_machine.cognition import world_reference_hypothesis as submission
from hegel_machine.cognition import world_response_hypotheses as hypotheses
from hegel_machine.cognition import world_response_consequences as consequences
from hegel_machine.cognition import world_response_constraints as constraints
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from scripts import consolidate_response_measurement as study
from scripts.run_discoveryworld_prerequisite_check import replay_prefix
from scripts.run_online_inquiry_llm import write_json
from scripts.run_response_family_recovery import declarations

OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'artifacts/response_controller_adoption_20261009'


def load_snapshot(name):
    spec = importlib.util.spec_from_file_location(
        'hegel_machine.cognition._boundary_before_' + name, OUTPUT / ('before_' + name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(case, event_index, state_path, output):
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    recorded = json.loads((SOURCE / case / 'live/result.json').read_text())
    proposal = recorded['events'][event_index]
    prefix = min(proposal['feedback']['new_observation_indices'])
    records = recorded['observations'][:prefix]
    state = json.loads(state_path.read_text())
    bundle = json.loads((SOURCE / 'initial_bundle.json').read_text())
    initial = declarations(bundle['files'][study.METHOD_PATH])
    code = Path(recipe.__file__).read_text()
    for declaration in initial:
        code = submission._retain_declaration(code, declaration)
    before_constraints = load_snapshot('constraints')
    before_consequences = load_snapshot('consequences')
    before_hypotheses = load_snapshot('hypotheses')
    before_consequences.query_response_family = before_constraints.query_response_family
    originals = (hypotheses.propose_response_research, execution.propose_response_research,
                 consequences.assess_response_consequences, constraints.query_response_family,
                 constraints.fit_response_constraints, submission.query_response_family)
    study.CASES = {case: protocol['cases'][case]['config']}
    limits = {**protocol['limits'], 'max_model_calls': 0}
    write_json(output / 'protocol.json', {
        'source': str(SOURCE), 'case': case, 'prefix': prefix,
        'prefix_state': str(state_path), 'new_external_model_calls': 0,
        'initial_declaration_ids': [d['id'] for d in initial],
        'hypothesis': proposal['arguments']['hypothesis'], 'limits': limits,
        'scope': 'Same paid public prefix, saved controller state, original nine declarations, '
                 'and recorded model proposal. Frozen hypothesis/consequence/constraint source '
                 'versus current source; same physical controller. Costs start at submission. '
                 'No model calls, new candidate hints, or future responses are supplied.'})
    rows = []
    try:
        for arm in ('before', 'boundary_progress'):
            if arm == 'before':
                implementation = before_hypotheses.propose_response_research
                consequences.assess_response_consequences = before_consequences.assess_response_consequences
                constraints.query_response_family = before_constraints.query_response_family
                constraints.fit_response_constraints = before_constraints.fit_response_constraints
                submission.query_response_family = before_constraints.query_response_family
            else:
                implementation = originals[0]
                consequences.assess_response_consequences = originals[2]
                constraints.query_response_family = originals[3]
                constraints.fit_response_constraints = originals[4]
                submission.query_response_family = originals[5]
            analyses = []

            def observe(state, estimators, path):
                result = implementation(state, estimators, path)
                family = result.get('family_research', {})
                selected = (result.get('selection') or {}).get('selected_experiment')
                analyses.append({'response_count': result['response_count'],
                    'status': result['status'], 'compute_cost': result['compute_cost'],
                    'selected': deepcopy(selected), 'family_research': deepcopy(family),
                    'conditional_simplification': deepcopy(result.get('conditional_simplification'))})
                return result

            hypotheses.propose_response_research = observe
            execution.propose_response_research = observe
            agent = DiscoveryWorldAgent(None, output / arm / 'agent', arm='hegel', limits=limits)
            lab = study.TransferSignalLab(case)
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
            write_json(output / arm / 'feedback.json', feedback)
            write_json(output / arm / 'analysis.json', analyses)
            write_json(output / arm / 'new_observations.json', agent.observations[prefix:])
            row = {'arm': arm, 'status': feedback['status'],
                'success': lab.score()['completedSuccessfully'],
                'new_actions': lab.environment_steps - prefix,
                'new_costs': {**agent.result()['costs'], 'environment_steps': lab.environment_steps - prefix},
                'replay': replay, 'completed_targets': final.get('completed', []),
                'new_tests': [{key: t.get(key) for key in
                              ('number', 'prediction', 'family', 'status', 'question')} for t in new_tests],
                'new_responses': [r for r in final.get('response_settings', [])
                                  if any(i >= prefix for i in r.get('evidence', []))],
                'retained_declaration_ids': [d['id'] for d in declarations(
                    (agent.workspace / study.METHOD_PATH).read_text())]}
            rows.append(row)
            write_json(output / 'results.json', rows)
            print(json.dumps({'case': case, **{key: row[key] for key in
                              ('arm', 'status', 'success', 'new_actions')}}), flush=True)
    finally:
        (hypotheses.propose_response_research, execution.propose_response_research,
         consequences.assess_response_consequences, constraints.query_response_family,
         constraints.fit_response_constraints, submission.query_response_family) = originals


if __name__ == '__main__':
    run_case('task_01', 5,
             ROOT / 'artifacts/response_revised_scope_20261009/task_01/revised/prefix_state.json', OUTPUT)
    run_case('task_02', 2,
             ROOT / 'artifacts/response_nested_support_20261009/success_control/prefix_state.json',
             OUTPUT / 'success_control')
