"""Run retained model-authored bases through empirical prediction trials, no LLM."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import runpy

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_discoveryworld_pilot import SELECTION, write_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'artifacts/discoveryworld_observed_settings_20261008/continued'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--expand', action='store_true')
    args = parser.parse_args()
    previous_dir, output = args.previous.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((previous_dir / 'result.json').read_text())
    plan = json.loads((previous_dir / 'protocol.json').read_text())
    task = next(t for t in json.loads(SELECTION.read_text())['tasks'] if t['id'] == 'dw_04_challenge_s1')
    inherited = (previous_dir / 'agent' / METHOD_PATH).read_text()
    constructor = runpy.run_path(str(ROOT / 'artifacts/discoveryworld_observed_settings_20261008/run.py'))
    source = constructor['revised_source'](inherited)
    (output / 'candidate.py').write_text(source)
    workspace = output / 'agent'
    workspace.mkdir(exist_ok=False)
    # A non-exhausted account lets the existing procedure run. The provider is
    # absent and the estimator-trial argument returns exhaustion without LLMs.
    limits = {**plan['limits'], 'max_model_calls': previous['costs']['model_calls'] + 1}
    write_json(output / 'protocol.json', {'task': task, 'source_result': str(previous_dir / 'result.json'),
        'runtime': plan['runtime'], 'limits': limits, 'provider': None,
        'finite_grammar_expansion': args.expand,
        'scope': 'Opened development task. Reuse all ten model-authored basis declarations and paid '
                 'observations with host empirical-fit and trial selection. '
                 + ('After exhaustion, reuse the old finite refit/composition grammar uniformly on all '
                    'measured variables, retaining its proposals in the source. ' if args.expand else '')
                 + 'No new model calls, no supplied target formula and no assumed activation width. '
                   'This is not autonomous discovery.'})
    agent = DiscoveryWorldAgent(None, workspace, arm='hegel', limits=limits)
    with DiscoveryWorldLab(task, plan['runtime'], workspace) as lab:
        agent.lab = lab
        replay = restore_repair(agent, lab, {**previous, 'episode_status': 'running'}, previous_dir / 'agent')
        write_json(output / 'replay.json', replay)
        agent.costs.update(observation_cost=lab.environment_steps,
                           compute_cost=agent.costs['research_compute_units'])
        agent._execute('write', {'path': METHOD_PATH, 'code': source}, lab)
        before = deepcopy(agent.result()['costs'])
        prefix = lab.environment_steps
        agent._on_checkpoint = lambda data: write_json(output / 'active.json', {
            **data, 'study_status': 'running', 'new_environment_steps': lab.environment_steps - prefix})
        execution = agent._run_procedure(lab, METHOD_PATH, max_steps=512,
                                        query_arguments={} if args.expand else {'estimator_trial': True})
        result = agent.result()
        result['costs'].update(observation_cost=lab.environment_steps,
                              compute_cost=result['costs']['research_compute_units'])
        result.update(study_status='finished', task_id=task['id'], execution=execution,
            prior_costs=before, replayed_environment_steps=prefix,
            new_environment_steps=lab.environment_steps - prefix,
            new_costs={key: value - before[key] for key, value in result['costs'].items()})
        write_json(output / 'result.json', result)
        state = json.loads((workspace / 'shared/reference_recalibration.json').read_text())
        brief = {key: result[key] for key in ('status', 'score', 'new_costs', 'new_environment_steps',
                                              'replayed_environment_steps')}
        brief.update(procedure_status=execution['status'],
            basis_expansions=state.get('basis_expansions', []),
            prediction_selection=state.get('prediction_selection'),
            questions=[q for q in agent.questions if q.get('opened_before_observation', -1) >= prefix],
            final_responses=agent.observations[-2:])
        old_observations = json.loads((previous_dir / 'agent/laboratory_observations.json').read_text())
        tests = list(state.get('tests', []))
        if state.get('test'):
            pending = state['test']
            question = next(q for q in agent.questions if q['id'] == pending['question'])
            tests.append({**pending, 'status': question['status'], 'evidence': question.get('evidence', [])})
        brief['new_physical_tests'] = [test for test in tests
                                      if test['started'] >= len(old_observations)]
        write_json(output / 'brief.json', brief)
        print(json.dumps({key: brief[key] for key in ('status', 'score', 'new_costs', 'new_physical_tests')}, indent=2))


if __name__ == '__main__':
    main()
