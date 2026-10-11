"""Frozen three-start response ablation using the existing world executor."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import difflib
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_discoveryworld_pilot import SELECTION, write_json


def prepare():
    constructor = runpy.run_path(str(ROOT / 'artifacts/discoveryworld_observed_settings_20261008/run.py'))
    inherited = ROOT / 'artifacts/discoveryworld_response_uncertainty_20261008/agent/shared/consolidated_method.py'
    source = constructor['revised_source'](inherited.read_text())
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_extend_hypotheses')
    assignments = {target.id: node for node in function.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    start, end = assignments['context'].lineno - 1, assignments['remaining'].lineno - 1
    lines = source.splitlines(keepends=True)
    assert "'operator': 'reference_response'" in ''.join(lines[start:end])
    control = ''.join(lines[:start] + lines[end:])
    ast.parse(control)
    for arm, text in [('with_response', source), ('without_response', control)]:
        (OUT / (arm + '.py')).write_text(text)
    (OUT / 'source_difference.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), control.splitlines(True), fromfile='with_response.py', tofile='without_response.py')))
    modules = ['world_response_hypotheses.py', 'world_reference_response.py',
               'world_reference_expansion.py', 'world_reference_hypothesis.py',
               'world_relation_research.py', 'world_experiment_choice.py',
               'world_reference_recalibration.py', 'synthesis.py', 'probing.py']
    (OUT / 'host_sources').mkdir(exist_ok=True)
    for module in modules:
        (OUT / 'host_sources' / module).write_text((ROOT / 'src/hegel_machine/cognition' / module).read_text())
    previous = ROOT / 'artifacts/discoveryworld_predictive_trial_20261008/grammar'
    old_plan = json.loads((previous / 'protocol.json').read_text())
    registered = {t['id']: t for t in json.loads(SELECTION.read_text())['tasks']}
    cases = [{'id': 'challenge_s0_initial', 'task': registered['dw_04_challenge_s0'], 'prefix': None},
             {'id': 'challenge_s1_initial', 'task': registered['dw_04_challenge_s1'], 'prefix': None},
             {'id': 'challenge_s1_prefix408', 'task': registered['dw_04_challenge_s1'], 'prefix': str(previous)}]
    plan = {'scope': 'Fixed host-method ablation; source and cases chosen before these six executions.',
        'authorship': 'The assistant designed the response-family adapter after inspecting seed1 failures. '
                     'Ten retained bases were previously declared by Luna; fourteen by the old finite grammar. '
                     'The frozen executions receive no manual parameter, setting or prompt changes.',
        'evidence_role': 'Seed0 is original development. Both seed1 starts are opened development diagnostics, '
                         'despite the old task registry evaluation label. Starts from the same world are not independent tasks.',
        'runtime': old_plan['runtime'], 'limits': old_plan['limits'], 'provider': None,
        'max_procedure_steps': 1024, 'cases': cases,
        'jobs': [{'case': case['id'], 'arm': arm} for case in cases
                 for arm in ('with_response', 'without_response')]}
    write_json(OUT / 'protocol.json', plan)
    return plan


def worker(index):
    plan = json.loads((OUT / 'protocol.json').read_text())
    job = plan['jobs'][index]
    case = next(c for c in plan['cases'] if c['id'] == job['case'])
    output = OUT / job['case'] / job['arm']
    workspace = output / 'agent'
    workspace.mkdir(parents=True, exist_ok=False)
    source = (OUT / (job['arm'] + '.py')).read_text()
    limits = deepcopy(plan['limits'])
    agent = DiscoveryWorldAgent(None, workspace, arm='hegel', limits=limits)
    with DiscoveryWorldLab(case['task'], plan['runtime'], workspace) as lab:
        agent.lab = lab
        if case['prefix']:
            previous = Path(case['prefix'])
            prior = json.loads((previous / 'result.json').read_text())
            replay = restore_repair(agent, lab, {**prior, 'episode_status': 'running'}, previous / 'agent')
            write_json(output / 'replay.json', replay)
        else:
            agent._question({'id': 'task', 'question': '\n'.join(t['description']
                for t in lab.public_observation()['taskProgress'])})
        agent._execute('write', {'path': METHOD_PATH, 'code': source}, lab)
        agent.costs.update(observation_cost=lab.environment_steps,
                           compute_cost=agent.costs['research_compute_units'])
        before = deepcopy(agent.result()['costs'])
        prefix = lab.environment_steps
        execution = agent._run_procedure(lab, METHOD_PATH, max_steps=plan['max_procedure_steps'])
        result = agent.result()
        result['costs'].update(observation_cost=lab.environment_steps,
                              compute_cost=result['costs']['research_compute_units'])
        result.update(study_status='finished', case=case['id'], arm=job['arm'],
            execution=execution, prior_costs=before, replayed_environment_steps=prefix,
            new_costs={key: value - before[key] for key, value in result['costs'].items()})
        write_json(output / 'result.json', result)
        state_file = workspace / 'shared/reference_recalibration.json'
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        brief = {'case': case['id'], 'arm': job['arm'], 'status': result['status'],
            'procedure_status': execution['status'], 'score': result['score'],
            'new_costs': result['new_costs'], 'replayed_environment_steps': prefix,
            'entry_source_changed': (workspace / METHOD_PATH).read_text() != source,
            'response_rounds': len(state.get('response_research_history', [])),
            'response_choices': [r['selected_experiment']['prediction'] for r in
                state.get('response_research_history', []) if r.get('selected_experiment')],
            'response_feedback': state.get('response_feedback')}
        write_json(output / 'brief.json', brief)
        return brief


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=int)
    args = parser.parse_args()
    if args.job is not None:
        worker(args.job)
    else:
        plan = prepare()
        def launch(index):
            with (OUT / f'job_{index}.log').open('w') as log:
                return subprocess.run([sys.executable, str(Path(__file__).resolve()), '--job', str(index)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        with ThreadPoolExecutor(max_workers=2) as pool:
            exits = list(pool.map(launch, range(len(plan['jobs']))))
        rows = [json.loads((OUT / job['case'] / job['arm'] / 'brief.json').read_text())
                for job, code in zip(plan['jobs'], exits) if code == 0]
        write_json(OUT / 'summary.json', {'exit_codes': exits, 'runs': rows})
        print(json.dumps({'exit_codes': exits, 'runs': [{k: r[k] for k in
            ('case', 'arm', 'status', 'new_costs', 'response_rounds')} for r in rows]}, indent=2))
