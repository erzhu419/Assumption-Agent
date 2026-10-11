"""Matched offline continuations: single-target versus joint-target experiment selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition import world_reference_response
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_discoveryworld_pilot import write_json


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / 'artifacts/discoveryworld_response_ablation_20261008/protocol.json').read_text())
    tasks = {case['task']['random_seed']: case['task'] for case in old['cases']}
    cases = [
        {'id': 'seed0_prefix529', 'task': tasks[0],
         'prefix': 'artifacts/discoveryworld_reference_discrimination_recovery_20261009/seed0_prefix528/reference_discrimination'},
        {'id': 'seed1_prefix408', 'task': tasks[1],
         'prefix': 'artifacts/discoveryworld_predictive_trial_20261008/grammar'},
    ]
    archive = OUT / 'host_sources'
    archive.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__).resolve(), OUT / 'submitted_run.py')
    for name in ('world_reference_recalibration', 'world_reference_response', 'world_response_hypotheses',
                 'world_experiment_choice', 'synthesis', 'probing', 'world_response_constraints'):
        shutil.copyfile(ROOT / f'src/hegel_machine/cognition/{name}.py', archive / f'{name}.py')
    shutil.copyfile(ROOT / 'artifacts/discoveryworld_joint_response_20261009/host_sources/world_response_hypotheses.py',
                    archive / 'single_target_response_hypotheses.py')
    revise = runpy.run_path(str(ROOT / 'artifacts/discoveryworld_observed_settings_20261008/run.py'))['revised_source']
    for case in cases:
        inherited = (ROOT / case['prefix'] / 'agent' / METHOD_PATH).read_text()
        (OUT / (case['id'] + '.py')).write_text(revise(inherited))
    plan = {'runtime': old['runtime'], 'limits': old['limits'], 'provider': None,
            'max_procedure_steps': 64, 'cases': cases,
            'jobs': [{'case': case['id'], 'arm': arm} for case in cases
                     for arm in ('single_target', 'joint_targets')],
            'intervention': 'Same paid prefix, declared bases, current controller, public evidence and joint LP solver. '
                'single_target uses the frozen prior selector: first unfinished target then references; '
                'joint_targets pools all measured unfinished targets and, when old fits are exhausted, '
                'constructs bounded joint fits for each eligible target. Both execute and judge actual object-specific predictions.',
            'scope': 'Opened development continuations, provider=None, matched local window of 64 program calls. '
                'This measures experiment selection and evidence use, not a completed full-budget task comparison. '
                'No target formula or successful setting supplied; no mid-run scientific prompts or source edits.'}
    write_json(OUT / 'protocol.json', plan)
    return plan


def worker(index):
    plan = json.loads((OUT / 'protocol.json').read_text())
    job = plan['jobs'][index]
    case = next(case for case in plan['cases'] if case['id'] == job['case'])
    if job['arm'] == 'single_target':
        spec = importlib.util.spec_from_file_location('hegel_machine.cognition._single_target_response',
            OUT / 'host_sources/single_target_response_hypotheses.py')
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        world_reference_response.propose_response_research = old.propose_response_research
    output = OUT / case['id'] / job['arm']
    workspace = output / 'agent'
    workspace.mkdir(parents=True, exist_ok=False)
    prefix = ROOT / case['prefix']
    prior = json.loads((prefix / 'result.json').read_text())
    agent = DiscoveryWorldAgent(None, workspace, arm='hegel', limits=plan['limits'])
    source = (OUT / (case['id'] + '.py')).read_text()
    with DiscoveryWorldLab(case['task'], plan['runtime'], workspace) as lab:
        agent.lab = lab
        replay = restore_repair(agent, lab, {**prior, 'episode_status': 'running'}, prefix / 'agent')
        write_json(output / 'replay.json', replay)
        agent._execute('write', {'path': 'shared/parent_before_multi_object_response.py',
            'code': (workspace / METHOD_PATH).read_text()}, lab)
        agent._execute('write', {'path': METHOD_PATH, 'code': source}, lab)
        agent.costs.update(observation_cost=lab.environment_steps, compute_cost=agent.costs['research_compute_units'])
        before = deepcopy(agent.result()['costs'])
        initial = json.loads((workspace / 'shared/reference_recalibration.json').read_text())
        history_start = len(initial.get('response_research_history', []))
        recovery_start = len(initial.get('setting_recoveries', []))
        tests_start = len(initial.get('tests', []))
        execution = agent._run_procedure(lab, METHOD_PATH, max_steps=plan['max_procedure_steps'])
        result = agent.result()
        result['costs'].update(observation_cost=lab.environment_steps, compute_cost=result['costs']['research_compute_units'])
        result.update(study_status='finished', execution=execution, prior_costs=before,
            replayed_environment_steps=replay['replayed_environment_steps'] if 'replayed_environment_steps' in replay
                else before['environment_steps'],
            new_costs={key: value - before[key] for key, value in result['costs'].items()})
        write_json(output / 'result.json', result)
        state = json.loads((workspace / 'shared/reference_recalibration.json').read_text())
        history = state.get('response_research_history', [])[history_start:]
        brief = {'case': case['id'], 'arm': job['arm'], 'status': execution['status'],
            'score': result['score'], 'new_costs': result['new_costs'],
            'initial_compatible': sum(c['status'] == 'compatible' for c in initial.get('response_research', {}).get('candidates', [])),
            'initial_hypotheses': len(initial.get('response_research', {}).get('hypotheses', [])),
            'response_rounds': [{k: row[k] for k in ('question_id', 'status', 'candidate_count', 'compatible_count',
                'selected_experiment', 'response_count')} for row in history],
            'new_tests': state.get('tests', [])[tests_start:],
            'pending_test': state.get('test'),
            'physical_answers': [{k:q.get(k) for k in ('id', 'status', 'answer', 'evidence')}
                for q in agent.questions if q['id'] in {t['question']
                    for t in state.get('tests', [])[tests_start:]}],
            'completed_targets': state.get('completed', []),
            'final_hypotheses': len(state.get('response_research', {}).get('hypotheses', [])),
            'joint_search': state.get('response_research', {}).get('joint_parameter_search'),
            'recoveries': state.get('setting_recoveries', [])[recovery_start:],
            'entry_source_changed': (workspace / METHOD_PATH).read_text() != source}
        write_json(output / 'brief.json', brief)
        print(json.dumps({k: brief[k] for k in ('case', 'arm', 'status', 'new_costs')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=int)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if args.job is not None:
        worker(args.job)
    else:
        plan = prepare()
        def launch(index):
            with (OUT / f'job_{index}.log').open('w') as log:
                return subprocess.run([sys.executable, str(Path(__file__).resolve()), '--job', str(index),
                    '--output', str(OUT)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(launch, range(len(plan['jobs']))))
        rows = [json.loads((OUT / job['case'] / job['arm'] / 'brief.json').read_text())
                for job, code in zip(plan['jobs'], codes) if code == 0]
        write_json(OUT / 'summary.json', {'exit_codes': codes, 'runs': rows})
        print(json.dumps({'exit_codes': codes, 'completed_runs': len(rows)}), flush=True)
