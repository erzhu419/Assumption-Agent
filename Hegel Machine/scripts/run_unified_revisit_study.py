#!/usr/bin/env python3
"""Compare sequential research and knowledge-triggered revisits under one total budget."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import threading
import time

import run_observation_action_study as common
import run_risk_action_study as risk

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/unified_revisit_study_20260920'
ARMS = ('baseline_v097', 'knowledge_revisit', 'no_successor_knowledge')
STAGES = ('control_risk_source', 'independent_signal_task', 'changed_response_successor')
RESOURCE_KEYS = ('observation_cost', 'measurement_cost', 'compute_cost')


def prepare(output):
    path = output / 'protocol.json'
    if path.exists():
        return
    baseline = 'artifacts/risk_experiment_study_20260920/runtime/src'
    development, main = [95401903], list(range(95602001, 95602009))
    common.save(path, {
        'experiment': 'successor_knowledge_reopens_unresolved_tasks',
        'fixed_at': datetime.now(timezone.utc).isoformat(),
        'cohorts': {'development': development, 'main': main},
        'known_optional_development_root': 95401901,
        'stages': list(STAGES),
        'tasks_by_seed': {str(seed): [{'environment': 'scalar', 'task_index': task,
            'profile': profile, 'seed': seed + 10007 * position}
            for position, (task, profile) in enumerate(((12, 'development'),
                (13, 'development'), (13, 'confirmation')))] for seed in [*development, *main]},
        'arms': {
            'baseline_v097': {'runtime': baseline, 'version': '0.97.0', 'schedule': 'sequential_only'},
            'knowledge_revisit': {'runtime': str((output / 'runtime/src').relative_to(ROOT)),
                'version': '0.98.0', 'schedule': 'knowledge_driven_pending_revisits'},
            'no_successor_knowledge': {'runtime': str((output / 'runtime/src').relative_to(ROOT)),
                'version': '0.98.0', 'schedule': 'matched_revisits_without_successor_knowledge'}},
        'development_runtime': 'src',
        'shared_laboratory_source': baseline + '/hegel_machine/cognition/unknown_task_lab.py',
        'research_config': {'observation_budget': 384, 'compute_budget': 240000,
            'minimum_improvement': 1e-7, 'max_rounds': 4, 'active_probes': False},
        'measurement_budget': 1536,
        'total_research_budget': {'observation_cost': 1152, 'measurement_cost': 4608,
                                  'compute_cost': 720000},
        'maximum_concurrent_research_processes': 4,
        'terminal': {'length': 96, 'trajectories': 16, 'seed_offset': 1000003,
                     'risk_quantile': .9, 'underage_cost': 9., 'overage_cost': 1.},
        'schedule_contract': 'Each stream starts empty and completes all three first visits before '
            'revisiting an unresolved task. The 0.98 session selects and resumes pending research '
            'from actual successor knowledge and the previous purchased evidence. Restore full JSON '
            'between visits. After the knowledge arm finishes, the ablation receives only its revisit '
            'task positions and allocated resource limits, never its candidate, predictions, source '
            'concept identifiers, or candidate scores. Terminal evaluation starts only after all '
            'first visits and revisits in all arms of the cohort have ended.',
        'knowledge_ablation_contract': 'All arms perform the same declared source tasks. The ablation '
            'retains the library immediately after each task first ended and uses that task-local '
            'snapshot for its matched revisit. It can retain knowledge acquired by its own subsequent '
            'revisits of that task, but receives no successor knowledge. candidate=None requests ordinary '
            'resumed research through the existing continuation interface. This is a matched-schedule '
            'knowledge ablation, not an autonomous scheduler baseline. A paired allocation is always '
            'clipped by the ablation stream own remaining total budget; exhaustion is recorded rather '
            'than receiving extra resources or silently dropping a task.',
        'budget_contract': 'The total stream budget is three times the original per-task allowance. '
            'First visits retain the original limits. Subsequent work consumes only the remaining '
            'stream pool; it is not restricted to the old task unused share. Old purchased observations '
            'are restored without buying them again. Source learning, candidate screening and fitting, '
            'failed and successful revisits, and all new purchases count. Session totals equal latest '
            'cumulative reports summed over unique tasks plus separately charged screening computation; '
            'do not add an old report again on top of its resumed cumulative report.',
        'evaluation_contract': 'Evaluate the latest returned Program for every unique task, including '
            'unresolved tasks, using the same independent controls and disturbances per paired stream. '
            'The fixed public quantile .9 and pinball loss are the primary risk endpoint. (9,1) action '
            'loss is ten times pinball, not independent evidence. Record all terminal resources '
            'separately, including the measurement cost of the selected deployment channels.',
        'oracle_contract': 'Only the terminal scorer uses evaluation_oracle, after every learning '
            'and revisit process has finished. Its full-state conditional quantile is a reference; '
            'neither it nor any terminal outcome enters scheduling, hypotheses, stopping, or memory.',
        'comparisons': 'Paired latest-model pinball loss per complete stream and original task stage '
            'against 0.97 and the matched no-successor-knowledge arm. Report all three resource totals, '
            'query counts, CPU and wall time, actual revisit count/cost, source-knowledge usage, and '
            'unresolved outcomes. No binary solved threshold or extra terminal tuning.',
        'scope': 'Eight prospectively fixed stochastic roots of the existing two noisy mechanisms '
            'and parameter profiles, yielding 72 initial task visits plus actual revisits. These are '
            'not eight independently parameterized apparatus. Development root 95401903 and the '
            'optional known root 95401901 were previously exposed and are not prospective evidence.',
        'execution_contract': 'Prepare protocol before outcomes. Use explicitly released current '
            'source for known development, then freeze the final released 0.98 source before any '
            'main learning. Run at most four independent stream processes at once. Preserve failed '
            'attempts and negative results; do not tune from the main cohort.',
        'formal_evidence': False})


def run_path(output, cohort, arm, seed):
    return output / cohort / 'runs' / arm / str(seed) / 'checkpoint.json'


def load(protocol, arm, cohort):
    source = ROOT / (protocol['development_runtime'] if cohort == 'development' and arm != ARMS[0]
                     else protocol['arms'][arm]['runtime'])
    sys.path.insert(0, str(source))
    canonical = 'hegel_machine.cognition.unknown_task_lab'
    spec = importlib.util.spec_from_file_location(canonical, ROOT / protocol['shared_laboratory_source'])
    laboratory = importlib.util.module_from_spec(spec)
    sys.modules[canonical] = laboratory
    spec.loader.exec_module(laboratory)
    session = importlib.import_module('hegel_machine.cognition.unified_session')
    if not Path(session.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError('each runtime must use its own process')
    def factory(item):
        arguments = dict(item)
        arguments.pop('environment')
        return laboratory.UnknownTaskLab(**arguments)
    return session, factory


def complete_learning(protocol, output, cohort):
    for arm in ARMS:
        for seed in protocol['cohorts'][cohort]:
            state = json.loads(run_path(output, cohort, arm, seed).read_text())
            if not state.get('study_learning_complete'):
                raise ValueError('finish all cohort initial visits and revisits before evaluation')


risk.STAGES = STAGES
risk.load = common.load = load
common.ARMS = ARMS
common.run_path = run_path
common.ensure_learning = complete_learning


def _save_learning(path, state, phase, cpu_start):
    state.setdefault('study_cpu', []).append({'phase': phase, 'seconds': time.process_time() - cpu_start})
    common.save(path, state)


def _physical_costs(laboratories):
    return {key: sum(lab.costs[key] for lab in laboratories)
            for key in ('observation_cost', 'measurement_cost', 'queries')}


def _check_purchases(actual, reported):
    for key in ('observation_cost', 'measurement_cost', 'queries'):
        if actual[key] != reported[key]:
            raise ValueError(f'physical purchases differ from visit accounting: {key}')


def import_development(protocol, output, prefix, source_study):
    """Reuse the already purchased, exactly matching known-development prefix."""
    seed = protocol['cohorts']['development'][0]
    queued = json.loads((prefix / 'queued.json').read_text())
    paused = json.loads((prefix / 'paused.json').read_text())
    old = json.loads(source_study.read_text())
    if len(queued['results']) != len(STAGES) or len(paused['results']) != 2:
        raise ValueError('the imported prefix must contain the declared three first visits')
    for position, record in enumerate(queued['results']):
        if record['research'] != old['results'][position]['research']:
            raise ValueError('reuse requires the exact same complete paid research reports')
        if {key: value for key, value in protocol['tasks_by_seed'][str(seed)][position].items()
                if key != 'environment'} != queued['tasks'][position]:
            raise ValueError('the imported tasks must match the fixed development protocol')
    imported = output / 'development' / 'imported_initial'
    common.save(imported / 'queued_original.json', queued)
    common.save(imported / 'paused_original.json', paused)
    common.save(imported / 'reuse.json', {'source_prefix': str(prefix),
        'source_study': str(source_study), 'complete_reports_equal': True,
        'initial_cpu_seconds': None,
        'reason': 'the previous runner measured CPU jointly for four tasks; the three-task prefix cannot be split',
        'old_screening_compute_cost': queued.get('screening_compute_cost', 0),
        'original_scheduler_events': queued.get('scheduler_events', [])})
    for arm in ARMS:
        path = run_path(output, 'development', arm, seed)
        if path.exists():
            raise ValueError('preserve existing development instead of importing over it')
        state = deepcopy(queued)
        state['tasks'] = deepcopy(protocol['tasks_by_seed'][str(seed)])
        state['results'] = deepcopy(old['results'][:len(STAGES)])
        state['evaluation'] = []
        state['study_initial_snapshots'] = {str(position): {
            'research': deepcopy(record['research']),
            'knowledge': deepcopy(paused['knowledge'] if position == 1 else
                                  queued['knowledge'] if position == 2 else None)}
            for position, record in enumerate(state['results'])}
        state.update(study_initial_complete=True, study_learning_complete=arm == ARMS[0],
            study_phase='imported_initial', study_imported_initial=str(imported / 'reuse.json'),
            study_cpu=[{'phase': 'imported_initial', 'seconds': None}])
        if arm == ARMS[0]:
            state.update(pending_revisits=[], revisits=[], scheduler_events=[], screening_compute_cost=0)
        else:
            state['study_rescreen_required'] = True
        state['costs'] = {key: sum(row['research']['costs'][key] for row in state['results'])
                          for key in RESOURCE_KEYS}
        state['costs']['compute_cost'] += state['screening_compute_cost']
        common.save(path, state)


def initial_visits(protocol, output, cohort, arm, seed):
    module, lab_type = load(protocol, arm, cohort)
    path, laboratories = run_path(output, cohort, arm, seed), []
    def factory(item):
        lab = lab_type(item)
        laboratories.append(lab)
        return lab
    state = json.loads(path.read_text()) if path.exists() else None
    cpu_start = time.process_time()
    session = (module.UnifiedResearchSession(state=state, lab_factory=factory) if state else
        module.UnifiedResearchSession(module.ResearchConfig(**protocol['research_config'], seed=seed),
            tasks=protocol['tasks_by_seed'][str(seed)], measurement_budget=protocol['measurement_budget'],
            lab_factory=factory))
    if session.state.pop('study_rescreen_required', False):
        before = session.state['screening_compute_cost']
        session._offer_new_knowledge(2, session.state['results'][2]['research']['concepts_added'])
        session.state['study_imported_rescreen'] = {
            'original_screening_compute_cost': before,
            'additional_compute_cost': session.state['screening_compute_cost'] - before,
            'pending_after_current_screening': len(session.state['pending_revisits'])}
    while len(session.state['results']) < len(STAGES):
        laboratories.clear()
        incoming = deepcopy(session.library.action_experience)
        started = time.perf_counter()
        state = session.run(max_tasks=1)
        record = state['results'][-1]
        physical = _physical_costs(laboratories)
        _check_purchases(physical, record['research']['costs'])
        record.update(learning_wall_seconds=time.perf_counter() - started, actual_costs=physical,
            action_experience_before=incoming, action_experience_after=state['knowledge'].get('action_experience', {}),
            acquisitions=[item for lab in laboratories for item in lab.acquisitions])
        position = str(record['task_position'])
        state.setdefault('study_initial_snapshots', {})[position] = {
            'research': deepcopy(record['research']), 'knowledge': deepcopy(state['knowledge'])}
        state['study_phase'] = 'initial_visits'
        common.save(path, state)
        print(json.dumps({'phase': 'initial', 'arm': arm, 'seed': seed,
            'task_position': record['task_position'], 'status': record['research']['status'],
            'costs': record['research']['costs']}), flush=True)
        session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=factory)
    state = session.to_dict()
    state['study_initial_complete'] = True
    state['study_learning_complete'] = arm == ARMS[0]
    _save_learning(path, state, 'initial', cpu_start)


def knowledge_revisits(protocol, output, cohort, seed):
    arm = 'knowledge_revisit'
    module, lab_type = load(protocol, arm, cohort)
    path, laboratories = run_path(output, cohort, arm, seed), []
    def factory(item):
        lab = lab_type(item)
        laboratories.append(lab)
        return lab
    state = json.loads(path.read_text())
    if not state['study_initial_complete']:
        raise ValueError('revisits follow all initial tasks')
    cpu_start = time.process_time()
    session = module.UnifiedResearchSession(state=state, lab_factory=factory)
    while session.state['pending_revisits']:
        laboratories.clear()
        before = len(session.state['revisits'])
        started = time.perf_counter()
        state = session.run(max_tasks=1)
        if len(state['revisits']) > before:
            visit = state['revisits'][-1]
            position = visit['task_position']
            record = state['results'][position]
            physical = _physical_costs(laboratories)
            _check_purchases(physical, visit['visit_costs'])
            elapsed = time.perf_counter() - started
            visit.update(actual_new_costs=physical, learning_wall_seconds=elapsed,
                         continuation=deepcopy(record['research'].get('continuation')),
                         confirmed_used_sources=list(record['research'].get('used_sources', ())))
            record['learning_wall_seconds'] += elapsed
            for key, value in physical.items():
                record['actual_costs'][key] += value
            record['acquisitions'].extend(item for lab in laboratories for item in lab.acquisitions)
            record['action_experience_after'] = state['knowledge'].get('action_experience', {})
            print(json.dumps({'phase': 'revisit', 'arm': arm, 'seed': seed, 'task_position': position,
                'status': visit['status'], 'visit_costs': visit['visit_costs']}), flush=True)
        state['study_phase'] = 'knowledge_revisits'
        common.save(path, state)
        session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=factory)
    state = session.to_dict()
    state['study_learning_complete'] = True
    _save_learning(path, state, 'revisit', cpu_start)


def ablated_revisits(protocol, output, cohort, seed):
    arm = 'no_successor_knowledge'
    module, lab_type = load(protocol, arm, cohort)
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    path = run_path(output, cohort, arm, seed)
    state = json.loads(path.read_text())
    full = json.loads(run_path(output, cohort, 'knowledge_revisit', seed).read_text())
    if not state['study_initial_complete'] or not full['study_learning_complete']:
        raise ValueError('finish initial visits and the matched knowledge schedule first')
    cpu_start = time.process_time()
    state['pending_revisits'] = []
    matched = state.setdefault('study_matched_revisits', [])
    local_knowledge = state.setdefault('study_revisit_knowledge', {})
    for index in range(len(matched), len(full['revisits'])):
        full_visit = full['revisits'][index]
        position = full_visit['task_position']
        previous = state['results'][position]['research']
        allocation = {key: full_visit['resource_limits'][key] - full_visit['previous']['costs'][key]
                      for key in RESOURCE_KEYS}
        session = module.UnifiedResearchSession(state=state, lab_factory=lab_type)
        remaining = session._remaining()
        allocated = {key: min(allocation[key], remaining[key]) for key in RESOURCE_KEYS}
        assignment = {'matched_visit_index': index, 'task_position': position,
            'requested_allocation': allocation, 'available_pool': remaining, 'allocation': allocated}
        if previous['status'] == 'completed' or not allocated['observation_cost'] or not allocated['compute_cost']:
            assignment.update(executed=False, reason='already_completed' if previous['status'] == 'completed'
                              else 'remaining_resources')
            matched.append(assignment)
            common.save(path, state)
            continue
        key = str(position)
        knowledge = local_knowledge.get(key, state['study_initial_snapshots'][key]['knowledge'])
        library = KnowledgeLibrary.from_dict(knowledge)
        available = [concept.concept_id for concept in library.concepts]
        limits = {name: previous['costs'][name] + allocated[name] for name in RESOURCE_KEYS}
        item = state['tasks'][position]
        started = time.perf_counter()
        with lab_type(item) as laboratory:
            report = UnifiedResearchAgent(replace(session.config, seed=item['seed'],
                observation_budget=limits['observation_cost'], compute_budget=limits['compute_cost']),
                measurement_budget=limits['measurement_cost'], library=library).run(laboratory,
                    previous=previous, continuation_candidate=None)
        elapsed, physical = time.perf_counter() - started, dict(laboratory.costs)
        _check_purchases(physical, report['visit_costs'])
        local_knowledge[key] = library.to_dict()
        record = state['results'][position]
        record['research'] = report
        record['learning_wall_seconds'] += elapsed
        for name, value in physical.items():
            record['actual_costs'][name] += value
        record['acquisitions'].extend(laboratory.acquisitions)
        record['action_experience_after'] = library.action_experience
        state.setdefault('revisits', []).append({'task_position': position, 'candidate': None,
            'previous': previous, 'resource_limits': limits, 'visit_costs': report['visit_costs'],
            'available_concepts': available, 'status': report['status'], 'actual_new_costs': physical,
            'learning_wall_seconds': elapsed, 'continuation': deepcopy(report.get('continuation')),
            'confirmed_used_sources': list(report.get('used_sources', ())), 'matched_visit_index': index})
        assignment.update(executed=True, resource_limits=limits, visit_costs=report['visit_costs'])
        matched.append(assignment)
        state['costs'] = module.UnifiedResearchSession(state=state, lab_factory=lab_type).to_dict()['costs']
        common.save(path, state)
        print(json.dumps({'phase': 'no_successor_revisit', 'arm': arm, 'seed': seed,
            'task_position': position, 'status': report['status'], 'visit_costs': report['visit_costs']}), flush=True)
    state.update(study_learning_complete=True, study_phase='matched_revisits', status='completed')
    _save_learning(path, state, 'matched_revisit', cpu_start)


def summarize(protocol, output, cohort):
    result = {'cohort': cohort, 'scope': protocol['scope'], 'arms': {}}
    for arm in ARMS:
        tasks, visits, costs, terminal, cpu, wall = [], [], Counter(), Counter(), 0., 0.
        screenings, cpu_complete, screening_events = 0, True, []
        for seed in protocol['cohorts'][cohort]:
            state = json.loads(run_path(output, cohort, arm, seed).read_text())
            if not state['study_learning_complete'] or len(state['evaluation']) != len(STAGES):
                raise ValueError('complete learning and evaluation before summary')
            for key, limit in protocol['total_research_budget'].items():
                if state['costs'][key] > limit:
                    raise ValueError('stream exceeded its shared budget')
            costs.update(state['costs'])
            costs['queries'] += sum(row['research']['costs']['queries'] for row in state['results'])
            screenings += state.get('screening_compute_cost', 0)
            screening_events.extend({'replicate_seed': seed, **event}
                for event in state.get('scheduler_events', []) if event['action'] == 'try_successor_knowledge')
            cpu_complete = cpu_complete and all(row['seconds'] is not None for row in state.get('study_cpu', []))
            cpu += sum(row['seconds'] for row in state.get('study_cpu', []) if row['seconds'] is not None)
            scores = {row['task_position']: row for row in state['evaluation']}
            for row in state['results']:
                position, report = row['task_position'], row['research']
                tasks.append({'replicate_seed': seed, 'position': position, 'stage': STAGES[position],
                    'status': report['status'], 'initial_status': state['study_initial_snapshots'][str(position)]['research']['status'],
                    'kind': report['selected_kind'], 'channels': report['selected_channels'],
                    'costs': report['costs'], 'evaluation': scores[position],
                    'used_sources': report.get('used_sources', []),
                    'revisits': sum(visit['task_position'] == position for visit in state.get('revisits', []))})
                terminal.update(scores[position]['costs'])
                wall += row['learning_wall_seconds']
            for visit in state.get('revisits', []):
                continuation = visit.get('continuation') or {}
                offered = continuation.get('source_concept_ids', [])
                used = visit.get('confirmed_used_sources', [])
                visits.append({'replicate_seed': seed, **{key: value for key, value in visit.items()
                    if key not in ('previous', 'candidate', 'continuation')},
                    'offered_sources': offered, 'confirmed_offered_sources_used': sorted(set(offered) & set(used)),
                    'previous_status': visit['previous']['status'],
                    'paid_rows_reused': continuation.get('paid_rows_reused', 0),
                    'fresh_comparisons': [{key: row[key] for key in
                        ('probe_id', 'channels', 'risk_quantile', 'before_loss', 'after_loss', 'observation_rows')}
                        for row in continuation.get('comparisons', [])]})
        def metrics(rows):
            return {key: mean(row['evaluation'][key] for row in rows) for key in risk.METRICS}
        result['arms'][arm] = {'tasks': tasks, 'revisits': visits, 'tasks_count': len(tasks),
            'revisit_count': len(visits), 'learning_costs': dict(costs), 'terminal_costs': dict(terminal),
            'screening_compute_cost': screenings, 'learning_process_cpu_seconds': cpu,
            'learning_process_cpu_complete': cpu_complete,
            'learning_wall_seconds': wall, 'learning_status': dict(Counter(row['status'] for row in tasks)),
            'metrics': metrics(tasks),
            'by_seed': {str(seed): metrics([row for row in tasks if row['replicate_seed'] == seed])
                        for seed in protocol['cohorts'][cohort]},
            'by_stage': {stage: metrics([row for row in tasks if row['stage'] == stage]) for stage in STAGES},
            'by_stage_costs': {stage: {'learning': {key: sum(row['costs'][key] for row in tasks
                if row['stage'] == stage) for key in common.COSTS}, 'terminal': {key: sum(
                    row['evaluation']['costs'][key] for row in tasks if row['stage'] == stage)
                    for key in common.COSTS}} for stage in STAGES},
            'revisit_status': dict(Counter(row['status'] for row in visits)),
            'screening_attempts': len(screening_events),
            'scheduled_offers': sum(row['scheduled'] for row in screening_events),
            'screening_events': screening_events,
            'visits_with_confirmed_offered_source_use': sum(bool(row['confirmed_offered_sources_used']) for row in visits),
            'revisit_costs': {key: sum(row['visit_costs'][key] for row in visits) for key in common.COSTS}}
    result['paired_pinball_changes'] = {}
    learned = result['arms']['knowledge_revisit']
    for arm in ('baseline_v097', 'no_successor_knowledge'):
        control = result['arms'][arm]
        changes = {str(seed): learned['by_seed'][str(seed)]['pinball_loss'] - control['by_seed'][str(seed)]['pinball_loss']
                   for seed in protocol['cohorts'][cohort]}
        result['paired_pinball_changes'][arm] = {'new_minus_control_by_seed': changes,
            'mean_change': mean(changes.values()), 'mean_change_percent': 100 * (
                learned['metrics']['pinball_loss'] / control['metrics']['pinball_loss'] - 1),
            'lower': sum(value < 0 for value in changes.values()), 'equal': sum(value == 0 for value in changes.values()),
            'higher': sum(value > 0 for value in changes.values()),
            'by_stage_mean_change': {stage: learned['by_stage'][stage]['pinball_loss']
                - control['by_stage'][stage]['pinball_loss'] for stage in STAGES}}
    common.save(output / cohort / 'summary.json', result)
    digest = {**result, 'arms': {arm: {key: value for key, value in data.items() if key not in ('tasks', 'revisits')}
                               for arm, data in result['arms'].items()}}
    common.save(output / cohort / 'final_digest.json', digest)
    print(json.dumps(digest, indent=2), flush=True)


def queue(protocol, output, cohort):
    seeds = protocol['cohorts'][cohort]
    execution = {'status': 'running', 'active': [], 'completed': [], 'failed': [],
                 'maximum_concurrent_processes': protocol['maximum_concurrent_research_processes']}
    lock = threading.Lock()
    def run(mode, arm, seed):
        job = [mode, arm, seed]
        log = output / cohort / 'logs' / f'{mode}_{arm}_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            execution['active'].append(job)
            common.save(output / cohort / 'execution.json', execution)
        with log.open('a') as stream:
            code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), mode,
                '--output', str(output), '--cohort', cohort, '--arm', arm, '--seed', str(seed)],
                stdout=stream, stderr=subprocess.STDOUT)
        with lock:
            execution['active'].remove(job)
            execution['failed' if code else 'completed'].append({'job': job, 'exit_code': code})
            common.save(output / cohort / 'execution.json', execution)
        print(json.dumps({'finished': job, 'exit_code': code}), flush=True)
        return code
    stages = (('initial', [(arm, seed) for i, seed in enumerate(seeds)
                  for arm in ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]]),
              ('revisit', [('knowledge_revisit', seed) for seed in seeds]),
              ('ablated_revisit', [('no_successor_knowledge', seed) for seed in seeds]),
              ('evaluate', [(arm, seed) for seed in seeds for arm in ARMS]))
    for mode, jobs in stages:
        execution['phase'] = mode
        with ThreadPoolExecutor(max_workers=protocol['maximum_concurrent_research_processes']) as pool:
            futures = [pool.submit(run, mode, arm, seed) for arm, seed in jobs]
            for future in as_completed(futures):
                if not future.cancelled() and future.result():
                    for pending in futures:
                        pending.cancel()
        if execution['failed']:
            common.save(output / cohort / 'execution.json', {**execution, 'status': 'failed', 'exit_code': 1})
            raise SystemExit(1)
    summarize(protocol, output, cohort)
    common.save(output / cohort / 'execution.json', {**execution, 'status': 'completed', 'exit_code': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'import_development', 'initial', 'revisit', 'ablated_revisit',
                                        'evaluate', 'queue', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cohort', choices=('development', 'main'), default='development')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--source-prefix', type=Path)
    parser.add_argument('--source-study', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == 'prepare':
        prepare(output)
        return
    protocol = json.loads((output / 'protocol.json').read_text())
    if args.mode == 'freeze':
        common.freeze(output)
    elif args.mode == 'import_development':
        import_development(protocol, output, args.source_prefix, args.source_study)
    elif args.mode == 'initial':
        initial_visits(protocol, output, args.cohort, args.arm, args.seed)
    elif args.mode == 'revisit':
        knowledge_revisits(protocol, output, args.cohort, args.seed)
    elif args.mode == 'ablated_revisit':
        ablated_revisits(protocol, output, args.cohort, args.seed)
    elif args.mode == 'evaluate':
        risk.evaluate(protocol, output, args.cohort, args.arm, args.seed)
        path = run_path(output, args.cohort, args.arm, args.seed)
        state = json.loads(path.read_text())
        for score in state['evaluation']:
            score['selected_channels'] = state['results'][score['task_position']]['research']['selected_channels']
        common.save(path, state)
    elif args.mode == 'queue':
        queue(protocol, output, args.cohort)
    else:
        summarize(protocol, output, args.cohort)


if __name__ == '__main__':
    main()
