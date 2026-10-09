#!/usr/bin/env python3
"""Fixed-precision paired study of compact revisions and shorter real experiments."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import time

import run_instrument_relation_diagnosis as base
from run_retained_relation_evidence import View, call, packed, renamed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'artifacts/compact_relation_revision_20260919'
COST_KEYS = ('observation_cost', 'measurement_cost', 'compute_cost', 'queries')


def protocol_for(protocol, seed):
    return {**protocol, 'apparatus_seed': seed,
            'initial_current_seeds': ([2401 + i for i in range(4)] if seed == 9013 else
                                      [seed + 2401 + i for i in range(4)]),
            'terminal_seeds': ([1009016 + i for i in range(4)] if seed == 9013 else
                               [seed + 1000003 + i for i in range(4)])}


def research_stage(module, protocol, library, source, input_entry, position, cached=None):
    from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
    from hegel_machine.cognition.competing_object_experiments import plan_competing_object_probe
    started = time.perf_counter()
    stage = protocol['stages'][position]
    seed = protocol['apparatus_seed']
    prefix, outer, radius = stage['prefix'], stage['outer'], stage['radius']
    params = {**protocol['apparatus_settings'], 'relation_radius': radius}
    with module.laboratory_for({'environment': 'objects', 'seed': seed, 'apparatus_seed': seed,
                                'objects': 4, **params}) as lab:
        view = View(lab, prefix, outer, stage['name'])
        agent = base.agent_for(protocol, seed + position, library)
        budget = base.SearchBudget(protocol['budgets_per_case']['compute_budget'])
        actions = []
        original_binding = agent._instrument_binding(lab.spec_for(('fine',)), lab.spec)
        mapping = {key: prefix + key for key in lab.spec.feature_names}
        binding = {prefix + key: renamed(value, mapping) for key, value in original_binding.items()}
        program = base.Program.from_dict(input_entry['program'])
        researcher = RelationIntervalResearch(program, library.meta_priors, budget,
                                              research_program=input_entry.get('research_program'))
        actions.append({'action': 'inherit_relation_intervals', 'intervals': researcher.intervals,
            'evidence_contexts': len(getattr(researcher, 'evidence_contexts', [])),
            'inherited_trajectories': sum(len(item['trajectories']) for item in
                                        getattr(researcher, 'evidence_contexts', []))})
        if cached:
            view.records = cached['paid_records']
            paid = [base.Observation(row['features'], row['target'], tuple(row['history']))
                    for record in view.records for row in record['rows']]
        else:
            paid = [row for _, row in view.acquire(view.random(protocol['initial_current_seeds'][position], 16), 'initial_current')]
        experiments, selected, updates = [], None, 0
        while researcher.parameters and budget.remaining > 1000:
            entries = researcher.alternatives(paid, budget, tolerance=1e-7, experiments=experiments)
            pending_updates = getattr(researcher, 'evidence_updates', [])[updates:]
            actions.extend({'action': 'reopen_inherited_relation_evidence', **item}
                           for item in pending_updates)
            updates += len(pending_updates)
            actions.append({'action': 'condition_inherited_relation_intervals',
                'status': researcher.status, 'candidate_count': len(entries),
                'intervals': deepcopy(researcher.intervals), 'compute_cost_so_far': budget.spent})
            if not entries:
                if researcher.status == 'resolved_to_precision' and budget.spend(len(paid)):
                    loss = sum((program.predict(row) - row.target) ** 2 for row in paid) / len(paid)
                    if loss <= 1e-7:
                        selected = researcher._entry(researcher.values, researcher.intervals[0]['radius_interval'],
                                                     0, loss, len(paid), budget.spent)
                break
            selected = min(entries, key=lambda item: item['training_loss'])
            researcher.adopt(selected)
            if all(item['radius_interval'][1] - item['radius_interval'][0] <= 1e-6
                   for item in researcher.intervals):
                break
            plan = plan_competing_object_probe(view.spec, entries, budget, binding=binding,
                serial=100 * position + len(view.records), length=9, tolerance=1e-7,
                methods=library.program_experiment_methods, task_id=stage['name'],
                questions=researcher.questions(entries))
            if protocol['active_arm'] == 'revised':
                from hegel_machine.cognition.competing_object_experiments import shorten_discriminating_probe
                plan = shorten_discriminating_probe(plan, budget, tolerance=1e-7)
            probe = plan['probe']
            actions.append({'action': 'plan_inherited_relation_experiment',
                **{key: value for key, value in plan.items() if key not in ('probe', 'hypotheses')}})
            if probe is None or lab.costs['observation_cost'] + probe.cost > 512 or \
                    lab.costs['measurement_cost'] + 12 * probe.cost > 6144:
                break
            pairs = view.acquire(probe, 'distinguish_inherited_relation')
            evidence = base.assess_competing_object_probe(plan, pairs, budget, tolerance=1e-7)
            actions.append({'action': 'test_inherited_relation_predictions',
                            'probe_id': probe.probe_id, 'evidence': evidence})
            feedback = library.program_experiment_methods.observe(plan, evidence,
                task_id=stage['name'], command_spec=view.spec, binding=binding, tolerance=1e-7)
            if feedback:
                actions.append(feedback)
            current = [row for _, row in pairs]
            paid.extend(current)
            if evidence['status'] == 'distinguished':
                experiments.append(tuple(current))
        saved = packed(selected) if selected else input_entry
        concept = None
        if selected:
            concept, added = library.learn(selected['candidate'], view.measured, paid)
            prior = next((item for item in library.meta_priors if item['source_concept_id'] == concept), None)
            if prior is None:
                library.meta_priors.append({'source_concept_id': concept, 'status': 'supported',
                    'kind': 'controlled_parameter_research', 'uses': 0, 'counterexamples': [],
                    'source_features': list(view.measured.feature_names),
                    'control_features': list(view.spec.feature_names),
                    'research_program': saved.get('research_program')})
            elif hasattr(agent, '_remember_relation_research'):
                agent._remember_relation_research(prior, selected, view.measured, actions)
        return {'stage': stage['name'], 'status': researcher.status,
            'stop_reason': ('parameter_precision' if researcher.intervals and all(
                item['radius_interval'][1] - item['radius_interval'][0] <= 1e-6
                for item in researcher.intervals) else researcher.status), 'entry': saved,
            'concept_id': concept, 'library': library.to_dict(), 'actions': actions,
            'paid_records': view.records, 'costs': {key: value + (cached['costs'].get(key, 0) if cached else 0)
                for key, value in {**lab.costs, 'compute_cost': budget.spent}.items()},
            'learning_wall_seconds': time.perf_counter() - started +
                (cached['learning_wall_seconds'] if cached else 0),
            'prior_attempt_actions': cached['actions'] if cached else []}


def terminal(module, protocol, result, position):
    stage = protocol['stages'][position]
    seed = protocol['apparatus_seed']
    program = base.Program.from_dict(result['entry']['program'])
    with module.laboratory_for({'environment': 'objects', 'seed': seed,
            'apparatus_seed': seed, 'objects': 4, **protocol['apparatus_settings'],
            'relation_radius': stage['radius']}) as lab:
        view = View(lab, stage['prefix'], stage['outer'], stage['name'])
        random_rows = [row for random_seed in protocol['terminal_seeds'] for _, row in
                       view.acquire(view.random(random_seed, 64), 'terminal_random')]
        random_mse = sum((program.predict(row) - row.target) ** 2
                         for row in random_rows) / len(random_rows)
        # Separate declared stress distribution: measured edge distance on either
        # side of the actual mechanism boundary. It never feeds learning or stop.
        boundary_rows = []
        for offset in protocol['boundary_offsets']:
            points = []
            for phase in protocol['first_counterexample_probe']['phases']:
                positions = ([0., stage['radius'] + offset, 2., -2.]
                             if phase == 'on' else [0., 1.5, 2., -2.])
                point = {stage['prefix'] + f'{role}_o{i}': value for role, values in
                    (('q', [.27, .81, .49, 0.]), ('u', [0., 0., 0., 0.]), ('x', positions))
                    for i, value in enumerate(values)}
                point[stage['prefix'] + 'focus'] = 1.
                points.append(point)
            probe = base.Probe(f'terminal_boundary:{offset}', tuple(points))
            boundary_rows.extend(row for _, row in view.acquire(probe, 'terminal_boundary'))
        boundary_mse = (sum((program.predict(row)-row.target)**2 for row in boundary_rows)
                        /len(boundary_rows)) if boundary_rows else None
        return {'random_mse': random_mse, 'solved': random_mse <= 1e-6,
                'boundary_mse': boundary_mse, 'boundary_status': 'available',
                'random_rows': len(random_rows), 'boundary_rows': len(boundary_rows),
                **lab.costs, 'compute_cost': len(random_rows)+len(boundary_rows)}


def prepare_sources(protocol, output):
    module, _ = base.load(protocol, 'baseline')
    for seed in protocol['apparatus_seeds']:
        path = output / 'sources' / f'{seed}.json'
        if path.exists():
            continue
        source = (json.loads((ROOT / protocol['known_source_file']).read_text()) if seed == 9013
                  else base.make_source(module, protocol, seed))
        base.save(path, source)
        print(json.dumps({'prepared_source': seed, 'costs': source['costs']}), flush=True)


def learn(protocol, output, arm, seed):
    module, source_path = base.load(protocol, arm)
    protocol = {**protocol_for(protocol, seed), 'active_arm': arm}
    source = json.loads((output / 'sources' / f'{seed}.json').read_text())
    folder = output / arm / str(seed)
    if (folder / 'learning_complete.json').exists():
        print(json.dumps({'already_complete': str(folder)})); return
    base.save(folder / 'runtime_info.json', {'source': str(source_path),
        'version': importlib.import_module('hegel_machine').__version__})
    library = base.KnowledgeLibrary.from_dict(source['knowledge'])
    program = base.Program.from_dict(source['measured_program'])
    def relations(node):
        return ([node] if node.op == 'relation_sum' else []) + [x for child in node.children for x in relations(child)]
    relation = relations(program)[0]
    source_entry = {'program': program.to_dict(), 'research_program': {'selection_steps': [{
        'operation': 'revise_relation_condition', 'radius': relation.children[0].value,
        'radius_interval': [.35, 1.2], 'relation': relation.name,
        'predicate': relation.children[1].to_dict()}]}}
    results = []
    for position, stage in enumerate(protocol['stages']):
        path = folder / f'stage_{position}.json'
        if path.exists():
            result = json.loads(path.read_text())
            results.append(result)
            library = base.KnowledgeLibrary.from_dict(result['library'])
            continue
        if position == 0:
            incoming = source_entry
        elif position == 1:
            incoming = {'program': call(results[0]['entry'], results[0]['concept_id'], 'next_', True).to_dict()}
        elif position == 2:
            incoming = {'program': call(results[1]['entry'], results[1]['concept_id']).to_dict()}
        else:
            incoming = {'program': call(results[0]['entry'], results[0]['concept_id']).to_dict()}
        result = research_stage(module, protocol, library, source, incoming, position)
        result['program_nodes'] = base.Program.from_dict(result['entry']['program']).size
        base.save(path, result)
        result = json.loads(path.read_text())
        library = base.KnowledgeLibrary.from_dict(result['library'])
        results.append(result)
        print(json.dumps({'arm': arm, 'seed': seed, 'stage': position,
            'status': result['status'], 'costs': result['costs'],
            'wall': result['learning_wall_seconds'], 'nodes': result['program_nodes']}), flush=True)
    base.save(folder / 'learning_complete.json', {'completed_stages': len(results)})


def evaluate(protocol, output, arm, seed):
    missing = [f'{a}/{s}' for a in ('baseline', 'revised') for s in protocol['apparatus_seeds']
               if not (output / a / str(s) / 'learning_complete.json').exists()]
    if missing:
        raise RuntimeError(f'Learning not complete: {missing}')
    module, _ = base.load(protocol, arm)
    protocol = protocol_for(protocol, seed)
    folder = output / arm / str(seed)
    for position in range(len(protocol['stages'])):
        path = folder / f'stage_{position}.json'
        result = json.loads(path.read_text())
        if 'terminal' not in result:
            result['terminal'] = terminal(module, protocol, result, position)
            base.save(path, result)
        print(json.dumps({'arm': arm, 'seed': seed, 'stage': position,
                          'terminal': result['terminal']}), flush=True)



def grouped_results(arms):
    groups = {}
    for name, arm in arms.items():
        groups[name] = {}
        for label in ('known', 'new', 'all'):
            apparatus = [item for item in arm['apparatus'] if label == 'all' or item['scope'] == label]
            stages = [stage for item in apparatus for stage in item['stages']]
            errors = [stage['terminal']['boundary_mse'] for stage in stages]
            failures = [error for error in errors if error > 1e-6]
            groups[name][label] = {
                'apparatus': len(apparatus), 'random_solved': sum(item['solved'] for item in apparatus),
                'random_stages': len(stages),
                'complete_sequences': sum(item['solved'] == 4 for item in apparatus),
                'costs': {key: sum(item['learning_costs_including_source'][key] for item in apparatus)
                          for key in COST_KEYS},
                'wall': sum(item['cumulative_learning_wall_seconds'] for item in apparatus),
                'max_random_mse': max(stage['terminal']['random_mse'] for stage in stages),
                'boundary_passed': sum(error <= 1e-6 for error in errors),
                'boundary_mse_min': min(errors), 'boundary_mse_max': max(errors),
                'boundary_failed_range': [min(failures), max(failures)] if failures else None,
                'boundary_by_stage': [{'seed': item['apparatus_seed'],
                    'mse': [stage['terminal']['boundary_mse'] for stage in item['stages']]}
                    for item in apparatus],
                'reopened_stages': sum(bool(stage['reopens']) for stage in stages),
                'shortened_probes': sum(len(stage['shortened_probes']) for stage in stages),
                'rows_saved_by_shortening': sum(probe['planned_rows'] - probe['purchased_rows']
                    for stage in stages for probe in stage['shortened_probes'])}
    return groups


def summarize(protocol, output):
    arms = {}
    for arm in ('baseline', 'revised'):
        apparatus = []
        for seed in protocol['apparatus_seeds']:
            source = json.loads((output / 'sources' / f'{seed}.json').read_text())
            stages = []
            for position in range(4):
                item = json.loads((output / arm / str(seed) / f'stage_{position}.json').read_text())
                stages.append({k: item[k] for k in ('stage', 'status', 'stop_reason', 'costs',
                    'learning_wall_seconds', 'program_nodes', 'terminal')} | {
                    'actual_probes': sum(a['action'] == 'test_inherited_relation_predictions' for a in item['actions']),
                    'reopens': [a for a in item['actions'] if a['action'] == 'reopen_inherited_relation_evidence'],
                    'shortened_probes': [a.get('prefix_design') for a in item['actions']
                        if a['action'] == 'plan_inherited_relation_experiment' and a.get('prefix_design')]})
            apparatus.append({'apparatus_seed': seed, 'scope': 'known' if seed == 9013 else 'new',
                'source_costs_once': source['costs'], 'stages': stages,
                'solved': sum(s['terminal']['solved'] for s in stages),
                'learning_costs_including_source': {key: source['costs'].get(key,0)+sum(s['costs'].get(key,0) for s in stages)
                                                   for key in COST_KEYS},
                'cumulative_learning_wall_seconds': sum(s['learning_wall_seconds'] for s in stages)})
        arms[arm] = {'apparatus': apparatus, 'solved': sum(a['solved'] for a in apparatus),
                     'complete_sequences': sum(a['solved'] == 4 for a in apparatus),
                     'learning_costs_including_source': {key:sum(a['learning_costs_including_source'][key] for a in apparatus)
                                                        for key in COST_KEYS},
                     'cumulative_learning_wall_seconds': sum(a['cumulative_learning_wall_seconds'] for a in apparatus)}
    result = {'completed_at': datetime.now(timezone.utc).isoformat(), 'scope': protocol['scope'],
              'status': 'completed', 'arms': arms, 'grouped_results': grouped_results(arms)}
    base.save(output / 'summary.json', result)
    print(json.dumps({a: {k:v for k,v in data.items() if k != 'apparatus'} for a,data in arms.items()},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'learn', 'evaluate', 'summarize'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--arm', choices=('baseline','revised'))
    parser.add_argument('--seed', type=int)
    args=parser.parse_args()
    protocol=json.loads((args.output/'protocol.json').read_text())
    if args.mode == 'prepare': prepare_sources(protocol,args.output)
    elif args.mode == 'learn': learn(protocol,args.output,args.arm,args.seed)
    elif args.mode == 'evaluate': evaluate(protocol,args.output,args.arm,args.seed)
    else: summarize(protocol,args.output)


if __name__ == '__main__':
    main()
