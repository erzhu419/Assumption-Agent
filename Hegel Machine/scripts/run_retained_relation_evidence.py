#!/usr/bin/env python3
"""Known-apparatus development of saved parameter evidence in a new consumer."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import time

import run_instrument_relation_diagnosis as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/retained_relation_evidence_20260914'


def renamed(program, mapping):
    children = tuple(renamed(child, mapping) for child in program.children)
    definition = program.definition
    if definition:
        definition = replace(definition, body=renamed(definition.body, mapping))
    return replace(program, name=mapping.get(program.name, program.name) if program.op == 'feature'
                   else program.name, children=children, definition=definition)


def spec_names(spec, prefix, task):
    context = deepcopy(spec.context)
    if 'object_schema' in context:
        schema = json.loads(context['object_schema'])
        schema['fields'] = {obj: {role: prefix + field for role, field in fields.items()}
                            for obj, fields in schema['fields'].items()}
        context['object_schema'] = json.dumps(schema)
    return replace(spec, task_id=task, context=context,
        feature_names=tuple(prefix + key for key in spec.feature_names),
        feature_bounds={prefix + key: ((1., 1.) if key == 'focus' else value)
                        for key, value in spec.feature_bounds.items()})


def call(entry, source, prefix='', outer=False):
    Program = base.Program
    Definition = importlib.import_module('hegel_machine.cognition.language').Definition
    program = Program.from_dict(entry['program'])
    names = tuple(sorted(program.required_features()))
    def args(node):
        if node.op == 'feature':
            return Program('arg', name=node.name)
        return replace(node, children=tuple(args(child) for child in node.children))
    definition = Definition('retained_' + source, names, args(program), source)
    current = Program('call', name=definition.definition_id, definition=definition,
        children=tuple(Program('feature', name=prefix + name) for name in names))
    if outer:
        quantities = [Program('feature', name=prefix + name) for name in names
                      if name.startswith('measured_q_')]
        current = Program('add', children=(Program('window_sum', lag=2, children=(current,)),
            Program('mul', children=(Program('const', value=.2),
                                    Program('object_sum', children=tuple(quantities))))))
    return current


class View:
    def __init__(self, lab, prefix, outer, task):
        self.lab, self.prefix, self.outer = lab, prefix, outer
        self.spec = spec_names(lab.spec, prefix, task)
        self.measured = spec_names(lab.spec_for(('fine',)), prefix, task)
        self.records = []

    def random(self, seed, length):
        probe = self.lab.random_probe(seed, length=length)
        return base.Probe(probe.probe_id, tuple({self.prefix + k: (1. if k == 'focus' else v)
                            for k, v in point.items()} for point in probe.sequence))

    def acquire(self, probe, purpose):
        raw = base.Probe(probe.probe_id, tuple({key.removeprefix(self.prefix): value
                        for key, value in point.items()} for point in probe.sequence))
        batch = self.lab.acquire(raw, ('fine',))
        history, command_history, targets, pairs = [], [], [], []
        for point, measured in zip(probe.sequence, batch.observations):
            fields = {self.prefix + key: value for key, value in measured.features.items()}
            targets.append(measured.target)
            target = (sum(targets[-2:]) + .2 * sum(value for key, value in measured.features.items()
                if key.startswith('measured_q_'))) if self.outer else measured.target
            row = base.Observation(fields, target, tuple(history))
            command = base.Observation(point, target, tuple(command_history))
            pairs.append((command, row))
            history.append({**fields, '__target__': target})
            command_history.append({**point, '__target__': target})
        self.records.append({'purpose': purpose, 'probe': asdict(probe),
            'rows': [asdict(row) for _, row in pairs],
            'observation_cost': batch.observation_cost, 'measurement_cost': batch.measurement_cost})
        return pairs


def packed(entry):
    return {key: value for key, value in entry.items() if key != 'candidate'} | {
        'program': entry['candidate'].program.to_dict()}


def research_stage(module, protocol, library, source, input_entry, position, cached=None):
    from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
    from hegel_machine.cognition.competing_object_experiments import plan_competing_object_probe
    started = time.perf_counter()
    stage = protocol['stages'][position]
    prefix, outer, radius = stage['prefix'], stage['outer'], stage['radius']
    params = {**protocol['apparatus_settings'], 'relation_radius': radius}
    with module.laboratory_for({'environment': 'objects', 'seed': 9013, 'apparatus_seed': 9013,
                                'objects': 4, **params}) as lab:
        view = View(lab, prefix, outer, stage['name'])
        agent = base.agent_for(protocol, 9013 + position, library)
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
            paid = [row for _, row in view.acquire(view.random(2401 + position, 16), 'initial_current')]
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
    program = base.Program.from_dict(result['entry']['program'])
    with module.laboratory_for({'environment': 'objects', 'seed': 9013, 'apparatus_seed': 9013,
            'objects': 4, **protocol['apparatus_settings'], 'relation_radius': stage['radius']}) as lab:
        view = View(lab, stage['prefix'], stage['outer'], stage['name'])
        rows = [row for offset in range(4) for _, row in view.acquire(
            view.random(1009016 + offset, 64), 'terminal')]
        mse = sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)
        return {'mse': mse, 'solved': mse <= 1e-6, **lab.costs, 'compute_cost': len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=OUT / 'protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--arm', choices=('baseline', 'revised'), required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    module, source_path = base.load(protocol, args.arm)
    output = args.output
    base.save(output / 'runtime_info.json', {'source': str(source_path),
        'version': importlib.import_module('hegel_machine').__version__})
    source = json.loads((ROOT / protocol['source_file']).read_text())
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
        path = output / f'stage_{position}.json'
        cached = json.loads(path.read_text()) if args.resume and path.exists() else None
        if cached and cached.get('concept_id'):
            results.append(cached)
            library = base.KnowledgeLibrary.from_dict(cached['library'])
            continue
        if position == 0:
            incoming = source_entry
        elif position == 1:
            incoming = {'program': call(results[0]['entry'], results[0]['concept_id'], 'next_', True).to_dict()}
        elif position == 2:
            incoming = {'program': call(results[1]['entry'], results[1]['concept_id']).to_dict()}
        else:
            incoming = {'program': call(results[0]['entry'], results[0]['concept_id']).to_dict()}
        result = research_stage(module, protocol, library, source, incoming, position, cached)
        base.save(output / f'stage_{position}.json', result)
        result = json.loads((output / f'stage_{position}.json').read_text())
        library = base.KnowledgeLibrary.from_dict(result['library'])
        results.append(result)
        print(json.dumps({'stage': position, 'status': result['status'], 'costs': result['costs']}), flush=True)
    for position, result in enumerate(results):
        result['terminal'] = terminal(module, protocol, result, position)
        base.save(output / f'stage_{position}.json', result)
    digest = {'scope': protocol['scope'], 'source_costs_once': source['costs'],
        'solved': sum(item['terminal']['solved'] for item in results),
        'stages': [{key: item[key] for key in ('stage', 'status', 'costs', 'terminal', 'learning_wall_seconds')}
                   | {'reopen_count': sum(a['action'] == 'reopen_inherited_relation_evidence' for a in item['actions']),
                      'actual_probes': sum(a['action'] == 'test_inherited_relation_predictions' for a in item['actions'])}
                   for item in results]}
    base.save(output / 'summary.json', digest)
    print(json.dumps(digest), flush=True)

if __name__ == '__main__':
    main()
