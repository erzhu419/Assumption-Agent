"""The public planner must produce a frozen, fundable reset question."""
from copy import deepcopy
from types import SimpleNamespace

from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_dependency_methods import propose_partial_role_experiment
from hegel_machine.cognition.object_induction import assess_position_experiment, position_owner_predictions
from hegel_machine.cognition.object_interventions import plan_paid_position_contrast, plan_position_owner_followup


def scene():
    qfields = ['qa', 'qb', 'qc']
    controls = ['selector', 'drive', *qfields, 'xa', 'xb', 'xc']
    spec = SimpleNamespace(feature_bounds={c: (-1., 1.) for c in controls}, max_probe_length=4)
    model = {'dynamic_fields': qfields, 'quantity_controls': {q: q for q in qfields},
             'position_candidates': {x: {'control': x, 'quantities': qfields} for x in controls[-3:]},
             'drive_links': {'drive': {'quantity': 'qa'}}, 'focus_control': 'selector',
             'focus_candidates': {'fa': {'quantities': qfields}, 'fb': {'quantities': qfields}},
             'raw_sensor_features': [*qfields, *controls[-3:]]}
    zero = {x: 0. for x in controls}
    def record(name, point, history=()):
        before = {c: point[c] for c in model['raw_sensor_features']}
        return {'probe_id': name, 'position': len(history), 'controls': point, 'history': list(history),
                'before': before, 'after': before.copy(), 'target': 0.}
    baseline = {**zero, 'xa': .1, 'xb': .5, 'xc': -.25}
    rows = [record('flat', zero), record('eligible', baseline)]
    def schema(xs):
        return {'fields': {q: {'q': q, 'x': x} for q, x in zip(qfields, xs)}}
    assignments = [schema(['xa', 'xb', 'xc']), schema(['xa', 'xc', 'xb'])]
    return spec, model, rows, assignments, record


def test_reset_design_quotes_actual_missing_corners_and_keeps_non_sampled_owners():
    spec, model, rows, assignments, record = scene()
    original = deepcopy(rows)
    def plan(limit):
        return plan_paid_position_contrast(spec, model, rows, SearchBudget(500), schemas=assignments,
                                           remaining_rows=limit)
    first = plan(3)
    assert first['status'] == 'proposed_position_experiment' and first['compute_cost'] > 0
    assert first['baseline_ref'] == {'probe_id': 'eligible', 'position': 0}
    assert len(set(first['alternative_owners'])) == 2 and len(first['owner_predictions']) == 3
    assert len({tuple(p['allowed_foreign_quantities']) for q,p in first['owner_predictions'].items()
                if q in first['alternative_owners']}) == 2
    assert first['field'] in ('xb', 'xc') and first['required_rows'] == 3 and rows == original
    changed = first['corners'][1]
    rows.append(record('same_controls_wrong_state', changed, [rows[0]['controls']]))
    assert plan(2)['status'] == 'complete_position_experiment_unfunded'
    rows.append(record('already_paid_reset', changed))
    result = plan(2)
    assert result['corners'] == first['corners'] and result['missing_corner_indices'] == [2, 3]
    assert plan_paid_position_contrast(spec, model, rows, SearchBudget(0),
                                       schemas=assignments)['status'] == 'position_planning_compute_exhausted'


def test_common_flat_selector_research_moves_to_position_and_resumes_original_square(monkeypatch):
    from hegel_machine.cognition.object_dependency_methods import ObjectDependencyMethods

    spec, model, rows, schemas, record = scene()
    def flat(self, *args, **kwargs):
        self.evidence = [{'status': 'uninformative_pilot'}]
        return None, {'purpose': 'measured_dependency_methods_lack_contrast'}
    monkeypatch.setattr(ObjectDependencyMethods, 'propose', flat)
    model['schema_alternatives'] = {'schemas': schemas}
    def call(available, previous=None):
        return propose_partial_role_experiment(spec, model, rows, selected_values={'fa': 0., 'fb': 1.},
            compute_limit=2000, remaining_rows=available, previous=previous)
    first = call(3)
    assert first['relation']['kind'] == 'position_membership_contrast' and len(first['actions']) == 3
    assert all(len(action['sequence']) == 1 for action in first['actions'])
    rows.append(record('p1', first['actions'][0]['sequence'][0]))
    partial = call(1, first)
    assert partial['status'] == 'scoped_unknown' and partial['required_observation_rows'] == 2
    assert not partial['actions'] and 'completed_relation' not in partial
    rest = call(2, partial)
    assert rest['actions'] == first['actions'][1:]
    rows.extend(record(f'p{i+2}', action['sequence'][0]) for i,action in enumerate(rest['actions']))
    finished = call(0, rest)
    assert finished['status'] == 'position_domain_unchanged' and not finished.get('actions')
    assert finished['completed_relation']['retained_owners'] == model['dynamic_fields']


def test_focus_only_disagreements_and_flat_backgrounds_do_not_dispatch_a_position_probe():
    spec, model, rows, assignments, _ = scene()
    first = assignments[0]
    second = deepcopy(first)
    for group in second['fields'].values():
        group['f'] = 'different_focus'
    response = plan_paid_position_contrast(spec, model, rows, SearchBudget(500), schemas=[first, second])
    assert response['status'] == 'no_competing_position_predictions' and 'corners' not in response
    flat = plan_paid_position_contrast(spec, model, rows[:1], SearchBudget(500), schemas=assignments)
    assert flat['status'] == 'no_paid_separated_reset' and 'corners' not in flat


def test_followup_holds_the_paid_backgrounds_and_distinguishes_both_remaining_owners():
    spec, model, rows, schemas, record = scene()
    model['model_id'] = 'parent'
    base = rows[1]['controls']
    moved = {**base, 'xb': -1.}
    square = [base, {**base, 'qb': -1.}, moved, {**moved, 'qb': -1.}]
    previous = dict(field='xb', position_control='xb', excited_quantity='qb', quantity_control='qb',
                    candidate_owners=model['dynamic_fields'], alternative_owners=['qb', 'qc'],
                    corners=square, baseline_ref={'probe_id': 'eligible', 'position': 0},
                    owner_predictions=position_owner_predictions(model['dynamic_fields'], 'qb', model['dynamic_fields']))
    rows.extend(record(name, point) for name, point in zip(['old-q', 'old-x', 'old-xq'], square[1:]))
    rows[-1]['after']['qc'] += 1.
    measured = assess_position_experiment(previous, model, rows, SearchBudget(200))
    assert measured['status'] == 'position_domain_narrowed' and measured['retained_owners'] == ['qb', 'qc']
    model['position_candidates']['xb']['quantities'] = measured['retained_owners']

    def plan(available=2, assignments=schemas):
        return plan_position_owner_followup(spec, model, rows, SearchBudget(200), previous_plan=previous,
                                            previous_result=measured, schemas=assignments, remaining_rows=available)
    assert plan(1)['status'] == 'complete_position_experiment_unfunded'
    assert plan(assignments=schemas[:1])['status'] == 'incomplete_position_owner_representation'
    result = plan()
    assert result['status'] == 'proposed_position_experiment' and result['missing_corner_indices'] == [1, 3]
    assert result['excited_quantity'] == 'qa' and result['owner_predictions'] == {
        'qb': {'allowed_foreign_quantities': ['qb']}, 'qc': {'allowed_foreign_quantities': ['qc']}}
    assert [result['paid_corner_refs'][i] for i in (0, 2)] == [measured['corner_refs'][i] for i in (0, 2)]
    rows.extend(record(name, result['corners'][i]) for name, i in [('new-q', 1), ('new-xq', 3)])
    paid = plan(0)
    assert paid['status'] == 'complete_paid_position_experiment' and paid['required_rows'] == 0
    assert assess_position_experiment(paid, model, rows, SearchBudget(200))['retained_owners'] == ['qb', 'qc']
    rows[-1]['after']['qb'] += 1.
    assert assess_position_experiment(paid, model, rows, SearchBudget(200))['retained_owners'] == ['qb']
    rows[3] = record('other-x', moved)
    assert plan()['status'] == 'unavailable_paid_position_background'
    assert assess_position_experiment(paid, model, rows, SearchBudget(200))['missing_corners'] == [2]


def test_normal_child_pays_only_incremental_corners_and_hands_parent_its_narrowed_domain(tmp_path, monkeypatch):
    from hegel_machine.cognition.common_object_research import CommonObjectDependencyMethods
    from hegel_machine.cognition.object_dependency_methods import ObjectDependencyMethods

    spec, model, rows, schemas, record = scene()
    # All prior focus work has been paid. Isolate the new position question.
    def measured_flat(self, spec, model, records, *args, **kwargs):
        assert self._index(records)
        self.evidence = [{'status': 'uninformative_pilot'}]
        return None, {'purpose': 'measured_dependency_methods_lack_contrast'}
    monkeypatch.setattr(ObjectDependencyMethods, 'propose', measured_flat)
    model.update(schema_alternatives={'schemas': schemas}, model_id='paid', unresolved=['x', 'f'],
        groups=[{'q': q, 'object_id': q, 'controls': {'q': q}} for q in model['dynamic_fields']])
    work = CommonObjectDependencyMethods(tmp_path, [], budget=SearchBudget(2000), actions=[])
    def propose(parent, available):
        return work.propose(spec, parent, rows, len(rows), {'fa': 0., 'fb': 1.}, remaining_rows=available)
    probe, metadata = propose(model, 3)
    assert metadata['position_experiment'] and metadata['interpret_dependencies'] is False
    plan = work.result['position_experiment']
    def set_response(row):
        for name in model['dynamic_fields']:
            if name != plan['excited_quantity']:
                row['after'][name] += row['before'][plan['field']] * row['before'][plan['excited_quantity']]
    set_response(rows[1])
    for i in range(3):
        purchase = record(f'new{i}', probe.sequence[0])
        set_response(purchase)
        rows.append(purchase)
        before = work.budget.spent
        work.observe(rows)
        readout_charge = 4 * (len(probe.sequence[0]) + 1) + 4 * len(model['raw_sensor_features']) + 3 * len(model['dynamic_fields'])
        assert work.budget.spent - before == len(probe.sequence[0]) + 1 + 4 + (readout_charge if i == 2 else 0)
        if i < 2:
            assert work.consume_position_result(model) is model
            probe, _ = propose(model, 2-i)
    updated = work.consume_position_result(model)
    assert updated['position_candidates'][plan['field']]['quantities'] == [plan['excited_quantity']]
    assert 'schema_alternatives' not in updated and model['schema_alternatives']['schemas'] == schemas
    assert work.consume_position_result(updated) is updated
    before = work.budget.spent
    probe, metadata = propose(updated, 0)
    assert probe is None and metadata['position_experiment'] and before == work.budget.spent


def test_all_ordinary_organizations_bind_the_recalled_position_experiment_locally(tmp_path, monkeypatch):
    import json
    from hegel_machine.cognition.common_object_research import CommonObjectResearchAgent
    from hegel_machine.cognition.object_dependency_methods import ObjectDependencyMethods
    from test_position_method_formation import measured_stages, rebuild

    # Every arm receives identical raw apparatus history. Only PR3 starts with
    # the frozen executable. No source instrument is named like a target one.
    old = json.dumps(measured_stages())
    for key in ('qa', 'qb', 'qc', 'xb', 'eligible', 'old-x'):
        old = old.replace(f'"{key}"', f'"donor-{key}"')
    history = json.loads(old)
    compiled = rebuild(history, SearchBudget(200))['method']
    raw = {'previous': {'position_history': history}, 'source_ref': compiled['source_ref'],
           'model_id': compiled['model_id']}

    def flat(self, spec, model, records, *args, **kwargs):
        assert self._index(records)
        self.evidence = [{'status': 'uninformative_pilot'}]
        return None, {'purpose': 'measured_dependency_methods_lack_contrast'}
    monkeypatch.setattr(ObjectDependencyMethods, 'propose', flat)
    for arm in ('archive', 'fixed_pr2', 'pr3'):
        spec, model, rows, schemas, record = scene()
        quantities = model['dynamic_fields']
        model['focus_candidates']['fc'] = {'quantities': quantities}
        model.update(model_id='target', schema_alternatives={'schemas': schemas}, unresolved=['x', 'f'],
            groups=[{'q': q, 'u': 'drive-' + q, 'object_id': q, 'controls': {'q': q, 'u': 'drive'}}
                    for q in quantities])
        agent = CommonObjectResearchAgent(workspace=tmp_path / arm, organization=arm,
            learned_methods=([compiled] if arm == 'pr3' else []), raw_sources=[raw])
        work = agent._dependency_methods([], budget=SearchBudget(5000), actions=[])
        def propose(remaining):
            return work.propose(spec, model, rows, len(rows), {'fa': 0., 'fb': 1.}, remaining_rows=remaining)
        probe, _ = propose(5)
        initial = work.result['position_experiment']
        field = initial['field']
        owner = next(q for q in reversed(quantities) if q != initial['excited_quantity'])
        def measure(row):
            row['after'][owner] += row['before'][field] * sum(row['before'][q] for q in quantities if q != owner)
        measure(rows[1])
        for i in range(5):
            if i:
                probe, metadata = propose(5 - i)
            assert probe is not None
            if i == 3:
                followup = work.result['position_experiment']
                assert metadata['source_object_model_id'] == compiled['model_id']
                assert followup['excited_quantity'] not in followup['candidate_owners']
                assert followup['missing_corner_indices'] == [1, 3]
                assert followup['field'] == field and 'donor-' not in str(followup)
            row = record(f'target-{i}', probe.sequence[0])
            measure(row)
            rows.append(row)
            work.observe(rows)
            model = work.consume_position_result(model)
        assert model['position_candidates'][field]['quantities'] == [owner]
        assert work.result['method_object']['position_method'] == compiled['position_method']
        assert len(agent.source_reconstruction_receipts) == (0 if arm == 'pr3' else 1)


def test_completed_selector_and_new_position_question_keep_separate_answers(tmp_path, monkeypatch):
    from hegel_machine.cognition.common_object_research import CommonObjectDependencyMethods
    from hegel_machine.cognition.object_dependency_methods import ObjectDependencyMethods

    spec, model, rows, schemas, record = scene()
    model['schema_alternatives'] = {'schemas': schemas}
    # The selector pilot and the position proposal legitimately share a return.
    # Every selector contrast is already measured with constant target.
    base = rows[0]['controls']
    paired = []
    for i, selector in enumerate((-.75, .75)):
        a, b = {**base, 'selector': selector}, {**base, 'selector': selector, 'qa': .5}
        paired.append([[a], [b]])
        rows.extend([record(f'sel-{i}-a', a), record(f'sel-{i}-b', b)])
    relation = {'excitation_role': 'quantity', 'matched_prefix_rows': 0, 'quantity_candidate': 'qa'}
    work = CommonObjectDependencyMethods(tmp_path, [], budget=SearchBudget(1500), actions=[])
    work.previous_ref = work._save('paid_selector.json', {'paired_contract': paired, 'relation': relation})
    old_child = work.common._open_question({'id': 'selector', 'question': 'Which membership?',
        'parent_id': 'task', 'depends_on': [], 'experiment_relation': relation, 'paired_contract': paired})
    work.child = old_child
    # All other selector recipes were flat, which exposes the production switch.
    monkeypatch.setattr(ObjectDependencyMethods, 'propose',
        lambda self, *args, **kwargs: (None, {'purpose': 'measured_dependency_methods_lack_contrast'}))
    probe, _ = work.propose(spec, model, rows, 0, {'fa': 0., 'fb': 1.}, remaining_rows=3)
    first = work.result
    assert first['completed_relation']['status'] == 'uninformative_pilot'
    assert old_child['answer'] == first['completed_relation']
    position_child = work.child
    assert position_child['experiment_relation']['kind'] == 'position_membership_contrast'
    plan = first['position_experiment']
    rows.append(record('pos-q', probe.sequence[0]))
    work.observe(rows)
    assert work.child is position_child and not position_child.get('answer')
    assert [action['sequence'][0] for action in work.result['actions']] == plan['corners'][2:]
    assert 'completed_relation' not in work.result
    rows.extend(record(f'pos-{i}', plan['corners'][i]) for i in (2, 3))
    work.observe(rows)
    assert position_child['answer'] == work.result['position_assessment']
    assert position_child['answer']['status'] == 'position_domain_unchanged'
