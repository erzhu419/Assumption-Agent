"""The ordinary Common route consumes real anonymous-object measurements."""

from copy import deepcopy
import json

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.common_object_lab import (CommonObjectLab,
    measured_selector_values, object_records_from_common)
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from test_inquiry_revision import make_agent


def prepare(tmp_path, lab):
    public_lab = CommonObjectLab(lab)
    agent = make_agent(tmp_path, 'hegel', cls=CommonArchiveInquiryAgent, compute=400000)
    agent._method_public_spec = public_lab.public_spec()
    budget = SearchBudget(400000)
    # Historical object studies also begin with a measured random trajectory;
    # it makes the response scope evidence independent of control iteration order.
    initial = lab.random_probe(73, length=12, name='common_initial')
    agent._execute('query', {'actions': [{'kind': 'trajectory', 'channels': ['fine'],
        'sequence': list(initial.sequence)}]}, public_lab)
    records = object_records_from_common(agent.observations)
    model = infer_object_model(lab.spec, records, budget)
    for serial in range(20):
        probe, question = propose_object_probe(lab.spec, model, records, serial)
        if question.get('purpose') == 'distinguish_membership_dependency':
            return agent, public_lab, model, records
        assert probe is not None
        result = agent._execute('query', {'actions': [{'kind': 'trajectory',
            'channels': ['fine'], 'sequence': list(probe.sequence)}]}, public_lab)
        assert 'error' not in result
        records = object_records_from_common(agent.observations)
        model = infer_object_model(lab.spec, records, budget, prior=model)
    raise AssertionError('The existing inducer never reached a public partial-role problem')


def call(agent, lab, model, records, *, sources=(), previous=None):
    values = {'spec': lab.public_spec(), 'model': model, 'records': records,
              'selected_values': measured_selector_values(model, records), 'sources': sources}
    if previous is not None:
        values['previous'] = previous
    arguments = {'operator': 'partial_role_experiment', 'allocation': {'compute': 400000}}
    for name, value in values.items():
        path = name + '.json'
        agent._save(path, json.dumps(value))
        arguments[name + '_ref'] = {'file': path}
    return agent._execute('run', arguments, lab)


def consume(agent, lab, proposal):
    result = agent._execute('query', {'actions': proposal['actions']}, lab)
    assert 'error' not in result
    return object_records_from_common(agent.observations)


def test_common_full_pair_creates_reusable_method_and_replay_never_rebuys(tmp_path):
    with AnonymousObjectLab(objects=3, readout='focus', apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as raw:
        agent, lab, model, records = prepare(tmp_path, raw)
        proposal = call(agent, lab, model, records)
        assert proposal['status'] == 'proposed_complete_partial_role_experiment'
        assert all(name.startswith('c') for name in proposal['actions'][0]['sequence'][0])
        records = consume(agent, lab, proposal)
        # An induction step may resolve the question before callback consumption.
        revised = infer_object_model(raw.spec, records, SearchBudget(400000), prior=model)
        complete = call(agent, lab, revised, records, previous=proposal)
        assert complete['completed_relation']['status'] == 'discriminating'
        method = complete['method_object']
        assert method['dependency_methods']
        assert not method['object_assignment_transferred']
        paid = dict(agent.costs)
        repeated = call(agent, lab, revised, records, previous=proposal)
        assert repeated['completed_relation'] == complete['completed_relation']
        assert agent.costs['observation_cost'] == paid['observation_cost']
        assert agent.costs['measurement_cost'] == paid['measurement_cost']
        assert agent.costs['model_calls'] == 0
        assert not repeated['target_independent_support']


def test_common_partial_pair_is_pending_until_every_selector_corner_is_measured(tmp_path):
    with AnonymousObjectLab(objects=3, readout='focus', apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as raw:
        agent, lab, model, records = prepare(tmp_path, raw)
        proposal = call(agent, lab, model, records)
        assert len(proposal['actions']) > 1
        records = consume(agent, lab, {'actions': proposal['actions'][:1]})
        partial = call(agent, lab, model, records, previous=proposal)
        assert partial['reason'] == 'paired_measurements_incomplete'
        assert 'method_object' not in partial and 'completed_relation' not in partial
        assert partial['actions'] == proposal['actions'][1:]
        records = consume(agent, lab, partial)
        result = call(agent, lab, model, records, previous=partial)
        assert result['completed_relation']['status'] == 'discriminating'


def test_common_query_callback_consumes_actual_pair_without_external_record_projection(tmp_path):
    with AnonymousObjectLab(objects=3, readout='focus', apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as raw:
        agent, lab, model, records = prepare(tmp_path, raw)
        proposal = call(agent, lab, model, records)
        agent._save('pending.json', json.dumps(proposal))
        callback = {'operator': 'partial_role_experiment', 'records': 'common',
            'allocation': {'compute': 400000}, 'previous_ref': {'file': 'pending.json'},
            **{name + '_ref': {'file': name + '.json'}
               for name in ('spec', 'model', 'selected_values', 'sources')}}
        partial = agent._execute('query', {'actions': proposal['actions'][:1],
            'after_observation': [callback]}, lab)
        assert 'after_observation_error' not in partial
        pending = partial['after_observation_results'][0]
        assert pending['reason'] == 'paired_measurements_incomplete'
        assert 'method_object' not in pending
        agent._save('pending.json', json.dumps(pending))
        complete = agent._execute('query', {'actions': pending['actions'],
            'after_observation': [callback]}, lab)
        assert 'after_observation_error' not in complete
        feedback = complete['after_observation_results'][0]
        assert feedback['completed_relation']['status'] == 'discriminating'
        assert feedback['method_object']['dependency_methods']
        paid = agent.costs['observation_cost']
        repeated = agent._execute('run', callback, lab)
        assert repeated['completed_relation'] == feedback['completed_relation']
        assert agent.costs['observation_cost'] == paid


def test_common_source_relation_binds_new_public_names_and_flat_target_revises_it(tmp_path):
    with AnonymousObjectLab(objects=3, readout='focus', apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as raw:
        agent, lab, model, records = prepare(tmp_path / 'source', raw)
        proposal = call(agent, lab, model, records)
        records = consume(agent, lab, proposal)
        result = call(agent, lab, model, records, previous=proposal)
        source = deepcopy(result['method_object'])
        source.update(model_id='actual_source_relation', source_ref={'file': 'source_completion.json'})
        original = deepcopy(source)
    with AnonymousObjectLab(objects=3, readout='total', apparatus_seed=9029,
                            name_seed=6011, wiring_seed=919) as raw:
        agent, lab, model, records = prepare(tmp_path / 'target', raw)
        proposal = call(agent, lab, model, records, sources=[source])
        tested = proposal['question'].get('source_object_model_id') == source['model_id']
        # A control scan sometimes already bought the donor's exact pilot.
        # Consuming that paid failure is also a test, without a new purchase.
        old_recipe = source['dependency_methods'][0]
        already_measured = any(all(item[key] == old_recipe[key] for key in
            ('excitation_role', 'matched_prefix_rows', 'scope'))
            for item in proposal['measured_methods'])
        assert tested or already_measured
        assert not set(source['source_role_bindings']['quantity_controls']) & set(model['quantity_controls'])
        records = consume(agent, lab, proposal)
        result = call(agent, lab, model, records, sources=[source], previous=proposal)
        assert result['completed_relation']['status'] == 'uninformative_pilot'
        assert any(item['action'] == 'rewrite_object_dependency_method'
                   for item in result['method_actions'])
        assert tuple(result['relation'][key] for key in ('excitation_role', 'matched_prefix_rows')) != tuple(
            proposal['relation'][key] for key in ('excitation_role', 'matched_prefix_rows'))
        assert source == original
        assert not result['world_support_updated']
