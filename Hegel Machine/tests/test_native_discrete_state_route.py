"""The Common research route learns and exports control-only state predictors."""

from dataclasses import replace

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.program_disagreement_probes import propose_program_disagreement_probes
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from hegel_machine.cognition.temporal_control_probes import propose_temporal_control_probes
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_native_representation_routing import PublicOnlyLab


def scene(tmp_path, *, validation_flip=False):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='hegel', source_task=False,
        config=replace(InquiryLLMConfig(), compute_budget=30000, max_model_calls=0))
    sequences = [
        [.1, .8, .4, .4, .1, .4, .9, .4, .2, .4],
        [.9, .5, .2, .5, .8, .4, .1, .5, .9, .4],
        [.4, .2, .9, .5, .4, .1, .5, .8, .5, .2],
        [.2, .9, .45, .5, .1, .45, .8, .55, .2],
    ]
    agent.rows, agent.observations = [], []
    for index, values in enumerate(sequences):
        state, history, rows = 0., [], []
        for value in values:
            state = 1. if value >= .7 else 0. if value < .3 else state
            target = 1. - state if index == 3 and validation_flip else state
            rows.append({'features': {'u': value}, 'history': list(history), 'target': target})
            history.append({'u': value})
        start = len(agent.rows)
        agent.observations.append({'action': {'kind': 'trajectory', 'channels': [],
            'sequence': [{'u': value} for value in values]}, 'rows': rows,
            'row_indices': list(range(start, start + len(rows)))})
        agent.rows.extend(rows)
    parent = Program('const', value=0.)
    agent._save('old.py', export_temporal_program(parent))
    arguments = {'operator': 'research_route', 'research_question': {
        'id': 'revision:old', 'kind': 'program_revision', 'parent_id': 'task', 'depends_on': [],
        'question': 'Explain the measured history-dependent response',
        'origin': {'kind': 'frozen_prediction_counterexample', 'row_indices': [1, 2]},
        'answer_type': 'program', 'use_sites': [
            {'kind': 'supply_component', 'model': 'old.py', 'slot': 'representation'}]},
        'theories': [{'id': 'old', 'program': parent.to_dict()}],
        'native_representation_plan': {'parent_theory_ids': ['old'],
            'validation_records': [3], 'compute_limit': 20000, 'max_candidates': 4}}
    return agent, arguments


def test_common_routes_actual_state_learning_without_teacher_forcing_or_validation_fit(tmp_path):
    agent, arguments = scene(tmp_path / 'original')
    old_source = (agent.workspace / 'old.py').read_bytes()
    result = agent._execute('run', arguments, PublicOnlyLab())
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    state_candidates = [item for item in native['candidates'] if item.get('state_research')]
    assert state_candidates and native['comparison_completed']
    assert any(item['direction'] == 'discrete_state' and item['compute_cost'] > 0
               for item in native['search_directions'])
    assert native['selected']['validation_mse'] == 0.
    selected = next(item for item in state_candidates if item['id'] == native['selected']['id'])
    assert selected['program']['op'] == 'state_replay'
    assert selected['state_research']['fit_record_indices'] == [0, 1, 2]
    assert selected['state_research']['fit_rows'] == 30
    assert selected['research_program'] is None
    module = {}
    exec((agent.workspace / result['selected_model']).read_text(), module)
    for row in agent.rows:
        assert module['predict'](row) == row['target']
        corrupted = {**row, 'history': [{**point, '__target__': 123.} for point in row['history']]}
        assert module['predict'](corrupted) == row['target']
    assert result['retained_parent_theory_ids'] == ['old']
    assert (agent.workspace / 'old.py').read_bytes() == old_source
    assert agent.model is None and not result['parent_revision_completed']
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0

    changed, changed_arguments = scene(tmp_path / 'changed_validation', validation_flip=True)
    rerouted = changed._execute('run', changed_arguments, PublicOnlyLab())
    other = load_native_representation_result(rerouted['native_representation_ref'], changed.workspace)
    original_programs = {Program.from_dict(item['program']) for item in state_candidates}
    changed_programs = {Program.from_dict(item['program']) for item in other['candidates']
                        if item.get('state_research')}
    assert original_programs == changed_programs


def learned_replay():
    # A learned transition is a legal prediction program, but the current
    # temporal symbolic solver cannot flatten its endogenous state as controls.
    mature = Program('if', children=(Program('ge', children=(
        Program('feature', name='u'), Program('const', value=.6))),
        Program('const', value=1.), Program('lag', name='__target__', lag=1)))
    body = Program('if', children=(Program('available', lag=1), mature,
                                  Program('const', value=0.)))
    return Program('state_replay', children=(body,))


def test_temporal_prefix_declines_replay_instead_of_requesting_internal_state():
    replay = learned_replay()
    spec = TaskSpec('state', '', ('u',), (), (), feature_bounds={'u': (0., 1.)},
                    max_probe_length=12)
    budget = SearchBudget(1000)
    assert propose_temporal_control_probes(spec, [replay, Program('const', value=0.)],
        budget.spend, length=6) == []
    assert budget.spent == 0


def test_ordinary_disagreement_forecasts_closed_replay_on_public_controls():
    replay = learned_replay()
    spec = TaskSpec('state', '', ('u',), (), (), feature_bounds={'u': (0., 1.)},
                    max_probe_length=12)
    proposals = propose_program_disagreement_probes(spec,
        [replay, Program('const', value=0.)], SearchBudget(6000), length=6)
    assert proposals
    for proposal in proposals:
        history = []
        for point, prediction in zip(proposal['probe'].sequence, proposal['predictions'][0]):
            assert set(point) == {'u'}
            assert replay.predict(Observation(point, 999., tuple(history))) == prediction
            history.append(point)
