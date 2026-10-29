"""Learned transitions execute from reset without observed response feedback."""

import ast
from dataclasses import asdict
from pathlib import Path

import pytest

from hegel_machine.cognition.code_agent import (
    _closed_temporal_export, literal_python_model_context_matches,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Definition, EvaluationContext, MissingHistory, Program
from hegel_machine.cognition.models import Observation, TaskSpec
from hegel_machine.cognition.temporal_operator_agent import (
    _PROGRAM_REPRESENTATION_RUNNER, _PROGRAM_RUNNER, _STATE_PROGRAM_RUNNER,
    export_temporal_program,
)


def transition(control=None):
    u = control or Program('feature', name='u')
    on = Program('ge', children=(u, Program('const', value=.6)))
    off = Program('lt', children=(u, Program('const', value=.2)))
    mature = Program('if', children=(on, Program('const', value=1.),
        Program('if', children=(off, Program('const', value=0.),
                                Program('lag', name='__target__', lag=1)))))
    return Program('if', children=(Program('available', lag=1), mature, on))


def observations(values, injected_target=None):
    history, rows = [], []
    for value in values:
        features = {'u': value}
        if injected_target is not None:
            features['__target__'] = injected_target
        rows.append(Observation(features, 999., tuple(history)))
        history.append(features)
    return rows


def test_replay_keeps_learned_transition_but_uses_only_own_state_and_resets():
    body = transition()
    program = Program('state_replay', children=(body,))
    values = [.8, *([.4] * 32), .1, .4]
    expected = [1.] * 33 + [0., 0.]
    rows = observations(values)
    contaminated = observations(values, injected_target=-42.)
    assert program.children[0] is body
    assert [program.predict(row) for row in rows] == expected
    assert [program.predict(row) for row in contaminated] == expected
    assert body.predict(contaminated[1]) == -42.  # Legacy lag semantics are unchanged.
    reset = Observation({'u': .4}, 1.)
    assert program.predict(reset) == 0.
    assert program.predict(rows[-3]) == 1.
    assert program.predict(reset) == 0.
    assert program.minimum_history_depth == 0 and program.history_depth >= 1
    assert program.observed_variables() == {('u', 0)}
    assert ('__target__', 1) in body.observed_variables()
    assert program.required_features() == {'u'}
    context = EvaluationContext(rows[-3])
    assert program.predict(rows[-3], context=context) == 1.
    assert context.primitive_steps > len(rows[-3].history)


def test_teacher_forced_loss_does_not_become_closed_loop_loss():
    body = Program('if', children=(Program('available', lag=1),
        Program('lag', name='__target__', lag=1), Program('const', value=0.)))
    rows, history = [], []
    for target in (0., 1., 1., 1.):
        rows.append(Observation({'u': 0.}, target, tuple(history)))
        history.append({'u': 0., '__target__': target})
    replay = Program('state_replay', children=(body,))
    assert loss(body, rows, 0., .5) == .25
    assert loss(replay, rows, 0., .5) == .75
    assert [replay.predict(row) for row in rows] == [0., 0., 0., 0.]


def test_replay_requires_the_learned_reset_branch():
    with pytest.raises(ValueError, match='executes from reset'):
        Program('state_replay', children=(Program('lag', name='__target__', lag=1),))


@pytest.mark.parametrize('callable_body', [False, True])
def test_native_serialized_and_exported_replay_have_identical_full_trajectories(callable_body):
    if callable_body:
        definition = Definition('learned_state', ('input',), Program('state_replay',
            children=(transition(Program('arg', name='input')),)))
        state = Program('call', name='learned_state', definition=definition,
                        children=(Program('feature', name='u'),))
    else:
        state = Program('state_replay', children=(transition(),))
    program = Program('add', children=(state, Program('feature', name='u')))
    restored = Program.from_dict(program.to_dict())
    namespace = {}
    source = export_temporal_program(program)
    assert source.endswith(_STATE_PROGRAM_RUNNER)
    exec(source, namespace)
    exec(_PROGRAM_REPRESENTATION_RUNNER, namespace)
    assert len(namespace['_representation_terms']) == 2
    assert namespace['_representation_terms'][0][0] == 'state_replay'
    rows = observations([.4, .8, .4, .4, .1, .4], injected_target=99.)
    for row in [*rows, *reversed(rows)]:
        expected = program.predict(row)
        assert restored.predict(row) == expected
        assert namespace['predict'](asdict(row)) == expected
        coordinates = namespace['represent'](asdict(row))
        assert namespace['predict_features'](coordinates) == expected
        assert coordinates[0] == [1, state.predict(row)]


def test_target_lag_is_only_exportable_inside_replay():
    with pytest.raises(ValueError, match='only control'):
        export_temporal_program(transition())
    replay = Program('state_replay', children=(transition(),))
    with pytest.raises(ValueError, match='only control'):
        export_temporal_program(Program('add', children=(replay,
            Program('lag', name='__target__', lag=1))))
    with pytest.raises(ValueError, match='only control'):
        export_temporal_program(Program('state_replay', children=(
            Program('feature', name='__target__'),)), feature_names=('u', '__target__'))


def test_ordinary_historical_export_bytes_and_literal_recognition_stay_unchanged(tmp_path):
    evidence = Path(__file__).resolve().parents[1] / 'docs/evidence/frozen_parent_terminal_259_455_20261025/models'
    source = (evidence / 'remove_candidate_0016_v0001.py').read_text()
    prefix = '# Frozen paid-data predictor; controls and recorded history only.\n_PROGRAM = '
    suffix = '\n' + _PROGRAM_RUNNER
    assert source.endswith(suffix)
    literal = ast.literal_eval(source[len(prefix):-len(suffix)])

    def restore(node):
        op, value, name, lag, children = node
        return Program(op, value, name, lag, tuple(map(restore, children)))

    assert export_temporal_program(restore(literal)) == source
    (tmp_path / 'old.py').write_text(source)
    assert _closed_temporal_export(tmp_path, 'old.py')


def test_replay_literal_reuses_only_its_actual_predictor_source(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    program = Program('state_replay', children=(transition(),))
    source = export_temporal_program(program)
    (first / 'model.py').write_text(source)
    (second / 'model.py').write_text(source)
    (second / 'unused_parameters.json').write_text('{"new": true}')
    assert _closed_temporal_export(first, 'model.py')
    assert _closed_temporal_export(second, 'model.py')
    assert literal_python_model_context_matches(first, 'model.py', second, 'model.py')
    changed = Program('add', children=(program, Program('const', value=.1)))
    (second / 'model.py').write_text(export_temporal_program(changed))
    assert not literal_python_model_context_matches(first, 'model.py', second, 'model.py')


def test_replay_literal_alias_reuses_the_paid_score_without_new_predictions(tmp_path, monkeypatch):
    from test_literal_program_score_reuse import scene
    from test_python_paid_score_reuse import choose, no_execution, scoring_costs

    agent, lab, calls = scene(tmp_path, monkeypatch)
    source = export_temporal_program(Program('state_replay', children=(transition(),)))
    model, alias = 'state/_evaluated_model/state.py', 'renamed/_evaluated_model/state.py'
    agent._save(model, source)
    agent._save(alias, source)
    first = choose(agent, lab, [model])
    assert first['new_scored_count'] == 1
    before = scoring_costs(agent)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    repeated = choose(agent, lab, [alias])
    assert repeated['reused_score_count'] == 1 and repeated['new_scored_count'] == 0
    assert scoring_costs(agent) == before
    assert calls[-1] == model


@pytest.mark.parametrize('callable_body', [False, True])
def test_method_profile_distinguishes_internal_state_from_observed_response(callable_body, monkeypatch):
    from hegel_machine.cognition.method_conditions import method_profile

    replay = Program('state_replay', children=(transition(),))
    if callable_body:
        definition = Definition('closed_transition', (), replay)
        replay = Program('call', name=definition.definition_id, definition=definition)
    # A raw response lag outside the closed replay retains its original role.
    program = Program('add', children=(replay, Program('lag', name='__target__', lag=1)))
    spec = TaskSpec('fixture', 'Existing profile path', ('u',), (), ())

    def no_predictions(*args, **kwargs):
        pytest.fail('method profiles must reuse the supplied comparison')

    monkeypatch.setattr(Program, 'predict', no_predictions)
    profile = method_profile(spec, observations([.8, .4]), program,
                             residual_loss=.1, baseline_loss=1.)

    def roles(node):
        return ([node['role']] if 'role' in node else []) + [role
            for child in node.get('children', []) for role in roles(child)]

    representation = profile['representation']
    assigned = roles(representation['structure'])
    assert assigned.count('predicted_state') == 1
    assert assigned.count('observed_target') == 1
    assert representation['kind'] == 'history'
    assert representation['uses_calls'] == callable_body
    assert representation['program'] == program.to_dict()


@pytest.mark.parametrize('lag', [1, 2, 3])
def test_measurement_time_binding_replays_the_actual_earlier_prefix(lag):
    from hegel_machine.cognition.scale_bindings import bind_measurement_arguments

    replay = Program('state_replay', children=(transition(),))
    body_before = replay.to_dict()
    previous = bind_measurement_arguments(Program('lag', name='measurement', lag=lag),
                                          {'measurement': replay})
    namespace = {}
    exec(export_temporal_program(previous), namespace)
    rows = observations([.8, .4, .1, .4, .8, .4, .4, .1], injected_target=-42.)
    for index, row in enumerate(rows):
        if index < lag:
            with pytest.raises(MissingHistory):
                previous.predict(row)
        else:
            expected = replay.predict(rows[index - lag])
            assert previous.predict(row) == expected
            assert namespace['predict'](asdict(row)) == expected
    assert replay.to_dict() == body_before
    # A later call on a new reset trajectory cannot retain the preceding state.
    restarted = observations([.4] * (lag + 2), injected_target=1.)
    assert previous.predict(restarted[-1]) == 0.
