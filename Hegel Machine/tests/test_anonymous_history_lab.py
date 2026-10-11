"""The adapter must preserve physics while keeping inference inputs public."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab, CONTROL_LEVELS
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig


def trajectory(*controls):
    return {"kind": "trajectory", "sequence": [{"u": value} for value in controls], "channels": []}


def test_development_mechanisms_have_the_same_public_interface():
    labs = [AnonymousHistoryLab(kind, seed) for kind in ("memoryless", "delay", "hysteresis")
            for seed in (31, 79)]
    assert all(lab.public_spec() == labs[0].public_spec() for lab in labs)
    assert all(lab.task_interface == labs[0].task_interface for lab in labs)
    assert labs[0].public_spec()["control_levels"]["u"] == list(CONTROL_LEVELS)


@pytest.mark.parametrize("kind", ["memoryless", "delay", "hysteresis"])
def test_reset_and_controls_only_history(kind):
    lab = AnonymousHistoryLab(kind, seed=31)
    action = trajectory(-1.0, 0.0, 1.0, 0.0)
    first = lab.execute(action)
    lab.execute(trajectory(1.0, 1.0, 1.0))
    repeated = lab.execute(action)
    assert first["rows"] == repeated["rows"]
    rows = lab.observation_rows(first)
    assert rows[0].history == ()
    for index, row in enumerate(rows):
        assert set(row.features) == {"u"}
        assert row.history == tuple(point["features"] for point in first["rows"][:index])
    assert lab.costs == {"observation_cost": 11, "measurement_cost": 0, "queries": 3}


def test_prediction_inputs_do_not_consult_physics_or_buy_evidence(monkeypatch):
    lab = AnonymousHistoryLab("hysteresis", seed=31)
    def unavailable(*args, **kwargs):
        raise AssertionError("prediction input construction cannot observe the apparatus")
    monkeypatch.setattr(lab._environment, "observe", unavailable)
    before = lab.costs
    rows = lab.prediction_rows(trajectory(0.0, 0.1 + 0.2, -0.5))
    assert [row.target for row in rows] == [0.0, 0.0, 0.0]
    assert rows[-1].history == ({"u": 0.0}, {"u": 0.3})
    assert lab.costs == before


@pytest.mark.parametrize("kind", ["memoryless", "delay", "hysteresis"])
def test_future_inputs_match_executed_inputs(kind):
    lab = AnonymousHistoryLab(kind, seed=79)
    action = trajectory(-1.0, 0.3, 1.0, 0.3)
    inputs = lab.prediction_rows(action)
    measured = lab.observation_rows(lab.execute(action))
    assert [(row.features, row.history) for row in inputs] == [
        (row.features, row.history) for row in measured]


def test_confirmation_and_longer_terminal_evaluation_are_separate():
    lab = AnonymousHistoryLab("hysteresis", seed=31)
    actions = lab.confirmation_actions(50, Random(101))
    assert [lab.action_cost(action) for action in actions] == [24, 24, 2]
    for action in actions:
        lab.execute(action)
    before = lab.costs
    evaluation = lab.evaluation_rows(length=110, seed=733)
    assert len(evaluation) == 110
    assert max(len(row.history) for row in evaluation) == 47
    assert [index for index, row in enumerate(evaluation) if not row.history] == [0, 48, 96]
    assert evaluation == lab.evaluation_rows(length=110, seed=733)
    assert evaluation != lab.evaluation_rows(length=110, seed=739)
    assert all(row.features["u"] in CONTROL_LEVELS for row in evaluation)
    assert all(set(past) == {"u"} for row in evaluation for past in row.history)
    assert lab.costs == before


def test_old_hysteresis_simulator_retains_path_dependence():
    lab = AnonymousHistoryLab("hysteresis", seed=31)
    low = lab.observation_rows(lab.execute(trajectory(-1.0, 0.0)))
    high = lab.observation_rows(lab.execute(trajectory(1.0, 0.0)))
    assert low[-1].features == high[-1].features
    assert low[-1].target == 0.0 and high[-1].target == 1.0


@pytest.mark.parametrize("action", [trajectory(0.15), trajectory(*([0.0] * 25))])
def test_unpublished_inputs_are_rejected_before_purchase(action):
    lab = AnonymousHistoryLab()
    with pytest.raises(ValueError):
        lab.execute(action)
    assert lab.costs["queries"] == 0


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_paid_counterexample_blocks_repeated_confirmation_and_allows_actual_revision(tmp_path, monkeypatch, arm):
    """Real runtime purchases and fits; the authored bias is an interface fixture."""
    lab = AnonymousHistoryLab('memoryless', seed=31)
    fit = """def fit(rows):
    global slope, intercept
    x = [row['features']['u'] for row in rows]
    y = [row['target'] for row in rows]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    slope = sum((a-mx)*(b-my) for a,b in zip(x,y)) / sum((a-mx)**2 for a in x)
    intercept = my - slope * mx
"""
    accurate = fit + "def predict(row):\n    return slope * row['features']['u'] + intercept\n"
    inaccurate = fit + """def predict(row):
    u = row['features']['u']
    return slope * u + intercept + (1.0 if abs(u) < 1.0 else 0.0)
"""
    experiment = """def select_actions(spec, records, memory):
    return {'actions': [{'kind': 'trajectory', 'channels': [],
                         'sequence': [{'u': -1.0}, {'u': 1.0}]}]}
"""
    manifest = {'hypotheses': [{'id': 'candidate', 'path': 'model.py'}],
                'research_programs': [{'id': 'endpoints', 'path': 'shared/probe.py'}],
                'support_questions': []}
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm=arm, source_task=False,
        config=InquiryLLMConfig(observation_budget=100, confirmation_rows=24, execution_seconds=30.),
        confirmation_seed=101, knowledge={'files': {
            'model.py': inaccurate, 'shared/probe.py': experiment,
            'shared/knowledge.json': json.dumps(manifest)}})
    agent._execute('query', {'path': 'shared/probe.py', 'question': {
        'id': 'q1', 'question': 'What do the two endpoints show?',
        'parent_id': None, 'depends_on': []}}, lab)
    calls = []
    score = agent._score

    def counted_score(model):
        calls.append((model, len(agent.rows)))
        return score(model)

    monkeypatch.setattr(agent, '_score', counted_score)
    first = agent._execute('finish', {'model': 'model.py'}, lab)
    assert first['mse'] <= 1e-10
    assert first['fresh_confirmation']['passed'] is False
    assert first['fresh_confirmation']['rows'] == 24
    assert agent.model is None and len(agent.rows) == 26
    assert agent.costs['observation_cost'] == lab.costs['observation_cost'] == 26
    assert agent.costs['compute_cost'] == 2 + 24

    prior_lab, prior_costs = lab.costs, deepcopy(agent.costs)
    repeated = agent._execute('finish', {'model': 'model.py'}, lab)
    assert repeated['returncode'] == 0 and repeated['mse'] > 1e-10
    assert isinstance(repeated.get('error'), str) and repeated['error']
    assert 'fresh_confirmation' not in repeated
    assert agent.model is None and len(agent.confirmations) == 1
    assert lab.costs == prior_lab
    assert agent.costs['observation_cost'] == prior_costs['observation_cost']
    assert agent.costs['compute_cost'] - prior_costs['compute_cost'] == len(agent.rows) == 26
    assert calls == [('model.py', 2), ('model.py', 26)]

    # The agent remains able to edit and test its own source against the same
    # accumulated evidence; the failed submission does not end the investigation.
    agent._execute('write', {'path': 'model.py', 'code': accurate}, lab)
    before_revision = deepcopy(agent.costs)
    revised = agent._execute('finish', {'model': 'model.py'}, lab)
    assert revised['mse'] <= 1e-10 and revised['fresh_confirmation']['passed']
    assert revised['fresh_confirmation']['rows'] == 24
    assert agent.model == 'model.py' and len(agent.confirmations) == 2
    assert agent.costs['observation_cost'] == lab.costs['observation_cost'] == 50
    assert agent.costs['compute_cost'] - before_revision['compute_cost'] == 26 + 24
    assert calls == [('model.py', 2), ('model.py', 26), ('model.py', 26)]
