"""Frozen fitting views stay executable without entering live checkpoints."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from test_common_archive_inquiry import configuration, laboratory, PLAN, QUESTION
from test_common_archive_resume import agent_at, checkpoint, SEED


def nested_snapshot(workspace):
    root = workspace / "shared/generalization/revision_0/_evaluated_model"
    root.mkdir(parents=True)
    (root / "observations.json").write_text('[{"target": 3}]')
    (root / "model.py").write_text("def predict(row): return 3\n")
    return root


def forbid_snapshot_text_reads(monkeypatch):
    read_text = Path.read_text

    def authored_only(path, *args, **kwargs):
        if "_evaluated_model" in path.parts:
            raise AssertionError("live context reread an immutable fitting snapshot")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", authored_only)


def test_live_request_prunes_stale_snapshot_entries_without_reading_them(tmp_path, monkeypatch):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm="code_react",
                                     config=configuration())
    frozen = nested_snapshot(agent.workspace)
    snapshot_name = str((frozen / "observations.json").relative_to(agent.workspace))
    agent.files[snapshot_name] = "stale serialized fitting rows"
    (agent.workspace / "shared/model.py").write_text("# current authored model\n")
    forbid_snapshot_text_reads(monkeypatch)

    request = agent._request(laboratory())

    assert snapshot_name not in request["files"]
    assert not any("_evaluated_model" in Path(name).parts for name in agent.files)
    assert request["files"]["shared/model.py"] == "# current authored model\n"
    assert (frozen / "observations.json").exists()
    assert request["session_knowledge"] == {}


def test_active_checkpoint_keeps_account_and_authored_files_not_fitting_copies(tmp_path, monkeypatch):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm="code_react",
                                     config=configuration())
    lab = laboratory()
    agent._execute("write", {"path": "shared/probe.py", "code": PLAN}, lab)
    agent._execute("query", {"path": "shared/probe.py", "question": QUESTION}, lab)
    frozen = nested_snapshot(agent.workspace)
    name = str((frozen / "observations.json").relative_to(agent.workspace))
    agent.files[name] = "old nested fitting view"
    account, observations = deepcopy(agent.costs), deepcopy(agent.observations)
    forbid_snapshot_text_reads(monkeypatch)

    active = agent.result("researching")

    assert active["outgoing_knowledge"]["files"]["shared/probe.py"] == PLAN
    assert not any("_evaluated_model" in Path(name).parts
                   for name in active["outgoing_knowledge"]["files"])
    assert active["costs"] == account
    assert active["paid_observations"] == observations
    assert (frozen / "observations.json").exists()


def test_restore_refreshes_authored_sources_without_reading_old_fitting_copies(tmp_path, monkeypatch):
    old, lab, previous = checkpoint(tmp_path)
    frozen = nested_snapshot(old.workspace)
    name = str((frozen / "observations.json").relative_to(old.workspace))
    context_path = old.workspace / "research_context.json"
    context = json.loads(context_path.read_text())
    context["files"][name] = "stale nested fitting view"
    context_path.write_text(json.dumps(context))
    previous["outgoing_knowledge"]["files"][name] = "another stale view"
    continued = agent_at(tmp_path / "continued")
    target = TemporalInquiryLab(task_index=1)
    forbid_snapshot_text_reads(monkeypatch)

    restored = continued.restore_research(previous, old.workspace, target,
                                         confirmation_seed=SEED)

    assert restored["new_observations"] == 0
    assert name not in continued.files
    assert (continued.workspace / name).exists()
    assert continued.rows == old.rows and continued.events == old.events
    assert continued.costs == old.costs and target.costs == lab.costs
    assert continued.confirmations == old.confirmations
    assert continued._evaluated_summary() == old._evaluated_summary()


@pytest.mark.parametrize('status', ['completed', 'model_call_budget_exhausted'])
def test_final_publication_prunes_unselected_snapshots_and_preserves_paid_account(
        tmp_path, monkeypatch, status):
    source = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react',
                                      config=configuration())
    lab = laboratory()
    source._execute('write', {'path': 'shared/probe.py', 'code': PLAN}, lab)
    source._execute('query', {'path': 'shared/probe.py', 'question': QUESTION}, lab)
    frozen = nested_snapshot(source.workspace)
    stale_model = str((frozen / 'model.py').relative_to(source.workspace))
    source.files[stale_model] = 'stale serialized snapshot'
    source.events.append({'action': 'evaluate', 'arguments': {'model': stale_model},
                          'feedback': {'returncode': 0, 'mse': 3.}})
    data = source.workspace / 'models'
    data.mkdir()
    (data / 'summary.json').write_text('{"coefficient": 1}')
    account, records, events = deepcopy(source.costs), deepcopy(source.observations), deepcopy(source.events)
    forbid_snapshot_text_reads(monkeypatch)

    result = source.result(status)

    files = result['outgoing_knowledge']['files']
    assert files['shared/probe.py'] == PLAN
    assert files['models/summary.json'] == '{"coefficient": 1}'
    for name in ('observations.json', 'model_observations.json', 'laboratory_observations.json'):
        assert files[name] == (source.workspace / name).read_text()
    assert not any('_evaluated_model' in Path(name).parts for name in files)
    assert result['costs'] == source.costs == account
    assert result['paid_observations'] == source.observations == records
    assert result['events'] == source.events == events
    assert (frozen / 'observations.json').exists()


def test_publication_keeps_frozen_fitting_view_and_parameters_for_consumers(tmp_path, monkeypatch):
    source = CommonArchiveInquiryAgent(None, tmp_path / "source", arm="code_react",
                                      config=configuration())
    lab = laboratory()
    source._execute("write", {"path": "shared/probe.py", "code": PLAN}, lab)
    source._execute("query", {"path": "shared/probe.py", "question": QUESTION}, lab)
    source._execute("write", {"path": "parameters.json", "code": '{"offset": 3}'}, lab)
    model = """import json
def fit(rows):
    global coefficient, offset
    offset = json.load(open('parameters.json'))['offset']
    coefficient = sum(row['target'] / next(iter(row['features'].values()))
                      for row in rows) / len(rows) + offset
def predict(row):
    return (coefficient - offset) * next(iter(row['features'].values()))
"""
    source._execute("write", {"path": "model.py", "code": model}, lab)
    evaluated = source._execute("evaluate", {"model": "model.py"}, lab)
    retained = evaluated["evaluated_model"]["model"]
    source._request(lab)
    assert source._execute("finish", {"model": retained}, lab)["fresh_confirmation"]["passed"]
    unselected = nested_snapshot(source.workspace)
    nested = nested_snapshot(source.workspace / '_evaluated_model')
    account, records, events = deepcopy(source.costs), deepcopy(source.observations), deepcopy(source.events)
    read_text = Path.read_text

    def selected_only(path, *args, **kwargs):
        if path.is_relative_to(unselected) or path.is_relative_to(nested):
            raise AssertionError('publication read an unselected fitting snapshot')
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', selected_only)

    result = source.result('completed')
    memory = result['outgoing_knowledge']

    assert "_evaluated_model/model_observations.json" in memory["files"]
    assert "_evaluated_model/parameters.json" in memory["files"]
    assert json.loads(memory['files']['_evaluated_model/parameters.json']) == {'offset': 3}
    assert len(json.loads(memory['files']['_evaluated_model/model_observations.json'])) == 2
    assert not any(name.startswith(str(unselected.relative_to(source.workspace)))
                   or name.startswith(str(nested.relative_to(source.workspace))) for name in memory['files'])
    assert result['costs'] == source.costs == account
    assert result['paid_observations'] == source.observations == records
    assert result['events'] == source.events == events
    assert (unselected / 'observations.json').exists() and (nested / 'observations.json').exists()
    consumer = CommonArchiveInquiryAgent(None, tmp_path / "consumer", arm="code_react",
        knowledge=memory, config=configuration(), source_task=False)
    consumer._request(laboratory())
    confirmed = consumer._execute("finish", {"model": retained}, laboratory())
    assert confirmed["fresh_confirmation"]["passed"]
    assert confirmed["fresh_confirmation"]["mse"] == pytest.approx(0.)


def test_publication_follows_selected_frozen_child_binding_for_independent_consumer(tmp_path):
    source = CommonArchiveInquiryAgent(None, tmp_path / 'source', arm='code_react',
                                      config=configuration())
    lab = laboratory()
    source._execute('write', {'path': 'shared/probe.py', 'code': PLAN}, lab)
    source._execute('query', {'path': 'shared/probe.py', 'question': QUESTION}, lab)
    child = 'shared/component_0/_evaluated_model'
    fitted = """import json
def fit(rows):
    global coefficient
    offset = json.load(open('parameters.json'))['offset']
    coefficient = sum(row['target'] / next(iter(row['features'].values()))
                      for row in rows) / len(rows) + offset
def predict(row):
    return (coefficient - 3) * next(iter(row['features'].values()))
"""
    parent = """def bind_models(models):
    global component
    component = models['component']
def predict(row):
    return component(row)
"""
    source._execute('write', {'path': child + '/model.py', 'code': fitted}, lab)
    source._execute('write', {'path': child + '/parameters.json', 'code': '{"offset": 3}'}, lab)
    source._execute('write', {'path': child + '/model_observations.json',
        'code': (source.workspace / 'model_observations.json').read_text()}, lab)
    source._execute('write', {'path': 'model.py', 'code': parent}, lab)
    source._execute('write', {'path': 'model_bindings.json',
        'code': json.dumps({'model.py': {'component': child + '/model.py'}})}, lab)
    evaluated = source._execute('evaluate', {'model': 'model.py'}, lab)
    assert evaluated['returncode'] == 0 and evaluated['mse'] == pytest.approx(0.)
    retained = evaluated['evaluated_model']['model']

    memory = source.result('model_call_budget_exhausted')['outgoing_knowledge']

    frozen_child = '_evaluated_model/' + child
    assert memory['files'][frozen_child + '/parameters.json'] == '{"offset": 3}'
    assert json.loads(memory['files'][frozen_child + '/model_observations.json'])
    consumer = CommonArchiveInquiryAgent(None, tmp_path / 'consumer', arm='code_react',
        knowledge=memory, config=configuration(), source_task=False)
    assert json.loads((consumer.workspace / 'model_observations.json').read_text()) == []
    confirmed = consumer._execute('finish', {'model': retained}, laboratory())
    assert confirmed['fresh_confirmation']['passed']
    assert confirmed['fresh_confirmation']['mse'] == pytest.approx(0.)
