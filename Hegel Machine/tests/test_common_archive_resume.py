"""Resume the actual temporal interface without fresh labels or renewed budgets."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from random import Random

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_common_archive_online_research import ScriptedProvider
from test_temporal_prospective_inputs import trajectory
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.models import Observation


SEED = 1709
CONFIG = InquiryLLMConfig(observation_budget=24, confirmation_rows=2,
    max_model_calls=8, token_budget=100, execution_seconds=30., compute_budget=1000)


def agent_at(path, *, provider=None, config=CONFIG):
    return CommonArchiveInquiryAgent(provider, path, arm="hegel", config=config,
        source_task=False, confirmation_seed=SEED,
        knowledge={"files": {
            "inherited.py": "def select_actions(spec, observations, memory): return []\n",
            "shared/knowledge.json": json.dumps({"hypotheses": [{"path": "model.py"}],
                "research_programs": [{"path": "inherited.py"}], "support_questions": []})}})


def execute(agent, lab, action, arguments):
    # Match the real provider's pre-action context and final post-action event.
    context = agent._request(lab)
    (agent.workspace / "research_context.json").write_text(json.dumps(context))
    step = agent.costs["model_calls"]
    agent.costs["model_calls"] += 1
    agent.costs["reported_tokens"] += 1
    feedback = agent._execute(action, arguments, lab)
    agent.events.append({"step": step, "action": action, "arguments": deepcopy(arguments),
                         "feedback": deepcopy(feedback), "usage": {"total_tokens": 1}})
    return feedback


def checkpoint(tmp_path):
    agent, lab = agent_at(tmp_path / "source"), TemporalInquiryLab(task_index=1)
    execute(agent, lab, "query", {"actions": [trajectory(0.)]})
    target = agent.rows[0].target
    execute(agent, lab, "evaluate", {"model": "model.py",
        "code": f"def predict(row): return {target!r}\n", "notes": "Current fit; long paths unresolved."})
    failed = execute(agent, lab, "finish", {"model": "model.py"})
    assert failed["fresh_confirmation"]["passed"] is False
    execute(agent, lab, "write", {"path": "latest.py", "code": "# A new unfinished draft\n",
        "research_question": {"id": "new_focus", "question": "Explain the new counterexamples.",
                              "parent_id": "task", "depends_on": []},
        "notes": "Latest notebook after failed confirmation."})
    return agent, lab, agent.result("token_budget_exhausted")


def test_restore_preserves_original_knowledge_latest_state_and_refuted_incumbent(tmp_path):
    old, lab, previous = checkpoint(tmp_path)
    (old.workspace / "evaluation.json").write_text('{"terminal_label": 999}')
    (old.workspace / "fit.bin").write_bytes(b"paid fitted dependency")
    before = {p.relative_to(old.workspace): p.read_bytes() for p in old.workspace.rglob("*") if p.is_file()}
    continued = agent_at(tmp_path / "continued", config=replace(CONFIG, token_budget=400))
    restored_lab = TemporalInquiryLab(task_index=1)
    restored = continued.restore_research(previous, old.workspace, restored_lab, confirmation_seed=SEED)
    assert restored["new_observations"] == 0
    assert continued.costs == old.costs and restored_lab.costs == lab.costs
    assert continued.events == old.events and continued.rows == old.rows
    assert continued.questions == old.questions and continued._active_question_id == "new_focus"
    assert continued.knowledge == old.knowledge
    assert continued.inherited_files == {"inherited.py", "shared/knowledge.json"}
    assert "latest.py" in continued.files and "latest.py" not in continued.inherited_files
    assert continued._evaluated_summary() == old._evaluated_summary()
    assert continued._evaluated_summary()["support"] == "new_paid_rows_unscored"
    assert (continued.workspace / "research_notes.md").read_text() == "Latest notebook after failed confirmation."
    assert (continued.workspace / "fit.bin").read_bytes() == b"paid fitted dependency"
    assert not (continued.workspace / "evaluation.json").exists()
    assert before == {p.relative_to(old.workspace): p.read_bytes() for p in old.workspace.rglob("*") if p.is_file()}
    denied = continued._execute("finish", {"model": "model.py"}, restored_lab)
    assert "paid evidence contradicts" in denied["error"]
    assert restored_lab.costs == lab.costs


def test_confirmation_stream_continues_after_paid_failure_without_repurchase(tmp_path, monkeypatch):
    old, lab, previous = checkpoint(tmp_path)
    expected = lab.confirmation_actions(2, old.confirmation_rng)
    resumed, target = agent_at(tmp_path / "continued"), TemporalInquiryLab(task_index=1)
    def no_purchase(*args, **kwargs):
        raise AssertionError("restoration must not buy or replay observations")
    monkeypatch.setattr(target, "execute", no_purchase)
    resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert target.confirmation_actions(2, resumed.confirmation_rng) == expected
    assert resumed.confirmations == old.confirmations
    assert not target._laboratory.acquisitions


def test_run_uses_remaining_total_calls_and_preserves_provider_usage_numbering(tmp_path):
    old, lab, previous = checkpoint(tmp_path)
    provider = ScriptedProvider([("run", {"code": "print('continue the same investigation')"})])
    config = replace(CONFIG, token_budget=400, max_model_calls=previous["costs"]["model_calls"] + 1)
    resumed = agent_at(tmp_path / "continued", provider=provider, config=config)
    target = TemporalInquiryLab(task_index=1)
    resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert provider.calls == previous["costs"]["model_calls"]
    result = resumed.run(target)
    assert result["status"] == "model_call_budget_exhausted"
    assert len(provider.requests) == 1
    assert result["costs"]["model_calls"] == config.max_model_calls
    assert result["costs"]["reported_tokens"] == previous["costs"]["reported_tokens"] + 1
    assert result["events"][-1]["step"] == previous["costs"]["model_calls"]
    assert result["costs"]["observation_cost"] == previous["costs"]["observation_cost"]


@pytest.mark.parametrize('recovery_name', ['interrupted_training.json', 'interrupted_training_0877.json'])
def test_restoration_exposes_recovered_dependency_but_excludes_recovery_checkpoint(tmp_path, recovery_name):
    old, lab, previous = checkpoint(tmp_path)
    (old.workspace / 'pending_dependency.py').write_text('value = 17\n')
    recovery = '{"host_archive": true}'
    (old.workspace / recovery_name).write_text(recovery)
    previous['outgoing_knowledge']['files']['pending_dependency.py'] = 'value = 17\n'
    # Older terminal exports included this host checkpoint as if it were code.
    previous['outgoing_knowledge']['files'][recovery_name] = recovery
    resumed = agent_at(tmp_path / 'continued')
    # An inherited copy must not get read back into the author's files, either.
    (resumed.workspace / recovery_name).write_text(recovery)
    resumed.restore_research(previous, old.workspace, TemporalInquiryLab(task_index=1),
                             confirmation_seed=SEED)
    assert resumed.files['pending_dependency.py'] == 'value = 17\n'
    assert recovery_name not in resumed.files
    assert recovery_name not in resumed._request(lab)['files']
    assert recovery_name not in resumed.result('unfinished')['outgoing_knowledge']['files']
    assert (old.workspace / recovery_name).read_text() == recovery
    assert (resumed.workspace / recovery_name).read_text() == recovery
    assert resumed.events == old.events and resumed.observations == old.observations
    assert resumed.costs == old.costs


def test_saved_comparison_and_child_return_remain_available_to_parent(tmp_path):
    old, lab = agent_at(tmp_path / "source"), TemporalInquiryLab(task_index=1)
    execute(old, lab, "query", {"actions": [trajectory(0.)]})
    base = old.rows[0].target
    for name, extra in (("a.py", "0."), ("b.py", "len(row['history'])")):
        execute(old, lab, "write", {"path": name,
            "code": f"def predict(row): return {base!r} + {extra}\n"})
    comparison = execute(old, lab, "run", {"operator": "compare_experiments",
        "models": ["a.py", "b.py"], "experiments": [{"id": "longer", "actions": [trajectory(0., 0.)]}]})
    execute(old, lab, "write", {"path": "readout.py", "code": (
        "def readout(result,spec,memory):\n"
        "    return {'status':'supported','answer':{'selected':'a.py'},"
        "'evidence':result['observations'][0]['row_indices']}\n")})
    query = deepcopy(comparison["comparison"]["query_arguments"])
    query["revision"]["readout"] = "readout.py"
    query["question"] = {"id": "child", "question": "Compare the authored predictions.",
                          "parent_id": "task", "depends_on": []}
    execute(old, lab, "query", query)
    previous = old.result("token_budget_exhausted")
    resumed, target = agent_at(tmp_path / "continued"), TemporalInquiryLab(task_index=1)
    resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert resumed.experiment_comparisons == old.experiment_comparisons
    assert resumed.revisions == old.revisions
    request = resumed._request(target)
    answer = request["online_research"]["parent_returns"][0]["answers"][-1]
    assert answer["answer"]["selected"] == "a.py"
    evaluated = resumed._execute("evaluate", {"model": answer["answer"]["selected"],
        "research_question": {"id": "task"}}, target)
    assert evaluated["mse"] < 1e-10
    assert resumed.costs["observation_cost"] == old.costs["observation_cost"]


def test_restored_exhausted_python_stops_without_another_model_call(tmp_path):
    old, _, previous = checkpoint(tmp_path)
    previous["costs"]["python_execution_seconds"] = CONFIG.execution_seconds + .1
    provider = ScriptedProvider([])
    resumed = agent_at(tmp_path / "continued", provider=provider,
                       config=replace(CONFIG, token_budget=400))
    lab = TemporalInquiryLab(task_index=1)
    resumed.restore_research(previous, old.workspace, lab, confirmation_seed=SEED)
    result = resumed.run(lab)
    assert result["status"] == "python_execution_budget_exhausted"
    assert provider.requests == []
    assert result["costs"] == previous["costs"]
    assert result["events"] == previous["events"]
    assert result["model"] is None


FROZEN_FIT = """value = 0.
def fit(rows):
    global value
    value = rows[0]['target']
def predict(row): return value
"""


def write_dependency(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def saved_research(agent, lab):
    agent._prepare_program_state()
    write_dependency(agent.workspace, 'research_context.json', json.dumps(agent._request(lab)))
    return agent.result('model_call_budget_exhausted')


def test_restore_keeps_primary_views_and_active_paid_prefix_without_nested_history(tmp_path, monkeypatch):
    old, lab, _ = checkpoint(tmp_path)
    primary = [
        '_evaluated_model', '_submitted_models/confirmation_0000/_evaluated_model',
        'shared/generalization/earlier/_evaluated_model',
        '_experiment_models/comparison_0000/candidate_0000/_evaluated_model',
        '_observable_models/meter/_evaluated_model', 'archive/source_000/_evaluated_model',
    ]
    training = json.dumps([asdict(old.rows[0])])
    blocked = set()
    views = {}
    for name in primary:
        context = old.workspace / name
        if not (context / 'model.py').is_file():
            write_dependency(context, 'model.py', FROZEN_FIT)
            write_dependency(context, 'model_observations.json', training)
        write_dependency(context, 'ordinary_data/fit.bin', 'Retain the ordinary fitted input.')
        views[name] = (context / 'model_observations.json').read_text()
        junk = context / 'shared/generalization/unreferenced/_evaluated_model'
        write_dependency(junk, 'history.bin', 'An unrelated historical context.')
        blocked.add(junk)
    old.provider = ScriptedProvider([])
    old._start_inquiry({'question': {'id': 'continue_local', 'parent_id': 'task',
                                    'question': 'Continue the paid local investigation.'},
                        'max_model_calls': 2}, parent_event_index=len(old.events) - 1)
    previous = saved_research(old, lab)
    original_scandir = os.scandir

    def scandir(path):
        if not isinstance(path, int):
            assert Path(path) not in blocked, 'restore descended into unreferenced nested history'
        return original_scandir(path)

    monkeypatch.setattr(os, 'scandir', scandir)
    if hasattr(Path, '_accessor'):
        monkeypatch.setattr(Path._accessor, 'scandir', scandir)
    provider = ScriptedProvider([])
    restored = agent_at(tmp_path / 'continued', provider=provider)
    target = TemporalInquiryLab(task_index=1)
    report = restored.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert report['new_observations'] == 0 and provider.requests == []
    assert restored.costs == old.costs and target.costs == lab.costs
    assert restored.events == old.events and restored.observations == old.observations
    assert restored.rows == old.rows and restored.confirmations == old.confirmations
    assert restored.questions == old.questions and restored.llm_inquiries == old.llm_inquiries
    assert restored._inquiry_stack[-1]['question_id'] == 'continue_local'
    assert restored.operator_results == old.operator_results
    assert restored.candidate_paths == old.candidate_paths
    submission = restored.workspace / '_submitted_models/confirmation_0000'
    assert submission.is_symlink()
    assert submission.resolve() == (old.workspace / '_submitted_models/confirmation_0000').resolve()
    for name in primary:
        context = restored.workspace / name
        assert (context / 'model.py').is_file()
        assert (context / 'model_observations.json').read_text() == views[name]
        assert (context / 'ordinary_data/fit.bin').is_file()
        history = context / 'shared/generalization/unreferenced/_evaluated_model'
        if name.startswith('_submitted_models/'):
            # Historical submissions reference the retained snapshot intact.
            assert (history / 'history.bin').read_text() == 'An unrelated historical context.'
        else:
            assert not history.exists()
        assert (old.workspace / name / 'model_observations.json').read_text() == views[name]
    # A new live fitting file must not replace a retained primary's own paid view.
    write_dependency(restored.workspace, 'model_observations.json',
                     '[{"features":{"u":0},"target":999,"history":[]}]')
    selected = primary[2] + '/model.py'
    score = evaluate_python_model(restored.workspace, selected,
                                 [Observation({'u': 0.}, old.rows[0].target)])
    assert score['returncode'] == 0 and score['mse'] == 0, score


@pytest.mark.parametrize('selection', ['exported_refit', 'direct_evaluation'])
def test_restore_keeps_explicit_deep_model_at_its_original_path(tmp_path, selection):
    old, lab, _ = checkpoint(tmp_path)
    directory = 'shared/generalization/primary/_evaluated_model/selected'
    if selection == 'exported_refit':
        execute(old, lab, 'write', {'path': 'selected.py', 'code': FROZEN_FIT})
        result = execute(old, lab, 'run', {'operator': 'refit', 'model': 'selected.py',
            'path': directory, 'records': [0, 1], 'validation_records': [1]})
        selected = result['tested_model']
        assert selected in old.candidate_paths
    else:
        selected = directory + '/_evaluated_model/selected.py'
        execute(old, lab, 'write', {'path': selected, 'code': FROZEN_FIT})
        write_dependency(old.workspace, directory + '/_evaluated_model/model_observations.json',
                         json.dumps([asdict(old.rows[0])]))
        result = execute(old, lab, 'evaluate', {'model': selected})
        assert result['returncode'] == 0
        assert result['executed_path'] == selected
        assert selected not in old.candidate_paths and selected not in old.research_candidates
    previous = saved_research(old, lab)
    continued = agent_at(tmp_path / 'continued')
    target = TemporalInquiryLab(task_index=1)
    continued.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert (continued.workspace / selected).read_text() == (old.workspace / selected).read_text()
    assert continued.costs == old.costs and target.costs == lab.costs
    assert continued.operator_results == old.operator_results
    assert continued.candidate_paths == old.candidate_paths
    score = evaluate_python_model(continued.workspace, selected,
                                 [Observation({'u': 0.}, old.rows[0].target)])
    assert score['returncode'] == 0 and score['mse'] == 0, score


def test_restore_keeps_same_context_binding_and_frozen_grandchild(tmp_path, monkeypatch):
    old, lab, _ = checkpoint(tmp_path)
    parent = 'shared/generalization/parent/_evaluated_model'
    grand = 'shared/generalization/grand/_evaluated_model'
    write_dependency(old.workspace / parent, 'parent.py', """def bind_models(models):
    global child
    child = models['local']
def predict(row): return child(row) + 1.
""")
    write_dependency(old.workspace / parent, 'child.py', """def bind_models(models):
    global grand
    grand = models['grand']
def predict(row): return grand(row) + 1.
""")
    write_dependency(old.workspace / parent, 'model_bindings.json', json.dumps({
        'parent.py': {'local': 'child.py'}, 'child.py': {'grand': grand + '/grand.py'}}))
    write_dependency(old.workspace / parent / grand, 'grand.py',
                     "import json\ndef predict(row): return json.load(open('parameter.json'))\n")
    write_dependency(old.workspace / parent / grand, 'parameter.json', '3')
    root_grand = parent + '/shared/root_dependency/_evaluated_model'
    write_dependency(old.workspace, 'root_parent.py', """def bind_models(models):
    global child
    child = models['local']
def predict(row): return child(row) + 1.
""")
    write_dependency(old.workspace, 'root_child.py', """def bind_models(models):
    global grand
    grand = models['grand']
def predict(row): return grand(row) + 1.
""")
    write_dependency(old.workspace, 'model_bindings.json', json.dumps({
        'root_parent.py': {'local': 'root_child.py'},
        'root_child.py': {'grand': root_grand + '/grand.py'}}))
    write_dependency(old.workspace / root_grand, 'grand.py',
                     "import json\ndef predict(row): return json.load(open('parameter.json'))\n")
    write_dependency(old.workspace / root_grand, 'parameter.json', '4')
    junk = old.workspace / parent / 'shared/generalization/unreferenced/_evaluated_model'
    write_dependency(junk, 'history.bin', 'Unreferenced history must not be copied.')
    previous = saved_research(old, lab)
    original_scandir = os.scandir

    def scandir(path):
        if not isinstance(path, int):
            assert Path(path) != junk, 'restore descended into an unrelated sibling context'
        return original_scandir(path)

    monkeypatch.setattr(os, 'scandir', scandir)
    if hasattr(Path, '_accessor'):
        monkeypatch.setattr(Path._accessor, 'scandir', scandir)
    continued = agent_at(tmp_path / 'continued')
    continued.restore_research(previous, old.workspace, TemporalInquiryLab(task_index=1),
                               confirmation_seed=SEED)
    assert not (continued.workspace / junk.relative_to(old.workspace)).exists()
    assert (continued.workspace / parent / grand / 'parameter.json').read_text() == '3'
    score = evaluate_python_model(continued.workspace, parent + '/parent.py',
                                 [Observation({'u': 0.}, 5.)])
    assert score['returncode'] == 0 and score['mse'] == 0, score
    score = evaluate_python_model(continued.workspace, 'root_parent.py', [Observation({'u': 0.}, 6.)])
    assert score['returncode'] == 0 and score['mse'] == 0, score
    assert continued.costs == old.costs and continued.operator_results == old.operator_results
