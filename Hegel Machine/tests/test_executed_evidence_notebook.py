"""Empirical memory follows executed records; these checks do not certify LLM adoption."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import code_agent
from hegel_machine.cognition.code_provider_context import prepare_code_context
from test_common_archive_online_research import READOUT, prepared
from test_common_archive_resume import SEED, agent_at, checkpoint, execute
from test_inquiry_revision import PublicLab, action, revision
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab


STALE_NOTE = "Revision 0 tested the fresh rows 2–3 and refuted their forecasts."


class AccountedPublicLab(PublicLab):
    @property
    def costs(self):
        return {"observation_cost": self.paid, "measurement_cost": 0,
                "queries": len(self.actions)}

    def restore_paid_account(self, observations, costs):
        self.paid = costs["observation_cost"]


def evidence_scene(path, arm):
    agent, lab = prepared(path, arm), AccountedPublicLab()
    execute(agent, lab, "query", {"actions": [action(0)]})
    execute(agent, lab, "write", {"path": "readout.py", "code": READOUT})
    execute(agent, lab, "query", {
        "actions": [action(2)],
        "revision": revision(["old.py", "new.py"], readout="readout.py"),
        "question": {"id": "contrast", "question": "Do these predictions agree?",
                     "parent_id": "task", "depends_on": []}})
    execute(agent, lab, "query", {"actions": [action(3, 4)], "notes": STALE_NOTE})
    execute(agent, lab, "run", {"operator": "analyze_revision", "revision_index": 0,
                                "path": "readout.py", "notes": STALE_NOTE})
    return agent, lab


@pytest.mark.parametrize("arm", ["code_react", "hegel"])
def test_old_revision_and_later_acquisition_keep_distinct_executed_facts(tmp_path, arm):
    agent, lab = evidence_scene(tmp_path, arm)
    before = deepcopy(agent.costs)
    request = agent._request(lab)
    notebook = request["evidence_notebook"]
    old = notebook["facts"]["revision:0"]
    fresh = notebook["facts"]["record:2"]

    assert notebook["current_paid_rows"] == 4
    assert old["record_indices"] == [1]
    assert old["response_row_ranges"] == [[1, 1]]
    assert old["forecast_evidence_cutoff"] == {"rows": 1, "records": 1}
    assert old["paid_rows_at_analysis"] == 4
    assert old["newer_paid_row_count"] == 2
    assert sorted(item["prospective_mse"] for item in old["forecast_results"]) == [0., 4.]
    assert old["evidence_status"] == "not_checked"
    assert old["source"] == {"file": "research_state.json", "json_path": ["revisions", 0]}
    assert fresh["response_row_range"] == [2, 3]
    assert fresh["forecast_revision_indices"] == []
    assert "forecast_results" not in fresh
    assert "mse" not in fresh
    assert fresh["source"] == {"file": "laboratory_observations.json", "json_path": [2]}
    assert request["research_notes"] == STALE_NOTE
    assert (tmp_path / "research_notes.md").read_text() == STALE_NOTE
    assert agent.events[-1]["arguments"]["notes"] == STALE_NOTE
    assert agent.costs == before and lab.paid == 4


def test_both_arms_receive_the_same_factual_notebook(tmp_path):
    code, code_lab = evidence_scene(tmp_path / "code", "code_react")
    hegel, hegel_lab = evidence_scene(tmp_path / "hegel", "hegel")
    assert code._request(code_lab)["evidence_notebook"] == hegel._request(hegel_lab)["evidence_notebook"]


def test_interrupted_silent_computation_is_returned_as_unfinished_analysis(tmp_path):
    agent, lab = evidence_scene(tmp_path, 'hegel')
    agent.events.append({'action': 'run', 'arguments': {'path': 'slow.py'},
        'feedback': {'status': 'runtime_interrupted', 'executed_path': 'slow.py',
                     'execution_seconds': 300., 'stdout': '', 'stderr': ''}})
    facts = agent._request(lab)['evidence_notebook']['facts']
    interrupted = facts[f'analysis:{len(agent.events) - 1}']
    assert interrupted['status'] == 'runtime_interrupted'
    assert interrupted['executed_path'] == 'slow.py'
    assert interrupted['execution_seconds'] == 300.
    assert 'mse' not in interrupted and 'returncode' not in interrupted
    data = facts['response_data']
    assert data['row_fields'] == list(json.loads((tmp_path / 'observations.json').read_text())[0])
    assert data['rows'] == len(agent.rows)


def test_computation_output_and_unscored_rows_have_their_own_references(tmp_path):
    agent, lab = evidence_scene(tmp_path, 'hegel')
    execute(agent, lab, 'evaluate', {'model': 'new.py'})
    execute(agent, lab, 'query', {'actions': [action(5)]})
    feedback = execute(agent, lab, 'run', {'code': "print('analysis of existing evidence')"})
    facts = agent._request(lab)['evidence_notebook']['facts']
    assert facts['current_paid_evaluation']['evaluated_response_row_range'] == [0, 3]
    assert facts['current_paid_evaluation']['unscored_response_row_range'] == [4, 4]
    index = len(agent.events) - 1
    analysis = facts[f'analysis:{index}']
    assert analysis['stdout'] == feedback['stdout']
    assert analysis['source']['json_path'] == ['events', index, 'feedback']
    assert facts['record:3']['forecast_revision_indices'] == []
    assert 'stdout' not in facts['record:3']


def test_restoration_preserves_fact_references_executed_event_and_authored_note(tmp_path):
    old, lab = evidence_scene(tmp_path / "source", "hegel")
    prior_notebook = old._request(lab)["evidence_notebook"]
    previous = old.result("model_call_budget_exhausted")
    resumed, target = prepared(tmp_path / "restored", "hegel"), AccountedPublicLab()
    restored = resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)

    assert restored["new_observations"] == 0
    assert target.actions == []
    assert resumed.events == previous["events"]
    assert resumed._request(target)["evidence_notebook"] == prior_notebook
    assert (resumed.workspace / "research_notes.md").read_text() == STALE_NOTE
    assert resumed.revisions[0]["returned_answer"]["evidence"] == [1]


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_cited_old_revision_retains_actual_inputs_and_existing_reanalysis_action(tmp_path, arm):
    agent, lab = evidence_scene(tmp_path, arm)
    # Move the old experiment beyond the four recent revisions and the active
    # question's latest answer. The author's citation must keep its real inputs.
    for _ in range(5):
        execute(agent, lab, 'run', {'operator': 'analyze_evidence', 'record_indices': [2],
            'path': 'readout.py', 'research_question': {'id': 'contrast'}})
    assert 'revision:0' not in agent._request(lab)['evidence_notebook']['facts']
    note = 'revision:0 established my interpretation; revision:999 is an invalid reference.'
    agent._save('research_notes.md', note)
    before = deepcopy((agent.revisions, agent.questions, agent.costs, lab.actions))

    request = agent._request(lab)
    context = prepare_code_context(tmp_path, request, schema={}, instructions='runtime test')
    fact = context['evidence_notebook']['facts']['revision:0']
    assert fact['record_indices'] == [1]
    assert fact['record_references'][0]['control_runs'] == [
        {'steps': [0, 0], 'controls': {'u': 2}}]
    assert fact['response_row_ranges'] == [[1, 1]]
    assert fact['analysis_version'] == 2
    assert fact['evidence_status'] == 'not_checked'
    assert 'do not validate experiment-role descriptions' in fact['check_scope']
    assert fact['reanalysis_action'] == {'action': 'run', 'arguments': {
        'operator': 'analyze_revision', 'revision_index': 0, 'path': 'readout.py',
        'research_question': {'id': 'contrast'}}}
    assert 'current authored file' in fact['reanalysis_source']
    assert 'old source is not restored' in fact['reanalysis_source']
    assert fact['executed_readout_source'] == {'file': 'research_state.json',
        'json_path': ['revisions', 0, 'analysis_attempts', 1, 'source']}
    state = json.loads((tmp_path / 'research_state.json').read_text())
    for key in fact['executed_readout_source']['json_path']:
        state = state[key]
    assert state == before[0][0]['analysis_attempts'][-1]['source']
    assert 'revision:999' not in context['evidence_notebook']['facts']
    assert context['research_notes'] == note
    assert (agent.revisions, agent.questions, agent.costs, lab.actions) == before

    # The same action accepts an authored correction without buying evidence,
    # discarding the previous readout or silently rewriting its interpretation.
    correction = """def readout(result,spec,memory):
    return {'status':'inconclusive', 'answer':'The original interpretation is withdrawn.',
            'evidence':result['observations'][0]['row_indices']}
"""
    execute(agent, lab, 'write', {'path': 'readout.py', 'code': correction})
    action = fact['reanalysis_action']
    execute(agent, lab, action['action'], action['arguments'])
    corrected = agent._request(lab)['evidence_notebook']['facts']['revision:0']
    assert corrected['analysis_version'] == 3
    assert corrected['authored_readout_status'] == 'inconclusive'
    assert corrected['executed_readout_source']['json_path'] == [
        'revisions', 0, 'analysis_attempts', 2, 'source']
    assert agent.revisions[0]['analysis_attempts'][-1]['source'] == correction
    assert agent.revisions[0]['analysis_attempts'][:2] == before[0][0]['analysis_attempts']
    assert lab.actions == before[3] and agent.costs['observation_cost'] == before[2]['observation_cost']


def test_failed_confirmation_retains_actual_scored_fields_after_restore(tmp_path):
    old, lab, previous = checkpoint(tmp_path)
    executed = old.confirmations[0]
    fact = old._request(lab)["evidence_notebook"]["facts"]["confirmation:0"]
    assert fact["kind"] == "independent_confirmation"
    assert fact["passed"] is False
    assert fact["mse"] == executed["mse"] > executed["mse_threshold"]
    for key in ("rows", "mse_threshold", "observation_indices", "returncode", "model"):
        if key in executed:
            assert fact[key] == executed[key]
    assert "observations" not in fact
    assert fact["confirmation_index"] == 0
    paid = old._evidence_notebook()["facts"]["current_paid_evaluation"]
    assert paid["kind"] == "paid_fit"
    assert paid["support"] == "new_paid_rows_unscored"

    resumed, target = agent_at(tmp_path / "restored"), TemporalInquiryLab(task_index=1)
    resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)
    assert resumed._evidence_notebook()["facts"]["confirmation:0"] == fact
    assert resumed.events == old.events


@pytest.mark.parametrize("empirical_notebook", [False, True])
def test_provider_uses_reference_contract_only_for_executed_notebook(tmp_path, monkeypatch,
                                                                  empirical_notebook):
    revised = "revision:0 rejects the old response; record:2 still needs interpretation."
    request = {"task_interface": "Use the public laboratory.", "research_notes": STALE_NOTE}
    if empirical_notebook:
        request["evidence_notebook"] = {"current_paid_rows": 4, "facts": {
            "revision:0": {"response_row_ranges": [[1, 1]]},
            "record:2": {"response_row_range": [2, 3], "forecast_revision_indices": []}}}

    def run(command, prompt, timeout, cwd):
        context = json.loads(prompt)
        assert context["research_notes"] == STALE_NOTE
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        instructions = (Path(cwd) / "code_agent_instructions.txt").read_text()
        description = schema["properties"]["research_notes"]["description"]
        assert schema["properties"]["research_notes"]["type"] == "string"
        assert "research_notes" in schema["required"]
        if empirical_notebook:
            assert context["evidence_notebook"] == request["evidence_notebook"]
            assert "citing evidence_notebook references" in description
            assert "research_notes is the prior author's interpretation and can be wrong" in instructions
            assert "revision:N, record:N, confirmation:N" in instructions
            assert "COMPLETE current notebook" not in instructions
        else:
            assert "observed evidence" in description
            assert "COMPLETE current notebook" in instructions
            assert "evidence_notebook is empirical memory" not in instructions
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps({
            "action": "run", "arguments_json": json.dumps({"code": "print('inspect evidence')"}),
            "source_code": None, "research_notes": revised}))
        return 0, json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 31, "output_tokens": 7}}), "", False

    monkeypatch.setattr(code_agent, "_run", run)
    provider = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test")
    result = provider.next_action(request)
    assert result["arguments"] == {"code": "print('inspect evidence')", "notes": revised}
    assert result["usage"]["total_tokens"] == 38
