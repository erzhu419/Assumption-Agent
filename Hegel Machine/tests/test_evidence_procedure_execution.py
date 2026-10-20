"""Scripted integration of checked evidence and an authored research procedure.

The toy formula is a runtime fixture, not a candidate for the live Luna task.
"""

from copy import deepcopy
import json

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_common_archive_online_research import ConfirmationLab
from test_executable_evidence_binding import return_local_checks, return_region
from test_inquiry_revision import action


PROGRAM = "shared/evidence_procedure.py"
PARENT = '''import json
def bind_research(inputs):
    global region
    region = inputs['slope']['region']
def fit(rows):
    global p, training_inputs
    training_inputs = [row['features']['u'] for row in rows]
    disk = json.load(open('laboratory_observations.json'))
    assert [r['features']['u'] for record in disk for r in record['rows']] == training_inputs
    estimate = sum(row['features']['u'] * row['target'] for row in rows) / sum(u*u for u in training_inputs)
    p = estimate if region['lower'] < estimate < region['upper'] else (region['lower'] + region['upper']) / 2
def get_parameters(): return {'slope': p, 'training_inputs': training_inputs}
def predict(row): return p * row['features']['u']
'''


def prepared(tmp_path):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm="hegel", source_task=False,
        config=InquiryLLMConfig(observation_budget=30, confirmation_rows=3,
            execution_seconds=30., compute_budget=10000))
    lab = ConfirmationLab()
    agent._execute("query", {"actions": [action(-1., 1.), action(-.5, .5),
                                          action(.75, .25)]}, lab)
    agent._execute("write", {"path": "parent.py", "code": PARENT}, lab)
    agent._execute("evaluate", {"model": "incumbent.py",
        "code": "def predict(row): return row['features']['u']\n"}, lab)
    return agent, lab


def procedure_source(inputs):
    # Selection and its source are supplied by the author. The executor performs
    # the whole existing bind/refit/compare sequence without another model turn.
    return f'''import json
def select_actions(spec, observations, memory):
    state = json.load(open('research_state.json'))
    own = [r for r in state['method_results'] if r['path'] == {PROGRAM!r}]
    stage = len(own) % 4
    if stage == 0:
        return {{'action':'run', 'arguments':{{'operator':'bind_research',
            'model':'parent.py', 'inputs':{inputs!r}}}}}
    if stage == 1:
        return {{'action':'run', 'arguments':{{'operator':'refit', 'model':'parent.py',
            'validation_records':[2], 'path':'shared/refits/round_' + str(len(own)//4)}}}}
    if stage == 2:
        return {{'action':'evaluate', 'arguments':{{'model':own[-1]['feedback']['path']}}}}
    return {{'stop':'Return the actual validation and candidate comparison to the author.'}}
'''


def execute(agent, lab):
    return agent._execute("run", {"operator": "procedure", "path": PROGRAM,
                                  "max_steps": 8}, lab)


def test_same_authored_procedure_consumes_changed_evidence_refits_and_preserves_incumbent(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, 0., .5)
    local = return_local_checks(agent, lab, expected=2.)
    agent._save(PROGRAM, procedure_source({"slope": "slope_question",
        "contrast": {"revision_index": local["index"], "check_ids": ["difference"]}}))
    original_program = (tmp_path / PROGRAM).read_text()
    original_candidate = deepcopy(agent.evaluated_model)
    paid_before = agent.costs["observation_cost"]

    first = execute(agent, lab)
    assert first["status"] == "method_stopped"
    assert len(first["steps"]) == 4
    fit = first["steps"][1]["feedback"]
    assert fit["split"]["validation_records"] == [2]
    assert fit["validation"]["mse"] > 0
    consumed = fit["validation"]["research_bindings"][0]
    assert consumed["inputs"]["slope"]["version"] == 1
    assert consumed["actual_parameters"] == {"slope": .25,
        "training_inputs": [-1., 1., -.5, .5]}
    assert consumed["local_relations"][0]["checks"][0]["residual"] == -1.5
    comparison = first["steps"][2]["feedback"]
    assert comparison["retention_decision"]["status"] == "incumbent_retained"
    assert agent.evaluated_model == original_candidate
    first_path = fit["tested_model"]

    # Re-analysis changes the declared child; no new candidate formula or paid
    # observation is inserted. The same complete program consumes that revision.
    return_region(agent, lab, .9, 1.1)
    assert agent.research_candidates["parent.py"]["status"] == "needs_rebuild"
    second = execute(agent, lab)
    assert second["status"] == "method_stopped"
    assert len(second["steps"]) == 4
    updated = second["steps"][1]["feedback"]
    assert updated["split"] == fit["split"]
    assert updated["tested_model"] != first_path
    assert updated["validation"]["mse"] == 0.
    assert updated["validation"]["research_bindings"][0]["inputs"]["slope"]["version"] == 2
    assert updated["validation"]["research_bindings"][0]["actual_parameters"]["slope"] == 1.
    assert second["steps"][2]["feedback"]["mse"] == 0.
    assert (tmp_path / PROGRAM).read_text() == original_program
    assert (tmp_path / "parent.py").read_text() == PARENT
    assert (tmp_path / first_path).is_file()
    assert agent.costs["observation_cost"] == paid_before == lab.paid
    assert agent.costs["model_calls"] == 0
    assert not agent.confirmations and agent.model is None


def test_procedure_returns_bound_validation_evidence_conflict_before_fitting(tmp_path):
    agent, lab = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1)
    local = return_local_checks(agent, lab, expected=2.)
    # The selected local contrast uses record 0. Requesting that very record as
    # held-out evidence must return for revision instead of scoring a leaked fit.
    source = procedure_source({"slope": "slope_question",
        "contrast": {"revision_index": local["index"]}}).replace(
            "'validation_records':[2]", "'validation_records':[0]")
    agent._save(PROGRAM, source)
    retained = deepcopy(agent.evaluated_model)
    paid_before = agent.costs["observation_cost"]
    result = execute(agent, lab)
    assert result["status"] == "method_error"
    assert len(result["steps"]) == 2
    assert "validation labels are already bound into fitting" in result["steps"][-1]["feedback"]["error"]
    assert agent.evaluated_model == retained
    assert not (tmp_path / "shared/refits/round_0").exists()
    assert agent.costs["observation_cost"] == paid_before
    assert agent.costs["model_calls"] == 0 and not agent.confirmations
