"""The author sees and can practice the existing unchanged-source refit operation."""

from copy import deepcopy
import json
from pathlib import Path
from random import Random

from hegel_machine.cognition.anonymous_method_learning import (
    AnonymousPreparedMethodAgent, AnonymousResearchMethodLearner, EMPTY_METHOD,
)
from hegel_machine.cognition.common_archive_inquiry import _REFIT_INTERFACE
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_temporal_python_refit import DEPENDENCY, MODEL, LawLab, content


def make_author(path):
    # Reuse the existing test-only affine law and dependency that asserts it
    # receives only the current fitting records, including during validation.
    source_lab = LawLab(slope=1., offset=0.)
    rng = Random(71)
    source_records = [source_lab.execute({"kind": "trajectory", "channels": [],
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(12)]}) for _ in range(3)]
    source_rows = [row for record in source_records for row in record["rows"]]
    archive = "archive/source_000/_evaluated_model/"
    knowledge = {"files": {archive + name: value for name, value in {
        "model.py": MODEL, "fitting_dependency.py": DEPENDENCY,
        "settings.json": json.dumps({"basis": "linear"}),
        "observations.json": json.dumps(source_rows),
        "model_observations.json": json.dumps(source_rows),
        "laboratory_observations.json": json.dumps(source_records),
        }.items()}, "research_roles": {"predictors": [archive + "model.py"]}}
    knowledge["files"][METHOD_PATH] = EMPTY_METHOD
    limits = InquiryLLMConfig(observation_budget=192, confirmation_rows=12,
        execution_seconds=60., compute_budget=10000, max_model_calls=0, token_budget=0)
    prepared = []

    def prepare(case, directory, source, config):
        lab = LawLab(slope=-2., offset=.4)
        agent = AnonymousPreparedMethodAgent(None, directory, arm="hegel",
            source_task=False, knowledge=deepcopy(knowledge), config=config,
            confirmation_seed=173)
        # Three distinct complete trajectories give the operator an actual
        # separate validation record, as in the historical refit fixture.
        random = Random(93)
        actions = [{"kind": "trajectory", "channels": [],
            "sequence": [{"u": random.uniform(-1., 1.)} for _ in range(12)]}
            for _ in range(3)]
        agent._execute("query", {"actions": actions}, lab)
        agent.bind_method(source)
        payload = {"agent": agent, "laboratory": lab, "inherited_candidates": []}
        prepared.append(payload)
        return payload

    author = AnonymousResearchMethodLearner(None, path, knowledge=knowledge,
        cases={"case_000": {"confirmation_seed": 173}},
        lab_factory=lambda case: LawLab(slope=-2., offset=.4),
        config=limits, practice_limits=limits, prepare_trial=prepare)
    return author, prepared


def test_actual_author_request_exposes_existing_trial_refit_contract(tmp_path):
    author, _ = make_author(tmp_path)
    request = author._request(author.interface)
    contract = request["task_interface"]
    assert _REFIT_INTERFACE in contract
    assert "fit/predict source in your self-contained" in contract
    assert "source-task fitted parameters and observations" in contract
    assert "validation_records selects whole laboratory records" in contract


def test_authored_method_refits_current_records_and_preserves_old_family(tmp_path):
    author, prepared = make_author(tmp_path)
    original = content(author.workspace / "archive")
    # Scripted method fixture; no scientific strategy is supplied to an LLM.
    method = '''import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    prior = memory['research_roles']['predictors'][0]
    if not steps:
        return {'action': 'evaluate', 'arguments': {'model': prior}}
    if len(steps) == 1:
        return {'action': 'run', 'arguments': {'operator': 'refit',
            'model': prior, 'path': 'shared/current_refit', 'validation_records': [2]}}
    return {'model': steps[-1]['feedback']['path']}
'''
    edited = author._execute("write", {"path": METHOD_PATH, "code": method}, author.interface)
    practice = edited["edit_practice"]
    assert practice["status"] == "completed", practice
    assert practice["fresh_confirmation"]["passed"]
    assert practice["fresh_confirmation"]["rows"] == 12
    assert practice["fresh_confirmation"]["mse"] < 1e-20
    assert practice["costs"]["observation_cost"] == 48
    assert practice["costs"]["model_calls"] == 0
    agent = prepared[0]["agent"]
    old_score, refit, submission = agent.method_results
    assert old_score["feedback"]["mse"] > .1
    result = refit["feedback"]
    assert result["status"] == "refit_supported_on_validation"
    assert result["validation"]["fit_available"]
    assert result["validation"]["mse"] < 1e-20
    assert result["split"]["training_records"] == [0, 1]
    assert result["split"]["validation_records"] == [2]
    assert result["final_fitting_rows"] == result["paid_rows_at_refit"] == 36
    assert submission["proposal"]["model"] == result["path"]
    candidate = agent.workspace / Path(result["path"]).parent
    trace = json.loads((candidate / "fit_trace.json").read_text())
    # The fixture records each real fit: validation first sees only records 0,1;
    # later evaluation/confirmation can fit all 36 current prefix observations.
    validation_fit = next(item for item in trace if item["rows"] == 24)
    assert validation_fit == {"rows": 24, "targets": [row["target"]
        for record in agent.observations[:2] for row in record["rows"]]}
    fitting = json.loads((candidate / "model_observations.json").read_text())
    paid_prefix = [row for record in agent.observations[:3] for row in record["rows"]]
    assert fitting == json.loads(json.dumps(paid_prefix))
    assert (candidate / "model.py").read_text() == MODEL
    retained = content(agent.workspace / "archive")
    assert {name: retained[name] for name in original} == original
    assert content(author.workspace / "archive") == original
