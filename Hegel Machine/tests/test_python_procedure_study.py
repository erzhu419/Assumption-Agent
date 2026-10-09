"""Keep the macro intervention isolated and service/failure costs intact."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.run_python_procedure_study import (
    DEFAULT_RESEARCH_PROCEDURE, PROCEDURE_INTERFACE, PROGRAM_PATH,
    execution_evidence, prepare_target, protocol, summarize, target_jobs,
)


def test_shared_program_and_single_step_work_but_atomic_rejects_macro(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "model.py").write_text("def predict(features):\n    return 0.0\n")
    candidate = {"candidate_id": "candidate_000", "training_rows": 72,
        "snapshot_workspace": str(archive), "inner_model": "model.py",
        "model": "shared/confirmed_models/candidate_000/_evaluated_model/model.py"}
    source = {"case": {"apparatus_seed": 971107010, "changed_response_seed": 981107010,
                       "seed": 991107010}, "candidate": candidate, "costs": {}}
    plan = protocol("live", [source])
    jobs = target_jobs([source], tmp_path, plan)
    payloads = {job["condition"]: prepare_target(job, plan) for job in jobs
                if job["kind"] == "response_parameter_change"}
    bundled, atomic = (payloads[name]["agent"] for name in ("bundled", "atomic"))
    assert bundled.common_interface == atomic.common_interface + PROCEDURE_INTERFACE
    assert PROCEDURE_INTERFACE not in atomic.common_interface
    assert bundled.files[PROGRAM_PATH] == atomic.files[PROGRAM_PATH] == DEFAULT_RESEARCH_PROCEDURE
    assert bundled.observations == atomic.observations
    assert all(bundled.costs[key] == atomic.costs[key] for key in
               ("observation_cost", "measurement_cost", "compute_cost", "model_calls", "reported_tokens"))
    # A generic program action remains available in both conditions. This tests
    # the actual shared single-invocation path, not a synthetic rejected command.
    program = ("def select_actions(spec, observations, memory):\n"
               "    return {'action': 'write', 'arguments': {'path': 'result.txt', 'code': 'done'}}\n")
    for payload in payloads.values():
        agent = payload["agent"]
        agent._save("step.py", program)
        feedback = agent._execute("query", {"path": "step.py"}, payload["laboratory"])
        assert feedback["written"] == "result.txt"
        assert len(agent.method_results) == 1
    before = deepcopy(atomic.costs)
    with pytest.raises(ValueError, match="unavailable in the atomic condition"):
        atomic._execute("run", {"operator": "procedure", "path": "step.py", "max_steps": 1},
                        payloads["atomic"]["laboratory"])
    assert atomic.costs == before and len(atomic.method_results) == 1
    outcome = bundled._execute("run", {"operator": "procedure", "path": "step.py", "max_steps": 1},
                               payloads["bundled"]["laboratory"])
    assert outcome["status"] == "method_step_limit" and len(bundled.method_results) == 2
    assert bundled.costs["model_calls"] == atomic.costs["model_calls"] == 0


def test_accounting_retains_prior_source_failure_unknown_service_usage_and_inner_steps(tmp_path):
    sources = [{"case": {"apparatus_seed": index, "changed_response_seed": 20 + index,
                         "seed": 30 + index}, "candidate": {"candidate_id": str(index)} if index else None,
                "status": "completed" if index else "model_call_budget_exhausted",
                "costs": {"model_calls": 3 if index else 12, "reported_tokens": 30 if index else 120,
                          "observation_cost": 72}} for index in range(4)]
    plan = protocol("live", sources)
    jobs = target_jobs(sources, tmp_path, plan)
    assert len(jobs) == 12 and all(job["case"]["apparatus_seed"] != 0 for job in jobs)
    assert {job["condition"] for job in jobs} == {"atomic", "bundled"}
    # Repeated copies of nested feedback do not turn internal operations into
    # additional provider calls or additional executions.
    operators = [{"operator": "research_choice"}, {"operator": "refit"}]
    methods = [{"proposal": {"action": "run", "arguments": {"operator": item["operator"]}},
                "feedback": deepcopy(item)} for item in operators]
    training = {"events": [{"action": "run", "arguments": {"operator": "procedure"},
                             "feedback": {"steps": deepcopy(methods), "operator_results": operators}}]}
    agent = SimpleNamespace(operator_results=operators, method_results=methods)
    evidence = execution_evidence(agent, training)
    assert evidence["operator_calls"] == {"research_choice": 1, "refit": 1}
    assert evidence["procedure_requests"] == 1 and evidence["method_steps"] == 2
    results = [{"condition": "bundled", "kind": "recurrent_instrument_change", "status": "provider_failed",
        "confirmed_and_correct": False, "false_confirmation": False,
        "costs": {"model_calls": 2, "reported_tokens": None, "observation_cost": 72},
        "known_tokens": 100, "input_tokens": 90, "cached_input_tokens": 40, "missing_usage_calls": 1,
        **evidence}]
    summary = summarize(sources, results, plan)
    assert summary["registered_episodes"] == 12
    assert summary["prior_sources_without_published_candidate"] == 1
    assert summary["existing_all_source_acquisition_costs"]["model_calls"] == 21
    assert summary["existing_reused_source_costs"]["model_calls"] == 9
    actual = summary["actual_new_targets"]
    assert actual["costs"]["model_calls"] == 2 and actual["costs"]["reported_tokens"] is None
    assert actual["statuses"] == {"provider_failed": 1}
    assert actual["known_tokens"] == 100 and actual["missing_usage_calls"] == 1
    assert actual["method_steps"] == 2 and actual["procedure_requests"] == 1
    assert summary["conditions"]["bundled"]["registered_targets"] == 6
    offline = protocol("offline", sources)
    assert len(target_jobs(sources, tmp_path, offline)) == 6 and offline["conditions"] == ["bundled"]
