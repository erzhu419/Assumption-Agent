"""The additional case changes actual instrument dynamics after refit validation."""

from copy import deepcopy
from random import Random

from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.python_scoped_method_cases import KINDS, NEW_KIND, case_spec, protocol
from scripts.python_selective_method_cases import protocol as previous_protocol
from scripts.run_python_method_learning_study import laboratory, trial_preparer
from test_python_method_learning import archived_predictor


def sources():
    return [{"case": {"apparatus_seed": 971106001 + i * 1009,
                      "changed_response_seed": 981106001 + i * 1009,
                      "seed": 991106001 + i * 1009},
             "candidate": None if i == 0 else {"candidate_id": "candidate_000"}}
            for i in range(4)]


def test_fifth_condition_preserves_all_prior_cases_and_transfer_order():
    inputs = sources()
    original = deepcopy(inputs)
    old, plan = previous_protocol(inputs), protocol(inputs)
    assert inputs == original and plan == protocol(inputs)
    assert tuple(plan["target_kinds"]) == KINDS
    assert len(plan["practice_cases"]) == 5 and len(plan["transfer_cases"]) == 10
    assert {key: plan["practice_cases"][key] for key in old["practice_cases"]} == old["practice_cases"]
    assert plan["transfer_cases"][:8] == old["transfer_cases"]
    assert plan["limits"] == old["limits"] and plan["prefix"] == old["prefix"]
    for spec in [plan["practice_cases"]["case_4"], *plan["transfer_cases"][8:]]:
        assert spec["kind"] == NEW_KIND
        assert spec["task"]["instrument_mode"] == "finite2"
        assert spec["task"]["response_seed"] == spec["case"]["changed_response_seed"]
        assert spec["task"]["calibration_change"] == {"after_rows": 96, "instrument_mode": "recurrent"}
    first = deepcopy(plan["practice_cases"]["case_3"]["task"])
    second = deepcopy(plan["practice_cases"]["case_4"]["task"])
    first.pop("calibration_change")
    second.pop("calibration_change")
    assert first == second


def test_actual_prefix_is_finite_then_fresh_and_terminal_use_recurrent():
    spec = protocol(sources())["practice_cases"]["case_4"]
    initial_spec = deepcopy(spec)
    initial_spec["task"].pop("calibration_change")
    settled_spec = deepcopy(initial_spec)
    settled_spec["task"]["instrument_mode"] = "recurrent"
    changing, initial, settled = map(laboratory, (spec, initial_spec, settled_spec))
    rng = Random(spec["task"]["seed"])
    actions = [{"kind": "trajectory", "channels": ["meter"],
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]} for _ in range(5)]
    for action in actions[:4]:
        actual, expected = changing.execute(action), initial.execute(action)
        assert actual["rows"] == expected["rows"] and actual["measurements"] == expected["measurements"]
    assert changing.costs["observation_cost"] == 96
    actual, finite, recurrent = changing.execute(actions[4]), initial.execute(actions[4]), settled.execute(actions[4])
    assert actual["rows"] == recurrent["rows"] and actual["rows"] != finite["rows"]
    assert [item["values"] for item in actual["measurements"]] == [item["values"] for item in recurrent["measurements"]]
    assert [item["values"] for item in actual["measurements"]] != [item["values"] for item in finite["measurements"]]
    unused = laboratory(spec)
    before = deepcopy(unused.costs)
    assert unused.evaluation_rows(length=48, seed=727) == settled.evaluation_rows(length=48, seed=727)
    assert unused.costs == before


def test_real_scoped_refit_still_fails_after_structure_change_but_meter_repair_works(tmp_path, archived_predictor):
    source = {"case": {"seed": 71, "apparatus_seed": 910061, "changed_response_seed": 910079},
              "candidate": archived_predictor}
    spec = case_spec(source, NEW_KIND, seed_offset=0, response_offset=0)
    plan = protocol([source])
    payload = trial_preparer(source, plan)(spec, tmp_path / "target",
        "def select_actions(spec, records, memory): return {'stop': 'test uses actual atomic operators'}\n",
        InquiryLLMConfig(**plan["limits"]))
    agent, lab = payload["agent"], payload["laboratory"]
    prior = archived_predictor["model"]
    old = agent._execute("finish", {"model": prior}, lab)
    assert not old["fresh_confirmation"]["passed"] and len(agent.rows) == 96
    before = agent._execute("run", {"operator": "refit", "model": prior, "path": "refit_before"}, lab)
    assert before["validation"]["mse"] < 1e-10
    changed = agent._execute("finish", {"model": before["path"]}, lab)
    assert not changed["fresh_confirmation"]["passed"]
    rng = Random(727)

    def query(count, channels):
        return agent._execute("query", {"actions": [{"kind": "trajectory", "channels": channels,
            "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]} for _ in range(count)]}, lab)

    query(1, [])
    after = agent._execute("run", {"operator": "refit", "model": prior, "path": "refit_settled",
        "records": [4, 5], "validation_records": [5]}, lab)
    assert after["status"] == "refit_inadequate_on_validation"
    assert after["validation"]["mse"] > 1e-10 and after["final_fitting_rows"] == 48
    assert agent.costs["measurement_cost"] == 0
    query(3, ["meter"])
    instrument = agent._execute("run", {"operator": "instrument", "path": "instrument.json",
        "records": [6, 7], "compute_limit": 100000}, lab)
    agent._execute("run", {"operator": "assess_instrument", "instrument": instrument["path"], "records": [8]}, lab)
    response = agent._execute("run", {"operator": "response", "instrument": instrument["path"],
        "records": [6, 7, 8], "validation_records": [8], "output_dir": "settled_response"}, lab)
    candidate = min(response["candidates"], key=lambda item: item["validation_mse"])
    assert candidate["validation_mse"] < 1e-10
    confirmed = agent._execute("finish", {"model": candidate["path"]}, lab)
    assert confirmed["fresh_confirmation"]["passed"]
    terminal = evaluate_python_model(agent.workspace, agent.model, lab.evaluation_rows(length=48, seed=2027))
    assert terminal["mse"] < 1e-10
    assert agent.costs["observation_cost"] == 240 and agent.costs["measurement_cost"] == 216
    assert agent.costs["model_calls"] == agent.costs["reported_tokens"] == 0
