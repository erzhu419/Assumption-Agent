"""New cases must separate validation evidence from the later settled apparatus."""

from copy import deepcopy
from random import Random

from scripts.python_selective_method_cases import case_spec, KINDS, protocol
from scripts.run_python_method_learning_study import laboratory, protocol as previous_protocol


def sources():
    return [{"case": {"apparatus_seed": 971106001 + index * 1009,
                      "changed_response_seed": 981106001 + index * 1009,
                      "seed": 991106001 + index * 1009},
             "candidate": None if index == 0 else {"candidate_id": "candidate_000"}}
            for index in range(4)]


def test_four_practices_and_eight_transfer_cases_use_new_deterministic_inputs():
    inputs = sources()
    saved = deepcopy(inputs)
    old = previous_protocol(inputs)
    plan = protocol(inputs)
    assert inputs == saved and plan == protocol(inputs)
    assert plan["author_source_index"] == 1
    assert list(plan["practice_cases"]) == [f"case_{i}" for i in range(4)]
    assert tuple(spec["kind"] for spec in plan["practice_cases"].values()) == KINDS
    assert len(plan["transfer_cases"]) == 8
    assert {spec["source_index"] for spec in plan["transfer_cases"]} == {2, 3}
    assert plan["limits"] == old["limits"] and plan["prefix"] == old["prefix"]
    for spec in [*plan["practice_cases"].values(), *plan["transfer_cases"]]:
        assert spec["confirmation_seed"] == spec["task"]["seed"] + 7919
        assert spec["task"]["seed"] not in {s["task"]["seed"] for s in
            [*old["practice_cases"].values(), *old["transfer_cases"]]}
        if spec["kind"] == "post_validation_change":
            change = spec["task"]["calibration_change"]
            assert spec["task"]["response_seed"] == spec["case"]["changed_response_seed"]
            assert change["after_rows"] == 96
            assert change["apparatus_seed"] == spec["task"]["apparatus_seed"]
            assert change["response_seed"] != spec["task"]["response_seed"]


def test_checkpoint_keeps_initial_96_rows_then_changes_fresh_and_terminal():
    spec = case_spec(sources()[1], "post_validation_change", seed_offset=300001, response_offset=400003)
    moving = laboratory(spec)
    initial_spec = deepcopy(spec)
    change = initial_spec["task"].pop("calibration_change")
    initial = laboratory(initial_spec)
    settled_spec = deepcopy(initial_spec)
    settled_spec["task"]["response_seed"] = change["response_seed"]
    settled = laboratory(settled_spec)
    rng = Random(spec["task"]["seed"])
    actions = [{"kind": "trajectory", "channels": [],
                "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]} for _ in range(5)]
    # Prefix and first rejected predictor's fresh trajectory precede the update.
    for action in actions[:4]:
        assert moving.execute(action)["rows"] == initial.execute(action)["rows"]
    assert moving.costs["observation_cost"] == 96
    changed_rows = moving.execute(actions[4])["rows"]
    assert changed_rows == settled.execute(actions[4])["rows"]
    assert changed_rows != initial.execute(actions[4])["rows"]
    assert moving.public_spec()["settled_after_observation_rows"] == 96
    # Terminal scoring uses the settled law even without buying a hidden prefix.
    unused = laboratory(spec)
    before = deepcopy(unused.costs)
    assert unused.evaluation_rows(length=48, seed=727) == settled.evaluation_rows(length=48, seed=727)
    assert unused.costs == before
