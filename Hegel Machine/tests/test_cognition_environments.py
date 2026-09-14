"""Check experiment properties that could invalidate cognitive comparisons."""

from dataclasses import asdict

from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.models import Probe


def test_probe_replay_and_history_reset_are_deterministic():
    environment = make_suite()[3]
    probe = environment.spec.available_probes[3]
    first = environment.observe(probe)
    environment.observe(environment.spec.available_probes[0])
    assert environment.observe(probe) == first
    assert first[0].history == ()
    assert len(first) == probe.cost
    for index, observation in enumerate(first):
        assert len(observation.history) == index
        assert set(observation.features) == {"x", "z"}
        for earlier, raw_history in zip(first, observation.history):
            assert raw_history == {**earlier.features, "__target__": earlier.target}


def test_evaluation_inputs_are_independent_and_do_not_change_observation_stream():
    environment = make_suite()[3]
    probe = environment.spec.initial_probes[0]
    before = environment.observe(probe)
    evaluation = environment.evaluation_rows()
    assert len(evaluation) == 192
    assert environment.evaluation_rows() == evaluation
    assert environment.observe(probe) == before
    training_inputs = {
        tuple(sorted(row.items()))
        for query in (*environment.spec.initial_probes, *environment.spec.available_probes)
        for row in query.sequence
    }
    assert all(tuple(sorted(row.features.items())) not in training_inputs for row in evaluation)
    assert make_suite(8)[3].evaluation_rows() != evaluation


def test_controller_specs_do_not_label_the_required_cognitive_action():
    environments = make_suite()
    assert len(environments) == 17
    assert {environment.evaluation_metadata["phase"] for environment in environments} == {
        "acquire", "transfer", "conflict", "unrelated", "regime_change"
    }
    first = environments[0].spec
    for environment in environments:
        spec = environment.spec
        assert spec.description == first.description
        assert spec.context == first.context
        assert len(spec.feature_names) == 2
        assert set(spec.feature_bounds) == set(spec.feature_names)
        if spec.feature_names == first.feature_names:
            assert spec.initial_probes == first.initial_probes
            assert spec.available_probes == first.available_probes


def test_common_probes_distinguish_history_and_feature_conflicts():
    environments = make_suite()
    history_environment = environments[1]
    probe = history_environment.spec.available_probes[3]
    history_rows = history_environment.observe(probe)
    # The same current controls can require different predictions after a jump.
    assert history_rows[1].features["x"] == history_rows[3].features["x"] == 0.0
    assert history_rows[1].target != history_rows[3].target

    conflict = environments[10]
    intervention = conflict.spec.available_probes[4]
    rows = conflict.observe(intervention)
    assert len({row.features["x"] for row in rows}) == 1
    assert len({row.target for row in rows}) == 2
    history_conflict = environments[11]
    assert history_conflict.observe(probe)[1].target == history_conflict.observe(probe)[3].target


def test_hysteresis_exposes_raw_history_without_supplying_a_state_feature():
    environment = make_suite()[2]
    rows = environment.observe(environment.spec.available_probes[8])
    assert rows[2].features["x"] == rows[6].features["x"] == 0.0
    assert rows[2].target == 0.0
    assert rows[6].target == 1.0
    assert rows[6].history[-1]["__target__"] == 1.0
    assert all(set(row.features) == {"x", "z"} for row in rows)


def test_forward_interface_has_no_mechanism_partition_or_repair_hints():
    environments = make_suite(101, partition="forward")
    assert len(environments) == 12
    for environment in environments:
        spec = asdict(environment.spec)
        assert "partition" not in spec and "mechanism" not in spec
        assert environment.spec.description == environments[0].spec.description
        assert environment.spec.context == environments[0].spec.context
        assert set(environment.spec.feature_bounds) == set(environment.spec.feature_names)
        assert environment.evaluation_metadata["partition"] == "forward"
    assert environments[5].spec.feature_names == ("p", "q")
    assert environments[6].spec.feature_names == ("r", "s")
    assert environments[1].spec.task_id == environments[-1].spec.task_id


def test_forward_history_can_require_more_than_the_last_three_observations():
    environment = make_suite(101, partition="forward")[1]
    positive = Probe("positive_history", ({"x": 1.0, "z": 0.0},) + ({"x": 0.0, "z": 0.0},) * 4)
    negative = Probe("negative_history", ({"x": -1.0, "z": 0.0},) + ({"x": 0.0, "z": 0.0},) * 4)
    left, right = environment.observe(positive), environment.observe(negative)
    assert left[-1].features == right[-1].features
    assert [row["x"] for row in left[-1].history[-3:]] == [row["x"] for row in right[-1].history[-3:]]
    assert left[-1].target != right[-1].target
    assert environment.observe(positive) == left


def test_old_partition_task_interfaces_and_reference_relations_remain_available():
    development = make_suite(7, partition="development")
    confirmation = make_suite(23, partition="confirmation")
    assert [len(development), len(confirmation)] == [17, 12]
    assert development[0].spec.task_id == "t01" and confirmation[0].spec.task_id == "c01"
    assert development[-1].spec.task_id == "t02" and confirmation[-1].spec.task_id == "c02"
    row = development[0].observe(development[0].spec.initial_probes[0])[0]
    assert row.target == 1.4 * row.features["x"] + 0.3
