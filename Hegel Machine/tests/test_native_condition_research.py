"""Historical condition research consumes only public contracts and paid rows."""

from copy import deepcopy
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.native_condition_research import research_native_conditions
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from hegel_machine.cognition import native_condition_research as native_research


def public_spec(*, discrete=False):
    spec = {
        "controls": ["drive", "switch"],
        "feature_bounds": {"drive": [-1., 1.], "switch": [-1., 1.]},
        "max_probe_length": 16,
        "action_kinds": ["trajectory"],
        "channels": [],
        "observation_cost_per_row": 1,
        "reset": "Every trajectory starts independently.",
        "prediction_contract": "Current and preceding controls only; no past targets.",
    }
    if discrete:
        spec["control_levels"] = {name: [-1., -.5, 0., .5, 1.]
                                  for name in spec["controls"]}
    return spec


def record(sequence, first, *, changed=False):
    """Synthetic labelled fixture independent of any running apparatus."""
    history, rows = [], []
    for point in sequence:
        window = [*history, point][-3:]
        active = point["switch"] >= .1
        if changed:
            target = sum(.3 + .2 * past["drive"] for past in window)
        elif active:
            target = sum(-.4 - .7 * past["drive"] for past in window)
        else:
            target = sum(.2 + 1.2 * past["drive"] for past in window)
        rows.append({"features": dict(point), "target": target,
                     "history": deepcopy(history)})
        history.append(dict(point))
    return {"action": {"kind": "trajectory", "channels": [],
                       "sequence": deepcopy(sequence)},
            "observation_index": first,
            "row_indices": list(range(first, first + len(rows))),
            "rows": rows, "measurements": [],
            "observation_cost": len(rows), "measurement_cost": 0}


def random_sequence(seed, length=16):
    rng = Random(seed)
    return [{"drive": rng.choice([-.9, -.4, .2, .7]),
             "switch": rng.choice([-1., -.5, 0., .5, 1.])}
            for _ in range(length)]


def paid_fixture():
    background = {"drive": .6, "switch": -.8}
    reference = [dict(background) for _ in range(16)]
    sweep = [dict(background) for _ in range(16)]
    for index in range(2, 16):
        sweep[index]["switch"] = [-1., -.5, 0., .5, 1.][min(4, (index - 2) // 3)]
    sequences = [reference, sweep, *(random_sequence(seed) for seed in (23, 29, 37))]
    return [record(sequence, 100 + index * 16) for index, sequence in enumerate(sequences)]


def observed(rows):
    return [Observation(dict(row["features"]), row["target"],
                        tuple(deepcopy(row["history"]))) for row in rows]


def test_matched_paid_condition_produces_executable_history_predictor():
    records = paid_fixture()
    before = deepcopy(records)
    budget = SearchBudget(600_000)
    result = research_native_conditions(public_spec(), records, budget,
        matched_pairs=[{"reference_record_index": 0, "record_index": 1}])
    assert records == before
    assert result["compute_cost"] == budget.spent > 0
    assert result["candidates"]
    best = min(result["candidates"], key=lambda candidate: candidate["fit_mse"])
    assert best["fit_mse"] < 1e-8
    model = Program.from_dict(best["program"])
    exported = {}
    exec(export_temporal_program(model, feature_names=public_spec()["controls"]), exported)
    fresh = record(random_sequence(79), 900)
    direct = [model.predict(row) for row in observed(fresh["rows"])]
    frozen = [exported["predict"]({"features": row["features"], "history": row["history"]})
              for row in fresh["rows"]]
    assert frozen == pytest.approx(direct)
    assert sum((value - row["target"]) ** 2 for value, row in zip(frozen, fresh["rows"])) / len(frozen) < 1e-8
    assert result["proposed_experiments"] == []
    assert "fit" not in exported


def test_zero_budget_and_no_pair_does_not_invent_condition_support():
    records = [record(random_sequence(4), 200)]
    before = deepcopy(records)
    result = research_native_conditions(public_spec(), records, SearchBudget(0))
    assert result["status"] == "compute_budget_exhausted"
    assert result["candidates"] == []
    assert result["proposed_experiments"] == []
    assert result["compute_cost"] == 0
    assert records == before


def test_equal_suffix_with_different_prefix_is_not_a_matched_axis_experiment():
    left, right = random_sequence(4), random_sequence(7)
    right[8:] = deepcopy(left[8:])
    records = [record(left, 0), record(right, 16)]
    result = research_native_conditions(public_spec(), records, SearchBudget(1000))
    assert result["status"] == "needs_matched_evidence"
    assert result["evidence"] == []
    assert result["candidates"] == []


def test_planned_experiments_respect_public_discrete_levels_without_buying():
    records = []
    budget = SearchBudget(2000)
    result = research_native_conditions(public_spec(discrete=True), records, budget,
        observation_budget=120)
    assert records == []
    assert result["candidates"] == []
    assert result["proposed_experiments"]
    total = 0
    for proposal in result["proposed_experiments"]:
        request = proposal["query_action"]
        assert request["action"] == "query"
        for action in request["arguments"]["actions"]:
            assert action["kind"] == "trajectory" and action["channels"] == []
            assert 1 <= len(action["sequence"]) <= 16
            total += len(action["sequence"])
            for point in action["sequence"]:
                assert set(point) == {"drive", "switch"}
                assert all(value in public_spec(discrete=True)["control_levels"][name]
                           for name, value in point.items())
    assert total <= 120
    assert 0 < result["compute_cost"] == budget.spent <= budget.limit


def test_paid_midpoint_refines_joint_fit_and_next_experiment_with_one_row_allowance():
    records = paid_fixture()
    original = deepcopy(records)
    intervals, programs, experiments = [], [], []
    for _ in range(3):
        budget = SearchBudget(600_000)
        result = research_native_conditions(public_spec(), records, budget,
            matched_pairs=[{"reference_record_index": 0, "record_index": 1}],
            observation_budget=1)
        proposal, = result["proposed_experiments"]
        assert proposal["purpose"] == "conditional_boundary"
        assert proposal["matched_action_pairs"] == []
        assert proposal["observation_cost"] == 1
        assert proposal["predicted_separation"] > 1e-3
        assert 0 < result["compute_cost"] == budget.spent <= budget.limit
        action, = proposal["query_action"]["arguments"]["actions"]
        point, = action["sequence"]
        left, right = proposal["threshold_interval"]
        assert left < .1 <= right
        assert point["switch"] == (left + right) / 2.
        assert result["candidates"][0]["fit_mse"] < 1e-8
        intervals.append((left, right))
        programs.append(result["candidates"][0]["program"])
        experiments.append(deepcopy(action))
        measured = record(action["sequence"], 100 + sum(len(r["rows"]) for r in records))
        assert min((prediction - measured["rows"][0]["target"]) ** 2
                   for prediction in proposal["branch_predictions"]) < 1e-8
        records.append(measured)
    assert records[:len(original)] == original
    assert len({str(action) for action in experiments}) == 3
    assert programs[0] != programs[1] != programs[2]
    assert [right - left for left, right in intervals] == pytest.approx([.5, .25, .125])


def test_boundary_does_not_offer_an_unavailable_discrete_midpoint():
    result = research_native_conditions(public_spec(discrete=True), paid_fixture(),
        SearchBudget(600_000), observation_budget=1)
    assert result["candidates"]
    assert result["proposed_experiments"] == []


def test_boundary_proposal_keeps_its_candidate_when_only_one_is_requested():
    from hegel_machine.cognition.native_condition_research import update_native_boundary

    records = paid_fixture()
    previous = research_native_conditions(public_spec(), records, SearchBudget(600_000),
                                          observation_budget=1)
    proposal = previous["proposed_experiments"][0]
    action = proposal["query_action"]["arguments"]["actions"][0]
    records.append(record(action["sequence"], 180))
    updated = update_native_boundary(public_spec(), records, SearchBudget(10_000),
        previous=previous, proposal=proposal, max_candidates=1, observation_budget=1)
    candidate, = updated["candidates"]
    boundary, = updated["proposed_experiments"]
    assert boundary["candidate_program"] == candidate["program"]
    assert boundary["threshold_interval"] == [0., .25]


@pytest.mark.parametrize("stage", [1, 2])
def test_cold_public_laboratory_tuple_histories_reach_joint_boundary_fit(stage):
    from scripts.run_conditional_discovery_study import laboratory

    lab = laboratory({"seed": 131011031, "apparatus_seed": 131011031,
                      "stage": stage, "noise_sigma": 0.})
    records = [lab.execute(action) for action in lab.confirmation_actions(72, Random(131111034))]
    assert isinstance(records[0]["rows"][0]["history"], tuple)
    budget = SearchBudget(640_000)
    first = research_native_conditions(lab.public_spec(), records, budget, observation_budget=96)
    assert first["status"] == "needs_matched_evidence"
    for action in first["proposed_experiments"][0]["query_action"]["arguments"]["actions"]:
        records.append(lab.execute(action))
    second = research_native_conditions(lab.public_spec(), records, budget, observation_budget=1)
    assert second["evidence"]
    assert min(item["fit_mse"] for item in second["candidates"]) < 1e-8
    boundary, = second["proposed_experiments"]
    assert boundary["purpose"] == "conditional_boundary"
    left, right = boundary["threshold_interval"]
    records.append(lab.execute(boundary["query_action"]["arguments"]["actions"][0]))
    third = research_native_conditions(lab.public_spec(), records, budget, observation_budget=1)
    refined, = third["proposed_experiments"]
    new_left, new_right = refined["threshold_interval"]
    assert new_right - new_left == pytest.approx((right - left) / 2)
    assert len(third["fit_evidence"]) == 121
    assert budget.remaining > 500_000


def test_tiny_compute_budget_never_overspends_matched_evidence():
    budget = SearchBudget(1)
    result = research_native_conditions(public_spec(), paid_fixture(), budget,
        matched_pairs=[{"reference_record_index": 0, "record_index": 1}])
    assert result["candidates"] == []
    assert result["compute_cost"] == budget.spent <= 1


def test_explicit_pair_rejects_changed_uncontrolled_history():
    records = paid_fixture()[:2]
    records[1]["action"]["sequence"][0]["drive"] = -.2
    records[1] = record(records[1]["action"]["sequence"], 116)
    result = research_native_conditions(public_spec(), records, SearchBudget(10_000),
        matched_pairs=[{"reference_record_index": 0, "record_index": 1}])
    assert result['status'] == 'research_not_executed'
    assert result['reason'] == 'invalid_matched_evidence'
    assert result['candidates'] == [] and result['evidence'] == []
    assert result['invalid_pairs'][0]['changed_controls'] == ['drive', 'switch']


def test_sequential_multi_control_selection_returns_actual_differences_not_scientific_refutation():
    records = paid_fixture()
    sequence = deepcopy(records[0]['action']['sequence'])
    sequence[0]['drive'] = .9
    sequence[2]['switch'] = .5
    records.append(record(sequence, 180))
    original = deepcopy(records)
    budget = SearchBudget(600_000)
    result = research_native_conditions(public_spec(), records, budget,
        matched_pairs=[{'reference_record_index': 0, 'record_index': 1},
                       {'reference_record_index': 0, 'record_index': 5}])
    assert result['status'] == 'research_not_executed' and result['error']
    assert not result['candidates'] and not result['evidence']
    assert not result['proposed_experiments'] and result['searched_candidates'] == 0
    invalid, = result['invalid_pairs']
    assert invalid['control_changes'] == {
        'drive': [{'step': 0, 'reference': .6, 'intervention': .9}],
        'switch': [{'step': 2, 'reference': -.8, 'intervention': .5}]}
    assert invalid['reference_evidence'] == records[0]['row_indices']
    assert invalid['evidence'] == records[5]['row_indices']
    assert [(pair['reference_record_index'], pair['record_index'])
            for pair in result['available_pairs']] == [(0, 1)]
    assert budget.spent == sum(len(item['rows']) for item in records) < budget.limit
    assert records == original


def test_longer_constant_reference_uses_only_the_aligned_paid_prefix():
    records = paid_fixture()
    records[1] = record(records[1]["action"]["sequence"][:12], 116)
    original = deepcopy(records)
    result = research_native_conditions(public_spec(), records, SearchBudget(30_000),
        matched_pairs=[{"reference_record_index": 0, "record_index": 1}])
    evidence = result["evidence"][0]
    assert evidence["matched_rows"] == 12
    assert evidence["reference_evidence"] == records[0]["row_indices"][:12]
    assert evidence["evidence"] == records[1]["row_indices"]
    assert result["fit_evidence"] == [index for item in records for index in item["row_indices"]]
    assert records == original


def pulse_levels_fixture(background_drive=.6):
    """Separate two-level interventions jointly distinguish a switched law."""
    background = {"drive": background_drive, "switch": -.8}
    sequences = [[dict(background) for _ in range(16)]]
    for level in (1., -.4, 0., .2, .6):
        sequence = [dict(background) for _ in range(16)]
        for index in (5, 10):
            sequence[index]["switch"] = level
        sequences.append(sequence)
    return [record(sequence, index * 16) for index, sequence in enumerate(sequences)]


def test_pooled_paid_levels_refute_smooth_explanation_of_single_pulse():
    records = pulse_levels_fixture()
    original = deepcopy(records)
    selected = research_native_conditions(public_spec(), records, SearchBudget(600_000),
        matched_pairs=[{"reference_record_index": 0, "record_index": 1}])
    assert selected["evidence"][0]["status"] == "smooth"
    budget = SearchBudget(600_000)
    result = research_native_conditions(public_spec(), records, budget)
    evidence, = result["evidence"]
    assert evidence["status"] == "conditional"
    assert evidence["smooth_loss"] > 1e-4
    assert evidence["conditional_loss"] < 1e-10
    assert evidence["matched_rows"] == 80
    assert evidence["reference_record_indices"] == [0]
    assert evidence["record_indices"] == [1, 2, 3, 4, 5]
    assert "record_index" not in evidence and "reference_record_index" not in evidence
    assert evidence["reference_evidence"] == records[0]["row_indices"] * 5
    assert evidence["evidence"] == [row for rec in records[1:] for row in rec["row_indices"]]
    assert result["candidates"][0]["fit_mse"] < 1e-8
    assert 0 < evidence["compute_cost"] <= result["compute_cost"] == budget.spent <= budget.limit
    assert records == original


def test_explicit_pairs_pool_only_selected_evidence_and_keep_fit_scope():
    records = pulse_levels_fixture()
    selected = [1, 3, 4]
    result = research_native_conditions(public_spec(), records, SearchBudget(600_000),
        record_indices=[0, 1], matched_pairs=[
            {"reference_record_index": 0, "record_index": index} for index in selected])
    evidence, = result["evidence"]
    assert evidence["record_indices"] == selected
    assert evidence["evidence"] == [row for index in selected for row in records[index]["row_indices"]]
    assert result["fit_record_indices"] == [0, 1]
    assert result["fit_evidence"] == records[0]["row_indices"] + records[1]["row_indices"]


def test_matched_levels_with_different_backgrounds_are_assessed_separately():
    records = pulse_levels_fixture(.6)[:2] + pulse_levels_fixture(-.3)[:2]
    for index, item in enumerate(records):
        item["observation_index"] = 16 * index
        item["row_indices"] = list(range(16 * index, 16 * (index + 1)))
    result = research_native_conditions(public_spec(), records, SearchBudget(600_000),
        matched_pairs=[{"reference_record_index": 0, "record_index": 1},
                       {"reference_record_index": 2, "record_index": 3}])
    assert len(result["evidence"]) == 2
    assert [item["background"]["drive"] for item in result["evidence"]] == [.6, -.3]
    assert [item["record_index"] for item in result["evidence"]] == [1, 3]
    assert all(item["matched_rows"] == 16 for item in result["evidence"])


def test_old_candidate_is_scored_on_selected_paid_rows_without_refitting():
    records = paid_fixture()
    for index in (2, 4):
        records[index] = record(records[index]["action"]["sequence"],
                                records[index]["observation_index"], changed=True)
    constant = Program("const", value=0.)
    original = deepcopy(records)
    result = research_native_conditions(public_spec(), records, SearchBudget(600_000),
        record_indices=[2, 4],
        matched_pairs=[{"reference_record_index": 0, "record_index": 1}],
        programs=[constant])
    assert records == original
    retained = next(item for item in result["candidates"]
                    if item["source"] == "retained_program")
    assert retained["program"] == constant.to_dict()
    targets = [row["target"] for index in (2, 4) for row in records[index]["rows"]]
    assert retained["fit_mse"] == pytest.approx(sum(value ** 2 for value in targets) / len(targets))
    assert result["fit_evidence"] == [*records[2]["row_indices"], *records[4]["row_indices"]]


@pytest.mark.parametrize("argument", ["basis", "programs"])
def test_fitting_terms_cannot_read_previous_response_targets(argument):
    # The historical native lab recorded these labels internally; the current
    # prediction interface exposes only controls, including at later steps.
    with pytest.raises(ValueError, match="public control"):
        research_native_conditions(public_spec(), paid_fixture(), SearchBudget(50_000),
            matched_pairs=[], **{argument: [Program("feature", name="__target__")]})


def test_returned_predictor_stays_frozen_and_can_be_refuted_by_new_records():
    result = research_native_conditions(public_spec(), paid_fixture(), SearchBudget(600_000),
        matched_pairs=[{"reference_record_index": 0, "record_index": 1}])
    candidate = min(result["candidates"], key=lambda item: item["fit_mse"])
    saved = deepcopy(candidate["program"])
    namespace = {}
    exec(export_temporal_program(Program.from_dict(saved),
                                feature_names=public_spec()["controls"]), namespace)
    counterexample = record(random_sequence(79), 900, changed=True)
    errors = [(namespace["predict"](row) - row["target"]) ** 2
              for row in counterexample["rows"]]
    assert sum(errors) / len(errors) > .01
    assert candidate["program"] == saved


def test_two_measured_controls_reuse_composed_condition_search():
    spec = public_spec()
    spec["controls"] = ["drive", "left", "right"]
    spec["feature_bounds"] = {name: [-1., 1.] for name in spec["controls"]}
    background = {"drive": .6, "left": .75, "right": .75}
    sequences = [[dict(background) for _ in range(16)] for _ in range(3)]
    for index, control in enumerate(("left", "right"), 1):
        for step in range(2, 16):
            sequences[index][step][control] = [-1., -.5, 0., .5, 1.][min(4, (step - 2) // 3)]
    rng = Random(812)
    sequences.extend([[{name: rng.choice([-1., -.5, 0., .5, 1.])
                        for name in spec["controls"]} for _ in range(16)] for _ in range(6)])

    def measured(sequence, first):
        history, rows = [], []
        for point in sequence:
            active = point["left"] >= .1 and point["right"] >= -.1
            target = sum((-.4 - .7 * old["drive"]) if active else
                         (.2 + 1.2 * old["drive"]) for old in [*history, point][-2:])
            rows.append({"features": dict(point), "target": target, "history": deepcopy(history)})
            history.append(dict(point))
        return {"action": {"kind": "trajectory", "channels": [], "sequence": deepcopy(sequence)},
                "rows": rows, "row_indices": list(range(first, first + len(rows))),
                "observation_index": first, "observation_cost": len(rows),
                "measurement_cost": 0, "measurements": []}

    records = [measured(sequence, index * 16) for index, sequence in enumerate(sequences)]
    result = research_native_conditions(spec, records, SearchBudget(1_000_000),
        matched_pairs=[{"reference_record_index": 0, "record_index": index} for index in (1, 2)])
    composed = [candidate for candidate in result["candidates"]
                if candidate["source"].startswith("composed_condition:")]
    assert composed
    best = min(composed, key=lambda candidate: candidate["fit_mse"])
    assert best["fit_mse"] < 1e-8
    model = Program.from_dict(best["program"])
    sequence = [{name: rng.choice([-1., -.5, 0., .5, 1.])
                 for name in spec["controls"]} for _ in range(16)]
    fresh = measured(sequence, 1000)
    predicted = [model.predict(row) for row in observed(fresh["rows"])]
    assert sum((value - row["target"]) ** 2 for value, row in zip(predicted, fresh["rows"])) / 16 < 1e-8


@pytest.fixture(scope="module")
def boundary_research_state():
    """A genuine fitted conditional model, shared only as immutable test input."""
    records = paid_fixture()
    result = research_native_conditions(public_spec(), records, SearchBudget(600_000),
        matched_pairs=[{"reference_record_index": 0, "record_index": 1}],
        observation_budget=1)
    proposal, = result["proposed_experiments"]
    candidate = next(item for item in result["candidates"]
                     if item["source"] == proposal["candidate_source"])
    result["candidates"] = [candidate]
    return records, result, proposal


def append_boundary_measurement(records, proposal):
    action, = proposal["query_action"]["arguments"]["actions"]
    first = max(index for item in records for index in item["row_indices"]) + 1
    return [*deepcopy(records), record(action["sequence"], first)]


def forbid_structure_refit(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A purchased boundary point must not restart structural fitting")

    for name in ("research_native_conditions", "propose_measured_window_response",
                 "propose_experimental_conditional_coordinates", "propose_composed_conditions"):
        monkeypatch.setattr(native_research, name, unexpected)


def assert_candidates_scored_on_paid_rows(result, records):
    rows = [row for item in records for row in observed(item["rows"])]
    evidence = [index for item in records for index in item["row_indices"]]
    assert result["fit_evidence"] == evidence
    for candidate in result["candidates"]:
        model = Program.from_dict(candidate["program"])
        actual = sum((model.predict(row) - row.target) ** 2 for row in rows) / len(rows)
        assert candidate["fit_mse"] == pytest.approx(actual, abs=1e-12)
        assert candidate["fit_evidence"] == evidence


def test_paid_boundary_updates_parameters_without_repeating_structure_fit(
        boundary_research_state, monkeypatch):
    records, previous, proposal = deepcopy(boundary_research_state)
    before = deepcopy(previous)
    records = append_boundary_measurement(records, proposal)
    forbid_structure_refit(monkeypatch)
    budget = SearchBudget(10_000)

    result = native_research.update_native_boundary(public_spec(), records, budget,
        previous=previous, proposal=proposal, observation_budget=1)

    assert previous == before
    update = result["boundary_update"]
    left, right = proposal["threshold_interval"]
    assert update["before_interval"] == [left, right]
    assert update["after_interval"] == pytest.approx([left, (left + right) / 2])
    assert result["research_decision"]["kind"] == "remaining_parameter_ambiguity"
    assert_candidates_scored_on_paid_rows(result, records)
    retained = next(item for item in result["candidates"]
                    if item["program"] == before["candidates"][0]["program"])
    assert retained["research_program"] == before["candidates"][0]["research_program"]
    assert min(item["fit_mse"] for item in result["candidates"]) < 1e-8
    refined, = result["proposed_experiments"]
    assert refined["threshold_interval"] == update["after_interval"]
    assert refined["candidate_program"] in [item["program"] for item in result["candidates"]]
    assert 0 < result["compute_cost"] == budget.spent < before["compute_cost"]


def test_sequential_boundary_updates_keep_old_models_and_research_metadata(
        boundary_research_state, monkeypatch):
    records, previous, proposal = deepcopy(boundary_research_state)
    original = deepcopy(previous["candidates"][0])
    forbid_structure_refit(monkeypatch)
    intervals = []

    for _ in range(2):
        records = append_boundary_measurement(records, proposal)
        previous = native_research.update_native_boundary(public_spec(), records,
            SearchBudget(10_000), previous=previous, proposal=proposal, observation_budget=1)
        assert_candidates_scored_on_paid_rows(previous, records)
        intervals.append(previous["boundary_update"]["after_interval"])
        proposal, = previous["proposed_experiments"]

    assert [right - left for left, right in intervals] == pytest.approx([.25, .125])
    retained = next(item for item in previous["candidates"]
                    if item["program"] == original["program"])
    assert retained["research_program"] == original["research_program"]
    # The second measurement refutes the original midpoint. Its executable
    # model remains available with its actual error, rather than a stale fit.
    assert retained["fit_mse"] > 1e-3
    assert min(item["fit_mse"] for item in previous["candidates"]) < 1e-8
    assert proposal["candidate_program"] != original["program"]


def test_boundary_contradicting_both_predictions_requests_structure_revision(
        boundary_research_state, monkeypatch):
    records, previous, proposal = deepcopy(boundary_research_state)
    records = append_boundary_measurement(records, proposal)
    records[-1]["rows"][0]["target"] = max(proposal["branch_predictions"]) + 10.
    forbid_structure_refit(monkeypatch)

    result = native_research.update_native_boundary(public_spec(), records,
        SearchBudget(10_000), previous=previous, proposal=proposal, observation_budget=1)

    assert result["research_decision"]["kind"] == "structural_mismatch"
    assert result["boundary_update"]["before_interval"] == proposal["threshold_interval"]
    assert result["boundary_update"]["after_interval"] == proposal["threshold_interval"]
    assert result["boundary_update"]["matched_branch"] is None
    assert min(result["boundary_update"]["branch_errors"]) > 1.
    assert_candidates_scored_on_paid_rows(result, records)
    assert any(item["program"] == previous["candidates"][0]["program"]
               for item in result["candidates"])
    assert min(item["fit_mse"] for item in result["candidates"]) > 1e-3
    assert result["proposed_experiments"] == []


def test_boundary_update_preserves_scorable_incumbent_when_compute_runs_out(
        boundary_research_state):
    records, previous, proposal = deepcopy(boundary_research_state)
    records = append_boundary_measurement(records, proposal)
    budget = SearchBudget(sum(len(item["rows"]) for item in records))

    result = native_research.update_native_boundary(public_spec(), records, budget,
        previous=previous, proposal=proposal, observation_budget=1)

    retained = next(item for item in result["candidates"]
                    if item["program"] == previous["candidates"][0]["program"])
    assert retained["research_program"] == previous["candidates"][0]["research_program"]
    assert_candidates_scored_on_paid_rows(result, records)
    assert result["status"] == "candidates_require_evaluation"
    assert result["research_decision"]["kind"] == "compute_budget_exhausted"
    assert result["compute_cost"] == budget.spent <= budget.limit


def test_full_search_retains_candidate_procedure_when_only_rescoring_is_affordable(
        boundary_research_state):
    records, previous, _ = deepcopy(boundary_research_state)
    candidate = previous["candidates"][0]
    before = deepcopy(candidate)
    budget = SearchBudget(2 * sum(len(item["rows"]) for item in records))

    result = research_native_conditions(public_spec(), records, budget,
        retained_candidates=[candidate], observation_budget=1)

    assert candidate == before
    retained = next(item for item in result["candidates"]
                    if item["program"] == candidate["program"])
    assert retained["research_program"] == candidate["research_program"]
    assert_candidates_scored_on_paid_rows(result, records)
    assert retained["fit_mse"] < 1e-8
    assert result["status"] == "search_incomplete"
    assert result["research_decision"]["kind"] == "computation_incomplete"
    assert result["compute_cost"] == budget.spent <= budget.limit


def test_completed_local_evidence_does_not_claim_unfunded_joint_fits_failed():
    records = paid_fixture()
    retained = Program('const', value=0.)
    result = research_native_conditions(public_spec(), records, SearchBudget(2600),
        programs=[retained])
    local, window, conditional, _ = result['search_stages']
    assert local['status'] == 'attempt_finished' and local['completed_fits'] > 0
    assert window['status'] == conditional['status'] == 'computation_incomplete'
    assert window['completed_fits'] == conditional['completed_fits'] == 0
    assert window['blocked_work'] == {'required_compute': 640,
        'available_compute': 264, 'additional_compute': 376}
    assert result['status'] == 'search_incomplete'
    assert result['research_decision']['completed_joint_fits'] == 0
    assert result['searched_candidates'] == 0
    candidate, = result['candidates']
    assert candidate['source'] == 'retained_program'
    assert candidate['program'] == retained.to_dict()
    assert_candidates_scored_on_paid_rows(result, records)

    # Giving the same operator enough work reaches actual fits and keeps the
    # earlier executable program alongside new evidence-selected candidates.
    completed = research_native_conditions(public_spec(), records, SearchBudget(6000),
        retained_candidates=result['candidates'])
    assert completed['status'] == 'candidates_require_evaluation'
    assert completed['searched_candidates'] > 0
    assert sum(item['completed_fits'] for item in completed['search_stages']
               if item['stage'].startswith('joint_')) > 0
    assert min(item['fit_mse'] for item in completed['candidates']) < 1e-8
    assert any(item['program'] == retained.to_dict() for item in completed['candidates'])


def test_unusable_fit_is_not_reported_as_unfunded_computation(monkeypatch):
    from hegel_machine.cognition.research_programs import _Fitter
    rows = observed(paid_fixture()[0]['rows'])
    budget = SearchBudget(10000)
    fitter = _Fitter(rows, budget, .5)
    assert fitter.fit([Program('feature', name='missing')]) is None
    assert fitter.fit([Program('const', value=1.)]) is None
    assert budget.denied_requests == 0 and budget.last_denied_request is None
    assert budget.completed_fits == 0

    # An executed search with no returned candidate must not infer a compute
    # shortfall from the candidate count alone.
    for name in ('propose_measured_window_response',
                 'propose_experimental_conditional_coordinates', 'propose_composed_conditions'):
        monkeypatch.setattr(native_research, name, lambda *args, **kwargs: [])
    result = research_native_conditions(public_spec(), paid_fixture(), SearchBudget(10000))
    assert result['status'] == 'matched_evidence_without_candidate'
    assert result['searched_candidates'] == 0
    assert all(stage['status'] == 'attempt_finished' for stage in result['search_stages'])
    assert result['research_decision']['kind'] != 'computation_incomplete'
