"""Reference boundaries constrain possible target activations, not point labels."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_response_constraints import (
    query_response_family,
    response_family_grid_support,
)


X = Program("feature", name="x")
ESTIMATOR = {"id": "linear", "basis": [X.to_dict()]}
FEATURES = {1: {"x": 0.}, 2: {"x": 1.}, 3: {"x": 3.}}
GRID = {"origin": 0., "step": .5, "lower": 0., "upper": 10.}


def response(number, setting, active):
    return {"crystal": number, "frequency": setting, "activated": active}


def reference_history():
    return [response(1, 0., True), response(1, -1., False),
            response(1, 1., False), response(2, 2., True),
            response(2, 1., False), response(2, 3., False)]


def indices(result):
    assert result["status"] == "COMPLETE", result
    support = {index for first, last in result["intervals"]
               for index in range(first, last + 1)}
    assert len(support) == result["grid_count"]
    return support


@pytest.mark.parametrize("offset,target_negative", ((0., False), (0., True), (.25, False)))
def test_scalar_projection_matches_exhaustive_family_queries(offset, target_negative):
    # Twenty-one public points include strict support endpoints. The negative
    # target case also requires preserving both disjoint activation regions.
    grid = {**GRID, "origin": offset, "lower": offset, "upper": 10. + offset}
    paid = reference_history()
    if target_negative:
        paid.append(response(3, 6., False))
    projected = response_family_grid_support(FEATURES, paid, ESTIMATOR, 3, grid)
    expected = set()
    for index in range(21):
        setting = offset + index * grid["step"]
        answer = query_response_family(FEATURES, paid, ESTIMATOR,
                                       queries=[response(3, setting, True)])
        assert answer["status"] in {"SAT_WITNESS", "UNSAT_VERIFIED"}, answer
        if answer["status"] == "SAT_WITNESS":
            expected.add(index)
    assert indices(projected) == expected
    if target_negative:
        assert projected["intervals"] == [[7, 11], [13, 17]]
    elif not offset:
        assert projected["intervals"] == [[7, 17]]


def test_redundant_reference_negative_does_not_claim_target_information():
    paid = reference_history()
    before = response_family_grid_support(FEATURES, paid, ESTIMATOR, 3, GRID)
    after = response_family_grid_support(FEATURES, paid, ESTIMATOR, 3, GRID,
                                        queries=[response(1, 100., False)])
    assert indices(after) == indices(before)


def test_useful_reference_boundary_reduces_unmeasured_target_support():
    paid = reference_history()
    assert all(item["crystal"] != 3 for item in paid)
    before = response_family_grid_support(FEATURES, paid, ESTIMATOR, 3, GRID)
    after = response_family_grid_support(FEATURES, paid, ESTIMATOR, 3, GRID,
                                        queries=[response(1, .5, False)])
    assert indices(after) == set(range(9, 18))
    assert indices(after) < indices(before)


def test_hypothetical_outcomes_leave_paid_evidence_and_bindings_unchanged():
    inputs = deepcopy((FEATURES, reference_history(), ESTIMATOR, GRID))
    original = deepcopy(inputs)
    features, paid, estimator, grid = inputs
    alternatives = [[response(1, .5, active)] for active in (True, False)]
    saved_alternatives = deepcopy(alternatives)
    for hypothetical in alternatives:
        answer = response_family_grid_support(features, paid, estimator, 3, grid,
                                              queries=hypothetical)
        assert answer["status"] == "COMPLETE"
    assert inputs == original
    assert alternatives == saved_alternatives
    assert indices(response_family_grid_support(features, paid, estimator, 3, grid)) \
        == set(range(7, 18))


@pytest.mark.parametrize("limit", (0, 8))
def test_incomplete_compute_cannot_be_read_as_empty_scientific_support(limit):
    result = response_family_grid_support(FEATURES, reference_history(), ESTIMATOR,
                                          3, GRID, compute_limit=limit)
    assert result["status"] == "UNKNOWN"
    assert "grid_count" not in result
    assert result["intervals"] == []
    assert result["compute_cost"] <= limit


@pytest.mark.parametrize("basis", (
    [Program("const", value=1.).to_dict()],
    [X.to_dict(), Program("mul", children=(X, X)).to_dict()],
), ids=("no_varying_term", "two_varying_terms"))
def test_unsupported_dimension_is_unknown_not_family_exclusion(basis):
    result = response_family_grid_support(FEATURES, reference_history()[:3],
                                          {"id": "unsupported", "basis": basis}, 3, GRID)
    assert result["status"] == "UNKNOWN"
    assert "grid_count" not in result


def test_unbounded_coefficient_and_unsearched_branches_are_unknown():
    unbounded = response_family_grid_support(FEATURES, reference_history()[:3],
                                             ESTIMATOR, 3, GRID)
    truncated = response_family_grid_support(FEATURES,
        [*reference_history(), response(3, 6., False)], ESTIMATOR, 3, GRID, branch_limit=1)
    for result in (unbounded, truncated):
        assert result["status"] == "UNKNOWN"
        assert "grid_count" not in result
        assert result["intervals"] == []


def test_many_target_negatives_outside_anchored_support_do_not_exhaust_branch_limit():
    paid = [*reference_history(), *(response(3, float(value), False) for value in range(100, 229))]
    grid = {**GRID, "upper": 300.}
    result = response_family_grid_support(FEATURES, paid, ESTIMATOR, 3, grid)
    assert indices(result) == set(range(7, 18))
    assert result["pruned_target_gaps"] == 129
    assert result["branch_count"] == result["branches_attempted"] == 1


def test_many_still_possible_target_gaps_remain_unknown_at_branch_limit():
    paid = [{**item, "frequency": 100. * item["frequency"]} for item in reference_history()]
    paid.extend(response(3, float(value), False) for value in range(350, 737, 3))
    grid = {"origin": 0., "step": 1., "lower": 0., "upper": 1000.}
    result = response_family_grid_support(FEATURES, paid, ESTIMATOR, 3, grid)
    assert result["status"] == "UNKNOWN"
    assert result["branch_count"] == 130
    assert result["pruned_target_gaps"] == 0
    assert "grid_count" not in result
    assert result["intervals"] == []


@pytest.mark.parametrize("upper,activated", ((.1, False), (.3, True)))
def test_target_search_beats_optimistic_reference_without_claiming_nonlocal_confirmation(upper, activated):
    from hegel_machine.cognition.world_response_consequences import (
        assess_response_consequences, select_response_grid_trial,
    )
    from hegel_machine.cognition.world_response_hypotheses import _propose_response_members

    def measured(number, value, setting=None):
        row = {"crystal": number, "measurements": {"x": {"value": value, "evidence": number}}}
        if setting is not None:
            row["frequency"] = {"value": setting, "evidence": number}
        return row

    # Distinct instrument grids are already supported. Here only three target
    # settings remain possible, while the reference has a finer control step.
    grid = {"origin": .25, "step": 1., "lower": .25, "upper": 10.25}
    features = {**FEATURES, 3: {"x": 1.5}}
    state = {"research_table": {
        "references": [measured(1, 0., 0.), measured(2, 1., 2.)],
        "targets": [measured(3, 1.5)]},
        "response_settings": reference_history(), "target_numbers": [3], "completed": [],
        "control_grid": grid, "control_grids": {"1": GRID, "2": GRID},
        "consequence_panel": [{"crystal": 3, "frequency": 4.25}]}
    original = deepcopy(state)
    members = _propose_response_members(state, [ESTIMATOR], "shared/research.py")
    assessed = assess_response_consequences(state, [ESTIMATOR], members)
    assert (assessed["selected_number"], assessed["selected_setting"]) == (1, .5)
    assert assessed["selection"]["guaranteed_nonlocal_eliminations"] == 0
    scientific = {key: deepcopy(assessed["family_research"][key]) for key in (
        "families", "signatures", "admitted_signatures", "possible_signatures",
        "excluded_prior_signatures", "nonlocal_progress")}

    chosen = select_response_grid_trial(state, [ESTIMATOR], assessed)
    assert chosen["status"] == "attainable_target_search"
    assert (chosen["selected_number"], chosen["selected_setting"]) == (3, 3.25)
    assert chosen["selection"]["guaranteed_nonlocal_eliminations"] == 0
    assert chosen["selection"]["guaranteed_search_reduction"] == 1
    assessment = chosen["family_research"]["operational_search"]
    reference = next(item for item in assessment["experiments"] if item["crystal"] == 1)
    target = next(item for item in assessment["experiments"] if item["crystal"] == 3)
    assert reference["reductions_by_outcome"] == [0, 1]
    assert reference["guaranteed_search_reduction"] == 0
    assert target["reductions_by_outcome"] == [1, 3]
    for key, value in scientific.items():
        assert chosen["family_research"][key] == value
    assert state == original

    # Two executable mechanisms share the entire paid history. Their actual
    # responses respectively remove a possible point or activate the target;
    # neither conclusion was inserted into history during experiment design.
    def actual_response(number, setting):
        return -.1 <= setting - 2. * features[number]["x"] <= upper

    assert all(actual_response(item["crystal"], item["frequency"]) == item["activated"]
               for item in state["response_settings"])
    actual = actual_response(chosen["selected_number"], chosen["selected_setting"])
    assert actual is activated
    before = indices(response_family_grid_support(features, state["response_settings"],
                                                  ESTIMATOR, 3, grid))
    state["response_settings"].append(response(3, chosen["selected_setting"], actual))
    if actual:
        assert (chosen["selected_setting"] - grid["origin"]) / grid["step"] in before
    else:
        after = indices(response_family_grid_support(features, state["response_settings"],
                                                     ESTIMATOR, 3, grid))
        assert after == before - {3}
        assert after < before


def test_stalled_research_retains_operational_option_without_automatically_resuming(monkeypatch):
    from hegel_machine.cognition import world_response_constraints
    from hegel_machine.cognition.world_response_consequences import assess_response_consequences
    from hegel_machine.cognition.world_response_hypotheses import (
        _propose_response_members, propose_response_research,
    )

    references = [{"crystal": number,
        "measurements": {"x": {"value": float(number - 1), "evidence": number}},
        "frequency": {"value": float(2 * (number - 1)), "evidence": number}}
        for number in (1, 2)]
    state = {"research_table": {"references": references, "targets": [
        {"crystal": 3, "measurements": {"x": {"value": 3., "evidence": 3}}}]},
        "response_settings": reference_history(), "target_numbers": [3], "completed": [],
        "control_grid": GRID}
    parent = {"id": "quadratic", "basis": [X.to_dict(),
        Program("mul", children=(X, X)).to_dict()]}
    path = "shared/research.py"
    for _ in range(2):
        result = propose_response_research(state, [parent], path)
        assert result["selection"] is not None
        number, setting = result["selected_number"], result["selected_setting"]
        # These negative responses come from a single mechanism compatible
        # with all six original reference observations, not arbitrary vetoes.
        active = -.21 <= setting - 2.2 * FEATURES[number]["x"] <= .01
        assert not active
        state["response_research"] = deepcopy(result)
        state["response_settings"].append(response(number, setting, active))

    members = _propose_response_members(state, [parent], path)
    full = assess_response_consequences(state, [parent], deepcopy(members))
    assert full["status"] == "family_research_needed"
    before = deepcopy(state)
    revised = propose_response_research(state, [parent], path)
    assert state == before
    # The finite paid coefficient/band batch may be tested even when its
    # first setting was already proposed, but never observed, by the old pool.
    # Refutations must not buy unlimited refits or erase scientific stagnation.
    assert revised["status"] == "conditional_joint_response_trial"
    number, setting = revised["selected_number"], revised["selected_setting"]
    assert (number, setting) not in {
        (item["crystal"], item["frequency"]) for item in state["response_settings"]}
    active = -.21 <= setting - 2.2 * FEATURES[number]["x"] <= .01
    assert not active
    for key in ("families", "admitted_signatures", "possible_signatures", "nonlocal_progress"):
        assert revised["family_research"][key] == full["family_research"][key]
    state["response_research"] = deepcopy(revised)
    state["response_settings"].append(response(number, setting, active))
    support_ids = {support["id"] for support in
        revised["conditional_simplification"]["assumption"]["supports"]}
    original_fit = world_response_constraints.fit_response_constraints
    repeated_joint = []

    def track_joint(*args, **kwargs):
        if support_ids & {family["id"] for family in args[2]}:
            repeated_joint.append(True)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(world_response_constraints, "fit_response_constraints", track_joint)
    batch = deepcopy(revised["conditional_simplification"]["joint_revision"]["models"])
    stopped = propose_response_research(state, [parent], path)
    for _ in range(len(batch) - 1):
        if stopped["status"] != "conditional_joint_response_trial":
            break
        assert stopped["conditional_simplification"]["joint_revision"]["models"] == batch
        number, setting = stopped["selected_number"], stopped["selected_setting"]
        assert (number, setting) in {
            (model["design_target_number"], model["target_setting"]) for model in batch}
        assert (number, setting) not in {
            (item["crystal"], item["frequency"]) for item in state["response_settings"]}
        active = -.21 <= setting - 2.2 * FEATURES[number]["x"] <= .01
        assert not active
        state["response_research"] = deepcopy(stopped)
        state["response_settings"].append(response(number, setting, active))
        stopped = propose_response_research(state, [parent], path)
    members = _propose_response_members(state, [parent], path)
    full = assess_response_consequences(state, [parent], deepcopy(members))
    before = deepcopy(state)
    stopped = propose_response_research(state, [parent], path)
    assert state == before
    assert not repeated_joint
    assert stopped["status"] == "family_research_needed"
    assert stopped["selection"] is None
    assert stopped["selected_number"] is stopped["selected_setting"] is None
    child = stopped["conditional_simplification"]
    assert child["status"] == "family_research_needed"
    assert not child["adopted"]
    assert child["family_research"]["selected_test"] is None
    option = stopped["family_research"]["research_obligation"]["conditional_operational_search"]
    assert option["assumption"] == child["assumption"]
    assert option["remaining_settings"] > 0
    assert option["grid_intervals"]
    suggestion = option["suggested_trial"]
    assert suggestion == child["family_research"]["operational_search"]["suggested_trial"]
    assert suggestion["guaranteed_search_reduction"] == 1
    assert suggestion["guaranteed_nonlocal_eliminations"] == 0
    point = suggestion["selected_experiment"]
    assert (point["number"], point["prediction"]) not in {
        (item["crystal"], item["frequency"]) for item in state["response_settings"]}
    for key in ("families", "admitted_signatures", "possible_signatures", "nonlocal_progress"):
        assert stopped["family_research"][key] == full["family_research"][key]
    # With every retained witness refuted, persisting the stopped decision
    # must not discard its scope and make the next comparison refit it anew.
    state["response_research"] = deepcopy(stopped)
    repeated = propose_response_research(state, [parent], path)
    assert repeated["status"] == "family_research_needed"
    assert repeated["selection"] is None
    assert repeated["conditional_simplification"]["joint_revision"] == child["joint_revision"]
    assert repeated["conditional_simplification"]["family_research"]["no_progress_cycles"] >= \
        child["family_research"]["no_progress_cycles"]
    assert not repeated_joint


@pytest.mark.parametrize("extra_support", (False, True))
def test_projected_trial_reports_actual_supports_and_executes_without_template_member(
        tmp_path, monkeypatch, extra_support):
    from hegel_machine.cognition import world_reference_recalibration as controller
    from hegel_machine.cognition.world_response_consequences import (
        assess_response_consequences, select_response_grid_trial,
    )
    from hegel_machine.cognition.world_response_hypotheses import _propose_response_members
    from test_world_reference_hypothesis import ReferenceLab

    def measured(number, value, setting=None):
        row = {"crystal": number, "measurements": {"x": {"value": value, "evidence": number}}}
        if setting is not None:
            row["frequency"] = {"value": setting, "evidence": number}
        return row

    state = {"research_table": {
        "references": [measured(1, 0., 0.), measured(2, 1., 2.)],
        "targets": [measured(3, 1.5)]},
        "response_settings": reference_history(), "target_numbers": [3], "completed": [],
        "control_grid": {"origin": .25, "step": 1., "lower": .25, "upper": 10.25},
        "control_grids": {"1": GRID, "2": GRID},
        "consequence_panel": [{"crystal": 3, "frequency": 4.25}]}
    families = [ESTIMATOR]
    if extra_support:
        families = [*families, {**ESTIMATOR, "id": "another_compatible_support"}]
    path = "shared/research.py"
    members = _propose_response_members(state, families, path)
    assessed = assess_response_consequences(state, families, members)
    # The finite sample can miss an attainable point. Retain only its compatible
    # members that predict a negative there, while the full supports remain.
    assessed["hypotheses"] = [h for h in assessed["hypotheses"] if not
        h["lower"] <= 3.25 - h["predictions_by_crystal"]["3"] <= h["upper"]]
    assert assessed["hypotheses"]
    for experiment in assessed["experiments"]:
        experiment.update(family="unrelated_composite", candidate_id="old_coefficient",
                          fit_mse=123., restore_frequency=0.)
    before_templates = deepcopy(assessed["experiments"])
    chosen = select_response_grid_trial(state, families, assessed)
    point = chosen["selection"]["selected_experiment"]
    assert point["family"] == "attainable_target_search"
    assert point["conditional_support_ids"] == sorted(f["id"] for f in families)
    assert "candidate_id" not in point and "fit_mse" not in point
    assert "restore_frequency" not in point
    assert point["predictions"] and not any(point["predictions"].values())
    assert point["procedure"]["path"] == path
    assert assessed["experiments"] == before_templates
    assert point["prediction"] == 3.25

    # Exercise the existing parent handoff, installation-aware trial creation,
    # and public tuning controller. The neutral label needs no estimator lookup.
    chosen["question_id"] = "projected_response"
    state["response_research"] = chosen
    monkeypatch.setattr(controller, "STATE", tmp_path / "controller_state.json")
    memory = {"program_arguments": {"estimator_trial": True},
              "limits": {"max_model_calls": 0}, "costs": {"model_calls": 0}}
    selected = controller._extend_hypotheses(state, memory, 3, path)["response_trial"]
    assert selected["family"] == point["family"]
    lab = ReferenceLab()
    lab.required[3], lab.frequencies[3] = 3.25, 2.25
    trial = controller._trial_action(state, selected, 3, {"3": 203}, {"3": 103},
                                     lab.public_observation(), [], None)
    records = []
    for action in trial["actions"]:
        records.append({"index": len(records), **lab.act(action)})
    tuning = controller._tune(lab.public_observation(), point["prediction"])
    for action in tuning["actions"]:
        records.append({"index": len(records), **lab.act(action)})
    measured_responses = controller._response_settings(records, {"3": 203}, {"3": 103})
    assert state["test"]["family"] == "attainable_target_search"
    assert lab.frequencies[3] == point["prediction"] and lab.active(3)
    assert any(r["frequency"] == point["prediction"] and r["activated"]
               for r in measured_responses)
