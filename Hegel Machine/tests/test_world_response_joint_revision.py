"""A stalled support can execute its finite paid revision without refitting it."""

from copy import deepcopy

from hegel_machine.cognition import world_response_constraints
from hegel_machine.cognition.world_response_consequences import assess_response_consequences
from hegel_machine.cognition.world_response_hypotheses import (
    _propose_response_members,
    propose_response_research,
)
from test_world_response_nested_support import PATH, X, X2, X3, state_for


def stalled_state(*terms):
    state = state_for((10., 15., 30.), [
        (number, value + offset)
        for number, value in enumerate((10., 15., 30.), 1)
        for offset in (-1., 1.)])
    estimators = [{"id": "full", "basis": [term.to_dict() for term in terms]}]
    for index in range(2):
        result = propose_response_research(state, estimators, PATH)
        assert result["selected_number"] == 4
        state["response_research"] = deepcopy(result)
        state["response_settings"].append({
            "crystal": result["selected_number"], "frequency": result["selected_setting"],
            "activated": False, "evidence": [100 + index]})
    return state, estimators


def conditional_supports(state):
    return state["response_research"]["conditional_simplification"]["assumption"]["supports"]


def test_stalled_compatible_support_gets_new_joint_member_without_scientific_reset():
    state, estimators = stalled_state(X, X2, X3)
    before = deepcopy(state)
    members = _propose_response_members(state, estimators, PATH)
    full = assess_response_consequences(deepcopy(state), estimators, deepcopy(members))
    support_ids = {support["id"] for support in conditional_supports(state)}
    previous = [h for h in members["hypotheses"] if h["family"] in support_ids]
    assert previous  # Compatibility alone used to block this joint research.
    assert not support_ids & {row["family"] for row in members["joint_parameter_search"]}

    result = propose_response_research(state, estimators, PATH)
    assert state == before
    assert result["status"] == "conditional_joint_response_trial"
    selected = result["selection"]["selected_experiment"]
    assert selected["number"] == 4
    assert selected["prediction"] in (54., 56.)
    assert all(not h["lower"] <= selected["prediction"] -
               h["predictions_by_crystal"]["4"] <= h["upper"] for h in previous)
    assert (4, selected["prediction"]) not in {
        (row["crystal"], row["frequency"]) for row in state["response_settings"]}
    hypothesis = next(h for h in result["hypotheses"]
                      if h["candidate_id"] == selected["candidate_id"])
    assert hypothesis["family"] in support_ids
    for row in state["response_settings"]:
        residual = row["frequency"] - hypothesis["predictions_by_crystal"][str(row["crystal"])]
        assert (hypothesis["lower"] <= residual <= hypothesis["upper"]) == row["activated"]
    assert hypothesis["lower"] <= selected["prediction"] - \
        hypothesis["predictions_by_crystal"]["4"] <= hypothesis["upper"]
    for key in ("bases", "families", "admitted_signatures", "possible_signatures",
                "nonlocal_progress", "no_progress_cycles"):
        assert result["family_research"][key] == full["family_research"][key]
    assert result["selection"].get("guaranteed_nonlocal_eliminations", 0) == 0
    assert result["conditional_simplification"]["joint_revision"]


def test_failed_joint_trial_uses_surviving_paid_batch_without_refitting(monkeypatch):
    state, estimators = stalled_state(X, X2, X3)
    result = propose_response_research(state, estimators, PATH)
    assert result["status"] == "conditional_joint_response_trial"
    batch = deepcopy(result["conditional_simplification"]["joint_revision"]["models"])
    assert len({model["target_setting"] for model in batch}) == 2
    support_ids = {support["id"] for support in conditional_supports(state)}
    state["response_research"] = deepcopy(result)
    state["response_settings"].append({"crystal": result["selected_number"],
        "frequency": result["selected_setting"], "activated": False, "evidence": [102]})
    calls = []
    original = world_response_constraints.fit_response_constraints

    def record(*args, **kwargs):
        calls.append({item["id"] for item in args[2]})
        return original(*args, **kwargs)

    monkeypatch.setattr(world_response_constraints, "fit_response_constraints", record)
    repeated = propose_response_research(state, estimators, PATH)
    assert not any(support_ids & families for families in calls)
    assert repeated["status"] == "conditional_joint_response_trial"
    selected = repeated["selection"]["selected_experiment"]
    assert selected["prediction"] != result["selected_setting"]
    assert selected["prediction"] in {model["target_setting"] for model in batch}
    hypothesis = next(h for h in repeated["hypotheses"]
                      if h["candidate_id"] == selected["candidate_id"])
    for row in state["response_settings"]:
        residual = row["frequency"] - hypothesis["predictions_by_crystal"][str(row["crystal"])]
        assert (hypothesis["lower"] <= residual <= hypothesis["upper"]) == row["activated"]
    assert repeated["conditional_simplification"]["joint_revision"]["models"] == batch
    assert repeated["conditional_simplification"]["family_research"]["no_progress_cycles"] >= 2

    # Both paid witnesses have now been contradicted. The batch is exhausted,
    # not permission to refit new members until a point happens to work.
    state["response_research"] = deepcopy(repeated)
    state["response_settings"].append({"crystal": selected["number"],
        "frequency": selected["prediction"], "activated": False, "evidence": [103]})
    exhausted = propose_response_research(state, estimators, PATH)
    assert not any(support_ids & families for families in calls)
    assert exhausted["status"] == "family_research_needed"
    assert exhausted["selection"] is None
    assert exhausted["conditional_simplification"]["joint_revision"]["models"] == batch
    assert exhausted["conditional_simplification"]["family_research"]["no_progress_cycles"] >= \
        repeated["conditional_simplification"]["family_research"]["no_progress_cycles"]
    state["response_research"] = deepcopy(exhausted)
    still_exhausted = propose_response_research(state, estimators, PATH)
    assert still_exhausted["status"] == "family_research_needed"
    assert still_exhausted["selection"] is None
    assert not any(support_ids & families for families in calls)


def test_unexecuted_paid_joint_trial_survives_same_evidence_recomparison(monkeypatch):
    state, estimators = stalled_state(X, X2, X3)
    result = propose_response_research(state, estimators, PATH)
    assert result["status"] == "conditional_joint_response_trial"
    support_ids = {support["id"] for support in conditional_supports(state)}
    state["response_research"] = deepcopy(result)
    original = world_response_constraints.fit_response_constraints
    attempts = []

    def record(*args, **kwargs):
        if support_ids & {item["id"] for item in args[2]}:
            attempts.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(world_response_constraints, "fit_response_constraints", record)
    repeated = propose_response_research(state, estimators, PATH)
    assert not attempts
    assert repeated["status"] == "conditional_joint_response_trial"
    selected = repeated["selection"]["selected_experiment"]
    assert (selected["number"], selected["prediction"]) == (
        result["selected_number"], result["selected_setting"])
    assert any(h["candidate_id"] == selected["candidate_id"] for h in repeated["hypotheses"])
    assert repeated["conditional_simplification"]["family_research"]["no_progress_cycles"] == \
        result["conditional_simplification"]["family_research"]["no_progress_cycles"]


def test_empty_joint_search_is_recorded_once_and_does_not_restart_research(monkeypatch):
    state, estimators = stalled_state(X, X2, X3)
    support_ids = {support["id"] for support in conditional_supports(state)}
    original = world_response_constraints.fit_response_constraints
    attempts = []

    def no_new_model(*args, **kwargs):
        ids = {item["id"] for item in args[2]}
        if ids == support_ids:
            attempts.append(ids)
            return {"models": [], "search": [{"family": item["id"],
                    "status": "infeasible", "models_retained": 0} for item in args[2]],
                    "compute_cost": 7}
        return original(*args, **kwargs)

    monkeypatch.setattr(world_response_constraints, "fit_response_constraints", no_new_model)
    # Both proposal sources are empty in this test. An empty LP result alone
    # no longer excludes the existing fixed-program interval witnesses.
    from hegel_machine.cognition import world_response_hypotheses as response
    monkeypatch.setattr(response, "_append_fixed_program_responses", lambda result, *args: result)
    result = propose_response_research(state, estimators, PATH)
    assert len(attempts) == 1
    assert result["status"] == "family_research_needed"
    assert result["selection"] is None
    assert result["conditional_simplification"]["joint_revision"]
    state["response_research"] = deepcopy(result)
    repeated = propose_response_research(state, estimators, PATH)
    assert len(attempts) == 1
    assert repeated["status"] == "family_research_needed"
    assert repeated["selection"] is None


def test_existing_unexecuted_joint_members_are_available_without_duplicate_fit(monkeypatch):
    # Here the global pool has already exhausted its target settings, so the
    # ordinary joint fallback has already offered alternatives. Pool presence
    # is not an observation: the first paid batch is still available to test.
    state, estimators = stalled_state(X, X2)
    support_ids = {support["id"] for support in conditional_supports(state)}
    original = world_response_constraints.fit_response_constraints
    attempts = []

    def record(*args, **kwargs):
        if support_ids & {item["id"] for item in args[2]}:
            attempts.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(world_response_constraints, "fit_response_constraints", record)
    result = propose_response_research(state, estimators, PATH)
    assert len(attempts) == 1
    assert result["status"] == "conditional_joint_response_trial"
    selected = result["selection"]["selected_experiment"]
    batch = result["conditional_simplification"]["joint_revision"]["models"]
    assert (selected["number"], selected["prediction"]) in {
        (model["design_target_number"], model["target_setting"]) for model in batch}
    assert (selected["number"], selected["prediction"]) not in {
        (row["crystal"], row["frequency"]) for row in state["response_settings"]}
    assert result["conditional_simplification"]["family_research"]["no_progress_cycles"] >= 2
