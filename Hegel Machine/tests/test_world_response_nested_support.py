"""Simpler members must answer the same Boolean evidence as their parent."""

from copy import deepcopy

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_response_hypotheses import (
    _propose_fitted_responses,
    _propose_response_members,
    propose_response_research,
)
from hegel_machine.cognition.world_response_consequences import assess_response_consequences


X = Program("feature", name="measurement")
X2 = Program("mul", children=(X, X))
X3 = Program("mul", children=(X2, X))
PATH = "shared/research.py"


def state_for(successes, negatives=()):
    references = [{"crystal": n, "measurements": {
        "measurement": {"value": float(n - 1), "evidence": n}},
        "frequency": {"value": value, "evidence": n}}
        for n, value in enumerate(successes, 1)]
    target = len(references) + 1
    responses = [{"crystal": n, "frequency": value, "activated": True,
                  "evidence": [n]} for n, value in enumerate(successes, 1)]
    responses.extend({"crystal": n, "frequency": value, "activated": False,
                      "evidence": [len(successes) + index + 1]}
                     for index, (n, value) in enumerate(negatives))
    return {"research_table": {"references": references, "targets": [
        {"crystal": target, "measurements": {
            "measurement": {"value": float(target - 1), "evidence": 99}}}]},
        "response_settings": responses, "target_numbers": [target], "completed": [],
        "control_grid": {"lower": 0., "upper": 300., "step": 1., "origin": 0.}}


def basis_key(basis):
    return frozenset(Program.from_dict(term).simplified() for term in basis)


def support_for(result, *terms):
    return next(item for item in result["nested_supports"]
                if basis_key(item["basis"]) == frozenset(terms))


def test_usable_parent_does_not_hide_a_simple_interval_refit():
    # The four successful settings are offset [-3,-3,+3,+3] from 10+5*x.
    # Interpolating them gives a usable cubic. Every singleton linear point
    # fit, including leave-one-out fits, contradicts a paid negative instead.
    state = state_for((7., 12., 23., 28.),
        [(n, 10. + 5. * (n - 1) + offset) for n in range(1, 5) for offset in (-4., 4.)])
    parent = {"id": "full", "basis": [term.to_dict() for term in (X, X2, X3)]}
    before = deepcopy((state, parent))
    old_parent = _propose_fitted_responses(state, [parent], PATH)
    old_simple = _propose_fitted_responses(state, [{"id": "linear", "basis": [X.to_dict()]}], PATH)
    assert old_parent["selection"] is not None
    assert old_simple["status"] == "no_compatible_response_hypotheses"

    result = propose_response_research(state, [parent], PATH)
    assert (state, parent) == before
    linear = support_for(result, X)
    simple = [h for h in result["hypotheses"] if h["family"] == linear["id"]]
    assert simple
    assert any(c["family"] == "full" and c["status"] == "compatible"
               for c in result["candidates"])
    assert any(row["family"] == linear["id"] and row["models_retained"] > 0
               for row in result["joint_parameter_search"])
    for hypothesis in simple:
        assert hypothesis["derived_from"] == ["full"]
        for response in state["response_settings"]:
            residual = response["frequency"] - hypothesis["predictions_by_crystal"][str(response["crystal"])]
            assert (hypothesis["lower"] <= residual <= hypothesis["upper"]) == response["activated"]
    # Derived proposals are members of the existing family, not new scope or
    # extra independently confirmed families.
    research = result["family_research"]
    assert [basis_key(basis) for basis in research["bases"]] == [basis_key(parent["basis"])]
    assert [(family["id"], family["status"]) for family in research["families"]] == [
        ("full", "SAT_WITNESS")]


def test_duplicate_and_reordered_declarations_do_not_multiply_nested_search():
    state = state_for((7., 12., 23., 28.),
        [(n, 10. + 5. * (n - 1) + offset) for n in range(1, 5) for offset in (-4., 4.)])
    basis = [term.to_dict() for term in (X, X2, X3)]
    original = [{"id": "full", "basis": basis}]
    repeated = original + [{"id": "same", "basis": list(reversed(basis))},
                           {"id": "repeated_term", "basis": basis + basis[:1]}]
    first = propose_response_research(state, original, PATH)
    second = propose_response_research(state, repeated, PATH)
    expected = {frozenset((X,)), frozenset((X2,)), frozenset((X3,)),
                frozenset((X, X2)), frozenset((X, X3)), frozenset((X2, X3))}
    for result in (first, second):
        supports = result["nested_supports"]
        assert len(supports) == len(expected)
        assert {basis_key(item["basis"]) for item in supports} == expected
        linear = support_for(result, X)
        # One real interval search for this support and design target, rather
        # than repeated numerical votes caused by duplicate parent syntax.
        assert sum(row["family"] == linear["id"]
                   for row in result["joint_parameter_search"]) == 1
        assert len(result["family_research"]["bases"]) == 1
    assert support_for(second, X)["derived_from"] == ["full", "same", "repeated_term"]


def test_negative_feedback_rejects_simple_members_without_refuting_parent():
    state = state_for((10., 15., 30.))
    parent = {"id": "quadratic", "basis": [X.to_dict(), X2.to_dict()]}
    first = propose_response_research(state, [parent], PATH)
    linear = support_for(first, X)
    assert any(h["family"] == linear["id"] for h in first["hypotheses"])
    for n, center in enumerate((10., 15., 30.), 1):
        state["response_settings"].extend({"crystal": n, "frequency": center + offset,
            "activated": False, "evidence": [100 + 2 * n + index]}
            for index, offset in enumerate((-1., 1.)))
    result = propose_response_research(state, [parent], PATH)
    linear = support_for(result, X)
    assert not any(h["family"] == linear["id"] for h in result["hypotheses"])
    assert any(row["family"] == linear["id"] and row["status"] == "infeasible"
               for row in result["joint_parameter_search"])
    assert any(c["family"] == "quadratic" and c["status"] == "compatible"
               for c in result["candidates"])
    assert [(family["id"], family["status"]) for family in result["family_research"]["families"]] == [
        ("quadratic", "SAT_WITNESS")]


def test_conditional_simple_research_changes_parent_action_and_consumes_feedback():
    state = state_for((10., 15., 30.),
        [(n, value + offset) for n, value in enumerate((10., 15., 30.), 1)
         for offset in (-1., 1.)])
    estimators = [{"id": "quadratic", "basis": [X.to_dict(), X2.to_dict()]}]
    # Conditional research is used before its own repeated failures establish
    # stagnation. Persist each executed decision and its actual feedback.
    previous_child = None
    for index in range(2):
        members = _propose_response_members(state, estimators, PATH)
        full_only = assess_response_consequences(state, estimators, deepcopy(members))
        result = propose_response_research(state, estimators, PATH)
        child = result["conditional_simplification"]
        assert child["adopted"]
        assert result["selection"]["consequence_scope"] == "assumed_simpler_supports"
        assert [basis_key(item["basis"]) for item in child["assumption"]["supports"]] == [frozenset((X2,))]
        for key in ("bases", "families", "admitted_signatures", "possible_signatures"):
            assert result["family_research"][key] == full_only["family_research"][key]
        selected = result["selection"]["selected_experiment"]
        assert (selected["number"], selected["prediction"]) not in {
            (row["crystal"], row["frequency"]) for row in state["response_settings"]}
        assert child["family_research"]["selected_test"] == {
            "crystal": selected["number"], "frequency": selected["prediction"]}
        if previous_child is not None:
            assert child["family_research"]["response_count"] == previous_child["response_count"] + 1
            assert 100 in child["family_research"]["evidence"]
        previous_child = deepcopy(child["family_research"])
        state["response_research"] = deepcopy(result)
        state["response_settings"].append({"crystal": result["selected_number"],
            "frequency": result["selected_setting"], "activated": False, "evidence": [100 + index]})

    # Scientific stagnation permits the retained finite joint batch, not an
    # unlimited restart. The shared family and conditional scope remain live.
    members = _propose_response_members(state, estimators, PATH)
    full_only = assess_response_consequences(state, estimators, deepcopy(members))
    result = propose_response_research(state, estimators, PATH)
    child = result["conditional_simplification"]
    assert full_only["selection"] is None
    assert result["status"] == child["status"] == "conditional_joint_response_trial"
    assert child["family_research"]["no_progress_cycles"] >= 2
    for key in ("bases", "families", "admitted_signatures", "possible_signatures"):
        assert result["family_research"][key] == full_only["family_research"][key]
    batch = deepcopy(child["joint_revision"]["models"])
    batch_responses = 0
    for index in range(len(batch)):
        if result["status"] != "conditional_joint_response_trial":
            break
        selected = result["selection"]["selected_experiment"]
        assert (selected["number"], selected["prediction"]) in {
            (model["design_target_number"], model["target_setting"]) for model in batch}
        assert (selected["number"], selected["prediction"]) not in {
            (row["crystal"], row["frequency"]) for row in state["response_settings"]}
        state["response_research"] = deepcopy(result)
        state["response_settings"].append({"crystal": selected["number"],
            "frequency": selected["prediction"], "activated": False, "evidence": [102 + index]})
        batch_responses += 1
        result = propose_response_research(state, estimators, PATH)
        assert result["conditional_simplification"]["joint_revision"]["models"] == batch
    members = _propose_response_members(state, estimators, PATH)
    full_only = assess_response_consequences(state, estimators, deepcopy(members))
    child = result["conditional_simplification"]
    assert result["status"] == child["status"] == "family_research_needed"
    assert not child["adopted"]
    assert result["selection"] is None
    assert result["selected_number"] is result["selected_setting"] is None
    for key in ("bases", "families", "admitted_signatures", "possible_signatures"):
        assert result["family_research"][key] == full_only["family_research"][key]
    research = child["family_research"]
    assert research["response_count"] == previous_child["response_count"] + 1 + batch_responses
    assert 101 in research["evidence"]
    assert research["selected_test"] is None
    option = result["family_research"]["research_obligation"]["conditional_operational_search"]
    assert option["assumption"] == child["assumption"]
    assert option["remaining_settings"] == 1
    assert option["grid_intervals"] == [[57, 57]]
    assert option["suggested_trial"] == research["operational_search"]["suggested_trial"]
    assert option["suggested_trial"]["guaranteed_nonlocal_eliminations"] == 0


def test_conditional_assumption_can_fail_while_full_explanation_survives():
    state = state_for((10., 15., 30.),
        [(n, value + offset) for n, value in enumerate((10., 15., 30.), 1)
         for offset in (-1., 1.)])
    estimators = [{"id": "quadratic", "basis": [X.to_dict(), X2.to_dict()]}]
    first = propose_response_research(state, estimators, PATH)
    assert [basis_key(item["basis"]) for item in
        first["conditional_simplification"]["assumption"]["supports"]] == [frozenset((X2,))]
    state["response_research"] = deepcopy(first)
    # Success at 58 is compatible with a*x+b*x² and the paid narrow reference
    # bands, but cannot be explained by any single x² term with a shared band.
    state["response_settings"].append({"crystal": 4, "frequency": 58.,
        "activated": True, "evidence": [100]})
    state["completed"] = [4]
    state["target_numbers"].append(5)
    state["research_table"]["targets"].append({"crystal": 5,
        "measurements": {"measurement": {"value": 4., "evidence": 101}}})
    result = propose_response_research(state, estimators, PATH)
    simple = support_for(result, X2)
    assert not any(h["family"] == simple["id"] for h in result["hypotheses"])
    assert any(row["family"] == simple["id"] and row["status"] == "infeasible"
               for row in result["joint_parameter_search"])
    assert [(family["id"], family["status"]) for family in result["family_research"]["families"]] == [
        ("quadratic", "SAT_WITNESS")]
    assert not result.get("conditional_simplification", {}).get("adopted", False)
