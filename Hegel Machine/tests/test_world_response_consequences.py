"""Family consequences must survive duplicate members and changing evidence."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition import world_response_consequences as consequences
from hegel_machine.cognition.world_response_constraints import query_response_family


X = Program("feature", name="x")
LINEAR = {"id": "linear", "basis": [X.to_dict()]}
QUADRATIC = {"id": "quadratic", "basis": [X.to_dict(), Program("mul", children=(X, X)).to_dict()]}


def response(crystal, frequency, activated):
    return {"crystal": crystal, "frequency": frequency, "activated": activated}


def experiment(name, crystal, frequency):
    return {"id": name, "number": crystal, "prediction": frequency,
            "predictions": {"member_a": True, "member_b": False}}


def fixture(panel_crystal=3, panel_frequency=14.):
    def measured(crystal, value):
        return {"crystal": crystal, "measurements": {"x": {"value": value, "evidence": crystal}}}
    return {"research_table": {"references": [measured(1, 0.), measured(2, 2.)],
                               "targets": [measured(3, 1.), measured(4, 0.)]},
            "response_settings": [response(1, 10., True), response(1, 9., False), response(2, 12., True)],
            "target_numbers": [3, 4], "completed": [],
            "consequence_panel": [{"crystal": panel_crystal, "frequency": panel_frequency}]}


def selected_result(experiments):
    first = experiments[0] if experiments else None
    return {"compute_cost": 0, "evidence": [1, 2], "experiments": deepcopy(experiments),
            "selection": {"selected_experiment": deepcopy(first)} if first else None,
            "selected_number": first["number"] if first else None,
            "selected_setting": first["prediction"] if first else None}


def assess(state, estimators=(LINEAR,), experiments=()):
    return consequences.assess_response_consequences(state, estimators, selected_result(experiments))


def remember(state, result):
    state["response_research"] = {"family_research": deepcopy(result["family_research"])}


def test_reference_beats_target_by_nonlocal_gain_and_duplicate_basis_ids_do_not_change_it():
    state = fixture(panel_crystal=4, panel_frequency=11.)
    actions = [experiment("target", 3, 14.), experiment("reference", 1, 11.)]
    results = [assess(state, [{**deepcopy(LINEAR), "id": f"copy_{i}"} for i in range(count)], actions)
               for count in (1, 10, 100)]
    for count, result in zip((1, 10, 100), results):
        research = result["family_research"]
        assert result["selection"]["selected_experiment"]["id"] == "reference"
        assert result["selected_number"] == 1
        assert research["selected_role"] == "nonlocal_family_test"
        assert len(research["families"]) == 1
        assert len(research["families"][0]["source_ids"]) == count
        value = next(value for value in research["experiment_values"] if value["experiment_id"] == "reference")
        assert value["guaranteed_nonlocal_eliminations"] == 1
        assert value["by_hypothetical_outcome"] == [1, 1]
    assert len({result["compute_cost"] for result in results}) == 1
    assert results[0]["family_research"]["experiment_values"] == results[-1]["family_research"]["experiment_values"]


def test_learning_only_the_tested_point_does_not_count_as_nonlocal_gain():
    state = fixture()
    first = assess(state, experiments=[experiment("target", 3, 14.)])
    assert first["family_research"]["admitted_signatures"] == [[False], [True]]
    assert first["family_research"]["experiment_values"] == []
    remember(state, first)
    state["response_settings"].append(response(3, 14., False))
    after = assess(state)
    assert after["family_research"]["panel"] == []
    assert after["family_research"]["nonlocal_progress"] == 0
    assert after["family_research"]["excluded_prior_signatures"] == []


def test_another_family_can_keep_a_signature_that_the_linear_family_excludes():
    state = fixture()
    first = assess(state, estimators=[LINEAR, QUADRATIC])
    remember(state, first)
    state["response_settings"].append(response(1, 11., False))
    features = {1: {"x": 0.}, 2: {"x": 2.}, 3: {"x": 1.}, 4: {"x": 0.}}
    query = [response(3, 14., True)]
    assert query_response_family(features, state["response_settings"], LINEAR, query)["status"] == "UNSAT_VERIFIED"
    assert query_response_family(features, state["response_settings"], QUADRATIC, query)["status"] == "SAT_WITNESS"
    result = assess(state, estimators=[LINEAR, QUADRATIC])
    assert result["family_research"]["admitted_signatures"] == [[False], [True]]
    assert result["family_research"]["nonlocal_progress"] == 0
    assert result["family_research"]["excluded_prior_signatures"] == []


@pytest.mark.parametrize("change", ("panel", "representation"))
def test_new_panel_or_representation_cannot_retroactively_claim_old_signature_progress(change):
    state = fixture()
    first = assess(state)
    assert first["family_research"]["admitted_signatures"] == [[False], [True]]
    remember(state, first)
    state["response_settings"].append(response(1, 11., False))
    if change == "panel":
        state["consequence_panel"][0]["frequency"] = 16.
    else:
        state["research_table"]["targets"][0]["measurements"]["x"]["value"] = .5
    result = assess(state)
    research = result["family_research"]
    assert research["epoch"] == first["family_research"]["epoch"] + 1
    assert research["admitted_signatures"] == [[False]]
    assert research["nonlocal_progress"] == 0
    assert research["excluded_prior_signatures"] == []


def test_unknown_queries_never_become_elimination_or_progress(monkeypatch):
    state = fixture()
    first = assess(state)
    remember(state, first)
    state["response_settings"].append(response(1, 11., False))
    monkeypatch.setattr(consequences, "query_response_family", lambda *args, **kwargs:
                        {"status": "UNKNOWN", "compute_cost": 0, "lp_calls": 0})
    result = assess(state, experiments=[experiment("reference", 1, 13.)])
    research = result["family_research"]
    assert research["possible_signatures"] == [[False], [True]]
    assert research["admitted_signatures"] == []
    assert research["nonlocal_progress"] == 0
    assert research["experiment_values"] == []
    assert research["selected_role"] == "operational_exploration"


def test_fresh_unknown_paid_check_retries_only_with_a_larger_allowance(monkeypatch):
    state = fixture()
    features = {1: {"x": 0.}, 2: {"x": 2.}, 3: {"x": 1.}, 4: {"x": 0.}}
    checked = query_response_family(features, state["response_settings"], LINEAR,
                                    branch_limit=None, compute_limit=10)
    assert checked["status"] == "UNKNOWN"
    assert checked["reason"] == "query_compute_limit"
    original = deepcopy(checked)
    calls = []

    def recorded_query(*args, **kwargs):
        calls.append(deepcopy(kwargs.get("queries", [])))
        return query_response_family(*args, **kwargs)

    monkeypatch.setattr(consequences, "query_response_family", recorded_query)
    monkeypatch.setattr(consequences, "QUERY_ALLOWANCE", 10)
    result = consequences.assess_response_consequences(
        state, [LINEAR], selected_result([]), paid_checks={"linear": checked})
    assert result["family_research"]["families"][0]["status"] == "UNKNOWN"
    # Different hypothetical conditions remain researchable; the unchanged
    # paid-history query alone must not consume its previous allowance again.
    assert calls and all(conditions for conditions in calls)
    assert result["family_research"]["nonlocal_progress"] == 0

    calls.clear()
    monkeypatch.setattr(consequences, "QUERY_ALLOWANCE", 20)
    consequences.assess_response_consequences(
        state, [LINEAR], selected_result([]), paid_checks={"linear": checked})
    assert calls[0] == []
    assert checked == original


def test_paid_executable_witness_proves_panel_signature_without_repeating_lp(monkeypatch):
    state = fixture()
    state["consequence_panel"].append({"crystal": 4, "frequency": 11.})
    # The old exact-query cache spent this allowance without proving a single
    # panel signature, despite both paid-history queries returning witnesses.
    monkeypatch.setattr(consequences, "QUERY_ALLOWANCE", 70)
    research = assess(state, [LINEAR, QUADRATIC])["family_research"]
    assert {family["status"] for family in research["families"]} == {"SAT_WITNESS"}
    assert [False, False] in research["admitted_signatures"]
    assert research["witness_reuses"] >= 1
    assert 2 <= research["witness_check_cost"] <= research["compute_cost"] <= 70
    assert research["nonlocal_progress"] == 0


def test_later_family_witness_precedes_expensive_unresolved_family(monkeypatch):
    calls = []

    def unresolved_first(features, responses, family, queries=(), **kwargs):
        calls.append((family["id"], list(queries)))
        if not queries and family["id"] == "quadratic":
            return query_response_family(features, responses, family, queries, **kwargs)
        return {"status": "UNKNOWN", "compute_cost": kwargs["compute_limit"] if queries else 1,
                "lp_calls": 0}

    monkeypatch.setattr(consequences, "QUERY_ALLOWANCE", 200)
    monkeypatch.setattr(consequences, "query_response_family", unresolved_first)
    research = assess(fixture(), [LINEAR, QUADRATIC])["family_research"]
    assert research["admitted_signatures"] == [[False]]
    assert research["possible_signatures"] == [[False], [True]]
    assert research["signatures"][1]["status"] == "UNKNOWN"
    assert not any(queries and not queries[0]["activated"] for _, queries in calls)
    assert research["compute_cost"] <= 200


@pytest.mark.parametrize("change", ("measurement", "paid_response"))
def test_witnesses_do_not_survive_new_evidence_or_binding_invocation(monkeypatch, change):
    state = fixture()
    first = assess(state)
    assert first["family_research"]["witness_reuses"] > 0
    remember(state, first)
    if change == "measurement":
        state["research_table"]["targets"][0]["measurements"]["x"]["value"] = .5
    else:
        state["response_settings"].append(response(1, 11., False))
    monkeypatch.setattr(consequences, "query_response_family", lambda *args, **kwargs:
                        {"status": "UNKNOWN", "compute_cost": 0, "lp_calls": 0})
    research = assess(state)["family_research"]
    assert research["admitted_signatures"] == []
    assert research["witness_check_cost"] == research["witness_reuses"] == 0


def test_later_real_witness_upgrades_cached_unknown_without_retrying_lp(monkeypatch):
    attempts = []

    def unresolved_conditions(features, responses, family, queries=(), **kwargs):
        queries = list(queries)
        unavailable = (len(queries) == 1 and queries[0]["crystal"] == 3
                       and queries[0]["activated"])
        unavailable |= bool(queries and queries[0]["crystal"] == 1 and queries[0]["activated"])
        if unavailable:
            attempts.append(queries)
            return {"status": "UNKNOWN", "compute_cost": 1, "lp_calls": 0}
        return query_response_family(features, responses, family, queries, **kwargs)

    monkeypatch.setattr(consequences, "query_response_family", unresolved_conditions)
    actions = [experiment("first_reference", 1, 10.5), experiment("other_reference", 2, 12.5),
               experiment("revisit_reference", 1, 10.5)]
    research = assess(fixture(), experiments=actions)["family_research"]
    values = {item["experiment_id"]: item for item in research["experiment_values"]}
    assert values["first_reference"]["outcome_statuses"][1] == "UNKNOWN"
    assert values["revisit_reference"]["outcome_statuses"][1] == "SAT_WITNESS"
    assert len([query for query in attempts if query[0]["crystal"] == 1]) == 2
    assert research["compute_cost"] <= consequences.QUERY_ALLOWANCE


def test_only_two_completed_negative_tests_without_progress_trigger_research_obligation():
    state = fixture(panel_crystal=4, panel_frequency=11.)
    first = assess(state, experiments=[experiment("already_implied_negative_7", 1, 7.)])
    assert first["family_research"]["no_progress_cycles"] == 0
    remember(state, first)
    repeated = assess(state, experiments=[experiment("already_implied_negative_7", 1, 7.)])
    assert repeated["family_research"]["no_progress_cycles"] == 0
    state["response_settings"].append(response(1, 7., False))
    second = assess(state, experiments=[experiment("already_implied_negative_8", 1, 8.)])
    assert second["family_research"]["no_progress_cycles"] == 1
    assert "research_obligation" not in second["family_research"]
    remember(state, second)
    repeated = assess(state, experiments=[experiment("already_implied_negative_8", 1, 8.)])
    assert repeated["family_research"]["no_progress_cycles"] == 1
    remember(state, repeated)
    state["response_settings"].append(response(1, 8., False))
    third = assess(state, experiments=[experiment("already_implied_negative_6", 1, 6.)])
    assert third["family_research"]["nonlocal_progress"] == 0
    assert third["family_research"]["no_progress_cycles"] == 2
    assert third["status"] == "family_research_needed"
    assert third["selection"] is None
    assert third["family_research"]["research_obligation"]["reason"] \
        == "completed_tests_changed_members_without_task_or_nonlocal_progress"


@pytest.mark.parametrize("change", ("new_family", "measurement"))
def test_revised_research_scope_does_not_inherit_old_stagnation_or_old_test_completion(change):
    state = fixture(panel_crystal=4, panel_frequency=11.)
    first = assess(state, experiments=[experiment("old_test", 1, 7.)])
    remember(state, first)
    state["response_research"]["family_research"]["no_progress_cycles"] = 2
    state["response_settings"].append(response(1, 7., False))
    estimators = [LINEAR]
    if change == "new_family":
        estimators.append(QUADRATIC)
    else:
        state["research_table"]["targets"][0]["measurements"]["x"]["value"] = .5
    paid = deepcopy(state["response_settings"])
    result = assess(state, estimators, [experiment("new_test", 1, 8.)])
    research = result["family_research"]
    assert research["no_progress_cycles"] == 0
    assert research["nonlocal_progress"] == 0
    assert research["excluded_prior_signatures"] == []
    assert research["panel"] == state["consequence_panel"]
    assert "research_obligation" not in research
    assert result["selection"]["selected_experiment"]["id"] == "new_test"
    assert state["response_settings"] == paid


@pytest.mark.parametrize("change", ("unchanged", "reordered", "duplicate", "renamed", "panel"))
def test_same_research_scope_preserves_stagnation_across_presentation_changes(change):
    state = fixture(panel_crystal=4, panel_frequency=11.)
    estimators = [LINEAR, QUADRATIC]
    first = assess(state, estimators)
    remember(state, first)
    state["response_research"]["family_research"]["no_progress_cycles"] = 2
    if change == "reordered":
        estimators = list(reversed(estimators))
    elif change == "duplicate":
        estimators += [{**deepcopy(LINEAR), "id": "linear_copy"}]
    elif change == "renamed":
        estimators = [{**deepcopy(item), "id": item["id"] + "_renamed"} for item in estimators]
    elif change == "panel":
        state["consequence_panel"][0]["frequency"] = 16.
    result = assess(state, estimators)
    assert result["family_research"]["no_progress_cycles"] == 2
    assert result["family_research"]["nonlocal_progress"] == 0
    assert result["status"] == "family_research_needed"


@pytest.mark.parametrize("change", ("new_family", "measurement", "reordered", "renamed", "duplicate"))
def test_automatic_panel_is_reselected_only_for_changed_research_scope(change):
    state = fixture()
    del state["consequence_panel"]
    estimators = [LINEAR] if change == "new_family" else [LINEAR, QUADRATIC]
    first = assess(state, estimators, [experiment("old_region", 3, 14.)])
    remember(state, first)
    if change == "new_family":
        estimators.append(QUADRATIC)
    elif change == "measurement":
        state["research_table"]["targets"][0]["measurements"]["x"]["value"] = .5
    elif change == "reordered":
        estimators = list(reversed(estimators))
    elif change == "renamed":
        estimators = [{**deepcopy(item), "id": item["id"] + "_renamed"} for item in estimators]
    else:
        estimators += [{**deepcopy(LINEAR), "id": "linear_copy"}]
    result = assess(state, estimators, [experiment("current_region", 3, 16.)])
    expected = 16. if change in {"new_family", "measurement"} else 14.
    assert result["family_research"]["panel"] == [{"crystal": 3, "frequency": expected}]
    assert result["family_research"]["nonlocal_progress"] == 0


@pytest.mark.parametrize("external_panel_change", (False, True))
def test_consumed_target_panel_point_preserves_only_real_progress_on_remaining_conditions(external_panel_change):
    state = fixture()
    state["consequence_panel"].append({"crystal": 4, "frequency": 11.})
    first = assess(state, experiments=[experiment("target_4", 4, 11.)])
    assert {tuple(signature) for signature in first["family_research"]["admitted_signatures"]} \
        == {(False, False), (False, True), (True, True)}
    assert first["family_research"]["selected_test"] == {"crystal": 4, "frequency": 11.}
    remember(state, first)
    state["response_settings"].append(response(4, 11., False))
    if external_panel_change:
        state["consequence_panel"][0]["frequency"] = 16.
    result = assess(state)
    research = result["family_research"]
    assert research["admitted_signatures"] == [[False]]
    if external_panel_change:
        assert research["nonlocal_progress"] == 0
        assert research["excluded_prior_signatures"] == []
        assert research["no_progress_cycles"] == 1
    else:
        assert research["progress_panel"] == [{"crystal": 3, "frequency": 14.}]
        assert research["nonlocal_progress"] == 1
        assert research["excluded_prior_signatures"] == [[True]]
        assert research["no_progress_cycles"] == 0


def test_small_query_budget_stays_bounded_with_many_negatives_and_families(monkeypatch):
    state = fixture()
    state["response_settings"] += [response(1, float(value), False)
                                   for value in range(21) if value not in (9, 10)]
    allowance = 20
    monkeypatch.setattr(consequences, "QUERY_ALLOWANCE", allowance)
    result = assess(state, estimators=[LINEAR, QUADRATIC], experiments=[experiment("target", 3, 14.)])
    research = result["family_research"]
    assert research["compute_cost"] <= allowance
    assert research["compute_cost"] + research["ordering_compute_cost"] == result["compute_cost"]
    assert len(research["families"]) == 2
    assert {family["status"] for family in research["families"]} == {"UNKNOWN"}
    assert research["admitted_signatures"] == []
    assert research["possible_signatures"] == [[False], [True]]
    assert research["nonlocal_progress"] == 0


def test_impossible_negative_outcome_cannot_supply_conditional_experiment_value():
    state = fixture()
    # The new object shares the reference's basis representation. Its negative
    # response at the known positive frequency is impossible in this family.
    result = assess(state, experiments=[experiment("entailed_positive", 4, 10.)])
    research = result["family_research"]
    value = research["experiment_values"][0]
    assert value["outcome_statuses"] == ["UNSAT_VERIFIED", "SAT_WITNESS"]
    assert value["by_hypothetical_outcome"] == [2, 0]
    assert value["guaranteed_nonlocal_eliminations"] == 0
    assert value["possible_nonlocal_eliminations"] == 0
    assert research["selected_role"] == "operational_exploration"


def test_reference_positive_without_new_target_consequence_counts_as_stagnation():
    state = fixture()
    first = assess(state, experiments=[experiment("reference_10_5", 1, 10.5)])
    assert first["family_research"]["admitted_signatures"] == [[False], [True]]
    assert first["family_research"]["selected_test"] == {"crystal": 1, "frequency": 10.5}
    remember(state, first)
    state["response_settings"].append(response(1, 10.5, True))
    result = assess(state)
    research = result["family_research"]
    assert research["admitted_signatures"] == [[False], [True]]
    assert research["nonlocal_progress"] == 0
    assert research["no_progress_cycles"] == 1


def test_paid_history_exclusion_is_reused_for_every_hypothetical_condition(monkeypatch):
    state = fixture(panel_frequency=13.)
    state["response_settings"] += [response(1, 11., False), response(3, 14., True)]
    calls = []
    original = consequences.query_response_family

    def counted(features, responses, family, queries=(), **kwargs):
        calls.append((family["id"], list(queries)))
        return original(features, responses, family, queries, **kwargs)

    monkeypatch.setattr(consequences, "query_response_family", counted)
    result = assess(state, [LINEAR, QUADRATIC], [experiment("reference", 1, 10.5)])
    families = {item["id"]: item["status"] for item in result["family_research"]["families"]}
    assert families == {"linear": "UNSAT_VERIFIED", "quadratic": "SAT_WITNESS"}
    assert [queries for family, queries in calls if family == "linear"] == [[]]
    assert any(queries for family, queries in calls if family == "quadratic")


def test_unknown_paid_history_does_not_suppress_hypothetical_queries(monkeypatch):
    calls = []

    def unresolved_history(features, responses, family, queries=(), **kwargs):
        calls.append(list(queries))
        status = "SAT_WITNESS" if queries and queries[0]["activated"] else "UNKNOWN"
        return {"status": status, "compute_cost": 1, "lp_calls": 0}

    monkeypatch.setattr(consequences, "query_response_family", unresolved_history)
    result = assess(fixture())
    research = result["family_research"]
    assert research["families"][0]["status"] == "UNKNOWN"
    assert research["admitted_signatures"] == [[True]]
    assert research["possible_signatures"] == [[False], [True]]
    assert len(calls) == 3
    assert research["nonlocal_progress"] == 0


def test_unspendable_remainder_does_not_restart_solver_for_each_condition(monkeypatch):
    state = fixture()
    original = consequences.query_response_family
    calls = []

    def counted(features, responses, family, queries=(), **kwargs):
        assert kwargs["compute_limit"] >= len(features) * len(family["basis"])
        calls.append((family["id"], list(queries)))
        return original(features, responses, family, queries, **kwargs)

    monkeypatch.setattr(consequences, "QUERY_ALLOWANCE", 5)
    monkeypatch.setattr(consequences, "query_response_family", counted)
    result = assess(state, [LINEAR, QUADRATIC], [experiment("reference", 1, 10.5)])
    research = result["family_research"]
    assert calls == [("linear", [])]
    assert research["query_count"] == 1
    assert research["compute_cost"] == 4
    assert {family["status"] for family in research["families"]} == {"UNKNOWN"}
    assert research["possible_signatures"] == [[False], [True]]


def test_expensive_comparisons_leave_compute_for_other_subject_and_both_outcomes(monkeypatch):
    calls = []

    def expensive_conditions(features, responses, family, queries=(), **kwargs):
        if len(queries) < 2:
            return {"status": "SAT_WITNESS", "compute_cost": 1, "lp_calls": 0}
        calls.append((queries[0]["crystal"], queries[0]["activated"], kwargs["compute_limit"]))
        return {"status": "UNKNOWN", "compute_cost": kwargs["compute_limit"], "lp_calls": 0}

    monkeypatch.setattr(consequences, "QUERY_ALLOWANCE", 1003)
    monkeypatch.setattr(consequences, "query_response_family", expensive_conditions)
    result = assess(fixture(), experiments=[experiment("target", 3, 15.),
                                            experiment("reference", 1, 10.5)])
    research = result["family_research"]
    assert calls == [(3, False, 250), (3, True, 250), (1, False, 250), (1, True, 250)]
    assert research["compute_cost"] == 1003
    assert len(research["experiment_values"]) == 2
    for value in research["experiment_values"]:
        assert value["compute_allowance"] == value["compute_cost"] == 500
        assert value["outcome_compute_costs"] == [250, 250]
        assert value["outcome_statuses"] == ["UNKNOWN", "UNKNOWN"]
        assert value["guaranteed_nonlocal_eliminations"] == 0


def test_second_outcome_can_use_first_outcomes_unspent_share(monkeypatch):
    calls = []

    def cheap_negative(features, responses, family, queries=(), **kwargs):
        if len(queries) < 2 or not queries[0]["activated"]:
            return {"status": "SAT_WITNESS", "compute_cost": 1, "lp_calls": 0}
        calls.append(kwargs["compute_limit"])
        return {"status": "UNKNOWN", "compute_cost": kwargs["compute_limit"], "lp_calls": 0}

    monkeypatch.setattr(consequences, "QUERY_ALLOWANCE", 1003)
    monkeypatch.setattr(consequences, "query_response_family", cheap_negative)
    research = assess(fixture(), experiments=[experiment("reference", 1, 10.5)])["family_research"]
    assert calls == [998]
    assert research["experiment_values"][0]["outcome_compute_costs"] == [2, 998]
    assert research["experiment_values"][0]["outcome_statuses"] == ["SAT_WITNESS", "UNKNOWN"]
    assert research["compute_cost"] == 1003


@pytest.mark.parametrize("conjunction_status", ("SAT_WITNESS", "UNSAT_VERIFIED"))
def test_only_satisfiable_conjunctions_establish_the_single_outcome(monkeypatch, conjunction_status):
    singles = []

    def query(features, responses, family, queries=(), **kwargs):
        if len(queries) == 1 and queries[0]["crystal"] == 1:
            singles.append(queries[0]["activated"])
        status = conjunction_status if len(queries) == 2 else "SAT_WITNESS"
        return {"status": status, "compute_cost": 1, "lp_calls": 0}

    monkeypatch.setattr(consequences, "query_response_family", query)
    research = assess(fixture(), experiments=[experiment("reference", 1, 10.5)])["family_research"]
    assert singles == ([] if conjunction_status == "SAT_WITNESS" else [False, True])
    assert research["experiment_values"][0]["outcome_statuses"] == ["SAT_WITNESS", "SAT_WITNESS"]


def test_unknown_same_condition_retries_only_when_later_share_is_larger(monkeypatch):
    attempts = []

    def query(features, responses, family, queries=(), **kwargs):
        if len(queries) == 1 and queries[0]["crystal"] == 3:
            status = "UNSAT_VERIFIED" if queries[0]["activated"] else "SAT_WITNESS"
        elif len(queries) == 2 and queries[0]["crystal"] == 1:
            attempts.append((queries[0]["activated"], kwargs["compute_limit"]))
            return {"status": "UNKNOWN", "compute_cost": kwargs["compute_limit"], "lp_calls": 0}
        else:
            status = "SAT_WITNESS"
        return {"status": status, "compute_cost": 1, "lp_calls": 0}

    monkeypatch.setattr(consequences, "QUERY_ALLOWANCE", 1003)
    monkeypatch.setattr(consequences, "query_response_family", query)
    # Separate proposals can ask the same physical question. An inconclusive
    # first comparison must not permanently poison a later, larger allocation.
    research = assess(fixture(), experiments=[experiment("reference_a", 1, 10.5),
        experiment("other_reference", 2, 12.5), experiment("reference_b", 1, 10.5)])["family_research"]
    false_limits = [limit for outcome, limit in attempts if not outcome]
    true_limits = [limit for outcome, limit in attempts if outcome]
    assert len(false_limits) == 1
    assert len(true_limits) == 2
    assert true_limits[1] > true_limits[0]
    assert research["compute_cost"] <= 1003
