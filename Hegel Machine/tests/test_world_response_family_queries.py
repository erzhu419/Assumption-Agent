"""Family consequences are existential queries, not representative failures."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition import world_response_constraints as constraints


X = Program("feature", name="x")
LINEAR = {"id": "linear", "basis": [X.to_dict()]}
SHARED_X_Z = {"id": "shared_x_z", "basis": [X.to_dict(), Program("feature", name="z").to_dict()]}


def response(crystal, frequency, activated):
    return {"crystal": crystal, "frequency": frequency, "activated": activated}


def check_witness(result, features, responses, queries=()):
    assert result["status"] == "SAT_WITNESS"
    witness = result["witness"]
    program = Program.from_dict(witness["program"])
    for item in [*responses, *queries]:
        prediction = program.predict(Observation(features[item["crystal"]], 0.))
        assert witness["predictions_by_crystal"][str(item["crystal"])] == prediction
        assert (witness["lower"] <= item["frequency"] - prediction <= witness["upper"]) \
            == item["activated"]
    assert witness["checked_responses"] == len(responses)
    assert witness["checked_queries"] == len(queries)
    json.dumps(result, allow_nan=False)


def test_history_can_be_feasible_without_positive_or_requested_target_activation():
    features = {1: {"x": 0.}, 2: {"x": 0.}}
    responses = [response(1, 10., False)]
    before = deepcopy((features, responses, LINEAR))
    base = constraints.query_response_family(features, responses, LINEAR)
    check_witness(base, features, responses)
    query = [response(2, 10., True)]
    requested = constraints.query_response_family(features, responses, LINEAR, query)
    assert requested["status"] == "UNSAT_VERIFIED"
    assert requested["reason"] == "equal_basis_connected_band_conflict"
    assert requested["lp_calls"] == 0
    assert requested["query_role"] == "hypothetical_conditions_not_observed_evidence"
    assert requested["paid_response_count"] == 1
    assert (features, responses, LINEAR) == before


def test_a_failed_representative_does_not_refute_its_linear_family():
    features = {1: {"x": 0.}, 2: {"x": 1.}}
    responses = [response(1, 10., True), response(1, 9., False), response(1, 11., False)]
    first = constraints.query_response_family(features, responses, LINEAR)
    check_witness(first, features, responses)
    witness = first["witness"]
    failed_setting = witness["predictions_by_crystal"]["2"] + (witness["lower"] + witness["upper"]) / 2.
    feedback = response(2, failed_setting, False)
    revised = constraints.query_response_family(features, [*responses, feedback], LINEAR)
    check_witness(revised, features, [*responses, feedback])
    assert revised["paid_response_count"] == 4


def test_truncated_gap_enumeration_cannot_exclude_unvisited_feasible_branch():
    features = {1: {"x": 0.}, 2: {"x": 1.}, 3: {"x": 2.}}
    responses = [response(1, 0., True), response(3, 2., True), response(2, 0., False)]
    truncated = constraints.query_response_family(features, responses, LINEAR, branch_limit=1)
    assert truncated["status"] == "UNKNOWN"
    assert truncated["branches_attempted"] == 1
    assert truncated["branch_count"] == 2
    assert truncated["branches_excluded_exactly"] == 1
    assert truncated["limited_by_branch_budget"]
    full = constraints.query_response_family(features, responses, LINEAR, branch_limit=2)
    check_witness(full, features, responses)


@pytest.mark.parametrize("degree", (1, 2))
def test_actual_compute_allowance_can_complete_more_than_128_paid_gaps(degree):
    features = {number: {"x": float(number - 1)} for number in range(1, 6)}
    # Tight response bounds on a cubic-shaped reference sequence exclude both
    # declared lower-degree families. The fifth object's negative-only history
    # has 131 possible band positions, as in the actual long-lived inquiry.
    responses = [response(number, float((number - 1) ** 3), True)
                 for number in range(1, 5)]
    responses += [response(number, (number - 1) ** 3 + offset, False)
                  for number in range(1, 5) for offset in (-0.1, 0.1)]
    responses += [response(5, float(value), False) for value in range(130)]
    terms = [X] if degree == 1 else [X, Program("mul", children=(X, X))]
    estimator = {"id": f"degree_{degree}", "basis": [term.to_dict() for term in terms]}
    before = deepcopy((features, responses, estimator))

    truncated = constraints.query_response_family(
        features, responses, estimator, compute_limit=50000)
    assert truncated["status"] == "UNKNOWN"
    assert truncated["reason"] == "branch_limit"
    assert truncated["branches_attempted"] == 128

    complete = constraints.query_response_family(
        features, responses, estimator, branch_limit=None, compute_limit=50000)
    assert complete["status"] == "UNSAT_VERIFIED"
    assert complete["branches_attempted"] == complete["branches_excluded_exactly"] == 131
    assert not complete["limited_by_branch_budget"]
    assert complete["compute_cost"] <= 50000
    assert complete["lp_calls"] == 0

    limited = constraints.query_response_family(
        features, responses, estimator, branch_limit=None, compute_limit=1000)
    assert limited["status"] == "UNKNOWN"
    assert limited["reason"] == "query_compute_limit"
    assert 0 < limited["compute_cost"] <= 1000
    assert limited["branches_excluded_exactly"] < limited["branch_count"]
    assert (features, responses, estimator) == before


def test_removing_branch_cap_requires_an_actual_compute_allowance():
    result = constraints.query_response_family(
        {1: {"x": 0.}}, [response(1, 10., True)], LINEAR, branch_limit=None)
    assert result["status"] == "INVALID_BINDING"
    assert result["compute_cost"] == 0
    assert result["lp_calls"] == 0


def test_new_reference_constraint_changes_unmeasured_target_consequences():
    features = {1: {"x": 0.}, 2: {"x": 2.}, 3: {"x": 1.}}
    responses = [response(1, 10., True), response(1, 9., False), response(2, 12., True)]
    target_positive = [response(3, 14., True)]
    target_negative = [response(3, 14., False)]
    before = constraints.query_response_family(features, responses, LINEAR, target_positive)
    check_witness(before, features, responses, target_positive)
    reference_feedback = response(1, 11., False)
    new_evidence = [*responses, reference_feedback]
    excluded = constraints.query_response_family(features, new_evidence, LINEAR, target_positive)
    assert excluded["status"] == "UNSAT_VERIFIED"
    assert excluded["reason"] == "exact_one_dimensional_branch_exclusion"
    assert excluded["branches_excluded_exactly"] == excluded["branch_count"]
    assert excluded["exact_comparisons"] > 0
    surviving = constraints.query_response_family(features, new_evidence, LINEAR, target_negative)
    check_witness(surviving, features, new_evidence, target_negative)
    # The changed consequence came exclusively from a reference observation.
    assert all(item["crystal"] != 3 for item in new_evidence)
    assert responses == [response(1, 10., True), response(1, 9., False), response(2, 12., True)]


@pytest.mark.parametrize("positive_frequencies,negative_frequency", (([10.], 10.), ([9., 11.], 10.)))
def test_identical_basis_representation_conflict_is_exactly_excluded(positive_frequencies, negative_frequency):
    features = {1: {"x": 4.}, 2: {"x": 4.}}
    responses = [response(1, value, True) for value in positive_frequencies]
    responses.append(response(2, negative_frequency, False))
    result = constraints.query_response_family(features, responses, LINEAR)
    assert result["status"] == "UNSAT_VERIFIED"
    assert result["reason"] == "equal_basis_connected_band_conflict"
    assert result["lp_calls"] == 0


def test_solver_infeasibility_alone_is_not_a_higher_dimensional_exclusion(monkeypatch):
    square = Program("mul", children=(X, X))
    estimator = {"id": "quadratic", "basis": [X.to_dict(), square.to_dict()]}
    features = {1: {"x": 0.}, 2: {"x": 1.}, 3: {"x": 2.}}
    monkeypatch.setattr(constraints, "linprog", lambda *args, **kwargs:
                        SimpleNamespace(success=False, status=2, nit=0))
    result = constraints.query_response_family(features, [response(1, 10., True)], estimator)
    assert result["status"] == "UNKNOWN"
    assert result["solver_statuses"] == {"2": 1}
    assert result["branches_excluded_exactly"] == 0


def test_no_rows_and_constant_basis_are_feasible_without_extra_bounds():
    features = {1: {}, 2: {}}
    estimator = {"id": "constant", "basis": []}
    result = constraints.query_response_family(features, [], estimator)
    check_witness(result, features, [])
    assert result["lp_calls"] == 1


def test_missing_measurement_is_invalid_binding_not_family_refutation():
    result = constraints.query_response_family({1: {}}, [response(1, 10., True)], LINEAR)
    assert result["status"] == "INVALID_BINDING"
    assert result["lp_calls"] == 0


def test_many_negative_comparisons_stop_at_the_actual_compute_allowance():
    features = {1: {"x": 0.}, 2: {"x": 1.}}
    responses = [response(1, 10., True)]
    responses += [response(1, float(value), False) for value in range(21) if value != 10]
    responses.append(response(2, 10., False))
    budget = 20
    limited = constraints.query_response_family(features, responses, LINEAR, compute_limit=budget)
    assert limited["status"] == "UNKNOWN"
    assert limited["reason"] == "query_compute_limit"
    assert limited["compute_cost"] <= budget
    # The allowance cannot even cover the endpoint-reduction scan. A partial
    # scan is not enough to claim that the remaining responses were considered.
    assert limited["constraint_rows_scanned"] == 0
    assert limited["exact_comparisons"] == 0
    assert limited["lp_calls"] == 0
    complete = constraints.query_response_family(features, responses, LINEAR)
    check_witness(complete, features, responses)
    assert complete["compute_cost"] > budget


def test_long_paid_history_reaches_feasible_late_branch_within_existing_allowance():
    square = Program("mul", children=(X, X))
    estimator = {"id": "quadratic", "basis": [X.to_dict(), square.to_dict()]}
    features = {n: {"x": x} for n, x in enumerate((1., 2., 3., 4., 19.887181801351343), 1)}
    responses = [response(n, value, True)
                 for n, value in enumerate((41.2, 48.2, 67.2, 104.), 1)]
    responses += [response(1, value, False) for value in (40.2, 42.2)]
    responses += [response(4, float(value), False)
                  for value in (*range(78, 104), *range(107, 125))]
    responses += [response(5, float(value), False) for value in (*range(0, 2701, 50), 2671)]
    before = deepcopy(responses)
    # The former all-pairs reduction spent 50,000 on the first 19 of 57
    # branches and returned UNKNOWN. No new responses or larger budget are used.
    result = constraints.query_response_family(features, responses, estimator, compute_limit=50000)
    check_witness(result, features, responses)
    assert result["branches_attempted"] > 19
    assert result["compute_cost"] < 10000
    assert result["constraint_rows_scanned"] > 0
    assert responses == before


def test_dominated_measurements_preserve_exact_exclusion_without_pair_explosion(monkeypatch):
    square = Program("mul", children=(X, X))
    estimator = {"id": "quadratic", "basis": [X.to_dict(), square.to_dict()]}
    features = {n: {"x": float(n)} for n in range(1, 5)}
    responses = [response(n, float(n ** 3), True) for n in features]
    responses += [response(n, n ** 3 + sign * offset / 2., False)
                  for n in features for sign in (-1., 1.) for offset in range(1, 51)]
    monkeypatch.setattr(constraints, "linprog", lambda *args, **kwargs:
                        pytest.fail("the inconsistent quadratic must be excluded exactly"))
    result = constraints.query_response_family(features, responses, estimator, compute_limit=2000)
    assert result["status"] == "UNSAT_VERIFIED"
    assert result["reason"] == "exact_two_dimensional_branch_exclusion"
    assert result["branches_excluded_exactly"] == result["branch_count"] == 1
    assert result["compute_cost"] < 2000


def test_endpoint_reduction_retains_both_positive_extremes_and_all_witness_checks():
    features = {1: {"x": 0.}, 2: {"x": 0.}, 3: {"x": 1.}}
    responses = [response(1, value, True) for value in (10., 10.5, 11.)]
    responses += [response(2, value, True) for value in (11., 11.5, 12.)]
    responses += [response(1, value, False) for value in (7., 8., 9.)]
    responses += [response(2, value, False) for value in (13., 14., 15.)]
    result = constraints.query_response_family(features, responses, LINEAR)
    check_witness(result, features, responses)
    # A positive interval of width two cannot fit between these two negatives,
    # whatever slope is chosen for the third object.
    excluded = constraints.query_response_family(features, responses, LINEAR,
        [response(3, 20., False), response(3, 21., True), response(3, 22., False)])
    assert excluded["status"] == "UNSAT_VERIFIED"


def test_compute_allowance_must_cover_basis_and_final_executable_checks():
    features = {1: {"x": 0.}, 2: {"x": 1.}}
    responses = [response(1, 10., True), response(1, 9., False), response(1, 11., False)]
    before_basis = constraints.query_response_family(features, responses, LINEAR, compute_limit=1)
    assert before_basis["status"] == "UNKNOWN"
    assert before_basis["reason"] == "query_compute_limit"
    assert before_basis["compute_cost"] == 0
    full = constraints.query_response_family(features, responses, LINEAR)
    check_witness(full, features, responses)
    budget = full["compute_cost"] - 1
    before_verification = constraints.query_response_family(features, responses, LINEAR, compute_limit=budget)
    assert before_verification["status"] == "UNKNOWN"
    assert before_verification["reason"] == "query_compute_limit"
    assert before_verification["lp_calls"] == 1
    assert before_verification["witness"] is None
    assert before_verification["compute_cost"] <= budget


def test_two_variable_family_excludes_negative_at_convex_positive_prediction(monkeypatch):
    features = {1: {"x": 0., "z": 0.}, 2: {"x": 2., "z": 0.},
                3: {"x": 0., "z": 2.}, 4: {"x": 1., "z": 1.}}
    responses = [response(1, 0., True), response(2, 8., True), response(3, 4., True)]
    positive = [response(4, 6., True)]
    check_witness(constraints.query_response_family(features, responses, SHARED_X_Z, positive),
                  features, responses, positive)
    # Exact rejection must not depend on the numerical solver's verdict.
    monkeypatch.setattr(constraints, "linprog", lambda *args, **kwargs:
                        pytest.fail("exactly inconsistent branches should not invoke the LP"))
    excluded = constraints.query_response_family(features, responses, SHARED_X_Z,
                                                  [response(4, 6., False)])
    assert excluded["status"] == "UNSAT_VERIFIED"
    assert excluded["reason"] == "exact_two_dimensional_branch_exclusion"
    assert excluded["varying_basis_terms"] == 2
    assert excluded["branches_excluded_exactly"] == excluded["branch_count"] == 2


def test_two_variable_elimination_keeps_strict_negative_boundary_and_feasible_neighbour():
    features = {1: {"x": 0., "z": 0.}, 2: {"x": 2., "z": 0.},
                3: {"x": 0., "z": 2.}, 4: {"x": 1., "z": 1.}}
    responses = [response(1, 0., True), response(2, 8., True), response(3, 4., True)]
    queries = [response(4, 6.000001, False)]
    result = constraints.query_response_family(features, responses, SHARED_X_Z, queries)
    check_witness(result, features, responses, queries)
    assert result["branches_excluded_exactly"] < result["branch_count"]


def test_two_variable_elimination_is_bounded_and_truncation_cannot_refute_family():
    features = {1: {"x": 0., "z": 0.}, 2: {"x": 2., "z": 0.},
                3: {"x": 0., "z": 2.}, 4: {"x": 1., "z": 1.}}
    responses = [response(1, 0., True), response(2, 8., True), response(3, 4., True)]
    queries = [response(4, 6., False)]
    full = constraints.query_response_family(features, responses, SHARED_X_Z, queries)
    for budget in (8, 11, full["compute_cost"] - 1):
        result = constraints.query_response_family(features, responses, SHARED_X_Z,
                                                    queries, compute_limit=budget)
        assert result["status"] == "UNKNOWN"
        assert result["reason"] == "query_compute_limit"
        assert result["compute_cost"] <= budget
    truncated = constraints.query_response_family(features, responses, SHARED_X_Z,
                                                   queries, branch_limit=1)
    assert truncated["status"] == "UNKNOWN"
    assert truncated["limited_by_branch_budget"]
    assert truncated["branches_excluded_exactly"] == 1


def test_shared_feature_pilot_acquired_reference_response_resolves_both_unmeasured_targets():
    features = {1: {"x": 0., "z": 0.}, 2: {"x": 0., "z": 2.},
                3: {"x": 2., "z": 0.}, 4: {"x": 1., "z": 1.},
                5: {"x": 1.5, "z": 0.}}
    history = [response(1, 0., True), response(2, 4., True), response(3, 10., True),
               response(1, 3., False)]
    assert constraints.query_response_family(features, history, LINEAR)["status"] == "UNSAT_VERIFIED"
    for number in (4, 5):
        negative = [response(number, 6., False)]
        check_witness(constraints.query_response_family(features, history, SHARED_X_Z, negative),
                      features, history, negative)
    acquired = [*history, response(3, 8., True)]
    for number in (4, 5):
        excluded = constraints.query_response_family(features, acquired, SHARED_X_Z,
                                                      [response(number, 6., False)])
        assert excluded["status"] == "UNSAT_VERIFIED"
        assert excluded["reason"] == "exact_two_dimensional_branch_exclusion"
    both_positive = [response(4, 6., True), response(5, 6., True)]
    check_witness(constraints.query_response_family(features, acquired, SHARED_X_Z, both_positive),
                  features, acquired, both_positive)
    assert all(row["crystal"] <= 3 for row in acquired)


def test_three_varying_terms_still_need_evidence_beyond_solver_infeasibility(monkeypatch):
    z = Program("feature", name="z")
    estimator = {"id": "xyz", "basis": [X.to_dict(), z.to_dict(),
                                          Program("feature", name="y").to_dict()]}
    features = {1: {"x": 0., "z": 0., "y": 0.}, 2: {"x": 1., "z": 2., "y": 3.}}
    monkeypatch.setattr(constraints, "linprog", lambda *args, **kwargs:
                        SimpleNamespace(success=False, status=2, nit=0))
    result = constraints.query_response_family(features, [response(1, 10., True)], estimator)
    assert result["varying_basis_terms"] == 3
    assert result["status"] == "UNKNOWN"
    assert result["branches_excluded_exactly"] == 0
    assert result["solver_statuses"] == {"2": 1}
