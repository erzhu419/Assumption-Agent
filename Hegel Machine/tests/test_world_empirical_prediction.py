"""Operational predictions remain testable after an exact law is contradicted."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_relation_research import research_relation


def request_for(values, *, degree=1, probe=3.):
    x = Program("feature", name="x")
    terms = [x]
    for _ in range(1, degree):
        terms.append(Program("mul", children=(terms[-1], x)))
    return {"rows": [{"features": {"x": float(index)}, "target": value, "evidence": [index]}
                     for index, value in enumerate(values)],
            "basis": [term.to_dict() for term in terms],
            "prediction_objective": "operational_setting",
            "probes": [{"id": "next_setting", "features": {"x": probe},
                        "actions": [{"action": "READ", "arg1": 4}]}]}


def public_for(request):
    return [{"index": index} for index in range(len(request["rows"]))]


def exact_fields(result):
    return {key: value for key, value in result.items()
            if key not in {"empirical_prediction", "compute_cost"}}


def test_refuted_exact_law_retains_an_empirical_prediction_with_measured_residuals():
    request = request_for([1., 4., 5.])
    result = research_relation(request, public_for(request))
    assert result["status"] == "refuted"
    assert result["selected_probe"] is None
    assert result["candidates"][0]["status"] == "refuted"
    empirical = result["empirical_prediction"]
    assert empirical["fit_mse"] == pytest.approx(2 / 9)
    assert [row["evidence"] for row in empirical["fit_residuals"]] == [[0], [1], [2]]
    assert [row["observed_setting"] for row in empirical["fit_residuals"]] == [1., 4., 5.]
    assert [row["error"] for row in empirical["fit_residuals"]] == pytest.approx([1 / 3, -2 / 3, 1 / 3])
    # Each held-out prediction has been fitted on the OTHER two observations.
    assert [row["prediction"] for row in empirical["leave_one_out_residuals"]] == pytest.approx([3., 3., 7.])
    assert empirical["leave_one_out_mse"] == pytest.approx(3.)
    assert empirical["probe_predictions"] == [{**request["probes"][0], "prediction": pytest.approx(22 / 3)}]
    assert empirical["evidence_use"] == "in_sample_fit_and_leave_one_out_reuse_not_fresh_confirmation"


def test_leave_one_out_exposes_interpolation_that_looks_better_in_sample():
    measured = [0., 1.1, 1.9, 3.]
    affine_request = request_for(measured, probe=4.)
    cubic_request = request_for(measured, degree=3, probe=4.)
    affine = research_relation(affine_request, public_for(affine_request))
    cubic = research_relation(cubic_request, public_for(cubic_request))
    assert cubic["empirical_prediction"]["fit_mse"] < 1e-6
    assert affine["empirical_prediction"]["fit_mse"] > 1e-6
    assert affine["empirical_prediction"]["leave_one_out_mse"] < cubic["empirical_prediction"]["leave_one_out_mse"]
    assert affine["status"] == "refuted"
    assert cubic["status"] == "inconclusive"
    assert not cubic["confirmation_evidence"]


@pytest.mark.parametrize("count", [2, 3, 4])
def test_empirical_work_is_charged_per_fit_row_and_prediction(count):
    request = request_for([1., 4., 5., 8.][:count], probe=5.)
    strict_request = deepcopy(request)
    strict_request.pop("prediction_objective")
    strict = research_relation(strict_request, public_for(request))
    empirical = research_relation(request, public_for(request))
    # All-row fit + in-sample predictions + n fits of n-1 rows and one
    # held-out prediction per fit + one supplied probe prediction.
    expected_work = 2 * count + (count * count if count >= 3 else 0) + 1
    assert empirical["compute_cost"] - strict["compute_cost"] == expected_work
    assert exact_fields(empirical) == exact_fields(strict)
    assert "empirical_prediction" not in strict
    if count < 3:
        assert empirical["empirical_prediction"]["leave_one_out_mse"] is None
        assert empirical["empirical_prediction"]["leave_one_out_residuals"] == []


def test_new_evidence_changes_operational_prediction_but_keeps_exact_counterexample():
    request = request_for([1., 3., 5.], probe=4.)
    first = research_relation(request, public_for(request))
    assert first["empirical_prediction"]["probe_predictions"][0]["prediction"] == pytest.approx(9.)
    request["rows"].append({"features": {"x": 3.}, "target": 7.1, "evidence": [3]})
    updated = research_relation(request, public_for(request), first)
    assert updated["status"] == "refuted"
    assert updated["diagnosis"] == "refit_needed"
    assert updated["rejected_candidates"][0]["program"] == first["selected_program"]
    assert updated["rejected_candidates"][0]["validation_mse"] == pytest.approx(.01)
    assert updated["empirical_prediction"]["probe_predictions"][0]["prediction"] == pytest.approx(9.1)
    assert updated["empirical_prediction"]["fit_residuals"][-1]["evidence"] == [3]


def test_empirical_refitting_does_not_change_frozen_confirmation():
    request = request_for([1., 3., 5.], probe=4.)
    request["tolerance"] = .001
    strict_request = deepcopy(request)
    strict_request.pop("prediction_objective")
    first = research_relation(request, public_for(request))
    strict_first = research_relation(strict_request, public_for(request))
    row = {"features": {"x": 3.}, "target": 7.01, "evidence": [3]}
    request["rows"].append(row)
    strict_request["rows"].append(deepcopy(row))
    updated = research_relation(request, public_for(request), first)
    strict_updated = research_relation(strict_request, public_for(request), strict_first)
    assert exact_fields(updated) == exact_fields(strict_updated)
    assert updated["status"] == "supported"
    assert updated["selected_program"] == first["selected_program"]
    assert updated["confirmation_evidence"] == [3]
    assert updated["empirical_prediction"]["probe_predictions"][0]["prediction"] == pytest.approx(9.01)


def test_operational_diagnostics_require_both_declared_basis_and_objective():
    request = request_for([1., 3., 5.])
    request["hypotheses"] = [Program("const", value=4.).to_dict()]
    request.pop("basis")
    assert "empirical_prediction" not in research_relation(request, public_for(request))
    request = request_for([1., 3., 5.])
    request["prediction_objective"] = "exact_law"
    assert "empirical_prediction" not in research_relation(request, public_for(request))
