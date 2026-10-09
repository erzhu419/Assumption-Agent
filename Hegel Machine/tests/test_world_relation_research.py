from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_relation_research import research_relation


def binding():
    x = Program("feature", name="x")
    return {"rows": [{"features": {"x": value}, "target": value, "evidence": [index]}
                     for index, value in enumerate((0., 1.))],
            "hypotheses": [x.to_dict(), Program("mul", children=(x, x)).to_dict()],
            "probes": [{"id": "uninformative", "features": {"x": 1.}, "actions": [{"action": "READ", "arg1": 1}]},
                       {"id": "distinguish", "features": {"x": 2.}, "actions": [{"action": "READ", "arg1": 2}]}]}


def test_frozen_candidates_choose_experiment_then_separately_confirm_selected_answer():
    public = [{"index": i} for i in range(4)]
    request = binding()
    first = research_relation(request, public[:2])
    assert first["status"] == "inconclusive"
    assert first["selected_probe"]["id"] == "distinguish"
    assert first["selected_probe"]["actions"] == request["probes"][1]["actions"]
    assert first["selected_probe"]["predictions"] == [2., 4.]
    request["rows"].append({"features": {"x": 2.}, "target": 4., "evidence": [2]})
    request["probes"].append({"id": "confirm", "features": {"x": 3.}, "actions": [{"action": "READ", "arg1": 3}]})
    second = research_relation(request, public[:3], first)
    assert second["status"] == "inconclusive"
    assert second["diagnosis"] == "needs_confirmation"
    assert second["selected_probe"]["id"] == "confirm"
    assert second["selected_probe"]["purpose"] == "confirmation"
    assert second["fit_evidence"] == [0, 1]
    assert second["validation_evidence"] == [2]
    assert [item["status"] for item in second["candidates"]] == ["refuted", "plausible"]
    assert second["candidates"][0]["validation_mse"] == 4.
    assert second["candidates"][1]["validation_mse"] == 0.
    request["rows"].append({"features": {"x": 3.}, "target": 9., "evidence": [3]})
    third = research_relation(request, public, second)
    assert third["status"] == "supported"
    assert third["confirmation_evidence"] == [3]
    assert third["selected_probe"] is None


def test_joint_mismatch_refits_without_erasing_frozen_counterexamples():
    request = binding()
    first = research_relation(request, [{"index": 0}, {"index": 1}])
    request.pop("hypotheses")
    request["rows"].append({"features": {"x": 2.}, "target": 8., "evidence": [2]})
    second = research_relation(request, [{"index": i} for i in range(3)], first)
    assert second["status"] == "refuted"
    assert second["diagnosis"] == "refit_needed"
    assert all(item["validation_mse"] > 0 for item in second["rejected_candidates"])
    assert second["fit_evidence"] == [0, 1, 2]
    assert second["validation_evidence"] == []
    assert all(item["validation_mse"] is None for item in second["candidates"])


def test_no_distinguishing_experiment_does_not_select_redundant_purchase():
    request = binding()
    request["probes"] = request["probes"][:1]
    result = research_relation(request, [{"index": 0}, {"index": 1}])
    assert result["status"] == "inconclusive"
    assert result["diagnosis"] == "observational_ambiguity"
    assert result["selected_probe"] is None


def test_fabricated_observation_reference_is_rejected():
    with pytest.raises(ValueError, match="actual observation indices"):
        research_relation(binding(), [{"index": 0}])


def test_overlapping_training_citations_cannot_be_new_validation():
    public = [{"index": i} for i in range(3)]
    request = binding()
    request["hypotheses"] = request["hypotheses"][:1]
    first = research_relation(request, public[:2])
    assert first["status"] == "inconclusive"
    request["rows"].append({"features": {"x": 2.}, "target": 2., "evidence": [1, 2]})
    second = research_relation(request, public, first)
    assert second["status"] == "inconclusive"
    assert second["validation_evidence"] == []


def test_changed_numeric_binding_does_not_inherit_old_support():
    request = binding()
    request["hypotheses"] = request["hypotheses"][:1]
    public = [{"index": i} for i in range(3)]
    first = research_relation(request, public[:2])
    request["rows"].append({"features": {"x": 2.}, "target": 2., "evidence": [2]})
    supported = research_relation(request, public, first)
    assert supported["status"] == "supported"
    changed = deepcopy(request)
    changed["rows"][0]["target"] = 9.
    result = research_relation(changed, public, supported)
    assert result["status"] != "supported"
    assert result["validation_evidence"] == []


def test_single_fitted_candidate_gets_a_fresh_confirmation_action():
    request = binding()
    request["hypotheses"] = request["hypotheses"][:1]
    first = research_relation(request, [{"index": 0}, {"index": 1}])
    assert first["diagnosis"] == "needs_confirmation"
    assert first["selected_probe"]["id"] == "distinguish"
    assert first["selected_probe"]["purpose"] == "confirmation"


def test_newly_bound_old_readout_is_not_fresh_confirmation():
    public = [{"index": i} for i in range(3)]
    request = binding()
    request["hypotheses"] = request["hypotheses"][:1]
    first = research_relation(request, public)
    request["rows"].append({"features": {"x": 2.}, "target": 2., "evidence": [2]})
    second = research_relation(request, public, first)
    assert second["status"] == "inconclusive"
    assert second["confirmation_evidence"] == []


def test_confirmation_failure_cannot_be_diluted_by_selection_evidence():
    public = [{"index": i} for i in range(4)]
    request = binding()
    first = research_relation(request, public[:2])
    request["rows"].append({"features": {"x": 2.}, "target": 4., "evidence": [2]})
    selected = research_relation(request, public[:3], first)
    request["rows"].append({"features": {"x": 3.}, "target": 9.0013, "evidence": [3]})
    result = research_relation(request, public, selected)
    assert result["status"] == "refuted"
    assert result["diagnosis"] == "refit_needed"
    rejected = result["rejected_candidates"][0]
    assert rejected["validation_mse"] < 1e-6 < rejected["confirmation_mse"]


def quadratic_basis_binding():
    x = Program("feature", name="x")
    return {"rows": [{"features": {"x": value}, "target": 2 + 3 * value + value * value,
                      "evidence": [index]} for index, value in enumerate((0., 1., 2.))],
            "basis": [x.to_dict(), Program("mul", children=(x, x)).to_dict()],
            "probes": [{"id": "new_point", "features": {"x": 3.},
                        "actions": [{"action": "READ", "arg1": 3}]}]}


def test_declared_basis_fits_one_prediction_and_requires_fresh_confirmation():
    request = quadratic_basis_binding()
    public = [{"index": i} for i in range(4)]
    first = research_relation(request, public[:3])

    assert len(first["candidates"]) == 1
    assert first["status"] == "inconclusive"
    assert first["diagnosis"] == "needs_confirmation"
    assert first["selected_probe"]["predictions"] == pytest.approx([20.])
    assert first["fit_evidence"] == [0, 1, 2]
    assert first["basis"] == request["basis"]

    request["rows"].append({"features": {"x": 3.}, "target": 20., "evidence": [3]})
    confirmed = research_relation(request, public, first)
    assert confirmed["status"] == "supported"
    assert confirmed["selected_program"] == first["selected_program"]
    assert confirmed["confirmation_evidence"] == [3]


def test_contradicted_basis_refits_that_family_without_widening_search():
    request = quadratic_basis_binding()
    request["basis"] = request["basis"][:1]
    request["rows"] = [{"features": {"x": value}, "target": value, "evidence": [index]}
                       for index, value in enumerate((0., 1.))]
    public = [{"index": i} for i in range(3)]
    first = research_relation(request, public[:2])
    request["rows"].append({"features": {"x": 2.}, "target": 4., "evidence": [2]})
    rejected = research_relation(request, public, first)

    assert rejected["status"] == "refuted"
    assert rejected["diagnosis"] == "refit_needed"
    assert rejected["rejected_candidates"][0]["program"] == first["selected_program"]
    assert rejected["rejected_candidates"][0]["validation_mse"] == pytest.approx(4.)
    assert len(rejected["candidates"]) == 1
    assert rejected["candidates"][0]["fit_mse"] > request.get("tolerance", 1e-6)
    assert rejected["basis"] == request["basis"]

    # The other declared family remains available, with its own confirmation.
    revised = quadratic_basis_binding()
    revised["rows"] = request["rows"]
    second = research_relation(revised, public)
    assert second["status"] == "inconclusive"
    assert second["selected_probe"]["predictions"] == pytest.approx([9.])


def test_changing_declared_basis_reopens_previously_supported_relation():
    request = quadratic_basis_binding()
    public = [{"index": i} for i in range(4)]
    first = research_relation(request, public[:3])
    request["rows"].append({"features": {"x": 3.}, "target": 20., "evidence": [3]})
    supported = research_relation(request, public, first)
    assert supported["status"] == "supported"

    request["basis"] = request["basis"][:1]
    revised = research_relation(request, public, supported)
    assert revised["status"] == "refuted"
    assert len(revised["candidates"]) == 1
    assert revised["validation_evidence"] == []
    assert "selected_program" not in revised


def test_explicit_hypotheses_take_precedence_over_declared_basis():
    request = quadratic_basis_binding()
    request["hypotheses"] = [Program("const", value=0.).to_dict()]
    result = research_relation(request, [{"index": i} for i in range(3)])
    assert result["status"] == "refuted"
    assert [item["program"] for item in result["candidates"]] == request["hypotheses"]
