"""Method experience follows the measured executable structure after adoption."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.conditional_coordinates import (
    _masked, adopt_conditional_research_program)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, SearchBudget
from hegel_machine.cognition.numeric_method_revisits import (
    METHOD_VISIT_ACTION, _method_applicability, method_visit_policy)
from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
from hegel_machine.cognition.research_drafts import remember_research_draft
from hegel_machine.cognition.research_programs import _entry, _Fitter

from test_cognition_decoded_observed_conditions import DecodedObservedLab, random_probe


def measured_method(width, placement, *, renamed=False, response_scale=1., threshold=.173,
                    omit_quadratic=False):
    names = ("u", "v", "w") if renamed else ("alpha", "beta", "gamma")
    reading = "reading" if renamed else "meter"
    lab = DecodedObservedLab(seed=5811, width=width, placement=placement,
        names=names, meter=reading, response_scale=response_scale, threshold=threshold)
    spec = lab.spec_for((reading,))
    rows = [row for seed in (5812, 5813) for row in
        lab.acquire(random_probe(lab.spec, seed, 64), (reading,)).observations]
    basis = [Program("feature", name=names[0]), Program("feature", name=reading),
             Program("const", value=1.)]
    if not omit_quadratic:
        basis.append(Program("mul", children=(Program("feature", name=names[1]),) * 2))
    condition = Program("ge", children=(Program("feature", name=reading),
        Program("const", value=threshold)))
    budget = SearchBudget(20000)
    fit = _Fitter(rows, budget, .5).fit([_masked(term, condition, side, width, placement)
        for term in basis for side in (True, False)])
    assert fit is not None
    fresh = lab.acquire(random_probe(lab.spec, 5814, 32), (reading,)).observations
    if not omit_quadratic:
        assert fit["loss"] < 1e-12 and loss(fit["program"], fresh, 0., .5) < 1e-12
    else:
        assert fit["loss"] > 1e-7
    readings = sorted({row.features[reading] for row in rows})
    interval = [max(value for value in readings if value < threshold),
                min(value for value in readings if value >= threshold)]
    step = {"operation": "discover_conditional_coordinates", "question": "Measured branch computation",
        "program": fit["program"], "loss": fit["loss"], "width": width, "placement": placement,
        "gate_coordinate": Program("feature", name=reading).to_dict(),
        "threshold": threshold, "threshold_interval": interval,
        "branch_basis": [term.to_dict() for term in basis]}
    selected = _entry(fit, [step], len(rows), budget.spent)
    selected["condition_uncertainty"] = assess_observed_condition_uncertainty(
        spec, rows, selected, budget, tolerance=1e-7)
    library = KnowledgeLibrary()
    record = remember_research_draft(library, selected, spec, [])
    assert record is not None
    retained = {"candidate": Candidate("retained_response", fit["program"], "measured_fit")}
    return record, spec, fresh, retained


def adopted(record):
    current = deepcopy(record)
    procedure = current["research_program"]
    program = Program.from_dict(procedure["graph"]["nodes"][procedure["output_node"]]["answer"])
    current["research_program"] = adopt_conditional_research_program(procedure, program,
        measured_loss=current["condition_uncertainty"]["original_target_loss"],
        fit_rows=current["condition_uncertainty"]["fit_rows"], compute_cost=0)
    return json.loads(json.dumps(current))


@pytest.fixture(scope="module")
def actual_methods():
    return {
        "five_outside": measured_method(5, "outside_window"),
        "seven_outside": measured_method(7, "outside_window"),
        "five_inside": measured_method(5, "inside_window"),
        "changed_basis": measured_method(5, "outside_window", omit_quadratic=True),
        "renamed_refit": measured_method(5, "outside_window", renamed=True,
            response_scale=2.3, threshold=-.271),
    }


def test_adoption_keeps_the_executed_window_condition_and_basis(actual_methods):
    record, spec, rows, incumbent = actual_methods["five_outside"]
    current = adopted(record)
    assert current["research_program"]["selection_steps"][0]["historical"]
    assert current["research_program"]["selection_steps"][-1]["operation"] == "adopt_conditional_response"
    applicability = _method_applicability(current, spec, incumbent)
    assert applicability == _method_applicability(record, spec, incumbent)
    assert applicability["response_basis"]
    assert "window_sum" in json.dumps(applicability["response_basis"])
    assert '"lag": 5' in json.dumps(applicability["response_basis"])
    assert "command_1" in json.dumps(applicability["response_basis"])
    # A superseded discovery no longer describes the adopted answer.
    history = current["research_program"]["selection_steps"][0]
    history.update(width=19, placement="inside_window", branch_basis=[])
    assert _method_applicability(current, spec, incumbent) == applicability


@pytest.mark.parametrize("different", ("seven_outside", "five_inside", "changed_basis"))
def test_different_executing_structures_cannot_borrow_the_same_visit_return(actual_methods, different):
    record, spec, rows, incumbent = actual_methods["five_outside"]
    applicability = _method_applicability(adopted(record), spec, incumbent)
    other, other_spec, other_rows, other_incumbent = actual_methods[different]
    alternative = _method_applicability(adopted(other), other_spec, other_incumbent)
    assert applicability != alternative
    library = KnowledgeLibrary()
    policy = method_visit_policy(library, spec, rows, applicability)
    measured_rows = record["condition_uncertainty"]["fit_rows"] + len(rows)
    policy.update(METHOD_VISIT_ACTION,
        before_loss=loss(Program("const", value=0.), rows, 0., .5),
        after_loss=loss(incumbent["candidate"].program, rows, 0., .5),
        observation_cost=measured_rows, measurement_cost=measured_rows,
        compute_cost=record["research_program"]["search_compute"])
    separate = method_visit_policy(library, other_spec, other_rows, alternative)
    assert separate.action_record(METHOD_VISIT_ACTION)["trials"] == 0


def test_public_role_renaming_coefficients_and_new_threshold_keep_real_structure(actual_methods):
    record, spec, rows, incumbent = actual_methods["five_outside"]
    other, other_spec, other_rows, other_incumbent = actual_methods["renamed_refit"]
    first = _method_applicability(adopted(record), spec, incumbent)
    second = _method_applicability(adopted(other), other_spec, other_incumbent)
    assert first == second
    assert json.loads(json.dumps(first)) == first
