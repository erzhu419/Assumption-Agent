"""Public coordinate research completes joint repairs inside the real local cap."""

from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import (
    _Fitter, _plan, propose_research_programs, propose_research_step_revisions,
    replay_research_program,
)


def guarded_lag(name, lag):
    return Program("if", children=(Program("available", lag=lag),
        Program("lag", name=name, lag=lag), Program("const", value=0.)))


def fixture(seed=212):
    rng, rows = Random(seed), []
    names = ("a", "b", "c", "d", "e", "f")
    for _ in range(5):
        history = []
        for _ in range(29):
            point = {name: rng.uniform(-2., 2.) for name in names}
            target = point["a"] + .7 * point["b"] * (history[-4]["a"] if len(history) >= 4 else 0.)
            rows.append(Observation(point, target, tuple(history)))
            history.append(point)
    atoms = [Program("feature", name=name) for name in ("a", "c", "d", "e", "f")]
    # Two current terms, eight historical distinctions and five supplied
    # parents contain fifteen distinct computations once the incumbent is
    # made a composite expression, as in the repaired production call.
    atoms[0] = Program("add", children=(atoms[0], Program("const", value=.1)))
    coordinates = [Program("feature", name=name) for name in ("a", "b")]
    coordinates.extend(guarded_lag(name, lag) for lag in range(1, 5) for name in ("a", "b"))
    spec = TaskSpec("public-coordinate-task", "Predict acquired responses", names, (), ())
    return spec, rows, atoms, coordinates


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def test_fifteen_atoms_complete_joint_repair_before_105_singleton_fits(monkeypatch):
    spec, rows, atoms, coordinates = fixture()
    assert len(rows) == 145 and len(set(atoms + coordinates)) == 15
    work, progress, fits = SearchBudget(41308), {}, []
    original = _Fitter.fit

    def observed(self, terms):
        fits.append((tuple(terms), len(self.columns), self.budget.spent))
        return original(self, terms)

    monkeypatch.setattr(_Fitter, "fit", observed)
    claims = propose_research_programs(spec, rows, work, atoms,
        coordinates=coordinates, progress=progress)
    assert claims
    best = min(claims, key=lambda claim: claim["training_loss"])
    assert best["training_loss"] < 1e-10
    assert mse(best["candidate"].program, fixture(901)[1]) < 1e-10
    assert fits and len(fits[0][0]) == 2 and fits[0][1] <= 15
    assert not any(len(terms) == 1 for terms, _, _ in fits)
    assert 0 < progress["completed_joint_fits"] == work.completed_fits <= 8
    assert progress["status"] == "training_residual_explained"
    assert progress["compute_cost"] == work.spent <= 41308
    first = best["research_program"]["selection_steps"][0]
    assert first["program"] == atoms[0].to_dict()
    assert first["loss"] == pytest.approx(mse(atoms[0], rows))
    assert best["research_program"]["fit_rows"] == 145
    replay = replay_research_program(best["research_program"], fixture(340)[1], SearchBudget(5000))
    assert replay and mse(replay["candidate"].program, fixture(902)[1]) < 1e-10


def test_unfunded_joint_repair_has_zero_new_compute():
    spec, rows, atoms, coordinates = fixture()
    work, progress = SearchBudget(7 * len(rows) - 1), {}
    assert propose_research_programs(spec, rows, work, atoms,
        coordinates=coordinates, progress=progress) == []
    assert work.spent == work.completed_fits == 0
    assert progress["status"] == "insufficient_joint_fit_budget"
    assert progress["last_denied_request"]["required_compute"] == 7 * len(rows)


def test_small_allocation_keeps_completed_joint_fits_and_reports_unsearched_tail():
    spec, rows, atoms, coordinates = fixture()
    work, progress = SearchBudget(4500, reserve=300), {}
    claims = propose_research_programs(spec, rows, work, atoms,
        coordinates=coordinates, progress=progress)
    assert claims and progress["completed_joint_fits"] > 0
    assert progress["screening_cutoffs"]
    assert progress["status"] == "incomplete_joint_search"
    assert work.reserve == 300 and work.spent <= 4200
    assert all(item["research_program"]["status"] == "requires_fresh_validation" for item in claims)


def test_interruption_restores_caller_reserve_and_retains_paid_account(monkeypatch):
    spec, rows, atoms, coordinates = fixture()
    work, progress = SearchBudget(41308, reserve=700), {}
    original = _Fitter.column

    def interrupted(self, term):
        result = original(self, term)
        if len(self.columns) == 3:
            raise RuntimeError("interrupted after the paid readout")
        return result

    monkeypatch.setattr(_Fitter, "column", interrupted)
    with pytest.raises(RuntimeError, match="paid readout"):
        propose_research_programs(spec, rows, work, atoms,
            coordinates=coordinates, progress=progress)
    assert work.reserve == 700
    assert progress["status"] == "interrupted_joint_search"
    assert progress["compute_cost"] == work.spent > 0


def test_history_only_private_coordinate_cannot_enter_the_new_search():
    spec, rows, atoms, _ = fixture()
    private = guarded_lag("__target__", 1)
    assert private.required_features() == set()
    work = SearchBudget(41308)
    assert propose_research_programs(spec, rows, work, atoms, coordinates=[private]) == []
    assert work.spent == 0


def test_historical_method_edits_change_gate_and_branch_using_public_history():
    _, rows, _, _ = fixture()
    term = Program("feature", name="a")
    recipe = _plan({"fit_terms": (term,), "coefficients": [0., 1.],
        "program": term, "loss": 1.}, [{"operation": "discover_conditional_coordinates",
        "question": "Compare conditional responses", "program": term, "loss": 1.,
        "gate_coordinate": term.to_dict(), "branch_basis": [term.to_dict()],
        "threshold": 0., "width": 1, "placement": "inside_window"}], len(rows), 100)
    # Actual histories can include the measured response, but it is not an
    # input available to a predictor of the current response.
    rows = [Observation(row.features, row.target, tuple({**past, "__target__": 99.}
        for past in row.history)) for row in rows]
    result = propose_research_step_revisions(recipe, rows, max_candidates=80, max_history=2)
    for kind in ("replace_gate", "add_branch_coordinate"):
        edits = [item for item in result["candidates"] if item["edit"]["kind"] == kind]
        assert any(Program.from_dict(item["edit"]["term"]).history_depth for item in edits)
        assert all(name != "__target__" for item in edits for name, _ in
            Program.from_dict(item["edit"]["term"]).observed_variables())
    assert recipe["selection_steps"][0]["branch_basis"] == [term.to_dict()]
