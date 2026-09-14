"""Actual fits determine when an unresolved assignment deserves another search."""

from dataclasses import replace
import json
from random import Random

from hegel_machine.cognition.competing_object_programs import research_competing_object_programs
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec


def _observed_problem():
    quantities = ["q_a", "q_b"]
    model = {"model_id": "observed_roles", "complete": False, "schema": None,
        "focus_control": "select", "unresolved": ["a:x", "b:x"],
        "groups": [{"object_id": name, "q": f"q_{name}", "u": f"u_{name}",
                    "f": f"f_{name}", "controls": {"q": f"set_q_{name}", "u": f"set_u_{name}"}}
                   for name in ("a", "b")],
        "position_candidates": {f"x_{name}": {"quantities": quantities,
            "control": f"place_{name}"} for name in ("a", "b")},
        "focus_candidates": {f"f_{name}": {"quantities": [f"q_{name}"]}
            for name in ("a", "b")}}
    features = tuple(f"{role}_{name}" for name in ("a", "b") for role in ("q", "u", "x", "f"))
    return TaskSpec("focal_readout", "Observed response", features, (), (),
                    context={"object_model": model})


def _observations(seed, count, *, total=False):
    rng, rows = Random(seed), []
    for _ in range(count):
        point = {f"{role}_{name}": rng.uniform(-1., 1.)
                 for name in ("a", "b") for role in ("q", "u", "x")}
        focus = rng.choice(("a", "b"))
        point.update({f"f_{name}": float(name == focus) for name in ("a", "b")})
        target = (point["q_a"] + point["q_b"] + .4 * (point["u_a"] + point["u_b"]) if total else
                  1.2 * point[f"q_{focus}"] + .7 * point[f"u_{focus}"] + 2.1 * point[f"x_{focus}"])
        rows.append(Observation(point, target))
    return rows


def test_supported_law_defers_empty_assignment_then_common_mismatch_reopens_it():
    spec, initial_rows = _observed_problem(), _observations(71, 48)
    first_budget = SearchBudget(60000)
    initial = research_competing_object_programs(spec, initial_rows, first_budget, max_candidates=2)
    assert len(initial["competition"]["schemas"]) == 2
    assert len(initial["entries"]) == 1
    assert all(item["compute_cost"] > 0 for item in initial["competition"]["searches"])

    fresh = _observations(97, 24)
    budget = SearchBudget(30000)
    revised = research_competing_object_programs(spec, initial_rows + fresh, budget,
        previous=initial["competition"], new_observations=fresh, max_candidates=2)
    state = json.loads(json.dumps(revised["competition"]))
    deferred = [item for item in state["searches"]
                if item.get("search") == "deferred_while_supported_laws_remain"]
    assert len(revised["entries"]) == 1 and len(deferred) == 1
    assert deferred[0]["compute_cost"] == 0 and not deferred[0]["search_complete"]
    assert state["schemas"] == initial["competition"]["schemas"]
    assert not state["counterexamples"] and budget.spent < first_budget.spent / 4

    # A new question on the same observed objects changes the response. Every
    # old executable fails on its paid data, so the deferred assignment must
    # participate again and independently fit the new conserved response.
    next_spec = replace(spec, task_id="total_readout")
    current = _observations(809, 48, total=True)
    reopened = research_competing_object_programs(next_spec, current, SearchBudget(30000),
        previous=state, new_observations=current, max_candidates=2)
    assert reopened["competition"]["counterexamples"]
    assert len(reopened["entries"]) == 2
    assert all(item["compute_cost"] > 0 and "deferred" not in item.get("search", "")
               for item in reopened["competition"]["searches"])
    assert all(item["structure"]["fits"][0]["family"] == "conserved_total"
               and item["training_loss"] < 1e-7 for item in reopened["entries"])
    assert not spec.context["object_model"]["complete"]
