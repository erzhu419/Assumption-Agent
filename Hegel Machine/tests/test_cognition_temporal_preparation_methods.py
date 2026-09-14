"""Measured methods can revise preparation while retaining independent evidence."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.program_disagreement_probes import propose_program_disagreement_probes
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods

from test_cognition_program_experiment_methods import _plan, _measure
from test_cognition_temporal_experiment_methods import binary, const, field, previous, window


def case(center=.21, *, amplitude=1., nonlinear=False):
    bounds = dict.fromkeys(("u", "v", "w"), (-1., 1.))
    spec = TaskSpec("prepared_product", "Public control trajectories", tuple(bounds), (), (),
                    feature_bounds=bounds, max_probe_length=24)
    current = binary("mul", field("v"), field("v")) if nonlinear else field("v")
    reading = binary("add", binary("mul", current, previous("u", 1)),
                     binary("mul", const(.2), field("w")))
    predicate = window(field("reading"), center)
    return spec, [binary("mul", const(amplitude), predicate),
                  binary("mul", const(2. * amplitude), predicate)], {"reading": reading}


def learned(*, nonlinear=False):
    spec, programs, binding = case(nonlinear=nonlinear)
    proposals = propose_program_disagreement_probes(spec, programs, SearchBudget(6000), binding=binding)
    proposal = next(item for item in proposals if item["sequence_mode"] == "prefix" and
                    item["boundary"]["temporal_recipe"]["coordinate_dependency"] == 0)
    plan = _plan(proposal, programs, binding)
    actual = _measure(plan, programs[-1], binding)
    assert actual["status"] == "distinguished"
    methods = ProgramExperimentMethods()
    methods.observe(plan, actual, task_id="source", command_spec=spec, binding=binding)
    return methods


def pay(methods, proposal, spec, programs, binding, *, truth=None, task_id="successor"):
    plan = _plan(proposal, programs, binding)
    actual = _measure(plan, programs[-1] if truth is None else truth, binding)
    action = methods.observe(plan, actual, task_id=task_id, command_spec=spec, binding=binding)
    return actual, action


def test_local_preparation_revision_preserves_other_current_control_and_replays_after_payment(monkeypatch):
    methods = learned()
    parent = deepcopy(methods.records[0])
    spec, programs, binding = case(.9)

    def no_fresh(*args, **kwargs):
        raise AssertionError("preparation revision restarted temporal generation")

    monkeypatch.setattr("hegel_machine.cognition.temporal_control_probes.propose_temporal_control_probes", no_fresh)
    budget = SearchBudget(6000)
    proposal = methods.propose(spec, programs, budget, binding=binding, task_id="successor")[0]
    assert 0 < budget.spent <= 4000 and proposal["compute_cost"] == budget.spent
    assert proposal["parent_method_id"] == parent["method_id"] and "method_id" not in proposal
    assert methods.records == [parent]
    recipe = parent["temporal_recipe"]
    roles = recipe["source_control_roles"]
    context = {role: [2. * value - 1. for value in fractions]
               for role, fractions in zip(roles, recipe["context_by_role"])}
    # Only earlier u prepares the product. Current w is a separate measured
    # contribution and must not be silently changed by the joint solve.
    assert proposal["probe"].sequence[0]["u"] != context["u"][0]
    for step in range(2):
        assert proposal["probe"].sequence[step]["w"] == pytest.approx(context["w"][step])
    changed = proposal["preparation_revision"]["changed_steps"]
    assert [(roles[item["control_role"]], item["lag"]) for item in changed] == [("u", 1)]
    assert methods.observe(_plan(proposal, programs, binding),
        {"status": "no_observations", "observation_rows": 0}, task_id="unpaid",
        command_spec=spec, binding=binding) is None
    assert methods.records == [parent]
    actual, action = pay(methods, proposal, spec, programs, binding)
    assert actual["status"] == "distinguished" and action["parent_method_id"] == parent["method_id"]
    child = deepcopy(methods.records[1])
    assert methods.records[0] == parent and len(child["support"]) == 1
    methods = ProgramExperimentMethods(json.loads(json.dumps(methods.to_dict())))
    spec, programs, binding = case(.88)
    replay = methods.propose(spec, programs, SearchBudget(4000), binding=binding, task_id="later")[0]
    assert replay["method_id"] == child["method_id"] and "parent_method_id" not in replay
    assert pay(methods, replay, spec, programs, binding, task_id="later")[0]["status"] == "distinguished"
    assert methods.records[0] == parent and len(methods.records[1]["support"]) == 2


def test_real_unresolved_revision_records_a_negative_child_and_limits_only_its_condition():
    methods = learned()
    parent = deepcopy(methods.records[0])
    # Their forecasts separate, but an actual midpoint leaves both nearby
    # response hypotheses within the measurement tolerance.
    spec, programs, binding = case(.9, amplitude=.001)
    proposal = methods.propose(spec, programs, SearchBudget(4000), binding=binding, task_id="ambiguous")[0]
    midpoint = binary("mul", const(.0015), window(field("reading"), .9))
    actual, action = pay(methods, proposal, spec, programs, binding, truth=midpoint, task_id="ambiguous")
    assert actual["status"] == "unresolved" and actual["calibration_status"] == "supported_on_probe"
    assert action["reason"] == "unresolved_under_current_condition"
    child = methods.records[1]
    assert child["parent_method_id"] == parent["method_id"]
    assert child["status"] == "counterexample" and child["support"] == []
    assert len(child["counterexamples"]) == 1 and methods.records[0] == parent
    assert methods.propose(spec, programs, SearchBudget(4000), binding=binding, task_id="ambiguous") == []
    assert methods.last_proposal["status"] == "blocked_condition"
    assert methods.last_proposal["blocked_scopes"][-1]["method_id"] == child["method_id"]
    assert len(methods.records) == 2 and methods.records[0] == parent
    retry = methods.propose(spec, programs, SearchBudget(4000), binding=binding, task_id="new_task")[0]
    assert retry["parent_method_id"] == parent["method_id"]
    assert pay(methods, retry, spec, programs, binding, task_id="new_task")[0]["status"] == "distinguished"


def test_each_existing_supported_recipe_is_tried_before_revising_a_failed_one(monkeypatch):
    methods = learned()
    spec, programs, binding = case(.9)
    proposal = methods.propose(spec, programs, SearchBudget(4000), binding=binding)[0]
    pay(methods, proposal, spec, programs, binding)
    parent, child = methods.to_dict()
    # Retrieval visits the failing parent first, then the supported child.
    methods = ProgramExperimentMethods([child, parent])

    def no_revision(*args, **kwargs):
        raise AssertionError("a stored usable method was bypassed by revision")

    monkeypatch.setattr("hegel_machine.cognition.temporal_experiment_methods.revise_temporal_preparation", no_revision)
    proposal = methods.propose(spec, programs, SearchBudget(4000), binding=binding)[0]
    assert proposal["method_id"] == child["method_id"]


def test_revised_preparation_measures_the_retained_nonlinear_coordinate_operation():
    methods = learned(nonlinear=True)
    spec, programs, binding = case(.9, nonlinear=True)
    proposal = methods.propose(spec, programs, SearchBudget(4000), binding=binding)[0]
    # Joint interpolation prepares a positive u and initially positive v.
    # The stored coordinate procedure selects the negative quadratic root;
    # that actual operation must run before its experiment acquires support.
    assert proposal["preparation_revision"]["changed_steps"][0]["value"] > 0.
    assert proposal["probe"].sequence[1]["v"] < 0.
    assert proposal["boundary"]["solve"]["solver_strategy"] == "coordinate"
    assert pay(methods, proposal, spec, programs, binding)[0]["status"] == "distinguished"
    replay = methods.propose(spec, programs, SearchBudget(4000), binding=binding)[0]
    assert replay["method_id"] == methods.records[1]["method_id"]
    assert replay["probe"].sequence == proposal["probe"].sequence


@pytest.mark.parametrize("allowance", (0, 100))
def test_revision_cannot_reset_an_exhausted_reuse_budget(allowance):
    methods = learned()
    records = methods.to_dict()
    spec, programs, binding = case(.9)
    budget = SearchBudget(allowance)
    assert methods.propose(spec, programs, budget, binding=binding) == []
    assert budget.spent <= allowance and methods.records == records
