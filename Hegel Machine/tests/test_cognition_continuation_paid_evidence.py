"""Internal experiments remain binding when a later random probe misses them."""

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.unified_research import compare_resumed_programs


def threshold(value):
    return Program("ge", children=(Program("feature", name="x"),
                                    Program("const", value=value)))


def compare(old, new, *, evidence=(), confirmation=None, budget=None,
            directed=False, external=False, refuted=False):
    confirmation = confirmation or [Observation({"x": -.8}, 0.),
                                    Observation({"x": .8}, 1.)]
    return compare_resumed_programs(old, new,
        before_loss=loss(old, confirmation, 0., .5),
        after_loss=loss(new, confirmation, 0., .5),
        incumbent_fallback=0., current_fallback=0.,
        directed_command_rows=evidence if directed else [],
        inherited_relation_evidence=[] if directed else evidence,
        feature_names=("x",), budget=budget if budget is not None else SearchBudget(100),
        minimum_improvement=1e-7, external_experiment=external,
        incumbent_refuted=refuted)


@pytest.mark.parametrize("directed", [False, True])
def test_internal_measured_counterexample_survives_random_confirmation_tie(directed):
    old, new = threshold(.5), threshold(.25)
    measured = [Observation({"x": .4}, 1.)]
    budget = SearchBudget(2)
    result = compare(old, new, evidence=measured, directed=directed, budget=budget)
    assert result["status"] == "compared"
    assert result["incumbent_refuted"] and not result["retain_incumbent"]
    kind = "program_experiment" if directed else "inherited_relation"
    assert result["checks"]["incumbent"][kind]["loss"] == 1.
    assert result["checks"]["current"][kind]["loss"] == 0.
    assert result["compute_cost"] == budget.spent == 2


def test_better_old_program_is_retained_when_paid_evidence_supports_it():
    old, new = threshold(.25), threshold(.5)
    rows = [Observation({"x": .4}, 1.)]
    result = compare(old, new, evidence=rows, confirmation=rows)
    assert not result["incumbent_refuted"] and result["retain_incumbent"]
    assert result["checks"]["current"]["inherited_relation"]["loss"] == 1.


def test_internal_tie_without_counterexample_keeps_the_supported_incumbent():
    result = compare(threshold(.5), threshold(.25), evidence=[Observation({"x": .8}, 1.)])
    assert result["retain_incumbent"] and not result["incumbent_refuted"]


def test_external_experiment_keeps_its_existing_tie_and_refutation_semantics():
    old, new = threshold(.5), threshold(.25)
    assert not compare(old, new, external=True)["retain_incumbent"]
    old_wins = [Observation({"x": .4}, 0.)]
    assert compare(old, new, confirmation=old_wins, external=True)["retain_incumbent"]
    assert not compare(old, new, confirmation=old_wins, external=True,
                       refuted=True)["retain_incumbent"]


def test_budget_shortfall_remains_unresolved_without_spending_partial_comparison():
    budget = SearchBudget(1)
    result = compare(threshold(.5), threshold(.25),
        evidence=[Observation({"x": .4}, 1.)], budget=budget)
    assert result["status"] == "compute_unresolved"
    assert result["retain_incumbent"] is None and result["checks"] == {}
    assert result["compute_required"] == 2
    assert result["compute_cost"] == budget.spent == 0


def test_two_failed_programs_do_not_favor_the_new_one():
    result = compare(threshold(.5), threshold(.25),
        evidence=[Observation({"x": .4}, .5)])
    assert result["retain_incumbent"] and not result["incumbent_refuted"]
    assert result["checks"]["current"]["inherited_relation"]["loss"] == .25


def test_different_readable_cohorts_do_not_supply_a_paired_replacement():
    old = threshold(.5)
    new = Program("feature", name="measured")
    rows = [Observation({"x": .4}, 1.), Observation({"measured": 1.}, 1.)]
    result = compare_resumed_programs(old, new, before_loss=0., after_loss=0.,
        incumbent_fallback=0., current_fallback=0., directed_command_rows=[],
        inherited_relation_evidence=rows, feature_names=("x",),
        budget=SearchBudget(2), minimum_improvement=1e-7)
    assert result["retain_incumbent"] and not result["incumbent_refuted"]
    assert result["checks"]["incumbent"]["inherited_relation"]["loss"] == 1.
    assert result["checks"]["current"]["inherited_relation"]["loss"] == 0.

