"""Partial measured response structure is reusable without a supported verdict."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.conditional_coordinates import _masked
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.observed_condition_duration import propose_observed_condition_duration
from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
from hegel_machine.cognition.research_drafts import (
    remember_research_draft, replay_numeric_research_drafts)
from hegel_machine.cognition.research_graph import ResearchGraph, bind_answers
from hegel_machine.cognition.research_programs import _Fitter, _entry
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _terms

from test_cognition_decoded_observed_conditions import DecodedObservedLab, random_probe
from test_cognition_research_drafts import numeric_procedure, numeric_rows


def broad_source():
    lab = DecodedObservedLab()
    spec = lab.spec_for(("meter",))
    rows = [row for seed in (4101, 4102) for row in
            lab.acquire(random_probe(lab.spec, seed, 64), ("meter",)).observations]
    budget = SearchBudget(100000)
    basis = [*_terms(spec.feature_names, quadratic=True), Program("const", value=1.)]
    gate = Program("feature", name="meter")
    condition = Program("ge", children=(gate, Program("const", value=.173)))
    fit = _Fitter(rows, budget, .5).fit([
        _masked(term, condition, side, 3, "inside_window")
        for term in basis for side in (True, False)])
    assert fit["loss"] < 1e-15
    readings = sorted({row.features["meter"] for row in rows})
    interval = [max(value for value in readings if value < .173),
                min(value for value in readings if value >= .173)]
    step = {"operation": "discover_conditional_coordinates", "question": "Paid branch response",
        "program": fit["program"], "loss": fit["loss"], "gate_coordinate": gate.to_dict(),
        "threshold": .173, "threshold_interval": interval, "width": 3,
        "placement": "inside_window", "branch_basis": [term.to_dict() for term in basis[:-1]],
        "parameter_terms": [term.to_dict() for term in fit["fit_terms"]]}
    selected = _entry(fit, [step], len(rows), budget.spent)
    selected["condition_uncertainty"] = assess_observed_condition_uncertainty(
        spec, rows, selected, budget, tolerance=1e-7)
    assert selected["condition_uncertainty"]["status"] == "unresolved"
    library = KnowledgeLibrary()
    record = remember_research_draft(library, selected, spec, [])
    return library, record


@pytest.mark.parametrize("placement", ["inside_window", "outside_window"])
def test_broad_old_draft_uses_paid_structure_and_refits_new_original_response(placement):
    library, record = broad_source()
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    before = deepcopy(library.to_dict())
    lab = DecodedObservedLab(seed=4201, width=5, placement=placement,
        names=("u", "v", "w"), meter="reading", threshold=-.271,
        control_scale=2.3, meter_scale=1.7, response_scale=1.25)
    spec = lab.spec_for(("reading",))
    training = [row for seed in (4301, 4302) for row in
        lab.acquire(random_probe(lab.spec, seed, 64), ("reading",)).observations]
    fixed = Probe("paid_fixed", tuple({name: .3 for name in lab.spec.feature_names}
                                     for _ in range(24)))
    experiment = lab.acquire(fixed, ("reading",)).observations
    hints = propose_observed_condition_duration(spec, experiment, SearchBudget(40000))
    evidence = [item["observed_condition_evidence"] for item in hints]
    assert evidence and min(item["conditional_loss"] for item in evidence) < 1e-15
    validation = lab.acquire(random_probe(lab.spec, 4401, 64), ("reading",)).observations
    budget, actions = SearchBudget(80000), []
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7, max_candidates=8),
        library=library)
    claims = replay_numeric_research_drafts(agent, spec, [*training, *experiment],
        budget, actions, evidence=evidence, max_candidates=4)
    assert claims
    checked = agent._assess(claims, validation, SearchBudget(5000), 0., .5)
    chosen = agent._best(checked)
    assert chosen["training_loss"] < 1e-12
    assert chosen["validation_loss"] < 1e-12
    assert chosen["research_draft_id"] == record["draft_id"]
    steps = chosen["research_program"]["selection_steps"]
    current = [step for step in steps if not step.get("historical")]
    assert any(step.get("width") == 5 and step.get("placement") == placement for step in current)
    assert all(step.get("gate_coordinate", {}).get("name", "reading") == "reading"
               for step in current)
    graph = ResearchGraph.from_dict(chosen["research_program"]["graph"])
    node = graph.nodes[chosen["research_program"]["output_node"]]
    rebuilt = bind_answers(node.expression, {key: graph.nodes[key].answer for key in node.depends_on})
    assert loss(rebuilt, validation, 0., .5) == pytest.approx(chosen["validation_loss"])
    assert actions and all(item["response_basis_terms"] < 8 for item in actions)
    assert all(item["comparison_basis"] == "original_response" for item in actions)
    assert 0 < budget.spent <= 80000
    assert sum(item["compute_cost"] for item in actions) == budget.spent
    assert library.to_dict() == before and not library.concepts
    assert chosen["research_program"]["status"] == "requires_fresh_validation"


def test_old_good_explanation_survives_limited_compute_and_no_supported_adoption():
    selected, spec, _ = numeric_procedure()
    library = KnowledgeLibrary()
    record = remember_research_draft(library, selected, spec, [])
    # A new instrument gain is still the same useful branch structure. With no
    # extra experiment allowance, its current refit must stay available.
    rows = numeric_rows(response_scale=1.8)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7), library=library)
    budget, actions = SearchBudget(1500), []
    claims = replay_numeric_research_drafts(agent, replace(spec, task_id="changed_gain"),
        rows, budget, actions)
    assert claims and min(item["training_loss"] for item in claims) < 1e-15
    assert all(item["research_draft_id"] == record["draft_id"] for item in claims)
    assert all(item["research_program"]["status"] == "requires_fresh_validation" for item in claims)
    assert not library.concepts and library.research_drafts[0]["status"] == "unresolved"
    assert 0 < budget.spent <= 1500
    chosen = min(claims, key=lambda item: item["training_loss"])
    state = assess_observed_condition_uncertainty(spec, rows, chosen, SearchBudget(10000),
        tolerance=1e-7)
    # The source gap was [.1, .35]; current evidence has the same readings here.
    assert state["conditions"][0]["threshold_interval"] == [.1, .35]


def test_retained_fit_replaces_the_source_gap_with_current_paid_reading_support():
    from test_cognition_observed_condition_uncertainty import observations

    selected, spec, _ = numeric_procedure()
    library = KnowledgeLibrary()
    remember_research_draft(library, selected, spec, [])
    # The response's cut is unchanged, but the new meter's sample gap is much
    # wider. Reusing [.1, .35] would understate its unresolved research question.
    rows = observations(selected["candidate"].program, [-1., 1., -.8, .8] * 10)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7), library=library)
    claims = replay_numeric_research_drafts(agent, spec, rows, SearchBudget(2000), [])
    chosen = min(claims, key=lambda item: item["training_loss"])
    assert chosen["training_loss"] < 1e-15
    state = assess_observed_condition_uncertainty(spec, rows, chosen, SearchBudget(10000),
        tolerance=1e-7)
    assert state["conditions"][0]["threshold_intervals"] == [[-.8, .8]]
    assert state["status"] == "unresolved" and state["expected_prediction_ambiguity"] > 1e-7


def test_repeated_newer_drafts_do_not_displace_the_compact_source_family():
    library, first = broad_source()
    repeated = deepcopy(first)
    repeated.update(draft_id="research_draft_0002", source_task_id="later_bad_boundary")
    library.research_drafts.append(repeated)
    lab = DecodedObservedLab(seed=4601)
    rows = lab.acquire(random_probe(lab.spec, 4602, 64), ("meter",)).observations
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7), library=library)
    budget, actions = SearchBudget(2000), []
    claims = replay_numeric_research_drafts(agent, lab.spec_for(("meter",)), rows, budget, actions)
    assert claims and all(item["research_draft_id"] == first["draft_id"] for item in claims)
    assert len(actions) == 1 and actions[0]["response_basis_terms"] == 3
    assert actions[0]["observed_condition_fields"] == ["meter"]


def test_fused_decoded_response_refits_branch_laws_and_reset_offsets_with_small_budget():
    from hegel_machine.cognition.decoded_conditional_coordinates import _forward_fit
    from hegel_machine.cognition.models import Observation
    from test_cognition_observed_condition_uncertainty import observations

    selected, spec, original = numeric_procedure(decoded=True)
    gate = Program("ge", children=(Program("feature", name="meter"),
                                   Program("const", value=.225)))
    old_inner = Program("if", children=(gate,
        Program("add", children=(Program("const", value=1.), Program("feature", name="x"))),
        Program("add", children=(Program("const", value=-.4), Program("feature", name="meter")))))
    old = Program("window_sum", lag=3, children=(old_inner,))
    old_rows = [Observation(row.features, old.predict(row), row.history) for row in original]
    fit = _forward_fit(old_inner, 3, old_rows, SearchBudget(5000), .5)
    steps = deepcopy(selected["research_program"]["selection_steps"])
    steps[0]["program"] = fit["program"]
    selected = _entry(fit, steps, len(old_rows), 0)
    selected["condition_uncertainty"] = {"status": "unresolved"}
    library = KnowledgeLibrary()
    record = remember_research_draft(library, selected, spec, [])
    # Each branch changes its coefficients and its per-step offset independently.
    revised_inner = Program("if", children=(gate,
        Program("add", children=(Program("const", value=2.),
            Program("mul", children=(Program("const", value=.4), Program("feature", name="x"))))),
        Program("add", children=(Program("const", value=-1.1),
            Program("mul", children=(Program("const", value=1.8), Program("feature", name="meter")))))))
    current = Program("window_sum", lag=3, children=(revised_inner,))
    rows = observations(current, [-.9, .35, -.4, .85, -.15, .6, -.65, .1] * 6)
    future = observations(current, [.8, -.8, .4, -.4, .6, -.6, .35, -.15] * 4)
    budget, actions = SearchBudget(1500), []
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7), library=library)
    claims = replay_numeric_research_drafts(agent, spec, rows, budget, actions)
    assert claims and claims[0]["research_draft_id"] == record["draft_id"]
    assert claims[0]["training_loss"] < 1e-15
    assert loss(claims[0]["candidate"].program, future, 0., .5) < 1e-15
    assert 0 < budget.spent <= 1500
