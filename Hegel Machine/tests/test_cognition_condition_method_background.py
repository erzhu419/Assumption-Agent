"""Measured context survives compact questions while current thresholds are solved."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.competing_object_experiments import (
    _forecast, assess_competing_object_probe, plan_competing_object_probe)
from hegel_machine.cognition.condition_questions import _question_program, plan_condition_questions
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.program_condition_sets import compound_predicate_regions, solve_compound_region
from hegel_machine.cognition.program_disagreement_probes import _experiment_sequence
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments
from test_cognition_conditional_relation_intervals import GATE, conditional_composition


def selectors():
    focus, c = Program("feature", name="focus"), lambda value: Program("const", value=value)
    return {"f0": Program("ge", children=(c(.5), focus)),
            "f1": Program("and", children=(Program("ge", children=(focus, c(.5))),
                                             Program("ge", children=(c(1.5), focus))))}


def actual_rows(sequence, *, prefix=""):
    """The physical focus chooses one object, unlike the imperfect saved map."""
    truth = conditional_composition(.630, "right")
    commands, measured, rows = [], [], []
    for point in sequence:
        fields = {name.removeprefix(prefix): value for name, value in point.items()}
        focus = round(fields["focus"])
        readings = {**fields, "f0": float(focus == 0), "f1": float(focus == 1)}
        target = truth.predict(Observation(readings, 0., tuple(measured)))
        rows.append(Observation(point, target, tuple(commands)))
        commands.append(point)
        measured.append(readings)
    return rows


def source_experiment():
    parent = bind_measurement_arguments(conditional_composition(.624, "right"), selectors())
    bounds = {"q0": (-1., 1.), "q1": (-1., 1.), "x0": (-2., 2.), "x1": (-2., 2.), "focus": (0., 3.)}
    spec = TaskSpec("source", "", tuple(bounds), (), (), feature_bounds=bounds)
    payload = {"selection_steps": [{"operation": "revise_relation_condition", "relation": "right",
        "predicate": GATE.to_dict(), "radius": .624, "radius_interval": [.616, .632]}]}
    context = {"q0": .8, "q1": .1, "x0": 0., "x1": 1.2, "focus": .8}
    paid = actual_rows([{**context, "x1": distance} for distance in (.1, 1.2) * 4])
    budget = SearchBudget(50000)
    research = RelationIntervalResearch(parent, [], budget, research_program=payload)
    entries = research.alternatives(paid, budget)
    question = research.questions(entries)[0]
    regions = compound_predicate_regions([_question_program(question), Program("const", value=0.)],
                                         spend=budget.spend)
    region = next(item for item in regions if len(item["constraints"]) == 2)
    solved, details = solve_compound_region(region, context, bounds, budget.spend)
    assert solved is not None
    sequence = _experiment_sequence(context, solved, 9, "cycle", history_depth=parent.history_depth)
    proposal = {"sequence_mode": "cycle", "boundary": {**details, "solver_strategy": "compound_conditions",
        "context": context, "constraints": [{**item, "expression": item["expression"].to_dict(),
            "measured_expression": item["measured_expression"].to_dict()} for item in region["constraints"]]}}
    plan = {"probe": Probe("paid_context", sequence), "proposal": proposal, "predicted_fields": {},
        "hypotheses": [{"hypothesis_id": item["candidate"].candidate_id,
            "candidate_id": item["candidate"].candidate_id, "program": item["candidate"].program.to_dict(),
            "predictions": _forecast(item["candidate"].program, sequence)} for item in entries]}
    observed = actual_rows(sequence)
    assessment = assess_competing_object_probe(plan, [(row, row) for row in observed], budget)
    assert assessment["status"] == "distinguished"
    methods = ProgramExperimentMethods()
    assert methods.observe(plan, assessment, task_id="source", command_spec=spec)
    research.alternatives([*paid, *observed], budget)
    assert research.intervals[0]["radius_interval"][0] > .616
    return spec, parent, payload, methods, research, [*paid, *observed], sequence


@pytest.mark.parametrize("renamed", [False, True])
def test_paid_background_and_current_radius_survive_compact_method_recall(renamed):
    spec, parent, payload, methods, research, paid, source = source_experiment()
    methods = ProgramExperimentMethods(json.loads(json.dumps(methods.to_dict())))
    saved = deepcopy(methods.records)
    prefix = "new_" if renamed else ""
    if renamed:
        mapping = {name: prefix + name for name in spec.feature_names}
        parent = parent.remap_features(mapping)
        spec = replace(spec, feature_names=tuple(mapping.values()),
                       feature_bounds={mapping[name]: bounds for name, bounds in spec.feature_bounds.items()})
        payload["selection_steps"][0]["radius_interval"] = research.intervals[0]["radius_interval"]
        research = RelationIntervalResearch(parent, [], SearchBudget(5000), research_program=payload)
        paid = [Observation({mapping[name]: value for name, value in row.features.items()}, row.target,
                  tuple({mapping[name]: value for name, value in point.items()} for point in row.history))
                for row in paid]
    budget = SearchBudget(50000)
    entries = research.alternatives(paid, budget)
    plan = plan_competing_object_probe(spec, entries, budget, serial=17, length=9,
        methods=methods, questions=research.questions(entries), task_id="narrower")
    assert plan["proposal_source"] == "learned_method" and plan["fresh_synthesis_skipped"]
    assert all(point[prefix + "focus"] == pytest.approx(.8) for point in plan["probe"].sequence)
    assert plan["probe"].sequence[1][prefix + "x1"] - plan["probe"].sequence[1][prefix + "x0"] != \
        pytest.approx(source[1]["x1"] - source[1]["x0"])
    assert methods.records == saved  # Only actual feedback may add support.
    acquired = actual_rows(plan["probe"].sequence, prefix=prefix)
    assessed = assess_competing_object_probe(plan, [(row, row) for row in acquired], budget)
    assert assessed["status"] == "distinguished"
    assert research.alternatives([*paid, *acquired], budget)
    assert any(low < .630 < high for low, high in research.parameters[0]["radius_intervals"])
    assert methods.observe(plan, assessed, task_id="narrower", command_spec=spec)["action"] == "use_program_experiment_method"
    assert len(methods.records[0]["support"]) == len(saved[0]["support"]) + 1
    assert plan["method_search_cost"] <= 4000 and plan["generation_cost"] <= 6000


def test_old_record_without_full_measured_context_does_not_invent_a_background():
    spec, _, _, methods, research, paid, _ = source_experiment()
    del methods.records[0]["source_boundary"]["context"]["focus"]
    original = deepcopy(methods.records)
    budget = SearchBudget(50000)
    entries = research.alternatives(paid, budget)
    plan = plan_condition_questions(spec, research.questions(entries), budget, methods=methods,
        response_programs=[item["candidate"].program for item in entries], length=9)
    assert plan["proposal_source"] == "learned_method"
    assert plan["proposal"]["boundary"]["context"]["focus"] == 1.5
    assert methods.records == original
