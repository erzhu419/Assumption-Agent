"""A response counterexample is diagnosed before it can shrink a relation prior."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.competing_object_experiments import _forecast, assess_competing_object_probe
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_research_actions import ObjectResearchActions
from hegel_machine.cognition.relation_input_diagnosis import diagnose_relation_inputs
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.research_programs import _entry
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_structure import _research_response_terms
from test_cognition_instrument_relation_diagnosis import (
    SETTINGS, acquire as measured_pairs, binding_error, law, legacy_source_model,
)
from hegel_machine.cognition.instrument_models import instrument_binding


def joint_probe(name, focus=1.5):
    point = {"q_o0": .2, "q_o1": .7, "q_o2": .5, "q_o3": 0.,
             "x_o0": 0., "x_o1": 1., "x_o2": 2., "x_o3": -2.,
             "u_o0": 0., "u_o1": 0., "u_o2": 0., "u_o3": 0., "focus": focus}
    return Probe(name, tuple({**point, "x_o1": .3 if active else 1., "x_o2": .6 if active else 2.}
                            for active in (False, True, True, True, True, False, False, False, False)))


class PaidInputs:
    def __init__(self, lab, batches=(), limit=512):
        self.lab, self.batches, self.purposes = lab, list(batches), []
        self.rows, self.limit = 0, limit

    def acquire(self, channels, purpose, *, probe=None):
        probe = probe or self.lab.random_probe(885 + len(self.purposes), length=24)
        if self.rows + probe.cost > self.limit:
            return None
        self.purposes.append(purpose)
        batch = self.lab.acquire(probe, channels)
        history, pairs = [], []
        for point, measured in zip(probe.sequence, batch.observations):
            pairs.append((Observation(point, measured.target, tuple(history)), measured))
            history.append({**point, "__target__": measured.target})
        self.rows += batch.observation_cost
        self.batches.append(pairs)
        return batch.observations

    def paired_for(self, channels):
        assert channels == ("fine",)
        return self.batches


def setup_source(source):
    spec = source.spec_for(("fine",))
    random, _ = measured_pairs(source, source.random_probe(183, length=48))
    active, batch = measured_pairs(source, joint_probe("source_law", focus=1.))
    before, after = batch.transitions[1]["before"], batch.transitions[1]["after"]
    # Two active outgoing edges; only paid transition values determine gain.
    coupling = (before["measured_q_o0"] - after["measured_q_o0"]) / (2 * before["measured_q_o0"])
    program = law(spec, coupling)
    assert max(abs(program.predict(row) - row.target) for _, row in active) < 1e-12
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7),
                                 library=KnowledgeLibrary())
    agent._learn_instruments = agent._revise_inputs = True
    budget = SearchBudget(120000)
    agent._fit_instruments(source.spec, spec, [random, active], budget, [])
    # Restore a genuinely paid pre-0.91 source: it retained one selected mapping
    # that passed these source readings, before the unseen selector boundary.
    old = legacy_source_model(agent.library.instrument_models[-1])
    assert set(instrument_binding(old)) == set(spec.feature_names)
    assert binding_error(instrument_binding(old), active) < 1e-7
    old["status"] = "supported_on_measured_trajectory"
    agent.library.instrument_models[-1] = old
    concept, _ = agent.library.learn(Candidate("measured_law", program, "law"), spec,
                                     [row for _, row in [*random, *active]])
    def entry(program):
        predicate = Program("lt", children=(Program("arg", name="edge_source_quantity"),
                                             Program("arg", name="edge_target_quantity")))
        fit = {"program": program, "terms": [program], "fit_terms": [program],
               "coefficients": [0., 1.], "loss": 0.}
        return _entry(fit, [{"operation": "revise_relation_condition", "question": "Retest the local relation",
            "program": program, "loss": 0., "relation": "right", "predicate": predicate.to_dict(),
            "radius": .8, "radius_interval": [.35, 1.2]}], len(random) + len(active), 0)
    agent.library.meta_priors.append({"source_concept_id": concept, "uses": 0, "counterexamples": [],
        "source_features": list(spec.feature_names), "control_features": list(source.spec.feature_bounds),
        "object_schema": json.loads(spec.context["object_schema"]), "research_program": entry(program)["research_program"]})
    call = next(call for call in agent._recall(source.spec)
                if max(abs(call.predict(row) - row.target) for row, _ in active) < 1e-8)
    return agent, entry(call), [random, active]


def command_evidence(lab, entry, probe, rivals=()):
    batch = lab.acquire(probe, ())
    plan = {"probe": probe, "predicted_fields": {}, "hypotheses": []}
    for index, program in enumerate((entry["candidate"].program, *rivals)):
        identifier = entry["candidate"].candidate_id if index == 0 else f"rival:{index}"
        plan["hypotheses"].append({"hypothesis_id": identifier, "candidate_id": identifier,
            "program": program.to_dict(), "predictions": _forecast(program, probe.sequence)})
    evidence = assess_competing_object_probe(plan, [(row, row) for row in batch.observations], SearchBudget(1000))
    return plan, evidence


def change_radius(program, radius):
    if program.op == "call":
        return replace(program, definition=replace(program.definition, body=change_radius(program.definition.body, radius)))
    children = tuple(change_radius(child, radius) for child in program.children)
    if program.op == "relation_sum":
        children = (Program("const", value=radius), *children[1:])
    return replace(program, children=children)


@pytest.mark.parametrize("condition", ["source_less", "source_greater"])
def test_same_trajectory_diagnosis_restores_initial_prior_and_keeps_true_relation_failure(condition):
    with UnifiedObjectLab(**SETTINGS, relation_condition="source_less") as source, \
         UnifiedObjectLab(**SETTINGS, relation_condition=condition) as target:
        agent, origin, batches = setup_source(source)
        initial = deepcopy(origin)
        definitions = deepcopy(agent.library.definitions)
        selected = deepcopy(origin)
        selected["research_program"]["selection_steps"][-1]["radius_interval"] = [.35, .6]  # Inference made with the old input map.
        probe = joint_probe("paid_parameter_counterexample")
        rival = change_radius(selected["candidate"].program, .4)
        plan, evidence = command_evidence(target, selected, probe, (rival,))
        assert evidence["hypotheses"][0]["status"] == "counterexample"
        if condition == "source_less":
            # A wrong radius masks the wrong selector, so all_refuted is false.
            assert evidence["status"] == "distinguished"
        paid = PaidInputs(target, batches)
        observations_before = target.costs["observation_cost"]
        budget, actions = SearchBudget(60000), []
        result = diagnose_relation_inputs(agent, target, selected, origin, plan, evidence,
            paid.acquire, paid.paired_for, budget, actions)
        assert result["status"] == "inputs_revised"
        assert result["evidence"]["calibration_status"] == "counterexample"
        assert result["entry"]["research_program"]["selection_steps"][-1]["radius_interval"] == [.35, 1.2]
        assert origin == initial and agent.library.definitions == definitions
        # Additional actual readings resolve the revised candidate pool.
        assert len(result["command_rows"]) == paid.rows == sum(
            len(batch) for batch in paid.batches[len(batches):])
        assert paid.rows == target.costs["observation_cost"] - observations_before
        assert paid.purposes == ["remeasure_relation_counterexample", "resolve_instrument_conditions",
                                 "compare_relation_input_mapping", "distinguish_instrument_hypotheses"]
        revised = result["entry"]["candidate"].program
        same_trace = result["command_rows"][:9]
        error = max(abs(revised.predict(row) - row.target) for row in same_trace)
        measured_loss = result["evidence"]["hypotheses"][0]["measured_loss"]
        assert result["response_loss"] == pytest.approx(
            loss(revised, result["command_rows"], 0., target.spec.risk_quantile))
        if condition == "source_less":
            assert error < 1e-8 and measured_loss < 1e-12
            assert result["response_status"] == "supported_on_paid_rows"
            assert result["response_loss"] < 1e-7
            # Other learned membership endpoints were not in the initial
            # counterexample. Their actual full responses must also recover.
            for focus in (.5, 2.5):
                fresh = target.acquire(joint_probe(f"new_endpoint:{focus}", focus=focus), ())
                assert max(abs(revised.predict(row) - row.target) for row in fresh.observations) < 1e-8
        else:
            assert error > .01 and measured_loss > 1e-5
            assert result["response_status"] == "counterexample"
            assert result["response_loss"] > 1e-7
        restored = RelationIntervalResearch(revised, agent.library.meta_priors, budget,
                                           research_program=result["entry"]["research_program"])
        assert restored.intervals[0]["radius_interval"] == [.35, 1.2]
        assert 0 < budget.spent <= budget.limit


def test_correct_measured_inputs_do_not_refit_or_buy_random_validation_for_a_changed_relation():
    with UnifiedObjectLab(**SETTINGS, relation_condition="source_less") as source, \
         UnifiedObjectLab(**SETTINGS, relation_condition="source_greater") as target:
        agent, origin, batches = setup_source(source)
        models = deepcopy(agent.library.instrument_models)
        plan, evidence = command_evidence(target, origin, joint_probe("relation_only", focus=1.))
        paid = PaidInputs(target, batches)
        result = diagnose_relation_inputs(agent, target, origin, origin, plan, evidence,
            paid.acquire, paid.paired_for, SearchBudget(60000), [])
        assert result["status"] == "inputs_supported" and result["entry"] is None
        assert result["evidence"]["hypotheses"][0]["measured_loss"] > 1e-5
        assert agent.library.instrument_models == models
        assert paid.purposes == ["remeasure_relation_counterexample"]
        assert len(result["command_rows"]) == paid.rows == 9


def test_structure_research_preserves_rebound_response_when_projecting_a_bare_command_parent(monkeypatch):
    with UnifiedObjectLab(**SETTINGS, relation_condition="source_less") as lab:
        agent, origin, batches = setup_source(lab)
        definitions = deepcopy(agent.library.definitions)
        plan, evidence = command_evidence(lab, origin, joint_probe("rebind_before_projection"))
        paid, budget, actions = PaidInputs(lab, batches), SearchBudget(60000), []
        diagnosis = diagnose_relation_inputs(agent, lab, origin, origin, plan, evidence,
            paid.acquire, paid.paired_for, budget, actions)
        assert diagnosis["response_status"] == "supported_on_paid_rows"
        # A retained callable can arrive without a recorded research procedure.
        parent = {"candidate": diagnosis["entry"]["candidate"]}
        before = deepcopy(parent)
        received = []

        def capture_parent(spec, training, validation, *args, parent=None, **kwargs):
            received.append(parent)
            return []

        monkeypatch.setattr(agent, "_research_view", capture_parent)
        controller = ObjectResearchActions(agent, lab, acquire=paid.acquire,
            paired_for=paid.paired_for,
            command_batches=lambda: [[command for command, _ in batch] for batch in paid.batches],
            view_rows=None, transitions_for=lambda channels, rows: (), budget=budget,
            actions=actions, transfer=[], graph=None,
            remaining=lambda: {"observation_remaining": 0, "measurement_remaining": 0,
                               "compute_remaining": budget.remaining}, length=24)
        controller._structure_view(("fine",), parent, ())
        projected, = received
        assert projected is not None and "research_program" not in projected
        program = projected["candidate"].program
        command_program = parent["candidate"].program
        assert program.definition == command_program.definition
        assert program.definition.source_concept_id is not None
        current_pairs = [pair for batch in paid.batches[len(batches):] for pair in batch]
        assert len(current_pairs) == paid.rows > 0
        for command, measured in current_pairs:
            assert program.predict(measured) == pytest.approx(command_program.predict(command), abs=1e-8)

        response = _research_response_terms(projected, budget)
        assert len(response) == 1
        schema = json.loads(lab.spec_for(("fine",)).context["object_schema"])
        total_fields = [fields[role] for fields in schema["fields"].values() for role in ("q", "u")]
        for _, measured in current_pairs:
            expected = sum(point[name] for point in [*measured.history[-2:], measured.features]
                           for name in total_fields)
            assert response[0].predict(measured) == pytest.approx(expected, abs=1e-8)
        assert parent == before and agent.library.definitions == definitions


def test_unfinished_diagnosis_preserves_paid_counterexample_without_claiming_a_revised_model():
    with UnifiedObjectLab(**SETTINGS, relation_condition="source_less") as lab:
        agent, origin, batches = setup_source(lab)
        models = deepcopy(agent.library.instrument_models)
        plan, evidence = command_evidence(lab, origin, joint_probe("bounded_counterexample"))
        paid, actions = PaidInputs(lab, batches, limit=9), []
        result = diagnose_relation_inputs(agent, lab, origin, origin, plan, evidence,
            paid.acquire, paid.paired_for, SearchBudget(60000), actions)
        assert result["status"] == "unresolved_inputs" and result["entry"] is None
        assert result["response_status"] == "unassessed" and result["response_loss"] is None
        assert len(result["command_rows"]) == paid.rows == 9
        assert result["evidence"]["calibration_status"] == "counterexample"
        assert agent.library.instrument_models == models
        assert actions[-1]["status"] == "unresolved_inputs"


@pytest.mark.parametrize("focus,selected_radius", [(2., .4), (1.5, .8)])
def test_supported_radius_only_buys_inputs_when_the_probe_touches_an_unmeasured_learned_cut(focus, selected_radius):
    with UnifiedObjectLab(**SETTINGS, relation_condition="source_less") as lab:
        agent, origin, batches = setup_source(lab)
        selected = deepcopy(origin)
        selected["candidate"] = replace(origin["candidate"],
            program=change_radius(origin["candidate"].program, selected_radius))
        competing_radius = .8 if selected_radius == .4 else .4
        plan, evidence = command_evidence(lab, selected, joint_probe("radius_selection", focus=focus),
            (change_radius(origin["candidate"].program, competing_radius),))
        assert evidence["status"] == "distinguished"
        assert evidence["hypotheses"][0]["status"] == "counterexample"
        before = deepcopy(agent.library.instrument_models)
        paid, actions = PaidInputs(lab, batches), []
        observations_before = lab.costs["observation_cost"]
        result = diagnose_relation_inputs(agent, lab, selected, origin, plan, evidence,
            paid.acquire, paid.paired_for, SearchBudget(60000), actions)
        if focus == 2.:
            assert result is None
            assert paid.rows == 0 and paid.purposes == []
            assert agent.library.instrument_models == before
        else:
            assert result["status"] == "inputs_revised"
            assert len(result["command_rows"]) == paid.rows == sum(
                len(batch) for batch in paid.batches[len(batches):])
            assert paid.rows == lab.costs["observation_cost"] - observations_before
            assert max(abs(result["entry"]["candidate"].program.predict(row) - row.target)
                       for row in result["command_rows"][:9]) < 1e-8
