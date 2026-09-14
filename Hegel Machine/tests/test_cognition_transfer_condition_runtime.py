"""Intact memory must test rare callable conditions before short adoption."""

from dataclasses import replace
import json

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, Probe

from test_cognition_compound_unified_research import (
    CONFIG, FinitePriorResearcher, WindowLaboratory, hypotheses,
)


def _independent_check(lab, report, *, temporal=False):
    # Terminal observations come from an independent laboratory instance and
    # are never returned to research, method memory or fitting.
    evaluator = type(lab)(names=lab.names, gain=lab.gain)
    points = [(0., 0.), (.271234, .683214), (.271234, .683214),
              (.2712, .683214), (.271234, .68325), (.9, .9)]
    probe = Probe("independent_active_and_outside", tuple(dict(zip(lab.names, point)) for point in points))
    rows = evaluator.acquire(probe).observations
    assert any(row.target != 0. for row in rows)
    assert any(row.target == 0. for row in rows)
    predictor = Program.from_dict(report["program"])
    errors = [(predictor.predict(row) - row.target) ** 2 for row in rows]
    assert max(errors) <= 1e-7, (report["selected_kind"], errors, predictor.to_dict())
    if temporal:
        assert any(row.target < 0. for row in rows)
    return predictor


def _paid_condition_evidence(lab):
    return [rows for _, rows in lab.acquisitions if any(abs(row.target) > 1e-8 for row in rows)]


def _run(lab, library, seed, researcher=FinitePriorResearcher):
    report = researcher(replace(CONFIG, seed=seed), library=library).run(lab)
    assert report["status"] == "completed"
    assert sum(len(rows) for _, rows in lab.acquisitions) == report["costs"]["observation_cost"]
    # A completed rare-response task must have purchased its active condition;
    # three all-zero random batches cannot establish a constant response.
    assert _paid_condition_evidence(lab)
    assert Program.from_dict(report["program"]).op != "const"
    return report


def test_full_memory_preserves_response_across_restore_rename_change_and_return():
    library = KnowledgeLibrary()
    source = WindowLaboratory(gain=2.)
    first = _run(source, library, 137)
    source_predictor = _independent_check(source, first)
    source_concepts = {concept.concept_id: concept.program.to_dict() for concept in library.concepts}
    source_methods = {method["method_id"]: json.loads(json.dumps(method["support"]))
                      for method in library.program_experiment_methods.records}
    assert source_concepts and source_methods

    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    for seed, lab in (
            (139, WindowLaboratory(("renamed_a", "renamed_b"), gain=2.)),
            (149, WindowLaboratory(("renamed_a", "renamed_b"), gain=1.)),
            (151, WindowLaboratory(gain=2.))):
        report = _run(lab, restored, seed)
        _independent_check(lab, report)
        assert any(action.get("accepted") for action in report["actions"]
                   if action["action"] == "validate_frozen_program")
        # Keep the entire library, including old executable concepts and
        # experimental methods, when testing every successor.
        restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(restored.to_dict())))

    saved_concepts = {concept.concept_id: concept for concept in restored.concepts}
    for identifier, payload in source_concepts.items():
        assert saved_concepts[identifier].program.to_dict() == payload
        assert saved_concepts[identifier].status != "revoked"
    saved_methods = {method["method_id"]: method for method in restored.program_experiment_methods.records}
    for identifier, evidence in source_methods.items():
        assert saved_methods[identifier]["support"][:len(evidence)] == evidence
    active_source = Observation({"x": .271234, "y": .683214}, 2.)
    assert source_predictor.predict(active_source) == 2.


class EventWindowLaboratory(WindowLaboratory):
    def acquire(self, probe, channels=()):
        rows, history, previous_level = [], [], 0.
        for point in probe.sequence:
            inside = (abs(point[self.names[0]] - .271234) < 1e-5 and
                      abs(point[self.names[1]] - .683214) < 1e-5)
            level = self.gain if inside else 0.
            target = level - previous_level if history else 0.
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
            previous_level = level
        self.acquisitions.append((probe, tuple(rows)))
        return MeasurementBatch(rows, len(rows), 0, ())


class EventFinitePriorResearcher(FinitePriorResearcher):
    def _research_view(self, spec, training, validation, budget, *args, **kwargs):
        result = []
        for index, law in enumerate(hypotheses(spec.feature_names)):
            if not budget.spend(len(training) + len(validation)):
                break
            program = Program("event_delta", lag=1, children=(law,))
            result.append({"candidate": Candidate(f"event_prior_{index}", program, "finite_prior"),
                "kind": "finite_prior", "family_id": "T02",
                "training_loss": loss(program, training, 0., .5),
                "validation_loss": loss(program, validation, 0., .5)})
        return result


def test_full_memory_condition_experiment_preserves_history_under_changed_gain():
    library = KnowledgeLibrary()
    source = EventWindowLaboratory(gain=2.)
    first = _run(source, library, 157, researcher=EventFinitePriorResearcher)
    _independent_check(source, first, temporal=True)
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    successor = EventWindowLaboratory(("later_a", "later_b"), gain=1.)
    second = _run(successor, restored, 163, researcher=EventFinitePriorResearcher)
    _independent_check(successor, second, temporal=True)
    assert any(row.target != 0. and row.history for _, rows in successor.acquisitions for row in rows)
