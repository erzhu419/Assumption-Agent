"""Finite competing programs must drive real purchases in the unified loop."""

from dataclasses import replace
import json

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def hypotheses(names):
    def window(name, center):
        field = Program("feature", name=name)
        return Program("and", children=(
            Program("ge", children=(field, Program("const", value=center - 1e-5))),
            Program("lt", children=(field, Program("const", value=center + 1e-5)))))
    gate = Program("and", children=(window(names[0], .271234), window(names[1], .683214)))
    return [Program("if", children=(gate, Program("const", value=gain), Program("const", value=0.)))
            for gain in (1., 2.)]


class WindowLaboratory:
    channels = ()

    def __init__(self, names=("x", "y"), gain=2.):
        self.names, self.gain, self.acquisitions = names, gain, []
        self.spec = TaskSpec("window_response", "Predict the response of public controls.",
            names, (), (), feature_bounds={name: (0., 1.) for name in names})

    def spec_for(self, channels=()):
        return self.spec

    def acquire(self, probe, channels=()):
        rows, history = [], []
        for point in probe.sequence:
            inside = (abs(point[self.names[0]] - .271234) < 1e-5 and
                      abs(point[self.names[1]] - .683214) < 1e-5)
            target = self.gain if inside else 0.
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        self.acquisitions.append((probe, tuple(rows)))
        return MeasurementBatch(rows, len(rows), 0, ())


class FinitePriorResearcher(UnifiedResearchAgent):
    """Supply two prior programs; only paid data chooses between their gains."""

    def _research_view(self, spec, training, validation, budget, *args, **kwargs):
        claims = []
        for index, program in enumerate(hypotheses(spec.feature_names)):
            if not budget.spend(len(training) + len(validation)):
                break
            claims.append({"candidate": Candidate(f"prior_{index}", program, "finite_prior"),
                "kind": "finite_prior", "family_id": "T02",
                "training_loss": loss(program, training, 0., .5),
                "validation_loss": loss(program, validation, 0., .5)})
        return claims


CONFIG = ResearchConfig(seed=137, observation_budget=192, compute_budget=120000,
                        minimum_improvement=1e-7, active_probes=True)


def test_unified_purchases_compound_evidence_refits_and_reuses_method_after_restore():
    library = KnowledgeLibrary()
    lab = WindowLaboratory()
    report = FinitePriorResearcher(CONFIG, library=library).run(lab)
    tests = [action for action in report["actions"] if action["action"] == "test_competing_program_predictions"]
    assert tests and tests[0]["evidence"]["status"] == "distinguished"
    assert tests[0]["plan"]["proposal"]["boundary"]["solver_strategy"] == "compound_conditions"
    assert any(action["action"] == "revise_from_program_experiment" for action in report["actions"])
    assert report["status"] == "completed"
    assert Program.from_dict(report["program"]).predict(Observation({"x": .271234, "y": .683214}, 2.)) == 2.
    assert sum(len(rows) for _, rows in lab.acquisitions) == report["costs"]["observation_cost"]

    # Isolate experimental knowledge: an intact learned response could solve
    # this successor directly and would not need another research experiment.
    method_memory = KnowledgeLibrary(program_experiment_methods=library.program_experiment_methods.to_dict())
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(method_memory.to_dict())))
    successor = WindowLaboratory(("renamed_a", "renamed_b"))
    second = FinitePriorResearcher(replace(CONFIG, seed=139), library=restored).run(successor)
    assert second["status"] == "completed"
    assert any(action["action"] == "use_program_experiment_method" and action["cross_task"]
               for action in second["actions"])
    assert any(action["action"] == "test_competing_program_predictions"
               and action["plan"]["proposal_source"] == "learned_method" for action in second["actions"])


def test_later_random_zeros_cannot_erase_a_paid_compound_counterexample():
    lab = WindowLaboratory(gain=3.)
    report = FinitePriorResearcher(CONFIG).run(lab)
    assert any(action["action"] == "test_competing_program_predictions"
               and action["evidence"]["status"] == "all_refuted" for action in report["actions"])
    assert report["status"] == "unresolved"
    assert any(action["action"] == "retain_program_experiment_counterexample"
               for action in report["actions"])
