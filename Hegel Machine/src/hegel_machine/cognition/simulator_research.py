"""A restorable experiment-research run with a misspecified observation channel."""
from copy import deepcopy
from random import Random

from .experiment_simulator import ExperimentSimulator
from .language import Program
from .models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from .probing import choose_valuable_probe


def _reading(row):
    return {"features": dict(row.features), "target": row.target, "history": list(row.history)}


def _probe(probe):
    return {"probe_id": probe.probe_id, "sequence": list(probe.sequence)}


class ObservationApparatus:
    """Public x controls; unobserved alternating disturbance with x-dependent size.

    Each experiment pays for actual readings. The hidden disturbance phase is
    apparatus state, rather than a target supplied to the simulator learner.
    """
    def __init__(self, state=None):
        self.state = deepcopy(state or {"quiet": 0, "loud": 0, "rows": 0})

    def observe(self, probe):
        result = []
        for point in probe.sequence:
            condition = "quiet" if point["x"] < 0 else "loud"
            sign = 1 if self.state[condition] % 2 else -1
            self.state[condition] += 1
            self.state["rows"] += 1
            bias, amplitude = (.2, .1) if condition == "quiet" else (.8, 2.)
            result.append(Observation(dict(point), bias + sign * amplitude))
        return result


def run_simulator_research(*, library=None, state=None, max_steps=None):
    """Revise a response simulator and execute its selected follow-up experiment.

    This bounded constructed apparatus is a functional demonstration. Its paid
    observations are never used as a new generalization benchmark.
    """
    report = deepcopy(state or {"status": "running", "phase": 0,
        "costs": {"observation_cost": 0, "compute_cost": 0, "measurement_cost": 0, "experiment_cost": 0},
        "observations": [], "selections": [], "selected_probe_changed": False,
        "task_objective": "squared prediction error with fixed public controls",
        "evidence_scope": "constructed observation-model revision apparatus"})
    simulator = ExperimentSimulator(report.get("simulator_state"))
    apparatus = ObservationApparatus(report.get("apparatus_state"))
    steps = 0
    while report["phase"] < 7 and (max_steps is None or steps < max_steps):
        work = SearchBudget(3000)
        phase = report["phase"]
        if phase < 6:
            points = [{"x": -.2}] * 4 + [{"x": 1.}] * 4
            Random(71 + phase).shuffle(points)
            probe = Probe(f"channel-research-{phase}", tuple(points))
            forecast = simulator.forecast(probe, [0.] * probe.cost, baseline_variance=1., budget=work)
            rows = apparatus.observe(probe)
            update = simulator.observe(forecast, rows, budget=work)
            report["observations"].append({"phase": phase, "probe": _probe(probe),
                "readings": [_reading(row) for row in rows], "forecast": forecast, "update": update})
        else:
            probes = [Probe("quiet-discrimination", ({"x": -.2},)),
                      Probe("loud-discrimination", ({"x": 1.},))]
            spec = TaskSpec("observation-channel-question", "Compare two explanations of the same response",
                ("x",), (), tuple(probes), feature_bounds={"x": (-.2, 1.)})
            candidates = [Candidate("negative-slope", Program("mul", children=(Program("const", value=-1.),
                              Program("feature", name="x"))), "refit"),
                          Candidate("slope", Program("feature", name="x"), "refit")]
            # These are real acquired controls and targets from an ordinary
            # apparatus reset at x=0, giving equal evidence for both hypotheses.
            reference = report.get("comparison_observations")
            if reference is None:
                reference_probe = Probe("comparison-reference", ({"x": 0.}, {"x": 0.}))
                acquired = apparatus.observe(reference_probe)
                report["comparison_observations"] = [_reading(row) for row in acquired]
                report["costs"]["observation_cost"] += len(acquired)
            else:
                acquired = [Observation(row["features"], row["target"]) for row in reference]
            old, old_details = choose_valuable_probe(probes, candidates, 0., work,
                observations=acquired, spec=spec, max_compute=800)
            selected, details = choose_valuable_probe(probes, candidates, 0., work,
                observations=acquired, spec=spec, max_compute=800, simulator=simulator)
            forecast = simulator.forecast(selected, [0.] * selected.cost, baseline_variance=1., budget=work)
            rows = apparatus.observe(selected)
            update = simulator.observe(forecast, rows, budget=work)
            report["selected_probe_changed"] = old.probe_id != selected.probe_id
            report["selections"].append({"previous_probe": old.probe_id, "selected_probe": selected.probe_id,
                "previous_design": old_details, "revised_design": details,
                "executed_probe": _probe(selected), "readings": [_reading(row) for row in rows], "update": update})
        report["costs"]["observation_cost"] += len(rows)
        report["costs"]["compute_cost"] += work.spent
        report["phase"] += 1
        steps += 1
    report["status"] = "completed" if report["phase"] == 7 else "paused"
    report["simulator_state"] = simulator.to_dict()
    report["apparatus_state"] = apparatus.state
    report["revisions"] = deepcopy(simulator.events)
    return report
