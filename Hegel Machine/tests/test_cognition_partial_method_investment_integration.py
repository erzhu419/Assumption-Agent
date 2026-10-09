"""A complete numeric research visit retains and resumes a priced partial method."""

from dataclasses import asdict, replace
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch, MeasurementChannel
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


class GapReadingLab:
    """Commands cannot bridge the instrument's actual unobserved reading gap."""

    channels = (MeasurementChannel("meter", 1, "Read the auxiliary instrument."),)

    def __init__(self):
        self.spec = TaskSpec("unlabelled_condition", "Predict the measured response.",
            ("drive", "bias"), (), (),
            feature_bounds={"drive": (-1., 1.), "bias": (-1., 1.)}, max_probe_length=32)
        self.observation_cost = self.measurement_cost = 0
        self.reading_amplitude = .8

    def spec_for(self, channels=()):
        return replace(self.spec, feature_names=(*self.spec.feature_names, *channels))

    def acquire(self, probe, channels=()):
        assert all(set(point) == set(self.spec.feature_bounds) for point in probe.sequence)
        rows, history = [], []
        for index, point in enumerate(probe.sequence):
            meter = .2 * point["bias"] + self.reading_amplitude * (1. if index % 2 else -1.)
            target = point["drive"] * (1. if meter >= .15 else -1.) + .3 * meter
            features = {**point, **({"meter": meter} if "meter" in channels else {})}
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
        self.observation_cost += probe.cost
        self.measurement_cost += probe.cost * len(channels)
        return MeasurementBatch(rows, probe.cost, probe.cost * len(channels), tuple(channels))


def retained_response():
    drive, meter = Program("feature", name="drive"), Program("feature", name="meter")
    branch = Program("if", children=(Program("ge", children=(meter, Program("const", value=0.))),
        drive, Program("mul", children=(Program("const", value=-1.), drive))))
    return Program("add", children=(branch, Program("mul", children=(Program("const", value=.3), meter))))


def append_paid(report, lab, probe):
    batch = lab.acquire(probe, ("meter",))
    report["batches"].append({"probe": asdict(probe), "channels": ["meter"],
                            "observations": [asdict(row) for row in batch.observations]})
    report["costs"]["observation_cost"] += batch.observation_cost
    report["costs"]["measurement_cost"] += batch.measurement_cost
    report["costs"]["queries"] += 1


def paid_partial_report(lab):
    program = retained_response()
    previous = {"task_id": lab.spec.task_id, "status": "unresolved", "program": program.to_dict(),
        "fallback": 0., "selected_kind": "conditional_response", "selected_channels": ["meter"],
        "research_program": {"selection_steps": [{"operation": "discover_conditional_coordinates",
            "program": program.to_dict(), "question": "Locate the measured response condition", "loss": 0.,
            "gate_coordinate": Program("feature", name="meter").to_dict(), "threshold_interval": [-.2, .2]}]},
        "actions": [], "transfer": [], "batches": [], "research_graph": {}, "noise_assessment": None,
        "costs": {"observation_cost": 0, "measurement_cost": 0, "compute_cost": 0, "queries": 0}}
    for index in range(2):
        probe = Probe(f"initial_paid:{index}", tuple({"drive": -.9 + .075 * step,
            "bias": (step % 3) - 1.} for step in range(24)))
        append_paid(previous, lab, probe)
    return previous


def config():
    return ResearchConfig(observation_budget=480, compute_budget=500000,
        minimum_improvement=1e-7, max_rounds=1, use_memory=False,
        active_probes=False, proposer="search", seed=870101)


def investment_events(report, *, after=0):
    return [event for event in report["actions"][after:]
            if event["action"] == "choose_condition_research_investment"]


def test_actual_low_information_procurement_stops_pending_and_survives_json_resume():
    lab = GapReadingLab()
    report = UnifiedResearchAgent(config(), measurement_budget=480).run(lab,
        previous=paid_partial_report(lab))
    decisions = investment_events(report)
    assert any(event["buy"] for event in decisions)
    assert not decisions[-1]["buy"]
    assert decisions[-1]["reason"] == "retain_question_for_more_valuable_future_evidence"
    assert report["status"] == "unresolved" and report["current_fit_confirmed"]
    assert report["condition_uncertainty"]["status"] == "unresolved"
    assert report["condition_uncertainty"]["original_target_loss"] <= 1e-7
    returns = report["condition_research_returns"]
    revisions = [event for event in report["actions"]
                 if event["action"] == "revise_competing_observed_conditions"]
    assert returns["batches"] == len(revisions) > 1
    assert returns["information_batches"] == 0
    assert returns["observation_cost"] == returns["measurement_cost"] == 24 * len(revisions)
    assert returns["compute_cost"] == sum(event["compute_cost"] for event in revisions)
    assert report["costs"]["observation_cost"] == lab.observation_cost < 480 - 48
    assert report["costs"]["measurement_cost"] == lab.measurement_cost
    assert report["costs"]["compute_cost"] < 500000 - 20000
    assert any(item["condition_research_returns"] == returns
               for item in report["continuation_state"]["alternatives"])
    action_count = len(report["actions"])
    restored = UnifiedResearchAgent(config(), measurement_budget=480).run(lab,
        previous=json.loads(json.dumps(report)))
    new_decisions = investment_events(restored, after=action_count)
    assert new_decisions and not any(event["buy"] for event in new_decisions)
    assert restored["condition_research_returns"] == returns
    assert restored["status"] == "unresolved"
    assert restored["visit_costs"]["observation_cost"] == 48
    assert restored["costs"]["observation_cost"] == lab.observation_cost


def test_new_real_gap_evidence_reopens_full_consumer_and_revises_response():
    lab = GapReadingLab()
    initial = UnifiedResearchAgent(config(), measurement_budget=480).run(lab,
        previous=paid_partial_report(lab))
    resumed = json.loads(json.dumps(initial))
    # This is a newly paid representative measurement, with its actual target
    # and legal raw commands. The consumer sees no private threshold or model.
    lab.reading_amplitude = .1
    append_paid(resumed, lab, Probe("new_paid_gap_evidence",
        tuple({"drive": .9 if step % 2 else -.9, "bias": 0.} for step in range(24))))
    count = len(resumed["actions"])
    result = UnifiedResearchAgent(config(), measurement_budget=480).run(lab, previous=resumed)
    decisions = investment_events(result, after=count)
    assert any(event["buy"] and event["reopened"] for event in decisions)
    assert any(event["action"] == "revise_competing_observed_conditions"
               and event["changed_program"] for event in result["actions"][count:])
    assert result["program"] != initial["program"]
    assert result["condition_research_returns"]["information_batches"] >= 1
    assert result["condition_uncertainty"]["original_target_loss"] <= 1e-7
    assert result["confirmation_loss"] <= 1e-7
    assert result["status"] == "unresolved"
    assert result["costs"]["observation_cost"] == lab.observation_cost
    assert result["costs"]["measurement_cost"] == lab.measurement_cost
