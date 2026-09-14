"""A no-active-experiments control must keep actual acquisitions fixed."""

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.passive_research import FixedExperimentResearchAgent
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


CONFIG = ResearchConfig(observation_budget=192, compute_budget=120000, minimum_improvement=1e-7)


def test_fixed_design_cannot_change_with_targets_or_available_knowledge():
    library = KnowledgeLibrary()
    reports = []
    # These laboratories have identical public controls and instruments, but
    # different current/history/measurement requirements.
    for task in (0, 1, 2, 2):
        laboratory = UnknownTaskLab(task, 301)
        report = FixedExperimentResearchAgent(CONFIG, library=library).run(laboratory)
        assert report["costs"]["observation_cost"] == 96
        assert report["costs"]["measurement_cost"] == 384
        assert report["status"] == "completed"
        assert report["selected_channels"] == ["sensor_a", "sensor_b"]
        reports.append(report)
    plans = [report["actions"][0] for report in reports]
    assert all(plan == plans[0] for plan in plans)
    acquired = [[batch["probe"] for batch in report["batches"]] for report in reports]
    assert all(probes == acquired[0] for probes in acquired)
    assert reports[-1]["used_sources"]


def test_fixed_design_still_discovers_joint_nonlinear_history():
    laboratory = UnknownTaskLab(7, 991)
    report = FixedExperimentResearchAgent(CONFIG).run(laboratory)
    rows = laboratory.evaluation_rows(report["selected_channels"], seed=871, length=48, trajectories=2)
    assert report["status"] == "completed"
    assert loss(Program.from_dict(report["program"]), rows, report["fallback"], .5) < 1e-12
    assert Program.from_dict(report["program"]).history_depth >= 1


def test_fixed_confirmation_failure_cannot_buy_another_experiment():
    original, changed = UnknownTaskLab(0, 771), UnknownTaskLab(1, 771)

    class ChangingLaboratory:
        spec, channels, spec_for = original.spec, original.channels, original.spec_for

        def __init__(self):
            self.count = 0

        def acquire(self, probe, channels):
            self.count += 1
            return (original if self.count <= 3 else changed).acquire(probe, channels)

    laboratory = ChangingLaboratory()
    library = KnowledgeLibrary()
    report = FixedExperimentResearchAgent(CONFIG, library=library).run(laboratory)
    assert laboratory.count == 4
    assert report["status"] == "unresolved"
    assert report["confirmation_loss"] > CONFIG.minimum_improvement
    assert not library.concepts
