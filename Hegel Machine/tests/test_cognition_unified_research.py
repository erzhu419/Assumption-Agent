"""Behavioral checks for common research, joint revision and causal reuse."""

from dataclasses import replace

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


CONFIG = ResearchConfig(observation_budget=192, compute_budget=120000,
                        minimum_improvement=1e-7, max_candidates=16)


class PublicLaboratory:
    """Expose exactly the instrument API to the researcher, without task labels."""

    def __init__(self, laboratory):
        self.spec = replace(laboratory.spec, task_id="unlabeled_experiment")
        self.channels = laboratory.channels
        self.acquire = laboratory.acquire
        self.spec_for = lambda channels=(): replace(laboratory.spec_for(channels),
                                                    task_id="unlabeled_experiment")


def evaluated(task, *, config=CONFIG, library=None, seed=113):
    laboratory = UnknownTaskLab(task, seed)
    report = UnifiedResearchAgent(config, library=library).run(PublicLaboratory(laboratory))
    rows = laboratory.evaluation_rows(report["selected_channels"], seed=733,
                                     length=40, trajectories=2)
    measured = loss(Program.from_dict(report["program"]), rows, report["fallback"], .5)
    return report, measured


@pytest.mark.parametrize("task,depth,channels", [(0, 0, []), (1, 1, []),
                                                (2, 1, ["sensor_a"]),
                                                (7, 1, ["meter_r"])])
def test_unknown_task_selects_joint_explanation_from_paid_evidence(task, depth, channels):
    report, measured = evaluated(task)
    assert report["status"] == "completed"
    assert measured < 1e-12
    assert report["selected_channels"] == channels
    assert Program.from_dict(report["program"]).history_depth == depth
    assert report["costs"]["observation_cost"] <= CONFIG.observation_budget
    assert report["costs"]["compute_cost"] <= CONFIG.compute_budget
    assert report["costs"]["measurement_cost"] <= 768
    confirmations = [action for action in report["actions"]
                     if action["action"] == "validate_frozen_program"]
    assert confirmations[-1]["accepted"]
    if channels:
        assert any(action["action"] == "open_joint_measurement_history_question"
                   for action in report["actions"])
        assert any(action["action"] == "execute_dependent_research" and action["answered"]
                   for action in report["actions"])


def test_renamed_successor_executes_learned_call_and_avoids_rediscovery():
    library = KnowledgeLibrary()
    source, _ = evaluated(2, library=library)
    successor, measured = evaluated(3, library=library, seed=211)
    independent, independent_error = evaluated(3, library=KnowledgeLibrary(), seed=211)
    assert measured < 1e-12 and independent_error < 1e-12
    assert set(successor["used_sources"]) & set(source["concepts_added"])
    assert successor["selected_kind"] == "short_callable_transfer"
    assert successor["costs"]["compute_cost"] < independent["costs"]["compute_cost"]
    assert successor["costs"]["observation_cost"] < independent["costs"]["observation_cost"]
    assert successor["costs"]["measurement_cost"] < independent["costs"]["measurement_cost"]
    assert any(event["accepted"] for event in successor["transfer"])


def test_changed_delay_rejects_old_binding_on_new_trajectory():
    library = KnowledgeLibrary()
    source, _ = evaluated(2, library=library)
    changed, measured = evaluated(4, library=library, seed=311)
    assert measured < 1e-12
    assert Program.from_dict(changed["program"]).history_depth == 2
    old_ids = set(source["concepts_added"])
    assert any(old_ids & set(event["source_ids"]) and event["status"] == "counterexample"
               for event in changed["transfer"])
    assert not old_ids & set(changed["used_sources"])
    assert any(prior["counterexamples"] for prior in library.meta_priors
               if prior["source_concept_id"] in old_ids)
    assert any(action["action"] == "test_learned_acquisition_method"
               and action["validation_loss"] > CONFIG.minimum_improvement
               for action in changed["actions"])
    assert changed["costs"]["observation_cost"] <= CONFIG.observation_budget


def test_no_active_probe_ablation_changes_experiment_policy():
    # Directed research compares controllable predictions. Final confirmation
    # deliberately uses a representative trajectory in both arms.
    spec = UnknownTaskLab(0, 113).spec
    coordinate = Program("feature", name=next(iter(spec.feature_bounds)))
    claims = [{"candidate": Candidate("left", coordinate, "test")},
              {"candidate": Candidate("right", Program("const", value=0.), "test")}]
    active_actions, passive_actions = [], []
    active = UnifiedResearchAgent(replace(CONFIG, active_probes=True))._probe(
        spec, 0, 24, claims, SearchBudget(10000), active_actions)
    passive = UnifiedResearchAgent(replace(CONFIG, active_probes=False))._probe(
        spec, 0, 24, claims, SearchBudget(10000), passive_actions)
    assert active_actions[-1]["policy"] == "prediction_disagreement"
    assert passive_actions[-1]["policy"] == "independent_random"
    assert active.sequence != passive.sequence


@pytest.mark.parametrize("task", [0, 2, 4, 5, 7])
def test_fixed_basis_control_can_solve_joint_tasks_without_learning_library(task):
    library = KnowledgeLibrary()
    report, measured = evaluated(task, config=replace(CONFIG, strategy="fixed"), library=library)
    assert measured < 1e-12
    assert report["selected_kind"] == "fixed_quadratic_two_step_history"
    assert len(report["selected_channels"]) == 2
    assert not library.concepts


def test_exhausted_instrument_budget_keeps_unresolved_research_visible():
    lab = UnknownTaskLab(2, 191)
    report = UnifiedResearchAgent(CONFIG, measurement_budget=0).run(PublicLaboratory(lab))
    assert report["status"] == "unresolved"
    assert report["costs"]["measurement_cost"] == 0
    assert not report["concepts_added"]


def test_shared_library_is_consumed_by_existing_meta_prior_research_and_session():
    from hegel_machine.cognition.meta_prior_research import MetaPriorResearch
    from hegel_machine.cognition.meta_prior_session import MetaPriorSession

    library = KnowledgeLibrary()
    learned, _ = evaluated(0, library=library)
    source_id = learned["concepts_added"][0]
    continued = MetaPriorResearch(CONFIG, library).run(UnknownTaskLab(0, 991))
    assert continued["status"] == "completed"
    assert any(source_id in update["used_sources"] for update in continued["prior_updates"])
    assert next(prior for prior in library.meta_priors
                if prior["source_concept_id"] == source_id)["uses"] >= 1

    restored = KnowledgeLibrary.from_dict(library.to_dict())
    prior_ids = [prior["prior_id"] for prior in restored.meta_priors]
    report = MetaPriorSession(CONFIG, library=restored).run(max_tasks=1)
    assert report["results"][0]["available_prior_ids"] == prior_ids
    assert report["results"][0]["research"]["status"] == "completed"


def test_failed_binding_counterexamples_support_existing_replacement_consumer():
    from hegel_machine.cognition.meta_prior_lab import ScalarApparatus
    from hegel_machine.cognition.meta_prior_research import MetaPriorResearch

    library = KnowledgeLibrary()
    evaluated(2, library=library)
    evaluated(4, library=library, seed=311)
    failures = [event for prior in library.meta_priors for event in prior["counterexamples"]]
    assert failures
    assert all("replacement_concept_id" in event for event in failures)
    environment = ScalarApparatus(lambda x, z: .3 + x * x + z, "unlabeled_experiment")
    continued = MetaPriorResearch(CONFIG, library).run(environment)
    assert continued["status"] == "completed"


def test_deleting_callable_knowledge_removes_short_acquisition_benefit():
    library = KnowledgeLibrary()
    source, _ = evaluated(2, library=library)
    deleted = library.without(set(source["concepts_added"]))
    retained, _ = evaluated(3, library=library, seed=211)
    absent, measured = evaluated(3, library=deleted, seed=211)
    assert measured < 1e-12
    assert retained["costs"]["observation_cost"] == 24
    assert absent["costs"]["observation_cost"] == 168
    assert not any(action["action"] == "test_learned_acquisition_method" for action in absent["actions"])


def test_short_trial_rebinds_measurement_role_and_pays_for_new_sensor():
    library = KnowledgeLibrary()
    source, _ = evaluated(2, library=library)
    swapped, measured = evaluated(6, library=library, seed=211)
    assert measured < 1e-12
    assert any(action["action"] == "validate_frozen_program"
               and action.get("short_trial") and action["accepted"] for action in swapped["actions"])
    assert swapped["selected_channels"] == ["reading_n"]
    assert swapped["costs"]["measurement_cost"] == 8 * 4 + 16 * 3
    assert swapped["costs"]["observation_cost"] == 24
    assert set(swapped["used_sources"]) & set(source["concepts_added"])


def test_short_trial_is_skipped_when_full_recovery_cannot_be_reserved():
    library = KnowledgeLibrary()
    evaluated(2, library=library)
    limited, measured = evaluated(3, config=replace(CONFIG, observation_budget=160), library=library)
    assert measured < 1e-12
    assert limited["costs"]["observation_cost"] <= 160
    assert not any(action["action"] == "test_learned_acquisition_method"
                   for action in limited["actions"])


def test_failed_short_confirmation_records_rejection_loss_and_reopens_research():
    library = KnowledgeLibrary()
    source, _ = evaluated(1, library=library)
    original, changed = UnknownTaskLab(1, 197), UnknownTaskLab(0, 197)

    class ChangingLaboratory:
        spec, channels, spec_for = original.spec, original.channels, original.spec_for

        def __init__(self):
            self.count = 0

        def acquire(self, probe, channels=()):
            self.count += 1
            return (original if self.count <= 2 else changed).acquire(probe, channels)

    result = UnifiedResearchAgent(CONFIG, library=library).run(ChangingLaboratory())
    short_failure = next(action for action in result["actions"]
                         if action["action"] == "validate_frozen_program" and action.get("short_trial"))
    assert not short_failure["accepted"]
    assert short_failure["loss"] > CONFIG.minimum_improvement
    assert result["status"] == "completed"
    assert not result["selected_channels"]
    assert Program.from_dict(result["program"]).history_depth == 0
    failures = [event for prior in library.meta_priors
                if prior["source_concept_id"] in source["concepts_added"]
                for event in prior["counterexamples"]]
    assert any(event["loss"] == short_failure["loss"] for event in failures)
