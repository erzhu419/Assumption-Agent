"""A second source world must actually challenge source-specific constants."""

from copy import deepcopy
import json

from hegel_machine.cognition import world_method_learning
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_discoveryworld_agent import LIMITS
from test_world_method_learning import PracticeLab, SOLVE, finish


class SeedLab(PracticeLab):
    def __init__(self, task):
        self.task = deepcopy(task)
        self.reading = 3 if task["random_seed"] == 0 else 7
        super().__init__((self.reading,))

    def act(self, action):
        result = super().act(action)
        if action.get("action") == "TUNE":
            self.success = action["arg1"] == 2 * self.reading
        return result


def multiple_sources(tmp_path, monkeypatch, source):
    first = {"id": "source0", "random_seed": 0, "split": "development"}
    other = {"id": "source3", "random_seed": 3, "split": "development"}
    original = SeedLab(first)
    records = [{"index": 0, **original.act({"action": "MEASURE"})},
               {"index": 1, **original.act({"action": "TUNE", "arg1": 6})}]
    labs = []

    def laboratory(task, runtime, workspace):
        assert runtime == "source-runtime"
        lab = SeedLab(task)
        labs.append(lab)
        return lab

    monkeypatch.setattr(world_method_learning, "DiscoveryWorldLab", laboratory)
    learner = world_method_learning.WorldMethodLearner(None, tmp_path,
        knowledge={"files": {METHOD_PATH: source}}, task=first,
        additional_tasks=[other], records=records, prefix_steps=1,
        public_spec={}, limits=LIMITS, calls=8, runtime="source-runtime",
        provider_factory=lambda _: None, max_steps=6)
    learner.questions = [{"id": "old_source_answer", "status": "supported",
                          "answer": {"frequency": 6}, "evidence": [0]}]
    return learner, labs, other


def test_source_constant_cannot_publish_after_only_the_original_world_passes(tmp_path, monkeypatch):
    constant = SOLVE.replace("2 * reading", "6")
    learner, labs, other = multiple_sources(tmp_path, monkeypatch, constant)

    refused = finish(learner)

    assert "published" not in refused and learner.published_source is None
    assert refused["current_source_coverage"]["confirmed_cases"] == [
        "source_continuation", "source_initial"]
    assert refused["unconfirmed_cases"] == ["source3_initial"]
    assert [lab.task["random_seed"] for lab in labs] == [0, 0, 3]
    assert learner.cases["source3_initial"]["task"] == other
    assert learner.cases["source3_initial"]["prefix_steps"] == 0
    trial = json.loads((tmp_path / "trials/trial_002/result.json").read_text())
    assert trial["replayed_environment_steps"] == 0
    assert trial["new_environment_steps"] == 2
    assert trial["paid_observations"][0]["feedback"]["reading"] == 7
    assert not trial["fresh_confirmation"]["passed"]
    assert not any(question["id"] == "old_source_answer"
                   for question in trial["research"]["questions"])


def test_measured_program_publishes_only_after_all_source_worlds_and_retains_costs(tmp_path, monkeypatch):
    learner, labs, _ = multiple_sources(tmp_path, monkeypatch, SOLVE)

    published = finish(learner)

    assert published["published"] == METHOD_PATH
    assert learner.published_source == SOLVE
    assert published["current_source_coverage"]["confirmed_cases"] == [
        "source3_initial", "source_continuation", "source_initial"]
    assert [lab.actions[-1]["arg1"] for lab in labs] == [6, 6, 14]
    assert learner.costs["environment_steps"] == learner.costs["observation_cost"] == 6
    assert learner.costs["model_calls"] == 0
    for field in ("environment_steps", "observation_cost", "python_execution_seconds"):
        assert learner.costs[field] == sum(trial["costs"][field] for trial in learner.practice_trials)
    before = deepcopy(learner.costs)
    assert finish(learner)["publication_practices"] == []
    assert learner.costs == before
