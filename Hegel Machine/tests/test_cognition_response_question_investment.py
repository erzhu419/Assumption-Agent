"""Uncertain response proposals compete for paid follow-up after restoration."""

from dataclasses import asdict
import json
from random import Random
from statistics import mean, stdev

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.mechanism_revision import propose_response_update
from hegel_machine.cognition.meta_prior_claims import compile_claims
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.noise_replication import assess_noise_replication
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.response_research_questions import response_question_offer
from hegel_machine.cognition.unified_session import UnifiedResearchSession


RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")
CONFIG = ResearchConfig(observation_budget=192, compute_budget=20000,
                        minimum_improvement=1e-7, seed=505, max_candidates=8)


class PaidResponseLab:
    channels = ()

    def __init__(self, slope, seed, noise=.1):
        self.spec = TaskSpec("measured_response", "Predict the measured response", ("x",), (), (),
                            feature_bounds={"x": (-1., 1.)}, max_probe_length=24)
        self._slope, self._noise, self._rng = slope, noise, Random(seed)
        self.acquisitions = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def spec_for(self, channels=()):
        assert not channels
        return self.spec

    def random_probe(self, seed, length=12):
        rng = Random(seed)
        return Probe(f"paid_{seed}", tuple({"x": rng.uniform(-1., 1.)} for _ in range(length)))

    def acquire(self, probe, channels=()):
        assert not channels
        rows, history = [], []
        for point in probe.sequence:
            target = .2 + self._slope * point["x"] + self._rng.gauss(0., self._noise)
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        self.acquisitions.append({"probe": probe, "observations": rows})
        return MeasurementBatch(rows, len(rows), 0, ())


def persisted(value):
    return json.loads(json.dumps(value))


def pending_response(slope, seed):
    # Both expressions come from actual measured responses. The current
    # proposal is fitted before its independent repeat supplies a pilot gain.
    source = PaidResponseLab(1., 301, noise=0.)
    training = source.acquire(source.random_probe(491)).observations
    work = SearchBudget(10000)
    parent = next(item for item in compile_claims(source.spec, training, work)
                  if item["kind"] == "affine")
    parent.update(fallback=0.)
    laboratory = PaidResponseLab(slope, seed)
    initial = laboratory.acquire(laboratory.random_probe(492)).observations
    probe = laboratory.random_probe(493)
    original = laboratory.acquire(probe).observations
    repeated = laboratory.acquire(Probe("pilot_repeat", probe.sequence)).observations
    work = SearchBudget(10000)
    proposal = propose_response_update(laboratory.spec, initial, original, repeated, work,
                                       parent=parent, minimum_improvement=CONFIG.minimum_improvement)
    candidate, pilot = proposal["selected"], proposal["event"]
    assert candidate is not None and 0 < pilot["gain"] < pilot["margin"]
    rows = [row for batch in laboratory.acquisitions for row in batch["observations"]]
    assert work.spend(len(rows))
    parent["validation_loss"] = loss(parent["candidate"].program, rows, 0., .5)
    noise = assess_noise_replication(parent["candidate"].program, original, repeated, work)
    question = {"question_id": "response_update:pilot", "parent": encode_entry(parent),
        "candidate": encode_entry(candidate), "observed_gain": pilot["gain"],
        "uncertainty_margin": pilot["margin"], "rows": pilot["heldout_rows"],
        "probe_ids": [probe.probe_id, "pilot_repeat"], "status": "pending", "pilot_only": True,
        "gains": [], "reference_losses": [], "candidate_losses": []}
    batches = [{"probe": asdict(batch["probe"]), "channels": [],
                "observations": [asdict(row) for row in batch["observations"]]}
               for batch in laboratory.acquisitions]
    previous = {"task_id": laboratory.spec.task_id, "status": "completed" if noise["noise_supported"]
                else "unresolved", "program": parent["candidate"].program.to_dict(),
        "expression": parent["candidate"].program.describe(), "selected_kind": parent["kind"],
        "selected_channels": [], "fallback": 0., "validation_loss": parent["validation_loss"],
        "selection_loss": parent["validation_loss"], "selection_basis": "later_trajectory",
        "confirmation_loss": noise["repeated_model_loss"], "noise_assessment": noise,
        "costs": {"observation_cost": len(rows), "measurement_cost": 0,
                  "compute_cost": work.spent, "queries": len(batches)},
        "batches": batches, "actions": [], "concepts_added": [], "used_sources": [],
        "transfer": [], "response_questions": [question]}
    return laboratory, persisted(previous), persisted(question)


def restored_session(laboratory, previous, *, unvisited=False):
    tasks = [{"seed": CONFIG.seed}]
    if unvisited:
        tasks.append({"seed": CONFIG.seed + 1, "research_limits": {
            "observation_budget": 500, "compute_budget": 50000}})
    opened = []

    def factory(item):
        opened.append(item["seed"])
        assert item["seed"] == CONFIG.seed, "a question spent an unvisited task's reserve"
        return laboratory

    session = UnifiedResearchSession(CONFIG, tasks=tasks, measurement_budget=0, lab_factory=factory)
    session.state["results"] = [{"task_position": 0, "available_concepts": [], "research": previous}]
    session.state["status"] = "paused"
    state = persisted(session.to_dict())
    return UnifiedResearchSession(state=state, lab_factory=factory), factory, opened


@pytest.fixture(scope="module")
def investigated_responses():
    cases = []
    for slope, seed in ((1.05, 16), (1., 12)):
        laboratory, previous, question = pending_response(slope, seed)
        count = len(laboratory.acquisitions)
        session, factory, opened = restored_session(laboratory, previous)
        result = persisted(session.run())
        cases.append({"lab": laboratory, "previous": previous, "question": question,
                      "result": result, "purchases": laboratory.acquisitions[count:],
                      "factory": factory, "opened": opened})
    # Private mean responses are bought only after both learning episodes finish.
    for position, item in enumerate(cases):
        terminal = PaidResponseLab((1.05, 1.)[position], 907, noise=0.)
        rows = terminal.acquire(terminal.random_probe(908, length=64)).observations
        current = item["result"]["results"][0]["research"]
        item["mean_mse"] = loss(Program.from_dict(current["program"]), rows, current["fallback"], .5)
        item["old_mean_mse"] = loss(Program.from_dict(item["previous"]["program"]), rows, 0., .5)
    return cases


def test_uncertain_frozen_response_question_survives_json_and_gets_real_followup(investigated_responses):
    item = investigated_responses[0]
    current = item["result"]["results"][0]["research"]
    question, = current["response_questions"]
    assert question["candidate"]["candidate"]["program"] == item["question"]["candidate"]["candidate"]["program"]
    assert question["parent"]["candidate"]["program"] == item["question"]["parent"]["candidate"]["program"]
    assert question["status"] == "settled"
    assert current["program"] == question["candidate"]["candidate"]["program"]
    assert item["mean_mse"] < item["old_mean_mse"] / 5
    assert current["status"] == "completed"


def test_unhelpful_measured_response_question_preserves_the_better_incumbent(investigated_responses):
    item = investigated_responses[1]
    current = item["result"]["results"][0]["research"]
    question, = current["response_questions"]
    assert question["status"] == "settled"
    assert current["program"] == item["previous"]["program"]
    assert current["status"] == "completed"
    assert question["outcome"] == "parent" and question["observed_gain"] < 0
    assert item["mean_mse"] < 1e-12
    assert question["candidate"]["candidate"]["program"] == item["question"]["candidate"]["candidate"]["program"]


def test_question_spends_fresh_trajectory_and_repeat_once_with_all_incremental_costs(investigated_responses):
    for item in investigated_responses:
        result, previous = item["result"], item["previous"]
        current = result["results"][0]["research"]
        purchases = item["purchases"]
        assert len(purchases) == 2
        assert purchases[0]["probe"].sequence == purchases[1]["probe"].sequence
        assert purchases[0]["observations"][0].target != purchases[1]["observations"][0].target
        old_sequences = [batch["probe"]["sequence"] for batch in previous["batches"]]
        assert all(persisted(purchases[0]["probe"].sequence) != sequence for sequence in old_sequences)
        assert current["batches"][:len(previous["batches"])] == previous["batches"]
        assert current["costs"]["queries"] == previous["costs"]["queries"] + 2
        assert current["visit_costs"]["observation_cost"] == sum(
            len(batch["observations"]) for batch in purchases)
        assert current["visit_costs"]["measurement_cost"] == 0
        length = len(purchases[0]["observations"])
        assert current["visit_costs"]["compute_cost"] == 10 * length
        question, = current["response_questions"]
        parent = Program.from_dict(question["parent"]["candidate"]["program"])
        candidate = Program.from_dict(question["candidate"]["candidate"]["program"])
        expected_before, expected_after = [], []
        for pair in zip(*(batch["observations"] for batch in purchases)):
            expected_before.append(mean((row.target - parent.predict(row)) ** 2 for row in pair))
            expected_after.append(mean((row.target - candidate.predict(row)) ** 2 for row in pair))
        assert question["reference_losses"] == pytest.approx(expected_before)
        assert question["candidate_losses"] == pytest.approx(expected_after)
        assert question["gains"] == pytest.approx([old - new
            for old, new in zip(expected_before, expected_after)])
        assert question["rows"] == length
        event, = [event for event in current["actions"] if event["action"] == "resolve_response_question"]
        assert event["gain"] == pytest.approx(mean(question["gains"]))
        assert event["fresh_rows"] == event["accumulated_rows"] == length
        for key in RESOURCES:
            assert current["costs"][key] == previous["costs"][key] + current["visit_costs"][key]
            assert result["costs"][key] == current["costs"][key]
        acquired = len(item["lab"].acquisitions)
        restored = UnifiedResearchSession(state=persisted(result), lab_factory=item["factory"])
        assert persisted(restored.run()) == result
        assert len(item["lab"].acquisitions) == acquired


def test_question_uses_only_visited_resources_and_insufficient_pool_stays_pending():
    laboratory, previous, question = pending_response(1.05, 16)
    session, _, opened = restored_session(laboratory, previous, unvisited=True)
    remaining = session._remaining()
    assert remaining["observation_cost"] == CONFIG.observation_budget - previous["costs"]["observation_cost"]
    assert response_question_offer(previous, question, 24,
        {**remaining, "observation_cost": 1}, CONFIG.minimum_improvement,
        resource_limits={"observation_budget": CONFIG.observation_budget,
                         "compute_budget": CONFIG.compute_budget}) is None
    assert question["status"] == "pending"
    session._offer_response_questions()
    offer, = session.state["pending_revisits"]
    session.state["pending_revisits"].remove(offer)
    assert session._perform_revisit(offer)
    visit, = session.state["revisits"]
    assert visit["resource_limits"]["observation_cost"] <= CONFIG.observation_budget
    assert visit["resource_limits"]["compute_cost"] <= CONFIG.compute_budget
    assert set(opened) == {CONFIG.seed}
    assert session.to_dict()["pending_tasks"] == [1]


def test_quote_uses_measured_loss_and_requires_executable_history():
    _, previous, question = pending_response(1.05, 16)
    limits = {"observation_budget": CONFIG.observation_budget,
              "compute_budget": CONFIG.compute_budget}
    remaining = {"observation_cost": 100, "measurement_cost": 0, "compute_cost": 10000}
    original = response_question_offer(previous, question, 24, remaining, resource_limits=limits)
    assert original is not None
    question["reference_loss"] = previous["selection_loss"]
    question["parent"]["validation_loss"] = None
    quoted = response_question_offer(previous, question, 24, remaining, resource_limits=limits)
    assert quoted["relative_gain"] == pytest.approx(original["relative_gain"])
    assert quoted["minimum_resources"] == original["minimum_resources"]
    frozen = persisted(question)
    measured_incumbent = {**previous, "selected_channels": ["sensor_a"]}
    assert response_question_offer(measured_incumbent, question, 24, remaining, resource_limits=limits) is None
    autoregressive_incumbent = {**previous,
        "program": Program("lag", name="__target__", lag=1).to_dict()}
    assert response_question_offer(autoregressive_incumbent, question, 24, remaining, resource_limits=limits) is None
    assert question == frozen
    # A saved lag can require more history than a successor apparatus permits.
    question["candidate"]["candidate"]["program"] = Program("lag", name="x", lag=3).to_dict()
    assert response_question_offer(previous, question, 3, remaining, resource_limits=limits) is None


def pending_response_rivals():
    entries = []
    for slope in (.8, 1., .95):
        source = PaidResponseLab(slope, 301, noise=0.)
        rows = source.acquire(source.random_probe(491)).observations
        fitted = next(item for item in compile_claims(source.spec, rows, SearchBudget(10000))
                      if item["kind"] == "affine")
        fitted["fallback"] = 0.
        entries.append(fitted)
    parent, first, rival = entries
    laboratory = PaidResponseLab(1., 16, noise=.01)
    laboratory.acquire(laboratory.random_probe(492))
    original_probe = laboratory.random_probe(493)
    original = laboratory.acquire(original_probe).observations
    repeated = laboratory.acquire(Probe("pilot_repeat", original_probe.sequence)).observations
    rows = [row for batch in laboratory.acquisitions for row in batch["observations"]]
    work = SearchBudget(10000)
    assert work.spend(len(entries) * len(rows))
    for item in entries:
        item["validation_loss"] = loss(item["candidate"].program, rows, 0., .5)
    questions = []
    for index, candidate in enumerate((first, rival)):
        gains = [(row.target - parent["candidate"].program.predict(row)) ** 2
                 - (row.target - candidate["candidate"].program.predict(row)) ** 2 for row in repeated]
        questions.append({"question_id": f"response_update:rival_{index}",
            "parent": encode_entry(parent), "candidate": encode_entry(candidate),
            "observed_gain": mean(gains), "uncertainty_margin": 3 * stdev(gains) / len(gains) ** .5,
            "rows": len(gains), "probe_ids": [original_probe.probe_id, "pilot_repeat"],
            "status": "pending", "pilot_only": True, "reference_loss": parent["validation_loss"],
            "gains": [], "reference_losses": [], "candidate_losses": []})
        assert questions[-1]["observed_gain"] > 0
    noise = assess_noise_replication(parent["candidate"].program, original, repeated, work)
    previous = {"task_id": laboratory.spec.task_id, "status": "unresolved",
        "program": parent["candidate"].program.to_dict(), "expression": parent["candidate"].program.describe(),
        "selected_kind": parent["kind"], "selected_channels": [], "fallback": 0.,
        "validation_loss": parent["validation_loss"], "selection_loss": parent["validation_loss"],
        "confirmation_loss": noise["repeated_model_loss"], "noise_assessment": noise,
        "costs": {"observation_cost": len(rows), "measurement_cost": 0,
                  "compute_cost": work.spent, "queries": len(laboratory.acquisitions)},
        "batches": [{"probe": asdict(batch["probe"]), "channels": [],
                     "observations": [asdict(row) for row in batch["observations"]]}
                    for batch in laboratory.acquisitions],
        "actions": [], "concepts_added": [], "used_sources": [], "transfer": [],
        "response_questions": questions}
    return laboratory, previous, questions, first, rival


def test_pending_rival_survives_adoption_and_cannot_overwrite_a_better_new_incumbent():
    laboratory, previous, questions, first, rival = pending_response_rivals()
    session, factory, _ = restored_session(laboratory, persisted(previous))
    session._offer_response_questions()
    first_offer = next(offer for offer in session.state["pending_revisits"]
                       if offer["question_id"] == questions[0]["question_id"])
    session.state["pending_revisits"].remove(first_offer)
    assert first_offer["minimum_resources"]["compute_cost"] == 10 * first_offer["candidate"]["probe_length"]
    assert session._perform_revisit(first_offer)
    after_first = persisted(session.to_dict())
    current = after_first["results"][0]["research"]
    assert current["program"] == first["candidate"].program.to_dict()
    assert current["status"] == "completed"
    assert current["response_questions"][0]["status"] == "settled"
    assert current["response_questions"][1]["status"] == "pending"
    # The rival beat the obsolete parent, but its newly priced comparison with
    # the better adopted answer no longer warrants another acquisition.
    assert after_first["pending_revisits"] == []
    rival_quote = current["response_questions"][1]["current_quote"]
    assert rival_quote["observed_gain"] < 0
    assert rival_quote["expected_gain"] <= CONFIG.minimum_improvement
    assert after_first["screening_compute_cost"] > 0
    count = len(laboratory.acquisitions)
    restored = UnifiedResearchSession(state=after_first, lab_factory=factory)
    final = restored.run()
    current = final["results"][0]["research"]
    assert len(laboratory.acquisitions) == count
    assert current["program"] == first["candidate"].program.to_dict()
    assert current["status"] == "completed"
    assert current["response_questions"][1]["status"] == "pending"
    assert current["response_questions"][1]["candidate"]["candidate"]["program"] == rival["candidate"].program.to_dict()
    assert final["costs"] == after_first["costs"]
    assert current["response_questions"][1]["observed_gain"] > 0
    assert final["pending_revisits"] == []
    visits = final["revisits"]
    assert len(visits) == 1
    for key in RESOURCES:
        screening = final["screening_compute_cost"] if key == "compute_cost" else 0
        assert final["costs"][key] == previous["costs"][key] + sum(
            visit["visit_costs"][key] for visit in visits) + screening
