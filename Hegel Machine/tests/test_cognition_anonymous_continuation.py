"""Anonymous dependency research resumes using its original sensor-bank evidence."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.anonymous_revisits import offer_anonymous_successor
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.meta_prior_claims import compile_claims
from hegel_machine.cognition.models import Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry, resume_entries
from hegel_machine.cognition.object_induction_continuation import paid_object_records
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession


CONFIG = ResearchConfig(observation_budget=512, compute_budget=480000,
    minimum_improvement=1e-7, max_rounds=6, max_candidates=16,
    active_probes=True, seed=71)
BASE = {"environment": "anonymous_objects", "objects": 3, "apparatus_seed": 9013,
        "name_seed": 42114, "wiring_seed": 1780, "variant": "exchange", "readout": "focus"}
TASKS = [
    {**BASE, "seed": 71, "research_limits": {"observation_budget": 64,
        "compute_budget": 30000, "measurement_budget": 1152}},
    {**BASE, "seed": 10078},
]
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def serialized(value):
    return json.loads(json.dumps(value))


def recorded_factory(purchases, projections, evaluations):
    class RecordedLab(AnonymousObjectLab):
        def acquire(self, probe, channels=()):
            batch = super().acquire(probe, channels)
            purchases.append({"seed": self.seed, "probe": probe,
                "channels": tuple(channels), "rows": list(batch.observations),
                "observation_cost": batch.observation_cost,
                "measurement_cost": batch.measurement_cost})
            return batch

        def project_rows(self, rows, source_channels, target_channels):
            projections.append((tuple(source_channels), tuple(target_channels)))
            return super().project_rows(rows, source_channels, target_channels)

        def project_transitions(self, transitions, source_channels, target_channels):
            projections.append((tuple(source_channels), tuple(target_channels)))
            return super().project_transitions(transitions, source_channels, target_channels)

        def evaluation_rows(self, *args, **kwargs):
            evaluations.append(self.seed)
            return super().evaluation_rows(*args, **kwargs)

    return lambda item: RecordedLab(**{key: value for key, value in item.items()
        if key not in {"environment", "research_limits"}})


@pytest.fixture(scope="module")
def resumed_anonymous_stream(tmp_path_factory):
    purchases, projections, evaluations = [], [], []
    factory = recorded_factory(purchases, projections, evaluations)
    directory = tmp_path_factory.mktemp("anonymous_continuation")

    def save(name, state):
        path = directory / name
        path.write_text(json.dumps(state))
        return json.loads(path.read_text())

    first = save("first.json", UnifiedResearchSession(CONFIG, tasks=TASKS,
        measurement_budget=6144, lab_factory=factory).run(max_tasks=1))
    successor = UnifiedResearchSession(state=first, lab_factory=factory)
    queued = save("queued.json", successor.run(max_tasks=1))
    boundary = len(purchases)
    resumed = UnifiedResearchSession(state=queued, lab_factory=factory)
    final = save("final.json", resumed.run(max_tasks=1))
    return {"first": first, "queued": queued, "final": final,
        "factory": factory, "purchases": purchases, "projections": projections,
        "evaluations": evaluations, "boundary": boundary}


def test_object_model_notification_can_schedule_research_without_a_new_response_concept(resumed_anonymous_stream):
    stream = resumed_anonymous_stream
    state = deepcopy(stream["queued"])
    old = state["results"][0]["research"]
    assert old["status"] == "unresolved"
    assert old["object_research"]["groups"] and not old["object_research"]["complete"]
    old_ids = {model["model_id"] for model in stream["first"]["knowledge"]["object_models"]}
    new_ids = [model["model_id"] for model in state["knowledge"]["object_models"]
               if model["model_id"] not in old_ids]
    assert new_ids
    # Replay only the actual model notification. Existing learned concepts are
    # left intact; they are not supplied as the reason to reopen this task.
    state["pending_revisits"] = []
    session = UnifiedResearchSession(state=state, lab_factory=stream["factory"])
    count = len(stream["purchases"])
    session._offer_new_knowledge(1, [], new_ids)
    proposed = session.to_dict()
    offer, = proposed["pending_revisits"]
    assert offer["candidate"]["kind"] == "successor_object_model"
    assert offer["candidate"]["source_concept_ids"] == []
    assert set(offer["candidate"]["source_object_model_ids"]) <= set(new_ids)
    assert offer["proposed_assignments"] and offer["old_evidence_gain"] == 0.
    assert max(offer["calibration_losses"].values()) <= CONFIG.minimum_improvement
    assert proposed["scheduler_events"][-1]["source_concept_ids"] == []
    assert len(stream["purchases"]) == count
    # Public labels can stay identical while their actual wiring changes.
    # Test the same learned source against real responses from that apparatus;
    # no second full source search is necessary to exercise calibration rejection.
    changed = deepcopy(old)
    with recorded_factory([], [], [])({**TASKS[0], "wiring_seed": 2680}) as lab:
        for batch in changed["batches"]:
            observed = lab.acquire(Probe(batch["probe"]["probe_id"],
                tuple(batch["probe"]["sequence"])), batch["channels"])
            batch["observations"] = [asdict(row) for row in observed.observations]
            batch["transitions"] = deepcopy(observed.transitions)
        agent = UnifiedResearchAgent(CONFIG, library=KnowledgeLibrary.from_dict(state["knowledge"]))
        work = SearchBudget(60000)
        assert offer_anonymous_successor(agent, lab, changed, set(new_ids), work) is None
        assert work.spent > 0


def test_json_resume_uses_new_dependency_evidence_in_the_response_research(resumed_anonymous_stream):
    stream = resumed_anonymous_stream
    old = stream["first"]["results"][0]["research"]
    current = stream["final"]["results"][0]["research"]
    assert stream["evaluations"] == []
    assert current["batches"][:len(old["batches"])] == old["batches"]
    assert current["actions"][:len(old["actions"])] == old["actions"]
    model = current["object_research"]
    assert model["model_id"] != old["object_research"]["model_id"]
    assert model["complete"] and current["learned_object_schema"] == model["schema"]
    assert len(model["unresolved"]) < len(old["object_research"]["unresolved"])
    events = current["actions"][len(old["actions"]):]
    resumed = next(event for event in events if event["action"] == "resume_object_induction")
    old_fine = [batch for batch in old["batches"] if batch["channels"] == ["fine"]]
    assert resumed["paid_transition_rows"] == sum(len(batch["transitions"]) for batch in old_fine)
    assert resumed["paid_transition_rows"] > old["object_research"]["acquisition_rows"]
    assert set(resumed["paid_probe_ids"]) == {batch["probe"]["probe_id"] for batch in old_fine}
    questions = [event for event in events if event["action"] == "test_object_dependency_question"]
    assert questions
    assert any(len(event["unresolved_after"]) < len(event["unresolved_before"]) for event in questions)
    acquired = [event for event in events if event["action"] == "acquire"]
    assert not any(event["purpose"] == "confirm_discovered_object_roles" for event in acquired)
    assert any(event["action"] == "validate_frozen_program" and event["accepted"] for event in events)
    confirmed_roles = next(index for index, event in enumerate(events)
                          if event["action"] == "confirm_resumed_object_roles" and event["supported"])
    used_roles = next(index for index, event in enumerate(events)
                     if event["action"] == "research_response_after_object_induction")
    relation = next(index for index, event in enumerate(events)
                    if event["action"] == "inherit_relation_intervals")
    confirmed_response = next(index for index, event in enumerate(events)
                             if event["action"] == "validate_frozen_program" and event["accepted"])
    first_response_purchase = next(index for index, event in enumerate(events)
        if event["action"] == "acquire" and event["purpose"] == "confirm_frozen_program")
    assert used_roles < relation < first_response_purchase < confirmed_roles < confirmed_response
    assert events[used_roles]["before_loss"] > CONFIG.minimum_improvement
    assert events[used_roles]["after_loss"] < events[used_roles]["before_loss"]
    assert events[used_roles]["status"] == "requires_fresh_confirmation"
    comparisons = [event for event in events
                   if event["action"] == "compare_object_continuation_candidates"]
    assert comparisons
    assert all(event["baseline_program"] == old["program"] for event in comparisons)
    role_probe = events[confirmed_roles]["probe_id"]
    assert any(event["probe_id"] == role_probe for event in comparisons)
    assert events[confirmed_roles]["shared_with_response"]
    assert model["continuation"]["confirmation"]["status"] == "supported"
    memberships = {fields["f"] for fields in model["schema"]["fields"].values()}
    # A later independently checked instrument substitution can express the
    # same answer in controls. The measured response that actually used these
    # memberships must already have passed its own fresh comparison.
    assert any(memberships <= Program.from_dict(event["program"]).required_features()
               and event["after_loss"] <= CONFIG.minimum_improvement for event in comparisons)
    assert current["status"] == "completed" and current["confirmation_loss"] <= 1e-7
    # The grouping is useful only if its executable response survives new
    # controls. The evaluator runs after all selection and purchases finish.
    session = UnifiedResearchSession(state=stream["final"], lab_factory=stream["factory"])
    scored = session.evaluate(length=32, trajectories=2, seed_offset=81173)
    assert scored["evaluation"][0]["loss"] <= CONFIG.minimum_improvement
    assert scored["costs"] == stream["final"]["costs"]


def test_resumed_anonymous_banks_and_shared_budget_do_not_repurchase_old_data(resumed_anonymous_stream):
    stream = resumed_anonymous_stream
    old = stream["first"]["results"][0]["research"]
    final = stream["final"]
    current = final["results"][0]["research"]
    paid = stream["purchases"][stream["boundary"]:]
    assert paid and all(item["seed"] == TASKS[0]["seed"] for item in paid)
    assert not {batch["probe"]["probe_id"] for batch in old["batches"]} & {
        item["probe"].probe_id for item in paid}
    assert all(source == target for source, target in stream["projections"])
    for resource in RESOURCES:
        assert current["costs"][resource] == old["costs"][resource] + current["visit_costs"][resource]
        expected = sum(item["research"]["costs"][resource] for item in final["results"])
        if resource == "compute_cost":
            expected += final["screening_compute_cost"]
        else:
            assert current["visit_costs"][resource] == sum(batch[resource] for batch in paid)
        assert final["costs"][resource] == expected
    for resource, cap in zip(RESOURCES, (608, 7296, 510000)):
        assert final["costs"][resource] <= cap
    session = UnifiedResearchSession(state=serialized(final), lab_factory=stream["factory"])
    count = len(stream["purchases"])
    assert serialized(session.run()) == final
    assert len(stream["purchases"]) == count


def test_unfunded_anonymous_resume_preserves_the_unresolved_model(resumed_anonymous_stream):
    state = resumed_anonymous_stream["queued"]
    previous = state["results"][0]["research"]
    offer, = state["pending_revisits"]
    purchases, projections, evaluations = [], [], []
    config = replace(CONFIG, observation_budget=previous["costs"]["observation_cost"],
                     compute_budget=previous["costs"]["compute_cost"])
    with recorded_factory(purchases, projections, evaluations)(TASKS[0]) as lab:
        report = UnifiedResearchAgent(config,
            measurement_budget=previous["costs"]["measurement_cost"],
            library=KnowledgeLibrary.from_dict(state["knowledge"])).run(lab,
                previous=previous, continuation_candidate=deepcopy(offer["candidate"]))
    assert report["status"] == "unresolved" and report["program"] == previous["program"]
    assert report["object_research"] == previous["object_research"]
    assert not report["object_research"]["complete"] and report["learned_object_schema"] is None
    assert report["batches"] == previous["batches"]
    assert not purchases and not evaluations
    assert all(report["visit_costs"][resource] == 0 for resource in RESOURCES)


def test_response_agreement_does_not_accept_a_refuted_joint_schema(resumed_anonymous_stream, monkeypatch):
    from hegel_machine.cognition import object_induction_continuation as joint

    state = resumed_anonymous_stream["queued"]
    previous = state["results"][0]["research"]
    offer, = state["pending_revisits"]
    check = joint.confirm_joint_object_roles

    def wrong_membership(agent, lab, channels, model, batches, budget, actions):
        # A competing role proposal can disagree even while the frozen
        # response happens to predict this trajectory correctly.
        pending = model["continuation"]["confirmation"]
        fields = pending["schema_before"]["fields"]
        first, second = list(fields)[:2]
        fields[first]["f"], fields[second]["f"] = fields[second]["f"], fields[first]["f"]
        return check(agent, lab, channels, model, batches, budget, actions)

    monkeypatch.setattr(joint, "confirm_joint_object_roles", wrong_membership)
    purchases, projections, evaluations = [], [], []
    with recorded_factory(purchases, projections, evaluations)(TASKS[0]) as lab:
        result = UnifiedResearchAgent(CONFIG, measurement_budget=6144,
            library=KnowledgeLibrary.from_dict(state["knowledge"])).run(lab,
                previous=previous, continuation_candidate=deepcopy(offer["candidate"]))
    events = result["actions"][len(previous["actions"]):]
    role = next(e for e in events if e["action"] == "confirm_resumed_object_roles")
    response = next(e for e in events if e["action"] == "compare_object_continuation_candidates")
    assert not role["supported"] and role["probe_id"] == response["probe_id"]
    assert response["after_loss"] <= CONFIG.minimum_improvement
    assert result["program"] == previous["program"] and result["status"] == "unresolved"
    assert result["learned_object_schema"] is None and not result["object_research"]["complete"]
    assert not evaluations


def test_unchanged_partial_induction_preserves_budget_for_a_real_successor_confirmation(
        resumed_anonymous_stream, monkeypatch):
    state = resumed_anonymous_stream["queued"]
    previous, source = [record["research"] for record in state["results"]]
    candidate = {"program": source["program"], "channels": source["selected_channels"],
        "fallback": source["fallback"], "kind": "successor_object_knowledge",
        "source_concept_ids": source["concepts_added"], "source_object_model_ids": []}
    # The learned program is already meaningful on the old paid responses.
    # There is enough compute to compare and confirm it, but not to re-infer
    # every role from the 52 paid sensor readings: that requires at least
    # 52 * (12 sensor fields + 10 controls) = 1144 compute before comparison.
    purchases, projections, evaluations = [], [], []
    config = replace(CONFIG, observation_budget=previous["costs"]["observation_cost"] + 24,
                     compute_budget=previous["costs"]["compute_cost"] + 1100)
    agent = UnifiedResearchAgent(config,
        measurement_budget=previous["costs"]["measurement_cost"] + 216,
        library=KnowledgeLibrary.from_dict(state["knowledge"]))
    instrument_refits = []
    fit = agent._fit_instruments

    def observed_fit(*args, **kwargs):
        instrument_refits.append(True)
        return fit(*args, **kwargs)

    monkeypatch.setattr(agent, "_fit_instruments", observed_fit)
    with recorded_factory(purchases, projections, evaluations)(TASKS[0]) as lab:
        rows = [row for batch in paid_view(lab, previous, candidate["channels"]) for row in batch]
        assert loss(Program.from_dict(candidate["program"]), rows, candidate["fallback"], .5) < 1e-7
        report = agent.run(lab, previous=previous, continuation_candidate=candidate)
    events = report["actions"][len(previous["actions"]):]
    induction = next(event for event in events if event["action"] == "retain_object_ambiguity")
    assert induction["new_observation_rows"] == 0 and induction["compute_cost"] == 0
    assert not instrument_refits
    assert report["object_research"]["groups"] == previous["object_research"]["groups"]
    assert not report["object_research"]["complete"]
    confirmation = next(event for event in events if event["action"] == "compare_object_continuation_candidates")
    assert any(item["probe"].probe_id == confirmation["probe_id"] for item in purchases)
    assert confirmation["after_loss"] < 1e-7
    assert report["status"] == "completed" and report["confirmation_loss"] < 1e-7
    assert report["visit_costs"]["compute_cost"] <= 1100
    assert 0 < report["visit_costs"]["observation_cost"] <= 24
    assert report["visit_costs"]["observation_cost"] == sum(item["observation_cost"] for item in purchases)
    assert not evaluations


def test_anonymous_bank_evidence_and_real_affine_frontiers_do_not_require_a_cross_bank_projection(resumed_anonymous_stream):
    stream = resumed_anonymous_stream
    previous = deepcopy(stream["first"]["results"][0]["research"])
    state = previous["object_continuation_state"]
    retained = [item for item in state["finalists"] if item.get("terms")]
    assert retained
    assert all(isinstance(term, dict) for item in retained for term in item["terms"])
    assert all(isinstance(term, Program) for item in retained
               for term in decode_entry(item)["terms"])
    with AnonymousObjectLab(**{key: value for key, value in TASKS[0].items()
            if key not in {"environment", "research_limits"}}) as lab:
        extra = lab.random_probe(81291, length=12, name="independent_coarse_bank")
        coarse = lab.acquire(extra, ("coarse",))
        previous["batches"].append({"probe": asdict(extra), "channels": ["coarse"],
            "observations": [asdict(row) for row in coarse.observations],
            "transitions": deepcopy(coarse.transitions)})
        records = paid_object_records(lab, previous["batches"], ("fine",))
        fine_batches = [batch for batch in previous["batches"] if batch["channels"] == ["fine"]]
        assert len(records) == sum(len(batch["observations"]) for batch in fine_batches)
        by_probe = {batch["probe"]["probe_id"]: batch for batch in fine_batches}
        assert {row["probe_id"] for row in records} == set(by_probe)
        for row in records:
            sequence = by_probe[row["probe_id"]]["probe"]["sequence"]
            assert row["controls"] == sequence[row["position"]]
            assert row["history"] == sequence[:row["position"]]
            assert set(row["before"]) == set(lab.spec_for(("fine",)).feature_names)
        budget = SearchBudget(50000)
        coarse_entry = next(entry for entry in compile_claims(lab.spec_for(("coarse",)),
            coarse.observations, budget) if entry["kind"] == "affine")
        state["candidate_frontiers"].append({"features": list(lab.spec_for(("coarse",)).feature_names),
                                           "entries": [encode_entry(coarse_entry)]})

        def commands():
            return paid_view(lab, previous, ())

        def pairs(view):
            result = []
            for batch in previous["batches"]:
                if tuple(view) not in lab.projectable_views(batch["channels"]):
                    continue
                rows = paid_view(lab, {"batches": [batch]}, view)[0]
                controls = paid_view(lab, {"batches": [batch]}, ())[0]
                result.append(list(zip(controls, rows)))
            return result

        agent = UnifiedResearchAgent(CONFIG, library=KnowledgeLibrary.from_dict(stream["first"]["knowledge"]))
        actions = []
        result = resume_entries(agent, lab, previous, None, paired_for=pairs,
            command_batches=commands, budget=budget, actions=actions)
        compared = actions[-1]["alternatives"]
        assert compared and any(item["channels"] == ["fine"] for item in compared)
        assert not any(item["channels"] == ["coarse"] for item in compared)
        assert result["training"]
