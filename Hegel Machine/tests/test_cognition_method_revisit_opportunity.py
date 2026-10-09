"""A successor's partial method buys a real test of an earlier paid failure."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.knowledge_changes import knowledge_changes, knowledge_snapshot
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.numeric_method_revisits import offer_numeric_method_experiment
from hegel_machine.cognition.research_programs import _Fitter
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _terms
from hegel_machine.cognition.unified_session import UnifiedResearchSession as CompleteResearchSession

from test_cognition_decoded_observed_conditions import DecodedObservedLab, random_probe
from test_cognition_numeric_draft_replay import broad_source


CONFIG = ResearchConfig(seed=4802, observation_budget=320, compute_budget=320000,
    max_rounds=2, max_candidates=8, minimum_improvement=1e-7)
TASKS = [dict(seed=4801, width=5, placement="outside_window"),
         dict(seed=4803, width=3, placement="inside_window")]
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


class UnifiedResearchSession(CompleteResearchSession):
    """Isolate the method path; instrument competition has its own session tests."""

    def _offer_instrument_conditions(self, pending):
        return pending


def persisted(value):
    return json.loads(json.dumps(value))


class MethodLab(DecodedObservedLab):
    def __init__(self, purchases, **task):
        super().__init__(**task)
        self.spec = replace(self.spec, task_id=f"method_task_{task['seed']}")
        self.purchases = purchases

    def acquire(self, probe, channels=()):
        batch = super().acquire(probe, channels)
        self.purchases.append(dict(task_id=self.spec.task_id, probe=probe,
            channels=tuple(channels), rows=batch.observations,
            observation_cost=batch.observation_cost, measurement_cost=batch.measurement_cost))
        return batch

    def restore(self, acquisitions):
        purchases, self.purchases = self.purchases, []
        super().restore(acquisitions)
        self.purchases = purchases

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        pass


@pytest.fixture(scope="module")
def real_method_stream():
    purchases = []
    factory = lambda task: MethodLab(purchases, **task)
    session = UnifiedResearchSession(CONFIG, tasks=TASKS, measurement_budget=320, lab_factory=factory)
    with factory(TASKS[0]) as lab:
        old = UnifiedResearchAgent(replace(CONFIG, observation_budget=192, compute_budget=6000),
            measurement_budget=192).run(lab)
    session.state["results"].append(dict(task_position=0, available_concepts=[], research=old))
    paused = persisted(session.to_dict())
    queued = persisted(UnifiedResearchSession(state=paused, lab_factory=factory).run(max_tasks=1))
    before = len(purchases)
    final = persisted(UnifiedResearchSession(state=queued, lab_factory=factory).run(max_tasks=1))
    return dict(paused=paused, queued=queued, final=final, purchases=purchases,
                before=before, factory=factory)


def test_unfinished_successor_method_reopens_a_real_failure_without_supported_concept(real_method_stream):
    stream = real_method_stream
    old = stream["paused"]["results"][0]["research"]
    queued = stream["queued"]
    assert old["status"] == "unresolved" and old["confirmation_loss"] > 1e-7
    assert old["selected_channels"] == ["meter"]
    assert not queued["knowledge"]["concepts"]
    assert queued["knowledge"]["research_drafts"]
    offer, = queued["pending_revisits"]
    event = queued["scheduler_events"][-1]
    assert offer["basis"] == "new_partial_method_distinguishing_experiment"
    assert offer["task_position"] == 0 and offer["source_task_position"] == 1
    assert event["scheduled"] and event["source_concept_ids"] == []
    assert event["method_opportunity"]["paid_counterexample_loss"] > 1e-7
    # The complete responses are still mismatched; their extrapolated gap is
    # not evidence of value. The visit is funded by actual paid-data progress.
    assert event["method_opportunity"]["predicted_separation"] == 0.
    assert (event["method_opportunity"]["candidate_paid_loss"]
            < event["method_opportunity"]["paid_counterexample_loss"])
    quote = offer["minimum_resources"]
    assert quote["observation_cost"] == quote["measurement_cost"] == 72
    assert quote["compute_cost"] >= (offer["candidate"]["method_opportunity"]["duration_compute_quote"]
        + offer["candidate"]["method_opportunity"]["response_compute_quote"])
    # Each unvisited task keeps its whole allocation, even with a pending visit.
    with_reserve = deepcopy(queued)
    with_reserve["tasks"].append(dict(seed=4805, width=7, placement="inside_window"))
    reserved = UnifiedResearchSession(state=with_reserve, lab_factory=stream["factory"])
    assert reserved._remaining() == UnifiedResearchSession(state=queued)._remaining()


def test_restored_offer_buys_the_plan_and_confirms_an_improved_original_response(real_method_stream):
    stream = real_method_stream
    queued, final = stream["queued"], stream["final"]
    old = queued["results"][0]["research"]
    current = final["results"][0]["research"]
    offer, = queued["pending_revisits"]
    visit, = final["revisits"]
    assert visit["basis"] == offer["basis"]
    assert current["batches"][:len(old["batches"])] == old["batches"]
    new_purchases = stream["purchases"][stream["before"]:]
    events = current["actions"][len(old["actions"]):]
    execution = next(item for item in events if item["action"] == "execute_successor_method_experiment")
    plan = offer["candidate"]["method_experiment"]
    purchased = next(item for item in new_purchases if item["probe"].probe_id == plan["probe"]["probe_id"])
    assert list(purchased["probe"].sequence) == plan["probe"]["sequence"]
    assert execution["source_draft_ids"] == plan["source_draft_ids"]
    assert execution["observation_rows"] == len(purchased["rows"]) == 24
    assert execution["incumbent_loss"] == pytest.approx(loss(
        Program.from_dict(old["program"]), purchased["rows"], old["fallback"], .5))
    proposal = offer["candidate"]
    assert execution["proposed_loss"] == pytest.approx(loss(
        Program.from_dict(proposal["program"]), purchased["rows"], proposal["fallback"], .5))
    acquisition = next(item for item in events if item["action"] == "acquire"
        and item["probe_id"] == plan["probe"]["probe_id"])
    assert acquisition["purpose"] == "successor_partial_method_experiment"
    assert all(set(point) == set(purchased["rows"][0].features) - {"meter"}
               for point in purchased["probe"].sequence)
    assert current["confirmation_loss"] < 1e-6
    assert current["confirmation_loss"] < old["confirmation_loss"]
    assert any(item["action"] == "validate_continued_program" for item in events)
    confirmation_ids = {item["probe_id"] for item in events if item["action"] == "validate_continued_program"}
    assert plan["probe"]["probe_id"] not in confirmation_ids
    fresh_lab = MethodLab([], seed=4807, width=5, placement="outside_window")
    fresh = fresh_lab.acquire(random_probe(fresh_lab.spec, 4808, 64), ("meter",)).observations
    future_loss = loss(Program.from_dict(current["program"]), fresh, current["fallback"], .5)
    assert future_loss < loss(Program.from_dict(old["program"]), fresh, old["fallback"], .5)
    # This fresh trajectory crosses the retained threshold bracket: point
    # confirmation does not settle its unmeasured response ambiguity.
    assert current["condition_uncertainty"]["status"] == "unresolved"
    assert current["status"] == "unresolved" and not current["concepts_added"]
    for resource in RESOURCES:
        assert current["costs"][resource] == old["costs"][resource] + current["visit_costs"][resource]
        expected = sum(item["research"]["costs"][resource] for item in final["results"])
        if resource == "compute_cost":
            expected += final["screening_compute_cost"]
        assert final["costs"][resource] == expected
        assert final["costs"][resource] <= 2 * (320000 if resource == "compute_cost" else 320)
    for resource in ("observation_cost", "measurement_cost"):
        assert current["visit_costs"][resource] == sum(item[resource] for item in new_purchases)


def test_json_replay_and_accounting_do_not_recycle_an_attempted_method(real_method_stream):
    stream = real_method_stream
    final = stream["final"]
    session = UnifiedResearchSession(state=final, lab_factory=stream["factory"])
    before = len(stream["purchases"])
    assert persisted(session.run()) == final
    assert len(stream["purchases"]) == before
    library = session.library
    snapshot = knowledge_snapshot(library)
    for draft in library.research_drafts:
        draft["outcomes"].append(dict(task_id="another_replay", status="current_counterexample"))
        draft["research_program"]["search_compute"] += 1
    assert knowledge_changes(snapshot, knowledge_snapshot(library)) == []
    repeated = [item for event in stream["queued"]["scheduler_events"]
                for item in event["knowledge_changes"] if item["kind"] == "draft"]
    screening_cost = session.state["screening_compute_cost"]
    session._offer_new_knowledge(1, [], changes=repeated)
    assert session.state["screening_compute_cost"] == screening_cost
    assert not session.state["pending_revisits"] and len(stream["purchases"]) == before


def test_unsupported_method_without_paid_gain_does_not_quote_its_forecast_as_value(real_method_stream):
    old = deepcopy(real_method_stream["paused"]["results"][0]["research"])
    lab = MethodLab([], **TASKS[0])
    rows = [row for batch in paid_view(lab, old, ("meter",)) for row in batch]
    # A richer smooth history already explains these purchased responses better
    # than a cheap replay of the successor's incompatible three-step boundary.
    fitting = SearchBudget(10000)
    fit = _Fitter(rows, fitting, .5).fit([Program("window_sum", lag=5, children=(term,))
        for term in _terms(lab.spec_for(("meter",)).feature_names, quadratic=True)])
    assert fit is not None and fit["loss"] > CONFIG.minimum_improvement
    old.update(program=fit["program"].to_dict(), selection_loss=fit["loss"],
               fallback=initial_prediction(rows, .5))
    old["costs"]["compute_cost"] += fitting.spent
    library, record = broad_source()
    original = persisted(library.to_dict())
    budget = SearchBudget(4000)
    offer = offer_numeric_method_experiment(UnifiedResearchAgent(CONFIG, library=library,
        measurement_budget=320), lab, old, {record["draft_id"]}, budget)
    assert offer is None
    assert 0 < budget.spent <= budget.limit
    assert not lab.purchases and persisted(library.to_dict()) == original
    unsupported = deepcopy(old)
    unsupported.update(selection_loss=0., confirmation_loss=0.)
    no_work = SearchBudget(4000)
    assert offer_numeric_method_experiment(UnifiedResearchAgent(CONFIG, library=library),
        lab, unsupported, {record["draft_id"]}, no_work) is None
    assert no_work.spent == 0


def test_shared_pool_funds_the_frozen_method_and_confirms_its_repair_without_rescreening(real_method_stream):
    purchases = []
    factory = lambda task: MethodLab(purchases, **task)
    tasks = deepcopy(TASKS)
    # Fund the real first visits while leaving enough to screen a method, but
    # less than its complete 72-row experiment/refit/confirmation purchase.
    paid_rows = sum(item["research"]["costs"]["observation_cost"]
                    for item in real_method_stream["queued"]["results"])
    tasks[0]["research_limits"] = {"observation_budget": paid_rows + 52 - CONFIG.observation_budget}
    tasks.append(dict(seed=4811, width=3, placement="inside_window", smooth=True))
    session = UnifiedResearchSession(CONFIG, tasks=tasks, measurement_budget=320, lab_factory=factory)
    with factory(TASKS[0]) as lab:
        old = UnifiedResearchAgent(replace(CONFIG, observation_budget=192, compute_budget=6000),
            measurement_budget=192).run(lab)
    session.state["results"].append(dict(task_position=0, available_concepts=[], research=old))
    deferred = persisted(session.run(max_tasks=1))
    attempt, = deferred["method_revisit_attempts"]
    offer = attempt["deferred_offer"]
    assert not deferred["pending_revisits"]
    assert session._remaining()["observation_cost"] == 52 < offer["minimum_resources"]["observation_cost"]
    # The next apparatus also supplies a supported law. Both are proposed from
    # paid evidence; the partial method's quote now uses the expanded pool.
    resumed = UnifiedResearchSession(state=deferred, lab_factory=factory)
    queued = persisted(resumed.run(max_tasks=1))
    actual, = queued["pending_revisits"]
    assert actual["candidate"]["kind"] == "successor_partial_method"
    assert actual["relative_gain"] > offer["relative_gain"]
    assert actual["candidate"] == offer["candidate"]
    assert actual["old_evidence_gain"] == offer["old_evidence_gain"]
    assert actual["minimum_resources"] == offer["minimum_resources"]
    assert actual["pricing_resources"] == resumed._remaining()
    assert queued["results"][0] == deferred["results"][0]
    assert "deferred_offer" not in queued["method_revisit_attempts"][0]
    funded, = [item for item in queued["scheduler_events"]
               if item["action"] == "fund_deferred_method_experiment"]
    assert funded["scheduled"] and funded["compute_cost"] == 0
    assert funded["method_opportunity"] == offer["candidate"]["method_opportunity"]
    assert offer["candidate"]["method_opportunity"]["planning_compute_cost"] > 0
    assert all(item["task_id"] != "method_task_4801" for item in purchases[len(old["batches"]):])
    # Rank alone is insufficient: the selected plan must actually repair the
    # paid failure and survive a separate newly purchased confirmation.
    selected = resumed.state["pending_revisits"].pop()
    assert resumed._perform_revisit(selected)
    current = resumed.state["results"][0]["research"]
    visit = resumed.state["revisits"][-1]
    paid_before = queued["results"][0]["research"]
    assert current["batches"][:len(paid_before["batches"])] == paid_before["batches"]
    fresh = current["continuation"]["method_fresh_comparisons"][-1]
    assert fresh["before_loss"] > CONFIG.minimum_improvement
    assert fresh["after_loss"] < CONFIG.minimum_improvement
    assert fresh["probe_id"] != actual["candidate"]["method_experiment"]["probe"]["probe_id"]
    assert current["confirmation_loss"] < old["confirmation_loss"]
    for resource in RESOURCES:
        assert current["costs"][resource] == old["costs"][resource] + visit["visit_costs"][resource]
        assert current["costs"][resource] <= visit["resource_limits"][resource]
        assert resumed._remaining()[resource] >= 0
