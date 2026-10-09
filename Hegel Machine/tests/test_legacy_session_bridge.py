"""Current paid records must participate in native revisits, not be repurchased."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.legacy_research_bridge import LEGACY_KNOWLEDGE_PATH, METHODS_PATH, run_legacy_research
from hegel_machine.cognition.legacy_session_bridge import LegacyInquirySession
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


def episodes(path, *, measured=True, consumer_budget=384):
    result = []
    for index, role in enumerate(("consumer", "source")):
        agent = TemporalOperatorAgent(None, path / role, arm="code_react", source_task=False,
            config=InquiryLLMConfig(observation_budget=consumer_budget if not index else 384,
                compute_budget=360000, confirmation_rows=24, execution_seconds=120))
        laboratory = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="finite2")
        result.append({"agent": agent, "laboratory": laboratory, "available_concepts": []})
    rng = Random(71)
    result[0]["agent"]._query({"actions": [{"kind": "trajectory",
        "sequence": [{"u": rng.uniform(-1, 1)} for _ in range(24)],
        "channels": ["meter"] if measured else []} for _ in range(3)]}, result[0]["laboratory"])
    source = result[1]
    learned = run_legacy_research(source["agent"], {}, source["laboratory"])
    source["research"] = json.loads(source["agent"].files[learned["report_path"]])
    return result, json.loads(source["agent"].files[LEGACY_KNOWLEDGE_PATH])


def test_native_revisit_uses_current_paid_rows_and_returns_real_provisional_value(tmp_path):
    items, knowledge = episodes(tmp_path)
    paid = deepcopy(items[0]["agent"].observations)
    before = [deepcopy(item["agent"].costs) for item in items]
    session = LegacyInquirySession(items, knowledge)
    feedback = items[0]["agent"]._execute("run", {"operator": "legacy_revisit", "max_tasks": 1},
        items[0]["laboratory"])
    result = session.to_dict()
    assert feedback["revisits"][0]["task_position"] == 0
    assert "legacy_revisit" in items[0]["agent"].common_interface
    items[0]["agent"]._prepare_program_state()
    public_state = json.loads((items[0]["agent"].workspace / "research_state.json").read_text())
    assert public_state["legacy_session"]["tasks"][0]["status"] == "completed"
    target = result["results"][0]["research"]
    assert target["status"] == "completed"
    assert len(result["revisits"]) == 1
    visit = result["revisits"][0]
    assert visit["task_position"] == 0 and visit["source_task_position"] == 1
    assert visit["visit_costs"]["observation_cost"] == 24
    assert target["continuation"]["paid_rows_reused"] == 72
    assert items[0]["agent"].observations[:3] == paid
    assert items[0]["agent"].costs["observation_cost"] == 96
    assert items[1]["agent"].costs == before[1]
    assert items[0]["agent"].costs["compute_cost"] - before[0]["compute_cost"] == (
        visit["visit_costs"]["compute_cost"] + result["screening_compute_cost"])
    returns = target["knowledge_return"]["returns"]
    assert returns and all(row["value_status"] == "positive" for row in returns)
    assert all(row["confirmation_status"] == "provisional" for row in returns)
    assert result["inquiry_exports"][0]["model_path"] is None
    assert "only control" in result["inquiry_exports"][0]["export_reason"]
    assert not items[0]["agent"].confirmations
    assert items[0]["agent"].model is None
    saved = json.loads(items[0]["agent"].files[LEGACY_KNOWLEDGE_PATH])
    assert saved == result["knowledge"]
    assert saved["concepts"][0]["production_methods"]
    before_confirm = deepcopy(items[0]["agent"].costs)
    confirmed = items[0]["agent"]._execute("run",
        {"operator": "legacy_return_confirmation", "length": 24}, items[0]["laboratory"])
    assert confirmed["status"] == "settled" and confirmed["confirmed_program"]
    assert confirmed["before_loss"] > 1e-4 and confirmed["after_loss"] < 1e-7
    assert confirmed["new_costs"] == {
        "observation_cost": 24, "measurement_cost": 72, "compute_cost": 48, "queries": 1}
    assert items[0]["agent"].costs["compute_cost"] - before_confirm["compute_cost"] == 48
    assert all(row["confirmation_status"] == "confirmed"
               for row in confirmed["knowledge_return"]["returns"])
    updated = session.to_dict()
    assert updated["results"][0]["research"]["costs"]["observation_cost"] == 120
    outcomes = [outcome for method in updated["knowledge"]["research_workflows"]["methods"]
                for outcome in method.get("knowledge_outcomes", [])]
    assert any(receipt["confirmation_status"] == "confirmed"
               for outcome in outcomes for receipt in outcome.get("source_returns", []))
    assert not items[0]["agent"].confirmations and items[0]["agent"].model is None


def test_native_pool_cannot_spend_reserved_confirmation_of_one_agent(tmp_path):
    items, knowledge = episodes(tmp_path, consumer_budget=119)
    before = deepcopy(items[0]["agent"].observations)
    result = LegacyInquirySession(items, knowledge).run()
    assert not result["revisits"]
    assert result["results"][0]["research"]["status"] == "unresolved"
    assert items[0]["agent"].observations == before
    assert result["results"][1]["research"]["status"] == "completed"


def test_saved_session_restores_without_recharging_or_replaying_evidence(tmp_path):
    items, knowledge = episodes(tmp_path)
    bridge = LegacyInquirySession(items, knowledge)
    state = json.loads(json.dumps(bridge.run()))
    before = [deepcopy(item["agent"].costs) for item in items]
    restored = LegacyInquirySession(items, state=state)
    result = restored.run()
    assert len(result["revisits"]) == 1
    assert [item["agent"].costs for item in items] == before
    assert result["knowledge"] == state["knowledge"]
    assert all(item["agent"].legacy_session is restored for item in items)


@pytest.mark.parametrize("confirmation_sign", [1., -1.])
def test_control_only_research_attributes_actual_call_and_measured_confirmation(
        tmp_path, monkeypatch, confirmation_sign):
    items, knowledge = episodes(tmp_path, measured=False)
    session = LegacyInquirySession(items, knowledge)
    result = session.run()
    target = result["results"][0]["research"]
    assert target["status"] == "completed" and not target["selected_channels"]
    assert result["inquiry_exports"][0]["model_path"]
    # The scheduler offered an approximate lag template. Research actually
    # found the complete learned response with a reconstructed instrument.
    # Crediting the queued template would reward a different computation.
    incoming = result["revisits"][0]["candidate"]["transfer_evidence"]
    assert not incoming[0]["template_id"].endswith(":abstract:0")
    program = Program.from_dict(target["program"])
    assert program.op == "call" and program.definition.definition_id.endswith(":abstract:0:definition")
    returned = target["knowledge_return"]["returns"]
    retained = [row for row in returned if row["retained"]]
    assert len(retained) == 1
    receipt = retained[0]
    source = program.definition.source_concept_id
    saved = KnowledgeLibrary.from_dict(knowledge)
    binding = saved.definition_bindings[program.definition.definition_id]
    assert receipt["source_id"] == source
    assert receipt["template_id"] == binding["template_id"]
    assert receipt["template_id"] != incoming[0]["template_id"]
    mapping = {role: Program.from_dict(value) if isinstance(value, dict) else value
               for role, value in receipt["mapping"].items()}
    assert any(isinstance(value, Program) and value.history_depth > 0 for value in mapping.values())
    assert program.children == tuple(bind_measurement_arguments(argument, mapping)
                                     for argument in binding["arguments"])
    assert receipt["retained_normalized_gain"] > 0
    assert receipt["confirmation_status"] == "provisional"
    # Native comparisons must be recomputable from the newly bought batch;
    # fitting the paid prefix alone cannot establish a return.
    by_probe = {batch["probe"]["probe_id"]: batch for batch in target["batches"]}
    comparisons = [event for event in target["actions"]
                   if event["action"] == "compare_successor_knowledge_on_fresh_response"
                   and event["program"] == target["program"]]
    assert comparisons
    for event in comparisons:
        rows = [Observation(row["features"], row["target"], tuple(row["history"]))
                for row in by_probe[event["probe_id"]]["observations"]]
        assert event["after_loss"] == pytest.approx(loss(program, rows, event["fallback"], .5))
        assert event["after_loss"] < event["before_loss"]
    assert not items[0]["agent"].confirmations and items[0]["agent"].model is None
    assert items[0]["agent"].costs["observation_cost"] <= 384 - 24
    agent, lab = items[0]["agent"], items[0]["laboratory"]
    real_execute = lab.execute

    def fresh_response(action):
        record = real_execute(action)
        for row in record["rows"]:
            row["target"] *= confirmation_sign
        return record

    monkeypatch.setattr(lab, "execute", fresh_response)
    before = deepcopy(agent.costs)
    submitted = items[0]["agent"]._execute("finish",
        {"model": result["inquiry_exports"][0]["model_path"]}, items[0]["laboratory"])
    assert submitted["fresh_confirmation"]["passed"] == (confirmation_sign > 0)
    state = session.to_dict()
    feedback = state["inquiry_confirmations"][-1]["feedback"]
    assert feedback["status"] == "settled" and feedback["compute_cost"] == 48
    assert agent.costs["observation_cost"] - before["observation_cost"] == 24
    fresh = [row for record in submitted["fresh_confirmation"]["observations"]
             for row in lab.observation_rows(record)]
    assert feedback["after_loss"] == pytest.approx(loss(program, fresh, target["fallback"], .5))
    second = feedback["knowledge_return"]["returns"]
    assert len(second) == 1 and second[0]["template_id"] == receipt["template_id"]
    assert second[0]["mapping"] == receipt["mapping"]
    assert second[0]["adaptation_compute"] == 0
    assert second[0]["confirmation_status"] == ("confirmed" if confirmation_sign > 0 else "rejected")
    assert (second[0]["retained_normalized_gain"] > 0) == (confirmation_sign > 0)
    assert feedback["knowledge_return"]["method_returns"]
    imported = state["results"][0]["research"]
    assert imported["costs"]["observation_cost"] == items[0]["agent"].costs["observation_cost"]
    assert imported["costs"]["compute_cost"] + state["screening_compute_cost"] == (
        items[0]["agent"].costs["compute_cost"])


def test_paid_research_discovers_source_during_continuation_without_a_queued_offer(tmp_path):
    items, knowledge = episodes(tmp_path, measured=False)
    agent, lab = items[0]["agent"], items[0]["laboratory"]
    agent._save(LEGACY_KNOWLEDGE_PATH, json.dumps(knowledge))
    result = run_legacy_research(agent, {}, lab)
    report = json.loads(agent.files[result["report_path"]])
    assert report["status"] == "completed" and report["selected_kind"] == "callable_transfer"
    assert report["continuation"]["source_concept_ids"] == []
    assert report["continuation"]["paid_rows_reused"] == 72
    program = Program.from_dict(report["program"])
    receipt, = report["knowledge_return"]["returns"]
    assert receipt["source_id"] == program.definition.source_concept_id
    library = KnowledgeLibrary.from_dict(knowledge)
    assert receipt["template_id"] == library.definition_bindings[program.name]["template_id"]
    assert receipt["retained_normalized_gain"] > 0 and receipt["comparisons"]
    assert report["knowledge_return"]["method_returns"]


def test_bound_native_research_and_two_agent_book_updates_survive_session_save(tmp_path):
    items, knowledge = episodes(tmp_path)
    bridge = LegacyInquirySession(items, knowledge)
    first, second = [item["agent"] for item in items]
    # These inactive records exercise shared book transport; neither claims an
    # invented empirical support nor enters the subsequent research search.
    entries = [{"method_id": f"inactive_{index}", "status": "counterexample",
                "solver_strategy": "instrument_sequence"} for index in range(2)]
    first._save(METHODS_PATH, json.dumps([entries[0]]))
    assert second._instrument_methods().to_dict() == [entries[0]]
    book = second._instrument_methods().to_dict()
    book.append(entries[1])
    second._save(METHODS_PATH, json.dumps(book))
    before = deepcopy(first.costs)
    result = first._execute("run", {"operator": "legacy_research", "observation_budget": 72},
        items[0]["laboratory"])
    after = bridge.to_dict()
    assert after["results"][0]["research"]["status"] == result["legacy_status"]
    assert after["knowledge"]["program_experiment_methods"] == entries
    assert first.costs["compute_cost"] - before["compute_cost"] == result["new_costs"]["compute_cost"]
    for item in items:
        assert json.loads(item["agent"].files[LEGACY_KNOWLEDGE_PATH]) == after["knowledge"]
        assert json.loads(item["agent"].files[METHODS_PATH]) == entries
    assert after["inquiry_bridge"]["charged_research_compute"][0] == (
        after["results"][0]["research"]["costs"]["compute_cost"])


def test_live_decisions_can_read_current_state_before_first_action_and_after_query(tmp_path):
    items, knowledge = episodes(tmp_path)
    session = LegacyInquirySession(items, knowledge)
    agent, lab = items[0]["agent"], items[0]["laboratory"]
    requests = []

    class QueryThenStop:
        def next_action(self, request):
            state = json.loads((agent.workspace / "research_state.json").read_text())
            requests.append(deepcopy(request))
            assert state["legacy_session"]["tasks"] == [
                {"task_position": 0, "status": "unresolved"},
                {"task_position": 1, "status": "completed"}]
            assert state["costs"]["observation_cost"] == request["costs"]["observation_cost"]
            if len(requests) == 1:
                assert state["costs"]["observation_cost"] == 72
                return {"action": "query", "arguments": {"actions": [{"kind": "trajectory",
                    "sequence": [{"u": .25}, {"u": -.5}], "channels": ["meter"]}]},
                    "usage": {"total_tokens": 0}}
            assert state["costs"]["observation_cost"] == 74
            assert len(request["laboratory_observations"]) == 4
            return {"action": "provider_error", "error": {"message": "stop test"},
                    "usage": {"total_tokens": 0}}

    agent.provider = QueryThenStop()
    result = agent.run(lab)
    assert result["status"] == "provider_failed" and len(requests) == 2
    assert result["model"] is None and not result["fresh_confirmation"]
    assert not session.to_dict()["revisits"]


def test_native_revisit_imports_new_plain_queries_without_repurchase_or_recharge(tmp_path):
    items, knowledge = episodes(tmp_path)
    session = LegacyInquirySession(items, knowledge)
    agent, lab = items[0]["agent"], items[0]["laboratory"]
    source_costs = deepcopy(items[1]["agent"].costs)
    model_path = session.to_dict()["inquiry_exports"][0]["model_path"]
    edited_model = (agent.workspace / model_path).read_text() + "\n# user revision\n"
    agent._save(model_path, edited_model)
    rng = Random(8181)
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": ["meter"],
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]}]}, lab)
    paid = deepcopy(agent.observations)
    before = deepcopy(agent.costs)
    session.sync_paid_observations()
    assert (agent.workspace / model_path).read_text() == edited_model
    assert agent.costs == before
    state = session.run(max_tasks=1)
    report = state["results"][0]["research"]
    visit, = state["revisits"]
    assert report["continuation"]["paid_rows_reused"] == 96
    assert agent.observations[:4] == paid
    assert visit["previous"]["costs"]["observation_cost"] == 96
    assert report["costs"]["observation_cost"] == agent.costs["observation_cost"]
    assert report["costs"]["measurement_cost"] == agent.costs["measurement_cost"]
    assert agent.costs["observation_cost"] - before["observation_cost"] == (
        visit["visit_costs"]["observation_cost"])
    assert agent.costs["compute_cost"] - before["compute_cost"] == (
        visit["visit_costs"]["compute_cost"] + state["screening_compute_cost"])
    assert items[1]["agent"].costs == source_costs
    after = deepcopy(agent.costs)
    session.sync_paid_observations()
    assert agent.costs == after
    assert len(session.to_dict()["results"][0]["research"]["batches"]) == len(agent.observations)
