"""Keep the event-only control intact and retain failed-source cohort slots."""

from copy import deepcopy
import json
from random import Random

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from scripts.run_python_advice_ablation import RecordingAdviceAgent
from scripts.run_python_pending_advice_study import (
    EventOnlyAdviceAgent, protocol, summarize, target_jobs,
)


def test_source_failures_keep_all_registered_targets_and_actual_source_costs(tmp_path):
    plan = protocol()
    costs = {"observation_cost": 72, "measurement_cost": 0, "compute_cost": 72,
             "python_execution_seconds": 1., "model_calls": 1, "reported_tokens": 42}
    sources = [{"case": case, "candidate": None, "episode": {"training": {
        "status": "model_call_budget_exhausted"}, "costs": deepcopy(costs), "known_tokens": 42,
        "input_tokens": 40, "cached_input_tokens": 20, "missing_usage_calls": 0}}
        for case in plan["cases"]]
    jobs, skipped = target_jobs(sources, tmp_path, plan)
    assert jobs == [] and len(skipped) == 16
    assert len({item["path"] for item in skipped}) == 16
    summary = summarize(sources, [], skipped, plan)
    assert summary["registered_sources"] == 4 and summary["registered_targets"] == 16
    assert summary["source_costs"]["observation_cost"] == 288
    assert summary["actual_source_plus_all_target_costs"]["model_calls"] == 4
    assert summary["known_tokens"] == 168
    for condition in summary["conditions"].values():
        assert condition["registered_targets"] == condition["source_blocked_targets"] == 8
        assert condition["attempted_targets"] == 0
        assert condition["source_plus_target_costs"]["observation_cost"] == 288
    # Source admission has no terminal field: actual fresh publication alone
    # opens both kinds and both conditions; all other failed slots remain.
    sources[0]["candidate"] = {"candidate_id": "actual_published_source"}
    jobs, skipped = target_jobs(sources, tmp_path, plan)
    assert len(jobs) == 4 and len(skipped) == 12
    assert {job["condition"] for job in jobs} == {"persistent", "event_only"}
    assert {job["kind"] for job in jobs} == set(plan["target_kinds"])


def test_event_only_preserves_original_advice_without_pending_feedback_or_state(tmp_path):
    agents = []
    for name, cls in (("persistent", RecordingAdviceAgent), ("event_only", EventOnlyAdviceAgent)):
        lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
            response_seed=910061, instrument_mode="recurrent")
        agent = cls(None, tmp_path / name, arm="code_react", source_task=False,
            config=InquiryLLMConfig(observation_budget=384, compute_budget=360000,
                                   confirmation_rows=24, execution_seconds=120.))
        agent._advice_learning = True
        rng = Random(71)
        agent._execute("query", {"actions": [{"kind": "trajectory", "channels": ["meter"],
            "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]} for _ in range(2)]}, lab)
        original = agent._execute("run", {"operator": "instrument", "path": "instrument.json",
                                          "records": [0]}, lab)
        assert "research_advice" in original
        before = deepcopy(agent.costs)
        incidental = agent._execute("run", {"code": "value = 1\n"}, lab)
        assert incidental["returncode"] == 0
        assert all(agent.costs[key] == before[key] for key in
                   ("observation_cost", "measurement_cost", "compute_cost", "model_calls"))
        agent._prepare_program_state()
        state = json.loads((agent.workspace / "research_state.json").read_text())
        agents.append((agent, lab, original, incidental, state))
    persistent, event_only = agents
    assert persistent[0].common_interface == event_only[0].common_interface
    assert persistent[2]["research_advice"] == event_only[2]["research_advice"]
    assert "research_advice" in persistent[3] and "research_advice" in persistent[4]
    assert "research_advice" not in event_only[3] and "research_advice" not in event_only[4]
    assert persistent[1].costs == event_only[1].costs
    assert persistent[0]._advice_ledger and event_only[0]._advice_ledger
    assert len(persistent[0]._advice_ledger) == len(event_only[0]._advice_ledger) + 1
