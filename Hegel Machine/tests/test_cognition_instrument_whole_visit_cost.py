"""The previously priced root07 revisit includes its paid stopping decision."""

from copy import deepcopy
from dataclasses import replace
import gzip
import importlib
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.condition_marginal_value import (
    CONDITION_FOLLOWUP_ACTION, condition_followup_policy,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, ResearchConfig
from hegel_machine.cognition.unified_continuation import paid_view
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def test_real_instrument_visit_records_its_tail_once_without_recrediting_gain(monkeypatch):
    path = Path(__file__).parent / "fixtures/instrument_whole_visit_root07.json.gz"
    case = json.loads(gzip.decompress(path.read_bytes()))
    previous = case["report"]
    for batch in previous["batches"]:
        history = []
        for row in batch["observations"]:
            row["history"] = list(history)
            history.append({**row["features"], "__target__": row["target"]})
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    factory = importlib.import_module("run_observed_condition_decoding_study").laboratory
    offer = case["offer"]
    assert offer["relative_gain"] == max(item["relative_gain"] for item in case["queue_priorities"])
    limits = {key: previous["costs"][key] + value for key, value in case["allocation"].items()}
    config = replace(ResearchConfig(**case["config"]), seed=case["task"]["seed"],
        observation_budget=limits["observation_cost"], compute_budget=limits["compute_cost"])
    library = KnowledgeLibrary.from_dict(case["knowledge"])
    with factory(case["task"]) as laboratory:
        channels = tuple(previous["selected_channels"])
        rows = [row for batch in paid_view(laboratory, previous, channels) for row in batch]
        policy = condition_followup_policy(laboratory.spec_for(channels), rows,
            {"candidate": Candidate("retained", Program.from_dict(previous["program"]), "retained")},
            library.action_experience)
        context = policy.action_context(CONDITION_FOLLOWUP_ACTION)
        before = deepcopy(policy.action_record(CONDITION_FOLLOWUP_ACTION))
        agent = UnifiedResearchAgent(config, library=library,
            measurement_budget=limits["measurement_cost"])
        report = agent.run(laboratory, previous=previous, continuation_candidate=offer["candidate"])

    record = library.action_experience[context][CONDITION_FOLLOWUP_ACTION]
    assert record["instrument_cycles"] == before["instrument_cycles"]
    assert len(record["instrument_revisits"]) == len(before.get("instrument_revisits", [])) + 1
    cycle = record["instrument_revisits"][-1]
    actual = report["visit_costs"]
    expected = case["recorded_execution"]
    assert actual["compute_cost"] == expected["visit_compute_cost"] == 36979
    assert actual["observation_cost"] == actual["measurement_cost"] == 48
    for resource in ("observation_cost", "measurement_cost", "compute_cost"):
        assert cycle[resource] == actual[resource]
        assert report["costs"][resource] <= limits[resource]
    assert cycle["screening_compute_cost"] == expected["paid_screening_compute_cost"] == 14221
    events = report["actions"][len(previous["actions"]):]
    paired, = [item for item in events if item["action"] == "compare_method_followup_on_fresh_response"]
    tail, = [item for item in events if item["action"] == "retain_unconfirmed_method_followup"]
    assert paired["before_loss"] == paired["after_loss"]
    assert tail["before_loss"] is None and tail["after_loss"] is None
    assert tail["stage_costs"]["observation_cost"] == tail["stage_costs"]["measurement_cost"] == 0
    first_compute = paired["stage_costs"]["compute_cost"] - cycle["screening_compute_cost"]
    assert first_compute == expected["first_paired_compute_cost"] == 23387
    assert cycle["compute_cost"] - first_compute == expected["unscored_tail_compute_cost"] == 13592
    assert tail["stage_costs"]["compute_cost"] == expected["unscored_tail_compute_cost"]
    assert record["trials"] == before["trials"] + 1
    assert record["unscored_attempts"] == before["unscored_attempts"] + 1
    assert record["expected_gain"] == before["expected_gain"] == 0.
    for resource in ("observation", "measurement", "compute"):
        assert (record[f"mean_{resource}_cost"] * record["trials"]
            - before[f"mean_{resource}_cost"] * before["trials"]) == pytest.approx(
                paired["stage_costs"][f"{resource}_cost"])
        assert (record.get(f"unscored_{resource}_cost", 0.)
            - before.get(f"unscored_{resource}_cost", 0.)) == tail["stage_costs"][f"{resource}_cost"]
    # Session-wide screening happens after this visit returns. Its recorded
    # 12052 compute is outside this action, just as the old screen is sunk.
    assert cycle["compute_cost"] != actual["compute_cost"] + expected["subsequent_session_screen_compute_cost"]
    assert report["batches"][:len(previous["batches"])] == previous["batches"]
