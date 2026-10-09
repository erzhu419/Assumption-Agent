"""Outstanding workflow demand comes from paid reports, before new research."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


@pytest.fixture(scope="module")
def partial_session():
    config = ResearchConfig(observation_budget=192, compute_budget=120000,
        minimum_improvement=1e-7, max_candidates=16, active_probes=False, seed=11811)
    tasks = [{"task_index": 8, "seed": 11811}, {"task_index": 10, "seed": 99991}]
    session = UnifiedResearchSession(config, tasks=tasks)
    report = UnifiedResearchAgent(replace(config, compute_budget=512),
        library=session.library).run(UnknownTaskLab(**tasks[0]))
    assert report["status"] == "unresolved"
    session.state["results"].append({"task_position": 0,
        "available_concepts": [], "research": report})
    return session.to_dict()


def test_demand_uses_paid_predecessor_without_opening_future_or_acquiring_data(partial_session):
    opened = []

    class ReadPaidOnly(UnknownTaskLab):
        def acquire(self, *args, **kwargs):
            raise AssertionError("demand extraction bought observations")

        def evaluation_rows(self, *args, **kwargs):
            raise AssertionError("demand extraction consulted terminal outcomes")

    def factory(item):
        assert item["seed"] == 11811, "unvisited task inspected"
        opened.append(item["seed"])
        return ReadPaidOnly(**item)

    session = UnifiedResearchSession(state=partial_session, lab_factory=factory)
    before = session.to_dict()
    demands = session._workflow_demands()
    assert opened == [11811]
    assert len(demands) == 1 and demands[0]["task_position"] == 0
    report = before["results"][0]["research"]
    selected = set(report["selected_channels"])
    assert selected, "This partial measured task should exercise view-specific demand."
    # Count the readings actually saved for this representation. Command-only
    # observations remain paid evidence, but contain no purchased sensor value
    # and cannot describe the selected measured view's research demand.
    saved_rows = [row for batch in report["batches"] for row in batch["observations"]]
    measured_rows = [row for row in saved_rows if selected <= row["features"].keys()]
    assert 0 < len(measured_rows) < len(saved_rows)
    assert demands[0]["channels"] == report["selected_channels"]
    assert demands[0]["paid_rows"] == len(measured_rows)
    assert len(saved_rows) == report["costs"]["observation_cost"]
    assert demands[0]["condition"]["response_signature"] is not None
    assert not {"task_index", "seed", "task_id", "program"} & demands[0].keys()
    assert session.to_dict() == before
    restored = UnifiedResearchSession(state=json.loads(json.dumps(before)), lab_factory=factory)
    assert restored._workflow_demands() == demands
    assert restored._workflow_demands(exclude_position=0) == []


def test_completed_and_unvisited_reports_are_not_outstanding_demand(partial_session):
    state = deepcopy(partial_session)
    state["results"][0]["research"]["status"] = "completed"

    def no_need_to_open(item):
        raise AssertionError("no unfinished paid task requires a laboratory")

    session = UnifiedResearchSession(state=state, lab_factory=no_need_to_open)
    assert session._workflow_demands() == []
