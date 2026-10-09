"""A queued method consumes a real target-binding failure before buying again."""

from copy import deepcopy
from dataclasses import replace
import gzip
import json
from pathlib import Path

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, TaskSpec
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_method_joint_mismatch import joint_case, joint_factory


def test_json_queued_failed_draft_binding_is_withdrawn_without_purchase(
        joint_case, joint_factory):
    path = Path(__file__).parent / "fixtures/draft_binding_counterexample_root10.json.gz"
    paid = json.loads(gzip.decompress(path.read_bytes()))
    candidate = deepcopy(paid["entry"])
    library = KnowledgeLibrary(research_drafts=[paid["draft"]])
    source_program = deepcopy(library.research_drafts[0]["research_program"])
    event = library.record_transfer_counterexample(TaskSpec(**paid["spec"]),
        Program.from_dict(candidate["program"]), (),
        source_draft_ids=(candidate["research_draft_id"],),
        fallback=candidate["fallback"], evidence=paid["evidence"],
        probe_id=paid["probe_id"])
    assert event["source_draft_ids"] == [candidate["research_draft_id"]]

    # The consumer needs the actual paid ledger, not another replay of those
    # observations. No fitting or prediction takes place on this path.
    costs = {key: value + paid["visit_costs"][key]
        for key, value in joint_case["report"]["costs"].items()}
    previous = {"task_id": paid["spec"]["task_id"], "status": "unresolved",
        "costs": costs, "program": deepcopy(joint_case["report"]["program"])}
    config = replace(ResearchConfig(**joint_case["config"]),
        observation_budget=costs["observation_cost"] + 72,
        compute_budget=costs["compute_cost"] + 240000)
    session = UnifiedResearchSession(config, tasks=[joint_case["task"]],
        library=library, measurement_budget=costs["measurement_cost"] + 72)
    session.state["results"] = [{"task_position": 0, "research": previous}]
    session.state["pending_revisits"] = [{"task_position": 0,
        "source_task_position": None, "candidate": candidate,
        "basis": "new_partial_method_distinguishing_experiment", "relative_gain": 1.,
        "minimum_resources": {"observation_cost": 48,
            "measurement_cost": 48, "compute_cost": 120000}}]
    opened = []

    def factory(arguments):
        laboratory = joint_factory(arguments)
        opened.append(laboratory)
        return laboratory

    state = json.loads(json.dumps(session.to_dict()))
    resumed = UnifiedResearchSession(state=state, lab_factory=factory)
    before = resumed._costs()
    assert not resumed._perform_revisit(resumed.state["pending_revisits"].pop())
    assert len(opened) == 1 and not opened[0].acquisitions
    assert resumed._costs() == before
    assert resumed.state["results"][0]["research"] == state["results"][0]["research"]
    assert not resumed.state["revisits"]
    assert resumed.state["scheduler_events"][-1]["action"] == "withdraw_refuted_revisit"
    assert resumed.library.research_drafts[0]["research_program"] == source_program
    assert resumed.library.research_drafts[0]["status"] == "unresolved"
