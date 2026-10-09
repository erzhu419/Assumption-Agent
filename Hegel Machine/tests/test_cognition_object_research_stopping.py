"""Paid object rows gate response recovery before fresh confirmation."""

from dataclasses import replace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_research_actions import ObjectResearchActions
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


CONFIG = ResearchConfig(
    observation_budget=128,
    compute_budget=50000,
    minimum_improvement=1e-7,
    seed=71,
)


def _candidate(name, value):
    return {
        "candidate": Candidate(name, Program("const", value=value), "test response"),
        "kind": "structural:test",
        "training_loss": 0.0,
        "validation_loss": 0.0,
    }


@pytest.fixture
def action_harness():
    lab = UnifiedObjectLab(objects=2, readout="focus", seed=71)
    probes = [lab.random_probe(701, length=8), lab.random_probe(702, length=8)]
    paired = []
    for probe in probes:
        batch = lab.acquire(probe, ("fine",))
        history, pairs = [], []
        for point, row in zip(probe.sequence, batch.observations):
            pairs.append((Observation(dict(point), row.target, tuple(history)), row))
            history.append({**point, "__target__": row.target})
        paired.append(pairs)

    def paired_for(channels):
        if tuple(channels) == ("fine",):
            return paired
        if tuple(channels) == ("coarse",):
            return [
                [
                    (command, measured)
                    for command, measured in zip(
                        (item[0] for item in batch),
                        lab.project_rows([item[1] for item in batch], ("fine",), ("coarse",)),
                    )
                ]
                for batch in paired
            ]
        return []

    def view_rows(rows, source, target):
        if not target:
            return rows
        return lab.project_rows(rows, source, target) if tuple(source) != tuple(target) else rows

    budget = SearchBudget(50000)

    def remaining():
        return {
            "observation_remaining": 128,
            "measurement_remaining": 10000,
            "compute_remaining": budget.remaining,
        }

    agent = UnifiedResearchAgent(replace(CONFIG, seed=71), measurement_budget=10000)
    acquired = []

    def acquire(*args, **kwargs):
        acquired.append((args, kwargs))
        return None

    actions = []
    object_actions = ObjectResearchActions(
        agent,
        lab,
        acquire=acquire,
        paired_for=paired_for,
        command_batches=lambda: [],
        view_rows=view_rows,
        transitions_for=lambda channels, rows=None: [],
        budget=budget,
        actions=actions,
        transfer=[],
        graph=None,
        remaining=remaining,
        length=8,
    )
    return object_actions, paired, actions, acquired


def _proposal(entry, view=("fine",)):
    rows = []
    return entry, view, rows, rows, [entry]


@pytest.mark.parametrize("offset", [0., 1.])
def test_same_or_worse_paid_response_is_rejected_before_confirmation(action_harness, offset):
    object_actions, paired, actions, acquired = action_harness
    paid = [row for batch in paired for _, row in batch]
    mean = sum(row.target for row in paid) / len(paid)
    parent = _candidate("parent", mean)
    worse = _candidate("worse", mean + offset)
    proposal = _proposal(worse)

    comparison = object_actions._compare_paid(proposal, parent, ("coarse",))

    assert comparison["status"] == "no_improvement"
    assert comparison["rows"] == len(paid)
    assert comparison["after_loss"] >= comparison["before_loss"]
    object_actions._structure = lambda view, current, current_channels: _proposal(worse, view)
    result = object_actions.propose(parent, channels=("coarse",))
    assert result["selected"] is None and object_actions.pending is None
    rejected = next(item for item in actions if item["action"] == "propose_object_research")
    assert rejected["status"] == "no_candidate"
    assert rejected["unscored_feedback"]["no_candidate_attempts"] == 1
    # The comparison uses the common fine acquisition and its coarse projection;
    # no new laboratory acquisition is authorized by this gate.
    assert not acquired


def test_improving_changed_view_enters_pending_fresh_confirmation(action_harness):
    object_actions, paired, actions, acquired = action_harness
    paid = [row for batch in paired for _, row in batch]
    mean = sum(row.target for row in paid) / len(paid)
    parent = _candidate("parent", mean + 1.0)
    improved = _candidate("improved", mean)
    object_actions._structure = lambda view, current, current_channels: _proposal(improved, view)

    result = object_actions.propose(parent, channels=("coarse",))

    assert result["selected"] is not None
    assert object_actions.pending is not None
    proposal = next(item for item in actions if item["action"] == "propose_object_research")
    assert proposal["status"] == "requires_fresh_confirmation"
    assert proposal["paid_comparison"]["status"] == "improved"
    assert proposal["paid_comparison"]["after_loss"] < proposal["paid_comparison"]["before_loss"]
    assert not acquired
    assert object_actions.confirmation_channels(("fine",)) == ("fine",)
