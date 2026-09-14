"""Paid stateful failures drive distinct boundary experiments, without answers."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget
from hegel_machine.cognition.probing import transition_probe
from hegel_machine.cognition.state_search import state_candidates


def _case(index):
    payload = json.loads((Path(__file__).parent / "fixtures" /
                          "cognition_v016_state_boundary_learning.json").read_text())
    case = next(item for item in payload["cases"] if item["source_task_index"] == index)
    rows = [Observation(item["features"], item["target"], tuple(item["history"]))
            for item in case["observations"]]
    used = [Probe(item["probe_id"], tuple(item["sequence"])) for item in case["used"]]
    environment = make_suite(payload["seed"], "stateful")[index - 53]
    return case, environment, rows, Program.from_dict(case["program"]), used


def _bracket(rows, name, threshold, state, within=None):
    relevant = sorted((row for row in rows if row.history and
                       bool(row.history[-1]["__target__"]) == state), key=lambda row: row.features[name])
    return next((left.features[name], right.features[name]) for left, right in zip(relevant, relevant[1:])
                if left.target != right.target and (
                    within[0] <= left.features[name] < right.features[name] <= within[1] if within else
                    left.features[name] <= threshold <= right.features[name]))


@pytest.mark.parametrize("renamed", [False, True])
def test_each_mature_boundary_gets_evidence_before_repeating_a_slice(renamed):
    case, environment, rows, program, used = _case(59)
    spec = environment.spec
    mapping = {name: f"control_{index}" for index, name in enumerate(spec.feature_names)} if renamed else {}
    if renamed:
        def remap(point):
            return {mapping.get(name, name): value for name, value in point.items()}
        spec = replace(spec, task_id="unseen_names", feature_names=tuple(mapping.values()),
                       feature_bounds={mapping[name]: bounds for name, bounds in spec.feature_bounds.items()})
        rows = [Observation(remap(row.features), row.target, tuple(remap(past) for past in row.history)) for row in rows]
        program = program.remap_features(mapping)
        used = [Probe(probe.probe_id, tuple(remap(point) for point in probe.sequence)) for probe in used]
    budget = SearchBudget(1500)
    probe, details = transition_probe(spec, [Candidate("retained", program, "state")], rows,
        budget=budget, remaining=22, round_number=case["source_round"], used=used)
    boundaries = details["boundary_support"]
    # The old experiment repeated the on-state boundary and used the startup
    # coordinate as a mature block, leaving the off-state bracket untouched.
    assert not any(item["previous_state"] is False for item in case["old_plan"]["boundary_support"])
    assert sum(item["previous_state"] is None for item in boundaries) == 1
    mature = [item for item in boundaries if item["previous_state"] is not None]
    assert {item["previous_state"] for item in mature[:2]} == {False, True}
    assert probe.cost == 13 and budget.spent <= budget.limit
    # The new experiment is run through the same public observation interface.
    reverse = {value: key for key, value in mapping.items()}
    actual = Probe(probe.probe_id, tuple({reverse.get(name, name): value for name, value in point.items()}
                                        for point in probe.sequence))
    fresh = environment.observe(actual)
    if renamed:
        fresh = [Observation(remap(row.features), row.target, tuple(remap(past) for past in row.history)) for row in fresh]
    assert any(row.history and not row.history[-1]["__target__"] and
               program.predict(row) != row.target for row in fresh)
    name = mapping.get("b1", "b1")
    threshold = program.children[1].children[2].children[1].value
    before = _bracket(rows, name, threshold, False)
    after = _bracket([*rows, *fresh], name, threshold, False, within=before)
    assert after[1] - after[0] <= (before[1] - before[0]) / 2 + 1e-12


def test_new_boundary_evidence_changes_the_executable_transition():
    case, environment, rows, program, used = _case(59)
    probe, _ = transition_probe(environment.spec, [Candidate("retained", program, "state")], rows,
        budget=SearchBudget(1500), remaining=22, round_number=case["source_round"], used=used)
    fresh = environment.observe(probe)
    acquired = [*rows, *fresh]
    candidates = state_candidates(acquired, budget=SearchBudget(5000), max_history=1)
    best = min(candidates, key=lambda candidate: (
        sum(candidate.program.predict(row) != row.target for row in acquired), candidate.program.size))
    assert sum(best.program.predict(row) != row.target for row in fresh) == 0
    assert sum(program.predict(row) != row.target for row in fresh) > 0
    assert best.program != program


def test_startup_choice_includes_the_real_cost_of_another_comparison():
    case, environment, rows, program, used = _case(58)
    candidates = [Candidate("retained", program, "state")]
    arguments = dict(remaining=21, round_number=case["source_round"], used=used)
    short, short_details = transition_probe(environment.spec, candidates, rows,
        budget=SearchBudget(1500), **arguments)
    joint, details = transition_probe(environment.spec, candidates, rows,
        budget=SearchBudget(1500), followup_observation_cost=12, **arguments)
    assert short.cost == 1 and short_details["startup_relative_width"] > short_details["mature_relative_width"]
    assert joint.cost == 13 and details["comparison_observation_quote"] == 12
    assert details["boundary_reduction_per_observation"] >= details["startup_reduction_per_observation"]
    assert {item["previous_state"] for item in details["boundary_support"]} == {None, False, True}
    fresh = environment.observe(joint)
    assert any(row.history and program.predict(row) != row.target for row in fresh)
    # On the same saved learning state a nearly exhausted observation budget
    # can still query startup; no incomplete mature preparation is inserted.
    limited, limited_details = transition_probe(environment.spec, candidates, rows,
        budget=SearchBudget(1500), remaining=4, round_number=case["source_round"], used=used,
        followup_observation_cost=12)
    assert limited.cost == 1
    assert all(item["previous_state"] is None for item in limited_details["boundary_support"])
