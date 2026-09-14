"""Repair a learned initial regime without replacing successful state dynamics."""

import json
from pathlib import Path

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget
from hegel_machine.cognition.state_search import startup_revision_candidates
from hegel_machine.cognition.workspace import ResearchWorkspace


def retained_startup_workspace(fixture_name="cognition_s08_startup_learning.json"):
    """Small copy of the paid s08 evidence before its second research visit."""
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                         fixture_name).read_text())
    observations, reference, queried = [], [], []
    for index, trajectory in enumerate(fixture["trajectories"]):
        history, rows, sequence = [], [], []
        for *values, target in trajectory["rows"]:
            features = dict(zip(fixture["feature_names"], values))
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
            sequence.append(features)
        observations.extend(rows)
        if index in fixture["reference_trajectories"]:
            reference.extend(rows)
        queried.append(Probe(trajectory["probe_id"], tuple(sequence)))
    incumbent = Candidate("retained_joint_state", Program.from_dict(fixture["program"]), "composition")
    parent = Candidate("prior_joint_state", Program.from_dict(fixture["parent_program"]), "composition")
    return ResearchWorkspace(fixture["task_id"], observations=observations, reference=reference,
        queried=queried, incumbent=incumbent, parent=parent, bank=[incumbent, parent],
        evidence_champion=incumbent, fallback=fixture["fallback"], initial_loss=fixture["initial_loss"],
        rounds_completed=fixture["rounds_completed"])


def test_paid_s08_starts_repair_the_joint_model_without_forgetting_mature_counterexamples():
    workspace = retained_startup_workspace()
    old = workspace.incumbent.program
    assert len(workspace.observations) == 74 and len(workspace.reference) == 40
    assert old.minimum_history_depth == 0
    assert loss(old, workspace.observations, workspace.fallback, .5) == 1 / 74
    budget = SearchBudget(256)
    candidates = startup_revision_candidates([old], workspace.observations, budget=budget)
    assert candidates
    revised = candidates[0].program
    assert revised.children[0] is old.children[0]
    assert revised.children[1] is old.children[1]
    assert revised.children[2].children[0] == old.children[2].children[0]
    assert revised.children[2].required_features() == {"x", "z"}
    assert loss(revised, workspace.observations, workspace.fallback, .5) == 0.
    assert loss(revised, workspace.reference, workspace.fallback, .5) == 0.
    assert budget.spent <= budget.limit
    assert startup_revision_candidates([revised], workspace.observations, budget=SearchBudget(256)) == []


def test_startup_revision_charges_discovery_and_predictions_and_funds_a_complete_first_fit():
    workspace = retained_startup_workspace()
    old, rows = workspace.incumbent.program, workspace.observations
    startup_rows = sum(not row.history for row in rows)
    short = SearchBudget(len(rows) + startup_rows - 1)
    assert startup_revision_candidates([old], rows, budget=short) == []
    assert short.spent == len(rows)
    complete = SearchBudget(len(rows) + 2 * startup_rows + 1)
    candidates = startup_revision_candidates([old], rows, budget=complete, max_candidates=1)
    assert len(candidates) == 1
    assert complete.spent == complete.limit
    assert loss(candidates[0].program, rows, workspace.fallback, .5) == 0.


def test_local_revision_reuses_bound_callable_coordinates_and_preserves_the_mature_subtree():
    definition = Definition("joint_drive", ("a", "b"), Program("add", children=(
        Program("arg", name="a"), Program("mul", children=(
            Program("const", value=.5), Program("arg", name="b"))))))
    coordinate = Program("call", name=definition.definition_id, definition=definition,
                         children=(Program("feature", name="u"), Program("feature", name="v")))

    def threshold(value):
        return Program("ge", children=(coordinate, Program("const", value=value)))

    mature = Program("if", children=(Program("ge", children=(
        Program("lag", name="__target__", lag=1), Program("const", value=.5))),
        threshold(-.7), threshold(.7)))
    old = Program("if", children=(Program("available", lag=1), mature, threshold(-.5)))
    rows = []
    for trajectory, start in enumerate((-1., -.4, .6, 1.)):
        history, active = [], False
        for index, drive in enumerate((start, 1.2, -.2, -1.2, .2)):
            v = .2 if (trajectory + index) % 2 else -.2
            features = {"u": drive - .5 * v, "v": v}
            active = drive >= .25 if index == 0 else drive >= (-.7 if active else .7)
            target = float(active)
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
    assert loss(old, rows, .5, .5) > 0.
    candidates = startup_revision_candidates([old], rows, budget=SearchBudget(160))
    assert candidates
    revised = candidates[0].program
    assert revised.children[1] is mature
    assert revised.children[2].children[0] is coordinate
    assert loss(revised, rows, .5, .5) == 0.
    assert loss(Program.from_dict(revised.to_dict()), rows, .5, .5) == 0.


def test_three_competing_mature_programs_share_the_first_startup_fit_and_candidate_slots():
    workspace = retained_startup_workspace()
    old, rows = workspace.incumbent.program, workspace.observations
    guard, mature, startup = old.children
    mature_programs = [mature, workspace.parent.program.children[1], mature.children[2]]
    programs = [Program("if", children=(guard, branch, startup)) for branch in mature_programs]
    startup_count = sum(not row.history for row in rows)
    # A single partition, one shared old prediction pass, one shared fit, and
    # three candidate comparisons suffice even when the input repeats a model.
    budget = SearchBudget(len(rows) + 2 * startup_count + 3)
    candidates = startup_revision_candidates([programs[0], *programs], rows,
                                             budget=budget, max_candidates=3)
    assert len(candidates) == 3 and budget.spent == budget.limit
    assert [candidate.program.children[1] for candidate in candidates] == mature_programs
    startup_rows = [row for row in rows if not row.history]
    assert all(loss(candidate.program, startup_rows, workspace.fallback, .5) == 0.
               for candidate in candidates)


def test_paid_s09_competing_mature_answer_receives_its_missing_startup_revision():
    workspace = retained_startup_workspace("cognition_s09_competing_startup_learning.json")
    rows, reference = workspace.observations, workspace.reference
    incumbent, competitor = workspace.incumbent.program, workspace.parent.program
    assert len(rows) == 124 and len(reference) == 80
    assert loss(incumbent, rows, workspace.fallback, .5) == 1 / 124
    assert loss(competitor, rows, workspace.fallback, .5) == 1 / 124
    assert loss(incumbent, reference, workspace.fallback, .5) == 1 / 80
    assert loss(competitor, reference, workspace.fallback, .5) == 1 / 80
    budget = SearchBudget(500)
    candidates = startup_revision_candidates([incumbent, competitor], rows, budget=budget)
    corrected = next(candidate.program for candidate in candidates
                     if candidate.program.children[1] is competitor.children[1]
                     and loss(candidate.program, rows, workspace.fallback, .5) == 0.)
    assert corrected.children[0] is competitor.children[0]
    assert corrected.children[2].children[0] == competitor.children[2].children[0]
    assert loss(corrected, reference, workspace.fallback, .5) == 0.
    assert budget.spent < 500
