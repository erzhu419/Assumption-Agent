"""Short startup evidence must not erase transferable mature dynamics."""

from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.selection import rank_candidates


def task(task_id, names):
    return TaskSpec(task_id, "predict measured output", names, (), ())


def theory(names=("x", "z"), history_coefficient=-1.3):
    current = Program("add", children=(
        Program("mul", children=(Program("const", value=2), Program("feature", name=names[0]))),
        Program("const", value=.5)))
    mature = Program("add", children=(current, Program("mul", children=(
        Program("const", value=history_coefficient), Program("lag", name=names[1], lag=4)))))
    return Program("if", children=(Program("available", lag=4), mature, current))


def trajectory(program, names, length, offset=0):
    history, rows = [], []
    for step in range(length):
        features = {names[0]: ((step * 7 + offset * 3) % 17 - 8) / 3,
                    names[1]: ((step * 3 + offset * 7) % 13 - 6) / 2}
        row = Observation(features, 0, tuple(history))
        target = program.predict(row)
        rows.append(Observation(features, target, tuple(history)))
        history.append({**features, "__target__": target})
    return rows


def learned_library():
    library = KnowledgeLibrary()
    source = theory()
    source_id, _ = library.learn(Candidate("learned", source, "representation"), task("source", ("x", "z")),
                                 trajectory(source, ("x", "z"), 12))
    return library, source_id


def rank(candidates, evidence, fresh):
    return rank_candidates(candidates, evidence, fresh, fallback=initial_prediction(evidence, .5),
                           q=.5, budget=SearchBudget(10000), complexity_weight=.001, fresh=fresh)


def test_short_initial_trajectories_retain_original_complete_history_theory():
    library, source_id = learned_library()
    target = task("renamed", ("u", "v"))
    transferred = theory(target.feature_names)
    initial = [row for offset in (0, 1) for row in trajectory(transferred, target.feature_names, 4, offset)]
    budget = SearchBudget(600)
    candidates = library.propose(target, initial, budget)

    original = next(candidate for candidate in candidates if candidate.program == transferred)
    fitted = [candidate for candidate in candidates if ":fit:" in candidate.candidate_id]
    assert original.memory_id == source_id
    assert original.parents == (source_id, source_id + ":abstract:0")
    assert original.operation == "abstract_reuse"
    assert loss(original.program, initial, 0, .5) == 0
    assert any(loss(candidate.program, initial, 0, .5) < 1e-12 for candidate in fitted)
    assert {key: library._pending_transfers[original.candidate_id][key]
            for key in ("template_id", "mapping", "parameter_mode")} == {
        "template_id": source_id + ":abstract:0", "mapping": {"role_0": "u", "role_1": "v"},
        "parameter_mode": "preserved"}
    assert library.last_search["parameter_compute"] > 0
    assert library.last_search["compute_cost"] == budget.spent <= budget.limit
    assert budget.spent == sum(library.last_search[key] for key in
                              ("parameter_compute", "screen_compute", "fit_compute"))

    # Only new measurements activating the mature branch select the old law.
    fresh = trajectory(transferred, target.feature_names, 16, 3)
    winner = rank(candidates, initial, fresh)[0].candidate
    assert winner.program == transferred
    assert loss(winner.program, fresh, 0, .5) == 0
    assert all(loss(candidate.program, fresh, 0, .5) > .01 for candidate in fitted)
    assert library.concepts[0].successful_tasks == ["source"]

    old_path = library.propose(target, initial, SearchBudget(600), preserve_parameters=False)
    assert all(candidate.program != transferred for candidate in old_path)
    assert library.last_search["parameter_compute"] == 0


def test_new_mature_coefficients_reject_preserved_parameters_without_revoking_source():
    library, source_id = learned_library()
    target = task("changed_coefficients", ("u", "v"))
    actual, old = theory(target.feature_names, 2.7), theory(target.feature_names)
    initial = [row for offset in (0, 1) for row in trajectory(actual, target.feature_names, 4, offset)]
    candidates = library.propose(target, initial, SearchBudget(600))
    original = next(candidate for candidate in candidates if candidate.program == old)
    assert loss(original.program, initial, 0, .5) == 0

    measured = trajectory(actual, target.feature_names, 16, 3)
    updated = library.propose(target, initial + measured, SearchBudget(2000))
    unseen = trajectory(actual, target.feature_names, 16, 5)
    winner = rank([original, *updated], initial + measured, unseen)[0].candidate
    old_loss = loss(original.program, unseen, 0, .5)
    assert old_loss > 1
    assert loss(winner.program, unseen, 0, .5) < 1e-12
    assert winner.program != original.program
    assert winner.memory_id == source_id

    library.record_transfer_result(original, target, accepted=False, loss=old_loss, baseline_loss=10)
    assert library.abstracts[0].transfer_history[-1]["parameter_mode"] == "preserved"
    assert library.abstracts[0].transfer_history[-1]["accepted"] is False
    assert library.concepts[0].status == "available"
    assert library.concepts[0].successful_tasks == ["source"]


def test_original_parameter_screen_is_bounded_and_shares_the_search_budget():
    library = KnowledgeLibrary()
    for index in range(12):
        program = theory(history_coefficient=index + 1)
        library.learn(Candidate(str(index), program, "representation"), task(str(index), ("x", "z")))
    target = task("renamed", ("u", "v"))
    initial = trajectory(theory(target.feature_names), target.feature_names, 4)
    budget = SearchBudget(120)
    candidates = library.propose(target, initial, budget)
    details = library.last_search
    assert 0 < details["parameter_compute"] <= (budget.limit - len(initial)) // 4
    assert details["parameter_trials"] <= 8
    assert 0 < details["preserved_parameter_candidates"] <= 4
    assert any(":fit:" in candidate.candidate_id for candidate in candidates)
    assert details["compute_cost"] == budget.spent <= budget.limit
