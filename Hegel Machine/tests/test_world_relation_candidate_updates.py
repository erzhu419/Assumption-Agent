"""Incremental models and prospective outcomes through the numerical researcher."""

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_relation_research import research_relation


X = Program("feature", name="x")
SQUARE = Program("mul", children=(X, X))


def row(x, y, outcome):
    return {"features": {"x": x}, "target": y, "evidence": [0, outcome],
            "conditioning_refs": [0], "outcome_refs": [outcome]}


def public(count):
    return [{"index": index} for index in range(count)]


def test_reused_conditioning_supports_new_response_only_after_selection():
    binding = {"rows": [row(0, 0, 1), row(1, 1, 2)],
               "hypotheses": [X.to_dict(), SQUARE.to_dict()]}
    first = research_relation(binding, public(3))
    binding["rows"].append(row(2, 4, 3))
    selected = research_relation(binding, public(4), first)
    assert selected["status"] == "inconclusive"
    assert selected["selected_program"] == SQUARE.to_dict()
    assert selected["validation_evidence"] == [0, 3]
    assert selected["confirmation_evidence"] == []
    binding["rows"].append(row(3, 9, 4))
    confirmed = research_relation(binding, public(5), selected)
    assert confirmed["status"] == "supported"
    assert confirmed["confirmation_evidence"] == [0, 4]
    assert confirmed["fit_evidence"] == [0, 1, 2]
    assert confirmed["candidates"][0]["status"] == "refuted"


def test_added_family_preserves_supported_old_model_but_needs_own_confirmation():
    # The new cubic matches the paid x=0,1,2 rows and differs beyond them.
    minus_one = Program("sub", children=(X, Program("const", value=1)))
    minus_two = Program("sub", children=(X, Program("const", value=2)))
    cubic = Program("add", children=(X, Program("mul", children=(
        X, Program("mul", children=(minus_one, minus_two)))))).simplified()
    binding = {"rows": [row(0, 0, 1), row(1, 1, 2)], "hypotheses": [X.to_dict()]}
    first = research_relation(binding, public(3))
    binding["rows"].append(row(2, 2, 3))
    supported = research_relation(binding, public(4), first)
    assert supported["status"] == "supported"
    old_after = supported["confirmation_after"]

    binding["candidate_additions"] = [cubic.to_dict(), {"program": cubic.to_dict()}]
    added = research_relation(binding, public(4), supported)
    assert len(added["candidates"]) == 2
    assert added["status"] == "inconclusive"
    old, new = added["candidates"]
    assert old["confirmation_after"] == old_after
    assert old["supported_evidence"] == [0, 3]
    assert new["fit_evidence"] == [0, 1, 2, 3]
    assert new["proposed_after"] == [0, 1, 2, 3]
    assert "confirmation_after" not in new

    # A repeated addition neither duplicates the candidate nor shifts its clock.
    repeated = research_relation(binding, public(5), added)
    assert len(repeated["candidates"]) == 2
    assert repeated["candidates"][1]["proposed_after"] == new["proposed_after"]
    binding["rows"].append(row(3, 9, 4))
    selected = research_relation(binding, public(5), repeated)
    assert selected["status"] == "inconclusive"
    assert selected["selected_program"] == cubic.to_dict()
    assert selected["confirmation_after"] == [0, 1, 2, 3, 4]
    assert selected["confirmation_after"] != old_after
    rejected = next(item for item in selected["rejected_candidates"]
                    if item["program"] == X.to_dict())
    assert rejected["supported_evidence"] == [0, 3]
    assert rejected["confirmation_mse"] > 0
    binding["rows"].append(row(4, 28, 5))
    confirmed = research_relation(binding, public(6), selected)
    assert confirmed["status"] == "supported"
    assert confirmed["confirmation_evidence"] == [0, 5]


def test_addition_screens_all_paid_rows_and_cannot_replace_selected_incumbent():
    binding = {"rows": [row(0, 0, 1), row(1, 1, 2)], "hypotheses": [X.to_dict()]}
    first = research_relation(binding, public(3))
    binding["rows"].append(row(2, 2, 3))
    old = research_relation(binding, public(4), first)
    binding["hypotheses"] = [SQUARE.to_dict()]  # Not an explicit addition.
    unchanged = research_relation(binding, public(4), old)
    assert [item["program"] for item in unchanged["candidates"]] == [X.to_dict()]
    binding["candidate_additions"] = [{"program": SQUARE.to_dict()}]
    result = research_relation(binding, public(4), unchanged)
    assert result["status"] == "supported" and result["selected_program"] == X.to_dict()
    added = result["candidates"][1]
    assert added["status"] == "refuted" and added["fit_mse"] > 0
    assert added["fit_rows"] == binding["rows"]
    assert added["proposed_after"] == [0, 1, 2, 3]
    repeated = research_relation(binding, public(5), result)
    assert repeated["candidates"][1] == added


def test_already_public_outcomes_bound_later_are_not_prospective():
    binding = {"rows": [row(0, 0, 1), row(1, 1, 2)], "hypotheses": [X.to_dict()]}
    first = research_relation(binding, public(4))  # Outcome 3 already public.
    binding["rows"].append(row(2, 2, 3))
    late = research_relation(binding, public(4), first)
    assert late["validation_evidence"] == []
    assert late["confirmation_evidence"] == []
    assert late["status"] == "inconclusive"
    assert late["candidates"][0]["screening_evidence"] == [0, 3]
    binding["rows"].append(row(3, 3, 4))
    fresh = research_relation(binding, public(5), late)
    assert fresh["status"] == "supported"
    assert fresh["confirmation_evidence"] == [0, 4]


@pytest.mark.parametrize("changes", [
    {"conditioning_refs": [99]}, {"outcome_refs": []}, {"outcome_refs": [99]},
    {"conditioning_refs": []}, {"outcome_refs": [True]},
])
def test_conditional_roles_cannot_drop_or_invent_supporting_records(changes):
    invalid = {**row(1, 1, 1), **changes}
    with pytest.raises(ValueError, match="refs"):
        research_relation({"rows": [invalid], "hypotheses": [X.to_dict()]}, public(2))


def test_parent_research_adopts_new_program_without_inheriting_old_confirmation(tmp_path):
    from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
    from test_discoveryworld_agent import LIMITS, ScriptedProvider
    from test_discoveryworld_research import BINDING, PARENT, RelationLab

    initial = BINDING.replace("[feature, {'op': 'mul', 'children': [feature, feature]}]", "[feature]")
    revised = initial.replace("'probes':", "'candidate_additions': [{'op': 'mul', 'children': [feature, feature]}], 'probes':")
    (tmp_path / "initial.py").write_text(initial)
    (tmp_path / "revised.py").write_text(revised)
    (tmp_path / "parent.py").write_text(PARENT)
    declaration = {"id": "relation", "parent_id": "task", "question": "Which response law?"}
    provider = ScriptedProvider([
        ("query", {"actions": [{"action": "READ", "arg1": x} for x in (0, 1)],
                   "research_question": {"id": "task", "question": "Tune the response", "parent_id": None}}),
        ("run", {"operator": "research_update", "path": "initial.py", "question": declaration}),
        ("query", {"question": {"id": "relation"}, "actions": [{"action": "READ", "arg1": 2}]}),
        ("run", {"operator": "research", "path": "revised.py", "question": declaration}),
    ])
    lab = RelationLab(square=True)
    agent = DiscoveryWorldAgent(provider, tmp_path, arm="hegel", limits=LIMITS)
    result = agent.run(lab)
    assert result["score"]["completedSuccessfully"]
    assert lab.actions == [{"action": "READ", "arg1": x} for x in (0, 1, 2, 3)] + [
        {"action": "TUNE", "arg1": 16}]
    relation = agent.relation_research["relation"]
    assert relation["status"] == "supported"
    assert relation["confirmation_after"] == [0, 1, 2]
    assert relation["confirmation_evidence"] == [3]
    assert relation["candidates"][0]["status"] == "refuted"
    assert relation["candidates"][1]["fit_evidence"] == [0, 1, 2]


def test_repeated_rejected_addition_does_not_restart_the_same_refit():
    binding = {"rows": [row(0, 0, 1), row(1, 1, 2)],
               "hypotheses": [X.to_dict()], "basis": [X.to_dict()]}
    first = research_relation(binding, public(3))
    binding["rows"].append(row(2, 4, 3))
    refitted = research_relation(binding, public(4), first)
    assert refitted["status"] == "refuted"
    binding["candidate_additions"] = [refitted["candidates"][0]["program"]]
    repeated = research_relation(binding, public(5), refitted)
    assert repeated["candidates"] == refitted["candidates"]
    assert repeated["compute_cost"] == 0


def test_late_bound_old_outcome_can_refute_without_becoming_confirmation():
    binding = {"rows": [row(0, 0, 1), row(1, 1, 2)],
               "hypotheses": [X.to_dict()], "basis": [X.to_dict()]}
    first = research_relation(binding, public(4))
    binding["rows"].append(row(2, 4, 3))
    result = research_relation(binding, public(4), first)
    old = result["rejected_candidates"][0]
    assert old["program"] == X.to_dict()
    assert old["validation_evidence"] == []
    assert old["confirmation_evidence"] == []
    assert old["screening_evidence"] == [0, 3]
    assert old["screening_mse"] == 4


def supported_incremental_square():
    binding = {"rows": [row(0, 0, 1), row(1, 1, 2)],
               "hypotheses": [X.to_dict()], "basis": [X.to_dict()]}
    first = research_relation(binding, public(3))
    binding["rows"].append(row(2, 4, 3))
    binding["candidate_additions"] = [SQUARE.to_dict()]
    selected = research_relation(binding, public(4), first)
    binding["rows"].append(row(3, 9, 4))
    confirmed = research_relation(binding, public(5), selected)
    assert confirmed["status"] == "supported"
    binding.pop("candidate_additions")  # Incremental adoption is already complete.
    return binding, confirmed


def test_basis_expansion_keeps_incremental_supported_program_without_resubmission():
    binding, confirmed = supported_incremental_square()
    binding["basis"].append(SQUARE.to_dict())
    expanded = research_relation(binding, public(5), confirmed)
    assert expanded["status"] == "supported"
    assert expanded["selected_program"] == SQUARE.to_dict()
    assert expanded["confirmation_after"] == confirmed["confirmation_after"]
    assert expanded["confirmation_evidence"] == confirmed["confirmation_evidence"]
    assert expanded["candidates"] == confirmed["candidates"]
    assert expanded["basis"] == binding["basis"]


def test_rebinding_confirming_row_preserves_program_history_without_old_support():
    binding, confirmed = supported_incremental_square()
    # Correct the extraction of a previous confirmation, not a fitting row.
    binding["rows"][-1]["target"] = 11
    correction = Program("mul", children=(Program("const", value=1 / 3),
        Program("mul", children=(X, Program("mul", children=(
            Program("sub", children=(X, Program("const", value=1))),
            Program("sub", children=(X, Program("const", value=2)))))))))
    revised_program = Program("add", children=(SQUARE, correction)).simplified()
    binding["hypotheses"] = [revised_program.to_dict()]
    rebound = research_relation(binding, public(5), confirmed)
    assert rebound["status"] == "inconclusive"
    assert rebound["selected_program"] == revised_program.to_dict()
    assert rebound["confirmation_after"] == list(range(5))
    assert rebound["confirmation_evidence"] == []
    old = next(item for item in rebound["candidates"] if item["program"] == SQUARE.to_dict())
    assert old["status"] == "refuted" and old["fit_mse"] > 0
    assert "supported_evidence" not in old
    history = rebound["binding_history"][-1]
    earlier = next(item for item in history["candidates"] if item["program"] == SQUARE.to_dict())
    assert earlier["supported_evidence"] == [0, 4]
    assert history["bound_rows"][-1]["target"] == 9
    binding["rows"].append(row(4, 24, 5))
    final = research_relation(binding, public(6), rebound)
    assert final["status"] == "supported"
    assert final["confirmation_evidence"] == [0, 5]
    assert final["binding_history"] == rebound["binding_history"]
