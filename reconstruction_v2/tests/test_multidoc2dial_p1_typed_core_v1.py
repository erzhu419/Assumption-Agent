from __future__ import annotations

from dataclasses import fields
import inspect

import pytest

from assumption_agent.benchmarks import (
    multidoc2dial_p1_typed_core_v1 as core,
)


def _history(
    *,
    suffix: str = "",
) -> tuple[core.DialogueTurn, ...]:
    return (
        core.DialogueTurn(
            "user",
            f"I need help with veterans health enrollment eligibility {suffix}",
        ),
        core.DialogueTurn(
            "agent",
            "We need establish service period and disability status.",
        ),
        core.DialogueTurn(
            "user",
            "I served during gulf war and have a disability.",
        ),
        core.DialogueTurn(
            "agent",
            "If eligible, complete enrollment application and provide "
            "discharge records.",
        ),
        core.DialogueTurn("user", "What should I do next?"),
    )


def _passages(
    *,
    suffix: str = "",
) -> tuple[core.Passage, ...]:
    rows = (
        (
            "General portal",
            "Overview",
            ("Portal",),
            "What should I do next using the general portal navigation.",
        ),
        (
            "General portal",
            "Contact",
            ("Portal",),
            "What should I do next contact support center.",
        ),
        (
            "General portal",
            "Account",
            ("Portal",),
            "What should I do next create an online account.",
        ),
        (
            "General portal",
            "Status",
            ("Portal",),
            "What should I do next check generic status.",
        ),
        (
            "General portal",
            "FAQ",
            ("Portal",),
            "What should I do next common questions.",
        ),
        (
            "Veterans health",
            "Eligibility conditions",
            ("Enrollment", "Eligibility"),
            "Service period and disability status determine enrollment "
            "eligibility.",
        ),
        (
            "Veterans health",
            "Application steps",
            ("Enrollment", "Apply"),
            "Complete the enrollment application after eligibility is "
            "established.",
        ),
        (
            "Veterans health",
            "Required records",
            ("Enrollment", "Apply"),
            "Provide discharge records and disability documentation.",
        ),
        (
            "Veterans health",
            "Service requirements",
            ("Enrollment", "Eligibility"),
            "Gulf war service may satisfy enrollment condition.",
        ),
        (
            "Veterans health",
            "Submission methods",
            ("Enrollment", "Apply"),
            "Submit the application online, by mail, or in person.",
        ),
        (
            "Veterans health",
            "After submission",
            ("Enrollment", "Apply"),
            "Track enrollment application after submitting records.",
        ),
        (
            "Benefits",
            "Unrelated",
            ("Benefits",),
            "Pension payment calendar and tax forms.",
        ),
        (
            "Education",
            "Unrelated",
            ("Education",),
            "School certification application.",
        ),
        (
            "Burial",
            "Unrelated",
            ("Burial",),
            "Memorial eligibility documentation.",
        ),
        (
            "General",
            "Other",
            ("Other",),
            "Website privacy and accessibility.",
        ),
    )
    return tuple(
        core.Passage(
            ordinal,
            title + suffix,
            section,
            path,
            text,
        )
        for ordinal, (title, section, path, text) in enumerate(rows)
    )


def _scores(
    passages: tuple[core.Passage, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dense = tuple(100 - passage.ordinal for passage in passages)
    ce = tuple(200 - passage.ordinal for passage in passages)
    return dense, ce


def _slate(
    *,
    suffix: str = "",
) -> core.ActionSlate:
    passages = _passages(suffix=suffix)
    dense, ce = _scores(passages)
    return core.build_action_slate(
        _history(suffix=suffix),
        passages,
        dense,
        ce,
    )


def _outcomes(
    *,
    selected_recipe: str,
    selected_utility: int,
    e0_utility: int = 500_000,
) -> tuple[int, ...]:
    values = [400_000] * len(core.AGENT_RECIPE_IDS)
    values[core.AGENT_RECIPE_IDS.index(core.E0_RECIPE_ID)] = e0_utility
    values[core.AGENT_RECIPE_IDS.index(selected_recipe)] = selected_utility
    return tuple(values)


def test_exact_public_projection_and_label_blind_api() -> None:
    assert tuple(field.name for field in fields(core.DialogueTurn)) == (
        "role",
        "text",
    )
    assert tuple(field.name for field in fields(core.Passage)) == (
        "ordinal",
        "title",
        "section",
        "path",
        "text",
    )
    assert tuple(inspect.signature(core.build_action_slate).parameters) == (
        "history",
        "passages",
        "raw_dense_scores",
        "raw_ce_scores",
    )
    assert tuple(inspect.signature(core.make_aform_example).parameters) == (
        "slate",
        "outcome_vector",
    )
    assert tuple(field.name for field in fields(core.AFormExample)) == (
        "signature",
        "outcome_vector",
        "changed_slots_from_e0",
    )

    turn = {"role": "user", "text": "Where is the application?"}
    passage = {
        "ordinal": 7,
        "title": "Enrollment",
        "section": "Apply",
        "path": ["Health", "Enrollment"],
        "text": "Submit the form.",
    }
    assert core.turn_public_payload(
        core.turn_from_public_fields(turn)
    ) == turn
    assert core.passage_public_payload(
        core.passage_from_public_fields(passage)
    ) == passage

    for forbidden in (
        "answer",
        "da",
        "doc_id",
        "domain",
        "family",
        "id_sp",
        "qrel",
        "references",
        "split",
    ):
        with pytest.raises(
            core.MultiDoc2DialP1TypedCoreError,
            match="exact public field set",
        ):
            core.turn_from_public_fields(
                {**turn, forbidden: "forbidden"}
            )
        with pytest.raises(
            core.MultiDoc2DialP1TypedCoreError,
            match="exact public field set",
        ):
            core.passage_from_public_fields(
                {**passage, forbidden: "forbidden"}
            )


def test_normalized_query_contract_preserves_role_but_excludes_da() -> None:
    first = (
        core.DialogueTurn("user", "  ＨＥＬＬＯ\tWorld  "),
        core.DialogueTurn("agent", "Use  FORM A."),
        core.DialogueTurn("user", "NEXT step"),
    )
    second = (
        core.DialogueTurn("user", "hello world"),
        core.DialogueTurn("agent", "use form a."),
        core.DialogueTurn("user", "next   STEP"),
    )
    expected = {
        "turns": [
            {"role": "user", "text": "hello world"},
            {"role": "agent", "text": "use form a."},
            {"role": "user", "text": "next step"},
        ]
    }
    assert core.normalized_query_payload(first) == expected
    assert core.normalized_query_sha256(first) == core.stable_hash(expected)
    assert (
        core.normalized_query_sha256(first)
        == core.normalized_query_sha256(second)
    )
    assert "da" not in core.canonical_bytes(expected).decode("ascii")

    consecutive_roles = (
        core.DialogueTurn("user", "first"),
        core.DialogueTurn("user", "second"),
    )
    assert core.normalized_query_payload(consecutive_roles) == {
        "turns": [
            {"role": "user", "text": "first"},
            {"role": "user", "text": "second"},
        ]
    }


def test_actions_are_deterministic_under_input_permutation() -> None:
    history = _history()
    passages = _passages()
    dense, ce = _scores(passages)
    first = core.build_action_slate(history, passages, dense, ce)
    second = core.build_action_slate(
        history,
        tuple(reversed(passages)),
        tuple(reversed(dense)),
        tuple(reversed(ce)),
    )
    assert first == second
    audit = first.audit_payload()
    claimed = audit.pop("self_sha256")
    assert claimed == core.stable_hash(audit)
    assert audit["hipporag_is_agent_input"] is False
    assert tuple(action.recipe_id for action in first.actions) == (
        core.R0_RAW_DENSE,
        core.R1_RAW_CE,
        core.R2_HISTORY_REFERENT,
        core.R3_TOPIC_PATH_EXPANSION,
        core.R4_CONDITION_SOLUTION_COVERAGE,
        core.R5_SECTION_NEIGHBOR_CLOSURE,
        core.R6_CONSERVATIVE_TYPED_CASCADE,
    )
    assert len(
        {action.behavior_digest for action in first.actions}
    ) == len(core.RECIPE_IDS)


def test_each_typed_recipe_changes_top5_set_coverage() -> None:
    slate = _slate()
    raw_dense = set(
        slate.action(core.R0_RAW_DENSE).top5_passage_ordinals
    )
    raw_ce = set(slate.action(core.R1_RAW_CE).top5_passage_ordinals)
    assert raw_dense == raw_ce == {0, 1, 2, 3, 4}

    typed_sets = {
        recipe_id: set(
            slate.action(recipe_id).top5_passage_ordinals
        )
        for recipe_id in core.AGENT_RECIPE_IDS
    }
    assert all(values != raw_ce for values in typed_sets.values())
    assert len({frozenset(values) for values in typed_sets.values()}) == len(
        core.AGENT_RECIPE_IDS
    )
    assert 5 in typed_sets[core.R2_HISTORY_REFERENT]
    assert typed_sets[core.R3_TOPIC_PATH_EXPANSION] == {
        5,
        6,
        7,
        8,
        9,
    }
    assert {5, 6, 7} <= typed_sets[
        core.R4_CONDITION_SOLUTION_COVERAGE
    ]
    assert {5, 6, 11} <= typed_sets[
        core.R5_SECTION_NEIGHBOR_CLOSURE
    ]
    assert {5, 7, 8, 11} <= typed_sets[
        core.R6_CONSERVATIVE_TYPED_CASCADE
    ]
    for recipe_id, values in typed_sets.items():
        action = slate.action(recipe_id)
        assert len(values) == core.TOP_K
        assert len(action.selection_trace) == core.TOP_K
        assert len(action.behavior_digest) == 64


def test_history_and_structural_coordinates_have_causal_effect() -> None:
    passages = _passages()
    dense, ce = _scores(passages)
    focused = core.build_action_slate(
        _history(),
        passages,
        dense,
        ce,
    )
    unrelated_history = (
        core.DialogueTurn(
            "user",
            "I need help with a pension payment calendar.",
        ),
        core.DialogueTurn(
            "agent",
            "Review tax forms and direct deposit dates.",
        ),
        core.DialogueTurn(
            "user",
            "My monthly payment is delayed.",
        ),
        core.DialogueTurn(
            "agent",
            "Check the payment schedule and banking details.",
        ),
        core.DialogueTurn("user", "What should I do next?"),
    )
    unrelated = core.build_action_slate(
        unrelated_history,
        passages,
        dense,
        ce,
    )
    assert focused.action(
        core.R2_HISTORY_REFERENT
    ).top5_passage_ordinals != unrelated.action(
        core.R2_HISTORY_REFERENT
    ).top5_passage_ordinals

    flattened = tuple(
        core.Passage(
            passage.ordinal,
            f"Unique title {passage.ordinal}",
            f"Unique section {passage.ordinal}",
            (f"Unique path {passage.ordinal}",),
            passage.text,
        )
        for passage in passages
    )
    flat = core.build_action_slate(
        _history(),
        flattened,
        dense,
        ce,
    )
    assert focused.action(
        core.R5_SECTION_NEIGHBOR_CLOSURE
    ).top5_passage_ordinals != flat.action(
        core.R5_SECTION_NEIGHBOR_CLOSURE
    ).top5_passage_ordinals
    assert focused.action(
        core.R0_RAW_DENSE
    ).top5_passage_ordinals == flat.action(
        core.R0_RAW_DENSE
    ).top5_passage_ordinals


def test_hipporag_is_a_separate_baseline_and_cannot_leak() -> None:
    passages = _passages()
    dense, ce = _scores(passages)
    slate_before = core.build_action_slate(
        _history(),
        passages,
        dense,
        ce,
    )
    assert "hippo" not in {
        name.casefold()
        for name in inspect.signature(core.build_action_slate).parameters
    }
    first = core.build_official_hipporag_baseline(
        passages,
        tuple(range(len(passages))),
    )
    second = core.build_official_hipporag_baseline(
        passages,
        tuple(reversed(range(len(passages)))),
    )
    slate_after = core.build_action_slate(
        _history(),
        passages,
        dense,
        ce,
    )
    assert first.top5_passage_ordinals != second.top5_passage_ordinals
    assert first.behavior_digest != second.behavior_digest
    assert slate_before == slate_after
    assert core.OFFICIAL_HIPPORAG_BASELINE_ID not in core.RECIPE_IDS


def test_e0_is_fixed_and_e1_promotes_only_repeated_safe_gain() -> None:
    slates = tuple(
        _slate(suffix=f" train-{index}")
        for index in range(core.MIN_SIGNATURE_SUPPORT)
    )
    selected_recipe = core.R3_TOPIC_PATH_EXPANSION
    outcomes = _outcomes(
        selected_recipe=selected_recipe,
        selected_utility=900_000,
    )
    examples = tuple(
        core.make_aform_example(slate, outcomes) for slate in slates
    )
    model = core.fit_e1(examples)
    qualified = [rule for rule in model.rules if rule.qualified]
    assert len(qualified) == 1
    assert qualified[0].recipe_id == selected_recipe
    assert qualified[0].minimum_delta == 400_000
    assert qualified[0].regularized_mean_delta > 0
    assert model.training_stage == "A_form"

    held = _slate(suffix=" held")
    e0 = core.apply_e0(held, stage="A_hold")
    assert e0.selected_recipe_id == core.E0_RECIPE_ID
    decisions = tuple(
        core.apply_e1(model, held, stage=stage)
        for stage in core.POLICY_STAGES
    )
    assert {
        decision.selected_recipe_id for decision in decisions
    } == {selected_recipe}
    assert all(not decision.fallback_to_e0 for decision in decisions)
    assert all(
        decision.selected_recipe_id in core.AGENT_RECIPE_IDS
        for decision in decisions
    )
    payload = model.payload()
    assert payload["training_stage"] == "A_form"
    assert "family" not in payload["feature_names"]
    assert "domain" not in payload["feature_names"]
    assert "da" not in payload["feature_names"]


def test_e1_under_support_conflict_regularization_and_unseen_fallback() -> None:
    slate = _slate()
    selected_recipe = core.R3_TOPIC_PATH_EXPANSION
    strong = _outcomes(
        selected_recipe=selected_recipe,
        selected_utility=900_000,
    )
    under = core.fit_e1(
        tuple(
            core.make_aform_example(slate, strong)
            for _ in range(core.MIN_SIGNATURE_SUPPORT - 1)
        )
    )
    assert core.apply_e1(
        under,
        slate,
        stage="A_hold",
    ).fallback_to_e0

    regression = _outcomes(
        selected_recipe=selected_recipe,
        selected_utility=400_000,
    )
    conflict = core.fit_e1(
        (
            core.make_aform_example(slate, strong),
            core.make_aform_example(slate, strong),
            core.make_aform_example(slate, strong),
            core.make_aform_example(slate, regression),
        )
    )
    assert not any(rule.qualified for rule in conflict.rules)
    assert core.apply_e1(
        conflict,
        slate,
        stage="A_hold",
    ).fallback_to_e0

    tied = list(strong)
    tied[core.AGENT_RECIPE_IDS.index(selected_recipe)] = tied[
        core.AGENT_RECIPE_IDS.index(core.E0_RECIPE_ID)
    ]
    stable_with_one_tie = core.fit_e1(
        (
            core.make_aform_example(slate, strong),
            core.make_aform_example(slate, strong),
            core.make_aform_example(slate, strong),
            core.make_aform_example(slate, tuple(tied)),
        )
    )
    stable_rule = next(
        rule
        for rule in stable_with_one_tie.rules
        if rule.recipe_id == selected_recipe
    )
    assert stable_rule.positive_count == 3
    assert stable_rule.minimum_delta == 0
    assert stable_rule.qualified

    tiny = _outcomes(
        selected_recipe=selected_recipe,
        selected_utility=501_000,
    )
    regularized = core.fit_e1(
        tuple(
            core.make_aform_example(slate, tiny)
            for _ in range(core.MIN_SIGNATURE_SUPPORT)
        )
    )
    tiny_rule = next(
        rule
        for rule in regularized.rules
        if rule.recipe_id == selected_recipe
    )
    assert tiny_rule.positive_count == core.MIN_SIGNATURE_SUPPORT
    assert tiny_rule.regularized_mean_delta < 0
    assert not tiny_rule.qualified

    qualified = core.fit_e1(
        tuple(
            core.make_aform_example(slate, strong)
            for _ in range(core.MIN_SIGNATURE_SUPPORT)
        )
    )
    unseen = _slate(suffix=" unseen")
    longer_history = (
        core.DialogueTurn("user", "First enrollment question."),
        core.DialogueTurn("agent", "First enrollment answer."),
        *tuple(
            turn
            for turn in _history(suffix=" unseen")
        ),
    )
    passages = _passages(suffix=" unseen")
    dense, ce = _scores(passages)
    unseen = core.build_action_slate(
        longer_history,
        passages,
        dense,
        ce,
    )
    decision = core.apply_e1(
        qualified,
        unseen,
        stage="M_search",
    )
    assert decision.fallback_to_e0
    assert decision.selected_recipe_id == core.E0_RECIPE_ID


def test_fit_accepts_only_fixed_integer_outcome_vector() -> None:
    slate = _slate()
    with pytest.raises(
        core.MultiDoc2DialP1TypedCoreError,
        match="AGENT_RECIPE_IDS",
    ):
        core.make_aform_example(slate, (1, 2))
    with pytest.raises(
        core.MultiDoc2DialP1TypedCoreError,
        match="outcome vector",
    ):
        core.make_aform_example(
            slate,
            (
                0,
                0,
                0,
                0,
                core.SCALE + 1,
            ),
        )
    assert set(core.AGENT_RECIPE_IDS).isdisjoint(
        core.BASELINE_RECIPE_IDS
    )
    assert core.E0_RECIPE_ID in core.AGENT_RECIPE_IDS
