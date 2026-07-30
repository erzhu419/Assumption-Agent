from dataclasses import FrozenInstanceError, replace

import pytest

from hegel_machine.bootstrap import initial_theory
from hegel_machine.ontology import UNIVERSAL_ASSUMPTIONS
from hegel_machine.schema import (
    AuthorityAssignment,
    AuthorityRole,
    EvidenceSplit,
    PatchCoordinate,
    PreregisteredPrediction,
    ScaleContext,
    TheoryPatch,
    freeze_pairs,
)


def _assignments():
    return tuple(
        AuthorityAssignment(role, f"actor_{role.value}") for role in AuthorityRole
    )


def _prediction():
    return PreregisteredPrediction(
        "prediction",
        "condition",
        "outcome",
        "increase",
        (0.0, 1.0),
        "outside range",
        "2026-07-30T08:00:00+08:00",
    )


def test_initial_theory_is_content_addressed_and_complete():
    first = initial_theory()
    second = initial_theory()
    assert first.version_id == second.version_id
    assert len(first.relation_laws) == 6
    assert len(first.hypothesis_families) == 22
    assert len(UNIVERSAL_ASSUMPTIONS) == 22
    assert not any(probe.semantic_only for probe in first.probes)


def test_initial_theory_is_immutable():
    theory = initial_theory()
    with pytest.raises(FrozenInstanceError):
        theory.scope = ("mutated",)


def test_scale_cannot_be_selected_on_holdout():
    with pytest.raises(ValueError, match="holdout"):
        ScaleContext(
            "bad_scale",
            "task",
            ("episode",),
            "mean",
            ("scope",),
            selected_on_split=EvidenceSplit.HOLDOUT,
        )


def test_five_authorities_must_be_distinct():
    assignments = list(_assignments())
    assignments[-1] = AuthorityAssignment(AuthorityRole.PROMOTER, "actor_generator")
    with pytest.raises(ValueError, match="distinct"):
        TheoryPatch(
            "patch_bad_roles",
            "candidate_bad_roles",
            initial_theory().version_id,
            PatchCoordinate.SCOPE,
            "repair scope",
            ("new_scope",),
            ("outside_scope",),
            (_prediction(),),
            ("hard_negative_1",),
            "reduction_1",
            0.2,
            freeze_pairs({"operation": "add_scope", "scope": "new_scope"}),
            tuple(assignments),
        )


def test_language_extension_requires_ontology_report():
    with pytest.raises(ValueError, match="ontology"):
        TheoryPatch(
            "patch_language",
            "candidate_language",
            initial_theory().version_id,
            PatchCoordinate.LANGUAGE,
            "invent relation",
            ("scope",),
            ("boundary",),
            (_prediction(),),
            ("hard_negative_1",),
            "reduction_1",
            0.4,
            freeze_pairs({"symbol": "R_new"}),
            _assignments(),
        )


def test_patch_payload_rejects_nested_mutable_values_and_duplicate_keys():
    parent = initial_theory()
    with pytest.raises(TypeError, match="scalar"):
        TheoryPatch(
            "patch_mutable",
            "candidate_mutable",
            parent.version_id,
            PatchCoordinate.SCOPE,
            "bad nested payload",
            ("new_scope",),
            ("boundary",),
            (_prediction(),),
            ("negative",),
            "reduction",
            0.1,
            (("operation", ["add_scope"]), ("scope", "new_scope")),
            _assignments(),
        )
    with pytest.raises(ValueError, match="duplicate"):
        freeze_pairs((("scope", "one"), ("scope", "two")))


def test_theory_rejects_shallow_mutability_before_hashing():
    theory = initial_theory()
    mutable_scope = list(theory.scope)
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(theory, scope=mutable_scope)
    assert theory.version_id == initial_theory().version_id


def test_patch_rejects_duplicate_authority_role():
    parent = initial_theory()
    duplicated = _assignments() + (
        AuthorityAssignment(AuthorityRole.PROMOTER, "second_promoter"),
    )
    with pytest.raises(ValueError, match="all five"):
        TheoryPatch(
            "patch_duplicate_role",
            "candidate_duplicate_role",
            parent.version_id,
            PatchCoordinate.SCOPE,
            "repair scope",
            ("new_scope",),
            ("boundary",),
            (_prediction(),),
            ("negative",),
            "reduction",
            0.1,
            freeze_pairs({"operation": "add_scope", "scope": "new_scope"}),
            duplicated,
        )


def test_patch_payload_has_one_canonical_order_and_content_hash():
    parent = initial_theory()
    common = (
        "patch_canonical",
        "candidate_canonical",
        parent.version_id,
        PatchCoordinate.SCOPE,
        "repair scope",
        ("new_scope",),
        ("boundary",),
        (_prediction(),),
        ("negative",),
        "reduction",
        0.1,
    )
    first = TheoryPatch(
        *common,
        (("scope", "new_scope"), ("operation", "add_scope")),
        _assignments(),
    )
    second = TheoryPatch(
        *common,
        (("operation", "add_scope"), ("scope", "new_scope")),
        _assignments(),
    )
    assert first.payload == second.payload
    assert first.content_id == second.content_id


def test_patch_rejects_repeated_prediction_and_negative():
    parent = initial_theory()
    with pytest.raises(ValueError, match="prediction"):
        TheoryPatch(
            "patch_repeat_prediction",
            "candidate",
            parent.version_id,
            PatchCoordinate.SCOPE,
            "repair",
            ("scope",),
            ("boundary",),
            (_prediction(), _prediction()),
            ("negative",),
            "reduction",
            0.1,
            freeze_pairs({"operation": "add_scope", "scope": "scope"}),
            _assignments(),
        )
    with pytest.raises(ValueError, match="hard negative"):
        TheoryPatch(
            "patch_repeat_negative",
            "candidate",
            parent.version_id,
            PatchCoordinate.SCOPE,
            "repair",
            ("scope",),
            ("boundary",),
            (_prediction(),),
            ("negative", "negative"),
            "reduction",
            0.1,
            freeze_pairs({"operation": "add_scope", "scope": "scope"}),
            _assignments(),
        )
