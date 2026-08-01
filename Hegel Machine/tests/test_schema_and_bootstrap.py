import json
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import pytest

import hegel_machine.laws as laws_module
from hegel_machine.bootstrap import initial_theory
from hegel_machine.governance import SEMANTIC_METRICS
from hegel_machine.hashing import stable_hash
from hegel_machine.laws import VERIFIER_REGISTRY_ID
from hegel_machine.ontology import (
    ACTIVE_FUNCTIONALS,
    ACTIVE_LAWS,
    FROZEN_ACTIVE_LAW_IDS,
    FROZEN_LEAF_IDS,
    FROZEN_ROOTS,
    UNIVERSAL_ASSUMPTIONS,
    validate_registry,
)
from hegel_machine.schema import (
    AuthorityAssignment,
    AuthorityRole,
    EvidenceSplit,
    LawKind,
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
    assert first.ontology_registry_id == stable_hash(
        UNIVERSAL_ASSUMPTIONS,
        prefix="ontology_registry_",
    )
    assert first.verifier_registry_id == VERIFIER_REGISTRY_ID
    assert first.evaluator.epoch == "phase2_epoch_0002"
    assert first.evaluator.version == "0.2.0"
    assert {probe.version for probe in first.probes} == {"2"}
    assert not any(probe.semantic_only for probe in first.probes)


def test_verifier_registry_id_binds_the_current_laws_source_bytes():
    expected = "verifier_registry_sha256_" + sha256(
        Path(laws_module.__file__).read_bytes()
    ).hexdigest()
    assert VERIFIER_REGISTRY_ID == expected


def test_real_ontology_config_matches_code_and_governance_exclusions():
    config_path = Path(__file__).resolve().parents[1] / "configs/initial_ontology.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert tuple(config["roots"]) == FROZEN_ROOTS
    assert tuple(config["leaf_ids"]) == tuple(
        item.template_id for item in UNIVERSAL_ASSUMPTIONS
    )
    assert tuple(config["active_law_ids"]) == tuple(
        law.law_id for law in ACTIVE_LAWS
    )
    forbidden_inputs = set(config["acceptance_forbidden_inputs"])
    assert forbidden_inputs == {
        "semantic_retrieval_score",
        "embedding_similarity",
        "llm_self_reported_confidence",
        "legacy_fixture_pass",
    }
    assert set(SEMANTIC_METRICS).issubset(forbidden_inputs)


def test_frozen_registry_exactly_matches_the_in_code_config_contract():
    assert (
        tuple(item.template_id for item in UNIVERSAL_ASSUMPTIONS)
        == FROZEN_LEAF_IDS
    )
    assert {item.root for item in UNIVERSAL_ASSUMPTIONS} == set(FROZEN_ROOTS)
    assert next(
        item for item in UNIVERSAL_ASSUMPTIONS if item.template_id == "T17"
    ).root == "invariance"
    assert tuple(law.law_id for law in ACTIVE_LAWS) == FROZEN_ACTIVE_LAW_IDS
    functionals = {
        functional.functional_id: functional for functional in ACTIVE_FUNCTIONALS
    }
    for law in ACTIVE_LAWS:
        functional = functionals[law.violation_functional_id]
        assert functional.law_kind is law.kind
        assert functional.required_observables == law.required_observables
    validate_registry()


def test_registry_validation_rejects_root_leaf_and_law_config_drift():
    t17_index = next(
        index
        for index, item in enumerate(UNIVERSAL_ASSUMPTIONS)
        if item.template_id == "T17"
    )
    wrong_root = list(UNIVERSAL_ASSUMPTIONS)
    wrong_root[t17_index] = replace(wrong_root[t17_index], root="shape")
    with pytest.raises(AssertionError, match="root"):
        validate_registry(assumptions=tuple(wrong_root))

    wrong_leaf = (replace(UNIVERSAL_ASSUMPTIONS[0], template_id="T99"),) + (
        UNIVERSAL_ASSUMPTIONS[1:]
    )
    with pytest.raises(AssertionError, match="leaves"):
        validate_registry(assumptions=wrong_leaf)

    wrong_law = (
        replace(ACTIVE_LAWS[0], law_id="law_unregistered"),
    ) + ACTIVE_LAWS[1:]
    with pytest.raises(AssertionError, match="active laws"):
        validate_registry(laws=wrong_law)


def test_registry_validation_rejects_law_functional_contract_drift():
    wrong_binding = (
        replace(
            ACTIVE_LAWS[0],
            violation_functional_id=ACTIVE_FUNCTIONALS[1].functional_id,
        ),
    ) + ACTIVE_LAWS[1:]
    with pytest.raises(AssertionError, match="wrong violation functional"):
        validate_registry(laws=wrong_binding)

    wrong_kind = (
        replace(ACTIVE_FUNCTIONALS[0], law_kind=LawKind.LOCALITY),
    ) + ACTIVE_FUNCTIONALS[1:]
    with pytest.raises(AssertionError, match="kinds"):
        validate_registry(functionals=wrong_kind)

    wrong_observables = (
        replace(ACTIVE_FUNCTIONALS[0], required_observables=("unexpected",)),
    ) + ACTIVE_FUNCTIONALS[1:]
    with pytest.raises(AssertionError, match="observables"):
        validate_registry(functionals=wrong_observables)


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


def test_theory_rejects_duplicate_functional_and_scale_identifiers():
    theory = initial_theory()
    with pytest.raises(ValueError, match="functional identifiers"):
        replace(
            theory,
            violation_functionals=(
                *theory.violation_functionals,
                theory.violation_functionals[0],
            ),
        )
    with pytest.raises(ValueError, match="scale identifiers"):
        replace(theory, scales=(*theory.scales, theory.scales[0]))


def test_theory_rejects_missing_registry_identity_and_unknown_law_scales():
    theory = initial_theory()
    with pytest.raises(ValueError, match="verifier registry"):
        replace(theory, verifier_registry_id="")

    law = theory.relation_laws[0]
    with pytest.raises(ValueError, match="registered scales"):
        replace(
            theory,
            relation_laws=(
                replace(law, scale_ids=("absent_scale",)),
                *theory.relation_laws[1:],
            ),
        )
    with pytest.raises(ValueError, match="registered scales"):
        replace(
            theory,
            relation_laws=(
                replace(law, scale_ids=()),
                *theory.relation_laws[1:],
            ),
        )


def test_theory_rejects_missing_or_misaligned_law_functional_contracts():
    theory = initial_theory()
    law = theory.relation_laws[0]
    with pytest.raises(ValueError, match="registered violation functional"):
        replace(
            theory,
            relation_laws=(
                replace(law, violation_functional_id="vf_missing"),
                *theory.relation_laws[1:],
            ),
        )

    functional = theory.violation_functionals[0]
    with pytest.raises(ValueError, match="kinds disagree"):
        replace(
            theory,
            violation_functionals=(
                replace(functional, law_kind=LawKind.LOCALITY),
                *theory.violation_functionals[1:],
            ),
        )
    with pytest.raises(ValueError, match="observables disagree"):
        replace(
            theory,
            violation_functionals=(
                replace(functional, required_observables=("unexpected",)),
                *theory.violation_functionals[1:],
            ),
        )


def test_theory_rejects_unused_functionals():
    theory = initial_theory()
    unused = replace(
        theory.violation_functionals[0],
        functional_id="vf_registered_but_unused",
    )
    with pytest.raises(ValueError, match="must be used"):
        replace(
            theory,
            violation_functionals=(*theory.violation_functionals, unused),
        )


def test_theory_rejects_probe_anchor_and_task_registry_drift():
    theory = initial_theory()
    probe = theory.probes[0]
    with pytest.raises(ValueError, match="probe anchors"):
        replace(
            theory,
            probes=(
                replace(probe, anchor_ids=("unregistered_anchor",)),
                *theory.probes[1:],
            ),
        )
    with pytest.raises(ValueError, match="probe tasks"):
        replace(
            theory,
            probes=(
                replace(probe, task_ids=("unregistered_task",)),
                *theory.probes[1:],
            ),
        )
    with pytest.raises(ValueError, match="probe tasks"):
        replace(
            theory,
            probes=(replace(probe, task_ids=()), *theory.probes[1:]),
        )


def test_theory_rejects_probe_cutoff_drift():
    theory = initial_theory()
    with pytest.raises(ValueError, match="theory data cutoff"):
        replace(
            theory,
            probes=(
                replace(theory.probes[0], data_cutoff="2026-07-31T00:00:00+08:00"),
                *theory.probes[1:],
            ),
        )


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
