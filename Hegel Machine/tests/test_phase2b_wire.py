import ast
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from hegel_machine.phase2b_wire import (
    PREDICTION_SCHEMA_VERSION,
    PUBLIC_EVIDENCE_SCHEMA_VERSION,
    NumericInterval,
    PredictionBundle,
    PredictionDisposition,
    PredictionReason,
    PublicEvidenceBundle,
)


def uid(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def digest(character: str) -> str:
    return character * 64


def public_evidence_mapping() -> dict[str, object]:
    return {
        "schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "bundle_id": uid(1),
        "entity_candidates": [
            {
                "entity_id": uid(3),
                "role_candidate_ids": [uid(6), uid(5)],
            },
            {
                "entity_id": uid(4),
                "role_candidate_ids": [uid(5), uid(6)],
            },
        ],
        "role_ids": [uid(6), uid(5)],
        "quantity_ids": [uid(7)],
        "observations": [
            {
                "observation_id": uid(2),
                "source_channel_id": uid(15),
                "entity_ids": [uid(4), uid(3)],
                "role_candidate_ids": [uid(6), uid(5)],
                "quantity_id": uid(7),
                "value": {"kind": "numeric", "values": [1.0, 2.0]},
                "unit_dimension": {
                    "si_exponents": [1, 0, -1, 0, 0, 0, 0]
                },
                "temporal_support": {
                    "clock_id": uid(12),
                    "start": 0.0,
                    "end": 1.0,
                },
                "spatial_support": {
                    "frame_id": uid(13),
                    "lower": [0.0, 0.0],
                    "upper": [1.0, 1.0],
                },
                "uncertainty": {
                    "model": "absolute_bound",
                    "radius": [0.1, 0.1],
                },
                "provenance_sha256": digest("a"),
                "missingness": "observed",
            }
        ],
        "task_target": {
            "task_id": uid(8),
            "entity_ids": [uid(4), uid(3)],
            "quantity_ids": [uid(7)],
        },
        "aggregation_graph": {
            "scale_ids": [uid(10), uid(9)],
            "root_scale_ids": [uid(9)],
            "edges": [
                {
                    "source_scale_id": uid(9),
                    "target_scale_id": uid(10),
                    "transform_id": uid(11),
                }
            ],
        },
        "transform_catalog": [
            {
                "transform_id": uid(11),
                "operation": "temporal_aggregation",
                "parameters": [2.0],
            }
        ],
        "missingness_mask": [],
    }


def unique_prediction_mapping() -> dict[str, object]:
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "bundle_id": uid(1),
        "input_root_sha256": digest("b"),
        "protocol_sha256": digest("c"),
        "freeze_manifest_sha256": digest("d"),
        "disposition": "unique_match",
        "reason": "unique_structural_match",
        "family_id": uid(14),
        "binding": [
            {"role_id": uid(6), "entity_id": uid(4)},
            {"role_id": uid(5), "entity_id": uid(3)},
        ],
        "admissible_scale_ids": [uid(10), uid(9)],
    }


def test_public_evidence_round_trip_is_canonical_and_content_addressed():
    bundle = PublicEvidenceBundle.from_mapping(public_evidence_mapping())
    assert PublicEvidenceBundle.from_json(bundle.canonical_json) == bundle
    assert json.loads(bundle.canonical_json) == bundle.to_mapping()
    assert bundle.content_id.startswith("phase2b_evidence_")
    assert len(bundle.content_id) == len("phase2b_evidence_") + 64
    with pytest.raises(FrozenInstanceError):
        bundle.bundle_id = uid(99)


def test_set_like_input_order_cannot_change_evidence_content_id():
    first_mapping = public_evidence_mapping()
    second_mapping = deepcopy(first_mapping)
    for key in ("entity_candidates", "role_ids", "quantity_ids", "observations"):
        second_mapping[key] = list(reversed(second_mapping[key]))  # type: ignore[arg-type]
    second_mapping["task_target"]["entity_ids"].reverse()  # type: ignore[index,union-attr]
    second_mapping["aggregation_graph"]["scale_ids"].reverse()  # type: ignore[index,union-attr]
    second_mapping["transform_catalog"].reverse()  # type: ignore[union-attr]
    second_mapping["observations"][0]["entity_ids"].reverse()  # type: ignore[index,union-attr]
    second_mapping["observations"][0]["role_candidate_ids"].reverse()  # type: ignore[index,union-attr]

    first = PublicEvidenceBundle.from_mapping(first_mapping)
    second = PublicEvidenceBundle.from_mapping(second_mapping)
    assert first == second
    assert first.canonical_json == second.canonical_json
    assert first.content_id == second.content_id


@pytest.mark.parametrize(
    "field",
    (
        "law_family",
        "expected_pass",
        "correct_binding",
        "gold_scale",
        "oracle_margin",
        "candidate_private_payload",
        "candidate_rank",
        "answer",
    ),
)
def test_public_evidence_rejects_oracle_and_candidate_private_fields(field):
    mapping = public_evidence_mapping()
    mapping[field] = "forbidden"
    with pytest.raises(ValueError, match="unknown or forbidden"):
        PublicEvidenceBundle.from_mapping(mapping)


@pytest.mark.parametrize(
    "field",
    (
        "forward",
        "transformed",
        "inflows",
        "response_delta",
        "markov_blanket",
        "family_specific_observable",
    ),
)
def test_observation_rejects_family_specific_or_unknown_fields(field):
    mapping = public_evidence_mapping()
    mapping["observations"][0][field] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown or forbidden"):
        PublicEvidenceBundle.from_mapping(mapping)


def test_every_public_identifier_is_an_opaque_canonical_uuid4():
    fields = (
        ("bundle_id", "symmetry"),
        ("bundle_id", "00000000-0000-1000-8000-000000000001"),
        ("bundle_id", uid(10).upper()),
    )
    for field, invalid in fields:
        mapping = public_evidence_mapping()
        mapping[field] = invalid
        with pytest.raises(ValueError, match="UUIDv4"):
            PublicEvidenceBundle.from_mapping(mapping)

    mapping = public_evidence_mapping()
    mapping["observations"][0]["observation_id"] = "target-positive"  # type: ignore[index]
    with pytest.raises(ValueError, match="UUIDv4"):
        PublicEvidenceBundle.from_mapping(mapping)


def test_typed_interval_missingness_uncertainty_and_support_are_validated():
    interval = public_evidence_mapping()
    observation = interval["observations"][0]  # type: ignore[index]
    observation["value"] = {  # type: ignore[index]
        "kind": "interval",
        "lower": [0.9, 1.9],
        "upper": [1.1, 2.1],
    }
    bundle = PublicEvidenceBundle.from_mapping(interval)
    assert isinstance(bundle.observations[0].value, NumericInterval)

    reversed_interval = deepcopy(interval)
    reversed_interval["observations"][0]["value"]["lower"][0] = 2.0  # type: ignore[index]
    with pytest.raises(ValueError, match="lower bound"):
        PublicEvidenceBundle.from_mapping(reversed_interval)

    missing = public_evidence_mapping()
    missing_observation = missing["observations"][0]  # type: ignore[index]
    missing_observation["value"] = None  # type: ignore[index]
    missing_observation["missingness"] = "missing"  # type: ignore[index]
    missing_observation["uncertainty"] = {  # type: ignore[index]
        "model": "not_applicable",
        "radius": [],
    }
    missing["missingness_mask"] = [uid(2)]
    parsed_missing = PublicEvidenceBundle.from_mapping(missing)
    assert parsed_missing.observations[0].value is None

    contradictory = public_evidence_mapping()
    contradictory["observations"][0]["missingness"] = "missing"  # type: ignore[index]
    with pytest.raises(ValueError, match="presence disagree|missingness mask"):
        PublicEvidenceBundle.from_mapping(contradictory)

    bad_mask = public_evidence_mapping()
    bad_mask["missingness_mask"] = [uid(99)]
    with pytest.raises(ValueError, match="unknown observation"):
        PublicEvidenceBundle.from_mapping(bad_mask)

    bad_uncertainty = public_evidence_mapping()
    bad_uncertainty["observations"][0]["uncertainty"]["radius"] = [0.1]  # type: ignore[index]
    with pytest.raises(ValueError, match="dimension"):
        PublicEvidenceBundle.from_mapping(bad_uncertainty)


def test_public_references_and_aggregation_graph_fail_closed():
    unknown_entity = public_evidence_mapping()
    unknown_entity["observations"][0]["entity_ids"] = [uid(99)]  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown entity"):
        PublicEvidenceBundle.from_mapping(unknown_entity)

    unknown_transform = public_evidence_mapping()
    unknown_transform["aggregation_graph"]["edges"][0]["transform_id"] = uid(99)  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown transform"):
        PublicEvidenceBundle.from_mapping(unknown_transform)

    cyclic = public_evidence_mapping()
    cyclic["aggregation_graph"]["edges"].append(  # type: ignore[index,union-attr]
        {
            "source_scale_id": uid(10),
            "target_scale_id": uid(9),
            "transform_id": uid(11),
        }
    )
    with pytest.raises(ValueError, match="acyclic"):
        PublicEvidenceBundle.from_mapping(cyclic)


def test_prediction_unique_match_allows_one_family_binding_and_scale_set():
    prediction = PredictionBundle.from_mapping(unique_prediction_mapping())
    assert prediction.disposition is PredictionDisposition.UNIQUE_MATCH
    assert prediction.reason is PredictionReason.UNIQUE_STRUCTURAL_MATCH
    assert prediction.family_id == uid(14)
    assert prediction.admissible_scale_ids == (uid(9), uid(10))
    assert tuple(item.role_id for item in prediction.binding) == (uid(5), uid(6))
    assert PredictionBundle.from_json(prediction.canonical_json) == prediction
    assert prediction.content_id.startswith("phase2b_prediction_")


def test_prediction_abstention_carries_no_partial_answer():
    mapping = unique_prediction_mapping()
    mapping.update(
        disposition="abstain",
        reason="insufficient_evidence",
        family_id=None,
        binding=[],
        admissible_scale_ids=[],
    )
    prediction = PredictionBundle.from_mapping(mapping)
    assert prediction.disposition is PredictionDisposition.ABSTAIN

    for field, leaked in (
        ("family_id", uid(14)),
        ("binding", [{"role_id": uid(5), "entity_id": uid(3)}]),
        ("admissible_scale_ids", [uid(9)]),
    ):
        forged = deepcopy(mapping)
        forged[field] = leaked
        with pytest.raises(ValueError, match="abstention cannot carry"):
            PredictionBundle.from_mapping(forged)


@pytest.mark.parametrize("field", ("confidence", "answer", "expected_family"))
def test_prediction_rejects_confidence_and_answer_fields(field):
    mapping = unique_prediction_mapping()
    mapping[field] = 0.99
    with pytest.raises(ValueError, match="unknown or forbidden"):
        PredictionBundle.from_mapping(mapping)


def test_prediction_requires_hash_bindings_and_consistent_disposition():
    mapping = unique_prediction_mapping()
    for field in (
        "input_root_sha256",
        "protocol_sha256",
        "freeze_manifest_sha256",
    ):
        malformed = deepcopy(mapping)
        malformed[field] = "not-a-hash"
        with pytest.raises(ValueError, match="SHA-256"):
            PredictionBundle.from_mapping(malformed)

    wrong_reason = deepcopy(mapping)
    wrong_reason["reason"] = "insufficient_margin"
    with pytest.raises(ValueError, match="incompatible reason"):
        PredictionBundle.from_mapping(wrong_reason)


def test_prediction_binding_and_scale_order_are_canonical():
    first_mapping = unique_prediction_mapping()
    second_mapping = deepcopy(first_mapping)
    second_mapping["binding"].reverse()  # type: ignore[union-attr]
    second_mapping["admissible_scale_ids"].reverse()  # type: ignore[union-attr]
    first = PredictionBundle.from_mapping(first_mapping)
    second = PredictionBundle.from_mapping(second_mapping)
    assert first == second
    assert first.canonical_json == second.canonical_json
    assert first.content_id == second.content_id


def test_wire_module_has_no_generator_recognizer_verifier_or_evaluator_imports():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hegel_machine"
        / "phase2b_wire.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden = {"phase2_exit", "recognition", "laws", "evaluator"}
    assert not {
        component
        for module in imported
        for component in module.split(".")
        if component in forbidden
    }


def test_replacing_a_hash_binding_changes_prediction_content_id():
    prediction = PredictionBundle.from_mapping(unique_prediction_mapping())
    changed = replace(prediction, protocol_sha256=digest("e"))
    assert changed.content_id != prediction.content_id
