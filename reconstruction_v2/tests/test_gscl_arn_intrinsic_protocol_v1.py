from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping
import unicodedata

import pytest

from assumption_agent.benchmarks import (
    gscl_arn_intrinsic_protocol_v1 as protocol,
)


REFERENCE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "reference/gscl_intrinsic_candidates_20260730/metadata"
)
OFFICIAL_DATASET_PATH = REFERENCE_ROOT / "arn_dataset_v1.csv"
OFFICIAL_METADATA_PATH = REFERENCE_ROOT / "arn_zenodo_11044026.json"
LINKAGE_SECRET = hashlib.sha256(
    b"pre-source synthetic qualification linkage secret"
).digest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _proverb_for_bucket(bucket: int, seed: int) -> str:
    for attempt in range(10_000):
        candidate = f"Synthetic group relation {seed}-{attempt}"
        assignment = protocol.split_proverb(
            candidate, linkage_secret=LINKAGE_SECRET
        )
        if assignment.bucket == bucket:
            return candidate
    raise AssertionError("could not form deterministic synthetic bucket")


def _rows() -> list[protocol.AdaptedArnRow]:
    rows: list[protocol.AdaptedArnRow] = []
    source_id = 1
    cells = (
        ("far", "high"),
        ("far", "low"),
        ("near", "high"),
        ("near", "low"),
    )
    for repeat in range(2):
        for cell_index, (level, similarity) in enumerate(cells):
            bucket = 1 + cell_index
            proverb = _proverb_for_bucket(bucket, 100 + cell_index)
            rows.append(
                protocol.AdaptedArnRow(
                    source_id=str(source_id),
                    proverb=proverb,
                    query_narrative=(
                        f"Entirely synthetic narrative {source_id}"
                    ),
                    first_choice=(
                        f"Entirely synthetic first choice {source_id}"
                    ),
                    second_choice=(
                        f"Entirely synthetic second choice {source_id}"
                    ),
                    gold_choice=(
                        "first_choice"
                        if source_id % 2
                        else "second_choice"
                    ),
                    analogy_level=level,
                    distractor_similarity=similarity,
                )
            )
            source_id += 1
    calibration_proverb = _proverb_for_bucket(0, 999)
    rows.append(
        protocol.AdaptedArnRow(
            source_id=str(source_id),
            proverb=calibration_proverb,
            query_narrative="Synthetic calibration narrative",
            first_choice="Synthetic calibration first",
            second_choice="Synthetic calibration second",
            gold_choice="first_choice",
            analogy_level="far",
            distractor_similarity="high",
        )
    )
    return rows


@pytest.fixture()
def bundle() -> protocol.PrivatePackBundle:
    return protocol._build_private_packs_from_adapted_fixtures(
        _rows(),
        source_sha256=_sha("synthetic-source"),
        linkage_secret=LINKAGE_SECRET,
    )


@pytest.fixture()
def protocol_receipt() -> dict[str, object]:
    return protocol.build_safe_protocol_receipt()


@pytest.fixture()
def private_tmp() -> Path:
    root = Path(tempfile.mkdtemp(prefix="gscl-arn-", dir="/tmp"))
    root.chmod(0o700)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _prediction_packs(
    bundle: protocol.PrivatePackBundle,
    protocol_receipt: dict[str, object],
) -> dict[str, dict[str, object]]:
    item_ids = [
        row["opaque_item_id"]
        for row in bundle.predictor_pack["rows"]
    ]
    by_arm: dict[str, dict[str, object]] = {}
    for arm_id in protocol.ARM_IDS:
        predictions: list[dict[str, object]] = []
        for item_id in item_ids:
            if arm_id == "semantic_only":
                prediction = {
                    "opaque_item_id": item_id,
                    "disposition": "ANSWER",
                    "selected_choice": "first_choice",
                    "error_code": None,
                }
            elif arm_id == "legacy_keyword":
                prediction = {
                    "opaque_item_id": item_id,
                    "disposition": "ANSWER",
                    "selected_choice": "second_choice",
                    "error_code": None,
                }
            elif arm_id == "flat_label_no_verifier":
                prediction = {
                    "opaque_item_id": item_id,
                    "disposition": "ABSTAIN",
                    "selected_choice": None,
                    "error_code": None,
                }
            else:
                prediction = {
                    "opaque_item_id": item_id,
                    "disposition": "ANSWER",
                    "selected_choice": (
                        "first_choice"
                        if int(item_id[-1], 16) % 2 == 0
                        else "second_choice"
                    ),
                    "error_code": None,
                }
            predictions.append(prediction)
        by_arm[arm_id] = protocol.make_prediction_pack(
            arm_id=arm_id,
            arm_implementation_sha256=_sha(
                f"{arm_id}-implementation"
            ),
            arm_qualification_receipt_sha256=_sha(
                f"{arm_id}-qualification"
            ),
            protocol_contract_sha256=protocol_receipt[
                "protocol_contract_sha256"
            ],
            predictor_pack_sha256=bundle.pack_commitments[
                "predictor_pack_sha256"
            ],
            linkage_pack_sha256=bundle.pack_commitments[
                "linkage_pack_sha256"
            ],
            predictions=predictions,
        )
    return by_arm


def _write_source_fixture(
    tmp_path: Path,
    *,
    license_id: str = protocol.OFFICIAL_LICENSE_ID,
) -> tuple[Path, Path, protocol.SourceBinding]:
    header = protocol.OFFICIAL_HEADER_BYTES
    dataset_raw = header + b"synthetic,row,never,decoded\n"
    dataset_path = tmp_path / "arn.csv"
    dataset_path.write_bytes(dataset_raw)
    dataset_md5 = hashlib.md5(dataset_raw).hexdigest()  # noqa: S324
    dataset_sha256 = hashlib.sha256(dataset_raw).hexdigest()
    metadata = {
        "doi": protocol.OFFICIAL_DOI,
        "conceptdoi": protocol.OFFICIAL_CONCEPT_DOI,
        "revision": protocol.OFFICIAL_ZENODO_REVISION,
        "metadata": {
            "doi": protocol.OFFICIAL_DOI,
            "license": {"id": license_id},
        },
        "files": [
            {
                "key": protocol.OFFICIAL_DATASET_FILENAME,
                "size": len(dataset_raw),
                "checksum": f"md5:{dataset_md5}",
            }
        ],
    }
    metadata_raw = json.dumps(
        metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    metadata_path = tmp_path / "zenodo.json"
    metadata_path.write_bytes(metadata_raw)
    binding = protocol.SourceBinding(
        doi=protocol.OFFICIAL_DOI,
        concept_doi=protocol.OFFICIAL_CONCEPT_DOI,
        revision=protocol.OFFICIAL_ZENODO_REVISION,
        license_id=protocol.OFFICIAL_LICENSE_ID,
        dataset_filename=protocol.OFFICIAL_DATASET_FILENAME,
        dataset_size=len(dataset_raw),
        dataset_md5=dataset_md5,
        dataset_sha256=dataset_sha256,
        metadata_size=len(metadata_raw),
        metadata_sha256=hashlib.sha256(metadata_raw).hexdigest(),
        header_bytes=header,
    )
    return dataset_path, metadata_path, binding


def _qualification_file(
    root: Path, component_id: str
) -> protocol.ValidatedImplementationQualification:
    component_root = root / component_id
    component_root.mkdir(mode=0o700)
    implementation = component_root / "implementation.py"
    test_file = component_root / "test_implementation.py"
    implementation.write_text(
        f"COMPONENT = {component_id!r}\n", encoding="utf-8"
    )
    test_file.write_text("def test_fixture(): pass\n", encoding="utf-8")
    implementation_hash = hashlib.sha256(
        implementation.read_bytes()
    ).hexdigest()
    test_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()
    closure = {
        "component_id": component_id,
        "implementation_file_sha256": implementation_hash,
        "qualification_test_file_sha256": test_hash,
        "qualification_scope": "synthetic_and_source_free_mechanics_only",
    }
    receipt = {
        "schema": protocol.IMPLEMENTATION_QUALIFICATION_SCHEMA,
        "status": "QUALIFIED_SOURCE_FREE_IMPLEMENTATION_CLOSURE",
        **closure,
        "implementation_closure_sha256": protocol._content_hash(closure),
        "source_free": True,
        "measurement_content_accessed": False,
        "formal_measurement_run": False,
    }
    receipt["self_hash"] = protocol._content_hash(receipt)
    receipt_path = component_root / "qualification.safe.json"
    receipt_file_hash = protocol._write_exclusive(
        receipt_path, receipt
    )
    return protocol.validate_implementation_qualification_file(
        receipt_path=receipt_path,
        expected_receipt_file_sha256=receipt_file_hash,
        implementation_path=implementation,
        qualification_test_path=test_file,
        expected_component_id=component_id,
    )


def test_protocol_receipt_is_non_scoring_and_honestly_not_ready(
    protocol_receipt: dict[str, object],
) -> None:
    assert protocol_receipt["new_study"] is False
    assert protocol_receipt["formal_measurement"] is False
    assert protocol_receipt["efficacy_evidence"] is False
    assert protocol_receipt["effect_gate_added"] is False
    assert protocol_receipt["freeze_ready"] is False
    assert protocol_receipt["blocker_ids"] == [
        "OFFICIAL_SOURCE_NOT_VERIFIED_IN_RECEIPT",
        "RAW_NARRATIVE_ADAPTER_NOT_READY",
        "FOUR_ARM_IMPLEMENTATIONS_NOT_READY",
        "OFFICIAL_ADAPTER_TO_PACK_CLOSURE_NOT_READY",
        "CAPABILITY_MATERIALIZATION_NOT_READY",
        "RUNTIME_ACCESS_QUALIFICATION_NOT_READY",
    ]
    assert (
        protocol_receipt["implementation_closure"][
            "raw_narrative_adapter"
        ]["status"]
        == "NOT_READY"
    )
    assert {
        value["status"]
        for key, value in protocol_receipt[
            "implementation_closure"
        ].items()
        if key in protocol.ARM_IDS
    } == {"NOT_READY"}
    with pytest.raises(protocol.ArnImplementationNotReady):
        protocol.build_raw_narrative_adapter()
    with pytest.raises(protocol.ArnImplementationNotReady):
        protocol.build_arm_algorithm("full_gscl")


def test_exposure_quarantine_is_safe_and_does_not_change_mod5_split(
    protocol_receipt: dict[str, object],
) -> None:
    exposure = protocol_receipt["implementation_exposure"]
    assert exposure == protocol.IMPLEMENTATION_EXPOSURE
    assert exposure["exposed_field_classes"] == [
        "proverb",
        "query_narrative_prefix",
    ]
    assert exposure["exposed_choice_count"] == 0
    assert exposure["exposed_answer_count"] == 0
    assert exposure["exposed_cell_label_count"] == 0
    assert exposure["immutable_mod5_bucket"] == 2
    assert exposure["measurement_excluded"] is True
    assert exposure["label_unseen_disposition"] is True
    assert exposure["label_unseen_is_cryptographic_claim"] is False
    assert exposure["public_group_digest_emitted"] is False
    assert exposure["split_assignment_changed"] is False
    assert exposure["rebalanced"] is False
    assert exposure["fallback_replacement"] is False


def test_whole_exposed_group_is_quarantined_by_private_hmac_only() -> None:
    exposed_proverb = _proverb_for_bucket(2, 700)
    rows = [
        protocol.AdaptedArnRow(
            source_id="1",
            proverb=exposed_proverb,
            query_narrative="Synthetic exposed anchor",
            first_choice="Synthetic first",
            second_choice="Synthetic second",
            gold_choice="first_choice",
            analogy_level="far",
            distractor_similarity="high",
        ),
        protocol.AdaptedArnRow(
            source_id="2",
            proverb=exposed_proverb,
            query_narrative="Synthetic same whole group",
            first_choice="Synthetic first two",
            second_choice="Synthetic second two",
            gold_choice="second_choice",
            analogy_level="near",
            distractor_similarity="low",
        ),
        protocol.AdaptedArnRow(
            source_id="3",
            proverb=_proverb_for_bucket(3, 701),
            query_narrative="Synthetic unexposed group",
            first_choice="Synthetic first three",
            second_choice="Synthetic second three",
            gold_choice="first_choice",
            analogy_level="near",
            distractor_similarity="high",
        ),
    ]
    bundle = protocol._build_private_packs(
        rows,
        source_sha256=protocol.OFFICIAL_DATASET_SHA256,
        linkage_secret=LINKAGE_SECRET,
        lineage="official_arn_measurement",
        schemas=(
            protocol.OFFICIAL_PREDICTOR_PACK_SCHEMA,
            protocol.OFFICIAL_LINKAGE_PACK_SCHEMA,
            protocol.OFFICIAL_LABEL_PACK_SCHEMA,
        ),
        source_verification_self_hash=_sha("source-verification"),
        adapter_qualification_self_hash=_sha("adapter-qualification"),
        quarantine_source_id="1",
    )
    quarantined = [
        row
        for row in bundle.linkage_pack["rows"]
        if row["exclusion_codes"] == ["IMPLEMENTATION_EXPOSURE"]
    ]
    assert len(quarantined) == 2
    assert len({row["private_group_id"] for row in quarantined}) == 1
    assert {row["bucket"] for row in quarantined} == {2}
    assert all(not row["measurement_eligible"] for row in quarantined)
    assert bundle.safe_split_aggregates[
        "public_group_digest_emitted"
    ] is False


def test_splitter_is_exact_nfkc_full_digest_mod5_and_whole_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ascii_value = "Synthetic A"
    compatibility_value = "Synthetic \N{FULLWIDTH LATIN CAPITAL LETTER A}"
    assignment_a = protocol.split_proverb(
        ascii_value, linkage_secret=LINKAGE_SECRET
    )
    assignment_b = protocol.split_proverb(
        compatibility_value, linkage_secret=LINKAGE_SECRET
    )
    assert assignment_a == assignment_b
    normalized = unicodedata.normalize("NFKC", compatibility_value)
    digest = hashlib.sha256(
        protocol.SPLIT_SALT + b"\0" + normalized.encode("utf-8")
    ).hexdigest()
    assert assignment_a.private_group_id == __import__("hmac").new(
        LINKAGE_SECRET,
        b"group\0" + normalized.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert assignment_a.bucket == int(digest, 16) % 5
    assert assignment_a.hash_partition == (
        "calibration" if assignment_a.bucket == 0 else "measurement"
    )

    monkeypatch.setattr(
        protocol,
        "FROZEN_UNIDATA_VERSION",
        "deliberately-different",
    )
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError,
        match="Unicode database version",
    ):
        protocol.split_proverb(
            "synthetic", linkage_secret=LINKAGE_SECRET
        )


def test_linkage_identity_is_private_hmac_not_public_split_digest() -> None:
    proverb = _proverb_for_bucket(3, 404)
    other_secret = hashlib.sha256(b"another pre-source secret").digest()
    first = protocol.split_proverb(
        proverb, linkage_secret=LINKAGE_SECRET
    )
    second = protocol.split_proverb(
        proverb, linkage_secret=other_secret
    )
    assert first.bucket == second.bucket == 3
    assert first.private_group_id != second.private_group_id
    public_digest = hashlib.sha256(
        protocol.SPLIT_SALT
        + b"\0"
        + unicodedata.normalize("NFKC", proverb).encode("utf-8")
    ).hexdigest()
    assert first.private_group_id != public_digest


def test_official_source_exact_hash_doi_license_and_header_verify() -> None:
    receipt = protocol.verify_official_source(
        OFFICIAL_DATASET_PATH, OFFICIAL_METADATA_PATH
    )
    assert receipt["verified"] is True
    assert receipt["dataset_sha256"] == protocol.OFFICIAL_DATASET_SHA256
    assert receipt["doi"] == protocol.OFFICIAL_DOI
    assert receipt["concept_doi"] == protocol.OFFICIAL_CONCEPT_DOI
    assert (
        receipt["source_qualification_report_sha256"]
        == protocol.SOURCE_QUALIFICATION_REPORT_SHA256
    )
    assert receipt["zenodo_revision"] == 4
    assert receipt["license_id"] == protocol.OFFICIAL_LICENSE_ID
    assert receipt["dataset_rows_decoded"] == 0
    assert receipt["item_content_emitted"] is False
    body = dict(receipt)
    claimed = body.pop("self_hash")
    assert protocol._content_hash(body) == claimed
    protocol_receipt = protocol.build_safe_protocol_receipt(
        source_verification=receipt
    )
    assert protocol_receipt["source_contract"]["source_verified"] is True
    assert protocol_receipt["freeze_ready"] is False
    assert protocol_receipt["blocker_ids"] == [
        "RAW_NARRATIVE_ADAPTER_NOT_READY",
        "FOUR_ARM_IMPLEMENTATIONS_NOT_READY",
        "OFFICIAL_ADAPTER_TO_PACK_CLOSURE_NOT_READY",
        "CAPABILITY_MATERIALIZATION_NOT_READY",
        "RUNTIME_ACCESS_QUALIFICATION_NOT_READY",
    ]


def test_source_tamper_and_license_drift_fail_closed(
    tmp_path: Path,
) -> None:
    dataset_path, metadata_path, binding = _write_source_fixture(tmp_path)
    assert protocol._verify_source_files(
        dataset_path, metadata_path, binding
    )["verified"]

    original = dataset_path.read_bytes()
    dataset_path.write_bytes(original[:-1] + b"X")
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="SHA256"
    ):
        protocol._verify_source_files(
            dataset_path, metadata_path, binding
        )

    license_root = tmp_path / "license"
    license_root.mkdir()
    dataset_path, metadata_path, binding = _write_source_fixture(
        license_root, license_id="not-cc-by"
    )
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="license"
    ):
        protocol._verify_source_files(
            dataset_path, metadata_path, binding
        )


def test_source_symlink_and_header_drift_fail_closed(
    tmp_path: Path,
) -> None:
    dataset_path, metadata_path, binding = _write_source_fixture(tmp_path)
    symlink = tmp_path / "source-link.csv"
    symlink.symlink_to(dataset_path)
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="topology"
    ):
        protocol._verify_source_files(symlink, metadata_path, binding)

    header_root = tmp_path / "header"
    header_root.mkdir()
    dataset_path, metadata_path, binding = _write_source_fixture(header_root)
    raw = dataset_path.read_bytes().replace(
        b"id,proverb", b"ix,proverb", 1
    )
    dataset_path.write_bytes(raw)
    drift_binding = protocol.SourceBinding(
        **{
            **binding.__dict__,
            "dataset_md5": hashlib.md5(raw).hexdigest(),  # noqa: S324
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="header"
    ):
        protocol._verify_source_files(
            dataset_path, metadata_path, drift_binding
        )


def test_only_exact_qualification_files_can_close_implementation(
    private_tmp: Path,
) -> None:
    validated = _qualification_file(
        private_tmp, "raw_narrative_adapter"
    )
    receipt = protocol.build_safe_protocol_receipt(
        implementation_qualifications={
            "raw_narrative_adapter": validated
        }
    )
    assert receipt["freeze_ready"] is False
    assert receipt["implementation_closure"][
        "raw_narrative_adapter"
    ]["status"] == "READY"
    assert "OFFICIAL_ADAPTER_TO_PACK_CLOSURE_NOT_READY" in receipt[
        "blocker_ids"
    ]

    forged = protocol.ValidatedImplementationQualification(
        component_id="raw_narrative_adapter",
        receipt=validated.receipt,
        receipt_path=validated.receipt_path,
        receipt_file_sha256="a" * 64,
        implementation_path=validated.implementation_path,
        implementation_file_sha256="b" * 64,
        qualification_test_path=validated.qualification_test_path,
        qualification_test_file_sha256="c" * 64,
        _validation_token=object(),
    )
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="exact files"
    ):
        protocol.build_safe_protocol_receipt(
            implementation_qualifications={
                "raw_narrative_adapter": forged
            }
        )

    implementation = (
        private_tmp
        / "raw_narrative_adapter"
        / "implementation.py"
    )
    implementation.write_text("TAMPERED = True\n", encoding="utf-8")
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="SHA256"
    ):
        protocol.build_safe_protocol_receipt(
            implementation_qualifications={
                "raw_narrative_adapter": validated
            }
        )
    receipt_path = (
        private_tmp
        / "raw_narrative_adapter"
        / "qualification.safe.json"
    )
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="SHA256"
    ):
        protocol.validate_implementation_qualification_file(
            receipt_path=receipt_path,
            expected_receipt_file_sha256=(
                validated.receipt_file_sha256
            ),
            implementation_path=implementation,
            qualification_test_path=(
                private_tmp
                / "raw_narrative_adapter"
                / "test_implementation.py"
            ),
            expected_component_id="raw_narrative_adapter",
        )


def test_capability_materialization_is_separate_but_not_formal_on_one_uid(
    private_tmp: Path,
    bundle: protocol.PrivatePackBundle,
) -> None:
    receipt = protocol.materialize_qualification_capabilities_once(
        root=private_tmp / "materialized", bundle=bundle
    )
    assert receipt["formal_ready"] is False
    assert receipt["uid_separated"] is False
    assert receipt["arm_visible_pack_classes"] == ["predictor"]
    assert receipt["custodian_visible_pack_classes"] == [
        "linkage",
        "label",
    ]
    for relative in (
        "arm_capability/predictor.private.json",
        "custodian_capability/linkage.private.json",
        "custodian_capability/labels.private.json",
    ):
        assert (
            (private_tmp / "materialized" / relative).stat().st_mode
            & 0o777
        ) == 0o600
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="distinct"
    ):
        protocol.audit_formal_capability_materialization(
            root=private_tmp / "materialized",
            arm_uid=__import__("os").getuid(),
            custodian_uid=__import__("os").getuid(),
            pack_commitments=bundle.pack_commitments,
        )


def test_private_packs_separate_column_capabilities_and_commitments(
    bundle: protocol.PrivatePackBundle,
) -> None:
    assert bundle.safe_split_aggregates["source_row_count"] == 9
    assert bundle.safe_split_aggregates["measurement_item_count"] == 8
    assert (
        bundle.safe_split_aggregates[
            "whole_proverb_group_cross_partition_count"
        ]
        == 0
    )
    assert bundle.safe_split_aggregates["rebalanced"] is False
    assert bundle.safe_split_aggregates["fallback_replacement"] is False
    assert bundle.predictor_pack["column_contract"] == list(
        protocol.COLUMN_ACCESS_MATRIX["arms"]
    )
    assert bundle.linkage_pack["column_contract"] == list(
        protocol.COLUMN_ACCESS_MATRIX["splitter"]
    )
    assert bundle.label_pack["column_contract"] == list(
        protocol.COLUMN_ACCESS_MATRIX["scorer_only"]
    )
    encoded_predictor = json.dumps(bundle.predictor_pack, sort_keys=True)
    assert "proverb" not in encoded_predictor
    assert "gold_choice" not in encoded_predictor
    assert "analogy_level" not in encoded_predictor
    assert "distractor_similarity" not in encoded_predictor
    for row in _rows():
        assert row.proverb not in encoded_predictor
        assert row.source_id not in {
            packed["opaque_item_id"]
            for packed in bundle.predictor_pack["rows"]
        }
    assert bundle.pack_commitments == {
        "predictor_pack_sha256": protocol._content_hash(
            bundle.predictor_pack
        ),
        "linkage_pack_sha256": protocol._content_hash(
            bundle.linkage_pack
        ),
        "label_pack_sha256": protocol._content_hash(bundle.label_pack),
    }


def test_prediction_schema_and_exact_common_input_fail_closed(
    bundle: protocol.PrivatePackBundle,
    protocol_receipt: dict[str, object],
) -> None:
    packs = _prediction_packs(bundle, protocol_receipt)
    missing = dict(packs)
    missing.pop("legacy_keyword")
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="all four"
    ):
        protocol.build_qualification_action_seal(
            protocol_contract_sha256=protocol_receipt[
                "protocol_contract_sha256"
            ],
            predictor_pack=bundle.predictor_pack,
            linkage_pack=bundle.linkage_pack,
            label_pack_sha256=bundle.pack_commitments[
                "label_pack_sha256"
            ],
            prediction_packs=missing,
        )

    malformed = dict(packs["semantic_only"])
    malformed_predictions = [
        dict(row) for row in malformed["predictions"]
    ]
    malformed_predictions[0]["explanation"] = "forbidden"
    malformed["predictions"] = malformed_predictions
    malformed_body = dict(malformed)
    malformed_body.pop("self_hash")
    malformed["self_hash"] = protocol._content_hash(malformed_body)
    bad_packs = dict(packs)
    bad_packs["semantic_only"] = malformed
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="schema"
    ):
        protocol.build_qualification_action_seal(
            protocol_contract_sha256=protocol_receipt[
                "protocol_contract_sha256"
            ],
            predictor_pack=bundle.predictor_pack,
            linkage_pack=bundle.linkage_pack,
            label_pack_sha256=bundle.pack_commitments[
                "label_pack_sha256"
            ],
            prediction_packs=bad_packs,
        )

    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="ABSTAIN"
    ):
        protocol.make_prediction_pack(
            arm_id="semantic_only",
            arm_implementation_sha256=_sha("implementation"),
            arm_qualification_receipt_sha256=_sha("qualification"),
            protocol_contract_sha256=protocol_receipt[
                "protocol_contract_sha256"
            ],
            predictor_pack_sha256=bundle.pack_commitments[
                "predictor_pack_sha256"
            ],
            linkage_pack_sha256=bundle.pack_commitments[
                "linkage_pack_sha256"
            ],
            predictions=[
                {
                    "opaque_item_id": bundle.predictor_pack["rows"][0][
                        "opaque_item_id"
                    ],
                    "disposition": "ABSTAIN",
                    "selected_choice": "first_choice",
                    "error_code": None,
                }
            ],
        )
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="exact canonical file"
    ):
        protocol.build_all_arm_action_seal(
            ready_freeze={"freeze_ready": True},
            adapter_output_receipt={"status": "forged"},
            predictor_pack=bundle.predictor_pack,
            linkage_pack=bundle.linkage_pack,
            label_pack_sha256=bundle.pack_commitments[
                "label_pack_sha256"
            ],
            prediction_packs=packs,
        )


def test_all_arm_barrier_then_labels_open_once_and_score_aggregates(
    private_tmp: Path,
    bundle: protocol.PrivatePackBundle,
    protocol_receipt: dict[str, object],
) -> None:
    packs = _prediction_packs(bundle, protocol_receipt)
    seal = protocol.build_qualification_action_seal(
        protocol_contract_sha256=protocol_receipt[
            "protocol_contract_sha256"
        ],
        predictor_pack=bundle.predictor_pack,
        linkage_pack=bundle.linkage_pack,
        label_pack_sha256=bundle.pack_commitments["label_pack_sha256"],
        prediction_packs=packs,
    )
    state_root = private_tmp / "state"
    seal_path = state_root / "all_arm_action_seal.safe.json"
    seal_file_hash = protocol.persist_action_seal_once(seal_path, seal)
    with pytest.raises(FileExistsError):
        protocol.persist_action_seal_once(seal_path, seal)

    calls = {"labels": 0}

    def load_labels() -> dict[str, object]:
        calls["labels"] += 1
        return dict(bundle.label_pack)

    score = protocol.open_labels_and_score_qualification_once(
        state_root=state_root,
        action_seal_path=seal_path,
        expected_action_seal_file_sha256=seal_file_hash,
        prediction_packs=packs,
        linkage_pack=bundle.linkage_pack,
        label_loader=load_labels,
    )
    assert calls == {"labels": 1}
    assert score["online_or_api_evaluator_used"] is False
    assert score["effect_gate"] is False
    assert score["retry_or_resample"] is False
    assert score["abstain_and_error_counted_wrong"] is True
    assert score["formal_terminal"] is False
    assert score["lifecycle"] == "synthetic_qualification_only"
    aggregates = score["arm_aggregates"]
    assert aggregates["semantic_only"]["overall"]["correct"] == 4
    assert aggregates["semantic_only"]["overall"]["total"] == 8
    assert aggregates["legacy_keyword"]["overall"]["correct"] == 4
    assert aggregates["flat_label_no_verifier"]["overall"]["correct"] == 0
    assert aggregates["flat_label_no_verifier"][
        "disposition_counts"
    ] == {"ANSWER": 0, "ABSTAIN": 8, "ERROR": 0}
    assert aggregates["full_gscl"]["disposition_counts"] == {
        "ANSWER": 8,
        "ABSTAIN": 0,
        "ERROR": 0,
    }
    for arm in aggregates.values():
        assert set(arm["cells"]) == {
            "far_high",
            "far_low",
            "near_high",
            "near_low",
        }
        assert all(cell["total"] == 2 for cell in arm["cells"].values())
        assert arm["overall"]["proverb_cluster_count"] == 4
    differences = score["paired_differences"]
    assert set(differences) == {
        "full_gscl_minus_semantic_only",
        "full_gscl_minus_legacy_keyword",
        "full_gscl_minus_flat_label_no_verifier",
    }
    full_correct = aggregates["full_gscl"]["overall"]["correct"]
    assert differences["full_gscl_minus_semantic_only"]["overall"][
        "difference_sum"
    ] == full_correct - 4
    assert differences["full_gscl_minus_flat_label_no_verifier"][
        "overall"
    ]["difference_sum"] == full_correct
    for comparison in differences.values():
        assert comparison["effect_gate"] is False
        assert set(comparison["cells"]) == {
            "far_high",
            "far_low",
            "near_high",
            "near_low",
        }

    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="already"
    ):
        protocol.open_labels_and_score_qualification_once(
            state_root=state_root,
            action_seal_path=seal_path,
            expected_action_seal_file_sha256=seal_file_hash,
            prediction_packs=packs,
            linkage_pack=bundle.linkage_pack,
            label_loader=load_labels,
        )
    assert calls == {"labels": 1}


def test_barrier_and_tamper_fail_before_label_loader(
    private_tmp: Path,
    bundle: protocol.PrivatePackBundle,
    protocol_receipt: dict[str, object],
) -> None:
    packs = _prediction_packs(bundle, protocol_receipt)
    calls = {"labels": 0}

    def load_labels() -> dict[str, object]:
        calls["labels"] += 1
        return dict(bundle.label_pack)

    with pytest.raises(protocol.ArnIntrinsicProtocolError):
        protocol.open_labels_and_score_qualification_once(
            state_root=private_tmp / "absent",
            action_seal_path=private_tmp / "absent-seal.json",
            expected_action_seal_file_sha256=_sha("absent"),
            prediction_packs=packs,
            linkage_pack=bundle.linkage_pack,
            label_loader=load_labels,
        )
    assert calls == {"labels": 0}

    seal = protocol.build_qualification_action_seal(
        protocol_contract_sha256=protocol_receipt[
            "protocol_contract_sha256"
        ],
        predictor_pack=bundle.predictor_pack,
        linkage_pack=bundle.linkage_pack,
        label_pack_sha256=bundle.pack_commitments["label_pack_sha256"],
        prediction_packs=packs,
    )
    state_root = private_tmp / "tampered"
    seal_path = state_root / "seal.json"
    seal_file_hash = protocol.persist_action_seal_once(seal_path, seal)
    tampered_linkage = dict(bundle.linkage_pack)
    tampered_rows = [dict(row) for row in tampered_linkage["rows"]]
    tampered_rows[0]["bucket"] = (
        tampered_rows[0]["bucket"] + 1
    ) % 5
    tampered_linkage["rows"] = tampered_rows
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="linkage"
    ):
        protocol.open_labels_and_score_qualification_once(
            state_root=state_root,
            action_seal_path=seal_path,
            expected_action_seal_file_sha256=seal_file_hash,
            prediction_packs=packs,
            linkage_pack=tampered_linkage,
            label_loader=load_labels,
        )
    assert calls == {"labels": 0}


def test_label_hash_tamper_consumes_one_shot_without_retry(
    private_tmp: Path,
    bundle: protocol.PrivatePackBundle,
    protocol_receipt: dict[str, object],
) -> None:
    packs = _prediction_packs(bundle, protocol_receipt)
    seal = protocol.build_qualification_action_seal(
        protocol_contract_sha256=protocol_receipt[
            "protocol_contract_sha256"
        ],
        predictor_pack=bundle.predictor_pack,
        linkage_pack=bundle.linkage_pack,
        label_pack_sha256=bundle.pack_commitments["label_pack_sha256"],
        prediction_packs=packs,
    )
    state_root = private_tmp / "label-tamper"
    seal_path = state_root / "seal.json"
    seal_file_hash = protocol.persist_action_seal_once(seal_path, seal)
    tampered_labels = dict(bundle.label_pack)
    tampered_rows = [dict(row) for row in tampered_labels["rows"]]
    tampered_rows[0]["gold_choice"] = (
        "second_choice"
        if tampered_rows[0]["gold_choice"] == "first_choice"
        else "first_choice"
    )
    tampered_labels["rows"] = tampered_rows
    calls = {"labels": 0}

    def load_tampered_labels() -> dict[str, object]:
        calls["labels"] += 1
        return tampered_labels

    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="commitment"
    ):
        protocol.open_labels_and_score_qualification_once(
            state_root=state_root,
            action_seal_path=seal_path,
            expected_action_seal_file_sha256=seal_file_hash,
            prediction_packs=packs,
            linkage_pack=bundle.linkage_pack,
            label_loader=load_tampered_labels,
        )
    assert calls == {"labels": 1}
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="already"
    ):
        protocol.open_labels_and_score_qualification_once(
            state_root=state_root,
            action_seal_path=seal_path,
            expected_action_seal_file_sha256=seal_file_hash,
            prediction_packs=packs,
            linkage_pack=bundle.linkage_pack,
            label_loader=load_tampered_labels,
        )
    assert calls == {"labels": 1}


def test_action_file_mode_and_hardlink_tamper_fail_before_labels(
    private_tmp: Path,
    bundle: protocol.PrivatePackBundle,
    protocol_receipt: dict[str, object],
) -> None:
    packs = _prediction_packs(bundle, protocol_receipt)
    seal = protocol.build_qualification_action_seal(
        protocol_contract_sha256=protocol_receipt[
            "protocol_contract_sha256"
        ],
        predictor_pack=bundle.predictor_pack,
        linkage_pack=bundle.linkage_pack,
        label_pack_sha256=bundle.pack_commitments["label_pack_sha256"],
        prediction_packs=packs,
    )
    state_root = private_tmp / "mode"
    seal_path = state_root / "seal.json"
    seal_hash = protocol.persist_action_seal_once(seal_path, seal)
    seal_path.chmod(0o644)
    calls = {"labels": 0}

    def loader() -> dict[str, object]:
        calls["labels"] += 1
        return dict(bundle.label_pack)

    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="mode"
    ):
        protocol.open_labels_and_score_qualification_once(
            state_root=state_root,
            action_seal_path=seal_path,
            expected_action_seal_file_sha256=seal_hash,
            prediction_packs=packs,
            linkage_pack=bundle.linkage_pack,
            label_loader=loader,
        )
    assert calls == {"labels": 0}
    seal_path.chmod(0o600)
    hardlink = state_root / "seal-hardlink.json"
    hardlink.hardlink_to(seal_path)
    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="topology"
    ):
        protocol.open_labels_and_score_qualification_once(
            state_root=state_root,
            action_seal_path=seal_path,
            expected_action_seal_file_sha256=seal_hash,
            prediction_packs=packs,
            linkage_pack=bundle.linkage_pack,
            label_loader=loader,
        )
    assert calls == {"labels": 0}


def test_loader_and_terminal_failures_write_immutable_failure(
    private_tmp: Path,
    bundle: protocol.PrivatePackBundle,
    protocol_receipt: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packs = _prediction_packs(bundle, protocol_receipt)

    def make_state(name: str) -> tuple[Path, Path, str]:
        state = private_tmp / name
        seal_path = state / "seal.json"
        seal = protocol.build_qualification_action_seal(
            protocol_contract_sha256=protocol_receipt[
                "protocol_contract_sha256"
            ],
            predictor_pack=bundle.predictor_pack,
            linkage_pack=bundle.linkage_pack,
            label_pack_sha256=bundle.pack_commitments[
                "label_pack_sha256"
            ],
            prediction_packs=packs,
        )
        return (
            state,
            seal_path,
            protocol.persist_action_seal_once(seal_path, seal),
        )

    state, seal_path, seal_hash = make_state("loader-failure")

    def broken_loader() -> dict[str, object]:
        raise RuntimeError("synthetic loader failure")

    with pytest.raises(
        protocol.ArnIntrinsicProtocolError, match="label loader"
    ):
        protocol.open_labels_and_score_qualification_once(
            state_root=state,
            action_seal_path=seal_path,
            expected_action_seal_file_sha256=seal_hash,
            prediction_packs=packs,
            linkage_pack=bundle.linkage_pack,
            label_loader=broken_loader,
        )
    failure = json.loads(
        (state / "qualification_failure_terminal.safe.json").read_text()
    )
    assert failure["failure_code"] == "LABEL_LOADER_EXCEPTION"
    assert failure["retry_or_replay_allowed"] is False

    state, seal_path, seal_hash = make_state("terminal-failure")
    original_write = protocol._write_exclusive

    def fail_only_terminal(
        path: Path,
        value: Mapping[str, object],
        *,
        expected_uid: int | None = None,
    ) -> str:
        if path.name == "qualification_aggregate_score.safe.json":
            raise OSError("synthetic terminal failure")
        return original_write(
            path, value, expected_uid=expected_uid
        )

    monkeypatch.setattr(protocol, "_write_exclusive", fail_only_terminal)
    with pytest.raises(OSError, match="terminal"):
        protocol.open_labels_and_score_qualification_once(
            state_root=state,
            action_seal_path=seal_path,
            expected_action_seal_file_sha256=seal_hash,
            prediction_packs=packs,
            linkage_pack=bundle.linkage_pack,
            label_loader=lambda: dict(bundle.label_pack),
        )
    terminal_failure = json.loads(
        (state / "qualification_failure_terminal.safe.json").read_text()
    )
    assert terminal_failure["failure_code"] == (
        "TERMINAL_PERSISTENCE_FAILED"
    )


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_all_mapping_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_all_mapping_keys(nested))
    return keys


def test_safe_receipt_exact_allowlist_nested_hashes_and_no_fixture_text(
    protocol_receipt: dict[str, object],
) -> None:
    assert set(protocol_receipt) == {
        "schema",
        "status",
        "new_study",
        "formal_measurement",
        "efficacy_evidence",
        "effect_gate_added",
        "freeze_ready",
        "blocker_ids",
        "source_contract",
        "split_contract",
        "column_access_contract",
        "pack_contract",
        "prediction_contract",
        "lifecycle_contract",
        "metric_contract",
        "implementation_exposure",
        "implementation_closure",
        "capability_closure",
        "nested_hashes",
        "protocol_contract_sha256",
        "self_hash",
    }
    section_names = {
        "source_contract",
        "split_contract",
        "column_access_contract",
        "pack_contract",
        "prediction_contract",
        "lifecycle_contract",
        "metric_contract",
        "implementation_exposure",
        "implementation_closure",
        "capability_closure",
    }
    assert set(protocol_receipt["nested_hashes"]) == section_names
    for section in section_names:
        assert (
            protocol_receipt["nested_hashes"][section]
            == protocol._content_hash(protocol_receipt[section])
        )
    body = dict(protocol_receipt)
    self_hash = body.pop("self_hash")
    assert protocol._content_hash(body) == self_hash
    forbidden_detail_keys = {
        "id",
        "proverb",
        "query_narrative",
        "first_choice",
        "second_choice",
        "correct_answer",
        "gold_choice",
        "analogy_level",
        "distractor_similarity",
        "predictions",
        "rows",
        "per_item_score",
    }
    assert not (_all_mapping_keys(protocol_receipt) & forbidden_detail_keys)
    encoded = json.dumps(protocol_receipt, sort_keys=True)
    for row in _rows():
        assert row.proverb not in encoded
        assert row.query_narrative not in encoded
        assert row.first_choice not in encoded
        assert row.second_choice not in encoded
        public_digest = hashlib.sha256(
            protocol.SPLIT_SALT
            + b"\0"
            + unicodedata.normalize("NFKC", row.proverb).encode("utf-8")
        ).hexdigest()
        private_group_id = protocol.split_proverb(
            row.proverb, linkage_secret=LINKAGE_SECRET
        ).private_group_id
        assert public_digest not in encoded
        assert private_group_id not in encoded
