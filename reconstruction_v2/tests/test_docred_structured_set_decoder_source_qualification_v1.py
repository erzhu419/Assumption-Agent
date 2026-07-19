from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from assumption_agent.benchmarks import (
    docred_structured_set_decoder_source_qualification_v1 as audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_MARKERS = (
    "PRIVATE_TITLE",
    "PRIVATE_HEAD",
    "PRIVATE_TAIL",
    "PRIVATE_SENTENCE",
    "PRIVATE_TEST_DO_NOT_OPEN",
    "PRIVATE_DISTANT_DO_NOT_OPEN",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _with_self_hash(value: dict[str, Any], field: str) -> bytes:
    body = copy.deepcopy(value)
    body.pop(field, None)
    body[field] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return json.dumps(
        body,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("ascii") + b"\n"


def _write_mode_0600(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


def _document(split: str, family: str, index: int) -> dict[str, Any]:
    marker = f"PRIVATE_{split}_{family}_{index}"
    relation = audit.FAMILY_PROPERTIES[family][0]
    head = f"PRIVATE_HEAD_{marker}"
    tail = f"PRIVATE_TAIL_{marker}"
    sentences = [
        [head, "PRIVATE_SENTENCE", "zero"],
        [tail, "PRIVATE_SENTENCE", "one"],
    ]
    sentences.extend(
        [[f"PRIVATE_SENTENCE_{marker}_{ordinal}"] for ordinal in range(2, 10)]
    )
    return {
        "title": f"PRIVATE_TITLE_{marker}",
        "sents": sentences,
        "vertexSet": [
            [
                {
                    "name": head,
                    "sent_id": 0,
                    "pos": [0, 1],
                    "type": "PER",
                }
            ],
            [
                {
                    "name": tail,
                    "sent_id": 1,
                    "pos": [0, 1],
                    "type": "ORG",
                }
            ],
        ],
        "labels": [{"h": 0, "t": 1, "r": relation, "evidence": [0, 1]}],
    }


def _documents(split: str, per_family: int) -> list[dict[str, Any]]:
    return [
        _document(split, family, index)
        for family in audit.FAMILIES
        for index in range(per_family)
    ]


def _manifest_template(name: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "manifests" / name).read_text("utf-8"))


def _rebind_source_chain(project: Path) -> None:
    manifest_root = project / "manifests"
    source_root = project / "artifacts/docred_official_source_v1"
    bindings = {
        "train_annotated": (
            source_root / "train_annotated.json",
            "train_sha256",
        ),
        "dev": (source_root / "dev.json", "dev_sha256"),
        "relation_metadata": (
            source_root / "rel_info.json",
            "relation_metadata_sha256",
        ),
    }
    access_path = (
        manifest_root / "docred_structured_set_decoder_source_access_v1.json"
    )
    access = json.loads(access_path.read_text("utf-8"))
    for access_key, (path, _prerequisite_key) in bindings.items():
        raw = path.read_bytes()
        record = access["allowed_local_files"][access_key]
        record["sha256"] = hashlib.sha256(raw).hexdigest()
        record["size"] = len(raw)
        record["mode"] = 0o600
    access_path.write_bytes(_with_self_hash(access, "source_access_sha256"))
    access = json.loads(access_path.read_text("utf-8"))

    k3_path = (
        manifest_root / "docred_structured_set_decoder_k3_pre_row_amendment_v1.json"
    )
    k3 = json.loads(k3_path.read_text("utf-8"))
    k3["prerequisite"]["source_access_self_sha256"] = access[
        "source_access_sha256"
    ]
    for access_key, (_path, prerequisite_key) in bindings.items():
        k3["prerequisite"][prerequisite_key] = access["allowed_local_files"][
            access_key
        ]["sha256"]
    k3_path.write_bytes(_with_self_hash(k3, "k3_amendment_sha256"))
    k3 = json.loads(k3_path.read_text("utf-8"))

    family_path = (
        manifest_root
        / "docred_structured_set_decoder_relation_family_freeze_v1.json"
    )
    family = json.loads(family_path.read_text("utf-8"))
    family["prerequisite"]["source_access_self_sha256"] = access[
        "source_access_sha256"
    ]
    family["prerequisite"]["k_amendment_self_sha256"] = k3[
        "k3_amendment_sha256"
    ]
    for access_key, (_path, prerequisite_key) in bindings.items():
        family["prerequisite"][prerequisite_key] = access[
            "allowed_local_files"
        ][access_key]["sha256"]
    family_path.write_bytes(
        _with_self_hash(family, "relation_family_freeze_sha256")
    )


def _fixture(
    tmp_path: Path,
    *,
    train_per_family: int = 60,
    dev_per_family: int = 20,
    cross_split_collision: bool = False,
) -> Path:
    # pytest's configured temp root can live on DrvFS, where chmod(0600) is not
    # represented faithfully.  The source contract deliberately binds POSIX
    # mode bits, so keep synthetic source files on the Linux filesystem.
    fixture_id = hashlib.sha256(str(tmp_path).encode("utf-8")).hexdigest()[:20]
    project = Path("/tmp/docred_source_qualification_v1_tests") / fixture_id
    shutil.rmtree(project, ignore_errors=True)
    manifests = project / "manifests"
    manifests.mkdir(parents=True)
    for name in (
        "docred_structured_set_decoder_source_custody_v1.json",
        "docred_structured_set_decoder_source_access_v1.json",
        "docred_structured_set_decoder_k3_pre_row_amendment_v1.json",
        "docred_structured_set_decoder_relation_family_freeze_v1.json",
    ):
        (manifests / name).write_bytes(
            (PROJECT_ROOT / "manifests" / name).read_bytes()
        )

    train = _documents("train", train_per_family)
    dev = _documents("dev", dev_per_family)
    if cross_split_collision:
        dev[0] = copy.deepcopy(train[0])
    relation_metadata = {
        property_id: f"public description for {property_id}"
        for property_id in sorted(audit.FAMILY_PROPERTY_UNION)
    }
    source_root = project / "artifacts/docred_official_source_v1"
    _write_mode_0600(source_root / "train_annotated.json", _json_bytes(train))
    _write_mode_0600(source_root / "dev.json", _json_bytes(dev))
    _write_mode_0600(
        source_root / "rel_info.json", _json_bytes(relation_metadata)
    )
    _write_mode_0600(
        source_root / "test.json", b"PRIVATE_TEST_DO_NOT_OPEN"
    )
    _write_mode_0600(
        source_root / "train_distant.json", b"PRIVATE_DISTANT_DO_NOT_OPEN"
    )
    _rebind_source_chain(project)
    return project


def _rewrite_train(project: Path, mutation: Callable[[Any], None]) -> None:
    path = project / audit.FORMAL_TRAIN_RELATIVE_PATH
    payload = json.loads(path.read_text("utf-8"))
    mutation(payload)
    _write_mode_0600(path, _json_bytes(payload))
    _rebind_source_chain(project)


def _qualify(project: Path) -> dict[str, Any]:
    return audit.qualify_source_files(project, enforce_formal_identity=False)


def _contains_list(value: Any) -> bool:
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return any(_contains_list(child) for child in value.values())
    return False


def test_committed_formal_manifest_chain_matches_constants() -> None:
    specs, binding = audit._validate_frozen_contracts(
        PROJECT_ROOT,
        enforce_formal_identity=True,
    )
    assert specs["train"].sha256 == audit.FORMAL_TRAIN_SHA256
    assert specs["dev"].sha256 == audit.FORMAL_DEV_SHA256
    assert (
        specs["relation_metadata"].sha256
        == audit.FORMAL_RELATION_METADATA_SHA256
    )
    assert binding["family_freeze_self_sha256"] == (
        audit.FORMAL_FAMILY_FREEZE_SELF_SHA256
    )


def test_passed_receipt_is_aggregate_only_and_never_opens_forbidden_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _fixture(tmp_path)
    opened: list[str] = []
    original_open = Path.open

    def spy_open(path: Path, *args: Any, **kwargs: Any):
        opened.append(path.name)
        if path.name in {"test.json", "train_distant.json"}:
            raise AssertionError("forbidden split was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)
    receipt = _qualify(project)
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    assert receipt["status"] == "passed_source_qualification_no_selection"
    assert not _contains_list(receipt)
    for marker in PRIVATE_MARKERS:
        assert marker not in serialized
    assert "train_annotated.json" in opened
    assert "dev.json" in opened
    assert "rel_info.json" in opened
    assert "test.json" not in opened
    assert "train_distant.json" not in opened

    capacity = receipt["simultaneous_document_assignment_capacity"]
    assert capacity["required_global_document_count"] == 240
    assert capacity["deterministic_max_flow_assigned_document_count"] == 240
    assert capacity["simultaneous_all_block_document_disjoint_feasible"] is True
    for family in audit.FAMILIES:
        train = capacity["per_split_family"]["train"][family]
        dev = capacity["per_split_family"]["dev"][family]
        assert train["required_assigned_document_count"] == 60
        assert train["deterministic_max_flow_assigned_document_count"] == 60
        assert dev["required_assigned_document_count"] == 20
        assert dev["deterministic_max_flow_assigned_document_count"] == 20
    body = dict(receipt)
    declared = body.pop("qualification_sha256")
    assert declared == audit._stable_hash(body)


def test_one_document_shortfall_is_terminal_without_secret_or_cohort(
    tmp_path: Path,
) -> None:
    project = _fixture(tmp_path, train_per_family=60)

    def remove_one(payload: list[dict[str, Any]]) -> None:
        relation = audit.FAMILY_PROPERTIES["GEO_SOVEREIGNTY"][0]
        for index, document in enumerate(payload):
            if document["labels"][0]["r"] == relation:
                del payload[index]
                return
        raise AssertionError("synthetic relation not found")

    _rewrite_train(project, remove_one)
    receipt = _qualify(project)
    assert receipt["status"] == "terminal_source_incompatible_no_selection"
    capacity = receipt["simultaneous_document_assignment_capacity"]
    assert capacity["deterministic_max_flow_assigned_document_count"] == 239
    assert capacity["simultaneous_all_block_document_disjoint_feasible"] is False
    assert receipt["terminal_reason_counts"] == {
        "invalid_root_schema_count": 0,
        "invalid_document_schema_count": 0,
        "simultaneous_assignment_shortfall_count": 1,
    }
    assert receipt["claim_boundary"]["selection_secret_generated_or_opened"] is False
    assert receipt["claim_boundary"]["cohort_selected_or_materialized"] is False


def test_cross_split_title_and_serialized_document_collision_reduce_max_flow(
    tmp_path: Path,
) -> None:
    receipt = _qualify(_fixture(tmp_path, cross_split_collision=True))
    assert receipt["status"] == "terminal_source_incompatible_no_selection"
    collisions = receipt["cross_split_collision_counts"]
    assert collisions["cross_split_normalized_title"] == {
        "collision_group_count": 1,
        "train_document_occurrence_count": 1,
        "dev_document_occurrence_count": 1,
    }
    assert collisions["cross_split_canonical_serialized_document"] == {
        "collision_group_count": 1,
        "train_document_occurrence_count": 1,
        "dev_document_occurrence_count": 1,
    }
    capacity = receipt["simultaneous_document_assignment_capacity"]
    assert capacity["multi_document_collision_component_count"] == 1
    assert capacity["deterministic_max_flow_assigned_document_count"] == 239


def _pop_document_field(document: dict[str, Any], field: str) -> None:
    document.pop(field)


@pytest.mark.parametrize(
    ("kind", "mutation"),
    [
        ("document_not_object", lambda document: None),
        (
            "document_required_fields",
            lambda document: _pop_document_field(document, "labels"),
        ),
        ("title", lambda document: document.__setitem__("title", "")),
        ("sentences", lambda document: document.__setitem__("sents", None)),
        ("sentence", lambda document: document["sents"].__setitem__(0, "bad")),
        (
            "sentence_token",
            lambda document: document["sents"][0].__setitem__(0, 7),
        ),
        (
            "vertex_set",
            lambda document: document.__setitem__("vertexSet", None),
        ),
        (
            "entity_cluster",
            lambda document: document["vertexSet"].__setitem__(0, []),
        ),
        (
            "mention_not_object",
            lambda document: document["vertexSet"][0].__setitem__(0, "bad"),
        ),
        (
            "mention_required_fields",
            lambda document: document["vertexSet"][0][0].pop("pos"),
        ),
        (
            "mention_name",
            lambda document: document["vertexSet"][0][0].__setitem__("name", ""),
        ),
        (
            "mention_sentence",
            lambda document: document["vertexSet"][0][0].__setitem__(
                "sent_id", True
            ),
        ),
        (
            "mention_position",
            lambda document: document["vertexSet"][0][0].__setitem__(
                "pos", [0, 999]
            ),
        ),
        (
            "mention_type",
            lambda document: document["vertexSet"][0][0].__setitem__("type", ""),
        ),
        ("labels", lambda document: document.__setitem__("labels", None)),
        (
            "label_not_object",
            lambda document: document["labels"].__setitem__(0, "bad"),
        ),
        (
            "label_required_fields",
            lambda document: document["labels"][0].pop("evidence"),
        ),
        (
            "label_endpoint",
            lambda document: document["labels"][0].__setitem__("h", True),
        ),
        (
            "label_relation",
            lambda document: document["labels"][0].__setitem__("r", "P999999"),
        ),
        (
            "label_evidence",
            lambda document: document["labels"][0].__setitem__(
                "evidence", [999]
            ),
        ),
    ],
)
def test_each_minimum_schema_branch_fails_closed_as_aggregate_terminal(
    tmp_path: Path,
    kind: str,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    project = _fixture(tmp_path)

    def mutate(payload: list[Any]) -> None:
        if kind == "document_not_object":
            payload[0] = "PRIVATE_TITLE_NOT_AN_OBJECT"
        else:
            mutation(payload[0])

    _rewrite_train(project, mutate)
    receipt = _qualify(project)
    assert receipt["status"] == "terminal_source_incompatible_no_selection"
    train_schema = receipt["split_aggregates"]["train"]["schema"]
    assert train_schema["invalid_document_count"] == 1
    assert train_schema["anomaly_counts"][kind] == 1
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for marker in PRIVATE_MARKERS:
        assert marker not in serialized


def test_non_list_split_root_is_terminal_aggregate(tmp_path: Path) -> None:
    project = _fixture(tmp_path)
    train_path = project / audit.FORMAL_TRAIN_RELATIVE_PATH
    _write_mode_0600(train_path, _json_bytes({"PRIVATE_TITLE": "not a list"}))
    _rebind_source_chain(project)
    receipt = _qualify(project)
    assert receipt["status"] == "terminal_source_incompatible_no_selection"
    schema = receipt["split_aggregates"]["train"]["schema"]
    assert schema["root_is_list"] is False
    assert receipt["terminal_reason_counts"]["invalid_root_schema_count"] == 1


def test_duplicate_label_query_and_evidence_are_counts_only(tmp_path: Path) -> None:
    project = _fixture(tmp_path)

    def duplicate(payload: list[dict[str, Any]]) -> None:
        label = copy.deepcopy(payload[0]["labels"][0])
        label["evidence"] = [0, 0]
        payload[0]["labels"].append(label)

    _rewrite_train(project, duplicate)
    receipt = _qualify(project)
    assert receipt["status"] == "passed_source_qualification_no_selection"
    train = receipt["split_aggregates"]["train"]
    assert train["evidence_cardinality_counts"][
        "duplicate_evidence_ordinal_occurrence_count"
    ] == 1
    assert train["duplicate_counts"][
        "duplicate_exact_h_r_t_label_occurrence_count"
    ] == 1
    assert train["duplicate_counts"]["eligible_derived_query"][
        "duplicate_group_count"
    ] == 1


def test_serialized_document_identity_excludes_labels_and_extra_metadata() -> None:
    relation_metadata = {
        property_id: f"public description for {property_id}"
        for property_id in sorted(audit.FAMILY_PROPERTY_UNION)
    }
    left = _document("train", audit.FAMILIES[0], 0)
    right = copy.deepcopy(left)
    right["labels"][0]["evidence"] = [2]
    right["private_extra_metadata"] = "PRIVATE_SENTENCE_not_document_text"
    left_record, _ = audit._parse_document(
        left,
        split="train",
        relation_metadata=relation_metadata,
    )
    right_record, _ = audit._parse_document(
        right,
        split="dev",
        relation_metadata=relation_metadata,
    )
    assert (
        left_record.serialized_document_sha256
        == right_record.serialized_document_sha256
    )


def test_source_binding_rejects_private_or_extra_receipt_fields() -> None:
    with pytest.raises(
        audit.DocredSourceQualificationError,
        match="keyset drifted",
    ):
        audit._validated_source_binding(
            {"PRIVATE_TITLE": "PRIVATE_SENTENCE"},
            formal_identity_enforced=False,
        )


def test_empty_sentence_is_schema_invalid_and_cannot_enter_capacity(
    tmp_path: Path,
) -> None:
    project = _fixture(tmp_path)

    def empty_sentence(payload: list[dict[str, Any]]) -> None:
        payload[0]["sents"][2] = []

    _rewrite_train(project, empty_sentence)
    receipt = _qualify(project)
    assert receipt["status"] == "terminal_source_incompatible_no_selection"
    schema = receipt["split_aggregates"]["train"]["schema"]
    assert schema["invalid_document_count"] == 1
    assert schema["anomaly_counts"]["sentence"] == 1


def test_semantic_manifest_tamper_fails_before_any_source_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _fixture(tmp_path)
    family_path = project / audit.FORMAL_FAMILY_FREEZE_RELATIVE_PATH
    family = json.loads(family_path.read_text("utf-8"))
    family["formation_and_block_contract"]["all_blocks_document_disjoint"] = False
    family_path.write_bytes(
        _with_self_hash(family, "relation_family_freeze_sha256")
    )
    opened_sources: list[str] = []
    original_open = Path.open

    def spy_open(path: Path, *args: Any, **kwargs: Any):
        if path.name in {"train_annotated.json", "dev.json", "rel_info.json"}:
            opened_sources.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)
    with pytest.raises(audit.DocredSourceQualificationError, match="drifted"):
        _qualify(project)
    assert opened_sources == []


def test_source_hash_drift_fails_before_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _fixture(tmp_path)
    train_path = project / audit.FORMAL_TRAIN_RELATIVE_PATH
    train_path.write_bytes(train_path.read_bytes() + b"PRIVATE_TITLE_DRIFT")
    train_path.chmod(0o600)
    decoded_labels: list[str] = []
    original_decode = audit._strict_json

    def spy_decode(raw: bytes, *, label: str) -> Any:
        decoded_labels.append(label)
        return original_decode(raw, label=label)

    monkeypatch.setattr(audit, "_strict_json", spy_decode)
    with pytest.raises(
        audit.DocredSourceQualificationError,
        match="byte binding drifted",
    ):
        _qualify(project)
    assert "authorized TRAIN source" not in decoded_labels


def test_cli_uses_project_root_and_emits_one_canonical_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    expected = {
        "schema": audit.SCHEMA,
        "status": "synthetic_cli_receipt",
        "qualification_sha256": "a" * 64,
    }
    observed: list[Path] = []

    def fake_build(project_root: Path) -> dict[str, Any]:
        observed.append(project_root)
        return expected

    monkeypatch.setattr(audit, "build_formal_qualification", fake_build)
    assert audit.main(["--project-root", str(tmp_path)]) == 0
    assert observed == [tmp_path]
    assert json.loads(capsysbinary.readouterr().out) == expected
