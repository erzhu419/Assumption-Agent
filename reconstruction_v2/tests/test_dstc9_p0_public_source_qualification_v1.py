from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any, Iterator
from unittest import mock

import pytest

from assumption_agent.benchmarks import (
    dstc9_p0_public_source_qualification_v1 as p0,
)


TYPED_SHA256 = hashlib.sha256(
    Path(p0.core.__file__).read_bytes()
).hexdigest()


@pytest.fixture
def posix_tmp() -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="qualifier-fixture-", dir="/tmp"))
    assert not str(root).startswith("/tmp/dstc9")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _events(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        yield "start_map", None
        for key, member in value.items():
            yield "map_key", key
            yield from _events(member)
        yield "end_map", None
    elif isinstance(value, list):
        yield "start_array", None
        for member in value:
            yield from _events(member)
        yield "end_array", None
    elif isinstance(value, str):
        yield "string", value
    elif type(value) is bool:
        yield "boolean", value
    elif type(value) is int:
        yield "number", value
    elif value is None:
        yield "null", None
    else:
        raise AssertionError(type(value))


def _synthetic_basic_parse(source) -> Iterator[tuple[str, Any]]:
    yield from _events(json.load(source))


def _knowledge() -> dict[str, Any]:
    return {
        family: {
            "0": {
                "name": (
                    None if family in {"taxi", "train"} else f"{family} name"
                ),
                "docs": {
                    "0": {
                        "title": f"{family} title",
                        "body": f"{family} grounded body",
                    }
                },
            }
        }
        for family in p0.FAMILIES
    }


def _target_label(family: str) -> dict[str, Any]:
    return {
        "target": True,
        "knowledge": [
            {
                "domain": family,
                "entity_id": 0,
                "doc_id": 0,
            }
        ],
        "response": f"Grounded response for {family}.",
    }


def _base_split(split: str) -> tuple[list[Any], list[Any]]:
    logs: list[Any] = []
    labels: list[Any] = []
    for family in p0.FAMILIES:
        logs.append(
            [
                {
                    "speaker": "U",
                    "text": f"Unique {split} {family} question?",
                }
            ]
        )
        labels.append(_target_label(family))
    logs.append(
        [
            {
                "speaker": "U",
                "text": f"Unique {split} non target question?",
            }
        ]
    )
    labels.append({"target": False})
    return logs, labels


def _values(
    *,
    train: tuple[list[Any], list[Any]] | None = None,
    validation: tuple[list[Any], list[Any]] | None = None,
    knowledge: dict[str, Any] | None = None,
) -> dict[str, bytes]:
    train_logs, train_labels = (
        train if train is not None else _base_split("TRAIN")
    )
    validation_logs, validation_labels = (
        validation
        if validation is not None
        else _base_split("VALIDATION")
    )
    semantic: dict[str, Any] = {
        p0.KNOWLEDGE_MEMBER: knowledge or _knowledge(),
        p0.TRAIN_LOGS_MEMBER: train_logs,
        p0.TRAIN_LABELS_MEMBER: train_labels,
        p0.VALIDATION_LOGS_MEMBER: validation_logs,
        p0.VALIDATION_LABELS_MEMBER: validation_labels,
    }
    return {
        p0.FAQ_MEMBER: b"identity-only FAQ fixture\n",
        p0.LICENSE_MEMBER: b"identity-only license fixture\n",
        p0.NOTICE_MEMBER: b"identity-only notice fixture\n",
        **{
            path: json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            for path, value in semantic.items()
        },
    }


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - test Git identity.
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()


def _bundle(
    root: Path,
    values: dict[str, bytes],
    *,
    minimum: int = 1,
    public_hashes: frozenset[str] = frozenset(),
    metadata_override: tuple[str, str, int] | None = None,
    extra_member: tuple[str, bytes] | None = None,
) -> tuple[Path, p0.QualificationContract]:
    path = root / "fixture.bundle.tar"
    with tarfile.open(
        path,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for member_path in sorted(values):
            raw = values[member_path]
            info = tarfile.TarInfo(member_path)
            info.size = len(raw)
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            info.uname = ""
            info.gname = ""
            if (
                metadata_override is not None
                and member_path == metadata_override[0]
            ):
                setattr(
                    info,
                    metadata_override[1],
                    metadata_override[2],
                )
            archive.addfile(info, fileobj=__import__("io").BytesIO(raw))
        if extra_member is not None:
            extra_path, raw = extra_member
            info = tarfile.TarInfo(extra_path)
            info.size = len(raw)
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            archive.addfile(info, fileobj=__import__("io").BytesIO(raw))
    path.chmod(0o600)
    members = tuple(
        sorted(
            (
                p0.MemberContract(
                    member_path,
                    len(raw),
                    hashlib.sha256(raw).hexdigest(),
                    _git_blob_sha1(raw),
                )
                for member_path, raw in values.items()
            ),
            key=lambda value: value.path,
        )
    )
    train_logs = json.loads(values[p0.TRAIN_LOGS_MEMBER])
    validation_logs = json.loads(values[p0.VALIDATION_LOGS_MEMBER])
    knowledge = json.loads(values[p0.KNOWLEDGE_MEMBER])
    snippet_count = sum(
        len(entity["docs"])
        for entities in knowledge.values()
        for entity in entities.values()
    )
    contract = p0.QualificationContract(
        bundle_filename=path.name,
        bundle_size_bytes=path.stat().st_size,
        bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        members=members,
        expected_knowledge_snippets=snippet_count,
        expected_split_rows={
            "TRAIN": len(train_logs),
            "VALIDATION": len(validation_logs),
        },
        minimum_unique_dialogue_groups={
            split: {family: minimum for family in p0.FAMILIES}
            for split in p0.SPLITS
        },
        public_example_utterance_sha256=public_hashes,
    )
    return path, contract


def _qualify(
    root: Path,
    path: Path,
    contract: p0.QualificationContract,
) -> dict[str, Any]:
    with mock.patch.object(
        p0,
        "_ijson_basic_parse",
        _synthetic_basic_parse,
    ):
        return p0.qualify_bundle(
            bundle_path=path,
            eligibility_manifest_path=(
                root / "eligibility.private.json"
            ),
            qualification_contract=contract,
            typed_core_sha256=TYPED_SHA256,
        )


def test_official_bundle_and_member_contract_is_exact() -> None:
    assert p0.BUNDLE_SIZE_BYTES == 116_961_280
    assert p0.BUNDLE_SHA256 == (
        "6c3efa690a0829a97836dbb55bf9069b581a39e278a9601cb80a5338d21ffb83"
    )
    assert {member.path for member in p0.OFFICIAL_MEMBER_CONTRACTS} == {
        "FAQ.md",
        "LICENSE",
        "NOTICE",
        "data/knowledge.json",
        "data/train/labels.json",
        "data/train/logs.json",
        "data/val/labels.json",
        "data/val/logs.json",
    }
    assert p0.EXPECTED_SPLIT_ROWS == {
        "TRAIN": 71_348,
        "VALIDATION": 9_663,
    }
    assert p0.TYPED_CORE_SHA256 == TYPED_SHA256
    assert p0.core.VERSION == "dstc9_p1_typed_core_v1"
    assert p0.core.STUDY_ID == p0.STUDY_ID


def test_success_is_safe_aggregate_and_private_manifest_is_opaque(
    posix_tmp: Path,
) -> None:
    values = _values()
    path, contract = _bundle(posix_tmp, values)
    receipt = _qualify(posix_tmp, path, contract)
    assert receipt["status"] == (
        "qualified_public_non_scoring_schema_prefix_group_and_capacity"
    )
    assert receipt["archive_topology"]["regular_member_count"] == 8
    assert receipt["archive_topology"]["test_member_count"] == 0
    assert set(
        receipt["access_boundary"]["payload_open_counts"].values()
    ) == {1}
    assert receipt["knowledge_aggregate"]["snippet_count"] == 4
    for split in p0.SPLITS:
        assert receipt["final_eligible_aggregate"][split][
            "family_unique_dialogue_group_count"
        ] == {family: 1 for family in p0.FAMILIES}
    rendered = p0.canonical_bytes(receipt).decode("ascii")
    for forbidden in (
        '"entity_id"',
        '"doc_id"',
        "Unique TRAIN",
        "grounded body",
        "Grounded response",
    ):
        assert forbidden not in rendered

    private_path = posix_tmp / "eligibility.private.json"
    manifest = json.loads(private_path.read_text("ascii"))
    assert private_path.stat().st_mode & 0o777 == 0o600
    assert manifest["self_sha256"] == p0.stable_hash(
        {
            key: value
            for key, value in manifest.items()
            if key != "self_sha256"
        }
    )
    for rows in manifest["eligible_rows_by_split"].values():
        for row in rows:
            assert set(row) == {
                "opaque_item_id",
                "domain",
                "family",
                "normalized_query_sha256",
                "dialogue_group_sha256",
            }
            assert row["domain"] == row["family"]


def test_archive_identity_fails_before_tar_open(
    posix_tmp: Path,
) -> None:
    path, contract = _bundle(posix_tmp, _values())
    wrong = replace(contract, bundle_sha256="f" * 64)
    with mock.patch.object(
        p0.tarfile,
        "open",
        side_effect=AssertionError("archive topology opened"),
    ):
        with pytest.raises(
            p0.Dstc9P0QualificationError,
            match="byte identity",
        ):
            _qualify(posix_tmp, path, wrong)


@pytest.mark.parametrize(
    ("extra_member", "metadata_override"),
    (
        (("data/test/logs.json", b"[]"), None),
        (None, (p0.FAQ_MEMBER, "mode", 0o644)),
    ),
)
def test_archive_rejects_test_member_or_metadata_drift(
    posix_tmp: Path,
    extra_member: tuple[str, bytes] | None,
    metadata_override: tuple[str, str, int] | None,
) -> None:
    path, contract = _bundle(
        posix_tmp,
        _values(),
        extra_member=extra_member,
        metadata_override=metadata_override,
    )
    with pytest.raises(p0.Dstc9P0QualificationError):
        _qualify(posix_tmp, path, contract)


def test_unresolved_singleton_qrel_fails_closed(
    posix_tmp: Path,
) -> None:
    train_logs, train_labels = _base_split("TRAIN")
    train_labels[0]["knowledge"][0]["doc_id"] = "unknown-doc"
    values = _values(train=(train_logs, train_labels))
    path, contract = _bundle(posix_tmp, values)
    with pytest.raises(
        p0.Dstc9P0QualificationError,
        match="does not resolve",
    ):
        _qualify(posix_tmp, path, contract)


@pytest.mark.parametrize(
    "bad_log",
    (
        [
            {"speaker": "S", "text": "System-first history"},
            {"speaker": "U", "text": "Final user history"},
        ],
        [
            {
                "speaker": "U" if ordinal % 2 == 0 else "S",
                "text": f"Bounded turn {ordinal}",
            }
            for ordinal in range(p0.core.MAX_HISTORY_TURNS + 1)
        ],
    ),
)
def test_logs_must_satisfy_typed_core_history_contract(
    posix_tmp: Path,
    bad_log: list[dict[str, str]],
) -> None:
    train_logs, train_labels = _base_split("TRAIN")
    train_logs[0] = bad_log
    path, contract = _bundle(
        posix_tmp,
        _values(train=(train_logs, train_labels)),
    )
    with pytest.raises(
        p0.Dstc9P0QualificationError,
        match="typed-core contract",
    ):
        _qualify(posix_tmp, path, contract)


def test_ambiguous_prefix_target_is_excluded(
    posix_tmp: Path,
) -> None:
    train_logs, train_labels = _base_split("TRAIN")
    prefix = [{"speaker": "U", "text": "Ambiguous prefix fixture"}]
    train_logs.extend(
        [
            prefix,
            prefix
            + [
                {"speaker": "S", "text": "Branch A"},
                {"speaker": "U", "text": "Leaf A"},
            ],
            prefix
            + [
                {"speaker": "S", "text": "Branch B"},
                {"speaker": "U", "text": "Leaf B"},
            ],
        ]
    )
    train_labels.extend(
        [_target_label("hotel"), {"target": False}, {"target": False}]
    )
    path, contract = _bundle(
        posix_tmp,
        _values(train=(train_logs, train_labels)),
    )
    receipt = _qualify(posix_tmp, path, contract)
    assert receipt["eligibility_exclusion_aggregate"]["TRAIN"][
        "reason_row_count"
    ]["ambiguous_prefix"] == 1
    assert receipt["prefix_trie_aggregate"][
        "ambiguous_log_row_count"
    ]["TRAIN"] >= 1


def test_within_split_duplicate_query_group_is_retained(
    posix_tmp: Path,
) -> None:
    train_logs, train_labels = _base_split("TRAIN")
    train_logs.append(json.loads(json.dumps(train_logs[0])))
    train_labels.append(_target_label("hotel"))
    path, contract = _bundle(
        posix_tmp,
        _values(train=(train_logs, train_labels)),
    )
    receipt = _qualify(posix_tmp, path, contract)
    grouping = receipt["final_eligible_aggregate"]["TRAIN"][
        "normalized_query_grouping"
    ]
    assert grouping["duplicate_group_count"] == 1
    assert grouping["duplicate_row_count"] == 2
    assert grouping["maximum_selected_items_per_group"] == 1


def test_public_example_hash_excludes_complete_dialogue_group(
    posix_tmp: Path,
) -> None:
    train_logs, train_labels = _base_split("TRAIN")
    public_text = "Public fixture utterance"
    prefix = [{"speaker": "U", "text": public_text}]
    train_logs.extend(
        [
            prefix,
            prefix
            + [
                {"speaker": "S", "text": "Unique system continuation"},
                {"speaker": "U", "text": "Unique final continuation"},
            ],
        ]
    )
    train_labels.extend(
        [_target_label("hotel"), _target_label("hotel")]
    )
    public_hash = p0.public_utterance_sha256(public_text)
    path, contract = _bundle(
        posix_tmp,
        _values(train=(train_logs, train_labels)),
        public_hashes=frozenset({public_hash}),
    )
    receipt = _qualify(posix_tmp, path, contract)
    assert receipt["eligibility_exclusion_aggregate"]["TRAIN"][
        "reason_row_count"
    ]["public_example_group"] == 2
    assert public_hash not in p0.canonical_bytes(receipt).decode("ascii")


def test_cross_split_normalized_query_group_is_excluded_bilaterally(
    posix_tmp: Path,
) -> None:
    train_logs, train_labels = _base_split("TRAIN")
    val_logs, val_labels = _base_split("VALIDATION")
    shared = [{"speaker": "U", "text": "Shared cross split fixture"}]
    train_logs.append(shared)
    train_labels.append(_target_label("hotel"))
    val_logs.append(json.loads(json.dumps(shared)))
    val_labels.append(_target_label("hotel"))
    path, contract = _bundle(
        posix_tmp,
        _values(
            train=(train_logs, train_labels),
            validation=(val_logs, val_labels),
        ),
    )
    receipt = _qualify(posix_tmp, path, contract)
    assert receipt["cross_split_query_aggregate"][
        "pre_exclusion_overlap_group_count"
    ] == 1
    for split in p0.SPLITS:
        assert receipt["eligibility_exclusion_aggregate"][split][
            "reason_row_count"
        ]["cross_split_query_group"] == 1
    manifest = json.loads(
        (posix_tmp / "eligibility.private.json").read_text("ascii")
    )
    train_hashes = {
        row["normalized_query_sha256"]
        for row in manifest["eligible_rows_by_split"]["TRAIN"]
    }
    val_hashes = {
        row["normalized_query_sha256"]
        for row in manifest["eligible_rows_by_split"]["VALIDATION"]
    }
    assert not train_hashes & val_hashes


def test_post_exclusion_unique_group_capacity_fails_before_manifest(
    posix_tmp: Path,
) -> None:
    path, contract = _bundle(
        posix_tmp,
        _values(),
        minimum=2,
    )
    with pytest.raises(
        p0.Dstc9P0QualificationError,
        match="capacity",
    ):
        _qualify(posix_tmp, path, contract)
    assert not (posix_tmp / "eligibility.private.json").exists()


def test_failure_main_writes_exclusive_safe_terminal_without_values(
    posix_tmp: Path,
) -> None:
    train_logs, train_labels = _base_split("TRAIN")
    train_labels[0]["knowledge"][0]["entity_id"] = "unresolved-entity"
    values = _values(train=(train_logs, train_labels))
    path, contract = _bundle(posix_tmp, values)
    private = posix_tmp / "failed.private.json"
    safe = posix_tmp / "safe.terminal.json"
    with mock.patch.object(
        p0,
        "_ijson_basic_parse",
        _synthetic_basic_parse,
    ):
        result = p0.main(
            [
                "--bundle",
                str(path),
                "--private-eligibility-manifest",
                str(private),
                "--safe-terminal",
                str(safe),
                "--typed-core-sha256",
                TYPED_SHA256,
            ],
            qualification_contract=contract,
        )
    assert result == 2
    assert not private.exists()
    terminal = json.loads(safe.read_text("ascii"))
    assert safe.stat().st_mode & 0o777 == 0o600
    assert terminal["status"] == "terminal_p0_failed_no_retry"
    assert terminal["error_code"] == "knowledge_reference_unresolved"
    assert terminal["stage"] == "TRAIN_labels_JSON"
    assert terminal["payload_open_counts"] == {
        "FAQ_identity": 1,
        "LICENSE_identity": 1,
        "NOTICE_identity": 1,
        "knowledge_JSON": 1,
        "TRAIN_labels_JSON": 1,
        "TRAIN_logs_JSON": 1,
        "VALIDATION_labels_JSON": 0,
        "VALIDATION_logs_JSON": 0,
    }
    assert terminal["self_sha256"] == p0.stable_hash(
        {
            key: value
            for key, value in terminal.items()
            if key != "self_sha256"
        }
    )
    rendered = p0.canonical_bytes(terminal).decode("ascii")
    assert "unresolved-entity" not in rendered
    assert "Unique TRAIN" not in rendered
