from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import tempfile
from typing import Any, Iterator

import pytest

from assumption_agent.benchmarks import (
    dstc9_p0_public_source_qualification_v1 as p0,
)
from assumption_agent.benchmarks import dstc9_p1_formal_source_v1 as formal
from assumption_agent.benchmarks import dstc9_p1_typed_core_v1 as core


@pytest.fixture
def posix_tmp() -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="dstc9-formal-", dir="/tmp"))
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


def _synthetic_basic_parse(source: Any) -> Iterator[tuple[str, Any]]:
    yield from _events(json.load(source))


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - exact Git blob identity fixture.
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()


def _canonical_file(path: Path, value: object, *, mode: int = 0o600) -> None:
    path.write_bytes(formal.canonical_bytes(value, newline=True))
    path.chmod(mode)


def _knowledge() -> dict[str, Any]:
    # Insertion order is intentionally nonnumeric.  Formal corpus order must
    # still be family -> source-native entity -> numeric document.  Taxi and
    # train exercise DSTC9's null-name ``*`` entity sentinel.
    result: dict[str, Any] = {}
    for family in formal.FAMILIES:
        if family in {"taxi", "train"}:
            result[family] = {
                "*": {
                    "name": None,
                    "docs": {
                        "20": {
                            "title": f"{family} wildcard document 20",
                            "body": f"{family} body wildcard d20",
                        },
                        "10": {
                            "title": f"{family} wildcard document 10",
                            "body": f"{family} body wildcard d10",
                        },
                        "2": {
                            "title": f"{family} wildcard document 2",
                            "body": f"{family} body wildcard d2",
                        },
                        "0": {
                            "title": f"{family} wildcard document 0",
                            "body": f"{family} body wildcard d0",
                        },
                    },
                }
            }
            continue
        result[family] = {
            "10": {
                "name": f"{family} 10",
                "docs": {
                    "10": {
                        "title": f"{family} entity 10 document 10",
                        "body": f"{family} body e10 d10",
                    },
                    "2": {
                        "title": f"{family} entity 10 document 2",
                        "body": f"{family} body e10 d2",
                    },
                },
            },
            "2": {
                "name": None if family in {"taxi", "train"} else f"{family} 2",
                "docs": {
                    "10": {
                        "title": f"{family} entity 2 document 10",
                        "body": f"{family} body e2 d10",
                    },
                    "2": {
                        "title": f"{family} entity 2 document 2",
                        "body": f"{family} body e2 d2",
                    },
                },
            },
        }
    return result


def _split_values(
    split: str,
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    logs: list[Any] = []
    labels: list[Any] = []
    manifest_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(formal.FAMILIES):
        references = (
            (("*", "0"), ("*", "2"), ("*", "10"))
            if family in {"taxi", "train"}
            else (("2", "2"), ("2", "10"), ("10", "2"))
        )
        duplicate_group = formal.stable_hash(
            {"family": family, "split": split, "synthetic_group": 0}
        )
        singleton_group = formal.stable_hash(
            {"family": family, "split": split, "synthetic_group": 1}
        )
        for local_index, (entity_id, doc_id) in enumerate(references):
            source_ordinal = len(logs)
            history = [
                {
                    "speaker": "U",
                    "text": (
                        f"{split} {family} unique dialogue "
                        f"{family_index}-{local_index}?"
                    ),
                }
            ]
            typed_history = [
                core.DialogueTurn(
                    speaker=turn["speaker"],
                    text=turn["text"],
                )
                for turn in history
            ]
            query_sha256 = core.normalized_query_sha256(typed_history)
            item_id = p0.stable_hash(
                {"source_ordinal": source_ordinal, "split": split}
            )
            logs.append(history)
            labels.append(
                {
                    "knowledge": [
                        {
                            "doc_id": int(doc_id),
                            "domain": family,
                            "entity_id": (
                                entity_id
                                if entity_id == "*"
                                else int(entity_id)
                            ),
                        }
                    ],
                    "response": f"Private response {split} {family} {local_index}",
                    "target": True,
                }
            )
            manifest_rows.append(
                {
                    "dialogue_group_sha256": (
                        duplicate_group if local_index < 2 else singleton_group
                    ),
                    "domain": family,
                    "family": family,
                    "normalized_query_sha256": query_sha256,
                    "opaque_item_id": item_id,
                }
            )
    return logs, labels, manifest_rows


def _source_values() -> tuple[dict[str, bytes], dict[str, list[dict[str, Any]]]]:
    train_logs, train_labels, train_rows = _split_values("TRAIN")
    validation_logs, validation_labels, validation_rows = _split_values(
        "VALIDATION"
    )
    semantic: dict[str, Any] = {
        p0.KNOWLEDGE_MEMBER: _knowledge(),
        p0.TRAIN_LOGS_MEMBER: train_logs,
        p0.TRAIN_LABELS_MEMBER: train_labels,
        p0.VALIDATION_LOGS_MEMBER: validation_logs,
        p0.VALIDATION_LABELS_MEMBER: validation_labels,
    }
    return (
        {
            p0.FAQ_MEMBER: b"synthetic identity-only FAQ\n",
            p0.LICENSE_MEMBER: b"synthetic identity-only license\n",
            p0.NOTICE_MEMBER: b"synthetic identity-only notice\n",
            **{
                path: json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                for path, value in semantic.items()
            },
        },
        {
            "TRAIN": train_rows,
            "VALIDATION": validation_rows,
        },
    )


def _bundle(
    root: Path,
    values: dict[str, bytes],
) -> tuple[Path, p0.QualificationContract]:
    bundle = root / "synthetic.bundle.tar"
    with tarfile.open(
        bundle,
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
            info.type = tarfile.REGTYPE
            archive.addfile(info, fileobj=_BytesReader(raw))
    bundle.chmod(0o600)
    member_contracts = tuple(
        p0.MemberContract(
            path=member_path,
            size_bytes=len(values[member_path]),
            sha256=hashlib.sha256(values[member_path]).hexdigest(),
            git_blob_sha1=_git_blob_sha1(values[member_path]),
        )
        for member_path in sorted(values)
    )
    return bundle, p0.QualificationContract(
        bundle_filename=bundle.name,
        bundle_size_bytes=bundle.stat().st_size,
        bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        members=member_contracts,
        expected_knowledge_snippets=16,
        expected_split_rows={"TRAIN": 12, "VALIDATION": 12},
        minimum_unique_dialogue_groups={
            split: {family: 2 for family in formal.FAMILIES}
            for split in formal.SPLITS
        },
        public_example_utterance_sha256=frozenset(),
    )


class _BytesReader:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.raw) - self.offset
        result = self.raw[self.offset : self.offset + size]
        self.offset += len(result)
        return result


def _member_identity(
    source_contract: p0.QualificationContract,
) -> dict[str, dict[str, Any]]:
    return {
        member.path: {
            "git_blob_sha1": member.git_blob_sha1,
            "payload_open_count": 1,
            "sha256": member.sha256,
            "size_bytes": member.size_bytes,
        }
        for member in source_contract.members
    }


def _typed_binding() -> dict[str, str]:
    return {
        "sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "study_id": formal.STUDY_ID,
        "version": core.VERSION,
    }


@dataclass(frozen=True)
class _Fixture:
    bundle: Path
    p0_receipt: Path
    private_manifest: Path
    contract: formal.FormalSourceContract
    outputs: formal.FormalOutputPaths
    manifest_rows: Mapping[str, list[dict[str, Any]]]


def _outputs(root: Path, label: str = "out") -> formal.FormalOutputPaths:
    output = root / label
    output.mkdir()
    return formal.FormalOutputPaths(
        public_corpus=output / "corpus.public.json",
        public_a_form=output / "A_form.public.json",
        public_f_search=output / "F_search.public.json",
        public_a_hold=output / "A_hold.public.json",
        public_m_search=output / "M_search.public.sealed.json",
        private_a_form_qrels=output / "A_form.qrels.private.json",
        private_a_hold_qrels=output / "A_hold.qrels.private.json",
        private_m_search_qrels=output / "M_search.qrels.private.sealed.json",
        safe_selection_receipt=output / "selection.safe.json",
    )


def _fixture(root: Path) -> _Fixture:
    values, rows_by_split = _source_values()
    bundle, source_contract = _bundle(root, values)
    typed_binding = _typed_binding()
    sorted_rows = {
        split: sorted(
            rows_by_split[split],
            key=lambda row: (
                row["family"],
                row["dialogue_group_sha256"],
                row["opaque_item_id"],
            ),
        )
        for split in formal.SPLITS
    }
    private_manifest = p0.self_hashed(
        {
            "eligibility_rule_version": p0.ELIGIBILITY_RULE_VERSION,
            "eligible_rows_by_split": sorted_rows,
            "query_group_contract": {
                "cross_split_policy": "exclude_all_rows",
                "group_field": "normalized_query_sha256",
                "maximum_selected_items_per_group": 1,
            },
            "source_binding": {
                "bundle_sha256": source_contract.bundle_sha256,
                "bundle_size_bytes": source_contract.bundle_size_bytes,
                "commit": p0.OFFICIAL_COMMIT,
                "member_identity": _member_identity(source_contract),
                "repository": p0.OFFICIAL_REPOSITORY,
            },
            "study_id": formal.STUDY_ID,
            "typed_core_binding": typed_binding,
            "version": p0.VERSION,
        }
    )
    private_path = root / "eligibility.private.json"
    _canonical_file(private_path, private_manifest)
    private_raw = private_path.read_bytes()

    row_counts = {
        split: len(sorted_rows[split]) for split in formal.SPLITS
    }
    final = {
        split: {
            "family_unique_dialogue_group_count": {
                family: 2 for family in formal.FAMILIES
            },
            "normalized_query_grouping": {
                "duplicate_group_count": 0,
                "duplicate_row_count": 0,
                "excess_duplicate_row_count": 0,
                "group_count": row_counts[split],
                "maximum_selected_items_per_group": 1,
            },
            "row_count": row_counts[split],
        }
        for split in formal.SPLITS
    }
    p0_receipt = p0.self_hashed(
        {
            "access_boundary": {
                "action_model_evaluator_score_or_secret_count": 0,
                "bundle_full_extraction_count": 0,
                "individual_identifier_text_entity_doc_qrel_or_row_hash_output_count": 0,
                "online_or_API_evaluation_count": 0,
                "payload_member_reopen_count": 0,
                "payload_open_counts": {
                    "FAQ_identity": 1,
                    "LICENSE_identity": 1,
                    "NOTICE_identity": 1,
                    "TRAIN_labels_JSON": 1,
                    "TRAIN_logs_JSON": 1,
                    "VALIDATION_labels_JSON": 1,
                    "VALIDATION_logs_JSON": 1,
                    "knowledge_JSON": 1,
                },
                "test_member_count": 0,
            },
            "archive_topology": {
                "directory_link_or_special_member_count": 0,
                "mode_0600_member_count": 8,
                "mtime_zero_member_count": 8,
                "regular_member_count": 8,
                "test_member_count": 0,
                "uid_gid_zero_member_count": 8,
                "ustar_header_count": 8,
            },
            "cross_split_query_aggregate": {
                "post_exclusion_overlap_group_count": 0,
                "pre_exclusion_overlap_group_count": 0,
            },
            "eligibility_exclusion_aggregate": {},
            "final_eligible_aggregate": final,
            "knowledge_aggregate": {
                "snippet_count": source_contract.expected_knowledge_snippets
            },
            "member_receipts": _member_identity(source_contract),
            "prefix_trie_aggregate": {},
            "private_manifest_binding": {
                "file_sha256": hashlib.sha256(private_raw).hexdigest(),
                "row_count": row_counts,
                "self_sha256": private_manifest["self_sha256"],
                "size_bytes": len(private_raw),
            },
            "public_example_exclusion_binding": {},
            "source": {
                "bundle_filename": source_contract.bundle_filename,
                "bundle_sha256": source_contract.bundle_sha256,
                "bundle_size_bytes": source_contract.bundle_size_bytes,
                "commit": p0.OFFICIAL_COMMIT,
                "repository": p0.OFFICIAL_REPOSITORY,
            },
            "split_source_aggregate": {},
            "status": (
                "qualified_public_non_scoring_schema_prefix_group_and_capacity"
            ),
            "study_id": formal.STUDY_ID,
            "typed_core_binding": typed_binding,
            "version": p0.VERSION,
        }
    )
    receipt_path = root / "qualification.safe.json"
    _canonical_file(receipt_path, p0_receipt)
    receipt_raw = receipt_path.read_bytes()
    contract = formal.FormalSourceContract(
        source_contract=source_contract,
        p0_receipt_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        p0_receipt_self_sha256=p0_receipt["self_sha256"],
        private_manifest_file_sha256=hashlib.sha256(private_raw).hexdigest(),
        private_manifest_self_sha256=private_manifest["self_sha256"],
        typed_core_sha256=typed_binding["sha256"],
        selection_seed=formal.SELECTION_SEED,
        block_family_quotas={
            block: {family: 1 for family in formal.FAMILIES}
            for block in formal.BLOCKS
        },
    )
    return _Fixture(
        bundle=bundle,
        p0_receipt=receipt_path,
        private_manifest=private_path,
        contract=contract,
        outputs=_outputs(root),
        manifest_rows=sorted_rows,
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _compile(
    fixture: _Fixture,
    *,
    outputs: formal.FormalOutputPaths | None = None,
) -> dict[str, Any]:
    return formal.compile_formal_source(
        p0_receipt_path=fixture.p0_receipt,
        private_eligibility_manifest_path=fixture.private_manifest,
        bundle_path=fixture.bundle,
        outputs=fixture.outputs if outputs is None else outputs,
        contract=fixture.contract,
    )


def test_exact_quota_disjoint_leak_free_mapping_and_one_open(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p0, "_ijson_basic_parse", _synthetic_basic_parse)
    fixture = _fixture(posix_tmp)
    receipt = _compile(fixture)

    assert receipt["status"] == "selected_and_sealed"
    for path in fixture.outputs.all_paths():
        payload = _load(path)
        body = dict(payload)
        claimed = body.pop("self_sha256")
        assert formal.stable_hash(body) == claimed
    assert receipt["quota"] == {
        block: {family: 1 for family in formal.FAMILIES}
        for block in formal.BLOCKS
    }
    access = receipt["source_access"]
    assert access["formal_source_access_count"] == 1
    assert access["test_member_count"] == 0
    assert access["identity_member_payload_open_count"] == 0
    assert access["payload_member_reopen_count"] == 0
    assert access["payload_member_open_counts"] == {
        "FAQ_identity": 0,
        "LICENSE_identity": 0,
        "NOTICE_identity": 0,
        "TRAIN_labels_JSON": 1,
        "TRAIN_logs_JSON": 1,
        "VALIDATION_labels_JSON": 1,
        "VALIDATION_logs_JSON": 1,
        "knowledge_JSON": 1,
    }
    assert receipt["compiler_boundary"] == {
        "action_count": 0,
        "model_call_count": 0,
        "online_or_API_evaluation_count": 0,
        "score_count": 0,
    }
    assert receipt["disjointness_aggregate"] == {
        "cross_block_dialogue_group_overlap_count": 0,
        "cross_block_item_overlap_count": 0,
        "cross_block_normalized_query_overlap_count": 0,
        "selected_dialogue_group_count": 16,
        "selected_item_count": 16,
        "selected_normalized_query_count": 16,
    }

    corpus = _load(fixture.outputs.public_corpus)
    snippets = corpus["snippets"]
    assert len(snippets) == 16
    assert [set(row) for row in snippets] == [
        {"body", "entity_name", "ordinal", "title"}
    ] * 16
    assert [row["ordinal"] for row in snippets] == list(range(16))
    assert [row["title"] for row in snippets[:4]] == [
        "hotel entity 2 document 2",
        "hotel entity 2 document 10",
        "hotel entity 10 document 2",
        "hotel entity 10 document 10",
    ]

    work_ids: set[str] = set()
    query_hashes: set[str] = set()
    for block in formal.BLOCKS:
        public = _load(fixture.outputs.public_blocks()[block])
        assert public["block_id"] == block
        assert len(public["items"]) == 4
        assert all(
            set(item) == formal.PUBLIC_ITEM_KEYS
            for item in public["items"]
        )
        assert all(
            set(turn) == formal.PUBLIC_TURN_KEYS
            for item in public["items"]
            for turn in item["history"]
        )
        for item in public["items"]:
            assert item["work_id"] not in work_ids
            assert re.fullmatch(
                r"dstc9-work-v1-[0-9a-f]{64}",
                item["work_id"],
            )
            assert item["normalized_query_sha256"] not in query_hashes
            assert core.normalized_query_sha256(
                [
                    core.turn_from_public_fields(turn)
                    for turn in item["history"]
                ]
            ) == item["normalized_query_sha256"]
            work_ids.add(item["work_id"])
            query_hashes.add(item["normalized_query_sha256"])

        raw_public = fixture.outputs.public_blocks()[block].read_bytes()
        for forbidden in (
            b'"split"',
            b'"family"',
            b'"domain"',
            b'"entity_id"',
            b'"doc_id"',
            b'"qrel"',
            b'"label"',
            b'"response"',
            b'"gold_ordinal"',
        ):
            assert forbidden not in raw_public

    assert set(fixture.outputs.private_qrels()) == set(
        formal.QREL_BLOCKS
    )
    assert set(receipt["artifact_binding"]["private_qrels"]) == set(
        formal.QREL_BLOCKS
    )
    for block in formal.QREL_BLOCKS:
        public = _load(fixture.outputs.public_blocks()[block])
        qrels = _load(fixture.outputs.private_qrels()[block])
        assert public["block_id"] == qrels["block_id"] == block
        assert len(qrels["qrels"]) == 4
        assert all(
            set(row) == formal.PRIVATE_QREL_ROW_KEYS
            for row in qrels["qrels"]
        )
        assert [item["work_id"] for item in public["items"]] == [
            row["work_id"] for row in qrels["qrels"]
        ]
        for row in qrels["qrels"]:
            assert row["family"] in formal.FAMILIES
            assert 0 <= row["gold_ordinal"] < 16

    assert len(work_ids) == len(query_hashes) == 16
    assert stat.S_IMODE(fixture.outputs.public_corpus.stat().st_mode) == 0o600
    assert (
        stat.S_IMODE(
            fixture.outputs.public_blocks()["M_search"].stat().st_mode
        )
        == 0o400
    )
    for block in ("A_form", "F_search", "A_hold"):
        assert (
            stat.S_IMODE(
                fixture.outputs.public_blocks()[block].stat().st_mode
            )
            == 0o600
        )
    for path in fixture.outputs.private_qrels().values():
        assert stat.S_IMODE(path.stat().st_mode) == 0o400

    # Exact gold mapping for numeric source id order.
    family_offset = {
        family: index * 4
        for index, family in enumerate(formal.FAMILIES)
    }
    expected_gold = {
        f"{split} {family} unique dialogue {family_index}-0?": (
            family_offset[family]
        )
        for split in formal.SPLITS
        for family_index, family in enumerate(formal.FAMILIES)
    }
    expected_gold.update(
        {
            f"{split} {family} unique dialogue {family_index}-1?": (
                family_offset[family] + 1
            )
            for split in formal.SPLITS
            for family_index, family in enumerate(formal.FAMILIES)
        }
    )
    expected_gold.update(
        {
            f"{split} {family} unique dialogue {family_index}-2?": (
                family_offset[family] + 2
            )
            for split in formal.SPLITS
            for family_index, family in enumerate(formal.FAMILIES)
        }
    )
    observed_by_work: dict[str, str] = {}
    for block in formal.BLOCKS:
        public = _load(fixture.outputs.public_blocks()[block])
        for item in public["items"]:
            observed_by_work[item["work_id"]] = item["history"][0]["text"]
    for block in formal.QREL_BLOCKS:
        qrels = _load(fixture.outputs.private_qrels()[block])
        for row in qrels["qrels"]:
            assert row["gold_ordinal"] == expected_gold[
                observed_by_work[row["work_id"]]
            ]


def test_selection_is_byte_deterministic_and_represents_each_group_once(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p0, "_ijson_basic_parse", _synthetic_basic_parse)
    fixture = _fixture(posix_tmp)
    first = _compile(fixture)
    second_outputs = _outputs(posix_tmp, "out-second")
    second = _compile(fixture, outputs=second_outputs)
    assert first == second
    for left, right in zip(
        fixture.outputs.all_paths(),
        second_outputs.all_paths(),
    ):
        assert left.read_bytes() == right.read_bytes()

    selected_queries = {
        item["normalized_query_sha256"]
        for block in formal.BLOCKS
        for item in _load(
            fixture.outputs.public_blocks()[block]
        )["items"]
    }
    # Each family/split fixture has two candidates in group 0 and one in
    # group 1.  Exactly one of the first two and the singleton are selected.
    for split in formal.SPLITS:
        for family in formal.FAMILIES:
            rows = [
                row
                for row in fixture.manifest_rows[split]
                if row["family"] == family
            ]
            duplicate_queries = {
                rows[0]["normalized_query_sha256"],
                rows[1]["normalized_query_sha256"],
            }
            assert len(selected_queries & duplicate_queries) == 1
            assert rows[2]["normalized_query_sha256"] in selected_queries


@pytest.mark.parametrize(
    "mutation",
    ("private_mode", "receipt_self_hash", "cross_split_group"),
)
def test_invalid_binding_fails_before_source_open_and_writes_nothing(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setattr(p0, "_ijson_basic_parse", _synthetic_basic_parse)
    fixture = _fixture(posix_tmp)
    open_calls: list[str] = []
    original_open_member = p0._open_member

    def _counting_open(*args: Any, **kwargs: Any) -> Any:
        member = args[1]
        open_calls.append(member.name)
        return original_open_member(*args, **kwargs)

    monkeypatch.setattr(p0, "_open_member", _counting_open)
    if mutation == "private_mode":
        fixture.private_manifest.chmod(0o644)
        match = "metadata"
    elif mutation == "receipt_self_hash":
        value = _load(fixture.p0_receipt)
        value["self_sha256"] = "f" * 64
        _canonical_file(fixture.p0_receipt, value)
        # Preserve file binding in a test-only derived contract so the
        # self-hash validation, not the byte hash, is the failing boundary.
        raw = fixture.p0_receipt.read_bytes()
        object.__setattr__(
            fixture.contract,
            "p0_receipt_file_sha256",
            hashlib.sha256(raw).hexdigest(),
        )
        match = "self hash"
    else:
        value = _load(fixture.private_manifest)
        rows = value["eligible_rows_by_split"]
        rows["VALIDATION"][0]["dialogue_group_sha256"] = (
            rows["TRAIN"][0]["dialogue_group_sha256"]
        )
        rows["VALIDATION"] = sorted(
            rows["VALIDATION"],
            key=lambda row: (
                row["family"],
                row["dialogue_group_sha256"],
                row["opaque_item_id"],
            ),
        )
        body = dict(value)
        body.pop("self_sha256")
        value["self_sha256"] = p0.stable_hash(body)
        _canonical_file(fixture.private_manifest, value)
        private_raw = fixture.private_manifest.read_bytes()
        # Rebind only the byte/self identities and corresponding P0 binding;
        # semantic cross-split overlap must still fail before source access.
        object.__setattr__(
            fixture.contract,
            "private_manifest_file_sha256",
            hashlib.sha256(private_raw).hexdigest(),
        )
        object.__setattr__(
            fixture.contract,
            "private_manifest_self_sha256",
            value["self_sha256"],
        )
        receipt = _load(fixture.p0_receipt)
        receipt["private_manifest_binding"]["file_sha256"] = (
            fixture.contract.private_manifest_file_sha256
        )
        receipt["private_manifest_binding"]["self_sha256"] = (
            fixture.contract.private_manifest_self_sha256
        )
        receipt["private_manifest_binding"]["size_bytes"] = len(private_raw)
        body = dict(receipt)
        body.pop("self_sha256")
        receipt["self_sha256"] = p0.stable_hash(body)
        _canonical_file(fixture.p0_receipt, receipt)
        receipt_raw = fixture.p0_receipt.read_bytes()
        object.__setattr__(
            fixture.contract,
            "p0_receipt_file_sha256",
            hashlib.sha256(receipt_raw).hexdigest(),
        )
        object.__setattr__(
            fixture.contract,
            "p0_receipt_self_sha256",
            receipt["self_sha256"],
        )
        match = "block-disjoint"

    with pytest.raises(formal.Dstc9P1FormalSourceError, match=match):
        _compile(fixture)
    assert open_calls == []
    assert not any(path.exists() for path in fixture.outputs.all_paths())


def test_extra_test_member_is_rejected_without_payload_open(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p0, "_ijson_basic_parse", _synthetic_basic_parse)
    fixture = _fixture(posix_tmp)
    values, _rows = _source_values()
    values["data/test/logs.json"] = b"[]"
    tampered = posix_tmp / "with-test.bundle.tar"
    with tarfile.open(
        tampered,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for member_path in sorted(values):
            raw = values[member_path]
            info = tarfile.TarInfo(member_path)
            info.size = len(raw)
            info.mode = 0o600
            info.uid = info.gid = info.mtime = 0
            archive.addfile(info, fileobj=_BytesReader(raw))
    tampered.chmod(0o600)
    # Rebind only whole-bundle identity.  The frozen eight-member topology
    # remains unchanged and must reject TEST before any payload extractfile.
    source_contract = p0.QualificationContract(
        bundle_filename=tampered.name,
        bundle_size_bytes=tampered.stat().st_size,
        bundle_sha256=hashlib.sha256(tampered.read_bytes()).hexdigest(),
        members=fixture.contract.source_contract.members,
        expected_knowledge_snippets=16,
        expected_split_rows={"TRAIN": 12, "VALIDATION": 12},
        minimum_unique_dialogue_groups={
            split: {family: 2 for family in formal.FAMILIES}
            for split in formal.SPLITS
        },
        public_example_utterance_sha256=frozenset(),
    )
    object.__setattr__(fixture.contract, "source_contract", source_contract)

    # Rebind source identities in the private/P0 envelopes to reach topology.
    private = _load(fixture.private_manifest)
    private["source_binding"]["bundle_sha256"] = source_contract.bundle_sha256
    private["source_binding"]["bundle_size_bytes"] = (
        source_contract.bundle_size_bytes
    )
    body = dict(private)
    body.pop("self_sha256")
    private["self_sha256"] = p0.stable_hash(body)
    _canonical_file(fixture.private_manifest, private)
    private_raw = fixture.private_manifest.read_bytes()
    object.__setattr__(
        fixture.contract,
        "private_manifest_file_sha256",
        hashlib.sha256(private_raw).hexdigest(),
    )
    object.__setattr__(
        fixture.contract,
        "private_manifest_self_sha256",
        private["self_sha256"],
    )
    receipt = _load(fixture.p0_receipt)
    receipt["source"]["bundle_filename"] = source_contract.bundle_filename
    receipt["source"]["bundle_sha256"] = source_contract.bundle_sha256
    receipt["source"]["bundle_size_bytes"] = source_contract.bundle_size_bytes
    receipt["private_manifest_binding"]["file_sha256"] = (
        fixture.contract.private_manifest_file_sha256
    )
    receipt["private_manifest_binding"]["self_sha256"] = (
        fixture.contract.private_manifest_self_sha256
    )
    receipt["private_manifest_binding"]["size_bytes"] = len(private_raw)
    body = dict(receipt)
    body.pop("self_sha256")
    receipt["self_sha256"] = p0.stable_hash(body)
    _canonical_file(fixture.p0_receipt, receipt)
    receipt_raw = fixture.p0_receipt.read_bytes()
    object.__setattr__(
        fixture.contract,
        "p0_receipt_file_sha256",
        hashlib.sha256(receipt_raw).hexdigest(),
    )
    object.__setattr__(
        fixture.contract,
        "p0_receipt_self_sha256",
        receipt["self_sha256"],
    )

    open_calls: list[str] = []
    original_open_member = p0._open_member

    def _counting_open(*args: Any, **kwargs: Any) -> Any:
        open_calls.append(args[1].name)
        return original_open_member(*args, **kwargs)

    monkeypatch.setattr(p0, "_open_member", _counting_open)
    with pytest.raises(
        formal.Dstc9P1FormalSourceError,
        match="inherited P0 source validation",
    ) as captured:
        formal.compile_formal_source(
            p0_receipt_path=fixture.p0_receipt,
            private_eligibility_manifest_path=fixture.private_manifest,
            bundle_path=tampered,
            outputs=fixture.outputs,
            contract=fixture.contract,
        )
    assert captured.value.error_code == "p0_archive_topology_mismatch"
    assert open_calls == []
    assert not any(path.exists() for path in fixture.outputs.all_paths())
