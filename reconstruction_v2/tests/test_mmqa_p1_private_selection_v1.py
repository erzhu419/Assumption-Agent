from __future__ import annotations

import ast
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import pytest

from assumption_agent.benchmarks import mmqa_p1_private_selection_v1 as selection


SECRET = bytes(range(32))


@dataclass(frozen=True)
class SyntheticSource:
    root: Path
    paths: dict[str, Path]
    contract: selection.SelectionContract
    receipt_path: Path
    receipt_sha256: str
    table_row_count: int


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    raw = b"".join(
        selection._canonical_bytes(row, newline=True)  # noqa: SLF001
        for row in rows
    )
    return gzip.compress(raw, compresslevel=6, mtime=0)


def _blob(raw: bytes) -> str:
    digest = hashlib.sha1()  # nosec: synthetic Git-object fixture
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _source_fixture(tmp_path: Path) -> SyntheticSource:
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    global_index = 0
    table_row_count = 0

    for exact_type, _family in selection.FAMILY_BY_EXACT_TYPE.items():
        for split, count in (("TRAIN", 40), ("DEV", 35)):
            destination = train_rows if split == "TRAIN" else dev_rows
            for family_index in range(count):
                index = global_index
                global_index += 1
                qid = f"QID_PRIVATE_{index:04d}"
                table_id = f"TABLE_PRIVATE_{index:04d}"
                gold_text_id = f"TEXT_PRIVATE_G_{index:04d}"
                distractor_text_id = f"TEXT_PRIVATE_D_{index:04d}"
                gold_title = f"Gold Topic {index:04d}"
                distractor_title = f"Distractor Topic {index:04d}"

                # The first item proves that gold rows are the union of an
                # exact-linked row and a distinct answer table-index row.
                rows = [
                    [
                        {
                            "text": f"linked cell {index}",
                            "links": [
                                {
                                    "wiki_title": gold_title,
                                    "url": (
                                        "https://en.wikipedia.org/wiki/"
                                        + gold_title.replace(" ", "_")
                                    ),
                                }
                            ],
                        }
                    ]
                ]
                answer_row = 0
                if index == 0:
                    rows.append([{"text": "answer-only row", "links": []}])
                    answer_row = 1
                table_row_count += len(rows)
                table_rows.append(
                    {
                        "id": table_id,
                        "title": f"Table title {index:04d}",
                        "url": f"https://example.test/table/{index}",
                        "table": {
                            "table_name": f"Table name {index:04d}",
                            "header": [{"column_name": "Value"}],
                            "table_rows": rows,
                        },
                    }
                )
                text_rows.extend(
                    (
                        {
                            "id": gold_text_id,
                            "title": gold_title,
                            "url": (
                                "https://en.wikipedia.org/wiki/"
                                + gold_title.replace(" ", "_")
                            ),
                            "text": f"Gold paragraph content {index:04d}.",
                        },
                        {
                            "id": distractor_text_id,
                            "title": distractor_title,
                            "url": f"https://example.test/text/{index}",
                            "text": f"Distractor paragraph content {index:04d}.",
                        },
                    )
                )
                destination.append(
                    {
                        "qid": qid,
                        "question": f"Which linked evidence resolves item {index:04d}?",
                        "answers": [
                            {
                                "answer": f"PRIVATE_ANSWER_{index:04d}",
                                "table_indices": [[answer_row, 0]],
                            }
                        ],
                        "metadata": {
                            "type": exact_type,
                            "modalities": ["table", "text"],
                            "table_id": table_id,
                            "text_doc_ids": [gold_text_id, distractor_text_id],
                            "private_extra": f"PRIVATE_METADATA_{index:04d}",
                        },
                        "supporting_context": [
                            {"doc_part": "table", "doc_id": table_id},
                            {"doc_part": "text", "doc_id": gold_text_id},
                        ],
                    }
                )

    raw_by_name = {
        "MMQA_train.jsonl.gz": _jsonl(train_rows),
        "MMQA_dev.jsonl.gz": _jsonl(dev_rows),
        "MMQA_tables.jsonl.gz": _jsonl(table_rows),
        "MMQA_texts.jsonl.gz": _jsonl(text_rows),
    }
    root = tmp_path / "source"
    root.mkdir(mode=0o700)
    paths: dict[str, Path] = {}
    contracts: dict[str, selection.SourceFileContract] = {}
    maxima = {
        "MMQA_train.jsonl.gz": (20_000_000, 1_000),
        "MMQA_dev.jsonl.gz": (20_000_000, 1_000),
        "MMQA_tables.jsonl.gz": (20_000_000, 1_000),
        "MMQA_texts.jsonl.gz": (20_000_000, 2_000),
    }
    for name, raw in raw_by_name.items():
        path = root / name
        path.write_bytes(raw)
        path.chmod(0o600)
        paths[name] = path
        contracts[name] = selection.SourceFileContract(
            name,
            len(raw),
            _blob(raw),
            maxima[name][0],
            maxima[name][1],
        )
    contract = selection.SelectionContract(
        files=contracts,
        expected_train_rows=len(train_rows),
        expected_dev_rows=len(dev_rows),
    )
    receipt_body = {
        "schema": "mmqa_p1_source_qualification_v1_result_v1",
        "status": "qualified_aggregate_only",
        "study_id": selection.STUDY_ID,
        "qualified": True,
        "binding_self_sha256": {
            "download_authorization": "1" * 64,
            "qualification_freeze": "2" * 64,
            "source_custody": selection.SOURCE_CUSTODY_SELF_SHA256,
            "study_design": selection.STUDY_DESIGN_SELF_SHA256,
        },
        "model_action_embedding_reranking_or_score_count": 0,
        "online_evaluator_call_count": 0,
        "source_item_query_document_answer_support_or_identifier_output_count": 0,
        "marker_file_sha256": "3" * 64,
        "source_open_marker_file_sha256": "4" * 64,
        "source_identity": {
            name: {
                "git_blob_sha1": contracts[name].git_blob_sha1,
                "sha256": hashlib.sha256(raw_by_name[name]).hexdigest(),
                "size_bytes": contracts[name].size_bytes,
            }
            for name in sorted(contracts)
        },
        "TRAIN": {
            "question_record_count": len(train_rows),
            "required_per_family": 40,
            "eligible_count_by_family": {family: 40 for family in selection.FAMILIES},
        },
        "DEV": {
            "question_record_count": len(dev_rows),
            "required_total_per_family": 35,
            "eligible_count_by_family": {family: 35 for family in selection.FAMILIES},
            "component_disjoint_capacity": {"qualified": True},
        },
        "exact_type_family_count": 3,
        "schema_aggregates": {
            "table_record_count": len(table_rows),
            "table_row_count": table_row_count,
            "text_record_count": len(text_rows),
        },
        "support_contract": {
            "answer_table_index_rows_union_exact_linked_rows": True,
            "gold_row_bounds_inclusive": [1, 4],
            "gold_text_bounds_inclusive": [1, 4],
            "identifier_or_content_output_count": 0,
            "requires_exact_gold_row_text_pair": True,
            "support_parts": ["table", "text"],
        },
    }
    receipt = selection.self_hashed(receipt_body)
    receipt_path = tmp_path / "qualification.json"
    receipt_path.write_bytes(selection._canonical_bytes(receipt, newline=True))  # noqa: SLF001
    return SyntheticSource(
        root=root,
        paths=paths,
        contract=contract,
        receipt_path=receipt_path,
        receipt_sha256=receipt["self_sha256"],
        table_row_count=table_row_count,
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("ascii"))
    assert isinstance(value, dict)
    return value


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _acquire(tmp_path: Path) -> tuple[SyntheticSource, Path, dict[str, Any]]:
    source = _source_fixture(tmp_path)
    output = tmp_path / "selection"
    calls: list[int] = []

    def random_bytes(size: int) -> bytes:
        calls.append(size)
        return SECRET

    receipt = selection.acquire_once(
        source_paths=source.paths,
        qualification_path=source.receipt_path,
        expected_qualification_self_sha256=source.receipt_sha256,
        output_root=output,
        contract=source.contract,
        random_bytes=random_bytes,
    )
    assert calls == [32]
    return source, output, receipt


def test_full_one_shot_projection_is_id_free_and_gold_is_separate(
    tmp_path: Path,
) -> None:
    _source, output, receipt = _acquire(tmp_path)
    assert receipt["selection_contract"]["selected_total"] == 225
    oof = receipt["selection_contract"]["A_form_five_fold_OOF"]
    assert oof["fold_count"] == 5
    assert sum(oof["fold_sizes"].values()) == 120
    assert all(value > 0 for value in oof["fold_sizes"].values())
    assert len(oof["assignment_commitment_sha256"]) == 64
    assert receipt["model_network_retrieval_evaluator_or_score_calls"] == 0
    assert _mode(output) == 0o700
    assert _mode(output / selection.SECRET_FILENAME) == 0o600
    assert _mode(output / selection.COMMITMENT_FILENAME) == 0o644
    assert _mode(output / selection.PUBLIC_RECEIPT_FILENAME) == 0o644
    assert (output / selection.SECRET_FILENAME).read_bytes() == SECRET

    ledger = _load(output / selection.PRIVATE_LEDGER_FILENAME)
    assert _mode(output / selection.PRIVATE_LEDGER_FILENAME) == 0o600
    assert ledger["item_count"] == 225
    assert "QID_PRIVATE_" in json.dumps(ledger)
    assert {row["source_family"] for row in ledger["items"]} == set(
        selection.FAMILIES
    )

    resources_by_block: dict[str, set[str]] = {
        block: set() for block in selection.DEV_BLOCK_ORDER
    }
    for row in ledger["items"]:
        if row["block"] in selection.DEV_BLOCK_ORDER:
            resources_by_block[row["block"]].add(row["source_table_id"])
            resources_by_block[row["block"]].update(row["source_text_doc_ids"])
    for left_index, left in enumerate(selection.DEV_BLOCK_ORDER):
        for right in selection.DEV_BLOCK_ORDER[left_index + 1 :]:
            assert resources_by_block[left].isdisjoint(resources_by_block[right])

    special_mapping = next(
        row for row in ledger["items"] if row["source_qid"] == "QID_PRIVATE_0000"
    )
    ledger_by_work = {row["work_id"]: row for row in ledger["items"]}
    for block in selection.BLOCK_ORDER:
        action_path = output / selection.ACTION_PACK_FILENAMES[block]
        gold_path = output / selection.GOLD_PACK_FILENAMES[block]
        assert _mode(action_path) == _mode(gold_path) == 0o600
        action = _load(action_path)
        gold = _load(gold_path)
        assert len(action["items"]) == len(gold["items"]) == selection.BLOCK_ITEM_COUNTS[block]
        assert [row["work_id"] for row in action["items"]] == [
            row["work_id"] for row in gold["items"]
        ]
        raw = action_path.read_text("ascii")
        for forbidden in (
            "QID_PRIVATE_",
            "TABLE_PRIVATE_",
            "TEXT_PRIVATE_",
            "PRIVATE_ANSWER_",
            "PRIVATE_METADATA_",
            *selection.FAMILIES,
            *selection.FAMILY_BY_EXACT_TYPE,
        ):
            assert forbidden not in raw
        for item in action["items"]:
            assert set(item) == {"work_id", "question", "nodes", "edges"}
            assert selection._WORK_ID.fullmatch(item["work_id"])  # noqa: SLF001
            assert [node["ordinal"] for node in item["nodes"]] == list(
                range(len(item["nodes"]))
            )
            assert all(set(node) == {"ordinal", "node_type", "content"} for node in item["nodes"])
            edge_keys = {
                (edge["source_ordinal"], edge["target_ordinal"], edge["edge_type"])
                for edge in item["edges"]
            }
            assert edge_keys
            for source_ordinal, target_ordinal, edge_type in edge_keys:
                reverse = (
                    target_ordinal,
                    source_ordinal,
                    "TEXT_TO_ROW" if edge_type == "ROW_TO_TEXT" else "ROW_TO_TEXT",
                )
                assert reverse in edge_keys
        for item in gold["items"]:
            expected = {
                "work_id",
                "gold_row_ordinals",
                "gold_text_ordinals",
                "exact_gold_pairs",
            }
            if block == "A_form":
                expected.add("oof_fold")
                assert 0 <= item["oof_fold"] < 5
            elif block in {"A_hold", "M_search"}:
                expected.add("evaluation_family")
                assert item["evaluation_family"] in selection.FAMILIES
            assert set(item) == expected
            assert 1 <= len(item["gold_row_ordinals"]) <= 4
            assert 1 <= len(item["gold_text_ordinals"]) <= 4
            assert item["exact_gold_pairs"]

    a_form_gold = _load(output / selection.GOLD_PACK_FILENAMES["A_form"])["items"]
    fold_resources: list[set[str]] = [set() for _ in range(5)]
    for item in a_form_gold:
        source_row = ledger_by_work[item["work_id"]]
        fold_resources[item["oof_fold"]].add(source_row["source_table_id"])
        fold_resources[item["oof_fold"]].update(source_row["source_text_doc_ids"])
    assert all(fold_resources)
    assert all(
        fold_resources[left].isdisjoint(fold_resources[right])
        for left in range(5)
        for right in range(left + 1, 5)
    )

    special_gold = _load(
        output / selection.GOLD_PACK_FILENAMES[special_mapping["block"]]
    )["items"][special_mapping["block_ordinal"]]
    assert 0 <= special_gold["oof_fold"] < 5
    assert special_gold["gold_row_ordinals"] == [0, 1]
    assert special_gold["exact_gold_pairs"] == [
        {"row_ordinal": 0, "text_ordinal": 2}
    ]

    public = (output / selection.PUBLIC_RECEIPT_FILENAME).read_text("ascii")
    assert "QID_PRIVATE_" not in public
    assert "mmqa-work-v1-" not in public


def test_same_secret_is_repeat_exact_and_root_replay_is_forbidden(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    outputs = [tmp_path / "one", tmp_path / "two"]
    receipts = []
    for output in outputs:
        receipts.append(
            selection.acquire_once(
                source_paths=source.paths,
                qualification_path=source.receipt_path,
                expected_qualification_self_sha256=source.receipt_sha256,
                output_root=output,
                contract=source.contract,
                random_bytes=lambda size: SECRET,
            )
        )
    assert receipts[0]["acquisition_sha256"] == receipts[1]["acquisition_sha256"]
    for block in selection.BLOCK_ORDER:
        assert (
            outputs[0] / selection.ACTION_PACK_FILENAMES[block]
        ).read_bytes() == (
            outputs[1] / selection.ACTION_PACK_FILENAMES[block]
        ).read_bytes()
    with pytest.raises(selection.MmqaP1PrivateSelectionError, match="replay"):
        selection.acquire_once(
            source_paths=source.paths,
            qualification_path=source.receipt_path,
            expected_qualification_self_sha256=source.receipt_sha256,
            output_root=outputs[0],
            contract=source.contract,
            random_bytes=lambda size: SECRET,
        )


def test_forged_gold_open_capability_is_rejected_and_f_is_never_opened(
    tmp_path: Path,
) -> None:
    _source, output, receipt = _acquire(tmp_path)
    nonexistent_archive = (tmp_path / "does-not-exist.archive.json").absolute()
    authorization_path = tmp_path / "A_form.authorization.json"
    with pytest.raises(
        selection.MmqaP1PrivateSelectionError, match="archive.*unavailable"
    ):
        selection.write_block_gold_open_authorization(
            authorization_path,
            output_root=output,
            block="A_form",
            action_archive_sha256s=("a" * 64,),
            action_archive_paths=(nonexistent_archive,),
        )

    action = selection._pack_binding(  # noqa: SLF001
        receipt, block="A_form", role="action"
    )
    gold = selection._pack_binding(  # noqa: SLF001
        receipt, block="A_form", role="gold"
    )
    forged = selection.self_hashed(
        {
            "schema": (
                f"{selection.VERSION}_block_gold_open_authorization_v1"
            ),
            "version": selection.VERSION,
            "study_id": selection.STUDY_ID,
            "status": "gold_open_authorized_after_immutable_action_archives",
            "block": "A_form",
            "acquisition_sha256": receipt["acquisition_sha256"],
            "action_pack_sha256": action["semantic_sha256"],
            "gold_pack_sha256": gold["semantic_sha256"],
            "action_archive_sha256s": ["a" * 64],
            "action_archive_paths": [str(nonexistent_archive)],
            "action_archive_semantic_sha256s": ["b" * 64],
            "action_archives_complete_and_immutable": True,
            "A_hold_promotion_sha256": None,
            "A_hold_promotion_file_sha256": None,
            "A_hold_promotion_receipt_path": None,
            "A_hold_promotion_action_archive_path": None,
            "same_block_replay_authorized": False,
        },
        "authorization_sha256",
    )
    selection._atomic_write_json(  # noqa: SLF001
        authorization_path, forged, mode=0o600
    )
    with pytest.raises(
        selection.MmqaP1PrivateSelectionError, match="archive.*unavailable"
    ):
        selection.open_block_gold(
            output_root=output,
            block="A_form",
            authorization_path=authorization_path,
            expected_authorization_sha256=forged["authorization_sha256"],
        )
    assert not (
        output / selection.GOLD_OPEN_MARKER_FILENAMES["A_form"]
    ).exists()

    nonexistent = tmp_path / "does-not-exist"
    with pytest.raises(selection.MmqaP1PrivateSelectionError, match="permanently sealed"):
        selection.open_block_gold(
            output_root=nonexistent,
            block="F_search",
            authorization_path=nonexistent,
            expected_authorization_sha256="0" * 64,
        )
    assert not (output / selection.GOLD_OPEN_MARKER_FILENAMES["F_search"]).exists()
    with pytest.raises(selection.MmqaP1PrivateSelectionError, match="permanently sealed"):
        selection.write_block_gold_open_authorization(
            tmp_path / "F.auth",
            output_root=output,
            block="F_search",
            action_archive_sha256s=("a" * 64,),
            action_archive_paths=(nonexistent_archive,),
        )
    with pytest.raises(selection.MmqaP1PrivateSelectionError, match="promotion"):
        selection.write_block_gold_open_authorization(
            tmp_path / "M.auth",
            output_root=output,
            block="M_search",
            action_archive_sha256s=("a" * 64,),
            action_archive_paths=(nonexistent_archive,),
        )


def test_all_four_identities_are_checked_before_semantic_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path)
    target = source.paths["MMQA_texts.jsonl.gz"]
    raw = bytearray(target.read_bytes())
    raw[len(raw) // 2] ^= 1
    target.write_bytes(bytes(raw))
    target.chmod(0o600)
    parsed = False

    def forbidden_parse(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal parsed
        parsed = True
        raise AssertionError("semantic parse must not begin")

    monkeypatch.setattr(selection, "_load_question_split", forbidden_parse)
    output = tmp_path / "failed"
    with pytest.raises(selection.MmqaP1PrivateSelectionError, match="SHA256"):
        selection.acquire_once(
            source_paths=source.paths,
            qualification_path=source.receipt_path,
            expected_qualification_self_sha256=source.receipt_sha256,
            output_root=output,
            contract=source.contract,
            random_bytes=lambda size: SECRET,
        )
    assert parsed is False
    assert (output / selection.COMMITMENT_FILENAME).exists()
    assert (output / selection.FAILURE_FILENAME).exists()


def test_qualification_receipt_and_study_bindings_fail_closed(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    value = _load(source.receipt_path)
    value["binding_self_sha256"]["study_design"] = "f" * 64
    body = dict(value)
    body.pop("self_sha256")
    forged = selection.self_hashed(body)
    forged_path = tmp_path / "forged.json"
    forged_path.write_bytes(selection._canonical_bytes(forged, newline=True))  # noqa: SLF001
    with pytest.raises(selection.MmqaP1PrivateSelectionError, match="receipt drifted"):
        selection.load_qualification_binding(
            forged_path,
            expected_self_sha256=forged["self_sha256"],
            contract=source.contract,
        )

    design = _load(selection.PROJECT_ROOT / selection.DESIGN_RELATIVE)
    design["blocks"]["A_hold"]["per_family"] = 16
    body = dict(design)
    body.pop("self_sha256")
    drifted = selection.self_hashed(body)
    design_path = tmp_path / "design.json"
    design_path.write_bytes(selection._canonical_bytes(drifted, newline=True))  # noqa: SLF001
    with pytest.raises(selection.MmqaP1PrivateSelectionError, match="bound manifest"):
        selection.verify_study_bindings(
            selection.PROJECT_ROOT / selection.CUSTODY_RELATIVE,
            design_path,
        )


def test_selection_hmac_exact_namespace_and_action_order_hide_family_groups() -> None:
    qid = "PRIVATE_QID"
    observed = selection.selection_hmac_digest(
        SECRET,
        split="TRAIN",
        family=selection.FAMILIES[0],
        qid=qid,
    )
    manual = hmac_sha256(
        SECRET,
        selection.ORDER_HMAC_DOMAIN
        + _frame(b"study", selection.STUDY_ID)
        + _frame(b"split", "TRAIN")
        + _frame(b"family", selection.FAMILIES[0])
        + _frame(b"qid", qid),
    )
    assert observed == manual
    assert selection.opaque_work_id(
        SECRET,
        block="A_form",
        split="TRAIN",
        family=selection.FAMILIES[0],
        qid=qid,
    ).startswith("mmqa-work-v1-")


def _frame(name: bytes, value: str) -> bytes:
    raw = value.encode("utf-8")
    return name + b"\0" + len(raw).to_bytes(8, "big") + raw


def hmac_sha256(secret: bytes, message: bytes) -> bytes:
    import hmac

    return hmac.new(secret, message, hashlib.sha256).digest()


def test_module_has_no_model_network_scoring_or_alternate_source_surface() -> None:
    path = (
        selection.PROJECT_ROOT
        / "assumption_agent/benchmarks/mmqa_p1_private_selection_v1.py"
    )
    tree = ast.parse(path.read_text("utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not imports.intersection(
        {
            "requests",
            "httpx",
            "openai",
            "torch",
            "transformers",
            "sentence_transformers",
            "numpy",
            "sklearn",
        }
    )
    parser_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]
    option_literals = {
        arg.value
        for call in parser_calls
        for arg in call.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    assert option_literals == {
        "--formal-acquire",
        "--project",
        "--qualification-self-sha256",
    }
