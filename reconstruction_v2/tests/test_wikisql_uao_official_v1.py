from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace

import pytest

from replication_runtime.wikisql_uao_official_v1 import contract
from replication_runtime.wikisql_uao_official_v1 import worker
from assumption_agent.benchmarks import (
    wikisql_uao_action_runtime_v1 as action_runtime,
)
from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality


def test_official_bounds_equal_the_common_action_contract() -> None:
    assert contract.MAX_COLUMN_COUNT == action_runtime.MAX_COLUMNS
    assert (
        contract.MAX_QUESTION_CHARACTERS
        == action_runtime.MAX_QUESTION_CHARACTERS
    )
    assert (
        contract.MAX_HEADER_CHARACTERS
        == action_runtime.MAX_HEADER_CHARACTERS
    )
    assert (
        contract.MAX_CELL_CHARACTERS
        == action_runtime.MAX_CELL_CHARACTERS
    )
    assert (
        contract.MAX_SERIALIZED_ROW_CHARACTERS
        == action_runtime.MAX_SERIALIZED_ROW_CHARACTERS
        == reality.MAX_SERIALIZED_ROW_CHARACTERS
    )


def _item(ordinal: int, *, row_count: int = 11) -> dict[str, object]:
    return {
        "headers": ["City", "Population", "Note"],
        "item_id": hashlib.sha256(
            f"opaque-item-{ordinal}".encode("ascii")
        ).hexdigest(),
        "question": f"Which row answers opaque question {ordinal}?",
        "rows": [
            [
                f"city-{row_ordinal}",
                row_ordinal * 100,
                "label is legitimate cell text"
                if row_ordinal == 0
                else f"note-{row_ordinal}",
            ]
            for row_ordinal in range(row_count)
        ],
        "types": ["text", "real", "text"],
    }


def _payload(item_count: int = 1) -> dict[str, object]:
    return contract.input_payload(
        items=[_item(ordinal) for ordinal in range(item_count)]
    )


def _forbidden_field_names(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.casefold()
            if lowered == "contains_labels" and child is False:
                continue
            if any(
                token in lowered
                for token in (
                    "answer",
                    "family",
                    "gold",
                    "label",
                    "qrel",
                    "score",
                    "sql",
                    "utility",
                )
            ):
                result.add(key)
            result.update(_forbidden_field_names(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_forbidden_field_names(child))
    return result


def test_input_contract_is_canonical_committed_and_recursively_label_free() -> None:
    payload = _payload(2)
    raw = contract.canonical_json_bytes(payload)
    parsed, items = contract.parse_input(raw)
    assert parsed == payload
    assert len(items) == 2
    assert payload["schema"] == action_runtime.VIEW_PACK_SCHEMA
    assert payload["block"] == "A_hold"
    assert payload["contains_labels"] is False
    assert len(payload["self_sha256"]) == 64
    assert _forbidden_field_names(payload) == set()
    assert len(items[0].item_sha256) == 64
    assert len(items[0].row_corpus_sha256) == 64
    assert "label is legitimate cell text" in items[0].rows[0]

    for forbidden in ("sql", "family", "gold", "label", "utility"):
        invalid = _item(0)
        invalid[forbidden] = "hidden"
        with pytest.raises(
            contract.WikiSQLUAOOfficialHippoRAGError
        ):
            contract.input_payload(items=[invalid])

    nested = _item(0)
    nested["rows"][0][0] = {"gold": "hidden"}  # type: ignore[index]
    with pytest.raises(contract.WikiSQLUAOOfficialHippoRAGError):
        contract.input_payload(items=[nested])


def test_contract_enforces_item_row_and_table_bounds() -> None:
    assert len(contract.validate_input(_payload(72))) == 72
    with pytest.raises(
        contract.WikiSQLUAOOfficialHippoRAGError,
        match="item count",
    ):
        _payload(73)
    for invalid_count in (10, 81):
        with pytest.raises(
            contract.WikiSQLUAOOfficialHippoRAGError,
            match="row count",
        ):
            contract.input_payload(
                items=[_item(0, row_count=invalid_count)]
            )
    invalid_type = _item(0)
    invalid_type["types"] = ["text", "number", "text"]
    with pytest.raises(
        contract.WikiSQLUAOOfficialHippoRAGError,
        match="column type",
    ):
        contract.input_payload(items=[invalid_type])
    invalid_width = _item(0)
    invalid_width["rows"][0] = ["short"]  # type: ignore[index]
    with pytest.raises(
        contract.WikiSQLUAOOfficialHippoRAGError,
        match="width",
    ):
        contract.input_payload(items=[invalid_width])


def test_row_serialization_is_byte_exact_cross_arm_reality_text() -> None:
    payload = contract.input_payload(items=[_item(0)])
    item = contract.validate_input(payload)[0]
    documents = contract.serialize_rows(item)
    table = reality.WikiSQLTable(
        table_id=item.item_id,
        header=item.headers,
        types=item.types,
        rows=item.rows,
    )
    assert len(documents) == 11
    assert documents == reality.serialize_table_rows(table)
    assert all(
        document == reality.serialize_table_row(table, ordinal)
        for ordinal, document in enumerate(documents)
    )
    assert all(item.item_id not in document for document in documents)
    assert all("row_ordinal" not in document for document in documents)


def test_duplicate_cross_arm_row_text_fails_closed_without_id_injection() -> None:
    raw = _item(0)
    raw["rows"][1] = list(raw["rows"][0])  # type: ignore[index]
    with pytest.raises(
        contract.WikiSQLUAOOfficialHippoRAGError,
        match="shared row-document contract drifted",
    ):
        contract.input_payload(items=[raw])


def test_stable_top_five_uses_score_then_row_ordinal() -> None:
    item = contract.validate_input(_payload())[0]
    documents = contract.serialize_rows(item)
    mapping = {
        document: ordinal
        for ordinal, document in enumerate(documents)
    }
    assert contract.stable_top_k(
        retrieved_documents=list(reversed(documents)),
        retrieved_scores=[1.0] * len(documents),
        document_to_ordinal=mapping,
    ) == (0, 1, 2, 3, 4)

    with pytest.raises(
        contract.WikiSQLUAOOfficialHippoRAGError,
        match="omitted",
    ):
        contract.stable_top_k(
            retrieved_documents=documents[:-1],
            retrieved_scores=[1.0] * (len(documents) - 1),
            document_to_ordinal=mapping,
        )
    with pytest.raises(contract.WikiSQLUAOOfficialHippoRAGError):
        contract.stable_top_k(
            retrieved_documents=documents,
            retrieved_scores=[float("nan")] * len(documents),
            document_to_ordinal=mapping,
        )


class _Graph:
    def __init__(self, ordinal: int) -> None:
        self.ordinal = ordinal

    def vcount(self) -> int:
        return 100 + self.ordinal

    def ecount(self) -> int:
        return 200 + self.ordinal


class _FakeCore:
    def __init__(
        self,
        *,
        ordinal: int,
        index_root: Path,
        fail_retrieve: bool = False,
    ) -> None:
        self.ordinal = ordinal
        self.index_root = index_root
        self.fail_retrieve = fail_retrieve
        self.graph = _Graph(ordinal)
        self.documents: list[str] | None = None
        self.index_call_count = 0
        self.retrieve_call_count = 0

    def index(self, documents: list[str]) -> None:
        self.index_call_count += 1
        self.documents = list(documents)
        (self.index_root / "index.bin").write_bytes(
            hashlib.sha256(
                "\n".join(documents).encode("utf-8")
            ).digest()
        )

    def retrieve(
        self, queries: list[str], num_to_retrieve: int
    ) -> list[SimpleNamespace]:
        self.retrieve_call_count += 1
        assert len(queries) == 1
        assert self.documents is not None
        assert num_to_retrieve == len(self.documents)
        if self.fail_retrieve:
            raise RuntimeError("injected one-shot failure")
        return [
            SimpleNamespace(
                docs=list(reversed(self.documents)),
                doc_scores=[1.0] * len(self.documents),
            )
        ]


def test_worker_uses_one_fresh_index_per_item_in_one_sequential_lane(
    tmp_path: Path,
) -> None:
    payload = _payload(3)
    created: list[_FakeCore] = []
    observed_ordinals: list[int] = []

    def factory(**kwargs: object) -> _FakeCore:
        item_ordinal = kwargs["item_ordinal"]
        index_root = kwargs["index_root"]
        assert isinstance(item_ordinal, int)
        assert isinstance(index_root, Path)
        assert index_root.is_dir()
        assert list(index_root.iterdir()) == []
        observed_ordinals.append(item_ordinal)
        core = _FakeCore(
            ordinal=item_ordinal,
            index_root=index_root,
        )
        created.append(core)
        return core

    artifacts = worker.run_with_core_factory(
        private_input=payload,
        index_parent=tmp_path / "indexes",
        core_factory=factory,
    )
    assert observed_ordinals == [0, 1, 2]
    assert all(core.index_call_count == 1 for core in created)
    assert all(core.retrieve_call_count == 1 for core in created)
    assert len({core.index_root for core in created}) == 3
    action_pack = artifacts.action_pack
    action_runtime.decode_action_pack(
        action_pack,
        expected_block="A_hold",
        expected_arm="HippoRAG",
        expected_action_view_pack_sha256=payload["self_sha256"],
    )
    assert [
        row["top5_row_ids"] for row in action_pack["items"]
    ] == [[0, 1, 2, 3, 4]] * 3
    assert set(action_pack) == {
        "action_view_pack_sha256",
        "arm",
        "block",
        "item_count",
        "items",
        "schema",
        "self_sha256",
        "study_id",
    }
    assert action_pack == action_runtime.build_action_pack(
        block="A_hold",
        arm="HippoRAG",
        action_view_pack_sha256=payload["self_sha256"],
        items=[
            {
                "opaque_item_id": row["opaque_item_id"],
                "top5_row_ids": [0, 1, 2, 3, 4],
            }
            for row in payload["items"]
        ],
    )
    assert action_pack["action_view_pack_sha256"] == (
        payload["self_sha256"]
    )
    assert artifacts.safe_receipt["runtime"] == {
        "core_config_sha256": contract.CORE_CONFIG_SHA256,
        "evaluator_call_count": 0,
        "fresh_index_per_item": True,
        "index_call_count": 3,
        "network_call_count": 0,
        "official_hipporag_commit": (
            contract.OFFICIAL_HIPPORAG_COMMIT
        ),
        "replay_count": 0,
        "retrieve_call_count": 3,
        "retry_count": 0,
        "sequential_item_execution": True,
        "single_gpu_lane_count": 1,
        "top_k": 5,
    }
    assert artifacts.safe_receipt["item_count"] == 3
    assert len(artifacts.safe_receipt["index_receipts_sha256"]) == 64
    assert artifacts.safe_receipt["action_pack_sha256"] == (
        action_pack["self_sha256"]
    )
    safe_json = json.dumps(artifacts.safe_receipt, sort_keys=True)
    assert "Which row" not in safe_json
    assert "Population" not in safe_json
    assert "city-" not in safe_json
    assert "opaque_item_id" not in safe_json
    assert "top5_row_ids" not in safe_json


def test_common_action_and_safe_aggregate_outputs_are_hash_bound(
    tmp_path: Path,
) -> None:
    payload = _payload()

    def factory(**kwargs: object) -> _FakeCore:
        return _FakeCore(
            ordinal=kwargs["item_ordinal"],  # type: ignore[arg-type]
            index_root=kwargs["index_root"],  # type: ignore[arg-type]
        )

    artifacts = worker.run_with_core_factory(
        private_input=payload,
        index_parent=tmp_path / "indexes",
        core_factory=factory,
    )
    action = dict(artifacts.action_pack)
    assert action["self_sha256"] == action_runtime.canonical_sha256(
        {
            key: value
            for key, value in action.items()
            if key != "self_sha256"
        }
    )
    tampered = json.loads(json.dumps(action))
    tampered["items"][0]["top5_row_ids"] = [4, 3, 2, 1, 0]
    with pytest.raises(action_runtime.WikiSQLUAOActionRuntimeError):
        action_runtime.decode_action_pack(
            tampered,
            expected_block="A_hold",
            expected_arm="HippoRAG",
            expected_action_view_pack_sha256=payload["self_sha256"],
        )
    receipt = artifacts.safe_receipt
    assert receipt["self_sha256"] == contract.semantic_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "self_sha256"
        }
    )


def test_run_once_writes_only_common_action_and_safe_receipt_exclusively(
) -> None:
    payload = _payload()

    def factory(**kwargs: object) -> _FakeCore:
        return _FakeCore(
            ordinal=kwargs["item_ordinal"],  # type: ignore[arg-type]
            index_root=kwargs["index_root"],  # type: ignore[arg-type]
        )

    # The formal node uses a native Linux filesystem.  Force that contract in
    # tests too because the configured pytest temp root may be a DrvFS mount
    # whose synthetic permission bits cannot represent mode 0600.
    with tempfile.TemporaryDirectory(
        prefix="wikisql-official-", dir="/tmp"
    ) as temporary:
        root = Path(temporary)
        action_path = root / "actions.json"
        receipt_path = root / "safe-receipt.json"
        artifacts = worker.run_once(
            private_input=payload,
            action_output_path=action_path,
            safe_receipt_output_path=receipt_path,
            index_parent=root / "indexes",
            core_factory=factory,
        )
        assert json.loads(action_path.read_text("ascii")) == (
            artifacts.action_pack
        )
        assert json.loads(receipt_path.read_text("ascii")) == (
            artifacts.safe_receipt
        )
        assert stat.S_IMODE(action_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
        with pytest.raises(
            contract.WikiSQLUAOOfficialHippoRAGError,
            match="not fresh and distinct",
        ):
            worker.run_once(
                private_input=payload,
                action_output_path=action_path,
                safe_receipt_output_path=receipt_path,
                index_parent=root / "unused-indexes",
                core_factory=factory,
            )
        assert not (root / "unused-indexes").exists()


def test_failure_aborts_without_retry_replay_or_output(
    tmp_path: Path,
) -> None:
    payload = _payload(3)
    constructed: list[_FakeCore] = []
    factory_ordinals: list[int] = []

    def factory(**kwargs: object) -> _FakeCore:
        item_ordinal = kwargs["item_ordinal"]
        assert isinstance(item_ordinal, int)
        factory_ordinals.append(item_ordinal)
        core = _FakeCore(
            ordinal=item_ordinal,
            index_root=kwargs["index_root"],  # type: ignore[arg-type]
            fail_retrieve=item_ordinal == 1,
        )
        constructed.append(core)
        return core

    action_output_path = tmp_path / "actions.json"
    safe_receipt_output_path = tmp_path / "receipt.json"
    with pytest.raises(
        contract.WikiSQLUAOOfficialHippoRAGError,
        match="without retry",
    ):
        worker.run_once(
            private_input=payload,
            action_output_path=action_output_path,
            safe_receipt_output_path=safe_receipt_output_path,
            index_parent=tmp_path / "indexes",
            core_factory=factory,
        )
    assert factory_ordinals == [0, 1]
    assert [core.index_call_count for core in constructed] == [1, 1]
    assert [core.retrieve_call_count for core in constructed] == [1, 1]
    assert not action_output_path.exists()
    assert not safe_receipt_output_path.exists()
    with pytest.raises(
        contract.WikiSQLUAOOfficialHippoRAGError,
        match="not fresh",
    ):
        worker.run_once(
            private_input=payload,
            action_output_path=action_output_path,
            safe_receipt_output_path=safe_receipt_output_path,
            index_parent=tmp_path / "indexes",
            core_factory=factory,
        )
    assert factory_ordinals == [0, 1]


def test_existing_index_parent_and_symlinked_index_artifact_fail_closed(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(
        contract.WikiSQLUAOOfficialHippoRAGError,
        match="not fresh",
    ):
        worker.run_with_core_factory(
            private_input=_payload(),
            index_parent=existing,
            core_factory=lambda **_kwargs: object(),
        )

    index_root = tmp_path / "index"
    index_root.mkdir()
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    (index_root / "link").symlink_to(target)
    with pytest.raises(
        contract.WikiSQLUAOOfficialHippoRAGError,
        match="symbolic link",
    ):
        worker.snapshot_index_tree(index_root)
