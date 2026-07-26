from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import replace
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading
import warnings
import zipfile

import pytest

from assumption_agent.benchmarks import hitab_p1_source_acquisition_v1 as h


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMPTY_EXPOSURE = {
    "id": frozenset(),
    "question": frozenset(),
    "table_id": frozenset(),
}


@pytest.fixture
def tmp_path() -> Path:
    """Use the Linux filesystem because the contract requires POSIX mode bits."""

    path = Path(tempfile.mkdtemp(prefix="hitab-p1-source-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _tree_node(
    name: str,
    line_idx: int | None,
    children: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "children_dict": [] if children is None else children,
        "extra_official_field": "ignored",
        "line_idx": line_idx,
        "name": name,
        "value": name,
    }


def _table_payload(
    *,
    seed: int = 0,
    rows: int = 2,
    columns: int = 5,
) -> dict[str, object]:
    assert rows >= 2 and columns >= 5
    top_children = [
        _tree_node(
            "Metric",
            0,
            [
                _tree_node(f"Metric {column}", column)
                for column in range(1, columns)
            ],
        )
    ]
    left_children = [
        _tree_node(
            "Region",
            0,
            [
                _tree_node(f"Region {row}", row)
                for row in range(1, rows)
            ],
        )
    ]
    values = []
    ordinal = 0
    for _row in range(rows):
        row = []
        for _column in range(columns):
            row.append(
                {
                    "top_coord": [0, 0],
                    "ignored": True,
                    "value": seed + ordinal,
                    "value_name": "synthetic",
                }
            )
            ordinal += 1
        values.append(row)
    return {
        "data": values,
        "ignored_table_field": {"safe": True},
        "left_root": _tree_node("<LEFT>", None, left_children),
        "title": f"Synthetic table {seed}",
        "top_root": _tree_node("<TOP>", None, top_children),
    }


def _raw_table_payload(
    hmt: Mapping[str, object],
    *,
    top_header_rows_num: int = 2,
    left_header_columns_num: int = 2,
) -> dict[str, object]:
    data = hmt["data"]
    assert isinstance(data, list) and data and isinstance(data[0], list)
    rows = top_header_rows_num + len(data)
    columns = left_header_columns_num + len(data[0])
    texts: list[list[object]] = [
        ["" for _column in range(columns)] for _row in range(rows)
    ]
    for row_index, row in enumerate(data):
        assert isinstance(row, list)
        for column_index, cell in enumerate(row):
            assert isinstance(cell, Mapping)
            texts[row_index + top_header_rows_num][
                column_index + left_header_columns_num
            ] = cell["value"]
    return {
        "ignored_raw_field": ["allowed"],
        "left_header_columns_num": left_header_columns_num,
        "texts": texts,
        "top_header_rows_num": top_header_rows_num,
    }


def _sample(
    *,
    item_id: str,
    table_id: str,
    aggregation: str,
    answer: object,
    coordinate: str = "(2, 2)",
    source: str = "statcan",
    extra: bool = True,
) -> dict[str, object]:
    value: dict[str, object] = {
        "aggregation": [aggregation],
        "id": item_id,
        "linked_cells": {
            "entity_link": {"left": {}, "top": {}},
            "quantity_link": {"[ANSWER]": {coordinate: answer}},
        },
        "question": f"Question for {item_id}?",
        "reference_cells_map": {"ignored": "(999, 999)"},
        "table_id": table_id,
        "table_source": source,
    }
    if extra:
        value["answer_formulas"] = ["not consumed"]
        value["future_official_field"] = {"allowed": True}
    return value


def _jsonl(rows: list[Mapping[str, object]]) -> bytes:
    return b"".join(h.canonical_bytes(row, newline=True) for row in rows)


def _write_private(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    os.chmod(path, 0o600)


def _write_tables_zip(
    path: Path,
    tables: Mapping[str, Mapping[str, object]],
    *,
    extras: Mapping[str, bytes] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in (extras or {}).items():
            archive.writestr(name, raw)
        for table_id, value in tables.items():
            archive.writestr(
                f"official/tables/hmt/{table_id}.json",
                h.canonical_bytes(value),
            )
            archive.writestr(
                f"official/tables/raw/{table_id}.json",
                h.canonical_bytes(_raw_table_payload(value)),
            )
    os.chmod(path, 0o600)


def _candidate(
    *,
    split: str,
    item_id: str,
    table_id: str,
    family: str,
) -> h.SampleCandidate:
    token = {
        "AGGREGATE": "sum",
        "COMPARATIVE": "diff",
        "SUPERLATIVE": "max",
    }[family]
    return h.parse_sample_row(
        _sample(
            item_id=item_id,
            table_id=table_id,
            aggregation=token,
            answer=0,
        ),
        split=split,
        public_exposure_hashes=EMPTY_EXPOSURE,
    )


def _eligible(
    *,
    split: str,
    item_id: str,
    table_id: str,
    family: str,
    seed: int,
) -> h.EligibleItem:
    candidate = _candidate(
        split=split,
        item_id=item_id,
        table_id=table_id,
        family=family,
    )
    payload = _table_payload(seed=seed)
    table = h.parse_hmt_table(payload, _raw_table_payload(payload))
    qrel = h.ProofDNF(
        alternatives=(((0,),),),
        corpus_commitment=table.corpus_commitment,
    )
    return h.EligibleItem(candidate=candidate, table=table, qrel=qrel)


def _assert_self_hash(value: Mapping[str, object]) -> None:
    body = dict(value)
    claimed = body.pop("self_sha256")
    assert claimed == h.stable_hash(body)


def _prepare_full_source_set(
    tmp_path: Path,
    *,
    test_rows: list[Mapping[str, object]],
    test_tables: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, Path], h.VerifiedSourceSet]:
    tables: dict[str, Mapping[str, object]] = dict(test_tables)
    rows_by_split: dict[str, list[Mapping[str, object]]] = {
        "TRAIN": [],
        "DEV": [],
    }
    token_by_family = {
        "AGGREGATE": "sum",
        "COMPARATIVE": "diff",
        "SUPERLATIVE": "max",
    }
    seed = 1_000
    for split in ("TRAIN", "DEV"):
        for family, token in token_by_family.items():
            table_id = f"{split.casefold()}-{family.casefold()}"
            tables[table_id] = _table_payload(seed=seed)
            rows_by_split[split].append(
                _sample(
                    item_id=f"{table_id}-item",
                    table_id=table_id,
                    aggregation=token,
                    answer=seed,
                )
            )
            seed += 100
    source_root = tmp_path / "source"
    source_paths = {
        "TRAIN": source_root / "train.jsonl",
        "DEV": source_root / "dev.jsonl",
        "TEST": source_root / "test.jsonl",
        "TABLES": source_root / "tables.zip",
    }
    _write_private(source_paths["TRAIN"], _jsonl(rows_by_split["TRAIN"]))
    _write_private(source_paths["DEV"], _jsonl(rows_by_split["DEV"]))
    _write_private(source_paths["TEST"], _jsonl(test_rows))
    _write_tables_zip(source_paths["TABLES"], tables)
    identities: dict[str, h.VerifiedFileIdentity] = {}
    for key, path in source_paths.items():
        raw = path.read_bytes()
        identities[key] = h.VerifiedFileIdentity(
            key=key,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            git_blob_sha1=h.git_blob_sha1(raw),
            raw_newline_count=raw.count(b"\n") if key != "TABLES" else None,
        )
    identity_payload = {
        key: identities[key].safe_payload()
        for key in ("TRAIN", "DEV", "TEST", "TABLES")
    }
    verified = h.VerifiedSourceSet(
        identities=identities,
        safe_receipt={
            "source_identity_commitment": h.stable_hash(identity_payload)
        },
    )
    return source_paths, verified


def _valid_four_arm_archive(
    view: h.BridgeBlockView,
) -> Mapping[str, object]:
    records = []
    for index, row in enumerate(view.items):
        top5 = [0, 1, 2, 3, 4]
        records.append(
            {
                "arms": {
                    "E0": {
                        "corpus_commitment": row.corpus_commitment,
                        "top5_ordinals": top5,
                    },
                    "E1": {
                        "corpus_commitment": row.corpus_commitment,
                        "top5_ordinals": top5,
                    },
                    "HippoRAG": {
                        "complete_rank_sha256": hashlib.sha256(
                            f"rank-{index}".encode()
                        ).hexdigest(),
                        "corpus_commitment": row.corpus_commitment,
                        "input_sha256": hashlib.sha256(
                            f"input-{index}".encode()
                        ).hexdigest(),
                        "output_sha256": hashlib.sha256(
                            f"output-{index}".encode()
                        ).hexdigest(),
                        "physical_gpu": index % 2,
                        "top5_ordinals": top5,
                    },
                    "RAW": {
                        "corpus_commitment": row.corpus_commitment,
                        "top5_ordinals": top5,
                    },
                },
                "tensor_sha256": hashlib.sha256(
                    f"tensor-{index}".encode()
                ).hexdigest(),
                "work_id": row.work_id,
            }
        )
    return h.self_hashed(
        {
            "block": view.block,
            "block_view_sha256": view.view_sha256,
            "e1_model_sha256": "e" * 64,
            "four_arm_corpus_commitment_exact": True,
            "gpu0_unused_cuda_cache_release_receipt": h.self_hashed(
                {
                    "model_offload_or_reload": False,
                    "physical_gpu": 0,
                    "schema": (
                        "hitab_p1_gpu0_unused_cuda_cache_release_v1"
                    ),
                    "study_id": h.STUDY_ID,
                    "torch_cuda_empty_cache_called": True,
                }
            ),
            "hipporag_queue_joined_before_archive": True,
            "item_count": len(records),
            "records": records,
            "schema": (
                "hitab_p1_formal_controller_v1_"
                f"{view.block}_four_arm_action_archive_v1"
            ),
            "study_id": h.STUDY_ID,
        }
    )


class _Response(io.BytesIO):
    def __init__(self, body: bytes, url: str) -> None:
        super().__init__(body)
        self.status = 200
        self.headers = {
            "Content-Encoding": "identity",
            "Content-Length": str(len(body)),
        }
        self._url = url

    def geturl(self) -> str:
        return self._url


def _download_contracts(
    payloads: Mapping[str, bytes],
) -> dict[str, h.SourceFileContract]:
    result = {}
    for key in ("TRAIN", "DEV", "TEST", "TABLES"):
        relative = f"data/{key.casefold()}.synthetic"
        result[key] = h.SourceFileContract(
            key=key,
            relative_path=relative,
            size_bytes=len(payloads[key]),
            git_blob_sha1=h.git_blob_sha1(payloads[key]),
            is_jsonl=key != "TABLES",
            raw_url=(
                "https://raw.githubusercontent.com/microsoft/HiTab/"
                f"{h.SOURCE_COMMIT}/{relative}"
            ),
        )
    return result


def test_frozen_bindings_and_family_registry_are_exact() -> None:
    h.verify_frozen_bindings(
        PROJECT_ROOT / h.CUSTODY_RELATIVE,
        PROJECT_ROOT / h.DESIGN_RELATIVE,
    )
    assert h.FAMILIES == ("AGGREGATE", "COMPARATIVE", "SUPERLATIVE")
    assert h.family_from_aggregation(["average", "sum"]) == "AGGREGATE"
    assert h.family_from_aggregation(["greater_than", "div"]) == "COMPARATIVE"
    assert h.family_from_aggregation(["topk-argmin", "range"]) == "SUPERLATIVE"
    for aggregation in (
        [],
        ["none"],
        ["sum", "max"],
        ["unknown"],
        "sum",
    ):
        with pytest.raises(h.HitabP1RowIneligible):
            h.family_from_aggregation(aggregation)


def test_parallel_four_get_acquisition_is_one_attempt_and_no_payload_is_decoded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = {
        "TRAIN": b"opaque train bytes\n",
        "DEV": b"opaque dev bytes\n",
        "TEST": b"not JSON and intentionally never decoded\n",
        "TABLES": b"opaque synthetic ZIP transport bytes",
    }
    contracts = _download_contracts(payloads)
    barrier = threading.Barrier(4)
    calls: list[str] = []
    lock = threading.Lock()

    def opener(url: str) -> _Response:
        with lock:
            calls.append(url)
        barrier.wait(timeout=5)
        key = next(
            key for key, contract in contracts.items() if contract.raw_url == url
        )
        return _Response(payloads[key], url)

    monkeypatch.setattr(h, "verify_frozen_bindings", lambda *_args, **_kwargs: None)
    result = h.download_source_set_once(
        source_root=tmp_path / "source",
        control_root=tmp_path / "control",
        contracts=contracts,
        opener=opener,
    )
    assert len(calls) == 4
    assert set(calls) == {
        contract.raw_url for contract in contracts.values()
    }
    for key, path in result.source_paths.items():
        assert path.read_bytes() == payloads[key]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert not path.with_name(f".{path.name}.one_shot.part").exists()
    receipt = result.verified_sources.safe_receipt
    assert receipt["network_attempt_count"] == 4
    assert receipt["parallel_transport_count"] == 4
    assert receipt["json_decode_count"] == 0
    assert receipt["test_json_decode_count"] == 0
    assert stat.S_IMODE(
        (tmp_path / "control" / h.DOWNLOAD_ATTEMPT_FILENAME).stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(
        (tmp_path / "control" / h.DOWNLOAD_RECEIPT_FILENAME).stat().st_mode
    ) == 0o600
    before = list(calls)
    with pytest.raises(FileExistsError):
        h.download_source_set_once(
            source_root=tmp_path / "source",
            control_root=tmp_path / "control",
            contracts=contracts,
            opener=opener,
        )
    assert calls == before


def test_any_parallel_get_failure_is_whole_attempt_terminal_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = {
        key: f"opaque-{key}\n".encode()
        for key in ("TRAIN", "DEV", "TEST", "TABLES")
    }
    contracts = _download_contracts(payloads)
    barrier = threading.Barrier(4)
    calls: list[str] = []
    lock = threading.Lock()

    def opener(url: str) -> _Response:
        with lock:
            calls.append(url)
        barrier.wait(timeout=5)
        key = next(
            key for key, contract in contracts.items() if contract.raw_url == url
        )
        if key == "DEV":
            raise OSError("SECRET transport diagnostic must not be persisted")
        return _Response(payloads[key], url)

    monkeypatch.setattr(h, "verify_frozen_bindings", lambda *_args, **_kwargs: None)
    with pytest.raises(
        h.HitabP1SourceError, match="four-file acquisition failed closed"
    ):
        h.download_source_set_once(
            source_root=tmp_path / "source",
            control_root=tmp_path / "control",
            contracts=contracts,
            opener=opener,
        )
    assert len(calls) == 4
    for contract in contracts.values():
        destination = tmp_path / "source" / contract.relative_path
        assert not destination.exists()
        assert not destination.with_name(
            f".{destination.name}.one_shot.part"
        ).exists()
    failure_path = tmp_path / "control" / h.DOWNLOAD_FAILURE_FILENAME
    failure_raw = failure_path.read_bytes()
    assert b"SECRET" not in failure_raw
    failure = json.loads(failure_raw)
    assert failure["attempted_file_count"] == 4
    assert failure["retry_resume_range_mirror_or_provider_switch_count"] == 0
    assert stat.S_IMODE(failure_path.stat().st_mode) == 0o600


def test_four_file_identity_is_exact_one_shot_and_test_is_not_decoded(
    tmp_path: Path,
) -> None:
    payloads = {
        "TRAIN": b'{"opaque":"train"}\n',
        "DEV": b'{"opaque":"dev"}\n',
        "TEST": b"NOT EVEN JSON\nSECOND OPAQUE LINE\n",
        "TABLES": b"PK synthetic opaque table archive bytes",
    }
    paths: dict[str, Path] = {}
    contracts: dict[str, h.SourceFileContract] = {}
    expected_sha256: dict[str, str] = {}
    for key, raw in payloads.items():
        path = tmp_path / "source" / key
        _write_private(path, raw)
        paths[key] = path
        contracts[key] = h.SourceFileContract(
            key=key,
            relative_path=f"data/{key.lower()}",
            size_bytes=len(raw),
            git_blob_sha1=h.git_blob_sha1(raw),
            is_jsonl=key != "TABLES",
        )
        expected_sha256[key] = hashlib.sha256(raw).hexdigest()

    verified = h.verify_source_set_once(
        paths,
        expected_sha256_by_key=expected_sha256,
        control_root=tmp_path / "work",
        contracts=contracts,
    )
    receipt = verified.safe_receipt
    _assert_self_hash(receipt)
    assert receipt["file_count"] == 4
    assert receipt["test_json_decode_count"] == 0
    assert receipt["json_decode_count"] == 0
    assert verified.identities["TEST"].raw_newline_count == 2
    identity_only = h.test_identity_only_summary(verified)
    assert identity_only["json_decode_count"] == 0
    assert identity_only["raw_newline_count"] == 2

    marker = tmp_path / "work" / h.SOURCE_ATTEMPT_FILENAME
    persisted = tmp_path / "work" / h.SOURCE_RECEIPT_FILENAME
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert stat.S_IMODE(persisted.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        h.verify_source_set_once(
            paths,
            expected_sha256_by_key=expected_sha256,
            control_root=tmp_path / "work",
            contracts=contracts,
        )


def test_identity_rejects_wrong_sha_git_blob_size_and_consumes_marker(
    tmp_path: Path,
) -> None:
    payloads = {
        key: (key + "\n").encode()
        for key in ("TRAIN", "DEV", "TEST", "TABLES")
    }
    paths: dict[str, Path] = {}
    contracts: dict[str, h.SourceFileContract] = {}
    expected: dict[str, str] = {}
    for key, raw in payloads.items():
        path = tmp_path / "source" / key
        _write_private(path, raw)
        paths[key] = path
        contracts[key] = h.SourceFileContract(
            key,
            f"data/{key}",
            len(raw),
            h.git_blob_sha1(raw),
            key != "TABLES",
        )
        expected[key] = hashlib.sha256(raw).hexdigest()
    expected["DEV"] = "0" * 64
    with pytest.raises(h.HitabP1SourceError, match="byte identity drifted"):
        h.verify_source_set_once(
            paths,
            expected_sha256_by_key=expected,
            control_root=tmp_path / "work",
            contracts=contracts,
        )
    assert (tmp_path / "work" / h.SOURCE_ATTEMPT_FILENAME).exists()
    failure = json.loads(
        (tmp_path / "work" / h.SOURCE_FAILURE_FILENAME).read_text("ascii")
    )
    assert failure["status"] == "terminal_attempt_consumed"
    assert stat.S_IMODE(
        (tmp_path / "work" / h.SOURCE_FAILURE_FILENAME).stat().st_mode
    ) == 0o600
    assert not (tmp_path / "work" / h.SOURCE_RECEIPT_FILENAME).exists()


def test_sample_jsonl_allows_extra_fields_and_excludes_bad_rows_by_reason() -> None:
    exposed_question = "Publicly observed synthetic question"
    exposure = {
        "id": frozenset(),
        "question": frozenset({h.normalized_text_sha256(exposed_question)}),
        "table_id": frozenset(),
    }
    valid = _sample(
        item_id="valid",
        table_id="t-valid",
        aggregation="sum",
        answer=1,
        extra=True,
    )
    # reference_cells_map is an allowed extra, not a qrel input or required
    # semantic field under the frozen coordinate rule.
    valid.pop("reference_cells_map")
    missing = dict(valid)
    missing["id"] = "missing"
    missing.pop("linked_cells")
    wrong_source = _sample(
        item_id="wrong-source",
        table_id="t-wrong",
        aggregation="sum",
        answer=1,
        source="tOtTo",
    )
    mixed = _sample(
        item_id="mixed",
        table_id="t-mixed",
        aggregation="sum",
        answer=1,
    )
    mixed["aggregation"] = ["sum", "max"]
    exposed = _sample(
        item_id="exposed",
        table_id="t-exposed",
        aggregation="max",
        answer=1,
    )
    exposed["question"] = exposed_question
    result = h.parse_sample_jsonl_bytes(
        _jsonl([valid, missing, wrong_source, mixed, exposed]),
        split="TRAIN",
        public_exposure_hashes=exposure,
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].item_id == "valid"
    assert result.candidates[0].family == "AGGREGATE"
    assert result.safe_summary["json_decode_count"] == 5
    assert result.safe_summary["row_exclusion_reason_counts"] == {
        "aggregation_cross_family": 1,
        "public_example_excluded": 1,
        "required_semantic_field_missing": 1,
        "table_source_not_statcan_or_nsf": 1,
    }
    serialized_summary = json.dumps(result.safe_summary)
    assert all(
        secret not in serialized_summary
        for secret in ("valid", "t-valid", "Publicly observed")
    )

    with pytest.raises(h.HitabP1SourceError, match="strict JSON"):
        h.parse_sample_jsonl_bytes(
            b'{"id": 1,}\n',
            split="DEV",
            public_exposure_hashes=EMPTY_EXPOSURE,
        )
    with pytest.raises(h.HitabP1SourceError, match="promotion"):
        h.parse_sample_jsonl_bytes(
            _jsonl([valid]),
            split="TEST",
            public_exposure_hashes=EMPTY_EXPOSURE,
        )


def test_official_hmt_schema_builds_paths_units_offsets_and_exact_typed_edges() -> None:
    payload = _table_payload(seed=10)
    raw = _raw_table_payload(payload)
    view = h.parse_hmt_table(payload, raw)
    assert len(view.units) == 10
    assert view.units[0].left_header_path == ("Region",)
    assert view.units[0].top_header_path == ("Metric",)
    assert view.units[6].left_header_path == ("Region", "Region 1")
    assert view.units[6].top_header_path == ("Metric", "Metric 1")
    assert view.units[0].value_type == "NUMBER"
    assert view.matrix_coordinate_to_ordinal[(2, 2)] == 0
    assert "row_index" not in view.units[0].serialized
    assert "column_index" not in view.units[0].serialized
    assert all(
        edge.source_ordinal < edge.target_ordinal
        and edge.edge_type == "FORWARD_SHARED_AXIS_OR_HEADER"
        for edge in view.typed_edges
    )
    assert len(view.typed_edges) == len(set(view.typed_edges))
    assert not any(
        edge.source_ordinal == edge.target_ordinal for edge in view.typed_edges
    )
    # Raw offsets, not tree depth, define full-matrix coordinates.
    unusual_raw = _raw_table_payload(
        payload, top_header_rows_num=3, left_header_columns_num=1
    )
    unusual = h.parse_hmt_table(payload, unusual_raw)
    assert unusual.matrix_coordinate_to_ordinal[(3, 1)] == 0

    # Equal tree level alone is not an edge.  With flat independent leaves,
    # only shared row/column axes connect off-header pairs.
    flat = _table_payload(seed=50)
    flat["top_root"] = _tree_node(
        "<TOP>",
        None,
        [_tree_node(f"T{column}", column) for column in range(5)],
    )
    flat["left_root"] = _tree_node(
        "<LEFT>",
        None,
        [_tree_node(f"L{row}", row) for row in range(2)],
    )
    flat_view = h.parse_hmt_table(flat, _raw_table_payload(flat))
    edge_pairs = {
        (edge.source_ordinal, edge.target_ordinal)
        for edge in flat_view.typed_edges
    }
    assert (0, 1) in edge_pairs  # same row
    assert (0, 5) in edge_pairs  # same column
    assert (0, 6) not in edge_pairs  # same level only
    assert (6, 0) not in edge_pairs  # reverse forbidden

    too_small = _table_payload(rows=2, columns=5)
    too_small["data"][-1][-1]["value"] = "  "  # type: ignore[index]
    too_small_raw = _raw_table_payload(too_small)
    with pytest.raises(
        h.HitabP1RowIneligible, match="ordered_corpus_size_outside_10_256"
    ):
        h.parse_hmt_table(too_small, too_small_raw)

    ragged = _table_payload()
    ragged["data"][1].pop()  # type: ignore[index]
    with pytest.raises(h.HitabP1SourceError, match="not rectangular"):
        h.parse_hmt_table(ragged, _raw_table_payload(_table_payload()))

    mismatch = _table_payload(seed=70)
    mismatch_raw = _raw_table_payload(mismatch)
    mismatch_raw["texts"][2][2] = 999_999  # type: ignore[index]
    with pytest.raises(h.HitabP1SourceError, match="typed value drifted"):
        h.parse_hmt_table(mismatch, mismatch_raw)


def test_qrel_uses_annotated_coordinates_as_distinct_singleton_requirements() -> None:
    payload = _table_payload(seed=100)
    data = payload["data"]
    assert isinstance(data, list)
    data[0][0]["value"] = 7
    data[0][1]["value"] = "7.0"
    data[0][2]["value"] = 9
    table = h.parse_hmt_table(payload, _raw_table_payload(payload))
    row = _sample(
        item_id="qrel",
        table_id="literal-table",
        aggregation="sum",
        answer=7,
    )
    linked = row["linked_cells"]
    assert isinstance(linked, dict)
    linked["quantity_link"]["[ANSWER]"] = {  # type: ignore[index]
        "(2, 2)": 7.0,
        "(2, 3)": 7,
        "(2, 4)": 9,
    }
    candidate = h.parse_sample_row(
        row,
        split="TRAIN",
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    qrel = h.build_coordinate_qrel_dnf(candidate, table)
    assert qrel.alternatives == ((((0,), (1,), (2,))),)
    assert qrel.ordinal_mapping_commitment == h.stable_hash(
        [[list(bucket) for bucket in qrel.alternatives[0]]]
    )

    six = dict(row)
    six_linked = {
        "quantity_link": {
            "[ANSWER]": {f"({2 + value}, 2)": value for value in range(6)}
        }
    }
    six["linked_cells"] = six_linked
    six_candidate = h.parse_sample_row(
        six,
        split="TRAIN",
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    with pytest.raises(
        h.HitabP1RowIneligible, match="proof_requirement_count_outside_1_5"
    ):
        h.build_coordinate_qrel_dnf(six_candidate, table)

    unresolved = dict(row)
    unresolved["linked_cells"] = {
        "quantity_link": {"[ANSWER]": {"(999, 999)": 999_999}}
    }
    unresolved_candidate = h.parse_sample_row(
        unresolved,
        split="TRAIN",
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    with pytest.raises(
        h.HitabP1RowIneligible, match="proof_coordinate_unresolved"
    ):
        h.build_coordinate_qrel_dnf(unresolved_candidate, table)

    mismatch = dict(row)
    mismatch["linked_cells"] = {
        "quantity_link": {"[ANSWER]": {"(2, 2)": 123_456}}
    }
    mismatch_candidate = h.parse_sample_row(
        mismatch,
        split="TRAIN",
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    with pytest.raises(
        h.HitabP1RowIneligible,
        match="annotation_raw_HMT_typed_value_mismatch",
    ):
        h.build_coordinate_qrel_dnf(mismatch_candidate, table)


def test_zip_opens_only_requested_member_without_blanket_member_grammar(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tables.zip"
    _write_tables_zip(
        path,
        {"wanted": _table_payload(seed=1), "other": _table_payload(seed=2)},
        extras={
            "README.any-extension": b"safe unrelated payload",
            "misc/nested/file.bin": b"also safe and never opened",
        },
    )

    class TrackingZip(zipfile.ZipFile):
        opened: list[str] = []

        def open(self, name, mode="r", pwd=None, *, force_zip64=False):  # type: ignore[no-untyped-def]
            filename = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
            self.__class__.opened.append(filename)
            return super().open(
                name, mode=mode, pwd=pwd, force_zip64=force_zip64
            )

    value = h.read_requested_table_from_zip(
        path,
        "wanted",
        zip_file_factory=TrackingZip,
    )
    assert value.hmt["title"] == "Synthetic table 1"
    assert value.raw["top_header_rows_num"] == 2
    assert TrackingZip.opened == [
        "official/tables/hmt/wanted.json",
        "official/tables/raw/wanted.json",
    ]


@pytest.mark.parametrize("case", ["traversal", "symlink", "duplicate", "oversize"])
def test_zip_rejects_traversal_symlink_duplicate_and_oversize(
    tmp_path: Path, case: str
) -> None:
    path = tmp_path / f"{case}.zip"
    target_name = "tables/hmt/wanted.json"
    raw_target_name = "tables/raw/wanted.json"
    with zipfile.ZipFile(path, "w") as archive:
        payload = _table_payload()
        archive.writestr(target_name, h.canonical_bytes(payload))
        archive.writestr(
            raw_target_name, h.canonical_bytes(_raw_table_payload(payload))
        )
        if case == "traversal":
            archive.writestr("../escape.bin", b"x")
        elif case == "symlink":
            info = zipfile.ZipInfo("misc/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b"target")
        elif case == "duplicate":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(target_name, h.canonical_bytes(_table_payload(seed=1)))
        else:
            archive.writestr("misc/large.bin", b"x" * 128)
    os.chmod(path, 0o600)
    kwargs = {"max_member_bytes": 64} if case == "oversize" else {}
    with pytest.raises(h.HitabP1SourceError):
        h.read_requested_table_from_zip(path, "wanted", **kwargs)


def test_hmac_fixed_quota_is_input_order_invariant_and_one_question_per_table() -> None:
    rows: dict[str, list[h.EligibleItem]] = {"TRAIN": [], "DEV": []}
    seed = 100
    for split, block_prefix in (("TRAIN", "train"), ("DEV", "dev")):
        # The first two families share a table.  A fixed greedy family order
        # consumes it at most once and uses the second comparative candidate.
        rows[split].extend(
            [
                _eligible(
                    split=split,
                    item_id=f"{block_prefix}-agg",
                    table_id=f"{block_prefix}-shared",
                    family="AGGREGATE",
                    seed=seed,
                ),
                _eligible(
                    split=split,
                    item_id=f"{block_prefix}-cmp-shared",
                    table_id=f"{block_prefix}-shared",
                    family="COMPARATIVE",
                    seed=seed,
                ),
                _eligible(
                    split=split,
                    item_id=f"{block_prefix}-cmp-spare",
                    table_id=f"{block_prefix}-cmp-spare",
                    family="COMPARATIVE",
                    seed=seed + 1,
                ),
                _eligible(
                    split=split,
                    item_id=f"{block_prefix}-sup",
                    table_id=f"{block_prefix}-sup",
                    family="SUPERLATIVE",
                    seed=seed + 2,
                ),
            ]
        )
        seed += 10
    quotas = {"A_form": 1, "A_hold": 1, "M_search": 1}
    secret = b"s" * 32
    first = h.select_fixed_quota(
        rows,
        secret=secret,
        quota_per_family=quotas,
    )
    second = h.select_fixed_quota(
        {split: tuple(reversed(values)) for split, values in rows.items()},
        secret=secret,
        quota_per_family=quotas,
    )
    assert first.safe_receipt == second.safe_receipt
    assert first.safe_receipt["per_block_family_count"] == {
        "A_form": {"AGGREGATE": 1, "COMPARATIVE": 1, "SUPERLATIVE": 1},
        "A_hold": {"AGGREGATE": 1, "COMPARATIVE": 1, "SUPERLATIVE": 1},
    }
    selected = [
        row.item.candidate
        for block in h.INITIAL_BLOCKS
        for row in first.selected_by_block[block]
    ]
    table_ids = [row.table_id for row in selected]
    assert len(table_ids) == len(set(table_ids))
    _assert_self_hash(first.safe_receipt)
    receipt_text = json.dumps(first.safe_receipt)
    assert all(
        secret_text not in receipt_text
        for secret_text in (
            "train-shared",
            "dev-shared",
            "Question for",
            "Region",
        )
    )


def test_collision_registry_uses_full_universe_transitive_closure() -> None:
    def with_question(row: h.EligibleItem, question: str) -> h.EligibleItem:
        return replace(
            row,
            candidate=replace(row.candidate, question=question),
        )

    bridge_a = with_question(
        _eligible(
            split="TRAIN",
            item_id="bridge-a",
            table_id="table-one",
            family="AGGREGATE",
            seed=10,
        ),
        "Question one",
    )
    # This row is the skipped bridge: same table as A, same normalized
    # question as future C.
    bridge_b = with_question(
        _eligible(
            split="TRAIN",
            item_id="bridge-b",
            table_id="table-one",
            family="AGGREGATE",
            seed=20,
        ),
        "  QUESTION   TWO ",
    )
    initial_rows = (
        bridge_a,
        bridge_b,
        _eligible(
            split="TRAIN",
            item_id="initial-cmp",
            table_id="initial-cmp-table",
            family="COMPARATIVE",
            seed=30,
        ),
        _eligible(
            split="TRAIN",
            item_id="initial-sup",
            table_id="initial-sup-table",
            family="SUPERLATIVE",
            seed=40,
        ),
    )
    quotas = {"A_form": 1, "A_hold": 1, "M_search": 1}
    initial = h.select_fixed_quota(
        {"TRAIN": initial_rows},
        secret=b"c" * 32,
        blocks=("A_form",),
        quota_per_family=quotas,
    )
    bridge_c = with_question(
        _eligible(
            split="TEST",
            item_id="bridge-c",
            table_id="table-two",
            family="AGGREGATE",
            seed=50,
        ),
        "question two",
    )
    spare = _eligible(
        split="TEST",
        item_id="aggregate-spare",
        table_id="aggregate-spare-table",
        family="AGGREGATE",
        seed=60,
    )
    m_rows = (
        bridge_c,
        spare,
        _eligible(
            split="TEST",
            item_id="m-cmp",
            table_id="m-cmp-table",
            family="COMPARATIVE",
            seed=70,
        ),
        _eligible(
            split="TEST",
            item_id="m-sup",
            table_id="m-sup-table",
            family="SUPERLATIVE",
            seed=80,
        ),
    )
    selected = h.select_fixed_quota(
        {"TEST": m_rows},
        secret=b"c" * 32,
        blocks=("M_search",),
        quota_per_family=quotas,
        prior_component_tokens=initial.used_component_tokens,
        prior_component_registry=initial.component_registry,
    )
    selected_ids = {
        row.item.candidate.item_id
        for row in selected.selected_by_block["M_search"]
    }
    assert "bridge-c" not in selected_ids
    assert "aggregate-spare" in selected_ids
    transitive = [
        component
        for component in selected.component_registry
        if "table:table-one" in component
    ]
    assert len(transitive) == 1
    assert {
        "table:table-one",
        "table:table-two",
        "question:" + h.stable_hash("question two"),
    }.issubset(transitive[0])


def test_initial_one_shot_secret_selection_and_source_free_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_table = _table_payload(seed=9_000)
    source_paths, verified = _prepare_full_source_set(
        tmp_path,
        test_rows=[
            _sample(
                item_id="future-test-item",
                table_id="future-test-table",
                aggregation="sum",
                answer=9_000,
            )
        ],
        test_tables={"future-test-table": test_table},
    )
    monkeypatch.setattr(h.os, "urandom", lambda count: b"z" * count)
    control_root = tmp_path / "work"
    result = h.run_initial_selection_once(
        source_paths=source_paths,
        verified_sources=verified,
        control_root=control_root,
        quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    assert not hasattr(result, "secret")
    assert not hasattr(result, "selection")
    assert set(result.block_views) == {"A_form", "A_hold"}
    assert result.safe_receipt["test_json_decode_count"] == 0
    assert result.safe_receipt["test_identity_only"]["json_decode_count"] == 0
    assert result.safe_receipt["safe_parse_summaries"]["TRAIN"][
        "json_decode_count"
    ] == 3
    assert result.safe_receipt["safe_parse_summaries"]["DEV"][
        "json_decode_count"
    ] == 3
    assert stat.S_IMODE(
        (control_root / h.INITIAL_ATTEMPT_FILENAME).stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(
        (control_root / h.INITIAL_SECRET_FILENAME).stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(
        (control_root / h.INITIAL_RECEIPT_FILENAME).stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(
        (control_root / h.COMPONENT_REGISTRY_FILENAME).stat().st_mode
    ) == 0o600
    for block in ("A_form", "A_hold"):
        assert stat.S_IMODE(
            (
                control_root / h.BLOCK_VIEW_FILENAMES[block]
            ).stat().st_mode
        ) == 0o600
        assert stat.S_IMODE(
            (
                control_root / h.QREL_CUSTODY_FILENAMES[block]
            ).stat().st_mode
        ) == 0o600

    view = result.block_views["A_hold"]
    action = _valid_four_arm_archive(view)
    qrels = h.release_qrels_after_action_seal(
        block="A_hold",
        qrel_custody_path=(
            control_root / h.QREL_CUSTODY_FILENAMES["A_hold"]
        ),
        sealed_action_archive=action,
    )
    assert len(view.items) == len(qrels.rows) == 3
    assert {row.work_id for row in view.items} == {
        row.work_id for row in qrels.rows
    }
    assert all(
        row.corpus_commitment in {
            qrel.corpus_commitment for qrel in qrels.rows
        }
        for row in view.items
    )
    assert all(row.typed_edges for row in view.items)
    # The label-free bridge carries no family/qrel field; only the late pack
    # contains family and singleton proof ordinals.
    view_text = repr(view)
    assert "family=" not in view_text and "qrel=" not in view_text
    assert {row.family for row in qrels.rows} == set(h.FAMILIES)
    assert all(
        all(len(bucket) == 1 for proof in row.qrel.alternatives for bucket in proof)
        for row in qrels.rows
    )
    assert _assert_bridge_hashes(view, qrels)
    with pytest.raises(FileExistsError):
        h.release_qrels_after_action_seal(
            block="A_hold",
            qrel_custody_path=(
                control_root / h.QREL_CUSTODY_FILENAMES["A_hold"]
            ),
            sealed_action_archive=action,
        )
    with pytest.raises(FileExistsError):
        h.run_initial_selection_once(
            source_paths=source_paths,
            verified_sources=verified,
            control_root=control_root,
            quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
            public_exposure_hashes=EMPTY_EXPOSURE,
        )


def _assert_bridge_hashes(
    view: h.BridgeBlockView, qrels: h.BridgeQrelPack
) -> bool:
    view_payload = {
        "block": view.block,
        "items": [row.private_payload() for row in view.items],
    }
    qrel_payload = {
        "action_archive_sha256": qrels.action_archive_sha256,
        "block": qrels.block,
        "rows": [
            {
                "corpus_commitment": row.corpus_commitment,
                "family": row.family,
                "proof": row.qrel.payload(),
                "qrel_ordinal_mapping_commitment": (
                    row.qrel_ordinal_mapping_commitment
                ),
                "work_id": row.work_id,
            }
            for row in qrels.rows
        ],
    }
    return (
        view.view_sha256 == h.stable_hash(view_payload)
        and qrels.pack_sha256 == h.stable_hash(qrel_payload)
    )


def test_qrel_release_rejects_preseal_and_consumes_incomplete_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths, verified = _prepare_full_source_set(
        tmp_path,
        test_rows=[
            _sample(
                item_id="unused-test",
                table_id="unused-test-table",
                aggregation="sum",
                answer=8_000,
            )
        ],
        test_tables={"unused-test-table": _table_payload(seed=8_000)},
    )
    monkeypatch.setattr(h.os, "urandom", lambda count: b"r" * count)
    control_root = tmp_path / "work"
    initial = h.run_initial_selection_once(
        source_paths=source_paths,
        verified_sources=verified,
        control_root=control_root,
        quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    custody = control_root / h.QREL_CUSTODY_FILENAMES["A_hold"]
    marker = control_root / h.QREL_RELEASE_MARKER_FILENAMES["A_hold"]
    qrel_read_count = 0
    original_loader = h._load_qrel_custody

    def tracking_loader(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal qrel_read_count
        qrel_read_count += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(h, "_load_qrel_custody", tracking_loader)
    with pytest.raises(h.HitabP1SourceError, match="self-hashed"):
        h.release_qrels_after_action_seal(
            block="A_hold",
            qrel_custody_path=custody,
            sealed_action_archive={},
        )
    assert not marker.exists()
    assert qrel_read_count == 0

    valid = _valid_four_arm_archive(initial.block_views["A_hold"])
    missing_cache_body = copy.deepcopy(dict(valid))
    missing_cache_body.pop("self_sha256")
    missing_cache_body.pop("gpu0_unused_cuda_cache_release_receipt")
    missing_cache = h.self_hashed(missing_cache_body)
    with pytest.raises(h.HitabP1SourceError, match="header drifted"):
        h.release_qrels_after_action_seal(
            block="A_hold",
            qrel_custody_path=custody,
            sealed_action_archive=missing_cache,
        )
    assert qrel_read_count == 0 and not marker.exists()

    forged_cache_body = copy.deepcopy(dict(valid))
    forged_cache_body.pop("self_sha256")
    cache_body = forged_cache_body[
        "gpu0_unused_cuda_cache_release_receipt"
    ]
    cache_body.pop("self_sha256")  # type: ignore[union-attr]
    cache_body["physical_gpu"] = 1  # type: ignore[index]
    forged_cache_body["gpu0_unused_cuda_cache_release_receipt"] = (
        h.self_hashed(cache_body)  # type: ignore[arg-type]
    )
    forged_cache = h.self_hashed(forged_cache_body)
    with pytest.raises(h.HitabP1SourceError, match="cache release"):
        h.release_qrels_after_action_seal(
            block="A_hold",
            qrel_custody_path=custody,
            sealed_action_archive=forged_cache,
        )
    assert qrel_read_count == 0 and not marker.exists()

    out_of_range_body = copy.deepcopy(dict(valid))
    out_of_range_body.pop("self_sha256")
    out_of_range_body["records"][0]["arms"]["E1"][  # type: ignore[index]
        "top5_ordinals"
    ][0] = 10_000
    out_of_range = h.self_hashed(out_of_range_body)
    with pytest.raises(h.HitabP1SourceError, match="output is incomplete"):
        h.release_qrels_after_action_seal(
            block="A_hold",
            qrel_custody_path=custody,
            sealed_action_archive=out_of_range,
        )
    assert qrel_read_count == 0 and not marker.exists()

    wrong_gpu_body = copy.deepcopy(dict(valid))
    wrong_gpu_body.pop("self_sha256")
    wrong_gpu_body["records"][0]["arms"]["HippoRAG"][  # type: ignore[index]
        "physical_gpu"
    ] = 2
    wrong_gpu = h.self_hashed(wrong_gpu_body)
    with pytest.raises(h.HitabP1SourceError, match="physical lane"):
        h.release_qrels_after_action_seal(
            block="A_hold",
            qrel_custody_path=custody,
            sealed_action_archive=wrong_gpu,
        )
    assert qrel_read_count == 0 and not marker.exists()

    incomplete_body = dict(valid)
    incomplete_body.pop("self_sha256")
    incomplete_body.pop("e1_model_sha256")
    incomplete = h.self_hashed(incomplete_body)
    with pytest.raises(h.HitabP1SourceError, match="header drifted"):
        h.release_qrels_after_action_seal(
            block="A_hold",
            qrel_custody_path=custody,
            sealed_action_archive=incomplete,
        )
    assert not marker.exists()
    assert qrel_read_count == 0
    released = h.release_qrels_after_action_seal(
        block="A_hold",
        qrel_custody_path=custody,
        sealed_action_archive=valid,
    )
    assert released.rows
    assert qrel_read_count == 1
    assert marker.is_file()
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600


def test_test_first_decode_requires_promotion_and_replay_is_impossible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = {
        "m-agg": _table_payload(seed=100),
        "m-cmp": _table_payload(seed=200),
        "m-sup": _table_payload(seed=300),
    }
    test_rows = [
        _sample(
            item_id="m-agg-id",
            table_id="m-agg",
            aggregation="sum",
            answer=100,
        ),
        _sample(
            item_id="m-cmp-id",
            table_id="m-cmp",
            aggregation="diff",
            answer=200,
        ),
        _sample(
            item_id="m-sup-id",
            table_id="m-sup",
            aggregation="max",
            answer=300,
        ),
    ]
    source_paths, verified = _prepare_full_source_set(
        tmp_path,
        test_rows=test_rows,
        test_tables=tables,
    )
    monkeypatch.setattr(h.os, "urandom", lambda count: b"m" * count)
    control_root = tmp_path / "work"
    initial = h.run_initial_selection_once(
        source_paths=source_paths,
        verified_sources=verified,
        control_root=control_root,
        quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    source_commitment = verified.source_identity_commitment
    initial_commitment = str(initial.safe_receipt["selection_commitment"])
    bad_authorization = h.self_hashed(
        {
            "aggregate_exact_utility_net_strictly_positive": False,
            "comparison": "E1_minus_E0",
            "initial_selection_commitment": initial_commitment,
            "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth": True,
            "schema": "hitab_p1_test_first_decode_authorization_v1",
            "source_identity_commitment": source_commitment,
            "status": "A_hold_E1_promoted",
            "study_id": h.STUDY_ID,
        }
    )
    with pytest.raises(h.HitabP1SourceError, match="did not authorize"):
        h.decode_and_select_test_once(
            test_path=source_paths["TEST"],
            tables_zip_path=source_paths["TABLES"],
            verified_sources=verified,
            initial_run=initial,
            authorization=bad_authorization,
            control_root=control_root,
            quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
            public_exposure_hashes=EMPTY_EXPOSURE,
        )
    assert not (control_root / h.TEST_DECODE_ATTEMPT_FILENAME).exists()

    authorization = h.self_hashed(
        {
            "aggregate_exact_utility_net_strictly_positive": True,
            "comparison": "E1_minus_E0",
            "initial_selection_commitment": initial_commitment,
            "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth": True,
            "schema": "hitab_p1_test_first_decode_authorization_v1",
            "source_identity_commitment": source_commitment,
            "status": "A_hold_E1_promoted",
            "study_id": h.STUDY_ID,
        }
    )
    selected = h.decode_and_select_test_once(
        test_path=source_paths["TEST"],
        tables_zip_path=source_paths["TABLES"],
        verified_sources=verified,
        initial_run=initial,
        authorization=authorization,
        control_root=control_root,
        quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    receipt = selected.safe_receipt
    assert len(selected.block_view.items) == 3
    assert receipt["test_json_decode_count"] == 3
    assert stat.S_IMODE(
        (control_root / h.TEST_DECODE_ATTEMPT_FILENAME).stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(
        (control_root / h.TEST_SELECTION_RECEIPT_FILENAME).stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(
        (
            control_root / h.QREL_CUSTODY_FILENAMES["M_search"]
        ).stat().st_mode
    ) == 0o600
    receipt_text = json.dumps(receipt)
    assert all(
        raw not in receipt_text
        for raw in (
            "m-agg-id",
            "m-agg",
            "Question for",
            "Synthetic table",
            "(ignored, coordinate)",
        )
    )
    with pytest.raises(FileExistsError):
        h.decode_and_select_test_once(
            test_path=source_paths["TEST"],
            tables_zip_path=source_paths["TABLES"],
            verified_sources=verified,
            initial_run=initial,
            authorization=authorization,
            control_root=control_root,
            quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
            public_exposure_hashes=EMPTY_EXPOSURE,
        )


@pytest.mark.parametrize("mutated", ["SECRET", "TEST", "TABLES"])
def test_test_decode_loads_persisted_secret_and_revalidates_bound_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated: str,
) -> None:
    tables = {
        "m-agg": _table_payload(seed=100),
        "m-cmp": _table_payload(seed=200),
        "m-sup": _table_payload(seed=300),
    }
    test_rows = [
        _sample(
            item_id="m-agg-id",
            table_id="m-agg",
            aggregation="sum",
            answer=100,
        ),
        _sample(
            item_id="m-cmp-id",
            table_id="m-cmp",
            aggregation="diff",
            answer=200,
        ),
        _sample(
            item_id="m-sup-id",
            table_id="m-sup",
            aggregation="max",
            answer=300,
        ),
    ]
    source_paths, verified = _prepare_full_source_set(
        tmp_path,
        test_rows=test_rows,
        test_tables=tables,
    )
    monkeypatch.setattr(h.os, "urandom", lambda count: b"k" * count)
    control_root = tmp_path / "work"
    initial = h.run_initial_selection_once(
        source_paths=source_paths,
        verified_sources=verified,
        control_root=control_root,
        quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    authorization = h.self_hashed(
        {
            "aggregate_exact_utility_net_strictly_positive": True,
            "comparison": "E1_minus_E0",
            "initial_selection_commitment": initial.safe_receipt[
                "selection_commitment"
            ],
            "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth": True,
            "schema": "hitab_p1_test_first_decode_authorization_v1",
            "source_identity_commitment": (
                verified.source_identity_commitment
            ),
            "status": "A_hold_E1_promoted",
            "study_id": h.STUDY_ID,
        }
    )
    assert "secret" not in inspect.signature(
        h.decode_and_select_test_once
    ).parameters
    if mutated == "SECRET":
        target = control_root / h.INITIAL_SECRET_FILENAME
        target.write_bytes(b"x" * 32)
    else:
        target = source_paths[mutated]
        raw = bytearray(target.read_bytes())
        raw[len(raw) // 2] ^= 1
        target.write_bytes(raw)
    os.chmod(target, 0o600)
    with pytest.raises(h.HitabP1SourceError):
        h.decode_and_select_test_once(
            test_path=source_paths["TEST"],
            tables_zip_path=source_paths["TABLES"],
            verified_sources=verified,
            initial_run=initial,
            authorization=authorization,
            control_root=control_root,
            quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
            public_exposure_hashes=EMPTY_EXPOSURE,
        )
    assert (control_root / h.TEST_DECODE_ATTEMPT_FILENAME).is_file()
    assert (control_root / h.TEST_SELECTION_FAILURE_FILENAME).is_file()
    assert not (
        control_root / h.QREL_CUSTODY_FILENAMES["M_search"]
    ).exists()


def test_production_formal_boundary_claim_load_release_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from assumption_agent.benchmarks import (
        hitab_p1_formal_controller_v1 as formal,
    )

    source_paths, verified = _prepare_full_source_set(
        tmp_path,
        test_rows=[
            _sample(
                item_id="unused-test",
                table_id="unused-test-table",
                aggregation="sum",
                answer=7_000,
            )
        ],
        test_tables={"unused-test-table": _table_payload(seed=7_000)},
    )
    monkeypatch.setattr(h.os, "urandom", lambda count: b"a" * count)
    control_root = tmp_path / "work"
    initial = h.run_initial_selection_once(
        source_paths=source_paths,
        verified_sources=verified,
        control_root=control_root,
        quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    formal_root = tmp_path / "formal"
    formal_root.mkdir(mode=0o700)
    marker = formal.self_hashed(
        {
            "execution_binding_sha256": "d" * 64,
            "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
            "schema": "hitab_p1_formal_controller_v1_one_shot_marker_v1",
            "study_id": h.STUDY_ID,
        }
    )
    marker_path = formal_root / formal.FORMAL_MARKER_FILENAME
    marker_path.write_bytes(formal.canonical_bytes(marker))
    os.chmod(marker_path, 0o400)
    boundary = h.ProductionFormalAcquisitionBoundary(
        source_paths=source_paths,
        verified_sources=verified,
        control_root=control_root,
        formal_work_root=formal_root,
        initial_run=initial,
        quota_per_family={"A_form": 1, "A_hold": 1, "M_search": 1},
        public_exposure_hashes=EMPTY_EXPOSURE,
    )
    claim = boundary.claim_formal_attempt(str(marker["self_sha256"]))
    assert isinstance(claim, formal.AcquisitionClaim)
    assert not hasattr(claim, "qrel") and not hasattr(claim, "secret")
    block = boundary.load_label_free_block("A_hold", None)
    assert isinstance(block, formal.BlockView)
    assert all(row.work_id.startswith("hitab-work-v1-") for row in block.items)
    assert "qrel" not in repr(block).casefold()
    assert "family" not in repr(block).casefold()

    action = _valid_four_arm_archive(initial.block_views["A_hold"])
    action_path = formal_root / "A_hold.actions.private.json"
    action_path.write_bytes(h.canonical_bytes(action))
    os.chmod(action_path, 0o400)
    qrels = boundary.release_qrels_after_action_seal(
        "A_hold",
        action_path,
        action,
    )
    assert isinstance(qrels, formal.QrelPack)
    assert qrels.action_archive_sha256 == action["self_sha256"]
    assert all(
        row.proof.corpus_commitment == row.corpus_commitment
        for row in qrels.rows
    )
    assert not hasattr(h, "split_private_materialization")


def test_frozen_production_factory_reverifies_once_without_test_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from replication_runtime.hitab_p1_formal_v1 import runner

    source_paths, prepared = _prepare_full_source_set(
        tmp_path,
        test_rows=[
            _sample(
                item_id="future-test",
                table_id="future-test-table",
                aggregation="sum",
                answer=6_000,
            )
        ],
        test_tables={"future-test-table": _table_payload(seed=6_000)},
    )
    identities = {
        key: prepared.identities[key].safe_payload()
        for key in ("TRAIN", "DEV", "TEST", "TABLES")
    }
    source_commitment = h.stable_hash(identities)
    control_root = tmp_path / "source-control"
    receipt_path = control_root / h.DOWNLOAD_RECEIPT_FILENAME
    receipt = h.self_hashed(
        {
            "file_count": 4,
            "files": identities,
            "json_decode_count": 0,
            "network_attempt_count": 4,
            "parallel_transport_count": 4,
            "retry_resume_range_mirror_or_provider_switch_count": 0,
            "schema": "hitab_p1_source_download_receipt_v1",
            "source_identity_commitment": source_commitment,
            "status": "four_exact_sources_acquired_once",
            "study_id": h.STUDY_ID,
            "test_json_decode_count": 0,
            "version": h.VERSION,
        }
    )
    _write_private(receipt_path, h.canonical_bytes(receipt, newline=True))
    os.chmod(control_root, 0o700)
    formal_root = tmp_path / "formal"
    formal_root.mkdir(mode=0o700)
    module_path = Path(h.__file__).resolve()
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    synthetic_runtime = runner.FrozenPythonRuntime(
        role="outer",
        executable=Path("/usr/bin/python3"),
        resolved_target=Path("/usr/bin/python3.10"),
        resolved_target_receipt={},
        pyvenv_cfg=tmp_path / "pyvenv.cfg",
        pyvenv_cfg_receipt={},
        python_version="3.10.12",
        stdlib_root=Path("/usr/lib/python3.10"),
        stdlib_tree_receipt={},
        python_zip_path=Path("/usr/lib/python310.zip"),
        ordered_roots=(),
        tree_receipts=(),
        import_probe={},
    )
    implementation = runner.FrozenImplementation(
        path=module_path,
        self_sha256="1" * 64,
        project_root=PROJECT_ROOT,
        outer_runtime=synthetic_runtime,
        hippo_runtime=synthetic_runtime,
        files={
            "hitab_source_acquisition": module_path,
            "hitab_source_custody": module_path,
            "hitab_study_design": module_path,
        },
        file_sha256s={"hitab_source_acquisition": module_sha256},
        models={},
        model_tree_sha256s={},
        minilm_asset_manifest=module_path,
        minilm_asset_manifest_sha256="3" * 64,
        hippo_source_root=tmp_path,
        hippo_source_tree_receipt={},
        hippo_source_file_count=0,
        hippo_source_size_bytes=0,
        hippo_source_tree_sha256="6" * 64,
        hippo_legacy_source_root=tmp_path,
        hippo_worker_module="synthetic",
        runtime_policy={},
    )
    execution = runner.FrozenExecution(
        path=tmp_path / "execution.freeze.json",
        self_sha256="4" * 64,
        implementation=implementation,
        canary_receipt_path=module_path,
        canary_receipt_self_sha256="5" * 64,
        source_receipt_path=receipt_path,
        source_receipt_self_sha256=str(receipt["self_sha256"]),
        source_identity_commitment=source_commitment,
        source_paths=source_paths,
        source_sha256s={
            key: str(identities[key]["sha256"])
            for key in ("TRAIN", "DEV", "TEST", "TABLES")
        },
        formal_work_root=formal_root,
        acquisition_factory_module=(
            "assumption_agent.benchmarks."
            "hitab_p1_source_acquisition_v1"
        ),
        acquisition_factory_attribute=(
            "build_production_boundary_from_execution"
        ),
        acquisition_factory_file_label="hitab_source_acquisition",
    )
    monkeypatch.setattr(h, "verify_frozen_bindings", lambda **_kwargs: None)
    for block in h.BLOCK_QUOTA_PER_FAMILY:
        monkeypatch.setitem(h.BLOCK_QUOTA_PER_FAMILY, block, 1)
    for key, path in source_paths.items():
        raw = path.read_bytes()
        monkeypatch.setitem(
            h.FORMAL_SOURCE_CONTRACTS,
            key,
            h.SourceFileContract(
                key=key,
                relative_path=f"synthetic/{key}",
                size_bytes=len(raw),
                git_blob_sha1=h.git_blob_sha1(raw),
                is_jsonl=key != "TABLES",
                raw_url=None,
            ),
        )
    monkeypatch.setattr(h.os, "urandom", lambda count: b"f" * count)
    assert tuple(
        inspect.signature(
            h.build_production_boundary_from_execution
        ).parameters
    ) == ("execution",)
    boundary = h.build_production_boundary_from_execution(execution)
    assert isinstance(boundary, h.ProductionFormalAcquisitionBoundary)
    assert (
        control_root / h.SOURCE_ATTEMPT_FILENAME
    ).is_file()
    assert (
        control_root / h.INITIAL_ATTEMPT_FILENAME
    ).is_file()
    assert not (
        control_root / h.TEST_DECODE_ATTEMPT_FILENAME
    ).exists()
    with pytest.raises(FileExistsError):
        h.build_production_boundary_from_execution(execution)
