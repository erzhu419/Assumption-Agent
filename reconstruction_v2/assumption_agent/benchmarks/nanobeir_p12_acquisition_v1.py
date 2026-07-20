"""One-shot private HMAC acquisition for the frozen NanoBEIR P12 v2 study."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p11_acquisition_v1 as p11,
)


SCHEMA = "nanobeir_p12_acquisition_result_v1"
ATTEMPT_SCHEMA = "nanobeir_p12_acquisition_attempt_v1"
FREEZE_SCHEMA = "nanobeir_p12_acquisition_implementation_freeze_v1"
FAMILIES = ("NanoMSMARCO", "NanoNQ", "NanoQuoraRetrieval")
BLOCK_COUNTS = p11.BLOCK_COUNTS
RESERVE_COUNT = p11.RESERVE_COUNT
DOCUMENT_PROJECTION_CHARACTERS = p11.DOCUMENT_PROJECTION_CHARACTERS

SOURCE_ROOT_RELATIVE = Path("artifacts/nanobeir_p12_source_v1/dataset")
RUN_ROOT_RELATIVE = Path("artifacts/nanobeir_p12_acquisition_v1")
RESULT_RELATIVE = Path("manifests/nanobeir_p12_acquisition_result_v1.json")
FREEZE_RELATIVE = Path(
    "manifests/nanobeir_p12_acquisition_implementation_freeze_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/nanobeir_p12_acquisition_v1.py"
)
TEST_RELATIVE = Path("tests/test_nanobeir_p12_acquisition_v1.py")

PRECONDITIONS = {
    "candidate": {
        "relative": "manifests/nanobeir_p12_candidate_freeze_v1.json",
        "file_sha256": "156343887ddda849f24e3d29fc8e577585e46e86b9f24473d2b953c22773f519",
        "self_sha256": "2421b8c9fec755f6a7087621771b376dd77a4a726ef23ee8c248268044a5bd9e",
    },
    "custody": {
        "relative": "manifests/nanobeir_p12_source_custody_v2.json",
        "file_sha256": "ffca29d0ca63ff3638bfeb2bb7ca49167e1af015c57e4c31cc9f9d4b5115f191",
        "self_sha256": "74aa37c5889869ed62a30190b726065f2b5ceb7617fc9d6fe0ed8578da8dfe53",
    },
    "design": {
        "relative": "manifests/nanobeir_p12_study_design_v2.json",
        "file_sha256": "3039743eda3008894751429d044447e35cd8e5ac4a937128c7c4f29b42f3336e",
        "self_sha256": "07d8bd294977696910a55c2e56e37870ffe04eab67262177d304c8f9a8bb78b4",
    },
    "source_access": {
        "relative": "manifests/nanobeir_p12_source_access_v2.json",
        "file_sha256": "68959af87fa7d56b306d47169b8aaffb84412b7ff988fe8b38699a1407ed8186",
        "self_sha256": "49d9dcedbb2800564fa94088dbea7212bc6ebdd72d77c371f41456cb1da0baa1",
    },
}

SOURCE_FILES = {
    "corpus/NanoMSMARCO-00000-of-00001.parquet": "685715c7e0a66d0219572dcd43c3905782868d1aae885259768431f7d7eda830",
    "corpus/NanoNQ-00000-of-00001.parquet": "85d306945cd09cb748ca5b198b281a4f1f034b8240f8c5ecacceb68e38a1db0a",
    "corpus/NanoQuoraRetrieval-00000-of-00001.parquet": "c1e1efd3ed13e3458788f973706ff0ef8cdaedd028ba8ca3454421a92eee0659",
    "qrels/NanoMSMARCO-00000-of-00001.parquet": "6cd84c97a6ed813ffccbbb0b7aacc3051641f40a5869e0a15415823caf65c0d1",
    "qrels/NanoNQ-00000-of-00001.parquet": "f08f73ba0246a9ec1282ca26b48faa24cffb7e1e223354b7fb14fa9f4339e112",
    "qrels/NanoQuoraRetrieval-00000-of-00001.parquet": "d7ea957c68ea0736465643ee388b37669e21b953aae715f7f22aeedbd0819b12",
    "queries/NanoMSMARCO-00000-of-00001.parquet": "7cb9d7534660847f303211b9bdf84bcb3a3530f6e20e3c6050e77fc7ae77d0cd",
    "queries/NanoNQ-00000-of-00001.parquet": "3731f4ac7be9dc1054783ea700ee883d8fd8ad2283da259b1216fff0b4107a5e",
    "queries/NanoQuoraRetrieval-00000-of-00001.parquet": "6c5d11240a0d3868bd4fd8c2d2dc76b4de0b21e777501832aae5ce659faaf633",
}


class NanoBEIRP12AcquisitionError(RuntimeError):
    """The frozen NanoBEIR P12 acquisition failed closed."""


class OneShotRefusal(NanoBEIRP12AcquisitionError):
    """The acquisition root or public result is already consumed."""


canonical_json_bytes = p11.canonical_json_bytes
stable_hash = p11.stable_hash
file_sha256 = p11.file_sha256
self_hashed = p11.self_hashed
_write_exclusive = p11._write_exclusive
_write_json = p11._write_json


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NanoBEIRP12AcquisitionError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NanoBEIRP12AcquisitionError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise NanoBEIRP12AcquisitionError(f"{name} is not an object")
    return value


def _verify_self_hash(value: Mapping[str, Any], expected: str) -> None:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if declared != expected or stable_hash(body) != expected:
        raise NanoBEIRP12AcquisitionError("manifest self hash drifted")


def _verify_preconditions(base: Path) -> Mapping[str, Any]:
    loaded: dict[str, Any] = {}
    for name, binding in PRECONDITIONS.items():
        path = base / binding["relative"]
        if file_sha256(path) != binding["file_sha256"]:
            raise NanoBEIRP12AcquisitionError(f"{name} manifest file drifted")
        value = _read_json(path, name)
        _verify_self_hash(value, binding["self_sha256"])
        loaded[name] = value
    if loaded["source_access"].get("qualification", {}).get("source_passed") is not True:
        raise NanoBEIRP12AcquisitionError("source qualification did not pass")
    for relative, expected in SOURCE_FILES.items():
        path = base / SOURCE_ROOT_RELATIVE / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise NanoBEIRP12AcquisitionError("pinned source file drifted")
    return loaded


def _verify_freeze(base: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "acquisition freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise NanoBEIRP12AcquisitionError("acquisition freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise NanoBEIRP12AcquisitionError("acquisition freeze hash is absent")
    _verify_self_hash(value, declared)
    rows = value.get("implementation_bindings")
    if not isinstance(rows, list):
        raise NanoBEIRP12AcquisitionError("acquisition freeze bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_paths = {IMPLEMENTATION_RELATIVE.as_posix(), TEST_RELATIVE.as_posix()}
    if set(observed) != expected_paths:
        raise NanoBEIRP12AcquisitionError("acquisition implementation set drifted")
    for relative, expected in observed.items():
        if not isinstance(expected, str) or file_sha256(base / str(relative)) != expected:
            raise NanoBEIRP12AcquisitionError("acquisition implementation drifted")
    return value


def project_document(text: object) -> str:
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise NanoBEIRP12AcquisitionError("document text is invalid")
    return text[:DOCUMENT_PROJECTION_CHARACTERS]


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise NanoBEIRP12AcquisitionError(f"{name} is invalid")
    return value


def _read_family(
    base: Path, family: str
) -> tuple[tuple[dict[str, str], ...], dict[str, tuple[str, ...]], int]:
    if family not in FAMILIES:
        raise NanoBEIRP12AcquisitionError("family is outside the frozen set")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NanoBEIRP12AcquisitionError("pyarrow is unavailable") from exc
    root = base / SOURCE_ROOT_RELATIVE
    corpus_table = pq.read_table(root / "corpus" / f"{family}-00000-of-00001.parquet")
    query_table = pq.read_table(root / "queries" / f"{family}-00000-of-00001.parquet")
    qrel_table = pq.read_table(root / "qrels" / f"{family}-00000-of-00001.parquet")
    if (
        corpus_table.column_names != ["_id", "text"]
        or query_table.column_names != ["_id", "text"]
        or qrel_table.column_names != ["query-id", "corpus-id"]
    ):
        raise NanoBEIRP12AcquisitionError("source Parquet schema drifted")
    corpus_ids: set[str] = set()
    projected_count = 0
    for row in corpus_table.to_pylist():
        identifier = _required_text(row.get("_id"), "corpus ID")
        if identifier in corpus_ids:
            raise NanoBEIRP12AcquisitionError("duplicate corpus ID")
        corpus_ids.add(identifier)
        text = _required_text(row.get("text"), "corpus text")
        projected_count += len(text) > DOCUMENT_PROJECTION_CHARACTERS
    queries: list[dict[str, str]] = []
    query_ids: set[str] = set()
    query_texts: set[str] = set()
    for row in query_table.to_pylist():
        identifier = _required_text(row.get("_id"), "query ID")
        text = _required_text(row.get("text"), "query text")
        if identifier in query_ids or text in query_texts:
            raise NanoBEIRP12AcquisitionError("duplicate query ID or text")
        query_ids.add(identifier)
        query_texts.add(text)
        queries.append({"query_id": identifier, "query": text})
    qrels: dict[str, list[str]] = {identifier: [] for identifier in query_ids}
    seen_pairs: set[tuple[str, str]] = set()
    for row in qrel_table.to_pylist():
        query_id = _required_text(row.get("query-id"), "qrel query ID")
        document_id = _required_text(row.get("corpus-id"), "qrel corpus ID")
        pair = (query_id, document_id)
        if query_id not in query_ids or document_id not in corpus_ids or pair in seen_pairs:
            raise NanoBEIRP12AcquisitionError("qrel referential integrity drifted")
        seen_pairs.add(pair)
        qrels[query_id].append(document_id)
    if any(not values for values in qrels.values()):
        raise NanoBEIRP12AcquisitionError("query without a positive qrel")
    return (
        tuple(queries),
        {key: tuple(sorted(values)) for key, values in qrels.items()},
        projected_count,
    )


def hmac_order(secret: bytes, family: str, query_ids: Sequence[str]) -> tuple[str, ...]:
    if len(secret) != 32 or family not in FAMILIES:
        raise NanoBEIRP12AcquisitionError("HMAC ordering input drifted")
    unique = tuple(query_ids)
    if len(set(unique)) != len(unique):
        raise NanoBEIRP12AcquisitionError("HMAC query IDs are duplicated")
    return tuple(
        sorted(
            unique,
            key=lambda query_id: (
                hmac.new(
                    secret,
                    (family + "\n" + query_id).encode("utf-8"),
                    hashlib.sha256,
                ).digest(),
                query_id,
            ),
        )
    )


def allocate_blocks(
    secret: bytes, family: str, query_ids: Sequence[str]
) -> Mapping[str, tuple[str, ...]]:
    ordered = hmac_order(secret, family, query_ids)
    required = sum(count for _block, count in BLOCK_COUNTS) + RESERVE_COUNT
    if len(ordered) != required:
        raise NanoBEIRP12AcquisitionError("family capacity is not exactly 50")
    result: dict[str, tuple[str, ...]] = {}
    offset = 0
    for block, count in BLOCK_COUNTS:
        result[block] = ordered[offset : offset + count]
        offset += count
    result["RESERVE"] = ordered[offset : offset + RESERVE_COUNT]
    if offset + RESERVE_COUNT != len(ordered):
        raise NanoBEIRP12AcquisitionError("block allocation drifted")
    return result


def _item_key(secret: bytes, family: str, query_id: str) -> str:
    digest = hmac.new(
        secret,
        ("item\n" + family + "\n" + query_id).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hashlib.sha256((family + "\n" + query_id + "\n" + digest).encode()).hexdigest()


def run_formal(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    root = base / RUN_ROOT_RELATIVE
    result_path = base / RESULT_RELATIVE
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("acquisition root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("acquisition result already exists")
    preconditions = _verify_preconditions(base)
    freeze = _verify_freeze(base)
    family_data = {family: _read_family(base, family) for family in FAMILIES}
    if any(len(value[0]) != 50 for value in family_data.values()):
        raise NanoBEIRP12AcquisitionError("source capacity drifted")
    root.mkdir(mode=0o700)
    private = root / "private"
    private.mkdir(mode=0o700)
    secret = os.urandom(32)
    secret_path = private / "selection.secret"
    _write_exclusive(secret_path, secret)
    marker = {
        "candidate_freeze_self_sha256": PRECONDITIONS["candidate"]["self_sha256"],
        "formal_implementation_commit": freeze["formal_implementation_commit"],
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
        "selection_secret_sha256": hashlib.sha256(secret).hexdigest(),
        "study_design_self_sha256": PRECONDITIONS["design"]["self_sha256"],
    }
    marker_path = root / "attempt.marker"
    _write_json(marker_path, marker)
    views: dict[str, list[dict[str, Any]]] = {
        block: [] for block, _count in BLOCK_COUNTS
    }
    labels: dict[str, list[dict[str, Any]]] = {
        block: [] for block in ("C_confirm", "A_form", "A_hold", "M_search")
    }
    allocation_commitments: dict[str, dict[str, list[str]]] = {}
    projected_counts: dict[str, int] = {}
    for family in FAMILIES:
        queries, qrels, projected_count = family_data[family]
        projected_counts[family] = projected_count
        by_id = {row["query_id"]: row for row in queries}
        allocated = allocate_blocks(secret, family, tuple(by_id))
        allocation_commitments[family] = {}
        for block, query_ids in allocated.items():
            allocation_commitments[family][block] = [
                _item_key(secret, family, query_id) for query_id in query_ids
            ]
            if block == "RESERVE":
                continue
            for family_ordinal, query_id in enumerate(query_ids):
                key = _item_key(secret, family, query_id)
                views[block].append(
                    {
                        "family": family,
                        "family_ordinal": family_ordinal,
                        "item_key": key,
                        "query": by_id[query_id]["query"],
                        "source_query_id": query_id,
                    }
                )
                if block in labels:
                    labels[block].append(
                        {
                            "family": family,
                            "gold_document_ids": list(qrels[query_id]),
                            "item_key": key,
                        }
                    )
    pack_bindings: dict[str, Any] = {}
    for block, rows in views.items():
        pack = self_hashed(
            {
                "block": block,
                "items": rows,
                "schema": "nanobeir_p12_private_view_v1",
            },
            field="pack_sha256",
        )
        path = private / f"{block}.view.json"
        _write_json(path, pack)
        pack_bindings[f"{block}_view"] = {
            "file_sha256": file_sha256(path),
            "item_count": len(rows),
            "pack_sha256": pack["pack_sha256"],
            "relative_path": path.relative_to(base).as_posix(),
            "size_bytes": path.stat().st_size,
        }
    for block, rows in labels.items():
        pack = self_hashed(
            {
                "block": block,
                "items": rows,
                "schema": "nanobeir_p12_private_labels_v1",
            },
            field="pack_sha256",
        )
        path = private / f"{block}.labels.json"
        _write_json(path, pack)
        pack_bindings[f"{block}_labels"] = {
            "file_sha256": file_sha256(path),
            "item_count": len(rows),
            "pack_sha256": pack["pack_sha256"],
            "relative_path": path.relative_to(base).as_posix(),
            "size_bytes": path.stat().st_size,
        }
    result = self_hashed(
        {
            "allocation": {
                "block_family_counts": {
                    block: {family: count for family in FAMILIES}
                    for block, count in BLOCK_COUNTS
                },
                "commitment_set_sha256": stable_hash(allocation_commitments),
                "reserve_count_per_family": RESERVE_COUNT,
                "total_selected_item_count": sum(len(rows) for rows in views.values()),
            },
            "attempt_binding": {
                "attempt_marker_sha256": file_sha256(marker_path),
                "formal_implementation_commit": freeze["formal_implementation_commit"],
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "selection_secret_sha256": hashlib.sha256(secret).hexdigest(),
            },
            "claim_boundary": {
                "action_model_evaluator_or_score_count": 0,
                "external_network_call_count": 0,
                "individual_item_or_label_value_published": False,
                "performance_claim": False,
            },
            "pack_bindings": pack_bindings,
            "projection": {
                "character_cap": DOCUMENT_PROJECTION_CHARACTERS,
                "documents_projected_by_family": projected_counts,
                "shared_across_all_arms": True,
            },
            "recorded_date": "2026-07-21",
            "schema": SCHEMA,
            "source_custody_self_sha256": preconditions["custody"]["self_sha256"],
            "status": "passed_138_item_private_acquisition_ready_for_P12_C_confirm_runtime",
            "study_design_self_sha256": preconditions["design"]["self_sha256"],
        }
    )
    _write_json(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.formal:
        raise SystemExit("--formal is required")
    value = run_formal(arguments.project_root)
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
