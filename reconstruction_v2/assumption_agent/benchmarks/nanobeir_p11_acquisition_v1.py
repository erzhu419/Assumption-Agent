"""One-shot private HMAC acquisition for the frozen NanoBEIR P11 study."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "nanobeir_p11_acquisition_result_v1"
ATTEMPT_SCHEMA = "nanobeir_p11_acquisition_attempt_v1"
FREEZE_SCHEMA = "nanobeir_p11_acquisition_implementation_freeze_v1"
FAMILIES = ("NanoClimateFEVER", "NanoDBPedia", "NanoHotpotQA")
BLOCK_COUNTS = (
    ("C_confirm", 12),
    ("A_form", 10),
    ("F_search", 8),
    ("A_hold", 8),
    ("M_search", 8),
)
RESERVE_COUNT = 4
DOCUMENT_PROJECTION_CHARACTERS = 3000

SOURCE_ROOT_RELATIVE = Path("artifacts/nanobeir_p11_source_v1/dataset")
RUN_ROOT_RELATIVE = Path("artifacts/nanobeir_p11_acquisition_v1")
RESULT_RELATIVE = Path("manifests/nanobeir_p11_acquisition_result_v1.json")
FREEZE_RELATIVE = Path(
    "manifests/nanobeir_p11_acquisition_implementation_freeze_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/nanobeir_p11_acquisition_v1.py"
)
TEST_RELATIVE = Path("tests/test_nanobeir_p11_acquisition_v1.py")

PRECONDITIONS = {
    "candidate": {
        "relative": "manifests/nanobeir_p11_candidate_freeze_v1.json",
        "file_sha256": "86bfca4f3ee70a72bb4e757343d09bf80ac8a37870be7840ea23b4872a7fed36",
        "self_sha256": "aa49d5c0c194bd600486d64b5e94b29576d746cf4d23eff74d17653771293791",
    },
    "custody": {
        "relative": "manifests/nanobeir_p11_source_custody_v1.json",
        "file_sha256": "366437a7f9b29329f9af83b33c4e435958fc2e3c8799357232c588d8bda7feab",
        "self_sha256": "e5e7290a09291ca64b25fac85d6df9695034fd0db568de4fcefe5626bd8b56be",
    },
    "design": {
        "relative": "manifests/nanobeir_p11_study_design_v1.json",
        "file_sha256": "fc8228fabc74a6c640bd48f08282d6cff1cf72a60347ab0eff4d4660de959d7b",
        "self_sha256": "992f817ff35aa9da0cde0c8c70b659338cce5ab0f4399fc87b041e16ac6ce17f",
    },
    "source_access": {
        "relative": "manifests/nanobeir_p11_source_access_v1.json",
        "file_sha256": "0fe6d142a3b4e747d4b9ca58adbee07844067c45da7d3142c68e7985fe097283",
        "self_sha256": "1697d4ad52e9c53a48db6fffb5f6b6fc67ceacd37e2635285e515816b9a9684b",
    },
}

SOURCE_FILES = {
    "corpus/NanoClimateFEVER-00000-of-00001.parquet": "01da7329c55ecc4a8e8a3544e3df99e6bea9f40aa511db18befcb10de39f2990",
    "corpus/NanoDBPedia-00000-of-00001.parquet": "fdaca6b15b4f2231c31ecd50569e31c4743bf2961598c577ce0f9a4a2ee4ac1f",
    "corpus/NanoHotpotQA-00000-of-00001.parquet": "2b57b69da195e7349d210405fc4250d5ab20373aadf2cae6c15247eae30727e7",
    "qrels/NanoClimateFEVER-00000-of-00001.parquet": "806b8b8f787b1f8f19c367512e92c73d933716d9f6b079999ba2b2ed442d8340",
    "qrels/NanoDBPedia-00000-of-00001.parquet": "3090230e42a469717641985eccd839b78b1d54aa6cabbb0a83f08ee2f963068a",
    "qrels/NanoHotpotQA-00000-of-00001.parquet": "2bb77ea2883970e1a48b00e5c24a5a61206bf534f7f892e0b8070d7112de7a1b",
    "queries/NanoClimateFEVER-00000-of-00001.parquet": "4cea2c60a71aca2efde3d2aa64e4e4a2441da8f460e0d6a43fe7657975edd3b7",
    "queries/NanoDBPedia-00000-of-00001.parquet": "d90da885f45533a77b7b671666afbe22fbba2a3a691ac06c044d990959b50b0a",
    "queries/NanoHotpotQA-00000-of-00001.parquet": "08a64060e167573c974a965a944effafa89e2aa9dd4d97a3411c82571e249387",
}


class NanoBEIRAcquisitionError(RuntimeError):
    """The frozen NanoBEIR acquisition failed closed."""


class OneShotRefusal(NanoBEIRAcquisitionError):
    """The acquisition root or public result is already consumed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise NanoBEIRAcquisitionError("value is not canonical JSON") from exc


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)[:-1]).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def self_hashed(value: Mapping[str, Any], field: str = "self_sha256") -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise NanoBEIRAcquisitionError("self hash field already exists")
    result[field] = stable_hash(result)
    return result


def verify_self_hash(value: Mapping[str, Any], expected: str) -> None:
    declared = value.get("self_sha256")
    body = dict(value)
    body.pop("self_sha256", None)
    if declared != expected or stable_hash(body) != expected:
        raise NanoBEIRAcquisitionError("manifest self hash drifted")


def _write_exclusive(path: Path, raw: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(path, canonical_json_bytes(value))


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NanoBEIRAcquisitionError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NanoBEIRAcquisitionError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise NanoBEIRAcquisitionError(f"{name} is not an object")
    return value


def _verify_preconditions(base: Path) -> Mapping[str, Any]:
    loaded: dict[str, Any] = {}
    for name, binding in PRECONDITIONS.items():
        path = base / binding["relative"]
        if file_sha256(path) != binding["file_sha256"]:
            raise NanoBEIRAcquisitionError(f"{name} manifest file drifted")
        value = _read_json(path, name)
        verify_self_hash(value, binding["self_sha256"])
        loaded[name] = value
    if loaded["source_access"].get("qualification", {}).get("source_passed") is not True:
        raise NanoBEIRAcquisitionError("source qualification did not pass")
    for relative, expected in SOURCE_FILES.items():
        path = base / SOURCE_ROOT_RELATIVE / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise NanoBEIRAcquisitionError("pinned source file drifted")
    return loaded


def _verify_freeze(base: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "acquisition freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise NanoBEIRAcquisitionError("acquisition freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise NanoBEIRAcquisitionError("acquisition freeze hash is absent")
    verify_self_hash(value, declared)
    rows = value.get("implementation_bindings")
    if not isinstance(rows, list):
        raise NanoBEIRAcquisitionError("acquisition freeze bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_paths = {IMPLEMENTATION_RELATIVE.as_posix(), TEST_RELATIVE.as_posix()}
    if set(observed) != expected_paths:
        raise NanoBEIRAcquisitionError("acquisition implementation set drifted")
    for relative, expected in observed.items():
        if not isinstance(expected, str) or file_sha256(base / str(relative)) != expected:
            raise NanoBEIRAcquisitionError("acquisition implementation drifted")
    return value


def project_document(text: object) -> str:
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise NanoBEIRAcquisitionError("document text is invalid")
    return text[:DOCUMENT_PROJECTION_CHARACTERS]


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise NanoBEIRAcquisitionError(f"{name} is invalid")
    return value


def _read_family(base: Path, family: str) -> tuple[tuple[dict[str, str], ...], dict[str, tuple[str, ...]], int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NanoBEIRAcquisitionError("pyarrow is unavailable") from exc
    root = base / SOURCE_ROOT_RELATIVE
    corpus_table = pq.read_table(root / "corpus" / f"{family}-00000-of-00001.parquet")
    query_table = pq.read_table(root / "queries" / f"{family}-00000-of-00001.parquet")
    qrel_table = pq.read_table(root / "qrels" / f"{family}-00000-of-00001.parquet")
    if corpus_table.column_names != ["_id", "text"] or query_table.column_names != ["_id", "text"] or qrel_table.column_names != ["query-id", "corpus-id"]:
        raise NanoBEIRAcquisitionError("source Parquet schema drifted")
    corpus_ids: set[str] = set()
    projected_count = 0
    for row in corpus_table.to_pylist():
        identifier = _required_text(row.get("_id"), "corpus ID")
        if identifier in corpus_ids:
            raise NanoBEIRAcquisitionError("duplicate corpus ID")
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
            raise NanoBEIRAcquisitionError("duplicate query ID or text")
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
            raise NanoBEIRAcquisitionError("qrel referential integrity drifted")
        seen_pairs.add(pair)
        qrels[query_id].append(document_id)
    if any(not values for values in qrels.values()):
        raise NanoBEIRAcquisitionError("query without a positive qrel")
    frozen_qrels = {key: tuple(sorted(values)) for key, values in qrels.items()}
    return tuple(queries), frozen_qrels, projected_count


def hmac_order(secret: bytes, family: str, query_ids: Sequence[str]) -> tuple[str, ...]:
    if len(secret) != 32 or family not in FAMILIES:
        raise NanoBEIRAcquisitionError("HMAC ordering input drifted")
    unique = tuple(query_ids)
    if len(set(unique)) != len(unique):
        raise NanoBEIRAcquisitionError("HMAC query IDs are duplicated")
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


def allocate_blocks(secret: bytes, family: str, query_ids: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
    ordered = hmac_order(secret, family, query_ids)
    required = sum(count for _block, count in BLOCK_COUNTS) + RESERVE_COUNT
    if len(ordered) != required:
        raise NanoBEIRAcquisitionError("family capacity is not exactly 50")
    result: dict[str, tuple[str, ...]] = {}
    offset = 0
    for block, count in BLOCK_COUNTS:
        result[block] = ordered[offset : offset + count]
        offset += count
    result["RESERVE"] = ordered[offset : offset + RESERVE_COUNT]
    if offset + RESERVE_COUNT != len(ordered):
        raise NanoBEIRAcquisitionError("block allocation drifted")
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
    family_data: dict[str, tuple[tuple[dict[str, str], ...], dict[str, tuple[str, ...]], int]] = {
        family: _read_family(base, family) for family in FAMILIES
    }
    if any(len(value[0]) != 50 for value in family_data.values()):
        raise NanoBEIRAcquisitionError("source capacity drifted")
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
    views: dict[str, list[dict[str, Any]]] = {block: [] for block, _count in BLOCK_COUNTS}
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
                "schema": "nanobeir_p11_private_view_v1",
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
                "schema": "nanobeir_p11_private_labels_v1",
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
            "recorded_date": "2026-07-20",
            "schema": SCHEMA,
            "source_custody_self_sha256": preconditions["custody"]["self_sha256"],
            "status": "passed_138_item_private_acquisition_ready_for_C_confirm_runtime",
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
