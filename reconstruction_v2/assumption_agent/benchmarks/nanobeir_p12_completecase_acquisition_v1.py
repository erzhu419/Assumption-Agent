"""One-shot HMAC acquisition from the frozen P12 complete-case screen.

The screen's terminal set is fixed before a fresh secret exists.  This module
selects exactly 36 eligible queries per family, keeps labels in separate packs,
and binds the exact HippoRAG output bytes produced by the screen so downstream
blocks never relaunch that comparator.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_acquisition_v1 as utilities,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_completecase_availability_v1 as availability,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_runtime_v1 as train,
)


SCHEMA = "nanobeir_p12_completecase_acquisition_result_v1"
ATTEMPT_SCHEMA = "nanobeir_p12_completecase_acquisition_attempt_v1"
FREEZE_SCHEMA = "nanobeir_p12_completecase_acquisition_freeze_v1"
VIEW_SCHEMA = "nanobeir_p12_completecase_private_view_v1"
LABEL_SCHEMA = "nanobeir_p12_completecase_private_labels_v1"
HIPPO_SCHEMA = "nanobeir_p12_completecase_private_hipporag_v1"

FAMILIES = availability.FAMILIES
BLOCK_COUNTS = (
    ("C_confirm", 10),
    ("A_form", 8),
    ("F_search", 5),
    ("A_hold", 5),
    ("M_search", 5),
)
RESERVE_COUNT = 3
SELECTED_PER_FAMILY = 36
LABEL_BLOCKS = frozenset(("C_confirm", "A_form", "A_hold", "M_search"))

SOURCE_ROOT_RELATIVE = availability.SOURCE_ROOT_RELATIVE
AVAILABILITY_ROOT_RELATIVE = availability.RUN_ROOT_RELATIVE
AVAILABILITY_RESULT_RELATIVE = availability.RESULT_RELATIVE
RUN_ROOT_RELATIVE = Path("artifacts/nanobeir_p12_completecase_acquisition_v1")
RESULT_RELATIVE = Path(
    "manifests/nanobeir_p12_completecase_acquisition_result_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/nanobeir_p12_completecase_acquisition_freeze_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/nanobeir_p12_completecase_acquisition_v1.py"
)
TEST_RELATIVE = Path(
    "tests/test_nanobeir_p12_completecase_acquisition_v1.py"
)
DEPENDENCY_RELATIVES = (
    availability.IMPLEMENTATION_RELATIVE,
    Path("assumption_agent/benchmarks/nanobeir_p12_acquisition_v1.py"),
    Path("assumption_agent/benchmarks/fiqa_bridge_expansion_train_runtime_v1.py"),
    Path("replication_runtime/bright_official_hipporag_v1/contract.py"),
)

PRECONDITIONS = {
    "candidate": availability.PRECONDITIONS["candidate"],
    "design": availability.PRECONDITIONS["design"],
    "source_access": availability.PRECONDITIONS["source_access"],
}

QREL_FILES = {
    "qrels/NanoArguAna-00000-of-00001.parquet": (
        "bb469a9580d0872c54d9f8ff118243f1b77abfe8809a033ce5817ecf25bd9370"
    ),
    "qrels/NanoFEVER-00000-of-00001.parquet": (
        "32758317a399a6569b1325bbbf2259c397bb43b5fe4aa59e9993f52a3d071902"
    ),
    "qrels/NanoSciFact-00000-of-00001.parquet": (
        "0bfe8b38bac22fafab42cc3d9f0161ff57e50cd13c49757ade474e9acc675f55"
    ),
}
SOURCE_FILES = availability.SOURCE_FILES
DOCUMENT_PROJECTION_CHARACTERS = availability.DOCUMENT_PROJECTION_CHARACTERS

canonical_json_bytes = utilities.canonical_json_bytes
stable_hash = utilities.stable_hash
file_sha256 = utilities.file_sha256
self_hashed = utilities.self_hashed
_write_exclusive = utilities._write_exclusive
_write_json = utilities._write_json


class CompleteCaseAcquisitionError(RuntimeError):
    """The frozen complete-case acquisition failed closed."""


class OneShotRefusal(CompleteCaseAcquisitionError):
    """The acquisition root or public result is already consumed."""


# Compatibility names used by the mature P11/P12 runtime controller.
NanoBEIRP12AcquisitionError = CompleteCaseAcquisitionError
NanoBEIRAcquisitionError = CompleteCaseAcquisitionError


def project_document(text: object) -> str:
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise CompleteCaseAcquisitionError("document text is invalid")
    return text[:DOCUMENT_PROJECTION_CHARACTERS]


def verify_self_hash(value: Mapping[str, Any], expected: str) -> None:
    _verify_self(value, expected, "manifest")


_verify_self_hash = verify_self_hash


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompleteCaseAcquisitionError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompleteCaseAcquisitionError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise CompleteCaseAcquisitionError(f"{name} is not an object")
    return value


def _verify_self(value: Mapping[str, Any], expected: str, name: str) -> None:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if declared != expected or utilities.stable_hash(body) != expected:
        raise CompleteCaseAcquisitionError(f"{name} self hash drifted")


def _verify_preconditions(base: Path) -> Mapping[str, Any]:
    loaded: dict[str, Any] = {}
    for name, binding in PRECONDITIONS.items():
        path = base / binding["relative"]
        if utilities.file_sha256(path) != binding["file_sha256"]:
            raise CompleteCaseAcquisitionError(f"{name} manifest file drifted")
        value = _read_json(path, name)
        _verify_self(value, binding["self_sha256"], name)
        loaded[name] = value
    for relative, expected in {
        **availability.SOURCE_FILES,
        **QREL_FILES,
    }.items():
        path = base / SOURCE_ROOT_RELATIVE / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or utilities.file_sha256(path) != expected
        ):
            raise CompleteCaseAcquisitionError("pinned source file drifted")
    return loaded


def _git_head(project_root: Path) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_is_ancestor(commit: str, project_root: Path) -> bool:
    import subprocess

    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=project_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _verify_freeze(base: Path, project_root: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "acquisition freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise CompleteCaseAcquisitionError("acquisition freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise CompleteCaseAcquisitionError("acquisition freeze hash is absent")
    _verify_self(value, declared, "acquisition freeze")
    bindings = value.get("implementation_bindings")
    if not isinstance(bindings, list):
        raise CompleteCaseAcquisitionError("implementation bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in bindings
        if isinstance(row, Mapping)
    }
    required = {IMPLEMENTATION_RELATIVE.as_posix(), TEST_RELATIVE.as_posix()}
    if set(observed) != required:
        raise CompleteCaseAcquisitionError("implementation set drifted")
    for relative, expected in observed.items():
        if (
            not isinstance(expected, str)
            or utilities.file_sha256(base / str(relative)) != expected
        ):
            raise CompleteCaseAcquisitionError("implementation file drifted")
    dependencies = value.get("dependency_bindings")
    if not isinstance(dependencies, list):
        raise CompleteCaseAcquisitionError("dependency bindings are absent")
    dependency_observed = {
        row.get("relative_path"): row.get("sha256")
        for row in dependencies
        if isinstance(row, Mapping)
    }
    dependency_required = {path.as_posix() for path in DEPENDENCY_RELATIVES}
    if set(dependency_observed) != dependency_required:
        raise CompleteCaseAcquisitionError("dependency set drifted")
    for relative, expected in dependency_observed.items():
        if (
            not isinstance(expected, str)
            or utilities.file_sha256(base / str(relative)) != expected
        ):
            raise CompleteCaseAcquisitionError("dependency file drifted")
    commit = value.get("formal_implementation_commit")
    if not isinstance(commit, str) or not _git_is_ancestor(commit, project_root):
        raise CompleteCaseAcquisitionError("formal implementation commit drifted")
    return value


def _load_availability(
    base: Path, freeze: Mapping[str, Any]
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    binding = freeze.get("availability_result_binding")
    if not isinstance(binding, Mapping):
        raise CompleteCaseAcquisitionError("availability result binding is absent")
    path = base / AVAILABILITY_RESULT_RELATIVE
    if utilities.file_sha256(path) != binding.get("file_sha256"):
        raise CompleteCaseAcquisitionError("availability result file drifted")
    result = _read_json(path, "availability result")
    expected_self = binding.get("self_sha256")
    if not isinstance(expected_self, str):
        raise CompleteCaseAcquisitionError("availability result hash is absent")
    _verify_self(result, expected_self, "availability result")
    if (
        result.get("schema") != availability.SCHEMA
        or result.get("eligibility_passed") is not True
        or result.get("status")
        != "passed_complete_case_eligible_set_ready_for_private_HMAC_acquisition"
    ):
        raise CompleteCaseAcquisitionError("availability screen did not pass")
    pack_binding = result.get("private_pack_binding")
    if not isinstance(pack_binding, Mapping):
        raise CompleteCaseAcquisitionError("availability pack binding is absent")
    relative = pack_binding.get("relative_path")
    if not isinstance(relative, str):
        raise CompleteCaseAcquisitionError("availability pack path drifted")
    pack_path = base / relative
    if (
        pack_path.is_symlink()
        or not pack_path.is_file()
        or pack_path.stat().st_size != pack_binding.get("size_bytes")
        or utilities.file_sha256(pack_path) != pack_binding.get("file_sha256")
    ):
        raise CompleteCaseAcquisitionError("availability pack file drifted")
    pack = _read_json(pack_path, "availability pack")
    body = dict(pack)
    observed = body.pop("pack_sha256", None)
    if (
        observed != pack_binding.get("pack_sha256")
        or utilities.stable_hash(body) != observed
        or pack.get("schema") != availability.PACK_SCHEMA
        or pack.get("label_or_qrel_open_count") != 0
        or pack.get("candidate_action_count") != 0
    ):
        raise CompleteCaseAcquisitionError("availability pack drifted")
    rows = pack.get("items")
    if not isinstance(rows, list) or len(rows) != availability.ITEM_COUNT:
        raise CompleteCaseAcquisitionError("availability item set drifted")
    return result, tuple(rows)


def hmac_order(
    secret: bytes, family: str, query_ids: Sequence[str]
) -> tuple[str, ...]:
    if len(secret) != 32 or family not in FAMILIES:
        raise CompleteCaseAcquisitionError("HMAC ordering input drifted")
    values = tuple(query_ids)
    if len(values) != len(set(values)):
        raise CompleteCaseAcquisitionError("HMAC query IDs are duplicated")
    return tuple(
        sorted(
            values,
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
    secret: bytes, family: str, eligible_query_ids: Sequence[str]
) -> Mapping[str, tuple[str, ...]]:
    ordered = hmac_order(secret, family, eligible_query_ids)
    if len(ordered) < SELECTED_PER_FAMILY:
        raise CompleteCaseAcquisitionError("eligible family capacity is below 36")
    selected = ordered[:SELECTED_PER_FAMILY]
    output: dict[str, tuple[str, ...]] = {}
    offset = 0
    for block, count in BLOCK_COUNTS:
        output[block] = selected[offset : offset + count]
        offset += count
    output["RESERVE"] = selected[offset : offset + RESERVE_COUNT]
    if offset + RESERVE_COUNT != SELECTED_PER_FAMILY:
        raise CompleteCaseAcquisitionError("block allocation drifted")
    return output


def _item_key(secret: bytes, family: str, query_id: str) -> str:
    digest = hmac.new(
        secret,
        ("item\n" + family + "\n" + query_id).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hashlib.sha256(
        (family + "\n" + query_id + "\n" + digest).encode("utf-8")
    ).hexdigest()


def _eligible_rows(
    base: Path, rows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    output: dict[str, dict[str, Mapping[str, Any]]] = {
        family: {} for family in FAMILIES
    }
    seen_ordinals: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise CompleteCaseAcquisitionError("availability row is invalid")
        family = row.get("family")
        query_id = row.get("query_id")
        ordinal = row.get("ordinal")
        if (
            family not in FAMILIES
            or not isinstance(query_id, str)
            or not query_id
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not 0 <= ordinal < availability.ITEM_COUNT
            or ordinal in seen_ordinals
        ):
            raise CompleteCaseAcquisitionError("availability identity drifted")
        seen_ordinals.add(ordinal)
        if row.get("availability") != "terminal":
            continue
        required = {
            "base_pool",
            "raw_top10",
            "HippoRAG_top_rows",
            "HippoRAG_output_file_sha256",
            "graph_node_count",
            "graph_edge_count",
            "query",
            "stderr_sha256",
            "stdout_sha256",
        }
        if any(name not in row for name in required):
            raise CompleteCaseAcquisitionError("terminal availability row drifted")
        base_pool = row["base_pool"]
        raw_top10 = row["raw_top10"]
        top_rows = row["HippoRAG_top_rows"]
        if (
            not isinstance(base_pool, list)
            or len(base_pool) != 32
            or len(set(base_pool)) != 32
            or not isinstance(raw_top10, list)
            or len(raw_top10) != 10
            or raw_top10 != base_pool[:10]
            or not isinstance(top_rows, list)
            or len(top_rows) != 10
            or len(set(top_rows)) != 10
            or any(value not in base_pool for value in top_rows)
            or not isinstance(row["stderr_sha256"], str)
            or not isinstance(row["stdout_sha256"], str)
        ):
            raise CompleteCaseAcquisitionError("terminal retrieval rows drifted")
        output_path = (
            base
            / AVAILABILITY_ROOT_RELATIVE
            / "hipporag"
            / f"item_{ordinal:03d}"
            / "output.json"
        )
        expected_sha = row["HippoRAG_output_file_sha256"]
        if (
            not isinstance(expected_sha, str)
            or utilities.file_sha256(output_path) != expected_sha
        ):
            raise CompleteCaseAcquisitionError("screen HippoRAG output drifted")
        try:
            payload = train.hippo_contract.parse_output(output_path.read_bytes())
        except Exception as exc:
            raise CompleteCaseAcquisitionError(
                "screen HippoRAG output is invalid"
            ) from exc
        derived = [base_pool[index] for index in payload["top_ordinals"]]
        if (
            derived != top_rows
            or payload["graph_node_count"] != row["graph_node_count"]
            or payload["graph_edge_count"] != row["graph_edge_count"]
        ):
            raise CompleteCaseAcquisitionError("screen HippoRAG receipt drifted")
        if query_id in output[family]:
            raise CompleteCaseAcquisitionError("eligible query ID duplicated")
        output[family][query_id] = row
    if seen_ordinals != set(range(availability.ITEM_COUNT)):
        raise CompleteCaseAcquisitionError("availability ordinal set drifted")
    if any(len(output[family]) < SELECTED_PER_FAMILY for family in FAMILIES):
        raise CompleteCaseAcquisitionError("eligible family capacity is below 36")
    return output


def _read_qrels(base: Path) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise CompleteCaseAcquisitionError("pyarrow is unavailable") from exc
    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for family in FAMILIES:
        path = (
            base
            / SOURCE_ROOT_RELATIVE
            / "qrels"
            / f"{family}-00000-of-00001.parquet"
        )
        table = pq.read_table(path)
        if table.column_names != ["query-id", "corpus-id"]:
            raise CompleteCaseAcquisitionError("qrel schema drifted")
        grouped: dict[str, list[str]] = {}
        seen: set[tuple[str, str]] = set()
        for row in table.to_pylist():
            query_id = row.get("query-id")
            document_id = row.get("corpus-id")
            pair = (query_id, document_id)
            if (
                not isinstance(query_id, str)
                or not query_id
                or not isinstance(document_id, str)
                or not document_id
                or pair in seen
            ):
                raise CompleteCaseAcquisitionError("qrel row drifted")
            seen.add(pair)
            grouped.setdefault(query_id, []).append(document_id)
        output[family] = {
            query_id: tuple(sorted(document_ids))
            for query_id, document_ids in grouped.items()
        }
    return output


def _write_pack(
    *,
    base: Path,
    private: Path,
    block: str,
    suffix: str,
    schema: str,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    pack = utilities.self_hashed(
        {"block": block, "items": list(rows), "schema": schema},
        field="pack_sha256",
    )
    path = private / f"{block}.{suffix}.json"
    utilities._write_json(path, pack)
    return {
        "file_sha256": utilities.file_sha256(path),
        "item_count": len(rows),
        "pack_sha256": pack["pack_sha256"],
        "relative_path": path.relative_to(base).as_posix(),
        "size_bytes": path.stat().st_size,
    }


def run_formal(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    root = base / RUN_ROOT_RELATIVE
    result_path = base / RESULT_RELATIVE
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("acquisition root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("acquisition result already exists")
    preconditions = _verify_preconditions(base)
    freeze = _verify_freeze(base, project_root)
    availability_result, screen_rows = _load_availability(base, freeze)
    eligible = _eligible_rows(base, screen_rows)
    qrels = _read_qrels(base)

    root.mkdir(mode=0o700)
    private = root / "private"
    private.mkdir(mode=0o700)
    secret = os.urandom(32)
    utilities._write_exclusive(private / "selection.secret", secret, mode=0o600)
    marker = {
        "availability_result_self_sha256": availability_result["self_sha256"],
        "candidate_freeze_self_sha256": PRECONDITIONS["candidate"]["self_sha256"],
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
        "selection_secret_sha256": hashlib.sha256(secret).hexdigest(),
        "study_design_self_sha256": PRECONDITIONS["design"]["self_sha256"],
    }
    marker_path = root / "attempt.marker"
    utilities._write_json(marker_path, marker)

    views: dict[str, list[dict[str, Any]]] = {
        block: [] for block, _count in BLOCK_COUNTS
    }
    hippo: dict[str, list[dict[str, Any]]] = {
        block: [] for block, _count in BLOCK_COUNTS
    }
    labels: dict[str, list[dict[str, Any]]] = {
        block: [] for block in LABEL_BLOCKS
    }
    commitments: dict[str, dict[str, list[str]]] = {}
    for family in FAMILIES:
        allocated = allocate_blocks(secret, family, tuple(eligible[family]))
        commitments[family] = {}
        for block, query_ids in allocated.items():
            commitments[family][block] = [
                _item_key(secret, family, query_id) for query_id in query_ids
            ]
            if block == "RESERVE":
                continue
            for family_ordinal, query_id in enumerate(query_ids):
                row = eligible[family][query_id]
                key = _item_key(secret, family, query_id)
                views[block].append(
                    {
                        "family": family,
                        "family_ordinal": family_ordinal,
                        "item_key": key,
                        "query": row["query"],
                        "source_query_id": query_id,
                    }
                )
                source_ordinal = int(row["ordinal"])
                source_path = (
                    AVAILABILITY_ROOT_RELATIVE
                    / "hipporag"
                    / f"item_{source_ordinal:03d}"
                    / "output.json"
                )
                hippo[block].append(
                    {
                        "base_pool": list(row["base_pool"]),
                        "family": family,
                        "graph_edge_count": row["graph_edge_count"],
                        "graph_node_count": row["graph_node_count"],
                        "item_key": key,
                        "raw_top10": list(row["raw_top10"]),
                        "source_output_file_sha256": row[
                            "HippoRAG_output_file_sha256"
                        ],
                        "source_output_relative_path": source_path.as_posix(),
                        "source_stderr_sha256": row["stderr_sha256"],
                        "source_stdout_sha256": row["stdout_sha256"],
                        "source_screen_ordinal": source_ordinal,
                        "top_rows": list(row["HippoRAG_top_rows"]),
                    }
                )
                if block in LABEL_BLOCKS:
                    gold = qrels[family].get(query_id)
                    if not gold:
                        raise CompleteCaseAcquisitionError(
                            "selected query has no positive qrel"
                        )
                    labels[block].append(
                        {
                            "family": family,
                            "gold_document_ids": list(gold),
                            "item_key": key,
                        }
                    )

    pack_bindings: dict[str, Any] = {}
    for block, _count in BLOCK_COUNTS:
        pack_bindings[f"{block}_view"] = _write_pack(
            base=base,
            private=private,
            block=block,
            suffix="view",
            schema=VIEW_SCHEMA,
            rows=views[block],
        )
        pack_bindings[f"{block}_hipporag"] = _write_pack(
            base=base,
            private=private,
            block=block,
            suffix="hipporag",
            schema=HIPPO_SCHEMA,
            rows=hippo[block],
        )
        if block in LABEL_BLOCKS:
            pack_bindings[f"{block}_labels"] = _write_pack(
                base=base,
                private=private,
                block=block,
                suffix="labels",
                schema=LABEL_SCHEMA,
                rows=labels[block],
            )

    result = utilities.self_hashed(
        {
            "allocation": {
                "block_family_counts": {
                    block: {family: count for family in FAMILIES}
                    for block, count in BLOCK_COUNTS
                },
                "commitment_set_sha256": utilities.stable_hash(commitments),
                "eligible_counts_by_family": {
                    family: len(eligible[family]) for family in FAMILIES
                },
                "reserve_count_per_family": RESERVE_COUNT,
                "selected_count_per_family": SELECTED_PER_FAMILY,
                "total_runtime_item_count": sum(
                    len(rows) for rows in views.values()
                ),
            },
            "attempt_binding": {
                "attempt_marker_sha256": utilities.file_sha256(marker_path),
                "formal_execution_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "selection_secret_sha256": hashlib.sha256(secret).hexdigest(),
            },
            "availability_binding": {
                "availability_result_self_sha256": availability_result[
                    "self_sha256"
                ],
                "selected_HippoRAG_outputs_byte_reused": True,
                "HippoRAG_relaunch_count": 0,
            },
            "claim_boundary": {
                "candidate_action_count": 0,
                "external_network_call_count": 0,
                "individual_item_or_label_value_published": False,
                "performance_score_count": 0,
            },
            "pack_bindings": pack_bindings,
            "recorded_date": "2026-07-21",
            "schema": SCHEMA,
            "source_access_self_sha256": preconditions["source_access"][
                "self_sha256"
            ],
            "status": (
                "passed_99_item_private_acquisition_ready_for_"
                "P12_completecase_C_confirm_runtime"
            ),
            "study_design_self_sha256": preconditions["design"]["self_sha256"],
        }
    )
    utilities._write_exclusive(
        result_path, utilities.canonical_json_bytes(result), mode=0o644
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.formal:
        raise SystemExit("--formal is required")
    result = run_formal(arguments.project_root)
    print(
        json.dumps(
            {
                "self_sha256": result["self_sha256"],
                "status": result["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
