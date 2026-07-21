"""One-shot view-only acquisition for the fresh P15 HMAC extension window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_acquisition_v1 as p14_acquisition,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_source_qualification_v1 as source,
)


SCHEMA = "bright_p15_extension_acquisition_result_v1"
ATTEMPT_SCHEMA = "bright_p15_extension_acquisition_attempt_v1"
FREEZE_SCHEMA = "bright_p15_extension_acquisition_freeze_v1"
VIEW_SCHEMA = "bright_p15_extension_private_view_v1"
FAMILIES = source.FAMILIES
POSITION_START = 72
POSITION_END = 92
ATTEMPTS_PER_FAMILY = POSITION_END - POSITION_START
ATTEMPT_COUNT = ATTEMPTS_PER_FAMILY * len(FAMILIES)

DESIGN_RELATIVE = Path(
    "manifests/bright_p15_all_remote_c_confirm_study_design_v1.json"
)
DESIGN_FILE_SHA256 = (
    "52dc2dc60fb3c1ac22ba29f37c2ef1270f06217e37300fdf389babea263de804"
)
DESIGN_SELF_SHA256 = (
    "2ca2a335fcc669bda4f715afbe95b664040240f6808ebf65ca1633b0ce9e6011"
)
P14_RESULT_RELATIVE = p14_acquisition.RESULT_RELATIVE
P14_RESULT_FILE_SHA256 = (
    "b9528887a80d5fb93b0b2840555b038f6b83907804f9e15f216613d68b5465d7"
)
P14_RESULT_SELF_SHA256 = (
    "062b1a2636ec7e756acb798264623375661d2e701c10a8fbac04df7b7f82b9e7"
)
SELECTION_SECRET_RELATIVE = Path(
    "artifacts/bright_p14_acquisition_v1/private/selection.secret"
)
SELECTION_SECRET_SHA256 = (
    "e81ccaf84a48171388b3c69d12604bff962a75e58355727875443ae3009bf82f"
)

RUN_ROOT_RELATIVE = Path("artifacts/bright_p15_extension_acquisition_v1")
PRIVATE_RELATIVE = RUN_ROOT_RELATIVE / "private"
RESULT_RELATIVE = Path(
    "manifests/bright_p15_extension_acquisition_result_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/bright_p15_extension_acquisition_freeze_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/bright_p15_extension_acquisition_v1.py"
)
TEST_RELATIVE = Path("tests/test_bright_p15_extension_acquisition_v1.py")


class P15AcquisitionError(RuntimeError):
    """The frozen P15 extension acquisition failed closed."""


class OneShotRefusal(P15AcquisitionError):
    """The P15 extension acquisition root or result is consumed."""


@dataclass(frozen=True)
class ViewSourceItem:
    item_id: str
    query: str
    excluded_ids: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeItem:
    ordinal: int
    family: str
    attempt_ordinal: int
    family_hmac_position: int
    item_key: str
    query: str
    source_query_id: str
    excluded_ids: tuple[str, ...]


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    return p14_acquisition._read_json(path, name)


def _verify_self(value: Mapping[str, Any], expected: str, name: str) -> None:
    p14_acquisition._verify_self(value, expected, name)


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_is_ancestor(commit: str, project_root: Path) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=project_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _verify_design_and_P14(base: Path) -> Mapping[str, Any]:
    design_path = base / DESIGN_RELATIVE
    if p14_acquisition.utilities.file_sha256(design_path) != DESIGN_FILE_SHA256:
        raise P15AcquisitionError("P15 design file drifted")
    design = _read_json(design_path, "P15 design")
    _verify_self(design, DESIGN_SELF_SHA256, "P15 design")
    p14_path = base / P14_RESULT_RELATIVE
    if p14_acquisition.utilities.file_sha256(p14_path) != P14_RESULT_FILE_SHA256:
        raise P15AcquisitionError("P14 acquisition result file drifted")
    result = _read_json(p14_path, "P14 acquisition result")
    _verify_self(result, P14_RESULT_SELF_SHA256, "P14 acquisition result")
    binding = result.get("attempt_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("selection_secret_sha256") != SELECTION_SECRET_SHA256
        or result.get("candidate_name")
        != "P13_P12_BRIDGE_SAFE_TYPED_QUERY_V1"
    ):
        raise P15AcquisitionError("P14 acquisition binding drifted")
    return result


def _verify_freeze(base: Path, project_root: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "P15 acquisition freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise P15AcquisitionError("P15 acquisition freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise P15AcquisitionError("P15 acquisition freeze hash is absent")
    _verify_self(value, declared, "P15 acquisition freeze")
    rows = value.get("implementation_bindings")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows if isinstance(row, Mapping)
    } if isinstance(rows, list) else {}
    required = {IMPLEMENTATION_RELATIVE.as_posix(), TEST_RELATIVE.as_posix()}
    if set(observed) != required:
        raise P15AcquisitionError("P15 acquisition implementation set drifted")
    for relative, expected in observed.items():
        if p14_acquisition.utilities.file_sha256(base / relative) != expected:
            raise P15AcquisitionError("P15 acquisition implementation drifted")
    commit = value.get("formal_implementation_commit")
    if not isinstance(commit, str) or not _git_is_ancestor(commit, project_root):
        raise P15AcquisitionError("P15 acquisition commit drifted")
    if value.get("study_design_self_sha256") != DESIGN_SELF_SHA256:
        raise P15AcquisitionError("P15 acquisition design binding drifted")
    return value


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise P15AcquisitionError(f"{name} is invalid")
    return value


def _text_list(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise P15AcquisitionError(f"{name} is invalid")
    output = tuple(_required_text(item, name) for item in value)
    if len(output) != len(set(output)):
        raise P15AcquisitionError(f"{name} is duplicated")
    return output


def load_view_sources(base: Path) -> Mapping[str, tuple[ViewSourceItem, ...]]:
    """Load only assignment fields; the gold_ids parquet column is not read."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise P15AcquisitionError("pyarrow is unavailable") from exc
    root = base / source.SOURCE_ROOT_RELATIVE / "examples"
    output: dict[str, tuple[ViewSourceItem, ...]] = {}
    for family in FAMILIES:
        slug = source.SLUGS[family]
        relative = f"examples/{slug}-00000-of-00001.parquet"
        path = root / f"{slug}-00000-of-00001.parquet"
        binding = source.SOURCE_FILES[relative]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != binding["size_bytes"]
            or p14_acquisition.utilities.file_sha256(path) != binding["sha256"]
        ):
            raise P15AcquisitionError("P15 example source drifted")
        table = pq.read_table(path, columns=["query", "id", "excluded_ids"])
        if table.column_names != ["query", "id", "excluded_ids"]:
            raise P15AcquisitionError("P15 view source schema drifted")
        rows: list[ViewSourceItem] = []
        seen_ids: set[str] = set()
        seen_queries: set[str] = set()
        for raw in table.to_pylist():
            item_id = _required_text(raw.get("id"), "example ID")
            query = _required_text(raw.get("query"), "query")
            excluded = _text_list(raw.get("excluded_ids"), "excluded ID")
            if item_id in seen_ids or query in seen_queries:
                raise P15AcquisitionError("P15 view source identity is duplicated")
            seen_ids.add(item_id)
            seen_queries.add(query)
            rows.append(ViewSourceItem(item_id, query, excluded))
        if len(rows) < POSITION_END:
            raise P15AcquisitionError("family capacity is below position 92")
        output[family] = tuple(rows)
    return output


def select_extension(
    secret: bytes, family: str, query_ids: Sequence[str]
) -> tuple[str, ...]:
    ordered = p14_acquisition.hmac_order(secret, family, query_ids)
    if len(ordered) < POSITION_END:
        raise P15AcquisitionError("family capacity is below position 92")
    return ordered[POSITION_START:POSITION_END]


def _write_view_pack(base: Path, private: Path, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    pack = p14_acquisition.utilities.self_hashed(
        {"block": "C_confirm", "items": list(rows), "schema": VIEW_SCHEMA},
        field="pack_sha256",
    )
    path = private / "C_confirm.view.json"
    p14_acquisition.utilities._write_json(path, pack)
    return {
        "file_sha256": p14_acquisition.utilities.file_sha256(path),
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
        raise OneShotRefusal("P15 acquisition root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("P15 acquisition result already exists")
    _verify_design_and_P14(base)
    freeze = _verify_freeze(base, project_root)
    root.mkdir(mode=0o700)
    private = base / PRIVATE_RELATIVE
    private.mkdir(mode=0o700)
    marker = p14_acquisition.utilities.self_hashed(
        {
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "P14_selection_secret_sha256": SELECTION_SECRET_SHA256,
            "schema": ATTEMPT_SCHEMA,
            "study_design_self_sha256": DESIGN_SELF_SHA256,
        },
        field="attempt_sha256",
    )
    marker_path = root / "attempt.marker"
    p14_acquisition.utilities._write_json(marker_path, marker)
    secret_path = base / SELECTION_SECRET_RELATIVE
    if (
        secret_path.is_symlink()
        or not secret_path.is_file()
        or secret_path.stat().st_size != 32
        or p14_acquisition.utilities.file_sha256(secret_path)
        != SELECTION_SECRET_SHA256
    ):
        raise P15AcquisitionError("P14 selection secret drifted")
    secret = secret_path.read_bytes()
    sources = load_view_sources(base)
    rows: list[dict[str, Any]] = []
    commitments: dict[str, list[str]] = {}
    for family in FAMILIES:
        examples = {item.item_id: item for item in sources[family]}
        query_ids = select_extension(secret, family, tuple(examples))
        commitments[family] = []
        for attempt_ordinal, query_id in enumerate(query_ids):
            item = examples[query_id]
            key = p14_acquisition._item_key(secret, family, query_id)
            commitments[family].append(key)
            rows.append(
                {
                    "attempt_ordinal": attempt_ordinal,
                    "excluded_document_ids": list(item.excluded_ids),
                    "family": family,
                    "family_HMAC_position": POSITION_START + attempt_ordinal,
                    "item_key": key,
                    "query": item.query,
                    "source_query_id": query_id,
                }
            )
    if len(rows) != ATTEMPT_COUNT:
        raise P15AcquisitionError("P15 assignment count drifted")
    view_binding = _write_view_pack(base, private, rows)
    result = p14_acquisition.utilities.self_hashed(
        {
            "allocation": {
                "attempt_count": ATTEMPT_COUNT,
                "attempt_count_per_family": ATTEMPTS_PER_FAMILY,
                "commitment_set_sha256": p14_acquisition.utilities.stable_hash(commitments),
                "HMAC_end_exclusive": POSITION_END,
                "HMAC_start": POSITION_START,
                "target_terminal_count_per_family": 10,
            },
            "attempt_binding": {
                "attempt_marker_sha256": p14_acquisition.utilities.file_sha256(marker_path),
                "formal_execution_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "selection_secret_sha256": SELECTION_SECRET_SHA256,
            },
            "candidate_name": "P13_P12_BRIDGE_SAFE_TYPED_QUERY_V1",
            "claim_boundary": {
                "action_model_comparator_or_score_count": 0,
                "gold_ID_column_read_count": 0,
                "label_pack_created": False,
                "prior_P14_item_action_or_score_reuse_count": 0,
                "view_item_count": ATTEMPT_COUNT,
            },
            "pack_bindings": {"C_confirm_view": view_binding},
            "recorded_date": "2026-07-22",
            "schema": SCHEMA,
            "source_qualification_self_sha256": "0f007f4d4159a37784150bfa8025c23375e69d6904528365924e3a033d71dd00",
            "status": "passed_view_only_ready_for_P15_all_remote_action",
            "study_design_self_sha256": DESIGN_SELF_SHA256,
        }
    )
    p14_acquisition.utilities._write_exclusive(
        result_path,
        p14_acquisition.utilities.canonical_json_bytes(result),
        mode=0o644,
    )
    return result


def load_views(base: Path, result: Mapping[str, Any]) -> tuple[RuntimeItem, ...]:
    bindings = result.get("pack_bindings")
    binding = bindings.get("C_confirm_view") if isinstance(bindings, Mapping) else None
    if not isinstance(binding, Mapping) or binding.get("item_count") != ATTEMPT_COUNT:
        raise P15AcquisitionError("P15 view binding drifted")
    path = base / str(binding.get("relative_path"))
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != binding.get("size_bytes")
        or p14_acquisition.utilities.file_sha256(path) != binding.get("file_sha256")
    ):
        raise P15AcquisitionError("P15 view file drifted")
    pack = _read_json(path, "P15 view")
    body = dict(pack)
    declared = body.pop("pack_sha256", None)
    if (
        pack.get("schema") != VIEW_SCHEMA
        or pack.get("block") != "C_confirm"
        or declared != binding.get("pack_sha256")
        or p14_acquisition.utilities.stable_hash(body) != declared
    ):
        raise P15AcquisitionError("P15 view envelope drifted")
    raw_rows = pack.get("items")
    if not isinstance(raw_rows, list) or len(raw_rows) != ATTEMPT_COUNT:
        raise P15AcquisitionError("P15 view rows drifted")
    output: list[RuntimeItem] = []
    counts = {family: 0 for family in FAMILIES}
    for ordinal, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise P15AcquisitionError("P15 view row drifted")
        family = row.get("family")
        attempt = row.get("attempt_ordinal")
        position = row.get("family_HMAC_position")
        if (
            family not in FAMILIES
            or attempt != counts[family]
            or position != POSITION_START + attempt
        ):
            raise P15AcquisitionError("P15 view order drifted")
        excluded = row.get("excluded_document_ids")
        if not isinstance(excluded, list) or len(excluded) != len(set(excluded)):
            raise P15AcquisitionError("P15 exclusions drifted")
        texts = (row.get("item_key"), row.get("query"), row.get("source_query_id"))
        if not all(isinstance(value, str) and value for value in texts):
            raise P15AcquisitionError("P15 view text drifted")
        output.append(
            RuntimeItem(
                ordinal=ordinal,
                family=family,
                attempt_ordinal=attempt,
                family_hmac_position=position,
                item_key=texts[0],
                query=texts[1],
                source_query_id=texts[2],
                excluded_ids=tuple(excluded),
            )
        )
        counts[family] += 1
    if any(value != ATTEMPTS_PER_FAMILY for value in counts.values()):
        raise P15AcquisitionError("P15 family count drifted")
    return tuple(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.formal:
        raise SystemExit("--formal is required")
    result = run_formal(arguments.project_root)
    print(json.dumps({"self_sha256": result["self_sha256"], "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
