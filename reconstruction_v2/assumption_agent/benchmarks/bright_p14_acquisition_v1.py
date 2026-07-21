"""One-shot private HMAC acquisition for the frozen P14 BRIGHT study."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_source_qualification_v1 as source,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_acquisition_v1 as utilities,
)


SCHEMA = "bright_p14_acquisition_result_v1"
ATTEMPT_SCHEMA = "bright_p14_acquisition_attempt_v1"
FREEZE_SCHEMA = "bright_p14_acquisition_freeze_v1"
VIEW_SCHEMA = "bright_p14_private_view_v1"
LABEL_SCHEMA = "bright_p14_private_labels_v1"
FAMILIES = source.FAMILIES
WINDOWS = (
    ("C_confirm", 0, 20, 10),
    ("A_form", 20, 36, 8),
    ("F_search", 36, 46, 5),
    ("A_hold", 46, 56, 5),
    ("M_search", 56, 66, 5),
    ("RESERVE", 66, 72, 3),
)
LABEL_BLOCKS = frozenset(("C_confirm", "A_form", "A_hold", "M_search"))
MAXIMUM_POSITION = 72

RUN_ROOT_RELATIVE = Path("artifacts/bright_p14_acquisition_v1")
RESULT_RELATIVE = Path("manifests/bright_p14_acquisition_result_v1.json")
FREEZE_RELATIVE = Path("manifests/bright_p14_acquisition_freeze_v1.json")
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/bright_p14_acquisition_v1.py"
)
TEST_RELATIVE = Path("tests/test_bright_p14_acquisition_v1.py")
DEPENDENCY_RELATIVES = (
    source.IMPLEMENTATION_RELATIVE,
    Path("assumption_agent/benchmarks/nanobeir_p12_acquisition_v1.py"),
)

PRECONDITIONS = {
    "candidate": {
        "relative": "manifests/nanobeir_p13_candidate_freeze_v1.json",
        "file_sha256": (
            "64482d0e0d4647327da0a74f1e14844854291f73f439928788eeeba7e7c6a1b2"
        ),
        "self_sha256": (
            "17f9865483cd3c4846db8a63c1047f8af6bdaa24b78ece09245f3e568e0457f0"
        ),
    },
    "custody": source.PRECONDITIONS["custody"],
    "design": source.PRECONDITIONS["design"],
    "qualification": {
        "relative": source.RESULT_RELATIVE.as_posix(),
        "file_sha256": (
            "1b4fb0c29d391e543380a491a5022539799c23763b4509617daefa5e0b6ef288"
        ),
        "self_sha256": (
            "0f007f4d4159a37784150bfa8025c23375e69d6904528365924e3a033d71dd00"
        ),
    },
}


class P14AcquisitionError(RuntimeError):
    """The frozen P14 private acquisition failed closed."""


class OneShotRefusal(P14AcquisitionError):
    """The formal P14 acquisition root or public result is consumed."""


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise P14AcquisitionError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P14AcquisitionError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise P14AcquisitionError(f"{name} is not an object")
    return value


def _verify_self(value: Mapping[str, Any], expected: str, name: str) -> None:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if declared != expected or utilities.stable_hash(body) != expected:
        raise P14AcquisitionError(f"{name} self hash drifted")


def _verify_preconditions(base: Path) -> Mapping[str, Any]:
    loaded: dict[str, Any] = {}
    for name, binding in PRECONDITIONS.items():
        path = base / binding["relative"]
        if utilities.file_sha256(path) != binding["file_sha256"]:
            raise P14AcquisitionError(f"{name} file drifted")
        value = _read_json(path, name)
        _verify_self(value, binding["self_sha256"], name)
        loaded[name] = value
    if loaded["qualification"].get("status") != (
        "passed_source_ready_for_private_HMAC_acquisition"
    ):
        raise P14AcquisitionError("source qualification did not pass")
    if loaded["candidate"].get("candidate", {}).get("candidate_name") != (
        "P13_P12_BRIDGE_SAFE_TYPED_QUERY_V1"
    ):
        raise P14AcquisitionError("candidate binding drifted")
    expected_windows = [
        {
            "attempt_count": end - start,
            "block": block,
            "end_exclusive": end,
            "start": start,
            "target_terminal_count": target,
        }
        for block, start, end, target in WINDOWS
    ]
    if loaded["design"].get("frozen_HMAC_windows_per_family") != (
        expected_windows
    ):
        raise P14AcquisitionError("study windows drifted")
    source._verify_preconditions(base)
    return loaded


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


def _verify_freeze(base: Path, project_root: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "acquisition freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise P14AcquisitionError("acquisition freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise P14AcquisitionError("acquisition freeze hash is absent")
    _verify_self(value, declared, "acquisition freeze")

    rows = value.get("implementation_bindings")
    if not isinstance(rows, list):
        raise P14AcquisitionError("implementation bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    required = {IMPLEMENTATION_RELATIVE.as_posix(), TEST_RELATIVE.as_posix()}
    if set(observed) != required:
        raise P14AcquisitionError("implementation set drifted")
    for relative, expected in observed.items():
        if (
            not isinstance(expected, str)
            or utilities.file_sha256(base / str(relative)) != expected
        ):
            raise P14AcquisitionError("implementation file drifted")

    dependencies = value.get("dependency_bindings")
    if not isinstance(dependencies, list):
        raise P14AcquisitionError("dependency bindings are absent")
    dependency_observed = {
        row.get("relative_path"): row.get("sha256")
        for row in dependencies
        if isinstance(row, Mapping)
    }
    dependency_required = {path.as_posix() for path in DEPENDENCY_RELATIVES}
    if set(dependency_observed) != dependency_required:
        raise P14AcquisitionError("dependency set drifted")
    for relative, expected in dependency_observed.items():
        if (
            not isinstance(expected, str)
            or utilities.file_sha256(base / str(relative)) != expected
        ):
            raise P14AcquisitionError("dependency file drifted")

    qualification = value.get("source_qualification_result_binding")
    if not isinstance(qualification, Mapping) or dict(qualification) != {
        "file_sha256": PRECONDITIONS["qualification"]["file_sha256"],
        "self_sha256": PRECONDITIONS["qualification"]["self_sha256"],
    }:
        raise P14AcquisitionError("source qualification binding drifted")
    commit = value.get("formal_implementation_commit")
    if not isinstance(commit, str) or not _git_is_ancestor(
        commit, project_root
    ):
        raise P14AcquisitionError("formal implementation commit drifted")
    return value


def hmac_order(
    secret: bytes, family: str, query_ids: Sequence[str]
) -> tuple[str, ...]:
    if len(secret) != 32 or family not in FAMILIES:
        raise P14AcquisitionError("HMAC ordering input drifted")
    values = tuple(query_ids)
    if len(values) != len(set(values)):
        raise P14AcquisitionError("HMAC query IDs are duplicated")
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


def allocate_windows(
    secret: bytes, family: str, query_ids: Sequence[str]
) -> Mapping[str, tuple[str, ...]]:
    ordered = hmac_order(secret, family, query_ids)
    if len(ordered) < MAXIMUM_POSITION:
        raise P14AcquisitionError("family query capacity is below 72")
    return {
        block: ordered[start:end]
        for block, start, end, _target in WINDOWS
    }


def _item_key(secret: bytes, family: str, query_id: str) -> str:
    digest = hmac.new(
        secret,
        ("item\n" + family + "\n" + query_id).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hashlib.sha256(
        (family + "\n" + query_id + "\n" + digest).encode("utf-8")
    ).hexdigest()


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
        raise OneShotRefusal("P14 acquisition root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("P14 acquisition result already exists")
    preconditions = _verify_preconditions(base)
    freeze = _verify_freeze(base, project_root)

    root.mkdir(mode=0o700)
    private = root / "private"
    private.mkdir(mode=0o700)
    secret = os.urandom(32)
    utilities._write_exclusive(
        private / "selection.secret", secret, mode=0o600
    )
    marker = {
        "candidate_freeze_self_sha256": PRECONDITIONS["candidate"][
            "self_sha256"
        ],
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
        "selection_secret_sha256": hashlib.sha256(secret).hexdigest(),
        "source_qualification_self_sha256": PRECONDITIONS[
            "qualification"
        ]["self_sha256"],
        "study_design_self_sha256": PRECONDITIONS["design"]["self_sha256"],
    }
    marker_path = root / "attempt.marker"
    utilities._write_json(marker_path, marker)

    sources = source.load_sources(base)
    views: dict[str, list[dict[str, Any]]] = {
        block: [] for block, _start, _end, _target in WINDOWS
    }
    labels: dict[str, list[dict[str, Any]]] = {
        block: [] for block in LABEL_BLOCKS
    }
    commitments: dict[str, dict[str, list[str]]] = {}
    for family in FAMILIES:
        examples = {item.item_id: item for item in sources[family].examples}
        allocated = allocate_windows(secret, family, tuple(examples))
        commitments[family] = {}
        for block, start, _end, _target in WINDOWS:
            query_ids = allocated[block]
            commitments[family][block] = [
                _item_key(secret, family, query_id) for query_id in query_ids
            ]
            for attempt_ordinal, query_id in enumerate(query_ids):
                item = examples[query_id]
                key = _item_key(secret, family, query_id)
                views[block].append(
                    {
                        "attempt_ordinal": attempt_ordinal,
                        "excluded_document_ids": list(item.excluded_ids),
                        "family": family,
                        "family_HMAC_position": start + attempt_ordinal,
                        "item_key": key,
                        "query": item.query,
                        "source_query_id": query_id,
                    }
                )
                if block in LABEL_BLOCKS:
                    labels[block].append(
                        {
                            "family": family,
                            "gold_document_ids": list(item.gold_ids),
                            "item_key": key,
                        }
                    )

    pack_bindings: dict[str, Any] = {}
    for block, _start, _end, _target in WINDOWS:
        pack_bindings[f"{block}_view"] = _write_pack(
            base=base,
            private=private,
            block=block,
            suffix="view",
            schema=VIEW_SCHEMA,
            rows=views[block],
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
                "block_family_attempt_counts": {
                    block: {family: end - start for family in FAMILIES}
                    for block, start, end, _target in WINDOWS
                },
                "block_family_target_terminal_counts": {
                    block: {family: target for family in FAMILIES}
                    for block, _start, _end, target in WINDOWS
                },
                "commitment_set_sha256": utilities.stable_hash(commitments),
                "query_counts_by_family": {
                    family: len(sources[family].examples)
                    for family in FAMILIES
                },
                "selected_HMAC_positions_per_family": MAXIMUM_POSITION,
                "total_nonreserve_attempt_count": sum(
                    len(rows)
                    for block, rows in views.items()
                    if block != "RESERVE"
                ),
                "total_view_item_count": sum(
                    len(rows) for rows in views.values()
                ),
            },
            "attempt_binding": {
                "attempt_marker_sha256": utilities.file_sha256(marker_path),
                "formal_execution_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "selection_secret_sha256": hashlib.sha256(secret).hexdigest(),
            },
            "candidate_name": "P13_P12_BRIDGE_SAFE_TYPED_QUERY_V1",
            "claim_boundary": {
                "candidate_action_count": 0,
                "external_network_call_count": 0,
                "gold_labels_read_and_private_sealed": True,
                "individual_item_or_label_value_published": False,
                "performance_score_count": 0,
            },
            "pack_bindings": pack_bindings,
            "recorded_date": "2026-07-21",
            "schema": SCHEMA,
            "source_custody_self_sha256": preconditions["custody"][
                "self_sha256"
            ],
            "source_qualification_self_sha256": preconditions[
                "qualification"
            ]["self_sha256"],
            "status": (
                "passed_private_acquisition_ready_for_P14_direct_C_confirm"
            ),
            "study_design_self_sha256": preconditions["design"][
                "self_sha256"
            ],
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
