"""Result-blind in-place completion of the interrupted BRIGHT reserve acquisition."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from assumption_agent.benchmarks import bright_reasoning_retrieval_acquisition_v1 as source
from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_acquisition_v1 as acquisition
from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_entrypoint_v1 as entrypoint


VERSION = "bright_reasoning_retrieval_reserve_acquisition_recovery_v1"
RECOVERY_FREEZE_SCHEMA = f"{VERSION}_implementation_freeze"
RECOVERY_DESIGN_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_reserve_acquisition_recovery_design_v1.json"
)
RECOVERY_FREEZE_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_reserve_acquisition_recovery_implementation_freeze_v1.json"
)
FAILURE_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_reserve_acquisition_failure_v1.json"
)

RECOVERY_DESIGN_FILE_SHA256 = (
    "001f6d8c74c0b31d5c8c8309907db9aa4c707bb9ca831904b4aaef6820b2eef2"
)
RECOVERY_DESIGN_SELF_SHA256 = (
    "a69d9591195cd0b720249a0f5337e80f88a5a53e9a355a9add6d64b2158581ca"
)
FAILURE_FILE_SHA256 = (
    "242ae6737cd328a7cc386ff2cc42ee70d760539bc2cc70f56adbe6de34fd62f9"
)
FAILURE_SELF_SHA256 = (
    "c67858a8c36d65c7f63a4fd087d5c6dd2d6eab96a9dc82254eb8dcf89fd3b181"
)
ORIGINAL_FREEZE_FILE_SHA256 = (
    "09ea1246be3318c44e58e1292ade60bce5cb1e2369681e58f254432be8e74a44"
)
ORIGINAL_FREEZE_SELF_SHA256 = (
    "095af5481d95adcaf3e77ea465b2bc6f90909648fb1e32562367e324b8f30c1a"
)
ATTEMPT_FILE_SHA256 = (
    "08153ff11f7b74f9360961e17bbec3c4276437564471be2401c20f99889d1c66"
)
VIEW_FILE_SHA256 = (
    "9a80300a90c531edb436bc1f2967bece16ff209af3c34d84a651c18ead11f80d"
)
LABEL_FILE_SHA256 = (
    "fd1c168eccbe9cde96de99bc72df00e09e7b2ba691d0b3c53c65a13800553be7"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class BrightReserveRecoveryError(RuntimeError):
    """The preregistered acquisition recovery failed closed."""


def _verify_self(payload: Mapping[str, Any], field: str, expected: str | None = None) -> str:
    try:
        observed = source.verify_self_hash(payload, field)
    except source.BrightAcquisitionError as exc:
        raise BrightReserveRecoveryError(str(exc)) from exc
    if expected is not None and observed != expected:
        raise BrightReserveRecoveryError(f"{field} binding drifted")
    return observed


def _read_bound_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BrightReserveRecoveryError(f"{field} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightReserveRecoveryError(f"{field} is invalid") from exc
    if not isinstance(value, dict):
        raise BrightReserveRecoveryError(f"{field} root is invalid")
    return value


def _read_canonical_pack(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BrightReserveRecoveryError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightReserveRecoveryError(f"{field} is invalid") from exc
    if (
        not isinstance(value, dict)
        or source.canonical_json_bytes(value) + b"\n" != raw
    ):
        raise BrightReserveRecoveryError(f"{field} is not canonical")
    return value


def _verify_recovery_freeze(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    design_path = project_root / RECOVERY_DESIGN_RELATIVE
    if source.file_sha256(design_path) != RECOVERY_DESIGN_FILE_SHA256:
        raise BrightReserveRecoveryError("recovery design file drifted")
    design = _read_bound_json(design_path, "recovery design")
    _verify_self(design, "self_sha256", RECOVERY_DESIGN_SELF_SHA256)

    failure_path = project_root / FAILURE_RELATIVE
    if source.file_sha256(failure_path) != FAILURE_FILE_SHA256:
        raise BrightReserveRecoveryError("failure receipt file drifted")
    failure = _read_bound_json(failure_path, "failure receipt")
    _verify_self(failure, "self_sha256", FAILURE_SELF_SHA256)
    if failure.get("status") != (
        "terminal_acquisition_attempt_implementation_invalid_recovery_unattempted"
    ):
        raise BrightReserveRecoveryError("failure receipt status drifted")

    original_freeze_path = project_root / acquisition.FREEZE_RELATIVE
    if source.file_sha256(original_freeze_path) != ORIGINAL_FREEZE_FILE_SHA256:
        raise BrightReserveRecoveryError("original implementation freeze file drifted")
    entrypoint._activate()
    original_freeze = acquisition._verify_freeze(project_root)
    if original_freeze.get("self_sha256") != ORIGINAL_FREEZE_SELF_SHA256:
        raise BrightReserveRecoveryError("original implementation freeze drifted")

    recovery_freeze = _read_bound_json(
        project_root / RECOVERY_FREEZE_RELATIVE, "recovery implementation freeze"
    )
    if (
        recovery_freeze.get("schema") != RECOVERY_FREEZE_SCHEMA
        or recovery_freeze.get("design_self_sha256")
        != RECOVERY_DESIGN_SELF_SHA256
    ):
        raise BrightReserveRecoveryError("recovery implementation freeze identity drifted")
    _verify_self(recovery_freeze, "self_sha256")
    bindings = recovery_freeze.get("implementation_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise BrightReserveRecoveryError("recovery implementation bindings are invalid")
    for row in bindings:
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256"}:
            raise BrightReserveRecoveryError("recovery implementation binding row drifted")
        if source.file_sha256(project_root / str(row["relative_path"])) != row["sha256"]:
            raise BrightReserveRecoveryError("recovery implementation file drifted")
    return original_freeze, recovery_freeze


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise BrightReserveRecoveryError(f"{field} is invalid")
    return value


def validate_existing_packs(
    view: Mapping[str, Any], labels: Mapping[str, Any]
) -> dict[str, Any]:
    if set(view) != {
        "block",
        "excluded_fields",
        "item_count",
        "items",
        "pack_sha256",
        "schema",
    }:
        raise BrightReserveRecoveryError("view envelope drifted")
    if (
        view.get("schema") != acquisition.VIEW_SCHEMA
        or view.get("block") != acquisition.BLOCK
        or view.get("item_count") != acquisition.ITEM_COUNT
        or view.get("excluded_fields")
        != [
            "source_example_id",
            "reasoning",
            "gold_ids_long",
            "gold_ids",
            "gold_answer",
        ]
    ):
        raise BrightReserveRecoveryError("view contract drifted")
    view_pack_sha256 = _verify_self(view, "pack_sha256")

    if set(labels) != {
        "block",
        "item_count",
        "items",
        "pack_sha256",
        "schema",
    }:
        raise BrightReserveRecoveryError("label envelope drifted")
    if (
        labels.get("schema") != acquisition.LABEL_SCHEMA
        or labels.get("block") != acquisition.BLOCK
        or labels.get("item_count") != acquisition.ITEM_COUNT
    ):
        raise BrightReserveRecoveryError("label contract drifted")
    label_pack_sha256 = _verify_self(labels, "pack_sha256")

    view_rows = view.get("items")
    label_rows = labels.get("items")
    if (
        not isinstance(view_rows, list)
        or not isinstance(label_rows, list)
        or len(view_rows) != acquisition.ITEM_COUNT
        or len(label_rows) != acquisition.ITEM_COUNT
    ):
        raise BrightReserveRecoveryError("pack row counts drifted")
    families: list[str] = []
    commitments: list[str] = []
    for ordinal, (view_row, label_row) in enumerate(zip(view_rows, label_rows)):
        if not isinstance(view_row, Mapping) or set(view_row) != {
            "excluded_ids",
            "family",
            "item_commitment_sha256",
            "ordinal",
            "query",
        }:
            raise BrightReserveRecoveryError("view row shape drifted")
        if not isinstance(label_row, Mapping) or set(label_row) != {
            "gold_ids",
            "item_commitment_sha256",
            "ordinal",
        }:
            raise BrightReserveRecoveryError("label row shape drifted")
        commitment = view_row.get("item_commitment_sha256")
        family = view_row.get("family")
        excluded = view_row.get("excluded_ids")
        gold = label_row.get("gold_ids")
        if (
            view_row.get("ordinal") != ordinal
            or label_row.get("ordinal") != ordinal
            or not isinstance(commitment, str)
            or _SHA256.fullmatch(commitment) is None
            or label_row.get("item_commitment_sha256") != commitment
            or family not in source.FAMILY_ORDER
            or isinstance(excluded, (str, bytes))
            or not isinstance(excluded, list)
            or isinstance(gold, (str, bytes))
            or not isinstance(gold, list)
            or not gold
        ):
            raise BrightReserveRecoveryError("pack row identity drifted")
        _required_text(view_row.get("query"), "query")
        excluded_ids = tuple(_required_text(value, "excluded ID") for value in excluded)
        gold_ids = tuple(_required_text(value, "gold ID") for value in gold)
        if len(set(excluded_ids)) != len(excluded_ids) or len(set(gold_ids)) != len(gold_ids):
            raise BrightReserveRecoveryError("pack row IDs are duplicated")
        families.append(str(family))
        commitments.append(commitment)
    expected_counts = Counter(
        {family: acquisition.COUNT_PER_FAMILY for family in source.FAMILY_ORDER}
    )
    observed_counts = Counter(families)
    if observed_counts != expected_counts or len(set(commitments)) != len(commitments):
        raise BrightReserveRecoveryError("pack balance or commitments drifted")
    return {
        "family_counts": dict(sorted(observed_counts.items())),
        "label_pack_sha256": label_pack_sha256,
        "view_pack_sha256": view_pack_sha256,
    }


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    original_freeze, recovery_freeze = _verify_recovery_freeze(project_root)
    result_path = project_root / acquisition.RESULT_RELATIVE
    if result_path.exists() or result_path.is_symlink():
        raise BrightReserveRecoveryError("public acquisition result already exists")
    root = project_root / acquisition.ROOT_RELATIVE
    marker_path = root / "attempt.marker"
    view_path = project_root / acquisition.PRIVATE_RELATIVE / f"{acquisition.BLOCK}.view.json"
    label_path = project_root / acquisition.PRIVATE_RELATIVE / f"{acquisition.BLOCK}.labels.json"
    expected_files = {
        marker_path: ATTEMPT_FILE_SHA256,
        view_path: VIEW_FILE_SHA256,
        label_path: LABEL_FILE_SHA256,
    }
    for path, expected in expected_files.items():
        if source.file_sha256(path) != expected:
            raise BrightReserveRecoveryError("frozen recovery artifact drifted")

    marker = _read_canonical_pack(marker_path, "attempt marker")
    if (
        marker.get("schema") != acquisition.ATTEMPT_SCHEMA
        or marker.get("design_self_sha256") != acquisition.DESIGN_SELF_SHA256
        or marker.get("implementation_freeze_self_sha256")
        != ORIGINAL_FREEZE_SELF_SHA256
    ):
        raise BrightReserveRecoveryError("attempt marker identity drifted")
    _verify_self(marker, "attempt_sha256")
    view = _read_canonical_pack(view_path, "frozen view pack")
    labels = _read_canonical_pack(label_path, "frozen label pack")
    validated = validate_existing_packs(view, labels)
    for path, expected in expected_files.items():
        if source.file_sha256(path) != expected:
            raise BrightReserveRecoveryError("recovery artifact changed during validation")

    body = {
        "claim_boundary": {
            "document_content_reasoning_gold_answer_or_gold_ids_long_read": False,
            "model_retrieval_or_score_count": 0,
            "network_call_count": 0,
            "recovery_existing_private_pack_parse_count": 2,
            "selection_used_gold_or_outcome": False,
        },
        "cohort": {
            "family_counts": validated["family_counts"],
            "item_count": acquisition.ITEM_COUNT,
            "label_pack_file_sha256": LABEL_FILE_SHA256,
            "label_pack_sha256": validated["label_pack_sha256"],
            "view_pack_file_sha256": VIEW_FILE_SHA256,
            "view_pack_sha256": validated["view_pack_sha256"],
        },
        "formal_binding": {
            "attempt_marker_file_sha256": ATTEMPT_FILE_SHA256,
            "design_self_sha256": acquisition.DESIGN_SELF_SHA256,
            "failure_receipt_self_sha256": FAILURE_SELF_SHA256,
            "implementation_freeze_self_sha256": original_freeze["self_sha256"],
            "original_acquisition_result_sha256": acquisition.ORIGINAL_RESULT_SHA256,
            "original_RESERVE_view_pack_sha256": acquisition.ORIGINAL_RESERVE_VIEW_SHA256,
            "recovery_design_self_sha256": RECOVERY_DESIGN_SELF_SHA256,
            "recovery_implementation_freeze_self_sha256": recovery_freeze["self_sha256"],
            "selection_secret_sha256": acquisition.ORIGINAL_SELECTION_SECRET_SHA256,
        },
        "schema": acquisition.RESULT_SCHEMA,
        "status": "fresh_RESERVE_R_search_acquired_labels_sealed",
    }
    result = source.self_hashed(body, "result_sha256")
    source._write_json(result_path, result, mode=0o644)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run(arguments.project_root)
    print(
        json.dumps(
            {"result_sha256": result["result_sha256"], "status": result["status"]},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
