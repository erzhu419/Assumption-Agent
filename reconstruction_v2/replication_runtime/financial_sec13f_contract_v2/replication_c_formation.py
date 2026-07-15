from __future__ import annotations

"""Form the preregistered SEC-13F replication-C pack exactly once.

The driver has one fixed study profile.  It constructs one deterministic pack,
checks it against all three committed public commitment views, and persists no
sealed query or answer content in its public receipt.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from replication_runtime.financial_semantic_v2.pack import (
    Sec13FSource,
    build_measurement_view,
    build_public_pack,
    payload_hash,
    read_json,
    sha256_file,
    verify_measurement_view,
    verify_public_pack,
    write_json,
)

from .formation import (
    FreshFormationError,
    assert_no_prior_commitment_collision_v1,
)


RECEIPT_VERSION = (
    "financial_sec13f_contract_v2_replication_c_pack_formation_v1"
)
STUDY_ID = "financial-sec13f-contract-v2-replication-c-2025q4-to-2026q1"
SELECTION_SEED = (
    "assumption-agent-financial-sec13f-contract-v2-replication-c-20260716"
)
PREREGISTRATION_RELATIVE_PATH = (
    "manifests/financial_sec13f_contract_v2_replication_c_preregistration_v1.json"
)
PREREGISTRATION_FILE_SHA256 = (
    "95a1bee00e75864d78d73aa88112629f277637d7e4fae008e2b05e200a462601"
)
PREREGISTRATION_HASH = (
    "2a9bee40bbcda9454712d7b046670591facb7c2bf1de459685019badcc2cd68b"
)
ACQUISITION_RELATIVE_PATH = (
    "manifests/financial_sec13f_contract_v2_fresh_acquisition_v1.json"
)
ACQUISITION_FILE_SHA256 = (
    "0d5629a8fe7360b76bc7343aa0064d58743a8ce0fb9552ff505705b1b2806e39"
)
ACQUISITION_HASH = (
    "0f19907600a5e1eb38e987f6ccbb3e28d2285de72eaafc0e19fa505432e815ee"
)
ARCHIVE_SET_HASH = (
    "d7261ed659f54408600d422a58996d65030826b10828cb1a4d0064f834ca966d"
)
PREVIOUS_LABEL = "2025 Q4"
CURRENT_LABEL = "2026 Q1"
PREVIOUS_CONTAINER_ROOT = "/root/2025-q2"
CURRENT_CONTAINER_ROOT = "/root/2025-q3"
EXPECTED_REPORT_DATES = {"previous": "2025-12-31", "current": "2026-03-31"}
DRIVER_RELATIVE_PATH = (
    "replication_runtime/financial_sec13f_contract_v2/replication_c_formation.py"
)
SOURCE_RELATIVE_PATHS = (
    "replication_runtime/financial_semantic_v2/pack.py",
    "replication_runtime/financial_sec13f_contract_v2/formation.py",
    DRIVER_RELATIVE_PATH,
)


def _self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or declared != payload_hash(body):
        raise FreshFormationError(f"{label} self hash drifted")
    return declared


def _git_root(project: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    root = Path(result.stdout.strip()).resolve(strict=True)
    project.relative_to(root)
    return root


def _repo_relative(project: Path, relative: str) -> str:
    return (project.relative_to(_git_root(project)) / relative).as_posix()


def _head(project: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _committed_file(project: Path, relative: str, expected_sha: str) -> Path:
    path = project / relative
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
        raise FreshFormationError(f"committed input drifted: {relative}")
    live = path.read_bytes()
    committed = subprocess.run(
        ["git", "-C", str(project), "show", f"HEAD:{_repo_relative(project, relative)}"],
        check=True,
        capture_output=True,
    ).stdout
    if committed != live:
        raise FreshFormationError(f"committed input is dirty: {relative}")
    return path.resolve(strict=True)


def _source_closure(project: Path) -> dict[str, Any]:
    head = _head(project)
    rows: list[dict[str, str]] = []
    for relative in SOURCE_RELATIVE_PATHS:
        path = project / relative
        if path.is_symlink() or not path.is_file():
            raise FreshFormationError("formation source closure is incomplete")
        rows.append({"relative_path": relative, "file_sha256": sha256_file(path)})
    diff = subprocess.run(
        ["git", "-C", str(project), "diff", "--quiet", "HEAD", "--", *SOURCE_RELATIVE_PATHS],
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(project), "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_RELATIVE_PATHS],
        check=True,
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0 or status.stdout.strip():
        raise FreshFormationError("formation source closure is not committed and clean")
    body = {
        "closure_version": "financial_sec13f_replication_c_formation_source_v1",
        "git_commit": head,
        "files": rows,
        "file_count": len(rows),
        "file_set_hash": payload_hash(rows),
    }
    return {**body, "closure_hash": payload_hash(body)}


def _validate_preregistration(project: Path) -> tuple[dict[str, Any], str]:
    path = _committed_file(
        project, PREREGISTRATION_RELATIVE_PATH, PREREGISTRATION_FILE_SHA256
    )
    value = read_json(path)
    declared = _self_hash(value, "manifest_hash", "replication-C preregistration")
    pack = value.get("pack")
    execution = value.get("measurement_execution")
    boundary = value.get("evidence_boundary")
    exclusions = value.get("exclusion_commitment_views")
    if (
        declared != PREREGISTRATION_HASH
        or value.get("study_id") != STUDY_ID
        or not isinstance(pack, Mapping)
        or pack.get("selection_seed") != SELECTION_SEED
        or pack.get("measurement_count") != 8
        or pack.get("sealed_count") != 4
        or pack.get("exclusion_view_count") != 3
        or pack.get("resplit_authorized") is not False
        or pack.get("collision_with_every_exclusion_view_forbidden") is not True
        or not isinstance(execution, Mapping)
        or execution.get("physical_calls") != 16
        or execution.get("retries") != 0
        or execution.get("model_replay_authorized") is not False
        or not isinstance(boundary, Mapping)
        or boundary.get("gold_formed") is not False
        or boundary.get("model_calls") != 0
        or boundary.get("new_pack_formed") is not False
        or boundary.get("new_sealed_content_read") is not False
        or not isinstance(exclusions, list)
        or len(exclusions) != 3
    ):
        raise FreshFormationError("replication-C preregistration drifted")
    return value, declared


def _validate_acquisition(
    project: Path, preregistration: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    path = _committed_file(project, ACQUISITION_RELATIVE_PATH, ACQUISITION_FILE_SHA256)
    value = read_json(path)
    declared = _self_hash(value, "receipt_hash", "SEC acquisition receipt")
    frozen = preregistration.get("source_acquisition")
    identity_fields = (
        "role",
        "archive_sha256",
        "size_bytes",
        "coverpage_sha256",
        "infotable_sha256",
        "source_fingerprint",
        "source_url",
        "source_path_persisted",
    )

    def identities(rows: object) -> list[dict[str, Any]]:
        if not isinstance(rows, list) or len(rows) != 2:
            raise FreshFormationError("SEC acquisition archive set is malformed")
        if not all(isinstance(row, Mapping) for row in rows):
            raise FreshFormationError("SEC acquisition archive row is malformed")
        return [
            {field: row.get(field) for field in identity_fields}
            for row in rows
        ]

    if (
        declared != ACQUISITION_HASH
        or value.get("archive_set_hash") != ARCHIVE_SET_HASH
        or not isinstance(frozen, Mapping)
        or frozen.get("receipt_hash") != declared
        or frozen.get("archive_set_hash") != ARCHIVE_SET_HASH
        or identities(frozen.get("archives")) != identities(value.get("archives"))
        or value.get("model_calls") != 0
        or value.get("online_judge_calls") != 0
        or value.get("secret_value_persisted") is not False
    ):
        raise FreshFormationError("inherited SEC acquisition drifted")
    return value, declared


def _validate_archive(path: Path, expected: Mapping[str, Any]) -> Sec13FSource:
    archive = path.expanduser().resolve(strict=True)
    if archive.is_symlink() or not archive.is_file():
        raise FreshFormationError("SEC archive is not a regular file")
    source = Sec13FSource.open(archive)
    if (
        sha256_file(archive) != expected.get("archive_sha256")
        or archive.stat().st_size != expected.get("size_bytes")
        or source.coverpage_sha256 != expected.get("coverpage_sha256")
        or source.infotable_sha256 != expected.get("infotable_sha256")
        or source.source_fingerprint != expected.get("source_fingerprint")
    ):
        raise FreshFormationError("SEC archive identity drifted")
    return source


def _load_exclusions(
    project: Path, preregistration: Mapping[str, Any], supplied: Sequence[Path]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    frozen = preregistration["exclusion_commitment_views"]
    expected_paths = [str(row["relative_path"]) for row in frozen]
    supplied_paths = [
        path.expanduser().resolve(strict=True).relative_to(project).as_posix()
        for path in supplied
    ]
    if supplied_paths != expected_paths:
        raise FreshFormationError("exclusion view order or identity drifted")
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for binding in frozen:
        path = _committed_file(
            project, str(binding["relative_path"]), str(binding["file_sha256"])
        )
        view = verify_measurement_view(read_json(path))
        if view["measurement_view_hash"] != binding["measurement_view_hash"]:
            raise FreshFormationError("exclusion measurement view drifted")
        result.append((dict(binding), view))
    return result


def form_replication_c_pack_v1(
    *,
    project_root: str | Path,
    previous_archive: str | Path,
    current_archive: str | Path,
    exclusion_views: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    project = Path(project_root).expanduser().resolve(strict=True)
    preregistration, preregistration_hash = _validate_preregistration(project)
    acquisition, acquisition_hash = _validate_acquisition(project, preregistration)
    rows = acquisition["archives"]
    if [row.get("role") for row in rows] != ["previous", "current"]:
        raise FreshFormationError("SEC acquisition role order drifted")
    previous_path = Path(previous_archive)
    current_path = Path(current_archive)
    before = [
        {"sha256": sha256_file(previous_path), "size": previous_path.stat().st_size},
        {"sha256": sha256_file(current_path), "size": current_path.stat().st_size},
    ]
    previous = _validate_archive(previous_path, rows[0])
    current = _validate_archive(current_path, rows[1])
    exclusions = _load_exclusions(project, preregistration, exclusion_views)
    source_closure = _source_closure(project)

    # The single construction call authorized by the preregistration.
    pack = build_public_pack(
        previous_source=previous,
        current_source=current,
        previous_period_label=PREVIOUS_LABEL,
        current_period_label=CURRENT_LABEL,
        preregistration_seed=SELECTION_SEED,
        previous_container_root=PREVIOUS_CONTAINER_ROOT,
        current_container_root=CURRENT_CONTAINER_ROOT,
    )
    pack = verify_public_pack(pack)
    if pack.get("snapshot_report_dates") != EXPECTED_REPORT_DATES:
        raise FreshFormationError("replication-C SEC report dates drifted")
    view = build_measurement_view(pack)
    audits: list[dict[str, Any]] = []
    for binding, prior_view in exclusions:
        collision = assert_no_prior_commitment_collision_v1(
            new_pack=pack, prior_measurement_view=prior_view
        )
        audit_body = {
            "exclusion_view_relative_path": binding["relative_path"],
            "exclusion_view_file_sha256": binding["file_sha256"],
            "exclusion_measurement_view_hash": binding["measurement_view_hash"],
            "collision_audit": collision,
        }
        audits.append({**audit_body, "binding_hash": payload_hash(audit_body)})
    after = [
        {"sha256": sha256_file(previous_path), "size": previous_path.stat().st_size},
        {"sha256": sha256_file(current_path), "size": current_path.stat().st_size},
    ]
    if after != before:
        raise FreshFormationError("SEC archives changed during pack formation")
    receipt_body = {
        "receipt_version": RECEIPT_VERSION,
        "study_id": STUDY_ID,
        "preregistration": {
            "relative_path": PREREGISTRATION_RELATIVE_PATH,
            "file_sha256": PREREGISTRATION_FILE_SHA256,
            "manifest_hash": preregistration_hash,
        },
        "source_acquisition": {
            "relative_path": ACQUISITION_RELATIVE_PATH,
            "file_sha256": ACQUISITION_FILE_SHA256,
            "receipt_hash": acquisition_hash,
            "archive_set_hash": ARCHIVE_SET_HASH,
            "redownloaded": False,
        },
        "formation_source_closure": source_closure,
        "selection_seed": SELECTION_SEED,
        "private_pack_hash": pack["pack_hash"],
        "measurement_view_hash": view["measurement_view_hash"],
        "measurement_count": 8,
        "sealed_commitment_count": 4,
        "exclusion_collision_audits": audits,
        "exclusion_collision_audit_set_hash": payload_hash(audits),
        "all_prior_query_and_instruction_commitments_disjoint": True,
        "formation_after_preregistration": True,
        "private_pack_path_persisted": False,
        "private_pack_content_persisted_in_receipt": False,
        "new_sealed_content_persisted_in_receipt": False,
        "prior_private_pack_accessed": False,
        "prior_sealed_content_accessed": False,
        "gold_formed": False,
        "oracle_calls": 0,
        "model_calls": 0,
        "network_calls": 0,
        "online_judge_calls": 0,
        "secret_value_persisted": False,
    }
    return pack, view, receipt_body


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--exclusion-view", type=Path, action="append", required=True)
    parser.add_argument("--private-pack-output", type=Path, required=True)
    parser.add_argument("--measurement-view-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = (
        args.private_pack_output,
        args.measurement_view_output,
        args.receipt_output,
    )
    if len(args.exclusion_view) != 3 or any(path.exists() or path.is_symlink() for path in outputs):
        raise FreshFormationError("replication-C requires three exclusions and fresh outputs")
    pack, view, receipt_body = form_replication_c_pack_v1(
        project_root=args.project_root,
        previous_archive=args.previous,
        current_archive=args.current,
        exclusion_views=args.exclusion_view,
    )
    write_json(args.private_pack_output, pack)
    write_json(args.measurement_view_output, view)
    receipt_body["private_pack_file_sha256"] = sha256_file(args.private_pack_output)
    receipt_body["measurement_view_file_sha256"] = sha256_file(
        args.measurement_view_output
    )
    receipt = {**receipt_body, "receipt_hash": payload_hash(receipt_body)}
    write_json(args.receipt_output, receipt)
    print(
        json.dumps(
            {
                "private_pack_hash": pack["pack_hash"],
                "measurement_view_hash": view["measurement_view_hash"],
                "receipt_hash": receipt["receipt_hash"],
                "collision_audit_count": len(receipt_body["exclusion_collision_audits"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
