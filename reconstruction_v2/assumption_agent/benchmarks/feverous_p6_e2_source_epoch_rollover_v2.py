"""Content-free authorization for the FEVEROUS formal source epoch v2.

The first formal acquisition terminated before cohort formation.  Its marker,
terminal failure, and selection secret remain immutable in the v1 root.  This
module verifies the two public JSON commitments, metadata for the private
secret, and the committed aggregate-only TRAIN-loader qualification.  It never
reads or hashes the predecessor secret.  A successor may start only in the
disjoint v2 root and only while that entire root is absent.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from assumption_agent.benchmarks import (
    feverous_p6_e2_train_loader_qualification_v2 as train_loader_qualification,
)


VERSION = "feverous_p6_e2_source_epoch_rollover_v2"
SCHEMA = f"{VERSION}_manifest"
MANIFEST_RELATIVE = Path("manifests/feverous_p6_e2_source_epoch_rollover_v2.json")

PREDECESSOR_FORMAL_ROOT_RELATIVE = Path("artifacts/feverous_p6_e2_formal_v1")
PREDECESSOR_ACQUISITION_ROOT_RELATIVE = (
    PREDECESSOR_FORMAL_ROOT_RELATIVE / "acquisition"
)
PREDECESSOR_MARKER_RELATIVE = (
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "acquisition.one_shot_marker.json"
)
PREDECESSOR_FAILURE_RELATIVE = (
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "acquisition.terminal_failure.json"
)
PREDECESSOR_SECRET_RELATIVE = (
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "selection_secret.private.bin"
)

SUCCESSOR_FORMAL_ROOT_RELATIVE = Path("artifacts/feverous_p6_e2_formal_v2")
SUCCESSOR_ACQUISITION_ROOT_RELATIVE = SUCCESSOR_FORMAL_ROOT_RELATIVE / "acquisition"
SUCCESSOR_CONTROLLER_ROOT_RELATIVE = SUCCESSOR_FORMAL_ROOT_RELATIVE / "controller"
SUCCESSOR_HIPPORAG_STAGE_RELATIVE = (
    SUCCESSOR_FORMAL_ROOT_RELATIVE / "official_hipporag_stage"
)
SUCCESSOR_HIPPORAG_WORK_RELATIVE = (
    SUCCESSOR_FORMAL_ROOT_RELATIVE / "hipporag_query_work"
)
SUCCESSOR_NER_PRIVATE_RELATIVE = SUCCESSOR_FORMAL_ROOT_RELATIVE / "ner_private"

PREDECESSOR_MARKER_SHA256 = (
    "f48037c471231e520b4c4c41a3bd73c607f89c5f332db4534c4ebb170e9d3f42"
)
PREDECESSOR_MARKER_FILE_SHA256 = (
    "7ad9474441d3601e11b51a56d839f47f14e04fbf4934be498d99d266bddd00c9"
)
PREDECESSOR_FAILURE_SHA256 = (
    "35a3c3ab0a1ba6fa3d5a55202a812f222e89889e044532bbf1f33a2e158f969e"
)
PREDECESSOR_FAILURE_FILE_SHA256 = (
    "51fefdf2a5a9d6bbd0ea367fa53d266efac4eb111bcdfdfbccb743c8d3a81042"
)
PREDECESSOR_IMPLEMENTATION_FREEZE_SHA256 = (
    "86825c33c63204d2782f53ce561be692200537de8845ac1af8a8ecf107db3d72"
)
PREDECESSOR_FAILURE_EXCEPTION_MESSAGE_SHA256 = (
    "2fc1abdeb8f9cebf5dc7da30d3e51c622cf83f4bf2839d9812c9c1e6d90e82e2"
)
TRAIN_LOADER_QUALIFICATION_SHA256 = (
    "33dee5047bf6fdfec818317c62479e858cc4bf8ee5385c57cf35c4fdf41bfeb0"
)
PREDECESSOR_ABSENT_RELATIVES = (
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "acquisition.public.json",
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "corpus.private.json",
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "A_form.view.private.json",
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "F_search.view.private.json",
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "A_hold.view.sealed.json",
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "M_search.view.sealed.json",
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "A_form.labels.sealed.json",
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "F_search.labels.sealed.json",
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "A_hold.labels.sealed.json",
    PREDECESSOR_ACQUISITION_ROOT_RELATIVE / "M_search.labels.sealed.json",
    PREDECESSOR_FORMAL_ROOT_RELATIVE / "controller",
    PREDECESSOR_FORMAL_ROOT_RELATIVE / "official_hipporag_stage",
    PREDECESSOR_FORMAL_ROOT_RELATIVE / "hipporag_query_work",
    PREDECESSOR_FORMAL_ROOT_RELATIVE / "ner_private",
)


class FeverousSourceEpochRolloverError(RuntimeError):
    """The immutable predecessor or preregistered successor binding drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousSourceEpochRolloverError(
            "rollover value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FeverousSourceEpochRolloverError(
            "public predecessor commitment cannot be hashed"
        ) from exc
    return digest.hexdigest()


def _canonical_project(project: str | Path) -> Path:
    try:
        root = Path(project).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FeverousSourceEpochRolloverError("project root is unavailable") from exc
    if not root.is_dir() or root.is_symlink():
        raise FeverousSourceEpochRolloverError("project root is unsafe")
    return root


def _load_public_json(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise FeverousSourceEpochRolloverError(
            "public predecessor commitment is unavailable"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverousSourceEpochRolloverError(
            "public predecessor commitment is invalid"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise FeverousSourceEpochRolloverError(
            "public predecessor commitment is noncanonical"
        )
    return value, hashlib.sha256(raw).hexdigest()


def manifest_body() -> dict[str, Any]:
    """Return the exact content-free rollover preregistration body."""

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "preregistered_pre_action_source_epoch_rollover",
        "source_split": "TRAIN",
        "predecessor": {
            "source_epoch": "feverous_p6_e2_formal_v1",
            "implementation_freeze_sha256": (
                PREDECESSOR_IMPLEMENTATION_FREEZE_SHA256
            ),
            "formal_root_relative": PREDECESSOR_FORMAL_ROOT_RELATIVE.as_posix(),
            "marker_relative": PREDECESSOR_MARKER_RELATIVE.as_posix(),
            "marker_sha256": PREDECESSOR_MARKER_SHA256,
            "marker_file_sha256": PREDECESSOR_MARKER_FILE_SHA256,
            "failure_relative": PREDECESSOR_FAILURE_RELATIVE.as_posix(),
            "failure_sha256": PREDECESSOR_FAILURE_SHA256,
            "failure_file_sha256": PREDECESSOR_FAILURE_FILE_SHA256,
            "failure_exception_message_sha256": (
                PREDECESSOR_FAILURE_EXCEPTION_MESSAGE_SHA256
            ),
            "terminal_status": "formal_acquisition_failed_no_retry_or_resample",
            "absent_receipt_pack_controller_and_runtime_relatives": [
                path.as_posix() for path in PREDECESSOR_ABSENT_RELATIVES
            ],
        },
        "predecessor_selection_secret_metadata": {
            "relative_path": PREDECESSOR_SECRET_RELATIVE.as_posix(),
            "exists": True,
            "regular_file": True,
            "symlink": False,
            "size_bytes": 32,
            "mode_octal": "0600",
            "content_or_hash_read_by_rollover_verifier": False,
            "content_or_hash_read_by_successor_acquisition": False,
        },
        "predecessor_secret_pre_freeze_audit_incident": {
            "status": "random_secret_sha256_accidentally_computed_by_diagnostic_glob",
            "reported_hash_computation_count": 1,
            "raw_secret_bytes_viewed_or_disclosed": False,
            "transient_diagnostic_output_observed": True,
            "hash_value_may_exist_in_agent_tool_log": True,
            "hash_value_written_to_project_artifact_committed_or_used_by_successor": (
                False
            ),
            "hash_value_forbidden_from_successor_inputs_and_artifacts": True,
            "predecessor_secret_applied_to_records_selection_or_cohort": False,
            "successor_uses_fresh_independent_os_random_secret": True,
            "scientific_selection_information_exposed": False,
        },
        "real_train_loader_qualification": {
            "relative_path": (
                train_loader_qualification.MANIFEST_RELATIVE.as_posix()
            ),
            "qualification_sha256": TRAIN_LOADER_QUALIFICATION_SHA256,
            "required_before_successor_marker_or_secret": True,
            "committed_and_implementation_freeze_bound": True,
        },
        "successor": {
            "source_epoch": "feverous_p6_e2_formal_v2",
            "formal_root_relative": SUCCESSOR_FORMAL_ROOT_RELATIVE.as_posix(),
            "acquisition_root_relative": (
                SUCCESSOR_ACQUISITION_ROOT_RELATIVE.as_posix()
            ),
            "controller_root_relative": SUCCESSOR_CONTROLLER_ROOT_RELATIVE.as_posix(),
            "hipporag_stage_relative": SUCCESSOR_HIPPORAG_STAGE_RELATIVE.as_posix(),
            "hipporag_work_relative": SUCCESSOR_HIPPORAG_WORK_RELATIVE.as_posix(),
            "ner_private_relative": SUCCESSOR_NER_PRIVATE_RELATIVE.as_posix(),
        },
        "predecessor_raw_train_rows_transiently_json_decoded": True,
        "predecessor_raw_train_claim_label_and_evidence_fields_transiently_decoded": (
            True
        ),
        "predecessor_records_adapted_selected_or_persisted": False,
        "predecessor_cohort_pack_corpus_retrieval_utility_or_evaluator_use": False,
        "predecessor_marker_secret_or_failure_overwrite_authorized": False,
        "same_epoch_retry_replay_or_resample_authorized": False,
        "successor_root_reuse_authorized": False,
        "successor_selection_secret_must_be_fresh": True,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }


def form_rollover_manifest() -> dict[str, Any]:
    body = manifest_body()
    return {**body, "source_epoch_rollover_sha256": stable_hash(body)}


def _verify_manifest_file(path: Path) -> dict[str, Any]:
    value, _file_sha = _load_public_json(path)
    expected = form_rollover_manifest()
    if value != expected:
        raise FeverousSourceEpochRolloverError("rollover manifest drifted")
    return value


def verify_rollover_manifest(
    project: str | Path, *, require_successor_absent: bool = True
) -> Mapping[str, Any]:
    """Verify v1 public commitments and secret metadata without opening secret."""

    root = _canonical_project(project)
    manifest = _verify_manifest_file(root / MANIFEST_RELATIVE)

    qualification = (
        train_loader_qualification.verify_train_loader_qualification(root)
    )
    if (
        qualification.get("qualification_sha256")
        != TRAIN_LOADER_QUALIFICATION_SHA256
    ):
        raise FeverousSourceEpochRolloverError(
            "real TRAIN loader qualification binding drifted"
        )

    marker, marker_file_sha = _load_public_json(root / PREDECESSOR_MARKER_RELATIVE)
    marker_body = dict(marker)
    marker_sha = marker_body.pop("marker_sha256", None)
    if (
        marker_sha != PREDECESSOR_MARKER_SHA256
        or stable_hash(marker_body) != marker_sha
        or marker_file_sha != PREDECESSOR_MARKER_FILE_SHA256
        or marker.get("implementation_freeze_sha256")
        != PREDECESSOR_IMPLEMENTATION_FREEZE_SHA256
    ):
        raise FeverousSourceEpochRolloverError(
            "predecessor public marker commitment drifted"
        )

    failure, failure_file_sha = _load_public_json(root / PREDECESSOR_FAILURE_RELATIVE)
    failure_body = dict(failure)
    failure_sha = failure_body.pop("failure_sha256", None)
    if (
        failure_sha != PREDECESSOR_FAILURE_SHA256
        or stable_hash(failure_body) != failure_sha
        or failure_file_sha != PREDECESSOR_FAILURE_FILE_SHA256
        or failure.get("implementation_freeze_sha256")
        != PREDECESSOR_IMPLEMENTATION_FREEZE_SHA256
        or failure.get("exception_message_sha256")
        != PREDECESSOR_FAILURE_EXCEPTION_MESSAGE_SHA256
        or failure.get("status") != "formal_acquisition_failed_no_retry_or_resample"
        or failure.get("online_evaluator_calls") != 0
    ):
        raise FeverousSourceEpochRolloverError(
            "predecessor public failure commitment drifted"
        )

    if any(
        os.path.lexists(root / relative)
        for relative in PREDECESSOR_ABSENT_RELATIVES
    ):
        raise FeverousSourceEpochRolloverError(
            "predecessor post-marker acquisition/action artifact unexpectedly exists"
        )

    secret_path = root / PREDECESSOR_SECRET_RELATIVE
    try:
        secret_stat = secret_path.lstat()
    except OSError as exc:
        raise FeverousSourceEpochRolloverError(
            "predecessor secret metadata is unavailable"
        ) from exc
    if (
        secret_path.is_symlink()
        or not stat.S_ISREG(secret_stat.st_mode)
        or stat.S_IMODE(secret_stat.st_mode) != 0o600
        or secret_stat.st_size != 32
    ):
        raise FeverousSourceEpochRolloverError(
            "predecessor secret metadata drifted"
        )

    successor_root = root / SUCCESSOR_FORMAL_ROOT_RELATIVE
    if require_successor_absent and os.path.lexists(successor_root):
        raise FeverousSourceEpochRolloverError(
            "successor formal root already exists"
        )
    return manifest


__all__ = [
    "FeverousSourceEpochRolloverError",
    "MANIFEST_RELATIVE",
    "PREDECESSOR_FAILURE_FILE_SHA256",
    "PREDECESSOR_FAILURE_EXCEPTION_MESSAGE_SHA256",
    "PREDECESSOR_FAILURE_SHA256",
    "PREDECESSOR_IMPLEMENTATION_FREEZE_SHA256",
    "PREDECESSOR_MARKER_FILE_SHA256",
    "PREDECESSOR_MARKER_SHA256",
    "PREDECESSOR_ABSENT_RELATIVES",
    "PREDECESSOR_SECRET_RELATIVE",
    "SCHEMA",
    "SUCCESSOR_ACQUISITION_ROOT_RELATIVE",
    "SUCCESSOR_CONTROLLER_ROOT_RELATIVE",
    "SUCCESSOR_FORMAL_ROOT_RELATIVE",
    "SUCCESSOR_HIPPORAG_STAGE_RELATIVE",
    "SUCCESSOR_HIPPORAG_WORK_RELATIVE",
    "SUCCESSOR_NER_PRIVATE_RELATIVE",
    "TRAIN_LOADER_QUALIFICATION_SHA256",
    "VERSION",
    "form_rollover_manifest",
    "manifest_body",
    "stable_hash",
    "verify_rollover_manifest",
]
