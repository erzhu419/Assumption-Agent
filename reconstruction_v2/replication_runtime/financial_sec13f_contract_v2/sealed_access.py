from __future__ import annotations

"""One-use access boundary for the Replication-C sealed private pack.

Importing this module never probes a private path.  The only operation which
may open one is :func:`read_authorized_private_pack_v1`; it first persists a
content-free claim in a caller-supplied private journal directory.  A consumed
authorization cannot be replayed, even when the first access fails.
"""

import json
import os
from pathlib import Path
from typing import Any, Mapping

from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    verify_public_pack,
)


ACCESS_VERSION = "financial_sec13f_replication_c_sealed_access_v1"
AUTHORIZATION_VERSION = (
    "financial_sec13f_contract_v2_sealed_authorization_v1"
)
CLAIM_FILENAME = "sealed_access.claim.json"
COMPLETION_FILENAME = "sealed_access.completed.json"
FAILURE_FILENAME = "sealed_access.failed.json"


class SealedAccessError(PermissionError):
    """The sealed access boundary failed closed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if not _is_sha256(declared) or declared != payload_hash(body):
        raise SealedAccessError(f"{label} self hash drifted")
    return str(declared)


def validate_sealed_authorization_v1(
    value: Mapping[str, Any],
    *,
    expected_study_id: str,
    expected_private_pack_hash: str,
    expected_measurement_view_hash: str,
    expected_candidate_id: str,
) -> str:
    """Validate the small, public post-controls authorization contract."""

    declared = _self_hash(value, "manifest_hash", "sealed authorization")
    authorization = value.get("authorization")
    decision = value.get("decision")
    prerequisites = value.get("prerequisite_bindings")
    cohort = value.get("sealed_cohort_binding")
    candidate = value.get("candidate_and_provider_binding")
    sequencing = value.get("sequencing")
    incident = value.get("incident_adjudication")
    if (
        value.get("manifest_version") != AUTHORIZATION_VERSION
        or value.get("study_id") != expected_study_id
        or not isinstance(authorization, Mapping)
        or authorization.get("private_pack_content_access_authorized") is not True
        or authorization.get("sealed_evaluation_authorized") is not True
        or authorization.get("sealed_preparation_authorized") is not True
        or authorization.get("sealed_scoring_authorized") is not True
        or authorization.get("sealed_item_count_authorized") != 4
        or authorization.get("online_judge_authorized") is not False
        or authorization.get("candidate_mutation_authorized") is not False
        or not isinstance(decision, Mapping)
        or decision.get("sealed_authorization_decision")
        != "authorize_exact_preregistered_replication_c_sealed_evaluation"
        or decision.get("post_controls") is not True
        or decision.get("controls_disposition_accepted") is not True
        or decision.get("family_out_disposition_accepted") is not True
        or decision.get("incident_explicitly_adjudicated") is not True
        or decision.get("sealed_item_set_changed") is not False
        or not isinstance(prerequisites, Mapping)
        or set(prerequisites)
        != {
            "controls_disposition",
            "family_out_disposition",
            "promotion_decision",
            "sealed_preregistration",
        }
        or any(
            not isinstance(row, Mapping)
            or not _is_sha256(row.get("manifest_hash"))
            for row in prerequisites.values()
        )
        or not isinstance(cohort, Mapping)
        or cohort.get("private_pack_hash") != expected_private_pack_hash
        or cohort.get("measurement_view_hash") != expected_measurement_view_hash
        or cohort.get("item_count") != 4
        or cohort.get("item_replacement_authorized") is not False
        or not _is_sha256(cohort.get("precommitted_private_pack_file_sha256"))
        or not isinstance(candidate, Mapping)
        or candidate.get("candidate_id") != expected_candidate_id
        or candidate.get("provider_label") != "plus"
        or candidate.get("pro_fallback_authorized") is not False
        or candidate.get("candidate_unchanged_after_promotion_and_controls") is not True
        or not isinstance(incident, Mapping)
        or incident.get("accepted_for_current_pack_continuation") is not True
        or incident.get("semantic_holdout_blindness_preserved") is not True
        or incident.get("strict_zero_byte_policy_satisfied") is not False
        or incident.get("original_zero_byte_pre_authorization_claim_waived") is not True
        or not isinstance(sequencing, Mapping)
        or sequencing.get("access_journal_must_open_before_next_private_pack_byte_read") is not True
        or sequencing.get("next_private_pack_access_must_be_recorded") is not True
    ):
        raise SealedAccessError("sealed authorization policy drifted")
    return declared


def _write_new_hashed_json(
    path: Path,
    body: Mapping[str, Any],
    *,
    hash_field: str,
) -> dict[str, Any]:
    payload = {**dict(body), hash_field: payload_hash(body)}
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # The file fsync makes the bytes durable; the directory fsync makes
        # the new O_EXCL directory entry durable before a caller may proceed
        # to the private-path probe.
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        try:
            os.unlink(path)
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_descriptor = os.open(path.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except FileNotFoundError:
            pass
        raise
    return payload


def read_authorized_private_pack_v1(
    *,
    unresolved_private_pack_path: str | Path,
    journal_root: str | Path,
    authorization: Mapping[str, Any],
    expected_study_id: str,
    expected_private_pack_hash: str,
    expected_measurement_view_hash: str,
    expected_candidate_id: str,
) -> dict[str, Any]:
    """Consume one authorization and read one private JSON object.

    The path is intentionally not expanded, resolved, stat'ed, or hashed until
    after the durable claim is written.  The journal contains only public
    identities and hashes; it never stores the path or private bytes.
    """

    authorization_hash = validate_sealed_authorization_v1(
        authorization,
        expected_study_id=expected_study_id,
        expected_private_pack_hash=expected_private_pack_hash,
        expected_measurement_view_hash=expected_measurement_view_hash,
        expected_candidate_id=expected_candidate_id,
    )
    unresolved_journal = Path(journal_root).expanduser()
    if unresolved_journal.is_symlink():
        raise SealedAccessError("sealed access journal is symlinked")
    journal = unresolved_journal.resolve()
    journal.mkdir(parents=True, exist_ok=True, mode=0o700)
    if journal.is_symlink() or not journal.is_dir():
        raise SealedAccessError("sealed access journal is unavailable")
    if any(journal.iterdir()):
        raise SealedAccessError("sealed authorization has already been consumed")
    claim = _write_new_hashed_json(
        journal / CLAIM_FILENAME,
        {
            "access_version": ACCESS_VERSION,
            "authorization_manifest_hash": authorization_hash,
            "study_id_hash": payload_hash({"study_id": expected_study_id}),
            "private_pack_hash": expected_private_pack_hash,
            "precommitted_private_pack_file_sha256": authorization[
                "sealed_cohort_binding"
            ]["precommitted_private_pack_file_sha256"],
            "measurement_view_hash": expected_measurement_view_hash,
            "candidate_id": expected_candidate_id,
            "private_path_persisted": False,
            "private_content_persisted": False,
            "access_claimed_before_path_probe": True,
            "single_access": True,
            "retry_authorized": False,
            "replay_authorized": False,
        },
        hash_field="claim_hash",
    )
    try:
        source = Path(unresolved_private_pack_path).expanduser()
        if source.is_symlink() or not source.is_file():
            raise SealedAccessError("authorized private pack is unavailable")
        source = source.resolve(strict=True)
        raw = source.read_bytes()
        expected_file_sha = authorization["sealed_cohort_binding"][
            "precommitted_private_pack_file_sha256"
        ]
        import hashlib

        if hashlib.sha256(raw).hexdigest() != expected_file_sha:
            raise SealedAccessError("authorized private pack file hash drifted")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise SealedAccessError("authorized private pack is not one object")
        verified = verify_public_pack(value)
        if verified.get("pack_hash") != expected_private_pack_hash:
            raise SealedAccessError("authorized private pack object hash drifted")
        _write_new_hashed_json(
            journal / COMPLETION_FILENAME,
            {
                "access_version": ACCESS_VERSION,
                "authorization_manifest_hash": authorization_hash,
                "claim_hash": claim["claim_hash"],
                "expected_private_pack_hash": expected_private_pack_hash,
                "precommitted_private_pack_file_sha256": expected_file_sha,
                "raw_file_sha256_matches_precommit": True,
                "verified_public_pack_hash_matches_commitment": True,
                "private_path_persisted": False,
                "private_content_persisted": False,
                "access_completed": True,
                "retry_authorized": False,
                "replay_authorized": False,
            },
            hash_field="completion_hash",
        )
        return verified
    except Exception as exc:
        try:
            _write_new_hashed_json(
                journal / FAILURE_FILENAME,
                {
                    "access_version": ACCESS_VERSION,
                    "authorization_manifest_hash": authorization_hash,
                    "claim_hash": claim["claim_hash"],
                    "error_type": type(exc).__name__,
                    "error_message_hash": payload_hash({"message": str(exc)}),
                    "raw_error_persisted": False,
                    "private_path_persisted": False,
                    "private_content_persisted": False,
                    "retry_authorized": False,
                    "replay_authorized": False,
                },
                hash_field="failure_hash",
            )
        except FileExistsError:
            pass
        raise
