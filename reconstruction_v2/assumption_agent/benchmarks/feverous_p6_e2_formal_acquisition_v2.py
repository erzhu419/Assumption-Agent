"""One-shot FEVEROUS formal pack formation in successor source epoch v2.

Epoch v1 terminated before cohort formation.  This module preserves the same
frozen selection and pack compiler while requiring the preregistered public
rollover commitment and writing exclusively beneath the disjoint v2 root.
There is no retry, replay, resampling, or root-reuse path within v2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import secrets
import stat
from typing import Any

from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks import (
    feverous_p6_e2_formal_acquisition_v1 as predecessor_acquisition,
)
from assumption_agent.benchmarks import feverous_p6_e2_formal_source_v1 as formal_source
from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as source_adapter
from assumption_agent.benchmarks import (
    feverous_p6_e2_source_epoch_rollover_v2 as rollover,
)


VERSION = "feverous_p6_e2_formal_acquisition_v2"
FORMAL_ROOT_RELATIVE = rollover.SUCCESSOR_FORMAL_ROOT_RELATIVE
ROOT_RELATIVE = rollover.SUCCESSOR_ACQUISITION_ROOT_RELATIVE
RECEIPT_RELATIVE = ROOT_RELATIVE / "acquisition.public.json"
MARKER_RELATIVE = ROOT_RELATIVE / "acquisition.one_shot_marker.json"
FAILURE_RELATIVE = ROOT_RELATIVE / "acquisition.terminal_failure.json"
SECRET_RELATIVE = ROOT_RELATIVE / "selection_secret.private.bin"
CORPUS_RELATIVE = ROOT_RELATIVE / "corpus.private.json"

VIEW_RELATIVES = {
    "A_form": ROOT_RELATIVE / "A_form.view.private.json",
    "F_search": ROOT_RELATIVE / "F_search.view.private.json",
    "A_hold": ROOT_RELATIVE / "A_hold.view.sealed.json",
    "M_search": ROOT_RELATIVE / "M_search.view.sealed.json",
}
LABEL_RELATIVES = {
    "A_form": ROOT_RELATIVE / "A_form.labels.sealed.json",
    "A_hold": ROOT_RELATIVE / "A_hold.labels.sealed.json",
    "M_search": ROOT_RELATIVE / "M_search.labels.sealed.json",
}
F_SEARCH_LABEL_RELATIVE = ROOT_RELATIVE / "F_search.labels.sealed.json"

_EXPECTED_BINDING_RELATIVES = {
    "corpus_view": CORPUS_RELATIVE,
    **{
        f"{block}_view": VIEW_RELATIVES[block]
        for block in acquisition.BLOCK_ORDER
    },
    **{
        f"{block}_labels": LABEL_RELATIVES[block]
        for block in ("A_form", "A_hold", "M_search")
    },
}
_EXPECTED_BINDING_ROLE_ORDER = tuple(_EXPECTED_BINDING_RELATIVES)
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "version",
        "status",
        "source_epoch",
        "source_epoch_rollover_sha256",
        "train_loader_qualification_sha256",
        "implementation_freeze_sha256",
        "identity_full_compile_equivalence_qualification_sha256",
        "selection_secret_sha256",
        "selection_secret_bytes",
        "selection_secret_persisted_publicly",
        "source_split",
        "annotation_receipt",
        "database_page_stream_receipt",
        "selected_page_lookup_receipt",
        "source_adapter_receipt",
        "selection_statistics",
        "corpus_statistics",
        "block_counts",
        "F_search_gold_pack_created",
        "private_file_bindings",
        "private_file_binding_set_sha256",
        "all_blocks_one_acquisition",
        "action_retrieval_utility_or_evaluator_calls",
        "development_or_test_source_accessed",
        "online_evaluator_calls",
        "acquisition_receipt_sha256",
    }
)
SOURCE_ROOT_RELATIVE = predecessor_acquisition.SOURCE_ROOT_RELATIVE
ANNOTATION_RELATIVE = predecessor_acquisition.ANNOTATION_RELATIVE
DATABASE_RELATIVE = predecessor_acquisition.DATABASE_RELATIVE

FeverousFormalAcquisitionError = (
    predecessor_acquisition.FeverousFormalAcquisitionError
)
_canonical_bytes = predecessor_acquisition._canonical_bytes
_semantic_hash = predecessor_acquisition._semantic_hash
_sha256_file = predecessor_acquisition._sha256_file
_require_sha256 = predecessor_acquisition._require_sha256
_canonical_project = predecessor_acquisition._canonical_project
_exclusive_bytes = predecessor_acquisition._exclusive_bytes
_exclusive_json = predecessor_acquisition._exclusive_json
_load_canonical_json = predecessor_acquisition._load_canonical_json
_file_binding = predecessor_acquisition._file_binding


def _create_private_successor_root(project: Path) -> None:
    """Exclusively claim the entire v2 root before creating any v2 secret."""

    artifacts = project / "artifacts"
    try:
        if not os.path.lexists(artifacts):
            os.mkdir(artifacts, 0o755)
        observed = artifacts.lstat()
        if artifacts.is_symlink() or not stat.S_ISDIR(observed.st_mode):
            raise FeverousFormalAcquisitionError("artifacts parent is unsafe")
        os.mkdir(project / FORMAL_ROOT_RELATIVE, 0o700)
        os.mkdir(project / ROOT_RELATIVE, 0o700)
    except OSError as exc:
        raise FeverousFormalAcquisitionError(
            "successor formal root already exists or cannot be claimed"
        ) from exc
    for path in (project / FORMAL_ROOT_RELATIVE, project / ROOT_RELATIVE):
        observed = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            raise FeverousFormalAcquisitionError(
                "successor private root mode drifted"
            )


@dataclass(frozen=True)
class AcquisitionPaths:
    marker: Path
    failure: Path
    receipt: Path
    secret: Path
    corpus: Path
    views: Mapping[str, Path]
    labels: Mapping[str, Path]


def acquisition_paths(project: str | Path) -> AcquisitionPaths:
    root = _canonical_project(project)
    return AcquisitionPaths(
        marker=root / MARKER_RELATIVE,
        failure=root / FAILURE_RELATIVE,
        receipt=root / RECEIPT_RELATIVE,
        secret=root / SECRET_RELATIVE,
        corpus=root / CORPUS_RELATIVE,
        views={block: root / relative for block, relative in VIEW_RELATIVES.items()},
        labels={block: root / relative for block, relative in LABEL_RELATIVES.items()},
    )


def _public_receipt(
    *,
    project: Path,
    source_epoch_rollover_sha256: str,
    train_loader_qualification_sha256: str,
    implementation_freeze_sha256: str,
    equivalence_qualification_sha256: str,
    secret: bytes,
    source: formal_source.ControlledTrainSource,
    adapter_receipt: Mapping[str, Any],
    selection_stats: Mapping[str, Any],
    corpus_stats: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = [
        _file_binding(project, CORPUS_RELATIVE, role="corpus_view"),
        *(
            _file_binding(project, VIEW_RELATIVES[block], role=f"{block}_view")
            for block in acquisition.BLOCK_ORDER
        ),
        *(
            _file_binding(project, LABEL_RELATIVES[block], role=f"{block}_labels")
            for block in ("A_form", "A_hold", "M_search")
        ),
    ]
    body: dict[str, Any] = {
        "schema": f"{VERSION}_receipt",
        "version": VERSION,
        "status": "all_four_train_blocks_acquired_before_any_action_or_outcome",
        "source_epoch": "feverous_p6_e2_formal_v2",
        "source_epoch_rollover_sha256": source_epoch_rollover_sha256,
        "train_loader_qualification_sha256": train_loader_qualification_sha256,
        "implementation_freeze_sha256": implementation_freeze_sha256,
        "identity_full_compile_equivalence_qualification_sha256": (
            equivalence_qualification_sha256
        ),
        "selection_secret_sha256": hashlib.sha256(secret).hexdigest(),
        "selection_secret_bytes": len(secret),
        "selection_secret_persisted_publicly": False,
        "source_split": "TRAIN",
        "annotation_receipt": dict(source.annotation_receipt),
        "database_page_stream_receipt": dict(source.database_receipt),
        "selected_page_lookup_receipt": dict(source.selected_lookup_receipt),
        "source_adapter_receipt": dict(adapter_receipt),
        "selection_statistics": dict(selection_stats),
        "corpus_statistics": dict(corpus_stats),
        "block_counts": dict(acquisition.BLOCK_COUNTS),
        "F_search_gold_pack_created": False,
        "private_file_bindings": bindings,
        "private_file_binding_set_sha256": _semantic_hash(bindings),
        "all_blocks_one_acquisition": True,
        "action_retrieval_utility_or_evaluator_calls": 0,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    return {**body, "acquisition_receipt_sha256": _semantic_hash(body)}


def perform_formal_acquisition_once(
    *,
    project: str | Path,
    source_epoch_rollover_sha256: str,
    train_loader_qualification_sha256: str,
    implementation_freeze_sha256: str,
    identity_full_compile_equivalence_qualification_sha256: str,
) -> Mapping[str, Any]:
    """Generate the sole v2 cohort and every sealed pack exactly once."""

    root = _canonical_project(project)
    rollover_sha = _require_sha256(
        source_epoch_rollover_sha256, "source epoch rollover"
    )
    verified_rollover = rollover.verify_rollover_manifest(root)
    if verified_rollover.get("source_epoch_rollover_sha256") != rollover_sha:
        raise FeverousFormalAcquisitionError("source epoch rollover drifted")
    qualification_sha = _require_sha256(
        train_loader_qualification_sha256, "real TRAIN loader qualification"
    )
    expected_qualification_sha = verified_rollover.get(
        "real_train_loader_qualification", {}
    ).get("qualification_sha256")
    if qualification_sha != expected_qualification_sha:
        raise FeverousFormalAcquisitionError(
            "real TRAIN loader qualification drifted"
        )
    implementation_sha = _require_sha256(
        implementation_freeze_sha256, "implementation freeze"
    )
    equivalence_sha = _require_sha256(
        identity_full_compile_equivalence_qualification_sha256,
        "identity/full-compiler qualification",
    )
    _create_private_successor_root(root)
    paths = acquisition_paths(root)
    annotation_path = root / ANNOTATION_RELATIVE
    database_path = root / DATABASE_RELATIVE
    try:
        marker_body = {
            "schema": f"{VERSION}_one_shot_marker",
            "version": VERSION,
            "source_epoch": "feverous_p6_e2_formal_v2",
            "source_epoch_rollover_sha256": rollover_sha,
            "train_loader_qualification_sha256": qualification_sha,
            "implementation_freeze_sha256": implementation_sha,
            "identity_full_compile_equivalence_qualification_sha256": equivalence_sha,
            "source_split": "TRAIN",
            "retry_replay_or_resample_authorized": False,
        }
        _exclusive_json(
            paths.marker,
            {**marker_body, "marker_sha256": _semantic_hash(marker_body)},
        )
        secret = secrets.token_bytes(32)
        if len(secret) != 32:
            raise FeverousFormalAcquisitionError("OS secret generation drifted")
        _exclusive_bytes(paths.secret, secret)
        with formal_source.ControlledTrainSource(
            annotation_path=annotation_path,
            database_path=database_path,
        ) as source:
            records = source.read_annotations_once()
            resolver = source.exact_resolver_for_candidate_screen()
            adapted = source_adapter.adapt_train_candidate_records(
                records,
                source_split="TRAIN",
                resolver=resolver,
                binding=source_adapter.FROZEN_TRAIN_BINDING,
            )
            source_adapter.verify_adapter_receipt(adapted.receipt)
            blocks, selection_stats = acquisition.select_private_blocks(
                adapted.candidates, secret
            )
            plan = source.plan_corpus_identities_parallel_once(
                blocks=blocks,
                secret=secret,
                identity_full_compile_equivalence_qualification_sha256=(
                    equivalence_sha
                ),
            )
            selected_units = source.iter_selected_corpus_units_once(plan)
            corpus, corpus_index, corpus_stats = (
                acquisition.materialize_fixed_corpus_from_selection_plan(
                    plan=plan,
                    units=selected_units,
                    secret=secret,
                    require_formal_source=True,
                )
            )
            acquisition.verify_formal_corpus_acquisition(corpus_stats)
            corpus_view, views, labels = acquisition.materialize_private_payloads(
                blocks=blocks,
                corpus=corpus,
                corpus_index=corpus_index,
            )
            _exclusive_json(paths.corpus, corpus_view)
            for block in acquisition.BLOCK_ORDER:
                _exclusive_json(paths.views[block], views[block])
            for block in ("A_form", "A_hold", "M_search"):
                _exclusive_json(paths.labels[block], labels[block])
            receipt = _public_receipt(
                project=root,
                source_epoch_rollover_sha256=rollover_sha,
                train_loader_qualification_sha256=qualification_sha,
                implementation_freeze_sha256=implementation_sha,
                equivalence_qualification_sha256=equivalence_sha,
                secret=secret,
                source=source,
                adapter_receipt=adapted.receipt,
                selection_stats=selection_stats,
                corpus_stats=corpus_stats,
            )
            _exclusive_json(paths.receipt, receipt)
            verify_acquisition_receipt(root)
            return receipt
    except BaseException as exc:
        if not paths.failure.exists():
            failure_body = {
                "schema": f"{VERSION}_terminal_failure",
                "version": VERSION,
                "status": "formal_acquisition_failed_no_retry_or_resample",
                "source_epoch": "feverous_p6_e2_formal_v2",
                "source_epoch_rollover_sha256": rollover_sha,
                "train_loader_qualification_sha256": qualification_sha,
                "exception_type": type(exc).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8", errors="replace")
                ).hexdigest(),
                "implementation_freeze_sha256": implementation_sha,
                "online_evaluator_calls": 0,
            }
            try:
                _exclusive_json(
                    paths.failure,
                    {**failure_body, "failure_sha256": _semantic_hash(failure_body)},
                )
            except FeverousFormalAcquisitionError:
                pass
        raise


def verify_acquisition_envelope(project: str | Path) -> Mapping[str, Any]:
    """Verify v2 public semantics and private metadata without sealed content."""

    root = _canonical_project(project)
    expected_rollover = rollover.verify_rollover_manifest(
        root, require_successor_absent=False
    ).get("source_epoch_rollover_sha256")
    receipt = _load_canonical_json(root / RECEIPT_RELATIVE)
    if set(receipt) != _RECEIPT_KEYS:
        raise FeverousFormalAcquisitionError("acquisition receipt schema drifted")
    body = dict(receipt)
    declared = _require_sha256(
        body.pop("acquisition_receipt_sha256", None), "acquisition receipt"
    )
    bindings = receipt.get("private_file_bindings")
    if (
        receipt.get("schema") != f"{VERSION}_receipt"
        or receipt.get("version") != VERSION
        or receipt.get("status")
        != "all_four_train_blocks_acquired_before_any_action_or_outcome"
        or receipt.get("source_epoch") != "feverous_p6_e2_formal_v2"
        or receipt.get("source_epoch_rollover_sha256") != expected_rollover
        or receipt.get("train_loader_qualification_sha256")
        != rollover.TRAIN_LOADER_QUALIFICATION_SHA256
        or _semantic_hash(body) != declared
        or receipt.get("source_split") != "TRAIN"
        or receipt.get("selection_secret_persisted_publicly") is not False
        or receipt.get("selection_secret_bytes") != 32
        or not isinstance(receipt.get("selection_secret_sha256"), str)
        or not isinstance(receipt.get("implementation_freeze_sha256"), str)
        or not isinstance(
            receipt.get("identity_full_compile_equivalence_qualification_sha256"),
            str,
        )
        or receipt.get("block_counts") != acquisition.BLOCK_COUNTS
        or receipt.get("F_search_gold_pack_created") is not False
        or receipt.get("all_blocks_one_acquisition") is not True
        or receipt.get("action_retrieval_utility_or_evaluator_calls") != 0
        or receipt.get("development_or_test_source_accessed") is not False
        or receipt.get("online_evaluator_calls") != 0
        or not isinstance(bindings, list)
        or receipt.get("private_file_binding_set_sha256") != _semantic_hash(bindings)
    ):
        raise FeverousFormalAcquisitionError("acquisition receipt drifted")
    for field in (
        "selection_secret_sha256",
        "implementation_freeze_sha256",
        "identity_full_compile_equivalence_qualification_sha256",
    ):
        _require_sha256(receipt.get(field), field)
    if len(bindings) != len(_EXPECTED_BINDING_ROLE_ORDER):
        raise FeverousFormalAcquisitionError("private binding count drifted")
    observed_roles: list[str] = []
    for expected_role, binding in zip(_EXPECTED_BINDING_ROLE_ORDER, bindings):
        if not isinstance(binding, Mapping) or set(binding) != {
            "role",
            "relative_path",
            "size_bytes",
            "file_sha256",
        }:
            raise FeverousFormalAcquisitionError("private binding schema drifted")
        relative = _EXPECTED_BINDING_RELATIVES[expected_role]
        path = root / relative
        try:
            observed = path.lstat()
        except OSError as exc:
            raise FeverousFormalAcquisitionError(
                "private binding metadata drifted"
            ) from exc
        if (
            binding.get("role") != expected_role
            or binding.get("relative_path") != relative.as_posix()
            or type(binding.get("size_bytes")) is not int
            or binding["size_bytes"] < 1
            or path.is_symlink()
            or not stat.S_ISREG(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o600
            or observed.st_size != binding.get("size_bytes")
        ):
            raise FeverousFormalAcquisitionError("private binding identity drifted")
        _require_sha256(binding.get("file_sha256"), "private binding")
        observed_roles.append(expected_role)
    if tuple(observed_roles) != _EXPECTED_BINDING_ROLE_ORDER:
        raise FeverousFormalAcquisitionError("private binding role set drifted")

    marker = _load_canonical_json(root / MARKER_RELATIVE)
    marker_body = dict(marker)
    marker_declared = _require_sha256(
        marker_body.pop("marker_sha256", None), "one-shot marker"
    )
    expected_marker_keys = {
        "schema",
        "version",
        "source_epoch",
        "source_epoch_rollover_sha256",
        "train_loader_qualification_sha256",
        "implementation_freeze_sha256",
        "identity_full_compile_equivalence_qualification_sha256",
        "source_split",
        "retry_replay_or_resample_authorized",
        "marker_sha256",
    }
    if (
        set(marker) != expected_marker_keys
        or marker.get("schema") != f"{VERSION}_one_shot_marker"
        or marker.get("version") != VERSION
        or marker.get("source_epoch") != "feverous_p6_e2_formal_v2"
        or marker.get("source_epoch_rollover_sha256") != expected_rollover
        or marker.get("train_loader_qualification_sha256")
        != rollover.TRAIN_LOADER_QUALIFICATION_SHA256
        or marker.get("implementation_freeze_sha256")
        != receipt.get("implementation_freeze_sha256")
        or marker.get("identity_full_compile_equivalence_qualification_sha256")
        != receipt.get("identity_full_compile_equivalence_qualification_sha256")
        or marker.get("source_split") != "TRAIN"
        or marker.get("retry_replay_or_resample_authorized") is not False
        or _semantic_hash(marker_body) != marker_declared
        or stat.S_IMODE((root / MARKER_RELATIVE).stat().st_mode) != 0o600
    ):
        raise FeverousFormalAcquisitionError("one-shot marker drifted")

    secret_path = root / SECRET_RELATIVE
    if (
        secret_path.is_symlink()
        or not secret_path.is_file()
        or stat.S_IMODE(secret_path.stat().st_mode) != 0o600
    ):
        raise FeverousFormalAcquisitionError("private selection secret is unavailable")
    try:
        secret = secret_path.read_bytes()
    except OSError as exc:
        raise FeverousFormalAcquisitionError(
            "private selection secret cannot be read"
        ) from exc
    if (
        len(secret) != receipt.get("selection_secret_bytes")
        or hashlib.sha256(secret).hexdigest()
        != receipt.get("selection_secret_sha256")
    ):
        raise FeverousFormalAcquisitionError("private selection secret drifted")
    if os.path.lexists(root / FAILURE_RELATIVE):
        raise FeverousFormalAcquisitionError(
            "successful acquisition has a terminal failure record"
        )
    if os.path.lexists(root / F_SEARCH_LABEL_RELATIVE):
        raise FeverousFormalAcquisitionError("F_search label pack exists")
    formal_source.verify_annotation_receipt(receipt["annotation_receipt"])
    formal_source.require_formal_database_page_stream_receipt(
        receipt["database_page_stream_receipt"]
    )
    formal_source.verify_selected_page_lookup_receipt(
        receipt["selected_page_lookup_receipt"]
    )
    source_adapter.verify_adapter_receipt(receipt["source_adapter_receipt"])
    acquisition.verify_formal_corpus_acquisition(receipt["corpus_statistics"])
    return receipt


def verify_acquisition_receipt(project: str | Path) -> Mapping[str, Any]:
    root = _canonical_project(project)
    receipt = verify_acquisition_envelope(root)
    for binding in receipt["private_file_bindings"]:
        path = root / str(binding["relative_path"])
        if _sha256_file(path) != binding.get("file_sha256"):
            raise FeverousFormalAcquisitionError("private binding content drifted")
    return receipt


def _binding_for_role(receipt: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    rows = receipt.get("private_file_bindings")
    if not isinstance(rows, list):
        raise FeverousFormalAcquisitionError("private binding list is absent")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    if len(matches) != 1:
        raise FeverousFormalAcquisitionError("private binding role is absent")
    return matches[0]


def _load_authorized_json_role(
    *, project: Path, receipt: Mapping[str, Any], role: str
) -> Mapping[str, Any]:
    binding = _binding_for_role(receipt, role)
    expected_relative = _EXPECTED_BINDING_RELATIVES.get(role)
    if (
        expected_relative is None
        or binding.get("relative_path") != expected_relative.as_posix()
    ):
        raise FeverousFormalAcquisitionError("authorized role path drifted")
    return _load_canonical_json(
        project / expected_relative,
        expected_sha256=str(binding["file_sha256"]),
    )


def load_corpus_view(project: str | Path) -> Mapping[str, Any]:
    root = _canonical_project(project)
    receipt = verify_acquisition_envelope(root)
    return _load_authorized_json_role(
        project=root, receipt=receipt, role="corpus_view"
    )


def load_block_view(project: str | Path, *, block: str) -> Mapping[str, Any]:
    if block not in acquisition.BLOCK_ORDER:
        raise FeverousFormalAcquisitionError("block view identity is invalid")
    root = _canonical_project(project)
    receipt = verify_acquisition_envelope(root)
    return _load_authorized_json_role(
        project=root, receipt=receipt, role=f"{block}_view"
    )


def load_block_labels(project: str | Path, *, block: str) -> Mapping[str, Any]:
    if block not in {"A_form", "A_hold", "M_search"}:
        raise FeverousFormalAcquisitionError("label pack identity is invalid")
    root = _canonical_project(project)
    receipt = verify_acquisition_envelope(root)
    return _load_authorized_json_role(
        project=root, receipt=receipt, role=f"{block}_labels"
    )


def load_private_secret(project: str | Path) -> bytes:
    root = _canonical_project(project)
    receipt = verify_acquisition_envelope(root)
    path = root / SECRET_RELATIVE
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FeverousFormalAcquisitionError(
            "private selection secret cannot be read"
        ) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or len(raw) != 32
        or hashlib.sha256(raw).hexdigest()
        != receipt.get("selection_secret_sha256")
    ):
        raise FeverousFormalAcquisitionError("private selection secret drifted")
    return raw


__all__ = [
    "AcquisitionPaths",
    "CORPUS_RELATIVE",
    "DATABASE_RELATIVE",
    "FORMAL_ROOT_RELATIVE",
    "FeverousFormalAcquisitionError",
    "LABEL_RELATIVES",
    "MARKER_RELATIVE",
    "RECEIPT_RELATIVE",
    "ROOT_RELATIVE",
    "SECRET_RELATIVE",
    "VERSION",
    "VIEW_RELATIVES",
    "acquisition_paths",
    "load_block_labels",
    "load_block_view",
    "load_corpus_view",
    "load_private_secret",
    "perform_formal_acquisition_once",
    "verify_acquisition_receipt",
    "verify_acquisition_envelope",
]
