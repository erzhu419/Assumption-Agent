"""One-shot formal pack formation for the frozen FEVEROUS P6/E2 study.

The module is the only filesystem bridge between controlled TRAIN custody and
the later lifecycle.  It creates every block in one acquisition before any
action, keeps claims/corpus/gold in mode-600 private packs, and publishes only
aggregate receipts and content hashes.  It has no retrieval, evaluator, or
online-service dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any

from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks import feverous_p6_e2_formal_source_v1 as formal_source
from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as source_adapter


VERSION = "feverous_p6_e2_formal_acquisition_v1"
ROOT_RELATIVE = Path("artifacts/feverous_p6_e2_formal_v1/acquisition")
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
SOURCE_ROOT_RELATIVE = Path("artifacts/feverous_official_source_v1")
ANNOTATION_RELATIVE = SOURCE_ROOT_RELATIVE / formal_source.FROZEN_ANNOTATION_BASENAME
DATABASE_RELATIVE = SOURCE_ROOT_RELATIVE / formal_source.FROZEN_DATABASE_BASENAME

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FeverousFormalAcquisitionError(RuntimeError):
    """A one-shot source, secret, private pack, or receipt invariant drifted."""


def _canonical_bytes(value: object) -> bytes:
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
        raise FeverousFormalAcquisitionError("value is not canonical JSON") from exc


def _semantic_hash(value: object) -> str:
    raw = _canonical_bytes(value)
    return hashlib.sha256(raw[:-1]).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FeverousFormalAcquisitionError("bound file cannot be hashed") from exc
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FeverousFormalAcquisitionError(f"{field} is not a SHA-256")
    return value


def _canonical_project(project: str | Path) -> Path:
    try:
        root = Path(project).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FeverousFormalAcquisitionError("project root is unavailable") from exc
    if not root.is_dir() or root.is_symlink():
        raise FeverousFormalAcquisitionError("project root is unsafe")
    return root


def _private_root(project: Path) -> Path:
    path = project / ROOT_RELATIVE
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise FeverousFormalAcquisitionError("private acquisition root failed") from exc
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o700:
        raise FeverousFormalAcquisitionError("private acquisition root mode drifted")
    return path


def _exclusive_bytes(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise FeverousFormalAcquisitionError("private output parent is unsafe")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FeverousFormalAcquisitionError("exclusive private output failed") from exc


def _exclusive_json(path: Path, value: object) -> None:
    _exclusive_bytes(path, _canonical_bytes(value))


def _load_canonical_json(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FeverousFormalAcquisitionError("private pack is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverousFormalAcquisitionError("private pack is invalid") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise FeverousFormalAcquisitionError("private pack is not canonical JSON")
    if expected_sha256 is not None and _sha256_bytes(raw) != expected_sha256:
        raise FeverousFormalAcquisitionError("private pack file hash drifted")
    return value


def _file_binding(project: Path, relative: Path, *, role: str) -> dict[str, object]:
    path = project / relative
    try:
        observed = path.lstat()
    except OSError as exc:
        raise FeverousFormalAcquisitionError("private pack is unavailable") from exc
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        raise FeverousFormalAcquisitionError("private pack mode drifted")
    return {
        "role": role,
        "relative_path": relative.as_posix(),
        "size_bytes": observed.st_size,
        "file_sha256": _sha256_file(path),
    }


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
    implementation_freeze_sha256: str,
    identity_full_compile_equivalence_qualification_sha256: str,
) -> Mapping[str, Any]:
    """Generate the sole formal cohort and every sealed pack exactly once."""

    root = _canonical_project(project)
    implementation_sha = _require_sha256(
        implementation_freeze_sha256, "implementation freeze"
    )
    equivalence_sha = _require_sha256(
        identity_full_compile_equivalence_qualification_sha256,
        "identity/full-compiler qualification",
    )
    _private_root(root)
    paths = acquisition_paths(root)
    marker_body = {
        "schema": f"{VERSION}_one_shot_marker",
        "version": VERSION,
        "implementation_freeze_sha256": implementation_sha,
        "identity_full_compile_equivalence_qualification_sha256": equivalence_sha,
        "source_split": "TRAIN",
    }
    _exclusive_json(
        paths.marker,
        {**marker_body, "marker_sha256": _semantic_hash(marker_body)},
    )
    annotation_path = root / ANNOTATION_RELATIVE
    database_path = root / DATABASE_RELATIVE
    try:
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
    """Verify public semantics and private metadata without opening sealed packs."""

    root = _canonical_project(project)
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
        or _semantic_hash(body) != declared
        or receipt.get("source_split") != "TRAIN"
        or receipt.get("selection_secret_persisted_publicly") is not False
        or receipt.get("selection_secret_bytes") != 32
        or not isinstance(receipt.get("selection_secret_sha256"), str)
        or _SHA256.fullmatch(str(receipt.get("selection_secret_sha256"))) is None
        or not isinstance(receipt.get("implementation_freeze_sha256"), str)
        or _SHA256.fullmatch(str(receipt.get("implementation_freeze_sha256"))) is None
        or not isinstance(
            receipt.get(
                "identity_full_compile_equivalence_qualification_sha256"
            ),
            str,
        )
        or _SHA256.fullmatch(
            str(
                receipt.get(
                    "identity_full_compile_equivalence_qualification_sha256"
                )
            )
        )
        is None
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
        role = binding.get("role")
        relative_text = binding.get("relative_path")
        size_bytes = binding.get("size_bytes")
        file_sha256 = binding.get("file_sha256")
        if (
            role != expected_role
            or not isinstance(relative_text, str)
            or relative_text
            != _EXPECTED_BINDING_RELATIVES[expected_role].as_posix()
            or type(size_bytes) is not int
            or size_bytes < 1
            or not isinstance(file_sha256, str)
            or _SHA256.fullmatch(file_sha256) is None
        ):
            raise FeverousFormalAcquisitionError("private binding identity drifted")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise FeverousFormalAcquisitionError("private binding path is unsafe")
        path = root / relative
        try:
            observed = path.lstat()
        except OSError:
            observed = None
        if (
            role in observed_roles
            or path.is_symlink()
            or not path.is_file()
            or observed is None
            or not stat.S_ISREG(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o600
            or path.stat().st_size != binding.get("size_bytes")
        ):
            raise FeverousFormalAcquisitionError("private binding metadata drifted")
        observed_roles.append(role)
    if tuple(observed_roles) != _EXPECTED_BINDING_ROLE_ORDER:
        raise FeverousFormalAcquisitionError("private binding role set drifted")

    marker = _load_canonical_json(root / MARKER_RELATIVE)
    marker_body = dict(marker)
    marker_declared = _require_sha256(
        marker_body.pop("marker_sha256", None), "one-shot marker"
    )
    if (
        set(marker)
        != {
            "schema",
            "version",
            "implementation_freeze_sha256",
            "identity_full_compile_equivalence_qualification_sha256",
            "source_split",
            "marker_sha256",
        }
        or marker.get("schema") != f"{VERSION}_one_shot_marker"
        or marker.get("version") != VERSION
        or marker.get("source_split") != "TRAIN"
        or marker.get("implementation_freeze_sha256")
        != receipt.get("implementation_freeze_sha256")
        or marker.get("identity_full_compile_equivalence_qualification_sha256")
        != receipt.get(
            "identity_full_compile_equivalence_qualification_sha256"
        )
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
    """Fully hash every private role; use only at outcome-free acquisition time."""

    root = _canonical_project(project)
    receipt = verify_acquisition_envelope(root)
    for binding in receipt["private_file_bindings"]:
        assert isinstance(binding, Mapping)
        path = root / str(binding["relative_path"])
        if _sha256_file(path) != binding.get("file_sha256"):
            raise FeverousFormalAcquisitionError("private binding content drifted")
    return receipt


def _binding_for_role(receipt: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    rows = receipt.get("private_file_bindings")
    if not isinstance(rows, list):
        raise FeverousFormalAcquisitionError("private binding list is absent")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("role") == role]
    if len(matches) != 1:
        raise FeverousFormalAcquisitionError("private binding role is absent")
    return matches[0]


def _load_authorized_json_role(
    *, project: Path, receipt: Mapping[str, Any], role: str
) -> Mapping[str, Any]:
    """Hash and decode exactly one role after its lifecycle authorization."""

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
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise FeverousFormalAcquisitionError("private selection secret is unavailable")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FeverousFormalAcquisitionError("private selection secret cannot be read") from exc
    if len(raw) != 32 or hashlib.sha256(raw).hexdigest() != receipt.get(
        "selection_secret_sha256"
    ):
        raise FeverousFormalAcquisitionError("private selection secret drifted")
    return raw


__all__ = [
    "AcquisitionPaths",
    "CORPUS_RELATIVE",
    "DATABASE_RELATIVE",
    "FeverousFormalAcquisitionError",
    "LABEL_RELATIVES",
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
