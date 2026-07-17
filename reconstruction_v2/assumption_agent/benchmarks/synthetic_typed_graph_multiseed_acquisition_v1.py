"""One-shot custody and acquisition for the post-terminal eight-seed study.

The acquisition side is deliberately score blind.  It obtains one 256-byte OS
random batch, splits it into eight ordered 32-byte seeds, and calls only the
public grammar's ``generate_block(seed, "A_hold")`` entry point.  It then emits
an opaque action pack and a separately sealed late-label pack.  It never runs a
retriever, model, evaluator, formation procedure, or recipe search.

Formal commands require an external actual-HEAD implementation freeze.  Unit
tests use temporary roots and are not formal attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

from . import contractnli_typed_clause_graph_v1 as core
from . import synthetic_typed_graph_causal_grammar_v1 as grammar


VERSION = "synthetic_typed_graph_multiseed_replication_v1"
MODULE_VERSION = "synthetic_typed_graph_multiseed_acquisition_v1"
DESIGN_SCHEMA = "synthetic_typed_graph_multiseed_replication_design_v1"
DESIGN_SHA256 = "584294aced1b5c953629d424b7d8d4fffbe61ca8506ff24541cb1d80fee934ce"
DESIGN_FILE_SHA256 = "ab2bf0e9a0ba759a016a2e8f1e10969441733d974dca0299153378adf63fd746"
GRAMMAR_SHA256 = "acb691846256e87603e92ff079e3db2a1c9df8ea36c193f1dba3c7d35893f9d9"
GRAPH_CORE_SHA256 = "7aef388172c08eecd227033111ce0e92845bca0b514a8bacbff205566963460c"
ORIGINAL_PUBLICATION_FILE_SHA256 = (
    "6186f35949f746ce3497060e2a5f67fd39e1b9582712c17c9c88f7f4f813c392"
)
ORIGINAL_PUBLICATION_SHA256 = (
    "02ab970fec045512f6411347a21161978c5c674c5700c33370b10af07d6aab13"
)
ORIGINAL_SEED_COMMITMENT_SHA256 = (
    "db88f5f76962821874fc70b3b205c9c1a98110ebf74534f1b4b825043bbd7208"
)
ORIGINAL_A_HOLD_COMMITMENT_SET_SHA256 = (
    "7ecbe779b414d4c6c0202b5d68c5f0a4c73bd33039395dce92726b9d2d47e5a4"
)

SEED_COUNT = 8
SEED_BYTES = 32
SEED_BATCH_BYTES = SEED_COUNT * SEED_BYTES
BLOCK = "A_hold"
ITEMS_PER_SEED = 64
TOTAL_ITEM_COUNT = SEED_COUNT * ITEMS_PER_SEED
PRIVATE_MODE = 0o600
PUBLIC_MODE = 0o644

FREEZE_SCHEMA = "synthetic_typed_graph_multiseed_replication_implementation_freeze_v1"
SEED_CUSTODY_SCHEMA = "synthetic_typed_graph_multiseed_replication_seed_custody_v1"
ACQUISITION_SCHEMA = "synthetic_typed_graph_multiseed_replication_acquisition_v1"
ACTION_PACK_SCHEMA = "synthetic_typed_graph_multiseed_action_pack_v1"
ACTION_ITEM_SCHEMA = "synthetic_typed_graph_multiseed_action_item_v1"
LABEL_PACK_SCHEMA = "synthetic_typed_graph_multiseed_label_pack_v1"
LABEL_ITEM_SCHEMA = "synthetic_typed_graph_multiseed_label_item_v1"
COMPILED_COHORT_PACK_SCHEMA = (
    "synthetic_typed_graph_multiseed_compiled_cohort_pack_v1"
)
RESULT_SCHEMA = "synthetic_typed_graph_multiseed_replication_result_v1"
PUBLICATION_SCHEMA = "synthetic_typed_graph_multiseed_terminal_reproducibility_v1"

DESIGN_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_design_v1.json"
)
IMPLEMENTATION_FREEZE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_implementation_freeze_v1.json"
)
ARTIFACT_ROOT_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_multiseed_replication_v1"
)
SEED_MARKER_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "seed_generation.attempt.marker"
SEED_BATCH_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "seed_batch.bin"
SEED_CUSTODY_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_seed_custody_v1.json"
)
SEED_FAILURE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_seed_failure_v1.json"
)
ACQUISITION_MARKER_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "acquisition.attempt.marker"
ACTION_PACK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "action_pack.json"
LABEL_PACK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "label_pack.json"
COMPILED_COHORT_PACK_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "full_compiled_cohort_pack.json"
)
ACQUISITION_RECEIPT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_acquisition_v1.json"
)
RESULT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_result_v1.json"
)
PUBLICATION_MARKER_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "publish_terminal.attempt.marker"
)
PUBLICATION_RELATIVE_PATH = Path(
    "published/synthetic_typed_graph_multiseed_replication_v1/formal_seeds_and_cohort.json"
)
PUBLICATION_FAILURE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_publication_failure_v1.json"
)
RUNNER_VERSION = "synthetic_typed_graph_multiseed_runner_v1"
RUNNER_MARKER_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "runner/formal.attempt.marker"
)
RUNNER_WORK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "runner/formal.work"
RUNNER_ACTION_SEAL_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "runner/formal.action.seal.json"
)
SUCCESS_RESULT_STATUS = "terminal_descriptive_eight_seed_replication_complete"
FAILURE_RESULT_STATUS = (
    "terminal_infrastructure_or_implementation_invalid_no_replay"
)
ORIGINAL_PUBLICATION_RELATIVE_PATH = Path(
    "published/synthetic_typed_graph_causal_v1/formal_seed_and_cohort.json"
)

REQUIRED_FREEZE_PATHS = frozenset(
    {
        DESIGN_RELATIVE_PATH.as_posix(),
        "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_acquisition_v1.py",
        "tests/test_synthetic_typed_graph_multiseed_acquisition_v1.py",
        "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v1.py",
        "tests/test_synthetic_typed_graph_multiseed_runner_v1.py",
        "assumption_agent/benchmarks/synthetic_typed_graph_causal_grammar_v1.py",
        "assumption_agent/benchmarks/contractnli_typed_clause_graph_v1.py",
        "manifests/synthetic_typed_graph_causal_design_v1.json",
        "manifests/synthetic_typed_graph_causal_preseed_amendment_v1.json",
        "manifests/synthetic_typed_graph_causal_implementation_freeze_v1.json",
        "manifests/synthetic_typed_graph_causal_formation_v1.json",
        "manifests/synthetic_typed_graph_causal_A_hold_v1.json",
        ORIGINAL_PUBLICATION_RELATIVE_PATH.as_posix(),
        "manifests/qasper_minilm_runtime_asset_v1.json",
        "manifests/musique_official_hipporag_runtime_attestation_v2.json",
        "manifests/musique_official_hipporag_retrieve_only_binding_v1.json",
        "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
        "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
        "replication_runtime/qasper_minilm_v1/__init__.py",
        "replication_runtime/qasper_minilm_v1/binding.py",
        "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
        "replication_runtime/musique_official_hipporag_v1/contract.py",
        "replication_runtime/musique_official_hipporag_v1/adapter.py",
        "replication_runtime/musique_official_hipporag_v1/binding.py",
        "assumption_agent/models.py",
    }
)

_FORMAL_ENTRY_ACTIVE = False


class SyntheticMultiseedAcquisitionError(RuntimeError):
    """A frozen custody, acquisition, or persistence invariant failed."""


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SyntheticMultiseedAcquisitionError("value is not canonical JSON") from exc


def semantic_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


stable_hash = semantic_hash


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise SyntheticMultiseedAcquisitionError("self-hash field already exists")
    return {**dict(body), field: semantic_hash(dict(body))}


def _assert_no_symlink_components(path: Path, field: str) -> Path:
    absolute = path.expanduser().absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise SyntheticMultiseedAcquisitionError(f"{field} contains a symbolic link")
    return absolute


def _write_exclusive(path: Path, raw: bytes, mode: int) -> str:
    absolute = path.expanduser().absolute()
    if absolute.exists() or absolute.is_symlink():
        raise SyntheticMultiseedAcquisitionError(
            f"exclusive output already exists: {absolute.name}"
        )
    absolute.parent.mkdir(
        parents=True,
        exist_ok=True,
        mode=0o755 if mode == PUBLIC_MODE else 0o700,
    )
    _assert_no_symlink_components(absolute.parent, "output parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(absolute, flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.chmod(absolute, mode)
    parent_descriptor = os.open(absolute.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return hashlib.sha256(raw).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], mode: int) -> str:
    return _write_exclusive(path, canonical_bytes(payload) + b"\n", mode)


def _read_json_with_mode(
    path: Path, *, expected_mode: int, field: str
) -> tuple[dict[str, Any], str]:
    absolute = _assert_no_symlink_components(path, field)
    if (
        not absolute.is_file()
        or absolute.is_symlink()
        or stat.S_IMODE(absolute.stat().st_mode) != expected_mode
    ):
        raise SyntheticMultiseedAcquisitionError(f"{field} mode or type drifted")
    raw = absolute.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticMultiseedAcquisitionError(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SyntheticMultiseedAcquisitionError(f"{field} root drifted")
    return payload, hashlib.sha256(raw).hexdigest()


def _git(project_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SyntheticMultiseedAcquisitionError("Git freeze verification failed") from exc
    return completed.stdout


def _git_project_prefix(project_root: Path) -> str:
    git_root = Path(
        _git(project_root, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    try:
        relative = project_root.resolve().relative_to(git_root)
    except ValueError as exc:
        raise SyntheticMultiseedAcquisitionError("project root is outside Git") from exc
    return "" if relative == Path(".") else relative.as_posix().rstrip("/") + "/"


def _committed_bytes(project_root: Path, relative_path: Path) -> bytes:
    prefix = _git_project_prefix(project_root)
    return _git(project_root, "show", f"HEAD:{prefix}{relative_path.as_posix()}")


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _load_committed_public_json(
    project_root: Path, relative_path: Path, field: str
) -> dict[str, Any]:
    path = _assert_no_symlink_components(project_root / relative_path, field)
    if not path.is_file() or path.is_symlink():
        raise SyntheticMultiseedAcquisitionError(f"{field} is unavailable")
    raw = path.read_bytes()
    if _committed_bytes(project_root, relative_path) != raw:
        raise SyntheticMultiseedAcquisitionError(f"{field} is not current-HEAD committed")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticMultiseedAcquisitionError(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SyntheticMultiseedAcquisitionError(f"{field} root drifted")
    return payload


def verify_frozen_design(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    path = _assert_no_symlink_components(root / DESIGN_RELATIVE_PATH, "design")
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != DESIGN_FILE_SHA256
    ):
        raise SyntheticMultiseedAcquisitionError("frozen multiseed design drifted")
    try:
        design = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticMultiseedAcquisitionError("frozen design is unreadable") from exc
    if not isinstance(design, dict) or design.get("schema") != DESIGN_SCHEMA:
        raise SyntheticMultiseedAcquisitionError("frozen design schema drifted")
    body = dict(design)
    declared = body.pop("design_sha256", None)
    if declared != DESIGN_SHA256 or semantic_hash(body) != declared:
        raise SyntheticMultiseedAcquisitionError("frozen design self-hash drifted")
    seed_contract = design.get("seed_contract")
    cohort_contract = design.get("cohort_contract")
    if (
        not isinstance(seed_contract, Mapping)
        or seed_contract.get("original_published_seed_commitment_sha256")
        != ORIGINAL_SEED_COMMITMENT_SHA256
        or not isinstance(cohort_contract, Mapping)
        or cohort_contract.get("block") != BLOCK
        or cohort_contract.get("total_items") != TOTAL_ITEM_COUNT
    ):
        raise SyntheticMultiseedAcquisitionError("frozen design semantic binding drifted")
    return design


def _formal_output_paths() -> tuple[Path, ...]:
    return (
        SEED_MARKER_RELATIVE_PATH,
        SEED_BATCH_RELATIVE_PATH,
        SEED_CUSTODY_RELATIVE_PATH,
        SEED_FAILURE_RELATIVE_PATH,
        ACQUISITION_MARKER_RELATIVE_PATH,
        ACTION_PACK_RELATIVE_PATH,
        LABEL_PACK_RELATIVE_PATH,
        COMPILED_COHORT_PACK_RELATIVE_PATH,
        ACQUISITION_RECEIPT_RELATIVE_PATH,
        RESULT_RELATIVE_PATH,
        PUBLICATION_MARKER_RELATIVE_PATH,
        PUBLICATION_RELATIVE_PATH,
        PUBLICATION_FAILURE_RELATIVE_PATH,
        RUNNER_MARKER_RELATIVE_PATH,
        RUNNER_WORK_RELATIVE_PATH,
        RUNNER_ACTION_SEAL_RELATIVE_PATH,
    )


def create_implementation_freeze(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    verify_frozen_design(root)
    _git_project_prefix(root)
    output = root / IMPLEMENTATION_FREEZE_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise SyntheticMultiseedAcquisitionError("implementation freeze already exists")
    if any((root / relative).exists() for relative in _formal_output_paths()):
        raise SyntheticMultiseedAcquisitionError("formal multiseed output already exists")
    bindings: list[dict[str, str]] = []
    for relative in sorted(REQUIRED_FREEZE_PATHS):
        path = _assert_no_symlink_components(root / relative, "freeze binding")
        if not path.is_file() or path.is_symlink():
            raise SyntheticMultiseedAcquisitionError(
                f"required freeze file unavailable: {relative}"
            )
        worktree = path.read_bytes()
        committed = _committed_bytes(root, Path(relative))
        if worktree != committed:
            raise SyntheticMultiseedAcquisitionError(
                f"required freeze file is not HEAD-clean: {relative}"
            )
        bindings.append(
            {
                "relative_path": relative,
                "file_sha256": hashlib.sha256(committed).hexdigest(),
                "git_blob_sha1": _git_blob_sha1(committed),
            }
        )
    body = {
        "schema": FREEZE_SCHEMA,
        "version": VERSION,
        "status": "complete_preseed_implementation_frozen_must_commit_before_seed",
        "creation_HEAD": _git(root, "rev-parse", "HEAD").decode("ascii").strip(),
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "formal_seed_or_cohort_exists": False,
        "bindings": bindings,
    }
    freeze = _self_hashed(body, "implementation_freeze_sha256")
    _write_json_exclusive(output, freeze, PUBLIC_MODE)
    return freeze


def verify_implementation_freeze(project_root: Path) -> tuple[dict[str, Any], str]:
    root = project_root.resolve(strict=True)
    actual_head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    freeze = _load_committed_public_json(
        root, IMPLEMENTATION_FREEZE_RELATIVE_PATH, "implementation freeze"
    )
    body = dict(freeze)
    declared = body.pop("implementation_freeze_sha256", None)
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("status")
        != "complete_preseed_implementation_frozen_must_commit_before_seed"
        or freeze.get("design_sha256") != DESIGN_SHA256
        or freeze.get("design_file_sha256") != DESIGN_FILE_SHA256
        or freeze.get("formal_seed_or_cohort_exists") is not False
        or not isinstance(declared, str)
        or semantic_hash(body) != declared
    ):
        raise SyntheticMultiseedAcquisitionError("implementation freeze drifted")
    rows = freeze.get("bindings")
    if not isinstance(rows, list):
        raise SyntheticMultiseedAcquisitionError("implementation bindings are absent")
    by_path: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "file_sha256",
            "git_blob_sha1",
        }:
            raise SyntheticMultiseedAcquisitionError("implementation binding schema drifted")
        relative = row.get("relative_path")
        if not isinstance(relative, str) or relative in by_path:
            raise SyntheticMultiseedAcquisitionError("implementation binding path drifted")
        by_path[relative] = row
    if set(by_path) != set(REQUIRED_FREEZE_PATHS):
        raise SyntheticMultiseedAcquisitionError("implementation freeze path set drifted")
    for relative, row in by_path.items():
        path = _assert_no_symlink_components(root / relative, "implementation binding")
        if not path.is_file() or path.is_symlink():
            raise SyntheticMultiseedAcquisitionError("implementation-bound file unavailable")
        worktree = path.read_bytes()
        committed = _committed_bytes(root, Path(relative))
        if worktree != committed:
            raise SyntheticMultiseedAcquisitionError(
                "implementation-bound worktree differs from HEAD"
            )
        if (
            row.get("file_sha256") != hashlib.sha256(committed).hexdigest()
            or row.get("git_blob_sha1") != _git_blob_sha1(committed)
        ):
            raise SyntheticMultiseedAcquisitionError("implementation binding hash drifted")
    return freeze, actual_head


def _require_formal_entry() -> None:
    if _FORMAL_ENTRY_ACTIVE is not True:
        raise SyntheticMultiseedAcquisitionError(
            "formal operation may only be consumed by the acquisition CLI"
        )


def create_seed_custody(*, project_root: Path) -> dict[str, Any]:
    """Consume the sole entropy call and persist public commitments only."""

    _require_formal_entry()
    root = project_root.resolve(strict=True)
    verify_frozen_design(root)
    freeze, actual_head = verify_implementation_freeze(root)
    forbidden = tuple(root / relative for relative in _formal_output_paths()[1:])
    marker_path = root / SEED_MARKER_RELATIVE_PATH
    if marker_path.exists() or marker_path.is_symlink() or any(
        path.exists() or path.is_symlink() for path in forbidden
    ):
        raise SyntheticMultiseedAcquisitionError("canonical seed attempt already exists")
    marker = _self_hashed(
        {
            "schema": f"{SEED_CUSTODY_SCHEMA}_attempt_marker",
            "version": VERSION,
            "status": "sole_eight_seed_batch_generation_attempt_consumed",
            "actual_HEAD": actual_head,
            "design_sha256": DESIGN_SHA256,
            "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
            "attempt_count": 1,
            "entropy_call_bytes": SEED_BATCH_BYTES,
        },
        "marker_sha256",
    )
    marker_file_hash = _write_json_exclusive(marker_path, marker, PUBLIC_MODE)
    try:
        # The only entropy access is strictly after the durable attempt marker.
        batch = os.urandom(SEED_BATCH_BYTES)
        if not isinstance(batch, bytes) or len(batch) != SEED_BATCH_BYTES:
            raise SyntheticMultiseedAcquisitionError(
                "OS random source did not return exactly 256 bytes"
            )
        seeds = tuple(
            batch[index * SEED_BYTES : (index + 1) * SEED_BYTES]
            for index in range(SEED_COUNT)
        )
        commitments = [hashlib.sha256(seed).hexdigest() for seed in seeds]
        if len(set(seeds)) != SEED_COUNT:
            raise SyntheticMultiseedAcquisitionError(
                "duplicate fresh seeds make the formal attempt terminal"
            )
        if ORIGINAL_SEED_COMMITMENT_SHA256 in commitments:
            raise SyntheticMultiseedAcquisitionError(
                "fresh seed collides with the original published seed"
            )
        batch_commitment = hashlib.sha256(batch).hexdigest()
        persisted = _write_exclusive(
            root / SEED_BATCH_RELATIVE_PATH, batch, PRIVATE_MODE
        )
        if persisted != batch_commitment:
            raise SyntheticMultiseedAcquisitionError("seed batch persistence drifted")
        custody = _self_hashed(
            {
                "schema": SEED_CUSTODY_SCHEMA,
                "version": VERSION,
                "status": "eight_fresh_seeds_committed_cohort_not_generated",
                "design_sha256": DESIGN_SHA256,
                "implementation_freeze_sha256": freeze[
                    "implementation_freeze_sha256"
                ],
                "seed_attempt_marker_sha256": marker["marker_sha256"],
                "seed_attempt_marker_file_sha256": marker_file_hash,
                "seed_count": SEED_COUNT,
                "seed_bytes_each": SEED_BYTES,
                "seed_batch_bytes": SEED_BATCH_BYTES,
                "seed_batch_commitment_sha256": batch_commitment,
                "ordered_seed_commitments_sha256": commitments,
                "original_seed_commitment_sha256": ORIGINAL_SEED_COMMITMENT_SHA256,
                "seed_generation": "one_os.urandom_256_call_then_ordered_8x32_split_after_marker",
                "seed_material_published": False,
                "cohort_generated": False,
                "attempts_allowed": 1,
                "retry_replacement_or_interim_allowed": False,
            },
            "custody_sha256",
        )
        _write_json_exclusive(
            root / SEED_CUSTODY_RELATIVE_PATH, custody, PUBLIC_MODE
        )
        return custody
    except Exception as exc:
        failure = _self_hashed(
            {
                "schema": f"{SEED_CUSTODY_SCHEMA}_failure_receipt",
                "version": VERSION,
                "status": "terminal_seed_batch_invalid_no_replacement",
                "design_sha256": DESIGN_SHA256,
                "implementation_freeze_sha256": freeze[
                    "implementation_freeze_sha256"
                ],
                "seed_attempt_marker_sha256": marker["marker_sha256"],
                "failure_class": type(exc).__name__,
                "secret_material_or_exception_message_persisted_publicly": False,
                "retry_replacement_or_smaller_N_authorized": False,
            },
            "receipt_sha256",
        )
        failure_path = root / SEED_FAILURE_RELATIVE_PATH
        if not failure_path.exists():
            _write_json_exclusive(failure_path, failure, PUBLIC_MODE)
        raise


def load_seed_custody(path: Path) -> dict[str, Any]:
    absolute = _assert_no_symlink_components(path, "seed custody")
    root = absolute.parent.parent
    if absolute != root / SEED_CUSTODY_RELATIVE_PATH:
        raise SyntheticMultiseedAcquisitionError("seed custody path is not canonical")
    custody, _file_hash = _read_json_with_mode(
        absolute, expected_mode=PUBLIC_MODE, field="seed custody"
    )
    body = dict(custody)
    declared = body.pop("custody_sha256", None)
    commitments = custody.get("ordered_seed_commitments_sha256")
    if (
        custody.get("schema") != SEED_CUSTODY_SCHEMA
        or custody.get("status")
        != "eight_fresh_seeds_committed_cohort_not_generated"
        or custody.get("design_sha256") != DESIGN_SHA256
        or custody.get("seed_count") != SEED_COUNT
        or custody.get("seed_bytes_each") != SEED_BYTES
        or custody.get("seed_batch_bytes") != SEED_BATCH_BYTES
        or custody.get("original_seed_commitment_sha256")
        != ORIGINAL_SEED_COMMITMENT_SHA256
        or custody.get("seed_material_published") is not False
        or custody.get("cohort_generated") is not False
        or custody.get("attempts_allowed") != 1
        or custody.get("retry_replacement_or_interim_allowed") is not False
        or not isinstance(declared, str)
        or semantic_hash(body) != declared
        or not isinstance(commitments, list)
        or len(commitments) != SEED_COUNT
        or len(set(commitments)) != SEED_COUNT
        or ORIGINAL_SEED_COMMITMENT_SHA256 in commitments
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in commitments
        )
    ):
        raise SyntheticMultiseedAcquisitionError("seed custody binding drifted")
    batch_path = _assert_no_symlink_components(
        root / SEED_BATCH_RELATIVE_PATH, "seed batch"
    )
    if (
        not batch_path.is_file()
        or batch_path.is_symlink()
        or batch_path.stat().st_size != SEED_BATCH_BYTES
        or stat.S_IMODE(batch_path.stat().st_mode) != PRIVATE_MODE
    ):
        raise SyntheticMultiseedAcquisitionError("seed batch mode or size drifted")
    marker, marker_file_hash = _read_json_with_mode(
        root / SEED_MARKER_RELATIVE_PATH,
        expected_mode=PUBLIC_MODE,
        field="seed attempt marker",
    )
    marker_body = dict(marker)
    marker_hash = marker_body.pop("marker_sha256", None)
    if (
        marker.get("schema") != f"{SEED_CUSTODY_SCHEMA}_attempt_marker"
        or marker.get("status") != "sole_eight_seed_batch_generation_attempt_consumed"
        or marker.get("attempt_count") != 1
        or marker.get("entropy_call_bytes") != SEED_BATCH_BYTES
        or marker.get("design_sha256") != DESIGN_SHA256
        or marker.get("implementation_freeze_sha256")
        != custody.get("implementation_freeze_sha256")
        or not isinstance(marker_hash, str)
        or semantic_hash(marker_body) != marker_hash
        or custody.get("seed_attempt_marker_sha256") != marker_hash
        or custody.get("seed_attempt_marker_file_sha256") != marker_file_hash
    ):
        raise SyntheticMultiseedAcquisitionError("seed attempt marker chain drifted")
    return custody


def _read_seed_batch(path: Path, custody: Mapping[str, Any]) -> tuple[bytes, ...]:
    absolute = _assert_no_symlink_components(path, "seed batch")
    if (
        not absolute.is_file()
        or absolute.is_symlink()
        or absolute.stat().st_size != SEED_BATCH_BYTES
        or stat.S_IMODE(absolute.stat().st_mode) != PRIVATE_MODE
    ):
        raise SyntheticMultiseedAcquisitionError("seed batch mode or size drifted")
    raw = absolute.read_bytes()
    if hashlib.sha256(raw).hexdigest() != custody.get(
        "seed_batch_commitment_sha256"
    ):
        raise SyntheticMultiseedAcquisitionError("seed batch commitment drifted")
    seeds = tuple(
        raw[index * SEED_BYTES : (index + 1) * SEED_BYTES]
        for index in range(SEED_COUNT)
    )
    commitments = [hashlib.sha256(seed).hexdigest() for seed in seeds]
    if commitments != custody.get("ordered_seed_commitments_sha256"):
        raise SyntheticMultiseedAcquisitionError("ordered seed commitments drifted")
    if len(set(seeds)) != SEED_COUNT or ORIGINAL_SEED_COMMITMENT_SHA256 in commitments:
        raise SyntheticMultiseedAcquisitionError("seed collision detected after custody")
    return seeds


def _load_original_A_hold_commitments_after_marker(root: Path) -> frozenset[str]:
    path = _assert_no_symlink_components(
        root / ORIGINAL_PUBLICATION_RELATIVE_PATH, "original public publication"
    )
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != ORIGINAL_PUBLICATION_FILE_SHA256
    ):
        raise SyntheticMultiseedAcquisitionError("original publication file drifted")
    try:
        publication = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticMultiseedAcquisitionError(
            "original publication is unreadable"
        ) from exc
    if not isinstance(publication, dict):
        raise SyntheticMultiseedAcquisitionError("original publication root drifted")
    body = dict(publication)
    declared = body.pop("reproducibility_sha256", None)
    blocks = publication.get("blocks")
    rows = blocks.get(BLOCK) if isinstance(blocks, Mapping) else None
    if (
        publication.get("schema")
        != "synthetic_typed_graph_causal_terminal_reproducibility_v1"
        or publication.get("status")
        != "terminal_seed_and_full_compiled_cohort_published"
        or publication.get("seed_commitment_sha256")
        != ORIGINAL_SEED_COMMITMENT_SHA256
        or declared != ORIGINAL_PUBLICATION_SHA256
        or semantic_hash(body) != declared
        or not isinstance(rows, list)
        or len(rows) != ITEMS_PER_SEED
    ):
        raise SyntheticMultiseedAcquisitionError("original publication binding drifted")
    commitments = [
        row.get("item_commitment_sha256") if isinstance(row, Mapping) else None
        for row in rows
    ]
    if (
        any(not isinstance(value, str) or len(value) != 64 for value in commitments)
        or len(set(commitments)) != ITEMS_PER_SEED
        or stable_hash(commitments) != ORIGINAL_A_HOLD_COMMITMENT_SET_SHA256
    ):
        raise SyntheticMultiseedAcquisitionError(
            "original A_hold commitment projection drifted"
        )
    return frozenset(commitments)


def _validate_compiled_item(item: grammar.CompiledItem, ordinal: int) -> None:
    if item.block != BLOCK or item.block_ordinal != ordinal:
        raise SyntheticMultiseedAcquisitionError("compiled A_hold ordering drifted")
    spans = tuple(
        core.SourceSpan(node.span_i, node.start, node.end, node.identity_text)
        for node in item.nodes
    )
    full = {
        (edge.edge_family, edge.left_span_i, edge.right_span_i)
        for edge in core.build_typed_clause_graph(spans)
    }
    designated = {
        (edge.edge_family, edge.left_span_i, edge.right_span_i)
        for edge in item.designated_edges
    }
    if not designated or not designated.issubset(full):
        raise SyntheticMultiseedAcquisitionError(
            "designated edges are not a subset of the frozen full graph"
        )


def _action_row(
    item: grammar.CompiledItem, *, seed_index: int, global_ordinal: int
) -> dict[str, Any]:
    body = {
        "schema": ACTION_ITEM_SCHEMA,
        "global_ordinal": global_ordinal,
        "seed_index": seed_index,
        "seed_ordinal": item.block_ordinal,
        "question": item.question,
        "context": item.context,
        "nodes": [
            {
                "span_i": node.span_i,
                "start": node.start,
                "end": node.end,
                "identity_text": node.identity_text,
            }
            for node in item.nodes
        ],
        "designated_edges": [
            {
                "edge_family": edge.edge_family,
                "left_span_i": edge.left_span_i,
                "right_span_i": edge.right_span_i,
            }
            for edge in item.designated_edges
        ],
    }
    return _self_hashed(body, "action_item_sha256")


def _label_row(
    item: grammar.CompiledItem,
    *,
    action_item_sha256: str,
    seed_index: int,
    global_ordinal: int,
) -> dict[str, Any]:
    body = {
        "schema": LABEL_ITEM_SCHEMA,
        "global_ordinal": global_ordinal,
        "seed_index": seed_index,
        "seed_ordinal": item.block_ordinal,
        "action_item_sha256": action_item_sha256,
        "gold_node_indices": list(item.gold_node_indices),
        "family_id": item.family_id,
        "family_role": item.family_role,
        "polarity": item.polarity,
        "edge_family": item.edge_family,
    }
    return _self_hashed(body, "label_item_sha256")


def _pack(
    *, schema: str, items: Sequence[Mapping[str, Any]], labels_included: bool | None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": schema,
        "version": VERSION,
        "block": BLOCK,
        "seed_count": SEED_COUNT,
        "item_count_per_seed": ITEMS_PER_SEED,
        "total_item_count": TOTAL_ITEM_COUNT,
    }
    if labels_included is not None:
        body["labels_included"] = labels_included
    body["items"] = list(items)
    return _self_hashed(body, "pack_sha256")


def acquire_formal_cohort(*, project_root: Path) -> dict[str, Any]:
    """Generate exactly 8x64 A_hold rows after consuming the one-shot marker."""

    _require_formal_entry()
    root = project_root.resolve(strict=True)
    verify_frozen_design(root)
    freeze, actual_head = verify_implementation_freeze(root)
    committed_custody = _load_committed_public_json(
        root, SEED_CUSTODY_RELATIVE_PATH, "seed custody"
    )
    custody = load_seed_custody(root / SEED_CUSTODY_RELATIVE_PATH)
    if committed_custody != custody or custody.get(
        "implementation_freeze_sha256"
    ) != freeze.get("implementation_freeze_sha256"):
        raise SyntheticMultiseedAcquisitionError("committed custody chain drifted")
    outputs = (
        root / ACQUISITION_MARKER_RELATIVE_PATH,
        root / ACTION_PACK_RELATIVE_PATH,
        root / LABEL_PACK_RELATIVE_PATH,
        root / COMPILED_COHORT_PACK_RELATIVE_PATH,
        root / ACQUISITION_RECEIPT_RELATIVE_PATH,
        root / RESULT_RELATIVE_PATH,
        root / RUNNER_MARKER_RELATIVE_PATH,
        root / RUNNER_WORK_RELATIVE_PATH,
        root / RUNNER_ACTION_SEAL_RELATIVE_PATH,
        root / PUBLICATION_MARKER_RELATIVE_PATH,
        root / PUBLICATION_RELATIVE_PATH,
    )
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise SyntheticMultiseedAcquisitionError("canonical acquisition output exists")
    marker = _self_hashed(
        {
            "schema": f"{ACQUISITION_SCHEMA}_attempt_marker",
            "version": VERSION,
            "status": "sole_multiseed_A_hold_acquisition_attempt_consumed",
            "actual_HEAD": actual_head,
            "design_sha256": DESIGN_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "custody_sha256": custody["custody_sha256"],
            "seed_batch_commitment_sha256": custody[
                "seed_batch_commitment_sha256"
            ],
            "attempt_count": 1,
        },
        "marker_sha256",
    )
    marker_file_hash = _write_json_exclusive(
        root / ACQUISITION_MARKER_RELATIVE_PATH, marker, PRIVATE_MODE
    )
    try:
        # Neither the private batch nor the public original cohort is opened
        # until the durable acquisition marker exists.
        seeds = _read_seed_batch(root / SEED_BATCH_RELATIVE_PATH, custody)
        original_commitments = _load_original_A_hold_commitments_after_marker(root)
        compiled: list[tuple[int, grammar.CompiledItem]] = []
        for seed_index, seed in enumerate(seeds):
            items = grammar.generate_block(seed, BLOCK)
            if len(items) != ITEMS_PER_SEED:
                raise SyntheticMultiseedAcquisitionError(
                    "public grammar did not return exactly 64 A_hold rows"
                )
            for seed_ordinal, item in enumerate(items):
                _validate_compiled_item(item, seed_ordinal)
                compiled.append((seed_index, item))
        if len(compiled) != TOTAL_ITEM_COUNT:
            raise SyntheticMultiseedAcquisitionError("compiled cohort count drifted")
        item_commitments = [item.item_commitment_sha256 for _, item in compiled]
        if len(set(item_commitments)) != TOTAL_ITEM_COUNT:
            raise SyntheticMultiseedAcquisitionError(
                "new cohort item commitments overlap each other"
            )
        if original_commitments.intersection(item_commitments):
            raise SyntheticMultiseedAcquisitionError(
                "new cohort overlaps the original published A_hold cohort"
            )
        action_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        compiled_rows: list[dict[str, Any]] = []
        for global_ordinal, (seed_index, item) in enumerate(compiled):
            action = _action_row(
                item, seed_index=seed_index, global_ordinal=global_ordinal
            )
            label = _label_row(
                item,
                action_item_sha256=action["action_item_sha256"],
                seed_index=seed_index,
                global_ordinal=global_ordinal,
            )
            action_rows.append(action)
            label_rows.append(label)
            compiled_body = _compiled_public_row(
                item,
                seed_index=seed_index,
                global_ordinal=global_ordinal,
            )
            compiled_rows.append(
                {
                    **compiled_body,
                    "compiled_row_sha256": semantic_hash(compiled_body),
                }
            )
        action_pack = _pack(
            schema=ACTION_PACK_SCHEMA, items=action_rows, labels_included=False
        )
        label_pack = _pack(
            schema=LABEL_PACK_SCHEMA, items=label_rows, labels_included=None
        )
        compiled_pack = _pack(
            schema=COMPILED_COHORT_PACK_SCHEMA,
            items=compiled_rows,
            labels_included=True,
        )
        action_file_hash = _write_json_exclusive(
            root / ACTION_PACK_RELATIVE_PATH, action_pack, PRIVATE_MODE
        )
        label_file_hash = _write_json_exclusive(
            root / LABEL_PACK_RELATIVE_PATH, label_pack, PRIVATE_MODE
        )
        compiled_file_hash = _write_json_exclusive(
            root / COMPILED_COHORT_PACK_RELATIVE_PATH,
            compiled_pack,
            PRIVATE_MODE,
        )
        commitments = {
            "action_pack_file_sha256": action_file_hash,
            "action_item_commitment_set_sha256": stable_hash(
                [row["action_item_sha256"] for row in action_rows]
            ),
            "label_pack_file_sha256": label_file_hash,
            "label_item_commitment_set_sha256": stable_hash(
                [row["label_item_sha256"] for row in label_rows]
            ),
            "compiled_cohort_pack_file_sha256": compiled_file_hash,
            "compiled_row_commitment_set_sha256": stable_hash(
                [row["compiled_row_sha256"] for row in compiled_rows]
            ),
        }
        receipt = _self_hashed(
            {
                "schema": ACQUISITION_SCHEMA,
                "version": VERSION,
                "status": "formal_multiseed_A_hold_cohort_acquired_private_labels_separated",
                "design_sha256": DESIGN_SHA256,
                "design_file_sha256": DESIGN_FILE_SHA256,
                "grammar_sha256": GRAMMAR_SHA256,
                "graph_core_sha256": GRAPH_CORE_SHA256,
                "implementation_freeze_sha256": freeze[
                    "implementation_freeze_sha256"
                ],
                "custody_sha256": custody["custody_sha256"],
                "seed_batch_commitment_sha256": custody[
                    "seed_batch_commitment_sha256"
                ],
                "ordered_seed_commitments_sha256": custody[
                    "ordered_seed_commitments_sha256"
                ],
                "attempt_marker_sha256": marker["marker_sha256"],
                "attempt_marker_file_sha256": marker_file_hash,
                "block": BLOCK,
                "seed_count": SEED_COUNT,
                "item_count_per_seed": ITEMS_PER_SEED,
                "total_item_count": TOTAL_ITEM_COUNT,
                "generated_item_commitment_set_sha256": stable_hash(
                    item_commitments
                ),
                "original_A_hold_commitment_set_sha256": ORIGINAL_A_HOLD_COMMITMENT_SET_SHA256,
                "new_and_original_A_hold_commitments_disjoint": True,
                "fixed_recipe_id": "R1_DEFINITION_1SWAP",
                "arms": ["RAW", "official_HippoRAG", "Agent_R1"],
                "commitments": commitments,
                "label_rows_persisted_publicly": False,
                "seed_material_published": False,
                "formation_candidate_pool_filter_or_recipe_search_used": False,
                "network_calls": 0,
                "retry_replacement_or_interim_allowed": False,
            },
            "receipt_sha256",
        )
        _write_json_exclusive(
            root / ACQUISITION_RECEIPT_RELATIVE_PATH, receipt, PUBLIC_MODE
        )
        return receipt
    except Exception as exc:
        failure = _self_hashed(
            {
                "schema": ACQUISITION_SCHEMA,
                "version": VERSION,
                "status": "terminal_multiseed_acquisition_invalid_no_replay",
                "design_sha256": DESIGN_SHA256,
                "implementation_freeze_sha256": freeze[
                    "implementation_freeze_sha256"
                ],
                "custody_sha256": custody["custody_sha256"],
                "attempt_marker_sha256": marker["marker_sha256"],
                "failure_class": type(exc).__name__,
                "exception_message_seed_or_item_rows_persisted_publicly": False,
                "retry_replacement_smaller_N_or_overlap_repair_authorized": False,
            },
            "receipt_sha256",
        )
        receipt_path = root / ACQUISITION_RECEIPT_RELATIVE_PATH
        if not receipt_path.exists():
            _write_json_exclusive(receipt_path, failure, PUBLIC_MODE)
        raise


def _verify_pack(
    root: Path,
    *,
    relative_path: Path,
    schema: str,
    item_hash_field: str,
    expected_file_hash: object,
    expected_set_hash: object,
) -> dict[str, Any]:
    pack, file_hash = _read_json_with_mode(
        root / relative_path, expected_mode=PRIVATE_MODE, field=schema
    )
    body = dict(pack)
    declared = body.pop("pack_sha256", None)
    rows = pack.get("items")
    if (
        pack.get("schema") != schema
        or pack.get("version") != VERSION
        or pack.get("block") != BLOCK
        or pack.get("seed_count") != SEED_COUNT
        or pack.get("item_count_per_seed") != ITEMS_PER_SEED
        or pack.get("total_item_count") != TOTAL_ITEM_COUNT
        or not isinstance(declared, str)
        or semantic_hash(body) != declared
        or file_hash != expected_file_hash
        or not isinstance(rows, list)
        or len(rows) != TOTAL_ITEM_COUNT
    ):
        raise SyntheticMultiseedAcquisitionError(f"{schema} binding drifted")
    hashes: list[str] = []
    for global_ordinal, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or row.get("global_ordinal") != global_ordinal
            or row.get("seed_index") != global_ordinal // ITEMS_PER_SEED
            or row.get("seed_ordinal") != global_ordinal % ITEMS_PER_SEED
        ):
            raise SyntheticMultiseedAcquisitionError(f"{schema} ordering drifted")
        row_body = dict(row)
        row_hash = row_body.pop(item_hash_field, None)
        if not isinstance(row_hash, str) or semantic_hash(row_body) != row_hash:
            raise SyntheticMultiseedAcquisitionError(f"{schema} row hash drifted")
        hashes.append(row_hash)
    if stable_hash(hashes) != expected_set_hash:
        raise SyntheticMultiseedAcquisitionError(f"{schema} ordered set drifted")
    return pack


_COMPILED_PUBLIC_ROW_FIELDS = frozenset(
    {
        "schema",
        "global_ordinal",
        "seed_index",
        "seed_ordinal",
        "block",
        "block_ordinal",
        "family_slot",
        "family_id",
        "family_role",
        "template_split",
        "polarity",
        "negative_kind",
        "edge_family",
        "pair_key",
        "item_commitment_sha256",
        "label_free_commitment_sha256",
        "matching_signature_sha256",
        "structural_draw_sha256",
        "question",
        "context",
        "nodes",
        "gold_node_indices",
        "designated_edges",
        "endpoint_permutation",
    }
)


def _verify_compiled_cohort_pack(
    root: Path,
    *,
    expected_file_hash: object,
    expected_row_set_hash: object,
    expected_item_set_hash: object,
) -> dict[str, Any]:
    pack, file_hash = _read_json_with_mode(
        root / COMPILED_COHORT_PACK_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="private full compiled cohort pack",
    )
    body = dict(pack)
    declared = body.pop("pack_sha256", None)
    rows = pack.get("items")
    if (
        set(pack)
        != {
            "schema",
            "version",
            "block",
            "seed_count",
            "item_count_per_seed",
            "total_item_count",
            "labels_included",
            "items",
            "pack_sha256",
        }
        or pack.get("schema") != COMPILED_COHORT_PACK_SCHEMA
        or pack.get("version") != VERSION
        or pack.get("block") != BLOCK
        or pack.get("seed_count") != SEED_COUNT
        or pack.get("item_count_per_seed") != ITEMS_PER_SEED
        or pack.get("total_item_count") != TOTAL_ITEM_COUNT
        or pack.get("labels_included") is not True
        or not isinstance(declared, str)
        or semantic_hash(body) != declared
        or file_hash != expected_file_hash
        or not isinstance(rows, list)
        or len(rows) != TOTAL_ITEM_COUNT
    ):
        raise SyntheticMultiseedAcquisitionError(
            "private full compiled cohort pack binding drifted"
        )
    row_hashes: list[str] = []
    item_hashes: list[str] = []
    for global_ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SyntheticMultiseedAcquisitionError(
                "private full compiled cohort row drifted"
            )
        row_body = dict(row)
        row_hash = row_body.pop("compiled_row_sha256", None)
        expected_seed, expected_within = divmod(global_ordinal, ITEMS_PER_SEED)
        item_hash = row.get("item_commitment_sha256")
        if (
            set(row_body) != _COMPILED_PUBLIC_ROW_FIELDS
            or row.get("schema") != f"{grammar.VERSION}_compiled_item"
            or row.get("global_ordinal") != global_ordinal
            or row.get("seed_index") != expected_seed
            or row.get("seed_ordinal") != expected_within
            or row.get("block") != BLOCK
            or row.get("block_ordinal") != expected_within
            or not isinstance(row_hash, str)
            or semantic_hash(row_body) != row_hash
            or not isinstance(item_hash, str)
            or len(item_hash) != 64
        ):
            raise SyntheticMultiseedAcquisitionError(
                "private full compiled cohort row binding drifted"
            )
        row_hashes.append(row_hash)
        item_hashes.append(item_hash)
    if (
        len(set(row_hashes)) != TOTAL_ITEM_COUNT
        or len(set(item_hashes)) != TOTAL_ITEM_COUNT
        or stable_hash(row_hashes) != expected_row_set_hash
        or stable_hash(item_hashes) != expected_item_set_hash
    ):
        raise SyntheticMultiseedAcquisitionError(
            "private full compiled cohort ordered commitments drifted"
        )
    return pack


def load_committed_acquisition_receipt(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    freeze, _actual_head = verify_implementation_freeze(root)
    receipt = _load_committed_public_json(
        root, ACQUISITION_RECEIPT_RELATIVE_PATH, "acquisition receipt"
    )
    body = dict(receipt)
    declared = body.pop("receipt_sha256", None)
    commitments = receipt.get("commitments")
    if (
        receipt.get("schema") != ACQUISITION_SCHEMA
        or receipt.get("status")
        != "formal_multiseed_A_hold_cohort_acquired_private_labels_separated"
        or receipt.get("design_sha256") != DESIGN_SHA256
        or receipt.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or receipt.get("block") != BLOCK
        or receipt.get("seed_count") != SEED_COUNT
        or receipt.get("item_count_per_seed") != ITEMS_PER_SEED
        or receipt.get("total_item_count") != TOTAL_ITEM_COUNT
        or receipt.get("new_and_original_A_hold_commitments_disjoint") is not True
        or receipt.get("fixed_recipe_id") != "R1_DEFINITION_1SWAP"
        or receipt.get("arms") != ["RAW", "official_HippoRAG", "Agent_R1"]
        or not isinstance(commitments, Mapping)
        or set(commitments) != {
            "action_pack_file_sha256",
            "action_item_commitment_set_sha256",
            "label_pack_file_sha256",
            "label_item_commitment_set_sha256",
            "compiled_cohort_pack_file_sha256",
            "compiled_row_commitment_set_sha256",
        }
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in commitments.values()
        )
        or not isinstance(declared, str)
        or semantic_hash(body) != declared
    ):
        raise SyntheticMultiseedAcquisitionError("acquisition receipt drifted")
    custody = _load_committed_public_json(
        root, SEED_CUSTODY_RELATIVE_PATH, "seed custody"
    )
    if custody != load_seed_custody(root / SEED_CUSTODY_RELATIVE_PATH) or (
        receipt.get("custody_sha256") != custody.get("custody_sha256")
        or receipt.get("seed_batch_commitment_sha256")
        != custody.get("seed_batch_commitment_sha256")
        or receipt.get("ordered_seed_commitments_sha256")
        != custody.get("ordered_seed_commitments_sha256")
    ):
        raise SyntheticMultiseedAcquisitionError("acquisition custody chain drifted")
    _verify_pack(
        root,
        relative_path=ACTION_PACK_RELATIVE_PATH,
        schema=ACTION_PACK_SCHEMA,
        item_hash_field="action_item_sha256",
        expected_file_hash=commitments.get("action_pack_file_sha256"),
        expected_set_hash=commitments.get("action_item_commitment_set_sha256"),
    )
    _verify_pack(
        root,
        relative_path=LABEL_PACK_RELATIVE_PATH,
        schema=LABEL_PACK_SCHEMA,
        item_hash_field="label_item_sha256",
        expected_file_hash=commitments.get("label_pack_file_sha256"),
        expected_set_hash=commitments.get("label_item_commitment_set_sha256"),
    )
    return receipt


def _compiled_public_row(
    item: grammar.CompiledItem, *, seed_index: int, global_ordinal: int
) -> dict[str, Any]:
    return {
        "schema": item.schema,
        "global_ordinal": global_ordinal,
        "seed_index": seed_index,
        "seed_ordinal": item.block_ordinal,
        "block": item.block,
        "block_ordinal": item.block_ordinal,
        "family_slot": item.family_slot,
        "family_id": item.family_id,
        "family_role": item.family_role,
        "template_split": item.template_split,
        "polarity": item.polarity,
        "negative_kind": item.negative_kind,
        "edge_family": item.edge_family,
        "pair_key": item.pair_key,
        "item_commitment_sha256": item.item_commitment_sha256,
        "label_free_commitment_sha256": item.label_free_commitment_sha256,
        "matching_signature_sha256": item.matching_signature_sha256,
        "structural_draw_sha256": item.structural_draw_sha256,
        "question": item.question,
        "context": item.context,
        "nodes": [
            {
                "span_i": node.span_i,
                "start": node.start,
                "end": node.end,
                "identity_text": node.identity_text,
                "latent_role": node.latent_role,
            }
            for node in item.nodes
        ],
        "gold_node_indices": list(item.gold_node_indices),
        "designated_edges": [
            {
                "edge_family": edge.edge_family,
                "left_span_i": edge.left_span_i,
                "right_span_i": edge.right_span_i,
            }
            for edge in item.designated_edges
        ],
        "endpoint_permutation": [list(pair) for pair in item.endpoint_permutation],
    }


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_committed_terminal_result(
    root: Path,
    *,
    freeze: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    committed = _load_committed_public_json(root, RESULT_RELATIVE_PATH, "terminal result")
    result, result_file_hash = _read_json_with_mode(
        root / RESULT_RELATIVE_PATH,
        expected_mode=PUBLIC_MODE,
        field="terminal result",
    )
    if result != committed:
        raise SyntheticMultiseedAcquisitionError("terminal result readback drifted")
    result_body = dict(result)
    result_hash = result_body.pop("receipt_sha256", None)
    status = result.get("status")
    acquisition_file_hash = sha256_file(root / ACQUISITION_RECEIPT_RELATIVE_PATH)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("version") != VERSION
        or status not in {SUCCESS_RESULT_STATUS, FAILURE_RESULT_STATUS}
        or result.get("design_sha256") != DESIGN_SHA256
        or result.get("design_file_sha256") != DESIGN_FILE_SHA256
        or result.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or result.get("acquisition_receipt_sha256")
        != acquisition.get("receipt_sha256")
        or result.get("acquisition_receipt_file_sha256") != acquisition_file_hash
        or result.get("generated_item_commitment_set_sha256")
        != acquisition.get("generated_item_commitment_set_sha256")
        or not _is_lower_hex(result.get("invocation_HEAD"), 40)
        or not _is_lower_hex(result_hash, 64)
        or semantic_hash(result_body) != result_hash
    ):
        raise SyntheticMultiseedAcquisitionError("terminal result binding drifted")

    marker, marker_file_hash = _read_json_with_mode(
        root / RUNNER_MARKER_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="runner formal attempt marker",
    )
    marker_body = dict(marker)
    marker_hash = marker_body.pop("marker_sha256", None)
    if (
        marker.get("schema") != f"{RUNNER_VERSION}_formal_attempt_marker"
        or marker.get("version") != RUNNER_VERSION
        or marker.get("status") != "sole_formal_replication_attempt_consumed"
        or marker.get("actual_HEAD") != result.get("invocation_HEAD")
        or marker.get("design_sha256") != DESIGN_SHA256
        or marker.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or marker.get("acquisition_receipt_sha256")
        != acquisition.get("receipt_sha256")
        or marker.get("acquisition_receipt_file_sha256") != acquisition_file_hash
        or marker.get("attempt_count") != 1
        or marker.get("private_packs_opened_before_marker") is not False
        or not _is_lower_hex(marker_hash, 64)
        or semantic_hash(marker_body) != marker_hash
        or result.get("formal_attempt_marker_sha256") != marker_hash
        or result.get("formal_attempt_marker_file_sha256") != marker_file_hash
    ):
        raise SyntheticMultiseedAcquisitionError("runner attempt marker chain drifted")

    seal_path = root / RUNNER_ACTION_SEAL_RELATIVE_PATH
    declared_seal_file_hash = result.get("action_seal_file_sha256")
    if status == SUCCESS_RESULT_STATUS:
        hashes = (
            "action_pack_file_sha256",
            "action_pack_sha256",
            "action_item_commitment_set_sha256",
            "label_pack_file_sha256",
            "label_pack_sha256",
            "label_item_commitment_set_sha256",
            "runtime_binding_sha256",
            "action_table_sha256",
            "action_seal_sha256",
            "action_seal_file_sha256",
        )
        if (
            result.get("block") != BLOCK
            or result.get("recipe_id") != "R1_DEFINITION_1SWAP"
            or result.get("seed_count") != SEED_COUNT
            or result.get("item_count_per_seed") != ITEMS_PER_SEED
            or result.get("total_item_count") != TOTAL_ITEM_COUNT
            or result.get("arms") != ["RAW", "official_HippoRAG", "Agent_R1"]
            or result.get("action_work_unit_count") != TOTAL_ITEM_COUNT * 3
            or result.get("official_retrieve_action_count") != TOTAL_ITEM_COUNT
            or result.get("interpretation")
            != "descriptive_fixed_cohort_replication_only"
            or result.get("seeds_or_item_rows_disclosed") is not False
            or result.get("result_must_be_committed_before_terminal_publication")
            is not True
            or not isinstance(result.get("aggregates"), Mapping)
            or not isinstance(result.get("cluster_differences"), Mapping)
            or any(not _is_lower_hex(result.get(field), 64) for field in hashes)
        ):
            raise SyntheticMultiseedAcquisitionError(
                "successful terminal result schema drifted"
            )
    else:
        if (
            not isinstance(result.get("failure_class"), str)
            or not result.get("failure_class")
            or result.get("retry_replacement_or_backup_attempt_authorized") is not False
            or result.get(
                "exception_message_seed_item_or_label_content_persisted_publicly"
            )
            is not False
            or result.get("result_must_be_committed_before_terminal_publication")
            is not True
            or declared_seal_file_hash is not None
            and not _is_lower_hex(declared_seal_file_hash, 64)
        ):
            raise SyntheticMultiseedAcquisitionError(
                "failed terminal result schema drifted"
            )

    if declared_seal_file_hash is None:
        if status == SUCCESS_RESULT_STATUS or seal_path.exists() or seal_path.is_symlink():
            raise SyntheticMultiseedAcquisitionError("runner action seal chain drifted")
    else:
        seal, seal_file_hash = _read_json_with_mode(
            seal_path,
            expected_mode=PRIVATE_MODE,
            field="runner private action seal",
        )
        seal_body = dict(seal)
        seal_hash = seal_body.pop("action_seal_sha256", None)
        if (
            seal.get("schema") != f"{RUNNER_VERSION}_private_action_seal"
            or seal.get("version") != RUNNER_VERSION
            or seal.get("status")
            != "all_1536_actions_joined_official_postflight_terminal"
            or seal.get("item_count") != TOTAL_ITEM_COUNT
            or seal.get("action_work_unit_count") != TOTAL_ITEM_COUNT * 3
            or seal.get("labels_opened_before_action_seal") is not False
            or seal.get("labels_opened_before_seal") is not False
            or not _is_lower_hex(seal_hash, 64)
            or semantic_hash(seal_body) != seal_hash
            or seal_file_hash != declared_seal_file_hash
            or status == SUCCESS_RESULT_STATUS
            and result.get("action_seal_sha256") != seal_hash
        ):
            raise SyntheticMultiseedAcquisitionError("runner action seal chain drifted")
    return result, result_file_hash


def publish_terminal(*, project_root: Path) -> dict[str, Any]:
    """Publish exact seeds and 512 compiled rows after one committed terminal."""

    _require_formal_entry()
    root = project_root.resolve(strict=True)
    freeze, actual_head = verify_implementation_freeze(root)
    acquisition = load_committed_acquisition_receipt(root)
    result, result_file_hash = _load_committed_terminal_result(
        root,
        freeze=freeze,
        acquisition=acquisition,
    )
    result_hash = str(result["receipt_sha256"])
    outputs = (
        root / PUBLICATION_MARKER_RELATIVE_PATH,
        root / PUBLICATION_RELATIVE_PATH,
        root / PUBLICATION_FAILURE_RELATIVE_PATH,
    )
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise SyntheticMultiseedAcquisitionError("publication attempt already exists")
    custody = _load_committed_public_json(
        root, SEED_CUSTODY_RELATIVE_PATH, "seed custody"
    )
    marker = _self_hashed(
        {
            "schema": f"{PUBLICATION_SCHEMA}_attempt_marker",
            "version": VERSION,
            "status": "sole_terminal_reproducibility_publication_attempt_consumed",
            "actual_HEAD": actual_head,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "terminal_result_receipt_sha256": result_hash,
            "seed_batch_commitment_sha256": custody[
                "seed_batch_commitment_sha256"
            ],
            "attempt_count": 1,
        },
        "marker_sha256",
    )
    marker_file_hash = _write_json_exclusive(
        root / PUBLICATION_MARKER_RELATIVE_PATH, marker, PRIVATE_MODE
    )
    try:
        seeds = _read_seed_batch(root / SEED_BATCH_RELATIVE_PATH, custody)
        commitments = acquisition.get("commitments")
        if not isinstance(commitments, Mapping):
            raise SyntheticMultiseedAcquisitionError(
                "publication acquisition commitments drifted"
            )
        compiled_pack = _verify_compiled_cohort_pack(
            root,
            expected_file_hash=commitments.get(
                "compiled_cohort_pack_file_sha256"
            ),
            expected_row_set_hash=commitments.get(
                "compiled_row_commitment_set_sha256"
            ),
            expected_item_set_hash=acquisition.get(
                "generated_item_commitment_set_sha256"
            ),
        )
        rows: list[dict[str, Any]] = []
        item_commitments: list[str] = []
        for stored in compiled_pack["items"]:
            row = dict(stored)
            row.pop("compiled_row_sha256")
            rows.append(row)
            item_commitments.append(str(row["item_commitment_sha256"]))
        artifact = _self_hashed(
            {
                "schema": PUBLICATION_SCHEMA,
                "version": VERSION,
                "status": "terminal_eight_seeds_and_full_compiled_cohort_published",
                "publication_HEAD": actual_head,
                "design_sha256": DESIGN_SHA256,
                "implementation_freeze_sha256": freeze[
                    "implementation_freeze_sha256"
                ],
                "acquisition_receipt_sha256": acquisition["receipt_sha256"],
                "terminal_result_receipt_sha256": result_hash,
                "terminal_result_file_sha256": result_file_hash,
                "publication_attempt_marker_sha256": marker["marker_sha256"],
                "publication_attempt_marker_file_sha256": marker_file_hash,
                "seed_encoding": "ordered_lowercase_hex_exact_32_bytes_each",
                "formal_seed_hexes": [seed.hex() for seed in seeds],
                "seed_batch_commitment_sha256": hashlib.sha256(
                    b"".join(seeds)
                ).hexdigest(),
                "ordered_seed_commitments_sha256": [
                    hashlib.sha256(seed).hexdigest() for seed in seeds
                ],
                "block": BLOCK,
                "seed_count": SEED_COUNT,
                "item_count_per_seed": ITEMS_PER_SEED,
                "total_item_count": TOTAL_ITEM_COUNT,
                "generated_item_commitment_set_sha256": stable_hash(
                    item_commitments
                ),
                "items": rows,
                "retrieval_actions_model_outputs_or_scores_included": False,
            },
            "reproducibility_sha256",
        )
        _write_json_exclusive(
            root / PUBLICATION_RELATIVE_PATH, artifact, PUBLIC_MODE
        )
        return artifact
    except Exception as exc:
        failure = _self_hashed(
            {
                "schema": f"{PUBLICATION_SCHEMA}_failure_receipt",
                "version": VERSION,
                "status": "terminal_publication_invalid_no_replay",
                "marker_sha256": marker["marker_sha256"],
                "terminal_result_receipt_sha256": result_hash,
                "failure_class": type(exc).__name__,
                "exception_message_seed_or_item_rows_persisted_publicly": False,
                "retry_or_replay_authorized": False,
            },
            "receipt_sha256",
        )
        failure_path = root / PUBLICATION_FAILURE_RELATIVE_PATH
        if not failure_path.exists():
            _write_json_exclusive(failure_path, failure, PUBLIC_MODE)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "seed-custody", "acquire", "publish-terminal"),
    )
    parser.add_argument("--project-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    global _FORMAL_ENTRY_ACTIVE
    if arguments.command == "freeze":
        result = create_implementation_freeze(arguments.project_root)
        result_hash = result["implementation_freeze_sha256"]
    else:
        if _FORMAL_ENTRY_ACTIVE:
            raise SyntheticMultiseedAcquisitionError("formal entry is already active")
        _FORMAL_ENTRY_ACTIVE = True
        try:
            if arguments.command == "seed-custody":
                result = create_seed_custody(project_root=arguments.project_root)
                result_hash = result["custody_sha256"]
            elif arguments.command == "acquire":
                result = acquire_formal_cohort(project_root=arguments.project_root)
                result_hash = result["receipt_sha256"]
            else:
                result = publish_terminal(project_root=arguments.project_root)
                result_hash = result["reproducibility_sha256"]
        finally:
            _FORMAL_ENTRY_ACTIVE = False
    print(
        json.dumps(
            {
                "command": arguments.command,
                "status": result["status"],
                "result_sha256": result_hash,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "ACQUISITION_RECEIPT_RELATIVE_PATH",
    "ACQUISITION_SCHEMA",
    "ACTION_ITEM_SCHEMA",
    "ACTION_PACK_RELATIVE_PATH",
    "ACTION_PACK_SCHEMA",
    "BLOCK",
    "COMPILED_COHORT_PACK_RELATIVE_PATH",
    "COMPILED_COHORT_PACK_SCHEMA",
    "DESIGN_FILE_SHA256",
    "DESIGN_SHA256",
    "ITEMS_PER_SEED",
    "LABEL_ITEM_SCHEMA",
    "LABEL_PACK_RELATIVE_PATH",
    "LABEL_PACK_SCHEMA",
    "PRIVATE_MODE",
    "PUBLIC_MODE",
    "SEED_BATCH_BYTES",
    "SEED_BYTES",
    "SEED_COUNT",
    "SyntheticMultiseedAcquisitionError",
    "TOTAL_ITEM_COUNT",
    "VERSION",
    "acquire_formal_cohort",
    "canonical_bytes",
    "create_implementation_freeze",
    "create_seed_custody",
    "load_committed_acquisition_receipt",
    "load_seed_custody",
    "publish_terminal",
    "semantic_hash",
    "sha256_file",
    "stable_hash",
    "verify_frozen_design",
    "verify_implementation_freeze",
]


if __name__ == "__main__":
    raise SystemExit(main())
