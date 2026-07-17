"""One-seed private cohort acquisition for the frozen synthetic causal study.

The public grammar creates a deterministic 4x64 cohort from exactly one secret
seed.  This module is the custody boundary: it creates (or verifies) the seed,
compiles every slot once, applies the preregistered graph interventions, and
separates label-free views from late-opened labels.  It performs no retrieval,
embedding, scoring, candidate selection, or online operation.

Formal callers must commit this module, its tests, and the runner before calling
``create_seed_custody``.  Unit tests use temporary paths only; they do not
constitute a formal attempt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

from . import contractnli_typed_clause_graph_v1 as core
from . import synthetic_typed_graph_causal_grammar_v1 as grammar


VERSION = "synthetic_typed_graph_causal_acquisition_v1"
DESIGN_SCHEMA = "synthetic_typed_graph_causal_design_v1"
DESIGN_SHA256 = "b1ca0187c5e24ee33a67ab8abc6b9b6abbb22acad4fb08015a0fa4054055517a"
DESIGN_FILE_SHA256 = "44926c4c5455508b16a867df49a01dc5eb8b0fdf339a29bd39275b3faa92bc31"
GRAMMAR_SHA256 = "acb691846256e87603e92ff079e3db2a1c9df8ea36c193f1dba3c7d35893f9d9"
GRAPH_CORE_SHA256 = "7aef388172c08eecd227033111ce0e92845bca0b514a8bacbff205566963460c"
AMENDMENT_SHA256 = "f96260b0b504422a6cda2b029bb5ac071e9f9166fff79b1573b0e05c8b288338"
AMENDMENT_FILE_SHA256 = "6234f8041d5f73f33b2432dae853aa0a851d49f9faf0c75b14bf0f5be1c1e870"

SEED_BYTES = 32
PRIVATE_MODE = 0o600
PUBLIC_MODE = 0o644
MAX_PRIVATE_BYTES = 32 * 1024 * 1024

CUSTODY_SCHEMA = f"{VERSION}_seed_custody"
ATTEMPT_SCHEMA = f"{VERSION}_attempt_marker"
VIEW_SCHEMA = f"{VERSION}_label_free_block"
VIEW_ITEM_SCHEMA = f"{VERSION}_label_free_item"
LABEL_SCHEMA = f"{VERSION}_label_block"
LABEL_ITEM_SCHEMA = f"{VERSION}_label_item"
RECEIPT_SCHEMA = f"{VERSION}_public_receipt"

IMPLEMENTATION_FREEZE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_causal_implementation_freeze_v1.json"
)
SEED_MARKER_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_causal_v1/seed_generation.attempt.marker"
)
SEED_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_causal_v1/private/formal_seed.bin"
)
SEED_CUSTODY_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_causal_seed_custody_v1.json"
)
SEED_FAILURE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_causal_seed_failure_v1.json"
)
COHORT_MARKER_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_causal_v1/cohort.attempt.marker"
)
PRIVATE_COHORT_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_causal_v1/private/cohort"
)
ACQUISITION_RECEIPT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_causal_acquisition_v1.json"
)
REPRODUCIBILITY_RELATIVE_PATH = Path(
    "published/synthetic_typed_graph_causal_v1/formal_seed_and_cohort.json"
)
REPRODUCIBILITY_MARKER_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_causal_v1/publish_reproducibility.attempt.marker"
)
REPRODUCIBILITY_FAILURE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_causal_reproducibility_failure_v1.json"
)
IMPLEMENTATION_FREEZE_SCHEMA = "synthetic_typed_graph_causal_implementation_freeze_v1"
REQUIRED_FREEZE_PATHS = frozenset(
    {
        "manifests/synthetic_typed_graph_causal_design_v1.json",
        "manifests/synthetic_typed_graph_causal_preseed_amendment_v1.json",
        "assumption_agent/benchmarks/synthetic_typed_graph_causal_grammar_v1.py",
        "tests/test_synthetic_typed_graph_causal_grammar_v1.py",
        "assumption_agent/benchmarks/contractnli_typed_clause_graph_v1.py",
        "tests/test_contractnli_typed_clause_graph_v1.py",
        "assumption_agent/benchmarks/synthetic_typed_graph_causal_acquisition_v1.py",
        "assumption_agent/benchmarks/synthetic_typed_graph_causal_runner_v1.py",
        "tests/test_synthetic_typed_graph_causal_formal_v1.py",
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
_FORMAL_ACQUISITION_ENTRY_ACTIVE = False
EXPECTED_PRIVATE_COHORT_FILES = frozenset(
    {f"{block}.label_free.sealed.json" for block in grammar.BLOCK_ORDER}
    | {f"{block}.labels.sealed.json" for block in ("A_form", "A_hold", "M_search")}
)
FORMAL_STAGE_FORBIDDEN_RELATIVE_PATHS = tuple(
    [
        Path(f"artifacts/synthetic_typed_graph_causal_v1/runner/{stage}")
        for stage in ("formation", "A_hold", "M_search")
    ]
    + [
        Path(f"manifests/synthetic_typed_graph_causal_{stage}_v1.json")
        for stage in ("formation", "A_hold", "M_search")
    ]
    + [
        Path(f"manifests/synthetic_typed_graph_causal_{stage}_failure_v1.json")
        for stage in ("formation", "A_hold", "M_search")
    ]
)


class SyntheticCausalAcquisitionError(RuntimeError):
    """A formal custody, cohort, or persistence invariant failed."""


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
        raise SyntheticCausalAcquisitionError("value is not canonical JSON") from exc


def semantic_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise SyntheticCausalAcquisitionError("self-hash field already exists")
    return {**dict(body), field: semantic_hash(dict(body))}


def _assert_no_symlink_components(path: Path, field: str) -> Path:
    absolute = path.expanduser().absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise SyntheticCausalAcquisitionError(f"{field} contains a symbolic link")
    return absolute


def _write_exclusive(path: Path, raw: bytes, mode: int) -> str:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise SyntheticCausalAcquisitionError(f"exclusive output already exists: {path.name}")
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
        mode=0o755 if mode == PUBLIC_MODE else 0o700,
    )
    _assert_no_symlink_components(path.parent, "output parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    os.chmod(path, mode)
    parent_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return hashlib.sha256(raw).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], mode: int) -> str:
    return _write_exclusive(path, canonical_bytes(payload) + b"\n", mode)


def verify_frozen_design(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    paths = {
        "design": root / "manifests/synthetic_typed_graph_causal_design_v1.json",
        "amendment": root
        / "manifests/synthetic_typed_graph_causal_preseed_amendment_v1.json",
        "grammar": root
        / "assumption_agent/benchmarks/synthetic_typed_graph_causal_grammar_v1.py",
        "core": root / "assumption_agent/benchmarks/contractnli_typed_clause_graph_v1.py",
    }
    expected = {
        "design": DESIGN_FILE_SHA256,
        "amendment": AMENDMENT_FILE_SHA256,
        "grammar": GRAMMAR_SHA256,
        "core": GRAPH_CORE_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected[name]:
            raise SyntheticCausalAcquisitionError(f"frozen {name} binding drifted")
    try:
        design = json.loads(paths["design"].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticCausalAcquisitionError("frozen design is unreadable") from exc
    if not isinstance(design, dict) or design.get("schema") != DESIGN_SCHEMA:
        raise SyntheticCausalAcquisitionError("frozen design schema drifted")
    body = dict(design)
    declared = body.pop("design_sha256", None)
    if declared != DESIGN_SHA256 or semantic_hash(body) != declared:
        raise SyntheticCausalAcquisitionError("frozen design self-hash drifted")
    binding = design.get("implementation_binding")
    if not isinstance(binding, Mapping) or not isinstance(binding.get("grammar"), Mapping):
        raise SyntheticCausalAcquisitionError("frozen grammar design binding is absent")
    if binding["grammar"].get("file_sha256") != GRAMMAR_SHA256:
        raise SyntheticCausalAcquisitionError("frozen grammar design binding drifted")
    try:
        amendment = json.loads(paths["amendment"].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticCausalAcquisitionError("pre-seed amendment is unreadable") from exc
    if not isinstance(amendment, dict):
        raise SyntheticCausalAcquisitionError("pre-seed amendment root drifted")
    amendment_body = dict(amendment)
    amendment_declared = amendment_body.pop("amendment_sha256", None)
    if amendment_declared != AMENDMENT_SHA256 or semantic_hash(amendment_body) != amendment_declared:
        raise SyntheticCausalAcquisitionError("pre-seed amendment self-hash drifted")
    return design


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
        raise SyntheticCausalAcquisitionError("Git freeze verification failed") from exc
    return completed.stdout


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _git_project_prefix(project_root: Path) -> str:
    git_root = Path(
        _git(project_root, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    try:
        relative = project_root.resolve().relative_to(git_root)
    except ValueError as exc:
        raise SyntheticCausalAcquisitionError("project root is outside the Git tree") from exc
    return "" if relative == Path(".") else relative.as_posix().rstrip("/") + "/"


def _committed_bytes(project_root: Path, relative_path: Path) -> bytes:
    project_prefix = _git_project_prefix(project_root)
    return _git(
        project_root,
        "show",
        f"HEAD:{project_prefix}{relative_path.as_posix()}",
    )


def verify_implementation_freeze(project_root: Path) -> tuple[dict[str, Any], str]:
    """Verify one external self-hashed freeze against actual current HEAD bytes."""

    root = project_root.resolve(strict=True)
    _git_project_prefix(root)
    actual_head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    freeze_path = _assert_no_symlink_components(
        root / IMPLEMENTATION_FREEZE_RELATIVE_PATH, "implementation freeze"
    )
    if not freeze_path.is_file() or freeze_path.is_symlink():
        raise SyntheticCausalAcquisitionError("implementation freeze is unavailable")
    raw = freeze_path.read_bytes()
    if _committed_bytes(root, IMPLEMENTATION_FREEZE_RELATIVE_PATH) != raw:
        raise SyntheticCausalAcquisitionError("implementation freeze is not current-HEAD committed")
    try:
        freeze = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticCausalAcquisitionError("implementation freeze is invalid JSON") from exc
    if not isinstance(freeze, dict) or freeze.get("schema") != IMPLEMENTATION_FREEZE_SCHEMA:
        raise SyntheticCausalAcquisitionError("implementation freeze schema drifted")
    body = dict(freeze)
    declared = body.pop("implementation_freeze_sha256", None)
    if not isinstance(declared, str) or semantic_hash(body) != declared:
        raise SyntheticCausalAcquisitionError("implementation freeze self-hash drifted")
    if (
        freeze.get("design_sha256") != DESIGN_SHA256
        or freeze.get("amendment_sha256") != AMENDMENT_SHA256
        or freeze.get("formal_seed_or_cohort_exists") is not False
    ):
        raise SyntheticCausalAcquisitionError("implementation freeze semantic binding drifted")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, list):
        raise SyntheticCausalAcquisitionError("implementation freeze bindings are absent")
    by_path: dict[str, Mapping[str, Any]] = {}
    for row in bindings:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path", "file_sha256", "git_blob_sha1"
        }:
            raise SyntheticCausalAcquisitionError("implementation binding schema drifted")
        relative = row.get("relative_path")
        if not isinstance(relative, str) or relative in by_path:
            raise SyntheticCausalAcquisitionError("implementation binding path drifted")
        by_path[relative] = row
    if set(by_path) != set(REQUIRED_FREEZE_PATHS):
        raise SyntheticCausalAcquisitionError("implementation freeze path set drifted")
    for relative, row in by_path.items():
        path = _assert_no_symlink_components(root / relative, "implementation binding")
        if not path.is_file() or path.is_symlink():
            raise SyntheticCausalAcquisitionError("implementation-bound file is unavailable")
        worktree_raw = path.read_bytes()
        head_raw = _committed_bytes(root, Path(relative))
        if worktree_raw != head_raw:
            raise SyntheticCausalAcquisitionError("implementation-bound worktree differs from HEAD")
        if row.get("file_sha256") != hashlib.sha256(head_raw).hexdigest() or (
            row.get("git_blob_sha1") != _git_blob_sha1(head_raw)
        ):
            raise SyntheticCausalAcquisitionError("implementation-bound hash drifted")
    return freeze, actual_head


def create_implementation_freeze(project_root: Path) -> dict[str, Any]:
    """Create the canonical external freeze from clean actual-HEAD file bytes.

    This is a pre-seed operation.  The resulting manifest must itself be
    committed before ``create_seed_custody`` can verify and consume the sole
    seed-generation attempt.
    """

    root = project_root.resolve(strict=True)
    verify_frozen_design(root)
    _git_project_prefix(root)
    freeze_path = root / IMPLEMENTATION_FREEZE_RELATIVE_PATH
    if freeze_path.exists() or freeze_path.is_symlink():
        raise SyntheticCausalAcquisitionError("implementation freeze already exists")
    forbidden_outputs = (
        SEED_MARKER_RELATIVE_PATH,
        SEED_RELATIVE_PATH,
        SEED_CUSTODY_RELATIVE_PATH,
        SEED_FAILURE_RELATIVE_PATH,
        COHORT_MARKER_RELATIVE_PATH,
        PRIVATE_COHORT_RELATIVE_PATH,
        ACQUISITION_RECEIPT_RELATIVE_PATH,
        REPRODUCIBILITY_RELATIVE_PATH,
        REPRODUCIBILITY_MARKER_RELATIVE_PATH,
        REPRODUCIBILITY_FAILURE_RELATIVE_PATH,
        *FORMAL_STAGE_FORBIDDEN_RELATIVE_PATHS,
    )
    if any((root / relative).exists() for relative in forbidden_outputs):
        raise SyntheticCausalAcquisitionError("formal seed or cohort output already exists")
    bindings: list[dict[str, str]] = []
    for relative in sorted(REQUIRED_FREEZE_PATHS):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise SyntheticCausalAcquisitionError("required implementation file is unavailable")
        worktree_raw = path.read_bytes()
        head_raw = _committed_bytes(root, Path(relative))
        if worktree_raw != head_raw:
            raise SyntheticCausalAcquisitionError("required implementation file is not HEAD-clean")
        bindings.append(
            {
                "relative_path": relative,
                "file_sha256": hashlib.sha256(head_raw).hexdigest(),
                "git_blob_sha1": _git_blob_sha1(head_raw),
            }
        )
    body = {
        "schema": IMPLEMENTATION_FREEZE_SCHEMA,
        "decision": "freeze_the_complete_preseed_synthetic_formal_implementation_against_actual_HEAD",
        "creation_HEAD": _git(root, "rev-parse", "HEAD").decode("ascii").strip(),
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "grammar_sha256": GRAMMAR_SHA256,
        "graph_core_sha256": GRAPH_CORE_SHA256,
        "formal_seed_or_cohort_exists": False,
        "bindings": bindings,
    }
    freeze = _self_hashed(body, "implementation_freeze_sha256")
    _write_json_exclusive(freeze_path, freeze, PUBLIC_MODE)
    return freeze


def _load_committed_public_json(
    project_root: Path, relative_path: Path, field: str
) -> dict[str, Any]:
    path = _assert_no_symlink_components(project_root / relative_path, field)
    if not path.is_file() or path.is_symlink():
        raise SyntheticCausalAcquisitionError(f"{field} is unavailable")
    raw = path.read_bytes()
    if _committed_bytes(project_root, relative_path) != raw:
        raise SyntheticCausalAcquisitionError(f"{field} is not current-HEAD committed")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticCausalAcquisitionError(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SyntheticCausalAcquisitionError(f"{field} root drifted")
    return payload


def create_seed_custody(
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Create the sole 32-byte OS-random seed and publish only its commitment.

    This function never compiles the cohort.  The custody file therefore exists
    before any generated world or outcome.  Both outputs are exclusive; callers
    must treat any partial failure as terminal rather than creating another seed.
    """

    if _FORMAL_ACQUISITION_ENTRY_ACTIVE is not True:
        raise SyntheticCausalAcquisitionError(
            "formal seed custody may only be consumed by the acquisition CLI"
        )
    root = project_root.resolve(strict=True)
    verify_frozen_design(root)
    freeze, actual_head = verify_implementation_freeze(root)
    seed_marker_path = root / SEED_MARKER_RELATIVE_PATH
    seed_path = root / SEED_RELATIVE_PATH
    custody_path = root / SEED_CUSTODY_RELATIVE_PATH
    failure_path = root / SEED_FAILURE_RELATIVE_PATH
    if any(path.exists() or path.is_symlink() for path in (
        seed_marker_path,
        seed_path,
        custody_path,
        failure_path,
        *(root / relative for relative in FORMAL_STAGE_FORBIDDEN_RELATIVE_PATHS),
        root / REPRODUCIBILITY_MARKER_RELATIVE_PATH,
        root / REPRODUCIBILITY_FAILURE_RELATIVE_PATH,
        root / REPRODUCIBILITY_RELATIVE_PATH,
    )):
        raise SyntheticCausalAcquisitionError("canonical seed-attempt output already exists")
    marker = _self_hashed(
        {
            "schema": f"{VERSION}_seed_generation_attempt_marker",
            "version": VERSION,
            "status": "sole_seed_generation_attempt_consumed",
            "actual_HEAD": actual_head,
            "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
            "design_sha256": DESIGN_SHA256,
            "attempt_count": 1,
        },
        "marker_sha256",
    )
    marker_file_sha256 = _write_json_exclusive(seed_marker_path, marker, PUBLIC_MODE)
    try:
        # First randomness access occurs strictly after the durable marker.
        seed = os.urandom(SEED_BYTES)
        if not isinstance(seed, bytes) or len(seed) != SEED_BYTES:
            raise SyntheticCausalAcquisitionError("OS random source did not return 32 bytes")
        commitment = hashlib.sha256(seed).hexdigest()
        seed_file_sha256 = _write_exclusive(seed_path, seed, PRIVATE_MODE)
        if seed_file_sha256 != commitment:
            raise SyntheticCausalAcquisitionError("private seed persistence drifted")
        body = {
            "schema": CUSTODY_SCHEMA,
            "version": VERSION,
            "status": "seed_committed_cohort_not_generated",
            "design_sha256": DESIGN_SHA256,
            "design_file_sha256": DESIGN_FILE_SHA256,
            "grammar_sha256": GRAMMAR_SHA256,
            "graph_core_sha256": GRAPH_CORE_SHA256,
            "amendment_sha256": AMENDMENT_SHA256,
            "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
            "seed_attempt_marker_sha256": marker["marker_sha256"],
            "seed_attempt_marker_file_sha256": marker_file_sha256,
            "seed_bytes": SEED_BYTES,
            "seed_generation": "os.urandom_exactly_once_after_marker_O_EXCL_mode_0600",
            "seed_commitment_sha256": commitment,
            "seed_material_published": False,
            "cohort_generated": False,
            "seed_trials_allowed": 1,
        }
        custody = _self_hashed(body, "custody_sha256")
        _write_json_exclusive(custody_path, custody, PUBLIC_MODE)
        return custody
    except Exception as exc:
        failure = _self_hashed(
            {
                "schema": f"{VERSION}_seed_generation_failure_receipt",
                "version": VERSION,
                "status": "terminal_seed_generation_invalid_no_replay",
                "design_sha256": DESIGN_SHA256,
                "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
                "seed_attempt_marker_sha256": marker["marker_sha256"],
                "failure_class": type(exc).__name__,
                "exception_message_or_secret_material_persisted": False,
            },
            "receipt_sha256",
        )
        if not failure_path.exists():
            _write_json_exclusive(failure_path, failure, PUBLIC_MODE)
        raise


def _read_private_seed(path: Path, expected_commitment: str) -> bytes:
    absolute = _assert_no_symlink_components(path, "private seed")
    if not absolute.is_file() or absolute.is_symlink():
        raise SyntheticCausalAcquisitionError("private seed is unavailable")
    info = absolute.stat()
    if stat.S_IMODE(info.st_mode) != PRIVATE_MODE or info.st_size != SEED_BYTES:
        raise SyntheticCausalAcquisitionError("private seed mode or size drifted")
    seed = absolute.read_bytes()
    if hashlib.sha256(seed).hexdigest() != expected_commitment:
        raise SyntheticCausalAcquisitionError("private seed commitment drifted")
    return seed


def _read_json_with_mode(
    path: Path, *, expected_mode: int, field: str
) -> tuple[dict[str, Any], str]:
    absolute = _assert_no_symlink_components(path, field)
    if not absolute.is_file() or stat.S_IMODE(absolute.stat().st_mode) != expected_mode:
        raise SyntheticCausalAcquisitionError(f"{field} mode or type drifted")
    raw = absolute.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticCausalAcquisitionError(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SyntheticCausalAcquisitionError(f"{field} root drifted")
    return payload, hashlib.sha256(raw).hexdigest()


def _require_ancestor(root: Path, commit: object, field: str) -> str:
    if not isinstance(commit, str) or len(commit) != 40:
        raise SyntheticCausalAcquisitionError(f"{field} commit is malformed")
    try:
        _git(root, "merge-base", "--is-ancestor", commit, "HEAD")
    except Exception as exc:
        raise SyntheticCausalAcquisitionError(f"{field} is not an ancestor of HEAD") from exc
    return commit


def _historical_bytes(root: Path, commit: str, relative_path: Path) -> bytes:
    prefix = _git_project_prefix(root)
    try:
        return _git(root, "show", f"{commit}:{prefix}{relative_path.as_posix()}")
    except Exception as exc:
        raise SyntheticCausalAcquisitionError(
            "historical custody binding is absent"
        ) from exc


def load_seed_custody(path: Path) -> dict[str, Any]:
    absolute = _assert_no_symlink_components(path, "seed custody")
    root = absolute.parent.parent
    if absolute != root / SEED_CUSTODY_RELATIVE_PATH:
        raise SyntheticCausalAcquisitionError("seed custody path is not canonical")
    payload, _custody_file_hash = _read_json_with_mode(
        absolute, expected_mode=PUBLIC_MODE, field="seed custody"
    )
    if not isinstance(payload, dict) or payload.get("schema") != CUSTODY_SCHEMA:
        raise SyntheticCausalAcquisitionError("seed custody schema drifted")
    body = dict(payload)
    declared = body.pop("custody_sha256", None)
    if not isinstance(declared, str) or semantic_hash(body) != declared:
        raise SyntheticCausalAcquisitionError("seed custody self-hash drifted")
    fixed = {
        "status": "seed_committed_cohort_not_generated",
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "grammar_sha256": GRAMMAR_SHA256,
        "graph_core_sha256": GRAPH_CORE_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "seed_bytes": SEED_BYTES,
        "seed_trials_allowed": 1,
        "seed_generation": "os.urandom_exactly_once_after_marker_O_EXCL_mode_0600",
        "seed_material_published": False,
        "cohort_generated": False,
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise SyntheticCausalAcquisitionError("seed custody binding drifted")
    commitment = payload.get("seed_commitment_sha256")
    freeze_hash = payload.get("implementation_freeze_sha256")
    if (
        not isinstance(commitment, str)
        or len(commitment) != 64
        or any(character not in "0123456789abcdef" for character in commitment)
        or not isinstance(freeze_hash, str)
        or len(freeze_hash) != 64
        or any(character not in "0123456789abcdef" for character in freeze_hash)
    ):
        raise SyntheticCausalAcquisitionError("seed commitment is malformed")
    marker, marker_file_hash = _read_json_with_mode(
        root / SEED_MARKER_RELATIVE_PATH,
        expected_mode=PUBLIC_MODE,
        field="seed attempt marker",
    )
    marker_body = dict(marker)
    marker_hash = marker_body.pop("marker_sha256", None)
    if (
        not isinstance(marker_hash, str)
        or semantic_hash(marker_body) != marker_hash
        or marker.get("schema") != f"{VERSION}_seed_generation_attempt_marker"
        or marker.get("version") != VERSION
        or marker.get("status") != "sole_seed_generation_attempt_consumed"
        or marker.get("attempt_count") != 1
        or marker.get("design_sha256") != DESIGN_SHA256
        or marker.get("implementation_freeze_sha256") != freeze_hash
        or payload.get("seed_attempt_marker_sha256") != marker_hash
        or payload.get("seed_attempt_marker_file_sha256") != marker_file_hash
    ):
        raise SyntheticCausalAcquisitionError("seed attempt marker chain drifted")
    invocation = _require_ancestor(root, marker.get("actual_HEAD"), "seed marker HEAD")
    freeze_path = _assert_no_symlink_components(
        root / IMPLEMENTATION_FREEZE_RELATIVE_PATH, "implementation freeze"
    )
    if (
        not freeze_path.is_file()
        or _historical_bytes(root, invocation, IMPLEMENTATION_FREEZE_RELATIVE_PATH)
        != freeze_path.read_bytes()
    ):
        raise SyntheticCausalAcquisitionError("seed marker historical freeze drifted")
    return payload


def _core_edges(item: grammar.CompiledItem) -> tuple[grammar.SyntheticEdge, ...]:
    spans = tuple(
        core.SourceSpan(node.span_i, node.start, node.end, node.identity_text)
        for node in item.nodes
    )
    return tuple(
        grammar.SyntheticEdge(edge.edge_family, edge.left_span_i, edge.right_span_i)
        for edge in core.build_typed_clause_graph(spans)
    )


def _edge_rows(edges: Sequence[grammar.SyntheticEdge]) -> list[list[object]]:
    return [
        [edge.edge_family, edge.left_span_i, edge.right_span_i]
        for edge in edges
    ]


def _view_row(item: grammar.CompiledItem) -> dict[str, Any]:
    full = _core_edges(item)
    if not set(item.designated_edges).issubset(full):
        raise SyntheticCausalAcquisitionError("designated edge is absent from graph core")
    edges_by_mode = {
        mode: _edge_rows(grammar.apply_graph_ablation(item, full, mode=mode))
        for mode in grammar.ABLATION_MODES
    }
    # Deliberately exclude item_commitment_sha256.  The grammar commitment is
    # unsalted over a tiny gold-index space and is therefore late-label data,
    # not a safe opaque identifier for an action pack.
    body = {
        "schema": VIEW_ITEM_SCHEMA,
        "block": item.block,
        "ordinal": item.block_ordinal,
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
        "edges_by_mode": edges_by_mode,
    }
    return _self_hashed(body, "opaque_view_sha256")


def _label_rows(
    block: str,
    items: Sequence[grammar.CompiledItem],
    *,
    view_ids: Mapping[str, str],
    derangement: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    by_label_free = {item.label_free_commitment_sha256: item for item in items}
    rows: list[dict[str, Any]] = []
    for item in items:
        row: dict[str, Any] = {
            "schema": LABEL_ITEM_SCHEMA,
            "block": block,
            "ordinal": item.block_ordinal,
            "opaque_view_sha256": view_ids[item.item_commitment_sha256],
            "item_commitment_sha256": item.item_commitment_sha256,
            "label_free_commitment_sha256": item.label_free_commitment_sha256,
            "matching_signature_sha256": item.matching_signature_sha256,
            "structural_draw_sha256": item.structural_draw_sha256,
            "family_slot": item.family_slot,
            "family_id": item.family_id,
            "family_role": item.family_role,
            "template_split": item.template_split,
            "polarity": item.polarity,
            "negative_kind": item.negative_kind,
            "edge_family": item.edge_family,
            "pair_key": item.pair_key,
            "gold_node_indices": list(item.gold_node_indices),
        }
        if block == "A_form":
            if derangement is None or item.label_free_commitment_sha256 not in derangement:
                raise SyntheticCausalAcquisitionError("A_form derangement is incomplete")
            source = by_label_free[derangement[item.label_free_commitment_sha256]]
            row["permuted_from_item_commitment_sha256"] = source.item_commitment_sha256
            row["permuted_from_opaque_view_sha256"] = view_ids[
                source.item_commitment_sha256
            ]
            row["permuted_gold_node_indices"] = list(source.gold_node_indices)
        rows.append(row)
    return rows


def _pack(schema: str, block: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    body = {
        "schema": schema,
        "block": block,
        "count": len(rows),
        "rows": list(rows),
    }
    return _self_hashed(body, "block_sha256")


def _family_aggregate(items: Sequence[grammar.CompiledItem]) -> dict[str, Any]:
    return {
        "family_id_counts": dict(sorted(Counter(item.family_id for item in items).items())),
        "edge_family_counts": dict(
            sorted(Counter(item.edge_family for item in items).items())
        ),
        "polarity_counts": dict(sorted(Counter(item.polarity for item in items).items())),
        "gold_cardinality_counts": {
            str(key): value
            for key, value in sorted(Counter(len(item.gold_node_indices) for item in items).items())
        },
        "pair_count": len({item.pair_key for item in items}),
    }


def acquire_formal_cohort(
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Consume the one cohort attempt and persist separated private packs.

    The attempt marker is written before the seed is opened.  Any exception
    after that point is terminal under the preregistration and must not be
    replayed with this or another seed.
    """

    if _FORMAL_ACQUISITION_ENTRY_ACTIVE is not True:
        raise SyntheticCausalAcquisitionError(
            "formal cohort may only be consumed by the acquisition CLI"
        )
    root = project_root.resolve(strict=True)
    verify_frozen_design(root)
    freeze, actual_head = verify_implementation_freeze(root)
    seed_path = root / SEED_RELATIVE_PATH
    custody_path = root / SEED_CUSTODY_RELATIVE_PATH
    private_root = root / PRIVATE_COHORT_RELATIVE_PATH
    attempt_marker_path = root / COHORT_MARKER_RELATIVE_PATH
    public_receipt_path = root / ACQUISITION_RECEIPT_RELATIVE_PATH
    committed_custody = _load_committed_public_json(
        root, SEED_CUSTODY_RELATIVE_PATH, "seed custody"
    )
    custody = load_seed_custody(custody_path)
    if committed_custody != custody or custody.get(
        "implementation_freeze_sha256"
    ) != freeze.get("implementation_freeze_sha256"):
        raise SyntheticCausalAcquisitionError("committed seed custody binding drifted")
    if any(path.exists() or path.is_symlink() for path in (
        attempt_marker_path, private_root, public_receipt_path
    )):
        raise SyntheticCausalAcquisitionError("canonical cohort-attempt output already exists")
    commitment = str(custody["seed_commitment_sha256"])
    marker = _self_hashed(
        {
            "schema": ATTEMPT_SCHEMA,
            "version": VERSION,
            "status": "formal_cohort_attempt_consumed",
            "design_sha256": DESIGN_SHA256,
            "custody_sha256": custody["custody_sha256"],
            "seed_commitment_sha256": commitment,
            "actual_HEAD": actual_head,
            "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
            "attempt_count": 1,
        },
        "marker_sha256",
    )
    marker_file_sha256 = _write_json_exclusive(
        attempt_marker_path, marker, PRIVATE_MODE
    )
    try:
        # First private-seed open occurs strictly after the durable cohort marker.
        seed = _read_private_seed(seed_path, commitment)
        blocks = grammar.generate_all_blocks(seed)
        derangement = dict(
            grammar.evaluator_label_derangement(blocks["A_form"], seed=seed)
        )
        private_root.mkdir(parents=True, exist_ok=False, mode=0o700)

        pack_receipts: dict[str, dict[str, Any]] = {}
        all_commitments: list[str] = []
        same_gold_vector_count = 0
        for block in grammar.BLOCK_ORDER:
            items = blocks[block]
            all_commitments.extend(item.item_commitment_sha256 for item in items)
            view_rows = [_view_row(item) for item in items]
            view_ids = {
                item.item_commitment_sha256: str(row["opaque_view_sha256"])
                for item, row in zip(items, view_rows)
            }
            view = _pack(VIEW_SCHEMA, block, view_rows)
            view_path = private_root / f"{block}.label_free.sealed.json"
            view_file_sha256 = _write_json_exclusive(view_path, view, PRIVATE_MODE)
            row: dict[str, Any] = {
                "count": len(items),
                "grammar_block_commitment_sha256": grammar.block_commitment(items),
                "view_block_sha256": view["block_sha256"],
                "view_file_sha256": view_file_sha256,
                "aggregate": _family_aggregate(items),
            }
            if block != "F_search":
                label_rows = _label_rows(
                    block,
                    items,
                    view_ids=view_ids,
                    derangement=derangement if block == "A_form" else None,
                )
                if block == "A_form":
                    same_gold_vector_count = sum(
                        label["gold_node_indices"] == label["permuted_gold_node_indices"]
                        for label in label_rows
                    )
                labels = _pack(LABEL_SCHEMA, block, label_rows)
                label_path = private_root / f"{block}.labels.sealed.json"
                row["label_block_sha256"] = labels["block_sha256"]
                row["label_file_sha256"] = _write_json_exclusive(
                    label_path, labels, PRIVATE_MODE
                )
            else:
                row["labels_created"] = False
            pack_receipts[block] = row

        if len(all_commitments) != 256 or len(set(all_commitments)) != 256:
            raise SyntheticCausalAcquisitionError("formal cohort commitments overlap")
        if {path.name for path in private_root.iterdir()} != set(
            EXPECTED_PRIVATE_COHORT_FILES
        ) or (private_root / "F_search.labels.sealed.json").exists():
            raise SyntheticCausalAcquisitionError("private cohort file set drifted")
        body = {
        "schema": RECEIPT_SCHEMA,
        "version": VERSION,
        "status": "formal_cohort_acquired_private_labels_separated",
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "grammar_sha256": GRAMMAR_SHA256,
        "graph_core_sha256": GRAPH_CORE_SHA256,
        "custody_sha256": custody["custody_sha256"],
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "seed_commitment_sha256": commitment,
        "attempt_marker_file_sha256": marker_file_sha256,
        "attempt_marker_sha256": marker["marker_sha256"],
        "block_order": list(grammar.BLOCK_ORDER),
        "total_count": 256,
        "packs": pack_receipts,
        "F_search_labels_created": False,
        "evaluator_derangement_fixed_point_definition": "different_label_free_item_identity",
        "evaluator_derangement_effective_same_gold_vector_count": same_gold_vector_count,
        "same_gold_vector_count_is_descriptive_not_a_gate_or_retry_trigger": True,
        "item_rows_persisted_publicly": False,
        "seed_material_published": False,
        "candidate_pool_or_filter_used": False,
        "network_calls": 0,
        }
        receipt = _self_hashed(body, "receipt_sha256")
        _write_json_exclusive(public_receipt_path, receipt, PUBLIC_MODE)
        return receipt
    except Exception as exc:
        failure = _self_hashed(
            {
                "schema": RECEIPT_SCHEMA,
                "version": VERSION,
                "status": "terminal_cohort_generation_invalid_no_replay",
                "design_sha256": DESIGN_SHA256,
                "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
                "custody_sha256": custody["custody_sha256"],
                "seed_commitment_sha256": commitment,
                "attempt_marker_sha256": marker["marker_sha256"],
                "failure_class": type(exc).__name__,
                "failure_message_private_path_seed_or_item_persisted_publicly": False,
                "replacement_seed_smaller_block_or_replay_authorized": False,
            },
            "receipt_sha256",
        )
        if not public_receipt_path.exists():
            _write_json_exclusive(public_receipt_path, failure, PUBLIC_MODE)
        raise


def load_committed_acquisition_receipt(project_root: Path) -> dict[str, Any]:
    """Load only the canonical success receipt committed in actual current HEAD."""

    root = project_root.resolve(strict=True)
    freeze, _head = verify_implementation_freeze(root)
    receipt = _load_committed_public_json(
        root, ACQUISITION_RECEIPT_RELATIVE_PATH, "acquisition receipt"
    )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise SyntheticCausalAcquisitionError("acquisition receipt schema drifted")
    body = dict(receipt)
    declared = body.pop("receipt_sha256", None)
    if not isinstance(declared, str) or semantic_hash(body) != declared:
        raise SyntheticCausalAcquisitionError("acquisition receipt self-hash drifted")
    if (
        receipt.get("status") != "formal_cohort_acquired_private_labels_separated"
        or receipt.get("design_sha256") != DESIGN_SHA256
        or receipt.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or receipt.get("F_search_labels_created") is not False
        or receipt.get("total_count") != 256
        or not isinstance(receipt.get("packs"), Mapping)
        or set(receipt["packs"]) != set(grammar.BLOCK_ORDER)
    ):
        raise SyntheticCausalAcquisitionError("acquisition receipt binding drifted")
    private_root = _assert_no_symlink_components(
        root / PRIVATE_COHORT_RELATIVE_PATH, "private cohort root"
    )
    if not private_root.is_dir() or {path.name for path in private_root.iterdir()} != set(
        EXPECTED_PRIVATE_COHORT_FILES
    ) or (private_root / "F_search.labels.sealed.json").exists():
        raise SyntheticCausalAcquisitionError("committed private cohort file set drifted")
    committed_custody = _load_committed_public_json(
        root, SEED_CUSTODY_RELATIVE_PATH, "seed custody"
    )
    custody = load_seed_custody(root / SEED_CUSTODY_RELATIVE_PATH)
    if committed_custody != custody or (
        receipt.get("custody_sha256") != custody.get("custody_sha256")
        or receipt.get("seed_commitment_sha256")
        != custody.get("seed_commitment_sha256")
    ):
        raise SyntheticCausalAcquisitionError("acquisition custody chain drifted")
    marker, marker_file_hash = _read_json_with_mode(
        root / COHORT_MARKER_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="cohort attempt marker",
    )
    marker_body = dict(marker)
    marker_hash = marker_body.pop("marker_sha256", None)
    if (
        not isinstance(marker_hash, str)
        or semantic_hash(marker_body) != marker_hash
        or marker.get("schema") != ATTEMPT_SCHEMA
        or marker.get("version") != VERSION
        or marker.get("status") != "formal_cohort_attempt_consumed"
        or marker.get("attempt_count") != 1
        or marker.get("design_sha256") != DESIGN_SHA256
        or marker.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or marker.get("custody_sha256") != custody.get("custody_sha256")
        or marker.get("seed_commitment_sha256")
        != custody.get("seed_commitment_sha256")
        or receipt.get("attempt_marker_sha256") != marker_hash
        or receipt.get("attempt_marker_file_sha256") != marker_file_hash
    ):
        raise SyntheticCausalAcquisitionError("cohort attempt marker chain drifted")
    invocation = _require_ancestor(root, marker.get("actual_HEAD"), "cohort marker HEAD")
    custody_path = _assert_no_symlink_components(
        root / SEED_CUSTODY_RELATIVE_PATH, "seed custody"
    )
    if _historical_bytes(root, invocation, SEED_CUSTODY_RELATIVE_PATH) != (
        custody_path.read_bytes()
    ):
        raise SyntheticCausalAcquisitionError("cohort marker historical custody drifted")
    return receipt


def _load_terminal_stage_receipt(root: Path) -> tuple[str, dict[str, Any], str]:
    # Import lazily to avoid the module-level acquisition -> runner ->
    # acquisition cycle.  Publication must use the same recursive validator as
    # next-stage authorization: public receipt self-hashes alone do not bind
    # the private attempt marker, pre-label action seal, or invocation history.
    from . import synthetic_typed_graph_causal_runner_v1 as runner

    acquisition = load_committed_acquisition_receipt(root)
    freeze, _head = verify_implementation_freeze(root)
    paths = {
        "formation": runner.FORMATION_RECEIPT_RELATIVE_PATH,
        "A_hold": runner.A_HOLD_RECEIPT_RELATIVE_PATH,
        "M_search": runner.M_SEARCH_RECEIPT_RELATIVE_PATH,
    }

    def load(stage: str) -> tuple[dict[str, Any], str]:
        relative = paths[stage]
        receipt = runner._load_validated_stage_receipt(
            root=root,
            stage=stage,
            acquisition=acquisition,
            freeze=freeze,
        )
        if receipt.get("item_rows_or_item_commitments_persisted_publicly") is not False:
            raise SyntheticCausalAcquisitionError(f"{stage} public projection drifted")
        return receipt, sha256_file(root / relative)

    formation, formation_file_hash = load("formation")
    if formation.get("parent_receipt_sha256") is not None:
        raise SyntheticCausalAcquisitionError("formation parent receipt drifted")
    if formation.get("status") == "terminal_unidentifiable_transition":
        return "formation", formation, formation_file_hash
    if (
        formation.get("status") != "formation_complete_identifiable"
        or formation.get("A_hold_authorized") is not True
    ):
        raise SyntheticCausalAcquisitionError("formation is not a valid terminal chain root")
    a_hold, a_hold_file_hash = load("A_hold")
    recipe_fields = ("real_recipe_id", "permuted_recipe_id", "fixed_e00_recipe_id")
    if a_hold.get("parent_receipt_sha256") != formation.get("receipt_sha256") or any(
        a_hold.get(field) != formation.get(field) for field in recipe_fields
    ):
        raise SyntheticCausalAcquisitionError("A_hold parent receipt drifted")
    if a_hold.get("status") == "valid_nonpromotion":
        return "A_hold", a_hold, a_hold_file_hash
    if a_hold.get("status") != "promoted" or a_hold.get("M_search_authorized") is not True:
        raise SyntheticCausalAcquisitionError("A_hold is neither terminal nor M-authorizing")
    m_search, m_file_hash = load("M_search")
    if (
        m_search.get("parent_receipt_sha256") != a_hold.get("receipt_sha256")
        or any(m_search.get(field) != a_hold.get(field) for field in recipe_fields)
        or m_search.get("status")
        not in {"terminal_positive_net", "terminal_nonpositive_net"}
    ):
        raise SyntheticCausalAcquisitionError("M_search terminal chain drifted")
    return "M_search", m_search, m_file_hash


def _compiled_public_row(item: grammar.CompiledItem) -> dict[str, Any]:
    return {
        "schema": item.schema,
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
        "designated_edges": _edge_rows(item.designated_edges),
        "endpoint_permutation": [list(pair) for pair in item.endpoint_permutation],
    }


def publish_terminal_reproducibility(project_root: Path) -> dict[str, Any]:
    """Publish the committed seed and full cohort only after a committed terminal."""

    if _FORMAL_ACQUISITION_ENTRY_ACTIVE is not True:
        raise SyntheticCausalAcquisitionError(
            "reproducibility publication may only be consumed by the acquisition CLI"
        )
    root = project_root.resolve(strict=True)
    freeze, actual_head = verify_implementation_freeze(root)
    terminal_stage, terminal, terminal_file_hash = _load_terminal_stage_receipt(root)
    output_path = root / REPRODUCIBILITY_RELATIVE_PATH
    marker_path = root / REPRODUCIBILITY_MARKER_RELATIVE_PATH
    failure_path = root / REPRODUCIBILITY_FAILURE_RELATIVE_PATH
    if any(path.exists() or path.is_symlink() for path in (
        output_path, marker_path, failure_path
    )):
        raise SyntheticCausalAcquisitionError("canonical reproducibility attempt output exists")
    committed_custody = _load_committed_public_json(
        root, SEED_CUSTODY_RELATIVE_PATH, "seed custody"
    )
    custody = load_seed_custody(root / SEED_CUSTODY_RELATIVE_PATH)
    if committed_custody != custody:
        raise SyntheticCausalAcquisitionError("seed custody is not current-HEAD committed")
    marker = _self_hashed(
        {
            "schema": f"{VERSION}_reproducibility_publication_attempt_marker",
            "version": VERSION,
            "status": "sole_reproducibility_publication_attempt_consumed",
            "actual_HEAD": actual_head,
            "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
            "terminal_stage": terminal_stage,
            "terminal_receipt_sha256": terminal["receipt_sha256"],
            "seed_commitment_sha256": custody["seed_commitment_sha256"],
            "attempt_count": 1,
        },
        "marker_sha256",
    )
    marker_file_hash = _write_json_exclusive(marker_path, marker, PRIVATE_MODE)
    try:
        # First private-seed access follows the durable one-shot publication marker.
        seed = _read_private_seed(
            root / SEED_RELATIVE_PATH, str(custody["seed_commitment_sha256"])
        )
        blocks = grammar.generate_all_blocks(seed)
        body = {
        "schema": "synthetic_typed_graph_causal_terminal_reproducibility_v1",
        "version": VERSION,
        "status": "terminal_seed_and_full_compiled_cohort_published",
        "publication_HEAD": actual_head,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "design_sha256": DESIGN_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "grammar_sha256": GRAMMAR_SHA256,
        "seed_encoding": "lowercase_hex_exact_32_bytes",
        "formal_seed_hex": seed.hex(),
        "seed_commitment_sha256": hashlib.sha256(seed).hexdigest(),
        "terminal_stage": terminal_stage,
        "terminal_receipt_sha256": terminal["receipt_sha256"],
        "terminal_receipt_file_sha256": terminal_file_hash,
        "publication_attempt_marker_sha256": marker["marker_sha256"],
        "publication_attempt_marker_file_sha256": marker_file_hash,
        "block_order": list(grammar.BLOCK_ORDER),
        "block_commitments": {
            block: grammar.block_commitment(blocks[block])
            for block in grammar.BLOCK_ORDER
        },
        "total_count": 256,
        "blocks": {
            block: [_compiled_public_row(item) for item in blocks[block]]
            for block in grammar.BLOCK_ORDER
        },
        "retrieval_actions_model_outputs_or_scores_included": False,
        }
        artifact = _self_hashed(body, "reproducibility_sha256")
        _write_json_exclusive(output_path, artifact, PUBLIC_MODE)
        return artifact
    except Exception as exc:
        failure = _self_hashed(
            {
                "schema": f"{VERSION}_reproducibility_failure_receipt",
                "version": VERSION,
                "status": "terminal_reproducibility_publication_invalid_no_replay",
                "marker_sha256": marker["marker_sha256"],
                "terminal_stage": terminal_stage,
                "terminal_receipt_sha256": terminal["receipt_sha256"],
                "failure_class": type(exc).__name__,
                "exception_message_seed_or_item_persisted_publicly": False,
                "retry_or_replay_authorized": False,
            },
            "receipt_sha256",
        )
        if not failure_path.exists():
            _write_json_exclusive(failure_path, failure, PUBLIC_MODE)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "seed", "cohort", "publish-reproducibility")
    )
    parser.add_argument("--project-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    global _FORMAL_ACQUISITION_ENTRY_ACTIVE
    if arguments.command == "freeze":
        result = create_implementation_freeze(arguments.project_root)
        status = "implementation_frozen_must_commit_before_seed"
        result_hash = result["implementation_freeze_sha256"]
    else:
        if _FORMAL_ACQUISITION_ENTRY_ACTIVE:
            raise SyntheticCausalAcquisitionError("formal acquisition entry is already active")
        _FORMAL_ACQUISITION_ENTRY_ACTIVE = True
        try:
            if arguments.command == "seed":
                result = create_seed_custody(project_root=arguments.project_root)
                status = str(result["status"])
                result_hash = str(result["custody_sha256"])
            elif arguments.command == "cohort":
                result = acquire_formal_cohort(project_root=arguments.project_root)
                status = str(result["status"])
                result_hash = str(result["receipt_sha256"])
            else:
                result = publish_terminal_reproducibility(arguments.project_root)
                status = str(result["status"])
                result_hash = str(result["reproducibility_sha256"])
        finally:
            _FORMAL_ACQUISITION_ENTRY_ACTIVE = False
    print(
        json.dumps(
            {"command": arguments.command, "status": status, "result_sha256": result_hash},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "ATTEMPT_SCHEMA",
    "CUSTODY_SCHEMA",
    "DESIGN_FILE_SHA256",
    "DESIGN_SHA256",
    "GRAPH_CORE_SHA256",
    "GRAMMAR_SHA256",
    "LABEL_ITEM_SCHEMA",
    "LABEL_SCHEMA",
    "PRIVATE_MODE",
    "PUBLIC_MODE",
    "RECEIPT_SCHEMA",
    "SEED_BYTES",
    "SyntheticCausalAcquisitionError",
    "VERSION",
    "VIEW_ITEM_SCHEMA",
    "VIEW_SCHEMA",
    "acquire_formal_cohort",
    "canonical_bytes",
    "create_seed_custody",
    "create_implementation_freeze",
    "load_seed_custody",
    "load_committed_acquisition_receipt",
    "semantic_hash",
    "publish_terminal_reproducibility",
    "sha256_file",
    "verify_frozen_design",
]


if __name__ == "__main__":
    raise SystemExit(main())
