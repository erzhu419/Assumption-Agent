"""Fresh-seed custody, acquisition, and terminal publication for replication v2.

This module is deliberately score blind.  A committed, successful public
integration diagnostic must exist before the formal implementation freeze can
be created.  After that freeze is committed, the formal path consumes one
256-byte OS-random draw, splits it into eight ordered seeds, generates exactly
one ``A_hold`` block per seed, and writes separate private action, late-label,
and full-compiled packs.  Publication reads the frozen compiled pack; it never
regenerates the cohort.

Formal commands require current-HEAD committed public predecessors.  Tests use
temporary roots and are not formal attempts.
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


VERSION = "synthetic_typed_graph_multiseed_replication_v2"
MODULE_VERSION = "synthetic_typed_graph_multiseed_acquisition_v2"
DESIGN_SCHEMA = "synthetic_typed_graph_multiseed_replication_design_v2"
DESIGN_SHA256 = "c429d455fdf20cc339e3ac9985ece38fbf78932a7b23ee4f01653c49f314d14a"
DESIGN_FILE_SHA256 = "41ae773e5e1c7a7a0b67374cc5946650060089b84bb62b165fccc2d28dfdb2a2"
CHUNK_SCHEDULE_SHA256 = (
    "faf5dd2b2a45b4a2b16b8913b5e38e930e99d7ccaa3ce35218d6de38a863a635"
)
GRAMMAR_SHA256 = "acb691846256e87603e92ff079e3db2a1c9df8ea36c193f1dba3c7d35893f9d9"
GRAPH_CORE_SHA256 = "7aef388172c08eecd227033111ce0e92845bca0b514a8bacbff205566963460c"

SEED_COUNT = 8
SEED_BYTES = 32
SEED_BATCH_BYTES = SEED_COUNT * SEED_BYTES
BLOCK = "A_hold"
ITEMS_PER_SEED = 64
TOTAL_ITEM_COUNT = SEED_COUNT * ITEMS_PER_SEED
PRIVATE_MODE = 0o600
PUBLIC_MODE = 0o644

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
V1_PUBLICATION_FILE_SHA256 = (
    "7ea28c298422191456ec976ddcc22450bd7021d4f14afb6bd283ae0f6d44b6e1"
)
V1_PUBLICATION_SHA256 = (
    "f54998cef3259ac196c7d4a767cc034df657f1e5221b6da6c3eb30b52d5ba13c"
)
V1_ITEM_COMMITMENT_SET_SHA256 = (
    "62f57fc07dfc95aafd1cb590787aa54a326ad2843711d397813a070398447bd6"
)
V1_ORDERED_SEED_COMMITMENTS = (
    "d7b2b8f364c812946e989f1f11c589824e76ed809e290a84c42c5ef289ecaa53",
    "8aef84f20314126f5117cce89d9b0a28ef8d2e1c36e93992ab471268a9e808ae",
    "57a8321762bfaa76419964eca8ea9ff3fbda985b6e6488774bf2b70b0da84b94",
    "2d3e0447f09eec17158dd22c2142d491cb8de329bed81d5bb5bad98218fff378",
    "26bc1534c3f250b90a59cf12bbb05e0e6cc244cb05b2aea5895c0f95bed4889a",
    "b0b4f20c5c8764dfe006d56f32370aecb1004350ea5ee754d47653ab5128b9e9",
    "d7978d34b901c6fdd8348791050d2269fa2fa631c4749f732522233ab1299096",
    "72e561d5c89550b0351185a6a313d08f2cc84e43c73190e5a9886c46f909a179",
)

INTEGRATION_DIAGNOSTIC_SCHEMA = (
    "synthetic_typed_graph_multiseed_replication_integration_diagnostic_v2"
)
INTEGRATION_SUCCESS_STATUS = "integration_diagnostic_complete_no_scores_or_claims"
INTEGRATION_FAILURE_STATUS = (
    "terminal_integration_diagnostic_invalid_fresh_formal_not_authorized"
)
FREEZE_SCHEMA = "synthetic_typed_graph_multiseed_replication_implementation_freeze_v2"
SEED_CUSTODY_SCHEMA = "synthetic_typed_graph_multiseed_replication_seed_custody_v2"
ACQUISITION_SCHEMA = "synthetic_typed_graph_multiseed_replication_acquisition_v2"
ACTION_PACK_SCHEMA = "synthetic_typed_graph_multiseed_action_pack_v2"
ACTION_ITEM_SCHEMA = "synthetic_typed_graph_multiseed_action_item_v2"
LABEL_PACK_SCHEMA = "synthetic_typed_graph_multiseed_label_pack_v2"
LABEL_ITEM_SCHEMA = "synthetic_typed_graph_multiseed_label_item_v2"
COMPILED_COHORT_PACK_SCHEMA = "synthetic_typed_graph_multiseed_compiled_cohort_pack_v2"
RESULT_SCHEMA = "synthetic_typed_graph_multiseed_replication_result_v2"
PUBLICATION_SCHEMA = "synthetic_typed_graph_multiseed_terminal_reproducibility_v2"

DESIGN_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_design_v2.json"
)
INTEGRATION_DIAGNOSTIC_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_integration_diagnostic_v2.json"
)
IMPLEMENTATION_FREEZE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_implementation_freeze_v2.json"
)
ARTIFACT_ROOT_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_multiseed_replication_v2"
)
INTEGRATION_DIAGNOSTIC_MARKER_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "integration_diagnostic/attempt.marker"
)
INTEGRATION_DIAGNOSTIC_ACTION_SEAL_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "integration_diagnostic/action_seal.json"
)
SEED_MARKER_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "seed_generation.attempt.marker"
SEED_BATCH_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "seed_batch.bin"
SEED_CUSTODY_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_seed_custody_v2.json"
)
SEED_FAILURE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_seed_failure_v2.json"
)
ACQUISITION_MARKER_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "acquisition.attempt.marker"
ACTION_PACK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "action_pack.json"
LABEL_PACK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "label_pack.json"
COMPILED_COHORT_PACK_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "full_compiled_cohort_pack.json"
)
ACQUISITION_RECEIPT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_acquisition_v2.json"
)
RESULT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_result_v2.json"
)
PUBLICATION_MARKER_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "publish_terminal.attempt.marker"
)
PUBLICATION_RELATIVE_PATH = Path(
    "published/synthetic_typed_graph_multiseed_replication_v2/formal_seeds_and_cohort.json"
)
PUBLICATION_FAILURE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_publication_failure_v2.json"
)
RUNNER_VERSION = "synthetic_typed_graph_multiseed_runner_v2"
RUNNER_MARKER_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "runner/formal.attempt.marker"
RUNNER_WORK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "runner/formal.work"
RUNNER_ACTION_SEAL_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "runner/action_seal.json"
)
SUCCESS_RESULT_STATUS = "terminal_descriptive_eight_seed_replication_complete"
FAILURE_RESULT_STATUS = "terminal_infrastructure_or_implementation_invalid_no_replay"
ORIGINAL_PUBLICATION_RELATIVE_PATH = Path(
    "published/synthetic_typed_graph_causal_v1/formal_seed_and_cohort.json"
)
V1_PUBLICATION_RELATIVE_PATH = Path(
    "published/synthetic_typed_graph_multiseed_replication_v1/formal_seeds_and_cohort.json"
)

REQUIRED_FREEZE_PATHS = frozenset(
    {
        DESIGN_RELATIVE_PATH.as_posix(),
        INTEGRATION_DIAGNOSTIC_RELATIVE_PATH.as_posix(),
        "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_acquisition_v2.py",
        "tests/test_synthetic_typed_graph_multiseed_acquisition_v2.py",
        "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v2.py",
        "tests/test_synthetic_typed_graph_multiseed_runner_v2.py",
        "assumption_agent/benchmarks/synthetic_typed_graph_causal_grammar_v1.py",
        "assumption_agent/benchmarks/contractnli_typed_clause_graph_v1.py",
        "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_acquisition_v1.py",
        "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v1.py",
        "manifests/synthetic_typed_graph_multiseed_replication_design_v1.json",
        "manifests/synthetic_typed_graph_multiseed_replication_implementation_freeze_v1.json",
        "manifests/synthetic_typed_graph_multiseed_replication_acquisition_v1.json",
        "manifests/synthetic_typed_graph_multiseed_replication_result_v1.json",
        ORIGINAL_PUBLICATION_RELATIVE_PATH.as_posix(),
        V1_PUBLICATION_RELATIVE_PATH.as_posix(),
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

DIAGNOSTIC_BINDING_PATHS = tuple(
    sorted(
        (
            "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_acquisition_v2.py",
            "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v2.py",
            "tests/test_synthetic_typed_graph_multiseed_acquisition_v2.py",
            "tests/test_synthetic_typed_graph_multiseed_runner_v2.py",
        )
    )
)

_FORMAL_ENTRY_ACTIVE = False


class SyntheticMultiseedV2AcquisitionError(RuntimeError):
    """A v2 custody, acquisition, publication, or persistence invariant failed."""


SyntheticMultiseedAcquisitionError = SyntheticMultiseedV2AcquisitionError


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
        raise SyntheticMultiseedV2AcquisitionError("value is not canonical JSON") from exc


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
        raise SyntheticMultiseedV2AcquisitionError("self-hash field already exists")
    return {**dict(body), field: semantic_hash(dict(body))}


def _assert_no_symlink_components(path: Path, field: str) -> Path:
    absolute = path.expanduser().absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise SyntheticMultiseedV2AcquisitionError(
                f"{field} contains a symbolic link"
            )
    return absolute


def _write_exclusive(path: Path, raw: bytes, mode: int) -> str:
    absolute = path.expanduser().absolute()
    if absolute.exists() or absolute.is_symlink():
        raise SyntheticMultiseedV2AcquisitionError(
            f"exclusive output already exists: {absolute.name}"
        )
    absolute.parent.mkdir(
        parents=True, exist_ok=True, mode=0o755 if mode == PUBLIC_MODE else 0o700
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
        raise SyntheticMultiseedV2AcquisitionError(f"{field} mode or type drifted")
    raw = absolute.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticMultiseedV2AcquisitionError(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SyntheticMultiseedV2AcquisitionError(f"{field} root drifted")
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
        raise SyntheticMultiseedV2AcquisitionError("Git freeze verification failed") from exc
    return completed.stdout


def _git_project_prefix(project_root: Path) -> str:
    git_root = Path(
        _git(project_root, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    try:
        relative = project_root.resolve().relative_to(git_root)
    except ValueError as exc:
        raise SyntheticMultiseedV2AcquisitionError("project root is outside Git") from exc
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
    if (
        not path.is_file()
        or path.is_symlink()
        or stat.S_IMODE(path.stat().st_mode) != PUBLIC_MODE
    ):
        raise SyntheticMultiseedV2AcquisitionError(f"{field} is unavailable")
    raw = path.read_bytes()
    if _committed_bytes(project_root, relative_path) != raw:
        raise SyntheticMultiseedV2AcquisitionError(
            f"{field} is not current-HEAD committed"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticMultiseedV2AcquisitionError(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SyntheticMultiseedV2AcquisitionError(f"{field} root drifted")
    return payload


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_frozen_design(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    path = _assert_no_symlink_components(root / DESIGN_RELATIVE_PATH, "design")
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != DESIGN_FILE_SHA256
    ):
        raise SyntheticMultiseedV2AcquisitionError("frozen v2 design drifted")
    try:
        design = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticMultiseedV2AcquisitionError("frozen design is unreadable") from exc
    if not isinstance(design, dict) or design.get("schema") != DESIGN_SCHEMA:
        raise SyntheticMultiseedV2AcquisitionError("frozen design schema drifted")
    body = dict(design)
    declared = body.pop("design_sha256", None)
    if declared != DESIGN_SHA256 or semantic_hash(body) != declared:
        raise SyntheticMultiseedV2AcquisitionError("frozen design self-hash drifted")
    artifact_bindings = design.get("artifact_bindings")
    if not isinstance(artifact_bindings, Mapping):
        raise SyntheticMultiseedV2AcquisitionError(
            "frozen design artifact bindings drifted"
        )
    for binding in artifact_bindings.values():
        if not isinstance(binding, Mapping):
            raise SyntheticMultiseedV2AcquisitionError(
                "frozen design artifact binding row drifted"
            )
        for key, relative_value in binding.items():
            if not key.endswith("relative_path"):
                continue
            hash_key = (
                "file_sha256"
                if key == "relative_path"
                else f"{key[:-len('_relative_path')]}_file_sha256"
            )
            expected_hash = binding.get(hash_key)
            if not isinstance(relative_value, str) or not _is_lower_hex(
                expected_hash, 64
            ):
                raise SyntheticMultiseedV2AcquisitionError(
                    "frozen design artifact path/hash pair drifted"
                )
            artifact_path = _assert_no_symlink_components(
                root / relative_value, "design-bound artifact"
            )
            if (
                not artifact_path.is_file()
                or artifact_path.is_symlink()
                or sha256_file(artifact_path) != expected_hash
            ):
                raise SyntheticMultiseedV2AcquisitionError(
                    "design-bound artifact bytes drifted"
                )
    return design


def _validate_integration_diagnostic_payload(
    diagnostic: Mapping[str, Any], *, expected_file_hash: str | None = None
) -> str:
    body = dict(diagnostic)
    declared = body.pop("diagnostic_sha256", None)
    if declared is None:
        declared = body.pop("receipt_sha256", None)
    else:
        if "receipt_sha256" in body:
            raise SyntheticMultiseedV2AcquisitionError(
                "integration diagnostic has ambiguous self hash"
            )
    bindings = diagnostic.get("bindings")
    source = diagnostic.get("source_v1_publication")
    schedule = diagnostic.get("chunk_schedule")
    counts = diagnostic.get("counts")
    expected_count_values = {
        "item_count": TOTAL_ITEM_COUNT,
        "action_work_unit_count": TOTAL_ITEM_COUNT * 3,
        "submitted_action_work_unit_count": TOTAL_ITEM_COUNT * 3,
        "terminal_action_work_unit_count": TOTAL_ITEM_COUNT * 3,
        "official_retrieve_action_count": TOTAL_ITEM_COUNT,
        "RAW_action_count": TOTAL_ITEM_COUNT,
        "Agent_R1_action_count": TOTAL_ITEM_COUNT,
    }
    if (
        diagnostic.get("schema") != INTEGRATION_DIAGNOSTIC_SCHEMA
        or diagnostic.get("version") != VERSION
        or diagnostic.get("status") != INTEGRATION_SUCCESS_STATUS
        or diagnostic.get("design_sha256") != DESIGN_SHA256
        or diagnostic.get("design_file_sha256") != DESIGN_FILE_SHA256
        or not isinstance(bindings, list)
        or len(bindings) != len(DIAGNOSTIC_BINDING_PATHS)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"relative_path", "file_sha256", "git_blob_sha1"}
            or row.get("relative_path") != DIAGNOSTIC_BINDING_PATHS[index]
            or not _is_lower_hex(row.get("file_sha256"), 64)
            or not _is_lower_hex(row.get("git_blob_sha1"), 40)
            for index, row in enumerate(bindings)
        )
        or not isinstance(source, Mapping)
        or set(source)
        != {
            "file_sha256",
            "reproducibility_sha256",
            "generated_item_commitment_set_sha256",
            "projected_action_pack_sha256",
            "projected_action_item_commitment_set_sha256",
            "source_label_free_commitment_set_sha256",
        }
        or source.get("file_sha256") != V1_PUBLICATION_FILE_SHA256
        or source.get("reproducibility_sha256") != V1_PUBLICATION_SHA256
        or source.get("generated_item_commitment_set_sha256")
        != V1_ITEM_COMMITMENT_SET_SHA256
        or not _is_lower_hex(source.get("projected_action_pack_sha256"), 64)
        or not _is_lower_hex(
            source.get("projected_action_item_commitment_set_sha256"), 64
        )
        or not _is_lower_hex(
            source.get("source_label_free_commitment_set_sha256"), 64
        )
        or not isinstance(schedule, Mapping)
        or dict(schedule)
        != {
            "chunk_count": 2,
            "texts_per_chunk": 8448,
            "total_text_count": 16896,
            "chunk_schedule_sha256": CHUNK_SCHEDULE_SHA256,
        }
        or not isinstance(counts, Mapping)
        or dict(counts) != expected_count_values
        or diagnostic.get("arms") != ["RAW", "official_HippoRAG", "Agent_R1"]
        or diagnostic.get("official_concurrency_cap") != 8
        or diagnostic.get("local_concurrency_cap") != 64
        or diagnostic.get("observed_encoder_output_row_counts") != [8448, 8448]
        or diagnostic.get("observed_encoder_input_row_counts") != [8448, 8448]
        or not isinstance(diagnostic.get("official_peak_concurrency_count"), int)
        or not 1 <= diagnostic["official_peak_concurrency_count"] <= 8
        or not isinstance(diagnostic.get("local_peak_concurrency_count"), int)
        or not 1 <= diagnostic["local_peak_concurrency_count"] <= 64
        or any(
            not _is_lower_hex(diagnostic.get(field), 64)
            for field in (
                "runtime_binding_sha256",
                "official_postflight_receipt_sha256",
                "action_table_sha256",
                "action_seal_sha256",
                "action_seal_file_sha256",
                "diagnostic_attempt_marker_sha256",
                "diagnostic_attempt_marker_file_sha256",
            )
        )
        or diagnostic.get("labels_opened") is not False
        or diagnostic.get("scores_computed") is not False
        or diagnostic.get("estimands_computed") is not False
        or diagnostic.get("claims_made") is not False
        or diagnostic.get("network_calls") != 0
        or diagnostic.get("retrieval_actions_model_outputs_or_scores_disclosed")
        is not False
        or diagnostic.get("action_rows_or_ranked_indices_persisted") is not False
        or diagnostic.get("action_identity_or_quality_used_for_decision") is not False
        or diagnostic.get("diagnostic_is_non_claim") is not True
        or diagnostic.get("fresh_formal_seed_authorized") is not True
        or not _is_lower_hex(diagnostic.get("invocation_HEAD"), 40)
        or not _is_lower_hex(declared, 64)
        or semantic_hash(body) != declared
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "successful integration diagnostic binding drifted"
        )
    if expected_file_hash is not None and not _is_lower_hex(expected_file_hash, 64):
        raise SyntheticMultiseedV2AcquisitionError(
            "integration diagnostic file hash drifted"
        )
    return str(declared)


def load_committed_integration_diagnostic(
    project_root: Path,
) -> tuple[dict[str, Any], str, str]:
    root = project_root.resolve(strict=True)
    diagnostic = _load_committed_public_json(
        root, INTEGRATION_DIAGNOSTIC_RELATIVE_PATH, "integration diagnostic"
    )
    file_hash = sha256_file(root / INTEGRATION_DIAGNOSTIC_RELATIVE_PATH)
    receipt_hash = _validate_integration_diagnostic_payload(
        diagnostic, expected_file_hash=file_hash
    )
    bindings = diagnostic["bindings"]
    for row, relative_text in zip(bindings, DIAGNOSTIC_BINDING_PATHS, strict=True):
        relative = Path(relative_text)
        path = _assert_no_symlink_components(root / relative, "diagnostic code binding")
        if not path.is_file() or path.is_symlink():
            raise SyntheticMultiseedV2AcquisitionError(
                "diagnostic-bound implementation file is unavailable"
            )
        worktree = path.read_bytes()
        committed = _committed_bytes(root, relative)
        if (
            worktree != committed
            or row.get("file_sha256") != hashlib.sha256(committed).hexdigest()
            or row.get("git_blob_sha1") != _git_blob_sha1(committed)
        ):
            raise SyntheticMultiseedV2AcquisitionError(
                "implementation drifted after the committed diagnostic"
            )
    marker, marker_file_hash = _read_json_with_mode(
        root / INTEGRATION_DIAGNOSTIC_MARKER_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="integration diagnostic attempt marker",
    )
    marker_body = dict(marker)
    marker_hash = marker_body.pop("marker_sha256", None)
    if (
        marker.get("schema")
        != f"{INTEGRATION_DIAGNOSTIC_SCHEMA}_attempt_marker"
        or marker.get("version") != VERSION
        or marker.get("status")
        != "sole_public_label_free_integration_diagnostic_attempt_consumed"
        or marker.get("actual_HEAD") != diagnostic.get("invocation_HEAD")
        or marker.get("design_sha256") != DESIGN_SHA256
        or marker.get("bindings_sha256") != semantic_hash(bindings)
        or marker.get("attempt_count") != 1
        or marker.get("fresh_formal_seed_or_cohort_exists") is not False
        or not _is_lower_hex(marker_hash, 64)
        or semantic_hash(marker_body) != marker_hash
        or diagnostic.get("diagnostic_attempt_marker_sha256") != marker_hash
        or diagnostic.get("diagnostic_attempt_marker_file_sha256")
        != marker_file_hash
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "integration diagnostic attempt marker chain drifted"
        )
    seal, seal_file_hash = _read_json_with_mode(
        root / INTEGRATION_DIAGNOSTIC_ACTION_SEAL_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="integration diagnostic private action seal",
    )
    if (
        set(seal)
        != {
            "schema",
            "total_action_count",
            "arm_terminal_counts",
            "ordered_action_commitment_set_sha256",
            "official_peak_concurrency",
            "local_peak_concurrency",
            "postflight_receipt_sha256",
            "action_rows_or_ranked_indices_persisted",
        }
        or seal.get("schema")
        != "synthetic_typed_graph_multiseed_replication_integration_diagnostic_private_action_seal_v2"
        or seal.get("total_action_count") != TOTAL_ITEM_COUNT * 3
        or seal.get("arm_terminal_counts")
        != {"RAW": TOTAL_ITEM_COUNT, "official_HippoRAG": TOTAL_ITEM_COUNT, "Agent_R1": TOTAL_ITEM_COUNT}
        or seal.get("ordered_action_commitment_set_sha256")
        != diagnostic.get("action_table_sha256")
        or seal.get("official_peak_concurrency")
        != diagnostic.get("official_peak_concurrency_count")
        or seal.get("local_peak_concurrency")
        != diagnostic.get("local_peak_concurrency_count")
        or seal.get("postflight_receipt_sha256")
        != diagnostic.get("official_postflight_receipt_sha256")
        or seal.get("action_rows_or_ranked_indices_persisted") is not False
        or semantic_hash(seal) != diagnostic.get("action_seal_sha256")
        or seal_file_hash != diagnostic.get("action_seal_file_sha256")
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "integration diagnostic private action seal chain drifted"
        )
    return diagnostic, file_hash, receipt_hash


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
    diagnostic, diagnostic_file_hash, diagnostic_hash = (
        load_committed_integration_diagnostic(root)
    )
    output = root / IMPLEMENTATION_FREEZE_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise SyntheticMultiseedV2AcquisitionError(
            "implementation freeze already exists"
        )
    if any((root / relative).exists() for relative in _formal_output_paths()):
        raise SyntheticMultiseedV2AcquisitionError("formal v2 output already exists")
    bindings: list[dict[str, str]] = []
    for relative in sorted(REQUIRED_FREEZE_PATHS):
        path = _assert_no_symlink_components(root / relative, "freeze binding")
        if not path.is_file() or path.is_symlink():
            raise SyntheticMultiseedV2AcquisitionError(
                f"required freeze file unavailable: {relative}"
            )
        worktree = path.read_bytes()
        committed = _committed_bytes(root, Path(relative))
        if worktree != committed:
            raise SyntheticMultiseedV2AcquisitionError(
                f"required freeze file is not HEAD-clean: {relative}"
            )
        bindings.append(
            {
                "relative_path": relative,
                "file_sha256": hashlib.sha256(committed).hexdigest(),
                "git_blob_sha1": _git_blob_sha1(committed),
            }
        )
    diagnostic_binding_rows = [
        row
        for row in bindings
        if row["relative_path"] in DIAGNOSTIC_BINDING_PATHS
    ]
    if diagnostic_binding_rows != diagnostic["bindings"]:
        raise SyntheticMultiseedV2AcquisitionError(
            "formal freeze code/test tuples differ from diagnostic tuples"
        )
    body = {
        "schema": FREEZE_SCHEMA,
        "version": VERSION,
        "status": "complete_postdiagnostic_preseed_implementation_frozen_must_commit_before_seed",
        "creation_HEAD": _git(root, "rev-parse", "HEAD").decode("ascii").strip(),
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "integration_diagnostic_sha256": diagnostic_hash,
        "integration_diagnostic_file_sha256": diagnostic_file_hash,
        "integration_diagnostic_invocation_HEAD": diagnostic["invocation_HEAD"],
        "integration_diagnostic_status": diagnostic["status"],
        "chunk_schedule_sha256": CHUNK_SCHEDULE_SHA256,
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
    diagnostic, diagnostic_file_hash, diagnostic_hash = (
        load_committed_integration_diagnostic(root)
    )
    body = dict(freeze)
    declared = body.pop("implementation_freeze_sha256", None)
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("status")
        != "complete_postdiagnostic_preseed_implementation_frozen_must_commit_before_seed"
        or freeze.get("design_sha256") != DESIGN_SHA256
        or freeze.get("design_file_sha256") != DESIGN_FILE_SHA256
        or freeze.get("integration_diagnostic_sha256") != diagnostic_hash
        or freeze.get("integration_diagnostic_file_sha256") != diagnostic_file_hash
        or freeze.get("integration_diagnostic_invocation_HEAD")
        != diagnostic.get("invocation_HEAD")
        or freeze.get("integration_diagnostic_status") != INTEGRATION_SUCCESS_STATUS
        or freeze.get("chunk_schedule_sha256") != CHUNK_SCHEDULE_SHA256
        or freeze.get("formal_seed_or_cohort_exists") is not False
        or not _is_lower_hex(declared, 64)
        or semantic_hash(body) != declared
    ):
        raise SyntheticMultiseedV2AcquisitionError("implementation freeze drifted")
    rows = freeze.get("bindings")
    if not isinstance(rows, list):
        raise SyntheticMultiseedV2AcquisitionError("implementation bindings absent")
    by_path: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "file_sha256",
            "git_blob_sha1",
        }:
            raise SyntheticMultiseedV2AcquisitionError(
                "implementation binding schema drifted"
            )
        relative = row.get("relative_path")
        if not isinstance(relative, str) or relative in by_path:
            raise SyntheticMultiseedV2AcquisitionError(
                "implementation binding path drifted"
            )
        by_path[relative] = row
    if set(by_path) != set(REQUIRED_FREEZE_PATHS):
        raise SyntheticMultiseedV2AcquisitionError(
            "implementation freeze path set drifted"
        )
    for relative, row in by_path.items():
        path = _assert_no_symlink_components(root / relative, "implementation binding")
        if not path.is_file() or path.is_symlink():
            raise SyntheticMultiseedV2AcquisitionError(
                "implementation-bound file unavailable"
            )
        worktree = path.read_bytes()
        committed = _committed_bytes(root, Path(relative))
        if worktree != committed:
            raise SyntheticMultiseedV2AcquisitionError(
                "implementation-bound worktree differs from HEAD"
            )
        if (
            row.get("file_sha256") != hashlib.sha256(committed).hexdigest()
            or row.get("git_blob_sha1") != _git_blob_sha1(committed)
        ):
            raise SyntheticMultiseedV2AcquisitionError(
                "implementation binding hash drifted"
            )
    return freeze, actual_head


def _require_formal_entry() -> None:
    if _FORMAL_ENTRY_ACTIVE is not True:
        raise SyntheticMultiseedV2AcquisitionError(
            "formal operation may only be consumed by the acquisition CLI"
        )


def _load_v1_publication_projection(root: Path) -> tuple[frozenset[str], frozenset[str]]:
    path = _assert_no_symlink_components(
        root / V1_PUBLICATION_RELATIVE_PATH, "v1 multiseed publication"
    )
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != V1_PUBLICATION_FILE_SHA256
    ):
        raise SyntheticMultiseedV2AcquisitionError("v1 publication file drifted")
    try:
        publication = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticMultiseedV2AcquisitionError(
            "v1 publication is unreadable"
        ) from exc
    if not isinstance(publication, dict):
        raise SyntheticMultiseedV2AcquisitionError("v1 publication root drifted")
    body = dict(publication)
    declared = body.pop("reproducibility_sha256", None)
    seeds = publication.get("ordered_seed_commitments_sha256")
    rows = publication.get("items")
    if (
        publication.get("schema")
        != "synthetic_typed_graph_multiseed_terminal_reproducibility_v1"
        or publication.get("status")
        != "terminal_eight_seeds_and_full_compiled_cohort_published"
        or declared != V1_PUBLICATION_SHA256
        or semantic_hash(body) != declared
        or seeds != list(V1_ORDERED_SEED_COMMITMENTS)
        or not isinstance(rows, list)
        or len(rows) != TOTAL_ITEM_COUNT
    ):
        raise SyntheticMultiseedV2AcquisitionError("v1 publication binding drifted")
    items = [
        row.get("item_commitment_sha256") if isinstance(row, Mapping) else None
        for row in rows
    ]
    if (
        any(not _is_lower_hex(value, 64) for value in items)
        or len(set(items)) != TOTAL_ITEM_COUNT
        or stable_hash(items) != V1_ITEM_COMMITMENT_SET_SHA256
        or publication.get("generated_item_commitment_set_sha256")
        != V1_ITEM_COMMITMENT_SET_SHA256
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "v1 publication item projection drifted"
        )
    return frozenset(V1_ORDERED_SEED_COMMITMENTS), frozenset(items)


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
        raise SyntheticMultiseedV2AcquisitionError(
            "canonical v2 seed attempt already exists"
        )
    marker = _self_hashed(
        {
            "schema": f"{SEED_CUSTODY_SCHEMA}_attempt_marker",
            "version": VERSION,
            "status": "sole_v2_eight_seed_batch_generation_attempt_consumed",
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
        forbidden_seed_commitments = frozenset(
            {ORIGINAL_SEED_COMMITMENT_SHA256, *V1_ORDERED_SEED_COMMITMENTS}
        )
        batch = os.urandom(SEED_BATCH_BYTES)
        if not isinstance(batch, bytes) or len(batch) != SEED_BATCH_BYTES:
            raise SyntheticMultiseedV2AcquisitionError(
                "OS random source did not return exactly 256 bytes"
            )
        seeds = tuple(
            batch[index * SEED_BYTES : (index + 1) * SEED_BYTES]
            for index in range(SEED_COUNT)
        )
        commitments = [hashlib.sha256(seed).hexdigest() for seed in seeds]
        if len(set(seeds)) != SEED_COUNT:
            raise SyntheticMultiseedV2AcquisitionError(
                "duplicate fresh seeds make the formal attempt terminal"
            )
        if forbidden_seed_commitments.intersection(commitments):
            raise SyntheticMultiseedV2AcquisitionError(
                "fresh seed collides with an original or v1 published seed"
            )
        batch_commitment = hashlib.sha256(batch).hexdigest()
        persisted = _write_exclusive(
            root / SEED_BATCH_RELATIVE_PATH, batch, PRIVATE_MODE
        )
        if persisted != batch_commitment:
            raise SyntheticMultiseedV2AcquisitionError(
                "seed batch persistence drifted"
            )
        custody = _self_hashed(
            {
                "schema": SEED_CUSTODY_SCHEMA,
                "version": VERSION,
                "status": "eight_fresh_v2_seeds_committed_cohort_not_generated",
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
                "v1_ordered_seed_commitments_sha256": list(
                    V1_ORDERED_SEED_COMMITMENTS
                ),
                "fresh_seeds_disjoint_from_original_and_v1": True,
                "seed_generation": "one_os.urandom_256_call_then_ordered_8x32_split_after_marker",
                "seed_material_published": False,
                "cohort_generated": False,
                "attempts_allowed": 1,
                "retry_replacement_or_interim_allowed": False,
            },
            "custody_sha256",
        )
        _write_json_exclusive(root / SEED_CUSTODY_RELATIVE_PATH, custody, PUBLIC_MODE)
        return custody
    except Exception as exc:
        failure = _self_hashed(
            {
                "schema": f"{SEED_CUSTODY_SCHEMA}_failure_receipt",
                "version": VERSION,
                "status": "terminal_v2_seed_batch_invalid_no_replacement",
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
    absolute = _assert_no_symlink_components(path, "v2 seed custody")
    root = absolute.parent.parent
    if absolute != root / SEED_CUSTODY_RELATIVE_PATH:
        raise SyntheticMultiseedV2AcquisitionError(
            "seed custody path is not canonical"
        )
    custody, _file_hash = _read_json_with_mode(
        absolute, expected_mode=PUBLIC_MODE, field="v2 seed custody"
    )
    body = dict(custody)
    declared = body.pop("custody_sha256", None)
    commitments = custody.get("ordered_seed_commitments_sha256")
    forbidden = frozenset(
        {ORIGINAL_SEED_COMMITMENT_SHA256, *V1_ORDERED_SEED_COMMITMENTS}
    )
    if (
        custody.get("schema") != SEED_CUSTODY_SCHEMA
        or custody.get("status")
        != "eight_fresh_v2_seeds_committed_cohort_not_generated"
        or custody.get("design_sha256") != DESIGN_SHA256
        or custody.get("seed_count") != SEED_COUNT
        or custody.get("seed_bytes_each") != SEED_BYTES
        or custody.get("seed_batch_bytes") != SEED_BATCH_BYTES
        or custody.get("original_seed_commitment_sha256")
        != ORIGINAL_SEED_COMMITMENT_SHA256
        or custody.get("v1_ordered_seed_commitments_sha256")
        != list(V1_ORDERED_SEED_COMMITMENTS)
        or custody.get("fresh_seeds_disjoint_from_original_and_v1") is not True
        or custody.get("seed_material_published") is not False
        or custody.get("cohort_generated") is not False
        or custody.get("attempts_allowed") != 1
        or custody.get("retry_replacement_or_interim_allowed") is not False
        or not _is_lower_hex(declared, 64)
        or semantic_hash(body) != declared
        or not isinstance(commitments, list)
        or len(commitments) != SEED_COUNT
        or len(set(commitments)) != SEED_COUNT
        or forbidden.intersection(commitments)
        or any(not _is_lower_hex(value, 64) for value in commitments)
    ):
        raise SyntheticMultiseedV2AcquisitionError("seed custody binding drifted")
    batch_path = _assert_no_symlink_components(
        root / SEED_BATCH_RELATIVE_PATH, "v2 seed batch"
    )
    if (
        not batch_path.is_file()
        or batch_path.is_symlink()
        or batch_path.stat().st_size != SEED_BATCH_BYTES
        or stat.S_IMODE(batch_path.stat().st_mode) != PRIVATE_MODE
    ):
        raise SyntheticMultiseedV2AcquisitionError("seed batch mode or size drifted")
    marker, marker_file_hash = _read_json_with_mode(
        root / SEED_MARKER_RELATIVE_PATH,
        expected_mode=PUBLIC_MODE,
        field="v2 seed attempt marker",
    )
    marker_body = dict(marker)
    marker_hash = marker_body.pop("marker_sha256", None)
    if (
        marker.get("schema") != f"{SEED_CUSTODY_SCHEMA}_attempt_marker"
        or marker.get("status")
        != "sole_v2_eight_seed_batch_generation_attempt_consumed"
        or marker.get("attempt_count") != 1
        or marker.get("entropy_call_bytes") != SEED_BATCH_BYTES
        or marker.get("design_sha256") != DESIGN_SHA256
        or marker.get("implementation_freeze_sha256")
        != custody.get("implementation_freeze_sha256")
        or not _is_lower_hex(marker_hash, 64)
        or semantic_hash(marker_body) != marker_hash
        or custody.get("seed_attempt_marker_sha256") != marker_hash
        or custody.get("seed_attempt_marker_file_sha256") != marker_file_hash
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "v2 seed attempt marker chain drifted"
        )
    return custody


def _read_seed_batch(path: Path, custody: Mapping[str, Any]) -> tuple[bytes, ...]:
    absolute = _assert_no_symlink_components(path, "v2 seed batch")
    if (
        not absolute.is_file()
        or absolute.is_symlink()
        or absolute.stat().st_size != SEED_BATCH_BYTES
        or stat.S_IMODE(absolute.stat().st_mode) != PRIVATE_MODE
    ):
        raise SyntheticMultiseedV2AcquisitionError("seed batch mode or size drifted")
    raw = absolute.read_bytes()
    if hashlib.sha256(raw).hexdigest() != custody.get(
        "seed_batch_commitment_sha256"
    ):
        raise SyntheticMultiseedV2AcquisitionError("seed batch commitment drifted")
    seeds = tuple(
        raw[index * SEED_BYTES : (index + 1) * SEED_BYTES]
        for index in range(SEED_COUNT)
    )
    commitments = [hashlib.sha256(seed).hexdigest() for seed in seeds]
    forbidden = frozenset(
        {ORIGINAL_SEED_COMMITMENT_SHA256, *V1_ORDERED_SEED_COMMITMENTS}
    )
    if commitments != custody.get("ordered_seed_commitments_sha256"):
        raise SyntheticMultiseedV2AcquisitionError(
            "ordered seed commitments drifted"
        )
    if len(set(seeds)) != SEED_COUNT or forbidden.intersection(commitments):
        raise SyntheticMultiseedV2AcquisitionError(
            "seed collision detected after custody"
        )
    return seeds


def _load_original_A_hold_commitments(root: Path) -> frozenset[str]:
    path = _assert_no_symlink_components(
        root / ORIGINAL_PUBLICATION_RELATIVE_PATH, "original publication"
    )
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != ORIGINAL_PUBLICATION_FILE_SHA256
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "original publication file drifted"
        )
    try:
        publication = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticMultiseedV2AcquisitionError(
            "original publication is unreadable"
        ) from exc
    if not isinstance(publication, dict):
        raise SyntheticMultiseedV2AcquisitionError(
            "original publication root drifted"
        )
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
        raise SyntheticMultiseedV2AcquisitionError(
            "original publication binding drifted"
        )
    commitments = [
        row.get("item_commitment_sha256") if isinstance(row, Mapping) else None
        for row in rows
    ]
    if (
        any(not _is_lower_hex(value, 64) for value in commitments)
        or len(set(commitments)) != ITEMS_PER_SEED
        or stable_hash(commitments) != ORIGINAL_A_HOLD_COMMITMENT_SET_SHA256
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "original A_hold projection drifted"
        )
    return frozenset(commitments)


def _load_prior_item_commitments_after_marker(root: Path) -> frozenset[str]:
    original = _load_original_A_hold_commitments(root)
    _v1_seeds, v1_items = _load_v1_publication_projection(root)
    if original.intersection(v1_items):
        raise SyntheticMultiseedV2AcquisitionError(
            "historical published cohorts unexpectedly overlap"
        )
    return frozenset((*original, *v1_items))


def _validate_compiled_item(item: grammar.CompiledItem, ordinal: int) -> None:
    if item.block != BLOCK or item.block_ordinal != ordinal:
        raise SyntheticMultiseedV2AcquisitionError(
            "compiled A_hold ordering drifted"
        )
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
        raise SyntheticMultiseedV2AcquisitionError(
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
    """Generate exactly one 64-row ``A_hold`` block for each fresh seed."""

    _require_formal_entry()
    root = project_root.resolve(strict=True)
    verify_frozen_design(root)
    freeze, actual_head = verify_implementation_freeze(root)
    committed_custody = _load_committed_public_json(
        root, SEED_CUSTODY_RELATIVE_PATH, "v2 seed custody"
    )
    custody = load_seed_custody(root / SEED_CUSTODY_RELATIVE_PATH)
    if (
        committed_custody != custody
        or custody.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "committed seed custody chain drifted"
        )
    outputs = (
        root / ACQUISITION_MARKER_RELATIVE_PATH,
        root / ACTION_PACK_RELATIVE_PATH,
        root / LABEL_PACK_RELATIVE_PATH,
        root / COMPILED_COHORT_PACK_RELATIVE_PATH,
        root / ACQUISITION_RECEIPT_RELATIVE_PATH,
        root / RESULT_RELATIVE_PATH,
        root / PUBLICATION_MARKER_RELATIVE_PATH,
        root / PUBLICATION_RELATIVE_PATH,
    )
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise SyntheticMultiseedV2AcquisitionError(
            "canonical v2 acquisition attempt already exists"
        )
    marker = _self_hashed(
        {
            "schema": f"{ACQUISITION_SCHEMA}_attempt_marker",
            "version": VERSION,
            "status": "sole_v2_cohort_generation_attempt_consumed",
            "actual_HEAD": actual_head,
            "design_sha256": DESIGN_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "custody_sha256": custody["custody_sha256"],
            "attempt_count": 1,
            "grammar_calls_authorized": SEED_COUNT,
            "block": BLOCK,
        },
        "marker_sha256",
    )
    marker_file_hash = _write_json_exclusive(
        root / ACQUISITION_MARKER_RELATIVE_PATH, marker, PRIVATE_MODE
    )
    try:
        seeds = _read_seed_batch(root / SEED_BATCH_RELATIVE_PATH, custody)
        prior_commitments = _load_prior_item_commitments_after_marker(root)
        compiled: list[tuple[int, grammar.CompiledItem]] = []
        for seed_index, seed in enumerate(seeds):
            items = grammar.generate_block(seed, BLOCK)
            if len(items) != ITEMS_PER_SEED:
                raise SyntheticMultiseedV2AcquisitionError(
                    "public grammar did not return exactly 64 A_hold rows"
                )
            for seed_ordinal, item in enumerate(items):
                _validate_compiled_item(item, seed_ordinal)
                compiled.append((seed_index, item))
        if len(compiled) != TOTAL_ITEM_COUNT:
            raise SyntheticMultiseedV2AcquisitionError(
                "compiled cohort count drifted"
            )
        item_commitments = [item.item_commitment_sha256 for _, item in compiled]
        if len(set(item_commitments)) != TOTAL_ITEM_COUNT:
            raise SyntheticMultiseedV2AcquisitionError(
                "new cohort item commitments overlap each other"
            )
        if prior_commitments.intersection(item_commitments):
            raise SyntheticMultiseedV2AcquisitionError(
                "new cohort overlaps the original or v1 published cohort"
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
            compiled_body = _compiled_public_row(
                item, seed_index=seed_index, global_ordinal=global_ordinal
            )
            action_rows.append(action)
            label_rows.append(label)
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
                "status": "formal_v2_multiseed_A_hold_cohort_acquired_private_labels_separated",
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
                "grammar_generate_block_call_count": SEED_COUNT,
                "generated_item_commitment_set_sha256": stable_hash(
                    item_commitments
                ),
                "original_A_hold_commitment_set_sha256": ORIGINAL_A_HOLD_COMMITMENT_SET_SHA256,
                "v1_multiseed_item_commitment_set_sha256": V1_ITEM_COMMITMENT_SET_SHA256,
                "new_original_and_v1_item_commitments_pairwise_disjoint": True,
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
                "status": "terminal_v2_multiseed_acquisition_invalid_no_replay",
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
        or not _is_lower_hex(declared, 64)
        or semantic_hash(body) != declared
        or file_hash != expected_file_hash
        or not isinstance(rows, list)
        or len(rows) != TOTAL_ITEM_COUNT
    ):
        raise SyntheticMultiseedV2AcquisitionError(f"{schema} binding drifted")
    hashes: list[str] = []
    for global_ordinal, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or row.get("global_ordinal") != global_ordinal
            or row.get("seed_index") != global_ordinal // ITEMS_PER_SEED
            or row.get("seed_ordinal") != global_ordinal % ITEMS_PER_SEED
        ):
            raise SyntheticMultiseedV2AcquisitionError(f"{schema} ordering drifted")
        row_body = dict(row)
        row_hash = row_body.pop(item_hash_field, None)
        if not _is_lower_hex(row_hash, 64) or semantic_hash(row_body) != row_hash:
            raise SyntheticMultiseedV2AcquisitionError(f"{schema} row hash drifted")
        hashes.append(str(row_hash))
    if stable_hash(hashes) != expected_set_hash:
        raise SyntheticMultiseedV2AcquisitionError(
            f"{schema} ordered commitment set drifted"
        )
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
        field="v2 private full compiled cohort pack",
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
        or not _is_lower_hex(declared, 64)
        or semantic_hash(body) != declared
        or file_hash != expected_file_hash
        or not isinstance(rows, list)
        or len(rows) != TOTAL_ITEM_COUNT
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "private full compiled cohort pack binding drifted"
        )
    row_hashes: list[str] = []
    item_hashes: list[str] = []
    for global_ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SyntheticMultiseedV2AcquisitionError(
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
            or not _is_lower_hex(row_hash, 64)
            or semantic_hash(row_body) != row_hash
            or not _is_lower_hex(item_hash, 64)
        ):
            raise SyntheticMultiseedV2AcquisitionError(
                "private full compiled cohort row binding drifted"
            )
        row_hashes.append(str(row_hash))
        item_hashes.append(str(item_hash))
    if (
        len(set(row_hashes)) != TOTAL_ITEM_COUNT
        or len(set(item_hashes)) != TOTAL_ITEM_COUNT
        or stable_hash(row_hashes) != expected_row_set_hash
        or stable_hash(item_hashes) != expected_item_set_hash
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "private compiled cohort ordered commitments drifted"
        )
    return pack


def load_committed_acquisition_receipt(
    project_root: Path, *, verify_private_packs: bool = True
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    freeze, _actual_head = verify_implementation_freeze(root)
    receipt = _load_committed_public_json(
        root, ACQUISITION_RECEIPT_RELATIVE_PATH, "v2 acquisition receipt"
    )
    body = dict(receipt)
    declared = body.pop("receipt_sha256", None)
    commitments = receipt.get("commitments")
    if (
        receipt.get("schema") != ACQUISITION_SCHEMA
        or receipt.get("status")
        != "formal_v2_multiseed_A_hold_cohort_acquired_private_labels_separated"
        or receipt.get("design_sha256") != DESIGN_SHA256
        or receipt.get("design_file_sha256") != DESIGN_FILE_SHA256
        or receipt.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or receipt.get("block") != BLOCK
        or receipt.get("seed_count") != SEED_COUNT
        or receipt.get("item_count_per_seed") != ITEMS_PER_SEED
        or receipt.get("total_item_count") != TOTAL_ITEM_COUNT
        or receipt.get("grammar_generate_block_call_count") != SEED_COUNT
        or receipt.get("new_original_and_v1_item_commitments_pairwise_disjoint")
        is not True
        or receipt.get("fixed_recipe_id") != "R1_DEFINITION_1SWAP"
        or receipt.get("arms") != ["RAW", "official_HippoRAG", "Agent_R1"]
        or not isinstance(commitments, Mapping)
        or set(commitments)
        != {
            "action_pack_file_sha256",
            "action_item_commitment_set_sha256",
            "label_pack_file_sha256",
            "label_item_commitment_set_sha256",
            "compiled_cohort_pack_file_sha256",
            "compiled_row_commitment_set_sha256",
        }
        or any(not _is_lower_hex(value, 64) for value in commitments.values())
        or not _is_lower_hex(declared, 64)
        or semantic_hash(body) != declared
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "v2 acquisition receipt drifted"
        )
    custody = _load_committed_public_json(
        root, SEED_CUSTODY_RELATIVE_PATH, "v2 seed custody"
    )
    if custody != load_seed_custody(root / SEED_CUSTODY_RELATIVE_PATH) or (
        receipt.get("custody_sha256") != custody.get("custody_sha256")
        or receipt.get("seed_batch_commitment_sha256")
        != custody.get("seed_batch_commitment_sha256")
        or receipt.get("ordered_seed_commitments_sha256")
        != custody.get("ordered_seed_commitments_sha256")
    ):
        raise SyntheticMultiseedV2AcquisitionError(
            "v2 acquisition custody chain drifted"
        )
    if verify_private_packs:
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


def _load_committed_terminal_result(
    root: Path,
    *,
    freeze: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    committed = _load_committed_public_json(root, RESULT_RELATIVE_PATH, "v2 terminal result")
    result, result_file_hash = _read_json_with_mode(
        root / RESULT_RELATIVE_PATH,
        expected_mode=PUBLIC_MODE,
        field="v2 terminal result",
    )
    if result != committed:
        raise SyntheticMultiseedV2AcquisitionError(
            "terminal result readback drifted"
        )
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
        raise SyntheticMultiseedV2AcquisitionError("terminal result binding drifted")
    marker, marker_file_hash = _read_json_with_mode(
        root / RUNNER_MARKER_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="v2 runner formal attempt marker",
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
        raise SyntheticMultiseedV2AcquisitionError(
            "v2 runner attempt marker chain drifted"
        )
    declared_seal_file_hash = result.get("action_seal_file_sha256")
    seal_path = root / RUNNER_ACTION_SEAL_RELATIVE_PATH
    if status == SUCCESS_RESULT_STATUS:
        required_hashes = (
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
            or result.get("official_concurrency_cap") != 8
            or result.get("local_concurrency_cap") != 64
            or not isinstance(result.get("official_peak_concurrency_count"), int)
            or not 1 <= result["official_peak_concurrency_count"] <= 8
            or not isinstance(result.get("local_peak_concurrency_count"), int)
            or not 1 <= result["local_peak_concurrency_count"] <= 64
            or result.get("chunk_schedule_sha256") != CHUNK_SCHEDULE_SHA256
            or result.get("observed_encoder_input_row_counts") != [8448, 8448]
            or result.get("observed_encoder_output_row_counts") != [8448, 8448]
            or not _is_lower_hex(
                result.get("official_postflight_receipt_sha256"), 64
            )
            or result.get("interpretation")
            != "descriptive_fixed_cohort_replication_only"
            or result.get("seeds_or_item_rows_disclosed") is not False
            or result.get("result_must_be_committed_before_terminal_publication")
            is not True
            or not isinstance(result.get("aggregates"), Mapping)
            or not isinstance(result.get("cluster_differences"), Mapping)
            or any(not _is_lower_hex(result.get(field), 64) for field in required_hashes)
        ):
            raise SyntheticMultiseedV2AcquisitionError(
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
            raise SyntheticMultiseedV2AcquisitionError(
                "failed terminal result schema drifted"
            )
    if declared_seal_file_hash is None:
        if status == SUCCESS_RESULT_STATUS or seal_path.exists() or seal_path.is_symlink():
            raise SyntheticMultiseedV2AcquisitionError(
                "v2 runner action seal chain drifted"
            )
    else:
        seal, seal_file_hash = _read_json_with_mode(
            seal_path,
            expected_mode=PRIVATE_MODE,
            field="v2 runner private action seal",
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
            or seal.get("submitted_action_work_unit_count")
            != TOTAL_ITEM_COUNT * 3
            or seal.get("terminal_action_work_unit_count")
            != TOTAL_ITEM_COUNT * 3
            or seal.get("official_retrieve_action_count") != TOTAL_ITEM_COUNT
            or seal.get("RAW_action_count") != TOTAL_ITEM_COUNT
            or seal.get("Agent_R1_action_count") != TOTAL_ITEM_COUNT
            or seal.get("official_concurrency_cap") != 8
            or seal.get("local_concurrency_cap") != 64
            or seal.get("chunk_schedule_sha256") != CHUNK_SCHEDULE_SHA256
            or seal.get("observed_encoder_input_row_counts") != [8448, 8448]
            or seal.get("observed_encoder_output_row_counts") != [8448, 8448]
            or status == SUCCESS_RESULT_STATUS
            and (
                seal.get("official_peak_concurrency_count")
                != result.get("official_peak_concurrency_count")
                or seal.get("local_peak_concurrency_count")
                != result.get("local_peak_concurrency_count")
                or seal.get("runtime_binding_sha256")
                != result.get("runtime_binding_sha256")
                or seal.get("official_postflight_receipt_sha256")
                != result.get("official_postflight_receipt_sha256")
                or seal.get("action_table_sha256")
                != result.get("action_table_sha256")
            )
            or seal.get("labels_opened_before_action_seal") is not False
            or seal.get("labels_opened_before_seal") is not False
            or not _is_lower_hex(seal_hash, 64)
            or semantic_hash(seal_body) != seal_hash
            or seal_file_hash != declared_seal_file_hash
            or status == SUCCESS_RESULT_STATUS
            and result.get("action_seal_sha256") != seal_hash
        ):
            raise SyntheticMultiseedV2AcquisitionError(
                "v2 runner action seal chain drifted"
            )
    return result, result_file_hash


def publish_terminal(*, project_root: Path) -> dict[str, Any]:
    """Publish exact fresh seeds and 512 stored rows after a committed terminal."""

    _require_formal_entry()
    root = project_root.resolve(strict=True)
    freeze, actual_head = verify_implementation_freeze(root)
    # Validate only committed public metadata first.  No private action, label,
    # seed, or compiled content is opened until the committed terminal result
    # has passed its canonical chain validation below.
    acquisition = load_committed_acquisition_receipt(
        root, verify_private_packs=False
    )
    result, result_file_hash = _load_committed_terminal_result(
        root, freeze=freeze, acquisition=acquisition
    )
    result_hash = str(result["receipt_sha256"])
    outputs = (
        root / PUBLICATION_MARKER_RELATIVE_PATH,
        root / PUBLICATION_RELATIVE_PATH,
        root / PUBLICATION_FAILURE_RELATIVE_PATH,
    )
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise SyntheticMultiseedV2AcquisitionError(
            "v2 publication attempt already exists"
        )
    custody = _load_committed_public_json(
        root, SEED_CUSTODY_RELATIVE_PATH, "v2 seed custody"
    )
    marker = _self_hashed(
        {
            "schema": f"{PUBLICATION_SCHEMA}_attempt_marker",
            "version": VERSION,
            "status": "sole_v2_terminal_reproducibility_publication_attempt_consumed",
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
            raise SyntheticMultiseedV2AcquisitionError(
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
                "status": "terminal_v2_eight_seeds_and_full_compiled_cohort_published",
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
                "cohort_regenerated_during_publication": False,
                "retrieval_actions_model_outputs_or_scores_included": False,
            },
            "reproducibility_sha256",
        )
        _write_json_exclusive(root / PUBLICATION_RELATIVE_PATH, artifact, PUBLIC_MODE)
        return artifact
    except Exception as exc:
        failure = _self_hashed(
            {
                "schema": f"{PUBLICATION_SCHEMA}_failure_receipt",
                "version": VERSION,
                "status": "terminal_v2_publication_invalid_no_replay",
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
            raise SyntheticMultiseedV2AcquisitionError(
                "formal entry is already active"
            )
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
    "CHUNK_SCHEDULE_SHA256",
    "COMPILED_COHORT_PACK_RELATIVE_PATH",
    "COMPILED_COHORT_PACK_SCHEMA",
    "DESIGN_FILE_SHA256",
    "DESIGN_SHA256",
    "IMPLEMENTATION_FREEZE_RELATIVE_PATH",
    "INTEGRATION_DIAGNOSTIC_RELATIVE_PATH",
    "INTEGRATION_DIAGNOSTIC_SCHEMA",
    "ITEMS_PER_SEED",
    "LABEL_ITEM_SCHEMA",
    "LABEL_PACK_RELATIVE_PATH",
    "LABEL_PACK_SCHEMA",
    "PRIVATE_MODE",
    "PUBLIC_MODE",
    "RESULT_RELATIVE_PATH",
    "RESULT_SCHEMA",
    "SEED_BATCH_BYTES",
    "SEED_BYTES",
    "SEED_COUNT",
    "SyntheticMultiseedAcquisitionError",
    "SyntheticMultiseedV2AcquisitionError",
    "TOTAL_ITEM_COUNT",
    "VERSION",
    "acquire_formal_cohort",
    "canonical_bytes",
    "create_implementation_freeze",
    "create_seed_custody",
    "load_committed_acquisition_receipt",
    "load_committed_integration_diagnostic",
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
