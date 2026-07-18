"""Runtime-identity normalization for the untouched v3 multiseed cohort.

V5 is a thin lifecycle wrapper around the exact v2 action/scoring kernel and
the already acquired, still-unscored v3 cohort.  It preserves the lexical
``venv/bin/python`` symlink and uses the prospectively frozen v3 official
runtime attestation, whose only identity normalization excludes validated
Hugging Face download timestamp lines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from replication_runtime.qasper_minilm_v1 import OfflineMiniLMEncoder
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding

from . import synthetic_typed_graph_multiseed_lifecycle_v3 as v3
from .musique_formal_runtime_binding_v3 import (
    PreparedFormalRuntimeV3,
    prepare_formal_runtime_v3,
    validate_formal_runtime_binding_v3,
)


VERSION = "synthetic_typed_graph_multiseed_runtime_normalization_v5"
RESULT_VERSION = "synthetic_typed_graph_multiseed_replication_v5"
SOURCE_COHORT_VERSION = v3.VERSION
DESIGN_SCHEMA = "synthetic_typed_graph_multiseed_runtime_normalization_design_v5"
DESIGN_SHA256 = "f32d5407c3ab094ac6545c692e1b7080931be169e38c309fbd382846e396ca98"
DESIGN_FILE_SHA256 = "e5852c614398c86261c814eee1dbe73ef03390ad1b9bafbcb8bb4f961643d1ac"

PRIVATE_MODE = 0o600
PUBLIC_MODE = 0o644

DESIGN_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_runtime_normalization_design_v5.json"
)
MODULE_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/"
    "synthetic_typed_graph_multiseed_runtime_normalization_v5.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/test_synthetic_typed_graph_multiseed_runtime_normalization_v5.py"
)
IMPLEMENTATION_FREEZE_RELATIVE_PATH = Path(
    "manifests/"
    "synthetic_typed_graph_multiseed_runtime_normalization_implementation_freeze_v5.json"
)
RESULT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_result_v5.json"
)
PUBLICATION_FAILURE_RELATIVE_PATH = Path(
    "manifests/"
    "synthetic_typed_graph_multiseed_replication_publication_failure_v5.json"
)
ARTIFACT_ROOT_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_multiseed_replication_v5"
)
FORMAL_LAUNCH_MARKER_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "runner/formal.launch.marker"
)
FORMAL_WORK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "runner/formal.work"
FORMAL_ACTION_SEAL_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "runner/action_seal.json"
)
PUBLICATION_MARKER_RELATIVE_PATH = (
    ARTIFACT_ROOT_RELATIVE_PATH / "publication.attempt.marker"
)
PUBLICATION_RELATIVE_PATH = Path(
    "published/synthetic_typed_graph_multiseed_replication_v5/reproducibility.json"
)

V4_CLOSURE_RECEIPT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_execution_repair_prefreeze_terminal_v4.json"
)
V4_CLOSURE_COMMIT = "8842a327f2519ed619427f2a6027c7ab66b35006"
V4_CLOSURE_RECEIPT_SHA256 = (
    "cbfd42ad6e8f8fa0b4251689d8375260e9b4388338df0c74eea597d936d70482"
)
V4_CLOSURE_RECEIPT_FILE_SHA256 = (
    "f80a2c631af77d9a243e0315f093daa1505cf4b6ffa46c1b65b26b6e4b8e1e43"
)

V3_CLOSURE_RECEIPT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_prelaunch_terminal_v3.json"
)
V3_CLOSURE_MARKER_RELATIVE_PATH = (
    v3.ARTIFACT_ROOT_RELATIVE_PATH / "prelaunch_terminal.attempt.marker"
)
V3_CLOSURE_COMMIT = "5b43dd741a79bb40e9af63b6c5df8c3c0b0dbc07"
V3_CLOSURE_RECEIPT_SHA256 = (
    "99d0e997dcf58f4ea9fcbaa4476bbd00196ff7e9c27c93e9944130a9800e1cc8"
)
V3_CLOSURE_RECEIPT_FILE_SHA256 = (
    "0103cc7d7bdca420d91fe90a15fa43c61e330cb56abe1cf0e09d9068d4ea8e24"
)
V3_CLOSURE_MARKER_SHA256 = (
    "122c20216f96977eb5b69422380af18a3265fd924bd41fdc846cdbbf59fc2b18"
)
V3_CLOSURE_MARKER_FILE_SHA256 = (
    "a04876dfbe7e751885d803ac682a9ef31192c17e657b3c66c6f4952c8167deff"
)

SOURCE_COHORT_COMMITMENTS = {
    "seed_batch_commitment_sha256": (
        "e0d47ab2a6266d4b45e2d72901011a7d93df7cd1f8d79642b543538f3757889c"
    ),
    "generated_item_commitment_set_sha256": (
        "22cdb51798a2acedd0b894172e1e8cddf16b5964dcc4d7532ce7ad66112fcc2f"
    ),
    "action_pack_file_sha256": (
        "56feaceab1237cd880767893818955689b7905a5e3db0ae403054fd059080f55"
    ),
    "action_item_commitment_set_sha256": (
        "dd22279bea305ced4499bbd1f7131cc6eee170ebcf75df4442a75c97c6f3d95a"
    ),
    "label_pack_file_sha256": (
        "caf548c9c487d6d9d0eafa84f441c3f0b9af443cff8142edb06f3f81b7d9ff78"
    ),
    "label_item_commitment_set_sha256": (
        "10af505c6567bd8c3c837a6602df89a200f671dfe812c0736984f5b1b9113310"
    ),
    "compiled_cohort_pack_file_sha256": (
        "e09d1ac5a928cd08f5227e85333afaa56cf6f00772ee6ba11fead5232394b561"
    ),
    "compiled_row_commitment_set_sha256": (
        "cf08e7e31b1ea1ba24f39cdfed046f912b621c51a28fadd6104ff86c89675b66"
    ),
}
SOURCE_ACQUISITION_RECEIPT_SHA256 = (
    "b70f74b4487f8f7a16aea7d5e31569b1b75203e2338ccc51e51e37f30e25035e"
)
SOURCE_ACQUISITION_RECEIPT_FILE_SHA256 = (
    "5c76f39742aad92fc6ef44d06a03ac4d2062f3ca0750c7e5ac2cf9733bc19043"
)
OFFICIAL_ATTESTATION_V3_RELATIVE_PATH = Path(
    "manifests/musique_official_hipporag_runtime_attestation_v3.json"
)
ALLOWED_SEMANTIC_CHANGES = (
    "preserve_lexical_runtime_python_symlink_path",
    "exclude_only_validated_huggingface_download_timestamp_line_from_runtime_identity",
    "substitute_adapter_v3_verifier_for_adapter_v2_verifier_with_mechanical_equivalence_proof",
    "correct_failure_receipt_pack_label_state_and_systemd_invocation_provenance",
)

FORMAL_SYSTEMD_UNIT = "assumption-synth-multiseed-v5-formal"
SYSTEMD_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TMPDIR": "/tmp",
    "TRANSFORMERS_OFFLINE": "1",
}
SYSTEMD_PROPERTIES = {
    "KillMode": "control-group",
    "RemainAfterExit": True,
    "Restart": "no",
    "ServiceType": "exec",
    "StandardError": "journal",
    "StandardOutput": "journal",
    "TimeoutStopSec": "60s",
    "UMask": "0077",
}

FREEZE_SCHEMA = "synthetic_typed_graph_multiseed_runtime_normalization_implementation_freeze_v5"
FREEZE_STATUS = "complete_v5_runtime_normalization_frozen_must_commit_before_formal"
RESULT_SCHEMA = "synthetic_typed_graph_multiseed_replication_result_v5"
SUCCESS_RESULT_STATUS = "terminal_descriptive_eight_seed_replication_complete"
FAILURE_RESULT_STATUS = "terminal_infrastructure_or_implementation_invalid_no_replay"
PUBLICATION_SCHEMA = "synthetic_typed_graph_multiseed_terminal_reproducibility_v5"

V4_REQUIRED_ABSENT_PATHS = (
    Path(
        "manifests/"
        "synthetic_typed_graph_multiseed_execution_repair_implementation_freeze_v4.json"
    ),
    Path(
        "artifacts/synthetic_typed_graph_multiseed_replication_v4/"
        "runner/formal.launch.marker"
    ),
    Path(
        "artifacts/synthetic_typed_graph_multiseed_replication_v4/runner/formal.work"
    ),
    Path(
        "artifacts/synthetic_typed_graph_multiseed_replication_v4/"
        "runner/action_seal.json"
    ),
    Path("manifests/synthetic_typed_graph_multiseed_replication_result_v4.json"),
    Path(
        "artifacts/synthetic_typed_graph_multiseed_replication_v4/"
        "publication.attempt.marker"
    ),
    Path(
        "published/synthetic_typed_graph_multiseed_replication_v4/"
        "reproducibility.json"
    ),
)
V3_REQUIRED_ABSENT_PATHS = (
    v3.FORMAL_LAUNCH_MARKER_RELATIVE_PATH,
    v3.FORMAL_WORK_RELATIVE_PATH,
    v3.FORMAL_ACTION_SEAL_RELATIVE_PATH,
    v3.RESULT_RELATIVE_PATH,
    v3.PUBLICATION_MARKER_RELATIVE_PATH,
    v3.PUBLICATION_RELATIVE_PATH,
    v3.PUBLICATION_FAILURE_RELATIVE_PATH,
)

V5_OUTPUT_PATHS = (
    FORMAL_LAUNCH_MARKER_RELATIVE_PATH,
    FORMAL_WORK_RELATIVE_PATH,
    FORMAL_ACTION_SEAL_RELATIVE_PATH,
    RESULT_RELATIVE_PATH,
    PUBLICATION_MARKER_RELATIVE_PATH,
    PUBLICATION_RELATIVE_PATH,
    PUBLICATION_FAILURE_RELATIVE_PATH,
)

REQUIRED_FREEZE_PATHS = frozenset(
    {
        *v3.REQUIRED_FREEZE_PATHS,
        DESIGN_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        V4_CLOSURE_RECEIPT_RELATIVE_PATH.as_posix(),
        V3_CLOSURE_RECEIPT_RELATIVE_PATH.as_posix(),
        v3.IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix(),
        v3.SEED_CUSTODY_RELATIVE_PATH.as_posix(),
        v3.ACQUISITION_RECEIPT_RELATIVE_PATH.as_posix(),
        "manifests/musique_official_hipporag_runtime_attestation_v3.json",
        "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v3.py",
        "replication_runtime/musique_official_hipporag_v1/adapter_v3.py",
        "assumption_agent/benchmarks/musique_formal_runtime_binding_v3.py",
        "tests/test_musique_runtime_attestation_v3.py",
    }
)

PRIVATE_PACK_BINDINGS = (
    (
        v3.ACTION_PACK_RELATIVE_PATH,
        "action_pack_file_sha256",
    ),
    (
        v3.LABEL_PACK_RELATIVE_PATH,
        "label_pack_file_sha256",
    ),
    (
        v3.COMPILED_COHORT_PACK_RELATIVE_PATH,
        "compiled_cohort_pack_file_sha256",
    ),
)

# Exact v2 kernel aliases; v5 never copies or mutates their implementation.
kernel_v2 = v3.kernel_v2
acquisition_v2 = v3.acquisition_v2
canonical_bytes = v3.canonical_bytes
semantic_hash = v3.semantic_hash
stable_hash = v3.stable_hash
sha256_file = v3.sha256_file
_write_json_exclusive = v3._write_json_exclusive
_read_seed_batch = v3._read_seed_batch
_is_lower_hex = v3._is_lower_hex


class SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(RuntimeError):
    """A v5 normalization, freeze, detached-run, or publication invariant failed."""


RunCallable = Callable[..., subprocess.CompletedProcess[str]]


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            f"self-hash field already exists: {field}"
        )
    return {**dict(body), field: semantic_hash(dict(body))}


def _validate_self_hash(
    payload: Mapping[str, Any], field: str, *, label: str
) -> str:
    body = dict(payload)
    declared = body.pop(field, None)
    if not _is_lower_hex(declared, 64) or semantic_hash(body) != declared:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            f"{label} self-hash drifted"
        )
    return str(declared)


def _require_canonical_root(project_root: Path) -> Path:
    try:
        return v3._require_canonical_root(project_root)
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "canonical project root is unavailable"
        ) from exc


def _committed_head(project_root: Path) -> str:
    value = v3._committed_head(project_root)
    if not _is_lower_hex(value, 40):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error("Git HEAD drifted")
    return value


def _assert_absent(root: Path, relatives: Sequence[Path], field: str) -> None:
    occupied = [
        relative.as_posix()
        for relative in relatives
        if (root / relative).exists() or (root / relative).is_symlink()
    ]
    if occupied:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            f"{field} unexpectedly exists: {occupied[0]}"
        )


def _read_public_json(path: Path, *, field: str) -> tuple[dict[str, Any], str]:
    try:
        payload, file_sha256 = v3._read_json_with_mode(
            path, expected_mode=PUBLIC_MODE, field=field
        )
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            f"{field} is unavailable"
        ) from exc
    return payload, file_sha256


def _read_private_canonical_json(
    path: Path, *, field: str
) -> tuple[dict[str, Any], str]:
    try:
        return v3._read_canonical_json(
            path, expected_mode=PRIVATE_MODE, field=field
        )
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            f"{field} is unavailable"
        ) from exc


def verify_frozen_design(project_root: Path) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    path = root / DESIGN_RELATIVE_PATH
    if (
        not path.is_file()
        or path.is_symlink()
        or stat.S_IMODE(path.stat().st_mode) != PUBLIC_MODE
        or sha256_file(path) != DESIGN_FILE_SHA256
        or v3._committed_bytes(root, DESIGN_RELATIVE_PATH) != path.read_bytes()
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "frozen v5 design drifted"
        )
    design, _ = _read_public_json(path, field="frozen v5 design")
    if (
        design.get("schema") != DESIGN_SCHEMA
        or design.get("version") != VERSION
        or _validate_self_hash(design, "design_sha256", label="v5 design")
        != DESIGN_SHA256
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "frozen v5 design binding drifted"
        )
    paths = design.get("path_contract")
    systemd = design.get("systemd_contract")
    repair = design.get("repair_scope")
    reuse = design.get("cohort_reuse_contract")
    if (
        not isinstance(paths, Mapping)
        or paths.get("formal_launch_marker")
        != FORMAL_LAUNCH_MARKER_RELATIVE_PATH.as_posix()
        or paths.get("formal_work_root") != FORMAL_WORK_RELATIVE_PATH.as_posix()
        or paths.get("formal_action_seal")
        != FORMAL_ACTION_SEAL_RELATIVE_PATH.as_posix()
        or paths.get("formal_result") != RESULT_RELATIVE_PATH.as_posix()
        or paths.get("implementation_freeze")
        != IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix()
        or paths.get("publication") != PUBLICATION_RELATIVE_PATH.as_posix()
        or paths.get("publication_attempt_marker")
        != PUBLICATION_MARKER_RELATIVE_PATH.as_posix()
        or paths.get("publication_failure")
        != PUBLICATION_FAILURE_RELATIVE_PATH.as_posix()
        or paths.get("attestation_v3")
        != OFFICIAL_ATTESTATION_V3_RELATIVE_PATH.as_posix()
        or not isinstance(systemd, Mapping)
        or systemd.get("formal_unit") != FORMAL_SYSTEMD_UNIT
        or systemd.get("environment") != SYSTEMD_ENVIRONMENT
        or systemd.get("properties") != SYSTEMD_PROPERTIES
        or systemd.get("child_module")
        != "assumption_agent.benchmarks.synthetic_typed_graph_multiseed_runtime_normalization_v5"
        or not isinstance(repair, Mapping)
        or repair.get("allowed_semantic_changes")
        != list(ALLOWED_SEMANTIC_CHANGES)
        or repair.get("frozen_v2_or_v3_file_mutation_authorized") is not False
        or repair.get("new_feasibility_or_performance_gate_authorized") is not False
        or not isinstance(reuse, Mapping)
        or reuse.get("source_cohort_version") != SOURCE_COHORT_VERSION
        or any(reuse.get(key) != value for key, value in SOURCE_COHORT_COMMITMENTS.items())
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v5 runtime-normalization contract drifted"
        )
    return design


def _verify_v4_closure_commit(root: Path, receipt_bytes: bytes) -> None:
    try:
        v3._git(root, "merge-base", "--is-ancestor", V4_CLOSURE_COMMIT, "HEAD")
        prefix = v3.acquisition_v2._git_project_prefix(root)
        committed = v3._git(
            root,
            "show",
            f"{V4_CLOSURE_COMMIT}:{prefix}{V4_CLOSURE_RECEIPT_RELATIVE_PATH.as_posix()}",
        )
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v4 closure commit is not an ancestor with the bound receipt"
        ) from exc
    if committed != receipt_bytes:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v4 closure bytes differ from the closure commit"
        )


def verify_v4_prefreeze_closure(project_root: Path) -> dict[str, Any]:
    """Verify the committed v4 closure without opening any cohort pack."""

    root = _require_canonical_root(project_root)
    _assert_absent(root, V4_REQUIRED_ABSENT_PATHS, "closed v4 output")
    _assert_absent(root, V3_REQUIRED_ABSENT_PATHS, "closed v3 formal output")
    receipt_path = root / V4_CLOSURE_RECEIPT_RELATIVE_PATH
    try:
        receipt = v3._load_committed_public_json(
            root, V4_CLOSURE_RECEIPT_RELATIVE_PATH, "committed v4 closure"
        )
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "committed v4 closure is unavailable"
        ) from exc
    receipt_bytes = receipt_path.read_bytes()
    if sha256_file(receipt_path) != V4_CLOSURE_RECEIPT_FILE_SHA256:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v4 closure receipt bytes drifted"
        )
    _verify_v4_closure_commit(root, receipt_bytes)
    if (
        receipt.get("schema")
        != "synthetic_typed_graph_multiseed_execution_repair_prefreeze_terminal_v4"
        or receipt.get("version")
        != "synthetic_typed_graph_multiseed_execution_repair_v4"
        or receipt.get("status")
        != "terminal_prefreeze_infrastructure_invalid_formal_attempt_unconsumed_no_scores_or_claims"
        or _validate_self_hash(receipt, "receipt_sha256", label="v4 closure")
        != V4_CLOSURE_RECEIPT_SHA256
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v4 closure receipt drifted"
        )
    closure = receipt.get("closure")
    accounting = receipt.get("execution_accounting")
    preservation = receipt.get("cohort_preservation")
    predecessors = receipt.get("predecessor_bindings")
    if (
        not isinstance(closure, Mapping)
        or closure.get("v4_formal_attempt_unconsumed") is not True
        or closure.get("v4_formal_authorization_revoked") is not True
        or closure.get("retry_or_v4_launch_authorized") is not False
        or any(
            closure.get(key) is not False
            for key in (
                "implementation_freeze_created",
                "formal_launch_marker_created",
                "formal_work_root_created",
                "formal_action_seal_created",
                "canonical_v4_formal_result_created",
                "publication_attempt_marker_created",
                "published_cohort_created",
            )
        )
        or not isinstance(accounting, Mapping)
        or any(
            accounting.get(key) != 0
            for key in (
                "actions_submitted",
                "agent_actions_completed",
                "formal_action_pack_open_calls",
                "formal_label_pack_open_calls",
                "hipporag_actions_completed",
                "minilm_scoring_calls",
                "raw_actions_completed",
                "scores_computed",
                "online_evaluator_calls",
            )
        )
        or not isinstance(preservation, Mapping)
        or any(
            preservation.get(key) != value
            for key, value in SOURCE_COHORT_COMMITMENTS.items()
        )
        or preservation.get("source_cohort_version") != SOURCE_COHORT_VERSION
        or not isinstance(predecessors, Mapping)
        or predecessors.get("v3_closure_commit") != V3_CLOSURE_COMMIT
        or predecessors.get("v3_closure_receipt_sha256")
        != V3_CLOSURE_RECEIPT_SHA256
        or predecessors.get("v3_closure_receipt_file_sha256")
        != V3_CLOSURE_RECEIPT_FILE_SHA256
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v4 closure invariants drifted"
        )
    try:
        acquisition = v3.load_committed_acquisition_receipt(
            root, verify_private_packs=False
        )
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "untouched v3 acquisition chain is unavailable"
        ) from exc
    commitments = acquisition.get("commitments")
    if (
        acquisition.get("receipt_sha256") != SOURCE_ACQUISITION_RECEIPT_SHA256
        or sha256_file(root / v3.ACQUISITION_RECEIPT_RELATIVE_PATH)
        != SOURCE_ACQUISITION_RECEIPT_FILE_SHA256
        or acquisition.get("generated_item_commitment_set_sha256")
        != SOURCE_COHORT_COMMITMENTS["generated_item_commitment_set_sha256"]
        or not isinstance(commitments, Mapping)
        or any(
            commitments.get(key) != value
            for key, value in SOURCE_COHORT_COMMITMENTS.items()
            if key not in {
                "seed_batch_commitment_sha256",
                "generated_item_commitment_set_sha256",
            }
        )
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v3 cohort commitments drifted after closure"
        )
    return {
        "closure_receipt_sha256": receipt["receipt_sha256"],
        "closure_receipt_file_sha256": V4_CLOSURE_RECEIPT_FILE_SHA256,
        "closure_commit": V4_CLOSURE_COMMIT,
        "acquisition_receipt_sha256": acquisition["receipt_sha256"],
        "acquisition_receipt_file_sha256": sha256_file(
            root / v3.ACQUISITION_RECEIPT_RELATIVE_PATH
        ),
        "generated_item_commitment_set_sha256": acquisition[
            "generated_item_commitment_set_sha256"
        ],
        "v4_required_outputs_absent": True,
        "private_cohort_packs_semantically_opened": False,
    }


def _lexical_runtime_python(path: Path) -> Path:
    """Return an absolute lexical venv launcher without resolving its symlink."""

    lexical = path.expanduser().absolute()
    if (
        lexical.name != "python"
        or lexical.parent.name != "bin"
        or not lexical.is_symlink()
        or not lexical.is_file()
        or not os.access(lexical, os.X_OK)
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "runtime Python must be an executable lexical venv/bin/python symlink"
        )
    return lexical


def _systemd_child_argv(
    root: Path,
    *,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> list[str]:
    lexical = _lexical_runtime_python(runtime_python)
    llm = local_llm_model.expanduser().absolute()
    embedding = local_embedding_model.expanduser().resolve(strict=True)
    return [
        str(Path(sys.executable).resolve(strict=True)),
        "-u",
        "-m",
        "assumption_agent.benchmarks."
        "synthetic_typed_graph_multiseed_runtime_normalization_v5",
        "formal-child",
        "--project-root",
        str(root),
        "--runtime-python",
        str(lexical),
        "--local-llm-model",
        str(llm),
        "--local-embedding-model",
        str(embedding),
    ]


def _systemd_run_argv(root: Path, child_argv: Sequence[str]) -> list[str]:
    return [
        "systemd-run",
        "--user",
        f"--unit={FORMAL_SYSTEMD_UNIT}",
        "--service-type=exec",
        "--remain-after-exit",
        f"--working-directory={root}",
        "--property=StandardOutput=journal",
        "--property=StandardError=journal",
        "--property=KillMode=control-group",
        "--property=Restart=no",
        "--property=UMask=0077",
        "--property=TimeoutStopSec=60s",
        "--setenv=TMPDIR=/tmp",
        "--setenv=HF_HUB_OFFLINE=1",
        "--setenv=TRANSFORMERS_OFFLINE=1",
        *child_argv,
    ]


def _systemd_contract_sha256(root: Path) -> str:
    python = str(Path(sys.executable).resolve(strict=True))
    template_child = [
        python,
        "-u",
        "-m",
        "assumption_agent.benchmarks."
        "synthetic_typed_graph_multiseed_runtime_normalization_v5",
        "formal-child",
        "--project-root",
        str(root),
        "--runtime-python",
        "BOUND_LEXICAL_RUNTIME_PYTHON",
        "--local-llm-model",
        "BOUND_LOCAL_LLM_MODEL",
        "--local-embedding-model",
        "BOUND_LOCAL_EMBEDDING_MODEL",
    ]
    return semantic_hash(
        {
            "environment": SYSTEMD_ENVIRONMENT,
            "properties": SYSTEMD_PROPERTIES,
            "formal_unit": FORMAL_SYSTEMD_UNIT,
            "formal_argv_template": _systemd_run_argv(root, template_child),
        }
    )


def _call_run(
    run: RunCallable, argv: Sequence[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    try:
        completed = run(
            list(argv),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "detached systemd command could not be executed"
        ) from exc
    if not hasattr(completed, "returncode"):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "run callable contract drifted"
        )
    return completed


def _offline_environment_exact() -> bool:
    return all(os.environ.get(key) == value for key, value in SYSTEMD_ENVIRONMENT.items())


def _current_binding_rows(
    project_root: Path, paths: Sequence[str]
) -> list[dict[str, str]]:
    try:
        return v3._current_binding_rows(project_root, paths)
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v5 required committed binding drifted"
        ) from exc


def _project_minilm_binding(project_root: Path) -> dict[str, Any]:
    try:
        receipt = minilm_binding.verify_runtime_asset(project_root)
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "project-local MiniLM binding is unavailable"
        ) from exc
    safe = {
        key: value
        for key, value in receipt.items()
        if key not in {"asset_manifest_path", "model_root"}
    }
    safe["runtime_asset_paths_persisted"] = False
    return _self_hashed(safe, "binding_sha256")


def _private_pack_binding_rows(project_root: Path) -> list[dict[str, Any]]:
    root = _require_canonical_root(project_root)
    rows: list[dict[str, Any]] = []
    for relative, commitment_key in PRIVATE_PACK_BINDINGS:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) != PRIVATE_MODE
        ):
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                f"private source pack is unavailable: {relative}"
            )
        digest = sha256_file(path)
        if digest != SOURCE_COHORT_COMMITMENTS[commitment_key]:
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                f"private source pack bytes drifted: {relative}"
            )
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "file_sha256": digest,
                "size_bytes": path.stat().st_size,
                "mode": "0600",
                "semantic_payload_opened": False,
            }
        )
    return rows


def create_implementation_freeze(project_root: Path) -> dict[str, Any]:
    """Freeze the four preregistered repairs plus the exact predecessor chain."""

    root = _require_canonical_root(project_root)
    verify_frozen_design(root)
    closure = verify_v4_prefreeze_closure(root)
    output = root / IMPLEMENTATION_FREEZE_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v5 implementation freeze already exists"
        )
    _assert_absent(root, V5_OUTPUT_PATHS, "pre-freeze v5 output")
    binding_paths = tuple(sorted(REQUIRED_FREEZE_PATHS))
    bindings = _current_binding_rows(root, binding_paths)
    private_pack_bindings = _private_pack_binding_rows(root)
    minilm = _project_minilm_binding(root)
    body = {
        "schema": FREEZE_SCHEMA,
        "version": VERSION,
        "status": FREEZE_STATUS,
        "creation_HEAD": _committed_head(root),
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "predecessor_closure": closure,
        "source_cohort_version": SOURCE_COHORT_VERSION,
        "source_cohort_commitments": dict(SOURCE_COHORT_COMMITMENTS),
        "private_pack_bindings": private_pack_bindings,
        "private_pack_semantics_opened_during_freeze": False,
        "project_minilm_binding": minilm,
        "systemd_contract_sha256": _systemd_contract_sha256(root),
        "allowed_semantic_changes": list(ALLOWED_SEMANTIC_CHANGES),
        "frozen_v2_or_v3_file_mutated": False,
        "new_seed_cohort_smoke_gate_or_candidate_search_performed": False,
        "bindings": bindings,
    }
    freeze = _self_hashed(body, "implementation_freeze_sha256")
    _write_json_exclusive(output, freeze, PUBLIC_MODE)
    return freeze


def verify_implementation_freeze(
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    root = _require_canonical_root(project_root)
    verify_frozen_design(root)
    closure = verify_v4_prefreeze_closure(root)
    try:
        freeze = v3._load_committed_public_json(
            root,
            IMPLEMENTATION_FREEZE_RELATIVE_PATH,
            "committed v5 implementation freeze",
        )
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "committed v5 implementation freeze is unavailable"
        ) from exc
    declared = _validate_self_hash(
        freeze, "implementation_freeze_sha256", label="v5 implementation freeze"
    )
    rows = freeze.get("bindings")
    private_rows = freeze.get("private_pack_bindings")
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("version") != VERSION
        or freeze.get("status") != FREEZE_STATUS
        or freeze.get("design_sha256") != DESIGN_SHA256
        or freeze.get("design_file_sha256") != DESIGN_FILE_SHA256
        or freeze.get("predecessor_closure") != closure
        or freeze.get("source_cohort_version") != SOURCE_COHORT_VERSION
        or freeze.get("source_cohort_commitments") != SOURCE_COHORT_COMMITMENTS
        or freeze.get("private_pack_semantics_opened_during_freeze") is not False
        or freeze.get("project_minilm_binding") != _project_minilm_binding(root)
        or freeze.get("systemd_contract_sha256") != _systemd_contract_sha256(root)
        or freeze.get("allowed_semantic_changes") != list(ALLOWED_SEMANTIC_CHANGES)
        or freeze.get("frozen_v2_or_v3_file_mutated") is not False
        or freeze.get("new_seed_cohort_smoke_gate_or_candidate_search_performed")
        is not False
        or not _is_lower_hex(declared, 64)
        or not isinstance(rows, list)
        or not isinstance(private_rows, list)
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "committed v5 implementation freeze drifted"
        )
    expected = _current_binding_rows(root, tuple(sorted(REQUIRED_FREEZE_PATHS)))
    if rows != expected:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v5 code/test bindings drifted after freeze"
        )
    # Before formal consumption, do not reopen private cohort bytes.  Their
    # committed hashes were checked while creating the freeze and are checked
    # by the frozen action/late-label loaders after the marker.
    if [row.get("relative_path") for row in private_rows] != [
        relative.as_posix() for relative, _key in PRIVATE_PACK_BINDINGS
    ] or any(
        row.get("file_sha256") != SOURCE_COHORT_COMMITMENTS[key]
        or row.get("mode") != "0600"
        or row.get("semantic_payload_opened") is not False
        for row, (_relative, key) in zip(private_rows, PRIVATE_PACK_BINDINGS)
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v5 private pack freeze bindings drifted"
        )
    return freeze, _committed_head(root)


def _path_free_preflight_receipt(
    encoder: OfflineMiniLMEncoder, runtime: PreparedFormalRuntimeV3
) -> dict[str, Any]:
    if not isinstance(encoder, OfflineMiniLMEncoder) or not isinstance(
        runtime, PreparedFormalRuntimeV3
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "formal resources are not the attested frozen types"
        )
    expected_runtime_keys = {
        "asset_file_sha256",
        "asset_manifest_path",
        "asset_sha256",
        "embedding_dimension",
        "maximum_sequence_length",
        "model_root",
        "model_tree_sha256",
        "runtime_versions",
        "status",
        "weights_sha256",
    }
    if set(encoder.runtime_receipt) != expected_runtime_keys:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "MiniLM runtime receipt key set drifted"
        )
    minilm_runtime = {
        key: value
        for key, value in encoder.runtime_receipt.items()
        if key not in {"asset_manifest_path", "model_root"}
    }
    minilm_runtime["runtime_asset_paths_persisted"] = False
    minilm_runtime = _self_hashed(minilm_runtime, "binding_sha256")
    canary = dict(encoder.canary_receipt)
    official = validate_formal_runtime_binding_v3(runtime.safe_binding)
    official_runtime_binding_sha256 = kernel_v2._runtime_binding(runtime)
    body = {
        "schema": "synthetic_typed_graph_multiseed_v5_path_free_preflight",
        "minilm_runtime_binding": minilm_runtime,
        "minilm_runtime_binding_sha256": minilm_runtime["binding_sha256"],
        "minilm_canary_receipt": canary,
        "minilm_canary_receipt_sha256": semantic_hash(canary),
        "official_hipporag_runtime_binding": official,
        "official_safe_binding_self_sha256": official["binding_sha256"],
        "official_runtime_filesystem_binding_sha256": official[
            "runtime_filesystem_binding_sha256"
        ],
        "official_runtime_binding_sha256": official_runtime_binding_sha256,
        "action_label_or_compiled_pack_open_calls": 0,
        "performance_signal_or_gate_computed": False,
        "runtime_asset_paths_persisted": False,
    }
    raw = json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if "/home/" in raw or "/tmp/" in raw or "\\" in raw:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "preflight receipt contains a host path"
        )
    return _self_hashed(body, "preflight_sha256")


def _prepare_formal_resources(
    *,
    project_root: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> tuple[OfflineMiniLMEncoder, PreparedFormalRuntimeV3, dict[str, Any]]:
    lexical = _lexical_runtime_python(runtime_python)
    llm = local_llm_model.expanduser().absolute()
    embedding = local_embedding_model.expanduser().resolve(strict=True)
    root = _require_canonical_root(project_root)
    encoder = OfflineMiniLMEncoder(
        asset_manifest_path=root / kernel_v2.MINILM_MANIFEST_RELATIVE_PATH,
        model_root=root / kernel_v2.MINILM_MODEL_ROOT_RELATIVE_PATH,
        run_canary=True,
    )
    runtime = prepare_formal_runtime_v3(
        project_root=root,
        attestation_receipt_path=root / OFFICIAL_ATTESTATION_V3_RELATIVE_PATH,
        base_binding_receipt_path=root / kernel_v2.OFFICIAL_BASE_RECEIPT_RELATIVE_PATH,
        runtime_python=lexical,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )
    if (
        encoder.runtime_receipt.get("asset_manifest_path")
        != str(root / kernel_v2.MINILM_MANIFEST_RELATIVE_PATH)
        or encoder.runtime_receipt.get("model_root")
        != str(root / kernel_v2.MINILM_MODEL_ROOT_RELATIVE_PATH)
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "MiniLM preflight paths differ from the frozen project-local binding"
        )
    return encoder, runtime, _path_free_preflight_receipt(encoder, runtime)


def launch_formal(
    project_root: Path,
    *,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    run: RunCallable = subprocess.run,
) -> dict[str, Any]:
    """Preflight installation bytes, then durably consume the sole v5 attempt."""

    root = _require_canonical_root(project_root)
    freeze, actual_head = verify_implementation_freeze(root)
    closure = verify_v4_prefreeze_closure(root)
    _assert_absent(root, V5_OUTPUT_PATHS, "v5 formal attempt output")
    lexical = _lexical_runtime_python(runtime_python)
    llm = local_llm_model.expanduser().absolute()
    embedding = local_embedding_model.expanduser().resolve(strict=True)

    # This installation-only preflight is deliberately before the marker and
    # has no path to any v3 cohort pack or performance metric.
    encoder, runtime, preflight = _prepare_formal_resources(
        project_root=root,
        runtime_python=lexical,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )
    del encoder, runtime
    _assert_absent(root, V5_OUTPUT_PATHS, "v5 output during formal preflight")
    _assert_absent(root, V3_REQUIRED_ABSENT_PATHS, "closed v3 output after preflight")

    child_argv = _systemd_child_argv(
        root,
        runtime_python=lexical,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )
    launcher_argv = _systemd_run_argv(root, child_argv)
    acquisition = v3.load_committed_acquisition_receipt(
        root, verify_private_packs=False
    )
    acquisition_file_sha256 = sha256_file(
        root / v3.ACQUISITION_RECEIPT_RELATIVE_PATH
    )
    marker = _self_hashed(
        {
            "schema": f"{RESULT_SCHEMA}_formal_launch_marker",
            "version": RESULT_VERSION,
            "status": "sole_detached_v5_formal_attempt_consumed",
            "actual_HEAD": actual_head,
            "design_sha256": DESIGN_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "v4_closure_receipt_sha256": closure["closure_receipt_sha256"],
            "source_cohort_version": SOURCE_COHORT_VERSION,
            "acquisition_receipt_sha256": acquisition["receipt_sha256"],
            "acquisition_receipt_file_sha256": acquisition_file_sha256,
            "generated_item_commitment_set_sha256": acquisition[
                "generated_item_commitment_set_sha256"
            ],
            "unit": FORMAL_SYSTEMD_UNIT,
            "systemd_contract_sha256": _systemd_contract_sha256(root),
            "launcher_argv_sha256": semantic_hash(launcher_argv),
            "runtime_python": str(lexical),
            "runtime_python_is_lexical_symlink": True,
            "local_llm_model": str(llm),
            "local_embedding_model": str(embedding),
            "path_free_preflight": preflight,
            "official_attestation_v3_receipt_sha256": preflight[
                "official_hipporag_runtime_binding"
            ]["attestation_receipt_sha256"],
            "official_attestation_v3_receipt_file_sha256": preflight[
                "official_hipporag_runtime_binding"
            ]["attestation_receipt_file_sha256"],
            "attempt_count": 1,
            "private_packs_opened_before_marker": False,
            "labels_opened_before_marker": False,
            "performance_signal_or_gate_computed_before_marker": False,
            "relaunch_authorized": False,
        },
        "marker_sha256",
    )
    marker_file_sha256 = _write_json_exclusive(
        root / FORMAL_LAUNCH_MARKER_RELATIVE_PATH, marker, PRIVATE_MODE
    )
    completed = _call_run(run, launcher_argv, cwd=root)
    if completed.returncode != 0:
        state, state_returncode = _read_systemd_state(root, run=run)
        safe_state = {
            key: value for key, value in state.items() if key != "ControlGroup"
        }
        safe_state["systemctl_returncode"] = state_returncode
        if _systemd_state_proves_child_never_started(
            state, returncode=state_returncode
        ):
            return _persist_formal_failure(
                project_root=root,
                marker=marker,
                marker_file_sha256=marker_file_sha256,
                freeze=freeze,
                acquisition=acquisition,
                acquisition_file_sha256=acquisition_file_sha256,
                failure_class="SystemdRunLaunchFailureProvenChildNeverStarted",
                administrative=True,
                systemd_invocation_id=None,
                pack_label_open_state="unopened",
                open_state_evidence=(
                    "systemd_launch_failure_before_child_start"
                ),
                systemd_state=safe_state,
            )
        try:
            terminal_state, invocation_id = _validated_terminal_systemd_state(
                state, returncode=state_returncode
            )
        except SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error as exc:
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "systemd-run failed but target unit is not proven unstarted or terminal; "
                "the consumed marker remains pending"
            ) from exc
        return _persist_formal_failure(
            project_root=root,
            marker=marker,
            marker_file_sha256=marker_file_sha256,
            freeze=freeze,
            acquisition=acquisition,
            acquisition_file_sha256=acquisition_file_sha256,
            failure_class="SystemdRunNonzeroWithProvenTerminalUnit",
            administrative=True,
            systemd_invocation_id=invocation_id,
            pack_label_open_state="unknown",
            open_state_evidence=(
                "launcher_nonzero_with_verified_terminal_unit"
            ),
            systemd_state={
                **terminal_state,
                "launcher_returncode": int(completed.returncode),
            },
        )
    return marker


def _load_formal_marker(project_root: Path) -> tuple[dict[str, Any], str]:
    root = _require_canonical_root(project_root)
    marker, file_sha256 = _read_private_canonical_json(
        root / FORMAL_LAUNCH_MARKER_RELATIVE_PATH,
        field="v5 formal launch marker",
    )
    _validate_self_hash(marker, "marker_sha256", label="v5 formal marker")
    preflight = marker.get("path_free_preflight")
    if (
        marker.get("schema") != f"{RESULT_SCHEMA}_formal_launch_marker"
        or marker.get("version") != RESULT_VERSION
        or marker.get("status") != "sole_detached_v5_formal_attempt_consumed"
        or marker.get("design_sha256") != DESIGN_SHA256
        or marker.get("source_cohort_version") != SOURCE_COHORT_VERSION
        or marker.get("unit") != FORMAL_SYSTEMD_UNIT
        or marker.get("systemd_contract_sha256") != _systemd_contract_sha256(root)
        or marker.get("runtime_python_is_lexical_symlink") is not True
        or marker.get("attempt_count") != 1
        or marker.get("private_packs_opened_before_marker") is not False
        or marker.get("labels_opened_before_marker") is not False
        or marker.get("performance_signal_or_gate_computed_before_marker") is not False
        or marker.get("relaunch_authorized") is not False
        or not isinstance(preflight, Mapping)
        or _validate_self_hash(preflight, "preflight_sha256", label="v5 preflight")
        != preflight.get("preflight_sha256")
        or marker.get("official_attestation_v3_receipt_sha256")
        != preflight.get("official_hipporag_runtime_binding", {}).get(
            "attestation_receipt_sha256"
        )
        or marker.get("official_attestation_v3_receipt_file_sha256")
        != preflight.get("official_hipporag_runtime_binding", {}).get(
            "attestation_receipt_file_sha256"
        )
        or any(
            not isinstance(marker.get(field), str) or not marker.get(field)
            for field in (
                "runtime_python",
                "local_llm_model",
                "local_embedding_model",
            )
        )
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v5 formal launch marker drifted"
        )
    return marker, file_sha256


def _formal_failure_body(
    *,
    marker: Mapping[str, Any],
    marker_file_sha256: str,
    freeze: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    acquisition_file_sha256: str,
    failure_class: str,
    administrative: bool,
    systemd_invocation_id: str | None,
    pack_label_open_state: str,
    open_state_evidence: str,
    systemd_state: Mapping[str, Any] | None,
    action_seal_file_sha256: str | None,
) -> dict[str, Any]:
    expected_open_state = {
        "systemd_launch_failure_before_child_start": "unopened",
        "formal_child_exception": "unknown",
        "administrative_finalizer_without_durable_child_evidence": "unknown",
        "launcher_nonzero_with_verified_terminal_unit": "unknown",
    }.get(open_state_evidence)
    if expected_open_state != pack_label_open_state:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "failure open-state evidence is inconsistent"
        )
    if systemd_invocation_id is not None and not _is_lower_hex(
        systemd_invocation_id, 32
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "failure invocation id is not verified lowercase hex"
        )
    body = {
        "schema": RESULT_SCHEMA,
        "version": RESULT_VERSION,
        "status": FAILURE_RESULT_STATUS,
        "invocation_HEAD": marker["actual_HEAD"],
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "v4_closure_receipt_sha256": marker["v4_closure_receipt_sha256"],
        "source_cohort_version": SOURCE_COHORT_VERSION,
        "acquisition_receipt_sha256": acquisition["receipt_sha256"],
        "acquisition_receipt_file_sha256": acquisition_file_sha256,
        "generated_item_commitment_set_sha256": acquisition[
            "generated_item_commitment_set_sha256"
        ],
        "formal_launch_marker_sha256": marker["marker_sha256"],
        "formal_launch_marker_file_sha256": marker_file_sha256,
        "systemd_unit": FORMAL_SYSTEMD_UNIT,
        "systemd_invocation_id": systemd_invocation_id,
        "systemd_invocation_id_source": (
            "formal_child_environment"
            if open_state_evidence == "formal_child_exception"
            else (
                "verified_target_unit_property"
                if systemd_invocation_id is not None
                else None
            )
        ),
        "systemd_contract_sha256": marker["systemd_contract_sha256"],
        "preflight_sha256": marker["path_free_preflight"]["preflight_sha256"],
        "official_attestation_v3_receipt_sha256": marker[
            "official_attestation_v3_receipt_sha256"
        ],
        "official_attestation_v3_receipt_file_sha256": marker[
            "official_attestation_v3_receipt_file_sha256"
        ],
        "action_seal_file_sha256": action_seal_file_sha256,
        "failure_class": failure_class,
        "administrative_finalization": administrative,
        "private_pack_open_state": pack_label_open_state,
        "label_pack_open_state": pack_label_open_state,
        "pack_label_open_state_evidence": open_state_evidence,
        "systemd_state": dict(systemd_state) if systemd_state is not None else None,
        "retry_replacement_or_relaunch_authorized": False,
        "exception_message_seed_item_or_label_content_persisted_publicly": False,
        "performance_gate_or_promotion_computed": False,
        "result_must_be_committed_before_terminal_publication": True,
    }
    return _self_hashed(body, "receipt_sha256")


def _persist_formal_failure(
    *,
    project_root: Path,
    marker: Mapping[str, Any],
    marker_file_sha256: str,
    freeze: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    acquisition_file_sha256: str,
    failure_class: str,
    administrative: bool,
    systemd_invocation_id: str | None,
    pack_label_open_state: str,
    open_state_evidence: str,
    systemd_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    seal_path = root / FORMAL_ACTION_SEAL_RELATIVE_PATH
    seal_file_sha256 = (
        sha256_file(seal_path)
        if seal_path.is_file() and not seal_path.is_symlink()
        else None
    )
    failure = _formal_failure_body(
        marker=marker,
        marker_file_sha256=marker_file_sha256,
        freeze=freeze,
        acquisition=acquisition,
        acquisition_file_sha256=acquisition_file_sha256,
        failure_class=failure_class,
        administrative=administrative,
        systemd_invocation_id=systemd_invocation_id,
        pack_label_open_state=pack_label_open_state,
        open_state_evidence=open_state_evidence,
        systemd_state=systemd_state,
        action_seal_file_sha256=seal_file_sha256,
    )
    path = root / RESULT_RELATIVE_PATH
    if not path.exists() and not path.is_symlink():
        _write_json_exclusive(path, failure, PUBLIC_MODE)
    return _load_terminal_result_local(root)


_V2_KERNEL_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "version",
        "status",
        "design_sha256",
        "design_file_sha256",
        "block",
        "recipe_id",
        "seed_count",
        "item_count_per_seed",
        "total_item_count",
        "arms",
        "action_work_unit_count",
        "official_retrieve_action_count",
        "official_concurrency_cap",
        "local_concurrency_cap",
        "official_peak_concurrency_count",
        "local_peak_concurrency_count",
        "chunk_schedule_sha256",
        "observed_encoder_input_row_counts",
        "observed_encoder_output_row_counts",
        "action_pack_file_sha256",
        "action_pack_sha256",
        "action_item_commitment_set_sha256",
        "label_pack_file_sha256",
        "label_pack_sha256",
        "label_item_commitment_set_sha256",
        "runtime_binding_sha256",
        "official_postflight_receipt_sha256",
        "action_table_sha256",
        "action_seal_sha256",
        "action_seal_file_sha256",
        "aggregates",
        "cluster_differences",
        "interpretation",
        "seeds_or_item_rows_disclosed",
        "receipt_sha256",
    }
)
_V2_ACTION_SEAL_KEYS = frozenset(
    {
        "schema",
        "version",
        "status",
        "purpose",
        "block",
        "recipe_id",
        "item_count",
        "action_work_unit_count",
        "submitted_action_work_unit_count",
        "terminal_action_work_unit_count",
        "official_retrieve_action_count",
        "official_call_count",
        "RAW_action_count",
        "Agent_R1_action_count",
        "official_concurrency_cap",
        "local_concurrency_cap",
        "official_peak_concurrency_count",
        "local_peak_concurrency_count",
        "chunk_schedule_sha256",
        "observed_encoder_input_row_counts",
        "observed_encoder_output_row_counts",
        "action_pack_file_sha256",
        "action_pack_sha256",
        "action_item_commitment_set_sha256",
        "runtime_binding_sha256",
        "official_postflight_receipt_sha256",
        "action_table_sha256",
        "action_rows",
        "labels_opened_before_action_seal",
        "labels_opened_before_seal",
        "scores_computed_before_action_seal",
        "action_seal_sha256",
    }
)
_V2_ACTION_ROW_KEYS = frozenset(
    {
        "global_ordinal",
        "action_item_sha256",
        "RAW_top5",
        "official_HippoRAG_top5",
        "Agent_R1_top5",
        "common_scan_sha256",
        "local_tensor_sha256",
    }
)


def _validate_v2_kernel_receipt(
    project_root: Path,
    payload: Mapping[str, Any],
    *,
    acquisition: Mapping[str, Any],
    marker: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact frozen v2 receipt and its durable private action seal."""

    root = _require_canonical_root(project_root)
    if set(payload) != _V2_KERNEL_RECEIPT_KEYS:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v2 kernel receipt key set drifted"
        )
    _validate_self_hash(payload, "receipt_sha256", label="v2 kernel receipt")
    commitments = acquisition.get("commitments")
    preflight = marker.get("path_free_preflight")
    if not isinstance(commitments, Mapping) or not isinstance(preflight, Mapping):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v2 kernel predecessor bindings are unavailable"
        )
    expected_runtime = preflight.get("official_runtime_binding_sha256")
    official_peak = payload.get("official_peak_concurrency_count")
    local_peak = payload.get("local_peak_concurrency_count")
    sha_fields = (
        "action_pack_sha256",
        "label_pack_sha256",
        "runtime_binding_sha256",
        "official_postflight_receipt_sha256",
        "action_table_sha256",
        "action_seal_sha256",
        "action_seal_file_sha256",
    )
    if (
        payload.get("schema") != kernel_v2.RESULT_SCHEMA
        or payload.get("version") != kernel_v2.DESIGN_VERSION
        or payload.get("status") != kernel_v2.SUCCESS_RESULT_STATUS
        or payload.get("design_sha256") != kernel_v2.DESIGN_SHA256
        or payload.get("design_file_sha256") != kernel_v2.DESIGN_FILE_SHA256
        or payload.get("block") != kernel_v2.BLOCK
        or payload.get("recipe_id") != kernel_v2.RECIPE_ID
        or payload.get("seed_count") != 8
        or payload.get("item_count_per_seed") != 64
        or payload.get("total_item_count") != 512
        or payload.get("arms") != list(kernel_v2.ARM_IDS)
        or payload.get("action_work_unit_count") != 1536
        or payload.get("official_retrieve_action_count") != 512
        or payload.get("official_concurrency_cap") != 8
        or payload.get("local_concurrency_cap") != 64
        or not isinstance(official_peak, int)
        or isinstance(official_peak, bool)
        or not 1 <= official_peak <= 8
        or not isinstance(local_peak, int)
        or isinstance(local_peak, bool)
        or not 1 <= local_peak <= 64
        or payload.get("chunk_schedule_sha256")
        != kernel_v2.CHUNK_SCHEDULE_SHA256
        or payload.get("observed_encoder_input_row_counts") != [8448, 8448]
        or payload.get("observed_encoder_output_row_counts") != [8448, 8448]
        or payload.get("action_pack_file_sha256")
        != commitments.get("action_pack_file_sha256")
        or payload.get("action_item_commitment_set_sha256")
        != commitments.get("action_item_commitment_set_sha256")
        or payload.get("label_pack_file_sha256")
        != commitments.get("label_pack_file_sha256")
        or payload.get("label_item_commitment_set_sha256")
        != commitments.get("label_item_commitment_set_sha256")
        or not _is_lower_hex(expected_runtime, 64)
        or payload.get("runtime_binding_sha256") != expected_runtime
        or payload.get("official_postflight_receipt_sha256") != expected_runtime
        or any(not _is_lower_hex(payload.get(field), 64) for field in sha_fields)
        or not isinstance(payload.get("aggregates"), Mapping)
        or set(payload["aggregates"]) != set(kernel_v2.ARM_IDS)
        or not isinstance(payload.get("cluster_differences"), Mapping)
        or set(payload["cluster_differences"])
        != {"Agent_R1_minus_official_HippoRAG", "Agent_R1_minus_RAW"}
        or payload.get("interpretation")
        != "descriptive_fixed_cohort_replication_only"
        or payload.get("seeds_or_item_rows_disclosed") is not False
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v2 kernel receipt invariant drifted"
        )

    seal, seal_file_sha256 = _read_private_canonical_json(
        root / FORMAL_ACTION_SEAL_RELATIVE_PATH,
        field="v5 frozen-v2 private action seal",
    )
    if set(seal) != _V2_ACTION_SEAL_KEYS:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v2 private action seal key set drifted"
        )
    seal_sha256 = _validate_self_hash(
        seal, "action_seal_sha256", label="v2 private action seal"
    )
    rows = seal.get("action_rows")
    if (
        payload.get("action_seal_sha256") != seal_sha256
        or payload.get("action_seal_file_sha256") != seal_file_sha256
        or seal.get("schema") != f"{kernel_v2.VERSION}_private_action_seal"
        or seal.get("version") != kernel_v2.VERSION
        or seal.get("status")
        != "all_1536_actions_joined_official_postflight_terminal"
        or seal.get("purpose") != "fresh_formal_replication"
        or seal.get("block") != kernel_v2.BLOCK
        or seal.get("recipe_id") != kernel_v2.RECIPE_ID
        or any(
            seal.get(field) != expected
            for field, expected in {
                "item_count": 512,
                "action_work_unit_count": 1536,
                "submitted_action_work_unit_count": 1536,
                "terminal_action_work_unit_count": 1536,
                "official_retrieve_action_count": 512,
                "official_call_count": 512,
                "RAW_action_count": 512,
                "Agent_R1_action_count": 512,
                "official_concurrency_cap": 8,
                "local_concurrency_cap": 64,
            }.items()
        )
        or seal.get("official_peak_concurrency_count") != official_peak
        or seal.get("local_peak_concurrency_count") != local_peak
        or seal.get("chunk_schedule_sha256") != kernel_v2.CHUNK_SCHEDULE_SHA256
        or seal.get("observed_encoder_input_row_counts") != [8448, 8448]
        or seal.get("observed_encoder_output_row_counts") != [8448, 8448]
        or seal.get("action_pack_file_sha256")
        != payload.get("action_pack_file_sha256")
        or seal.get("action_pack_sha256") != payload.get("action_pack_sha256")
        or seal.get("action_item_commitment_set_sha256")
        != payload.get("action_item_commitment_set_sha256")
        or seal.get("runtime_binding_sha256") != expected_runtime
        or seal.get("official_postflight_receipt_sha256") != expected_runtime
        or seal.get("action_table_sha256") != payload.get("action_table_sha256")
        or seal.get("labels_opened_before_action_seal") is not False
        or seal.get("labels_opened_before_seal") is not False
        or seal.get("scores_computed_before_action_seal") is not False
        or not isinstance(rows, list)
        or len(rows) != 512
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v2 private action seal invariant drifted"
        )
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _V2_ACTION_ROW_KEYS:
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "v2 private action row schema drifted"
            )
        ranks = (
            row.get("RAW_top5"),
            row.get("official_HippoRAG_top5"),
            row.get("Agent_R1_top5"),
        )
        if (
            row.get("global_ordinal") != ordinal
            or any(
                not _is_lower_hex(row.get(field), 64)
                for field in (
                    "action_item_sha256",
                    "common_scan_sha256",
                    "local_tensor_sha256",
                )
            )
            or any(
                not isinstance(rank, list)
                or len(rank) != 5
                or len(set(rank)) != 5
                or any(
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < kernel_v2.NODE_COUNT
                    for index in rank
                )
                for rank in ranks
            )
        ):
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "v2 private action row invariant drifted"
            )
    recomputed_action_table = semantic_hash(
        [
            [
                row["RAW_top5"],
                row["official_HippoRAG_top5"],
                row["Agent_R1_top5"],
            ]
            for row in rows
        ]
    )
    if recomputed_action_table != payload.get("action_table_sha256"):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v2 action table binding drifted"
        )
    return dict(payload)


def run_formal_child(
    project_root: Path,
    *,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> dict[str, Any]:
    """Run the frozen v2 kernel on v5 work paths and the v3 source packs."""

    root = _require_canonical_root(project_root)
    marker, marker_file_sha256 = _load_formal_marker(root)
    freeze, _ = verify_implementation_freeze(root)
    closure = verify_v4_prefreeze_closure(root)
    acquisition = v3.load_committed_acquisition_receipt(
        root, verify_private_packs=False
    )
    acquisition_file_sha256 = sha256_file(
        root / v3.ACQUISITION_RECEIPT_RELATIVE_PATH
    )
    result_path = root / RESULT_RELATIVE_PATH
    if result_path.exists() or result_path.is_symlink():
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "canonical v5 formal result already exists"
        )
    child_invocation_id: str | None = None
    try:
        lexical = _lexical_runtime_python(runtime_python)
        llm = local_llm_model.expanduser().absolute()
        embedding = local_embedding_model.expanduser().resolve(strict=True)
        if (
            marker["runtime_python"] != str(lexical)
            or marker["local_llm_model"] != str(llm)
            or marker["local_embedding_model"] != str(embedding)
        ):
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "formal child runtime arguments differ from launch marker"
            )
        if Path.cwd().resolve() != root or not _offline_environment_exact():
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "formal child detached offline environment drifted"
            )
        invocation_id = os.environ.get("INVOCATION_ID")
        if not _is_lower_hex(invocation_id, 32):
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "formal child is not running in an attested systemd invocation"
            )
        child_invocation_id = str(invocation_id)
        if (
            marker.get("implementation_freeze_sha256")
            != freeze.get("implementation_freeze_sha256")
            or marker.get("v4_closure_receipt_sha256")
            != closure.get("closure_receipt_sha256")
            or marker.get("acquisition_receipt_sha256")
            != acquisition.get("receipt_sha256")
            or marker.get("acquisition_receipt_file_sha256")
            != acquisition_file_sha256
        ):
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "formal launch predecessor chain drifted"
            )
        encoder, runtime, preflight = _prepare_formal_resources(
            project_root=root,
            runtime_python=lexical,
            local_llm_model=llm,
            local_embedding_model=embedding,
        )
        if preflight != marker["path_free_preflight"]:
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "formal child preflight differs from launcher preflight"
            )
        commitments = acquisition.get("commitments")
        if not isinstance(commitments, Mapping):
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "formal acquisition commitments drifted"
            )
        action_pack = kernel_v2.load_action_pack(
            root / v3.ACTION_PACK_RELATIVE_PATH
        )
        kernel_v2._pack_matches_commitments(
            pack_file_sha256=action_pack.file_sha256,
            item_set_sha256=action_pack.item_commitment_set_sha256,
            commitments=commitments,
            prefix="action",
        )
        label_open_count = 0

        def load_late_labels() -> kernel_v2.LabelPack:
            nonlocal label_open_count
            label_open_count += 1
            if label_open_count != 1:
                raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                    "late labels opened more than once"
                )
            labels = kernel_v2.load_label_pack(root / v3.LABEL_PACK_RELATIVE_PATH)
            kernel_v2._pack_matches_commitments(
                pack_file_sha256=labels.file_sha256,
                item_set_sha256=labels.item_commitment_set_sha256,
                commitments=commitments,
                prefix="label",
            )
            return labels

        outcome = kernel_v2.run_multiseed_replication(
            action_pack,
            label_loader=load_late_labels,
            encoder=encoder,
            runtime=runtime,
            work_root=root / FORMAL_WORK_RELATIVE_PATH,
            action_seal_path=root / FORMAL_ACTION_SEAL_RELATIVE_PATH,
        )
        if label_open_count != 1:
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "late labels were not opened exactly once"
            )
        kernel_receipt = kernel_v2.multiseed_public_result(outcome)
        kernel_receipt = _validate_v2_kernel_receipt(
            root,
            kernel_receipt,
            acquisition=acquisition,
            marker=marker,
        )
        body = {
            "schema": RESULT_SCHEMA,
            "version": RESULT_VERSION,
            "status": SUCCESS_RESULT_STATUS,
            "invocation_HEAD": marker["actual_HEAD"],
            "design_sha256": DESIGN_SHA256,
            "design_file_sha256": DESIGN_FILE_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "v4_closure_receipt_sha256": closure["closure_receipt_sha256"],
            "source_cohort_version": SOURCE_COHORT_VERSION,
            "acquisition_receipt_sha256": acquisition["receipt_sha256"],
            "acquisition_receipt_file_sha256": acquisition_file_sha256,
            "generated_item_commitment_set_sha256": acquisition[
                "generated_item_commitment_set_sha256"
            ],
            "formal_launch_marker_sha256": marker["marker_sha256"],
            "formal_launch_marker_file_sha256": marker_file_sha256,
            "systemd_unit": FORMAL_SYSTEMD_UNIT,
            "systemd_invocation_id": str(invocation_id),
            "systemd_contract_sha256": marker["systemd_contract_sha256"],
            "preflight_sha256": preflight["preflight_sha256"],
            "official_attestation_v3_receipt_sha256": marker[
                "official_attestation_v3_receipt_sha256"
            ],
            "official_attestation_v3_receipt_file_sha256": marker[
                "official_attestation_v3_receipt_file_sha256"
            ],
            "wire_format_version": acquisition_v2.VERSION,
            "execution_kernel_version": kernel_v2.VERSION,
            "execution_kernel_receipt": kernel_receipt,
            "interpretation": (
                "descriptive_fixed_untouched_v3_cohort_runtime_identity_normalization_only"
            ),
            "performance_gate_or_promotion_computed": False,
            "seeds_or_item_rows_disclosed": False,
            "retry_replacement_or_relaunch_authorized": False,
            "result_must_be_committed_before_terminal_publication": True,
        }
        result = _self_hashed(body, "receipt_sha256")
        _write_json_exclusive(result_path, result, PUBLIC_MODE)
        return result
    except BaseException as exc:
        return _persist_formal_failure(
            project_root=root,
            marker=marker,
            marker_file_sha256=marker_file_sha256,
            freeze=freeze,
            acquisition=acquisition,
            acquisition_file_sha256=acquisition_file_sha256,
            failure_class=type(exc).__name__,
            administrative=False,
            systemd_invocation_id=child_invocation_id,
            pack_label_open_state="unknown",
            open_state_evidence="formal_child_exception",
        )


def _load_terminal_result_local(project_root: Path) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    result, _ = _read_public_json(
        root / RESULT_RELATIVE_PATH, field="v5 terminal result"
    )
    _validate_self_hash(result, "receipt_sha256", label="v5 terminal result")
    marker, marker_file_sha256 = _load_formal_marker(root)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("version") != RESULT_VERSION
        or result.get("status") not in {SUCCESS_RESULT_STATUS, FAILURE_RESULT_STATUS}
        or result.get("design_sha256") != DESIGN_SHA256
        or result.get("design_file_sha256") != DESIGN_FILE_SHA256
        or result.get("implementation_freeze_sha256")
        != marker.get("implementation_freeze_sha256")
        or result.get("v4_closure_receipt_sha256")
        != marker.get("v4_closure_receipt_sha256")
        or result.get("source_cohort_version") != SOURCE_COHORT_VERSION
        or result.get("acquisition_receipt_sha256")
        != marker.get("acquisition_receipt_sha256")
        or result.get("acquisition_receipt_file_sha256")
        != marker.get("acquisition_receipt_file_sha256")
        or result.get("generated_item_commitment_set_sha256")
        != marker.get("generated_item_commitment_set_sha256")
        or result.get("formal_launch_marker_sha256") != marker.get("marker_sha256")
        or result.get("formal_launch_marker_file_sha256") != marker_file_sha256
        or result.get("systemd_unit") != FORMAL_SYSTEMD_UNIT
        or result.get("systemd_contract_sha256")
        != marker.get("systemd_contract_sha256")
        or result.get("preflight_sha256")
        != marker.get("path_free_preflight", {}).get("preflight_sha256")
        or result.get("official_attestation_v3_receipt_sha256")
        != marker.get("official_attestation_v3_receipt_sha256")
        or result.get("official_attestation_v3_receipt_file_sha256")
        != marker.get("official_attestation_v3_receipt_file_sha256")
        or result.get("retry_replacement_or_relaunch_authorized") is not False
        or result.get("performance_gate_or_promotion_computed") is not False
        or result.get("result_must_be_committed_before_terminal_publication")
        is not True
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "v5 terminal result chain drifted"
        )
    if result.get("status") == SUCCESS_RESULT_STATUS:
        kernel_receipt = result.get("execution_kernel_receipt")
        if (
            result.get("wire_format_version") != acquisition_v2.VERSION
            or result.get("execution_kernel_version") != kernel_v2.VERSION
            or not _is_lower_hex(result.get("systemd_invocation_id"), 32)
            or not isinstance(kernel_receipt, Mapping)
            or result.get("seeds_or_item_rows_disclosed") is not False
        ):
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "successful v5 kernel receipt drifted"
            )
        acquisition = v3.load_committed_acquisition_receipt(
            root, verify_private_packs=False
        )
        _validate_v2_kernel_receipt(
            root,
            kernel_receipt,
            acquisition=acquisition,
            marker=marker,
        )
    else:
        evidence = result.get("pack_label_open_state_evidence")
        expected_state = {
            "systemd_launch_failure_before_child_start": "unopened",
            "formal_child_exception": "unknown",
            "administrative_finalizer_without_durable_child_evidence": "unknown",
            "launcher_nonzero_with_verified_terminal_unit": "unknown",
        }.get(evidence)
        invocation_id = result.get("systemd_invocation_id")
        invocation_source = result.get("systemd_invocation_id_source")
        if (
            not isinstance(result.get("failure_class"), str)
            or not result.get("failure_class")
            or expected_state is None
            or result.get("private_pack_open_state") != expected_state
            or result.get("label_pack_open_state") != expected_state
            or (
                invocation_id is not None
                and not _is_lower_hex(invocation_id, 32)
            )
            or (
                evidence == "formal_child_exception"
                and (
                    result.get("administrative_finalization") is not False
                    or invocation_source != "formal_child_environment"
                )
            )
            or (
                evidence == "systemd_launch_failure_before_child_start"
                and (
                    result.get("administrative_finalization") is not True
                    or invocation_id is not None
                    or invocation_source is not None
                )
            )
            or (
                evidence
                in {
                    "administrative_finalizer_without_durable_child_evidence",
                    "launcher_nonzero_with_verified_terminal_unit",
                }
                and (
                    result.get("administrative_finalization") is not True
                    or invocation_source
                    != (
                        "verified_target_unit_property"
                        if invocation_id is not None
                        else None
                    )
                )
            )
            or result.get(
                "exception_message_seed_item_or_label_content_persisted_publicly"
            )
            is not False
        ):
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "failed v5 terminal result drifted"
            )
    return result


def load_committed_terminal_result(project_root: Path) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    try:
        committed = v3._load_committed_public_json(
            root, RESULT_RELATIVE_PATH, "committed v5 terminal result"
        )
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "committed v5 terminal result is unavailable"
        ) from exc
    local = _load_terminal_result_local(root)
    if committed != local:
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "committed v5 terminal result readback drifted"
        )
    return local


_SYSTEMD_SHOW_FIELDS = (
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "ControlGroup",
    "Result",
    "ExecMainCode",
    "ExecMainStatus",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
    "InvocationID",
)
_SYSTEMD_TERMINAL_STATE_PAIRS = frozenset(
    {("active", "exited"), ("inactive", "dead"), ("failed", "failed")}
)
_SYSTEMD_TERMINAL_RESULTS = frozenset(
    {
        "success",
        "exit-code",
        "signal",
        "core-dump",
        "watchdog",
        "start-limit-hit",
        "timeout",
        "resources",
        "protocol",
        "oom-kill",
    }
)
_SYSTEMD_UNSET_VALUES = frozenset({"", "n/a"})


def _complete_systemd_state(state: Mapping[str, str]) -> bool:
    return set(state) == set(_SYSTEMD_SHOW_FIELDS)


def _safe_systemd_state(
    state: Mapping[str, str], *, returncode: int
) -> dict[str, Any]:
    safe: dict[str, Any] = {
        key: value for key, value in state.items() if key != "ControlGroup"
    }
    safe["systemctl_returncode"] = returncode
    return safe


def _systemd_state_proves_child_never_started(
    state: Mapping[str, str], *, returncode: int
) -> bool:
    """Accept only a complete, positive systemd proof of zero child execution."""

    if returncode not in {0, 4} or not _complete_systemd_state(state):
        return False
    execution_never_started = (
        state.get("MainPID") == "0"
        and state.get("ExecMainCode") in {"", "0"}
        and state.get("ExecMainStatus") in {"", "0"}
        and state.get("ExecMainStartTimestamp") in _SYSTEMD_UNSET_VALUES
        and state.get("ExecMainExitTimestamp") in _SYSTEMD_UNSET_VALUES
    )
    if not execution_never_started:
        return False
    unit_absent = (
        state.get("LoadState") == "not-found"
        and state.get("ActiveState") == "inactive"
        and state.get("SubState") == "dead"
        and state.get("InvocationID") in _SYSTEMD_UNSET_VALUES
    )
    unit_terminal_before_exec = (
        state.get("LoadState") == "loaded"
        and (state.get("ActiveState"), state.get("SubState"))
        in {("inactive", "dead"), ("failed", "failed")}
        and state.get("Result") in _SYSTEMD_TERMINAL_RESULTS
        and (
            state.get("InvocationID") in _SYSTEMD_UNSET_VALUES
            or _is_lower_hex(state.get("InvocationID"), 32)
        )
    )
    return unit_absent or unit_terminal_before_exec


def _validated_terminal_systemd_state(
    state: Mapping[str, str], *, returncode: int
) -> tuple[dict[str, Any], str]:
    """Require complete positive evidence that the exact target unit is terminal."""

    invocation_id = state.get("InvocationID")
    timestamps_complete = all(
        isinstance(state.get(field), str)
        and state.get(field) not in _SYSTEMD_UNSET_VALUES
        for field in ("ExecMainStartTimestamp", "ExecMainExitTimestamp")
    )
    if (
        returncode != 0
        or not _complete_systemd_state(state)
        or state.get("LoadState") != "loaded"
        or (state.get("ActiveState"), state.get("SubState"))
        not in _SYSTEMD_TERMINAL_STATE_PAIRS
        or state.get("MainPID") != "0"
        or state.get("Result") not in _SYSTEMD_TERMINAL_RESULTS
        or state.get("ExecMainCode") not in {"1", "2", "3"}
        or not str(state.get("ExecMainStatus", "")).isdigit()
        or not timestamps_complete
        or not _is_lower_hex(invocation_id, 32)
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "target systemd unit lacks complete positive terminal evidence"
        )
    return _safe_systemd_state(state, returncode=returncode), str(invocation_id)


def _read_systemd_state(
    project_root: Path, *, run: RunCallable
) -> tuple[dict[str, str], int]:
    root = _require_canonical_root(project_root)
    argv = [
        "systemctl",
        "--user",
        "show",
        f"{FORMAL_SYSTEMD_UNIT}.service",
        "--no-pager",
        *[f"--property={field}" for field in _SYSTEMD_SHOW_FIELDS],
    ]
    completed = _call_run(run, argv, cwd=root)
    state: dict[str, str] = {}
    for line in (completed.stdout or "").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in _SYSTEMD_SHOW_FIELDS and key not in state:
            state[key] = value
    return state, int(completed.returncode)


def finalize_formal(
    project_root: Path, *, run: RunCallable = subprocess.run
) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    result_path = root / RESULT_RELATIVE_PATH
    if result_path.is_file() and not result_path.is_symlink():
        return _load_terminal_result_local(root)
    marker, marker_file_sha256 = _load_formal_marker(root)
    freeze, _ = verify_implementation_freeze(root)
    acquisition = v3.load_committed_acquisition_receipt(
        root, verify_private_packs=False
    )
    acquisition_file_sha256 = sha256_file(
        root / v3.ACQUISITION_RECEIPT_RELATIVE_PATH
    )
    state, returncode = _read_systemd_state(root, run=run)
    safe_state, verified_invocation_id = _validated_terminal_systemd_state(
        state, returncode=returncode
    )
    failure_class = (
        "DetachedSystemdServiceTerminalWithoutCanonicalResult"
    )
    return _persist_formal_failure(
        project_root=root,
        marker=marker,
        marker_file_sha256=marker_file_sha256,
        freeze=freeze,
        acquisition=acquisition,
        acquisition_file_sha256=acquisition_file_sha256,
        failure_class=failure_class,
        administrative=True,
        systemd_invocation_id=verified_invocation_id,
        pack_label_open_state="unknown",
        open_state_evidence="administrative_finalizer_without_durable_child_evidence",
        systemd_state=safe_state,
    )


def publish_terminal(project_root: Path) -> dict[str, Any]:
    """Publish only the stored v3 seeds/cohort after a committed v5 result."""

    root = _require_canonical_root(project_root)
    freeze, actual_head = verify_implementation_freeze(root)
    closure = verify_v4_prefreeze_closure(root)
    acquisition = v3.load_committed_acquisition_receipt(
        root, verify_private_packs=False
    )
    custody = v3.load_committed_seed_custody(root, verify_private_batch=False)
    result = load_committed_terminal_result(root)
    if (
        result.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or result.get("v4_closure_receipt_sha256")
        != closure.get("closure_receipt_sha256")
        or result.get("acquisition_receipt_sha256")
        != acquisition.get("receipt_sha256")
        or result.get("generated_item_commitment_set_sha256")
        != acquisition.get("generated_item_commitment_set_sha256")
    ):
        raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
            "terminal publication predecessor chain drifted"
        )
    _assert_absent(
        root,
        (
            PUBLICATION_MARKER_RELATIVE_PATH,
            PUBLICATION_RELATIVE_PATH,
            PUBLICATION_FAILURE_RELATIVE_PATH,
        ),
        "v5 terminal publication output",
    )
    result_file_sha256 = sha256_file(root / RESULT_RELATIVE_PATH)
    marker = _self_hashed(
        {
            "schema": f"{PUBLICATION_SCHEMA}_attempt_marker",
            "version": RESULT_VERSION,
            "status": "sole_v5_terminal_reproducibility_publication_attempt_consumed",
            "actual_HEAD": actual_head,
            "design_sha256": DESIGN_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "terminal_result_receipt_sha256": result["receipt_sha256"],
            "terminal_result_file_sha256": result_file_sha256,
            "source_cohort_version": SOURCE_COHORT_VERSION,
            "seed_batch_commitment_sha256": custody[
                "seed_batch_commitment_sha256"
            ],
            "attempt_count": 1,
            "private_seed_or_compiled_pack_opened_before_marker": False,
        },
        "marker_sha256",
    )
    marker_file_sha256 = _write_json_exclusive(
        root / PUBLICATION_MARKER_RELATIVE_PATH, marker, PRIVATE_MODE
    )
    try:
        seeds = _read_seed_batch(root / v3.SEED_BATCH_RELATIVE_PATH, custody)
        commitments = acquisition.get("commitments")
        if not isinstance(commitments, Mapping):
            raise SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error(
                "publication acquisition commitments drifted"
            )
        compiled_pack = v3._verify_compiled_cohort_pack(
            root,
            expected_file_sha256=commitments[
                "compiled_cohort_pack_file_sha256"
            ],
            expected_row_set_sha256=commitments[
                "compiled_row_commitment_set_sha256"
            ],
            expected_item_set_sha256=acquisition[
                "generated_item_commitment_set_sha256"
            ],
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
                "version": RESULT_VERSION,
                "status": (
                    "terminal_v5_source_v3_eight_seeds_and_full_compiled_cohort_published"
                ),
                "publication_HEAD": actual_head,
                "design_sha256": DESIGN_SHA256,
                "design_file_sha256": DESIGN_FILE_SHA256,
                "implementation_freeze_sha256": freeze[
                    "implementation_freeze_sha256"
                ],
                "v4_closure_receipt_sha256": closure["closure_receipt_sha256"],
                "source_cohort_version": SOURCE_COHORT_VERSION,
                "source_cohort_claim": (
                    "prospectively_generated_untouched_v3_cohort_reused_for_runtime_identity_normalization"
                ),
                "acquisition_receipt_sha256": acquisition["receipt_sha256"],
                "terminal_result_status": result["status"],
                "terminal_result_receipt_sha256": result["receipt_sha256"],
                "terminal_result_file_sha256": result_file_sha256,
                "publication_attempt_marker_sha256": marker["marker_sha256"],
                "publication_attempt_marker_file_sha256": marker_file_sha256,
                "seed_encoding": "ordered_lowercase_hex_exact_32_bytes_each",
                "formal_seed_hexes": [seed.hex() for seed in seeds],
                "seed_batch_commitment_sha256": hashlib.sha256(
                    b"".join(seeds)
                ).hexdigest(),
                "ordered_seed_commitments_sha256": [
                    hashlib.sha256(seed).hexdigest() for seed in seeds
                ],
                "block": v3.BLOCK,
                "seed_count": v3.SEED_COUNT,
                "item_count_per_seed": v3.ITEMS_PER_SEED,
                "total_item_count": v3.TOTAL_ITEM_COUNT,
                "generated_item_commitment_set_sha256": stable_hash(
                    item_commitments
                ),
                "items": rows,
                "cohort_regenerated_during_publication": False,
                "grammar_generate_block_call_count_during_publication": 0,
                "retrieval_actions_model_outputs_or_scores_included": False,
            },
            "reproducibility_sha256",
        )
        _write_json_exclusive(
            root / PUBLICATION_RELATIVE_PATH, artifact, PUBLIC_MODE
        )
        return artifact
    except BaseException as exc:
        failure = _self_hashed(
            {
                "schema": f"{PUBLICATION_SCHEMA}_failure_receipt",
                "version": RESULT_VERSION,
                "status": "terminal_v5_publication_invalid_no_replay",
                "marker_sha256": marker["marker_sha256"],
                "terminal_result_receipt_sha256": result["receipt_sha256"],
                "failure_class": type(exc).__name__,
                "exception_message_seed_or_item_rows_persisted_publicly": False,
                "retry_or_replay_authorized": False,
            },
            "receipt_sha256",
        )
        failure_path = root / PUBLICATION_FAILURE_RELATIVE_PATH
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_json_exclusive(failure_path, failure, PUBLIC_MODE)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "finalize-formal", "publish-terminal"):
        child = subparsers.add_parser(command)
        child.add_argument("--project-root", required=True, type=Path)
    for command in ("launch-formal", "formal-child"):
        child = subparsers.add_parser(command)
        child.add_argument("--project-root", required=True, type=Path)
        child.add_argument("--runtime-python", required=True, type=Path)
        child.add_argument("--local-llm-model", required=True, type=Path)
        child.add_argument("--local-embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "freeze":
        result = create_implementation_freeze(arguments.project_root)
    elif arguments.command == "launch-formal":
        result = launch_formal(
            arguments.project_root,
            runtime_python=arguments.runtime_python,
            local_llm_model=arguments.local_llm_model,
            local_embedding_model=arguments.local_embedding_model,
        )
    elif arguments.command == "formal-child":
        result = run_formal_child(
            arguments.project_root,
            runtime_python=arguments.runtime_python,
            local_llm_model=arguments.local_llm_model,
            local_embedding_model=arguments.local_embedding_model,
        )
    elif arguments.command == "finalize-formal":
        result = finalize_formal(arguments.project_root)
    else:
        result = publish_terminal(arguments.project_root)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
