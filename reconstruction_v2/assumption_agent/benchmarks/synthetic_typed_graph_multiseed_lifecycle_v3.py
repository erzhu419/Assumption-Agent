"""Detached, fresh-seed formal lifecycle for synthetic multiseed replication v3.

V3 is deliberately a thin lifecycle around the exact committed v2 execution
kernel.  It never reuses or mutates the terminal v2 diagnostic workspace and
does not add another 512-item diagnostic or a performance gate.  All durable
attempt markers are created before launching work, all formal paths are new,
and every formal outcome is terminal and non-replayable.
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
import time
from typing import Any, Callable, Mapping, Sequence

from replication_runtime.qasper_minilm_v1 import OfflineMiniLMEncoder

from . import synthetic_typed_graph_causal_grammar_v1 as grammar
from . import synthetic_typed_graph_multiseed_acquisition_v2 as acquisition_v2
from . import synthetic_typed_graph_multiseed_runner_v2 as kernel_v2
from .musique_formal_runtime_binding_v2 import PreparedFormalRuntimeV2


VERSION = "synthetic_typed_graph_multiseed_replication_v3"
MODULE_VERSION = "synthetic_typed_graph_multiseed_lifecycle_v3"
DESIGN_SCHEMA = "synthetic_typed_graph_multiseed_replication_design_v3"
# Filled from the committed design before any lifecycle command is consumed.
DESIGN_SHA256 = "8a654522fd0e74f565ebe9a5bf4d7ea8565ed3fb542918facb7961c15ef3a739"
DESIGN_FILE_SHA256 = "a9c72e4e4b9fe820ba21717cfd45f3ca27fdfc5e3fe9b8d6dac6e50ae3ef2b33"

SEED_COUNT = 8
SEED_BYTES = 32
SEED_BATCH_BYTES = SEED_COUNT * SEED_BYTES
BLOCK = "A_hold"
ITEMS_PER_SEED = 64
TOTAL_ITEM_COUNT = SEED_COUNT * ITEMS_PER_SEED
PRIVATE_MODE = 0o600
PUBLIC_MODE = 0o644

DESIGN_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_design_v3.json"
)
SMOKE_RECEIPT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_preseed_verification_v3.json"
)
IMPLEMENTATION_FREEZE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_implementation_freeze_v3.json"
)
SEED_CUSTODY_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_seed_custody_v3.json"
)
SEED_FAILURE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_seed_failure_v3.json"
)
ACQUISITION_RECEIPT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_acquisition_v3.json"
)
RESULT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_result_v3.json"
)
PUBLICATION_FAILURE_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_publication_failure_v3.json"
)
ARTIFACT_ROOT_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_multiseed_replication_v3"
)
SMOKE_LAUNCH_MARKER_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "smoke/launch.marker"
SEED_MARKER_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "seed_generation.attempt.marker"
SEED_BATCH_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "seed_batch.bin"
ACQUISITION_MARKER_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "acquisition.attempt.marker"
ACTION_PACK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "action_pack.json"
LABEL_PACK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "label_pack.json"
COMPILED_COHORT_PACK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "full_compiled_cohort_pack.json"
FORMAL_LAUNCH_MARKER_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "runner/formal.launch.marker"
FORMAL_WORK_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "runner/formal.work"
FORMAL_ACTION_SEAL_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "runner/action_seal.json"
PUBLICATION_MARKER_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "publish_terminal.attempt.marker"
PUBLICATION_RELATIVE_PATH = Path(
    "published/synthetic_typed_graph_multiseed_replication_v3/formal_seeds_and_cohort.json"
)

SMOKE_SYSTEMD_UNIT = "assumption-synth-multiseed-v3-smoke"
FORMAL_SYSTEMD_UNIT = "assumption-synth-multiseed-v3-formal"

SMOKE_SUCCESS_STATUS = "detached_preseed_process_custody_verified_no_data_or_models_opened"
SMOKE_FAILURE_STATUS = "terminal_preseed_infrastructure_invalid_no_formal_authorized"
FREEZE_STATUS = "complete_preseed_v3_implementation_frozen_must_commit_before_seed"
SEED_CUSTODY_STATUS = "eight_fresh_v3_seeds_committed_cohort_not_generated"
ACQUISITION_STATUS = "formal_v3_multiseed_A_hold_cohort_acquired_private_labels_separated"
SUCCESS_RESULT_STATUS = "terminal_descriptive_eight_seed_replication_complete"
FAILURE_RESULT_STATUS = "terminal_infrastructure_or_implementation_invalid_no_replay"

SMOKE_SCHEMA = "synthetic_typed_graph_multiseed_replication_preseed_verification_v3"
FREEZE_SCHEMA = "synthetic_typed_graph_multiseed_replication_implementation_freeze_v3"
SEED_CUSTODY_SCHEMA = "synthetic_typed_graph_multiseed_replication_seed_custody_v3"
ACQUISITION_SCHEMA = "synthetic_typed_graph_multiseed_replication_acquisition_v3"
RESULT_SCHEMA = "synthetic_typed_graph_multiseed_replication_result_v3"
PUBLICATION_SCHEMA = "synthetic_typed_graph_multiseed_terminal_reproducibility_v3"

V2_INTERRUPTION_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_integration_diagnostic_v2.json"
)
V2_INTERRUPTION_FILE_SHA256 = "eb1f93013c40eb0507231e62b9d7707b11d1bfce66890298dd4a53eb286d5471"
V2_INTERRUPTION_SHA256 = "e4e1d4df0505c727376ce348010ca16f69d093935583fcb5d3cd57478370a51c"
V2_INTERRUPTION_MARKER_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_multiseed_replication_v2/integration_diagnostic/attempt.marker"
)
V2_INTERRUPTION_MARKER_FILE_SHA256 = "91292adad61cb2bad6c298da7c54abad87de48bdaaf07379b064f1a73fef19f2"
V2_ACQUISITION_KERNEL_FILE_SHA256 = "aefe03d9227d3b86eb075212b023f6409228b1e2949a0037cdba772b18694ded"
V2_RUNNER_KERNEL_FILE_SHA256 = "ed2b1d81676645669e327054dbe0a10856727cc50e02cb29f423736de371ca31"
V2_ACQUISITION_TEST_FILE_SHA256 = "8c8adf87255b9891c6182148c6d3867f4dab52dbce85fba9ea03587f7ac472a0"
V2_RUNNER_TEST_FILE_SHA256 = "9949d3e67419857be53213aaaa2af4ad1450f9874084ede6a9259a4a0b682c8f"

V3_DRIVER_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_lifecycle_v3.py"
)
V3_TEST_RELATIVE_PATH = Path("tests/test_synthetic_typed_graph_multiseed_lifecycle_v3.py")

SMOKE_BINDING_PATHS = (
    V3_DRIVER_RELATIVE_PATH.as_posix(),
    V3_TEST_RELATIVE_PATH.as_posix(),
    "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_acquisition_v2.py",
    "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v2.py",
    "tests/test_synthetic_typed_graph_multiseed_acquisition_v2.py",
    "tests/test_synthetic_typed_graph_multiseed_runner_v2.py",
)

SYSTEMD_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TMPDIR": "/tmp",
    "TRANSFORMERS_OFFLINE": "1",
}
SYSTEMD_PROPERTIES = {
    "KillMode": "control-group",
    "RemainAfterExit": "yes",
    "Restart": "no",
    "StandardError": "journal",
    "StandardOutput": "journal",
    "TimeoutStopSec": "60s",
    "Type": "exec",
    "UMask": "0077",
}

ORDERED_LAUNCHER_PREFIX_CONTRACT = [
    "systemd-run",
    "--user",
    "--unit=BOUND_UNIT",
    "--service-type=exec",
    "--remain-after-exit",
    "--working-directory=CANONICAL_PROJECT_ROOT",
    "--property=StandardOutput=journal",
    "--property=StandardError=journal",
    "--property=KillMode=control-group",
    "--property=Restart=no",
    "--property=UMask=0077",
    "--property=TimeoutStopSec=60s",
    "--setenv=TMPDIR=/tmp",
    "--setenv=HF_HUB_OFFLINE=1",
    "--setenv=TRANSFORMERS_OFFLINE=1",
    "ABSOLUTE_CURRENT_PYTHON",
]
SMOKE_CHILD_ORDERED_ARGV_SUFFIX_CONTRACT = [
    "-u",
    "-m",
    "assumption_agent.benchmarks.synthetic_typed_graph_multiseed_lifecycle_v3",
    "smoke-child",
    "--project-root",
    "CANONICAL_PROJECT_ROOT",
]
FORMAL_CHILD_ORDERED_ARGV_SUFFIX_CONTRACT = [
    "-u",
    "-m",
    "assumption_agent.benchmarks.synthetic_typed_graph_multiseed_lifecycle_v3",
    "formal-child",
    "--project-root",
    "CANONICAL_PROJECT_ROOT",
    "--runtime-python",
    "BOUND_RUNTIME_PYTHON",
    "--local-llm-model",
    "BOUND_LOCAL_LLM_MODEL",
    "--local-embedding-model",
    "BOUND_LOCAL_EMBEDDING_MODEL",
]
V1_V2_TEST_COMMAND = (
    "TMPDIR=/tmp python3 -m pytest -q -s "
    "tests/test_synthetic_typed_graph_causal_grammar_v1.py "
    "tests/test_synthetic_typed_graph_causal_formal_v1.py "
    "tests/test_synthetic_typed_graph_multiseed_acquisition_v1.py "
    "tests/test_synthetic_typed_graph_multiseed_runner_v1.py "
    "tests/test_synthetic_typed_graph_multiseed_acquisition_v2.py "
    "tests/test_synthetic_typed_graph_multiseed_runner_v2.py"
)

V2_FORMAL_ABSENCE_PATHS = (
    Path("artifacts/synthetic_typed_graph_multiseed_replication_v2/seed_generation.attempt.marker"),
    Path("artifacts/synthetic_typed_graph_multiseed_replication_v2/seed_batch.bin"),
    Path("manifests/synthetic_typed_graph_multiseed_replication_seed_custody_v2.json"),
    Path("artifacts/synthetic_typed_graph_multiseed_replication_v2/acquisition.attempt.marker"),
    Path("artifacts/synthetic_typed_graph_multiseed_replication_v2/action_pack.json"),
    Path("artifacts/synthetic_typed_graph_multiseed_replication_v2/label_pack.json"),
    Path("artifacts/synthetic_typed_graph_multiseed_replication_v2/full_compiled_cohort_pack.json"),
    Path("manifests/synthetic_typed_graph_multiseed_replication_acquisition_v2.json"),
    Path("artifacts/synthetic_typed_graph_multiseed_replication_v2/runner/formal.attempt.marker"),
    Path("artifacts/synthetic_typed_graph_multiseed_replication_v2/runner/formal.work"),
    Path("artifacts/synthetic_typed_graph_multiseed_replication_v2/runner/action_seal.json"),
    Path("manifests/synthetic_typed_graph_multiseed_replication_result_v2.json"),
    Path("published/synthetic_typed_graph_multiseed_replication_v2/formal_seeds_and_cohort.json"),
)

REQUIRED_FREEZE_PATHS = frozenset(
    {
        DESIGN_RELATIVE_PATH.as_posix(),
        SMOKE_RECEIPT_RELATIVE_PATH.as_posix(),
        V3_DRIVER_RELATIVE_PATH.as_posix(),
        V3_TEST_RELATIVE_PATH.as_posix(),
        "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_acquisition_v2.py",
        "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v2.py",
        "tests/test_synthetic_typed_graph_multiseed_acquisition_v2.py",
        "tests/test_synthetic_typed_graph_multiseed_runner_v2.py",
        *acquisition_v2.REQUIRED_FREEZE_PATHS,
    }
)

# Deliberate aliases: v3 reuses the exact frozen, audited v2 durability helpers.
canonical_bytes = acquisition_v2.canonical_bytes
semantic_hash = acquisition_v2.semantic_hash
sha256_file = acquisition_v2.sha256_file
stable_hash = acquisition_v2.stable_hash
_assert_no_symlink_components = acquisition_v2._assert_no_symlink_components
_write_exclusive = acquisition_v2._write_exclusive
_write_json_exclusive = acquisition_v2._write_json_exclusive
_read_json_with_mode = acquisition_v2._read_json_with_mode
_git = acquisition_v2._git
_committed_bytes = acquisition_v2._committed_bytes
_git_blob_sha1 = acquisition_v2._git_blob_sha1
_load_committed_public_json = acquisition_v2._load_committed_public_json
_is_lower_hex = acquisition_v2._is_lower_hex
_read_seed_batch = acquisition_v2._read_seed_batch
_load_prior_item_commitments_after_marker = (
    acquisition_v2._load_prior_item_commitments_after_marker
)


class SyntheticTypedGraphMultiseedLifecycleV3Error(RuntimeError):
    """A v3 lifecycle, custody, detached-run, or publication invariant failed."""


RunCallable = Callable[..., subprocess.CompletedProcess[str]]


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            f"self-hash field already exists: {field}"
        )
    return {**dict(body), field: semantic_hash(dict(body))}


def _committed_head(project_root: Path) -> str:
    value = _git(project_root, "rev-parse", "HEAD").decode("ascii").strip()
    if not _is_lower_hex(value, 40):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error("Git HEAD drifted")
    return value


def _require_canonical_root(project_root: Path) -> Path:
    root = _assert_no_symlink_components(project_root.resolve(strict=True), "project root")
    if not root.is_dir():
        raise SyntheticTypedGraphMultiseedLifecycleV3Error("project root is unavailable")
    return root


def _read_canonical_json(
    path: Path, *, expected_mode: int, field: str
) -> tuple[dict[str, Any], str]:
    payload, file_sha256 = _read_json_with_mode(
        path, expected_mode=expected_mode, field=field
    )
    if path.read_bytes() != canonical_bytes(payload) + b"\n":
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            f"{field} canonical encoding drifted"
        )
    return payload, file_sha256


def _validate_self_hash(
    payload: Mapping[str, Any], field: str, *, label: str
) -> str:
    body = dict(payload)
    declared = body.pop(field, None)
    if not _is_lower_hex(declared, 64) or semantic_hash(body) != declared:
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(f"{label} self-hash drifted")
    return str(declared)


def _assert_absent(root: Path, relatives: Sequence[Path], field: str) -> None:
    occupied = [
        relative.as_posix()
        for relative in relatives
        if (root / relative).exists() or (root / relative).is_symlink()
    ]
    if occupied:
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            f"{field} unexpectedly exists: {occupied[0]}"
        )


def _v3_formal_output_paths() -> tuple[Path, ...]:
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
        FORMAL_LAUNCH_MARKER_RELATIVE_PATH,
        FORMAL_WORK_RELATIVE_PATH,
        FORMAL_ACTION_SEAL_RELATIVE_PATH,
        RESULT_RELATIVE_PATH,
        PUBLICATION_MARKER_RELATIVE_PATH,
        PUBLICATION_RELATIVE_PATH,
        PUBLICATION_FAILURE_RELATIVE_PATH,
    )


def verify_frozen_design(project_root: Path) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    path = root / DESIGN_RELATIVE_PATH
    if (
        not path.is_file()
        or path.is_symlink()
        or stat.S_IMODE(path.stat().st_mode) != PUBLIC_MODE
        or sha256_file(path) != DESIGN_FILE_SHA256
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error("frozen v3 design drifted")
    if _committed_bytes(root, DESIGN_RELATIVE_PATH) != path.read_bytes():
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "frozen v3 design is not current-HEAD committed"
        )
    # The design is human-readable JSON whose exact bytes are already bound by
    # DESIGN_FILE_SHA256; lifecycle receipts use the stricter canonical form.
    design, _ = _read_json_with_mode(
        path, expected_mode=PUBLIC_MODE, field="frozen v3 design"
    )
    if (
        design.get("schema") != DESIGN_SCHEMA
        or design.get("version") != VERSION
        or _validate_self_hash(design, "design_sha256", label="v3 design")
        != DESIGN_SHA256
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error("frozen v3 design binding drifted")
    paths = design.get("path_contract")
    systemd_contract = design.get("systemd_contract")
    smoke_contract = design.get("smoke_contract")
    test_contract = design.get("test_contract")
    if (
        not isinstance(paths, Mapping)
        or paths.get("driver") != V3_DRIVER_RELATIVE_PATH.as_posix()
        or paths.get("test") != V3_TEST_RELATIVE_PATH.as_posix()
        or paths.get("smoke_receipt") != SMOKE_RECEIPT_RELATIVE_PATH.as_posix()
        or paths.get("formal_result") != RESULT_RELATIVE_PATH.as_posix()
        or not isinstance(systemd_contract, Mapping)
        or systemd_contract.get("formal_unit") != FORMAL_SYSTEMD_UNIT
        or systemd_contract.get("smoke_unit") != SMOKE_SYSTEMD_UNIT
        or systemd_contract.get("environment") != SYSTEMD_ENVIRONMENT
        or systemd_contract.get("properties") != SYSTEMD_PROPERTIES
        or systemd_contract.get("ordered_launcher_prefix")
        != ORDERED_LAUNCHER_PREFIX_CONTRACT
        or systemd_contract.get("smoke_child_ordered_argv_suffix")
        != SMOKE_CHILD_ORDERED_ARGV_SUFFIX_CONTRACT
        or systemd_contract.get("formal_child_ordered_argv_suffix")
        != FORMAL_CHILD_ORDERED_ARGV_SUFFIX_CONTRACT
        or not isinstance(smoke_contract, Mapping)
        or smoke_contract.get("child_sleep_seconds") != 10
        or smoke_contract.get("success_status") != SMOKE_SUCCESS_STATUS
        or smoke_contract.get("failure_status") != SMOKE_FAILURE_STATUS
        or smoke_contract.get("successful_receipt_must_bind_exact_code_and_test_tuples")
        != list(SMOKE_BINDING_PATHS)
        or not isinstance(test_contract, Mapping)
        or test_contract.get("existing_v1_v2_test_count") != 57
        or test_contract.get("exact_command") != V1_V2_TEST_COMMAND
        or test_contract.get("required_result")
        != "exit_zero_with_57_passed_before_the_v3_smoke"
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error("v3 lifecycle contract drifted")
    return design


def _validate_v2_interruption_and_absence(project_root: Path) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    interruption = _load_committed_public_json(
        root, V2_INTERRUPTION_RELATIVE_PATH, "committed v2 interruption"
    )
    if sha256_file(root / V2_INTERRUPTION_RELATIVE_PATH) != V2_INTERRUPTION_FILE_SHA256:
        raise SyntheticTypedGraphMultiseedLifecycleV3Error("v2 interruption bytes drifted")
    if (
        interruption.get("status")
        != "terminal_integration_diagnostic_invalid_fresh_formal_not_authorized"
        or interruption.get("failure_class") != "ExternalProcessTermination"
        or interruption.get("fresh_formal_seed_authorized") is not False
        or interruption.get("labels_opened") is not False
        or interruption.get("scores_computed") is not False
        or interruption.get("estimands_computed") is not False
        or interruption.get("claims_made") is not False
        or _validate_self_hash(
            interruption, "diagnostic_sha256", label="v2 interruption"
        )
        != V2_INTERRUPTION_SHA256
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "v2 interruption provenance drifted"
        )
    marker, marker_file_sha256 = _read_canonical_json(
        root / V2_INTERRUPTION_MARKER_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="v2 interruption marker",
    )
    marker_sha256 = _validate_self_hash(marker, "marker_sha256", label="v2 marker")
    if (
        marker_file_sha256 != V2_INTERRUPTION_MARKER_FILE_SHA256
        or marker.get("status")
        != "sole_public_label_free_integration_diagnostic_attempt_consumed"
        or marker.get("fresh_formal_seed_or_cohort_exists") is not False
        or interruption.get("diagnostic_attempt_marker_sha256") != marker_sha256
        or interruption.get("diagnostic_attempt_marker_file_sha256")
        != marker_file_sha256
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "v2 interruption marker chain drifted"
        )
    _assert_absent(root, V2_FORMAL_ABSENCE_PATHS, "v2 formal output")
    return {
        "diagnostic_sha256": V2_INTERRUPTION_SHA256,
        "file_sha256": V2_INTERRUPTION_FILE_SHA256,
        "marker_sha256": marker_sha256,
        "marker_file_sha256": marker_file_sha256,
        "formal_outputs_absent": True,
    }


def _systemd_child_argv(
    root: Path,
    command: str,
    *,
    runtime_python: Path | None = None,
    local_llm_model: Path | None = None,
    local_embedding_model: Path | None = None,
) -> list[str]:
    argv = [
        str(Path(sys.executable).resolve(strict=True)),
        "-u",
        "-m",
        "assumption_agent.benchmarks.synthetic_typed_graph_multiseed_lifecycle_v3",
        command,
        "--project-root",
        str(root),
    ]
    optional = (
        ("--runtime-python", runtime_python),
        ("--local-llm-model", local_llm_model),
        ("--local-embedding-model", local_embedding_model),
    )
    for flag, value in optional:
        if value is not None:
            argv.extend((flag, str(value.resolve(strict=True))))
    return argv


def _systemd_run_argv(root: Path, unit: str, child_argv: Sequence[str]) -> list[str]:
    argv = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
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
    ]
    argv.extend(child_argv)
    return argv


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
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "detached systemd command could not be executed"
        ) from exc
    if not hasattr(completed, "returncode"):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error("run callable contract drifted")
    return completed


def _offline_environment_exact() -> bool:
    return all(os.environ.get(key) == value for key, value in SYSTEMD_ENVIRONMENT.items())


def _load_smoke_receipt_local(project_root: Path) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    payload, _ = _read_canonical_json(
        root / SMOKE_RECEIPT_RELATIVE_PATH,
        expected_mode=PUBLIC_MODE,
        field="v3 smoke receipt",
    )
    declared = _validate_self_hash(payload, "receipt_sha256", label="v3 smoke")
    if (
        payload.get("schema") != SMOKE_SCHEMA
        or payload.get("version") != VERSION
        or payload.get("status") not in {SMOKE_SUCCESS_STATUS, SMOKE_FAILURE_STATUS}
        or payload.get("design_sha256") != DESIGN_SHA256
        or not _is_lower_hex(declared, 64)
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error("v3 smoke receipt drifted")
    return payload


def load_committed_smoke_receipt(project_root: Path) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    committed = _load_committed_public_json(
        root, SMOKE_RECEIPT_RELATIVE_PATH, "committed v3 smoke receipt"
    )
    local = _load_smoke_receipt_local(root)
    marker, marker_file_sha256 = _load_smoke_marker(root)
    if (
        committed != local
        or local.get("status") != SMOKE_SUCCESS_STATUS
        or local.get("invocation_HEAD") != marker.get("actual_HEAD")
        or local.get("unit") != SMOKE_SYSTEMD_UNIT
        or not _is_lower_hex(local.get("systemd_invocation_id"), 32)
        or local.get("systemd_contract_sha256")
        != marker.get("systemd_contract_sha256")
        or local.get("bindings") != marker.get("bindings")
        or local.get("launch_marker_sha256") != marker.get("marker_sha256")
        or local.get("launch_marker_file_sha256") != marker_file_sha256
        or local.get("child_sleep_seconds") != 10
        or local.get("offline_environment_verified") is not True
        or local.get(
            "cohort_action_label_model_retrieval_score_or_estimand_opened"
        )
        is not False
        or local.get("formal_seed_authorized") is not True
        or local.get("attempts_allowed") != 1
        or local.get("retry_replacement_or_backup_attempt_authorized") is not False
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "committed successful v3 smoke receipt is unavailable"
        )
    return local


def _current_binding_rows(
    project_root: Path, paths: Sequence[str]
) -> list[dict[str, str]]:
    root = _require_canonical_root(project_root)
    rows: list[dict[str, str]] = []
    for relative_text in paths:
        relative = Path(relative_text)
        path = _assert_no_symlink_components(root / relative, "binding path")
        if not path.is_file() or path.is_symlink():
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                f"required binding is unavailable: {relative_text}"
            )
        current = path.read_bytes()
        committed = _committed_bytes(root, relative)
        if current != committed:
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                f"required binding is not current-HEAD clean: {relative_text}"
            )
        rows.append(
            {
                "relative_path": relative_text,
                "file_sha256": hashlib.sha256(committed).hexdigest(),
                "git_blob_sha1": _git_blob_sha1(committed),
            }
        )
    return rows


def _systemd_contract_sha256(root: Path) -> str:
    smoke_child = _systemd_child_argv(root, "smoke-child")
    formal_suffix = [
        "-u",
        "-m",
        "assumption_agent.benchmarks.synthetic_typed_graph_multiseed_lifecycle_v3",
        "formal-child",
        "--project-root",
        str(root),
        "--runtime-python",
        "BOUND_RUNTIME_PYTHON",
        "--local-llm-model",
        "BOUND_LOCAL_LLM_MODEL",
        "--local-embedding-model",
        "BOUND_LOCAL_EMBEDDING_MODEL",
    ]
    python = str(Path(sys.executable).resolve(strict=True))
    formal_child = [python, *formal_suffix]
    return semantic_hash(
        {
            "environment": SYSTEMD_ENVIRONMENT,
            "properties": SYSTEMD_PROPERTIES,
            "smoke_unit": SMOKE_SYSTEMD_UNIT,
            "formal_unit": FORMAL_SYSTEMD_UNIT,
            "smoke_argv": _systemd_run_argv(root, SMOKE_SYSTEMD_UNIT, smoke_child),
            "formal_argv_template": _systemd_run_argv(
                root, FORMAL_SYSTEMD_UNIT, formal_child
            ),
        }
    )


def _load_smoke_marker(project_root: Path) -> tuple[dict[str, Any], str]:
    root = _require_canonical_root(project_root)
    marker, file_sha256 = _read_canonical_json(
        root / SMOKE_LAUNCH_MARKER_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="v3 smoke launch marker",
    )
    marker_sha256 = _validate_self_hash(marker, "marker_sha256", label="v3 smoke marker")
    bindings = marker.get("bindings")
    if (
        marker.get("schema") != f"{SMOKE_SCHEMA}_launch_marker"
        or marker.get("version") != VERSION
        or marker.get("status") != "sole_detached_preseed_smoke_attempt_consumed"
        or marker.get("design_sha256") != DESIGN_SHA256
        or marker.get("unit") != SMOKE_SYSTEMD_UNIT
        or marker.get("attempt_count") != 1
        or marker.get("child_sleep_seconds") != 10
        or marker.get("systemd_contract_sha256") != _systemd_contract_sha256(root)
        or not isinstance(bindings, list)
        or [row.get("relative_path") if isinstance(row, Mapping) else None for row in bindings]
        != list(SMOKE_BINDING_PATHS)
        or not _is_lower_hex(marker_sha256, 64)
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error("v3 smoke marker drifted")
    return marker, file_sha256


def _smoke_receipt_body(
    *,
    marker: Mapping[str, Any],
    marker_file_sha256: str,
    status: str,
    failure_class: str | None = None,
    invocation_id: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": SMOKE_SCHEMA,
        "version": VERSION,
        "status": status,
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "invocation_HEAD": marker["actual_HEAD"],
        "unit": SMOKE_SYSTEMD_UNIT,
        "systemd_invocation_id": invocation_id,
        "systemd_contract_sha256": marker["systemd_contract_sha256"],
        "bindings": marker["bindings"],
        "launch_marker_sha256": marker["marker_sha256"],
        "launch_marker_file_sha256": marker_file_sha256,
        "child_sleep_seconds": 10,
        "offline_environment_verified": status == SMOKE_SUCCESS_STATUS,
        "cohort_action_label_model_retrieval_score_or_estimand_opened": False,
        "formal_seed_authorized": status == SMOKE_SUCCESS_STATUS,
        "attempts_allowed": 1,
        "retry_replacement_or_backup_attempt_authorized": False,
    }
    if failure_class is not None:
        body["failure_class"] = failure_class
    return _self_hashed(body, "receipt_sha256")


def run_systemd_smoke(
    project_root: Path,
    *,
    run: RunCallable = subprocess.run,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Durably consume and launch the fixed detached runtime smoke attempt."""

    root = _require_canonical_root(project_root)
    verify_frozen_design(root)
    _validate_v2_interruption_and_absence(root)
    if (root / IMPLEMENTATION_FREEZE_RELATIVE_PATH).exists():
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "smoke cannot be consumed after implementation freeze"
        )
    _assert_absent(
        root,
        (SMOKE_LAUNCH_MARKER_RELATIVE_PATH, SMOKE_RECEIPT_RELATIVE_PATH),
        "v3 smoke attempt output",
    )
    _assert_absent(root, _v3_formal_output_paths(), "v3 formal output before smoke")
    bindings = _current_binding_rows(root, SMOKE_BINDING_PATHS)
    child_argv = _systemd_child_argv(root, "smoke-child")
    launcher_argv = _systemd_run_argv(root, SMOKE_SYSTEMD_UNIT, child_argv)
    marker = _self_hashed(
        {
            "schema": f"{SMOKE_SCHEMA}_launch_marker",
            "version": VERSION,
            "status": "sole_detached_preseed_smoke_attempt_consumed",
            "actual_HEAD": _committed_head(root),
            "design_sha256": DESIGN_SHA256,
            "unit": SMOKE_SYSTEMD_UNIT,
            "systemd_contract_sha256": _systemd_contract_sha256(root),
            "launcher_argv_sha256": semantic_hash(launcher_argv),
            "bindings": bindings,
            "attempt_count": 1,
            "child_sleep_seconds": 10,
            "cohort_or_models_opened_before_marker": False,
        },
        "marker_sha256",
    )
    marker_file_sha256 = _write_json_exclusive(
        root / SMOKE_LAUNCH_MARKER_RELATIVE_PATH, marker, PRIVATE_MODE
    )
    completed = _call_run(run, launcher_argv, cwd=root)
    if completed.returncode != 0:
        failure = _smoke_receipt_body(
            marker=marker,
            marker_file_sha256=marker_file_sha256,
            status=SMOKE_FAILURE_STATUS,
            failure_class="SystemdRunLaunchFailure",
        )
        _write_json_exclusive(root / SMOKE_RECEIPT_RELATIVE_PATH, failure, PUBLIC_MODE)
        return failure
    monotonic_fn = time.monotonic if monotonic is None else monotonic
    sleep_fn = time.sleep if sleep is None else sleep
    deadline = monotonic_fn() + 60.0
    receipt_path = root / SMOKE_RECEIPT_RELATIVE_PATH
    while monotonic_fn() < deadline:
        if receipt_path.is_file() and not receipt_path.is_symlink():
            return _load_smoke_receipt_local(root)
        sleep_fn(0.25)
    failure = _smoke_receipt_body(
        marker=marker,
        marker_file_sha256=marker_file_sha256,
        status=SMOKE_FAILURE_STATUS,
        failure_class="DetachedSmokeReceiptTimeout",
    )
    if not receipt_path.exists() and not receipt_path.is_symlink():
        _write_json_exclusive(receipt_path, failure, PUBLIC_MODE)
    return _load_smoke_receipt_local(root)


def run_smoke_child(project_root: Path) -> dict[str, Any]:
    """Validate the detached/offline environment and persist the smoke receipt."""

    root = _require_canonical_root(project_root)
    marker, marker_file_sha256 = _load_smoke_marker(root)
    output = root / SMOKE_RECEIPT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "canonical v3 smoke receipt already exists"
        )
    try:
        if Path.cwd().resolve() != root:
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "smoke child working directory drifted"
            )
        if not _offline_environment_exact():
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "smoke child offline environment drifted"
            )
        invocation_id = os.environ.get("INVOCATION_ID")
        if not _is_lower_hex(invocation_id, 32):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "smoke child is not running in an attested systemd invocation"
            )
        if _current_binding_rows(root, SMOKE_BINDING_PATHS) != marker["bindings"]:
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "smoke child code/test tuples drifted after launch"
            )
        time.sleep(10)
        receipt = _smoke_receipt_body(
            marker=marker,
            marker_file_sha256=marker_file_sha256,
            status=SMOKE_SUCCESS_STATUS,
            invocation_id=str(invocation_id),
        )
    except BaseException as exc:
        receipt = _smoke_receipt_body(
            marker=marker,
            marker_file_sha256=marker_file_sha256,
            status=SMOKE_FAILURE_STATUS,
            failure_class=type(exc).__name__,
        )
    _write_json_exclusive(output, receipt, PUBLIC_MODE)
    return receipt


def create_implementation_freeze(project_root: Path) -> dict[str, Any]:
    """Freeze v3 lifecycle plus the exact unmodified v2 execution kernel."""

    root = _require_canonical_root(project_root)
    verify_frozen_design(root)
    smoke = load_committed_smoke_receipt(root)
    provenance = _validate_v2_interruption_and_absence(root)
    output = root / IMPLEMENTATION_FREEZE_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "v3 implementation freeze already exists"
        )
    _assert_absent(root, _v3_formal_output_paths(), "pre-freeze v3 formal output")
    binding_paths = (
        *SMOKE_BINDING_PATHS,
        *tuple(sorted(set(REQUIRED_FREEZE_PATHS).difference(SMOKE_BINDING_PATHS))),
    )
    bindings = _current_binding_rows(root, binding_paths)
    by_path = {row["relative_path"]: row for row in bindings}
    smoke_tuples = [by_path[path] for path in SMOKE_BINDING_PATHS]
    if smoke_tuples != smoke.get("bindings"):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "smoke-to-freeze code/test tuples drifted"
        )
    if (
        by_path["assumption_agent/benchmarks/synthetic_typed_graph_multiseed_acquisition_v2.py"]["file_sha256"]
        != V2_ACQUISITION_KERNEL_FILE_SHA256
        or by_path["assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v2.py"]["file_sha256"]
        != V2_RUNNER_KERNEL_FILE_SHA256
        or by_path["tests/test_synthetic_typed_graph_multiseed_acquisition_v2.py"]["file_sha256"]
        != V2_ACQUISITION_TEST_FILE_SHA256
        or by_path["tests/test_synthetic_typed_graph_multiseed_runner_v2.py"]["file_sha256"]
        != V2_RUNNER_TEST_FILE_SHA256
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "bound v2 execution kernel or test bytes drifted"
        )
    smoke_file_sha256 = sha256_file(root / SMOKE_RECEIPT_RELATIVE_PATH)
    body = {
        "schema": FREEZE_SCHEMA,
        "version": VERSION,
        "status": FREEZE_STATUS,
        "creation_HEAD": _committed_head(root),
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "smoke_receipt_sha256": smoke["receipt_sha256"],
        "smoke_receipt_file_sha256": smoke_file_sha256,
        "smoke_status": smoke["status"],
        "smoke_to_freeze_binding_equality": True,
        "smoke_binding_tuples": smoke_tuples,
        "v2_interruption_provenance": provenance,
        "v2_formal_outputs_absent": True,
        "systemd_contract_sha256": _systemd_contract_sha256(root),
        "formal_seed_or_cohort_exists": False,
        "bindings": bindings,
    }
    freeze = _self_hashed(body, "implementation_freeze_sha256")
    _write_json_exclusive(output, freeze, PUBLIC_MODE)
    return freeze


def verify_implementation_freeze(project_root: Path) -> tuple[dict[str, Any], str]:
    root = _require_canonical_root(project_root)
    verify_frozen_design(root)
    smoke = load_committed_smoke_receipt(root)
    provenance = _validate_v2_interruption_and_absence(root)
    freeze = _load_committed_public_json(
        root, IMPLEMENTATION_FREEZE_RELATIVE_PATH, "committed v3 implementation freeze"
    )
    declared = _validate_self_hash(
        freeze, "implementation_freeze_sha256", label="v3 implementation freeze"
    )
    rows = freeze.get("bindings")
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("version") != VERSION
        or freeze.get("status") != FREEZE_STATUS
        or freeze.get("design_sha256") != DESIGN_SHA256
        or freeze.get("design_file_sha256") != DESIGN_FILE_SHA256
        or freeze.get("smoke_receipt_sha256") != smoke.get("receipt_sha256")
        or freeze.get("smoke_receipt_file_sha256")
        != sha256_file(root / SMOKE_RECEIPT_RELATIVE_PATH)
        or freeze.get("smoke_status") != SMOKE_SUCCESS_STATUS
        or freeze.get("smoke_to_freeze_binding_equality") is not True
        or freeze.get("smoke_binding_tuples") != smoke.get("bindings")
        or freeze.get("v2_interruption_provenance") != provenance
        or freeze.get("v2_formal_outputs_absent") is not True
        or freeze.get("systemd_contract_sha256") != _systemd_contract_sha256(root)
        or freeze.get("formal_seed_or_cohort_exists") is not False
        or not _is_lower_hex(declared, 64)
        or not isinstance(rows, list)
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "committed v3 implementation freeze drifted"
        )
    binding_paths = (
        *SMOKE_BINDING_PATHS,
        *tuple(sorted(set(REQUIRED_FREEZE_PATHS).difference(SMOKE_BINDING_PATHS))),
    )
    expected = _current_binding_rows(root, binding_paths)
    if rows != expected:
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "v3 implementation bindings drifted after freeze"
        )
    by_path = {row["relative_path"]: row for row in expected}
    if [by_path[path] for path in SMOKE_BINDING_PATHS] != smoke.get("bindings"):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "committed smoke/freeze tuple equality drifted"
        )
    return freeze, _committed_head(root)


def create_seed_custody(project_root: Path) -> dict[str, Any]:
    """Consume one fresh ordered 8x32-byte seed batch after the committed freeze."""

    root = _require_canonical_root(project_root)
    freeze, actual_head = verify_implementation_freeze(root)
    _assert_absent(root, _v3_formal_output_paths(), "v3 seed attempt output")
    marker = _self_hashed(
        {
            "schema": f"{SEED_CUSTODY_SCHEMA}_attempt_marker",
            "version": VERSION,
            "status": "sole_v3_eight_seed_batch_generation_attempt_consumed",
            "actual_HEAD": actual_head,
            "design_sha256": DESIGN_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "attempt_count": 1,
            "entropy_call_bytes": SEED_BATCH_BYTES,
            "v2_formal_outputs_absent": True,
        },
        "marker_sha256",
    )
    marker_file_sha256 = _write_json_exclusive(
        root / SEED_MARKER_RELATIVE_PATH, marker, PRIVATE_MODE
    )
    try:
        batch = os.urandom(SEED_BATCH_BYTES)
        if type(batch) is not bytes or len(batch) != SEED_BATCH_BYTES:
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "OS random source did not return exactly 256 bytes"
            )
        seeds = tuple(
            batch[index * SEED_BYTES : (index + 1) * SEED_BYTES]
            for index in range(SEED_COUNT)
        )
        commitments = [hashlib.sha256(seed).hexdigest() for seed in seeds]
        forbidden = frozenset(
            {
                acquisition_v2.ORIGINAL_SEED_COMMITMENT_SHA256,
                *acquisition_v2.V1_ORDERED_SEED_COMMITMENTS,
            }
        )
        if len(set(seeds)) != SEED_COUNT or forbidden.intersection(commitments):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "fresh v3 seeds collide with each other, original, or v1"
            )
        batch_commitment = hashlib.sha256(batch).hexdigest()
        persisted = _write_exclusive(
            root / SEED_BATCH_RELATIVE_PATH, batch, PRIVATE_MODE
        )
        if persisted != batch_commitment:
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "v3 seed batch persistence drifted"
            )
        custody = _self_hashed(
            {
                "schema": SEED_CUSTODY_SCHEMA,
                "version": VERSION,
                "status": SEED_CUSTODY_STATUS,
                "design_sha256": DESIGN_SHA256,
                "design_file_sha256": DESIGN_FILE_SHA256,
                "implementation_freeze_sha256": freeze[
                    "implementation_freeze_sha256"
                ],
                "seed_attempt_marker_sha256": marker["marker_sha256"],
                "seed_attempt_marker_file_sha256": marker_file_sha256,
                "seed_count": SEED_COUNT,
                "seed_bytes_each": SEED_BYTES,
                "seed_batch_bytes": SEED_BATCH_BYTES,
                "seed_batch_commitment_sha256": batch_commitment,
                "ordered_seed_commitments_sha256": commitments,
                "original_seed_commitment_sha256": (
                    acquisition_v2.ORIGINAL_SEED_COMMITMENT_SHA256
                ),
                "v1_ordered_seed_commitments_sha256": list(
                    acquisition_v2.V1_ORDERED_SEED_COMMITMENTS
                ),
                "fresh_seeds_disjoint_from_original_and_v1": True,
                "v2_formal_seed_or_cohort_absent": True,
                "seed_generation": (
                    "one_os.urandom_256_call_then_ordered_8x32_split_after_marker"
                ),
                "seed_material_published": False,
                "cohort_generated": False,
                "attempts_allowed": 1,
                "retry_replacement_or_seed_count_change_authorized": False,
            },
            "custody_sha256",
        )
        _write_json_exclusive(
            root / SEED_CUSTODY_RELATIVE_PATH, custody, PUBLIC_MODE
        )
        return custody
    except BaseException as exc:
        failure = _self_hashed(
            {
                "schema": f"{SEED_CUSTODY_SCHEMA}_failure_receipt",
                "version": VERSION,
                "status": "terminal_v3_seed_batch_invalid_no_replacement",
                "design_sha256": DESIGN_SHA256,
                "implementation_freeze_sha256": freeze[
                    "implementation_freeze_sha256"
                ],
                "seed_attempt_marker_sha256": marker["marker_sha256"],
                "failure_class": type(exc).__name__,
                "secret_material_or_exception_message_persisted_publicly": False,
                "retry_replacement_or_seed_count_change_authorized": False,
            },
            "receipt_sha256",
        )
        failure_path = root / SEED_FAILURE_RELATIVE_PATH
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_json_exclusive(failure_path, failure, PUBLIC_MODE)
        raise


def load_committed_seed_custody(
    project_root: Path, *, verify_private_batch: bool = False
) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    freeze, _ = verify_implementation_freeze(root)
    custody = _load_committed_public_json(
        root, SEED_CUSTODY_RELATIVE_PATH, "committed v3 seed custody"
    )
    declared = _validate_self_hash(custody, "custody_sha256", label="v3 seed custody")
    commitments = custody.get("ordered_seed_commitments_sha256")
    forbidden = frozenset(
        {
            acquisition_v2.ORIGINAL_SEED_COMMITMENT_SHA256,
            *acquisition_v2.V1_ORDERED_SEED_COMMITMENTS,
        }
    )
    if (
        custody.get("schema") != SEED_CUSTODY_SCHEMA
        or custody.get("version") != VERSION
        or custody.get("status") != SEED_CUSTODY_STATUS
        or custody.get("design_sha256") != DESIGN_SHA256
        or custody.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or custody.get("seed_count") != SEED_COUNT
        or custody.get("seed_bytes_each") != SEED_BYTES
        or custody.get("seed_batch_bytes") != SEED_BATCH_BYTES
        or custody.get("fresh_seeds_disjoint_from_original_and_v1") is not True
        or custody.get("v2_formal_seed_or_cohort_absent") is not True
        or custody.get("seed_material_published") is not False
        or custody.get("cohort_generated") is not False
        or custody.get("attempts_allowed") != 1
        or custody.get("retry_replacement_or_seed_count_change_authorized") is not False
        or not isinstance(commitments, list)
        or len(commitments) != SEED_COUNT
        or len(set(commitments)) != SEED_COUNT
        or forbidden.intersection(commitments)
        or any(not _is_lower_hex(value, 64) for value in commitments)
        or not _is_lower_hex(declared, 64)
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "committed v3 seed custody drifted"
        )
    marker, marker_file_sha256 = _read_canonical_json(
        root / SEED_MARKER_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="v3 seed marker",
    )
    marker_sha256 = _validate_self_hash(marker, "marker_sha256", label="v3 seed marker")
    if (
        marker.get("schema") != f"{SEED_CUSTODY_SCHEMA}_attempt_marker"
        or marker.get("status")
        != "sole_v3_eight_seed_batch_generation_attempt_consumed"
        or marker.get("attempt_count") != 1
        or marker.get("entropy_call_bytes") != SEED_BATCH_BYTES
        or marker.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or custody.get("seed_attempt_marker_sha256") != marker_sha256
        or custody.get("seed_attempt_marker_file_sha256") != marker_file_sha256
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "v3 seed marker chain drifted"
        )
    if verify_private_batch:
        batch_path = root / SEED_BATCH_RELATIVE_PATH
        if (
            not batch_path.is_file()
            or batch_path.is_symlink()
            or batch_path.stat().st_size != SEED_BATCH_BYTES
            or stat.S_IMODE(batch_path.stat().st_mode) != PRIVATE_MODE
        ):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "v3 private seed batch mode or size drifted"
            )
        _read_seed_batch(batch_path, custody)
    return custody


def acquire_formal_cohort(project_root: Path) -> dict[str, Any]:
    """Generate one fresh 8x64 cohort and separated private kernel-wire packs."""

    root = _require_canonical_root(project_root)
    freeze, actual_head = verify_implementation_freeze(root)
    custody = load_committed_seed_custody(root)
    outputs = (
        ACQUISITION_MARKER_RELATIVE_PATH,
        ACTION_PACK_RELATIVE_PATH,
        LABEL_PACK_RELATIVE_PATH,
        COMPILED_COHORT_PACK_RELATIVE_PATH,
        ACQUISITION_RECEIPT_RELATIVE_PATH,
        FORMAL_LAUNCH_MARKER_RELATIVE_PATH,
        FORMAL_WORK_RELATIVE_PATH,
        FORMAL_ACTION_SEAL_RELATIVE_PATH,
        RESULT_RELATIVE_PATH,
        PUBLICATION_MARKER_RELATIVE_PATH,
        PUBLICATION_RELATIVE_PATH,
        PUBLICATION_FAILURE_RELATIVE_PATH,
    )
    _assert_absent(root, outputs, "v3 acquisition output")
    marker = _self_hashed(
        {
            "schema": f"{ACQUISITION_SCHEMA}_attempt_marker",
            "version": VERSION,
            "status": "sole_v3_cohort_generation_attempt_consumed",
            "actual_HEAD": actual_head,
            "design_sha256": DESIGN_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "custody_sha256": custody["custody_sha256"],
            "attempt_count": 1,
            "grammar_calls_authorized": SEED_COUNT,
            "block": BLOCK,
            "private_seed_opened_before_marker": False,
        },
        "marker_sha256",
    )
    marker_file_sha256 = _write_json_exclusive(
        root / ACQUISITION_MARKER_RELATIVE_PATH, marker, PRIVATE_MODE
    )
    try:
        seeds = _read_seed_batch(root / SEED_BATCH_RELATIVE_PATH, custody)
        prior_commitments = _load_prior_item_commitments_after_marker(root)
        compiled: list[tuple[int, grammar.CompiledItem]] = []
        for seed_index, seed in enumerate(seeds):
            items = grammar.generate_block(seed, BLOCK)
            if len(items) != ITEMS_PER_SEED:
                raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                    "grammar did not return exactly 64 A_hold rows"
                )
            for seed_ordinal, item in enumerate(items):
                acquisition_v2._validate_compiled_item(item, seed_ordinal)
                compiled.append((seed_index, item))
        if len(compiled) != TOTAL_ITEM_COUNT:
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "fresh v3 compiled cohort count drifted"
            )
        item_commitments = [item.item_commitment_sha256 for _, item in compiled]
        if len(set(item_commitments)) != TOTAL_ITEM_COUNT:
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "fresh v3 cohort contains duplicate item commitments"
            )
        if prior_commitments.intersection(item_commitments):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "fresh v3 cohort overlaps original or v1"
            )
        action_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        compiled_rows: list[dict[str, Any]] = []
        for global_ordinal, (seed_index, item) in enumerate(compiled):
            action = acquisition_v2._action_row(
                item, seed_index=seed_index, global_ordinal=global_ordinal
            )
            label = acquisition_v2._label_row(
                item,
                action_item_sha256=action["action_item_sha256"],
                seed_index=seed_index,
                global_ordinal=global_ordinal,
            )
            compiled_body = acquisition_v2._compiled_public_row(
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
        action_pack = acquisition_v2._pack(
            schema=acquisition_v2.ACTION_PACK_SCHEMA,
            items=action_rows,
            labels_included=False,
        )
        label_pack = acquisition_v2._pack(
            schema=acquisition_v2.LABEL_PACK_SCHEMA,
            items=label_rows,
            labels_included=None,
        )
        compiled_pack = acquisition_v2._pack(
            schema=acquisition_v2.COMPILED_COHORT_PACK_SCHEMA,
            items=compiled_rows,
            labels_included=True,
        )
        action_file_sha256 = _write_json_exclusive(
            root / ACTION_PACK_RELATIVE_PATH, action_pack, PRIVATE_MODE
        )
        label_file_sha256 = _write_json_exclusive(
            root / LABEL_PACK_RELATIVE_PATH, label_pack, PRIVATE_MODE
        )
        compiled_file_sha256 = _write_json_exclusive(
            root / COMPILED_COHORT_PACK_RELATIVE_PATH,
            compiled_pack,
            PRIVATE_MODE,
        )
        commitments = {
            "action_pack_file_sha256": action_file_sha256,
            "action_item_commitment_set_sha256": stable_hash(
                [row["action_item_sha256"] for row in action_rows]
            ),
            "label_pack_file_sha256": label_file_sha256,
            "label_item_commitment_set_sha256": stable_hash(
                [row["label_item_sha256"] for row in label_rows]
            ),
            "compiled_cohort_pack_file_sha256": compiled_file_sha256,
            "compiled_row_commitment_set_sha256": stable_hash(
                [row["compiled_row_sha256"] for row in compiled_rows]
            ),
        }
        receipt = _self_hashed(
            {
                "schema": ACQUISITION_SCHEMA,
                "version": VERSION,
                "status": ACQUISITION_STATUS,
                "design_sha256": DESIGN_SHA256,
                "design_file_sha256": DESIGN_FILE_SHA256,
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
                "attempt_marker_file_sha256": marker_file_sha256,
                "block": BLOCK,
                "seed_count": SEED_COUNT,
                "item_count_per_seed": ITEMS_PER_SEED,
                "total_item_count": TOTAL_ITEM_COUNT,
                "grammar_generate_block_call_count": SEED_COUNT,
                "generated_item_commitment_set_sha256": stable_hash(
                    item_commitments
                ),
                "original_A_hold_commitment_set_sha256": (
                    acquisition_v2.ORIGINAL_A_HOLD_COMMITMENT_SET_SHA256
                ),
                "v1_multiseed_item_commitment_set_sha256": (
                    acquisition_v2.V1_ITEM_COMMITMENT_SET_SHA256
                ),
                "new_original_and_v1_item_commitments_pairwise_disjoint": True,
                "v2_formal_seed_or_cohort_absent": True,
                "fixed_recipe_id": kernel_v2.RECIPE_ID,
                "arms": list(kernel_v2.ARM_IDS),
                "wire_format_version": acquisition_v2.VERSION,
                "wire_format_is_bound_v2_execution_kernel_only": True,
                "commitments": commitments,
                "packs_mode": "0600",
                "label_rows_persisted_publicly": False,
                "seed_material_published": False,
                "formation_candidate_pool_filter_or_recipe_search_used": False,
                "network_calls": 0,
                "retry_replacement_or_smaller_cohort_authorized": False,
            },
            "receipt_sha256",
        )
        _write_json_exclusive(
            root / ACQUISITION_RECEIPT_RELATIVE_PATH, receipt, PUBLIC_MODE
        )
        return receipt
    except BaseException as exc:
        failure = _self_hashed(
            {
                "schema": ACQUISITION_SCHEMA,
                "version": VERSION,
                "status": "terminal_v3_multiseed_acquisition_invalid_no_replay",
                "design_sha256": DESIGN_SHA256,
                "implementation_freeze_sha256": freeze[
                    "implementation_freeze_sha256"
                ],
                "custody_sha256": custody["custody_sha256"],
                "attempt_marker_sha256": marker["marker_sha256"],
                "failure_class": type(exc).__name__,
                "exception_message_seed_or_item_rows_persisted_publicly": False,
                "retry_replacement_or_smaller_cohort_authorized": False,
            },
            "receipt_sha256",
        )
        receipt_path = root / ACQUISITION_RECEIPT_RELATIVE_PATH
        if not receipt_path.exists() and not receipt_path.is_symlink():
            _write_json_exclusive(receipt_path, failure, PUBLIC_MODE)
        raise


def _verify_compiled_cohort_pack(
    project_root: Path,
    *,
    expected_file_sha256: object,
    expected_row_set_sha256: object,
    expected_item_set_sha256: object,
) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    pack, file_sha256 = _read_canonical_json(
        root / COMPILED_COHORT_PACK_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="v3 stored full compiled cohort pack",
    )
    declared = _validate_self_hash(pack, "pack_sha256", label="compiled cohort pack")
    rows = pack.get("items")
    if (
        pack.get("schema") != acquisition_v2.COMPILED_COHORT_PACK_SCHEMA
        or pack.get("version") != acquisition_v2.VERSION
        or pack.get("block") != BLOCK
        or pack.get("seed_count") != SEED_COUNT
        or pack.get("item_count_per_seed") != ITEMS_PER_SEED
        or pack.get("total_item_count") != TOTAL_ITEM_COUNT
        or pack.get("labels_included") is not True
        or file_sha256 != expected_file_sha256
        or not _is_lower_hex(declared, 64)
        or not isinstance(rows, list)
        or len(rows) != TOTAL_ITEM_COUNT
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "v3 stored compiled pack binding drifted"
        )
    row_hashes: list[str] = []
    item_hashes: list[str] = []
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "v3 stored compiled row drifted"
            )
        row = dict(raw)
        row_hash = row.pop("compiled_row_sha256", None)
        expected_seed, expected_within = divmod(ordinal, ITEMS_PER_SEED)
        item_hash = raw.get("item_commitment_sha256")
        if (
            set(row) != acquisition_v2._COMPILED_PUBLIC_ROW_FIELDS
            or raw.get("global_ordinal") != ordinal
            or raw.get("seed_index") != expected_seed
            or raw.get("seed_ordinal") != expected_within
            or raw.get("block") != BLOCK
            or raw.get("block_ordinal") != expected_within
            or not _is_lower_hex(row_hash, 64)
            or semantic_hash(row) != row_hash
            or not _is_lower_hex(item_hash, 64)
        ):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "v3 stored compiled row binding drifted"
            )
        row_hashes.append(str(row_hash))
        item_hashes.append(str(item_hash))
    if (
        len(set(row_hashes)) != TOTAL_ITEM_COUNT
        or len(set(item_hashes)) != TOTAL_ITEM_COUNT
        or stable_hash(row_hashes) != expected_row_set_sha256
        or stable_hash(item_hashes) != expected_item_set_sha256
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "v3 compiled cohort commitment set drifted"
        )
    return pack


def load_committed_acquisition_receipt(
    project_root: Path, *, verify_private_packs: bool = True
) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    freeze, _ = verify_implementation_freeze(root)
    custody = load_committed_seed_custody(
        root, verify_private_batch=verify_private_packs
    )
    receipt = _load_committed_public_json(
        root, ACQUISITION_RECEIPT_RELATIVE_PATH, "committed v3 acquisition receipt"
    )
    declared = _validate_self_hash(receipt, "receipt_sha256", label="v3 acquisition")
    commitments = receipt.get("commitments")
    required_commitment_fields = {
        "action_pack_file_sha256",
        "action_item_commitment_set_sha256",
        "label_pack_file_sha256",
        "label_item_commitment_set_sha256",
        "compiled_cohort_pack_file_sha256",
        "compiled_row_commitment_set_sha256",
    }
    if (
        receipt.get("schema") != ACQUISITION_SCHEMA
        or receipt.get("version") != VERSION
        or receipt.get("status") != ACQUISITION_STATUS
        or receipt.get("design_sha256") != DESIGN_SHA256
        or receipt.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or receipt.get("custody_sha256") != custody.get("custody_sha256")
        or receipt.get("block") != BLOCK
        or receipt.get("seed_count") != SEED_COUNT
        or receipt.get("item_count_per_seed") != ITEMS_PER_SEED
        or receipt.get("total_item_count") != TOTAL_ITEM_COUNT
        or receipt.get("grammar_generate_block_call_count") != SEED_COUNT
        or receipt.get("new_original_and_v1_item_commitments_pairwise_disjoint")
        is not True
        or receipt.get("v2_formal_seed_or_cohort_absent") is not True
        or receipt.get("fixed_recipe_id") != kernel_v2.RECIPE_ID
        or receipt.get("arms") != list(kernel_v2.ARM_IDS)
        or receipt.get("wire_format_version") != acquisition_v2.VERSION
        or receipt.get("wire_format_is_bound_v2_execution_kernel_only") is not True
        or not isinstance(commitments, Mapping)
        or set(commitments) != required_commitment_fields
        or any(not _is_lower_hex(value, 64) for value in commitments.values())
        or not _is_lower_hex(receipt.get("generated_item_commitment_set_sha256"), 64)
        or not _is_lower_hex(declared, 64)
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "committed v3 acquisition receipt drifted"
        )
    if verify_private_packs:
        acquisition_v2._verify_pack(
            root,
            relative_path=ACTION_PACK_RELATIVE_PATH,
            schema=acquisition_v2.ACTION_PACK_SCHEMA,
            item_hash_field="action_item_sha256",
            expected_file_hash=commitments["action_pack_file_sha256"],
            expected_set_hash=commitments[
                "action_item_commitment_set_sha256"
            ],
        )
        acquisition_v2._verify_pack(
            root,
            relative_path=LABEL_PACK_RELATIVE_PATH,
            schema=acquisition_v2.LABEL_PACK_SCHEMA,
            item_hash_field="label_item_sha256",
            expected_file_hash=commitments["label_pack_file_sha256"],
            expected_set_hash=commitments["label_item_commitment_set_sha256"],
        )
        _verify_compiled_cohort_pack(
            root,
            expected_file_sha256=commitments[
                "compiled_cohort_pack_file_sha256"
            ],
            expected_row_set_sha256=commitments[
                "compiled_row_commitment_set_sha256"
            ],
            expected_item_set_sha256=receipt[
                "generated_item_commitment_set_sha256"
            ],
        )
    return receipt


def launch_formal(
    project_root: Path,
    *,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    run: RunCallable = subprocess.run,
) -> dict[str, Any]:
    """Durably consume the formal attempt before launching its fixed systemd unit."""

    root = _require_canonical_root(project_root)
    freeze, actual_head = verify_implementation_freeze(root)
    acquisition = load_committed_acquisition_receipt(
        root, verify_private_packs=False
    )
    paths = (
        FORMAL_LAUNCH_MARKER_RELATIVE_PATH,
        FORMAL_WORK_RELATIVE_PATH,
        FORMAL_ACTION_SEAL_RELATIVE_PATH,
        RESULT_RELATIVE_PATH,
        PUBLICATION_MARKER_RELATIVE_PATH,
        PUBLICATION_RELATIVE_PATH,
        PUBLICATION_FAILURE_RELATIVE_PATH,
    )
    _assert_absent(root, paths, "v3 formal attempt output")
    resolved_runtime_python = runtime_python.resolve(strict=True)
    resolved_llm = local_llm_model.resolve(strict=True)
    resolved_embedding = local_embedding_model.resolve(strict=True)
    child_argv = _systemd_child_argv(
        root,
        "formal-child",
        runtime_python=resolved_runtime_python,
        local_llm_model=resolved_llm,
        local_embedding_model=resolved_embedding,
    )
    launcher_argv = _systemd_run_argv(root, FORMAL_SYSTEMD_UNIT, child_argv)
    acquisition_file_sha256 = sha256_file(
        root / ACQUISITION_RECEIPT_RELATIVE_PATH
    )
    marker = _self_hashed(
        {
            "schema": f"{RESULT_SCHEMA}_formal_launch_marker",
            "version": VERSION,
            "status": "sole_detached_v3_formal_attempt_consumed",
            "actual_HEAD": actual_head,
            "design_sha256": DESIGN_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "acquisition_receipt_sha256": acquisition["receipt_sha256"],
            "acquisition_receipt_file_sha256": acquisition_file_sha256,
            "unit": FORMAL_SYSTEMD_UNIT,
            "systemd_contract_sha256": _systemd_contract_sha256(root),
            "launcher_argv_sha256": semantic_hash(launcher_argv),
            "runtime_python": str(resolved_runtime_python),
            "local_llm_model": str(resolved_llm),
            "local_embedding_model": str(resolved_embedding),
            "attempt_count": 1,
            "private_packs_opened_before_marker": False,
            "labels_opened_before_marker": False,
            "relaunch_authorized": False,
        },
        "marker_sha256",
    )
    marker_file_sha256 = _write_json_exclusive(
        root / FORMAL_LAUNCH_MARKER_RELATIVE_PATH, marker, PRIVATE_MODE
    )
    completed = _call_run(run, launcher_argv, cwd=root)
    if completed.returncode != 0:
        return _persist_formal_failure(
            project_root=root,
            marker=marker,
            marker_file_sha256=marker_file_sha256,
            freeze=freeze,
            acquisition=acquisition,
            acquisition_file_sha256=acquisition_file_sha256,
            failure_class="SystemdRunLaunchFailure",
            administrative=True,
            systemd_state={"launcher_returncode": int(completed.returncode)},
        )
    return marker


def _load_formal_marker(project_root: Path) -> tuple[dict[str, Any], str]:
    root = _require_canonical_root(project_root)
    marker, file_sha256 = _read_canonical_json(
        root / FORMAL_LAUNCH_MARKER_RELATIVE_PATH,
        expected_mode=PRIVATE_MODE,
        field="v3 formal launch marker",
    )
    _validate_self_hash(marker, "marker_sha256", label="v3 formal marker")
    if (
        marker.get("schema") != f"{RESULT_SCHEMA}_formal_launch_marker"
        or marker.get("version") != VERSION
        or marker.get("status") != "sole_detached_v3_formal_attempt_consumed"
        or marker.get("design_sha256") != DESIGN_SHA256
        or marker.get("unit") != FORMAL_SYSTEMD_UNIT
        or marker.get("systemd_contract_sha256")
        != _systemd_contract_sha256(root)
        or marker.get("attempt_count") != 1
        or marker.get("private_packs_opened_before_marker") is not False
        or marker.get("labels_opened_before_marker") is not False
        or marker.get("relaunch_authorized") is not False
        or any(
            not isinstance(marker.get(field), str) or not marker.get(field)
            for field in (
                "runtime_python",
                "local_llm_model",
                "local_embedding_model",
            )
        )
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "v3 formal launch marker drifted"
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
    systemd_state: Mapping[str, Any] | None,
    action_seal_file_sha256: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "version": VERSION,
        "status": FAILURE_RESULT_STATUS,
        "invocation_HEAD": marker["actual_HEAD"],
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "implementation_freeze_sha256": freeze[
            "implementation_freeze_sha256"
        ],
        "acquisition_receipt_sha256": acquisition["receipt_sha256"],
        "acquisition_receipt_file_sha256": acquisition_file_sha256,
        "generated_item_commitment_set_sha256": acquisition[
            "generated_item_commitment_set_sha256"
        ],
        "formal_launch_marker_sha256": marker["marker_sha256"],
        "formal_launch_marker_file_sha256": marker_file_sha256,
        "systemd_unit": FORMAL_SYSTEMD_UNIT,
        "systemd_invocation_id": (
            os.environ.get("INVOCATION_ID")
            if _is_lower_hex(os.environ.get("INVOCATION_ID"), 32)
            else None
        ),
        "systemd_contract_sha256": marker["systemd_contract_sha256"],
        "action_seal_file_sha256": action_seal_file_sha256,
        "failure_class": failure_class,
        "administrative_finalization_without_private_pack_or_label_open": administrative,
        "systemd_state": dict(systemd_state) if systemd_state is not None else None,
        "retry_replacement_or_relaunch_authorized": False,
        "exception_message_seed_item_or_label_content_persisted_publicly": False,
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
        systemd_state=systemd_state,
        action_seal_file_sha256=seal_file_sha256,
    )
    path = root / RESULT_RELATIVE_PATH
    if not path.exists() and not path.is_symlink():
        _write_json_exclusive(path, failure, PUBLIC_MODE)
    return _load_terminal_result_local(root)


def run_formal_child(
    project_root: Path,
    *,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> dict[str, Any]:
    """Run the frozen v2 kernel on v3 paths and persist one canonical result."""

    root = _require_canonical_root(project_root)
    marker, marker_file_sha256 = _load_formal_marker(root)
    freeze, _ = verify_implementation_freeze(root)
    acquisition = load_committed_acquisition_receipt(
        root, verify_private_packs=False
    )
    acquisition_file_sha256 = sha256_file(
        root / ACQUISITION_RECEIPT_RELATIVE_PATH
    )
    result_path = root / RESULT_RELATIVE_PATH
    if result_path.exists() or result_path.is_symlink():
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "canonical v3 formal result already exists"
        )
    try:
        resolved = (
            runtime_python.resolve(strict=True),
            local_llm_model.resolve(strict=True),
            local_embedding_model.resolve(strict=True),
        )
        if tuple(marker[field] for field in (
            "runtime_python", "local_llm_model", "local_embedding_model"
        )) != tuple(str(path) for path in resolved):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "formal child runtime arguments differ from launch marker"
            )
        if Path.cwd().resolve() != root or not _offline_environment_exact():
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "formal child detached offline environment drifted"
            )
        invocation_id = os.environ.get("INVOCATION_ID")
        if not _is_lower_hex(invocation_id, 32):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "formal child is not running in an attested systemd invocation"
            )
        if (
            marker.get("implementation_freeze_sha256")
            != freeze.get("implementation_freeze_sha256")
            or marker.get("acquisition_receipt_sha256")
            != acquisition.get("receipt_sha256")
            or marker.get("acquisition_receipt_file_sha256")
            != acquisition_file_sha256
        ):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "formal launch predecessor chain drifted"
            )
        encoder, runtime = kernel_v2._prepare_formal_resources(
            project_root=root,
            runtime_python=resolved[0],
            local_llm_model=resolved[1],
            local_embedding_model=resolved[2],
        )
        if not isinstance(encoder, OfflineMiniLMEncoder) or not isinstance(
            runtime, PreparedFormalRuntimeV2
        ):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "formal resources are not the attested frozen types"
            )
        commitments = acquisition.get("commitments")
        if not isinstance(commitments, Mapping):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "formal acquisition commitments drifted"
            )
        action_pack = kernel_v2.load_action_pack(root / ACTION_PACK_RELATIVE_PATH)
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
                raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                    "late labels opened more than once"
                )
            labels = kernel_v2.load_label_pack(root / LABEL_PACK_RELATIVE_PATH)
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
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "late labels were not opened exactly once"
            )
        kernel_receipt = kernel_v2.multiseed_public_result(outcome)
        body = {
            "schema": RESULT_SCHEMA,
            "version": VERSION,
            "status": SUCCESS_RESULT_STATUS,
            "invocation_HEAD": marker["actual_HEAD"],
            "design_sha256": DESIGN_SHA256,
            "design_file_sha256": DESIGN_FILE_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
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
            "wire_format_version": acquisition_v2.VERSION,
            "execution_kernel_version": kernel_v2.VERSION,
            "execution_kernel_receipt": kernel_receipt,
            "interpretation": "descriptive_fixed_fresh_v3_cohort_replication_only",
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
        )


def _load_terminal_result_local(project_root: Path) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    result, _ = _read_canonical_json(
        root / RESULT_RELATIVE_PATH,
        expected_mode=PUBLIC_MODE,
        field="v3 terminal result",
    )
    _validate_self_hash(result, "receipt_sha256", label="v3 terminal result")
    marker, marker_file_sha256 = _load_formal_marker(root)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("version") != VERSION
        or result.get("status") not in {SUCCESS_RESULT_STATUS, FAILURE_RESULT_STATUS}
        or result.get("design_sha256") != DESIGN_SHA256
        or result.get("design_file_sha256") != DESIGN_FILE_SHA256
        or result.get("formal_launch_marker_sha256") != marker.get("marker_sha256")
        or result.get("formal_launch_marker_file_sha256") != marker_file_sha256
        or result.get("systemd_unit") != FORMAL_SYSTEMD_UNIT
        or result.get("systemd_contract_sha256")
        != marker.get("systemd_contract_sha256")
        or result.get("retry_replacement_or_relaunch_authorized") is not False
        or result.get("result_must_be_committed_before_terminal_publication")
        is not True
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "v3 terminal result chain drifted"
        )
    if result.get("status") == SUCCESS_RESULT_STATUS:
        kernel_receipt = result.get("execution_kernel_receipt")
        if (
            result.get("wire_format_version") != acquisition_v2.VERSION
            or result.get("execution_kernel_version") != kernel_v2.VERSION
            or not _is_lower_hex(result.get("systemd_invocation_id"), 32)
            or not isinstance(kernel_receipt, Mapping)
            or kernel_receipt.get("status") != kernel_v2.SUCCESS_RESULT_STATUS
            or _validate_self_hash(
                kernel_receipt, "receipt_sha256", label="v2 kernel receipt"
            )
            != kernel_receipt.get("receipt_sha256")
            or result.get("performance_gate_or_promotion_computed") is not False
            or result.get("seeds_or_item_rows_disclosed") is not False
        ):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "successful v3 kernel receipt drifted"
            )
    else:
        if (
            not isinstance(result.get("failure_class"), str)
            or not result.get("failure_class")
            or result.get(
                "exception_message_seed_item_or_label_content_persisted_publicly"
            )
            is not False
        ):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "failed v3 terminal result drifted"
            )
    return result


def load_committed_terminal_result(project_root: Path) -> dict[str, Any]:
    root = _require_canonical_root(project_root)
    committed = _load_committed_public_json(
        root, RESULT_RELATIVE_PATH, "committed v3 terminal result"
    )
    local = _load_terminal_result_local(root)
    if committed != local:
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "committed v3 terminal result readback drifted"
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
)


def _read_systemd_state(
    project_root: Path, *, unit: str, run: RunCallable
) -> tuple[dict[str, str], int]:
    root = _require_canonical_root(project_root)
    argv = [
        "systemctl",
        "--user",
        "show",
        f"{unit}.service",
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
    """Return a child result or close a terminal service-without-result as failure."""

    root = _require_canonical_root(project_root)
    result_path = root / RESULT_RELATIVE_PATH
    if result_path.is_file() and not result_path.is_symlink():
        return _load_terminal_result_local(root)
    marker, marker_file_sha256 = _load_formal_marker(root)
    freeze, _ = verify_implementation_freeze(root)
    acquisition = load_committed_acquisition_receipt(
        root, verify_private_packs=False
    )
    acquisition_file_sha256 = sha256_file(
        root / ACQUISITION_RECEIPT_RELATIVE_PATH
    )
    state, returncode = _read_systemd_state(
        root, unit=FORMAL_SYSTEMD_UNIT, run=run
    )
    active = state.get("ActiveState", "")
    sub = state.get("SubState", "")
    if active in {"activating", "reloading"} or (
        active == "active" and sub not in {"exited", "dead", "failed"}
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "detached v3 formal service is still running"
        )
    failure_class = (
        "SystemdStateUnavailableAfterConsumedFormalMarker"
        if returncode != 0 and not state
        else "DetachedSystemdServiceTerminalWithoutCanonicalResult"
    )
    safe_state: dict[str, Any] = {
        key: value
        for key, value in state.items()
        if key != "ControlGroup"
    }
    safe_state["systemctl_returncode"] = returncode
    return _persist_formal_failure(
        project_root=root,
        marker=marker,
        marker_file_sha256=marker_file_sha256,
        freeze=freeze,
        acquisition=acquisition,
        acquisition_file_sha256=acquisition_file_sha256,
        failure_class=failure_class,
        administrative=True,
        systemd_state=safe_state,
    )


def publish_terminal(project_root: Path) -> dict[str, Any]:
    """Publish stored seeds/cohort only after a committed terminal v3 result."""

    root = _require_canonical_root(project_root)
    freeze, actual_head = verify_implementation_freeze(root)
    acquisition = load_committed_acquisition_receipt(
        root, verify_private_packs=False
    )
    custody = load_committed_seed_custody(root)
    result = load_committed_terminal_result(root)
    if (
        result.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or result.get("acquisition_receipt_sha256")
        != acquisition.get("receipt_sha256")
        or result.get("generated_item_commitment_set_sha256")
        != acquisition.get("generated_item_commitment_set_sha256")
    ):
        raise SyntheticTypedGraphMultiseedLifecycleV3Error(
            "terminal publication predecessor chain drifted"
        )
    outputs = (
        PUBLICATION_MARKER_RELATIVE_PATH,
        PUBLICATION_RELATIVE_PATH,
        PUBLICATION_FAILURE_RELATIVE_PATH,
    )
    _assert_absent(root, outputs, "v3 terminal publication output")
    result_file_sha256 = sha256_file(root / RESULT_RELATIVE_PATH)
    marker = _self_hashed(
        {
            "schema": f"{PUBLICATION_SCHEMA}_attempt_marker",
            "version": VERSION,
            "status": "sole_v3_terminal_reproducibility_publication_attempt_consumed",
            "actual_HEAD": actual_head,
            "design_sha256": DESIGN_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "terminal_result_receipt_sha256": result["receipt_sha256"],
            "terminal_result_file_sha256": result_file_sha256,
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
        seeds = _read_seed_batch(root / SEED_BATCH_RELATIVE_PATH, custody)
        commitments = acquisition.get("commitments")
        if not isinstance(commitments, Mapping):
            raise SyntheticTypedGraphMultiseedLifecycleV3Error(
                "publication acquisition commitments drifted"
            )
        compiled_pack = _verify_compiled_cohort_pack(
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
                "version": VERSION,
                "status": "terminal_v3_eight_seeds_and_full_compiled_cohort_published",
                "publication_HEAD": actual_head,
                "design_sha256": DESIGN_SHA256,
                "design_file_sha256": DESIGN_FILE_SHA256,
                "implementation_freeze_sha256": freeze[
                    "implementation_freeze_sha256"
                ],
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
                "block": BLOCK,
                "seed_count": SEED_COUNT,
                "item_count_per_seed": ITEMS_PER_SEED,
                "total_item_count": TOTAL_ITEM_COUNT,
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
                "version": VERSION,
                "status": "terminal_v3_publication_invalid_no_replay",
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
    simple = (
        "launch-smoke",
        "smoke-child",
        "freeze",
        "seed-custody",
        "acquire",
        "finalize-formal",
        "publish-terminal",
    )
    for command in simple:
        child = subparsers.add_parser(command)
        child.add_argument("--project-root", required=True, type=Path)
    for command in ("launch-formal", "formal-child"):
        child = subparsers.add_parser(command)
        child.add_argument("--project-root", required=True, type=Path)
        child.add_argument("--runtime-python", required=True, type=Path)
        child.add_argument("--local-llm-model", required=True, type=Path)
        child.add_argument("--local-embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "launch-smoke":
        result = run_systemd_smoke(arguments.project_root)
    elif arguments.command == "smoke-child":
        result = run_smoke_child(arguments.project_root)
    elif arguments.command == "freeze":
        result = create_implementation_freeze(arguments.project_root)
    elif arguments.command == "seed-custody":
        result = create_seed_custody(arguments.project_root)
    elif arguments.command == "acquire":
        result = acquire_formal_cohort(arguments.project_root)
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
    result_hash = next(
        (
            result[field]
            for field in (
                "receipt_sha256",
                "implementation_freeze_sha256",
                "custody_sha256",
                "marker_sha256",
                "reproducibility_sha256",
            )
            if isinstance(result.get(field), str)
        ),
        semantic_hash(result),
    )
    print(
        json.dumps(
            {
                "command": arguments.command,
                "status": result.get("status"),
                "result_sha256": result_hash,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
