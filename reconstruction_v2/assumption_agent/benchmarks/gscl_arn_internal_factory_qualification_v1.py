"""One fixed, source-free execution qualification for the GSCL ARN core.

This is not an effect study and cannot open the official ARN source or any
label.  It owns one small public synthetic fixture, executes the exact
Qwen-to-MiniLM internal factory, seals the four-arm pre-label barrier, and
publishes only a content-free aggregate terminal.  The caller may select only
the already-frozen local asset paths and one fresh ``/var/tmp`` evidence root;
it cannot supply source rows, predictions, labels, scores, or callbacks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePath
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import (
    gscl_arn_formal_supervisor_v1 as supervisor,
)
from assumption_agent.benchmarks import (
    gscl_arn_intrinsic_protocol_v1 as protocol,
)
from assumption_agent.gscl_arn_raw_adapter_v1 import ArnTopology


VERSION = "gscl_arn_internal_factory_qualification_v1"
SAFE_TERMINAL_SCHEMA = f"{VERSION}.safe_terminal.v1"
DEFERRED_EXIT_CODE = 75
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
_EXPECTED_GPU_INDICES = (0, 1)
FROZEN_DEPLOYMENT_ROOT = Path(
    "/var/tmp/gscl_unified_nonscoring_harness_20260730"
)
FROZEN_WORKSPACE_CODE_ROOT = FROZEN_DEPLOYMENT_ROOT / "code"
FROZEN_CODE_ROOT = FROZEN_DEPLOYMENT_ROOT / "code/reconstruction_v2"
FROZEN_RUNTIME_ROOT = (
    FROZEN_DEPLOYMENT_ROOT / "assets/gscl_runtime_ext4_v1"
)
FROZEN_MAIN_RUNTIME_ROOT = FROZEN_RUNTIME_ROOT / "typed_venv"
FROZEN_MAIN_PYTHON = FROZEN_MAIN_RUNTIME_ROOT / "bin/python"
FROZEN_MAIN_PYTHON_SHA256 = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
FROZEN_MAIN_PYVENV_CONFIG = (
    FROZEN_MAIN_RUNTIME_ROOT / "pyvenv.cfg"
)
FROZEN_MAIN_PYVENV_CONFIG_SHA256 = (
    "da6c0ab165bd098b86649d2af4da536e7c91ee921c20b4f03d5d631f7172a503"
)
FROZEN_MAIN_BASE_PREFIX = FROZEN_RUNTIME_ROOT / "python310"
FROZEN_RUNTIME_BINDING_MANIFEST = (
    FROZEN_CODE_ROOT
    / "manifests/gscl_ext4_runtime_binding_20260730.json"
)
FROZEN_RUNTIME_BINDING_FILE_SHA256 = (
    "4ee4c6be40af92b0c24e540735621576502e6d5a097bf265d90071386405f1a3"
)
FROZEN_RUNTIME_BINDING_SELF_SHA256 = (
    "8929d1d96581373b1c2a13c1c2330fceb56c26283eab415534c2e3543217c356"
)
FROZEN_TEST_VENV_ROOT = FROZEN_RUNTIME_ROOT / "test_venv"
FROZEN_TEST_PYTHON = FROZEN_TEST_VENV_ROOT / "bin/python"
FROZEN_TEST_PYTHON_LINK_TARGET = str(FROZEN_MAIN_PYTHON)
FROZEN_TEST_PYTHON_RESOLVED = Path(
    FROZEN_TEST_PYTHON_LINK_TARGET
)
FROZEN_TEST_PYTHON_RESOLVED_SHA256 = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
FROZEN_TEST_PYVENV_CONFIG = FROZEN_TEST_VENV_ROOT / "pyvenv.cfg"
FROZEN_TEST_PYVENV_CONFIG_SHA256 = (
    "71f98c0e02f9dbb439cbda7b1ffc40999bd963123abf4934a1266988cacb71a0"
)
FROZEN_PYTEST_WHEEL_BUNDLE_MANIFEST = (
    FROZEN_DEPLOYMENT_ROOT
    / "assets/pytest_wheels/bundle_manifest.json"
)
FROZEN_PARENT_SITE = Path(
    FROZEN_MAIN_RUNTIME_ROOT / "lib/python3.10/site-packages"
)
FROZEN_TEST_PARENT_PTH = (
    FROZEN_TEST_VENV_ROOT
    / (
        "lib/python3.10/site-packages/"
        "gscl_parent_runtime.pth"
    )
)
FROZEN_TEST_CODE_PTH = (
    FROZEN_TEST_VENV_ROOT
    / "lib/python3.10/site-packages/gscl_code_root.pth"
)
FROZEN_PYTEST_KNOWN_ABSENT_RECORD_ENTRIES = (
    {
        "declared_path": "../../bin/py.test",
        "path": str(FROZEN_TEST_VENV_ROOT / "lib/bin/py.test"),
    },
    {
        "declared_path": "../../bin/pytest",
        "path": str(FROZEN_TEST_VENV_ROOT / "lib/bin/pytest"),
    },
)
FROZEN_MINILM_TARGET_MANIFEST = (
    FROZEN_DEPLOYMENT_ROOT
    / (
        "assets/minilm_target_qualification_ext4_r1/"
        "target_manifest.json"
    )
)
FROZEN_MINILM_TARGET_FILE_SHA256 = (
    "ea4054549bd22fe53029568b69e8652589f585a54985c54f09207ee3e6149d0e"
)
FROZEN_MINILM_TARGET_SELF_SHA256 = (
    "bd555bd3ed51c1d570ef168d503b02c2f9fd2c09507df9d4e06ff5fd46939dd7"
)
SOURCE_FREE_DESELECTED_TEST_NODES = (
    (
        "tests/test_gscl_arn_intrinsic_protocol_v1.py::"
        "test_official_source_exact_hash_doi_license_and_header_verify"
    ),
)
_EXPECTED_TEST_PTH_BYTES = {
    FROZEN_TEST_PARENT_PTH: (
        f"{FROZEN_PARENT_SITE}\n".encode("ascii")
    ),
    FROZEN_TEST_CODE_PTH: (
        (
            f"{FROZEN_WORKSPACE_CODE_ROOT}\n"
            f"{FROZEN_CODE_ROOT}\n"
        ).encode("ascii")
    ),
}

# Public, program-owned, source-free rows.  Each query and choice contains
# several disjoint lexical spans so the fixed three-span closed-choice grammar
# is representable without consulting an official item.
_PUBLIC_ROWS = (
    (
        "1",
        "Patient rivers carve silent stone over many seasons.",
        "Morning frost slowly covers the quiet valley floor.",
        "Gentle rainfall steadily fills the sheltered garden basin.",
        "Sudden sunlight quickly dries the exposed mountain path.",
        "high",
        "far",
        "A",
    ),
    (
        "3",
        "Careful builders measure twice before shaping strong timber.",
        "A careful pilot checks each instrument before departure.",
        "A patient baker weighs each ingredient before mixing.",
        "A hurried runner ignores every warning before starting.",
        "low",
        "near",
        "B",
    ),
)
PUBLIC_TOPOLOGY = ArnTopology(
    row_count=2,
    id_minimum=1,
    id_maximum=3,
    missing_ids=(2,),
    cell_counts={
        "far_high": 1,
        "far_low": 0,
        "near_high": 0,
        "near_low": 1,
    },
)


class QualificationRunnerError(RuntimeError):
    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


class QualificationDeferred(QualificationRunnerError):
    """A pre-attempt shared-node resource condition was not satisfied."""


def _public_fixture_bytes() -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(
        buffer,
        dialect="excel",
        lineterminator="\r\n",
    )
    writer.writerow(protocol.OFFICIAL_HEADER)
    writer.writerows(_PUBLIC_ROWS)
    return buffer.getvalue().encode("utf-8")


PUBLIC_FIXTURE_BYTES = _public_fixture_bytes()
PUBLIC_FIXTURE_SHA256 = hashlib.sha256(PUBLIC_FIXTURE_BYTES).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_absolute_fresh_root(path: Path) -> Path:
    if not isinstance(path, Path):
        raise QualificationRunnerError("qualification_root_invalid")
    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
        parent = absolute.parent.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise QualificationRunnerError(
            "qualification_root_invalid"
        ) from exc
    if (
        not path.is_absolute()
        or absolute != path
        or absolute.exists()
        or absolute == Path("/var/tmp")
        or absolute.name in {"", ".", ".."}
        or parent != absolute.parent
    ):
        raise QualificationRunnerError("qualification_root_not_fresh")
    try:
        absolute.relative_to(Path("/var/tmp"))
    except ValueError as exc:
        raise QualificationRunnerError(
            "qualification_root_outside_var_tmp"
        ) from exc
    if PurePath(absolute).parts.count(".."):
        raise QualificationRunnerError("qualification_root_invalid")
    return absolute


def _preflight_exact_assets(
    *,
    root: Path,
    qwen_model_root: Path,
    qwen_model_manifest: Path,
    qwen_actual_canary_lineage_terminal: Path,
    minilm_model_root: Path,
    minilm_asset_manifest: Path,
    minilm_target_manifest: Path,
) -> None:
    _require_absolute_fresh_root(root)
    for path in (qwen_model_root, minilm_model_root):
        supervisor._safe_absolute_path(  # noqa: SLF001
            path, allow_file=False
        )
    for path in (
        qwen_model_manifest,
        qwen_actual_canary_lineage_terminal,
        minilm_asset_manifest,
        minilm_target_manifest,
    ):
        supervisor._safe_absolute_path(  # noqa: SLF001
            path, allow_file=True
        )


def _preflight_fixed_source_free_test_runtime() -> None:
    if (  # noqa: SLF001
        supervisor._RECONSTRUCTION_ROOT != FROZEN_CODE_ROOT
        or supervisor._WORKSPACE_ROOT  # noqa: SLF001
        != FROZEN_WORKSPACE_CODE_ROOT
    ):
        raise QualificationRunnerError(
            "qualification_deployment_root_not_frozen"
        )
    reference_root = FROZEN_CODE_ROOT / "reference"
    if reference_root.exists():
        raise QualificationRunnerError(
            "official_source_tree_present"
        )
    for path in supervisor._walk_regular_files(FROZEN_CODE_ROOT):  # noqa: SLF001
        relative = path.relative_to(FROZEN_CODE_ROOT)
        if (
            "reference" in relative.parts
            or path.name.lower() in {
                "arn.csv",
                "arn_dataset_v1.csv",
            }
        ):
            raise QualificationRunnerError(
                "official_source_tree_present"
            )
    # The invocation path is intentionally one exact, hash-bound symlink.
    # Validate every parent component first; validate the link itself below
    # with lstat/readlink/realpath rather than passing it to the generic
    # no-symlink closure helper.
    supervisor._safe_absolute_path(  # noqa: SLF001
        FROZEN_TEST_PYTHON.parent, allow_file=False
    )
    try:
        python_lstat = FROZEN_TEST_PYTHON.lstat()
        python_link_target = os.readlink(FROZEN_TEST_PYTHON)
        python_resolved = FROZEN_TEST_PYTHON.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise QualificationRunnerError(
            "frozen_test_python_invalid"
        ) from exc
    if (
        not stat.S_ISLNK(python_lstat.st_mode)
        or python_link_target != FROZEN_TEST_PYTHON_LINK_TARGET
        or python_resolved != FROZEN_TEST_PYTHON_RESOLVED
        or supervisor._hash_regular_absolute(  # noqa: SLF001
            python_resolved
        )
        != FROZEN_TEST_PYTHON_RESOLVED_SHA256
        or supervisor._hash_regular_absolute(  # noqa: SLF001
            FROZEN_TEST_PYVENV_CONFIG
        )
        != FROZEN_TEST_PYVENV_CONFIG_SHA256
    ):
        raise QualificationRunnerError(
            "frozen_test_python_invalid"
        )
    supervisor._safe_absolute_path(  # noqa: SLF001
        FROZEN_PYTEST_WHEEL_BUNDLE_MANIFEST,
        allow_file=True,
    )
    for path, expected_raw in _EXPECTED_TEST_PTH_BYTES.items():
        actual = supervisor._read_regular_absolute_exact(  # noqa: SLF001
            path,
            expected_sha256=hashlib.sha256(
                expected_raw
            ).hexdigest(),
            maximum_bytes=4096,
        )
        if actual != expected_raw:
            raise QualificationRunnerError(
                "frozen_test_pth_invalid"
            )


def _preflight_frozen_main_runtime() -> None:
    try:
        executable = Path(os.path.abspath(sys.executable))
        metadata = executable.lstat()
        prefix = Path(sys.prefix)
        base_prefix = Path(sys.base_prefix)
    except (OSError, TypeError, ValueError) as exc:
        raise QualificationRunnerError(
            "frozen_main_runtime_invalid"
        ) from exc
    if (
        executable != FROZEN_MAIN_PYTHON
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or prefix != FROZEN_MAIN_RUNTIME_ROOT
        or base_prefix != FROZEN_MAIN_BASE_PREFIX
        or supervisor._hash_regular_absolute(  # noqa: SLF001
            executable
        )
        != FROZEN_MAIN_PYTHON_SHA256
        or supervisor._hash_regular_absolute(  # noqa: SLF001
            FROZEN_MAIN_PYVENV_CONFIG
        )
        != FROZEN_MAIN_PYVENV_CONFIG_SHA256
    ):
        raise QualificationRunnerError(
            "frozen_main_runtime_invalid"
        )


def _preflight_frozen_runtime_binding_manifest() -> None:
    raw = supervisor._read_regular_absolute_exact(  # noqa: SLF001
        FROZEN_RUNTIME_BINDING_MANIFEST,
        expected_sha256=FROZEN_RUNTIME_BINDING_FILE_SHA256,
        maximum_bytes=64 * 1024,
    )
    try:
        manifest = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationRunnerError(
            "frozen_runtime_binding_invalid"
        ) from exc
    if not isinstance(manifest, dict):
        raise QualificationRunnerError(
            "frozen_runtime_binding_invalid"
        )
    body = dict(manifest)
    claimed = body.pop("self_sha256", None)
    if (
        manifest.get("schema") != "gscl_ext4_runtime_binding_v1"
        or manifest.get("status")
        != "PASS_EXT4_RUNTIME_EXACT_BINDING"
        or claimed != FROZEN_RUNTIME_BINDING_SELF_SHA256
        or _content_hash(body) != claimed
        or manifest.get("formal_measurement") is not False
        or manifest.get("effect_gate_count") != 0
        or manifest.get("official_source_access_count") != 0
        or manifest.get("scorer_call_count") != 0
        or manifest.get("api_evaluation_count") != 0
    ):
        raise QualificationRunnerError(
            "frozen_runtime_binding_invalid"
        )


def _validate_fixed_test_attestation(
    attestation: supervisor.TestAttestation,
) -> None:
    receipt = attestation.receipt
    test_runner = receipt.get("test_runner")
    expected_pth_hashes = {
        str(path): hashlib.sha256(raw).hexdigest()
        for path, raw in _EXPECTED_TEST_PTH_BYTES.items()
    }
    observed_versions = {
        row.get("distribution"): row.get("version")
        for row in (
            test_runner.get("distribution_closures", [])
            if isinstance(test_runner, dict)
            else []
        )
        if isinstance(row, dict)
    }
    pytest_closure = next(
        (
            row
            for row in (
                test_runner.get("distribution_closures", [])
                if isinstance(test_runner, dict)
                else []
            )
            if isinstance(row, dict)
            and row.get("distribution") == "pytest"
        ),
        None,
    )
    expected_test_files = {
        str(path)
        for path in (
            supervisor._INTERNAL_QUALIFICATION_TEST_PATHS.values()  # noqa: SLF001
        )
    }
    if (
        not isinstance(test_runner, dict)
        or test_runner.get("interpreter_path")
        != str(FROZEN_TEST_PYTHON)
        or test_runner.get("interpreter_invocation_is_symlink")
        is not True
        or test_runner.get("interpreter_resolved_path")
        != str(FROZEN_TEST_PYTHON_RESOLVED)
        or test_runner.get("interpreter_sha256")
        != FROZEN_TEST_PYTHON_RESOLVED_SHA256
        or test_runner.get(
            "interpreter_invocation_binding_sha256"
        )
        != hashlib.sha256(
            FROZEN_TEST_PYTHON_LINK_TARGET.encode("utf-8")
        ).hexdigest()
        or test_runner.get("pyvenv_config_path")
        != str(FROZEN_TEST_PYVENV_CONFIG)
        or test_runner.get("pyvenv_config_sha256")
        != FROZEN_TEST_PYVENV_CONFIG_SHA256
        or test_runner.get("pytest_version") != "8.3.3"
        or test_runner.get("cuda_visible_devices") != ""
        or test_runner.get(
            "bytecode_writes_disabled_by_cli"
        )
        is not True
        or test_runner.get("pytest_config_file") != "/dev/null"
        or test_runner.get("pytest_rootdir")
        != str(FROZEN_CODE_ROOT)
        or test_runner.get("pth_file_sha256s")
        != dict(sorted(expected_pth_hashes.items()))
        or test_runner.get(
            "pytest_wheel_bundle_manifest", {}
        ).get("path")
        != str(FROZEN_PYTEST_WHEEL_BUNDLE_MANIFEST)
        or observed_versions.get("numpy") != "2.2.6"
        or not isinstance(pytest_closure, dict)
        or pytest_closure.get("absent_entries")
        != list(FROZEN_PYTEST_KNOWN_ABSENT_RECORD_ENTRIES)
        or pytest_closure.get("declared_entry_count")
        != (
            pytest_closure.get("present_file_count", -1)
            + len(FROZEN_PYTEST_KNOWN_ABSENT_RECORD_ENTRIES)
        )
        or receipt.get("deselected_test_nodes")
        != list(SOURCE_FREE_DESELECTED_TEST_NODES)
        or receipt.get("official_source_access_count") != 0
        or receipt.get("source_content_supplied") is not False
        or receipt.get("code_tree_unchanged") is not True
        or set(receipt.get("test_file_sha256s", {}))
        != expected_test_files
    ):
        raise QualificationRunnerError(
            "frozen_test_attestation_invalid"
        )


def _run_nvidia_smi(arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            (str(_NVIDIA_SMI), *tuple(arguments)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationRunnerError(
            "gpu_preflight_unavailable"
        ) from exc
    if completed.returncode != 0:
        raise QualificationRunnerError("gpu_preflight_failed")
    return completed.stdout


def _decimal_lines(raw: bytes, *, issue_id: str) -> tuple[int, ...]:
    try:
        lines = tuple(
            line.strip()
            for line in raw.decode("ascii").splitlines()
            if line.strip()
        )
    except UnicodeError as exc:
        raise QualificationRunnerError(issue_id) from exc
    if any(not line.isdecimal() for line in lines):
        raise QualificationRunnerError(issue_id)
    return tuple(int(line) for line in lines)


def _preflight_exactly_two_idle_gpus() -> None:
    try:
        resolved = _NVIDIA_SMI.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise QualificationRunnerError(
            "gpu_preflight_unavailable"
        ) from exc
    if not _NVIDIA_SMI.is_absolute() or not resolved.is_file():
        raise QualificationRunnerError("gpu_preflight_unavailable")
    indices = _decimal_lines(
        _run_nvidia_smi(
            (
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            )
        ),
        issue_id="gpu_inventory_invalid",
    )
    if indices != _EXPECTED_GPU_INDICES:
        raise QualificationRunnerError("gpu_inventory_invalid")
    compute_pids = _decimal_lines(
        _run_nvidia_smi(
            (
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            )
        ),
        issue_id="gpu_process_inventory_invalid",
    )
    if compute_pids:
        raise QualificationDeferred("gpu_compute_process_present")


def _validate_inner_terminal(
    terminal: Mapping[str, Any],
    *,
    action: supervisor.FrozenAction,
    invocation: supervisor.FormalInvocation,
    barrier: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(terminal, Mapping):
        raise QualificationRunnerError(
            "internal_qualification_terminal_invalid"
        )
    body = dict(terminal)
    claimed = body.pop("self_hash", None)
    expected_canary = {
        key: value
        for key, value in action.receipt[
            "qwen_actual_canary_lineage_terminal"
        ].items()
        if key != "path"
    }
    if (
        terminal.get("schema")
        != supervisor.INTERNAL_FACTORY_QUALIFICATION_SCHEMA
        or terminal.get("status")
        != "PASS_SOURCE_FREE_EXACT_INTERNAL_FACTORY_QUALIFICATION"
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or _content_hash(body) != claimed
        or terminal.get("one_shot_key")
        != invocation.receipt["one_shot_key"]
        or terminal.get("qualification_action_self_hash")
        != action.receipt["self_hash"]
        or terminal.get(
            "qualification_runtime_closure_self_hash"
        )
        != action.closure.manifest["self_hash"]
        or terminal.get("four_arm_barrier_self_hash")
        != barrier.get("self_hash")
        or terminal.get("synthetic_source_sha256")
        != PUBLIC_FIXTURE_SHA256
        or terminal.get("common_item_count") != len(_PUBLIC_ROWS)
        or terminal.get("closed_choice_selection_count")
        != 3 * len(_PUBLIC_ROWS)
        or terminal.get("free_form_generation_count") != 0
        or terminal.get("score_operation")
        != "teacher_forced_forward_log_softmax"
        or terminal.get("qwen_actual_canary_lineage_binding")
        != expected_canary
        or terminal.get("official_source_content_supplied_to_model")
        is not False
        or terminal.get("public_synthetic_content_supplied_to_model")
        is not True
        or terminal.get("official_source_access_count") != 0
        or terminal.get("label_open_count") != 0
        or terminal.get("online_or_api_evaluation_count") != 0
        or terminal.get("formal_measurement_authorized") is not False
        or terminal.get("formal_root_used") is not False
        or terminal.get("formal_result") is not False
        or terminal.get("efficacy_evidence") is not False
        or terminal.get("effect_gate_added") is not False
        or terminal.get("item_content_emitted") is not False
    ):
        raise QualificationRunnerError(
            "internal_qualification_terminal_invalid"
        )
    return terminal


def _outer_terminal(
    *,
    action: supervisor.FrozenAction,
    invocation: supervisor.FormalInvocation,
    source_receipt: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    barrier: Mapping[str, Any],
    inner_terminal: Mapping[str, Any],
    test_attestation: supervisor.TestAttestation,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": SAFE_TERMINAL_SCHEMA,
        "status": "PASS_FIXED_SOURCE_FREE_INTERNAL_FACTORY_QUALIFICATION",
        "runner_version": VERSION,
        "claim_scope": (
            "runtime_qualification_only_no_effect_or_formal_measurement"
        ),
        "public_fixture_sha256": PUBLIC_FIXTURE_SHA256,
        "public_fixture_item_count": len(_PUBLIC_ROWS),
        "executed_test_attestation_self_hash": (
            test_attestation.receipt["self_hash"]
        ),
        "runtime_closure_self_hash": action.closure.manifest[
            "self_hash"
        ],
        "action_self_hash": action.receipt["self_hash"],
        "one_shot_key": invocation.receipt["one_shot_key"],
        "source_receipt_self_hash": source_receipt["self_hash"],
        "factory_execution_receipt_self_hash": execution_receipt[
            "self_hash"
        ],
        "four_arm_barrier_self_hash": barrier["self_hash"],
        "inner_terminal_self_hash": inner_terminal["self_hash"],
        "common_item_count": inner_terminal["common_item_count"],
        "closed_choice_selection_count": inner_terminal[
            "closed_choice_selection_count"
        ],
        "free_form_generation_count": 0,
        "score_operation": "teacher_forced_forward_log_softmax",
        "qwen_model_manifest_sha256": inner_terminal[
            "qwen_model_manifest_sha256"
        ],
        "qwen_actual_canary_lineage_binding": inner_terminal[
            "qwen_actual_canary_lineage_binding"
        ],
        "minilm_asset_manifest_sha256": inner_terminal[
            "minilm_asset_manifest_sha256"
        ],
        "minilm_target_manifest_file_sha256": inner_terminal[
            "minilm_target_manifest_file_sha256"
        ],
        "minilm_target_manifest_self_sha256": inner_terminal[
            "minilm_target_manifest_self_sha256"
        ],
        "outer_systemd_attestation_self_hash": inner_terminal[
            "outer_systemd_attestation_self_hash"
        ],
        "outer_systemd_stable_binding_sha256": inner_terminal[
            "outer_systemd_stable_binding_sha256"
        ],
        "official_source_access_count": 0,
        "label_open_count": 0,
        "offline_scorer_call_count": 0,
        "online_or_api_evaluation_count": 0,
        "official_source_content_supplied_to_model": False,
        "public_synthetic_content_supplied_to_model": True,
        "formal_measurement": False,
        "formal_result": False,
        "efficacy_evidence": False,
        "effect_gate_added": False,
        "caller_source_rows_accepted": False,
        "caller_predictions_accepted": False,
        "caller_labels_accepted": False,
        "caller_scores_accepted": False,
        "item_content_emitted": False,
    }
    return {**body, "self_hash": _content_hash(body)}


def run_qualification(
    *,
    root: Path,
    qwen_model_root: Path,
    qwen_model_manifest: Path,
    qwen_actual_canary_lineage_terminal: Path,
    minilm_model_root: Path,
    minilm_asset_manifest: Path,
) -> Mapping[str, Any]:
    """Execute the fixed public fixture through the exact internal core once."""

    _preflight_exact_assets(
        root=root,
        qwen_model_root=qwen_model_root,
        qwen_model_manifest=qwen_model_manifest,
        qwen_actual_canary_lineage_terminal=(
            qwen_actual_canary_lineage_terminal
        ),
        minilm_model_root=minilm_model_root,
        minilm_asset_manifest=minilm_asset_manifest,
        minilm_target_manifest=FROZEN_MINILM_TARGET_MANIFEST,
    )
    target_raw = supervisor._read_regular_absolute_exact(  # noqa: SLF001
        FROZEN_MINILM_TARGET_MANIFEST,
        expected_sha256=FROZEN_MINILM_TARGET_FILE_SHA256,
        maximum_bytes=2 * 1024 * 1024,
    )
    target_manifest = (
        supervisor._validate_minilm_target_manifest_bytes(  # noqa: SLF001
            target_raw
        )
    )
    if (
        target_manifest.get("self_sha256")
        != FROZEN_MINILM_TARGET_SELF_SHA256
    ):
        raise QualificationRunnerError(
            "frozen_minilm_target_invalid"
        )
    _preflight_frozen_main_runtime()
    _preflight_frozen_runtime_binding_manifest()
    _preflight_fixed_source_free_test_runtime()
    _preflight_exactly_two_idle_gpus()
    runner_path = Path(__file__).resolve()
    test_attestation = supervisor.run_source_free_tests(
        code_root=supervisor._RECONSTRUCTION_ROOT,  # noqa: SLF001
        test_files=tuple(
            supervisor._INTERNAL_QUALIFICATION_TEST_PATHS.values()  # noqa: SLF001
        ),
        deselected_test_nodes=SOURCE_FREE_DESELECTED_TEST_NODES,
        test_python=FROZEN_TEST_PYTHON,
        pytest_wheel_bundle_manifest=(
            FROZEN_PYTEST_WHEEL_BUNDLE_MANIFEST
        ),
    )
    _validate_fixed_test_attestation(test_attestation)
    closure = supervisor.attest_runtime_closure(
        code_roots=(supervisor._RECONSTRUCTION_ROOT,),  # noqa: SLF001
        entry_files=(
            runner_path,
            *tuple(
                supervisor._INTERNAL_FORMAL_IMPLEMENTATION_PATHS.values()  # noqa: SLF001
            ),
        ),
        config_files=(
            qwen_model_manifest,
            qwen_actual_canary_lineage_terminal,
            minilm_asset_manifest,
            FROZEN_MINILM_TARGET_MANIFEST,
            FROZEN_RUNTIME_BINDING_MANIFEST,
            FROZEN_PYTEST_WHEEL_BUNDLE_MANIFEST,
        ),
        asset_roots=(qwen_model_root, minilm_model_root),
        test_attestation=test_attestation,
        support_module_files=(  # noqa: SLF001
            supervisor._INTERNAL_SUPPORT_MODULE_PATHS
        ),
    )
    # Repeat immediately before root creation.  A deferral here still creates
    # no attempt, source pack, secret, or evidence root.
    _preflight_exactly_two_idle_gpus()
    freeze_commitments = {
        "fixed_public_fixture": PUBLIC_FIXTURE_SHA256,
        "qualification_protocol": _content_hash(
            {
                "four_arm_barrier_required": True,
                "formal_measurement": False,
                "label_open_count": 0,
                "official_source_access_count": 0,
                "runner_version": VERSION,
                "scorer_call_count": 0,
            }
        ),
        "runner_implementation": hashlib.sha256(
            runner_path.read_bytes()
        ).hexdigest(),
        "runtime_binding_manifest": (
            FROZEN_RUNTIME_BINDING_FILE_SHA256
        ),
    }
    with supervisor.FormalSupervisor._source_free_qualification(
        root
    ) as runtime:
        action = (
            runtime.freeze_internal_factory_qualification_action_once(
                closure=closure,
                freeze_commitments=freeze_commitments,
                qwen_model_root=qwen_model_root,
                qwen_model_manifest=qwen_model_manifest,
                qwen_actual_canary_lineage_terminal=(
                    qwen_actual_canary_lineage_terminal
                ),
                minilm_model_root=minilm_model_root,
                minilm_asset_manifest=minilm_asset_manifest,
                minilm_target_manifest=(
                    FROZEN_MINILM_TARGET_MANIFEST
                ),
                source_sha256=PUBLIC_FIXTURE_SHA256,
            )
        )
        invocation = runtime.begin_once(action)
        source_receipt = runtime.materialize_synthetic_packs_once(
            invocation,
            raw=PUBLIC_FIXTURE_BYTES,
            expected_topology=PUBLIC_TOPOLOGY,
        )
        execution_receipt = runtime.run_internal_factory_once(invocation)
        barrier = runtime.seal_four_arm_barrier_once(invocation)
        key = invocation.receipt["one_shot_key"]
        inner = runtime.store.read_json(
            f"state/attempts/{key}."
            "internal_factory_qualification.safe.json"
        )
        inner_terminal = _validate_inner_terminal(
            inner,
            action=action,
            invocation=invocation,
            barrier=barrier,
        )
        terminal = _outer_terminal(
            action=action,
            invocation=invocation,
            source_receipt=source_receipt,
            execution_receipt=execution_receipt,
            barrier=barrier,
            inner_terminal=inner_terminal,
            test_attestation=test_attestation,
        )
        runtime.store.ensure_directory("control")
        runtime.store.write_json_exclusive(
            "control/outer_terminal.safe.json", terminal
        )
        return terminal


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--qwen-model-root", required=True, type=Path)
    parser.add_argument("--qwen-model-manifest", required=True, type=Path)
    parser.add_argument(
        "--qwen-actual-canary-lineage-terminal",
        required=True,
        type=Path,
    )
    parser.add_argument("--minilm-model-root", required=True, type=Path)
    parser.add_argument(
        "--minilm-asset-manifest", required=True, type=Path
    )
    arguments = parser.parse_args(argv)
    try:
        run_qualification(
            root=arguments.root,
            qwen_model_root=arguments.qwen_model_root,
            qwen_model_manifest=arguments.qwen_model_manifest,
            qwen_actual_canary_lineage_terminal=(
                arguments.qwen_actual_canary_lineage_terminal
            ),
            minilm_model_root=arguments.minilm_model_root,
            minilm_asset_manifest=arguments.minilm_asset_manifest,
        )
    except QualificationDeferred as exc:
        print(
            f"{VERSION} deferred before attempt: {exc.issue_id}",
            file=sys.stderr,
        )
        return DEFERRED_EXIT_CODE
    except (
        QualificationRunnerError,
        supervisor.FormalSupervisorError,
    ) as exc:
        issue_id = getattr(exc, "issue_id", type(exc).__name__)
        print(
            f"{VERSION} failed closed: {issue_id}",
            file=sys.stderr,
        )
        return 2
    return 0


__all__ = [
    "DEFERRED_EXIT_CODE",
    "PUBLIC_FIXTURE_BYTES",
    "PUBLIC_FIXTURE_SHA256",
    "PUBLIC_TOPOLOGY",
    "QualificationDeferred",
    "QualificationRunnerError",
    "SAFE_TERMINAL_SCHEMA",
    "VERSION",
    "run_qualification",
]


if __name__ == "__main__":
    raise SystemExit(_main())
