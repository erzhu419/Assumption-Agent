"""One-shot public ARN intrinsic measurement for the frozen GSCL factory.

This module is deliberately thin.  It accepts no source path, item, prediction,
label, score, callback, model, provider, threshold, or effect gate from its
caller.  The official public ARN bytes must be staged below the supervisor's
compiled-in formal root before this entry point starts.  The supervisor then
owns the only content-sensitive lifecycle:

``freeze -> begin -> materialize -> internal factory -> four-arm barrier
-> one fixed offline scorer``.

Only aggregate score fields cross into ``control/outer_terminal.safe.json``.
Any failure after ``begin_once`` seals a content-free terminal and permanently
disallows retry or replay of this fixed root.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import (
    gscl_arn_formal_supervisor_v1 as supervisor,
)
from assumption_agent.benchmarks import (
    gscl_arn_internal_factory_qualification_v1 as qualification,
)
from assumption_agent.benchmarks import (
    gscl_arn_intrinsic_protocol_v1 as protocol,
)


VERSION = "gscl_arn_formal_measurement_v1"
SAFE_TERMINAL_SCHEMA = f"{VERSION}.safe_terminal.v1"
FAILED_TERMINAL_SCHEMA = f"{VERSION}.failed_after_begin.safe.v1"
DEFERRED_EXIT_CODE = 75
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ISSUE_ID = re.compile(r"[a-z0-9_]{1,160}\Z")

# The source identity is duplicated here intentionally: the formal entry point
# fails closed if the independently frozen protocol constants ever drift.
FORMAL_ROOT = Path("/var/tmp/gscl_arn_intrinsic_formal_v1")
OFFICIAL_DATASET_SIZE = 1_256_913
OFFICIAL_DATASET_SHA256 = (
    "a866fe5341ce4a29f00f24987a12278303b2b8ad788352f549b0fe051ad4a7a8"
)
OFFICIAL_METADATA_SIZE = 5_562
OFFICIAL_METADATA_SHA256 = (
    "c9e91d7a49ea383eeccec5421cce9f1b0d8713c243187d840482eb1764f3317f"
)
OFFICIAL_DOI = "10.5281/zenodo.11044026"
OFFICIAL_LICENSE_ID = "cc-by-4.0"
FROZEN_SOURCE_DATASET = FORMAL_ROOT / "source/arn.csv"
FROZEN_SOURCE_METADATA = FORMAL_ROOT / "source/metadata.json"

FROZEN_DEPLOYMENT_ROOT = qualification.FROZEN_DEPLOYMENT_ROOT
FROZEN_CODE_ROOT = qualification.FROZEN_CODE_ROOT
FROZEN_QWEN_MODEL_ROOT = Path(
    "/var/tmp/gscl_closed_choice_actual_qualification_20260730/"
    "model_snapshot"
)
FROZEN_QWEN_MODEL_MANIFEST = Path(
    "/var/tmp/gscl_closed_choice_actual_qualification_20260730/"
    "qwen_manifest.json"
)
FROZEN_QWEN_MODEL_MANIFEST_SHA256 = (
    "970fd38542fc3e00f9c98e2efda0bcb4e9355e0974f0a9cd5ae38cc57a82e658"
)
FROZEN_QWEN_ACTUAL_CANARY_LINEAGE_TERMINAL = (
    FROZEN_CODE_ROOT
    / (
        "manifests/"
        "gscl_closed_choice_actual_canary_lineage_terminal_ext4_20260730.json"
    )
)
FROZEN_QWEN_ACTUAL_CANARY_LINEAGE_FILE_SHA256 = (
    "4a15b4b209896a7999f62371df1c69e65b9d7b95b5821de58a9b8497ccac5f6a"
)
FROZEN_MINILM_MODEL_ROOT = (
    FROZEN_DEPLOYMENT_ROOT / "assets/minilm_model"
)
FROZEN_MINILM_ASSET_MANIFEST = (
    FROZEN_DEPLOYMENT_ROOT / "assets/minilm_asset_manifest.json"
)
FROZEN_MINILM_ASSET_MANIFEST_SHA256 = (
    "62b85c7752f2e46932fb9fb13ae2f3aac9eb750a33c8f07102739040feb6cc75"
)
FROZEN_MINILM_TARGET_MANIFEST = (
    qualification.FROZEN_MINILM_TARGET_MANIFEST
)
FROZEN_MINILM_TARGET_FILE_SHA256 = (
    qualification.FROZEN_MINILM_TARGET_FILE_SHA256
)

@dataclass(frozen=True)
class FrozenReceiptBinding:
    path: Path
    file_sha256: str
    self_sha256: str


FROZEN_QWEN_RUNTIME_QUALIFICATION_RECEIPTS = (
    FrozenReceiptBinding(
        path=Path(
            "/var/tmp/gscl_unified_nonscoring_harness_20260730/work/"
            "source_free_internal_factory_qualification_ext4_repair_r4/"
            "work/formal_factory/"
            "341903418662da7d71bea349f9eaefef8c8abbf4cc321713e6f8f6d872c813ee/"
            "extractor_shard_0/runtime.safe.json"
        ),
        file_sha256=(
            "2c9dcf9279b998020829c207935a11212816a1768fb66160c9bb8e9daf674664"
        ),
        self_sha256=(
            "271a7fc0bbe4b8b17f444aa4afa76f871a86526a4498a1b12933150695f0fed0"
        ),
    ),
    FrozenReceiptBinding(
        path=Path(
            "/var/tmp/gscl_unified_nonscoring_harness_20260730/work/"
            "source_free_internal_factory_qualification_ext4_repair_r4/"
            "work/formal_factory/"
            "341903418662da7d71bea349f9eaefef8c8abbf4cc321713e6f8f6d872c813ee/"
            "extractor_shard_1/runtime.safe.json"
        ),
        file_sha256=(
            "efe6997ed172fbb240ff9d30550197c3faaf7571496ceb0cf31c2cfada6df62a"
        ),
        self_sha256=(
            "a90a9c2a919c81caf0fbc5c999d9477158f2ddfa2a12999c6885243ce6a8e3da"
        ),
    ),
)
FROZEN_INTERNAL_FACTORY_QUALIFICATION_RECEIPT = FrozenReceiptBinding(
    path=Path(
        "/var/tmp/gscl_unified_nonscoring_harness_20260730/work/"
        "source_free_internal_factory_qualification_ext4_repair_r4/"
        "state/attempts/"
        "341903418662da7d71bea349f9eaefef8c8abbf4cc321713e6f8f6d872c813ee."
        "internal_factory_qualification.safe.json"
    ),
    file_sha256=(
        "75597701af9c3761509471d6c54109006851117419f301ddb3c57e0d150a2111"
    ),
    self_sha256=(
        "5c734f946295338f64a84ba6bf5bd2ebc1bc5b8b97c934de437c697cc391f9ad"
    ),
)

class FormalMeasurementError(RuntimeError):
    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


class FormalMeasurementDeferred(FormalMeasurementError):
    """A shared-node resource was unavailable before a formal attempt."""


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


def _hash_regular_file(path: Path, *, maximum_bytes: int) -> str:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FormalMeasurementError("frozen_file_unavailable") from exc
    if (
        not path.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_size > maximum_bytes
    ):
        raise FormalMeasurementError("frozen_file_invalid")
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise FormalMeasurementError("frozen_file_invalid")
                digest.update(chunk)
    except OSError as exc:
        raise FormalMeasurementError("frozen_file_unavailable") from exc
    return digest.hexdigest()


def _validate_source_constants() -> None:
    if (
        FORMAL_ROOT != supervisor.FORMAL_ROOT
        or OFFICIAL_DATASET_SIZE != protocol.OFFICIAL_DATASET_SIZE
        or OFFICIAL_DATASET_SHA256 != protocol.OFFICIAL_DATASET_SHA256
        or OFFICIAL_METADATA_SIZE != protocol.OFFICIAL_METADATA_SIZE
        or OFFICIAL_METADATA_SHA256 != protocol.OFFICIAL_METADATA_SHA256
        or OFFICIAL_DOI != protocol.OFFICIAL_DOI
        or OFFICIAL_LICENSE_ID != protocol.OFFICIAL_LICENSE_ID
        or FROZEN_SOURCE_DATASET
        != FORMAL_ROOT / supervisor.FORMAL_SOURCE_DATASET_RELATIVE
        or FROZEN_SOURCE_METADATA
        != FORMAL_ROOT / supervisor.FORMAL_SOURCE_METADATA_RELATIVE
    ):
        raise FormalMeasurementError("official_source_binding_drifted")


def _validate_receipt_binding(
    binding: FrozenReceiptBinding,
) -> Mapping[str, Any]:
    if (
        not isinstance(binding, FrozenReceiptBinding)
        or not binding.path.is_absolute()
        or _SHA256.fullmatch(binding.file_sha256) is None
        or _SHA256.fullmatch(binding.self_sha256) is None
    ):
        raise FormalMeasurementError(
            "qualification_receipt_binding_pending"
        )
    observed = _hash_regular_file(
        binding.path, maximum_bytes=16 * 1024 * 1024
    )
    if observed != binding.file_sha256:
        raise FormalMeasurementError(
            "qualification_receipt_file_drifted"
        )
    try:
        value = json.loads(
            binding.path.read_text(encoding="ascii"),
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FormalMeasurementError(
            "qualification_receipt_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise FormalMeasurementError("qualification_receipt_invalid")
    body = dict(value)
    claimed = body.pop("self_hash", body.pop("self_sha256", None))
    if (
        claimed != binding.self_sha256
        or _content_hash(body) != binding.self_sha256
    ):
        raise FormalMeasurementError("qualification_receipt_invalid")
    return value


def _validate_fixed_assets_and_receipts() -> tuple[Path, ...]:
    fixed_files = (
        (
            FROZEN_QWEN_MODEL_MANIFEST,
            FROZEN_QWEN_MODEL_MANIFEST_SHA256,
        ),
        (
            FROZEN_QWEN_ACTUAL_CANARY_LINEAGE_TERMINAL,
            FROZEN_QWEN_ACTUAL_CANARY_LINEAGE_FILE_SHA256,
        ),
        (
            FROZEN_MINILM_ASSET_MANIFEST,
            FROZEN_MINILM_ASSET_MANIFEST_SHA256,
        ),
        (
            FROZEN_MINILM_TARGET_MANIFEST,
            FROZEN_MINILM_TARGET_FILE_SHA256,
        ),
    )
    for path, expected in fixed_files:
        if (
            _hash_regular_file(path, maximum_bytes=4 * 1024 * 1024)
            != expected
        ):
            raise FormalMeasurementError("frozen_asset_manifest_drifted")
    qwen_receipts: list[Path] = []
    for binding in FROZEN_QWEN_RUNTIME_QUALIFICATION_RECEIPTS:
        _validate_receipt_binding(binding)
        qwen_receipts.append(binding.path)
    _validate_receipt_binding(
        FROZEN_INTERNAL_FACTORY_QUALIFICATION_RECEIPT
    )
    return tuple(qwen_receipts)


def _preflight_staged_source_without_opening_content() -> None:
    """Check custody and size only; formal bytes first open after begin."""

    source_root = FORMAL_ROOT / "source"
    try:
        root_metadata = FORMAL_ROOT.lstat()
        source_metadata = source_root.lstat()
        root_entries = {path.name for path in FORMAL_ROOT.iterdir()}
        source_entries = {path.name for path in source_root.iterdir()}
    except OSError as exc:
        raise FormalMeasurementError(
            "formal_source_not_staged"
        ) from exc
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.getuid()
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
        or not stat.S_ISDIR(source_metadata.st_mode)
        or source_metadata.st_uid != os.getuid()
        or stat.S_IMODE(source_metadata.st_mode) != 0o700
        or root_entries != {"source"}
        or source_entries != {"arn.csv", "metadata.json"}
    ):
        raise FormalMeasurementError("formal_root_topology_invalid")
    for path, expected_size in (
        (FROZEN_SOURCE_DATASET, OFFICIAL_DATASET_SIZE),
        (FROZEN_SOURCE_METADATA, OFFICIAL_METADATA_SIZE),
    ):
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise FormalMeasurementError(
                "formal_source_not_staged"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size != expected_size
        ):
            raise FormalMeasurementError(
                "formal_source_topology_invalid"
            )


def _preflight_never_started() -> None:
    forbidden_files = (
        FORMAL_ROOT / "state/action.freeze.json",
        FORMAL_ROOT / "control/outer_terminal.safe.json",
    )
    if any(path.exists() for path in forbidden_files):
        raise FormalMeasurementError(
            "formal_root_already_started_no_retry"
        )
    attempts = FORMAL_ROOT / "state/attempts"
    if attempts.exists():
        try:
            metadata = attempts.lstat()
            populated = next(attempts.iterdir(), None) is not None
        except OSError as exc:
            raise FormalMeasurementError(
                "formal_root_attempt_state_invalid"
            ) from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or populated
        ):
            raise FormalMeasurementError(
                "formal_root_already_started_no_retry"
            )


def _source_free_runtime_closure(
    *,
    qwen_receipts: Sequence[Path],
) -> tuple[supervisor.RuntimeClosure, supervisor.TestAttestation]:
    runner_path = Path(__file__).resolve()
    test_attestation = supervisor.run_source_free_tests(
        code_root=supervisor._RECONSTRUCTION_ROOT,  # noqa: SLF001
        test_files=tuple(
            supervisor._INTERNAL_QUALIFICATION_TEST_PATHS.values()  # noqa: SLF001
        ),
        deselected_test_nodes=qualification.SOURCE_FREE_DESELECTED_TEST_NODES,
        test_python=qualification.FROZEN_TEST_PYTHON,
        pytest_wheel_bundle_manifest=(
            qualification.FROZEN_PYTEST_WHEEL_BUNDLE_MANIFEST
        ),
    )
    qualification._validate_fixed_test_attestation(  # noqa: SLF001
        test_attestation
    )
    closure = supervisor.attest_runtime_closure(
        code_roots=(supervisor._RECONSTRUCTION_ROOT,),  # noqa: SLF001
        entry_files=(
            runner_path,
            *tuple(
                supervisor._INTERNAL_FORMAL_IMPLEMENTATION_PATHS.values()  # noqa: SLF001
            ),
        ),
        config_files=(
            FROZEN_QWEN_MODEL_MANIFEST,
            FROZEN_QWEN_ACTUAL_CANARY_LINEAGE_TERMINAL,
            *tuple(qwen_receipts),
            FROZEN_INTERNAL_FACTORY_QUALIFICATION_RECEIPT.path,
            FROZEN_MINILM_ASSET_MANIFEST,
            FROZEN_MINILM_TARGET_MANIFEST,
            qualification.FROZEN_RUNTIME_BINDING_MANIFEST,
            qualification.FROZEN_PYTEST_WHEEL_BUNDLE_MANIFEST,
        ),
        asset_roots=(
            FROZEN_QWEN_MODEL_ROOT,
            FROZEN_MINILM_MODEL_ROOT,
        ),
        test_attestation=test_attestation,
        support_module_files=(  # noqa: SLF001
            supervisor._INTERNAL_SUPPORT_MODULE_PATHS
        ),
    )
    return closure, test_attestation


def _freeze_commitments(
    *,
    closure: supervisor.RuntimeClosure,
) -> Mapping[str, str]:
    runner_path = Path(__file__).resolve()
    source_contract = {
        "dataset_sha256": OFFICIAL_DATASET_SHA256,
        "dataset_size": OFFICIAL_DATASET_SIZE,
        "doi": OFFICIAL_DOI,
        "license_id": OFFICIAL_LICENSE_ID,
        "metadata_sha256": OFFICIAL_METADATA_SHA256,
        "metadata_size": OFFICIAL_METADATA_SIZE,
        "official_row_count": protocol.OFFICIAL_ROW_COUNT,
    }
    receipt_contract = {
        "internal_factory": {
            "file_sha256": (
                FROZEN_INTERNAL_FACTORY_QUALIFICATION_RECEIPT.file_sha256
            ),
            "self_sha256": (
                FROZEN_INTERNAL_FACTORY_QUALIFICATION_RECEIPT.self_sha256
            ),
        },
        "qwen": [
            {
                "file_sha256": binding.file_sha256,
                "self_sha256": binding.self_sha256,
            }
            for binding in FROZEN_QWEN_RUNTIME_QUALIFICATION_RECEIPTS
        ],
    }
    measurement_contract = {
        "arm_ids": list(protocol.ARM_IDS),
        "begin_after_freeze": True,
        "effect_gate_added": False,
        "four_arm_barrier_before_label_open": True,
        "offline_scorer_call_count": 1,
        "online_or_api_evaluation_count": 0,
        "retry_or_replay_allowed_after_begin": False,
        "runner_version": VERSION,
    }
    return {
        "formal_runtime_closure": closure.manifest["self_hash"],
        "official_public_source": _content_hash(source_contract),
        "qualification_receipts": _content_hash(receipt_contract),
        "runner_implementation": _hash_regular_file(
            runner_path, maximum_bytes=2 * 1024 * 1024
        ),
        "one_shot_measurement_contract": _content_hash(
            measurement_contract
        ),
    }


def _validate_score_receipt(
    score: Mapping[str, Any],
    *,
    action: supervisor.FrozenAction,
    invocation: supervisor.FormalInvocation,
    barrier: Mapping[str, Any],
) -> None:
    expected_fields = {
        "schema",
        "status",
        "one_shot_key",
        "action_self_hash",
        "four_arm_barrier_self_hash",
        "label_open_claim_self_hash",
        "arm_aggregates",
        "paired_aggregate_differences",
        "uncertainty_method",
        "abstain_and_error_counted_wrong",
        "online_or_api_evaluator_used",
        "effect_gate_added",
        "item_content_emitted",
        "self_hash",
    }
    body = dict(score)
    claimed = body.pop("self_hash", None)
    if (
        set(score) != expected_fields
        or score.get("schema") != supervisor.SCORE_RECEIPT_SCHEMA
        or score.get("status") != "FIXED_OFFLINE_SCORER_COMPLETED"
        or score.get("one_shot_key")
        != invocation.receipt["one_shot_key"]
        or score.get("action_self_hash") != action.receipt["self_hash"]
        or score.get("four_arm_barrier_self_hash")
        != barrier.get("self_hash")
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or _content_hash(body) != claimed
        or set(score.get("arm_aggregates", {}))
        != set(protocol.ARM_IDS)
        or score.get("online_or_api_evaluator_used") is not False
        or score.get("effect_gate_added") is not False
        or score.get("item_content_emitted") is not False
    ):
        raise FormalMeasurementError("aggregate_score_receipt_invalid")


def _success_terminal(
    *,
    action: supervisor.FrozenAction,
    invocation: supervisor.FormalInvocation,
    source_receipt: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    barrier: Mapping[str, Any],
    score: Mapping[str, Any],
    test_attestation: supervisor.TestAttestation,
) -> Mapping[str, Any]:
    _validate_score_receipt(
        score,
        action=action,
        invocation=invocation,
        barrier=barrier,
    )
    if (
        source_receipt.get("source_sha256") != OFFICIAL_DATASET_SHA256
        or source_receipt.get("metadata_sha256") != OFFICIAL_METADATA_SHA256
        or source_receipt.get("adapted_row_count")
        != protocol.OFFICIAL_ROW_COUNT
        or source_receipt.get("item_content_emitted") is not False
        or execution_receipt.get("item_content_emitted") is not False
        or barrier.get("item_content_emitted") is not False
        or barrier.get("label_opened") is not False
    ):
        raise FormalMeasurementError("formal_safe_receipt_invalid")
    body: dict[str, Any] = {
        "schema": SAFE_TERMINAL_SCHEMA,
        "status": "COMPLETED_ONE_SHOT_OFFLINE_ARN_INTRINSIC_MEASUREMENT",
        "version": VERSION,
        "formal_root": str(FORMAL_ROOT),
        "one_shot_key": invocation.receipt["one_shot_key"],
        "action_self_hash": action.receipt["self_hash"],
        "runtime_closure_self_hash": action.closure.manifest["self_hash"],
        "source_binding": {
            "dataset_sha256": OFFICIAL_DATASET_SHA256,
            "dataset_size": OFFICIAL_DATASET_SIZE,
            "doi": OFFICIAL_DOI,
            "license_id": OFFICIAL_LICENSE_ID,
            "metadata_sha256": OFFICIAL_METADATA_SHA256,
            "metadata_size": OFFICIAL_METADATA_SIZE,
            "official_row_count": protocol.OFFICIAL_ROW_COUNT,
        },
        "source_receipt_self_hash": source_receipt["self_hash"],
        "factory_execution_receipt_self_hash": execution_receipt[
            "self_hash"
        ],
        "four_arm_barrier_self_hash": barrier["self_hash"],
        "aggregate_score_receipt_self_hash": score["self_hash"],
        "common_item_count": barrier["common_item_count"],
        "arm_aggregates": score["arm_aggregates"],
        "paired_aggregate_differences": score[
            "paired_aggregate_differences"
        ],
        "uncertainty_method": score["uncertainty_method"],
        "source_free_test_attestation_self_hash": (
            test_attestation.receipt["self_hash"]
        ),
        "formal_attempt_count": 1,
        "offline_scorer_call_count": 1,
        "online_or_api_evaluation_count": 0,
        "retry_or_replay_allowed": False,
        "effect_gate_added": False,
        "item_content_emitted": False,
    }
    return {**body, "self_hash": _content_hash(body)}


def _fixed_issue_id(exc: BaseException) -> str:
    candidate = getattr(exc, "issue_id", None)
    if candidate is None and exc.args:
        candidate = exc.args[0]
    if isinstance(candidate, str) and _ISSUE_ID.fullmatch(candidate):
        return candidate
    return "unexpected_formal_runtime_error"


def _seal_failed_after_begin(
    *,
    runtime: supervisor.FormalSupervisor,
    action: supervisor.FrozenAction,
    invocation: supervisor.FormalInvocation,
    stage: str,
    exc: BaseException,
) -> None:
    body = {
        "schema": FAILED_TERMINAL_SCHEMA,
        "status": "FAILED_AFTER_FORMAL_BEGIN_NO_RETRY_OR_REPLAY",
        "version": VERSION,
        "formal_root": str(FORMAL_ROOT),
        "one_shot_key": invocation.receipt["one_shot_key"],
        "action_self_hash": action.receipt["self_hash"],
        "failed_stage": stage,
        "issue_id": _fixed_issue_id(exc),
        "retry_or_replay_allowed": False,
        "online_or_api_evaluation_count": 0,
        "effect_gate_added": False,
        "item_content_emitted": False,
    }
    terminal = {**body, "self_hash": _content_hash(body)}
    runtime.store.ensure_directory("control")
    runtime.store.write_json_exclusive(
        "control/outer_terminal.safe.json", terminal
    )


def _execute_formal_once(
    *,
    closure: supervisor.RuntimeClosure,
    test_attestation: supervisor.TestAttestation,
) -> Mapping[str, Any]:
    """Execute the sole formal call chain; no caller data enters this method."""

    with supervisor.FormalSupervisor() as runtime:
        action = runtime.freeze_internal_factory_action_once(
            closure=closure,
            freeze_commitments=_freeze_commitments(closure=closure),
            qwen_model_root=FROZEN_QWEN_MODEL_ROOT,
            qwen_model_manifest=FROZEN_QWEN_MODEL_MANIFEST,
            qwen_actual_canary_lineage_terminal=(
                FROZEN_QWEN_ACTUAL_CANARY_LINEAGE_TERMINAL
            ),
            qwen_runtime_qualification_receipts=tuple(
                binding.path
                for binding in (
                    FROZEN_QWEN_RUNTIME_QUALIFICATION_RECEIPTS
                )
            ),
            internal_factory_qualification_receipt=(
                FROZEN_INTERNAL_FACTORY_QUALIFICATION_RECEIPT.path
            ),
            minilm_model_root=FROZEN_MINILM_MODEL_ROOT,
            minilm_asset_manifest=FROZEN_MINILM_ASSET_MANIFEST,
            minilm_target_manifest=FROZEN_MINILM_TARGET_MANIFEST,
        )
        invocation: supervisor.FormalInvocation | None = None
        stage = "begin_once"
        try:
            invocation = runtime.begin_once(action)
            stage = "materialize_official_packs_once"
            source_receipt = runtime.materialize_official_packs_once(
                invocation
            )
            stage = "run_internal_factory_once"
            execution_receipt = runtime.run_internal_factory_once(
                invocation
            )
            stage = "seal_four_arm_barrier_once"
            barrier = runtime.seal_four_arm_barrier_once(invocation)
            stage = "run_fixed_scorer_once"
            score = runtime.run_fixed_scorer_once(invocation)
            stage = "seal_safe_aggregate_terminal"
            terminal = _success_terminal(
                action=action,
                invocation=invocation,
                source_receipt=source_receipt,
                execution_receipt=execution_receipt,
                barrier=barrier,
                score=score,
                test_attestation=test_attestation,
            )
            runtime.store.ensure_directory("control")
            runtime.store.write_json_exclusive(
                "control/outer_terminal.safe.json", terminal
            )
            return terminal
        except Exception as exc:
            if invocation is not None:
                try:
                    _seal_failed_after_begin(
                        runtime=runtime,
                        action=action,
                        invocation=invocation,
                        stage=stage,
                        exc=exc,
                    )
                except Exception:
                    pass
            raise


def run_measurement() -> Mapping[str, Any]:
    """Run the fixed formal measurement once after all pre-root checks pass."""

    _validate_source_constants()
    _preflight_staged_source_without_opening_content()
    _preflight_never_started()
    qwen_receipts = _validate_fixed_assets_and_receipts()
    qualification._preflight_frozen_main_runtime()  # noqa: SLF001
    qualification._preflight_frozen_runtime_binding_manifest()  # noqa: SLF001
    qualification._preflight_fixed_source_free_test_runtime()  # noqa: SLF001
    try:
        qualification._preflight_exactly_two_idle_gpus()  # noqa: SLF001
    except qualification.QualificationDeferred as exc:
        raise FormalMeasurementDeferred(exc.issue_id) from exc
    closure, test_attestation = _source_free_runtime_closure(
        qwen_receipts=qwen_receipts
    )
    # The shared node can change while source-free tests execute.  This second
    # check still occurs before action freeze, secret creation, and begin.
    try:
        qualification._preflight_exactly_two_idle_gpus()  # noqa: SLF001
    except qualification.QualificationDeferred as exc:
        raise FormalMeasurementDeferred(exc.issue_id) from exc
    _preflight_never_started()
    return _execute_formal_once(
        closure=closure,
        test_attestation=test_attestation,
    )


def _main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        print(f"{VERSION} accepts no arguments", file=sys.stderr)
        return 2
    try:
        run_measurement()
    except FormalMeasurementDeferred as exc:
        print(
            f"{VERSION} deferred before formal attempt: {exc.issue_id}",
            file=sys.stderr,
        )
        return DEFERRED_EXIT_CODE
    except (
        FormalMeasurementError,
        qualification.QualificationRunnerError,
        supervisor.FormalSupervisorError,
    ) as exc:
        print(
            f"{VERSION} failed closed: {_fixed_issue_id(exc)}",
            file=sys.stderr,
        )
        return 2
    return 0


__all__ = [
    "DEFERRED_EXIT_CODE",
    "FAILED_TERMINAL_SCHEMA",
    "FORMAL_ROOT",
    "FormalMeasurementDeferred",
    "FormalMeasurementError",
    "FROZEN_INTERNAL_FACTORY_QUALIFICATION_RECEIPT",
    "FROZEN_QWEN_RUNTIME_QUALIFICATION_RECEIPTS",
    "OFFICIAL_DATASET_SHA256",
    "OFFICIAL_METADATA_SHA256",
    "SAFE_TERMINAL_SCHEMA",
    "VERSION",
    "run_measurement",
]


if __name__ == "__main__":
    raise SystemExit(_main())
