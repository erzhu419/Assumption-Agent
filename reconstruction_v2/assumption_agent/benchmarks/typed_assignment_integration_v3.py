from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import secrets
import shutil
import tempfile
import threading
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

from ..events import Event
from ..models import stable_hash
from ..typed_assignment_contract_v3 import PublicDestinationSpec
from .execution_contract_integration_v2 import (
    EXECUTION_CONTRACT_RUNTIME_VERSION,
    ExecutionContractRuntimeContextV2,
    ExecutionContractSubprocessBackendV2,
    ExecutionContractTrialEvidenceV2,
)
from .execution_contract_prompt_v2 import (
    EXECUTION_CONTRACT_PROMPT_CONTAINER_PATH,
    ExecutionContractPromptInjectionReceiptV2,
)
from .skilllearn_lifecycle import (
    DockerEgressPolicy,
    OfflineVerifierRuntime,
    SkillLearnAgentTerminalError,
    SkillLearnTrialRequest,
    TrialVariant,
)
from .typed_task_capability import (
    INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY,
)
from .typed_assignment_runtime_tool_v3 import (
    TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
)


TYPED_ASSIGNMENT_RUNTIME_VERSION = (
    "content_evidence_typed_plan_harness_apply_reconcile_v3"
)
TYPED_ASSIGNMENT_PROMPT_DELIVERY_VERSION = (
    "verified_typed_assignment_plan_only_launch_prompt_v3"
)
TYPED_ASSIGNMENT_PROMPT_CONTAINER_PATH = (
    "/tmp/assumption-v2-typed-assignment-context-v3.txt"
)
TYPED_ASSIGNMENT_TOOL_CONTAINER_PATH = (
    "/tmp/assumption-v2-typed-assignment-runtime-v3.py"
)
TYPED_ASSIGNMENT_FRESH_TOOL_CONTAINER_PREFIX = (
    "/tmp/.assumption-v2-typed-assignment-runtime-v3-"
)
TYPED_ASSIGNMENT_PUBLIC_INSTRUCTION_CONTAINER_PATH = (
    "/tmp/assumption-v2-typed-assignment-public-instruction-v3.txt"
)
TYPED_ASSIGNMENT_SIDECAR_CONTAINER_PATH = (
    "/root/.assumption-v2/typed-assignment-v3"
)
TYPED_ASSIGNMENT_EVIDENCE_CONTAINER_PATH = (
    f"{TYPED_ASSIGNMENT_SIDECAR_CONTAINER_PATH}/evidence.json"
)
TYPED_ASSIGNMENT_PLAN_CONTAINER_PATH = (
    f"{TYPED_ASSIGNMENT_SIDECAR_CONTAINER_PATH}/plan.json"
)
TYPED_ASSIGNMENT_PREPARE_RECEIPT_CONTAINER_PATH = (
    f"{TYPED_ASSIGNMENT_SIDECAR_CONTAINER_PATH}/prepare_receipt.json"
)
TYPED_ASSIGNMENT_RECONCILIATION_RECEIPT_CONTAINER_PATH = (
    f"{TYPED_ASSIGNMENT_SIDECAR_CONTAINER_PATH}/reconciliation_receipt.json"
)
MAX_TYPED_ASSIGNMENT_PROMPT_BYTES = 128 * 1024
MAX_TYPED_ASSIGNMENT_RECEIPT_BYTES = 512 * 1024

_PREPARE_RECEIPT_FIELDS = frozenset(
    {
        "runtime_policy",
        "runtime_tool_sha256",
        "contract_hash",
        "evidence_set_hash",
        "evidence_file_sha256",
        "pre_manifest_hash",
        "pre_manifest_file_sha256",
        "plan_schema_file_sha256",
        "prepare_state_file_sha256",
        "public_instruction_sha256",
        "destination_set_hash",
        "file_count",
        "evidence_count",
        "extraction_unavailable_count",
        "container_evidence_profile_persisted",
        "raw_public_instruction_in_receipt",
        "raw_content_evidence_in_receipt",
        "source_filenames_in_receipt",
        "host_safe_receipt",
        "receipt_hash",
    }
)
_RECONCILIATION_RECEIPT_FIELDS = frozenset(
    {
        "runtime_policy",
        "runtime_tool_sha256",
        "mode",
        "contract_hash",
        "evidence_set_hash",
        "prepare_receipt_file_sha256",
        "plan_file_sha256",
        "normalized_plan_hash",
        "assignment_count",
        "positive_evidence_assignment_count",
        "public_default_assignment_count",
        "reopened_file_count",
        "source_empty",
        "destination_set_hash",
        "destination_distribution_hash",
        "final_task_manifest_hash",
        "all_destination_content_hashes_match",
        "transactional_apply",
        "rollback_required",
        "raw_public_instruction_in_receipt",
        "raw_content_evidence_in_receipt",
        "source_filenames_in_receipt",
        "host_safe_receipt",
        "receipt_hash",
    }
)

_V2_RUN_TEMPLATE_FRAGMENT = (
    f"$(cat {{instruction_file}} {EXECUTION_CONTRACT_PROMPT_CONTAINER_PATH})"
)
_TYPED_RUN_TEMPLATE_FRAGMENT = (
    "$(cat {instruction_file} "
    f"{EXECUTION_CONTRACT_PROMPT_CONTAINER_PATH} "
    f"{TYPED_ASSIGNMENT_PROMPT_CONTAINER_PATH})"
)


class TypedAssignmentIntegrationError(PermissionError):
    """The v3 typed-assignment runtime crossed its registered boundary."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_tool_source() -> Path:
    source = Path(__file__).resolve().with_name(
        "typed_assignment_runtime_tool_v3.py"
    )
    if source.is_symlink() or not source.is_file():
        raise TypedAssignmentIntegrationError(
            "typed-assignment runtime tool source is unavailable"
        )
    return source


def _fresh_runtime_tool_container_path() -> str:
    # The name is chosen only after agent exit.  The exited agent therefore
    # cannot prepare a replacement at the path the harness will execute.
    return (
        TYPED_ASSIGNMENT_FRESH_TOOL_CONTAINER_PREFIX
        + secrets.token_hex(24)
        + ".py"
    )


def _runtime_component_paths() -> tuple[Path, ...]:
    package_root = Path(__file__).resolve().parents[1]
    return (
        package_root / "typed_assignment_contract_v3.py",
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name(
            "typed_assignment_runtime_tool_v3.py"
        ),
    )


def typed_assignment_runtime_class_hash() -> str:
    rows: list[dict[str, str]] = []
    for path in _runtime_component_paths():
        if path.is_symlink() or not path.is_file():
            raise TypedAssignmentIntegrationError(
                "typed-assignment runtime component is unavailable"
            )
        rows.append(
            {
                "component": path.name,
                "sha256": _sha256_file(path),
            }
        )
    return stable_hash(
        {
            "runtime_version": TYPED_ASSIGNMENT_RUNTIME_VERSION,
            "components": sorted(rows, key=lambda row: row["component"]),
        }
    )


# This value is deliberately computed from the complete implementation bytes.
# A candidate/preregistration therefore cannot silently swap the runtime class.
TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH = typed_assignment_runtime_class_hash()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_bounded_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TypedAssignmentIntegrationError(
            "typed-assignment receipt is unavailable"
        )
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_TYPED_ASSIGNMENT_RECEIPT_BYTES:
        raise TypedAssignmentIntegrationError(
            "typed-assignment receipt is outside its byte bound"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TypedAssignmentIntegrationError(
            "typed-assignment receipt is not JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise TypedAssignmentIntegrationError(
            "typed-assignment receipt is malformed"
        )
    return payload


def _receipt_digest(payload: Mapping[str, Any]) -> str:
    without_hash = dict(payload)
    embedded = without_hash.pop("receipt_hash", None)
    computed = stable_hash(without_hash)
    if not _is_sha256(embedded) or embedded != computed:
        raise TypedAssignmentIntegrationError(
            "typed-assignment receipt hash is missing or drifted"
        )
    return computed


def _is_nonnegative_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _validate_prepare_receipt_v3(
    payload: Mapping[str, Any],
    *,
    expected_runtime_tool_sha256: str,
    expected_public_instruction_sha256: str,
) -> str:
    if set(payload) != _PREPARE_RECEIPT_FIELDS:
        raise TypedAssignmentIntegrationError(
            "typed-assignment prepare receipt schema drifted"
        )
    receipt_hash = _receipt_digest(payload)
    sha_fields = (
        "runtime_tool_sha256",
        "contract_hash",
        "evidence_set_hash",
        "evidence_file_sha256",
        "pre_manifest_hash",
        "pre_manifest_file_sha256",
        "plan_schema_file_sha256",
        "prepare_state_file_sha256",
        "public_instruction_sha256",
        "destination_set_hash",
    )
    file_count = payload.get("file_count")
    evidence_count = payload.get("evidence_count")
    unavailable_count = payload.get("extraction_unavailable_count")
    if (
        payload.get("runtime_policy")
        != TYPED_ASSIGNMENT_RUNTIME_POLICY_V3
        or any(not _is_sha256(payload.get(key)) for key in sha_fields)
        or payload.get("runtime_tool_sha256")
        != expected_runtime_tool_sha256
        or payload.get("public_instruction_sha256")
        != expected_public_instruction_sha256
        or not _is_nonnegative_int(file_count)
        or file_count == 0
        or not _is_nonnegative_int(evidence_count)
        or not _is_nonnegative_int(unavailable_count)
        or evidence_count + unavailable_count != file_count
        or payload.get("container_evidence_profile_persisted") is not True
        or payload.get("raw_public_instruction_in_receipt") is not False
        or payload.get("raw_content_evidence_in_receipt") is not False
        or payload.get("source_filenames_in_receipt") is not False
        or payload.get("host_safe_receipt") is not True
    ):
        raise TypedAssignmentIntegrationError(
            "typed-assignment prepare receipt is invalid"
        )
    return receipt_hash


def _validate_reconciliation_receipt_v3(
    payload: Mapping[str, Any],
    *,
    prepare_payload: Mapping[str, Any],
    expected_prepare_receipt_file_sha256: str,
    expected_runtime_tool_sha256: str,
) -> str:
    if set(payload) != _RECONCILIATION_RECEIPT_FIELDS:
        raise TypedAssignmentIntegrationError(
            "typed-assignment reconciliation receipt schema drifted"
        )
    receipt_hash = _receipt_digest(payload)
    sha_fields = (
        "runtime_tool_sha256",
        "contract_hash",
        "evidence_set_hash",
        "prepare_receipt_file_sha256",
        "plan_file_sha256",
        "normalized_plan_hash",
        "destination_set_hash",
        "destination_distribution_hash",
        "final_task_manifest_hash",
    )
    file_count = prepare_payload.get("file_count")
    assignment_count = payload.get("assignment_count")
    positive_count = payload.get("positive_evidence_assignment_count")
    default_count = payload.get("public_default_assignment_count")
    reopened_count = payload.get("reopened_file_count")
    if (
        any(not _is_sha256(payload.get(key)) for key in sha_fields)
        or payload.get("runtime_policy")
        != TYPED_ASSIGNMENT_RUNTIME_POLICY_V3
        or payload.get("runtime_tool_sha256")
        != expected_runtime_tool_sha256
        or payload.get("runtime_tool_sha256")
        != prepare_payload.get("runtime_tool_sha256")
        or payload.get("mode") != "apply_and_reconcile"
        or payload.get("contract_hash")
        != prepare_payload.get("contract_hash")
        or payload.get("evidence_set_hash")
        != prepare_payload.get("evidence_set_hash")
        or payload.get("prepare_receipt_file_sha256")
        != expected_prepare_receipt_file_sha256
        or payload.get("destination_set_hash")
        != prepare_payload.get("destination_set_hash")
        or not _is_nonnegative_int(file_count)
        or not _is_nonnegative_int(assignment_count)
        or not _is_nonnegative_int(positive_count)
        or not _is_nonnegative_int(default_count)
        or not _is_nonnegative_int(reopened_count)
        or assignment_count != file_count
        or reopened_count != file_count
        or positive_count + default_count != assignment_count
        or payload.get("source_empty") is not True
        or payload.get("all_destination_content_hashes_match") is not True
        or payload.get("transactional_apply") is not True
        or payload.get("rollback_required") is not False
        or payload.get("raw_public_instruction_in_receipt") is not False
        or payload.get("raw_content_evidence_in_receipt") is not False
        or payload.get("source_filenames_in_receipt") is not False
        or payload.get("host_safe_receipt") is not True
    ):
        raise TypedAssignmentIntegrationError(
            "typed-assignment reconciliation receipt is invalid"
        )
    return receipt_hash


@dataclass(frozen=True)
class TypedAssignmentPromptInjectionReceiptV3:
    request_hash: str
    base_execution_prompt_receipt_hash: str
    runtime_class_hash: str
    prepare_receipt_hash: str
    contract_hash: str
    evidence_set_hash: str
    prompt_fragment_sha256: str
    prompt_fragment_size: int
    container_readback_sha256: str
    run_template_before_hash: str
    run_template_after_hash: str
    effective_prompt_sha256: str

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "delivery_policy": TYPED_ASSIGNMENT_PROMPT_DELIVERY_VERSION,
            "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_VERSION,
            "request_hash": self.request_hash,
            "base_execution_prompt_receipt_hash": (
                self.base_execution_prompt_receipt_hash
            ),
            "runtime_class_hash": self.runtime_class_hash,
            "prepare_receipt_hash": self.prepare_receipt_hash,
            "contract_hash": self.contract_hash,
            "evidence_set_hash": self.evidence_set_hash,
            "prompt_fragment_sha256": self.prompt_fragment_sha256,
            "prompt_fragment_size": self.prompt_fragment_size,
            "container_path_hash": stable_hash(
                {"path": TYPED_ASSIGNMENT_PROMPT_CONTAINER_PATH}
            ),
            "container_readback_sha256": self.container_readback_sha256,
            "run_template_before_hash": self.run_template_before_hash,
            "run_template_after_hash": self.run_template_after_hash,
            "effective_prompt_sha256": self.effective_prompt_sha256,
            "agent_started_at_receipt_time": False,
            "model_invoked_at_receipt_time": False,
            "plan_only_agent_authority": True,
            "harness_apply_required": True,
            "verifier_content_accessed": False,
            "validation_or_test_content_accessed": False,
            "raw_document_content_persisted_host_side": False,
        }


@dataclass(frozen=True)
class TypedAssignmentRuntimeEvidenceV3:
    request_hash: str
    runtime_class_hash: str
    prompt_receipt_hash: str
    prepare_receipt_hash: str
    reconciliation_receipt_hash: str
    contract_hash: str
    evidence_set_hash: str
    file_count: int
    prepare_receipt_body: Mapping[str, Any]
    reconciliation_receipt_body: Mapping[str, Any]
    post_agent_runtime_tool_sha256: str
    post_agent_runtime_tool_readback_sha256: str
    post_agent_runtime_tool_container_path_hash: str

    @property
    def evidence_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_VERSION,
            "request_hash": self.request_hash,
            "runtime_class_hash": self.runtime_class_hash,
            "prompt_receipt_hash": self.prompt_receipt_hash,
            "prepare_receipt_hash": self.prepare_receipt_hash,
            "reconciliation_receipt_hash": (
                self.reconciliation_receipt_hash
            ),
            "contract_hash": self.contract_hash,
            "evidence_set_hash": self.evidence_set_hash,
            "file_count": self.file_count,
            "prepare_receipt_body": dict(self.prepare_receipt_body),
            "reconciliation_receipt_body": dict(
                self.reconciliation_receipt_body
            ),
            "post_agent_runtime_delivery": {
                "runtime_tool_sha256": self.post_agent_runtime_tool_sha256,
                "container_readback_sha256": (
                    self.post_agent_runtime_tool_readback_sha256
                ),
                "container_path_hash": (
                    self.post_agent_runtime_tool_container_path_hash
                ),
                "fresh_unpredictable_path_selected_after_agent_exit": True,
                "pre_agent_prepare_tool_removed_before_agent_start": True,
            },
            "agent_wrote_plan_only": True,
            "harness_applied_plan": True,
            "post_apply_reconciliation_passed": True,
            "reconciliation_completed_before_verifier_invocation": True,
            "verifier_invoked_at_receipt_time": False,
            "verifier_materialized_at_receipt_time": False,
            "validation_or_test_content_accessed": False,
            "online_judge_calls": 0,
            "raw_document_content_persisted_host_side": False,
        }


@dataclass
class _TypedAssignmentRunStateV3:
    request_hash: str
    context: ExecutionContractRuntimeContextV2
    source_locator: str = ""
    task_root: str = ""
    prepare_payload: Mapping[str, Any] | None = None
    prepare_receipt_hash: str = ""
    prepare_receipt_file_sha256: str = ""
    destination_spec: PublicDestinationSpec | None = None
    prompt_receipt: TypedAssignmentPromptInjectionReceiptV3 | None = None
    runtime_evidence: TypedAssignmentRuntimeEvidenceV3 | None = None
    verifier_triggered: bool = False


def _prompt_fragment(
    *,
    request_hash: str,
    runtime_class_hash: str,
    prepare_payload: Mapping[str, Any],
    destination_spec: PublicDestinationSpec,
) -> bytes:
    contract_hash = prepare_payload.get("contract_hash")
    evidence_set_hash = prepare_payload.get("evidence_set_hash")
    destination_spec.verify()
    destinations = list(destination_spec.destinations)
    default_destination = destination_spec.default_destination
    if (
        not _is_sha256(contract_hash)
        or not _is_sha256(evidence_set_hash)
        or not isinstance(destinations, list)
        or not destinations
        or any(not isinstance(value, str) or not value for value in destinations)
        or default_destination not in {*destinations, None}
    ):
        raise TypedAssignmentIntegrationError(
            "typed-assignment prepare receipt lacks its closed contract"
        )
    envelope = {
        "delivery_policy": TYPED_ASSIGNMENT_PROMPT_DELIVERY_VERSION,
        "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_VERSION,
        "request_hash": request_hash,
        "runtime_class_hash": runtime_class_hash,
        "contract_hash": contract_hash,
        "evidence_set_hash": evidence_set_hash,
        "allowed_destinations": destinations,
        "public_default_destination": default_destination,
        "evidence_path": TYPED_ASSIGNMENT_EVIDENCE_CONTAINER_PATH,
        "plan_path": TYPED_ASSIGNMENT_PLAN_CONTAINER_PATH,
        "required_plan_schema": {
            "contract_hash": "<exact contract_hash above>",
            "evidence_set_hash": "<exact evidence_set_hash above>",
            "assignments": [
                {
                    "file_id": "<exact file_id from evidence.json>",
                    "destination": "<one allowed destination>",
                    "basis": (
                        "positive_content_evidence or public_default"
                    ),
                    "evidence_ids": [
                        "<same-file evidence id; empty only for public_default>"
                    ],
                }
            ],
        },
        "compiler_owned_instructions": [
            (
                "Read the complete evidence JSON and assign every listed "
                "file_id exactly once. Base non-default assignments on the "
                "same-file extracted content evidence and cite its evidence_id."
            ),
            (
                "Use public_default only when the public task's explicit "
                "default rule applies; then select the registered default "
                "destination and use an empty evidence_ids list."
            ),
            (
                "Write exactly one JSON plan at plan_path. Assignment array "
                "order is immaterial. Do not create destination folders and "
                "do not move, rename, edit, or delete any task file."
            ),
            (
                "The harness, not the agent, validates and applies the plan "
                "after the agent exits, reopens the result, and reconciles "
                "the exact one-to-one mapping before the offline verifier."
            ),
            (
                "Do not inspect tests, solutions, verifier files, or any "
                "evaluation artifact. Finish immediately after the plan is "
                "written and locally parsed as valid JSON."
            ),
        ],
        "model_authored_instruction_fields": [],
        "runtime_enforcement_claimed": True,
        "online_judge_calls": 0,
    }
    fragment = (
        "\n\n[ASSUMPTION_V2_TYPED_ASSIGNMENT_CONTEXT_V3]\n"
        "This block is a harness-verified, plan-only runtime contract.\n"
        + _canonical_json_bytes(envelope).decode("utf-8")
        + "[/ASSUMPTION_V2_TYPED_ASSIGNMENT_CONTEXT_V3]\n"
    ).encode("utf-8")
    if not fragment or len(fragment) > MAX_TYPED_ASSIGNMENT_PROMPT_BYTES:
        raise TypedAssignmentIntegrationError(
            "typed-assignment prompt fragment exceeds its byte bound"
        )
    return fragment


class _TypedAssignmentVerifierProxyV3:
    def __init__(
        self,
        delegate: Any,
        *,
        backend: "TypedAssignmentExecutionContractSubprocessBackendV3",
    ) -> None:
        self.delegate = delegate
        self.backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def run(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        command = list(args) if isinstance(args, (list, tuple)) else args
        if (
            isinstance(command, list)
            and len(command) >= 4
            and command[:2] == ["docker", "exec"]
            and "/tests/test.sh" in {str(value) for value in command[3:]}
        ):
            self.backend._apply_and_reconcile_typed_assignment_v3(
                delegate=self.delegate,
                container_name=str(command[2]),
            )
        return self.delegate.run(command, *positional, **kwargs)


class TypedAssignmentExecutionContractSubprocessBackendV3(
    ExecutionContractSubprocessBackendV2
):
    """Execute a closed typed plan between agent exit and offline scoring.

    The agent receives bounded content evidence and can author only a typed
    assignment plan.  A standard-library harness validates the exact bijection,
    applies it, and reconciles the reopened tree before the existing verifier
    isolation proxy is allowed to materialize ``/tests``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._typed_assignment_local = threading.local()
        self._typed_assignment_evidence_lock = threading.Lock()
        self._typed_assignment_evidence: list[
            TypedAssignmentRuntimeEvidenceV3
        ] = []

    @property
    def execution_backend_instance_hash(self) -> str:
        return stable_hash(
            {
                "base_execution_backend_instance_hash": (
                    super().execution_backend_instance_hash
                ),
                "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_VERSION,
                "runtime_class_hash": TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH,
            }
        )

    @property
    def typed_assignment_evidence(self) -> tuple[
        TypedAssignmentRuntimeEvidenceV3, ...
    ]:
        with self._typed_assignment_evidence_lock:
            return tuple(self._typed_assignment_evidence)

    def _load_portable_task_capability_context(
        self,
        *,
        request: SkillLearnTrialRequest,
        source_receipt: Any,
        compile_root: Path,
    ) -> Any:
        context = super()._load_portable_task_capability_context(
            request=request,
            source_receipt=source_receipt,
            compile_root=compile_root,
        )
        self._typed_assignment_local.state = None
        execution_context = getattr(
            self._execution_contract_local,
            "context",
            None,
        )
        if (
            request.variant is TrialVariant.POLICY_ON
            and isinstance(
                execution_context,
                ExecutionContractRuntimeContextV2,
            )
        ):
            self._typed_assignment_local.state = _TypedAssignmentRunStateV3(
                request_hash=request.request_hash,
                context=execution_context,
            )
        return context

    def _install_treatment_receipt_adapter(self, runner: ModuleType) -> None:
        super()._install_treatment_receipt_adapter(runner)
        base_inject = runner._inject_skills_runtime

        def inject_with_typed_assignment(
            container_name: str,
            skill_source_dir: Path,
            copies: list[tuple[str, str]],
        ) -> None:
            runner._assumption_v2_typed_assignment_prompt_receipt = None
            base_inject(container_name, skill_source_dir, copies)
            state = getattr(self._typed_assignment_local, "state", None)
            if not isinstance(state, _TypedAssignmentRunStateV3):
                return
            try:
                self._prepare_and_inject_typed_assignment_v3(
                    runner=runner,
                    container_name=container_name,
                    state=state,
                )
            except Exception as exc:
                runner._assumption_v2_typed_assignment_prompt_receipt = None
                self.event_sink.emit(
                    Event(
                        event=(
                            "skilllearn_trial_blocked_invalid_"
                            "typed_assignment_pre_agent_v3"
                        ),
                        stage="benchmark.skilllearn.typed_assignment_v3",
                        trace_id=state.request_hash[:20],
                        payload={
                            "request_hash": state.request_hash,
                            "runtime_class_hash": (
                                TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH
                            ),
                            "error_type": type(exc).__name__,
                            "agent_started": False,
                            "model_invoked": False,
                            "raw_content_persisted": False,
                        },
                    )
                )
                if isinstance(exc, SkillLearnAgentTerminalError):
                    raise
                raise SkillLearnAgentTerminalError(
                    "typed_assignment_pre_agent_invalid"
                ) from exc

        runner._inject_skills_runtime = inject_with_typed_assignment
        runner._assumption_v2_typed_assignment_prompt_receipt = None

    def _collection_locator(
        self,
        *,
        runner: ModuleType,
        container_name: str,
        context: ExecutionContractRuntimeContextV2,
    ) -> str:
        metadata_rows = tuple(
            row
            for row in context.base_metadata
            if row.role_spec.capability
            == INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY
        )
        if len(metadata_rows) != 1:
            raise TypedAssignmentIntegrationError(
                "typed-assignment route lacks one collection capability"
            )
        candidates = self._discover_portable_container_collection_locators(
            delegate=runner.subprocess,
            container_name=container_name,
            metadata=metadata_rows[0],
        )
        if len(candidates) != 1:
            raise TypedAssignmentIntegrationError(
                "typed-assignment source collection is ambiguous"
            )
        locator = candidates[0]
        path = PurePosixPath(locator)
        if not path.is_absolute() or path.parent == PurePosixPath("/"):
            raise TypedAssignmentIntegrationError(
                "typed-assignment source collection is unsafe"
            )
        return locator

    @staticmethod
    def _docker_run_checked(delegate: Any, command: Sequence[str]) -> Any:
        result = delegate.run(
            list(command),
            capture_output=True,
            text=True,
        )
        if int(getattr(result, "returncode", 1)) != 0:
            raise TypedAssignmentIntegrationError(
                "typed-assignment container command failed"
            )
        return result

    def _copy_container_json(
        self,
        *,
        delegate: Any,
        container_name: str,
        container_path: str,
        host_root: Path,
        name: str,
    ) -> dict[str, Any]:
        target = host_root / name
        self._docker_run_checked(
            delegate,
            [
                "docker",
                "cp",
                f"{container_name}:{container_path}",
                str(target),
            ],
        )
        return _load_bounded_json(target)

    def _install_fresh_runtime_tool_v3(
        self,
        *,
        delegate: Any,
        container_name: str,
        container_path: str,
        host_root: Path,
    ) -> tuple[str, str]:
        if not container_path.startswith(
            TYPED_ASSIGNMENT_FRESH_TOOL_CONTAINER_PREFIX
        ) or not container_path.endswith(".py"):
            raise TypedAssignmentIntegrationError(
                "typed-assignment fresh runtime path is invalid"
            )
        tool_source = _runtime_tool_source()
        source_bytes = tool_source.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        self._docker_run_checked(
            delegate,
            [
                "docker",
                "exec",
                container_name,
                "test",
                "!",
                "-e",
                container_path,
            ],
        )
        self._docker_run_checked(
            delegate,
            [
                "docker",
                "cp",
                str(tool_source),
                f"{container_name}:{container_path}",
            ],
        )
        self._docker_run_checked(
            delegate,
            [
                "docker",
                "exec",
                container_name,
                "chmod",
                "0500",
                container_path,
            ],
        )
        readback = host_root / "fresh-runtime-tool-readback.py"
        self._docker_run_checked(
            delegate,
            [
                "docker",
                "cp",
                f"{container_name}:{container_path}",
                str(readback),
            ],
        )
        readback_bytes = readback.read_bytes()
        readback_sha256 = hashlib.sha256(readback_bytes).hexdigest()
        if readback_bytes != source_bytes or readback_sha256 != source_sha256:
            raise TypedAssignmentIntegrationError(
                "typed-assignment fresh runtime readback drifted"
            )
        return source_sha256, readback_sha256

    def _prepare_and_inject_typed_assignment_v3(
        self,
        *,
        runner: ModuleType,
        container_name: str,
        state: _TypedAssignmentRunStateV3,
    ) -> None:
        base_receipt = getattr(
            runner,
            "_assumption_v2_execution_contract_prompt_receipt",
            None,
        )
        if not isinstance(base_receipt, ExecutionContractPromptInjectionReceiptV2):
            raise TypedAssignmentIntegrationError(
                "typed assignment lacks the base execution prompt receipt"
            )
        source_locator = self._collection_locator(
            runner=runner,
            container_name=container_name,
            context=state.context,
        )
        task_root = str(PurePosixPath(source_locator).parent)
        state.source_locator = source_locator
        state.task_root = task_root

        receipt_root = Path(
            tempfile.mkdtemp(prefix="skilllearn_typed_assignment_v3-")
        )
        try:
            tool_source = _runtime_tool_source()
            runtime_tool_sha256 = _sha256_file(tool_source)
            instruction_source = receipt_root / "public-instruction.txt"
            instruction_source.write_text(
                state.context.public_instruction,
                encoding="utf-8",
            )
            self._docker_run_checked(
                runner.subprocess,
                [
                    "docker",
                    "cp",
                    str(tool_source),
                    f"{container_name}:{TYPED_ASSIGNMENT_TOOL_CONTAINER_PATH}",
                ],
            )
            self._docker_run_checked(
                runner.subprocess,
                [
                    "docker",
                    "cp",
                    str(instruction_source),
                    (
                        f"{container_name}:"
                        f"{TYPED_ASSIGNMENT_PUBLIC_INSTRUCTION_CONTAINER_PATH}"
                    ),
                ],
            )
            self._docker_run_checked(
                runner.subprocess,
                [
                    "docker",
                    "exec",
                    container_name,
                    "python3",
                    TYPED_ASSIGNMENT_TOOL_CONTAINER_PATH,
                    "prepare",
                    "--task-root",
                    task_root,
                    "--source-dir",
                    source_locator,
                    "--public-instruction-file",
                    TYPED_ASSIGNMENT_PUBLIC_INSTRUCTION_CONTAINER_PATH,
                    "--sidecar-dir",
                    TYPED_ASSIGNMENT_SIDECAR_CONTAINER_PATH,
                ],
            )
            # The prepare copy is never the copy executed after agent exit.
            # Remove it before any model invocation so an agent-authored
            # replacement at this known path is inert.
            self._docker_run_checked(
                runner.subprocess,
                [
                    "docker",
                    "exec",
                    container_name,
                    "rm",
                    "-f",
                    "--",
                    TYPED_ASSIGNMENT_TOOL_CONTAINER_PATH,
                ],
            )
            prepare_payload = self._copy_container_json(
                delegate=runner.subprocess,
                container_name=container_name,
                container_path=(
                    TYPED_ASSIGNMENT_PREPARE_RECEIPT_CONTAINER_PATH
                ),
                host_root=receipt_root,
                name="prepare-receipt.json",
            )
            prepare_receipt_file_sha256 = _sha256_file(
                receipt_root / "prepare-receipt.json"
            )
            prepare_receipt_hash = _validate_prepare_receipt_v3(
                prepare_payload,
                expected_runtime_tool_sha256=runtime_tool_sha256,
                expected_public_instruction_sha256=hashlib.sha256(
                    state.context.public_instruction.encode("utf-8")
                ).hexdigest(),
            )
            destination_spec = PublicDestinationSpec.from_public_instruction(
                state.context.public_instruction
            )
            state.prepare_payload = prepare_payload
            state.prepare_receipt_hash = prepare_receipt_hash
            state.prepare_receipt_file_sha256 = (
                prepare_receipt_file_sha256
            )
            state.destination_spec = destination_spec

            fragment = _prompt_fragment(
                request_hash=state.request_hash,
                runtime_class_hash=TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH,
                prepare_payload=prepare_payload,
                destination_spec=destination_spec,
            )
            prompt_source = receipt_root / "prompt-fragment.txt"
            prompt_source.write_bytes(fragment)
            self._docker_run_checked(
                runner.subprocess,
                [
                    "docker",
                    "cp",
                    str(prompt_source),
                    f"{container_name}:{TYPED_ASSIGNMENT_PROMPT_CONTAINER_PATH}",
                ],
            )
            self._docker_run_checked(
                runner.subprocess,
                [
                    "docker",
                    "exec",
                    container_name,
                    "chmod",
                    "0444",
                    TYPED_ASSIGNMENT_PUBLIC_INSTRUCTION_CONTAINER_PATH,
                    TYPED_ASSIGNMENT_PROMPT_CONTAINER_PATH,
                    TYPED_ASSIGNMENT_EVIDENCE_CONTAINER_PATH,
                    TYPED_ASSIGNMENT_PREPARE_RECEIPT_CONTAINER_PATH,
                ],
            )
            readback = receipt_root / "prompt-readback.txt"
            self._docker_run_checked(
                runner.subprocess,
                [
                    "docker",
                    "cp",
                    (
                        f"{container_name}:"
                        f"{TYPED_ASSIGNMENT_PROMPT_CONTAINER_PATH}"
                    ),
                    str(readback),
                ],
            )
            if readback.read_bytes() != fragment:
                raise TypedAssignmentIntegrationError(
                    "typed-assignment prompt readback drifted"
                )
            agent = runner.get_agent(self.agent_id)
            if not isinstance(agent, dict):
                raise TypedAssignmentIntegrationError(
                    "typed-assignment agent definition is unavailable"
                )
            run_before = str(agent.get("run") or "")
            if run_before.count(_V2_RUN_TEMPLATE_FRAGMENT) != 1:
                raise TypedAssignmentIntegrationError(
                    "typed-assignment prompt has no unique base binding"
                )
            run_after = run_before.replace(
                _V2_RUN_TEMPLATE_FRAGMENT,
                _TYPED_RUN_TEMPLATE_FRAGMENT,
                1,
            )
            if (
                run_after == run_before
                or run_after.count(TYPED_ASSIGNMENT_PROMPT_CONTAINER_PATH) != 1
            ):
                raise TypedAssignmentIntegrationError(
                    "typed-assignment final prompt binding drifted"
                )
            base_fragment = receipt_root / "base-prompt-readback.txt"
            self._docker_run_checked(
                runner.subprocess,
                [
                    "docker",
                    "cp",
                    (
                        f"{container_name}:"
                        f"{EXECUTION_CONTRACT_PROMPT_CONTAINER_PATH}"
                    ),
                    str(base_fragment),
                ],
            )
            effective_prompt = (
                state.context.public_instruction.encode("utf-8")
                + base_fragment.read_bytes()
                + fragment
            ).rstrip(b"\n")
            prompt_receipt = TypedAssignmentPromptInjectionReceiptV3(
                request_hash=state.request_hash,
                base_execution_prompt_receipt_hash=base_receipt.receipt_hash,
                runtime_class_hash=TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH,
                prepare_receipt_hash=prepare_receipt_hash,
                contract_hash=str(prepare_payload["contract_hash"]),
                evidence_set_hash=str(prepare_payload["evidence_set_hash"]),
                prompt_fragment_sha256=hashlib.sha256(fragment).hexdigest(),
                prompt_fragment_size=len(fragment),
                container_readback_sha256=hashlib.sha256(
                    readback.read_bytes()
                ).hexdigest(),
                run_template_before_hash=stable_hash(
                    {"run_template": run_before}
                ),
                run_template_after_hash=stable_hash(
                    {"run_template": run_after}
                ),
                effective_prompt_sha256=hashlib.sha256(
                    effective_prompt
                ).hexdigest(),
            )
            agent["run"] = run_after
            state.prompt_receipt = prompt_receipt
            runner._assumption_v2_typed_assignment_prompt_receipt = (
                prompt_receipt
            )
            self.event_sink.emit(
                Event(
                    event=(
                        "skilllearn_pre_agent_typed_assignment_v3_injected"
                    ),
                    stage="benchmark.skilllearn.typed_assignment_v3",
                    trace_id=state.request_hash[:20],
                    payload={
                        **prompt_receipt.safe_payload(),
                        "receipt_hash": prompt_receipt.receipt_hash,
                        "file_count": int(prepare_payload["file_count"]),
                    },
                )
            )
        finally:
            shutil.rmtree(receipt_root, ignore_errors=True)

    def _apply_and_reconcile_typed_assignment_v3(
        self,
        *,
        delegate: Any,
        container_name: str,
    ) -> None:
        state = getattr(self._typed_assignment_local, "state", None)
        if not isinstance(state, _TypedAssignmentRunStateV3):
            return
        if state.verifier_triggered:
            if state.runtime_evidence is None:
                raise SkillLearnAgentTerminalError(
                    "typed_assignment_reconciliation_invalid"
                )
            return
        state.verifier_triggered = True
        if (
            not isinstance(
                state.prompt_receipt,
                TypedAssignmentPromptInjectionReceiptV3,
            )
            or not isinstance(state.prepare_payload, Mapping)
        ):
            raise SkillLearnAgentTerminalError(
                "typed_assignment_prompt_delivery_missing"
            )
        receipt_root = Path(
            tempfile.mkdtemp(prefix="skilllearn_typed_assignment_apply_v3-")
        )
        try:
            try:
                fresh_tool_path = _fresh_runtime_tool_container_path()
                runtime_tool_sha256 = ""
                runtime_tool_readback_sha256 = ""
                try:
                    (
                        runtime_tool_sha256,
                        runtime_tool_readback_sha256,
                    ) = self._install_fresh_runtime_tool_v3(
                        delegate=delegate,
                        container_name=container_name,
                        container_path=fresh_tool_path,
                        host_root=receipt_root,
                    )
                    if (
                        runtime_tool_sha256
                        != state.prepare_payload.get("runtime_tool_sha256")
                    ):
                        raise TypedAssignmentIntegrationError(
                            "typed-assignment runtime changed after prepare"
                        )
                    self._docker_run_checked(
                        delegate,
                        [
                            "docker",
                            "exec",
                            container_name,
                            "python3",
                            fresh_tool_path,
                            "apply",
                            "--sidecar-dir",
                            TYPED_ASSIGNMENT_SIDECAR_CONTAINER_PATH,
                            "--expected-prepare-receipt-sha256",
                            state.prepare_receipt_file_sha256,
                        ],
                    )
                finally:
                    # Remove even a partially delivered fresh path before the
                    # verifier delegate is allowed to materialize /tests.
                    self._docker_run_checked(
                        delegate,
                        [
                            "docker",
                            "exec",
                            container_name,
                            "rm",
                            "-f",
                            "--",
                            fresh_tool_path,
                        ],
                    )
                payload = self._copy_container_json(
                    delegate=delegate,
                    container_name=container_name,
                    container_path=(
                        TYPED_ASSIGNMENT_RECONCILIATION_RECEIPT_CONTAINER_PATH
                    ),
                    host_root=receipt_root,
                    name="reconciliation-receipt.json",
                )
                file_count = int(state.prepare_payload.get("file_count") or 0)
                reconciliation_hash = (
                    _validate_reconciliation_receipt_v3(
                        payload,
                        prepare_payload=state.prepare_payload,
                        expected_prepare_receipt_file_sha256=(
                            state.prepare_receipt_file_sha256
                        ),
                        expected_runtime_tool_sha256=runtime_tool_sha256,
                    )
                )
                evidence = TypedAssignmentRuntimeEvidenceV3(
                    request_hash=state.request_hash,
                    runtime_class_hash=TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH,
                    prompt_receipt_hash=state.prompt_receipt.receipt_hash,
                    prepare_receipt_hash=state.prepare_receipt_hash,
                    reconciliation_receipt_hash=reconciliation_hash,
                    contract_hash=str(payload["contract_hash"]),
                    evidence_set_hash=str(payload["evidence_set_hash"]),
                    file_count=file_count,
                    prepare_receipt_body=dict(state.prepare_payload),
                    reconciliation_receipt_body=dict(payload),
                    post_agent_runtime_tool_sha256=runtime_tool_sha256,
                    post_agent_runtime_tool_readback_sha256=(
                        runtime_tool_readback_sha256
                    ),
                    post_agent_runtime_tool_container_path_hash=stable_hash(
                        {"container_path": fresh_tool_path}
                    ),
                )
                state.runtime_evidence = evidence
                with self._typed_assignment_evidence_lock:
                    self._typed_assignment_evidence.append(evidence)
                self.event_sink.emit(
                    Event(
                        event=(
                            "skilllearn_typed_assignment_v3_reconciled_"
                            "before_verifier"
                        ),
                        stage="benchmark.skilllearn.typed_assignment_v3",
                        trace_id=state.request_hash[:20],
                        payload={
                            **evidence.safe_payload(),
                            "evidence_hash": evidence.evidence_hash,
                        },
                    )
                )
            except SkillLearnAgentTerminalError:
                raise
            except Exception as exc:
                self.event_sink.emit(
                    Event(
                        event=(
                            "skilllearn_typed_assignment_v3_"
                            "reconciliation_blocked"
                        ),
                        stage="benchmark.skilllearn.typed_assignment_v3",
                        trace_id=state.request_hash[:20],
                        payload={
                            "request_hash": state.request_hash,
                            "runtime_class_hash": (
                                TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH
                            ),
                            "error_type": type(exc).__name__,
                            "verifier_materialized": False,
                            "raw_content_persisted": False,
                        },
                    )
                )
                raise SkillLearnAgentTerminalError(
                    "typed_assignment_reconciliation_invalid"
                ) from exc
        finally:
            shutil.rmtree(receipt_root, ignore_errors=True)

    @contextmanager
    def _verifier_isolation(
        self,
        runner: ModuleType,
        *,
        agent_runtime_volume: str | None = None,
        egress_policy: DockerEgressPolicy,
        offline_verifier_runtime: OfflineVerifierRuntime | None = None,
        trace_id: str = "skilllearn-typed-assignment-verifier-isolation",
    ) -> Iterator[None]:
        with super()._verifier_isolation(
            runner,
            agent_runtime_volume=agent_runtime_volume,
            egress_policy=egress_policy,
            offline_verifier_runtime=offline_verifier_runtime,
            trace_id=trace_id,
        ):
            base_proxy = runner.subprocess
            runner.subprocess = _TypedAssignmentVerifierProxyV3(
                base_proxy,
                backend=self,
            )
            try:
                yield
            finally:
                runner.subprocess = base_proxy

    def _run_serialized_evidence(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> ExecutionContractTrialEvidenceV2:
        try:
            evidence = super()._run_serialized_evidence(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
            state = getattr(self._typed_assignment_local, "state", None)
            expected_route = (
                request.variant is TrialVariant.POLICY_ON
                and self.execution_contract_bundle.has_item(request.item_id)
            )
            runtime_valid = (
                isinstance(state, _TypedAssignmentRunStateV3)
                and isinstance(
                    state.prompt_receipt,
                    TypedAssignmentPromptInjectionReceiptV3,
                )
                and isinstance(
                    state.runtime_evidence,
                    TypedAssignmentRuntimeEvidenceV3,
                )
            )
            if expected_route and evidence.observation.valid and not runtime_valid:
                observation = replace(
                    evidence.observation,
                    success=False,
                    score=0.0,
                    metrics={"evaluation_valid": 0.0},
                    error_type="typed_assignment_runtime_receipt_missing",
                )
                evidence = ExecutionContractTrialEvidenceV2(
                    observation=observation,
                    prompt_receipt=evidence.prompt_receipt,
                    execution_backend_instance_hash=(
                        self.execution_backend_instance_hash
                    ),
                    contract_route_expected=evidence.contract_route_expected,
                    prompt_receipt_valid=evidence.prompt_receipt_valid,
                )
                evidence.verify()
            return evidence
        finally:
            self._typed_assignment_local.state = None
