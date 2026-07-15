from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from assumption_agent.benchmarks import typed_assignment_integration_v3 as integration
from assumption_agent.models import stable_hash
from assumption_agent.typed_assignment_contract_v3 import PublicDestinationSpec


PUBLIC_INSTRUCTION = """
Organize the documents into LLM, trapped_ion_and_qc, black_hole, DNA, and
music_history. Each document belongs to one folder, so if a file does not fit
into any other 4 folders, it should fit into the last one.
"""
RECONCILIATION_RECEIPT_CONTAINER_PATH = (
    integration.TYPED_ASSIGNMENT_RECONCILIATION_RECEIPT_CONTAINER_PATH
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(body: dict[str, object]) -> dict[str, object]:
    return {**body, "receipt_hash": stable_hash(body)}


def _prepare_receipt(runtime_tool_sha256: str) -> dict[str, object]:
    return _receipt(
        {
            "runtime_policy": integration.TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
            "runtime_tool_sha256": runtime_tool_sha256,
            "contract_hash": "1" * 64,
            "evidence_set_hash": "2" * 64,
            "evidence_file_sha256": "3" * 64,
            "pre_manifest_hash": "4" * 64,
            "pre_manifest_file_sha256": "5" * 64,
            "plan_schema_file_sha256": "6" * 64,
            "prepare_state_file_sha256": "7" * 64,
            "public_instruction_sha256": hashlib.sha256(
                PUBLIC_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            "destination_set_hash": "8" * 64,
            "file_count": 1,
            "evidence_count": 1,
            "extraction_unavailable_count": 0,
            "container_evidence_profile_persisted": True,
            "raw_public_instruction_in_receipt": False,
            "raw_content_evidence_in_receipt": False,
            "source_filenames_in_receipt": False,
            "host_safe_receipt": True,
        }
    )


def _reconciliation_receipt(
    runtime_tool_sha256: str,
    *,
    prepare_receipt_file_sha256: str,
) -> dict[str, object]:
    return _receipt(
        {
            "runtime_policy": integration.TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
            "runtime_tool_sha256": runtime_tool_sha256,
            "mode": "apply_and_reconcile",
            "contract_hash": "1" * 64,
            "evidence_set_hash": "2" * 64,
            "prepare_receipt_file_sha256": prepare_receipt_file_sha256,
            "plan_file_sha256": "9" * 64,
            "normalized_plan_hash": "a" * 64,
            "assignment_count": 1,
            "positive_evidence_assignment_count": 1,
            "public_default_assignment_count": 0,
            "reopened_file_count": 1,
            "source_empty": True,
            "destination_set_hash": "8" * 64,
            "destination_distribution_hash": "b" * 64,
            "final_task_manifest_hash": "c" * 64,
            "all_destination_content_hashes_match": True,
            "transactional_apply": True,
            "rollback_required": False,
            "raw_public_instruction_in_receipt": False,
            "raw_content_evidence_in_receipt": False,
            "source_filenames_in_receipt": False,
            "host_safe_receipt": True,
        }
    )


def _prompt_receipt(
    prepare_receipt_hash: str = "f" * 64,
) -> integration.TypedAssignmentPromptInjectionReceiptV3:
    return integration.TypedAssignmentPromptInjectionReceiptV3(
        request_hash="d" * 64,
        base_execution_prompt_receipt_hash="e" * 64,
        runtime_class_hash=integration.TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH,
        prepare_receipt_hash=prepare_receipt_hash,
        contract_hash="1" * 64,
        evidence_set_hash="2" * 64,
        prompt_fragment_sha256="3" * 64,
        prompt_fragment_size=100,
        container_readback_sha256="3" * 64,
        run_template_before_hash="4" * 64,
        run_template_after_hash="5" * 64,
        effective_prompt_sha256="6" * 64,
    )


def test_runtime_class_hash_binds_all_three_implementation_files() -> None:
    rows = sorted(
        (
            {"component": path.name, "sha256": _sha256_file(path)}
            for path in integration._runtime_component_paths()
        ),
        key=lambda row: row["component"],
    )
    expected = stable_hash(
        {
            "runtime_version": integration.TYPED_ASSIGNMENT_RUNTIME_VERSION,
            "components": rows,
        }
    )

    assert integration.typed_assignment_runtime_class_hash() == expected
    assert integration.TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH == expected


def test_prompt_fragment_delivers_plan_only_closed_contract() -> None:
    spec = PublicDestinationSpec.from_public_instruction(PUBLIC_INSTRUCTION)
    prepare = {
        "contract_hash": "a" * 64,
        "evidence_set_hash": "b" * 64,
    }

    fragment = integration._prompt_fragment(
        request_hash="c" * 64,
        runtime_class_hash="d" * 64,
        prepare_payload=prepare,
        destination_spec=spec,
    )
    text = fragment.decode("utf-8")

    assert integration.TYPED_ASSIGNMENT_EVIDENCE_CONTAINER_PATH in text
    assert integration.TYPED_ASSIGNMENT_PLAN_CONTAINER_PATH in text
    assert '"public_default_destination":"music_history"' in text
    assert "Do not create destination folders" in text
    assert "harness, not the agent" in text
    assert "/root/papers" not in text


def test_safe_runtime_receipt_digest_rejects_drift() -> None:
    body = {
        "runtime_policy": "test",
        "contract_hash": "a" * 64,
        "host_safe_receipt": True,
    }
    receipt = {**body, "receipt_hash": stable_hash(body)}

    assert integration._receipt_digest(receipt) == stable_hash(body)

    drifted = {**receipt, "host_safe_receipt": False}
    try:
        integration._receipt_digest(drifted)
    except integration.TypedAssignmentIntegrationError:
        pass
    else:  # pragma: no cover - explicit fail keeps the exception contract clear.
        raise AssertionError("drifted receipt was accepted")

    with pytest.raises(integration.TypedAssignmentIntegrationError):
        integration._receipt_digest(body)


def test_typed_proxy_reconciles_before_forwarding_verifier() -> None:
    order: list[str] = []

    class Backend:
        def _apply_and_reconcile_typed_assignment_v3(
            self, *, delegate, container_name: str
        ) -> None:
            assert container_name == "container-1"
            assert delegate is base
            order.append("reconcile")

    class Delegate:
        def run(self, command, *_args, **_kwargs):
            order.append("verifier")
            return command

    base = Delegate()
    proxy = integration._TypedAssignmentVerifierProxyV3(
        base,
        backend=Backend(),
    )

    returned = proxy.run(
        ["docker", "exec", "container-1", "bash", "/tests/test.sh"]
    )

    assert order == ["reconcile", "verifier"]
    assert returned[-1] == "/tests/test.sh"


def test_prompt_receipt_safe_payload_contains_no_raw_evidence() -> None:
    receipt = integration.TypedAssignmentPromptInjectionReceiptV3(
        request_hash="1" * 64,
        base_execution_prompt_receipt_hash="2" * 64,
        runtime_class_hash="3" * 64,
        prepare_receipt_hash="4" * 64,
        contract_hash="5" * 64,
        evidence_set_hash="6" * 64,
        prompt_fragment_sha256="7" * 64,
        prompt_fragment_size=100,
        container_readback_sha256="7" * 64,
        run_template_before_hash="8" * 64,
        run_template_after_hash="9" * 64,
        effective_prompt_sha256="a" * 64,
    )

    serialized = json.dumps(receipt.safe_payload(), sort_keys=True)
    assert "raw_document_content_persisted_host_side" in serialized
    assert "/root/papers" not in serialized
    assert receipt.receipt_hash == stable_hash(receipt.safe_payload())


def test_post_agent_fresh_tool_ignores_tampered_known_prepare_path() -> None:
    runtime_source = integration._runtime_tool_source()
    runtime_bytes = runtime_source.read_bytes()
    runtime_sha256 = hashlib.sha256(runtime_bytes).hexdigest()
    prepare_receipt_file_sha256 = "f" * 64
    prepare = _prepare_receipt(runtime_sha256)
    reconciliation = _reconciliation_receipt(
        runtime_sha256,
        prepare_receipt_file_sha256=prepare_receipt_file_sha256,
    )

    class EventSink:
        def __init__(self) -> None:
            self.events: list[object] = []

        def emit(self, event: object) -> None:
            self.events.append(event)

    class Delegate:
        def __init__(self) -> None:
            self.files = {
                integration.TYPED_ASSIGNMENT_TOOL_CONTAINER_PATH: (
                    b"agent-authored replacement"
                )
            }
            self.executed: list[tuple[str, bytes]] = []

        def run(self, command, *_args, **_kwargs):
            command = list(command)
            if command[:2] == ["docker", "cp"]:
                source, destination = command[2], command[3]
                if source.startswith("container-1:"):
                    container_path = source.split(":", 1)[1]
                    Path(destination).write_bytes(self.files[container_path])
                else:
                    container_path = destination.split(":", 1)[1]
                    self.files[container_path] = Path(source).read_bytes()
            elif command[:3] == ["docker", "exec", "container-1"]:
                operation = command[3]
                if operation == "test":
                    return SimpleNamespace(
                        returncode=int(command[-1] in self.files)
                    )
                if operation == "rm":
                    self.files.pop(command[-1], None)
                elif operation == "python3":
                    runtime_path = command[4]
                    runtime_payload = self.files[runtime_path]
                    self.executed.append((runtime_path, runtime_payload))
                    self.files[RECONCILIATION_RECEIPT_CONTAINER_PATH] = (
                        json.dumps(reconciliation, sort_keys=True) + "\n"
                    ).encode("utf-8")
            return SimpleNamespace(returncode=0)

    backend = object.__new__(
        integration.TypedAssignmentExecutionContractSubprocessBackendV3
    )
    backend._typed_assignment_local = threading.local()
    backend._typed_assignment_evidence_lock = threading.Lock()
    backend._typed_assignment_evidence = []
    backend.event_sink = EventSink()
    state = integration._TypedAssignmentRunStateV3(
        request_hash="d" * 64,
        context=None,  # type: ignore[arg-type]
        prepare_payload=prepare,
        prepare_receipt_hash=str(prepare["receipt_hash"]),
        prepare_receipt_file_sha256=prepare_receipt_file_sha256,
        prompt_receipt=_prompt_receipt(str(prepare["receipt_hash"])),
    )
    backend._typed_assignment_local.state = state
    delegate = Delegate()

    backend._apply_and_reconcile_typed_assignment_v3(
        delegate=delegate,
        container_name="container-1",
    )

    assert len(delegate.executed) == 1
    executed_path, executed_bytes = delegate.executed[0]
    assert executed_path != integration.TYPED_ASSIGNMENT_TOOL_CONTAINER_PATH
    assert executed_path.startswith(
        integration.TYPED_ASSIGNMENT_FRESH_TOOL_CONTAINER_PREFIX
    )
    assert executed_bytes == runtime_bytes
    assert delegate.files[integration.TYPED_ASSIGNMENT_TOOL_CONTAINER_PATH] == (
        b"agent-authored replacement"
    )
    assert executed_path not in delegate.files
    assert len(backend.typed_assignment_evidence) == 1
    safe = backend.typed_assignment_evidence[0].safe_payload()
    assert safe["prepare_receipt_body"] == prepare
    assert safe["reconciliation_receipt_body"] == reconciliation
    assert safe["post_agent_runtime_delivery"][
        "container_readback_sha256"
    ] == runtime_sha256
    assert safe["reconciliation_completed_before_verifier_invocation"] is True
    assert safe["verifier_invoked_at_receipt_time"] is False
    assert safe["verifier_materialized_at_receipt_time"] is False
    assert "verifier_materialized_after_reconciliation" not in safe


def test_fresh_tool_readback_must_match_exact_host_bytes(tmp_path: Path) -> None:
    class Delegate:
        def __init__(self) -> None:
            self.files: dict[str, bytes] = {}

        def run(self, command, *_args, **_kwargs):
            command = list(command)
            if command[:2] == ["docker", "cp"]:
                source, destination = command[2], command[3]
                if source.startswith("container-1:"):
                    container_path = source.split(":", 1)[1]
                    Path(destination).write_bytes(
                        self.files[container_path] + b"tampered"
                    )
                else:
                    self.files[destination.split(":", 1)[1]] = Path(
                        source
                    ).read_bytes()
            return SimpleNamespace(returncode=0)

    backend = object.__new__(
        integration.TypedAssignmentExecutionContractSubprocessBackendV3
    )
    with pytest.raises(integration.TypedAssignmentIntegrationError):
        backend._install_fresh_runtime_tool_v3(
            delegate=Delegate(),
            container_name="container-1",
            container_path=integration._fresh_runtime_tool_container_path(),
            host_root=tmp_path,
        )


def test_reconciliation_receipt_binds_committed_runtime_tool_hash() -> None:
    runtime_sha256 = _sha256_file(integration._runtime_tool_source())
    prepare = _prepare_receipt(runtime_sha256)
    prepare_file_sha256 = "f" * 64
    drifted = _reconciliation_receipt(
        "0" * 64,
        prepare_receipt_file_sha256=prepare_file_sha256,
    )

    with pytest.raises(integration.TypedAssignmentIntegrationError):
        integration._validate_reconciliation_receipt_v3(
            drifted,
            prepare_payload=prepare,
            expected_prepare_receipt_file_sha256=prepare_file_sha256,
            expected_runtime_tool_sha256=runtime_sha256,
        )
