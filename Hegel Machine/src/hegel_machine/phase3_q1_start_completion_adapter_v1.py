"""Read-only adapter from persisted Q1 actor evidence to a completion receipt.

The adapter owns no controller state and performs no publication.  Successor
controllers provide their frozen JSON parser, successor-bundle validator, and
the matching supervisor helper module so the persisted v3-domain roots can be
replayed before the v1 completion receipt is built.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import re
import stat
from typing import Any, Final, NoReturn

from . import phase3_q1_start_completion_wire_v1 as completion


_TAGGED: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ACTOR_ORDER: Final = ("python", "rust", "host")

StrictJsonBytes = Callable[[bytes, str], Mapping[str, object]]
SuccessorBundleValidator = Callable[[Mapping[str, object]], object]


class CompletionAdapterError(ValueError):
    """A classified failure while replaying persisted completion evidence."""

    def __init__(
        self,
        failure_code: str,
        detail: str,
        *,
        pending_phase: str | None = None,
    ) -> None:
        super().__init__(f"{failure_code}: {detail}")
        self.failure_code = failure_code
        self.detail = detail
        self.pending_phase = pending_phase


def _fail(
    failure_code: str,
    detail: str,
    *,
    pending_phase: str | None = None,
) -> NoReturn:
    raise CompletionAdapterError(
        failure_code,
        detail,
        pending_phase=pending_phase,
    )


def build_completion_artifact_v1(
    supervisor_outcome: Mapping[str, object],
    capsule: Mapping[str, object],
    run_spec: Mapping[str, object],
    *,
    supervisor_module: Any,
    strict_json_bytes: StrictJsonBytes,
    validate_successor_bundle: SuccessorBundleValidator,
) -> bytes:
    """Replay persisted actor evidence and return canonical receipt bytes.

    All filesystem access is read-only.  The supplied supervisor module is used
    only for its frozen root-domain helpers and constants.
    """

    def root(value: object, label: str) -> bytes:
        if type(value) is not str or _TAGGED.fullmatch(value) is None:
            _fail("Q1_START_COMPLETION_ADAPTER_INVALID", f"{label} root differs")
        return bytes.fromhex(value.removeprefix("sha256:"))

    def files(root_path: Path) -> tuple[Any, ...]:
        try:
            rows = []
            for path in root_path.rglob("*"):
                info = path.lstat()
                if path.is_symlink() or not (
                    stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
                ):
                    raise OSError(f"unsupported output entry {path}")
                if stat.S_ISREG(info.st_mode):
                    rows.append(
                        completion.Q1FileEvidenceV1(
                            path.relative_to(root_path).as_posix(), path.read_bytes()
                        )
                    )
        except OSError as error:
            _fail("Q1_START_COMPLETION_OUTPUT_UNAVAILABLE", str(error))
        return tuple(sorted(rows, key=lambda row: row.relative_path))

    gate16 = supervisor_outcome.get("gate16_candidate")
    gate18 = supervisor_outcome.get("gate18_runtime_receipt_inputs")
    if type(gate16) is not dict or type(gate18) is not dict:
        _fail("Q1_START_COMPLETION_ADAPTER_INVALID", "Gate 16/18 inputs differ")
    signature_preimages = gate16.get("signature_state_preimages")
    if type(signature_preimages) is not list or len(signature_preimages) != 2:
        _fail("Q1_START_COMPLETION_ADAPTER_INVALID", "signature preimages differ")
    signatures = tuple(
        completion.Q1SignatureCompletionRowV1(
            row["input_signature_id"],
            root(row["fixed_point_record_root"], "fixed point"),
            root(row["signature_manifest_root"], "signature manifest"),
            root(row["saturation_state_root"], "saturation state"),
            completion.signature_guard_ceiling_rows_v1(row["input_signature_id"]),
        )
        for row in signature_preimages
        if type(row) is dict
    )
    if len(signatures) != 2:
        _fail("Q1_START_COMPLETION_ADAPTER_INVALID", "signature rows differ")
    run_genesis = root(run_spec["run_genesis_root"], "run genesis")
    prestart_receipt = root(
        run_spec["prestart_admission_receipt_root"], "prestart receipt"
    )
    host_result = supervisor_outcome.get("host_result")
    if type(host_result) is not dict:
        _fail("Q1_START_COMPLETION_ADAPTER_INVALID", "host result differs")
    closure_bundle = root(host_result.get("closure_bundle_root"), "closure bundle")
    binding_kwargs = {
        "expected_capsule_digest": run_spec["start_capsule_digest"],
        "expected_source_commit": run_spec["source_commit"],
        "expected_run_genesis_root": run_genesis,
        "expected_prestart_admission_receipt_root": prestart_receipt,
        "expected_closure_bundle_root": closure_bundle,
        "expected_signature_rows": signatures,
    }
    stage_rows_value = gate18.get("stage_rows")
    if type(stage_rows_value) is not list or len(stage_rows_value) != 3:
        _fail("Q1_START_COMPLETION_ADAPTER_INVALID", "Gate 18 stage rows differ")
    stage_rows = {
        row.get("stage"): row for row in stage_rows_value if type(row) is dict
    }
    if tuple(
        row.get("stage") for row in stage_rows_value if type(row) is dict
    ) != _ACTOR_ORDER:
        _fail("Q1_START_COMPLETION_ADAPTER_INVALID", "Gate 18 actor order differs")

    output_files: dict[str, tuple[Any, ...]] = {}
    isolation_adapters: dict[str, Any] = {}
    for stage in _ACTOR_ORDER:
        stage_row = stage_rows[stage]
        observed_files = files(Path(run_spec[f"{stage}_output"]))
        output_files[stage] = observed_files
        source_rows = tuple(
            completion.Q1SourceManifestEvidenceRowV1(
                row["relative_path"],
                row["byte_count"],
                root(row["raw_sha256"], f"{stage} source"),
            )
            for row in run_spec["actor_source_manifests"][stage]
        )
        runtime_evidence = stage_row.get("runtime_evidence")
        if type(runtime_evidence) is not dict or type(
            runtime_evidence.get("mounts")
        ) is not list:
            _fail(
                "Q1_START_COMPLETION_ADAPTER_INVALID",
                f"{stage} runtime evidence differs",
            )
        mount_rows = tuple(
            completion.Q1MountManifestEvidenceRowV1(
                row["destination"], row["source"], row["read_write"]
            )
            for row in runtime_evidence["mounts"]
            if type(row) is dict
        )
        adapter = completion.Q1IsolationEvidenceAdapterV1(
            actor_id=stage.encode("ascii"),
            actor_implementation_root=root(
                run_spec["actor_implementation_roots"][stage],
                f"{stage} implementation",
            ),
            source_manifest_rows=source_rows,
            mount_manifest_rows=mount_rows,
            output_files=observed_files,
            scratch_path=run_spec[f"{stage}_scratch"],
        )
        source_replay = stage_row.get("source_manifest_replay")
        supervisor_source_root = supervisor_module._object_root(
            supervisor_module._SOURCE_CLOSURE_DOMAIN,
            run_spec["actor_source_manifests"][stage],
        )
        supervisor_mount_root = supervisor_module._object_root(
            supervisor_module._MOUNT_REGISTRY_DOMAIN,
            runtime_evidence["mounts"],
        )
        supervisor_output_root = supervisor_module._output_tree_closure_root(
            Path(run_spec[f"{stage}_output"])
        )
        supervisor_scratch_root = supervisor_module._object_root(
            supervisor_module._SCRATCH_ABSENCE_DOMAIN,
            {"path": run_spec[f"{stage}_scratch"], "absent": True},
        )
        if (
            type(source_replay) is not dict
            or source_replay.get("source_closure_root") != supervisor_source_root
            or stage_row.get("mount_registry_root") != supervisor_mount_root
            or stage_row.get("output_tree_closure_root") != supervisor_output_root
            or stage_row.get("scratch_absence_root") != supervisor_scratch_root
        ):
            _fail(
                "Q1_START_COMPLETION_ADAPTER_INVALID",
                f"{stage} persisted closure replay differs",
            )
        isolation_adapters[stage] = adapter

    formal_paths = gate16.get("formal_relative_paths")
    if type(formal_paths) is not list or not formal_paths:
        _fail("Q1_START_COMPLETION_ADAPTER_INVALID", "formal path registry differs")
    python_by_path = {row.relative_path: row for row in output_files["python"]}
    try:
        formal_files = tuple(python_by_path[path] for path in formal_paths)
    except KeyError as error:
        _fail(
            "Q1_START_COMPLETION_ADAPTER_INVALID",
            f"formal file absent: {error}",
        )
    supervisor_formal_file_root = supervisor_module._file_closure_root(
        Path(run_spec["python_output"]),
        formal_paths,
        supervisor_module._FORMAL_FILE_CLOSURE_DOMAIN,
    )
    if gate16.get("formal_file_closure_root") != supervisor_formal_file_root:
        _fail("Q1_START_COMPLETION_ADAPTER_INVALID", "formal file closure differs")
    formal_file_root = completion.formal_output_file_closure_root_v1(formal_files)

    def runtime_values(stage: str) -> tuple[int, int]:
        runtime = stage_rows[stage]["runtime_evidence"]
        started = runtime.get("measurement_interval_start_monotonic_ns")
        returned = runtime.get("actor_return_monotonic_ns")
        exited = runtime.get("container_exit_monotonic_ns")
        wall_ns = runtime.get("measurement_wall_time_ns")
        peak = runtime.get("memory_peak_bytes")
        if (
            type(started) is not int
            or type(returned) is not int
            or type(exited) is not int
            or type(wall_ns) is not int
            or not started <= returned <= exited
            or wall_ns != exited - started
            or type(peak) is not int
            or peak < 0
        ):
            _fail(
                "Q1_START_COMPLETION_ADAPTER_INVALID",
                f"{stage} runtime counters differ",
            )
        return peak, wall_ns

    endpoints: dict[str, Any] = {}
    for stage in ("python", "rust"):
        by_path = {row.relative_path: row for row in output_files[stage]}
        try:
            result_payload = by_path["endpoint-result.json"].payload
        except KeyError as error:
            _fail("Q1_START_COMPLETION_OUTPUT_UNAVAILABLE", f"{stage}: {error}")
        try:
            result_value = strict_json_bytes(
                result_payload, f"{stage} endpoint result"
            )
        except Exception as error:
            _fail("Q1_START_COMPLETION_ADAPTER_INVALID", str(error))
        memory_peak, wall_ns = runtime_values(stage)
        endpoints[stage] = completion.Q1EndpointCompletionEvidenceV1(
            actor_id=stage.encode("ascii"),
            actor_implementation_root=root(
                run_spec["actor_implementation_roots"][stage],
                f"{stage} implementation",
            ),
            run_genesis_root=run_genesis,
            prestart_admission_receipt_root=prestart_receipt,
            endpoint_result_root=completion.endpoint_result_root_v1(
                stage.encode("ascii"), result_payload, **binding_kwargs
            ),
            closure_bundle_root=closure_bundle,
            signature_rows=signatures,
            actual_output_bytes=sum(len(row.payload) for row in output_files[stage]),
            actual_peak_scratch_bytes=result_value[
                "actual_formal_peak_scratch_bytes"
            ],
            actual_peak_rss_bytes=memory_peak,
            actual_wall_time_ns=wall_ns,
        )

    host_files = {row.relative_path: row for row in output_files["host"]}
    try:
        host_payload = host_files["host-result.json"].payload
    except KeyError as error:
        _fail("Q1_START_COMPLETION_OUTPUT_UNAVAILABLE", f"host: {error}")
    host_memory_peak, host_wall_ns = runtime_values("host")
    host_runtime = stage_rows["host"]["runtime_evidence"]
    if (
        host_runtime.get("actor_terminal_scratch_payload_bytes") != 0
        or host_runtime.get("supervisor_control_peak_payload_bytes") != 0
        or "--scratch-dir" in host_runtime.get("command", [])
    ):
        _fail("Q1_START_COMPLETION_ADAPTER_INVALID", "host scratch evidence differs")
    agreement = completion.Q1DualReplayAgreementV1(
        run_genesis_root=run_genesis,
        prestart_admission_receipt_root=prestart_receipt,
        python_endpoint_evidence_root=endpoints["python"].evidence_root,
        rust_endpoint_evidence_root=endpoints["rust"].evidence_root,
        formal_file_closure_root=formal_file_root,
        signature_rows=signatures,
        closure_bundle_root=closure_bundle,
        host_actor_implementation_root=root(
            run_spec["actor_implementation_roots"]["host"],
            "host implementation",
        ),
        host_replay_result_root=completion.host_replay_result_root_v1(
            host_payload, **binding_kwargs
        ),
        actual_host_output_bytes=sum(
            len(row.payload) for row in output_files["host"]
        ),
        actual_host_peak_scratch_bytes=0,
        actual_host_peak_rss_bytes=host_memory_peak,
        actual_host_wall_time_ns=host_wall_ns,
    )
    actor_rows = tuple(
        isolation_adapters[stage].isolation_row() for stage in _ACTOR_ORDER
    )
    try:
        mount_manifest_root = completion.runtime_mount_registry_root_v1(actor_rows)
        input_absence_root = completion.formal_input_key_registry_absence_root_v1()
    except AttributeError as error:
        _fail(
            "COMPLETION_AGGREGATE_ROOT_ADAPTER_UNAVAILABLE",
            str(error),
            pending_phase="LOCAL_DEPENDENCY",
        )
    # The successor validator proves the supervisor-domain empty-input
    # manifest, while the completion wire independently proves the v1-domain
    # empty formal-input-key registry.  Their digests intentionally differ by
    # domain and are not compared byte-for-byte.
    validate_successor_bundle(capsule["successor_prestart_bundle"])
    ledger = completion.Q1TargetBlindAccessLedgerV1(
        run_genesis_root=run_genesis,
        prestart_admission_receipt_root=prestart_receipt,
        dual_replay_agreement_root=agreement.agreement_root,
        mount_manifest_root=mount_manifest_root,
        input_key_registry_absence_root=input_absence_root,
        actor_rows=actor_rows,
    )
    subjects = (
        (
            run_genesis,
            prestart_receipt,
            (
                endpoints["python"].canonical_object(),
                endpoints["rust"].canonical_object(),
            ),
        ),
        (
            endpoints["python"].evidence_root,
            endpoints["rust"].evidence_root,
            formal_file_root,
            tuple(row.canonical_object() for row in signatures),
            closure_bundle,
            True,
            True,
        ),
        (agreement.canonical_object(), agreement.agreement_root),
        (ledger.canonical_object(), ledger.ledger_root),
    )
    receipt = completion.Q1CompletionReceiptV1(
        completion.Q1CompletionCoreV1(
            run_genesis_root=run_genesis,
            prestart_admission_receipt_root=prestart_receipt,
            gate_subject_rows_15_through_18=subjects,
            output_slots_1_through_7=completion.output_slots_1_through_7_v1(
                agreement, ledger
            ),
        )
    )
    completion.decode_completion_receipt_v1(receipt.canonical_bytes)
    return receipt.canonical_bytes
