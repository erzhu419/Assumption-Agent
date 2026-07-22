from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading

import numpy as np
import pytest

from assumption_agent.benchmarks import tatqa_p19_acquisition_v1 as acquisition
from assumption_agent.benchmarks import tatqa_p19_formal_adapters_v1 as adapters
from assumption_agent.benchmarks import tatqa_p19_formal_controller_v1 as controller
from replication_runtime.tatqa_p19_v1 import hipporag_contract
from replication_runtime.tatqa_p19_v1 import typed_plan_contract


@pytest.fixture
def local_tmp() -> Path:
    # DrvFS does not preserve the 0600 custody contract used by the formal
    # acquisition.  The production boundary is intentionally strict, so these
    # tests use the native Linux filesystem just like the remote formal root.
    root = Path(tempfile.mkdtemp(prefix="tatqa-p18-adapter-", dir="/tmp"))
    try:
        yield root
    finally:
        for directory, _children, _files in os.walk(root):
            Path(directory).chmod(0o700)
        shutil.rmtree(root)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _write(path: Path, value: object) -> tuple[str, int]:
    raw = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _self_hashed(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: acquisition.stable_hash(body)}


def _systemd_unit_closure(
    *, unit_name_sha256: str, control_group_sha256: str, scope: str
) -> dict[str, object]:
    return {
        "active_state": "inactive",
        "control_group_process_count": 0,
        "control_group_sha256": control_group_sha256,
        "control_group_thread_count": 0,
        "load_state": "not-found",
        "main_pid": 0,
        "schema": adapters.SYSTEMD_UNIT_CLOSURE_SCHEMA,
        "sub_state": "dead",
        "systemctl_reset_failed_returncode": 0,
        "systemctl_reset_failed_stderr_sha256": hashlib.sha256(
            f"{scope}-reset-stderr".encode()
        ).hexdigest(),
        "systemctl_reset_failed_stdout_sha256": hashlib.sha256(
            f"{scope}-reset-stdout".encode()
        ).hexdigest(),
        "systemctl_show_returncode": 0,
        "systemctl_show_stderr_sha256": hashlib.sha256(
            f"{scope}-closed-show-stderr".encode()
        ).hexdigest(),
        "systemctl_show_stdout_sha256": hashlib.sha256(
            f"{scope}-closed-show-stdout".encode()
        ).hexdigest(),
        "unit_name_sha256": unit_name_sha256,
    }


def _systemd_start_policy(
    *,
    unit_name_sha256: str,
    control_group_sha256: str,
    worker_pid: int,
    scope: str,
) -> dict[str, object]:
    return {
        "active_state": "active",
        "control_group_sha256": control_group_sha256,
        "kill_mode": "control-group",
        "load_state": "loaded",
        "main_pid": worker_pid,
        "schema": adapters.SYSTEMD_START_POLICY_SCHEMA,
        "sub_state": "running",
        "systemctl_show_returncode": 0,
        "systemctl_show_stderr_sha256": hashlib.sha256(
            f"{scope}-start-show-stderr".encode()
        ).hexdigest(),
        "systemctl_show_stdout_sha256": hashlib.sha256(
            f"{scope}-start-show-stdout".encode()
        ).hexdigest(),
        "tasks_max": 3,
        "unit_name_sha256": unit_name_sha256,
    }


def _runtime_receipts(root: Path) -> adapters.RuntimeReceiptPaths:
    fingerprint_body = {
        "api_or_online_evaluator_calls": 0,
        "controller_or_worker_label_reads": 0,
        "controller_or_worker_source_reads": 0,
        "external_network_calls": 0,
        "filesystem_isolation": adapters.SYSTEMD_FILESYSTEM_ISOLATION,
        "formal_source_opened": False,
        "hippo_concurrency_cap": 8,
        "maximum_cpu_threads_per_hippo_process": 2,
        "retry_replay_resample_provider_switch": 0,
        "schema": adapters.RUNTIME_FINGERPRINT_SCHEMA,
        "source_identifiers_answers_families_mappings_or_labels_present": False,
        "status": "verified_before_formal_source_open",
        "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
    }
    fingerprint = _self_hashed(
        fingerprint_body, adapters.RUNTIME_FINGERPRINT_SELF_FIELD
    )
    fingerprint_path = root / "runtime.fingerprint.json"
    _write(fingerprint_path, fingerprint)
    canary_body = {
        "api_or_online_evaluator_calls": 0,
        "external_network_calls": 0,
        "filesystem_isolation": adapters.SYSTEMD_FILESYSTEM_ISOLATION,
        "formal_source_opened": False,
        "hippo_worker_receipt_sha256": "3" * 64,
        "minilm_worker_receipt_sha256": "2" * 64,
        "production_note": "public synthetic fixture only",
        "public_synthetic_distinct_rankings": True,
        "public_synthetic_p0_behavior_sha256": "a" * 64,
        "public_synthetic_p1_behavior_sha256": "b" * 64,
        "qualified": True,
        "retry_replay_resample_provider_switch": 0,
        "runtime_fingerprint_self_sha256": fingerprint[
            adapters.RUNTIME_FINGERPRINT_SELF_FIELD
        ],
        "schema": adapters.PRODUCTION_CANARY_SCHEMA,
        "source_identifiers_answers_families_mappings_or_labels_present": False,
        "status": "qualified_before_formal_source_open",
        "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
        "typed_plan_worker_receipt_sha256": "1" * 64,
    }
    canary = _self_hashed(canary_body, adapters.PRODUCTION_CANARY_SELF_FIELD)
    canary_path = root / "production.canary.json"
    _write(canary_path, canary)
    return adapters.RuntimeReceiptPaths(
        runtime_fingerprint=fingerprint_path,
        production_canary=canary_path,
    )


def _custody_payloads(project: Path) -> None:
    private_root = project / acquisition.ACQUISITION_ROOT_RELATIVE
    private_root.mkdir(parents=True)
    bindings: dict[str, dict[str, object]] = {}
    view_hashes: dict[str, str] = {}
    for block in acquisition.BLOCK_ORDER:
        items: list[dict[str, object]] = []
        for ordinal in range(acquisition.BLOCK_COUNTS[block]):
            question = f"Compare Acme revenue {block} item {ordinal}."
            units = (
                acquisition.CanonicalUnit("T:0", "company | revenue | year"),
                acquisition.CanonicalUnit("T:1", f"Acme | {ordinal + 10} | 2023"),
                acquisition.CanonicalUnit("T:2", f"Acme | {ordinal + 20} | 2024"),
                acquisition.CanonicalUnit("P:1", "Acme described annual revenue."),
                acquisition.CanonicalUnit("P:2", "The report discussed costs."),
            )
            commitment = acquisition.item_commitment(
                block=block, ordinal=ordinal, question=question, units=units
            )
            items.append(
                {
                    "canonical_units": [row.payload() for row in units],
                    "item_commitment_sha256": commitment,
                    "ordinal": ordinal,
                    "question": question,
                }
            )
        view_body = {
            "access_state": (
                "presealed_until_valid_A_hold_promotion"
                if block == "M_search"
                else "available_only_at_frozen_lifecycle_stage"
            ),
            "block": block,
            "item_count": len(items),
            "items": items,
            "late_fields_included": False,
            "schema": f"{acquisition.VERSION}_block_view",
            "version": acquisition.VERSION,
        }
        view = _self_hashed(view_body, "block_view_sha256")
        view_hashes[block] = view["block_view_sha256"]  # type: ignore[assignment]
        filename = acquisition.VIEW_FILENAMES[block]
        file_sha, size = _write(private_root / filename, view)
        bindings[filename] = {
            "file_sha256": file_sha,
            "filename": filename,
            "mode": "0600",
            "semantic_sha256": view["block_view_sha256"],
            "size_bytes": size,
        }

        if block != "F_search":
            per_family = acquisition.PER_FAMILY_QUOTA[block]
            label_items = [
                {
                    "family": acquisition.FAMILY_ORDER[ordinal // per_family],
                    "gold_unit_ids": ["T:1"],
                    "item_commitment_sha256": row["item_commitment_sha256"],
                    "ordinal": ordinal,
                }
                for ordinal, row in enumerate(items)
            ]
            label_body = {
                "access_state": (
                    "presealed_until_valid_A_hold_promotion"
                    if block == "M_search"
                    else "sealed_until_corresponding_actions_and_postflight"
                ),
                "block": block,
                "block_view_sha256": view["block_view_sha256"],
                "item_count": len(label_items),
                "items": label_items,
                "schema": f"{acquisition.VERSION}_sealed_labels",
                "version": acquisition.VERSION,
            }
            labels = _self_hashed(label_body, "label_pack_sha256")
            filename = acquisition.LABEL_FILENAMES[block]
            file_sha, size = _write(private_root / filename, labels)
            bindings[filename] = {
                "file_sha256": file_sha,
                "filename": filename,
                "mode": "0600",
                "semantic_sha256": labels["label_pack_sha256"],
                "size_bytes": size,
            }

    ledger = _self_hashed(
        {
            "item_count": acquisition.TOTAL_SELECTED_ITEMS,
            "schema": f"{acquisition.VERSION}_private_ledger",
            "version": acquisition.VERSION,
        },
        "ledger_sha256",
    )
    file_sha, size = _write(private_root / acquisition.LEDGER_FILENAME, ledger)
    bindings[acquisition.LEDGER_FILENAME] = {
        "file_sha256": file_sha,
        "filename": acquisition.LEDGER_FILENAME,
        "mode": "0600",
        "semantic_sha256": ledger["ledger_sha256"],
        "size_bytes": size,
    }

    receipt_body = {
        "F_search_label_pack_created": False,
        "M_search_view_and_labels_presealed": True,
        "aggregate_qualification": {},
        "fixed_block_counts": acquisition.BLOCK_COUNTS,
        "fixed_per_family_quota": acquisition.PER_FAMILY_QUOTA,
        "implementation_freeze_self_sha256": "4" * 64,
        "label_file_count": 3,
        "ledger_file_count": 1,
        "network_download_online_evaluator_or_model_calls": 0,
        "private_file_bindings": bindings,
        "retry_replay_resample_or_smaller_blocks": 0,
        "schema": f"{acquisition.VERSION}_public_receipt",
        "selected_context_count": acquisition.TOTAL_SELECTED_ITEMS,
        "selected_question_count": acquisition.TOTAL_SELECTED_ITEMS,
        "selection_secret_commitment_sha256": "5" * 64,
        "selection_secret_persisted_publicly": False,
        "selection_secret_size_bytes": 32,
        "source_custody_self_sha256": acquisition.CUSTODY_SELF_SHA256,
        "source_download_receipt_self_sha256": "6" * 64,
        "source_item_or_identifier_persisted_publicly": False,
        "status": "trusted_one_shot_acquisition_complete",
        "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
        "version": acquisition.VERSION,
        "view_file_count": 4,
    }
    receipt = _self_hashed(receipt_body, "acquisition_receipt_sha256")
    _write(private_root / acquisition.PUBLIC_RECEIPT_FILENAME, receipt)
    private_root.chmod(0o500)


class FakePlanRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes]] = []
        self.receipts: dict[str, dict[str, object]] = {}
        self.abort_calls = 0
        self.verify_closed_calls = 0

    def __call__(self, block: str, canonical_input: bytes) -> bytes:
        self.calls.append((block, canonical_input))
        inputs = typed_plan_contract.parse_input(canonical_input)
        rows = []
        completion = json.dumps(
            {
                "entity_facets": ["Acme"],
                "metric_facets": ["revenue"],
                "operation": "COMPARE",
                "relation_query": "Acme revenue comparison",
                "time_facets": ["2023", "2024"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for row in inputs:
            rows.append(
                typed_plan_contract.build_output_item(
                    item=row,
                    completion=completion,
                    completion_token_count=20,
                    prompt_projection_sha256=hashlib.sha256(
                        f"projection-{block}-{row.ordinal}".encode()
                    ).hexdigest(),
                    prompt_sha256=hashlib.sha256(
                        f"prompt-{block}-{row.ordinal}".encode()
                    ).hexdigest(),
                    prompt_token_count=100,
                )
            )
        raw_output = typed_plan_contract.canonical_json_bytes(
            typed_plan_contract.output_payload(rows)
        )
        worker_pid = 1000 + len(self.calls)
        scope = f"qwen-{block}-{worker_pid}"
        unit_name_sha256 = hashlib.sha256(f"{scope}-unit".encode()).hexdigest()
        control_group_sha256 = hashlib.sha256(
            f"{scope}-control-group".encode()
        ).hexdigest()
        self.receipts[block] = {
            "batch_size": 4,
            "block": block,
            "filesystem_isolation": adapters.SYSTEMD_FILESYSTEM_ISOLATION,
            "input_sha256": hashlib.sha256(canonical_input).hexdigest(),
            "item_count": len(inputs),
            "model_execution_finished_monotonic_ns": 500,
            "model_execution_started_monotonic_ns": 100,
            "model_context_tokens": 32768,
            "network_properties": list(adapters.SYSTEMD_NETWORK_PROPERTIES),
            "output_sha256": hashlib.sha256(raw_output).hexdigest(),
            "physical_GPU": "1",
            "schema": adapters.TYPED_PLAN_TRANSPORT_SCHEMA,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stdout_sha256": hashlib.sha256(b"passed").hexdigest(),
            "systemd_unit_closure": _systemd_unit_closure(
                unit_name_sha256=unit_name_sha256,
                control_group_sha256=control_group_sha256,
                scope=scope,
            ),
            "systemd_unit_name_sha256": unit_name_sha256,
            "worker_pid": worker_pid,
        }
        return raw_output

    def transport_receipt(self, block: str):
        return dict(self.receipts[block])

    def abort_all_workers(self):
        self.abort_calls += 1
        return ()

    def verify_all_workers_closed(self):
        self.verify_closed_calls += 1
        return ()


class FakeEncoder:
    def encode(self, texts):
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            first = int.from_bytes(digest[:2], "big") % 384
            second = int.from_bytes(digest[2:4], "big") % 384
            matrix[index, first] = np.float32(1.0)
            if second != first:
                matrix[index, second] = np.float32(0.5)
            matrix[index] /= np.linalg.norm(matrix[index]).astype(np.float32)
        return matrix


class FakeHippoRunner:
    def __init__(self, *, outside_corpus: bool = False) -> None:
        self.outside_corpus = outside_corpus
        self.calls: list[tuple[str, str]] = []
        self.receipts: dict[tuple[str, str], dict[str, object]] = {}
        self.abort_calls = 0
        self.verify_closed_calls = 0

    def __call__(self, block: str, commitment: str, canonical_input: bytes) -> bytes:
        self.calls.append((block, commitment))
        query, units = hipporag_contract.parse_input(canonical_input)
        top = [row.unit_id for row in units[:5]]
        if self.outside_corpus:
            top[-1] = "P:99"
        payload = hipporag_contract.output_payload(
            top_unit_ids=top,
            graph_nodes=7,
            graph_edges=6,
            unit_count=len(units),
            input_sha256=hipporag_contract.input_binding_sha256(query, units),
        )
        raw_output = hipporag_contract.canonical_json_bytes(payload)
        worker_pid = 2000 + len(self.calls)
        scope = f"hippo-{block}-{commitment}-{worker_pid}"
        unit_name_sha256 = hashlib.sha256(f"{scope}-unit".encode()).hexdigest()
        control_group_sha256 = hashlib.sha256(
            f"{scope}-control-group".encode()
        ).hexdigest()
        start_policy = _systemd_start_policy(
            unit_name_sha256=unit_name_sha256,
            control_group_sha256=control_group_sha256,
            worker_pid=worker_pid,
            scope=scope,
        )
        self.receipts[(block, commitment)] = {
            "CPU_threads": 2,
            "block": block,
            "configured_torch_interop_threads": 1,
            "configured_torch_intraop_threads": 1,
            "filesystem_isolation": adapters.SYSTEMD_FILESYSTEM_ISOLATION,
            "input_file_sha256": hashlib.sha256(canonical_input).hexdigest(),
            "input_semantic_sha256": payload["input_sha256"],
            "item_commitment_sha256": commitment,
            "model_execution_finished_monotonic_ns": 300,
            "model_execution_started_monotonic_ns": 200,
            "network_properties": list(adapters.SYSTEMD_NETWORK_PROPERTIES),
            "observed_process_thread_peak": 1,
            "output_file_sha256": hashlib.sha256(raw_output).hexdigest(),
            "schema": adapters.HIPPO_TRANSPORT_SCHEMA,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stdout_sha256": hashlib.sha256(b"passed").hexdigest(),
            "systemd_start_policy": start_policy,
            "systemd_start_policy_sha256": acquisition.stable_hash(start_policy),
            "systemd_tasks_max": 3,
            "systemd_unit_closure": _systemd_unit_closure(
                unit_name_sha256=unit_name_sha256,
                control_group_sha256=control_group_sha256,
                scope=scope,
            ),
            "systemd_unit_name_sha256": unit_name_sha256,
            "thread_monitor_process_reservation": 1,
            "visible_GPU": "",
            "worker_pid": worker_pid,
            "maximum_worker_process_threads": 2,
        }
        return raw_output

    def transport_receipt(self, block: str, commitment: str):
        return dict(self.receipts[(block, commitment)])

    def abort_all_workers(self):
        self.abort_calls += 1
        return ()

    def verify_all_workers_closed(self):
        self.verify_closed_calls += 1
        return ()


class InferenceOverlapProbe:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.qwen_entered = threading.Event()
        self.hippo_entered = threading.Event()
        self.qwen_live = False
        self.hippo_live = 0
        self.runner_overlap_observed = False

    def enter_qwen(self) -> None:
        with self.lock:
            self.qwen_live = True
        self.qwen_entered.set()
        assert self.hippo_entered.wait(timeout=5.0)
        with self.lock:
            self.runner_overlap_observed = (
                self.runner_overlap_observed or self.hippo_live > 0
            )

    def leave_qwen(self) -> None:
        with self.lock:
            self.qwen_live = False

    def enter_hippo(self) -> None:
        with self.lock:
            self.hippo_live += 1
            self.runner_overlap_observed = (
                self.runner_overlap_observed or self.qwen_live
            )
        self.hippo_entered.set()
        assert self.qwen_entered.wait(timeout=5.0)

    def leave_hippo(self) -> None:
        with self.lock:
            self.hippo_live -= 1


class ProbedPlanRunner(FakePlanRunner):
    def __init__(self, probe: InferenceOverlapProbe) -> None:
        super().__init__()
        self.probe = probe

    def __call__(self, block: str, canonical_input: bytes) -> bytes:
        if block != "A_hold":
            return super().__call__(block, canonical_input)
        self.probe.enter_qwen()
        try:
            return super().__call__(block, canonical_input)
        finally:
            self.probe.leave_qwen()


class ProbedHippoRunner(FakeHippoRunner):
    def __init__(self, probe: InferenceOverlapProbe) -> None:
        super().__init__()
        self.probe = probe

    def __call__(self, block: str, commitment: str, canonical_input: bytes) -> bytes:
        if block != "A_hold":
            return super().__call__(block, commitment, canonical_input)
        self.probe.enter_hippo()
        try:
            return super().__call__(block, commitment, canonical_input)
        finally:
            self.probe.leave_hippo()


class NonOverlappingIntervalHippoRunner(ProbedHippoRunner):
    def __call__(self, block: str, commitment: str, canonical_input: bytes) -> bytes:
        raw = super().__call__(block, commitment, canonical_input)
        receipt = self.receipts[(block, commitment)]
        receipt["model_execution_started_monotonic_ns"] = 600
        receipt["model_execution_finished_monotonic_ns"] = 700
        return raw


class DriftingTransportHippoRunner(FakeHippoRunner):
    def __init__(self) -> None:
        super().__init__()
        self.receipt_reads: dict[tuple[str, str], int] = {}

    def transport_receipt(self, block: str, commitment: str):
        key = (block, commitment)
        self.receipt_reads[key] = self.receipt_reads.get(key, 0) + 1
        receipt = super().transport_receipt(block, commitment)
        if self.receipt_reads[key] > 1:
            receipt["stdout_sha256"] = "f" * 64
        return receipt


class InvalidPlanRunner(FakePlanRunner):
    def __call__(self, block: str, canonical_input: bytes) -> bytes:
        super().__call__(block, canonical_input)
        return b"{}\n"


def _subject(tmp_path: Path, *, hippo=None, plan=None):
    project = tmp_path / "project"
    project.mkdir(parents=True)
    _custody_payloads(project)
    receipt_paths = _runtime_receipts(tmp_path / "receipts")
    control_root = project / acquisition.FORMAL_ROOT_RELATIVE / "execution"
    plan = plan or FakePlanRunner()
    hippo = hippo or FakeHippoRunner()
    runtime = adapters.ProductionRuntimeAdapter(
        control_root=control_root,
        receipt_paths=receipt_paths,
        typed_plan_runner=plan,
        minilm_encoder=FakeEncoder(),
        hippo_runner=hippo,
    )
    custody = adapters.TrustedAcquisitionAdapter(
        project_root=project,
        runtime=runtime,
        control_root=control_root,
    )
    return project, runtime, custody, plan, hippo


def _seal_test_evidence(
    runtime: adapters.ProductionRuntimeAdapter,
    name: str,
) -> str:
    schemas = {
        "A_form_fit": f"{controller.VERSION}_A_form_evaluator_fit_evidence_v1",
        "F_search_policy_freeze": f"{controller.VERSION}_F_search_policy_freeze_v1",
        "A_hold_score": f"{controller.VERSION}_A_hold_offline_score_v1",
        "M_search_score": f"{controller.VERSION}_M_search_offline_score_v1",
    }
    payload = {"schema": schemas[name], "synthetic_adapter_test": True}
    evidence_sha = hashlib.sha256(_canonical(payload).rstrip(b"\n")).hexdigest()
    assert runtime.persist_offline_evidence(name, payload, evidence_sha) == evidence_sha
    return evidence_sha


def test_preflight_claim_and_private_view_compile_through_byte_contract(local_tmp) -> None:
    _project, runtime, custody, plan, _hippo = _subject(local_tmp)
    preflight = runtime.preflight()
    receipt_sha = custody.claim_one_shot()
    view = custody.load_block_view("A_form", None)
    assert preflight.qualified is True
    assert len(receipt_sha) == 64
    assert view.block == "A_form"
    assert len(view.items) == 48
    assert len(plan.calls) == 1
    request_items = typed_plan_contract.parse_input(plan.calls[0][1])
    assert len(request_items) == 48
    assert all("item_commitment" not in repr(row) for row in request_items)
    assert set(controller.ItemView.__dataclass_fields__) == {
        "item_commitment_sha256",
        "plan",
        "redundancy_features",
        "units",
    }
    assert (runtime.control_root / "stages/A_form/block.preparation.json").is_file()
    assert stat.S_IMODE(
        (runtime.control_root / "execution.claim.json").stat().st_mode
    ) == 0o600


def test_private_view_file_and_semantic_hashes_are_both_enforced(local_tmp) -> None:
    project, runtime, custody, _plan, _hippo = _subject(local_tmp)
    runtime.preflight()
    custody.claim_one_shot()
    path = (
        project
        / acquisition.ACQUISITION_ROOT_RELATIVE
        / acquisition.VIEW_FILENAMES["A_form"]
    )
    path.chmod(0o700)
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="mode"):
        custody.load_block_view("A_form", None)


def test_m_view_is_never_touched_before_durable_epoch_authorization(local_tmp) -> None:
    _project, runtime, custody, _plan, _hippo = _subject(local_tmp / "unauthorized")
    runtime.preflight()
    custody.claim_one_shot()
    touched: list[bool] = []
    original = custody._read_bound_private

    def wrapped(**kwargs):
        if kwargs["filename"] == acquisition.VIEW_FILENAMES["M_search"]:
            seal = runtime.control_root / "M_search.epoch_authorization.sealed.json"
            touched.append(seal.is_file() and bool(seal.read_bytes()))
        return original(**kwargs)

    custody._read_bound_private = wrapped  # type: ignore[method-assign]
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="lacks"):
        custody.load_block_view("M_search", None)
    assert touched == []
    _project, runtime, custody, _plan, _hippo = _subject(local_tmp / "authorized")
    runtime.preflight()
    custody.claim_one_shot()
    touched = []
    original = custody._read_bound_private

    def wrapped_authorized(**kwargs):
        if kwargs["filename"] == acquisition.VIEW_FILENAMES["M_search"]:
            seal = runtime.control_root / "M_search.epoch_authorization.sealed.json"
            touched.append(seal.is_file() and bool(seal.read_bytes()))
        return original(**kwargs)

    custody._read_bound_private = wrapped_authorized  # type: ignore[method-assign]
    policy_sha = _seal_test_evidence(runtime, "F_search_policy_freeze")
    hold_sha = _seal_test_evidence(runtime, "A_hold_score")
    authorization = controller.EpochAuthorization(
        a_hold_score_sha256=hold_sha,
        policy_freeze_sha256=policy_sha,
    )
    view = custody.load_block_view("M_search", authorization)
    assert view.block == "M_search"
    assert touched == [True]


def test_hippo_output_must_bind_input_and_remain_in_common_corpus(local_tmp) -> None:
    _project, runtime, custody, _plan, _hippo = _subject(
        local_tmp, hippo=FakeHippoRunner(outside_corpus=True)
    )
    runtime.preflight()
    custody.claim_one_shot()
    _seal_test_evidence(runtime, "F_search_policy_freeze")
    view = custody.load_block_view("A_hold", None)
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="item binding"):
        runtime.run_hippo("A_hold", view.items[0])
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="replay"):
        runtime.run_hippo("A_hold", view.items[0])


def test_worker_transport_receipt_is_not_a_self_reported_zero_counter(local_tmp) -> None:
    _project, runtime, custody, plan, _hippo = _subject(local_tmp)
    original = plan.transport_receipt

    def tampered(block: str):
        receipt = original(block)
        receipt["physical_GPU"] = "0"
        return receipt

    plan.transport_receipt = tampered  # type: ignore[method-assign]
    runtime.preflight()
    custody.claim_one_shot()
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="transport binding"):
        custody.load_block_view("A_form", None)


@pytest.mark.parametrize("drifted_pid", (True, 1000.5))
def test_adapter_rejects_bool_or_float_worker_pid_transport_tamper(
    local_tmp, drifted_pid: object
) -> None:
    _project, runtime, custody, plan, _hippo = _subject(local_tmp)
    original = plan.transport_receipt

    def tampered(block: str):
        receipt = original(block)
        receipt["worker_pid"] = drifted_pid
        return receipt

    plan.transport_receipt = tampered  # type: ignore[method-assign]
    runtime.preflight()
    custody.claim_one_shot()
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="worker PID"):
        custody.load_block_view("A_form", None)


def test_real_controller_uses_durable_archive_postflight_before_label_open(local_tmp) -> None:
    _project, runtime, custody, _plan, hippo = _subject(local_tmp)
    label_open_checks: list[tuple[str, bool, bool]] = []
    original = custody._read_bound_private

    def wrapped(**kwargs):
        filename = kwargs["filename"]
        for block, expected in acquisition.LABEL_FILENAMES.items():
            if filename == expected:
                stage = runtime.control_root / "stages" / block
                label_open_checks.append(
                    (
                        block,
                        (stage / "action.archive.json").is_file(),
                        (stage / "runtime.postflight.json").is_file(),
                    )
                )
        return original(**kwargs)

    custody._read_bound_private = wrapped  # type: ignore[method-assign]
    result = controller.TatqaP19FormalController(
        acquisition=custody, runtime=runtime
    ).run()
    assert result.status == "valid_nonpromotion"
    assert result.primary_evaluated is True
    assert result.primary_value is False
    assert result.m_view_released is False
    assert [row[0] for row in label_open_checks] == ["A_form", "A_hold"]
    assert all(archive and postflight for _block, archive, postflight in label_open_checks)
    assert len(hippo.calls) == controller.BLOCK_ITEM_COUNTS["A_hold"]
    assert runtime.sealed_stage_receipts("A_hold") == (
        result.a_hold_archive.archive_sha256,  # type: ignore[union-attr]
        result.a_hold_score.postflight_sha256,  # type: ignore[union-attr]
    )
    archive_envelope = json.loads(
        (runtime.control_root / "stages/A_hold/action.archive.json").read_text()
    )
    postflight_envelope = json.loads(
        (runtime.control_root / "stages/A_hold/runtime.postflight.json").read_text()
    )
    assert len(archive_envelope["worker_pids"]) == 31
    assert len(archive_envelope["transport_receipts"]) == 31
    assert all(
        hashlib.sha256(
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        == declared
        for row, declared in zip(
            archive_envelope["transport_receipts"],
            archive_envelope["transport_receipt_sha256s"],
            strict=True,
        )
    )
    assert archive_envelope["transport_receipt_aggregate_sha256"] == (
        postflight_envelope["transport_receipt_aggregate_sha256"]
    )
    assert len(archive_envelope["durable_archive_receipt_sha256"]) == 64
    assert len(postflight_envelope["durable_postflight_receipt_sha256"]) == 64


def test_preflight_receipt_tamper_is_terminal_and_not_replayed(local_tmp) -> None:
    _project, runtime, _custody, _plan, _hippo = _subject(local_tmp)
    path = runtime.receipt_paths.production_canary
    path.write_bytes(path.read_bytes().replace(b'"qualified":true', b'"qualified":false'))
    with pytest.raises(adapters.TatqaP19FormalAdapterError):
        runtime.preflight()
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="replay"):
        runtime.preflight()


def test_actual_qwen_and_hippo_runners_overlap_with_eager_model_submission(
    local_tmp,
) -> None:
    probe = InferenceOverlapProbe()
    plan = ProbedPlanRunner(probe)
    hippo = ProbedHippoRunner(probe)
    _project, runtime, custody, _plan, _hippo = _subject(
        local_tmp, plan=plan, hippo=hippo
    )
    result = controller.TatqaP19FormalController(
        acquisition=custody, runtime=runtime
    ).run()
    assert result.status == "valid_nonpromotion"
    assert probe.runner_overlap_observed is True
    receipt = dict(runtime.preparation_inference_receipt("A_hold"))
    assert receipt["actual_model_future_expected_count"] == 31
    assert receipt["actual_model_future_submit_count_before_first_join"] == 31
    assert receipt["all_actual_model_futures_submitted_before_first_join"] is True
    assert receipt["qwen_batch_item_count"] == 30
    assert receipt["qwen_batch_submitted_count"] == 1
    assert receipt["qwen_batch_terminal_count"] == 1
    assert receipt["hippo_future_submitted_count"] == 30
    assert receipt["hippo_future_terminal_count"] == 30
    assert receipt["hippo_future_consumed_count"] == 30
    assert receipt["hippo_actual_concurrency_cap"] == 8
    assert 1 <= receipt["hippo_actual_observed_peak"] <= 8
    assert receipt["qwen_hippo_independent_executors"] is True
    assert receipt["qwen_hippo_overlap_observed"] is True
    assert len(receipt["qwen_hippo_overlap_witness_item_commitments"]) == 30
    assert len(receipt["hippo_transport_receipts"]) == 30
    assert receipt["qwen_transport_receipt"][
        "model_execution_started_monotonic_ns"
    ] < receipt["qwen_transport_receipt"][
        "model_execution_finished_monotonic_ns"
    ]
    assert receipt["minilm_raw_compiled_item_count"] == 30
    assert len(receipt["hippo_worker_pids"]) == 30
    assert len(receipt["hippo_worker_receipt_sha256s"]) == 30
    assert receipt["retry_replay_resample_provider_switch"] == 0
    persisted = json.loads(
        (
            runtime.control_root
            / "stages/A_hold/preparation.inference.json"
        ).read_text(encoding="ascii")
    )
    assert persisted == receipt
    assert runtime._inference_preparations["A_hold"].executors_closed is True


def test_parent_runner_overlap_cannot_replace_worker_interval_overlap(local_tmp) -> None:
    probe = InferenceOverlapProbe()
    plan = ProbedPlanRunner(probe)
    hippo = NonOverlappingIntervalHippoRunner(probe)
    _project, runtime, custody, _plan, _hippo = _subject(
        local_tmp, plan=plan, hippo=hippo
    )
    result = controller.TatqaP19FormalController(
        acquisition=custody, runtime=runtime
    ).run()
    assert probe.runner_overlap_observed is True
    assert result.status == "implementation_or_runtime_invalid"
    assert result.failure_stage == "A_hold_eager_actions"
    assert runtime._inference_preparations["A_hold"].executors_closed is True


def test_postflight_reopens_and_rejects_full_transport_receipt_drift(local_tmp) -> None:
    hippo = DriftingTransportHippoRunner()
    _project, runtime, custody, _plan, _hippo = _subject(
        local_tmp, hippo=hippo
    )
    result = controller.TatqaP19FormalController(
        acquisition=custody, runtime=runtime
    ).run()
    assert result.status == "implementation_or_runtime_invalid"
    assert result.failure_stage == "A_hold_eager_actions"
    assert runtime._inference_preparations["A_hold"].executors_closed is True


def test_archive_validation_failure_after_prepare_reaps_every_executor(local_tmp) -> None:
    _project, runtime, custody, plan, hippo = _subject(local_tmp)
    original = runtime.preparation_inference_receipt

    def invalid_receipt(block: str):
        receipt = dict(original(block))
        receipt["qwen_batch_terminal_count"] = 0
        return receipt

    runtime.preparation_inference_receipt = invalid_receipt  # type: ignore[method-assign]
    result = controller.TatqaP19FormalController(
        acquisition=custody, runtime=runtime
    ).run()
    assert result.status == "implementation_or_runtime_invalid"
    assert result.failure_stage == "A_form_eager_actions"
    runtime.verify_all_inference_closed()
    assert runtime._inference_preparations["A_form"].executors_closed is True
    assert plan.abort_calls >= 2
    assert hippo.abort_calls >= 2
    assert plan.verify_closed_calls >= 1
    assert hippo.verify_closed_calls >= 1
    assert not any(
        thread.name.startswith("p18-A_form-qwen-inference")
        for thread in threading.enumerate()
    )


def test_offline_evidence_is_exact_exclusive_and_reopened_before_m_seal(
    local_tmp,
) -> None:
    _project, runtime, custody, _plan, _hippo = _subject(local_tmp)
    runtime.preflight()
    custody.claim_one_shot()
    policy_sha = _seal_test_evidence(runtime, "F_search_policy_freeze")
    hold_sha = _seal_test_evidence(runtime, "A_hold_score")
    evidence_path = runtime.control_root / "evidence/A_hold_score.json"
    assert stat.S_IMODE(evidence_path.stat().st_mode) == 0o600
    runtime.verify_offline_evidence("A_hold_score", hold_sha)
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="replay"):
        runtime.persist_offline_evidence(
            "A_hold_score",
            {
                "schema": f"{controller.VERSION}_A_hold_offline_score_v1",
                "synthetic_adapter_test": True,
            },
            hold_sha,
        )
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="predeclared"):
        runtime.persist_offline_evidence("not_allowed", {}, "a" * 64)

    raw = evidence_path.read_bytes()
    evidence_path.write_bytes(raw.replace(b'"synthetic_adapter_test":true', b'"synthetic_adapter_test":false'))
    authorization = controller.EpochAuthorization(
        a_hold_score_sha256=hold_sha,
        policy_freeze_sha256=policy_sha,
    )
    with pytest.raises(adapters.TatqaP19FormalAdapterError):
        custody.load_block_view("M_search", authorization)
    assert not (
        runtime.control_root / "M_search.epoch_authorization.sealed.json"
    ).exists()


@pytest.mark.parametrize(
    ("block", "required_name"),
    (
        ("F_search", "A_form_fit"),
        ("A_hold", "F_search_policy_freeze"),
    ),
)
def test_prior_offline_evidence_is_reopened_before_downstream_view_touch(
    local_tmp, block: str, required_name: str
) -> None:
    _project, runtime, custody, _plan, _hippo = _subject(local_tmp)
    runtime.preflight()
    custody.claim_one_shot()
    touched: list[str] = []
    original = custody._read_bound_private

    def wrapped(**kwargs):
        if kwargs["filename"] == acquisition.VIEW_FILENAMES[block]:
            touched.append(block)
        return original(**kwargs)

    custody._read_bound_private = wrapped  # type: ignore[method-assign]
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="absent"):
        custody.load_block_view(block, None)
    assert touched == []
    assert not (runtime.control_root / f"evidence/{required_name}.json").exists()


def test_invalid_qwen_terminal_synchronously_reaps_prestarted_executors(
    local_tmp,
) -> None:
    _project, runtime, custody, _plan, _hippo = _subject(
        local_tmp, plan=InvalidPlanRunner()
    )
    runtime.preflight()
    custody.claim_one_shot()
    _seal_test_evidence(runtime, "F_search_policy_freeze")
    with pytest.raises(adapters.TatqaP19FormalAdapterError, match="output drifted"):
        custody.load_block_view("A_hold", None)
    assert not any(
        thread.name.startswith("p18-A_hold-qwen-inference")
        or thread.name.startswith("p18-A_hold-hippo-inference")
        for thread in threading.enumerate()
    )
