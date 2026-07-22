from __future__ import annotations

import concurrent.futures
from collections import Counter
from dataclasses import replace
import hashlib
import inspect
import threading
from typing import Sequence

import pytest

from assumption_agent.benchmarks import tatqa_p18_formal_controller_v1 as controller
from assumption_agent.benchmarks import tatqa_p18_typed_evaluator_core_v1 as core


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


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
        "schema": controller.SYSTEMD_UNIT_CLOSURE_SCHEMA,
        "sub_state": "dead",
        "systemctl_reset_failed_returncode": 0,
        "systemctl_reset_failed_stderr_sha256": _sha(f"{scope}-reset-stderr"),
        "systemctl_reset_failed_stdout_sha256": _sha(f"{scope}-reset-stdout"),
        "systemctl_show_returncode": 0,
        "systemctl_show_stderr_sha256": _sha(f"{scope}-closed-show-stderr"),
        "systemctl_show_stdout_sha256": _sha(f"{scope}-closed-show-stdout"),
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
        "schema": controller.SYSTEMD_START_POLICY_SCHEMA,
        "sub_state": "running",
        "systemctl_show_returncode": 0,
        "systemctl_show_stderr_sha256": _sha(f"{scope}-start-show-stderr"),
        "systemctl_show_stdout_sha256": _sha(f"{scope}-start-show-stdout"),
        "tasks_max": 3,
        "unit_name_sha256": unit_name_sha256,
    }


def _hippo_systemd_fields(*, block: str, item: str, worker_pid: int) -> dict[str, object]:
    scope = f"hippo-{block}-{item}-{worker_pid}"
    unit_name_sha256 = _sha(f"{scope}-unit")
    control_group_sha256 = _sha(f"{scope}-control-group")
    start_policy = _systemd_start_policy(
        unit_name_sha256=unit_name_sha256,
        control_group_sha256=control_group_sha256,
        worker_pid=worker_pid,
        scope=scope,
    )
    return {
        "maximum_worker_process_threads": 2,
        "systemd_start_policy": start_policy,
        "systemd_start_policy_sha256": controller._canonical_hash(start_policy),
        "systemd_tasks_max": 3,
        "systemd_unit_closure": _systemd_unit_closure(
            unit_name_sha256=unit_name_sha256,
            control_group_sha256=control_group_sha256,
            scope=scope,
        ),
        "systemd_unit_name_sha256": unit_name_sha256,
        "thread_monitor_process_reservation": 1,
    }


def _plan() -> core.TypedPlan:
    return core.TypedPlan(
        entity_facets=("Acme",),
        metric_facets=("revenue",),
        time_facets=("2024",),
        operation="COMPARE",
        relation_query="Acme revenue in 2024",
    )


def _unit(
    unit_id: str,
    facets: tuple[int, int, int, int],
    *,
    edges: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    operand: int = 0,
    similarity: int = 0,
) -> core.CanonicalUnit:
    return core.CanonicalUnit(
        unit_id=unit_id,
        facet_coverage=facets,
        typed_edge_features=edges,
        numeric_or_time_operand_coverage=operand,
        full_question_similarity=similarity,
    )


def _units() -> tuple[core.CanonicalUnit, ...]:
    # P0: T:0,T:1,P:1,T:2,T:3.  P1 retains the first three and uses the
    # query-anchored cross-modal residuals P:2,P:3.  Paragraph order is
    # strictly positive under the canonical serialization contract.
    return (
        _unit("T:0", (1, 1, 1, 1), edges=(5, 0, 0, 0, 0), similarity=100),
        _unit("T:1", (1, 1, 1, 0), edges=(4, 0, 0, 0, 0), similarity=90),
        _unit("P:1", (1, 1, 0, 0), edges=(3, 0, 0, 0, 0), similarity=80),
        _unit("T:2", (1, 0, 1, 0), edges=(2, 0, 0, 0, 0), operand=1, similarity=70),
        _unit("T:3", (1, 0, 1, 0), edges=(2, 0, 0, 0, 0), operand=1, similarity=60),
        _unit("P:2", (0, 0, 0, 0), edges=(0, 0, 0, 9, 0), operand=5, similarity=1),
        _unit("P:3", (0, 0, 0, 0), edges=(0, 0, 0, 9, 0), operand=5, similarity=1),
    )


def _units_for_candidate_expansion(
    outside_p0_count: int,
) -> tuple[core.CanonicalUnit, ...]:
    if outside_p0_count not in {0, 1, 2}:
        raise ValueError("synthetic candidate expansion must be zero, one, or two")
    units = list(_units())
    if outside_p0_count == 0:
        units[5] = _unit("P:2", (0, 0, 0, 0), similarity=1)
    if outside_p0_count <= 1:
        units[6] = _unit("P:3", (0, 0, 0, 0), similarity=1)
    return tuple(units)


def _views() -> dict[str, controller.BlockView]:
    return {
        block: controller.BlockView(
            block=block,
            items=tuple(
                controller.ItemView(
                    item_commitment_sha256=_sha(f"{block}-{index}"),
                    plan=_plan(),
                    units=(
                        _units_for_candidate_expansion(index % 3)
                        if block in {"A_hold", "M_search"}
                        else _units()
                    ),
                )
                for index in range(count)
            ),
        )
        for block, count in controller.BLOCK_ITEM_COUNTS.items()
    }


def _label_pack(
    view: controller.BlockView, *, favor_p1: bool
) -> controller.LabelPack:
    per_family = controller.BLOCK_FAMILY_COUNTS[view.block]
    gold = ("P:2", "P:3") if favor_p1 else ("T:2", "T:3")
    return controller.LabelPack(
        block=view.block,
        rows=tuple(
            controller.LabelRow(
                item_commitment_sha256=item.item_commitment_sha256,
                family=controller.FAMILY_ORDER[index // per_family],
                canonical_gold_units=gold,
            )
            for index, item in enumerate(view.items)
        ),
    )


class SyntheticAcquisition:
    def __init__(
        self,
        *,
        promote: bool,
        events: list[tuple[object, ...]],
        invalid_gold_block: str | None = None,
    ) -> None:
        self.views = _views()
        self.labels = {
            "A_form": _label_pack(self.views["A_form"], favor_p1=promote),
            "A_hold": _label_pack(self.views["A_hold"], favor_p1=promote),
            "M_search": _label_pack(self.views["M_search"], favor_p1=True),
        }
        if invalid_gold_block is not None:
            pack = self.labels[invalid_gold_block]
            rows = list(pack.rows)
            rows[0] = replace(rows[0], canonical_gold_units=("P:99",))
            self.labels[invalid_gold_block] = controller.LabelPack(
                block=invalid_gold_block,
                rows=tuple(rows),
            )
        self.events = events
        self.claimed = False
        self.claim_count = 0
        self.view_calls: list[str] = []
        self.label_calls: list[str] = []
        self.m_authorizations: list[controller.EpochAuthorization] = []
        self.release_receipts: list[tuple[str, str, str]] = []

    def claim_one_shot(self) -> str:
        self.events.append(("claim",))
        self.claim_count += 1
        if self.claimed:
            raise RuntimeError("synthetic durable marker already consumed")
        self.claimed = True
        return _sha("synthetic-acquisition-receipt")

    def load_block_view(
        self,
        block: str,
        authorization: controller.EpochAuthorization | None,
    ) -> controller.BlockView:
        self.events.append(("view", block))
        self.view_calls.append(block)
        if block == "M_search":
            if not isinstance(authorization, controller.EpochAuthorization):
                raise RuntimeError("M view lacks authorization")
            self.m_authorizations.append(authorization)
        elif authorization is not None:
            raise RuntimeError("premature synthetic authorization")
        return self.views[block]

    def release_label_pack(
        self, block: str, archive_sha256: str, postflight_sha256: str
    ) -> controller.LabelPack:
        # The synthetic trusted boundary itself enforces the late-release API:
        # a completed postflight must bind these exact archive and postflight
        # receipt hashes before this capability is released.
        receipts = [
            event
            for event in self.events
            if event[:2] == ("postflight", block)
        ]
        if not receipts or receipts[-1][2:] != (
            archive_sha256,
            postflight_sha256,
        ):
            raise RuntimeError("labels requested without exact sealed receipts")
        self.events.append(("labels", block))
        self.label_calls.append(block)
        self.release_receipts.append((block, archive_sha256, postflight_sha256))
        return self.labels[block]


class RecordingFuture:
    def __init__(
        self,
        future: concurrent.futures.Future[object],
        runtime: "SyntheticRuntime",
        block: str,
    ) -> None:
        self._future = future
        self._runtime = runtime
        self._block = block

    def result(self) -> object:
        expected = self._runtime.expected_submissions[self._block]
        actual = self._runtime.submit_counts[self._block]
        self._runtime.events.append(("result", self._block, actual, expected))
        if actual != expected:
            raise AssertionError("future joined before the complete eager cohort")
        self._runtime.result_counts[self._block] += 1
        return self._future.result(timeout=10)


class RecordingExecutor:
    def __init__(
        self,
        *,
        runtime: "SyntheticRuntime",
        block: str,
        kind: str,
        workers: int,
    ) -> None:
        self.runtime = runtime
        self.block = block
        self.kind = kind
        self.inner = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.runtime.executor_ids.setdefault(block, {})[kind] = id(self)

    def __enter__(self) -> "RecordingExecutor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.inner.shutdown(wait=True, cancel_futures=False)

    def submit(self, function, /, *args, **kwargs) -> RecordingFuture:
        self.runtime.submit_counts[self.block] += 1
        logical_arm = (
            args[2]
            if self.kind == "action" and len(args) >= 3
            else controller.HIPPO_ARM
        )
        self.runtime.events.append(
            (
                "submit",
                self.block,
                self.kind,
                logical_arm,
                self.runtime.submit_counts[self.block],
            )
        )
        future = self.inner.submit(function, *args, **kwargs)
        return RecordingFuture(future, self.runtime, self.block)


class SyntheticRuntime:
    def __init__(
        self,
        *,
        events: list[tuple[object, ...]],
        fail_block: str | None = None,
    ) -> None:
        self.events = events
        self.fail_block = fail_block
        self.failure_emitted = False
        self.expected_submissions = {
            block: controller.BLOCK_ITEM_COUNTS[block]
            * len(controller.BLOCK_ARMS[block])
            for block in controller.BLOCK_ARMS
        }
        self.submit_counts = CounterMap(controller.BLOCK_ARMS)
        self.result_counts = CounterMap(controller.BLOCK_ARMS)
        self.executor_ids: dict[str, dict[str, int]] = {}
        self.hippo_barriers = {
            block: threading.Barrier(controller.HIPPO_CONCURRENCY_CAP)
            for block in ("A_hold", "M_search")
        }
        self.hippo_call_counts = CounterMap(("A_hold", "M_search"))
        self.hippo_live = CounterMap(("A_hold", "M_search"))
        self.hippo_peak = CounterMap(("A_hold", "M_search"))
        self.hippo_lock = threading.Lock()
        self.hippo_items = {
            block: set() for block in ("A_hold", "M_search")
        }
        self.offline_evidence: dict[str, tuple[dict[str, object], str]] = {}
        self.inference_closed = True
        self.abort_all_count = 0

    def preflight(self) -> controller.RuntimePreflight:
        self.events.append(("preflight",))
        p0, p1 = core.build_action_pair(_plan(), _units())
        return controller.RuntimePreflight(
            qualified=True,
            public_synthetic_distinct_rankings=True,
            public_synthetic_p0_behavior_sha256=p0.behavior_sha256,
            public_synthetic_p1_behavior_sha256=p1.behavior_sha256,
            external_network_calls=0,
            api_or_online_evaluator_calls=0,
            retry_replay_resample_provider_switch=0,
        )

    def action_executor(
        self, block: str, standard_work_count: int
    ) -> RecordingExecutor:
        return RecordingExecutor(
            runtime=self,
            block=block,
            kind="action",
            workers=min(16, standard_work_count),
        )

    def hippo_executor(
        self, block: str, concurrency_cap: int
    ) -> RecordingExecutor:
        assert concurrency_cap == controller.HIPPO_CONCURRENCY_CAP
        return RecordingExecutor(
            runtime=self,
            block=block,
            kind="hippo",
            workers=concurrency_cap,
        )

    def run_raw(
        self, block: str, item: controller.ItemView
    ) -> Sequence[str]:
        if block == self.fail_block and not self.failure_emitted:
            self.failure_emitted = True
            raise RuntimeError("synthetic required RAW failure")
        # Independently injected RAW ranking: it is neither a P0 alias nor the
        # official-Hippo ranking, while remaining on the exact common corpus.
        return ("T:0", "T:1", "P:1", "T:2", "P:2")

    def run_hippo(
        self, block: str, item: controller.ItemView
    ) -> Sequence[str]:
        with self.hippo_lock:
            self.hippo_items[block].add(item.item_commitment_sha256)
            call_index = self.hippo_call_counts[block]
            self.hippo_call_counts[block] += 1
            self.hippo_live[block] += 1
            self.hippo_peak[block] = max(
                self.hippo_peak[block], self.hippo_live[block]
            )
            if self.hippo_live[block] > controller.HIPPO_CONCURRENCY_CAP:
                raise AssertionError("runtime observed a cap violation")
        try:
            # Exactly the first cap-sized cohort rendezvous.  Later work does
            # not wait, so a final partial cohort cannot deadlock.
            if call_index < controller.HIPPO_CONCURRENCY_CAP:
                self.hippo_barriers[block].wait(timeout=5)
            # Independently injected official-Hippo ranking.
            return ("T:0", "T:1", "P:1", "T:3", "P:3")
        finally:
            with self.hippo_lock:
                self.hippo_live[block] -= 1

    def preparation_inference_receipt(self, block: str) -> dict[str, object]:
        hippo_count = (
            controller.BLOCK_ITEM_COUNTS[block]
            if controller.HIPPO_ARM in controller.BLOCK_WORK_ARMS[block]
            else 0
        )
        qwen_scope = f"qwen-{block}-9999"
        qwen_unit_name_sha256 = _sha(f"{qwen_scope}-unit")
        qwen_control_group_sha256 = _sha(f"{qwen_scope}-control-group")
        qwen_transport = {
            "batch_size": 4,
            "block": block,
            "filesystem_isolation": controller.SYSTEMD_FILESYSTEM_ISOLATION,
            "input_sha256": _sha(f"{block}-qwen-input"),
            "item_count": controller.BLOCK_ITEM_COUNTS[block],
            "model_context_tokens": 32_768,
            "model_execution_finished_monotonic_ns": 10_000,
            "model_execution_started_monotonic_ns": 1_000,
            "network_properties": list(controller.SYSTEMD_NETWORK_PROPERTIES),
            "output_sha256": _sha(f"{block}-qwen-output"),
            "physical_GPU": "1",
            "schema": controller.TYPED_PLAN_TRANSPORT_SCHEMA,
            "stderr_sha256": _sha(f"{block}-qwen-stderr"),
            "stdout_sha256": _sha(f"{block}-qwen-stdout"),
            "systemd_unit_closure": _systemd_unit_closure(
                unit_name_sha256=qwen_unit_name_sha256,
                control_group_sha256=qwen_control_group_sha256,
                scope=qwen_scope,
            ),
            "systemd_unit_name_sha256": qwen_unit_name_sha256,
            "worker_pid": 9_999,
        }
        hippo_item_ids = sorted(self.hippo_items.get(block, set()))
        hippo_receipts = [
            {
                "CPU_threads": controller.HIPPO_CPU_THREADS_PER_PROCESS,
                "block": block,
                "configured_torch_interop_threads": 1,
                "configured_torch_intraop_threads": 1,
                "filesystem_isolation": controller.SYSTEMD_FILESYSTEM_ISOLATION,
                "input_file_sha256": _sha(f"{block}-{item}-hippo-input-file"),
                "input_semantic_sha256": _sha(
                    f"{block}-{item}-hippo-input-semantic"
                ),
                "item_commitment_sha256": item,
                "model_execution_finished_monotonic_ns": 3_000 + index,
                "model_execution_started_monotonic_ns": 2_000 + index,
                "network_properties": list(controller.SYSTEMD_NETWORK_PROPERTIES),
                "observed_process_thread_peak": 1,
                "output_file_sha256": _sha(f"{block}-{item}-hippo-output"),
                "schema": controller.HIPPO_TRANSPORT_SCHEMA,
                "stderr_sha256": _sha(f"{block}-{item}-hippo-stderr"),
                "stdout_sha256": _sha(f"{block}-{item}-hippo-stdout"),
                **_hippo_systemd_fields(
                    block=block,
                    item=item,
                    worker_pid=10_000 + index,
                ),
                "visible_GPU": "",
                "worker_pid": 10_000 + index,
            }
            for index, item in enumerate(hippo_item_ids)
        ]
        body: dict[str, object] = {
            "actual_model_future_expected_count": 1 + hippo_count,
            "actual_model_future_submit_count_before_first_join": 1 + hippo_count,
            "all_actual_model_futures_submitted_before_first_join": True,
            "block": block,
            "hippo_actual_concurrency_cap": (
                controller.HIPPO_CONCURRENCY_CAP if hippo_count else 0
            ),
            "hippo_actual_observed_peak": (
                self.hippo_peak[block] if hippo_count else 0
            ),
            "hippo_future_submitted_count": hippo_count,
            "hippo_future_terminal_count": hippo_count,
            "hippo_future_consumed_count": hippo_count,
            "hippo_transport_receipt_sha256s": [
                controller._canonical_hash(row) for row in hippo_receipts
            ],
            "hippo_transport_receipts": hippo_receipts,
            "hippo_worker_pids": [10_000 + index for index in range(hippo_count)],
            "hippo_worker_receipt_sha256s": [
                _sha(f"{block}-hippo-worker-{index}")
                for index in range(hippo_count)
            ],
            "hippo_executor_dedicated": bool(hippo_count),
            "minilm_raw_compiled_item_count": controller.BLOCK_ITEM_COUNTS[block],
            "qwen_batch_item_count": controller.BLOCK_ITEM_COUNTS[block],
            "qwen_batch_submitted_count": 1,
            "qwen_batch_terminal_count": 1,
            "qwen_executor_dedicated": True,
            "qwen_hippo_independent_executors": bool(hippo_count),
            "qwen_hippo_overlap_observed": bool(hippo_count),
            "qwen_hippo_overlap_witness_item_commitments": hippo_item_ids,
            "qwen_transport_receipt": qwen_transport,
            "qwen_transport_receipt_sha256": controller._canonical_hash(
                qwen_transport
            ),
            "qwen_worker_pid": 9_999,
            "qwen_worker_receipt_sha256": _sha(f"{block}-qwen-worker"),
            "retry_replay_resample_provider_switch": 0,
            "schema": (
                "tatqa_p18_formal_adapters_v1_"
                "preparation_inference_receipt_v1"
            ),
        }
        return {
            **body,
            "preparation_inference_receipt_sha256": controller._canonical_hash(body),
        }

    def postflight(
        self, block: str, archive: controller.StageArchive
    ) -> controller.RuntimePostflight:
        if self.result_counts[block] != self.expected_submissions[block]:
            raise AssertionError("postflight occurred before every future join")
        postflight = controller.RuntimePostflight(
            block=block,
            archive_sha256=archive.archive_sha256,
            runtime_ok=True,
            external_network_calls=0,
            api_or_online_evaluator_calls=0,
            retry_replay_resample_provider_switch=0,
            controller_or_worker_source_reads=0,
            controller_or_worker_label_reads=0,
            maximum_cpu_threads_per_hippo_process=(
                controller.HIPPO_CPU_THREADS_PER_PROCESS
                if controller.HIPPO_ARM in controller.BLOCK_ARMS[block]
                else 0
            ),
        )
        self.events.append(
            (
                "postflight",
                block,
                archive.archive_sha256,
                postflight.postflight_sha256,
            )
        )
        return postflight

    def abort_all_inference(self) -> None:
        self.abort_all_count += 1
        self.inference_closed = True

    def verify_all_inference_closed(self) -> None:
        if not self.inference_closed:
            raise RuntimeError("synthetic inference cohort remains live")

    def persist_offline_evidence(
        self,
        name: str,
        payload: dict[str, object],
        evidence_sha256: str,
    ) -> str:
        if name in self.offline_evidence:
            raise RuntimeError("synthetic evidence replay")
        copied = dict(payload)
        assert controller._canonical_hash(copied) == evidence_sha256
        self.offline_evidence[name] = (copied, evidence_sha256)
        self.events.append(("evidence", name, evidence_sha256))
        return evidence_sha256

    def verify_offline_evidence(self, name: str, evidence_sha256: str) -> None:
        payload, observed = self.offline_evidence[name]
        assert observed == evidence_sha256
        assert controller._canonical_hash(payload) == evidence_sha256


class CounterMap(dict[str, int]):
    def __init__(self, keys) -> None:
        super().__init__((key, 0) for key in keys)


def _run(
    *,
    promote: bool,
    fail_block: str | None = None,
    invalid_gold_block: str | None = None,
) -> tuple[
    controller.FormalDisposition,
    SyntheticAcquisition,
    SyntheticRuntime,
    controller.TatqaP18FormalController,
    list[tuple[object, ...]],
]:
    events: list[tuple[object, ...]] = []
    acquisition = SyntheticAcquisition(
        promote=promote,
        events=events,
        invalid_gold_block=invalid_gold_block,
    )
    runtime = SyntheticRuntime(events=events, fail_block=fail_block)
    subject = controller.TatqaP18FormalController(
        acquisition=acquisition, runtime=runtime
    )
    return subject.run(), acquisition, runtime, subject, events


def test_every_stage_bulk_submits_complete_cohort_before_first_join() -> None:
    result, _acquisition, runtime, _subject, events = _run(promote=True)
    assert result.status == "valid_primary_true"
    for block, expected in runtime.expected_submissions.items():
        submissions = [
            event for event in events if event[:2] == ("submit", block)
        ]
        joins = [event for event in events if event[:2] == ("result", block)]
        assert len(submissions) == expected
        assert len(joins) == expected
        assert all(event[2:] == (expected, expected) for event in joins)
        first_join_index = events.index(joins[0])
        assert all(events.index(event) < first_join_index for event in submissions)
    assert result.a_form_archive is not None
    assert result.a_form_archive.submit_count_before_first_join == 96
    assert result.f_search_archive is not None
    assert result.f_search_archive.submit_count_before_first_join == 72
    assert result.a_hold_archive is not None
    assert result.a_hold_archive.submit_count_before_first_join == 120
    assert result.m_search_archive is not None
    assert result.m_search_archive.submit_count_before_first_join == 120

    for block in ("A_hold", "M_search"):
        submitted_arms = CounterMap(controller.BLOCK_WORK_ARMS[block])
        for event in events:
            if event[:2] == ("submit", block):
                submitted_arms[event[3]] += 1
        assert submitted_arms == {
            arm: controller.BLOCK_ITEM_COUNTS[block]
            for arm in controller.BLOCK_WORK_ARMS[block]
        }


def test_hippo_uses_distinct_executor_and_actual_peak_never_exceeds_eight() -> None:
    result, _acquisition, runtime, _subject, _events = _run(promote=True)
    assert result.status == "valid_primary_true"
    for block, archive in (
        ("A_hold", result.a_hold_archive),
        ("M_search", result.m_search_archive),
    ):
        assert archive is not None
        assert runtime.executor_ids[block]["action"] != runtime.executor_ids[block]["hippo"]
        assert archive.hippo_executor_dedicated is True
        assert archive.hippo_concurrency_cap == 8
        assert archive.hippo_observed_peak == 8
        assert runtime.hippo_peak[block] == 8
        assert runtime.hippo_live[block] == 0


def test_archive_main_actions_are_design_arms_and_candidate_work_is_not_a_gate() -> None:
    result, _acquisition, _runtime, _subject, _events = _run(promote=True)
    for archive in (result.a_hold_archive, result.m_search_archive):
        assert archive is not None
        by_item: dict[str, set[str]] = {}
        for row in archive.actions:
            by_item.setdefault(row.item_commitment_sha256, set()).add(row.logical_arm)
        assert all(arms == set(controller.BLOCK_ARMS[archive.block]) for arms in by_item.values())
        candidate_by_item: dict[str, set[str]] = {}
        for row in archive.candidate_work_actions:
            candidate_by_item.setdefault(row.item_commitment_sha256, set()).add(
                row.logical_arm
            )
        assert all(arms == {"P0", "P1"} for arms in candidate_by_item.values())
        assert Counter(
            row.p1_minus_p0_features[-1]
            for row in archive.candidate_work_actions
            if row.logical_arm == "P1"
        ) == Counter({0: 10, 1: 10, 2: 10})
        payload = archive.payload()
        assert payload["predeclared_logical_arms"] == list(
            controller.BLOCK_ARMS[archive.block]
        )
        assert payload["submitted_work_arms"] == list(
            controller.BLOCK_WORK_ARMS[archive.block]
        )
        assert archive.submitted_work_count == 120
        assert archive.submitted_work_terminal_count == 120
        assert archive.logical_action_count == 120
        assert archive.logical_action_terminal_count == 120
        assert len(payload["submitted_work_action_result_sha256s"]) == 120
        assert len(payload["logical_action_result_sha256s"]) == 120
        assert "logical_work_count" not in payload
        assert "terminal_count" not in payload
        assert payload["shared_candidate_work_only_not_effect_gate"] is True
        assert payload["effect_gate_count"] == 1
        assert payload["effect_gate_scope"] == "single_predeclared_E1_effect_rule"
        assert payload["candidate_work_additional_effect_gate_count"] == 0


def test_labels_are_released_only_after_archive_postflight_and_f_has_no_labels() -> None:
    result, acquisition, _runtime, _subject, events = _run(promote=True)
    assert result.status == "valid_primary_true"
    assert acquisition.label_calls == ["A_form", "A_hold", "M_search"]
    assert "F_search" not in acquisition.label_calls
    for block in acquisition.label_calls:
        postflight_index = next(
            index
            for index, event in enumerate(events)
            if event[:2] == ("postflight", block)
        )
        label_index = events.index(("labels", block))
        last_result_index = max(
            index
            for index, event in enumerate(events)
            if event[:2] == ("result", block)
        )
        assert last_result_index < postflight_index < label_index
    assert result.policy_freeze is not None
    assert result.policy_freeze.payload()["label_pack_created_or_released"] is False
    for block, archive_sha, postflight_sha in acquisition.release_receipts:
        assert len(archive_sha) == 64
        assert len(postflight_sha) == 64
        assert block != "F_search"
    block, _archive_sha, postflight_sha = acquisition.release_receipts[0]
    with pytest.raises(RuntimeError, match="exact sealed receipts"):
        acquisition.release_label_pack(
            block,
            _sha("wrong-archive-receipt"),
            postflight_sha,
        )


def test_nonpromotion_is_valid_primary_false_and_never_releases_m() -> None:
    result, acquisition, _runtime, _subject, _events = _run(promote=False)
    assert result.status == "valid_nonpromotion"
    assert result.primary_evaluated is True
    assert result.primary_value is False
    assert result.efficacy == "false"
    assert result.a_hold_promoted is False
    assert result.epoch_transition_count == 0
    assert result.m_view_released is False
    assert result.m_labels_released is False
    assert "M_search" not in acquisition.view_calls
    assert "M_search" not in acquisition.label_calls
    assert result.m_search_archive is None
    assert result.m_search_score is None


def test_same_e0_e1_behavior_is_ordinary_nonpromotion_not_invalidation() -> None:
    result, _acquisition, _runtime, _subject, _events = _run(promote=False)
    assert result.status == "valid_nonpromotion"
    assert result.policy_freeze is not None
    assert all(row.same_behavior for row in result.policy_freeze.rows)
    assert result.a_hold_archive is not None
    by_item: dict[str, dict[str, controller.LogicalActionResult]] = {}
    for row in result.a_hold_archive.actions:
        if row.logical_arm not in {"E0", "E1"}:
            continue
        by_item.setdefault(row.item_commitment_sha256, {})[row.logical_arm] = row
    assert all(
        rows["E0"].behavior_sha256 == rows["E1"].behavior_sha256
        for rows in by_item.values()
    )
    assert result.a_hold_score is not None
    comparison = result.a_hold_score.comparison("E0")
    assert comparison.net_u == 0
    assert comparison.exact_test.exact_p == 1
    assert comparison.exact_test.promoted is False


def test_promotion_transitions_once_then_runs_m_and_unique_and_primary() -> None:
    result, acquisition, _runtime, _subject, _events = _run(promote=True)
    assert result.status == "valid_primary_true"
    assert result.primary_evaluated is True
    assert result.primary_value is True
    assert result.a_hold_promoted is True
    assert result.epoch_transition_count == 1
    assert result.epoch_authorization is not None
    assert len(acquisition.m_authorizations) == 1
    assert acquisition.m_authorizations[0] == result.epoch_authorization
    assert acquisition.view_calls.count("M_search") == 1
    assert acquisition.label_calls.count("M_search") == 1
    assert result.m_search_score is not None
    assert result.m_search_score.comparison("E0").exact_test.promoted is True
    assert result.m_search_score.comparison(
        "RAW"
    ).aggregate_and_all_families_positive
    assert result.m_search_score.comparison(
        controller.HIPPO_ARM
    ).aggregate_and_all_families_positive
    for score in (result.a_hold_score, result.m_search_score):
        assert score is not None
        assert dict(score.arm_complete_counts) == {
            "E0": 0,
            "E1": controller.BLOCK_ITEM_COUNTS[score.block] // 3,
            "RAW": 0,
            controller.HIPPO_ARM: 0,
        }
        assert score.candidate_expansion_item_count == (
            2 * controller.BLOCK_ITEM_COUNTS[score.block] // 3
        )
        assert score.candidate_expansion_unit_count == (
            controller.BLOCK_ITEM_COUNTS[score.block]
        )
    payload = result.payload()
    assert payload["primary_count"] == 1
    assert payload["primary_operator"] == "AND"


def test_raw_and_hippo_are_independent_same_corpus_actions_not_p0_aliases() -> None:
    result, _acquisition, _runtime, _subject, _events = _run(promote=True)
    assert result.a_hold_archive is not None
    by_item: dict[str, dict[str, controller.LogicalActionResult]] = {}
    for row in (
        *result.a_hold_archive.actions,
        *result.a_hold_archive.candidate_work_actions,
    ):
        by_item.setdefault(row.item_commitment_sha256, {})[row.logical_arm] = row
    for rows in by_item.values():
        assert rows["RAW"].behavior_sha256 != rows["P0"].behavior_sha256
        assert rows[controller.HIPPO_ARM].behavior_sha256 != rows["P0"].behavior_sha256
        assert rows["RAW"].behavior_sha256 != rows[controller.HIPPO_ARM].behavior_sha256
        assert rows["RAW"].selected_unit_ids == (
            "T:0",
            "T:1",
            "P:1",
            "T:2",
            "P:2",
        )
        assert rows[controller.HIPPO_ARM].selected_unit_ids == (
            "T:0",
            "T:1",
            "P:1",
            "T:3",
            "P:3",
        )


def test_required_action_failure_is_unknown_invalid_without_label_or_replay() -> None:
    result, acquisition, runtime, _subject, _events = _run(
        promote=True, fail_block="A_hold"
    )
    assert result.status == "implementation_or_runtime_invalid"
    assert result.primary_evaluated is False
    assert result.primary_value is None
    assert result.efficacy == "unknown"
    assert result.replay_authorized is False
    assert result.failure_stage == "A_hold_eager_actions"
    assert runtime.failure_emitted is True
    assert acquisition.label_calls == ["A_form"]
    assert "M_search" not in acquisition.view_calls
    assert result.m_search_archive is None


def test_controller_cannot_return_terminal_without_all_closed_proof() -> None:
    events: list[tuple[object, ...]] = []
    acquisition = SyntheticAcquisition(promote=True, events=events)
    runtime = SyntheticRuntime(events=events, fail_block="A_hold")

    def failed_abort() -> None:
        runtime.abort_all_count += 1
        runtime.inference_closed = False
        raise RuntimeError("injected abort failure")

    def failed_closure_proof() -> None:
        raise RuntimeError("injected unclosed worker")

    runtime.abort_all_inference = failed_abort  # type: ignore[method-assign]
    runtime.verify_all_inference_closed = failed_closure_proof  # type: ignore[method-assign]
    subject = controller.TatqaP18FormalController(
        acquisition=acquisition, runtime=runtime
    )
    with pytest.raises(
        controller.TatqaP18FormalControllerError,
        match="closure could not be proved",
    ):
        subject.run()
    assert runtime.abort_all_count == 1


def test_late_label_unit_must_belong_to_its_bound_item_corpus() -> None:
    result, acquisition, _runtime, _subject, _events = _run(
        promote=True,
        invalid_gold_block="A_hold",
    )
    assert result.status == "implementation_or_runtime_invalid"
    assert result.failure_stage == "A_hold_late_labels"
    assert result.primary_evaluated is False
    assert result.efficacy == "unknown"
    assert result.replay_authorized is False
    assert acquisition.label_calls == ["A_form", "A_hold"]
    assert result.m_view_released is False


def test_controller_and_acquisition_marker_are_one_shot() -> None:
    first, acquisition, runtime, subject, events = _run(promote=False)
    assert first.status == "valid_nonpromotion"
    event_count = len(events)
    second = subject.run()
    assert second.status == "implementation_or_runtime_invalid"
    assert second.failure_stage == "one_shot_reentry"
    assert second.primary_evaluated is False
    assert second.efficacy == "unknown"
    assert second.replay_authorized is False
    assert len(events) == event_count
    assert acquisition.claim_count == 1

    another = controller.TatqaP18FormalController(
        acquisition=acquisition, runtime=runtime
    ).run()
    assert another.status == "implementation_or_runtime_invalid"
    assert another.failure_stage == "one_shot_acquisition_claim"
    assert acquisition.claim_count == 2


def test_action_result_archive_and_disposition_hashes_are_canonical_and_strict() -> None:
    result, _acquisition, _runtime, _subject, _events = _run(promote=False)
    assert result.disposition_sha256 == result.disposition_sha256
    assert result.a_hold_archive is not None
    archive = result.a_hold_archive
    assert archive.archive_sha256 == archive.archive_sha256
    action = archive.actions[0]
    assert action.action_result_sha256 == action.action_result_sha256
    assert action.behavior_sha256 == core.canonical_behavior_hash(
        action.selected_unit_ids
    )
    with pytest.raises(controller.TatqaP18FormalControllerError):
        replace(action, source_action_sha256="bad")
    with pytest.raises(controller.TatqaP18FormalControllerError):
        replace(
            archive,
            logical_action_terminal_count=(
                archive.logical_action_terminal_count - 1
            ),
        )
    with pytest.raises(controller.TatqaP18FormalControllerError):
        replace(
            archive,
            submitted_work_terminal_count=(
                archive.submitted_work_terminal_count - 1
            ),
        )


def test_offline_evidence_is_full_recomputable_and_durably_sealed_in_order() -> None:
    result, _acquisition, runtime, _subject, events = _run(promote=True)
    assert result.status == "valid_primary_true"
    assert result.a_form_fit is not None
    assert result.a_hold_archive is not None
    assert result.a_hold_score is not None
    assert result.m_search_score is not None

    fit = result.a_form_fit.payload()
    assert len(fit["feature_rows"]) == controller.BLOCK_ITEM_COUNTS["A_form"]
    assert len(fit["utility_deltas"]) == controller.BLOCK_ITEM_COUNTS["A_form"]
    assert fit["E1_model_sha256"] == result.model_sha256
    archive = result.a_hold_archive.payload()
    assert len(archive["logical_action_results"]) == (
        controller.BLOCK_ITEM_COUNTS["A_hold"]
        * len(controller.BLOCK_ARMS["A_hold"])
    )
    assert [
        controller._canonical_hash(row)
        for row in archive["logical_action_results"]
    ] == archive["logical_action_result_sha256s"]
    score = result.a_hold_score.payload()
    assert len(score["item_exact_utility_rows"]) == controller.BLOCK_ITEM_COUNTS[
        "A_hold"
    ]
    assert all(
        len(row["paired_deltas"]) == controller.BLOCK_ITEM_COUNTS["A_hold"]
        for row in score["comparisons"]
    )
    assert tuple(runtime.offline_evidence) == (
        "A_form_fit",
        "F_search_policy_freeze",
        "A_hold_score",
        "M_search_score",
    )
    positions = {
        name: next(
            index
            for index, event in enumerate(events)
            if event[:2] == ("evidence", name)
        )
        for name in runtime.offline_evidence
    }
    assert positions["A_form_fit"] < next(
        index for index, event in enumerate(events) if event == ("view", "F_search")
    )
    assert positions["F_search_policy_freeze"] < next(
        index for index, event in enumerate(events) if event == ("view", "A_hold")
    )
    assert positions["A_hold_score"] < next(
        index for index, event in enumerate(events) if event == ("view", "M_search")
    )


def test_paragraph_zero_is_rejected_by_view_action_and_label_boundaries() -> None:
    with pytest.raises(core.TatqaP18TypedEvaluatorError, match="positive official order"):
        _unit("P:0", (1, 1, 1, 1), edges=(5, 0, 0, 0, 0), similarity=100)
    # Forge only to prove the controller independently rejects an object that
    # bypassed the core dataclass constructor.
    forged = object.__new__(core.CanonicalUnit)
    object.__setattr__(forged, "unit_id", "P:0")
    object.__setattr__(forged, "facet_coverage", (1, 1, 1, 1))
    object.__setattr__(forged, "typed_edge_features", (5, 0, 0, 0, 0))
    object.__setattr__(forged, "numeric_or_time_operand_coverage", 0)
    object.__setattr__(forged, "full_question_similarity", 100)
    invalid_units = (
        forged,
        *_units()[1:],
    )
    with pytest.raises(controller.TatqaP18FormalControllerError, match="unit ID"):
        controller.ItemView(
            item_commitment_sha256=_sha("invalid-paragraph-zero"),
            plan=_plan(),
            units=invalid_units,
        )
    with pytest.raises(controller.TatqaP18FormalControllerError, match="unit ID"):
        controller.LogicalActionResult(
            block="A_form",
            item_commitment_sha256=_sha("invalid-action-paragraph-zero"),
            logical_arm="P0",
            selected_policy_id=core.P0_POLICY_ID,
            selected_unit_ids=("P:0", "T:0", "T:1", "T:2", "T:3"),
            p1_minus_p0_features=(0,) * len(core.FEATURE_ORDER),
            source_action_sha256=_sha("invalid-source-action"),
        )
    with pytest.raises(controller.TatqaP18FormalControllerError, match="unit ID"):
        controller.LabelRow(
            item_commitment_sha256=_sha("invalid-label-paragraph-zero"),
            family="TEXT",
            canonical_gold_units=("P:0",),
        )


def test_controller_has_no_direct_source_file_network_or_provider_api() -> None:
    source = inspect.getsource(controller)
    assert "from pathlib" not in source
    assert "import os" not in source
    assert ".stat(" not in source
    assert ".open(" not in source
    assert "requests." not in source
    assert "urllib." not in source
    assert "socket." not in source
    assert "provider_switch(" not in source
    assert set(controller.ItemView.__dataclass_fields__) == {
        "item_commitment_sha256",
        "plan",
        "units",
        "redundancy_features",
    }
