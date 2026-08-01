from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    gscl_arn_formal_measurement_v1 as measurement,
)
from assumption_agent.benchmarks import (
    gscl_arn_formal_supervisor_v1 as supervisor,
)
from assumption_agent.benchmarks import (
    gscl_arn_intrinsic_protocol_v1 as protocol,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def test_public_source_and_fixed_root_bindings_are_exact() -> None:
    assert measurement.FORMAL_ROOT == Path(
        "/var/tmp/gscl_arn_intrinsic_formal_v1"
    )
    assert measurement.FORMAL_ROOT == supervisor.FORMAL_ROOT
    assert (
        measurement.OFFICIAL_DATASET_SHA256
        == protocol.OFFICIAL_DATASET_SHA256
        == "a866fe5341ce4a29f00f24987a12278303b2b8ad788352f549b0fe051ad4a7a8"
    )
    assert (
        measurement.OFFICIAL_METADATA_SHA256
        == protocol.OFFICIAL_METADATA_SHA256
        == "c9e91d7a49ea383eeccec5421cce9f1b0d8713c243187d840482eb1764f3317f"
    )


def test_pending_qualification_bindings_fail_before_formal_root_use() -> None:
    binding = measurement.FrozenReceiptBinding(
        path=Path("/var/tmp/pending.safe.json"),
        file_sha256="PENDING",
        self_sha256="PENDING",
    )
    with pytest.raises(
        measurement.FormalMeasurementError,
        match="qualification_receipt_binding_pending",
    ):
        measurement._validate_receipt_binding(binding)  # noqa: SLF001


def test_all_green_qualification_bindings_are_frozen() -> None:
    bindings = (
        *measurement.FROZEN_QWEN_RUNTIME_QUALIFICATION_RECEIPTS,
        measurement.FROZEN_INTERNAL_FACTORY_QUALIFICATION_RECEIPT,
    )
    assert len({binding.path for binding in bindings}) == 3
    for binding in bindings:
        assert binding.path.is_absolute()
        assert measurement._SHA256.fullmatch(  # noqa: SLF001
            binding.file_sha256
        )
        assert measurement._SHA256.fullmatch(  # noqa: SLF001
            binding.self_sha256
        )


def test_source_preflight_reads_topology_not_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "formal"
    source = root / "source"
    source.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    source.chmod(0o700)
    dataset = source / "arn.csv"
    metadata = source / "metadata.json"
    dataset.write_bytes(b"data")
    metadata.write_bytes(b"meta")
    dataset.chmod(0o600)
    metadata.chmod(0o600)
    monkeypatch.setattr(measurement, "FORMAL_ROOT", root)
    monkeypatch.setattr(measurement, "FROZEN_SOURCE_DATASET", dataset)
    monkeypatch.setattr(measurement, "FROZEN_SOURCE_METADATA", metadata)
    monkeypatch.setattr(measurement, "OFFICIAL_DATASET_SIZE", 4)
    monkeypatch.setattr(measurement, "OFFICIAL_METADATA_SIZE", 4)
    measurement._preflight_staged_source_without_opening_content()  # noqa: SLF001
    unexpected = root / "unexpected"
    unexpected.mkdir()
    with pytest.raises(
        measurement.FormalMeasurementError,
        match="formal_root_topology_invalid",
    ):
        measurement._preflight_staged_source_without_opening_content()  # noqa: SLF001


class _Store:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.written: dict[str, Any] = {}

    def ensure_directory(self, relative: str) -> None:
        assert relative == "control"

    def write_json_exclusive(
        self, relative: str, value: Any
    ) -> None:
        self.calls.append("safe_terminal")
        assert relative == "control/outer_terminal.safe.json"
        assert relative not in self.written
        self.written[relative] = value


class _FakeRuntime:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.store = _Store(calls)
        closure = SimpleNamespace(manifest={"self_hash": _digest("closure")})
        self.action = SimpleNamespace(
            receipt={"self_hash": _digest("action")},
            closure=closure,
        )
        self.invocation = SimpleNamespace(
            receipt={"one_shot_key": _digest("attempt")}
        )

    def __enter__(self) -> "_FakeRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def freeze_internal_factory_action_once(
        self, **_: Any
    ) -> Any:
        self.calls.append("freeze")
        return self.action

    def begin_once(self, action: Any) -> Any:
        assert action is self.action
        self.calls.append("begin")
        return self.invocation

    def materialize_official_packs_once(
        self, invocation: Any
    ) -> dict[str, Any]:
        assert invocation is self.invocation
        self.calls.append("materialize")
        return {
            "self_hash": _digest("source"),
            "source_sha256": measurement.OFFICIAL_DATASET_SHA256,
            "metadata_sha256": measurement.OFFICIAL_METADATA_SHA256,
            "adapted_row_count": protocol.OFFICIAL_ROW_COUNT,
            "item_content_emitted": False,
        }

    def run_internal_factory_once(
        self, invocation: Any
    ) -> dict[str, Any]:
        assert invocation is self.invocation
        self.calls.append("factory")
        return {
            "self_hash": _digest("factory"),
            "item_content_emitted": False,
        }

    def seal_four_arm_barrier_once(
        self, invocation: Any
    ) -> dict[str, Any]:
        assert invocation is self.invocation
        self.calls.append("barrier")
        return {
            "self_hash": _digest("barrier"),
            "common_item_count": 876,
            "label_opened": False,
            "item_content_emitted": False,
        }

    def run_fixed_scorer_once(
        self, invocation: Any
    ) -> dict[str, Any]:
        assert invocation is self.invocation
        self.calls.append("score")
        body = {
            "schema": supervisor.SCORE_RECEIPT_SCHEMA,
            "status": "FIXED_OFFLINE_SCORER_COMPLETED",
            "one_shot_key": self.invocation.receipt["one_shot_key"],
            "action_self_hash": self.action.receipt["self_hash"],
            "four_arm_barrier_self_hash": _digest("barrier"),
            "label_open_claim_self_hash": _digest("labels"),
            "arm_aggregates": {
                arm_id: {"accuracy": 0.5}
                for arm_id in protocol.ARM_IDS
            },
            "paired_aggregate_differences": {},
            "uncertainty_method": "fixed_offline_fixture",
            "abstain_and_error_counted_wrong": True,
            "online_or_api_evaluator_used": False,
            "effect_gate_added": False,
            "item_content_emitted": False,
        }
        return {
            **body,
            "self_hash": measurement._content_hash(body),  # noqa: SLF001
        }


def test_exact_one_shot_call_order_and_safe_aggregate_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    runtime = _FakeRuntime(calls)
    monkeypatch.setattr(
        measurement.supervisor,
        "FormalSupervisor",
        lambda: runtime,
    )
    monkeypatch.setattr(
        measurement,
        "_freeze_commitments",
        lambda **_: {"runner": _digest("runner")},
    )
    closure = runtime.action.closure
    attestation = SimpleNamespace(receipt={"self_hash": _digest("tests")})
    terminal = measurement._execute_formal_once(  # noqa: SLF001
        closure=closure,
        test_attestation=attestation,
    )
    assert calls == [
        "freeze",
        "begin",
        "materialize",
        "factory",
        "barrier",
        "score",
        "safe_terminal",
    ]
    assert terminal["offline_scorer_call_count"] == 1
    assert terminal["online_or_api_evaluation_count"] == 0
    assert terminal["effect_gate_added"] is False
    assert terminal["item_content_emitted"] is False
    assert set(terminal["arm_aggregates"]) == set(protocol.ARM_IDS)
    serialized = measurement._canonical_bytes(terminal)  # noqa: SLF001
    for forbidden in (
        b"opaque_item_id",
        b"predictions",
        b"labels",
        b"query_narrative",
        b"correct_answer",
    ):
        assert forbidden not in serialized


def test_post_begin_failure_is_sealed_once_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    runtime = _FakeRuntime(calls)

    def fail(_: Any) -> dict[str, Any]:
        calls.append("factory")
        raise supervisor.FormalSupervisorError(
            "internal_stage_subprocess_failed"
        )

    runtime.run_internal_factory_once = fail  # type: ignore[method-assign]
    monkeypatch.setattr(
        measurement.supervisor,
        "FormalSupervisor",
        lambda: runtime,
    )
    monkeypatch.setattr(
        measurement,
        "_freeze_commitments",
        lambda **_: {"runner": _digest("runner")},
    )
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="internal_stage_subprocess_failed",
    ):
        measurement._execute_formal_once(  # noqa: SLF001
            closure=runtime.action.closure,
            test_attestation=SimpleNamespace(
                receipt={"self_hash": _digest("tests")}
            ),
        )
    assert calls == [
        "freeze",
        "begin",
        "materialize",
        "factory",
        "safe_terminal",
    ]
    failed = runtime.store.written[
        "control/outer_terminal.safe.json"
    ]
    assert failed["status"] == "FAILED_AFTER_FORMAL_BEGIN_NO_RETRY_OR_REPLAY"
    assert failed["retry_or_replay_allowed"] is False
    assert failed["issue_id"] == "internal_stage_subprocess_failed"


def test_runner_has_no_effect_gate_or_online_evaluator_input() -> None:
    signature = set(inspect.signature(measurement.run_measurement).parameters)
    assert signature == set()
    source = inspect.getsource(measurement._execute_formal_once)  # noqa: SLF001
    required_in_order = (
        "freeze_internal_factory_action_once",
        "begin_once",
        "materialize_official_packs_once",
        "run_internal_factory_once",
        "seal_four_arm_barrier_once",
        "run_fixed_scorer_once",
    )
    offsets = [source.index(name) for name in required_in_order]
    assert offsets == sorted(offsets)
    assert "threshold" not in source
    assert "provider" not in source
    assert "online" not in source
    assert measurement._main(("--unfrozen-input",)) == 2  # noqa: SLF001
