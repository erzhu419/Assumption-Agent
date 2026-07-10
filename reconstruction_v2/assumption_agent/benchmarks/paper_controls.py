from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

from ..events import Event, EventSink, JsonlEventSink, NullEventSink
from ..models import SplitName, stable_hash
from ..secure_env import load_dotenv, map_legacy_model_env
from ..splits import AccessPhase, SplitAccessGuard, SplitManifest
from .paper_protocol import PaperProtocol
from .paper_report import PaperTrialRecord, read_records
from .skilllearn_lifecycle import (
    SkillLearnBackendPool,
    SkillLearnPrebuiltImageCache,
    SkillLearnSubprocessBackend,
    SkillLearnTrialBackend,
    SkillLearnTrialRequest,
    TrialVariant,
)
from .skilllearnbench import SkillLearnBenchAdapter


@dataclass(frozen=True)
class ControlSource:
    id: str
    root: Path | None


class PaperRecordStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        existing = read_records(self.path) if self.path.is_file() else ()
        self.records: dict[tuple[str, str, int, str], PaperTrialRecord] = {
            _record_key(row): row for row in existing
        }

    def assert_run_identity(
        self,
        *,
        protocol_hash: str,
        manifest_hash: str,
        evaluator_epoch: str,
    ) -> None:
        for record in self.records.values():
            if (
                record.protocol_hash != protocol_hash
                or record.manifest_hash != manifest_hash
                or record.evaluator_epoch != evaluator_epoch
            ):
                raise PermissionError("paper record file contains a different frozen run identity")

    def get(
        self,
        *,
        item_id_hash: str,
        control_id: str,
        repeat: int,
        split: str,
    ) -> PaperTrialRecord | None:
        return self.records.get((item_id_hash, control_id, repeat, split))

    def append(self, record: PaperTrialRecord) -> None:
        with self._lock:
            issues = record.validate()
            if issues:
                raise ValueError(f"invalid paper record: {issues}")
            key = _record_key(record)
            incumbent = self.records.get(key)
            if incumbent is not None and record.attempt <= incumbent.attempt:
                raise ValueError("paper trial replacement attempt must increase")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                handle.flush()
            self.records[key] = record


class PaperControlRunner:
    def __init__(
        self,
        *,
        adapter: SkillLearnBenchAdapter,
        manifest: SplitManifest,
        guard: SplitAccessGuard,
        backend: SkillLearnTrialBackend,
        protocol: PaperProtocol,
        controls: Sequence[ControlSource],
        record_store: PaperRecordStore,
        evaluator_epoch: str,
        event_sink: EventSink | None = None,
    ) -> None:
        self.adapter = adapter
        self.manifest = manifest
        self.guard = guard
        self.backend = backend
        self.protocol = protocol
        self.controls = tuple(controls)
        self.record_store = record_store
        self.evaluator_epoch = evaluator_epoch
        self.event_sink = event_sink or NullEventSink()
        self.items = {row.id: row for row in adapter.discover()}
        expected_controls = {str(row["id"]) for row in protocol.payload["controls"]}
        supplied_controls = {row.id for row in self.controls}
        if supplied_controls != expected_controls:
            raise ValueError(
                "paper controls must exactly match the frozen protocol: "
                f"missing={sorted(expected_controls - supplied_controls)}, "
                f"extra={sorted(supplied_controls - expected_controls)}"
            )
        if len(supplied_controls) != len(self.controls):
            raise ValueError("paper control IDs must be unique")
        self.record_store.assert_run_identity(
            protocol_hash=self.protocol.protocol_hash,
            manifest_hash=self.manifest.manifest_hash,
            evaluator_epoch=self.evaluator_epoch,
        )

    def run(
        self,
        item_ids: Sequence[str],
        *,
        split: SplitName,
        repeats: int,
        retry_invalid: bool = True,
        parallel_workers: int = 1,
        trace_id: str = "paper_controls",
    ) -> tuple[PaperTrialRecord, ...]:
        if split not in {SplitName.VALIDATION, SplitName.TEST}:
            raise ValueError("paper controls are restricted to validation or sealed test")
        if repeats <= 0:
            raise ValueError("paper control repeats must be positive")
        if parallel_workers <= 0:
            raise ValueError("paper control worker count must be positive")
        allowed = self.manifest.ids_for(split)
        unexpected = sorted(set(item_ids) - set(allowed))
        if unexpected:
            raise PermissionError("paper control selection is outside the frozen split")
        phase = AccessPhase.PROMOTION if split is SplitName.VALIDATION else AccessPhase.FINAL_REPORT
        for item_id in item_ids:
            self.guard.authorize(item_id, phase)
        produced: list[PaperTrialRecord] = []
        ordered_items = sorted(
            item_ids,
            key=lambda item_id: stable_hash(
                {"protocol": self.protocol.protocol_hash, "item_id": item_id, "split": split.value}
            ),
        )
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            for repeat in range(1, repeats + 1):
                pending = {
                    item_id: executor.submit(
                        self._run_item_controls,
                        item_id=item_id,
                        split=split,
                        repeat=repeat,
                        retry_invalid=retry_invalid,
                        trace_id=trace_id,
                    )
                    for item_id in ordered_items
                }
                for item_id in ordered_items:
                    produced.extend(pending[item_id].result())
        return tuple(produced)

    def _run_item_controls(
        self,
        *,
        item_id: str,
        split: SplitName,
        repeat: int,
        retry_invalid: bool,
        trace_id: str,
    ) -> tuple[PaperTrialRecord, ...]:
        controls = self._balanced_controls(item_id, repeat)
        pair_id = stable_hash(
            {
                "protocol": self.protocol.protocol_hash,
                "manifest": self.manifest.manifest_hash,
                "item_id": item_id,
                "repeat": repeat,
                "split": split.value,
            }
        )[:20]
        rows: list[PaperTrialRecord] = []
        item_id_hash = stable_hash({"item_id": item_id})
        self.event_sink.emit(
            Event(
                event="paper_control_item_schedule_started",
                stage="benchmark.skilllearn.paper_controls",
                trace_id=f"{trace_id}:{pair_id}",
                payload={
                    "item_id_hash": item_id_hash,
                    "pair_id": pair_id,
                    "repeat": repeat,
                    "split": split.value,
                    "control_order": [control.id for control in controls],
                    "within_item_parallel": False,
                },
            )
        )
        for order_index, control in enumerate(controls):
            incumbent = self.record_store.get(
                item_id_hash=item_id_hash,
                control_id=control.id,
                repeat=repeat,
                split=split.value,
            )
            if incumbent is not None and (incumbent.valid or not retry_invalid):
                rows.append(incumbent)
                self._emit_skip(incumbent, trace_id)
                continue
            attempt = (incumbent.attempt + 1) if incumbent else 1
            record = self._run_control(
                item_id=item_id,
                split=split,
                repeat=repeat,
                pair_id=pair_id,
                control=control,
                attempt=attempt,
                trace_id=f"{trace_id}:order-{order_index + 1}",
            )
            self.record_store.append(record)
            rows.append(record)
            self._emit_record(record, trace_id)
        return tuple(rows)

    def _run_control(
        self,
        *,
        item_id: str,
        split: SplitName,
        repeat: int,
        pair_id: str,
        control: ControlSource,
        attempt: int,
        trace_id: str,
    ) -> PaperTrialRecord:
        item = self.items[item_id]
        item_id_hash = stable_hash({"item_id": item_id})
        source = self._source_for(control, item.family)
        variant = TrialVariant.POLICY_OFF if source is None else TrialVariant.POLICY_ON
        request = SkillLearnTrialRequest(
            item_id=item_id,
            family=item.family,
            split=split,
            variant=variant,
            evaluator_epoch=self.evaluator_epoch,
            pair_id=pair_id,
            repeat=repeat,
            agent_id=self.backend.agent_id,
            model=self.backend.model,
            max_steps=self.backend.max_steps,
            manifest_hash=self.manifest.manifest_hash,
            program_id=None if control.root is None else control.id,
        )
        observation = self.backend.run(
            request,
            skill_source_dir=source,
            trace_id=f"{trace_id}:{pair_id}:{control.id}:r{repeat}:a{attempt}",
        )
        record = PaperTrialRecord(
            item_id_hash=item_id_hash,
            family_hash=stable_hash({"family": item.family}),
            split=split.value,
            control_id=control.id,
            protocol_hash=self.protocol.protocol_hash,
            manifest_hash=self.manifest.manifest_hash,
            evaluator_epoch=self.evaluator_epoch,
            pair_id=pair_id,
            repeat=repeat,
            attempt=attempt,
            success=observation.success,
            score=observation.score,
            valid=observation.valid,
            provider_fingerprint=observation.provider_fingerprint,
            fairness_fingerprint=observation.fairness_fingerprint,
            total_tokens=observation.total_tokens,
            steps=observation.steps,
            duration_seconds=observation.duration_seconds,
            metrics=dict(observation.metrics),
            error_type=observation.error_type,
            observation_hash=observation.observation_hash,
            prebuilt_image_key=observation.prebuilt_image_key,
            prebuilt_image_id=observation.prebuilt_image_id,
            prebuilt_cache_reused=observation.prebuilt_cache_reused,
            agent_runtime_key=observation.agent_runtime_key,
            agent_runtime_version=observation.agent_runtime_version,
        )
        return record

    def _emit_record(self, record: PaperTrialRecord, trace_id: str) -> None:
        self.event_sink.emit(
            Event(
                event="paper_control_trial_recorded",
                stage="benchmark.skilllearn.paper_controls",
                trace_id=f"{trace_id}:{record.pair_id}:{record.control_id}",
                payload={
                    "item_id_hash": record.item_id_hash,
                    "family_hash": record.family_hash,
                    "control_id": record.control_id,
                    "repeat": record.repeat,
                    "attempt": record.attempt,
                    "split": record.split,
                    "valid": record.valid,
                    "success": record.success,
                    "error_type": record.error_type,
                    "provider_fingerprint": record.provider_fingerprint,
                    "fairness_fingerprint": record.fairness_fingerprint,
                    "prebuilt_image_key": record.prebuilt_image_key,
                    "prebuilt_image_id": record.prebuilt_image_id,
                    "prebuilt_cache_reused": record.prebuilt_cache_reused,
                    "agent_runtime_key": record.agent_runtime_key,
                    "agent_runtime_version": record.agent_runtime_version,
                    "observation_hash": record.observation_hash,
                    "raw_content_persisted": False,
                },
            )
        )

    def _balanced_controls(self, item_id: str, repeat: int) -> tuple[ControlSource, ...]:
        controls = sorted(self.controls, key=lambda row: row.id)
        offset = int(
            stable_hash(
                {
                    "protocol": self.protocol.protocol_hash,
                    "item_id": item_id,
                    "repeat": repeat,
                    "order": "control_rotation",
                }
            )[:8],
            16,
        ) % len(controls)
        return tuple(controls[offset:] + controls[:offset])

    def _source_for(self, control: ControlSource, family: str) -> Path | None:
        if control.root is None:
            return None
        family_source = control.root / family
        if family_source.is_dir():
            return family_source
        return None

    def _emit_skip(self, record: PaperTrialRecord, trace_id: str) -> None:
        self.event_sink.emit(
            Event(
                event="paper_control_trial_reused",
                stage="benchmark.skilllearn.paper_controls",
                trace_id=trace_id,
                payload={
                    "item_id_hash": record.item_id_hash,
                    "control_id": record.control_id,
                    "repeat": record.repeat,
                    "attempt": record.attempt,
                    "split": record.split,
                    "valid": record.valid,
                    "observation_hash": record.observation_hash,
                },
            )
        )


def validate_freeze_receipt(
    receipt: Mapping[str, Any],
    *,
    protocol: PaperProtocol,
    protocol_lock: Mapping[str, Any],
    manifest: SplitManifest,
) -> str:
    declared_receipt_hash = str(receipt.get("receipt_hash") or "")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if declared_receipt_hash != stable_hash(unhashed):
        raise PermissionError("freeze receipt content hash mismatch")
    if receipt.get("frozen") is not True:
        raise PermissionError("sealed test requires a frozen archive receipt")
    if receipt.get("protocol_hash") != protocol.protocol_hash:
        raise PermissionError("freeze receipt protocol mismatch")
    if receipt.get("protocol_lock_hash") != protocol_lock.get("lock_hash"):
        raise PermissionError("freeze receipt protocol lock mismatch")
    if receipt.get("manifest_hash") != manifest.manifest_hash:
        raise PermissionError("freeze receipt manifest mismatch")
    evaluator_epoch = str(receipt.get("evaluator_epoch") or "")
    if not evaluator_epoch:
        raise PermissionError("freeze receipt evaluator epoch missing")
    return evaluator_epoch


def open_sealed_journal(
    path: str | Path,
    *,
    protocol: PaperProtocol,
    protocol_lock: Mapping[str, Any],
    manifest: SplitManifest,
    controls: Sequence[ControlSource],
    record_path: Path,
) -> dict[str, Any]:
    destination = Path(path)
    expected = {
        "journal_version": "sealed_access_journal_v1",
        "protocol_hash": protocol.protocol_hash,
        "protocol_lock_hash": protocol_lock.get("lock_hash"),
        "manifest_hash": manifest.manifest_hash,
        "control_config_hash": control_config_hash(controls),
        "record_path_hash": stable_hash({"path": str(record_path.resolve())}),
        "test_content_accessed": True,
        "secret_value_persisted": False,
    }
    if destination.exists():
        incumbent = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(incumbent, Mapping):
            raise PermissionError("sealed journal is malformed")
        _validate_journal_hash(incumbent)
        for key, value in expected.items():
            if incumbent.get(key) != value:
                raise PermissionError(f"sealed journal mismatch: {key}")
        history = incumbent.get("completion_history")
        if incumbent.get("status") == "complete":
            if not isinstance(history, list) or not history:
                raise PermissionError("completed sealed journal has no completion history")
            latest = history[-1]
            if not isinstance(latest, Mapping):
                raise PermissionError("sealed completion history is malformed")
            if latest.get("records_hash") != _record_file_hash(record_path):
                raise PermissionError("sealed records changed after the prior completion")
        resumed = {
            **dict(incumbent),
            "status": "in_progress",
            "access_invocation_count": int(incumbent.get("access_invocation_count") or 0) + 1,
            "previous_journal_hash": incumbent.get("journal_hash"),
        }
        resumed["journal_hash"] = _journal_hash(resumed)
        _replace_json(destination, resumed)
        return resumed
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **expected,
        "status": "in_progress",
        "access_invocation_count": 1,
        "completion_history": [],
        "previous_journal_hash": None,
    }
    payload["journal_hash"] = _journal_hash(payload)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def finalize_sealed_journal(
    path: str | Path,
    *,
    records: Sequence[PaperTrialRecord],
) -> dict[str, Any]:
    destination = Path(path)
    incumbent = json.loads(destination.read_text(encoding="utf-8"))
    if not isinstance(incumbent, Mapping):
        raise PermissionError("sealed journal is malformed")
    _validate_journal_hash(incumbent)
    if incumbent.get("status") != "in_progress":
        raise PermissionError("sealed journal is not open for completion")
    ordered = sorted(
        records,
        key=lambda row: (row.item_id_hash, row.control_id, row.repeat, row.attempt),
    )
    completion = {
        "invocation": int(incumbent.get("access_invocation_count") or 0),
        "record_count": len(ordered),
        "valid_record_count": sum(row.valid for row in ordered),
        "invalid_record_count": sum(not row.valid for row in ordered),
        "records_hash": stable_hash([row.to_dict() for row in ordered]),
    }
    history = incumbent.get("completion_history")
    if not isinstance(history, list):
        raise PermissionError("sealed completion history is malformed")
    finalized = {
        **dict(incumbent),
        "status": "complete",
        "completion_history": [*history, completion],
        "previous_journal_hash": incumbent.get("journal_hash"),
    }
    finalized["journal_hash"] = _journal_hash(finalized)
    _replace_json(destination, finalized)
    return finalized


def _journal_hash(payload: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key != "journal_hash"})


def _validate_journal_hash(payload: Mapping[str, Any]) -> None:
    if payload.get("journal_hash") != _journal_hash(payload):
        raise PermissionError("sealed journal content hash mismatch")


def _record_file_hash(path: Path) -> str:
    if not path.is_file():
        raise PermissionError("sealed paper record file is missing")
    records = sorted(
        read_records(path),
        key=lambda row: (row.item_id_hash, row.control_id, row.repeat, row.attempt),
    )
    return stable_hash([row.to_dict() for row in records])


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_key(record: PaperTrialRecord) -> tuple[str, str, int, str]:
    return (record.item_id_hash, record.control_id, record.repeat, record.split)


def source_tree_hash(root: Path) -> str:
    rows = [
        {
            "path": str(path.relative_to(root)),
            "content_hash": stable_hash({"bytes": path.read_bytes().hex()}),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return stable_hash(rows)


def control_config_hash(controls: Sequence[ControlSource]) -> str:
    return stable_hash(
        {
            "controls": [
                {
                    "id": row.id,
                    "source_hash": source_tree_hash(row.root) if row.root else None,
                }
                for row in sorted(controls, key=lambda value: value.id)
            ]
        }
    )


def controls_from_freeze_receipt(
    receipt: Mapping[str, Any],
    *,
    split: str,
) -> tuple[ControlSource, ...]:
    control_sets = receipt.get("control_sets")
    if not isinstance(control_sets, Mapping):
        raise PermissionError("freeze receipt control sets are missing")
    declared = control_sets.get(split)
    if not isinstance(declared, Mapping):
        raise PermissionError(f"freeze receipt has no {split} control set")
    rows = declared.get("controls")
    if not isinstance(rows, list):
        raise PermissionError("freeze receipt control rows are malformed")
    controls: list[ControlSource] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("id"):
            raise PermissionError("freeze receipt control row is malformed")
        raw_root = row.get("root")
        root = Path(str(raw_root)).resolve() if raw_root else None
        if root is not None and not root.is_dir():
            raise FileNotFoundError(f"frozen control source is missing: {row['id']}")
        actual_hash = source_tree_hash(root) if root else None
        if actual_hash != row.get("source_hash"):
            raise PermissionError(f"frozen control source changed: {row['id']}")
        controls.append(ControlSource(str(row["id"]), root))
    frozen_hash = str(declared.get("config_hash") or "")
    if frozen_hash != control_config_hash(controls):
        raise PermissionError("freeze receipt control configuration hash mismatch")
    return tuple(controls)


def _parse_control(value: str, project_root: Path) -> ControlSource:
    if "=" not in value:
        raise ValueError("controls must use ID=PATH or ID=none")
    control_id, raw_path = value.split("=", 1)
    control_id = control_id.strip()
    raw_path = raw_path.strip()
    if not control_id:
        raise ValueError("control ID cannot be empty")
    if raw_path.lower() == "none":
        return ControlSource(control_id, None)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    if not path.is_dir():
        raise FileNotFoundError(f"control source is not a directory: {control_id}")
    return ControlSource(control_id, path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable paper controls on a frozen split.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--sealed-journal", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parallel-workers", type=int)
    parser.add_argument(
        "--retry-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    load_dotenv(args.env_file)
    map_legacy_model_env()
    protocol = PaperProtocol.read(args.protocol)
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if not isinstance(lock, Mapping):
        raise ValueError("protocol lock must contain one JSON object")
    manifest = SplitManifest.read(args.manifest)
    split = SplitName(args.split)
    receipt: Mapping[str, Any] | None = None
    if args.freeze_receipt is not None:
        loaded_receipt = json.loads(args.freeze_receipt.read_text(encoding="utf-8"))
        if not isinstance(loaded_receipt, Mapping):
            raise ValueError("freeze receipt must contain one JSON object")
        receipt = loaded_receipt
    if receipt is not None:
        if args.control:
            raise ValueError("use either --freeze-receipt controls or explicit --control values")
        controls = controls_from_freeze_receipt(receipt, split=split.value)
    else:
        controls = tuple(_parse_control(value, project_root) for value in args.control)
    if not controls:
        raise ValueError("paper controls are required")
    guard = SplitAccessGuard(manifest)
    if split is SplitName.TEST:
        if args.limit is not None:
            raise PermissionError("claim-eligible sealed test cannot use --limit")
        if args.freeze_receipt is None or args.sealed_journal is None:
            raise PermissionError("sealed test requires --freeze-receipt and --sealed-journal")
        assert receipt is not None
        evaluator_epoch = validate_freeze_receipt(
            receipt,
            protocol=protocol,
            protocol_lock=lock,
            manifest=manifest,
        )
        open_sealed_journal(
            args.sealed_journal,
            protocol=protocol,
            protocol_lock=lock,
            manifest=manifest,
            controls=controls,
            record_path=args.records,
        )
        guard.freeze_archive()
    else:
        evaluator_epoch = f"skilllearn-eval-{manifest.manifest_hash[:12]}"
    item_ids = manifest.ids_for(split)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        item_ids = item_ids[: args.limit]
    secondary = manifest.manifest_hash == lock.get("secondary_manifest_hash")
    if split is SplitName.TEST:
        phase_name = "family_out_transfer" if secondary else "sealed_test"
    else:
        phase_name = "family_out_development" if secondary else "development"
    phase = protocol.payload["phases"][phase_name]
    repeats = int(phase["repeats"])
    parallel_workers = args.parallel_workers or int(phase.get("parallel_workers") or 1)
    if parallel_workers <= 0:
        raise ValueError("parallel worker count must be positive")
    sink = JsonlEventSink(args.events)
    prebuilt_cache = SkillLearnPrebuiltImageCache(
        args.benchmark_root,
        event_sink=sink,
    )
    backends = tuple(
        SkillLearnSubprocessBackend(
            args.benchmark_root,
            agent_id=str(protocol.payload["agent_id"]),
            model=str(protocol.payload["model"]),
            max_steps=int(protocol.payload["max_steps"]),
            provider_mode=str(protocol.payload["trial_provider_mode"]),
            trials_dir=args.trials_dir,
            prebuilt_cache=prebuilt_cache,
            event_sink=sink,
        )
        for _ in range(parallel_workers)
    )
    backend: SkillLearnTrialBackend = (
        backends[0] if len(backends) == 1 else SkillLearnBackendPool(backends)
    )
    record_store = PaperRecordStore(args.records)
    runner = PaperControlRunner(
        adapter=SkillLearnBenchAdapter(args.benchmark_root),
        manifest=manifest,
        guard=guard,
        backend=backend,
        protocol=protocol,
        controls=controls,
        record_store=record_store,
        evaluator_epoch=evaluator_epoch,
        event_sink=sink,
    )
    records = runner.run(
        item_ids,
        split=split,
        repeats=repeats,
        retry_invalid=args.retry_invalid,
        parallel_workers=parallel_workers,
        trace_id=f"paper-controls-{split.value}-{protocol.protocol_hash[:12]}",
    )
    if split is SplitName.TEST:
        assert args.sealed_journal is not None
        finalize_sealed_journal(
            args.sealed_journal,
            records=tuple(record_store.records.values()),
        )
    print(
        json.dumps(
            {
                "record_count": len(records),
                "valid_count": sum(row.valid for row in records),
                "split": split.value,
                "parallel_workers": parallel_workers,
                "test_content_accessed": guard.test_accessed,
                "secret_value_persisted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
