from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .events import Event, EventSink, NullEventSink
from .models import SplitName, stable_hash


class AccessPhase(str, Enum):
    PROPOSAL = "proposal"
    REPAIR = "repair"
    PROMOTION = "promotion"
    FINAL_REPORT = "final_report"


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    family: str
    features: Mapping[str, Any]
    content_ref: str
    verifier_ref_hash: str

    @property
    def id_hash(self) -> str:
        return stable_hash({"item_id": self.id})


@dataclass(frozen=True)
class SplitManifest:
    benchmark: str
    protocol: str
    seed: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    family_by_id: Mapping[str, str]
    sealed_test: bool = True

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self.to_dict(include_hash=False))

    def split_for(self, item_id: str) -> SplitName:
        if item_id in self.train_ids:
            return SplitName.TRAIN
        if item_id in self.validation_ids:
            return SplitName.VALIDATION
        if item_id in self.test_ids:
            return SplitName.TEST
        raise KeyError(f"item is not in split manifest: {item_id}")

    def validate(self) -> list[str]:
        issues: list[str] = []
        train = set(self.train_ids)
        validation = set(self.validation_ids)
        test = set(self.test_ids)
        if train & validation or train & test or validation & test:
            issues.append("split_overlap")
        if not train:
            issues.append("train_split_empty")
        if not validation:
            issues.append("validation_split_empty")
        if not test:
            issues.append("test_split_empty")
        all_ids = train | validation | test
        if not all(item_id in self.family_by_id for item_id in all_ids):
            issues.append("family_mapping_incomplete")
        if self.protocol == "family_out":
            train_families = {self.family_by_id[item_id] for item_id in train}
            validation_families = {self.family_by_id[item_id] for item_id in validation}
            test_families = {self.family_by_id[item_id] for item_id in test}
            if train_families & validation_families or train_families & test_families or validation_families & test_families:
                issues.append("family_out_leakage")
        return issues

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["train_ids"] = list(self.train_ids)
        payload["validation_ids"] = list(self.validation_ids)
        payload["test_ids"] = list(self.test_ids)
        payload["family_by_id"] = dict(sorted(self.family_by_id.items()))
        payload["item_counts"] = {
            "train": len(self.train_ids),
            "validation": len(self.validation_ids),
            "test": len(self.test_ids),
        }
        payload["family_counts"] = {
            split.value: len({self.family_by_id[item_id] for item_id in self.ids_for(split)})
            for split in SplitName
        }
        payload["validation_issues"] = self.validate()
        payload["raw_content_persisted"] = False
        if include_hash:
            payload["manifest_hash"] = self.manifest_hash
        return payload

    def ids_for(self, split: SplitName) -> tuple[str, ...]:
        if split is SplitName.TRAIN:
            return self.train_ids
        if split is SplitName.VALIDATION:
            return self.validation_ids
        return self.test_ids

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitManifest":
        manifest = cls(
            benchmark=str(data.get("benchmark") or ""),
            protocol=str(data.get("protocol") or ""),
            seed=str(data.get("seed") or ""),
            train_ids=tuple(str(value) for value in data.get("train_ids", [])),
            validation_ids=tuple(str(value) for value in data.get("validation_ids", [])),
            test_ids=tuple(str(value) for value in data.get("test_ids", [])),
            family_by_id={
                str(key): str(value)
                for key, value in dict(data.get("family_by_id") or {}).items()
            },
            sealed_test=bool(data.get("sealed_test", True)),
        )
        issues = manifest.validate()
        if issues:
            raise ValueError(f"invalid split manifest: {issues}")
        declared_hash = str(data.get("manifest_hash") or "")
        if declared_hash and declared_hash != manifest.manifest_hash:
            raise ValueError("split manifest hash does not match its content")
        return manifest

    @classmethod
    def read(cls, path: str | Path) -> "SplitManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("split manifest must contain one JSON object")
        return cls.from_dict(payload)


def build_instance_holdout_manifest(
    items: Iterable[BenchmarkItem],
    *,
    benchmark: str,
    seed: str,
) -> SplitManifest:
    grouped: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for item in items:
        grouped[item.family].append(item)
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    family_by_id: dict[str, str] = {}
    for family, family_items in sorted(grouped.items()):
        ordered = sorted(family_items, key=lambda item: stable_hash({"seed": seed, "item_id": item.id}))
        for item in ordered:
            family_by_id[item.id] = family
        count = len(ordered)
        if count == 1:
            train.append(ordered[0].id)
            continue
        if count == 2:
            train.append(ordered[0].id)
            test.append(ordered[1].id)
            continue
        train_count = max(1, count // 2)
        validation_count = max(1, (count - train_count) // 2)
        if train_count + validation_count >= count:
            validation_count = 1
            train_count = count - 2
        train.extend(item.id for item in ordered[:train_count])
        validation.extend(item.id for item in ordered[train_count : train_count + validation_count])
        test.extend(item.id for item in ordered[train_count + validation_count :])
    return _validated_manifest(
        benchmark=benchmark,
        protocol="instance_holdout",
        seed=seed,
        train=train,
        validation=validation,
        test=test,
        family_by_id=family_by_id,
    )


def build_family_out_manifest(
    items: Iterable[BenchmarkItem],
    *,
    benchmark: str,
    seed: str,
) -> SplitManifest:
    rows = list(items)
    family_by_id = {item.id: item.family for item in rows}
    families = sorted({item.family for item in rows}, key=lambda family: stable_hash({"seed": seed, "family": family}))
    if len(families) < 3:
        raise ValueError("family-out protocol requires at least three families")
    train_family_count = max(1, int(len(families) * 0.6))
    validation_family_count = max(1, int(len(families) * 0.2))
    if train_family_count + validation_family_count >= len(families):
        train_family_count = len(families) - 2
        validation_family_count = 1
    train_families = set(families[:train_family_count])
    validation_families = set(families[train_family_count : train_family_count + validation_family_count])
    test_families = set(families[train_family_count + validation_family_count :])
    return _validated_manifest(
        benchmark=benchmark,
        protocol="family_out",
        seed=seed,
        train=[item.id for item in rows if item.family in train_families],
        validation=[item.id for item in rows if item.family in validation_families],
        test=[item.id for item in rows if item.family in test_families],
        family_by_id=family_by_id,
    )


class SplitAccessGuard:
    def __init__(self, manifest: SplitManifest, *, event_sink: EventSink | None = None) -> None:
        if manifest.validate():
            raise ValueError(f"cannot guard an invalid split manifest: {manifest.validate()}")
        self.manifest = manifest
        self.event_sink = event_sink or NullEventSink()
        self.archive_frozen = False
        self.test_accessed = False

    def freeze_archive(self) -> None:
        self.archive_frozen = True
        self.event_sink.emit(
            Event(
                event="archive_frozen_for_test",
                stage="split_guard",
                trace_id=self.manifest.manifest_hash[:20],
                payload={"manifest_hash": self.manifest.manifest_hash},
            )
        )

    def authorize(self, item_id: str, phase: AccessPhase) -> SplitName:
        split = self.manifest.split_for(item_id)
        allowed = {
            AccessPhase.PROPOSAL: {SplitName.TRAIN},
            AccessPhase.REPAIR: {SplitName.TRAIN},
            AccessPhase.PROMOTION: {SplitName.VALIDATION},
            AccessPhase.FINAL_REPORT: {SplitName.TEST},
        }[phase]
        if split not in allowed:
            raise PermissionError(f"{phase.value} cannot access {split.value} item {item_id}")
        if phase is AccessPhase.FINAL_REPORT and not self.archive_frozen:
            raise PermissionError("archive must be frozen before sealed test access")
        if split is SplitName.TEST:
            self.test_accessed = True
        self.event_sink.emit(
            Event(
                event="split_access_authorized",
                stage="split_guard",
                trace_id=self.manifest.manifest_hash[:20],
                payload={
                    "item_id_hash": stable_hash({"item_id": item_id}),
                    "phase": phase.value,
                    "split": split.value,
                    "archive_frozen": self.archive_frozen,
                },
            )
        )
        return split


def _validated_manifest(
    *,
    benchmark: str,
    protocol: str,
    seed: str,
    train: list[str],
    validation: list[str],
    test: list[str],
    family_by_id: Mapping[str, str],
) -> SplitManifest:
    manifest = SplitManifest(
        benchmark=benchmark,
        protocol=protocol,
        seed=seed,
        train_ids=tuple(sorted(train)),
        validation_ids=tuple(sorted(validation)),
        test_ids=tuple(sorted(test)),
        family_by_id=dict(family_by_id),
    )
    issues = manifest.validate()
    if issues:
        raise ValueError(f"invalid split manifest: {issues}")
    return manifest
