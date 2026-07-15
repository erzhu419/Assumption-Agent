from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import zipfile

import pytest

from replication_runtime.financial_semantic_v2 import materialize
from replication_runtime.financial_semantic_v2 import oracle_pandas
from replication_runtime.financial_semantic_v2 import oracle_streaming
from replication_runtime.financial_semantic_v2 import pack as period_pack
from replication_runtime.financial_semantic_v2 import prewarm


MANAGER_COUNT = 32
ISSUER_COUNT = 20


def _write_tsv(
    path: Path,
    header: list[str],
    rows: list[list[object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_period(root: Path, *, current: bool) -> None:
    report_date = "31-MAR-2026" if current else "31-DEC-2025"
    prefix = "C" if current else "P"
    cover_rows: list[list[object]] = []
    info_rows: list[list[object]] = []
    for manager_index in range(MANAGER_COUNT):
        accession = f"{prefix}{manager_index:05d}"
        manager = f"Period Fund {manager_index:02d} LLC"
        cover_rows.append(
            [accession, report_date, "13F HOLDINGS REPORT", manager]
        )
        for issuer_index in range(ISSUER_COUNT):
            previous_value = 10_000 + manager_index * 100 + issuer_index
            value = (
                previous_value
                + (issuer_index + 1) * 1_000
                + manager_index
                if current
                else previous_value
            )
            info_rows.append(
                [
                    accession,
                    f"Issuer Corporation {issuer_index:02d}",
                    "COM",
                    f"{100_000_000 + issuer_index:09d}",
                    value,
                ]
            )
        info_rows.append(
            [
                accession,
                f"Private Note {manager_index:02d}",
                "PUT",
                f"{900_000_000 + manager_index:09d}",
                777 + manager_index,
            ]
        )
    if current:
        cover_rows.reverse()
        info_rows.reverse()
    _write_tsv(
        root / "COVERPAGE.tsv",
        [
            "ACCESSION_NUMBER",
            "REPORTCALENDARORQUARTER",
            "REPORTTYPE",
            "FILINGMANAGER_NAME",
        ],
        cover_rows,
    )
    _write_tsv(
        root / "INFOTABLE.tsv",
        [
            "ACCESSION_NUMBER",
            "NAMEOFISSUER",
            "TITLEOFCLASS",
            "CUSIP",
            "VALUE",
        ],
        info_rows,
    )


def _zip_period(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(source.iterdir()):
            archive.write(path, arcname=f"official-sec-period/{path.name}")


@dataclass(frozen=True)
class _Inputs:
    previous: Path
    current: Path
    previous_zip: Path
    current_zip: Path
    upstream: Path
    private_pack: dict[str, Any]
    measurement_view: dict[str, Any]
    measurement_gold: dict[str, Any]


@pytest.fixture()
def inputs(tmp_path: Path) -> _Inputs:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    _write_period(previous, current=False)
    _write_period(current, current=True)
    previous_zip = tmp_path / "previous.zip"
    current_zip = tmp_path / "current.zip"
    _zip_period(previous, previous_zip)
    _zip_period(current, current_zip)
    private = period_pack.build_public_pack(
        previous_source=previous_zip,
        current_source=current_zip,
        previous_period_label="2025 Q4",
        current_period_label="2026 Q1",
        preregistration_seed="materialization-prewarm-test",
        previous_container_root=materialize.PREVIOUS_ALIAS,
        current_container_root=materialize.CURRENT_ALIAS,
    )
    view = period_pack.build_measurement_view(private)
    left = oracle_pandas.evaluate_partition(
        pack=private,
        previous_source=previous,
        current_source=current,
        partition="measurement",
    )
    right = oracle_streaming.evaluate_partition(
        pack=private,
        previous_source=previous,
        current_source=current,
        partition="measurement",
    )
    gold = period_pack.build_consensus_gold(
        pack=private,
        left=left,
        right=right,
        partition="measurement",
    )
    upstream = tmp_path / "upstream"
    (upstream / "core").mkdir(parents=True)
    (upstream / "agents").mkdir()
    (upstream / "core" / "eval_runner.py").write_text(
        "# frozen local runner fixture\n",
        encoding="utf-8",
    )
    (upstream / "agents" / "__init__.py").write_text(
        "# frozen local agents fixture\n",
        encoding="utf-8",
    )
    return _Inputs(
        previous=previous,
        current=current,
        previous_zip=previous_zip,
        current_zip=current_zip,
        upstream=upstream,
        private_pack=private,
        measurement_view=view,
        measurement_gold=gold,
    )


def _materialize(inputs: _Inputs, output: Path) -> dict[str, Any]:
    return materialize.materialize_measurement_benchmark_v1(
        upstream_benchmark_root=inputs.upstream,
        private_pack=inputs.private_pack,
        measurement_view=inputs.measurement_view,
        measurement_gold=inputs.measurement_gold,
        previous_archive=inputs.previous_zip,
        current_archive=inputs.current_zip,
        output_root=output,
    )


def test_materialization_is_measurement_only_deterministic_and_auditable(
    inputs: _Inputs,
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "benchmark-first"
    second_root = tmp_path / "benchmark-second"
    first = _materialize(inputs, first_root)
    second = _materialize(inputs, second_root)

    assert first == second
    body = dict(first)
    declared = body.pop("materialization_hash")
    assert declared == period_pack.payload_hash(body)
    assert first["item_count"] == 8
    assert first["sealed_task_count_materialized"] == 0
    assert first["sealed_content_persisted"] is False
    assert first["sealed_gold_accessed"] is False
    assert first["benchmark_tree_hash"] == (
        materialize.measurement_benchmark_tree_receipt_v1(first_root)[
            "tree_hash"
        ]
    )
    item_ids = [
        item["item_id"]
        for item in inputs.measurement_view["measurement_items"]
    ]
    task_root = first_root / "tasks" / materialize.FAMILY
    assert sorted(path.name for path in task_root.iterdir()) == sorted(item_ids)
    assert all("sealed" not in path.as_posix() for path in first_root.rglob("*"))

    dockerfile = (
        task_root / item_ids[0] / "environment" / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert f"mkdir -p /tmp/period-previous {materialize.PREVIOUS_ALIAS}" in dockerfile
    assert f"mkdir -p /tmp/period-current {materialize.CURRENT_ALIAS}" in dockerfile
    assert dockerfile.count("INFOTABLE.tsv") == 4
    for role, alias in (
        ("previous", materialize.PREVIOUS_ALIAS),
        ("current", materialize.CURRENT_ALIAS),
    ):
        receipt = first["period_source_receipts"][role]
        assert receipt["container_alias"] == alias
        assert receipt["source_fingerprint"] == inputs.private_pack[
            "sources"
        ][role]["source_fingerprint"]
        assert receipt["source_path_persisted"] is False

    materialized_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in first_root.rglob("*")
        if path.is_file() and path.suffix != ".zip"
    )
    for sealed in period_pack.partition_items(inputs.private_pack, "sealed"):
        assert sealed["item_id"] not in materialized_text
        assert sealed["instruction"] not in materialized_text


def test_materialization_rejects_archive_role_swap_before_writing(
    inputs: _Inputs,
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"

    with pytest.raises(
        period_pack.PeriodOutPackError,
        match="previous SEC source differs",
    ):
        materialize.materialize_measurement_benchmark_v1(
            upstream_benchmark_root=inputs.upstream,
            private_pack=inputs.private_pack,
            measurement_view=inputs.measurement_view,
            measurement_gold=inputs.measurement_gold,
            previous_archive=inputs.current_zip,
            current_archive=inputs.previous_zip,
            output_root=output,
        )

    assert not output.exists()


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "cache_only": [],
        "image_calls": [],
        "offline_calls": [],
        "preparation_calls": [],
    }

    class FakeImageCache:
        def __init__(self, _benchmark: Path, *, cache_only: bool, **_: Any):
            self.cache_only = cache_only
            state["cache_only"].append(cache_only)

        def ensure(self, *, item_id: str, **_: Any) -> SimpleNamespace:
            state["image_calls"].append((self.cache_only, item_id))
            return SimpleNamespace(
                tag="assumption-v2-item:fixture",
                cache_key="a" * 64,
                environment_hash="b" * 64,
                source_environment_hash="c" * 64,
                image_id="sha256:" + "d" * 64,
                agent_runtime_key="e" * 64,
                agent_runtime_version="codex-runtime-fixture-v1",
                reused=self.cache_only,
            )

    class FakeBackend:
        def __init__(self, *_: Any, **__: Any):
            self.runner = SimpleNamespace(subprocess=object())

        def _load_runner(self) -> SimpleNamespace:
            return self.runner

    class FakeOfflineCache:
        def __init__(self, **_: Any):
            pass

        def ensure(self, *, profile: Any, base_image_id: str, **_: Any) -> SimpleNamespace:
            assert state["preparation_calls"]
            state["offline_calls"].append(base_image_id)
            return SimpleNamespace(
                runtime_key=prewarm.offline_verifier_runtime_key(
                    profile=profile
                ),
                reused=True,
            )

    def fake_prepare(
        *,
        profile: Any,
        base_image_tag: str,
        report_path: Path,
        **_: Any,
    ) -> dict[str, Any]:
        state["preparation_calls"].append(
            (profile.profile_id, tuple(profile.requirements), base_image_tag)
        )
        body = {
            "report_version": "offline_verifier_preparation_receipt_v2",
            "policy": prewarm.OFFLINE_VERIFIER_POLICY_VERSION,
            "profile_id": profile.profile_id,
            "profile_hash": profile.profile_hash,
            "runtime_key": prewarm.offline_verifier_runtime_key(
                profile=profile
            ),
            "base_image_tag": base_image_tag,
            "base_image_id": "sha256:" + "d" * 64,
            "docker_install_network": "none",
            "probe_passed": True,
            "raw_content_persisted": False,
        }
        report = {**body, "receipt_hash": period_pack.payload_hash(body)}
        period_pack.write_json(report_path, report)
        return report

    monkeypatch.setattr(prewarm, "SkillLearnPrebuiltImageCache", FakeImageCache)
    monkeypatch.setattr(prewarm, "SkillLearnSubprocessBackend", FakeBackend)
    monkeypatch.setattr(
        prewarm,
        "SkillLearnOfflineVerifierRuntimeCache",
        FakeOfflineCache,
    )
    monkeypatch.setattr(
        prewarm,
        "prepare_offline_verifier_runtime",
        fake_prepare,
    )
    return state


def test_prewarm_prepares_dependencies_then_rechecks_cache_only_idempotently(
    inputs: _Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = tmp_path / "benchmark"
    _materialize(inputs, benchmark)
    state = _install_fake_runtime(monkeypatch)

    first = prewarm.prepare_measurement_runtime_v1(
        benchmark_root=benchmark,
        measurement_view=inputs.measurement_view,
        output_root=tmp_path / "prewarm-first",
    )
    second = prewarm.prepare_measurement_runtime_v1(
        benchmark_root=benchmark,
        measurement_view=inputs.measurement_view,
        output_root=tmp_path / "prewarm-second",
    )

    assert first == second
    body = dict(first)
    declared = body.pop("prewarm_hash")
    assert declared == period_pack.payload_hash(body)
    assert state["cache_only"] == [False, True, False, True]
    assert len(state["preparation_calls"]) == 2
    assert all(
        call[:2]
        == (
            prewarm.OFFLINE_VERIFIER_PROFILE_ID,
            prewarm.OFFLINE_VERIFIER_REQUIREMENTS,
        )
        for call in state["preparation_calls"]
    )
    assert len(state["image_calls"]) == 32
    assert sum(cache_only for cache_only, _ in state["image_calls"]) == 16
    assert len(state["offline_calls"]) == 16
    assert first["item_count"] == 8
    assert first["formal_execution_cache_only"] is True
    assert first["formal_image_cache_only"] is True
    assert first["formal_offline_verifier_cache_only"] is True
    assert first["formal_verifier_network"] == "none"
    assert first["offline_verifier_requirements"] == list(
        prewarm.OFFLINE_VERIFIER_REQUIREMENTS
    )
    assert all(
        row["prebuilt_cache_reused"]
        and row["offline_verifier_runtime_reused"]
        for row in first["formal_cache_rows"]
    )
    assert first["model_calls"] == 0
    assert first["online_judge_calls"] == 0
    assert first["sealed_task_count"] == 0
    assert first["sealed_content_accessed"] is False


def test_prewarm_rejects_added_sealed_task_before_runtime_construction(
    inputs: _Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = tmp_path / "benchmark"
    _materialize(inputs, benchmark)
    sealed = (
        benchmark
        / "tasks"
        / materialize.FAMILY
        / "financial-period-out-sealed-0"
    )
    sealed.mkdir()
    (sealed / "secret.txt").write_text("sealed", encoding="utf-8")

    class MustNotConstruct:
        def __init__(self, *_: Any, **__: Any):
            raise AssertionError("runtime construction must not start")

    monkeypatch.setattr(
        prewarm,
        "SkillLearnPrebuiltImageCache",
        MustNotConstruct,
    )
    output = tmp_path / "prewarm-must-not-exist"
    with pytest.raises(
        prewarm.PeriodOutPrewarmError,
        match="benchmark tree drifted",
    ):
        prewarm.prepare_measurement_runtime_v1(
            benchmark_root=benchmark,
            measurement_view=inputs.measurement_view,
            output_root=output,
        )

    assert not output.exists()
