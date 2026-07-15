from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import zipfile

import pytest

from replication_runtime.financial_sec13f_contract_v2 import (
    hygienic_materialize,
)
from replication_runtime.financial_sec13f_contract_v2 import hygienic_prewarm
from replication_runtime.financial_semantic_v2 import oracle_pandas
from replication_runtime.financial_semantic_v2 import oracle_streaming
from replication_runtime.financial_semantic_v2 import pack as period_pack


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
        cover_rows.append(
            [
                accession,
                report_date,
                "13F HOLDINGS REPORT",
                f"Period Fund {manager_index:02d} LLC",
            ]
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
    previous_zip: Path
    current_zip: Path
    upstream: Path
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
        preregistration_seed="hygienic-materialization-v2-test",
        previous_container_root=hygienic_materialize.PREVIOUS_ALIAS,
        current_container_root=hygienic_materialize.CURRENT_ALIAS,
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
        previous_zip=previous_zip,
        current_zip=current_zip,
        upstream=upstream,
        measurement_view=view,
        measurement_gold=gold,
    )


def _materialize(inputs: _Inputs, output: Path) -> dict[str, Any]:
    return hygienic_materialize.materialize_measurement_benchmark_v2(
        upstream_benchmark_root=inputs.upstream,
        measurement_view=inputs.measurement_view,
        measurement_gold=inputs.measurement_gold,
        previous_archive=inputs.previous_zip,
        current_archive=inputs.current_zip,
        output_root=output,
    )


def test_materialization_is_deterministic_versioned_and_gold_isolated(
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
    assert first["materialization_version"] == (
        hygienic_materialize.MATERIALIZATION_VERSION
    )
    assert first["tree_receipt_version"] == (
        hygienic_materialize.TREE_RECEIPT_VERSION
    )
    assert first["private_pack_accessed"] is False
    assert first["sealed_content_accessed_by_materializer"] is False
    assert first["sealed_task_count_materialized"] == 0
    isolation = first["verifier_isolation"]
    assert isolation["candidate_visible_gold_path_count"] == 0
    assert isolation["gold_copied_into_environment_image"] is False
    assert isolation["gold_container_path"] == "/tests/expected_output.json"
    evidence = first["verifier_evidence"]
    assert evidence["fixed_failure_code"] == (
        hygienic_materialize.FIXED_FAILURE_CODE
    )
    assert evidence["pytest_traceback_mode"] == "none"

    receipt = hygienic_materialize.measurement_benchmark_tree_receipt_v2(
        first_root
    )
    assert receipt["tree_receipt_version"] == (
        hygienic_materialize.TREE_RECEIPT_VERSION
    )
    assert receipt["tree_hash"] == first["benchmark_tree_hash"]

    item_ids = [
        item["item_id"]
        for item in inputs.measurement_view["measurement_items"]
    ]
    task_root = first_root / "tasks" / hygienic_materialize.FAMILY
    assert sorted(path.name for path in task_root.iterdir()) == sorted(item_ids)
    for item_id in item_ids:
        task = task_root / item_id
        environment = task / "environment"
        tests = task / "tests"
        assert sorted(path.name for path in tests.iterdir()) == [
            "expected_output.json",
            "test.sh",
            "test_outputs.py",
        ]
        assert not any(
            path.name == "expected_output.json"
            for path in environment.rglob("*")
        )
        dockerfile = (environment / "Dockerfile").read_text(
            encoding="utf-8"
        )
        assert "/tests" not in dockerfile
        assert "expected_output" not in dockerfile
        script = (tests / "test.sh").read_text(encoding="utf-8")
        assert "--tb=no" in script
        assert "/tests/expected_output.json" not in script
        source = (tests / "test_outputs.py").read_text(encoding="utf-8")
        assert "pytest.fail(FAILURE_CODE, pytrace=False)" in source
        assert "assert " not in source
        assert source.count("/tests/expected_output.json") == 1


def test_materializer_never_calls_private_or_sealed_pack_validators(
    inputs: _Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("private or sealed pack path was accessed")

    for name in (
        "verify_public_pack",
        "build_measurement_view",
        "verify_consensus_gold",
        "validate_source_against_pack",
        "partition_items",
    ):
        monkeypatch.setattr(period_pack, name, forbidden)

    report = _materialize(inputs, tmp_path / "benchmark")

    assert report["private_pack_accessed"] is False
    assert report["sealed_content_accessed_by_materializer"] is False


def test_fixed_failure_output_contains_neither_expected_nor_observed_values(
    tmp_path: Path,
) -> None:
    tests = tmp_path / "tests"
    root = tmp_path / "root"
    tests.mkdir()
    root.mkdir()
    expected_marker = 1_234_567_891
    observed_marker = 9_876_543_219
    (tests / "expected_output.json").write_text(
        json.dumps(
            {
                "q1_answer": expected_marker,
                "q2_answer": 3,
                "q3_answer": ["expected-secret-marker"],
                "q4_answer": ["another-expected-marker"],
            }
        ),
        encoding="utf-8",
    )
    (root / "answers.json").write_text(
        json.dumps(
            {
                "q1_answer": observed_marker,
                "q2_answer": 4,
                "q3_answer": ["observed-secret-marker"],
                "q4_answer": ["another-observed-marker"],
            }
        ),
        encoding="utf-8",
    )
    source = hygienic_materialize._test_source().replace(
        'Path("/tests/expected_output.json")',
        f"Path({str(tests / 'expected_output.json')!r})",
    ).replace(
        'Path("/root/answers.json")',
        f"Path({str(root / 'answers.json')!r})",
    )
    test_file = tests / "test_outputs.py"
    test_file.write_text(source, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--tb=no",
            "-q",
            "-s",
            str(test_file),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    evidence = completed.stdout + completed.stderr

    assert completed.returncode == 1
    for secret in (
        str(expected_marker),
        str(observed_marker),
        "expected-secret-marker",
        "observed-secret-marker",
        "another-expected-marker",
        "another-observed-marker",
    ):
        assert secret not in evidence


def test_tree_receipt_version_and_exclusion_are_hash_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    (root / hygienic_materialize.MATERIALIZATION_REPORT_NAME).write_text(
        "report\n",
        encoding="utf-8",
    )

    receipt = hygienic_materialize.measurement_benchmark_tree_receipt_v2(root)
    body = dict(receipt)
    tree_hash = body.pop("tree_hash")

    assert tree_hash == period_pack.payload_hash(body)
    assert body["tree_receipt_version"] == (
        hygienic_materialize.TREE_RECEIPT_VERSION
    )
    assert body["excluded_relative_paths"] == [
        hygienic_materialize.MATERIALIZATION_REPORT_NAME
    ]
    assert all(
        row["path"] != hygienic_materialize.MATERIALIZATION_REPORT_NAME
        for row in body["rows"]
    )


def test_hygienic_prewarm_is_cache_only_and_preserves_v2_tree(
    inputs: _Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = tmp_path / "benchmark"
    materialization = _materialize(inputs, benchmark)
    before = hygienic_materialize.measurement_benchmark_tree_receipt_v2(
        benchmark
    )
    state: dict[str, Any] = {
        "cache_modes": [],
        "image_calls": [],
        "offline_calls": [],
    }

    class FakeImageCache:
        def __init__(
            self,
            _benchmark: Path,
            *,
            cache_only: bool,
            **_: Any,
        ):
            self.cache_only = cache_only
            state["cache_modes"].append(cache_only)

        def ensure(self, *, item_id: str, **_: Any) -> SimpleNamespace:
            state["image_calls"].append((self.cache_only, item_id))
            return SimpleNamespace(
                tag="assumption-sec13f-v2:fixture",
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

        def ensure(
            self,
            *,
            profile: Any,
            base_image_id: str,
            **_: Any,
        ) -> SimpleNamespace:
            state["offline_calls"].append(base_image_id)
            return SimpleNamespace(
                runtime_key=hygienic_prewarm.offline_verifier_runtime_key(
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
        body = {
            "report_version": "offline_verifier_preparation_receipt_v2",
            "policy": hygienic_prewarm.OFFLINE_VERIFIER_POLICY_VERSION,
            "profile_id": profile.profile_id,
            "profile_hash": profile.profile_hash,
            "runtime_key": hygienic_prewarm.offline_verifier_runtime_key(
                profile=profile
            ),
            "base_image_tag": base_image_tag,
            "base_image_id": "sha256:" + "d" * 64,
            "python_version": profile.python_version,
            "python_abi": profile.python_abi,
            "docker_install_network": "none",
            "probe_passed": True,
            "raw_content_persisted": False,
        }
        report = {**body, "receipt_hash": period_pack.payload_hash(body)}
        period_pack.write_json(report_path, report)
        return report

    monkeypatch.setattr(
        hygienic_prewarm,
        "SkillLearnPrebuiltImageCache",
        FakeImageCache,
    )
    monkeypatch.setattr(
        hygienic_prewarm,
        "SkillLearnSubprocessBackend",
        FakeBackend,
    )
    monkeypatch.setattr(
        hygienic_prewarm,
        "SkillLearnOfflineVerifierRuntimeCache",
        FakeOfflineCache,
    )
    monkeypatch.setattr(
        hygienic_prewarm,
        "prepare_offline_verifier_runtime",
        fake_prepare,
    )

    report = hygienic_prewarm.prepare_measurement_runtime_v2(
        benchmark_root=benchmark,
        measurement_view=inputs.measurement_view,
        output_root=tmp_path / "prewarm",
    )
    after = hygienic_materialize.measurement_benchmark_tree_receipt_v2(
        benchmark
    )

    assert state["cache_modes"] == [False, True]
    assert len(state["image_calls"]) == 16
    assert len(state["offline_calls"]) == 8
    assert report["tree_receipt_version"] == (
        hygienic_materialize.TREE_RECEIPT_VERSION
    )
    assert report["benchmark_tree_unchanged"] is True
    assert report["pre_prewarm_tree_hash"] == before["tree_hash"]
    assert report["post_prewarm_tree_hash"] == after["tree_hash"]
    assert after == before
    assert after["tree_hash"] == materialization["benchmark_tree_hash"]
    assert report["python_dont_write_bytecode"] is True
    assert report["python_dont_write_bytecode_env"] == "1"
    assert not any(path.name == "__pycache__" for path in benchmark.rglob("*"))
