from __future__ import annotations

"""Materialize only the SEC-13F measurement partition as a local benchmark.

The full deterministic pack and sealed gold may exist in a private artifact
directory, but this module writes only the eight measurement tasks.  The
result is SkillLearn runner-compatible without claiming to be an official
SkillLearnBench score.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Mapping, Sequence

from .pack import (
    build_measurement_view,
    payload_hash,
    read_json,
    sha256_file,
    verify_consensus_gold,
    verify_measurement_view,
    verify_public_pack,
    validate_source_against_pack,
    write_json,
)


MATERIALIZATION_VERSION = (
    "financial_semantic_sec13f_measurement_materialization_v1"
)
MATERIALIZATION_REPORT_NAME = "measurement.materialization.json"
FAMILY = "financial-analysis"
PREVIOUS_ALIAS = "/root/2025-q2"
CURRENT_ALIAS = "/root/2025-q3"


class PeriodOutMaterializationError(RuntimeError):
    """Measurement task materialization failed closed."""


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_text(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755 if executable else 0o644)


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _tree_receipt(
    root: Path,
    *,
    excluded_relative_paths: frozenset[str] = frozenset(),
    file_hash_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    hashes = file_hash_cache if file_hash_cache is not None else {}
    rows: list[dict[str, Any]] = []
    for path in sorted(
        resolved.rglob("*"),
        key=lambda value: value.relative_to(resolved).as_posix(),
    ):
        if path.is_symlink():
            raise PeriodOutMaterializationError(
                "materialized benchmark contains a symbolic link"
            )
        relative = path.relative_to(resolved).as_posix()
        if relative in excluded_relative_paths:
            continue
        stat = path.stat()
        if path.is_dir():
            rows.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "mode": stat.st_mode & 0o777,
                }
            )
        elif path.is_file():
            identity = (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            )
            digest = hashes.get(identity)
            if digest is None:
                digest = sha256_file(path)
                hashes[identity] = digest
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": stat.st_mode & 0o777,
                    "size_bytes": stat.st_size,
                    "sha256": digest,
                }
            )
        else:
            raise PeriodOutMaterializationError(
                "materialized benchmark contains a special file"
            )
    return {
        "file_and_directory_count": len(rows),
        "rows": rows,
        "tree_hash": payload_hash(rows),
    }


def measurement_benchmark_tree_receipt_v1(
    benchmark_root: str | Path,
    *,
    file_hash_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> dict[str, Any]:
    """Hash the immutable runner/task tree, excluding its self-report."""

    return _tree_receipt(
        Path(benchmark_root),
        excluded_relative_paths=frozenset({MATERIALIZATION_REPORT_NAME}),
        file_hash_cache=file_hash_cache,
    )


def _require_shared_table_root(
    *,
    coverpage_ref: str,
    infotable_ref: str,
    role: str,
) -> None:
    if PurePosixPath(coverpage_ref).parent != PurePosixPath(
        infotable_ref
    ).parent:
        raise PeriodOutMaterializationError(
            f"{role} archive tables do not share one compatibility root"
        )


def _dockerfile() -> str:
    return f"""FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-venv \\
    wget \\
    unzip \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root

COPY period-previous.zip /tmp/period-previous.zip
COPY period-current.zip /tmp/period-current.zip
RUN set -eu; \\
    mkdir -p /tmp/period-previous {PREVIOUS_ALIAS}; \\
    unzip -q /tmp/period-previous.zip -d /tmp/period-previous; \\
    cover="$(find /tmp/period-previous -type f -iname COVERPAGE.tsv)"; \\
    info="$(find /tmp/period-previous -type f -iname INFOTABLE.tsv)"; \\
    test "$(find /tmp/period-previous -type f -iname COVERPAGE.tsv | wc -l)" -eq 1; \\
    test "$(find /tmp/period-previous -type f -iname INFOTABLE.tsv | wc -l)" -eq 1; \\
    test "$(dirname "$cover")" = "$(dirname "$info")"; \\
    source_root="$(dirname "$cover")"; \\
    cp -a "$source_root"/. {PREVIOUS_ALIAS}/; \\
    rm -rf /tmp/period-previous /tmp/period-previous.zip
RUN set -eu; \\
    mkdir -p /tmp/period-current {CURRENT_ALIAS}; \\
    unzip -q /tmp/period-current.zip -d /tmp/period-current; \\
    cover="$(find /tmp/period-current -type f -iname COVERPAGE.tsv)"; \\
    info="$(find /tmp/period-current -type f -iname INFOTABLE.tsv)"; \\
    test "$(find /tmp/period-current -type f -iname COVERPAGE.tsv | wc -l)" -eq 1; \\
    test "$(find /tmp/period-current -type f -iname INFOTABLE.tsv | wc -l)" -eq 1; \\
    test "$(dirname "$cover")" = "$(dirname "$info")"; \\
    source_root="$(dirname "$cover")"; \\
    cp -a "$source_root"/. {CURRENT_ALIAS}/; \\
    rm -rf /tmp/period-current /tmp/period-current.zip

COPY skills /root/.claude/skills
COPY skills /root/.codex/skills
COPY skills /root/.opencode/skill
COPY skills /root/.goose/skills
COPY skills /root/.factory/skills
COPY skills /root/.agents/skills
COPY skills /root/.gemini/skills

RUN pip3 install --break-system-packages pandas==2.3.3 rapidfuzz==3.14.3
"""


def _test_script() -> str:
    return """#!/bin/bash
set +e
python3 -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
status=$?
if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit "$status"
"""


def _test_source() -> str:
    return '''from __future__ import annotations

import json
from pathlib import Path


GROUND_TRUTH = Path("/tests/expected_output.json")
OUTPUT_FILE = Path("/root/answers.json")


def test_file_exists() -> None:
    assert OUTPUT_FILE.is_file()


def test_answer_quality() -> None:
    expected = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    observed = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    assert list(observed) == list(expected)
    first = float(observed["q1_answer"])
    truth = float(expected["q1_answer"])
    assert abs(first - truth) <= max(1e-9, abs(truth) * 0.001)
    for key in list(expected)[1:]:
        left = observed[key]
        right = expected[key]
        if isinstance(right, list):
            assert isinstance(left, list)
            assert [str(value).casefold() for value in left] == [
                str(value).casefold() for value in right
            ]
        else:
            assert left == right
'''


def _task_toml() -> str:
    return """version = "1.0"

[metadata]
author_name = "Assumption Agent period-out extension"
difficulty = "hard"
category = "finance"
tags = ["data processing", "financial analysis", "project-authored"]

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 900.0

[environment]
build_timeout_sec = 1200.0
cpus = 1
memory_mb = 4096
storage_mb = 12288
"""


def materialize_measurement_benchmark_v1(
    *,
    upstream_benchmark_root: str | Path,
    private_pack: Mapping[str, Any],
    measurement_view: Mapping[str, Any],
    measurement_gold: Mapping[str, Any],
    previous_archive: str | Path,
    current_archive: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    upstream = Path(upstream_benchmark_root).expanduser().resolve(strict=True)
    previous = Path(previous_archive).expanduser().resolve(strict=True)
    current = Path(current_archive).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not previous.is_file() or not current.is_file():
        raise PeriodOutMaterializationError(
            "period archives must be regular local files"
        )

    private = verify_public_pack(private_pack)
    view = verify_measurement_view(
        measurement_view,
        private_pack=private,
    )
    if build_measurement_view(private) != view:
        raise PeriodOutMaterializationError(
            "measurement view is not the deterministic private-pack view"
        )
    gold = verify_consensus_gold(
        measurement_gold,
        pack=private,
        expected_partition="measurement",
    )
    roots = view["container_roots"]
    if roots != {"previous": PREVIOUS_ALIAS, "current": CURRENT_ALIAS}:
        raise PeriodOutMaterializationError(
            "candidate compatibility aliases drifted"
        )
    previous_source = validate_source_against_pack(
        previous,
        private,
        role="previous",
    )
    current_source = validate_source_against_pack(
        current,
        private,
        role="current",
    )
    for role, source in (
        ("previous", previous_source),
        ("current", current_source),
    ):
        if source.source_kind != "zip":
            raise PeriodOutMaterializationError(
                f"{role} period source must be a local ZIP archive"
            )
        _require_shared_table_root(
            coverpage_ref=source.coverpage_ref,
            infotable_ref=source.infotable_ref,
            role=role,
        )
    expected_ids = [row["item_id"] for row in view["measurement_items"]]
    gold_by_id = {row["item_id"]: row for row in gold["items"]}
    if set(gold_by_id) != set(expected_ids):
        raise PeriodOutMaterializationError(
            "measurement gold item set drifted"
        )

    core = upstream / "core"
    agents = upstream / "agents"
    if not (core / "eval_runner.py").is_file() or not (
        agents / "__init__.py"
    ).is_file():
        raise PeriodOutMaterializationError(
            "upstream SkillLearn runner closure is unavailable"
        )
    previous_archive_sha256 = sha256_file(previous)
    current_archive_sha256 = sha256_file(current)
    tree_hash_cache = {
        _file_identity(previous): previous_archive_sha256,
        _file_identity(current): current_archive_sha256,
    }

    destination.mkdir(parents=True)
    try:
        ignore_runtime_cache = shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
        )
        shutil.copytree(
            core,
            destination / "core",
            ignore=ignore_runtime_cache,
        )
        shutil.copytree(
            agents,
            destination / "agents",
            ignore=ignore_runtime_cache,
        )
        item_receipts: list[dict[str, Any]] = []
        for item in view["measurement_items"]:
            item_id = str(item["item_id"])
            task = destination / "tasks" / FAMILY / item_id
            environment = task / "environment"
            tests = task / "tests"
            environment.mkdir(parents=True)
            tests.mkdir(parents=True)
            _write_text(task / "instruction.md", str(item["instruction"]))
            _write_text(task / "task.toml", _task_toml())
            _write_text(environment / "Dockerfile", _dockerfile())
            _link_or_copy(previous, environment / "period-previous.zip")
            _link_or_copy(current, environment / "period-current.zip")
            _write_text(tests / "test.sh", _test_script(), executable=True)
            _write_text(tests / "test_outputs.py", _test_source())
            expected_path = tests / "expected_output.json"
            expected_path.write_text(
                json.dumps(
                    gold_by_id[item_id]["answers"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            expected_path.chmod(0o644)
            environment_receipt = _tree_receipt(
                environment,
                file_hash_cache=tree_hash_cache,
            )
            tests_receipt = _tree_receipt(
                tests,
                file_hash_cache=tree_hash_cache,
            )
            item_receipts.append(
                {
                    "item_id": item_id,
                    "item_id_hash": payload_hash({"item_id": item_id}),
                    "fold": item["fold"],
                    "template": item["template"],
                    "instruction_sha256": sha256_file(
                        task / "instruction.md"
                    ),
                    "task_toml_sha256": sha256_file(task / "task.toml"),
                    "environment_tree_hash": environment_receipt[
                        "tree_hash"
                    ],
                    "tests_tree_hash": tests_receipt["tree_hash"],
                    "expected_output_sha256": sha256_file(expected_path),
                    "answers_hash": gold_by_id[item_id]["answers_hash"],
                    "raw_content_persisted_in_report": False,
                }
            )
        complete_tree = measurement_benchmark_tree_receipt_v1(
            destination,
            file_hash_cache=tree_hash_cache,
        )
        body = {
            "materialization_version": MATERIALIZATION_VERSION,
            "project_authored_extension": True,
            "official_skilllearnbench_score": False,
            "private_pack_hash": private["pack_hash"],
            "measurement_view_hash": view["measurement_view_hash"],
            "measurement_gold_hash": gold["gold_hash"],
            "previous_archive_sha256": previous_archive_sha256,
            "current_archive_sha256": current_archive_sha256,
            "period_source_receipts": {
                "previous": {
                    "container_alias": PREVIOUS_ALIAS,
                    "archive_sha256": previous_archive_sha256,
                    "coverpage_sha256": previous_source.coverpage_sha256,
                    "infotable_sha256": previous_source.infotable_sha256,
                    "source_fingerprint": previous_source.source_fingerprint,
                    "source_path_persisted": False,
                },
                "current": {
                    "container_alias": CURRENT_ALIAS,
                    "archive_sha256": current_archive_sha256,
                    "coverpage_sha256": current_source.coverpage_sha256,
                    "infotable_sha256": current_source.infotable_sha256,
                    "source_fingerprint": current_source.source_fingerprint,
                    "source_path_persisted": False,
                },
            },
            "period_aliases": {
                "previous": PREVIOUS_ALIAS,
                "current": CURRENT_ALIAS,
                "aliases_are_calendar_labels": False,
            },
            "item_count": len(item_receipts),
            "items": item_receipts,
            "item_set_hash": payload_hash(item_receipts),
            "benchmark_tree_hash": complete_tree["tree_hash"],
            "sealed_task_count_materialized": 0,
            "sealed_content_accessed_by_measurement_root": False,
            "sealed_content_persisted": False,
            "sealed_gold_accessed": False,
            "model_calls": 0,
            "online_judge_calls": 0,
            "secret_value_persisted": False,
        }
        report = {**body, "materialization_hash": payload_hash(body)}
        write_json(destination / MATERIALIZATION_REPORT_NAME, report)
        return report
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-benchmark-root", type=Path, required=True)
    parser.add_argument("--private-pack", type=Path, required=True)
    parser.add_argument("--measurement-view", type=Path, required=True)
    parser.add_argument("--measurement-gold", type=Path, required=True)
    parser.add_argument("--previous-archive", type=Path, required=True)
    parser.add_argument("--current-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = materialize_measurement_benchmark_v1(
        upstream_benchmark_root=args.upstream_benchmark_root,
        private_pack=read_json(args.private_pack),
        measurement_view=read_json(args.measurement_view),
        measurement_gold=read_json(args.measurement_gold),
        previous_archive=args.previous_archive,
        current_archive=args.current_archive,
        output_root=args.output_root,
    )
    # Output only content-free identities; never print instructions or gold.
    print(
        json.dumps(
            {
                "materialization_hash": report["materialization_hash"],
                "benchmark_tree_hash": report["benchmark_tree_hash"],
                "item_count": report["item_count"],
                "sealed_task_count_materialized": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
