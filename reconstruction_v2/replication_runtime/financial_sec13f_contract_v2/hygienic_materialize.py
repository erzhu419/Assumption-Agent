from __future__ import annotations

"""Materialize a measurement-only SEC-13F benchmark with hygienic failures.

The v1 materializer remains frozen.  This additive implementation reuses its
audited Docker/task templates and filesystem helpers, but changes the verifier
contract in three important ways:

* only the redacted measurement view is accepted; the private pack (and thus
  sealed item content) is neither required nor read;
* verifier failures have one fixed, answer-free message and pytest is invoked
  with traceback rendering disabled; and
* every tree digest is bound to an explicit receipt schema version.

Gold is written exactly once per task, at ``tests/expected_output.json``.  The
environment Dockerfile does not copy ``tests`` into the image, so its runtime
location (``/tests/expected_output.json``) is outside the candidate namespace
until the runner's post-agent verifier mount.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from replication_runtime.financial_semantic_v2 import materialize as _v1
from replication_runtime.financial_semantic_v2 import pack as _pack


MATERIALIZATION_VERSION = (
    "financial_sec13f_contract_hygienic_measurement_materialization_v2"
)
TREE_RECEIPT_VERSION = "financial_sec13f_contract_tree_receipt_v2"
VERIFIER_EVIDENCE_POLICY_VERSION = (
    "financial_sec13f_fixed_failure_evidence_v2"
)
GOLD_ISOLATION_POLICY_VERSION = (
    "financial_sec13f_post_agent_tests_namespace_v2"
)
MATERIALIZATION_REPORT_NAME = "measurement.materialization.json"
FAMILY = _v1.FAMILY
PREVIOUS_ALIAS = _v1.PREVIOUS_ALIAS
CURRENT_ALIAS = _v1.CURRENT_ALIAS
GOLD_CONTAINER_PATH = "/tests/expected_output.json"
FIXED_FAILURE_CODE = "answer_verification_failed"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HygienicMaterializationError(RuntimeError):
    """The redacted measurement benchmark failed closed."""


def _versioned_tree_receipt(
    root: Path,
    *,
    excluded_relative_paths: frozenset[str] = frozenset(),
    file_hash_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> dict[str, Any]:
    """Return a path-independent, schema-bound tree receipt."""

    resolved = root.resolve(strict=True)
    hashes = file_hash_cache if file_hash_cache is not None else {}
    rows: list[dict[str, Any]] = []
    for path in sorted(
        resolved.rglob("*"),
        key=lambda value: value.relative_to(resolved).as_posix(),
    ):
        if path.is_symlink():
            raise HygienicMaterializationError(
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
                digest = _pack.sha256_file(path)
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
            raise HygienicMaterializationError(
                "materialized benchmark contains a special file"
            )
    body = {
        "tree_receipt_version": TREE_RECEIPT_VERSION,
        "excluded_relative_paths": sorted(excluded_relative_paths),
        "file_and_directory_count": len(rows),
        "rows": rows,
    }
    return {**body, "tree_hash": _pack.payload_hash(body)}


def measurement_benchmark_tree_receipt_v2(
    benchmark_root: str | Path,
    *,
    file_hash_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> dict[str, Any]:
    """Hash the immutable runner/task tree, excluding its self-report."""

    return _versioned_tree_receipt(
        Path(benchmark_root),
        excluded_relative_paths=frozenset({MATERIALIZATION_REPORT_NAME}),
        file_hash_cache=file_hash_cache,
    )


def _verify_measurement_gold_without_private_pack(
    value: Mapping[str, Any],
    *,
    measurement_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate measurement gold against the redacted view only.

    ``financial_semantic_v2.pack.verify_consensus_gold`` deliberately requires
    the full public pack.  Calling it here would touch sealed item content, so
    v2 validates the same measurement invariants against the committed view.
    """

    if not isinstance(value, Mapping):
        raise HygienicMaterializationError("measurement gold must be an object")
    payload = dict(value)
    declared_hash = payload.get("gold_hash")
    if (
        not isinstance(declared_hash, str)
        or _SHA256.fullmatch(declared_hash) is None
    ):
        raise HygienicMaterializationError("measurement gold hash is malformed")
    body = dict(payload)
    del body["gold_hash"]
    if _pack.payload_hash(body) != declared_hash:
        raise HygienicMaterializationError("measurement gold self hash drifted")
    expected_fields = {
        "gold_version",
        "partition",
        "public_pack_hash",
        "source_fingerprints",
        "oracle_ids",
        "oracle_output_hashes",
        "item_count",
        "items",
        "cross_oracle_agreement",
        "candidate_imports",
        "model_calls",
        "network_calls",
        "gold_hash",
    }
    if set(payload) != expected_fields:
        raise HygienicMaterializationError("measurement gold fields drifted")
    if (
        payload.get("gold_version") != _pack.CONSENSUS_GOLD_VERSION
        or payload.get("partition") != "measurement"
        or payload.get("public_pack_hash")
        != measurement_view["private_pack_hash"]
        or payload.get("cross_oracle_agreement") is not True
        or payload.get("candidate_imports") != 0
        or payload.get("model_calls") != 0
        or payload.get("network_calls") != 0
    ):
        raise HygienicMaterializationError("measurement gold policy drifted")
    expected_fingerprints = {
        role: measurement_view["sources"][role]["source_fingerprint"]
        for role in ("previous", "current")
    }
    if payload.get("source_fingerprints") != expected_fingerprints:
        raise HygienicMaterializationError(
            "measurement gold source binding drifted"
        )
    if payload.get("oracle_ids") != sorted(_pack.REQUIRED_ORACLE_IDS):
        raise HygienicMaterializationError(
            "measurement gold oracle binding drifted"
        )
    oracle_hashes = payload.get("oracle_output_hashes")
    if (
        not isinstance(oracle_hashes, list)
        or len(oracle_hashes) != 2
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in oracle_hashes
        )
        or len(set(oracle_hashes)) != 2
    ):
        raise HygienicMaterializationError(
            "measurement gold oracle receipts drifted"
        )
    measurement_items = measurement_view["measurement_items"]
    rows = payload.get("items")
    if (
        payload.get("item_count") != len(measurement_items)
        or not isinstance(rows, list)
        or len(rows) != len(measurement_items)
    ):
        raise HygienicMaterializationError("measurement gold rows drifted")
    for item, row in zip(measurement_items, rows):
        if not isinstance(row, Mapping) or set(row) != {
            "item_id",
            "answers",
            "answers_hash",
        }:
            raise HygienicMaterializationError(
                "measurement gold row fields drifted"
            )
        if row.get("item_id") != item["item_id"]:
            raise HygienicMaterializationError(
                "measurement gold item order drifted"
            )
        answers = row.get("answers")
        if not isinstance(answers, Mapping):
            raise HygienicMaterializationError(
                "measurement gold answers are malformed"
            )
        # Reuse the frozen answer-schema validator.  It sees only one redacted
        # measurement item and its corresponding measurement gold row.
        _pack._validate_answers(item, answers)
        if row.get("answers_hash") != _pack.payload_hash(answers):
            raise HygienicMaterializationError(
                "measurement gold answer hash drifted"
            )
    return payload


def _validate_source_against_view(
    source: Path,
    *,
    measurement_view: Mapping[str, Any],
    role: str,
) -> _pack.Sec13FSource:
    opened = _pack.Sec13FSource.open(source)
    expected = measurement_view["sources"][role]
    if (
        opened.source_fingerprint != expected["source_fingerprint"]
        or opened.coverpage_sha256 != expected["coverpage_sha256"]
        or opened.infotable_sha256 != expected["infotable_sha256"]
    ):
        raise HygienicMaterializationError(
            f"{role} SEC source differs from measurement view"
        )
    return opened


def _test_script() -> str:
    return """#!/bin/bash
set +e
python3 -m pytest --tb=no --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -q
status=$?
if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit "$status"
"""


def _test_source() -> str:
    # Deliberately avoid bare ``assert`` statements.  Pytest assertion
    # rewriting can serialize expected/observed representations into CTRF.
    return f'''from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


GROUND_TRUTH = Path("/tests/expected_output.json")
OUTPUT_FILE = Path("/root/answers.json")
FAILURE_CODE = "{FIXED_FAILURE_CODE}"


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _answer_matches() -> bool:
    expected = _read_json(GROUND_TRUTH)
    observed = _read_json(OUTPUT_FILE)
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        return False
    try:
        if list(observed) != list(expected):
            return False
        first = float(observed["q1_answer"])
        truth = float(expected["q1_answer"])
        if abs(first - truth) > max(1e-9, abs(truth) * 0.001):
            return False
        for key in list(expected)[1:]:
            left = observed[key]
            right = expected[key]
            if isinstance(right, list):
                if not isinstance(left, list):
                    return False
                if [str(value).casefold() for value in left] != [
                    str(value).casefold() for value in right
                ]:
                    return False
            elif left != right:
                return False
        return True
    except Exception:
        return False


def test_answer_contract() -> None:
    if not _answer_matches():
        pytest.fail(FAILURE_CODE, pytrace=False)
'''


def materialize_measurement_benchmark_v2(
    *,
    upstream_benchmark_root: str | Path,
    measurement_view: Mapping[str, Any],
    measurement_gold: Mapping[str, Any],
    previous_archive: str | Path,
    current_archive: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create a deterministic, redacted-view-only measurement benchmark."""

    upstream = Path(upstream_benchmark_root).expanduser().resolve(strict=True)
    previous = Path(previous_archive).expanduser().resolve(strict=True)
    current = Path(current_archive).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not previous.is_file() or not current.is_file():
        raise HygienicMaterializationError(
            "period archives must be regular local files"
        )

    view = _pack.verify_measurement_view(measurement_view)
    gold = _verify_measurement_gold_without_private_pack(
        measurement_gold,
        measurement_view=view,
    )
    roots = view["container_roots"]
    if roots != {"previous": PREVIOUS_ALIAS, "current": CURRENT_ALIAS}:
        raise HygienicMaterializationError(
            "candidate compatibility aliases drifted"
        )
    previous_source = _validate_source_against_view(
        previous,
        measurement_view=view,
        role="previous",
    )
    current_source = _validate_source_against_view(
        current,
        measurement_view=view,
        role="current",
    )
    for role, source in (
        ("previous", previous_source),
        ("current", current_source),
    ):
        if source.source_kind != "zip":
            raise HygienicMaterializationError(
                f"{role} period source must be a local ZIP archive"
            )
        _v1._require_shared_table_root(
            coverpage_ref=source.coverpage_ref,
            infotable_ref=source.infotable_ref,
            role=role,
        )
    expected_ids = [row["item_id"] for row in view["measurement_items"]]
    gold_by_id = {row["item_id"]: row for row in gold["items"]}
    if set(gold_by_id) != set(expected_ids):
        raise HygienicMaterializationError(
            "measurement gold item set drifted"
        )

    core = upstream / "core"
    agents = upstream / "agents"
    if not (core / "eval_runner.py").is_file() or not (
        agents / "__init__.py"
    ).is_file():
        raise HygienicMaterializationError(
            "upstream SkillLearn runner closure is unavailable"
        )
    previous_archive_sha256 = _pack.sha256_file(previous)
    current_archive_sha256 = _pack.sha256_file(current)
    tree_hash_cache = {
        _v1._file_identity(previous): previous_archive_sha256,
        _v1._file_identity(current): current_archive_sha256,
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
            _v1._write_text(task / "instruction.md", str(item["instruction"]))
            _v1._write_text(task / "task.toml", _v1._task_toml())
            _v1._write_text(environment / "Dockerfile", _v1._dockerfile())
            _v1._link_or_copy(previous, environment / "period-previous.zip")
            _v1._link_or_copy(current, environment / "period-current.zip")
            _v1._write_text(tests / "test.sh", _test_script(), executable=True)
            _v1._write_text(tests / "test_outputs.py", _test_source())
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
            environment_receipt = _versioned_tree_receipt(
                environment,
                file_hash_cache=tree_hash_cache,
            )
            tests_receipt = _versioned_tree_receipt(
                tests,
                file_hash_cache=tree_hash_cache,
            )
            item_receipts.append(
                {
                    "item_id": item_id,
                    "item_id_hash": _pack.payload_hash({"item_id": item_id}),
                    "fold": item["fold"],
                    "template": item["template"],
                    "instruction_sha256": _pack.sha256_file(
                        task / "instruction.md"
                    ),
                    "task_toml_sha256": _pack.sha256_file(task / "task.toml"),
                    "tree_receipt_version": TREE_RECEIPT_VERSION,
                    "environment_tree_hash": environment_receipt["tree_hash"],
                    "tests_tree_hash": tests_receipt["tree_hash"],
                    "expected_output_sha256": _pack.sha256_file(expected_path),
                    "answers_hash": gold_by_id[item_id]["answers_hash"],
                    "gold_container_path": GOLD_CONTAINER_PATH,
                    "gold_in_environment_tree": False,
                    "failure_evidence_policy": (
                        VERIFIER_EVIDENCE_POLICY_VERSION
                    ),
                    "raw_content_persisted_in_report": False,
                }
            )
        complete_tree = measurement_benchmark_tree_receipt_v2(
            destination,
            file_hash_cache=tree_hash_cache,
        )
        body = {
            "materialization_version": MATERIALIZATION_VERSION,
            "tree_receipt_version": TREE_RECEIPT_VERSION,
            "project_authored_extension": True,
            "official_skilllearnbench_score": False,
            "private_pack_hash": view["private_pack_hash"],
            "private_pack_accessed": False,
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
            "verifier_isolation": {
                "policy_version": GOLD_ISOLATION_POLICY_VERSION,
                "candidate_namespace": "/root",
                "verifier_namespace": "/tests",
                "gold_container_path": GOLD_CONTAINER_PATH,
                "gold_written_outside_environment_tree": True,
                "gold_copied_into_environment_image": False,
                "candidate_visible_gold_path_count": 0,
                "post_agent_verifier_mount_required": True,
            },
            "verifier_evidence": {
                "policy_version": VERIFIER_EVIDENCE_POLICY_VERSION,
                "failure_message_kind": "fixed_code_only",
                "fixed_failure_code": FIXED_FAILURE_CODE,
                "pytest_traceback_mode": "none",
                "raw_expected_in_failure_evidence": False,
                "raw_observed_in_failure_evidence": False,
            },
            "item_count": len(item_receipts),
            "items": item_receipts,
            "item_set_hash": _pack.payload_hash(item_receipts),
            "benchmark_tree_hash": complete_tree["tree_hash"],
            "sealed_task_count_materialized": 0,
            "sealed_content_accessed_by_materializer": False,
            "sealed_content_accessed_by_measurement_root": False,
            "sealed_content_persisted": False,
            "sealed_gold_accessed": False,
            "model_calls": 0,
            "online_judge_calls": 0,
            "secret_value_persisted": False,
        }
        report = {**body, "materialization_hash": _pack.payload_hash(body)}
        _pack.write_json(destination / MATERIALIZATION_REPORT_NAME, report)
        return report
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-benchmark-root", type=Path, required=True)
    parser.add_argument("--measurement-view", type=Path, required=True)
    parser.add_argument("--measurement-gold", type=Path, required=True)
    parser.add_argument("--previous-archive", type=Path, required=True)
    parser.add_argument("--current-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = materialize_measurement_benchmark_v2(
        upstream_benchmark_root=args.upstream_benchmark_root,
        measurement_view=_pack.read_json(args.measurement_view),
        measurement_gold=_pack.read_json(args.measurement_gold),
        previous_archive=args.previous_archive,
        current_archive=args.current_archive,
        output_root=args.output_root,
    )
    # Content-free identities only; never print instructions, gold, or output.
    print(
        json.dumps(
            {
                "materialization_hash": report["materialization_hash"],
                "benchmark_tree_hash": report["benchmark_tree_hash"],
                "tree_receipt_version": report["tree_receipt_version"],
                "item_count": report["item_count"],
                "sealed_task_count_materialized": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
