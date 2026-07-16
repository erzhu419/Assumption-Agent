from __future__ import annotations

"""Materialize exactly four authorized Replication-C sealed tasks."""

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from replication_runtime.financial_semantic_v2 import materialize as _v1
from replication_runtime.financial_semantic_v2 import pack as _pack

from .hygienic_materialize import (
    CURRENT_ALIAS,
    FAMILY,
    FIXED_FAILURE_CODE,
    GOLD_CONTAINER_PATH,
    GOLD_ISOLATION_POLICY_VERSION,
    PREVIOUS_ALIAS,
    TREE_RECEIPT_VERSION,
    VERIFIER_EVIDENCE_POLICY_VERSION,
    _test_script,
    _test_source,
    _versioned_tree_receipt,
)
from .sealed_prepare import verify_sealed_payload_v1


MATERIALIZATION_VERSION = (
    "financial_sec13f_replication_c_sealed_materialization_v1"
)
MATERIALIZATION_REPORT_NAME = "sealed.materialization.json"


class SealedMaterializationError(RuntimeError):
    """The sealed benchmark failed closed."""


def sealed_benchmark_tree_receipt_v1(
    benchmark_root: str | Path,
    *,
    file_hash_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> dict[str, Any]:
    return _versioned_tree_receipt(
        Path(benchmark_root),
        excluded_relative_paths=frozenset({MATERIALIZATION_REPORT_NAME}),
        file_hash_cache=file_hash_cache,
    )


def _verify_sealed_gold(
    value: Mapping[str, Any],
    *,
    sealed_payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(value)
    body = dict(payload)
    declared = body.pop("gold_hash", None)
    items = sealed_payload["sealed_items"]
    rows = payload.get("items")
    if (
        declared != _pack.payload_hash(body)
        or payload.get("gold_version") != _pack.CONSENSUS_GOLD_VERSION
        or payload.get("partition") != "sealed"
        or payload.get("public_pack_hash") != sealed_payload["private_pack_hash"]
        or payload.get("source_fingerprints")
        != {
            role: sealed_payload["sources"][role]["source_fingerprint"]
            for role in ("previous", "current")
        }
        or payload.get("oracle_ids") != sorted(_pack.REQUIRED_ORACLE_IDS)
        or not isinstance(payload.get("oracle_output_hashes"), list)
        or len(payload["oracle_output_hashes"]) != 2
        or len(set(payload["oracle_output_hashes"])) != 2
        or payload.get("item_count") != 4
        or not isinstance(rows, list)
        or len(rows) != 4
        or payload.get("cross_oracle_agreement") is not True
        or payload.get("candidate_imports") != 0
        or payload.get("model_calls") != 0
        or payload.get("network_calls") != 0
    ):
        raise SealedMaterializationError("sealed gold policy drifted")
    for item, row in zip(items, rows or ()):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"item_id", "answers", "answers_hash"}
            or row.get("item_id") != item["item_id"]
            or not isinstance(row.get("answers"), Mapping)
        ):
            raise SealedMaterializationError("sealed gold rows drifted")
        _pack._validate_answers(item, row["answers"])
        if row.get("answers_hash") != _pack.payload_hash(row["answers"]):
            raise SealedMaterializationError("sealed gold answer hash drifted")
    return payload


def _validate_source(
    path: Path,
    *,
    payload: Mapping[str, Any],
    role: str,
) -> _pack.Sec13FSource:
    source = _pack.Sec13FSource.open(path)
    expected = payload["sources"][role]
    if (
        source.source_kind != "zip"
        or source.source_fingerprint != expected["source_fingerprint"]
        or source.coverpage_sha256 != expected["coverpage_sha256"]
        or source.infotable_sha256 != expected["infotable_sha256"]
    ):
        raise SealedMaterializationError(f"{role} source differs from sealed payload")
    _v1._require_shared_table_root(
        coverpage_ref=source.coverpage_ref,
        infotable_ref=source.infotable_ref,
        role=role,
    )
    return source


def materialize_sealed_benchmark_v1(
    *,
    upstream_benchmark_root: str | Path,
    measurement_view: Mapping[str, Any],
    sealed_payload: Mapping[str, Any],
    sealed_gold: Mapping[str, Any],
    previous_archive: str | Path,
    current_archive: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    upstream = Path(upstream_benchmark_root).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    payload = verify_sealed_payload_v1(
        sealed_payload,
        measurement_view=measurement_view,
    )
    gold = _verify_sealed_gold(sealed_gold, sealed_payload=payload)
    if payload["container_roots"] != {
        "previous": PREVIOUS_ALIAS,
        "current": CURRENT_ALIAS,
    }:
        raise SealedMaterializationError("sealed container aliases drifted")
    previous = Path(previous_archive).expanduser().resolve(strict=True)
    current = Path(current_archive).expanduser().resolve(strict=True)
    previous_source = _validate_source(previous, payload=payload, role="previous")
    current_source = _validate_source(current, payload=payload, role="current")
    core = upstream / "core"
    agents = upstream / "agents"
    if not (core / "eval_runner.py").is_file() or not (agents / "__init__.py").is_file():
        raise SealedMaterializationError("upstream runner closure is unavailable")
    gold_by_id = {row["item_id"]: row for row in gold["items"]}
    item_ids = [str(item["item_id"]) for item in payload["sealed_items"]]
    if len(gold_by_id) != 4 or set(gold_by_id) != set(item_ids):
        raise SealedMaterializationError("sealed gold item set drifted")
    previous_sha = _pack.sha256_file(previous)
    current_sha = _pack.sha256_file(current)
    tree_hash_cache = {
        _v1._file_identity(previous): previous_sha,
        _v1._file_identity(current): current_sha,
    }
    destination.mkdir(parents=True, mode=0o700)
    try:
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
        shutil.copytree(core, destination / "core", ignore=ignore)
        shutil.copytree(agents, destination / "agents", ignore=ignore)
        receipts: list[dict[str, Any]] = []
        for item in payload["sealed_items"]:
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
            expected = tests / "expected_output.json"
            expected.write_text(
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
            # Match the already validated measurement materializer.  Gold is
            # isolated by post-agent mount timing, while the offline verifier
            # must be able to read the read-only /tests bind regardless of
            # container UID/capability details.
            expected.chmod(0o644)
            environment_receipt = _versioned_tree_receipt(
                environment, file_hash_cache=tree_hash_cache
            )
            tests_receipt = _versioned_tree_receipt(
                tests, file_hash_cache=tree_hash_cache
            )
            receipts.append(
                {
                    "item_id_hash": _pack.payload_hash({"item_id": item_id}),
                    "replicate": item["replicate"],
                    "template": item["template"],
                    "instruction_sha256": item["instruction_sha256"],
                    "environment_tree_hash": environment_receipt["tree_hash"],
                    "tests_tree_hash": tests_receipt["tree_hash"],
                    "expected_output_sha256": _pack.sha256_file(expected),
                    "answers_hash": gold_by_id[item_id]["answers_hash"],
                    "raw_content_persisted_in_report": False,
                }
            )
        tree = sealed_benchmark_tree_receipt_v1(
            destination, file_hash_cache=tree_hash_cache
        )
        body = {
            "materialization_version": MATERIALIZATION_VERSION,
            "tree_receipt_version": TREE_RECEIPT_VERSION,
            "project_authored_extension": True,
            "official_skilllearnbench_score": False,
            "private_pack_hash": payload["private_pack_hash"],
            "measurement_view_hash": payload["measurement_view_hash"],
            "sealed_payload_hash": payload["sealed_payload_hash"],
            "sealed_gold_hash": gold["gold_hash"],
            "previous_archive_sha256": previous_sha,
            "current_archive_sha256": current_sha,
            "period_source_receipts": {
                "previous": {
                    "archive_sha256": previous_sha,
                    "source_fingerprint": previous_source.source_fingerprint,
                    "source_path_persisted": False,
                },
                "current": {
                    "archive_sha256": current_sha,
                    "source_fingerprint": current_source.source_fingerprint,
                    "source_path_persisted": False,
                },
            },
            "verifier_isolation_policy": GOLD_ISOLATION_POLICY_VERSION,
            "verifier_evidence_policy": VERIFIER_EVIDENCE_POLICY_VERSION,
            "fixed_failure_code": FIXED_FAILURE_CODE,
            "gold_container_path": GOLD_CONTAINER_PATH,
            "item_count": 4,
            "items": receipts,
            "item_set_hash": _pack.payload_hash(receipts),
            "benchmark_tree_hash": tree["tree_hash"],
            "sealed_task_count_materialized": 4,
            "measurement_task_count_materialized": 0,
            "sealed_content_persisted_in_private_benchmark": True,
            "sealed_content_persisted_in_report": False,
            "sealed_gold_persisted_in_private_verifier_tree": True,
            "sealed_gold_persisted_in_report": False,
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
    parser.add_argument("--sealed-payload", type=Path, required=True)
    parser.add_argument("--sealed-gold", type=Path, required=True)
    parser.add_argument("--previous-archive", type=Path, required=True)
    parser.add_argument("--current-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = materialize_sealed_benchmark_v1(
        upstream_benchmark_root=args.upstream_benchmark_root,
        measurement_view=_pack.read_json(args.measurement_view),
        sealed_payload=_pack.read_json(args.sealed_payload),
        sealed_gold=_pack.read_json(args.sealed_gold),
        previous_archive=args.previous_archive,
        current_archive=args.current_archive,
        output_root=args.output_root,
    )
    print(json.dumps({"materialization_hash": report["materialization_hash"], "item_count": 4}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
