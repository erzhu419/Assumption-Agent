#!/usr/bin/env python3
"""One-shot CLI for the consumed HybridQA set-interaction qualification."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import stat
import sys
import traceback
from typing import Mapping


PROJECT_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCHEMA = "hybridqa_set_interaction_meta_development_freeze_v1"
DIAGNOSTIC_VERSION = "hybridqa_set_interaction_meta_development_v1"
ARCHITECTURE_DECISION_SHA256 = (
    "fe9bc18d100a190faea21121ef0d934ea1fa222885fa031aba22d9d830ff9421"
)
PRIOR_MARGINAL_RESULT_SELF_SHA256 = (
    "78c882902ded2830b84a987c68262bb56d1459a444eed58046da923a88f636f5"
)
REQUIRED_IMPLEMENTATION_PATHS = frozenset(
    {
        "artifacts/hybridqa_marginal_replacement_meta_development_v1/result.safe.json",
        "artifacts/hybridqa_marginal_replacement_meta_development_v1/terminal.json",
        "assumption_agent/__init__.py",
        "assumption_agent/benchmarks/__init__.py",
        "assumption_agent/benchmarks/feverous_e2_evaluator_v1.py",
        "assumption_agent/benchmarks/hybridqa_direct_acquisition_v2.py",
        "assumption_agent/benchmarks/hybridqa_marginal_replacement_meta_development_v1.py",
        "assumption_agent/benchmarks/hybridqa_query_anchored_formal_runner_v1.py",
        "assumption_agent/benchmarks/hybridqa_query_anchored_operator_v1.py",
        "assumption_agent/benchmarks/hybridqa_set_interaction_meta_development_v1.py",
        "assumption_agent/benchmarks/hybridqa_source_qualification_v1.py",
        "assumption_agent/models.py",
        "manifests/hybridqa_marginal_replacement_gpu_runtime_qualification_v1.json",
        "manifests/hybridqa_marginal_replacement_meta_development_result_v1.json",
        "manifests/hybridqa_set_interaction_meta_development_v1.service",
        "manifests/hybridqa_set_interaction_numeric_runtime_qualification_v1.json",
        "manifests/qasper_minilm_runtime_asset_v1.json",
        "manifests/red_queen_set_interaction_architecture_decision_v1.json",
        "replication_runtime/__init__.py",
        "replication_runtime/bright_minilm_v1/__init__.py",
        "replication_runtime/bright_minilm_v1/encoder.py",
        "replication_runtime/multihoprag_minilm_v1/__init__.py",
        "replication_runtime/multihoprag_minilm_v1/adapter.py",
        "replication_runtime/qasper_minilm_v1/__init__.py",
        "replication_runtime/qasper_minilm_v1/binding.py",
        "scripts/run_hybridqa_set_interaction_meta_development_v1.py",
        "tests/test_hybridqa_marginal_replacement_meta_development_v1.py",
        "tests/test_hybridqa_set_interaction_meta_development_v1.py",
        "tests/test_run_hybridqa_set_interaction_meta_development_v1.py",
    }
)


class SetInteractionBootstrapError(RuntimeError):
    """A pre-import deployment, freeze, or source binding drifted."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)[:-1]).hexdigest()


def self_hashed(body: Mapping[str, object], field: str) -> dict[str, object]:
    if field in body:
        raise SetInteractionBootstrapError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def verify_self_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or stable_hash(body) != declared
    ):
        raise SetInteractionBootstrapError(f"{field} drifted")
    return declared


def write_exclusive(path: Path, value: object, *, mode: int = 0o600) -> str:
    raw = canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_symlink_ancestors(path: Path, *, missing_leaf_ok: bool = False) -> None:
    if not path.is_absolute():
        raise SetInteractionBootstrapError("bound path is not absolute")
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if missing_leaf_ok and index == len(parts) - 1:
                return
            raise SetInteractionBootstrapError(
                "bound path ancestor is unavailable"
            ) from None
        if stat.S_ISLNK(metadata.st_mode):
            raise SetInteractionBootstrapError(
                "bound path contains a symlink ancestor"
            )


def load_freeze(path: Path) -> tuple[dict[str, object], str]:
    try:
        reject_symlink_ancestors(path)
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SetInteractionBootstrapError(
            "freeze unavailable or invalid"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or not isinstance(value, dict)
        or raw != canonical_bytes(value)
        or value.get("schema") != FREEZE_SCHEMA
    ):
        raise SetInteractionBootstrapError("freeze envelope drifted")
    self_sha = verify_self_hash(value, "self_sha256")
    return value, self_sha


def validate_frozen_implementation(freeze: Mapping[str, object]) -> None:
    required = freeze.get("required_file_sha256s")
    runtime_resolved = freeze.get("runtime_python_resolved")
    runtime_sha = freeze.get("runtime_python_sha256")
    observed_runtime = Path(sys.executable).resolve(strict=True)
    if (
        not isinstance(required, dict)
        or set(required) != REQUIRED_IMPLEMENTATION_PATHS
        or not isinstance(runtime_resolved, str)
        or not isinstance(runtime_sha, str)
        or str(observed_runtime) != runtime_resolved
        or sha256_file(observed_runtime) != runtime_sha
    ):
        raise SetInteractionBootstrapError(
            "runtime or implementation binding drifted"
        )
    for relative, expected in sorted(required.items()):
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise SetInteractionBootstrapError(
                "required implementation binding is invalid"
            )
        path = PROJECT_PACKAGE_ROOT / relative
        try:
            reject_symlink_ancestors(path)
            metadata = path.lstat()
        except OSError as exc:
            raise SetInteractionBootstrapError(
                "required implementation file unavailable"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or sha256_file(path) != expected
        ):
            raise SetInteractionBootstrapError(
                "required implementation file drifted"
            )


def validate_output_parent(output_root: Path) -> None:
    reject_symlink_ancestors(output_root, missing_leaf_ok=True)
    if output_root.exists() or output_root.is_symlink():
        raise SetInteractionBootstrapError(
            "fresh output root already exists"
        )
    try:
        metadata = output_root.parent.lstat()
    except OSError as exc:
        raise SetInteractionBootstrapError(
            "output parent unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SetInteractionBootstrapError("output parent mode drifted")


def validate_acquisition_binding(
    acquisition_root: Path,
    freeze: Mapping[str, object],
) -> dict[str, object]:
    binding = freeze.get("acquisition_binding")
    if not isinstance(binding, Mapping):
        raise SetInteractionBootstrapError("acquisition binding is unavailable")
    public_filename = binding.get("public_filename")
    public_file_sha256 = binding.get("public_file_sha256")
    public_self_sha256 = binding.get("acquisition_receipt_sha256")
    private_hashes = binding.get("private_pack_file_sha256s")
    if (
        not isinstance(public_filename, str)
        or "/" in public_filename
        or public_filename in {"", ".", ".."}
        or not isinstance(public_file_sha256, str)
        or len(public_file_sha256) != 64
        or not isinstance(public_self_sha256, str)
        or len(public_self_sha256) != 64
        or not isinstance(private_hashes, Mapping)
        or len(private_hashes) != 7
    ):
        raise SetInteractionBootstrapError("acquisition binding is invalid")
    reject_symlink_ancestors(acquisition_root)
    root_metadata = acquisition_root.lstat()
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o500
    ):
        raise SetInteractionBootstrapError("acquisition root drifted")
    public_path = acquisition_root / public_filename
    reject_symlink_ancestors(public_path)
    public_metadata = public_path.lstat()
    public_raw = public_path.read_bytes()
    try:
        public = json.loads(public_raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SetInteractionBootstrapError(
            "acquisition public receipt is invalid"
        ) from exc
    if (
        stat.S_ISLNK(public_metadata.st_mode)
        or not stat.S_ISREG(public_metadata.st_mode)
        or not isinstance(public, dict)
        or public_raw != canonical_bytes(public)
        or hashlib.sha256(public_raw).hexdigest() != public_file_sha256
        or verify_self_hash(public, "acquisition_receipt_sha256")
        != public_self_sha256
    ):
        raise SetInteractionBootstrapError(
            "acquisition public receipt drifted"
        )
    declared_private = public.get("private_pack_file_sha256s")
    if not isinstance(declared_private, Mapping):
        raise SetInteractionBootstrapError(
            "acquisition private binding is unavailable"
        )
    verified: dict[str, str] = {}
    for filename, expected in sorted(private_hashes.items()):
        if (
            not isinstance(filename, str)
            or "/" in filename
            or filename in {"", ".", ".."}
            or not isinstance(expected, str)
            or len(expected) != 64
            or declared_private.get(filename) != expected
        ):
            raise SetInteractionBootstrapError(
                "acquisition private binding is invalid"
            )
        path = acquisition_root / filename
        reject_symlink_ancestors(path)
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or sha256_file(path) != expected
        ):
            raise SetInteractionBootstrapError(
                "acquisition private pack drifted"
            )
        verified[filename] = expected
    return {
        "public_file_sha256": public_file_sha256,
        "acquisition_receipt_sha256": public_self_sha256,
        "private_pack_file_sha256s": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    arguments = parser.parse_args()

    freeze, freeze_self = load_freeze(arguments.freeze)
    validate_frozen_implementation(freeze)
    if (
        freeze.get("architecture_decision_self_sha256")
        != ARCHITECTURE_DECISION_SHA256
        or freeze.get("prior_marginal_result_self_sha256")
        != PRIOR_MARGINAL_RESULT_SELF_SHA256
        or freeze.get("output_root") != str(arguments.output_root)
        or freeze.get("acquisition_root") != str(arguments.acquisition_root)
        or freeze.get("minilm_asset_manifest") != str(arguments.asset_manifest)
        or freeze.get("minilm_model_root") != str(arguments.model_root)
    ):
        raise SetInteractionBootstrapError("CLI/freeze binding drifted")
    validate_output_parent(arguments.output_root)
    os.mkdir(arguments.output_root, mode=0o700)
    attempt = self_hashed(
        {
            "schema": f"{DIAGNOSTIC_VERSION}_attempt",
            "version": DIAGNOSTIC_VERSION,
            "status": "attempt_started",
            "architecture_decision_self_sha256": (
                ARCHITECTURE_DECISION_SHA256
            ),
            "prior_marginal_result_self_sha256": (
                PRIOR_MARGINAL_RESULT_SELF_SHA256
            ),
            "implementation_freeze_self_sha256": freeze_self,
            "retry_replay_resample": 0,
        },
        "attempt_self_sha256",
    )
    write_exclusive(arguments.output_root / "attempt.json", attempt)

    try:
        sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))
        base = import_module(
            "assumption_agent.benchmarks."
            "hybridqa_marginal_replacement_meta_development_v1"
        )
        diagnostic = import_module(
            "assumption_agent.benchmarks."
            "hybridqa_set_interaction_meta_development_v1"
        )
        if (
            diagnostic.VERSION != DIAGNOSTIC_VERSION
            or diagnostic.ARCHITECTURE_DECISION_SHA256
            != ARCHITECTURE_DECISION_SHA256
            or diagnostic.PRIOR_MARGINAL_RESULT_SELF_SHA256
            != PRIOR_MARGINAL_RESULT_SELF_SHA256
        ):
            raise SetInteractionBootstrapError(
                "imported diagnostic identity drifted"
            )
        diagnostic.verify_set_energy_numeric_canary()
        validate_acquisition_binding(arguments.acquisition_root, freeze)
        corpus, views, labels = base.load_consumed_packs(
            arguments.acquisition_root
        )
        encoder = base.open_gpu_encoder(
            asset_manifest_path=arguments.asset_manifest,
            model_root=arguments.model_root,
        )
        index = base.build_portable_corpus_index(
            articles=corpus.articles,
            encoder=encoder,
        )
        items = base.form_items(
            corpus=corpus,
            views=views,
            labels=labels,
            index=index,
            encoder=encoder,
        )
        evaluation = diagnostic.evaluate_crossfit(
            items=items,
            graph=corpus.graph,
        )
        result = diagnostic.build_safe_result(
            corpus=corpus,
            items=items,
            evaluation=evaluation,
            encoder=encoder,
            freeze_self_sha256=freeze_self,
        )
        result_file_sha = write_exclusive(
            arguments.output_root / "result.safe.json",
            result,
        )
        terminal = diagnostic.self_hashed(
            {
                "schema": f"{diagnostic.VERSION}_terminal",
                "version": diagnostic.VERSION,
                "status": "complete",
                "decision": evaluation["decision"],
                "result_self_sha256": result["result_self_sha256"],
                "result_file_sha256": result_file_sha,
                "implementation_freeze_self_sha256": freeze_self,
                "retry_replay_resample": 0,
            },
            "terminal_self_sha256",
        )
        write_exclusive(arguments.output_root / "terminal.json", terminal)
        return 0
    except BaseException as exc:
        failure = self_hashed(
            {
                "schema": f"{DIAGNOSTIC_VERSION}_terminal_failure",
                "version": DIAGNOSTIC_VERSION,
                "status": "implementation_or_runtime_invalid",
                "exception_class": type(exc).__name__,
                "exception_message": str(exc)[:512],
                "traceback_sha256": hashlib.sha256(
                    traceback.format_exc().encode("utf-8")
                ).hexdigest(),
                "implementation_freeze_self_sha256": freeze_self,
                "retry_replay_resample": 0,
                "efficacy_inference_permitted": False,
            },
            "failure_self_sha256",
        )
        write_exclusive(arguments.output_root / "failure.safe.json", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
