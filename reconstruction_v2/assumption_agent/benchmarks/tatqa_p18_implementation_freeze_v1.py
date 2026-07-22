"""Pre-source runtime fingerprint and implementation freeze tooling for P18.

Both artifacts are deliberately deterministic and exclusive.  The runtime
fingerprint binds five immutable runtime/model trees, the concrete inventory,
and the successful user-systemd IP-network denial preflight.  The subsequent
implementation freeze binds *exactly* the registry declared by trusted
acquisition to bytes at one committed Git revision, plus already committed and
clean fingerprint/canary receipts.  Neither operation has a TAT-QA source
loader or accepts a formal item identifier.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import tatqa_p18_acquisition_v1 as acquisition
from assumption_agent.benchmarks import tatqa_p18_public_canary_v1 as canary
from replication_runtime.tatqa_p18_v1 import formal_runtime


VERSION = "tatqa_p18_implementation_freeze_tool_v1"
RUNTIME_FINGERPRINT_SCHEMA = "tatqa_p18_remote_runtime_fingerprint_v1"
IMPLEMENTATION_FREEZE_SCHEMA = "tatqa_p18_implementation_freeze_v1"
ASSET_NAMES = (
    "HippoRAG_LLM",
    "HippoRAG_embedding",
    "HippoRAG_source",
    "MiniLM",
    "Qwen",
)
REQUIRED_BINDING_PATHS = frozenset(
    {
        "assumption_agent/benchmarks/tatqa_p18_acquisition_v1.py",
        "assumption_agent/benchmarks/tatqa_p18_formal_adapters_v1.py",
        "assumption_agent/benchmarks/tatqa_p18_formal_controller_v1.py",
        "assumption_agent/benchmarks/tatqa_p18_formal_study_v1.py",
        "assumption_agent/benchmarks/tatqa_p18_implementation_freeze_v1.py",
        "assumption_agent/benchmarks/tatqa_p18_label_free_runtime_v1.py",
        "assumption_agent/benchmarks/tatqa_p18_offline_finalize_v1.py",
        "assumption_agent/benchmarks/tatqa_p18_public_canary_v1.py",
        "assumption_agent/benchmarks/tatqa_p18_runtime_qualification_v1.py",
        "assumption_agent/benchmarks/tatqa_p18_source_download_v1.py",
        "assumption_agent/benchmarks/tatqa_p18_typed_evaluator_core_v1.py",
        "replication_runtime/qasper_minilm_v1/__init__.py",
        "replication_runtime/qasper_minilm_v1/binding.py",
        "replication_runtime/tatqa_p18_v1/__init__.py",
        "replication_runtime/tatqa_p18_v1/formal_runtime.py",
        "replication_runtime/tatqa_p18_v1/hipporag_contract.py",
        "replication_runtime/tatqa_p18_v1/hipporag_worker.py",
        "replication_runtime/tatqa_p18_v1/typed_plan_contract.py",
        "replication_runtime/tatqa_p18_v1/typed_plan_worker.py",
        "tests/test_tatqa_p18_acquisition_v1.py",
        "tests/test_tatqa_p18_formal_adapters_v1.py",
        "tests/test_tatqa_p18_formal_controller_v1.py",
        "tests/test_tatqa_p18_formal_runtime_v1.py",
        "tests/test_tatqa_p18_formal_study_v1.py",
        "tests/test_tatqa_p18_hipporag_runtime_v1.py",
        "tests/test_tatqa_p18_implementation_freeze_v1.py",
        "tests/test_tatqa_p18_label_free_runtime_v1.py",
        "tests/test_tatqa_p18_offline_finalize_v1.py",
        "tests/test_tatqa_p18_public_canary_v1.py",
        "tests/test_tatqa_p18_runtime_qualification_v1.py",
        "tests/test_tatqa_p18_source_download_v1.py",
        "tests/test_tatqa_p18_study_design_v1.py",
        "tests/test_tatqa_p18_typed_evaluator_core_v1.py",
        "tests/test_tatqa_p18_typed_plan_runtime_v1.py",
    }
)

_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_INVENTORY_KEYS = re.compile(
    r"(?:answer|family|gold|label|mapping|question_uid|table_uid|source_id|"
    r"api[_-]?key|ruoli|secret|credential)",
    flags=re.IGNORECASE,
)


class TatqaP18ImplementationFreezeError(RuntimeError):
    """A pre-source fingerprint, Git binding, or exclusive seal drifted."""


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TatqaP18ImplementationFreezeError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value, newline=False)).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_json(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TatqaP18ImplementationFreezeError(f"{field} is unavailable") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise TatqaP18ImplementationFreezeError(f"{field} is not canonical JSON")
    return value, raw


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TatqaP18ImplementationFreezeError(
            f"{field} is not a lowercase SHA-256"
        )
    return value


def _verify_self_hash(value: Mapping[str, Any]) -> str:
    claimed = _require_sha256(value.get("self_sha256"), "self_sha256")
    body = dict(value)
    del body["self_sha256"]
    if _stable_hash(body) != claimed:
        raise TatqaP18ImplementationFreezeError("receipt self hash drifted")
    return claimed


def _safe_relative(value: str | Path, *, field: str) -> str:
    raw = Path(value).as_posix()
    pure = PurePosixPath(raw)
    if (
        not raw
        or pure.is_absolute()
        or raw != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise TatqaP18ImplementationFreezeError(f"{field} is unsafe")
    return raw


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise TatqaP18ImplementationFreezeError(
            "exclusive receipt path is already consumed"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != raw:
        raise TatqaP18ImplementationFreezeError("receipt reopen verification failed")
    return hashlib.sha256(raw).hexdigest()


def _validate_inventory(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise TatqaP18ImplementationFreezeError("runtime inventory is absent")

    def walk(row: object) -> None:
        if isinstance(row, Mapping):
            for key, child in row.items():
                if not isinstance(key, str) or _FORBIDDEN_INVENTORY_KEYS.search(key):
                    raise TatqaP18ImplementationFreezeError(
                        "runtime inventory contains a forbidden field"
                    )
                walk(child)
        elif isinstance(row, (list, tuple)):
            for child in row:
                walk(child)
        elif row is None or isinstance(row, (str, int, float, bool)):
            return
        else:
            raise TatqaP18ImplementationFreezeError(
                "runtime inventory is not canonical JSON"
            )

    copied = dict(value)
    walk(copied)
    _canonical_bytes(copied)
    return copied


def _validate_network_preflight(value: Mapping[str, object]) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or value.get("network_properties")
        != list(formal_runtime.SYSTEMD_NETWORK_PROPERTIES)
        or value.get("returncode") != 0
    ):
        raise TatqaP18ImplementationFreezeError(
            "systemd network-isolation preflight did not pass"
        )
    for field in ("stdout_sha256", "stderr_sha256"):
        _require_sha256(value.get(field), field)
    return dict(value)


def build_runtime_fingerprint(
    *,
    output_path: str | Path,
    asset_roots: Mapping[str, str | Path],
    runtime_inventory: Mapping[str, object],
    systemd_network_preflight: Mapping[str, object],
    runtime_implementation_commit: str | None = None,
) -> dict[str, Any]:
    """Bind source-free production assets and network isolation before data."""

    if set(asset_roots) != set(ASSET_NAMES):
        raise TatqaP18ImplementationFreezeError("five-asset registry drifted")
    if runtime_implementation_commit is not None and (
        not isinstance(runtime_implementation_commit, str)
        or _SHA1.fullmatch(runtime_implementation_commit) is None
    ):
        raise TatqaP18ImplementationFreezeError(
            "runtime implementation commit is malformed"
        )
    bindings: dict[str, object] = {}
    for name in ASSET_NAMES:
        root = Path(asset_roots[name])
        try:
            bindings[name] = formal_runtime.tree_receipt(root)
        except Exception as exc:
            raise TatqaP18ImplementationFreezeError(
                f"{name} asset tree cannot be frozen"
            ) from exc
    body: dict[str, Any] = {
        "schema": RUNTIME_FINGERPRINT_SCHEMA,
        "status": "verified_before_formal_source_open",
        "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
        "source_custody_self_sha256": acquisition.CUSTODY_SELF_SHA256,
        "runtime_implementation_commit": runtime_implementation_commit,
        "asset_bindings": bindings,
        "runtime_inventory": _validate_inventory(runtime_inventory),
        "systemd_network_preflight": _validate_network_preflight(
            systemd_network_preflight
        ),
        "filesystem_isolation": canary.FILESYSTEM_ISOLATION,
        "formal_source_opened": False,
        "source_identifiers_answers_families_mappings_or_labels_present": False,
        "api_environment_variables_exposed_to_workers": [],
        "external_network_calls": 0,
        "api_or_online_evaluator_calls": 0,
        "retry_replay_resample_provider_switch": 0,
        "controller_or_worker_source_reads": 0,
        "controller_or_worker_label_reads": 0,
        "hippo_concurrency_cap": 8,
        "maximum_cpu_threads_per_hippo_process": 2,
    }
    receipt = {**body, "self_sha256": _stable_hash(body)}
    _write_exclusive(Path(output_path), receipt)
    return receipt


def _git(project: Path, args: Sequence[str], *, field: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise TatqaP18ImplementationFreezeError(f"Git {field} failed") from exc
    if completed.returncode != 0:
        raise TatqaP18ImplementationFreezeError(f"Git {field} failed")
    return completed.stdout


def _git_repo_binding(project: Path) -> tuple[Path, str]:
    repo = Path(
        _git(project, ["rev-parse", "--show-toplevel"], field="repository discovery")
        .decode("utf-8")
        .strip()
    ).resolve(strict=True)
    try:
        prefix = project.resolve(strict=True).relative_to(repo).as_posix()
    except ValueError as exc:
        raise TatqaP18ImplementationFreezeError(
            "project is outside its Git repository"
        ) from exc
    return repo, "" if prefix == "." else prefix


def _repo_path(prefix: str, relative: str) -> str:
    return f"{prefix}/{relative}" if prefix else relative


def _committed_clean_bytes(
    *, project: Path, prefix: str, commit: str, relative: str
) -> bytes:
    git_path = _repo_path(prefix, relative)
    committed = _git(
        project, ["show", f"{commit}:{git_path}"], field=f"read {relative} at commit"
    )
    current_path = project / relative
    try:
        if current_path.is_symlink() or not current_path.is_file():
            raise OSError("not a regular file")
        current = current_path.read_bytes()
    except OSError as exc:
        raise TatqaP18ImplementationFreezeError(
            f"current frozen member {relative} is unavailable"
        ) from exc
    if current != committed:
        raise TatqaP18ImplementationFreezeError(
            f"current frozen member {relative} differs from formal commit"
        )
    status = _git(
        project,
        ["status", "--porcelain", "--untracked-files=all", "--", git_path],
        field=f"cleanliness check for {relative}",
    )
    if status:
        raise TatqaP18ImplementationFreezeError(
            f"frozen member {relative} is not clean"
        )
    return current


def _receipt_binding(
    *,
    project: Path,
    prefix: str,
    commit: str,
    relative: str,
    expected_schema: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    raw = _committed_clean_bytes(
        project=project, prefix=prefix, commit=commit, relative=relative
    )
    value, observed_raw = _strict_json(project / relative, field=relative)
    if observed_raw != raw or value.get("schema") != expected_schema:
        raise TatqaP18ImplementationFreezeError(f"{relative} schema drifted")
    self_sha = _verify_self_hash(value)
    return value, {
        "relative_path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "self_sha256": self_sha,
    }


def build_implementation_freeze(
    *,
    project_root: str | Path,
    formal_implementation_commit: str,
    runtime_fingerprint_relative: str | Path,
    production_canary_relative: str | Path,
    runtime_qualification_terminal_relative: str | Path,
    output_relative: str | Path = acquisition.IMPLEMENTATION_FREEZE_RELATIVE,
) -> dict[str, Any]:
    """Seal exact committed implementation and pre-source runtime receipts."""

    project = Path(project_root).resolve(strict=True)
    if not project.is_dir():
        raise TatqaP18ImplementationFreezeError("project root is not a directory")
    if (
        not isinstance(formal_implementation_commit, str)
        or _SHA1.fullmatch(formal_implementation_commit) is None
    ):
        raise TatqaP18ImplementationFreezeError(
            "formal implementation commit is malformed"
        )
    fingerprint_relative = _safe_relative(
        runtime_fingerprint_relative, field="runtime fingerprint path"
    )
    canary_relative = _safe_relative(
        production_canary_relative, field="production canary path"
    )
    qualification_terminal_relative = _safe_relative(
        runtime_qualification_terminal_relative,
        field="runtime qualification terminal path",
    )
    output_name = _safe_relative(output_relative, field="freeze output path")
    if (project / output_name).exists():
        raise TatqaP18ImplementationFreezeError("freeze output already exists")

    # Formal source bytes and the acquisition one-shot root must not yet exist.
    forbidden = (
        project / acquisition.SOURCE_RECEIPT_RELATIVE,
        project / acquisition.SOURCE_ROOT_RELATIVE,
        project / acquisition.ACQUISITION_ROOT_RELATIVE,
    )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise TatqaP18ImplementationFreezeError(
            "formal source/acquisition state exists before implementation freeze"
        )

    _repo, prefix = _git_repo_binding(project)
    resolved = (
        _git(
            project,
            ["rev-parse", "--verify", f"{formal_implementation_commit}^{{commit}}"],
            field="formal commit verification",
        )
        .decode("ascii")
        .strip()
    )
    if resolved != formal_implementation_commit:
        raise TatqaP18ImplementationFreezeError("formal commit resolution drifted")
    _git(
        project,
        ["merge-base", "--is-ancestor", formal_implementation_commit, "HEAD"],
        field="formal commit ancestry",
    )

    registry = sorted(acquisition.REQUIRED_IMPLEMENTATION_PATHS)
    if (
        not registry
        or registry != sorted(set(registry))
        or frozenset(registry) != acquisition.REQUIRED_IMPLEMENTATION_PATHS
        or acquisition.REQUIRED_IMPLEMENTATION_PATHS != REQUIRED_BINDING_PATHS
    ):
        raise TatqaP18ImplementationFreezeError(
            "trusted acquisition implementation registry is not exact"
        )
    bindings: list[dict[str, str]] = []
    for name in registry:
        safe = _safe_relative(name, field="implementation registry path")
        raw = _committed_clean_bytes(
            project=project,
            prefix=prefix,
            commit=formal_implementation_commit,
            relative=safe,
        )
        bindings.append(
            {"relative_path": safe, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    if [row["relative_path"] for row in bindings] != registry:
        raise TatqaP18ImplementationFreezeError(
            "implementation binding order drifted"
        )

    fingerprint, fingerprint_binding = _receipt_binding(
        project=project,
        prefix=prefix,
        commit=formal_implementation_commit,
        relative=fingerprint_relative,
        expected_schema=RUNTIME_FINGERPRINT_SCHEMA,
    )
    production_canary, canary_binding = _receipt_binding(
        project=project,
        prefix=prefix,
        commit=formal_implementation_commit,
        relative=canary_relative,
        expected_schema=canary.SCHEMA,
    )
    qualification_terminal, qualification_terminal_binding = _receipt_binding(
        project=project,
        prefix=prefix,
        commit=formal_implementation_commit,
        relative=qualification_terminal_relative,
        expected_schema="tatqa_p18_runtime_qualification_v1_terminal_success_v1",
    )
    qualification_commit = fingerprint.get("runtime_implementation_commit")
    if (
        not isinstance(qualification_commit, str)
        or _SHA1.fullmatch(qualification_commit) is None
    ):
        raise TatqaP18ImplementationFreezeError(
            "runtime qualification implementation commit is malformed"
        )
    _git(
        project,
        [
            "merge-base",
            "--is-ancestor",
            qualification_commit,
            formal_implementation_commit,
        ],
        field="runtime qualification commit ancestry",
    )
    for row in bindings:
        relative = row["relative_path"]
        qualified = _git(
            project,
            [
                "show",
                f"{qualification_commit}:{_repo_path(prefix, relative)}",
            ],
            field=f"runtime-qualified {relative}",
        )
        if hashlib.sha256(qualified).hexdigest() != row["sha256"]:
            raise TatqaP18ImplementationFreezeError(
                "production implementation changed after runtime qualification"
            )
    if (
        fingerprint.get("status") != "verified_before_formal_source_open"
        or fingerprint.get("study_design_self_sha256")
        != acquisition.DESIGN_SELF_SHA256
        or fingerprint.get("formal_source_opened") is not False
        or fingerprint.get("source_identifiers_answers_families_mappings_or_labels_present")
        is not False
        or fingerprint.get("external_network_calls") != 0
        or fingerprint.get("api_or_online_evaluator_calls") != 0
        or production_canary.get("status")
        != "qualified_before_formal_source_open"
        or production_canary.get("qualified") is not True
        or production_canary.get("runtime_fingerprint_self_sha256")
        != fingerprint_binding["self_sha256"]
        or production_canary.get("formal_source_opened") is not False
        or production_canary.get("hippo_canary_ran") is not True
        or production_canary.get("P1_retains_ordered_P0_top3") is not True
        or isinstance(
            production_canary.get("P1_outside_P0_unit_count"), bool
        )
        or not isinstance(
            production_canary.get("P1_outside_P0_unit_count"), int
        )
        or production_canary["P1_outside_P0_unit_count"] < 1
        or production_canary.get("typed_plan_worker_receipt_source")
        != "capability_receipt_snapshot"
        or production_canary.get("minilm_worker_receipt_source")
        != "capability_receipt_snapshot"
        or production_canary.get("hippo_worker_receipt_source")
        != "capability_receipt_snapshot"
        or production_canary.get("external_network_calls") != 0
        or production_canary.get("api_or_online_evaluator_calls") != 0
        or qualification_terminal.get("status")
        != "qualified_before_formal_source_open"
        or qualification_terminal.get("runtime_fingerprint_self_sha256")
        != fingerprint_binding["self_sha256"]
        or qualification_terminal.get("production_canary_self_sha256")
        != canary_binding["self_sha256"]
        or qualification_terminal.get("formal_source_opened") is not False
        or qualification_terminal.get("retry_requalification") != 0
    ):
        raise TatqaP18ImplementationFreezeError(
            "runtime fingerprint/canary is not a qualified pre-source pair"
        )
    try:
        acquisition.validate_production_canary_capability_receipts(
            production_canary
        )
    except Exception as exc:
        raise TatqaP18ImplementationFreezeError(
            "production canary capability evidence drifted"
        ) from exc

    body: dict[str, Any] = {
        "schema": IMPLEMENTATION_FREEZE_SCHEMA,
        "status": "implementation_frozen",
        "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
        "source_custody_self_sha256": acquisition.CUSTODY_SELF_SHA256,
        "formal_implementation_commit": formal_implementation_commit,
        "runtime_qualification_implementation_commit": qualification_commit,
        "implementation_bindings": bindings,
        "implementation_binding_registry_is_exact": True,
        "implementation_bytes_unchanged_since_runtime_qualification": True,
        "formal_implementation_tree_sha256": _stable_hash(bindings),
        "runtime_fingerprint_binding": fingerprint_binding,
        "production_canary_binding": canary_binding,
        "runtime_qualification_terminal_binding": qualification_terminal_binding,
        "runtime_and_canary_committed_and_clean": True,
        "runtime_qualification_terminal_committed_and_clean": True,
        "formal_source_opened": False,
        "formal_source_download_receipt_present": False,
        "formal_acquisition_root_present": False,
        "external_network_calls_by_freeze_builder": 0,
        "api_or_online_evaluator_calls_by_freeze_builder": 0,
        "retry_replay_resample_provider_switch": 0,
    }
    freeze = {**body, "self_sha256": _stable_hash(body)}
    _write_exclusive(project / output_name, freeze)
    return freeze


__all__ = [
    "ASSET_NAMES",
    "IMPLEMENTATION_FREEZE_SCHEMA",
    "REQUIRED_BINDING_PATHS",
    "RUNTIME_FINGERPRINT_SCHEMA",
    "TatqaP18ImplementationFreezeError",
    "VERSION",
    "build_implementation_freeze",
    "build_runtime_fingerprint",
]
