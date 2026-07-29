"""Append-only continuation of the missing WikiSQL v5 HippoRAG action arm.

The original repaired-v5 terminal, Agent action, RAW action, compiled
label-free view, and private labels remain immutable.  This controller:

1. verifies those frozen inputs and the qualified one-edit HippoRAG source;
2. partitions the label-free A_hold view into independent deterministic
   shards and runs only the original official HippoRAG worker;
3. seals one full-view HippoRAG action pack beside the byte-identical
   original Agent and RAW packs;
4. opens the frozen A_hold labels only after the three-arm barrier; and
5. invokes the original offline scorer exactly once.

This is explicitly a user-authorized post-terminal protocol exception in the
same v5 lineage.  It is not a rewrite or restart of the failed formal root.
The first recovery invocation also remains immutable: it reached no index,
action, barrier, label, or score before four workers blocked in the shared
311linux ZFS dependency read path.  This continuation binds that zero-effect
evidence, substitutes an exact-version dependency overlay on ext4, and runs
the same deterministic missing arm without changing any effect variable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


VERSION = "wikisql_uao_v5_hipporag_continuation_v1"
CONFIG_SCHEMA = f"{VERSION}_config_v1"
CONTINUATION_SCHEMA = f"{VERSION}_receipt_v1"
HIPPO_RECEIPT_SCHEMA = f"{VERSION}_hipporag_aggregate_receipt_v1"
BARRIER_SCHEMA = f"{VERSION}_three_arm_barrier_v1"
TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
FAILURE_SCHEMA = f"{VERSION}_safe_failure_terminal_v1"
STUDY_ID = "UAO_P4_WIKISQL_TYPED_RELATIONAL_ROW_EVIDENCE_V1"
PATCHED_SOURCE_SHA256 = (
    "6d0938da96757504e88ec15ea88f15bc6a6605e006eeb00c780598330b4c698b"
)
INPUT_SOURCE_SHA256 = (
    "960561b080531fe4d668bde635e81f8e65620ce50bdacdd9a25531e856fa3e05"
)
PATCH_SHA256 = (
    "a4a5584e0906d89eb09b59b4ee244d0a80b78a64cae9dbeafb50a923f7eddce5"
)
ITEM_COUNT = 72
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_INVOCATION = re.compile(r"[0-9a-f]{32}\Z")
_CONFIG_KEYS = frozenset(
    {
        "bindings",
        "continuation",
        "lane_assignments",
        "lineage",
        "recovery_root",
        "schema",
        "self_sha256",
        "shard_count",
        "study_id",
        "unit_name",
        "user_authorization",
    }
)


class RecoveryError(RuntimeError):
    """The append-only recovery contract failed closed."""


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise RecoveryError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value, newline=False)).hexdigest()


def self_hashed(value: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in value:
        raise RecoveryError("self hash already exists")
    return {**value, "self_sha256": semantic_sha256(value)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise RecoveryError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise RecoveryError(f"{field} is not a SHA-256 value")
    return value


def _absolute(value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise RecoveryError(f"{field} path drifted")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(str(path))):
        raise RecoveryError(f"{field} path is not normalized and absolute")
    return path


def _read_json(
    path: Path,
    *,
    field: str,
    expected_mode: int = 0o600,
) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"{field} is unreadable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != expected_mode
        or not isinstance(value, dict)
        or canonical_json_bytes(value) != raw
    ):
        raise RecoveryError(f"{field} metadata or canonical bytes drifted")
    return value


def _write_once(
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int = 0o600,
) -> str:
    raw = canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise RecoveryError(f"exclusive output already exists: {path}") from exc
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise RecoveryError("exclusive write stalled")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if file_sha256(path) != hashlib.sha256(raw).hexdigest():
        raise RecoveryError("exclusive output verification failed")
    return hashlib.sha256(raw).hexdigest()


def _mkdir(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise RecoveryError(f"fresh directory cannot be created: {path}") from exc


@dataclass(frozen=True, slots=True)
class FileBinding:
    path: Path
    sha256: str
    mode: int

    @classmethod
    def parse(cls, value: object, field: str) -> "FileBinding":
        if not isinstance(value, dict) or set(value) != {
            "mode_octal",
            "path",
            "sha256",
        }:
            raise RecoveryError(f"{field} shape drifted")
        mode = value["mode_octal"]
        if not isinstance(mode, str) or not re.fullmatch(r"0[0-7]{3}", mode):
            raise RecoveryError(f"{field} mode drifted")
        return cls(
            path=_absolute(value["path"], field),
            sha256=_hex64(value["sha256"], field),
            mode=int(mode, 8),
        )

    def verify(self, field: str) -> None:
        try:
            metadata = self.path.lstat()
        except OSError as exc:
            raise RecoveryError(f"{field} is unavailable") from exc
        if (
            self.path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != self.mode
            or file_sha256(self.path) != self.sha256
        ):
            raise RecoveryError(f"{field} binding drifted")


@dataclass(frozen=True, slots=True)
class TreeBinding:
    path: Path
    sha256: str
    file_count: int
    total_bytes: int

    @classmethod
    def parse(cls, value: object, field: str) -> "TreeBinding":
        if not isinstance(value, dict) or set(value) != {
            "file_count",
            "path",
            "sha256",
            "total_bytes",
        }:
            raise RecoveryError(f"{field} shape drifted")
        count = value["file_count"]
        size = value["total_bytes"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
        ):
            raise RecoveryError(f"{field} counts drifted")
        return cls(
            path=_absolute(value["path"], field),
            sha256=_hex64(value["sha256"], field),
            file_count=count,
            total_bytes=size,
        )

    def verify(self, field: str, identity_fn: Any) -> None:
        try:
            identity = identity_fn(self.path)
        except BaseException as exc:
            raise RecoveryError(f"{field} tree is unavailable") from exc
        if identity != (self.sha256, self.file_count, self.total_bytes):
            raise RecoveryError(f"{field} tree binding drifted")


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    path: Path
    recovery_root: Path
    unit_name: str
    shard_count: int
    lane_assignments: tuple[str, ...]
    files: Mapping[str, FileBinding]
    trees: Mapping[str, TreeBinding]
    self_sha256: str
    continuation: Mapping[str, object] | None = None


def load_config(path: Path) -> RecoveryConfig:
    value = _read_json(path, field="recovery config")
    if set(value) != _CONFIG_KEYS:
        raise RecoveryError("recovery config shape drifted")
    supplied = _hex64(value["self_sha256"], "config self")
    body = {key: child for key, child in value.items() if key != "self_sha256"}
    if semantic_sha256(body) != supplied:
        raise RecoveryError("recovery config self hash drifted")
    root = _absolute(value["recovery_root"], "recovery root")
    if path != root / "control/recovery_continuation_config.json":
        raise RecoveryError("recovery config path drifted")
    shard_count = value["shard_count"]
    lanes = value["lane_assignments"]
    if (
        value["schema"] != CONFIG_SCHEMA
        or value["study_id"] != STUDY_ID
        or value["lineage"] != "formal_v5_repair_r1"
        or not isinstance(value["unit_name"], str)
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count != 4
        or not isinstance(lanes, list)
        or tuple(lanes) != ("0", "0", "1", "1")
        or value["user_authorization"]
        != {
            "authorized": True,
            "no_v6": True,
            "reuse_agent_and_raw": True,
            "run_only_missing_patched_hipporag_then_offline_score": True,
        }
    ):
        raise RecoveryError("recovery identity or authorization drifted")
    bindings = value["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {"files", "trees"}:
        raise RecoveryError("binding registry drifted")
    raw_files = bindings["files"]
    raw_trees = bindings["trees"]
    required_files = {
        "agent_action",
        "agent_policy",
        "agent_receipt",
        "bound_worker",
        "ext4_overlay_receipt",
        "original_config",
        "original_model_alias_receipt",
        "original_terminal",
        "original_unit",
        "preindex_dstate_evidence",
        "prior_attempt",
        "prior_intent",
        "prior_recovery_config",
        "prior_recovery_unit",
        "raw_action",
        "recovery_runner",
        "recovery_unit",
        "source_labels",
        "source_view",
    }
    required_trees = {"ext4_overlay", "original_code", "patched_import"}
    if (
        not isinstance(raw_files, dict)
        or set(raw_files) != required_files
        or not isinstance(raw_trees, dict)
        or set(raw_trees) != required_trees
    ):
        raise RecoveryError("binding names drifted")
    files = {
        name: FileBinding.parse(raw_files[name], name)
        for name in sorted(required_files)
    }
    trees = {
        name: TreeBinding.parse(raw_trees[name], name)
        for name in sorted(required_trees)
    }
    if files["recovery_runner"].path.resolve() != Path(__file__).resolve():
        raise RecoveryError("executed recovery runner is not frozen")
    continuation = value["continuation"]
    if continuation != {
        "authorized_after_preindex_zero_effect": True,
        "effect_retry_or_resample_count": 0,
        "prior_invocation_id": "a1a4ba82c5974d6b93d58bfc78c92225",
        "prior_shard_launch_count": 4,
        "reuse_existing_attempt_and_intent": True,
        "runtime_substitution_only": (
            "exact_version_dependency_overlay_moved_from_ZFS_to_ext4"
        ),
    }:
        raise RecoveryError("continuation contract drifted")
    return RecoveryConfig(
        path=path,
        recovery_root=root,
        unit_name=value["unit_name"],
        shard_count=shard_count,
        lane_assignments=tuple(lanes),
        files=files,
        trees=trees,
        self_sha256=supplied,
        continuation=continuation,
    )


def _bootstrap_original(config: RecoveryConfig):
    config.files["original_config"].verify("original formal config")
    original_value = _read_json(
        config.files["original_config"].path,
        field="original formal config",
    )
    bindings = original_value.get("bindings")
    if not isinstance(bindings, dict) or not isinstance(
        bindings.get("trees"), dict
    ):
        raise RecoveryError("original formal config bindings drifted")
    original_trees = bindings["trees"]
    dependency_names = (
        "python_dependency_tree",
        "babel_dependency_tree",
        "official_base_dependency_tree",
    )
    dependency_roots: list[Path] = []
    for name in dependency_names:
        row = original_trees.get(name)
        if not isinstance(row, dict):
            raise RecoveryError(f"original dependency {name} drifted")
        dependency_roots.append(
            _absolute(row.get("path"), f"original dependency {name}")
        )
    code = config.trees["original_code"].path
    for path in reversed((code, *dependency_roots)):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from assumption_agent.benchmarks import (
        wikisql_uao_action_runtime_v1 as action_runtime,
    )
    from assumption_agent.benchmarks import wikisql_uao_scorer_v1 as scorer
    from replication_runtime.wikisql_uao_formal_v5_repair_r1 import (
        runner as formal,
    )
    from replication_runtime.wikisql_uao_official_v1 import contract
    from replication_runtime.wikisql_uao_runtime_qualification import (
        alias_runtime,
    )

    return action_runtime, scorer, formal, contract, alias_runtime


def _service_attestation(config: RecoveryConfig) -> dict[str, str]:
    runtime = Path(f"/run/user/{os.getuid()}")
    command = [
        "/usr/bin/systemctl",
        "--user",
        "show",
        config.unit_name,
        "--no-pager",
        "--property=ActiveState",
        "--property=FragmentPath",
        "--property=InvocationID",
        "--property=NRestarts",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env={
                "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime / 'bus'}",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "XDG_RUNTIME_DIR": str(runtime),
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RecoveryError("systemd attestation failed") from exc
    rows: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, child = line.partition("=")
        if separator != "=" or key in rows:
            raise RecoveryError("systemd attestation shape drifted")
        rows[key] = child
    if (
        set(rows) != {"ActiveState", "FragmentPath", "InvocationID", "NRestarts"}
        or rows["ActiveState"] != "activating"
        or rows["NRestarts"] != "0"
        or _INVOCATION.fullmatch(rows["InvocationID"]) is None
        or Path(rows["FragmentPath"]).resolve(strict=True)
        != config.files["recovery_unit"].path.resolve(strict=True)
    ):
        raise RecoveryError("systemd execution identity drifted")
    return rows


def _verify_static_bindings(config: RecoveryConfig, formal: Any) -> Any:
    for name, binding in config.files.items():
        if name == "source_labels":
            # The commitment is already frozen in the config, but the private
            # label bytes are not opened until after the action barrier.
            continue
        binding.verify(name)
    original_config = formal.load_config(config.files["original_config"].path)
    for name in (
        "official_python_executable",
        "python_executable",
    ):
        original_config.file(name).verify(name)
    patched = config.trees["patched_import"]
    patched_source = patched.path / "hipporag/HippoRAG.py"
    try:
        patched_metadata = patched_source.lstat()
    except OSError as exc:
        raise RecoveryError("qualified patched source is unavailable") from exc
    if (
        patched_source.is_symlink()
        or not stat.S_ISREG(patched_metadata.st_mode)
        or file_sha256(patched_source) != PATCHED_SOURCE_SHA256
    ):
        raise RecoveryError("qualified patched source drifted")
    _verify_original_model_aliases(
        config=config,
        original_config=original_config,
    )
    original_terminal = _read_json(
        config.files["original_terminal"].path,
        field="original terminal",
    )
    if (
        original_terminal.get("status")
        != "formal_failed_no_retry_efficacy_unknown"
        or original_terminal.get("stage")
        != "launch_three_actions_concurrently"
        or original_terminal.get("action_child_launch_count") != 3
        or original_terminal.get("action_barrier_count") != 0
        or original_terminal.get("a_hold_label_projection_count") != 0
        or original_terminal.get("scorer_launch_count") != 0
        or original_terminal.get("API_or_online_evaluation_count") != 0
    ):
        raise RecoveryError("original failure terminal is not the bound failure")
    config.trees["ext4_overlay"].verify(
        "exact-version ext4 dependency overlay",
        formal.tree_identity,
    )
    _verify_ext4_overlay_receipt(config)
    continuation_evidence = _verify_continuation_evidence(config)
    return original_config, continuation_evidence


def _verify_original_model_aliases(
    *,
    config: RecoveryConfig,
    original_config: Any,
) -> tuple[Path, Mapping[str, object]]:
    receipt_binding = config.files["original_model_alias_receipt"]
    receipt = _read_json(
        receipt_binding.path,
        field="original verified model alias receipt",
    )
    if (
        receipt.get("schema")
        != "wikisql_uao_short_model_alias_runtime_receipt_v1"
        or receipt.get("status") != "short_model_aliases_bound_and_verified"
        or receipt.get("derived_hipporag_component")
        != "Transformers_smollm2_Transformers_minilm"
        or receipt.get("derived_hipporag_component_utf8_bytes") != 40
        or not isinstance(receipt.get("self_sha256"), str)
    ):
        raise RecoveryError("original verified model alias receipt drifted")
    body = {key: child for key, child in receipt.items() if key != "self_sha256"}
    if semantic_sha256(body) != receipt["self_sha256"]:
        raise RecoveryError("original model alias receipt self hash drifted")
    alias_root = receipt_binding.path.parent / "model_aliases"
    try:
        metadata = alias_root.lstat()
    except OSError as exc:
        raise RecoveryError("original model alias root is unavailable") from exc
    if (
        alias_root.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RecoveryError("original model alias root drifted")
    expected = {
        "smollm2": original_config.tree("hippo_llm_model_tree").path,
        "minilm": original_config.tree("encoder_model_tree").path,
    }
    if {path.name for path in alias_root.iterdir()} != {
        *expected,
        "stderr.log",
        "stdout.log",
    }:
        raise RecoveryError("original model alias registry drifted")
    for alias, target in expected.items():
        path = alias_root / alias
        try:
            same = os.path.samefile(path, target)
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise RecoveryError("original model alias cannot be resolved") from exc
        if (
            not path.is_symlink()
            or os.readlink(path) != str(target)
            or resolved != target
            or same is not True
        ):
            raise RecoveryError("original model alias binding drifted")
    return alias_root, receipt


def _verify_self_hashed_json(
    binding: FileBinding,
    *,
    field: str,
) -> dict[str, object]:
    value = _read_json(binding.path, field=field, expected_mode=binding.mode)
    supplied = _hex64(value.get("self_sha256"), f"{field} self")
    body = {key: child for key, child in value.items() if key != "self_sha256"}
    if semantic_sha256(body) != supplied:
        raise RecoveryError(f"{field} self hash drifted")
    return value


def _verify_continuation_evidence(
    config: RecoveryConfig,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    attempt = _verify_self_hashed_json(
        config.files["prior_attempt"],
        field="prior recovery attempt",
    )
    intent = _verify_self_hashed_json(
        config.files["prior_intent"],
        field="prior recovery intent",
    )
    evidence = _verify_self_hashed_json(
        config.files["preindex_dstate_evidence"],
        field="preindex D-state evidence",
    )
    prior_invocation = "a1a4ba82c5974d6b93d58bfc78c92225"
    if (
        attempt.get("schema")
        != "wikisql_uao_v5_hipporag_recovery_v1_attempt_v1"
        or attempt.get("status") != "claimed_once"
        or attempt.get("config_file_sha256")
        != config.files["prior_recovery_config"].sha256
        or attempt.get("invocation_id_sha256")
        != hashlib.sha256(prior_invocation.encode("ascii")).hexdigest()
        or attempt.get("nrestarts") != 0
        or intent.get("schema")
        != "wikisql_uao_v5_hipporag_recovery_v1_intent_v1"
        or intent.get("status") != "missing_HippoRAG_only_intent_frozen"
        or intent.get("Agent_and_RAW_rerun_count") != 0
        or intent.get(
            "A_hold_label_open_count_before_three_arm_barrier"
        )
        != 0
        or evidence.get("schema")
        != (
            "wikisql_uao_v5_hipporag_recovery_v1_"
            "preindex_dstate_evidence_v1"
        )
        or evidence.get("status")
        != "PRESERVED_PREINDEX_INFRASTRUCTURE_INVALID_ZERO_EFFECT"
        or evidence.get("service_invocation_id") != prior_invocation
        or evidence.get("original_attempt_file_sha256")
        != config.files["prior_attempt"].sha256
        or evidence.get("original_attempt_self_sha256")
        != attempt["self_sha256"]
        or evidence.get("original_intent_file_sha256")
        != config.files["prior_intent"].sha256
        or evidence.get("original_intent_self_sha256")
        != intent["self_sha256"]
        or evidence.get("effect_measurement_consumed") is not False
        or evidence.get("shard_launch_count") != 4
        or evidence.get("completed_HippoRAG_shard_count") != 0
        or evidence.get("index_directory_count") != 0
        or evidence.get("action_file_count") != 0
        or evidence.get("worker_receipt_count") != 0
        or evidence.get("action_barrier_count") != 0
        or evidence.get("label_projection_count") != 0
        or evidence.get("scorer_launch_count") != 0
        or evidence.get("Agent_and_RAW_rerun_count") != 0
        or evidence.get("API_or_online_evaluation_count") != 0
    ):
        raise RecoveryError("prior zero-effect continuation evidence drifted")
    preserved = _absolute(
        evidence.get("preserved_work_path"),
        "preserved preindex work",
    )
    try:
        metadata = preserved.lstat()
        children = tuple(preserved.rglob("*"))
    except OSError as exc:
        raise RecoveryError("preserved preindex work is unavailable") from exc
    if (
        preserved.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or any(path.name == "indexes" and path.is_dir() for path in children)
        or any(
            path.is_file()
            and (
                path.name.endswith(".actions.json")
                or path.name.endswith(".safe.json")
            )
            for path in children
        )
    ):
        raise RecoveryError("preserved preindex work gained effect outputs")
    return attempt, intent, evidence


def _verify_ext4_overlay_receipt(
    config: RecoveryConfig,
) -> Mapping[str, object]:
    receipt = _verify_self_hashed_json(
        config.files["ext4_overlay_receipt"],
        field="exact-version ext4 overlay receipt",
    )
    tree = config.trees["ext4_overlay"]
    required_versions = {
        "accelerate": "1.8.1",
        "networkx": "3.3",
        "numpy": "2.1.3",
        "pandas": "2.2.3",
        "pyarrow": "17.0.0",
        "safetensors": "0.4.5",
        "scikit-learn": "1.5.2",
        "scipy": "1.14.1",
        "sentence-transformers": "3.1.1",
        "tokenizers": "0.20.3",
        "torch": "2.4.1+cu118",
        "transformers": "4.45.2",
    }
    prior_attested_origin_hashes = {
        "networkx": (
            "8152d75a7e98997ebc0a5f66986a6a3d92ed88821552a1a6532ceda917e1af2e"
        ),
        "numpy": (
            "39c42db027548f958e096e8babe3fa0e3e773d24aa39eb6363fc0e3abbec34b1"
        ),
        "sentence-transformers": (
            "73bd39dc1269cd422ec9969ee6a7df45ac524cf49ce12fdcea51b1ac77b1bf8f"
        ),
        "torch": (
            "34fd26c3046775a70a6a654df60afb362b2f2e98b2f9b8e713c8763b7a80ff83"
        ),
        "transformers": (
            "e31022d9850a13c409d5b0cf901ba1934030be0190f790df3d5cc28c564d9ed4"
        ),
    }
    if (
        receipt.get("schema")
        != "wikisql_uao_p4_v5_ext4_overlay_receipt_v1"
        or receipt.get("status")
        != "EXACT_VERSION_OVERLAY_VERIFIED_ON_EXT4"
        or receipt.get("overlay_root") != str(tree.path)
        or receipt.get("tree_sha256") != tree.sha256
        or receipt.get("tree_file_count") != tree.file_count
        or receipt.get("tree_total_bytes") != tree.total_bytes
        or receipt.get("required_versions") != required_versions
        or receipt.get("prior_attested_origin_hashes")
        != prior_attested_origin_hashes
        or receipt.get("origin_hashes_match_prior_attestation") is not True
        or receipt.get("local_import_probe_passed") is not True
        or receipt.get("remote_import_probe_passed") is not True
        or receipt.get("source_runtime_substitution_only") is not True
    ):
        raise RecoveryError("exact-version ext4 overlay receipt drifted")
    return receipt


def _view_item_payload(item: Any) -> dict[str, object]:
    return {
        "opaque_item_id": item.item_id,
        "physical_rows": [list(row) for row in item.rows],
        "question": item.question,
        "table_header": list(item.header),
        "table_types": list(item.types),
    }


def _make_shards(
    *,
    config: RecoveryConfig,
    action_runtime: Any,
    work: Path,
) -> tuple[dict[str, object], tuple[Any, ...], list[Path]]:
    full_view = _read_json(
        config.files["source_view"].path,
        field="original A_hold action view",
    )
    full_items = action_runtime.decode_view_pack(
        full_view,
        expected_block="A_hold",
        expected_count=ITEM_COUNT,
    )
    size = ITEM_COUNT // config.shard_count
    paths: list[Path] = []
    for shard in range(config.shard_count):
        start = shard * size
        stop = start + size
        shard_value = action_runtime.build_view_pack(
            block="A_hold",
            items=[
                _view_item_payload(item) for item in full_items[start:stop]
            ],
        )
        shard_root = work / "hippo" / f"shard-{shard:02d}"
        _mkdir(shard_root)
        for child in ("home", "tmp"):
            _mkdir(shard_root / child)
        path = shard_root / "A_hold.action_views.json"
        _write_once(path, shard_value)
        paths.append(path)
    return full_view, full_items, paths


def _open_log(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
    except OSError as exc:
        raise RecoveryError("worker log cannot be created") from exc
    return descriptor


def _launch_shards(
    *,
    config: RecoveryConfig,
    action_runtime: Any,
    original_config: Any,
    formal: Any,
    alias_runtime: Any,
    shard_views: Sequence[Path],
    work: Path,
) -> tuple[list[dict[str, object]], Mapping[str, object]]:
    alias_root, alias_receipt = _verify_original_model_aliases(
        config=config,
        original_config=original_config,
    )
    patched_import = config.trees["patched_import"].path
    code = config.trees["original_code"].path
    ext4_overlay = config.trees["ext4_overlay"].path
    python = original_config.file("official_python_executable").path
    all_devices = formal._all_gpu_device_paths()
    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    for shard, (view, gpu) in enumerate(
        zip(shard_views, config.lane_assignments, strict=True)
    ):
        shard_root = view.parent
        action = shard_root / "A_hold.HippoRAG.actions.json"
        receipt = shard_root / "hipporag.safe.json"
        environment = formal._lane_environment(
            original_config,
            shard_root,
            cuda_visible_devices=gpu,
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(patched_import),
                str(code),
                str(
                    original_config.tree(
                        "official_python_dependency_tree"
                    ).path
                ),
                str(original_config.tree("babel_dependency_tree").path),
                str(original_config.tree("official_hipporag_tree").path),
                str(ext4_overlay),
                str(
                    original_config.tree(
                        "official_base_dependency_tree"
                    ).path
                ),
            )
        )
        environment["PYTHONPYCACHEPREFIX"] = str(
            shard_root / "tmp/pycache"
        )
        argv = (
            str(python),
            "-S",
            "-B",
            "-s",
            str(config.files["bound_worker"].path),
            "--input",
            str(view),
            "--action-output",
            str(action),
            "--safe-receipt-output",
            str(receipt),
            "--index-parent",
            str(shard_root / "indexes"),
            "--llm-model",
            alias_runtime.LLM_ALIAS,
            "--embedding-model",
            alias_runtime.EMBEDDING_ALIAS,
        )
        read_paths = (
            *formal._existing_system_read_paths(),
            python,
            original_config.tree("python_runtime_tree").path,
            code,
            config.files["bound_worker"].path,
            patched_import,
            original_config.tree("official_python_dependency_tree").path,
            original_config.tree("babel_dependency_tree").path,
            original_config.tree("official_hipporag_tree").path,
            ext4_overlay,
            original_config.tree("official_base_dependency_tree").path,
            original_config.tree("encoder_model_tree").path,
            original_config.tree("hippo_llm_model_tree").path,
            alias_root,
            view,
        )

        def isolate(
            reads: tuple[Path, ...] = tuple(read_paths),
            root: Path = shard_root,
        ) -> None:
            formal.apply_landlock(
                read_paths=reads,
                write_paths=(root, Path("/proc/self/task")),
                device_paths=all_devices,
            )

        stdout = _open_log(shard_root / "stdout.log")
        stderr = _open_log(shard_root / "stderr.log")
        try:
            process = subprocess.Popen(
                argv,
                cwd=alias_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                preexec_fn=isolate,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RecoveryError(f"HippoRAG shard {shard} launch failed") from exc
        finally:
            os.close(stdout)
            os.close(stderr)
        processes.append((shard, process))
    statuses = {shard: process.wait() for shard, process in processes}
    if statuses != {0: 0, 1: 0, 2: 0, 3: 0}:
        raise RecoveryError(
            "one or more patched HippoRAG shards failed: "
            + semantic_sha256(statuses)
        )
    shard_rows: list[dict[str, object]] = []
    shard_commitments: list[dict[str, object]] = []
    for shard, view in enumerate(shard_views):
        shard_root = view.parent
        action_path = shard_root / "A_hold.HippoRAG.actions.json"
        receipt_path = shard_root / "hipporag.safe.json"
        action_value = _read_json(
            action_path,
            field=f"HippoRAG shard {shard} action",
        )
        receipt_value = _read_json(
            receipt_path,
            field=f"HippoRAG shard {shard} receipt",
        )
        view_value = _read_json(view, field=f"HippoRAG shard {shard} view")
        rows = action_runtime.decode_action_pack(
            action_value,
            expected_block="A_hold",
            expected_arm="HippoRAG",
            expected_action_view_pack_sha256=view_value["self_sha256"],
        )
        shard_rows.extend(dict(row) for row in rows)
        shard_commitments.append(
            {
                "action_file_sha256": file_sha256(action_path),
                "action_pack_self_sha256": action_value["self_sha256"],
                "gpu_lane": config.lane_assignments[shard],
                "item_count": len(rows),
                "receipt_file_sha256": file_sha256(receipt_path),
                "receipt_self_sha256": receipt_value["self_sha256"],
                "shard": shard,
                "view_file_sha256": file_sha256(view),
                "view_self_sha256": view_value["self_sha256"],
            }
        )
    return shard_rows, {
        "alias_receipt_file_sha256": config.files[
            "original_model_alias_receipt"
        ].sha256,
        "alias_receipt_self_sha256": alias_receipt["self_sha256"],
        "shards": shard_commitments,
    }


def _seal_hippo(
    *,
    config: RecoveryConfig,
    action_runtime: Any,
    full_view: Mapping[str, object],
    full_items: Sequence[Any],
    shard_rows: Sequence[Mapping[str, object]],
    shard_receipts: Mapping[str, object],
    work: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    action = action_runtime.build_action_pack(
        block="A_hold",
        arm="HippoRAG",
        action_view_pack_sha256=full_view["self_sha256"],
        items=shard_rows,
    )
    decoded = action_runtime.decode_action_pack(
        action,
        expected_block="A_hold",
        expected_arm="HippoRAG",
        expected_action_view_pack_sha256=full_view["self_sha256"],
    )
    if (
        len(decoded) != ITEM_COUNT
        or tuple(row["opaque_item_id"] for row in decoded)
        != tuple(item.item_id for item in full_items)
    ):
        raise RecoveryError("aggregated HippoRAG item coverage drifted")
    action_path = work / "hippo/A_hold.HippoRAG.actions.json"
    _write_once(action_path, action)
    receipt = self_hashed(
        {
            "API_or_online_evaluation_count": 0,
            "action_file_sha256": file_sha256(action_path),
            "action_pack_self_sha256": action["self_sha256"],
            "action_view_pack_self_sha256": full_view["self_sha256"],
            "arm": "HippoRAG",
            "input_source_sha256": INPUT_SOURCE_SHA256,
            "item_count": ITEM_COUNT,
            "lane_assignments": list(config.lane_assignments),
            "official_worker_preserved_per_item_fresh_index": True,
            "parallel_shard_count": config.shard_count,
            "patched_source_sha256": PATCHED_SOURCE_SHA256,
            "protocol_exception": (
                "user_authorized_post_terminal_same_v5_missing_arm_recovery"
            ),
            "effect_retry_or_resample_count": 0,
            "infrastructure_continuation_count": 1,
            "schema": HIPPO_RECEIPT_SCHEMA,
            "shard_commitments": shard_receipts,
            "status": "patched_HippoRAG_action_pack_sealed",
            "study_id": STUDY_ID,
            "unified_patch_sha256": PATCH_SHA256,
        }
    )
    _write_once(work / "hippo/hipporag.safe.json", receipt)
    return action, receipt


def _validate_existing_arms(
    *,
    config: RecoveryConfig,
    action_runtime: Any,
    view_sha256: str,
    hippo: Mapping[str, object],
) -> dict[str, object]:
    packs: dict[str, Mapping[str, object]] = {
        "Agent": _read_json(
            config.files["agent_action"].path,
            field="original Agent action",
        ),
        "RAW": _read_json(
            config.files["raw_action"].path,
            field="original RAW action",
        ),
        "HippoRAG": hippo,
    }
    identifiers: dict[str, tuple[str, ...]] = {}
    result: dict[str, object] = {}
    for arm, value in packs.items():
        rows = action_runtime.decode_action_pack(
            value,
            expected_block="A_hold",
            expected_arm=arm,
            expected_action_view_pack_sha256=view_sha256,
        )
        if len(rows) != ITEM_COUNT:
            raise RecoveryError(f"{arm} action count drifted")
        identifiers[arm] = tuple(row["opaque_item_id"] for row in rows)
        path = (
            config.files["agent_action"].path
            if arm == "Agent"
            else config.files["raw_action"].path
            if arm == "RAW"
            else config.recovery_root / "work/hippo/A_hold.HippoRAG.actions.json"
        )
        result[arm] = {
            "file_sha256": file_sha256(path),
            "pack_self_sha256": value["self_sha256"],
        }
    if not (
        identifiers["Agent"] == identifiers["RAW"] == identifiers["HippoRAG"]
    ):
        raise RecoveryError("three action packs do not share item IDs")
    result["item_id_set_sha256"] = semantic_sha256(list(identifiers["Agent"]))
    return result


def _project_labels(
    *,
    config: RecoveryConfig,
    scorer: Any,
    view_sha256: str,
    output: Path,
) -> str:
    config.files["source_labels"].verify("post-barrier source labels")
    source = _read_json(
        config.files["source_labels"].path,
        field="post-barrier source labels",
    )
    supplied = _hex64(source.get("self_sha256"), "source label self")
    body = {key: child for key, child in source.items() if key != "self_sha256"}
    if (
        semantic_sha256(body) != supplied
        or source.get("study_id") != STUDY_ID
        or source.get("block") != "A_hold"
        or source.get("item_count") != ITEM_COUNT
        or source.get("release_policy")
        != "after_all_A_hold_three_arm_actions_are_sealed"
        or not isinstance(source.get("items"), list)
    ):
        raise RecoveryError("source label pack drifted")
    projected = scorer.build_minimal_label_pack(
        action_view_pack_sha256=view_sha256,
        items=source["items"],
    )
    return _write_once(output, projected)


def _run_scorer(
    *,
    config: RecoveryConfig,
    original_config: Any,
    formal: Any,
    scorer_root: Path,
    barrier: Path,
) -> int:
    python = original_config.file("python_executable").path
    environment = formal._lane_environment(
        original_config,
        scorer_root,
        cuda_visible_devices="",
    )
    environment["PYTHONPYCACHEPREFIX"] = str(
        scorer_root / "tmp/pycache"
    )
    argv = (
        str(python),
        "-S",
        "-B",
        "-s",
        "-m",
        "assumption_agent.benchmarks.wikisql_uao_scorer_v1",
        "--action-view-pack",
        str(config.files["source_view"].path),
        "--minimal-label-pack",
        str(scorer_root / "A_hold.minimal.labels.json"),
        "--agent-action-pack",
        str(config.files["agent_action"].path),
        "--raw-action-pack",
        str(config.files["raw_action"].path),
        "--hipporag-action-pack",
        str(config.recovery_root / "work/hippo/A_hold.HippoRAG.actions.json"),
        "--private-score-output",
        str(scorer_root / "three_arm.private.json"),
        "--safe-receipt-output",
        str(scorer_root / "score.safe.json"),
        "--terminal-output",
        str(scorer_root / "scorer_terminal.safe.json"),
    )
    read_paths = (
        *formal._existing_system_read_paths(),
        python,
        original_config.tree("python_runtime_tree").path,
        original_config.tree("python_dependency_tree").path,
        original_config.tree("babel_dependency_tree").path,
        config.trees["original_code"].path,
        config.files["source_view"].path,
        scorer_root / "A_hold.minimal.labels.json",
        config.files["agent_action"].path,
        config.files["raw_action"].path,
        config.recovery_root / "work/hippo/A_hold.HippoRAG.actions.json",
        barrier,
    )

    def isolate() -> None:
        formal.apply_landlock(
            read_paths=read_paths,
            write_paths=(scorer_root,),
        )

    stdout = _open_log(scorer_root / "stdout.log")
    stderr = _open_log(scorer_root / "stderr.log")
    try:
        process = subprocess.Popen(
            argv,
            cwd=scorer_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            preexec_fn=isolate,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RecoveryError("offline scorer launch failed") from exc
    finally:
        os.close(stdout)
        os.close(stderr)
    return process.wait()


@dataclass(slots=True)
class State:
    stage: str = "load_config"
    prior_shard_launch_count: int = 4
    shard_launch_count: int = 0
    action_barrier_count: int = 0
    label_projection_count: int = 0
    scorer_launch_count: int = 0


def run(config_path: Path) -> Mapping[str, object]:
    config = load_config(config_path)
    root = config.recovery_root
    control = root / "control"
    terminal_path = control / "outer_terminal.safe.json"
    state = State()
    if terminal_path.exists() or terminal_path.is_symlink():
        raise RecoveryError("recovery terminal already exists")
    try:
        action_runtime, scorer, formal, _contract, alias_runtime = (
            _bootstrap_original(config)
        )
        state.stage = "attest_unique_user_service"
        service = _service_attestation(config)
        state.stage = "verify_frozen_inputs_and_qualified_patch"
        original_config, continuation_evidence = _verify_static_bindings(
            config,
            formal,
        )
        prior_attempt, prior_intent, preindex_evidence = (
            continuation_evidence
        )
        continuation = self_hashed(
            {
                "API_or_online_evaluation_count": 0,
                "Agent_and_RAW_rerun_count": 0,
                "A_hold_label_open_count_before_three_arm_barrier": 0,
                "config_file_sha256": file_sha256(config.path),
                "config_self_sha256": config.self_sha256,
                "effect_retry_or_resample_count": 0,
                "invocation_id_sha256": hashlib.sha256(
                    service["InvocationID"].encode("ascii")
                ).hexdigest(),
                "nrestarts": 0,
                "preindex_evidence_file_sha256": config.files[
                    "preindex_dstate_evidence"
                ].sha256,
                "preindex_evidence_self_sha256": preindex_evidence[
                    "self_sha256"
                ],
                "prior_attempt_file_sha256": config.files[
                    "prior_attempt"
                ].sha256,
                "prior_attempt_self_sha256": prior_attempt["self_sha256"],
                "prior_intent_file_sha256": config.files[
                    "prior_intent"
                ].sha256,
                "prior_intent_self_sha256": prior_intent["self_sha256"],
                "prior_invocation_id": preindex_evidence[
                    "service_invocation_id"
                ],
                "prior_preindex_shard_launch_count": 4,
                "protocol_exception": (
                    "user_authorized_same_v5_zero_effect_"
                    "infrastructure_continuation"
                ),
                "runtime_substitution_only": (
                    "exact_version_dependency_overlay_moved_from_ZFS_to_ext4"
                ),
                "schema": CONTINUATION_SCHEMA,
                "status": "continuation_claimed_once",
                "study_id": STUDY_ID,
            }
        )
        continuation_file = _write_once(
            control / "recovery_continuation.safe.json",
            continuation,
        )
        state.stage = "prepare_four_label_free_shards"
        work = root / "work"
        _mkdir(work)
        _mkdir(work / "hippo")
        full_view, full_items, shard_views = _make_shards(
            config=config,
            action_runtime=action_runtime,
            work=work,
        )
        intent = prior_intent
        intent_file = config.files["prior_intent"].sha256
        state.stage = "launch_four_patched_HippoRAG_shards"
        state.shard_launch_count = config.shard_count
        shard_rows, shard_receipts = _launch_shards(
            config=config,
            action_runtime=action_runtime,
            original_config=original_config,
            formal=formal,
            alias_runtime=alias_runtime,
            shard_views=shard_views,
            work=work,
        )
        state.stage = "seal_full_HippoRAG_action_pack"
        hippo, hippo_receipt = _seal_hippo(
            config=config,
            action_runtime=action_runtime,
            full_view=full_view,
            full_items=full_items,
            shard_rows=shard_rows,
            shard_receipts=shard_receipts,
            work=work,
        )
        state.stage = "validate_and_seal_three_arm_barrier"
        actions = _validate_existing_arms(
            config=config,
            action_runtime=action_runtime,
            view_sha256=full_view["self_sha256"],
            hippo=hippo,
        )
        barrier = self_hashed(
            {
                "A_hold_label_open_count_before_barrier": 0,
                "Agent_and_RAW_reused_byte_exact": True,
                "all_three_actions_durable": True,
                "action_commitments": actions,
                "hipporag_receipt_self_sha256": hippo_receipt["self_sha256"],
                "continuation_file_sha256": continuation_file,
                "continuation_self_sha256": continuation["self_sha256"],
                "intent_file_sha256": intent_file,
                "intent_self_sha256": intent["self_sha256"],
                "schema": BARRIER_SCHEMA,
                "status": "three_common_action_packs_sealed",
                "study_id": STUDY_ID,
            }
        )
        barrier_path = control / "action_barrier.safe.json"
        barrier_file = _write_once(barrier_path, barrier)
        state.action_barrier_count = 1
        state.stage = "post_barrier_project_minimal_labels"
        scorer_root = work / "scorer"
        _mkdir(scorer_root)
        for child in ("home", "tmp"):
            _mkdir(scorer_root / child)
        label_file = _project_labels(
            config=config,
            scorer=scorer,
            view_sha256=full_view["self_sha256"],
            output=scorer_root / "A_hold.minimal.labels.json",
        )
        state.label_projection_count = 1
        state.stage = "run_original_offline_scorer_once"
        state.scorer_launch_count = 1
        if (
            _run_scorer(
                config=config,
                original_config=original_config,
                formal=formal,
                scorer_root=scorer_root,
                barrier=barrier_path,
            )
            != 0
        ):
            raise RecoveryError("original offline scorer failed")
        state.stage = "validate_safe_aggregate_and_write_terminal"
        scorer_paths = formal.FormalPaths.for_root(root)
        scorer_artifacts = formal._verify_scorer_outputs(scorer_paths)
        terminal = self_hashed(
            {
                "API_or_online_evaluation_count": 0,
                "Agent_and_RAW_rerun_count": 0,
                "a_hold_label_opened_only_after_action_barrier": True,
                "a_hold_minimal_label_file_sha256": label_file,
                "action_barrier_file_sha256": barrier_file,
                "action_barrier_self_sha256": barrier["self_sha256"],
                "config_self_sha256": config.self_sha256,
                "nrestarts": 0,
                "effect_retry_or_resample_count": 0,
                "infrastructure_continuation_count": 1,
                "original_failed_terminal_preserved": True,
                "original_terminal_file_sha256": config.files[
                    "original_terminal"
                ].sha256,
                "parallel_HippoRAG_shard_count": config.shard_count,
                "prior_preindex_HippoRAG_shard_launch_count": 4,
                "total_HippoRAG_process_launch_count": 8,
                "patched_source_sha256": PATCHED_SOURCE_SHA256,
                "primary_passed": scorer_artifacts.terminal[
                    "primary_passed"
                ],
                "protocol_exception": (
                    "user_authorized_post_terminal_same_v5_missing_arm_recovery"
                ),
                "schema": TERMINAL_SCHEMA,
                "scorer_safe_aggregate_file_sha256": (
                    scorer_artifacts.safe_receipt_file_sha256
                ),
                "scorer_safe_terminal_file_sha256": (
                    scorer_artifacts.terminal_file_sha256
                ),
                "scorer_safe_terminal_self_sha256": (
                    scorer_artifacts.terminal["self_sha256"]
                ),
                "status": "completed_post_terminal_protocol_exception",
                "study_id": STUDY_ID,
            }
        )
        _write_once(terminal_path, terminal)
        return terminal
    except BaseException as exc:
        if not terminal_path.exists() and not terminal_path.is_symlink():
            failure = self_hashed(
                {
                    "API_or_online_evaluation_count": 0,
                    "Agent_and_RAW_rerun_count": 0,
                    "action_barrier_count": state.action_barrier_count,
                    "failure_fingerprint_sha256": hashlib.sha256(
                        f"{type(exc).__name__}:{exc}".encode(
                            "utf-8", errors="replace"
                        )
                    ).hexdigest(),
                    "label_projection_count": state.label_projection_count,
                    "protocol_exception": True,
                    "schema": FAILURE_SCHEMA,
                    "scorer_launch_count": state.scorer_launch_count,
                    "prior_preindex_shard_launch_count": (
                        state.prior_shard_launch_count
                    ),
                    "shard_launch_count": state.shard_launch_count,
                    "total_shard_process_launch_count": (
                        state.prior_shard_launch_count
                        + state.shard_launch_count
                    ),
                    "stage": state.stage,
                    "status": "failed_post_terminal_recovery_efficacy_unknown",
                    "study_id": STUDY_ID,
                }
            )
            _write_once(terminal_path, failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    terminal = run(arguments.config)
    print(
        json.dumps(
            terminal,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by remote service.
    raise SystemExit(main())
