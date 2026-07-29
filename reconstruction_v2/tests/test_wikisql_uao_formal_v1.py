from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping

import pytest

from assumption_agent.benchmarks import (
    wikisql_uao_action_runtime_v1 as action_runtime,
)
from assumption_agent.benchmarks import (
    wikisql_uao_reality_v1 as reality,
)
from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v1 as source_compiler,
)
from replication_runtime.wikisql_uao_formal_v1 import runner


GPU_UUIDS = {
    "0": "GPU-00000000-0000-0000-0000-000000000000",
    "1": "GPU-11111111-1111-1111-1111-111111111111",
}


@pytest.fixture()
def posix_root() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix="wikisql-uao-formal-test-",
        dir="/tmp",
    ) as raw:
        yield Path(raw)


def _raw(value: object) -> bytes:
    return runner.canonical_json_bytes(value)


def _addressed(value: Mapping[str, object]) -> dict[str, object]:
    return {**value, "self_sha256": runner.semantic_sha256(value)}


def _write(path: Path, raw: bytes, mode: int = 0o600) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _file_binding(path: Path) -> dict[str, object]:
    metadata = path.lstat()
    raw = path.read_bytes()
    return {
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _tree_binding(path: Path) -> dict[str, object]:
    digest, file_count, total_bytes = runner.tree_identity(path)
    return {
        "file_count": file_count,
        "path": str(path),
        "sha256": digest,
        "total_bytes": total_bytes,
    }


def _view_item(block: str, ordinal: int) -> dict[str, object]:
    item_id = hashlib.sha256(f"{block}-{ordinal}".encode()).hexdigest()
    return {
        "opaque_item_id": item_id,
        "physical_rows": [
            [f"name-{row}", row] for row in range(11)
        ],
        "question": f"Which row is relevant for {block} item {ordinal}?",
        "table_header": ["Name", "Score"],
        "table_types": ["text", "real"],
    }


def _source_pack_payloads() -> dict[str, Mapping[str, object] | bytes]:
    form_items = [_view_item("A_form", index) for index in range(192)]
    hold_items = [_view_item("A_hold", index) for index in range(72)]
    form_view = action_runtime.build_view_pack(
        block="A_form", items=form_items
    )
    hold_view = action_runtime.build_view_pack(
        block="A_hold", items=hold_items
    )
    sorted_form = form_view["items"]
    sorted_hold = hold_view["items"]
    assert isinstance(sorted_form, list)
    assert isinstance(sorted_hold, list)

    form_labels: list[dict[str, object]] = []
    for family_index, family in enumerate(reality.FAMILY_ORDER):
        family_rows = sorted_form[
            family_index * 64 : (family_index + 1) * 64
        ]
        for rank, item in enumerate(family_rows):
            form_labels.append(
                {
                    "action_view_sha256": reality.canonical_sha256(item),
                    "family": family,
                    "fold_index": rank % 4,
                    "gold_row_ids": [0],
                    "item_commitment_sha256": hashlib.sha256(
                        (
                            "form-source-"
                            + str(item["opaque_item_id"])
                        ).encode()
                    ).hexdigest(),
                    "opaque_item_id": item["opaque_item_id"],
                    "sqlite_rowid_cross_checked": True,
                    "table_row_count": 11,
                }
            )
    form_label_pack = action_runtime.build_label_pack(
        block="A_form", items=form_labels
    )

    hold_labels: list[dict[str, object]] = []
    for family_index, family in enumerate(reality.FAMILY_ORDER):
        family_rows = sorted_hold[
            family_index * 24 : (family_index + 1) * 24
        ]
        for item in family_rows:
            hold_labels.append(
                {
                    "action_view_sha256": reality.canonical_sha256(item),
                    "family": family,
                    "gold_row_ids": [0],
                    "item_commitment_sha256": hashlib.sha256(
                        (
                            "hold-source-"
                            + str(item["opaque_item_id"])
                        ).encode()
                    ).hexdigest(),
                    "opaque_item_id": item["opaque_item_id"],
                    "sqlite_rowid_cross_checked": True,
                    "table_row_count": 11,
                }
            )
    hold_labels.sort(key=lambda row: str(row["opaque_item_id"]))
    hold_label_pack = _addressed(
        {
            "block": "A_hold",
            "item_count": 72,
            "items": hold_labels,
            "release_policy": (
                "after_all_A_hold_three_arm_actions_are_sealed"
            ),
            "schema": (
                f"{source_compiler.VERSION}_private_label_pack_v1"
            ),
            "study_id": runner.STUDY_ID,
        }
    )
    provenance = _addressed(
        {
            "access_policy": "controller_only_never_Agent_or_scorer_input",
            "item_count": 264,
            "items": [],
            "schema": (
                f"{source_compiler.VERSION}_controller_only_provenance_pack_v1"
            ),
            "study_id": runner.STUDY_ID,
        }
    )
    return {
        "private/selection_secret.bin": b"s" * 32,
        "private/A_form.action_views.json": form_view,
        "private/A_form.labels.json": form_label_pack,
        "private/A_hold.action_views.json": hold_view,
        "private/A_hold.labels.json": hold_label_pack,
        "private/controller_only.provenance.json": provenance,
    }


def _common_action_pack(
    view_pack: Mapping[str, object],
    arm: str,
) -> Mapping[str, object]:
    items = view_pack["items"]
    assert isinstance(items, list)
    return action_runtime.build_action_pack(
        block="A_hold",
        arm=arm,
        action_view_pack_sha256=str(view_pack["self_sha256"]),
        items=[
            {
                "opaque_item_id": item["opaque_item_id"],
                "top5_row_ids": [0, 1, 2, 3, 4],
            }
            for item in items
        ],
    )


def _service_text(root: Path) -> str:
    return f"""[Unit]
Description=WikiSQL formal synthetic test

[Service]
Type=oneshot
WorkingDirectory={root}/reconstruction_v2
UMask=0077
ExecStart=/usr/bin/env -i PYTHONPATH={root}/reconstruction_v2:{root}/assets/python_dependency_tree:{root}/assets/babel_dependency_tree {root}/assets/python_executable -m replication_runtime.wikisql_uao_formal_v1.runner --config {root}/control/formal_config.json
CPUQuota=700%
MemoryMax=42949672960
TasksMax=128
KillMode=control-group
Restart=no
RestrictAddressFamilies=AF_UNIX
IPAddressDeny=any
NoNewPrivileges=yes
PrivateTmp=yes
TimeoutStartSec=infinity
"""


def _effective_exec_start(root: Path) -> str:
    exec_line = next(
        line
        for line in _service_text(root).splitlines()
        if line.startswith("ExecStart=")
    )
    argv = exec_line.removeprefix("ExecStart=")
    return (
        "{ path=/usr/bin/env ; argv[]="
        + argv
        + " ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; "
        "pid=0 ; code=(null) ; status=0/0 }"
    )


def _action_script(
    *,
    label_path: Path,
    action_packs: Mapping[str, Mapping[str, object]],
) -> str:
    encoded = {
        arm: base64.b64encode(_raw(pack)).decode("ascii")
        for arm, pack in action_packs.items()
    }
    return f"""import base64, os, pathlib, sys, time
LABEL = pathlib.Path({str(label_path)!r})
PACKS = {encoded!r}

def write(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)

arm, output, coord, mode, *extra = sys.argv[1:]
coord = pathlib.Path(coord)
write(coord / (arm + ".started"), b"started\\n")
deadline = time.monotonic() + 10
while len(list(coord.glob("*.started"))) != 3:
    if time.monotonic() > deadline:
        raise SystemExit(91)
    time.sleep(0.01)
if arm == "Agent":
    try:
        LABEL.read_bytes()
        result = b"ALLOWED\\n"
    except PermissionError:
        result = b"DENIED\\n"
    write(pathlib.Path(output).parent / "label_probe.txt", result)
if mode == "exit":
    raise SystemExit(7)
raw = b"{{}}\\n" if mode == "tamper" else base64.b64decode(PACKS[arm])
write(pathlib.Path(output), raw)
for path in extra:
    write(pathlib.Path(path), b"{{}}\\n")
"""


def _scorer_script() -> str:
    return """import hashlib, json, os, pathlib, sys

def canonical(value, newline=True):
    raw = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
    return raw + (b"\\n" if newline else b"")

def addressed(value):
    return {**value, "self_sha256": hashlib.sha256(canonical(value, False)).hexdigest()}

def write(path, value):
    raw = canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(raw).hexdigest()

barrier, labels, private_path, safe_path, terminal_path = map(pathlib.Path, sys.argv[1:])
observation = {"barrier_seen": barrier.exists(), "labels_read": bool(labels.read_bytes())}
write(pathlib.Path(private_path).parent / "scorer_observation.json", observation)
private = addressed({"schema": "synthetic_private_score", "study_id": "UAO_P4_WIKISQL_TYPED_RELATIONAL_ROW_EVIDENCE_V1"})
private_file = write(private_path, private)
comparison_raw = {
    "baseline": "raw",
    "exact_p_denominator": 16,
    "exact_p_numerator": 1,
    "family_net_u": {"EQ": 1, "GT": 1, "LT": 1},
    "nonzero_pair_count": 4,
    "observed_net_u": 3,
    "passed": True,
}
comparison_hippo = dict(comparison_raw)
comparison_hippo["baseline"] = "hipporag"
safe = addressed({
    "Agent_vs_HippoRAG": comparison_hippo,
    "Agent_vs_RAW": comparison_raw,
    "alpha_denominator": 10,
    "alpha_numerator": 1,
    "block": "A_hold",
    "family_counts": {"EQ": 24, "GT": 24, "LT": 24},
    "input_commitments": {
        "Agent_action_pack_sha256": "a" * 64,
        "HippoRAG_action_pack_sha256": "b" * 64,
        "RAW_action_pack_sha256": "c" * 64,
        "action_view_pack_sha256": "d" * 64,
        "minimal_label_pack_sha256": "e" * 64,
    },
    "item_count": 72,
    "offline_aggregate_primary_call_count": 1,
    "online_evaluation_count": 0,
    "primary_passed": True,
    "private_score_pack_sha256": private["self_sha256"],
    "schema": "wikisql_uao_scorer_v1_safe_aggregate_receipt_v1",
    "status": "PASS_REALITY_PRIMARY",
    "study_id": "UAO_P4_WIKISQL_TYPED_RELATIONAL_ROW_EVIDENCE_V1",
})
safe_file = write(safe_path, safe)
terminal = addressed({
    "block": "A_hold",
    "primary_passed": True,
    "private_score_file_sha256": private_file,
    "private_score_pack_sha256": private["self_sha256"],
    "safe_aggregate_file_sha256": safe_file,
    "safe_aggregate_receipt_sha256": safe["self_sha256"],
    "schema": "wikisql_uao_scorer_v1_safe_terminal_v1",
    "status": "completed",
    "study_id": "UAO_P4_WIKISQL_TYPED_RELATIONAL_ROW_EVIDENCE_V1",
})
write(terminal_path, terminal)
"""


@dataclass
class SyntheticFormal:
    root: Path
    config_path: Path
    paths: runner.FormalPaths
    source_payloads: Mapping[str, Mapping[str, object] | bytes]
    action_script: Path
    scorer_script: Path
    action_packs: Mapping[str, Mapping[str, object]]
    source_calls: list[str]
    mutable_tree: Path

    def dependencies(
        self,
        *,
        bad_arm: str | None = None,
        tamper_arm: str | None = None,
    ) -> runner.Dependencies:
        def source_compile(
            config: runner.FormalConfig,
            paths: runner.FormalPaths,
        ) -> Mapping[str, str]:
            self.source_calls.append("source_compile")
            hashes: dict[str, str] = {}
            paths.compiled.mkdir(mode=0o700)
            (paths.compiled / "private").mkdir(mode=0o700)
            (paths.compiled / "safe").mkdir(mode=0o700)
            pack_commitments: dict[str, dict[str, str]] = {}
            for relative, value in self.source_payloads.items():
                raw = value if isinstance(value, bytes) else _raw(value)
                hashes[relative] = _write(
                    paths.compiled / relative, raw
                )
                if isinstance(value, Mapping) and "self_sha256" in value:
                    key = {
                        "private/A_form.action_views.json": "A_form_action_view",
                        "private/A_form.labels.json": "A_form_label",
                        "private/A_hold.action_views.json": "A_hold_action_view",
                        "private/A_hold.labels.json": "A_hold_label",
                        "private/controller_only.provenance.json": (
                            "controller_only_provenance"
                        ),
                    }.get(relative)
                    if key is not None:
                        pack_commitments[key] = {
                            "canonical_payload_sha256": reality.canonical_sha256(
                                value
                            ),
                            "self_sha256": str(value["self_sha256"]),
                        }
            receipt = _addressed(
                {
                    "authorized_member_open_count": len(
                        source_compiler.REQUIRED_MEMBERS
                    ),
                    "babel_locale": source_compiler.BABEL_LOCALE,
                    "babel_required_production_version": (
                        source_compiler.PRODUCTION_BABEL_VERSION
                    ),
                    "babel_runtime_version": (
                        source_compiler.PRODUCTION_BABEL_VERSION
                    ),
                    "eligibility_contract": {
                        "condition_count": 1,
                        "condition_operator_indices": [0, 1, 2],
                        "table_physical_row_count_minimum": (
                            reality.MIN_TABLE_ROWS
                        ),
                        "table_physical_row_count_maximum": (
                            reality.MAX_TABLE_ROWS
                        ),
                        "column_count_minimum": 1,
                        "column_count_maximum": source_compiler.MAX_COLUMNS,
                        "question_character_count_maximum": (
                            source_compiler.MAX_QUESTION_CHARACTERS
                        ),
                        "header_or_cell_character_count_maximum": (
                            source_compiler.MAX_HEADER_OR_CELL_CHARACTERS
                        ),
                        "canonical_serialized_row_character_count_maximum": (
                            reality.MAX_SERIALIZED_ROW_CHARACTERS
                        ),
                        "canonical_serialized_rows_must_round_trip": True,
                        "canonical_serialized_rows_must_be_unique": True,
                        "sqlite_schema_rowid_order_and_normalized_cells_must_match_json_before_gold_derivation": True,
                        "sqlite_gold_row_count_minimum": (
                            reality.MIN_GOLD_ROWS
                        ),
                        "sqlite_gold_row_count_maximum": (
                            reality.MAX_GOLD_ROWS
                        ),
                        "sqlite_gold_authoritative_before_HMAC": True,
                    },
                    "pack_commitments": pack_commitments,
                    "schema": (
                        f"{source_compiler.VERSION}_safe_aggregate_receipt_v1"
                    ),
                    "selected_item_count": 264,
                    "selected_sqlite_consistency_assert_count": 264,
                    "source_archive_git_blob_sha1": (
                        source_compiler.PRODUCTION_ARCHIVE_GIT_BLOB_SHA1
                    ),
                    "source_archive_sha256": config.file(
                        "source_archive"
                    ).sha256,
                    "sqlite_rowid_eligible_count": 264,
                    "status": "compiled_source_and_sealed_private_packs",
                    "study_id": runner.STUDY_ID,
                    "train_test_table_overlap_count": 0,
                }
            )
            relative = "safe/source_compiler_receipt.json"
            hashes[relative] = _write(
                paths.compiled / relative, _raw(receipt)
            )
            return hashes

        def action_commands(
            config: runner.FormalConfig,
            paths: runner.FormalPaths,
            _source: runner.SourceArtifacts,
        ) -> Mapping[str, runner.CommandSpec]:
            coordination = paths.work / "coordination"
            coordination.mkdir(mode=0o700)
            commands: dict[str, runner.CommandSpec] = {}
            outputs = {
                "Agent": paths.agent_action,
                "RAW": paths.raw_action,
                "HippoRAG": paths.hippo_action,
            }
            extras = {
                "Agent": (paths.agent_policy, paths.agent_receipt),
                "RAW": (),
                "HippoRAG": (paths.hippo_receipt,),
            }
            for arm in ("Agent", "RAW", "HippoRAG"):
                mode = (
                    "exit"
                    if arm == bad_arm
                    else "tamper"
                    if arm == tamper_arm
                    else "pass"
                )
                environment = {
                    "CUDA_VISIBLE_DEVICES": (
                        "1" if arm == "Agent" else "0" if arm == "HippoRAG" else ""
                    ),
                    "HOME": str(
                        {
                            "Agent": paths.agent_root,
                            "RAW": paths.raw_root,
                            "HippoRAG": paths.hippo_root,
                        }[arm]
                        / "home"
                    ),
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
                lane_root = {
                    "Agent": paths.agent_root,
                    "RAW": paths.raw_root,
                    "HippoRAG": paths.hippo_root,
                }[arm]
                read_paths = [
                    Path("/usr"),
                    Path("/etc"),
                    config.file(
                        "official_python_executable"
                        if arm == "HippoRAG"
                        else "python_executable"
                    ).path,
                    self.action_script,
                    paths.a_hold_view,
                ]
                if arm == "Agent":
                    read_paths.extend(
                        (paths.a_form_view, paths.a_form_labels)
                    )
                commands[arm] = runner.CommandSpec(
                    name=arm,
                    argv=(
                        str(
                            config.file(
                                "official_python_executable"
                                if arm == "HippoRAG"
                                else "python_executable"
                            ).path
                        ),
                        "-I",
                        "-S",
                        "-B",
                        str(self.action_script),
                        arm,
                        str(outputs[arm]),
                        str(coordination),
                        mode,
                        *(str(path) for path in extras[arm]),
                    ),
                    cwd=lane_root,
                    environment=environment,
                    read_paths=tuple(read_paths),
                    write_paths=(lane_root, coordination),
                )
            return commands

        def scorer_command(
            config: runner.FormalConfig,
            paths: runner.FormalPaths,
            _source: runner.SourceArtifacts,
        ) -> runner.CommandSpec:
            return runner.CommandSpec(
                name="scorer",
                argv=(
                    str(config.file("python_executable").path),
                    "-I",
                    "-S",
                    "-B",
                    str(self.scorer_script),
                    str(paths.barrier),
                    str(paths.scorer_labels),
                    str(paths.score_private),
                    str(paths.score_safe),
                    str(paths.score_terminal),
                ),
                cwd=paths.scorer_root,
                environment={
                    "CUDA_VISIBLE_DEVICES": "",
                    "HOME": str(paths.scorer_root / "home"),
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                read_paths=(
                    Path("/usr"),
                    Path("/etc"),
                    config.file("python_executable").path,
                    self.scorer_script,
                    paths.a_hold_view,
                    paths.scorer_labels,
                    paths.agent_action,
                    paths.raw_action,
                    paths.hippo_action,
                    paths.barrier,
                ),
                write_paths=(paths.scorer_root,),
            )

        return runner.Dependencies(
            service_probe=lambda config: runner.ServiceAttestation(
                nrestarts=0,
                invocation_id="a" * 32,
                active_state="activating",
                sub_state="start",
                fragment_path=config.file("service_unit").path,
                drop_in_paths="",
                restart="no",
                exec_start=_effective_exec_start(self.root),
                service_type="oneshot",
                timeout_start_usec="infinity",
                cpu_quota_per_sec_usec="7s",
                memory_max="42949672960",
                tasks_max="128",
                ip_address_deny="::/0 0.0.0.0/0",
                umask="0077",
                private_tmp="yes",
                no_new_privileges="yes",
                restrict_address_families="AF_UNIX",
                kill_mode="control-group",
            ),
            gpu_probe=lambda _config: runner.GPUAttestation(
                uuids=GPU_UUIDS,
                compute_process_count=0,
            ),
            abi_probe=runner.landlock_abi_version,
            outer_landlock=lambda _config, _paths: None,
            child_landlock=runner.apply_landlock,
            source_compile=source_compile,
            action_commands=action_commands,
            label_projector=runner._project_minimal_labels_production,
            scorer_command=scorer_command,
        )


def _build_synthetic_formal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SyntheticFormal:
    root = tmp_path / "formal_v1"
    control = root / "control"
    work = root / "work"
    code = root / "reconstruction_v2"
    for directory in (root, control, work, code):
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
    monkeypatch.setattr(runner, "FORMAL_ROOT", root)

    payloads = _source_pack_payloads()
    hold_view = payloads["private/A_hold.action_views.json"]
    assert isinstance(hold_view, Mapping)
    action_packs = {
        arm: _common_action_pack(hold_view, arm)
        for arm in ("Agent", "RAW", "HippoRAG")
    }

    manifests = code / "manifests"
    manifests.mkdir(mode=0o700)
    design_body = {
        "decision": "GO_SYNTHETIC_CONTROLLER_TEST",
        "schema": "wikisql_uao_p4_study_design_v1",
        "study_id": runner.STUDY_ID,
    }
    design = _addressed(design_body)
    design_path = code / runner.DESIGN_RELATIVE_PATH
    _write(design_path, _raw(design), mode=0o644)
    service_path = code / runner.SERVICE_RELATIVE_PATH
    _write(
        service_path,
        _service_text(root).encode("utf-8"),
        mode=0o644,
    )

    source_path = root / runner.SOURCE_RELATIVE_PATH
    _write(source_path, b"synthetic archive bytes", mode=0o600)

    action_script = code / "synthetic_action.py"
    _write(
        action_script,
        _action_script(
            label_path=runner.FormalPaths.for_root(root).a_hold_labels,
            action_packs=action_packs,
        ).encode("utf-8"),
        mode=0o644,
    )
    scorer_script = code / "synthetic_scorer.py"
    _write(scorer_script, _scorer_script().encode("utf-8"), mode=0o644)

    assets = root / "assets"
    assets.mkdir(mode=0o700)
    file_paths: dict[str, Path] = {
        "design": design_path,
        "service_unit": service_path,
        "source_archive": source_path,
    }
    for name in (
        "nvidia_smi_executable",
        "official_python_executable",
        "python_executable",
        "systemctl_executable",
    ):
        path = assets / name
        payload = (
            b"#!/bin/sh\nexec /usr/bin/python3.10 \"$@\"\n"
            if name
            in {"official_python_executable", "python_executable"}
            else f"{name}\n".encode()
        )
        _write(path, payload, mode=0o755)
        file_paths[name] = path

    tree_paths: dict[str, Path] = {"code_tree": code}
    for name in (
        "babel_dependency_tree",
        "encoder_model_tree",
        "hippo_llm_model_tree",
        "official_base_dependency_tree",
        "official_hipporag_tree",
        "official_overlay_dependency_tree",
        "official_python_dependency_tree",
        "python_dependency_tree",
        "python_runtime_tree",
    ):
        path = assets / name
        path.mkdir(mode=0o700)
        _write(path / "identity.bin", name.encode(), mode=0o600)
        tree_paths[name] = path
    mutable_tree = tree_paths["encoder_model_tree"]

    config_body = {
        "bindings": {
            "files": {
                name: _file_binding(path)
                for name, path in sorted(file_paths.items())
            },
            "trees": {
                name: _tree_binding(path)
                for name, path in sorted(tree_paths.items())
            },
        },
        "design_self_sha256": design["self_sha256"],
        "encoder_model_semantic_sha256": (
            runner.action_runtime.directory_tree_sha256(
                tree_paths["encoder_model_tree"]
            )
        ),
        "formal_root": str(root),
        "gpu_uuids": GPU_UUIDS,
        "schema": runner.CONFIG_SCHEMA,
        "study_id": runner.STUDY_ID,
        "unit_name": runner.UNIT_NAME,
    }
    config = _addressed(config_body)
    config_path = control / "formal_config.json"
    _write(config_path, _raw(config), mode=0o600)
    return SyntheticFormal(
        root=root,
        config_path=config_path,
        paths=runner.FormalPaths.for_root(root),
        source_payloads=payloads,
        action_script=action_script,
        scorer_script=scorer_script,
        action_packs=action_packs,
        source_calls=[],
        mutable_tree=mutable_tree,
    )


def test_formal_submits_three_lanes_concurrently_and_opens_labels_late(
    posix_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _build_synthetic_formal(posix_root, monkeypatch)
    terminal = runner._run_with_dependencies(
        formal.config_path, formal.dependencies()
    )
    assert terminal["status"] == "completed_protocol_valid"
    assert terminal["primary_passed"] is True
    assert terminal["action_child_launch_count"] == 3
    assert terminal["source_compiler_invocation_count"] == 1
    assert formal.source_calls == ["source_compile"]
    assert (
        formal.paths.agent_root / "label_probe.txt"
    ).read_text() == "DENIED\n"
    observation = json.loads(
        (
            formal.paths.scorer_root / "scorer_observation.json"
        ).read_text()
    )
    assert observation == {"barrier_seen": True, "labels_read": True}
    assert formal.paths.barrier.stat().st_mtime_ns <= (
        formal.paths.scorer_labels.stat().st_mtime_ns
    )
    assert {
        path.name
        for path in (formal.paths.work / "coordination").glob("*.started")
    } == {"Agent.started", "RAW.started", "HippoRAG.started"}
    assert _load_json(formal.paths.live)["nrestarts"] == 0


def _load_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text("ascii"))
    assert isinstance(value, Mapping)
    return value


def test_bound_tree_tamper_fails_before_attempt_or_source_open_and_no_retry(
    posix_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _build_synthetic_formal(posix_root, monkeypatch)
    marker = formal.mutable_tree / "identity.bin"
    marker.write_bytes(b"tampered")
    terminal = runner._run_with_dependencies(
        formal.config_path, formal.dependencies()
    )
    assert terminal["status"] == "formal_failed_no_retry_efficacy_unknown"
    assert terminal["stage"] == "verify_pre_attempt_bindings"
    assert terminal["attempt_claimed"] is False
    assert terminal["source_compiler_invocation_count"] == 0
    assert not formal.paths.attempt.exists()
    assert formal.source_calls == []
    with pytest.raises(
        runner.WikiSQLUAOFormalError, match="retry is forbidden"
    ):
        runner._run_with_dependencies(
            formal.config_path, formal.dependencies()
        )


def test_source_binding_is_first_read_only_after_unique_attempt(
    posix_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _build_synthetic_formal(posix_root, monkeypatch)
    source_path = runner.load_config(formal.config_path).file(
        "source_archive"
    ).path
    source_path.write_bytes(b"tampered-after-config-freeze")
    terminal = runner._run_with_dependencies(
        formal.config_path, formal.dependencies()
    )
    assert terminal["status"] == "formal_failed_no_retry_efficacy_unknown"
    assert terminal["stage"] == "verify_formal_source_binding_after_attempt"
    assert terminal["attempt_claimed"] is True
    assert terminal["source_compiler_invocation_count"] == 0
    assert formal.paths.attempt.exists()
    assert formal.source_calls == []
    with pytest.raises(
        runner.WikiSQLUAOFormalError, match="retry is forbidden"
    ):
        runner._run_with_dependencies(
            formal.config_path, formal.dependencies()
        )


@pytest.mark.parametrize(
    ("bad_arm", "tamper_arm", "expected_stage"),
    (
        ("RAW", None, "launch_three_actions_concurrently"),
        (None, "HippoRAG", "validate_and_durably_seal_common_actions"),
    ),
)
def test_action_failure_or_tamper_is_terminal_without_scorer_or_retry(
    posix_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_arm: str | None,
    tamper_arm: str | None,
    expected_stage: str,
) -> None:
    formal = _build_synthetic_formal(posix_root, monkeypatch)
    terminal = runner._run_with_dependencies(
        formal.config_path,
        formal.dependencies(bad_arm=bad_arm, tamper_arm=tamper_arm),
    )
    assert terminal["status"] == "formal_failed_no_retry_efficacy_unknown"
    assert terminal["stage"] == expected_stage
    assert terminal["attempt_claimed"] is True
    assert terminal["source_compiler_invocation_count"] == 1
    assert terminal["scorer_launch_count"] == 0
    assert not formal.paths.barrier.exists()
    assert not formal.paths.scorer_labels.exists()
    assert not formal.paths.score_terminal.exists()
    with pytest.raises(
        runner.WikiSQLUAOFormalError, match="retry is forbidden"
    ):
        runner._run_with_dependencies(
            formal.config_path, formal.dependencies()
        )
    assert formal.source_calls == ["source_compile"]


def test_successful_attempt_is_exclusive_and_never_replayed(
    posix_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _build_synthetic_formal(posix_root, monkeypatch)
    first = runner._run_with_dependencies(
        formal.config_path, formal.dependencies()
    )
    terminal_bytes = formal.paths.terminal.read_bytes()
    with pytest.raises(
        runner.WikiSQLUAOFormalError, match="retry is forbidden"
    ):
        runner._run_with_dependencies(
            formal.config_path, formal.dependencies()
        )
    assert formal.paths.terminal.read_bytes() == terminal_bytes
    assert first["status"] == "completed_protocol_valid"
    assert formal.source_calls == ["source_compile"]


def test_safe_score_receipt_rejects_extra_self_hashed_text_fields(
    posix_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _build_synthetic_formal(posix_root, monkeypatch)
    runner._run_with_dependencies(
        formal.config_path, formal.dependencies()
    )
    safe = dict(_load_json(formal.paths.score_safe))
    safe.pop("self_sha256")
    safe["note"] = "forbidden covert payload"
    tampered_safe = _addressed(safe)
    safe_raw = _raw(tampered_safe)
    formal.paths.score_safe.write_bytes(safe_raw)

    terminal = dict(_load_json(formal.paths.score_terminal))
    terminal.pop("self_sha256")
    terminal["safe_aggregate_file_sha256"] = hashlib.sha256(
        safe_raw
    ).hexdigest()
    terminal["safe_aggregate_receipt_sha256"] = tampered_safe[
        "self_sha256"
    ]
    formal.paths.score_terminal.write_bytes(
        _raw(_addressed(terminal))
    )
    with pytest.raises(
        runner.WikiSQLUAOFormalError,
        match="safe aggregate receipt drifted",
    ):
        runner._verify_scorer_outputs(formal.paths)


def test_production_commands_bind_distinct_common_and_official_lanes(
    posix_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _build_synthetic_formal(posix_root, monkeypatch)
    config = runner.load_config(formal.config_path)
    source = runner.SourceArtifacts(
        output_file_sha256={},
        compiler_receipt_self_sha256="0" * 64,
        a_hold_view_self_sha256="1" * 64,
        a_hold_label_pack_self_sha256="2" * 64,
    )
    commands = runner._production_action_commands(
        config, formal.paths, source
    )
    scorer = runner._production_scorer_command(
        config, formal.paths, source
    )
    common_python = config.file("python_executable").path
    official_python = config.file("official_python_executable").path
    common_pythonpath = os.pathsep.join(
        str(config.tree(name).path)
        for name in (
            "code_tree",
            "python_dependency_tree",
            "babel_dependency_tree",
        )
    )
    official_pythonpath = os.pathsep.join(
        str(config.tree(name).path)
        for name in (
            "code_tree",
            "official_python_dependency_tree",
            "babel_dependency_tree",
            "official_hipporag_tree",
            "official_overlay_dependency_tree",
            "official_base_dependency_tree",
        )
    )
    assert commands["Agent"].argv[0] == str(common_python)
    assert commands["RAW"].argv[0] == str(common_python)
    assert scorer.argv[0] == str(common_python)
    assert commands["HippoRAG"].argv[0] == str(official_python)
    semantic_model_hash = runner.action_runtime.directory_tree_sha256(
        config.tree("encoder_model_tree").path
    )
    semantic_hash_index = (
        commands["Agent"].argv.index("--encoder-model-sha256") + 1
    )
    assert semantic_model_hash == config.encoder_model_semantic_sha256
    assert (
        commands["Agent"].argv[semantic_hash_index]
        == semantic_model_hash
    )
    assert semantic_model_hash != config.tree("encoder_model_tree").sha256
    assert commands["Agent"].environment["PYTHONPATH"] == common_pythonpath
    assert commands["RAW"].environment["PYTHONPATH"] == common_pythonpath
    assert scorer.environment["PYTHONPATH"] == common_pythonpath
    assert (
        commands["HippoRAG"].environment["PYTHONPATH"]
        == official_pythonpath
    )
    assert official_python not in commands["Agent"].read_paths
    assert official_python not in commands["RAW"].read_paths
    assert official_python in commands["HippoRAG"].read_paths
    assert common_python not in commands["HippoRAG"].read_paths
    for name in (
        "official_python_dependency_tree",
        "official_overlay_dependency_tree",
        "official_base_dependency_tree",
    ):
        tree = config.tree(name).path
        assert tree in commands["HippoRAG"].read_paths
        assert tree not in commands["Agent"].read_paths
        assert tree not in commands["RAW"].read_paths
        assert tree not in scorer.read_paths


def test_outer_landlock_admits_both_gpus_before_child_narrowing(
    posix_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _build_synthetic_formal(posix_root, monkeypatch)
    config = runner.load_config(formal.config_path)
    captured: dict[str, object] = {}
    common = Path("/dev/frozen-common")
    gpu0 = Path("/dev/frozen-gpu0")
    gpu1 = Path("/dev/frozen-gpu1")
    monkeypatch.setattr(
        runner,
        "_gpu_device_paths",
        lambda index: (
            (gpu0, common) if index == "0" else (gpu1, common)
        ),
    )
    monkeypatch.setattr(
        runner,
        "apply_landlock",
        lambda **kwargs: captured.update(kwargs),
    )
    runner._outer_landlock(config, formal.paths)
    assert tuple(captured["device_paths"]) == (
        gpu0,
        common,
        gpu1,
        common,
    )


def test_service_is_minimal_uao_v3_without_capability_or_mount_hardening() -> None:
    service = (
        Path(__file__).resolve().parents[1]
        / "manifests/wikisql-uao-p4-formal-v1.service"
    ).read_text("utf-8")
    assert "RestrictAddressFamilies=AF_UNIX" in service
    assert "IPAddressDeny=any" in service
    assert "NoNewPrivileges=yes" in service
    assert "PrivateTmp=yes" in service
    assert (
        "/home/erzhu419/p19_runtime_assets_20260723/"
        "typed_venv/bin/python"
    ) in service
    assert (
        "/home/erzhu419/wikisql_uao_p4_20260729/formal_v1/"
        "runtime_assets/babel_2_10_3_clean"
    ) in service
    assert "/usr/bin/python3.10" not in service
    for forbidden in runner._FORBIDDEN_SERVICE_PREFIXES:
        assert forbidden not in service


def test_effective_service_profile_rejects_drop_in_override(
    posix_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _build_synthetic_formal(posix_root, monkeypatch)
    config = runner.load_config(formal.config_path)
    service = formal.dependencies().service_probe(config)
    with pytest.raises(
        runner.WikiSQLUAOFormalError,
        match="effective user-service properties drifted",
    ):
        runner._verify_effective_service_profile(
            replace(
                service,
                drop_in_paths="/home/erzhu419/.config/systemd/user/"
                "wikisql-uao-p4-formal-v1.service.d/override.conf",
            ),
            config.file("service_unit").path.read_bytes(),
        )
