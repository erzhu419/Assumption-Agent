"""One-shot, source-free production-path canary for WikiSQL UAO P4.

One exact content-addressed config launches three Landlocked children before
waiting: GPU1 loads/encodes with the real MiniLM, CPU runs the real RAW path,
and GPU0 runs the candidate-restricted official HippoRAG worker through one
complete index/retrieve over a fixed synthetic 11-row table.  Both frozen
``python -S`` dependency orders and the clean Babel 2.10.3 origin are checked
inside the child processes.  No source, label, evaluator, API, network,
retry, replay, or fallback channel is representable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from assumption_agent.benchmarks import (
    wikisql_uao_action_runtime_v1 as action_runtime,
)
from replication_runtime.wikisql_uao_formal_v1 import runner as formal
from replication_runtime.wikisql_uao_official_v1 import contract as official_contract
from replication_runtime.wikisql_uao_official_v1 import worker as official_worker


VERSION = "wikisql_uao_source_free_production_canary_v1"
STUDY_ID = formal.STUDY_ID
CONFIG_SCHEMA = f"{VERSION}_content_addressed_config_v1"
CANARY_ROOT = Path(
    "/home/erzhu419/wikisql_uao_p4_20260729/source_free_canary_v1"
)
FORMAL_SOURCE_ROOT = formal.FORMAL_ROOT / "source"
UNIT_NAME = "wikisql-uao-p4-source-free-canary-v1.service"
INSTALLED_UNIT_PATH = (
    Path("/home/erzhu419/.config/systemd/user") / UNIT_NAME
)
SERVICE_RELATIVE_PATH = Path(
    "manifests/wikisql-uao-p4-source-free-canary-v1.service"
)
MODULE = "replication_runtime.wikisql_uao_source_free_canary_v1.runner"
EXPECTED_BABEL_VERSION = "2.10.3"
COMMON_ORDER = (
    "code_tree",
    "python_dependency_tree",
    "babel_dependency_tree",
)
OFFICIAL_ORDER = (
    "code_tree",
    "official_python_dependency_tree",
    "babel_dependency_tree",
    "official_hipporag_tree",
    "official_overlay_dependency_tree",
    "official_base_dependency_tree",
)
FILES = frozenset(
    {
        "nvidia_smi_executable",
        "official_python_executable",
        "python_executable",
        "service_unit",
        "systemctl_executable",
    }
)
TREES = frozenset(
    {
        *COMMON_ORDER,
        *OFFICIAL_ORDER,
        "encoder_model_tree",
        "hippo_llm_model_tree",
        "official_python_runtime_tree",
        "python_runtime_tree",
    }
)
LANE_SCHEMAS = {
    lane: f"{VERSION}_{lane.casefold()}_lane_safe_v1"
    for lane in ("Agent", "RAW", "HippoRAG")
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
GPU_UUID = re.compile(r"GPU-[A-Za-z0-9-]{8,}\Z")


class WikiSQLUAOCanaryError(RuntimeError):
    """The source-free production-path qualification failed closed."""


@dataclass(frozen=True, slots=True)
class Config:
    path: Path
    files: Mapping[str, formal.FileBinding]
    trees: Mapping[str, formal.TreeBinding]
    gpu_uuids: Mapping[str, str]
    encoder_model_semantic_sha256: str
    self_sha256: str

    def file(self, name: str) -> formal.FileBinding:
        return self.files[name]

    def tree(self, name: str) -> formal.TreeBinding:
        return self.trees[name]


@dataclass(frozen=True, slots=True)
class Paths:
    root: Path
    control: Path
    work: Path
    attempt: Path
    terminal: Path
    failure: Path
    input: Path
    agent: Path
    raw: Path
    hippo: Path

    @classmethod
    def fixed(cls) -> "Paths":
        control, work = CANARY_ROOT / "control", CANARY_ROOT / "work"
        return cls(
            CANARY_ROOT,
            control,
            work,
            control / "canary_attempt.json",
            control / "canary_terminal.safe.json",
            control / "canary_failure.safe.json",
            work / "synthetic.action_views.json",
            work / "agent",
            work / "raw",
            work / "hipporag",
        )


def _addressed(value: Mapping[str, object]) -> dict[str, object]:
    return {**value, "self_sha256": formal.semantic_sha256(value)}


def _write(path: Path, value: Mapping[str, object]) -> str:
    return formal._write_once(path, value, mode=0o600)


def _load(path: Path, field: str) -> dict[str, object]:
    return formal._load_canonical_json(path, mode=0o600, field=field)


def _reject_source_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(FORMAL_SOURCE_ROOT)
    except ValueError:
        pass
    else:
        raise WikiSQLUAOCanaryError("formal source path is forbidden")
    if any(part.casefold() == "source" for part in resolved.parts):
        raise WikiSQLUAOCanaryError("source path is forbidden")


def load_config(path: Path) -> Config:
    if path != CANARY_ROOT / "control/canary_config.json":
        raise WikiSQLUAOCanaryError("config path drifted")
    value = _load(path, "canary config")
    expected = {
        "bindings",
        "canary_root",
        "encoder_model_semantic_sha256",
        "expected_babel_version",
        "gpu_uuids",
        "pythonpath_order",
        "schema",
        "self_sha256",
        "study_id",
        "unit_name",
    }
    body = {key: child for key, child in value.items() if key != "self_sha256"}
    if (
        set(value) != expected
        or value.get("schema") != CONFIG_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("canary_root") != str(CANARY_ROOT)
        or value.get("unit_name") != UNIT_NAME
        or value.get("expected_babel_version") != EXPECTED_BABEL_VERSION
        or value.get("pythonpath_order")
        != {"common": list(COMMON_ORDER), "official": list(OFFICIAL_ORDER)}
        or formal.semantic_sha256(body) != value.get("self_sha256")
    ):
        raise WikiSQLUAOCanaryError("config identity or capability shape drifted")
    bindings = value.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"files", "trees"}:
        raise WikiSQLUAOCanaryError("binding envelope drifted")
    raw_files, raw_trees = bindings["files"], bindings["trees"]
    if (
        not isinstance(raw_files, dict)
        or set(raw_files) != FILES
        or not isinstance(raw_trees, dict)
        or set(raw_trees) != TREES
    ):
        raise WikiSQLUAOCanaryError("binding registry drifted")
    files = {
        name: formal.FileBinding.parse(raw_files[name], name)
        for name in FILES
    }
    trees = {
        name: formal.TreeBinding.parse(raw_trees[name], name)
        for name in TREES
    }
    for binding in (*files.values(), *trees.values()):
        _reject_source_path(binding.path)
    semantic = value.get("encoder_model_semantic_sha256")
    gpu = value.get("gpu_uuids")
    if (
        not isinstance(semantic, str)
        or HEX64.fullmatch(semantic) is None
        or not isinstance(gpu, dict)
        or set(gpu) != {"0", "1"}
        or any(
            not isinstance(gpu[index], str)
            or GPU_UUID.fullmatch(gpu[index]) is None
            for index in ("0", "1")
        )
        or gpu["0"] == gpu["1"]
        or trees["code_tree"].path != CANARY_ROOT / "reconstruction_v2"
        or files["service_unit"].path
        != trees["code_tree"].path / SERVICE_RELATIVE_PATH
        or trees["babel_dependency_tree"].path
        != CANARY_ROOT / "runtime_assets/babel_2_10_3_clean"
    ):
        raise WikiSQLUAOCanaryError("fixed runtime layout drifted")
    return Config(
        path,
        files,
        trees,
        {"0": gpu["0"], "1": gpu["1"]},
        semantic,
        str(value["self_sha256"]),
    )


def _roots(config: Config, order: Sequence[str]) -> tuple[Path, ...]:
    return tuple(config.tree(name).path for name in order)


def _environment(root: Path, cuda: str, module_roots: Sequence[Path]) -> dict[str, str]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": cuda,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(root / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, module_roots)),
        "TEMP": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TMPDIR": str(root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _dep_args(config: Config, official: bool, output: Path) -> tuple[str, ...]:
    executable = config.file(
        "official_python_executable" if official else "python_executable"
    )
    order = OFFICIAL_ORDER if official else COMMON_ORDER
    result = [
        "--expected-python",
        str(executable.path),
        "--expected-python-sha256",
        executable.sha256,
        "--config-self-sha256",
        config.self_sha256,
        "--babel-root",
        str(config.tree("babel_dependency_tree").path),
        "--lane-receipt-output",
        str(output),
    ]
    for root in _roots(config, order):
        result += ["--pythonpath-root", str(root)]
    return tuple(result)


def synthetic_input() -> dict[str, object]:
    return official_contract.input_payload(
        items=[
            {
                "headers": ["City", "Score"],
                "item_id": hashlib.sha256(b"UAO-P4-synthetic-v1").hexdigest(),
                "question": "Which synthetic city has score seven?",
                "rows": [[f"City-{index}", index] for index in range(11)],
                "types": ["text", "real"],
            }
        ]
    )


def build_commands(config: Config, paths: Paths) -> Mapping[str, formal.CommandSpec]:
    common, official = _roots(config, COMMON_ORDER), _roots(config, OFFICIAL_ORDER)
    common_read = (
        *formal._existing_system_read_paths(),
        config.file("python_executable").path,
        config.tree("python_runtime_tree").path,
        *common,
    )
    official_read = (
        *formal._existing_system_read_paths(),
        config.file("official_python_executable").path,
        config.tree("official_python_runtime_tree").path,
        *official,
    )

    def prefix(official_lane: bool) -> tuple[str, ...]:
        name = "official_python_executable" if official_lane else "python_executable"
        return (str(config.file(name).path), "-S", "-B", "-s", "-m", MODULE)

    commands = {
        "Agent": formal.CommandSpec(
            "Agent",
            (
                *prefix(False),
                "agent",
                *_dep_args(config, False, paths.agent / "lane.safe.json"),
                "--model",
                str(config.tree("encoder_model_tree").path),
                "--model-semantic-sha256",
                config.encoder_model_semantic_sha256,
            ),
            paths.agent,
            _environment(paths.agent, "1", common),
            (*common_read, config.tree("encoder_model_tree").path),
            (paths.agent,),
            formal._gpu_device_paths("1"),
        ),
        "RAW": formal.CommandSpec(
            "RAW",
            (
                *prefix(False),
                "raw",
                *_dep_args(config, False, paths.raw / "lane.safe.json"),
                "--input",
                str(paths.input),
            ),
            paths.raw,
            _environment(paths.raw, "", common),
            (*common_read, paths.input),
            (paths.raw,),
        ),
        "HippoRAG": formal.CommandSpec(
            "HippoRAG",
            (
                *prefix(True),
                "hippo",
                *_dep_args(config, True, paths.hippo / "lane.safe.json"),
                "--input",
                str(paths.input),
                "--action-output",
                str(paths.hippo / "action.private.json"),
                "--official-receipt-output",
                str(paths.hippo / "official.safe.json"),
                "--index-parent",
                str(paths.hippo / "indexes"),
                "--llm-model",
                str(config.tree("hippo_llm_model_tree").path),
                "--embedding-model",
                str(config.tree("encoder_model_tree").path),
            ),
            paths.hippo,
            _environment(paths.hippo, "0", official),
            (
                *official_read,
                config.tree("encoder_model_tree").path,
                config.tree("hippo_llm_model_tree").path,
                paths.input,
            ),
            (paths.hippo,),
            formal._gpu_device_paths("0"),
        ),
    }
    if any("-S" not in command.argv for command in commands.values()):
        raise WikiSQLUAOCanaryError("fixed -S command drifted")
    return commands


def _dependency(arguments: argparse.Namespace) -> dict[str, object]:
    roots = tuple(Path(value).resolve() for value in arguments.pythonpath_root)
    if (
        Path(sys.executable).resolve() != arguments.expected_python.resolve()
        or sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or os.environ.get("PYTHONPATH") != os.pathsep.join(map(str, roots))
        or tuple(Path(value).resolve() for value in sys.path[1 : 1 + len(roots)])
        != roots
    ):
        raise WikiSQLUAOCanaryError("fixed -S dependency order drifted")
    python_sha, _ = formal._file_sha256(arguments.expected_python)
    if python_sha != arguments.expected_python_sha256:
        raise WikiSQLUAOCanaryError("interpreter binding drifted")
    import babel

    origin = Path(babel.__file__).resolve()
    if (
        babel.__version__ != EXPECTED_BABEL_VERSION
        or origin != arguments.babel_root.resolve() / "babel/__init__.py"
    ):
        raise WikiSQLUAOCanaryError("Babel 2.10.3 origin drifted")
    babel_sha, _ = formal._file_sha256(origin)
    return {
        "babel_origin_file_sha256": babel_sha,
        "babel_version": babel.__version__,
        "config_self_sha256": arguments.config_self_sha256,
        "interpreter_file_sha256": python_sha,
        "pythonpath_order_sha256": formal.semantic_sha256(list(map(str, roots))),
    }


def _lane_receipt(
    arguments: argparse.Namespace, lane: str, fields: Mapping[str, object]
) -> None:
    _write(
        arguments.lane_receipt_output,
        _addressed(
            {
                **_dependency(arguments),
                **fields,
                "API_call_count": 0,
                "lane": lane,
                "network_call_count": 0,
                "online_evaluator_call_count": 0,
                "replay_count": 0,
                "retry_count": 0,
                "schema": LANE_SCHEMAS[lane],
                "status": "passed",
            }
        ),
    )


def _read_input(path: Path) -> dict[str, object]:
    value = _load(path, "synthetic input")
    items = official_contract.validate_input(value)
    if len(items) != 1 or len(items[0].rows) != 11:
        raise WikiSQLUAOCanaryError("synthetic item shape drifted")
    return value


def _agent(arguments: argparse.Namespace) -> None:
    encoder = action_runtime.LocalSentenceTransformerEncoder(
        model_root=arguments.model,
        expected_model_sha256=arguments.model_semantic_sha256,
        device="cuda:0",
    )
    matrix = tuple(
        tuple(map(float, row))
        for row in encoder.encode(
            ("fixed synthetic question", "fixed synthetic evidence"),
            batch_size=action_runtime.ENCODER_BATCH_SIZE,
        )
    )
    import torch

    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or len(matrix) != 2
        or not matrix[0]
        or len(matrix[0]) != len(matrix[1])
        or any(not math.isfinite(value) for row in matrix for value in row)
    ):
        raise WikiSQLUAOCanaryError("real GPU1 MiniLM probe drifted")
    _lane_receipt(
        arguments,
        "Agent",
        {
            "cuda_logical_device_count": 1,
            "embedding_dimension": len(matrix[0]),
            "embedding_matrix_sha256": action_runtime.canonical_sha256(matrix),
            "model_semantic_sha256": arguments.model_semantic_sha256,
            "request_count": 2,
        },
    )


def _raw(arguments: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise WikiSQLUAOCanaryError("RAW unexpectedly sees a GPU")
    view = _read_input(arguments.input)
    action = action_runtime.run_raw(view_pack=view)
    action_runtime.decode_action_pack(
        action,
        expected_block="A_hold",
        expected_arm="RAW",
        expected_action_view_pack_sha256=str(view["self_sha256"]),
    )
    _lane_receipt(
        arguments,
        "RAW",
        {
            "action_pack_sha256": action["self_sha256"],
            "cpu_only": True,
            "input_pack_sha256": view["self_sha256"],
            "item_count": 1,
            "row_count": 11,
        },
    )


def _hippo(arguments: argparse.Namespace) -> None:
    dependency = _dependency(arguments)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise WikiSQLUAOCanaryError("HippoRAG GPU0 assignment drifted")
    status = official_worker.main(
        [
            "--input",
            str(arguments.input),
            "--action-output",
            str(arguments.action_output),
            "--safe-receipt-output",
            str(arguments.official_receipt_output),
            "--index-parent",
            str(arguments.index_parent),
            "--llm-model",
            str(arguments.llm_model),
            "--embedding-model",
            str(arguments.embedding_model),
        ]
    )
    action = _load(arguments.action_output, "official action")
    receipt = _load(arguments.official_receipt_output, "official receipt")
    runtime = receipt.get("runtime")
    receipt_body = {
        key: child for key, child in receipt.items() if key != "self_sha256"
    }
    if (
        status != 0
        or receipt.get("schema") != official_contract.SAFE_RECEIPT_SCHEMA
        or receipt.get("official_hipporag_commit")
        != official_contract.OFFICIAL_HIPPORAG_COMMIT
        or formal.semantic_sha256(receipt_body) != receipt.get("self_sha256")
        or receipt.get("item_count") != 1
        or not isinstance(runtime, dict)
        or runtime.get("index_call_count") != 1
        or runtime.get("retrieve_call_count") != 1
        or runtime.get("network_call_count") != 0
        or runtime.get("evaluator_call_count") != 0
        or runtime.get("retry_count") != 0
        or runtime.get("replay_count") != 0
    ):
        raise WikiSQLUAOCanaryError("official full index/retrieve probe drifted")
    action_file, _ = formal._file_sha256(arguments.action_output)
    official_file, _ = formal._file_sha256(arguments.official_receipt_output)
    _write(
        arguments.lane_receipt_output,
        _addressed(
            {
                **dependency,
                "API_call_count": 0,
                "action_file_sha256": action_file,
                "action_pack_sha256": action["self_sha256"],
                "cuda_logical_device_count": 1,
                "index_call_count": 1,
                "item_count": 1,
                "lane": "HippoRAG",
                "network_call_count": 0,
                "official_hipporag_commit": official_contract.OFFICIAL_HIPPORAG_COMMIT,
                "official_receipt_file_sha256": official_file,
                "official_receipt_self_sha256": receipt["self_sha256"],
                "online_evaluator_call_count": 0,
                "replay_count": 0,
                "retrieve_call_count": 1,
                "retry_count": 0,
                "row_count": 11,
                "schema": LANE_SCHEMAS["HippoRAG"],
                "status": "passed",
            }
        ),
    )


def _pristine(paths: Paths) -> None:
    for directory in (paths.root, paths.control, paths.work):
        metadata = directory.lstat()
        if (
            directory.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise WikiSQLUAOCanaryError("canary root layout drifted")
    if any(
        path.exists() or path.is_symlink()
        for path in (
            paths.attempt,
            paths.terminal,
            paths.failure,
            paths.input,
            paths.agent,
            paths.raw,
            paths.hippo,
        )
    ):
        raise WikiSQLUAOCanaryError("canary already attempted; retry is forbidden")


def _attested_invocation_id(
    service: formal.ServiceAttestation,
) -> str:
    invocation = service.invocation_id
    if re.fullmatch(r"[0-9a-f]{32}", invocation) is None:
        raise WikiSQLUAOCanaryError("systemd InvocationID drifted")
    return invocation


def _preflight(config: Config) -> tuple[str, int]:
    for name, binding in config.files.items():
        binding.verify(name)
    for name, binding in config.trees.items():
        binding.verify(name)
    if (
        action_runtime.directory_tree_sha256(config.tree("encoder_model_tree").path)
        != config.encoder_model_semantic_sha256
    ):
        raise WikiSQLUAOCanaryError("encoder semantic binding drifted")
    service_raw = config.file("service_unit").path.read_bytes()
    try:
        service = service_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WikiSQLUAOCanaryError("minimal user-service is not UTF-8") from exc
    required = {
        "Type=oneshot",
        "Restart=no",
        "RestrictAddressFamilies=AF_UNIX",
        "IPAddressDeny=any",
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
    }
    lines = {line.strip() for line in service.splitlines()}
    service_pythonpath = "PYTHONPATH=" + os.pathsep.join(
        str(path) for path in _roots(config, COMMON_ORDER)
    )
    if (
        not required <= lines
        or any(
            line.startswith(prefix)
            for line in lines
            for prefix in formal._FORBIDDEN_SERVICE_PREFIXES
        )
        or " -S " not in service
        or service_pythonpath not in service
        or str(config.file("python_executable").path) not in service
        or f"-m {MODULE} controller" not in service
        or str(config.path) not in service
    ):
        raise WikiSQLUAOCanaryError("minimal user-service drifted")
    try:
        service_state = formal._systemctl_attestation(  # type: ignore[arg-type]
            config,
            unit_name=UNIT_NAME,
            installed_unit_path=INSTALLED_UNIT_PATH,
        )
        formal._verify_effective_service_profile(service_state, service_raw)
        fragment = service_state.fragment_path.read_bytes()
    except (OSError, formal.WikiSQLUAOFormalError) as exc:
        raise WikiSQLUAOCanaryError(
            "effective user-service attestation failed"
        ) from exc
    invocation = _attested_invocation_id(service_state)
    gpu = formal._gpu_attestation(config)  # type: ignore[arg-type]
    abi = formal.landlock_abi_version()
    if (
        service_state.nrestarts != 0
        or service_state.drop_in_paths != ""
        or service_state.active_state not in {"activating", "active"}
        or service_state.sub_state not in {"start", "running"}
        or hashlib.sha256(fragment).hexdigest()
        != config.file("service_unit").sha256
        or dict(gpu.uuids) != dict(config.gpu_uuids)
        or gpu.compute_process_count != 0
        or abi < 3
    ):
        raise WikiSQLUAOCanaryError("service/GPU/Landlock attestation drifted")
    return invocation, abi


def _outer(config: Config, paths: Paths) -> None:
    devices = (*formal._gpu_device_paths("0"), *formal._gpu_device_paths("1"))
    if not {"nvidia0", "nvidia1"} <= {path.name for path in devices}:
        raise WikiSQLUAOCanaryError("exact GPU device nodes are unavailable")
    formal.apply_landlock(
        read_paths=(
            *formal._existing_system_read_paths(),
            *(binding.path for binding in config.files.values()),
            *(binding.path for binding in config.trees.values()),
        ),
        write_paths=(paths.root, Path("/tmp")),
        device_paths=devices,
    )


def _verify_lane(path: Path, lane: str, config: Config) -> dict[str, object]:
    value = _load(path, f"{lane} receipt")
    body = {key: child for key, child in value.items() if key != "self_sha256"}
    order = OFFICIAL_ORDER if lane == "HippoRAG" else COMMON_ORDER
    executable = config.file(
        "official_python_executable" if lane == "HippoRAG" else "python_executable"
    )
    babel_sha, _ = formal._file_sha256(
        config.tree("babel_dependency_tree").path / "babel/__init__.py"
    )
    if (
        value.get("schema") != LANE_SCHEMAS[lane]
        or formal.semantic_sha256(body) != value.get("self_sha256")
        or value.get("status") != "passed"
        or value.get("lane") != lane
        or value.get("config_self_sha256") != config.self_sha256
        or value.get("interpreter_file_sha256") != executable.sha256
        or value.get("pythonpath_order_sha256")
        != formal.semantic_sha256([str(config.tree(name).path) for name in order])
        or value.get("babel_version") != EXPECTED_BABEL_VERSION
        or value.get("babel_origin_file_sha256") != babel_sha
        or any(value.get(key) != 0 for key in (
            "API_call_count",
            "network_call_count",
            "online_evaluator_call_count",
            "retry_count",
            "replay_count",
        ))
    ):
        raise WikiSQLUAOCanaryError(f"{lane} safe receipt drifted")
    return value


Launcher = Callable[
    [Mapping[str, formal.CommandSpec], Callable[..., None], Callable[[], None]],
    Mapping[str, int],
]


def _launch(
    commands: Mapping[str, formal.CommandSpec],
    child_landlock: Callable[..., None],
    on_launch: Callable[[], None],
) -> Mapping[str, int]:
    return formal._launch_actions_concurrently(
        commands, child_landlock=child_landlock, on_launch=on_launch
    )


def run_controller(
    config_path: Path,
    *,
    preflight: Callable[[Config], tuple[str, int]] = _preflight,
    outer: Callable[[Config, Paths], None] = _outer,
    launcher: Launcher = _launch,
    child_landlock: Callable[..., None] = formal.apply_landlock,
) -> Mapping[str, object]:
    paths = Paths.fixed()
    _pristine(paths)
    for forbidden in (
        FORMAL_SOURCE_ROOT,
        formal.FORMAL_ROOT / "control/formal_attempt.json",
        formal.FORMAL_ROOT / "control/outer_terminal.safe.json",
    ):
        try:
            forbidden.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise WikiSQLUAOCanaryError(
                "formal pre-source boundary metadata is unavailable"
            ) from exc
        raise WikiSQLUAOCanaryError(
            "formal source or attempt evidence already exists"
        )
    config = load_config(config_path)
    attempt = _addressed(
        {
            "config_self_sha256": config.self_sha256,
            "formal_source_access_count": 0,
            "schema": f"{VERSION}_attempt_v1",
            "status": "claimed_once",
            "study_id": STUDY_ID,
        }
    )
    _write(paths.attempt, attempt)
    stage = "preflight"
    try:
        invocation, abi = preflight(config)
        stage = "outer_landlock"
        outer(config, paths)
        _write(paths.input, synthetic_input())
        for lane in (paths.agent, paths.raw, paths.hippo):
            lane.mkdir(mode=0o700)
            (lane / "home").mkdir(mode=0o700)
            (lane / "tmp").mkdir(mode=0o700)
        commands = build_commands(config, paths)
        launches = 0

        def launched() -> None:
            nonlocal launches
            launches += 1

        stage = "three_lane_launch"
        statuses = launcher(commands, child_landlock, launched)
        if statuses != {"Agent": 0, "RAW": 0, "HippoRAG": 0} or launches != 3:
            raise WikiSQLUAOCanaryError("three lanes failed; retry is forbidden")
        receipts = {
            "Agent": _verify_lane(paths.agent / "lane.safe.json", "Agent", config),
            "RAW": _verify_lane(paths.raw / "lane.safe.json", "RAW", config),
            "HippoRAG": _verify_lane(
                paths.hippo / "lane.safe.json", "HippoRAG", config
            ),
        }
        if (
            receipts["Agent"].get("request_count") != 2
            or receipts["RAW"].get("cpu_only") is not True
            or receipts["HippoRAG"].get("index_call_count") != 1
            or receipts["HippoRAG"].get("retrieve_call_count") != 1
        ):
            raise WikiSQLUAOCanaryError("lane qualification semantics drifted")
        terminal = _addressed(
            {
                "API_or_online_evaluation_count": 0,
                "attempt_self_sha256": attempt["self_sha256"],
                "babel_2_10_3_both_interpreters": True,
                "config_self_sha256": config.self_sha256,
                "formal_source_access_count": 0,
                "gpu0_official_full_index_retrieve_count": 1,
                "gpu1_real_minilm_encode_count": 1,
                "invocation_id_sha256": hashlib.sha256(invocation.encode()).hexdigest(),
                "landlock_abi": abi,
                "lane_receipt_self_sha256": {
                    lane: receipt["self_sha256"] for lane, receipt in receipts.items()
                },
                "network_call_count": 0,
                "raw_cpu_action_count": 1,
                "retry_replay_resample_or_fallback_count": 0,
                "schema": f"{VERSION}_safe_terminal_v1",
                "status": "PASS_WIKISQL_UAO_SOURCE_FREE_PRODUCTION_CANARY",
                "study_id": STUDY_ID,
            }
        )
        _write(paths.terminal, terminal)
        return terminal
    except Exception as exc:
        failure = _addressed(
            {
                "error_class": type(exc).__name__,
                "failure_stage": stage,
                "formal_source_access_count": 0,
                "retry_count": 0,
                "schema": f"{VERSION}_safe_failure_v1",
                "status": "FAILED_NO_RETRY",
                "study_id": STUDY_ID,
            }
        )
        _write(paths.failure, failure)
        return failure


def _dep_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-python", required=True, type=Path)
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--config-self-sha256", required=True)
    parser.add_argument("--pythonpath-root", action="append", required=True)
    parser.add_argument("--babel-root", required=True, type=Path)
    parser.add_argument("--lane-receipt-output", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    ctl = sub.add_parser("controller")
    ctl.add_argument("--config", required=True, type=Path)
    agent = sub.add_parser("agent")
    _dep_parser(agent)
    agent.add_argument("--model", required=True, type=Path)
    agent.add_argument("--model-semantic-sha256", required=True)
    raw = sub.add_parser("raw")
    _dep_parser(raw)
    raw.add_argument("--input", required=True, type=Path)
    hippo = sub.add_parser("hippo")
    _dep_parser(hippo)
    hippo.add_argument("--input", required=True, type=Path)
    hippo.add_argument("--action-output", required=True, type=Path)
    hippo.add_argument("--official-receipt-output", required=True, type=Path)
    hippo.add_argument("--index-parent", required=True, type=Path)
    hippo.add_argument("--llm-model", required=True, type=Path)
    hippo.add_argument("--embedding-model", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "controller":
        result = run_controller(arguments.config)
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        return int(
            result.get("status")
            != "PASS_WIKISQL_UAO_SOURCE_FREE_PRODUCTION_CANARY"
        )
    {"agent": _agent, "raw": _raw, "hippo": _hippo}[arguments.mode](arguments)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
