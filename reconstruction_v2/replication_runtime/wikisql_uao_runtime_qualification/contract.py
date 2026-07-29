"""Strict contract for the stable WikiSQL UAO runtime qualification layer.

This package is intentionally not a study and not a one-shot canary.  It has
no dataset, label, qrel, scoring, evaluator, provider, or API capability.  A
single stable service may execute any number of source-free synthetic
qualification attempts.  Those attempts never consume an effect-study
attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Mapping

from replication_runtime.wikisql_uao_formal_v1 import runner as formal


QUALIFICATION_ID = "WIKISQL_UAO_RUNTIME_QUALIFICATION"
CONFIG_SCHEMA = "wikisql_uao_runtime_qualification_config_v1"
CHECK_SCHEMA = "wikisql_uao_runtime_qualification_check_v1"
TERMINAL_SCHEMA = "wikisql_uao_runtime_qualification_terminal_v1"
LANE_SCHEMA_PREFIX = "wikisql_uao_runtime_qualification"
QUALIFICATION_ROOT = Path(
    "/home/erzhu419/wikisql_uao_runtime_qualification"
)
UNIT_NAME = "wikisql-uao-runtime-qualification.service"
INSTALLED_UNIT_PATH = (
    Path("/home/erzhu419/.config/systemd/user") / UNIT_NAME
)
SERVICE_RELATIVE_PATH = Path(
    "manifests/wikisql-uao-runtime-qualification.service"
)
CONFIG_RELATIVE_PATH = Path("control/runtime_config.json")
PYTHONHOME_ROOT = (
    QUALIFICATION_ROOT / "runtime_assets/python310_clean"
)
BABEL_ROOT = (
    QUALIFICATION_ROOT / "runtime_assets/babel_2_10_3_clean"
)
OFFICIAL_HIPPORAG_ROOT = (
    QUALIFICATION_ROOT / "runtime_assets/hipporag_source_clean"
)
OFFICIAL_BASE_ROOT = (
    QUALIFICATION_ROOT / "runtime_assets/official_base_import_clean"
)
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
REQUIRED_FILES = frozenset(
    {
        "nvidia_smi_executable",
        "official_python_executable",
        "python_executable",
        "service_unit",
        "systemctl_executable",
    }
)
REQUIRED_TREES = frozenset(
    {
        *COMMON_ORDER,
        *OFFICIAL_ORDER,
        "encoder_model_tree",
        "hippo_llm_model_tree",
        "official_python_runtime_tree",
        "python_runtime_tree",
    }
)
CAPABILITY_BOUNDARY = {
    "api_or_network_evaluation_authorized": False,
    "classification": "non_scoring_iterative_runtime_qualification",
    "effect_study_attempt_count": 0,
    "evaluator_or_score_paths_bound": 0,
    "formal_source_paths_bound": 0,
    "label_or_qrel_paths_bound": 0,
}
PYTHONPATH_ORDER = {
    "common": list(COMMON_ORDER),
    "official": list(OFFICIAL_ORDER),
}
THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
GPU_UUID = re.compile(r"GPU-[A-Za-z0-9-]{8,}\Z")
INVOCATION_ID = re.compile(r"[0-9a-f]{32}\Z")
_CONFIG_KEYS = frozenset(
    {
        "bindings",
        "capability_boundary",
        "encoder_model_semantic_sha256",
        "expected_babel_version",
        "gpu_uuids",
        "pythonpath_order",
        "qualification_id",
        "qualification_root",
        "resource_policy",
        "schema",
        "self_sha256",
        "unit_name",
    }
)
_PATH_COMPONENTS_FORBIDDEN = frozenset(
    {
        "dataset",
        "datasets",
        "label",
        "labels",
        "qrel",
        "qrels",
        "score",
        "scores",
        "source",
    }
)


class QualificationContractError(RuntimeError):
    """The non-scoring qualification contract was malformed or drifted."""


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
        raise QualificationContractError(
            "value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value, newline=False)).hexdigest()


def addressed(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    if "self_sha256" in body:
        raise QualificationContractError(
            "addressed payload already contains self_sha256"
        )
    return {**body, "self_sha256": semantic_sha256(body)}


def _exact_dict(
    value: object, keys: frozenset[str], field: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise QualificationContractError(f"{field} shape drifted")
    return value


def _load_canonical(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise QualificationContractError(
            "qualification config is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise QualificationContractError(
            "qualification config metadata drifted"
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationContractError(
            "qualification config is not canonical JSON"
        ) from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise QualificationContractError(
            "qualification config is not canonical JSON"
        )
    return value


def _reject_privileged_path(path: Path) -> None:
    if any(
        component.casefold() in _PATH_COMPONENTS_FORBIDDEN
        for component in path.parts
    ):
        raise QualificationContractError(
            "scoring or formal-source path is not representable"
        )


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    path: Path
    files: Mapping[str, formal.FileBinding]
    trees: Mapping[str, formal.TreeBinding]
    gpu_uuids: Mapping[str, str]
    encoder_model_semantic_sha256: str
    resource_policy: Mapping[str, object]
    self_sha256: str

    def file(self, name: str) -> formal.FileBinding:
        try:
            return self.files[name]
        except KeyError as exc:
            raise QualificationContractError(
                "file binding is absent"
            ) from exc

    def tree(self, name: str) -> formal.TreeBinding:
        try:
            return self.trees[name]
        except KeyError as exc:
            raise QualificationContractError(
                "tree binding is absent"
            ) from exc


def load_config(path: Path) -> QualificationConfig:
    expected_path = QUALIFICATION_ROOT / CONFIG_RELATIVE_PATH
    if path != expected_path:
        raise QualificationContractError(
            "qualification config path drifted"
        )
    value = _exact_dict(_load_canonical(path), _CONFIG_KEYS, "config")
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    supplied_self = value["self_sha256"]
    if (
        not isinstance(supplied_self, str)
        or HEX64.fullmatch(supplied_self) is None
        or semantic_sha256(body) != supplied_self
        or value["schema"] != CONFIG_SCHEMA
        or value["qualification_id"] != QUALIFICATION_ID
        or value["qualification_root"] != str(QUALIFICATION_ROOT)
        or value["unit_name"] != UNIT_NAME
        or value["expected_babel_version"] != EXPECTED_BABEL_VERSION
        or value["pythonpath_order"] != PYTHONPATH_ORDER
        or value["capability_boundary"] != CAPABILITY_BOUNDARY
    ):
        raise QualificationContractError(
            "qualification config identity drifted"
        )
    bindings = _exact_dict(
        value["bindings"], frozenset({"files", "trees"}), "bindings"
    )
    raw_files = bindings["files"]
    raw_trees = bindings["trees"]
    if (
        type(raw_files) is not dict
        or set(raw_files) != REQUIRED_FILES
        or type(raw_trees) is not dict
        or set(raw_trees) != REQUIRED_TREES
    ):
        raise QualificationContractError(
            "qualification binding registry drifted"
        )
    try:
        files = {
            name: formal.FileBinding.parse(
                raw_files[name], f"file binding {name}"
            )
            for name in sorted(REQUIRED_FILES)
        }
        trees = {
            name: formal.TreeBinding.parse(
                raw_trees[name], f"tree binding {name}"
            )
            for name in sorted(REQUIRED_TREES)
        }
    except formal.WikiSQLUAOFormalError as exc:
        raise QualificationContractError(
            "qualification binding payload drifted"
        ) from exc
    for binding in (*files.values(), *trees.values()):
        _reject_privileged_path(binding.path)
    gpu = value["gpu_uuids"]
    semantic = value["encoder_model_semantic_sha256"]
    resource_policy = value["resource_policy"]
    if (
        type(gpu) is not dict
        or set(gpu) != {"0", "1"}
        or any(
            not isinstance(gpu[index], str)
            or GPU_UUID.fullmatch(gpu[index]) is None
            for index in ("0", "1")
        )
        or gpu["0"] == gpu["1"]
        or not isinstance(semantic, str)
        or HEX64.fullmatch(semantic) is None
        or type(resource_policy) is not dict
        or trees["code_tree"].path
        != QUALIFICATION_ROOT / "reconstruction_v2"
        or trees["python_runtime_tree"].path != PYTHONHOME_ROOT
        or trees["official_python_runtime_tree"].path
        != PYTHONHOME_ROOT
        or trees["babel_dependency_tree"].path != BABEL_ROOT
        or trees["official_hipporag_tree"].path
        != OFFICIAL_HIPPORAG_ROOT
        or trees["official_base_dependency_tree"].path
        != OFFICIAL_BASE_ROOT
        or files["service_unit"].path
        != trees["code_tree"].path / SERVICE_RELATIVE_PATH
    ):
        raise QualificationContractError(
            "qualification runtime layout drifted"
        )
    return QualificationConfig(
        path=path,
        files=files,
        trees=trees,
        gpu_uuids={"0": gpu["0"], "1": gpu["1"]},
        encoder_model_semantic_sha256=semantic,
        resource_policy=resource_policy,
        self_sha256=supplied_self,
    )


@dataclass(frozen=True, slots=True)
class AttemptPaths:
    root: Path
    input: Path
    terminal: Path
    checks: Path
    agent: Path
    raw: Path
    hippo: Path

    @classmethod
    def for_attempt(cls, attempt_id: str) -> "AttemptPaths":
        if re.fullmatch(r"[0-9a-f]{32}-[0-9a-f]{12}", attempt_id) is None:
            raise QualificationContractError("attempt id drifted")
        root = QUALIFICATION_ROOT / "attempts" / attempt_id
        return cls(
            root=root,
            input=root / "synthetic.action_views.json",
            terminal=root / "terminal.safe.json",
            checks=root / "checks.safe.json",
            agent=root / "agent",
            raw=root / "raw",
            hippo=root / "hipporag",
        )


def lane_schema(lane: str) -> str:
    if lane not in {"Agent", "RAW", "HippoRAG"}:
        raise QualificationContractError("unknown lane")
    return f"{LANE_SCHEMA_PREFIX}_{lane.casefold()}_lane_safe_v1"


def invocation_id_from_environment(value: str | None) -> str:
    if not isinstance(value, str) or INVOCATION_ID.fullmatch(value) is None:
        raise QualificationContractError(
            "systemd InvocationID is absent or malformed"
        )
    return value


def attempt_id(invocation_id: str, config_self_sha256: str) -> str:
    invocation_id_from_environment(invocation_id)
    if HEX64.fullmatch(config_self_sha256) is None:
        raise QualificationContractError("config self hash drifted")
    return f"{invocation_id}-{config_self_sha256[:12]}"
