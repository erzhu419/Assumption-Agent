#!/usr/bin/env python3
"""Isolated Python actor for one admitted Q1 formal archive materialization.

The actor accepts only the 14-of-20 formal action.  In one process it measures
Odd then Sink exactly once through the existing full-node6 partition runner and
passes the returned immutable same-run snapshots directly to the formal archive
materializer.  It has no diagnostic-action compatibility route.

Execute in the bound Python endpoint image as ``/usr/local/bin/python3 -I -S
-B <this-file> --action materialize-full-node6-formal-archive-v1 ...``.
"""

from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Callable, Final, NoReturn


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_ROOT: Final = PROJECT_ROOT / "src"
PACKAGE_ROOT: Final = SOURCE_ROOT / "hegel_machine"
ACTION_ID: Final = "materialize-full-node6-formal-archive-v1"
IMPLEMENTATION_ID: Final = "hegel-python-q1-formal-archive-materializer-v1"
ERROR_SCHEMA_VERSION: Final = "hegel-q1-python-formal-archive-materializer-error/1"

MODULE_NAMES: Final = (
    "phase3_m3_bounded_enumerator_shrink2_v1",
    "phase3_m3_bounded_enumerator_shrink3_v1",
    "phase3_m3_bounded_enumerator_shrink4_v1",
    "phase3_m3_bounded_enumerator_shrink5_v1",
    "phase3_m3_bounded_enumerator_shrink6_v1",
    "phase3_m3_bounded_enumerator_v1",
    "phase3_m3_dsl_core_v1",
    "phase3_m3_record_wire_v1",
    "phase3_m3_shrink1_core_v1",
    "phase3_m3_shrink2_core_v1",
    "phase3_m3_shrink3_core_v1",
    "phase3_m3_shrink4_core_v1",
    "phase3_m3_shrink5_core_v1",
    "phase3_m3_shrink6_core_v1",
    "phase3_q0_evaluator_v1",
    "phase3_q0_input_adapter_v1",
    "phase3_q1_archive_projection_v1",
    "phase3_q1_capacity_preflight_v1",
    "phase3_q1_external_sort_profile_v1",
    "phase3_q1_formal_archive_contract_v1",
    "phase3_q1_full_node6_capacity_preflight_v1",
    "phase3_q1_full_node6_formal_materializer_v1",
    "phase3_q1_full_node6_projection_v1",
    "phase3_q1_partition_snapshot_v1",
    "phase3_q1_prestart_wire_v1",
    "phase3_q1_quotient_contract_v1",
    "phase3_q1_universe_v1",
    "strict_ast_shrink1_v1",
    "strict_ast_shrink2_v1",
    "strict_ast_shrink3_v1",
    "strict_ast_shrink4_v1",
    "strict_ast_shrink5_v1",
    "strict_ast_shrink6_v1",
    "strict_ast_v1",
    "strict_cbor_v1",
)
MODULE_SOURCE_PATHS: Final = tuple(
    f"src/hegel_machine/{name}.py" for name in MODULE_NAMES
)


class EndpointError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise EndpointError(code, detail)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail("FAIL_Q1_FORMAL_MATERIALIZER_JSON", str(error))


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            _fail("FAIL_Q1_FORMAL_MATERIALIZER_JSON", f"duplicate key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> NoReturn:
    _fail("FAIL_Q1_FORMAL_MATERIALIZER_JSON", f"non-finite token {value!r}")


def _canonical_file(path_text: str) -> Path:
    path = Path(path_text)
    try:
        status = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
    except OSError as error:
        _fail("REJECT_Q1_FORMAL_MATERIALIZER_PATH", str(error))
    if (
        not path.is_absolute()
        or resolved != path
        or path.is_symlink()
        or not stat.S_ISREG(status.st_mode)
    ):
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_PATH",
            "admission must be one canonical nonsymlink regular file",
        )
    return path


def _canonical_empty_directory(
    path_text: str,
    label: str,
) -> tuple[Path, tuple[int, int]]:
    path = Path(path_text)
    try:
        status = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
        nonempty = any(path.iterdir())
    except OSError as error:
        _fail("REJECT_Q1_FORMAL_MATERIALIZER_PATH", f"{label}: {error}")
    if (
        not path.is_absolute()
        or resolved != path
        or path.is_symlink()
        or not stat.S_ISDIR(status.st_mode)
        or nonempty
    ):
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_PATH",
            f"{label} must be one canonical nonsymlink empty directory",
        )
    probe = path / f".hegel-q1-formal-{label}-probe"
    try:
        with probe.open("xb", buffering=0):
            pass
        probe.unlink()
    except OSError as error:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_PATH",
            f"{label} write probe: {error}",
        )
    return path, (status.st_dev, status.st_ino)


def _parse_invocation(
    argv: list[str],
) -> tuple[Path, str, str, str, str, Path, Path]:
    if (
        len(argv) != 16
        or argv[0:2] != ["--action", ACTION_ID]
        or argv[2] != "--admission-json"
        or argv[4] != "--capsule-digest"
        or argv[6] != "--source-commit"
        or argv[8] != "--run-genesis-root"
        or argv[10] != "--prestart-admission-receipt-root"
        or argv[12] != "--output-dir"
        or argv[14] != "--scratch-dir"
    ):
        _fail(
            "Q1_FORMAL_MATERIALIZER_ACTION_NOT_ADMITTED",
            "only the exact formal action with admission, identities, output and scratch is admitted",
        )
    admission_path = _canonical_file(argv[3])
    output_root, output_identity = _canonical_empty_directory(
        argv[13],
        "output",
    )
    scratch_root, scratch_identity = _canonical_empty_directory(
        argv[15],
        "scratch",
    )
    if output_root == scratch_root or output_identity == scratch_identity:
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_PATH",
            "output and scratch directories alias",
        )
    return (
        admission_path,
        argv[5],
        argv[7],
        argv[9],
        argv[11],
        output_root,
        scratch_root,
    )


sys.path.insert(0, str(SOURCE_ROOT))
if "hegel_machine" in sys.modules:
    if __name__ == "__main__":
        _fail("FAIL_Q1_FORMAL_MATERIALIZER_ISOLATION", "package existed before bootstrap")
else:
    package = ModuleType("hegel_machine")
    package.__path__ = [str(PACKAGE_ROOT)]  # type: ignore[attr-defined]
    package.__package__ = "hegel_machine"
    sys.modules["hegel_machine"] = package

from hegel_machine import phase3_q1_full_node6_formal_materializer_v1 as materializer  # noqa: E402


def _loaded_project_modules() -> tuple[str, ...]:
    expected = {f"hegel_machine.{name}" for name in MODULE_NAMES}
    loaded: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("hegel_machine."):
            continue
        source = getattr(module, "__file__", None)
        if type(source) is not str:
            _fail("FAIL_Q1_FORMAL_MATERIALIZER_ISOLATION", f"{name} has no source")
        try:
            relative = Path(source).resolve().relative_to(PROJECT_ROOT).as_posix()
        except (OSError, ValueError):
            _fail("FAIL_Q1_FORMAL_MATERIALIZER_ISOLATION", f"{name} escaped project")
        if name not in expected or relative not in MODULE_SOURCE_PATHS:
            _fail(
                "FAIL_Q1_FORMAL_MATERIALIZER_ISOLATION",
                f"unallowlisted project module {name}",
            )
        loaded.add(name)
    if loaded != expected:
        _fail(
            "FAIL_Q1_FORMAL_MATERIALIZER_ISOLATION",
            f"module closure differs; missing={sorted(expected - loaded)}",
        )
    return tuple(sorted(loaded))


def _load_admission(
    path: Path,
    capsule_digest: str,
    source_commit: str,
    run_genesis_root: str,
    prestart_admission_receipt_root: str,
) -> materializer.Q1FormalArchiveMaterializerAdmissionV1:
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except EndpointError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("FAIL_Q1_FORMAL_MATERIALIZER_JSON", str(error))
    fields = tuple(
        materializer.Q1FormalArchiveMaterializerAdmissionV1.__dataclass_fields__
    )
    if type(value) is not dict or tuple(sorted(value)) != tuple(sorted(fields)):
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION",
            "formal admission key registry differs",
        )
    if type(value.get("q1_output_slots")) is not list:
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION",
            "q1_output_slots must be a JSON array",
        )
    value["q1_output_slots"] = tuple(value["q1_output_slots"])
    try:
        admission = materializer.Q1FormalArchiveMaterializerAdmissionV1(**value)
    except (TypeError, materializer.Q1FormalArchiveMaterializationError) as error:
        _fail("REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION", str(error))
    if (
        admission.capsule_digest != capsule_digest
        or admission.source_commit != source_commit
        or admission.run_genesis_root != run_genesis_root
        or admission.prestart_admission_receipt_root
        != prestart_admission_receipt_root
    ):
        _fail(
            "REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION",
            "CLI and admission identities differ",
        )
    return admission


def run_endpoint_v1(
    argv: list[str],
    *,
    partition_runner: Callable[..., tuple[object, object]] = (
        materializer._full._run_partition_once_v1
    ),
    projection_runner: Callable[..., object] = (
        materializer._projection.project_full_node6_same_run_v1
    ),
    module_verifier: Callable[[], object] = _loaded_project_modules,
) -> bytes:
    (
        admission_path,
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_admission_receipt_root,
        output_root,
        scratch_root,
    ) = _parse_invocation(argv)
    admission = _load_admission(
        admission_path,
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_admission_receipt_root,
    )
    module_verifier()
    result = materializer.run_full_node6_formal_materializer_v1(
        admission,
        output_root=output_root,
        scratch_root=scratch_root,
        partition_runner=partition_runner,
        projection_runner=projection_runner,
    )
    if tuple(scratch_root.iterdir()) != ():
        _fail(
            "FAIL_Q1_FORMAL_MATERIALIZER_SCRATCH",
            "transient scratch remains at terminal",
        )
    return result.endpoint_result_bytes


def _error_object(error: BaseException) -> dict[str, object]:
    return {
        "diagnostic_only": False,
        "error_code": getattr(error, "code", "FAIL_Q1_FORMAL_MATERIALIZER"),
        "error_detail": getattr(error, "detail", str(error)),
        "formal_archive_materialized": False,
        "q1_output_slots": [None] * 8,
        "q1_state": "NOT_RUN",
        "role_evaluation_performed": False,
        "schema_version": ERROR_SCHEMA_VERSION,
        "split_accessed": False,
        "target_truth_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run_endpoint_v1(sys.argv[1:] if argv is None else argv)
        code = 0
    except BaseException as error:
        payload = _canonical_json_bytes(_error_object(error))
        code = 1
    try:
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
    except OSError:
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
