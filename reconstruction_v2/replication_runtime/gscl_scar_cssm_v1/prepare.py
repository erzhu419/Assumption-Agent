"""One-shot private source materialization for the SCAR CSSM study.

This stage has no model, action former, scorer, network, or evaluator
capability.  It creates one opaque secret, opens the pinned private SCAR copy
through the frozen compiler exactly once, and durably separates action and
label packs.  Any failure after the preparation sentinel is terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import gscl_scar_cssm_source_v1 as source


VERSION = "gscl_scar_cssm_prepare_v1"
ATTEMPT_SCHEMA = f"{VERSION}.attempt.v1"
TERMINAL_SCHEMA = f"{VERSION}.safe_terminal.v1"
_STUDY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ScarCssmPrepareError(RuntimeError):
    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ScarCssmPrepareError("PREPARE_CANONICAL_JSON_INVALID") from exc


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, raw: bytes) -> Mapping[str, object]:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ScarCssmPrepareError("PREPARE_OUTPUT_ALREADY_EXISTS")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return MappingProxyType(
        {
            "mode_octal": "0600",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    )


def _mount_filesystem_type(path: Path) -> str:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        rows = Path("/proc/self/mountinfo").read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    except OSError as exc:
        raise ScarCssmPrepareError("PREPARE_ROOT_INVALID") from exc
    matches: list[tuple[int, str]] = []
    for line in rows:
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, suffix = left.split(), right.split()
        if len(fields) < 5 or not suffix:
            continue
        mount = Path(fields[4].replace("\\040", " ").replace("\\134", "\\"))
        try:
            absolute.relative_to(mount)
        except ValueError:
            continue
        matches.append((len(mount.parts), suffix[0]))
    if not matches:
        raise ScarCssmPrepareError("PREPARE_ROOT_INVALID")
    return max(matches)[1]


def _require_empty_private_root(path: Path) -> None:
    try:
        metadata = path.lstat()
        entries = tuple(path.iterdir())
    except OSError as exc:
        raise ScarCssmPrepareError("PREPARE_ROOT_INVALID") from exc
    if (
        not path.is_absolute()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or _mount_filesystem_type(path) != "ext4"
        or entries
    ):
        raise ScarCssmPrepareError("PREPARE_ROOT_INVALID")


def _default_secret_factory() -> bytes:
    try:
        return os.getrandom(source.HMAC_SECRET_BYTES)
    except (AttributeError, OSError) as exc:
        raise ScarCssmPrepareError("PREPARE_SECRET_GENERATION_FAILED") from exc


def prepare_once(
    *,
    source_path: Path,
    output_root: Path,
    study_id: str,
    secret_factory: Callable[[], bytes] = _default_secret_factory,
    compiler: Callable[..., source.ScarCssmSourceCompilation] = (
        source.compile_scar_cssm_source_v1
    ),
) -> Mapping[str, object]:
    if (
        not isinstance(source_path, Path)
        or not source_path.is_absolute()
        or not isinstance(study_id, str)
        or _STUDY_ID.fullmatch(study_id) is None
    ):
        raise ScarCssmPrepareError("PREPARE_COORDINATE_INVALID")
    _require_empty_private_root(output_root)
    paths = {
        "action_pack": output_root / "action_pack.private.json",
        "label_pack": output_root / "label_pack.private.json",
        "safe_source": output_root / "source.safe.json",
        "secret": output_root / "compiler_secret.private.bin",
        "attempt": output_root / "prepare.attempt.sentinel",
        "terminal": output_root / "prepare.terminal.safe.json",
    }
    attempt_body = {
        "schema": ATTEMPT_SCHEMA,
        "source_expected_sha256": source.EXPECTED_SOURCE_SHA256,
        "study_id": study_id,
        "version": VERSION,
    }
    attempt = {**attempt_body, "self_sha256": _content_hash(attempt_body)}
    attempt_receipt = _publish_once(paths["attempt"], _canonical_bytes(attempt))

    secret = secret_factory()
    if type(secret) is not bytes or len(secret) != source.HMAC_SECRET_BYTES:
        raise ScarCssmPrepareError("PREPARE_SECRET_GENERATION_FAILED")
    secret_receipt = _publish_once(paths["secret"], secret)

    try:
        compilation = compiler(
            source_path=source_path,
            secret=secret,
            study_id=study_id,
        )
    except Exception as exc:
        raise ScarCssmPrepareError("PREPARE_SOURCE_COMPILATION_FAILED") from exc
    if type(compilation) is not source.ScarCssmSourceCompilation:
        raise ScarCssmPrepareError("PREPARE_SOURCE_COMPILATION_INVALID")
    action_raw = _canonical_bytes(compilation.action_pack)
    label_raw = _canonical_bytes(compilation.label_pack)
    safe_raw = _canonical_bytes(compilation.safe_aggregate)
    action_receipt = _publish_once(paths["action_pack"], action_raw)
    label_receipt = _publish_once(paths["label_pack"], label_raw)
    safe_receipt = _publish_once(paths["safe_source"], safe_raw)

    terminal_body = {
        "action_pack": dict(action_receipt),
        "action_pack_commitment_sha256": compilation.action_pack[
            "action_commitment_sha256"
        ],
        "attempt": dict(attempt_receipt),
        "external_network_call_count": 0,
        "formal_source_access_count": 1,
        "label_pack": dict(label_receipt),
        "model_action_or_scorer_call_count": 0,
        "online_or_api_evaluator_call_count": 0,
        "safe_source_aggregate": dict(safe_receipt),
        "schema": TERMINAL_SCHEMA,
        "secret": dict(secret_receipt),
        "source_expected_sha256": source.EXPECTED_SOURCE_SHA256,
        "status": "complete_private_action_label_separation",
        "study_id": study_id,
        "version": VERSION,
    }
    terminal = {
        **terminal_body,
        "self_sha256": _content_hash(terminal_body),
    }
    _publish_once(paths["terminal"], _canonical_bytes(terminal))
    return MappingProxyType(terminal)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize one private SCAR CSSM source capability"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--study-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    terminal = prepare_once(
        source_path=arguments.source,
        output_root=arguments.output_root,
        study_id=arguments.study_id,
    )
    print(terminal["self_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_SCHEMA",
    "ScarCssmPrepareError",
    "TERMINAL_SCHEMA",
    "VERSION",
    "prepare_once",
]
