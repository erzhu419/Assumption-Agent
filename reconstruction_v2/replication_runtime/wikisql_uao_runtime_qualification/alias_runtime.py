"""Fail-closed short-model-alias binding for WikiSQL runtime qualification.

This module is deliberately independent of every formal-study and canary
version.  It does not load a model, inspect a benchmark source, or construct a
HippoRAG core.  It only binds two already-verified direct model directories to
the fixed cwd-local aliases required by the pinned HippoRAG path convention.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Callable


SCHEMA = "wikisql_uao_short_model_alias_runtime_receipt_v1"
ALIAS_DIRECTORY = "model_aliases"
LLM_ALIAS = "smollm2"
EMBEDDING_ALIAS = "minilm"
DERIVED_HIPPORAG_COMPONENT = (
    f"Transformers_{LLM_ALIAS}_Transformers_{EMBEDDING_ALIAS}"
)
DERIVED_HIPPORAG_COMPONENT_UTF8_BYTES = 40

TreeIdentityFn = Callable[[Path], object]


class WikiSQLUAOAliasRuntimeError(RuntimeError):
    """The short-alias runtime binding failed closed."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise WikiSQLUAOAliasRuntimeError(
            "tree identity or alias receipt is not serializable"
        ) from exc


def _json_projection(value: object) -> object:
    raw = _canonical_json_bytes(value)
    try:
        return json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WikiSQLUAOAliasRuntimeError(
            "tree identity JSON projection failed"
        ) from exc


def _direct_absolute_directory(value: Path, label: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise WikiSQLUAOAliasRuntimeError(
            f"{label} must be an absolute Path"
        )
    try:
        metadata = value.lstat()
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise WikiSQLUAOAliasRuntimeError(
            f"{label} is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != value
    ):
        raise WikiSQLUAOAliasRuntimeError(
            f"{label} must be a direct canonical directory"
        )
    return value


def _tree_identity(
    identity_fn: TreeIdentityFn,
    target: Path,
    label: str,
) -> object:
    if not callable(identity_fn):
        raise WikiSQLUAOAliasRuntimeError(
            "tree identity function is unavailable"
        )
    try:
        observed = identity_fn(target)
    except Exception as exc:
        raise WikiSQLUAOAliasRuntimeError(
            f"{label} tree identity failed"
        ) from exc
    return _json_projection(observed)


def _path_sha256(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def bind_and_verify_short_model_aliases(
    *,
    writable_root: Path,
    llm_model_root: Path,
    embedding_model_root: Path,
    identity_fn: TreeIdentityFn,
) -> dict[str, object]:
    """Create and verify the two fixed short aliases exactly once.

    ``identity_fn`` must return a JSON-serializable complete tree identity.
    It is called on each direct target immediately before alias creation and
    again after all link, cwd-component, and ``NAME_MAX`` checks.  Any content
    drift leaves the newly created alias root in place and fails closed.
    """

    root = _direct_absolute_directory(writable_root, "writable root")
    targets = (
        (LLM_ALIAS, _direct_absolute_directory(
            llm_model_root, "LLM model root"
        )),
        (EMBEDDING_ALIAS, _direct_absolute_directory(
            embedding_model_root, "embedding model root"
        )),
    )
    try:
        targets_are_distinct = not os.path.samefile(
            targets[0][1], targets[1][1]
        )
    except OSError as exc:
        raise WikiSQLUAOAliasRuntimeError(
            "model target identity is unavailable"
        ) from exc
    if not targets_are_distinct:
        raise WikiSQLUAOAliasRuntimeError(
            "LLM and embedding model targets must be distinct"
        )

    identities_before = {
        alias: _tree_identity(identity_fn, target, alias)
        for alias, target in targets
    }

    alias_root = root / ALIAS_DIRECTORY
    if alias_root.exists() or alias_root.is_symlink():
        raise WikiSQLUAOAliasRuntimeError(
            "model alias root is not fresh"
        )
    try:
        alias_root.mkdir(mode=0o700)
        alias_root_metadata = alias_root.lstat()
        alias_root_resolved = alias_root.resolve(strict=True)
    except OSError as exc:
        raise WikiSQLUAOAliasRuntimeError(
            "model alias root cannot be created"
        ) from exc
    if (
        stat.S_ISLNK(alias_root_metadata.st_mode)
        or not stat.S_ISDIR(alias_root_metadata.st_mode)
        or stat.S_IMODE(alias_root_metadata.st_mode) != 0o700
        or alias_root_resolved != alias_root
    ):
        raise WikiSQLUAOAliasRuntimeError(
            "model alias root mode or identity drifted"
        )

    aliases: dict[str, object] = {}
    for alias, target in targets:
        alias_path = alias_root / alias
        try:
            os.symlink(
                str(target),
                alias_path,
                target_is_directory=True,
            )
            alias_metadata = alias_path.lstat()
            link_target = os.readlink(alias_path)
            resolved = alias_path.resolve(strict=True)
            samefile = os.path.samefile(alias_path, target)
        except OSError as exc:
            raise WikiSQLUAOAliasRuntimeError(
                f"{alias} model alias cannot be bound"
            ) from exc
        if (
            not stat.S_ISLNK(alias_metadata.st_mode)
            or link_target != str(target)
            or resolved != target
            or samefile is not True
        ):
            raise WikiSQLUAOAliasRuntimeError(
                f"{alias} model alias binding drifted"
            )
        aliases[alias] = {
            "alias_is_single_relative_component": True,
            "link_target_path_sha256": _path_sha256(target),
            "resolved_path_sha256": _path_sha256(resolved),
            "samefile": True,
            "tree_identity": identities_before[alias],
        }

    component_bytes = len(DERIVED_HIPPORAG_COMPONENT.encode("utf-8"))
    if (
        DERIVED_HIPPORAG_COMPONENT
        != "Transformers_smollm2_Transformers_minilm"
        or component_bytes != DERIVED_HIPPORAG_COMPONENT_UTF8_BYTES
    ):
        raise WikiSQLUAOAliasRuntimeError(
            "fixed HippoRAG working component drifted"
        )
    try:
        name_max = os.pathconf(alias_root, "PC_NAME_MAX")
    except (OSError, ValueError) as exc:
        raise WikiSQLUAOAliasRuntimeError(
            "filesystem NAME_MAX is unavailable"
        ) from exc
    if (
        isinstance(name_max, bool)
        or not isinstance(name_max, int)
        or name_max < component_bytes
    ):
        raise WikiSQLUAOAliasRuntimeError(
            "short HippoRAG working component exceeds NAME_MAX"
        )

    identities_after = {
        alias: _tree_identity(identity_fn, target, alias)
        for alias, target in targets
    }
    if identities_after != identities_before:
        raise WikiSQLUAOAliasRuntimeError(
            "model tree identity changed during alias binding"
        )

    body: dict[str, object] = {
        "alias_directory": ALIAS_DIRECTORY,
        "alias_root_mode_octal": "0700",
        "aliases": aliases,
        "derived_hipporag_component": DERIVED_HIPPORAG_COMPONENT,
        "derived_hipporag_component_utf8_bytes": component_bytes,
        "filesystem_name_max_bytes": name_max,
        "model_content_changed": False,
        "schema": SCHEMA,
        "status": "short_model_aliases_bound_and_verified",
    }
    receipt = {
        **body,
        "self_sha256": hashlib.sha256(
            _canonical_json_bytes(body)
        ).hexdigest(),
    }
    _canonical_json_bytes(receipt)
    return receipt


__all__ = [
    "ALIAS_DIRECTORY",
    "DERIVED_HIPPORAG_COMPONENT",
    "DERIVED_HIPPORAG_COMPONENT_UTF8_BYTES",
    "EMBEDDING_ALIAS",
    "LLM_ALIAS",
    "SCHEMA",
    "WikiSQLUAOAliasRuntimeError",
    "bind_and_verify_short_model_aliases",
]
