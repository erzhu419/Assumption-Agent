from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..events import Event, EventSink, NullEventSink
from ..models import ActionNode, HypothesisProgram, HypothesisStatus, stable_hash
from ..splits import BenchmarkItem, SplitManifest
from ..typed_execution_contract import (
    InvariantKind,
    TypedExecutionContract,
    TypedExecutionContractRegistry,
)
from ..typed_operator_grammar import (
    BoundTypedRecipe,
    TypedProgramBindingRegistry,
    is_typed_recipe_materialization,
)
from ..validation import backend_action_contract_issues
from .typed_task_capability import (
    PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
    CompiledPortableTaskCapability,
    build_compiled_portable_task_capability,
    deterministic_portable_capability_output_locator,
    portable_role_spec_for_bound_recipe,
    validate_compiled_portable_task_capability,
)


SKILL_ROUTING_VERSION = "per_item_trigger_routing_v2"
LEGACY_SKILL_ACTION_LOWERING_VERSION = "skilllearn_prompt_directive_lowering_v1"
SKILL_ACTION_LOWERING_VERSION = "skilllearn_prompt_directive_lowering_v2"
SKILL_FALLBACK_SEMANTICS_VERSION = "baseline_on_nonactivation_only_v1"
SKILLLEARN_ALLOWED_ACTION_OPERATIONS = frozenset(
    {"execute_step", "check_condition", "produce_artifact", "request_evidence"}
)
NO_SKILL_TREATMENT_HASH = stable_hash(
    {
        "routing_version": SKILL_ROUTING_VERSION,
        "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
        "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
        "skill_contents": [],
    }
)


@dataclass(frozen=True)
class LoweredSkillAction:
    action_id: str
    semantics: str
    instruction: str

    def to_dict(self) -> dict[str, str]:
        return {
            "action_id": self.action_id,
            "semantics": self.semantics,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class SkillCompileResult:
    output_root: Path
    skill_paths: tuple[Path, ...]
    family_count: int
    hypothesis_ids: tuple[str, ...]
    manifest_hash: str
    item_sources: Mapping[str, Path]
    program_set_hash: str
    treatment_hash: str
    item_treatment_hashes: Mapping[str, str]
    typed_binding_hashes: tuple[str, ...] = ()
    typed_binding_set_hash: str = ""
    typed_snapshot_hashes: tuple[str, ...] = ()
    typed_snapshot_ledger_hash: str = ""
    portable_capability_compiler_mode: str = ""
    portable_capability_role_spec_set_hash: str = ""
    item_portable_capability_role_spec_hashes: Mapping[
        str, tuple[str, ...]
    ] = field(default_factory=dict)
    item_portable_capability_metadata_paths: Mapping[
        str, tuple[Path, ...]
    ] = field(default_factory=dict)
    typed_execution_contract_hashes: tuple[str, ...] = ()

    def source_for(self, item_id: str) -> Path | None:
        return self.item_sources.get(stable_hash({"item_id": item_id}))

    def treatment_hash_for(self, item_id: str) -> str:
        return self.item_treatment_hashes.get(
            stable_hash({"item_id": item_id}),
            NO_SKILL_TREATMENT_HASH,
        )

    def source_receipt_for(self, item_id: str) -> "SkillSourceReceipt":
        return verify_compiled_skill_source(
            compile_root=self.output_root,
            item_id=item_id,
            skill_source_dir=self.source_for(item_id),
            expected_compile_manifest_hash=self.manifest_hash,
            expected_program_set_hash=self.program_set_hash,
            expected_treatment_hash=self.treatment_hash_for(item_id),
            expected_typed_binding_set_hash=self.typed_binding_set_hash,
            expected_typed_snapshot_hashes=self.typed_snapshot_hashes,
            expected_typed_snapshot_ledger_hash=(
                self.typed_snapshot_ledger_hash
            ),
            expected_portable_capability_compiler_mode=(
                self.portable_capability_compiler_mode
            ),
            expected_portable_capability_role_spec_set_hash=(
                self.portable_capability_role_spec_set_hash
            ),
            expected_portable_capability_role_spec_hashes=(
                self.item_portable_capability_role_spec_hashes.get(
                    stable_hash({"item_id": item_id}),
                    (),
                )
            ),
        )


@dataclass(frozen=True)
class SkillSourceReceipt:
    """Content-addressed compiler-to-runtime treatment receipt for one item."""

    compile_manifest_hash: str
    item_id_hash: str
    source_route: str | None
    source_file_hashes: tuple[tuple[str, str], ...]
    source_tree_hash: str
    program_set_hash: str
    treatment_hash: str
    typed_binding_set_hash: str
    typed_snapshot_hashes: tuple[str, ...]
    typed_snapshot_ledger_hash: str
    portable_capability_compiler_mode: str = ""
    portable_capability_role_spec_set_hash: str = ""
    portable_capability_role_spec_hashes: tuple[str, ...] = ()
    portable_capability_metadata_file_hashes: tuple[
        tuple[str, str], ...
    ] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "compile_manifest_hash": self.compile_manifest_hash,
            "item_id_hash": self.item_id_hash,
            "source_route": self.source_route,
            "source_file_hashes": [
                {"path": path, "sha256": sha256}
                for path, sha256 in self.source_file_hashes
            ],
            "source_tree_hash": self.source_tree_hash,
            "program_set_hash": self.program_set_hash,
            "treatment_hash": self.treatment_hash,
            "typed_binding_set_hash": self.typed_binding_set_hash,
            "typed_snapshot_hashes": list(self.typed_snapshot_hashes),
            "typed_snapshot_ledger_hash": self.typed_snapshot_ledger_hash,
        }
        if self.portable_capability_compiler_mode:
            payload.update(
                {
                    "portable_capability_compiler_mode": (
                        self.portable_capability_compiler_mode
                    ),
                    "portable_capability_role_spec_set_hash": (
                        self.portable_capability_role_spec_set_hash
                    ),
                    "portable_capability_role_spec_hashes": list(
                        self.portable_capability_role_spec_hashes
                    ),
                    "portable_capability_metadata_file_hashes": [
                        {"path": path, "sha256": sha256}
                        for path, sha256 in (
                            self.portable_capability_metadata_file_hashes
                        )
                    ],
                    "portable_capability_metadata_tree_hash": stable_hash(
                        {
                            "files": [
                                {"path": path, "sha256": sha256}
                                for path, sha256 in (
                                    self.portable_capability_metadata_file_hashes
                                )
                            ]
                        }
                    ),
                }
            )
        return payload

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class SkillSourceTreeReceipt:
    """Path-independent byte receipt for a non-compiler skill source tree."""

    source_file_hashes: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file_hashes": [
                {"path": path, "sha256": sha256}
                for path, sha256 in self.source_file_hashes
            ],
            "source_tree_hash": self.source_tree_hash,
        }

    @property
    def source_tree_hash(self) -> str:
        return stable_hash(
            {
                "files": [
                    {"path": path, "sha256": sha256}
                    for path, sha256 in self.source_file_hashes
                ]
            }
        )

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.to_dict())


def verify_skill_source_tree(
    source_root: str | Path,
) -> SkillSourceTreeReceipt:
    """Hash every regular source byte and reject links or special files."""

    source_path = Path(source_root).expanduser()
    if source_path.is_symlink() or not source_path.is_dir():
        raise PermissionError("skill source tree must be a real directory")
    source = source_path.resolve(strict=True)
    rows: list[tuple[str, str]] = []
    for path in source.rglob("*"):
        if path.is_symlink():
            raise PermissionError("skill source tree contains a link")
        if path.is_file():
            rows.append(
                (
                    path.relative_to(source).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        elif not path.is_dir():
            raise PermissionError("skill source tree contains a special file")
    if not rows:
        raise PermissionError("skill source tree is empty")
    return SkillSourceTreeReceipt(tuple(sorted(rows)))


def verify_compiled_skill_source(
    *,
    compile_root: str | Path,
    item_id: str,
    skill_source_dir: str | Path | None,
    expected_compile_manifest_hash: str,
    expected_program_set_hash: str,
    expected_treatment_hash: str,
    expected_typed_binding_set_hash: str = "",
    expected_typed_snapshot_hashes: Sequence[str] = (),
    expected_typed_snapshot_ledger_hash: str = "",
    expected_portable_capability_compiler_mode: str = "",
    expected_portable_capability_role_spec_set_hash: str = "",
    expected_portable_capability_role_spec_hashes: Sequence[str] = (),
) -> SkillSourceReceipt:
    """Verify the exact compiled item tree immediately before runtime use.

    The receipt intentionally excludes host paths.  It binds the manifest, item
    route, every source byte (through raw SHA-256 file hashes), treatment, and
    the complete typed-selection provenance surface.  The manifest's canonical
    text hashes are independently rechecked when recomputing treatment identity.
    """

    root_path = Path(compile_root).expanduser()
    if root_path.is_symlink() or not root_path.is_dir():
        raise PermissionError("compiled skill root must be a real directory")
    root = root_path.resolve(strict=True)
    manifest_path = root / "compile_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PermissionError("compiled skill manifest is missing or linked")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PermissionError("compiled skill manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise PermissionError("compiled skill manifest must be an object")
    manifest_hash = stable_hash(manifest)
    if manifest_hash != expected_compile_manifest_hash:
        raise PermissionError("compiled skill manifest hash mismatch")
    if manifest.get("program_set_hash") != expected_program_set_hash:
        raise PermissionError("compiled skill program set mismatch")

    typed_rows = manifest.get("typed_binding_rows")
    if not isinstance(typed_rows, list):
        raise PermissionError("compiled skill typed binding rows are malformed")
    manifest_binding_set_hash = str(
        manifest.get("typed_binding_set_hash") or ""
    )
    recomputed_binding_set_hash = (
        stable_hash({"bindings": typed_rows}) if typed_rows else ""
    )
    if manifest_binding_set_hash != recomputed_binding_set_hash:
        raise PermissionError("compiled skill typed binding set is inconsistent")
    if manifest_binding_set_hash != expected_typed_binding_set_hash:
        raise PermissionError("compiled skill typed binding set mismatch")
    manifest_snapshot_hashes = manifest.get("typed_snapshot_hashes")
    if not isinstance(manifest_snapshot_hashes, list) or any(
        not isinstance(value, str) for value in manifest_snapshot_hashes
    ):
        raise PermissionError("compiled skill snapshot hashes are malformed")
    if tuple(manifest_snapshot_hashes) != tuple(expected_typed_snapshot_hashes):
        raise PermissionError("compiled skill snapshot hashes mismatch")
    if str(manifest.get("typed_snapshot_ledger_hash") or "") != (
        expected_typed_snapshot_ledger_hash
    ):
        raise PermissionError("compiled skill snapshot ledger mismatch")

    item_hash = stable_hash({"item_id": item_id})
    (
        portable_role_spec_hashes,
        portable_metadata_file_hashes,
    ) = _verify_portable_capability_item_metadata(
        root=root,
        manifest=manifest,
        item_id_hash=item_hash,
        expected_compiler_mode=(
            expected_portable_capability_compiler_mode
        ),
        expected_role_spec_set_hash=(
            expected_portable_capability_role_spec_set_hash
        ),
        expected_role_spec_hashes=tuple(
            expected_portable_capability_role_spec_hashes
        ),
    )
    item_routes = manifest.get("item_routes")
    item_treatments = manifest.get("item_treatment_hashes")
    skill_paths = manifest.get("skill_paths")
    content_hashes = manifest.get("skill_content_hashes")
    if not isinstance(item_routes, dict) or not isinstance(item_treatments, dict):
        raise PermissionError("compiled skill item routing is malformed")
    if not isinstance(skill_paths, list) or not isinstance(content_hashes, dict):
        raise PermissionError("compiled skill content index is malformed")
    if item_hash not in item_routes or item_hash not in item_treatments:
        raise PermissionError("compiled skill item is outside the manifest")
    source_route_value = item_routes[item_hash]
    if source_route_value is not None and not isinstance(source_route_value, str):
        raise PermissionError("compiled skill source route is malformed")
    source_route = source_route_value
    canonical_route = str(Path("items") / item_hash)
    if source_route not in {None, canonical_route}:
        raise PermissionError("compiled skill source route escaped its item")
    if item_treatments[item_hash] != expected_treatment_hash:
        raise PermissionError("compiled skill treatment hash mismatch")

    expected_rows: list[tuple[str, str]] = []
    if source_route is not None:
        prefix = f"{source_route}/"
        for raw_path in skill_paths:
            if not isinstance(raw_path, str):
                raise PermissionError("compiled skill path is malformed")
            if not raw_path.startswith(prefix):
                continue
            relative_path = raw_path[len(prefix) :]
            parts = Path(relative_path).parts
            if (
                not parts
                or Path(relative_path).is_absolute()
                or ".." in parts
                or parts[-1] != "SKILL.md"
            ):
                raise PermissionError("compiled skill item path is unsafe")
            content_hash = content_hashes.get(raw_path)
            if not isinstance(content_hash, str):
                raise PermissionError("compiled skill content hash is missing")
            expected_rows.append((relative_path, content_hash))
    expected_rows.sort()

    if source_route is None:
        if skill_source_dir is not None:
            raise PermissionError("no-skill manifest item received a source tree")
        if expected_rows or expected_treatment_hash != NO_SKILL_TREATMENT_HASH:
            raise PermissionError("no-skill manifest item has a nonempty treatment")
        actual_treatment_rows: tuple[tuple[str, str], ...] = ()
        actual_receipt_rows: tuple[tuple[str, str], ...] = ()
    else:
        if skill_source_dir is None:
            raise PermissionError("compiled treatment source tree is missing")
        source_path = Path(skill_source_dir).expanduser()
        if source_path.is_symlink() or not source_path.is_dir():
            raise PermissionError("compiled treatment source must be a real directory")
        source = source_path.resolve(strict=True)
        expected_source = (root / source_route).resolve(strict=True)
        if source != expected_source:
            raise PermissionError("compiled treatment source route mismatch")
        actual_files: list[Path] = []
        for path in source.rglob("*"):
            if path.is_symlink():
                raise PermissionError("compiled treatment source contains a link")
            if path.is_file():
                actual_files.append(path)
            elif not path.is_dir():
                raise PermissionError("compiled treatment source has a special file")
        actual_relative_paths = tuple(
            sorted(path.relative_to(source).as_posix() for path in actual_files)
        )
        expected_relative_paths = tuple(path for path, _ in expected_rows)
        if actual_relative_paths != expected_relative_paths:
            raise PermissionError("compiled treatment source tree is not exact")
        actual_treatment_row_list: list[tuple[str, str]] = []
        actual_receipt_row_list: list[tuple[str, str]] = []
        expected_by_path = dict(expected_rows)
        for relative_path in actual_relative_paths:
            path = source / relative_path
            try:
                raw_content = path.read_bytes()
                text = raw_content.decode("utf-8")
            except (OSError, UnicodeError) as exc:
                raise PermissionError("compiled treatment source is unreadable") from exc
            content_hash = stable_hash({"content": text})
            if content_hash != expected_by_path[relative_path]:
                raise PermissionError("compiled treatment source content mismatch")
            actual_treatment_row_list.append((relative_path, content_hash))
            actual_receipt_row_list.append(
                (relative_path, hashlib.sha256(raw_content).hexdigest())
            )
        actual_treatment_rows = tuple(actual_treatment_row_list)
        actual_receipt_rows = tuple(actual_receipt_row_list)

    treatment_payload = {
        "routing_version": SKILL_ROUTING_VERSION,
        "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
        "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
        "skill_content_hashes": sorted(
            content_hash for _, content_hash in actual_treatment_rows
        ),
    }
    if expected_portable_capability_compiler_mode:
        treatment_payload.update(
            {
                "portable_capability_compiler_mode": (
                    expected_portable_capability_compiler_mode
                ),
                "portable_capability_role_spec_hashes": list(
                    portable_role_spec_hashes
                ),
                "portable_capability_metadata_content_hashes": sorted(
                    row["metadata_content_hash"]
                    for row in manifest[
                        "portable_capability_role_spec_rows"
                    ]
                    if row["item_id_hash"] == item_hash
                ),
            }
        )
    recomputed_treatment_hash = (
        stable_hash(treatment_payload)
        if actual_treatment_rows
        else NO_SKILL_TREATMENT_HASH
    )
    if recomputed_treatment_hash != expected_treatment_hash:
        raise PermissionError("compiled treatment content does not match treatment hash")
    source_tree_hash = stable_hash(
        {
            "item_id_hash": item_hash,
            "files": [
                {"path": path, "sha256": sha256}
                for path, sha256 in actual_receipt_rows
            ],
        }
    )
    return SkillSourceReceipt(
        compile_manifest_hash=manifest_hash,
        item_id_hash=item_hash,
        source_route=source_route,
        source_file_hashes=actual_receipt_rows,
        source_tree_hash=source_tree_hash,
        program_set_hash=expected_program_set_hash,
        treatment_hash=expected_treatment_hash,
        typed_binding_set_hash=expected_typed_binding_set_hash,
        typed_snapshot_hashes=tuple(expected_typed_snapshot_hashes),
        typed_snapshot_ledger_hash=expected_typed_snapshot_ledger_hash,
        portable_capability_compiler_mode=(
            expected_portable_capability_compiler_mode
        ),
        portable_capability_role_spec_set_hash=(
            expected_portable_capability_role_spec_set_hash
        ),
        portable_capability_role_spec_hashes=portable_role_spec_hashes,
        portable_capability_metadata_file_hashes=(
            portable_metadata_file_hashes
        ),
    )


_PORTABLE_CAPABILITY_MANIFEST_KEYS = frozenset(
    {
        "portable_capability_compiler_mode",
        "portable_capability_role_spec_rows",
        "portable_capability_role_spec_set_hash",
        "item_portable_capability_role_spec_hashes",
        "item_portable_capability_metadata_paths",
        "source_artifact_locators_persisted",
    }
)


def _verify_portable_capability_item_metadata(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    item_id_hash: str,
    expected_compiler_mode: str,
    expected_role_spec_set_hash: str,
    expected_role_spec_hashes: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    present_keys = _PORTABLE_CAPABILITY_MANIFEST_KEYS.intersection(manifest)
    if not expected_compiler_mode:
        if present_keys:
            raise PermissionError(
                "compiled skill has unexpected portable capability metadata"
            )
        if expected_role_spec_set_hash or expected_role_spec_hashes:
            raise PermissionError(
                "portable capability receipt is partial"
            )
        return (), ()
    if expected_compiler_mode != PORTABLE_TASK_CAPABILITY_COMPILER_VERSION:
        raise PermissionError("portable capability compiler mode is unsupported")
    if present_keys != _PORTABLE_CAPABILITY_MANIFEST_KEYS:
        raise PermissionError(
            "compiled skill portable capability manifest is partial"
        )
    if manifest.get("portable_capability_compiler_mode") != (
        expected_compiler_mode
    ):
        raise PermissionError("portable capability compiler mode mismatch")
    if manifest.get("source_artifact_locators_persisted") is not False:
        raise PermissionError("portable capability source locator was persisted")

    raw_rows = manifest.get("portable_capability_role_spec_rows")
    raw_role_map = manifest.get(
        "item_portable_capability_role_spec_hashes"
    )
    raw_path_map = manifest.get("item_portable_capability_metadata_paths")
    if (
        not isinstance(raw_rows, list)
        or not isinstance(raw_role_map, dict)
        or not isinstance(raw_path_map, dict)
    ):
        raise PermissionError(
            "compiled skill portable capability index is malformed"
        )
    row_keys = {
        "item_id_hash",
        "program_id_hash",
        "typed_binding_hash",
        "bound_recipe_hash",
        "role_spec_hash",
        "metadata_hash",
        "metadata_path",
        "metadata_content_hash",
    }
    rows: list[dict[str, str]] = []
    for raw_row in raw_rows:
        if (
            not isinstance(raw_row, dict)
            or set(raw_row) != row_keys
            or any(not isinstance(value, str) for value in raw_row.values())
        ):
            raise PermissionError(
                "compiled skill portable capability row is malformed"
            )
        row = dict(raw_row)
        for key in row_keys - {"metadata_path"}:
            if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise PermissionError(
                    "compiled skill portable capability hash is malformed"
                )
        expected_path = str(
            Path("task_capabilities")
            / row["item_id_hash"]
            / f'{row["program_id_hash"]}.json'
        )
        if row["metadata_path"] != expected_path:
            raise PermissionError(
                "compiled skill portable capability path is not canonical"
            )
        rows.append(row)
    canonical_rows = sorted(
        rows,
        key=lambda row: (
            row["item_id_hash"],
            row["program_id_hash"],
            row["metadata_path"],
        ),
    )
    if rows != canonical_rows or len(
        {row["metadata_path"] for row in rows}
    ) != len(rows):
        raise PermissionError(
            "compiled skill portable capability rows are not canonical"
        )
    role_spec_set_hash = stable_hash({"rows": rows})
    if (
        manifest.get("portable_capability_role_spec_set_hash")
        != role_spec_set_hash
        or role_spec_set_hash != expected_role_spec_set_hash
    ):
        raise PermissionError(
            "compiled skill portable capability set hash mismatch"
        )

    item_keys = set(manifest.get("item_routes") or {})
    if set(raw_role_map) != item_keys or set(raw_path_map) != item_keys:
        raise PermissionError(
            "compiled skill portable capability item coverage is malformed"
        )
    computed_role_map = {
        item_hash: sorted(
            row["role_spec_hash"]
            for row in rows
            if row["item_id_hash"] == item_hash
        )
        for item_hash in sorted(item_keys)
    }
    computed_path_map = {
        item_hash: sorted(
            row["metadata_path"]
            for row in rows
            if row["item_id_hash"] == item_hash
        )
        for item_hash in sorted(item_keys)
    }
    if raw_role_map != computed_role_map or raw_path_map != computed_path_map:
        raise PermissionError(
            "compiled skill portable capability item index drifted"
        )
    item_role_hashes = tuple(computed_role_map.get(item_id_hash, ()))
    if item_role_hashes != expected_role_spec_hashes:
        raise PermissionError(
            "compiled skill portable capability item roles mismatch"
        )

    receipt_rows: list[tuple[str, str]] = []
    for row in rows:
        if row["item_id_hash"] != item_id_hash:
            continue
        relative_path = Path(row["metadata_path"])
        current = root
        for component in relative_path.parts:
            current = current / component
            if current.is_symlink():
                raise PermissionError(
                    "compiled skill portable capability path contains a link"
                )
        if not current.is_file():
            raise PermissionError(
                "compiled skill portable capability metadata is missing"
            )
        try:
            raw_content = current.read_bytes()
            text = raw_content.decode("utf-8")
            payload = json.loads(text)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PermissionError(
                "compiled skill portable capability metadata is unreadable"
            ) from exc
        metadata = validate_compiled_portable_task_capability(payload)
        canonical_text = json.dumps(
            metadata.safe_payload(),
            indent=2,
            sort_keys=True,
        ) + "\n"
        if text != canonical_text:
            raise PermissionError(
                "compiled skill portable capability encoding drifted"
            )
        if (
            metadata.item_id_hash != row["item_id_hash"]
            or metadata.program_id_hash != row["program_id_hash"]
            or metadata.typed_binding_hash != row["typed_binding_hash"]
            or metadata.bound_recipe_hash != row["bound_recipe_hash"]
            or metadata.role_spec.role_spec_hash != row["role_spec_hash"]
            or metadata.metadata_hash != row["metadata_hash"]
            or stable_hash({"content": text})
            != row["metadata_content_hash"]
        ):
            raise PermissionError(
                "compiled skill portable capability metadata drifted"
            )
        receipt_rows.append(
            (
                relative_path.as_posix(),
                hashlib.sha256(raw_content).hexdigest(),
            )
        )
    return item_role_hashes, tuple(sorted(receipt_rows))


class SkillLearnProgramCompiler:
    """Compile promoted programs into SkillLearnBench-compatible SKILL.md files."""

    def __init__(
        self,
        *,
        event_sink: EventSink | None = None,
        typed_program_registry: TypedProgramBindingRegistry | None = None,
        require_typed_bindings: bool = False,
        portable_capability_compiler_mode: str | None = None,
        typed_execution_contract_registry: (
            TypedExecutionContractRegistry | None
        ) = None,
    ) -> None:
        self.event_sink = event_sink or NullEventSink()
        self.typed_program_registry = typed_program_registry
        self.require_typed_bindings = require_typed_bindings
        self.portable_capability_compiler_mode = (
            str(portable_capability_compiler_mode or "")
        )
        self.typed_execution_contract_registry = (
            typed_execution_contract_registry
        )
        if require_typed_bindings and typed_program_registry is None:
            raise ValueError(
                "typed compiler mode requires a binding registry"
            )
        if self.portable_capability_compiler_mode and (
            self.portable_capability_compiler_mode
            != PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
        ):
            raise ValueError("portable capability compiler mode is unsupported")
        if self.portable_capability_compiler_mode and (
            not require_typed_bindings or typed_program_registry is None
        ):
            raise ValueError(
                "portable capability compiler mode requires typed bindings"
            )
        if typed_execution_contract_registry is not None and not (
            self.portable_capability_compiler_mode
            and require_typed_bindings
            and typed_program_registry is not None
        ):
            raise ValueError(
                "execution contracts require portable typed compiler mode"
            )

    def require_program_bindings(
        self,
        programs: Sequence[HypothesisProgram],
    ) -> tuple[str, ...]:
        if not self.require_typed_bindings:
            if any(
                is_typed_recipe_materialization(program)
                for program in programs
            ):
                raise PermissionError(
                    "typed recipe materialization requires the receipt-bound "
                    "typed compiler"
                )
            return ()
        assert self.typed_program_registry is not None
        return tuple(
            self.typed_program_registry.require(program).binding_hash
            for program in programs
        )

    def compile(
        self,
        *,
        programs: Sequence[HypothesisProgram],
        items: Sequence[BenchmarkItem],
        split_manifest: SplitManifest,
        output_root: str | Path,
        method_name: str = "assumption-agent-v2",
        allowed_statuses: set[HypothesisStatus] | None = None,
        target_item_ids: Sequence[str] | None = None,
        target_split: str = "train",
        trace_id: str = "skill_compile",
    ) -> SkillCompileResult:
        self.require_program_bindings(programs)
        allowed = allowed_statuses or {HypothesisStatus.PROMOTED}
        destination = Path(output_root) / method_name
        target_ids = set(target_item_ids or split_manifest.train_ids)
        known_manifest_ids = {
            *split_manifest.train_ids,
            *split_manifest.validation_ids,
            *split_manifest.test_ids,
        }
        if not target_ids <= known_manifest_ids:
            raise ValueError("compiler target IDs are outside the split manifest")
        split_ids = {
            "train": set(split_manifest.train_ids),
            "validation": set(split_manifest.validation_ids),
            "test": set(split_manifest.test_ids),
        }
        if target_split not in split_ids:
            raise ValueError("compiler target split must be train, validation, or test")
        if not target_ids <= split_ids[target_split]:
            raise PermissionError("compiler target IDs do not belong to the declared target split")
        target_items = [item for item in items if item.id in target_ids]
        if len(target_items) != len(target_ids):
            raise ValueError("split manifest references missing SkillLearnBench target items")
        program_rows: list[
            tuple[
                HypothesisProgram,
                str,
                tuple[LoweredSkillAction, ...],
                str,
                str,
            ]
        ] = []
        typed_bindings: dict[str, Any] = {}
        bound_typed_recipes: dict[str, BoundTypedRecipe] = {}
        portable_role_specs: dict[str, Any] = {}
        typed_execution_contracts: dict[
            str, TypedExecutionContract
        ] = {}
        seen_program_ids: set[str] = set()
        for program in sorted(programs, key=lambda row: row.id):
            if program.status not in allowed:
                continue
            if self.require_typed_bindings:
                assert self.typed_program_registry is not None
                if self.portable_capability_compiler_mode:
                    bound_recipe = (
                        self.typed_program_registry.require_bound_recipe(
                            program
                        )
                    )
                    typed_bindings[program.id] = bound_recipe.binding
                    bound_typed_recipes[program.id] = bound_recipe
                    portable_role_specs[program.id] = (
                        portable_role_spec_for_bound_recipe(bound_recipe)
                    )
                    if self.typed_execution_contract_registry is not None:
                        contract_registry = (
                            self.typed_execution_contract_registry
                        )
                        typed_execution_contracts[program.id] = (
                            contract_registry.require_for_bound_recipe(
                                bound_recipe
                            )
                        )
                else:
                    typed_bindings[program.id] = (
                        self.typed_program_registry.require(program)
                    )
            validation_issues = program.validate()
            if validation_issues:
                raise ValueError(
                    "SkillLearn compiler received an invalid program: "
                    f"{program.id}: {validation_issues}"
                )
            if program.id in seen_program_ids:
                raise ValueError("SkillLearn compiler program IDs must be unique")
            seen_program_ids.add(program.id)
            skill_name = _slug(program.id)
            if self.portable_capability_compiler_mode:
                role_spec = portable_role_specs[program.id]
                output_locator = (
                    deterministic_portable_capability_output_locator(
                        role_spec_hash=role_spec.role_spec_hash,
                        typed_binding_hash=(
                            typed_bindings[program.id].binding_hash
                        ),
                    )
                )
                lowered_actions = _lower_portable_capability_skill(
                    role_spec_hash=role_spec.role_spec_hash,
                    output_container_locator=output_locator,
                    role=role_spec.role,
                    artifact_format=role_spec.artifact_format.value,
                    capability=role_spec.capability,
                    workflow=(
                        bound_typed_recipes[program.id].recipe.workflow.value
                    ),
                    operator_kinds=tuple(
                        node.kind.value
                        for node in bound_typed_recipes[
                            program.id
                        ].recipe.nodes
                    ),
                    execution_contract=typed_execution_contracts.get(
                        program.id
                    ),
                )
                skill_text = _render_portable_capability_skill(
                    program,
                    skill_name,
                    lowered_actions,
                    role_spec_hash=role_spec.role_spec_hash,
                    output_container_locator=output_locator,
                    role=role_spec.role,
                    artifact_format=role_spec.artifact_format.value,
                    capability=role_spec.capability,
                    workflow=(
                        bound_typed_recipes[program.id].recipe.workflow.value
                    ),
                    execution_contract=typed_execution_contracts.get(
                        program.id
                    ),
                )
            else:
                lowered_actions = _lower_skilllearn_program(program)
                skill_text = _render_skill(
                    program,
                    skill_name,
                    lowered_actions,
                )
            treatment_hash = skilllearn_program_treatment_hash(
                program,
                lowered_actions=lowered_actions,
                rendered_skill=skill_text,
                portable_capability_role_spec_hash=(
                    portable_role_specs[program.id].role_spec_hash
                    if self.portable_capability_compiler_mode
                    else ""
                ),
            )
            program_rows.append(
                (
                    program,
                    skill_name,
                    lowered_actions,
                    skill_text,
                    treatment_hash,
                )
            )

        typed_binding_rows = [
            {
                **binding.safe_payload(),
                "program_id": program_id,
            }
            for program_id, binding in sorted(typed_bindings.items())
        ]
        typed_binding_hashes = tuple(
            row["binding_hash"] for row in typed_binding_rows
        )
        typed_snapshot_hashes = tuple(
            sorted({row["snapshot_hash"] for row in typed_binding_rows})
        )
        typed_snapshot_ledger_hashes = {
            row["snapshot_ledger_hash"] for row in typed_binding_rows
        }
        if len(typed_snapshot_ledger_hashes) > 1:
            raise PermissionError(
                "typed compiler programs crossed snapshot ledgers"
            )
        typed_snapshot_ledger_hash = (
            next(iter(typed_snapshot_ledger_hashes))
            if typed_snapshot_ledger_hashes
            else ""
        )
        typed_binding_set_hash = (
            stable_hash({"bindings": typed_binding_rows})
            if typed_binding_rows
            else ""
        )

        program_set_payload = {
            "routing_version": SKILL_ROUTING_VERSION,
            "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
            "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
            "program_treatment_hashes": sorted(
                row[4] for row in program_rows
            ),
        }
        if self.portable_capability_compiler_mode:
            program_set_payload["portable_capability_compiler_mode"] = (
                self.portable_capability_compiler_mode
            )
            program_set_payload["portable_capability_role_spec_hashes"] = (
                sorted(
                    row.role_spec_hash
                    for row in portable_role_specs.values()
                )
            )
            if self.typed_execution_contract_registry is not None:
                program_set_payload["typed_execution_contract_hashes"] = sorted(
                    row.contract_hash
                    for row in typed_execution_contracts.values()
                )
        program_set_hash = stable_hash(program_set_payload)
        rendered_skills: dict[str, tuple[HypothesisProgram, str, str]] = {}
        used_hypotheses: set[str] = set()
        families: set[str] = set()
        routed_item_hashes: set[str] = set()
        action_lowering_hashes: dict[str, str] = {
            program.id: stable_hash(
                [
                    {
                        "semantics": row.semantics,
                        "instruction": row.instruction,
                    }
                    for row in lowered_actions
                ]
            )
            for program, _, lowered_actions, _, _ in program_rows
        }
        program_treatment_hashes = {
            program.id: treatment_hash
            for program, _, _, _, treatment_hash in program_rows
        }
        item_skill_content_hashes: dict[str, list[str]] = {
            item.id_hash: [] for item in target_items
        }
        portable_metadata_sources: dict[
            str, tuple[CompiledPortableTaskCapability, str, str]
        ] = {}
        portable_role_spec_rows: list[dict[str, str]] = []
        item_portable_role_spec_hashes: dict[str, list[str]] = {
            item.id_hash: [] for item in target_items
        }
        item_portable_metadata_content_hashes: dict[str, list[str]] = {
            item.id_hash: [] for item in target_items
        }
        for program, skill_name, _, skill_text, _ in program_rows:
            matched_items = sorted(
                (
                    item
                    for item in target_items
                    if program.matches({**dict(item.features), "family": item.family})
                ),
                key=lambda item: item.id_hash,
            )
            for item in matched_items:
                item_hash = item.id_hash
                relative_path = str(
                    Path("items") / item_hash / skill_name / "SKILL.md"
                )
                if relative_path in rendered_skills:
                    raise ValueError(
                        "SkillLearn compiler produced colliding skill paths: "
                        f"{relative_path}"
                    )
                content_hash = stable_hash({"content": skill_text})
                rendered_skills[relative_path] = (
                    program,
                    skill_text,
                    content_hash,
                )
                item_skill_content_hashes[item_hash].append(content_hash)
                if self.portable_capability_compiler_mode:
                    bound_recipe = bound_typed_recipes[program.id]
                    role_spec = portable_role_specs[program.id]
                    metadata = build_compiled_portable_task_capability(
                        role_spec,
                        item_id=item.id,
                        program_id=program.id,
                        typed_binding_hash=(
                            bound_recipe.binding.binding_hash
                        ),
                        bound_recipe_hash=bound_recipe.bound_recipe_hash,
                        execution_contract=typed_execution_contracts.get(
                            program.id
                        ),
                    )
                    metadata_path = str(
                        Path("task_capabilities")
                        / item_hash
                        / f"{metadata.program_id_hash}.json"
                    )
                    if metadata_path in portable_metadata_sources:
                        raise ValueError(
                            "portable capability metadata path collision"
                        )
                    metadata_text = json.dumps(
                        metadata.safe_payload(),
                        indent=2,
                        sort_keys=True,
                    ) + "\n"
                    metadata_content_hash = stable_hash(
                        {"content": metadata_text}
                    )
                    portable_metadata_sources[metadata_path] = (
                        metadata,
                        metadata_text,
                        metadata_content_hash,
                    )
                    portable_role_spec_rows.append(
                        {
                            "item_id_hash": item_hash,
                            "program_id_hash": metadata.program_id_hash,
                            "typed_binding_hash": (
                                metadata.typed_binding_hash
                            ),
                            "bound_recipe_hash": metadata.bound_recipe_hash,
                            "role_spec_hash": role_spec.role_spec_hash,
                            "metadata_hash": metadata.metadata_hash,
                            "metadata_path": metadata_path,
                            "metadata_content_hash": metadata_content_hash,
                        }
                    )
                    item_portable_role_spec_hashes[item_hash].append(
                        role_spec.role_spec_hash
                    )
                    item_portable_metadata_content_hashes[item_hash].append(
                        metadata_content_hash
                    )
                routed_item_hashes.add(item_hash)
                used_hypotheses.add(program.id)
                families.add(item.family)

        portable_role_spec_rows.sort(
            key=lambda row: (
                row["item_id_hash"],
                row["program_id_hash"],
                row["metadata_path"],
            )
        )
        item_treatment_hashes: dict[str, str] = {}
        for item_hash, content_hashes in sorted(
            item_skill_content_hashes.items()
        ):
            if not content_hashes:
                item_treatment_hashes[item_hash] = NO_SKILL_TREATMENT_HASH
                continue
            item_treatment_payload = {
                "routing_version": SKILL_ROUTING_VERSION,
                "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
                "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
                "skill_content_hashes": sorted(content_hashes),
            }
            if self.portable_capability_compiler_mode:
                item_treatment_payload.update(
                    {
                        "portable_capability_compiler_mode": (
                            self.portable_capability_compiler_mode
                        ),
                        "portable_capability_role_spec_hashes": sorted(
                            item_portable_role_spec_hashes[item_hash]
                        ),
                        "portable_capability_metadata_content_hashes": sorted(
                            item_portable_metadata_content_hashes[item_hash]
                        ),
                    }
                )
            item_treatment_hashes[item_hash] = stable_hash(
                item_treatment_payload
            )
        treatment_hash = stable_hash(
            {
                "program_set_hash": program_set_hash,
                "item_treatment_hashes": item_treatment_hashes,
            }
        )
        skill_content_hashes = {
            relative_path: row[2]
            for relative_path, row in sorted(rendered_skills.items())
        }
        compile_manifest = {
            "method_name": method_name,
            "routing_version": SKILL_ROUTING_VERSION,
            "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
            "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
            "external_verifier_exposed_to_agent": False,
            "action_lowering_hashes": action_lowering_hashes,
            "program_treatment_hashes": program_treatment_hashes,
            "program_set_hash": program_set_hash,
            "skill_content_hashes": skill_content_hashes,
            "item_treatment_hashes": item_treatment_hashes,
            "treatment_hash": treatment_hash,
            "typed_binding_required": self.require_typed_bindings,
            "typed_binding_rows": typed_binding_rows,
            "typed_binding_set_hash": typed_binding_set_hash,
            "typed_snapshot_hashes": list(typed_snapshot_hashes),
            "typed_snapshot_ledger_hash": typed_snapshot_ledger_hash,
            "skill_paths": sorted(rendered_skills),
            "item_routes": {
                item.id_hash: (
                    str(Path("items") / item.id_hash)
                    if item.id_hash in routed_item_hashes
                    else None
                )
                for item in sorted(target_items, key=lambda row: row.id_hash)
            },
            "family_count": len(families),
            "hypothesis_ids": sorted(used_hypotheses),
            "split_manifest_hash": split_manifest.manifest_hash,
            "source_split": "train",
            "target_split": target_split,
            "target_item_set_hash": stable_hash({"item_ids": sorted(target_ids)}),
            "test_content_accessed": False,
            "raw_content_persisted": False,
        }
        portable_capability_role_spec_set_hash = ""
        item_portable_capability_metadata_paths: dict[
            str, tuple[Path, ...]
        ] = {}
        if self.portable_capability_compiler_mode:
            portable_capability_role_spec_set_hash = stable_hash(
                {"rows": portable_role_spec_rows}
            )
            role_hash_map = {
                item_hash: sorted(role_hashes)
                for item_hash, role_hashes in sorted(
                    item_portable_role_spec_hashes.items()
                )
            }
            metadata_path_map = {
                item_hash: sorted(
                    row["metadata_path"]
                    for row in portable_role_spec_rows
                    if row["item_id_hash"] == item_hash
                )
                for item_hash in sorted(item_portable_role_spec_hashes)
            }
            compile_manifest.update(
                {
                    "portable_capability_compiler_mode": (
                        self.portable_capability_compiler_mode
                    ),
                    "portable_capability_role_spec_rows": (
                        portable_role_spec_rows
                    ),
                    "portable_capability_role_spec_set_hash": (
                        portable_capability_role_spec_set_hash
                    ),
                    "item_portable_capability_role_spec_hashes": (
                        role_hash_map
                    ),
                    "item_portable_capability_metadata_paths": (
                        metadata_path_map
                    ),
                    "source_artifact_locators_persisted": False,
                }
            )
            if self.typed_execution_contract_registry is not None:
                compile_manifest.update(
                    {
                        "typed_execution_contract_hashes": sorted(
                            row.contract_hash
                            for row in typed_execution_contracts.values()
                        ),
                        "typed_execution_contract_runtime_enforcement_claimed": (
                            False
                        ),
                    }
                )
            item_portable_capability_metadata_paths = {
                item_hash: tuple(
                    destination / relative_path
                    for relative_path in relative_paths
                )
                for item_hash, relative_paths in metadata_path_map.items()
            }
            _require_portable_compiler_outputs_locator_free(
                train_locators=tuple(
                    artifact.locator
                    for bound_recipe in bound_typed_recipes.values()
                    for artifact in bound_recipe.snapshot.graph.artifacts
                ),
                skill_texts=tuple(
                    row[1] for row in rendered_skills.values()
                ),
                metadata_texts=tuple(
                    row[1] for row in portable_metadata_sources.values()
                ),
                compile_manifest=compile_manifest,
            )
        compile_manifest_hash = stable_hash(compile_manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-",
                dir=destination.parent,
            )
        )
        try:
            for relative_path, (_, skill_text, _) in sorted(
                rendered_skills.items()
            ):
                skill_path = staging / relative_path
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                skill_path.write_text(skill_text, encoding="utf-8")
            for relative_path, (_, metadata_text, _) in sorted(
                portable_metadata_sources.items()
            ):
                metadata_path = staging / relative_path
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(metadata_text, encoding="utf-8")
            (staging / "compile_manifest.json").write_text(
                json.dumps(compile_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _atomic_replace_directory(staging, destination)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

        skill_paths = tuple(
            destination / relative_path for relative_path in sorted(rendered_skills)
        )
        item_sources = {
            item_hash: destination / "items" / item_hash
            for item_hash in sorted(routed_item_hashes)
        }
        items_by_hash = {item.id_hash: item for item in target_items}
        for relative_path, (program, _, content_hash) in sorted(
            rendered_skills.items()
        ):
            item_hash = Path(relative_path).parts[1]
            item = items_by_hash[item_hash]
            skill_path = destination / relative_path
            portable_event_payload: dict[str, Any] = {}
            if self.portable_capability_compiler_mode:
                program_id_hash = stable_hash({"program_id": program.id})
                matching_rows = [
                    row
                    for row in portable_role_spec_rows
                    if row["item_id_hash"] == item_hash
                    and row["program_id_hash"] == program_id_hash
                ]
                if len(matching_rows) != 1:
                    raise PermissionError(
                        "portable capability event binding is not unique"
                    )
                portable_row = matching_rows[0]
                portable_event_payload = {
                    "portable_capability_compiler_mode": (
                        self.portable_capability_compiler_mode
                    ),
                    "portable_capability_role_spec_hash": (
                        portable_row["role_spec_hash"]
                    ),
                    "portable_capability_metadata_hash": (
                        portable_row["metadata_hash"]
                    ),
                    "portable_capability_metadata_content_hash": (
                        portable_row["metadata_content_hash"]
                    ),
                    "portable_capability_metadata_path_hash": stable_hash(
                        {"path": portable_row["metadata_path"]}
                    ),
                    "portable_capability_bound_recipe_hash": (
                        portable_row["bound_recipe_hash"]
                    ),
                    "portable_capability_executes_before_agent_start": True,
                    "source_artifact_locator_disclosed": False,
                }
            self.event_sink.emit(
                Event(
                    event="skilllearn_skill_compiled",
                    stage="benchmark.skilllearn.compile",
                    trace_id=trace_id,
                    payload={
                        "hypothesis_id": program.id,
                        "hypothesis_hash": program.payload_hash,
                        "program_treatment_hash": (
                            program_treatment_hashes[program.id]
                        ),
                        "program_set_hash": program_set_hash,
                        "item_id_hash": item_hash,
                        "item_treatment_hash": item_treatment_hashes[item_hash],
                        "family_hash": stable_hash({"family": item.family}),
                        "skill_path_hash": stable_hash({"path": str(skill_path)}),
                        "skill_content_hash": content_hash,
                        "compile_manifest_hash": compile_manifest_hash,
                        "typed_binding_hash": (
                            typed_bindings[program.id].binding_hash
                            if program.id in typed_bindings
                            else ""
                        ),
                        "typed_binding_set_hash": typed_binding_set_hash,
                        "typed_binding_hashes": list(typed_binding_hashes),
                        "typed_snapshot_hash": (
                            typed_bindings[program.id].snapshot_hash
                            if program.id in typed_bindings
                            else ""
                        ),
                        "typed_recipe_id": (
                            typed_bindings[program.id].recipe_id
                            if program.id in typed_bindings
                            else ""
                        ),
                        "typed_snapshot_ledger_hash": (
                            typed_snapshot_ledger_hash
                        ),
                        "split_manifest_hash": split_manifest.manifest_hash,
                        "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
                        "action_lowering_hash": action_lowering_hashes[program.id],
                        "external_verifier_exposed_to_agent": False,
                        "source_split": "train",
                        "target_split": target_split,
                        **portable_event_payload,
                    },
                )
            )
        return SkillCompileResult(
            output_root=destination,
            skill_paths=skill_paths,
            family_count=len(families),
            hypothesis_ids=tuple(sorted(used_hypotheses)),
            manifest_hash=compile_manifest_hash,
            item_sources=dict(item_sources),
            program_set_hash=program_set_hash,
            treatment_hash=treatment_hash,
            item_treatment_hashes=item_treatment_hashes,
            typed_binding_hashes=typed_binding_hashes,
            typed_binding_set_hash=typed_binding_set_hash,
            typed_snapshot_hashes=typed_snapshot_hashes,
            typed_snapshot_ledger_hash=typed_snapshot_ledger_hash,
            portable_capability_compiler_mode=(
                self.portable_capability_compiler_mode
            ),
            portable_capability_role_spec_set_hash=(
                portable_capability_role_spec_set_hash
            ),
            item_portable_capability_role_spec_hashes={
                item_hash: tuple(sorted(role_hashes))
                for item_hash, role_hashes in sorted(
                    item_portable_role_spec_hashes.items()
                )
            }
            if self.portable_capability_compiler_mode
            else {},
            item_portable_capability_metadata_paths=(
                item_portable_capability_metadata_paths
            ),
            typed_execution_contract_hashes=tuple(
                sorted(
                    row.contract_hash
                    for row in typed_execution_contracts.values()
                )
            ),
        )


_EXECUTION_INVARIANT_INSTRUCTIONS: Mapping[InvariantKind, str] = {
    InvariantKind.PRIMARY_ARTIFACT_READ_BEFORE_MUTATION: (
        "Open and inspect the current bound primary artifact before any mutation; "
        "derive changes only from that observed current state."
    ),
    InvariantKind.TASK_DELTA_ONLY: (
        "Apply only the concrete delta required by the current public task and "
        "preserve content outside that delta."
    ),
    InvariantKind.PRESERVE_UNTARGETED_CONTENT: (
        "Preserve every field, record, and artifact not targeted by the current "
        "public task."
    ),
    InvariantKind.EACH_SOURCE_ITEM_ASSIGNED_EXACTLY_ONCE: (
        "Before moving files, construct a complete one-to-one assignment in which "
        "each current source item appears exactly once."
    ),
    InvariantKind.SOURCE_COLLECTION_EMPTY_AFTER_SUCCESS: (
        "Treat a nonempty current source collection after organization as an "
        "incomplete result."
    ),
    InvariantKind.INPUT_DERIVATION_PRESERVED: (
        "Derive the rendered result from the current bound input data and retain "
        "that data-to-output relationship."
    ),
    InvariantKind.OBSERVABLE_INTERACTION_POSTCONDITION: (
        "Define and replay the task-required interaction; success requires an "
        "observable visible state change, not only event-handler source code."
    ),
    InvariantKind.FINITE_SEARCH_SPACE_DECLARED: (
        "Evaluate only the receipt-bound finite candidate set and record the exact "
        "number of evaluated candidates."
    ),
    InvariantKind.FINAL_METRICS_FROM_FINAL_OUTPUT: (
        "Recompute every reported metric from the final materialized output using "
        "one canonical computation; do not copy an intermediate-run metric."
    ),
    InvariantKind.FINAL_OUTPUT_REOPENED: (
        "After materializing the result, reopen it from task-local storage and "
        "check the reopened state before finishing."
    ),
}


def _lower_portable_capability_skill(
    *,
    role_spec_hash: str,
    output_container_locator: str,
    role: str,
    artifact_format: str,
    capability: str,
    workflow: str,
    operator_kinds: Sequence[str],
    execution_contract: TypedExecutionContract | None = None,
) -> tuple[LoweredSkillAction, ...]:
    """Fixed compiler-owned instructions for the supported capability mode."""

    base_actions = (
        LoweredSkillAction(
            action_id="portable-current-item-role",
            semantics="prompt_directive",
            instruction=(
                f"Use the harness-resolved `{role}` role for the current "
                f"`{artifact_format}` task artifact only; do not reuse a "
                "path from another item."
            ),
        ),
        LoweredSkillAction(
            action_id="portable-read-harness-profile",
            semantics="harness_prepared_evidence",
            instruction=(
                "Use the harness-created, receipt-verified task-artifact "
                f"profile at `{output_container_locator}` as supplemental "
                f"task-local evidence via the fixed `{capability}` capability "
                f"(role receipt `{role_spec_hash}`)."
            ),
        ),
        LoweredSkillAction(
            action_id="portable-selected-typed-workflow",
            semantics="harness_selected_typed_workflow",
            instruction=(
                "The pre-agent capability supplies read-only artifact "
                "evidence only; it does not execute task writes, rendering, "
                "or transforms. Follow the harness-selected "
                f"`{workflow}` workflow as the fixed agent plan `"
                + " -> ".join(operator_kinds)
                + "`, using only current-task tools and arguments grounded "
                "in the public instruction and current environment."
            ),
        ),
        LoweredSkillAction(
            action_id="portable-complete-current-task",
            semantics="agent_local_self_check",
            instruction=(
                "Complete the request in the current public task instruction "
                "and check the resulting task-local state before finishing."
            ),
        ),
    )
    if execution_contract is None:
        return base_actions
    issues = execution_contract.validate_closed()
    if issues:
        raise PermissionError(
            f"portable execution contract is invalid: {list(issues)}"
        )
    invariant_actions = tuple(
        LoweredSkillAction(
            action_id=f"execution-invariant-{row.kind.value}",
            semantics="harness_selected_execution_contract",
            instruction=_EXECUTION_INVARIANT_INSTRUCTIONS[row.kind],
        )
        for row in execution_contract.invariants
    )
    resource_action = LoweredSkillAction(
        action_id="execution-contract-resource-receipt",
        semantics="harness_selected_execution_contract",
        instruction=(
            "Keep this task within the declared contract limits: at most "
            f"{execution_contract.resources.max_action_starts} action starts, "
            f"{execution_contract.resources.max_mutations} mutations, "
            f"{execution_contract.resources.max_repair_attempts} repair attempts, "
            f"{execution_contract.resources.max_completion_checks} completion "
            "checks, and "
            f"{execution_contract.resources.max_search_evaluations} finite-search "
            "evaluations; record the observed counts in the final effect receipt."
        ),
    )
    completion_action = LoweredSkillAction(
        action_id="execution-contract-completion-loop",
        semantics="agent_local_self_check",
        instruction=(
            "Use the fixed completion loop: apply the registered mutation, reopen "
            "the materialized output, check all closed invariants, perform only "
            "bounded repairs, reopen and recheck after a repair, then finalize one "
            "effect receipt. Recompute self-evaluation only from the final reopened "
            "output."
        ),
    )
    return (
        *base_actions[:-1],
        *invariant_actions,
        resource_action,
        completion_action,
        base_actions[-1],
    )


def _render_portable_capability_skill(
    program: HypothesisProgram,
    skill_name: str,
    lowered_actions: Sequence[LoweredSkillAction],
    *,
    role_spec_hash: str,
    output_container_locator: str,
    role: str,
    artifact_format: str,
    capability: str,
    workflow: str,
    execution_contract: TypedExecutionContract | None = None,
) -> str:
    description = (
        "Use a harness-prepared profile and a closed typed workflow for the "
        "current task artifact."
    )
    lines = [
        "---",
        f"name: {skill_name}",
        f"description: {json.dumps(description, ensure_ascii=True)}",
        "---",
        "# Portable current-item typed workflow",
        "",
        "## Activation",
        "",
    ]
    lines.extend(_render_trigger(program))
    receipt_lines = [
        "",
        "## Harness receipt",
        "",
        f"- Input role receipt: `{role_spec_hash}`",
        f"- Input role: `{role}` (`{artifact_format}`)",
        f"- Fixed capability: `{capability}`",
        f"- Selected workflow: `{workflow}`",
    ]
    if execution_contract is not None:
        issues = execution_contract.validate_closed()
        if issues:
            raise PermissionError(
                f"portable execution contract is invalid: {list(issues)}"
            )
        receipt_lines.extend(
            [
                f"- Execution contract receipt: `{execution_contract.contract_hash}`",
                "- Closed invariants: "
                + ", ".join(
                    f"`{row.kind.value}`"
                    for row in execution_contract.invariants
                ),
                "- Completion phases: "
                + " -> ".join(
                    f"`{row.value}`"
                    for row in execution_contract.completion.phase_order
                ),
                "- Declared resource limits: "
                f"actions={execution_contract.resources.max_action_starts}, "
                f"mutations={execution_contract.resources.max_mutations}, "
                f"repairs={execution_contract.resources.max_repair_attempts}, "
                f"checks={execution_contract.resources.max_completion_checks}, "
                "search_evaluations="
                f"{execution_contract.resources.max_search_evaluations}.",
                "- Contract delivery is receipt-bound; runtime enforcement and "
                "semantic compliance are not claimed by the compiler.",
            ]
        )
    receipt_lines.extend(
        [
            "- Capability scope: read-only pre-agent artifact evidence; remaining workflow operators are agent-executed.",
            f"- Derived profile: `{output_container_locator}`",
            "- The profile must exist and have a verified effect receipt before the agent starts.",
            "- No source input path is supplied by this skill.",
            "",
            "## Procedure",
            "",
        ]
    )
    lines.extend(receipt_lines)
    for index, action in enumerate(lowered_actions, start=1):
        if action.semantics == "harness_prepared_evidence":
            label = "Harness-prepared evidence"
        elif action.semantics == "harness_selected_typed_workflow":
            label = "Harness-selected typed workflow"
        elif action.semantics == "harness_selected_execution_contract":
            label = "Closed execution contract"
        elif action.semantics == "agent_local_self_check":
            label = "Agent-local self-check"
        else:
            label = "Agent instruction"
        lines.append(f"{index}. **{label}:** {action.instruction}")
    lines.extend(
        [
            "",
            "## Evaluation boundary",
            "",
            "- Use only current-task files, the public task instruction, and the verified derived profile.",
            "- The hidden benchmark verifier is unavailable until after the agent exits.",
            "- Runtime package installation and network access are not part of this capability.",
            "",
        ]
    )
    return "\n".join(lines)


def _require_portable_compiler_outputs_locator_free(
    *,
    train_locators: Sequence[str],
    skill_texts: Sequence[str],
    metadata_texts: Sequence[str],
    compile_manifest: Mapping[str, Any],
) -> None:
    corpus = "\n".join(
        (
            *skill_texts,
            *metadata_texts,
            json.dumps(compile_manifest, ensure_ascii=True, sort_keys=True),
        )
    )
    leaked = sorted(
        {
            locator
            for locator in train_locators
            if isinstance(locator, str) and locator and locator in corpus
        }
    )
    if leaked:
        raise PermissionError(
            "portable capability compiler attempted to persist a TRAIN locator"
        )


def _render_skill(
    program: HypothesisProgram,
    skill_name: str,
    lowered_actions: Sequence[LoweredSkillAction],
) -> str:
    description = program.statement.replace("\n", " ").strip()
    lines = [
        "---",
        f"name: {skill_name}",
        f"description: {json.dumps(description, ensure_ascii=True)}",
        "---",
        f"# {description}",
        "",
        "## Activation",
        "",
    ]
    lines.extend(_render_trigger(program))
    lines.extend(["", "## Procedure", ""])
    for index, action in enumerate(lowered_actions, start=1):
        label = (
            "Agent-local self-check"
            if action.semantics == "agent_local_self_check"
            else "Agent instruction"
        )
        lines.append(f"{index}. **{label}:** {action.instruction}")
    lines.extend(
        [
            "",
            "## Evaluation boundary",
            "",
            "- Only the task-local instructions and evidence available during this run may be used.",
            "- The benchmark verifier runs after the agent exits and is not callable from this skill.",
            "",
            "## Fallback",
            "",
            "The frozen router omits this skill when activation does not match. Once this skill is injected, the benchmark does not replace the result with a post-hoc baseline output.",
            "",
        ]
    )
    return "\n".join(lines)


def _lower_skilllearn_program(
    program: HypothesisProgram,
) -> tuple[LoweredSkillAction, ...]:
    contract_issues = backend_action_contract_issues(
        program,
        allowed_operations=SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
        external_evidence_is_hidden=True,
    )
    if contract_issues:
        summary = (
            "contains operations without a backend lowering"
            if any(
                issue.startswith("unsupported_action_operation:")
                for issue in contract_issues
            )
            else "references hidden external evaluation evidence"
        )
        raise ValueError(
            f"SkillLearn action graph {summary}: "
            f"{list(contract_issues)}"
        )
    lowered: list[LoweredSkillAction] = []
    for action in _ordered_actions(program.action_graph):
        value = _display_action_value(action.value).strip()
        target = action.target.strip()
        if action.operation == "execute_step":
            instruction = (
                f"Execute the task step `{target}`: {value}"
                if value
                else f"Execute the task step `{target}`."
            )
            semantics = "prompt_directive"
        elif action.operation == "produce_artifact":
            detail = f": {value}" if value else "."
            instruction = f"Produce the requested artifact `{target}`{detail}"
            semantics = "prompt_directive"
        elif action.operation == "request_evidence":
            detail = f": {value}" if value else "."
            instruction = (
                f"Gather task-local evidence `{target}`{detail} Do not request "
                "policy-off/on results or the hidden benchmark verifier."
            )
            semantics = "prompt_directive"
        else:
            detail = f"`{target}`: {value}" if value else target
            instruction = f"Before completion, check locally that {detail}"
            if not instruction.endswith((".", "!", "?")):
                instruction += "."
            semantics = "agent_local_self_check"
        lowered.append(
            LoweredSkillAction(
                action_id=action.id,
                semantics=semantics,
                instruction=instruction,
            )
        )
    return tuple(lowered)


def skilllearn_program_treatment_hash(
    program: HypothesisProgram,
    *,
    lowered_actions: Sequence[LoweredSkillAction] | None = None,
    rendered_skill: str | None = None,
    portable_capability_role_spec_hash: str = "",
) -> str:
    """Hash only the external treatment that can reach the SkillLearn agent."""

    lowered = tuple(lowered_actions or _lower_skilllearn_program(program))
    skill_text = rendered_skill or _render_skill(
        program,
        _slug(program.id),
        lowered,
    )
    payload = {
        "routing_version": SKILL_ROUTING_VERSION,
        "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
        "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
        "external_verifier_exposed_to_agent": False,
        "rendered_skill_hash": stable_hash({"content": skill_text}),
    }
    if portable_capability_role_spec_hash:
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            portable_capability_role_spec_hash,
        ):
            raise PermissionError(
                "portable capability role spec hash is malformed"
            )
        payload.update(
            {
                "portable_capability_compiler_mode": (
                    PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
                ),
                "portable_capability_role_spec_hash": (
                    portable_capability_role_spec_hash
                ),
            }
        )
    return stable_hash(payload)


def skilllearn_program_set_treatment_hash(
    programs: Sequence[HypothesisProgram],
) -> str:
    return stable_hash(
        {
            "routing_version": SKILL_ROUTING_VERSION,
            "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
            "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
            "program_treatment_hashes": sorted(
                skilllearn_program_treatment_hash(program)
                for program in programs
            ),
        }
    )


def _atomic_replace_directory(staging: Path, destination: Path) -> None:
    """Publish a complete compiler tree without merging stale files into it."""

    backup: Path | None = None
    if destination.exists():
        backup = destination.with_name(
            f".{destination.name}.backup-{uuid.uuid4().hex}"
        )
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None and backup.exists():
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def _render_trigger(program: HypothesisProgram) -> list[str]:
    rows: list[str] = []
    for group_name, predicates in (
        ("Require all", program.trigger.all_of),
        ("Require any", program.trigger.any_of),
        ("Exclude", (*program.trigger.none_of, *program.anti_trigger.all_of, *program.anti_trigger.any_of)),
    ):
        for predicate in predicates:
            rows.append(f"- {group_name}: `{predicate.key}` `{predicate.op}` `{_display_value(predicate.value)}`")
    return rows or ["- Apply only when the structured task router selects this program."]


def _ordered_actions(actions: tuple[ActionNode, ...]) -> tuple[ActionNode, ...]:
    by_id = {action.id: action for action in actions}
    pending = {action.id: set(action.depends_on) for action in actions}
    ordered: list[ActionNode] = []
    while pending:
        ready = sorted(action_id for action_id, dependencies in pending.items() if not dependencies)
        if not ready:
            raise ValueError("cannot compile a cyclic action graph")
        for action_id in ready:
            ordered.append(by_id[action_id])
            pending.pop(action_id)
            for dependencies in pending.values():
                dependencies.discard(action_id)
    return tuple(ordered)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "hypothesis-program"


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _display_action_value(value: Any, *, humanize_identifiers: bool = False) -> str:
    """Render structured action values as deterministic, agent-readable prose."""

    if value is None:
        return ""
    if isinstance(value, str):
        if humanize_identifiers and re.fullmatch(r"[A-Za-z0-9_]+", value):
            return value.replace("_", " ")
        return value
    if isinstance(value, Mapping):
        return "; ".join(
            f"{_humanize_action_identifier(str(key))}: "
            f"{_display_action_value(value[key], humanize_identifiers=True)}"
            for key in sorted(value, key=lambda row: str(row))
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return ", ".join(
            _display_action_value(row, humanize_identifiers=True)
            for row in value
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _humanize_action_identifier(value: str) -> str:
    return value.replace("_", " ")
