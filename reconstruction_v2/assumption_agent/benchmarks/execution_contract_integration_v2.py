from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import shutil
import tempfile
import threading
import uuid
from types import ModuleType
from typing import Any, Mapping, Sequence

from ..events import Event
from ..models import HypothesisProgram, stable_hash
from ..splits import BenchmarkItem
from ..typed_execution_contract import (
    TypedExecutionContract,
    TypedExecutionContractRegistry,
    load_typed_execution_contract,
)
from ..typed_operator_grammar import TypedProgramBindingRegistry
from .execution_contract_prompt_v2 import (
    EXECUTION_CONTRACT_PROMPT_CONTAINER_PATH,
    EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION,
    ExecutionContractPromptInjectionReceiptV2,
    bind_execution_contract_prompt_v2,
    build_execution_contract_prompt_capsule_v2,
)
from .runtime_profile_injection import VerifiedRuntimeProfile
from .skilllearn_compiler import SkillCompileResult, SkillSourceReceipt
from .skilllearn_lifecycle import (
    PortableTaskCapabilityRuntimeContext,
    SkillLearnAgentTerminalError,
    SkillLearnSubprocessBackend,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from .typed_task_capability import CompiledPortableTaskCapability


EXECUTION_CONTRACT_BUNDLE_VERSION = (
    "immutable_companion_to_frozen_portable_compile_v2"
)
EXECUTION_CONTRACT_RUNTIME_VERSION = (
    "companion_bundle_verified_combined_prompt_runtime_v2"
)
EXECUTION_CONTRACT_BUNDLE_FILENAME = "execution_contract_bundle.v2.json"


def _canonical_manifest_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"


def _immutable_container(*_args: Any, **_kwargs: Any) -> None:
    raise TypeError("execution-contract manifest is immutable")


class _FrozenJsonDict(dict[str, Any]):
    __setitem__ = _immutable_container
    __delitem__ = _immutable_container
    clear = _immutable_container
    pop = _immutable_container
    popitem = _immutable_container
    setdefault = _immutable_container
    update = _immutable_container
    __ior__ = _immutable_container


class _FrozenJsonList(list[Any]):
    __setitem__ = _immutable_container
    __delitem__ = _immutable_container
    append = _immutable_container
    clear = _immutable_container
    extend = _immutable_container
    insert = _immutable_container
    pop = _immutable_container
    remove = _immutable_container
    reverse = _immutable_container
    sort = _immutable_container
    __iadd__ = _immutable_container
    __imul__ = _immutable_container


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenJsonDict(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return _FrozenJsonList(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class ExecutionContractRuntimeContextV2:
    request_hash: str
    base_runtime_context_hash: str
    source_receipt_hash: str
    typed_binding_set_hash: str
    public_instruction_hash: str
    bundle_manifest_hash: str
    item_route_hash: str
    base_metadata: tuple[CompiledPortableTaskCapability, ...]
    contracts: tuple[TypedExecutionContract, ...]
    public_instruction: str = field(compare=False, repr=False)

    @property
    def context_hash(self) -> str:
        return stable_hash(self.safe_payload())

    @property
    def profile_contract_bindings(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "metadata_hash": metadata.metadata_hash,
                "execution_contract_hash": contract.contract_hash,
                "binding_hash": stable_hash(
                    {
                        "metadata_hash": metadata.metadata_hash,
                        "execution_contract_hash": contract.contract_hash,
                    }
                ),
            }
            for metadata, contract in sorted(
                zip(
                    self.base_metadata,
                    self.contracts,
                    strict=True,
                ),
                key=lambda row: row[0].metadata_hash,
            )
        )

    @property
    def profile_contract_binding_set_hash(self) -> str:
        return stable_hash(
            {
                "profile_contract_bindings": list(
                    self.profile_contract_bindings
                )
            }
        )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "runtime_policy": EXECUTION_CONTRACT_RUNTIME_VERSION,
            "request_hash": self.request_hash,
            "base_runtime_context_hash": self.base_runtime_context_hash,
            "source_receipt_hash": self.source_receipt_hash,
            "typed_binding_set_hash": self.typed_binding_set_hash,
            "public_instruction_hash": self.public_instruction_hash,
            "bundle_manifest_hash": self.bundle_manifest_hash,
            "item_route_hash": self.item_route_hash,
            "execution_contract_hashes": [
                row.contract_hash for row in self.contracts
            ],
            "execution_contract_set_hash": stable_hash(
                {
                    "execution_contract_hashes": [
                        value
                        for value in sorted(
                            {row.contract_hash for row in self.contracts}
                        )
                    ]
                }
            ),
            "profile_contract_binding_set_hash": (
                self.profile_contract_binding_set_hash
            ),
            "profile_contract_binding_hashes": [
                row["binding_hash"]
                for row in self.profile_contract_bindings
            ],
            "public_instruction_persisted": False,
            "source_artifact_locator_persisted": False,
            "runtime_enforcement_claimed": False,
        }


@dataclass(frozen=True)
class ExecutionContractTrialEvidenceV2:
    """Post-run evidence returned without exposing mutable runner state."""

    observation: SkillLearnTrialObservation = field(
        compare=False,
        repr=False,
    )
    prompt_receipt: ExecutionContractPromptInjectionReceiptV2 | None = field(
        compare=False,
        repr=False,
    )
    execution_backend_instance_hash: str
    contract_route_expected: bool
    prompt_receipt_valid: bool

    @property
    def evidence_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "runtime_policy": EXECUTION_CONTRACT_RUNTIME_VERSION,
            "request_hash": self.observation.request.request_hash,
            "observation_hash": self.observation.observation_hash,
            "execution_backend_instance_hash": (
                self.execution_backend_instance_hash
            ),
            "contract_route_expected": self.contract_route_expected,
            "prompt_receipt_valid": self.prompt_receipt_valid,
            "prompt_receipt_hash": (
                self.prompt_receipt.receipt_hash
                if self.prompt_receipt is not None
                else None
            ),
            "effective_prompt_sha256": (
                self.prompt_receipt.effective_prompt_sha256
                if self.prompt_receipt is not None
                else None
            ),
            "raw_observation_or_prompt_content_persisted": False,
        }

    def verify(self) -> None:
        receipt = self.prompt_receipt
        observation = self.observation
        # The backend hash is opaque to this value object.  Requiring a
        # canonical digest still rejects empty or malformed evidence.
        if (
            not isinstance(self.execution_backend_instance_hash, str)
            or len(self.execution_backend_instance_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.execution_backend_instance_hash
            )
        ):
            raise PermissionError(
                "execution-contract backend instance hash is invalid"
            )
        if self.prompt_receipt_valid != (receipt is not None):
            raise PermissionError(
                "execution-contract evidenced receipt validity drifted"
            )
        if receipt is not None and not self.contract_route_expected:
            raise PermissionError(
                "execution-contract receipt appeared outside its route"
            )
        if (
            self.contract_route_expected
            and observation.request.variant is not TrialVariant.POLICY_ON
        ):
            raise PermissionError(
                "execution-contract route is not a policy-on request"
            )
        if (
            self.contract_route_expected
            and observation.valid
            and receipt is None
        ):
            raise PermissionError(
                "valid execution-contract route lacks an evidenced receipt"
            )
        if receipt is not None and (
            receipt.request_hash != observation.request.request_hash
            or observation.runtime_profile_prompt_delivery_policy
            != EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION
            or observation.runtime_profile_prompt_injection_receipt_hash
            != receipt.receipt_hash
            or observation.runtime_profile_effective_prompt_sha256
            != receipt.effective_prompt_sha256
        ):
            raise PermissionError(
                "execution-contract evidenced prompt receipt drifted"
            )


@dataclass(frozen=True)
class ExecutionContractCompileBundleV2:
    root: Path = field(compare=False)
    manifest_hash: str
    manifest: Mapping[str, Any]
    base_compile_result: SkillCompileResult = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", _freeze_json(self.manifest))

    @property
    def manifest_path(self) -> Path:
        return self.root / EXECUTION_CONTRACT_BUNDLE_FILENAME

    def verify(self) -> None:
        path = self.manifest_path
        if path.is_symlink() or not path.is_file():
            raise PermissionError("execution-contract bundle manifest is missing")
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
            payload = json.loads(text)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PermissionError(
                "execution-contract bundle manifest is unreadable"
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or text != _canonical_manifest_text(payload)
            or dict(payload) != dict(self.manifest)
            or stable_hash(payload) != self.manifest_hash
        ):
            raise PermissionError(
                "execution-contract bundle manifest or encoding drifted"
            )
        _validate_bundle_manifest(payload, self.base_compile_result)

    def has_item(self, item_id: str) -> bool:
        self.verify()
        item_hash = stable_hash({"item_id": item_id})
        return item_hash in dict(self.manifest["item_routes"])

    def runtime_context_for(
        self,
        *,
        request: SkillLearnTrialRequest,
        source_receipt: SkillSourceReceipt,
        base_context: PortableTaskCapabilityRuntimeContext,
    ) -> ExecutionContractRuntimeContextV2:
        self.verify()
        if request.variant is not TrialVariant.POLICY_ON:
            raise PermissionError(
                "execution-contract bundle is policy-on only"
            )
        if request.portable_capability_delivery_mode:
            raise PermissionError(
                "v2 companion runtime cannot share the frozen v1 delivery mode"
            )
        item_hash = stable_hash({"item_id": request.item_id})
        raw_route = dict(self.manifest["item_routes"]).get(item_hash)
        if not isinstance(raw_route, Mapping):
            raise PermissionError(
                "execution-contract bundle has no current-item route"
            )
        request_family_hash = stable_hash({"family": request.family})
        if (
            raw_route.get("family_hash") != request_family_hash
            or base_context.family != request.family
        ):
            raise PermissionError(
                "execution-contract bundle family route drifted"
            )
        if request.program_id == "":
            raise PermissionError(
                "execution-contract request program identity is empty"
            )
        if request.program_id is not None:
            program_id_hash = stable_hash(
                {"program_id": request.program_id}
            )
            if program_id_hash not in tuple(raw_route["program_id_hashes"]):
                raise PermissionError(
                    "execution-contract request program is outside the item route"
                )
        if (
            request.compile_manifest_hash
            != self.manifest["base_compile_manifest_hash"]
            or request.program_set_hash
            != self.manifest["base_program_set_hash"]
            or request.typed_binding_set_hash
            != self.manifest["base_typed_binding_set_hash"]
            or request.treatment_hash != raw_route["base_treatment_hash"]
            or source_receipt.receipt_hash
            != raw_route["base_source_receipt_hash"]
            or base_context.request_hash != request.request_hash
            or base_context.source_receipt_hash != source_receipt.receipt_hash
            or base_context.typed_binding_set_hash
            != request.typed_binding_set_hash
        ):
            raise PermissionError(
                "execution-contract bundle does not match base treatment receipts"
            )
        rows_by_program_hash = {
            row["program_id_hash"]: row
            for row in self.manifest["contract_rows"]
        }
        contracts_by_program_hash = {
            program_hash: load_typed_execution_contract(
                row["execution_contract"]
            )
            for program_hash, row in rows_by_program_hash.items()
        }
        route_program_hashes = tuple(raw_route["program_id_hashes"])
        metadata_program_hashes = tuple(
            row.program_id_hash for row in base_context.metadata
        )
        if (
            not metadata_program_hashes
            or metadata_program_hashes
            != tuple(sorted(metadata_program_hashes))
            or set(metadata_program_hashes) != set(route_program_hashes)
        ):
            raise PermissionError(
                "execution-contract and portable metadata routes differ"
            )
        try:
            contracts = tuple(
                contracts_by_program_hash[value]
                for value in metadata_program_hashes
            )
        except KeyError as exc:
            raise PermissionError(
                "execution-contract item route references a missing contract"
            ) from exc
        if len(contracts) != len(base_context.metadata) or any(
            contract.graph_hash != metadata.role_spec.source_graph_hash
            or contract.recipe_id != metadata.role_spec.source_recipe_id
            or contract.target_family_hash != request_family_hash
            or rows_by_program_hash[metadata.program_id_hash][
                "typed_binding_hash"
            ]
            != metadata.typed_binding_hash
            or rows_by_program_hash[metadata.program_id_hash][
                "bound_recipe_hash"
            ]
            != metadata.bound_recipe_hash
            for metadata, contract in zip(
                base_context.metadata,
                contracts,
                strict=True,
            )
        ):
            raise PermissionError(
                "execution-contract and portable metadata coverage differ"
            )
        context = ExecutionContractRuntimeContextV2(
            request_hash=request.request_hash,
            base_runtime_context_hash=base_context.context_hash,
            source_receipt_hash=source_receipt.receipt_hash,
            typed_binding_set_hash=request.typed_binding_set_hash,
            public_instruction_hash=base_context.public_instruction_hash,
            bundle_manifest_hash=self.manifest_hash,
            item_route_hash=str(raw_route["item_route_hash"]),
            base_metadata=base_context.metadata,
            contracts=contracts,
            public_instruction=base_context.public_instruction,
        )
        if stable_hash(
            {"public_instruction": context.public_instruction}
        ) != context.public_instruction_hash:
            raise PermissionError(
                "execution-contract public instruction receipt drifted"
            )
        return context


def build_execution_contract_compile_bundle_v2(
    *,
    base_compile_result: SkillCompileResult,
    programs: Sequence[HypothesisProgram],
    items: Sequence[BenchmarkItem],
    typed_program_registry: TypedProgramBindingRegistry,
    execution_contract_registry: TypedExecutionContractRegistry,
    output_root: str | Path,
) -> ExecutionContractCompileBundleV2:
    program_by_id = {row.id: row for row in programs}
    if set(base_compile_result.hypothesis_ids) - set(program_by_id):
        raise PermissionError(
            "execution-contract bundle is missing a compiled program"
        )
    contract_rows: list[dict[str, Any]] = []
    contract_by_program_hash: dict[str, TypedExecutionContract] = {}
    for program_id in sorted(base_compile_result.hypothesis_ids):
        program = program_by_id[program_id]
        bound = typed_program_registry.require_bound_recipe(program)
        contract = execution_contract_registry.require_for_bound_recipe(bound)
        program_id_hash = stable_hash({"program_id": program.id})
        contract_by_program_hash[program_id_hash] = contract
        contract_rows.append(
            {
                "program_id_hash": program_id_hash,
                "typed_binding_hash": bound.binding.binding_hash,
                "bound_recipe_hash": bound.bound_recipe_hash,
                "source_graph_hash": bound.snapshot.graph.graph_hash,
                "source_recipe_id": bound.recipe.recipe_id,
                "source_recipe_payload_hash": stable_hash(
                    bound.recipe.payload()
                ),
                "execution_contract_hash": contract.contract_hash,
                "execution_contract": contract.safe_payload(),
                "raw_program_or_task_content_persisted": False,
            }
        )
    contract_rows.sort(key=lambda row: row["program_id_hash"])
    item_routes: dict[str, dict[str, Any]] = {}
    for item in sorted(items, key=lambda row: row.id_hash):
        if base_compile_result.source_for(item.id) is None:
            continue
        matched_program_hashes = tuple(
            sorted(
                stable_hash({"program_id": program.id})
                for program in programs
                if program.id in base_compile_result.hypothesis_ids
                and program.matches(
                    {**dict(item.features), "family": item.family}
                )
            )
        )
        if not matched_program_hashes:
            raise PermissionError(
                "execution-contract routed item has no matched program"
            )
        try:
            contract_hashes = tuple(
                contract_by_program_hash[value].contract_hash
                for value in matched_program_hashes
            )
        except KeyError as exc:
            raise PermissionError(
                "execution-contract matched program has no contract"
            ) from exc
        family_hash = stable_hash({"family": item.family})
        if any(
            contract_by_program_hash[value].target_family_hash != family_hash
            for value in matched_program_hashes
        ):
            raise PermissionError(
                "execution-contract item route crosses family identity"
            )
        source_receipt = base_compile_result.source_receipt_for(item.id)
        route_payload = {
            "item_id_hash": item.id_hash,
            "family_hash": family_hash,
            "program_id_hashes": list(matched_program_hashes),
            "execution_contract_hashes": list(contract_hashes),
            "base_treatment_hash": base_compile_result.treatment_hash_for(
                item.id
            ),
            "base_source_receipt_hash": source_receipt.receipt_hash,
            "raw_item_or_family_content_persisted": False,
        }
        route_payload["item_route_hash"] = stable_hash(route_payload)
        item_routes[item.id_hash] = route_payload
    if not item_routes:
        raise PermissionError("execution-contract bundle has no routed items")
    manifest: dict[str, Any] = {
        "bundle_policy": EXECUTION_CONTRACT_BUNDLE_VERSION,
        "base_compile_manifest_hash": base_compile_result.manifest_hash,
        "base_program_set_hash": base_compile_result.program_set_hash,
        "base_typed_binding_set_hash": (
            base_compile_result.typed_binding_set_hash
        ),
        "base_typed_snapshot_hashes": list(
            base_compile_result.typed_snapshot_hashes
        ),
        "base_typed_snapshot_ledger_hash": (
            base_compile_result.typed_snapshot_ledger_hash
        ),
        "contract_rows": contract_rows,
        "execution_contract_set_hash": stable_hash(
            {
                "execution_contract_hashes": sorted(
                    {
                        row["execution_contract_hash"]
                        for row in contract_rows
                    }
                )
            }
        ),
        "item_routes": item_routes,
        "item_route_set_hash": stable_hash(
            {"item_routes": item_routes}
        ),
        "frozen_v1_files_modified": False,
        "runtime_enforcement_claimed": False,
        "validation_or_test_content_accessed": False,
        "raw_program_task_or_evaluator_content_persisted": False,
    }
    _validate_bundle_manifest(manifest, base_compile_result)
    manifest_hash = stable_hash(manifest)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / EXECUTION_CONTRACT_BUNDLE_FILENAME
    if path.exists() or path.is_symlink():
        raise FileExistsError("execution-contract bundle output already exists")
    path.write_text(_canonical_manifest_text(manifest), encoding="utf-8")
    bundle = ExecutionContractCompileBundleV2(
        root=root,
        manifest_hash=manifest_hash,
        manifest=manifest,
        base_compile_result=base_compile_result,
    )
    bundle.verify()
    return bundle


def _validate_bundle_manifest(
    manifest: Mapping[str, Any],
    base_compile_result: SkillCompileResult,
) -> None:
    if manifest.get("bundle_policy") != EXECUTION_CONTRACT_BUNDLE_VERSION:
        raise PermissionError("execution-contract bundle policy drifted")
    if (
        manifest.get("base_compile_manifest_hash")
        != base_compile_result.manifest_hash
        or manifest.get("base_program_set_hash")
        != base_compile_result.program_set_hash
        or manifest.get("base_typed_binding_set_hash")
        != base_compile_result.typed_binding_set_hash
        or tuple(manifest.get("base_typed_snapshot_hashes") or ())
        != base_compile_result.typed_snapshot_hashes
        or manifest.get("base_typed_snapshot_ledger_hash")
        != base_compile_result.typed_snapshot_ledger_hash
    ):
        raise PermissionError(
            "execution-contract bundle base compile receipt drifted"
        )
    raw_rows = manifest.get("contract_rows")
    raw_routes = manifest.get("item_routes")
    if not isinstance(raw_rows, list) or not isinstance(raw_routes, Mapping):
        raise PermissionError("execution-contract bundle shape is malformed")
    program_hashes: list[str] = []
    contract_hashes: list[str] = []
    contract_hash_by_program_hash: dict[str, str] = {}
    family_hash_by_contract_hash: dict[str, str] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise PermissionError(
                "execution-contract bundle row is malformed"
            )
        contract = load_typed_execution_contract(row["execution_contract"])
        if (
            row.get("execution_contract_hash") != contract.contract_hash
            or row.get("source_graph_hash") != contract.graph_hash
            or row.get("source_recipe_id") != contract.recipe_id
            or row.get("raw_program_or_task_content_persisted") is not False
        ):
            raise PermissionError(
                "execution-contract bundle row binding drifted"
            )
        for key in (
            "program_id_hash",
            "typed_binding_hash",
            "bound_recipe_hash",
            "source_graph_hash",
            "source_recipe_payload_hash",
            "execution_contract_hash",
        ):
            value = row.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise PermissionError(
                    "execution-contract bundle hash is malformed"
                )
        program_hashes.append(str(row["program_id_hash"]))
        contract_hashes.append(contract.contract_hash)
        contract_hash_by_program_hash[str(row["program_id_hash"])] = (
            contract.contract_hash
        )
        family_hash_by_contract_hash[contract.contract_hash] = (
            contract.target_family_hash
        )
    if (
        not raw_rows
        or program_hashes != sorted(set(program_hashes))
        or manifest.get("execution_contract_set_hash")
        != stable_hash(
            {
                "execution_contract_hashes": sorted(
                    set(contract_hashes)
                )
            }
        )
    ):
        raise PermissionError(
            "execution-contract bundle contract set is not canonical"
        )
    for item_hash, route in raw_routes.items():
        if not isinstance(item_hash, str) or not isinstance(route, Mapping):
            raise PermissionError(
                "execution-contract item route is malformed"
            )
        route_program_hashes = tuple(route.get("program_id_hashes") or ())
        route_contract_hashes = tuple(
            route.get("execution_contract_hashes") or ()
        )
        route_family_hash = route.get("family_hash")
        try:
            expected_route_contract_hashes = tuple(
                contract_hash_by_program_hash[value]
                for value in route_program_hashes
            )
        except (KeyError, TypeError) as exc:
            raise PermissionError(
                "execution-contract item route references an unknown program"
            ) from exc
        route_without_hash = {
            key: value for key, value in route.items() if key != "item_route_hash"
        }
        if (
            len(item_hash) != 64
            or any(character not in "0123456789abcdef" for character in item_hash)
            or route.get("item_id_hash") != item_hash
            or not isinstance(route_family_hash, str)
            or len(route_family_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in route_family_hash
            )
            or route.get("item_route_hash") != stable_hash(route_without_hash)
            or not route_program_hashes
            or route_program_hashes != tuple(sorted(set(route_program_hashes)))
            or route_contract_hashes != expected_route_contract_hashes
            or any(
                family_hash_by_contract_hash[value] != route_family_hash
                for value in route_contract_hashes
            )
            or any(
                not isinstance(route.get(key), str)
                or len(str(route.get(key))) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(route.get(key))
                )
                for key in (
                    "base_treatment_hash",
                    "base_source_receipt_hash",
                    "item_route_hash",
                )
            )
            or route.get("raw_item_or_family_content_persisted") is not False
        ):
            raise PermissionError(
                "execution-contract item route binding drifted"
            )
    if manifest.get("item_route_set_hash") != stable_hash(
        {"item_routes": dict(raw_routes)}
    ):
        raise PermissionError(
            "execution-contract item route set drifted"
        )
    for flag in (
        "frozen_v1_files_modified",
        "runtime_enforcement_claimed",
        "validation_or_test_content_accessed",
        "raw_program_task_or_evaluator_content_persisted",
    ):
        if manifest.get(flag) is not False:
            raise PermissionError(
                "execution-contract bundle boundary flag drifted"
            )


class ExecutionContractSubprocessBackendV2(SkillLearnSubprocessBackend):
    """Opt-in runtime wrapper that leaves every frozen v1 module untouched."""

    def __init__(
        self,
        *args: Any,
        execution_contract_bundle: ExecutionContractCompileBundleV2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        execution_contract_bundle.verify()
        self.execution_contract_bundle = execution_contract_bundle
        self._execution_contract_local = threading.local()
        self._execution_contract_instance_nonce = uuid.uuid4().hex
        # The frozen v1 backend caches one mutable runner module.  A single
        # wrapper instance is therefore serialized; flat experiment
        # parallelism uses one backend instance per worker.
        self._execution_contract_run_lock = threading.Lock()

    @property
    def execution_backend_instance_hash(self) -> str:
        return stable_hash(
            {
                "runtime_policy": EXECUTION_CONTRACT_RUNTIME_VERSION,
                "runner_instance_token": self._runner_instance_token,
                "execution_contract_instance_nonce": (
                    self._execution_contract_instance_nonce
                ),
                "bundle_manifest_hash": (
                    self.execution_contract_bundle.manifest_hash
                ),
            }
        )

    def _load_portable_task_capability_context(
        self,
        *,
        request: SkillLearnTrialRequest,
        source_receipt: SkillSourceReceipt,
        compile_root: Path,
    ) -> PortableTaskCapabilityRuntimeContext | None:
        base_context = super()._load_portable_task_capability_context(
            request=request,
            source_receipt=source_receipt,
            compile_root=compile_root,
        )
        self._execution_contract_local.context = None
        self._execution_contract_local.receipt = None
        if request.variant is TrialVariant.POLICY_ON and self.execution_contract_bundle.has_item(
            request.item_id
        ):
            if base_context is None:
                raise PermissionError(
                    "execution-contract route has no portable runtime context"
                )
            self._execution_contract_local.context = (
                self.execution_contract_bundle.runtime_context_for(
                    request=request,
                    source_receipt=source_receipt,
                    base_context=base_context,
                )
            )
        return base_context

    def _install_treatment_receipt_adapter(self, runner: ModuleType) -> None:
        super()._install_treatment_receipt_adapter(runner)
        base_inject = runner._inject_skills_runtime

        def inject_with_execution_contract(
            container_name: str,
            skill_source_dir: Path,
            copies: list[tuple[str, str]],
        ) -> None:
            runner._assumption_v2_execution_contract_prompt_receipt = None
            base_inject(container_name, skill_source_dir, copies)
            context = getattr(
                self._execution_contract_local,
                "context",
                None,
            )
            if context is None:
                return
            try:
                self._inject_execution_contract_prompt_v2(
                    runner=runner,
                    container_name=container_name,
                    context=context,
                )
            except Exception as exc:
                runner._assumption_v2_execution_contract_prompt_receipt = None
                self.event_sink.emit(
                    Event(
                        event=(
                            "skilllearn_trial_blocked_invalid_"
                            "execution_contract_prompt_v2"
                        ),
                        stage=(
                            "benchmark.skilllearn.execution_contract_prompt_v2"
                        ),
                        trace_id=context.request_hash[:20],
                        payload={
                            "delivery_policy": (
                                EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION
                            ),
                            "request_hash": context.request_hash,
                            "bundle_manifest_hash": (
                                context.bundle_manifest_hash
                            ),
                            "error_type": type(exc).__name__,
                            "agent_started": False,
                            "model_invoked": False,
                            "raw_content_persisted": False,
                        },
                    )
                )
                if isinstance(exc, SkillLearnAgentTerminalError):
                    raise
                raise SkillLearnAgentTerminalError(
                    "execution_contract_prompt_delivery_invalid"
                ) from exc

        runner._inject_skills_runtime = inject_with_execution_contract
        runner._assumption_v2_execution_contract_prompt_receipt = None

    def _inject_execution_contract_prompt_v2(
        self,
        *,
        runner: ModuleType,
        container_name: str,
        context: ExecutionContractRuntimeContextV2,
    ) -> None:
        effects = getattr(
            runner,
            "_assumption_v2_task_capability_effects",
            None,
        )
        if (
            not isinstance(effects, tuple)
            or len(effects) != len(context.contracts)
            or len(effects) != len(context.base_metadata)
            or any(
                effect.metadata_hash != metadata.metadata_hash
                or effect.item_id_hash != metadata.item_id_hash
                or effect.role_spec_hash
                != metadata.role_spec.role_spec_hash
                for effect, metadata in zip(
                    effects,
                    context.base_metadata,
                    strict=True,
                )
            )
        ):
            raise PermissionError(
                "execution-contract profile effect coverage is not exact"
            )
        profiles = tuple(
            VerifiedRuntimeProfile(
                metadata_hash=row.metadata_hash,
                item_id_hash=row.item_id_hash,
                role_spec_hash=row.role_spec_hash,
                effect_receipt_hash=row.effect_receipt_hash,
                output_sha256=row.output_sha256,
                profile_bytes=row.profile_bytes,
            )
            for row in effects
        )
        capsule = build_execution_contract_prompt_capsule_v2(
            request_hash=context.request_hash,
            base_runtime_context_hash=context.base_runtime_context_hash,
            source_receipt_hash=context.source_receipt_hash,
            typed_binding_set_hash=context.typed_binding_set_hash,
            public_instruction_hash=context.public_instruction_hash,
            bundle_manifest_hash=context.bundle_manifest_hash,
            profiles=profiles,
            contracts=context.contracts,
        )
        agent = runner.get_agent(self.agent_id)
        if not isinstance(agent, dict):
            raise PermissionError("execution-contract agent definition is missing")
        target = EXECUTION_CONTRACT_PROMPT_CONTAINER_PATH
        self._require_portable_container_path_without_links(
            delegate=runner.subprocess,
            container_name=container_name,
            locator=target,
        )
        if not self._portable_capability_docker_condition(
            runner.subprocess,
            [
                "docker",
                "exec",
                container_name,
                "test",
                "!",
                "-e",
                target,
            ],
        ):
            raise PermissionError(
                "execution-contract prompt target already exists"
            )
        receipt_root = Path(
            tempfile.mkdtemp(prefix="skilllearn_execution_contract_prompt_v2-")
        )
        try:
            source = receipt_root / "fragment.txt"
            source.write_bytes(capsule.fragment_bytes)
            self._portable_capability_docker_run(
                runner.subprocess,
                ["docker", "cp", str(source), f"{container_name}:{target}"],
            )
            self._portable_capability_docker_run(
                runner.subprocess,
                ["docker", "exec", container_name, "chmod", "0444", target],
            )
            self._require_portable_container_path_without_links(
                delegate=runner.subprocess,
                container_name=container_name,
                locator=target,
            )
            readback = receipt_root / "container-readback.txt"
            self._portable_capability_docker_run(
                runner.subprocess,
                ["docker", "cp", f"{container_name}:{target}", str(readback)],
            )
            bound = bind_execution_contract_prompt_v2(
                capsule,
                container_readback=readback.read_bytes(),
                run_template=str(agent.get("run") or ""),
                public_instruction=context.public_instruction,
            )
            agent["run"] = bound.run_template
            runner._assumption_v2_execution_contract_prompt_receipt = (
                bound.receipt
            )
            self._execution_contract_local.receipt = bound.receipt
            self.event_sink.emit(
                Event(
                    event=(
                        "skilllearn_pre_agent_execution_contract_prompt_v2_"
                        "injected"
                    ),
                    stage=(
                        "benchmark.skilllearn.execution_contract_prompt_v2"
                    ),
                    trace_id=context.request_hash[:20],
                    payload={
                        **bound.receipt.safe_payload(),
                        "receipt_hash": bound.receipt.receipt_hash,
                    },
                )
            )
        finally:
            shutil.rmtree(receipt_root, ignore_errors=True)

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        with self._execution_contract_run_lock:
            return self._run_serialized_evidence(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            ).observation

    def run_with_evidence(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> ExecutionContractTrialEvidenceV2:
        with self._execution_contract_run_lock:
            return self._run_serialized_evidence(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )

    def _run_serialized_evidence(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> ExecutionContractTrialEvidenceV2:
        self._execution_contract_local.context = None
        self._execution_contract_local.receipt = None
        expected_route = (
            request.variant is TrialVariant.POLICY_ON
            and self.execution_contract_bundle.has_item(request.item_id)
        )
        try:
            observation = super().run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
            context = getattr(
                self._execution_contract_local,
                "context",
                None,
            )
            receipt = getattr(
                self._execution_contract_local,
                "receipt",
                None,
            )
            receipt_valid = False
            if (
                expected_route
                and isinstance(context, ExecutionContractRuntimeContextV2)
                and isinstance(
                    receipt,
                    ExecutionContractPromptInjectionReceiptV2,
                )
            ):
                receipt_valid = (
                    receipt.request_hash == request.request_hash
                    and receipt.base_runtime_context_hash
                    == context.base_runtime_context_hash
                    and receipt.source_receipt_hash
                    == context.source_receipt_hash
                    and receipt.typed_binding_set_hash
                    == context.typed_binding_set_hash
                    and receipt.public_instruction_hash
                    == context.public_instruction_hash
                    and receipt.bundle_manifest_hash
                    == self.execution_contract_bundle.manifest_hash
                    and receipt.profile_count == len(context.base_metadata)
                    and receipt.contract_hashes
                    == tuple(
                        sorted(
                            {
                                row.contract_hash
                                for row in context.contracts
                            }
                        )
                    )
                    and receipt.profile_contract_binding_set_hash
                    == context.profile_contract_binding_set_hash
                    and receipt.profile_contract_binding_hashes
                    == tuple(
                        row["binding_hash"]
                        for row in context.profile_contract_bindings
                    )
                )
            if observation.valid and expected_route and not receipt_valid:
                error_type = (
                    "execution_contract_prompt_delivery_missing"
                    if receipt is None
                    else "execution_contract_prompt_delivery_invalid"
                )
                observation = replace(
                    observation,
                    success=False,
                    score=0.0,
                    metrics={"evaluation_valid": 0.0},
                    error_type=error_type,
                    runtime_profile_prompt_delivery_policy="",
                    runtime_profile_prompt_injection_receipt_hash="",
                    runtime_profile_effective_prompt_sha256="",
                )
                self.event_sink.emit(
                    Event(
                        event=(
                            "skilllearn_execution_contract_trial_"
                            "blocked_missing_receipt_v2"
                        ),
                        stage=(
                            "benchmark.skilllearn."
                            "execution_contract_prompt_v2"
                        ),
                        trace_id=trace_id,
                        payload={
                            "request_hash": request.request_hash,
                            "bundle_manifest_hash": (
                                self.execution_contract_bundle.manifest_hash
                            ),
                            "error_type": error_type,
                            "task_effect_attributed": False,
                            "runtime_enforcement_claimed": False,
                        },
                    )
                )
            elif observation.valid and receipt is not None and not receipt_valid:
                observation = replace(
                    observation,
                    success=False,
                    score=0.0,
                    metrics={"evaluation_valid": 0.0},
                    error_type="execution_contract_prompt_delivery_unexpected",
                    runtime_profile_prompt_delivery_policy="",
                    runtime_profile_prompt_injection_receipt_hash="",
                    runtime_profile_effective_prompt_sha256="",
                )
            elif receipt_valid:
                assert isinstance(
                    receipt,
                    ExecutionContractPromptInjectionReceiptV2,
                )
                observation = replace(
                    observation,
                    runtime_profile_prompt_delivery_policy=(
                        EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION
                    ),
                    runtime_profile_prompt_injection_receipt_hash=(
                        receipt.receipt_hash
                    ),
                    runtime_profile_effective_prompt_sha256=(
                        receipt.effective_prompt_sha256
                    ),
                )
                self.event_sink.emit(
                    Event(
                        event=(
                            "skilllearn_execution_contract_trial_"
                            "receipted_v2"
                        ),
                        stage=(
                            "benchmark.skilllearn."
                            "execution_contract_prompt_v2"
                        ),
                        trace_id=trace_id,
                        payload={
                            "request_hash": request.request_hash,
                            "observation_hash": observation.observation_hash,
                            "prompt_receipt_hash": receipt.receipt_hash,
                            "bundle_manifest_hash": (
                                receipt.bundle_manifest_hash
                            ),
                            "semantic_consumption_claimed": False,
                            "runtime_enforcement_claimed": False,
                        },
                    )
                )
            if expected_route or receipt is not None:
                self.event_sink.emit(
                    Event(
                        event=(
                            "skilllearn_execution_contract_trial_completed_v2"
                        ),
                        stage=(
                            "benchmark.skilllearn."
                            "execution_contract_prompt_v2"
                        ),
                        trace_id=trace_id,
                        payload={
                            "request_hash": request.request_hash,
                            "observation_hash": observation.observation_hash,
                            "success": observation.success,
                            "valid": observation.valid,
                            "score": observation.score,
                            "metrics": dict(observation.metrics),
                            "error_type": observation.error_type,
                            "contract_route_expected": expected_route,
                            "prompt_receipt_valid": receipt_valid,
                            "supersedes_frozen_v1_trial_completed": True,
                            "semantic_consumption_claimed": False,
                            "runtime_enforcement_claimed": False,
                        },
                    )
                )
            evidence = ExecutionContractTrialEvidenceV2(
                observation=observation,
                prompt_receipt=(receipt if receipt_valid else None),
                execution_backend_instance_hash=(
                    self.execution_backend_instance_hash
                ),
                contract_route_expected=expected_route,
                prompt_receipt_valid=receipt_valid,
            )
            evidence.verify()
            return evidence
        finally:
            self._execution_contract_local.context = None
            self._execution_contract_local.receipt = None
