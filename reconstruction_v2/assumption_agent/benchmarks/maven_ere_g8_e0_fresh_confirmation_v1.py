"""Fresh document-disjoint confirmation of the fixed MAVEN-ERE G8/E0 policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import maven_ere_g8_e1_acquisition_v1 as base_acquisition
from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core
from assumption_agent.benchmarks import maven_ere_g8_e1_formal_controller_v1 as v1
from assumption_agent.benchmarks import maven_ere_g8_e1_result_blind_recovery_controller_v2 as recovery
from assumption_agent.benchmarks import maven_ere_global_family_e2_train_crossfit_v1 as e2
from assumption_agent.benchmarks import maven_ere_local_runtime_v1 as local_runtime
from assumption_agent.benchmarks.maven_ere_nli_runtime_v1 import verify_maven_design
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)
from replication_runtime.qasc_nli_v1 import binding as nli_binding
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding


VERSION = "v1"
SCHEMA = "maven_ere_g8_e0_fresh_confirmation_v1"
DESIGN_RELATIVE = Path("manifests/maven_ere_g8_e0_fresh_confirmation_design_v1.json")
DESIGN_FILE_SHA256 = "109e3fca6e9e407942cc2477a740eaf67b496679f8921a1c151e3ce8496c0bac"
DESIGN_SELF_SHA256 = "b9e996f99175770faf969291165daf6fd9163dc974b4f2a3f0ad94b5c38bf107"
E2_DISPOSITION_RELATIVE = Path(
    "manifests/maven_ere_global_family_e2_train_crossfit_result_disposition_v1.json"
)
E2_DISPOSITION_FILE_SHA256 = (
    "4eb5df267b547092b155d7cee8bbf791e8b0d88a1283bc9b8aeca5ad4918e5d4"
)
E2_DISPOSITION_SELF_SHA256 = (
    "043e4c8c4ac371d99958cad3321cdead404ce9acdfb7829ba2c786ada86f3c2e"
)
FREEZE_RELATIVE = Path(
    "manifests/maven_ere_g8_e0_fresh_confirmation_implementation_freeze_v1.json"
)
FREEZE_SCHEMA = "maven_ere_g8_e0_fresh_confirmation_implementation_freeze_v1"
ROOT_RELATIVE = Path("artifacts/maven_ere_g8_e0_fresh_confirmation_v1")
ACQUISITION_ROOT_NAME = "acquisition"
CONTROLLER_ROOT_NAME = "controller"
HIPPORAG_STAGE_NAME = "official_hipporag_item_work"
V1_ACQUISITION_RELATIVE = Path("artifacts/maven_ere_g8_e1_formal_v1/acquisition")
V1_CONTROLLER_RELATIVE = Path("artifacts/maven_ere_g8_e1_formal_v1/controller")
ORIGINAL_ACQUISITION_RECEIPT_FILE_SHA256 = (
    "5ced719eaa73e62234fbdc21530a34f927fb0921a5fe1ee50b68c958c5737704"
)
ORIGINAL_ACQUISITION_RECEIPT_SELF_SHA256 = (
    "7842feff65de8e659bd0953c84e4c8f7a0a4a96ff005628a4d3f1369f4e03d74"
)
G8_MODEL_FILE_SHA256 = "7fc8bee5e794e774a4b82ddf44b477279b1e0d6a78137a570834738db27a0db4"
ORIGINAL_VALID_SPECS = (("A_hold", 10), ("M_search", 10))
FRESH_SPECS = (("A_hold", 20),)
PRIMARY_ALPHA = Fraction(1, 10)
IMPLEMENTATION_ROLE_PATHS: Mapping[str, Path] = {
    "design": DESIGN_RELATIVE,
    "E2_terminal_disposition": E2_DISPOSITION_RELATIVE,
    "implementation": Path(
        "assumption_agent/benchmarks/maven_ere_g8_e0_fresh_confirmation_v1.py"
    ),
    "test": Path("tests/test_maven_ere_g8_e0_fresh_confirmation_v1.py"),
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MavenEreFreshConfirmationError(RuntimeError):
    """The frozen fresh-confirmation contract drifted."""


class OneShotRefusal(MavenEreFreshConfirmationError):
    """The fresh formal root is not pristine."""


@dataclass(frozen=True)
class AssignmentBundle:
    selected: Mapping[tuple[str, str], tuple[base_acquisition._Candidate, ...]]
    selected_components: frozenset[int]
    collision_component_count: int
    eligible_candidate_occurrence_count: int
    total_cost: int


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise MavenEreFreshConfirmationError("bound file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return base_acquisition.stable_hash(value)


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = stable_hash(result)
    return result


def parse_valid_member(raw: bytes) -> tuple[base_acquisition._Document, ...]:
    lines = raw.splitlines()
    if len(lines) != base_acquisition.EXPECTED_LINE_COUNTS["valid"]:
        raise MavenEreFreshConfirmationError("valid member line count drifted")
    documents: list[base_acquisition._Document] = []
    for index, line in enumerate(lines):
        try:
            value = base_acquisition.qualification._strict_json_line(line)
            document = base_acquisition._parse_document(
                value, split="valid", split_index=index
            )
        except Exception as exc:
            raise MavenEreFreshConfirmationError(
                "valid member reader equivalence failed"
            ) from exc
        documents.append(document)
    return tuple(documents)


def assign_candidates(
    documents: Sequence[base_acquisition._Document],
    *,
    secret: bytes,
    specs: Sequence[tuple[str, int]],
    excluded_components: frozenset[int] = frozenset(),
) -> AssignmentBundle:
    if not documents or not specs:
        raise MavenEreFreshConfirmationError("assignment inputs are empty")
    if len(secret) != 32:
        raise MavenEreFreshConfirmationError("assignment secret length drifted")
    if len(set(block for block, _quota in specs)) != len(specs):
        raise MavenEreFreshConfirmationError("assignment block names collide")
    components, document_to_component = base_acquisition._collision_components(documents)
    if any(value < 0 or value >= len(components) for value in excluded_components):
        raise MavenEreFreshConfirmationError("excluded component ordinal drifted")
    demands: dict[tuple[str, str], int] = {}
    best: dict[
        int, dict[tuple[str, str], base_acquisition._Candidate]
    ] = {component: {} for component in range(len(components))}
    eligible = 0
    for block, quota in specs:
        if not isinstance(block, str) or not block or quota <= 0:
            raise MavenEreFreshConfirmationError("assignment spec drifted")
        for family in core.FAMILY_ORDER:
            demands[(block, family)] = quota
        for document_index, document in enumerate(documents):
            component = document_to_component[document_index]
            if component in excluded_components:
                continue
            for family in core.FAMILY_ORDER:
                candidate = base_acquisition._candidate_for_target(
                    secret=secret,
                    document=document,
                    document_index=document_index,
                    block=block,
                    family=family,
                )
                if candidate is None:
                    continue
                eligible += len(document.family_pairs[family])
                target = (block, family)
                incumbent = best[component].get(target)
                if incumbent is None or (
                    candidate.selection_digest,
                    candidate.tie_break,
                ) < (incumbent.selection_digest, incumbent.tie_break):
                    best[component][target] = candidate
    choices = {
        component: {
            target: base_acquisition._EdgeChoice(
                candidate.cost, candidate.tie_break, candidate
            )
            for target, candidate in rows.items()
        }
        for component, rows in best.items()
        if rows and component not in excluded_components
    }
    solution = base_acquisition.deterministic_min_cost_assignment(choices, demands)
    if solution.assigned_count != solution.required_count:
        raise MavenEreFreshConfirmationError("fresh exact assignment shortfall")
    selected: dict[
        tuple[str, str], tuple[base_acquisition._Candidate, ...]
    ] = {}
    selected_components: set[int] = set()
    for target, quota in demands.items():
        rows = tuple(solution.selected[target])
        if len(rows) != quota:
            raise MavenEreFreshConfirmationError("fresh target quota drifted")
        selected[target] = rows
        for row in rows:
            selected_components.add(document_to_component[row.document_index])
    if (
        len(selected_components) != solution.required_count
        or selected_components.intersection(excluded_components)
    ):
        raise MavenEreFreshConfirmationError("fresh component disjointness drifted")
    return AssignmentBundle(
        selected=selected,
        selected_components=frozenset(selected_components),
        collision_component_count=len(components),
        eligible_candidate_occurrence_count=eligible,
        total_cost=solution.total_cost,
    )


def build_packs(
    documents: Sequence[base_acquisition._Document],
    bundle: AssignmentBundle,
    *,
    specs: Sequence[tuple[str, int]],
) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    views: dict[str, Mapping[str, Any]] = {}
    labels: dict[str, Mapping[str, Any]] = {}
    all_ids: set[str] = set()
    for block, quota in specs:
        selected: list[base_acquisition._Candidate] = []
        for family in core.FAMILY_ORDER:
            rows = list(bundle.selected[(block, family)])
            if len(rows) != quota:
                raise MavenEreFreshConfirmationError("pack target quota drifted")
            selected.extend(rows)
        selected.sort(key=lambda row: row.item_id)
        if any(row.item_id in all_ids for row in selected):
            raise MavenEreFreshConfirmationError("pack private item ID collision")
        all_ids.update(row.item_id for row in selected)
        view_rows = [
            base_acquisition._candidate_view(row, documents[row.document_index])
            for row in selected
        ]
        label_rows = [
            {"family": row.family, "item_id": row.item_id} for row in selected
        ]
        views[block] = base_acquisition._pack(
            "maven_ere_g8_e1_action_view_pack_v1", block, view_rows
        )
        labels[block] = base_acquisition._pack(
            "maven_ere_g8_e1_family_label_pack_v1", block, label_rows
        )
    return views, labels


def _pack_binding(pack: Mapping[str, Any]) -> dict[str, Any]:
    raw = base_acquisition.canonical_json(pack)
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "item_count": pack["item_count"],
        "pack_sha256": pack["pack_sha256"],
        "size_bytes": len(raw),
    }


def _read_original_acquisition_receipt(project: Path) -> Mapping[str, Any]:
    path = project / V1_ACQUISITION_RELATIVE / "acquisition.receipt.json"
    if _sha256_file(path) != ORIGINAL_ACQUISITION_RECEIPT_FILE_SHA256:
        raise MavenEreFreshConfirmationError("original acquisition receipt drifted")
    try:
        value = json.loads(path.read_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreFreshConfirmationError(
            "original acquisition receipt is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise MavenEreFreshConfirmationError("original acquisition receipt root drifted")
    body = dict(value)
    declared = body.pop("acquisition_sha256", None)
    if (
        declared != ORIGINAL_ACQUISITION_RECEIPT_SELF_SHA256
        or stable_hash(body) != ORIGINAL_ACQUISITION_RECEIPT_SELF_SHA256
        or value.get("status")
        != "passed_one_shot_private_document_disjoint_acquisition"
    ):
        raise MavenEreFreshConfirmationError(
            "original acquisition receipt self hash drifted"
        )
    return value


def _validate_reconstructed_original_packs(
    receipt: Mapping[str, Any],
    views: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    view_bindings = receipt.get("view_pack_bindings")
    label_bindings = receipt.get("label_pack_bindings")
    if not isinstance(view_bindings, Mapping) or not isinstance(label_bindings, Mapping):
        raise MavenEreFreshConfirmationError("original public pack bindings drifted")
    proofs: dict[str, Any] = {}
    for block, _quota in ORIGINAL_VALID_SPECS:
        generated_view = _pack_binding(views[block])
        generated_label = _pack_binding(labels[block])
        original_view = view_bindings.get(block)
        original_label = label_bindings.get(block)
        for generated, original in (
            (generated_view, original_view),
            (generated_label, original_label),
        ):
            if not isinstance(original, Mapping) or any(
                generated[key] != original.get(key)
                for key in ("file_sha256", "item_count", "pack_sha256", "size_bytes")
            ):
                raise MavenEreFreshConfirmationError(
                    "original assignment regeneration proof failed"
                )
        proofs[block] = {
            "label_file_sha256": generated_label["file_sha256"],
            "label_pack_sha256": generated_label["pack_sha256"],
            "view_file_sha256": generated_view["file_sha256"],
            "view_pack_sha256": generated_view["pack_sha256"],
        }
    return proofs


def load_g8_model(
    path: str | Path,
    *,
    expected_file_sha256: str = G8_MODEL_FILE_SHA256,
) -> core.G8Model:
    model_path = Path(path).absolute()
    if _sha256_file(model_path) != expected_file_sha256:
        raise MavenEreFreshConfirmationError("fixed G8 model file hash drifted")
    try:
        value = json.loads(model_path.read_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreFreshConfirmationError("fixed G8 model is invalid") from exc
    exact = {
        "centered_target_sha256",
        "coefficient_sha256",
        "fit_sha256",
        "item_count",
        "model_sha256",
        "normal_equation_sha256",
        "observation_weight_sha256",
        "schema",
        "set_observation_count",
        "weights_hex",
    }
    if not isinstance(value, dict) or set(value) != exact:
        raise MavenEreFreshConfirmationError("fixed G8 model shape drifted")
    body = dict(value)
    declared = body.pop("model_sha256", None)
    if (
        not isinstance(declared, str)
        or stable_hash(body) != declared
        or value.get("schema") != "maven_ere_G8_model_v1"
        or value.get("item_count") != 96
    ):
        raise MavenEreFreshConfirmationError("fixed G8 model self hash drifted")
    weights_hex = value.get("weights_hex")
    if not isinstance(weights_hex, list) or len(weights_hex) != len(core.G8_FEATURE_ORDER):
        raise MavenEreFreshConfirmationError("fixed G8 weight shape drifted")
    try:
        weights = tuple(float.fromhex(item) for item in weights_hex)
    except (TypeError, ValueError) as exc:
        raise MavenEreFreshConfirmationError("fixed G8 weight encoding drifted") from exc
    return core.G8Model(
        weights=weights,
        normal_equation_sha256=value["normal_equation_sha256"],
        observation_weight_sha256=value["observation_weight_sha256"],
        centered_target_sha256=value["centered_target_sha256"],
        coefficient_sha256=value["coefficient_sha256"],
        fit_sha256=value["fit_sha256"],
        item_count=value["item_count"],
        set_observation_count=value["set_observation_count"],
    )


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise MavenEreFreshConfirmationError("git provenance command failed")
    return completed.stdout


def _validate_public_binding(
    path: Path,
    *,
    file_sha256: str,
    self_field: str,
    self_sha256: str,
    schema: str,
    status: str,
) -> None:
    if _sha256_file(path) != file_sha256:
        raise MavenEreFreshConfirmationError("fresh public binding file hash drifted")
    try:
        value = json.loads(path.read_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreFreshConfirmationError("fresh public binding is invalid") from exc
    if not isinstance(value, dict):
        raise MavenEreFreshConfirmationError("fresh public binding root drifted")
    body = dict(value)
    declared = body.pop(self_field, None)
    if (
        declared != self_sha256
        or stable_hash(body) != self_sha256
        or value.get("schema") != schema
        or value.get("status") != status
    ):
        raise MavenEreFreshConfirmationError("fresh public binding self hash drifted")


def validate_formal_provenance(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    base = e2.validate_formal_provenance(project)
    repository = Path(
        _git_bytes(project, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve(strict=True)
    if (repository / "reconstruction_v2").resolve(strict=True) != project:
        raise MavenEreFreshConfirmationError("project/repository relationship drifted")
    _validate_public_binding(
        project / DESIGN_RELATIVE,
        file_sha256=DESIGN_FILE_SHA256,
        self_field="design_sha256",
        self_sha256=DESIGN_SELF_SHA256,
        schema="maven_ere_g8_e0_fresh_confirmation_design_v1",
        status=(
            "frozen_before_original_secret_or_valid_source_reopen_and_before_new_"
            "secret_cohort_or_model_inference"
        ),
    )
    _validate_public_binding(
        project / E2_DISPOSITION_RELATIVE,
        file_sha256=E2_DISPOSITION_FILE_SHA256,
        self_field="disposition_sha256",
        self_sha256=E2_DISPOSITION_SELF_SHA256,
        schema="maven_ere_global_family_e2_train_crossfit_result_disposition_v1",
        status="stopped_train_crossfit_no_formal_measurement",
    )
    freeze_path = project / FREEZE_RELATIVE
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise MavenEreFreshConfirmationError("fresh implementation freeze is unavailable")
    raw = freeze_path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreFreshConfirmationError("fresh implementation freeze is invalid") from exc
    if not isinstance(payload, dict):
        raise MavenEreFreshConfirmationError("fresh implementation freeze root drifted")
    body = dict(payload)
    declared = body.pop("implementation_freeze_sha256", None)
    if (
        not isinstance(declared, str)
        or not _SHA256.fullmatch(declared)
        or stable_hash(body) != declared
        or payload.get("schema") != FREEZE_SCHEMA
        or payload.get("version") != VERSION
        or payload.get("status")
        != "frozen_before_original_secret_valid_source_new_secret_cohort_or_model_inference"
    ):
        raise MavenEreFreshConfirmationError("fresh implementation freeze self hash drifted")
    implementation_commit = payload.get("implementation_commit")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise MavenEreFreshConfirmationError("fresh implementation commit drifted")
    binding = payload.get("implementation_binding")
    files = binding.get("files") if isinstance(binding, Mapping) else None
    if (
        not isinstance(files, list)
        or binding.get("file_count") != len(IMPLEMENTATION_ROLE_PATHS)
        or len(files) != len(IMPLEMENTATION_ROLE_PATHS)
    ):
        raise MavenEreFreshConfirmationError("fresh implementation registry drifted")
    expected: list[tuple[Path, str]] = []
    for row, (role, relative) in zip(
        files, IMPLEMENTATION_ROLE_PATHS.items(), strict=True
    ):
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "role",
            "sha256",
        }:
            raise MavenEreFreshConfirmationError("fresh implementation row shape drifted")
        digest = row.get("sha256")
        if (
            row.get("role") != role
            or row.get("relative_path") != relative.as_posix()
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or _sha256_file(project / relative) != digest
        ):
            raise MavenEreFreshConfirmationError("fresh implementation row drifted")
        expected.append((relative, digest))
    if payload.get("claim_boundary") != {
        "model_or_HippoRAG_inference_run": False,
        "new_selection_secret_or_cohort_generated": False,
        "official_valid_source_reopened": False,
        "online_or_external_evaluation_used": False,
        "original_private_view_or_label_pack_opened": False,
        "original_selection_secret_reopened": False,
    }:
        raise MavenEreFreshConfirmationError("fresh implementation claim boundary drifted")
    if _git_bytes(repository, "cat-file", "-t", implementation_commit).strip() != b"commit":
        raise MavenEreFreshConfirmationError("fresh implementation commit unavailable")
    _git_bytes(repository, "merge-base", "--is-ancestor", implementation_commit, "HEAD")
    repository_paths = [
        "reconstruction_v2/" + relative.as_posix() for relative, _digest in expected
    ]
    freeze_repository_path = "reconstruction_v2/" + FREEZE_RELATIVE.as_posix()
    for path in (*repository_paths, freeze_repository_path):
        _git_bytes(repository, "ls-files", "--error-unmatch", "--", path)
    _git_bytes(
        repository,
        "diff",
        "--quiet",
        "HEAD",
        "--",
        *repository_paths,
        freeze_repository_path,
    )
    for relative, digest in expected:
        repository_path = "reconstruction_v2/" + relative.as_posix()
        committed = _git_bytes(
            repository, "show", f"{implementation_commit}:{repository_path}"
        )
        if hashlib.sha256(committed).hexdigest() != digest:
            raise MavenEreFreshConfirmationError("fresh committed blob drifted")
    return {
        "base_E2_provenance": base,
        "implementation_commit": implementation_commit,
        "implementation_freeze_file_sha256": hashlib.sha256(raw).hexdigest(),
        "implementation_freeze_self_sha256": declared,
    }


def _runtime_config(project: Path) -> local_runtime.FormalRuntimeConfig:
    canonical = local_runtime.default_formal_runtime_config(project)
    return replace(
        canonical,
        hippo_stage_root=project / ROOT_RELATIVE / HIPPORAG_STAGE_NAME,
    )


def preflight_runtime(config: local_runtime.FormalRuntimeConfig) -> dict[str, Any]:
    canonical = local_runtime.default_formal_runtime_config(config.project)
    expected_stage = config.project / ROOT_RELATIVE / HIPPORAG_STAGE_NAME
    if (
        config.hippo_stage_root != expected_stage
        or replace(config, hippo_stage_root=canonical.hippo_stage_root) != canonical
        or os.path.lexists(expected_stage)
    ):
        raise MavenEreFreshConfirmationError("fresh runtime config drifted")
    try:
        minilm = minilm_binding.verify_runtime_binding(
            asset_manifest_path=config.minilm_asset_manifest,
            model_root=config.minilm_model_root,
        )
        design = verify_maven_design(config.project)
        nli = nli_binding.verify_runtime_binding(
            asset_manifest_path=config.nli_asset_manifest,
            model_root=config.nli_model_root,
        )
        hippo = verify_formal_runtime_attestation_v3(
            project_root=config.project,
            attestation_receipt_path=config.hippo_attestation_receipt,
            base_binding_receipt_path=config.hippo_base_binding_receipt,
            runtime_python=config.hippo_runtime_python,
            local_llm_model=config.hippo_llm_model,
            local_embedding_model=config.hippo_embedding_model,
        )
    except Exception as exc:
        raise MavenEreFreshConfirmationError("fresh offline runtime preflight failed") from exc
    return {
        "external_network_calls": 0,
        "hipporag_runtime_attestation": dict(hippo),
        "minilm_runtime_binding": dict(minilm),
        "model_inference_calls": 0,
        "nli_design_binding": dict(design),
        "nli_runtime_binding": dict(nli),
        "original_private_pack_reads": 0,
        "original_secret_reads": 0,
        "schema": "maven_ere_g8_e0_fresh_confirmation_runtime_preflight_v1",
        "valid_source_reads": 0,
        "version": VERSION,
    }


def load_fresh_labels(
    path: str | Path,
    *,
    expected_item_ids: Sequence[str],
) -> Mapping[str, str]:
    pack = local_runtime._read_pack(
        path,
        schema="maven_ere_g8_e1_family_label_pack_v1",
        block="A_hold",
    )
    items = pack["items"]
    if len(items) != 60:
        raise MavenEreFreshConfirmationError("fresh label count drifted")
    result: dict[str, str] = {}
    for row in items:
        if not isinstance(row, Mapping) or set(row) != {"family", "item_id"}:
            raise MavenEreFreshConfirmationError("fresh label row shape drifted")
        item_id, family = row.get("item_id"), row.get("family")
        if (
            not isinstance(item_id, str)
            or not _SHA256.fullmatch(item_id)
            or item_id in result
            or family not in core.FAMILY_ORDER
        ):
            raise MavenEreFreshConfirmationError("fresh label row drifted")
        result[item_id] = str(family)
    if set(result) != set(expected_item_ids):
        raise MavenEreFreshConfirmationError("fresh label/view keyset drifted")
    counts = {
        family: sum(value == family for value in result.values())
        for family in core.FAMILY_ORDER
    }
    if counts != {family: 20 for family in core.FAMILY_ORDER}:
        raise MavenEreFreshConfirmationError("fresh label family balance drifted")
    return result


def score_confirmation(
    execution: v1.BlockExecution,
    labels: Mapping[str, str],
) -> dict[str, Any]:
    if execution.block != "A_hold" or len(execution.items) != 60:
        raise MavenEreFreshConfirmationError("fresh scored execution drifted")
    families = [labels[row.prepared.view.item_id] for row in execution.items]
    utilities: dict[str, list[int]] = {
        arm: [] for arm in ("RAW", "HippoRAG", "E0")
    }
    for row, family in zip(execution.items, families, strict=True):
        if row.agent.e1_selected is not None:
            raise MavenEreFreshConfirmationError("fresh E0 execution contains E1")
        utilities["RAW"].append(
            core.utility(row.raw.selected, family, row.prepared.item)
        )
        utilities["HippoRAG"].append(
            core.utility(row.hippo.selected, family, row.prepared.item)
        )
        utilities["E0"].append(
            core.utility(row.agent.e0_selected, family, row.prepared.item)
        )
    return {
        "arm_correct_count": {arm: sum(values) for arm, values in utilities.items()},
        "comparisons": {
            "E0_minus_HippoRAG": v1._comparison(
                utilities["E0"], utilities["HippoRAG"], families
            ),
            "E0_minus_RAW": v1._comparison(
                utilities["E0"], utilities["RAW"], families
            ),
        },
        "edge_deletion_action_change_item_count": sum(
            row.agent.edge_deletion_action_change_count > 0
            for row in execution.items
        ),
        "edge_deletion_witness_count": sum(
            row.agent.edge_deletion_witness_count for row in execution.items
        ),
        "family_item_count": {
            family: sum(value == family for value in families)
            for family in core.FAMILY_ORDER
        },
        "item_count": len(families),
    }


def _comparison_passed(comparison: Mapping[str, Any]) -> bool:
    p = comparison["p_value"]
    return bool(
        comparison["net_utility"] > 0
        and int(p["numerator"]) * PRIMARY_ALPHA.denominator
        <= int(p["denominator"]) * PRIMARY_ALPHA.numerator
        and all(value > 0 for value in comparison["family_net"].values())
    )


def primary_passed(score: Mapping[str, Any]) -> bool:
    comparisons = score["comparisons"]
    return bool(
        _comparison_passed(comparisons["E0_minus_HippoRAG"])
        and _comparison_passed(comparisons["E0_minus_RAW"])
    )


def run_fresh_confirmation(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    if project.name != "reconstruction_v2":
        raise MavenEreFreshConfirmationError("project root must be reconstruction_v2")
    provenance = validate_formal_provenance(project)
    receipt = _read_original_acquisition_receipt(project)
    config = _runtime_config(project)
    preflight = preflight_runtime(config)
    root = project / ROOT_RELATIVE
    if os.path.lexists(root):
        raise OneShotRefusal("fresh confirmation root already exists")
    root.mkdir(mode=0o700, parents=True)
    acquisition_root = root / ACQUISITION_ROOT_NAME
    controller_root = root / CONTROLLER_ROOT_NAME
    acquisition_root.mkdir(mode=0o700)
    controller_root.mkdir(mode=0o700)
    marker = _self_hashed(
        {
            "design_sha256": DESIGN_SELF_SHA256,
            "provenance": provenance,
            "schema": "maven_ere_g8_e0_fresh_confirmation_authorization_v1",
            "status": "consumed_before_original_secret_or_valid_source_open",
            "version": VERSION,
        },
        "marker_sha256",
    )
    v1._exclusive_write(root / "authorization.consumed.json", marker)
    v1._durable_roundtrip(controller_root / "runtime.preflight.json", preflight)
    phase = "original_assignment_reconstruction"
    try:
        secret_path = project / V1_ACQUISITION_RELATIVE / "selection.secret"
        if (
            secret_path.is_symlink()
            or not secret_path.is_file()
            or secret_path.stat().st_size != 32
            or stat.S_IMODE(secret_path.stat().st_mode) != 0o600
        ):
            raise MavenEreFreshConfirmationError("original selection secret drifted")
        original_secret = secret_path.read_bytes()
        if base_acquisition._secret_commitment(original_secret) != receipt.get(
            "secret_commitment"
        ):
            raise MavenEreFreshConfirmationError(
                "original selection secret commitment drifted"
            )
        valid_relative, valid_size, valid_sha = base_acquisition.SOURCE_SPECS["valid"]
        valid_raw = base_acquisition._read_bound_source(
            project / valid_relative, size=valid_size, digest=valid_sha
        )
        documents = parse_valid_member(valid_raw)
        original_assignment = assign_candidates(
            documents,
            secret=original_secret,
            specs=ORIGINAL_VALID_SPECS,
        )
        original_views, original_labels = build_packs(
            documents, original_assignment, specs=ORIGINAL_VALID_SPECS
        )
        regeneration_proof = _validate_reconstructed_original_packs(
            receipt, original_views, original_labels
        )
        if len(original_assignment.selected_components) != 60:
            raise MavenEreFreshConfirmationError(
                "original valid exclusion component count drifted"
            )

        phase = "fresh_assignment"
        fresh_secret = os.urandom(32)
        fresh_assignment = assign_candidates(
            documents,
            secret=fresh_secret,
            specs=FRESH_SPECS,
            excluded_components=original_assignment.selected_components,
        )
        fresh_views, fresh_labels = build_packs(
            documents, fresh_assignment, specs=FRESH_SPECS
        )
        if len(fresh_assignment.selected_components) != 60:
            raise MavenEreFreshConfirmationError("fresh component count drifted")
        private = acquisition_root / "private_packs"
        private.mkdir(mode=0o700)
        base_acquisition._exclusive_write(
            acquisition_root / "selection.secret", fresh_secret
        )
        view_path = private / "A_hold.view.json"
        label_path = private / "A_hold.labels.json"
        base_acquisition._atomic_write(
            view_path, base_acquisition.canonical_json(fresh_views["A_hold"])
        )
        base_acquisition._atomic_write(
            label_path, base_acquisition.canonical_json(fresh_labels["A_hold"])
        )
        acquisition_receipt = _self_hashed(
            {
                "claim_boundary": {
                    "hidden_TEST_opened": False,
                    "new_secret_generation_count": 1,
                    "online_or_external_evaluator_calls": 0,
                    "original_private_pack_opens": 0,
                    "original_selection_secret_open_count": 1,
                    "released_train_source_open_count": 0,
                    "released_valid_source_open_count": 1,
                },
                "collision_component_count": fresh_assignment.collision_component_count,
                "eligible_candidate_occurrence_count_after_exclusion": (
                    fresh_assignment.eligible_candidate_occurrence_count
                ),
                "excluded_original_valid_component_count": 60,
                "fresh_label_pack_binding": base_acquisition._public_pack_binding(
                    acquisition_root, label_path, fresh_labels["A_hold"]
                ),
                "fresh_secret_commitment": base_acquisition._secret_commitment(
                    fresh_secret
                ),
                "fresh_selected_item_count": 60,
                "fresh_view_pack_binding": base_acquisition._public_pack_binding(
                    acquisition_root, view_path, fresh_views["A_hold"]
                ),
                "original_assignment_regeneration_proof": regeneration_proof,
                "schema": "maven_ere_g8_e0_fresh_confirmation_acquisition_v1",
                "status": "passed_exact_original_exclusion_and_fresh_assignment",
                "version": VERSION,
            },
            "acquisition_sha256",
        )
        v1._durable_roundtrip(
            acquisition_root / "acquisition.receipt.json", acquisition_receipt
        )

        phase = "fixed_G8_and_label_free_actions"
        g8 = load_g8_model(project / V1_CONTROLLER_RELATIVE / "G8.model.json")
        with recovery.RecoveryRuntimeBundle(config) as runtime:
            assert runtime.hippo is not None
            runtime.hippo.prepare_blocks(("A_hold",))
            views = local_runtime.load_view_pack(view_path, block="A_hold")
            if len(views) != 60:
                raise MavenEreFreshConfirmationError("fresh view count drifted")
            prepared = runtime.prepare_block("A_hold", views)
            semantic_file_sha = v1._durable_roundtrip(
                controller_root / "A_hold.semantic.archive.json",
                v1._semantic_archive(prepared),
            )
            execution = v1.execute_block(
                prepared=prepared,
                g8_model=g8,
                e1_model=None,
                hippo=runtime.hippo,
                causal_audit=True,
            )
            action_file_sha = v1._durable_roundtrip(
                controller_root / "A_hold.action.archive.json",
                recovery.normalized_action_archive(execution),
            )
            phase = "fresh_label_open_and_score"
            labels = load_fresh_labels(
                label_path,
                expected_item_ids=tuple(row.view.item_id for row in prepared.items),
            )
            score = score_confirmation(execution, labels)
            passed = primary_passed(score)
            report = _self_hashed(
                {
                    "A_hold_action_file_sha256": action_file_sha,
                    "A_hold_semantic_file_sha256": semantic_file_sha,
                    "G8_fit_sha256": g8.fit_sha256,
                    "fresh_primary_passed": passed,
                    "schema": "maven_ere_g8_e0_fresh_confirmation_report_v1",
                    "score": score,
                    "version": VERSION,
                },
                "report_sha256",
            )
            report_file_sha = v1._durable_roundtrip(
                controller_root / "A_hold.aggregate.report.json", report
            )

        phase = "terminal_result"
        terminal = _self_hashed(
            {
                "A_hold": report,
                "acquisition_receipt_sha256": acquisition_receipt[
                    "acquisition_sha256"
                ],
                "claim_boundary": {
                    "hidden_TEST_opened": False,
                    "online_or_external_evaluator_calls": 0,
                    "original_M_search_private_path_opened": False,
                    "original_private_pack_opens": 0,
                    "released_train_source_open_count": 0,
                    "released_valid_source_open_count": 1,
                    "same_cohort_retry_or_resample": False,
                },
                "fresh_primary_passed": passed,
                "provenance": provenance,
                "report_file_sha256": report_file_sha,
                "schema": "maven_ere_g8_e0_fresh_confirmation_terminal_v1",
                "status": (
                    "valid_fresh_confirmation_primary_passed"
                    if passed
                    else "valid_fresh_confirmation_primary_failed"
                ),
                "version": VERSION,
            },
            "terminal_result_sha256",
        )
        v1._durable_roundtrip(controller_root / "terminal.result.json", terminal)
        return terminal
    except BaseException as exc:
        failure = _self_hashed(
            {
                "category": type(exc).__name__,
                "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                "phase": phase,
                "schema": "maven_ere_g8_e0_fresh_confirmation_failure_v1",
                "status": "terminal_no_retry",
                "version": VERSION,
            },
            "failure_sha256",
        )
        try:
            v1._atomic_write(root / "formal.failure.json", failure)
        except BaseException:
            pass
        raise


__all__ = [
    "AssignmentBundle",
    "MavenEreFreshConfirmationError",
    "assign_candidates",
    "build_packs",
    "load_g8_model",
    "parse_valid_member",
    "preflight_runtime",
    "primary_passed",
    "run_fresh_confirmation",
    "score_confirmation",
    "validate_formal_provenance",
]
