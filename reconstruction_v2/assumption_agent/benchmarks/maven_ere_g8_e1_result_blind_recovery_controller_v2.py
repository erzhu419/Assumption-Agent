"""Result-blind continuation of the terminal MAVEN-ERE v1 serializer failure."""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import maven_ere_g8_e1_acquisition_v1 as acquisition
from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core
from assumption_agent.benchmarks import maven_ere_g8_e1_formal_controller_v1 as v1
from assumption_agent.benchmarks import maven_ere_local_runtime_v1 as local_runtime
from assumption_agent.benchmarks.maven_ere_nli_runtime_v1 import (
    MavenEreNLIWorkerPool,
    verify_maven_design,
)
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)
from replication_runtime.qasc_nli_v1 import binding as nli_binding
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding


VERSION = "v2"
SCHEMA = "maven_ere_g8_e1_result_blind_recovery_controller_v2"
RECOVERY_DESIGN_RELATIVE = Path(
    "manifests/maven_ere_g8_e1_result_blind_recovery_design_v2.json"
)
RECOVERY_DESIGN_FILE_SHA256 = (
    "b9747342f160578c0bbc222419ee15ddec0e1d84d829c415ed753d5d82e0e777"
)
RECOVERY_DESIGN_SELF_SHA256 = (
    "abd62be274705da4d442fbc79be11251a655ff0e368dd18126f13aad48c64c26"
)
V1_DISPOSITION_RELATIVE = Path(
    "manifests/maven_ere_g8_e1_formal_v1_implementation_failure_disposition_v1.json"
)
V1_DISPOSITION_FILE_SHA256 = (
    "b78f1b18f7797844eb05fdcc1c159ae3247f2c8d419f7b027c5014e05c94991c"
)
V1_DISPOSITION_SELF_SHA256 = (
    "c532a18f54393902dd9eb3db6e413c6195f97a288b9b1f8e3f3b60a5b1654d08"
)
RECOVERY_FREEZE_RELATIVE = Path(
    "manifests/maven_ere_g8_e1_result_blind_recovery_implementation_freeze_v2.json"
)
RECOVERY_FREEZE_SCHEMA = (
    "maven_ere_g8_e1_result_blind_recovery_implementation_freeze_v2"
)
V1_FORMAL_ROOT_RELATIVE = Path("artifacts/maven_ere_g8_e1_formal_v1")
V1_ACQUISITION_RELATIVE = V1_FORMAL_ROOT_RELATIVE / "acquisition"
V1_CONTROLLER_RELATIVE = V1_FORMAL_ROOT_RELATIVE / "controller"
RECOVERY_ROOT_RELATIVE = Path(
    "artifacts/maven_ere_g8_e1_result_blind_recovery_v2"
)
RECOVERY_CONTROLLER_RELATIVE = RECOVERY_ROOT_RELATIVE / "controller"
RECOVERY_HIPPO_RELATIVE = RECOVERY_ROOT_RELATIVE / "official_hipporag_item_work"

V1_ARTIFACT_SHA256: Mapping[str, tuple[Path, str]] = {
    "acquisition_receipt": (
        V1_ACQUISITION_RELATIVE / "acquisition.receipt.json",
        "5ced719eaa73e62234fbdc21530a34f927fb0921a5fe1ee50b68c958c5737704",
    ),
    "failure_receipt": (
        V1_CONTROLLER_RELATIVE / "lifecycle.failure.json",
        "8e04cc2e062b91d5e8f96aa0cfaf2c2a82f10aa5960c2257a2a0f8eeab931188",
    ),
    "G_form_semantic_archive": (
        V1_CONTROLLER_RELATIVE / "G_form.semantic.archive.json",
        "d550eb9146e513bbc4f78bbae2b074d5b199d1b48385eff16469e6547fb044e6",
    ),
    "G8_model": (
        V1_CONTROLLER_RELATIVE / "G8.model.json",
        "7fc8bee5e794e774a4b82ddf44b477279b1e0d6a78137a570834738db27a0db4",
    ),
    "A_form_semantic_archive": (
        V1_CONTROLLER_RELATIVE / "A_form.semantic.archive.json",
        "51eeee0f1a2cafb5f587d5c230148f9d31f9c6611cef2f03ff7f919bfc749b2c",
    ),
    "A_form_action_archive": (
        V1_CONTROLLER_RELATIVE / "A_form.action.archive.json",
        "9d71572e4fb5197a19c98310e948da5c8e0a5009a95aa90921985a1f42cb6238",
    ),
}
RECOVERY_IMPLEMENTATION_ROLE_PATHS: Mapping[str, Path] = {
    "recovery_design": RECOVERY_DESIGN_RELATIVE,
    "v1_failure_disposition": V1_DISPOSITION_RELATIVE,
    "recovery_controller": Path(
        "assumption_agent/benchmarks/maven_ere_g8_e1_result_blind_recovery_controller_v2.py"
    ),
    "test_recovery_controller": Path(
        "tests/test_maven_ere_g8_e1_result_blind_recovery_controller_v2.py"
    ),
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MavenEreRecoveryError(RuntimeError):
    """The frozen result-blind recovery contract drifted."""


class OneShotRefusal(MavenEreRecoveryError):
    """The recovery root is not pristine."""


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise MavenEreRecoveryError("bound file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hashed(body: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    result = dict(body)
    result[field_name] = v1.stable_hash(result)
    return result


def _json_normalize(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON-domain value that the durable reader actually observes."""

    decoded = json.loads(v1._canonical_bytes(value).decode("ascii"))
    if not isinstance(decoded, dict):
        raise MavenEreRecoveryError("JSON normalization root drifted")
    return decoded


def normalized_action_archive(execution: v1.BlockExecution) -> dict[str, Any]:
    return _json_normalize(v1._action_archive(execution))


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise MavenEreRecoveryError("git provenance command failed")
    return completed.stdout


def _validate_self_hashed_public_file(
    path: Path,
    *,
    file_sha256: str,
    self_field: str,
    self_sha256: str,
    schema: str,
    status: str,
) -> Mapping[str, Any]:
    if _sha256_file(path) != file_sha256:
        raise MavenEreRecoveryError("public binding file hash drifted")
    try:
        value = json.loads(path.read_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreRecoveryError("public binding is invalid") from exc
    if not isinstance(value, dict):
        raise MavenEreRecoveryError("public binding root drifted")
    body = dict(value)
    declared = body.pop(self_field, None)
    if (
        declared != self_sha256
        or acquisition.stable_hash(body) != self_sha256
        or value.get("schema") != schema
        or value.get("status") != status
    ):
        raise MavenEreRecoveryError("public binding self hash drifted")
    return value


def validate_recovery_provenance(project_root: str | Path) -> dict[str, Any]:
    """Verify both the untouched v1 closure and the separately frozen recovery."""

    project = Path(project_root).resolve(strict=True)
    base = acquisition.validate_formal_provenance(project)
    repository = Path(
        _git_bytes(project, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve(strict=True)
    if (repository / "reconstruction_v2").resolve(strict=True) != project:
        raise MavenEreRecoveryError("project/repository relationship drifted")
    _validate_self_hashed_public_file(
        project / RECOVERY_DESIGN_RELATIVE,
        file_sha256=RECOVERY_DESIGN_FILE_SHA256,
        self_field="recovery_design_sha256",
        self_sha256=RECOVERY_DESIGN_SELF_SHA256,
        schema="maven_ere_g8_e1_result_blind_recovery_design_v2",
        status="frozen_before_v1_private_reopen_or_recovery_model_inference",
    )
    _validate_self_hashed_public_file(
        project / V1_DISPOSITION_RELATIVE,
        file_sha256=V1_DISPOSITION_FILE_SHA256,
        self_field="disposition_sha256",
        self_sha256=V1_DISPOSITION_SELF_SHA256,
        schema="maven_ere_g8_e1_formal_v1_implementation_failure_disposition_v1",
        status="implementation_invalid_efficacy_unknown_terminal_no_v1_replay",
    )
    freeze_path = project / RECOVERY_FREEZE_RELATIVE
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise MavenEreRecoveryError("recovery implementation freeze is unavailable")
    raw = freeze_path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreRecoveryError("recovery implementation freeze is invalid") from exc
    if not isinstance(payload, dict):
        raise MavenEreRecoveryError("recovery implementation freeze root drifted")
    body = dict(payload)
    declared = body.pop("implementation_freeze_sha256", None)
    if (
        not isinstance(declared, str)
        or not _SHA256.fullmatch(declared)
        or acquisition.stable_hash(body) != declared
        or payload.get("schema") != RECOVERY_FREEZE_SCHEMA
        or payload.get("version") != VERSION
        or payload.get("status")
        != "frozen_before_v1_private_reopen_or_recovery_model_inference"
    ):
        raise MavenEreRecoveryError("recovery implementation freeze self hash drifted")
    implementation_commit = payload.get("implementation_commit")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise MavenEreRecoveryError("recovery implementation commit is invalid")
    binding = payload.get("implementation_binding")
    files = binding.get("files") if isinstance(binding, Mapping) else None
    if (
        not isinstance(files, list)
        or binding.get("file_count") != len(RECOVERY_IMPLEMENTATION_ROLE_PATHS)
        or len(files) != len(RECOVERY_IMPLEMENTATION_ROLE_PATHS)
    ):
        raise MavenEreRecoveryError("recovery implementation registry drifted")
    expected: list[tuple[str, Path, str]] = []
    for row, (role, relative) in zip(
        files, RECOVERY_IMPLEMENTATION_ROLE_PATHS.items(), strict=True
    ):
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "role",
            "sha256",
        }:
            raise MavenEreRecoveryError("recovery implementation row shape drifted")
        digest = row.get("sha256")
        if (
            row.get("role") != role
            or row.get("relative_path") != relative.as_posix()
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or _sha256_file(project / relative) != digest
        ):
            raise MavenEreRecoveryError("recovery implementation row drifted")
        expected.append((role, relative, digest))
    if payload.get("claim_boundary") != {
        "A_form_three_arm_actions_reexecuted": False,
        "hidden_TEST_opened": False,
        "new_selection_secret_or_cohort_generated": False,
        "online_or_external_evaluation_used": False,
        "private_efficacy_label_or_score_opened_after_v1_failure": False,
        "released_train_or_valid_source_rows_reopened": False,
    }:
        raise MavenEreRecoveryError("recovery implementation claim boundary drifted")
    if _git_bytes(repository, "cat-file", "-t", implementation_commit).strip() != b"commit":
        raise MavenEreRecoveryError("recovery implementation commit is unavailable")
    _git_bytes(repository, "merge-base", "--is-ancestor", implementation_commit, "HEAD")
    repository_paths = [
        "reconstruction_v2/" + relative.as_posix()
        for _role, relative, _digest in expected
    ]
    freeze_repository_path = "reconstruction_v2/" + RECOVERY_FREEZE_RELATIVE.as_posix()
    for repository_path in (*repository_paths, freeze_repository_path):
        _git_bytes(repository, "ls-files", "--error-unmatch", "--", repository_path)
    _git_bytes(
        repository,
        "diff",
        "--quiet",
        "HEAD",
        "--",
        *repository_paths,
        freeze_repository_path,
    )
    for _role, relative, digest in expected:
        repository_path = "reconstruction_v2/" + relative.as_posix()
        committed = _git_bytes(
            repository, "show", f"{implementation_commit}:{repository_path}"
        )
        if hashlib.sha256(committed).hexdigest() != digest:
            raise MavenEreRecoveryError("recovery committed blob drifted")
    return {
        "base_implementation_provenance": base,
        "recovery_implementation_commit": implementation_commit,
        "recovery_implementation_freeze_file_sha256": hashlib.sha256(raw).hexdigest(),
        "recovery_implementation_freeze_self_sha256": declared,
    }


def _recovery_runtime_config(project: Path) -> local_runtime.FormalRuntimeConfig:
    canonical = local_runtime.default_formal_runtime_config(project)
    return replace(
        canonical,
        hippo_stage_root=project / RECOVERY_HIPPO_RELATIVE,
    )


def preflight_recovery_runtime(
    config: local_runtime.FormalRuntimeConfig,
) -> dict[str, Any]:
    canonical = local_runtime.default_formal_runtime_config(config.project)
    expected_stage = config.project / RECOVERY_HIPPO_RELATIVE
    if (
        config.hippo_stage_root != expected_stage
        or replace(config, hippo_stage_root=canonical.hippo_stage_root) != canonical
        or os.path.lexists(expected_stage)
    ):
        raise MavenEreRecoveryError("recovery runtime config drifted")
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
        raise MavenEreRecoveryError("offline recovery runtime preflight failed") from exc
    return {
        "benchmark_source_or_private_pack_reads": 0,
        "external_network_calls": 0,
        "hipporag_runtime_attestation": dict(hippo),
        "minilm_runtime_binding": dict(minilm),
        "model_inference_calls": 0,
        "nli_design_binding": dict(design),
        "nli_runtime_binding": dict(nli),
        "schema": "maven_ere_result_blind_recovery_runtime_preflight_v2",
        "version": VERSION,
    }


class RecoveryRuntimeBundle(AbstractContextManager["RecoveryRuntimeBundle"]):
    def __init__(self, config: local_runtime.FormalRuntimeConfig) -> None:
        self.config = config
        self.encoder: minilm_binding.OfflineMiniLMEncoder | None = None
        self.nli: MavenEreNLIWorkerPool | None = None
        self.hippo: local_runtime.OfficialHippoGateway | None = None
        self._stack: ExitStack | None = None

    def __enter__(self) -> "RecoveryRuntimeBundle":
        if os.path.lexists(self.config.hippo_stage_root):
            raise MavenEreRecoveryError("recovery HippoRAG stage is not pristine")
        stack = ExitStack()
        try:
            self.encoder = minilm_binding.OfflineMiniLMEncoder(
                asset_manifest_path=self.config.minilm_asset_manifest,
                model_root=self.config.minilm_model_root,
            )
            self.nli = stack.enter_context(
                MavenEreNLIWorkerPool(
                    self.config.nli_model_root,
                    project_root=self.config.project,
                    runtime_python=self.config.local_python,
                )
            )
            self.hippo = local_runtime.OfficialHippoGateway(self.config)
            self._stack = stack
            return self
        except BaseException:
            stack.close()
            raise

    def __exit__(self, *_exc: object) -> None:
        if self._stack is not None:
            self._stack.close()

    def prepare_block(
        self, block: str, views: Sequence[local_runtime.ItemView]
    ) -> local_runtime.PreparedBlock:
        if self.encoder is None or self.nli is None:
            raise MavenEreRecoveryError("recovery semantic runtime is not open")
        return local_runtime.prepare_block(
            block=block,
            views=views,
            encoder=self.encoder,
            nli_pool=self.nli,
        )


def _selection(
    value: object,
    *,
    sentence_count: int,
    field: str,
) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise MavenEreRecoveryError(f"{field} selection shape drifted")
    result: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw < sentence_count:
            raise MavenEreRecoveryError(f"{field} selection ordinal drifted")
        result.append(raw)
    if len(set(result)) != 3:
        raise MavenEreRecoveryError(f"{field} selection uniqueness drifted")
    return tuple(result)  # type: ignore[return-value]


def validate_reused_a_form_archive(
    path: str | Path,
    prepared: local_runtime.PreparedBlock,
    g8_model: core.G8Model,
    *,
    expected_file_sha256: str,
    expected_item_count: int = 48,
) -> dict[str, Any]:
    """Validate existing label-free outputs without reopening an efficacy label."""

    archive_path = Path(path).absolute()
    if _sha256_file(archive_path) != expected_file_sha256:
        raise MavenEreRecoveryError("reused A_form action archive file hash drifted")
    try:
        value = json.loads(archive_path.read_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreRecoveryError("reused A_form action archive is invalid") from exc
    exact_root = {
        "all_3n_tasks_submitted_before_first_result",
        "archive_sha256",
        "block",
        "hipporag_physical_cap",
        "item_count",
        "items",
        "local_physical_cap",
        "logical_task_count",
        "schema",
        "version",
    }
    if not isinstance(value, dict) or set(value) != exact_root:
        raise MavenEreRecoveryError("reused A_form action archive root drifted")
    body = dict(value)
    declared = body.pop("archive_sha256", None)
    if (
        not isinstance(declared, str)
        or not _SHA256.fullmatch(declared)
        or v1.stable_hash(body) != declared
        or value.get("schema") != "maven_ere_three_arm_action_archive_v1"
        or value.get("version") != "v1"
        or value.get("block") != "A_form"
        or value.get("item_count") != expected_item_count
        or value.get("logical_task_count") != 3 * expected_item_count
        or value.get("all_3n_tasks_submitted_before_first_result") is not True
        or value.get("local_physical_cap") != 16
        or value.get("hipporag_physical_cap") != 2
        or prepared.block != "A_form"
        or len(prepared.items) != expected_item_count
    ):
        raise MavenEreRecoveryError("reused A_form action archive header drifted")
    items = value.get("items")
    if not isinstance(items, list) or len(items) != expected_item_count:
        raise MavenEreRecoveryError("reused A_form action item count drifted")
    exact_item = {
        "agent",
        "hipporag_selected",
        "item_id",
        "raw_selected",
        "semantic_receipt_sha256",
    }
    exact_agent = {
        "e0_behavior_sha256",
        "e0_selected",
        "e1_behavior_sha256",
        "e1_selected",
        "edge_deletion_action_change_count",
        "edge_deletion_witness_count",
        "frontier_sha256",
    }
    for raw_row, expected in zip(items, prepared.items, strict=True):
        if not isinstance(raw_row, dict) or set(raw_row) != exact_item:
            raise MavenEreRecoveryError("reused A_form action item shape drifted")
        if (
            raw_row.get("item_id") != expected.view.item_id
            or raw_row.get("semantic_receipt_sha256")
            != expected.semantic_receipt_sha256
        ):
            raise MavenEreRecoveryError("reused A_form action item binding drifted")
        raw_selected = _selection(
            raw_row.get("raw_selected"),
            sentence_count=expected.item.sentence_count,
            field="RAW",
        )
        _selection(
            raw_row.get("hipporag_selected"),
            sentence_count=expected.item.sentence_count,
            field="HippoRAG",
        )
        agent = raw_row.get("agent")
        if not isinstance(agent, dict) or set(agent) != exact_agent:
            raise MavenEreRecoveryError("reused A_form agent row shape drifted")
        e0_selected = _selection(
            agent.get("e0_selected"),
            sentence_count=expected.item.sentence_count,
            field="E0",
        )
        if (
            agent.get("e1_selected") is not None
            or agent.get("e1_behavior_sha256") is not None
            or agent.get("edge_deletion_witness_count") != 0
            or agent.get("edge_deletion_action_change_count") != 0
            or core.raw3(expected.item) != raw_selected
        ):
            raise MavenEreRecoveryError("reused A_form fixed-arm contract drifted")
        space = core.build_action_space(expected.item)
        frontier = core.g8_frontier(expected.item, g8_model, space=space)
        frontier_sha = v1.stable_hash(
            [
                {
                    "energy_hex": row.generator_energy.hex(),
                    "ordinals": row.ordinals,
                }
                for row in frontier.entries
            ]
        )
        if (
            e0_selected != frontier.e0.ordinals
            or agent.get("frontier_sha256") != frontier_sha
            or agent.get("e0_behavior_sha256")
            != core.behavior_hash(expected.item, space, frontier, e0_selected)
        ):
            raise MavenEreRecoveryError("reused A_form E0 recomputation drifted")
    return _self_hashed(
        {
            "A_form_three_arm_actions_reexecuted": 0,
            "all_3n_tasks_submitted_before_first_result": True,
            "archive_file_sha256": expected_file_sha256,
            "archive_self_sha256": declared,
            "e0_frontier_and_behavior_recomputed_item_count": expected_item_count,
            "hipporag_selection_shape_validated_item_count": expected_item_count,
            "logical_task_count": 3 * expected_item_count,
            "raw3_recomputed_item_count": expected_item_count,
            "schema": "maven_ere_A_form_reused_action_validation_v2",
            "validated_item_count": expected_item_count,
            "version": VERSION,
        },
        "validation_sha256",
    )


def _require_exact_payload(
    payload: Mapping[str, Any],
    path: Path,
    expected_file_sha256: str,
    *,
    label: str,
) -> str:
    raw = v1._canonical_bytes(payload)
    if (
        _sha256_file(path) != expected_file_sha256
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
        or path.read_bytes() != raw
    ):
        raise MavenEreRecoveryError(f"{label} byte binding drifted")
    return expected_file_sha256


def _verify_v1_artifact_hashes(project: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for role, (relative, expected) in V1_ARTIFACT_SHA256.items():
        if _sha256_file(project / relative) != expected:
            raise MavenEreRecoveryError("v1 recovery artifact hash drifted")
        result[role] = expected
    failure = _validate_self_hashed_public_file(
        project / V1_CONTROLLER_RELATIVE / "lifecycle.failure.json",
        file_sha256=V1_ARTIFACT_SHA256["failure_receipt"][1],
        self_field="failure_sha256",
        self_sha256="ba59fac227bfa6a79d2103f92f432a930a3473b3f9d102c91ddf0ea27ab969f2",
        schema="maven_ere_g8_e1_lifecycle_failure_v1",
        status="terminal_no_retry",
    )
    if (
        failure.get("phase") != "A_form_label_free_actions"
        or failure.get("category") != "MavenEreFormalControllerError"
        or failure.get("message_sha256")
        != "c7d5383503fe3981b59403731f2fd751d04cfc6c9c05fd2f6a1b0c4b31a7ec7a"
    ):
        raise MavenEreRecoveryError("v1 failure boundary drifted")
    return result


def _prepared(
    runtime: RecoveryRuntimeBundle,
    acquisition_root: Path,
    block: str,
) -> local_runtime.PreparedBlock:
    view_path = acquisition_root / "private_packs" / f"{block}.view.json"
    views = local_runtime.load_view_pack(view_path, block=block)
    expected = {
        "G_form": 96,
        "A_form": 48,
        "F_search": 36,
        "A_hold": 30,
        "M_search": 30,
    }[block]
    if len(views) != expected:
        raise MavenEreRecoveryError("recovery block view count drifted")
    return runtime.prepare_block(block, views)


def _labels(
    acquisition_root: Path,
    prepared: local_runtime.PreparedBlock,
) -> Mapping[str, str]:
    return v1.load_family_labels(
        acquisition_root / "private_packs" / f"{prepared.block}.labels.json",
        block=prepared.block,
        expected_item_ids=tuple(row.view.item_id for row in prepared.items),
    )


def _write_action(path: Path, execution: v1.BlockExecution) -> str:
    return v1._durable_roundtrip(path, normalized_action_archive(execution))


def run_result_blind_recovery(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    if project.name != "reconstruction_v2":
        raise MavenEreRecoveryError("project root must be reconstruction_v2")
    provenance = validate_recovery_provenance(project)
    v1_hashes = _verify_v1_artifact_hashes(project)
    recovery_root = project / RECOVERY_ROOT_RELATIVE
    if os.path.lexists(recovery_root):
        raise OneShotRefusal("result-blind recovery root already exists")
    config = _recovery_runtime_config(project)
    preflight = preflight_recovery_runtime(config)
    recovery_root.mkdir(mode=0o700, parents=True)
    controller_root = project / RECOVERY_CONTROLLER_RELATIVE
    controller_root.mkdir(mode=0o700)
    marker = _self_hashed(
        {
            "A_form_three_arm_actions_reexecuted": False,
            "design_sha256": RECOVERY_DESIGN_SELF_SHA256,
            "provenance": provenance,
            "schema": "maven_ere_result_blind_recovery_authorization_consumed_v2",
            "status": "consumed_before_private_pack_or_recovery_model_inference",
            "v1_artifact_hashes": v1_hashes,
            "version": VERSION,
        },
        "marker_sha256",
    )
    v1._exclusive_write(
        controller_root / "recovery.authorization.consumed.json", marker
    )
    v1._durable_roundtrip(controller_root / "runtime.preflight.json", preflight)
    acquisition_root = project / V1_ACQUISITION_RELATIVE
    phase = "runtime_open"
    try:
        with RecoveryRuntimeBundle(config) as runtime:
            assert runtime.hippo is not None
            hippo = runtime.hippo

            phase = "G_form_exact_training_reconstruction"
            g_prepared = _prepared(runtime, acquisition_root, "G_form")
            g_semantic = v1._semantic_archive(g_prepared)
            _require_exact_payload(
                g_semantic,
                project / V1_CONTROLLER_RELATIVE / "G_form.semantic.archive.json",
                V1_ARTIFACT_SHA256["G_form_semantic_archive"][1],
                label="G_form semantic archive",
            )
            g_labels = _labels(acquisition_root, g_prepared)
            g8_model = v1._fit_g8(g_prepared, g_labels)
            g8_payload = _self_hashed(core.g8_model_payload(g8_model), "model_sha256")
            _require_exact_payload(
                g8_payload,
                project / V1_CONTROLLER_RELATIVE / "G8.model.json",
                V1_ARTIFACT_SHA256["G8_model"][1],
                label="G8 model",
            )
            g_receipt = _self_hashed(
                {
                    "G8_fit_sha256": g8_model.fit_sha256,
                    "G8_model_file_sha256": V1_ARTIFACT_SHA256["G8_model"][1],
                    "G_form_semantic_file_sha256": V1_ARTIFACT_SHA256[
                        "G_form_semantic_archive"
                    ][1],
                    "item_count": 96,
                    "schema": "maven_ere_G_form_exact_reconstruction_receipt_v2",
                    "version": VERSION,
                },
                "receipt_sha256",
            )
            v1._durable_roundtrip(
                controller_root / "G_form.exact_reconstruction.receipt.json", g_receipt
            )

            phase = "A_form_semantic_and_existing_action_validation"
            a_prepared = _prepared(runtime, acquisition_root, "A_form")
            a_semantic = v1._semantic_archive(a_prepared)
            _require_exact_payload(
                a_semantic,
                project / V1_CONTROLLER_RELATIVE / "A_form.semantic.archive.json",
                V1_ARTIFACT_SHA256["A_form_semantic_archive"][1],
                label="A_form semantic archive",
            )
            action_validation = validate_reused_a_form_archive(
                project / V1_CONTROLLER_RELATIVE / "A_form.action.archive.json",
                a_prepared,
                g8_model,
                expected_file_sha256=V1_ARTIFACT_SHA256["A_form_action_archive"][1],
            )
            action_validation_file_sha = v1._durable_roundtrip(
                controller_root / "A_form.reused_action.validation.json",
                action_validation,
            )
            phase = "A_form_label_open_and_E1_fit"
            a_labels = _labels(acquisition_root, a_prepared)
            e1_model = v1._fit_e1(a_prepared, a_labels, g8_model)
            e1_payload = _self_hashed(core.e1_model_payload(e1_model), "model_sha256")
            e1_file_sha = v1._durable_roundtrip(
                controller_root / "E1.model.json", e1_payload
            )

            phase = "F_search_label_free_actions"
            hippo.prepare_blocks(("F_search",))
            f_prepared = _prepared(runtime, acquisition_root, "F_search")
            f_semantic_file_sha = v1._durable_roundtrip(
                controller_root / "F_search.semantic.archive.json",
                v1._semantic_archive(f_prepared),
            )
            f_execution = v1.execute_block(
                prepared=f_prepared,
                g8_model=g8_model,
                e1_model=e1_model,
                hippo=hippo,
                causal_audit=False,
            )
            f_action_file_sha = _write_action(
                controller_root / "F_search.action.archive.json", f_execution
            )
            f_distinct = sum(
                row.agent.e1_selected != row.agent.e0_selected for row in f_execution.items
            )
            hold_freeze = _self_hashed(
                {
                    "A_form_reused_action_validation_file_sha256": action_validation_file_sha,
                    "E1_file_sha256": e1_file_sha,
                    "E1_fit_sha256": e1_model.fit_sha256,
                    "F_search_E1_vs_E0_behavior_distinct_item_count": f_distinct,
                    "F_search_action_file_sha256": f_action_file_sha,
                    "G8_fit_sha256": g8_model.fit_sha256,
                    "design_sha256": RECOVERY_DESIGN_SELF_SHA256,
                    "promotion_alpha": {"denominator": 10, "numerator": 1},
                    "schema": "maven_ere_recovery_A_hold_pre_run_freeze_v2",
                    "status": "frozen_before_A_hold_view_open",
                    "version": VERSION,
                },
                "freeze_sha256",
            )
            hold_freeze_file_sha = v1._durable_roundtrip(
                controller_root / "A_hold.pre_run.freeze.json", hold_freeze
            )

            phase = "A_hold_label_free_actions"
            hippo.prepare_blocks(("A_hold",))
            hold_prepared = _prepared(runtime, acquisition_root, "A_hold")
            hold_semantic_file_sha = v1._durable_roundtrip(
                controller_root / "A_hold.semantic.archive.json",
                v1._semantic_archive(hold_prepared),
            )
            hold_execution = v1.execute_block(
                prepared=hold_prepared,
                g8_model=g8_model,
                e1_model=e1_model,
                hippo=hippo,
                causal_audit=True,
            )
            hold_action_file_sha = _write_action(
                controller_root / "A_hold.action.archive.json", hold_execution
            )
            phase = "A_hold_label_open_and_score"
            hold_labels = _labels(acquisition_root, hold_prepared)
            hold_score = v1.score_block(hold_execution, hold_labels)
            promotion = v1.evaluator_promoted(hold_score)
            primary = v1.real_domain_primary_passed(hold_score)
            hold_report = _self_hashed(
                {
                    "A_hold_action_file_sha256": hold_action_file_sha,
                    "A_hold_pre_run_freeze_file_sha256": hold_freeze_file_sha,
                    "A_hold_semantic_file_sha256": hold_semantic_file_sha,
                    "evaluator_promoted": promotion,
                    "real_domain_primary_passed": primary,
                    "schema": "maven_ere_recovery_A_hold_aggregate_report_v2",
                    "score": hold_score,
                    "version": VERSION,
                },
                "report_sha256",
            )
            hold_report_file_sha = v1._durable_roundtrip(
                controller_root / "A_hold.aggregate.report.json", hold_report
            )

            m_score: Mapping[str, Any] | None = None
            m_report_file_sha: str | None = None
            if promotion:
                phase = "M_search_pre_run_freeze"
                m_freeze = _self_hashed(
                    {
                        "A_hold_report_file_sha256": hold_report_file_sha,
                        "E1_fit_sha256": e1_model.fit_sha256,
                        "G8_fit_sha256": g8_model.fit_sha256,
                        "authorization": "evaluator_promoted_true",
                        "schema": "maven_ere_recovery_M_search_pre_run_freeze_v2",
                        "status": "frozen_before_M_search_view_open",
                        "version": VERSION,
                    },
                    "freeze_sha256",
                )
                m_freeze_file_sha = v1._durable_roundtrip(
                    controller_root / "M_search.pre_run.freeze.json", m_freeze
                )
                phase = "M_search_label_free_actions"
                hippo.prepare_blocks(("M_search",))
                m_prepared = _prepared(runtime, acquisition_root, "M_search")
                m_semantic_file_sha = v1._durable_roundtrip(
                    controller_root / "M_search.semantic.archive.json",
                    v1._semantic_archive(m_prepared),
                )
                m_execution = v1.execute_block(
                    prepared=m_prepared,
                    g8_model=g8_model,
                    e1_model=e1_model,
                    hippo=hippo,
                    causal_audit=True,
                )
                m_action_file_sha = _write_action(
                    controller_root / "M_search.action.archive.json", m_execution
                )
                phase = "M_search_label_open_and_score"
                m_labels = _labels(acquisition_root, m_prepared)
                m_score = v1.score_block(m_execution, m_labels)
                m_report = _self_hashed(
                    {
                        "M_L5_passed": v1.evaluator_promoted(m_score),
                        "M_search_action_file_sha256": m_action_file_sha,
                        "M_search_pre_run_freeze_file_sha256": m_freeze_file_sha,
                        "M_search_semantic_file_sha256": m_semantic_file_sha,
                        "schema": "maven_ere_recovery_M_search_aggregate_report_v2",
                        "score": m_score,
                        "version": VERSION,
                    },
                    "report_sha256",
                )
                m_report_file_sha = v1._durable_roundtrip(
                    controller_root / "M_search.aggregate.report.json", m_report
                )

            phase = "terminal_result"
            terminal = _self_hashed(
                {
                    "A_hold": hold_report,
                    "M_L5_passed": (
                        v1.evaluator_promoted(m_score) if m_score is not None else None
                    ),
                    "M_search": m_score,
                    "M_search_opened": promotion,
                    "artifact_bindings": {
                        "A_form_reused_action_validation_file_sha256": action_validation_file_sha,
                        "A_hold_report_file_sha256": hold_report_file_sha,
                        "F_search_action_file_sha256": f_action_file_sha,
                        "F_search_semantic_file_sha256": f_semantic_file_sha,
                        "M_search_report_file_sha256": m_report_file_sha,
                        "v1_A_form_action_archive_file_sha256": V1_ARTIFACT_SHA256[
                            "A_form_action_archive"
                        ][1],
                    },
                    "claim_boundary": {
                        "A_form_three_arm_actions_reexecuted": False,
                        "hidden_TEST_opened": False,
                        "new_selection_secret_or_cohort_generated": False,
                        "online_or_external_evaluator_calls": 0,
                        "released_train_or_valid_source_rows_reopened": False,
                        "same_v1_controller_root_replayed": False,
                    },
                    "evaluator_promoted": promotion,
                    "provenance": provenance,
                    "real_domain_primary_passed": primary,
                    "schema": "maven_ere_g8_e1_result_blind_recovery_terminal_v2",
                    "status": (
                        "valid_recovery_promotion_M_measured"
                        if promotion
                        else "valid_recovery_no_promotion_M_unopened"
                    ),
                    "version": VERSION,
                },
                "terminal_result_sha256",
            )
            v1._durable_roundtrip(
                controller_root / "recovery.terminal_result.json", terminal
            )
            return terminal
    except BaseException as exc:
        failure = _self_hashed(
            {
                "category": type(exc).__name__,
                "message_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                "phase": phase,
                "schema": "maven_ere_g8_e1_result_blind_recovery_failure_v2",
                "status": "terminal_no_retry",
                "version": VERSION,
            },
            "failure_sha256",
        )
        try:
            v1._atomic_write(controller_root / "recovery.failure.json", failure)
        except BaseException:
            pass
        raise


__all__ = [
    "MavenEreRecoveryError",
    "OneShotRefusal",
    "RECOVERY_ROOT_RELATIVE",
    "normalized_action_archive",
    "preflight_recovery_runtime",
    "run_result_blind_recovery",
    "validate_recovery_provenance",
    "validate_reused_a_form_archive",
]
