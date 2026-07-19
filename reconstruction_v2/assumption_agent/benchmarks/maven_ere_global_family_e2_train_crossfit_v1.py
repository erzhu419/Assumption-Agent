"""One-shot TRAIN-only cross-fit for the MAVEN-ERE global-family E2 selector."""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from assumption_agent.benchmarks import maven_ere_g8_e1_acquisition_v1 as acquisition
from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core
from assumption_agent.benchmarks import maven_ere_g8_e1_formal_controller_v1 as v1
from assumption_agent.benchmarks import maven_ere_g8_e1_result_blind_recovery_controller_v2 as recovery
from assumption_agent.benchmarks import maven_ere_local_runtime_v1 as local_runtime
from assumption_agent.benchmarks.maven_ere_nli_runtime_v1 import (
    MavenEreNLIWorkerPool,
    verify_maven_design,
)
from replication_runtime.qasc_nli_v1 import binding as nli_binding
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding


VERSION = "v1"
SCHEMA = "maven_ere_global_family_e2_train_crossfit_v1"
DESIGN_RELATIVE = Path(
    "manifests/maven_ere_global_family_e2_train_crossfit_design_v1.json"
)
DESIGN_FILE_SHA256 = "2d0cf7c1d4316de5e9b3ca399026d21280d9af366bc620b872954773ba88f135"
DESIGN_SELF_SHA256 = "2a3ae6817a5ea83dc25bdad1853f781922d99197adc651582e05778119f875fb"
RECOVERY_DISPOSITION_RELATIVE = Path(
    "manifests/maven_ere_g8_e1_result_blind_recovery_result_disposition_v2.json"
)
RECOVERY_DISPOSITION_FILE_SHA256 = (
    "16887616c36ef07f2d739f102067335990c5b232d09e9cd2f2c0051f760f7e3d"
)
RECOVERY_DISPOSITION_SELF_SHA256 = (
    "1903cefd75625a405cd68208b536bba41845d9c860bb26264d4e0431ed70d655"
)
FREEZE_RELATIVE = Path(
    "manifests/maven_ere_global_family_e2_train_crossfit_implementation_freeze_v1.json"
)
FREEZE_SCHEMA = (
    "maven_ere_global_family_e2_train_crossfit_implementation_freeze_v1"
)
ROOT_RELATIVE = Path("artifacts/maven_ere_global_family_e2_train_crossfit_v1")
ACQUISITION_RELATIVE = Path("artifacts/maven_ere_g8_e1_formal_v1/acquisition")
V1_CONTROLLER_RELATIVE = Path("artifacts/maven_ere_g8_e1_formal_v1/controller")
G8_MODEL_FILE_SHA256 = "7fc8bee5e794e774a4b82ddf44b477279b1e0d6a78137a570834738db27a0db4"
FOLD_COUNT = 4
RIDGE_LAMBDA = 1.0
FAMILY_STAT_ORDER = (
    "all_sentence_max",
    "all_sentence_top3_mean",
    "authorized_sentence_max",
    "head_mention_sentence_max",
    "tail_mention_sentence_max",
    "singleton_argmax_vote_fraction",
)
FEATURE_ORDER = tuple(
    f"{family}_{stat}" for family in core.FAMILY_ORDER for stat in FAMILY_STAT_ORDER
)
IMPLEMENTATION_ROLE_PATHS: Mapping[str, Path] = {
    "design": DESIGN_RELATIVE,
    "recovery_result_disposition": RECOVERY_DISPOSITION_RELATIVE,
    "implementation": Path(
        "assumption_agent/benchmarks/maven_ere_global_family_e2_train_crossfit_v1.py"
    ),
    "test": Path("tests/test_maven_ere_global_family_e2_train_crossfit_v1.py"),
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MavenEreE2Error(RuntimeError):
    """The frozen TRAIN-only E2 diagnostic contract drifted."""


class OneShotRefusal(MavenEreE2Error):
    """The diagnostic root is not pristine."""


@dataclass(frozen=True)
class E2Model:
    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]
    class_means: tuple[float, float, float]
    weights: tuple[tuple[float, float, float], ...]
    normal_equation_sha256: str
    coefficient_sha256: str
    training_examples_sha256: str
    fit_sha256: str
    item_count: int

    def __post_init__(self) -> None:
        dimension = len(FEATURE_ORDER)
        if (
            len(self.feature_means) != dimension
            or len(self.feature_stds) != dimension
            or len(self.weights) != dimension
            or any(len(row) != len(core.FAMILY_ORDER) for row in self.weights)
            or len(self.class_means) != len(core.FAMILY_ORDER)
            or self.item_count <= 0
        ):
            raise MavenEreE2Error("E2 model shape drifted")
        values = (
            *self.feature_means,
            *self.feature_stds,
            *self.class_means,
            *(value for row in self.weights for value in row),
        )
        if not all(math.isfinite(value) for value in values):
            raise MavenEreE2Error("E2 model contains nonfinite values")
        if any(value < 0 for value in self.feature_stds):
            raise MavenEreE2Error("E2 feature scales are negative")
        for digest in (
            self.normal_equation_sha256,
            self.coefficient_sha256,
            self.training_examples_sha256,
            self.fit_sha256,
        ):
            if not _SHA256.fullmatch(digest):
                raise MavenEreE2Error("E2 model digest drifted")


@dataclass(frozen=True)
class E2Selection:
    target_family: str
    selected: tuple[int, int, int]
    used_fallback: bool
    target_margin_q6: int
    generator_energy: float


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise MavenEreE2Error("bound file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
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
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MavenEreE2Error("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = stable_hash(result)
    return result


def _finite_vector(values: Sequence[object], *, expected: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != expected:
        raise MavenEreE2Error("feature vector shape drifted")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise MavenEreE2Error("feature value is not numeric")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise MavenEreE2Error("feature value is nonfinite")
        result.append(parsed)
    return tuple(result)


def item_family_features(
    item: core.ValidatedActionItem,
    *,
    space: core.TypedActionSpace | None = None,
) -> tuple[float, ...]:
    if not isinstance(item, core.ValidatedActionItem):
        raise MavenEreE2Error("validated action item required")
    action_space = space if space is not None else core.build_action_space(item)
    if action_space.item is not item:
        raise MavenEreE2Error("action space/item identity drifted")
    event_sentences = tuple(
        frozenset(mention.sentence_ordinal for mention in event.mentions)
        for event in item.events
    )
    head = event_sentences[item.head_event]
    tail = event_sentences[item.tail_event]
    if not head or not tail or item.sentence_count < 3:
        raise MavenEreE2Error("endpoint or sentence capacity drifted")
    singleton_votes = [
        max(
            range(len(core.FAMILY_ORDER)),
            key=lambda index: (row[index], -index),
        )
        for row in item.sentence_family_nli_scores
    ]
    result: list[float] = []
    for family_index, _family in enumerate(core.FAMILY_ORDER):
        all_values = [
            row[family_index] / core.Q6_SCALE
            for row in item.sentence_family_nli_scores
        ]
        top3 = sorted(all_values, reverse=True)[:3]
        authorized = [
            item.sentence_family_nli_scores[ordinal][family_index] / core.Q6_SCALE
            for ordinal in action_space.authorized_ordinals
        ]
        head_values = [
            item.sentence_family_nli_scores[ordinal][family_index] / core.Q6_SCALE
            for ordinal in head
        ]
        tail_values = [
            item.sentence_family_nli_scores[ordinal][family_index] / core.Q6_SCALE
            for ordinal in tail
        ]
        result.extend(
            (
                max(all_values),
                math.fsum(top3) / 3.0,
                max(authorized),
                max(head_values),
                max(tail_values),
                sum(value == family_index for value in singleton_votes)
                / item.sentence_count,
            )
        )
    parsed = _finite_vector(result, expected=len(FEATURE_ORDER))
    return parsed


def _float64_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


def fit_e2(examples: Sequence[core.LabelledItem]) -> E2Model:
    if not examples:
        raise MavenEreE2Error("E2 training examples are empty")
    ordered = sorted(
        examples,
        key=lambda row: (
            core.FAMILY_ORDER.index(row.family),
            core.action_item_commitment(row.item),
        ),
    )
    counts = {
        family: sum(row.family == family for row in ordered)
        for family in core.FAMILY_ORDER
    }
    if len(set(counts.values())) != 1 or min(counts.values()) < 4:
        raise MavenEreE2Error("E2 family-balanced training contract drifted")
    feature_rows: list[tuple[float, ...]] = []
    target_rows: list[tuple[float, float, float]] = []
    training_receipts: list[dict[str, str]] = []
    for row in ordered:
        space = core.build_action_space(row.item)
        feature_rows.append(item_family_features(row.item, space=space))
        target_rows.append(
            tuple(float(family == row.family) for family in core.FAMILY_ORDER)  # type: ignore[arg-type]
        )
        training_receipts.append(
            {
                "family": row.family,
                "item_sha256": core.action_item_commitment(row.item),
            }
        )
    x = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(target_rows, dtype=np.float64)
    means = x.mean(axis=0, dtype=np.float64)
    centered = x - means
    stds = np.sqrt((centered * centered).mean(axis=0, dtype=np.float64))
    standardized = np.zeros_like(centered)
    nonzero = stds > 0
    standardized[:, nonzero] = centered[:, nonzero] / stds[nonzero]
    class_means = y.mean(axis=0, dtype=np.float64)
    target = y - class_means
    weight = 1.0 / len(ordered)
    matrix = np.eye(len(FEATURE_ORDER), dtype=np.float64) * RIDGE_LAMBDA
    matrix += weight * standardized.T @ standardized
    right = weight * standardized.T @ target
    try:
        coefficients = np.linalg.solve(matrix, right)
    except np.linalg.LinAlgError as exc:
        raise MavenEreE2Error("E2 normal equation solve failed") from exc
    if not np.isfinite(coefficients).all():
        raise MavenEreE2Error("E2 coefficients are nonfinite")
    normal_hash = stable_hash(
        {
            "matrix_sha256": _float64_sha(matrix),
            "right_sha256": _float64_sha(right),
        }
    )
    coefficient_hash = _float64_sha(coefficients)
    training_hash = stable_hash(training_receipts)
    fit_hash = stable_hash(
        {
            "class_means_hex": [value.hex() for value in class_means],
            "coefficient_sha256": coefficient_hash,
            "feature_means_hex": [value.hex() for value in means],
            "feature_order": FEATURE_ORDER,
            "feature_stds_hex": [value.hex() for value in stds],
            "lambda": RIDGE_LAMBDA,
            "normal_equation_sha256": normal_hash,
            "training_examples_sha256": training_hash,
        }
    )
    return E2Model(
        feature_means=tuple(float(value) for value in means),
        feature_stds=tuple(float(value) for value in stds),
        class_means=tuple(float(value) for value in class_means),  # type: ignore[arg-type]
        weights=tuple(
            tuple(float(value) for value in row)  # type: ignore[misc]
            for row in coefficients
        ),
        normal_equation_sha256=normal_hash,
        coefficient_sha256=coefficient_hash,
        training_examples_sha256=training_hash,
        fit_sha256=fit_hash,
        item_count=len(ordered),
    )


def e2_logits(model: E2Model, features: Sequence[object]) -> tuple[float, float, float]:
    parsed = _finite_vector(features, expected=len(FEATURE_ORDER))
    standardized = np.zeros(len(FEATURE_ORDER), dtype=np.float64)
    for index, value in enumerate(parsed):
        if model.feature_stds[index] > 0:
            standardized[index] = (
                value - model.feature_means[index]
            ) / model.feature_stds[index]
    weights = np.asarray(model.weights, dtype=np.float64)
    logits = np.asarray(model.class_means, dtype=np.float64) + standardized @ weights
    if not np.isfinite(logits).all():
        raise MavenEreE2Error("E2 logits are nonfinite")
    return tuple(float(value) for value in logits)  # type: ignore[return-value]


def predict_item_family(
    model: E2Model,
    item: core.ValidatedActionItem,
    *,
    space: core.TypedActionSpace | None = None,
) -> str:
    logits = e2_logits(model, item_family_features(item, space=space))
    index = max(range(len(logits)), key=lambda value: (logits[value], -value))
    return core.FAMILY_ORDER[index]


def e2_select(
    item: core.ValidatedActionItem,
    g8_model: core.G8Model,
    e2_model: E2Model,
    *,
    space: core.TypedActionSpace | None = None,
    frontier: core.G8Frontier | None = None,
) -> E2Selection:
    action_space = space if space is not None else core.build_action_space(item)
    g8_frontier = (
        frontier
        if frontier is not None
        else core.g8_frontier(item, g8_model, space=action_space)
    )
    if action_space.item is not item:
        raise MavenEreE2Error("E2 action space/item identity drifted")
    target_family = predict_item_family(e2_model, item, space=action_space)
    target_index = core.FAMILY_ORDER.index(target_family)
    candidates: list[tuple[int, float, tuple[int, int, int]]] = []
    for selected in core.iter_authorized_set3(action_space):
        scores = core.selected_set_family_scores(item, selected)
        predicted_index = max(
            range(len(scores)), key=lambda index: (scores[index], -index)
        )
        if predicted_index != target_index:
            continue
        other = max(
            score for index, score in enumerate(scores) if index != target_index
        )
        margin = scores[target_index] - other
        energy = core.g8_energy(g8_model, core.phi_features(action_space, selected))
        candidates.append((margin, energy, selected))
    if not candidates:
        selected = g8_frontier.e0.ordinals
        scores = core.selected_set_family_scores(item, selected)
        margin = scores[target_index] - max(
            score for index, score in enumerate(scores) if index != target_index
        )
        return E2Selection(
            target_family,
            selected,
            True,
            margin,
            g8_frontier.e0.generator_energy,
        )
    margin, energy, selected = min(
        candidates,
        key=lambda row: (-row[0], -row[1], row[2]),
    )
    return E2Selection(target_family, selected, False, margin, energy)


def model_payload(model: E2Model) -> dict[str, Any]:
    return {
        "class_means_hex": [value.hex() for value in model.class_means],
        "coefficient_sha256": model.coefficient_sha256,
        "feature_means_hex": [value.hex() for value in model.feature_means],
        "feature_order": list(FEATURE_ORDER),
        "feature_stds_hex": [value.hex() for value in model.feature_stds],
        "fit_sha256": model.fit_sha256,
        "item_count": model.item_count,
        "lambda": RIDGE_LAMBDA,
        "normal_equation_sha256": model.normal_equation_sha256,
        "schema": "maven_ere_global_family_E2_model_v1",
        "training_examples_sha256": model.training_examples_sha256,
        "version": VERSION,
        "weights_hex": [[value.hex() for value in row] for row in model.weights],
    }


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise MavenEreE2Error("git provenance command failed")
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
        raise MavenEreE2Error("public E2 binding file hash drifted")
    try:
        value = json.loads(path.read_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreE2Error("public E2 binding is invalid") from exc
    if not isinstance(value, dict):
        raise MavenEreE2Error("public E2 binding root drifted")
    body = dict(value)
    declared = body.pop(self_field, None)
    if (
        declared != self_sha256
        or stable_hash(body) != self_sha256
        or value.get("schema") != schema
        or value.get("status") != status
    ):
        raise MavenEreE2Error("public E2 binding self hash drifted")


def validate_formal_provenance(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    base = recovery.validate_recovery_provenance(project)
    repository = Path(
        _git_bytes(project, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve(strict=True)
    if (repository / "reconstruction_v2").resolve(strict=True) != project:
        raise MavenEreE2Error("project/repository relationship drifted")
    _validate_public_binding(
        project / DESIGN_RELATIVE,
        file_sha256=DESIGN_FILE_SHA256,
        self_field="design_sha256",
        self_sha256=DESIGN_SELF_SHA256,
        schema="maven_ere_global_family_e2_train_crossfit_design_v1",
        status="frozen_before_new_training_pack_reopen_or_E2_model_inference",
    )
    _validate_public_binding(
        project / RECOVERY_DISPOSITION_RELATIVE,
        file_sha256=RECOVERY_DISPOSITION_FILE_SHA256,
        self_field="disposition_sha256",
        self_sha256=RECOVERY_DISPOSITION_SELF_SHA256,
        schema="maven_ere_g8_e1_result_blind_recovery_result_disposition_v2",
        status="valid_recovery_no_promotion_M_unopened_primary_failed",
    )
    freeze_path = project / FREEZE_RELATIVE
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise MavenEreE2Error("E2 implementation freeze is unavailable")
    raw = freeze_path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreE2Error("E2 implementation freeze is invalid") from exc
    if not isinstance(payload, dict):
        raise MavenEreE2Error("E2 implementation freeze root drifted")
    body = dict(payload)
    declared = body.pop("implementation_freeze_sha256", None)
    if (
        not isinstance(declared, str)
        or not _SHA256.fullmatch(declared)
        or stable_hash(body) != declared
        or payload.get("schema") != FREEZE_SCHEMA
        or payload.get("version") != VERSION
        or payload.get("status")
        != "frozen_before_new_training_pack_reopen_or_E2_model_inference"
    ):
        raise MavenEreE2Error("E2 implementation freeze self hash drifted")
    implementation_commit = payload.get("implementation_commit")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise MavenEreE2Error("E2 implementation commit drifted")
    binding = payload.get("implementation_binding")
    files = binding.get("files") if isinstance(binding, Mapping) else None
    if (
        not isinstance(files, list)
        or binding.get("file_count") != len(IMPLEMENTATION_ROLE_PATHS)
        or len(files) != len(IMPLEMENTATION_ROLE_PATHS)
    ):
        raise MavenEreE2Error("E2 implementation registry drifted")
    expected: list[tuple[Path, str]] = []
    for row, (role, relative) in zip(
        files, IMPLEMENTATION_ROLE_PATHS.items(), strict=True
    ):
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "role",
            "sha256",
        }:
            raise MavenEreE2Error("E2 implementation row shape drifted")
        digest = row.get("sha256")
        if (
            row.get("role") != role
            or row.get("relative_path") != relative.as_posix()
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or _sha256_file(project / relative) != digest
        ):
            raise MavenEreE2Error("E2 implementation row drifted")
        expected.append((relative, digest))
    if payload.get("claim_boundary") != {
        "A_hold_F_search_or_M_search_private_pack_opened": False,
        "formal_claim_or_promotion_run": False,
        "hidden_TEST_opened": False,
        "online_or_external_evaluation_used": False,
        "released_train_or_valid_source_rows_reopened": False,
    }:
        raise MavenEreE2Error("E2 implementation claim boundary drifted")
    if _git_bytes(repository, "cat-file", "-t", implementation_commit).strip() != b"commit":
        raise MavenEreE2Error("E2 implementation commit is unavailable")
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
            raise MavenEreE2Error("E2 committed blob drifted")
    return {
        "base_recovery_provenance": base,
        "implementation_commit": implementation_commit,
        "implementation_freeze_file_sha256": hashlib.sha256(raw).hexdigest(),
        "implementation_freeze_self_sha256": declared,
    }


def preflight_semantic_runtime(project: Path) -> dict[str, Any]:
    config = local_runtime.default_formal_runtime_config(project)
    try:
        minilm = minilm_binding.verify_runtime_binding(
            asset_manifest_path=config.minilm_asset_manifest,
            model_root=config.minilm_model_root,
        )
        design = verify_maven_design(project)
        nli = nli_binding.verify_runtime_binding(
            asset_manifest_path=config.nli_asset_manifest,
            model_root=config.nli_model_root,
        )
    except Exception as exc:
        raise MavenEreE2Error("E2 offline semantic runtime preflight failed") from exc
    return {
        "external_network_calls": 0,
        "minilm_runtime_binding": dict(minilm),
        "model_inference_calls": 0,
        "nli_design_binding": dict(design),
        "nli_runtime_binding": dict(nli),
        "private_pack_reads": 0,
        "released_source_row_reads": 0,
        "schema": "maven_ere_global_family_e2_semantic_preflight_v1",
        "version": VERSION,
    }


class SemanticRuntime(AbstractContextManager["SemanticRuntime"]):
    def __init__(self, project: Path) -> None:
        self.project = project
        self.encoder: minilm_binding.OfflineMiniLMEncoder | None = None
        self.nli: MavenEreNLIWorkerPool | None = None
        self._stack: ExitStack | None = None

    def __enter__(self) -> "SemanticRuntime":
        config = local_runtime.default_formal_runtime_config(self.project)
        stack = ExitStack()
        try:
            self.encoder = minilm_binding.OfflineMiniLMEncoder(
                asset_manifest_path=config.minilm_asset_manifest,
                model_root=config.minilm_model_root,
            )
            self.nli = stack.enter_context(
                MavenEreNLIWorkerPool(
                    config.nli_model_root,
                    project_root=self.project,
                    runtime_python=config.local_python,
                )
            )
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
            raise MavenEreE2Error("E2 semantic runtime is not open")
        return local_runtime.prepare_block(
            block=block,
            views=views,
            encoder=self.encoder,
            nli_pool=self.nli,
        )


def _prepare_training_block(
    runtime: SemanticRuntime,
    acquisition_root: Path,
    block: str,
) -> local_runtime.PreparedBlock:
    if block not in {"G_form", "A_form"}:
        raise MavenEreE2Error("non-training private block access forbidden")
    views = local_runtime.load_view_pack(
        acquisition_root / "private_packs" / f"{block}.view.json",
        block=block,
    )
    expected = {"G_form": 96, "A_form": 48}[block]
    if len(views) != expected:
        raise MavenEreE2Error("training block count drifted")
    return runtime.prepare_block(block, views)


def _training_labels(
    acquisition_root: Path,
    prepared: local_runtime.PreparedBlock,
) -> Mapping[str, str]:
    if prepared.block not in {"G_form", "A_form"}:
        raise MavenEreE2Error("non-training label pack access forbidden")
    return v1.load_family_labels(
        acquisition_root / "private_packs" / f"{prepared.block}.labels.json",
        block=prepared.block,
        expected_item_ids=tuple(row.view.item_id for row in prepared.items),
    )


def _fold_map(
    prepared: local_runtime.PreparedBlock,
    labels: Mapping[str, str],
) -> Mapping[str, int]:
    if prepared.block != "A_form" or len(prepared.items) != 48:
        raise MavenEreE2Error("A_form fold source drifted")
    result: dict[str, int] = {}
    for family in core.FAMILY_ORDER:
        rows = sorted(
            row.view.item_id
            for row in prepared.items
            if labels[row.view.item_id] == family
        )
        if len(rows) != 16:
            raise MavenEreE2Error("A_form family fold count drifted")
        for rank, item_id in enumerate(rows):
            result[item_id] = rank % FOLD_COUNT
    if len(result) != 48:
        raise MavenEreE2Error("A_form fold map drifted")
    return result


def _comparison(
    left: Sequence[int], right: Sequence[int], families: Sequence[str]
) -> dict[str, Any]:
    return v1._comparison(left, right, families)


def _feasible(report: Mapping[str, Any]) -> bool:
    comparison = report["comparisons"]["E2_minus_E0"]
    p_value = comparison["p_value"]
    return bool(
        comparison["net_utility"] > 0
        and int(p_value["numerator"]) * 5 <= int(p_value["denominator"])
        and all(value >= 0 for value in comparison["family_net"].values())
        and report["behavior_distinct_E2_vs_E0_count"] >= 3
    )


def run_train_crossfit(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    if project.name != "reconstruction_v2":
        raise MavenEreE2Error("project root must be reconstruction_v2")
    provenance = validate_formal_provenance(project)
    root = project / ROOT_RELATIVE
    if os.path.lexists(root):
        raise OneShotRefusal("E2 TRAIN cross-fit root already exists")
    preflight = preflight_semantic_runtime(project)
    root.mkdir(mode=0o700, parents=True)
    marker = _self_hashed(
        {
            "design_sha256": DESIGN_SELF_SHA256,
            "provenance": provenance,
            "schema": "maven_ere_global_family_e2_train_crossfit_authorization_v1",
            "status": "consumed_before_G_or_A_training_pack_open",
            "version": VERSION,
        },
        "marker_sha256",
    )
    v1._exclusive_write(root / "authorization.consumed.json", marker)
    v1._durable_roundtrip(root / "runtime.preflight.json", preflight)
    acquisition_root = project / ACQUISITION_RELATIVE
    phase = "semantic_runtime_open"
    try:
        with SemanticRuntime(project) as runtime:
            phase = "G_form_training_reconstruction"
            g_prepared = _prepare_training_block(runtime, acquisition_root, "G_form")
            g_labels = _training_labels(acquisition_root, g_prepared)
            g8 = v1._fit_g8(g_prepared, g_labels)
            g8_payload = _self_hashed(core.g8_model_payload(g8), "model_sha256")
            recovery._require_exact_payload(
                g8_payload,
                project / V1_CONTROLLER_RELATIVE / "G8.model.json",
                G8_MODEL_FILE_SHA256,
                label="E2 incumbent G8 model",
            )
            phase = "A_form_training_reconstruction"
            a_prepared = _prepare_training_block(runtime, acquisition_root, "A_form")
            a_labels = _training_labels(acquisition_root, a_prepared)

            g_examples = tuple(
                core.labelled_item(row.item, g_labels[row.view.item_id])
                for row in g_prepared.items
            )
            folds = _fold_map(a_prepared, a_labels)
            utilities: dict[str, list[int]] = {
                arm: [] for arm in ("RAW", "E0", "E2")
            }
            families: list[str] = []
            distinct = 0
            fallback_count = 0
            fold_receipts: list[dict[str, Any]] = []
            phase = "four_fold_crossfit"
            for fold in range(FOLD_COUNT):
                training_a = tuple(
                    core.labelled_item(row.item, a_labels[row.view.item_id])
                    for row in a_prepared.items
                    if folds[row.view.item_id] != fold
                )
                test_rows = tuple(
                    row
                    for row in a_prepared.items
                    if folds[row.view.item_id] == fold
                )
                if len(training_a) != 36 or len(test_rows) != 12:
                    raise MavenEreE2Error("E2 fold cardinality drifted")
                model = fit_e2((*g_examples, *training_a))
                fold_actions: list[dict[str, Any]] = []
                fold_family_counts = {family: 0 for family in core.FAMILY_ORDER}
                for row in test_rows:
                    family = a_labels[row.view.item_id]
                    fold_family_counts[family] += 1
                    space = core.build_action_space(row.item)
                    frontier = core.g8_frontier(row.item, g8, space=space)
                    e0 = frontier.e0.ordinals
                    e2 = e2_select(
                        row.item,
                        g8,
                        model,
                        space=space,
                        frontier=frontier,
                    )
                    raw = core.raw3(row.item)
                    utilities["RAW"].append(core.utility(raw, family, row.item))
                    utilities["E0"].append(core.utility(e0, family, row.item))
                    utilities["E2"].append(core.utility(e2.selected, family, row.item))
                    families.append(family)
                    distinct += e2.selected != e0
                    fallback_count += e2.used_fallback
                    fold_actions.append(
                        {
                            "E0": e0,
                            "E2": e2.selected,
                            "fallback": e2.used_fallback,
                            "item_id": row.view.item_id,
                            "target_family": e2.target_family,
                        }
                    )
                if fold_family_counts != {family: 4 for family in core.FAMILY_ORDER}:
                    raise MavenEreE2Error("E2 fold family balance drifted")
                fold_receipts.append(
                    {
                        "action_sha256": stable_hash(fold_actions),
                        "evaluation_count": 12,
                        "family_counts": fold_family_counts,
                        "fit_sha256": model.fit_sha256,
                        "fold": fold,
                        "training_count": model.item_count,
                    }
                )
            if len(families) != 48:
                raise MavenEreE2Error("E2 aggregate cross-fit count drifted")
            score = {
                "arm_correct_count": {
                    arm: sum(values) for arm, values in utilities.items()
                },
                "behavior_distinct_E2_vs_E0_count": distinct,
                "comparisons": {
                    "E2_minus_E0": _comparison(
                        utilities["E2"], utilities["E0"], families
                    ),
                    "E2_minus_RAW": _comparison(
                        utilities["E2"], utilities["RAW"], families
                    ),
                },
                "fallback_to_E0_count": fallback_count,
                "family_item_count": {
                    family: sum(value == family for value in families)
                    for family in core.FAMILY_ORDER
                },
                "item_count": 48,
            }
            eligible = _feasible(score)
            final_model_file_sha: str | None = None
            final_model_fit_sha: str | None = None
            if eligible:
                phase = "eligible_final_E2_fit"
                all_a = tuple(
                    core.labelled_item(row.item, a_labels[row.view.item_id])
                    for row in a_prepared.items
                )
                final_model = fit_e2((*g_examples, *all_a))
                final_payload = _self_hashed(
                    model_payload(final_model), "model_sha256"
                )
                final_model_file_sha = v1._durable_roundtrip(
                    root / "E2.model.json", final_payload
                )
                final_model_fit_sha = final_model.fit_sha256
            phase = "terminal_result"
            result = _self_hashed(
                {
                    "claim_boundary": {
                        "A_hold_F_search_or_M_search_private_pack_opened": False,
                        "formal_claim_or_promotion_run": False,
                        "hidden_TEST_opened": False,
                        "online_or_external_evaluator_calls": 0,
                        "released_train_or_valid_source_rows_reopened": False,
                    },
                    "design_sha256": DESIGN_SELF_SHA256,
                    "final_E2_model_file_sha256": final_model_file_sha,
                    "final_E2_model_fit_sha256": final_model_fit_sha,
                    "fold_receipts": fold_receipts,
                    "formal_fresh_cohort_eligible": eligible,
                    "schema": SCHEMA,
                    "score": score,
                    "status": (
                        "passed_train_crossfit_fresh_cohort_authorized"
                        if eligible
                        else "stopped_train_crossfit_no_formal_measurement"
                    ),
                    "version": VERSION,
                },
                "result_sha256",
            )
            v1._durable_roundtrip(root / "crossfit.result.json", result)
            return result
    except BaseException as exc:
        failure = _self_hashed(
            {
                "category": type(exc).__name__,
                "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                "phase": phase,
                "schema": "maven_ere_global_family_e2_train_crossfit_failure_v1",
                "status": "terminal_no_retry",
                "version": VERSION,
            },
            "failure_sha256",
        )
        try:
            v1._atomic_write(root / "crossfit.failure.json", failure)
        except BaseException:
            pass
        raise


__all__ = [
    "E2Model",
    "E2Selection",
    "FEATURE_ORDER",
    "MavenEreE2Error",
    "e2_logits",
    "e2_select",
    "fit_e2",
    "item_family_features",
    "model_payload",
    "predict_item_family",
    "run_train_crossfit",
    "validate_formal_provenance",
]
