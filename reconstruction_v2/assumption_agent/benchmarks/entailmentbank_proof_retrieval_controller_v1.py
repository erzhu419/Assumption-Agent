"""Fail-closed formal controller for EntailmentBank G1/E1 and L5 measurement."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import (
    entailmentbank_proof_retrieval_acquisition_v1 as acquisition,
)
from assumption_agent.benchmarks import entailmentbank_proof_retrieval_core_v1 as core
from assumption_agent.benchmarks import (
    entailmentbank_proof_retrieval_runtime_v1 as runtime,
)
from assumption_agent.benchmarks.eraser_evidence_inference_official_hipporag_v1 import (
    adapter as official_adapter,
)
from assumption_agent.benchmarks import eraser_evidence_inference_local_runtime_v1
from replication_runtime.qasc_nli_v1.contract import NLIPair


VERSION = "entailmentbank_proof_retrieval_controller_v2"
OFFICIAL_CONCURRENCY_CAP = 8
FORMAL_ROOT_RELATIVE_PATH = acquisition.PRIVATE_ROOT_RELATIVE_PATH
STUDY_ATTEMPT_RELATIVE_PATH = FORMAL_ROOT_RELATIVE_PATH / "study_attempt.json"
STAGE_ROOT_RELATIVE_PATH = FORMAL_ROOT_RELATIVE_PATH / "study_private"
FORMATION_RESULT_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_g1_e1_formation_result_v2.json"
)
AHOLD_RESULT_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_g1_e1_ahold_result_v2.json"
)
M_RESULT_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_g1_e1_m_result_v2.json"
)
FINAL_RESULT_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_g1_e1_final_result_v2.json"
)


class EntailmentBankControllerError(RuntimeError):
    """A stage, label barrier, action seal, control, or score drifted."""


class FormalStudyRefusal(EntailmentBankControllerError):
    """The one-shot efficacy lifecycle is no longer pristine."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EntailmentBankControllerError(f"{field} is not an object")
    return value


def _read_public(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntailmentBankControllerError(f"{label} is unreadable") from exc
    if acquisition.canonical_json_bytes(value) + b"\n" != raw:
        raise EntailmentBankControllerError(f"{label} is not canonical JSON")
    return _mapping(value, label)


def _write_public_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise FormalStudyRefusal("public stage result already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = acquisition.canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_view_labels(
    items: Sequence[core.LabelFreeItem], labels: Sequence[core.ItemLabel]
) -> None:
    if tuple(item.item_commitment_sha256 for item in items) != tuple(
        label.item_commitment_sha256 for label in labels
    ):
        raise EntailmentBankControllerError("view/label commitment order drifted")


def bind_view_tensors(
    items: Sequence[core.LabelFreeItem], tensors: Sequence[core.ItemTensor]
) -> None:
    if tuple(item.item_commitment_sha256 for item in items) != tuple(
        tensor.item_commitment_sha256 for tensor in tensors
    ):
        raise EntailmentBankControllerError("view/tensor commitment order drifted")


def build_action_feature_matrix(
    items: Sequence[core.LabelFreeItem],
    tensors: Sequence[core.ItemTensor],
    g_model: core.QuantizedRidgeModel,
) -> tuple[
    Mapping[str, Mapping[str, core.Action]],
    Mapping[str, Mapping[str, tuple[int, ...]]],
]:
    bind_view_tensors(items, tensors)
    actions: dict[str, dict[str, core.Action]] = {}
    features: dict[str, dict[str, tuple[int, ...]]] = {}
    for item, tensor in zip(items, tensors, strict=True):
        per_action = {}
        per_feature = {}
        for recipe in core.RECIPE_REGISTRY:
            action = core.execute_recipe(tensor, g_model, recipe.recipe_id)
            per_action[recipe.recipe_id] = action
            per_feature[recipe.recipe_id] = core.evaluator_features(
                item, tensor, g_model, action
            )
        actions[item.item_commitment_sha256] = per_action
        features[item.item_commitment_sha256] = per_feature
    return actions, features


def action_feature_pack(
    *,
    block: str,
    items: Sequence[core.LabelFreeItem],
    actions: Mapping[str, Mapping[str, core.Action]],
    features: Mapping[str, Mapping[str, Sequence[int]]],
    g_model_sha256: str,
) -> Mapping[str, Any]:
    if block not in {"A_form", "F_search"}:
        raise EntailmentBankControllerError("formation action block drifted")
    if len(items) != acquisition.BLOCK_COUNTS[block]:
        raise EntailmentBankControllerError("formation action item count drifted")
    rows = []
    recipe_ids = tuple(recipe.recipe_id for recipe in core.RECIPE_REGISTRY)
    for ordinal, item in enumerate(items):
        commitment = item.item_commitment_sha256
        if set(actions.get(commitment, {})) != set(recipe_ids) or set(
            features.get(commitment, {})
        ) != set(recipe_ids):
            raise EntailmentBankControllerError("formation recipe matrix drifted")
        rows.append(
            {
                "ordinal": ordinal,
                "item_commitment_sha256": commitment,
                "recipes": [
                    {
                        "recipe_id": recipe_id,
                        "action": actions[commitment][recipe_id].payload(),
                        "evaluator_features": list(features[commitment][recipe_id]),
                    }
                    for recipe_id in recipe_ids
                ],
            }
        )
    body = {
        "schema": f"{VERSION}_action_feature_pack",
        "block": block,
        "item_count": len(rows),
        "recipe_count": len(recipe_ids),
        "g_model_sha256": g_model_sha256,
        "items": rows,
        "labels_opened_before_pack_seal": False,
    }
    return acquisition.self_hashed(body, "action_feature_pack_sha256")


def fit_e1_from_a_form(
    *,
    items: Sequence[core.LabelFreeItem],
    labels: Sequence[core.ItemLabel],
    actions: Mapping[str, Mapping[str, core.Action]],
    features: Mapping[str, Mapping[str, Sequence[int]]],
) -> tuple[core.QuantizedRidgeModel, Mapping[str, int]]:
    bind_view_labels(items, labels)
    rows = []
    targets = []
    utility_totals = {recipe.recipe_id: 0 for recipe in core.RECIPE_REGISTRY}
    for item, label in zip(items, labels, strict=True):
        commitment = item.item_commitment_sha256
        for recipe in core.RECIPE_REGISTRY:
            action = actions[commitment][recipe.recipe_id]
            feature = tuple(features[commitment][recipe.recipe_id])
            utility = core.direct_utility(action.selected_ordinals, label)
            rows.append(feature)
            targets.append(utility)
            utility_totals[recipe.recipe_id] += utility
    return core.fit_e1_model(rows, targets), dict(sorted(utility_totals.items()))


@dataclass(frozen=True)
class FrozenPolicies:
    g_model: core.QuantizedRidgeModel
    e1_model: core.QuantizedRidgeModel
    q0_recipe_id: str
    q1_recipe_id: str
    e0_search_totals: Mapping[str, int]
    e1_search_totals: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.q0_recipe_id not in core.RECIPE_BY_ID or self.q1_recipe_id not in core.RECIPE_BY_ID:
            raise EntailmentBankControllerError("frozen policy recipe drifted")


def freeze_policies(
    *,
    g_model: core.QuantizedRidgeModel,
    e1_model: core.QuantizedRidgeModel,
    f_features: Mapping[str, Mapping[str, Sequence[int]]],
) -> FrozenPolicies:
    q0, e0_totals = core.select_global_recipe(f_features, evaluator="E0")
    q1, e1_totals = core.select_global_recipe(
        f_features, evaluator="E1", e1_model=e1_model
    )
    return FrozenPolicies(g_model, e1_model, q0, q1, e0_totals, e1_totals)


OfficialRunner = Callable[[core.LabelFreeItem, Path], tuple[int, ...]]


def execute_measurement_actions(
    *,
    block: str,
    items: Sequence[core.LabelFreeItem],
    tensors: Sequence[core.ItemTensor],
    policies: FrozenPolicies,
    official_runner: OfficialRunner,
    official_work_parent: Path,
    official_workers: int = OFFICIAL_CONCURRENCY_CAP,
) -> Mapping[str, Mapping[str, Any]]:
    if block not in {"A_hold", "M_search"}:
        raise EntailmentBankControllerError("measurement block drifted")
    if len(items) != acquisition.BLOCK_COUNTS[block]:
        raise EntailmentBankControllerError("measurement item count drifted")
    if (
        isinstance(official_workers, bool)
        or not 1 <= official_workers <= OFFICIAL_CONCURRENCY_CAP
    ):
        raise EntailmentBankControllerError("official concurrency drifted")
    bind_view_tensors(items, tensors)
    official_work_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(official_work_parent, 0o700)

    def execute_agent(item_index: int) -> tuple[core.Action, core.Action]:
        tensor = tensors[item_index]
        return (
            core.execute_recipe(tensor, policies.g_model, policies.q0_recipe_id),
            core.execute_recipe(tensor, policies.g_model, policies.q1_recipe_id),
        )

    def execute_raw(_item_index: int) -> tuple[int, ...]:
        return tuple(range(core.TOP_K))

    agent_futures: list[Future[tuple[core.Action, core.Action]]] = []
    raw_futures: list[Future[tuple[int, ...]]] = []
    official_futures: list[Future[tuple[int, ...]]] = []
    with ThreadPoolExecutor(max_workers=max(1, 2 * len(items))) as local_pool, ThreadPoolExecutor(
        max_workers=official_workers
    ) as official_pool:
        # All 3*n arm tasks are submitted before any future result is joined.
        for item_index, item in enumerate(items):
            agent_futures.append(local_pool.submit(execute_agent, item_index))
            raw_futures.append(local_pool.submit(execute_raw, item_index))
            work_root = official_work_parent / f"{block}-{item_index:03d}"
            official_futures.append(
                official_pool.submit(official_runner, item, work_root)
            )
        rows: dict[str, Mapping[str, Any]] = {}
        for item_index, item in enumerate(items):
            q0_action, q1_action = agent_futures[item_index].result()
            raw = tuple(raw_futures[item_index].result())
            hippo = tuple(official_futures[item_index].result())
            for name, ordinals in (("RAW", raw), ("official_HippoRAG", hippo)):
                if (
                    len(ordinals) != core.TOP_K
                    or len(set(ordinals)) != core.TOP_K
                    or any(type(value) is not int or not 0 <= value < 25 for value in ordinals)
                ):
                    raise EntailmentBankControllerError(f"{name} action drifted")
            rows[item.item_commitment_sha256] = {
                "Q0": q0_action,
                "Q1": q1_action,
                "official_HippoRAG": hippo,
                "RAW": raw,
            }
    if len(rows) != len(items):
        raise EntailmentBankControllerError("measurement action registry drifted")
    return rows


def measurement_action_pack(
    *,
    block: str,
    items: Sequence[core.LabelFreeItem],
    policies: FrozenPolicies,
    actions: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if block not in {"A_hold", "M_search"} or len(items) != acquisition.BLOCK_COUNTS.get(block):
        raise EntailmentBankControllerError("measurement action pack shape drifted")
    rows = []
    for ordinal, item in enumerate(items):
        row = actions.get(item.item_commitment_sha256)
        if row is None or set(row) != {"Q0", "Q1", "official_HippoRAG", "RAW"}:
            raise EntailmentBankControllerError("measurement item action set drifted")
        q0 = row["Q0"]
        q1 = row["Q1"]
        if not isinstance(q0, core.Action) or not isinstance(q1, core.Action):
            raise EntailmentBankControllerError("measurement Agent action drifted")
        rows.append(
            {
                "ordinal": ordinal,
                "item_commitment_sha256": item.item_commitment_sha256,
                "Q0_action": q0.payload(),
                "Q1_action": q1.payload(),
                "official_HippoRAG_ordinals": list(row["official_HippoRAG"]),
                "RAW_ordinals": list(row["RAW"]),
            }
        )
    body = {
        "schema": f"{VERSION}_measurement_action_pack",
        "block": block,
        "item_count": len(rows),
        "q0_recipe_id": policies.q0_recipe_id,
        "q1_recipe_id": policies.q1_recipe_id,
        "all_3_times_n_tasks_submitted_before_any_result_join": True,
        "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
        "labels_opened_before_action_pack_seal": False,
        "items": rows,
    }
    return acquisition.self_hashed(body, "measurement_action_pack_sha256")


def _pair_summary(differences: Sequence[int]) -> Mapping[str, Any]:
    test = core.exact_one_sided_signflip(differences)
    return {
        "net_difference": sum(differences),
        "positive_pair_count": sum(value > 0 for value in differences),
        "zero_pair_count": sum(value == 0 for value in differences),
        "negative_pair_count": sum(value < 0 for value in differences),
        "exact_one_sided_signflip": test,
    }


def score_measurement(
    *,
    block: str,
    items: Sequence[core.LabelFreeItem],
    labels: Sequence[core.ItemLabel],
    actions: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if block not in {"A_hold", "M_search"}:
        raise EntailmentBankControllerError("score block drifted")
    if len(items) != acquisition.BLOCK_COUNTS[block] or len(labels) != len(items):
        raise EntailmentBankControllerError("score item count drifted")
    bind_view_labels(items, labels)
    family_counts = Counter(label.family for label in labels)
    if {
        family: family_counts[family] for family in core.FAMILY_ORDER
    } != acquisition.BLOCK_FAMILY_COUNTS[block]:
        raise EntailmentBankControllerError("score family balance drifted")
    utilities = {arm: [] for arm in ("Q0", "Q1", "official_HippoRAG", "RAW")}
    family_utilities = {
        family: {arm: [] for arm in utilities} for family in core.FAMILY_ORDER
    }
    private_rows = []
    for item, label in zip(items, labels, strict=True):
        row = actions[item.item_commitment_sha256]
        ordinals = {
            "Q0": row["Q0"].selected_ordinals,
            "Q1": row["Q1"].selected_ordinals,
            "official_HippoRAG": row["official_HippoRAG"],
            "RAW": row["RAW"],
        }
        values = {
            arm: core.direct_utility(tuple(action_ordinals), label)
            for arm, action_ordinals in ordinals.items()
        }
        for arm, value in values.items():
            utilities[arm].append(value)
            family_utilities[label.family][arm].append(value)
        private_rows.append(
            {
                "item_commitment_sha256": item.item_commitment_sha256,
                "family": label.family,
                "utilities": values,
            }
        )
    differences = {
        "Q1_minus_Q0": [right - left for right, left in zip(utilities["Q1"], utilities["Q0"], strict=True)],
        "Q1_minus_official_HippoRAG": [
            right - left
            for right, left in zip(utilities["Q1"], utilities["official_HippoRAG"], strict=True)
        ],
        "Q1_minus_RAW": [right - left for right, left in zip(utilities["Q1"], utilities["RAW"], strict=True)],
    }
    summaries = {name: _pair_summary(values) for name, values in differences.items()}
    primary = summaries["Q1_minus_Q0"]
    exact = primary["exact_one_sided_signflip"]
    promoted = (
        primary["net_difference"] > 0
        and 10 * exact["tail_numerator"] <= exact["tail_denominator"]
    )
    family_totals = {
        family: {
            arm: sum(family_utilities[family][arm]) for arm in utilities
        }
        for family in core.FAMILY_ORDER
    }
    l5_success = None
    if block == "M_search":
        l5_success = (
            promoted
            and sum(utilities["Q1"]) > sum(utilities["official_HippoRAG"])
            and sum(utilities["Q1"]) > sum(utilities["RAW"])
            and all(
                family_totals[family]["Q1"]
                >= family_totals[family]["official_HippoRAG"]
                and family_totals[family]["Q1"] >= family_totals[family]["RAW"]
                for family in core.FAMILY_ORDER
            )
        )
    return {
        "block": block,
        "item_count": len(items),
        "arm_totals": {arm: sum(values) for arm, values in utilities.items()},
        "family_arm_totals": family_totals,
        "paired_comparisons": summaries,
        "evaluator_promoted": promoted if block == "A_hold" else None,
        "M_search_L5_success": l5_success,
        "private_item_scores": private_rows,
    }


def public_score_result(
    *,
    score: Mapping[str, Any],
    action_pack_sha256: str,
    label_pack_sha256: str,
) -> Mapping[str, Any]:
    block = score["block"]
    body = {
        "schema": f"entailmentbank_proof_retrieval_g1_e1_{block}_result_v2",
        "status": (
            "evaluator_promoted_M_search_authorized"
            if block == "A_hold" and score["evaluator_promoted"]
            else "evaluator_not_promoted_terminal_without_M_search"
            if block == "A_hold"
            else "untouched_M_search_completed"
        ),
        "block": block,
        "item_count": score["item_count"],
        "measurement_action_pack_sha256": action_pack_sha256,
        "label_pack_file_sha256": label_pack_sha256,
        "arm_totals": score["arm_totals"],
        "family_arm_totals": score["family_arm_totals"],
        "paired_comparisons": score["paired_comparisons"],
        "evaluator_promoted": score["evaluator_promoted"],
        "M_search_L5_success": score["M_search_L5_success"],
        "online_or_external_evaluator_calls": 0,
    }
    return acquisition.self_hashed(body, "stage_result_sha256")


def _validate_acquisition_receipt(root: Path) -> Mapping[str, Any]:
    receipt = _read_public(
        root / acquisition.RECEIPT_RELATIVE_PATH, label="acquisition receipt"
    )
    try:
        acquisition.verify_self_hash(receipt, "acquisition_receipt_sha256")
    except acquisition.EntailmentBankAcquisitionError as exc:
        raise FormalStudyRefusal("acquisition receipt hash drifted") from exc
    if (
        receipt.get("status") != "formal_186_item_private_cohort_acquired_before_any_action"
        or receipt.get("design_sha256") != acquisition.DESIGN_SHA256
        or receipt.get("F_search_label_pack_created") is not False
    ):
        raise FormalStudyRefusal("acquisition receipt is not formal-valid")
    return receipt


def _verified_pack_path(
    root: Path, receipt: Mapping[str, Any], *, block: str, role: str
) -> Path:
    name = f"{block}.{role}.private.json"
    hashes = _mapping(receipt.get("private_pack_file_sha256s"), "pack hashes")
    expected = hashes.get(name)
    path = root / acquisition.PACK_ROOT_RELATIVE_PATH / name
    if not isinstance(expected, str) or _file_sha256(path) != expected:
        raise FormalStudyRefusal("private acquisition pack hash drifted")
    return path


def _default_official_runner(project_root: Path) -> OfficialRunner:
    config = eraser_evidence_inference_local_runtime_v1.default_formal_runtime_config(
        project_root
    )

    def run(item: core.LabelFreeItem, work_root: Path) -> tuple[int, ...]:
        return official_adapter.run_item_local_official_hipporag_v1(
            query=item.hypothesis,
            sentence_texts=item.node_texts,
            runtime_python=config.hippo_runtime_python,
            local_llm_model=config.hippo_llm_model,
            local_embedding_model=config.hippo_embedding_model,
            base_binding_receipt_path=config.hippo_base_binding_receipt,
            attestation_receipt_path=config.hippo_attestation_receipt,
            work_root=work_root,
            timeout_seconds=900,
        )

    return run


def _preflight_runtimes(project_root: Path):
    encoder = runtime.create_minilm_encoder(project_root)
    nli_pool = runtime.LocalTwoWorkerNLIPool(project_root=project_root)
    try:
        canary_items = (
            (
                "canary-left",
                (NLIPair("Every mineral is matter.", "Quartz is matter."),),
            ),
            (
                "canary-right",
                (NLIPair("Every bird has wings.", "A robin has wings."),),
            ),
        )
        nli_scores = nli_pool.score_items(canary_items)
        if set(nli_scores) != {"canary-left", "canary-right"}:
            raise EntailmentBankControllerError("two-worker NLI canary drifted")
        official = _default_official_runner(project_root)
        canary_parent = project_root / FORMAL_ROOT_RELATIVE_PATH / "preflight"
        canary_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        canary = core.LabelFreeItem(
            "f" * 64,
            "Which facts are relevant?",
            "Synthetic answer",
            "Synthetic matter relation",
            tuple(f"Synthetic exact fact {index}" for index in range(25)),
        )
        official_output = official(canary, canary_parent / "official-canary")
        if len(official_output) != core.TOP_K:
            raise EntailmentBankControllerError("official HippoRAG canary drifted")
        return encoder, nli_pool, official, {
            "MiniLM_canary": encoder.canary_receipt,
            "NLI_two_worker_canary_scores": {
                key: list(value) for key, value in sorted(nli_scores.items())
            },
            "official_HippoRAG_canary_retrieval_count": len(official_output),
        }
    except BaseException:
        nli_pool.close()
        raise


def run_formal_study(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve()
    root = project_root / "reconstruction_v2"
    receipt = _validate_acquisition_receipt(root)
    marker_path = root / STUDY_ATTEMPT_RELATIVE_PATH
    stage_root = root / STAGE_ROOT_RELATIVE_PATH
    public_paths = (
        root / FORMATION_RESULT_RELATIVE_PATH,
        root / AHOLD_RESULT_RELATIVE_PATH,
        root / M_RESULT_RELATIVE_PATH,
        root / FINAL_RESULT_RELATIVE_PATH,
    )
    if marker_path.exists() or stage_root.exists() or any(path.exists() for path in public_paths):
        raise FormalStudyRefusal("formal study stage is not pristine")
    encoder, nli_pool, official_runner, preflight = _preflight_runtimes(root)
    try:
        marker = acquisition.self_hashed(
            {
                "schema": f"{VERSION}_attempt",
                "status": "started_after_row_free_runtime_preflight_before_private_view_open",
                "acquisition_receipt_sha256": receipt["acquisition_receipt_sha256"],
                "preflight_sha256": acquisition.stable_hash(preflight),
            },
            "study_attempt_sha256",
        )
        runtime.write_private_json(marker_path, marker)

        # G is tensorized and sealed before its permitted formation labels open.
        g_items = runtime.load_view_pack(
            _verified_pack_path(root, receipt, block="G_form", role="view"),
            block="G_form",
        )
        g_tensors = runtime.build_item_tensors(
            g_items, minilm_encoder=encoder, nli_scorer=nli_pool
        )
        g_tensor_value = runtime.tensor_pack("G_form", g_tensors)
        g_tensor_file = stage_root / "G_form.tensor.private.json"
        g_tensor_file_sha = runtime.write_private_json(g_tensor_file, g_tensor_value)
        g_labels_path = _verified_pack_path(
            root, receipt, block="G_form", role="labels"
        )
        g_labels = runtime.load_label_pack(g_labels_path, block="G_form")
        bind_view_labels(g_items, g_labels)
        g_model = core.fit_g_model(g_tensors, g_labels)

        # A and F actions are both sealed while A labels remain physically unopened.
        a_items = runtime.load_view_pack(
            _verified_pack_path(root, receipt, block="A_form", role="view"),
            block="A_form",
        )
        f_items = runtime.load_view_pack(
            _verified_pack_path(root, receipt, block="F_search", role="view"),
            block="F_search",
        )
        a_tensors = runtime.build_item_tensors(
            a_items, minilm_encoder=encoder, nli_scorer=nli_pool
        )
        f_tensors = runtime.build_item_tensors(
            f_items, minilm_encoder=encoder, nli_scorer=nli_pool
        )
        tensor_hashes = {"G_form": g_tensor_file_sha}
        for block, tensors in (("A_form", a_tensors), ("F_search", f_tensors)):
            tensor_hashes[block] = runtime.write_private_json(
                stage_root / f"{block}.tensor.private.json",
                runtime.tensor_pack(block, tensors),
            )
        a_actions, a_features = build_action_feature_matrix(a_items, a_tensors, g_model)
        f_actions, f_features = build_action_feature_matrix(f_items, f_tensors, g_model)
        a_action_value = action_feature_pack(
            block="A_form",
            items=a_items,
            actions=a_actions,
            features=a_features,
            g_model_sha256=g_model.model_sha256,
        )
        f_action_value = action_feature_pack(
            block="F_search",
            items=f_items,
            actions=f_actions,
            features=f_features,
            g_model_sha256=g_model.model_sha256,
        )
        a_action_file_sha = runtime.write_private_json(
            stage_root / "A_form.actions.private.json", a_action_value
        )
        f_action_file_sha = runtime.write_private_json(
            stage_root / "F_search.actions.private.json", f_action_value
        )

        a_labels_path = _verified_pack_path(
            root, receipt, block="A_form", role="labels"
        )
        a_labels = runtime.load_label_pack(a_labels_path, block="A_form")
        e1_model, a_utility_totals = fit_e1_from_a_form(
            items=a_items,
            labels=a_labels,
            actions=a_actions,
            features=a_features,
        )
        policies = freeze_policies(
            g_model=g_model, e1_model=e1_model, f_features=f_features
        )
        formation_body = {
            "schema": "entailmentbank_proof_retrieval_g1_e1_formation_result_v2",
            "status": "G1_E1_Q0_Q1_frozen_before_A_hold_view_open",
            "acquisition_receipt_sha256": receipt["acquisition_receipt_sha256"],
            "G_model": g_model.payload(),
            "E1_model": e1_model.payload(),
            "Q0_recipe_id": policies.q0_recipe_id,
            "Q1_recipe_id": policies.q1_recipe_id,
            "A_form_actual_utility_totals_by_recipe": a_utility_totals,
            "F_search_E0_predicted_totals_by_recipe": policies.e0_search_totals,
            "F_search_E1_predicted_totals_by_recipe": policies.e1_search_totals,
            "private_tensor_file_sha256s": dict(sorted(tensor_hashes.items())),
            "A_form_action_feature_file_sha256": a_action_file_sha,
            "F_search_action_feature_file_sha256": f_action_file_sha,
            "F_search_label_pack_created_or_opened": False,
            "A_hold_or_M_view_or_label_opened": False,
            "external_network_calls": 0,
        }
        formation = acquisition.self_hashed(formation_body, "formation_result_sha256")
        _write_public_once(root / FORMATION_RESULT_RELATIVE_PATH, formation)

        a_hold_score = _run_formal_measurement_block(
            root=root,
            receipt=receipt,
            block="A_hold",
            policies=policies,
            encoder=encoder,
            nli_pool=nli_pool,
            official_runner=official_runner,
            stage_root=stage_root,
        )
        a_hold_public = public_score_result(
            score=a_hold_score["score"],
            action_pack_sha256=a_hold_score["action_pack_sha256"],
            label_pack_sha256=a_hold_score["label_pack_file_sha256"],
        )
        _write_public_once(root / AHOLD_RESULT_RELATIVE_PATH, a_hold_public)

        m_public = None
        if a_hold_score["score"]["evaluator_promoted"]:
            m_score = _run_formal_measurement_block(
                root=root,
                receipt=receipt,
                block="M_search",
                policies=policies,
                encoder=encoder,
                nli_pool=nli_pool,
                official_runner=official_runner,
                stage_root=stage_root,
            )
            m_public = public_score_result(
                score=m_score["score"],
                action_pack_sha256=m_score["action_pack_sha256"],
                label_pack_sha256=m_score["label_pack_file_sha256"],
            )
            _write_public_once(root / M_RESULT_RELATIVE_PATH, m_public)

        final_body = {
            "schema": "entailmentbank_proof_retrieval_g1_e1_final_result_v2",
            "status": (
                "completed_untouched_M_search"
                if m_public is not None
                else "terminal_nonpromotion_M_search_unopened"
            ),
            "formation_result_sha256": formation["formation_result_sha256"],
            "A_hold_result_sha256": a_hold_public["stage_result_sha256"],
            "M_search_result_sha256": (
                m_public["stage_result_sha256"] if m_public is not None else None
            ),
            "evaluator_promoted": a_hold_public["evaluator_promoted"],
            "M_search_L5_success": (
                m_public["M_search_L5_success"] if m_public is not None else None
            ),
            "M_search_view_or_label_opened": m_public is not None,
            "test_split_opened_hashed_or_parsed": False,
            "online_or_external_evaluator_calls": 0,
            "same_source_retry_or_gate_patch": False,
        }
        final = acquisition.self_hashed(final_body, "final_result_sha256")
        _write_public_once(root / FINAL_RESULT_RELATIVE_PATH, final)
        return final
    finally:
        nli_pool.close()


def _run_formal_measurement_block(
    *,
    root: Path,
    receipt: Mapping[str, Any],
    block: str,
    policies: FrozenPolicies,
    encoder: Any,
    nli_pool: runtime.LocalTwoWorkerNLIPool,
    official_runner: OfficialRunner,
    stage_root: Path,
) -> Mapping[str, Any]:
    view_path = _verified_pack_path(root, receipt, block=block, role="view")
    items = runtime.load_view_pack(view_path, block=block)
    tensors = runtime.build_item_tensors(
        items, minilm_encoder=encoder, nli_scorer=nli_pool
    )
    tensor_file_sha = runtime.write_private_json(
        stage_root / f"{block}.tensor.private.json",
        runtime.tensor_pack(block, tensors),
    )
    actions = execute_measurement_actions(
        block=block,
        items=items,
        tensors=tensors,
        policies=policies,
        official_runner=official_runner,
        official_work_parent=stage_root / "official_hipporag" / block,
        official_workers=OFFICIAL_CONCURRENCY_CAP,
    )
    action_value = measurement_action_pack(
        block=block, items=items, policies=policies, actions=actions
    )
    action_file_sha = runtime.write_private_json(
        stage_root / f"{block}.actions.private.json", action_value
    )
    # The late label pack is not even hashed until every arm action is durable.
    labels_path = _verified_pack_path(root, receipt, block=block, role="labels")
    labels_file_sha = _file_sha256(labels_path)
    labels = runtime.load_label_pack(labels_path, block=block)
    score = score_measurement(
        block=block, items=items, labels=labels, actions=actions
    )
    private_rows = score["private_item_scores"]
    public_score = {
        key: value for key, value in score.items() if key != "private_item_scores"
    }
    private_score_body = {
        "schema": f"{VERSION}_private_measurement_scores",
        "block": block,
        "tensor_file_sha256": tensor_file_sha,
        "action_pack_sha256": action_value["measurement_action_pack_sha256"],
        "action_pack_file_sha256": action_file_sha,
        "label_pack_file_sha256": labels_file_sha,
        "items": private_rows,
    }
    private_score = acquisition.self_hashed(
        private_score_body, "private_score_pack_sha256"
    )
    runtime.write_private_json(
        stage_root / f"{block}.scores.private.json", private_score
    )
    return {
        "score": public_score,
        "action_pack_sha256": action_value["measurement_action_pack_sha256"],
        "label_pack_file_sha256": labels_file_sha,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_formal_study(args.project_root)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EntailmentBankControllerError",
    "FormalStudyRefusal",
    "FrozenPolicies",
    "action_feature_pack",
    "bind_view_labels",
    "bind_view_tensors",
    "build_action_feature_matrix",
    "execute_measurement_actions",
    "fit_e1_from_a_form",
    "freeze_policies",
    "measurement_action_pack",
    "public_score_result",
    "run_formal_study",
    "score_measurement",
]
