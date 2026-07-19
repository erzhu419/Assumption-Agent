"""One-shot offline formal lifecycle for the frozen MAVEN-ERE G8/E1 study."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core
from assumption_agent.benchmarks import maven_ere_local_runtime_v1 as local_runtime


VERSION = "v1"
SCHEMA = "maven_ere_g8_e1_formal_controller_v1"
DESIGN_SELF_SHA256 = "314a9804d32a3c3fb848e0100bc62bc693a468e8e3ac09c9baf018c7cfeee417"
FORMAL_ROOT_RELATIVE = Path("artifacts/maven_ere_g8_e1_formal_v1")
ACQUISITION_RELATIVE = FORMAL_ROOT_RELATIVE / "acquisition"
CONTROLLER_RELATIVE = FORMAL_ROOT_RELATIVE / "controller"
PROMOTION_ALPHA = Fraction(1, 10)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MavenEreFormalControllerError(RuntimeError):
    """The frozen lifecycle, access order, action, or score contract drifted."""


class OneShotRefusal(MavenEreFormalControllerError):
    """The formal controller root is not pristine."""


@dataclass(frozen=True)
class RawArtifact:
    item_id: str
    selected: tuple[int, int, int]


@dataclass(frozen=True)
class HippoArtifact:
    item_id: str
    selected: tuple[int, int, int]


@dataclass(frozen=True)
class AgentArtifact:
    item_id: str
    e0_selected: tuple[int, int, int]
    e1_selected: tuple[int, int, int] | None
    e0_behavior_sha256: str
    e1_behavior_sha256: str | None
    frontier_sha256: str
    edge_deletion_witness_count: int
    edge_deletion_action_change_count: int


@dataclass(frozen=True)
class ItemExecution:
    prepared: local_runtime.PreparedItem
    raw: RawArtifact
    hippo: HippoArtifact
    agent: AgentArtifact


@dataclass(frozen=True)
class BlockExecution:
    block: str
    items: tuple[ItemExecution, ...]
    all_3n_tasks_submitted_before_first_result: bool
    logical_task_count: int
    local_physical_cap: int
    hipporag_physical_cap: int


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
        raise MavenEreFormalControllerError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _self_hashed(body: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    output = dict(body)
    output[field_name] = stable_hash(output)
    return output


def _exclusive_write(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _exclusive_write(temporary, value)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_roundtrip(path: Path, value: Mapping[str, Any]) -> str:
    _atomic_write(path, value)
    raw = path.read_bytes()
    if raw != _canonical_bytes(value):
        raise MavenEreFormalControllerError("durable artifact bytes drifted")
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreFormalControllerError("durable artifact cannot be decoded") from exc
    if decoded != value:
        raise MavenEreFormalControllerError("durable artifact semantic drifted")
    return hashlib.sha256(raw).hexdigest()


def _read_self_hashed_pack(
    path: Path, *, schema: str, block: str
) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise MavenEreFormalControllerError("label pack is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MavenEreFormalControllerError("label pack is invalid") from exc
    if not isinstance(value, dict):
        raise MavenEreFormalControllerError("label pack root must be an object")
    body = dict(value)
    declared = body.pop("pack_sha256", None)
    if (
        value.get("schema") != schema
        or value.get("version") != "v1"
        or value.get("block") != block
        or declared != stable_hash(body)
    ):
        raise MavenEreFormalControllerError("label pack binding drifted")
    return value


def load_family_labels(
    path: str | Path,
    *,
    block: str,
    expected_item_ids: Sequence[str],
) -> Mapping[str, str]:
    if block not in {"G_form", "A_form", "A_hold", "M_search"}:
        raise MavenEreFormalControllerError("label access is forbidden for this block")
    pack = _read_self_hashed_pack(
        Path(path),
        schema="maven_ere_g8_e1_family_label_pack_v1",
        block=block,
    )
    items = pack.get("items")
    if not isinstance(items, list) or len(items) != pack.get("item_count"):
        raise MavenEreFormalControllerError("label item count drifted")
    result: dict[str, str] = {}
    for row in items:
        if not isinstance(row, Mapping) or set(row) != {"family", "item_id"}:
            raise MavenEreFormalControllerError("label row shape drifted")
        item_id, family = row.get("item_id"), row.get("family")
        if (
            not isinstance(item_id, str)
            or not _SHA256.fullmatch(item_id)
            or item_id in result
            or family not in core.FAMILY_ORDER
        ):
            raise MavenEreFormalControllerError("label row is invalid")
        result[item_id] = str(family)
    if set(result) != set(expected_item_ids) or len(result) != len(expected_item_ids):
        raise MavenEreFormalControllerError("label/view item keysets differ")
    counts = {family: sum(value == family for value in result.values()) for family in core.FAMILY_ORDER}
    expected_per_family = {
        "G_form": 32,
        "A_form": 16,
        "A_hold": 10,
        "M_search": 10,
    }[block]
    if counts != {family: expected_per_family for family in core.FAMILY_ORDER}:
        raise MavenEreFormalControllerError("label family balance drifted")
    return result


def _raw_task(prepared: local_runtime.PreparedItem) -> RawArtifact:
    # Independent execution: no Agent future or cached action is accepted.
    return RawArtifact(prepared.view.item_id, core.raw3(prepared.item))


def _agent_task(
    prepared: local_runtime.PreparedItem,
    g8_model: core.G8Model,
    e1_model: core.E1Model | None,
    causal_audit: bool,
) -> AgentArtifact:
    space = core.build_action_space(prepared.item)
    frontier = core.g8_frontier(prepared.item, g8_model, space=space)
    e0 = frontier.e0.ordinals
    e1 = (
        core.e1_select(space, frontier, e1_model).entry.ordinals
        if e1_model is not None
        else None
    )
    deletion = (
        core.edge_deletion_redecode(prepared.item, g8_model, e1_model=e1_model)
        if causal_audit
        else ()
    )
    return AgentArtifact(
        item_id=prepared.view.item_id,
        e0_selected=e0,
        e1_selected=e1,
        e0_behavior_sha256=core.behavior_hash(prepared.item, space, frontier, e0),
        e1_behavior_sha256=(
            core.behavior_hash(prepared.item, space, frontier, e1)
            if e1 is not None
            else None
        ),
        frontier_sha256=stable_hash(
            [
                {"energy_hex": row.generator_energy.hex(), "ordinals": row.ordinals}
                for row in frontier.entries
            ]
        ),
        edge_deletion_witness_count=len(deletion),
        edge_deletion_action_change_count=sum(
            row.e0_changed or row.e1_changed is True for row in deletion
        ),
    )


def _hippo_task(
    gateway: local_runtime.OfficialHippoGateway,
    block: str,
    prepared: local_runtime.PreparedItem,
) -> HippoArtifact:
    return HippoArtifact(
        prepared.view.item_id,
        gateway.retrieve(block=block, view=prepared.view),
    )


def execute_block(
    *,
    prepared: local_runtime.PreparedBlock,
    g8_model: core.G8Model,
    e1_model: core.E1Model | None,
    hippo: local_runtime.OfficialHippoGateway,
    causal_audit: bool,
    local_cap: int = local_runtime.LOCAL_TASK_PHYSICAL_CAP,
    hippo_cap: int = local_runtime.HIPPORAG_PHYSICAL_CAP,
) -> BlockExecution:
    """Eagerly submit exactly three logical arm tasks per item before collect."""

    if prepared.block not in local_runtime.BLOCK_ORDER or not prepared.items:
        raise MavenEreFormalControllerError("prepared block is invalid")
    if local_cap != 16 or hippo_cap != 2:
        raise MavenEreFormalControllerError("formal physical caps drifted")
    futures: list[
        tuple[
            local_runtime.PreparedItem,
            Future[RawArtifact],
            Future[HippoArtifact],
            Future[AgentArtifact],
        ]
    ] = []
    with ThreadPoolExecutor(max_workers=local_cap) as local_executor, ThreadPoolExecutor(
        max_workers=hippo_cap
    ) as hippo_executor:
        for row in prepared.items:
            raw_future = local_executor.submit(_raw_task, row)
            hippo_future = hippo_executor.submit(_hippo_task, hippo, prepared.block, row)
            agent_future = local_executor.submit(
                _agent_task, row, g8_model, e1_model, causal_audit
            )
            futures.append((row, raw_future, hippo_future, agent_future))
        submitted = len(futures) == len(prepared.items)
        results: list[ItemExecution] = []
        # This is deliberately the first Future.result call in the function.
        for row, raw_future, hippo_future, agent_future in futures:
            raw = raw_future.result()
            hippo_result = hippo_future.result()
            agent = agent_future.result()
            if not (
                raw.item_id == hippo_result.item_id == agent.item_id == row.view.item_id
            ):
                raise MavenEreFormalControllerError("arm item identity drifted")
            results.append(ItemExecution(row, raw, hippo_result, agent))
    return BlockExecution(
        block=prepared.block,
        items=tuple(results),
        all_3n_tasks_submitted_before_first_result=submitted,
        logical_task_count=3 * len(results),
        local_physical_cap=local_cap,
        hipporag_physical_cap=hippo_cap,
    )


def _semantic_archive(prepared: local_runtime.PreparedBlock) -> dict[str, Any]:
    return _self_hashed(
        {
            "block": prepared.block,
            "item_count": len(prepared.items),
            "items": [
                {
                    "action_item_commitment": core.action_item_commitment(row.item),
                    "item_id": row.view.item_id,
                    "semantic_receipt_sha256": row.semantic_receipt_sha256,
                }
                for row in prepared.items
            ],
            "preparation_sha256": prepared.preparation_sha256,
            "schema": "maven_ere_semantic_action_item_archive_v1",
            "version": VERSION,
        },
        "archive_sha256",
    )


def _action_archive(execution: BlockExecution) -> dict[str, Any]:
    return _self_hashed(
        {
            "all_3n_tasks_submitted_before_first_result": execution.all_3n_tasks_submitted_before_first_result,
            "block": execution.block,
            "hipporag_physical_cap": execution.hipporag_physical_cap,
            "item_count": len(execution.items),
            "items": [
                {
                    "agent": {
                        "e0_behavior_sha256": row.agent.e0_behavior_sha256,
                        "e0_selected": row.agent.e0_selected,
                        "e1_behavior_sha256": row.agent.e1_behavior_sha256,
                        "e1_selected": row.agent.e1_selected,
                        "edge_deletion_action_change_count": row.agent.edge_deletion_action_change_count,
                        "edge_deletion_witness_count": row.agent.edge_deletion_witness_count,
                        "frontier_sha256": row.agent.frontier_sha256,
                    },
                    "hipporag_selected": row.hippo.selected,
                    "item_id": row.prepared.view.item_id,
                    "raw_selected": row.raw.selected,
                    "semantic_receipt_sha256": row.prepared.semantic_receipt_sha256,
                }
                for row in execution.items
            ],
            "local_physical_cap": execution.local_physical_cap,
            "logical_task_count": execution.logical_task_count,
            "schema": "maven_ere_three_arm_action_archive_v1",
            "version": VERSION,
        },
        "archive_sha256",
    )


def _comparison(
    left: Sequence[int],
    right: Sequence[int],
    families: Sequence[str],
) -> dict[str, Any]:
    if not (len(left) == len(right) == len(families)):
        raise MavenEreFormalControllerError("comparison vector length drifted")
    deltas = [a - b for a, b in zip(left, right, strict=True)]
    sign = core.exact_sign_flip(deltas)
    return {
        "family_net": {
            family: sum(
                delta for delta, observed_family in zip(deltas, families, strict=True)
                if observed_family == family
            )
            for family in core.FAMILY_ORDER
        },
        "gain_count": sum(value > 0 for value in deltas),
        "harm_count": sum(value < 0 for value in deltas),
        "net_utility": sum(deltas),
        "nonzero_pair_count": sign.nonzero_pair_count,
        "p_value": {
            "denominator": sign.p_value.denominator,
            "numerator": sign.p_value.numerator,
        },
        "tie_count": sum(value == 0 for value in deltas),
    }


def _p_at_most(comparison: Mapping[str, Any], alpha: Fraction) -> bool:
    value = comparison["p_value"]
    return int(value["numerator"]) * alpha.denominator <= int(
        value["denominator"]
    ) * alpha.numerator


def score_block(
    execution: BlockExecution,
    labels: Mapping[str, str],
) -> dict[str, Any]:
    if any(row.agent.e1_selected is None for row in execution.items):
        raise MavenEreFormalControllerError("scored block lacks E1 actions")
    families = [labels[row.prepared.view.item_id] for row in execution.items]
    utilities: dict[str, list[int]] = {arm: [] for arm in ("RAW", "HippoRAG", "E0", "E1")}
    for row, family in zip(execution.items, families, strict=True):
        utilities["RAW"].append(core.utility(row.raw.selected, family, row.prepared.item))
        utilities["HippoRAG"].append(
            core.utility(row.hippo.selected, family, row.prepared.item)
        )
        utilities["E0"].append(
            core.utility(row.agent.e0_selected, family, row.prepared.item)
        )
        assert row.agent.e1_selected is not None
        utilities["E1"].append(
            core.utility(row.agent.e1_selected, family, row.prepared.item)
        )
    comparisons = {
        "E1_minus_E0": _comparison(utilities["E1"], utilities["E0"], families),
        "E1_minus_HippoRAG": _comparison(
            utilities["E1"], utilities["HippoRAG"], families
        ),
        "E1_minus_RAW": _comparison(utilities["E1"], utilities["RAW"], families),
    }
    return {
        "arm_correct_count": {arm: sum(values) for arm, values in utilities.items()},
        "behavior_distinct_E1_vs_E0_count": sum(
            row.agent.e1_selected != row.agent.e0_selected for row in execution.items
        ),
        "comparisons": comparisons,
        "edge_deletion_action_change_item_count": sum(
            row.agent.edge_deletion_action_change_count > 0 for row in execution.items
        ),
        "edge_deletion_witness_count": sum(
            row.agent.edge_deletion_witness_count for row in execution.items
        ),
        "family_item_count": {
            family: sum(value == family for value in families)
            for family in core.FAMILY_ORDER
        },
        "item_count": len(execution.items),
    }


def evaluator_promoted(score: Mapping[str, Any]) -> bool:
    comparison = score["comparisons"]["E1_minus_E0"]
    return bool(
        comparison["net_utility"] > 0
        and _p_at_most(comparison, PROMOTION_ALPHA)
        and all(value >= 0 for value in comparison["family_net"].values())
        and score["behavior_distinct_E1_vs_E0_count"] > 0
    )


def real_domain_primary_passed(score: Mapping[str, Any]) -> bool:
    hippo = score["comparisons"]["E1_minus_HippoRAG"]
    raw = score["comparisons"]["E1_minus_RAW"]
    return bool(
        hippo["net_utility"] > 0
        and _p_at_most(hippo, PROMOTION_ALPHA)
        and all(value > 0 for value in hippo["family_net"].values())
        and raw["net_utility"] > 0
        and _p_at_most(raw, PROMOTION_ALPHA)
    )


def _block_paths(acquisition_root: Path, block: str) -> tuple[Path, Path]:
    private = acquisition_root / "private_packs"
    return private / f"{block}.view.json", private / f"{block}.labels.json"


def _load_and_prepare(
    runtime: local_runtime.RuntimeBundle,
    acquisition_root: Path,
    block: str,
) -> local_runtime.PreparedBlock:
    view_path, _label_path = _block_paths(acquisition_root, block)
    views = local_runtime.load_view_pack(view_path, block=block)
    expected = {"G_form": 96, "A_form": 48, "F_search": 36, "A_hold": 30, "M_search": 30}[block]
    if len(views) != expected:
        raise MavenEreFormalControllerError("formal block view count drifted")
    return runtime.prepare_block(block, views)


def _labels_for(
    acquisition_root: Path,
    prepared: local_runtime.PreparedBlock,
) -> Mapping[str, str]:
    _view_path, label_path = _block_paths(acquisition_root, prepared.block)
    return load_family_labels(
        label_path,
        block=prepared.block,
        expected_item_ids=tuple(row.view.item_id for row in prepared.items),
    )


def _fit_g8(
    prepared: local_runtime.PreparedBlock, labels: Mapping[str, str]
) -> core.G8Model:
    return core.fit_g8(
        tuple(
            core.labelled_item(row.item, labels[row.view.item_id])
            for row in prepared.items
        )
    )


def _fit_e1(
    prepared: local_runtime.PreparedBlock,
    labels: Mapping[str, str],
    g8_model: core.G8Model,
) -> core.E1Model:
    return core.fit_e1(
        tuple(
            core.labelled_item(row.item, labels[row.view.item_id])
            for row in prepared.items
        ),
        g8_model,
    )


def run_formal_lifecycle(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    acquisition_root = project / ACQUISITION_RELATIVE
    controller_root = project / CONTROLLER_RELATIVE
    if os.path.lexists(controller_root):
        raise OneShotRefusal("formal controller root already exists")
    if not (acquisition_root / "acquisition.receipt.json").is_file():
        raise MavenEreFormalControllerError("completed acquisition receipt is unavailable")
    config = local_runtime.default_formal_runtime_config(project)
    # Hash every runtime and the official transport before consuming lifecycle.
    preflight = local_runtime.preflight_formal_runtime_config(config)
    controller_root.mkdir(mode=0o700)
    marker = _self_hashed(
        {
            "design_sha256": DESIGN_SELF_SHA256,
            "schema": "maven_ere_g8_e1_lifecycle_authorization_consumed_v1",
            "status": "consumed_before_private_view_or_label_open",
            "version": VERSION,
        },
        "marker_sha256",
    )
    _exclusive_write(controller_root / "lifecycle.authorization.consumed.json", marker)
    _durable_roundtrip(controller_root / "runtime.preflight.json", preflight)
    phase = "runtime_open"
    try:
        with local_runtime.RuntimeBundle(config) as runtime:
            assert runtime.hippo is not None
            hippo = runtime.hippo

            phase = "G_form_label_free_semantics"
            g_prepared = _load_and_prepare(runtime, acquisition_root, "G_form")
            g_semantic = _semantic_archive(g_prepared)
            g_semantic_file_sha = _durable_roundtrip(
                controller_root / "G_form.semantic.archive.json", g_semantic
            )
            phase = "G_form_label_open_and_G8_fit"
            g_labels = _labels_for(acquisition_root, g_prepared)
            g8_model = _fit_g8(g_prepared, g_labels)
            g8_payload = _self_hashed(core.g8_model_payload(g8_model), "model_sha256")
            g8_file_sha = _durable_roundtrip(controller_root / "G8.model.json", g8_payload)

            phase = "A_form_label_free_actions"
            hippo.prepare_blocks(("A_form",))
            a_prepared = _load_and_prepare(runtime, acquisition_root, "A_form")
            a_semantic_file_sha = _durable_roundtrip(
                controller_root / "A_form.semantic.archive.json",
                _semantic_archive(a_prepared),
            )
            a_execution = execute_block(
                prepared=a_prepared,
                g8_model=g8_model,
                e1_model=None,
                hippo=hippo,
                causal_audit=False,
            )
            a_action_file_sha = _durable_roundtrip(
                controller_root / "A_form.action.archive.json",
                _action_archive(a_execution),
            )
            phase = "A_form_label_open_and_E1_fit"
            a_labels = _labels_for(acquisition_root, a_prepared)
            e1_model = _fit_e1(a_prepared, a_labels, g8_model)
            e1_payload = _self_hashed(core.e1_model_payload(e1_model), "model_sha256")
            e1_file_sha = _durable_roundtrip(controller_root / "E1.model.json", e1_payload)

            phase = "F_search_label_free_actions"
            hippo.prepare_blocks(("F_search",))
            f_prepared = _load_and_prepare(runtime, acquisition_root, "F_search")
            f_semantic_file_sha = _durable_roundtrip(
                controller_root / "F_search.semantic.archive.json",
                _semantic_archive(f_prepared),
            )
            f_execution = execute_block(
                prepared=f_prepared,
                g8_model=g8_model,
                e1_model=e1_model,
                hippo=hippo,
                causal_audit=False,
            )
            f_action_file_sha = _durable_roundtrip(
                controller_root / "F_search.action.archive.json",
                _action_archive(f_execution),
            )
            f_distinct = sum(
                row.agent.e1_selected != row.agent.e0_selected for row in f_execution.items
            )
            freeze = _self_hashed(
                {
                    "A_form_action_file_sha256": a_action_file_sha,
                    "E1_file_sha256": e1_file_sha,
                    "E1_fit_sha256": e1_model.fit_sha256,
                    "F_search_E1_vs_E0_behavior_distinct_item_count": f_distinct,
                    "F_search_action_file_sha256": f_action_file_sha,
                    "G8_file_sha256": g8_file_sha,
                    "G8_fit_sha256": g8_model.fit_sha256,
                    "design_sha256": DESIGN_SELF_SHA256,
                    "promotion_alpha": {"denominator": 10, "numerator": 1},
                    "schema": "maven_ere_A_hold_pre_run_freeze_v1",
                    "status": "frozen_before_A_hold_view_open",
                    "version": VERSION,
                },
                "freeze_sha256",
            )
            a_hold_freeze_sha = _durable_roundtrip(
                controller_root / "A_hold.pre_run.freeze.json", freeze
            )

            phase = "A_hold_label_free_actions"
            hippo.prepare_blocks(("A_hold",))
            hold_prepared = _load_and_prepare(runtime, acquisition_root, "A_hold")
            hold_semantic_file_sha = _durable_roundtrip(
                controller_root / "A_hold.semantic.archive.json",
                _semantic_archive(hold_prepared),
            )
            hold_execution = execute_block(
                prepared=hold_prepared,
                g8_model=g8_model,
                e1_model=e1_model,
                hippo=hippo,
                causal_audit=True,
            )
            hold_action_file_sha = _durable_roundtrip(
                controller_root / "A_hold.action.archive.json",
                _action_archive(hold_execution),
            )
            phase = "A_hold_label_open_and_score"
            hold_labels = _labels_for(acquisition_root, hold_prepared)
            hold_score = score_block(hold_execution, hold_labels)
            promotion = evaluator_promoted(hold_score)
            primary = real_domain_primary_passed(hold_score)
            hold_report = _self_hashed(
                {
                    "A_hold_action_file_sha256": hold_action_file_sha,
                    "A_hold_pre_run_freeze_file_sha256": a_hold_freeze_sha,
                    "A_hold_semantic_file_sha256": hold_semantic_file_sha,
                    "evaluator_promoted": promotion,
                    "real_domain_primary_passed": primary,
                    "schema": "maven_ere_A_hold_aggregate_report_v1",
                    "score": hold_score,
                    "version": VERSION,
                },
                "report_sha256",
            )
            hold_report_file_sha = _durable_roundtrip(
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
                        "schema": "maven_ere_M_search_pre_run_freeze_v1",
                        "status": "frozen_before_M_search_view_open",
                        "version": VERSION,
                    },
                    "freeze_sha256",
                )
                m_freeze_file_sha = _durable_roundtrip(
                    controller_root / "M_search.pre_run.freeze.json", m_freeze
                )
                phase = "M_search_label_free_actions"
                hippo.prepare_blocks(("M_search",))
                m_prepared = _load_and_prepare(runtime, acquisition_root, "M_search")
                m_semantic_file_sha = _durable_roundtrip(
                    controller_root / "M_search.semantic.archive.json",
                    _semantic_archive(m_prepared),
                )
                m_execution = execute_block(
                    prepared=m_prepared,
                    g8_model=g8_model,
                    e1_model=e1_model,
                    hippo=hippo,
                    causal_audit=True,
                )
                m_action_file_sha = _durable_roundtrip(
                    controller_root / "M_search.action.archive.json",
                    _action_archive(m_execution),
                )
                phase = "M_search_label_open_and_score"
                m_labels = _labels_for(acquisition_root, m_prepared)
                m_score = score_block(m_execution, m_labels)
                m_l5 = evaluator_promoted(m_score)
                m_report = _self_hashed(
                    {
                        "M_L5_passed": m_l5,
                        "M_search_action_file_sha256": m_action_file_sha,
                        "M_search_pre_run_freeze_file_sha256": m_freeze_file_sha,
                        "M_search_semantic_file_sha256": m_semantic_file_sha,
                        "schema": "maven_ere_M_search_aggregate_report_v1",
                        "score": m_score,
                        "version": VERSION,
                    },
                    "report_sha256",
                )
                m_report_file_sha = _durable_roundtrip(
                    controller_root / "M_search.aggregate.report.json", m_report
                )

            phase = "terminal_result"
            terminal = _self_hashed(
                {
                    "A_hold": hold_report,
                    "M_L5_passed": (
                        evaluator_promoted(m_score) if m_score is not None else None
                    ),
                    "M_search": m_score,
                    "M_search_opened": promotion,
                    "artifact_bindings": {
                        "A_form_action_file_sha256": a_action_file_sha,
                        "A_form_semantic_file_sha256": a_semantic_file_sha,
                        "A_hold_report_file_sha256": hold_report_file_sha,
                        "F_search_action_file_sha256": f_action_file_sha,
                        "F_search_semantic_file_sha256": f_semantic_file_sha,
                        "G_form_semantic_file_sha256": g_semantic_file_sha,
                        "M_search_report_file_sha256": m_report_file_sha,
                    },
                    "claim_boundary": {
                        "hidden_TEST_opened": False,
                        "online_or_external_evaluator_calls": 0,
                        "same_source_retry_or_resample": False,
                    },
                    "evaluator_promoted": promotion,
                    "real_domain_primary_passed": primary,
                    "schema": "maven_ere_g8_e1_terminal_result_v1",
                    "status": (
                        "valid_promotion_M_measured"
                        if promotion
                        else "valid_no_promotion_M_unopened"
                    ),
                    "version": VERSION,
                },
                "terminal_result_sha256",
            )
            _durable_roundtrip(controller_root / "lifecycle.terminal_result.json", terminal)
            return terminal
    except BaseException as exc:
        failure = _self_hashed(
            {
                "category": type(exc).__name__,
                "message_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                "phase": phase,
                "schema": "maven_ere_g8_e1_lifecycle_failure_v1",
                "status": "terminal_no_retry",
                "version": VERSION,
            },
            "failure_sha256",
        )
        try:
            _atomic_write(controller_root / "lifecycle.failure.json", failure)
        except BaseException:
            pass
        raise


__all__ = [
    "AgentArtifact",
    "BlockExecution",
    "HippoArtifact",
    "ItemExecution",
    "MavenEreFormalControllerError",
    "OneShotRefusal",
    "PROMOTION_ALPHA",
    "RawArtifact",
    "evaluator_promoted",
    "execute_block",
    "load_family_labels",
    "real_domain_primary_passed",
    "run_formal_lifecycle",
    "score_block",
    "stable_hash",
]
