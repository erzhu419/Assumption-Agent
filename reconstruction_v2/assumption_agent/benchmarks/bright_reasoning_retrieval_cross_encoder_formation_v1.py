"""Reproduce the historically informed P9 candidate on consumed BRIGHT TRAIN45.

This stage is deliberately post-terminal and non-prospective.  It binds one
candidate mechanism only: relation/mechanism cross-encoder ranking fused by
equal-weight RRF(k=60) with the frozen RAW and HippoRAG rankings.  It does not
touch any unconsumed reserve item.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_measurement_v1 as reserve
from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v1 as base
from replication_runtime.bright_cross_encoder_v1 import contract as cross_contract
from replication_runtime.bright_cross_encoder_v1 import worker as cross_worker
from replication_runtime.bright_official_hipporag_v1 import contract as hippo_contract
from replication_runtime.bright_query_generator_v1 import contract as qwen_contract


VERSION = "bright_reasoning_retrieval_cross_encoder_formation_v1"
P9 = "P9_RELATION_MECHANISM_CE_RAW_HIPPORAG_RRF"
ROOT_RELATIVE = Path("artifacts/bright_reasoning_retrieval_cross_encoder_formation_v1")
RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_cross_encoder_formation_result_v1.json"
)
MODEL_RELATIVE = Path("artifacts/bright_cross_encoder_runtime_v1/model")
ASSET_MANIFEST_RELATIVE = Path("manifests/bright_cross_encoder_runtime_asset_v1.json")
RUNTIME_FREEZE_RELATIVE = Path(
    "manifests/bright_cross_encoder_runtime_implementation_freeze_v1.json"
)

ASSET_MANIFEST_FILE_SHA256 = (
    "cbb90ef21571b94e41e3fcb501228dbd130edcba87dcd40f95f15d7e805c133c"
)
ASSET_SELF_SHA256 = (
    "56c550fd1224096dad64ebf7ed5ae8552d55ee8a1216376f39b1dc11be32ff43"
)
RUNTIME_FREEZE_FILE_SHA256 = (
    "8d8d8df7ebecdeb619c5f10da6b719ebc5461a932af5e85756935749414f44d8"
)
RUNTIME_FREEZE_SELF_SHA256 = (
    "596adcb76cab5c98b8420c1e41d72daee681cd66df302136b5f490dc9b32d0ec"
)
ACQUISITION_RESULT_FILE_SHA256 = (
    "7e25eb23cbe1741d64d7f367d7b1922fbdb1f6bde682e7159b251c7d6f6e151a"
)
PREPARE_RESULT_FILE_SHA256 = (
    "578155cfb89ba3d9b09b3a198a922100e84bbc8122d06a1dc82b98d875a0bbfe"
)
ACTION_RESULT_FILE_SHA256 = (
    "2345f9939ad9721b836ec52110a04451011208f8d8e3ee33359a0aae010bc651"
)
FINAL_RESULT_FILE_SHA256 = (
    "ff1e22fd7d151321dbce889341ecf0424c6f66dc066c0832156e498cdb3b9d4c"
)


class BrightCrossEncoderFormationError(RuntimeError):
    """The consumed-cohort P9 formation contract failed closed."""


def _read_json(path: Path, field: str, *, canonical: bool) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BrightCrossEncoderFormationError(f"{field} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightCrossEncoderFormationError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise BrightCrossEncoderFormationError(f"{field} root drifted")
    if canonical and base.canonical_json_bytes(value) != raw:
        raise BrightCrossEncoderFormationError(f"{field} is not canonical")
    return value


def _verify_public_result(
    path: Path,
    *,
    file_sha256: str,
    schema: str,
    status: str,
) -> dict[str, Any]:
    if base.file_sha256(path) != file_sha256:
        raise BrightCrossEncoderFormationError(f"{path.name} file binding drifted")
    value = _read_json(path, path.name, canonical=True)
    if value.get("schema") != schema or value.get("status") != status:
        raise BrightCrossEncoderFormationError(f"{path.name} identity drifted")
    base.verify_self_hash(value, "result_sha256")
    return value


def _verify_runtime(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    reconstruction_root = project_root / "reconstruction_v2"
    asset_path = reconstruction_root / ASSET_MANIFEST_RELATIVE
    freeze_path = reconstruction_root / RUNTIME_FREEZE_RELATIVE
    if base.file_sha256(asset_path) != ASSET_MANIFEST_FILE_SHA256:
        raise BrightCrossEncoderFormationError("cross-encoder asset manifest drifted")
    if base.file_sha256(freeze_path) != RUNTIME_FREEZE_FILE_SHA256:
        raise BrightCrossEncoderFormationError("cross-encoder runtime freeze drifted")
    asset = _read_json(asset_path, "cross-encoder asset manifest", canonical=False)
    freeze = _read_json(freeze_path, "cross-encoder runtime freeze", canonical=False)
    if base.verify_self_hash(asset, "asset_sha256") != ASSET_SELF_SHA256:
        raise BrightCrossEncoderFormationError("cross-encoder asset self-hash drifted")
    if base.verify_self_hash(freeze, "self_sha256") != RUNTIME_FREEZE_SELF_SHA256:
        raise BrightCrossEncoderFormationError("cross-encoder freeze self-hash drifted")
    if freeze.get("schema") != "bright_cross_encoder_runtime_implementation_freeze_v1":
        raise BrightCrossEncoderFormationError("cross-encoder freeze schema drifted")
    bindings = freeze.get("implementation_bindings")
    if not isinstance(bindings, list) or len(bindings) != 4:
        raise BrightCrossEncoderFormationError("runtime implementation bindings drifted")
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != {"relative_path", "sha256"}:
            raise BrightCrossEncoderFormationError("runtime implementation binding shape drifted")
        path = project_root / "reconstruction_v2" / str(binding["relative_path"])
        if base.file_sha256(path) != binding["sha256"]:
            raise BrightCrossEncoderFormationError("runtime implementation file drifted")
    model_root = project_root / "reconstruction_v2" / MODEL_RELATIVE
    model_files = asset.get("local_binding", {}).get("required_files")
    if not isinstance(model_files, list) or len(model_files) != 6:
        raise BrightCrossEncoderFormationError("model asset binding drifted")
    expected_names: set[str] = set()
    for binding in model_files:
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256", "size"}:
            raise BrightCrossEncoderFormationError("model file binding shape drifted")
        name = str(binding["path"])
        expected_names.add(name)
        path = model_root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != binding["size"]
            or base.file_sha256(path) != binding["sha256"]
        ):
            raise BrightCrossEncoderFormationError("model file binding drifted")
    actual_names = {path.name for path in model_root.iterdir() if path.is_file()}
    if actual_names != expected_names or any(path.is_symlink() for path in model_root.iterdir()):
        raise BrightCrossEncoderFormationError("model root file set drifted")
    return asset, freeze


def p9_rows(
    *,
    candidate_rows: Sequence[int],
    cross_encoder_ranked_ordinals: Sequence[int],
    raw_rows: Sequence[int],
    hipporag_rows: Sequence[int],
) -> tuple[int, ...]:
    """Return the fixed equal-weight RRF(k=60) P9 top ten."""

    pool = tuple(candidate_rows)
    ordinals = tuple(cross_encoder_ranked_ordinals)
    raw = tuple(raw_rows)
    hippo = tuple(hipporag_rows)
    if (
        len(pool) != base.core.POOL_SIZE
        or len(set(pool)) != base.core.POOL_SIZE
        or any(isinstance(value, bool) or not isinstance(value, int) for value in pool)
        or len(ordinals) != base.core.POOL_SIZE
        or set(ordinals) != set(range(base.core.POOL_SIZE))
        or len(raw) != base.core.TOP_K
        or len(set(raw)) != base.core.TOP_K
        or len(hippo) != base.core.TOP_K
        or len(set(hippo)) != base.core.TOP_K
        or not set(raw) <= set(pool)
        or not set(hippo) <= set(pool)
    ):
        raise BrightCrossEncoderFormationError("P9 input ranking drifted")
    cross_rows = tuple(pool[index] for index in ordinals)
    return base.core._rrf_ranking((cross_rows, raw, hippo), pool)[: base.core.TOP_K]


def _paired(left: Sequence[int], right: Sequence[int]) -> dict[str, int]:
    if len(left) != len(right):
        raise BrightCrossEncoderFormationError("paired score lengths drifted")
    return {
        "gain": sum(a > b for a, b in zip(left, right)),
        "harm": sum(a < b for a, b in zip(left, right)),
        "tie": sum(a == b for a, b in zip(left, right)),
    }


def _source_state(project_root: Path) -> dict[str, Any]:
    acquisition = _verify_public_result(
        project_root / "reconstruction_v2" / reserve.ACQUISITION_RESULT_RELATIVE,
        file_sha256=ACQUISITION_RESULT_FILE_SHA256,
        schema="bright_reasoning_retrieval_reserve_acquisition_v1_result",
        status="fresh_RESERVE_R_search_acquired_labels_sealed",
    )
    prepare = _verify_public_result(
        project_root / "reconstruction_v2" / reserve.PREPARE_RESULT_RELATIVE,
        file_sha256=PREPARE_RESULT_FILE_SHA256,
        schema=reserve.PREPARE_SCHEMA,
        status="reserve_action_intents_prepared_labels_sealed",
    )
    actions = _verify_public_result(
        project_root / "reconstruction_v2" / reserve.ACTION_RESULT_RELATIVE,
        file_sha256=ACTION_RESULT_FILE_SHA256,
        schema=reserve.ACTION_RESULT_SCHEMA,
        status="reserve_three_arm_actions_complete_labels_sealed",
    )
    final = _verify_public_result(
        project_root / "reconstruction_v2" / reserve.FINAL_RESULT_RELATIVE,
        file_sha256=FINAL_RESULT_FILE_SHA256,
        schema=reserve.FINAL_SCHEMA,
        status="fresh_RESERVE_measurement_complete",
    )
    if (
        prepare.get("formal_binding", {}).get("acquisition_result_sha256")
        != acquisition["result_sha256"]
        or actions.get("formal_binding", {}).get("prepare_result_sha256")
        != prepare["result_sha256"]
        or final.get("formal_binding", {}).get("action_result_sha256")
        != actions["result_sha256"]
        or final.get("item_count") != 45
    ):
        raise BrightCrossEncoderFormationError("terminal TRAIN45 chain drifted")
    return {
        "acquisition": acquisition,
        "actions": actions,
        "final": final,
        "prepare": prepare,
    }


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    reconstruction_root = project_root / "reconstruction_v2"
    result_path = reconstruction_root / RESULT_RELATIVE
    root = reconstruction_root / ROOT_RELATIVE
    if root.exists() or result_path.exists():
        raise BrightCrossEncoderFormationError("P9 formation is one-shot")
    source = _source_state(project_root)
    asset, freeze = _verify_runtime(project_root)
    preconditions = {"acquisition": source["acquisition"]}
    items = reserve._load_view(reconstruction_root, preconditions)
    if len(items) != 45 or Counter(item.family for item in items) != Counter(
        {family: 15 for family in base.core.FAMILY_ORDER}
    ):
        raise BrightCrossEncoderFormationError("consumed TRAIN45 identity drifted")
    prepare_result, intent = reserve._load_prepare(reconstruction_root)
    if prepare_result["result_sha256"] != source["prepare"]["result_sha256"]:
        raise BrightCrossEncoderFormationError("prepare reload drifted")
    intent_rows = intent.get("items")
    if not isinstance(intent_rows, list) or len(intent_rows) != len(items):
        raise BrightCrossEncoderFormationError("intent rows drifted")

    qwen_path = reconstruction_root / reserve.ROOT_RELATIVE / "qwen.output.json"
    local_path = reconstruction_root / reserve.ROOT_RELATIVE / "local.action.json"
    action_path = reconstruction_root / reserve.ROOT_RELATIVE / "three_arm.action.json"
    if (
        base.file_sha256(qwen_path)
        != source["prepare"]["private_bindings"]["qwen"]["output_file_sha256"]
        or base.file_sha256(local_path)
        != source["prepare"]["private_bindings"]["local_action_file_sha256"]
        or base.file_sha256(action_path)
        != source["actions"]["private_bindings"]["three_arm_action_file_sha256"]
    ):
        raise BrightCrossEncoderFormationError("terminal private file binding drifted")
    qwen = qwen_contract.parse_output(qwen_path.read_bytes())
    local = _read_json(local_path, "local action", canonical=True)
    local_rows = base._validate_action_pack(local, items)
    action = _read_json(action_path, "three-arm action", canonical=True)
    if (
        action.get("schema") != reserve.THREE_ARM_SCHEMA
        or base.verify_self_hash(action, "pack_sha256")
        != source["actions"]["private_bindings"]["three_arm_action_pack_sha256"]
    ):
        raise BrightCrossEncoderFormationError("terminal action pack drifted")
    action_rows = action.get("items")
    if not isinstance(action_rows, list) or len(action_rows) != len(items):
        raise BrightCrossEncoderFormationError("terminal action rows drifted")

    root.mkdir(mode=0o700, parents=True)
    marker = base.self_hashed(
        {
            "asset_self_sha256": asset["asset_sha256"],
            "candidate": P9,
            "consumed_final_result_sha256": source["final"]["result_sha256"],
            "runtime_freeze_self_sha256": freeze["self_sha256"],
            "schema": f"{VERSION}_attempt",
        },
        "attempt_sha256",
    )
    base._write_json(root / "attempt.marker", marker)

    cross_items: list[dict[str, Any]] = []
    document_id_rows: list[tuple[str, ...]] = []
    for position, (item, qwen_row, local_row, intent_row, action_row) in enumerate(
        zip(items, qwen["items"], local_rows, intent_rows, action_rows)
    ):
        if (
            qwen_row.get("ordinal") != position
            or qwen_row.get("generation_valid") is not True
            or len(qwen_row.get("expansions", ())) != 4
            or not isinstance(intent_row, Mapping)
            or intent_row.get("ordinal") != position
            or not isinstance(action_row, Mapping)
            or action_row.get("ordinal") != position
            or action_row.get("item_commitment_sha256") != item.commitment
            or list(local_row["raw_rows"]) != list(intent_row.get("RAW_rows", ()))
            or list(local_row["raw_document_ids"])
            != list(action_row.get("RAW_document_ids", ()))
        ):
            raise BrightCrossEncoderFormationError("consumed item identity drifted")
        hippo_path = (
            reconstruction_root
            / reserve.ROOT_RELATIVE
            / "hipporag"
            / f"item_{position:03d}"
            / "input.json"
        )
        if base.file_sha256(hippo_path) != intent_row.get("HippoRAG_input_file_sha256"):
            raise BrightCrossEncoderFormationError("HippoRAG input binding drifted")
        hippo_raw = hippo_path.read_bytes()
        try:
            hippo_input = json.loads(hippo_raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BrightCrossEncoderFormationError("HippoRAG input is invalid") from exc
        if (
            not isinstance(hippo_input, Mapping)
            or set(hippo_input) != {"documents", "query", "schema"}
            or hippo_input.get("schema") != hippo_contract.INPUT_SCHEMA
            or hippo_contract.canonical_json_bytes(hippo_input) != hippo_raw
        ):
            raise BrightCrossEncoderFormationError("HippoRAG input envelope drifted")
        query, documents = hippo_contract.validate_input(
            hippo_input.get("query"), hippo_input.get("documents")
        )
        if query != item.query:
            raise BrightCrossEncoderFormationError("HippoRAG query identity drifted")
        candidate_rows = tuple(local_row["candidate_rows"])
        candidate_ids = tuple(local_row["candidate_document_ids"])
        if (
            len(candidate_rows) != base.core.POOL_SIZE
            or len(set(candidate_rows)) != base.core.POOL_SIZE
            or len(candidate_ids) != base.core.POOL_SIZE
            or len(set(candidate_ids)) != base.core.POOL_SIZE
            or list(candidate_rows) != list(intent_row.get("candidate_rows", ()))
        ):
            raise BrightCrossEncoderFormationError("candidate identity drifted")
        document_id_rows.append(candidate_ids)
        expansions = qwen_row["expansions"]
        cross_items.append(
            {
                "documents": [
                    {"content": document.content, "ordinal": document.ordinal}
                    for document in documents
                ],
                "mechanism_query": expansions[2],
                "ordinal": position,
                "relation_query": expansions[1],
            }
        )
    cross_input = cross_contract.input_payload(cross_items)
    cross_input_path = root / "cross_encoder.input.json"
    base._write_json(cross_input_path, cross_input)
    cross_output_path = root / "cross_encoder.output.json"
    cross_receipt = cross_worker.run(
        input_path=cross_input_path,
        output_path=cross_output_path,
        model_root=reconstruction_root / MODEL_RELATIVE,
    )
    cross_output = cross_contract.parse_output(cross_output_path.read_bytes())

    p9_action_rows: list[dict[str, Any]] = []
    ce_document_ids: list[tuple[str, ...]] = []
    for position, (item, local_row, action_row, cross_row, candidate_ids) in enumerate(
        zip(items, local_rows, action_rows, cross_output["items"], document_id_rows)
    ):
        hippo_rows = tuple(action_row.get("HippoRAG", {}).get("top_rows", ()))
        candidate_rows = tuple(local_row["candidate_rows"])
        p9 = p9_rows(
            candidate_rows=candidate_rows,
            cross_encoder_ranked_ordinals=cross_row["ranked_ordinals"],
            raw_rows=tuple(local_row["raw_rows"]),
            hipporag_rows=hippo_rows,
        )
        row_to_id = dict(zip(candidate_rows, candidate_ids))
        if len(row_to_id) != base.core.POOL_SIZE:
            raise BrightCrossEncoderFormationError("candidate row-to-ID map drifted")
        p9_ids = tuple(row_to_id[row] for row in p9)
        ce_ids = tuple(
            candidate_ids[index] for index in cross_row["ranked_ordinals"][: base.core.TOP_K]
        )
        ce_document_ids.append(ce_ids)
        p9_action_rows.append(
            {
                "P9_document_ids": list(p9_ids),
                "P9_rows": list(p9),
                "cross_encoder_mean_logit_quantized": list(
                    cross_row["mean_logit_quantized"]
                ),
                "cross_encoder_ranked_ordinals": list(cross_row["ranked_ordinals"]),
                "family": item.family,
                "item_commitment_sha256": item.commitment,
                "ordinal": position,
            }
        )
    p9_actions = base.self_hashed(
        {
            "candidate": P9,
            "formation": {
                "cross_encoder_queries": ["relation_query", "mechanism_query"],
                "fusion": "equal_weight_RRF",
                "fusion_rankings": [
                    "cross_encoder_full32",
                    "RAW_top10",
                    "HippoRAG_top10",
                ],
                "rrf_k": base.core.RRF_K,
                "top_k": base.core.TOP_K,
            },
            "item_count": len(items),
            "items": p9_action_rows,
            "schema": f"{VERSION}_action",
        },
        "pack_sha256",
    )
    p9_action_path = root / "p9.action.json"
    base._write_json(p9_action_path, p9_actions)

    labels = reserve._load_labels(reconstruction_root, preconditions, items)
    arm_scores: dict[str, list[int]] = {
        "P9": [],
        "CrossEncoder_RM": [],
        "P6": [],
        "HippoRAG": [],
        "RAW": [],
    }
    score_rows: list[dict[str, Any]] = []
    for item, gold, p9_action, ce_ids, old_action in zip(
        items, labels, p9_action_rows, ce_document_ids, action_rows
    ):
        values = {
            "P9": base.core.integer_ndcg_at_10(p9_action["P9_document_ids"], gold),
            "CrossEncoder_RM": base.core.integer_ndcg_at_10(ce_ids, gold),
            "P6": base.core.integer_ndcg_at_10(old_action["Agent_document_ids"], gold),
            "HippoRAG": base.core.integer_ndcg_at_10(
                old_action["HippoRAG"]["document_ids"], gold
            ),
            "RAW": base.core.integer_ndcg_at_10(old_action["RAW_document_ids"], gold),
        }
        for arm, value in values.items():
            arm_scores[arm].append(value)
        score_rows.append(
            {
                "arm_scores": values,
                "family": item.family,
                "item_commitment_sha256": item.commitment,
                "ordinal": item.ordinal,
            }
        )
    scored = base.self_hashed(
        {
            "item_count": len(items),
            "items": score_rows,
            "schema": f"{VERSION}_scored",
        },
        "pack_sha256",
    )
    scored_path = root / "scored.json"
    base._write_json(scored_path, scored)
    aggregates = base._family_arm_aggregates(items, arm_scores)
    family_delta_raw = {
        family: aggregates["P9"]["family_sum_integer_ndcg"][family]
        - aggregates["RAW"]["family_sum_integer_ndcg"][family]
        for family in base.core.FAMILY_ORDER
    }
    family_delta_hippo = {
        family: aggregates["P9"]["family_sum_integer_ndcg"][family]
        - aggregates["HippoRAG"]["family_sum_integer_ndcg"][family]
        for family in base.core.FAMILY_ORDER
    }
    result = base.self_hashed(
        {
            "arm_aggregates": aggregates,
            "candidate": P9,
            "claim_boundary": {
                "candidate_selected_after_consumed_TRAIN45_label_access": True,
                "external_network_call_count": 0,
                "formal_remaining_RESERVE_content_read_count": 0,
                "formal_remaining_RESERVE_label_open_count": 0,
                "L5_claim": False,
                "population_inference": False,
                "prospective_confirmation_claim": False,
                "search_multiplicity_formally_accounted": False,
            },
            "descriptive_evidence": {
                "P9_minus_HippoRAG_family_sum_integer_ndcg": family_delta_hippo,
                "P9_minus_HippoRAG_positive_in_all_three_families": all(
                    value > 0 for value in family_delta_hippo.values()
                ),
                "P9_minus_HippoRAG_sum_integer_ndcg": sum(arm_scores["P9"])
                - sum(arm_scores["HippoRAG"]),
                "P9_minus_RAW_family_sum_integer_ndcg": family_delta_raw,
                "P9_minus_RAW_positive_in_all_three_families": all(
                    value > 0 for value in family_delta_raw.values()
                ),
                "P9_minus_RAW_sum_integer_ndcg": sum(arm_scores["P9"])
                - sum(arm_scores["RAW"]),
            },
            "formation_contract": p9_actions["formation"],
            "formal_binding": {
                "asset_self_sha256": asset["asset_sha256"],
                "consumed_final_result_sha256": source["final"]["result_sha256"],
                "implementation_commit": base._git_head(project_root),
                "runtime_freeze_self_sha256": freeze["self_sha256"],
            },
            "item_count": len(items),
            "paired": {
                "P9_minus_HippoRAG": _paired(
                    arm_scores["P9"], arm_scores["HippoRAG"]
                ),
                "P9_minus_RAW": _paired(arm_scores["P9"], arm_scores["RAW"]),
            },
            "private_bindings": {
                "attempt_marker_file_sha256": base.file_sha256(root / "attempt.marker"),
                "cross_encoder_input_file_sha256": base.file_sha256(cross_input_path),
                "cross_encoder_output_file_sha256": base.file_sha256(cross_output_path),
                "cross_encoder_pair_count": cross_receipt["pair_count"],
                "p9_action_file_sha256": base.file_sha256(p9_action_path),
                "p9_action_pack_sha256": p9_actions["pack_sha256"],
                "scored_file_sha256": base.file_sha256(scored_path),
                "scored_pack_sha256": scored["pack_sha256"],
            },
            "schema": f"{VERSION}_result",
            "status": "consumed_TRAIN45_postterminal_candidate_formation_complete",
        },
        "result_sha256",
    )
    base._write_json(result_path, result, mode=0o644)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run(arguments.project_root)
    print(
        json.dumps(
            {
                "result_sha256": result["result_sha256"],
                "status": result["status"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


