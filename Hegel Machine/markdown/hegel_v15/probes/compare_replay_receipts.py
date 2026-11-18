"""Compare saved D2 first-dispatch telemetry; no Agent or physical experiment runs.

This is a portable packaging of the existing 24-pair receipt comparison. It reads
the exported first nonempty dispatch, normalizes branch IDs by source concept,
and reports recorded state differences. It does not prove that every prediction
function differs or attribute task outcomes to a particular replay code path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RECEIPT = Path(
    "receipts/v14_continuous_object_and_fair_comparison_20261118/"
    "deliverable_2_unconfounded_noH_source_paired_study_20261118.json"
)
LIST_FIELDS = (
    "support_invalidated",
    "empirically_excluded_on_context",
    "empirically_excluded_bound_instance_only",
    "not_scheduled_by_router",
    "scope_not_established",
)
MATCH_FIELDS = {
    "matching_task_id_pairs": "task_id",
    "matching_episode_uid_pairs": "episode_uid",
    "matching_episode_tag_pairs": "episode_tag",
    "matching_target_role_pairs": "target_regime_role",
}


def normalize_dispatch(telemetry: dict) -> dict:
    """Use recorded source links, not branch:/archive_src: spelling, as identity."""
    identity = {}
    levels = {}
    for item in telemetry["branch_identity_map"]:
        sources = item["source_concept_ids"]
        if len(sources) != 1:
            raise ValueError("This V14 comparison requires one source concept per branch.")
        source = sources[0]
        if source in levels:
            raise ValueError(f"Multiple branch versions for source {source}; explicit matching needed.")
        identity[item["branch_id"]] = source
        levels[source] = item.get("scope_claim_level")
    normalized = {
        field: sorted(identity[branch_id] for branch_id in telemetry[field])
        for field in LIST_FIELDS
    }
    normalized["scope_claim_level"] = dict(sorted(levels.items()))
    return normalized


def compare(receipt: dict) -> dict:
    pairs = []
    matching = {key: 0 for key in MATCH_FIELDS}
    matching["matching_observable_regime_features_pairs"] = 0
    difference_counts = {field: 0 for field in (*LIST_FIELDS, "scope_claim_level")}
    for chain in receipt["unexposed_evaluation_chains"]:
        arms = chain["arms_target_results"]
        r_episodes = arms["R_lazy_noH"]
        k_episodes = arms["K_eager_noH"]
        if len(r_episodes) != len(k_episodes):
            raise ValueError(f"Different target counts in {chain['chain_id']}.")
        for r_episode, k_episode in zip(r_episodes, k_episodes):
            r_dispatch = r_episode["branch_dispatch_telemetry"]
            k_dispatch = k_episode["branch_dispatch_telemetry"]
            if not r_dispatch or not k_dispatch:
                raise ValueError("A paired target is missing first-dispatch telemetry.")
            for count_key, episode_key in MATCH_FIELDS.items():
                matching[count_key] += r_episode[episode_key] == k_episode[episode_key]
            same_features = (
                r_dispatch["observable_regime_features"]
                == k_dispatch["observable_regime_features"]
            )
            matching["matching_observable_regime_features_pairs"] += same_features
            r_state = normalize_dispatch(r_dispatch)
            k_state = normalize_dispatch(k_dispatch)
            differences = {}
            for field in difference_counts:
                if r_state[field] != k_state[field]:
                    differences[field] = {"R": r_state[field], "K": k_state[field]}
                    difference_counts[field] += 1
            pairs.append({
                "chain": chain["chain_id"],
                "target": r_episode["target_regime_role"],
                "identical_observable_features": same_features,
                "differences": differences,
                "R_pass": r_episode["strict_pass"],
                "K_pass": k_episode["strict_pass"],
            })
    matching["stage"] = (
        "first nonempty branch_dispatch_telemetry from first "
        "apply_world_structure_commitments action (script lines172-178)"
    )
    matching["construction"] = (
        "both derived from deepcopy same reconciled source library; K retains active "
        "branches, R clears then reconstructs; target _freeze_branch_writes=True; "
        "per-target deepcopy with only R cache persisted"
    )
    return {
        "pair_count": len(pairs),
        "pairs": pairs,
        "matching_scope": matching,
        "difference_counts": difference_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path,
                        help="Directory containing the extracted V14 review bundle.")
    parser.add_argument("--output", type=Path,
                        default=Path("scope_replay_receipt_comparison.json"),
                        help="Output JSON path; defaults to the current directory.")
    args = parser.parse_args()
    source = args.bundle.resolve() / RECEIPT
    raw = source.read_bytes()
    result = compare(json.loads(raw))
    result["verification_mode"] = "comparison_of_saved_first_dispatch_receipts_only"
    result["full_physical_rerun"] = False
    result["source_receipt"] = RECEIPT.as_posix()
    result["source_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    result["limitations"] = [
        "The stage and source-library construction descriptions are based on the "
        "audited V14 exporter code; this script does not rerun the source or target agents.",
        "Recorded scope, support and routing differences do not establish that every "
        "prediction function differs or identify the cause of an individual task outcome.",
        "Pass flags are copied from the saved receipt; scientific measurements are not rerun.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "pair_count": result["pair_count"],
        "matching_scope": result["matching_scope"],
        "difference_counts": result["difference_counts"],
        "full_physical_rerun": False,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
