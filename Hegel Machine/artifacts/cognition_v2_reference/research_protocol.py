"""Seed-level research summaries without pass gates or pooled task pseudo-replicates."""

from __future__ import annotations

from math import isfinite
import random
from statistics import mean


def fresh_evidence_summary(episodes: list[dict], minimum_improvement: float) -> dict:
    counts = {"revision_comparisons": 0, "adverse_revision_comparisons": 0,
              "retained_comparisons": 0, "adverse_retained_comparisons": 0,
              "rejected_comparisons": 0, "beneficial_rejected_comparisons": 0,
              "revisions": 0, "revisions_with_fresh_check": 0,
              "reuse_checks": 0, "failed_reuse_checks": 0,
              "failed_reuse_checks_retained_at_episode_end": 0,
              "transfer_candidates_tested": 0, "direct_candidates_tested": 0,
              "abstract_candidates_tested": 0, "scope_restrictions": 0, "revocations": 0}
    transfer_counts_complete = True
    for episode in episodes:
        current_revision, checked = None, set()
        for index, action in enumerate(episode.get("actions", [])):
            kind, details = action["action"], action.get("details", {})
            if kind == "revise":
                counts["revisions"] += 1
                current_revision = index
            elif kind in {"prediction_check", "rollback"}:
                revised = details.get("revision_loss", details.get("revised_loss"))
                parent = details.get("parent_loss")
                if revised is None or parent is None:
                    continue
                counts["revision_comparisons"] += 1
                adverse = revised > parent + minimum_improvement
                beneficial = revised + minimum_improvement < parent
                counts["adverse_revision_comparisons"] += adverse
                if current_revision is not None:
                    checked.add(current_revision)
                if kind == "prediction_check":
                    counts["retained_comparisons"] += 1
                    counts["adverse_retained_comparisons"] += adverse
                else:
                    counts["rejected_comparisons"] += 1
                    counts["beneficial_rejected_comparisons"] += beneficial
                    current_revision = None
            elif kind == "reuse_check":
                counts["reuse_checks"] += 1
                failed = details["loss"] > minimum_improvement
                counts["failed_reuse_checks"] += failed
                if failed and details.get("expression") is not None:
                    counts["failed_reuse_checks_retained_at_episode_end"] += (
                        details["expression"] == episode.get("expression")
                        and details.get("memory_id") == episode.get("selected_memory_id"))
            elif kind == "retrieve":
                if "tested" not in details:
                    transfer_counts_complete = False
                    continue
                for tested in details["tested"]:
                    counts["transfer_candidates_tested"] += 1
                    counts["direct_candidates_tested"] += tested["operation"] == "reuse"
                    counts["abstract_candidates_tested"] += tested["operation"] == "abstract_reuse"
        counts["revisions_with_fresh_check"] += len(checked)
        counts["scope_restrictions"] += sum(event["action"] == "restricted" for event in episode.get("memory_events", []))
        counts["revocations"] += sum(event["action"] == "revoked" for event in episode.get("memory_events", []))

    def rate(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None

    return {**counts, "transfer_test_counts_complete": transfer_counts_complete,
            "observed_revision_failure_rate": rate("adverse_revision_comparisons", "revision_comparisons"),
            "false_extension_rate": rate("adverse_retained_comparisons", "retained_comparisons"),
            "false_rejection_rate": rate("beneficial_rejected_comparisons", "rejected_comparisons"),
            "revision_check_coverage": rate("revisions_with_fresh_check", "revisions"),
            "reuse_failure_rate": rate("failed_reuse_checks", "reuse_checks"),
            "definitions": {
                "evidence": "the same fresh observations used by the controller for retention or rollback; these rates describe evidence-action consistency, not independent rejection accuracy",
                "adverse": f"fresh revision prediction loss exceeds its parent by more than {minimum_improvement}",
                "false_extension_rate": "adverse fresh comparisons followed by retention / all fresh retained comparisons",
                "false_rejection_rate": "fresh beneficial revisions followed by rollback / all fresh rejected comparisons",
                "scope": "prediction-loss evidence only; excludes complexity tradeoffs and does not identify the globally best cognitive operation",
                "transfer_counts": "only explicitly logged evaluated candidates; a retrieved but unscored candidate does not count",
            }}


def final_decision_summary(episodes: list[dict]) -> dict:
    measured = [episode['evaluation']['final_counterfactual'] for episode in episodes
                if 'final_counterfactual' in episode.get('evaluation', {})]
    fields = ('revision_events', 'scored_revision_events', 'wrongly_accepted_events',
              'rollback_events', 'scored_rollback_events', 'wrongly_rejected_events',
              'missing_program_records', 'additional_evaluation_units')
    counts = {field: sum(row[field] for row in measured) for field in fields}
    return {**counts, 'episodes_measured': len(measured), 'attempted_episodes': len(episodes),
            'wrong_acceptance_rate': (counts['wrongly_accepted_events'] / counts['scored_revision_events']
                                      if counts['scored_revision_events'] else None),
            'wrong_rejection_rate': (counts['wrongly_rejected_events'] / counts['scored_rollback_events']
                                     if counts['scored_rollback_events'] else None),
            'wall_time_seconds': sum(row['wall_time_seconds'] for row in measured),
            'definitions': {
                'wrong_acceptance': 'on final evaluation, after loss exceeds before loss plus minimum_improvement',
                'wrong_rejection': 'on final evaluation, rejected loss is below restored loss minus minimum_improvement',
                'denominators': 'all scored revise events and all scored rollback events respectively; missing AST records are separate',
                'scope': 'recorded pairwise alternatives only, not the globally best operation; these outcomes never update the agent or memory',
            }}


def confidence_summary(values: list[float], *, seed: int = 0) -> dict:
    """Percentile bootstrap over independent suite seeds, not individual rows."""
    if not values:
        return {"n": 0, "mean": None, "interval_95": None}
    result = {"n": len(values), "mean": mean(values), "interval_95": None}
    if len(values) < 2:
        return result
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
    result["interval_95"] = [draws[49], draws[1949]]
    return result


def _episode_key(episode: dict, index: int) -> tuple:
    return episode.get("episode_index", index), episode["task_id"]


def paired_seed_effect(reference: dict, alternative: dict) -> dict:
    """Alternative minus reference on the exact same completed episodes."""
    reference_rows = {_episode_key(row, index): row for index, row in enumerate(reference["episodes"])}
    alternative_rows = {_episode_key(row, index): row for index, row in enumerate(alternative["episodes"])}
    differences: dict[str, list[float]] = {}
    denominators: dict[str, int] = {}
    for key, first in reference_rows.items():
        kind = first["evaluation"]["loss_kind"]
        denominators[kind] = denominators.get(kind, 0) + 1
        second = alternative_rows.get(key)
        if second is None:
            continue
        first_loss, second_loss = first["evaluation"].get("loss"), second["evaluation"].get("loss")
        if first_loss is None or second_loss is None:
            continue
        if not isfinite(first_loss) or not isfinite(second_loss):
            continue
        differences.setdefault(kind, []).append(second_loss - first_loss)
    return {
        kind: {"mean_difference": mean(differences[kind]) if differences.get(kind) else None,
               "paired_tasks": len(differences.get(kind, [])), "attempted_tasks": count}
        for kind, count in denominators.items()
    }


def knowledge_balance(full: dict, without: dict) -> dict:
    """Keep quality, observation, compute and time currencies separate."""
    def phase_cost(run, phases, field):
        return sum(row.get(field) or 0 for row in run["episodes"]
                   if row["evaluation"]["task_metadata"]["phase"] in phases)

    result = {"quality_without_minus_with": paired_seed_effect(full, without)}
    for field in ("observation_cost", "compute_cost", "wall_time_seconds"):
        result[f"transfer_{field}_saved"] = (
            phase_cost(without, {"transfer", "composition"}, field)
            - phase_cost(full, {"transfer", "composition"}, field)
        )
        result[f"acquisition_extra_{field}"] = (
            phase_cost(full, {"acquire"}, field) - phase_cost(without, {"acquire"}, field)
        )
        result[f"net_{field}_saved"] = (
            sum(row.get(field) or 0 for row in without["episodes"])
            - sum(row.get(field) or 0 for row in full["episodes"])
        )
    result["costs_complete"] = full.get("costs_complete", True) and without.get("costs_complete", True)
    return result


def summarize_research(comparisons: list[dict], seeds: list[int], partition: str) -> dict:
    systems = sorted({system for comparison in comparisons for system in comparison["runs"]})
    summary = {}
    seed_rows = []
    paired: dict[str, dict[str, list[float]]] = {}
    pairing_counts: dict[str, dict[str, dict[str, int]]] = {}
    balances = []
    for seed, comparison in zip(seeds, comparisons):
        runs = comparison["runs"]
        seed_rows.append({"seed": seed, "systems": {
            system: {key: run[key] for key in (
                "task_count", "completed_tasks", "metrics_by_loss", "rates",
                "total_observation_cost", "total_compute_cost", "total_wall_time_seconds",
                "costs_complete", "direct_reuse_tasks", "abstract_reuse_tasks", "memory_revisions",
                "fresh_evidence", "final_counterfactual", "llm_usage", "decision_summary",
            )} for system, run in runs.items()
        }})
        reference = runs.get("cognitive_agent")
        if reference:
            for system, run in runs.items():
                if system == "cognitive_agent":
                    continue
                for kind, effect in paired_seed_effect(reference, run).items():
                    if effect["mean_difference"] is not None:
                        paired.setdefault(system, {}).setdefault(kind, []).append(effect["mean_difference"])
                    count = pairing_counts.setdefault(system, {}).setdefault(
                        kind, {"paired_tasks": 0, "attempted_tasks": 0})
                    count["paired_tasks"] += effect["paired_tasks"]
                    count["attempted_tasks"] += effect["attempted_tasks"]
            if "without_knowledge" in runs:
                balances.append(knowledge_balance(reference, runs["without_knowledge"]))
    for system in systems:
        runs = [comparison["runs"][system] for comparison in comparisons if system in comparison["runs"]]
        kinds = sorted({kind for run in runs for kind in run["metrics_by_loss"]})
        summary[system] = {
            "test_loss": {kind: confidence_summary([
                run["metrics_by_loss"][kind]["mean_test_loss"] for run in runs
                if kind in run["metrics_by_loss"] and run["metrics_by_loss"][kind]["mean_test_loss"] is not None
            ]) for kind in kinds},
            "rates": {name: confidence_summary([run["rates"][name] for run in runs])
                      for name in ("prediction_coverage", "failure_rate", "refusal_rate", "revision_rate", "rollback_rate")},
            "costs": {name: confidence_summary([run[name] for run in runs])
                      for name in ("total_observation_cost", "total_compute_cost", "total_wall_time_seconds")},
            "risk_decisions": {name: confidence_summary([
                run["decision_summary"][name] for run in runs
                if run.get("decision_summary", {}).get(name) is not None
            ]) for name in ("mean_decision_loss", "coverage", "mean_interval_width", "worst_case_loss")},
            "fresh_evidence": {name: confidence_summary([
                run["fresh_evidence"][name] for run in runs if run["fresh_evidence"][name] is not None
            ]) for name in ("observed_revision_failure_rate", "false_extension_rate", "false_rejection_rate",
                           "revision_check_coverage", "reuse_failure_rate", "transfer_candidates_tested",
                           "direct_candidates_tested", "abstract_candidates_tested", "scope_restrictions", "revocations")},
            "fresh_evidence_counts": {name: sum(run["fresh_evidence"][name] for run in runs)
                                      for name in ("revision_comparisons", "adverse_revision_comparisons",
                                                   "retained_comparisons", "adverse_retained_comparisons",
                                                   "rejected_comparisons", "beneficial_rejected_comparisons",
                                                   "revisions", "revisions_with_fresh_check", "reuse_checks",
                                                   "failed_reuse_checks", "scope_restrictions", "revocations")},
            "final_counterfactual": {
                **{name: confidence_summary([run['final_counterfactual'][name] for run in runs
                                              if run['final_counterfactual'][name] is not None])
                   for name in ('wrong_acceptance_rate', 'wrong_rejection_rate')},
                'counts': {name: sum(run['final_counterfactual'][name] for run in runs)
                           for name in ('revision_events', 'scored_revision_events', 'wrongly_accepted_events',
                                        'rollback_events', 'scored_rollback_events', 'wrongly_rejected_events',
                                        'missing_program_records', 'episodes_measured', 'attempted_episodes',
                                        'additional_evaluation_units')},
                'evaluation_wall_time_seconds': sum(run['final_counterfactual']['wall_time_seconds'] for run in runs),
            },
            "transfer_test_counts_complete": all(run["fresh_evidence"]["transfer_test_counts_complete"] for run in runs),
            "llm_usage": {name: (
                confidence_summary([run["llm_usage"][name] for run in runs])
                if all(run["llm_usage"][name] is not None for run in runs) else None
            ) for name in ("calls", "input_tokens", "output_tokens", "total_tokens")},
            "all_costs_reported": all(run["costs_complete"] for run in runs),
        }
    paired_summary = {
        system: {kind: {**confidence_summary(values), **pairing_counts[system][kind]}
                 for kind, values in by_kind.items()}
        for system, by_kind in paired.items()
    }
    balance_summary = {}
    if balances:
        numeric_names = [name for name, value in balances[0].items()
                         if isinstance(value, (int, float)) and not isinstance(value, bool)]
        balance_summary = {name: confidence_summary([balance[name] for balance in balances])
                           for name in numeric_names}
        balance_summary["all_costs_reported"] = all(balance["costs_complete"] for balance in balances)
    return {
        "experiment": "cognition_multi_seed_research", "partition": partition, "seeds": seeds,
        "systems": summary, "paired_loss_alternative_minus_cognitive": paired_summary,
        "knowledge_resource_balance": balance_summary, "seed_results": seed_rows,
        "inference": {
            "unit": "independent_suite_seed", "interval": "95 percent percentile bootstrap; 2000 resamples",
            "one_seed_interval": "unavailable", "paired_sign": "positive favors cognitive_agent",
            "quality_and_costs": "reported separately; no conversion of prediction loss into compute units",
            "final_counterfactual": {
                "wrong_acceptance": "final after-program loss > before-program loss + minimum_improvement",
                "wrong_rejection": "final rejected-program loss < restored-program loss - minimum_improvement",
                "denominators": "scored revise and rollback events respectively; missing AST records remain separate",
                "scope": "only alternatives actually involved in recorded decisions; final labels never enter learning",
            },
        },
        "limitations": [
            "Both partitions use public synthetic generators; confirmation reserves combinations, intervals and history lengths.",
            "A confirmation run is exploratory if its results are used to revise the system; a later untouched confirmation is then required.",
            "Intervals resample suite seeds and can be coarse or unstable with few seeds.",
            "Paired losses include only episodes completed by both systems; coverage and failures remain in the reported denominator.",
            "Simulator queries are reproducible; generated-query noise is assigned on first query and replayed for identical trajectories.",
        ],
    }
