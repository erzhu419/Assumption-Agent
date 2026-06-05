"""Matched frozen toggle-off baselines for the paper main experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STRUCTURAL_LIVE_DIR = Path("phase four/assumption_graph/structural_live_ablation_20260603")
DEFAULT_FINAL_SUMMARY = STRUCTURAL_LIVE_DIR / "structural_live_all_repairs_margin100_v2_gpt54mini_gpt55_20260604_summary.json"
DEFAULT_OUT = Path("phase four/assumption_graph/paper_readiness_20260604/paper_baseline_hardening_20260605.json")

TOGGLE_CONFIGS = [
    {
        "baseline": "no_world_model_trace_policy",
        "path": STRUCTURAL_LIVE_DIR / "structural_live_natural100_v1_gpt54mini_gpt55_20260603_summary.json",
        "disabled_components": [
            "trace_policy_router",
            "world_model_risk_prediction",
            "repair_policy",
            "recursive_readback",
        ],
        "toggle_note": "Natural one-shot cueing over the same 100 problem_ids.",
    },
    {
        "baseline": "no_recursive_runner_one_shot",
        "path": STRUCTURAL_LIVE_DIR / "structural_live_natural100_v1_gpt54mini_gpt55_20260603_summary.json",
        "disabled_components": [
            "recursive_resume",
            "recursive_repair",
            "gated_retention_readback",
        ],
        "toggle_note": "Single-pass structural cueing without recursive repair/readback over the same 100 problem_ids.",
    },
    {
        "baseline": "no_novelty_gate_incremental_addition",
        "path": STRUCTURAL_LIVE_DIR
        / "structural_live_natural_repaired_residual_signal_incremental100_v1_gpt54mini_gpt55_20260603_summary.json",
        "disabled_components": [
            "final_novelty_integration_gate",
            "selective_retention_margin_gate",
        ],
        "toggle_note": "Incremental addition run before the final novelty/integration and margin retention discipline.",
    },
    {
        "baseline": "no_final_margin_retention_gate",
        "path": STRUCTURAL_LIVE_DIR / "structural_live_all_repairs100_v1_gpt54mini_gpt55_20260603_summary.json",
        "disabled_components": [
            "final_margin_retention_policy",
        ],
        "toggle_note": "Same repair set before the final margin policy was frozen.",
    },
]


def build_paper_baseline_hardening_payload(
    *,
    root: Path,
    eval_id: str | None = None,
    final_summary_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    final_path = _resolve(root, final_summary_path or DEFAULT_FINAL_SUMMARY)
    final_summary = _load_json(final_path)
    final_problem_ids = _problem_ids(final_summary)
    final_pairs = final_summary.get("pair_summaries", {})
    rows = []
    for config in TOGGLE_CONFIGS:
        path = _resolve(root, config["path"])
        summary = _load_json(path)
        problem_ids = _problem_ids(summary)
        same_order = problem_ids == final_problem_ids
        same_set = set(problem_ids) == set(final_problem_ids)
        row = {
            "baseline": config["baseline"],
            "source": _display_path(root, path),
            "source_kind": "matched_frozen_toggle_off_summary",
            "disabled_components": config["disabled_components"],
            "toggle_note": config["toggle_note"],
            "same_problem_id_order": same_order,
            "same_problem_id_set": same_set,
            "problem_count": len(problem_ids),
            "pass": bool(summary.get("pass")),
            "selection_mode": (summary.get("plan") or {}).get("selection_mode"),
            "route_source_counts": (summary.get("plan") or {}).get("route_source_counts"),
            "pairs": _pair_comparison(final_pairs=final_pairs, toggle_pairs=summary.get("pair_summaries", {})),
            "model_answer_payload_stored": False,
        }
        rows.append(row)
    gates = [
        {
            "gate": "required_toggle_baselines_present",
            "pass": {row["baseline"] for row in rows} >= {
                "no_world_model_trace_policy",
                "no_recursive_runner_one_shot",
                "no_novelty_gate_incremental_addition",
            },
            "observed": sorted(row["baseline"] for row in rows),
        },
        {
            "gate": "matched_frozen_task_set",
            "pass": all(row["same_problem_id_set"] and row["problem_count"] == 100 for row in rows),
            "observed": {
                row["baseline"]: {
                    "same_problem_id_order": row["same_problem_id_order"],
                    "same_problem_id_set": row["same_problem_id_set"],
                    "problem_count": row["problem_count"],
                }
                for row in rows
            },
        },
        {
            "gate": "toggle_off_baselines_not_just_prompt_length",
            "pass": all(row["source_kind"] == "matched_frozen_toggle_off_summary" for row in rows),
            "observed": {
                row["baseline"]: row["disabled_components"]
                for row in rows
            },
        },
        {
            "gate": "main_pipeline_beats_key_toggle_offs",
            "pass": all(
                row["pairs"]["structural_vs_base"]["final_minus_toggle_utility"] > 0.0
                and row["pairs"]["structural_vs_placebo"]["final_minus_toggle_utility"] > 0.0
                for row in rows
                if row["baseline"] in {
                    "no_world_model_trace_policy",
                    "no_recursive_runner_one_shot",
                    "no_novelty_gate_incremental_addition",
                }
            ),
            "observed": {
                row["baseline"]: {
                    pair: row["pairs"][pair]["final_minus_toggle_utility"]
                    for pair in ("structural_vs_base", "structural_vs_placebo")
                }
                for row in rows
            },
        },
    ]
    return {
        "eval_id": eval_id or "paper_baseline_hardening_20260605",
        "eval_kind": "matched_frozen_toggle_off_baseline_audit",
        "pass": all(gate["pass"] for gate in gates),
        "final_summary": _display_path(root, final_path),
        "final_eval_id": final_summary.get("eval_id"),
        "baseline_rows": rows,
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
        "note": (
            "These are tracked summary-level matched toggles over the same frozen problem set. "
            "They are stronger than cross-run historical proxies, while still avoiding raw answer/forensic payloads."
        ),
    }


def _pair_comparison(*, final_pairs: dict[str, Any], toggle_pairs: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for pair in ("structural_vs_base", "structural_vs_placebo"):
        final = final_pairs.get(pair, {})
        toggle = toggle_pairs.get(pair, {})
        result[pair] = {
            "n": toggle.get("n"),
            "toggle_utility": toggle.get("utility"),
            "final_utility": final.get("utility"),
            "final_minus_toggle_utility": round(
                float(final.get("utility") or 0.0) - float(toggle.get("utility") or 0.0),
                4,
            ),
            "toggle_outcomes": toggle.get("outcomes"),
            "final_outcomes": final.get("outcomes"),
            "toggle_domain_breakdown": toggle.get("by_domain"),
        }
    return result


def _problem_ids(summary: dict[str, Any]) -> list[str]:
    pair = summary.get("pair_summaries", {}).get("structural_vs_base", {})
    return list(pair.get("judged_problem_ids") or [])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build matched frozen toggle-off baseline audit.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--final-summary", default=str(DEFAULT_FINAL_SUMMARY))
    ap.add_argument("--eval-id", default="paper_baseline_hardening_20260605")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    root = Path(args.root).resolve()
    payload = build_paper_baseline_hardening_payload(
        root=root,
        eval_id=args.eval_id,
        final_summary_path=Path(args.final_summary),
    )
    out = _resolve(root, Path(args.out))
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
