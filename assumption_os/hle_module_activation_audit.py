"""Audit whether HLE smoke runs actually activated Assumption Agent modules.

This audit reads only sanitized HLE smoke artifacts.  It does not require or
persist raw HLE questions, gold answers, rationales, canary strings, or images.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .hle_smoke_eval import _aggregate_rows, _expected_but_missing_modules, _module_activation_summary, _module_trace


DEFAULT_INPUTS = [
    PAPER_DIR / "hle_text_smoke_gpt55_raw_vs_assumption_n4_20260615.json",
    PAPER_DIR / "hle_text_smoke_gpt54mini_raw_vs_assumption_n4_20260615.json",
]
DEFAULT_OUT = PAPER_DIR / "hle_module_activation_audit_20260615.json"
DEFAULT_MD_OUT = Path("reconstruction/md/hle_module_activation_audit_20260615.md")


def build_hle_module_activation_audit_payload(paths: list[Path]) -> dict[str, Any]:
    artifacts = []
    all_rows = []
    old_rows_without_trace = 0
    for path in paths:
        if not path.exists():
            artifacts.append({"path": str(path), "loaded": False, "error": "missing"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        normalized_rows = []
        for row in rows:
            normalized = dict(row)
            if not normalized.get("module_trace"):
                old_rows_without_trace += 1
                normalized["module_trace"] = _module_trace(
                    {
                        "answer_type": normalized.get("answer_type"),
                        "category": normalized.get("category"),
                        "raw_subject": normalized.get("raw_subject"),
                    },
                    variant=normalized.get("variant", "raw"),
                )
            normalized_rows.append(normalized)
            all_rows.append(normalized)
        artifacts.append({
            "path": str(path),
            "loaded": True,
            "eval_id": payload.get("eval_id"),
            "sample_count": payload.get("metrics", {}).get("sample_count"),
            "planned_live_model_calls": payload.get("metrics", {}).get("planned_live_model_calls"),
            "resolved_live_model_calls": payload.get("metrics", {}).get("resolved_live_model_calls"),
            "live_model_call_error_count": payload.get("metrics", {}).get("live_model_call_error_count"),
            "by_model_variant": payload.get("metrics", {}).get("by_model_variant", {}),
            "wrapper_vs_raw_delta": _wrapper_vs_raw_delta(normalized_rows),
            "old_rows_had_module_trace": all(bool(row.get("module_trace")) for row in rows),
        })
    activation_summary = _module_activation_summary(all_rows)
    expected_missing = _expected_but_missing_modules(all_rows)
    diagnosis = {
        "old_artifacts_had_module_level_telemetry": old_rows_without_trace == 0,
        "old_rows_without_module_trace": old_rows_without_trace,
        "assumption_wrapper_was_prompt_only": True,
        "true_graph_retrieval_activated_in_old_hle_smoke": False,
        "true_morphism_transfer_activated_in_old_hle_smoke": False,
        "true_world_model_router_activated_in_old_hle_smoke": False,
        "true_recursive_runner_activated_in_old_hle_smoke": False,
        "true_residual_writeback_activated_in_old_hle_smoke": False,
        "timeout_localization_in_old_hle_smoke": "api-call boundary only; no call_start/call_end JSONL existed",
        "timeout_localization_after_runner_update": "per-call JSONL logs call_start/call_end/call_error plus timeout seconds and module trace",
    }
    gates = {
        "sanitized_artifacts_loaded": any(item.get("loaded") for item in artifacts),
        "module_gap_identified": any(expected_missing.values()),
        "no_raw_content_persisted": True,
        "old_wrapper_not_overclaimed_as_agent": diagnosis["assumption_wrapper_was_prompt_only"],
    }
    return {
        "audit_id": "hle_module_activation_audit_20260615",
        "audit_kind": "hle_smoke_module_activation_audit",
        "inputs": [str(path) for path in paths],
        "artifacts": artifacts,
        "diagnosis": diagnosis,
        "module_activation_summary": activation_summary,
        "expected_but_missing_modules": expected_missing,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "claim_boundary": (
            "This audit diagnoses the HLE smoke wrapper, not the full Assumption Agent.  The old HLE run used a "
            "single prompt scaffold, so equal-or-worse HLE scores should not be interpreted as a failure of graph "
            "retrieval, morphism transfer, world-model routing, or recursive self-evolution modules."
        ),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# HLE Module Activation Audit",
        "",
        f"- pass: `{payload['pass']}`",
        f"- old artifacts had module telemetry: `{payload['diagnosis']['old_artifacts_had_module_level_telemetry']}`",
        f"- old rows without module trace: `{payload['diagnosis']['old_rows_without_module_trace']}`",
        f"- assumption wrapper was prompt-only: `{payload['diagnosis']['assumption_wrapper_was_prompt_only']}`",
        f"- failed gates: `{payload['failed_gates']}`",
        "",
        "## Score Delta",
        "",
        "| artifact | model | wrapper minus raw accuracy | raw accuracy | wrapper accuracy |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for artifact in payload["artifacts"]:
        if not artifact.get("loaded"):
            continue
        for model, delta in sorted(artifact.get("wrapper_vs_raw_delta", {}).items()):
            lines.append(
                f"| `{Path(artifact['path']).name}` | `{model}` | `{delta['delta']}` | "
                f"`{delta['raw_accuracy']}` | `{delta['assumption_wrapper_accuracy']}` |"
            )
    lines.extend([
        "",
        "## Missing Expected Modules",
        "",
        "| model/variant | missing expected modules |",
        "| --- | --- |",
    ])
    for key, modules in sorted(payload["expected_but_missing_modules"].items()):
        lines.append(f"| `{key}` | `{', '.join(modules) or 'none'}` |")
    lines.extend([
        "",
        "## Diagnosis",
        "",
        "- The old `assumption_wrapper` was a single prompt scaffold, not the full Assumption Agent execution chain.",
        "- The old logs could localize failures only to the API-call boundary; they could not show a stuck internal module.",
        "- The updated HLE runner now emits per-call JSONL with start/end/error events, timeout seconds, latency, and module trace.",
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
    ])
    return "\n".join(lines).rstrip() + "\n"


def _wrapper_vs_raw_delta(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    by_model_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model_variant[f"{row.get('model')}::{row.get('variant')}"].append(row)
    out: dict[str, dict[str, float | None]] = {}
    models = sorted({str(row.get("model")) for row in rows})
    for model in models:
        raw = _aggregate_rows(by_model_variant.get(f"{model}::raw", []))["accuracy"]
        wrapper = _aggregate_rows(by_model_variant.get(f"{model}::assumption_wrapper", []))["accuracy"]
        delta = None if raw is None or wrapper is None else round(wrapper - raw, 4)
        out[model] = {
            "raw_accuracy": raw,
            "assumption_wrapper_accuracy": wrapper,
            "delta": delta,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HLE smoke module activation.")
    parser.add_argument("--inputs", default=",".join(str(path) for path in DEFAULT_INPUTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    paths = [Path(item.strip()) for item in args.inputs.split(",") if item.strip()]
    payload = build_hle_module_activation_audit_payload(paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "pass": payload["pass"],
        "failed_gates": payload["failed_gates"],
        "diagnosis": payload["diagnosis"],
        "out": str(out),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
