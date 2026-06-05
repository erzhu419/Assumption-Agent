"""Reproducibility package manifest for the paper-facing audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_FINAL_SUMMARY = Path(
    "phase four/assumption_graph/structural_live_ablation_20260603/"
    "structural_live_all_repairs_margin100_v2_gpt54mini_gpt55_20260604_summary.json"
)
DEFAULT_OUT = PAPER_DIR / "paper_repro_pack_20260605.json"

DEFAULT_ARTIFACTS = [
    PAPER_DIR / "paper_main_experiment_20260605.json",
    PAPER_DIR / "paper_baseline_hardening_20260605.json",
    PAPER_DIR / "paper_retrieval_baselines_20260605.json",
    PAPER_DIR / "morphism_claim_bundle_20260605.json",
    PAPER_DIR / "paper_negative_results_20260605.json",
    PAPER_DIR / "paper_benchmark_line_20260604.json",
    PAPER_DIR / "formal_engine_depth_audit_20260604.json",
    PAPER_DIR / "first_party_world_model_scale_20260604.json",
    DEFAULT_FINAL_SUMMARY,
    Path(
        "phase four/assumption_graph/structural_live_ablation_20260603/"
        "structural_live_natural100_v1_gpt54mini_gpt55_20260603_summary.json"
    ),
    Path(
        "phase four/assumption_graph/structural_live_ablation_20260603/"
        "structural_live_natural_repaired_residual_signal_incremental100_v1_gpt54mini_gpt55_20260603_summary.json"
    ),
]

DEFAULT_CODE_FILES = [
    Path("assumption_os/paper_main_experiment.py"),
    Path("assumption_os/paper_baseline_hardening.py"),
    Path("assumption_os/paper_retrieval_baselines.py"),
    Path("assumption_os/morphism_claims.py"),
    Path("assumption_os/paper_negative_results.py"),
    Path("assumption_os/paper_repro_pack.py"),
    Path("tests/test_assumption_os.py"),
]


def build_paper_repro_pack_payload(
    *,
    root: Path,
    eval_id: str | None = None,
    final_summary_path: Path | None = None,
    artifact_paths: list[Path] | None = None,
    code_paths: list[Path] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    final_path = _resolve(root, final_summary_path or DEFAULT_FINAL_SUMMARY)
    final_summary = _load_json(final_path)
    artifacts = artifact_paths or DEFAULT_ARTIFACTS
    code = code_paths or DEFAULT_CODE_FILES
    manifest = [
        _manifest_row(root, _resolve(root, path), kind="artifact")
        for path in artifacts
        if _resolve(root, path).exists()
    ] + [
        _manifest_row(root, _resolve(root, path), kind="code")
        for path in code
        if _resolve(root, path).exists()
    ]
    domain_counts = final_summary.get("pair_summaries", {}).get("structural_vs_base", {}).get("by_domain", {})
    gates = [
        {
            "gate": "exact_commands_present",
            "pass": len(_exact_commands()) >= 6,
            "observed": len(_exact_commands()),
        },
        {
            "gate": "artifact_hash_manifest_present",
            "pass": len(manifest) >= 10 and all(row.get("sha256") for row in manifest),
            "observed": {"row_count": len(manifest)},
        },
        {
            "gate": "env_vars_are_names_only",
            "pass": all("value" not in row for row in _api_env_vars()),
            "observed": [row["name"] for row in _api_env_vars()],
        },
        {
            "gate": "redacted_data_card_present",
            "pass": int(final_summary.get("plan", {}).get("selected_case_count") or 0) == 100,
            "observed": {
                "selected_case_count": final_summary.get("plan", {}).get("selected_case_count"),
                "domain_count": len(domain_counts),
            },
        },
    ]
    return {
        "eval_id": eval_id or "paper_repro_pack_20260605",
        "eval_kind": "paper_reproducibility_package_manifest",
        "pass": all(gate["pass"] for gate in gates),
        "exact_commands": _exact_commands(),
        "frozen_configs": {
            "final_summary": _display_path(root, final_path),
            "final_eval_id": final_summary.get("eval_id"),
            "sample_path": _redact_absolute_path(final_summary.get("plan", {}).get("sample_path")),
            "graph_dir": _redact_absolute_path(final_summary.get("plan", {}).get("graph_dir")),
            "selected_case_count": final_summary.get("plan", {}).get("selected_case_count"),
            "selection_mode": final_summary.get("plan", {}).get("selection_mode"),
            "solver_model_alias": final_summary.get("plan", {}).get("solver_model"),
            "judge_model_alias": final_summary.get("plan", {}).get("judge_model"),
            "repair_patterns": final_summary.get("plan", {}).get("repair_patterns"),
            "abstain_patterns": final_summary.get("plan", {}).get("abstain_patterns"),
        },
        "data_card": {
            "task_count": final_summary.get("plan", {}).get("selected_case_count"),
            "domains": {
                domain: row.get("n")
                for domain, row in sorted(domain_counts.items())
            },
            "unit_of_analysis": "problem_id",
            "stored_public_artifacts": "summary/audit JSON and Markdown only",
            "excluded_from_repro_pack": [
                "raw model answers",
                "forensic JSONL with judge raw text",
                "API keys",
                "local cache files",
            ],
            "redaction_policy": "No secret values are written; absolute local roots are redacted to <REPO>.",
        },
        "api_env_vars": _api_env_vars(),
        "artifact_source_manifest": manifest,
        "one_click_main_table_script": "python3 -m assumption_os.paper_main_experiment --root . --out phase\\ four/assumption_graph/paper_readiness_20260604/paper_main_experiment_20260605.json",
        "gates": gates,
        "failed_gates": [gate["gate"] for gate in gates if not gate["pass"]],
    }


def _exact_commands() -> list[dict[str, str]]:
    return [
        {
            "name": "main_experiment",
            "command": "python3 -m assumption_os.paper_main_experiment --root . --out 'phase four/assumption_graph/paper_readiness_20260604/paper_main_experiment_20260605.json'",
        },
        {
            "name": "matched_toggle_baselines",
            "command": "python3 -m assumption_os.paper_baseline_hardening --root . --out 'phase four/assumption_graph/paper_readiness_20260604/paper_baseline_hardening_20260605.json'",
        },
        {
            "name": "full_text_rag_retrieval_baselines",
            "command": "python3 -m assumption_os.paper_retrieval_baselines --include-neural --out 'phase four/assumption_graph/paper_readiness_20260604/paper_retrieval_baselines_20260605.json'",
        },
        {
            "name": "morphism_claim_bundle",
            "command": "python3 -m assumption_os.morphism_claims --root . --out 'phase four/assumption_graph/paper_readiness_20260604/morphism_claim_bundle_20260605.json'",
        },
        {
            "name": "negative_results",
            "command": "python3 -m assumption_os.paper_negative_results --root . --out 'phase four/assumption_graph/paper_readiness_20260604/paper_negative_results_20260605.json'",
        },
        {
            "name": "paper_benchmark_line",
            "command": "python3 -m assumption_os.paper_benchmark_line --root . --out 'phase four/assumption_graph/paper_readiness_20260604/paper_benchmark_line_20260604.json'",
        },
        {
            "name": "repro_pack",
            "command": "python3 -m assumption_os.paper_repro_pack --root . --out 'phase four/assumption_graph/paper_readiness_20260604/paper_repro_pack_20260605.json'",
        },
        {
            "name": "test_suite",
            "command": "python3 -m unittest tests.test_assumption_os",
        },
    ]


def _api_env_vars() -> list[dict[str, str]]:
    return [
        {"name": "RUOLI_BASE_URL", "purpose": "OpenAI-compatible base URL for live runs."},
        {"name": "RUOLI_GPT_KEY", "purpose": "GPT model API key; set in environment only."},
        {"name": "GPT5_API_KEY", "purpose": "Fallback GPT API key name."},
        {"name": "GPT5_BASE_URL", "purpose": "Fallback GPT base URL."},
        {"name": "GPT55_MODEL", "purpose": "High-quality judge/single-run model, default gpt-5.5."},
        {"name": "GPT_MINI_MODEL", "purpose": "Repeated-call cheaper GPT model, default gpt-5.4-mini."},
        {"name": "RUOLI_GEMINI_KEY", "purpose": "Gemini API key; set in environment only."},
        {"name": "GEMINI_PROXY_API_KEY", "purpose": "Fallback Gemini API key name."},
        {"name": "GEMINI_PROXY_BASE_URL", "purpose": "Fallback Gemini base URL."},
        {"name": "GEMINI_FLASH_LOW_MODEL", "purpose": "Repeated-call cheaper Gemini model, default gemini-3.5-flash-low."},
        {"name": "ASSUMPTION_OS_EMBEDDING_API_KEY", "purpose": "Optional embedding API key for external embedding baseline."},
        {"name": "ASSUMPTION_OS_EMBEDDING_BASE_URL", "purpose": "Optional embedding API base URL."},
        {"name": "ASSUMPTION_OS_EMBEDDING_MODEL", "purpose": "Optional embedding model name."},
    ]


def _manifest_row(root: Path, path: Path, *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": _display_path(root, path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _redact_absolute_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    marker = "/home/erzhu419/mine_code/Asumption Agent/"
    if value.startswith(marker):
        return "<REPO>/" + value[len(marker):]
    return value


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
    ap = argparse.ArgumentParser(description="Build paper reproducibility pack manifest.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--final-summary", default=str(DEFAULT_FINAL_SUMMARY))
    ap.add_argument("--eval-id", default="paper_repro_pack_20260605")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    root = Path(args.root).resolve()
    payload = build_paper_repro_pack_payload(
        root=root,
        eval_id=args.eval_id,
        final_summary_path=Path(args.final_summary),
    )
    out = _resolve(root, Path(args.out))
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "manifest_rows": len(payload["artifact_source_manifest"]),
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
