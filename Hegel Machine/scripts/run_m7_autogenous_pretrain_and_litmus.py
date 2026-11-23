#!/usr/bin/env python3
"""Hegel M7.0 Runner: Blank-Seed Offline Forest Pretraining + Zero-Handcrafted-Skill Wave-1 -> Wave-2 Litmus Test.

Proves genuine Autogenous Philosophical Forest Growth (`0` hand-authored `SKILL_01..07`, `0` `node_F01..F12`):
  Phase 1 (`pretrain`): Starts from `K0 = 4` methodological axioms in `hegel_clean_scipy_sandbox:v1`, solves 6
    strictly disjoint multi-regime physics/biostats/optimization pretraining problems via live official ChatGPT
    Codex (`auth_mode == "chatgpt"`), compiles container-unit-tested `autogenous_*.py` wormhole modules via
    `AutonomousAufhebungCompiler`, splits nodes on regime preconditions, anti-unifies operator pipelines across
    trees, and locks `pretrained_autogenous_forest_bundle.json`.
  Phase 2 (`wave_litmus`): Loads the locked `pretrained_autogenous_forest_bundle.json` (with `SKILL_01..07` 100%
    physically excluded!), runs Wave-1 `tb__baseline-free-localization`, performs blind online Wave-1 -> Wave-2
    `AutonomousAufhebungCompiler` consolidation (`zero` verifier peek), and evaluates Wave-2
    `tb__guided-wave-localization` + `tb__virtual-baseline-localization` head-to-head against `Thesis (Bare)`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import (
    run_m6_9_a_two_branch_episode,
)
from hegel_machine.cognition.m7_autogenous_forest_pretraining import (
    consolidate_wave1_via_aufhebung_compiler,
    load_autogenous_forest_bundle,
    run_blank_seed_forest_pretraining,
    serialize_autogenous_forest_bundle,
)


def write_sha256_manifest(evidence_root: Path) -> int:
    lines: list[str] = []
    for fp in sorted(evidence_root.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(evidence_root).as_posix()
            digest = hashlib.sha256(fp.read_bytes()).hexdigest()
            lines.append(f"{digest}  {rel}")
    (evidence_root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def run_wave_litmus(
    *,
    evidence_root: Path,
    pretrained_bundle_path: Path,
    model: str,
    max_turns: int,
    reasoning_effort: str,
    task_filter: list[str] | None = None,
) -> dict[str, Any]:
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    assert codex_info.get("auth_mode") == "chatgpt", f"Official ChatGPT OAuth required: {codex_info}"

    forest, archive, meta_dag, bundle_meta = load_autogenous_forest_bundle(pretrained_bundle_path)
    waves_dict = m6f.load_frontier_scout_20_tasks()
    all_tasks = {t.task_id: t for tl in waves_dict.values() for t in tl}

    w1_ids = ["tb__baseline-free-localization"]
    w2_ids = ["tb__guided-wave-localization", "tb__virtual-baseline-localization"]
    if task_filter:
        allow = set(task_filter)
        w1_ids = [tid for tid in w1_ids if tid in allow]
        w2_ids = [tid for tid in w2_ids if tid in allow]

    litmus_dir = evidence_root / "wave_continual_litmus"
    litmus_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{evidence_root.name}__autogenous_litmus"

    w1_summaries: list[dict[str, Any]] = []
    w1_anti_records: list[tuple[m6f.FrontierBenchmarkTask, dict[str, Any], Path]] = []

    # 1. Wave-1 Execution (using ONLY Blank-Seed Pretrained Forest, ZERO SKILL_01..07!)
    branches_root = litmus_dir / "branches"
    hippo_w1 = HippoRAGWormholeIndex(forest)
    for tid in w1_ids:
        task = all_tasks[tid]
        ep_dir = litmus_dir / "runs" / tid
        if (ep_dir / "00_episode_summary.json").is_file():
            ep_sum = json.loads((ep_dir / "00_episode_summary.json").read_text(encoding="utf-8"))
        else:
            print(f"[M7-LITMUS-W1] Starting `{tid}`...", flush=True)
            ep_sum = run_m6_9_a_two_branch_episode(
                task=task,
                model=model,
                codex_info=codex_info,
                forest=forest,
                forest_archive=archive,
                hippo_idx=hippo_w1,
                episode_out_dir=ep_dir,
                branches_work_root=branches_root,
                run_id=run_id,
                max_turns_per_branch=max_turns,
                cmd_timeout_sec=180,
                reasoning_effort=reasoning_effort,
            )
            print(f"[M7-LITMUS-W1] Finished `{tid}`: SYNTH={ep_sum['strict_pass']} | {ep_sum['postlock_branch_outcomes']}", flush=True)
        w1_summaries.append(ep_sum)
        anti_dir = branches_root / f"{task.wave}__{model}__{task.task_id}" / "branch_2_antithesis_wormhole"
        anti_sum_fp = anti_dir / "00_episode_summary.json"
        if anti_sum_fp.is_file():
            w1_anti_records.append((task, json.loads(anti_sum_fp.read_text(encoding="utf-8")), anti_dir))

    # 2. Blind Online Wave-1 -> Wave-2 Autonomous Aufhebung Consolidation (`Zero` verifier peek!)
    w1_cons_fp = litmus_dir / "wave1_aufhebung_consolidation.json"
    post_w1_bundle_fp = litmus_dir / "post_wave1_autogenous_forest_bundle.json"
    if w1_cons_fp.is_file() and post_w1_bundle_fp.is_file():
        w1_cons = json.loads(w1_cons_fp.read_text(encoding="utf-8"))
        forest, archive, meta_dag, _ = load_autogenous_forest_bundle(post_w1_bundle_fp)
    else:
        w1_cons = consolidate_wave1_via_aufhebung_compiler(
            wave1_antithesis_episodes=w1_anti_records,
            forest=forest,
            archive=archive,
            codex_info=codex_info,
            model=model,
            run_id=run_id,
            reasoning_effort=reasoning_effort,
        )
        w1_cons_fp.write_text(json.dumps(w1_cons, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        serialize_autogenous_forest_bundle(
            forest=forest,
            archive=archive,
            meta_dag=meta_dag,
            out_path=post_w1_bundle_fp,
            extra_metadata={"stage": "post_wave1_aufhebung"},
        )

    # 3. Wave-2 Execution (with Pretrained + Wave-1 Aufhebung Forest, ZERO SKILL_05 / node_F05!)
    hippo_w2 = HippoRAGWormholeIndex(forest)
    w2_summaries: list[dict[str, Any]] = []
    for tid in w2_ids:
        task = all_tasks[tid]
        ep_dir = litmus_dir / "runs" / tid
        if (ep_dir / "00_episode_summary.json").is_file():
            ep_sum = json.loads((ep_dir / "00_episode_summary.json").read_text(encoding="utf-8"))
        else:
            print(f"[M7-LITMUS-W2] Starting `{tid}`...", flush=True)
            ep_sum = run_m6_9_a_two_branch_episode(
                task=task,
                model=model,
                codex_info=codex_info,
                forest=forest,
                forest_archive=archive,
                hippo_idx=hippo_w2,
                episode_out_dir=ep_dir,
                branches_work_root=branches_root,
                run_id=run_id,
                max_turns_per_branch=max_turns,
                cmd_timeout_sec=180,
                reasoning_effort=reasoning_effort,
            )
            print(f"[M7-LITMUS-W2] Finished `{tid}`: SYNTH={ep_sum['strict_pass']} | {ep_sum['postlock_branch_outcomes']}", flush=True)
        w2_summaries.append(ep_sum)

    all_s = [*w1_summaries, *w2_summaries]
    report = {
        "model": model,
        "auth_mode": codex_info["auth_mode"],
        "handcrafted_domain_skills_mounted": 0,
        "pretrained_bundle_sha256": bundle_meta["bundle_sha256"],
        "wave1_aufhebung_consolidation": w1_cons,
        "total_tasks": len(all_s),
        "synthesis_strict_passes": sum(1 for s in all_s if s["strict_pass"]),
        "antithesis_strict_passes": sum(
            1 for s in all_s if s["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"]
        ),
        "thesis_bare_strict_passes": sum(
            1 for s in all_s if s["postlock_branch_outcomes"]["thesis_bare_run2_strict_pass"]
        ),
        "total_autogenous_skill_calls": sum(int(s.get("skill_activation_count", 0)) for s in all_s),
        "per_task_results": [
            {
                "task_id": s["task_id"],
                "wave": s["wave"],
                "synthesis_pass": s["strict_pass"],
                "antithesis_pass": s["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"],
                "thesis_bare_pass": s["postlock_branch_outcomes"]["thesis_bare_run2_strict_pass"],
                "selected_branch": s["synthesis_decision"]["selected_branch"],
                "selection_mechanism": s["synthesis_decision"]["selection_mechanism"],
                "staged_autogenous_modules": [m["module_import"] for m in s.get("staged_wormhole_modules", [])],
                "autogenous_skill_calls": s.get("skill_activation_count", 0),
            }
            for s in all_s
        ],
    }
    (litmus_dir / "00_litmus_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--evidence-dir",
        default="docs/evidence/m7_0_autogenous_forest_pretrain_and_litmus_20261123",
    )
    ap.add_argument("--phase", choices=["pretrain", "wave_litmus", "all"], default="all")
    ap.add_argument("--model", default="gpt-6.2-sol")
    ap.add_argument("--pretrain-turns", type=int, default=16)
    ap.add_argument("--eval-turns", type=int, default=28)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--pretrain-tasks", nargs="*", default=None)
    ap.add_argument("--litmus-tasks", nargs="*", default=None)
    args = ap.parse_args()

    ev_root = (REPO_ROOT / args.evidence_dir).resolve()
    ev_root.mkdir(parents=True, exist_ok=True)
    pt_dir = ev_root / "pretraining"
    bundle_fp = pt_dir / "pretrained_autogenous_forest_bundle.json"

    if args.phase in {"pretrain", "all"}:
        pt_manifest = run_blank_seed_forest_pretraining(
            out_dir=pt_dir,
            model=args.model,
            max_turns_per_task=args.pretrain_turns,
            reasoning_effort=args.reasoning_effort,
            selected_task_ids=args.pretrain_tasks,
        )
        print(
            json.dumps(
                {
                    "phase": "pretrain_done",
                    "pass_count": sum(1 for s in pt_manifest["episodes"] if s["strict_pass"]),
                    "total_tasks": len(pt_manifest["episodes"]),
                    "bundle_info": pt_manifest["bundle_info"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if args.phase in {"wave_litmus", "all"}:
        assert bundle_fp.is_file(), f"Missing pretrained bundle: {bundle_fp}"
        litmus_rep = run_wave_litmus(
            evidence_root=ev_root,
            pretrained_bundle_path=bundle_fp,
            model=args.model,
            max_turns=args.eval_turns,
            reasoning_effort=args.reasoning_effort,
            task_filter=args.litmus_tasks,
        )
        print(json.dumps({"phase": "wave_litmus_done", **litmus_rep}, ensure_ascii=False), flush=True)

    file_cnt = write_sha256_manifest(ev_root)
    print(f"SHA256SUMS_LOCKED: {file_cnt} files in {ev_root}", flush=True)


if __name__ == "__main__":
    main()
