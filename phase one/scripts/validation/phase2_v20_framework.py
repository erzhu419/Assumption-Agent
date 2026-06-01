"""
v20: Combined v19a (frame directive) + v19b (problem rewrite).

Hypothesis: v19a (+10pp) and v19b (+14pp) both won by foregrounding frame
decision. If we do BOTH — rewrite the problem AND place frame as PRIMARY
directive — the effects should compose.

Architecture:
  Turn 0 (frame + rewrite combined):
    Produce JSON with {frame, critical_reframe, anti_patterns, evaluation_criteria,
                        rewritten_problem, what_changed}
  Turn 1 (solve):
    PRIMARY FRAME block (v19a-style) + rewritten problem (v19b-style)
    + secondary wisdom/cases
  Turn 2 (audit):
    Check both frame adherence and whether solved rewritten vs original

Math/sci bypass to v12c hygiene.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client, parse_json_from_llm  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry
from phase2_v12_framework import (EXECUTE_MATH, EXECUTE_SCIENCE, format_priors,
                                   MATH_SCI)
from phase2_v15_exemplar_framework import (build_same_domain_exemplar,
                                            format_wisdom_with_cases)


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH_DEFAULT = CACHE / "wisdom_library.json"
SELECTIONS_PATH_DEFAULT = CACHE / "phase2_v3_selections.json"
EXEMPLARS_PATH = CACHE / "wisdom_diverse_exemplars.json"
EMB_PATH = CACHE / "signal_embeddings.npz"
V13_REFLECT = ANSWERS_DIR / "phase2_v13_reflect_answers.json"
OURS_27 = ANSWERS_DIR / "ours_27_answers.json"


FRAME_REWRITE_PROMPT = """# 问题诊断 + 重写任务

读下面问题，**同时** 做两件事：
1. 诊断问题的 frame（对象层 / 范式层 / 混合）及 critical reframe
2. 把问题重写成一个更精确、显式化了真正该回答维度的版本

## 原始问题
{problem}

## 输出 JSON（不要代码块）
{{
  "frame": "object" | "paradigm" | "hybrid",
  "critical_reframe": "如果误读会被误读成什么？真正该解的问题说清楚 (30-60 字)",
  "anti_patterns": ["要避免的常见错误做法 1", "2", ...],
  "evaluation_criteria": "什么样答案算好 (30-50 字)",
  "rewritten_problem": "重写版本（完整段落，可比原文稍长，显式化 stakeholder/criteria/constraints）",
  "what_changed": "相比原文凸显了什么 (20-50 字)"
}}

## 指导
- **对象层** (object): 纯计算/证明/实现，保留原问题为重写版，rewritten ≈ original
- **范式层** (paradigm): 显式化 stakeholder / 监管 / 投入 / criteria，rewrite 可大改
- **hybrid**: 两层都显式化
"""


EXECUTE_V20 = """# 你要解决下面的问题。

## ⚡ PRIMARY FRAME (顶层 directive，必须遵守)

**问题类型**: {frame}
**Critical reframe**: {critical_reframe}
**评判标准**: {evaluation_criteria}
**要避免的反模式**:
{anti_patterns_block}

---

## 📝 你要解决的问题（已经被诊断并重写，显式化真正该回答的维度）

### 原始问题
{problem}

### 重写版本（基于上面 frame 的显式化）
{rewritten_problem}

*重写点评*: {what_changed}

**针对重写版本作答**；若原问题本身清晰（object-level），重写 ≈ 原文，直接按原文答。

---

## 次要参考（仅在与 PRIMARY FRAME 一致时纳入）

{assumption_context_block}

{domain_execution_block}

### 本类别 attention priors
{priors_block}

### Wisdom + 判例集
{wisdom_case_block}

---

## 要求
- 完全服从 PRIMARY FRAME 与重写版本的 framing
- 次要参考是补充，不是主体
- 不超过 600 字

开始：
"""


REFLECT_PROMPT_V20 = """草稿已产。

## 原问题
{problem}

## Primary Frame (Turn 0 已定)
{frame_summary}

## 重写版本
{rewritten_problem}

## 草稿
{draft}

## 自检
1. 草稿是否按 PRIMARY FRAME 作答？滑回了错误 frame 吗？
2. 草稿是针对 original 还是 rewritten？如果重写凸显了什么但 draft 没 address，补上。
3. 踩了 anti_patterns 里的某条吗？

输出最终答案（不要 audit 过程）。不超过 650 字。
"""


def cache_load(p):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def format_anti_patterns(aps):
    if not aps:
        return "  (无)"
    return "\n".join(f"  - {ap}" for ap in aps)


def load_assumption_graph(
    graph_dir: str | None,
    *,
    proposals_path: str | None = None,
    proposal_ids: list[str] | None = None,
    proposal_types: set[str] | None = None,
    force_proposal_route: bool = False,
    force_proposal_domains: set[str] | None = None,
    force_proposal_difficulties: set[str] | None = None,
    route_scope_proposals: bool = False,
):
    if not graph_dir:
        return None
    from assumption_os.context import format_assumption_context
    from assumption_os.graph_memory import JsonlGraphStore, SimpleAssumptionGraph
    from assumption_os.proposal_overlay import apply_proposal_overlay, load_proposal_payload
    from assumption_os.retrieval_policy import format_policy_context, retrieve_phase2_assumptions

    graph_path = Path(graph_dir)
    if not graph_path.is_absolute():
        graph_path = PROJECT.parent / graph_path
    store = JsonlGraphStore(graph_path)
    proposal_payload = None
    if proposals_path:
        proposal_path = Path(proposals_path)
        if not proposal_path.is_absolute():
            proposal_path = PROJECT.parent / proposal_path
        proposal_payload = load_proposal_payload(proposal_path)
        applied = apply_proposal_overlay(
            store,
            proposal_payload,
            proposal_ids=proposal_ids,
            proposal_types=proposal_types,
        )
        if applied:
            print(f"  [assumption overlay] applied candidate nodes: {len(applied)}")
    graph = SimpleAssumptionGraph(store)
    return (
        graph,
        format_assumption_context,
        format_policy_context,
        retrieve_phase2_assumptions,
        proposal_payload,
        proposal_ids,
        proposal_types,
        force_proposal_route,
        force_proposal_domains or set(),
        force_proposal_difficulties or set(),
        route_scope_proposals,
    )


def parse_csv_set(raw: str) -> set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


def resolve_repo_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else PROJECT.parent / path


def build_assumption_context(
    graph_bundle,
    problem: str,
    meta: dict,
    pid: str,
    dom: str,
    diff: str,
    coverage_tags: list[str] | None = None,
    skip_domains: set[str] | None = None,
    trace_recorder=None,
) -> str:
    if not graph_bundle:
        return ""
    if skip_domains and dom in skip_domains:
        return ""
    (
        graph,
        formatter,
        policy_formatter,
        retrieve_policy,
        proposal_payload,
        proposal_ids,
        proposal_types,
        force_proposal_route,
        force_proposal_domains,
        force_proposal_difficulties,
        route_scope_proposals,
    ) = graph_bundle
    result = retrieve_policy(
        graph,
        problem=problem,
        meta=meta,
        pid=pid,
        domain=dom,
        difficulty=diff,
        top_k=8,
        pool_k=24,
        skip_domains=skip_domains,
    )
    if result and proposal_payload and (force_proposal_route or route_scope_proposals):
        from assumption_os.proposal_overlay import proposal_candidate_ids, proposal_route_target_ids

        target_ids: list[str] = []
        route_domain_ok = not force_proposal_domains or dom in force_proposal_domains
        route_difficulty_ok = (
            not force_proposal_difficulties or diff in force_proposal_difficulties
        )
        if route_domain_ok and route_difficulty_ok:
            target_ids = proposal_route_target_ids(
                graph.store,
                proposal_payload,
                problem={
                    "problem_id": pid,
                    "domain": dom,
                    "difficulty": diff,
                    "description": problem,
                    "coverage_tags": coverage_tags or [],
                },
                meta=meta,
                proposal_ids=proposal_ids,
                proposal_types=proposal_types,
            )
        if route_scope_proposals:
            candidates = proposal_candidate_ids(
                proposal_payload,
                proposal_ids=proposal_ids,
                proposal_types=proposal_types,
            )
            _remove_nodes_from_result(result, set(candidates) - set(target_ids))
        if force_proposal_route and route_domain_ok and route_difficulty_ok:
            _force_nodes_into_result(result, graph, target_ids)
    if trace_recorder and getattr(trace_recorder, "enabled", False):
        nodes = list(getattr(result.subgraph, "nodes", [])) if result else []
        trace_recorder.record_retrieval(
            problem_id=pid,
            component="phase2_assumption_graph_retrieval",
            assumption="Activated Assumption Graph context should improve the phase2 answer when routed to the current problem.",
            expected_effect="Retrieve method, harness, runtime, and residual assumptions that shape the draft rather than decorate it.",
            activated_assumption_ids=[node.id for node in nodes],
            artifacts={
                "domain": dom,
                "difficulty": diff,
                "node_types": [str(getattr(node.type, "value", node.type)) for node in nodes],
                "policy_notes": list(getattr(result, "policy_notes", [])) if result else [],
                "coverage_tags": coverage_tags or [],
            },
            metadata={
                "skip_domains": sorted(skip_domains or []),
                "force_proposal_route": force_proposal_route,
                "route_scope_proposals": route_scope_proposals,
            },
        )
    return policy_formatter(result, formatter, max_nodes=8)


def _force_nodes_into_result(result, graph, node_ids: list[str]) -> None:
    if not node_ids:
        return
    subgraph = result.subgraph
    existing = {node.id for node in subgraph.nodes}
    forced = [graph.store.nodes[nid] for nid in node_ids if nid in graph.store.nodes and nid not in existing]
    if not forced:
        return
    subgraph.nodes = [*forced, *subgraph.nodes]
    for node in forced:
        subgraph.scores[node.id] = max(1.25, subgraph.scores.get(node.id, 0.0))
    result.policy_notes.append("Forced proposal-route nodes: " + ", ".join(node.id for node in forced))


def _remove_nodes_from_result(result, node_ids: set[str]) -> None:
    if not node_ids:
        return
    subgraph = result.subgraph
    before = len(subgraph.nodes)
    subgraph.nodes = [node for node in subgraph.nodes if node.id not in node_ids]
    removed = before - len(subgraph.nodes)
    if not removed:
        return
    for node_id in node_ids:
        subgraph.scores.pop(node_id, None)
    result.policy_notes.append(f"Route-scoped proposal candidates removed: {removed}")


def build_domain_execution_block(enabled: bool, dom: str, problem: str, meta: dict) -> str:
    if not enabled:
        return ""
    from assumption_os.domain_templates import format_phase2_domain_execution_template

    return format_phase2_domain_execution_template(dom, problem, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v20")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-wisdoms", type=int, default=2)
    ap.add_argument("--sample", default="sample_100.json")
    ap.add_argument("--selections", default=None,
                    help="override selections file (e.g. phase2_v3_selections_v2.json)")
    ap.add_argument("--wisdom", default=None,
                    help="override wisdom library file")
    ap.add_argument("--assumption-graph", default=None,
                    help="optional Assumption Graph dir generated by `python -m assumption_os.build_graph`")
    ap.add_argument("--assumption-graph-skip-domains", default="software_engineering",
                    help="comma-separated domains where graph context is gated off")
    ap.add_argument("--assumption-proposals", default=None,
                    help="optional proposal JSON to overlay candidate nodes for a test run")
    ap.add_argument("--assumption-proposal-ids", nargs="*", default=None,
                    help="specific proposal ids to overlay; default overlays all proposal candidates")
    ap.add_argument("--assumption-proposal-types", default="",
                    help="comma-separated proposal types to overlay")
    ap.add_argument("--assumption-force-proposal-route", action="store_true",
                    help="force overlaid proposal target nodes only on their routed should-fire problems")
    ap.add_argument("--assumption-force-proposal-domains", default="",
                    help="comma-separated domains where proposal-route forcing is allowed")
    ap.add_argument("--assumption-force-proposal-difficulties", default="",
                    help="comma-separated difficulties where proposal-route forcing is allowed")
    ap.add_argument("--assumption-route-scope-proposals", action="store_true",
                    help="remove proposal candidate nodes outside their routed should-fire subset")
    ap.add_argument("--math-science-bridge", action="store_true",
                    help="use intent-aware math/science bypass prompts for research-bridge and science-decision rows")
    ap.add_argument("--disable-domain-execution-template", action="store_true",
                    help="disable domain execution templates that are enabled with --assumption-graph")
    ap.add_argument("--runtime-trace-events-out", default=None,
                    help="optional JSONL path for first-party LLM/retrieval runtime trace events")
    ap.add_argument("--runtime-trace-summary-out", default=None,
                    help="optional JSON summary path for TrialManifest-normalized runtime trace events")
    ap.add_argument("--runtime-trace-eval-id", default=None,
                    help="eval id for runtime trace manifests; defaults to <variant>_runtime_trace")
    ap.add_argument("--runtime-trace-writeback", action="store_true",
                    help="append runtime trace TrialManifests to --assumption-graph")
    args = ap.parse_args()
    if args.assumption_proposals and not args.assumption_graph:
        raise SystemExit("--assumption-proposals requires --assumption-graph")
    if args.runtime_trace_writeback and not args.assumption_graph:
        raise SystemExit("--runtime-trace-writeback requires --assumption-graph")

    selections_path = (CACHE / args.selections) if args.selections else SELECTIONS_PATH_DEFAULT
    wisdom_path = (CACHE / args.wisdom) if args.wisdom else WISDOM_PATH_DEFAULT

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    meta_path = ANSWERS_DIR / f"{args.variant}_meta.json"  # frame + rewrite combined
    drafts_path = ANSWERS_DIR / f"{args.variant}_drafts.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)
    meta = cache_load(meta_path)
    drafts = cache_load(drafts_path)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
    structures = cache_load(struct_path)

    library = json.loads(wisdom_path.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    selections = cache_load(selections_path)
    diverse_exs = cache_load(EXEMPLARS_PATH)
    v13 = json.loads(V13_REFLECT.read_text(encoding="utf-8"))
    ours = json.loads(OURS_27.read_text(encoding="utf-8"))

    emb = np.load(EMB_PATH, allow_pickle=True)
    prob_emb = emb["problem_emb"]
    prob_ids_emb = emb["problem_ids"].tolist()
    pid_to_emb_idx = {pid: i for i, pid in enumerate(prob_ids_emb)}

    sample = json.loads((CACHE / args.sample).read_text(encoding="utf-8"))[: args.n]

    assumption_graph = load_assumption_graph(
        args.assumption_graph,
        proposals_path=args.assumption_proposals,
        proposal_ids=args.assumption_proposal_ids,
        proposal_types=parse_csv_set(args.assumption_proposal_types),
        force_proposal_route=args.assumption_force_proposal_route,
        force_proposal_domains=parse_csv_set(args.assumption_force_proposal_domains),
        force_proposal_difficulties=parse_csv_set(args.assumption_force_proposal_difficulties),
        route_scope_proposals=args.assumption_route_scope_proposals,
    )
    assumption_graph_skip_domains = parse_csv_set(args.assumption_graph_skip_domains)
    domain_execution_enabled = bool(args.assumption_graph) and not args.disable_domain_execution_template
    trace_recorder = None
    if args.runtime_trace_events_out or args.runtime_trace_summary_out or args.runtime_trace_writeback:
        from assumption_os.runtime_trace import RuntimeTraceRecorder

        trace_recorder = RuntimeTraceRecorder(
            eval_id=args.runtime_trace_eval_id or f"{args.variant}_runtime_trace",
            events_out=resolve_repo_path(args.runtime_trace_events_out),
            summary_out=resolve_repo_path(args.runtime_trace_summary_out),
            graph_dir=resolve_repo_path(args.assumption_graph),
            writeback=args.runtime_trace_writeback,
        )
    client = create_client()
    t0 = time.time()
    new = hit = hyg = full = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        problem = p.get("description", "")

        if dom in MATH_SCI:
            if dom == "mathematics":
                prompt = EXECUTE_MATH.format(problem=problem)
                max_tok = 1100
            else:
                prompt = EXECUTE_SCIENCE.format(problem=problem)
                max_tok = 900
            if args.math_science_bridge:
                from assumption_os.math_science_policy import format_math_science_prompt

                prompt, max_tok, route = format_math_science_prompt(dom, problem)
                meta[pid] = {
                    "frame": "object" if route in {"math_formal", "science_mechanism"} else "hybrid",
                    "critical_reframe": f"math/science bypass route: {route}",
                    "rewritten_problem": problem,
                    "what_changed": "intent-aware math/science bypass",
                    "anti_patterns": [],
                    "bypass_route": route,
                }
                cache_save(meta_path, meta)
            try:
                resp = _generate_with_retry(client, prompt, max_tokens=max_tok, temperature=0.3)
                answers[pid] = resp["text"].strip()
                if trace_recorder:
                    trace_recorder.record_llm_call(
                        problem_id=pid,
                        component="phase2_math_science_bypass",
                        prompt_kind="math_science_bridge" if args.math_science_bridge else "math_science_hygiene",
                        assumption="A compact domain-specific math/science execution prompt is safer than generic graph context on bypass rows.",
                        expected_effect="Produce a concrete answer without importing irrelevant method assumptions.",
                        observed_effect=f"answer_chars={len(answers[pid])}",
                        artifacts={
                            "domain": dom,
                            "difficulty": diff,
                            "max_tokens": max_tok,
                            "temperature": 0.3,
                            "bypass_route": meta.get(pid, {}).get("bypass_route"),
                        },
                    )
            except Exception as e:
                print(f"  [err {pid}] {e}")
                continue
            hyg += 1
        else:
            # Turn 0: combined frame + rewrite
            if pid in meta:
                m = meta[pid]
            else:
                try:
                    r0 = _generate_with_retry(client, FRAME_REWRITE_PROMPT.format(problem=problem),
                                              max_tokens=700, temperature=0.2)
                    m = parse_json_from_llm(r0["text"])
                    required = {"frame", "critical_reframe", "anti_patterns",
                                "evaluation_criteria", "rewritten_problem", "what_changed"}
                    if not required.issubset(m.keys()):
                        raise ValueError(f"missing: {required - m.keys()}")
                    meta[pid] = m
                    cache_save(meta_path, meta)
                    if trace_recorder:
                        trace_recorder.record_llm_call(
                            problem_id=pid,
                            component="phase2_turn0_frame_rewrite",
                            prompt_kind="frame_rewrite",
                            assumption="Explicitly framing and rewriting the problem improves downstream assumption selection.",
                            expected_effect="Expose domain, evaluation criteria, anti-patterns, and a rewritten problem for the executor.",
                            observed_effect=f"frame={m.get('frame', '?')}; anti_patterns={len(m.get('anti_patterns', []))}",
                            artifacts={
                                "domain": dom,
                                "difficulty": diff,
                                "frame": m.get("frame"),
                                "what_changed": m.get("what_changed"),
                            },
                        )
                except Exception as e:
                    print(f"  [err meta {pid}] {e}")
                    m = {"frame": "hybrid", "critical_reframe": "", "anti_patterns": [],
                         "evaluation_criteria": "", "rewritten_problem": problem,
                         "what_changed": "fallback"}
                    meta[pid] = m
                    cache_save(meta_path, meta)

            key = f"{dom}__{diff}"
            struct = structures.get(key, {"attention_priors": []})
            priors = struct.get("attention_priors", [])

            sel_ids = selections.get(pid, [])[: args.max_wisdoms]
            wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]
            wisdom_entries = [w for w in wisdom_entries if w["id"] in diverse_exs]

            w_blocks = []
            for w in wisdom_entries:
                pre_mined = diverse_exs.get(w["id"], [])
                same_dom_ex = build_same_domain_exemplar(
                    pid, dom, sample, prob_emb, pid_to_emb_idx, v13, ours)
                cross = [e for e in pre_mined if e.get("domain") != dom][:3]
                if len(cross) < 3:
                    cross += [e for e in pre_mined if e not in cross][:3 - len(cross)]
                w_blocks.append(format_wisdom_with_cases(w, cross, same_dom_ex))
            wisdom_case_block = "\n\n---\n\n".join(w_blocks) if w_blocks else "  (无)"
            assumption_context_block = build_assumption_context(
                assumption_graph, problem, m, pid, dom, diff,
                coverage_tags=p.get("coverage_tags", []),
                skip_domains=assumption_graph_skip_domains,
                trace_recorder=trace_recorder,
            )
            domain_execution_block = build_domain_execution_block(
                domain_execution_enabled, dom, problem, m)

            if pid in drafts:
                draft = drafts[pid]
            else:
                try:
                    r1 = _generate_with_retry(client, EXECUTE_V20.format(
                        frame=m.get("frame", "hybrid"),
                        critical_reframe=m.get("critical_reframe", ""),
                        evaluation_criteria=m.get("evaluation_criteria", ""),
                        anti_patterns_block=format_anti_patterns(m.get("anti_patterns", [])),
                        problem=problem,
                        rewritten_problem=m.get("rewritten_problem", problem),
                        what_changed=m.get("what_changed", ""),
                        assumption_context_block=assumption_context_block,
                        domain_execution_block=domain_execution_block,
                        priors_block=format_priors(priors),
                        wisdom_case_block=wisdom_case_block,
                    ), max_tokens=1100, temperature=0.3)
                    draft = r1["text"].strip()
                    drafts[pid] = draft
                    cache_save(drafts_path, drafts)
                    if trace_recorder:
                        trace_recorder.record_llm_call(
                            problem_id=pid,
                            component="phase2_turn1_draft",
                            prompt_kind="execute_v20",
                            assumption="Combining frame, retrieved assumptions, domain execution constraints, priors, and cases should yield a better draft.",
                            expected_effect="Generate a task answer that operationalizes selected assumptions.",
                            observed_effect=f"draft_chars={len(draft)}; graph_context_chars={len(assumption_context_block)}",
                            artifacts={
                                "domain": dom,
                                "difficulty": diff,
                                "wisdom_count": len(wisdom_entries),
                                "has_assumption_context": bool(assumption_context_block.strip()),
                                "has_domain_execution_block": bool(domain_execution_block.strip()),
                                "max_tokens": 1100,
                                "temperature": 0.3,
                            },
                        )
                except Exception as e:
                    print(f"  [err draft {pid}] {e}")
                    continue

            # Turn 2 audit
            frame_summary = f"{m.get('frame', '?')}: {m.get('critical_reframe', '')}"
            try:
                r2 = _generate_with_retry(client, REFLECT_PROMPT_V20.format(
                    problem=problem, frame_summary=frame_summary,
                    rewritten_problem=m.get("rewritten_problem", problem),
                    draft=draft
                ), max_tokens=1100, temperature=0.3)
                answers[pid] = r2["text"].strip()
                cache_save(answers_path, answers)
                if trace_recorder:
                    trace_recorder.record_llm_call(
                        problem_id=pid,
                        component="phase2_turn2_audit_revise",
                        prompt_kind="reflect_v20",
                        assumption="A second-pass audit can detect decorative assumption use and revise toward concrete execution.",
                        expected_effect="Convert the draft into a final answer that better follows the frame and task constraints.",
                        observed_effect=f"final_chars={len(answers[pid])}",
                        artifacts={
                            "domain": dom,
                            "difficulty": diff,
                            "draft_chars": len(draft),
                            "max_tokens": 1100,
                            "temperature": 0.3,
                        },
                    )
            except Exception as e:
                print(f"  [err audit {pid}] {e}")
                continue
            full += 1

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(meta_path, meta)
            cache_save(drafts_path, drafts)
            print(f"  [{args.variant}] {i+1}/{len(sample)} hyg={hyg} full={full} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(meta_path, meta)
    cache_save(drafts_path, drafts)
    print(f"\n  [{args.variant}] done: hyg={hyg} full={full} ({time.time()-t0:.0f}s)")

    from collections import Counter
    fc = Counter(x.get("frame", "?") for x in meta.values())
    print(f"  Frame distribution: {dict(fc)}")
    if trace_recorder:
        trace_payload = trace_recorder.flush()
        print(f"  Runtime trace events: {trace_payload.get('event_count', 0)}")


if __name__ == "__main__":
    main()
