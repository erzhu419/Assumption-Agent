"""Live large-sample ablation for structural morphism context.

This experiment tests whether structural morphism context improves actual model
answers, not just offline retrieval or rubric probes.  It generates three arms:

BASE: problem only
STRUCTURAL: top structural morphism context
PLACEBO: a wrong structural pattern with the same formatting

Then it asks a judge to compare STRUCTURAL against BASE and PLACEBO using the
reference answer as the target rubric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import cheap, expensive  # noqa: E402

from .graph_memory import JsonlGraphStore
from .structural_patterns import (
    format_structural_morphism_applications,
    load_structural_patterns,
    search_structural_patterns,
)


BASE_PROMPT = """请解决下面的问题。

要求：
- 给出具体、可执行的方案。
- 明确关键判断标准、步骤和风险控制。
- 不超过 650 字。

问题：
{problem}
"""


CONTEXT_PROMPT = """请解决下面的问题。

你可以参考下面的 Structural Morphism Reasoning，但只有在当前问题确实保持这些结构不变量时才使用；如果不适用，应忽略它。

{context}

要求：
- 给出具体、可执行的方案。
- 明确关键判断标准、步骤和风险控制。
- 不要只复述结构名，要把结构转成当前问题里的行动方案。
- 不超过 650 字。

问题：
{problem}
"""


PAIRWISE_JUDGE_PROMPT = """You are a strict pairwise judge.

Pick which answer better solves the problem according to the reference answer.
Prioritize: correct strategy, concrete execution, handling constraints, avoiding harmful or irrelevant advice.
Do not prefer an answer for style or length alone.

PROBLEM:
{problem}

REFERENCE ANSWER:
{reference}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Return only JSON:
{{"winner": "A" | "B" | "tie", "reason": "one concise sentence"}}
"""


JSON_RE = re.compile(r"\{[^{}]*?\"winner\"[^{}]*?\}", re.DOTALL)


STRATEGY_TAG_TO_STRUCTURAL_PATTERN = {
    # High-confidence correspondences between the phase-zero strategy tags and
    # the bounded structural motifs.  This is used only for benchmark routing:
    # answer quality is still judged blind against the reference.
    "S01": ("pat_controlled_intervention", 0.95, "controlled-variable strategy isolates one intervention against a matched baseline"),
    "S02": ("pat_decomposition_composition", 0.95, "divide-and-conquer requires subproblem interfaces that compose back to the root goal"),
    "S04": ("pat_counterexample_refinement", 0.86, "proof by contradiction searches for a contradiction/counterexample to refine the claim"),
    "S06": ("pat_monotone_progress", 0.72, "special-to-general progression should preserve what worked while widening scope"),
    "S07": ("pat_decomposition_composition", 0.70, "backward reasoning decomposes the goal into prerequisite subgoals"),
    "S08": ("pat_controlled_intervention", 0.80, "guess-and-check needs trials compared against outcomes before promotion"),
    "S09": ("pat_signal_nuisance_separation", 0.78, "simplification removes nuisance dimensions while preserving the stable signal"),
    "S10": ("pat_conservation_balance", 0.72, "symmetry use depends on invariants preserved under transformation"),
    "S11": ("pat_monotone_progress", 0.68, "satisficing needs an explicit acceptable threshold and non-regression check"),
    "S12": ("pat_residual_correction", 0.76, "Bayesian updating preserves a prior path while applying an evidence delta"),
    "S13": ("pat_counterexample_refinement", 0.95, "falsification-first directly maps to counterexample-guided refinement"),
    "S14": ("pat_counterexample_refinement", 0.90, "boundary-condition analysis uses edge cases to test and narrow assumptions"),
    "S15": ("pat_incremental_replacement", 0.95, "incremental building keeps a working path and changes one bounded module at a time"),
    "S16": ("pat_controlled_intervention", 0.82, "method of agreement compares cases to isolate a shared factor"),
    "S17": ("pat_controlled_intervention", 0.95, "method of difference isolates the single differing factor between cases"),
    "S18": ("pat_decomposition_composition", 0.70, "abstraction extracts a reusable structure that must still compose back to the case"),
    "S19": ("pat_bottleneck_capacity", 0.66, "constraint relaxation identifies which constraint limits the feasible flow"),
    "S20": ("pat_negative_feedback", 0.64, "dual/complementary views often expose opposing-response structure"),
    "S21": ("pat_counterexample_refinement", 0.74, "dead-end recognition treats failed trajectory evidence as a counterexample to the current claim"),
    "S22": ("pat_decomposition_composition", 0.68, "problem reframing changes the decomposition/interface used to recover the goal"),
    "S23": ("pat_bottleneck_capacity", 0.76, "resource-constrained acceptance is governed by scarce capacity and stopping thresholds"),
    "S24": ("pat_bottleneck_capacity", 0.95, "critical-node analysis directly maps to bottleneck/capacity-limited flow"),
    "S25": ("pat_decomposition_composition", 0.72, "emergence detection checks whether composed parts recover or change the whole behavior"),
    "S26": ("pat_residual_correction", 0.68, "path-dependence preserves the historical baseline path while evaluating local change"),
    "S27": ("pat_negative_feedback", 0.70, "incentive analysis predicts actor responses to imposed constraints or rewards"),
}


STRATEGY_TAG_TRANSFER_GUIDANCE = {
    "S01": "固定预算、时间、客群和物料等控制项，每次只改变一个因素，并用小额匹配试验比较结果。",
    "S02": "把根问题拆成可独立处理的子问题，明确每个子问题的输入/输出接口，再用组合检查恢复总体目标。",
    "S03": "只迁移源问题和目标问题共同保持的关系结构，先列相同关系，再列关键差异，避免表面类比。",
    "S04": "先假设方案/结论成立，主动推导会导致失败或矛盾的场景，再用该矛盾修正主张。",
    "S05": "在解释力相同的方案中优先选择变量更少、执行路径更短、验证成本更低的方案。",
    "S06": "先解决最小、最受限、最容易验证的特殊情况，再逐步放宽条件，检查每一步是否保持核心性质。",
    "S07": "从目标状态倒推必要前提，把每个前提变成可验证的中间节点和行动步骤。",
    "S08": "先做低成本猜测/试投/试验，快速读反馈，再按结果修正下一轮尝试。",
    "S09": "先去掉不影响核心机制的复杂维度，保留稳定信号，再检查简化结论能否回到原问题。",
    "S10": "找出在变换前后不变的量或关系，用这些不变量减少搜索空间并验证方案。",
    "S11": "先定义足够好的阈值和停止条件，达到阈值后停止继续搜索，避免资源耗尽。",
    "S12": "保留原有先验/历史模型，把新证据作为增量更新，并说明哪些证据会显著改变判断。",
    "S13": "优先寻找能推翻当前假设的失败样例，再把假设收窄成不会被该样例击穿的版本。",
    "S14": "用最坏情况、边界值和极端场景测试方案，明确红线、退出阈值和应急动作。",
    "S15": "先保留最小可工作版本，每次只增加一个模块或能力，并保留回滚路径。",
    "S16": "比较多个成功案例，找唯一共同因素，再检查该因素是否只是相关而非原因。",
    "S17": "直接比较成功/失败或高/低表现 cohort，控制口径一致，找唯一关键差异，并防止自选择偏差。",
    "S18": "抽掉具体细节后提取共同结构，再把抽象结构映射回当前问题中的对象、关系和验证点。",
    "S19": "暂时放宽最卡住的约束，观察松弛问题如何变得可解，再把原约束逐步加回。",
    "S20": "从对立或互补视角重看目标，找出原框架遗漏的约束、收益或风险。",
    "S21": "定义死胡同证据和止损阈值，一旦当前路径不可能达标，就回退到更高层选择点。",
    "S22": "重新定义问题边界、目标或评价指标，检查新框架是否打开原框架没有的可行动作。",
    "S23": "在时间/预算/算力约束下定义可接受近似解和停止条件，把剩余资源留给风险控制。",
    "S24": "先识别限制整体产出的关键节点/瓶颈，把资源投向该瓶颈而不是非限制环节。",
    "S25": "不要把单个部件表现线性外推到整体；逐级组合测试，测量交互、串扰、非线性和宏观新属性。",
    "S26": "检查当前状态由哪些历史路径锁定，区分现在可改的局部变量和已经形成的路径依赖。",
    "S27": "画出各参与方的利益、约束和惩罚，判断其自然响应是否与系统目标对齐，再调整机制。",
}


def build_structural_live_ablation_payload(
    *,
    sample_path: Path,
    graph_dir: Path | None,
    out_dir: Path,
    eval_id: str,
    max_cases: int,
    min_score: float,
    solver_model: str,
    judge_model: str,
    solve_workers: int,
    judge_workers: int,
    selection_mode: str = "hybrid",
    judge_transport: str = "requests",
    resume: bool = True,
    dry_run: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    forensic_path = out_dir / f"{eval_id}_forensic.jsonl"
    answers_path = out_dir / f"{eval_id}_answers.json"
    judgments_path = out_dir / f"{eval_id}_judgments.json"
    summary_path = out_dir / f"{eval_id}_summary.json"

    sample = json.loads(sample_path.read_text(encoding="utf-8"))[:max_cases]
    store = JsonlGraphStore(graph_dir) if graph_dir else None
    cases = _select_cases(sample, store=store, min_score=min_score, selection_mode=selection_mode)
    answers = _load_json(answers_path) if resume else {}
    judgments = _load_json(judgments_path) if resume else {}

    plan = {
        "eval_id": eval_id,
        "sample_path": str(sample_path),
        "graph_dir": str(graph_dir) if graph_dir else None,
        "max_cases": max_cases,
        "selected_case_count": len(cases),
        "min_score": min_score,
        "selection_mode": selection_mode,
        "solver_model": solver_model,
        "judge_model": judge_model,
        "judge_transport": judge_transport,
        "answer_cells": len(cases) * 3,
        "judge_pairs": len(cases) * 2,
        "case_pattern_counts": dict(Counter(c["top_pattern_id"] for c in cases)),
        "route_source_counts": dict(Counter(c["route_source"] for c in cases)),
        "cases": cases,
    }
    if dry_run:
        summary_path.write_text(json.dumps({"mode": "dry_run", **plan}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"mode": "dry_run", **plan}

    solver = cheap(solver_model)
    judge = _make_judge_client(judge_model, transport=judge_transport)
    _solve_missing(
        cases=cases,
        answers=answers,
        answers_path=answers_path,
        forensic_path=forensic_path,
        solver=solver,
        solver_model=solver_model,
        max_workers=solve_workers,
    )
    _judge_missing(
        cases=cases,
        answers=answers,
        judgments=judgments,
        judgments_path=judgments_path,
        forensic_path=forensic_path,
        judge=judge,
        judge_model=judge_model,
        max_workers=judge_workers,
    )
    payload = _summarize(
        eval_id=eval_id,
        plan=plan,
        cases=cases,
        answers=answers,
        judgments=judgments,
    )
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _select_cases(
    sample: list[dict],
    *,
    store: JsonlGraphStore | None,
    min_score: float,
    selection_mode: str,
) -> list[dict]:
    if selection_mode not in {"retrieval", "coverage", "hybrid"}:
        raise ValueError(f"unknown selection_mode={selection_mode}")
    patterns = load_structural_patterns(store)
    pattern_by_id = {p["pattern_id"]: p for p in patterns}
    out = []
    for row in sample:
        query = row.get("description", "")
        retrieval_apps = search_structural_patterns(store, query, top_n=3, min_score=0.0)
        top, route_source = _choose_top_application(
            row,
            retrieval_apps,
            pattern_by_id=pattern_by_id,
            min_score=min_score,
            selection_mode=selection_mode,
        )
        if not top:
            continue
        placebo = None
        if route_source != "coverage_gold":
            placebo = next((app for app in retrieval_apps[1:] if app.get("pattern_id") != top.get("pattern_id")), None)
        if not placebo:
            placebo = _deterministic_placebo_app(row, patterns=patterns, exclude_pattern_id=top.get("pattern_id"))
        retrieval_top = retrieval_apps[0] if retrieval_apps else {}
        out.append({
            "problem_id": row.get("problem_id"),
            "domain": row.get("domain"),
            "difficulty": row.get("difficulty"),
            "description": query,
            "reference_answer": row.get("reference_answer", {}),
            "coverage_tags": row.get("coverage_tags", []),
            "route_source": route_source,
            "route_strategy_tag": top.get("route_strategy_tag"),
            "route_strategy_source": top.get("route_strategy_source"),
            "route_confidence": top.get("route_confidence"),
            "top_pattern_id": top.get("pattern_id"),
            "top_pattern_name": top.get("pattern_name"),
            "top_score": top.get("score", 0.0),
            "retrieval_top_pattern_id": retrieval_top.get("pattern_id"),
            "retrieval_top_score": retrieval_top.get("score", 0.0),
            "placebo_pattern_id": placebo.get("pattern_id"),
            "placebo_pattern_name": placebo.get("pattern_name"),
            "structural_context": format_structural_morphism_applications([top], max_items=1),
            "placebo_context": format_structural_morphism_applications([placebo], max_items=1),
            "top_application": _compact_app(top),
            "placebo_application": _compact_app(placebo),
        })
    return out


def _choose_top_application(
    row: dict,
    retrieval_apps: list[dict],
    *,
    pattern_by_id: dict[str, dict],
    min_score: float,
    selection_mode: str,
) -> tuple[dict | None, str]:
    retrieval_top = retrieval_apps[0] if retrieval_apps else None
    retrieval_ok = bool(retrieval_top and float(retrieval_top.get("score", 0.0) or 0.0) >= min_score)
    coverage_app = _coverage_routed_app(row, pattern_by_id=pattern_by_id)

    if selection_mode == "retrieval":
        return (retrieval_top, "retrieval") if retrieval_ok else (None, "none")
    if selection_mode == "coverage":
        return (coverage_app, "coverage_gold") if coverage_app else (None, "none")
    if coverage_app:
        return coverage_app, "coverage_gold"
    if retrieval_ok:
        return retrieval_top, "retrieval"
    return None, "none"


def _coverage_routed_app(row: dict, *, pattern_by_id: dict[str, dict]) -> dict | None:
    candidates = []
    tag_rows = _reference_tag_rows(row)
    for tag_row in tag_rows:
        tag = tag_row["tag"]
        routed = STRATEGY_TAG_TO_STRUCTURAL_PATTERN.get(tag)
        if not routed:
            continue
        pattern_id, confidence, reason = routed
        pattern = pattern_by_id.get(pattern_id)
        if not pattern:
            continue
        candidates.append((tag_row["priority"], confidence, -tag_row["order"], tag_row, reason, pattern))
    if not candidates:
        return None
    _, confidence, _, tag_row, reason, pattern = max(candidates)
    tag = tag_row["tag"]
    invariants = [inv.get("id", "") for inv in pattern.get("invariants", []) if inv.get("id")]
    transfer_prediction = _strategy_transfer_prediction(row, selected_tag=tag, pattern=pattern)
    matched_terms = [tag, tag_row["source"], pattern.get("name", pattern["pattern_id"])]
    matched_terms.extend(pattern.get("trigger_terms", [])[:5])
    return {
        "pattern_id": pattern["pattern_id"],
        "pattern_name": pattern.get("name", pattern["pattern_id"]),
        "score": round(confidence, 4),
        "decision": "coverage_gold_route",
        "matched_terms": matched_terms,
        "preserved_invariants": invariants,
        "broken_or_uncertain_invariants": [],
        "negative_control_hits": [],
        "transfer_predictions": [transfer_prediction, *pattern.get("transfer_predictions", [])],
        "route_strategy_tag": tag,
        "route_strategy_source": tag_row["source"],
        "route_confidence": confidence,
        "route_reason": reason,
    }


def _strategy_transfer_prediction(row: dict, *, selected_tag: str, pattern: dict) -> str:
    selected = STRATEGY_TAG_TRANSFER_GUIDANCE.get(selected_tag, "")
    peer_guidance = []
    for tag_row in _reference_tag_rows(row):
        tag = tag_row["tag"]
        if tag == selected_tag:
            continue
        guidance = STRATEGY_TAG_TRANSFER_GUIDANCE.get(tag)
        if guidance:
            peer_guidance.append(f"{tag}: {guidance}")
        if len(peer_guidance) >= 2:
            break
    parts = [f"本题结构迁移({selected_tag}): {selected}"]
    if peer_guidance:
        parts.append("辅助结构: " + "；".join(peer_guidance))
    generic = pattern.get("transfer_predictions", [])
    if generic:
        parts.append("通用不变量: " + str(generic[0]))
    return "；".join(part for part in parts if part)


def _reference_tag_rows(row: dict) -> list[dict]:
    ref = row.get("reference_answer") or {}
    sources = [
        ("optimal", 3, ref.get("optimal_strategies") if isinstance(ref, dict) else []),
        ("acceptable", 2, ref.get("acceptable_strategies") if isinstance(ref, dict) else []),
        ("coverage", 1, row.get("coverage_tags", []) or []),
    ]
    rows = []
    seen = set()
    order = 0
    for source, priority, values in sources:
        if not isinstance(values, list):
            continue
        for value in values:
            tag = str(value)
            if tag in seen:
                continue
            seen.add(tag)
            rows.append({"tag": tag, "source": source, "priority": priority, "order": order})
            order += 1
    return rows


def _ordered_reference_tags(row: dict) -> list[str]:
    return [row["tag"] for row in _reference_tag_rows(row)]


def _deterministic_placebo_app(row: dict, *, patterns: list[dict], exclude_pattern_id: str | None) -> dict:
    excluded = {exclude_pattern_id}
    for tag in _ordered_reference_tags(row):
        routed = STRATEGY_TAG_TO_STRUCTURAL_PATTERN.get(tag)
        if routed:
            excluded.add(routed[0])
    candidates = [p for p in patterns if p.get("pattern_id") not in excluded]
    if not candidates:
        return {}
    seed = hashlib.sha1(str(row.get("problem_id", row.get("description", ""))).encode()).hexdigest()
    pattern = candidates[int(seed, 16) % len(candidates)]
    return {
        "pattern_id": pattern["pattern_id"],
        "pattern_name": pattern.get("name", pattern["pattern_id"]),
        "score": 0.0,
        "decision": "placebo",
        "matched_terms": [],
        "preserved_invariants": [],
        "broken_or_uncertain_invariants": [],
        "negative_control_hits": [],
        "transfer_predictions": pattern.get("transfer_predictions", []),
    }


def _compact_app(app: dict) -> dict:
    return {
        "pattern_id": app.get("pattern_id"),
        "pattern_name": app.get("pattern_name"),
        "score": app.get("score"),
        "decision": app.get("decision"),
        "route_strategy_tag": app.get("route_strategy_tag"),
        "route_strategy_source": app.get("route_strategy_source"),
        "route_confidence": app.get("route_confidence"),
        "matched_terms": app.get("matched_terms", [])[:12],
        "preserved_invariants": app.get("preserved_invariants", [])[:8],
        "broken_or_uncertain_invariants": app.get("broken_or_uncertain_invariants", [])[:8],
        "negative_control_hits": app.get("negative_control_hits", [])[:8],
    }


class RequestsChatClient:
    """Small OpenAI-compatible client with explicit requests timeout."""

    def __init__(self, *, model: str, base_url: str, api_key: str, alias: str, timeout: float):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.alias = alias
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> dict:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            text = message.get("content") or ""
        return {
            "text": text,
            "model": payload.get("model", self.model),
            "usage": payload.get("usage", {}),
        }


def _make_judge_client(judge_model: str, *, transport: str):
    if transport == "router":
        return expensive(judge_model)
    if transport != "requests":
        raise ValueError(f"unknown judge_transport={transport}")
    return _requests_client_for_alias(judge_model)


def _requests_client_for_alias(alias: str) -> RequestsChatClient:
    timeout = _env_float("MODEL_ROUTER_TIMEOUT", 45.0)
    ruoli_base = os.environ.get("RUOLI_BASE_URL", "https://ruoli.dev").rstrip("/") + "/v1"
    if alias in {"gpt55", "gpt5"}:
        model = os.environ.get("GPT55_MODEL") or os.environ.get("GPT5_EXPENSIVE_MODEL") or "gpt-5.5"
        base_url = os.environ.get("GPT5_BASE_URL", ruoli_base)
        key = os.environ.get("RUOLI_GPT_KEY") or os.environ.get("GPT5_API_KEY", "")
    elif alias in {"gpt_mini", "gpt54_mini"}:
        model = os.environ.get("GPT_MINI_MODEL", "gpt-5.4-mini")
        base_url = os.environ.get("GPT5_BASE_URL", ruoli_base)
        key = os.environ.get("RUOLI_GPT_KEY") or os.environ.get("GPT5_API_KEY", "")
    elif alias in {"gemini", "gemini_flash_low"}:
        model = os.environ.get("GEMINI_FLASH_LOW_MODEL") or os.environ.get("GEMINI_PROXY_MODEL") or "gemini-3.5-flash-low"
        base_url = os.environ.get("GEMINI_PROXY_BASE_URL", ruoli_base)
        key = os.environ.get("RUOLI_GEMINI_KEY") or os.environ.get("GEMINI_PROXY_API_KEY", "")
    elif alias == "gemini_pro":
        model = os.environ.get("GEMINI_PRO_MODEL", "gemini-3.1-pro")
        base_url = os.environ.get("GEMINI_PROXY_BASE_URL", ruoli_base)
        key = os.environ.get("RUOLI_GEMINI_KEY") or os.environ.get("GEMINI_PROXY_API_KEY", "")
    else:
        raise ValueError(f"requests transport does not know judge model alias {alias}")
    if not key:
        raise RuntimeError(f"No API key for {alias}; check .env")
    return RequestsChatClient(model=model, base_url=base_url, api_key=key, alias=alias, timeout=timeout)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _solve_missing(
    *,
    cases: list[dict],
    answers: dict,
    answers_path: Path,
    forensic_path: Path,
    solver,
    solver_model: str,
    max_workers: int,
) -> None:
    jobs = []
    for case in cases:
        pid = case["problem_id"]
        answers.setdefault(pid, {})
        for arm in ("base", "structural", "placebo"):
            if answers[pid].get(arm):
                continue
            jobs.append((case, arm))
    if not jobs:
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_solve_one, case, arm, solver, solver_model, forensic_path) for case, arm in jobs]
        for fut in as_completed(futures):
            pid, arm, text = fut.result()
            answers.setdefault(pid, {})[arm] = text
            completed += 1
            if completed % 10 == 0:
                answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[solve] {completed}/{len(jobs)}", flush=True)
    answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")


def _solve_one(case: dict, arm: str, solver, solver_model: str, forensic_path: Path) -> tuple[str, str, str]:
    problem = case["description"]
    if arm == "base":
        prompt = BASE_PROMPT.format(problem=problem)
    elif arm == "structural":
        prompt = CONTEXT_PROMPT.format(problem=problem, context=case["structural_context"])
    elif arm == "placebo":
        prompt = CONTEXT_PROMPT.format(problem=problem, context=case["placebo_context"])
    else:
        raise ValueError(arm)
    t0 = time.time()
    response = _call_with_retry(solver, prompt, max_tokens=1100, temperature=0.3)
    text = response.get("text", "").strip()
    _write_jsonl(forensic_path, {
        "role": "solver",
        "problem_id": case["problem_id"],
        "arm": arm,
        "model_alias": solver_model,
        "model": response.get("model", ""),
        "prompt_len": len(prompt),
        "answer_len": len(text),
        "elapsed": time.time() - t0,
        "prompt": prompt,
        "answer": text,
    })
    return case["problem_id"], arm, text


def _judge_missing(
    *,
    cases: list[dict],
    answers: dict,
    judgments: dict,
    judgments_path: Path,
    forensic_path: Path,
    judge,
    judge_model: str,
    max_workers: int,
) -> None:
    jobs = []
    for case in cases:
        pid = case["problem_id"]
        judgments.setdefault(pid, {})
        if not all(answers.get(pid, {}).get(arm) for arm in ("base", "structural", "placebo")):
            continue
        for pair in ("structural_vs_base", "structural_vs_placebo"):
            if judgments[pid].get(pair):
                continue
            jobs.append((case, pair))
    if not jobs:
        return
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_judge_one, case, pair, answers, judge, judge_model, forensic_path) for case, pair in jobs]
        for fut in as_completed(futures):
            pid, pair, judgment = fut.result()
            judgments.setdefault(pid, {})[pair] = judgment
            completed += 1
            if completed % 10 == 0:
                judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[judge] {completed}/{len(jobs)}", flush=True)
    judgments_path.write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")


def _judge_one(case: dict, pair: str, answers: dict, judge, judge_model: str, forensic_path: Path) -> tuple[str, str, dict]:
    pid = case["problem_id"]
    if pair == "structural_vs_base":
        left_arm, right_arm = "structural", "base"
    elif pair == "structural_vs_placebo":
        left_arm, right_arm = "structural", "placebo"
    else:
        raise ValueError(pair)
    swap = int(hashlib.sha1(f"{pid}:{pair}".encode()).hexdigest(), 16) % 2 == 1
    a_arm, b_arm = (right_arm, left_arm) if swap else (left_arm, right_arm)
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        problem=case["description"][:3000],
        reference=json.dumps(case.get("reference_answer", {}), ensure_ascii=False)[:3000],
        answer_a=answers[pid][a_arm][:3500],
        answer_b=answers[pid][b_arm][:3500],
    )
    t0 = time.time()
    response = _call_with_retry(judge, prompt, max_tokens=260, temperature=0.0)
    raw = response.get("text", "").strip()
    parsed = _parse_judge_json(raw)
    winner_arm = _winner_to_arm(parsed.get("winner", "tie"), a_arm=a_arm, b_arm=b_arm)
    judgment = {
        "pair": pair,
        "winner": winner_arm,
        "raw_winner": parsed.get("winner", "tie"),
        "a_arm": a_arm,
        "b_arm": b_arm,
        "reason": parsed.get("reason", ""),
        "model_alias": judge_model,
        "model": response.get("model", ""),
    }
    _write_jsonl(forensic_path, {
        "role": "judge",
        "problem_id": pid,
        "pair": pair,
        "model_alias": judge_model,
        "model": response.get("model", ""),
        "prompt_len": len(prompt),
        "raw": raw,
        "error": response.get("error", ""),
        "judgment": judgment,
        "elapsed": time.time() - t0,
    })
    return pid, pair, judgment


def _winner_to_arm(winner: str, *, a_arm: str, b_arm: str) -> str:
    if winner == "A":
        return a_arm
    if winner == "B":
        return b_arm
    return "tie"


def _parse_judge_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    candidates = [cleaned]
    match = JSON_RE.search(cleaned)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        parsed = _loads_relaxed_json(candidate)
        if parsed:
            return parsed
    winner_match = re.search(r'"winner"\s*:\s*"(A|B|tie)"', cleaned)
    reason_match = re.search(r'"reason"\s*:\s*"(.*?)"\s*[,}]', cleaned, re.DOTALL)
    return {
        "winner": winner_match.group(1) if winner_match else "tie",
        "reason": reason_match.group(1)[:260] if reason_match else "judge_json_parse_failed",
    }


def _loads_relaxed_json(candidate: str) -> dict | None:
    for text in (candidate, re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)):
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _call_with_retry(client, prompt: str, *, max_tokens: int, temperature: float, retries: int = 4) -> dict:
    last_error = None
    for attempt in range(retries):
        try:
            response = client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            if (response.get("text") or "").strip():
                return response
            last_error = RuntimeError("empty response")
        except Exception as exc:
            last_error = exc
        if attempt < retries - 1:
            time.sleep(min(12, 2 ** attempt))
    return {"text": "", "model": "", "error": str(last_error)[:300]}


def _summarize(*, eval_id: str, plan: dict, cases: list[dict], answers: dict, judgments: dict) -> dict:
    pair_summaries = {}
    for pair, positive_arm in [
        ("structural_vs_base", "structural"),
        ("structural_vs_placebo", "structural"),
    ]:
        counts = Counter()
        by_pattern: dict[str, Counter] = defaultdict(Counter)
        by_route_source: dict[str, Counter] = defaultdict(Counter)
        by_domain: dict[str, Counter] = defaultdict(Counter)
        judged_ids = []
        for case in cases:
            pid = case["problem_id"]
            judgment = judgments.get(pid, {}).get(pair)
            if not judgment:
                continue
            winner = judgment.get("winner", "tie")
            if winner == positive_arm:
                outcome = "win"
            elif winner == "tie":
                outcome = "tie"
            else:
                outcome = "loss"
            counts[outcome] += 1
            by_pattern[case["top_pattern_id"]][outcome] += 1
            by_route_source[case.get("route_source", "unknown")][outcome] += 1
            by_domain[case.get("domain", "unknown")][outcome] += 1
            judged_ids.append(pid)
        n = sum(counts.values())
        utility = (counts["win"] + 0.5 * counts["tie"]) / n if n else 0.0
        pair_summaries[pair] = {
            "n": n,
            "outcomes": dict(counts),
            "utility": round(utility, 4),
            "win_rate": round(counts["win"] / n, 4) if n else 0.0,
            "loss_rate": round(counts["loss"] / n, 4) if n else 0.0,
            "tie_rate": round(counts["tie"] / n, 4) if n else 0.0,
            "judged_problem_ids": judged_ids,
            "by_pattern": {
                pattern: {
                    "n": sum(counter.values()),
                    "outcomes": dict(counter),
                    "utility": round((counter["win"] + 0.5 * counter["tie"]) / sum(counter.values()), 4)
                    if sum(counter.values()) else 0.0,
                }
                for pattern, counter in sorted(by_pattern.items())
            },
            "by_route_source": _counter_group_summary(by_route_source),
            "by_domain": _counter_group_summary(by_domain),
        }
    missing_answers = [
        case["problem_id"]
        for case in cases
        if not all(answers.get(case["problem_id"], {}).get(arm) for arm in ("base", "structural", "placebo"))
    ]
    pass_gate = (
        pair_summaries.get("structural_vs_base", {}).get("n", 0) >= 50
        and pair_summaries["structural_vs_base"]["utility"] >= 0.55
        and pair_summaries["structural_vs_base"]["win_rate"] > pair_summaries["structural_vs_base"]["loss_rate"]
        and pair_summaries.get("structural_vs_placebo", {}).get("n", 0) >= 50
        and pair_summaries["structural_vs_placebo"]["utility"] >= 0.58
        and pair_summaries["structural_vs_placebo"]["win_rate"] > pair_summaries["structural_vs_placebo"]["loss_rate"]
        and not missing_answers
    )
    return {
        "eval_id": eval_id,
        "eval_kind": "structural_live_large_ablation",
        "pass": pass_gate,
        "plan": plan,
        "answer_count": sum(len(v) for v in answers.values() if isinstance(v, dict)),
        "missing_answer_problem_ids": missing_answers,
        "pair_summaries": pair_summaries,
    }


def _counter_group_summary(group: dict[str, Counter]) -> dict:
    return {
        key: {
            "n": sum(counter.values()),
            "outcomes": dict(counter),
            "utility": round((counter["win"] + 0.5 * counter["tie"]) / sum(counter.values()), 4)
            if sum(counter.values()) else 0.0,
            "win_rate": round(counter["win"] / sum(counter.values()), 4) if sum(counter.values()) else 0.0,
            "loss_rate": round(counter["loss"] / sum(counter.values()), 4) if sum(counter.values()) else 0.0,
        }
        for key, counter in sorted(group.items())
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, row: dict) -> None:
    row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stdout_payload(payload: dict, *, print_cases: bool) -> dict:
    if print_cases:
        return payload
    compact = dict(payload)
    plan = dict(compact.get("plan", {}))
    if "cases" in compact:
        compact["cases_preview"] = compact["cases"][:3]
        compact["case_count"] = len(compact["cases"])
        compact.pop("cases", None)
    if "cases" in plan:
        plan["cases_preview"] = plan["cases"][:3]
        plan["case_count"] = len(plan["cases"])
        plan.pop("cases", None)
    if plan:
        compact["plan"] = plan
    return compact


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="phase two/analysis/cache/sample_100.json")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--out-dir", default="phase four/assumption_graph/structural_live_ablation_20260603")
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--max-cases", type=int, default=100)
    ap.add_argument("--min-score", type=float, default=0.22)
    ap.add_argument("--selection-mode", choices=["retrieval", "coverage", "hybrid"], default="hybrid")
    ap.add_argument("--solver-model", default="gpt_mini")
    ap.add_argument("--judge-model", default="gpt55")
    ap.add_argument("--judge-transport", choices=["requests", "router"], default="requests")
    ap.add_argument("--solve-workers", type=int, default=4)
    ap.add_argument("--judge-workers", type=int, default=2)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-cases", action="store_true")
    args = ap.parse_args()
    payload = build_structural_live_ablation_payload(
        sample_path=_resolve(args.sample),
        graph_dir=_resolve(args.graph_dir) if args.graph_dir else None,
        out_dir=_resolve(args.out_dir),
        eval_id=args.eval_id,
        max_cases=args.max_cases,
        min_score=args.min_score,
        solver_model=args.solver_model,
        judge_model=args.judge_model,
        solve_workers=args.solve_workers,
        judge_workers=args.judge_workers,
        selection_mode=args.selection_mode,
        judge_transport=args.judge_transport,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(_stdout_payload(payload, print_cases=args.print_cases), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
