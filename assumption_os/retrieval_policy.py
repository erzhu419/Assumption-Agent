"""Domain-aware Assumption Graph retrieval policies.

The base graph retrieval is intentionally generic.  Some domains need a second
policy layer before prompt injection.  In software engineering, the n=21-50
heldout audit showed that generic strategy retrieval often produced plausible
but overly broad context; winning answers needed concrete acceptance metrics,
rollout/rollback paths, adapter boundaries, and MVP validation criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .formal_mapping import (
    build_formal_mapping_payload,
    format_formal_mapping_applications,
    search_formal_mappings,
)
from .schema import ActivatedSubgraph, AssumptionNode, AssumptionType


PRIMARY_TYPES = {
    AssumptionType.METHOD,
    AssumptionType.HARNESS,
    AssumptionType.RETRIEVAL,
    AssumptionType.WORLD_MODEL,
    AssumptionType.EVALUATOR,
    AssumptionType.ALIGNMENT,
}


@dataclass
class RetrievalPolicyResult:
    subgraph: ActivatedSubgraph
    policy_notes: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    formal_mapping_applications: list[dict] = field(default_factory=list)


def retrieve_phase2_assumptions(
    graph,
    *,
    problem: str,
    meta: dict,
    pid: str,
    domain: str,
    difficulty: str,
    top_k: int = 8,
    pool_k: int = 24,
    skip_domains: set[str] | None = None,
) -> RetrievalPolicyResult | None:
    """Retrieve and rerank assumptions for Phase2 v20 prompt injection."""

    if skip_domains and domain in skip_domains:
        return None

    query = "\n".join([
        problem,
        meta.get("critical_reframe", ""),
        meta.get("rewritten_problem", ""),
        meta.get("what_changed", ""),
    ])
    seeds = [
        pid,
        domain,
        difficulty,
        meta.get("frame", ""),
        *meta.get("anti_patterns", [])[:3],
    ]
    subgraph = graph.retrieve(query, seeds=seeds, top_k=max(top_k, pool_k), candidate_types=PRIMARY_TYPES)
    formal_mapping_applications = search_formal_mappings(
        build_formal_mapping_payload(graph.store),
        query,
        top_n=2,
    )
    if domain == "software_engineering":
        result = _rerank_software_engineering(subgraph, query=query, top_k=top_k)
        result.formal_mapping_applications = formal_mapping_applications
        result.diagnostics["formal_mapping_hits"] = [
            app["source_key"] for app in formal_mapping_applications
        ]
        return result
    return RetrievalPolicyResult(
        subgraph=_slice_subgraph(subgraph, top_k),
        diagnostics={
            "policy": "generic_primary",
            "domain": domain,
            "formal_mapping_hits": [app["source_key"] for app in formal_mapping_applications],
        },
        formal_mapping_applications=formal_mapping_applications,
    )


def format_policy_context(result: RetrievalPolicyResult | None, formatter, *, max_nodes: int = 8) -> str:
    if not result:
        return ""
    text = formatter(result.subgraph, max_nodes=max_nodes)
    if not result.policy_notes:
        formal_text = format_formal_mapping_applications(result.formal_mapping_applications)
        return "\n\n".join(x for x in [text, formal_text] if x).strip()
    lines = [
        text,
        "",
        "## Domain Execution Checks",
        "Treat these as acceptance constraints for using the assumptions above.",
    ]
    lines.extend(f"- {note}" for note in result.policy_notes)
    formal_text = format_formal_mapping_applications(result.formal_mapping_applications)
    if formal_text:
        lines.extend(["", formal_text])
    return "\n".join(lines).strip()


def software_route(query: str) -> str:
    """Classify a software-engineering problem into a prompt policy route."""

    return _software_route(query)


def _rerank_software_engineering(
    subgraph: ActivatedSubgraph,
    *,
    query: str,
    top_k: int,
) -> RetrievalPolicyResult:
    route = _software_route(query)
    boosts = _software_boost_map(route)
    adjusted = {}
    for node in subgraph.nodes:
        base = subgraph.scores.get(node.id, 0.0)
        adjusted[node.id] = base + boosts.get(node.id, 0.0) + _software_text_bonus(node, query)
    ranked = sorted(subgraph.nodes, key=lambda n: adjusted[n.id], reverse=True)[:top_k]
    ranked_ids = {node.id for node in ranked}
    reranked = ActivatedSubgraph(
        query=subgraph.query,
        seed_ids=subgraph.seed_ids,
        nodes=ranked,
        edges=[e for e in subgraph.edges if e.source in ranked_ids and e.target in ranked_ids],
        scores={node.id: adjusted[node.id] for node in ranked},
        cases=[n for n in ranked if n.type == AssumptionType.CASE],
        residuals=[n for n in ranked if n.type == AssumptionType.RESIDUAL],
        verifiers=[n for n in ranked if n.type == AssumptionType.VERIFIER],
    )
    return RetrievalPolicyResult(
        subgraph=reranked,
        policy_notes=_software_policy_notes(route),
        diagnostics={
            "policy": "software_engineering_rerank",
            "route": route,
            "boosted_ids": {k: v for k, v in sorted(boosts.items()) if v > 0},
        },
    )


def _slice_subgraph(subgraph: ActivatedSubgraph, top_k: int) -> ActivatedSubgraph:
    nodes = subgraph.nodes[:top_k]
    ids = {n.id for n in nodes}
    return ActivatedSubgraph(
        query=subgraph.query,
        seed_ids=subgraph.seed_ids,
        nodes=nodes,
        edges=[e for e in subgraph.edges if e.source in ids and e.target in ids],
        scores={n.id: subgraph.scores[n.id] for n in nodes if n.id in subgraph.scores},
        cases=[n for n in nodes if n.type == AssumptionType.CASE],
        residuals=[n for n in nodes if n.type == AssumptionType.RESIDUAL],
        verifiers=[n for n in nodes if n.type == AssumptionType.VERIFIER],
    )


def _software_route(query: str) -> str:
    text = query.lower()
    routes = [
        ("performance_regression", ["性能下降", "性能回退", "存储过程", "批量交易", "执行时间异常", "哪一次提交", "提交导致"]),
        ("platform_migration", ["迁移", "unity", "虚幻", "scada", "wincc", "ignition", "资产商店"]),
        ("online_offline_gap", ["a/b", "ab测试", "推荐系统", "准确率", "点击率", "转化率", "数据泄露", "漂移", "线上", "线下"]),
        ("tech_choice", ["kms", "elixir", "go语言", "go 语言", "技术选型"]),
        ("legacy_increment", ["遗留", "脚本", "重构", "测试覆盖", "ci/cd", "wms", "库存", "硬编码"]),
        ("startup_gtm", ["医疗", "医院", "医生", "报销", "首类客户", "商业化", "生态", "责任边界"]),
        ("adapter_discovery", ["api", "adapter", "适配器", "端口", "ip地址", "未公开", "sdk", "兼容"]),
        ("safety_latency", ["自动驾驶", "延迟", "毫秒", "路径规划", "紧急刹车", "舒适"]),
        ("release_quality", ["bug", "qa", "崩溃", "商城", "玩家留存", "pvp", "boss"]),
    ]
    for route, needles in routes:
        if any(n in text for n in needles):
            return route
    return "software_general"


def _software_boost_map(route: str) -> dict[str, float]:
    base = {
        "strategy_S14": 0.025,
        "strategy_S15": 0.025,
        "strategy_S21": 0.025,
        "strategy_S23": 0.02,
        "strategy_S24": 0.02,
    }
    route_boosts = {
        "release_quality": {
            "strategy_S24": 0.16,
            "strategy_S12": 0.18,
            "strategy_S11": 0.16,
            "strategy_S14": 0.08,
            "strategy_S21": 0.07,
            "strategy_S23": 0.05,
            "strategy_S22": -0.08,
        },
        "adapter_discovery": {
            "strategy_S08": 0.16,
            "strategy_S14": 0.14,
            "strategy_S15": 0.08,
            "strategy_S13": 0.07,
            "strategy_S23": 0.05,
            "strategy_S01": 0.04,
        },
        "platform_migration": {
            "strategy_S26": 0.16,
            "strategy_S27": 0.14,
            "strategy_S21": 0.12,
            "strategy_S24": 0.09,
            "strategy_S15": 0.06,
            "strategy_S23": 0.05,
        },
        "online_offline_gap": {
            "strategy_S13": 0.16,
            "strategy_S17": 0.14,
            "strategy_S01": 0.13,
            "strategy_S12": 0.09,
            "strategy_S14": 0.05,
        },
        "performance_regression": {
            "strategy_S01": 0.17,
            "strategy_S17": 0.15,
            "strategy_S13": 0.12,
            "strategy_S12": 0.08,
            "strategy_S14": 0.06,
        },
        "legacy_increment": {
            "strategy_S15": 0.16,
            "strategy_S06": 0.12,
            "strategy_S09": 0.11,
            "strategy_S23": 0.08,
            "strategy_S21": 0.06,
            "strategy_S24": 0.05,
        },
        "tech_choice": {
            "strategy_S05": 0.16,
            "strategy_S11": 0.14,
            "strategy_S14": 0.08,
            "strategy_S21": 0.06,
            "strategy_S23": 0.05,
        },
        "startup_gtm": {
            "strategy_S21": 0.15,
            "strategy_S23": 0.13,
            "strategy_S27": 0.08,
            "strategy_S24": 0.06,
            "strategy_S15": 0.04,
        },
        "safety_latency": {
            "strategy_S20": 0.16,
            "strategy_S19": 0.13,
            "strategy_S22": 0.11,
            "strategy_S14": 0.08,
            "strategy_S21": 0.05,
        },
    }
    out = dict(base)
    for node_id, boost in route_boosts.get(route, {}).items():
        out[node_id] = out.get(node_id, 0.0) + boost
    return out


def _software_text_bonus(node: AssumptionNode, query: str) -> float:
    text = query.lower()
    node_text = " ".join([node.claim, " ".join(node.tags), " ".join(node.context_conditions)]).lower()
    bonus = 0.0
    if "回滚" in text or "rollback" in text or "降级" in text:
        if any(token in node_text for token in ["增量", "边界", "止损", "resource", "约束"]):
            bonus += 0.025
    if "指标" in text or "验收" in text or "阈值" in text or "metric" in text:
        if any(token in node_text for token in ["边界", "满意", "瓶颈", "falsification", "证伪"]):
            bonus += 0.025
    if "mvp" in text or "试点" in text or "灰度" in text:
        if any(token in node_text for token in ["增量", "试错", "近似", "controlled"]):
            bonus += 0.025
    return bonus


def _software_policy_notes(route: str) -> list[str]:
    common = [
        "Name concrete acceptance metrics, Go/No-Go thresholds, and who owns the decision.",
        "Include staged rollout, rollback/kill-switch, monitoring, and post-release verification.",
    ]
    route_notes = {
        "release_quality": [
            "Rank defects by player-impact/revenue/safety surface, not only generic severity.",
            "Give release gates, regression scope, downgrade/workaround options, and escalation thresholds.",
        ],
        "adapter_discovery": [
            "Stay inside legal/authorization boundaries; build a capability matrix before implementation.",
            "Specify the adapter contract, protocol discovery steps, MVP demo criteria, and fallback path.",
        ],
        "platform_migration": [
            "Separate license cost from total migration cost, path dependency, team skill, and release risk.",
            "Use spike/prototype evidence, dual-run or phased migration, and explicit no-go thresholds.",
        ],
        "online_offline_gap": [
            "List competing hypotheses: leakage, distribution drift, logging/serving skew, metric mismatch, and experiment design.",
            "Use controlled reproduction, canary analysis, and one-factor interventions before changing the model.",
        ],
        "performance_regression": [
            "Establish a reproducible baseline, then isolate the causal commit with bisect/revert/cherry-pick confirmation.",
            "Control workload, data snapshot, environment, and instrumentation before declaring the culprit.",
        ],
        "legacy_increment": [
            "Do not rewrite first; isolate the legacy system with tests, observability, and a sidecar/MVP path.",
            "Define reversible increments, data contracts, and manual override before automation expands.",
        ],
        "tech_choice": [
            "Choose against security/compliance/operability constraints before language aesthetics.",
            "Require a time-boxed spike covering failure modes, auditability, team ramp cost, and long-term ownership.",
        ],
        "startup_gtm": [
            "Pick a narrow wedge customer and measurable workflow before broad commercialization.",
            "Make trust, liability, integration, reimbursement, and adoption metrics explicit.",
        ],
        "safety_latency": [
            "Treat safety as a hard real-time constraint and comfort as an optimization under that constraint.",
            "Specify arbitration, fallback behavior, tail-latency budgets, and scenario-based validation.",
        ],
    }
    return [*common, *route_notes.get(route, [
        "Prefer execution detail over abstract governance: milestones, validation data, rollback, and failure thresholds.",
    ])]
