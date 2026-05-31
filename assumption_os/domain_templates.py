"""Domain execution templates for Phase2 prompt interventions.

These templates are intentionally separate from graph retrieval.  They encode
answer-level execution constraints that were repeatedly useful in heldout
software-engineering failures, without attributing the effect to any retrieved
assumption node.
"""

from __future__ import annotations

from .retrieval_policy import software_route


SOFTWARE_COMMON_CHECKS = [
    "先给出可验收的目标、Go/No-Go 阈值、决策 owner；不要只给原则。",
    "把技术可行性、用户/业务影响、安全/合规、运维归属分开判断。",
    "包含分阶段 rollout、rollback/kill-switch、监控指标和上线后复核。",
    "如果建议实验或 spike，写清样本、环境、成功/失败阈值和下一步动作。",
]


SOFTWARE_ROUTE_CHECKS = {
    "release_quality": [
        "按玩家影响、收入面、回归范围、数据/安全风险给缺陷分级。",
        "明确 release gate、必须修复项、可降级/绕过项、延期或热修策略。",
    ],
    "adapter_discovery": [
        "先限定授权边界和禁止事项，再做 capability matrix。",
        "写出协议/线缆/API 探测步骤、adapter contract、MVP demo 标准和 fallback。",
    ],
    "platform_migration": [
        "区分 license 成本、迁移 TCO、路径依赖、团队能力和发布风险。",
        "用 spike/prototype、dual-run 或分阶段迁移验证，并给 no-go 阈值。",
    ],
    "online_offline_gap": [
        "列出 leakage、distribution drift、logging-serving skew、metric mismatch、实验设计五类假设。",
        "要求 controlled reproduction、canary 分析和单因素干预，再谈模型改动。",
    ],
    "performance_regression": [
        "先固定 workload、数据快照、环境、版本和 instrumentation，建立可复现 baseline。",
        "用 bisect/revert/cherry-pick 或等价方法确认 causal commit。",
    ],
    "legacy_increment": [
        "避免先重写；先加 characterization tests、observability、data contract 和 manual override。",
        "给 sidecar/MVP 或 strangler path，并确保每步可回滚。",
    ],
    "tech_choice": [
        "先按安全、合规、可运维性、审计、团队 ramp cost、长期 owner 设约束。",
        "要求 time-boxed spike 覆盖失败模式，而不是按语言偏好选型。",
    ],
    "startup_gtm": [
        "先选窄 wedge customer 和一个可度量工作流，不要直接泛化到大市场。",
        "显式处理信任、责任边界、集成、报销/付费和采用指标。",
    ],
    "safety_latency": [
        "把安全当硬实时约束，舒适性只是该约束下的优化目标。",
        "说明 arbitration、fallback、tail-latency budget 和场景验证。",
    ],
}


def format_phase2_domain_execution_template(domain: str, problem: str, meta: dict | None = None) -> str:
    """Return an optional domain execution block for Phase2 v20 prompts."""

    if domain != "software_engineering":
        return ""
    meta = meta or {}
    query = "\n".join([
        problem,
        meta.get("critical_reframe", ""),
        meta.get("rewritten_problem", ""),
        meta.get("what_changed", ""),
    ])
    route = software_route(query)
    route_checks = SOFTWARE_ROUTE_CHECKS.get(route, [
        "优先给执行细节：里程碑、验证数据、失败阈值、回滚路径和责任人。",
        "避免抽象治理建议；每条建议都要能被工程团队执行或否决。",
    ])
    lines = [
        "### Software Engineering execution template",
        "仅作为执行约束；若与 PRIMARY FRAME 冲突，服从 PRIMARY FRAME。",
        f"- route: {route}",
    ]
    lines.extend(f"- {check}" for check in [*SOFTWARE_COMMON_CHECKS, *route_checks])
    return "\n".join(lines)
