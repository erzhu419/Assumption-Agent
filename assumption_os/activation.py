"""Activation profiles for conditioned assumption evaluation.

Retrieval can be broad; evaluation routing should be stricter.  An activation
profile captures the subset where a node claims it should help.  Strategy nodes
prefer explicit coverage tags.  Wisdom nodes prefer trigger-signal keywords and
cross-domain examples.  Generic nodes may still fall back to lexical routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import AssumptionNode


CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
LATIN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")

STOP_CJK = {
    "一个", "一种", "这个", "那个", "这些", "那些", "自己", "当前", "现在", "时候", "如果",
    "因为", "因此", "先把", "是否", "已经", "需要", "问题", "情况", "进行", "判断",
    "分析", "解决", "目标", "激活", "这条", "应当", "不要", "而是", "不是", "没有",
    "什么", "如何", "可以", "不能", "开始", "之间", "面对", "使用",
}


@dataclass(frozen=True)
class ActivationProfile:
    node_id: str
    family: str
    strategy_code: str | None = None
    domains: set[str] = field(default_factory=set)
    excluded_domains: set[str] = field(default_factory=set)
    difficulties: set[str] = field(default_factory=set)
    problem_ids: set[str] = field(default_factory=set)
    coverage_tags: set[str] = field(default_factory=set)
    keywords: tuple[str, ...] = ()
    min_keyword_hits: int = 2
    allow_lexical_fallback: bool = False


def build_activation_profile(node: AssumptionNode) -> ActivationProfile:
    activation = node.payload.get("activation", {}) if isinstance(node.payload, dict) else {}
    family = _node_family(node)
    explicit_keywords = tuple(str(x).lower() for x in activation.get("keywords", []) if str(x).strip())
    strategy_code = _strategy_code(node)
    coverage_tags = set(str(x) for x in activation.get("coverage_tags", []))
    if strategy_code:
        coverage_tags.add(strategy_code)

    if family == "wisdom":
        domains = {
            str(case.get("domain"))
            for case in (node.payload.get("cross_domain_examples", []) if isinstance(node.payload, dict) else [])
            if case.get("domain")
        }
        keywords = explicit_keywords or tuple(_wisdom_keywords(node))
        return ActivationProfile(
            node_id=node.id,
            family=family,
            domains=domains | set(activation.get("domains", [])),
            excluded_domains=set(activation.get("excluded_domains", [])),
            difficulties=set(activation.get("difficulties", [])),
            problem_ids=set(activation.get("problem_ids", [])),
            coverage_tags=coverage_tags,
            keywords=keywords,
            min_keyword_hits=int(activation.get("min_keyword_hits", 2)),
            allow_lexical_fallback=False,
        )

    if family == "strategy":
        return ActivationProfile(
            node_id=node.id,
            family=family,
            strategy_code=strategy_code,
            domains=set(activation.get("domains", [])),
            excluded_domains=set(activation.get("excluded_domains", [])),
            difficulties=set(activation.get("difficulties", [])),
            problem_ids=set(activation.get("problem_ids", [])),
            coverage_tags=coverage_tags,
            keywords=explicit_keywords,
            min_keyword_hits=int(activation.get("min_keyword_hits", 2)),
            allow_lexical_fallback=False,
        )

    tag_domains = {str(t).split(":", 1)[1] for t in node.tags if str(t).startswith("domain:")}
    return ActivationProfile(
        node_id=node.id,
        family=family,
        domains=tag_domains | set(activation.get("domains", [])),
        excluded_domains=set(activation.get("excluded_domains", [])),
        difficulties=set(activation.get("difficulties", [])),
        problem_ids=set(activation.get("problem_ids", [])),
        coverage_tags=coverage_tags,
        keywords=explicit_keywords,
        min_keyword_hits=int(activation.get("min_keyword_hits", 2)),
        allow_lexical_fallback=True,
    )


def keyword_hit_count(profile: ActivationProfile, text: str) -> int:
    low = text.lower()
    return sum(1 for keyword in profile.keywords if keyword and keyword in low)


def _node_family(node: AssumptionNode) -> str:
    tag_set = {str(t).lower() for t in node.tags}
    if node.id.startswith("wisdom_") or "wisdom" in tag_set:
        return "wisdom"
    if _strategy_code(node):
        return "strategy"
    return str(node.type.value if hasattr(node.type, "value") else node.type)


def _strategy_code(node: AssumptionNode) -> str | None:
    if node.id.startswith("strategy_"):
        return node.id.split("_", 1)[1]
    for tag in node.tags:
        tag = str(tag)
        if len(tag) >= 2 and tag[0].upper() == "S" and tag[1:].isdigit():
            return tag
    return None


def _wisdom_keywords(node: AssumptionNode) -> list[str]:
    payload = node.payload if isinstance(node.payload, dict) else {}
    parts = [
        payload.get("signal", ""),
        payload.get("unpacked_for_llm", ""),
        payload.get("aphorism", ""),
        node.claim,
    ]
    for case in payload.get("cross_domain_examples", []) or []:
        parts.append(case.get("scenario", ""))
    return _extract_keywords("\n".join(str(p) for p in parts if p))


def _extract_keywords(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for word in LATIN_RE.findall(text.lower()):
        if word not in seen:
            seen.add(word)
            out.append(word)

    for run in CJK_RE.findall(text):
        chars = list(run)
        for n in (4, 3, 2):
            for i in range(0, max(0, len(chars) - n + 1)):
                gram = "".join(chars[i:i + n])
                if gram in STOP_CJK:
                    continue
                if _too_generic_cjk(gram):
                    continue
                if gram not in seen:
                    seen.add(gram)
                    out.append(gram)
    return out[:200]


def _too_generic_cjk(gram: str) -> bool:
    generic_chars = set("的了一是在和与及或但若把被为对中上下来去时先后更最这那其")
    return sum(1 for ch in gram if ch in generic_chars) >= max(1, len(gram) - 1)
