"""
Precompute structural features for every problem using enhanced heuristics.

Produces phase one/cache/features.json keyed by problem_id. Features vary by
domain, difficulty, and keyword signals so the Phase 2 distiller has real
feature variation to threshold on.

Run once; the generator reads from this cache.
"""

import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
import _config as cfg


# Domain default bias (base features; further adjusted by keywords & difficulty)
DOMAIN_DEFAULTS = {
    "software_engineering": dict(coupling_estimate=0.55, decomposability=0.65, has_baseline=True,
                                 randomness_level=0.20, information_completeness=0.60,
                                 reversibility=0.70),
    "mathematics":          dict(coupling_estimate=0.40, decomposability=0.70, has_baseline=False,
                                 randomness_level=0.15, information_completeness=0.75,
                                 reversibility=0.85),
    "science":              dict(coupling_estimate=0.55, decomposability=0.60, has_baseline=True,
                                 randomness_level=0.50, information_completeness=0.55,
                                 reversibility=0.55),
    "business":             dict(coupling_estimate=0.60, decomposability=0.45, has_baseline=False,
                                 randomness_level=0.60, information_completeness=0.45,
                                 reversibility=0.35),
    "engineering":          dict(coupling_estimate=0.60, decomposability=0.55, has_baseline=True,
                                 randomness_level=0.35, information_completeness=0.55,
                                 reversibility=0.40),
    "daily_life":           dict(coupling_estimate=0.55, decomposability=0.40, has_baseline=False,
                                 randomness_level=0.55, information_completeness=0.40,
                                 reversibility=0.45),
}

DIFFICULTY_DELTA = {
    "easy":   dict(coupling=-0.15, decomposability=+0.10, info=+0.15, component=-2, constraint=-1),
    "medium": dict(coupling=0.0,   decomposability=0.0,   info=0.0,   component=0,  constraint=0),
    "hard":   dict(coupling=+0.15, decomposability=-0.10, info=-0.15, component=+3, constraint=+2),
}

KEYWORDS = {
    "coupling_up":      ["耦合", "相互影响", "交叉", "依赖", "牵一发", "联动", "连锁"],
    "coupling_down":    ["独立", "隔离", "解耦", "互不影响"],
    "decomp_up":        ["分解", "模块", "子问题", "步骤", "阶段", "部件", "组件"],
    "decomp_down":      ["整体", "一体", "不可分", "笼统"],
    "baseline_yes":     ["基准", "正常", "以前可以", "之前", "参考", "对照"],
    "random_up":        ["随机", "概率", "不确定", "偶然", "噪声", "波动", "无规律"],
    "random_down":      ["确定", "固定", "规律", "稳定", "重复"],
    "info_up":          ["数据充足", "详细", "清晰", "完整", "明确"],
    "info_down":        ["模糊", "信息不足", "缺少", "未知", "不清楚"],
    "reversible_up":    ["可逆", "可撤销", "可恢复", "可调整"],
    "reversible_down":  ["不可逆", "不能撤销", "一次性", "破坏性", "永久"],
}


def _count_hits(text: str, patterns):
    return sum(1 for p in patterns if p in text)


def extract(problem: dict) -> dict:
    domain = problem.get("domain", "daily_life")
    difficulty = problem.get("difficulty", "medium")
    desc = problem.get("description", "") or ""

    base = dict(DOMAIN_DEFAULTS.get(domain, DOMAIN_DEFAULTS["daily_life"]))
    delta = DIFFICULTY_DELTA.get(difficulty, DIFFICULTY_DELTA["medium"])

    coupling = base["coupling_estimate"] + delta["coupling"]
    coupling += 0.08 * _count_hits(desc, KEYWORDS["coupling_up"])
    coupling -= 0.08 * _count_hits(desc, KEYWORDS["coupling_down"])

    decomp = base["decomposability"] + delta["decomposability"]
    decomp += 0.07 * _count_hits(desc, KEYWORDS["decomp_up"])
    decomp -= 0.07 * _count_hits(desc, KEYWORDS["decomp_down"])

    has_baseline = base["has_baseline"] or _count_hits(desc, KEYWORDS["baseline_yes"]) > 0

    rnd = base["randomness_level"]
    rnd += 0.10 * _count_hits(desc, KEYWORDS["random_up"])
    rnd -= 0.08 * _count_hits(desc, KEYWORDS["random_down"])

    info = base["information_completeness"] + delta["info"]
    info += 0.08 * _count_hits(desc, KEYWORDS["info_up"])
    info -= 0.10 * _count_hits(desc, KEYWORDS["info_down"])

    rev = base["reversibility"]
    rev += 0.08 * _count_hits(desc, KEYWORDS["reversible_up"])
    rev -= 0.12 * _count_hits(desc, KEYWORDS["reversible_down"])

    component_count = max(1, 5 + delta["component"] + len(re.findall(r"[、；;]", desc)) // 3)
    constraint_count = max(0, 3 + delta["constraint"] + desc.count("必须") + desc.count("不能"))

    clamp = lambda x: max(0.0, min(1.0, x))

    return {
        "domain": domain,
        "coupling_estimate": round(clamp(coupling), 3),
        "decomposability": round(clamp(decomp), 3),
        "has_baseline": bool(has_baseline),
        "randomness_level": round(clamp(rnd), 3),
        "information_completeness": round(clamp(info), 3),
        "component_count": int(component_count),
        "constraint_count": int(constraint_count),
        "reversibility": round(clamp(rev), 3),
        "difficulty": difficulty,
    }


def main():
    problems = []
    for f in sorted(cfg.PROBLEMS_DIR.glob("*.json")):
        if "error" in f.name:
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            problems.extend(data)

    features = {p["problem_id"]: extract(p) for p in problems}

    out = PROJECT / "cache" / "features.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(features, ensure_ascii=False, indent=2))

    # Distribution report
    import statistics
    for key in ["coupling_estimate", "decomposability", "randomness_level",
                "information_completeness", "reversibility"]:
        vals = [f[key] for f in features.values()]
        print(f"  {key:>28}: mean={statistics.mean(vals):.2f} "
              f"min={min(vals):.2f} max={max(vals):.2f} "
              f"n_high(>=.65)={sum(v >= .65 for v in vals)} "
              f"n_low(<=.35)={sum(v <= .35 for v in vals)}")
    print(f"\n  Saved {len(features)} features to {out}")


if __name__ == "__main__":
    main()
