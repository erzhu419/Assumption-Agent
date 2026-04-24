"""Clean data API for agent-generated gate code (Exp 21).

Purpose: hide the project's idiosyncratic file naming from the agent.
Agent only needs:
  - list_candidates()          -> list of candidate_ids
  - candidate_info(cid)        -> {cid, aphorism, wid, unpacked, signal}
  - per_pid_records(cid)       -> list of {pid, problem, domain, difficulty,
                                            base_what_changed, ext_what_changed,
                                            base_anti_patterns, ext_anti_patterns,
                                            base_answer, ext_answer}
"""

import json
from pathlib import Path

PROJECT = Path(__file__).parent.parent
CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO = PROJECT / "phase four" / "autonomous"


def _load(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


_CANDIDATES = [
    {"cid": "WCAND01", "aphorism": "上工治未病，不治已病", "base": "_valp_v20p1_base"},
    {"cid": "WCAND02", "aphorism": "别高效解决一个被看错的问题",
     "base": "_valp_v20p1_base"},
    {"cid": "WCAND03", "aphorism": "凡事预则立，不预则废", "base": "_valp_v20p1_base"},
    {"cid": "WCAND04", "aphorism": "急则治其标，缓则治其本", "base": "_valp_v20p1_base"},
    {"cid": "WCAND05", "aphorism": "凡益之道，与时偕行",
     "base": "_valp_v20p1_base", "wid": "W076"},
    {"cid": "WCAND06", "aphorism": "覆水难收，向前算账", "base": "_valp_v20p1_base"},
    {"cid": "WCAND07", "aphorism": "亲兄弟，明算账", "base": "_valp_v20p1_base"},
    {"cid": "WCAND08", "aphorism": "想理解行为，先看激励", "base": "_valp_v20p1_base"},
    {"cid": "WCAND09", "aphorism": "不谋全局者，不足谋一域",
     "base": "_valp_v20p1_base"},
    {"cid": "WCAND10", "aphorism": "没有调查，就没有发言权",
     "base": "_valp_v20p1_base", "wid": "W077"},
    {"cid": "WCAND11", "aphorism": "若不是品牌，你就只是商品。",
     "base": "_valp_v20p1_base"},
    {"cid": "WCROSSL01", "aphorism": "是骡子是马，拉出来遛遛",
     "base": "_valp_v20_base", "wid": "W078"},
]


def list_candidates():
    return [c["cid"] for c in _CANDIDATES]


def _candidate_row(cid):
    for c in _CANDIDATES:
        if c["cid"] == cid: return c
    raise KeyError(cid)


def _load_wisdom_record(aphorism):
    for src in ("success_distilled_candidates.json", "cross_llm_candidates.json"):
        data = _load(AUTO / src, default=[])
        for c in data:
            if c.get("aphorism", "").strip() == aphorism.strip():
                return c
    return None


def candidate_info(cid):
    row = _candidate_row(cid)
    rec = _load_wisdom_record(row["aphorism"]) or {}
    return {
        "cid": cid, "aphorism": row["aphorism"],
        "wid": row.get("wid"),
        "unpacked": rec.get("unpacked_for_llm", ""),
        "signal": rec.get("signal", ""),
        "source": rec.get("source", ""),
    }


def per_pid_records(cid):
    row = _candidate_row(cid)
    base_stem = row["base"]
    base_meta = _load(CACHE / "answers" / f"{base_stem}_meta.json", default={})
    ext_meta = _load(CACHE / "answers" / f"_valp_v20_ext_{cid}_meta.json", default={})
    base_ans = _load(CACHE / "answers" / f"{base_stem}_answers.json", default={})
    ext_ans = _load(CACHE / "answers" / f"_valp_v20_ext_{cid}_answers.json", default={})
    problems = _load(CACHE / "sample_holdout_50.json", default=[])
    pid_to_prob = {p["problem_id"]: p for p in problems if "description" in p}

    out = []
    for pid in sorted(set(base_meta.keys()) & set(ext_meta.keys())):
        if pid not in pid_to_prob: continue
        b = base_meta[pid] or {}
        e = ext_meta[pid] or {}
        p = pid_to_prob[pid]
        out.append({
            "pid": pid,
            "problem": p.get("description", ""),
            "domain": p.get("domain", ""),
            "difficulty": p.get("difficulty", ""),
            "base_what_changed": b.get("what_changed", ""),
            "ext_what_changed": e.get("what_changed", ""),
            "base_anti_patterns": b.get("anti_patterns", []),
            "ext_anti_patterns": e.get("anti_patterns", []),
            "base_answer": base_ans.get(pid, ""),
            "ext_answer": ext_ans.get(pid, ""),
        })
    return out


if __name__ == "__main__":
    print(f"Candidates: {len(list_candidates())}")
    info = candidate_info("WCAND05")
    print(f"WCAND05: {info['aphorism']} (wid={info['wid']})")
    rows = per_pid_records("WCAND05")
    print(f"per_pid_records: {len(rows)} rows")
    if rows:
        r = rows[0]
        print(f"  sample: pid={r['pid']} domain={r['domain']}")
        print(f"    base_wc: {r['base_what_changed'][:60]}")
        print(f"    ext_wc: {r['ext_what_changed'][:60]}")
        print(f"    ext_answer: {r['ext_answer'][:60]}")
