"""Exp 14 — Fragment decomposition + refined faithfulness.

Hypothesis (architectural change #1): the unit of retrieval is too
coarse. Each wisdom is a 60-120 char `unpacked_for_llm` that contains
multiple distinct claims (a trigger, a set of key moves, a canonical
anti-pattern). Perturbation-alignment to the whole wisdom is near
zero (Exp 9/13); perhaps alignment to a SPECIFIC fragment is high.

We:
  1. Use GPT-5.4 to decompose each wisdom into 3 fragments:
       TRIGGER   — "what problem pattern does this wisdom fire on"
       MOVE      — "what cognitive move does it prescribe"
       ANTIPAT   — "what failure mode does it warn against"
  2. Embed each fragment.
  3. For each (wisdom, pid) pair, compute cosine alignment of the
     Turn-0 perturbation direction with EACH fragment separately and
     take the MAXIMUM. Report per-fragment alignment distribution.
  4. If max-over-fragment alignment is materially higher than
     whole-wisdom alignment (Exp 9), it is evidence that the
     retrieval unit is too coarse. If it is also low, the problem is
     deeper than unit granularity.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase four"))

from sentence_transformers import SentenceTransformer
from gpt5_client import GPT5Client
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
FRAGMENTS_PATH = AUTO_DIR / "exp14_fragments.json"
OUT_LOG = AUTO_DIR / "exp14_faithfulness_log.json"

CANDIDATES = [
    {"cid": "WCAND01", "aphorism": "上工治未病，不治已病"},
    {"cid": "WCAND02", "aphorism": "别高效解决一个被看错的问题"},
    {"cid": "WCAND03", "aphorism": "凡事预则立，不预则废"},
    {"cid": "WCAND04", "aphorism": "急则治其标，缓则治其本"},
    {"cid": "WCAND05", "aphorism": "凡益之道，与时偕行", "committed_id": "W076"},
    {"cid": "WCAND06", "aphorism": "覆水难收，向前算账"},
    {"cid": "WCAND07", "aphorism": "亲兄弟，明算账"},
    {"cid": "WCAND08", "aphorism": "想理解行为，先看激励"},
    {"cid": "WCAND09", "aphorism": "不谋全局者，不足谋一域"},
    {"cid": "WCAND10", "aphorism": "没有调查，就没有发言权", "committed_id": "W077"},
    {"cid": "WCAND11", "aphorism": "若不是品牌，你就只是商品。"},
    {"cid": "WCROSSL01", "aphorism": "是骡子是马，拉出来遛遛",
     "committed_id": "W078"},
]

DECOMP_PROMPT = """把下面这条 wisdom 拆成 3 条**可独立检索**的 fragment。

aphorism: {aphorism}
unpacked_for_llm: {unpacked}
signal: {signal}

三条 fragment:
  TRIGGER: 这条 wisdom 在什么问题模式上触发（30-50 字，描述问题特征，不写"做什么"）
  MOVE   : 它规定的具体认知动作（30-50 字，描述"做什么"，不写"为什么"）
  ANTIPAT: 它警告的典型错误（30-50 字，描述失败模式）

输出 JSON (不要代码块):
{{"TRIGGER": "...", "MOVE": "...", "ANTIPAT": "..."}}
"""


def load_candidate_record(aphorism):
    for src in ("success_distilled_candidates.json", "cross_llm_candidates.json"):
        p = AUTO_DIR / src
        if not p.exists(): continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for c in data:
            if c.get("aphorism", "").strip() == aphorism.strip():
                return c
    return None


def load_meta(stem):
    p = CACHE / "answers" / f"{stem}_meta.json"
    if not p.exists(): return {}
    return json.loads(p.read_text(encoding="utf-8"))


def decompose_all(gpt):
    """Use GPT-5.4 to decompose each wisdom into fragments; cache."""
    frags = {}
    if FRAGMENTS_PATH.exists():
        frags = json.loads(FRAGMENTS_PATH.read_text(encoding="utf-8"))

    for cand in CANDIDATES:
        cid = cand["cid"]
        if cid in frags and all(k in frags[cid] for k in ("TRIGGER", "MOVE", "ANTIPAT")):
            continue
        rec = load_candidate_record(cand["aphorism"])
        if not rec:
            print(f"  [{cid}] record missing"); continue
        prompt = DECOMP_PROMPT.format(
            aphorism=rec["aphorism"],
            unpacked=rec["unpacked_for_llm"],
            signal=rec.get("signal", ""),
        )
        try:
            r = gpt.generate(prompt, max_tokens=500, temperature=0.2)
            parsed = parse_json_from_llm(r["text"])
            frags[cid] = {
                "TRIGGER": parsed.get("TRIGGER", ""),
                "MOVE": parsed.get("MOVE", ""),
                "ANTIPAT": parsed.get("ANTIPAT", ""),
                "whole": rec["unpacked_for_llm"],
                "aphorism": rec["aphorism"],
                "wid": cand.get("committed_id"),
            }
            print(f"  [{cid}] decomposed")
        except Exception as e:
            print(f"  [{cid}] decomp err: {e}")
    FRAGMENTS_PATH.write_text(json.dumps(frags, ensure_ascii=False, indent=2))
    return frags


def alignment_stats(fragments, base_meta, ext_meta, model):
    """For each (base, ext) pid, compute cosine alignment of
    (ext_emb - base_emb) direction with each fragment + whole wisdom.
    Returns per-fragment and whole alignments as lists."""
    pids = sorted(set(base_meta.keys()) & set(ext_meta.keys()))
    base_texts, ext_texts = [], []
    for pid in pids:
        b = base_meta[pid].get("what_changed", "")
        e = ext_meta[pid].get("what_changed", "")
        if b and e:
            base_texts.append(b); ext_texts.append(e)
    if not base_texts: return None
    bem = model.encode(base_texts, normalize_embeddings=True, show_progress_bar=False)
    eem = model.encode(ext_texts, normalize_embeddings=True, show_progress_bar=False)
    diffs = eem - bem
    mags = np.linalg.norm(diffs, axis=1)
    keep = mags > 0.10
    if not keep.any(): return None
    diff_normed = diffs[keep] / mags[keep, None]  # (n_kept, d)

    # Encode all fragments + whole
    frag_texts = [fragments[k] for k in ("TRIGGER", "MOVE", "ANTIPAT", "whole")]
    frag_emb = model.encode(frag_texts, normalize_embeddings=True,
                             show_progress_bar=False)  # (4, d)

    # align[i,j] = cos(diff_i, frag_j)
    aligns = diff_normed @ frag_emb.T  # (n_kept, 4)
    return {
        "TRIGGER": (float(aligns[:, 0].mean()), float(aligns[:, 0].max())),
        "MOVE":    (float(aligns[:, 1].mean()), float(aligns[:, 1].max())),
        "ANTIPAT": (float(aligns[:, 2].mean()), float(aligns[:, 2].max())),
        "whole":   (float(aligns[:, 3].mean()), float(aligns[:, 3].max())),
        "max_any_frag_per_pid_mean": float(aligns[:, :3].max(axis=1).mean()),
        "n_kept": int(keep.sum()),
    }


def main():
    gpt = GPT5Client()
    print(f"Decomposer: {gpt.model}\n[1/2] Decomposing wisdoms to fragments...")
    frags = decompose_all(gpt)

    print(f"\n[2/2] Loading sentence-transformer...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    base_defaults = {"WCROSSL01": "_valp_v20_base"}
    default_base = "_valp_v20p1_base"

    print(f"\n{'cid':10s} {'wid':5s} {'whole':7s} {'trigger':9s} {'move':7s} "
          f"{'antipat':9s} {'max_frag_per_pid'}")
    print("-" * 80)
    results = []
    for cand in CANDIDATES:
        cid = cand["cid"]; wid = cand.get("committed_id") or "----"
        if cid not in frags: continue
        f = frags[cid]
        base_stem = base_defaults.get(cid, default_base)
        base_meta = load_meta(base_stem)
        ext_meta = load_meta(f"_valp_v20_ext_{cid}")
        if not base_meta or not ext_meta:
            print(f"  {cid:8s} meta missing"); continue
        st = alignment_stats(f, base_meta, ext_meta, model)
        if st is None:
            print(f"  {cid:8s} no valid diff"); continue
        print(f"  {cid:9s} {wid:4s} {st['whole'][0]:+.3f}  {st['TRIGGER'][0]:+.3f}    "
              f"{st['MOVE'][0]:+.3f}  {st['ANTIPAT'][0]:+.3f}    "
              f"{st['max_any_frag_per_pid_mean']:+.3f}")
        results.append({
            "cid": cid, "wid": cand.get("committed_id"),
            "aphorism": f["aphorism"],
            "fragments": {k: f[k] for k in ("TRIGGER", "MOVE", "ANTIPAT")},
            "alignment_whole_mean": st["whole"][0],
            "alignment_whole_max": st["whole"][1],
            "alignment_trigger_mean": st["TRIGGER"][0],
            "alignment_move_mean":   st["MOVE"][0],
            "alignment_antipat_mean": st["ANTIPAT"][0],
            "max_frag_per_pid_mean": st["max_any_frag_per_pid_mean"],
            "n_kept": st["n_kept"],
        })

    # Summary: does max-of-fragments beat whole?
    print(f"\n=== SUMMARY ===")
    ups = 0; total = len(results)
    for r in results:
        if r["max_frag_per_pid_mean"] > r["alignment_whole_mean"] + 0.05:
            ups += 1
    print(f"  Candidates where max-over-fragments beats whole by >0.05: {ups}/{total}")
    print(f"  Candidates where max-over-fragments >= 0.15 (meaningful): "
          f"{sum(1 for r in results if r['max_frag_per_pid_mean'] >= 0.15)}/{total}")
    print(f"  Candidates where max-over-fragments >= 0.25 (strong):    "
          f"{sum(1 for r in results if r['max_frag_per_pid_mean'] >= 0.25)}/{total}")

    OUT_LOG.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decomposer_model": gpt.model,
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
