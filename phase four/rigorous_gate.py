"""Rigorous gate — three orthogonal validation stages + documented
requirements for a properly-powered pair-A/B stage.

Orthogonal stages (this file implements & runs them):

  Stage B — Basis coverage.
    For each candidate W, compute the residual norm of embed(W.unpacked)
    projected orthogonal to the span of the current 75-wisdom library
    embeddings. Pass iff residual > τ_B: W adds a direction the library
    does not yet cover. Does not involve pair-wr at all.

  Stage C — Perturbation magnitude.
    For each pid, compute ||embed(ext.what_changed) − embed(base.what_changed)||.
    Average across 50 pids. Pass iff mean > τ_C: W actually changes the
    v20 solver's Turn-0 reasoning, not just its output. Uses meta files
    that were never judged; completely independent of pair-wr.

  Stage A — Faithfulness alignment.
    Same per-pid diff as Stage C, but normalised. Cosine-align with
    embed(W.unpacked). Average over pids whose diff is non-trivial.
    Pass iff alignment > τ_A: W's perturbation direction matches W's
    own self-description of where it should fire.

Documented but NOT implemented here (requires fresh compute):

  Stage 4 — Properly-powered pair-A/B.
    - n ≥ 200 problems (SE(0.6, 200) ≈ 0.035; +10pp threshold is ≈2.9 SE;
      single-test p ≈ 0.002).
    - Bonferroni for 12 candidates: per-candidate threshold p < 0.001,
      equivalent to requiring |wr − 0.50| ≥ 0.10 at n = 250 instead of 50.
    - Three judge families, each pre-registered.
    - Each pair judged with BOTH side orderings; mean pair-verdict used.
    - Pass iff ≥ 2/3 families have corrected-p < 0.01 AND
      ≥ 2/3 families have mean wr ≥ 0.55.

All three orthogonal stages are pass/fail independently from pair-wr,
so failures and passes can be compared with the original gate's
outcomes to diagnose *what kind* of signal it was picking up.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase four"))

from sentence_transformers import SentenceTransformer

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "rigorous_gate_log.json"

TAU_B = 0.55      # basis-residual threshold; >0 means *some* orthogonality
TAU_C = 0.30      # mean what_changed diff threshold (unit-sphere distance)
TAU_A = 0.25      # mean cosine alignment threshold

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
    {"cid": "WCROSSL01", "aphorism": "是骡子是马，拉出来遛遛", "committed_id": "W078"},
]


def load_candidate_record(aphorism):
    """Find the full candidate record (with unpacked_for_llm) by aphorism match."""
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
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ---------- Stage B: basis coverage ----------

def stage_B(cand_emb, library_embs):
    """Residual norm of cand_emb orthogonal to span(library_embs).

    library_embs: (75, d) normalized rows.
    cand_emb:     (d,) normalized.

    Uses SVD of library_embs to get orthonormal basis U of its row span.
    Residual = cand_emb − U U^T cand_emb. Returns ||residual||.
    A value of 0 = cand is fully in library span; value close to 1 = fully
    orthogonal.
    """
    U, _, _ = np.linalg.svd(library_embs.T, full_matrices=False)
    # U has shape (d, rank). Columns are orthonormal basis for library row span.
    proj = U @ (U.T @ cand_emb)
    residual = cand_emb - proj
    return float(np.linalg.norm(residual))


# ---------- Stage C: perturbation magnitude ----------

def stage_C(base_meta, ext_meta, model):
    """Mean ||embed(ext.wc) - embed(base.wc)|| across shared pids."""
    pids = sorted(set(base_meta.keys()) & set(ext_meta.keys()))
    if not pids: return 0.0, 0
    base_texts, ext_texts = [], []
    for pid in pids:
        b = base_meta[pid].get("what_changed", "")
        e = ext_meta[pid].get("what_changed", "")
        if not b or not e:
            continue
        base_texts.append(b)
        ext_texts.append(e)
    if not base_texts: return 0.0, 0
    bem = model.encode(base_texts, normalize_embeddings=True, show_progress_bar=False)
    eem = model.encode(ext_texts, normalize_embeddings=True, show_progress_bar=False)
    diffs = np.linalg.norm(eem - bem, axis=1)
    return float(np.mean(diffs)), len(diffs)


# ---------- Stage A: faithfulness alignment ----------

def stage_A(cand_emb, base_meta, ext_meta, model, nontrivial_threshold=0.10):
    """Mean cosine alignment of (ext.wc - base.wc) with cand direction, over
    pids whose perturbation is non-trivial (|diff| > nontrivial_threshold)."""
    pids = sorted(set(base_meta.keys()) & set(ext_meta.keys()))
    if not pids: return 0.0, 0
    base_texts, ext_texts = [], []
    for pid in pids:
        b = base_meta[pid].get("what_changed", "")
        e = ext_meta[pid].get("what_changed", "")
        if not b or not e: continue
        base_texts.append(b); ext_texts.append(e)
    if not base_texts: return 0.0, 0
    bem = model.encode(base_texts, normalize_embeddings=True, show_progress_bar=False)
    eem = model.encode(ext_texts, normalize_embeddings=True, show_progress_bar=False)
    diffs = eem - bem
    mags = np.linalg.norm(diffs, axis=1)
    # Only use pids where diff is meaningful
    keep_idx = mags > nontrivial_threshold
    if not keep_idx.any(): return 0.0, 0
    diffs_kept = diffs[keep_idx]
    mags_kept = mags[keep_idx]
    # normalize
    diffs_norm = diffs_kept / mags_kept[:, None]
    aligns = diffs_norm @ cand_emb  # (n_kept,)
    return float(np.mean(aligns)), int(keep_idx.sum())


def main():
    print("Loading sentence-transformer...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Library embeddings (75 original)
    lib = json.loads((CACHE / "wisdom_library.json").read_text(encoding="utf-8"))
    lib_texts = [w["unpacked_for_llm"] for w in lib]
    print(f"Embedding library: {len(lib_texts)} wisdoms...")
    lib_embs = model.encode(lib_texts, normalize_embeddings=True,
                             show_progress_bar=False)

    # Determine base meta file per candidate
    base_meta_defaults = {
        "WCROSSL01": "_valp_v20_base",  # W078 used post-prune base
    }
    default_base = "_valp_v20p1_base"

    results = []
    print(f"\nValidating {len(CANDIDATES)} candidates...\n")
    print(f"{'cid':10s} {'wid':5s} {'B_resid':>8s} {'C_pert':>8s} {'A_align':>8s}  "
          f"passes   composite")
    print("-" * 78)

    for cand in CANDIDATES:
        cid = cand["cid"]; wid = cand.get("committed_id") or "----"
        rec = load_candidate_record(cand["aphorism"])
        if not rec:
            print(f"  {cid:9s} {wid:4s} record-missing"); continue
        cand_text = rec.get("unpacked_for_llm", "")
        cand_emb = model.encode([cand_text], normalize_embeddings=True,
                                 show_progress_bar=False)[0]

        # Stage B
        b_resid = stage_B(cand_emb, lib_embs)
        b_pass = b_resid > TAU_B

        # Stage C, A: need base_meta + ext_meta
        base_stem = base_meta_defaults.get(cid, default_base)
        base_meta = load_meta(base_stem)
        ext_meta = load_meta(f"_valp_v20_ext_{cid}")
        if base_meta is None or ext_meta is None:
            print(f"  {cid:9s} {wid:4s} meta missing ({base_stem} or ext)")
            continue

        c_pert, c_n = stage_C(base_meta, ext_meta, model)
        c_pass = c_pert > TAU_C
        a_align, a_n = stage_A(cand_emb, base_meta, ext_meta, model)
        a_pass = a_align > TAU_A

        n_pass = int(b_pass) + int(c_pass) + int(a_pass)
        composite = "PASS_ALL" if n_pass == 3 else \
                    "PASS_MAJ" if n_pass == 2 else \
                    "FAIL"
        marks = ("P" if b_pass else "F") + ("P" if c_pass else "F") + ("P" if a_pass else "F")
        print(f"  {cid:9s} {wid:4s} {b_resid:8.3f} {c_pert:8.3f} {a_align:8.3f}  "
              f"B{marks[0]} C{marks[1]} A{marks[2]}   {composite}")

        results.append({
            "cid": cid, "wid": cand.get("committed_id"),
            "aphorism": cand["aphorism"],
            "stage_B_residual": b_resid, "stage_B_pass": bool(b_pass),
            "stage_C_perturbation": c_pert, "stage_C_n": c_n,
            "stage_C_pass": bool(c_pass),
            "stage_A_alignment": a_align, "stage_A_n": a_n,
            "stage_A_pass": bool(a_pass),
            "n_orthogonal_passes": n_pass,
            "composite": composite,
        })

    # Summary vs pair-wr outcomes
    print(f"\n{'='*78}\nSummary — compare orthogonal vs pair-wr outcomes\n{'='*78}")
    pair_wr_n100 = {"W076": 0.57, "W077": 0.52, "W078": 0.52}  # from Exp 10
    print(f"  {'cid':10s} {'wid':5s} {'orthog':9s} {'pair_wr_n100':12s} {'final_call'}")
    for r in results:
        wid = r["wid"] or "----"
        pwr = pair_wr_n100.get(r["wid"])
        pwr_s = f"{pwr:.2f}" if pwr is not None else "N/A"
        ortho = r["composite"]
        if ortho == "PASS_ALL" and pwr and pwr >= 0.55:
            final = "STRONG_VALIDATION"
        elif ortho == "PASS_ALL" and (pwr is None or pwr < 0.55):
            final = "ORTHO_YES_PAIR_NO (novel direction but no answer-quality uplift)"
        elif ortho in ("FAIL", "PASS_MAJ") and pwr and pwr >= 0.55:
            final = "PAIR_YES_ORTHO_NO (answer uplift but possibly noise)"
        else:
            final = "BOTH_NEGATIVE"
        print(f"  {r['cid']:9s} {wid:4s} {ortho:9s} {pwr_s:12s} {final}")

    n_total = len(results)
    n_pass_all = sum(1 for r in results if r["composite"] == "PASS_ALL")
    n_pass_maj = sum(1 for r in results if r["composite"] == "PASS_MAJ")
    print(f"\n  Orthogonal outcomes: PASS_ALL={n_pass_all}/{n_total}  "
          f"PASS_MAJ={n_pass_maj}/{n_total}")

    OUT_LOG.write_text(json.dumps({
        "thresholds": {"tau_B": TAU_B, "tau_C": TAU_C, "tau_A": TAU_A},
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
