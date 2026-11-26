import argparse, json, os, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
LOG = Path(os.environ.get('HEGEL_SKILL_INSPECTION_LOG', '/tmp/.hegel_skill_inspections.jsonl'))
cat = json.loads((HERE / '_skill_contracts.json').read_text(encoding='utf-8'))

def _shared_search_catalog(catalog: dict, query: str = "", *, top_k: int = 6) -> tuple[list[str], list[str], list[str]]:
    """Single canonical `M9.2.1` `A3` pure-data search engine shared identically across `K` and `E` (`0` skill import)."""
    import re as _re

    q_low = str(query or "").lower()
    raw_words = set(_re.findall(r"[a-zA-Z0-9]{2,}", _re.sub(r"[_./-]+", " ", q_low)))
    syn_groups = (
        {"robust", "regression", "outlier", "outliers", "irls", "cauchy", "huber", "lorentzian", "glitch", "spikes", "residual", "weights"},
        {"certified", "certificate", "lower", "bound", "bounds", "l0", "sparse", "box", "subcube", "cube", "minorant", "branch", "blocks"},
        {"preconditioner", "multigrid", "amg", "galerkin", "cg", "conjugate", "spd", "elliptic", "poisson", "diffusion", "robin"},
        {"panel", "reference", "control", "controls", "cohort", "batch", "center", "stratum", "strata", "imputation", "missing", "align", "balanced", "fill"},
        {"nuisance", "profile", "profiled", "profiling", "whitener", "schur", "tomography", "amplitudes", "matching"},
        {"relaxation", "creep", "viscoelastic", "memory", "hereditary", "prony", "step", "breakpoint", "tau", "trend", "volterra"},
        {"kronecker", "sylvester", "separable", "spectral", "kkt", "simplex", "active", "transport"},
    )
    expanded = set(raw_words)
    for grp in syn_groups:
        if raw_words & grp:
            expanded |= grp
    neg_rx = r"(?:avoid|no|not|never|without|non|exclude|excluding)\s+[a-z0-9_\-\s]{0,24}"
    query_affirms_complex_or_indef = bool(
        (_re.search(r"\b(?:complex128|complex|helmholtz|indefinite)\b", q_low))
        and not _re.search(neg_rx + r"\b(?:complex128|complex|helmholtz|indefinite)\b", q_low)
        and not (raw_words & {"real", "float64", "spd"})
    )
    scored = []
    rejected_notes = []
    for entry in catalog.get("skills", []):
        m_stem = str(entry.get("module_stem") or "")
        fn_app_map = entry.get("function_level_applicability") or {}
        is_l1 = 1.0 if entry.get("granularity_tier") == "L1_compositional_primitive" else 0.0
        fns = entry.get("functions", [])
        cards = entry.get("index_cards", [])
        for idx_f, c in enumerate(cards):
            fn_info = fns[idx_f] if idx_f < len(fns) else {}
            fn_nm = str(fn_info.get("function") or "")
            app_status = str(fn_app_map.get(fn_nm) or "")
            if query_affirms_complex_or_indef and m_stem in {
                "autogenous_prim_conservative_spd_solve",
                "autogenous_prim_robust_trend_decomposition",
                "autogenous_prim_robust_affine_projection",
                "autogenous_prim_box_ridge_bounds",
                "autogenous_prim_reference_panel_balancing",
            }:
                rejected_notes.append(f"{m_stem}.{fn_nm}: REJECTED_PRECONDITION(real_or_spd_required_for_query)")
                continue
            if app_status.startswith("REJECTED_"):
                rejected_notes.append(f"{m_stem}.{fn_nm}: {app_status}")
                continue
            fn_blob = " ".join([m_stem, fn_nm, str(fn_info.get("signature") or ""), str(fn_info.get("doc_summary") or "")]).lower()
            fn_words = set(_re.findall(r"[a-zA-Z0-9]{2,}", _re.sub(r"[_./-]+", " ", fn_blob)))
            direct_hit = len(raw_words & fn_words)
            syn_hit = len(expanded & fn_words)
            exact_id_bonus = 6.0 if (fn_nm and fn_nm.lower() in q_low) else (3.0 if m_stem.lower() in q_low else 0.0)
            sc = 3.0 * direct_hit + 1.2 * syn_hit + exact_id_bonus + (0.5 * is_l1 if (direct_hit + syn_hit) > 0 else 0.0)
            if not raw_words:
                sc = 1.0 + is_l1
            if sc > 0.0:
                card_out = f"{c} | STATUS: {app_status}" if app_status and "STATUS:" not in c else c
                # Sort by canonical `(m_stem, fn_nm)` so `K` vs `E` command prefixes NEVER change tie-break order!
                scored.append((round(float(sc), 9), f"{m_stem}.{fn_nm}", card_out))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top_slice = scored[: max(1, int(top_k))]
    return [c for _, _, c in top_slice], [fid for _, fid, _ in top_slice], rejected_notes


def _shared_describe_catalog(catalog: dict, target: str = "", *, mode: str = "executable", ref_dir: str = "") -> dict:
    """Single canonical `M9.2.1` `A3` name resolver & contract renderer shared identically across `K` and `E`."""
    q = str(target or "").strip()
    if q.startswith("hegel_wormhole_skills."):
        q = q[len("hegel_wormhole_skills.") :]
    is_exec = mode == "executable"
    matches = []
    for entry in catalog.get("skills", []):
        m_stem = str(entry.get("module_stem") or "")
        matched_fns = None
        if not q or q == "*" or q == m_stem:
            matched_fns = list(entry.get("functions", []))
        elif "." in q and q.split(".", 1)[0] == m_stem:
            fn_q = q.split(".", 1)[1]
            sub_fns = [f for f in entry.get("functions", []) if f.get("function") == fn_q]
            if sub_fns:
                matched_fns = sub_fns
        else:
            sub_fns = [f for f in entry.get("functions", []) if f.get("function") == q]
            if sub_fns:
                matched_fns = sub_fns
        if matched_fns is not None:
            can_item = {
                "module_stem": m_stem,
                "granularity_tier": entry.get("granularity_tier"),
                "compiled_code_sha256": entry.get("compiled_code_sha256"),
                "qualification_status": entry.get("qualification_status"),
                "unit_test_verified": entry.get("unit_test_verified"),
                "mathematical_preconditions": list(entry.get("mathematical_preconditions") or []),
                "counter_regime_guards": list(entry.get("counter_regime_guards") or []),
                "function_level_applicability": dict(entry.get("function_level_applicability") or {}),
                "functions": matched_fns,
                "index_cards": list(entry.get("index_cards") or []),
                "execution_available": is_exec,
                "module_import": (f"hegel_wormhole_skills.{m_stem}" if is_exec else None),
                "reference_source_txt": (str(Path(ref_dir) / f"{m_stem}.reference.txt") if (not is_exec and ref_dir) else None),
            }
            matches.append(can_item)
    out = {
        "ok": len(matches) > 0,
        "target": q or "*",
        "access_mode": mode,
        "execution_available": is_exec,
        "matches": matches,
    }
    if not matches:
        out["error"] = f"unknown_reference:{target}"
    return out

ap = argparse.ArgumentParser()
ap.add_argument('--list', action='store_true')
ap.add_argument('--search', type=str, default='')
ap.add_argument('--describe', type=str, default='')
args = ap.parse_args()
def _log(ev, tgt, **extra):
    try:
        with LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'event': ev, 'target': tgt, 'mode': 'knowledge_only_ref', 'ts': round(time.time(), 3), **extra}) + '\n')
    except OSError:
        pass
if args.search:
    top_cards, ranked_ids, rejected_notes = _shared_search_catalog(cat, args.search, top_k=6)
    _log('skill_search_executed', args.search, query=args.search, returned_card_count=len(top_cards), ranked_function_ids=ranked_ids, rejected_precondition_count=len(rejected_notes))
    if top_cards:
        print('\n'.join(top_cards))
    elif rejected_notes:
        print('No applicable skill cards (preconditions rejected for query): ' + '; '.join(rejected_notes[:3]))
    else:
        print('No matching skill cards found.')
elif args.list and not args.describe:
    _log('skill_index_listed', '__all__')
    payload = _shared_describe_catalog(cat, '*', mode='knowledge_only', ref_dir=str(HERE))
    print('\n'.join(c for m in payload['matches'] for c in m.get('index_cards', [])))
else:
    q = args.describe.strip()
    _log('skill_contract_requested', q or '*')
    payload = _shared_describe_catalog(cat, q, mode='knowledge_only', ref_dir=str(HERE))
    if payload['ok']:
        _log('skill_contract_delivered', q or '*', matched_module_count=len(payload['matches']), matched_modules=[m.get('module_stem') for m in payload['matches']])
    print(json.dumps(payload, indent=2, ensure_ascii=False))
