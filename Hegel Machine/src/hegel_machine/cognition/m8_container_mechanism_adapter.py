"""Hegel M8.4 Process-Isolated Public Calibration & Symmetry Probes, Transactional Repair & Graph Writeback.

Upgrades M8.3 (`b3fdc3134`) to close all findings in `Hegel_M8_2_Audit` and `Hegel_M8_3_Audit`:
  1. **Process-Isolated Candidate Execution & Fail-Closed Finite/Equivariance Probes**:
     - Candidate Python modules execute inside a separate untrusted `python3 -I` child RPC worker (`stdout`
       redirected to `stderr`, pure JSON+ndarray wire codec, zero `pickle`), while the trusted supervisor
       evaluates schemas, `isfinite`, physical bounds, public calibration (using ONLY public bundle targets,
       ZERO private verifier seeds), and general non-automorphism equivariances!
     - `NaN` / `Inf` outputs in `ho06`, `ho10`, `ho04`, `ho05`, `ho08`, `ho12` fail closed immediately (`REFUTED`).
     - `ho05` tests 2 non-cyclic general vertex permutations on BOTH `adjacency` and `sparse_precision`
       (rejecting data-blind constant ring templates).
     - `ho12` verifies cohort circular-tag membership (`>= 80%` prevalence), empirical `[3.2, 7.5]` depth ratio,
       `[A-Z]\\d+[A-Z]` wildtype codon match against `reference_wildtype_cds`, AND positive plasmid-tag renaming
       equivariance (rejecting constant outputs).
     - `ho08` verifies affine thermal-drift invariance and parallel-spring (`+3.5 * strain`) equivariance on BOTH
       `G_inf` and `test_stress`, while recording `65%`-prefix `G_inf` shift as soft `identifiability_warning`
       (`SUPPORTED_IN_PARTIAL_SCOPE`, NEVER falsely refuting valid full-horizon solvers!).
  2. **Exception-Safe Transactional Repair & Deferred Content-Addressed Guard Commit**:
     - `execute_in_container_local_repair` wraps staging + probing in `try/except/finally`: any probe exception
       restores `orig_code` in `probe_cid` and returns a typed failure receipt preserving 100% of spent LLM tokens!
     - `defer_guard_commit=True` holds the distilled guard as a `pending_guard` (`CANDIDATE_REPAIR_HYPOTHESIS`)
       until the caller confirms `verify_candidate_artifact_contract_in_container` ALSO accepts!
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import tempfile
from time import monotonic
from types import SimpleNamespace
from typing import Any

from . import m6_frontier_real_benchmarks as m6f
from .m7_holdout_ood_blind_benchmark import CLEAN_SCIPY_SANDBOX_IMAGE, HoldoutOODTask, ensure_clean_scipy_sandbox_image
from .m8_persistent_variational_planner import (
    PersistentMechanismGraph,
    ResearchDecisionRecord,
    TypedMechanism,
)
from .wormhole_forest import evaluate_predicate_rule

M8_PUBLIC_PROBE_PROTOCOL_VERSION = "m8_4_isolated_rpc_v1"
NOMINAL_LOCAL_REPAIR_COST = 0.35

M8_CONTAINER_LOCAL_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "repaired_python_code",
        "invariance_guard_name",
        "invariance_principle",
        "defect_diagnosis",
    ],
    "properties": {
        "repaired_python_code": {"type": "string", "minLength": 40, "maxLength": 24000},
        "invariance_guard_name": {"type": "string", "minLength": 3, "maxLength": 80},
        "invariance_principle": {"type": "string", "minLength": 10, "maxLength": 600},
        "defect_diagnosis": {"type": "string", "minLength": 10, "maxLength": 600},
    },
}


# Self-contained supervisor script executing ONLY against public workspace files in `/holdout_task`
# while isolating untrusted candidate module imports/calls inside a separate `python3 -I` child RPC worker!
_GENERAL_PUBLIC_INVARIANCE_PROBE_PY = r'''
import copy, glob, json, math, os, pickle, re, shutil, subprocess, sys, tempfile
import numpy as np

TASK_DIR = "/holdout_task"
if TASK_DIR not in sys.path:
    sys.path.append(TASK_DIR)
SENTINEL = os.environ.pop("HEGEL_PUB_PROBE_SENTINEL", "")

out = {
    "probe_applicable": False,
    "probe_protocol_version": "m8_5_isolated_rpc_v2",
    "symmetry_probe_family": None,
    "public_output_contract_ok": None,
    "public_calibration_ok": None,
    "id_renaming_invariant_ok": None,
    "identifiability_warning": False,
    "identifiability_metrics": {},
    "missing_or_null_keys": [],
    "renaming_discrepancies": [],
    "public_consistency_metrics": {},
}

_CODEC_SRC = """
def _enc_val(v):
    if isinstance(v, np.ndarray):
        return {'__hegel_ndarray__': _enc_val(v.tolist()), 'dtype': str(v.dtype), 'shape': list(v.shape)}
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        if math.isnan(fv): return {'__hegel_nonfinite__': 'nan'}
        if math.isinf(fv): return {'__hegel_nonfinite__': 'inf' if fv > 0 else '-inf'}
        return fv
    if isinstance(v, tuple): return {'__hegel_tuple__': [_enc_val(x) for x in v]}
    if isinstance(v, list): return [_enc_val(x) for x in v]
    if isinstance(v, dict): return {str(k): _enc_val(val) for k, val in v.items()}
    return v

def _dec_val(v):
    if isinstance(v, dict):
        if '__hegel_nonfinite__' in v: return float(v['__hegel_nonfinite__'])
        if '__hegel_tuple__' in v: return tuple(_dec_val(x) for x in v['__hegel_tuple__'])
        if '__hegel_ndarray__' in v:
            return np.asarray(_dec_val(v['__hegel_ndarray__']), dtype=v['dtype']).reshape(tuple(v['shape']))
        return {k: _dec_val(val) for k, val in v.items()}
    if isinstance(v, list): return [_dec_val(x) for x in v]
    return v
"""
exec(_CODEC_SRC, globals())

def _start_candidate_worker(mod_name):
    worker_src = (
        "import importlib, json, math, os, sys\n"
        "import numpy as np\n"
        "rpc_fd = os.dup(1)\n"
        "os.dup2(2, 1)\n"
        "sys.stdout = sys.stderr\n"
        f"if {TASK_DIR!r} not in sys.path: sys.path.append({TASK_DIR!r})\n"
        + _CODEC_SRC +
        f"\nmod = importlib.import_module({mod_name!r})\n"
        "for raw_ln in sys.stdin:\n"
        "    if not raw_ln.strip(): continue\n"
        "    req = json.loads(raw_ln)\n"
        "    rid = req['id']\n"
        "    try:\n"
        "        fn = getattr(mod, req['fn'])\n"
        "        res = fn(*_dec_val(req['args']), **_dec_val(req['kwargs']))\n"
        "        resp = {'id': rid, 'ok': True, 'result': _enc_val(res)}\n"
        "    except Exception as exc:\n"
        "        resp = {'id': rid, 'ok': False, 'error': f'{type(exc).__name__}: {exc}'}\n"
        "    os.write(rpc_fd, (json.dumps(resp, allow_nan=False) + '\\n').encode('utf-8'))\n"
        "os.close(rpc_fd)\n"
    )
    err_fp = tempfile.TemporaryFile(mode="w+b")
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("HEGEL_")}
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", worker_src],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err_fp, env=clean_env, text=True, bufsize=1,
    )
    proc._hegel_err_fp = err_fp
    seq = {"i": 0}
    def _call(fn_name, *args, **kwargs):
        seq["i"] += 1
        rid = seq["i"]
        proc.stdin.write(json.dumps({"id": rid, "fn": fn_name, "args": _enc_val(list(args)), "kwargs": _enc_val(kwargs)}, allow_nan=False) + "\n")
        proc.stdin.flush()
        ln = proc.stdout.readline()
        if not ln:
            err_tail = ""
            try:
                err_fp.flush()
                sz = err_fp.tell()
                err_fp.seek(max(0, sz - 400))
                err_tail = err_fp.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"candidate_worker_eof:{err_tail}")
        resp = json.loads(ln)
        if resp.get("id") != rid or not resp.get("ok"):
            raise RuntimeError(str(resp.get("error") or "candidate_rpc_error"))
        return _dec_val(resp.get("result"))
    return proc, _call

def _stop_worker(proc):
    if proc is None: return
    try:
        if proc.stdin: proc.stdin.close()
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
    try:
        efp = getattr(proc, "_hegel_err_fp", None)
        if efp is not None:
            efp.close()
    except Exception:
        pass

def _is_finite_scalar(v):
    try:
        if isinstance(v, bool):
            return False
        return math.isfinite(float(v))
    except Exception:
        return False

def _is_finite_array(x, expected_shape=None):
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return False
    if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        return False
    if expected_shape is not None and arr.shape != tuple(expected_shape):
        return False
    return True

def _is_binary_matrix(x, expected_shape):
    try:
        raw = np.asarray(x)
        if raw.dtype == object:
            return False
        af = np.asarray(raw, dtype=float)
    except Exception:
        return False
    if af.shape != tuple(expected_shape) or not bool(np.all(np.isfinite(af))):
        return False
    return bool(np.all((af == 0.0) | (af == 1.0)))

# 1. `ho12` Cohort Consensus: Cohort prevalence + depth witness + WT codon witness + negative & positive tag renaming!
if os.path.isdir("/holdout_task/public_cohort") and os.path.isfile("/holdout_task/plasmid_audit.py"):
    out["probe_applicable"] = True
    out["symmetry_probe_family"] = "cohort_identifier_renaming_symmetry"
    proc = None
    try:
        proc, call_fn = _start_candidate_worker("plasmid_audit")
        pub_dir = "/holdout_task/public_cohort"
        ref_panel = json.load(open(os.path.join(pub_dir, "reference_panel.json"), "r", encoding="utf-8"))
        wt_cds = str(ref_panel.get("reference_wildtype_cds") or "")
        _CODON = {
            "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
            "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
            "TCT":"S","TCC":"S","TCA":"S","TCG":"S","AGT":"S","AGC":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
            "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
            "TAT":"Y","TAC":"Y","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","AAT":"N","AAC":"N","AAA":"K","AAG":"K",
            "GAT":"D","GAC":"D","GAA":"E","GAG":"E","TGT":"C","TGC":"C","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
            "GGT":"G","GGC":"G","GGA":"G","GGG":"G","TAA":"*","TAG":"*","TGA":"*",
        }
        isos = sorted(glob.glob(os.path.join(pub_dir, "isolate_*.json")))
        tag_iso_counts = {}
        tag_cns = {}
        tag_seqs = {}
        _RC_TRANS = str.maketrans("ACGTacgt", "TGCAtgca")
        def _revcomp(s):
            return s.translate(_RC_TRANS)[::-1]
        def _find_aligned_plasmid_cds(plasmid_seq, wt_seq):
            L = len(wt_seq)
            if L < 48 or len(plasmid_seq) < L:
                return None
            anchors = [(0, wt_seq[:24]), (24, wt_seq[24:48]), (L - 24, wt_seq[-24:]), (L - 48, wt_seq[-48:-24])]
            for strand_seq in (plasmid_seq + plasmid_seq[:L], _revcomp(plasmid_seq) + _revcomp(plasmid_seq)[:L]):
                tried = set()
                for off_a, seed_a in anchors:
                    idx = strand_seq.find(seed_a)
                    while idx != -1:
                        st = idx - off_a
                        if 0 <= st <= len(strand_seq) - L and st not in tried:
                            tried.add(st)
                            cand_cds = strand_seq[st : st + L]
                            diffs_nt = sum(1 for a, b in zip(cand_cds, wt_seq) if a != b)
                            if diffs_nt <= 9:
                                return cand_cds
                        idx = strand_seq.find(seed_a, idx + 1)
            return None

        for ifp in isos:
            b_iso = json.load(open(ifp, "r", encoding="utf-8"))
            chr_ds = [float(c["mean_depth"]) for c in b_iso.get("contigs", []) if not bool(c.get("circular", False))]
            med_chr = float(np.median(chr_ds)) if chr_ds else 1.0
            seen_t = set()
            for c in b_iso.get("contigs", []):
                c_seq = str(c.get("sequence") or "").upper()
                if bool(c.get("circular", False)) and len(c_seq) >= 2400:
                    ft = str(c.get("family_tag") or "")
                    seen_t.add(ft)
                    tag_cns.setdefault(ft, []).append(float(c["mean_depth"]) / max(med_chr, 1e-9))
                    tag_seqs.setdefault(ft, []).append(c_seq)
            for ft in seen_t:
                tag_iso_counts[ft] = tag_iso_counts.get(ft, 0) + 1

        r0 = call_fn("audit_outbreak_cohort", pub_dir)
        miss = []
        if not isinstance(r0, dict):
            miss.append("return_not_dict")
        else:
            ft0 = r0.get("shared_plasmid_family_tag")
            sub0 = r0.get("amino_acid_substitution")
            cn0_raw = r0.get("median_plasmid_copy_number")
            if not isinstance(ft0, str) or not ft0.strip() or ft0.strip().lower() == "none":
                miss.append("shared_plasmid_family_tag")
            elif tag_iso_counts.get(ft0, 0) < max(1, int(math.ceil(0.80 * len(isos)))):
                miss.append(f"tag_not_cohort_conserved({ft0}:count={tag_iso_counts.get(ft0, 0)}/{len(isos)})")
            if not _is_finite_scalar(cn0_raw):
                miss.append("median_plasmid_copy_number")
            else:
                cn0 = float(cn0_raw)
                if not (3.2 <= cn0 <= 7.5):
                    miss.append(f"copy_number_out_of_plasmid_bounds({cn0:.4f})")
                elif isinstance(ft0, str) and ft0 in tag_cns:
                    emp_cn = float(np.median(tag_cns[ft0]))
                    if not (abs(cn0 - emp_cn) <= 0.35):
                        miss.append(f"copy_number_mismatch_vs_cohort_depths({cn0:.3f}!={emp_cn:.3f})")
            if not isinstance(sub0, str) or not re.match(r"^[A-Z]\d+[A-Z]$", sub0.strip()):
                miss.append("amino_acid_substitution_invalid_format")
            else:
                s_clean = sub0.strip()
                wt_aa, pos_1, mut_aa = s_clean[0], int(s_clean[1:-1]), s_clean[-1]
                wt_up = wt_cds.upper()
                n_cod = len(wt_up) // 3
                if not (1 <= pos_1 <= n_cod) or wt_aa == mut_aa:
                    miss.append("substitution_position_or_synonymous_invalid")
                else:
                    ref_cod = wt_up[3 * (pos_1 - 1) : 3 * pos_1]
                    if _CODON.get(ref_cod) != wt_aa:
                        miss.append(f"substitution_wt_mismatch({wt_aa}!=ref_{_CODON.get(ref_cod)})")
                    elif isinstance(ft0, str) and ft0 in tag_seqs:
                        seq_list = tag_seqs[ft0]
                        mut_matches = 0
                        aligned_cnt = 0
                        for p_seq in seq_list:
                            al_cds = _find_aligned_plasmid_cds(p_seq, wt_up)
                            if al_cds is not None:
                                aligned_cnt += 1
                                pl_cod = al_cds[3 * (pos_1 - 1) : 3 * pos_1]
                                if _CODON.get(pl_cod) == mut_aa:
                                    mut_matches += 1
                        if aligned_cnt == 0 or mut_matches < max(1, int(math.ceil(0.80 * aligned_cnt))):
                            miss.append(f"substitution_mut_not_supported_by_plasmid_sequence({s_clean}:matches={mut_matches}/{aligned_cnt})")
        out["missing_or_null_keys"] = miss
        out["public_output_contract_ok"] = (len(miss) == 0)
        out["public_calibration_ok"] = (len(miss) == 0)
        out["public_consistency_metrics"]["target_coverage"] = {
            "shared_plasmid_family_tag": "verified_cohort_prevalence_and_equivariance",
            "median_plasmid_copy_number": "verified_cohort_depth_ratio",
            "amino_acid_substitution": "verified_wildtype_and_plasmid_cds_codon",
        }
        if isinstance(r0, dict) and all(_is_finite_scalar(v) if isinstance(v, (int, float)) else True for v in r0.values()):
            out["public_consistency_metrics"]["public_output_preview"] = {
                k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))
                for k, v in r0.items()
            }

        # (a) Background contig-tag renaming + isolate order reversal invariance
        ren_dir = "/tmp/_m8_pub_cohort_renamed"
        shutil.rmtree(ren_dir, ignore_errors=True)
        os.makedirs(ren_dir, exist_ok=True)
        shutil.copy2(os.path.join(pub_dir, "reference_panel.json"), os.path.join(ren_dir, "reference_panel.json"))
        for idx, ifp in enumerate(reversed(isos), start=1):
            obj = json.load(open(ifp, "r", encoding="utf-8"))
            ctgs = []
            for ci, c in enumerate(obj.get("contigs", [])):
                c2 = dict(c)
                c2["contig_id"] = f"sym_ctg_{idx}_{ci}"
                if not bool(c2.get("circular", False)):
                    c2["family_tag"] = f"BG_CHROM_UNIT_{ci}"
                ctgs.append(c2)
            obj["isolate_id"] = f"SYM_ISO_{idx:02d}"
            obj["contigs"] = ctgs
            json.dump(obj, open(os.path.join(ren_dir, f"isolate_{idx:02d}.json"), "w"))
        r1 = call_fn("audit_outbreak_cohort", ren_dir)
        diffs = []
        if not isinstance(r1, dict) or not _is_finite_scalar(r1.get("median_plasmid_copy_number")):
            diffs.append("renamed_bg_invalid_or_non_finite")
        elif isinstance(r0, dict) and _is_finite_scalar(r0.get("median_plasmid_copy_number")):
            for sk in ("shared_plasmid_family_tag", "amino_acid_substitution"):
                if r0.get(sk) != r1.get(sk):
                    diffs.append(f"{sk}:{r0.get(sk)}!={r1.get(sk)}")
            if not (abs(float(r0["median_plasmid_copy_number"]) - float(r1["median_plasmid_copy_number"])) <= 1e-5):
                diffs.append("copy_number_drift_under_bg_renaming")

        # (b) Positive outbreak plasmid-tag equivariance (`true_tag -> OUTBREAK_SYM_PLASMID_Z7` MUST rename output tag!)
        pos_dir = ren_dir + "_pos"
        shutil.rmtree(pos_dir, ignore_errors=True)
        os.makedirs(pos_dir, exist_ok=True)
        shutil.copy2(os.path.join(pub_dir, "reference_panel.json"), os.path.join(pos_dir, "reference_panel.json"))
        conserved_tags = {t for t, cnt in tag_iso_counts.items() if cnt >= max(1, int(math.ceil(0.80 * len(isos))))}
        for idx, ifp in enumerate(isos, start=1):
            obj = json.load(open(ifp, "r", encoding="utf-8"))
            for c in obj.get("contigs", []):
                if bool(c.get("circular", False)) and c.get("family_tag") in conserved_tags:
                    c["family_tag"] = "OUTBREAK_SYM_PLASMID_Z7"
            json.dump(obj, open(os.path.join(pos_dir, f"isolate_{idx:02d}.json"), "w"))
        r2 = call_fn("audit_outbreak_cohort", pos_dir)
        shutil.rmtree(pos_dir, ignore_errors=True)
        shutil.rmtree(ren_dir, ignore_errors=True)
        if not isinstance(r2, dict) or r2.get("shared_plasmid_family_tag") != "OUTBREAK_SYM_PLASMID_Z7":
            diffs.append(f"plasmid_tag_not_equivariant(got={r2.get('shared_plasmid_family_tag') if isinstance(r2, dict) else type(r2).__name__})")
        elif isinstance(r0, dict) and (
            r2.get("amino_acid_substitution") != r0.get("amino_acid_substitution")
            or not _is_finite_scalar(r2.get("median_plasmid_copy_number"))
        ):
            diffs.append("plasmid_tag_rename_perturbed_substitution_or_copy")
        out["renaming_discrepancies"] = diffs
        out["id_renaming_invariant_ok"] = (len(diffs) == 0)
    except Exception as exc:
        if out["public_output_contract_ok"] is None:
            out["public_output_contract_ok"] = False
        out["public_calibration_ok"] = False
        out["id_renaming_invariant_ok"] = False
        out["renaming_discrepancies"] = [f"{type(exc).__name__}:{exc}"]
    finally:
        _stop_worker(proc)

# 2. `ho08` Viscoelastic Prony Memory: Public calibration + Affine drift invariance + Spring equivariance + Soft prefix warning!
elif os.path.isfile("/holdout_task/public_rheology.pkl") and os.path.isfile("/holdout_task/prony_rheology.py"):
    out["probe_applicable"] = True
    out["symmetry_probe_family"] = "affine_thermal_drift_gauge_invariance"
    proc = None
    try:
        proc, call_fn = _start_candidate_worker("prony_rheology")
        blob = pickle.load(open("/holdout_task/public_rheology.pkl", "rb"))
        exp0 = copy.deepcopy(blob["experiment"])
        gt_te = np.asarray(blob["gt_test_stress"], dtype=float)
        gt_g = float(blob["gt_G_inf"])
        r0 = call_fn("fit_prony_and_predict_stress", copy.deepcopy(exp0))
        miss = []
        if not isinstance(r0, dict) or "test_stress" not in r0 or "G_inf" not in r0:
            miss.append("missing_test_stress_or_G_inf")
        elif not (_is_finite_array(r0["test_stress"], gt_te.shape) and _is_finite_scalar(r0["G_inf"]) and float(r0["G_inf"]) >= 0.0):
            miss.append("non_finite_or_invalid_shape_or_negative_G_inf")
        else:
            pred0 = np.asarray(r0["test_stress"], dtype=float)
            rel_rmse = float(np.sqrt(np.mean((pred0 - gt_te) ** 2)) / np.sqrt(np.mean(gt_te ** 2)))
            g_rel = abs(float(r0["G_inf"]) - gt_g) / max(abs(gt_g), 1e-9)
            out["public_consistency_metrics"] = {"pub_rel_rmse": round(rel_rmse, 5), "pub_G_inf_rel_err": round(g_rel, 5)}
            if not (rel_rmse <= 0.022 and g_rel <= 0.06):
                miss.append(f"public_calibration_exceeded(rel_rmse={rel_rmse:.4f},G_err={g_rel:.4f})")
        out["missing_or_null_keys"] = miss
        out["public_output_contract_ok"] = (len(miss) == 0)
        out["public_calibration_ok"] = (len(miss) == 0)

        diffs = []
        if isinstance(r0, dict) and _is_finite_array(r0.get("test_stress"), gt_te.shape) and _is_finite_scalar(r0.get("G_inf")):
            pred0 = np.asarray(r0["test_stress"], dtype=float)
            g0_val = float(r0["G_inf"])
            t_tr = np.asarray(exp0["train_t"], dtype=float)
            te_strain = np.asarray(exp0["test_strain"], dtype=float)

            # (a) Strict gauge invariance under extra affine load-cell thermal drift `+(1.35 - 0.09 * t)`
            exp_gauge = copy.deepcopy(exp0)
            exp_gauge["train_stress_raw"] = np.asarray(exp0["train_stress_raw"], dtype=float) + (1.35 - 0.09 * t_tr)
            r1 = call_fn("fit_prony_and_predict_stress", exp_gauge)
            if not (isinstance(r1, dict) and _is_finite_array(r1.get("test_stress"), gt_te.shape) and _is_finite_scalar(r1.get("G_inf"))):
                diffs.append("affine_drift_output_non_finite_or_bad_shape")
            else:
                pred1 = np.asarray(r1["test_stress"], dtype=float)
                gauge_stress_drift = float(np.sqrt(np.mean((pred1 - pred0) ** 2)) / max(np.sqrt(np.mean(pred0 ** 2)), 1e-9))
                gauge_g_drift = abs(float(r1["G_inf"]) - g0_val) / max(abs(g0_val), 1e-9)
                if not (gauge_stress_drift <= 0.015 and gauge_g_drift <= 0.03):
                    diffs.append(f"affine_drift_non_invariant(stress_drift={gauge_stress_drift:.4f},G_drift={gauge_g_drift:.4f})")

            # (b) Parallel-spring equivariance: `+3.5 * strain` MUST shift `G_inf` by `+3.5` AND `test_stress` by `+3.5 * test_strain`!
            exp_eq = copy.deepcopy(exp0)
            exp_eq["train_stress_raw"] = np.asarray(exp0["train_stress_raw"], dtype=float) + 3.5 * np.asarray(exp0["train_strain"], dtype=float)
            r_eq = call_fn("fit_prony_and_predict_stress", exp_eq)
            if not (isinstance(r_eq, dict) and _is_finite_array(r_eq.get("test_stress"), gt_te.shape) and _is_finite_scalar(r_eq.get("G_inf"))):
                diffs.append("parallel_spring_output_non_finite_or_bad_shape")
            else:
                pred_eq = np.asarray(r_eq["test_stress"], dtype=float)
                expected_eq_stress = pred0 + 3.5 * te_strain
                eq_g_err = abs((float(r_eq["G_inf"]) - g0_val) - 3.5) / 3.5
                eq_stress_err = float(np.sqrt(np.mean((pred_eq - expected_eq_stress) ** 2)) / max(np.sqrt(np.mean(expected_eq_stress ** 2)), 1e-9))
                if not (eq_g_err <= 0.045 and eq_stress_err <= 0.025):
                    diffs.append(f"equilibrium_modulus_or_stress_non_equivariant(G_err={eq_g_err:.4f},stress_err={eq_stress_err:.4f})")

            # (c) Soft identifiability diagnostic (NOT a symmetry refutation!): 65% prefix `[0, 19.5s]` checks slow-mode collinearity
            n_sub = int(0.65 * len(t_tr))
            exp_sub = copy.deepcopy(exp0)
            for k_s in ("train_t", "train_strain", "train_stress_raw"):
                exp_sub[k_s] = np.asarray(exp0[k_s], dtype=float)[:n_sub]
            try:
                r_sub = call_fn("fit_prony_and_predict_stress", exp_sub)
                if isinstance(r_sub, dict) and _is_finite_scalar(r_sub.get("G_inf")):
                    sub_g_drift = abs(float(r_sub["G_inf"]) - g0_val) / max(abs(g0_val), 1e-9)
                    out["identifiability_metrics"] = {"prefix_65pct_G_inf_rel_drift": round(sub_g_drift, 5)}
                    if not (sub_g_drift <= 0.08):
                        out["identifiability_warning"] = True
                else:
                    out["identifiability_warning"] = True
            except Exception:
                out["identifiability_warning"] = True
        else:
            diffs.append("nominal_output_invalid_cannot_verify_invariance")

        out["renaming_discrepancies"] = diffs
        out["id_renaming_invariant_ok"] = (len(diffs) == 0)
    except Exception as exc:
        if out["public_output_contract_ok"] is None:
            out["public_output_contract_ok"] = False
        out["public_calibration_ok"] = False
        out["id_renaming_invariant_ok"] = False
        out["renaming_discrepancies"] = [f"{type(exc).__name__}:{exc}"]
    finally:
        _stop_worker(proc)

# 3. `ho06` 384-Well Confounded IC50: Strict finite check (`NaN` fail-closed!) + IC50 & Hill-slope renaming equivariance!
elif os.path.isfile("/holdout_task/public_screen.pkl") and os.path.isfile("/holdout_task/assay_ic50.py"):
    out["probe_applicable"] = True
    out["symmetry_probe_family"] = "compound_renaming_and_well_permutation_equivariance"
    proc = None
    try:
        proc, call_fn = _start_candidate_worker("assay_ic50")
        blob = pickle.load(open("/holdout_task/public_screen.pkl", "rb"))
        plates0 = copy.deepcopy(blob["plates"])
        gt = blob["gt"]
        est0 = call_fn("estimate_compound_hill_parameters", copy.deepcopy(plates0))
        miss = []
        if not isinstance(est0, dict) or set(est0.keys()) != set(gt.keys()):
            miss.append("missing_or_extra_compound_keys")
        elif not all(
            isinstance(est0[k], dict)
            and _is_finite_scalar(est0[k].get("log10_ic50_uM"))
            and _is_finite_scalar(est0[k].get("hill_slope"))
            for k in gt
        ):
            miss.append("non_finite_ic50_or_hill_slope")
        else:
            ic_err = max(abs(float(est0[k]["log10_ic50_uM"]) - float(v["log10_ic50_uM"])) for k, v in gt.items())
            sl_err = max(abs(float(est0[k]["hill_slope"]) - float(v["hill_slope"])) for k, v in gt.items())
            out["public_consistency_metrics"] = {"pub_max_ic50_err": round(ic_err, 4), "pub_max_slope_err": round(sl_err, 4)}
            if not (ic_err <= 0.09 and sl_err <= 0.18):
                miss.append(f"public_ic50_or_slope_err_exceeded(ic50={ic_err:.4f},slope={sl_err:.4f})")
        out["missing_or_null_keys"] = miss
        out["public_output_contract_ok"] = (len(miss) == 0)
        out["public_calibration_ok"] = (len(miss) == 0)

        cmap = {k: f"DRUG_SYM_{i:02d}" for i, k in enumerate(sorted(gt.keys()))}
        plates1 = []
        for p in reversed(plates0):
            w_new = []
            for w in reversed(p["wells"]):
                w2 = dict(w)
                if w2.get("compound_id") in cmap:
                    w2["compound_id"] = cmap[w2["compound_id"]]
                w_new.append(w2)
            plates1.append({"plate_id": "REN_" + str(p["plate_id"]), "wells": w_new})
        est1 = call_fn("estimate_compound_hill_parameters", plates1)
        diffs = []
        if not isinstance(est1, dict) or set(est1.keys()) != set(cmap.values()):
            diffs.append("renamed_keyset_mismatch")
        elif not isinstance(est0, dict) or set(est0.keys()) != set(gt.keys()):
            diffs.append("nominal_est_invalid")
        else:
            for orig_k, ren_k in cmap.items():
                v0, v1 = est0[orig_k], est1[ren_k]
                if not (
                    isinstance(v0, dict) and isinstance(v1, dict)
                    and _is_finite_scalar(v0.get("log10_ic50_uM")) and _is_finite_scalar(v1.get("log10_ic50_uM"))
                    and _is_finite_scalar(v0.get("hill_slope")) and _is_finite_scalar(v1.get("hill_slope"))
                ):
                    diffs.append(f"non_finite_under_rename:{orig_k}"); break
                if not (
                    abs(float(v0["log10_ic50_uM"]) - float(v1["log10_ic50_uM"])) <= 1e-3
                    and abs(float(v0["hill_slope"]) - float(v1["hill_slope"])) <= 1e-3
                ):
                    diffs.append(f"ic50_or_slope_rename_drift:{orig_k}"); break
        out["renaming_discrepancies"] = diffs
        out["id_renaming_invariant_ok"] = (len(diffs) == 0)
    except Exception as exc:
        out["public_output_contract_ok"] = False
        out["public_calibration_ok"] = False
        out["id_renaming_invariant_ok"] = False
        out["renaming_discrepancies"] = [f"{type(exc).__name__}:{exc}"]
    finally:
        _stop_worker(proc)

# 4. `ho04` Strapdown Magnetometer: Multi-permutation equivariance + calibration invariance + finite checks!
elif os.path.isfile("/holdout_task/public_mag.pkl") and os.path.isfile("/holdout_task/mag_calibrator.py"):
    out["probe_applicable"] = True
    out["symmetry_probe_family"] = "query_row_permutation_equivariance"
    proc = None
    try:
        proc, call_fn = _start_candidate_worker("mag_calibrator")
        blob = pickle.load(open("/holdout_task/public_mag.pkl", "rb"))
        s0 = copy.deepcopy(blob["session"]); gt = blob["gt"]
        n_te = len(s0["test_m_raw"])
        r0 = call_fn("calibrate_and_estimate_heading", copy.deepcopy(s0))
        miss = []
        if not (isinstance(r0, dict) and _is_finite_array(r0.get("heading_deg"), (n_te,)) and _is_finite_array(r0.get("b_hard"), (3,))):
            miss.append("non_finite_or_bad_shape_output")
        else:
            b_err = float(np.linalg.norm(np.asarray(r0["b_hard"], float) - np.asarray(gt["b_hard"], float)))
            dh = (np.asarray(r0["heading_deg"], float) - np.asarray(gt["gt_heading_deg"], float) + 180.0) % 360.0 - 180.0
            h_rmse = float(np.sqrt(np.mean(dh ** 2)))
            out["public_consistency_metrics"] = {"pub_heading_rmse_deg": round(h_rmse, 4), "pub_b_hard_err_uT": round(b_err, 4)}
            if not (h_rmse <= 0.45 and b_err <= 0.35):
                miss.append(f"public_mag_err_exceeded(h_rmse={h_rmse:.4f},b_err={b_err:.4f})")
        out["missing_or_null_keys"] = miss
        out["public_output_contract_ok"] = (len(miss) == 0)
        out["public_calibration_ok"] = (len(miss) == 0)

        diffs = []
        if len(miss) == 0:
            rng_p = np.random.default_rng(8404)
            perms = [np.arange(n_te)[::-1], rng_p.permutation(n_te)]
            for p_idx, perm in enumerate(perms):
                s1 = copy.deepcopy(s0)
                s1["test_m_raw"] = np.asarray(s0["test_m_raw"])[perm]
                s1["test_roll_pitch_rad"] = np.asarray(s0["test_roll_pitch_rad"])[perm]
                r1 = call_fn("calibrate_and_estimate_heading", s1)
                if not (isinstance(r1, dict) and _is_finite_array(r1.get("heading_deg"), (n_te,)) and _is_finite_array(r1.get("b_hard"), (3,))):
                    diffs.append(f"perm_{p_idx}_non_finite_or_bad_shape"); break
                dh_perm = (np.asarray(r1["heading_deg"], float) - np.asarray(r0["heading_deg"], float)[perm] + 180.0) % 360.0 - 180.0
                b_drift = float(np.linalg.norm(np.asarray(r1["b_hard"], float) - np.asarray(r0["b_hard"], float)))
                if not (float(np.max(np.abs(dh_perm))) <= 1e-2 and b_drift <= 1e-3):
                    diffs.append(f"query_permutation_not_equivariant(p={p_idx})"); break
        else:
            diffs.append("nominal_mag_invalid")
        out["renaming_discrepancies"] = diffs
        out["id_renaming_invariant_ok"] = (len(diffs) == 0)
    except Exception as exc:
        out["public_output_contract_ok"] = False
        out["public_calibration_ok"] = False
        out["id_renaming_invariant_ok"] = False
        out["renaming_discrepancies"] = [f"{type(exc).__name__}:{exc}"]
    finally:
        _stop_worker(proc)

# 5. `ho05` Graphical Precision: Non-cyclic general vertex permutations on BOTH `adjacency` & `sparse_precision`!
elif os.path.isfile("/holdout_task/public_graph.npz") and os.path.isfile("/holdout_task/deconfound_graph.py"):
    out["probe_applicable"] = True
    out["symmetry_probe_family"] = "vertex_permutation_equivariance"
    proc = None
    try:
        proc, call_fn = _start_candidate_worker("deconfound_graph")
        z = np.load("/holdout_task/public_graph.npz")
        X = np.asarray(z["X"], float); gt_adj = np.asarray(z["gt_adj"], bool); p = X.shape[1]
        r0 = call_fn("recover_sparse_precision_support", X.copy())
        miss = []
        if not (isinstance(r0, dict) and "adjacency" in r0 and "sparse_precision" in r0):
            miss.append("missing_adjacency_or_sparse_precision")
        elif not (_is_binary_matrix(r0["adjacency"], (p, p)) and _is_finite_array(r0["sparse_precision"], (p, p))):
            miss.append("invalid_shape_nonbinary_adjacency_or_non_finite_precision")
        else:
            adj0 = np.asarray(r0["adjacency"], float).astype(bool)
            S0 = np.asarray(r0["sparse_precision"], float)
            iu = np.triu_indices(p, 1)
            tp = int(np.sum(adj0[iu] & gt_adj[iu])); fp = int(np.sum(adj0[iu] & ~gt_adj[iu])); fn = int(np.sum(~adj0[iu] & gt_adj[iu]))
            f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
            min_eig = float(np.linalg.eigvalsh(0.5 * (S0 + S0.T))[0])
            supp_ok = bool(np.any(adj0) and float(np.min(np.abs(S0[adj0]))) > 1e-5)
            out["public_consistency_metrics"] = {"pub_edge_f1": round(f1, 4), "min_eig": round(min_eig, 6)}
            if (
                np.any(np.diag(adj0))
                or not np.array_equal(adj0, adj0.T)
                or not np.allclose(S0, S0.T, atol=1e-3)
                or not (min_eig > 1e-6 and f1 >= 0.86 and supp_ok)
            ):
                miss.append(f"invalid_spd_or_low_pub_f1(f1={f1:.4f},eig={min_eig:.5f})")
        out["missing_or_null_keys"] = miss
        out["public_output_contract_ok"] = (len(miss) == 0)
        out["public_calibration_ok"] = (len(miss) == 0)

        diffs = []
        if isinstance(r0, dict) and "adjacency" in r0 and "sparse_precision" in r0 and _is_binary_matrix(r0["adjacency"], (p, p)) and _is_finite_array(r0["sparse_precision"], (p, p)):
            adj0 = np.asarray(r0["adjacency"], float).astype(bool); S0 = np.asarray(r0["sparse_precision"], float)
            # Two fixed NON-CYCLIC general permutations (breaks 24-cycle ring automorphisms completely!)
            rng_g = np.random.default_rng(8505)
            perm_a = rng_g.permutation(p)
            perm_b = np.array([(5 * i + (1 if i % 2 == 0 else -1)) % p for i in range(p)], dtype=int)
            if len(set(perm_b.tolist())) != p:
                perm_b = rng_g.permutation(p)
            for p_idx, perm in enumerate((perm_a, perm_b)):
                r1 = call_fn("recover_sparse_precision_support", X[:, perm].copy())
                if not (isinstance(r1, dict) and "adjacency" in r1 and _is_binary_matrix(r1.get("adjacency"), (p, p)) and _is_finite_array(r1.get("sparse_precision"), (p, p))):
                    diffs.append(f"perm_{p_idx}_invalid_or_non_finite"); break
                adj1 = np.asarray(r1["adjacency"], float).astype(bool)
                S1 = np.asarray(r1["sparse_precision"], float)
                adj0_perm = adj0[np.ix_(perm, perm)]
                S0_perm = S0[np.ix_(perm, perm)]
                s_rel = float(np.linalg.norm(S1 - S0_perm) / max(np.linalg.norm(S0_perm), 1e-9))
                if not np.array_equal(adj1, adj0_perm) or not (s_rel <= 0.05):
                    diffs.append("vertex_permutation_not_equivariant"); break
        else:
            diffs.append("nominal_graph_invalid")
        out["renaming_discrepancies"] = diffs
        out["id_renaming_invariant_ok"] = (len(diffs) == 0)
    except Exception as exc:
        out["public_output_contract_ok"] = False
        out["public_calibration_ok"] = False
        out["id_renaming_invariant_ok"] = False
        out["renaming_discrepancies"] = [f"{type(exc).__name__}:{exc}"]
    finally:
        _stop_worker(proc)

# 6. `ho10` Haldane Bioreactor Control: Step-by-step finite control validation (`NaN` fail-closed!) + rollout envelope!
elif os.path.isfile("/holdout_task/plant_sim.py") and os.path.isfile("/holdout_task/bioreactor_ctrl.py"):
    out["probe_applicable"] = True
    out["symmetry_probe_family"] = "closed_loop_safety_and_perturbation_invariance"
    proc = None
    try:
        import plant_sim
        proc, call_fn = _start_candidate_worker("bioreactor_ctrl")

        def _checked_step(obs, mem):
            res_step = call_fn("compute_control_step", obs, mem)
            if not (isinstance(res_step, (tuple, list)) and len(res_step) == 2 and isinstance(res_step[0], dict)):
                raise ValueError("control_step_must_return_tuple_of_cmd_dict_and_state")
            u_cmd, next_mem = res_step
            if not (
                "feed_rate_L_per_h" in u_cmd
                and "coolant_flow_frac" in u_cmd
                and _is_finite_scalar(u_cmd["feed_rate_L_per_h"])
                and _is_finite_scalar(u_cmd["coolant_flow_frac"])
            ):
                raise ValueError("non_finite_or_missing_control_command")
            return {"feed_rate_L_per_h": float(u_cmd["feed_rate_L_per_h"]), "coolant_flow_frac": float(u_cmd["coolant_flow_frac"])}, next_mem

        miss = []
        scen_logs = []
        for sc in plant_sim.NOMINAL_SCENARIOS:
            rr = plant_sim.run_closed_loop(_checked_step, sc, seed=10)
            t_max, v_max, bio = rr.get("max_T_C"), rr.get("max_V_L"), rr.get("final_biomass_g")
            if not (_is_finite_scalar(t_max) and _is_finite_scalar(v_max) and _is_finite_scalar(bio)):
                miss.append(f"nominal_non_finite_state({sc['name']})")
            else:
                scen_logs.append({"scenario": sc["name"], "seed": 10, "max_T_C": round(float(t_max), 3), "max_V_L": round(float(v_max), 3), "final_biomass_g": round(float(bio), 3)})
                if not (float(t_max) <= 34.5 and float(v_max) <= 12.0 and float(bio) >= 62.0):
                    miss.append(f"nominal_violation({sc['name']}:T={float(t_max):.2f},V={float(v_max):.2f},B={float(bio):.2f})")
        out["missing_or_null_keys"] = miss
        out["public_output_contract_ok"] = (len(miss) == 0)
        out["public_calibration_ok"] = (len(miss) == 0)

        nom0 = dict(plant_sim.NOMINAL_SCENARIOS[0])
        diffs = []
        for idx_p, p_sc in enumerate([
            {**nom0, "name": "pub_env_hi", "mu0": round(float(nom0["mu0"]) * 1.10, 4), "Ki": round(float(nom0["Ki"]) * 0.85, 4), "ua_scale": 0.78, "heat_pulse": 2.0},
            {**nom0, "name": "pub_env_lo", "mu0": round(float(nom0["mu0"]) * 0.88, 4), "Ki": round(float(nom0["Ki"]) * 1.12, 4), "ua_scale": 0.78, "heat_pulse": 1.6},
        ]):
            rr = plant_sim.run_closed_loop(_checked_step, p_sc, seed=20 + idx_p)
            t_max, v_max, bio = rr.get("max_T_C"), rr.get("max_V_L"), rr.get("final_biomass_g")
            if not (_is_finite_scalar(t_max) and _is_finite_scalar(v_max) and _is_finite_scalar(bio)):
                diffs.append(f"perturbed_non_finite_state({p_sc['name']})")
            else:
                scen_logs.append({"scenario": p_sc["name"], "seed": 20 + idx_p, "max_T_C": round(float(t_max), 3), "max_V_L": round(float(v_max), 3), "final_biomass_g": round(float(bio), 3)})
                if not (float(t_max) <= 34.5 and float(v_max) <= 12.0 and float(bio) >= 62.0):
                    diffs.append(f"perturbed_violation({p_sc['name']}:T={float(t_max):.2f},V={float(v_max):.2f},B={float(bio):.2f})")
        out["public_consistency_metrics"]["rollouts"] = scen_logs
        out["renaming_discrepancies"] = diffs
        out["id_renaming_invariant_ok"] = (len(diffs) == 0)
    except Exception as exc:
        out["public_output_contract_ok"] = False
        out["public_calibration_ok"] = False
        out["id_renaming_invariant_ok"] = False
        out["renaming_discrepancies"] = [f"{type(exc).__name__}:{exc}"]
    finally:
        _stop_worker(proc)

sys.stdout.write(SENTINEL + json.dumps(out, allow_nan=False) + "\n")
sys.stdout.flush()
'''


def probe_general_public_container_invariants_in_container(
    cid: str,
    task: Any,
) -> dict[str, Any]:
    """Run strictly PUBLIC output-contract & symmetry/gauge-invariance checks inside an active container `cid`."""
    _ = task
    t0 = monotonic()
    probe_sentinel = f"__HEGEL_PUB_PROBE_{secrets.token_hex(16)}__:"
    with tempfile.TemporaryDirectory(prefix="hegel_m84_genprobe_") as td:
        p_fp = Path(td) / "_m84_probe.py"
        p_fp.write_text(_GENERAL_PUBLIC_INVARIANCE_PROBE_PY, encoding="utf-8")
        subprocess.run(["docker", "cp", str(p_fp), f"{cid}:/tmp/_m84_probe.py"], check=True, timeout=20)
    ex = m6f.exec_container_bash(
        cid,
        f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HEGEL_PUB_PROBE_SENTINEL={probe_sentinel} "
        "python3 -I /tmp/_m84_probe.py; rc=$?; rm -f /tmp/_m84_probe.py; exit $rc",
        timeout_sec=75,
    )
    parsed: dict[str, Any] | None = None
    if int(ex.get("exit_code", 1)) == 0:
        for ln in reversed(str(ex.get("stdout") or "").splitlines()):
            s = ln.strip()
            if s.startswith(probe_sentinel):
                try:
                    cand_obj = json.loads(s[len(probe_sentinel) :].strip())
                    if isinstance(cand_obj, dict):
                        parsed = cand_obj
                        break
                except json.JSONDecodeError:
                    pass
    if parsed is None:
        parsed = {
            "probe_applicable": False,
            "public_output_contract_ok": False,
            "id_renaming_invariant_ok": False,
            "probe_exit_code": ex.get("exit_code"),
            "error": str(ex.get("stderr") or "missing_authenticated_probe_sentinel")[-400:],
        }
    parsed["wall_ms"] = round((monotonic() - t0) * 1000.0, 2)
    return parsed


def probe_public_cohort_contract_and_invariance(
    *,
    task: HoldoutOODTask,
    candidate_py_code: str,
    required_non_null_str_keys: tuple[str, ...] = ("shared_plasmid_family_tag", "amino_acid_substitution"),
    required_finite_float_keys: tuple[str, ...] = ("median_plasmid_copy_number",),
) -> dict[str, Any]:
    """Standalone container wrapper running `probe_general_public_container_invariants_in_container` on `task`."""
    _ = (required_non_null_str_keys, required_finite_float_keys)
    ensure_clean_scipy_sandbox_image()
    artifact_rel = task.declared_artifacts[0]
    art_dir = str(Path(artifact_rel).parent)
    cid = m6f.start_detached_container(
        CLEAN_SCIPY_SANDBOX_IMAGE,
        memory_limit="2048m",
        cpus="2",
        network_mode="none",
    )
    try:
        m6f.exec_container_bash(cid, f"mkdir -p {art_dir} /tmp", timeout_sec=10)
        with tempfile.TemporaryDirectory(prefix="hegel_m8_pubprobe_") as td:
            td_p = Path(td)
            (td_p / "_setup.py").write_text(task.public_setup_py, encoding="utf-8")
            (td_p / "cand.py").write_text(candidate_py_code, encoding="utf-8")
            subprocess.run(["docker", "cp", str(td_p / "_setup.py"), f"{cid}:/tmp/_setup.py"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(td_p / "cand.py"), f"{cid}:{artifact_rel}"], check=True, timeout=20)
        m6f.exec_container_bash(cid, "python3 /tmp/_setup.py && rm -f /tmp/_setup.py", timeout_sec=45)
        return probe_general_public_container_invariants_in_container(cid, task)
    finally:
        m6f.stop_container_safely(cid)


def recall_persisted_container_invariance_guards(
    graph: PersistentMechanismGraph | None,
    *,
    domain_tag: str = "",
) -> list[TypedMechanism]:
    """Autonomously recall verified `container_invariance_guard` mechanisms matching declared `scope_guard`."""
    if graph is None:
        return []
    out: list[TypedMechanism] = []
    obs_ctx = {"symmetry_probe_family": domain_tag} if domain_tag else {}
    for _, mech in sorted(graph.mechanisms.items()):
        if mech.role == "container_invariance_guard" and mech.epistemic_status == "SUPPORTED_IN_SCOPE":
            if domain_tag and mech.scope_guard and mech.scope_guard.get("field") == "symmetry_probe_family":
                if evaluate_predicate_rule(mech.scope_guard, obs_ctx) == "FALSE":
                    continue
            elif domain_tag and mech.provenance_note and domain_tag not in mech.provenance_note:
                continue
            out.append(mech)
    return out


def commit_verified_container_guard(
    graph: PersistentMechanismGraph | None,
    pending_guard: dict[str, Any] | None,
    *,
    checkpoint_path: Path | None = None,
) -> str | None:
    """Commit a `pending_guard` to `graph` ONLY AFTER all public probes AND artifact contracts pass!"""
    if graph is None or not isinstance(pending_guard, dict):
        return None
    mech = TypedMechanism.from_dict(pending_guard)
    mech.epistemic_status = "SUPPORTED_IN_SCOPE"
    if mech.mechanism_id in graph.mechanisms:
        existing = graph.mechanisms[mech.mechanism_id]
        existing.dependent_task_ids |= mech.dependent_task_ids
        existing.epistemic_status = "SUPPORTED_IN_SCOPE"
        graph.sync_authoritative_state_to_forest()
    else:
        graph.register_mechanism(mech, tree_id="tree_container_invariance")
    if checkpoint_path is not None:
        graph.save_checkpoint(checkpoint_path)
    return mech.mechanism_id


def merge_worker_graph_into_shared(
    shared_graph: PersistentMechanismGraph,
    worker_graph: PersistentMechanismGraph,
) -> dict[str, int]:
    """Merge content-addressed mechanisms, `decision_ledger` receipts, and `action_return_history` without collision loss!"""
    merged_mechs = 0
    merged_decisions = 0
    for mid, mech in sorted(worker_graph.mechanisms.items()):
        if mid in shared_graph.mechanisms:
            shared_graph.mechanisms[mid].dependent_task_ids |= set(mech.dependent_task_ids)
            shared_graph.mechanisms[mid].inapplicable_task_ids |= set(mech.inapplicable_task_ids)
        else:
            shared_graph.register_mechanism(deepcopy(mech), tree_id="tree_container_invariance")
            merged_mechs += 1

    seen_dec_ids = {r.decision_id for r in shared_graph.decision_ledger}
    for rec in worker_graph.decision_ledger:
        if rec.decision_id not in seen_dec_ids:
            shared_graph.decision_ledger.append(deepcopy(rec))
            seen_dec_ids.add(rec.decision_id)
            merged_decisions += 1

    merged_returns = 0
    for act_k, rows in worker_graph.action_return_history.items():
        cur_rows = shared_graph.action_return_history.setdefault(act_k, [])
        seen_event_ids = {str(r["event_id"]) for r in cur_rows if isinstance(r, dict) and r.get("event_id")}
        seen_legacy = {json.dumps(r, sort_keys=True) for r in cur_rows if isinstance(r, dict) and not r.get("event_id")}
        for r in rows:
            ev_id = str(r.get("event_id") or "") if isinstance(r, dict) else ""
            if ev_id:
                if ev_id not in seen_event_ids:
                    cur_rows.append(deepcopy(r))
                    seen_event_ids.add(ev_id)
                    merged_returns += 1
            else:
                rk = json.dumps(r, sort_keys=True)
                if rk not in seen_legacy:
                    cur_rows.append(deepcopy(r))
                    seen_legacy.add(rk)
                    merged_returns += 1

    shared_graph.sync_authoritative_state_to_forest()
    return {"merged_mechanisms": merged_mechs, "merged_decisions": merged_decisions, "merged_return_events": merged_returns}


def compile_verified_repair_to_executable_skill(
    *,
    task: Any,
    repair_summary: dict[str, Any],
    forest: Any,
    archive: dict[str, dict[str, Any]],
    codex_client: Any,
    run_id: str = "m8_5_repair_compile",
    graph: PersistentMechanismGraph | None = None,
    allow_horizontal_wormholes: bool = True,
) -> dict[str, Any]:
    """Compile a public-probe & contract-verified `LocalContainerRepair` into a unit-tested `@track_wormhole_skill` Python module (`M8.5`).

    Strictly rejects unverified/syntax-only fallbacks (`unit_test_receipt['passed'] == True` inside `hegel_clean_scipy_sandbox:v1` required!)
    before consolidating into `PhilosophicalForest` + `archive` so downstream tasks execute real Python functions rather than prompt hints.
    """
    from . import m7_autogenous_forest_pretraining as m7a

    if not bool(repair_summary.get("repaired_verified")):
        return {"compiled_and_consolidated": False, "reason": "repair_not_verified"}
    rep_code = str(repair_summary.get("repaired_python_code") or "").strip()
    if not rep_code:
        return {"compiled_and_consolidated": False, "reason": "missing_repaired_python_code"}

    ensure_clean_scipy_sandbox_image()
    sandbox_cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2", network_mode="none")
    try:
        sig_task = (
            task
            if hasattr(task, "benchmark")
            else SimpleNamespace(benchmark="Hegel-OOD-Holdout-12", instruction_text=str(task.instruction_text))
        )
        pub_sigs = m6f.extract_public_task_signatures(sig_task)
        root_nid = m7a._select_axiomatic_root_for_signatures(archive, pub_sigs)
        parent_summary = {
            "node_id": root_nid,
            "principle": archive[root_nid].get("methodology_principle", ""),
        }
        art_name = Path(str(task.declared_artifacts[0])).name if getattr(task, "declared_artifacts", None) else "repaired_module.py"
        repair_nogoods = [
            {
                "nogood_rung": "RUNG_2_INVARIANCE_REPAIR",
                "assumption_refuted": str(repair_summary.get("defect_diagnosis") or "Naive primary solver violated public invariance or contract."),
                "empirical_witness": json.dumps(repair_summary.get("probe_after", {}).get("public_consistency_metrics", {}), sort_keys=True)[:240],
                "sublated_pivot": str(repair_summary.get("invariance_principle") or "Enforce symmetry/gauge invariance and finite output completeness."),
            }
        ]
        short_slug = str(getattr(task, "short_name", None) or getattr(task, "task_id", "repair_task")).split("__")[-1]
        comp = m7a.compile_episode_dialectical_sublation(
            episode_id=f"m8_5_repair_{short_slug}",
            source_slug=f"repair_{short_slug}",
            public_task_brief=str(task.instruction_text),
            inferred_signatures=pub_sigs,
            raw_python_sources={art_name: rep_code},
            nogoods_recorded=repair_nogoods,
            parent_node_summary=parent_summary,
            client=codex_client,
            sandbox_cid=sandbox_cid,
            stage_subdir=f"{run_id}__exec_repair__{short_slug}".replace(":", "_"),
            module_prefix="autogenous_repair",
        )
        ut_passed = bool((comp.get("unit_test_receipt") or {}).get("passed")) and str((comp.get("unit_test_receipt") or {}).get("mode") or "") != "static_compile_only_unverified"
        if not (comp.get("compiled_ok") and ut_passed):
            return {
                "compiled_and_consolidated": False,
                "reason": "unit_test_verification_required_failed",
                "compilation": comp,
            }
        comp["routed_signatures"] = sorted(
            set(comp.get("routed_signatures") or [])
            | {s for s in pub_sigs if s in m7a.ALL_PUBLIC_ROUTING_SIGNATURES}
        ) or ["sig:robust_physical_calibration_and_safety_margins"]
        comp["source_qualification_status"] = "verified_container_repair_operator"
        cons = m7a.consolidate_autogenous_episode_into_forest(
            forest=forest,
            archive=archive,
            compilation=comp,
            nogoods_recorded=repair_nogoods,
            allow_horizontal_wormholes=allow_horizontal_wormholes,
        )
        gid = str(repair_summary.get("persisted_guard_id") or "")
        if graph is not None and gid and gid in graph.mechanisms:
            g_mech = graph.mechanisms[gid]
            g_mech.provenance_note = (
                f"{g_mech.provenance_note} | [EXECUTABLE_SKILL node={cons.get('target_node_id')} "
                f"module={comp.get('filename')} funcs={comp.get('function_names')} sha256={str(comp.get('compiled_code_sha256'))[:16]}]"
            )
            graph.sync_authoritative_state_to_forest()
        return {
            "compiled_and_consolidated": bool(cons.get("consolidated")),
            "target_node_id": cons.get("target_node_id"),
            "module_filename": comp.get("filename"),
            "exported_functions": comp.get("function_names", []),
            "compiled_code_sha256": comp.get("compiled_code_sha256"),
            "unit_test_receipt": comp.get("unit_test_receipt"),
            "routed_signatures": comp.get("routed_signatures", []),
            "consolidation": cons,
            "llm_tokens": comp.get("llm_tokens", {}),
        }
    finally:
        m6f.stop_container_safely(sandbox_cid)


def execute_in_container_local_repair(
    *,
    probe_cid: str,
    task: Any,
    public_probe_before: dict[str, Any],
    codex_client: Any,
    graph: PersistentMechanismGraph | None = None,
    checkpoint_path: Path | None = None,
    step_index: int = 1,
    stage_subdir: str = "m8_local_container_repair",
    defer_guard_commit: bool = False,
) -> dict[str, Any]:
    """Execute 1-turn targeted `LocalContainerRepair` inside live `probe_cid` with exception-safe rollback & token preservation!"""
    _ = step_index
    t0 = monotonic()
    if not getattr(task, "declared_artifacts", None) or len(task.declared_artifacts) != 1:
        return {
            "attempted": False,
            "repaired_verified": False,
            "reason": "multi_or_non_python_artifact_unsupported",
            "llm_tokens": {"prompt": 0, "cached": 0, "completion": 0, "wall_ms": 0.0},
        }
    artifact_rel = str(task.declared_artifacts[0])
    if not artifact_rel.endswith(".py"):
        return {
            "attempted": False,
            "repaired_verified": False,
            "reason": "non_python_artifact",
            "llm_tokens": {"prompt": 0, "cached": 0, "completion": 0, "wall_ms": 0.0},
        }

    ex_read = m6f.exec_container_bash(probe_cid, f"cat {artifact_rel}", timeout_sec=15)
    orig_code = ex_read["stdout"]
    if ex_read["exit_code"] != 0 or not orig_code.strip():
        return {
            "attempted": False,
            "repaired_verified": False,
            "reason": "empty_or_missing_primary_python_file",
            "llm_tokens": {"prompt": 0, "cached": 0, "completion": 0, "wall_ms": 0.0},
        }

    fam_before = str(public_probe_before.get("symmetry_probe_family") or "")
    recalled_guards = recall_persisted_container_invariance_guards(graph, domain_tag=fam_before)
    guard_hints = [{"guard_id": m.mechanism_id, "principle": m.provenance_note} for m in recalled_guards]
    blinded_task_ref = "container_task_" + hashlib.sha256(str(task.task_id).encode("utf-8")).hexdigest()[:8]

    sys_prompt = (
        "You are the `Hegel-M8.4` Budgeted Container Mechanism & Symmetry Repair Planner.\n"
        "A primary scientific Python deliverable failed our strictly PUBLIC contract, calibration, or symmetry/gauge-invariance "
        "probe inside the container.\n"
        "Synthesize a complete, self-contained replacement Python module (`repaired_python_code`) for `declared_artifact` "
        "that fixes the diagnosed defect from first principles (projecting out declared nuisance drifts/gradients, preserving "
        "sample/variable permutation equivariance and identifier-renaming invariance, and populating all required keys with "
        "finite values) without any hardcoded test-seed or prefix shortcuts!"
    )
    user_payload = json.dumps(
        {
            "task_ref": blinded_task_ref,
            "task_instruction": task.instruction_text,
            "declared_artifact": artifact_rel,
            "public_probe_diagnostics": public_probe_before,
            "recalled_persistent_invariance_guards": guard_hints,
            "current_candidate_python_code": orig_code,
        },
        ensure_ascii=False,
    )
    llm_res = codex_client.call_structured(
        stage_subdir=stage_subdir,
        system_prompt=sys_prompt,
        conversation_turns=[{"role": "user", "content": user_payload}],
        output_schema=M8_CONTAINER_LOCAL_REPAIR_SCHEMA,
    )
    p_tok = int(llm_res.get("prompt_tokens") or 0)
    k_tok = int(llm_res.get("cached_tokens") or 0)
    c_tok = int(llm_res.get("completion_tokens") or 0)
    llm_w_ms = float(llm_res.get("wall_ms") or 0.0)
    tok_bundle = {"prompt": p_tok, "cached": k_tok, "completion": c_tok, "wall_ms": llm_w_ms}

    if not (llm_res.get("ok") and isinstance(llm_res.get("parsed"), dict)):
        err_msg = str(llm_res.get("error") or "local_repair_llm_error")
        return {
            "attempted": True,
            "repaired_verified": False,
            "reason": err_msg,
            "recalled_guard_ids": [m.mechanism_id for m in recalled_guards],
            "persisted_guard_id": None,
            "pending_guard": None,
            "llm_tokens": tok_bundle,
            "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        }

    parsed_repair = llm_res["parsed"]
    repaired_code = str(parsed_repair["repaired_python_code"])
    try:
        ast.parse(repaired_code)
    except SyntaxError as exc:
        return {
            "attempted": True,
            "repaired_verified": False,
            "attempted_repaired_python_code": repaired_code,
            "reason": f"repaired_syntax_error:{exc}",
            "recalled_guard_ids": [m.mechanism_id for m in recalled_guards],
            "persisted_guard_id": None,
            "pending_guard": None,
            "llm_tokens": tok_bundle,
            "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        }

    orig_sha = hashlib.sha256(orig_code.encode("utf-8")).hexdigest()
    rep_sha = hashlib.sha256(repaired_code.encode("utf-8")).hexdigest()

    # Exception-safe staging + re-probing: ALWAYS restore `orig_code` on False OR any exception, and NEVER lose `tok_bundle`!
    probe_after: dict[str, Any] = {}
    repaired_ok = False
    probe_exc_reason: str | None = None
    staged_repaired = False
    with tempfile.TemporaryDirectory(prefix="hegel_m84_stage_") as td:
        td_p = Path(td)
        (td_p / "repaired.py").write_text(repaired_code, encoding="utf-8")
        (td_p / "orig_backup.py").write_text(orig_code, encoding="utf-8")
        try:
            subprocess.run(["docker", "cp", str(td_p / "repaired.py"), f"{probe_cid}:{artifact_rel}"], check=True, timeout=20)
            staged_repaired = True
            probe_after = probe_general_public_container_invariants_in_container(probe_cid, task)
            repaired_ok = bool(
                probe_after.get("public_output_contract_ok") is True
                and probe_after.get("id_renaming_invariant_ok") is True
            )
        except Exception as exc:
            repaired_ok = False
            probe_exc_reason = f"probe_after_exception:{type(exc).__name__}:{exc}"
        finally:
            if staged_repaired and not repaired_ok:
                try:
                    subprocess.run(["docker", "cp", str(td_p / "orig_backup.py"), f"{probe_cid}:{artifact_rel}"], check=True, timeout=20)
                except Exception as rb_exc:
                    probe_exc_reason = f"{probe_exc_reason or 'probe_failed'};rollback_error:{type(rb_exc).__name__}:{rb_exc}"

    pending_guard_dict: dict[str, Any] | None = None
    persisted_guard_id: str | None = None
    if repaired_ok:
        fam_tag = str(probe_after.get("symmetry_probe_family") or fam_before or "cohort_invariance")
        raw_gname = "".join(ch if ch.isalnum() else "_" for ch in str(parsed_repair["invariance_guard_name"]).lower()).strip("_")
        principle_str = str(parsed_repair["invariance_principle"]).strip()
        content_sha8 = hashlib.sha256(f"{fam_tag}|{principle_str}".encode("utf-8")).hexdigest()[:8]
        guard_id = f"mech_guard_{raw_gname[:20]}_{content_sha8}"
        guard_mech = TypedMechanism(
            mechanism_id=guard_id,
            parent_id=None,
            role="container_invariance_guard",
            feature_names=("public_output_contract_ok", "id_renaming_invariant_ok"),
            scope_guard={"field": "symmetry_probe_family", "op": "==", "value": fam_tag},
            dependent_task_ids={str(task.task_id)},
            epistemic_status="CANDIDATE_REPAIR_HYPOTHESIS",
            provenance_note=(
                f"{fam_tag} [{M8_PUBLIC_PROBE_PROTOCOL_VERSION}, orig={orig_sha[:10]}->rep={rep_sha[:10]}]: "
                f"{principle_str}"
            ),
        )
        pending_guard_dict = guard_mech.to_dict()
        if graph is not None and not defer_guard_commit:
            persisted_guard_id = commit_verified_container_guard(graph, pending_guard_dict, checkpoint_path=checkpoint_path)

    return {
        "attempted": True,
        "repaired_verified": repaired_ok,
        "reason": ("ok" if repaired_ok else (probe_exc_reason or "public_probe_rejected")),
        "raw_candidate_sha256": orig_sha,
        "repaired_candidate_sha256": rep_sha,
        "repaired_python_code": repaired_code if repaired_ok else None,
        "attempted_repaired_python_code": repaired_code,
        "probe_after": probe_after,
        "recalled_guard_ids": [m.mechanism_id for m in recalled_guards],
        "persisted_guard_id": persisted_guard_id,
        "pending_guard": pending_guard_dict,
        "invariance_guard_name": parsed_repair.get("invariance_guard_name"),
        "invariance_principle": parsed_repair.get("invariance_principle"),
        "defect_diagnosis": parsed_repair.get("defect_diagnosis"),
        "llm_tokens": tok_bundle,
        "wall_ms": round((monotonic() - t0) * 1000.0, 2),
    }


def choose_and_execute_container_mechanism_action(
    *,
    task: HoldoutOODTask,
    primary_candidate_py: str,
    graph: PersistentMechanismGraph,
    codex_client: Any | None,
    budget_remaining: float = 1.0,
    step_index: int = 1,
) -> dict[str, Any]:
    """Standalone pre-payment budgeted container planner (`CommitPrimary1x` / `LocalContainerRepair` / `SpawnDialecticalChallenger`)."""
    t0 = monotonic()
    ensure_clean_scipy_sandbox_image()
    artifact_rel = task.declared_artifacts[0]
    art_dir = str(Path(artifact_rel).parent)

    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2", network_mode="none")
    try:
        m6f.exec_container_bash(cid, f"mkdir -p {art_dir} /tmp", timeout_sec=10)
        with tempfile.TemporaryDirectory(prefix="hegel_m8_act_") as td:
            td_p = Path(td)
            (td_p / "_setup.py").write_text(task.public_setup_py, encoding="utf-8")
            (td_p / "cand.py").write_text(primary_candidate_py, encoding="utf-8")
            subprocess.run(["docker", "cp", str(td_p / "_setup.py"), f"{cid}:/tmp/_setup.py"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(td_p / "cand.py"), f"{cid}:{artifact_rel}"], check=True, timeout=20)
        m6f.exec_container_bash(cid, "python3 /tmp/_setup.py && rm -f /tmp/_setup.py", timeout_sec=45)
        probe_before = probe_general_public_container_invariants_in_container(cid, task)
        fam_b = str(probe_before.get("symmetry_probe_family") or "")
        recalled_guards = recall_persisted_container_invariance_guards(graph, domain_tag=fam_b)

        both_ok = bool(probe_before.get("public_output_contract_ok") and probe_before.get("id_renaming_invariant_ok"))
        pre_state = {
            "task_id": task.task_id,
            "probe_protocol_version": M8_PUBLIC_PROBE_PROTOCOL_VERSION,
            "symmetry_probe_family": probe_before.get("symmetry_probe_family"),
            "public_output_contract_ok": bool(probe_before.get("public_output_contract_ok")),
            "public_calibration_ok": bool(probe_before.get("public_calibration_ok")),
            "id_renaming_invariant_ok": bool(probe_before.get("id_renaming_invariant_ok")),
            "identifiability_warning": bool(probe_before.get("identifiability_warning")),
            "missing_or_null_keys": list(probe_before.get("missing_or_null_keys") or []),
            "renaming_discrepancies": list(probe_before.get("renaming_discrepancies") or []),
            "recalled_invariance_guard_ids": [m.mechanism_id for m in recalled_guards],
            "budget_remaining_before_step": round(float(budget_remaining), 4),
        }

        if both_ok:
            rec = ResearchDecisionRecord(
                decision_id=f"m8_cnt_mech_{step_index:03d}_{task.task_id}",
                step_index=step_index,
                task_id=task.task_id,
                pre_action_state=pre_state,
                available_actions=["CommitPrimary1x"],
                selected_action="CommitPrimary1x",
                selected_candidate_id="primary_candidate_public_envelope_supported",
                execution_status="EXECUTED_OK",
                actual_cost=0.0,
                estimated_selected_cost=0.0,
                public_val_log_predictive_delta=0.0,
                collateral_min_delta=0.0,
                rationale="Primary candidate satisfied public output contract, calibration/witnesses, and symmetry probe.",
            )
            graph.decision_ledger.append(rec)
            graph.action_return_history.setdefault("CommitPrimary1x", []).append(
                {
                    "event_id": rec.decision_id,
                    "task_id": str(task.task_id),
                    "step_index": int(step_index),
                    "metric_kind": "observed_public_envelope_pass",
                    "observed_public_acceptance": 1.0,
                    "gain_nats": 0.0,
                    "cost_spent": 0.0,
                }
            )
            graph.sync_authoritative_state_to_forest()
            return {
                "selected_action": "CommitPrimary1x",
                "final_python_code": primary_candidate_py,
                "probe_before": probe_before,
                "probe_after": probe_before,
                "recalled_guard_ids": [m.mechanism_id for m in recalled_guards],
                "persisted_guard_id": None,
                "decision_record": rec.to_dict(),
                "wall_ms": round((monotonic() - t0) * 1000.0, 2),
            }

        if codex_client is None or float(budget_remaining) < NOMINAL_LOCAL_REPAIR_COST:
            fallback_act = "SpawnDialecticalChallenger" if float(budget_remaining) >= 1.0 else "CommitPrimary1x"
            rec_fb = ResearchDecisionRecord(
                decision_id=f"m8_cnt_mech_{step_index:03d}_{task.task_id}",
                step_index=step_index,
                task_id=task.task_id,
                pre_action_state=pre_state,
                available_actions=["CommitPrimary1x", "LocalContainerRepair", "SpawnDialecticalChallenger"],
                selected_action=fallback_act,
                selected_candidate_id="budget_or_client_gated_fallback",
                execution_status="BUDGET_GATED_SKIP",
                actual_cost=0.0,
                estimated_selected_cost=(1.0 if fallback_act == "SpawnDialecticalChallenger" else 0.0),
                public_val_log_predictive_delta=0.0,
                collateral_min_delta=0.0,
                rationale="Insufficient budget or missing codex_client for paid `LocalContainerRepair`.",
            )
            graph.decision_ledger.append(rec_fb)
            graph.sync_authoritative_state_to_forest()
            return {
                "selected_action": fallback_act,
                "final_python_code": primary_candidate_py,
                "probe_before": probe_before,
                "probe_after": probe_before,
                "recalled_guard_ids": [m.mechanism_id for m in recalled_guards],
                "persisted_guard_id": None,
                "decision_record": rec_fb.to_dict(),
                "wall_ms": round((monotonic() - t0) * 1000.0, 2),
            }

        rep_out = execute_in_container_local_repair(
            probe_cid=cid,
            task=task,
            public_probe_before=probe_before,
            codex_client=codex_client,
            graph=graph,
            step_index=step_index,
            stage_subdir=f"m8_local_container_repair_{step_index:02d}",
        )
        tok = rep_out["llm_tokens"]
        repaired_ok = bool(rep_out.get("repaired_verified"))
        attempt_events = [
            {
                "action": "LocalContainerRepair",
                "cost_spent": NOMINAL_LOCAL_REPAIR_COST,
                "prompt_tokens": tok["prompt"],
                "completion_tokens": tok["completion"],
                "status": "OK" if repaired_ok else "FAILED_OR_REJECTED",
                "note": str(rep_out.get("defect_diagnosis") or rep_out.get("reason") or ""),
            }
        ]
        if repaired_ok:
            selected_act = "LocalContainerRepair"
            final_code = str(rep_out["repaired_python_code"])
            exec_st = "EXECUTED_OK"
        elif float(budget_remaining) - NOMINAL_LOCAL_REPAIR_COST >= 1.0:
            selected_act = "SpawnDialecticalChallenger"
            final_code = primary_candidate_py
            exec_st = "ESCALATED_AFTER_LOCAL_REPAIR_REJECTED"
        else:
            selected_act = "CommitPrimary1x"
            final_code = primary_candidate_py
            exec_st = "EXECUTION_ERROR" if not rep_out.get("probe_after") else "PROBE_REJECTED"

        rec_done = ResearchDecisionRecord(
            decision_id=f"m8_cnt_mech_{step_index:03d}_{task.task_id}",
            step_index=step_index,
            task_id=task.task_id,
            pre_action_state=deepcopy(pre_state),
            available_actions=["CommitPrimary1x", "LocalContainerRepair", "SpawnDialecticalChallenger"],
            selected_action=selected_act,
            selected_candidate_id=rep_out.get("persisted_guard_id") or "local_repair_unverified",
            execution_status=exec_st,
            actual_cost=NOMINAL_LOCAL_REPAIR_COST,
            estimated_selected_cost=NOMINAL_LOCAL_REPAIR_COST,
            total_search_cost_spent=NOMINAL_LOCAL_REPAIR_COST,
            total_prompt_tokens=int(tok["prompt"]),
            total_completion_tokens=int(tok["completion"]),
            attempt_cost_events=attempt_events,
            public_val_log_predictive_delta=0.0,
            collateral_min_delta=0.0,
            rationale=(
                f"Executed 1-turn `LocalContainerRepair` (`repaired_ok={repaired_ok}`, "
                f"`guard={rep_out.get('persisted_guard_id')}`) -> `{selected_act}`."
            ),
        )
        graph.decision_ledger.append(rec_done)
        graph.action_return_history.setdefault("LocalContainerRepair", []).append(
            {
                "event_id": rec_done.decision_id,
                "task_id": str(task.task_id),
                "step_index": int(step_index),
                "metric_kind": "observed_public_repair_acceptance",
                "observed_public_acceptance": 1.0 if repaired_ok else 0.0,
                "gain_nats": 0.0,
                "cost_spent": NOMINAL_LOCAL_REPAIR_COST,
            }
        )
        graph.sync_authoritative_state_to_forest()
        return {
            "selected_action": selected_act,
            "final_python_code": final_code,
            "probe_before": probe_before,
            "probe_after": rep_out.get("probe_after") or probe_before,
            "recalled_guard_ids": rep_out.get("recalled_guard_ids", []),
            "persisted_guard_id": rep_out.get("persisted_guard_id"),
            "repair_metadata": {
                "invariance_guard_name": rep_out.get("invariance_guard_name"),
                "invariance_principle": rep_out.get("invariance_principle"),
                "defect_diagnosis": rep_out.get("defect_diagnosis"),
                "prompt_tokens": tok["prompt"],
                "completion_tokens": tok["completion"],
            },
            "decision_record": rec_done.to_dict(),
            "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        }
    finally:
        m6f.stop_container_safely(cid)
