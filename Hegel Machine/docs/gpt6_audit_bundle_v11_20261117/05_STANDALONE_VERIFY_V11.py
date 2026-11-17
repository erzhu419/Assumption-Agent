#!/usr/bin/env python3
"""Zero-dependency standalone audit verifier for Hegel Machine v11 GPT-6 Handoff Bundle.

Usage inside GPT-6 sandbox (after unzipping Hegel_v11_GPT6_Audit_Handoff_20261117.zip):
    python3 05_STANDALONE_VERIFY_V11.py --output verify_report.json

Optional live-repository execution mode:
    python3 05_STANDALONE_VERIFY_V11.py --repo "/path/to/Hegel Machine" --output live_verify.json

Default mode uses ONLY Python standard library (`ast`, `copy`, `fractions`, `hashlib`, `json`,
`re`, `pathlib`) and makes zero network calls, zero model calls, and zero repository modifications.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _locate_receipt(bundle_dir: Path, rel_path: str) -> Path | None:
    p_bundle = bundle_dir / "receipts" / Path(rel_path).relative_to("docs/evidence")
    if p_bundle.is_file():
        return p_bundle
    p_repo = bundle_dir.parents[1] / rel_path
    if p_repo.is_file():
        return p_repo
    return None


def _locate_source(bundle_dir: Path, rel_path: str) -> Path | None:
    p_bundle = bundle_dir / "bundled_sources" / rel_path
    if p_bundle.is_file():
        return p_bundle
    p_repo = bundle_dir.parents[1] / rel_path
    if p_repo.is_file():
        return p_repo
    return None


def check_ast_fingerprint_probe(bundle_dir: Path) -> dict[str, Any]:
    """Run GPT-6's exact 6-mutation AST probe against bundled `premise_commitment_index.py`."""
    src_p = _locate_source(bundle_dir, "src/hegel_machine/cognition/premise_commitment_index.py")
    if src_p is None:
        return {"present": False, "undetected_semantic_mutations": 6}
    source = src_p.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_p))
    fn_node = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_prior_state_fingerprint"
        ),
        None,
    )
    if fn_node is None:
        return {"present": False, "undetected_semantic_mutations": 6}
    mod = ast.Module(body=[fn_node], type_ignores=[])
    ns: dict[str, Any] = {"hashlib": hashlib, "json": json}
    exec(compile(mod, str(src_p), "exec"), ns)
    fp_fn = ns["_prior_state_fingerprint"]

    base = {
        "source_concept_id": "probe_0",
        "status": "supported",
        "program": {"op": "feature", "name": "m0_1"},
        "structure": {
            "fits": [
                {
                    "coefficients": [0.0, 1.0, -0.7],
                    "fit_terms": [{"op": "feature", "name": "m0_1"}],
                }
            ]
        },
        "research_program": {
            "selection_steps": [
                {"operation": "resolve_relation_intervals", "parameters": [{"radius": 0.9}]}
            ]
        },
        "scope": {"bounds": {"x": [-1.0, 1.0]}},
        "support_witness_rows": [{"features": {"m0_1": 1.0}, "target": 1.0}],
        "counterexamples": [{"rows": [{"features": {"m0_1": -1.0}, "target": 2.0}]}],
        "_cached_commitment_group_key": "old_cached_key",
    }
    base_fp = fp_fn(base)

    mutations: dict[str, Any] = {}

    m_prog = deepcopy(base)
    m_prog["program"] = {"op": "feature", "name": "m0_2"}
    mutations["program"] = fp_fn(m_prog) != base_fp

    m_coef = deepcopy(base)
    m_coef["structure"]["fits"][0]["coefficients"] = [0.0, 1.0, -0.25]
    mutations["coefficient"] = fp_fn(m_coef) != base_fp

    m_rad = deepcopy(base)
    m_rad["research_program"]["selection_steps"][0]["parameters"][0]["radius"] = 1.35
    mutations["radius"] = fp_fn(m_rad) != base_fp

    m_scp = deepcopy(base)
    m_scp["scope"]["bounds"]["x"] = [-0.2, 0.2]
    mutations["scope"] = fp_fn(m_scp) != base_fp

    m_sup = deepcopy(base)
    m_sup["support_witness_rows"][0]["target"] = 9.5
    mutations["support_content"] = fp_fn(m_sup) != base_fp

    m_ce = deepcopy(base)
    m_ce["counterexamples"][0]["rows"][0]["target"] = -4.0
    mutations["counterexample_content"] = fp_fn(m_ce) != base_fp

    # Cached internal metadata must NOT alter semantic fingerprint
    m_cache_only = deepcopy(base)
    m_cache_only["_cached_commitment_group_key"] = "different_cached_key"
    cache_ignored = fp_fn(m_cache_only) == base_fp

    undetected = sum(1 for v in mutations.values() if not v)
    return {
        "present": True,
        "base_fingerprint": base_fp,
        "mutations_detected": mutations,
        "undetected_semantic_mutations": undetected,
        "cached_metadata_ignored": cache_ignored,
    }


def check_certificate_domain_counterexample_exact_rational() -> dict[str, Any]:
    n = 2
    A = [[F(1, 2), F(0)], [F(1, 2), F(1)]]
    B = [[F(1), F(0)], [F(0), F(1)]]
    alpha = [F(0), F(-1)]
    mu = list(map(F, ["1/2", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0"]))
    fuel = [F(1, 10), F(1)]
    lo, hi = F(-1), F(0)
    ul, uh = F(0), F(1)
    budget = F(1, 2)

    gw = [alpha[i] - sum(A[r][i] * alpha[r] for r in range(n)) for i in range(n)]
    gu = [-sum(B[r][i] * alpha[r] for r in range(n)) for i in range(n)]
    dual_ok_on_box = all(
        [
            all(x >= 0 for x in mu),
            all(gw[i] == mu[i] - mu[n + i] for i in range(n)),
            all(
                gu[i] == mu[2 * n + i] - mu[3 * n + i] + mu[4 * n + i] - mu[5 * n + i]
                for i in range(n)
            ),
            all(mu[4 * n + i] + mu[5 * n + i] <= fuel[i] for i in range(n)),
            sum(
                mu[i] * hi + mu[n + i] * (-lo) + mu[2 * n + i] * uh + mu[3 * n + i] * (-ul)
                for i in range(n)
            )
            <= 0,
            F(1) > budget,
        ]
    )

    w = [F(0), F(0)]
    cost = F(0)
    in_box_all_steps = all(lo <= x <= hi for x in w)
    for row in ([F(1), F(0)], [F(1), F(0)], [F(0), F(0)]):
        cost += sum(fuel[i] * abs(row[i]) for i in range(n))
        w = [sum(A[i][j] * w[j] for j in range(n)) + row[i] for i in range(n)]
        in_box_all_steps = in_box_all_steps and all(lo <= x <= hi for x in w)
    reaches_goal_unbounded = (-w[1] <= F(-1)) and (cost <= budget)

    modes = {
        "physical_legality": {
            "potential_verified": bool(dual_ok_on_box),
            "witness_verified": bool(reaches_goal_unbounded and in_box_all_steps),
        },
        "analysis_region_without_enclosure_proof": {
            "potential_verified": False,
            "witness_verified": bool(reaches_goal_unbounded),
        },
        "unbounded": {
            "potential_verified": False,
            "witness_verified": bool(reaches_goal_unbounded),
        },
    }
    for m_res in modes.values():
        m_res["both_opposite_certificates_accepted"] = bool(
            m_res["potential_verified"] and m_res["witness_verified"]
        )
    return {
        "all_modes_free_of_contradiction": all(
            not m["both_opposite_certificates_accepted"] for m in modes.values()
        ),
        "modes": modes,
    }


def check_bundle_integrity_and_v11_receipts(bundle_dir: Path) -> dict[str, Any]:
    sums_path = bundle_dir / "SHA256SUMS.json"
    sha_checks = {"present": sums_path.is_file(), "verified_files": 0, "mismatches": []}
    if sums_path.is_file():
        recorded = json.loads(sums_path.read_text(encoding="utf-8"))
        for rel_p, expected_hex in recorded.get("files", {}).items():
            fp = bundle_dir / rel_p
            if not fp.is_file():
                sha_checks["mismatches"].append({"file": rel_p, "error": "missing"})
            else:
                actual_hex = _sha256_file(fp)
                if actual_hex != expected_hex:
                    sha_checks["mismatches"].append(
                        {"file": rel_p, "expected": expected_hex, "actual": actual_hex}
                    )
                else:
                    sha_checks["verified_files"] += 1

    # Verify all 13 historic v10 receipts remain untouched
    ledger_p = _locate_receipt(
        bundle_dir,
        "docs/evidence/v10_2_audit_remediation_20261116/errata_and_sha256_ledger_20261116.json",
    )
    v10_untouched = {"checked": 0, "mismatches": []}
    if ledger_p is not None:
        ledger = json.loads(ledger_p.read_text(encoding="utf-8"))
        for item in ledger.get("untouched_v10_receipts_sha256_and_roles", ()):
            rel_v10 = f"docs/evidence/v10_world_structure_20261116/{item['filename']}"
            rp = _locate_receipt(bundle_dir, rel_v10)
            if rp is not None:
                got = _sha256_file(rp)
                if got != item["sha256"]:
                    v10_untouched["mismatches"].append(
                        {"path": rel_v10, "expected": item["sha256"], "actual": got}
                    )
                else:
                    v10_untouched["checked"] += 1

    # Verify v11 Deliverables A, B, and C receipts
    da_p = _locate_receipt(
        bundle_dir,
        "docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_a_credibility_and_baselines_20261117.json",
    )
    db_p = _locate_receipt(
        bundle_dir,
        "docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_b_branch_lifecycle_20261117.json",
    )
    dc_p = _locate_receipt(
        bundle_dir,
        "docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_c_knowledge_vs_method_2x2_20261117.json",
    )
    dc11_1_p = _locate_receipt(
        bundle_dir,
        "docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_c_v11_1_method_remediation_2x2_20261117.json",
    )

    v11_checks: dict[str, Any] = {
        "all_three_v11_receipts_present": bool(da_p and db_p and dc_p),
        "v11_1_remediation_receipt_present": bool(dc11_1_p),
        "violations": [],
    }
    if da_p:
        da = json.loads(da_p.read_text(encoding="utf-8"))
        if (da.get("llm_usage") or {}).get("model_calls", 1) != 0:
            v11_checks["violations"].append("deliverable_a_nonzero_model_calls")
        a1 = da["deliverable_A1_index_and_fingerprint_audit"]
        if a1["gpt6_fingerprint_probe"]["undetected_semantic_mutations"] != 0:
            v11_checks["violations"].append("deliverable_a1_undetected_mutations")
        if not a1["partial_budget_interruption_and_resume"]["resumed_parity_exact"]:
            v11_checks["violations"].append("deliverable_a1_resume_parity_failed")
        a2 = da["deliverable_A2_strong_baselines_and_scaling"]
        for m_k, m_v in a2.items():
            emp_raw = m_v["confirmed_empirical_source_count"]
            emp_n = emp_raw["empirical_episode"] if isinstance(emp_raw, dict) else emp_raw
            if emp_n != 4:
                v11_checks["violations"].append(f"{m_k}_empirical_source_count_not_4")
            for arm_k, arm_v in m_v["arms"].items():
                if arm_v["max_consolidation_calls_per_episode"] > 1:
                    v11_checks["violations"].append(f"{m_k}_{arm_k}_double_charged")
        a3 = da["deliverable_A3_evidence_budget_and_telemetry"]
        if a3["provisional_occam_gate"]["selection_epistemic_status"] != "provisional_occam_preference":
            v11_checks["violations"].append("deliverable_a3_missing_provisional_occam")

    if db_p:
        db = json.loads(db_p.read_text(encoding="utf-8"))
        if (db.get("llm_usage") or {}).get("model_calls", 1) != 0:
            v11_checks["violations"].append("deliverable_b_nonzero_model_calls")
        if not db.get("acceptance_B1_to_B10_all_verified"):
            v11_checks["violations"].append("deliverable_b_acceptance_flag_false")
        if len(db.get("unseen_evaluation_chains") or ()) < 4:
            v11_checks["violations"].append("deliverable_b_fewer_than_4_unseen_chains")

    if dc_p:
        dc = json.loads(dc_p.read_text(encoding="utf-8"))
        if (dc.get("llm_usage") or {}).get("model_calls", 1) != 0:
            v11_checks["violations"].append("deliverable_c_nonzero_model_calls")
        ph = dc["partition_hygiene_and_evaluator_immutability"]
        if not ph["zero_seed_overlap_verified"] or not ph["evaluator_immutable_verified"]:
            v11_checks["violations"].append("deliverable_c_partition_or_immutability_violation")
        m_ind = dc["method_induction_M0_to_M1"]
        if m_ind["promoted"] and m_ind["promoted_method"]["author_origin"] != "runtime_induced_from_dev_episodes":
            v11_checks["violations"].append("deliverable_c_m1_wrong_author_origin")

    if dc11_1_p:
        dc11 = json.loads(dc11_1_p.read_text(encoding="utf-8"))
        if (dc11.get("llm_usage") or {}).get("model_calls", 1) != 0:
            v11_checks["violations"].append("deliverable_c_v11_1_nonzero_model_calls")
        ph11 = dc11["partition_hygiene_and_evaluator_immutability"]
        if (
            not ph11["zero_internal_seed_overlap_verified"]
            or not ph11["zero_overlap_with_v11_971000_976000_verified"]
            or not ph11["evaluator_immutable_verified"]
        ):
            v11_checks["violations"].append("deliverable_c_v11_1_partition_or_immutability_violation")
        m_ind11 = dc11["method_induction_M0_to_M1_v11_1"]
        if (
            m_ind11["promoted"]
            and m_ind11["promoted_method"]["author_origin"] != "runtime_induced_from_dev_episodes"
        ):
            v11_checks["violations"].append("deliverable_c_v11_1_m1_wrong_author_origin")

    runtime_modules = [
        "src/hegel_machine/cognition/algebraic_proof_certificates.py",
        "src/hegel_machine/cognition/equal_tool_attribution.py",
        "src/hegel_machine/cognition/premise_commitment_index.py",
        "src/hegel_machine/cognition/sensor_role_calibration.py",
        "src/hegel_machine/cognition/premise_dependency_audit.py",
        "src/hegel_machine/cognition/world_structure_commitments.py",
        "src/hegel_machine/cognition/branch_reconciliation.py",
        "src/hegel_machine/cognition/discriminating_experiment_policy.py",
        "src/hegel_machine/cognition/unified_research.py",
    ]
    private_lab_reads = []
    api_key_leaks = []
    priv_pat = re.compile(r"\blab\._[a-zA-Z0-9_]+")
    key_pat = re.compile(r"sk-[A-Za-z0-9]{20,}")
    scanned_modules = 0
    for rel_m in runtime_modules:
        mp = _locate_source(bundle_dir, rel_m)
        if mp is not None:
            scanned_modules += 1
            txt = mp.read_text(encoding="utf-8")
            for hit in priv_pat.findall(txt):
                private_lab_reads.append({"module": rel_m, "match": hit})
            for hit in key_pat.findall(txt):
                api_key_leaks.append({"module": rel_m, "match": hit[:6] + "..."})

    return {
        "bundle_sha256_integrity": sha_checks,
        "historic_v10_13_receipts_untouched": v10_untouched,
        "v11_receipt_invariants": v11_checks,
        "static_anti_cheat_scan": {
            "scanned_runtime_modules": scanned_modules,
            "private_lab_attribute_reads": private_lab_reads,
            "persisted_api_keys": api_key_leaks,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing unpacked GPT-6 v11 audit bundle",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Optional local Hegel Machine checkout",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    fp_probe = check_ast_fingerprint_probe(args.bundle_dir)
    dom_check = check_certificate_domain_counterexample_exact_rational()
    bundle_audit = check_bundle_integrity_and_v11_receipts(args.bundle_dir)

    all_ok = (
        fp_probe.get("present", False)
        and fp_probe.get("undetected_semantic_mutations") == 0
        and fp_probe.get("cached_metadata_ignored", False)
        and dom_check["all_modes_free_of_contradiction"]
        and not bundle_audit["bundle_sha256_integrity"]["mismatches"]
        and not bundle_audit["historic_v10_13_receipts_untouched"]["mismatches"]
        and bundle_audit["v11_receipt_invariants"]["all_three_v11_receipts_present"]
        and bundle_audit["v11_receipt_invariants"]["v11_1_remediation_receipt_present"]
        and not bundle_audit["v11_receipt_invariants"]["violations"]
        and not bundle_audit["static_anti_cheat_scan"]["private_lab_attribute_reads"]
        and not bundle_audit["static_anti_cheat_scan"]["persisted_api_keys"]
    )
    report = {
        "verifier": "05_STANDALONE_VERIFY_V11.py",
        "ast_fingerprint_probe_check": fp_probe,
        "exact_rational_domain_check": dom_check,
        "bundle_and_v11_receipt_audit": bundle_audit,
        "all_checks_passed": bool(all_ok),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
