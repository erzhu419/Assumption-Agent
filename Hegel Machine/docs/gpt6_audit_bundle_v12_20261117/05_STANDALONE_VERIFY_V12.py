#!/usr/bin/env python3
"""Zero-dependency standalone audit verifier for Hegel Machine v12 GPT-6 Handoff Bundle.

Usage inside GPT-6 sandbox (after unzipping Hegel_v12_GPT6_Audit_Handoff_20261117.zip):
    python3 05_STANDALONE_VERIFY_V12.py --output verify_v12_report.json

Uses ONLY Python standard library (`ast`, `copy`, `dataclasses`, `fractions`, `hashlib`, `json`,
`re`, `pathlib`) and makes zero network calls, zero model calls, and zero repository modifications.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _MinimalProgram:
    op: str
    value: float | None = None
    name: str | None = None
    lag: int = 0
    children: tuple = ()
    definition: Any = None

    def predict(self, x: dict[str, float]) -> float:
        if self.op == "const":
            return float(self.value or 0.0)
        if self.op in {"arg", "feature"}:
            return float(x[self.name])  # type: ignore[index]
        if self.op == "lt":
            return float(self.children[0].predict(x) < self.children[1].predict(x))
        raise ValueError(self.op)


def check_v12_source_excerpt_probes_ast(bundle_dir: Path) -> dict[str, Any]:
    """Extract verbatim AST functions from bundled `branch_reconciliation.py` and run GPT-6's v12 probes."""
    src_p = _locate_source(bundle_dir, "src/hegel_machine/cognition/branch_reconciliation.py")
    if src_p is None:
        return {"present": False, "all_v12_excerpt_probes_remediated": False}
    source = src_p.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_p))
    wanted = {
        "_is_theta_slot_arg",
        "_programs_algebraically_identical",
        "anti_unify_programs",
        "propagate_premise_retraction_invalidation",
    }
    fn_nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    if len(fn_nodes) < 4:
        return {
            "present": True,
            "found_functions": [n.name for n in fn_nodes],
            "all_v12_excerpt_probes_remediated": False,
        }
    mod = ast.Module(body=fn_nodes, type_ignores=[])
    ns: dict[str, Any] = {
        "Program": _MinimalProgram,
        "deepcopy": deepcopy,
        "Any": Any,
    }
    exec(compile(mod, str(src_p), "exec"), ns)

    alg_id = ns["_programs_algebraically_identical"]
    au_fn = ns["anti_unify_programs"]
    prop_fn = ns["propagate_premise_retraction_invalidation"]

    def predicate(t: float) -> _MinimalProgram:
        return _MinimalProgram(
            "lt",
            children=(_MinimalProgram("const", value=t), _MinimalProgram("arg", name="x")),
        )

    p0, p1 = predicate(0.0), predicate(5e-10)
    x_wit = {"x": 2.5e-10}
    eq_diff = alg_id(p0, p1)
    eq_same = alg_id(p0, p0)

    au = au_fn(predicate(0.22), predicate(0.46))
    third = au_fn(au[0], predicate(0.34)) if au is not None else None

    unknown = prop_fn([{"claim_id": "new_root", "status": "candidate"}])[0]
    cycle = prop_fn(
        [
            {"claim_id": "A", "depends_on_claim_ids": ["B"], "independent_empirical_support": False},
            {"claim_id": "B", "depends_on_claim_ids": ["A"], "independent_empirical_support": False},
        ]
    )

    remediated = (
        eq_diff is False
        and eq_same is True
        and au is not None
        and third is not None
        and [s["slot"] for s in third[1]] == ["theta_0"]
        and unknown["support_validity"] == "unverified_missing_support"
        and all(c["support_validity"] == "invalid_ungrounded_cycle" for c in cycle)
    )
    return {
        "present": True,
        "threshold_equivalence": {
            "0_vs_5e_minus_10_returns_identical": eq_diff,
            "identical_control": eq_same,
            "witness_x": x_wit["x"],
            "prediction_a": p0.predict(x_wit),
            "prediction_b": p1.predict(x_wit),
        },
        "incremental_anti_unification": {
            "two_thresholds_succeed": au is not None,
            "third_compatible_threshold_returns_none": third is None,
            "slot_names": [s["slot"] for s in third[1]] if third is not None else None,
        },
        "unsupported_root": unknown,
        "unsupported_cycle_control": cycle,
        "all_v12_excerpt_probes_remediated": bool(remediated),
    }


def check_ast_fingerprint_probe(bundle_dir: Path) -> dict[str, Any]:
    """Run GPT-6's 6-mutation AST probe against bundled `premise_commitment_index.py`."""
    src_p = _locate_source(bundle_dir, "src/hegel_machine/cognition/premise_commitment_index.py")
    if src_p is None:
        return {"present": False, "undetected_semantic_mutations": 6}
    source = src_p.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_p))
    fn_node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prior_state_fingerprint"),
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
        "structure": {"fits": [{"coefficients": [0.0, 1.0, -0.7], "fit_terms": [{"op": "feature", "name": "m0_1"}]}]},
        "research_program": {"selection_steps": [{"operation": "resolve_relation_intervals", "parameters": [{"radius": 0.9}]}]},
        "scope": {"bounds": {"x": [-1.0, 1.0]}},
        "support_witness_rows": [{"features": {"m0_1": 1.0}, "target": 1.0}],
        "counterexamples": [{"rows": [{"features": {"m0_1": -1.0}, "target": 2.0}]}],
        "_cached_commitment_group_key": "old_cached_key",
    }
    base_fp = fp_fn(base)

    mutations: dict[str, bool] = {}
    for name, mutator in [
        ("program", lambda d: d.__setitem__("program", {"op": "feature", "name": "m0_2"})),
        ("coefficient", lambda d: d["structure"]["fits"][0].__setitem__("coefficients", [0.0, 1.0, -0.25])),
        ("radius", lambda d: d["research_program"]["selection_steps"][0]["parameters"][0].__setitem__("radius", 1.35)),
        ("scope", lambda d: d["scope"]["bounds"].__setitem__("x", [-0.2, 0.2])),
        ("support_content", lambda d: d["support_witness_rows"][0].__setitem__("target", 9.5)),
        ("counterexample_content", lambda d: d["counterexamples"][0]["rows"][0].__setitem__("target", -4.0)),
    ]:
        mc = deepcopy(base)
        mutator(mc)
        mutations[name] = fp_fn(mc) != base_fp

    m_cache_only = deepcopy(base)
    m_cache_only["_cached_commitment_group_key"] = "different_cached_key"
    cache_ignored = fp_fn(m_cache_only) == base_fp
    undetected = sum(1 for v in mutations.values() if not v)
    return {
        "present": True,
        "undetected_semantic_mutations": undetected,
        "cached_metadata_ignored": cache_ignored,
        "mutations_detected": mutations,
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
            all(gu[i] == mu[2 * n + i] - mu[3 * n + i] + mu[4 * n + i] - mu[5 * n + i] for i in range(n)),
            all(mu[4 * n + i] + mu[5 * n + i] <= fuel[i] for i in range(n)),
            sum(mu[i] * hi + mu[n + i] * (-lo) + mu[2 * n + i] * uh + mu[3 * n + i] * (-ul) for i in range(n)) <= 0,
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
        "physical_legality": {"potential_verified": bool(dual_ok_on_box), "witness_verified": bool(reaches_goal_unbounded and in_box_all_steps)},
        "analysis_region_without_enclosure_proof": {"potential_verified": False, "witness_verified": bool(reaches_goal_unbounded)},
        "unbounded": {"potential_verified": False, "witness_verified": bool(reaches_goal_unbounded)},
    }
    return {
        "all_modes_free_of_contradiction": all(
            not (m["potential_verified"] and m["witness_verified"]) for m in modes.values()
        ),
        "modes": modes,
    }


def check_bundle_integrity_and_v12_receipts(bundle_dir: Path) -> dict[str, Any]:
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
                    v10_untouched["mismatches"].append({"path": rel_v10, "expected": item["sha256"], "actual": got})
                else:
                    v10_untouched["checked"] += 1

    v12_dir = "docs/evidence/v12_autonomous_reconciliation_and_utility_20261117"
    da12_p = _locate_receipt(bundle_dir, f"{v12_dir}/deliverable_a_v12_semantics_and_errata_20261117.json")
    db12_p = _locate_receipt(bundle_dir, f"{v12_dir}/deliverable_b_v12_autonomous_lifecycle_20261117.json")
    dc12_p = _locate_receipt(bundle_dir, f"{v12_dir}/deliverable_c_v12_knowledge_utility_20261117.json")

    v12_checks: dict[str, Any] = {
        "all_three_v12_receipts_present": bool(da12_p and db12_p and dc12_p),
        "violations": [],
    }
    if da12_p:
        da12 = json.loads(da12_p.read_text(encoding="utf-8"))
        if (da12.get("paid_api_budget") or {}).get("model_calls", 1) != 0:
            v12_checks["violations"].append("v12_deliverable_a_nonzero_model_calls")
        if not da12.get("all_deliverable_a_checks_passed"):
            v12_checks["violations"].append("v12_deliverable_a_checks_not_passed")
        # Cross-check historic v11 hashes recorded in Deliverable A errata
        v11_hashes = da12["a1_historic_v11_readonly_reclassification"]["audited_v11_files_readonly"]
        p_b11 = _locate_receipt(bundle_dir, "docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_b_branch_lifecycle_20261117.json")
        p_c11 = _locate_receipt(bundle_dir, "docs/evidence/v11_branch_lifecycle_and_km_20261117/deliverable_c_knowledge_vs_method_2x2_20261117.json")
        if p_b11 and _sha256_file(p_b11) != v11_hashes["deliverable_b_sha256"]:
            v12_checks["violations"].append("historic_v11_deliverable_b_modified")
        if p_c11 and _sha256_file(p_c11) != v11_hashes["deliverable_c_sha256"]:
            v12_checks["violations"].append("historic_v11_deliverable_c_modified")

    if db12_p:
        db12 = json.loads(db12_p.read_text(encoding="utf-8"))
        if (db12.get("paid_api_budget") or {}).get("model_calls", 1) != 0:
            v12_checks["violations"].append("v12_deliverable_b_nonzero_model_calls")
        hg = db12.get("harness_isolation_guarantee") or {}
        if any(
            hg.get(k, True)
            for k in (
                "routes_on_hidden_relation_condition",
                "manual_merge_pair_lists",
                "manual_scope_guard_overrides",
                "asserts_dropping_failed_episodes",
            )
        ):
            v12_checks["violations"].append("v12_deliverable_b_harness_isolation_violation")
        if len(db12.get("unseen_evaluation_chains") or ()) < 4:
            v12_checks["violations"].append("v12_deliverable_b_fewer_than_4_unseen_chains")
        if not isinstance(db12["unseen_evaluation_summary"].get("eval_total_false_internal_confirmations"), int):
            v12_checks["violations"].append("v12_deliverable_b_missing_false_confirmation_metric")
        for ch_b in db12.get("unseen_evaluation_chains") or ():
            for ep_b in [
                *ch_b.get("source_episodes", ()),
                ch_b["prospective_target_and_removal_checks"]["unseen_family_with_H_star"],
                ch_b["prospective_target_and_removal_checks"]["unseen_old_domain_return"],
            ]:
                if ep_b.get("strict_pass") and (
                    ep_b.get("false_internal_confirmation")
                    or float(ep_b.get("external_fresh48_mse", 1.0)) > 1e-7
                ):
                    v12_checks["violations"].append("v12_deliverable_b_dishonest_strict_pass_with_external_failure")

    if dc12_p:
        dc12 = json.loads(dc12_p.read_text(encoding="utf-8"))
        if (dc12.get("paid_api_budget") or {}).get("model_calls", 1) != 0:
            v12_checks["violations"].append("v12_deliverable_c_nonzero_model_calls")
        prot = dc12.get("equal_tool_and_api_protocol") or {}
        if prot.get("experiment_policy_fixed") != "developer_default_M0":
            v12_checks["violations"].append("v12_deliverable_c_not_fixed_M0")
        if not prot.get("pass0_parametric_family_calibration_enabled_on_all_arms"):
            v12_checks["violations"].append("v12_deliverable_c_pass0_disabled_on_baseline")
        if not prot.get("r_ordinary_allowed_on_demand_family_reconstruction"):
            v12_checks["violations"].append("v12_deliverable_c_r_ordinary_reconstruction_disabled")
        if len(dc12.get("unseen_evaluation_chains") or ()) < 8:
            v12_checks["violations"].append("v12_deliverable_c_fewer_than_8_unseen_chains")

    runtime_and_v12_files = [
        "src/hegel_machine/cognition/branch_reconciliation.py",
        "src/hegel_machine/cognition/competing_object_programs.py",
        "src/hegel_machine/cognition/discriminating_experiment_policy.py",
        "src/hegel_machine/cognition/equal_tool_attribution.py",
        "src/hegel_machine/cognition/premise_commitment_index.py",
        "src/hegel_machine/cognition/premise_dependency_audit.py",
        "src/hegel_machine/cognition/world_structure_commitments.py",
        "src/hegel_machine/cognition/unified_research.py",
        "scripts/run_v12_deliverable_a_errata_and_probes.py",
        "scripts/run_v12_autonomous_lifecycle_study.py",
        "scripts/run_v12_knowledge_utility_interventions.py",
    ]
    private_lab_reads = []
    fake_probe_labels = []
    harness_cond_routes = []
    api_key_leaks = []
    priv_pat = re.compile(r"\blab\._[a-zA-Z0-9_]+")
    key_pat = re.compile(r"sk-[A-Za-z0-9]{20,}")
    fake_probe_pat = re.compile(
        r'"(parametric_slot_calibration|active_threshold_or_interval_probe|premise_revised_probe)"'
    )
    scanned_count = 0
    for rel_m in runtime_and_v12_files:
        mp = _locate_source(bundle_dir, rel_m)
        if mp is not None:
            scanned_count += 1
            txt = mp.read_text(encoding="utf-8")
            for hit in priv_pat.findall(txt):
                private_lab_reads.append({"file": rel_m, "match": hit})
            for hit in key_pat.findall(txt):
                api_key_leaks.append({"file": rel_m, "match": hit[:6] + "..."})
            if rel_m.startswith("src/"):
                for hit in fake_probe_pat.findall(txt):
                    fake_probe_labels.append({"file": rel_m, "match": hit})
            if rel_m.startswith("scripts/run_v12_") and rel_m != "scripts/run_v12_deliverable_a_errata_and_probes.py":
                mod_ast = ast.parse(txt, filename=rel_m)
                for node in ast.walk(mod_ast):
                    if isinstance(node, (ast.If, ast.IfExp)):
                        test_src = ast.unparse(node.test)
                        if "relation_condition" in test_src:
                            harness_cond_routes.append({"file": rel_m, "lineno": getattr(node, "lineno", 0), "match": test_src})

    return {
        "bundle_sha256_integrity": sha_checks,
        "historic_v10_13_receipts_untouched": v10_untouched,
        "v12_receipt_invariants": v12_checks,
        "static_anti_cheat_scan": {
            "scanned_files": scanned_count,
            "private_lab_attribute_reads": private_lab_reads,
            "fake_probe_string_labels_in_runtime": fake_probe_labels,
            "harness_hidden_condition_routes": harness_cond_routes,
            "persisted_api_keys": api_key_leaks,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing unpacked GPT-6 v12 audit bundle",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    v12_probes = check_v12_source_excerpt_probes_ast(args.bundle_dir)
    fp_probe = check_ast_fingerprint_probe(args.bundle_dir)
    dom_check = check_certificate_domain_counterexample_exact_rational()
    bundle_audit = check_bundle_integrity_and_v12_receipts(args.bundle_dir)

    scan = bundle_audit["static_anti_cheat_scan"]
    all_ok = (
        v12_probes.get("all_v12_excerpt_probes_remediated", False)
        and fp_probe.get("present", False)
        and fp_probe.get("undetected_semantic_mutations") == 0
        and fp_probe.get("cached_metadata_ignored", False)
        and dom_check["all_modes_free_of_contradiction"]
        and not bundle_audit["bundle_sha256_integrity"]["mismatches"]
        and not bundle_audit["historic_v10_13_receipts_untouched"]["mismatches"]
        and bundle_audit["v12_receipt_invariants"]["all_three_v12_receipts_present"]
        and not bundle_audit["v12_receipt_invariants"]["violations"]
        and not scan["private_lab_attribute_reads"]
        and not scan["fake_probe_string_labels_in_runtime"]
        and not scan["harness_hidden_condition_routes"]
        and not scan["persisted_api_keys"]
    )
    report = {
        "verifier": "05_STANDALONE_VERIFY_V12.py",
        "v12_source_excerpt_probes_ast_check": v12_probes,
        "v11_ast_fingerprint_probe_check": fp_probe,
        "v10_2_exact_rational_domain_check": dom_check,
        "bundle_and_v12_receipt_audit": bundle_audit,
        "all_checks_passed": bool(all_ok),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
