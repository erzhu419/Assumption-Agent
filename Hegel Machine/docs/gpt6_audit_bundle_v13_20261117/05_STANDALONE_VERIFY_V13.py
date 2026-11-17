#!/usr/bin/env python3
"""Zero-dependency standalone audit verifier for Hegel Machine v13 GPT-6 Handoff Bundle.

Usage inside GPT-6 sandbox (after unzipping Hegel_v13_GPT6_Audit_Handoff_20261117.zip):
    python3 05_STANDALONE_VERIFY_V13.py --output verify_v13_report.json

Uses ONLY Python standard library (`ast`, `copy`, `dataclasses`, `fractions`, `hashlib`, `json`,
`math`, `re`, `types`, `pathlib`) and makes zero network calls, zero model calls, and zero edits.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
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
class _MinimalObservation:
    features: dict[str, float]
    target: float
    history: tuple = ()


@dataclass(frozen=True)
class _MinimalProgram:
    op: str
    value: float | None = None
    name: str | None = None
    lag: int = 0
    children: tuple = ()
    definition: Any = None

    def predict(self, row: Any) -> float:
        d = row.features if hasattr(row, "features") else row
        if self.op == "const":
            return float(self.value or 0.0)
        if self.op in {"arg", "feature"}:
            return float(d[self.name])  # type: ignore[index]
        c = [ch.predict(row) for ch in self.children]
        if self.op == "lt":
            return float(c[0] < c[1])
        if self.op == "and":
            return float(all(x >= 0.5 for x in c))
        if self.op == "or":
            return float(any(x >= 0.5 for x in c))
        if self.op == "not":
            return float(c[0] < 0.5)
        if self.op == "add":
            return float(sum(c))
        if self.op == "sub":
            return float(c[0] - c[1])
        if self.op == "mul":
            return float(math.prod(c))
        raise ValueError(self.op)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "value": self.value,
            "name": self.name,
            "lag": self.lag,
            "children": [ch.to_dict() for ch in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_MinimalProgram":
        return cls(
            d["op"],
            d.get("value"),
            d.get("name"),
            d.get("lag", 0),
            tuple(cls.from_dict(c) for c in d.get("children", ())),
        )


@dataclass
class _MinimalResearchQuestion:
    question: str
    evidence: dict[str, Any]
    operations: tuple[str, ...]


def _min_loss(prog: _MinimalProgram, rows: list[_MinimalObservation], *args: Any) -> float:
    return sum((prog.predict(r) - r.target) ** 2 for r in rows) / max(1, len(rows))


def check_v13_gpt6_review_probes_ast(bundle_dir: Path) -> dict[str, Any]:
    """Re-run GPT-6's exact `markdown/hegel_v13/review_probes.py` AST excerpt controls on v13 code."""
    b_p = _locate_source(bundle_dir, "src/hegel_machine/cognition/branch_reconciliation.py")
    u_p = _locate_source(bundle_dir, "src/hegel_machine/cognition/unified_research.py")
    if b_p is None or u_p is None:
        return {"present": False, "all_v13_review_probes_remediated": False}

    b_tree = ast.parse(b_p.read_text(encoding="utf-8"), filename=str(b_p))
    ns: dict[str, Any] = {
        "Program": _MinimalProgram,
        "Observation": _MinimalObservation,
        "ResearchQuestion": _MinimalResearchQuestion,
        "deepcopy": deepcopy,
        "isfinite": math.isfinite,
        "hashlib": hashlib,
        "json": json,
        "loss": _min_loss,
        "_as_schema": lambda x: x,
        "Any": Any,
    }
    varnames = {"FORBIDDEN_SCOPE_TOKENS", "PUBLIC_OBSERVABLE_GUARD_MANIFEST", "OBSERVABLE_REGIME_FEATURES"}
    for n in b_tree.body:
        names = []
        if isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names = [n.target.id]
        if varnames.intersection(names):
            mod_v = ast.fix_missing_locations(
                ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), n], type_ignores=[])
            )
            exec(compile(mod_v, str(b_p), "exec"), ns)

    wanted_fns = {
        "_program_semantic_version",
        "extract_guard_ast_dependencies",
        "validate_observable_scope_guard",
        "compute_observable_regime_features",
        "evaluate_scope_guard",
        "shrink_claim_to_observable_scope",
        "adjudicate_branches_on_paid_evidence",
        "reconcile_competing_branches",
    }
    fn_nodes = [n for n in b_tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted_fns]
    if len(fn_nodes) < len(wanted_fns):
        return {"present": True, "found": [n.name for n in fn_nodes], "all_v13_review_probes_remediated": False}
    mod_f = ast.fix_missing_locations(
        ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *fn_nodes], type_ignores=[])
    )
    exec(compile(mod_f, str(b_p), "exec"), ns)

    # 1. N3: Empty pilot -> `None` (not `0.0`) and `evaluate_scope_guard` -> `'unknown'` (not `True`)!
    g = _MinimalProgram("lt", children=(_MinimalProgram("feature", name="obs_unconditional_block_rate"), _MinimalProgram("const", value=0.1)))
    empty = ns["compute_observable_regime_features"]([], None)
    empty_eval = ns["evaluate_scope_guard"](g, empty)

    # 2. N1: `_exec_reconcile_probe` when `acquire` returns `None` -> `status == 'unavailable'`, zero fake paid probes!
    u_tree = ast.parse(u_p.read_text(encoding="utf-8"), filename=str(u_p))
    cb_fn = next(n for n in ast.walk(u_tree) if isinstance(n, ast.FunctionDef) and n.name == "_exec_reconcile_probe")
    oldrows = [_MinimalObservation({"x": 0.0}, 0.0)]
    oldtrans = [{"old_record": True}]
    cbns: dict[str, Any] = {
        "reconcile_channels": ("fine",),
        "acquire": lambda *a, **k: None,
        "ep_obs_rec": oldrows,
        "ep_trans_rec": oldtrans,
        "batches": [],
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cb_fn], type_ignores=[])), str(u_p), "exec"), cbns)
    requested = SimpleNamespace(probe_id="new_probe_not_executed", sequence=({"x": 1.0},), reset=True)
    cb_ret = cbns["_exec_reconcile_probe"](requested)

    # 3. N2: Both programs refuted -> `adjudicate` & `reconcile` return `pending_questions` and restrict universal status!
    old_c = {"claim_id": "old", "status": "universal", "independent_empirical_support": True}
    new_c = {"claim_id": "new", "status": "universal", "independent_empirical_support": True}
    a_al = {"program": _MinimalProgram("const", value=1.0)}
    b_al = {"program": _MinimalProgram("const", value=2.0)}
    direct = ns["adjudicate_branches_on_paid_evidence"](new_c, old_c, a_al, b_al, oldrows, oldtrans, None)

    ns["align_semantics_roles_units_and_time"] = lambda *args, **kw: {
        "aligned_new": {"aligned": True, **a_al},
        "compatible_candidates": [{"raw_claim": old_c, "aligned": b_al}],
        "incompatible_semantics": [],
    }
    ns["assess_scope_overlap"] = lambda *args, **kw: {"certified_disjoint": False, "unknown": False}
    ns["find_feasible_prediction_difference"] = lambda *args, **kw: {
        "proved_absent": False,
        "witness_available": True,
        "probe": requested,
        "max_abs_disagreement": 1.0,
    }
    rec_denied = ns["reconcile_competing_branches"](new_c, [old_c], None, None, execute_probe_fn=cbns["_exec_reconcile_probe"])
    rec_exec = ns["reconcile_competing_branches"](
        new_c,
        [old_c],
        None,
        None,
        execute_probe_fn=lambda p, **kw: {
            "status": "executed_new",
            "requested_probe_id": p.probe_id,
            "actual_probe_id": p.probe_id,
            "rows": oldrows,
            "transitions": oldtrans,
        },
    )

    remediated = (
        empty["obs_unconditional_block_rate"] is None
        and empty_eval == "unknown"
        and isinstance(cb_ret, dict)
        and cb_ret.get("status") == "unavailable"
        and rec_denied["paid_probes_executed"] == []
        and len(direct.get("pending_questions", ())) == 1
        and len(rec_exec.get("pending_questions", ())) == 1
        and "research_question" in rec_exec
        and rec_exec["updated_old_claims"][0]["status"] != "universal"
        and rec_exec["updated_new_claim"]["status"] != "universal"
    )
    return {
        "present": True,
        "empty_pilot_guard": {
            "computed_empty_block_rate": empty["obs_unconditional_block_rate"],
            "ordinary_evaluate_scope_guard": empty_eval,
        },
        "denied_probe_fallback": {
            "callback_status": cb_ret.get("status") if isinstance(cb_ret, dict) else "untyped",
            "recorded_paid_probes_when_denied": rec_denied["paid_probes_executed"],
        },
        "both_refuted_subquestion_return": {
            "reconciler_pending_questions_count": len(rec_exec.get("pending_questions", ())),
            "reconciler_has_research_question": "research_question" in rec_exec,
            "old_status_after": rec_exec["updated_old_claims"][0]["status"],
            "new_status_after": rec_exec["updated_new_claim"]["status"],
        },
        "all_v13_review_probes_remediated": bool(remediated),
    }


def check_v12_and_v11_probes_ast(bundle_dir: Path) -> dict[str, Any]:
    src_b = _locate_source(bundle_dir, "src/hegel_machine/cognition/branch_reconciliation.py")
    src_p = _locate_source(bundle_dir, "src/hegel_machine/cognition/premise_commitment_index.py")
    if src_b is None or src_p is None:
        return {"present": False, "all_prior_probes_ok": False}

    tree_b = ast.parse(src_b.read_text(encoding="utf-8"), filename=str(src_b))
    wanted_b = {"_is_theta_slot_arg", "_programs_algebraically_identical", "anti_unify_programs", "propagate_premise_retraction_invalidation"}
    fn_b = [n for n in tree_b.body if isinstance(n, ast.FunctionDef) and n.name in wanted_b]
    ns_b: dict[str, Any] = {"Program": _MinimalProgram, "deepcopy": deepcopy, "Any": Any}
    exec(compile(ast.Module(body=fn_b, type_ignores=[]), str(src_b), "exec"), ns_b)

    def pred(t: float) -> _MinimalProgram:
        return _MinimalProgram("lt", children=(_MinimalProgram("const", value=t), _MinimalProgram("arg", name="x")))

    eq_diff = ns_b["_programs_algebraically_identical"](pred(0.0), pred(5e-10))
    au = ns_b["anti_unify_programs"](pred(0.22), pred(0.46))
    third = ns_b["anti_unify_programs"](au[0], pred(0.34)) if au is not None else None
    unk_root = ns_b["propagate_premise_retraction_invalidation"]([{"claim_id": "new_root", "status": "candidate"}])[0]
    multi_fam = ns_b["propagate_premise_retraction_invalidation"](
        [
            {"claim_id": "b1", "independent_empirical_support": True},
            {"claim_id": "b2", "independent_empirical_support": True},
            {"claim_id": "b3", "independent_empirical_support": True},
            {"claim_id": "fam", "independent_empirical_support": False, "depends_on_claim_ids": ["b1", "b2", "b3"], "min_grounded_dependencies": 2},
        ],
        retracted_premise_or_claim_ids={"b1"},
    )
    fam_valid = next(c for c in multi_fam if c["claim_id"] == "fam")["support_validity"] == "valid_grounded"

    tree_p = ast.parse(src_p.read_text(encoding="utf-8"), filename=str(src_p))
    fn_fp = next(n for n in tree_p.body if isinstance(n, ast.FunctionDef) and n.name == "_prior_state_fingerprint")
    ns_p: dict[str, Any] = {"hashlib": hashlib, "json": json}
    exec(compile(ast.Module(body=[fn_fp], type_ignores=[]), str(src_p), "exec"), ns_p)
    fp_fn = ns_p["_prior_state_fingerprint"]
    base = {
        "source_concept_id": "probe_0",
        "status": "supported",
        "program": {"op": "feature", "name": "m0_1"},
        "structure": {"fits": [{"coefficients": [0.0, 1.0, -0.7], "fit_terms": [{"op": "feature", "name": "m0_1"}]}]},
        "research_program": {"selection_steps": [{"operation": "resolve_relation_intervals", "parameters": [{"radius": 0.9}]}]},
        "scope": {"bounds": {"x": [-1.0, 1.0]}},
        "support_witness_rows": [{"features": {"m0_1": 1.0}, "target": 1.0}],
        "counterexamples": [{"rows": [{"features": {"m0_1": -1.0}, "target": 2.0}]}],
        "_cached_commitment_group_key": "old",
    }
    base_fp = fp_fn(base)
    undetected = 0
    for mut in (
        lambda d: d.__setitem__("program", {"op": "feature", "name": "m0_2"}),
        lambda d: d["structure"]["fits"][0].__setitem__("coefficients", [0.0, 1.0, -0.25]),
        lambda d: d["research_program"]["selection_steps"][0]["parameters"][0].__setitem__("radius", 1.35),
        lambda d: d["scope"]["bounds"].__setitem__("x", [-0.2, 0.2]),
        lambda d: d["support_witness_rows"][0].__setitem__("target", 9.5),
        lambda d: d["counterexamples"][0]["rows"][0].__setitem__("target", -4.0),
    ):
        mc = deepcopy(base)
        mut(mc)
        if fp_fn(mc) == base_fp:
            undetected += 1

    ok = (
        eq_diff is False
        and third is not None
        and [s["slot"] for s in third[1]] == ["theta_0"]
        and unk_root["support_validity"] == "unverified_missing_support"
        and fam_valid is True
        and undetected == 0
    )
    return {
        "present": True,
        "v12_threshold_exact_ok": eq_diff is False,
        "v12_3branch_incremental_ok": third is not None,
        "v12_unverified_root_ok": unk_root["support_validity"] == "unverified_missing_support",
        "v13_multisource_family_survival_ok": fam_valid,
        "v11_undetected_fingerprint_mutations": undetected,
        "all_prior_probes_ok": bool(ok),
    }


def check_certificate_domain_counterexample_exact_rational() -> dict[str, Any]:
    n = 2
    A = [[F(1, 2), F(0)], [F(1, 2), F(1)]]
    alpha = [F(0), F(-1)]
    mu = list(map(F, ["1/2", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0"]))
    fuel = [F(1, 10), F(1)]
    lo, hi, ul, uh, budget = F(-1), F(0), F(0), F(1), F(1, 2)
    gw = [alpha[i] - sum(A[r][i] * alpha[r] for r in range(n)) for i in range(n)]
    gu = [-alpha[i] for i in range(n)]
    dual_ok = (
        all(x >= 0 for x in mu)
        and all(gw[i] == mu[i] - mu[n + i] for i in range(n))
        and all(gu[i] == mu[2 * n + i] - mu[3 * n + i] + mu[4 * n + i] - mu[5 * n + i] for i in range(n))
        and all(mu[4 * n + i] + mu[5 * n + i] <= fuel[i] for i in range(n))
        and sum(mu[i] * hi + mu[n + i] * (-lo) + mu[2 * n + i] * uh + mu[3 * n + i] * (-ul) for i in range(n)) <= 0
    )
    w = [F(0), F(0)]
    in_box = True
    for row in ([F(1), F(0)], [F(1), F(0)], [F(0), F(0)]):
        w = [sum(A[i][j] * w[j] for j in range(n)) + row[i] for i in range(n)]
        in_box = in_box and all(lo <= x <= hi for x in w)
    return {"all_modes_free_of_contradiction": not (dual_ok and in_box)}


def check_bundle_integrity_and_v13_receipts(bundle_dir: Path) -> dict[str, Any]:
    sums_path = bundle_dir / "SHA256SUMS.json"
    sha_checks: dict[str, Any] = {"present": sums_path.is_file(), "verified_files": 0, "mismatches": []}
    if sums_path.is_file():
        recorded = json.loads(sums_path.read_text(encoding="utf-8"))
        for rel_p, expected_hex in recorded.get("files", {}).items():
            fp = bundle_dir / rel_p
            if not fp.is_file():
                sha_checks["mismatches"].append({"file": rel_p, "error": "missing"})
            elif _sha256_file(fp) != expected_hex:
                sha_checks["mismatches"].append({"file": rel_p, "error": "sha256_mismatch"})
            else:
                sha_checks["verified_files"] += 1

    ledger_p = _locate_receipt(bundle_dir, "docs/evidence/v10_2_audit_remediation_20261116/errata_and_sha256_ledger_20261116.json")
    v10_untouched: dict[str, Any] = {"checked": 0, "mismatches": []}
    if ledger_p is not None:
        ledger = json.loads(ledger_p.read_text(encoding="utf-8"))
        for item in ledger.get("untouched_v10_receipts_sha256_and_roles", ()):
            rp = _locate_receipt(bundle_dir, f"docs/evidence/v10_world_structure_20261116/{item['filename']}")
            if rp is not None:
                if _sha256_file(rp) != item["sha256"]:
                    v10_untouched["mismatches"].append(item["filename"])
                else:
                    v10_untouched["checked"] += 1

    v13_dir = "docs/evidence/v13_evidence_recursion_and_routing_20261117"
    pa_p = _locate_receipt(bundle_dir, f"{v13_dir}/pr_a_evidence_and_recursion_controls_20261117.json")
    pb_p = _locate_receipt(bundle_dir, f"{v13_dir}/pr_b_routing_vs_applicability_and_cache_20261117.json")
    pc_p = _locate_receipt(bundle_dir, f"{v13_dir}/pr_c_tightened_source_paired_study_20261117.json")

    v13_checks: dict[str, Any] = {"all_three_v13_receipts_present": bool(pa_p and pb_p and pc_p), "violations": []}
    for nm, rp in (("pr_a", pa_p), ("pr_b", pb_p), ("pr_c", pc_p)):
        if rp is not None:
            dj = json.loads(rp.read_text(encoding="utf-8"))
            if (dj.get("paid_api_budget") or {}).get("model_calls", 1) != 0:
                v13_checks["violations"].append(f"{nm}_nonzero_model_calls")
    if pa_p is not None:
        dpa = json.loads(pa_p.read_text(encoding="utf-8"))
        for k, v in dpa.get("remediated_gpt6_v13_review_probes", {}).items():
            if not v.get("remediated"):
                v13_checks["violations"].append(f"pr_a_probe_not_remediated:{k}")
    if pb_p is not None:
        dpb = json.loads(pb_p.read_text(encoding="utf-8"))
        if not dpb["ordinary_archive_3layer_cache_audit"]["second_call_unchanged_sources"]["cache_hit"]:
            v13_checks["violations"].append("pr_b_ordinary_cache_second_call_missed")
    if pc_p is not None:
        dpc = json.loads(pc_p.read_text(encoding="utf-8"))
        if len(dpc.get("unexposed_evaluation_chains") or ()) < 8:
            v13_checks["violations"].append("pr_c_fewer_than_8_unexposed_source_clusters")
        for ch_c in dpc.get("unexposed_evaluation_chains") or ():
            for eps_arm in (ch_c.get("arms_target_results") or {}).values():
                for ep_c in eps_arm:
                    if ep_c.get("strict_pass") and (
                        ep_c.get("false_internal_confirmation") or float(ep_c.get("external_fresh48_mse", 1.0)) > 1e-7
                    ):
                        v13_checks["violations"].append("pr_c_dishonest_strict_pass_with_external_failure")

    scan_files = [
        "src/hegel_machine/cognition/branch_reconciliation.py",
        "src/hegel_machine/cognition/competing_object_programs.py",
        "src/hegel_machine/cognition/equal_tool_attribution.py",
        "src/hegel_machine/cognition/premise_commitment_index.py",
        "src/hegel_machine/cognition/unified_research.py",
        "scripts/run_v13_pr_ab_controls.py",
        "scripts/run_v13_pr_c_tightened_study.py",
    ]
    priv_pat = re.compile(r"\blab\._[a-zA-Z0-9_]+")
    key_pat = re.compile(r"sk-[A-Za-z0-9]{20,}")
    fake_probe_pat = re.compile(r'"(parametric_slot_calibration|active_threshold_or_interval_probe|premise_revised_probe)"')

    priv_hits, fake_hits, route_hits, key_hits = [], [], [], []
    scanned = 0
    for rel_m in scan_files:
        mp = _locate_source(bundle_dir, rel_m)
        if mp is not None:
            scanned += 1
            txt = mp.read_text(encoding="utf-8")
            for h in priv_pat.findall(txt):
                priv_hits.append({"file": rel_m, "match": h})
            for h in key_pat.findall(txt):
                key_hits.append({"file": rel_m, "match": h[:6] + "..."})
            if rel_m.startswith("src/"):
                for h in fake_probe_pat.findall(txt):
                    fake_hits.append({"file": rel_m, "match": h})
            if rel_m == "scripts/run_v13_pr_c_tightened_study.py":
                mod_ast = ast.parse(txt, filename=rel_m)
                for node in ast.walk(mod_ast):
                    if isinstance(node, (ast.If, ast.IfExp)):
                        ts = ast.unparse(node.test)
                        if "relation_condition" in ts:
                            route_hits.append({"file": rel_m, "lineno": getattr(node, "lineno", 0), "match": ts})

    return {
        "bundle_sha256_integrity": sha_checks,
        "historic_v10_13_receipts_untouched": v10_untouched,
        "v13_receipt_invariants": v13_checks,
        "static_anti_cheat_scan": {
            "scanned_files": scanned,
            "private_lab_attribute_reads": priv_hits,
            "fake_probe_string_labels_in_runtime": fake_hits,
            "harness_hidden_condition_routes": route_hits,
            "persisted_api_keys": key_hits,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    v13_probes = check_v13_gpt6_review_probes_ast(args.bundle_dir)
    prior_probes = check_v12_and_v11_probes_ast(args.bundle_dir)
    dom_check = check_certificate_domain_counterexample_exact_rational()
    bundle_audit = check_bundle_integrity_and_v13_receipts(args.bundle_dir)
    scan = bundle_audit["static_anti_cheat_scan"]

    all_ok = (
        v13_probes.get("all_v13_review_probes_remediated", False)
        and prior_probes.get("all_prior_probes_ok", False)
        and dom_check["all_modes_free_of_contradiction"]
        and not bundle_audit["bundle_sha256_integrity"]["mismatches"]
        and not bundle_audit["historic_v10_13_receipts_untouched"]["mismatches"]
        and bundle_audit["v13_receipt_invariants"]["all_three_v13_receipts_present"]
        and not bundle_audit["v13_receipt_invariants"]["violations"]
        and not scan["private_lab_attribute_reads"]
        and not scan["fake_probe_string_labels_in_runtime"]
        and not scan["harness_hidden_condition_routes"]
        and not scan["persisted_api_keys"]
    )
    report = {
        "verifier": "05_STANDALONE_VERIFY_V13.py",
        "v13_gpt6_review_probes_ast_check": v13_probes,
        "v12_and_v11_regression_probes_ast_check": prior_probes,
        "v10_2_exact_rational_domain_check": dom_check,
        "bundle_and_v13_receipt_audit": bundle_audit,
        "all_checks_passed": bool(all_ok),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
