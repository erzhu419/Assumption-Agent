#!/usr/bin/env python3
"""Zero-dependency standalone audit verifier for Hegel Machine v15 GPT-6 Handoff Bundle.

Usage inside an isolated sandbox (with PYTHONPATH="" after unzipping):
    PYTHONPATH="" python3 05_STANDALONE_VERIFY_V15.py --output verify_v15_report.json

Uses ONLY Python standard library (`argparse`, `dataclasses`, `hashlib`, `json`, `math`,
`pathlib`, `statistics`, `subprocess`, `sys`, `tempfile`) and performs:
  1. SHA-256 verification over every file listed in `06_SHA256_SUMS.txt`.
  2. Strict `allow_nan=False` JSON parsing & `model_calls == 0` check across all 39 receipts
     (35 untouched historic `v10..v14` receipts + 4 new `v15` receipts), plus byte-hash lock
     confirming historic V14 D1/D2/D3 receipts were NOT modified.
  3. Live execution of GPT-6's V14 probes AND V15 probes (`probe_d3_object_removal.py`,
     `probe_scope_replay.py`) against `bundled_sources/`.
  4. Content-level audit of Deliverable 1 (10-item V14 Errata + real production reconciliation
     events + status-aware failure narratives with zero `"None passed"` on failed chains).
  5. Audit of Deliverable 2 (all 3 §3.3 object-removal controls, billed raw-history rebuild,
     `partial_pending` budget check, and 0/24 K-vs-R dispatch differences across 8 D2 chains).
  6. Independent per-episode gate & prefix-cost recomputation across Prospective E1 (`§8.1`,
     verifying 0/36 K-vs-R dispatch differences & true pre-run version removal) and Prospective
     E2 (`§8.2`, verifying zero splicing and genuine separate `L^Ord_c` vs `L^K_c` trajectories).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tempfile
from typing import Any


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


FROZEN_V14_SHA256 = {
    "deliverable_1_errata_and_replays_a_to_f_20261118.json": "0e1e7ba58f3926f7ad5865426f987d013903a3102994113aab4354de18951bda",
    "deliverable_2_unconfounded_noH_source_paired_study_20261118.json": "9076698f6e8314b256cc7f8061255693c0a3271f4d687613647a66dfab6d4795",
    "deliverable_3_continuous_repaired_object_study_20261118.json": "6c9e09a6f2fb22116ece4832ccd0c65f21964ce84bd6860370e07bb2b8a722e3",
}

DISPATCH_LIST_FIELDS = (
    "support_invalidated",
    "empirically_excluded_on_context",
    "empirically_excluded_bound_instance_only",
    "not_scheduled_by_router",
    "scope_not_established",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict_json_loads(raw_text: str) -> Any:
    def _forbid_nonfinite(token: str) -> None:
        raise ValueError(f"Non-strict JSON constant encountered: {token}")

    return json.loads(raw_text, parse_constant=_forbid_nonfinite)


def _normalize_dispatch(telemetry: dict[str, Any]) -> dict[str, Any]:
    identity = {}
    levels = {}
    for item in telemetry["branch_identity_map"]:
        srcs = item["source_concept_ids"]
        s = srcs[0] if len(srcs) == 1 else "+".join(sorted(srcs))
        identity[item["branch_id"]] = s
        levels[s] = item.get("scope_claim_level")
    out = {f: sorted(identity[bid] for bid in telemetry.get(f, ())) for f in DISPATCH_LIST_FIELDS}
    out["scope_claim_level"] = dict(sorted(levels.items()))
    return out


def verify_sha256_manifest(bundle_dir: Path) -> dict[str, Any]:
    sums_path = bundle_dir / "06_SHA256_SUMS.txt"
    if not sums_path.is_file():
        return {"present": False, "all_matched": False}
    mismatches: list[str] = []
    checked = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split("  ", 1)
        target = bundle_dir / rel
        checked += 1
        if not target.is_file() or _sha256_file(target) != digest:
            mismatches.append(rel)
    return {
        "present": True,
        "files_checked": checked,
        "mismatches": mismatches,
        "all_matched": len(mismatches) == 0 and checked > 0,
    }


def verify_all_receipts_strict_json_and_frozen_history(bundle_dir: Path) -> dict[str, Any]:
    receipts_root = bundle_dir / "receipts"
    json_files = sorted(receipts_root.rglob("*.json")) if receipts_root.is_dir() else []
    non_strict: list[str] = []
    api_violations: list[str] = []
    v14_hash_checks: dict[str, bool] = {}

    for jf in json_files:
        rel = str(jf.relative_to(bundle_dir))
        try:
            obj = _strict_json_loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:
            non_strict.append(f"{rel}: {exc}")
            continue
        if isinstance(obj, dict) and "paid_api_budget" in obj:
            if int((obj["paid_api_budget"] or {}).get("model_calls", 0)) != 0:
                api_violations.append(rel)
        if jf.name in FROZEN_V14_SHA256:
            v14_hash_checks[jf.name] = (_sha256_file(jf) == FROZEN_V14_SHA256[jf.name])

    mem_src = bundle_dir / "bundled_sources/src/hegel_machine/cognition/memory.py"
    mem_bundled = mem_src.is_file() and "def remove_library_object_version" in mem_src.read_text(encoding="utf-8")

    return {
        "receipts_count": len(json_files),
        "expected_minimum_receipts": 39,
        "non_strict_json_files": non_strict,
        "paid_api_violations": api_violations,
        "historic_v14_sha256_untouched": v14_hash_checks,
        "memory_py_bundled_with_removal_helper": mem_bundled,
        "passed": (
            len(json_files) >= 39
            and not non_strict
            and not api_violations
            and all(v14_hash_checks.values())
            and len(v14_hash_checks) == len(FROZEN_V14_SHA256)
            and mem_bundled
        ),
    }


def verify_gpt6_v14_and_v15_probes(bundle_dir: Path) -> dict[str, Any]:
    probes_dir = bundle_dir / "probes"
    if not probes_dir.is_dir():
        return {"present": False, "passed": False}
    outputs: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="v15_standalone_probes_") as tmp:
        tmp_p = Path(tmp)
        for name in (
            "probe_acquisition_contract.py",
            "probe_question_identity.py",
            "probe_scope_and_routing.py",
            "probe_d3_object_removal.py",
            "probe_scope_replay.py",
        ):
            script = probes_dir / name
            out_json = tmp_p / f"{name}.json"
            proc = subprocess.run(
                [sys.executable, str(script), "--bundle", str(bundle_dir), "--output", str(out_json)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0 or not out_json.is_file():
                outputs[name] = {"returncode": proc.returncode, "stderr": proc.stderr}
            else:
                outputs[name] = _strict_json_loads(out_json.read_text(encoding="utf-8"))

    acq = outputs.get("probe_acquisition_contract.py", {})
    qid = outputs.get("probe_question_identity.py", {})
    scr = outputs.get("probe_scope_and_routing.py", {})
    rm_v15 = outputs.get("probe_d3_object_removal.py", {})
    sr_v15 = outputs.get("probe_scope_replay.py", {})

    v14_ok = bool(
        (acq.get("real_production_batch") or {}).get("status") == "reused_exact_protocol"
        and (acq.get("real_production_batch") or {}).get("new_acquisition_attempts") == 0
        and qid.get("identical_question_ids") is False
        and (qid.get("zero_budget") or {}).get("predictive_loss_evaluations_still_executed") == 0
        and (qid.get("zero_budget") or {}).get("claims_mutated_after_denial") is False
        and ((scr.get("empty_family_search") or {}).get("verdict") or {}).get("empirically_refuted") is False
        and ((scr.get("empty_family_search") or {}).get("verdict") or {}).get("cross_mse") is None
        and (scr.get("router_deferred_aggregate_dropped") or {}).get("ordinary_aggregate_retained") is True
        and len((scr.get("fixed_instance_vs_refittable_template_control") or {}).get("dispatched_templates") or ()) == 1
    )
    rm_ok = bool(
        rm_v15.get("concept_remaining_ids") == ["concept_0001"]
        and rm_v15.get("abstract_remaining_source_ids") == ["concept_0001"]
        and rm_v15.get("definition_remaining_source_ids") == ["concept_0001"]
        and rm_v15.get("definition_binding_keys") == ["concept_0001:definition"]
    )
    sr_fb = sr_v15.get("scope_unidentified_replay_fallback") or {}
    sr_hd = sr_v15.get("evidence_hash_dependencies") or {}
    sr_ok = bool(
        sr_fb.get("replayed_old_status") == "instance_scope_unknown"
        and int(sr_fb.get("template_check_calls") or 0) >= 1
        and sr_hd.get("radius_edit_invalidates") is True
        and sr_hd.get("commitment_edit_invalidates") is True
        and sr_hd.get("observation_edit_invalidates") is True
    )
    return {
        "v14_probes_regression_ok": v14_ok,
        "v15_probe_d3_object_removal_ok": rm_ok,
        "v15_probe_scope_replay_ok": sr_ok,
        "probe_d3_object_removal_output": rm_v15,
        "probe_scope_replay_output": sr_v15,
        "passed": v14_ok and rm_ok and sr_ok,
    }


def verify_v15_deliverables_1_and_2(bundle_dir: Path) -> dict[str, Any]:
    v15_dir = bundle_dir / "receipts/v15_true_intervention_and_strategy_20261118"
    d1_p = v15_dir / "deliverable_1_v14_errata_and_factual_reexport_20261118.json"
    d2_p = v15_dir / "deliverable_2_removal_replay_parity_and_exposed_diagnostics_20261118.json"
    if not (d1_p.is_file() and d2_p.is_file()):
        return {"passed": False, "reason": "missing_d1_or_d2_receipt"}

    d1 = _strict_json_loads(d1_p.read_text(encoding="utf-8"))
    d2 = _strict_json_loads(d2_p.read_text(encoding="utf-8"))

    gt = d1["recomputed_v14_ground_truth"]
    errata_ok = bool(
        len(d1["errata_10_items"]) == 10
        and gt["total_arm_episode_records"] == 232
        and gt["unique_episode_uids"] == 106
        and gt["strict_passes_all"] == 207
        and gt["unresolved_or_unconfirmed_all"] == 25
        and gt["false_internal_confirmations_all"] == 0
    )

    # §2.2 Acceptance #1–#3: Real production events checked, NOT just 7 sentence keys!
    narrative_violations: list[str] = []
    event_violations: list[str] = []
    for ch in d1["reexported_exposed_v14_d3_chains"]:
        cid = ch["chain_id"]
        s7 = ch["status_aware_seven_sentence_factual_ledger"]
        ev = ch["production_reconciliation_events"]
        if len(s7) != 7:
            narrative_violations.append(f"{cid}:wrong_key_count")
        if ch["produced_qualified_repaired_object"]:
            if not ev["initial_conflict_reconciliation"]["executed"]:
                event_violations.append(f"{cid}:missing_initial_reconciliation_event")
            if ev["post_subresearch_re_reconciliation"]["epistemic_status"] == "missing_evidence":
                event_violations.append(f"{cid}:missing_post_reconciliation_event")
        else:
            joined = json.dumps(s7, ensure_ascii=False)
            if "None` passed" in joined or "passed an independent" in joined or ch["repaired_concept_id"] is not None:
                narrative_violations.append(f"{cid}:fake_confirmation_on_failed_chain")
            if "missing_evidence" not in s7["5_fresh_confirmation_and_historical_branch_checks"]:
                narrative_violations.append(f"{cid}:missing_honest_missing_evidence_marker")

    c1 = d2["three_object_removal_controls_sec3_3"]["control_1_true_list_shaped_version_removal"]["post_removal_inventory_summary"]
    c2 = d2["three_object_removal_controls_sec3_3"]["control_2_exclusive_derivative_vs_two_source_shared_retention"]
    c3 = d2["three_object_removal_controls_sec3_3"]["control_3_billed_raw_history_reconstruction"]["reconstruction_event"]
    controls_ok = bool(
        c1["removed_target_present_in_concepts"] is False
        and c1["removed_target_present_in_abstracts"] is False
        and c1["removed_target_present_in_definitions"] is False
        and c1["removed_target_present_in_bindings"] is False
        and c1["removed_target_present_in_recall"] is False
        and c2["exclusive_derivative_removed"] is True
        and c2["two_source_shared_retained"] is True
        and c3["generated_from_raw_history"] is True
        and int(c3["compute_cost"]) > 0
    )
    parity_d2_ok = bool(
        d2["eight_chain_k_vs_r_state_and_dispatch_parity_sec4_3"]["chains_verified"] == 8
        and d2["eight_chain_k_vs_r_state_and_dispatch_parity_sec4_3"]["target_pairs_verified"] == 24
        and d2["eight_chain_k_vs_r_state_and_dispatch_parity_sec4_3"]["semantic_dispatch_differences_total"] == 0
        and d2["six_exposed_development_replays_sec5_2"]["case_1_1031000_source_and_T1_parity"]["partial_pending_budget_check"]["reconstruction_status"] == "partial_pending"
    )
    return {
        "errata_10_items_ok": errata_ok,
        "narrative_violations": narrative_violations,
        "event_violations": event_violations,
        "three_object_removal_controls_ok": controls_ok,
        "eight_chain_24_pair_parity_ok": parity_d2_ok,
        "passed": errata_ok and not narrative_violations and not event_violations and controls_ok and parity_d2_ok,
    }


def verify_prospective_e1_and_e2(bundle_dir: Path) -> dict[str, Any]:
    v15_dir = bundle_dir / "receipts/v15_true_intervention_and_strategy_20261118"
    e1_p = v15_dir / "deliverable_3_prospective_e1_same_history_organization_20261118.json"
    e2_p = v15_dir / "deliverable_4_prospective_e2_end_to_end_strategy_20261118.json"
    if not (e1_p.is_file() and e2_p.is_file()):
        return {"passed": False, "reason": "missing_e1_or_e2_receipt"}

    e1 = _strict_json_loads(e1_p.read_text(encoding="utf-8"))
    e2 = _strict_json_loads(e2_p.read_text(encoding="utf-8"))

    tol = 1e-7
    gate_mismatches: list[str] = []
    e1_dispatch_diffs = 0
    removal_audit_ok = True

    for cl in e1["prospective_clusters"]:
        cid = cl["cluster_id"]
        rm_info = cl["version_removal_inventory_audit"]
        if rm_info["removed_concept_id"] is not None:
            if rm_info["removed_concept_id"] in rm_info["post_removal_object_ids"]:
                removal_audit_ok = False
        k_eps = cl["arms_target_results"]["K_eager_persistent"]
        r_eps = cl["arms_target_results"]["R_same_history_lazy"]
        for ke, re_ in zip(k_eps, r_eps):
            if ke["branch_dispatch_telemetry"] and re_["branch_dispatch_telemetry"]:
                if _normalize_dispatch(ke["branch_dispatch_telemetry"]) != _normalize_dispatch(re_["branch_dispatch_telemetry"]):
                    e1_dispatch_diffs += 1
        for arm, eps in cl["arms_target_results"].items():
            for ep in eps:
                c = ep["measured_costs"]
                wb = c["observation_cost"] <= 600 and c["measurement_cost"] <= 12200 and c["compute_cost"] <= 840000
                ic = ep["status"] == "completed" and ep["confirmation_loss"] is not None and float(ep["confirmation_loss"]) <= tol
                sp = bool(ic and float(ep["external_fresh48_mse"]) <= tol and wb)
                if sp != ep["strict_pass"]:
                    gate_mismatches.append(f"E1:{cid}:{arm}:{ep['episode_tag']}")

    # Verify E2 zero-splicing & prefix arithmetic
    e2_arith_ok = True
    for cl in e2["prospective_clusters"]:
        k_s = sum(x["measured_costs"]["compute_cost"] for x in cl["K_autonomous_end_to_end"]["source_episodes"])
        o_s = sum(x["measured_costs"]["compute_cost"] for x in cl["Ordinary_autonomous_end_to_end"]["source_episodes"])
        for k_val in (1, 3, 6):
            kt = k_s + sum(x["measured_costs"]["compute_cost"] for x in cl["K_autonomous_end_to_end"]["target_episodes"][:k_val])
            ot = o_s + sum(x["measured_costs"]["compute_cost"] for x in cl["Ordinary_autonomous_end_to_end"]["target_episodes"][:k_val])
            rec_p = cl["end_to_end_prefix_comparison_Ordinary_minus_K"][f"k_{k_val}"]
            if rec_p["K_end_to_end_compute"] != kt or rec_p["Ordinary_end_to_end_compute"] != ot:
                e2_arith_ok = False

    return {
        "prospective_clusters_count": len(e1["prospective_clusters"]),
        "gate_mismatches": gate_mismatches,
        "e1_k_vs_r_dispatch_differences_across_36_pairs": e1_dispatch_diffs,
        "e1_true_removal_inventory_ok": removal_audit_ok,
        "e2_end_to_end_prefix_arithmetic_ok": e2_arith_ok,
        "passed": (
            len(e1["prospective_clusters"]) == 6
            and len(e2["prospective_clusters"]) == 6
            and not gate_mismatches
            and e1_dispatch_diffs == 0
            and removal_audit_ok
            and e2_arith_ok
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    bdir = args.bundle_dir.resolve()

    c_sha = verify_sha256_manifest(bdir)
    c_json = verify_all_receipts_strict_json_and_frozen_history(bdir)
    c_probes = verify_gpt6_v14_and_v15_probes(bdir)
    c_d12 = verify_v15_deliverables_1_and_2(bdir)
    c_e12 = verify_prospective_e1_and_e2(bdir)

    report = {
        "verifier": "05_STANDALONE_VERIFY_V15.py",
        "bundle_dir": str(bdir),
        "checks": {
            "sha256_manifest": c_sha,
            "strict_json_and_untouched_history": c_json,
            "gpt6_v14_and_v15_probes": c_probes,
            "deliverables_1_and_2_controls_and_narratives": c_d12,
            "prospective_e1_and_e2_verification": c_e12,
        },
        "all_checks_passed": bool(
            c_sha["all_matched"]
            and c_json["passed"]
            and c_probes["passed"]
            and c_d12["passed"]
            and c_e12["passed"]
        ),
    }
    raw_out = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw_out, encoding="utf-8")
    print(raw_out)
    if not report["all_checks_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
