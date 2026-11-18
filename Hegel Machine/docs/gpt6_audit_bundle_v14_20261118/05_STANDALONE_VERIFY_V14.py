#!/usr/bin/env python3
"""Zero-dependency standalone audit verifier for Hegel Machine v14 GPT-6 Handoff Bundle.

Usage inside GPT-6 sandbox (after unzipping Hegel_v14_GPT6_Audit_Handoff_20261118.zip):
    python3 05_STANDALONE_VERIFY_V14.py --output verify_v14_report.json

Uses ONLY Python standard library (`argparse`, `dataclasses`, `hashlib`, `itertools`,
`json`, `math`, `pathlib`, `statistics`, `subprocess`, `sys`, `tempfile`) and makes
zero network calls, zero paid API model calls (`model_calls = 0`), and zero file edits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import product
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


def verify_all_receipts_strict_json(bundle_dir: Path) -> dict[str, Any]:
    receipts_root = bundle_dir / "receipts"
    json_files = sorted(receipts_root.rglob("*.json")) if receipts_root.is_dir() else []
    non_strict: list[str] = []
    zero_api_violations: list[str] = []
    for jf in json_files:
        rel = str(jf.relative_to(bundle_dir))
        try:
            obj = _strict_json_loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:
            non_strict.append(f"{rel}: {exc}")
            continue
        if isinstance(obj, dict) and "paid_api_budget" in obj:
            if int((obj["paid_api_budget"] or {}).get("model_calls", 0)) != 0:
                zero_api_violations.append(rel)
    return {
        "receipts_count": len(json_files),
        "expected_minimum_receipts": 35,
        "non_strict_json_files": non_strict,
        "paid_api_violations": zero_api_violations,
        "passed": len(json_files) >= 35 and not non_strict and not zero_api_violations,
    }


def verify_gpt6_v14_probes(bundle_dir: Path) -> dict[str, Any]:
    probes_dir = bundle_dir / "probes"
    if not probes_dir.is_dir():
        return {"present": False, "passed": False}
    outputs: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="v14_standalone_probes_") as tmp:
        tmp_p = Path(tmp)
        for name in (
            "probe_acquisition_contract.py",
            "probe_question_identity.py",
            "probe_scope_and_routing.py",
        ):
            out_file = tmp_p / name.replace(".py", ".json")
            subprocess.run(
                [sys.executable, str(probes_dir / name), "--bundle", str(bundle_dir), "--output", str(out_file)],
                check=True,
                capture_output=True,
                text=True,
            )
            outputs[name.replace(".py", "")] = _strict_json_loads(out_file.read_text(encoding="utf-8"))

    p_acq = outputs["probe_acquisition_contract"]
    p_qid = outputs["probe_question_identity"]
    p_sr = outputs["probe_scope_and_routing"]
    checks = {
        "real_production_batch_reused_exact_protocol": p_acq["real_production_batch"]["status"] == "reused_exact_protocol",
        "real_production_batch_zero_new_acquisitions": p_acq["real_production_batch"]["new_acquisition_attempts"] == 0,
        "distinct_parent_programs_distinct_question_ids": p_qid["identical_question_ids"] is False,
        "zero_budget_zero_loss_evaluations": p_qid["zero_budget"]["predictive_loss_evaluations_still_executed"] == 0,
        "zero_budget_zero_claim_mutations": p_qid["zero_budget"]["claims_mutated_after_denial"] is False,
        "empty_family_search_not_empirically_refuted": p_sr["empty_family_search"]["verdict"]["empirically_refuted"] is False,
        "empty_family_search_strict_json_null_mse": p_sr["empty_family_search"]["verdict"]["cross_mse"] is None,
        "router_deferred_ordinary_aggregate_retained": p_sr["router_deferred_aggregate_dropped"]["ordinary_aggregate_retained"] is True,
        "deferred_router_families_consumed": p_sr["router_deferred_family_not_in_consumed_list"]["consumer_reference_count"] > 0,
        "denied_fixed_cross_check_zero_fit_calls": p_sr["denied_fixed_cross_check_still_executed"]["fit_calls"] == 0,
        "fixed_instance_scope_preserves_refittable_template": len(p_sr["fixed_instance_vs_refittable_template_control"]["dispatched_templates"]) == 1,
    }
    return {
        "present": True,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_v14_deliverables_recomputation(bundle_dir: Path) -> dict[str, Any]:
    v14_dir = bundle_dir / "receipts" / "v14_continuous_object_and_fair_comparison_20261118"
    d1 = _strict_json_loads((v14_dir / "deliverable_1_errata_and_replays_a_to_f_20261118.json").read_text(encoding="utf-8"))
    d2 = _strict_json_loads((v14_dir / "deliverable_2_unconfounded_noH_source_paired_study_20261118.json").read_text(encoding="utf-8"))
    d3 = _strict_json_loads((v14_dir / "deliverable_3_continuous_repaired_object_study_20261118.json").read_text(encoding="utf-8"))

    # 1. Verify Deliverable 1 Errata & Replays A–F
    e1 = d1["v13_five_item_errata"]["erratum_1_ordinary_source_strict_count"]["actual_recomputed_counts"]
    d1_ok = (
        d1["all_gpt6_v14_probes_remediated"] is True
        and d1["all_replays_a_to_f_passed"] is True
        and e1["ordinary_source_total"] == 32
        and e1["ordinary_source_strict_pass"] == 31
        and e1["ordinary_source_false_confirmations_count"] == 1
        and e1["common_calibration_anchors_strict_pass"] == 8
    )

    # 2. Recompute Deliverable 2 Prefix Full Costs & Cluster Paired Diffs from raw chain records
    chains_d2 = d2["unexposed_evaluation_chains"]
    prefix_recompute_ok = True
    for ch in chains_d2:
        c_k_maint = ch["source_formation_measured_ledger"]["source_reconciliation_maintenance_compute_C_K_maint"]
        r_eps = ch["arms_target_results"]["R_lazy_noH"]
        k_eps = ch["arms_target_results"]["K_eager_noH"]
        for k_step in (1, 2, 3):
            expected_delta = sum(r_eps[i]["measured_costs"]["compute_cost"] for i in range(k_step)) - (
                c_k_maint + sum(k_eps[i]["measured_costs"]["compute_cost"] for i in range(k_step))
            )
            recorded_delta = ch["prefix_full_cost_comparison_R_minus_K"][f"k_{k_step}"]["delta_C_c_k_R_minus_K"]
            if expected_delta != recorded_delta:
                prefix_recompute_ok = False

    r_arm = d2["separate_tables"]["target_arms_24_per_arm"]["R_lazy_noH"]
    k_arm = d2["separate_tables"]["target_arms_24_per_arm"]["K_eager_noH"]
    d2_ok = (
        len(chains_d2) == 8
        and prefix_recompute_ok
        and r_arm["H_star_front_priority_enabled"] is False
        and k_arm["H_star_front_priority_enabled"] is False
        and r_arm["cache_cold_vs_warm_breakdown"]["unconsumed_layer2_H_star_built_total"] == 0
    )

    # 3. Verify Deliverable 3 Continuous Object Ledgers & 7-Sentence Ledgers
    chains_d3 = d3["all_attempted_continuous_chains"]
    seven_keys = {
        "1_prior_ordinary_experiences_and_versions",
        "2_refuting_real_evidence_and_failed_versions",
        "3_reconciliation_choice_rationale",
        "4_subanswer_parent_mutation",
        "5_fresh_confirmation_and_historical_branch_checks",
        "6_unseen_target_consumption_and_action_change",
        "7_equal_capability_ordinary_method_comparison",
    }
    d3_ok = (
        len(chains_d3) == 6
        and d3["protocol_guarantees"]["question_A_equal_latent_api_and_equal_budget"] is True
        and d3["protocol_guarantees"]["question_B_minimal_single_object_removals_preserve_other_sources_and_latent_api"] is True
        and all(set(ch["seven_sentence_factual_ledger"].keys()) == seven_keys for ch in chains_d3)
    )

    return {
        "deliverable_1_verified": d1_ok,
        "deliverable_2_verified": d2_ok,
        "deliverable_3_verified": d3_ok,
        "passed": d1_ok and d2_ok and d3_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    bundle_dir = args.bundle.expanduser().resolve()
    sha_check = verify_sha256_manifest(bundle_dir)
    json_check = verify_all_receipts_strict_json(bundle_dir)
    probes_check = verify_gpt6_v14_probes(bundle_dir)
    deliv_check = verify_v14_deliverables_recomputation(bundle_dir)

    report = {
        "verifier": "05_STANDALONE_VERIFY_V14.py",
        "bundle_dir": str(bundle_dir),
        "all_checks_passed": (
            sha_check["all_matched"]
            and json_check["passed"]
            and probes_check["passed"]
            and deliv_check["passed"]
        ),
        "sha256_manifest_check": sha_check,
        "strict_json_and_zero_api_check": json_check,
        "gpt6_v14_probes_check": probes_check,
        "v14_deliverables_recomputation_check": deliv_check,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["all_checks_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
