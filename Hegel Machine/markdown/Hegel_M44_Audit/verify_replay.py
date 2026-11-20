"""Independently check M4.4 committed/replayed evidence, counts, costs and every JSON difference.

No source edits, scientific acquisitions or model calls. Retains differences rather than
normalizing them away. Adapted from the M4.3 audit's verified read-only comparison.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
from statistics import mean


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walkdiff(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        result = []
        for k in sorted(x.keys() | y.keys()):
            if k not in x or k not in y:
                result.append({"path": path + "/" + str(k), "committed": x.get(k), "replay": y.get(k)})
            else:
                result.extend(walkdiff(x[k], y[k], path + "/" + str(k)))
        return result
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return [{"path": path, "committed_length": len(x), "replay_length": len(y)}]
        return [d for i, (xx, yy) in enumerate(zip(x, y)) for d in walkdiff(xx, yy, path + "/" + str(i))]
    if x == y:
        return []
    d = {"path": path, "committed": x, "replay": y}
    if isinstance(x, (float, int)) and isinstance(y, (float, int)) and not isinstance(x, bool) and not isinstance(y, bool):
        d["numeric_delta"] = y - x
    return [d]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_receipts(rows, extra_key=None):
    keys = [tuple([r["arm"], r["seed"], r["target_id"]] + ([r[extra_key]] if extra_key else [])) for r in rows]
    grouped = defaultdict(list)
    ledger_bad = []
    for r in rows:
        label = r["arm"] + (" / " + str(r[extra_key]) if extra_key else "")
        grouped[label].append(r)
        c = r["cost"]
        phases = c["phase_breakdown"]
        errors = {}
        for phase_field, total_field, eps in (("meas", "measurement_cost", 1e-5), ("compute", "compute_cost", 1e-4), ("obs", "observation_cost", 1e-5)):
            delta = sum(p[phase_field] for p in phases.values()) - c[total_field]
            if abs(delta) > eps:
                errors[total_field] = delta
        delta = c["measurement_cost"] + .05 * c["compute_cost"] - c["composite_cost"]
        if abs(delta) > .000051:
            errors["composite_cost"] = delta
        if errors:
            ledger_bad.append({"key": [r["arm"], r["seed"], r["target_id"]], "errors": errors})
    reports = {}
    for name, group in sorted(grouped.items()):
        inside = [r for r in group if not r["is_out_of_scope"]]
        outside = [r for r in group if r["is_out_of_scope"]]
        reports[name] = {
            "in_scope": len(inside), "strict_pass": sum(r["strict_pass"] for r in inside),
            "false_confirmation": sum(r["false_confirmation"] for r in inside),
            "oos": len(outside), "oos_unknown": sum(r["status"] == "scoped_unknown" for r in outside),
            "max_overlap": max(r["prefix_confirm_overlap_count"] for r in group),
            "confirmed_via_recall": sum(r["confirmed_via_recall"] for r in inside),
            "non_direct_walk_wins": sum(r["non_direct_walk_win"] for r in inside),
            "total_candidates": sum(r["evaluated_candidate_count"] for r in inside),
            "total_M": round(sum(r["cost"]["measurement_cost"] for r in inside), 6),
            "total_K": round(sum(r["cost"]["compute_cost"] for r in inside), 6),
            "total_C": round(sum(r["cost"]["composite_cost"] for r in inside), 6),
        }
    return {"count": len(rows), "unique_keys": len(set(keys)), "duplicate_keys": [list(k) for k, n in Counter(keys).items() if n > 1], "ledger_mismatches": ledger_bad, "arms": reports}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--committed", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = {"committed": args.committed, "replay": args.replay}
    hash_checks = {}
    for label, root in roots.items():
        bad = []; count = 0
        for line in (root / "SHA256SUMS.txt").read_text().splitlines():
            if not line.strip():
                continue
            expected, name = line.split(maxsplit=1); name = name.lstrip("*"); count += 1
            if not (root / name).is_file() or sha(root / name) != expected:
                bad.append(name)
        hash_checks[label] = {"count": count, "mismatches": bad}
    files = {label: {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts} for label, root in roots.items()}
    old, new = files["committed"], files["replay"]
    file_diffs = []; structured = {}
    for name in sorted(old.keys() | new.keys()):
        if name not in old or name not in new:
            file_diffs.append({"path": name, "missing_in": "committed" if name not in old else "replay"})
        elif sha(old[name]) != sha(new[name]):
            file_diffs.append({"path": name, "committed_sha256": sha(old[name]), "replay_sha256": sha(new[name])})
            if name.endswith(".json"):
                structured[name] = walkdiff(json.loads(old[name].read_text()), json.loads(new[name].read_text()))
            elif name.endswith(".jsonl"):
                structured[name] = walkdiff(read_rows(old[name]), read_rows(new[name]))
    datasets = {
        "part2": ("part2_m4_4_execution_receipts.jsonl", None),
        "fork": ("part2_m4_4_fork_noise_receipts.jsonl", "noise_regime"),
        "part3": ("part3_m4_4_online_writeback_receipts.jsonl", None),
    }
    checks = {}
    for label, (fname, key) in datasets.items():
        checks[label] = check_receipts(read_rows(args.replay / fname), key)
        differences = structured.get(fname, [])
        checks[label]["all_receipt_differences"] = len(differences)
        checks[label]["selected_model_sha_differences"] = sum(d["path"].endswith("/selected_model_sha256") for d in differences)
        checks[label]["other_differences"] = [d for d in differences if not d["path"].endswith("/selected_model_sha256")]
    summary_names = ["part1_m4_3_audit_closure_verification.json", "part2_m4_4_continual_benchmark_summary.json", "part3_m4_4_online_writeback_stream_summary.json"]
    result = {
        "python": platform.python_version(), "sha_checks": hash_checks,
        "committed_file_count": len(old), "replay_file_count": len(new),
        "byte_identical_files": len(old.keys() & new.keys()) - sum("missing_in" not in d for d in file_diffs),
        "file_differences": file_diffs, "structured_differences": structured,
        "summaries_exact": {name: not structured.get(name) for name in summary_names},
        "receipt_checks": checks,
        "online_writeback_artifacts_in_delivered_bundle": [n for n in new if "online_writebacks" in n or "post_stream" in n],
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    brief = {k: v for k, v in result.items() if k not in {"structured_differences", "file_differences", "receipt_checks"}}
    brief["receipt_counts"] = {k: {x: v[x] for x in ("count", "unique_keys", "all_receipt_differences", "selected_model_sha_differences")} for k, v in checks.items()}
    brief["ledger_mismatch_counts"] = {k: len(v["ledger_mismatches"]) for k, v in checks.items()}
    brief["non_sha_receipt_differences"] = {k: len(v["other_differences"]) for k, v in checks.items()}
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
