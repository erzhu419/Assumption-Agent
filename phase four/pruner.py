"""Darwinian pruner (脑洞 E): deprecate/remove stale wisdoms.

Reads wisdom selection files (phase2_v3_selections*.json) — these map
problem_id → ranked list of wisdom IDs chosen by the retrieval stage.
v20 consumes only the first `max_wisdoms` (default 2) per problem; that
is the true activation signal.

Rules:
  1. active wisdom with 0 activations across scan → mark deprecated
  2. deprecated wisdom with 0 activations across a second scan → remove
  3. wisdom newer than the most recent scan file → skipped (not yet had a
     chance to activate)
  4. any activation counted: update `last_activated` + `activation_count`

Decisions are logged to `autonomous/prune_log.json` so trajectory is
reconstructable. Run with `--dry-run` to preview without writing.

Usage:
  python pruner.py                    # prune using all selection files
  python pruner.py --dry-run          # preview only
  python pruner.py --max-wisdoms 2    # match v20 default
  python pruner.py --selections phase2_v3_selections.json,phase2_v3_selections_v18.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase four"))

from wisdom_registry import (
    load_or_init_registry, save_registry, active_wisdoms,
    deprecate_wisdom, remove_wisdom, record_activation,
)

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
PRUNE_LOG = AUTO_DIR / "prune_log.json"


def discover_selection_files():
    """Return sorted list of selection file paths (oldest → newest)."""
    files = sorted(
        CACHE.glob("phase2_v*selections*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return files


def collect_activations(files, max_wisdoms):
    """Return {wid: count} across first `max_wisdoms` of each pid selection."""
    c = Counter()
    per_file = {}
    for f in files:
        try:
            sel = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [skip {f.name}] {e}")
            continue
        sub = Counter()
        for pid, ids in sel.items():
            for wid in ids[:max_wisdoms]:
                c[wid] += 1
                sub[wid] += 1
        per_file[f.name] = {
            "problems": len(sel),
            "activations": sum(sub.values()),
            "unique_wisdoms": len(sub),
        }
    return c, per_file


def _parse_dt(s):
    if not s or s == "original":
        return datetime(2020, 1, 1, tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime(2020, 1, 1, tzinfo=timezone.utc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wisdoms", type=int, default=2,
                    help="top-K per problem counted as activation (match v20)")
    ap.add_argument("--selections", default=None,
                    help="comma-sep list of selection files (relative to cache/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print decisions without writing registry/log")
    args = ap.parse_args()

    # 1. Load scan inputs
    if args.selections:
        files = [CACHE / f.strip() for f in args.selections.split(",")]
        files = [f for f in files if f.exists()]
    else:
        files = discover_selection_files()

    if not files:
        print("No selection files found — abort.")
        return

    print(f"Scanning {len(files)} selection files (top-{args.max_wisdoms} per problem):")
    for f in files:
        print(f"  • {f.name}  ({time.strftime('%Y-%m-%d', time.localtime(f.stat().st_mtime))})")

    newest_mtime = max(f.stat().st_mtime for f in files)
    newest_dt = datetime.fromtimestamp(newest_mtime, tz=timezone.utc)

    activations, per_file = collect_activations(files, args.max_wisdoms)
    total_acts = sum(activations.values())
    print(f"\nTotal activations: {total_acts}  unique wisdoms activated: {len(activations)}")

    # 2. Load registry
    registry = load_or_init_registry()

    # 3. Decisions
    decisions = []
    skipped_newborn = []

    # Snapshot existing activation counts first so record_activation only
    # ADDS to them, not double-counts prior rounds.
    # (record_activation is designed as append, so we need to only count NEW acts.)
    # For this Darwinian prune we don't try to do incremental: we recompute
    # contribution_gain on full scan. The `activation_count` field becomes a
    # rolling total — which is what we want.

    for w in registry["wisdoms"]:
        if w["status"] == "removed":
            continue

        wid = w["id"]
        act_count = activations.get(wid, 0)

        # Newborn protection: wisdom created after the newest selection file
        # had no opportunity to be activated.
        created_dt = _parse_dt(w.get("created_at"))
        if created_dt > newest_dt:
            skipped_newborn.append(wid)
            continue

        # Fold this scan into lifetime activation_count
        w["activation_count"] = w.get("activation_count", 0) + act_count
        if act_count > 0:
            w["last_activated"] = datetime.fromtimestamp(
                newest_mtime, tz=timezone.utc
            ).isoformat()

        if act_count > 0:
            decisions.append({"wid": wid, "status": w["status"],
                              "scan_activations": act_count, "action": "kept"})
            continue

        # No activation in this scan — take Darwinian action
        if w["status"] == "active":
            reason = (f"prune scan: 0 activations in {len(files)} selection files "
                      f"(total top-{args.max_wisdoms} acts = {total_acts})")
            w["status"] = "deprecated"
            w["deprecated_at"] = datetime.now(timezone.utc).isoformat()
            w["deprecation_reason"] = reason
            decisions.append({"wid": wid, "status": "deprecated",
                              "scan_activations": 0, "action": "deprecate",
                              "reason": reason})
        elif w["status"] == "deprecated":
            reason = f"prune scan: still 0 activations after deprecation ({len(files)} files)"
            w["status"] = "removed"
            w["removed_at"] = datetime.now(timezone.utc).isoformat()
            w["removal_reason"] = reason
            decisions.append({"wid": wid, "status": "removed",
                              "scan_activations": 0, "action": "remove",
                              "reason": reason})

    # bump version if any action was taken
    any_action = any(d["action"] in ("deprecate", "remove") for d in decisions)
    if any_action and not args.dry_run:
        # bump manually: minor for any state change
        v = registry["version"].replace("v", "")
        major, minor = v.split(".")
        registry["version"] = f"v{major}.{int(minor) + 1}"

    # 4. Summary
    n_active = sum(1 for w in registry["wisdoms"] if w["status"] == "active")
    n_dep = sum(1 for w in registry["wisdoms"] if w["status"] == "deprecated")
    n_rem = sum(1 for w in registry["wisdoms"] if w["status"] == "removed")

    deprecates = [d for d in decisions if d["action"] == "deprecate"]
    removes = [d for d in decisions if d["action"] == "remove"]

    print(f"\n=== Prune summary ===")
    print(f"  Active:     {n_active}")
    print(f"  Deprecated: {n_dep}  (+{len(deprecates)} this scan)")
    print(f"  Removed:    {n_rem}  (+{len(removes)} this scan)")
    print(f"  Skipped (newborn, created after scan): {len(skipped_newborn)} "
          f"{skipped_newborn[:5]}{'...' if len(skipped_newborn) > 5 else ''}")

    if deprecates:
        print(f"\n  → deprecated this round:")
        for d in deprecates[:20]:
            print(f"    - {d['wid']}")
    if removes:
        print(f"\n  → removed this round:")
        for d in removes[:20]:
            print(f"    - {d['wid']}")

    if args.dry_run:
        print("\n[DRY RUN] no files written")
        return

    # 5. Persist
    save_registry(registry)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_wisdoms": args.max_wisdoms,
        "scan_files": [f.name for f in files],
        "per_file_stats": per_file,
        "total_activations": total_acts,
        "deprecated": [d["wid"] for d in deprecates],
        "removed": [d["wid"] for d in removes],
        "registry_version_after": registry["version"],
        "active_count_after": n_active,
        "skipped_newborn": skipped_newborn,
    }
    log = json.loads(PRUNE_LOG.read_text(encoding="utf-8")) if PRUNE_LOG.exists() else []
    log.append(log_entry)
    PRUNE_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n  Prune log appended → {PRUNE_LOG.name}")


if __name__ == "__main__":
    main()
