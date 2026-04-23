"""Version-aware wisdom library + activation/performance tracking.

Loads and manages `wisdom_registry.json`:
  - Each wisdom has status (active/deprecated/removed)
  - Tracks activation count, last activated, contribution_gain
  - Supports version bump, wisdom append, wisdom deprecation

This is the persistent state of the autonomous loop — the "memory" of the
agent's library evolution.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).parent.parent
CACHE = PROJECT / "phase two" / "analysis" / "cache"
ORIG_WISDOM = CACHE / "wisdom_library.json"
REGISTRY_PATH = PROJECT / "phase four" / "autonomous" / "wisdom_registry.json"
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_or_init_registry() -> dict:
    """Initialize registry from wisdom_library.json if not exists."""
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    orig = json.loads(ORIG_WISDOM.read_text(encoding="utf-8"))
    registry = {
        "version": "v20.0",
        "created_at": _now_iso(),
        "last_updated": _now_iso(),
        "wisdoms": [
            {
                **w,
                "status": "active",
                "created_at": "original",
                "last_activated": None,
                "activation_count": 0,
                "contribution_gain": None,
                "source": "original",
            }
            for w in orig
        ],
    }
    save_registry(registry)
    return registry


def save_registry(registry: dict):
    registry["last_updated"] = _now_iso()
    REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2))


def active_wisdoms(registry: dict) -> list:
    return [w for w in registry["wisdoms"] if w.get("status") == "active"]


def append_wisdom(registry: dict, new_wisdom: dict, source: str,
                   keep_reason: str) -> str:
    """Append a new wisdom; auto-assigns W{N:03d} if no id."""
    active = active_wisdoms(registry)
    existing_ids = {w["id"] for w in registry["wisdoms"]}
    # next id: W + max numeric suffix + 1
    max_n = max(
        (int(w["id"][1:]) for w in registry["wisdoms"] if w["id"].startswith("W")),
        default=0
    )
    new_id = f"W{max_n+1:03d}"
    while new_id in existing_ids:
        max_n += 1
        new_id = f"W{max_n+1:03d}"

    entry = {
        **new_wisdom,
        "id": new_id,
        "status": "active",
        "created_at": _now_iso(),
        "last_activated": None,
        "activation_count": 0,
        "contribution_gain": None,
        "source": source,
        "keep_reason": keep_reason,
    }
    registry["wisdoms"].append(entry)
    bump_version(registry)
    return new_id


def deprecate_wisdom(registry: dict, wid: str, reason: str):
    for w in registry["wisdoms"]:
        if w["id"] == wid and w["status"] == "active":
            w["status"] = "deprecated"
            w["deprecated_at"] = _now_iso()
            w["deprecation_reason"] = reason
            bump_version(registry)
            return True
    return False


def remove_wisdom(registry: dict, wid: str, reason: str):
    for w in registry["wisdoms"]:
        if w["id"] == wid and w["status"] == "deprecated":
            w["status"] = "removed"
            w["removed_at"] = _now_iso()
            w["removal_reason"] = reason
            bump_version(registry)
            return True
    return False


def record_activation(registry: dict, wid: str):
    for w in registry["wisdoms"]:
        if w["id"] == wid:
            w["activation_count"] = w.get("activation_count", 0) + 1
            w["last_activated"] = _now_iso()
            return True
    return False


def update_contribution_gain(registry: dict, wid: str, gain: float, weight: float = 1.0):
    """Rolling average of contribution gain, weighted by `weight`."""
    for w in registry["wisdoms"]:
        if w["id"] == wid:
            cur = w.get("contribution_gain")
            n = w.get("gain_samples", 0)
            new_sum = (cur * n if cur is not None else 0) + gain * weight
            new_n = n + weight
            w["contribution_gain"] = new_sum / new_n
            w["gain_samples"] = new_n
            return True
    return False


def bump_version(registry: dict):
    """Bump semantic version: minor for add, patch for deprecate/remove."""
    v = registry["version"]
    major, minor, *patch = v.replace("v", "").split(".")
    patch_num = int(patch[0]) if patch else 0
    # Just bump patch each time for simplicity
    registry["version"] = f"v{major}.{int(minor) + 1}"


def export_to_wisdom_library(registry: dict, out_path: Path):
    """Export only active wisdoms in v16-compatible format (strips metadata)."""
    META_FIELDS = {"status", "created_at", "last_activated",
                    "activation_count", "contribution_gain", "source",
                    "keep_reason", "gain_samples", "deprecated_at",
                    "deprecation_reason", "removed_at", "removal_reason"}
    export = [
        {k: v for k, v in w.items() if k not in META_FIELDS}
        for w in active_wisdoms(registry)
    ]
    out_path.write_text(json.dumps(export, ensure_ascii=False, indent=2))
    return len(export)


if __name__ == "__main__":
    r = load_or_init_registry()
    print(f"Registry version: {r['version']}")
    print(f"Total wisdoms: {len(r['wisdoms'])}")
    print(f"Active: {len(active_wisdoms(r))}")
