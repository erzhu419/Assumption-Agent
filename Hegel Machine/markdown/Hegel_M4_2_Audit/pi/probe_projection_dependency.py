"""Read-only dependency controls for M4.2 Pi certificate.

Only transient function arguments are altered; repository sources are unchanged.
This isolates which trajectory values are consumed, not a physical ODE counterexample.
"""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from hegel_machine.cognition import m4_2_unified_closure_and_pi_lifecycle as mod

OUT = Path(__file__).resolve().parent
original = mod.apply_executable_pi_s_projection


def wrapped(kind):
    def apply(*args, **kwargs):
        traj = kwargs.get("fine_trajectory")
        if traj is not None:
            traj = deepcopy(traj)
            if kind == "nonterminal_window_shift":
                n = max(1, min(5, len(traj["x_slow"]) // 5))
                for idx in range(len(traj["x_slow"]) - n, len(traj["x_slow"]) - 1):
                    traj["x_slow"][idx] += 100.0
            elif kind == "endpoint_fast_shift":
                traj["x_fast"][-1] += 1.0
            kwargs["fine_trajectory"] = traj
        return original(*args, **kwargs)
    return apply


results = {}
for name in ("baseline", "nonterminal_window_shift", "endpoint_fast_shift"):
    with patch.object(mod, "apply_executable_pi_s_projection", wrapped(name)):
        cert = mod.certify_regional_pi_s_closure_with_trajectory_projection(
            OUT / "projection_workspaces" / name, t_bounds=(2.4, 5.2)
        )
    results[name] = cert

b, w, e = (results[n] for n in ("baseline", "nonterminal_window_shift", "endpoint_fast_shift"))
results["dependency_summary"] = {
    "baseline_certified": b["certified"],
    "window_shift_certified": w["certified"],
    "endpoint_shift_certified": e["certified"],
    "window_projection_max_change": max(abs(x["window_projected_y"] - y["window_projected_y"]) for x, y in zip(b["projection_records"], w["projection_records"])),
    "endpoint_defect_unchanged_by_window": b["max_closure_defect"] == w["max_closure_defect"],
    "slaving_defect_unchanged_by_window": b["max_trajectory_slaving_defect"] == w["max_trajectory_slaving_defect"],
    "baseline_slaving_defect": b["max_trajectory_slaving_defect"],
    "endpoint_shift_slaving_defect": e["max_trajectory_slaving_defect"],
    "note": "Dependency isolation, not a new physical ODE counterexample; final state preserved only in nonterminal_window_shift.",
}
(OUT / "projection_dependency_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps(results["dependency_summary"], ensure_ascii=False, indent=2))
