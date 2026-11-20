"""Read-only controls for M4.1 Part 3, commit 99b1fdfeb.

No source patches. Capture the existing nested solver before its first call,
then call it with transparent instrumentation. One control changes only the
fallback acquisition policy (reuse already acquired pilot; one confirmation,
matching the recall branch). All outputs are outside the git worktree.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.repo / "src"))
    from hegel_machine.cognition import m4_1_unified_discovery_and_renormalization as m
    import scipy.optimize

    captured = {}

    class CapturedFirstSolver(Exception):
        pass

    def capture(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "_solve_pi_s_lab":
            captured["solve"] = frame.f_back.f_locals["_solve_pi_s_lab"]
            captured["lab"] = deepcopy(frame.f_locals["base_lab"])
            captured["pilot"] = deepcopy(frame.f_locals["pilot_pts"])
            captured["confirm"] = deepcopy(frame.f_locals["confirm_pts"])
            captured["forest"] = deepcopy(frame.f_back.f_locals["forest"])
            captured["ops"] = deepcopy(frame.f_back.f_locals["ops"])
            raise CapturedFirstSolver

    sys.setprofile(capture)
    try:
        m.run_executable_pi_s_renormalization_suite(a.out / "capture_only")
    except CapturedFirstSolver:
        pass
    finally:
        sys.setprofile(None)
    assert "solve" in captured

    original_acquire = m.ExecutableFastSlowRenormalizationLab.acquire_channel
    original_evaluate = m.evaluate_isolated_python_model
    original_index = m.HippoRAGWormholeIndex
    original_fine = m.integrate_fast_slow_ode_rk4
    original_projection = m.apply_executable_pi_s_projection
    original_authorize = m.authorize_abstraction_use
    original_least_squares = scipy.optimize.least_squares
    trace = {}
    policy = "original"

    def acquire(self, points, *, channel="coarse_macro", phase="pilot"):
        requested = deepcopy(points)
        if policy == "reuse_pilot_one_confirm":
            if phase == "fallback_extra":
                points = []
            elif phase == "fallback_confirm":
                points = points[:1]
        rows = original_acquire(self, points, channel=channel, phase=phase)
        trace["acquisitions"].append({
            "phase": phase, "channel": channel, "requested_points": requested,
            "actual_points": deepcopy(points), "returned_rows": deepcopy(rows),
            "cumulative_cost": self.paid_measurement_cost,
        })
        return rows

    def evaluate(ws, rel, rows, *, isolated_prefix_rows):
        out = original_evaluate(ws, rel, rows, isolated_prefix_rows=isolated_prefix_rows)
        trace["evaluations"].append({
            "model": rel, "eval_n": len(rows), "prefix_n": len(isolated_prefix_rows),
            "eval_points": [r["features"] for r in rows],
            "prefix_points": [r["features"] for r in isolated_prefix_rows],
            "result": out,
        })
        return out

    class Index(original_index):
        def retrieve(self, *args, **kwargs):
            out = super().retrieve(*args, **kwargs)
            trace["retrievals"].append({
                "signatures": args[0], "task_view": kwargs["task_view"],
                "edge_counts": self.edge_counts,
                "candidates": [{k: c[k] for k in (
                    "node_id", "node_ppr", "ppr_score", "matched_signatures", "reached_only_by_graph_walk"
                )} for c in out["candidates"]],
            })
            return out

    def fine(*args, **kwargs):
        trace["fine_integrations"] += 1
        return original_fine(*args, **kwargs)

    def projection(*args, **kwargs):
        trace["coarse_integrations"] += 1
        trace["projections_with_fine_trajectory"] += int(kwargs.get("fine_trajectory") is not None)
        return original_projection(*args, **kwargs)

    def authorize(*args, **kwargs):
        out = original_authorize(*args, **kwargs)
        trace["authorizations"].append({"request": deepcopy(kwargs), "result": out})
        return out

    def least_squares(*args, **kwargs):
        trace["lens_least_squares_calls"] += 1
        return original_least_squares(*args, **kwargs)

    m.ExecutableFastSlowRenormalizationLab.acquire_channel = acquire
    m.evaluate_isolated_python_model = evaluate
    m.HippoRAGWormholeIndex = Index
    m.integrate_fast_slow_ode_rk4 = fine
    m.apply_executable_pi_s_projection = projection
    m.authorize_abstraction_use = authorize
    scipy.optimize.least_squares = least_squares

    cases = []

    def run(name, *, mode="regional_with_cg", changed_policy="original", heldout=None, confirm=None):
        nonlocal trace, policy
        policy = changed_policy
        trace = {
            "acquisitions": [], "evaluations": [], "retrievals": [], "authorizations": [],
            "fine_integrations": 0, "coarse_integrations": 0,
            "projections_with_fine_trajectory": 0, "lens_least_squares_calls": 0,
        }
        lab = deepcopy(captured["lab"])
        if heldout is not None:
            lab.heldout_points = deepcopy(heldout)
        result = captured["solve"](
            lab, pilot_pts=captured["pilot"],
            confirm_pts=captured["confirm"] if confirm is None else confirm,
            mode=mode, ws_dir=a.out / "workspaces" / name,
        )
        point_ids = []
        for rec in trace["acquisitions"]:
            point_ids.extend((rec["channel"], json.dumps(pt, sort_keys=True)) for pt in rec["actual_points"])
        rec = {
            "name": name, "mode": mode, "policy": policy,
            "heldout_override": heldout, "confirm_override": confirm,
            "result": result,
            "summary": {
                "total_acquired_rows": len(point_ids),
                "unique_channel_point_rows": len(set(point_ids)),
                "duplicate_acquired_rows": len(point_ids) - len(set(point_ids)),
                "isolated_model_fit_calls": len(trace["evaluations"]),
                "lens_least_squares_calls": trace["lens_least_squares_calls"],
                "fine_integrations": trace["fine_integrations"],
                "coarse_integrations": trace["coarse_integrations"],
                "projections_with_fine_trajectory": trace["projections_with_fine_trajectory"],
                "authorization_call_count": len(trace["authorizations"]),
            },
            "trace": trace,
        }
        cases.append(rec)
        (a.out / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        print(json.dumps({"name": name, "result": result, "summary": rec["summary"]}), flush=True)

    try:
        run("original_with_edge")
        run("original_without_edge", mode="regional_no_cg")
        run("without_edge_reuse_pilot_one_confirm", mode="regional_no_cg", changed_policy="reuse_pilot_one_confirm")
        run("outside_confirm_fallback", confirm=[
            {"drive_u": 1.9, "horizon_t": 0.12},
            {"drive_u": 2.6, "horizon_t": 0.12},
        ])
        run("outside_confirm_fallback_near_box", confirm=[
            {"drive_u": 1.9, "horizon_t": 1.2},
            {"drive_u": 2.6, "horizon_t": 1.2},
        ])
        run("outside_heldout_query", heldout=[{"drive_u": 1.5, "horizon_t": 0.12}])
    finally:
        m.ExecutableFastSlowRenormalizationLab.acquire_channel = original_acquire
        m.evaluate_isolated_python_model = original_evaluate
        m.HippoRAGWormholeIndex = original_index
        m.integrate_fast_slow_ode_rk4 = original_fine
        m.apply_executable_pi_s_projection = original_projection
        m.authorize_abstraction_use = original_authorize
        scipy.optimize.least_squares = original_least_squares

    # Independent analytic check of the actual two-dimensional linear ODE and
    # its stated quasistatic Schur reduction; no new research solver is added.
    import numpy as np
    from scipy.linalg import expm
    ode = captured["lab"].ode_params
    red = {k: v for k, v in ode.items() if k not in ("tau_f", "x_fast_init_scale")}
    amat = np.array([[-1.0 / ode["tau_s"], ode["g_sf"]], [-ode["g_fs"] / ode["tau_f"], -1.0 / ode["tau_f"]]])
    bvec = np.array([ode["a_s"] / ode["tau_s"], ode["a_f"] / ode["tau_f"]])
    numeric = []
    for u in (0.4, 1.7, 3.0):
        for t in (0.08, 0.12, 2.2, 4.0, 5.2):
            z0 = np.array([0.0, ode["x_fast_init_scale"] * u])
            zeq = np.linalg.solve(amat, -bvec * u)
            exact = zeq + expm(amat * t) @ (z0 - zeq)
            fine_r = original_fine(u, t, **ode)
            coarse_r = original_projection(u, t, **red, fine_trajectory=fine_r)
            gamma = 1.0 / ode["tau_s"] + ode["g_sf"] * ode["g_fs"]
            drive = ode["a_s"] / ode["tau_s"] + ode["g_sf"] * ode["a_f"]
            coarse_exact_s = drive / gamma * u * (1.0 - math.exp(-gamma * t))
            numeric.append({
                "u": u, "t": t,
                "fine_state_rk4_vs_matrix_exponential_max_abs": float(np.max(np.abs(exact - [fine_r["x_slow"][-1], fine_r["x_fast"][-1]]))),
                "coarse_rk4_vs_reduced_closed_form_abs": abs(coarse_r["x_slow_eff"] - coarse_exact_s),
                "lens_scaled_fine_coarse_defect": 1.15 * abs(fine_r["y_micro"] - coarse_r["y_coarse"]),
                "window_projected_y": coarse_r["window_projected_y"],
            })
    cert = captured["lab"].certify_regional_scale_box()
    record = {
        "commit": "99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c",
        "source_sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        "case_files": [f"{r['name']}.json" for r in cases],
        "cases": [{k: v for k, v in r.items() if k != "trace"} for r in cases],
        "analytic_ode_check": numeric,
        "original_certificate": cert,
        "scoped_api_allows_outside_acquisition": captured["lab"].acquire_channel(
            [{"drive_u": 1.5, "horizon_t": 0.12}], channel="coarse_macro", phase="audit_outside_query"
        ),
    }
    (a.out / "probe_results.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print("wrote", a.out / "probe_results.json", flush=True)


if __name__ == "__main__":
    main()
