"""Run v10 Phase 4A controlled study: Active Probing & Cross-Scale Transfer on Curved Hypersurface Thresholds.

Produces reproducible JSON receipt (`model_calls = 0`, `total_tokens = 0`):
  ``docs/evidence/v10_world_structure_20261116/v10_phase4a_curved_hypersurface_probing_20261116.json``

Evaluates coupled 3-parameter ``(r_0, c_0, c_1)`` nonlinear hypersurface edge thresholds in 4D
``(q_src, q_dst, x_src, x_dst)`` space:
  1. ``distance_attenuated_gap``:
       q_src > 0  and  (q_src - q_dst) - c_1 * (x_dst - x_src)^2 > c_0
  2. ``bilinear_product_barrier``:
       q_src > q_dst  and  (q_src^2 - c_1 * q_src * q_dst) > c_0

Stages:
  - Stage 4A-1 (Active Experimental Design on Curved Hypersurfaces, 8 tasks x 4 matched arms, 64 rows/task):
      * ``frozen_source_hypersurface``
      * ``passive_polytope_centroid``
      * ``axis_aligned_1d_bisection``
      * ``v10_phase4a_coupled_hypersurface_probing``
    Evaluated on polytope diameter ``diam(P)``, parameter L2 error ``||c_hat - c*||_2``, random held-out MSE,
    and near-hypersurface multi-regime OOD boundary stress MSE (threshold ``<= 1e-6``).
  - Stage 4A-2 (Anonymous ``N=4 -> N=5`` Cross-Scale 3-Parameter Shift Transfer via ``UnifiedResearchAgent``,
    4 tasks across both curved hypersurface families with ``selected_channels = []``).
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from math import hypot
from pathlib import Path
from random import Random
from statistics import mean
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.curved_hypersurface_probing import (
    calibrate_hypersurface_on_transitions,
    describe_linear_in_params_hypersurface,
    evaluate_edge_hypersurface_features,
    plan_coupled_hypersurface_probe,
    rebind_hypersurface_constants,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.instrument_models import fit_instrument_model, instrument_binding
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_dependency_audit import extract_necessary_active_transfer_geometry
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
    instantiate_commitment_on_schema,
)

EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _distance_attenuated_predicate(c0: float = 0.14, c1: float = 0.30) -> Program:
    qs = Program("arg", name="edge_source_quantity")
    qd = Program("arg", name="edge_target_quantity")
    xs = Program("arg", name="edge_source_position")
    xd = Program("arg", name="edge_target_position")
    dx = Program("sub", children=(xd, xs))
    dx2 = Program("mul", children=(dx, dx))
    rhs = Program(
        "sub",
        children=(
            Program("sub", children=(qs, qd)),
            Program("mul", children=(Program("const", value=c1), dx2)),
        ),
    )
    return Program(
        "and",
        children=(
            Program("lt", children=(Program("const", value=0.0), qs)),
            Program("lt", children=(Program("const", value=c0), rhs)),
        ),
    )


def _bilinear_barrier_predicate(c0: float = 0.14, c1: float = 0.35) -> Program:
    qs = Program("arg", name="edge_source_quantity")
    qd = Program("arg", name="edge_target_quantity")
    qs2 = Program("mul", children=(qs, qs))
    qsqd = Program("mul", children=(qs, qd))
    rhs = Program(
        "sub",
        children=(
            qs2,
            Program("mul", children=(Program("const", value=c1), qsqd)),
        ),
    )
    return Program(
        "and",
        children=(
            Program("lt", children=(qd, qs)),
            Program("lt", children=(Program("const", value=c0), rhs)),
        ),
    )


def _predicate_for(mechanism: str, c0: float, c1: float) -> Program:
    if mechanism == "distance_attenuated_gap":
        return _distance_attenuated_predicate(c0, c1)
    return _bilinear_barrier_predicate(c0, c1)


def _build_commitment_dict(mechanism: str, src_r: float, src_c0: float, src_c1: float) -> dict:
    pred = _predicate_for(mechanism, src_c0, src_c1)
    return {
        "commitment_id": f"world_commitment:hs_{mechanism[:8]}",
        "kind": "world_structure_commitment",
        "reorganization_type": "equivalence_reorganization",
        "source_concept_ids": [],
        "source_arities": [4],
        "family": "joint_event",
        "family_id": "T11",
        "role_terms": [
            {
                "kind": "all_object_sum",
                "summand": Program("feature", name="object_role:q").to_dict(),
                "roles": ["q"],
            },
            {
                "kind": "all_object_sum",
                "summand": Program("feature", name="object_role:u").to_dict(),
                "roles": ["u"],
            },
            {
                "kind": "program_op",
                "op": "event_delta",
                "name": None,
                "value": None,
                "lag": 1,
                "children": [
                    {
                        "kind": "all_relation_sum",
                        "direction": "right",
                        "symbol": "r_0",
                        "predicate": pred.to_dict(),
                    }
                ],
            },
        ],
        "radius_parameters": [
            {
                "symbol": "r_0",
                "direction": "right",
                "edge_condition": pred.to_dict(),
                "binding": "saved_parameter_symbol",
                "source_values": [src_r],
                "source_intervals": [[src_r - 0.03, src_r + 0.03]],
            }
        ],
        "support_premises": [],
    }


def _adversarial_boundary_probe(
    mechanism: str,
    true_r: float,
    true_c0: float,
    true_c1: float,
    actuator_curvature: float,
    seed: int,
    length: int = 64,
) -> Probe:
    """Construct held-out trajectories spanning extreme psi_1 lever arms near the curved boundary."""
    rng = Random(seed)
    pred_true = _predicate_for(mechanism, true_c0, true_c1)
    desc = describe_linear_in_params_hypersurface(pred_true)
    assert desc is not None

    def _inv_act(q_phys: float) -> float:
        lo, hi = -1.0, 1.0
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if mid + actuator_curvature * (mid ** 3) < q_phys:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    pts = []
    for step in range(length):
        reg = step % 4
        if reg == 0:
            dx = rng.uniform(0.03, 0.20 * true_r)
            qd = rng.uniform(-0.85, 0.25)
        elif reg == 1:
            dx = rng.uniform(0.82 * true_r, 0.99 * true_r)
            qd = rng.uniform(-0.85, 0.55)
        elif reg == 2:
            dx = rng.uniform(0.96 * true_r, 1.04 * true_r)
            qd = rng.uniform(-0.7, 0.2)
        else:
            dx = rng.uniform(0.25 * true_r, 0.85 * true_r)
            qd = rng.uniform(-0.90, -0.35) if mechanism == "bilinear_product_barrier" else rng.uniform(-0.5, 0.4)

        # Solve qs near true hypersurface + signed jitter delta in [-0.03, +0.03]
        jitter = rng.choice((-1.0, 1.0)) * rng.uniform(0.004, 0.035)
        best_qs = rng.uniform(0.25, 0.95)
        best_gap = float("inf")
        for k in range(1, 45):
            qs_try = -0.5 + 1.6 * k / 45.0
            g_ok, phi0, psi_l = evaluate_edge_hypersurface_features(desc, 0.0, qs_try, dx, qd)
            if not g_ok:
                continue
            target_phi = true_c0 + true_c1 * psi_l[0] + jitter
            if abs(phi0 - target_phi) < best_gap:
                best_gap = abs(phi0 - target_phi)
                best_qs = qs_try

        xs = rng.uniform(-1.5, 0.2)
        xd = min(1.98, xs + dx)
        pts.append(
            {
                "q_o0": _inv_act(best_qs),
                "x_o0": xs,
                "u_o0": rng.uniform(-0.2, 0.2),
                "q_o1": _inv_act(qd),
                "x_o1": xd,
                "u_o1": rng.uniform(-0.2, 0.2),
                "q_o2": 0.0,
                "x_o2": -1.98,
                "u_o2": rng.uniform(-0.1, 0.1),
                "q_o3": 0.0,
                "x_o3": -1.96,
                "u_o3": rng.uniform(-0.1, 0.1),
                "focus": float(step % 4),
            }
        )
    return Probe(f"adv_boundary_{seed}", tuple(pts))


def _run_stage4a1_arm(spec: dict, arm_name: str) -> dict:
    t0 = time.monotonic()
    mech = spec["mechanism"]
    src_r, src_c0, src_c1 = 0.86, 0.12, 0.24
    template_pred = _predicate_for(mech, src_c0, src_c1)
    desc = describe_linear_in_params_hypersurface(template_pred)
    assert desc is not None

    lab = UnifiedObjectLab(
        seed=spec["seed"],
        objects=4,
        readout="total_event",
        relation_direction="right",
        relation_condition=mech,
        relation_radius=spec["radius"],
        condition_threshold=spec["c0"],
        condition_curvature=spec["c1"],
        actuator_curvature=0.12,
    )
    schema = lab._schema(("fine",))
    q_fields = [schema.fields[g]["q"] for g in schema.objects]
    u_fields = [schema.fields[g]["u"] for g in schema.objects]
    x_fields = [schema.fields[g]["x"] for g in schema.objects]

    b0 = lab.acquire(lab.random_probe(spec["seed"] + 11, length=24, name="init_0"), ("fine",))
    b1 = lab.acquire(lab.random_probe(spec["seed"] + 23, length=24, name="init_1"), ("fine",))
    batches = [b0, b1]
    probe_Seqs = [b0.observations, b1.observations]
    cmd_batches = [
        [
            (Observation(pt, 0.0), obs)
            for pt, obs in zip(pr.sequence, bt.observations)
        ]
        for pr, bt in (
            (lab.random_probe(spec["seed"] + 11, length=24, name="init_0"), b0),
            (lab.random_probe(spec["seed"] + 23, length=24, name="init_1"), b1),
        )
    ]

    bud = SearchBudget(400000)
    inst_model = fit_instrument_model(
        lab.spec_for(()),
        lab.spec_for(("fine",)),
        [p for cb in cmd_batches for p in cb],
        bud,
    )
    binding = instrument_binding(inst_model)
    q_fns = {}
    for g in schema.objects:
        qf = schema.fields[g]["q"]
        qc = schema.controls[g]["q"]
        if qf in binding:
            prog_q = binding[qf]
            q_fns[g] = lambda val, _p=prog_q, _c=qc: float(_p.predict(Observation({_c: float(val)}, 0.0)))

    def _calibrate_current():
        qr, xr, dr = [], [], []
        for b in batches:
            for tr in b.transitions:
                bef, aft = tr["before"], tr["after"]
                qr.append([float(bef[f]) for f in q_fields])
                xr.append([float(bef[f]) for f in x_fields])
                dr.append(
                    [float(aft[qf]) - float(bef[qf]) - float(bef[uf]) for qf, uf in zip(q_fields, u_fields)]
                )
        ng = extract_necessary_active_transfer_geometry(qr, xr, dr, [tuple(range(4))], bud, 1e-10)
        return calibrate_hypersurface_on_transitions(
            template_pred, qr, xr, dr, q_fields, x_fields, ng, ("right",), budget=bud, eff_tol=1e-10
        )

    active_steps = 16
    if arm_name == "frozen_source_hypersurface":
        batches.append(
            lab.acquire(lab.random_probe(spec["seed"] + 37, length=active_steps, name="extra_passive"), ("fine",))
        )
        cal_final = _calibrate_current()
        est_r = src_r
        est_c0, est_c1 = src_c0, src_c1
        pred_chosen = template_pred
        diam_final = cal_final["hypersurface_polytope"]["diameter"] if cal_final else None
        r_width = (
            cal_final["matches"][0]["radius_interval"][1] - cal_final["matches"][0]["radius_interval"][0]
            if cal_final
            else None
        )
    elif arm_name == "passive_polytope_centroid":
        batches.append(
            lab.acquire(lab.random_probe(spec["seed"] + 37, length=active_steps, name="extra_passive"), ("fine",))
        )
        cal_final = _calibrate_current()
        assert cal_final is not None
        poly = cal_final["hypersurface_polytope"]
        r_iv = cal_final["matches"][0]["radius_interval"]
        est_r = 0.5 * (r_iv[0] + min(r_iv[1], 2.0))
        est_c0, est_c1 = poly["chebyshev_center"]
        pred_chosen = rebind_hypersurface_constants(template_pred, desc, [est_c0, est_c1])
        diam_final = poly["diameter"]
        r_width = r_iv[1] - r_iv[0]
    else:
        probe_mode = "coupled_diametral" if arm_name == "v10_phase4a_coupled_hypersurface_probing" else "axis_aligned_1d"
        for k in range(active_steps):
            st = _calibrate_current()
            if st is None:
                break
            plan = plan_coupled_hypersurface_probe(
                template_pred,
                st["hypersurface_polytope"],
                st["matches"][0]["radius_interval"],
                "right",
                schema.controls,
                lab.spec.feature_bounds,
                q_forward_fns=q_fns,
                mode=probe_mode,
                axis_toggle=k,
                probe_id=f"{arm_name}_{k}",
            )
            if plan is None:
                break
            batches.append(lab.acquire(Probe(plan["probe_id"], (plan["point"],)), ("fine",)))
        cal_final = _calibrate_current()
        assert cal_final is not None
        poly = cal_final["hypersurface_polytope"]
        r_iv = cal_final["matches"][0]["radius_interval"]
        est_r = 0.5 * (r_iv[0] + min(r_iv[1], 2.0))
        est_c0, est_c1 = poly["chebyshev_center"]
        pred_chosen = rebind_hypersurface_constants(template_pred, desc, [est_c0, est_c1])
        diam_final = poly["diameter"]
        r_width = r_iv[1] - r_iv[0]

    # Fit linear readout weights on all 64 paid rows using (est_r, pred_chosen)
    all_train = [obs for b in batches for obs in b.observations]
    comm = _build_commitment_dict(mech, est_r, est_c0, est_c1)
    entries = instantiate_commitment_on_schema(
        lab.spec_for(("fine",)),
        all_train,
        schema,
        comm,
        bud,
        radius_bounds=(est_r, est_r),
        predicate_overrides={"r_0": pred_chosen.to_dict()},
    )
    prog = entries[0]["candidate"].program

    # Held-out random evaluation + held-out multi-regime boundary stress evaluation
    rand_eval = lab.evaluation_rows(("fine",), seed=spec["seed"] + 50000, length=48, trajectories=4)
    rand_mse = float(loss(prog, rand_eval, 0.0, 0.5))

    with UnifiedObjectLab(**lab._settings) as adv_lab:
        adv_rows = [
            obs
            for idx_a in range(2)
            for obs in adv_lab.observe(
                _adversarial_boundary_probe(
                    mech,
                    spec["radius"],
                    spec["c0"],
                    spec["c1"],
                    0.12,
                    spec["seed"] + 90000 + 97 * idx_a,
                    length=64,
                ),
                ("fine",),
            )
        ]
    adv_mse = float(loss(prog, adv_rows, 0.0, 0.5))
    param_l2 = hypot(est_c0 - spec["c0"], est_c1 - spec["c1"])
    rad_err = abs(est_r - spec["radius"])

    return {
        "estimated_params": {"r_0": round(est_r, 6), "c_0": round(est_c0, 6), "c_1": round(est_c1, 6)},
        "polytope_diameter": round(float(diam_final), 6) if diam_final is not None else None,
        "radius_interval_width": round(float(r_width), 6) if r_width is not None else None,
        "param_l2_error": round(param_l2, 6),
        "radius_abs_error": round(rad_err, 6),
        "random_eval_mse": rand_mse,
        "adversarial_boundary_mse": adv_mse,
        "passed_boundary_1e6": bool(adv_mse <= 1e-6),
        "costs": lab.costs,
        "compute_cost": bud.spent,
        "wall_seconds": round(time.monotonic() - t0, 3),
    }


def main() -> int:
    t_start = time.monotonic()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    stage_4a1_specs = [
        {"task_id": "970401", "mechanism": "distance_attenuated_gap", "seed": 970401, "radius": 1.06, "c0": 0.21, "c1": 0.46},
        {"task_id": "970402", "mechanism": "distance_attenuated_gap", "seed": 970402, "radius": 1.14, "c0": 0.28, "c1": 0.38},
        {"task_id": "970403", "mechanism": "distance_attenuated_gap", "seed": 970403, "radius": 0.98, "c0": 0.17, "c1": 0.54},
        {"task_id": "970404", "mechanism": "distance_attenuated_gap", "seed": 970404, "radius": 1.10, "c0": 0.24, "c1": 0.49},
        {"task_id": "970405", "mechanism": "bilinear_product_barrier", "seed": 970405, "radius": 1.08, "c0": 0.19, "c1": 0.62},
        {"task_id": "970406", "mechanism": "bilinear_product_barrier", "seed": 970406, "radius": 1.15, "c0": 0.25, "c1": 0.48},
        {"task_id": "970407", "mechanism": "bilinear_product_barrier", "seed": 970407, "radius": 1.02, "c0": 0.16, "c1": 0.70},
        {"task_id": "970408", "mechanism": "bilinear_product_barrier", "seed": 970408, "radius": 1.12, "c0": 0.22, "c1": 0.56},
    ]
    stage_4a1_arms = (
        "frozen_source_hypersurface",
        "passive_polytope_centroid",
        "axis_aligned_1d_bisection",
        "v10_phase4a_coupled_hypersurface_probing",
    )

    stage_4a1_episodes = []
    for sp in stage_4a1_specs:
        ep = {**sp, "arms": {}}
        for arm in stage_4a1_arms:
            ep["arms"][arm] = _run_stage4a1_arm(sp, arm)
        stage_4a1_episodes.append(ep)

    # Stage 4A-2: End-to-End Anonymous N=4 -> N=5 3-Parameter Shift Transfer via UnifiedResearchAgent
    stage_4a2_specs = [
        {"task_id": "970501", "mechanism": "distance_attenuated_gap", "seed": 970501, "radius": 1.12, "c0": 0.25, "c1": 0.48},
        {"task_id": "970502", "mechanism": "distance_attenuated_gap", "seed": 970502, "radius": 1.04, "c0": 0.19, "c1": 0.53},
        {"task_id": "970503", "mechanism": "bilinear_product_barrier", "seed": 970503, "radius": 1.10, "c0": 0.21, "c1": 0.60},
        {"task_id": "970504", "mechanism": "bilinear_product_barrier", "seed": 970504, "radius": 1.16, "c0": 0.24, "c1": 0.52},
    ]

    # Seed N=4 confirmed priors per curved hypersurface family
    source_libs: dict[str, KnowledgeLibrary] = {}
    for mech in ("distance_attenuated_gap", "bilinear_product_barrier"):
        src_lab = UnifiedObjectLab(
            seed=970011 if mech == "distance_attenuated_gap" else 970022,
            objects=4,
            readout="total_event",
            relation_direction="right",
            relation_condition=mech,
            relation_radius=0.92,
            condition_threshold=0.14,
            condition_curvature=0.32,
            actuator_curvature=0.12,
        )
        sc4 = src_lab._schema(("fine",))
        sb0 = src_lab.acquire(src_lab.random_probe(101, length=32, name="s0"), ("fine",))
        sb1 = src_lab.acquire(src_lab.random_probe(102, length=24, name="s1"), ("fine",))
        lib_m = KnowledgeLibrary()
        ag_m = UnifiedResearchAgent(
            ResearchConfig(observation_budget=380, compute_budget=420000, minimum_improvement=1e-7),
            measurement_budget=8400,
            library=lib_m,
        )
        comm_src = _build_commitment_dict(mech, 0.92, 0.14, 0.32)
        ents = instantiate_commitment_on_schema(
            src_lab.spec_for(("fine",)), sb0.observations, sc4, comm_src, SearchBudget(120000), transitions=sb0.transitions
        )
        ag_m.remember_confirmed_selection(
            ents[0],
            src_lab.spec_for(("fine",)),
            sb0.observations,
            sb1.observations,
            confirmed=True,
            confirmation_loss=0.0,
            confirmation_rows=len(sb1.observations),
            selected_channels=("fine",),
        )
        source_libs[mech] = lib_m

    stage_4a2_episodes = []
    for sp in stage_4a2_specs:
        ep = {**sp, "objects": 5, "arms": {}}
        for arm_name, use_wsc in (("flat_unconsolidated_baseline", False), ("v10_phase4a_coupled_world_commitment", True)):
            t0 = time.monotonic()
            lib_run = deepcopy(source_libs[sp["mechanism"]]) if use_wsc else KnowledgeLibrary()
            cfg = ResearchConfig(
                observation_budget=380,
                compute_budget=420000,
                minimum_improvement=1e-7,
                max_candidates=4,
                world_structure_commitments=use_wsc,
                transition_premise_audit=use_wsc,
                expanded_premise_revisions=use_wsc,
            )
            with AnonymousObjectLab(
                seed=sp["seed"],
                objects=5,
                readout="total_event",
                relation_direction="right",
                relation_condition=sp["mechanism"],
                relation_radius=sp["radius"],
                condition_threshold=sp["c0"],
                condition_curvature=sp["c1"],
                actuator_curvature=0.12,
                name_seed=sp["seed"] % 997 + 19,
                wiring_seed=sp["seed"] % 991 + 41,
            ) as lab5:
                res = UnifiedResearchAgent(cfg, measurement_budget=8400, library=lib_run).run(lab5)
                prog = Program.from_dict(res["program"]) if res.get("program") else None
                ch = tuple(res.get("selected_channels") or ())
                ev_rows = lab5.evaluation_rows(ch, seed=sp["seed"] + 100000, length=48, trajectories=1)
                mse = float(loss(prog, ev_rows, res["fallback"], 0.5)) if prog else None
            ep["arms"][arm_name] = {
                "status": res["status"],
                "selected_channels": res.get("selected_channels"),
                "fresh48_mse": mse,
                "passed_1e7": bool(mse is not None and mse <= 1e-7),
                "costs": res["costs"],
                "compute_cost": res["costs"]["compute_cost"],
                "wall_seconds": round(time.monotonic() - t0, 3),
            }
        stage_4a2_episodes.append(ep)

    summary_4a1 = {}
    for arm in stage_4a1_arms:
        rows_a = [ep["arms"][arm] for ep in stage_4a1_episodes]
        summary_4a1[arm] = {
            "pass_count_boundary_1e6": sum(1 for r in rows_a if r["passed_boundary_1e6"]),
            "total_tasks": len(rows_a),
            "pass_rate": round(sum(1 for r in rows_a if r["passed_boundary_1e6"]) / len(rows_a), 4),
            "mean_polytope_diameter": round(mean(r["polytope_diameter"] for r in rows_a if r["polytope_diameter"] is not None), 6),
            "mean_param_l2_error": round(mean(r["param_l2_error"] for r in rows_a), 6),
            "mean_radius_abs_error": round(mean(r["radius_abs_error"] for r in rows_a), 6),
            "mean_random_eval_mse": mean(r["random_eval_mse"] for r in rows_a),
            "mean_adversarial_boundary_mse": mean(r["adversarial_boundary_mse"] for r in rows_a),
        }

    summary_4a2 = {}
    for arm_name in ("flat_unconsolidated_baseline", "v10_phase4a_coupled_world_commitment"):
        rows_b = [ep["arms"][arm_name] for ep in stage_4a2_episodes]
        summary_4a2[arm_name] = {
            "pass_count_1e7": sum(1 for r in rows_b if r["passed_1e7"]),
            "total_tasks": len(rows_b),
            "pass_rate": round(sum(1 for r in rows_b if r["passed_1e7"]) / len(rows_b), 4),
            "pure_command_channel_count": sum(1 for r in rows_b if r["selected_channels"] == []),
            "mean_fresh48_mse": mean(r["fresh48_mse"] for r in rows_b if r["fresh48_mse"] is not None),
            "mean_compute_cost": round(mean(r["compute_cost"] for r in rows_b), 1),
            "mean_observation_cost": round(mean(r["costs"]["observation_cost"] for r in rows_b), 1),
        }

    payload = {
        "experiment_id": "v10_phase4a_curved_hypersurface_probing_20261116",
        "generated_at_utc": _utc_now(),
        "total_wall_seconds": round(time.monotonic() - t_start, 3),
        "llm_ledger": {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "summary": {
            "stage_4a1_active_boundary_design": summary_4a1,
            "stage_4a2_anonymous_n4_to_n5_transfer": summary_4a2,
        },
        "stage_4a1_episodes": stage_4a1_episodes,
        "stage_4a2_episodes": stage_4a2_episodes,
    }
    out_path = EVIDENCE_DIR / "v10_phase4a_curved_hypersurface_probing_20261116.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
