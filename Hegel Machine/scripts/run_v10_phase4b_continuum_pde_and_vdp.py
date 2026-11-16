"""Run v10 Phase 4B controlled study: Position-Free Continuum PDE Subgraph Growth & Supercritical Van der Pol Limit-Cycle POMDPs.

Produces reproducible JSON receipt (`model_calls = 0`, `total_tokens = 0`):
  ``docs/evidence/v10_world_structure_20261116/v10_phase4b_continuum_pde_and_vdp_20261116.json``

Evaluates two architectural breakthroughs beyond Phase 3:
  1. Cohort 4B-1 (Breaking ``P_role_partition`` — Anonymous Position-Free Continuum PDE Field Discovery
     & Cross-Resolution Transfer up to ``N = 7`` / ``7! = 5,040`` site permutations):
       - ``N=4`` Discovery (2 tasks: ``focal_pde_event``, ``window_focal_pde``)
       - ``N=4 -> N=6`` Shifted-Parameter Cross-Resolution Transfer (2 tasks, ``720 x 720`` permutations)
       - ``N=5 -> N=7`` Shifted-Parameter Cross-Resolution Transfer (2 tasks, ``5,040 x 5,040 = 25,401,600`` permutations)
     Compared across 3 matched ``UnifiedResearchAgent`` arms:
       * ``flat_baseline_no_continuum``
       * ``fixed_role_partition_v10_phase3`` (``continuum_field_discovery=False``)
       * ``v10_phase4b_continuum_subgraph_and_commitment`` (``continuum_field_discovery=True``)
  2. Cohort 4B-2 (Supercritical Van der Pol Self-Excited Limit-Cycle Hidden Dynamics, ``rho > 1.0``):
       - ``N=4`` Discovery (2 tasks with ``rho = 1.02`` and ``rho = 1.05``, linearly unstable at origin!)
       - ``N=4 -> N=5`` Shifted-Parameter Cross-Scale Transfer (2 tasks with shifted ``(rho*, omega*, beta*, r_0*)``)
     Compared across 3 matched ``UnifiedResearchAgent`` arms:
       * ``flat_baseline_continuous_state``
       * ``linear_ar2_pomdp_v10_phase3a`` (``nonlinear_limit_cycle_discovery=False``)
       * ``v10_phase4b_supercritical_vdp_pomdp`` (``nonlinear_limit_cycle_discovery=True``)
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from scripts.run_v10_phase3a_higher_order_and_coupled_pomdp import (
    _make_markov_right_prior,
    _schema,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.continuum_field_lab import (
    ContinuumFieldLab,
    consolidate_continuum_field_commitments,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
)

EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eval_heldout_48(lab, res: dict, *, eval_seed: int) -> float | None:
    if not res.get("program"):
        return None
    prog = Program.from_dict(res["program"])
    channels = tuple(res.get("selected_channels") or ())
    rows = lab.evaluation_rows(channels, seed=eval_seed, length=48, trajectories=1)
    return float(loss(prog, rows, res["fallback"], 0.5))


def main() -> int:
    t_start = time.monotonic()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cohort 4B-1: Anonymous Position-Free Continuum PDE Field Benchmark
    # ------------------------------------------------------------------
    continuum_arms = (
        ("flat_baseline_no_continuum", False, False),
        ("fixed_role_partition_v10_phase3", True, False),
        ("v10_phase4b_continuum_subgraph_and_commitment", True, True),
    )

    # Source discovery on N=4 & N=5, followed by shifted-PDE transfer on N=6 & N=7
    continuum_chains = [
        {
            "chain_id": "chain_n4_to_n6_event",
            "readout": "focal_pde_event",
            "source": {"task_id": "980401", "sites": 4, "seed": 980401, "D": 0.13, "adv": 0.08, "sink": 0.05},
            "target": {"task_id": "980601", "sites": 6, "seed": 980601, "D": 0.16, "adv": 0.11, "sink": 0.03},
        },
        {
            "chain_id": "chain_n4_to_n6_window",
            "readout": "window_focal_pde",
            "source": {"task_id": "980402", "sites": 4, "seed": 980402, "D": 0.15, "adv": 0.07, "sink": 0.04},
            "target": {"task_id": "980602", "sites": 6, "seed": 980602, "D": 0.12, "adv": 0.10, "sink": 0.06},
        },
        {
            "chain_id": "chain_n5_to_n7_event",
            "readout": "focal_pde_event",
            "source": {"task_id": "980501", "sites": 5, "seed": 980501, "D": 0.14, "adv": 0.08, "sink": 0.05},
            "target": {"task_id": "980701", "sites": 7, "seed": 980701, "D": 0.17, "adv": 0.11, "sink": 0.03},
        },
        {
            "chain_id": "chain_n5_to_n7_window",
            "readout": "window_focal_pde",
            "source": {"task_id": "980502", "sites": 5, "seed": 980502, "D": 0.12, "adv": 0.09, "sink": 0.04},
            "target": {"task_id": "980702", "sites": 7, "seed": 980702, "D": 0.16, "adv": 0.07, "sink": 0.06},
        },
    ]

    cohort1_episodes = []
    for chain in continuum_chains:
        libs_by_arm = {arm_name: KnowledgeLibrary() for arm_name, _, _ in continuum_arms}
        for stage_label, sp in (("discovery", chain["source"]), ("cross_resolution_transfer", chain["target"])):
            ep = {
                "chain_id": chain["chain_id"],
                "stage": stage_label,
                "readout": chain["readout"],
                **sp,
                "arms": {},
            }
            for arm_name, use_wsc, use_cont in continuum_arms:
                t0 = time.monotonic()
                n_sites = sp["sites"]
                cfg = ResearchConfig(
                    observation_budget=420 if n_sites >= 6 else 360,
                    compute_budget=480000 if n_sites >= 6 else 380000,
                    minimum_improvement=1e-7,
                    max_candidates=4,
                    world_structure_commitments=use_wsc,
                    transition_premise_audit=use_wsc,
                    expanded_premise_revisions=use_wsc,
                    continuum_field_discovery=use_cont,
                )
                lib_cur = libs_by_arm[arm_name]
                with ContinuumFieldLab(
                    seed=sp["seed"],
                    sites=n_sites,
                    readout=chain["readout"],
                    diffusion=sp["D"],
                    advection=sp["adv"],
                    reaction_sink=sp["sink"],
                    actuator_curvature=0.11,
                    name_seed=sp["seed"] % 997 + 17,
                    wiring_seed=sp["seed"] % 991 + 31,
                ) as lab_c:
                    res = UnifiedResearchAgent(
                        cfg, measurement_budget=9600 if n_sites >= 6 else 7600, library=lib_cur
                    ).run(lab_c)
                    mse = _eval_heldout_48(lab_c, res, eval_seed=sp["seed"] + 100000)

                c_comms = (
                    consolidate_continuum_field_commitments(lib_cur.meta_priors, budget=SearchBudget(30000))
                    if use_cont
                    else []
                )
                ep["arms"][arm_name] = {
                    "status": res["status"],
                    "selected_channels": res.get("selected_channels"),
                    "fresh48_mse": mse,
                    "passed_1e7": bool(mse is not None and mse <= 1e-7),
                    "consolidated_continuum_commitments": len(c_comms),
                    "costs": res["costs"],
                    "compute_cost": res["costs"]["compute_cost"],
                    "wall_seconds": round(time.monotonic() - t0, 3),
                }
            cohort1_episodes.append(ep)

    # ------------------------------------------------------------------
    # Cohort 4B-2: Supercritical Van der Pol Self-Excited Limit-Cycle POMDPs
    # ------------------------------------------------------------------
    v4 = _schema(4, "src4")
    base_te_priors = [
        _make_markov_right_prior(v4, concept_id="m_te_1", radius=0.91, readout_kind="total_event"),
        _make_markov_right_prior(v4, concept_id="m_te_2", radius=0.98, readout_kind="total_event"),
    ]

    vdp_arms = (
        ("flat_baseline_continuous_state", False, False),
        ("linear_ar2_pomdp_v10_phase3a", True, False),
        ("v10_phase4b_supercritical_vdp_pomdp", True, True),
    )

    vdp_chains = [
        {
            "chain_id": "vdp_chain_1",
            "source": {
                "task_id": "990401",
                "objects": 4,
                "seed": 990401,
                "rho": 1.02,
                "freq": 0.65,
                "beta": 0.28,
                "weight": 0.44,
                "radius": 0.96,
            },
            "target": {
                "task_id": "990501",
                "objects": 5,
                "seed": 990501,
                "rho": 1.04,
                "freq": 0.58,
                "beta": 0.33,
                "weight": 0.48,
                "radius": 1.12,
            },
        },
        {
            "chain_id": "vdp_chain_2",
            "source": {
                "task_id": "990402",
                "objects": 4,
                "seed": 990402,
                "rho": 1.05,
                "freq": 0.78,
                "beta": 0.34,
                "weight": 0.42,
                "radius": 0.94,
            },
            "target": {
                "task_id": "990502",
                "objects": 5,
                "seed": 990502,
                "rho": 1.03,
                "freq": 0.86,
                "beta": 0.26,
                "weight": 0.46,
                "radius": 1.09,
            },
        },
    ]

    cohort2_episodes = []
    for chain in vdp_chains:
        libs_vdp = {}
        for arm_name, use_pomdp, _ in vdp_arms:
            lb = KnowledgeLibrary()
            if use_pomdp:
                lb.meta_priors = deepcopy(base_te_priors)
            libs_vdp[arm_name] = lb

        for stage_label, sp in (("n4_discovery", chain["source"]), ("n4_to_n5_transfer", chain["target"])):
            ep = {
                "chain_id": chain["chain_id"],
                "stage": stage_label,
                **sp,
                "arms": {},
            }
            for arm_name, use_pomdp, use_vdp in vdp_arms:
                t0 = time.monotonic()
                is_n5 = sp["objects"] >= 5
                cfg = ResearchConfig(
                    observation_budget=400 if is_n5 else 360,
                    compute_budget=480000 if is_n5 else 380000,
                    minimum_improvement=1e-7,
                    max_candidates=4,
                    world_structure_commitments=use_pomdp,
                    transition_premise_audit=use_pomdp,
                    expanded_premise_revisions=use_pomdp,
                    latent_state_discovery=use_pomdp,
                    higher_order_latent_discovery=use_pomdp,
                    nonlinear_limit_cycle_discovery=use_vdp,
                )
                lib_cur = libs_vdp[arm_name]
                with AnonymousObjectLab(
                    seed=sp["seed"],
                    objects=sp["objects"],
                    readout="total_event",
                    relation_direction="right",
                    relation_radius=sp["radius"],
                    latent_mode="van_der_pol_transition_memory",
                    latent_decay=sp["rho"],
                    latent_freq=sp["freq"],
                    latent_nonlinearity=sp["beta"],
                    latent_weight=sp["weight"],
                    actuator_curvature=0.13,
                    name_seed=sp["seed"] % 997 + 23,
                    wiring_seed=sp["seed"] % 991 + 47,
                ) as lab_v:
                    res = UnifiedResearchAgent(
                        cfg, measurement_budget=8800 if is_n5 else 7600, library=lib_cur
                    ).run(lab_v)
                    mse = _eval_heldout_48(lab_v, res, eval_seed=sp["seed"] + 100000)

                w_comms = (
                    consolidate_world_structure_commitments(lib_cur.meta_priors, budget=SearchBudget(30000))
                    if use_pomdp
                    else []
                )
                ep["arms"][arm_name] = {
                    "status": res["status"],
                    "selected_channels": res.get("selected_channels"),
                    "fresh48_mse": mse,
                    "passed_1e7": bool(mse is not None and mse <= 1e-7),
                    "vdp_commitments": sum(
                        1 for c in w_comms if c.get("latent_state_mechanism") == "van_der_pol_transition_memory"
                    ),
                    "costs": res["costs"],
                    "compute_cost": res["costs"]["compute_cost"],
                    "wall_seconds": round(time.monotonic() - t0, 3),
                }
            cohort2_episodes.append(ep)

    def _summarize_cohort(eps: list[dict], arm_names: tuple[str, ...]) -> dict:
        out = {}
        for a_name in arm_names:
            rows = [e["arms"][a_name] for e in eps]
            xfer = [e["arms"][a_name] for e in eps if "transfer" in e["stage"]]
            out[a_name] = {
                "overall_pass_count": sum(1 for r in rows if r["passed_1e7"]),
                "overall_total": len(rows),
                "overall_pass_rate": round(sum(1 for r in rows if r["passed_1e7"]) / len(rows), 4),
                "transfer_pass_count": sum(1 for r in xfer if r["passed_1e7"]),
                "transfer_total": len(xfer),
                "transfer_pure_command_count": sum(1 for r in xfer if r["selected_channels"] == []),
                "mean_fresh48_mse": mean(r["fresh48_mse"] for r in rows if r["fresh48_mse"] is not None),
                "mean_compute_cost": round(mean(r["compute_cost"] for r in rows), 1),
                "mean_observation_cost": round(mean(r["costs"]["observation_cost"] for r in rows), 1),
            }
        return out

    payload = {
        "experiment_id": "v10_phase4b_continuum_pde_and_vdp_20261116",
        "generated_at_utc": _utc_now(),
        "total_wall_seconds": round(time.monotonic() - t_start, 3),
        "llm_ledger": {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "summary": {
            "cohort_4b1_continuum_pde_discovery_and_transfer": _summarize_cohort(
                cohort1_episodes, tuple(a[0] for a in continuum_arms)
            ),
            "cohort_4b2_supercritical_vdp_limit_cycle": _summarize_cohort(
                cohort2_episodes, tuple(a[0] for a in vdp_arms)
            ),
        },
        "cohort_4b1_episodes": cohort1_episodes,
        "cohort_4b2_episodes": cohort2_episodes,
    }
    out_path = EVIDENCE_DIR / "v10_phase4b_continuum_pde_and_vdp_20261116.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
