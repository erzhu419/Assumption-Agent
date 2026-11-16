"""Run v10 Phase 5A controlled study: 2D Periodic Torus, Bounded Neumann/Dirichlet, & General Graph-Laplacian PDEs.

Produces reproducible JSON receipt (`model_calls = 0`, `total_tokens = 0`):
  ``docs/evidence/v10_world_structure_20261116/v10_phase5a_manifold_and_graph_pde_20261116.json``

Evaluates 12 episodes across three generalized position-free spatial manifold PDE families:
  1. Cohort 5A-1 — 2D Periodic Torus ``Z_{Nx} x Z_{Ny}`` Discovery & Cross-Resolution Transfer:
       - ``3x3 (9 sites)`` -> ``4x4 (16 sites, 16! = 2.09e13 permutations, 48 sensors)``
         on ``focal_pde_event`` and ``window_focal_pde`` readouts (4 episodes).
  2. Cohort 5A-2 — Non-Periodic Bounded Domain PDEs & Conservative Boundary Premise Revision:
       - ``N=5`` Reflecting Neumann (``d_n q = 0``) -> ``N=8`` Absorbing Dirichlet (``q|_{dOmega} = 0``)
       - ``N=5`` Absorbing Dirichlet -> ``N=8`` Reflecting Neumann (4 episodes).
  3. Cohort 5A-3 — General Unknown Irregular Graph-Laplacian Reaction-Diffusion PDEs:
       - ``N=6 -> N=9`` ``branched_wheel`` & ``ladder_pendant`` non-regular graphs (4 episodes).

Compared across 3 matched ``UnifiedResearchAgent`` arms:
  * ``flat_baseline_no_continuum`` (``continuum_field_discovery=False, manifold_pde_discovery=False``)
  * ``ring_only_v10_phase4b`` (``continuum_field_discovery=True, manifold_pde_discovery=False``)
  * ``v10_phase5a_manifold_pde_and_commitment`` (``continuum_field_discovery=True, manifold_pde_discovery=True``)
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

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.manifold_pde_lab import (
    ManifoldPDELab,
    consolidate_manifold_pde_commitments,
)
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eval_heldout_48(lab: ManifoldPDELab, res: dict, *, eval_seed: int) -> float | None:
    if not res.get("program"):
        return None
    prog = Program.from_dict(res["program"])
    channels = tuple(res.get("selected_channels") or ())
    rows = lab.evaluation_rows(channels, seed=eval_seed, length=48, trajectories=1)
    return float(loss(prog, rows, res["fallback"], 0.5))


def _run_chain_suite(chains: list[dict], arms: tuple) -> list[dict]:
    episodes = []
    for chain in chains:
        libs = {arm_name: KnowledgeLibrary() for arm_name, _, _, _ in arms}
        for stage_label, sp in (("discovery", chain["source"]), ("transfer_or_revision", chain["target"])):
            ep = {
                "chain_id": chain["chain_id"],
                "cohort": chain["cohort"],
                "stage": stage_label,
                "readout": chain["readout"],
                **{k: (list(v) if isinstance(v, tuple) else v) for k, v in sp.items()},
                "arms": {},
            }
            for arm_name, use_wsc, use_cont, use_man in arms:
                t0 = time.monotonic()
                n_sites = (
                    sp["grid_shape"][0] * sp["grid_shape"][1]
                    if sp["topology"] == "torus_2d"
                    else sp["sites"]
                )
                cfg = ResearchConfig(
                    observation_budget=560 if n_sites >= 16 else 440,
                    compute_budget=1400000 if n_sites >= 16 else 600000,
                    minimum_improvement=1e-7,
                    max_candidates=4,
                    world_structure_commitments=use_wsc,
                    transition_premise_audit=use_wsc,
                    expanded_premise_revisions=use_wsc,
                    continuum_field_discovery=use_cont,
                    manifold_pde_discovery=use_man,
                )
                lab_kwargs = dict(
                    seed=sp["seed"],
                    topology=sp["topology"],
                    readout=chain["readout"],
                    actuator_curvature=0.11,
                    name_seed=sp["seed"] % 997 + 17,
                    wiring_seed=sp["seed"] % 991 + 31,
                )
                if sp["topology"] == "torus_2d":
                    lab_kwargs.update(
                        grid_shape=tuple(sp["grid_shape"]),
                        diffusion_x=sp["dx"],
                        diffusion_y=sp["dy"],
                        advection_x=sp["cx"],
                        advection_y=sp["cy"],
                        reaction_sink=sp["sink"],
                    )
                elif sp["topology"] == "bounded_chain":
                    lab_kwargs.update(
                        sites=sp["sites"],
                        boundary_condition=sp["bc"],
                        diffusion=sp["D"],
                        advection=sp["adv"],
                        reaction_sink=sp["sink"],
                    )
                else:
                    lab_kwargs.update(
                        sites=sp["sites"],
                        graph_family=sp["graph_family"],
                        diffusion=sp["D"],
                        reaction_sink=sp["sink"],
                    )

                lib_cur = libs[arm_name]
                with ManifoldPDELab(**lab_kwargs) as lab_m:
                    res = UnifiedResearchAgent(
                        cfg,
                        measurement_budget=26000 if n_sites >= 16 else 15000,
                        library=lib_cur,
                    ).run(lab_m)
                    mse = _eval_heldout_48(lab_m, res, eval_seed=sp["seed"] + 100000)

                m_comms = (
                    consolidate_manifold_pde_commitments(lib_cur.meta_priors, budget=SearchBudget(40000))
                    if use_man
                    else []
                )
                audit_acts = [
                    a for a in res["actions"] if a.get("action") == "audit_continuum_field_premises"
                ]
                last_audit = audit_acts[-1] if audit_acts else {}
                ep["arms"][arm_name] = {
                    "status": res["status"],
                    "selected_channels": res.get("selected_channels"),
                    "fresh48_mse": mse,
                    "passed_1e7": bool(mse is not None and mse <= 1e-7),
                    "discovered_topology": last_audit.get("topology"),
                    "discovered_boundary_condition": last_audit.get("boundary_condition"),
                    "consolidated_manifold_commitments": len(m_comms),
                    "costs": res["costs"],
                    "compute_cost": res["costs"]["compute_cost"],
                    "wall_seconds": round(time.monotonic() - t0, 3),
                }
            episodes.append(ep)
    return episodes


def main() -> int:
    t_start = time.monotonic()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    arms = (
        ("flat_baseline_no_continuum", False, False, False),
        ("ring_only_v10_phase4b", True, True, False),
        ("v10_phase5a_manifold_pde_and_commitment", True, True, True),
    )

    chains = [
        # Cohort 5A-1: 2D Periodic Torus (3x3 = 9 -> 4x4 = 16 sites, 16! = 2.09e13)
        {
            "chain_id": "torus_3x3_to_4x4_event",
            "cohort": "5A-1_torus_2d",
            "readout": "focal_pde_event",
            "source": {
                "task_id": "500901",
                "topology": "torus_2d",
                "grid_shape": (3, 3),
                "seed": 500901,
                "dx": 0.14,
                "dy": 0.09,
                "cx": 0.08,
                "cy": 0.04,
                "sink": 0.04,
            },
            "target": {
                "task_id": "501601",
                "topology": "torus_2d",
                "grid_shape": (4, 4),
                "seed": 501601,
                "dx": 0.16,
                "dy": 0.10,
                "cx": 0.09,
                "cy": 0.05,
                "sink": 0.03,
            },
        },
        {
            "chain_id": "torus_3x3_to_4x4_window",
            "cohort": "5A-1_torus_2d",
            "readout": "window_focal_pde",
            "source": {
                "task_id": "500902",
                "topology": "torus_2d",
                "grid_shape": (3, 3),
                "seed": 500902,
                "dx": 0.15,
                "dy": 0.08,
                "cx": 0.07,
                "cy": 0.05,
                "sink": 0.05,
            },
            "target": {
                "task_id": "501602",
                "topology": "torus_2d",
                "grid_shape": (4, 4),
                "seed": 501602,
                "dx": 0.13,
                "dy": 0.10,
                "cx": 0.08,
                "cy": 0.04,
                "sink": 0.04,
            },
        },
        # Cohort 5A-2: Non-Periodic Bounded Domain & Boundary Premise Revision (Neumann <-> Dirichlet)
        {
            "chain_id": "chain_n5_neumann_to_n8_dirichlet",
            "cohort": "5A-2_bounded_neumann_dirichlet",
            "readout": "focal_pde_event",
            "source": {
                "task_id": "510501",
                "topology": "bounded_chain",
                "sites": 5,
                "bc": "neumann",
                "seed": 510501,
                "D": 0.13,
                "adv": 0.07,
                "sink": 0.00,
            },
            "target": {
                "task_id": "510801",
                "topology": "bounded_chain",
                "sites": 8,
                "bc": "dirichlet",
                "seed": 510801,
                "D": 0.16,
                "adv": 0.09,
                "sink": 0.03,
            },
        },
        {
            "chain_id": "chain_n5_dirichlet_to_n8_neumann",
            "cohort": "5A-2_bounded_neumann_dirichlet",
            "readout": "window_focal_pde",
            "source": {
                "task_id": "510502",
                "topology": "bounded_chain",
                "sites": 5,
                "bc": "dirichlet",
                "seed": 510502,
                "D": 0.14,
                "adv": 0.08,
                "sink": 0.02,
            },
            "target": {
                "task_id": "510802",
                "topology": "bounded_chain",
                "sites": 8,
                "bc": "neumann",
                "seed": 510802,
                "D": 0.12,
                "adv": 0.06,
                "sink": 0.00,
            },
        },
        # Cohort 5A-3: General Irregular Graph-Laplacian Reaction-Diffusion (N=6 -> N=9)
        {
            "chain_id": "graph_n6_to_n9_branched_wheel",
            "cohort": "5A-3_general_graph_laplacian",
            "readout": "focal_pde_event",
            "source": {
                "task_id": "520601",
                "topology": "general_graph",
                "sites": 6,
                "graph_family": "branched_wheel",
                "seed": 520601,
                "D": 0.14,
                "sink": 0.04,
            },
            "target": {
                "task_id": "520901",
                "topology": "general_graph",
                "sites": 9,
                "graph_family": "branched_wheel",
                "seed": 520901,
                "D": 0.17,
                "sink": 0.02,
            },
        },
        {
            "chain_id": "graph_n6_to_n9_ladder_pendant",
            "cohort": "5A-3_general_graph_laplacian",
            "readout": "window_focal_pde",
            "source": {
                "task_id": "520602",
                "topology": "general_graph",
                "sites": 6,
                "graph_family": "ladder_pendant",
                "seed": 520602,
                "D": 0.15,
                "sink": 0.03,
            },
            "target": {
                "task_id": "520902",
                "topology": "general_graph",
                "sites": 9,
                "graph_family": "ladder_pendant",
                "seed": 520902,
                "D": 0.13,
                "sink": 0.05,
            },
        },
    ]

    out_path = EVIDENCE_DIR / "v10_phase5a_manifold_and_graph_pde_20261116.json"
    if "--refresh-v10" in sys.argv and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        v10_only = _run_chain_suite(chains, (arms[2],))
        prev_by_tid = {ep["task_id"]: ep for ep in prev["episodes"]}
        episodes = []
        for ep_new in v10_only:
            ep_prev = deepcopy(prev_by_tid[ep_new["task_id"]])
            ep_prev["arms"]["v10_phase5a_manifold_pde_and_commitment"] = ep_new["arms"][
                "v10_phase5a_manifold_pde_and_commitment"
            ]
            episodes.append(ep_prev)
        wall_sec = round(
            sum(
                ep["arms"][a_name]["wall_seconds"]
                for ep in episodes
                for a_name, _, _, _ in arms
            ),
            3,
        )
    else:
        episodes = _run_chain_suite(chains, arms)
        wall_sec = round(time.monotonic() - t_start, 3)

    summary_by_arm = {}
    for arm_name, _, _, _ in arms:
        rows = [ep["arms"][arm_name] for ep in episodes]
        mses = [r["fresh48_mse"] for r in rows if r["fresh48_mse"] is not None]
        summary_by_arm[arm_name] = {
            "episodes_total": len(rows),
            "completed_count": sum(1 for r in rows if r["status"] == "completed"),
            "passed_1e7_count": sum(1 for r in rows if r["passed_1e7"]),
            "zero_channel_count": sum(1 for r in rows if r["selected_channels"] == []),
            "mean_fresh48_mse": float(mean(mses)) if mses else None,
            "mean_compute_cost": float(mean(r["compute_cost"] for r in rows)),
            "mean_observation_cost": float(mean(r["costs"]["observation_cost"] for r in rows)),
        }

    receipt = {
        "experiment_id": "v10_phase5a_manifold_and_graph_pde_20261116",
        "generated_at": _utc_now(),
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": wall_sec,
        "summary_by_arm": summary_by_arm,
        "episodes": episodes,
    }
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(summary_by_arm, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
