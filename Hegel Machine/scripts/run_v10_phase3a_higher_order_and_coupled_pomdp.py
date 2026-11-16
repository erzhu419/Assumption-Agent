"""Run v10 Phase 3A controlled study: 2nd-order oscillatory (AR(2)) & spatially coupled POMDP discovery.

Produces reproducible JSON receipt (`model_calls = 0`, `total_tokens = 0`):
  ``docs/evidence/v10_world_structure_20261116/v10_phase3a_higher_order_and_coupled_pomdp_20261116.json``

Evaluates 8 unseen anonymous-sensor POMDP apparatuses across 4 mechanisms beyond 1st-order independent reservoirs:
  1. ``oscillatory_transition_memory``: 2nd-order damped oscillator (AR(2) complex-conjugate poles)
     coupled into physical state transitions (``P_latent_order`` refuted on transitions)
  2. ``oscillatory_focal_memory``: 2nd-order damped oscillator (AR(2)) coupled into focal readout
     across switching focus states (``P_latent_order`` refuted on trajectory readout)
  3. ``coupled_transition_memory``: per-object latent state exchanging directed spatial flux across
     neighboring objects inside physical transitions (``P_latent_coupling`` refuted on transitions)
  4. ``coupled_focal_memory``: per-object latent reservoir with directed neighbor latent flux in
     focal readout (``P_latent_coupling`` refuted on trajectory readout)

Stages:
  - Stage 3A-1 (``N=4`` Discovery, 4 tasks = 1 per mechanism)
  - Stage 3A-2 (``N=4 -> N=5`` Cross-Scale Transfer under shifted poles, frequencies, couplings, and radii, 4 tasks)

Compared across three matched single-factor ``UnifiedResearchAgent`` arms:
  - ``flat_baseline_continuous_state``: baseline archive + built-in 1st/2nd-order scalar ``continuous_state``
  - ``first_order_pomdp_v10_phase2c``: full Phase 2C 1st-order independent POMDP (``higher_order_latent_discovery=False``)
  - ``v10_phase3a_higher_order_and_coupled_pomdp``: Phase 3A with ``P_latent_order`` & ``P_latent_coupling`` audit,
    deflation, consolidation, and cross-scale recalibration (``higher_order_latent_discovery=True``).
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

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import consolidate_world_structure_commitments

EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schema(count: int, prefix: str = "obj") -> ObjectSchema:
    objects = tuple(f"{prefix}_{i}" for i in range(count))
    return ObjectSchema(
        objects,
        {item: {role: f"{role}:{item}" for role in ("q", "u", "x", "f")} for item in objects},
        {item: (item,) for item in objects},
        {item: {role: f"ctl:{role}:{item}" for role in ("q", "u", "x")} for item in objects},
    )


def _feature(view: ObjectSchema, item: str, role: str) -> Program:
    return Program("feature", name=view.fields[item][role])


def _total(view: ObjectSchema, role: str) -> Program:
    return Program("object_sum", children=tuple(_feature(view, item, role) for item in view.objects))


def _make_markov_right_prior(
    view: ObjectSchema,
    *,
    concept_id: str,
    radius: float,
    readout_kind: str,
) -> dict:
    triples = tuple(
        _feature(view, item, role) for item in view.objects for role in ("f", "x", "q")
    )
    if readout_kind == "focus":
        foc_q = Program(
            "object_sum",
            children=tuple(
                Program("mul", children=(_feature(view, item, "f"), _feature(view, item, "q")))
                for item in view.objects
            ),
        )
        foc_u = Program(
            "object_sum",
            children=tuple(
                Program("mul", children=(_feature(view, item, "f"), _feature(view, item, "u")))
                for item in view.objects
            ),
        )
        sym_terms = (
            foc_q,
            foc_u,
            Program("neighbor_sum", name="right", children=(Program("arg", name="r_0"), *triples)),
        )
        actual_terms = (
            foc_q,
            foc_u,
            Program("neighbor_sum", name="right", children=(Program("const", value=radius), *triples)),
        )
        weights = [1.0, 1.0, 0.22]
        fam, fam_id = "local_exchange", "T08"
    else:
        sym_terms = (
            _total(view, "q"),
            _total(view, "u"),
            Program(
                "event_delta",
                lag=1,
                children=(
                    Program("neighbor_sum", name="right", children=(Program("arg", name="r_0"), *triples)),
                ),
            ),
        )
        actual_terms = (
            sym_terms[0],
            sym_terms[1],
            Program(
                "event_delta",
                lag=1,
                children=(
                    Program("neighbor_sum", name="right", children=(Program("const", value=radius), *triples)),
                ),
            ),
        )
        weights = [1.0, 1.0, 0.16]
        fam, fam_id = "joint_event", "T11"

    sym_prog = _sum_terms(list(sym_terms), weights, 0.0)
    act_prog = _sum_terms(list(actual_terms), weights, 0.0)
    plan = _plan(
        {
            "fit_terms": actual_terms,
            "coefficients": [0.0, *weights],
            "program": act_prog,
            "loss": 0.0,
        },
        [
            {
                "operation": "resolve_relation_intervals",
                "question": "Identify source radius",
                "parameter_template": sym_prog.to_dict(),
                "parameter_terms": [t.to_dict() for t in sym_terms],
                "parameters": [
                    {
                        "symbol": "r_0",
                        "radius": radius,
                        "radius_intervals": [[radius - 0.05, radius + 0.05]],
                        "prior_radius_intervals": [[radius - 0.05, radius + 0.05]],
                        "relation": "right",
                        "source_concept_id": concept_id,
                    }
                ],
                "evidence_contexts": [],
                "compact_replacement": False,
                "program": act_prog,
                "loss": 0.0,
            }
        ],
        48,
        1200,
    )
    return {
        "prior_id": f"prior:{concept_id}",
        "source_concept_id": concept_id,
        "family_id": fam_id,
        "kind": f"structural:{fam}",
        "status": "supported",
        "task_id": f"task:{concept_id}",
        "object_schema": view.to_dict(),
        "research_program": plan,
        "counterexamples": [],
    }


def _eval_heldout_48(lab: AnonymousObjectLab, res: dict, *, eval_seed: int) -> float | None:
    if not res.get("program"):
        return None
    prog = Program.from_dict(res["program"])
    channels = tuple(res.get("selected_channels") or ())
    rows = lab.evaluation_rows(channels, seed=eval_seed, length=48, trajectories=1)
    return float(loss(prog, rows, res["fallback"], 0.5))


def main() -> int:
    t_start = time.monotonic()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    v4 = _schema(4, "src4")

    base_te_priors = [
        _make_markov_right_prior(v4, concept_id="m_te_1", radius=0.91, readout_kind="total_event"),
        _make_markov_right_prior(v4, concept_id="m_te_2", radius=0.98, readout_kind="total_event"),
    ]
    base_foc_priors = [
        _make_markov_right_prior(v4, concept_id="m_foc_1", radius=0.91, readout_kind="focus"),
        _make_markov_right_prior(v4, concept_id="m_foc_2", radius=0.97, readout_kind="focus"),
    ]

    arms = (
        ("flat_baseline_continuous_state", False, False, False, False, False),
        ("first_order_pomdp_v10_phase2c", True, True, True, True, False),
        ("v10_phase3a_higher_order_and_coupled_pomdp", True, True, True, True, True),
    )

    stage_3a1_n4_specs = [
        {
            "task_id": "910401",
            "mechanism": "oscillatory_transition_memory",
            "objects": 4,
            "seed": 910401,
            "readout": "total_event",
            "latent_decay": 0.68,
            "latent_freq": 0.95,
            "latent_coupling": 0.0,
            "latent_weight": 0.41,
            "radius": 0.95,
        },
        {
            "task_id": "920401",
            "mechanism": "oscillatory_focal_memory",
            "objects": 4,
            "seed": 920401,
            "readout": "focus",
            "latent_decay": 0.68,
            "latent_freq": 0.95,
            "latent_coupling": 0.0,
            "latent_weight": 0.41,
            "radius": 0.95,
        },
        {
            "task_id": "930401",
            "mechanism": "coupled_transition_memory",
            "objects": 4,
            "seed": 930401,
            "readout": "total_event",
            "latent_decay": 0.65,
            "latent_freq": 0.0,
            "latent_coupling": 0.22,
            "latent_weight": 0.39,
            "radius": 0.96,
        },
        {
            "task_id": "940401",
            "mechanism": "coupled_focal_memory",
            "objects": 4,
            "seed": 940401,
            "readout": "focus",
            "latent_decay": 0.65,
            "latent_freq": 0.0,
            "latent_coupling": 0.22,
            "latent_weight": 0.39,
            "radius": 0.96,
        },
    ]

    stage_3a2_n5_specs = [
        {
            "task_id": "910501",
            "mechanism": "oscillatory_transition_memory",
            "objects": 5,
            "seed": 910501,
            "readout": "total_event",
            "latent_decay": 0.76,
            "latent_freq": 1.14,
            "latent_coupling": 0.0,
            "latent_weight": 0.46,
            "radius": 1.11,
        },
        {
            "task_id": "920501",
            "mechanism": "oscillatory_focal_memory",
            "objects": 5,
            "seed": 920501,
            "readout": "focus",
            "latent_decay": 0.76,
            "latent_freq": 1.14,
            "latent_coupling": 0.0,
            "latent_weight": 0.46,
            "radius": 1.11,
        },
        {
            "task_id": "930501",
            "mechanism": "coupled_transition_memory",
            "objects": 5,
            "seed": 930501,
            "readout": "total_event",
            "latent_decay": 0.74,
            "latent_freq": 0.0,
            "latent_coupling": 0.26,
            "latent_weight": 0.45,
            "radius": 1.12,
        },
        {
            "task_id": "940501",
            "mechanism": "coupled_focal_memory",
            "objects": 5,
            "seed": 940501,
            "readout": "focus",
            "latent_decay": 0.74,
            "latent_freq": 0.0,
            "latent_coupling": 0.26,
            "latent_weight": 0.45,
            "radius": 1.12,
        },
    ]

    mechanisms = (
        ("oscillatory_transition_memory", "total_event"),
        ("oscillatory_focal_memory", "focus"),
        ("coupled_transition_memory", "total_event"),
        ("coupled_focal_memory", "focus"),
    )

    learned_libs: dict[tuple[str, str], KnowledgeLibrary] = {}
    for arm_name, *_ in arms:
        for mech_name, r_kind in mechanisms:
            lib = KnowledgeLibrary()
            lib.meta_priors = deepcopy(base_te_priors if r_kind == "total_event" else base_foc_priors)
            learned_libs[(arm_name, mech_name)] = lib

    episodes = []
    for stage_tag, spec_list in (
        ("stage_3a1_n4_discovery", stage_3a1_n4_specs),
        ("stage_3a2_n5_transfer", stage_3a2_n5_specs),
    ):
        for sp in spec_list:
            row_out = {
                "stage": stage_tag,
                "task_id": sp["task_id"],
                "mechanism": sp["mechanism"],
                "objects": sp["objects"],
                "latent_decay": sp["latent_decay"],
                "latent_freq": sp["latent_freq"],
                "latent_coupling": sp["latent_coupling"],
                "latent_weight": sp["latent_weight"],
                "radius": sp["radius"],
                "arms": {},
            }
            for arm_name, use_wsc, use_tpa, use_exp, use_lat, use_ho in arms:
                t0 = time.monotonic()
                is_n5 = sp["objects"] >= 5
                cfg = ResearchConfig(
                    observation_budget=400 if is_n5 else 340,
                    compute_budget=500000 if is_n5 else 340000,
                    minimum_improvement=1e-7,
                    max_candidates=4,
                    world_structure_commitments=use_wsc,
                    transition_premise_audit=use_tpa,
                    expanded_premise_revisions=use_exp,
                    latent_state_discovery=use_lat,
                    higher_order_latent_discovery=use_ho,
                )
                lib_cur = learned_libs[(arm_name, sp["mechanism"])]
                with AnonymousObjectLab(
                    seed=sp["seed"],
                    objects=sp["objects"],
                    readout=sp["readout"],
                    relation_direction="right",
                    latent_mode=sp["mechanism"],
                    latent_decay=sp["latent_decay"],
                    latent_freq=sp["latent_freq"],
                    latent_coupling=sp["latent_coupling"],
                    latent_weight=sp["latent_weight"],
                    relation_radius=sp["radius"],
                    actuator_curvature=0.14,
                    name_seed=sp["seed"] % 997 + 13,
                    wiring_seed=sp["seed"] % 991 + 29,
                ) as lab:
                    res = UnifiedResearchAgent(
                        cfg, measurement_budget=8600 if is_n5 else 7200, library=lib_cur
                    ).run(lab)
                    mse = _eval_heldout_48(lab, res, eval_seed=sp["seed"] + 100000)

                row_out["arms"][arm_name] = {
                    "status": res["status"],
                    "selected_channels": res.get("selected_channels"),
                    "fresh48_mse": mse,
                    "passed_1e7": bool(mse is not None and mse <= 1e-7),
                    "costs": res["costs"],
                    "wall_seconds": round(time.monotonic() - t0, 3),
                }
            print(
                f"[{stage_tag} {sp['mechanism']} N={sp['objects']}] "
                f"flat={row_out['arms']['flat_baseline_continuous_state']['passed_1e7']} | "
                f"1st_order_2c={row_out['arms']['first_order_pomdp_v10_phase2c']['passed_1e7']} | "
                f"phase3a={row_out['arms']['v10_phase3a_higher_order_and_coupled_pomdp']['passed_1e7']} "
                f"(mse={row_out['arms']['v10_phase3a_higher_order_and_coupled_pomdp']['fresh48_mse']})",
                flush=True,
            )
            episodes.append(row_out)

    summary_by_arm = {}
    for arm_name, *_ in arms:
        arm_eps = [ep["arms"][arm_name] for ep in episodes]
        passes = sum(1 for item in arm_eps if item["passed_1e7"])
        mses = [item["fresh48_mse"] for item in arm_eps if item["fresh48_mse"] is not None]
        n4_eps = [ep["arms"][arm_name] for ep in episodes if ep["stage"] == "stage_3a1_n4_discovery"]
        n5_eps = [ep["arms"][arm_name] for ep in episodes if ep["stage"] == "stage_3a2_n5_transfer"]
        summary_by_arm[arm_name] = {
            "passed_count": passes,
            "total_tasks": len(episodes),
            "pass_rate": passes / len(episodes),
            "stage_3a1_n4_passed": sum(1 for item in n4_eps if item["passed_1e7"]),
            "stage_3a2_n5_passed": sum(1 for item in n5_eps if item["passed_1e7"]),
            "mean_fresh48_mse": float(mean(mses)) if mses else None,
            "mean_observation_cost": float(mean(item["costs"]["observation_cost"] for item in arm_eps)),
            "mean_compute_cost": float(mean(item["costs"]["compute_cost"] for item in arm_eps)),
            "mean_n5_compute_cost": float(mean(item["costs"]["compute_cost"] for item in n5_eps)),
        }

    consolidated_summary = {}
    for mech_name, _ in mechanisms:
        comms = consolidate_world_structure_commitments(
            learned_libs[("v10_phase3a_higher_order_and_coupled_pomdp", mech_name)].meta_priors
        )
        consolidated_summary[mech_name] = [
            {
                "commitment_id": c["commitment_id"],
                "generation": c.get("generation"),
                "latent_state_mechanism": c.get("latent_state_mechanism"),
                "source_latent_poles": c.get("source_latent_poles"),
                "source_ar2_params": c.get("source_ar2_params"),
                "source_arities": c.get("source_arities"),
            }
            for c in comms
            if c.get("latent_state_mechanism") == mech_name
        ]

    receipt = {
        "study_id": "v10_phase3a_higher_order_and_coupled_pomdp_20261116",
        "generated_at": _utc_now(),
        "wall_seconds": round(time.monotonic() - t_start, 3),
        "llm_ledger": {
            "model": None,
            "backend": "none",
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "episodes": episodes,
        "summary_by_arm": summary_by_arm,
        "consolidated_phase3a_commitments": consolidated_summary,
    }
    out_path = EVIDENCE_DIR / "v10_phase3a_higher_order_and_coupled_pomdp_20261116.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved Phase 3A receipt -> {out_path}")
    print("Phase 3A summary:", json.dumps(summary_by_arm, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
