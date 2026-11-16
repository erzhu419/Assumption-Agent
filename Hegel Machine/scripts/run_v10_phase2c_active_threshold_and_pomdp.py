"""Run v10 Phase 2C controlled study: active open-grammar threshold probing & POMDP latent-state growth.

Produces reproducible JSON receipt (`model_calls = 0`, `reported_tokens = 0`):
  ``docs/evidence/v10_world_structure_20261116/v10_phase2c_active_threshold_and_pomdp_20261116.json``

Evaluates two controlled experimental cohorts:
1. Cohort 1 — Active Boundary Probing on Sparse & Shifted Open-Grammar Continuous Thresholds:
   Inherits the consolidated ``donor_positive_gap`` commitment (``q_src > 0 and q_src - q_dst > c``,
   discovered on N=4 with ``c_source = 0.25`` in Phase 2B) and evaluates 4 unseen shifted-threshold
   apparatuses (``N=4`` and ``N=5``, ``c* in {0.43, 0.16, 0.52, 0.36}``) under sparse initial
   observations (16 rows) across three matched arms:
     - ``frozen_source_threshold``: retains literal ``c = 0.25`` without recalibration or probing
     - ``passive_midpoint_calibration``: calibrates ``[c_low, c_high]`` passively on initial rows only
     - ``v10_active_threshold_probing``: ``RelationIntervalResearch`` + ``solve_compound_region``
       active boundary bisection probes narrowing ``[c_low, c_high]`` to precision
   Plus end-to-end anonymous-sensor ``N=4 -> N=5`` shifted-threshold ``UnifiedResearchAgent`` validation.

2. Cohort 2 — Partially Observable / Latent State (POMDP) Premise-Graph Growth & Cross-Scale Transfer:
   Evaluates 6 unseen anonymous-sensor POMDP apparatuses across BOTH unobserved internal-memory
   mechanisms (``transition_memory`` where hidden per-object state ``z_{i,t-1}`` couples into physical
   transitions ``after[i]``, and ``focal_memory`` where hidden per-object reservoir ``z_{i,t}`` drives
   focal readout while ``focus`` switches across objects):
     - Stage 2A: ``N=4`` POMDP discovery & ``P_markov_state`` refutation (4 tasks = 2 mechs x 2 seeds)
     - Stage 2B: ``N=4 -> N=5`` consolidated POMDP cross-scale transfer under shifted continuous decay
       poles ``alpha_target != alpha_source`` and shifted radii (2 tasks = 2 mechs)
   Compared across three matched ``UnifiedResearchAgent`` arms:
     - ``flat_baseline_continuous_state``: baseline archive + built-in scalar ``continuous_state``
     - ``markov_v10_phase2a``: v10 Phase 2A without ``P_markov_state`` latent-state growth
     - ``v10_pomdp_premise_growth``: v10 Phase 2C with ``P_markov_state`` audit, transition deflation,
       1D continuous pole optimization, POMDP commitment consolidation, and cross-scale recalibration.
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
from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe,
    plan_competing_object_probe,
)
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
    instantiate_commitment_on_schema,
)

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


def _donor_positive_gap_predicate(threshold: float = 0.25) -> Program:
    q_src = Program("arg", name="edge_source_quantity")
    q_dst = Program("arg", name="edge_target_quantity")
    return Program(
        "and",
        children=(
            Program("lt", children=(Program("const", value=0.0), q_src)),
            Program(
                "lt",
                children=(
                    Program("const", value=float(threshold)),
                    Program("sub", children=(q_src, q_dst)),
                ),
            ),
        ),
    )


def _make_custom_threshold_prior(
    view: ObjectSchema,
    *,
    concept_id: str,
    threshold: float = 0.25,
    radius: float = 0.95,
) -> dict:
    pred_const = _donor_positive_gap_predicate(threshold)
    triples = tuple(
        _feature(view, item, role) for item in view.objects for role in ("f", "x", "q")
    )
    sym_terms = (
        _total(view, "q"),
        _total(view, "u"),
        Program(
            "event_delta",
            lag=1,
            children=(
                Program(
                    "relation_sum",
                    name="right",
                    children=(Program("arg", name="r_0"), pred_const, *triples),
                ),
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
                Program(
                    "relation_sum",
                    name="right",
                    children=(Program("const", value=radius), pred_const, *triples),
                ),
            ),
        ),
    )
    weights = [1.0, 1.0, 0.16]
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
                        "predicate": pred_const.to_dict(),
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
        "family_id": "T11",
        "kind": "world_commitment:joint_event",
        "status": "supported",
        "task_id": f"task:{concept_id}",
        "object_schema": view.to_dict(),
        "research_program": plan,
        "counterexamples": [],
    }


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


def _eval_heldout_48(lab: AnonymousObjectLab, report: dict, eval_seed: int) -> float | None:
    if report.get("program") is None:
        return None
    prog = Program.from_dict(report["program"])
    channels = tuple(report.get("selected_channels") or ())
    rows = lab.evaluation_rows(channels, seed=eval_seed, length=48, trajectories=1)
    return float(loss(prog, rows, report["fallback"], 0.5))


def run_cohort_1_active_threshold_study() -> dict:
    """Run 3-arm controlled study on sparse & shifted continuous open-grammar thresholds."""
    v4 = _schema(4, "src4")
    priors = [
        _make_custom_threshold_prior(v4, concept_id="og_src_1", threshold=0.25, radius=0.91),
        _make_custom_threshold_prior(v4, concept_id="og_src_2", threshold=0.25, radius=0.98),
    ]
    comm = consolidate_world_structure_commitments(priors)[0]

    tasks = [
        {"task_id": "810401", "objects": 4, "seed": 810401, "true_c": 0.43, "true_r": 1.02},
        {"task_id": "810402", "objects": 4, "seed": 810402, "true_c": 0.16, "true_r": 0.96},
        {"task_id": "810501", "objects": 5, "seed": 810501, "true_c": 0.52, "true_r": 1.11},
        {"task_id": "810502", "objects": 5, "seed": 810502, "true_c": 0.36, "true_r": 1.06},
    ]

    task_records = []
    for cfg_t in tasks:
        with UnifiedObjectLab(
            seed=cfg_t["seed"],
            objects=cfg_t["objects"],
            readout="total_event",
            relation_direction="right",
            relation_condition="donor_positive_gap",
            condition_threshold=cfg_t["true_c"],
            relation_radius=cfg_t["true_r"],
            actuator_curvature=0.0,
        ) as lab:
            spec = lab.spec_for(("fine",))
            command_spec = lab.spec_for(())
            schema = lab._schema(("fine",))
            init_batch = lab.acquire(
                lab.random_probe(cfg_t["seed"] + 19, length=16, name="sparse_init"), ("fine",)
            )
            eval_rows = lab.evaluation_rows(
                ("fine",), seed=cfg_t["seed"] + 9001, length=64, trajectories=4
            )

            # Arm 1: frozen_source_threshold (keeps source c = 0.25 without transition recalibration)
            b1 = SearchBudget(80000)
            e1 = instantiate_commitment_on_schema(
                spec,
                init_batch.observations,
                schema,
                comm,
                b1,
                radius_bounds=[cfg_t["true_r"] - 1e-8, cfg_t["true_r"] + 1e-8],
                transitions=(),
                active_threshold_probing=False,
                tolerance=1e-8,
            )[0]
            mse1 = float(loss(e1["candidate"].program, eval_rows, 0.0, 0.5))

            # Arm 2: passive_midpoint_calibration (recalibrates [c_low, c_high] on sparse rows, no active probes)
            b2 = SearchBudget(80000)
            e2 = instantiate_commitment_on_schema(
                spec,
                init_batch.observations,
                schema,
                comm,
                b2,
                radius_bounds=[cfg_t["true_r"] - 1e-8, cfg_t["true_r"] + 1e-8],
                transitions=init_batch.transitions,
                active_threshold_probing=False,
                tolerance=1e-8,
            )[0]
            mse2 = float(loss(e2["candidate"].program, eval_rows, 0.0, 0.5))
            passive_iv = e2["world_commitment_application"]["threshold_interval"]
            passive_width = float(passive_iv[1] - passive_iv[0]) if passive_iv else None

            # Arm 3: v10_active_threshold_probing (RelationIntervalResearch + solve_compound_region bisection)
            b3 = SearchBudget(160000)
            e3 = instantiate_commitment_on_schema(
                spec,
                init_batch.observations,
                schema,
                comm,
                b3,
                radius_bounds=[cfg_t["true_r"] - 1e-8, cfg_t["true_r"] + 1e-8],
                transitions=init_batch.transitions,
                active_threshold_probing=True,
                tolerance=1e-8,
            )[0]
            binding = {
                schema.fields[g][r]: Program("feature", name=schema.controls[g][r])
                for g in schema.objects
                for r in ("q", "x", "u")
            }
            for g in schema.objects:
                idx = int(g.removeprefix("o"))
                fc = Program("feature", name="focus")
                binding[schema.fields[g]["f"]] = Program(
                    "and",
                    children=(
                        Program("ge", children=(fc, Program("const", value=idx - 0.5))),
                        Program("lt", children=(fc, Program("const", value=idx + 0.5))),
                    ),
                )

            research = RelationIntervalResearch(
                e3["candidate"].program,
                [],
                b3,
                structure=e3.get("structure"),
                research_program=e3.get("research_program"),
            )
            paid = list(init_batch.observations)
            experiments = []
            active_rows_bought = 0
            bisection_rounds = 0
            for s_idx in range(15):
                cands = research.alternatives(paid, b3, tolerance=1e-8, experiments=experiments)
                if not cands:
                    break
                research.adopt(cands[0])
                qs = research.questions(cands)
                if not qs:
                    break
                plan = plan_competing_object_probe(
                    command_spec, cands, b3, binding=binding, serial=s_idx, length=3, questions=qs
                )
                if plan["probe"] is None:
                    break
                b_cmd = lab._lab.acquire(plan["probe"], objects=(), scale=1)
                b_fine = lab.acquire(plan["probe"], ("fine",))
                ev = assess_competing_object_probe(
                    plan, list(zip(b_cmd.observations, b_fine.observations)), b3, tolerance=1e-8
                )
                paid.extend(b_fine.observations)
                active_rows_bought += b_fine.observation_cost
                bisection_rounds += 1
                if ev["status"] == "distinguished":
                    experiments.append(tuple(b_fine.observations))

            c_param = next(
                p for p in research.parameters if p.get("parameter_kind") == "predicate_threshold"
            )
            final_iv = c_param["radius_intervals"][0]
            final_width = float(final_iv[1] - final_iv[0])
            final_c = float(research.values[c_param["symbol"]])
            final_prog, _ = research._program(research.values)
            mse3 = float(loss(final_prog, eval_rows, 0.0, 0.5))

            task_records.append(
                {
                    "task_id": cfg_t["task_id"],
                    "objects": cfg_t["objects"],
                    "true_threshold": cfg_t["true_c"],
                    "sparse_initial_rows": len(init_batch.observations),
                    "frozen_source_threshold": {
                        "heldout256_mse": mse1,
                        "passed_1e7": mse1 <= 1e-7,
                        "compute_cost": b1.spent,
                    },
                    "passive_midpoint_calibration": {
                        "threshold_interval": passive_iv,
                        "interval_width": passive_width,
                        "heldout256_mse": mse2,
                        "passed_1e7": mse2 <= 1e-7,
                        "compute_cost": b2.spent,
                    },
                    "v10_active_threshold_probing": {
                        "bisection_rounds": bisection_rounds,
                        "active_rows_purchased": active_rows_bought,
                        "final_threshold_estimate": final_c,
                        "abs_threshold_error": abs(final_c - cfg_t["true_c"]),
                        "final_threshold_interval": final_iv,
                        "final_interval_width": final_width,
                        "width_reduction_factor": (passive_width / max(final_width, 1e-15))
                        if passive_width
                        else None,
                        "heldout256_mse": mse3,
                        "passed_1e7": mse3 <= 1e-7,
                        "compute_cost": b3.spent,
                    },
                }
            )

    # Also verify full anonymous-sensor N=4 -> N=5 UnifiedResearchAgent with shifted threshold
    anon_lib = KnowledgeLibrary()
    anon_lib.meta_priors = deepcopy(priors)
    anon_cfg = ResearchConfig(
        observation_budget=380,
        compute_budget=460000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        active_threshold_probing=True,
    )
    with AnonymousObjectLab(
        seed=810599,
        objects=5,
        readout="total_event",
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=0.44,
        relation_radius=1.08,
        actuator_curvature=0.14,
        name_seed=31,
        wiring_seed=41,
    ) as anon_lab:
        anon_res = UnifiedResearchAgent(anon_cfg, measurement_budget=8200, library=anon_lib).run(
            anon_lab
        )
        anon_mse = _eval_heldout_48(anon_lab, anon_res, eval_seed=910599)

    summary = {}
    for arm_key in (
        "frozen_source_threshold",
        "passive_midpoint_calibration",
        "v10_active_threshold_probing",
    ):
        passes = sum(1 for r in task_records if r[arm_key]["passed_1e7"])
        summary[arm_key] = {
            "passed_count": passes,
            "total_tasks": len(task_records),
            "pass_rate": passes / len(task_records),
            "mean_heldout256_mse": float(mean(r[arm_key]["heldout256_mse"] for r in task_records)),
            "mean_compute_cost": float(mean(r[arm_key]["compute_cost"] for r in task_records)),
        }
    summary["v10_active_threshold_probing"]["mean_abs_threshold_error"] = float(
        mean(r["v10_active_threshold_probing"]["abs_threshold_error"] for r in task_records)
    )
    summary["v10_active_threshold_probing"]["mean_final_interval_width"] = float(
        mean(r["v10_active_threshold_probing"]["final_interval_width"] for r in task_records)
    )
    summary["passive_midpoint_calibration"]["mean_initial_interval_width"] = float(
        mean(r["passive_midpoint_calibration"]["interval_width"] for r in task_records)
    )

    return {
        "tasks": task_records,
        "summary_by_arm": summary,
        "anonymous_n5_end_to_end_verification": {
            "task_id": "810599",
            "objects": 5,
            "true_threshold": 0.44,
            "status": anon_res["status"],
            "selected_channels": anon_res.get("selected_channels"),
            "fresh48_mse": anon_mse,
            "passed_1e7": bool(anon_mse is not None and anon_mse <= 1e-7),
            "costs": anon_res["costs"],
        },
    }


def run_cohort_2_pomdp_study() -> dict:
    """Run 3-arm controlled study on N=4 POMDP discovery & N=5 cross-scale POMDP transfer."""
    v4 = _schema(4, "src4")
    base_transition_priors = [
        _make_markov_right_prior(v4, concept_id="m_te_1", radius=0.91, readout_kind="total_event"),
        _make_markov_right_prior(v4, concept_id="m_te_2", radius=0.98, readout_kind="total_event"),
    ]
    base_focal_priors = [
        _make_markov_right_prior(v4, concept_id="m_foc_1", radius=0.91, readout_kind="focus"),
        _make_markov_right_prior(v4, concept_id="m_foc_2", radius=0.97, readout_kind="focus"),
    ]

    arms = (
        ("flat_baseline_continuous_state", False, False, False, False),
        ("markov_v10_phase2a", True, True, True, False),
        ("v10_pomdp_premise_growth", True, True, True, True),
    )

    stage_2a_specs = [
        {
            "task_id": "820401",
            "mechanism": "transition_memory",
            "objects": 4,
            "seed": 820401,
            "readout": "total_event",
            "latent_decay": 0.64,
            "latent_weight": 0.41,
            "radius": 0.95,
        },
        {
            "task_id": "820402",
            "mechanism": "transition_memory",
            "objects": 4,
            "seed": 820402,
            "readout": "total_event",
            "latent_decay": 0.71,
            "latent_weight": 0.45,
            "radius": 1.01,
        },
        {
            "task_id": "830401",
            "mechanism": "focal_memory",
            "objects": 4,
            "seed": 830401,
            "readout": "focus",
            "latent_decay": 0.63,
            "latent_weight": 0.39,
            "radius": 0.94,
        },
        {
            "task_id": "830402",
            "mechanism": "focal_memory",
            "objects": 4,
            "seed": 830402,
            "readout": "focus",
            "latent_decay": 0.69,
            "latent_weight": 0.44,
            "radius": 0.99,
        },
    ]

    stage_2b_specs = [
        {
            "task_id": "820501",
            "mechanism": "transition_memory",
            "objects": 5,
            "seed": 820501,
            "readout": "total_event",
            "latent_decay": 0.75,
            "latent_weight": 0.47,
            "radius": 1.11,
        },
        {
            "task_id": "830501",
            "mechanism": "focal_memory",
            "objects": 5,
            "seed": 830501,
            "readout": "focus",
            "latent_decay": 0.73,
            "latent_weight": 0.46,
            "radius": 1.09,
        },
    ]

    learned_libs_by_arm_and_mech: dict[tuple[str, str], KnowledgeLibrary] = {}
    for arm_name, _, _, _, _ in arms:
        lib_tm = KnowledgeLibrary()
        lib_tm.meta_priors = deepcopy(base_transition_priors)
        learned_libs_by_arm_and_mech[(arm_name, "transition_memory")] = lib_tm

        lib_fm = KnowledgeLibrary()
        lib_fm.meta_priors = deepcopy(base_focal_priors)
        learned_libs_by_arm_and_mech[(arm_name, "focal_memory")] = lib_fm

    episodes = []
    for stage_tag, spec_list in (("stage_2a_n4_discovery", stage_2a_specs), ("stage_2b_n5_transfer", stage_2b_specs)):
        for sp in spec_list:
            row_out = {
                "stage": stage_tag,
                "task_id": sp["task_id"],
                "mechanism": sp["mechanism"],
                "objects": sp["objects"],
                "latent_decay": sp["latent_decay"],
                "latent_weight": sp["latent_weight"],
                "arms": {},
            }
            for arm_name, use_wsc, use_tpa, use_exp, use_lat in arms:
                t0 = time.monotonic()
                is_n5 = sp["objects"] >= 5
                cfg = ResearchConfig(
                    observation_budget=380 if is_n5 else 320,
                    compute_budget=420000 if is_n5 else 260000,
                    minimum_improvement=1e-7,
                    max_candidates=4,
                    world_structure_commitments=use_wsc,
                    transition_premise_audit=use_tpa,
                    expanded_premise_revisions=use_exp,
                    latent_state_discovery=use_lat,
                )
                lib_cur = learned_libs_by_arm_and_mech[(arm_name, sp["mechanism"])]
                with AnonymousObjectLab(
                    seed=sp["seed"],
                    objects=sp["objects"],
                    readout=sp["readout"],
                    relation_direction="right",
                    latent_mode=sp["mechanism"],
                    latent_decay=sp["latent_decay"],
                    latent_weight=sp["latent_weight"],
                    relation_radius=sp["radius"],
                    actuator_curvature=0.14,
                    name_seed=sp["seed"] % 997 + 11,
                    wiring_seed=sp["seed"] % 991 + 23,
                ) as lab:
                    res = UnifiedResearchAgent(
                        cfg, measurement_budget=8200 if is_n5 else 6800, library=lib_cur
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
            episodes.append(row_out)

    summary_by_arm = {}
    for arm_name, _, _, _, _ in arms:
        arm_eps = [ep["arms"][arm_name] for ep in episodes]
        passes = sum(1 for item in arm_eps if item["passed_1e7"])
        mses = [item["fresh48_mse"] for item in arm_eps if item["fresh48_mse"] is not None]
        summary_by_arm[arm_name] = {
            "passed_count": passes,
            "total_tasks": len(episodes),
            "pass_rate": passes / len(episodes),
            "stage_2a_n4_passed": sum(
                1 for ep in episodes if ep["stage"] == "stage_2a_n4_discovery" and ep["arms"][arm_name]["passed_1e7"]
            ),
            "stage_2b_n5_passed": sum(
                1 for ep in episodes if ep["stage"] == "stage_2b_n5_transfer" and ep["arms"][arm_name]["passed_1e7"]
            ),
            "mean_fresh48_mse": float(mean(mses)) if mses else None,
            "mean_observation_cost": float(mean(item["costs"]["observation_cost"] for item in arm_eps)),
            "mean_compute_cost": float(mean(item["costs"]["compute_cost"] for item in arm_eps)),
        }

    tm_comms = consolidate_world_structure_commitments(
        learned_libs_by_arm_and_mech[("v10_pomdp_premise_growth", "transition_memory")].meta_priors
    )
    fm_comms = consolidate_world_structure_commitments(
        learned_libs_by_arm_and_mech[("v10_pomdp_premise_growth", "focal_memory")].meta_priors
    )

    return {
        "episodes": episodes,
        "summary_by_arm": summary_by_arm,
        "consolidated_pomdp_commitments": {
            "transition_memory": [
                {
                    "commitment_id": c["commitment_id"],
                    "generation": c.get("generation"),
                    "latent_state_mechanism": c.get("latent_state_mechanism"),
                    "source_latent_poles": c.get("source_latent_poles"),
                    "source_arities": c.get("source_arities"),
                }
                for c in tm_comms
                if c.get("latent_state_mechanism") == "transition_memory"
            ],
            "focal_memory": [
                {
                    "commitment_id": c["commitment_id"],
                    "generation": c.get("generation"),
                    "latent_state_mechanism": c.get("latent_state_mechanism"),
                    "source_latent_poles": c.get("source_latent_poles"),
                    "source_arities": c.get("source_arities"),
                }
                for c in fm_comms
                if c.get("latent_state_mechanism") == "focal_memory"
            ],
        },
    }


def main() -> int:
    t_start = time.monotonic()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    cohort_1 = run_cohort_1_active_threshold_study()
    cohort_2 = run_cohort_2_pomdp_study()

    receipt = {
        "study_id": "v10_phase2c_active_threshold_and_pomdp_20261116",
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
        "cohort_1_active_threshold_probing": cohort_1,
        "cohort_2_pomdp_premise_growth": cohort_2,
    }
    out_path = EVIDENCE_DIR / "v10_phase2c_active_threshold_and_pomdp_20261116.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved Phase 2C receipt -> {out_path}")
    print("Cohort 1 summary:", json.dumps(cohort_1["summary_by_arm"], indent=2))
    print("Cohort 1 N=5 e2e:", json.dumps(cohort_1["anonymous_n5_end_to_end_verification"], indent=2))
    print("Cohort 2 summary:", json.dumps(cohort_2["summary_by_arm"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
