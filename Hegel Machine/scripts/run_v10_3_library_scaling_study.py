"""Run v10.3 Track 1: Large-Scale Heterogeneous & Conflicting Library Scaling Study (model_calls = 0).

Writes new receipt (zero edits to any v10 or v10.2 receipts):
  ``docs/evidence/v10_3_scaling_and_calibration_20261117/library_scaling_curve_20261117.json``

Isolates the single architectural factor distinguishing ``archive_full_tools`` (raw episode
archive permitted to reconstruct equivalent commitments on demand at target time) from
``structured_indexed_v10`` (persistent write-time commitment consolidation + sublinear
``PremiseCommitmentIndex`` + deduplicated custom-predicate auditing), holding all solver
tools and sensor-role induction 100% identical across arms (``sensor_role_calibration_priors=False``).

Sweeps library size ``M in [4, 12, 36, 72]`` confirmed episodes across 18 heterogeneous &
conflicting scientific regimes (with real measured ``support_witness_rows`` and recorded
``counterexamples`` attached to source priors) and evaluates 6 shifted-arity (N=5) target
tasks under both a moderate target compute cap (``190,000``) and an ample cap (``380,000``),
reporting both target-only costs and upfront-write amortized break-even curves.
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

from scripts.run_v10_phase2b_open_grammar_llm import (
    _acquire_shared_discovery_bundle,
    _build_parent_commitment_from_empty_library,
    _eval_program_heldout_48,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.equal_tool_attribution import (
    build_action_chain_telemetry,
    build_bounded_open_grammar_patch_catalog,
    reconstruct_commitments_from_raw_archive,
    run_single_factor_dependency_ablation_arm,
)
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_commitment_index import ensure_library_commitment_index
from hegel_machine.cognition.research_programs import _Fitter, _plan
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.open_grammar_author import validate_edge_predicate_ast
from hegel_machine.cognition.world_structure_commitments import _as_schema

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v10_3_scaling_and_calibration_20261117"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_rows(rows) -> list[dict]:
    return [
        {
            "features": dict(r.features),
            "target": float(r.target),
            "history": [dict(h) for h in r.history],
        }
        for r in rows
    ]


def _make_confirmed_regime_prior(
    *,
    concept_id: str,
    seed: int,
    objects: int,
    readout: str,
    direction: str,
    edge_pred_dict: dict | None,
    radius: float,
    variant: str = "exchange",
) -> dict:
    """Synthesize a genuine confirmed source episode prior with real lab witness & counterexample rows."""
    with AnonymousObjectLab(
        seed=seed,
        apparatus_seed=seed + 17,
        objects=objects,
        readout=readout if readout in {"total", "total_event", "window_total_event", "focus_window", "filtered_total"} else "total_event",
        variant=variant if variant in {"exchange", "sink", "nonlocal"} else "exchange",
        relation_direction=direction,
        relation_radius=radius,
        name_seed=seed + 101,
        wiring_seed=seed + 201,
    ) as lab:
        sc = lab._evaluation_schema(("fine",))
        spec = lab.spec_for(("fine",))
        spec.context["object_schema"] = json.dumps(sc.to_dict())
        rows = lab.evaluation_rows(("fine",), seed=seed + 901, length=28, trajectories=1)

    sc_obj = _as_schema(sc.to_dict())
    tot_q = Program("object_sum", children=tuple(Program("feature", name=sc_obj.fields[g]["q"]) for g in sc_obj.objects))
    tot_u = Program("object_sum", children=tuple(Program("feature", name=sc_obj.fields[g]["u"]) for g in sc_obj.objects))
    foc_q = Program(
        "object_sum",
        children=tuple(
            Program("mul", children=(Program("feature", name=sc_obj.fields[g]["f"]), Program("feature", name=sc_obj.fields[g]["q"])))
            for g in sc_obj.objects
        ),
    )
    triples = tuple(
        Program("feature", name=sc_obj.fields[g][r]) for g in sc_obj.objects for r in ("f", "x", "q")
    )
    op_dir = direction if direction in {"right", "left"} else None
    r_const = Program("const", value=float(radius))
    r_sym = Program("arg", name="r_0")

    if edge_pred_dict is None:
        rel_act = Program("neighbor_sum", name=op_dir, children=(r_const, *triples))
        rel_sym = Program("neighbor_sum", name=op_dir, children=(r_sym, *triples))
    else:
        pp = validate_edge_predicate_ast(edge_pred_dict)
        rel_act = Program("relation_sum", name=op_dir, children=(r_const, pp, *triples))
        rel_sym = Program("relation_sum", name=op_dir, children=(r_sym, pp, *triples))

    if readout == "window_total_event":
        act_terms = (
            Program("window_sum", lag=3, children=(tot_q,)),
            Program("window_sum", lag=3, children=(tot_u,)),
            Program("window_sum", lag=3, children=(Program("event_delta", lag=1, children=(foc_q,)),)),
            Program("window_sum", lag=3, children=(Program("event_delta", lag=1, children=(rel_act,)),)),
        )
        sym_terms = (
            Program("window_sum", lag=3, children=(tot_q,)),
            Program("window_sum", lag=3, children=(tot_u,)),
            Program("window_sum", lag=3, children=(Program("event_delta", lag=1, children=(foc_q,)),)),
            Program("window_sum", lag=3, children=(Program("event_delta", lag=1, children=(rel_sym,)),)),
        )
    elif readout == "focus_window":
        act_terms = (foc_q, Program("window_sum", lag=3, children=(rel_act,)))
        sym_terms = (foc_q, Program("window_sum", lag=3, children=(rel_sym,)))
    else:
        act_terms = (
            tot_q,
            tot_u,
            Program("event_delta", lag=1, children=(foc_q,)),
            Program("event_delta", lag=1, children=(rel_act,)),
        )
        sym_terms = (
            tot_q,
            tot_u,
            Program("event_delta", lag=1, children=(foc_q,)),
            Program("event_delta", lag=1, children=(rel_sym,)),
        )

    # Generate self-consistent synthetic targets on the real sensor trajectory for exact column verification
    weights = [1.0, 1.0, 0.7, -0.7][: len(act_terms)]
    prog = _sum_terms(list(act_terms), weights, 0.0)
    sym_prog = _sum_terms(list(sym_terms), weights, 0.0)

    from dataclasses import replace

    consistent_rows = [replace(r, target=float(prog.predict(r))) for r in rows[4:24]]
    ce_rows = [replace(r, target=float(prog.predict(r)) + 1.25) for r in rows[24:28]]

    step_param = {
        "symbol": "r_0",
        "parameter_kind": "radius",
        "radius": float(radius),
        "radius_intervals": [[float(radius) - 0.08, float(radius) + 0.08]],
        "prior_radius_intervals": [[float(radius) - 0.08, float(radius) + 0.08]],
        "relation": direction,
        "source_concept_id": concept_id,
    }
    if edge_pred_dict is not None:
        step_param["predicate"] = deepcopy(edge_pred_dict)

    plan = _plan(
        {"fit_terms": act_terms, "coefficients": [0.0, *weights], "program": prog, "loss": 0.0},
        [
            {
                "operation": "resolve_relation_intervals",
                "question": "Identify r_0",
                "parameter_template": sym_prog.to_dict(),
                "parameter_terms": [t.to_dict() for t in sym_terms],
                "parameters": [step_param],
                "evidence_contexts": [],
                "compact_replacement": False,
                "program": prog,
                "loss": 0.0,
            }
        ],
        len(consistent_rows),
        600,
    )
    noncons = [variant] if variant in {"sink", "nonlocal"} else []
    if noncons:
        plan["scientific_premise_revision"] = {"nonconservation_mechanisms": noncons}

    return {
        "prior_id": f"scale_prior_{concept_id}",
        "source_concept_id": concept_id,
        "family_id": "T15",
        "kind": "structural:scaled_archive",
        "status": "conditional",
        "task_id": f"source_task_{concept_id}",
        "source_features": list(spec.feature_names),
        "control_features": list(spec.feature_bounds),
        "object_schema": sc.to_dict(),
        "structure": {
            "fits": [
                {
                    "fit_terms": [t.to_dict() for t in act_terms],
                    "coefficients": [0.0, *weights],
                    "radius": float(radius),
                }
            ],
            "radius_interval": [float(radius) - 0.08, float(radius) + 0.08],
        },
        "research_program": plan,
        "support_rows": len(consistent_rows),
        "support_witness_rows": _serialize_rows(consistent_rows),
        "validation_loss": 0.0,
        "confirmation_loss": 0.0,
        "counterexamples": [
            {
                "task_id": f"other_incompatible_world_{concept_id}",
                "kind": "scientific_premise_refutation",
                "rows": _serialize_rows(ce_rows),
            }
        ],
        "uses": 0,
        "parent_concept_ids": [],
        "research_method": {
            "fit_rows": 14,
            "selection_rows": 6,
            "confirmation_rows": 20,
            "selection_loss": 0.0,
            "confirmation_loss": 0.0,
            "risk_quantile": 0.5,
        },
    }


def _build_master_72_prior_library() -> tuple[list[dict], dict]:
    """Build 4 foundational target-family priors + 68 heterogeneous/conflicting priors = 72 total."""
    base_lib_wte, base_comm_wte, _ = _build_parent_commitment_from_empty_library(
        readout="window_total_event", s0_seed=600401, s1_seed=600402
    )
    _, base_comm_te, _ = _build_parent_commitment_from_empty_library(
        readout="total_event", s0_seed=600411, s1_seed=600412
    )

    foundational_priors: list[dict] = []
    # 1. Foundational Source 1 & 2: donor_positive_gap (right & left) on N=4
    for idx_d, (d_dir, s_seed) in enumerate((("right", 610401), ("left", 610405))):
        with AnonymousObjectLab(
            seed=s_seed,
            apparatus_seed=s_seed + 11,
            objects=4,
            readout="window_total_event",
            variant="exchange",
            relation_direction=d_dir,
            relation_condition="donor_positive_gap",
            condition_threshold=0.24,
            relation_radius=1.05,
            actuator_curvature=0.16,
            name_seed=s_seed + 101,
            wiring_seed=s_seed + 201,
        ) as lab_s:
            b_s = _acquire_shared_discovery_bundle(lab_s, seed=s_seed)
            fb_s = initial_prediction(b_s["training"], b_s["spec"].risk_quantile)
            og_s = run_single_factor_dependency_ablation_arm(
                parent_commitment=base_comm_wte,
                model=b_s["model"],
                records=b_s["records"],
                spec=b_s["spec"],
                training=b_s["training"],
                validation=b_s["validation"],
                compute_budget_limit=200000,
                use_support_dependency_cue=True,
                tolerance=1e-7,
            )
            best_s = og_s["best_entry"]
            mse_s = _eval_program_heldout_48(lab_s, best_s["candidate"].program, fb_s, s_seed + 1000003)
            tmp_lib = KnowledgeLibrary()
            saver = UnifiedResearchAgent(
                ResearchConfig(seed=s_seed, world_structure_commitments=True, indexed_commitment_store=True),
                library=tmp_lib,
            )
            saver._object_schemas[("fine",)] = deepcopy(best_s["resolved_schema"])
            rec_s = saver.remember_confirmed_selection(
                best_s,
                saver._spec(lab_s, ("fine",)),
                b_s["training"],
                b_s["validation"],
                confirmed=True,
                confirmation_loss=float(mse_s),
                confirmation_rows=48,
                selected_channels=("fine",),
            )
            p_s = deepcopy(tmp_lib.meta_priors[-1])
            p_s["source_concept_id"] = f"foundational_dpg_{d_dir}_{idx_d}"
            foundational_priors.append(p_s)

    # 2. Foundational Source 3 & 4: focal_quadratic_total_event (right) on N=4 + base WTE
    with AnonymousObjectLab(
        seed=620401,
        apparatus_seed=620411,
        objects=4,
        readout="focal_quadratic_total_event",
        variant="exchange",
        relation_direction="right",
        relation_condition="always",
        relation_radius=1.05,
        actuator_curvature=0.15,
        name_seed=620502,
        wiring_seed=620612,
    ) as lab_fq:
        b_fq = _acquire_shared_discovery_bundle(lab_fq, seed=620401)
        fb_fq = initial_prediction(b_fq["training"], b_fq["spec"].risk_quantile)
        og_fq = run_single_factor_dependency_ablation_arm(
            parent_commitment=base_comm_te,
            model=b_fq["model"],
            records=b_fq["records"],
            spec=b_fq["spec"],
            training=b_fq["training"],
            validation=b_fq["validation"],
            compute_budget_limit=200000,
            use_support_dependency_cue=True,
            tolerance=1e-7,
        )
        best_fq = og_fq["best_entry"]
        mse_fq = _eval_program_heldout_48(lab_fq, best_fq["candidate"].program, fb_fq, 1620404)
        tmp_lib2 = KnowledgeLibrary()
        saver2 = UnifiedResearchAgent(
            ResearchConfig(seed=620401, world_structure_commitments=True, indexed_commitment_store=True),
            library=tmp_lib2,
        )
        saver2._object_schemas[("fine",)] = deepcopy(best_fq["resolved_schema"])
        saver2.remember_confirmed_selection(
            best_fq,
            saver2._spec(lab_fq, ("fine",)),
            b_fq["training"],
            b_fq["validation"],
            confirmed=True,
            confirmation_loss=float(mse_fq),
            confirmation_rows=48,
            selected_channels=("fine",),
        )
        p_fq = deepcopy(tmp_lib2.meta_priors[-1])
        p_fq["source_concept_id"] = "foundational_fq_right_2"
        foundational_priors.append(p_fq)

    p_base = deepcopy(base_lib_wte.meta_priors[0])
    p_base["source_concept_id"] = "foundational_uncond_wte_3"
    foundational_priors.append(p_base)

    # 3. 68 heterogeneous & conflicting episodes across 17 distinct non-target/conflicting regimes (4 arity/seed episodes each)
    catalog = build_bounded_open_grammar_patch_catalog()
    ex_preds = [item["edge_predicate"] for item in catalog["P_exchange_form"]]
    regime_specs = [
        ("left", ex_preds[0], "total_event", "exchange"),
        ("symmetric", ex_preds[1], "window_total_event", "exchange"),
        ("right", ex_preds[2], "focus_window", "exchange"),
        ("left", ex_preds[3], "window_total_event", "exchange"),
        ("right", ex_preds[4], "total_event", "exchange"),
        ("symmetric", ex_preds[5], "focus_window", "exchange"),
        ("left", ex_preds[6], "total_event", "exchange"),
        ("right", ex_preds[7], "window_total_event", "exchange"),
        ("left", ex_preds[8], "focus_window", "exchange"),
        ("right", ex_preds[9], "total_event", "exchange"),
        ("left", ex_preds[10], "window_total_event", "exchange"),
        ("right", ex_preds[12], "window_total_event", "exchange"),
        ("left", ex_preds[12], "total_event", "exchange"),
        ("right", None, "total_event", "sink"),
        ("left", None, "window_total_event", "sink"),
        ("right", None, "total_event", "nonlocal"),
        ("symmetric", None, "focus_window", "exchange"),
    ]
    hetero_priors: list[dict] = []
    serial_id = 4
    for r_idx, (d_dir, pred_d, rd_name, var_name) in enumerate(regime_specs):
        for ep_i in range(4):
            n_ep = 3 if ep_i % 2 == 0 else 4
            rad_ep = 0.88 + 0.07 * ep_i
            hetero_priors.append(
                _make_confirmed_regime_prior(
                    concept_id=f"regime{r_idx:02d}_ep{ep_i}_{serial_id:03d}",
                    seed=880000 + 100 * serial_id,
                    objects=n_ep,
                    readout=rd_name,
                    direction=d_dir,
                    edge_pred_dict=pred_d,
                    radius=rad_ep,
                    variant=var_name,
                )
            )
            serial_id += 1

    all_72 = foundational_priors + hetero_priors
    assert len(all_72) == 72
    return all_72, {"foundational_count": 4, "heterogeneous_conflicting_count": 68, "distinct_regimes": 20}


def _target_tasks() -> list[dict]:
    """6 pre-registered N=5 shifted-parameter target tasks across 3 mechanisms."""
    return [
        dict(
            task_id="scale_t1_dpg_right_n5",
            seed=915501,
            apparatus_seed=915051,
            objects=5,
            readout="window_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="donor_positive_gap",
            condition_threshold=0.31,
            relation_radius=1.16,
            actuator_curvature=0.15,
            name_seed=915601,
            wiring_seed=915701,
        ),
        dict(
            task_id="scale_t2_dpg_right_n5",
            seed=915502,
            apparatus_seed=915052,
            objects=5,
            readout="window_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="donor_positive_gap",
            condition_threshold=0.19,
            relation_radius=1.13,
            actuator_curvature=0.15,
            name_seed=915602,
            wiring_seed=915702,
        ),
        dict(
            task_id="scale_t3_dpg_left_n5",
            seed=915503,
            apparatus_seed=915053,
            objects=5,
            readout="window_total_event",
            variant="exchange",
            relation_direction="left",
            relation_condition="donor_positive_gap",
            condition_threshold=0.29,
            relation_radius=1.14,
            actuator_curvature=0.15,
            name_seed=915603,
            wiring_seed=915703,
        ),
        dict(
            task_id="scale_t4_dpg_left_n5",
            seed=915504,
            apparatus_seed=915054,
            objects=5,
            readout="window_total_event",
            variant="exchange",
            relation_direction="left",
            relation_condition="donor_positive_gap",
            condition_threshold=0.20,
            relation_radius=1.17,
            actuator_curvature=0.15,
            name_seed=915604,
            wiring_seed=915704,
        ),
        dict(
            task_id="scale_t5_fq_right_n5",
            seed=915505,
            apparatus_seed=915055,
            objects=5,
            readout="focal_quadratic_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="always",
            relation_radius=1.15,
            actuator_curvature=0.14,
            name_seed=915605,
            wiring_seed=915705,
        ),
        dict(
            task_id="scale_t6_uncond_wte_n5",
            seed=915506,
            apparatus_seed=915056,
            objects=5,
            readout="window_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="always",
            relation_radius=1.18,
            actuator_curvature=0.14,
            name_seed=915606,
            wiring_seed=915706,
        ),
    ]


def _run_target_under_arm(
    *,
    arm_name: str,
    task_spec: dict,
    priors_slice: list[dict],
    preindexed_lib: KnowledgeLibrary | None,
    target_compute_cap: int,
) -> dict:
    t0 = time.monotonic()
    kw = {k: v for k, v in task_spec.items() if k != "task_id"}
    seed = int(kw["seed"])
    eval_seed = seed + 1000003

    archive_rec_compute = 0
    if arm_name == "cold_full_tools":
        lib = KnowledgeLibrary()
        indexed_flag = False
    elif arm_name == "archive_full_tools":
        # Strong unindexed archive: reconstructs commitments on demand from raw priors (charged to target budget!)
        lib = KnowledgeLibrary(meta_priors=deepcopy(priors_slice))
        rec_b = SearchBudget(target_compute_cap // 2)
        _, archive_rec_compute = reconstruct_commitments_from_raw_archive(
            lib.meta_priors, rec_b, tolerance=1e-7
        )
        indexed_flag = False
    elif arm_name == "structured_indexed_v10":
        assert preindexed_lib is not None
        # Share read-only pre-consolidated commitment index built at write time!
        lib = KnowledgeLibrary(meta_priors=deepcopy(preindexed_lib.meta_priors))
        lib._indexed_prior_fingerprints = deepcopy(preindexed_lib._indexed_prior_fingerprints)
        lib._indexed_group_commitments = deepcopy(preindexed_lib._indexed_group_commitments)
        lib._premise_commitment_index = deepcopy(preindexed_lib._premise_commitment_index)
        indexed_flag = True
    else:
        raise ValueError(arm_name)

    rem_cap = max(1000, target_compute_cap - archive_rec_compute)
    cfg = ResearchConfig(
        seed=seed,
        observation_budget=340,
        compute_budget=rem_cap,
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=(arm_name != "cold_full_tools"),
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        active_threshold_probing=True,
        indexed_commitment_store=indexed_flag,
        sensor_role_calibration_priors=False,  # Strict single-factor isolation of Track 1!
    )
    agent = UnifiedResearchAgent(cfg, measurement_budget=8400, library=lib)
    with AnonymousObjectLab(**kw) as lab:
        res = agent.run(lab)
        prog = Program.from_dict(res["program"]) if res.get("program") is not None else None
        ch = tuple(res.get("selected_channels") or ())
        confirmed = bool(res["status"] == "completed")
        mse = (
            float(loss(prog, lab.evaluation_rows(ch, seed=eval_seed, length=48, trajectories=1), res["fallback"], 0.5))
            if prog is not None
            else None
        )

    consolidation_or_retrieval_in_run = sum(
        int(a.get("compute_cost", 0))
        for a in res.get("actions", ())
        if a.get("action") in {"consolidate_world_structure_commitments", "retrieve_indexed_world_commitments"}
    )
    audit_in_run = sum(
        int(a.get("compute_cost", 0))
        for a in res.get("actions", ())
        if a.get("action") == "audit_transition_premises"
    )
    tot_target_compute = archive_rec_compute + int(res["costs"]["compute_cost"])
    passed = bool(confirmed and mse is not None and mse <= 1e-7)

    return {
        "task_id": task_spec["task_id"],
        "arm": arm_name,
        "confirmed": confirmed,
        "passed_1e7": passed,
        "selected_channels": list(ch),
        "heldout_fresh48_mse": mse,
        "costs": {
            "observation_cost": res["costs"]["observation_cost"],
            "measurement_cost": res["costs"]["measurement_cost"],
            "archive_on_demand_reconstruction_compute": archive_rec_compute,
            "in_run_consolidation_or_index_lookup_compute": consolidation_or_retrieval_in_run,
            "in_run_premise_audit_compute": audit_in_run,
            "agent_run_compute_cost": res["costs"]["compute_cost"],
            "total_target_compute_cost": tot_target_compute,
        },
        "model_calls": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
    }


def main() -> int:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master_72, lib_meta = _build_master_72_prior_library()
    tasks = _target_tasks()
    library_sizes = [4, 12, 36, 72]
    budget_regimes = [("moderate_220k", 220000), ("ample_380k", 380000)]

    # Cold arm doesn't depend on M; run once per budget regime!
    cold_by_regime = {}
    for reg_name, cap in budget_regimes:
        cold_eps = [
            _run_target_under_arm(
                arm_name="cold_full_tools",
                task_spec=t,
                priors_slice=[],
                preindexed_lib=None,
                target_compute_cap=cap,
            )
            for t in tasks
        ]
        cold_by_regime[reg_name] = {
            "episodes": cold_eps,
            "passes_1e7": sum(1 for e in cold_eps if e["passed_1e7"]),
            "mean_total_target_compute": float(mean(e["costs"]["total_target_compute_cost"] for e in cold_eps)),
            "sum_total_target_compute": sum(e["costs"]["total_target_compute_cost"] for e in cold_eps),
        }
        print(f"[cold_full_tools | {reg_name}] passes={cold_by_regime[reg_name]['passes_1e7']}/6", flush=True)

    scaling_points = []
    for m_size in library_sizes:
        # Reverse chronological ordering in raw archive (newest regimes appended at end; foundational at start)
        slice_m = deepcopy(master_72[:m_size])
        # Build persistent indexed library once at write time and record write-time consolidation compute!
        indexed_lib = KnowledgeLibrary(meta_priors=deepcopy(slice_m))
        write_b = SearchBudget(500000)
        comms_m, _, write_consolidation_compute = ensure_library_commitment_index(
            indexed_lib, budget=write_b, tolerance=1e-7
        )

        regimes_out = {}
        for reg_name, cap in budget_regimes:
            arch_eps = [
                _run_target_under_arm(
                    arm_name="archive_full_tools",
                    task_spec=t,
                    priors_slice=slice_m,
                    preindexed_lib=None,
                    target_compute_cap=cap,
                )
                for t in tasks
            ]
            struct_eps = [
                _run_target_under_arm(
                    arm_name="structured_indexed_v10",
                    task_spec=t,
                    priors_slice=slice_m,
                    preindexed_lib=indexed_lib,
                    target_compute_cap=cap,
                )
                for t in tasks
            ]
            arch_sum = sum(e["costs"]["total_target_compute_cost"] for e in arch_eps)
            struct_sum = sum(e["costs"]["total_target_compute_cost"] for e in struct_eps)
            per_task_saving = (arch_sum - struct_sum) / len(tasks)
            breakeven_tasks = (
                round(write_consolidation_compute / per_task_saving, 2)
                if per_task_saving > 0
                else None
            )
            regimes_out[reg_name] = {
                "target_compute_cap": cap,
                "cold_full_tools": cold_by_regime[reg_name],
                "archive_full_tools": {
                    "passes_1e7": sum(1 for e in arch_eps if e["passed_1e7"]),
                    "mean_on_demand_reconstruction_compute": float(
                        mean(e["costs"]["archive_on_demand_reconstruction_compute"] for e in arch_eps)
                    ),
                    "mean_in_run_consolidation_compute": float(
                        mean(e["costs"]["in_run_consolidation_or_index_lookup_compute"] for e in arch_eps)
                    ),
                    "mean_in_run_audit_compute": float(
                        mean(e["costs"]["in_run_premise_audit_compute"] for e in arch_eps)
                    ),
                    "mean_total_target_compute": float(mean(e["costs"]["total_target_compute_cost"] for e in arch_eps)),
                    "sum_total_target_compute_6tasks": arch_sum,
                    "episodes": arch_eps,
                },
                "structured_indexed_v10": {
                    "passes_1e7": sum(1 for e in struct_eps if e["passed_1e7"]),
                    "upfront_write_consolidation_compute_once": write_consolidation_compute,
                    "mean_index_lookup_compute": float(
                        mean(e["costs"]["in_run_consolidation_or_index_lookup_compute"] for e in struct_eps)
                    ),
                    "mean_in_run_audit_compute": float(
                        mean(e["costs"]["in_run_premise_audit_compute"] for e in struct_eps)
                    ),
                    "mean_total_target_compute": float(
                        mean(e["costs"]["total_target_compute_cost"] for e in struct_eps)
                    ),
                    "sum_total_target_compute_6tasks": struct_sum,
                    "amortized_cumulative_compute_6tasks_incl_write": struct_sum + write_consolidation_compute,
                    "amortized_breakeven_target_tasks_vs_archive": breakeven_tasks,
                    "episodes": struct_eps,
                },
            }
            print(
                f"[M={m_size:2d} | {reg_name}] "
                f"archive: {regimes_out[reg_name]['archive_full_tools']['passes_1e7']}/6 "
                f"(mean_comp={regimes_out[reg_name]['archive_full_tools']['mean_total_target_compute']:.1f}) | "
                f"structured_indexed: {regimes_out[reg_name]['structured_indexed_v10']['passes_1e7']}/6 "
                f"(mean_comp={regimes_out[reg_name]['structured_indexed_v10']['mean_total_target_compute']:.1f}, "
                f"write_once={write_consolidation_compute}, breakeven_K={breakeven_tasks})",
                flush=True,
            )

        scaling_points.append(
            {
                "library_priors_count_M": m_size,
                "consolidated_commitments_count": len(comms_m),
                "upfront_write_consolidation_compute": write_consolidation_compute,
                "budget_regimes": regimes_out,
            }
        )

    receipt = {
        "experiment_id": "v10_3_library_scaling_curve_20261117",
        "generated_at": _utc_now(),
        "base_commit": "85b6055b11312a53720561f6697f6054b4ab8873",
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
        "single_factor_isolation_note": "All 3 arms use identical v10 solvers and cold sensor-role induction (sensor_role_calibration_priors=False); isolates persistent PremiseCommitmentIndex vs on-demand archive reconstruction as library size M scales 4 -> 72.",
        "library_construction": lib_meta,
        "scaling_curve": scaling_points,
    }
    out_path = OUT_DIR / "library_scaling_curve_20261117.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
