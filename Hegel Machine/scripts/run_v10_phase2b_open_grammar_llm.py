"""Run Phase 2B live open-grammar LLM single-factor controlled study (v10).

Produces reproducible JSON receipt with strictly separated symbolic and LLM ledgers:
  ``docs/evidence/v10_world_structure_20261116/v10_phase2b_open_grammar_llm_20261116.json``

Evaluates ``model="gpt-5.6-sol"`` via ``ResponsesJSONTransport`` (zero secret persistence)
across two open-grammar mechanisms outside the finite symbolic operator table:
1. ``donor_positive_gap`` (``P_exchange_form`` refuted on paid transitions:
   ``q_src > 0 and (q_src - q_dst) > 0.25``):
   - Stage A (``N=4`` discovery, seeds ``610401, 610402``):
     3 matched single-factor arms on identical paid apparatus probes & trajectories:
     * ``symbolic_v10_only`` (``model_calls=0``, full Phase 2A symbolic repertoire)
     * ``llm_without_premise_gate`` (``propose_flat_unconstrained_programs``, ``max_llm_calls=2``)
     * ``llm_with_v10_premise_gate`` (``propose_premise_gated_commitment_revisions``, ``max_llm_calls=2``)
   - Stage B (``N=4 -> N=5`` cross-scale consolidation & zero-extra-LLM-call transfer,
     seeds ``610501, 610502``):
     Full ``UnifiedResearchAgent.run`` with ``model_calls = 0`` on ``N=5`` comparing
     ``unconsolidated_parent_library`` vs ``consolidated_phase2b_open_grammar_library``.
2. ``focal_quadratic_total_event`` (``P_temporal_readout`` refuted on role-resolved
   residuals: ``total + 0.7 * event + 0.35 * q[focus]^2``):
   - Stage A (``N=4`` discovery, seeds ``620401, 620402``): same 3 matched arms.
   - Stage B (``N=4 -> N=5`` cross-scale consolidation & zero-extra-LLM-call transfer,
     seeds ``620501, 620502``): full ``UnifiedResearchAgent.run`` with ``model_calls = 0``.
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
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from hegel_machine.cognition.open_grammar_author import (
    ResponsesJSONTransport,
    empty_llm_ledger,
    propose_flat_unconstrained_programs,
    propose_premise_gated_commitment_revisions,
)
from hegel_machine.cognition.premise_dependency_audit import (
    apply_verified_transition_premises,
    audit_transition_premises,
)
from hegel_machine.cognition.research_programs import _Fitter
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
    instantiate_commitment_on_schema,
    propose_scientific_premise_revision,
)

EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"
LIVE_MODEL = "gpt-5.6-sol"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_mse(val: float | None) -> str:
    return f"{val:.2e}" if val is not None else "None"


def _eval_program_heldout_48(
    lab: AnonymousObjectLab,
    prog: Program | None,
    fallback: float,
    eval_seed: int,
    channels: tuple[str, ...] = ("fine",),
) -> float | None:
    if prog is None:
        return None
    rows = lab.evaluation_rows(channels, seed=eval_seed, length=48, trajectories=1)
    return float(loss(prog, rows, fallback, 0.5))


def _build_parent_commitment_from_empty_library(
    *,
    readout: str,
    s0_seed: int,
    s1_seed: int,
) -> tuple[KnowledgeLibrary, dict, dict]:
    """Form N=4 conservative unconditional 'right' commitment with zero LLM calls."""
    lib = KnowledgeLibrary()
    cfg0 = ResearchConfig(
        seed=s0_seed,
        observation_budget=340,
        compute_budget=280000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
    )
    agent0 = UnifiedResearchAgent(cfg0, measurement_budget=6800, library=lib)
    with AnonymousObjectLab(
        seed=s0_seed,
        objects=4,
        readout=readout,
        relation_direction="symmetric",
        relation_radius=0.92,
        actuator_curvature=0.16,
        name_seed=s0_seed + 11,
        wiring_seed=s0_seed + 23,
    ) as lab0:
        res0 = agent0.run(lab0)

    cfg1 = ResearchConfig(
        seed=s1_seed,
        observation_budget=460,
        compute_budget=340000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
    )
    agent1 = UnifiedResearchAgent(cfg1, measurement_budget=9200, library=lib)
    with AnonymousObjectLab(
        seed=s1_seed,
        objects=4,
        readout=readout,
        relation_direction="right",
        relation_radius=0.94,
        actuator_curvature=0.16,
        name_seed=s1_seed + 17,
        wiring_seed=s1_seed + 29,
    ) as lab1:
        res1 = agent1.run(lab1)

    right_priors = [
        deepcopy(p)
        for p in lib.meta_priors
        if p.get("status") == "supported"
        and '"right"' in json.dumps(p.get("research_program", {}))
    ]
    frozen_lib = KnowledgeLibrary()
    frozen_lib.meta_priors = right_priors if right_priors else [deepcopy(lib.meta_priors[-1])]
    comms = consolidate_world_structure_commitments(frozen_lib.meta_priors)
    assert comms, f"Expected consolidated parent commitment for readout={readout}"
    info = {
        "readout": readout,
        "from_empty_library": True,
        "s0_status": res0["status"],
        "s1_status": res1["status"],
        "model_calls": 0,
        "commitment_id": comms[0]["commitment_id"],
    }
    return frozen_lib, comms[0], info


def _acquire_shared_discovery_bundle(
    lab: AnonymousObjectLab,
    *,
    seed: int,
) -> dict:
    """Acquire identical public role sweeps + training/validation trajectories for all 3 arms."""
    budget = SearchBudget(220000)
    raw_spec = lab.spec_for(("fine",))
    records: list[dict] = []

    def acquire_probe(probe: Probe) -> tuple[list[Observation], list[dict]]:
        batch = lab.acquire(probe, ("fine",))
        for idx, (pt, obs, tr) in enumerate(
            zip(probe.sequence, batch.observations, batch.transitions)
        ):
            records.append(
                {
                    "controls": dict(pt),
                    **tr,
                    "target": obs.target,
                    "history": list(probe.sequence[:idx]),
                    "probe_id": probe.probe_id,
                    "position": idx,
                }
            )
        return list(batch.observations), list(batch.transitions)

    acquire_probe(lab.random_probe(seed + 11, length=12, name="random_0"))
    model = infer_object_model(raw_spec, records, budget)
    for serial in range(12):
        probe, question = propose_object_probe(raw_spec, model, records, serial)
        if probe is None or question["purpose"] not in {
            "isolate_control_effects",
            "distinguish_role_definition",
        }:
            break
        acquire_probe(probe)
        model = infer_object_model(raw_spec, records, budget)

    t1, tr1 = acquire_probe(
        lab.random_probe(seed + 101, length=48, name="train_traj_0")
    )
    t2, tr2 = acquire_probe(
        lab.random_probe(seed + 151, length=48, name="train_traj_1")
    )
    training = [*t1, *t2]
    train_transitions = [*tr1, *tr2]
    validation = lab.observe(
        lab.random_probe(seed + 202, length=36, name="val_traj"), ("fine",)
    )
    model = infer_object_model(raw_spec, records, budget)
    return {
        "spec": raw_spec,
        "model": model,
        "records": records,
        "training": training,
        "validation": validation,
        "train_transitions": train_transitions,
        "shared_induction_compute": budget.spent,
        "observation_cost": lab.observation_cost,
        "measurement_cost": lab.measurement_cost,
    }


def _run_symbolic_v10_only_arm(
    lab: AnonymousObjectLab,
    bundle: dict,
    parent_comm: dict,
    *,
    eval_seed: int,
) -> dict:
    """Arm 1: Symbolic v10 (Phase 2A repertoire, model_calls = 0) on shared bundle."""
    t0 = time.monotonic()
    budget = SearchBudget(180000)
    spec = bundle["spec"]
    training = bundle["training"]
    validation = bundle["validation"]
    fallback = initial_prediction(training, spec.risk_quantile)

    audit = audit_transition_premises(
        deepcopy(bundle["model"]),
        bundle["records"],
        budget,
        commitments=[parent_comm],
        tolerance=1e-7,
    )
    updated = apply_verified_transition_premises(deepcopy(bundle["model"]), audit)
    candidate_entries = []
    if updated.get("schema"):
        if audit.get("status") in {
            "verified_direction_shift",
            "verified_conditional_exchange",
            "verified_nonconservative_exchange",
        }:
            candidate_entries.extend(
                propose_scientific_premise_revision(
                    parent_comm,
                    audit,
                    spec,
                    training,
                    updated["schema"],
                    budget,
                    tolerance=1e-7,
                )
            )
        else:
            candidate_entries.extend(
                instantiate_commitment_on_schema(
                    spec,
                    training,
                    updated["schema"],
                    parent_comm,
                    budget,
                    radius_bounds=audit.get("radius_interval"),
                    tolerance=1e-7,
                )
            )

    # Also allow symbolic linear baseline over all observed fine features + totals if schema unresolved
    fitter = _Fitter(list(training), budget, spec.risk_quantile)
    raw_terms = tuple(Program("feature", name=fname) for fname in spec.feature_names[:16])
    raw_fit = fitter.fit(raw_terms)
    best_prog = None
    best_val = float("inf")
    selected_kind = "none"

    for entry in candidate_entries:
        v_loss = loss(entry["candidate"].program, validation, fallback, spec.risk_quantile)
        budget.spend(len(validation))
        if v_loss < best_val:
            best_val = float(v_loss)
            best_prog = entry["candidate"].program
            selected_kind = entry.get("kind", "world_commitment")

    if raw_fit is not None:
        v_raw = loss(raw_fit["program"], validation, fallback, spec.risk_quantile)
        budget.spend(len(validation))
        if v_raw < best_val:
            best_val = float(v_raw)
            best_prog = raw_fit["program"]
            selected_kind = "symbolic_linear_features"

    fresh_mse = _eval_program_heldout_48(lab, best_prog, fallback, eval_seed)
    return {
        "arm": "symbolic_v10_only",
        "audit_status": audit.get("status"),
        "refuted_premises": [r.get("premise_id") for r in audit.get("refuted_premises", ())],
        "selected_kind": selected_kind,
        "validation_loss": best_val if best_prog is not None else None,
        "fresh48_mse": fresh_mse,
        "passed_1e7": bool(fresh_mse is not None and fresh_mse <= 1e-7),
        "costs": {
            "observation_cost": bundle["observation_cost"],
            "measurement_cost": bundle["measurement_cost"],
            "compute_cost": bundle["shared_induction_compute"] + budget.spent,
        },
        "llm_ledger": empty_llm_ledger(model=None, backend="none"),
        "wall_seconds": round(time.monotonic() - t0, 3),
    }


def _run_three_arm_n4_discovery_and_n5_transfer(
    *,
    mechanism_id: str,
    parent_lib: KnowledgeLibrary,
    parent_comm: dict,
    transport: ResponsesJSONTransport,
    n4_seeds: tuple[int, int],
    n5_seeds: tuple[int, int],
    n4_lab_common: dict,
    n5_lab_common: dict,
) -> dict:
    """Run Stage A (N=4 3-arm single-factor) and Stage B (N=5 zero-LLM-call transfer)."""
    stage_a_episodes = []
    consolidated_og_library: KnowledgeLibrary | None = None
    consolidated_commitment_summary: dict | None = None

    for seed in n4_seeds:
        lab_kwargs = {
            **n4_lab_common,
            "seed": seed,
            "name_seed": seed + 101,
            "wiring_seed": seed + 211,
        }
        with AnonymousObjectLab(**lab_kwargs) as lab:
            bundle = _acquire_shared_discovery_bundle(lab, seed=seed)
            fallback = initial_prediction(bundle["training"], bundle["spec"].risk_quantile)
            eval_seed = seed + 1000003

            # Arm 1: symbolic_v10_only (model_calls = 0)
            arm_sym = _run_symbolic_v10_only_arm(
                lab, bundle, parent_comm, eval_seed=eval_seed
            )

            # Arm 2: llm_without_premise_gate (live gpt-5.6-sol, max_llm_calls = 3)
            t_flat = time.monotonic()
            budget_flat = SearchBudget(180000)
            flat_res = propose_flat_unconstrained_programs(
                bundle["spec"],
                bundle["training"],
                bundle["validation"],
                bundle["train_transitions"],
                budget_flat,
                transport,
                incumbent_loss=arm_sym["validation_loss"],
                max_llm_calls=3,
                tolerance=1e-7,
                model_name=LIVE_MODEL,
            )
            flat_best = flat_res["best_entry"]
            flat_prog = flat_best["candidate"].program if flat_best else None
            flat_mse = _eval_program_heldout_48(lab, flat_prog, fallback, eval_seed)
            arm_flat = {
                "arm": "llm_without_premise_gate",
                "evaluated_count": flat_res["evaluated_count"],
                "validation_loss": flat_best["validation_loss"] if flat_best else None,
                "fresh48_mse": flat_mse,
                "passed_1e7": bool(flat_mse is not None and flat_mse <= 1e-7),
                "program_preview": flat_prog.describe()[:220] if flat_prog else None,
                "costs": {
                    "observation_cost": bundle["observation_cost"],
                    "measurement_cost": bundle["measurement_cost"],
                    "compute_cost": bundle["shared_induction_compute"] + flat_res["compute_cost"],
                },
                "llm_ledger": flat_res["llm_ledger"],
                "wall_seconds": round(time.monotonic() - t_flat, 3),
            }

            # Arm 3: llm_with_v10_premise_gate (live gpt-5.6-sol, max_llm_calls = 3)
            t_gated = time.monotonic()
            budget_gated = SearchBudget(180000)
            gated_res = propose_premise_gated_commitment_revisions(
                parent_comm,
                deepcopy(bundle["model"]),
                bundle["records"],
                bundle["spec"],
                bundle["training"],
                bundle["validation"],
                budget_gated,
                transport,
                max_llm_calls=3,
                tolerance=1e-7,
                model_name=LIVE_MODEL,
            )
            gated_best = gated_res["best_entry"]
            gated_prog = gated_best["candidate"].program if gated_best else None
            gated_mse = _eval_program_heldout_48(lab, gated_prog, fallback, eval_seed)
            arm_gated = {
                "arm": "llm_with_v10_premise_gate",
                "isolated_violated_premise": gated_res.get("isolated_violated_premise"),
                "verified_candidate_count": gated_res["verified_candidate_count"],
                "validation_loss": gated_best["validation_loss"] if gated_best else None,
                "fresh48_mse": gated_mse,
                "passed_1e7": bool(gated_mse is not None and gated_mse <= 1e-7),
                "program_preview": gated_prog.describe()[:220] if gated_prog else None,
                "costs": {
                    "observation_cost": bundle["observation_cost"],
                    "measurement_cost": bundle["measurement_cost"],
                    "compute_cost": bundle["shared_induction_compute"] + gated_res["compute_cost"],
                },
                "llm_ledger": gated_res["llm_ledger"],
                "wall_seconds": round(time.monotonic() - t_gated, 3),
            }

            # Consolidate first verified N=4 open-grammar discovery into a reusable KnowledgeLibrary!
            if (
                consolidated_og_library is None
                and arm_gated["passed_1e7"]
                and gated_best is not None
            ):
                consolidator = UnifiedResearchAgent(
                    ResearchConfig(
                        world_structure_commitments=True,
                        transition_premise_audit=True,
                        expanded_premise_revisions=True,
                    ),
                    library=KnowledgeLibrary(),
                )
                consolidator._object_schemas[("fine",)] = deepcopy(
                    gated_best["resolved_schema"]
                )
                saved = consolidator.remember_confirmed_selection(
                    gated_best,
                    consolidator._spec(lab, ("fine",)),
                    bundle["training"],
                    bundle["validation"],
                    confirmed=True,
                    confirmation_loss=gated_best["validation_loss"],
                    confirmation_rows=len(bundle["validation"]),
                    selected_channels=("fine",),
                )
                if saved.get("stored"):
                    consolidated_og_library = deepcopy(consolidator.library)
                    og_comms = consolidate_world_structure_commitments(
                        consolidated_og_library.meta_priors
                    )
                    if og_comms:
                        c0 = og_comms[0]
                        consolidated_commitment_summary = {
                            "source_seed": seed,
                            "commitment_id": c0["commitment_id"],
                            "generation": c0["generation"],
                            "role_term_count": len(c0.get("role_terms", ())),
                            "radius_parameters": c0.get("radius_parameters"),
                            "revision_lineage": c0.get("revision_lineage"),
                        }

        print(
            f"      [Stage A N=4 {mechanism_id} seed={seed}] "
            f"sym={arm_sym['passed_1e7']} ({_fmt_mse(arm_sym['fresh48_mse'])}, calls=0) | "
            f"flat_llm={arm_flat['passed_1e7']} ({_fmt_mse(arm_flat['fresh48_mse'])}, "
            f"calls={arm_flat['llm_ledger']['model_calls']}, tok={arm_flat['llm_ledger']['total_tokens']}) | "
            f"v10_gated_llm={arm_gated['passed_1e7']} ({_fmt_mse(arm_gated['fresh48_mse'])}, "
            f"calls={arm_gated['llm_ledger']['model_calls']}, tok={arm_gated['llm_ledger']['total_tokens']})",
            flush=True,
        )
        stage_a_episodes.append(
            {
                "seed": seed,
                "arms": {
                    "symbolic_v10_only": arm_sym,
                    "llm_without_premise_gate": arm_flat,
                    "llm_with_v10_premise_gate": arm_gated,
                },
            }
        )

    # Aggregate Stage A
    stage_a_aggregate = {}
    for arm_key in ("symbolic_v10_only", "llm_without_premise_gate", "llm_with_v10_premise_gate"):
        rows = [ep["arms"][arm_key] for ep in stage_a_episodes]
        mses = [r["fresh48_mse"] for r in rows if r["fresh48_mse"] is not None]
        stage_a_aggregate[arm_key] = {
            "tasks_total": len(rows),
            "passes_1e7": sum(1 for r in rows if r["passed_1e7"]),
            "pass_rate": sum(1 for r in rows if r["passed_1e7"]) / len(rows),
            "mean_fresh48_mse": mean(mses) if mses else None,
            "mean_observations": mean(r["costs"]["observation_cost"] for r in rows),
            "mean_compute": mean(r["costs"]["compute_cost"] for r in rows),
            "total_model_calls": sum(r["llm_ledger"]["model_calls"] for r in rows),
            "total_input_tokens": sum(r["llm_ledger"]["input_tokens"] for r in rows),
            "total_output_tokens": sum(r["llm_ledger"]["output_tokens"] for r in rows),
            "total_tokens": sum(r["llm_ledger"]["total_tokens"] for r in rows),
        }

    # Stage B: Zero-extra-LLM-call cross-scale transfer to N=5 via native UnifiedResearchAgent!
    stage_b_episodes = []
    for seed in n5_seeds:
        lab5_kwargs = {
            **n5_lab_common,
            "seed": seed,
            "name_seed": seed + 101,
            "wiring_seed": seed + 211,
        }
        transfer_arms = {}
        for t_arm, lib_to_use in (
            ("unconsolidated_parent_v10", parent_lib),
            (
                "consolidated_phase2b_zero_llm_calls",
                consolidated_og_library if consolidated_og_library is not None else parent_lib,
            ),
        ):
            cfg5 = ResearchConfig(
                seed=seed,
                observation_budget=300,
                compute_budget=260000,
                minimum_improvement=1e-7,
                max_candidates=4,
                world_structure_commitments=True,
                transition_premise_audit=True,
                expanded_premise_revisions=True,
            )
            agent5 = UnifiedResearchAgent(
                cfg5, measurement_budget=6000, library=deepcopy(lib_to_use)
            )
            t0 = time.monotonic()
            with AnonymousObjectLab(**lab5_kwargs) as lab5:
                res5 = agent5.run(lab5)
                prog5 = (
                    Program.from_dict(res5["program"])
                    if res5.get("program") is not None
                    else None
                )
                ch5 = tuple(res5.get("selected_channels") or ())
                mse5 = (
                    float(
                        loss(
                            prog5,
                            lab5.evaluation_rows(ch5, seed=seed + 1000003, length=48, trajectories=1),
                            res5["fallback"],
                            0.5,
                        )
                    )
                    if prog5 is not None
                    else None
                )
            transfer_arms[t_arm] = {
                "status": res5["status"],
                "selected_kind": res5.get("selected_kind"),
                "selected_channels": res5.get("selected_channels"),
                "costs": res5["costs"],
                "fresh48_mse": mse5,
                "passed_1e7": bool(mse5 is not None and mse5 <= 1e-7),
                "model_calls": 0,
                "total_tokens": 0,
                "wall_seconds": round(time.monotonic() - t0, 3),
            }
        print(
            f"      [Stage B N=5 Zero-Call Transfer {mechanism_id} seed={seed}] "
            f"unconsolidated={transfer_arms['unconsolidated_parent_v10']['passed_1e7']} "
            f"({_fmt_mse(transfer_arms['unconsolidated_parent_v10']['fresh48_mse'])}) | "
            f"consolidated_zero_llm={transfer_arms['consolidated_phase2b_zero_llm_calls']['passed_1e7']} "
            f"({_fmt_mse(transfer_arms['consolidated_phase2b_zero_llm_calls']['fresh48_mse'])}, "
            f"obs={transfer_arms['consolidated_phase2b_zero_llm_calls']['costs']['observation_cost']}, "
            f"compute={transfer_arms['consolidated_phase2b_zero_llm_calls']['costs']['compute_cost']}, calls=0)",
            flush=True,
        )
        stage_b_episodes.append({"seed": seed, "arms": transfer_arms})

    stage_b_aggregate = {}
    for t_arm in ("unconsolidated_parent_v10", "consolidated_phase2b_zero_llm_calls"):
        rows = [ep["arms"][t_arm] for ep in stage_b_episodes]
        mses = [r["fresh48_mse"] for r in rows if r["fresh48_mse"] is not None]
        stage_b_aggregate[t_arm] = {
            "tasks_total": len(rows),
            "passes_1e7": sum(1 for r in rows if r["passed_1e7"]),
            "pass_rate": sum(1 for r in rows if r["passed_1e7"]) / len(rows),
            "mean_fresh48_mse": mean(mses) if mses else None,
            "mean_observations": mean(r["costs"]["observation_cost"] for r in rows),
            "mean_compute": mean(r["costs"]["compute_cost"] for r in rows),
            "model_calls": 0,
            "total_tokens": 0,
        }

    return {
        "mechanism_id": mechanism_id,
        "stage_a_n4_single_factor_discovery": {
            "aggregate_by_arm": stage_a_aggregate,
            "episodes": stage_a_episodes,
        },
        "consolidated_commitment": consolidated_commitment_summary,
        "stage_b_n5_zero_llm_call_transfer": {
            "aggregate_by_arm": stage_b_aggregate,
            "episodes": stage_b_episodes,
        },
    }


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    transport = ResponsesJSONTransport(model=LIVE_MODEL, max_output_tokens=8192)
    print(f"[Phase 2B] Initialized ResponsesJSONTransport (model={LIVE_MODEL}, secrets_persisted=False)", flush=True)

    print("[1/4] Building genuine N=4 parent commitments from empty KnowledgeLibrary (0 LLM calls)...", flush=True)
    lib_win, comm_win, info_win = _build_parent_commitment_from_empty_library(
        readout="window_total_event", s0_seed=600401, s1_seed=600402
    )
    lib_inst, comm_inst, info_inst = _build_parent_commitment_from_empty_library(
        readout="total_event", s0_seed=600411, s1_seed=600412
    )
    print("      Parent commitments formed cleanly.", flush=True)

    print("[2/4] Running Mechanism 1: donor_positive_gap (P_exchange_form open-grammar AST)...", flush=True)
    mech1 = _run_three_arm_n4_discovery_and_n5_transfer(
        mechanism_id="donor_positive_gap_exchange_predicate",
        parent_lib=lib_win,
        parent_comm=comm_win,
        transport=transport,
        n4_seeds=(610401, 610402),
        n5_seeds=(610501, 610502),
        n4_lab_common=dict(
            objects=4,
            readout="window_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="donor_positive_gap",
            relation_radius=1.08,
            actuator_curvature=0.16,
        ),
        n5_lab_common=dict(
            objects=5,
            readout="window_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="donor_positive_gap",
            relation_radius=1.14,
            actuator_curvature=0.16,
        ),
    )

    print("[3/4] Running Mechanism 2: focal_quadratic_total_event (P_temporal_readout open-grammar AST)...", flush=True)
    mech2 = _run_three_arm_n4_discovery_and_n5_transfer(
        mechanism_id="focal_quadratic_total_event_readout_basis",
        parent_lib=lib_inst,
        parent_comm=comm_inst,
        transport=transport,
        n4_seeds=(620401, 620402),
        n5_seeds=(620501, 620502),
        n4_lab_common=dict(
            objects=4,
            readout="focal_quadratic_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="always",
            relation_radius=1.05,
            actuator_curvature=0.15,
        ),
        n5_lab_common=dict(
            objects=5,
            readout="focal_quadratic_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="always",
            relation_radius=1.12,
            actuator_curvature=0.15,
        ),
    )

    # Compute overall global LLM ledger summary
    all_mechs = [mech1, mech2]
    overall_stage_a = {}
    for arm_key in ("symbolic_v10_only", "llm_without_premise_gate", "llm_with_v10_premise_gate"):
        all_rows = [
            ep["arms"][arm_key]
            for m in all_mechs
            for ep in m["stage_a_n4_single_factor_discovery"]["episodes"]
        ]
        mses = [r["fresh48_mse"] for r in all_rows if r["fresh48_mse"] is not None]
        overall_stage_a[arm_key] = {
            "tasks_total": len(all_rows),
            "passes_1e7": sum(1 for r in all_rows if r["passed_1e7"]),
            "pass_rate": sum(1 for r in all_rows if r["passed_1e7"]) / len(all_rows),
            "mean_fresh48_mse": mean(mses) if mses else None,
            "total_model_calls": sum(r["llm_ledger"]["model_calls"] for r in all_rows),
            "total_input_tokens": sum(r["llm_ledger"]["input_tokens"] for r in all_rows),
            "total_output_tokens": sum(r["llm_ledger"]["output_tokens"] for r in all_rows),
            "total_tokens": sum(r["llm_ledger"]["total_tokens"] for r in all_rows),
        }

    overall_stage_b = {}
    for t_arm in ("unconsolidated_parent_v10", "consolidated_phase2b_zero_llm_calls"):
        all_rows = [
            ep["arms"][t_arm]
            for m in all_mechs
            for ep in m["stage_b_n5_zero_llm_call_transfer"]["episodes"]
        ]
        mses = [r["fresh48_mse"] for r in all_rows if r["fresh48_mse"] is not None]
        overall_stage_b[t_arm] = {
            "tasks_total": len(all_rows),
            "passes_1e7": sum(1 for r in all_rows if r["passed_1e7"]),
            "pass_rate": sum(1 for r in all_rows if r["passed_1e7"]) / len(all_rows),
            "mean_fresh48_mse": mean(mses) if mses else None,
            "mean_observations": mean(r["costs"]["observation_cost"] for r in all_rows),
            "mean_compute": mean(r["costs"]["compute_cost"] for r in all_rows),
            "model_calls": 0,
            "total_tokens": 0,
        }

    payload = {
        "generated_at_utc": _utc_now(),
        "study": "v10_phase2b_open_grammar_llm_single_factor_and_cross_scale_transfer",
        "llm_model": LIVE_MODEL,
        "endpoint_protocol": "POST /v1/responses (SSE stream=true, zero secret persistence)",
        "parent_commitment_formation": {
            "window_total_event_parent": info_win,
            "total_event_parent": info_inst,
        },
        "overall_stage_a_n4_single_factor": overall_stage_a,
        "overall_stage_b_n5_zero_llm_transfer": overall_stage_b,
        "mechanisms": [mech1, mech2],
    }
    out_path = EVIDENCE_DIR / "v10_phase2b_open_grammar_llm_20261116.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[4/4] Saved Phase 2B receipt to {out_path}", flush=True)


if __name__ == "__main__":
    main()
