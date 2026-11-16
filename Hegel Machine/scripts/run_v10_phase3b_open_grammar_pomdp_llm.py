"""Run Phase 3B live open-grammar + POMDP LLM single-factor controlled study (v10).

Produces reproducible JSON receipt with strictly separated symbolic and LLM ledgers:
  ``docs/evidence/v10_world_structure_20261116/v10_phase3b_open_grammar_pomdp_llm_20261116.json``

Evaluates ``model="gpt-5.6-sol"`` via ``ResponsesJSONTransport`` (zero secret persistence)
across two nonlinear POMDP mechanisms outside the entire symbolic operator repertoire (including Phase 3A):
1. ``bilinear_work_transition_memory`` (``P_latent_transition_drive`` refuted on paid transitions:
   ``z_{i,t} = alpha * z_{i,t-1} + q_{i,t} * u_{i,t}`` coupled into ``after[i]``):
   - Stage A (``N=4`` discovery, seeds ``950401, 950402``):
     3 matched single-factor arms on identical paid apparatus probes & trajectories:
     * ``symbolic_pomdp_v10_only`` (``model_calls=0``, full Phase 3A POMDP symbolic repertoire)
     * ``llm_without_premise_gate`` (``propose_flat_unconstrained_programs`` with ``accumulate``, ``max_llm_calls=2``)
     * ``llm_with_v10_pomdp_premise_gate`` (``propose_premise_gated_commitment_revisions``, ``max_llm_calls=2``)
   - Stage B (``N=4 -> N=5`` cross-scale shifted-pole & shifted-radius zero-LLM-call transfer,
     seeds ``950501, 950502``, ``alpha: 0.65 -> 0.74``, ``r_0: 0.96 -> 1.12``):
     Full ``UnifiedResearchAgent.run`` with ``model_calls = 0`` comparing
     ``unconsolidated_parent_pomdp_v10`` vs ``consolidated_phase3b_zero_llm_calls``.
2. ``quadratic_energy_focal_memory`` (``P_temporal_readout`` + ``P_markov_state`` refuted on
   role-resolved switching-focus trajectory residuals: ``z_{i,t} = alpha * z_{i,t-1} + q_{i,t}^2``):
   - Stage A (``N=4`` discovery, seeds ``960401, 960402``): same 3 matched arms.
   - Stage B (``N=4 -> N=5`` cross-scale shifted-pole & shifted-radius zero-LLM-call transfer,
     seeds ``960501, 960502``, ``alpha: 0.66 -> 0.75``, ``r_0: 0.95 -> 1.10``):
     Full ``UnifiedResearchAgent.run`` with ``model_calls = 0``.
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
    VERIFIED_EXCHANGE_STATUSES,
    apply_verified_transition_premises,
    audit_transition_premises,
)
from hegel_machine.cognition.research_programs import _Fitter, _plan
from hegel_machine.cognition.structure_lab import ObjectSchema
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


def _acquire_shared_pomdp_bundle(
    lab: AnonymousObjectLab,
    *,
    seed: int,
    include_focus_probes: bool,
) -> dict:
    """Acquire identical public role sweeps + multi-step trajectories for all 3 arms."""
    budget = SearchBudget(240000)
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
    allowed_purposes = {"isolate_control_effects", "distinguish_role_definition"}
    if include_focus_probes:
        allowed_purposes |= {"observe_missing_membership_states", "distinguish_membership_dependency"}
    for serial in range(28 if include_focus_probes else 14):
        probe, question = propose_object_probe(raw_spec, model, records, serial)
        if probe is None or question["purpose"] not in allowed_purposes:
            break
        acquire_probe(probe)
        model = infer_object_model(raw_spec, records, budget)

    training: list[Observation] = []
    train_transitions: list[dict] = []
    for tr_idx in range(3):
        t_obs, t_tr = acquire_probe(
            lab.random_probe(seed + 101 + 37 * tr_idx, length=32, name=f"train_traj_{tr_idx}")
        )
        training.extend(t_obs)
        train_transitions.extend(t_tr)
    validation = list(
        lab.observe(lab.random_probe(seed + 303, length=36, name="val_traj"), ("fine",))
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


def _run_symbolic_pomdp_v10_arm(
    lab: AnonymousObjectLab,
    bundle: dict,
    parent_comm: dict,
    *,
    eval_seed: int,
) -> dict:
    """Arm 1: Full Symbolic POMDP v10 (Phase 3A repertoire, model_calls = 0) on shared bundle."""
    t0 = time.monotonic()
    budget = SearchBudget(240000)
    spec = bundle["spec"]
    training = bundle["training"]
    validation = bundle["validation"]
    fallback = initial_prediction(training, spec.risk_quantile)

    audit = audit_transition_premises(
        deepcopy(bundle["model"]),
        bundle["records"],
        budget,
        commitments=[parent_comm],
        allow_expanded_revisions=True,
        allow_latent_state=True,
        allow_higher_order_latent=True,
        observations=training,
        tolerance=1e-7,
    )
    updated = apply_verified_transition_premises(deepcopy(bundle["model"]), audit)
    candidate_entries = []
    if updated.get("schema"):
        if audit.get("scientific_revision_triggers"):
            candidate_entries.extend(
                propose_scientific_premise_revision(
                    parent_comm,
                    audit,
                    spec,
                    training,
                    updated["schema"],
                    budget,
                    allow_latent_state=True,
                    allow_higher_order_latent=True,
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
                    audit_report=audit,
                    allow_latent_state=True,
                    allow_higher_order_latent=True,
                    tolerance=1e-7,
                )
            )

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
        "arm": "symbolic_pomdp_v10_only",
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


def _run_three_arm_pomdp_discovery_and_n5_transfer(
    *,
    mechanism_id: str,
    parent_lib: KnowledgeLibrary,
    parent_comm: dict,
    transport: ResponsesJSONTransport,
    n4_seeds: tuple[int, int],
    n5_seeds: tuple[int, int],
    n4_lab_common: dict,
    n5_lab_common: dict,
    include_focus_probes: bool,
) -> dict:
    stage_a_episodes = []
    consolidated_og_library: KnowledgeLibrary | None = None
    consolidated_commitment_summary: dict | None = None

    for seed in n4_seeds:
        lab_kwargs = {
            **n4_lab_common,
            "seed": seed,
            "name_seed": seed % 997 + 17,
            "wiring_seed": seed % 991 + 31,
        }
        with AnonymousObjectLab(**lab_kwargs) as lab:
            bundle = _acquire_shared_pomdp_bundle(
                lab, seed=seed, include_focus_probes=include_focus_probes
            )
            fallback = initial_prediction(bundle["training"], bundle["spec"].risk_quantile)
            eval_seed = seed + 1000003

            # Arm 1: symbolic_pomdp_v10_only (full Phase 3A repertoire, model_calls = 0)
            arm_sym = _run_symbolic_pomdp_v10_arm(
                lab, bundle, parent_comm, eval_seed=eval_seed
            )

            # Arm 2: llm_without_premise_gate (live gpt-5.6-sol, accumulate enabled, max_llm_calls = 2)
            t_flat = time.monotonic()
            budget_flat = SearchBudget(220000)
            flat_res = propose_flat_unconstrained_programs(
                bundle["spec"],
                bundle["training"],
                bundle["validation"],
                bundle["train_transitions"],
                budget_flat,
                transport,
                incumbent_loss=arm_sym["validation_loss"],
                max_llm_calls=2,
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

            # Arm 3: llm_with_v10_pomdp_premise_gate (live gpt-5.6-sol, max_llm_calls = 2)
            t_gated = time.monotonic()
            budget_gated = SearchBudget(220000)
            gated_res = propose_premise_gated_commitment_revisions(
                parent_comm,
                deepcopy(bundle["model"]),
                bundle["records"],
                bundle["spec"],
                bundle["training"],
                bundle["validation"],
                budget_gated,
                transport,
                max_llm_calls=2,
                tolerance=1e-7,
                model_name=LIVE_MODEL,
            )
            gated_best = gated_res["best_entry"]
            gated_prog = gated_best["candidate"].program if gated_best else None
            gated_mse = _eval_program_heldout_48(lab, gated_prog, fallback, eval_seed)
            arm_gated = {
                "arm": "llm_with_v10_pomdp_premise_gate",
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
                        latent_state_discovery=True,
                        higher_order_latent_discovery=True,
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
                    confirmation_loss=float(gated_mse),
                    confirmation_rows=48,
                    selected_channels=("fine",),
                    confirmation_ref={
                        "kind": "heldout_fresh48_trajectory",
                        "mechanism_id": mechanism_id,
                        "source_seed": seed,
                        "eval_seed": eval_seed,
                        "confirmation_rows": 48,
                        "selection_role": "first_passing_source_gate",
                    },
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
                            "latent_state_mechanism": c0.get("latent_state_mechanism"),
                            "source_latent_poles": c0.get("source_latent_poles"),
                            "role_term_count": len(c0.get("role_terms", ())),
                            "radius_parameters": c0.get("radius_parameters"),
                            "revision_lineage": c0.get("revision_lineage"),
                        }

        print(
            f"      [Stage A N=4 {mechanism_id} seed={seed}] "
            f"sym_pomdp={arm_sym['passed_1e7']} ({_fmt_mse(arm_sym['fresh48_mse'])}, calls=0) | "
            f"flat_llm={arm_flat['passed_1e7']} ({_fmt_mse(arm_flat['fresh48_mse'])}, "
            f"calls={arm_flat['llm_ledger']['model_calls']}, tok={arm_flat['llm_ledger']['total_tokens']}) | "
            f"v10_pomdp_gated_llm={arm_gated['passed_1e7']} ({_fmt_mse(arm_gated['fresh48_mse'])}, "
            f"calls={arm_gated['llm_ledger']['model_calls']}, tok={arm_gated['llm_ledger']['total_tokens']})",
            flush=True,
        )
        stage_a_episodes.append(
            {
                "seed": seed,
                "arms": {
                    "symbolic_pomdp_v10_only": arm_sym,
                    "llm_without_premise_gate": arm_flat,
                    "llm_with_v10_pomdp_premise_gate": arm_gated,
                },
            }
        )

    stage_a_aggregate = {}
    for arm_key in (
        "symbolic_pomdp_v10_only",
        "llm_without_premise_gate",
        "llm_with_v10_pomdp_premise_gate",
    ):
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

    # Stage B: Zero-extra-LLM-call cross-scale shifted-pole & shifted-radius transfer to N=5!
    stage_b_episodes = []
    for seed in n5_seeds:
        lab5_kwargs = {
            **n5_lab_common,
            "seed": seed,
            "name_seed": seed % 997 + 19,
            "wiring_seed": seed % 991 + 37,
        }
        transfer_arms = {}
        for t_arm, lib_to_use in (
            ("unconsolidated_parent_pomdp_v10", parent_lib),
            (
                "consolidated_phase3b_zero_llm_calls",
                consolidated_og_library if consolidated_og_library is not None else parent_lib,
            ),
        ):
            cfg5 = ResearchConfig(
                seed=seed,
                observation_budget=400,
                compute_budget=480000,
                minimum_improvement=1e-7,
                max_candidates=4,
                world_structure_commitments=True,
                transition_premise_audit=True,
                expanded_premise_revisions=True,
                latent_state_discovery=True,
                higher_order_latent_discovery=True,
            )
            agent5 = UnifiedResearchAgent(
                cfg5, measurement_budget=8600, library=deepcopy(lib_to_use)
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
            f"      [Stage B N=5 Zero-Call Shifted-Pole Transfer {mechanism_id} seed={seed}] "
            f"unconsolidated={transfer_arms['unconsolidated_parent_pomdp_v10']['passed_1e7']} "
            f"({_fmt_mse(transfer_arms['unconsolidated_parent_pomdp_v10']['fresh48_mse'])}) | "
            f"consolidated_zero_llm={transfer_arms['consolidated_phase3b_zero_llm_calls']['passed_1e7']} "
            f"({_fmt_mse(transfer_arms['consolidated_phase3b_zero_llm_calls']['fresh48_mse'])}, "
            f"obs={transfer_arms['consolidated_phase3b_zero_llm_calls']['costs']['observation_cost']}, "
            f"compute={transfer_arms['consolidated_phase3b_zero_llm_calls']['costs']['compute_cost']}, calls=0)",
            flush=True,
        )
        stage_b_episodes.append({"seed": seed, "arms": transfer_arms})

    stage_b_aggregate = {}
    for t_arm in ("unconsolidated_parent_pomdp_v10", "consolidated_phase3b_zero_llm_calls"):
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
    t_start = time.monotonic()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    transport = ResponsesJSONTransport(model=LIVE_MODEL, max_output_tokens=8192)
    print(
        f"[Phase 3B] Initialized ResponsesJSONTransport (model={LIVE_MODEL}, secrets_persisted=False)",
        flush=True,
    )

    v4 = _schema(4, "src4")
    lib_te = KnowledgeLibrary()
    lib_te.meta_priors = [
        _make_markov_right_prior(v4, concept_id="p_te_1", radius=0.91, readout_kind="total_event"),
        _make_markov_right_prior(v4, concept_id="p_te_2", radius=0.98, readout_kind="total_event"),
    ]
    comm_te = consolidate_world_structure_commitments(lib_te.meta_priors)[0]

    lib_foc = KnowledgeLibrary()
    lib_foc.meta_priors = [
        _make_markov_right_prior(v4, concept_id="p_foc_1", radius=0.91, readout_kind="focus"),
        _make_markov_right_prior(v4, concept_id="p_foc_2", radius=0.97, readout_kind="focus"),
    ]
    comm_foc = consolidate_world_structure_commitments(lib_foc.meta_priors)[0]

    print("[1/3] Running Mechanism 1: bilinear_work_transition_memory (P_latent_transition_drive)...", flush=True)
    mech1 = _run_three_arm_pomdp_discovery_and_n5_transfer(
        mechanism_id="bilinear_work_transition_memory",
        parent_lib=lib_te,
        parent_comm=comm_te,
        transport=transport,
        n4_seeds=(950401, 950402),
        n5_seeds=(950501, 950502),
        n4_lab_common=dict(
            objects=4,
            readout="total_event",
            variant="exchange",
            relation_direction="right",
            latent_mode="bilinear_work_transition_memory",
            latent_decay=0.65,
            latent_weight=0.44,
            relation_radius=0.96,
            actuator_curvature=0.14,
        ),
        n5_lab_common=dict(
            objects=5,
            readout="total_event",
            variant="exchange",
            relation_direction="right",
            latent_mode="bilinear_work_transition_memory",
            latent_decay=0.74,
            latent_weight=0.48,
            relation_radius=1.12,
            actuator_curvature=0.15,
        ),
        include_focus_probes=False,
    )

    print("[2/3] Running Mechanism 2: quadratic_energy_focal_memory (P_temporal_readout + accumulate)...", flush=True)
    mech2 = _run_three_arm_pomdp_discovery_and_n5_transfer(
        mechanism_id="quadratic_energy_focal_memory",
        parent_lib=lib_foc,
        parent_comm=comm_foc,
        transport=transport,
        n4_seeds=(960401, 960402),
        n5_seeds=(960501, 960502),
        n4_lab_common=dict(
            objects=4,
            readout="focus",
            variant="exchange",
            relation_direction="right",
            latent_mode="quadratic_energy_focal_memory",
            latent_decay=0.66,
            latent_weight=0.42,
            relation_radius=0.95,
            actuator_curvature=0.14,
        ),
        n5_lab_common=dict(
            objects=5,
            readout="focus",
            variant="exchange",
            relation_direction="right",
            latent_mode="quadratic_energy_focal_memory",
            latent_decay=0.75,
            latent_weight=0.47,
            relation_radius=1.10,
            actuator_curvature=0.15,
        ),
        include_focus_probes=True,
    )

    all_mechs = [mech1, mech2]
    overall_stage_a = {}
    for arm_key in (
        "symbolic_pomdp_v10_only",
        "llm_without_premise_gate",
        "llm_with_v10_pomdp_premise_gate",
    ):
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
    for t_arm in ("unconsolidated_parent_pomdp_v10", "consolidated_phase3b_zero_llm_calls"):
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
        "wall_seconds": round(time.monotonic() - t_start, 3),
        "study": "v10_phase3b_open_grammar_pomdp_llm_single_factor_and_cross_scale_transfer",
        "llm_model": LIVE_MODEL,
        "endpoint_protocol": "POST /v1/responses (SSE stream=true, zero secret persistence)",
        "overall_stage_a_n4_single_factor": overall_stage_a,
        "overall_stage_b_n5_zero_llm_transfer": overall_stage_b,
        "mechanisms": [mech1, mech2],
    }
    out_path = EVIDENCE_DIR / "v10_phase3b_open_grammar_pomdp_llm_20261116.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[3/3] Saved Phase 3B receipt to {out_path}", flush=True)


if __name__ == "__main__":
    main()
