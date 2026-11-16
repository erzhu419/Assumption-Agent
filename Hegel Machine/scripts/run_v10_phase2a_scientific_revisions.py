"""Run Phase 2A expanded scientific premise revision and N=3->4->5->6 evolution study.

Produces reproducible JSON receipt (`model_calls = 0`, `reported_tokens = 0`):
  ``docs/evidence/v10_world_structure_20261116/v10_phase2a_expanded_revisions_20261116.json``

Evaluates:
1. Genuine 2-episode N=4 source formation from an empty ``KnowledgeLibrary``
   (S0: N=4 symmetric -> S1: N=4 directional ``right``, conservative unconditional
   ``window_total_event``).
2. Matched 3-arm cross-scale (``N=4 -> N=5``) ablation across 4 mechanism shifts
   (2 seeds per shift = 8 target apparatuses, 24 episodes total):
   - ``baseline_archive``: ``world_structure_commitments=False, transition_premise_audit=False``
   - ``v10_phase1_direction_only``: ``world_structure_commitments=True, transition_premise_audit=True, expanded_premise_revisions=False``
   - ``v10_phase2a_full_premise_revision``: ``world_structure_commitments=True, transition_premise_audit=True, expanded_premise_revisions=True``
   Shifts:
   (a) ``conditional_source_greater`` (``P_exchange_form`` refuted -> ``source_greater``)
   (b) ``conditional_source_less_and_flip`` (``P_exchange_form`` + ``P_interaction_direction`` refuted -> ``source_less`` + ``left``)
   (c) ``nonconservative_sink`` (``P_conservation`` refuted -> local dissipation ``sink``)
   (d) ``nonconservative_nonlocal_and_flip`` (``P_conservation`` + ``P_interaction_direction`` refuted -> global coupling ``nonlocal`` + ``left``)
3. Multi-generation commitment evolution chain ``N=3 -> N=4 -> N=5 -> N=6`` from an
   empty ``KnowledgeLibrary`` (culminating in ``6! = 720 -> 1`` position & focus
   bijection collapse under simultaneous direction flip + non-conservation ``sink``).
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
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
)

EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_mse(val: float | None) -> str:
    return f"{val:.2e}" if val is not None else "None"


def _eval_heldout_48(lab: AnonymousObjectLab, report: dict, eval_seed: int) -> float | None:
    if report.get("program") is None:
        return None
    prog = Program.from_dict(report["program"])
    channels = tuple(report.get("selected_channels") or ())
    rows = lab.evaluation_rows(channels, seed=eval_seed, length=48, trajectories=1)
    return float(loss(prog, rows, report["fallback"], 0.5))


def _extract_last_audit(actions: list[dict]) -> dict | None:
    audits = [a for a in actions if a.get("action") == "audit_transition_premises"]
    return audits[-1] if audits else None


def build_n4_source_library() -> tuple[KnowledgeLibrary, dict]:
    """Form N=4 conservative unconditional 'right' commitment from empty library."""
    lib = KnowledgeLibrary()
    cfg_s0 = ResearchConfig(
        observation_budget=340,
        compute_budget=280000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
    )
    agent_s0 = UnifiedResearchAgent(cfg_s0, measurement_budget=6800, library=lib)
    with AnonymousObjectLab(
        seed=520401,
        objects=4,
        readout="window_total_event",
        relation_direction="symmetric",
        relation_radius=0.92,
        actuator_curvature=0.16,
        name_seed=13,
        wiring_seed=23,
    ) as lab_s0:
        res_s0 = agent_s0.run(lab_s0)
        mse_s0 = _eval_heldout_48(lab_s0, res_s0, eval_seed=620401)

    cfg_s1 = ResearchConfig(
        observation_budget=460,
        compute_budget=340000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
    )
    agent_s1 = UnifiedResearchAgent(cfg_s1, measurement_budget=9200, library=lib)
    with AnonymousObjectLab(
        seed=520402,
        objects=4,
        readout="window_total_event",
        relation_direction="right",
        relation_radius=0.94,
        actuator_curvature=0.16,
        name_seed=17,
        wiring_seed=29,
    ) as lab_s1:
        res_s1 = agent_s1.run(lab_s1)
        mse_s1 = _eval_heldout_48(lab_s1, res_s1, eval_seed=620402)

    right_priors = [
        deepcopy(p)
        for p in lib.meta_priors
        if p.get("status") == "supported"
        and '"right"' in json.dumps(p.get("research_program", {}))
    ]
    frozen_lib = KnowledgeLibrary()
    frozen_lib.meta_priors = right_priors if right_priors else deepcopy(lib.meta_priors)
    comms = consolidate_world_structure_commitments(frozen_lib.meta_priors)

    summary = {
        "from_empty_library": True,
        "s0_n4_symmetric": {
            "status": res_s0["status"],
            "costs": res_s0["costs"],
            "fresh48_mse": mse_s0,
            "passed_1e7": bool(mse_s0 is not None and mse_s0 <= 1e-7),
        },
        "s1_n4_right": {
            "status": res_s1["status"],
            "costs": res_s1["costs"],
            "fresh48_mse": mse_s1,
            "passed_1e7": bool(mse_s1 is not None and mse_s1 <= 1e-7),
        },
        "retained_right_priors": len(frozen_lib.meta_priors),
        "consolidated_commitments": len(comms),
    }
    return frozen_lib, summary


def run_cohort_1_three_arm_ablation(source_lib: KnowledgeLibrary) -> dict:
    """Run 3-arm cross-scale N=4 -> N=5 ablation across 4 Phase 2A mechanism shifts."""
    arms = (
        ("baseline_archive", False, False, False),
        ("v10_phase1_direction_only", True, True, False),
        ("v10_phase2a_full_premise_revision", True, True, True),
    )
    shifts = [
        {
            "shift_id": "conditional_source_greater",
            "seeds": (520501, 520502),
            "lab_common": dict(
                objects=5,
                readout="window_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="source_greater",
                relation_radius=1.12,
                actuator_curvature=0.17,
            ),
        },
        {
            "shift_id": "conditional_source_less_and_flip",
            "seeds": (520511, 520512),
            "lab_common": dict(
                objects=5,
                readout="window_total_event",
                variant="exchange",
                relation_direction="left",
                relation_condition="source_less",
                relation_radius=1.10,
                actuator_curvature=0.17,
            ),
        },
        {
            "shift_id": "nonconservative_sink",
            "seeds": (520521, 520522),
            "lab_common": dict(
                objects=5,
                readout="window_total_event",
                variant="sink",
                relation_direction="right",
                relation_condition="always",
                relation_radius=1.08,
                actuator_curvature=0.15,
            ),
        },
        {
            "shift_id": "nonconservative_nonlocal_and_flip",
            "seeds": (520531, 520532),
            "lab_common": dict(
                objects=5,
                readout="window_total_event",
                variant="nonlocal",
                relation_direction="left",
                relation_condition="always",
                relation_radius=1.11,
                actuator_curvature=0.15,
            ),
        },
    ]

    episodes = []
    for shift in shifts:
        for seed in shift["seeds"]:
            lab_kwargs = {
                **shift["lab_common"],
                "seed": seed,
                "name_seed": seed + 101,
                "wiring_seed": seed + 211,
            }
            per_arm = {}
            for arm_name, wsc, tpa, exp_rev in arms:
                cfg = ResearchConfig(
                    seed=seed,
                    observation_budget=300,
                    compute_budget=260000,
                    minimum_improvement=1e-7,
                    max_candidates=4,
                    world_structure_commitments=wsc,
                    transition_premise_audit=tpa,
                    expanded_premise_revisions=exp_rev,
                )
                lib_copy = deepcopy(source_lib)
                agent = UnifiedResearchAgent(cfg, measurement_budget=6000, library=lib_copy)
                t0 = time.monotonic()
                with AnonymousObjectLab(**lab_kwargs) as lab:
                    res = agent.run(lab)
                    mse = _eval_heldout_48(lab, res, eval_seed=seed + 1000003)
                elapsed = round(time.monotonic() - t0, 3)
                last_audit = _extract_last_audit(res.get("actions", []))
                per_arm[arm_name] = {
                    "status": res["status"],
                    "selected_kind": res.get("selected_kind"),
                    "selected_channels": res.get("selected_channels"),
                    "costs": res["costs"],
                    "fresh48_mse": mse,
                    "passed_1e7": bool(mse is not None and mse <= 1e-7),
                    "wall_seconds": elapsed,
                    "last_audit_status": last_audit.get("status") if last_audit else None,
                    "verified_direction": last_audit.get("verified_direction") if last_audit else None,
                    "verified_edge_condition": last_audit.get("verified_edge_condition") if last_audit else None,
                    "nonconservation_mechanisms": last_audit.get("nonconservation_mechanisms") if last_audit else None,
                    "surviving_bijection_count": last_audit.get("surviving_bijection_count") if last_audit else None,
                }
            print(
                f"      [{shift['shift_id']} seed={seed}] "
                f"base={per_arm['baseline_archive']['passed_1e7']} ({_fmt_mse(per_arm['baseline_archive']['fresh48_mse'])}) | "
                f"p1={per_arm['v10_phase1_direction_only']['passed_1e7']} ({_fmt_mse(per_arm['v10_phase1_direction_only']['fresh48_mse'])}) | "
                f"p2a={per_arm['v10_phase2a_full_premise_revision']['passed_1e7']} ({_fmt_mse(per_arm['v10_phase2a_full_premise_revision']['fresh48_mse'])})",
                flush=True,
            )
            episodes.append(
                {
                    "shift_id": shift["shift_id"],
                    "seed": seed,
                    "arms": per_arm,
                }
            )

    aggregate = {}
    for arm_name, *_ in arms:
        arm_rows = [ep["arms"][arm_name] for ep in episodes]
        passes = sum(1 for r in arm_rows if r["passed_1e7"])
        valid_mses = [r["fresh48_mse"] for r in arm_rows if r["fresh48_mse"] is not None]
        aggregate[arm_name] = {
            "tasks_total": len(arm_rows),
            "passes_1e7": passes,
            "pass_rate": passes / len(arm_rows),
            "mean_fresh48_mse": mean(valid_mses) if valid_mses else None,
            "mean_observations": mean(r["costs"]["observation_cost"] for r in arm_rows),
            "mean_compute": mean(r["costs"]["compute_cost"] for r in arm_rows),
            "mean_measurement_cost": mean(r["costs"]["measurement_cost"] for r in arm_rows),
            "model_calls": 0,
        }
    return {"aggregate_by_arm": aggregate, "episodes": episodes}


def run_cohort_2_four_generation_evolution() -> dict:
    """Run N=3 -> N=4 -> N=5 -> N=6 end-to-end commitment evolution from empty library."""
    stages_spec = [
        {
            "generation_expected": 0,
            "stage_label": "Gen0_N3_symmetric_unconditional_conserved",
            "obs_budget": 320,
            "compute_budget": 260000,
            "meas_budget": 6400,
            "lab_kwargs": dict(
                seed=530301,
                objects=3,
                readout="window_total_event",
                variant="exchange",
                relation_direction="symmetric",
                relation_condition="always",
                relation_radius=0.92,
                actuator_curvature=0.14,
                name_seed=301,
                wiring_seed=401,
            ),
        },
        {
            "generation_expected": 1,
            "stage_label": "Gen1_N4_revise_direction_to_right",
            "obs_budget": 360,
            "compute_budget": 300000,
            "meas_budget": 7200,
            "lab_kwargs": dict(
                seed=530401,
                objects=4,
                readout="window_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="always",
                relation_radius=0.98,
                actuator_curvature=0.15,
                name_seed=302,
                wiring_seed=402,
            ),
        },
        {
            "generation_expected": 2,
            "stage_label": "Gen2_N5_revise_condition_to_source_greater",
            "obs_budget": 360,
            "compute_budget": 300000,
            "meas_budget": 7200,
            "lab_kwargs": dict(
                seed=530501,
                objects=5,
                readout="window_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="source_greater",
                relation_radius=1.10,
                actuator_curvature=0.16,
                name_seed=303,
                wiring_seed=403,
            ),
        },
        {
            "generation_expected": 3,
            "stage_label": "Gen3_N6_revise_direction_left_and_nonconservation_sink",
            "obs_budget": 460,
            "compute_budget": 500000,
            "meas_budget": 10800,
            "lab_kwargs": dict(
                seed=530601,
                objects=6,
                readout="window_total_event",
                variant="sink",
                relation_direction="left",
                relation_condition="source_greater",
                relation_radius=1.12,
                actuator_curvature=0.15,
                name_seed=304,
                wiring_seed=404,
            ),
        },
    ]

    active_library = KnowledgeLibrary()
    stage_records = []
    for st in stages_spec:
        cfg = ResearchConfig(
            seed=st["lab_kwargs"]["seed"],
            observation_budget=st["obs_budget"],
            compute_budget=st["compute_budget"],
            minimum_improvement=1e-7,
            max_candidates=4,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
        )
        agent = UnifiedResearchAgent(
            cfg, measurement_budget=st["meas_budget"], library=active_library
        )
        t0 = time.monotonic()
        with AnonymousObjectLab(**st["lab_kwargs"]) as lab:
            res = agent.run(lab)
            mse = _eval_heldout_48(lab, res, eval_seed=st["lab_kwargs"]["seed"] + 1000003)
        elapsed = round(time.monotonic() - t0, 3)

        # Retain latest confirmed world commitment prior at front of active_library for clean lineage evolution
        supported_priors = [
            p for p in active_library.meta_priors if p.get("status") == "supported"
        ]
        if supported_priors:
            active_library.meta_priors = [deepcopy(supported_priors[-1])]
        comms = consolidate_world_structure_commitments(active_library.meta_priors)
        top_comm = comms[0] if comms else {}
        last_audit = _extract_last_audit(res.get("actions", []))

        rec = {
            "stage_label": st["stage_label"],
            "arity": st["lab_kwargs"]["objects"],
            "status": res["status"],
            "selected_kind": res.get("selected_kind"),
            "selected_channels": res.get("selected_channels"),
            "costs": res["costs"],
            "fresh48_mse": mse,
            "passed_1e7": bool(mse is not None and mse <= 1e-7),
            "wall_seconds": elapsed,
            "audit_status": last_audit.get("status") if last_audit else None,
            "surviving_bijections": last_audit.get("surviving_bijection_count") if last_audit else None,
            "consolidated_generation": top_comm.get("generation"),
            "consolidated_direction": (top_comm.get("radius_parameters") or [{}])[0].get("direction"),
            "consolidated_edge_condition": (top_comm.get("radius_parameters") or [{}])[0].get("edge_condition"),
            "consolidated_nonconservation": top_comm.get("nonconservation_mechanism"),
            "revision_lineage": top_comm.get("revision_lineage", []),
        }
        print(
            f"      [{st['stage_label']}] pass={rec['passed_1e7']} mse={_fmt_mse(mse)} "
            f"gen={rec['consolidated_generation']} obs={res['costs']['observation_cost']} "
            f"compute={res['costs']['compute_cost']}",
            flush=True,
        )
        stage_records.append(rec)

    return {
        "from_empty_knowledge_library": True,
        "all_four_generations_passed_1e7": all(r["passed_1e7"] for r in stage_records),
        "final_generation": stage_records[-1]["consolidated_generation"],
        "final_revision_lineage": stage_records[-1]["revision_lineage"],
        "stages": stage_records,
    }


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] Forming N=4 conservative directional source library from empty KnowledgeLibrary...", flush=True)
    source_lib, src_summary = build_n4_source_library()
    print(
        f"      Source formed: S0_pass={src_summary['s0_n4_symmetric']['passed_1e7']}, "
        f"S1_pass={src_summary['s1_n4_right']['passed_1e7']}, "
        f"commitments={src_summary['consolidated_commitments']}",
        flush=True,
    )

    print("[2/3] Running Cohort 1: 3-arm cross-scale N=4 -> N=5 ablation across 4 Phase 2A shifts (24 episodes)...", flush=True)
    cohort1 = run_cohort_1_three_arm_ablation(source_lib)

    print("[3/3] Running Cohort 2: 4-generation N=3 -> N=4 -> N=5 -> N=6 commitment evolution...", flush=True)
    cohort2 = run_cohort_2_four_generation_evolution()

    payload = {
        "generated_at_utc": _utc_now(),
        "study": "v10_phase2a_expanded_scientific_revisions_and_multigen_evolution",
        "model_calls": 0,
        "reported_tokens": 0,
        "source_formation_n4": src_summary,
        "cohort_1_three_arm_ablation_n4_to_n5": cohort1,
        "cohort_2_four_generation_n3_to_n6": cohort2,
    }
    out_path = EVIDENCE_DIR / "v10_phase2a_expanded_revisions_20261116.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Phase 2A receipt to {out_path}", flush=True)


if __name__ == "__main__":
    main()
