"""Verification suite for v10 Phase 2C: active open-grammar threshold probing & POMDP latent-state growth."""

from copy import deepcopy

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe,
    plan_competing_object_probe,
)
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from hegel_machine.cognition.premise_dependency_audit import (
    apply_verified_transition_premises,
    audit_transition_premises,
    describe_comparator_threshold_parameter,
    extract_comparator_threshold_interval,
)
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
    instantiate_commitment_on_schema,
)


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
    concept_id: str = "src_n4_gap",
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
    concept_id: str = "src_n4_markov",
    radius: float = 0.94,
    readout_kind: str = "total_event",
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
        fam = "local_exchange"
        fam_id = "T08"
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
        fam = "joint_event"
        fam_id = "T11"

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


def test_comparator_threshold_interval_and_active_compound_bisection():
    """Verify exact 1D threshold interval extraction and active compound-region bisection."""
    true_thresh = 0.43
    true_rad = 1.02
    v4 = _schema(4, "src4")
    comms = consolidate_world_structure_commitments(
        [
            _make_custom_threshold_prior(v4, concept_id="g1", threshold=0.25, radius=0.91),
            _make_custom_threshold_prior(v4, concept_id="g2", threshold=0.25, radius=0.98),
        ]
    )
    assert len(comms) == 1
    comm = comms[0]

    with UnifiedObjectLab(
        seed=710401,
        objects=4,
        readout="total_event",
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=true_thresh,
        relation_radius=true_rad,
        actuator_curvature=0.0,
    ) as lab:
        spec = lab.spec_for(("fine",))
        command_spec = lab.spec_for(())
        schema = lab._schema(("fine",))
        batch = lab.acquire(lab.random_probe(91, length=20, name="sparse_init"), ("fine",))
        budget = SearchBudget(120000)

        desc = describe_comparator_threshold_parameter(_donor_positive_gap_predicate(0.25))
        assert desc is not None
        assert desc["comparator_monotonicity"] == "decreasing"

        entries = instantiate_commitment_on_schema(
            spec,
            batch.observations,
            schema,
            comm,
            budget,
            radius_bounds=[true_rad - 1e-8, true_rad + 1e-8],
            transitions=batch.transitions,
            active_threshold_probing=True,
            tolerance=1e-8,
        )
        assert entries and entries[0]["training_loss"] <= 1e-12
        entry = entries[0]
        saved_params = entry["research_program"]["selection_steps"][-1]["parameters"]
        thresh_params = [p for p in saved_params if p.get("parameter_kind") == "predicate_threshold"]
        assert len(thresh_params) == 1
        init_iv = thresh_params[0]["radius_intervals"][0]
        assert init_iv[0] < true_thresh < init_iv[1]
        init_width = init_iv[1] - init_iv[0]
        assert init_width > 0.01

        # Build exact linear command->fine binding (actuator_curvature=0.0 for direct analytic binding)
        binding = {}
        for g in schema.objects:
            for r in ("q", "x", "u"):
                binding[schema.fields[g][r]] = Program("feature", name=schema.controls[g][r])
            idx = int(g.removeprefix("o"))
            f_ctl = Program("feature", name="focus")
            binding[schema.fields[g]["f"]] = Program(
                "and",
                children=(
                    Program("ge", children=(f_ctl, Program("const", value=idx - 0.5))),
                    Program("lt", children=(f_ctl, Program("const", value=idx + 0.5))),
                ),
            )

        research = RelationIntervalResearch(
            entry["candidate"].program,
            [],
            budget,
            structure=entry.get("structure"),
            research_program=entry.get("research_program"),
        )
        paid = list(batch.observations)
        experiments = []
        for step_idx in range(14):
            cands = research.alternatives(paid, budget, tolerance=1e-8, experiments=experiments)
            if not cands:
                break
            research.adopt(cands[0])
            qs = research.questions(cands)
            if not qs:
                break
            plan = plan_competing_object_probe(
                command_spec, cands, budget, binding=binding, serial=step_idx, length=3, questions=qs
            )
            assert plan["probe"] is not None
            assert plan["forecast_status"] == "predicted_distinction"
            b_cmd = lab._lab.acquire(plan["probe"], objects=(), scale=1)
            b_fine = lab.acquire(plan["probe"], ("fine",))
            paired = list(zip(b_cmd.observations, b_fine.observations))
            ev = assess_competing_object_probe(plan, paired, budget, tolerance=1e-8)
            paid.extend(b_fine.observations)
            if ev["status"] == "distinguished":
                experiments.append(tuple(b_fine.observations))

        c_param = next(p for p in research.parameters if p.get("parameter_kind") == "predicate_threshold")
        final_iv = c_param["radius_intervals"][0]
        final_width = final_iv[1] - final_iv[0]
        assert final_iv[0] <= true_thresh <= final_iv[1]
        assert final_width < init_width * 1e-3
        final_prog, _ = research._program(research.values)
        eval_rows = lab.evaluation_rows(("fine",), seed=8801, length=48, trajectories=2)
        assert loss(final_prog, eval_rows, 0.0, 0.5) < 1e-18


def test_unified_agent_cross_scale_shifted_threshold_transfer():
    """Verify N=4 -> N=5 cross-scale transfer on AnonymousObjectLab with shifted open-grammar threshold."""
    v4 = _schema(4, "src4")
    lib = KnowledgeLibrary()
    lib.meta_priors = [
        _make_custom_threshold_prior(v4, concept_id="g1", threshold=0.25, radius=0.91),
        _make_custom_threshold_prior(v4, concept_id="g2", threshold=0.25, radius=0.98),
    ]
    cfg = ResearchConfig(
        observation_budget=380,
        compute_budget=460000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        active_threshold_probing=True,
    )
    agent = UnifiedResearchAgent(cfg, measurement_budget=8200, library=lib)
    with AnonymousObjectLab(
        seed=710501,
        objects=5,
        readout="total_event",
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=0.44,
        relation_radius=1.08,
        actuator_curvature=0.14,
        name_seed=31,
        wiring_seed=41,
    ) as lab:
        res = agent.run(lab)
        assert res["status"] == "completed"
        prog = Program.from_dict(res["program"])
        channels = tuple(res.get("selected_channels") or ())
        eval_rows = lab.evaluation_rows(channels, seed=910501, length=48, trajectories=1)
        mse = loss(prog, eval_rows, res["fallback"], 0.5)
        assert mse <= 1e-7


def test_pomdp_latent_transition_audit_and_cross_scale_transfer():
    """Verify P_markov_state refutation, transition deflation, consolidation, and N=4->N=5 POMDP transfer."""
    with AnonymousObjectLab(
        seed=720401,
        objects=4,
        readout="total_event",
        relation_direction="right",
        latent_mode="transition_memory",
        latent_decay=0.67,
        latent_weight=0.42,
        relation_radius=0.96,
        actuator_curvature=0.14,
        name_seed=53,
        wiring_seed=67,
    ) as lab:
        raw_spec = lab.spec_for(("fine",))
        budget = SearchBudget(120000)
        records = []

        def acquire_p(p: Probe):
            b = lab.acquire(p, ("fine",))
            for idx, (pt, obs, tr) in enumerate(zip(p.sequence, b.observations, b.transitions)):
                records.append(
                    {
                        "controls": dict(pt),
                        **tr,
                        "target": obs.target,
                        "history": list(p.sequence[:idx]),
                        "probe_id": p.probe_id,
                        "position": idx,
                    }
                )

        acquire_p(lab.random_probe(101, length=16, name="rand_init"))
        model = infer_object_model(raw_spec, records, budget)
        for serial in range(16):
            p, q = propose_object_probe(raw_spec, model, records, serial)
            if p is None or q["purpose"] not in {"isolate_control_effects", "distinguish_role_definition"}:
                break
            acquire_p(p)
            model = infer_object_model(raw_spec, records, budget)

        audit = audit_transition_premises(model, records, budget, tolerance=1e-8, allow_latent_state=True)
        assert audit["status"] == "verified_latent_transition_exchange"
        assert abs(audit["latent_transition_state"]["pole"] - 0.67) < 1e-6
        assert audit["surviving_bijection_count"] == 1
        updated = apply_verified_transition_premises(model, audit)
        assert updated["complete"] is True

    # End-to-end N=4 discovery -> consolidation -> N=5 shifted-pole POMDP transfer
    v4 = _schema(4, "src4")
    markov_lib = KnowledgeLibrary()
    markov_lib.meta_priors = [
        _make_markov_right_prior(v4, concept_id="m1", radius=0.91, readout_kind="total_event"),
        _make_markov_right_prior(v4, concept_id="m2", radius=0.98, readout_kind="total_event"),
    ]

    # Without latent_state_discovery, Markovian agent fails on POMDP
    cfg_no_latent = ResearchConfig(
        observation_budget=300,
        compute_budget=240000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        latent_state_discovery=False,
    )
    with AnonymousObjectLab(
        seed=720402,
        objects=4,
        readout="total_event",
        relation_direction="right",
        latent_mode="transition_memory",
        latent_decay=0.64,
        latent_weight=0.41,
        relation_radius=0.95,
        actuator_curvature=0.14,
    ) as lab_fail:
        res_fail = UnifiedResearchAgent(cfg_no_latent, measurement_budget=6400, library=deepcopy(markov_lib)).run(lab_fail)
        mse_fail = (
            loss(
                Program.from_dict(res_fail["program"]),
                lab_fail.evaluation_rows(tuple(res_fail.get("selected_channels") or ()), seed=820402, length=48, trajectories=1),
                res_fail["fallback"],
                0.5,
            )
            if res_fail.get("program")
            else 1.0
        )
        assert mse_fail > 1e-3

    # With latent_state_discovery=True, solves N=4, consolidates, and transfers to N=5 with shifted pole 0.74!
    pomdp_lib = deepcopy(markov_lib)
    cfg_latent = ResearchConfig(
        observation_budget=320,
        compute_budget=260000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        latent_state_discovery=True,
    )
    with AnonymousObjectLab(
        seed=720402,
        objects=4,
        readout="total_event",
        relation_direction="right",
        latent_mode="transition_memory",
        latent_decay=0.64,
        latent_weight=0.41,
        relation_radius=0.95,
        actuator_curvature=0.14,
    ) as lab_n4:
        res_n4 = UnifiedResearchAgent(cfg_latent, measurement_budget=6800, library=pomdp_lib).run(lab_n4)
        assert res_n4["status"] == "completed"
        mse_n4 = loss(
            Program.from_dict(res_n4["program"]),
            lab_n4.evaluation_rows(tuple(res_n4.get("selected_channels") or ()), seed=820402, length=48, trajectories=1),
            res_n4["fallback"],
            0.5,
        )
        assert mse_n4 < 1e-12

    pomdp_comms = consolidate_world_structure_commitments(pomdp_lib.meta_priors)
    assert any(c.get("latent_state_mechanism") == "transition_memory" for c in pomdp_comms)

    # Transfer to N=5 with shifted decay pole (0.64 -> 0.74) and shifted radius (0.95 -> 1.11)
    cfg_latent_n5 = ResearchConfig(
        observation_budget=380,
        compute_budget=420000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        latent_state_discovery=True,
    )
    with AnonymousObjectLab(
        seed=720501,
        objects=5,
        readout="total_event",
        relation_direction="right",
        latent_mode="transition_memory",
        latent_decay=0.74,
        latent_weight=0.46,
        relation_radius=1.11,
        actuator_curvature=0.15,
        name_seed=71,
        wiring_seed=83,
    ) as lab_n5:
        res_n5 = UnifiedResearchAgent(cfg_latent_n5, measurement_budget=8200, library=pomdp_lib).run(lab_n5)
        assert res_n5["status"] == "completed"
        mse_n5 = loss(
            Program.from_dict(res_n5["program"]),
            lab_n5.evaluation_rows(tuple(res_n5.get("selected_channels") or ()), seed=820501, length=48, trajectories=1),
            res_n5["fallback"],
            0.5,
        )
        assert mse_n5 < 1e-12


def test_pomdp_focal_memory_readout_discovery_and_n5_transfer():
    """Verify per-object unobserved focal reservoir discovery (1D continuous pole) and N=4->N=5 transfer."""
    v4 = _schema(4, "src4")
    lib = KnowledgeLibrary()
    lib.meta_priors = [
        _make_markov_right_prior(v4, concept_id="f1", radius=0.91, readout_kind="focus"),
        _make_markov_right_prior(v4, concept_id="f2", radius=0.97, readout_kind="focus"),
    ]
    cfg = ResearchConfig(
        observation_budget=320,
        compute_budget=260000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        latent_state_discovery=True,
    )
    with AnonymousObjectLab(
        seed=730401,
        objects=4,
        readout="focus",
        relation_direction="right",
        latent_mode="focal_memory",
        latent_decay=0.63,
        latent_weight=0.39,
        relation_radius=0.94,
        actuator_curvature=0.13,
    ) as lab_n4:
        res_n4 = UnifiedResearchAgent(cfg, measurement_budget=6800, library=lib).run(lab_n4)
        assert res_n4["status"] == "completed"
        mse_n4 = loss(
            Program.from_dict(res_n4["program"]),
            lab_n4.evaluation_rows(tuple(res_n4.get("selected_channels") or ()), seed=830401, length=48, trajectories=1),
            res_n4["fallback"],
            0.5,
        )
        assert mse_n4 < 1e-12

    comms = consolidate_world_structure_commitments(lib.meta_priors)
    assert any(c.get("latent_state_mechanism") == "focal_memory" for c in comms)

    cfg_n5 = ResearchConfig(
        observation_budget=380,
        compute_budget=420000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        latent_state_discovery=True,
    )
    with AnonymousObjectLab(
        seed=730501,
        objects=5,
        readout="focus",
        relation_direction="right",
        latent_mode="focal_memory",
        latent_decay=0.72,
        latent_weight=0.47,
        relation_radius=1.09,
        actuator_curvature=0.15,
        name_seed=89,
        wiring_seed=97,
    ) as lab_n5:
        res_n5 = UnifiedResearchAgent(cfg_n5, measurement_budget=8200, library=lib).run(lab_n5)
        assert res_n5["status"] == "completed"
        mse_n5 = loss(
            Program.from_dict(res_n5["program"]),
            lab_n5.evaluation_rows(tuple(res_n5.get("selected_channels") or ()), seed=830501, length=48, trajectories=1),
            res_n5["fallback"],
            0.5,
        )
        assert mse_n5 < 1e-12
