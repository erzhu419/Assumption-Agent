"""Verification suite for v10 Phase 3A (2nd-order oscillatory & spatially coupled POMDPs)
and Phase 3B (joint Open-Grammar + POMDP latent-state equation discovery and N=4->N=5 transfer)."""

from copy import deepcopy
from math import cos

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from hegel_machine.cognition.open_grammar_author import propose_premise_gated_commitment_revisions
from hegel_machine.cognition.premise_dependency_audit import (
    apply_verified_transition_premises,
    audit_transition_premises,
)
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
    expand_role_terms_for_ar2_focal_readout,
    expand_role_terms_for_coupled_latent_transition,
    instantiate_commitment_terms,
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


def _collect_pomdp_roles_and_trajectories(
    lab: AnonymousObjectLab,
    budget: SearchBudget,
    *,
    include_focus_probes: bool = False,
) -> tuple[dict, list[dict], list[Observation], list[Observation]]:
    """Collect role sweeps + multi-step trajectories for POMDP open-grammar audits."""
    raw_spec = lab.spec_for(("fine",))
    records: list[dict] = []

    def acquire_p(probe: Probe) -> list[Observation]:
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
        return list(batch.observations)

    acquire_p(lab.random_probe(901, length=12, name="rand_0"))
    model = infer_object_model(raw_spec, records, budget)
    allowed_purposes = {"isolate_control_effects", "distinguish_role_definition"}
    if include_focus_probes:
        allowed_purposes |= {"observe_missing_membership_states", "distinguish_membership_dependency"}
    for serial in range(28 if include_focus_probes else 14):
        probe, question = propose_object_probe(raw_spec, model, records, serial)
        if probe is None or question["purpose"] not in allowed_purposes:
            break
        acquire_p(probe)
        model = infer_object_model(raw_spec, records, budget)

    train_obs: list[Observation] = []
    for tr_idx in range(3):
        train_obs.extend(
            acquire_p(lab.random_probe(710 + tr_idx, length=32, name=f"train_traj_{tr_idx}"))
        )
    val_obs = list(lab.observe(lab.random_probe(820, length=32, name="val_traj"), ("fine",)))
    model = infer_object_model(raw_spec, records, budget)
    return model, records, train_obs, val_obs


def test_ar2_accumulate_exact_resonator_and_value_summand_lifting():
    """Verify exact causal AR(2) resonator semantics and coupled value_summand lifting."""
    rho, omega = 0.72, 0.96
    a1 = 2.0 * rho * cos(omega)
    a2 = -(rho * rho)
    weight = 0.37
    radius = 0.96

    with UnifiedObjectLab(
        seed=910101,
        objects=4,
        readout="focus",
        relation_direction="right",
        latent_mode="oscillatory_focal_memory",
        latent_decay=rho,
        latent_freq=omega,
        latent_weight=weight,
        relation_radius=radius,
        actuator_curvature=0.12,
    ) as lab:
        schema = lab._schema(("fine",))
        rows = lab.evaluation_rows(("fine",), seed=910102, length=36, trajectories=2)
        base_comm = consolidate_world_structure_commitments(
            [_make_markov_right_prior(schema, concept_id="b1", radius=radius, readout_kind="focus")]
        )[0]
        ar2_rt = expand_role_terms_for_ar2_focal_readout(base_comm["role_terms"], a1, a2)
        terms = instantiate_commitment_terms(
            base_comm, schema, radius_bindings={"r_0": radius}, role_terms_override=ar2_rt
        )
        assert terms is not None and len(terms) == 5
        # Exact analytical coefficients: q_f + u_f + coupling*neighbor_sum + weight*ar2_focal(q) + 0.5*weight*ar2_focal(u)
        exact_prog = _sum_terms(list(terms), [1.0, 1.0, lab._lab._coupling, weight, 0.5 * weight], 0.0)
        assert loss(exact_prog, rows, 0.0, 0.5) < 1e-24

    # Also verify coupled_transition_memory exact algebraic identity on UnifiedObjectLab
    pole_c = 0.66
    coup_k = 0.15
    with UnifiedObjectLab(
        seed=910201,
        objects=4,
        readout="total_event",
        relation_direction="right",
        latent_mode="coupled_transition_memory",
        latent_decay=pole_c,
        latent_weight=weight,
        latent_coupling=coup_k,
        relation_radius=radius,
        actuator_curvature=0.12,
    ) as lab_c:
        schema_c = lab_c._schema(("fine",))
        rows_c = lab_c.evaluation_rows(("fine",), seed=910202, length=36, trajectories=2)
        base_te = consolidate_world_structure_commitments(
            [_make_markov_right_prior(schema_c, concept_id="c1", radius=radius, readout_kind="total_event")]
        )[0]
        coup_rt = expand_role_terms_for_coupled_latent_transition(
            base_te["role_terms"], pole_c, "r_0", "right", {"u": 1.0, "q": 0.0}
        )
        terms_c = instantiate_commitment_terms(
            base_te, schema_c, radius_bindings={"r_0": radius}, role_terms_override=coup_rt
        )
        assert terms_c is not None and len(terms_c) >= 6
        from hegel_machine.cognition.continuous_state import _linear_fit

        cols = [[t.predict(r) for r in rows_c] for t in terms_c]
        coeffs = _linear_fit(cols, [r.target for r in rows_c])
        fitted_prog = _sum_terms(list(terms_c), coeffs[1:], coeffs[0])
        assert loss(fitted_prog, rows_c, 0.0, 0.5) < 1e-19


def test_phase3a_oscillatory_ar2_pomdp_discovery_and_n5_transfer():
    """Verify 2nd-order oscillatory transition & focal POMDP discovery and N=4->N=5 transfer."""
    v4 = _schema(4, "src4")

    for mode, readout, seed4, seed5 in (
        ("oscillatory_transition_memory", "total_event", 920401, 920501),
        ("oscillatory_focal_memory", "focus", 920402, 920502),
    ):
        base_lib = KnowledgeLibrary()
        base_lib.meta_priors = [
            _make_markov_right_prior(v4, concept_id=f"m1_{mode}", radius=0.91, readout_kind=readout),
            _make_markov_right_prior(v4, concept_id=f"m2_{mode}", radius=0.98, readout_kind=readout),
        ]

        # Single-factor ablation: Phase 2C 1st-order POMDP (higher_order_latent_discovery=False) fails!
        cfg_1st = ResearchConfig(
            observation_budget=320,
            compute_budget=280000,
            minimum_improvement=1e-7,
            max_candidates=4,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            latent_state_discovery=True,
            higher_order_latent_discovery=False,
        )
        with AnonymousObjectLab(
            seed=seed4,
            objects=4,
            readout=readout,
            relation_direction="right",
            latent_mode=mode,
            latent_decay=0.68,
            latent_freq=0.95,
            latent_weight=0.41,
            relation_radius=0.95,
            actuator_curvature=0.14,
        ) as lab_fail:
            res_1st = UnifiedResearchAgent(cfg_1st, measurement_budget=6800, library=deepcopy(base_lib)).run(lab_fail)
            mse_1st = (
                loss(
                    Program.from_dict(res_1st["program"]),
                    lab_fail.evaluation_rows(tuple(res_1st.get("selected_channels") or ()), seed=seed4 + 1000, length=48, trajectories=1),
                    res_1st["fallback"],
                    0.5,
                )
                if res_1st.get("program")
                else 1.0
            )
            assert mse_1st > 1e-3

        # Phase 3A (higher_order_latent_discovery=True) discovers on N=4 and transfers to N=5!
        lib_3a = deepcopy(base_lib)
        cfg_3a_n4 = ResearchConfig(
            observation_budget=340,
            compute_budget=340000,
            minimum_improvement=1e-7,
            max_candidates=4,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            latent_state_discovery=True,
            higher_order_latent_discovery=True,
        )
        with AnonymousObjectLab(
            seed=seed4,
            objects=4,
            readout=readout,
            relation_direction="right",
            latent_mode=mode,
            latent_decay=0.68,
            latent_freq=0.95,
            latent_weight=0.41,
            relation_radius=0.95,
            actuator_curvature=0.14,
        ) as lab_n4:
            res_n4 = UnifiedResearchAgent(cfg_3a_n4, measurement_budget=7200, library=lib_3a).run(lab_n4)
            assert res_n4["status"] == "completed"
            mse_n4 = loss(
                Program.from_dict(res_n4["program"]),
                lab_n4.evaluation_rows(tuple(res_n4.get("selected_channels") or ()), seed=seed4 + 1000, length=48, trajectories=1),
                res_n4["fallback"],
                0.5,
            )
            assert mse_n4 < 1e-10

        comms = consolidate_world_structure_commitments(lib_3a.meta_priors)
        assert any(c.get("latent_state_mechanism") == mode for c in comms)

        # Transfer to unseen N=5 with shifted damping (0.68 -> 0.76), shifted freq (0.95 -> 1.14), shifted radius (0.95 -> 1.11)
        cfg_3a_n5 = ResearchConfig(
            observation_budget=400,
            compute_budget=500000,
            minimum_improvement=1e-7,
            max_candidates=4,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            latent_state_discovery=True,
            higher_order_latent_discovery=True,
        )
        with AnonymousObjectLab(
            seed=seed5,
            objects=5,
            readout=readout,
            relation_direction="right",
            latent_mode=mode,
            latent_decay=0.76,
            latent_freq=1.14,
            latent_weight=0.46,
            relation_radius=1.11,
            actuator_curvature=0.15,
            name_seed=73,
            wiring_seed=89,
        ) as lab_n5:
            res_n5 = UnifiedResearchAgent(cfg_3a_n5, measurement_budget=8600, library=lib_3a).run(lab_n5)
            assert res_n5["status"] == "completed"
            mse_n5 = loss(
                Program.from_dict(res_n5["program"]),
                lab_n5.evaluation_rows(tuple(res_n5.get("selected_channels") or ()), seed=seed5 + 1000, length=48, trajectories=1),
                res_n5["fallback"],
                0.5,
            )
            assert mse_n5 < 1e-10


def test_phase3a_spatially_coupled_pomdp_discovery_and_n5_transfer():
    """Verify spatially coupled cross-object latent transition & focal POMDP discovery and N=4->N=5 transfer."""
    v4 = _schema(4, "src4")

    for mode, readout, seed4, seed5 in (
        ("coupled_transition_memory", "total_event", 930401, 930501),
        ("coupled_focal_memory", "focus", 930402, 930502),
    ):
        base_lib = KnowledgeLibrary()
        base_lib.meta_priors = [
            _make_markov_right_prior(v4, concept_id=f"c1_{mode}", radius=0.91, readout_kind=readout),
            _make_markov_right_prior(v4, concept_id=f"c2_{mode}", radius=0.98, readout_kind=readout),
        ]

        # Single-factor ablation: 1st-order independent POMDP (higher_order_latent_discovery=False)
        # Note: for total_event readout, sum_i coupled_flux_i = 0 by conservation of directed latent flux!
        # Wait: in coupled_transition_memory, does total_event readout cancel coupled_flux?!
        # LET'S CHECK: total_event = sum_i q'_i + alpha_ev * event_delta(right_neighbor_sum(q))!
        # In event_delta(right_neighbor_sum(q)), q_{t} depends on q_{t-1} + coupled_flux(z_{t-2})!
        # So right_neighbor_sum(q_t) DOES depend on individual object q_{j,t} which DOES depend on coupled_flux!
        cfg_1st = ResearchConfig(
            observation_budget=320,
            compute_budget=280000,
            minimum_improvement=1e-7,
            max_candidates=4,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            latent_state_discovery=True,
            higher_order_latent_discovery=False,
        )
        with AnonymousObjectLab(
            seed=seed4,
            objects=4,
            readout=readout,
            relation_direction="right",
            latent_mode=mode,
            latent_decay=0.65,
            latent_weight=0.39,
            latent_coupling=0.22,
            relation_radius=0.96,
            actuator_curvature=0.14,
        ) as lab_fail:
            res_1st = UnifiedResearchAgent(cfg_1st, measurement_budget=6800, library=deepcopy(base_lib)).run(lab_fail)
            mse_1st = (
                loss(
                    Program.from_dict(res_1st["program"]),
                    lab_fail.evaluation_rows(tuple(res_1st.get("selected_channels") or ()), seed=seed4 + 1000, length=48, trajectories=1),
                    res_1st["fallback"],
                    0.5,
                )
                if res_1st.get("program")
                else 1.0
            )
            assert mse_1st > 1e-5

        # Phase 3A (higher_order_latent_discovery=True) solves N=4 and transfers to N=5!
        lib_3a = deepcopy(base_lib)
        cfg_3a_n4 = ResearchConfig(
            observation_budget=340,
            compute_budget=340000,
            minimum_improvement=1e-7,
            max_candidates=4,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            latent_state_discovery=True,
            higher_order_latent_discovery=True,
        )
        with AnonymousObjectLab(
            seed=seed4,
            objects=4,
            readout=readout,
            relation_direction="right",
            latent_mode=mode,
            latent_decay=0.65,
            latent_weight=0.39,
            latent_coupling=0.22,
            relation_radius=0.96,
            actuator_curvature=0.14,
        ) as lab_n4:
            res_n4 = UnifiedResearchAgent(cfg_3a_n4, measurement_budget=7200, library=lib_3a).run(lab_n4)
            assert res_n4["status"] == "completed"
            mse_n4 = loss(
                Program.from_dict(res_n4["program"]),
                lab_n4.evaluation_rows(tuple(res_n4.get("selected_channels") or ()), seed=seed4 + 1000, length=48, trajectories=1),
                res_n4["fallback"],
                0.5,
            )
            assert mse_n4 < 1e-11

        comms = consolidate_world_structure_commitments(lib_3a.meta_priors)
        assert any(c.get("latent_state_mechanism") == mode for c in comms)

        cfg_3a_n5 = ResearchConfig(
            observation_budget=400,
            compute_budget=500000,
            minimum_improvement=1e-7,
            max_candidates=4,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            latent_state_discovery=True,
            higher_order_latent_discovery=True,
        )
        with AnonymousObjectLab(
            seed=seed5,
            objects=5,
            readout=readout,
            relation_direction="right",
            latent_mode=mode,
            latent_decay=0.74,
            latent_weight=0.45,
            latent_coupling=0.26,
            relation_radius=1.12,
            actuator_curvature=0.15,
            name_seed=79,
            wiring_seed=97,
        ) as lab_n5:
            res_n5 = UnifiedResearchAgent(cfg_3a_n5, measurement_budget=8600, library=lib_3a).run(lab_n5)
            assert res_n5["status"] == "completed"
            mse_n5 = loss(
                Program.from_dict(res_n5["program"]),
                lab_n5.evaluation_rows(tuple(res_n5.get("selected_channels") or ()), seed=seed5 + 1000, length=48, trajectories=1),
                res_n5["fallback"],
                0.5,
            )
            assert mse_n5 < 1e-11


def test_phase3b_open_grammar_pomdp_offline_gate_and_zero_call_n5_transfer():
    """Verify open-grammar POMDP synthesis on bilinear_work_transition_memory & quadratic_energy_focal_memory,
    followed by zero-LLM-call cross-scale N=4 -> N=5 transfer with shifted decay poles and radii."""
    v4 = _schema(4, "og_pomdp4")

    # --- Case 1: bilinear_work_transition_memory (z_{i,t} = alpha*z_{i,t-1} + q_{i,t}*u_{i,t}) ---
    comm_te = consolidate_world_structure_commitments(
        [
            _make_markov_right_prior(v4, concept_id="bw1", radius=0.92, readout_kind="total_event"),
            _make_markov_right_prior(v4, concept_id="bw2", radius=0.98, readout_kind="total_event"),
        ]
    )[0]

    wrong_drive = Program(
        "mul",
        children=(
            Program("feature", name="object_role:u"),
            Program("feature", name="object_role:u"),
        ),
    ).to_dict()
    true_drive = Program(
        "mul",
        children=(
            Program("feature", name="object_role:q"),
            Program("feature", name="object_role:u"),
        ),
    ).to_dict()

    def mock_bilinear_transport(instructions, payload, *, purpose="mock"):
        assert payload["isolated_violated_premise"] == "P_latent_transition_drive"
        return (
            {
                "candidates": [
                    {"rationale": "Wrong u^2 drive (must be rejected)", "latent_drive_ast": wrong_drive},
                    {"rationale": "Bilinear work q*u drive", "latent_drive_ast": true_drive},
                ]
            },
            {
                "backend": "mock",
                "model": "mock-gpt",
                "calls": 1,
                "input_tokens": 510,
                "output_tokens": 160,
                "total_tokens": 670,
                "wall_time_seconds": 0.01,
                "purpose": purpose,
                "status": "ok",
            },
        )

    lib_bw = KnowledgeLibrary()
    with AnonymousObjectLab(
        seed=940401,
        objects=4,
        readout="total_event",
        relation_direction="right",
        latent_mode="bilinear_work_transition_memory",
        latent_decay=0.65,
        latent_weight=0.44,
        relation_radius=0.96,
        actuator_curvature=0.14,
    ) as lab_bw4:
        budget_bw4 = SearchBudget(220000)
        m_bw4, rec_bw4, tr_bw4, val_bw4 = _collect_pomdp_roles_and_trajectories(lab_bw4, budget_bw4)
        res_bw4 = propose_premise_gated_commitment_revisions(
            comm_te,
            m_bw4,
            rec_bw4,
            lab_bw4.spec_for(("fine",)),
            tr_bw4,
            val_bw4,
            budget_bw4,
            mock_bilinear_transport,
            max_llm_calls=2,
            tolerance=1e-8,
            model_name="mock-gpt",
        )
        assert res_bw4["supported"] is True
        assert res_bw4["isolated_violated_premise"] == "P_latent_transition_drive"
        assert res_bw4["llm_ledger"]["model_calls"] == 1
        best_bw4 = res_bw4["best_entry"]
        assert best_bw4["validation_loss"] <= 1e-11

        agent_save_bw = UnifiedResearchAgent(
            ResearchConfig(world_structure_commitments=True, transition_premise_audit=True),
            library=lib_bw,
        )
        agent_save_bw._object_schemas[("fine",)] = deepcopy(best_bw4["resolved_schema"])
        saved_bw = agent_save_bw.remember_confirmed_selection(
            best_bw4,
            agent_save_bw._spec(lab_bw4, ("fine",)),
            tr_bw4,
            val_bw4,
            confirmed=True,
            confirmation_loss=best_bw4["validation_loss"],
            confirmation_rows=len(val_bw4),
            selected_channels=("fine",),
        )
        assert saved_bw["stored"] is True

    bw_comms = consolidate_world_structure_commitments(lib_bw.meta_priors)
    assert len(bw_comms) == 1
    assert bw_comms[0]["latent_state_mechanism"] == "custom_transition_memory"

    # Zero-LLM-call cross-scale transfer to N=5 with shifted decay (0.65 -> 0.74) & radius (0.96 -> 1.12)!
    cfg_bw_n5 = ResearchConfig(
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
    with AnonymousObjectLab(
        seed=940501,
        objects=5,
        readout="total_event",
        relation_direction="right",
        latent_mode="bilinear_work_transition_memory",
        latent_decay=0.74,
        latent_weight=0.48,
        relation_radius=1.12,
        actuator_curvature=0.15,
        name_seed=67,
        wiring_seed=83,
    ) as lab_bw5:
        res_bw5 = UnifiedResearchAgent(cfg_bw_n5, measurement_budget=8600, library=lib_bw).run(lab_bw5)
        assert res_bw5["status"] == "completed"
        mse_bw5 = loss(
            Program.from_dict(res_bw5["program"]),
            lab_bw5.evaluation_rows(tuple(res_bw5.get("selected_channels") or ()), seed=950501, length=48, trajectories=1),
            res_bw5["fallback"],
            0.5,
        )
        assert mse_bw5 < 1e-10

    # --- Case 2: quadratic_energy_focal_memory (z_{i,t} = alpha*z_{i,t-1} + q_{i,t}^2) ---
    comm_foc = consolidate_world_structure_commitments(
        [
            _make_markov_right_prior(v4, concept_id="qe1", radius=0.92, readout_kind="focus"),
            _make_markov_right_prior(v4, concept_id="qe2", radius=0.98, readout_kind="focus"),
        ]
    )[0]

    # LLM proposes rough pole 0.50; gate recalibrates 1D pole to true 0.66 automatically!
    quad_acc_term = {
        "kind": "all_object_sum",
        "summand": Program(
            "mul",
            children=(
                Program("feature", name="object_role:f"),
                Program(
                    "accumulate",
                    value=0.50,
                    children=(
                        Program(
                            "mul",
                            children=(
                                Program("feature", name="object_role:q"),
                                Program("feature", name="object_role:q"),
                            ),
                        ),
                        Program("const", value=0.0),
                    ),
                ),
            ),
        ).to_dict(),
    }

    def mock_quad_focal_transport(instructions, payload, *, purpose="mock"):
        assert payload["isolated_violated_premise"] == "P_temporal_readout"
        return (
            {
                "candidates": [
                    {
                        "rationale": "Focused leaky integrator of q^2 energy",
                        "extra_role_terms": [quad_acc_term],
                    }
                ]
            },
            {
                "backend": "mock",
                "model": "mock-gpt",
                "calls": 1,
                "input_tokens": 490,
                "output_tokens": 150,
                "total_tokens": 640,
                "wall_time_seconds": 0.01,
                "purpose": purpose,
                "status": "ok",
            },
        )

    lib_qe = KnowledgeLibrary()
    with AnonymousObjectLab(
        seed=940402,
        objects=4,
        readout="focus",
        relation_direction="right",
        latent_mode="quadratic_energy_focal_memory",
        latent_decay=0.66,
        latent_weight=0.42,
        relation_radius=0.95,
        actuator_curvature=0.14,
    ) as lab_qe4:
        budget_qe4 = SearchBudget(220000)
        m_qe4, rec_qe4, tr_qe4, val_qe4 = _collect_pomdp_roles_and_trajectories(
            lab_qe4, budget_qe4, include_focus_probes=True
        )
        res_qe4 = propose_premise_gated_commitment_revisions(
            comm_foc,
            m_qe4,
            rec_qe4,
            lab_qe4.spec_for(("fine",)),
            tr_qe4,
            val_qe4,
            budget_qe4,
            mock_quad_focal_transport,
            max_llm_calls=2,
            tolerance=1e-8,
            model_name="mock-gpt",
        )
        assert res_qe4["supported"] is True
        assert res_qe4["isolated_violated_premise"] == "P_temporal_readout"
        best_qe4 = res_qe4["best_entry"]
        assert best_qe4["validation_loss"] <= 1e-10

        agent_save_qe = UnifiedResearchAgent(
            ResearchConfig(world_structure_commitments=True, transition_premise_audit=True),
            library=lib_qe,
        )
        agent_save_qe._object_schemas[("fine",)] = deepcopy(best_qe4["resolved_schema"])
        saved_qe = agent_save_qe.remember_confirmed_selection(
            best_qe4,
            agent_save_qe._spec(lab_qe4, ("fine",)),
            tr_qe4,
            val_qe4,
            confirmed=True,
            confirmation_loss=best_qe4["validation_loss"],
            confirmation_rows=len(val_qe4),
            selected_channels=("fine",),
        )
        assert saved_qe["stored"] is True

    qe_comms = consolidate_world_structure_commitments(lib_qe.meta_priors)
    assert len(qe_comms) == 1
    assert qe_comms[0]["latent_state_mechanism"] == "custom_focal_memory"

    # Zero-LLM-call cross-scale transfer to N=5 with shifted decay (0.66 -> 0.75) & radius (0.95 -> 1.10)!
    cfg_qe_n5 = ResearchConfig(
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
    with AnonymousObjectLab(
        seed=940502,
        objects=5,
        readout="focus",
        relation_direction="right",
        latent_mode="quadratic_energy_focal_memory",
        latent_decay=0.75,
        latent_weight=0.47,
        relation_radius=1.10,
        actuator_curvature=0.15,
        name_seed=71,
        wiring_seed=89,
    ) as lab_qe5:
        res_qe5 = UnifiedResearchAgent(cfg_qe_n5, measurement_budget=8600, library=lib_qe).run(lab_qe5)
        assert res_qe5["status"] == "completed"
        mse_qe5 = loss(
            Program.from_dict(res_qe5["program"]),
            lab_qe5.evaluation_rows(tuple(res_qe5.get("selected_channels") or ()), seed=950502, length=48, trajectories=1),
            res_qe5["fallback"],
            0.5,
        )
        assert mse_qe5 < 1e-10
