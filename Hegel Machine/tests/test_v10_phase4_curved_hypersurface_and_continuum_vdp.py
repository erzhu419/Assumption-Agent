"""v10 Phase 4 tests: Curved Hypersurface Probing, Continuum PDE Subgraph Growth & Supercritical VdP."""

from __future__ import annotations

from math import cos, hypot

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.continuum_field_lab import (
    ContinuumFieldLab,
    audit_continuum_field_premises,
    consolidate_continuum_field_commitments,
)
from hegel_machine.cognition.curved_hypersurface_probing import (
    calibrate_hypersurface_on_transitions,
    describe_linear_in_params_hypersurface,
    extract_hypersurface_polytope_2d,
    plan_coupled_hypersurface_probe,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.premise_dependency_audit import (
    audit_transition_premises,
    extract_necessary_active_transfer_geometry,
)
from hegel_machine.cognition.program_condition_sets import solve_compound_region
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
    instantiate_commitment_on_schema,
)


def _distance_attenuated_predicate(c0: float = 0.18, c1: float = 0.42) -> Program:
    qs = Program("arg", name="edge_source_quantity")
    qd = Program("arg", name="edge_target_quantity")
    xs = Program("arg", name="edge_source_position")
    xd = Program("arg", name="edge_target_position")
    dx = Program("sub", children=(xd, xs))
    dx2 = Program("mul", children=(dx, dx))
    rhs = Program(
        "sub",
        children=(
            Program("sub", children=(qs, qd)),
            Program("mul", children=(Program("const", value=c1), dx2)),
        ),
    )
    return Program(
        "and",
        children=(
            Program("lt", children=(Program("const", value=0.0), qs)),
            Program("lt", children=(Program("const", value=c0), rhs)),
        ),
    )


def _bilinear_barrier_predicate(c0: float = 0.16, c1: float = 0.55) -> Program:
    qs = Program("arg", name="edge_source_quantity")
    qd = Program("arg", name="edge_target_quantity")
    qs2 = Program("mul", children=(qs, qs))
    qsqd = Program("mul", children=(qs, qd))
    rhs = Program(
        "sub",
        children=(
            qs2,
            Program("mul", children=(Program("const", value=c1), qsqd)),
        ),
    )
    return Program(
        "and",
        children=(
            Program("lt", children=(qd, qs)),
            Program("lt", children=(Program("const", value=c0), rhs)),
        ),
    )


def test_solve_compound_region_handles_zero_jacobian_quadratic_saddle():
    qs = Program("feature", name="q_a")
    qd = Program("feature", name="q_b")
    quad = Program(
        "sub",
        children=(
            Program("mul", children=(qs, qs)),
            Program("mul", children=(Program("const", value=0.6), Program("mul", children=(qs, qd)))),
        ),
    )
    gap = Program("sub", children=(qs, qd))
    region = {
        "constraints": [
            {
                "expression": gap,
                "measured_expression": gap,
                "lower": 0.05,
                "upper": None,
                "target": 0.4,
                "lower_inclusive": False,
                "upper_inclusive": False,
            },
            {
                "expression": quad,
                "measured_expression": quad,
                "lower": 0.18,
                "upper": 0.55,
                "target": 0.32,
                "lower_inclusive": False,
                "upper_inclusive": False,
            },
        ]
    }
    # Start exactly at the zero-Jacobian saddle (0.0, 0.0)!
    pt0 = {"q_a": 0.0, "q_b": 0.0}
    bounds = {"q_a": (-1.0, 1.0), "q_b": (-1.0, 1.0)}
    budget = SearchBudget(10000)
    solved, meta = solve_compound_region(region, pt0, bounds, budget.spend)
    assert solved is not None
    assert meta["solver_strategy"] == "compound_conditions"
    assert solved["q_a"] - solved["q_b"] > 0.05
    v_quad = solved["q_a"] ** 2 - 0.6 * solved["q_a"] * solved["q_b"]
    assert 0.18 < v_quad < 0.55


def test_phase4a_coupled_hypersurface_probing_and_cross_scale_transfer():
    # 1. Verify active coupled diametral probing crushes polytope diameter vs passive & 1D bisection
    for cond_name, pred_builder, true_c0, true_c1 in (
        ("distance_attenuated_gap", _distance_attenuated_predicate, 0.21, 0.46),
        ("bilinear_product_barrier", _bilinear_barrier_predicate, 0.19, 0.62),
    ):
        template_pred = pred_builder(0.10, 0.20)
        desc = describe_linear_in_params_hypersurface(template_pred)
        assert desc is not None and desc["dimension"] == 2

        def _run_mode(mode_name: str):
            lab = UnifiedObjectLab(
                seed=701,
                objects=4,
                readout="total_event",
                relation_direction="right",
                relation_condition=cond_name,
                relation_radius=1.08,
                condition_threshold=true_c0,
                condition_curvature=true_c1,
                actuator_curvature=0.0,
            )
            schema = lab._schema(("fine",))
            q_fields = [schema.fields[g]["q"] for g in schema.objects]
            u_fields = [schema.fields[g]["u"] for g in schema.objects]
            x_fields = [schema.fields[g]["x"] for g in schema.objects]

            batches = [
                lab.acquire(lab.random_probe(801, length=24, name="init_0"), ("fine",)),
                lab.acquire(lab.random_probe(802, length=24, name="init_1"), ("fine",)),
            ]

            def _state():
                qr, xr, dr = [], [], []
                for b in batches:
                    for tr in b.transitions:
                        bef, aft = tr["before"], tr["after"]
                        qr.append([float(bef[f]) for f in q_fields])
                        xr.append([float(bef[f]) for f in x_fields])
                        dr.append(
                            [float(aft[qf]) - float(bef[qf]) - float(bef[uf]) for qf, uf in zip(q_fields, u_fields)]
                        )
                bud = SearchBudget(200000)
                id_bij = tuple(range(4))
                ng = extract_necessary_active_transfer_geometry(qr, xr, dr, [id_bij], bud, 1e-10)
                cal = calibrate_hypersurface_on_transitions(
                    template_pred, qr, xr, dr, q_fields, x_fields, ng, ("right",), budget=bud, eff_tol=1e-10
                )
                assert cal is not None and cal["supported"]
                return cal

            if mode_name == "passive":
                # Spend 14 extra random rows passively
                batches.append(lab.acquire(lab.random_probe(803, length=14, name="passive_extra"), ("fine",)))
                return _state()

            for step_k in range(14):
                st = _state()
                poly = st["hypersurface_polytope"]
                r_iv = st["matches"][0]["radius_interval"]
                plan = plan_coupled_hypersurface_probe(
                    template_pred,
                    poly,
                    r_iv,
                    "right",
                    schema.controls,
                    lab.spec.feature_bounds,
                    mode="coupled_diametral" if mode_name == "coupled" else "axis_aligned_1d",
                    axis_toggle=step_k,
                    probe_id=f"{mode_name}_{step_k}",
                )
                assert plan is not None
                batches.append(lab.acquire(Probe(plan["probe_id"], (plan["point"],)), ("fine",)))
            return _state()

        res_passive = _run_mode("passive")
        res_1d = _run_mode("1d")
        res_coupled = _run_mode("coupled")

        c_passive = res_passive["hypersurface_polytope"]["chebyshev_center"]
        c_coupled = res_coupled["hypersurface_polytope"]["chebyshev_center"]
        err_passive = hypot(c_passive[0] - true_c0, c_passive[1] - true_c1)
        err_coupled = hypot(c_coupled[0] - true_c0, c_coupled[1] - true_c1)

        assert res_coupled["hypersurface_polytope"]["diameter"] < res_passive["hypersurface_polytope"]["diameter"] * 0.25
        assert res_coupled["hypersurface_polytope"]["diameter"] < res_1d["hypersurface_polytope"]["diameter"] * 0.45
        assert err_coupled < 0.025
        assert err_coupled < err_passive

    # 2. Verify anonymous N=4 -> N=5 3-parameter shifted curved hypersurface transfer via UnifiedResearchAgent
    src_lab = UnifiedObjectLab(
        seed=711,
        objects=4,
        readout="total_event",
        relation_direction="right",
        relation_condition="distance_attenuated_gap",
        relation_radius=0.96,
        condition_threshold=0.16,
        condition_curvature=0.36,
    )
    src_schema = src_lab._schema(("fine",))
    b0 = src_lab.acquire(src_lab.random_probe(901, length=32, name="s0"), ("fine",))
    b1 = src_lab.acquire(src_lab.random_probe(902, length=24, name="s1"), ("fine",))

    # Build source world commitment with distance_attenuated_gap predicate
    pred_src = _distance_attenuated_predicate(0.16, 0.36)
    tot_q = Program("object_sum", children=tuple(Program("feature", name=src_schema.fields[g]["q"]) for g in src_schema.objects))
    tot_u = Program("object_sum", children=tuple(Program("feature", name=src_schema.fields[g]["u"]) for g in src_schema.objects))
    rel_src = Program(
        "relation_sum",
        name="right",
        children=(
            Program("const", value=0.96),
            pred_src,
            *(
                Program("feature", name=src_schema.fields[g][r])
                for g in src_schema.objects
                for r in ("f", "x", "q")
            ),
        ),
    )
    evt_rel = Program("event_delta", lag=1, children=(rel_src,))
    raw_comm = {
        "commitment_id": "world_commitment:curved_0001",
        "kind": "world_structure_commitment",
        "reorganization_type": "equivalence_reorganization",
        "source_concept_ids": [],
        "source_arities": [4],
        "family": "joint_event",
        "family_id": "T11",
        "role_terms": [
            {"kind": "all_object_sum", "summand": Program("feature", name="object_role:q").to_dict(), "roles": ["q"]},
            {"kind": "all_object_sum", "summand": Program("feature", name="object_role:u").to_dict(), "roles": ["u"]},
            {
                "kind": "program_op",
                "op": "event_delta",
                "name": None,
                "value": None,
                "lag": 1,
                "children": [
                    {
                        "kind": "all_relation_sum",
                        "direction": "right",
                        "symbol": "r_0",
                        "predicate": pred_src.to_dict(),
                    }
                ],
            },
        ],
        "radius_parameters": [
            {
                "symbol": "r_0",
                "direction": "right",
                "edge_condition": pred_src.to_dict(),
                "binding": "saved_parameter_symbol",
                "source_values": [0.96],
                "source_intervals": [[0.92, 1.00]],
            }
        ],
        "support_premises": [],
    }
    lib = KnowledgeLibrary()
    cfg = ResearchConfig(
        observation_budget=380,
        compute_budget=420000,
        minimum_improvement=1e-7,
        world_structure_commitments=True,
        transition_premise_audit=True,
    )
    agent = UnifiedResearchAgent(cfg, measurement_budget=8200, library=lib)
    bud_src = SearchBudget(120000)
    src_entries = instantiate_commitment_on_schema(
        src_lab.spec_for(("fine",)),
        b0.observations,
        src_schema,
        raw_comm,
        bud_src,
        transitions=b0.transitions,
    )
    assert src_entries and src_entries[0]["training_loss"] <= 1e-12
    agent.remember_confirmed_selection(
        src_entries[0],
        src_lab.spec_for(("fine",)),
        b0.observations,
        b1.observations,
        confirmed=True,
        confirmation_loss=0.0,
        confirmation_rows=len(b1.observations),
        selected_channels=("fine",),
    )
    consolidated = consolidate_world_structure_commitments(lib.meta_priors, budget=SearchBudget(50000))
    assert len(consolidated) == 1

    # Now transfer to anonymous N=5 with shifted (r_0=1.14, c_0=0.26, c_1=0.52) with ZERO LLM calls!
    with AnonymousObjectLab(
        seed=712,
        objects=5,
        readout="total_event",
        relation_direction="right",
        relation_condition="distance_attenuated_gap",
        relation_radius=1.14,
        condition_threshold=0.26,
        condition_curvature=0.52,
        name_seed=9101,
        wiring_seed=42,
    ) as target_lab:
        res_n5 = agent.run(target_lab)
        assert res_n5["status"] == "completed"
        assert res_n5["selected_channels"] == []
        ev_rows = target_lab.evaluation_rows((), seed=171200, length=48, trajectories=1)
        mse_n5 = loss(Program.from_dict(res_n5["program"]), ev_rows, res_n5["fallback"], 0.5)
        assert mse_n5 <= 1e-7


def test_phase4b1_continuum_pde_subgraph_growth_and_n5_to_n7_transfer():
    # 1. Verify P_role_partition refutation + P_continuum_topology growth + N=5 -> N=7 transfer
    lib = KnowledgeLibrary()
    cfg = ResearchConfig(
        observation_budget=420,
        compute_budget=460000,
        minimum_improvement=1e-7,
        world_structure_commitments=True,
        transition_premise_audit=True,
    )
    agent = UnifiedResearchAgent(cfg, measurement_budget=9400, library=lib)

    with ContinuumFieldLab(
        seed=811,
        sites=5,
        readout="focal_pde_event",
        diffusion=0.14,
        advection=0.08,
        reaction_sink=0.05,
        name_seed=5101,
        wiring_seed=17,
    ) as lab_n5:
        res_n5 = agent.run(lab_n5)
        assert res_n5["status"] == "completed"
        assert res_n5["selected_channels"] == []
        ev5 = lab_n5.evaluation_rows((), seed=181100, length=48, trajectories=1)
        assert loss(Program.from_dict(res_n5["program"]), ev5, res_n5["fallback"], 0.5) <= 1e-10

    comms = consolidate_continuum_field_commitments(lib.meta_priors, budget=SearchBudget(40000))
    assert len(comms) == 1
    assert comms[0]["equivalence_audit"]["accepted"] is True

    # Cross-resolution transfer to N=7 (7! = 5,040 site permutations x 5,040 focus permutations!) with shifted PDE constants
    with ContinuumFieldLab(
        seed=812,
        sites=7,
        readout="focal_pde_event",
        diffusion=0.17,
        advection=0.11,
        reaction_sink=0.03,
        name_seed=7703,
        wiring_seed=93,
    ) as lab_n7:
        res_n7 = agent.run(lab_n7)
        assert res_n7["status"] == "completed"
        assert res_n7["selected_channels"] == []
        ev7 = lab_n7.evaluation_rows((), seed=181200, length=48, trajectories=1)
        assert loss(Program.from_dict(res_n7["program"]), ev7, res_n7["fallback"], 0.5) <= 1e-10
        assert any(a["action"] == "apply_continuum_field_commitments" and a["supported"] for a in res_n7["actions"])


def test_phase4b2_supercritical_van_der_pol_limit_cycle_discovery_and_transfer():
    # Establish a parent conservative commitment on N=4 first, then discover supercritical VdP (rho=1.02 > 1.0)
    # on anonymous N=4 and transfer to shifted supercritical VdP (rho=1.04 > 1.0) on anonymous N=5!
    from hegel_machine.cognition.continuous_state import _sum_terms
    from hegel_machine.cognition.research_programs import _plan
    from hegel_machine.cognition.structure_lab import ObjectSchema

    objs4 = tuple(f"src4_{i}" for i in range(4))
    v4 = ObjectSchema(
        objs4,
        {item: {role: f"{role}:{item}" for role in ("q", "u", "x", "f")} for item in objs4},
        {item: (item,) for item in objs4},
        {item: {role: f"ctl:{role}:{item}" for role in ("q", "u", "x")} for item in objs4},
    )

    def _mk_prior(concept_id: str, radius: float) -> dict:
        triples = tuple(
            Program("feature", name=v4.fields[item][role])
            for item in v4.objects
            for role in ("f", "x", "q")
        )
        tot_q = Program("object_sum", children=tuple(Program("feature", name=v4.fields[i]["q"]) for i in v4.objects))
        tot_u = Program("object_sum", children=tuple(Program("feature", name=v4.fields[i]["u"]) for i in v4.objects))
        sym_terms = (
            tot_q,
            tot_u,
            Program(
                "event_delta",
                lag=1,
                children=(Program("neighbor_sum", name="right", children=(Program("arg", name="r_0"), *triples)),),
            ),
        )
        act_terms = (
            tot_q,
            tot_u,
            Program(
                "event_delta",
                lag=1,
                children=(Program("neighbor_sum", name="right", children=(Program("const", value=radius), *triples)),),
            ),
        )
        weights = [1.0, 1.0, 0.16]
        act_prog = _sum_terms(act_terms, weights, 0.0)
        sym_prog = _sum_terms(sym_terms, weights, 0.0)
        fit = {"fit_terms": act_terms, "coefficients": [0.0, *weights], "program": act_prog, "loss": 0.0}
        plan = _plan(
            fit,
            [
                {
                    "operation": "resolve_relation_intervals",
                    "question": "Prior radius binding",
                    "parameter_template": sym_prog.to_dict(),
                    "parameter_terms": [t.to_dict() for t in sym_terms],
                    "parameters": [
                        {
                            "symbol": "r_0",
                            "parameter_kind": "radius",
                            "radius": radius,
                            "radius_intervals": [[radius - 0.04, radius + 0.04]],
                            "prior_radius_intervals": [[radius - 0.04, radius + 0.04]],
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
            "family_id": "T11",
            "kind": "structural:joint_event",
            "status": "supported",
            "task_id": f"task:{concept_id}",
            "object_schema": v4.to_dict(),
            "research_program": plan,
            "counterexamples": [],
        }

    lib = KnowledgeLibrary()
    lib.meta_priors = [_mk_prior("m_te_1", 0.91), _mk_prior("m_te_2", 0.98)]
    cfg = ResearchConfig(
        observation_budget=400,
        compute_budget=460000,
        minimum_improvement=1e-7,
        world_structure_commitments=True,
        transition_premise_audit=True,
    )
    agent = UnifiedResearchAgent(cfg, measurement_budget=8800, library=lib)

    # Anonymous N=4 with supercritical Van der Pol (rho = 1.02 > 1.0, linearly unstable at origin!)
    with AnonymousObjectLab(
        seed=922,
        objects=4,
        readout="total_event",
        relation_direction="right",
        relation_radius=0.98,
        latent_mode="van_der_pol_transition_memory",
        latent_decay=1.02,
        latent_freq=0.65,
        latent_nonlinearity=0.28,
        latent_weight=0.44,
        name_seed=3201,
        wiring_seed=23,
    ) as vdp_n4:
        res_vdp4 = agent.run(vdp_n4)
        assert res_vdp4["status"] == "completed"
        assert res_vdp4["selected_channels"] == []
        ev4 = vdp_n4.evaluation_rows((), seed=192200, length=48, trajectories=1)
        assert loss(Program.from_dict(res_vdp4["program"]), ev4, res_vdp4["fallback"], 0.5) <= 1e-10

    vdp_comms = consolidate_world_structure_commitments(lib.meta_priors, budget=SearchBudget(50000))
    assert any(c.get("latent_state_mechanism") == "van_der_pol_transition_memory" for c in vdp_comms)

    # Anonymous N=5 cross-scale transfer with shifted supercritical VdP parameters (rho=1.04, freq=0.58, beta=0.33, r=1.12)
    with AnonymousObjectLab(
        seed=923,
        objects=5,
        readout="total_event",
        relation_direction="right",
        relation_radius=1.12,
        latent_mode="van_der_pol_transition_memory",
        latent_decay=1.04,
        latent_freq=0.58,
        latent_nonlinearity=0.33,
        latent_weight=0.48,
        name_seed=3301,
        wiring_seed=37,
    ) as vdp_n5:
        res_vdp5 = agent.run(vdp_n5)
        assert res_vdp5["status"] == "completed"
        assert res_vdp5["selected_channels"] == []
        ev5_v = vdp_n5.evaluation_rows((), seed=192300, length=48, trajectories=1)
        assert loss(Program.from_dict(res_vdp5["program"]), ev5_v, res_vdp5["fallback"], 0.5) <= 1e-10
