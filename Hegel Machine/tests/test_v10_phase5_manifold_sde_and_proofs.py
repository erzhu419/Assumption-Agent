"""Unit & integration tests for v10 Phase 5 (5A Manifold PDEs, 5B SDE IV & Resonance, 5C Symbolic Proofs)."""

from __future__ import annotations

from copy import deepcopy

from hegel_machine.cognition.algebraic_proof_certificates import (
    BudgetedControlLab,
    SymbolicProofKernel,
    prove_task_solvability,
    repair_proof_from_discrepancy,
    synthesize_emergent_invariants,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.manifold_pde_lab import (
    ManifoldPDELab,
    consolidate_manifold_pde_commitments,
)
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.stochastic_premise_audit import (
    audit_stochastic_transition_premises,
)
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def test_manifold_sum_serialization_and_torus_gauge_invariance():
    nx, ny = 3, 3
    pairs = tuple(
        part
        for k in range(nx * ny)
        for part in (Program("feature", name=f"f_{k}"), Program("feature", name=f"q_{k}"))
    )
    prog = Program("manifold_sum", name="torus2d_lap_x:3x3", children=pairs)
    rebuilt = Program.from_dict(prog.to_dict()).simplified()
    assert rebuilt == prog

    # Shift coordinates by (+1 mod 3, +2 mod 3) and verify torus translation invariance
    shifted_pairs = tuple(
        part
        for r in range(nx)
        for c in range(ny)
        for k in ((((r + 1) % nx) * ny + ((c + 2) % ny)),)
        for part in (Program("feature", name=f"f_{k}"), Program("feature", name=f"q_{k}"))
    )
    shifted_prog = Program("manifold_sum", name="torus2d_lap_x:3x3", children=shifted_pairs)

    feats = {f"q_{k}": 0.17 * k - 0.03 * (k * k) for k in range(nx * ny)}
    for focus_idx in range(nx * ny):
        row_f = {**feats, **{f"f_{k}": (1.0 if k == focus_idx else 0.0) for k in range(nx * ny)}}
        obs = Observation(row_f, 0.0)
        assert abs(prog.predict(obs) - shifted_prog.predict(obs)) <= 1e-15


def test_phase5a_2d_torus_discovery_and_3x3_to_4x4_transfer():
    lib = KnowledgeLibrary()
    for shape, seed, dx, dy, cx, cy, sk in (
        ((3, 3), 500911, 0.14, 0.09, 0.08, 0.04, 0.04),
        ((4, 4), 501611, 0.16, 0.10, 0.09, 0.05, 0.03),
    ):
        n_s = shape[0] * shape[1]
        cfg = ResearchConfig(
            observation_budget=560 if n_s >= 16 else 420,
            compute_budget=1400000 if n_s >= 16 else 520000,
            minimum_improvement=1e-7,
            max_candidates=4,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            continuum_field_discovery=True,
            manifold_pde_discovery=True,
        )
        with ManifoldPDELab(
            seed=seed,
            topology="torus_2d",
            grid_shape=shape,
            readout="focal_pde_event",
            diffusion_x=dx,
            diffusion_y=dy,
            advection_x=cx,
            advection_y=cy,
            reaction_sink=sk,
            actuator_curvature=0.11,
            name_seed=seed % 997 + 17,
            wiring_seed=seed % 991 + 31,
        ) as lab:
            res = UnifiedResearchAgent(
                cfg, measurement_budget=26000 if n_s >= 16 else 12000, library=lib
            ).run(lab)
            assert res["status"] == "completed"
            assert res["selected_channels"] == []
            prog = Program.from_dict(res["program"])
            mse = loss(prog, lab.evaluation_rows((), seed=seed + 50000, length=48, trajectories=1), res["fallback"], 0.5)
            assert mse <= 1e-14

    comms = consolidate_manifold_pde_commitments(lib.meta_priors, budget=SearchBudget(50000))
    assert len(comms) == 1
    assert comms[0]["topology"] == "2d_periodic_torus"
    assert comms[0]["source_resolutions"] == [9, 16]


def test_phase5a_neumann_to_dirichlet_boundary_revision_and_irregular_graph():
    # 1. N=5 Neumann discovery -> N=8 Dirichlet conservative boundary premise revision
    lib_bc = KnowledgeLibrary()
    for n_s, bc, seed in ((5, "neumann", 510511), (8, "dirichlet", 510811)):
        cfg = ResearchConfig(
            observation_budget=420,
            compute_budget=550000,
            minimum_improvement=1e-7,
            max_candidates=4,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            continuum_field_discovery=True,
            manifold_pde_discovery=True,
        )
        with ManifoldPDELab(
            seed=seed,
            topology="bounded_chain",
            sites=n_s,
            boundary_condition=bc,
            readout="focal_pde_event",
            diffusion=0.14,
            advection=0.08,
            reaction_sink=0.02,
            actuator_curvature=0.10,
            name_seed=seed % 997 + 17,
            wiring_seed=seed % 991 + 31,
        ) as lab:
            res = UnifiedResearchAgent(cfg, measurement_budget=14000, library=lib_bc).run(lab)
            assert res["status"] == "completed"
            assert res["selected_channels"] == []
            prog = Program.from_dict(res["program"])
            mse = loss(prog, lab.evaluation_rows((), seed=seed + 50000, length=48, trajectories=1), res["fallback"], 0.5)
            assert mse <= 1e-14

    # 2. N=6 -> N=9 irregular graph-Laplacian discovery & transfer
    lib_g = KnowledgeLibrary()
    for n_s, seed in ((6, 520611), (9, 520911)):
        cfg = ResearchConfig(
            observation_budget=440,
            compute_budget=580000,
            minimum_improvement=1e-7,
            max_candidates=4,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            continuum_field_discovery=True,
            manifold_pde_discovery=True,
        )
        with ManifoldPDELab(
            seed=seed,
            topology="general_graph",
            sites=n_s,
            graph_family="branched_wheel",
            readout="focal_pde_event",
            diffusion=0.15,
            reaction_sink=0.03,
            actuator_curvature=0.10,
            name_seed=seed % 997 + 17,
            wiring_seed=seed % 991 + 31,
        ) as lab:
            res = UnifiedResearchAgent(cfg, measurement_budget=15000, library=lib_g).run(lab)
            assert res["status"] == "completed"
            assert res["selected_channels"] == []
            prog = Program.from_dict(res["program"])
            mse = loss(prog, lab.evaluation_rows((), seed=seed + 50000, length=48, trajectories=1), res["fallback"], 0.5)
            assert mse <= 1e-14


def test_phase5b_stochastic_iv_vs_ols_endogeneity_and_near_critical_vdp():
    # 1. Verify 2SLS-IV overcomes Hurwicz-Nickell OLS endogeneity bias under SDE process noise
    with AnonymousObjectLab(
        seed=530411,
        objects=4,
        readout="total_event",
        relation_direction="right",
        relation_radius=0.95,
        latent_mode="transition_memory",
        latent_decay=0.75,
        latent_weight=0.50,
        process_noise_std=0.04,
    ) as lab:
        raw_spec = lab.spec_for(("fine",))
        bounds = lab.spec.feature_bounds
        center = {k: 0.5 * (lo + hi) for k, (lo, hi) in bounds.items()}
        single_pts = [center, *({**center, c: v} for c, iv in bounds.items() for v in iv)]
        from hegel_machine.cognition.models import Probe

        records = []
        b0 = lab.acquire(Probe("single_scan", tuple(single_pts)), ("fine",))
        for pos, (pt, obs, tr) in enumerate(zip(single_pts, b0.observations, b0.transitions)):
            records.append({"controls": dict(pt), **tr, "target": obs.target, "probe_id": "single_scan", "position": pos})
        for idx in range(3):
            pr = lab.random_probe(9001 + idx, length=48, name=f"traj_{idx}")
            bt = lab.acquire(pr, ("fine",))
            for pos, (pt, obs, tr) in enumerate(zip(pr.sequence, bt.observations, bt.transitions)):
                records.append({"controls": dict(pt), **tr, "target": obs.target, "probe_id": pr.probe_id, "position": pos})

        model = infer_object_model(raw_spec, records, SearchBudget(80000))
        audit_iv = audit_stochastic_transition_premises(
            model, records, SearchBudget(80000), estimation_method="2sls_iv"
        )
        assert audit_iv is not None
        assert audit_iv["status"] == "verified_latent_transition_exchange"
        assert "P_stochastic_process_noise" in audit_iv["supported_premises"]
        assert "P_conservation" in audit_iv["supported_premises"]
        diag = audit_iv["latent_transition_state"]["iv_diagnostics"]
        # Naive OLS suffers strong negative endogeneity bias; IV recovers pole close to 0.75!
        assert abs(diag["refined_iv_pole"] - 0.75) < abs(diag["naive_ols_pole"] - 0.75)
        assert abs(diag["refined_iv_pole"] - 0.75) < 0.05
        assert diag["hurwicz_nickell_ols_bias"] < -0.05

    # 2. Verify near-critical Van der Pol (exact seed 990401, rho = 1.02) achieves 100% completion!
    from scripts.run_v10_phase3a_higher_order_and_coupled_pomdp import _make_markov_right_prior, _schema

    v4 = _schema(4, "src4")
    lb = KnowledgeLibrary()
    lb.meta_priors = [
        _make_markov_right_prior(v4, concept_id="m_te_1", radius=0.91, readout_kind="total_event"),
        _make_markov_right_prior(v4, concept_id="m_te_2", radius=0.98, readout_kind="total_event"),
    ]
    cfg = ResearchConfig(
        observation_budget=380,
        compute_budget=380000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        latent_state_discovery=True,
        higher_order_latent_discovery=True,
        nonlinear_limit_cycle_discovery=True,
        active_resonance_probing=True,
    )
    with AnonymousObjectLab(
        seed=990401,
        objects=4,
        readout="total_event",
        relation_direction="right",
        relation_radius=0.96,
        latent_mode="van_der_pol_transition_memory",
        latent_decay=1.02,
        latent_freq=0.65,
        latent_nonlinearity=0.28,
        latent_weight=0.44,
        name_seed=990401 % 997 + 17,
        wiring_seed=990401 % 991 + 31,
    ) as lab_vdp:
        res_vdp = UnifiedResearchAgent(cfg, measurement_budget=7800, library=lb).run(lab_vdp)
        assert res_vdp["status"] == "completed"
        assert res_vdp["selected_channels"] == []
        prog_vdp = Program.from_dict(res_vdp["program"])
        mse_vdp = loss(
            prog_vdp,
            lab_vdp.evaluation_rows((), seed=990401 + 100000, length=48, trajectories=1),
            res_vdp["fallback"],
            0.5,
        )
        assert mse_vdp <= 1e-14


def test_phase5c_symbolic_proof_kernel_tamper_rejection_and_discrepancy_repair():
    # 1. Cross-commitment symbolic flux elimination across all 4 invariant/monotonicity classes
    for spec in (
        {"system_class": "coupled_state_reservoir_pomdp", "alpha": "3/4", "eta": "1/2", "c_u": "1"},
        {"system_class": "two_species_reaction_transport", "stoichiometric_ratio": "3"},
        {"system_class": "manifold_neumann_conservation", "diffusion": "1/7", "advection": "1/11"},
        {"system_class": "manifold_dirichlet_gauss_green", "diffusion": "1/7", "advection": "1/11"},
        {"system_class": "directional_moment_monotonicity", "gamma": "1/5"},
    ):
        certs = synthesize_emergent_invariants(spec)
        assert len(certs) >= 1
        assert certs[0]["kernel_verification"]["verified"] is True

    # Negative tamper test: perturbing invariant weight must be rejected by SymbolicProofKernel!
    good_pomdp = synthesize_emergent_invariants(
        {"system_class": "coupled_state_reservoir_pomdp", "alpha": "3/4", "eta": "1/2", "c_u": "1"}
    )[0]
    tampered_inv = deepcopy(good_pomdp)
    tampered_inv["invariant_weights"]["sum_z"] = "199/100"
    assert SymbolicProofKernel.verify_emergent_invariant_certificate(tampered_inv)["verified"] is False

    # 2. Spatially heterogeneous directional reachability vs impossibility
    # Initial mass q_2 = 2.0 at x_2 = 1.1; Goal: q_3 >= 0.8 at x_3 = 1.8 within tiny budget B = 0.15!
    # Under RIGHT transport (gamma = 0.30), passive flow carries 0.60/step into q_3 -> WINNABLE!
    # Under LEFT transport, flow goes away from q_3 -> IMPOSSIBLE (non-uniform spatial Farkas potential)!
    spatial_task = {
        "initial_state": [0.0, 0.0, 2.0, 0.0],
        "control_bounds": [-0.25, 0.25],
        "state_bounds": [0.0, 10.0],
        "budget": 0.15,
        "step_cost": 0.0,
        "fuel_weights": [1.0, 1.0, 1.0, 1.0],
        "goal_matrix": [[0.0, 0.0, 0.0, -1.0]],
        "goal_rhs": [-0.8],
    }
    right_model = {
        "objects": 4,
        "positions": [-1.8, -0.4, 1.1, 1.8],
        "coupling": 0.30,
        "radius": 1.0,
        "direction": "right",
    }
    left_model = {**right_model, "direction": "left"}
    cert_right = prove_task_solvability(right_model, spatial_task, certificate_id="dir_right")
    cert_left = prove_task_solvability(left_model, spatial_task, certificate_id="dir_left")
    assert cert_right["verdict"] == "winnable"
    assert cert_right["kernel_verification"]["verified"] is True
    assert cert_left["verdict"] == "impossible"
    assert cert_left["kernel_verification"]["verified"] is True

    # Negative tamper test on PotentialLowerBoundCertificate: inflating alpha_0 must fail Goal obligation!
    tampered_pot = deepcopy(cert_left)
    tampered_pot["alpha_0"] = "99"
    assert (
        SymbolicProofKernel.verify_solvability_certificate(tampered_pot, left_model, spatial_task)["verified"]
        is False
    )

    # 3. AWM §4.3 Discrepancy-Triggered Premise Audit & Re-Proof Loop
    lab = BudgetedControlLab(
        objects=4,
        positions=(-1.8, -0.5, 0.4, 1.8),
        coupling=0.25,
        radius=1.0,
        direction="right",
        boundary_port_gain=1.0,
        boundary_port_threshold=1.5,
        control_bounds=(-0.5, 0.5),
    )
    init_model = {
        "objects": 4,
        "positions": [-1.8, -0.5, 0.4, 1.8],
        "coupling": 0.25,
        "radius": 1.0,
        "direction": "right",
        "boundary_port_gain": 0.0,
        "boundary_port_threshold": 1.5,
        "premise_ids": ["P_conservation", "P_interaction_direction", "P_locality_radius", "P_control_input_gain"],
    }
    disc_task = {
        "initial_state": [0.0, 0.0, 0.0, 0.0],
        "control_bounds": [-0.5, 0.5],
        "state_bounds": [-10.0, 10.0],
        "budget": 1.25,
        "step_cost": 0.0,
        "fuel_weights": [1.0, 1.0, 1.0, 1.0],
        "goal_matrix": [[-1.0, -1.0, -1.0, -1.0]],
        "goal_rhs": [-2.0],
    }
    cert_init = prove_task_solvability(init_model, disc_task, certificate_id="disc_init")
    assert cert_init["verdict"] == "impossible"
    assert abs(cert_init["certified_min_budget"] - 2.0) <= 1e-9
    assert "P_focus_membership" in cert_init["unused_premise_ids"]

    p_graph = {
        "nodes": [
            {"id": "P_conservation", "status": "supported"},
            {"id": "P_control_input_gain", "status": "supported"},
            {"id": "P_interaction_direction", "status": "supported"},
            {"id": "P_focus_membership", "status": "supported"},
        ]
    }
    rep = repair_proof_from_discrepancy(
        lab,
        init_model,
        p_graph,
        disc_task,
        cert_init,
        discrepancy_feedback={"oracle_winnable": True},
    )
    assert rep["repaired"] is True
    assert rep["real_lab_execution_verification"]["winnable_verified_on_lab"] is True
    assert abs(rep["true_certified_min_budget"] - 1.0) <= 1e-9
