"""Verification suite for v10 Phase 2A expanded scientific revisions and Phase 2B open-grammar author."""

from copy import deepcopy

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe
from hegel_machine.cognition.open_grammar_author import (
    propose_flat_unconstrained_programs,
    propose_premise_gated_commitment_revisions,
)
from hegel_machine.cognition.premise_dependency_audit import (
    apply_verified_transition_premises,
    audit_transition_premises,
)
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
    instantiate_commitment_on_schema,
    propose_scientific_premise_revision,
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


def _neighbor(view: ObjectSchema, radius_node: Program, direction: str | None = "right") -> Program:
    return Program(
        "neighbor_sum",
        name=direction,
        children=(
            radius_node,
            *(_feature(view, item, role) for item in view.objects for role in ("f", "x", "q")),
        ),
    )


def _make_directional_prior(
    view: ObjectSchema,
    *,
    concept_id: str = "src_n4_right",
    direction: str | None = "right",
    radius: float = 0.95,
    weights: tuple[float, float, float] = (1.1, 1.1, 0.0),
    window: int = 3,
) -> dict:
    if window > 1:
        sym_terms = (
            Program("window_sum", lag=window, children=(_total(view, "q"),)),
            Program("window_sum", lag=window, children=(_total(view, "u"),)),
            Program(
                "window_sum",
                lag=window,
                children=(
                    Program(
                        "event_delta",
                        lag=1,
                        children=(_neighbor(view, Program("arg", name="r_0"), direction),),
                    ),
                ),
            ),
        )
        actual_terms = (
            sym_terms[0],
            sym_terms[1],
            Program(
                "window_sum",
                lag=window,
                children=(
                    Program(
                        "event_delta",
                        lag=1,
                        children=(_neighbor(view, Program("const", value=radius), direction),),
                    ),
                ),
            ),
        )
    else:
        sym_terms = (
            _total(view, "q"),
            _total(view, "u"),
            Program(
                "event_delta",
                lag=1,
                children=(_neighbor(view, Program("arg", name="r_0"), direction),),
            ),
        )
        actual_terms = (
            sym_terms[0],
            sym_terms[1],
            Program(
                "event_delta",
                lag=1,
                children=(_neighbor(view, Program("const", value=radius), direction),),
            ),
        )
    symbolic_program = _sum_terms(list(sym_terms), list(weights), 0.0)
    program = _sum_terms(actual_terms, list(weights), 0.0)
    plan = _plan(
        {
            "fit_terms": actual_terms,
            "coefficients": [0.0, *weights],
            "program": program,
            "loss": 0.0,
        },
        [
            {
                "operation": "resolve_relation_intervals",
                "question": "Identify source radius",
                "parameter_template": symbolic_program.to_dict(),
                "parameter_terms": [t.to_dict() for t in sym_terms],
                "parameters": [
                    {
                        "symbol": "r_0",
                        "radius": radius,
                        "radius_intervals": [[radius - 0.1, radius + 0.1]],
                        "prior_radius_intervals": [[radius - 0.1, radius + 0.1]],
                        "relation": direction,
                        "source_concept_id": concept_id,
                    }
                ],
                "evidence_contexts": [],
                "compact_replacement": False,
                "program": program,
                "loss": 0.0,
            }
        ],
        48,
        1200,
    )
    return {
        "prior_id": f"prior:{concept_id}",
        "source_concept_id": concept_id,
        "family_id": "T12",
        "kind": "structural:joint_window_event",
        "status": "supported",
        "task_id": f"task:{concept_id}",
        "object_schema": view.to_dict(),
        "research_program": plan,
        "counterexamples": [],
    }


def _collect_roles_and_trajectory(
    lab: AnonymousObjectLab,
    budget: SearchBudget,
    *,
    traj_seed: int = 701,
    traj_len: int = 32,
    val_seed: int = 702,
    val_len: int = 24,
) -> tuple[dict, list[dict], list[Observation], list[Observation], list[dict]]:
    """Run public role sweeps (without focus sweeps) plus training/validation trajectories."""
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

    acquire_probe(lab.random_probe(901, length=12, name="random_0"))
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

    training, train_transitions = acquire_probe(
        lab.random_probe(traj_seed, length=traj_len, name="train_traj")
    )
    validation = lab.observe(lab.random_probe(val_seed, length=val_len, name="val_traj"), ("fine",))
    model = infer_object_model(raw_spec, records, budget)
    return model, records, training, validation, train_transitions


def test_phase2a_conditional_exchange_audit_and_scientific_revision():
    # Source commitment: N=4 symmetric unconditional window_total_event
    prior_n4_sym = _make_directional_prior(
        _schema(4, "c4"),
        concept_id="src_n4_sym_uncond",
        direction=None,
        radius=0.95,
        weights=(1.0, 1.0, 0.5),
        window=3,
    )
    comm_sym = consolidate_world_structure_commitments([prior_n4_sym])[0]
    assert comm_sym["radius_parameters"][0]["direction"] == "symmetric"
    assert comm_sym["radius_parameters"][0]["edge_condition"] == "unconditional"

    # Target N=5: directional 'right' + state-dependent 'source_greater'
    with AnonymousObjectLab(
        seed=420501,
        objects=5,
        readout="window_total_event",
        relation_direction="right",
        relation_condition="source_greater",
        relation_radius=1.12,
        actuator_curvature=0.18,
    ) as lab:
        budget = SearchBudget(160000)
        model, records, training, validation, _ = _collect_roles_and_trajectory(lab, budget)
        audit = audit_transition_premises(
            model, records, budget, commitments=[comm_sym], tolerance=1e-8
        )
        assert audit["status"] == "verified_conditional_exchange"
        assert audit["conservation_supported"] is True
        assert audit["verified_direction"] == "right"
        assert audit["verified_edge_condition"] == "source_greater"
        assert audit["surviving_bijection_count"] == 1

        updated = apply_verified_transition_premises(model, audit)
        assert updated["complete"] is True
        spec = lab.spec_for(("fine",))
        entries = propose_scientific_premise_revision(
            comm_sym, audit, spec, training, updated["schema"], budget, tolerance=1e-8
        )
        assert len(entries) >= 1
        best = entries[0]
        assert best["training_loss"] <= 1e-10
        val_mse = loss(best["candidate"].program, validation, 0.0, 0.5)
        assert val_mse <= 1e-10
        rev = best["scientific_premise_revision"]
        assert rev["revision_kind"] == "exchange_condition_revision"
        retracted_ids = {p["premise_id"] for p in rev["retracted_premises"]}
        assert "exchange_form:r_0" in retracted_ids
        assert "interaction_direction:r_0" in retracted_ids

        # Confirm & consolidate the revised conditional law -> verify lossless round-trip!
        agent = UnifiedResearchAgent(
            ResearchConfig(world_structure_commitments=True, transition_premise_audit=True)
        )
        spec_with_schema = agent._spec(lab, ("fine",))
        agent._object_schemas[("fine",)] = deepcopy(updated["schema"])
        spec_with_schema = agent._spec(lab, ("fine",))
        saved = agent.remember_confirmed_selection(
            best,
            spec_with_schema,
            training,
            validation,
            confirmed=True,
            confirmation_loss=val_mse,
            confirmation_rows=len(validation),
            selected_channels=("fine",),
        )
        assert saved["stored"] is True
        consolidated_next = consolidate_world_structure_commitments(agent.library.meta_priors)
        assert len(consolidated_next) == 1
        assert consolidated_next[0]["radius_parameters"][0]["direction"] == "right"
        assert consolidated_next[0]["radius_parameters"][0]["edge_condition"] == "source_greater"
        assert consolidated_next[0]["generation"] == 1


def test_phase2a_nonconservative_sink_and_nonlocal_decoupling_and_revision():
    prior_n4 = _make_directional_prior(
        _schema(4, "nc4"),
        concept_id="src_n4_right_conserved",
        direction="right",
        radius=0.95,
        weights=(1.0, 1.0, 0.5),
        window=3,
    )
    comm_conserved = consolidate_world_structure_commitments([prior_n4])[0]

    for variant_name, target_dir, seed_val in (
        ("sink", "right", 420601),
        ("nonlocal", "left", 420602),
    ):
        with AnonymousObjectLab(
            seed=seed_val,
            objects=5,
            readout="window_total_event",
            variant=variant_name,
            relation_direction=target_dir,
            relation_radius=1.08,
            actuator_curvature=0.16,
        ) as lab:
            budget = SearchBudget(180000)
            model, records, training, validation, _ = _collect_roles_and_trajectory(lab, budget)
            audit = audit_transition_premises(
                model, records, budget, commitments=[comm_conserved], tolerance=1e-8
            )
            assert audit["status"] == "verified_nonconservative_exchange"
            assert audit["conservation_supported"] is False
            assert variant_name in audit["nonconservation_mechanisms"]
            assert audit["verified_direction"] == target_dir
            assert audit["surviving_bijection_count"] == 1

            updated = apply_verified_transition_premises(model, audit)
            assert updated["complete"] is True
            entries = propose_scientific_premise_revision(
                comm_conserved,
                audit,
                lab.spec_for(("fine",)),
                training,
                updated["schema"],
                budget,
                tolerance=1e-8,
            )
            assert len(entries) >= 1
            best = entries[0]
            assert best["training_loss"] <= 1e-10
            val_mse = loss(best["candidate"].program, validation, 0.0, 0.5)
            assert val_mse <= 1e-10
            assert (
                best["world_commitment_application"]["nonconservation_mechanism"] == variant_name
            )


def test_phase2a_four_generation_n3_to_n6_commitment_evolution():
    # Gen 0 (N=3): symmetric, unconditional, conserved
    prior_n3 = _make_directional_prior(
        _schema(3, "g0"),
        concept_id="gen0_n3",
        direction=None,
        radius=0.90,
        weights=(1.0, 1.0, 0.5),
        window=3,
    )
    current_comm = consolidate_world_structure_commitments([prior_n3])[0]
    assert current_comm["generation"] == 0

    stages = [
        # Gen 1 (N=4): direction symmetric -> right
        {"arity": 4, "seed": 510004, "dir": "right", "cond": "always", "variant": "exchange", "expected_gen": 1},
        # Gen 2 (N=5): condition unconditional -> source_greater
        {"arity": 5, "seed": 510005, "dir": "right", "cond": "source_greater", "variant": "exchange", "expected_gen": 2},
        # Gen 3 (N=6): direction right -> left AND conservation -> sink (6! = 720 -> 1!)
        {"arity": 6, "seed": 510006, "dir": "left", "cond": "source_greater", "variant": "sink", "expected_gen": 3},
    ]

    for stage in stages:
        with AnonymousObjectLab(
            seed=stage["seed"],
            objects=stage["arity"],
            readout="window_total_event",
            variant=stage["variant"],
            relation_direction=stage["dir"],
            relation_condition=stage["cond"],
            relation_radius=1.10,
            actuator_curvature=0.15,
        ) as lab:
            budget = SearchBudget(260000)
            model, records, training, validation, _ = _collect_roles_and_trajectory(lab, budget)
            audit = audit_transition_premises(
                model, records, budget, commitments=[current_comm], tolerance=1e-8
            )
            assert audit["surviving_bijection_count"] == 1
            updated = apply_verified_transition_premises(model, audit)
            assert updated["complete"] is True
            entries = propose_scientific_premise_revision(
                current_comm,
                audit,
                lab.spec_for(("fine",)),
                training,
                updated["schema"],
                budget,
                tolerance=1e-8,
            )
            assert entries and entries[0]["training_loss"] <= 1e-9
            val_mse = loss(entries[0]["candidate"].program, validation, 0.0, 0.5)
            assert val_mse <= 1e-9

            agent = UnifiedResearchAgent(
                ResearchConfig(world_structure_commitments=True, transition_premise_audit=True)
            )
            agent._object_schemas[("fine",)] = deepcopy(updated["schema"])
            spec_with_schema = agent._spec(lab, ("fine",))
            saved = agent.remember_confirmed_selection(
                entries[0],
                spec_with_schema,
                training,
                validation,
                confirmed=True,
                confirmation_loss=val_mse,
                confirmation_rows=len(validation),
                selected_channels=("fine",),
            )
            assert saved["stored"] is True
            current_comm = consolidate_world_structure_commitments(agent.library.meta_priors)[0]
            assert current_comm["generation"] == stage["expected_gen"]
            assert len(current_comm["revision_lineage"]) == stage["expected_gen"]


def test_phase2b_open_grammar_author_offline_gate_and_zero_call_n5_transfer():
    # 1. Open-grammar edge predicate ('donor_positive_gap': q_src > 0 and q_src - q_dst > 0.25)
    prior_n4 = _make_directional_prior(
        _schema(4, "og4"),
        concept_id="og_parent_n4",
        direction="right",
        radius=0.95,
        weights=(1.0, 1.0, 0.5),
        window=3,
    )
    comm_parent = consolidate_world_structure_commitments([prior_n4])[0]

    wrong_pred = Program(
        "lt",
        children=(
            Program("const", value=0.9),
            Program("arg", name="edge_source_quantity"),
        ),
    ).to_dict()
    true_pred = Program(
        "and",
        children=(
            Program(
                "lt",
                children=(
                    Program("const", value=0.0),
                    Program("arg", name="edge_source_quantity"),
                ),
            ),
            Program(
                "lt",
                children=(
                    Program("const", value=0.25),
                    Program(
                        "sub",
                        children=(
                            Program("arg", name="edge_source_quantity"),
                            Program("arg", name="edge_target_quantity"),
                        ),
                    ),
                ),
            ),
        ),
    ).to_dict()

    def mock_edge_transport(instructions, payload, *, purpose="mock"):
        assert payload["isolated_violated_premise"] == "P_exchange_form"
        return (
            {
                "candidates": [
                    {"rationale": "Overly strict threshold (should be rejected)", "directions": ["right"], "edge_predicate": wrong_pred},
                    {"rationale": "Positive donor and gap > 0.25", "directions": ["right", "left"], "edge_predicate": true_pred},
                ]
            },
            {
                "backend": "mock",
                "model": "mock-gpt",
                "calls": 1,
                "input_tokens": 420,
                "output_tokens": 180,
                "total_tokens": 600,
                "wall_time_seconds": 0.01,
                "purpose": purpose,
                "status": "ok",
            },
        )

    with AnonymousObjectLab(
        seed=610401,
        objects=4,
        readout="window_total_event",
        relation_direction="right",
        relation_condition="donor_positive_gap",
        relation_radius=1.08,
        actuator_curvature=0.16,
    ) as lab4:
        budget = SearchBudget(180000)
        model4, records4, train4, val4, raw_tr4 = _collect_roles_and_trajectory(lab4, budget)
        res_gated = propose_premise_gated_commitment_revisions(
            comm_parent,
            model4,
            records4,
            lab4.spec_for(("fine",)),
            train4,
            val4,
            budget,
            mock_edge_transport,
            max_llm_calls=2,
            tolerance=1e-8,
            model_name="mock-gpt",
        )
        assert res_gated["supported"] is True
        assert res_gated["isolated_violated_premise"] == "P_exchange_form"
        assert res_gated["llm_ledger"]["model_calls"] == 1
        assert res_gated["llm_ledger"]["total_tokens"] == 600
        best4 = res_gated["best_entry"]
        assert best4["validation_loss"] <= 1e-10

        # Consolidate into a verified world_structure_commitment!
        agent = UnifiedResearchAgent(
            ResearchConfig(world_structure_commitments=True, transition_premise_audit=True)
        )
        agent._object_schemas[("fine",)] = deepcopy(best4["resolved_schema"])
        saved = agent.remember_confirmed_selection(
            best4,
            agent._spec(lab4, ("fine",)),
            train4,
            val4,
            confirmed=True,
            confirmation_loss=best4["validation_loss"],
            confirmation_rows=len(val4),
            selected_channels=("fine",),
        )
        assert saved["stored"] is True
        og_comm = consolidate_world_structure_commitments(agent.library.meta_priors)[0]
        assert isinstance(og_comm["radius_parameters"][0]["edge_condition"], dict)

    # Now transfer og_comm to N=5 with ZERO LLM calls!
    with AnonymousObjectLab(
        seed=610501,
        objects=5,
        readout="window_total_event",
        relation_direction="right",
        relation_condition="donor_positive_gap",
        relation_radius=1.14,
        actuator_curvature=0.16,
    ) as lab5:
        budget5 = SearchBudget(180000)
        model5, records5, train5, val5, _ = _collect_roles_and_trajectory(lab5, budget5)
        audit5 = audit_transition_premises(
            model5, records5, budget5, commitments=[og_comm], tolerance=1e-8
        )
        assert audit5["status"] == "verified_unconditional_exchange"
        assert audit5["surviving_bijection_count"] == 1
        updated5 = apply_verified_transition_premises(model5, audit5)
        assert updated5["complete"] is True
        entries5 = instantiate_commitment_on_schema(
            lab5.spec_for(("fine",)),
            train5,
            updated5["schema"],
            og_comm,
            budget5,
            radius_bounds=audit5["radius_interval"],
            tolerance=1e-8,
        )
        assert entries5 and entries5[0]["training_loss"] <= 1e-10
        heldout5 = lab5.evaluation_rows(("fine",), seed=9901, length=48, trajectories=1)
        assert loss(entries5[0]["candidate"].program, heldout5, 0.0, 0.5) <= 1e-10

    # 2. Open-grammar temporal readout ('focal_quadratic_total_event': total + 0.7*event + 0.35*q[focus]^2)
    prior_te4 = _make_directional_prior(
        _schema(4, "rq4"),
        concept_id="rq_parent_n4",
        direction="right",
        radius=0.95,
        weights=(1.0, 1.0, 0.5),
        window=1,
    )
    comm_te = consolidate_world_structure_commitments([prior_te4])[0]
    quad_role_term = {
        "kind": "all_object_sum",
        "summand": Program(
            "mul",
            children=(
                Program("feature", name="object_role:f"),
                Program(
                    "mul",
                    children=(
                        Program("feature", name="object_role:q"),
                        Program("feature", name="object_role:q"),
                    ),
                ),
            ),
        ).to_dict(),
    }

    def mock_readout_transport(instructions, payload, *, purpose="mock"):
        assert payload["isolated_violated_premise"] == "P_temporal_readout"
        return (
            {"candidates": [{"rationale": "Focal quadratic q^2", "extra_role_terms": [quad_role_term]}]},
            {"backend": "mock", "model": "mock-gpt", "calls": 1, "input_tokens": 350, "output_tokens": 120, "total_tokens": 470, "wall_time_seconds": 0.01, "purpose": purpose, "status": "ok"},
        )

    with AnonymousObjectLab(
        seed=620401,
        objects=4,
        readout="focal_quadratic_total_event",
        relation_direction="right",
        relation_radius=1.05,
        actuator_curvature=0.15,
    ) as lab_rq4:
        budget_rq = SearchBudget(180000)
        m_rq4, rec_rq4, tr_rq4, val_rq4, raw_rq4 = _collect_roles_and_trajectory(lab_rq4, budget_rq)
        res_rq = propose_premise_gated_commitment_revisions(
            comm_te,
            m_rq4,
            rec_rq4,
            lab_rq4.spec_for(("fine",)),
            tr_rq4,
            val_rq4,
            budget_rq,
            mock_readout_transport,
            max_llm_calls=2,
            tolerance=1e-8,
            model_name="mock-gpt",
        )
        assert res_rq["supported"] is True
        assert res_rq["isolated_violated_premise"] == "P_temporal_readout"
        assert res_rq["best_entry"]["validation_loss"] <= 1e-10
