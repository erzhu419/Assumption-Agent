"""Focused verification for v10 world-structure commitments and premise dependency audit."""

from copy import deepcopy
from random import Random

import pytest

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import (
    plan_generalized_position_followup,
    plan_position_owner_followup,
    propose_object_probe,
)
from hegel_machine.cognition.premise_dependency_audit import (
    apply_verified_transition_premises,
    audit_transition_premises,
    compare_flat_error_vs_premise_graph,
)
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    check_equivalence_reorganization,
    consolidate_world_structure_commitments,
    extract_parametric_role_pattern,
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
    """Build a realistic confirmed directional prior (including zero-weight columns)."""
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


def _collect_role_records(lab: AnonymousObjectLab, budget: SearchBudget) -> tuple[dict, list[dict]]:
    """Run public random + single-control + additive role sweeps without focus sweeps."""
    raw_spec = lab.spec_for(("fine",))
    records = []

    def acquire_probe(probe: Probe):
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

    acquire_probe(lab.random_probe(901, length=12, name="random_0"))
    model = infer_object_model(raw_spec, records, budget)
    for serial in range(16):
        probe, question = propose_object_probe(raw_spec, model, records, serial)
        if probe is None or question["purpose"] not in {
            "isolate_control_effects",
            "distinguish_role_definition",
        }:
            break
        acquire_probe(probe)
        model = infer_object_model(raw_spec, records, budget)
    return model, records


def test_equivalence_reorganization_preserves_zero_weight_columns_symbols_and_rejection():
    view4 = _schema(4, "s4")
    # Notice weight[2] == 0.0: saved zero-weight column MUST survive consolidation!
    prior = _make_directional_prior(view4, weights=(1.2, 1.2, 0.0))
    budget = SearchBudget(50000)
    commitments = consolidate_world_structure_commitments([prior], budget=budget)
    assert len(commitments) == 1
    comm = commitments[0]
    assert comm["reorganization_type"] == "equivalence_reorganization"
    assert len(comm["role_terms"]) == 3
    assert comm["radius_parameters"][0]["direction"] == "right"
    assert comm["equivalence_audit"]["accepted"] is True

    extracted = extract_parametric_role_pattern(prior)
    assert extracted is not None

    # 1. Dropping the zero-weight column is rejected!
    dropped = deepcopy(comm)
    dropped["role_terms"] = dropped["role_terms"][:2]
    dropped["radius_parameters"] = []
    res_dropped = check_equivalence_reorganization([extracted], dropped, budget=budget)
    assert res_dropped["accepted"] is False
    assert res_dropped["reason"] in {
        "symbolic_parameter_groups_mismatch",
        "column_count_or_instantiation_mismatch",
    }

    # 2. Coarsening direction ('right' -> 'symmetric') is rejected!
    wrong_dir = deepcopy(comm)
    wrong_dir["radius_parameters"][0]["direction"] = "symmetric"
    res_dir = check_equivalence_reorganization([extracted], wrong_dir, budget=budget)
    assert res_dir["accepted"] is False
    assert res_dir["reason"] == "operator_direction_or_condition_changed"

    # 3. Shrinking rejection set Rej(T) < Rej(S) on a counterexample is rejected!
    rng = Random(42)
    ce_rows = []
    history = []
    for step in range(12):
        f_group = view4.objects[step % 4]
        feats = {}
        for g in view4.objects:
            feats[view4.fields[g]["q"]] = rng.uniform(-1.0, 1.0)
            feats[view4.fields[g]["u"]] = rng.uniform(-0.2, 0.2)
            feats[view4.fields[g]["x"]] = rng.uniform(-2.0, 2.0)
            feats[view4.fields[g]["f"]] = 1.0 if g == f_group else 0.0
        # Target generated by focal_x (orthogonal to conserved window_sum(q, u, neighbor))
        target = feats[view4.fields[f_group]["x"]] * 2.5
        ce_rows.append(Observation(feats, target, tuple(history[-3:])))
        history.append(dict(feats))

    res_valid_ce = check_equivalence_reorganization(
        [extracted],
        comm,
        counterexample_rows_by_source={extracted["source_concept_id"]: ce_rows},
        budget=budget,
    )
    assert res_valid_ce["accepted"] is True
    assert res_valid_ce["checks"]["source_details"][0]["source_counterexample_loss"] > 1e-3
    assert res_valid_ce["checks"]["source_details"][0]["commitment_counterexample_loss"] > 1e-3


def test_transition_premise_audit_collapses_n5_bijections_and_isolates_violations():
    # 1. N=5 directional ('right') anonymous apparatus: 5! = 120 position bijections -> 1!
    with AnonymousObjectLab(
        seed=20261116,
        objects=5,
        readout="window_total_event",
        relation_direction="right",
        relation_radius=1.05,
        actuator_curvature=0.2,
        name_seed=77,
        wiring_seed=88,
    ) as lab:
        budget = SearchBudget(80000)
        model, records = _collect_role_records(lab, budget)
        # Before transition premise audit, centered role checks leave multiple x owners unresolved
        # and focus f is still unresolved.
        assert any("f" in item for item in model["unresolved"])

        view4 = _schema(4, "src")
        prior = _make_directional_prior(view4, direction="right")
        comm = consolidate_world_structure_commitments([prior])[0]

        audit = audit_transition_premises(
            model, records, budget, commitments=[comm], tolerance=1e-8
        )
        assert audit["status"] == "verified_unconditional_exchange"
        assert audit["conservation_supported"] is True
        assert audit["supported_directions"] == ["right"]
        assert audit["surviving_bijection_count"] == 1
        low, high = audit["radius_interval"]
        assert low < 1.05 < high

        updated = apply_verified_transition_premises(model, audit)
        truth = lab._evaluation_schema(("fine",))
        true_x_to_q = {
            truth.fields[obj]["x"]: truth.fields[obj]["q"] for obj in truth.objects
        }
        inferred_x_to_q = {
            x_field: cand["quantities"][0]
            for x_field, cand in updated["position_candidates"].items()
        }
        assert inferred_x_to_q == true_x_to_q
        assert not any(item.endswith(":x") for item in updated["unresolved"])

    # 2a. Unmodeled inhomogeneous non-conserved ('anchored') apparatus -> conservation refuted, ZERO false pruning!
    with AnonymousObjectLab(
        seed=20261117,
        objects=4,
        readout="focus",
        variant="anchored",
        relation_direction="symmetric",
        relation_radius=1.0,
    ) as anchored_lab:
        budget = SearchBudget(80000)
        anc_model, anc_records = _collect_role_records(anchored_lab, budget)
        before_domains = deepcopy(anc_model["position_candidates"])
        anc_audit = audit_transition_premises(anc_model, anc_records, budget, tolerance=1e-8)
        assert anc_audit["status"] == "conservation_refuted"
        assert anc_audit["conservation_supported"] is False
        assert anc_audit["refuted_premises"][0]["premise_id"] == "P_conservation"
        after_anc = apply_verified_transition_premises(anc_model, anc_audit)
        assert after_anc["position_candidates"] == before_domains

    # 2b. Proportional dissipation ('sink') apparatus -> conservation refuted & de-coupled without false pruning!
    with AnonymousObjectLab(
        seed=20261117,
        objects=4,
        readout="focus",
        variant="sink",
        relation_direction="symmetric",
        relation_radius=1.0,
    ) as sink_lab:
        budget = SearchBudget(80000)
        sink_model, sink_records = _collect_role_records(sink_lab, budget)
        sink_audit = audit_transition_premises(sink_model, sink_records, budget, tolerance=1e-8)
        assert sink_audit["status"] == "verified_nonconservative_exchange"
        assert sink_audit["conservation_supported"] is False
        assert sink_audit["refuted_premises"][0]["premise_id"] == "P_conservation"
        assert "sink" in sink_audit["nonconservation_mechanisms"]
        after_sink = apply_verified_transition_premises(sink_model, sink_audit)
        truth_sink = sink_lab._evaluation_schema(("fine",))
        for obj in truth_sink.objects:
            x_f = truth_sink.fields[obj]["x"]
            q_f = truth_sink.fields[obj]["q"]
            assert q_f in after_sink["position_candidates"][x_f]["quantities"]

    # 3. AWM C.6 analogue: 'right' commitment evaluated on 'left' apparatus!
    with AnonymousObjectLab(
        seed=20261118,
        objects=5,
        readout="window_total_event",
        relation_direction="left",
        relation_radius=1.1,
        actuator_curvature=0.2,
    ) as left_lab:
        budget = SearchBudget(100000)
        left_model, left_records = _collect_role_records(left_lab, budget)
        traj_rows = left_lab.observe(left_lab.random_probe(555, length=24, name="traj"), ("fine",))
        comm_right = consolidate_world_structure_commitments(
            [_make_directional_prior(_schema(4, "src"), direction="right")]
        )[0]
        comparison = compare_flat_error_vs_premise_graph(
            comm_right, left_model, left_records, traj_rows, budget, tolerance=1e-7
        )
        assert comparison["flat_item_error"]["isolated_violated_premise"] is None
        assert (
            comparison["premise_dependency_graph"]["isolated_violated_premise"]
            == "P_interaction_direction"
        )
        triggers = comparison["premise_dependency_graph"]["scientific_revision_triggers"]
        assert len(triggers) == 1
        assert triggers[0]["retracted_direction"] == "right"
        assert triggers[0]["replacement_direction"] == "left"


def test_generalized_position_followup_supports_k3_and_delegates_k2():
    spec_bounds = {f"c{i}": (-1.0, 1.0) for i in range(13)}
    from hegel_machine.cognition.models import TaskSpec

    spec = TaskSpec(
        "gen_pos",
        "",
        tuple(f"m_{i}" for i in range(16)),
        (),
        (),
        feature_bounds=spec_bounds,
        max_probe_length=12,
    )
    base = {k: 0.0 for k in spec_bounds}
    moved = {**base, "c_pos0": 1.0}
    spec_bounds["c_pos0"] = (-1.0, 1.0)
    for idx in range(4):
        spec_bounds[f"cq_{idx}"] = (-1.0, 1.0)
        base[f"cq_{idx}"] = 0.0
        moved[f"cq_{idx}"] = 0.0
    spec = TaskSpec("gen_pos", "", tuple(f"m_{i}" for i in range(16)), (), (), feature_bounds=spec_bounds)

    records = [
        {"probe_id": "p0", "position": 0, "history": [], "controls": dict(base), "before": {}, "after": {}},
        {"probe_id": "p1", "position": 0, "history": [], "controls": {**base, "cq_0": 1.0}, "before": {}, "after": {}},
        {"probe_id": "p2", "position": 0, "history": [], "controls": dict(moved), "before": {}, "after": {}},
        {"probe_id": "p3", "position": 0, "history": [], "controls": {**moved, "cq_0": 1.0}, "before": {}, "after": {}},
    ]
    prev_plan = {
        "field": "x_0",
        "position_control": "c_pos0",
        "corners": [r["controls"] for r in records],
        "baseline_ref": {"probe_id": "p0", "position": 0},
    }
    # k = 3 remaining owners!
    model_k3 = {
        "model_id": "objects_0001",
        "dynamic_fields": ["q_0", "q_1", "q_2", "q_3"],
        "quantity_controls": {f"q_{i}": f"cq_{i}" for i in range(4)},
        "position_candidates": {
            "x_0": {"control": "c_pos0", "quantities": ["q_1", "q_2", "q_3"]}
        },
    }
    prev_res_k3 = {
        "status": "position_domain_narrowed",
        "field": "x_0",
        "source_model_id": "objects_0001",
        "retained_owners": ["q_1", "q_2", "q_3"],
        "corner_refs": [{"probe_id": f"p{i}", "position": 0} for i in range(4)],
    }
    # Old k=2 planner declines k=3; generalized planner succeeds!
    old_res = plan_position_owner_followup(
        spec,
        model_k3,
        records,
        SearchBudget(5000),
        previous_plan=prev_plan,
        previous_result=prev_res_k3,
        schemas=[],
        remaining_rows=2,
    )
    assert old_res["status"] == "no_unresolved_measured_owner_pair"

    gen_res = plan_generalized_position_followup(
        spec,
        model_k3,
        records,
        SearchBudget(5000),
        previous_plan=prev_plan,
        previous_result=prev_res_k3,
        remaining_rows=2,
    )
    assert gen_res["status"] == "proposed_position_experiment"
    assert gen_res["required_rows"] == 2
    assert gen_res["missing_corner_indices"] == [1, 3]
    assert set(gen_res["candidate_owners"]) == {"q_1", "q_2", "q_3"}


def test_end_to_end_cross_scale_n4_to_n5_and_scientific_revision():
    # Source library holds a confirmed N=4 directional ('right') window_total_event prior
    view4 = _schema(4, "src4")
    prior_n4 = _make_directional_prior(
        view4,
        concept_id="concept_n4_right",
        direction="right",
        radius=0.92,
        weights=(0.85, 0.85, 0.42),
        window=3,
    )

    # 1. Cross-scale N=4 -> N=5 on 'right' directional target with shifted radius (1.15)
    lib_baseline = KnowledgeLibrary()
    lib_baseline.meta_priors.append(deepcopy(prior_n4))
    cfg_baseline = ResearchConfig(
        observation_budget=280,
        compute_budget=240000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=False,
        transition_premise_audit=False,
    )
    agent_baseline = UnifiedResearchAgent(
        cfg_baseline, measurement_budget=5600, library=lib_baseline
    )
    with AnonymousObjectLab(
        seed=310501,
        objects=5,
        readout="window_total_event",
        relation_direction="right",
        relation_radius=1.15,
        actuator_curvature=0.2,
        name_seed=41,
        wiring_seed=53,
    ) as lab_base:
        res_base = agent_baseline.run(lab_base)
        heldout_base = lab_base.evaluation_rows(
            tuple(res_base["selected_channels"]), seed=9001, length=48, trajectories=1
        )
        mse_base = loss(
            Program.from_dict(res_base["program"]), heldout_base, res_base["fallback"], 0.5
        )

    lib_v10 = KnowledgeLibrary()
    lib_v10.meta_priors.append(deepcopy(prior_n4))
    cfg_v10 = ResearchConfig(
        observation_budget=280,
        compute_budget=240000,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
    )
    agent_v10 = UnifiedResearchAgent(cfg_v10, measurement_budget=5600, library=lib_v10)
    with AnonymousObjectLab(
        seed=310501,
        objects=5,
        readout="window_total_event",
        relation_direction="right",
        relation_radius=1.15,
        actuator_curvature=0.2,
        name_seed=41,
        wiring_seed=53,
    ) as lab_v10:
        res_v10 = agent_v10.run(lab_v10)
        heldout_v10 = lab_v10.evaluation_rows(
            tuple(res_v10["selected_channels"]), seed=9001, length=48, trajectories=1
        )
        mse_v10 = loss(
            Program.from_dict(res_v10["program"]), heldout_v10, res_v10["fallback"], 0.5
        )

    assert mse_base > 1e-4
    assert mse_v10 <= 1e-7

    # 2. Cross-scale N=4 ('right') -> N=5 ('left') requiring Scientific Premise Revision!
    lib_rev = KnowledgeLibrary()
    lib_rev.meta_priors.append(deepcopy(prior_n4))
    agent_rev = UnifiedResearchAgent(cfg_v10, measurement_budget=5600, library=lib_rev)
    with AnonymousObjectLab(
        seed=310502,
        objects=5,
        readout="window_total_event",
        relation_direction="left",
        relation_radius=1.10,
        actuator_curvature=0.2,
        name_seed=61,
        wiring_seed=73,
    ) as lab_left:
        res_rev = agent_rev.run(lab_left)
        heldout_left = lab_left.evaluation_rows(
            tuple(res_rev["selected_channels"]), seed=9002, length=48, trajectories=1
        )
        mse_rev = loss(
            Program.from_dict(res_rev["program"]), heldout_left, res_rev["fallback"], 0.5
        )

    assert mse_rev <= 1e-7
    audit_actions = [
        a for a in res_rev["actions"] if a.get("action") == "audit_transition_premises"
    ]
    assert len(audit_actions) >= 1
    assert audit_actions[-1]["verified_direction"] == "left"
    rev_metas = [
        (p.get("research_program") or {}).get("scientific_premise_revision")
        for p in lib_rev.meta_priors
        if (p.get("research_program") or {}).get("scientific_premise_revision") is not None
    ]
    assert len(rev_metas) >= 1
    rev_meta = rev_metas[-1]
    assert rev_meta["retracted_premises"][0]["premise_id"] == "interaction_direction:r_0"
    assert rev_meta["retracted_premises"][0]["direction"] == "right"
    assert rev_meta["replacement_premises"][0]["direction"] == "left"
