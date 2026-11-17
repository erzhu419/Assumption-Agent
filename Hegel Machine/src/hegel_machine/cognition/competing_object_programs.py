"""Retain executable laws attached to still-competing object assignments.

Every assignment fits its own coefficients to acquired observations. Agreement
on those observations does not establish algebraic equivalence or object
identity. Later observations can refute a frozen law and support a revision;
the old law and its counterexample remain in the research state.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations
import json
from math import ceil

from .equivalent_object_programs import canonical_object_program
from .evaluation import loss, predict, point_loss
from .language import Program
from .models import Candidate, SearchBudget
from .object_alternatives import enumerate_object_schemas
from .unified_structure import propose_structures


def _assess(hypothesis, rows, budget, tolerance, quantile, revision):
    """Assess the old executable, before fitting anything to the new response."""
    result = deepcopy(hypothesis)
    if not budget.spend(len(rows)):
        result["status"] = "unassessed"
        return result
    program = Program.from_dict(result["program"])
    predictions = [predict(program, row, 0.) for row in rows]
    errors = [point_loss(value, row.target, quantile) for value, row in zip(predictions, rows)]
    error = sum(errors) / len(rows)
    result["assessment_loss"] = error
    result["assessment_rows"] = len(rows)
    result["status"] = "supported" if error <= tolerance else "refuted"
    if error > tolerance:
        index = max(range(len(rows)), key=errors.__getitem__)
        result.setdefault("counterexamples", []).append({
            "revision": revision, "loss": error, "observation_count": len(rows),
            "observation": asdict(rows[index]), "prediction": predictions[index]})
    return result


def _entry(hypothesis):
    parents = tuple(dict.fromkeys(filter(None, (
        hypothesis.get("parent_hypothesis_id"),
        *hypothesis.get("method_source_ids", ()),
    ))))
    candidate = Candidate(hypothesis["hypothesis_id"], Program.from_dict(hypothesis["program"]),
        "representation", "Execute one evidence-supported object explanation", parents)
    entry = {"candidate": candidate, "kind": "competing_objects",
        "hypothesis_id": hypothesis["hypothesis_id"], "schema": hypothesis["schema"],
        "family_id": hypothesis["family_id"], "training_loss": hypothesis["training_loss"],
        "structure": hypothesis["structure"], "structure_search": hypothesis["structure_search"]}
    for key in ("research_program", "method_source_ids",
                "world_commitment_application", "scientific_premise_revision",
                "_consumed_object_provenance"):
        if hypothesis.get(key) is not None:
            entry[key] = deepcopy(hypothesis[key])
    return entry


def working_role_entry(fit, parent, *, schema_index, prefix, training_rows, parent_loss):
    """Keep the fitted graph under its assumed assignment until the role is resolved."""
    candidate = fit["candidate"]
    return {
        "candidate": replace(candidate, candidate_id=f"{prefix}:{schema_index}:{candidate.candidate_id}"),
        "kind": "working_role_revision", "family_id": fit["family_id"],
        "training_loss": fit["training_loss"],
        "working_role_revision": {"schema_index": schema_index, "structure": fit["structure"],
            "structure_search": fit["structure_search"], "training_rows": training_rows,
            "research_program": fit.get("research_program"), "fit_candidate_id": candidate.candidate_id,
            "parent_candidate_id": parent["candidate"].candidate_id, "parent_training_loss": parent_loss}}


def _refresh_radius_interval(hypothesis, rows, budget):
    """Absorb measured boundary information without refitting a supported law."""
    structure = hypothesis["structure"]
    interval = structure.get("radius_interval")
    radius = structure["fits"][0].get("radius")
    if interval is None or radius is None:
        return hypothesis
    fields = structure["schema"]["fields"]
    pairs = list(combinations(fields.values(), 2))
    if not budget.spend(len(rows) * len(pairs)):
        return hypothesis
    low, high = interval
    for left, right in pairs:
        for row in rows:
            values = row.features
            # The same observable identifiability condition used by structural
            # discovery: equal quantities or selectors cancel this edge.
            if (values[left["f"]] - values[right["f"]]) * (values[left["q"]] - values[right["q"]]) == 0.:
                continue
            distance = abs(values[left["x"]] - values[right["x"]])
            if distance < radius:
                low = max(low, distance)
            else:
                high = min(high, distance)
    if [low, high] == list(interval):
        return hypothesis
    result = deepcopy(hypothesis)
    result["structure"]["radius_interval"] = [low, high]
    result["structure"]["fits"][0]["radius_interval"] = [low, high]
    result["structure_search"] = "retained_boundary_update"
    return result


def _radius_alternatives(hypotheses, rows, fresh, budget, tolerance, quantile, limit):
    """Keep distinct executable edges inside the observed unresolved interval."""
    if not hypotheses or limit < 2:
        return hypotheses
    source = hypotheses[0]
    structure = source["structure"]
    interval, radius = structure.get("radius_interval"), structure["fits"][0].get("radius")
    if interval is None or radius is None or interval[1] <= interval[0]:
        return hypotheses
    original = Program.from_dict(source["program"])

    def at_radius(program, value):
        children = tuple(at_radius(child, value) for child in program.children)
        if program.op == "neighbor_sum" and children[0].op == "const" and children[0].value == radius:
            children = (Program("const", value=value), *children[1:])
        return replace(program, children=children)

    variants, seen = [], {canonical_object_program(original)}
    for numerator in (1, 3):
        value = interval[0] + (interval[1] - interval[0]) * numerator / 4
        program = at_radius(original, value)
        canonical = canonical_object_program(program)
        if canonical in seen:
            continue
        if not budget.spend(len(rows) + len(fresh)):
            break
        old_loss = loss(program, rows, 0., quantile)
        new_loss = loss(program, fresh, 0., quantile) if fresh else None
        if old_loss > tolerance or (new_loss is not None and new_loss > tolerance):
            continue
        seen.add(canonical)
        item = deepcopy(source)
        identifier = f"{source['hypothesis_id']}:radius:{numerator}"
        item.update(hypothesis_id=identifier, parent_hypothesis_id=source["hypothesis_id"],
            program=program.to_dict(), training_loss=old_loss, new_observation_loss=new_loss,
            structure_search="radius_alternative")
        item["structure"]["fits"][0].update(candidate_id=identifier, radius=value, training_loss=old_loss)
        item["structure"]["candidates"][0].update(candidate_id=identifier, program=program.to_dict())
        for claim in item["structure"].get("hypotheses", []):
            if claim["kind"] == "locality":
                claim["radius"] = value
        variants.append(item)
    # Each assignment keeps its first fitted law. Radius alternatives occupy
    # that assignment's own slots, never another assignment's research turn.
    retained, seen = [], set()
    for item in [source, *variants, *hypotheses[1:]]:
        canonical = canonical_object_program(Program.from_dict(item["program"]))
        if canonical not in seen:
            retained.append(item)
            seen.add(canonical)
        if len(retained) >= limit:
            break
    return retained


def research_competing_object_programs(spec, training, budget, *, previous=None,
                                      new_observations=(), transitions=(),
                                      tolerance=1e-7, max_candidates=8, preferred_schemas=(),
                                      response_parent=None, working_parent_loss=None,
                                      world_commitments=(), premise_audit=None,
                                      active_threshold_probing=True,
                                      allow_latent_state=True,
                                      allow_higher_order_latent=True,
                                      library=None,
                                      branch_reconciliation=False):
    """Fit and revise finite competing assignments within the supplied budget.

    ``training`` contains all paid observations for this task, including any
    ``new_observations``. The latter are scored against the frozen previous
    programs before refitting. The JSON-safe ``competition`` is returned even
    if no law survives; ``entries`` contains only currently supported laws.
    A caller with the partial parent's measured loss on these training rows
    may also compare its fitted revisions in a separate working pool.

    Search allowances are divided across remaining assignments, so the first
    assignment cannot consume all computation. ``exhaustive`` describes the
    assignment enumeration only; it never claims the finite program grammar is
    complete, nor authorizes a unique object schema.
    """
    previous = previous or {}
    start, rows = budget.spent, list(training)
    revision = int(previous.get("revision", -1)) + 1
    state = {"revision": revision, "source_model_id": None, "schemas": [],
        "exhaustive": False, "unresolved": [], "hypotheses": [],
        "unassessed_hypotheses": [],
        "counterexamples": deepcopy(previous.get("counterexamples", [])),
        "searches": [], "compute_cost": 0, "status": "no_supported_program"}
    model = spec.context.get("object_model")
    if not model or not rows or max_candidates < 1:
        return {"entries": [], "competition": state}
    model = json.loads(model) if isinstance(model, str) else model
    enumeration = enumerate_object_schemas(model, budget, max_schemas=max_candidates,
                                           preferred_schemas=preferred_schemas)
    state.update({key: enumeration[key] for key in (
        "schemas", "source_model_id", "exhaustive", "unresolved")})
    if "position_owner_coverage" in enumeration:
        state["position_owner_coverage"] = enumeration["position_owner_coverage"]
    state["enumeration_compute_cost"] = enumeration["compute_cost"]
    state["reused_schema_count"] = enumeration.get("reused_schema_count", 0)
    schemas = enumeration["schemas"]
    old = []
    fresh = list(new_observations)
    evidence = fresh or rows
    for hypothesis in [*previous.get("hypotheses", []), *previous.get("unassessed_hypotheses", [])]:
        if hypothesis["schema"] not in schemas:
            if not enumeration["exhaustive"]:
                state["unassessed_hypotheses"].append({**deepcopy(hypothesis),
                    "status": "unassessed", "reason": "assignment_search_incomplete"})
                continue
            # Membership evidence can exclude an assignment independently of
            # this readout; retain its former program without claiming failure.
            state["counterexamples"].append({**deepcopy(hypothesis),
                "status": "assignment_no_longer_permitted", "revision": revision})
            continue
        assessed = _assess(hypothesis, evidence, budget, tolerance, spec.risk_quantile, revision)
        if assessed["status"] == "supported":
            if evidence == rows:
                assessed["training_loss"] = assessed["assessment_loss"]
                assessed["training_rows"] = len(rows)
            elif budget.spend(len(rows)):
                # Reuse is justified by the entire current paid data, including
                # older trajectories that can disagree with the latest batch.
                error = loss(Program.from_dict(assessed["program"]), rows, 0., spec.risk_quantile)
                assessed["training_loss"] = error
                assessed["training_rows"] = len(rows)
                if error > tolerance:
                    assessed["status"] = "refuted"
                    assessed["counterexamples"].append({"revision": revision,
                        "loss": error, "observation_count": len(rows), "scope": "all_paid_data"})
            else:
                assessed["status"] = "unassessed"
        old.append(assessed)
        if assessed["status"] == "refuted":
            state["counterexamples"].append(assessed)
        elif assessed["status"] == "unassessed":
            state["unassessed_hypotheses"].append(assessed)

    # Keep at least one fitted law per searched assignment before retaining
    # additional families. No coefficient or radius is copied between schemas.
    per_schema = max(1, ceil(max_candidates / max(1, len(schemas))))
    surviving_laws = any(item["status"] == "supported" for item in old)
    earlier_schemas = previous.get("schemas", [])
    earlier_searches = {item["schema_index"]: item for item in previous.get("searches", [])}
    choices, working_revisions = [], []
    all_consumption_events = []
    last_branch_dispatch_telemetry = None
    for index, schema in enumerate(schemas):
        allowance = budget.remaining // (len(schemas) - index)
        local_budget = SearchBudget(allowance)
        local_spec = replace(spec, context={**spec.context, "object_schema": schema})
        prior = [item for item in old if item["schema"] == schema]
        parent = min(prior, key=lambda item: item["training_loss"]) if prior else None
        supported = [item for item in prior if item["status"] == "supported"]
        earlier_index = earlier_schemas.index(schema) if schema in earlier_schemas else None
        earlier_search = earlier_searches.get(earlier_index, {})
        searched_before = (earlier_search.get("compute_cost", 0) > 0 or
            earlier_search.get("search") == "deferred_while_supported_laws_remain")
        if surviving_laws and not prior and searched_before:
            # A still-supported law gives the next experiment something to
            # distinguish. Repeating an unsuccessful assignment's whole search
            # now adds no executable rival. Keep that assignment unresolved;
            # when all current laws fail it receives a fresh research turn.
            choices.append([])
            state["searches"].append({"schema_index": index, "compute_cost": 0,
                "candidate_count": 0, "search_complete": False,
                "search": "deferred_while_supported_laws_remain"})
            continue
        if supported:
            # No new fit is needed to keep an executable that the full paid
            # evidence still supports. Other assignments retain their turn.
            retained = []
            for rank, item in enumerate(supported[:per_schema]):
                frozen = {**item, "hypothesis_id": f"object_competition:{revision}:{index}:{rank}",
                    "parent_hypothesis_id": item["hypothesis_id"],
                    "structure_search": "retained_frozen_law"}
                retained.append(_refresh_radius_interval(frozen, rows, local_budget))
            retained = _radius_alternatives(retained, rows, fresh, local_budget,
                tolerance, spec.risk_quantile, per_schema)
            choices.append(retained)
            budget.spend(local_budget.spent)
            state["searches"].append({"schema_index": index, "compute_cost": local_budget.spent,
                "candidate_count": len(retained), "search_complete": False,
                "search": "retained_frozen_law"})
            continue
        # A repaired program must explain the very observations that caused
        # revision, even when a long old prefix dilutes their average error.
        # The structural proposer returns at most eight programs per schema.
        validation_allowance = min(allowance // 2, len(fresh) * (16 if parent else 8))
        validation_budget = SearchBudget(validation_allowance)
        search_budget = SearchBudget(allowance - validation_allowance)

        def supports_current_evidence(fit):
            if fit["training_loss"] > tolerance:
                return False
            if fresh and "new_observation_loss" not in fit:
                if not validation_budget.spend(len(fresh)):
                    return False
                fit["new_observation_loss"] = loss(fit["candidate"].program, fresh, 0., spec.risk_quantile)
            return fit.get("new_observation_loss", 0.) <= tolerance

        fits, fit_attempts = [], []

        def record_attempt(stage, attempted, work, spent_before=0):
            # Keep references until the existing admission loop has attached
            # any paid fresh-data score. Only scalar diagnostics are persisted.
            fit_attempts.append({"stage": stage, "fits": attempted,
                "compute_limit": work.limit, "compute_available_at_start": work.limit - spent_before,
                "compute_cost": work.spent - spent_before,
                "compute_remaining": work.remaining,
                "last_denied_request": deepcopy(work.last_denied_request)})

        if parent:
            # A refuted radius or family must not use the whole assignment's
            # allowance before alternative families get a chance to explain it.
            parent_budget = SearchBudget(search_budget.remaining // 3)
            fits = propose_structures(local_spec, rows, parent_budget,
                parent=parent, tolerance=tolerance, transitions=transitions)
            record_attempt("parent_revision", fits, parent_budget)
            search_budget.spend(parent_budget.spent)
        if (world_commitments or (branch_reconciliation and library is not None)) and not any(supports_current_evidence(fit) for fit in fits):
            from .world_structure_commitments import (
                commitment_applicable_triggers,
                commitment_verified_radius_bounds,
                instantiate_commitment_on_schema,
                is_commitment_empirically_consistent,
                propose_scientific_premise_revision,
            )
            commitment_budget = SearchBudget(max(1, search_budget.remaining * 2 // 3))
            commitment_fits = []
            schema_x_to_q = {
                group["x"]: group["q"]
                for group in schema.get("fields", {}).values()
                if "x" in group and "q" in group
            }
            ctl_sig = ",".join(sorted(local_spec.feature_bounds)[:4])
            task_sig = f"{local_spec.task_id}:{ctl_sig}" if ctl_sig else local_spec.task_id
            consistent_comms = [
                c
                for c in world_commitments
                if is_commitment_empirically_consistent(c, premise_audit, task_id=task_sig)
            ]
            parent_id_before = parent["hypothesis_id"] if parent else None

            if branch_reconciliation and library is not None:
                from .branch_reconciliation import (
                    dispatch_branches_and_families_for_target,
                    instantiate_parametric_family_on_target,
                )
                disp = dispatch_branches_and_families_for_target(
                    library=library,
                    consistent_comms=consistent_comms,
                    transitions=transitions,
                    schema=schema,
                    audit=premise_audit,
                    task_sig=task_sig,
                    comm_budget=commitment_budget,
                    tolerance=tolerance,
                )
                consistent_comms = disp["ordered_consistent_comms"]
                for gf in disp["scheduled_families"]:
                    fam_entries = instantiate_parametric_family_on_target(
                        gf, local_spec, rows, schema, commitment_budget,
                        audit_report=premise_audit, transitions=transitions, tolerance=tolerance,
                    )
                    src_ids_gf = list(gf.get("source_concept_ids") or gf.get("source_branch_ids") or ())
                    for e in fam_entries:
                        e["_consumed_object_provenance"] = {
                            "object_id": gf["commitment_id"],
                            "semantic_version": gf.get("semantic_group_fingerprint") or gf["commitment_id"],
                            "source_concept_ids": src_ids_gf,
                            "consumption_site": "pass0_parametric_family_instantiation",
                            "parent_before": parent_id_before,
                        }
                    gen_cids = [e["candidate"].candidate_id for e in fam_entries if "candidate" in e]
                    if index == 0:
                        all_consumption_events.append({
                            "event_stage": "candidate_generation",
                            "object_id": gf["commitment_id"],
                            "semantic_version": gf.get("semantic_group_fingerprint") or gf["commitment_id"],
                            "source_concept_ids": src_ids_gf,
                            "consumption_site": "pass0_parametric_family_instantiation",
                            "generated_candidate_ids": gen_cids,
                            "selected_candidate_id": None,
                            "adopted_confirmed": False,
                            "parent_before": parent_id_before,
                            "parent_after": parent_id_before,
                            "proposed_probe_ids": [],
                            "executed_probe_ids": [],
                            "probe_before": None,
                            "probe_after": None,
                        })
                    commitment_fits.extend(fam_entries)
                    if any(supports_current_evidence(item) for item in commitment_fits):
                        break
                if index == 0:
                    last_branch_dispatch_telemetry = disp["telemetry"]

            if not any(supports_current_evidence(item) for item in commitment_fits):
                for comm in consistent_comms:
                    if commitment_budget.remaining <= 0:
                        break
                    audit_bounds = commitment_verified_radius_bounds(
                        comm, premise_audit, schema_x_to_q
                    )
                    pred_overrides = None
                    if (
                        premise_audit is not None
                        and isinstance(premise_audit.get("verified_edge_condition"), dict)
                        and comm.get("radius_parameters")
                    ):
                        pred_overrides = {
                            comm["radius_parameters"][0]["symbol"]: premise_audit["verified_edge_condition"]
                        }
                    comm_entries = instantiate_commitment_on_schema(
                        local_spec, rows, schema, comm, commitment_budget,
                        radius_bounds=audit_bounds,
                        predicate_overrides=pred_overrides,
                        audit_report=premise_audit,
                        transitions=transitions,
                        active_threshold_probing=active_threshold_probing,
                        allow_latent_state=allow_latent_state,
                        allow_higher_order_latent=allow_higher_order_latent,
                        tolerance=tolerance,
                    )
                    src_ids_c = list(comm.get("source_concept_ids") or ())
                    for e in comm_entries:
                        e["_consumed_object_provenance"] = {
                            "object_id": comm["commitment_id"],
                            "semantic_version": comm.get("semantic_group_fingerprint"),
                            "source_concept_ids": src_ids_c,
                            "consumption_site": "pass1_scope_instantiation",
                            "parent_before": parent_id_before,
                        }
                    if index == 0:
                        all_consumption_events.append({
                            "event_stage": "candidate_generation",
                            "object_id": comm["commitment_id"],
                            "semantic_version": comm.get("semantic_group_fingerprint"),
                            "source_concept_ids": src_ids_c,
                            "consumption_site": "pass1_scope_instantiation",
                            "generated_candidate_ids": [e["candidate"].candidate_id for e in comm_entries if "candidate" in e],
                            "selected_candidate_id": None,
                            "adopted_confirmed": False,
                            "parent_before": parent_id_before,
                            "parent_after": parent_id_before,
                            "proposed_probe_ids": [],
                            "executed_probe_ids": [],
                            "probe_before": None,
                            "probe_after": None,
                        })
                    commitment_fits.extend(comm_entries)
                    if any(supports_current_evidence(item) for item in commitment_fits):
                        break

            if not any(supports_current_evidence(item) for item in commitment_fits):
                for comm in world_commitments:
                    if commitment_budget.remaining <= 0:
                        break
                    triggers = commitment_applicable_triggers(comm, premise_audit)
                    if triggers or premise_audit is None:
                        comm_entries = propose_scientific_premise_revision(
                            comm, premise_audit, local_spec, rows, schema,
                            commitment_budget, transitions=transitions,
                            active_threshold_probing=active_threshold_probing,
                            allow_latent_state=allow_latent_state,
                            allow_higher_order_latent=allow_higher_order_latent,
                            tolerance=tolerance,
                        )
                        src_ids_r = list(comm.get("source_concept_ids") or ())
                        for e in comm_entries:
                            e["_consumed_object_provenance"] = {
                                "object_id": comm["commitment_id"],
                                "semantic_version": comm.get("semantic_group_fingerprint"),
                                "source_concept_ids": src_ids_r,
                                "consumption_site": "pass2_premise_revision",
                                "parent_before": comm["commitment_id"],
                            }
                        if index == 0:
                            all_consumption_events.append({
                                "event_stage": "candidate_generation",
                                "object_id": comm["commitment_id"],
                                "semantic_version": comm.get("semantic_group_fingerprint"),
                                "source_concept_ids": src_ids_r,
                                "consumption_site": "pass2_premise_revision",
                                "generated_candidate_ids": [e["candidate"].candidate_id for e in comm_entries if "candidate" in e],
                                "selected_candidate_id": None,
                                "adopted_confirmed": False,
                                "parent_before": comm["commitment_id"],
                                "parent_after": comm["commitment_id"],
                                "proposed_probe_ids": [],
                                "executed_probe_ids": [],
                                "probe_before": None,
                                "probe_after": None,
                            })
                        commitment_fits.extend(comm_entries)
                        if any(supports_current_evidence(item) for item in commitment_fits):
                            break
            if (
                not any(supports_current_evidence(item) for item in commitment_fits)
                and premise_audit is not None
                and premise_audit.get("verified_edge_condition") == "unconditional"
                and commitment_budget.remaining > 0
            ):
                # Provisional Tier-0 unconditional preference was empirically refuted by paid trajectory rows;
                # calibrate any archived conditional commitments on this schema's transitions + rows.
                for comm in world_commitments:
                    if commitment_budget.remaining <= 0:
                        break
                    rp_c = (comm.get("radius_parameters") or [{}])[0]
                    if rp_c.get("edge_condition") in (None, "unconditional"):
                        continue
                    if (
                        premise_audit.get("verified_direction")
                        and rp_c.get("direction") != premise_audit.get("verified_direction")
                    ):
                        continue
                    comm_entries = instantiate_commitment_on_schema(
                        local_spec, rows, schema, comm, commitment_budget,
                        radius_bounds=None,
                        predicate_overrides=None,
                        audit_report=None,
                        transitions=transitions,
                        active_threshold_probing=active_threshold_probing,
                        allow_latent_state=allow_latent_state,
                        allow_higher_order_latent=allow_higher_order_latent,
                        tolerance=tolerance,
                    )
                    src_ids_c2 = list(comm.get("source_concept_ids") or ())
                    for e in comm_entries:
                        e["_consumed_object_provenance"] = {
                            "object_id": comm["commitment_id"],
                            "semantic_version": comm.get("semantic_group_fingerprint"),
                            "source_concept_ids": src_ids_c2,
                            "consumption_site": "pass1_scope_instantiation",
                            "parent_before": parent_id_before,
                        }
                    if index == 0:
                        all_consumption_events.append({
                            "event_stage": "candidate_generation",
                            "object_id": comm["commitment_id"],
                            "semantic_version": comm.get("semantic_group_fingerprint"),
                            "source_concept_ids": src_ids_c2,
                            "consumption_site": "pass1_scope_instantiation",
                            "generated_candidate_ids": [e["candidate"].candidate_id for e in comm_entries if "candidate" in e],
                            "selected_candidate_id": None,
                            "adopted_confirmed": False,
                            "parent_before": parent_id_before,
                            "parent_after": parent_id_before,
                            "proposed_probe_ids": [],
                            "executed_probe_ids": [],
                            "probe_before": None,
                            "probe_after": None,
                        })
                    commitment_fits.extend(comm_entries)
                    if any(supports_current_evidence(item) for item in commitment_fits):
                        break
            record_attempt("world_commitment", commitment_fits, commitment_budget)
            search_budget.spend(commitment_budget.spent)
            fits = [*fits, *commitment_fits]
        if not any(supports_current_evidence(fit) for fit in fits):
            before_search = search_budget.spent
            # A new role assignment must test the retained response, not lose
            # its fitted columns and restart from unrelated default responses.
            # An existing same-schema hypothesis retains its original path.
            retained_response = response_parent if not prior else None
            structural_fits = propose_structures(local_spec, rows, search_budget,
                parent=retained_response, tolerance=tolerance, transitions=transitions)
            record_attempt("parent_response" if retained_response else "structural_search",
                           structural_fits, search_budget, before_search)
            fits = [*fits, *structural_fits]
        local_budget.spend(search_budget.spent)
        accepted, seen = [], set()
        for fit in fits:
            if not supports_current_evidence(fit):
                continue
            program = fit["candidate"].program
            canonical = canonical_object_program(program)
            if canonical in seen:
                continue
            seen.add(canonical)
            identifier = f"object_competition:{revision}:{index}:{len(accepted)}"
            same_family = [item for item in prior if
                item["structure"]["fits"][0]["family"] == fit["structure"]["fits"][0]["family"]]
            ancestor = min(same_family, key=lambda item: item["training_loss"]) if same_family else parent
            hypothesis = {"hypothesis_id": identifier, "schema": schema,
                "program": program.to_dict(), "training_loss": fit["training_loss"],
                "new_observation_loss": fit.get("new_observation_loss"),
                "training_rows": len(rows), "status": "supported", "counterexamples": [],
                "family_id": fit["family_id"], "structure": fit["structure"],
                "structure_search": fit["structure_search"],
                "parent_hypothesis_id": ancestor["hypothesis_id"] if ancestor else None}
            for extra_key in ("research_program", "method_source_ids",
                              "world_commitment_application", "scientific_premise_revision",
                              "_consumed_object_provenance"):
                if fit.get(extra_key) is not None:
                    hypothesis[extra_key] = deepcopy(fit[extra_key])
            accepted.append(hypothesis)
            if len(accepted) >= per_schema:
                break
        local_budget.spend(validation_budget.spent)
        accepted = _radius_alternatives(accepted, rows, fresh, local_budget,
            tolerance, spec.risk_quantile, per_schema)
        budget.spend(local_budget.spent)
        if (working_parent_loss is not None and response_parent
                and response_parent.get("partial_object_knowledge") and response_parent.get("research_program")):
            # Only a fit actually initialized from this response can claim
            # improvement over it. Keep the native executable before the
            # diagnostic projection below drops its coefficients.
            improving = [fit for attempt in fit_attempts if attempt["stage"] == "parent_response"
                         for fit in attempt["fits"] if tolerance < fit["training_loss"] < working_parent_loss]
            if improving:
                fit = min(improving, key=lambda item: item["training_loss"])
                working_revisions.append(working_role_entry(fit, response_parent, schema_index=index,
                    prefix=f"working_role:{revision}", training_rows=len(rows), parent_loss=working_parent_loss))
        for attempt in fit_attempts:
            diagnostics = []
            for fit in attempt["fits"]:
                support = ("training_loss_above_tolerance" if fit["training_loss"] > tolerance else
                    "fresh_loss_not_evaluated" if fresh and "new_observation_loss" not in fit else
                    "new_observation_loss_above_tolerance" if fit.get("new_observation_loss", 0.) > tolerance else
                    "supported_on_evaluated_rows")
                diagnostics.append({"candidate_id": fit["candidate"].candidate_id,
                    "family_id": fit["family_id"], "family": fit["structure"]["fits"][0]["family"],
                    "training_loss": fit["training_loss"],
                    "new_observation_loss": fit.get("new_observation_loss"),
                    "search_complete": fit["structure"].get("search_complete", False),
                    "structure_search": fit["structure_search"], "support_status": support})
            attempt["fits"] = diagnostics
        state["searches"].append({"schema_index": index, "compute_cost": local_budget.spent,
            "candidate_count": len(fits), "search_complete": bool(fits) and all(
                item["structure"].get("search_complete", False) for item in fits),
            "fit_attempts": fit_attempts,
            "fresh_validation_budget": {"compute_limit": validation_budget.limit,
                "compute_cost": validation_budget.spent,
                "last_denied_request": deepcopy(validation_budget.last_denied_request)}})
        choices.append(accepted)

    # Round-robin retention avoids erasing an assignment just because another
    # assignment generated more fitting families on an underdetermined prefix.
    for rank in range(per_schema):
        for accepted in choices:
            if rank < len(accepted) and len(state["hypotheses"]) < max_candidates:
                state["hypotheses"].append(accepted[rank])
    active = state["hypotheses"]
    canonical = {canonical_object_program(Program.from_dict(item["program"])) for item in active}
    state["program_classes"] = len(canonical)
    state["status"] = ("competing" if len(canonical) > 1 else
                       "supported" if active else "no_supported_program")
    state["compute_cost"] = budget.spent - start
    entries = [_entry(item) for item in active]
    for entry in entries:
        entry["object_competition"] = state
    result = {
        "entries": entries,
        "competition": state,
        "consumption_events": all_consumption_events,
        "branch_dispatch_telemetry": last_branch_dispatch_telemetry,
    }
    if working_parent_loss is not None:
        working = []
        for revision_entry in sorted(working_revisions, key=lambda item: item["training_loss"]):
            if len(entries) + len(working) >= min(4, max_candidates):
                break
            key = canonical_object_program(revision_entry["candidate"].program)
            if key not in canonical:
                canonical.add(key)
                working.append(revision_entry)
        result["working_entries"] = working
    return result
