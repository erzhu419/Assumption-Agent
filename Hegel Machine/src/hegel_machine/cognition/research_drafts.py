"""Retain unfinished research procedures for subsequent paid evidence."""

from copy import deepcopy
from dataclasses import replace
from itertools import islice, permutations, product

from .abstractions import AbstractTemplate
from .memory import _object_schema
from .meta_prior_research import _sources
from .language import Program
from .research_program_inputs import abstract_research_inputs
from .research_programs import _entry, _Fitter, replay_research_program
from .scale_bindings import aggregate_role_binding, bind_measurement_arguments
from .models import SearchBudget


def remember_research_draft(library, selected, measured_spec, actions):
    """Keep a paid, unfinished procedure without creating a supported concept.

    Procedures are stored in measured coordinates, so replay can rebind their
    roles and independently refit their response on a later task. Interval
    evidence retains its original consuming response and trajectories; a later
    counterexample can withdraw those exclusions through relation research.
    """
    procedure = selected.get("research_program")
    steps = (procedure or {}).get("selection_steps", ())
    condition = selected.get("condition_uncertainty")
    numeric = (condition is not None and condition["status"] != "supported"
               and any(step["operation"] in ("discover_conditional_coordinates",
                                             "discover_decoded_conditional_coordinates")
                       for step in steps))
    if not numeric and not any(step["operation"] == "resolve_relation_intervals" for step in steps):
        return None
    if numeric:
        names = tuple(measured_spec.feature_names)
        roles = tuple(f"input_{index}" for index in range(len(names)))
        template = AbstractTemplate("research_inputs", "research_procedure",
            Program("object_sum", children=tuple(Program("feature", name=role) for role in roles)),
            roles, names, ())
        payload = deepcopy(procedure)
        payload.update(input_template=template.to_dict(), source_schema=None,
                       input_features=list(names), input_source_ids=[])
    else:
        payload = abstract_research_inputs(procedure, measured_spec, library)
    if payload is None:
        actions.append({"action": "defer_research_draft",
                        "reason": "measurement_inputs_unresolved"})
        return None
    record = next((item for item in library.research_drafts
                   if item.get("risk_method") is None
                   and item["source_task_id"] == measured_spec.task_id
                   and item["research_program"] == payload
                   and (not numeric or item.get("condition_uncertainty") == condition)), None)
    if record is not None:
        return record
    parent = next((item for item in library.research_drafts
                   if item["draft_id"] == selected.get("research_draft_id")), {})
    next_index = 1 + max((int(item["draft_id"].rsplit("_", 1)[1])
                          for item in library.research_drafts), default=0)
    record = {
        "draft_id": f"research_draft_{next_index:04d}",
        "source_task_id": measured_spec.task_id,
        "status": "unresolved", "risk_quantile": measured_spec.risk_quantile,
        "source_schema": deepcopy(payload["source_schema"]),
        "source_features": list(payload["input_features"]),
        "control_features": list(measured_spec.feature_bounds),
        "input_template": deepcopy(payload["input_template"]),
        "research_program": payload,
        "source_ids": list(dict.fromkeys(source for source in [
            *_sources(selected["candidate"].program), *payload.get("input_source_ids", ()),
            *parent.get("source_ids", ())] if source is not None)),
        "parent_draft_id": selected.get("research_draft_id"),
        "outcomes": [],
    }
    if numeric:
        record.update(condition_uncertainty=deepcopy(condition),
            observed_features=[name for name in names if name not in measured_spec.feature_bounds],
            input_mapping=dict(zip(names, names)))
    library.research_drafts.append(record)
    actions.append({"action": "preserve_research_draft",
        "draft_id": record["draft_id"], "parent_draft_id": record["parent_draft_id"],
        "source_task_id": measured_spec.task_id,
        "evidence_contexts": sum(len(step.get("evidence_contexts", ()))
            for step in payload["selection_steps"]),
        "status": "unresolved"})
    return record


def _numeric_mappings(record, template, spec, structural_transfer):
    """A command can transfer to another command, never to a paid reading."""
    command_roles = [role for role, name in zip(template.roles, template.source_features)
                     if name in record["control_features"]]
    observed_roles = [role for role, name in zip(template.roles, template.source_features)
                      if name in record["observed_features"]]
    commands = tuple(name for name in spec.feature_names if name in spec.feature_bounds)
    observed = tuple(name for name in spec.feature_names if name not in spec.feature_bounds)
    identity = dict(zip(template.roles, template.source_features))
    identity_valid = (all(identity[role] in commands for role in command_roles)
                      and all(identity[role] in observed for role in observed_roles))
    if identity_valid:
        yield identity
    if structural_transfer:
        for command_names, observed_names in product(
                permutations(commands, len(command_roles)),
                permutations(observed, len(observed_roles))):
            mapping = {**dict(zip(command_roles, command_names)),
                       **dict(zip(observed_roles, observed_names))}
            if not identity_valid or mapping != identity:
                yield mapping


def _response_terms(program, *, branch_basis=False):
    """Recover the coordinates actually used, rather than the search grammar."""
    if program.op in ("add", "sub"):
        return [term for child in program.children
                for term in _response_terms(child, branch_basis=branch_basis)]
    if program.op == "mul" and any(child.op == "const" for child in program.children):
        child = program.children[1 if program.children[0].op == "const" else 0]
        return _response_terms(child, branch_basis=branch_basis)
    if program.op == "window_sum":
        if branch_basis:
            return _response_terms(program.children[0], branch_basis=True)
        inner = _response_terms(program.children[0])
        terms = [replace(program, children=(term,)) for term in inner]
        # A decoded inner intercept has a reset-dependent window count; it
        # cannot be replaced by the fitter's constant output intercept.
        if any(node.op == "const" for node in _additive_atoms(program.children[0])):
            terms.append(replace(program, children=(Program("const", value=1.),)))
        return terms
    if branch_basis and program.op == "if":
        return [term for child in program.children[1:]
                for term in _response_terms(child, branch_basis=True)]
    if not branch_basis and program.op == "if":
        terms = []
        for side, branch in enumerate(program.children[1:]):
            atoms = _response_terms(branch)
            if any(node.op == "const" and node.value != 0.
                   for node in _additive_atoms(branch)):
                atoms.append(Program("const", value=1.))
            for atom in atoms:
                branches = ((atom, Program("const", value=0.)) if side == 0
                            else (Program("const", value=0.), atom))
                terms.append(Program("if", children=(program.children[0], *branches)))
        return terms
    if program.op == "const":
        return []
    return [program]


def _additive_atoms(program):
    if program.op in ("add", "sub"):
        return [term for child in program.children for term in _additive_atoms(child)]
    if program.op == "mul" and any(child.op == "const" for child in program.children):
        return _additive_atoms(program.children[1 if program.children[0].op == "const" else 0])
    return [program]


def _rebind_step(value, mapping):
    if isinstance(value, dict):
        if "op" in value:
            return bind_measurement_arguments(Program.from_dict(value), mapping).to_dict()
        return {key: ([mapping.get(name, name) for name in item] if key == "gate_names"
                      else _rebind_step(item, mapping)) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebind_step(item, mapping) for item in value]
    return value


def replay_numeric_research_drafts(agent, spec, training, budget, actions, *,
                                  evidence=(), validation=(), max_candidates=4):
    """Refit partial response structure before paying for a new broad grammar.

    Old thresholds are one competing computation, not restrictions on the new
    task. Current measured duration evidence can nominate new windows, branch
    placement and boundaries. Every fit still predicts the original response;
    its caller must compare validation and purchase fresh confirmation.
    """
    from .conditional_coordinates import (propose_conditional_coordinates,
                                          propose_experimental_conditional_coordinates)
    from .decoded_conditional_coordinates import propose_decoded_conditional_coordinates
    from .observed_condition_uncertainty import _gates
    from .evaluation import initial_prediction, loss

    rows, validation = list(training), list(validation)
    if (not agent.config.use_memory or agent.config.max_question_depth <= 0
            or _object_schema(spec) is not None or len(rows) < 8 or max_candidates < 1):
        return []
    fallback = initial_prediction(rows, spec.risk_quantile)
    excluded = set()

    def unrefuted(proposals, draft_id):
        retained = []
        for proposal in proposals:
            program = proposal["candidate"].program
            if agent.library.transfer_refuted(spec, program, fallback=fallback):
                if program not in excluded:
                    actions.append({"action": "exclude_refuted_draft_candidate",
                        "draft_id": draft_id, "task_id": spec.task_id,
                        "program": program.to_dict(), "fallback": fallback, "compute_cost": 0})
                    excluded.add(program)
            else:
                retained.append(proposal)
        return retained
    choices, families = [], set()
    for record in agent.library.research_drafts:
        if (record.get("condition_uncertainty") is None or record.get("resolved_concept_id")
                or record["risk_quantile"] != spec.risk_quantile):
            continue
        template = AbstractTemplate.from_dict(record["input_template"])
        mappings = list(islice(_numeric_mappings(record, template, spec,
            agent.config.structural_transfer), 2))
        # When public names and their command/reading roles agree, that binding
        # already answers the input question. Renamed tasks still try alternatives.
        if mappings and all(mappings[0][role] == name
                            for role, name in zip(template.roles, template.source_features)):
            mappings = mappings[:1]
        payload = record["research_program"]
        output = payload["graph"]["nodes"][payload["output_node"]]["answer"]
        saved = next((step for step in reversed(payload["selection_steps"])
            if step["operation"] in ("discover_conditional_coordinates",
                                     "discover_decoded_conditional_coordinates")), None)
        if saved is None:
            continue
        for mapping in mappings:
            features = {name: mapping[role]
                        for role, name in zip(template.roles, template.source_features)}
            program = bind_measurement_arguments(Program.from_dict(output), features)
            basis = tuple(dict.fromkeys(term for term in _response_terms(program, branch_basis=True)
                if term.history_depth == 0 and term.required_features()))
            gate_names = saved.get("gate_names", [saved.get("gate_coordinate", {}).get("name")])
            gates = tuple(features[name] for name in gate_names if name in features
                          and features[name] not in spec.feature_bounds)
            if not basis or not gates:
                continue
            family = frozenset(_response_terms(program)), frozenset(gates)
            if family in families:
                continue
            families.add(family)
            choices.append((record, features, program, saved, basis, gates))
    # Term ordering does not make a new response family; different windows and
    # branch placements do. When several
    # families compete, let their refits on the current paid response choose
    # which procedures receive deeper boundary and duration research.
    choices.sort(key=lambda item: (not any(trial.get("control") in item[5]
        and trial.get("gain", 0.) > 1e-10 for trial in evidence), len(item[4])))
    total = SearchBudget(min(80000, budget.remaining))
    limit = min(2, max_candidates)
    retained_fits = {}
    if len(choices) > limit and not evidence:
        screening = _Fitter(rows, total, spec.risk_quantile)
        alternatives = []
        for choice in choices:
            record, features, program, saved, basis, gates = choice
            start = total.spent
            fit = screening.fit(tuple(dict.fromkeys(_response_terms(program))))
            retained_fits[program] = fit
            alternatives.append({"draft_id": record["draft_id"],
                "source_task_id": record["source_task_id"],
                "response_basis_terms": len(basis), "observed_condition_fields": list(gates),
                "training_loss": fit["loss"] if fit is not None else None,
                "compute_cost": total.spent - start})
        choices.sort(key=lambda item: (not any(trial.get("control") in item[5]
            and trial.get("gain", 0.) > 1e-10 for trial in evidence),
            (0. if retained_fits[item[2]]["loss"] <= agent.config.minimum_improvement
             else retained_fits[item[2]]["loss"]) if retained_fits[item[2]] is not None
            else float("inf"), len(item[4])))
        actions.append({"action": "select_research_drafts_on_paid_response",
            "task_id": spec.task_id, "fitting_rows": len(rows),
            "comparison_basis": "original_response", "alternatives": alternatives,
            "selected_draft_ids": [], "compute_cost": total.spent})
    if not evidence:
        # Alternative bindings of one procedure must not take every research
        # slot before another available procedure has been considered.
        first, alternatives, sources = [], [], set()
        for choice in choices:
            identifier = choice[0]["draft_id"]
            if identifier in sources:
                alternatives.append(choice)
            else:
                first.append(choice)
                sources.add(identifier)
        choices = (first + alternatives)[:limit]
        if retained_fits:
            actions[-1]["selected_draft_ids"] = [item[0]["draft_id"] for item in choices]
    elif len(choices) > limit:
        # A new measured structure changes the procedure being tested. Old
        # parameter-fit loss cannot price that rewritten procedure. Continue
        # through distinct methods when its actual response fails validation.
        actions.append({"action": "revise_research_draft_agenda",
            "task_id": spec.task_id, "evidence": deepcopy(evidence),
            "validation_rows": len(validation),
            "comparison_basis": "current_measured_structure", "compute_cost": 0})
    claims = []
    for position, (record, features, program, saved, basis, gates) in enumerate(choices):
        current = [item for item in evidence if item.get("control") in gates
                   and item.get("gain", 0.) > 1e-10]
        allowance = (total.remaining if current else
                     total.remaining // (len(choices) - position))
        assessment_reserve = len(validation) * (max_candidates + 1)
        local = SearchBudget(max(0, allowance - assessment_reserve))
        proposals = []
        current_cuts = []
        for field, threshold, gate_width in dict.fromkeys(_gates(program)):
            if field in spec.feature_bounds or not local.spend(len(rows)):
                continue
            values = sorted({point[field] for row in rows for point in (row.features, *row.history)
                             if field in point})
            below = [value for value in values if value < threshold]
            above = [value for value in values if value >= threshold]
            current_cuts.append({"gate_coordinate": Program("feature", name=field).to_dict(),
                "threshold": threshold,
                "threshold_interval": [max(below) if below else threshold,
                                       min(above) if above else threshold],
                "width": saved["width"], "placement": ("inside_window" if gate_width > 1
                    else saved.get("placement", "outside_window"))})
        retained = (retained_fits[program] if program in retained_fits else
                    _Fitter(rows, local, spec.risk_quantile).fit(
                        tuple(dict.fromkeys(_response_terms(program)))))
        if retained is not None:
            steps = [_rebind_step(deepcopy(step), features)
                     for step in record["research_program"]["selection_steps"]]
            for step in steps:
                step["program"] = Program.from_dict(step["program"])
                step["historical"] = True
            for cut in current_cuts:
                steps.append({"operation": "refit_research_program",
                    "question": "Re-establish the retained cut interval using this task's paid readings",
                    "program": retained["program"], "loss": retained["loss"], **cut})
            steps.append({"operation": "refit_research_program",
                "question": "Retest the retained conditional computation on current original responses",
                "program": retained["program"], "loss": retained["loss"]})
            proposals.append(_entry(retained, steps, len(rows), local.spent))
        if current:
            # Rival fits from one trajectory are alternatives, not independent
            # boundary constraints that should be intersected.
            current = [min((item for item in current if item["control"] == gate),
                key=lambda item: item.get("conditional_loss", float("inf")))
                for gate in gates if any(item["control"] == gate for item in current)]
            proposals.extend(propose_experimental_conditional_coordinates(spec, rows, local,
                basis=basis, evidence=current, max_candidates=max_candidates))
        adequate = any(item["training_loss"] <= agent.config.minimum_improvement
                       for item in proposals)
        if not adequate and local.remaining:
            upper = max(1, max(len(row.history) for row in rows))
            widths = list(dict.fromkeys(width for item in current
                for width in item.get("width_candidates", [item["width"]])))
            widths = widths or list(dict.fromkeys([saved["width"], *range(1, upper + 1)]))
            inside = not current or any("inside_window" in item.get("placement_candidates",
                [item.get("placement", "inside_window")]) for item in current)
            if inside:
                decoded = SearchBudget(local.remaining * 3 // 4)
                proposals.extend(propose_decoded_conditional_coordinates(spec, rows, decoded,
                    basis=basis, gate_names=gates, width_candidates=widths,
                    duration_evidence=current, max_candidates=max_candidates))
                local.spend(decoded.spent)
            if (local.remaining and not any(item["training_loss"] <= agent.config.minimum_improvement
                                          for item in proposals)):
                proposals.extend(propose_conditional_coordinates(spec, rows, local,
                    basis=basis, gate_names=gates, max_candidates=max_candidates,
                    width_candidates=widths if current else None,
                    placement_candidates=("outside_window",)))
        # Withdrawal removes an already refuted offer. It does not purchase a
        # new broad search merely because that old offer had fitted its prefix.
        proposals = unrefuted(proposals, record["draft_id"])
        fitting_cost = local.spent
        if validation:
            assessment = SearchBudget(min(assessment_reserve, allowance - fitting_cost))
            checked = []
            for proposal in proposals:
                if not assessment.spend(len(validation)):
                    break
                proposal["validation_loss"] = loss(proposal["candidate"].program,
                    validation, fallback, spec.risk_quantile)
                checked.append(proposal)
            proposals = checked
            adequate = any(max(item["training_loss"], item["validation_loss"])
                <= agent.config.minimum_improvement for item in proposals)
            fitting_cost += assessment.spent
        total.spend(fitting_cost)
        actions.append({"action": "replay_research_draft", "draft_id": record["draft_id"],
            "source_task_id": record["source_task_id"], "task_id": spec.task_id,
            "compute_cost": fitting_cost, "response_basis_terms": len(basis),
            "observed_condition_fields": list(gates), "current_duration_evidence": bool(current),
            "fitting_priority": "current_supported_structure" if current else "retained_method",
            "comparison_basis": "original_response",
            "training_loss": min((item["training_loss"] for item in proposals), default=None),
            "validation_loss": min((item.get("validation_loss", float("inf"))
                for item in proposals), default=None) if validation else None,
            "input_mapping": dict(features),
            "status": "requires_current_validation" if proposals else "no_candidate"})
        for claim in proposals:
            claim["research_draft_id"] = record["draft_id"]
            claim["research_program"]["reused_procedure"] = True
            claim["research_program"]["search_compute"] = fitting_cost
            output_node = claim["research_program"]["output_node"]
            claim["research_program"]["graph"]["nodes"][output_node]["compute_cost"] = fitting_cost
            claims.append(claim)
        if current and adequate and any(item["training_loss"] <= agent.config.minimum_improvement
                                        for item in proposals):
            break
    budget.spend(total.spent)
    unique = {claim["candidate"].program: claim for claim in reversed(claims)}
    return sorted(unique.values(), key=lambda claim: (
        0. if max(claim["training_loss"], claim.get("validation_loss", 0.)) <= 1e-12
        else max(claim["training_loss"], claim.get("validation_loss", 0.)),
        claim["candidate"].program.size))[:max_candidates]


def replay_research_drafts(agent, spec, training, budget, actions, *, evidence=(),
                           max_candidates=None):
    """Rebind unfinished procedures and price their fit on current paid rows.

    Current training loss proposes an answer; the ordinary controller supplies
    validation, experiments and confirmation. A failed transfer remains local
    to that task, leaving the source draft available when its setting recurs.
    """
    if not agent.config.use_memory or agent.config.max_question_depth <= 0:
        return []
    limit = min(4, agent.config.max_candidates) if max_candidates is None else max_candidates
    claims = replay_numeric_research_drafts(agent, spec, training, budget, actions,
        evidence=evidence, max_candidates=limit)
    schema = _object_schema(spec)
    queues = []
    for record in reversed(agent.library.research_drafts):
        if record.get("resolved_concept_id") or record["risk_quantile"] != spec.risk_quantile:
            continue
        if record.get("risk_method") is not None:
            continue
        if record.get("condition_uncertainty") is not None:
            continue
        template = AbstractTemplate.from_dict(record["input_template"])
        aggregate = aggregate_role_binding(template, record["source_schema"], schema)
        prior = {**record, "object_schema": record["source_schema"]}
        command_binding = (agent._instrument_role_binding(template, prior, spec)
                           if schema is None else None)
        if schema is None and template.object_roles and command_binding is None:
            continue
        mappings = ([command_binding] if command_binding is not None else
                    [aggregate] if aggregate is not None else list(islice(
                        template.mappings(spec.feature_names,
                            structural_transfer=agent.config.structural_transfer,
                            object_schema=schema), 2)))
        if mappings:
            queues.append((record, template, iter(mappings)))
    while queues and len(claims) < limit and budget.remaining:
        remaining = []
        for record, template, mappings in queues:
            mapping = next(mappings, None)
            if mapping is None:
                continue
            remaining.append((record, template, mappings))
            features = {original: mapping[role]
                        for role, original in zip(template.roles, template.source_features)}
            before = budget.spent
            claim = replay_research_program(record["research_program"], training, budget,
                risk_quantile=spec.risk_quantile, feature_mapping=features)
            actions.append({"action": "replay_research_draft", "draft_id": record["draft_id"],
                "source_task_id": record["source_task_id"], "task_id": spec.task_id,
                "compute_cost": budget.spent - before,
                "training_loss": claim["training_loss"] if claim is not None else None,
                "status": "requires_current_validation" if claim is not None else "no_candidate"})
            if claim is not None:
                claim["research_draft_id"] = record["draft_id"]
                claims.append(claim)
            if len(claims) >= limit or not budget.remaining:
                break
        queues = remaining
    return claims
