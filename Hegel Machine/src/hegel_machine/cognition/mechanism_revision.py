"""Refit acquired responses after a local mechanism is removed or replaced."""

from copy import deepcopy
from dataclasses import replace
from itertools import islice
from statistics import mean, stdev

from .continuous_state import _sum_terms
from .evaluation import point_loss, predict
from .language import Program
from .models import Candidate, SearchBudget
from .nonlinear_parameters import fit_nonlinear_parameters
from .relation_revisions import _substitute
from .relation_uncertainty import RelationIntervalResearch, _bind
from .research_programs import _Fitter, _plan


def _open_calls(program, budget, sources):
    """Expose bound response bodies while retaining relation predicates."""
    if not budget.spend(1):
        return None
    if program.op == "call":
        source = program.definition.source_concept_id
        if source is not None and source not in sources:
            sources.append(source)
        body = _substitute(program.definition.body,
                           dict(zip(program.definition.arguments, program.children)))
        return _open_calls(body, budget, sources)
    children = tuple(_open_calls(child, budget, sources) for child in program.children)
    if any(child is None for child in children):
        return None
    return replace(program, children=children)


def _additive_terms(program):
    if program.op in {"add", "sub"}:
        return [term for child in program.children for term in _additive_terms(child)]
    if program.op == "mul":
        for index, child in enumerate(program.children):
            if child.op == "const":
                return [] if child.value == 0 else _additive_terms(program.children[1 - index])
    if program.op in {"window_sum", "event_delta"}:
        # These consumers are linear, including their reset behavior. A
        # nonlinear readout or an object reduction remains one whole term.
        return [replace(program, children=(term,)) for term in _additive_terms(program.children[0])]
    return [program]


def _mechanism(program):
    return program.history_depth > 0 or program.op in {"neighbor_sum", "relation_sum"} or any(
        _mechanism(child) for child in program.children)


def _signed_terms(program, negative=False):
    """Expose a local sum without changing coefficients inside its consumer."""
    if program.op in {"add", "sub"}:
        return [*_signed_terms(program.children[0], negative),
                *_signed_terms(program.children[1], negative != (program.op == "sub"))]
    return [Program("mul", children=(Program("const", value=-1.), program))
            if negative else program]


def _local_sums(program, nonlinear=False):
    # Object reductions and relation predicates are already indivisible learned
    # computations. Revise their response composition, not their input schema.
    if program.op in {"object_sum", "neighbor_sum", "relation_sum"}:
        return
    if nonlinear and program.op in {"add", "sub"}:
        yield program
        # One local sum can span several AST nodes. Its signed terms already
        # include all those additions; only a new consumer opens another sum.
        for term in _signed_terms(program):
            yield from _local_sums(term, nonlinear)
        return
    nonlinear = nonlinear or program.op in {"abs", "min", "max"} or (
        program.op == "mul" and all(child.op != "const" for child in program.children))
    for child in program.children:
        yield from _local_sums(child, nonlinear)


def _replace_shared(program, context, replacement):
    if program == context:
        return replacement, 1
    children, count = [], 0
    for child in program.children:
        revised, occurrences = _replace_shared(child, context, replacement)
        children.append(revised)
        count += occurrences
    return replace(program, children=tuple(children)), count


def _local_deletions(program):
    for context in dict.fromkeys(_local_sums(program)):
        terms = _signed_terms(context)
        for index, term in enumerate(terms):
            if not term.required_features() and not _mechanism(term):
                continue
            retained = terms[:index] + terms[index + 1:]
            replacement = retained[0]
            for other in retained[1:]:
                replacement = Program("add", children=(replacement, other))
            revised, count = _replace_shared(program, context, replacement)
            yield revised, {"removed_term": term, "edit_context": context,
                            "replacement_context": replacement, "shared_occurrences": count}


def _temporal_operators(program):
    current = [(program.op, program.lag)] if program.op in {"window_sum", "event_delta"} else []
    return [*current, *(key for child in program.children for key in _temporal_operators(child))]


def _without_temporal(program, operator):
    # Repeated consumers across additive terms describe the same temporal
    # mechanism. Remove one matching layer per path, preserving inner layers.
    if (program.op, program.lag) == operator:
        return program.children[0]
    return replace(program, children=tuple(_without_temporal(child, operator)
                                           for child in program.children))


def _relation_plan(fit, symbolic_terms, relation, count, cost, sources):
    template = _sum_terms([symbolic_terms[term] for term in fit["fit_terms"]],
                          fit["coefficients"][1:], fit["coefficients"][0])

    def symbols(node):
        return ({node.name} if node.op == "arg" else set()).union(
            *(symbols(child) for child in node.children))

    retained = symbols(template)
    parameters = [deepcopy(item) for item in relation.parameters if item["symbol"] in retained]
    if not parameters:
        return None
    contexts = []
    for prior in relation.evidence_contexts:
        if not retained.intersection(prior["parameters"]):
            continue
        context = deepcopy(prior)
        # Historical observations still constrain their complete old response.
        # A removed parameter keeps its historical value in that old context.
        fixed = {key: Program("const", value=value) for key, value in context["parameters"].items()
                 if key not in retained}
        context["parameter_template"] = _bind(
            Program.from_dict(context["parameter_template"]), fixed).to_dict()
        context["parameters"] = {key: value for key, value in context["parameters"].items()
                                 if key in retained}
        contexts.append(context)
    step = {"operation": "resolve_relation_intervals",
        "question": "Resolve the retained relation inside the revised response",
        "program": fit["program"], "loss": fit["loss"],
        "parameter_template": template.to_dict(), "parameters": parameters,
        "parameter_terms": [template.to_dict()], "evidence_contexts": contexts,
        "compact_replacement": False}
    plan = _plan(fit, [step], count, cost)
    plan["source_ids"] = list(sources)
    return plan


def _revision_entries(fitted, parent, original, relation, sources, count, cost):
    entries = []
    for fit, symbolic, kind, metadata in fitted:
        entry = {
            "candidate": Candidate(f"mechanism:{kind}:{len(entries)}:{cost}",
                fit["program"], "mechanism_revision", kind, parents=tuple(sources)),
            "kind": kind, "family_id": parent.get("family_id", "T03"),
            "training_loss": fit["loss"], "method_source_ids": list(sources),
            "mechanism_revision": {"original_program": original.to_dict(),
                "source_ids": list(sources), "status": "proposed", **metadata},
        }
        for key in ("research_program", "structure"):
            if parent.get(key) is not None:
                entry["mechanism_revision"][f"original_{key}"] = deepcopy(parent[key])
        if relation.parameters:
            plan = _relation_plan(fit, symbolic, relation, count, cost, sources)
            if plan is not None:
                entry["research_program"] = plan
        procedure = parent.get("research_program") or {}
        if (entry.get("research_program") is None and any(
                step["operation"] in ("discover_conditional_coordinates", "discover_composed_conditions")
                for step in procedure.get("selection_steps", []))):
            # Preserve how this response was researched, while rebuilding its
            # executable graph from the actual refit/deletion. Old coefficients
            # and confirmation status never describe this new candidate.
            steps = [{**deepcopy(step), "program": Program.from_dict(step["program"])}
                     for step in procedure["selection_steps"]]
            steps.append({"operation": "revise_conditional_response",
                "question": "Refit or simplify the measured conditional response",
                "program": fit["program"], "loss": fit["loss"],
                "revision_kind": kind, **deepcopy(metadata)})
            entry["research_program"] = _plan(fit, steps, count, cost)
            entry["research_program"]["source_ids"] = list(sources)
        if kind == "mechanism_temporal_refit":
            step = {"operation": "revise_temporal_scale",
                "question": "Retest the duration of the acquired temporal mechanism",
                "program": fit["program"], "loss": fit["loss"],
                "temporal_revisions": deepcopy(metadata["temporal_revisions"])}
            if entry.get("research_program") is None:
                entry["research_program"] = _plan(fit, [step], count, cost)
                entry["research_program"]["source_ids"] = list(sources)
            else:
                # The relation plan still owns its parameterized fit terms and
                # historical constraints; record the temporal question beside it.
                entry["research_program"].setdefault("selection_steps", []).append({
                    **step, "program": fit["program"].to_dict()})
        entries.append(entry)
    return entries


def propose_response_refits(spec, training, budget, *, parent, max_candidates=4, priors=()):
    """Refit the selected response's acquired basis, including inside time sums.

    An inherited response can retain biased coefficients even when its output
    genuinely varies on repetition. Refit those existing terms jointly and
    propose the least costly single-term omissions in training loss. No new
    basis or temporal operator is guessed here; adoption needs later evidence.
    """
    if parent is None or not training or max_candidates < 1:
        return []
    original = parent["candidate"].program
    sources = list(dict.fromkeys(source for source in (
        *parent.get("method_source_ids", ()), *parent["candidate"].parents,
        parent["candidate"].memory_id) if source is not None))
    relation = RelationIntervalResearch(original, priors, budget,
        structure=parent.get("structure"), research_program=parent.get("research_program"))
    sources = list(dict.fromkeys([*sources, *(item["source_concept_id"]
        for item in relation.parameters if item.get("source_concept_id") is not None)]))
    opened = _open_calls(relation.template, budget, sources)
    if opened is None:
        return []
    terms = tuple(dict.fromkeys(term for term in _additive_terms(opened) if term.op != "const"))
    if not terms or any(not term.required_features() <= set(spec.feature_names) for term in terms):
        return []
    values = {key: Program("const", value=value) for key, value in relation.values.items()}
    fitter = _Fitter(training, budget, spec.risk_quantile)
    fitted = []
    choices = [(None, terms)]
    if max_candidates > 1:
        choices.extend((index, terms[:index] + terms[index + 1:])
                       for index in range(len(terms)) if len(terms) > 1)
    for removed, retained in choices:
        numerical = tuple(_bind(term, values) for term in retained)
        symbolic = dict(zip(numerical, retained))
        if len(symbolic) != len(set(retained)):
            continue
        fit = fitter.fit(numerical)
        if fit is None:
            continue
        kind = "mechanism_full_refit" if removed is None else "mechanism_deletion"
        fitted.append((fit, symbolic, kind, {"removed_operator": None,
            "removed_term": _bind(terms[removed], values).to_dict() if removed is not None else None}))
    if not fitted or fitted[0][2] != "mechanism_full_refit":
        return []
    fitted = [fitted[0], *sorted(fitted[1:], key=lambda item:
        (item[0]["loss"], item[0]["program"].size))[:max_candidates - 1]]
    return _revision_entries(fitted, parent, original, relation, sources, len(training), budget.spent)


def propose_response_update(spec, training, original, repeated, budget, *, parent,
                            priors=(), fallback=0., minimum_improvement=1e-7):
    """Use an independent repeat to price a new response-learning opportunity.

    The original acquisition updates the existing basis; its independent repeat
    supplies the pilot gain. A positive, possibly uncertain pilot can justify
    further research, but never confirms the response on a new trajectory.
    """
    from .noise_replication import _inputs

    start = budget.spent
    training, original, repeated = list(training), list(original), list(repeated)
    fitted_rows = list({id(row): row for row in [*training, *original]}.values())
    event = {"training_rows": len(fitted_rows), "heldout_rows": len(repeated),
        "before_loss": None, "after_loss": None, "gain": None, "margin": None,
        "lower_bound": None, "compute_cost": 0, "requires_fresh_confirmation": False}

    def finish(reason, selected=None):
        event.update(reason=reason, compute_cost=budget.spent - start,
                     requires_fresh_confirmation=selected is not None)
        return {"selected": selected, "event": event}

    if parent is None or not training or len(repeated) < 2:
        return finish("insufficient_pilot_evidence")
    if len(original) != len(repeated):
        return finish("different_trajectory_length")
    fitted_ids = {id(row) for row in fitted_rows}
    if any(id(row) in fitted_ids for row in repeated):
        return finish("repeat_already_used_for_fitting")
    program = parent["candidate"].program
    reads_target = any(name == "__target__" for name, _ in program.observed_variables())
    if not budget.spend(len(original)):
        return finish("insufficient_budget")
    if any(_inputs(left, reads_target) != _inputs(right, reads_target)
           for left, right in zip(original, repeated)):
        return finish("unmatched_trajectory_inputs")
    comparison_cost = 3 * len(repeated)
    if budget.remaining <= comparison_cost:
        return finish("insufficient_budget")
    local = SearchBudget(budget.remaining - comparison_cost)
    entries = propose_response_refits(spec, fitted_rows, local, parent=parent,
                                      max_candidates=1, priors=priors)
    budget.spend(local.spent)
    if not entries:
        return finish("no_candidate")
    budget.spend(comparison_cost)
    candidate = entries[0]
    before = [point_loss(predict(program, row, fallback), row.target, spec.risk_quantile)
              for row in repeated]
    after = [point_loss(predict(candidate["candidate"].program, row, fallback),
                        row.target, spec.risk_quantile) for row in repeated]
    gains = [old - new for old, new in zip(before, after)]
    gain, margin = mean(gains), 3 * stdev(gains) / len(gains) ** .5
    event.update(before_loss=mean(before), after_loss=mean(after), gain=gain,
                 margin=margin, lower_bound=gain - margin)
    if gain <= minimum_improvement:
        return finish("no_positive_pilot_gain")
    candidate.update(validation_loss=event["after_loss"],
                     selection_basis="repeated_response_pilot")
    return finish("positive_response_update_pilot", candidate)


def propose_mechanism_revisions(spec, training, budget, *, parent, max_candidates=8, priors=()):
    """Refit after deleting a shared temporal layer or a local response term.

    The caller retains the original parent as an experimental rival. Neither
    fitting a zero coefficient nor deleting its term confirms irrelevance;
    returned candidates still require actual distinguishing measurements.
    ``max_candidates`` includes the complete refitted control.
    Local response coefficients can be revised inside the acquired nonlinear
    consumer; its structure and relation parameters remain explicit hypotheses.
    """
    if parent is None or not training or max_candidates < 2:
        return []
    original = parent["candidate"].program
    sources = list(dict.fromkeys(source for source in (
        *parent.get("method_source_ids", ()), *parent["candidate"].parents,
        parent["candidate"].memory_id) if source is not None))
    relation = RelationIntervalResearch(original, priors, budget,
        structure=parent.get("structure"), research_program=parent.get("research_program"))
    sources = list(dict.fromkeys([*sources, *(item["source_concept_id"]
        for item in relation.parameters if item.get("source_concept_id") is not None)]))
    opened = _open_calls(relation.template, budget, sources)
    if opened is None:
        return []
    terms = tuple(dict.fromkeys(term for term in _additive_terms(opened) if term.op != "const"))
    removable = [index for index, term in enumerate(terms) if _mechanism(term)]
    local = list(islice(_local_deletions(opened), max_candidates - 1))
    operators = list(dict.fromkeys(key for term in terms for key in _temporal_operators(term)))
    if (not terms or not (local or operators or len(terms) > 1 and removable) or any(
            not term.required_features() <= set(spec.feature_names) for term in terms)):
        return []

    fitter = _Fitter(training, budget, spec.risk_quantile)
    values = {key: Program("const", value=value) for key, value in relation.values.items()}
    choices = [(None, None, terms, None),
        *((None, None, tuple(dict.fromkeys(term for term in _additive_terms(revised)
                                          if term.op != "const")), edit)
          for revised, edit in local),
        *((None, operator, tuple(_without_temporal(term, operator) for term in terms), None)
          for operator in operators),
        *((index, None, terms[:index] + terms[index + 1:], None)
          for index in removable if len(terms) > 1)]
    fitted = []
    for removed, operator, retained, edit in choices[:max_candidates]:
        numerical = tuple(_bind(term, values) for term in retained)
        symbolic = dict(zip(numerical, retained))
        # Equal current columns cannot identify two distinct parameterized
        # branches. Do not silently discard one branch's range obligation.
        if len(symbolic) != len(set(retained)):
            continue
        fit = fitter.fit(numerical)
        if fit is None:
            continue
        kind = ("mechanism_full_refit" if removed is None and operator is None and edit is None
                else "mechanism_deletion")
        metadata = {
            "removed_term": _bind(terms[removed], values).to_dict() if removed is not None else None,
            "removed_operator": {"op": operator[0], "lag": operator[1]} if operator else None}
        if edit is not None:
            metadata.update({key: _bind(value, values).to_dict()
                if isinstance(value, Program) else value for key, value in edit.items()})
        fitted.append((fit, symbolic, kind, metadata))

    reference_loss = min((fit["loss"] for fit, *_ in fitted), default=float("inf"))
    if local and reference_loss > 1e-12:
        parameter_budget = SearchBudget(min(10000, budget.remaining * 2 // 3))
        revised = fit_nonlinear_parameters(opened, training, parameter_budget,
            values=values, quantile=spec.risk_quantile, reference_loss=reference_loss)
        budget.spend(parameter_budget.spent)
        if revised is not None:
            metadata = {"removed_term": None, "removed_operator": None,
                "parameter_revisions": [{key: _bind(value, values).to_dict()
                    if isinstance(value, Program) else value for key, value in edit.items()}
                    for edit in revised["parameter_revisions"]],
                "parameter_search_compute": parameter_budget.spent,
                "parameter_projection_trials": revised["projection_trials"],
                "parameter_iterations": revised["iterations"]}
            fitted.insert(1, (revised["fit"], revised["symbolic"], "mechanism_parameter_refit", metadata))

    reference_loss = min((fit["loss"] for fit, *_ in fitted), default=float("inf"))
    if any(operator == "window_sum" for operator, _ in operators) and reference_loss > 1e-12:
        from .temporal_scales import fit_temporal_scales
        temporal_budget = SearchBudget(min(6000, budget.remaining))
        revised = fit_temporal_scales(opened, training, temporal_budget,
            values=values, quantile=spec.risk_quantile, reference_loss=reference_loss)
        budget.spend(temporal_budget.spent)
        if revised is not None:
            metadata = {"removed_term": None, "removed_operator": None,
                "temporal_revisions": revised["temporal_revisions"],
                "temporal_trials": revised["trials"], "temporal_search_compute": temporal_budget.spent}
            fitted.insert(1, (revised["fit"], revised["symbolic"], "mechanism_temporal_refit", metadata))

    # Keep deletion's research question even when the full fit happens to
    # produce the same program. Those are not two independent predictions.
    return _revision_entries(fitted[:max_candidates], parent, original, relation,
                             sources, len(training), budget.spent)
