"""Public object representations and boundary experiments for unified research."""

import json
from dataclasses import replace

from .evaluation import loss
from .language import Program
from .models import Probe, SearchBudget
from .structural_discovery import (
    StructuralDiscovery, _discover_structure, boundary_probe, confirm_structure, discover_structure,
    distinguishing_probes, _relation_distances,
)
from .structure_lab import ObjectSchema


_FAMILIES = {
    "conserved_total": "T15", "conserved_combination": "T15",
    "focal_coordinates": "T04", "object_specific": "T04",
    "local_exchange": "T08", "local_and_global": "T08",
    "object_specific_exchange": "T08", "event_change": "T11",
    "window_exchange": "T12",
    "joint_current": "T08", "joint_event": "T11",
    "joint_window": "T12", "joint_window_event": "T12",
}


def _research_response_terms(parent, budget):
    procedure = (parent or {}).get("research_program")
    if procedure and (parent or {}).get("partial_object_knowledge"):
        return [Program.from_dict(procedure["graph"]["nodes"][key]["answer"])
                for key in procedure["term_nodes"]]
    if (parent or {}).get("candidate") is not None:
        from .response_basis import relation_free_response_terms
        if procedure and any(step["operation"] == "resolve_relation_intervals"
                for step in procedure.get("selection_steps", [])):
            from .structural_research_program import research_program_fit_contract
            fitted = research_program_fit_contract(procedure)
            if fitted["program"] == parent["candidate"].program:
                # A saved zero-weight response coordinate can become useful
                # when a new relation range is fitted. Keep the actual graph
                # columns, rather than recovering only the current nonzeros.
                return list(dict.fromkeys(term for column in fitted["fit_terms"]
                    for term in relation_free_response_terms(column, budget)))
        return relation_free_response_terms(parent["candidate"].program, budget)
    return []


def propose_structures(spec, training, budget, *, parent=None, tolerance=1e-8,
                       transitions=()):
    """Fit competing structures from observed fields, with no apparatus labels.

    A previously supported radius interval permits a cheaper local revision.
    Its family must still explain all supplied evidence; otherwise the ordinary
    structural search is reopened. Every returned entry owns its experiment
    metadata, so a validation winner never inherits the training winner's
    interaction boundary.
    """
    payload = spec.context.get("object_schema")
    if payload is None or not training:
        return []
    schema = ObjectSchema.from_dict(json.loads(payload) if isinstance(payload, str) else payload)
    if not set(schema.feature_names).issubset(spec.feature_names):
        return []
    discovery, search, retained = None, "full", []
    previous = (parent or {}).get("structure")
    if previous and ObjectSchema.from_dict(previous["schema"]) == schema:
        family = previous["fits"][0]["family"]
        response_terms = [Program.from_dict(term) for term in previous["fits"][0].get("response_terms", [])]
        interval = previous.get("radius_interval")
        if interval and interval[1] > interval[0]:
            low, high = interval
            radii = [low + (high - low) * fraction for fraction in (.25, .5, .75)]
            # A fitted cell is a local question, not evidence against both
            # neighboring cells. Refit those neighbors on the current paid
            # response as well, including weak edges hidden by average error.
            distances, _ = _relation_distances(training, schema, budget)
            if distances:
                left = max((value for value in distances if value < low), default=0.)
                right = min((value for value in distances if value > high),
                            default=distances[-1] + 1e-6)
                if left < low:
                    radii.append((left + low) / 2)
                if high < right:
                    radii.append((high + right) / 2)
            narrowed = discover_structure(training, schema, budget=budget, transitions=transitions,
                                          radius_candidates=radii, families=(family,), response_terms=response_terms)
            if narrowed.fits and min(fit["training_loss"] for fit in narrowed.fits) <= tolerance:
                discovery, search = narrowed, "boundary_revision"
        if discovery is None:
            # The family can remain useful even when its current radius has no
            # consistent interval. Resolve it against all acquired evidence
            # before spending again on unrelated structural explanations.
            local = SearchBudget(budget.remaining // 2)
            focused = discover_structure(training, schema, budget=local, transitions=transitions,
                                          families=(family,), response_terms=response_terms)
            budget.spend(local.spent)
            if focused.fits and min(fit["training_loss"] for fit in focused.fits) <= tolerance:
                discovery, search = focused, "family_revision"
            else:
                retained.append((focused, "family_revision"))
    if discovery is None:
        response_terms = _research_response_terms(parent, budget)
        # Resolve this response's relation in its own research allowance.
        # Changing the response is a separately proposed research direction.
        search_response = _discover_structure if response_terms else discover_structure
        discovery = search_response(training, schema, budget=budget, transitions=transitions,
                                    response_terms=response_terms)
    retained.append((discovery, search))
    entries = []
    for discovery, search in retained:
        entries.extend(_entries(discovery, search, spec, training, budget, tolerance))
    return entries


def propose_response_revisions(spec, training, budget, *, parent, tolerance=1e-8,
                               transitions=()):
    """Reopen a response after its existing structure conflicts with paid rows."""
    payload = spec.context.get("object_schema")
    if payload is None or not training or parent is None:
        return []
    schema = ObjectSchema.from_dict(json.loads(payload) if isinstance(payload, str) else payload)
    if not set(schema.feature_names).issubset(spec.feature_names):
        return []
    previous = parent.get("structure") or {}
    fits = previous.get("fits", [])
    response_terms = [Program.from_dict(term) for term in fits[0].get("response_terms", [])] if fits else []
    if not response_terms:
        response_terms = _research_response_terms(parent, budget)
    if not response_terms or any(not term.required_features() <= set(spec.feature_names)
                                 for term in response_terms):
        return []
    discovery = discover_structure(training, schema, budget=budget, transitions=transitions,
                                   response_terms=response_terms)
    return _entries(discovery, "response_revision", spec, training, budget, tolerance)


def _entries(discovery, search, spec, training, budget, tolerance):
    from .structural_research_program import attach_structural_research_program

    entries = []
    for candidate, fit in zip(discovery.candidates, discovery.fits):
        candidate = replace(candidate, candidate_id=f"{candidate.candidate_id}:{search}")
        fit = {**fit, "candidate_id": candidate.candidate_id}
        interval = fit["radius_interval"] if fit["training_loss"] <= tolerance else None
        hypotheses = [dict(hypothesis) for hypothesis in discovery.hypotheses
                      if hypothesis["kind"] != "locality"]
        if fit["radius"] is not None:
            hypotheses.append({"kind": "locality", "status": "proposed",
                               "radius": fit["radius"], "counterexamples": []})
        structure = StructuralDiscovery([candidate], [dict(fit)], hypotheses, discovery.schema,
                                        discovery.compute_cost, interval,
                                        discovery.search_complete).to_dict()
        training_loss = fit["training_loss"]
        if spec.risk_quantile != .5:
            if not budget.spend(len(training)):
                break
            training_loss = loss(candidate.program, training, 0., spec.risk_quantile)
        entry = {"candidate": candidate, "kind": f"structural:{fit['family']}",
                        "family_id": _FAMILIES[fit["family"]], "training_loss": training_loss,
                        "structure": structure, "structure_search": search}
        entries.append(attach_structural_research_program(entry, fit_rows=len(training)))
    return entries


def structural_probe(entry, spec, serial, length, *, paired_observations=()):
    """Query the live boundary of this particular executable explanation."""
    payload = entry.get("structure")
    if not payload or payload.get("radius_interval") is None:
        return None
    discovery = StructuralDiscovery.from_dict(payload)
    return boundary_probe(discovery, spec, index=serial, length=length,
                          paired_observations=paired_observations)


def structural_experiments(entry, spec, serial, length=4, *, paired_observations=()):
    """Propose paired measured interventions on the current structural laws."""
    payload = entry.get("structure")
    if not payload:
        return []
    discovery = StructuralDiscovery.from_dict(payload)
    plans = distinguishing_probes(discovery, spec, seed=71 + 997 * serial, length=length,
                                  paired_observations=paired_observations)
    return [{"kind": plan["kind"], **{
        key: Probe(f"unified_structure_{serial}_{plan['kind']}_{key}", plan[key].sequence)
        for key in ("baseline", "probe")}} for plan in plans]


def assess_structural_experiments(entry, pairs, budget):
    """Support or refute laws using acquired outcomes and measured transitions."""
    payload = entry.get("structure")
    if not payload:
        return []
    return confirm_structure(StructuralDiscovery.from_dict(payload), pairs, budget=budget)
