"""Public object representations and boundary experiments for unified research."""

import json

from .evaluation import loss
from .models import Probe
from .structural_discovery import (
    StructuralDiscovery, boundary_probe, confirm_structure, discover_structure,
    distinguishing_probes,
)
from .structure_lab import ObjectSchema


_FAMILIES = {
    "conserved_total": "T15", "conserved_combination": "T15",
    "focal_coordinates": "T04", "object_specific": "T04",
    "local_exchange": "T08", "local_and_global": "T08",
    "object_specific_exchange": "T08", "event_change": "T11",
    "window_exchange": "T12",
}


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
    discovery, search = None, "full"
    previous = (parent or {}).get("structure")
    if previous and ObjectSchema.from_dict(previous["schema"]) == schema:
        family = previous["fits"][0]["family"]
        interval = previous.get("radius_interval")
        if interval and interval[1] > interval[0]:
            low, high = interval
            radii = [low + (high - low) * fraction for fraction in (.25, .5, .75)]
            narrowed = discover_structure(training, schema, budget=budget, transitions=transitions,
                                          radius_candidates=radii, families=(family,))
            if narrowed.fits and min(fit["training_loss"] for fit in narrowed.fits) <= tolerance:
                discovery, search = narrowed, "boundary_revision"
        if discovery is None:
            # The family can remain useful even when its current radius has no
            # consistent interval. Resolve it against all acquired evidence
            # before spending again on unrelated structural explanations.
            focused = discover_structure(training, schema, budget=budget, transitions=transitions,
                                          families=(family,))
            if focused.fits and min(fit["training_loss"] for fit in focused.fits) <= tolerance:
                discovery, search = focused, "family_revision"
    if discovery is None:
        discovery = discover_structure(training, schema, budget=budget, transitions=transitions)
    entries = []
    for candidate, fit in zip(discovery.candidates, discovery.fits):
        interval = fit["radius_interval"] if fit["training_loss"] <= tolerance else None
        hypotheses = [dict(hypothesis) for hypothesis in discovery.hypotheses
                      if hypothesis["kind"] != "locality"]
        if fit["radius"] is not None:
            hypotheses.append({"kind": "locality", "status": "proposed",
                               "radius": fit["radius"], "counterexamples": []})
        structure = StructuralDiscovery([candidate], [dict(fit)], hypotheses, schema,
                                        discovery.compute_cost, interval,
                                        discovery.search_complete).to_dict()
        training_loss = fit["training_loss"]
        if spec.risk_quantile != .5:
            if not budget.spend(len(training)):
                break
            training_loss = loss(candidate.program, training, 0., spec.risk_quantile)
        entries.append({"candidate": candidate, "kind": f"structural:{fit['family']}",
                        "family_id": _FAMILIES[fit["family"]], "training_loss": training_loss,
                        "structure": structure, "structure_search": search})
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
