"""Transfer learned structural experiment families to current measurements."""

import json

from .evaluation import loss
from .models import Candidate
from .structural_discovery import StructuralDiscovery, discover_structure
from .structure_lab import ObjectSchema
from .unified_structure import _FAMILIES


def transfer_structure(spec, training, budget, source_prior, *, tolerance=1e-8,
                       transitions=()):
    """Reuse a learned experiment family while reopening its measured boundary.

    A source's interaction interval describes its old apparatus. Only the
    learned family is transferred; all coefficients and graph cells are fitted
    from currently paid observations. An inconsistent family yields no answer,
    allowing ordinary research to reopen the wider hypothesis space.
    """
    source = source_prior.get("structure")
    payload = spec.context.get("object_schema")
    if not source or not payload or not training:
        return []
    schema = ObjectSchema.from_dict(json.loads(payload) if isinstance(payload, str) else payload)
    if not set(schema.feature_names).issubset(spec.feature_names):
        return []
    family = source["fits"][0]["family"]
    source_id = source_prior["source_concept_id"]
    discovery = discover_structure(training, schema, budget=budget, transitions=transitions,
                                   families=(family,))
    entries = []
    for fitted, fit in zip(discovery.candidates, discovery.fits):
        if fit["training_loss"] > tolerance:
            continue
        candidate = Candidate(f"transfer_structure:{source_id}:{fitted.candidate_id}",
            fitted.program, "structural_method_transfer",
            f"Refit the learned {family.replace('_', ' ')} experiment family on current observations",
            parents=(source_id,), memory_id=source_id)
        fit = {**fit, "candidate_id": candidate.candidate_id}
        hypotheses = [dict(hypothesis) for hypothesis in discovery.hypotheses
                      if hypothesis["kind"] != "locality"]
        if fit["radius"] is not None:
            hypotheses.append({"kind": "locality", "status": "proposed",
                               "radius": fit["radius"], "counterexamples": []})
        structure = StructuralDiscovery([candidate], [fit], hypotheses, schema,
            discovery.compute_cost, fit["radius_interval"], discovery.search_complete).to_dict()
        training_loss = fit["training_loss"]
        if spec.risk_quantile != .5:
            if not budget.spend(len(training)):
                break
            training_loss = loss(candidate.program, training, 0., spec.risk_quantile)
        entries.append({"candidate": candidate, "kind": f"structural_method_transfer:{family}",
            "family_id": _FAMILIES[family], "training_loss": training_loss,
            "structure": structure, "structure_search": "transferred_family",
            "method_source_ids": [source_id]})
    return entries
