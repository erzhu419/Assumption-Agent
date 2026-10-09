"""Fit new responses from learned relation computations before structure search."""

from copy import deepcopy
from dataclasses import replace

from .language import Program
from .relation_conditions import TRUE
from .relation_kernel_priors import (
    interval_union, kernel_key, parameter_relations, rebind_relation_inputs)
from .relation_revisions import _relations
from .relation_uncertainty import _bind, _mark_relations
from .research_programs import _Fitter, _entry, _rank, _temporal, propose_research_programs


def collect_relation_kernels(spec, priors, budget, *, input_bindings=None):
    """Extract supported subcomputations under an explicit input binding.

    Without ``input_bindings`` this keeps the historical same-coordinate path.
    With one, the caller has already supplied paid instrument calibration from
    source measured leaves to executable current-input programs.  The source
    radius intervals remain untouched and no response target is transferred.
    """
    records = {}
    for prior in priors:
        if prior.get("status") != "supported":
            continue
        saved = next((step for step in reversed((prior.get("research_program") or {}).get("selection_steps", []))
                      if step["operation"] == "resolve_relation_intervals"), None)
        if saved is None:
            continue
        template = Program.from_dict(saved["parameter_template"])
        if not budget.spend(template.size):
            break
        values = {item["symbol"]: Program("const", value=item["radius"]) for item in saved["parameters"]}
        for item in saved["parameters"]:
            domains = item["radius_intervals"]
            for relation in parameter_relations(template, item["symbol"]):
                for low, high in domains:
                    radius = item["radius"] if low <= item["radius"] <= high else (low + high) / 2
                    kernel = _bind(relation, {**values, item["symbol"]: Program("const", value=radius)})
                    rebound = rebind_relation_inputs(kernel, input_bindings,
                                                      spec.feature_names)
                    if rebound is None:
                        continue
                    source = {"source_concept_id": prior["source_concept_id"],
                              "radius_intervals": deepcopy(domains),
                              "prior_radius_intervals": deepcopy(item.get("prior_radius_intervals", domains))}
                    if input_bindings:
                        source["input_binding"] = {
                            name: (value.to_dict() if isinstance(value, Program) else value)
                            for name, value in input_bindings.items()
                            if name in kernel.required_features()}
                    records.setdefault(rebound, [])
                    if source not in records[rebound]:
                        records[rebound].append(source)
    return records


def _attach_parameters(entry, records):
    """Save the new fitted consumer, rather than the source task's coefficients."""
    program = entry["candidate"].program
    relations = set(_relations(program))
    actual = []
    for kernel in records:
        if kernel not in relations:
            continue
        sources = []
        for other, records_for_kernel in records.items():
            if kernel_key(other) == kernel_key(kernel):
                for source in records_for_kernel:
                    if source not in sources:
                        sources.append(source)
        actual.append((kernel, sources))
    if not actual:
        return None
    template, parameters, sources_used = program, [], []
    plan = entry["research_program"]
    term_templates = [Program.from_dict(plan["graph"]["nodes"][key]["answer"])
                      for key in plan["term_nodes"]]
    for kernel, sources in actual:
        symbol = f"relation_parameter_{len(parameters)}"
        template = _mark_relations(template, {kernel}, symbol)
        term_templates = [_mark_relations(term, {kernel}, symbol) for term in term_templates]
        ids = list(dict.fromkeys(source["source_concept_id"] for source in sources
            if any(low <= kernel.children[0].value <= high for low, high in source["radius_intervals"])))
        domains = interval_union(interval for source in sources for interval in source["radius_intervals"])
        previous = interval_union(interval for source in sources for interval in source["prior_radius_intervals"])
        parameters.append({"symbol": symbol, "radius": kernel.children[0].value,
            "radius_intervals": domains, "prior_radius_intervals": previous,
            "source_concept_id": ids[0] if len(ids) == 1 else None,
            "source_definition_id": None, "relation": kernel.name or "symmetric",
            "predicate": (kernel.children[1] if kernel.op == "relation_sum" else TRUE).to_dict(),
            "kernel_priors": deepcopy(sources), "kernel_prior_status": "supported",
            "kernel_source_ids": ids, "kernel_origin_source_concept_id": None})
        sources_used.extend(ids)
    plan["selection_steps"].append({"operation": "resolve_relation_intervals",
        "question": "Test learned relation uncertainty inside the newly fitted response",
        "program": program.to_dict(), "loss": entry["training_loss"],
        "parameter_template": template.to_dict(), "parameters": parameters,
        "parameter_terms": [term.to_dict() for term in term_templates], "evidence_contexts": [],
        "compact_replacement": False})
    sources_used = list(dict.fromkeys(sources_used))
    plan["source_ids"] = list(dict.fromkeys([*plan["source_ids"], *sources_used]))
    entry["method_source_ids"] = sources_used
    entry["candidate"] = replace(entry["candidate"], parents=tuple(plan["source_ids"]))
    return entry


def compose_relation_kernels(spec, training, budget, records, *, max_candidates=16, tolerance=1e-7):
    """Compare current, temporal and joint consumers under the existing grammar."""
    rows = list(training)
    if len(rows) < 6 or not records:
        return []
    names = set.intersection(*(set(row.features) for row in rows))
    kernels = [kernel for kernel in records if kernel.required_features() <= names][:max_candidates]
    fitter, start = _Fitter(rows, budget, spec.risk_quantile), budget.spent
    found = {}
    # Give each remembered relation its own initial comparison before expanding
    # one source's temporal alternatives. Never copy the old response wrapper.
    for transform in (None, "temporal"):
        for kernel in kernels:
            terms = [kernel] if transform is None else list(_temporal(kernel))
            for term in terms:
                fit = fitter.fit([term])
                if fit is not None:
                    found.setdefault(fit["program"], fit)
        if found and min(fit["loss"] for fit in found.values()) <= tolerance:
            break
    claims = []
    for fit in sorted(found.values(), key=_rank)[:max_candidates]:
        step = {"operation": "compose_relation_kernel",
                "question": "Refit a learned relation under the current response and history",
                "program": fit["program"], "loss": fit["loss"]}
        claims.append(_entry(fit, [step], len(rows), budget.spent - start, len(claims)))
    if not claims or min(item["training_loss"] for item in claims) > tolerance:
        atoms = list(dict.fromkeys([*kernels,
            *(term for fit in sorted(found.values(), key=_rank) for term in fit["terms"])]))
        claims.extend(propose_research_programs(spec, rows, budget, atoms, max_candidates=max_candidates))
    ranked = sorted(claims, key=lambda entry: (
        0. if entry["training_loss"] < 1e-12 else entry["training_loss"], entry["candidate"].program.size))
    result = []
    for entry in ranked[:max_candidates]:
        attached = _attach_parameters(entry, records)
        if attached is not None:
            attached["candidate"] = replace(attached["candidate"], candidate_id=f"kernel_composition:{len(result)}")
            result.append(attached)
    return result
