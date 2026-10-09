"""Offer successor object knowledge on the same already purchased response views."""

from .evaluation import initial_prediction, loss
from .language import Program
from .meta_prior_research import _sources
from .models import Candidate, SearchBudget
from .unified_continuation import paid_view


def offer_object_successor(agent, laboratory, previous, concept_ids, budget):
    """Screen a proposal; the resumed structural frontier must still test it."""
    if previous["status"] != "unresolved":
        return None
    offered = tuple(laboratory.measurement_views[-1])
    if previous.get("learned_object_schema"):
        agent._object_schemas[offered] = previous["learned_object_schema"]
    if previous.get("object_research"):
        agent._object_models[offered] = previous["object_research"]
    agent._learn_instruments = agent._instrument_knowledge(laboratory)
    agent._revise_inputs = agent._revisable_inputs(laboratory)
    relevant = {concept.concept_id for concept in agent.library.concepts
                if concept.concept_id in concept_ids and concept.status != "revoked"
                and concept.risk_quantile == laboratory.spec.risk_quantile}
    if not relevant:
        return None
    views = list(dict.fromkeys([(), *(tuple(view) for batch in previous["batches"]
        for view in laboratory.projectable_views(batch["channels"]))]))
    old_channels = tuple(previous["selected_channels"])
    incumbent = Program.from_dict(previous["program"])
    rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
    best = None
    for index, channels in enumerate(views):
        # A measured view may only be compared where the very same batch also
        # provides the old predictor's view. Fine can supply coarse, not vice versa.
        batches = [batch for batch in previous["batches"] if {channels, old_channels}
            <= {(), *map(tuple, laboratory.projectable_views(batch["channels"]))}]
        if not batches:
            continue
        paid = {"batches": batches}
        rows = [row for batch in paid_view(laboratory, paid, channels) for row in batch]
        old_rows = [row for batch in paid_view(laboratory, paid, old_channels) for row in batch]
        if not rows or not budget.spend(len(old_rows)):
            continue
        spec = agent._spec(laboratory, channels)
        baseline = loss(incumbent, old_rows, previous["fallback"], spec.risk_quantile)
        fallback = initial_prediction(rows, spec.risk_quantile)
        calls = [Candidate(f"successor_object_call:{n}", call, "reuse",
                           parents=tuple(_sources(call)))
                 for n, call in enumerate(agent._recall(spec)) if relevant & set(_sources(call))]
        allowance = budget.remaining // max(1, len(views) - index)
        fitting = SearchBudget(max(0, allowance - (len(calls) + agent.config.max_candidates) * len(rows)))
        fitted = agent.library.propose(spec, rows, fitting,
            structural_transfer=agent.config.structural_transfer,
            preserve_parameters=True, source_concept_ids=relevant)
        budget.spend(fitting.spent)
        seen = set()
        for candidate in [*calls, *fitted[:agent.config.max_candidates]]:
            if candidate.program in seen:
                continue
            seen.add(candidate.program)
            if not budget.spend(len(rows)):
                break
            measured = loss(candidate.program, rows, fallback, spec.risk_quantile)
            gain = baseline - measured
            if gain <= agent.config.minimum_improvement:
                continue
            sources = sorted(relevant & {candidate.memory_id, *candidate.parents,
                                         *_sources(candidate.program)})
            if not sources:
                continue
            needed = {channels, old_channels}
            cover = min((tuple(view) for view in laboratory.measurement_views
                if needed <= {(), *map(tuple, laboratory.projectable_views(view))}),
                key=lambda view: sum(rates[name] for name in view))
            length = min(24, laboratory.spec.max_probe_length)
            prior = next((item for item in reversed(agent.library.meta_priors)
                          if item["source_concept_id"] in sources), {})
            proposal = {"candidate": {"program": candidate.program.to_dict(),
                "channels": list(channels), "fallback": fallback, "source_concept_ids": sources,
                "kind": "successor_object_knowledge", "family_id": prior.get("family_id", "T03"),
                "old_loss": baseline, "selection_loss": measured},
                "basis": "same_paid_object_responses", "old_evidence_gain": gain,
                "old_evidence_rows": len(rows),
                "relative_gain": gain / max(baseline, agent.config.minimum_improvement),
                "minimum_resources": {"observation_cost": 2 * length,
                    "measurement_cost": 2 * length * sum(rates[name] for name in cover),
                    "compute_cost": 6 * len(rows) + 32 * length}}
            if best is None or proposal["relative_gain"] > best["relative_gain"]:
                best = proposal
    return best
