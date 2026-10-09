"""Restore object research into the ordinary structural experiment loop."""

from copy import deepcopy

from .evaluation import initial_prediction, loss
from .language import Program
from .models import Candidate, Observation


def encode_entry(entry):
    if entry is None:
        return None
    candidate = entry["candidate"]
    payload = {**deepcopy({key: value for key, value in entry.items() if key not in {
        "candidate", "research_action_origin", "object_action_origin", "risk_action_origin"}}),
        "candidate": {"candidate_id": candidate.candidate_id, "program": candidate.program.to_dict(),
            "operation": candidate.operation, "rationale": candidate.rationale,
            "parents": list(candidate.parents), "memory_id": candidate.memory_id}}
    if "terms" in payload:
        payload["terms"] = [term.to_dict() for term in entry["terms"]]
    return payload


def decode_entry(payload):
    if payload is None:
        return None
    entry = deepcopy(payload)
    candidate = entry["candidate"]
    entry["candidate"] = Candidate(candidate["candidate_id"], Program.from_dict(candidate["program"]),
        candidate["operation"], candidate.get("rationale", ""), tuple(candidate.get("parents", ())),
        candidate.get("memory_id"))
    if "terms" in entry:
        entry["terms"] = [Program.from_dict(term) for term in entry["terms"]]
    return entry


def observations(payload):
    return [Observation(dict(row["features"]), row["target"],
                        tuple(dict(past) for past in row["history"])) for row in payload]


def restore_object_state(agent, previous, offered):
    """Restore learned schemas; a missing anonymous schema stays missing."""
    state = previous.get("object_continuation_state", {})
    agent._object_schemas = {tuple(item["channels"]): deepcopy(item["schema"])
                             for item in state.get("schemas", [])}
    agent._object_models = {tuple(item["channels"]): deepcopy(item["model"])
                            for item in state.get("models", [])}
    agent._object_competitions = {tuple(item["features"]): deepcopy(item["competition"])
                                  for item in state.get("competitions", [])}
    agent._object_candidate_frontiers = {tuple(item["features"]): deepcopy(item["entries"])
                                        for item in state.get("candidate_frontiers", [])}
    if not state:
        if previous.get("learned_object_schema"):
            agent._object_schemas[tuple(offered)] = deepcopy(previous["learned_object_schema"])
        if previous.get("object_research"):
            agent._object_models[tuple(offered)] = deepcopy(previous["object_research"])
    return deepcopy(state)


def resume_entries(agent, laboratory, previous, candidate, *, paired_for,
                   command_batches, budget, actions):
    """Compare stored and incoming programs on the same already paid view."""
    state = previous.get("object_continuation_state", {})
    channels = tuple(previous["selected_channels"])
    incumbent = decode_entry(state.get("selected"))
    if incumbent is None:
        incumbent = {"candidate": Candidate("continued_object:incumbent",
            Program.from_dict(previous["program"]), "retained_object_incumbent"),
            "kind": previous["selected_kind"], "family_id": "T03",
            "fallback": previous["fallback"], "structure": deepcopy(previous.get("selected_structure")),
            "research_program": deepcopy(previous.get("research_program")),
            "object_competition": deepcopy(previous.get("object_competition"))}
    choices = [(incumbent, channels)]
    for payload in state.get("finalists", []):
        proposal = decode_entry(payload)
        if proposal["candidate"].program != incumbent["candidate"].program:
            choices.append((proposal, tuple(payload.get("continuation_channels", channels))))
    feature_views = {tuple(agent._spec(laboratory, view).feature_names): tuple(view)
                     for view in laboratory.measurement_views}
    for frontier in state.get("candidate_frontiers", []):
        view = feature_views.get(tuple(frontier["features"]))
        if view is None:
            continue
        for payload in frontier["entries"]:
            proposal = decode_entry(payload)
            if not any(proposal["candidate"].program == old["candidate"].program and view == old_view
                       for old, old_view in choices):
                choices.append((proposal, view))
    if candidate is not None:
        sources = list(candidate.get("source_concept_ids", []))
        proposed_channels = tuple(candidate.get("channels", ()))
        proposed = {"candidate": Candidate("continued_object:successor",
            Program.from_dict(candidate["program"]), "successor_object_knowledge", parents=tuple(sources)),
            "kind": candidate.get("kind", "successor_object_knowledge"),
            "family_id": candidate.get("family_id", "T03"),
            "method_source_ids": sources, "fallback": candidate.get("fallback", 0.)}
        # A source schema is not the current apparatus's measured schema. Only
        # already-bound structural metadata supplied by the caller is usable.
        from .memory import _object_schema
        schema = _object_schema(agent._spec(laboratory, proposed_channels))
        structure = candidate.get("structure")
        if structure and structure.get("schema") == schema:
            proposed["structure"] = deepcopy(structure)
        for name in ("research_program", "object_competition"):
            if candidate.get(name) is not None:
                proposed[name] = deepcopy(candidate[name])
        choices.append((proposed, proposed_channels))

    views = [tuple(view) for view in laboratory.measurement_views]
    rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
    needed = {view for _, view in choices}
    cover = next((view for view in sorted(views, key=lambda view: sum(rates[name] for name in view))
                  if needed <= {(), *map(tuple, laboratory.projectable_views(view))}), None)
    common = (paired_for(cover) if cover else [list(zip(batch, batch)) for batch in command_batches()]) if cover is not None else []
    compared = []
    paired_gains = {}
    for entry, view in choices:
        comparison_cover, pairs = cover, common
        if cover is None:
            # Anonymous banks have no declared cross-bank projection. An
            # unrelated finalist must not prevent a legal comparison with the
            # incumbent on responses purchased from this candidate's bank.
            comparison_cover = next((source for source in sorted(views,
                key=lambda source: sum(rates[name] for name in source))
                if {view, channels} <= {(), *map(tuple, laboratory.projectable_views(source))}), None)
            if comparison_cover is None:
                continue
            pairs = paired_for(comparison_cover) if comparison_cover else [
                list(zip(batch, batch)) for batch in command_batches()]
        rows = [command if not view else measured for batch in pairs for command, measured in batch]
        if view and view != comparison_cover:
            rows = laboratory.project_rows(rows, comparison_cover, view)
        if not rows or not budget.spend(len(rows)):
            continue
        fallback = entry.get("fallback", initial_prediction(rows, laboratory.spec.risk_quantile))
        value = loss(entry["candidate"].program, rows, fallback, laboratory.spec.risk_quantile)
        entry.update(fallback=fallback, training_loss=value, validation_loss=value,
                     selection_basis="all_acquired_evidence")
        if cover is None:
            old_rows = [command if not channels else measured for batch in pairs for command, measured in batch]
            if channels and channels != comparison_cover:
                old_rows = laboratory.project_rows(old_rows, comparison_cover, channels)
            if not budget.spend(len(old_rows)):
                continue
            baseline = loss(incumbent["candidate"].program, old_rows,
                incumbent.get("fallback", previous["fallback"]), laboratory.spec.risk_quantile)
            paired_gains[id(entry)] = baseline - value
            entry["continuation_comparison"] = {"rows": len(rows), "incumbent_loss": baseline,
                                                "candidate_loss": value, "channels": list(comparison_cover)}
        compared.append((entry, view, rows))
    selected, selected_channels, rows = (incumbent, channels,
        [row for batch in (command_batches() if not channels else
                          [[row for _, row in pairs] for pairs in paired_for(channels)]) for row in batch])
    if compared:
        if cover is not None:
            best = agent._best([entry for entry, _, _ in compared])
        else:
            best = max((entry for entry, _, _ in compared), key=lambda entry: paired_gains[id(entry)])
            if paired_gains[id(best)] <= agent.config.minimum_improvement:
                best = incumbent
        chosen = next((item for item in compared if item[0] is best), None)
        if chosen is not None:
            selected, selected_channels, rows = chosen
    selected.setdefault("validation_loss", previous.get("selection_loss"))
    selected.setdefault("training_loss", previous.get("selection_loss"))
    actions.append({"action": "resume_object_research", "previous_status": previous["status"],
        "source_concept_ids": (candidate or {}).get("source_concept_ids", []),
        "source_object_model_ids": (candidate or {}).get("source_object_model_ids", []),
        "paid_batches": len(previous["batches"]), "paid_rows": len(rows),
        "selected_channels": list(selected_channels), "selected_kind": selected["kind"],
        "alternatives": [{"program": item["candidate"].program.to_dict(), "channels": list(view),
                          "loss": item["validation_loss"]} for item, view, _ in compared],
        "selection_basis": "same_purchased_responses"})
    return {"selected": selected, "channels": selected_channels, "training": list(rows),
            "validation": list(rows), "finalists": [entry for entry, view, _ in compared
                if view == selected_channels] or [selected], "incumbent": incumbent,
            "incumbent_channels": channels}
