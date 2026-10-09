"""Use changed research knowledge to propose an experiment on an old frontier."""

from copy import deepcopy
from dataclasses import asdict, replace

from .competing_object_experiments import plan_competing_object_probe, shorten_discriminating_probe
from .evaluation import loss
from .equivalent_object_programs import canonical_object_program
from .instrument_models import rebind_instrument_model
from .language import MissingHistory, Program
from .meta_prior_research import _sources
from .models import Probe, SearchBudget
from .object_continuation import decode_entry, encode_entry, restore_object_state
from .relation_uncertainty import RelationIntervalResearch
from .unified_continuation import paid_view


def _merge_experiment_entries(current, retained):
    """Keep every paid-compatible explanation in the prospective comparison."""
    merged, programs, identifiers = [], set(), set()
    for index, entry in enumerate([*current, *retained]):
        candidate = entry["candidate"]
        program = canonical_object_program(candidate.program.expanded())
        if program in programs:
            continue
        programs.add(program)
        if candidate.candidate_id in identifiers:
            candidate = replace(candidate, candidate_id=f"{candidate.candidate_id}:revisit:{index}")
            entry = {**entry, "candidate": candidate}
        identifiers.add(candidate.candidate_id)
        merged.append(entry)
    return merged


def offer_object_experiment(agent, laboratory, previous, changes, budget):
    """Plan on paid evidence; only the later visit can purchase and support it."""
    if (previous["status"] != "unresolved" or not changes or not agent.config.active_probes
            or laboratory.spec.risk_quantile != .5):
        return None
    offered = tuple(laboratory.measurement_views[-1])
    state = restore_object_state(agent, previous, offered)
    agent._learn_instruments = agent._instrument_knowledge(laboratory)
    agent._revise_inputs = agent._revisable_inputs(laboratory)
    views = [(), *map(tuple, laboratory.measurement_views)]
    frontiers = {}

    def add(payload, channels):
        if payload is None:
            return
        entry = decode_entry(payload)
        group = frontiers.setdefault(tuple(channels), [])
        if not any(old["candidate"].program == entry["candidate"].program for old in group):
            group.append(entry)

    add(state.get("selected"), previous["selected_channels"])
    for payload in state.get("finalists", []):
        add(payload, payload.get("continuation_channels", previous["selected_channels"]))
    for saved in state.get("candidate_frontiers", []):
        view = next((view for view in views
            if list(agent._spec(laboratory, view).feature_names) == saved["features"]), None)
        if view is not None:
            for payload in saved["entries"]:
                add(payload, view)

    available = [item for item in changes if item["available"]]
    if not available:
        return None
    relation_ids = {item["knowledge_id"] for item in available if item["kind"] == "relation"}
    instrument_ids = {item["knowledge_id"] for item in available if item["kind"] == "instrument"}
    method_ids = {item["knowledge_id"] for item in available if item["kind"] == "method"}
    command_spec = agent._spec(laboratory, ())
    rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
    for channels, entries in frontiers.items():
        if not (instrument_ids or method_ids) and not any(
                relation_ids & set(_sources(entry["candidate"].program)) for entry in entries):
            continue
        rows = [row for batch in paid_view(laboratory, previous, channels) for row in batch]
        if not rows:
            continue
        spec = agent._spec(laboratory, channels)
        supported = []
        for entry in entries[:agent.config.max_candidates]:
            if not budget.spend(len(rows)):
                break
            value = loss(entry["candidate"].program, rows,
                         entry.get("fallback", previous["fallback"]), spec.risk_quantile)
            if value <= agent.config.minimum_improvement:
                supported.append({**entry, "training_loss": value, "validation_loss": value,
                                  "selection_basis": "all_acquired_evidence"})
        if not supported:
            continue
        questions = []
        # Saved relation cells can supply executable competitors even when
        # only their previously selected representative was serialized.
        local = SearchBudget(budget.remaining // 2)
        first = next((entry for entry in supported
            if relation_ids & set(_sources(entry["candidate"].program))), supported[0])
        program = first["candidate"].program
        relation, used_relations = None, set()
        if relation_ids & set(_sources(program)):
            # An unchanged concept can now carry better range evidence. Its
            # current prior binds to the actual call through stored/_match;
            # the old entry's own metadata would otherwise hide that update.
            updated = RelationIntervalResearch(program, agent.library.meta_priors, local)
            consumed = relation_ids & {item.get("source_concept_id") for item in updated.parameters}
            if consumed:
                relation, used_relations = updated, consumed
        if relation is None:
            relation = RelationIntervalResearch(program, agent.library.meta_priors,
                local, structure=first.get("structure"), research_program=first.get("research_program"))
        if relation.parameters:
            alternatives = relation.alternatives(rows, local, tolerance=agent.config.minimum_improvement)
            # New evidence about a source does not by itself refute the old
            # task's incumbent. Put both into the actual distinguishing test.
            supported = _merge_experiment_entries(alternatives, supported) if alternatives else []
            questions = relation.questions(alternatives) if alternatives else []
        budget.spend(local.spent)
        if len(supported) < 2:
            continue
        binding = agent._instrument_binding(spec, command_spec)
        required = {name for entry in supported
            for name, _ in entry["candidate"].program.observed_variables()
            if name != "__target__"} - set(command_spec.feature_names)
        used_instruments = []
        for model in agent.library.instrument_models:
            if model["model_id"] not in instrument_ids or model.get("status") == "counterexample":
                continue
            mapped = rebind_instrument_model(model, spec)
            if any(name in mapped and mapped[name] == binding.get(name) for name in required):
                used_instruments.append(model["model_id"])
        if not (used_instruments or used_relations or method_ids):
            continue
        paired_report = {"batches": [batch for batch in previous["batches"]
            if not channels or channels in map(tuple, laboratory.projectable_views(batch["channels"]))]}
        commands = [row for batch in paid_view(laboratory, paired_report, ()) for row in batch]
        compatible = required <= set(binding)
        for name in sorted(required):
            if not compatible or not budget.spend(len(rows)):
                compatible = False
                break
            errors = []
            for command, row in zip(commands, rows):
                try:
                    errors.append((binding[name].predict(command) - row.features[name]) ** 2)
                except MissingHistory:
                    continue
            if not errors or sum(errors) / len(errors) > agent.config.minimum_improvement:
                compatible = False
                break
        if not compatible:
            continue
        plan = plan_competing_object_probe(command_spec, supported, budget, binding=binding,
            serial=len(previous["batches"]), length=5, tolerance=agent.config.minimum_improvement,
            methods=agent.library.program_experiment_methods,
            task_id=f"{laboratory.spec.task_id}:{agent.config.seed}", questions=questions)
        if plan["forecast_status"] != "predicted_distinction" or plan["probe"] is None:
            continue
        plan = shorten_discriminating_probe(plan, budget, tolerance=agent.config.minimum_improvement)
        proposal = plan.get("proposal") or {}
        used_methods = method_ids & {proposal.get("method_id"), proposal.get("parent_method_id")}
        if not (used_instruments or used_relations or used_methods):
            continue
        sequence = list(plan["probe"].sequence)
        if any(sequence == batch["probe"]["sequence"] for batch in previous["batches"]):
            continue
        chosen = supported[0]
        separation = plan["predicted_separation"]
        length = min(24, laboratory.spec.max_probe_length)
        rate = sum(rates[name] for name in channels)
        needed = {channels, tuple(previous["selected_channels"])}
        covers = [tuple(view) for view in laboratory.measurement_views
            if needed <= {(), *map(tuple, laboratory.projectable_views(view))}]
        if not covers:
            continue
        confirmation_rate = min(sum(rates[name] for name in view) for view in covers)
        candidate = {"program": chosen["candidate"].program.to_dict(), "channels": list(channels),
            "fallback": previous["fallback"], "kind": "successor_object_experiment",
            "family_id": chosen.get("family_id", "T03"), "source_concept_ids": sorted(used_relations),
            "knowledge_changes": deepcopy(changes),
            "experiment": {**plan, "probe": asdict(plan["probe"])},
            "experiment_binding": {name: program.to_dict() for name, program in binding.items()},
            "experiment_entries": [encode_entry(entry) for entry in supported]}
        for name in ("structure", "research_program"):
            if chosen.get(name):
                candidate[name] = deepcopy(chosen[name])
        return {"candidate": candidate, "basis": "new_knowledge_distinguishing_experiment",
            "old_evidence_gain": 0., "old_evidence_rows": len(rows),
            "predicted_separation": separation,
            "relative_gain": separation / (separation + agent.config.minimum_improvement),
            "source_instrument_ids": used_instruments,
            "source_method_ids": sorted(used_methods),
            "minimum_resources": {"observation_cost": plan["probe"].cost + length,
                "measurement_cost": plan["probe"].cost * rate + length * confirmation_rate,
                "compute_cost": (6 + len(supported)) * len(rows) + 32 * length + plan["probe"].cost *
                    (len(plan["hypotheses"]) + len(plan["predicted_fields"]) + 1)}}
    return None


def execute_object_experiment(agent, laboratory, candidate, *, acquire, paired_for, budget, actions):
    """Buy the offered experiment and assess its frozen hypotheses before refits."""
    from .competing_object_experiments import assess_competing_object_probe
    plan = deepcopy(candidate["experiment"])
    probe = plan["probe"] = Probe(plan["probe"]["probe_id"], tuple(plan["probe"]["sequence"]))
    channels = tuple(candidate["channels"])
    assessment_cost = probe.cost * (len(plan["hypotheses"]) + len(plan["predicted_fields"]))
    if budget.remaining < assessment_cost:
        return None
    rows = acquire(channels, "successor_knowledge_experiment", probe=probe)
    if rows is None:
        return None
    evidence = assess_competing_object_probe(plan, paired_for(channels)[-1], budget,
                                            tolerance=agent.config.minimum_improvement)
    actions.append({"action": "test_successor_object_experiment", "probe_id": probe.probe_id,
        "knowledge_changes": deepcopy(candidate["knowledge_changes"]),
        "plan": {key: value for key, value in plan.items() if key != "probe"}, "evidence": evidence})
    feedback = agent.library.program_experiment_methods.observe(plan, evidence,
        task_id=f"{laboratory.spec.task_id}:{agent.config.seed}",
        command_spec=agent._spec(laboratory, ()), binding={name: Program.from_dict(value)
            for name, value in candidate["experiment_binding"].items()},
        tolerance=agent.config.minimum_improvement)
    if feedback is not None:
        actions.append(feedback)
    refuted_programs = [Program.from_dict(hypothesis["program"])
        for hypothesis, assessment in zip(plan["hypotheses"], evidence["hypotheses"])
        if assessment["status"] == "counterexample"]
    refuted = {canonical_object_program(program.expanded()) for program in refuted_programs}
    entries = [decode_entry(item) for item in candidate["experiment_entries"]]
    return {"rows": rows, "channels": channels, "evidence": evidence,
            "refuted_programs": refuted_programs,
            "entries": [entry for entry in entries
                        if canonical_object_program(entry["candidate"].program.expanded()) not in refuted]}
