"""Carry callable knowledge through a stream of publicly instrumented tasks."""

from copy import deepcopy
from dataclasses import asdict, replace

from .evaluation import loss
from .language import Program
from .knowledge_changes import knowledge_snapshot, knowledge_changes
from .memory import KnowledgeLibrary
from .models import ResearchConfig, SearchBudget
from .numeric_method_revisits import reprice_numeric_method_offer, refresh_numeric_method_quote
from .response_research_questions import refresh_response_question_quotes, response_question_offer
from .research_portfolio import choose_research_visit
from .research_resources import allocate_complete_research
from .knowledge_returns import matching_returns, return_priority


TASK_ORDER = (0, 1, 2, 3, 4, 5, 6, 7, 3, 0, 7, 4, 2, 6, 1, 5)


def retain_revisit_offer(pending, offer):
    """Keep distinct numeric plans until their full allocations can compete."""
    existing = [item for item in pending if item["task_position"] == offer["task_position"]
                and "question_id" not in item]
    if offer.get("loss_kind") == "squared_error" and all(
            item.get("loss_kind") == "squared_error" for item in existing):
        # Refresh the same executable plan; competing plans may have different
        # continuation caps even when their minimum costs happen to match.
        pending[:] = [item for item in pending if not (
            item in existing and item["candidate"] == offer["candidate"])]
    elif existing:
        if offer["relative_gain"] <= max(item["relative_gain"] for item in existing):
            return False
        pending[:] = [item for item in pending if item not in existing]
    pending.append(offer)
    return True


def task_sequence(seed, order=TASK_ORDER):
    return [{"task_index": index, "seed": seed + 10007 * position,
             "profile": "development"} for position, index in enumerate(order)]


def laboratory_for(item):
    """Select the physical interface; mechanism choice stays inside research."""
    arguments = dict(item)
    environment = arguments.pop("environment", "scalar")
    arguments.pop("research_limits", None)
    if environment == "objects":
        from .unified_object_lab import UnifiedObjectLab
        return UnifiedObjectLab(**arguments)
    if environment == "anonymous_objects":
        from .anonymous_object_lab import AnonymousObjectLab
        return AnonymousObjectLab(**arguments)
    if environment == "dynamic_instruments":
        from .dynamic_instrument_lab import DynamicInstrumentLab
        return DynamicInstrumentLab(**arguments)
    if environment == "stateful_instruments":
        from .stateful_instrument_lab import StatefulInstrumentLab
        return StatefulInstrumentLab(**arguments)
    if environment == "ambiguous_instruments":
        from .ambiguous_instrument_lab import AmbiguousInstrumentLab
        return AmbiguousInstrumentLab(**arguments)
    if environment == "instrument_methods":
        from .instrument_method_lab import InstrumentMethodLab
        return InstrumentMethodLab(**arguments)
    if environment != "scalar":
        raise ValueError(f"unknown laboratory interface: {environment}")
    from .unknown_task_lab import UnknownTaskLab
    return UnknownTaskLab(**arguments)


class UnifiedResearchSession:
    """Keep research memory, with task-boundary continuation and separate scoring."""

    def __init__(self, config=None, *, tasks=None, measurement_budget=768,
                 state=None, library=None, lab_factory=None):
        if state is not None:
            self.state = deepcopy(state)
            self.config = ResearchConfig(**state["config"])
            self.library = KnowledgeLibrary.from_dict(state["knowledge"])
        else:
            self.config = config or ResearchConfig(observation_budget=192,
                compute_budget=120000, minimum_improvement=1e-7, max_rounds=4,
                active_probes=False)
            self.library = library if library is not None else KnowledgeLibrary()
            self.state = {"experiment": "unified_unknown_task_research", "status": "ready",
                "config": asdict(self.config), "measurement_budget": measurement_budget,
                "tasks": list(tasks if tasks is not None else task_sequence(self.config.seed)),
                "results": [], "evaluation": [], "knowledge": self.library.to_dict()}
        # Existing saved sequences also have paid reports that can be revisited.
        for key in ("pending_revisits", "revisits", "scheduler_events"):
            self.state.setdefault(key, [])
        self.state.setdefault("screening_compute_cost", 0)
        self.state.setdefault("method_revisit_attempts", [])
        self.state.setdefault("condition_revisit_quotes", [])
        if lab_factory is None:
            lab_factory = laboratory_for
        self.lab_factory = lab_factory

    def _costs(self):
        costs = {key: sum(row["research"]["costs"][key] for row in self.state["results"])
                 for key in ("observation_cost", "measurement_cost", "compute_cost")}
        costs["compute_cost"] += self.state["screening_compute_cost"]
        return costs

    def _remaining(self):
        """Pool unused resources while reserving every unvisited task's budget."""
        spent = self._costs()
        limits = [self._task_limits(row["task_position"]) for row in self.state["results"]]
        return {key: max(0, sum(item[key.replace("_cost", "_budget")] for item in limits) - spent[key])
                for key in spent}

    def _task_limits(self, position):
        limits = {"observation_budget": self.config.observation_budget,
                  "compute_budget": self.config.compute_budget,
                  "measurement_budget": self.state["measurement_budget"]}
        limits.update(self.state["tasks"][position].get("research_limits", {}))
        return limits

    def _laboratory_arguments(self, position):
        return {key: value for key, value in self.state["tasks"][position].items()
                if key != "research_limits"}

    def _workflow_demands(self, *, exclude_position=None):
        """Describe unfinished consumers from their purchased view alone."""
        if not self.config.use_memory or not self.config.learn_research_policy:
            return []
        from .unified_continuation import paid_view
        from .knowledge_returns import transfer_condition
        demands = []
        for item in self.state["results"]:
            previous = item["research"]
            if (item["task_position"] == exclude_position or previous["status"] != "unresolved"
                    or not previous.get("batches")):
                continue
            position, channels = item["task_position"], tuple(previous["selected_channels"])
            with self.lab_factory(self._laboratory_arguments(position)) as laboratory:
                rows = [row for batch in paid_view(laboratory, previous, channels) for row in batch]
                if not rows:
                    continue
                spec = laboratory.spec_for(channels)
                if previous.get("learned_object_schema"):
                    spec = replace(spec, context={**spec.context,
                        "object_schema": deepcopy(previous["learned_object_schema"])})
                demands.append({"task_position": position, "status": "unresolved",
                    "condition": transfer_condition(spec, rows), "paid_rows": len(rows),
                    "channels": list(channels)})
        return demands

    def _method_screened(self, position, method, previous):
        return any(attempt["task_position"] == position and attempt["method"] == method
            and attempt.get("paid_batches", len(previous["batches"])) == len(previous["batches"])
            and attempt.get("program", previous["program"]) == previous["program"]
            for attempt in self.state["method_revisit_attempts"])

    def _remember_method_screens(self, position, previous, methods, actions):
        quotes = [action["quote"] for action in actions
                  if action["action"] == "cache_numeric_method_quote"]
        # Keep each executable plan once, beside the paid attempt that formed
        # it. Scheduler summaries need no duplicate copy of its full program.
        actions[:] = [action for action in actions
                      if action["action"] != "cache_numeric_method_quote"]
        tried = set()
        for action in actions:
            if action["action"] == "select_research_drafts_on_paid_response":
                tried.update(item["draft_id"] for item in action["alternatives"]
                    if item["compute_cost"] or item["training_loss"] is not None)
            elif action["action"] == "replay_research_draft":
                tried.add(action["draft_id"])
            elif action["action"] == "replay_risk_method":
                tried.add(action["draft_id"])
        attempts = []
        for identifier in methods:
            if identifier not in tried:
                continue
            attempt = {"task_position": position, "draft_id": identifier,
                "method": deepcopy(methods[identifier]),
                "paid_batches": len(previous["batches"]), "program": deepcopy(previous["program"])}
            quote = next((item for item in quotes
                if item["candidate"]["research_draft_id"] == identifier), None)
            if quote is not None:
                source = next(record["source_task_id"] for record in self.library.research_drafts
                              if record["draft_id"] == identifier)
                source_position = next((row["task_position"] for row in self.state["results"]
                    if row["research"]["task_id"] == source), None)
                quote.update(task_position=position, source_task_position=source_position,
                    screening_compute_cost=quote["candidate"]["method_opportunity"]["planning_compute_cost"])
                attempt["cached_quote"] = quote
            self.state["method_revisit_attempts"].append(attempt)
            attempts.append(attempt)
        return attempts

    def _refresh_method_feedback(self):
        """Let measured method returns reopen or withdraw an already paid plan."""
        if not self.config.use_memory or not self.config.learn_research_policy:
            return
        methods = knowledge_snapshot(self.library)["draft"]
        remaining = self._remaining()
        by_position, changed_positions = {}, set()

        def remove_queued(position, identifier):
            self.state["pending_revisits"] = [offer for offer in self.state["pending_revisits"]
                if not (offer["task_position"] == position
                    and offer["candidate"].get("research_draft_id") == identifier
                    and offer["candidate"].get("method_experiment") is not None)]

        for attempt in self.state["method_revisit_attempts"]:
            quote = attempt.get("cached_quote")
            if quote is None:
                continue
            position, identifier = attempt["task_position"], attempt["draft_id"]
            previous = self.state["results"][position]["research"]
            if (attempt["paid_batches"] != len(previous["batches"])
                    or attempt["program"] != previous["program"]
                    or methods.get(identifier) != attempt["method"]):
                attempt.pop("cached_quote")
                attempt.pop("deferred_offer", None)
                remove_queued(position, identifier)
                changed_positions.add(position)
                continue
            before = deepcopy(quote["method_feedback"])
            refresh_numeric_method_quote(quote, self.library.action_experience, remaining)
            changed = before != quote["method_feedback"]
            by_position.setdefault(position, []).append((attempt, quote, before, changed))
            if changed:
                changed_positions.add(position)

        for position in sorted(changed_positions):
            entries = by_position.get(position, [])
            # A negative update can displace the current plan. Retained rival
            # plans must still compete even when their own feedback is unchanged.
            for attempt, quote, before, changed in entries:
                remove_queued(position, attempt["draft_id"])
                attempt.pop("deferred_offer", None)
            for attempt, quote, before, changed in entries:
                offer = deepcopy(quote) if quote["expected_marginal_value"] > 0. else None
                event = {"action": "refresh_method_feedback" if changed else "reconsider_cached_method",
                    "task_position": position, "source_task_position": quote["source_task_position"],
                    "draft_id": attempt["draft_id"], "compute_cost": 0,
                    "previous_feedback": before, "method_feedback": deepcopy(quote["method_feedback"]),
                    "expected_marginal_value": quote["expected_marginal_value"],
                    "scheduled": False, "withdrawn": offer is None}
                self._queue_knowledge_offer(position, quote["source_task_position"], offer,
                    quote["screening_compute_cost"], [attempt], event)

    def _refresh_source_feedback(self):
        """Apply measured source returns to pending offers without rescreening."""
        if not self.config.use_memory or self.config.strategy == "fixed":
            return
        withdrawn = []
        for offer in list(self.state["pending_revisits"]):
            state = self._source_feedback_state(offer)
            if state is None:
                continue
            score, matches, statuses, origins = state
            if origins and len(statuses) == len(origins) and all(
                    status == "negative" for status in statuses):
                withdrawn.append(offer)
                self.state["scheduler_events"].append({
                    "action": "withdraw_successor_after_negative_return",
                    "task_position": offer["task_position"],
                    "source_task_position": offer.get("source_task_position"),
                    "source_concept_ids": [origin.get("source_id") for origin in origins],
                    "conditions": [origin.get("condition") for origin in origins],
                    "matching_returns": matches, "compute_cost": 0, "scheduled": False})
                continue
            previous = offer.get("knowledge_return_score", 0.0)
            offer["knowledge_return_score"] = score
            offer["knowledge_return_matches"] = matches
            offer["source_return_status"] = statuses
            if score != previous:
                self.state["scheduler_events"].append({
                    "action": "refresh_source_feedback",
                    "task_position": offer["task_position"],
                    "source_task_position": offer.get("source_task_position"),
                    "source_concept_ids": [origin.get("source_id") for origin in origins],
                    "previous_score": previous, "knowledge_return_score": score,
                    "matching_returns": matches, "compute_cost": 0, "scheduled": True})
        if withdrawn:
            self.state["pending_revisits"] = [offer for offer in self.state["pending_revisits"]
                if offer not in withdrawn]

    def _source_feedback_state(self, offer):
        """Return source score/status for one offer using already paid records."""
        origins = offer.get("candidate", {}).get("transfer_evidence", [])
        if not origins:
            return None
        offer_program = Program.from_dict(offer["candidate"]["program"])
        source_ids = {origin["source_id"] for origin in origins}
        scores, matches, statuses = [], 0, []
        for origin in origins:
            source = next((item for item in self.library.concepts
                           if item.concept_id == origin.get("source_id")), None)
            if source is None:
                continue
            condition = origin.get("condition")
            priority = return_priority(source.reuse_history, condition,
                program=offer_program, source_ids=source_ids)
            scores.append(-priority[0])
            records = matching_returns(source.reuse_history, condition,
                program=offer_program, source_ids=source_ids)
            matches += len(records)
            if records:
                confirmed = [record for record in records
                             if record.get("confirmation_status", "confirmed") == "confirmed"]
                actionable = [record for record in records
                               if record.get("confirmation_status", "confirmed") != "provisional"]
                provisional_positive = any(
                    record.get("confirmation_status") == "provisional"
                    and record.get("retained_normalized_gain", 0.) > 0.
                    for record in records)
                statuses.append("negative" if actionable and all(
                    record.get("retained_normalized_gain", 0.) < 0. for record in actionable)
                    else "positive" if any(
                    record.get("retained_normalized_gain", 0.) > 0. for record in confirmed)
                    else "provisional" if provisional_positive
                    else "neutral")
            else:
                statuses.append("unknown")
        if len(statuses) != len(origins):
            return None
        return (sum(scores) / len(scores) if scores else 0.0, matches, statuses, origins)

    def _queue_knowledge_offer(self, position, source_position, offer, screening_cost,
                               attempts, event):
        if offer and offer.get("basis") == "new_partial_method_distinguishing_experiment":
            event.update(method_opportunity=deepcopy(offer["candidate"]["method_opportunity"]),
                         minimum_resources=dict(offer["minimum_resources"]))
        if offer:
            offer.update(task_position=position, source_task_position=source_position,
                         screening_compute_cost=screening_cost)
            feedback = self._source_feedback_state(offer)
            if feedback is not None:
                score, matches, statuses, origins = feedback
                offer.update(knowledge_return_score=score,
                    knowledge_return_matches=matches, source_return_status=statuses)
                if origins and len(statuses) == len(origins) and all(
                        status == "negative" for status in statuses):
                    event.update(scheduled=False, withdrawn=True,
                        reason="negative_source_return")
                    self.state["scheduler_events"].append(event)
                    return
            reprice_numeric_method_offer(offer, self._remaining())
        for queued in self.state["pending_revisits"]:
            reprice_numeric_method_offer(queued, self._remaining())
        if offer and all(self._remaining()[key] >= amount
                         for key, amount in offer["minimum_resources"].items()):
            event["scheduled"] = retain_revisit_offer(self.state["pending_revisits"], offer)
        elif offer and offer.get("basis") == "new_partial_method_distinguishing_experiment":
            attempt = next(item for item in attempts
                if item["draft_id"] in offer["candidate"]["method_opportunity"]["draft_ids"])
            attempt["deferred_offer"] = deepcopy(offer)
            event["reason"] = "preserve_priced_method_until_visited_resources_are_available"
        self.state["scheduler_events"].append(event)

    def _offer_retained_methods(self, position):
        """A new paid failure can make an existing procedure worth an experiment."""
        if not self.config.use_memory or self.config.strategy == "fixed":
            return
        previous = self.state["results"][position]["research"]
        if (previous["status"] != "unresolved"
                or (not previous["selected_channels"] and previous.get("risk_quantile", .5) == .5)
                or max(previous.get("selection_loss") or 0., previous.get("confirmation_loss") or 0.)
                    <= self.config.minimum_improvement):
            return
        methods = knowledge_snapshot(self.library)["draft"]
        drafts = {record["draft_id"]: methods[record["draft_id"]]
            for record in self.library.research_drafts if record["draft_id"] in methods
            and methods[record["draft_id"]]["available"]
            and (record.get("source_seed") != self.state["tasks"][position]["seed"]
                 if record.get("risk_method") is not None else record["source_task_id"] != previous["task_id"])
            and not self._method_screened(position, methods[record["draft_id"]], previous)}
        if not drafts:
            return
        from .numeric_method_revisits import offer_numeric_method_experiment
        from .unified_research import UnifiedResearchAgent
        remaining = self._remaining()
        with self.lab_factory(self._laboratory_arguments(position)) as laboratory:
            length = min(24, laboratory.spec.max_probe_length)
            if remaining["observation_cost"] < 2 * length:
                return
            work = min(self.config.compute_budget // 8,
                max(0, remaining["compute_cost"] - 6 * previous["costs"]["observation_cost"] - 32 * length))
            if not work:
                return
            budget, actions = SearchBudget(work), []
            agent = UnifiedResearchAgent(self.config, library=self.library,
                measurement_budget=self._task_limits(position)["measurement_budget"])
            offer = offer_numeric_method_experiment(agent, laboratory, previous, set(drafts), budget,
                actions=actions, remaining_resources=remaining)
        self.state["screening_compute_cost"] += budget.spent
        attempts = self._remember_method_screens(position, previous, drafts, actions)
        if not budget.spent:
            return
        source = (offer["candidate"]["method_opportunity"]["source_task_ids"][0] if offer else None)
        chosen_source = next((record for record in self.library.research_drafts if offer is not None
            and record["draft_id"] == offer["candidate"]["research_draft_id"]), {})
        source_seed = chosen_source.get("source_seed")
        source_position = next((item["task_position"] for item in self.state["results"]
            if (self.state["tasks"][item["task_position"]]["seed"] == source_seed
                if source_seed is not None else item["research"]["task_id"] == source)), None)
        event = {"action": "try_retained_method", "task_position": position,
            "source_task_position": source_position, "screening": deepcopy(actions),
            "compute_cost": budget.spent, "old_evidence_gain": offer["old_evidence_gain"] if offer else 0.,
            "basis": offer.get("basis", "old_paid_evidence") if offer else "old_paid_evidence",
            "scheduled": False}
        self._queue_knowledge_offer(position, source_position, offer, budget.spent, attempts, event)

    def _offer_new_knowledge(self, source_position, concept_ids, object_model_ids=(), changes=()):
        if not self.config.use_memory or self.config.strategy == "fixed":
            return
        from .unified_research import UnifiedResearchAgent
        from .unified_revisits import offer_successor_knowledge
        current_methods = knowledge_snapshot(self.library)["draft"]
        for record in self.state["results"]:
            position, previous = record["task_position"], record["research"]
            if position == source_position or previous["status"] not in {"unresolved", "completed"}:
                continue
            task_changes = [change for change in changes if change["kind"] != "draft"
                or not self._method_screened(position, current_methods.get(change["knowledge_id"]), previous)]
            remaining = self._remaining()
            unseen = set()
            if previous["status"] == "unresolved":
                available = set(record.get("available_concepts", ())) | set(previous.get("concepts_added", ()))
                unseen = {concept.concept_id for concept in self.library.concepts
                    if concept.status != "revoked" and concept.concept_id not in available}
                if source_position is None and any(offer["task_position"] == position
                        for offer in self.state["pending_revisits"]):
                    continue
            relevant = set(concept_ids) | unseen
            if not (relevant or object_model_ids or task_changes):
                continue
            with self.lab_factory(self._laboratory_arguments(position)) as laboratory:
                length = min(24, laboratory.spec.max_probe_length)
                first_length = (max(3, max(len(batch["probe"]["sequence"])
                    for batch in previous["batches"]) + 1)
                    if previous["status"] == "completed" else length)
                minimum_rows = (length if previous["status"] == "unresolved"
                    and laboratory.spec.risk_quantile == .5
                    and not getattr(laboratory, "measurement_views", None)
                    else length + 1 if task_changes and previous["status"] == "unresolved"
                    and getattr(laboratory, "measurement_views", None)
                    else first_length + max(length, first_length))
                if remaining["observation_cost"] < minimum_rows:
                    continue
                work = min(self.config.compute_budget // 8,
                           max(0, remaining["compute_cost"]
                               - 6 * previous["costs"]["observation_cost"] - 32 * length))
                if not work:
                    continue
                screen = record.get("successor_screen", {})
                if (screen.get("paid_batches") == len(previous["batches"])
                        and screen.get("program") == previous["program"]
                        and screen.get("compute_limit", 0) >= work):
                    unseen.difference_update(screen.get("source_ids", ()))
                relevant = set(concept_ids) | unseen
                if not (relevant or object_model_ids or task_changes):
                    continue
                budget = SearchBudget(work)
                agent = UnifiedResearchAgent(self.config, library=self.library,
                    measurement_budget=self._task_limits(position)["measurement_budget"])
                method_actions = []
                offer = offer_successor_knowledge(agent, laboratory, previous, relevant, budget,
                    object_model_ids=set(object_model_ids), knowledge_changes=task_changes,
                    method_screening_actions=method_actions, remaining_resources=remaining)
            self.state["screening_compute_cost"] += budget.spent
            if unseen:
                old_screen = record.get("successor_screen", {})
                seen = set(old_screen.get("source_ids", ())) if (
                    old_screen.get("paid_batches") == len(previous["batches"])
                    and old_screen.get("program") == previous["program"]) else set()
                record["successor_screen"] = {"source_ids": sorted(seen | relevant),
                    "paid_batches": len(previous["batches"]), "program": deepcopy(previous["program"]),
                    "compute_limit": work}
            producing_position = source_position
            if offer and offer["candidate"].get("source_concept_ids"):
                producers = {next((item["task_position"] for item in self.state["results"]
                    if source in item["research"].get("concepts_added", ())), None)
                    for source in offer["candidate"]["source_concept_ids"]}
                producing_position = next(iter(producers)) if len(producers) == 1 else None
            attempted_methods = self._remember_method_screens(position, previous,
                {change["knowledge_id"]: current_methods[change["knowledge_id"]]
                    for change in task_changes if change["kind"] == "draft" and change["available"]},
                method_actions)
            event = {"action": "try_successor_knowledge", "task_position": position,
                "source_task_position": producing_position, "source_concept_ids": sorted(relevant),
                "previously_unavailable_concept_ids": sorted(unseen),
                "source_object_model_ids": list(object_model_ids),
                "knowledge_changes": list(task_changes),
                "compute_cost": budget.spent, "old_evidence_gain": offer["old_evidence_gain"] if offer else 0.,
                "basis": offer.get("basis", "old_paid_evidence") if offer else "old_paid_evidence",
                "method_screening": deepcopy(method_actions), "scheduled": False}
            self._queue_knowledge_offer(position, producing_position, offer, budget.spent, attempted_methods, event)

    def _offer_instrument_conditions(self, pending):
        """Price unresolved measured boundaries against the visited-task pool."""
        if not self.config.active_probes or self.config.strategy == "fixed":
            return pending
        from .instrument_condition_revisits import (
            screen_instrument_condition_revisit, instrument_condition_offer)
        from .unified_research import UnifiedResearchAgent
        quotes = self.state["condition_revisit_quotes"]
        priced = []
        for record in self.state["results"]:
            position, previous = record["task_position"], record["research"]
            condition = previous.get("condition_uncertainty") or {}
            if (previous["status"] != "unresolved" or not previous["selected_channels"]
                    or condition.get("status") != "unresolved"
                    or (condition.get("expected_prediction_ambiguity") or 0.)
                        <= self.config.minimum_improvement):
                continue
            identity = {"program": previous["program"], "paid_batches": len(previous["batches"]),
                        "channels": previous["selected_channels"]}
            cached = next((item for item in quotes if item["task_position"] == position), None)
            remaining = self._remaining()
            work = min(self.config.compute_budget // 8, remaining["compute_cost"])
            if (cached is None or cached["evidence"] != identity
                    or (cached["screen"] is None and work > cached["screening_budget"])):
                with self.lab_factory(self._laboratory_arguments(position)) as laboratory:
                    length = min(24, laboratory.spec.max_probe_length)
                    rate = sum(channel.cost_per_row for channel in laboratory.channels
                               if channel.name in previous["selected_channels"])
                    if (laboratory.spec.risk_quantile != .5
                            or remaining["observation_cost"] < 2 * length
                            or remaining["measurement_cost"] < 2 * length * rate):
                        continue
                    if not work:
                        continue
                    budget = SearchBudget(work)
                    agent = UnifiedResearchAgent(self.config, library=self.library,
                        measurement_budget=self._task_limits(position)["measurement_budget"])
                    screen = screen_instrument_condition_revisit(agent, laboratory, previous, budget)
                self.state["screening_compute_cost"] += budget.spent
                paid_screening = budget.spent + (
                    cached["screening_compute_cost"] if cached is not None
                    and cached["evidence"] == identity else 0)
                if screen is not None:
                    screen["compute_cost"] = paid_screening
                replacement = {"task_position": position, "evidence": deepcopy(identity),
                    "screen": screen, "screening_compute_cost": paid_screening,
                    "screening_budget": work}
                if cached is None:
                    quotes.append(replacement)
                else:
                    cached.clear()
                    cached.update(replacement)
                cached = replacement
                self.state["scheduler_events"].append({
                    "action": "screen_instrument_condition_revisit", "task_position": position,
                    "compute_cost": budget.spent, "priced": screen is not None,
                    "prediction_ambiguity": condition["expected_prediction_ambiguity"]})
            if cached["screen"] is None:
                continue
            priced.append((position, cached))
        # Every candidate competes after all newly required screens are paid.
        # Pricing earlier instruments before later screens uses a larger pool
        # and lets task iteration order change the research ranking.
        remaining = self._remaining()
        for position, cached in priced:
            # New shared resources or applicable experience can change the
            # investment decision without paying to reconstruct the same plan.
            companion = self.state.get("portfolio_companion")
            admission = None
            if companion is not None and companion["task_position"] == position:
                if companion["evidence"] == cached["evidence"]:
                    admission = companion["admission_resources"]
                else:
                    self.state.pop("portfolio_companion")
            offer = instrument_condition_offer(cached["screen"], remaining,
                experience=self.library.action_experience, admission_resources=admission)
            if offer is not None:
                offer.update(task_position=position, source_task_position=position,
                    question_id=f"instrument_condition:{position}",
                    screening_compute_cost=cached["screening_compute_cost"])
                pending.append(offer)
                self.state["scheduler_events"].append({
                    "action": "price_instrument_condition_revisit", "task_position": position,
                    "compute_cost": 0, "scheduled": True,
                    "relative_gain": offer["relative_gain"],
                    "minimum_resources": dict(offer["minimum_resources"])})
        return pending

    def _offer_response_questions(self):
        """Refresh frozen questions from paid reports against the visited-task pool."""
        self._refresh_method_feedback()
        self._refresh_source_feedback()
        for record in self.state["results"]:
            self._offer_retained_methods(record["task_position"])
        pending = [item for item in self.state["pending_revisits"] if "question_id" not in item]
        remaining = self._remaining()
        for offer in pending:
            reprice_numeric_method_offer(offer, remaining)
        # A newly visited task can fund an already priced method. Reuse its
        # frozen plan; repeated library accounting does not pay for screening.
        for attempt in self.state["method_revisit_attempts"]:
            offer = attempt.get("deferred_offer")
            if offer is None:
                continue
            position = attempt["task_position"]
            previous = self.state["results"][position]["research"]
            if (len(previous["batches"]) != attempt["paid_batches"]
                    or previous["program"] != attempt["program"]):
                attempt.pop("deferred_offer")
                continue
            reprice_numeric_method_offer(offer, remaining)
            if not all(remaining[key] >= amount for key, amount in offer["minimum_resources"].items()):
                continue
            scheduled = retain_revisit_offer(pending, offer)
            if scheduled:
                attempt.pop("deferred_offer")
            self.state["scheduler_events"].append({"action": "fund_deferred_method_experiment",
                "task_position": position, "source_task_position": offer["source_task_position"],
                "method_opportunity": deepcopy(offer["candidate"]["method_opportunity"]),
                "minimum_resources": dict(offer["minimum_resources"]),
                "compute_cost": 0, "scheduled": scheduled})
        for record in self.state["results"]:
            previous = record["research"]
            questions = [item for item in previous.get("response_questions", [])
                         if item["status"] == "pending"]
            if not questions:
                continue
            position = record["task_position"]
            with self.lab_factory(self._laboratory_arguments(position)) as laboratory:
                # Instrument and risk questions have their own executable
                # continuations. These offers concern ordinary scalar responses.
                if laboratory.spec.risk_quantile != .5 or getattr(laboratory, "measurement_views", None):
                    continue
                is_refuted = (lambda program: self.library.transfer_refuted(laboratory.spec, program)) \
                    if self.config.use_memory else None
                budget = SearchBudget(self._remaining()["compute_cost"])
                events = refresh_response_question_quotes(previous, questions, budget,
                    minimum_improvement=self.config.minimum_improvement, is_refuted=is_refuted)
                self.state["screening_compute_cost"] += budget.spent
                self.state["scheduler_events"].extend({**event, "task_position": position}
                    for event in events)
                remaining = self._remaining()
                for question in questions:
                    offer = response_question_offer(previous, question,
                        laboratory.spec.max_probe_length, remaining, self.config.minimum_improvement,
                        resource_limits=self._task_limits(position),
                        is_refuted=is_refuted)
                    if offer is not None:
                        offer.update(task_position=position, source_task_position=position)
                        pending.append(offer)
        pending = self._offer_instrument_conditions(pending)
        remaining = self._remaining()
        for offer in pending:
            reprice_numeric_method_offer(offer, remaining)
        self.state["pending_revisits"] = pending

    def _remember_portfolio_companion(self, choice, remaining):
        """Keep the chosen companion eligible until the next portfolio decision."""
        previous = self.state.pop("portfolio_companion", None)
        companion = choice["decision"]["companion"] if choice is not None else None
        if companion is None:
            return
        offer = self.state["pending_revisits"][companion["index"]]
        if offer.get("basis") != "unresolved_instrument_condition":
            return
        experiment = offer["candidate"]["instrument_condition_experiment"]
        retained = {
            "task_position": offer["task_position"],
            "evidence": {"program": deepcopy(experiment["program"]),
                "paid_batches": experiment["paid_batches"],
                "channels": list(offer["candidate"]["channels"])},
            "admission_resources": dict(remaining)}
        # Reselecting the same still-unexecuted companion renews its existing
        # admission. Moving that reference to the shrunken pool would recreate
        # the rejection of a pair we have just explicitly chosen again.
        if (previous is not None and previous["task_position"] == retained["task_position"]
                and previous["evidence"] == retained["evidence"]):
            retained["admission_resources"] = previous["admission_resources"]
        self.state["portfolio_companion"] = retained

    def _revisit_allocation(self, offer, pending, remaining):
        """Quote the same complete execution limits used by the continuation."""
        position = offer["task_position"]
        candidate = offer["candidate"]
        if not all(remaining[key] >= amount for key, amount in offer["minimum_resources"].items()):
            return None
        # Share remaining resources between currently payable visits. A task
        # may borrow unused first-visit resources, never another task's reserve.
        pending = [item for item in pending
                   if all(
                       remaining[key] >= value for key, value in item["minimum_resources"].items())]
        method = offer.get("basis") == "new_partial_method_distinguishing_experiment"
        instrument_condition = offer.get("basis") == "unresolved_instrument_condition"
        allocation = (dict(offer["minimum_resources"]) if method else
            allocate_complete_research(offer["minimum_resources"],
                [item["minimum_resources"] for item in pending], remaining))
        if method and candidate.get("method_experiment") is not None:
            # Refitting can expose a rare reading boundary absent from the
            # initial quote. Let its measured information value choose how much
            # of the visited-task share to use; unused capacity stays pooled.
            for key in ("observation_cost", "measurement_cost"):
                other_minimum = sum(item["minimum_resources"][key] for item in pending
                    if item["task_position"] != position)
                share = min(remaining[key] // (1 + len(pending)),
                            remaining[key] - other_minimum)
                allocation[key] = min(remaining[key], max(allocation[key], share))
        if instrument_condition:
            # Execute this quoted experiment and leave other payable visits
            # their minimum complete experiment and confirmation resources.
            for key in allocation:
                other_minimum = sum(item["minimum_resources"][key] for item in pending
                                    if item["task_position"] != position)
                allocation[key] = max(offer["minimum_resources"][key],
                    min(allocation[key], max(0, remaining[key] - other_minimum)))
        return allocation

    def _perform_revisit(self, offer, *, allocation_limits=None):
        from .unified_research import UnifiedResearchAgent
        position = offer["task_position"]
        previous = self.state["results"][position]["research"]
        candidate = deepcopy(offer["candidate"])
        remaining = self._remaining()
        allocation = ({key: min(value, allocation_limits[key]) for key, value in remaining.items()}
            if allocation_limits is not None else
            self._revisit_allocation(offer, self.state["pending_revisits"], remaining))
        if allocation is None or any(allocation[key] < amount
                for key, amount in offer["minimum_resources"].items()):
            self.state["scheduler_events"].append({"action": "defer_revisit", "task_position": position,
                                                   "reason": "remaining_resources"})
            return False
        method = offer.get("basis") == "new_partial_method_distinguishing_experiment"
        instrument_condition = offer.get("basis") == "unresolved_instrument_condition"
        limits = {key: previous["costs"][key] + value for key, value in allocation.items()}
        item = self.state["tasks"][position]
        available = [concept.concept_id for concept in self.library.concepts]
        available_models = {model["model_id"] for model in self.library.object_models}
        before_knowledge = knowledge_snapshot(self.library)
        with self.lab_factory(self._laboratory_arguments(position)) as laboratory:
            if not instrument_condition and candidate.get("program") is not None and self.library.transfer_refuted(
                    laboratory.spec_for(candidate.get("channels", ())),
                    Program.from_dict(candidate["program"]), fallback=candidate.get("fallback", 0.)):
                self.state["scheduler_events"].append({
                    "action": "withdraw_refuted_revisit", "task_position": position,
                    "source_task_position": offer["source_task_position"],
                    "reason": "paid_counterexample_to_this_bound_program"})
                return False
            report = UnifiedResearchAgent(replace(self.config, seed=item["seed"],
                observation_budget=limits["observation_cost"], compute_budget=limits["compute_cost"]),
                measurement_budget=limits["measurement_cost"], library=self.library,
                downstream_demands=self._workflow_demands(exclude_position=position)).run(laboratory,
                    previous=previous, continuation_candidate=candidate)
            if method:
                if laboratory.spec.risk_quantile == .5:
                    from .numeric_method_revisits import record_numeric_method_visit as record_method
                else:
                    from .risk_research_methods import record_risk_method_visit as record_method
                record_method(self.library,
                    laboratory.spec_for(offer["candidate"]["channels"]), previous, report,
                    offer["candidate"]["method_opportunity"], offer["screening_compute_cost"])
        self.state["results"][position]["research"] = report
        if method:
            # This visit has already tested its source on the enlarged evidence;
            # JSON restoration must not price that same measured attempt again.
            for attempt in self.state["method_revisit_attempts"]:
                if (attempt["task_position"] == position
                        and attempt["draft_id"] in offer["candidate"]["method_opportunity"]["draft_ids"]):
                    attempt.update(paid_batches=len(report["batches"]), program=deepcopy(report["program"]))
                    attempt.pop("cached_quote", None)
                    attempt.pop("deferred_offer", None)
        if "question_id" in offer and not any(report["visit_costs"].values()):
            for question in report.get("response_questions", []):
                if question["question_id"] == offer["question_id"]:
                    question["deferred_resources"] = dict(remaining)
        self.state["revisits"].append({"task_position": position,
            "source_task_position": offer["source_task_position"], "candidate": candidate,
            "previous": previous, "visit_costs": report["visit_costs"], "resource_limits": limits,
            "available_concepts": available, "status": report["status"],
            "basis": offer.get("basis", "old_paid_evidence"),
            **({"screening_compute_cost": offer["screening_compute_cost"],
                "method_visit_outcome": deepcopy(report["continuation"]["method_visit_outcome"])}
               if method else {})})
        new_ids = [concept.concept_id for concept in self.library.concepts if concept.concept_id not in available]
        new_models = [model["model_id"] for model in self.library.object_models
                      if model["model_id"] not in available_models]
        # This completed visit replaces the evidence used by the other plans
        # for the same task. A rejected, unexecuted offer leaves rivals intact.
        if (len(report["batches"]) != len(previous["batches"])
                or report["program"] != previous["program"]):
            self.state["pending_revisits"] = [item for item in self.state["pending_revisits"]
                if item["task_position"] != position]
        self._offer_new_knowledge(position, new_ids, new_models,
            knowledge_changes(before_knowledge, knowledge_snapshot(self.library)))
        self._offer_response_questions()
        return True

    def run(self, *, max_tasks=None, on_checkpoint=None):
        from .unified_research import UnifiedResearchAgent
        if max_tasks is not None and max_tasks < 1:
            raise ValueError("task steps must be positive")
        if not self.state["evaluation"]:
            # A restored unfinished visit may never have seen knowledge that
            # was already in the incoming library. It need not wait for another
            # concept to be created before competing for a paid confirmation.
            self._offer_new_knowledge(None, ())
        self._offer_response_questions()
        visited = 0
        while len(self.state["results"]) < len(self.state["tasks"]) or self.state["pending_revisits"]:
            if max_tasks is not None and visited >= max_tasks:
                break
            position = len(self.state["results"])
            if position < len(self.state["tasks"]):
                item = self.state["tasks"][position]
                available = [concept.concept_id for concept in self.library.concepts]
                available_models = {model["model_id"] for model in self.library.object_models}
                before_knowledge = knowledge_snapshot(self.library)
                with self.lab_factory(self._laboratory_arguments(position)) as laboratory:
                    limits = self._task_limits(position)
                    report = UnifiedResearchAgent(replace(self.config, seed=item["seed"],
                        observation_budget=limits["observation_budget"],
                        compute_budget=limits["compute_budget"]),
                        measurement_budget=limits["measurement_budget"],
                        library=self.library,
                        downstream_demands=self._workflow_demands()).run(laboratory)
                self.state["results"].append({"task_position": position,
                    "available_concepts": available, "research": report})
                new_ids = [concept.concept_id for concept in self.library.concepts if concept.concept_id not in available]
                new_models = [model["model_id"] for model in self.library.object_models
                              if model["model_id"] not in available_models]
                self._offer_new_knowledge(position, new_ids, new_models,
                    knowledge_changes(before_knowledge, knowledge_snapshot(self.library)))
                self._offer_response_questions()
            else:
                remaining = self._remaining()
                budget = SearchBudget(remaining["compute_cost"])
                choice = choose_research_visit(self.state["pending_revisits"], remaining,
                    budget, allocation_for=self._revisit_allocation,
                    knowledge_tie_tolerance=self.config.knowledge_tie_tolerance)
                self.state["screening_compute_cost"] += budget.spent
                self._remember_portfolio_companion(choice, remaining)
                if choice is not None:
                    offer, allocation = choice["offer"], choice["allocation"]
                    self.state["scheduler_events"].append({"action": "choose_research_portfolio",
                        **choice["decision"], "task_position": offer["task_position"] if offer else None,
                        "allocation": allocation})
                else:
                    # Pinball/object queues retain their existing ranking.
                    offer = max(self.state["pending_revisits"],
                        key=lambda item: (item["relative_gain"],
                                          item.get("knowledge_return_score", 0.)), default=None)
                    allocation = None
                if offer is None:
                    self.state["pending_revisits"] = []
                else:
                    self.state["pending_revisits"].remove(offer)
                    self._perform_revisit(offer, allocation_limits=allocation)
            visited += 1
            self.state["status"] = ("completed" if len(self.state["results"]) ==
                                    len(self.state["tasks"]) and not self.state["pending_revisits"] else "paused")
            if on_checkpoint:
                on_checkpoint(self.to_dict())
        return self.to_dict()

    def evaluate(self, *, length=64, trajectories=4, seed_offset=1000003):
        if len(self.state["results"]) != len(self.state["tasks"]) or self.state["pending_revisits"]:
            raise ValueError("finish learning before independent evaluation")
        for position in range(len(self.state["evaluation"]), len(self.state["results"])):
            item = self.state["tasks"][position]
            result = self.state["results"][position]["research"]
            with self.lab_factory(self._laboratory_arguments(position)) as laboratory:
                rows = laboratory.evaluation_rows(result["selected_channels"],
                    seed=item["seed"] + seed_offset, length=length, trajectories=trajectories)
                rate = sum(channel.cost_per_row for channel in laboratory.channels
                           if channel.name in result["selected_channels"])
                risk_quantile = laboratory.spec.risk_quantile
            program = Program.from_dict(result["program"])
            measured = loss(program, rows, result["fallback"], risk_quantile)
            score = {"task_position": position, "rows": len(rows),
                "observation_cost": len(rows), "measurement_cost": len(rows) * rate,
                "compute_cost": len(rows),
                "loss": measured, "risk_quantile": risk_quantile,
                "loss_kind": "mse" if risk_quantile == .5 else "pinball"}
            if risk_quantile == .5:
                score["mse"] = measured
            self.state["evaluation"].append(score)
        return self.to_dict()

    def to_dict(self):
        report = {**deepcopy(self.state), "knowledge": self.library.to_dict()}
        report["pending_tasks"] = list(range(len(report["results"]), len(report["tasks"])))
        report["costs"] = self._costs()
        return report


def unified_summary(report):
    return {key: report[key] for key in ("status", "pending_tasks", "costs", "evaluation")} | {
        "learned_concepts": len(report["knowledge"]["concepts"]),
        "pending_revisits": [row["task_position"] for row in report["pending_revisits"]],
        "revisits": len(report["revisits"]),
        "results": [{"task_position": row["task_position"],
            "status": row["research"]["status"], "costs": row["research"]["costs"],
            "selected_channels": row["research"]["selected_channels"],
            "expression": Program.from_dict(row["research"]["program"]).describe()}
            for row in report["results"]]}
