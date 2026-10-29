"""Choose object research proposals; the ordinary experiment frontier confirms them."""

from copy import deepcopy

from .action_policy import ActionOption, ResearchActionPolicy
from .competing_object_experiments import _forecast, assess_competing_object_probe
from .evaluation import initial_prediction, loss
from .language import Program
from .models import SearchBudget
from .relation_input_diagnosis import bind_relation_entry, diagnose_relation_inputs, measured_relation_entry
from .relation_revisions import _has_relation, propose_relation_revisions


class ObjectResearchActions:
    """Keep one task's attempted directions and its separately owned return.

    Proposals do not bypass structural interventions or relation questions.
    The caller supplies their final, freshly purchased confirmation to settle.
    """

    def __init__(self, agent, laboratory, *, acquire, paired_for, command_batches,
                 view_rows, transitions_for, budget, actions, transfer, graph,
                 remaining, length):
        self.agent, self.lab = agent, laboratory
        self.acquire, self.paired_for, self.command_batches = acquire, paired_for, command_batches
        self.view_rows, self.transitions_for = view_rows, transitions_for
        self.budget, self.actions, self.transfer, self.graph = budget, actions, transfer, graph
        self.remaining, self.length = remaining, min(length, laboratory.spec.max_probe_length)
        self.confirmation_length = self.length
        self.views = [tuple(view) for view in laboratory.measurement_views]
        self.rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        self.learn = agent.config.use_memory and agent.config.learn_research_policy
        self.policy = ResearchActionPolicy([], agent._spec(laboratory, ()),
            agent.library.action_experience if self.learn else {})
        self.attempted, self.pending, self.field_status = set(), None, "unassessed_inputs"

    def _rate(self, channels):
        return sum(self.rates[name] for name in channels)

    def _costs(self, before):
        after = self.remaining()
        return {name + "_cost": before[name + "_remaining"] - after[name + "_remaining"]
                for name in ("observation", "measurement", "compute")}

    def _data(self, channels):
        batches = ([[measured for _, measured in batch] for batch in self.paired_for(channels)]
                   if channels else self.command_batches())
        if len(batches) > 1:
            return [row for batch in batches[:-1] for row in batch], list(batches[-1])
        rows = list(batches[0]) if batches else []
        return rows, list(rows)

    def confirmation_channels(self, selected_channels):
        """Find a legal acquisition that supplies both frozen response views."""
        needed = {tuple(selected_channels)}
        if self.pending:
            needed.add(tuple(self.pending["baseline_channels"]))
        return self._cover(needed)

    def _cover(self, needed):
        return next(view for view in sorted(self.views, key=self._rate)
                    if needed <= {(), *map(tuple, self.lab.projectable_views(view))})

    def _acquire(self, channels, purpose, *, probe=None, rows_count=None, reserve_channels=()):
        count = probe.cost if probe is not None else self.length if rows_count is None else rows_count
        remaining = self.remaining()
        if (remaining["observation_remaining"] < count + self.confirmation_length or
                remaining["measurement_remaining"] < count * self._rate(channels)
                + self.confirmation_length * self._rate(reserve_channels)):
            return None
        return self.acquire(channels, purpose, probe=probe,
                            rows_count=count if probe is None else rows_count)

    def _relation(self, channels, calls):
        training, validation = self._data(channels)
        paid = [*training, *(row for row in validation if row not in training)]
        available = max(0, self.budget.remaining - 2 * self.confirmation_length)
        local = SearchBudget(min(20000, available, max(4 * len(paid), available // 3)))
        fitting = SearchBudget(local.remaining * 3 // 4)
        claims = propose_relation_revisions(self.agent._spec(self.lab, channels), paid, fitting, calls,
                                            tolerance=self.agent.config.minimum_improvement)
        local.spend(fitting.spent)
        checked = self.agent._assess(claims, paid, local,
            initial_prediction(paid, self.lab.spec.risk_quantile), self.lab.spec.risk_quantile) if paid else []
        self.budget.spend(local.spent)
        for entry in checked:
            entry["selection_basis"] = "all_acquired_evidence"
        return self.agent._best(checked), channels, paid, list(paid), checked

    def _inputs(self, parent, failed_probe, failed_rows):
        count = len(failed_rows)
        if not self.budget.spend(count):
            return None
        candidate = parent["candidate"]
        plan = {"probe": failed_probe, "predicted_fields": {}, "hypotheses": [{
            "hypothesis_id": candidate.candidate_id, "candidate_id": candidate.candidate_id,
            "program": candidate.program.to_dict(), "predictions": _forecast(candidate.program, failed_probe.sequence)}]}
        evidence = assess_competing_object_probe(plan, [(row, row) for row in failed_rows], self.budget,
                                                tolerance=self.agent.config.minimum_improvement)
        diagnosis = diagnose_relation_inputs(self.agent, self.lab, parent, parent, plan, evidence,
            self._acquire, self.paired_for, self.budget, self.actions)
        if diagnosis is None:
            return None
        self.counterexamples.extend(diagnosis["command_rows"])
        self.field_status = diagnosis["status"]
        entry = diagnosis.get("entry")
        if entry is None:
            return None
        entry["input_recovery_response_status"] = diagnosis["response_status"]
        # The old direction already met a response counterexample. Rebinding
        # its input contract creates this direction's proposal, not its success.
        entry.pop("research_action_origin", None)
        training, validation = self._data(())
        paid = [*training, *(row for row in validation if row not in training)]
        if not self.budget.spend(len(paid)):
            return None
        value = loss(entry["candidate"].program, paid,
                     initial_prediction(paid, self.lab.spec.risk_quantile), self.lab.spec.risk_quantile)
        entry.update(training_loss=value, validation_loss=value, selection_basis="all_acquired_evidence")
        return entry, (), paid, list(paid), [entry]

    def _structure_view(self, view, parent, parent_channels, *, remaining_views=1):
        if not self.paired_for(view):
            if self._acquire(view, "prepare_object_structure",
                             reserve_channels=self._cover({view, tuple(parent_channels)})) is None:
                return None
        # Dependency and calibration probes are useful fitting evidence, but
        # their last one- or two-step reset is not a response holdout. Compare
        # on a new varied trajectory when affordable; otherwise select from
        # all paid evidence and leave support to the reserved confirmation.
        training = [row for batch in self.paired_for(view) for _, row in batch]
        validation = self._acquire(view, "compare_on_new_trajectory",
            reserve_channels=self._cover({view, tuple(parent_channels)})) or []
        if not training:
            return None
        cover = self._cover({view, tuple(parent_channels)})
        comparison_rows = sum(len(batch) for batch in self.paired_for(cover)) if parent else 0
        available = max(0, self.budget.remaining - 2 * self.confirmation_length - 2 * comparison_rows)
        entry_cost = len(training) * (len(self.agent._spec(self.lab, view).feature_names) + 2) + len(validation)
        local = SearchBudget(min(90000, available, max(entry_cost, available // remaining_views)))
        response_parent = parent if tuple(parent_channels) == view else None
        if parent is not None and not parent_channels and response_parent is None:
            # A command law can still describe the measured response after
            # calibration changes. Recover its input contract instead of
            # discarding its response composition when changing coordinates.
            spec = self.agent._spec(self.lab, view)
            prepared = measured_relation_entry(parent, spec, self.agent.library)
            if prepared is not None:
                response_parent = bind_relation_entry(prepared,
                    {name: Program("feature", name=name) for name in spec.feature_names})
        claims = self.agent._research_view(self.agent._spec(self.lab, view), training, validation,
            local, self.actions, self.transfer, self.graph,
            parent=response_parent,
            transitions=self.transitions_for(view, training),
            observation_remaining=self.remaining()["observation_remaining"],
            measurement_remaining=self.remaining()["measurement_remaining"])
        self.budget.spend(local.spent)
        return self.agent._best(claims), view, training, validation, claims

    def _structure(self, first_view, parent, parent_channels):
        views = [first_view, *(view for view in sorted(self.views, key=self._rate)
            if view and view != first_view and self.agent._spec(self.lab, view).context.get("object_schema"))]
        best = None
        for index, view in enumerate(views):
            proposal = self._structure_view(view, parent, parent_channels,
                                            remaining_views=len(views) - index)
            entry = proposal[0] if proposal else None
            if entry is None:
                continue
            if best is None or entry["validation_loss"] < best[0]["validation_loss"]:
                best = proposal
            if max(entry["training_loss"], entry["validation_loss"]) <= self.agent.config.minimum_improvement:
                return proposal
        return best

    def _compare_paid(self, proposal, parent, parent_channels):
        """Price a new confirmation only after comparing the paid counterexamples.

        These rows select a proposal; they cannot certify its generalization.
        A changed instrument view must meet the incumbent on common measured
        trajectories, rather than winning by using a different validation set.
        """
        entry, view = proposal[:2]
        cover = self._cover({tuple(view), tuple(parent_channels)})
        batches = self.paired_for(cover) if cover else [
            [(row, row) for row in batch] for batch in self.command_batches()]
        common = [pair for batch in batches for pair in batch]
        if not common or not self.budget.spend(2 * len(common)):
            return {"status": "comparison_unavailable", "rows": len(common)}
        measured = [row for _, row in common]

        def rows_for(channels):
            return (self.view_rows(measured, cover, tuple(channels)) if channels
                    else [row for row, _ in common])

        prior_rows, new_rows = rows_for(parent_channels), rows_for(view)
        fallback = initial_prediction(prior_rows, self.lab.spec.risk_quantile)
        before = loss(parent["candidate"].program, prior_rows, fallback, self.lab.spec.risk_quantile)
        after = loss(entry["candidate"].program, new_rows, fallback, self.lab.spec.risk_quantile)
        improved = after + self.agent.config.minimum_improvement < before
        return {"status": "improved" if improved else "no_improvement", "rows": len(common),
                "before_loss": before, "after_loss": after}

    def propose(self, parent, channels=(), failed_probe=None):
        """Execute one viable direction; skip exhausted proposals without fake returns."""
        channels = tuple(channels)
        result = {"handled": True, "selected": None, "channels": channels,
                  "training": [], "validation": [], "finalists": [], "confirmed": False,
                  "counterexamples": []}
        self.counterexamples = result["counterexamples"]
        if self.pending is not None:
            self.settle(None, None, channels, selected_channels=channels, confirmed=False)
            self.finish()
            self.pending = None
        while len(self.attempted) < 3:
            before = self.remaining()
            spec = self.agent._spec(self.lab, channels)
            training, validation = self._data(channels)
            paid = [*training, *(row for row in validation if row not in training)]
            calls = list(dict.fromkeys([*([parent["candidate"].program] if parent else []),
                                       *self.agent._recall(spec)]))
            definitions = {}
            relational = [call for call in calls if _has_relation(call, self.budget, definitions)]
            measured_views = [view for view in self.views if view and
                             self.agent._spec(self.lab, view).context.get("object_schema")]
            failed_rows = next((batch for batch in reversed(self.command_batches())
                if failed_probe is not None and len(batch) == failed_probe.cost and
                all(row.features == point for row, point in zip(batch, failed_probe.sequence))), [])
            self.counterexamples.extend(row for row in failed_rows if row not in self.counterexamples)
            measured_contract = (measured_relation_entry(parent, self.agent._spec(self.lab, measured_views[-1]),
                self.agent.library) if parent and measured_views and not channels else None)
            context = "object_recovery:" + ("relation_ready" if relational and len(paid) >= 6 else "no_relation")
            context += ":" + self.field_status + (":object_schema" if measured_views else ":no_schema")
            self.policy.set_condition(context)
            options = []
            confirmation_cost = self.confirmation_length * self._rate(channels)
            if relational and len(paid) >= 6:
                options.append(ActionOption("revise_object_relation", self.confirmation_length,
                    4 * len(paid) + 2 * self.confirmation_length, measurement_cost=confirmation_cost))
            if (measured_contract is not None and failed_rows and self.agent._learn_instruments
                    and self.agent._revise_inputs):
                options.append(ActionOption("remeasure_object_inputs", len(failed_rows) + self.confirmation_length,
                    3 * len(failed_rows) + 2 * self.confirmation_length,
                    measurement_cost=len(failed_rows) * self._rate(measured_views[-1])))
            available_views = []
            for view in measured_views:
                missing = int(not self.paired_for(view))
                row_count = sum(len(batch) for batch in self.paired_for(view)) or self.length
                compute = row_count * (len(self.agent._spec(self.lab, view).feature_names) + 2) + self.length + 2 * self.confirmation_length
                confirm_view = self._cover({view, channels})
                quote = ActionOption("research_object_structure", missing * self.length + self.confirmation_length,
                    compute, payload=(view, {}), measurement_cost=missing * self.length * self._rate(view)
                    + self.confirmation_length * self._rate(confirm_view))
                if all(getattr(quote, name + "_cost") <= self.remaining()[name + "_remaining"]
                       for name in ("observation", "measurement", "compute")):
                    available_views.append(quote)
            if available_views:
                options.append(min(available_views, key=lambda option: (option.measurement_cost, option.compute_cost)))
            options = [option for option in options if option.name not in self.attempted]
            resources = self.remaining()
            eligible = [option for option in options if all(getattr(option, name + "_cost") <= resources[name + "_remaining"]
                        for name in ("observation", "measurement", "compute"))]
            chosen = self.policy.choose(options, **resources,
                minimum_improvement=self.agent.config.minimum_improvement) if self.learn else None
            decision = dict(self.policy.last_decision) if self.learn else {
                "context": self.policy.context, "eligible_actions": [option.name for option in eligible]}
            if chosen is None and eligible:
                chosen = eligible[0]
                decision.update(selected_action=chosen.name,
                    reason="current_counterexample_requires_research" if self.learn else "fixed_eligible_order")
            self.actions.append({"action": "choose_object_research", "domain": "object_recovery", **decision,
                "options": [{"name": option.name, "observation_cost": option.observation_cost,
                    "measurement_cost": option.measurement_cost, "compute_cost": option.compute_cost} for option in options],
                "relation_sources": len(relational), "paid_rows": len(paid)})
            if chosen is None:
                break
            self.attempted.add(chosen.name)
            result["handled"] = True
            baseline = parent["candidate"].program if parent else Program("const", value=initial_prediction(paid, spec.risk_quantile))
            origin = {"context": self.policy.context, "action": chosen.name, "trial": None,
                "baseline_program": baseline.to_dict(), "baseline_channels": list(channels),
                "baseline_fallback": initial_prediction(paid, spec.risk_quantile),
                "start_resources": before, "charged_resources": before, "status": "pending"}
            if chosen.name == "revise_object_relation":
                proposal = self._relation(channels, relational)
            elif chosen.name == "remeasure_object_inputs":
                proposal = self._inputs(parent, failed_probe, failed_rows)
            else:
                proposal = self._structure(chosen.payload[0], parent, channels)
            entry = proposal[0] if proposal else None
            comparison = self._compare_paid(proposal, parent, channels) if entry is not None and parent else None
            if comparison is not None and comparison["status"] != "improved":
                entry = None
            proposal_costs = self._costs(before)
            unscored = (self.policy.record_unscored_attempt(chosen.name, context=origin["context"],
                status="no_candidate", **proposal_costs) if self.learn and entry is None else None)
            self.actions.append({"action": "propose_object_research", "domain": "object_recovery", "direction": chosen.name,
                "context": origin["context"], "unscored_feedback": unscored,
                "selected_kind": entry.get("kind") if entry else None, "status": "requires_fresh_confirmation" if entry else "no_candidate",
                "field_status": self.field_status, "paid_comparison": comparison, **proposal_costs})
            if entry is not None:
                self.pending = origin
                entry["object_action_origin"] = origin
                _, view, train, check, finalists = proposal
                return {**result, "selected": entry, "channels": view, "training": train,
                        "validation": check, "finalists": finalists, "object_action_origin": origin}
        return result

    def settle(self, selected, rows, acquired_channels, *, selected_channels, confirmed,
               fallback=None, probe_id=None):
        """Charge the complete visit and compare both frozen programs on these new rows."""
        origin = self.pending
        if origin is None or (not rows and origin["status"] != "pending"):
            return None
        status = ("supported" if confirmed else "refuted") if rows else "unresolved"
        before_loss = after_loss = None
        selected_fallback = fallback if fallback is not None else origin["baseline_fallback"]
        if rows and selected is not None and self.budget.spend(2 * len(rows)):
            baseline_rows = self.view_rows(rows, acquired_channels, tuple(origin["baseline_channels"]))
            selected_rows = self.view_rows(rows, acquired_channels, tuple(selected_channels))
            before_loss = loss(Program.from_dict(origin["baseline_program"]), baseline_rows,
                               origin["baseline_fallback"], self.lab.spec.risk_quantile)
            after_loss = loss(selected["candidate"].program, selected_rows, selected_fallback,
                              self.lab.spec.risk_quantile)
        costs = self._costs(origin["charged_resources"])
        feedback = None
        if self.learn:
            if origin["trial"] is None and before_loss is not None:
                feedback = self.policy.update(origin["action"], before_loss=before_loss, after_loss=after_loss,
                                              context=origin["context"], **self._costs(origin["start_resources"]))
                origin.update(trial=feedback["trials"], credited_gain=feedback["last_gain"])
            elif origin["trial"] is not None:
                ResearchActionPolicy.charge_followup(self.agent.library.action_experience, origin, **costs)
            if origin["trial"] is not None:
                feedback = ResearchActionPolicy.revise_feedback(self.agent.library.action_experience, origin,
                    before_loss=before_loss, after_loss=after_loss, status=status,
                    evidence_rows=len(rows or []))
        origin["charged_resources"] = self.remaining()
        # Final cleanup must not turn a genuinely refuted visit into mere lack
        # of confirmation. No rows never invent a measured numeric return.
        if rows or origin["status"] == "pending":
            origin["status"] = status
        if rows and status == "refuted" and origin["action"] == "research_object_structure":
            # A fresh counterexample adds fitting evidence to the measured
            # views. Revisit their structure using those paid rows instead of
            # treating the first inadequate fit as an exhausted direction.
            self.attempted.discard("research_object_structure")
        event = {"action": "object_research_feedback", "domain": "object_recovery", "context": origin["context"],
            "direction": origin["action"], "trial": origin["trial"], "status": origin["status"],
            "confirmed": confirmed, "baseline_program": origin["baseline_program"],
            "baseline_channels": origin["baseline_channels"], "baseline_fallback": origin["baseline_fallback"],
            "program": selected["candidate"].program.to_dict() if selected else None,
            "channels": list(selected_channels), "fallback": selected_fallback,
            "risk_quantile": self.lab.spec.risk_quantile, "probe_id": probe_id,
            "before_loss": before_loss, "after_loss": after_loss, "observation_rows": len(rows or []),
            "feedback": feedback, **costs}
        self.actions.append(event)
        return event

    def finish(self):
        """Charge subsequent input replacement and memory work to its owning visit."""
        origin = self.pending
        if origin is None:
            return None
        costs = self._costs(origin["charged_resources"])
        if origin["trial"] is None and not origin.get("unscored_recorded"):
            total = self._costs(origin["start_resources"])
            feedback = (self.policy.record_unscored_attempt(origin["action"], context=origin["context"],
                status="unresolved", **total) if self.learn else None)
            origin["unscored_recorded"] = True
            origin["charged_resources"] = self.remaining()
            event = {"action": "object_research_unscored", "domain": "object_recovery",
                "context": origin["context"], "direction": origin["action"], "trial": None,
                "status": "unresolved", "cost_basis": "whole_unscored_attempt", "feedback": feedback, **total}
            self.actions.append(event)
            return event
        if not any(costs.values()):
            return None
        feedback = (ResearchActionPolicy.charge_followup(self.agent.library.action_experience, origin, **costs)
                    if self.learn and origin["trial"] is not None else None)
        origin["charged_resources"] = self.remaining()
        event = {"action": "object_research_followup_cost", "domain": "object_recovery",
            "context": origin["context"], "direction": origin["action"], "trial": origin["trial"],
            "status": origin["status"], "feedback": feedback, **costs}
        self.actions.append(event)
        return event
