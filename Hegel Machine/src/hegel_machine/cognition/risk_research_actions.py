"""Propose competing revisions of a fixed-risk predictor from purchased evidence."""

from copy import deepcopy
from math import ceil

from .action_policy import ActionOption, ResearchActionPolicy
from .evaluation import initial_prediction, loss
from .language import Program
from .models import Candidate, SearchBudget
from .risk_search import trajectory_blocks
from .risk_research_methods import refit_risk_method, remember_risk_method, replay_risk_methods
from .unified_uncertainty import residual_risk_candidates


class RiskResearchActions:
    """Leave adoption and noise replication to the ordinary frozen confirmation."""

    def __init__(self, agent, laboratory, *, acquire, paired_for, command_batches,
                 view_rows, transitions_for, budget, actions, transfer, graph, remaining, length):
        self.agent, self.lab, self.budget = agent, laboratory, budget
        self.acquire, self.paired_for, self.command_batches = acquire, paired_for, command_batches
        self.view_rows, self.transitions_for = view_rows, transitions_for
        self.actions, self.transfer, self.graph, self.remaining = actions, transfer, graph, remaining
        self.length = min(laboratory.spec.max_probe_length, length)
        self.offered = tuple(channel.name for channel in laboratory.channels)
        self.rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        self.learn = agent.config.use_memory and agent.config.learn_research_policy
        self.policy = ResearchActionPolicy([], agent._spec(laboratory, ()),
            agent.library.action_experience if self.learn else {})
        self.attempted, self.pending, self.comparison_parent = set(), None, None
        self.uncertainty_bases, self.retained_alternatives = [], []
        self.current_method, self.source_draft_id = None, None
        self.method_steps, self.methods_replayed = [], False

    def _rate(self, channels):
        return sum(self.rates[name] for name in channels)

    def _costs(self, before):
        after = self.remaining()
        return {resource + "_cost": before[resource + "_remaining"] - after[resource + "_remaining"]
                for resource in ("observation", "measurement", "compute")}

    def _batches(self, channels):
        return ([[row for _, row in batch] for batch in self.paired_for(channels)]
                if channels else self.command_batches())

    def _preparation(self, channels):
        rows = [row for batch in self._batches(channels) for row in batch]
        missing = max(0, 2 - len(trajectory_blocks(rows)), ceil((32 - len(rows)) / self.length))
        return rows, missing

    def _options(self, channels):
        options = []
        for direction in ("revise_uncertainty", "revise_mechanism", "remeasure_inputs"):
            if direction in self.attempted or (direction == "remeasure_inputs" and not self.offered):
                continue
            view = self.offered if direction == "remeasure_inputs" else channels
            paid, preparation = self._preparation(view)
            preparation = max(preparation, self._preparation(self.offered)[1])
            count = len(paid) + preparation * self.length
            features = len(self.agent._spec(self.lab, view).feature_names)
            fit = count * (19 + 2 * features if direction == "revise_uncertainty" else features + 4)
            options.append(ActionOption(direction, (preparation + 2) * self.length,
                fit + count * 6 + self.length * 16,
                measurement_cost=(preparation + 2) * self.length * self._rate(self.offered)))
        return options

    def _retain_counterexplanation(self, parent, channels, support):
        """Keep the executable explanation already supported on later paid rows."""
        self.retained_alternatives = []
        counterexample = (support or {}).get("structural_counterexample")
        baseline = (support or {}).get("counterexample_baseline_program")
        if not counterexample or baseline is None:
            return parent, channels, False
        pairs = [pair for batch in self.paired_for(self.offered) for pair in batch] if self.offered else []
        rows = [row for _, row in pairs] if self.offered else [row for batch in self.command_batches() for row in batch]
        if not rows or not self.budget.spend(2 * len(rows)):
            return parent, channels, False
        fallback = support["counterexample_fallback"]
        base, correction = Program.from_dict(baseline), Program.from_dict(counterexample["program"])
        # Residual assessment predicts these components separately at reset.
        # Preserve their existing finite-history startup fallbacks in the sum.
        def at_start(program, value):
            depth = program.minimum_history_depth
            return (Program("if", children=(Program("available", lag=depth), program,
                    Program("const", value=value))) if depth else program)
        program = Program("add", children=(at_start(base, fallback), at_start(correction, 0.))).simplified()
        q = self.lab.spec.risk_quantile
        measured = loss(program, rows, fallback, q)
        previous = deepcopy(parent) if parent is not None else {
            "candidate": Candidate("risk:initial", Program("const", value=initial_prediction(rows, q)), "initial"),
            "kind": "initial", "family_id": "T02"}
        previous.pop("research_action_origin", None)
        previous.pop("risk_action_origin", None)
        previous_rows = ([command for command, _ in pairs] if self.offered and not channels
                         else self.view_rows(rows, self.offered, channels))
        previous.setdefault("fallback", initial_prediction(previous_rows, q))
        previous_loss = loss(previous["candidate"].program, previous_rows, previous["fallback"], q)
        previous.update(training_loss=previous_loss, validation_loss=previous_loss,
                        selection_basis="all_acquired_evidence")
        retained = {"candidate": Candidate("risk:retained_counterexplanation", program,
                        "retained_mechanism", "Reuse the independently supported residual explanation"),
            "kind": "retained_risk_counterexplanation", "family_id": previous.get("family_id", "T02"),
            "counterexplanation_kind": counterexample["kind"], "fallback": fallback,
            "training_loss": measured, "validation_loss": measured,
            "selection_basis": "all_acquired_evidence"}
        improved = measured + self.agent.config.minimum_improvement < previous_loss
        self.actions.append({"action": "retain_risk_counterexplanation", "domain": "risk_recovery",
            "program": program.to_dict(), "channels": list(self.offered), "fallback": fallback,
            "baseline_program": baseline, "correction": counterexample["program"],
            "previous_program": previous["candidate"].program.to_dict(),
            "previous_channels": list(channels), "previous_fallback": previous["fallback"],
            "support_gain": counterexample["gain"], "support_margin": counterexample["margin"],
            "previous_loss": previous_loss, "retained_loss": measured,
            "selected": improved, "observation_rows": len(rows), "compute_cost": 2 * len(rows),
            "selection_basis": "all_acquired_evidence", "confirmed": False})
        if improved:
            if previous["candidate"].program.required_features() <= set(rows[0].features):
                self.retained_alternatives = [previous]
            return retained, self.offered, program != previous["candidate"].program or self.offered != channels
        if channels == self.offered:
            self.retained_alternatives = [retained]
        return previous, channels, False

    def confirmation_channels(self, selected_channels):
        # A noise explanation must still face the other publicly offered
        # inputs on this very trajectory, even when its program reads fewer.
        return self.offered

    def _prepare(self, channels):
        _, fit_missing = self._preparation(channels)
        _, field_missing = self._preparation(self.offered)
        missing = max(fit_missing, field_missing)
        for _ in range(missing):
            available = self.remaining()
            if (available["observation_remaining"] < 3 * self.length or
                    available["measurement_remaining"] < 3 * self.length * self._rate(self.offered)):
                return None
            # These same paired rows support a partial predictor and the
            # subsequent challenge from all publicly available input fields.
            if self.acquire(self.offered, "prepare_risk_revision", rows_count=self.length) is None:
                return None
        rows, missing = self._preparation(channels)
        return rows if not missing else None

    def _fit(self, direction, parent, channels):
        view = self.offered if direction == "remeasure_inputs" else channels
        rows = self._prepare(view)
        if not rows:
            return None
        spec, q = self.agent._spec(self.lab, view), self.lab.spec.risk_quantile
        fallback = parent.get("fallback", initial_prediction(rows, q)) if view == channels else initial_prediction(rows, q)
        # Keep the later support comparison, fixed-objective paired score and
        # possible repeat funded; fitting cannot consume their search budget.
        reserve = len(rows) * 6 + self.length * 16
        available = max(0, self.budget.remaining - reserve)
        cap = 12000 if direction == "revise_uncertainty" else 90000
        entry_cost = len(rows) * (19 + 2 * len(spec.feature_names)
            if direction == "revise_uncertainty" else len(spec.feature_names) + 4)
        local = SearchBudget(min(cap, available, max(entry_cost, available // 2)))
        if direction == "revise_uncertainty":
            original = deepcopy(parent)
            original.pop("research_action_origin", None)
            fitting = SearchBudget(max(0, local.remaining - 4 * len(rows)))
            learned = (refit_risk_method(self.current_method, spec, rows, fitting,
                mechanism=original["candidate"].program) if self.current_method is not None else None)
            claims = ([learned] if learned is not None else residual_risk_candidates(
                spec, original["candidate"].program, rows, fitting, fallback=fallback))
            for claim in claims:
                claim["fallback"] = fallback
            local.spend(fitting.spent)
            if local.spend(len(rows)):
                original["training_loss"] = loss(original["candidate"].program, rows, fallback, q)
                original["validation_loss"] = original["training_loss"]
                original["selection_basis"] = "all_acquired_evidence"
                claims.append(original)
            claims.extend(self.retained_alternatives)
            checked = [item for claim in claims for item in self.agent._assess(
                [claim], rows, local, claim.get("fallback", fallback), q)]
            self.comparison_parent = original if original in checked else None
        else:
            if direction == "remeasure_inputs" and self.agent._learn_instruments:
                self.agent._fit_instruments(self.agent._spec(self.lab, ()), spec,
                    self.paired_for(view), local, self.actions)
            checked = self.agent._research_view(spec, rows, [], local, self.actions,
                self.transfer, self.graph, parent=parent if view == channels else None,
                transitions=self.transitions_for(view, rows), include_risk_revisions=False,
                observation_remaining=self.remaining()["observation_remaining"],
                measurement_remaining=self.remaining()["measurement_remaining"])
        self.budget.spend(local.spent)
        for claim in checked:
            claim["selection_basis"] = "all_acquired_evidence"
        chosen = self.agent._best(checked)
        if chosen is not None:
            chosen["risk_method"] = {**deepcopy(chosen.get("risk_method") or {}),
                "mechanism_program": (parent["candidate"].program.to_dict()
                    if direction == "revise_uncertainty" else chosen["candidate"].program.to_dict()),
                "uncertainty_kind": (self.current_method.get("uncertainty_kind", "global")
                    if learned is not None else "absolute" if "absolute_scale" in chosen["kind"]
                    else "quadratic" if "quadratic_scale" in chosen["kind"] else "global")
                    if direction == "revise_uncertainty" else "global",
                "response_program": chosen["candidate"].program.to_dict(), "steps": list(self.method_steps)}
        return (chosen, view, rows, list(rows), checked) if chosen else None

    def _reuse_method(self, parent, channels):
        """Compare executable methods on later paid rows before changing a plan."""
        if not self.learn or self.methods_replayed or self.current_method is not None:
            return parent, channels
        self.methods_replayed = True
        rows = [row for batch in self._batches(self.offered) for row in batch]
        blocks = trajectory_blocks(rows)
        if len(blocks) < 2 or len(rows) < 32:
            return parent, channels
        training, validation = [row for block in blocks[:-1] for row in block], blocks[-1]
        local = SearchBudget(min(12000, max(0, self.budget.remaining - 6 * len(rows) - 16 * self.length)))
        checked = replay_risk_methods(self.agent, self.agent._spec(self.lab, self.offered),
            training, validation, local, self.actions)
        self.budget.spend(local.spent)
        if not checked or not self.budget.spend(len(validation)):
            return parent, channels
        selected, source = min(checked, key=lambda item: (item[0]["validation_loss"], item[0]["candidate"].program.size))
        self.current_method, self.source_draft_id = deepcopy(selected["risk_method"]), source["draft_id"]
        old_rows = self.view_rows(validation, self.offered, channels)
        fallback = parent.get("fallback", initial_prediction(old_rows, self.lab.spec.risk_quantile))
        before = loss(parent["candidate"].program, old_rows, fallback, self.lab.spec.risk_quantile)
        improved = selected["validation_loss"] + self.agent.config.minimum_improvement < before
        self.actions.append({"action": "choose_risk_method_on_paid_response", "draft_id": source["draft_id"],
            "before_loss": before, "after_loss": selected["validation_loss"], "selected": improved,
            "observation_rows": len(validation), "fixed_quantile": self.lab.spec.risk_quantile,
            "compute_cost": len(validation), "requires_fresh_confirmation": True})
        return (selected, self.offered) if improved else (parent, channels)

    def _method_support(self, parent, channels, support):
        if (parent is None or self.current_method is None or parent.get("risk_method") is None
                or (support or {}).get("counterexample_baseline_program") == parent["candidate"].program.to_dict()):
            return support
        rows = [row for batch in self._batches(channels) for row in batch]
        blocks = trajectory_blocks(rows)
        if len(blocks) < 2:
            return {"adequate": False, "reason": "insufficient_calibration_rows"}
        training, validation = [row for block in blocks[:-1] for row in block], blocks[-1]
        local = SearchBudget(min(24000, max(0, self.budget.remaining // 4 - 6 * len(rows) - 16 * self.length)))
        assessed = self.agent._support(parent["candidate"].program, training, validation,
            self.agent._spec(self.lab, channels), local)
        self.budget.spend(local.spent)
        self.actions.append({"action": "assess_replayed_risk_method", "draft_id": self.source_draft_id,
            "reason": assessed["reason"], "adequate": assessed["adequate"],
            "training_rows": len(training), "validation_rows": len(validation),
            "fixed_quantile": self.lab.spec.risk_quantile, "compute_cost": local.spent})
        return assessed

    def _rewrite_method(self, support, retained_change, previous):
        if self.current_method is None:
            return None
        old = list(self.current_method["steps"])
        reason = (support or {}).get("reason", "unassessed_residual")
        if retained_change:
            steps = ["retain_supported_residual", "revise_uncertainty"]
        elif reason == "quantile_miscalibration":
            steps = ["revise_uncertainty"]
        elif reason in {"predictable_residual", "repeatable_unexplained_response"}:
            steps = ["revise_mechanism", "revise_uncertainty"]
        else:
            steps = old
        counterevidence = (reason in {"quantile_miscalibration", "predictable_residual",
            "repeatable_unexplained_response"} or bool((support or {}).get("structural_counterexample")))
        if (previous and previous["status"] == "refuted" and previous["action"] == "revise_uncertainty"
                and counterevidence):
            # New independent counterevidence rejects this particular tail
            # family. The next fit can propose other current residual families.
            self.current_method = None
            if reason == "quantile_miscalibration":
                self.attempted.discard("revise_uncertainty")
            self.actions.append({"action": "revise_risk_research_method", "draft_id": self.source_draft_id,
                "before_steps": old, "after_steps": steps, "reason": "fresh_refuted_recorded_tail",
                "evidence_condition": reason, "fixed_quantile": self.lab.spec.risk_quantile,
                "refuted_probe_status": previous["status"]})
            return next((step for step in steps if step in {
                "revise_uncertainty", "revise_mechanism", "remeasure_inputs"}), None)
        self.current_method["steps"] = steps
        if old != steps:
            self.actions.append({"action": "revise_risk_research_method", "draft_id": self.source_draft_id,
                "before_steps": old, "after_steps": steps, "reason": reason,
                "fixed_quantile": self.lab.spec.risk_quantile})
        return next((step for step in steps if step in {"revise_uncertainty", "revise_mechanism", "remeasure_inputs"}
                     and step not in self.attempted), None)

    def _charge_method(self, origin, costs):
        identifier = origin.get("saved_method_id")
        if identifier is not None:
            record = next(item for item in self.agent.library.research_drafts if item["draft_id"] == identifier)
            for key, value in costs.items():
                record["outcomes"][-1]["whole_visit_costs"][key] += value

    def _unscored(self, origin, status):
        if origin.get("unscored_recorded"):
            return None
        costs = self._costs(origin["start_resources"])
        feedback = (self.policy.record_unscored_attempt(origin["action"], status=status,
                    context=origin["context"], **costs) if self.learn else None)
        origin.update(unscored_recorded=True, charged_resources=self.remaining(), status=status)
        event = {"action": "risk_research_unscored", "domain": "risk_recovery",
            "context": origin["context"], "direction": origin["action"], "status": status,
            "feedback": feedback, **costs}
        self.actions.append(event)
        self._charge_method(origin, costs)
        return event

    def propose(self, parent, channels=(), failed_probe=None, noise_assessment=None):
        """Choose by public prerequisites; no residual fit is a noise certificate."""
        channels = tuple(channels)
        previous_origin = self.pending
        if parent is not None and parent.get("risk_method") is not None and self.current_method is None:
            source = next((item for item in self.agent.library.research_drafts
                if item["draft_id"] == parent.get("research_draft_id")
                and item.get("source_seed") != self.agent.config.seed), None)
            if source is not None:
                self.current_method = deepcopy(parent["risk_method"])
                self.source_draft_id = source["draft_id"]
        changed_base = (self.pending is not None and parent is not None
            and self.pending["action"] in {"revise_mechanism", "remeasure_inputs"}
            and (self.pending["baseline_program"] != parent["candidate"].program.to_dict()
                 or tuple(self.pending["baseline_channels"]) != channels))
        if self.pending is not None:
            self.finish()
        self.pending, self.comparison_parent = None, None
        preparation_start = self.remaining()
        noise_assessment = self._method_support(parent, channels, noise_assessment)
        parent, channels, retained_change = self._retain_counterexplanation(parent, channels, noise_assessment)
        if retained_change:
            self.method_steps.append("retain_supported_residual")
        if parent is not None:
            old_parent = parent
            parent, channels = self._reuse_method(parent, channels)
            if parent is not old_parent:
                noise_assessment = self._method_support(parent, channels, noise_assessment)
        preferred = self._rewrite_method(noise_assessment, retained_change, previous_origin)
        if parent is not None and (changed_base or retained_change):
            basis = (parent["candidate"].program, channels)
            if basis not in self.uncertainty_bases:
                self.attempted.discard("revise_uncertainty")
        result = {"handled": True, "selected": None, "channels": channels, "training": [],
                  "validation": [], "finalists": [], "confirmed": False, "counterexamples": []}
        if retained_change:
            paid_rows = [row for batch in self._batches(channels) for row in batch]
            result.update(selected=parent, training=paid_rows, validation=list(paid_rows), finalists=[parent])
        while len(self.attempted) < 3:
            before, preparation_start = preparation_start or self.remaining(), None
            rows, missing = self._preparation(channels)
            spec = self.agent._spec(self.lab, channels)
            fallback = parent.get("fallback", initial_prediction(rows, spec.risk_quantile)) if parent else initial_prediction(rows, spec.risk_quantile)
            if parent is None:
                parent = {"candidate": Candidate("risk:initial", Program("const", value=fallback), "initial"),
                    "kind": "initial", "family_id": "T02", "training_loss": None, "validation_loss": None}
            reason = (noise_assessment or {}).get("reason", "unassessed_residual")
            self.policy.set_condition("risk_recovery:" + reason
                + (":calibration_ready" if not missing else ":needs_calibration_rows")
                + (":all_inputs" if channels == self.offered else ":partial_inputs"))
            options = self._options(channels)
            available = self.remaining()
            eligible = [option for option in options if all(getattr(option, name + "_cost") <= available[name + "_remaining"]
                        for name in ("observation", "measurement", "compute"))]
            choice = self.policy.choose(options, **available,
                minimum_improvement=self.agent.config.minimum_improvement) if self.learn else None
            decision = dict(self.policy.last_decision) if self.learn else {
                "context": self.policy.context, "eligible_actions": [option.name for option in eligible]}
            if choice is None and eligible:
                choice = eligible[0]
                decision.update(selected_action=choice.name,
                    reason="current_evidence_requires_research" if self.learn else "fixed_eligible_order")
            method_choice = next((option for option in eligible if option.name == preferred), None)
            if method_choice is not None:
                choice = method_choice
                decision.update(selected_action=choice.name, reason="execute_revised_paid_risk_method",
                                method_draft_id=self.source_draft_id)
            self.actions.append({"action": "choose_risk_research", "domain": "risk_recovery", **decision,
                "options": [{"name": option.name, "observation_cost": option.observation_cost,
                    "measurement_cost": option.measurement_cost, "compute_cost": option.compute_cost} for option in options],
                "paid_rows": len(rows), "training_trajectories": len(trajectory_blocks(rows))})
            if choice is None:
                break
            self.attempted.add(choice.name)
            self.method_steps.append(choice.name)
            if choice.name == "revise_uncertainty":
                self.uncertainty_bases.append((parent["candidate"].program, channels))
            origin = {"context": self.policy.context, "action": choice.name, "trial": None,
                "baseline_program": parent["candidate"].program.to_dict(), "baseline_channels": list(channels),
                "baseline_fallback": fallback, "start_resources": before, "charged_resources": before,
                "status": "pending", "source_seed": self.agent.config.seed,
                "method_steps": list(self.method_steps), "source_draft_id": self.source_draft_id,
                "evidence_condition": reason}
            proposal = self._fit(choice.name, parent, channels)
            selected = proposal[0] if proposal else None
            self.actions.append({"action": "propose_risk_research", "domain": "risk_recovery",
                "direction": choice.name, "selected_kind": selected["kind"] if selected else None,
                "status": "requires_fresh_confirmation" if selected else "no_candidate", **self._costs(before)})
            if selected is None:
                self._unscored(origin, "no_candidate")
                continue
            self.pending = origin
            selected["risk_action_origin"] = origin
            _, view, training, validation, finalists = proposal
            return {**result, "selected": selected, "channels": view, "training": training,
                "validation": validation, "finalists": finalists, "risk_action_origin": origin,
                "comparison_parent": self.comparison_parent}
        return result

    def settle(self, selected, rows, acquired_channels, *, selected_channels, confirmed,
               fallback=None, probe_id=None):
        origin = self.pending
        if origin is None or not rows:
            return None
        status, before_loss, after_loss = ("supported" if confirmed else "refuted"), None, None
        selected_fallback = origin["baseline_fallback"] if fallback is None else fallback
        if self.budget.spend(2 * len(rows)):
            before_loss = loss(Program.from_dict(origin["baseline_program"]),
                self.view_rows(rows, acquired_channels, tuple(origin["baseline_channels"])),
                origin["baseline_fallback"], self.lab.spec.risk_quantile)
            after_loss = loss(selected["candidate"].program,
                self.view_rows(rows, acquired_channels, tuple(selected_channels)),
                selected_fallback, self.lab.spec.risk_quantile)
        costs, feedback = self._costs(origin["charged_resources"]), None
        if self.learn:
            if origin["trial"] is None and before_loss is not None:
                feedback = self.policy.update(origin["action"], context=origin["context"],
                    before_loss=before_loss, after_loss=after_loss, **self._costs(origin["start_resources"]))
                origin.update(trial=feedback["trials"], credited_gain=feedback["last_gain"])
            elif origin["trial"] is not None:
                ResearchActionPolicy.charge_followup(self.agent.library.action_experience, origin, **costs)
            if origin["trial"] is not None:
                feedback = ResearchActionPolicy.revise_feedback(self.agent.library.action_experience, origin,
                    status=status, before_loss=before_loss, after_loss=after_loss, evidence_rows=len(rows))
        origin.update(status=status, charged_resources=self.remaining(),
                      paired_return=origin.get("paired_return", False) or before_loss is not None)
        event = {"action": "risk_research_feedback", "domain": "risk_recovery", "context": origin["context"],
            "direction": origin["action"], "trial": origin["trial"], "status": status, "confirmed": confirmed,
            "baseline_program": origin["baseline_program"], "baseline_channels": origin["baseline_channels"],
            "baseline_fallback": origin["baseline_fallback"], "program": selected["candidate"].program.to_dict(),
            "channels": list(selected_channels), "fallback": selected_fallback, "risk_quantile": self.lab.spec.risk_quantile,
            "probe_id": probe_id, "before_loss": before_loss, "after_loss": after_loss,
            "observation_rows": len(rows), "feedback": feedback, **costs}
        self.actions.append(event)
        if self.learn and before_loss is not None:
            comparison = {key: deepcopy(event[key]) for key in ("probe_id", "before_loss", "after_loss",
                "observation_rows", "risk_quantile", "baseline_program", "baseline_fallback", "program", "fallback")}
            record = remember_risk_method(self.agent.library, self.agent._spec(self.lab, selected_channels),
                selected, origin, self._costs(origin["start_resources"]), comparison, confirmed)
            origin["saved_method_id"] = record["draft_id"]
            self.actions.append({"action": "preserve_research_draft", "domain": "risk_method",
                "draft_id": record["draft_id"], "parent_draft_id": record["parent_draft_id"],
                "source_task_id": record["source_task_id"], "status": record["status"],
                "steps": list(record["risk_method"]["steps"]), "fixed_quantile": self.lab.spec.risk_quantile})
        return event

    def finish(self):
        origin = self.pending
        if origin is None:
            return None
        if origin["trial"] is None and not origin.get("paired_return"):
            return self._unscored(origin, "unresolved")
        costs = self._costs(origin["charged_resources"])
        if not any(costs.values()):
            return None
        feedback = (ResearchActionPolicy.charge_followup(self.agent.library.action_experience, origin, **costs)
                    if self.learn and origin["trial"] is not None else None)
        origin["charged_resources"] = self.remaining()
        event = {"action": "risk_research_followup_cost", "domain": "risk_recovery", "context": origin["context"],
            "direction": origin["action"], "trial": origin["trial"], "status": origin["status"],
            "feedback": feedback, **costs}
        self.actions.append(event)
        self._charge_method(origin, costs)
        return event
