"""Continue paid inquiry tasks through the existing native session scheduler."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import json

from .language import Program
from .legacy_research_bridge import (
    InquiryResearchLab, LEGACY_KNOWLEDGE_PATH, METHODS_PATH, _paid_question,
)
from .memory import KnowledgeLibrary
from .models import Probe, ResearchConfig
from .unified_session import UnifiedResearchSession


SESSION_MODEL_PATH = "shared/legacy_session_model.py"
SESSION_REPORT_PATH = "legacy_research/session_report.json"


class _PaidInquirySession(UnifiedResearchSession):
    """Keep native choices within the actual inquiry operators' remaining caps."""

    def __init__(self, bridge, **kwargs):
        self.bridge = bridge
        super().__init__(**kwargs)

    def _remaining(self):
        remaining = super()._remaining()
        # Screening belongs to the controlling Agent, rather than being free
        # host work. Its unbilled native costs are included while a run executes.
        position = self.state["inquiry_bridge"]["controller_position"]
        remaining["compute_cost"] = min(remaining["compute_cost"],
            self.bridge._allowance(position)["compute_cost"])
        return remaining

    def _revisit_allocation(self, offer, pending, remaining):
        allowed = self.bridge._allowance(offer["task_position"])
        bounded = {key: min(value, allowed[key]) for key, value in remaining.items()}
        return super()._revisit_allocation(offer, pending, bounded)

    def _perform_revisit(self, offer, *, allocation_limits=None):
        if allocation_limits is not None:
            allowed = self.bridge._allowance(offer["task_position"])
            allocation_limits = {key: min(value, allowed[key])
                                 for key, value in allocation_limits.items()}
        return super()._perform_revisit(offer, allocation_limits=allocation_limits)


class LegacyInquirySession:
    """Join existing Agent evidence and knowledge, without replaying purchases.

    Every episode supplies ``agent`` and its public ``laboratory``. ``research``
    is a real native report, or None to import that Agent's paid observations as
    an unresolved question. ``available_concepts`` records what that earlier
    visit actually knew. The caller restores the same paid Agents when loading
    a JSON state; this object does not recreate apparatuses from hidden truth.

    Global screening is charged to controller_position. Each native revisit is
    charged to its own Agent. Existing independent Agent finish confirmations
    remain reserved and are never implied by a native completed status.
    """

    def __init__(self, episodes, knowledge=None, *, config=None, state=None,
                 controller_position=0):
        if not episodes:
            raise ValueError("a legacy inquiry session needs paid episodes")
        self.episodes = list(episodes)
        if state is not None:
            if len(state["tasks"]) != len(episodes):
                raise ValueError("restore the paid agents for every saved episode")
            self.session = _PaidInquirySession(self, state=state, lab_factory=self._laboratory)
        else:
            library = (KnowledgeLibrary.from_dict(knowledge) if isinstance(knowledge, dict)
                       else knowledge if knowledge is not None else KnowledgeLibrary())
            config = config or ResearchConfig(observation_budget=384, compute_budget=360000,
                minimum_improvement=1e-7, max_rounds=4, active_probes=True)
            tasks, reports = [], []
            for position, episode in enumerate(self.episodes):
                agent = episode["agent"]
                adapter = InquiryResearchLab(agent, episode["laboratory"])
                report = deepcopy(episode.get("research"))
                if report is None:
                    report = _paid_question(agent, adapter)
                if report is None:
                    raise ValueError("buy evidence before registering an unfinished inquiry")
                tasks.append({"episode_position": position,
                    "seed": episode.get("seed", config.seed + position),
                    "research_limits": {
                        "observation_budget": agent.config.observation_budget - agent.config.confirmation_rows,
                        "compute_budget": agent.config.compute_budget,
                        "measurement_budget": episode.get("measurement_budget",
                            agent.config.observation_budget * sum(c.cost_per_row for c in adapter.channels))}})
                reports.append({"task_position": position,
                    "available_concepts": list(episode.get("available_concepts", ())), "research": report})
            self.session = _PaidInquirySession(self, config=config, tasks=tasks,
                library=library, lab_factory=self._laboratory)
            self.session.state["results"] = reports
            self.session.state["inquiry_bridge"] = {
                "controller_position": controller_position,
                "charged_screening_compute": 0,
                "charged_research_compute": [row["research"]["costs"]["compute_cost"] for row in reports]}
        for episode in self.episodes:
            episode["agent"].legacy_session = self
            from .temporal_operator_agent import LEGACY_SESSION_INTERFACE
            if LEGACY_SESSION_INTERFACE not in episode["agent"].common_interface:
                episode["agent"].common_interface += "\n" + LEGACY_SESSION_INTERFACE
        self._persist()

    def _laboratory(self, task):
        episode = self.episodes[task["episode_position"]]
        return nullcontext(InquiryResearchLab(episode["agent"], episode["laboratory"]))

    def _unbilled_compute(self, position):
        state = self.session.state
        bridge = state["inquiry_bridge"]
        amount = (state["results"][position]["research"]["costs"]["compute_cost"]
                  - bridge["charged_research_compute"][position])
        if position == bridge["controller_position"]:
            amount += state["screening_compute_cost"] - bridge["charged_screening_compute"]
        return amount

    def _allowance(self, position):
        agent = self.episodes[position]["agent"]
        reserve = agent.config.confirmation_rows
        rows = max(0, agent.config.observation_budget - agent.costs["observation_cost"] - reserve)
        compute = max(0, agent._remaining_compute() - self._unbilled_compute(position)
                      - len(agent.rows) - rows - reserve)
        limit = self.session._task_limits(position)["measurement_budget"]
        return {"observation_cost": rows, "measurement_cost": max(0,
            limit - agent.costs["measurement_cost"]), "compute_cost": compute}

    def _settle(self):
        state = self.session.state
        for position, episode in enumerate(self.episodes):
            episode["agent"]._charge(self._unbilled_compute(position))
        state["inquiry_bridge"]["charged_research_compute"] = [
            row["research"]["costs"]["compute_cost"] for row in state["results"]]
        state["inquiry_bridge"]["charged_screening_compute"] = state["screening_compute_cost"]

    def sync_paid_observations(self, position=None):
        """Import ordinary Agent queries before entering the native researcher.

        The public adapter produces one native batch per trajectory record.
        Native acquisitions already append both records; comparing their counts
        here imports only ordinary queries made between native calls.
        """
        state = self.session.state
        positions = range(len(self.episodes)) if position is None else (position,)
        for index in positions:
            episode = self.episodes[index]
            agent = episode["agent"]
            report = state["results"][index]["research"]
            start = len(report["batches"])
            records = agent.observations[start:]
            if not records:
                continue
            adapter = InquiryResearchLab(agent, episode["laboratory"])
            for record_index, record in enumerate(records, start):
                rows = adapter.measured_rows(record)
                report["batches"].append({
                    "probe": {"probe_id": f"paid_inquiry_{record_index}",
                              "sequence": deepcopy(record["action"]["sequence"])},
                    "channels": list(record["action"].get("channels", [])),
                    "observations": [asdict(row) for row in rows], "transitions": []})
                report["costs"]["observation_cost"] += len(rows)
                report["costs"]["measurement_cost"] += record.get("measurement_cost", 0)
                report["costs"]["queries"] += 1
            report["actions"].append({"action": "import_paid_observations",
                "record_indices": list(range(start, len(agent.observations)))})
            # These quotes used the earlier paid evidence. The existing native
            # scheduler rescreens the enlarged report on the next run.
            state["pending_revisits"] = [offer for offer in state["pending_revisits"]
                                         if offer["task_position"] != index]

    def _persist(self, *, export_models=True):
        from .temporal_operator_agent import export_temporal_program
        library = self.session.library.to_dict()
        exports = []
        for position, episode in enumerate(self.episodes):
            agent = episode["agent"]
            report = self.session.state["results"][position]["research"]
            agent._save(LEGACY_KNOWLEDGE_PATH, json.dumps(library, indent=2))
            agent._save(METHODS_PATH, json.dumps(library["program_experiment_methods"], indent=2))
            agent._save(SESSION_REPORT_PATH, json.dumps(report, indent=2))
            path, reason = None, None
            try:
                code = export_temporal_program(Program.from_dict(report["program"]))
            except ValueError as error:
                reason = str(error)
            else:
                path = SESSION_MODEL_PATH
                if export_models:
                    agent._save(path, code)
                    self.session.state.setdefault("inquiry_confirmation_contexts", {})[str(position)] = {
                        "model_path": path, "code": code, "program": deepcopy(report["program"])}
            exports.append({"task_position": position, "status": report["status"],
                "model_path": path, "research_path": SESSION_REPORT_PATH,
                "export_reason": reason, "requires_agent_confirmation": True})
        self.session.state["inquiry_exports"] = exports

    def record_confirmation(self, position, feedback):
        """Pass actual fresh finish observations to the old attribution loop."""
        fresh = feedback.get("fresh_confirmation")
        if fresh is None:
            return {"status": "deferred", "reason": "no_fresh_confirmation", "compute_cost": 0}
        state = self.session.state
        receipts = state.setdefault("inquiry_confirmations", [])
        comparison_id = "agent_finish:" + str(position) + ":" + ",".join(
            str(index) for index in fresh["observation_indices"])
        previous_receipt = next((row for row in receipts if row["comparison_id"] == comparison_id), None)
        if previous_receipt is not None:
            return deepcopy(previous_receipt["feedback"])
        episode = self.episodes[position]
        agent = episode["agent"]
        selected = state["results"][position]["research"]
        adapter = InquiryResearchLab(agent, episode["laboratory"])
        fresh_rows = [row for record in fresh["observations"] for row in adapter.measured_rows(record)]
        visit = next((item for item in reversed(state["revisits"])
                      if item["task_position"] == position), None)
        context = state.get("inquiry_confirmation_contexts", {}).get(str(position))
        path = feedback.get("executed_path")
        unchanged = (context is not None and path == context["model_path"]
                     and context["program"] == selected["program"]
                     and (agent.workspace / path).read_text() == context["code"])
        if (visit is None or context is None or path != context["model_path"]
                or context["program"] != selected["program"]):
            result = {"status": "deferred", "reason": "no_matching_frozen_source_program", "compute_cost": 0}
        else:
            from .legacy_session_feedback import settle_source_confirmation
            from .unified_continuation import paid_view
            paid_rows = [row for batch in paid_view(adapter, visit["previous"], ()) for row in batch]
            result = settle_source_confirmation(self.session.library, adapter.spec,
                previous=visit["previous"], selected=selected,
                source_returns=selected.get("knowledge_return", {}).get("returns", []),
                paid_rows=paid_rows, fresh_rows=fresh_rows, frozen_code=context["code"],
                submitted_code=(agent.workspace / path).read_text(), comparison_id=comparison_id,
                available_compute=agent._remaining_compute(), passed=bool(fresh.get("passed")))
            agent._charge(result["compute_cost"])
            selected["costs"]["compute_cost"] += result["compute_cost"]
            state["inquiry_bridge"]["charged_research_compute"][position] += result["compute_cost"]
        # finish already paid for scoring the old rows and then the new rows.
        # Import that evidence and bill into native continuation without a
        # second purchase or a second charge to the current Agent.
        finish_compute = len(agent.rows)
        for index, record in enumerate(fresh["observations"]):
            selected["batches"].append({"probe": {"probe_id": f"{comparison_id}:{index}",
                "sequence": deepcopy(record["action"]["sequence"])},
                "channels": list(record["action"].get("channels", [])),
                "observations": [asdict(row) for row in adapter.measured_rows(record)], "transitions": []})
        selected["costs"]["observation_cost"] += len(fresh_rows)
        selected["costs"]["measurement_cost"] += sum(record.get("measurement_cost", 0)
            for record in fresh["observations"])
        selected["costs"]["queries"] += len(fresh["observations"])
        selected["costs"]["compute_cost"] += finish_compute
        state["inquiry_bridge"]["charged_research_compute"][position] += finish_compute
        if unchanged:
            selected["confirmation_loss"] = fresh.get("mse")
            selected["status"] = "completed" if fresh.get("passed") else "unresolved"
        if result.get("knowledge_return") is not None:
            selected["knowledge_return"] = deepcopy(result["knowledge_return"])
        selected["actions"].append({"action": "import_agent_finish_evidence",
            "comparison_id": comparison_id, "rows": len(fresh_rows),
            "unchanged_native_predictor": unchanged, "compute_cost": finish_compute})
        receipts.append({"task_position": position, "comparison_id": comparison_id,
            "feedback": deepcopy(result)})
        self._persist(export_models=False)
        return result

    def record_research(self, position, result):
        """Consume a bound Agent's actual legacy_research result and full book."""
        agent = self.episodes[position]["agent"]
        state = self.session.state
        previous = state["results"][position]["research"]
        report = json.loads((agent.workspace / result["report_path"]).read_text())
        # The one-task operator starts its imported paid report's compute at
        # zero. Retain the earlier session work alongside its newly paid bill.
        report["costs"]["compute_cost"] = previous["costs"]["compute_cost"] + result["new_costs"]["compute_cost"]
        report["concepts_added"] = list(dict.fromkeys((*previous.get("concepts_added", ()),
                                                     *report.get("concepts_added", ()))))
        self.session.library = KnowledgeLibrary.from_dict(
            json.loads((agent.workspace / result["knowledge_path"]).read_text()))
        state["results"][position].update(research=report,
            available_concepts=list(result["available_concepts"]))
        state["inquiry_bridge"]["charged_research_compute"][position] = report["costs"]["compute_cost"]
        state["pending_revisits"] = [offer for offer in state["pending_revisits"]
                                     if offer["task_position"] != position]
        state["status"] = "paused"
        self._persist()

    def confirm_return(self, position, *, length=24):
        """Buy a separate paired source test using the public measurement view."""
        from .legacy_session_feedback import confirm_source_return
        from .unified_continuation import paid_view
        self.sync_paid_observations(position)
        episode = self.episodes[position]
        agent, laboratory = episode["agent"], episode["laboratory"]
        state = self.session.state
        selected = state["results"][position]["research"]
        visit = next((item for item in reversed(state["revisits"])
                      if item["task_position"] == position), None)
        if visit is None:
            return {"status": "deferred", "reason": "no_native_source_visit", "compute_cost": 0}
        adapter = InquiryResearchLab(agent, laboratory)
        if not 1 <= length <= adapter.spec.max_probe_length:
            raise ValueError("source confirmation length must fit one public trajectory")
        actions = laboratory.confirmation_actions(length, agent.confirmation_rng)
        if len(actions) != 1 or actions[0]["kind"] != "trajectory":
            raise ValueError("source confirmation requires a public trajectory")
        serial = len(state.get("inquiry_source_confirmations", ()))
        probe = Probe(f"inquiry_source_confirmation_{position}_{serial}", tuple(actions[0]["sequence"]))
        channels = tuple(dict.fromkeys((*selected["selected_channels"],
                                        *visit["previous"]["selected_channels"])))
        paid_rows = [row for batch in paid_view(adapter, visit["previous"], channels) for row in batch]
        allowed = self._allowance(position)
        result = confirm_source_return(self.session.library, adapter,
            previous=visit["previous"], selected=selected,
            source_returns=deepcopy(selected.get("knowledge_return", {}).get("returns", [])),
            paid_rows=paid_rows, probe=probe, available_compute=allowed["compute_cost"],
            observation_allowance=allowed["observation_cost"],
            measurement_allowance=allowed["measurement_cost"], comparison_id=probe.probe_id,
            tolerance=self.session.config.minimum_improvement)
        agent._charge(result["compute_cost"])
        if result.get("batch") is not None:
            selected["batches"].append(result["batch"])
            for key, value in result["new_costs"].items():
                selected["costs"][key] += value
            state["inquiry_bridge"]["charged_research_compute"][position] += result["compute_cost"]
            selected["confirmation_loss"] = result["after_loss"]
            selected["status"] = "completed" if result["confirmed_program"] else "unresolved"
            selected["knowledge_return"] = deepcopy(result["knowledge_return"])
            selected["actions"].append({"action": "confirm_source_on_fresh_public_trajectory",
                "probe_id": probe.probe_id, "before_loss": result["before_loss"],
                "after_loss": result["after_loss"], "costs": deepcopy(result["new_costs"])})
        state.setdefault("inquiry_source_confirmations", []).append({
            "task_position": position, "feedback": deepcopy(result)})
        self._persist(export_models=False)
        return result

    def run(self, *, max_tasks=None):
        self.sync_paid_observations()
        try:
            self.session.run(max_tasks=max_tasks)
        finally:
            self._settle()
            self._persist()
        return self.to_dict()

    def to_dict(self):
        return self.session.to_dict()
