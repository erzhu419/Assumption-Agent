"""Continue world-prior research across apparatuses with shared executable memory."""

from copy import deepcopy
from dataclasses import asdict, replace
from random import Random

from .evaluation import loss
from .language import Program
from .memory import KnowledgeLibrary
from .meta_prior_research import MetaPriorResearch
from .models import Candidate, Probe, ResearchConfig


class MetaPriorSession:
    def __init__(self, config=None, *, source=None, library=None, state=None):
        self.state = deepcopy(state) if state else {
            "experiment": "continuing_meta_prior_research", "status": "ready",
            "config": asdict(config or ResearchConfig(observation_budget=96, compute_budget=60000,
                minimum_improvement=1e-7, max_rounds=5, seed=31001)),
            "source": source or {"kind": "scalar", "profile": "nested_readout", "seed": 31001},
            "results": [], "active_task": None, "evaluation": []}
        self.config = ResearchConfig(**self.state["config"])
        self.library = (KnowledgeLibrary.from_dict(state["knowledge"]) if state else
                        library if library is not None else KnowledgeLibrary())
        self.source = self.state["source"]
        if self.source["kind"] not in {"scalar", "sc_kg"}:
            raise ValueError("source kind must be scalar or sc_kg")
        self.task_count = len(self.source["problems"]) if self.source["kind"] == "sc_kg" else 4

    def _environment(self, index):
        if self.source["kind"] == "scalar":
            from .meta_prior_lab import make_meta_prior_sequence
            return make_meta_prior_sequence(self.source["seed"], self.source["profile"])[index]
        from .sc_kg_environment import SCKGEnvironment
        return SCKGEnvironment(self.source["project_root"], problem=self.source["problems"][index],
            dimension=self.source.get("dimension", 5), seed=self.source["seed"] + 10007 * index,
            noise=self.source.get("noise", 0.))

    def run(self, *, max_tasks=None, research_steps=None, on_checkpoint=None):
        if max_tasks is not None and max_tasks < 1:
            raise ValueError("task steps must be positive")
        completed = 0
        while len(self.state["results"]) < self.task_count:
            if max_tasks is not None and completed >= max_tasks:
                self.state["status"] = "paused"
                break
            index = len(self.state["results"])
            environment = self._environment(index)
            active = self.state["active_task"]
            provenance = {}
            if active and "available_prior_ids" in active:
                provenance["available_prior_ids"] = deepcopy(active["available_prior_ids"])
            elif not active:
                provenance["available_prior_ids"] = [prior["prior_id"] for prior in self.library.meta_priors]
            researcher = MetaPriorResearch(replace(self.config, seed=self.config.seed + 1009 * index), self.library)

            def checkpoint(result):
                self.state["active_task"] = {"position": index, **provenance, "research": result}
                self.state["status"] = "paused"
                if on_checkpoint:
                    on_checkpoint(self.to_dict())

            result = researcher.run(environment,
                research_state=active["research"]["research_state"] if active else None,
                max_steps=research_steps, on_checkpoint=checkpoint)
            if result["status"] == "paused":
                checkpoint(result)
                break
            self.state["results"].append({"position": index, **provenance, "research": result})
            self.state["active_task"] = None
            completed += 1
            self.state["status"] = "running"
            if on_checkpoint:
                on_checkpoint(self.to_dict())
        else:
            self.state["status"] = "completed"
        return self.to_dict()

    def run_deploy(self, config=None, *, max_tasks=None, research_steps=None,
                   max_optimizer_tasks=None, on_checkpoint=None):
        """Research and deploy each task before its successor can use the library.

        Task limits count newly completed research and deployments in this call.
        Previously researched tasks are deployed first, reusing their paid rows.
        A paused research task or a reached deployment limit stops the sequence.
        """
        if max_tasks is not None and max_tasks < 1:
            raise ValueError("task steps must be positive")
        if max_optimizer_tasks is not None and max_optimizer_tasks < 1:
            raise ValueError("optimizer task steps must be positive")
        self._configure_optimizer(config)
        researched = deployed = 0
        while len(self.state["optimizations"]) < self.task_count:
            if max_optimizer_tasks is not None and deployed >= max_optimizer_tasks:
                break
            if len(self.state["optimizations"]) < len(self.state["results"]):
                self.optimize(max_tasks=1, on_checkpoint=on_checkpoint)
                deployed += 1
                continue
            if max_tasks is not None and researched >= max_tasks:
                break
            self.run(max_tasks=1, research_steps=research_steps, on_checkpoint=on_checkpoint)
            if self.state["active_task"]:
                break
            researched += 1
        return self.to_dict()

    def evaluate(self, *, count=256):
        if len(self.state["results"]) != self.task_count:
            raise ValueError("finish the learning sequence before independent evaluation")
        if count < 1:
            raise ValueError("evaluation count must be positive")
        for row in self.state["results"][len(self.state["evaluation"]):]:
            index, result = row["position"], row["research"]
            environment = self._environment(index)
            seed = self.source["seed"] + 1000003 + index * 137
            if self.source["kind"] == "sc_kg":
                rows = environment.evaluation_rows(seed=seed, count=count)
            else:
                rng, rows = Random(seed), []
                while len(rows) < count:
                    points = tuple({name: rng.uniform(*bounds)
                        for name, bounds in environment.spec.feature_bounds.items()}
                        for _ in range(min(count - len(rows), environment.spec.max_probe_length)))
                    rows.extend(environment.observe(Probe(f"evaluation_{len(rows)}", points)))
            program = Program.from_dict(result["program"]) if result["program"] else Program("const", value=result["fallback"])
            self.state["evaluation"].append({"position": index, "rows": len(rows),
                "observation_cost": len(rows), "compute_cost": len(rows),
                "mse": loss(program, rows, result["fallback"], .5)})
        return self.to_dict()

    def _configure_optimizer(self, config):
        if self.source["kind"] != "sc_kg":
            raise ValueError("optimizer continuation requires an SC-KG source")
        if "optimizer_config" not in self.state:
            if not config or "N" not in config:
                raise ValueError("optimizer config must specify the total simulator budget N")
            self.state["optimizer_config"] = deepcopy(config)
            self.state["optimizations"] = []
        elif config and config != self.state["optimizer_config"]:
            raise ValueError("continue with the saved optimizer config")

    def optimize(self, config=None, *, max_tasks=None, on_checkpoint=None):
        """Deploy the completed research prefix with its existing paid data."""
        if max_tasks is not None and max_tasks < 1:
            raise ValueError("optimizer task steps must be positive")
        self._configure_optimizer(config)
        from .prior_adoption import PriorAdoption
        from .joint_model_revision import JointModelRevision
        from .sc_kg_environment import compile_sc_kg_program, initialize_sc_kg_optimizer

        completed = 0
        for row in self.state["results"][len(self.state["optimizations"]):]:
            if max_tasks is not None and completed >= max_tasks:
                break
            index, research = row["position"], row["research"]
            environment = self._environment(index)
            environment.restore(research["research_state"]["environment_state"])
            from algorithms.single_olhkg import SingleOLHKGConfig
            settings = dict(self.state["optimizer_config"])
            settings.setdefault("seed", self.config.seed + 4000009 + 1009 * index)
            problem = compile_sc_kg_program(environment, Program.from_dict(research["program"]))
            policy = PriorAdoption(minimum_gain=self.config.minimum_improvement)
            recovery = JointModelRevision(environment.spec,
                minimum_error=self.config.minimum_improvement,
                compute_budget=self.config.compute_budget, seed=settings["seed"] + 101)
            optimizer = initialize_sc_kg_optimizer(problem, SingleOLHKGConfig(**settings),
                                                   adoption_policy=policy, recovery=recovery)
            final = optimizer.run()
            result = {"position": index, "task_id": research["task_id"],
                "research_calls": problem.research_simulator_calls,
                "new_optimizer_calls": problem.simulator_calls,
                "total_simulator_calls": problem.research_simulator_calls + problem.simulator_calls,
                "adoption": policy.to_dict(), "adoption_events": deepcopy(optimizer.adoption_events),
                "adoption_costs": deepcopy(optimizer.adoption_diagnostics),
                "hypothesis_revision": recovery.to_dict(),
                "revision_costs": deepcopy(optimizer.recovery_diagnostics),
                "objective_research": optimizer.objective_research.to_dict(),
                "objective_costs": deepcopy(optimizer.objective_diagnostics),
                "objective_scope_revisions": deepcopy(optimizer.objective_scope_revisions),
                "joint_evidence": deepcopy(optimizer.joint_evidence),
                "scope_revisions": deepcopy(optimizer.scope_revisions),
                "final_program": problem.basis.program.to_dict(),
                "final": {key: final.get(key) for key in ("x_recommended", "true_objective",
                    "true_constraint_mean", "true_feasible", "simple_regret", "n_simulations", "total_time_sec")}}
            self.state["optimizations"].append(result)
            source_ids = {item["source_concept_id"] for item in research["prior_updates"]}
            result["revision_prior_updates"] = self._retain_optimizer_revisions(
                environment.spec, recovery.to_dict(), source_ids)
            result["deployment_segments"] = self._record_optimizer_trials(research, result)
            completed += 1
            if on_checkpoint:
                on_checkpoint(self.to_dict())
        return self.to_dict()

    def _retain_optimizer_revisions(self, spec, recovery, source_ids):
        """Keep previous knowledge conditional and retain supported new scopes."""
        from .workspace import observation_from_dict
        if not self.config.use_memory:
            return []
        updates = []
        for episode in recovery["episodes"]:
            previous_records = [prior for prior in self.library.meta_priors
                                if prior["source_concept_id"] in source_ids]
            accepted = episode.get("accepted_revision")
            scope_id = f"{spec.task_id}:deployment_scope_{episode['scope_start']}"
            replacement_id = None
            if accepted:
                rows = [observation_from_dict(recovery["observations"][i]["observation"])
                        for i in episode["row_indices"]
                        if recovery["observations"][i].get("observation_status") != "disputed"]
                scope = replace(spec, task_id=scope_id, context={**spec.context,
                    "evidence_scope": f"observations_from_{episode['scope_start']}"})
                program = Program.from_dict(accepted["program"])
                replacement_id, created = self.library.learn(
                    Candidate(f"revision:{scope_id}", program, "meta_prior_revision",
                              accepted["claim"], parents=tuple(sorted(source_ids))), scope, rows)
                record = {"prior_id": f"meta_prior_{len(self.library.meta_priors) + 1:04d}",
                    "source_concept_id": replacement_id, "parent_concept_ids": sorted(source_ids),
                    "task_id": scope_id, "source_features": list(spec.feature_names),
                    "status": "supported", "support_rows": len(rows),
                    "validation_loss": accepted["validation_mse"], "counterexamples": [], "uses": 0,
                    "probe_ids": [], "evidence_scope_start": episode["scope_start"],
                    **{key: deepcopy(accepted[key]) for key in
                       ("family_id", "kind", "claim", "counter_prediction")}}
                self.library.meta_priors.append(record)
                updates.append({"prior_id": record["prior_id"], "source_concept_id": replacement_id,
                                "created_concept": created, "scope_start": episode["scope_start"]})
            evidence = [deepcopy(recovery["observations"][i]) for i in episode["trigger_indices"]]
            for prior in previous_records:
                prior["status"] = "conditional"
                prior["counterexamples"].append({"task_id": scope_id,
                    "reason": "joint_predictive_mismatch", "observations": evidence,
                    "replacement_concept_id": replacement_id})
            if replacement_id:
                source_ids = {replacement_id}
        return updates

    def _record_optimizer_trials(self, research, result):
        """Attribute each new prediction to the prior used in that data scope."""
        if not self.config.use_memory:
            return []
        policy = result["adoption"]
        observations = policy["observations"]
        scopes = policy.get("scopes", [])
        replacements = {item["scope_start"]: item
                        for item in result["revision_prior_updates"]}
        segments = [{"observation_start": 0, "initial_selected": "default",
                     "prior_ids": [item["prior_id"] for item in research["prior_updates"]],
                     "context": {"task_id": research["task_id"]}}]
        for scope in scopes:
            replacement = replacements.get(scope["context"]["scope_start"])
            segments.append({"observation_start": scope["observation_start"],
                "initial_selected": scope["selected"], "context": deepcopy(scope["context"]),
                "prior_ids": [replacement["prior_id"]] if replacement else []})
        priors = {prior["prior_id"]: prior for prior in self.library.meta_priors}
        for index, segment in enumerate(segments):
            start = segment["observation_start"]
            end = scopes[index]["observation_start"] if index < len(scopes) else len(observations)
            selected = scopes[index]["previous"] if index < len(scopes) else policy["selected"]
            segment.update(observation_end=end, new_predictions=end - start, selected=selected)
            evidence = deepcopy(observations[start:end])
            observation_calls = {f"{purpose}_calls": sum(
                row["context"].get("purpose", "optimization") == purpose for row in evidence)
                for purpose in ("optimization", "confirmation", "coverage", "validation")}
            for purpose in ("objective_exploration", "objective_confirmation"):
                count = sum(row["context"].get("purpose") == purpose for row in evidence)
                if count:
                    observation_calls[f"{purpose}_calls"] = count
            segment["observation_calls"] = observation_calls
            decisions = [deepcopy(item) for item in policy["decisions"]
                if "superseded_by_observation_repair" not in item
                and item["observation_indices"] and start <= min(item["observation_indices"])
                and max(item["observation_indices"]) < end]
            for prior_id in segment["prior_ids"]:
                prior = priors[prior_id]
                prior.setdefault("deployment_trials", []).append({
                    "task_id": research["task_id"], "prior_task_id": prior["task_id"],
                    "prior_id": prior_id, "source_concept_id": prior["source_concept_id"],
                    "context": deepcopy(segment["context"]),
                    "initial_selected": segment["initial_selected"], "selected": selected,
                    "observation_start": start, "observation_end": end,
                    "new_predictions": end - start, "observations": evidence,
                    "observation_calls": deepcopy(observation_calls),
                    "decisions": decisions,
                    "comparison_costs": {"paired_prediction_rows": 2 * (end - start),
                        "shadow_gp_updates": end - start, "shadow_hvd_updates": 2 * (end - start),
                        "policy_observations": end - start}})
        return segments

    def to_dict(self):
        records = [item["research"] for item in self.state["results"]]
        if self.state["active_task"]:
            records.append(self.state["active_task"]["research"])
        status = self.state["status"]
        optimizer_pending = []
        if "optimizer_config" in self.state:
            optimizer_pending = list(range(len(self.state["optimizations"]), self.task_count))
            status = ("completed" if not optimizer_pending else
                      "paused" if self.state["active_task"] else "optimization_paused")
        return {**deepcopy(self.state), "status": status, "knowledge": self.library.to_dict(),
            "pending_tasks": list(range(len(self.state["results"]), self.task_count)),
            "optimizer_pending_tasks": optimizer_pending,
            "costs": {key: sum(row["costs"][key] for row in records)
                      for key in ("observation_cost", "compute_cost")}}


def meta_prior_summary(report):
    optimizations = report.get("optimizations", [])
    task_count = len(report["results"]) + len(report["pending_tasks"])
    revision_costs = {key: sum(result.get("hypothesis_revision", {}).get("costs", {}).get(key, 0)
                              for result in optimizations)
                      for key in ("compute_cost", "confirmation_calls", "coverage_calls", "validation_calls")}
    return {"status": report["status"], "pending_tasks": report["pending_tasks"],
        "active_task": report["active_task"]["position"] if report["active_task"] else None,
        "costs": report["costs"], "evaluation": report["evaluation"],
        "retained_priors": len(report["knowledge"]["meta_priors"]),
        "optimizer_costs": {
            "total_optimizer_calls": sum(result["total_simulator_calls"] for result in optimizations),
            "research_calls_included": sum(result["research_calls"] for result in optimizations),
            "new_optimizer_calls": sum(result["new_optimizer_calls"] for result in optimizations),
            "objective_research_compute": sum(result.get("objective_research", {}).get("compute_cost", 0)
                                              for result in optimizations),
            "revision_research_costs": revision_costs},
        "optimizer_pending_tasks": list(range(len(optimizations), task_count))
            if "optimizer_config" in report else [],
        "optimizations": [{key: result[key] for key in ("position", "total_simulator_calls",
            "new_optimizer_calls", "adoption_costs", "final")} | {"selected": result["adoption"]["selected"]}
            | {"hypothesis_revisions": len(result.get("scope_revisions", [])),
               "revision_phase": result.get("hypothesis_revision", {}).get("phase"),
               "revision_costs": result.get("revision_costs", {}),
               "objective_costs": result.get("objective_costs", {}),
               "objective_research_compute": result.get("objective_research", {}).get("compute_cost", 0),
               "objective_scope_revisions": len(result.get("objective_scope_revisions", [])),
               "revision_research_costs": result.get("hypothesis_revision", {}).get("costs", {})}
            for result in optimizations],
        "results": [{"position": row["position"], **{key: row["research"][key]
            for key in ("status", "family_id", "kind", "costs", "prior_updates")},
            "counterexamples": len(row["research"]["research_state"]["counterexamples"])}
            for row in report["results"]]}
