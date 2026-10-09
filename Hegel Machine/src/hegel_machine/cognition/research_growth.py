"""Joint growth of executable explanations, measurements and research methods.

The researcher consumes the ordinary TaskSpec/Observation interface. Instruments
are optional and paid through MeasurementBatch. Learned programs enter the same
KnowledgeLibrary used by the other cognitive researchers. No evaluation stream
or simulator mechanism is available to the research loop.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import math
from random import Random
from statistics import mean

from .growth_methods import ResearchMethodBook, instantiate
from .growth_models import abstract_program, bind_prior, discover
from .growth_coordinates import normalized_role_bindings
from .growth_uncertainty import estimate_uncertainty, predictive_variance
from .language import Definition, Program
from .memory import KnowledgeLibrary
from .models import Candidate, Observation, Probe, ResearchConfig, SearchBudget


def _mse(program, rows, budget):
    if not rows or not budget.spend(len(rows)):
        return math.inf
    return mean((program.predict(row) - row.target) ** 2 for row in rows)


def _split(rows):
    # Each acquisition is randomized; interleaving prevents whole late regimes
    # from being placed exclusively in either fit or validation data.
    return ([row for i, row in enumerate(rows) if i % 4 != 3],
            [row for i, row in enumerate(rows) if i % 4 == 3])


class GrowthResearcher:
    """Persist successful explanations and rewrite procedures after failures.

    Primitive expression operators and laboratory commands are fixed. Their
    composition, role binding, applicability conditions, and execution order
    are learned within the task's resource limits.
    """

    def __init__(self, config=None, *, library=None, state=None, measurement_budget=1024):
        state = state or {}
        self.config = config or ResearchConfig(**state.get("config", {
            "observation_budget": 384, "compute_budget": 200000,
            "max_rounds": 6, "seed": 71}))
        self.library = (library if library is not None else
                        KnowledgeLibrary.from_dict(state["library"]) if "library" in state
                        else KnowledgeLibrary())
        if (library is None and "methods" in state
                and "research_workflows" not in state.get("library", {})):
            self.library.research_workflows = ResearchMethodBook.from_dict(state["methods"])
        self.methods = self.library.research_workflows
        self.measurement_budget = state.get("measurement_budget", measurement_budget)
        self.tasks = deepcopy(state.get("tasks", []))
        self.abstractions = deepcopy(state.get("abstractions", []))

    def to_dict(self):
        return {"config": asdict(self.config), "library": self.library.to_dict(),
                "methods": self.methods.to_dict(), "tasks": deepcopy(self.tasks),
                "abstractions": deepcopy(self.abstractions),
                "measurement_budget": self.measurement_budget}

    def run(self, environment):
        spec = environment.spec
        if spec.risk_quantile != .5 or spec.decision_costs is not None:
            raise ValueError("research growth currently uses the fixed squared-error task objective")
        budget = SearchBudget(self.config.compute_budget)
        rng = Random(self.config.seed)
        channels = tuple(getattr(environment, "channels", ()))
        channel_names = [channel.name for channel in channels]
        channel_prices = {channel.name: channel.cost_per_row for channel in channels}
        selected = []
        events, trace, rows = [], [], []
        costs = {"observation_cost": 0, "measurement_cost": 0}
        resource_exhausted = False
        serial = 0
        method = None
        replayed = False
        replay_failed = False
        replay_candidate_id = None
        replay_outcome_recorded = False
        diagnostics = {}
        noise_variance = 0.
        noise_dof = 0
        uncertainty = {"kind": "deterministic", "variance": 0., "variance_program": None,
                       "repeat_degrees_of_freedom": 0}
        representation_changed = False
        confirmed = False
        model = {"program": Program("const", value=0.), "loss": math.inf,
                 "events": [], "alternatives": [], "terms": []}
        used_sources = []
        latest_evidence = "initial"
        data_evidence = {}
        fit_cache = {}

        def event(action, *, success=True, gain=0., recipe=None, consumes=None, **details):
            nonlocal latest_evidence
            identifier = f"step_{len(events)}"
            record = {"event_id": identifier, "action": action, "success": success,
                      "gain": gain, "details": details, "recipe": recipe or {},
                      "consumes": list(consumes) if consumes is not None else [latest_evidence],
                      "produces": [identifier]}
            if recipe:
                record.update({"feature_bounds": spec.feature_bounds,
                               "channel_names": channel_names})
            events.append(record)
            trace.append(record)
            latest_evidence = identifier
            return record

        def probe(count=24, *, repeated=False):
            nonlocal serial
            serial += 1
            count = min(count, spec.max_probe_length)
            points = []
            for _ in range(max(1, count // 6) if repeated else count):
                points.append({name: rng.uniform(*spec.feature_bounds.get(name, (-1., 1.)))
                               for name in spec.feature_names if name not in channel_names})
            sequence = ([point for point in points for _ in range(6)] if repeated else points)
            return Probe(f"growth_{len(self.tasks)}_{serial}", tuple(sequence))

        def acquire(query, selected_channels, action, *, consumes=(), binding=None, **details):
            nonlocal resource_exhausted
            price = query.cost * sum(channel_prices[name] for name in selected_channels)
            if (costs["observation_cost"] + query.cost > self.config.observation_budget
                    or costs["measurement_cost"] + price > self.measurement_budget):
                resource_exhausted = True
                return []
            if channels:
                batch = environment.acquire(query, tuple(selected_channels))
                acquired = batch.observations
                costs["observation_cost"] += batch.observation_cost
                costs["measurement_cost"] += batch.measurement_cost
            else:
                acquired = environment.observe(query)
                costs["observation_cost"] += len(acquired)
            receipt = event(action, consumes=consumes, recipe={"sequence": list(query.sequence),
                                  "control_names": list(query.sequence[0]) if query.sequence else [],
                                  "channels": list(selected_channels),
                                  "channel_ranks": [channel_names.index(name) for name in selected_channels],
                                  **({"channel_binding": binding} if binding else {})},
                  rows=len(acquired), channels=list(selected_channels), **details)
            data_evidence.update({id(row): receipt["event_id"] for row in acquired})
            return acquired

        def priors_for(data):
            names = sorted(set.intersection(*(set(row.features) for row in data))) if data else []
            proposals = []
            # Bound by semantic arity rather than a task/domain label.
            for concept in self.library.concepts[-12:]:
                if (concept.status == "revoked" or concept.risk_quantile != .5
                        or concept.program.history_depth):
                    continue
                saved = next((item for item in self.abstractions
                              if item.get("source_concept_id") == concept.concept_id), None)
                definition = Definition.from_dict(saved) if saved is not None else None
                source = concept.program
                if definition is not None:
                    source_names = sorted(concept.program.required_features())
                    source = Program("call", name=definition.definition_id, definition=definition,
                                     children=tuple(Program("feature", name=name) for name in source_names))
                bindings = bind_prior(source, names, data, budget)
                if saved and sorted(map(tuple, saved.get("source_bounds", {}).values())) != sorted(
                        map(tuple, spec.feature_bounds.values())):
                    bindings.extend(normalized_role_bindings(source, saved.get("source_bounds", {}),
                                                              spec.feature_bounds, data, budget))
                    bindings.sort(key=lambda item: item["loss"])
                for binding in bindings[:2]:
                    proposals.append({**binding, "source": concept.concept_id})
            proposals.sort(key=lambda item: item["loss"])
            unique = []
            for item in proposals:
                if not any(item["program"] == old["program"] for old in unique):
                    unique.append(item)
                if len(unique) == 8:
                    break
            return unique

        def fit(data, *, allowance=1e-6):
            nonlocal used_sources, resource_exhausted
            key = (tuple(id(row) for row in data), allowance)
            if key in fit_cache:
                used_sources = fit_cache[key][0]["used_sources"][:]
                return dict(fit_cache[key][0])
            training, validation = _split(data)
            if len(training) < 3 or not validation:
                return model
            if budget.remaining < 2 * len(data):
                resource_exhausted = True
                return model
            proposed = priors_for(training)
            search = SearchBudget(min(budget.remaining, max(1000, self.config.compute_budget // 6)))
            fitted = discover(training, validation, search,
                              priors=[item["program"] for item in proposed],
                              tolerance=allowance, max_depth=3)
            budget.spend(search.spent)
            if not math.isfinite(fitted["loss"]):
                resource_exhausted = True
                return model
            used_sources = list(dict.fromkeys(item["source"] for item in proposed
                            if item["program"] in fitted.get("used_priors", [fitted["program"]])))
            for item in fitted.get("events", []):
                action = item.get("action", item.get("operation", "compose"))
                event("condition" if action == "learn_executable_scope" else "search_candidate",
                      consumes=(), model_event=item)
            receipt = event("fit", consumes=list(dict.fromkeys(data_evidence.get(id(row), "initial")
                                                                   for row in data)),
                  recipe={"estimator": "residual_expression_search"},
                  model_matches=fitted["loss"] <= allowance, loss=fitted["loss"],
                  expression=fitted["program"].describe(),
                  binding_candidates=[{"source": item["source"], "mapping": item["mapping"],
                                       "loss": item["loss"]} for item in proposed],
                  used_sources=used_sources[:])
            fitted.update(evidence_id=receipt["event_id"], used_sources=used_sources[:])
            # Keep the inputs alive for the duration of the episode; an object's
            # identity denotes these observations, not a later allocator reuse.
            fit_cache[key] = dict(fitted), tuple(data)
            return dict(fitted)

        def project(data, chosen):
            names = set(spec.feature_names) | set(chosen)
            projected = []
            for row in data:
                item = Observation({key: value for key, value in row.features.items() if key in names}, row.target,
                                   tuple({key: value for key, value in previous.items()
                                          if key in names or key == "__target__"} for previous in row.history))
                data_evidence[id(item)] = data_evidence[id(row)]
                projected.append(item)
            return projected

        def compare_instruments(pilot, candidates):
            trials = []
            for subset in candidates:
                projected = project(pilot, subset)
                trial = fit(projected)
                trials.append((trial, subset, projected))
            if not trials:
                return None
            best = min(trials, key=lambda item: item[0]["loss"])
            receipt = next(item for item in trace if item["event_id"] == best[0]["evidence_id"])
            receipt["recipe"]["instrument_comparison"] = {
                "candidate_channels": list(dict.fromkeys(name for _, subset, _ in trials for name in subset)),
                "selected_channel": best[1][-1],
                "losses": {subset[-1]: trial["loss"] for trial, subset, _ in trials}}
            return best

        def learn_uncertainty(repeats):
            nonlocal uncertainty, noise_variance, noise_dof
            estimate = estimate_uncertainty(repeats, model["program"], spec.feature_names, budget,
                                            bounds=spec.feature_bounds)
            if estimate is not None:
                uncertainty = estimate
                noise_variance, noise_dof = estimate["variance"], estimate["repeat_degrees_of_freedom"]

        def adopt_view(best, source, **details):
            nonlocal model, selected, rows, representation_changed, uncertainty, noise_variance, noise_dof
            previous_loss = model["loss"]
            model, selected, rows = best
            representation_changed = True
            uncertainty = {"kind": "deterministic", "variance": 0.,
                           "variance_program": None, "repeat_degrees_of_freedom": 0}
            noise_variance, noise_dof = 0., 0
            choice = event("change_representation", consumes=[model["evidence_id"]],
                           gain=previous_loss - model["loss"], selected_channels=selected[:],
                           source=source, **details)
            # Later refits need both the purchased readings and the comparison
            # that selected their view; otherwise method compilation loses it.
            data_evidence.update({id(row): choice["event_id"] for row in rows})

        for initial in spec.initial_probes:
            rows.extend(acquire(initial, selected, "vary_controls"))
        if len(rows) < 12:
            rows.extend(acquire(probe(24), selected, "vary_controls"))
        model = fit(rows)
        initial_error = model["loss"]

        for round_index in range(self.config.max_rounds):
            just_replayed = False
            if budget.remaining < max(24, 2 * len(rows)):
                resource_exhausted = True
                break
            # A learned switch between old and new explanations creates its
            # own follow-up question: where does the switch actually lie?
            boundaries = model.get("scope_boundaries", [])
            if boundaries:
                points = []
                for boundary in boundaries:
                    if boundary["feature"] not in spec.feature_bounds:
                        continue
                    low, high = boundary["lower"], boundary["upper"]
                    if low is None or high is None:
                        continue
                    if high - low <= 1e-5:
                        continue
                    for fraction in (.2, .4, .6, .8):
                        point = dict(probe(1).sequence[0])
                        point[boundary["feature"]] = low + fraction * (high - low)
                        points.append(point)
                if points:
                    query = Probe(f"boundary_{len(self.tasks)}_{round_index}", tuple(points[:spec.max_probe_length]))
                    extra = acquire(query, selected, "vary_controls", purpose="distinguish_scope_boundary")
                    if extra:
                        rows.extend(extra)
                        model = fit(rows, allowance=max(1e-6, noise_variance * 1.5))
            # A fresh intervention is required even for a perfect initial fit:
            # correlated controls can make structurally different laws agree.
            confirmation = acquire(probe(24), selected, "confirm", round=round_index,
                                   consumes=[model.get("evidence_id", "initial")])
            if not confirmation:
                break
            fresh_error = _mse(model["program"], confirmation, budget)
            predicted_noise = noise_variance
            if uncertainty.get("variance_program") and budget.spend(len(confirmation)):
                predicted_noise = mean(predictive_variance(uncertainty, row) for row in confirmation)
            threshold = max(1e-6, predicted_noise * 2.)
            events[-1]["details"].update({"mse": fresh_error, "threshold": threshold,
                                         "model_confirmed": fresh_error <= threshold})
            if (method is not None and not replay_outcome_recorded
                    and replay_candidate_id == model.get("evidence_id")):
                self.methods.record_outcome(method["method_id"], diagnostics,
                                            success=fresh_error <= threshold, task_id=spec.task_id)
                replay_outcome_recorded = True
                replay_failed = fresh_error > threshold
            rows.extend(confirmation)
            if fresh_error <= threshold:
                confirmed = True
                model["loss"] = fresh_error
                break

            diagnostics = {"alias_error": noise_variance, "residual_error": fresh_error,
                           "prior_error": fresh_error, "repeat_variance": noise_variance,
                           "feature_count": len(rows[-1].features),
                           "feature_bounds": spec.feature_bounds, "channel_names": channel_names}
            if method is None:
                method = self.methods.propose(diagnostics)
            # Replay measured recipes, then refit on the new task. A stored
            # conclusion is never used as the result of the experiment.
            if method is not None and not replayed:
                replayed = True
                just_replayed = True
                event("replay_method", method_id=method.get("method_id"))
                pending_fit = False
                pending_pilot = None
                for step in instantiate(method, spec, channels=channels, attempt=len(self.tasks)):
                    if step.get("action") in {"fit", "compose", "condition"}:
                        if pending_pilot is not None and step.get("subquestions"):
                            best = compare_instruments(pending_pilot, [list(question["channels"])
                                                       for question in step["subquestions"]
                                                       if question.get("channels")])
                            if best and best[0]["loss"] < model["loss"] * .5:
                                adopt_view(best, "method", subquestions=len(step["subquestions"]))
                            pending_pilot, pending_fit = None, False
                        elif pending_fit:
                            model = fit(rows, allowance=max(1e-6, noise_variance * 1.5))
                            pending_fit = False
                        continue
                    if step.get("action") not in {"measure", "vary_controls", "replicate"}:
                        continue
                    query = step.get("probe")
                    if query is None:
                        continue
                    requested = step.get("channels", step.get("recipe", {}).get("channels", selected))
                    requested = [name for name in requested if name in channel_names]
                    requested = list(dict.fromkeys([*selected, *requested]))
                    binding = step.get("recipe", {}).get("channel_binding")
                    obtained = acquire(query, requested, step["action"], replay=True, binding=binding)
                    if not obtained:
                        continue
                    pending_fit = True
                    if binding == "compare_available":
                        pending_pilot = obtained
                        continue
                    if step["action"] == "replicate":
                        learn_uncertainty(obtained)
                    if requested != selected:
                        trial = fit(obtained)
                        if trial["loss"] < model["loss"] * .5:
                            adopt_view((trial, requested, obtained), "method")
                    else:
                        rows.extend(obtained)
                # Only replay one complete learned acquisition subgraph before
                # asking whether its new measurements actually distinguish.
                replay_candidate_id = model.get("evidence_id")

            model = fit(rows, allowance=max(1e-6, noise_variance * 1.5))
            if resource_exhausted:
                break
            if just_replayed and method is not None and not replay_failed:
                replay_failed = model["loss"] > max(1e-6, noise_variance * 1.5)
                if replay_failed:
                    self.methods.record_outcome(method["method_id"], diagnostics,
                                                success=False, task_id=spec.task_id)
                    replay_outcome_recorded = True
                else:
                    replay_candidate_id = model.get("evidence_id")
            if model["loss"] <= max(1e-6, noise_variance * 1.5):
                continue

            repeats = acquire(probe(48, repeated=True), selected, "replicate")
            if repeats:
                rows.extend(repeats)
                learn_uncertainty(repeats)
                diagnostics.update(alias_error=noise_variance, repeat_variance=noise_variance)
                event("identify_observation_limit", conditional_variance=noise_variance,
                      repeat_degrees_of_freedom=noise_dof,
                      conclusion="observed_coordinates_insufficient_or_stochastic" if noise_variance > 1e-6
                      else "deterministic_residual_requires_new_expression")

            # Compare each instrument on the same paid pilot. The full pilot is
            # never passed to a smaller representation's fit or validation.
            remaining_channels = [name for name in channel_names if name not in selected]
            if remaining_channels and noise_variance > 1e-6:
                pilot = acquire(probe(48), channel_names, "measure", binding="compare_available",
                                purpose="compare_observability")
                if pilot:
                    best = compare_instruments(pilot, [[*selected, name] for name in remaining_channels])
                    if best and best[0]["loss"] < model["loss"] * .5:
                        adopt_view(best, "measured_comparison")
                        if model["loss"] <= 1e-6:
                            continue

            if noise_variance > 1e-6 and noise_dof >= 12:
                model = fit(rows, allowance=noise_variance * 1.5)
                event("expand_uncertainty", variance=noise_variance,
                      reason="variation_remains_after_accounting_for_selected_measurements")
            else:
                # The new residual may need a larger data basis before a
                # composed expression can be distinguished from its rivals.
                rows.extend(acquire(probe(32), selected, "vary_controls"))
                model = fit(rows)

        status = "confirmed" if confirmed else (
            "budget_exhausted" if resource_exhausted or budget.remaining == 0 or
            costs["observation_cost"] >= self.config.observation_budget else "unresolved")
        abstraction_ids = []
        if confirmed:
            current_spec = replace(spec, feature_names=tuple(rows[-1].features))
            identifier, added = self.library.learn(
                Candidate(f"growth_{len(self.tasks)}", model["program"], "research_growth",
                          parents=tuple(model.get("used_sources", used_sources))), current_spec, rows)
            abstraction_ids.append(identifier)
            if added:
                definition = replace(abstract_program(model["program"], f"growth:{identifier}"),
                                     source_concept_id=identifier)
                self.abstractions.append({**definition.to_dict(), "source_bounds": deepcopy(spec.feature_bounds)})
                event("retain_abstraction", concept_id=identifier,
                      definition=definition.to_dict())
            previous = self.tasks[-1] if self.tasks else None
            if previous:
                before = Program.from_dict(previous["program"])
                if model["program"].size < before.size:
                    event("adopt_simplification", previous_size=before.size,
                          selected_size=model["program"].size,
                          confirmed_loss=model["loss"])
                elif model["program"].size > before.size:
                    event("restore_complexity", previous_size=before.size,
                          selected_size=model["program"].size,
                          confirmed_loss=model["loss"])
            # Only evidence ancestors of the accepted response survive method
            # compilation. A comparative pilot keeps its actual purchased view.
            method_trace = deepcopy(trace)
            if method is not None and replay_failed:
                learned = self.methods.revise(method["method_id"], method_trace, diagnostics,
                                              success=True, task_id=spec.task_id)
            else:
                learned = self.methods.learn(method_trace, diagnostics or {
                    "residual_error": initial_error, "feature_count": len(spec.feature_names),
                    "feature_bounds": spec.feature_bounds, "channel_names": channel_names},
                    success=True, task_id=spec.task_id)
            event("retain_method", method=learned)
        elif method is not None:
            learned = self.methods.revise(method["method_id"], trace, diagnostics,
                                          success=False, task_id=spec.task_id)
            event("revise_failed_method", method=learned)

        # Count every program evaluation and fitted candidate row through the
        # shared SearchBudget. Evaluation of the frozen output is external.
        costs["compute_cost"] = budget.spent
        result = {"task_id": spec.task_id, "program": model["program"].to_dict(),
                  "expression": model["program"].describe(), "status": status,
                  "validation_loss": model["loss"], "selected_channels": selected[:],
                  "uncertainty": deepcopy(uncertainty),
                  "representation_changed": representation_changed,
                  "abstraction_ids": abstraction_ids, "events": events, "costs": costs,
                  "observations_acquired": costs["observation_cost"]}
        self.tasks.append({key: deepcopy(value) for key, value in result.items() if key != "events"})
        return result
