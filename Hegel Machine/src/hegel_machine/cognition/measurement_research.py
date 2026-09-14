"""Choose instruments, infer continuous state, and continue ordinary research."""

from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations
from random import Random
from time import perf_counter

from .continuous_state import continuous_state_candidates
from .controller import CognitiveAgent
from .evaluation import initial_prediction, loss
from .memory import KnowledgeLibrary
from .language import Program
from .models import Candidate, Observation, Probe, SearchBudget
from .sparse_search import sparse_candidates
from .synthesis import _constant, _fit_linear
from .workspace import ResearchWorkspace


def measured_view(observations, names):
    """Project acquired measurements into a chosen representation of the task."""
    names = set(names)
    return [Observation({name: value for name, value in row.features.items() if name in names},
                        row.target, tuple({name: value for name, value in past.items()
                                           if name in names or name == "__target__"}
                                          for past in row.history)) for row in observations]


def _pilot_probe(spec, seed, index, length):
    rng = Random(seed + 977 * index)
    columns = {}
    for name, (low, high) in spec.feature_bounds.items():
        values = [low + (high - low) * (i + rng.random()) / length for i in range(length)]
        rng.shuffle(values)
        columns[name] = values
    return Probe(f"instrument_pilot_{index}", tuple(
        {name: column[i] for name, column in columns.items()} for i in range(length)))


def _operating_rows(state):
    unique = []
    for batch in state["batches"]:
        if batch["probe"] not in unique:
            unique.append(batch["probe"])
    return sum(len(probe["sequence"]) for probe in unique)


def _trajectories(rows):
    groups = []
    for row in rows:
        if not groups or not row.history:
            groups.append([])
        groups[-1].append(row)
    return groups


def _fit_representation(training, validation, budget, seed_programs=(), counterexamples=()):
    fallback = initial_prediction(training, .5)
    candidates = [Candidate("measurement_constant", _constant(fallback), "refit")]
    if seed_programs:
        names = sorted(set.intersection(*(set(row.features) for row in training)))
        depths = sorted({1, *(lag for program in seed_programs for name, lag in program.observed_variables()
                              if name == "__target__")})
        for depth in depths:
            terms = [Program("feature", name=name) for name in names]
            terms.append(Program("if", children=(Program("available", lag=depth),
                Program("lag", name="__target__", lag=depth), _constant(0.))))
            if budget.spend((len(terms) + 1) * len(training)):
                candidates.append(Candidate(f"instrument_refit:{depth}", _fit_linear(terms, training),
                    "refit", "Refit the existing response history with the newly available readings"))
    ordinary = SearchBudget(min(1800, budget.remaining // 6))
    candidates.extend(sparse_candidates(training, budget=ordinary))
    budget.spend(ordinary.spent)
    # Keep enough work to compare every returned explanation on the separate
    # trajectory. The best training fit alone must not stand in for the value
    # of an instrument: a different representation may generalize better.
    comparison_rows = len(validation) + sum(map(len, counterexamples))
    states = SearchBudget(max(0, budget.remaining - (len(candidates) + 6) * comparison_rows))
    candidates.extend(continuous_state_candidates(training, budget=states, seed_programs=seed_programs))
    budget.spend(states.spent)
    ranked = []
    for candidate in candidates:
        if not budget.spend(comparison_rows):
            break
        measured_loss = max(loss(candidate.program, rows, fallback, .5)
                            for rows in (validation, *counterexamples))
        ranked.append((measured_loss, candidate.program.size, candidate))
    if not ranked:
        return None
    measured_loss, _, candidate = min(ranked, key=lambda item: item[:2])
    return candidate, fallback, measured_loss


class InstrumentedEnvironment:
    """Execute controls while collecting the instruments chosen by research."""

    def __init__(self, laboratory, channels, batches=None):
        self.laboratory = laboratory
        self.batches = batches
        self.channels = tuple(channels)
        self.spec = laboratory.spec_for(self.channels)

    def observe(self, probe):
        rows = self.laboratory.acquire(probe, self.channels).observations
        if self.batches is not None:
            self.batches.append({"probe": asdict(probe), "channels": list(self.channels),
                                 "observations": [asdict(row) for row in rows]})
        return rows


class MeasurementResearchAgent:
    """Use measured predictive benefit to choose a sustainable observation view."""

    def __init__(self, config, *, measurement_budget=512, library=None):
        self.config = config
        self.measurement_budget = measurement_budget
        self.library = library if library is not None else KnowledgeLibrary()

    def _fit_view(self, spec, training, validation, budget, seed_programs=(), counterexamples=()):
        """Test known representations before paying to rediscover this view."""
        fallback = initial_prediction(training, .5)
        retrieved = (self.library.retrieve(spec, observations=training)[:self.config.max_candidates]
                     if self.config.use_memory else [])
        calls = (self.library.callable_atoms(spec, structural_transfer=False,
                                            limit=len(self.library.definitions)) if retrieved else [])
        tested = []
        before = budget.spent
        for candidate in retrieved:
            # Use the complete learned definition, preserving its measured
            # scope and source. A matching subexpression is not a whole model.
            call = next((call for call in calls
                if call.definition.source_concept_id == candidate.memory_id
                and call.expanded().simplified() == candidate.program.expanded().simplified()), None)
            if call is not None:
                candidate = replace(candidate, program=call)
            if not budget.spend(len(training) + len(validation) + sum(map(len, counterexamples))):
                break
            training_loss = loss(candidate.program, training, fallback, spec.risk_quantile)
            validation_loss = loss(candidate.program, validation, fallback, spec.risk_quantile)
            counterexample_loss = max((loss(candidate.program, rows, fallback, spec.risk_quantile)
                                      for rows in counterexamples), default=0.)
            tested.append((candidate, training_loss, validation_loss, counterexample_loss))
        reuse = {"compute_cost": budget.spent - before,
                 "trials": [{"source_concept_id": candidate.memory_id,
                             "training_loss": train_loss, "validation_loss": valid_loss,
                             "counterexample_loss": old_loss,
                             "call": candidate.program.op == "call"}
                            for candidate, train_loss, valid_loss, old_loss in tested],
                 "selected_source": None, "skipped_search": False}
        supported = [item for item in tested
                     if max(item[1:]) <= self.config.minimum_improvement]
        if supported:
            candidate, _, valid_loss, old_loss = min(supported,
                key=lambda item: (max(item[2:]), item[1], item[0].program.size))
            reuse.update(selected_source=candidate.memory_id, skipped_search=True)
            return (candidate, fallback, max(valid_loss, old_loss)), reuse
        # A failed cross-task explanation must not take over the fresh search
        # coordinates. The current workspace can still seed local revision;
        # unrelated old states remain available to ordinary transfer research.
        # Keep an already measured model if fresh search cannot improve it.
        ranked = sorted(tested, key=lambda item: (max(item[2:]), item[1]))
        fitted = _fit_representation(training, validation, budget, seed_programs=seed_programs,
                                     counterexamples=counterexamples)
        if ranked and (fitted is None or max(ranked[0][2:]) < fitted[2]):
            candidate, _, valid_loss, old_loss = ranked[0]
            fitted = candidate, fallback, max(valid_loss, old_loss)
            reuse["selected_source"] = candidate.memory_id
        return fitted, reuse

    def _initialize(self, laboratory, strategy):
        if strategy not in {"learned", "controls_only", "all"}:
            raise ValueError("measurement strategy must be learned, controls_only or all")
        initial_costs = dict(laboratory.costs)
        config = self.config
        instruments = sorted(laboratory.channels, key=lambda channel: channel.cost_per_row)
        acquisition_plan = (self.library.measurements.plan(laboratory)
                            if strategy == "learned" and config.use_memory else None)
        length = min(24, laboratory.spec.max_probe_length, config.observation_budget // 4)
        if acquisition_plan:
            length = min(length, acquisition_plan["pilot_length"])
        if strategy == "all":
            total_rate = sum(channel.cost_per_row for channel in instruments)
            length = min(length, max(0, (self.measurement_budget // total_rate - 3) // 2))
        if length < 6:
            raise ValueError("measurement research requires two pilot trajectories of at least six rows")
        offered = []
        rate = 0
        if strategy != "controls_only":
            for instrument in instruments:
                if acquisition_plan and instrument.name not in acquisition_plan["initial_channels"]:
                    continue
                if (2 * length + 3) * (rate + instrument.cost_per_row) <= self.measurement_budget:
                    offered.append(instrument.name)
                    rate += instrument.cost_per_row
        probes = [_pilot_probe(laboratory.spec, config.seed, index, length) for index in range(2)]
        pilots = [laboratory.acquire(probe, tuple(offered)).observations for probe in probes]
        selection_rows = [*pilots[0], *pilots[1]]
        if strategy == "learned":
            subsets = [subset for size in range(len(offered) + 1) for subset in combinations(offered, size)]
        else:
            subsets = [tuple(offered)]
        selection = SearchBudget(config.compute_budget // 2)
        if not selection.spend(len(selection_rows)):
            raise ValueError("computation budget cannot evaluate the initial observation representation")
        rate_by_name = {channel.name: channel.cost_per_row for channel in instruments}
        assessed = []
        for index, channels in enumerate(subsets):
            names = laboratory.spec_for(channels).feature_names
            training, validation = [measured_view(rows, names) for rows in pilots]
            remaining_subsets = len(subsets) - index
            local = SearchBudget(selection.remaining // remaining_subsets)
            fitted, reuse = self._fit_view(laboratory.spec_for(channels), training, validation, local)
            selection.spend(local.spent)
            if fitted is None:
                break
            candidate, fallback, measured_loss = fitted
            assessed.append({"channels": channels, "rate": sum(rate_by_name[name] for name in channels),
                             "candidate": candidate, "fallback": fallback, "loss": measured_loss,
                             "compute_cost": local.spent, "reuse": reuse})
        if not assessed:
            raise ValueError("computation budget cannot fit and compare an observation representation")
        baseline = next((item["loss"] for item in assessed if not item["channels"]), assessed[0]["loss"])
        best_loss = min(item["loss"] for item in assessed)
        tolerance = max(config.minimum_improvement, .05 * baseline)
        supported = [item for item in assessed if item["loss"] <= best_loss + tolerance]
        chosen = min(supported, key=lambda item: (item["rate"], item["loss"]))
        selected = chosen["channels"]
        environment = InstrumentedEnvironment(laboratory, selected)
        rows = measured_view(selection_rows, environment.spec.feature_names)
        reference = measured_view(pilots[1], environment.spec.feature_names)
        workspace = ResearchWorkspace(environment.spec.task_id, observations=rows, reference=reference,
            queried=probes, incumbent=chosen["candidate"], evidence_champion=chosen["candidate"],
            bank=[chosen["candidate"]], fallback=chosen["fallback"],
            initial_loss=loss(_constant(chosen["fallback"]), rows, chosen["fallback"], .5))
        return {"task_id": environment.spec.task_id, "strategy": strategy, "stage": "research",
                "selected_channels": list(selected), "workspace": workspace.to_dict(),
                "selection": [{"channels": list(item["channels"]), "cost_per_row": item["rate"],
                               "validation_loss": item["loss"], "compute_cost": item["compute_cost"],
                               "reuse": item["reuse"],
                               "expression": item["candidate"].program.describe()} for item in assessed],
                "initial_comparison_compute_cost": len(selection_rows),
                "pilot_observations": [[asdict(row) for row in trajectory] for trajectory in pilots],
                "batches": [{"probe": asdict(probe), "channels": list(offered),
                             "observations": [asdict(row) for row in trajectory]}
                            for probe, trajectory in zip(probes, pilots)],
                "episodes": [], "compute_cost": selection.spent, "measurement_events": [],
                "acquisition_plan": acquisition_plan, "coverage_events": [],
                "measurement_question": {}, "view_validation_loss": chosen["loss"],
                "initial_costs": initial_costs, "wall_time_seconds": 0.}

    def _remaining(self, laboratory, state):
        paid = {name: laboratory.costs[name] - state["initial_costs"][name]
                for name in ("observation_cost", "measurement_cost")}
        return {"observation": self.config.observation_budget - paid["observation_cost"],
                "measurement": self.measurement_budget - paid["measurement_cost"],
                "compute": self.config.compute_budget - state["compute_cost"]}

    def _run_visit(self, laboratory, state):
        remaining = self._remaining(laboratory, state)
        rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        rate = sum(rates[name] for name in state["selected_channels"])
        missing = set(rates) - set(state["selected_channels"])
        # Preserve the cost of another pair of measured trajectories and a
        # subsequent comparison when the current view can still be expanded.
        reserve = 24 if state["strategy"] == "learned" and missing else 0
        allowance = remaining["observation"]
        if rate:
            allowance = min(allowance, remaining["measurement"] // rate)
        observations = allowance - reserve if allowance >= reserve + 8 else allowance
        compute = remaining["compute"] // 2 if reserve and remaining["compute"] >= 4000 else remaining["compute"]
        if observations < 3 or compute < 128:
            state["status"] = "budget_exhausted"
            return False
        environment = InstrumentedEnvironment(laboratory, state["selected_channels"], state["batches"])
        episode = CognitiveAgent(replace(self.config, observation_budget=observations,
                                compute_budget=compute), self.library).run(
                                    environment, research_state=state["workspace"])
        payload = episode.to_dict()
        state["episodes"].append(payload)
        state["workspace"] = payload["research_state"]
        state["compute_cost"] += episode.compute_cost
        # Compare the completed explanation on newly acquired trajectories.
        # These rows are paid learning evidence, not the external final test.
        new_loss = episode.validation_loss
        changed = (new_loss is not None and new_loss > max(self.config.minimum_improvement,
                   state["view_validation_loss"] + self.config.minimum_improvement))
        persistent = (new_loss is not None and new_loss > self.config.minimum_improvement
                      and state["view_validation_loss"] > self.config.minimum_improvement)
        if state["strategy"] == "learned" and (changed or persistent):
            state["measurement_question"] = {
                "reason": "new_trajectory_residual" if changed else "persistent_prediction_residual",
                "before_channels": list(state["selected_channels"]),
                "previous_validation_loss": state["view_validation_loss"], "current_validation_loss": new_loss,
                "available_channels": sorted(missing), "evidence_rows": len(state["workspace"]["observations"]),
                "observation_end": _operating_rows(state),
                "question": "Does an available instrument explain the residual on subsequent trajectories?"}
            state["stage"] = "reselect"
        else:
            plan = state.get("acquisition_plan")
            if plan and _operating_rows(state) < plan["coverage_rows"]:
                state["stage"] = "coverage"
            else:
                state["stage"], state["status"] = "completed", "completed"
        return True

    def _check_coverage(self, laboratory, state):
        """Test beyond the operating progress that supported earlier reuse.

        A cheap view may pass its pilots while a later disturbance is dormant.
        Its source's paid coverage and compatible late counterexamples remain
        obligations even when ordinary program research stops early.
        """
        plan = state.get("acquisition_plan")
        required = plan["coverage_rows"] if plan else 0
        # Replaying instruments on an old experiment costs money but does not
        # extend operation. Compare the actual queries, without a fingerprint.
        observed = _operating_rows(state)
        if observed >= required:
            if state["measurement_question"].get("reason") == "unreached_learned_coverage":
                state["measurement_question"] = {}
            state["stage"], state["status"] = "completed", "completed"
            return True
        remaining = self._remaining(laboratory, state)
        rate = sum(channel.cost_per_row for channel in laboratory.channels
                   if channel.name in state["selected_channels"])
        missing = {channel.name for channel in laboratory.channels} - set(state["selected_channels"])
        reserve = 24 if missing else 0
        length = min(required - observed, laboratory.spec.max_probe_length,
                     max(0, remaining["observation"] - reserve), remaining["compute"])
        if rate:
            length = min(length, remaining["measurement"] // rate)
        if length <= 0:
            state["status"] = "budget_exhausted"
            state["measurement_question"] = {"reason": "unreached_learned_coverage",
                "status": "awaiting_budget", "observed_rows": observed, "required_rows": required,
                "remaining": remaining}
            return False
        workspace = ResearchWorkspace.from_dict(state["workspace"])
        probe = _pilot_probe(laboratory.spec, self.config.seed,
                             300 + len(state["coverage_events"]), length)
        environment = InstrumentedEnvironment(laboratory, state["selected_channels"], state["batches"])
        rows = environment.observe(probe)
        measured_loss = loss(workspace.incumbent.program, rows, workspace.fallback, .5)
        state["compute_cost"] += len(rows)
        workspace.observations.extend(rows)
        workspace.reference = rows
        workspace.queried.append(probe)
        state["workspace"] = workspace.to_dict()
        event = {"action": "measure_learned_coverage", "probe_id": probe.probe_id,
                 "observation_end": observed + len(rows), "required_rows": required,
                 "channels": list(state["selected_channels"]), "validation_loss": measured_loss,
                 "costs": {"observation_cost": len(rows), "measurement_cost": len(rows) * rate,
                           "compute_cost": len(rows)}}
        state["coverage_events"].append(event)
        episode = state["episodes"][-1]
        previous_loss = episode.get("validation_loss")
        if previous_loss is None:
            previous_loss = state["view_validation_loss"]
        episode["observation_cost"] += len(rows)
        episode["compute_cost"] += len(rows)
        episode["validation_loss"] = measured_loss
        episode["actions"].append(event)
        episode["research_state"] = deepcopy(state["workspace"])
        changed = measured_loss > max(self.config.minimum_improvement,
                                      previous_loss + self.config.minimum_improvement)
        persistent = min(measured_loss, previous_loss) > self.config.minimum_improvement
        if changed or persistent:
            state["measurement_question"] = {
                "reason": "new_trajectory_residual", "before_channels": list(state["selected_channels"]),
                "previous_validation_loss": previous_loss,
                "current_validation_loss": measured_loss, "available_channels": sorted(missing),
                "evidence_rows": observed + len(rows),
                "observation_end": observed + len(rows),
                "question": "Does an available instrument explain the residual after learned operating coverage?"}
            state["stage"] = "reselect"
        elif observed + len(rows) >= required:
            state["measurement_question"] = {}
            state["stage"], state["status"] = "completed", "completed"
        return True

    def _reselect(self, laboratory, state):
        from .workspace import observation_from_dict
        # If this representation never explained the paid initial evidence,
        # refine it on the longer existing trajectories before buying inputs.
        # The old independent trajectories remain separate obligations.
        if (state["view_validation_loss"] > self.config.minimum_improvement
                and not any(event["action"] == "revise_measurement_representation"
                            for event in state["measurement_events"])):
            if self._revise_current_view(laboratory, state):
                return True
        remaining = self._remaining(laboratory, state)
        rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        offered = tuple(rates)
        total_rate = sum(rates.values())
        # A poor initial fit is not evidence that its measured inputs are
        # useless. Revisit the already acquired, separate pilot trajectories
        # before charging another full instrument sweep. A later change in
        # mechanism still needs current measurements, not these old pilots.
        reusable = [batch for batch in state["batches"]
                    if set(offered) <= set(batch["channels"])
                    and len(batch["observations"]) >= 6]
        reuse = (state["measurement_question"]["reason"] == "persistent_prediction_residual"
                 and len(reusable) >= 2)
        length = min(16, laboratory.spec.max_probe_length,
                     (remaining["observation"] - 8) // 2,
                     max(0, (remaining["measurement"] - 8 * total_rate) // (2 * total_rate)))
        if (not reuse and length < 6) or remaining["compute"] < 4000:
            state["status"] = "budget_exhausted"
            state["measurement_question"].update(status="awaiting_budget",
                minimum_observation_cost=0 if reuse else 20,
                minimum_measurement_cost=0 if reuse else 20 * total_rate,
                minimum_compute_cost=4000, remaining=remaining)
            return False
        before_cost = dict(laboratory.costs)
        recent = None
        if reuse:
            batches = reusable[-2:]
            first, second = [Probe(batch["probe"]["probe_id"], tuple(batch["probe"]["sequence"]))
                             for batch in batches]
            pilots = [[observation_from_dict(row) for row in batch["observations"]] for batch in batches]
        else:
            recent = next((item["probe"] for item in reversed(state["batches"])
                           if 6 <= len(item["probe"]["sequence"]) <= length), None)
            first = Probe(recent["probe_id"], tuple(recent["sequence"])) if recent else _pilot_probe(
                laboratory.spec, self.config.seed, 100 + len(state["episodes"]), length)
            second = _pilot_probe(laboratory.spec, self.config.seed, 200 + len(state["episodes"]), length)
            pilots = []
            for probe in (first, second):
                rows = laboratory.acquire(probe, offered).observations
                pilots.append(rows)
                state["batches"].append({"probe": asdict(probe), "channels": list(offered),
                                          "observations": [asdict(row) for row in rows]})
        subsets = [subset for size in range(len(offered) + 1) for subset in combinations(offered, size)]
        selection = SearchBudget(remaining["compute"] // 2)
        workspace = ResearchWorkspace.from_dict(state["workspace"])
        assessed = []
        for index, channels in enumerate(subsets):
            before_work = selection.spent
            names = laboratory.spec_for(channels).feature_names
            training, validation = [measured_view(rows, names) for rows in pilots]
            counterexamples = [measured_view(rows, names) for rows in _trajectories(workspace.reference)
                               if all(set(names) <= set(row.features) for row in rows)]
            comparison_rows = len(validation) + sum(map(len, counterexamples))
            local = SearchBudget(max(0, selection.remaining // (len(subsets) - index) - comparison_rows))
            fitted, view_reuse = self._fit_view(laboratory.spec_for(channels), training, validation, local,
                seed_programs=[workspace.incumbent.program], counterexamples=counterexamples)
            selection.spend(local.spent)
            if fitted is None:
                break
            candidate, fallback, measured_loss = fitted
            prior = workspace.incumbent.program
            prior_loss = None
            if prior.required_features() <= set(names) and selection.spend(comparison_rows):
                prior_loss = max(loss(prior, rows, workspace.fallback, .5)
                                 for rows in (validation, *counterexamples))
                if prior_loss < measured_loss:
                    candidate, fallback, measured_loss = workspace.incumbent, workspace.fallback, prior_loss
            assessed.append({"channels": channels, "rate": sum(rates[name] for name in channels),
                             "candidate": candidate, "fallback": fallback, "loss": measured_loss,
                             "counterexample_rows": sum(map(len, counterexamples)),
                             "prior_loss": prior_loss,
                             "compute_cost": selection.spent - before_work, "reuse": view_reuse})
        state["compute_cost"] += selection.spent
        current = next(item for item in assessed if set(item["channels"]) == set(state["selected_channels"]))
        best_loss = min(item["loss"] for item in assessed)
        supported = [item for item in assessed if item["loss"] <= best_loss + self.config.minimum_improvement]
        chosen = min(supported, key=lambda item: (item["rate"], item["loss"]))
        representation_changed = chosen["candidate"].program != workspace.incumbent.program
        adopted = ((set(chosen["channels"]) != set(state["selected_channels"]) or representation_changed)
                   and current["prior_loss"] is not None
                   and chosen["loss"] + self.config.minimum_improvement < current["prior_loss"])
        event = {"action": "recompare_measurements", **state["measurement_question"],
            "status": "completed", "remaining": remaining,
            "evidence_source": "acquired_trajectories" if reuse else "new_measurements",
            "selected_channels": list(chosen["channels"]), "adopted": adopted,
            "representation_changed": representation_changed and adopted,
            "replayed_probe_id": first.probe_id if recent else None,
            "validation_probe_id": second.probe_id,
            "selection": [{"channels": list(item["channels"]), "validation_loss": item["loss"],
                "cost_per_row": item["rate"], "compute_cost": item["compute_cost"],
                "reuse": item["reuse"],
                "counterexample_rows": item["counterexample_rows"],
                "expression": item["candidate"].program.describe()} for item in assessed],
            "costs": {"observation_cost": laboratory.costs["observation_cost"] - before_cost["observation_cost"],
                      "measurement_cost": laboratory.costs["measurement_cost"] - before_cost["measurement_cost"],
                      "compute_cost": selection.spent}}
        state["measurement_events"].append(event)
        if adopted:
            names = laboratory.spec_for(chosen["channels"]).feature_names
            complete = []
            for batch in state["batches"]:
                if set(chosen["channels"]) <= set(batch["channels"]):
                    complete.extend(observation_from_dict(row) for row in batch["observations"])
            # Unmeasured channels are never filled from hidden simulator state.
            # Every older trajectory stays in the saved batches; this view uses
            # only trajectories whose required instruments were actually paid.
            workspace.observations = measured_view(complete, names)
            retained_reference = [row for row in workspace.reference if set(names) <= set(row.features)]
            workspace.reference = measured_view([*retained_reference, *pilots[1]], names)
            workspace.bank = [chosen["candidate"], *(candidate for candidate in workspace.bank
                if set(candidate.program.required_features()) <= set(names))]
            if (set(chosen["channels"]) == set(state["selected_channels"])
                    or not workspace.incumbent.program.required_features() <= set(names)):
                workspace.incumbent = chosen["candidate"]
            if (set(chosen["channels"]) == set(state["selected_channels"])
                    or workspace.evidence_champion
                    and not workspace.evidence_champion.program.required_features() <= set(names)):
                workspace.evidence_champion = chosen["candidate"]
            if workspace.parent and not workspace.parent.program.required_features() <= set(names):
                workspace.parent = None
            workspace.experiment_frontier = {}
            if not reuse:
                workspace.queried.extend((first, second))
            state["workspace"] = workspace.to_dict()
            state["selected_channels"] = list(chosen["channels"])
            state["view_validation_loss"] = chosen["loss"]
            state["measurement_question"] = {}
            state["stage"] = "research"
        else:
            state["stage"], state["status"] = "completed", "completed"
            state["measurement_question"]["status"] = "no_supported_instrument_change"
        return True

    def _revise_current_view(self, laboratory, state):
        workspace = ResearchWorkspace.from_dict(state["workspace"])
        training = [row for row in workspace.observations if row not in workspace.reference]
        references = _trajectories(workspace.reference)
        remaining = self._remaining(laboratory, state)
        # Leave one funded instrument comparison if the current explanation
        # cannot be repaired, and a subsequent ordinary research visit if it can.
        amount = remaining["compute"] - max(4000, remaining["compute"] // 4)
        if len(training) < 6 or not references or amount < 128:
            return False
        local = SearchBudget(amount)
        if not local.spend(len(workspace.reference)):
            return False
        previous_loss = max(loss(workspace.incumbent.program, rows, workspace.fallback, .5)
                            for rows in references)
        fitted, reuse = self._fit_view(laboratory.spec_for(state["selected_channels"]),
            training, references[-1], local, seed_programs=[workspace.incumbent.program],
            counterexamples=references[:-1])
        state["compute_cost"] += local.spent
        adopted = (fitted is not None and fitted[2] <= self.config.minimum_improvement
                   and fitted[2] + self.config.minimum_improvement < previous_loss)
        event = {"action": "revise_measurement_representation", **state["measurement_question"],
            "status": "completed", "evidence_source": "acquired_trajectories",
            "selected_channels": list(state["selected_channels"]), "adopted": adopted,
            "representation_changed": adopted, "training_rows": len(training),
            "counterexample_rows": len(workspace.reference), "before_loss": previous_loss,
            "validation_loss": fitted[2] if fitted is not None else None, "reuse": reuse,
            "costs": {"observation_cost": 0, "measurement_cost": 0, "compute_cost": local.spent}}
        state["measurement_events"].append(event)
        if not adopted:
            return False
        candidate, fallback, measured_loss = fitted
        workspace.bank = [candidate, *workspace.bank]
        workspace.incumbent = workspace.evidence_champion = candidate
        workspace.fallback = fallback
        workspace.experiment_frontier = {}
        state["workspace"] = workspace.to_dict()
        state["view_validation_loss"] = measured_loss
        state["measurement_question"] = {}
        state["stage"] = "research"
        return True

    def _report(self, laboratory, state):
        if (self.config.use_memory and state.get("status") in {"completed", "budget_exhausted"}
                and not state.get("acquisition_experience_recorded")):
            self.library.measurements.record(laboratory, state,
                                            minimum_improvement=self.config.minimum_improvement)
            state["acquisition_experience_recorded"] = True
        costs = {name: laboratory.costs[name] - state["initial_costs"][name] for name in state["initial_costs"]}
        costs["compute_cost"] = state["compute_cost"]
        episode = deepcopy(state["episodes"][-1]) if state["episodes"] else None
        if episode is not None:
            episode["observation_cost"] = sum(item["observation_cost"] for item in state["episodes"])
            episode["compute_cost"] = sum(item["compute_cost"] for item in state["episodes"])
            episode["actions"] = [action for item in state["episodes"] for action in item["actions"]]
        state["apparatus_acquisitions"] = deepcopy(laboratory.acquisitions)
        return {"task_id": state["task_id"], "strategy": state["strategy"],
                "status": state.get("status", "paused"), "selected_channels": state["selected_channels"],
                "measurement_budget": self.measurement_budget,
                "initial_comparison_compute_cost": state["initial_comparison_compute_cost"],
                "selection": state["selection"], "costs": costs,
                "pilot_observations": state["pilot_observations"], "episode": episode,
                "measurement_events": state["measurement_events"],
                "measurement_question": state["measurement_question"],
                "acquisition_plan": state.get("acquisition_plan"),
                "coverage_events": state.get("coverage_events", []),
                "research_state": deepcopy(state), "wall_time_seconds": state["wall_time_seconds"]}

    def run(self, laboratory, *, strategy="learned", research_state=None, max_steps=None, on_checkpoint=None):
        """Continue one apparatus question across research and instrument changes.

        Each step is initial instrument selection, a complete ordinary research
        visit, or an instrument comparison. Saved state resumes the next step.
        """
        if max_steps is not None and max_steps < 1:
            raise ValueError("research steps must be positive")
        started = perf_counter()
        if research_state is not None:
            state = deepcopy(research_state)
            laboratory.restore(state["apparatus_acquisitions"])
            if state["stage"] != "completed":
                state.pop("status", None)
                state.pop("acquisition_experience_recorded", None)
            steps = 0
        else:
            state = self._initialize(laboratory, strategy)
            steps = 1
            if on_checkpoint:
                on_checkpoint(self._report(laboratory, state))
        while state["stage"] != "completed" and (max_steps is None or steps < max_steps):
            advance = {"research": self._run_visit, "reselect": self._reselect,
                       "coverage": self._check_coverage}[state["stage"]]
            if not advance(laboratory, state):
                break
            steps += 1
            if on_checkpoint:
                on_checkpoint(self._report(laboratory, state))
        state["wall_time_seconds"] += perf_counter() - started
        if state["stage"] != "completed" and state.get("status") != "budget_exhausted":
            state["status"] = "paused"
        return self._report(laboratory, state)
