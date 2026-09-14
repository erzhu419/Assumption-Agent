"""Public SC-KG measurements and executable hypotheses returned as GP features.

The optional SC-KG checkout supplies the simulator and its ordinary parametric
GP.  Research sees normalized controls and acquired outputs; the learned
stateless program can subsequently become a constraint basis in that GP.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import replace
import importlib
from math import isfinite
from pathlib import Path
import random
import sys

from .language import Program
from .models import Observation, Probe, TaskSpec


def _sc_kg(project_root):
    root = Path(project_root).resolve()
    if not (root / "problems" / "rzdt.py").is_file():
        raise ValueError("project_root must contain the SC-KG problems/rzdt.py module")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("problems.rzdt")
    if Path(module.__file__).resolve() != root / "problems" / "rzdt.py":
        raise ValueError("a different SC-KG checkout is already imported in this process")
    return module, importlib.import_module("numpy")


def _normalized(point, names, lower, upper):
    return {name: (float(value) - lo) / (hi - lo)
            for name, value, lo, hi in zip(names, point, lower, upper)}


def _probe_catalog(names, seed):
    rng = random.Random(seed)
    probes = []
    for index in range(10):
        columns = []
        for _ in names:
            values = [(cell + rng.random()) / 8 for cell in range(8)]
            rng.shuffle(values)
            columns.append(values)
        probes.append(Probe(f"controls_{index:02d}", tuple(
            {name: column[row] for name, column in zip(names, columns)}
            for row in range(8))))
    return tuple(probes[:2]), tuple(probes[2:])


class SCKGEnvironment:
    """Query the constraint output of an existing local SC-KG benchmark.

    Every returned scalar costs one simulator call, including repeat queries.
    Inputs are quantized to the simulator's public integer bounds, and returned
    features describe the actual executed point. No mechanism name enters spec.
    """

    def __init__(self, project_root, problem="rzdt1", dimension=5, seed=7, noise=0.0):
        rzdt, np = _sc_kg(project_root)
        classes = {"rzdt1": rzdt.RZDT1, "rzdt2": rzdt.RZDT2,
                   "state_policy_rzdt1": rzdt.StatePolicyRZDT1}
        if problem not in classes:
            raise ValueError("problem must be rzdt1, rzdt2 or state_policy_rzdt1")
        if dimension < 2 or noise < 0 or not isfinite(noise):
            raise ValueError("dimension must be at least two and noise finite and nonnegative")
        self.project_root = str(Path(project_root).resolve())
        self._settings = dict(problem=problem, dimension=dimension, seed=seed, noise=noise)
        self._problem = classes[problem](d=dimension, L=100, sigma=noise,
                                         heteroscedastic=False)
        self._lower, self._upper = self._problem.int_bounds()
        self._rng = np.random.default_rng(seed)
        names = tuple(f"x{index + 1}" for index in range(dimension))
        initial, available = _probe_catalog(names, seed + 1009)
        self.spec = TaskSpec(
            task_id=f"sc_kg_observations_{seed}",
            description="Predict the scalar response to normalized independent controls.",
            feature_names=names, initial_probes=initial, available_probes=available,
            feature_bounds={name: (0.0, 1.0) for name in names}, max_probe_length=64,
            context={"query_reset": "independent_observation", "observation_unit": "dimensionless",
                     "objective_direction": "minimize", "constraint_upper_bound": "0.0"},
        )
        self.acquisitions = []
        self.evaluation_acquisitions = []

    @property
    def costs(self):
        count = sum(row["observation_cost"] for row in self.acquisitions)
        return {"observation_cost": count, "simulator_calls": count}

    @property
    def evaluation_costs(self):
        count = sum(row["observation_cost"] for row in self.evaluation_acquisitions)
        return {"observation_cost": count, "simulator_calls": count}

    def snapshot(self):
        """Save the paid observations and random-stream position for continuation."""
        return {"settings": dict(self._settings),
                "rng_state": deepcopy(self._rng.bit_generator.state),
                "acquisitions": deepcopy(self.acquisitions),
                "evaluation_acquisitions": deepcopy(self.evaluation_acquisitions)}

    def restore(self, snapshot):
        """Resume a matching apparatus without reacquiring its saved observations."""
        if snapshot["settings"] != self._settings:
            raise ValueError("snapshot belongs to a different SC-KG apparatus configuration")
        self._rng.bit_generator.state = deepcopy(snapshot["rng_state"])
        self.acquisitions = deepcopy(snapshot["acquisitions"])
        self.evaluation_acquisitions = deepcopy(snapshot["evaluation_acquisitions"])

    def integer_point(self, features):
        if set(features) != set(self.spec.feature_names):
            raise ValueError("a control point must supply every public feature")
        values = [float(features[name]) for name in self.spec.feature_names]
        if any(not isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("normalized controls must be finite and lie in [0, 1]")
        return tuple(int(round(lo + value * (hi - lo)))
                     for value, lo, hi in zip(values, self._lower, self._upper))

    def observe(self, probe):
        if not probe.sequence or len(probe.sequence) > self.spec.max_probe_length:
            raise ValueError("probe length must lie within the public observation limit")
        points = [self.integer_point(features) for features in probe.sequence]
        rows, vectors = [], []
        for point in points:
            vector = [float(value) for value in self._problem.simulate(point, self._rng)]
            vectors.append(vector)
            target = vector[2]
            rows.append(Observation(_normalized(point, self.spec.feature_names,
                                                self._lower, self._upper), target))
        self.acquisitions.append({
            "probe_id": probe.probe_id, "integer_points": [list(point) for point in points],
            "sequence": [dict(row.features) for row in rows],
            "targets": [row.target for row in rows], "observation_cost": len(rows),
            "vector_outputs": vectors,
        })
        return rows

    def evaluation_rows(self, *, seed=None, count=128):
        """Pay for fresh measurements on a separate stream after learning."""
        evaluation_seed = self._settings["seed"] + 1_000_003 if seed is None else seed
        settings = {**self._settings, "seed": evaluation_seed}
        fresh = SCKGEnvironment(self.project_root, **settings)
        rng = random.Random(evaluation_seed + 47)
        rows = []
        while len(rows) < count:
            sequence = tuple({name: rng.random() for name in self.spec.feature_names}
                             for _ in range(min(self.spec.max_probe_length, count - len(rows))))
            rows.extend(fresh.observe(Probe(f"evaluation_{len(rows)}", sequence)))
        self.evaluation_acquisitions.extend(deepcopy(fresh.acquisitions))
        return tuple(rows)


class HegelProgramBasis:
    """One executable learned coordinate, consumed by SC-KG ParametricGPR."""

    feature_dim = 1

    def __init__(self, program, feature_names, bounds):
        expanded = program.expanded()
        pending = [expanded]
        while pending:
            node = pending.pop()
            if node.op in {"lag", "available", "accumulate", "window_sum", "event_delta", "arg"}:
                raise ValueError("the SC-KG point basis requires a stateless program")
            pending.extend(node.children)
        if not expanded.required_features().issubset(feature_names):
            raise ValueError("program uses features absent from the SC-KG public controls")
        self.program = program
        self.feature_names = tuple(feature_names)
        self.lower, self.upper = bounds
        self.evaluations = 0

    def features(self, point):
        if len(point) != len(self.feature_names):
            raise ValueError("point dimension differs from the learned basis")
        features = _normalized(point, self.feature_names, self.lower, self.upper)
        result = self.program.predict(Observation(features, 0.0))
        self.evaluations += 1
        return [result]

    def features_many(self, points):
        import numpy as np
        return np.asarray([self.features(point) for point in points], dtype=float).reshape(-1, 1)

    def diagnostics(self):
        return {"basis": "hegel_executable_program", "feature_dim": self.feature_dim,
                "evaluations": self.evaluations, "program": self.program.to_dict()}


class HegelProgramTermBasis(HegelProgramBasis):
    """Allow an accepted program's additive mechanisms to change separately."""

    def __init__(self, original_basis):
        from .constraint_retention import program_terms
        super().__init__(original_basis.program, original_basis.feature_names,
                         (original_basis.lower, original_basis.upper))
        intercept, self.terms, coefficients = program_terms(self.program)
        self.program_coefficients = (intercept, *coefficients)
        self.feature_dim = len(self.terms)

    def features(self, point):
        features = _normalized(point, self.feature_names, self.lower, self.upper)
        row = Observation(features, 0.)
        self.evaluations += len(self.terms)
        return [term.predict(row) for term in self.terms]

    def features_many(self, points):
        import numpy as np
        return np.asarray([self.features(point) for point in points], dtype=float).reshape(-1, self.feature_dim)

    def diagnostics(self):
        return {**super().diagnostics(), "basis": "hegel_program_terms",
                "terms": [term.to_dict() for term in self.terms],
                "program_coefficients": list(self.program_coefficients)}


class HegelProgramProblem:
    """The ordinary SC-KG scalar problem with a learned constraint coordinate."""

    prefer_direct_gpr_basis = True

    def __init__(self, environment, program):
        ScalarizedProblem = importlib.import_module("problems.single_objective").ScalarizedProblem
        self.base = ScalarizedProblem(environment._problem)
        self.spec = environment.spec
        self.basis = HegelProgramBasis(program, environment.spec.feature_names, self.base.int_bounds())
        self.simulator_calls = 0
        self.research_simulator_calls = environment.costs["simulator_calls"]
        self.loaded_observation_count = 0
        self.paid_observations = []
        for acquisition in environment.acquisitions:
            vectors = acquisition.get("vector_outputs", [])
            if len(vectors) != len(acquisition["integer_points"]):
                self.paid_observations = None
                break
            for point, vector in zip(acquisition["integer_points"], vectors):
                self.paid_observations.append({
                    "point": tuple(point), "vector_output": list(vector),
                    "output": [float(self.base.weights[0] * vector[0]
                                     + self.base.weights[1] * vector[1]), vector[2]],
                    "probe_id": acquisition["probe_id"],
                })

    def __getattr__(self, name):
        return getattr(self.base, name)

    def simulate(self, point, rng=None):
        result = self.base.simulate(point, rng)
        self.simulator_calls += 1
        return result

    def gpr_basis_map(self, output_index=1):
        return self.basis if output_index == 1 else None


def compile_sc_kg_program(environment: SCKGEnvironment, program: Program) -> HegelProgramProblem:
    """Return an SC-KG problem whose constraint GP executes the learned program."""
    return HegelProgramProblem(environment, program)


def initialize_sc_kg_optimizer(problem: HegelProgramProblem, config, adoption_policy=None, recovery=None):
    """Load every paid vector observation before the original SC-KG search.

    SC-KG has a simulator-based initialization hook but no dataset loader. This
    small subclass supplies the recorded initial observations through that
    hook, retaining repeated noisy draws. GP/HVD fitting, candidate scoring,
    posterior updates and subsequent simulation calls use the original code.
    ``config.N`` is the total budget including these already paid observations.
    An adoption policy compares both constraint models only on subsequent new
    observations. Each model retains its own GP and HVD; switching happens
    after the current observation update and before the next decision.
    """
    import numpy as np
    from time import perf_counter
    from algorithms.single_olhkg import SingleOLHKGAlgorithm
    from core.gpr import ParametricGPR

    if recovery is not None and adoption_policy is None:
        from .prior_adoption import PriorAdoption
        adoption_policy = PriorAdoption()

    if problem.paid_observations is None:
        raise ValueError("research records lack actual vector outputs; acquire new data instead of reconstructing them")
    records = deepcopy(problem.paid_observations)
    if not records or len(records) != problem.research_simulator_calls:
        raise ValueError("initialization needs every paid research observation")
    if config.N < len(records):
        raise ValueError("the total optimizer budget is smaller than the paid research cost")

    class ResearchInitializedSCAlgorithm(SingleOLHKGAlgorithm):
        def __init__(self):
            super().__init__(problem, replace(config, n0=len(records)))
            self.adoption_policy = adoption_policy
            self.recovery = recovery
            self.objective_research = None
            self.objective_replacement = None
            if recovery is not None:
                from .objective_research import ObjectiveResearch
                from .objective_replacement import ObjectiveReplacement
                self.objective_research = ObjectiveResearch(problem.spec)
                self.objective_replacement = ObjectiveReplacement()
                self.objective_research.state["replacement_research"] = self.objective_replacement.state
            self.objective_diagnostics = {
                "forecast_rows": 0, "mean_prediction_rows": 0,
                "epistemic_prediction_rows": 0, "noise_prediction_rows": 0,
                "objective_exploration_requests": 0, "objective_confirmation_requests": 0,
                "observations": 0, "controller_seconds": 0., "forecast_seconds": 0.,
                "requests": [],
                "head_refits": 0, "head_fit_records": 0, "head_hvd_initializations": 0,
                "head_refit_seconds": 0., "scope_filtered_updates": 0,
                "head_posterior_updates": 0, "head_posterior_noise_predictions": 0,
                "head_posterior_seconds": 0.,
                "scope_filtered_fantasy_updates": 0,
                "replacement_fits": 0, "replacement_fit_records": 0,
                "replacement_prediction_rows": 0, "replacement_seconds": 0.,
                "replacement_installs": [], "replacement_install_fit_records": 0,
                "replacement_install_seconds": 0.,
                "terminal_rounds": 0, "terminal_candidates": 0,
                "terminal_selected_offers": 0, "terminal_new_actions": 0,
                "terminal_repeat_actions": 0, "terminal_outside_offer_actions": 0,
                "terminal_native_pool_rounds": 0, "terminal_native_pool_candidates": 0,
                "terminal_candidate_sets": [], "terminal_choices": [],
                "boundary_noise": {"models": [], "posteriors": [],
                    "certification_calls": 0, "certification_rows": 0,
                    "weight_updates": 0, "component_evaluations": 0, "seconds": 0.},
                "value_precision": {"evaluation_budget": 512, "evaluations": 0,
                    "tie_forecast_rows": 0, "score_one_calls": 0,
                    "terminal_value_calls": 0, "terminal_pool_rows": 0,
                    "seconds": 0., "rounds": []},
            }
            self._terminal_goal_offers = {}
            self._terminal_noise_evidence = None
            self._terminal_noise_model = None
            self._terminal_noise_reference = {}
            self._terminal_noise_values = {}
            self._terminal_noise_live = None
            self._value_precision_round = None
            self.scope_revisions = []
            self.active_scope_start = 0
            self.active_objective_scope_start = 0
            self.objective_scope_revisions = []
            self.joint_evidence = {"checked_goal_scopes": [], "applicability_requests": [],
                                   "prediction_rows": 0, "shared_predictions": 0}
            self.rejected_constraint_indices = set()
            self.observation_repairs = []
            self._confirmation_predictors = None
            self.recovery_diagnostics = {
                "candidate_prediction_rows": 0, "confirmation_requests": 0,
                "coverage_requests": 0, "validation_requests": 0,
                "applicability_probe_requests": 0, "applicability_confirmation_requests": 0,
                "recovery_observations": 0, "scope_rebases": 0,
                "scope_model_fits": 0, "scope_fit_records": 0,
                "scope_hvd_initializations": 0, "setup_seconds": 0.,
                "candidate_prediction_seconds": 0., "controller_seconds": 0.,
                "scope_rebase_seconds": 0.,
                "predictor_snapshots": 0, "observation_repairs": 0,
                "repair_gp_updates": 0, "repair_hvd_updates": 0,
                "filtered_replicate_updates": 0, "filtered_fantasy_updates": 0,
                "observation_repair_seconds": 0., "predictor_snapshot_seconds": 0.,
                "repair_mean_predictions": 0, "repair_epistemic_predictions": 0,
                "repair_noise_predictions": 0, "filtered_epistemic_predictions": 0,
            }
            self.constraint_retention = {
                "scopes": [], "observations": [],
                "prediction_rows": 0, "component_updates": 0,
                "comparison_rows": 0, "setup_seconds": 0.,
                "coefficient_fits": 0, "coefficient_fit_records": 0,
                "paid_design_fits": 0, "paid_design_fit_records": 0,
                "paid_design_unidentified": 0, "paid_design_seconds": 0.,
            }
            self.recovery_diagnostics["constraint_retention"] = self.constraint_retention
            if recovery is not None:
                recovery_started = perf_counter()
                self._scope_variance_template = deepcopy(self.variance_model)
                self.recovery_diagnostics["setup_seconds"] = perf_counter() - recovery_started
            self.adoption_events = []
            self.adoption_diagnostics = {
                "additional_model_fits": 0, "additional_fit_records": 0,
                "paired_prediction_rows": 0, "shadow_gp_updates": 0,
                "shadow_hvd_updates": 0, "policy_observations": 0,
                "additional_setup_seconds": 0.,
                "additional_model_fit_seconds": 0., "comparison_seconds": 0.,
                "shadow_update_seconds": 0., "policy_seconds": 0.,
            }
            self._pending_adoption = None
            self._recovery_control_purpose = "optimization"
            self.active_constraint_basis = (
                "learned" if self.gpr[1].basis_map is problem.basis else "default")
            if adoption_policy is not None:
                setup_started = perf_counter()
                self.constraint_models = {
                    "learned": self.gpr[1],
                    "default": ParametricGPR(
                        problem.d, self.config.lambda_i, self.config.prior_var,
                        normalize_func=problem.normalize,
                        numeric_backend=self.config.numeric_backend,
                        numeric_backend_device=self.config.numeric_backend_device,
                        torch_dtype=self.config.torch_dtype,
                        torch_min_rows=self.config.torch_min_rows),
                }
                self.constraint_variances = {
                    "default": self.variance_model,
                    "learned": deepcopy(self.variance_model),
                }
                self.active_constraint_basis = "default"
                self.gpr[1] = self.constraint_models["default"]
                self.adoption_diagnostics["additional_setup_seconds"] = perf_counter() - setup_started

        def _initial_samples(self):
            return [record["point"] for record in records]

        def _generate_candidates(self, iteration):
            self._recovery_control_purpose = "optimization"
            self._terminal_goal_offers = {}
            if (self._terminal_noise_evidence is not None and
                    self._terminal_noise_evidence["constraint_scope_start"] != self.active_scope_start):
                self._terminal_noise_evidence = None
                self._terminal_noise_model = None
                self._terminal_noise_live = None
            if self.recovery is not None:
                started = perf_counter()
                self._consider_constraint_applicability()
                control = self.recovery.next_control()
                self.recovery_diagnostics["controller_seconds"] += perf_counter() - started
                if control is not None:
                    values = [control[name] for name in self.problem.basis.feature_names]
                    point = tuple(map(int, self.problem.continuous_to_int(values)))
                    purpose = self.recovery.control_purpose
                    self._recovery_control_purpose = purpose
                    self.recovery_diagnostics[f"{purpose}_requests"] += 1
                    return [point], {point: f"revision_{purpose}"}
                research = self.objective_research.state["scope_research"]
                pending_goal = self.objective_research.state["pending"]
                remaining = self.config.N - len(self.history)
                if (self.recovery.state["phase"] == "unresolved"
                        and research is not None and research["phase"] == "validating"
                        and research["candidate"] is not None
                        and 0 < research.get("validation_target", 4) - len(research["validation"]) <= remaining
                        and (pending_goal is None
                             or pending_goal["reason"] == "validate_objective_scope")):
                    # A failed constraint search cannot strand an independently
                    # frozen objective candidate with an affordable validation.
                    started = perf_counter()
                    forecast_seconds = self.objective_diagnostics["forecast_seconds"]
                    request = self.objective_research.next_control(
                        self._objective_evidence(), self._objective_forecast,
                        scope_start=self.active_scope_start, stage=len(self.history),
                        remaining_calls=remaining)
                    self.objective_diagnostics["controller_seconds"] += max(0., perf_counter() - started
                        - (self.objective_diagnostics["forecast_seconds"] - forecast_seconds))
                    if request is not None:
                        values = [request["features"][name] for name in problem.basis.feature_names]
                        point = tuple(map(int, problem.continuous_to_int(values)))
                        self._register_objective_request(request, point)
                        return [point], {point: request["purpose"]}
                if getattr(self.recovery, "state", {}).get("phase") == "monitoring":
                    started = perf_counter()
                    forecast_seconds = self.objective_diagnostics["forecast_seconds"]
                    remaining = self.config.N - len(self.history)
                    if self.objective_scope_revisions and 1 <= remaining <= 4:
                        # This measurement question must get its share before
                        # the goal pool consumes the last planning allowance.
                        valuation = self.objective_diagnostics["value_precision"]
                        allowance = (valuation["evaluation_budget"] - valuation["evaluations"]) // remaining
                        evidence = None
                        if allowance >= 64 and self._terminal_noise_evidence is None:
                            evidence = self._boundary_noise_evidence(remaining)
                        offers = self.objective_research.terminal_offers(
                            self._objective_evidence(), self._objective_forecast,
                            scope_start=self.active_scope_start, stage=len(self.history),
                            remaining_calls=remaining)
                        # A repeat bought to compare objectives does not itself
                        # establish a constraint-noise research question. First
                        # nominate actions under the retained explanation; use
                        # scale uncertainty when a promising control actually
                        # needs a boundary decision. The same nominated actions
                        # then compete under the revised predictive belief.
                        questions = [offer for offer in offers if offer.get("reason") in {
                            "supported_goal_with_uncertain_constraint",
                            "research_promising_constraint_certificate"}]
                        if evidence is not None and questions:
                            evidence["research_questions"] = [{
                                "reason": offer["reason"],
                                "features": deepcopy(offer["features"])} for offer in questions]
                            self._install_boundary_noise(evidence)
                        for offer in offers:
                            values = [offer["features"][name] for name in problem.basis.feature_names]
                            point = tuple(map(int, problem.continuous_to_int(values)))
                            self._terminal_goal_offers.setdefault(point, deepcopy(offer))
                        if self._terminal_goal_offers:
                            self.objective_diagnostics["controller_seconds"] += max(0., perf_counter() - started
                                - (self.objective_diagnostics["forecast_seconds"] - forecast_seconds))
                            points = list(self._terminal_goal_offers)
                            self.objective_diagnostics["terminal_rounds"] += 1
                            self.objective_diagnostics["terminal_candidates"] += len(points)
                            self.objective_diagnostics["terminal_candidate_sets"].append({
                                "stage": len(self.history) + 1, "remaining_calls": remaining,
                                "offers": [{**offer, "point": list(point)}
                                           for point, offer in self._terminal_goal_offers.items()]})
                            if len(points) == 1 and any(control == points[0] for control, _ in self.history):
                                candidates, sources = super()._generate_candidates(iteration)
                                self.objective_diagnostics["terminal_native_pool_rounds"] += 1
                                self.objective_diagnostics["terminal_native_pool_candidates"] += len(candidates)
                                if points[0] not in sources:
                                    candidates.append(points[0])
                                sources[points[0]] = "objective_terminal_choice"
                                return candidates, sources
                            # SC's run builds its full recommendation pool and
                            # scores these actions together against that pool.
                            return points, {point: "objective_terminal_choice" for point in points}
                    request = self.objective_research.next_control(
                        self._objective_evidence(), self._objective_forecast,
                        scope_start=self.active_scope_start, stage=len(self.history),
                        remaining_calls=self.config.N - len(self.history))
                    self.objective_diagnostics["controller_seconds"] += max(0., perf_counter() - started
                        - (self.objective_diagnostics["forecast_seconds"] - forecast_seconds))
                    if request is not None:
                        values = [request["features"][name] for name in problem.basis.feature_names]
                        point = tuple(map(int, problem.continuous_to_int(values)))
                        self._register_objective_request(request, point)
                        return [point], {point: request["purpose"]}
            return super()._generate_candidates(iteration)

        def _effective_exact_terminal_mode(self):
            if (self._terminal_noise_model is not None and
                    self._terminal_noise_evidence["constraint_scope_start"] == self.active_scope_start):
                return "certified_lexicographic"
            return super()._effective_exact_terminal_mode()

        def _terminal_value_from_models(self, gpr_models, variance_model, pool,
                task_ensemble=None, observations=None, terminal_mode_override=None):
            if getattr(self, "_value_precision_round", None) is not None:
                self._value_precision_round["terminal_value_calls"] += 1
                self._value_precision_round["terminal_pool_rows"] += len(pool)
            return super()._terminal_value_from_models(gpr_models, variance_model, pool,
                task_ensemble=task_ensemble, observations=observations,
                terminal_mode_override=terminal_mode_override)

        def _boundary_noise_evidence(self, remaining):
            def noise_prediction(features):
                values = [features[name] for name in problem.basis.feature_names]
                point = tuple(map(int, problem.continuous_to_int(values)))
                self.objective_diagnostics["noise_prediction_rows"] += 1
                return self.variance_model.predict_variance(1, point, problem)

            return self.objective_research.boundary_noise_evidence(
                self._objective_evidence(), noise_prediction,
                scope_start=self.active_scope_start, remaining_calls=remaining)

        def _install_boundary_noise(self, evidence):
            started = perf_counter()
            self._terminal_noise_evidence = deepcopy(evidence)
            self._terminal_noise_model = self._clone_variance_model_for_exact_kg(self.variance_model)
            self._terminal_noise_reference = {point: len(values) for point, values in self.observations.items()}
            self._terminal_noise_values = {}
            for index, (point, output) in enumerate(self.history):
                if index >= self.active_scope_start and index not in self.rejected_constraint_indices:
                    self._terminal_noise_values.setdefault(point, []).append(float(output[1]))
            self._terminal_noise_live = None
            self.objective_diagnostics["boundary_noise"]["models"].append({
                "stage": len(self.history), "scope_start": self.active_scope_start,
                "evidence": deepcopy(evidence), "noise_shape": "frozen_current_hvd",
                "decision": "posterior_predictive_chance_quantile"})
            self.objective_diagnostics["boundary_noise"]["seconds"] += perf_counter() - started

        def _boundary_noise_posterior(self, observations=None):
            if (self._terminal_noise_model is None or self.task_ensemble is not None
                    or self._terminal_noise_evidence["constraint_scope_start"] != self.active_scope_start):
                return None
            from .boundary_noise import noise_tail_factor, update_noise_weights

            observations = self.observations if observations is None else observations
            live = observations is self.observations
            live_key = (len(self.history), tuple(sorted(self.rejected_constraint_indices)))
            if live and self._terminal_noise_live is not None and self._terminal_noise_live[0] == live_key:
                return self._terminal_noise_live[1]
            started = perf_counter()
            evidence = deepcopy(self._terminal_noise_evidence)
            updates = 0
            indices = {}
            for index, (point, _) in enumerate(self.history):
                indices.setdefault(point, []).append(index)
            for point, values in observations.items():
                first_new = self._terminal_noise_reference.get(point, 0)
                extra = values[first_new:]
                if not extra:
                    continue
                previous = list(self._terminal_noise_values.get(point, []))
                noise = self._terminal_noise_model.predict_variance(1, point, problem)
                self.objective_diagnostics["noise_prediction_rows"] += 1
                for offset, value in enumerate(extra, first_new):
                    paid_indices = indices.get(point, [])
                    if offset < len(paid_indices) and paid_indices[offset] in self.rejected_constraint_indices:
                        continue
                    evidence = update_noise_weights(evidence, previous_values=previous,
                        target=float(value[1]), predicted_noise=float(noise))
                    updates += bool(previous)
                    previous.append(float(value[1]))
            tail = noise_tail_factor(evidence, alpha=float(problem.alpha))
            result = {"evidence": evidence, "tail": tail, "new_variance_comparisons": updates}
            diagnostics = self.objective_diagnostics["boundary_noise"]
            diagnostics["weight_updates"] += updates
            diagnostics["component_evaluations"] += updates * len(evidence["scales"]) + tail["component_evaluations"]
            diagnostics["seconds"] += perf_counter() - started
            if live:
                self._terminal_noise_live = (live_key, result)
                diagnostics["posteriors"].append({"stage": len(self.history),
                    "weights": list(evidence["weights"]), "tail": deepcopy(tail),
                    "new_variance_comparisons": updates})
            return result

        def _boundary_noise_variance(self, pool, observations=None):
            posterior = self._boundary_noise_posterior(observations)
            if posterior is None:
                return None
            shape = np.asarray(self._terminal_noise_model.predict_variance_many(1, pool, problem), dtype=float)
            self.objective_diagnostics["noise_prediction_rows"] += len(pool)
            diagnostics = self.objective_diagnostics["boundary_noise"]
            diagnostics["certification_calls"] += 1
            diagnostics["certification_rows"] += len(pool)
            return np.maximum(shape * posterior["tail"]["variance_factor"], 1e-12)

        def _certification_result(self, mu_con, candidates, v_con=None, epistemic=None):
            # The native final recommender enters here, while its fantasies
            # enter _terminal_certificate_components. Both use the same mixture
            # tail before selecting a single recommended control.
            mixed = self._boundary_noise_variance(candidates)
            if mixed is not None:
                v_con = mixed
            result = super()._certification_result(mu_con, candidates, v_con, epistemic)
            if mixed is not None:
                self.objective_diagnostics["boundary_noise"]["last_recommendation"] = {
                    "stage": len(self.history), "pool_size": len(candidates),
                    "minimum_margin": float(np.min(result.margin)),
                    "certified_count": int(np.sum(np.asarray(result.margin) <= 0.))}
            return result

        def _terminal_certificate_components(self, gpr_models, variance_model, pool,
                *, task_ensemble=None, observations=None):
            result = super()._terminal_certificate_components(gpr_models, variance_model, pool,
                task_ensemble=task_ensemble, observations=observations)
            mixed = self._boundary_noise_variance(pool, observations) if task_ensemble is None and pool else None
            if mixed is not None:
                from statistics import NormalDist
                reference = np.asarray(self._terminal_noise_model.predict_variance_many(1, pool, problem), dtype=float)
                self.objective_diagnostics["noise_prediction_rows"] += len(pool)
                z = NormalDist().inv_cdf(1. - float(problem.alpha))
                original = np.asarray(result["margin"], dtype=float)
                location = original - z * np.sqrt(np.maximum(result["aleatoric"], 0.))
                result.update(plug_in_margin=original.copy(), noise_reference=reference,
                    noise_reference_margin=location + z * np.sqrt(np.maximum(reference, 0.)),
                    margin=location + z * np.sqrt(np.maximum(mixed, 0.)), aleatoric=mixed,
                    source="repeat_posterior_predictive_noise")
            return result

        def _exact_posterior_update_scores(self, candidates, terminal_pool,
                secondary_terminal_mode=None):
            original_started = perf_counter()
            scores = super()._exact_posterior_update_scores(candidates, terminal_pool,
                secondary_terminal_mode=secondary_terminal_mode)
            original_seconds = perf_counter() - original_started
            remaining = self.config.N - len(self.history)
            if not (self.objective_scope_revisions and self._terminal_goal_offers
                    and 1 <= remaining <= 4 and len(candidates) >= 2
                    and self.config.acquisition_mode == "exact_mc"
                    and self._effective_exact_kg_mc_samples() == 1):
                return scores
            raw = np.asarray(self._last_exact_kg_raw_scores, dtype=float)
            current = self._last_exact_kg_current_value
            if not (self.task_ensemble is None and secondary_terminal_mode is None
                    and np.asarray(current).ndim in (0, 1)):
                return scores
            vector_value = np.asarray(current).ndim == 1

            from .boundary_noise import scale_sensitive_repeats, scaled_predictive_innovation
            from .value_precision import compare_action_values
            started = perf_counter()
            costs = self.objective_diagnostics["value_precision"]
            allowance = (costs["evaluation_budget"] - costs["evaluations"]) // remaining
            event = {"stage": len(self.history) + 1, "remaining_calls": remaining,
                "allowance": allowance, "initial_raw": raw.tolist(),
                "candidate_points": [list(map(int, point)) for point in candidates],
                "initial_clipped": np.asarray(scores).tolist(),
                "initial_expected_terminal": np.asarray(self._last_exact_kg_expected_values).tolist(),
                "original_mc1_seconds": original_seconds,
                "current_terminal_value": np.asarray(current).tolist(), "terminal_pool_size": len(terminal_pool),
                "terminal_value_contract": "certified_lexicographic" if vector_value else "scalar_goal",
                "selection_semantics": "policy_indicator", "tie_forecast_rows": 0,
                "score_one_calls": 0, "terminal_value_calls": 0, "terminal_pool_rows": 0}
            actions = []
            for point, initial in zip(candidates, raw):
                forecast = self._objective_forecast(_normalized(point,
                    problem.basis.feature_names, *problem.int_bounds()))
                count = sum(tuple(control) == tuple(point) for control, _ in self.history)
                actions.append({"point": list(map(int, point)), "initial_raw": float(initial),
                    "feasible": forecast["feasible"], "posterior_value": forecast["objective_mean"],
                    "replicate_count": count, "prediction": forecast,
                    "tie_break": [not forecast["feasible"], forecast["objective_mean"], count,
                                  *map(int, point)]})
                event["tie_forecast_rows"] += 1

            noise_evidence = None
            # Four paired blocks, for every action and every noise hypothesis,
            # must fit the same existing terminal valuation budget.
            if allowance >= 8 * 4 * len(actions) and any(action["replicate_count"] for action in actions):
                posterior = self._boundary_noise_posterior()
                proposed = posterior["evidence"] if posterior is not None else None
                if proposed is not None:
                    from statistics import NormalDist
                    views = [{**action, "prediction": {**action["prediction"],
                        "constraint_noise": action["prediction"].get("constraint_noise_reference",
                                                                    action["prediction"]["constraint_noise"]),
                        "constraint_margin": action["prediction"].get("constraint_margin_reference",
                                                                     action["prediction"]["constraint_margin"])}}
                        for action in actions]
                    sensitive = scale_sensitive_repeats(views, proposed,
                        chance_z=NormalDist().inv_cdf(1. - float(problem.alpha)))
                    if sensitive:
                        noise_evidence = {**proposed, "sensitive_repeats": sensitive}
            clipped = np.max(raw) < 0. and np.all(np.asarray(scores) == 0.)
            if noise_evidence is None and not clipped:
                return scores
            scales = noise_evidence["scales"] if noise_evidence is not None else [1.]
            event["noise_evidence"] = noise_evidence
            event["noise_quadrature_points"] = len(scales)
            if noise_evidence is not None:
                for action in actions:
                    point = tuple(action["point"])
                    action["predictive_noise"] = {
                        "constraint_noise": float(self.variance_model.predict_variance(1, point, problem)),
                        "constraint_variance": float(self.gpr[1].posterior_var(point))}
                    action["noise_shape_ratio"] = (float(self._terminal_noise_model.predict_variance(1, point, problem))
                                                    / max(action["predictive_noise"]["constraint_noise"], 1e-12))
                    self.objective_diagnostics["noise_prediction_rows"] += 2
                    self.objective_diagnostics["epistemic_prediction_rows"] += 1

            def score(action, innovations):
                values = []
                for innovation in innovations:
                    gains = []
                    for scale in scales:
                        draw = (scaled_predictive_innovation(innovation, action["predictive_noise"],
                                                            scale * action["noise_shape_ratio"])
                                if noise_evidence is not None else innovation)
                        # Unknown noise changes the sampled label. The native
                        # learner receives neither that noise scale nor any
                        # evaluator truth; ordinary GP/HVD updates decide the
                        # same terminal objective and feasibility loss.
                        result = self._exact_posterior_update_score_one(action["point"],
                            np.asarray([draw]), terminal_pool, current,
                            return_diagnostics=True)
                        event["score_one_calls"] += 1
                        gains.append(result["component_gain"] if vector_value else float(result["raw_score"]))
                    weights = noise_evidence["weights"] if noise_evidence is not None else [1.]
                    average = np.asarray(weights) @ np.asarray(gains)
                    values.append(average.tolist() if vector_value else float(average))
                return values

            self._value_precision_round = event
            try:
                from .terminal_value_precision import compare_terminal_values
                compare = compare_terminal_values if vector_value else compare_action_values
                comparison = compare(actions, score,
                    seed=int(self.config.seed) + 1700047 * (len(self.history) + 1),
                    max_evaluations=allowance // len(scales))
            finally:
                self._value_precision_round = None
            event.update(comparison)
            event["innovation_evaluations"] = comparison["evaluations"]
            event["evaluations"] = comparison["evaluations"] * len(scales)
            if vector_value:
                for row in comparison["action_statistics"]:
                    row["mean_raw"] = (self._terminal_diagnostic_gain(current,
                        np.asarray(current) - row["mean_gain"]) if row["mean_gain"] is not None else None)
            by_point = {tuple(row["point"]): row for row in comparison["action_statistics"]}
            estimates = [by_point[tuple(point)]["mean_raw"] for point in candidates]
            if comparison["evaluations"]:
                self._last_exact_kg_raw_scores = np.asarray(estimates)
                gains = np.asarray([by_point[tuple(point)]["mean_gain"] for point in candidates]
                                   if vector_value else estimates)
                self._last_exact_kg_expected_values = np.asarray(current) - gains
                self._last_exact_kg_component_gains = gains
            selected = tuple(comparison["selected_point"])
            indicator = np.asarray([float(tuple(point) == selected) for point in candidates])
            # The native run uses argmax here. These are policy indicators;
            # the actual KG estimates remain in the raw and expected fields.
            event["selected_raw_value"] = by_point[selected]["mean_raw"]
            event["selected_tie_break"] = by_point[selected]["tie_break"]
            event["policy_indicator"] = indicator.tolist()
            event["seconds"] = perf_counter() - started
            costs["evaluations"] += event["evaluations"]
            for name in ("tie_forecast_rows", "score_one_calls", "terminal_value_calls", "terminal_pool_rows", "seconds"):
                costs[name] += event[name]
            costs["rounds"].append(event)
            return indicator

        def _register_objective_request(self, request, point):
            purpose = request["purpose"]
            self._recovery_control_purpose = purpose
            self.objective_diagnostics[f"{purpose}_requests"] += 1
            self.objective_diagnostics["requests"].append({
                **deepcopy(request), "point": list(point), "stage": len(self.history) + 1})

        def _objective_evidence(self):
            return [{"observation": {
                        "features": _normalized(point, problem.basis.feature_names, *problem.int_bounds()),
                        "target": float(output[1])},
                     "context": {"objective": float(output[0]), "global_index": index,
                                 "point": list(point), "stage": index + 1},
                     **({"observation_status": "disputed"}
                        if index in self.rejected_constraint_indices else {})}
                    for index, (point, output) in enumerate(self.history)]

        def _consider_constraint_applicability(self):
            """A goal counterexample makes the constraint scope worth testing."""
            remaining = self.config.N - len(self.history)
            if remaining < 9:
                return
            state = self.recovery.state
            if state["phase"] not in {"monitoring", "collecting", "validating", "unresolved"}:
                return
            research = self.objective_research.state["scope_research"]
            if research is None:
                return
            start = research["scope_start"]
            if start in self.joint_evidence["checked_goal_scopes"]:
                return
            if state["phase"] != "monitoring" and state["episodes"][-1]["scope_start"] >= start:
                return
            grouped = {}
            for index in range(self.active_scope_start, min(start, len(self.history))):
                if index not in self.rejected_constraint_indices:
                    point, output = self.history[index]
                    grouped.setdefault(tuple(point), []).append(
                        {"global_index": index, "target": float(output[1])})
            if not grouped:
                return
            compute = 2 * len(grouped)
            if state["compute_budget"] - state["costs"]["compute_cost"] < compute:
                return
            points = list(grouped)
            means = [model.posterior_mean_many(points) for model in self.constraint_models.values()]
            state["costs"]["compute_cost"] += compute
            self.joint_evidence["prediction_rows"] += compute
            # Both current predictors drifting from a paid old reading identifies
            # a useful repeat even when they already assimilated the new regime.
            departures = [min(abs(float(values[i]) - np.mean([row["target"] for row in grouped[point]]))
                              for values in means) for i, point in enumerate(points)]
            selected = max(range(len(points)), key=departures.__getitem__)
            point = points[selected]
            features = _normalized(point, problem.basis.feature_names, *problem.int_bounds())
            if self.recovery.start_applicability_probe(features, grouped[point],
                    goal_scope_start=start, context={"stage": len(self.history),
                        "remaining_calls": remaining}):
                self.joint_evidence["checked_goal_scopes"].append(start)
                self.joint_evidence["applicability_requests"].append({
                    "stage": len(self.history) + 1, "goal_scope_start": start,
                    "point": list(point), "predicted_departure": departures[selected],
                    "reference_indices": [row["global_index"] for row in grouped[point]],
                    "remaining_calls": remaining})

        def _objective_forecast(self, features):
            started = perf_counter()
            values = [features[name] for name in problem.basis.feature_names]
            point = tuple(map(int, problem.continuous_to_int(values)))
            certificate = self._terminal_certificate_components(
                self.gpr, self.variance_model, [point], observations=self.observations)
            result = {
                "features": _normalized(point, problem.basis.feature_names, *problem.int_bounds()),
                "objective_mean": float(certificate["objective"][0]),
                "objective_variance": float(self.gpr[0].posterior_var(point)),
                "objective_noise": float(self.variance_model.predict_variance(0, point, problem)),
                "constraint_mean": float(certificate["mu_con"][0]),
                "constraint_variance": float(certificate["epistemic"][0]),
                "constraint_noise": float(certificate["aleatoric"][0]),
                "constraint_margin": float(certificate["margin"][0]),
                "feasible": bool(certificate["margin"][0] <= 0.),
                **({"constraint_noise_reference": float(certificate["noise_reference"][0]),
                    "constraint_margin_reference": float(certificate["noise_reference_margin"][0]),
                    "constraint_plug_in_margin": float(certificate["plug_in_margin"][0])}
                   if "noise_reference" in certificate else {}),
            }
            self.objective_diagnostics["forecast_rows"] += 1
            self.objective_diagnostics["mean_prediction_rows"] += 2
            self.objective_diagnostics["epistemic_prediction_rows"] += 2
            self.objective_diagnostics["noise_prediction_rows"] += 2
            self.objective_diagnostics["forecast_seconds"] += perf_counter() - started
            return result

        def _condition_constraint_readout(self, model, history):
            from .constraint_retention import condition_coefficient_readout
            started = perf_counter()
            fitted = condition_coefficient_readout(model, history,
                max(float(getattr(problem, "sigma_level", 0.)) ** 2, 1e-6))
            self.constraint_retention["paid_design_fits"] += int(fitted)
            self.constraint_retention["paid_design_fit_records"] += len(history) if fitted else 0
            self.constraint_retention["paid_design_unidentified"] += int(not fitted)
            self.constraint_retention["paid_design_seconds"] += perf_counter() - started
            return fitted

        def _fit_initial_belief(self, samples):
            started = perf_counter()
            for record in records:
                point = record["point"]
                output = np.asarray(record["output"], dtype=float)
                self.observations.setdefault(point, []).append(output)
                self.history.append((point, output))
                problem.loaded_observation_count += 1
            for index, model in enumerate(self.gpr):
                basis = model.basis_map
                if basis is not None and hasattr(basis, "fit_from_observations"):
                    basis.fit_from_observations(self.observations, output_index=index)
                self._rebuild_gpr_from_history(index, replay_sequential=False)
            self._configure_hvd_source_task_posterior(self.variance_model, self.gpr)
            # HVD itself iterates every replicate in observations[point]; its
            # point inventory must therefore contain each control only once.
            unique_points = list(dict.fromkeys(samples))
            self.variance_model.initialize(unique_points, self.observations, self.gpr, self.problem)
            if self.adoption_policy is not None:
                shadow_started = perf_counter()
                self.gpr[1] = self.constraint_models["learned"]
                self._rebuild_gpr_from_history(1, replay_sequential=False)
                shadow_variance = self.constraint_variances["learned"]
                self._configure_hvd_source_task_posterior(shadow_variance, self.gpr)
                shadow_variance.initialize(unique_points, self.observations, self.gpr, self.problem)
                self.gpr[1] = self.constraint_models["default"]
                self.adoption_diagnostics["additional_model_fits"] += 1
                self.adoption_diagnostics["additional_fit_records"] += len(records)
                self.adoption_diagnostics["additional_model_fit_seconds"] += perf_counter() - shadow_started
            self._initialize_task_ensemble(unique_points)
            self._initialize_certification_recheck_targets(unique_points)
            self._last_initial_belief_timing = {"load_paid_observations": perf_counter() - started,
                                               "total": perf_counter() - started}

        def _simulate_and_store(self, point):
            if self._terminal_goal_offers:
                offers = self._terminal_goal_offers
                self._terminal_goal_offers = {}
                chosen = offers.get(tuple(map(int, point)))
                if chosen is not None:
                    started = perf_counter()
                    request = self.objective_research.select_terminal_offer(chosen, stage=len(self.history))
                    self._register_objective_request(request, point)
                    self.objective_diagnostics["controller_seconds"] += perf_counter() - started
                    self.objective_diagnostics["terminal_selected_offers"] += 1
                else:
                    # Native recheck/finalist policies may select another paid
                    # action. None of the unpurchased offers becomes a request.
                    self.objective_diagnostics["terminal_outside_offer_actions"] += 1
                repeated = bool(self.observations.get(tuple(point)))
                self.objective_diagnostics["terminal_repeat_actions" if repeated else "terminal_new_actions"] += 1
                self.objective_diagnostics["terminal_choices"].append({
                    "stage": len(self.history) + 1, "point": list(map(int, point)),
                    "purpose": self._recovery_control_purpose,
                    "selected_offer": chosen is not None, "action_kind": "replicate" if repeated else "new",
                    "terminal_pool_size": len(self._last_terminal_pool),
                    "offered_candidate_count": len(offers)})
            if self.adoption_policy is None:
                return super()._simulate_and_store(point)
            started = perf_counter()
            before = self.active_constraint_basis
            shadow = "learned" if before == "default" else "default"
            models = self.constraint_models
            means = {name: float(model.posterior_mean(point)) for name, model in models.items()}
            shadow_model = models[shadow]
            shadow_variance = self.constraint_variances[shadow]
            pending = {
                "point": tuple(int(value) for value in point),
                "selected_before": before, "shadow": shadow,
                "means": means,
                "shadow_mean": [float(self.gpr[0].posterior_mean(point)), means[shadow]],
                "shadow_epistemic": [float(self.gpr[0].posterior_var(point)),
                                     float(shadow_model.posterior_var(point))],
                "shadow_sigma2": [float(shadow_variance.predict_variance(i, point, self.problem))
                                  for i in range(2)],
            }
            noise = float(self.variance_model.predict_variance(1, point, self.problem))
            components = getattr(models["learned"], "_finite_mixture_components", [])
            if components:
                pending["retained_prediction"] = {
                    "stage": len(self.history) + 1, "point": list(point),
                    "scope_start": self.active_scope_start,
                    "component_names": list(models["learned"]._finite_mixture_component_names),
                    "means": [float(component.posterior_mean(point)) for component in components],
                    "weights_before": models["learned"]._finite_mixture_weights.tolist()}
                self.constraint_retention["prediction_rows"] += len(components)
            if self._has_rejected_constraint(point):
                pending["active_epistemic"] = float(models[before].posterior_var(point))
                pending["active_variance_before"] = deepcopy(self.variance_model)
                self.recovery_diagnostics["filtered_epistemic_predictions"] += 1
            if self._has_previous_objective(point):
                pending["objective_variance_before"] = deepcopy(self.variance_model)
            self.adoption_diagnostics["paired_prediction_rows"] += 2
            self.adoption_diagnostics["comparison_seconds"] += perf_counter() - started
            candidate_mean = None
            candidate_predictions = {}
            features = None
            if self.recovery is not None:
                features = _normalized(point, problem.basis.feature_names, *problem.int_bounds())
                shared_started = perf_counter()
                forecast_seconds = self.objective_diagnostics["forecast_seconds"]
                shared = self.objective_research.prepare_shared_control(
                    self._objective_evidence(), self._objective_forecast, features,
                    scope_start=self.active_scope_start, stage=len(self.history),
                    remaining_calls=self.config.N - len(self.history),
                    purpose=self._recovery_control_purpose)
                self.joint_evidence["shared_predictions"] += (
                    shared is not None and self.objective_research.state["shared_pending"] is not None)
                self.objective_diagnostics["controller_seconds"] += max(0., perf_counter() - shared_started
                    - (self.objective_diagnostics["forecast_seconds"] - forecast_seconds))
                started = perf_counter()
                candidates = self.recovery.candidates
                for candidate_id, candidate in candidates.items():
                    candidate_predictions[candidate_id] = float(candidate.predict(Observation(features, 0.)))
                preferred = self.recovery.candidate
                candidate_mean = next((candidate_predictions[candidate_id]
                    for candidate_id, candidate in candidates.items() if candidate == preferred), None)
                self.recovery_diagnostics["candidate_prediction_rows"] += len(candidate_predictions)
                self.recovery_diagnostics["candidate_prediction_seconds"] += perf_counter() - started
                self._prepare_objective_replacement(point)
            output = super()._simulate_and_store(point)
            if self.objective_research is not None:
                started = perf_counter()
                self.objective_replacement.observe(features, float(output[0]),
                    global_index=len(self.history) - 1)
                self.objective_research.observe(features, list(map(float, output)),
                    context={"stage": len(self.history), "global_index": len(self.history) - 1,
                             "purpose": self._recovery_control_purpose})
                self.objective_diagnostics["observations"] += 1
                self.objective_diagnostics["controller_seconds"] += perf_counter() - started
            started = perf_counter()
            comparison_state = self.adoption_policy.comparison_snapshot() if self.recovery is not None else None
            selected = self.adoption_policy.observe(
                means["default"], means["learned"], float(output[1]),
                noise_variance=noise,
                context={"point": list(pending["point"]), "stage": len(self.history),
                         "purpose": self._recovery_control_purpose})
            if selected not in models:
                raise ValueError("adoption policy must select default or learned")
            self.adoption_diagnostics["policy_observations"] += 1
            self.adoption_diagnostics["policy_seconds"] += perf_counter() - started
            if self.recovery is not None:
                started = perf_counter()
                self.recovery.observe(
                    Observation(features, float(output[1])), means["default"], means["learned"],
                    candidate_mean=candidate_mean, candidate_predictions=candidate_predictions,
                    noise_variance=noise,
                    context={"stage": len(self.history), "global_index": len(self.history) - 1,
                             "point": list(pending["point"]), "objective": float(output[0]),
                             "purpose": self._recovery_control_purpose,
                             "remaining_calls": self.config.N - len(self.history)})
                self.recovery_diagnostics["recovery_observations"] += 1
                self.recovery_diagnostics["controller_seconds"] += perf_counter() - started
                if (self._recovery_control_purpose not in {"confirmation", "applicability_confirmation"}
                        and self.recovery.control_purpose in {"confirmation", "applicability_confirmation"}):
                    snapshot_started = perf_counter()
                    self._confirmation_predictors = {
                        "global_index": len(self.history) - 1,
                        "models": {name: self._clone_gpr_for_exact_kg(model)
                                   for name, model in models.items()},
                        "variances": deepcopy(self.constraint_variances),
                        "comparison_state": comparison_state,
                    }
                    self.recovery_diagnostics["predictor_snapshots"] += 1
                    self.recovery_diagnostics["predictor_snapshot_seconds"] += perf_counter() - snapshot_started
            pending.update({"output": output, "selected_after": selected,
                            "noise_variance": noise, "candidate_mean": candidate_mean,
                            "candidate_predictions": candidate_predictions})
            self._pending_adoption = pending
            return output

        @staticmethod
        def _restore_constraint_variance(variance, saved, output_index=1):
            # OrthogonalHVD stores its learned statistics in output-indexed
            # dictionaries. Restore channel 1 while keeping both real objective
            # observations and every output-0 statistic from the running HVD.
            for name, values in saved.__dict__.items():
                if isinstance(values, dict) and output_index in values:
                    getattr(variance, name)[output_index] = deepcopy(values[output_index])

        def _has_previous_objective(self, point):
            point = tuple(map(int, point))
            lower = min(self.active_objective_scope_start, self.active_scope_start)
            upper = max(self.active_objective_scope_start, self.active_scope_start)
            return lower != upper and any(
                control == point for control, output in
                self.history[lower:upper])

        def _objective_values(self, point):
            point = tuple(map(int, point))
            return [float(output[0]) for control, output in self.history[self.active_objective_scope_start:]
                    if control == point]

        def _has_rejected_constraint(self, point):
            point = tuple(map(int, point))
            return any(index >= self.active_scope_start and self.history[index][0] == point
                       for index in self.rejected_constraint_indices)

        def _constraint_values(self, point):
            point = tuple(map(int, point))
            if not self.rejected_constraint_indices:
                return [float(output[1]) for output in self.observations.get(point, [])]
            return [float(output[1]) for index, (control, output) in enumerate(self.history)
                    if index >= self.active_scope_start and index not in self.rejected_constraint_indices
                    and control == point]

        def _update_constraint_variance(self, variance, model, point, target, mean, epistemic, values):
            return variance.update(1, np.asarray(point, dtype=int), target, mean, model, self.problem,
                epistemic_var=epistemic,
                replicate_variance=float(np.var(values, ddof=1)) if len(values) >= 2 else None,
                replicate_count=len(values))

        def _repair_constraint_observation(self, repair, pending, stage):
            started = perf_counter()
            saved = self._confirmation_predictors
            anchor_index = int(repair["global_index"])
            if saved is None or saved["global_index"] != anchor_index:
                raise ValueError("observation repair requires the pre-counterexample predictors")
            self.rejected_constraint_indices.add(anchor_index)
            point, output = pending["point"], pending["output"]
            means, noise = {}, {}
            values = self._constraint_values(point)
            for name, model in saved["models"].items():
                variance = self.constraint_variances[name]
                self._restore_constraint_variance(variance, saved["variances"][name])
                means[name] = float(model.posterior_mean(point))
                epistemic = float(model.posterior_var(point))
                noise[name] = float(variance.predict_variance(1, point, self.problem))
                self.recovery_diagnostics["repair_mean_predictions"] += 1
                self.recovery_diagnostics["repair_epistemic_predictions"] += 1
                self.recovery_diagnostics["repair_noise_predictions"] += 1
                model.update(point, output[1], noise[name])
                self._configure_hvd_source_task_posterior(variance, [self.gpr[0], model])
                self._update_constraint_variance(variance, model, point, output[1],
                                                means[name], epistemic, values)
            self.constraint_models = saved["models"]
            selected = self.adoption_policy.repair_observation(
                int(repair["observation_index"]), int(repair["confirmation_index"]),
                comparison_state=saved["comparison_state"],
                default_mean=means["default"], learned_mean=means["learned"],
                noise_variance=noise[saved["comparison_state"]["selected"]])
            self.active_constraint_basis = selected
            self.gpr[1] = self.constraint_models[selected]
            self.variance_model = self.constraint_variances[selected]
            record = {**deepcopy(repair), "stage": int(stage), "selected_after": selected,
                      "replayed_predictions": means, "replayed_noise": noise,
                      "total_paid_records": len(self.history),
                      "retained_constraint_records": len(self.history) - self.active_scope_start
                                                     - sum(i >= self.active_scope_start
                                                           for i in self.rejected_constraint_indices)}
            self.observation_repairs.append(record)
            self.recovery.acknowledge_observation_repair()
            self.recovery_diagnostics["observation_repairs"] += 1
            self.recovery_diagnostics["repair_gp_updates"] += 2
            self.recovery_diagnostics["repair_hvd_updates"] += 2
            self.recovery_diagnostics["observation_repair_seconds"] += perf_counter() - started

        def _exact_primary_fantasy_update(self, x_arr, y, existing_observations,
                                         mu_before, sigma2_before, epistemic_before):
            models, variance = super()._exact_primary_fantasy_update(
                x_arr, y, existing_observations, mu_before, sigma2_before, epistemic_before)
            if self._has_rejected_constraint(x_arr):
                self._restore_constraint_variance(variance, self.variance_model)
                self._configure_hvd_source_task_posterior(variance, models)
                self._update_constraint_variance(variance, models[1], x_arr, y[1], mu_before[1],
                    epistemic_before[1], self._constraint_values(x_arr) + [float(y[1])])
                self.recovery_diagnostics["filtered_fantasy_updates"] += 1
            if self._has_previous_objective(x_arr):
                self._restore_constraint_variance(variance, self.variance_model, output_index=0)
                values = self._objective_values(x_arr) + [float(y[0])]
                variance.update(0, x_arr, y[0], mu_before[0], models[0], self.problem,
                    epistemic_var=epistemic_before[0],
                    replicate_variance=float(np.var(values, ddof=1)) if len(values) >= 2 else None,
                    replicate_count=len(values))
                self.objective_diagnostics["scope_filtered_fantasy_updates"] += 1
            return models, variance

        def _rebase_objective_scope(self, revision, stage):
            started = perf_counter()
            start = int(revision["scope_start"])
            history = self.history[start:]
            observations = {}
            for point, output in history:
                observations.setdefault(point, []).append(output)
            program = Program.from_dict(revision["program"])
            basis = HegelProgramBasis(program, problem.basis.feature_names, problem.int_bounds())
            model = ParametricGPR(problem.d, self.config.lambda_i, self.config.prior_var,
                normalize_func=problem.normalize, basis_map=basis,
                numeric_backend=self.config.numeric_backend,
                numeric_backend_device=self.config.numeric_backend_device,
                torch_dtype=self.config.torch_dtype, torch_min_rows=self.config.torch_min_rows)
            fitting = copy(self)
            fitting.history, fitting.observations = history, observations
            fitting.config = replace(self.config, n0=len(history))
            fitting.gpr = [model, self.gpr[1]]
            fitting._rebuild_gpr_from_history(0, replay_sequential=False)
            for name, variance in self.constraint_variances.items():
                clean = deepcopy(self._scope_variance_template)
                fitting.gpr[1] = self.constraint_models[name]
                fitting._configure_hvd_source_task_posterior(clean, fitting.gpr)
                clean.initialize(list(observations), observations, fitting.gpr, problem)
                self._restore_constraint_variance(variance, clean, output_index=0)
            # OLS above estimates the residual/HVD scales. Its fitted
            # coefficient covariance is not another likelihood contribution.
            # Conditional on the frozen learned feature, start from the
            # configured calibration prior and consume each paid label once.
            posterior_started = perf_counter()
            deviation_variance = float(model.lambda_i)
            model.set_parametric_prior(np.asarray([0., 1.]), deviation_variance,
                                       self.config.prior_var)
            for point, output in history:
                noise = self.variance_model.predict_variance(0, point, problem)
                model.update(point, float(output[0]), noise)
            posterior_seconds = perf_counter() - posterior_started
            self.objective_diagnostics["head_posterior_updates"] += len(history)
            self.objective_diagnostics["head_posterior_noise_predictions"] += len(history)
            self.objective_diagnostics["head_posterior_seconds"] += posterior_seconds
            self.gpr[0] = model
            self.active_objective_scope_start = start
            self.objective_scope_revisions.append({**deepcopy(revision), "stage": int(stage),
                "scope_start": start, "fit_indices": list(range(start, len(self.history))),
                "fit_records": len(history), "total_paid_records": len(self.history),
                "posterior_updates": len(history), "posterior_seconds": posterior_seconds,
                "posterior_prior": {"mean": [0., 1.], "variance": float(self.config.prior_var),
                                    "deviation_variance": deviation_variance}})
            self.objective_research.acknowledge_revision()
            self.objective_replacement.finish(stage, "supported_objective_program")
            self.objective_diagnostics["head_refits"] += 1
            self.objective_diagnostics["head_fit_records"] += len(history)
            self.objective_diagnostics["head_hvd_initializations"] += 2
            self.objective_diagnostics["head_refit_seconds"] += perf_counter() - started

        def _constraint_decision_view(self):
            # These two legacy decision routines consume control-wise means
            # and constraint replication counts. Keep the mean of every actual
            # objective output while aggregating only retained constraints.
            decision = copy(self)
            decision.observations = dict(self.observations)
            for point, outputs in self.observations.items():
                if self._has_rejected_constraint(point):
                    objective_mean = float(np.mean([output[0] for output in outputs]))
                    decision.observations[point] = [np.asarray([objective_mean, target])
                                                    for target in self._constraint_values(point)]
            return decision

        def _observed_nominal_incumbent(self):
            if self.objective_research is not None:
                # Empirical winners remain available to objective research.
                # They no longer override the GP ranking with a single noisy
                # target while the original robust constraint rule stays active.
                return None
            if not self.rejected_constraint_indices:
                return super()._observed_nominal_incumbent()
            return SingleOLHKGAlgorithm._observed_nominal_incumbent(self._constraint_decision_view())

        def _recommendation_calibration_fit(self):
            if not self.rejected_constraint_indices:
                return super()._recommendation_calibration_fit()
            return SingleOLHKGAlgorithm._recommendation_calibration_fit(self._constraint_decision_view())

        def _fit_objective_model(self, history, observations):
            """The same prefix-fitting rule serves prediction and installation."""
            model = ParametricGPR(problem.d, self.config.lambda_i, self.config.prior_var,
                normalize_func=problem.normalize, basis_map=None,
                numeric_backend=self.config.numeric_backend,
                numeric_backend_device=self.config.numeric_backend_device,
                torch_dtype=self.config.torch_dtype, torch_min_rows=self.config.torch_min_rows)
            fitting = copy(self)
            fitting.history, fitting.observations = history, observations
            fitting.config = replace(self.config, n0=len(history))
            fitting.gpr = [model, self.gpr[1]]
            fitting._rebuild_gpr_from_history(0, replay_sequential=False)
            return model

        def _goal_candidate_tag(self, revision=None):
            research = revision or self.objective_research.state["scope_research"]
            if not research:
                return None
            model = research.get("model") if revision else research.get("candidate")
            return (research["scope_start"], model["training_end"],
                    *(f"interaction:{first}:{second}"
                      for first, second in model.get("interactions", ()))) if model else None

        def _objective_prefix(self, start):
            history = self.history[start:]
            observations = {}
            for point, output in history:
                observations.setdefault(point, []).append(output)
            return history, observations

        def _prepare_objective_replacement(self, point):
            replacement = self.objective_replacement
            if self.gpr[0].basis_map is not None:
                research = self.objective_research.state["scope_research"]
                if not replacement.state["confirmed"]:
                    failures = [row for row in self.objective_research.state["monitoring"][-4:]
                                if row["error"] > row["threshold"]]
                    trigger = research.get("trigger", []) if research else []
                    if not failures and not trigger:
                        return
                    # A failed program can reflect an inadequate representation,
                    # as well as a new mechanism. Compare the ordinary GP on the
                    # already accepted scope while goal research investigates the
                    # newer scope; future predictions decide between them.
                    start = self.active_objective_scope_start
                    replacement.confirm(start, len(self.history))
                    replacement.state["events"].append({
                        "action": "reopen_installed_objective_model", "stage": len(self.history),
                        "scope_start": start,
                        "counterexample_scope": (research["scope_start"] if trigger
                                                  else failures[0]["global_index"])})
            elif self.recovery.state["phase"] == "validating":
                start = max(self.recovery.state["episodes"][-1]["scope_start"],
                            self.active_objective_scope_start)
                replacement.propose(start, len(self.history))
            elif not replacement.state["confirmed"]:
                return
            features = _normalized(point, problem.basis.feature_names, *problem.int_bounds())
            # Coverage belongs to this proposed opponent, independently of the
            # objective candidate's earlier comparisons at the same control.
            if any(row["features"] == features for row in replacement.state["validation"]):
                return
            started = perf_counter()
            history, observations = self._objective_prefix(replacement.state["scope_start"])
            proposed = self._fit_objective_model(history, observations)
            candidate, tag = None, self._goal_candidate_tag()
            if tag is not None:
                from .decision_experiments import _predict
                candidate = {"tag": tag, "mean": _predict(problem.spec,
                    self.objective_research.state["scope_research"]["candidate"], features)}
            replacement.prepare(features, float(self.gpr[0].posterior_mean(point)),
                float(proposed.posterior_mean(point)), stage=len(self.history), candidate=candidate)
            self.objective_diagnostics["replacement_fits"] += 1
            self.objective_diagnostics["replacement_fit_records"] += len(history)
            self.objective_diagnostics["replacement_prediction_rows"] += 2 + (candidate is not None)
            self.objective_diagnostics["mean_prediction_rows"] += 2 + (candidate is not None)
            self.objective_diagnostics["replacement_seconds"] += perf_counter() - started

        def _rebase_goal_comparison(self, stage, start):
            tag = self._goal_candidate_tag()
            comparison = self.objective_replacement.comparisons(tag) if tag is not None else []
            self.objective_research.incumbent_rebuilt(
                stage=stage, scope_start=start, comparison=comparison)
            self.objective_diagnostics["replacement_installs"].append({
                "stage": int(stage), "scope_start": start, "prepared_comparisons": len(comparison)})
            self.objective_replacement.finish(stage, "supported_gp_replacement")

        def _consider_objective_replacement(self, stage):
            replacement = self.objective_replacement
            if not replacement.state["confirmed"]:
                return
            accepted = self.objective_research.accepted_revision
            decision = replacement.decide(stage=stage,
                accepted_candidate=self._goal_candidate_tag(accepted) if accepted else None)
            if not decision["replace"]:
                return
            started = perf_counter()
            start = replacement.state["scope_start"]
            history, observations = self._objective_prefix(start)
            model = self._fit_objective_model(history, observations)
            fitting = copy(self)
            fitting.history, fitting.observations = history, observations
            fitting.config = replace(self.config, n0=len(history))
            # Replace only output zero's uncertainty; the independently
            # accepted constraint models keep their current posterior state.
            for name, variance in self.constraint_variances.items():
                fitting.gpr = [model, self.constraint_models[name]]
                fresh = deepcopy(self._scope_variance_template)
                fitting._configure_hvd_source_task_posterior(fresh, fitting.gpr)
                fresh.initialize(list(observations), observations, fitting.gpr, problem)
                self._restore_constraint_variance(variance, fresh, output_index=0)
            self.gpr[0] = model
            self.active_objective_scope_start = start
            self._rebase_goal_comparison(stage, start)
            self.objective_diagnostics["replacement_install_fit_records"] += len(history)
            self.objective_diagnostics["replacement_install_seconds"] += perf_counter() - started

        def _rebase_scope(self, revision, stage):
            started = perf_counter()
            scope_start = int(revision["scope_start"])
            scope_history = self.history[scope_start:]
            preserve_objective = self.gpr[0].basis_map is not None
            if not preserve_objective:
                proposed_start = max(scope_start, self.active_objective_scope_start)
                self.objective_replacement.confirm(proposed_start, stage)
                accepted = self.objective_research.accepted_revision
                decision = self.objective_replacement.decide(stage=stage,
                    accepted_candidate=self._goal_candidate_tag(accepted) if accepted else None)
                preserve_objective = not decision["replace"]
            objective_start = (self.active_objective_scope_start if preserve_objective
                               else max(scope_start, self.active_objective_scope_start))
            objective_history = self.history[objective_start:]
            objective_observations = {}
            for point, output in objective_history:
                objective_observations.setdefault(point, []).append(output)
            if not scope_history or scope_start < self.active_scope_start:
                raise ValueError("revision must select a nonempty current data segment")
            scope_observations = {}
            for point, output in scope_history:
                scope_observations.setdefault(point, []).append(output)
            constraint_history = [record for index, record in enumerate(self.history)
                                  if index >= scope_start and index not in self.rejected_constraint_indices]
            constraint_observations = {}
            for point, output in constraint_history:
                constraint_observations.setdefault(point, []).append(output)
            program = revision["program"]
            new_basis = HegelProgramBasis(program, problem.basis.feature_names, problem.int_bounds())

            def fresh_model(basis=None):
                return ParametricGPR(problem.d, self.config.lambda_i, self.config.prior_var,
                    normalize_func=problem.normalize, basis_map=basis,
                    numeric_backend=self.config.numeric_backend,
                    numeric_backend_device=self.config.numeric_backend_device,
                    torch_dtype=self.config.torch_dtype, torch_min_rows=self.config.torch_min_rows)

            objective = (self.gpr[0] if preserve_objective else
                         self._fit_objective_model(objective_history, objective_observations))
            constraints = {"default": fresh_model(), "learned": fresh_model(new_basis)}
            # Use the original SC-KG fitting method with an explicit data view.
            # The running optimizer keeps its complete paid history and N/n0.
            fitting = copy(self)
            fitting.history = objective_history
            fitting.observations = objective_observations
            fitting.config = replace(self.config, n0=len(objective_history))
            fitting.gpr = [objective, constraints["default"]]
            variances = {}
            for name, model in constraints.items():
                fitting.gpr[1] = model
                fitting.history = constraint_history
                fitting.observations = constraint_observations
                fitting.config = replace(self.config, n0=len(constraint_history))
                fitting._rebuild_gpr_from_history(1, replay_sequential=False)
                if name == "learned":
                    from .constraint_retention import install_program_competition
                    retention_started = perf_counter()
                    self._condition_constraint_readout(model, constraint_history)
                    term_basis = HegelProgramTermBasis(new_basis)
                    coefficient_model = None
                    if term_basis.feature_dim and len(constraint_observations) > term_basis.feature_dim:
                        candidate = fresh_model(term_basis)
                        fitting.gpr[1] = candidate
                        fitting._rebuild_gpr_from_history(1, replay_sequential=False)
                        if self._condition_constraint_readout(candidate, constraint_history):
                            coefficient_model = candidate
                        self.constraint_retention["coefficient_fits"] += 1
                        self.constraint_retention["coefficient_fit_records"] += len(constraint_history)
                        self.recovery_diagnostics["scope_model_fits"] += 1
                        self.recovery_diagnostics["scope_fit_records"] += len(constraint_history)
                    model = install_program_competition(model, self._clone_gpr_for_exact_kg,
                                                       coefficient_model=coefficient_model)
                    constraints[name] = fitting.gpr[1] = model
                    self.constraint_retention["scopes"].append({
                        "stage": int(stage), "scope_start": scope_start,
                        "program": program.to_dict(),
                        "component_names": list(model._finite_mixture_component_names),
                        "initial_weights": model._finite_mixture_weights.tolist(),
                        "coefficient_dimension": coefficient_model.p if coefficient_model is not None else None})
                    self.constraint_retention["setup_seconds"] += perf_counter() - retention_started
                variance = deepcopy(self._scope_variance_template)
                fitting._configure_hvd_source_task_posterior(variance, fitting.gpr)
                variance.initialize(list(scope_observations), scope_observations, fitting.gpr, problem)
                if preserve_objective:
                    self._restore_constraint_variance(variance, self.constraint_variances[name], output_index=0)
                elif len(objective_history) != len(scope_history):
                    clean_objective = deepcopy(self._scope_variance_template)
                    fitting._configure_hvd_source_task_posterior(clean_objective, fitting.gpr)
                    clean_objective.initialize(list(objective_observations), objective_observations,
                                               fitting.gpr, problem)
                    self._restore_constraint_variance(variance, clean_objective, output_index=0)
                if len(constraint_history) != len(scope_history):
                    clean = deepcopy(self._scope_variance_template)
                    fitting._configure_hvd_source_task_posterior(clean, fitting.gpr)
                    clean.initialize(list(constraint_observations), constraint_observations, fitting.gpr, problem)
                    self._restore_constraint_variance(variance, clean)
                variances[name] = variance
            self.scope_revisions.append({
                "stage": int(stage), "previous_scope_start": self.active_scope_start,
                "scope_start": scope_start, "scope_records": len(scope_history),
                "constraint_scope_records": len(constraint_history),
                "objective_scope_start": objective_start, "objective_scope_records": len(objective_history),
                "preserved_objective_head": preserve_objective,
                "total_paid_records": len(self.history), "old_program": problem.basis.program.to_dict(),
                "program": program.to_dict(), "family_id": revision.get("family_id"),
                "kind": revision.get("kind"), "validation_mse": revision.get("validation_mse"),
            })
            problem.basis = new_basis
            self.active_scope_start = scope_start
            self.active_objective_scope_start = objective_start
            self.observations = scope_observations
            self.constraint_models, self.constraint_variances = constraints, variances
            self.gpr = [objective, constraints["learned"]]
            self.variance_model = variances["learned"]
            if not preserve_objective:
                self._rebase_goal_comparison(stage, objective_start)
            self.active_constraint_basis = "learned"
            self.adoption_policy.begin_scope(context={"stage": int(stage), "scope_start": scope_start},
                                            selected="learned")
            self.recovery.acknowledge_revision()
            self.objective_research.register_constraint_concept(
                program.to_dict(), scope_start=scope_start, stage=stage)
            self.recovery_diagnostics["scope_rebases"] += 1
            self.recovery_diagnostics["scope_model_fits"] += 2 + int(not preserve_objective)
            self.recovery_diagnostics["scope_fit_records"] += (2 * len(constraint_history)
                + (len(objective_history) if not preserve_objective else 0))
            self.recovery_diagnostics["scope_hvd_initializations"] += (2
                + 2 * (len(constraint_history) != len(scope_history))
                + 2 * (not preserve_objective and len(objective_history) != len(scope_history)))
            self.recovery_diagnostics["scope_rebase_seconds"] += perf_counter() - started

        def _update_posterior_dominance_incumbent(self, *, stage, reason):
            # Original run() calls this immediately after both active GP/HVD
            # updates, including the final paid observation. Do not switch in
            # _simulate_and_store: that would apply an old model's residual
            # and observation variance to the newly selected model.
            pending = self._pending_adoption
            if pending is not None:
                started = perf_counter()
                point, output = pending["point"], pending["output"]
                shadow = pending["shadow"]
                model = self.constraint_models[shadow]
                variance = self.constraint_variances[shadow]
                model.update(point, output[1], pending["shadow_sigma2"][1])
                self._configure_hvd_source_task_posterior(variance, [self.gpr[0], model])
                for i, updated_model in enumerate((self.gpr[0], model)):
                    values = self._constraint_values(point) if i == 1 else self._objective_values(point)
                    variance.update(
                        i, np.asarray(point, dtype=int), output[i], pending["shadow_mean"][i],
                        updated_model, self.problem,
                        epistemic_var=pending["shadow_epistemic"][i],
                        replicate_variance=float(np.var(values, ddof=1)) if len(values) >= 2 else None,
                        replicate_count=len(values))
                self.adoption_diagnostics["shadow_gp_updates"] += 1
                self.adoption_diagnostics["shadow_hvd_updates"] += 2
                self.adoption_diagnostics["shadow_update_seconds"] += perf_counter() - started
                if "active_variance_before" in pending:
                    active_variance = self.constraint_variances[pending["selected_before"]]
                    self._restore_constraint_variance(active_variance, pending["active_variance_before"])
                    self._update_constraint_variance(active_variance,
                        self.constraint_models[pending["selected_before"]], point, output[1],
                        pending["means"][pending["selected_before"]], pending["active_epistemic"],
                        self._constraint_values(point))
                    self.recovery_diagnostics["filtered_replicate_updates"] += 1
                if "objective_variance_before" in pending:
                    variance = self.constraint_variances[pending["selected_before"]]
                    self._restore_constraint_variance(variance, pending["objective_variance_before"], output_index=0)
                    values = self._objective_values(point)
                    variance.update(0, np.asarray(point), output[0], pending["shadow_mean"][0], self.gpr[0], problem,
                        epistemic_var=pending["shadow_epistemic"][0],
                        replicate_variance=float(np.var(values, ddof=1)) if len(values) >= 2 else None,
                        replicate_count=len(values))
                    self.objective_diagnostics["scope_filtered_updates"] += 1
                selected = pending["selected_after"]
                if "retained_prediction" in pending:
                    model = self.constraint_models["learned"]
                    self.constraint_retention["observations"].append({
                        **pending["retained_prediction"], "target": float(output[1]),
                        "weights_after": model._finite_mixture_weights.tolist(),
                        "purpose": self._recovery_control_purpose})
                    count = len(pending["retained_prediction"]["means"])
                    self.constraint_retention["component_updates"] += count
                    self.constraint_retention["comparison_rows"] += count
                self.active_constraint_basis = selected
                self.gpr[1] = self.constraint_models[selected]
                self.variance_model = self.constraint_variances[selected]
                self.adoption_events.append({
                    "stage": int(stage), "point": list(point),
                    "purpose": self._recovery_control_purpose,
                    "default_mean": pending["means"]["default"],
                    "learned_mean": pending["means"]["learned"],
                    "target": float(output[1]), "noise_variance": pending["noise_variance"],
                    "candidate_mean": pending["candidate_mean"],
                    "candidate_predictions": pending["candidate_predictions"],
                    "selected_before": pending["selected_before"], "selected_after": selected,
                })
                self._pending_adoption = None
                repair = getattr(self.recovery, "observation_repair", None)
                if repair is not None:
                    self._repair_constraint_observation(repair, pending, stage)
                    self.adoption_events[-1]["selected_after_observation_repair"] = self.active_constraint_basis
                if self._recovery_control_purpose in {"confirmation", "applicability_confirmation"}:
                    self._confirmation_predictors = None
                if self.recovery is not None and self.recovery.accepted_revision is not None:
                    self._rebase_scope(self.recovery.accepted_revision, stage)
                    self.adoption_events[-1]["selected_after_recovery"] = self.active_constraint_basis
                if self.objective_replacement is not None:
                    self._consider_objective_replacement(stage)
                if self.objective_research is not None and self.objective_research.accepted_revision is not None:
                    self._rebase_objective_scope(self.objective_research.accepted_revision, stage)
            return super()._update_posterior_dominance_incumbent(stage=stage, reason=reason)

    return ResearchInitializedSCAlgorithm()
