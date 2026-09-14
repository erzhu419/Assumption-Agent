"""Research measured coordinates, history and representation through one interface.

The laboratory supplies controllable bounds and instruments, never a mechanism
label. All model selection uses purchased trajectories; the selected executable
program is frozen before a further trajectory can support its adoption.
"""

from dataclasses import asdict, replace
from copy import deepcopy
from functools import partial
from itertools import combinations, combinations_with_replacement, islice
import json
from random import Random

from .continuous_state import _linear_fit, _sum_terms, continuous_state_candidates
from .evaluation import initial_prediction, loss, predict
from .language import Definition, Program
from .memory import KnowledgeLibrary, _object_schema
from .meta_prior_claims import compile_claims
from .meta_prior_research import _sources
from .models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from .research_graph import ResearchGraph
from .scale_bindings import aggregate_role_binding, bind_measurement_arguments, control_history_depth


def _view(rows, names):
    allowed = set(names)
    return [Observation({key: value for key, value in row.features.items() if key in allowed},
                        row.target, tuple({key: value for key, value in past.items()
                                           if key in allowed or key == "__target__"}
                                          for past in row.history)) for row in rows]


def _history_terms(names, depth):
    # Empty prehistory is a hypothesis about a reset experiment. The guards
    # make that hypothesis executable, and validation includes startup rows.
    return [Program("if", children=(Program("available", lag=lag),
                Program("lag", name=name, lag=lag), Program("const", value=0.)))
            for lag in range(1, depth + 1) for name in names]


def _terms(names, depth=0, quadratic=False):
    current = [Program("feature", name=name) for name in names]
    products = [Program("mul", children=pair)
                for pair in combinations_with_replacement(current, 2)] if quadratic else []
    return [*current, *products, *_history_terms(names, depth)]


def _fit(terms, rows, budget, kind, *, family="T02", q=.5):
    terms = list(dict.fromkeys(terms))
    if not rows or not budget.spend(len(rows) * (len(terms) + 2)):
        return None
    columns = [[term.predict(row) for row in rows] for term in terms]
    coefficients = _linear_fit(columns, [row.target for row in rows])
    program = _sum_terms(terms, coefficients[1:], coefficients[0]).simplified()
    return {"candidate": Candidate(f"unified:{kind}:{budget.spent}", program,
                                    "unified_research", kind),
            "kind": kind, "family_id": family,
            "training_loss": loss(program, rows, initial_prediction(rows, q), q)}


class UnifiedResearchAgent:
    """A bounded common research loop, including paid joint revision and reuse."""

    def __init__(self, config=None, *, measurement_budget=768, library=None):
        self.config = config or ResearchConfig(observation_budget=192,
            compute_budget=120000, minimum_improvement=1e-7, active_probes=False)
        self.measurement_budget = measurement_budget
        self.library = library if library is not None else KnowledgeLibrary()
        self._learn_instruments = False
        self._revise_inputs = False
        self._object_schemas = {}
        self._object_models = {}
        self._object_competitions = {}

    def _spec(self, laboratory, channels=()):
        spec = laboratory.spec_for(channels)
        model = self._object_models.get(tuple(channels))
        if model is not None:
            spec = replace(spec, context={**spec.context, "object_model": model})
        schema = self._object_schemas.get(tuple(channels))
        return replace(spec, context={**spec.context, "object_schema": json.dumps(schema)}) if schema else spec

    def _knowledge_spec(self, spec):
        if not _object_schema(spec) and spec.context.get("object_model"):
            from .partial_object_knowledge import partial_object_schema
            schema = partial_object_schema(spec.context["object_model"])
            if schema:
                spec = replace(spec, context={**spec.context, "object_schema": json.dumps(schema)})
        return spec

    def _object_induction(self, laboratory):
        views = getattr(laboratory, "measurement_views", ())
        return (self.config.strategy != "fixed" and bool(views)
                and hasattr(laboratory, "project_transitions")
                and not any(_object_schema(laboratory.spec_for(view)) for view in views))

    def _object_probe(self, spec, model, records, serial):
        from .object_interventions import propose_object_probe
        return propose_object_probe(spec, model, records, serial)

    def _research_objects(self, laboratory, channels, acquire, batches, budget, actions):
        """Discover a usable grouping from paid anonymous state transitions."""
        from .object_induction import infer_object_model
        from .instrument_models import rebind_instrument_model
        from .models import Probe
        raw_spec = laboratory.spec_for(channels)
        command_spec = laboratory.spec_for(())
        start = budget.spent
        initial_batch = len(batches)
        row_limit = min(192, self.config.observation_budget // 3)

        def records():
            evidence = []
            for batch in batches[initial_batch:]:
                for index, (point, row, transition) in enumerate(zip(batch["probe"]["sequence"],
                        batch["observations"], batch.get("transitions", []))):
                    evidence.append({"controls": dict(point), **transition, "target": row["target"],
                        "history": list(batch["probe"]["sequence"][:index]),
                        "probe_id": batch["probe"]["probe_id"], "position": index})
            return evidence

        previous = next((item for item in reversed(self.library.object_models)
            if item.get("status") != "counterexample" and item.get("groups")
            and set(item["raw_control_features"]) == set(command_spec.feature_names)
            and set(item["raw_sensor_features"]) == set(raw_spec.feature_names)), None)
        # Random simultaneous setups also identify nonlinear calibration that
        # an endpoint-only control sweep cannot distinguish from a linear map.
        if acquire(channels, "observe_anonymous_dependencies", rows_count=12) is None:
            return {"status": "unresolved", "complete": False, "schema": None, "reason": "observation_budget"}
        if previous is not None and previous.get("sensor_model"):
            witnesses = previous.get("membership_witnesses", [])
            if witnesses:
                acquire(channels, "check_retained_object_membership",
                    probe=Probe(f"object_membership_{len(batches)}", tuple(witnesses)))
            evidence = records()
            previous_context = {**raw_spec.context, "object_model": previous}
            if previous.get("schema"):
                previous_context["object_schema"] = json.dumps(previous["schema"])
            binding = rebind_instrument_model(previous["sensor_model"],
                replace(raw_spec, context=previous_context))
            errors = {}
            for name, program in binding.items():
                if not budget.spend(len(evidence)):
                    break
                errors[name] = sum((program.predict(Observation(row["controls"], row["target"]))
                    - row["before"][name]) ** 2 for row in evidence) / len(evidence)
            required = {group[role] for group in previous["groups"]
                        for role in ("q", "u", "x", "f") if role in group}
            stable = bool(errors) and required <= set(errors) and max(errors.values()) <= self.config.minimum_improvement
            if stable and previous.get("dependency_witnesses"):
                from .object_induction import validate_object_roles
                points = []
                for witness in previous["dependency_witnesses"]:
                    for point in witness["sequence"]:
                        if point not in points:
                            points.append(point)
                for offset in range(0, len(points), raw_spec.max_probe_length):
                    acquire(channels, "test_retained_role_dependencies", probe=Probe(
                        f"object_role_witness_{len(batches)}", tuple(points[offset:offset + raw_spec.max_probe_length])))
                role_check = validate_object_roles(previous, records(), budget)
                stable = role_check["status"] == "supported"
                actions.append({"action": "compare_retained_role_definitions",
                    "object_model_id": previous["model_id"], **role_check})
                if role_check["status"] == "counterexample":
                    previous["status"] = "counterexample"
                    previous.setdefault("counterexamples", []).append(role_check)
                    actions.append({"action": "reopen_object_dependencies", "object_model_id": previous["model_id"],
                        "reason": "observed_role_dependency_changed", "counterexamples": role_check["counterexamples"]})
            retained_alternatives = previous.get("schema_alternatives", {})
            reusable_ambiguity = (retained_alternatives.get("exhaustive")
                and len(retained_alternatives.get("schemas", [])) > 1
                and all(group.get("f") for group in previous["groups"]))
            if stable and reusable_ambiguity:
                actions.append({"action": "reuse_object_alternatives", "object_model_id": previous["model_id"],
                    "alternatives": len(retained_alternatives["schemas"]),
                    "reason": "current_role_witnesses_support_retained_partition_candidates"})
            if stable and not previous.get("complete") and not reusable_ambiguity:
                from .partial_object_knowledge import propose_partial_object_claims
                training = [Observation(row["features"], row["target"], tuple(row["history"]))
                            for row in batches[initial_batch]["observations"]]
                validation = acquire(channels, "test_partial_object_task_explanation", rows_count=12)
                local = SearchBudget(min(12000, budget.remaining // 3))
                claims = propose_partial_object_claims(replace(raw_spec, context=previous_context), training, local)
                budget.spend(local.spent)
                assessed = self._assess(claims, validation, budget,
                    initial_prediction(training, raw_spec.risk_quantile), raw_spec.risk_quantile) if validation else []
                best = self._best(assessed)
                stable = best is not None and best["validation_loss"] <= self.config.minimum_improvement
                actions.append({"action": "test_partial_object_explanation", "object_model_id": previous["model_id"],
                    "supported": stable, "validation_loss": best["validation_loss"] if best else None,
                    "unresolved": list(previous.get("unresolved", []))})
                if not stable:
                    actions.append({"action": "refine_partial_object_dependencies", "object_model_id": previous["model_id"],
                        "reason": "current_response_requires_unresolved_roles"})
                    for probe in previous.get("dependency_probes", []):
                        if len(records()) + len(probe["sequence"]) > row_limit:
                            break
                        acquire(channels, "replay_object_dependency_experiment", probe=Probe(
                            f"object_refine_{len(batches)}", tuple(probe["sequence"])))
            if stable:
                self._object_models[tuple(channels)] = deepcopy(previous)
                if previous.get("complete"):
                    self._object_schemas[tuple(channels)] = deepcopy(previous["schema"])
                actions.append({"action": "reuse_discovered_objects", "object_model_id": previous["model_id"],
                    "rows": len(records()), "field_losses": errors, "complete": previous.get("complete", False)})
                return {**deepcopy(previous), "reuse_rows": len(records()), "reused": True}
            if any(error > self.config.minimum_improvement for error in errors.values()):
                previous["status"] = "counterexample"
                previous.setdefault("counterexamples", []).append({"field_losses": errors, "rows": len(evidence)})
                actions.append({"action": "reopen_object_dependencies", "object_model_id": previous["model_id"],
                    "reason": "observed_control_to_sensor_mapping_changed", "field_losses": errors})

        model = infer_object_model(raw_spec, records(), budget)
        tested_partial_roles = set()
        for serial in range(row_limit):
            evidence = records()
            if model.get("complete") or len(evidence) >= row_limit:
                break
            groups = model.get("groups", [])
            role_signature = tuple(tuple(group.get(role) for role in ("q", "u", "x")) for group in groups)
            if (groups and all(group.get("u") and group.get("controls", {}).get("q") for group in groups)
                    and role_signature not in tested_partial_roles):
                from .partial_object_knowledge import propose_partial_object_claims
                tested_partial_roles.add(role_signature)
                training = [Observation(row["features"], row["target"], tuple(row["history"]))
                            for row in batches[initial_batch]["observations"]]
                local = SearchBudget(min(12000, budget.remaining // 3))
                claims = propose_partial_object_claims(replace(raw_spec,
                    context={**raw_spec.context, "object_model": model}), training, local)
                budget.spend(local.spent)
                promising = [claim for claim in claims if claim["training_loss"] <= self.config.minimum_improvement]
                if promising:
                    validation = acquire(channels, "test_identified_roles_for_current_task", rows_count=12)
                    assessed = self._assess(promising, validation, budget,
                        initial_prediction(training, raw_spec.risk_quantile), raw_spec.risk_quantile) if validation else []
                    best = self._best(assessed)
                    adequate = best is not None and best["validation_loss"] <= self.config.minimum_improvement
                    actions.append({"action": "use_partial_roles_for_current_question", "supported": adequate,
                        "validation_loss": best["validation_loss"] if best else None,
                        "unresolved": list(model.get("unresolved", []))})
                    if adequate:
                        break
            probe, question = self._object_probe(raw_spec, model, evidence, serial)
            if probe is None or len(evidence) + probe.cost > row_limit:
                break
            if acquire(channels, "distinguish_object_dependencies", probe=probe) is None:
                break
            before = deepcopy(model.get("unresolved", []))
            model = infer_object_model(raw_spec, records(), budget)
            actions.append({"action": "test_object_dependency_question", "question": question,
                "probe_id": probe.probe_id, "observation_rows": probe.cost,
                "unresolved_before": before, "unresolved_after": deepcopy(model.get("unresolved", []))})
        # Discovery data selected the grouping. A separate simultaneous setup
        # can still contradict those field roles before they guide prediction.
        if model.get("complete"):
            if acquire(channels, "confirm_discovered_object_roles", rows_count=12) is not None:
                model = infer_object_model(raw_spec, records(), budget, prior=model)
            else:
                model = {**model, "complete": False, "schema": None, "status": "unresolved"}
        model = deepcopy(model)
        model.update(model_id=f"objects_{len(self.library.object_models) + 1:04d}",
            acquisition_rows=len(records()), compute_cost=budget.spent - start,
            parent_model_id=previous["model_id"] if previous else None, counterexamples=[], reused=False)
        if not model.get("complete"):
            from .object_alternatives import enumerate_object_schemas
            model["schema_alternatives"] = enumerate_object_schemas(model, budget)
        model["dependency_probes"] = [deepcopy(batch["probe"]) for batch in batches[initial_batch:]
            if any(action.get("probe_id") == batch["probe"]["probe_id"]
                   and action.get("purpose") == "distinguish_object_dependencies" for action in actions)]
        self._object_models[tuple(channels)] = deepcopy(model)
        if model.get("complete"):
            self._object_schemas[tuple(channels)] = deepcopy(model["schema"])
            model["membership_witnesses"] = []
            for group in model["schema"]["objects"]:
                field = model["schema"]["fields"][group]["f"]
                witness = next((row["controls"] for row in records() if row["before"][field] > .5), None)
                if witness is not None and witness not in model["membership_witnesses"]:
                    model["membership_witnesses"].append(witness)
        if self.config.use_memory:
            self.library.object_models.append(deepcopy(model))
        actions.append({"action": "adopt_discovered_objects" if model.get("complete") else "retain_object_ambiguity",
            "object_model_id": model["model_id"], "rows": model["acquisition_rows"],
            "groups": deepcopy(model.get("groups", {})), "unresolved": deepcopy(model.get("unresolved", [])),
            "compute_cost": model["compute_cost"]})
        return model

    def _revisable_inputs(self, laboratory):
        return self.config.use_memory and self._instrument_knowledge(laboratory)

    def _input_prior(self, prior):
        payload = prior.get("input_program") if self._revise_inputs and prior else None
        if payload is None:
            return prior
        return {**prior, "source_features": payload["required_fields"],
                "object_schema": payload["source_schema"]}

    def _remember_inputs(self, prior, program, measured_spec, command_spec, rows, budget, actions):
        """Keep a learned mechanism's measured inputs separate from its calibration."""
        if prior.get("object_schema") or prior.get("input_program") or not _object_schema(measured_spec):
            return
        from .abstractions import extract_templates
        from .relocatable_programs import abstract_measurements, instantiate_measurements
        payload = abstract_measurements(program, measured_spec, self.library)
        if not payload["complete"] or not payload["required_fields"]:
            return
        binding = self._instrument_binding(measured_spec, command_spec)
        if not set(payload["required_fields"]) <= set(binding) or not budget.spend(2 * len(rows)):
            return
        rebound = instantiate_measurements(payload, binding)
        error = sum((rebound.predict(row) - program.predict(row)) ** 2 for row in rows) / len(rows)
        if error > self.config.minimum_improvement:
            actions.append({"action": "reject_input_abstraction", "loss": error,
                            "source_concept_id": prior["source_concept_id"]})
            return
        canonical = Program.from_dict(payload["program"])
        source = prior["source_concept_id"]
        template = extract_templates(canonical, f"{source}:measurement_inputs",
                                     object_schema=payload["source_schema"])[0]
        template.source_concept_id = source
        template.parents = [source]
        pattern = canonical.remap_features(dict(zip(template.source_features, template.roles)))
        arguments = []

        def parameterize(node):
            if node.op in {"feature", "lag", "available"}:
                if node not in arguments:
                    arguments.append(node)
                return Program("arg", name=f"input_{arguments.index(node)}")
            return replace(node, children=tuple(parameterize(child) for child in node.children))

        body = parameterize(pattern)
        definition = Definition(f"{source}:measurement_inputs:definition",
            tuple(f"input_{index}" for index in range(len(arguments))), body, source)
        payload.update(definition=definition.to_dict(), template=template.to_dict(),
            arguments=[argument.to_dict() for argument in arguments],
            roundtrip_loss=error, support_rows=len(rows))
        prior["input_program"] = payload
        if prior.get("research_program"):
            from .research_program_inputs import abstract_research_inputs
            procedure = abstract_research_inputs(prior["research_program"], measured_spec, self.library)
            if procedure is not None:
                prior["input_research_program"] = procedure
        actions.append({"action": "preserve_revisable_input_dependencies", "source_concept_id": source,
            "source_dependencies": payload["source_dependencies"], "fields": payload["required_fields"],
            "research_program_preserved": bool(prior.get("input_research_program")), "roundtrip_loss": error})

    def _remember_relation_research(self, prior, selected, measured_spec, actions):
        """Keep new paid relation evidence even when the executable concept is unchanged."""
        procedure = selected.get("research_program")
        steps = (procedure or {}).get("selection_steps", [])
        if not any(step["operation"] == "resolve_relation_intervals" for step in steps):
            return False
        inputs = None
        if prior.get("input_program") or prior.get("input_research_program"):
            from .research_program_inputs import abstract_research_inputs
            inputs = abstract_research_inputs(procedure, measured_spec, self.library)
            if inputs is None:
                actions.append({"action": "unresolved_relation_research_inputs",
                                "source_concept_id": prior["source_concept_id"]})
                return False
        prior["research_program"] = deepcopy(procedure)
        if inputs is not None:
            prior["input_research_program"] = inputs
        actions.append({"action": "preserve_relation_research", "source_concept_id": prior["source_concept_id"],
            "evidence_contexts": sum(len(step.get("evidence_contexts", [])) for step in steps),
            "input_research_program_preserved": inputs is not None})
        return True

    def _instrument_knowledge(self, laboratory):
        """Research whether commands and their history explain paid readings."""
        return self.config.strategy != "fixed" and bool(laboratory.channels)

    def _instrument_binding(self, measured_spec, command_spec):
        if not self._learn_instruments or not self.config.use_memory:
            return {}
        from .instrument_models import rebind_instrument_model
        binding = {}
        for model in reversed(self.library.instrument_models):
            if model.get("status") == "counterexample":
                continue
            mapped = rebind_instrument_model(model, measured_spec)
            for name, program in mapped.items():
                if name not in binding and {field for field, _ in program.observed_variables()} <= set(command_spec.feature_names):
                    binding[name] = program
        return binding

    def _instrument_role_binding(self, template, prior, command_spec):
        if not prior:
            return None
        if not prior.get("object_schema"):
            measured_spec = replace(command_spec, feature_names=tuple(prior["source_features"]))
            instruments = self._instrument_binding(measured_spec, command_spec)
            controls = set(prior.get("control_features", ())) & set(command_spec.feature_names)
            mapping = {name: Program("feature", name=name) for name in controls}
            mapping.update(instruments)
            if not set(template.source_features) <= set(mapping):
                return None
            return {role: mapping[name] for role, name in zip(template.roles, template.source_features)}
        if self._object_schemas:
            # Discovered object identities bind source roles to current
            # instruments before replacing those readings with commands. Old
            # control names can denote different sites after a wiring change.
            schema = next(iter(self._object_schemas.values()))
            features = tuple(name for group in schema["objects"] for name in schema["fields"][group].values())
            current = replace(command_spec, feature_names=features,
                context={"object_schema": json.dumps(schema)})
            binding = self._instrument_binding(current, command_spec)
            roles = aggregate_role_binding(template, prior["object_schema"], schema)
            if roles is None:
                roles = next(iter(template.mappings(features,
                    structural_transfer=self.config.structural_transfer, object_schema=schema)), None)
            if roles is None:
                return None
            mapped = {role: bind_measurement_arguments(
                Program("feature", name=value) if isinstance(value, str) else value, binding)
                for role, value in roles.items()}
            if any(program.required_features() - set(command_spec.feature_names) for program in mapped.values()):
                return None
            return mapped
        for view, model in self._object_models.items():
            if model.get("complete"):
                continue
            current = replace(command_spec, feature_names=tuple(model["raw_sensor_features"]),
                context={"object_model": model})
            binding = self._instrument_binding(current, command_spec)
            if set(template.source_features) <= set(binding):
                return {role: binding[name] for role, name in zip(template.roles, template.source_features)}
        measured_spec = replace(command_spec, feature_names=tuple(prior["source_features"]),
            context={"object_schema": json.dumps(prior["object_schema"])})
        binding = self._instrument_binding(measured_spec, command_spec)
        if not set(template.source_features) <= set(binding):
            return None
        return {role: binding[name] for role, name in zip(template.roles, template.source_features)}

    def _fit_instruments(self, command_spec, measured_spec, paired_batches, budget, actions):
        """Fit purchased readings, then compare on a later purchased trajectory."""
        from .instrument_models import fit_instrument_model, rebind_instrument_model
        if len(paired_batches) < 2:
            return None
        training = [pair for batch in paired_batches[:-1] for pair in batch]
        validation = paired_batches[-1]

        def compare(binding, observations=validation):
            if not binding or not budget.spend(len(observations) * len(binding)):
                return {}
            return {name: sum((program.predict(commands) - measured.features[name]) ** 2
                              for commands, measured in observations) / len(observations)
                    for name, program in binding.items()}

        # Response mismatch alone does not revoke calibration. Newly purchased
        # readings distinguish a changed instrument map from a changed law.
        if self.config.use_memory:
            for old in reversed(self.library.instrument_models):
                if old.get("status") == "counterexample":
                    continue
                binding = rebind_instrument_model(old, measured_spec)
                # An earlier measured counterexample cannot be cleared by a
                # later random trajectory that happens to miss its boundary.
                errors = compare(binding, [*training, *validation])
                failed = {name: error for name, error in errors.items()
                          if error > self.config.minimum_improvement}
                if failed:
                    old["status"] = "counterexample"
                    old.setdefault("counterexamples", []).append({"task_id": measured_spec.task_id,
                        "field_losses": failed, "rows": len(training) + len(validation)})
                    actions.append({"action": "instrument_mapping_counterexample",
                        "model_id": old["model_id"], "field_losses": failed,
                        "rows": len(training) + len(validation)})
                elif set(measured_spec.feature_names) <= set(errors):
                    actions.append({"action": "reuse_instrument_mapping", "model_id": old["model_id"],
                        "field_losses": errors, "rows": len(training) + len(validation)})
                    return old
        before = budget.spent
        local = SearchBudget(min(24000, budget.remaining // 2))
        model = fit_instrument_model(command_spec, measured_spec, training, local,
                                     max_history=self.config.max_history)
        budget.spend(local.spent)
        errors = compare(rebind_instrument_model(model, measured_spec))
        # Only fields supported by an unseen measured trajectory become
        # executable substitutes. Partial answers can remain useful knowledge.
        model["fields"] = {name: field for name, field in model["fields"].items()
                           if name in errors and errors[name] <= self.config.minimum_improvement}
        model.update(model_id=f"instrument_{len(self.library.instrument_models) + 1:04d}",
                     status="supported_on_measured_trajectory", validation_losses=errors,
                     validation_rows=len(validation), counterexamples=[])
        actions.append({"action": "research_instrument_mapping", "model_id": model["model_id"],
            "fields": list(model["fields"]), "field_losses": errors,
            "field_history_depths": {name: control_history_depth(Program.from_dict(field["program"]))
                                    for name, field in model["fields"].items()},
            "fit_rows": len(training), "validation_rows": len(validation),
            "compute_cost": budget.spent - before})
        if not model["fields"]:
            return None
        if self.config.use_memory:
            self.library.instrument_models.append(model)
        return model

    def _probe(self, spec, serial, length, candidates, budget, actions):
        bounds = tuple(spec.feature_bounds.items())
        rng = Random(self.config.seed + 1009 * serial)
        random_points = tuple({name: rng.uniform(*limits) for name, limits in bounds}
                              for _ in range(length))
        selected, score = random_points, None
        mode = "independent_random"
        if self.config.active_probes and self.config.strategy != "fixed" and candidates:
            # Evaluate proposed interventions only where the proposed model
            # actually has all its input coordinates. Unmeasured instruments
            # are never invented to score a hypothetical observation.
            controllable = set(spec.feature_bounds)
            programs = [item["candidate"].program for item in candidates
                        if {name for name, _ in item["candidate"].program.observed_variables()}
                        <= controllable][:4]
            pool = [random_points]
            pool.append(tuple({name: limits[(step + axis) % 2]
                               for axis, (name, limits) in enumerate(bounds)}
                              for step in range(length)))
            pool.append(tuple({name: limits[(step // 3 + axis) % 2]
                               for axis, (name, limits) in enumerate(bounds)}
                              for step in range(length)))
            if len(programs) >= 2 and budget.spend(len(pool) * length * len(programs)):
                scores = []
                for points in pool:
                    history, disagreement = [], 0.
                    for point in points:
                        row = Observation(point, 0., tuple(history))
                        values = [program.predict(row) for program in programs]
                        disagreement += max(values) - min(values)
                        history.append(point)
                    scores.append(disagreement / length)
                index = max(range(len(pool)), key=scores.__getitem__)
                selected, score, mode = pool[index], scores[index], "prediction_disagreement"
            elif any(item["candidate"].program.history_depth for item in candidates):
                # A recorded-history claim motivates a long constant suffix
                # after differing controls. Sensor realizations remain unknown
                # until the actual acquisition is paid.
                selected = pool[2]
                mode = "history_boundary_intervention"
        actions.append({"action": "choose_experiment", "probe_id": f"unified_{serial}",
                        "policy": mode, "predicted_disagreement": score})
        return Probe(f"unified_{serial}", selected)

    def _recall(self, spec):
        if not self.config.use_memory or self.config.strategy == "fixed":
            return []
        concepts = {concept.concept_id: concept for concept in self.library.concepts
                    if concept.status != "revoked"}
        templates = {template.template_id: template for template in self.library.abstracts}
        records = []
        for definition_id, binding in self.library.definition_bindings.items():
            definition = self.library.definitions[definition_id]
            source = concepts.get(definition.source_concept_id)
            template = templates.get(binding["template_id"])
            if (source is None or template is None or template.status == "revoked"
                    or binding["template_id"] != f"{source.concept_id}:abstract:0"
                    or source.risk_quantile != spec.risk_quantile):
                continue
            prior = next((item for item in reversed(self.library.meta_priors)
                          if item["source_concept_id"] == source.concept_id), None)
            payload = prior.get("input_program") if self._revise_inputs and prior else None
            if payload is not None:
                from .abstractions import AbstractTemplate
                template = AbstractTemplate.from_dict(payload["template"])
                definition = Definition.from_dict(payload["definition"])
                binding = {"arguments": tuple(Program.from_dict(argument) for argument in payload["arguments"])}
            records.append((source, template, definition, binding))
        # Give complete learned mechanisms a turn before multiple role bindings
        # of the same mechanism. Successful bindings are already remembered by
        # AbstractTemplate.mappings; names are not used as mechanism labels.
        records.sort(key=lambda item: (-len(item[0].successful_tasks), -item[0].generation,
                                       item[0].program.size))

        def role_bindings(source, template):
            prior = next((prior for prior in reversed(self.library.meta_priors)
                          if prior["source_concept_id"] == source.concept_id), None)
            prior = self._input_prior(prior)
            source_controls = prior.get("control_features") if prior else None
            schema = _object_schema(spec)
            if (not template.object_roles and prior and source_controls is not None and
                    set(spec.feature_names) <= set(spec.feature_bounds) and
                    set(template.source_features) - set(source_controls)):
                mapping = self._instrument_role_binding(template, prior, spec)
                if mapping is not None:
                    yield mapping
                return
            if template.object_roles and schema is None:
                mapping = self._instrument_role_binding(template, prior, spec)
                if mapping is not None:
                    yield mapping
                elif spec.context.get("object_model") and set(template.source_features) <= set(spec.feature_names):
                    yield dict(zip(template.roles, template.source_features))
                return
            if template.object_roles and self.config.structural_transfer:
                aggregate = aggregate_role_binding(template,
                    prior.get("object_schema") if prior else None, schema)
                if aggregate is not None:
                    yield aggregate
                    # A source group covering the apparatus denotes its whole
                    # extent. Binding that role to one fine object changes the
                    # learned measurement, rather than changing its scale.
                    return
            if source_controls is not None and not template.object_roles:
                required_controls = sum(name in source_controls for name in template.source_features)
                available_controls = sum(name in spec.feature_bounds for name in spec.feature_names)
                if (required_controls > available_controls or
                        len(template.roles) - required_controls > len(spec.feature_names) - available_controls):
                    return
            for mapping in template.mappings(spec.feature_names,
                                             structural_transfer=self.config.structural_transfer,
                                             object_schema=schema):
                if source_controls is not None and not template.object_roles and any(
                        (original in source_controls) != (mapping[role] in spec.feature_bounds)
                        for role, original in zip(template.roles, template.source_features)):
                    continue
                yield mapping

        queues = [(definition, binding, iter(role_bindings(source, template)))
                  for source, template, definition, binding in records]
        calls = []
        while queues and len(calls) < self.config.max_candidates:
            next_queues = []
            for definition, binding, mappings in queues:
                mapping = next(mappings, None)
                if mapping is None:
                    continue
                next_queues.append((definition, binding, mappings))
                call = Program("call", name=definition.definition_id, definition=definition,
                    children=tuple(bind_measurement_arguments(argument, mapping)
                                   for argument in binding["arguments"]))
                if call not in calls:
                    calls.append(call)
                if len(calls) >= self.config.max_candidates:
                    break
            queues = next_queues
        return calls

    def _program_claims(self, spec, training, budget, calls, actions):
        """Execute remembered research procedures, then compose unexplained parts."""
        if self.config.max_question_depth <= 0 or not calls:
            return []
        from .abstractions import AbstractTemplate
        from .research_programs import propose_research_programs, replay_research_program
        before = budget.spent
        claims = []
        templates = {template.template_id: template for template in self.library.abstracts
                     if template.status != "revoked"}
        recalled = {source for call in calls for source in _sources(call)}
        queues, queued_sources = [], set()
        for prior in reversed(self.library.meta_priors):
            source = prior["source_concept_id"]
            inputs = prior.get("input_research_program") if self._revise_inputs else None
            payload = inputs or prior.get("research_program")
            input_program = prior.get("input_program", {})
            if inputs:
                saved_template = inputs.get("input_template") or input_program.get("template")
                template = AbstractTemplate.from_dict(saved_template) if saved_template else None
                source_schema = inputs.get("source_schema") or input_program.get("source_schema")
            else:
                template = templates.get(f"{source}:abstract:0")
                source_schema = prior.get("object_schema")
            if not payload or source not in recalled or template is None or source in queued_sources:
                continue
            schema = _object_schema(spec)
            aggregate = aggregate_role_binding(template, source_schema, schema)
            binding_prior = {**prior, "object_schema": source_schema,
                             "source_features": list(template.source_features)}
            command_binding = self._instrument_role_binding(template, binding_prior, spec) if schema is None else None
            if inputs and schema is None and template.object_roles and command_binding is None:
                continue
            mappings = ([command_binding] if command_binding is not None else
                        [aggregate] if aggregate is not None else list(islice(
                template.mappings(spec.feature_names, structural_transfer=self.config.structural_transfer,
                                  object_schema=schema), 2)))
            if mappings:
                queues.append((source, payload, template, iter(mappings[:2])))
                queued_sources.add(source)
        # Give distinct learned procedures an initial evidence comparison before
        # another role binding of a recent procedure can consume the next slot.
        while queues and len(claims) < 4:
            remaining = []
            for source, payload, template, mappings in queues:
                mapping = next(mappings, None)
                if mapping is None:
                    continue
                remaining.append((source, payload, template, mappings))
                features = {original: mapping[role]
                            for role, original in zip(template.roles, template.source_features)}
                replayed = replay_research_program(payload, training, budget,
                    risk_quantile=spec.risk_quantile, feature_mapping=features)
                if replayed is not None:
                    replayed["method_source_ids"] = [source]
                    replayed["candidate"] = replace(replayed["candidate"], parents=tuple(dict.fromkeys(
                        (*replayed["candidate"].parents, source))))
                    claims.append(replayed)
                if len(claims) >= 4:
                    break
            queues = remaining
        if not any(item["training_loss"] <= self.config.minimum_improvement for item in claims):
            from .relation_revisions import propose_relation_revisions
            local = SearchBudget(min(12000, budget.remaining // 2))
            claims.extend(propose_relation_revisions(spec, training, local, calls,
                tolerance=self.config.minimum_improvement))
            budget.spend(local.spent)
        if not any(item["training_loss"] <= self.config.minimum_improvement for item in claims):
            claims.extend(propose_research_programs(spec, training, budget, calls,
                                                    max_candidates=self.config.max_candidates))
        actions.append({"action": "execute_research_programs", "compute_cost": budget.spent - before,
            "features": list(spec.feature_names), "alternatives": [
                {"training_loss": item["training_loss"],
                 "reused_procedure": item["research_program"].get("reused_procedure", False),
                 "source_ids": list(item["candidate"].parents),
                 "steps": len(item["research_program"]["selection_steps"])} for item in claims]})
        return claims

    def _assess(self, claims, validation, budget, fallback, q):
        results = []
        for claim in claims:
            if claim is None or not budget.spend(len(validation)):
                continue
            item = dict(claim)
            item["validation_loss"] = loss(claim["candidate"].program, validation, fallback, q)
            results.append(item)
        return results

    def _best(self, claims):
        if not claims:
            return None
        lowest = min(item["validation_loss"] for item in claims)
        adequate = [item for item in claims
                    if item["validation_loss"] <= lowest + self.config.minimum_improvement]
        return min(adequate, key=lambda item: (item["candidate"].program.size,
                                              item["validation_loss"]))

    def _support(self, program, training, validation, spec, budget):
        """Keep the external loss fixed while testing residual explanations."""
        from .unified_uncertainty import assess_noisy_adequacy
        return assess_noisy_adequacy(spec, program, training, validation, budget,
            minimum_improvement=self.config.minimum_improvement)

    def _risk_revisions(self, spec, claims, training, budget):
        from .unified_uncertainty import residual_risk_candidates
        if not self.config.risk_learning or spec.risk_quantile == .5:
            return []
        parents = sorted((claim for claim in claims if claim is not None),
                         key=lambda claim: claim["training_loss"])[:3]
        revised = []
        for parent in parents:
            local = SearchBudget(min(12000, budget.remaining // 2))
            results = residual_risk_candidates(spec, parent["candidate"].program,
                                               training, local)
            budget.spend(local.spent)
            for result in results:
                result["parent_kind"] = parent["kind"]
            revised.extend(results)
        return revised

    def _tail_calibration(self, claim, rows, spec, budget, actions):
        """Use out-of-fit residuals to propose a further fixed-risk revision."""
        if (spec.risk_quantile == .5 or not self.config.risk_learning
                or self.config.strategy == "fixed" or not rows or not budget.spend(len(rows))):
            return None
        from .risk_search import empirical_quantile
        program = claim["candidate"].program
        offset = empirical_quantile([row.target - predict(program, row, 0.) for row in rows],
                                    spec.risk_quantile)
        if abs(offset) < 1e-12:
            return None
        calibrated = Program("add", children=(program, Program("const", value=offset))).simplified()
        actions.append({"action": "propose_tail_calibration", "source_rows": len(rows),
            "fixed_quantile": spec.risk_quantile, "offset": offset,
            "parent_program": program.to_dict(), "program": calibrated.to_dict()})
        return {**claim, "candidate": Candidate(f"tail_calibration:{budget.spent}", calibrated,
                    "risk_calibration", "Revise the fixed tail from previously held-out residuals"),
                "kind": "independent_tail_calibration", "calibration_parent_kind": claim["kind"]}

    def _compare_calibration(self, original, calibrated, rows, spec, budget, fallback, actions):
        if not budget.spend(len(rows)):
            return original, None
        old_loss = loss(original["candidate"].program, rows, fallback, spec.risk_quantile)
        if calibrated is None or not budget.spend(len(rows)):
            return original, old_loss
        new_loss = loss(calibrated["candidate"].program, rows, fallback, spec.risk_quantile)
        chosen = calibrated if new_loss + self.config.minimum_improvement < old_loss else original
        actions.append({"action": "compare_tail_calibration", "rows": len(rows),
            "original_loss": old_loss, "calibrated_loss": new_loss,
            "accepted": chosen is calibrated, "fixed_quantile": spec.risk_quantile})
        if chosen is calibrated:
            chosen["calibration_parent_validation_loss"] = chosen["validation_loss"]
            chosen["validation_loss"] = new_loss
        return chosen, new_loss if chosen is calibrated else old_loss

    def _object_explanations(self, spec, rows, budget, *, actions, parent=None,
                             new_observations=(), transitions=(), tolerance=1e-7):
        from .equivalent_object_programs import propose_equivalent_object_programs
        from .competing_object_programs import research_competing_object_programs
        previous = parent.get("object_competition") if parent else None
        if previous is None:
            shared = propose_equivalent_object_programs(spec, rows, budget,
                parent=parent, transitions=transitions, tolerance=tolerance)
            supported = []
            for item in shared:
                if item["training_loss"] > tolerance:
                    continue
                if new_observations:
                    if not budget.spend(len(new_observations)):
                        break
                    if loss(item["candidate"].program, new_observations, 0., spec.risk_quantile) > tolerance:
                        continue
                supported.append(item)
            if supported:
                return supported
        result = research_competing_object_programs(spec, rows, budget, previous=previous,
            new_observations=new_observations, transitions=transitions, tolerance=tolerance)
        if parent is not None and previous is not None and not result["entries"]:
            parent["object_competition"] = result["competition"]
        self._object_competitions[tuple(spec.feature_names)] = result["competition"]
        actions.append({"action": "compare_competing_object_programs",
            "features": list(spec.feature_names), "status": result["competition"]["status"],
            "hypotheses": len(result["competition"]["hypotheses"]),
            "program_classes": result["competition"].get("program_classes", 0),
            "compute_cost": result["competition"]["compute_cost"]})
        return result["entries"]

    def _object_program_refuted(self, entry, spec):
        competition = self._object_competitions.get(tuple(spec.feature_names),
            entry.get("object_competition") or {})
        counterexamples = competition.get("counterexamples", [])
        if not counterexamples:
            return False
        program = entry["candidate"].program.to_dict()
        return any(item.get("status") == "refuted" and item.get("program") == program
                   for item in counterexamples)

    def _research_view(self, spec, training, validation, budget, actions, transfer,
                       graph, *, parent=None, transitions=()):
        before = budget.spent
        fallback = initial_prediction(training, spec.risk_quantile)
        if self.config.strategy == "fixed":
            claims = [_fit(_terms(spec.feature_names, 2, True), training, budget,
                           "fixed_quadratic_two_step_history", family="T08", q=spec.risk_quantile)]
            claims.extend(self._risk_revisions(spec, claims, training, budget))
            return self._assess(claims, validation, budget, fallback, spec.risk_quantile)

        calls = self._recall(spec)
        reused = [_fit([call, Program("mul", children=(call, call)),
                        *_terms(spec.feature_names)], training, budget,
                       "callable_transfer", family="T03", q=spec.risk_quantile) for call in calls]
        assessed = self._assess(reused, validation, budget, fallback, spec.risk_quantile)
        for item in assessed:
            transfer.append({"source_ids": _sources(item["candidate"].program),
                             "program": item["candidate"].program.to_dict(),
                             "validation_loss": item["validation_loss"],
                             "status": "supported" if item["validation_loss"] <=
                             self.config.minimum_improvement else "counterexample",
                             "validation_rows": len(validation), "accepted": False})
        # Reuse competes with an inexpensive current-coordinate explanation.
        # A successful learned computation does not force a full rediscovery.
        affine = _fit(_terms(spec.feature_names), training, budget, "current_affine", q=spec.risk_quantile)
        assessed.extend(self._assess([affine], validation, budget, fallback, spec.risk_quantile))
        if spec.context.get("object_model") and not _object_schema(spec):
            from .partial_object_knowledge import propose_partial_object_claims
            partial_claims = propose_partial_object_claims(spec, training, budget)
            checked = self._assess(partial_claims, validation, budget, fallback, spec.risk_quantile)
            assessed.extend(checked)
            actions.append({"action": "compare_partial_object_claims",
                "alternatives": [{"kind": item["kind"], "validation_loss": item["validation_loss"]} for item in checked],
                "unresolved": list(spec.context["object_model"].get("unresolved", []))})
        if affine is None or affine["training_loss"] > self.config.minimum_improvement:
            local = SearchBudget(min(32000, budget.remaining // 2))
            programs = self._program_claims(spec, training, local, calls, actions)
            budget.spend(local.spent)
            assessed.extend(self._assess(programs, validation, budget, fallback, spec.risk_quantile))
            best = self._best(assessed)
            if best and best["validation_loss"] <= self.config.minimum_improvement:
                return assessed
        if _object_schema(spec) is not None or spec.context.get("object_model"):
            if _object_schema(spec) is not None:
                from .unified_structure import propose_structures
            else:
                propose_structures = partial(self._object_explanations, actions=actions)
            structural_allowance = (max(0, budget.remaining - 16 * len(validation))
                if spec.context.get("object_model") and not _object_schema(spec)
                else budget.remaining // 2)
            local = SearchBudget(min(90000, structural_allowance))
            structures = propose_structures(spec, training, local, parent=parent,
                                            tolerance=self.config.minimum_improvement,
                                            transitions=transitions)
            budget.spend(local.spent)
            structural_claims = self._assess(structures, validation, budget, fallback, spec.risk_quantile)
            assessed.extend(structural_claims)
            leading = self._best(structural_claims)
            explained_families = {item["kind"] for item in structural_claims
                                  if item["training_loss"] <= self.config.minimum_improvement}
            if (leading and leading["kind"] in explained_families
                    and max(leading["training_loss"], leading["validation_loss"])
                    > self.config.minimum_improvement):
                # The same relation explains one trajectory and nearly explains
                # the other, but its unresolved edge gives conflicting fits.
                # Use both paid trajectories to revise that relation before
                # spending the recovery budget on unrelated model families.
                revision_budget = SearchBudget(min(12000, budget.remaining // 2))
                revised = propose_structures(spec, [*training, *validation], revision_budget,
                    parent=leading, tolerance=self.config.minimum_improvement, transitions=transitions,
                    **({"new_observations": validation} if not _object_schema(spec) else {}))
                budget.spend(revision_budget.spent)
                consistent = [{**item, "validation_loss": item["training_loss"],
                               "selection_basis": "all_acquired_evidence"} for item in revised
                              if item["training_loss"] <= self.config.minimum_improvement]
                assessed.extend(consistent)
                actions.append({"action": "revise_structure_from_combined_evidence",
                    "family": leading["kind"], "rows": len(training) + len(validation),
                    "compute_cost": revision_budget.spent, "answered": bool(consistent),
                    "selection_basis": "all_acquired_evidence"})
            actions.append({"action": "compare_structural_explanations",
                "features": list(spec.feature_names), "compute_cost": local.spent,
                "alternatives": [{"kind": item["kind"], "training_loss": item["training_loss"],
                                  "validation_loss": item["validation_loss"]}
                                 for item in assessed]})
            if any(item.get("object_equivalence") for item in structural_claims):
                actions.append({"action": "compare_equivalent_object_programs",
                    "alternatives": [{"schemas": len(item["object_equivalence"]["schemas"]),
                        "validation_loss": item["validation_loss"], "kind": item["kind"]}
                        for item in structural_claims if item.get("object_equivalence")]})
        best = self._best(assessed)
        if best and (best["validation_loss"] <= self.config.minimum_improvement or
                     best.get("structure") and best["training_loss"] <= self.config.minimum_improvement):
            actions.append({"action": "compare_reuse_and_parameters", "features": list(spec.feature_names),
                "selected_kind": best["kind"], "validation_loss": best["validation_loss"],
                "compute_cost": budget.spent - before})
            return assessed

        # Existing ontology-linked claim compilation is shared with the meta
        # prior researcher. Its current, nonlinear and finite-history claims
        # compete on the same later trajectory.
        local = SearchBudget(min(budget.remaining // 2, 24000))
        claims = compile_claims(spec, training, local)
        budget.spend(local.spent)
        assessed.extend(self._assess(claims, validation, budget, fallback, spec.risk_quantile))
        joint = []
        for depth in range(1, self.config.max_history + 1):
            for quadratic in (False, True):
                if quadratic and not self.config.joint_state_coordinates:
                    continue
                joint.append(_fit(_terms(spec.feature_names, depth, quadratic), training, budget,
                    "joint_quadratic_history" if quadratic else "reset_history",
                    family="T08", q=spec.risk_quantile))
        assessed.extend(self._assess(joint, validation, budget, fallback, spec.risk_quantile))

        best = self._best(assessed)
        if (self.config.state_search and best is not None
                and best["validation_loss"] > self.config.minimum_improvement):
            local = SearchBudget(min(48000, budget.remaining // 2))
            states = continuous_state_candidates(training, budget=local,
                                                   seed_programs=calls)
            budget.spend(local.spent)
            state_claims = []
            for candidate in states:
                if budget.spend(len(training)):
                    state_claims.append({"candidate": candidate, "kind": "continuous_latent_state",
                        "family_id": "T08", "training_loss": loss(candidate.program,
                            training, fallback, spec.risk_quantile)})
            assessed.extend(self._assess(state_claims, validation, budget, fallback,
                                         spec.risk_quantile))

        # A retained parent can remain useful even when measurement alone or
        # history alone fails. Execute a dependent residual question using the
        # newly paid coordinates, and compare its recombination with joint fit.
        if (parent is not None and self.config.max_question_depth > 0
                and parent["candidate"].program.required_features() <= set(spec.feature_names)
                and budget.spend(len(training))):
            base = parent["candidate"].program
            residuals = [Observation(row.features, row.target - base.predict(row), row.history)
                         for row in training]
            children = [_fit(_terms(spec.feature_names, depth, True), residuals, budget,
                             "residual_child", family="T09", q=spec.risk_quantile)
                        for depth in range(1, min(self.config.max_history, 2) + 1)]
            children = [child for child in children if child is not None]
            if children:
                child = min(children, key=lambda item: item["training_loss"])
                child_id = f"residual_{len(graph.nodes)}"
                node = graph.add_residual_question(child_id, base, training)
                node.required_measurements = list(set(spec.feature_names) - base.required_features())
                node.expression = child["candidate"].program
                answers = graph.execute_ready(training, budget, max_nodes=2,
                                              risk_quantile=spec.risk_quantile)
                combined = next((answer for answer in answers
                                 if answer.candidate_id == f"answer:{child_id}:combined"), None)
                if combined is not None:
                    assessed.extend(self._assess([{"candidate": combined,
                        "kind": "dependent_residual_revision", "family_id": "T09",
                        "training_loss": graph.nodes[f"{child_id}:combined"].loss}],
                        validation, budget, fallback, spec.risk_quantile))
                actions.append({"action": "execute_dependent_research", "question_id": child_id,
                                "answered": combined is not None,
                                "required_measurements": node.required_measurements})
        risks = self._risk_revisions(spec, assessed, training, budget)
        assessed.extend(self._assess(risks, validation, budget, fallback, spec.risk_quantile))
        best = self._best(assessed)
        actions.append({"action": "compare_research_directions", "features": list(spec.feature_names),
            "alternatives": [{"kind": item["kind"], "family_id": item["family_id"],
                               "validation_loss": item["validation_loss"],
                               "history_depth": item["candidate"].program.history_depth}
                              for item in assessed],
            "selected_kind": best["kind"] if best else None,
            "compute_cost": budget.spent - before})
        return assessed

    def run(self, laboratory):
        self._object_schemas = {}
        self._object_models = {}
        self._object_competitions = {}
        self._learn_instruments = self._instrument_knowledge(laboratory)
        self._revise_inputs = self._revisable_inputs(laboratory)
        budget = SearchBudget(self.config.compute_budget)
        actions, transfer, batches = [], [], []
        command_observations = {}
        observation_probes, noise_replications = {}, {}
        graph = ResearchGraph()
        rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        alternative_views = getattr(laboratory, "measurement_views", None)
        offered = tuple(alternative_views[-1]) if alternative_views else tuple(rates)

        def views_from(channels):
            if alternative_views:
                return list(laboratory.projectable_views(channels))
            return [subset for width in range(1, len(channels) + 1)
                    for subset in combinations(channels, width)] if channels else [()]

        def view_rows(rows, source, target):
            if source and not target:
                return [command_observations[id(row)] for row in rows]
            if alternative_views:
                projected = laboratory.project_rows(rows, source, target)
                for original, row in zip(rows, projected):
                    if id(original) in command_observations:
                        command_observations[id(row)] = command_observations[id(original)]
                    if id(original) in observation_probes:
                        observation_probes[id(row)] = observation_probes[id(original)]
                return projected
            return _view(rows, self._spec(laboratory, target).feature_names)

        def noise_coordinates(rows, source, evidence_channels):
            if not source and evidence_channels:
                # Commands and instrument readings were recorded together. Test
                # the original predictor and measured residual alternatives on
                # that same evidence, without inverting the instrument model.
                return [Observation({**row.features, **command_observations[id(row)].features},
                    row.target, tuple({**past, **command_past}
                        for past, command_past in zip(row.history,
                            command_observations[id(row)].history))) for row in rows]
            return rows

        def challenge_noise(support, program, validation, source, evidence_channels, fallback):
            if not support["adequate"] or support["reason"] != "noise_compatible":
                return support
            from .noise_replication import assess_noise_replication
            key = tuple(evidence_channels)
            required_rows = len(noise_replications[key][0]) if key in noise_replications else len(validation)
            if budget.remaining < 3 * required_rows:
                return {**support, "adequate": False, "reason": "noise_experiment_unfunded"}
            if key not in noise_replications:
                probe = observation_probes.get(id(validation[0])) if validation else None
                if probe is None:
                    return support
                repeated = acquire(evidence_channels, "distinguish_noise_from_repeatable_error",
                    probe=Probe(f"repeat_noise_{serial}", probe.sequence))
                if repeated is None:
                    return {**support, "adequate": False, "reason": "noise_experiment_unfunded"}
                noise_replications[key] = (validation, repeated)
            original, repeated = noise_replications[key]
            compared_program = measured_program(program, source, evidence_channels)
            evidence = assess_noise_replication(compared_program,
                noise_coordinates(original, source, evidence_channels),
                noise_coordinates(repeated, source, evidence_channels), budget,
                minimum_improvement=self.config.minimum_improvement,
                risk_quantile=laboratory.spec.risk_quantile, fallback=fallback)
            actions.append({"action": "compare_noise_replication", **evidence})
            return {**support, "replication": evidence,
                **({"adequate": False, "reason": "repeatable_unexplained_response"}
                   if evidence["repeatable_error"] else {})}

        def assess_trial_noise(program, training, validation, source, evidence_channels):
            spec = self._spec(laboratory, evidence_channels)
            original_validation = validation
            if not source and evidence_channels:
                training = noise_coordinates(training, source, evidence_channels)
                validation = noise_coordinates(validation, source, evidence_channels)
                spec = replace(spec, feature_names=tuple(dict.fromkeys(
                    [*spec.feature_names, *self._spec(laboratory, ()).feature_names])))
            else:
                program = measured_program(program, source, evidence_channels)
            support = self._support(program, training, validation, spec, budget)
            return challenge_noise(support, program, original_validation, source, evidence_channels,
                initial_prediction(training, laboratory.spec.risk_quantile))

        def measured_program(program, source, target):
            if alternative_views and tuple(source) != tuple(target):
                from .scale_bindings import aggregate_measurement_binding, bind_measurement_arguments
                binding = aggregate_measurement_binding(_object_schema(self._spec(laboratory, source)),
                    _object_schema(self._spec(laboratory, target)))
                if binding is not None:
                    return bind_measurement_arguments(program, binding)
            return program

        def transitions_for(channels, rows=None):
            observed = [row.features for row in rows] if rows is not None else None
            return [row for batch in batches if batch.get("transitions")
                    and tuple(channels) in views_from(tuple(batch["channels"]))
                    for row in laboratory.project_transitions(batch["transitions"], batch["channels"], channels)
                    if observed is None or row["before"] in observed]

        def paired_for(channels):
            result = []
            for batch in batches:
                if tuple(channels) not in views_from(tuple(batch["channels"])):
                    continue
                measured = view_rows([Observation(row["features"], row["target"], tuple(row["history"]))
                    for row in batch["observations"]], tuple(batch["channels"]), channels)
                history, pairs = [], []
                for point, row in zip(batch["probe"]["sequence"], measured):
                    pairs.append((Observation(dict(point), row.target, tuple(history)), row))
                    history.append({**point, "__target__": row.target})
                result.append(pairs)
            return result

        def channels_for(required):
            if alternative_views:
                return next((tuple(view) for view in alternative_views
                    if required <= set(self._spec(laboratory, view).feature_names)), offered)
            return tuple(name for name in offered if name in required)
        spent_observations, spent_measurements, serial = 0, 0, 0
        directed_command_rows = []
        length = min(24, laboratory.spec.max_probe_length,
                     max(1, self.config.observation_budget // 8))

        def acquire(channels, purpose, candidates=(), *, rows_count=None, probe=None, questions=()):
            nonlocal spent_observations, spent_measurements, serial
            count = probe.cost if probe is not None else length if rows_count is None else rows_count
            rate = sum(rates[name] for name in channels)
            if (spent_observations + count > self.config.observation_budget
                    or spent_measurements + count * rate > self.measurement_budget):
                return None
            # Endpoint history interventions can make every event response
            # zero. A later adoption comparison must also cover varied setups;
            # targeted structural interventions are acquired explicitly below.
            comparison = purpose in {"compare_on_new_trajectory", "validate_short_learned_trial",
                                     "confirm_short_frozen_program", "confirm_frozen_program",
                                     "confirm_instrument_replacement"}
            program_plan = None
            if (probe is None and not comparison and candidates and self.config.active_probes
                    and self.config.strategy != "fixed"):
                from .program_disagreement_probes import predicate_boundary_groups
                from .competing_object_experiments import (
                    plan_competing_object_probe, assess_competing_object_probe)
                command_spec = self._spec(laboratory, ())
                controls = set(command_spec.feature_bounds)
                controllable = [item for item in candidates
                    if {name for name, _ in item["candidate"].program.observed_variables()}
                    <= controls | {"__target__"}][:4]
                programs = [item["candidate"].program for item in controllable]
                # A set of independent conditions can hide all disagreement
                # from the ordinary random/end-point pool. Use the shared
                # program planner for these directly controllable candidates.
                if len(programs) > 1 and (questions or len(predicate_boundary_groups(programs, spend=budget.spend)) > 1):
                    program_plan = plan_competing_object_probe(command_spec, controllable, budget,
                        serial=serial, length=count, tolerance=self.config.minimum_improvement,
                        methods=self.library.program_experiment_methods if self.config.use_memory else None,
                        task_id=f"{laboratory.spec.task_id}:{self.config.seed}", questions=questions)
                    actions.append({"action": "plan_competing_program_experiment",
                        **{key: value for key, value in program_plan.items()
                           if key not in ("probe", "hypotheses", "predicted_fields")},
                        "probe_id": program_plan["probe"].probe_id if program_plan["probe"] else None})
                    probe = program_plan["probe"]
                    if probe is not None and (
                            spent_observations + probe.cost > self.config.observation_budget
                            or spent_measurements + probe.cost * rate > self.measurement_budget):
                        return None
            if purpose == "distinguish_program_conditions" and (
                    program_plan is None or program_plan["probe"] is None):
                return None
            probe = probe or self._probe(self._spec(laboratory, channels), serial, count,
                                        () if comparison else candidates, budget, actions)
            serial += 1
            batch = laboratory.acquire(probe, channels)
            command_history = []
            for point, row in zip(probe.sequence, batch.observations):
                command_observations[id(row)] = Observation(dict(point), row.target, tuple(command_history))
                observation_probes[id(row)] = probe
                observation_probes[id(command_observations[id(row)])] = probe
                command_history.append({**point, "__target__": row.target})
            spent_observations += batch.observation_cost
            spent_measurements += batch.measurement_cost
            batches.append({"probe": asdict(probe), "channels": list(channels),
                            "observations": [asdict(row) for row in batch.observations],
                            "transitions": getattr(batch, "transitions", [])})
            actions.append({"action": "acquire", "purpose": purpose, "probe_id": probe.probe_id,
                            "channels": list(channels), "observation_cost": batch.observation_cost,
                            "measurement_cost": batch.measurement_cost})
            if program_plan is not None and program_plan["probe"] is not None:
                actual = [(command_observations[id(row)], command_observations[id(row)])
                          for row in batch.observations]
                evidence = assess_competing_object_probe(program_plan, actual, budget,
                    tolerance=self.config.minimum_improvement)
                directed_command_rows.extend(pair[0] for pair in actual)
                actions.append({"action": "test_competing_program_predictions", "probe_id": probe.probe_id,
                    "plan": {key: value for key, value in program_plan.items() if key != "probe"},
                    "evidence": evidence})
                if self.config.use_memory and purpose != "distinguish_program_conditions":
                    feedback = self.library.program_experiment_methods.observe(program_plan, evidence,
                        task_id=f"{laboratory.spec.task_id}:{self.config.seed}", command_spec=command_spec,
                        tolerance=self.config.minimum_improvement)
                    if feedback is not None:
                        actions.append(feedback)
            return batch.observations

        object_research = None
        if self._object_induction(laboratory):
            object_research = self._research_objects(laboratory, offered, acquire, batches, budget, actions)
            self._learn_instruments = self._instrument_knowledge(laboratory)
            self._revise_inputs = self._revisable_inputs(laboratory)
            if object_research.get("groups"):
                model = self._fit_instruments(self._spec(laboratory, ()), self._spec(laboratory, offered),
                                              paired_for(offered), budget, actions)
                if model is not None:
                    object_research["sensor_model"] = deepcopy(model)
                    saved = next((item for item in self.library.object_models
                                  if item["model_id"] == object_research["model_id"]), None)
                    if saved is not None:
                        saved["sensor_model"] = deepcopy(model)

        selected_channels, finalists, parent, selected = (), [], None, None
        selected_training, selected_validation = [], []
        inherited_relation_evidence = []
        confirmation_loss, confirmed = None, False
        confirmation_rows, noisy_support = 0, None
        method_trial = False
        # Later compositions already use commands and can depend on measured
        # concepts through their function bodies. Keep these complete learned
        # calls alongside freshly rebound sensor mechanisms.
        command_calls = (self._recall(self._spec(laboratory, ()))
                         if self._learn_instruments and self.library.instrument_models else [])
        calls = command_calls or list(dict.fromkeys(call for names in views_from(offered)
                                                    for call in self._recall(self._spec(laboratory, names))))
        calls = calls[:self.config.max_candidates]
        trial_length = min(8, laboratory.spec.max_probe_length)
        recalled_sources = {source for call in calls for source in _sources(call)}
        learned_methods = [prior["research_method"] for prior in self.library.meta_priors
            if prior["source_concept_id"] in recalled_sources and prior.get("research_method")]
        noisy_methods = [method for method in learned_methods if method["noise_supported"]]
        if noisy_methods:
            trial_length = min(laboratory.spec.max_probe_length,
                max(32, min(max(method["fit_rows"], method["confirmation_rows"])
                            for method in noisy_methods)))
        required = {name for call in calls for name, _ in call.observed_variables()}
        trial_channels = channels_for(required)
        if noisy_methods:
            # A new independent instrument can explain an apparently random
            # residual. A reused noise model must face that alternative too.
            trial_channels = offered
        # A failed short trial must leave the ordinary two-view investigation
        # and its independent confirmation funded. Knowledge buys an earlier
        # empirical decision, not permission to spend the recovery budget.
        trial_funded = (self.config.observation_budget >= 7 * length + 3 * trial_length
            and self.measurement_budget >= 4 * length * sum(rates[name] for name in offered)
            + 3 * trial_length * sum(rates[name] for name in trial_channels)
            and budget.remaining >= 12000)
        from .shared_transfer_fitting import SharedTransferFitter
        transfer_fitter = SharedTransferFitter()

        def transfer_work(action, before, start, observations, candidates):
            actions.append({"action": action, "compute_cost": budget.spent - start,
                "observation_rows": len(observations), "candidate_count": len(candidates),
                "sharing": {name: value - before.get(name, 0)
                            for name, value in transfer_fitter.stats().items()}})

        def fit_transfer_candidates(observations, observation_channels):
            start, before = budget.spent, transfer_fitter.stats()
            proposed = []
            for call in calls:
                names = {name for name, _ in call.observed_variables()}
                channels = channels_for(names)
                spec = self._spec(laboratory, channels)
                rows = view_rows(observations, observation_channels, channels)
                # An inactive pilot does not erase the learned computation.
                # Keep it alongside the fitted change for real comparison.
                fit_view = (tuple(channels), "fit")
                retained_loss = transfer_fitter.loss(call, rows, budget,
                    initial_prediction(rows, spec.risk_quantile), spec.risk_quantile, view=fit_view)
                if retained_loss is not None:
                    proposed.append(({"candidate": Candidate(f"retained_call:{len(proposed)}",
                        call, "retain_learned_call", parents=tuple(_sources(call))),
                        "kind": "retained_calibrated_call" if command_calls else "retained_callable_transfer",
                        "family_id": "T03", "training_loss": retained_loss,
                        "method_source_ids": _sources(call)}, channels, rows))
                fitted = transfer_fitter.fit([call, Program("mul", children=(call, call)),
                              *(_terms(spec.feature_names) if not _object_schema(spec) else [])], rows, budget,
                              "short_callable_transfer", family="T03", q=spec.risk_quantile, view=fit_view)
                if fitted is not None:
                    fitted["method_source_ids"] = _sources(call)
                    proposed.append((fitted, channels, rows))
                    for revised in self._risk_revisions(spec, [fitted], rows, budget):
                        revised["method_source_ids"] = _sources(call)
                        proposed.append((revised, channels, rows))
            transfer_work("shared_transfer_fit", before, start, observations, proposed)
            return proposed

        def assess_transfer_candidates(candidates, observations, observation_channels, *, fallback_quantile=None):
            start, before = budget.spent, transfer_fitter.stats()
            compared = []
            for item, channels, training in candidates:
                rows = view_rows(observations, observation_channels, channels)
                value = transfer_fitter.loss(item["candidate"].program, rows, budget,
                    initial_prediction(training, laboratory.spec.risk_quantile
                                       if fallback_quantile is None else fallback_quantile),
                    laboratory.spec.risk_quantile, view=(tuple(channels), "validation"))
                if value is not None:
                    compared.append(({**item, "validation_loss": value}, channels, training))
            transfer_work("shared_transfer_validation", before, start, observations, compared)
            return compared

        transfer_observations = []
        if calls and trial_funded:
            if noisy_methods:
                first = acquire(trial_channels, "short_learned_coordinate_trial", rows_count=trial_length // 2)
                second = acquire(trial_channels, "short_learned_coordinate_trial",
                                 rows_count=trial_length - trial_length // 2)
                pilot = [*first, *second]
            else:
                pilot = list(acquire(trial_channels, "short_learned_coordinate_trial", rows_count=trial_length))
            proposed = fit_transfer_candidates(pilot, trial_channels)
            if proposed:
                shared_validation = None
                criterion = "training_loss"
                if command_calls:
                    # Several expressive fits can interpolate a short pilot.
                    # Let their different predictions face the same new data
                    # before discarding an already correct larger mechanism.
                    shared_validation = acquire(trial_channels, "validate_short_learned_trial", rows_count=trial_length)
                    checked = assess_transfer_candidates(proposed, shared_validation, trial_channels)
                    if checked:
                        proposed = checked
                        criterion = "validation_loss"
                lowest = min(item[0][criterion] for item in proposed)
                adequate = [item for item in proposed
                            if item[0][criterion] <= lowest + self.config.minimum_improvement]
                trial, channels, training = min(adequate,
                    key=lambda item: (sum(rates[name] for name in item[1]),
                                      item[0]["candidate"].program.size))
                check_channels = trial_channels if noisy_methods else channels
                validation = shared_validation if shared_validation is not None else acquire(
                    check_channels, "validate_short_learned_trial", [trial], rows_count=trial_length)
                validation_view = view_rows(validation, check_channels, channels)
                assessed = [trial] if criterion == "validation_loss" else [item for item, _, _ in
                    assess_transfer_candidates([(trial, channels, training)], validation, check_channels,
                                               fallback_quantile=.5)]
                if assessed:
                    trial = assessed[0]
                    supported = trial["validation_loss"] <= self.config.minimum_improvement
                    trial_support = None
                    if not supported and noisy_methods and self.config.risk_learning:
                        trial_support = assess_trial_noise(trial["candidate"].program, pilot,
                            validation, channels, check_channels)
                        supported = trial_support["adequate"]
                    condition_ready, condition_status = True, "not_requested"
                    if supported and self.config.active_probes and self.config.strategy != "fixed":
                        from .transfer_condition_research import TransferConditionResearch
                        from .competing_object_experiments import assess_competing_object_probe
                        command_spec = self._spec(laboratory, ())
                        measured_spec = self._spec(laboratory, trial_channels)
                        condition_binding = self._instrument_binding(measured_spec, command_spec)
                        condition_research = TransferConditionResearch(calls)
                        for step in range(self.config.max_rounds + 1):
                            paid_pairs = [pair for batch in paired_for(trial_channels) for pair in batch]
                            plan = condition_research.plan(command_spec, trial["candidate"].program,
                                budget, paid_rows=[pair[1] for pair in paid_pairs],
                                paired_rows=paid_pairs, binding=condition_binding,
                                methods=self.library.program_experiment_methods if self.config.use_memory else None,
                                serial=serial, length=4, tolerance=self.config.minimum_improvement,
                                task_id=f"{laboratory.spec.task_id}:{self.config.seed}")
                            condition_status = plan["forecast_status"]
                            scope = plan["transfer_scope"]
                            if not scope["has_conditional_calls"]:
                                break
                            transfer_observations = list(pilot)
                            actions.append({"action": "plan_transfer_condition_experiment",
                                **{key: value for key, value in plan.items()
                                   if key not in ("probe", "hypotheses", "predicted_fields")},
                                "probe_id": plan["probe"].probe_id if plan["probe"] else None})
                            condition_ready = False
                            if condition_status == "covered_by_paid_evidence":
                                # Removing the old conditional mechanism is a
                                # new explanation requiring ordinary research,
                                # even if the old active region now outputs zero.
                                condition_ready = bool(_sources(trial["candidate"].program))
                                if not condition_ready:
                                    condition_status = "conditional_mechanism_removed"
                                break
                            if plan["probe"] is None or step == self.config.max_rounds:
                                break
                            ordinary_rows = (7 if offered else 4) * length
                            query_rows = plan["probe"].cost + trial_length
                            if (spent_observations + query_rows + ordinary_rows > self.config.observation_budget
                                    or spent_measurements + query_rows * sum(rates[name] for name in trial_channels)
                                    + 4 * length * sum(rates[name] for name in offered) > self.measurement_budget
                                    or budget.remaining < 12000):
                                condition_status = "recovery_budget_reserved"
                                break
                            observed = acquire(trial_channels, "distinguish_transfer_conditions", probe=plan["probe"])
                            if observed is None:
                                break
                            actual = paired_for(trial_channels)[-1]
                            evidence = assess_competing_object_probe(plan, actual, budget,
                                tolerance=self.config.minimum_improvement)
                            actions.append({"action": "test_transfer_condition_predictions",
                                "probe_id": plan["probe"].probe_id,
                                "plan": {key: value for key, value in plan.items() if key != "probe"},
                                "evidence": evidence})
                            if self.config.use_memory:
                                feedback = self.library.program_experiment_methods.observe(plan, evidence,
                                    task_id=f"{laboratory.spec.task_id}:{self.config.seed}",
                                    command_spec=command_spec, binding=condition_binding,
                                    tolerance=self.config.minimum_improvement)
                                if feedback is not None:
                                    actions.append(feedback)
                            if all({name for name, _ in Program.from_dict(item["program"]).observed_variables()}
                                   <= set(command_spec.feature_bounds) | {"__target__"} for item in plan["hypotheses"]):
                                directed_command_rows.extend(pair[0] for pair in actual)
                            pilot.extend(observed)
                            transfer_observations = list(pilot)
                            if (condition_status == "requires_paired_measurements"
                                    or evidence["calibration_status"] == "counterexample"):
                                model = self._fit_instruments(command_spec, measured_spec,
                                    paired_for(trial_channels), budget, actions)
                                if model is not None:
                                    from .instrument_models import rebind_instrument_model
                                    condition_binding = rebind_instrument_model(model, measured_spec)
                                if not condition_binding:
                                    condition_status = "calibration_unresolved"
                                    break
                            before = budget.spent
                            revised = fit_transfer_candidates(pilot, trial_channels)
                            check_channels = trial_channels
                            validation = acquire(check_channels, "validate_short_learned_trial", rows_count=trial_length)
                            if validation is None:
                                break
                            compared = assess_transfer_candidates(revised, validation, check_channels)
                            if not compared:
                                break
                            best_loss = min(max(item["training_loss"], item["validation_loss"])
                                            for item, _, _ in compared)
                            adequate = [entry for entry in compared
                                if max(entry[0]["training_loss"], entry[0]["validation_loss"])
                                <= best_loss + self.config.minimum_improvement]
                            trial, channels, training = min(adequate, key=lambda entry:
                                (sum(rates[name] for name in entry[1]), entry[0]["candidate"].program.size))
                            validation_view = view_rows(validation, check_channels, channels)
                            supported = max(trial["training_loss"], trial["validation_loss"]) <= self.config.minimum_improvement
                            actions.append({"action": "refit_transfer_after_experiment", "supported": supported,
                                "program": trial["candidate"].program.to_dict(), "training_rows": len(training),
                                "training_loss": trial["training_loss"], "validation_loss": trial["validation_loss"],
                                "compute_cost": budget.spent - before})
                            if not supported:
                                condition_status = "current_call_family_refuted"
                                break
                    event = {"source_ids": _sources(trial["candidate"].program),
                        "method_source_ids": trial["method_source_ids"],
                        "program": trial["candidate"].program.to_dict(),
                        "validation_loss": trial["validation_loss"],
                        "validation_rows": len(validation), "accepted": False,
                        "status": "supported" if supported and condition_ready else
                                  "counterexample" if not supported else "unresolved",
                        "condition_status": condition_status}
                    transfer.append(event)
                    actions.append({"action": "test_learned_acquisition_method",
                        "pilot_channels": list(trial_channels), "selected_channels": list(channels),
                        "source_ids": event["source_ids"], "method_source_ids": trial["method_source_ids"],
                        "training_rows": len(training),
                        "validation_rows": len(validation), "validation_loss": trial["validation_loss"],
                        "learned_trial_rows": trial_length, "noise_assessment": trial_support,
                        "validation_channels": list(check_channels),
                        "ordinary_observation_reserve": 7 * length})
                    if not condition_ready:
                        actions.append({"action": "reopen_transfer_condition_research",
                            "reason": condition_status, "retained_observation_rows": len(transfer_observations),
                            "method_source_ids": trial["method_source_ids"]})
                    source_prior = next((prior for prior in reversed(self.library.meta_priors)
                        if prior["source_concept_id"] in event["source_ids"] and prior.get("structure")), None)
                    requires_structure_test = (source_prior is not None
                        and _object_schema(self._spec(laboratory, channels)) is not None
                        and self.config.active_probes)
                    if requires_structure_test:
                        from .unified_structure_transfer import transfer_structure
                        compared_rows = view_rows(validation, check_channels, channels)
                        local = SearchBudget(min(24000, budget.remaining // 3))
                        structures = transfer_structure(self._spec(laboratory, channels),
                            [*training, *compared_rows], local, source_prior,
                            tolerance=self.config.minimum_improvement,
                            transitions=transitions_for(channels, [*training, *compared_rows]))
                        budget.spend(local.spent)
                        if structures:
                            finalists = [{**item, "validation_loss": item["training_loss"],
                                          "selection_basis": "all_acquired_evidence"} for item in structures]
                            selected, selected_channels = self._best(finalists), channels
                            selected_training, selected_validation = training, compared_rows
                            method_trial = True
                            actions.append({"action": "reuse_structural_research_method",
                                "method_source_ids": selected["method_source_ids"],
                                "family": selected["structure"]["fits"][0]["family"],
                                "old_program_supported": supported,
                                "training_loss": selected["training_loss"], "compute_cost": local.spent})
                    if event["status"] == "supported" and not method_trial and not requires_structure_test:
                        calibrated = self._tail_calibration(trial, validation_view,
                            self._spec(laboratory, channels), budget, actions)
                        confirmation = acquire(check_channels, "confirm_short_frozen_program", [trial],
                                               rows_count=trial_length)
                        confirmation_view = view_rows(confirmation, check_channels, channels)
                        trial, confirmation_loss = self._compare_calibration(trial, calibrated,
                            confirmation_view, self._spec(laboratory, channels), budget,
                            initial_prediction(training, laboratory.spec.risk_quantile), actions)
                        if confirmation_loss is not None:
                            confirmation_rows = len(confirmation)
                            confirmed = confirmation_loss <= self.config.minimum_improvement
                            if not confirmed and noisy_methods and self.config.risk_learning:
                                noisy_support = assess_trial_noise(trial["candidate"].program, pilot,
                                    confirmation, channels, check_channels)
                                confirmed = noisy_support["adequate"]
                            actions.append({"action": "validate_frozen_program", "attempt": 0,
                                "program": trial["candidate"].program.to_dict(),
                                "loss": confirmation_loss, "accepted": confirmed,
                                "noise_assessment": noisy_support,
                                "observation_rows": len(confirmation), "short_trial": True})
                            if confirmed:
                                selected, selected_channels, finalists = trial, channels, [trial]
                                selected_training, selected_validation = training, validation_view
                            else:
                                event.update(status="counterexample", confirmation_loss=confirmation_loss)
        prefetched_inputs = None
        if self._revise_inputs and command_calls and not confirmed and not method_trial:
            contradiction = next((action for action in reversed(actions)
                if action["action"] == "test_transfer_condition_predictions"
                and action["evidence"]["status"] == "all_refuted"), None)
            if contradiction is not None and self._learn_instruments:
                # All retained response laws failed on this actual trajectory.
                # Measure the same complete experiment to distinguish a wrong
                # input map from an internal mechanism change, including reset.
                batch = next(batch for batch in batches
                    if batch["probe"]["probe_id"] == contradiction["probe_id"])
                first = acquire(offered, "diagnose_input_mapping")
                boundary = acquire(offered, "distinguish_instrument_boundary", probe=Probe(
                    f"measure_counterexample_{serial}", tuple(batch["probe"]["sequence"])))
                checked = acquire(offered, "compare_input_mapping")
                if first is not None and boundary is not None and checked is not None:
                    prefetched_inputs = ([*first, *boundary], checked)
                    self._fit_instruments(self._spec(laboratory, ()), self._spec(laboratory, offered),
                        paired_for(offered), budget, actions)
                    command_calls = self._recall(self._spec(laboratory, ()))
            # Use the paid command pilot and its boundary counterexamples to
            # ask whether retained mechanisms explain the current response.
            command_batches = [batch for batch in batches if not batch["channels"]]
            if len(command_batches) >= 2:
                pilot, checked = [[Observation(row["features"], row["target"], tuple(row["history"]))
                    for row in batch["observations"]] for batch in command_batches[:2]]
                # The short-call test may have bought boundary counterexamples
                # before reopening this question. They are fitting evidence for
                # its replacement, with the same recorded trajectory histories.
                initial_rows = (*pilot, *checked)
                pilot.extend(row for row in directed_command_rows
                             if row not in initial_rows)
                local = SearchBudget(min(24000, budget.remaining // 3))
                proposed = self._program_claims(self._spec(laboratory, ()), pilot, local, command_calls, actions)
                budget.spend(local.spent)
                assessed = self._assess(proposed, checked, budget,
                    initial_prediction(pilot, laboratory.spec.risk_quantile), laboratory.spec.risk_quantile)
                best = self._best([item for item in assessed
                    if item["training_loss"] <= self.config.minimum_improvement])
                if best is not None and best["validation_loss"] <= self.config.minimum_improvement:
                    selected, selected_channels, finalists = best, (), assessed
                    selected_training, selected_validation = pilot, checked
                    method_trial = True
                    actions.append({"action": "reuse_dependencies_for_new_question",
                        "validation_loss": best["validation_loss"], "compute_cost": local.spent})
                if not method_trial:
                    from .relation_revisions import propose_relation_revisions
                    local = SearchBudget(min(16000, budget.remaining // 3))
                    fit_budget = SearchBudget(local.remaining * 3 // 4)
                    relation_training = [*pilot, *checked]
                    repaired = propose_relation_revisions(self._spec(laboratory, ()),
                        relation_training, fit_budget, command_calls, tolerance=self.config.minimum_improvement)
                    local.spend(fit_budget.spent)
                    if repaired:
                        relation_check = acquire((), "compare_relation_revision")
                        assessed = (self._assess(repaired, relation_check, local,
                            initial_prediction(relation_training, laboratory.spec.risk_quantile),
                            laboratory.spec.risk_quantile) if relation_check else [])
                        best = self._best(assessed)
                        if best is not None and best["validation_loss"] <= self.config.minimum_improvement:
                            selected, selected_channels, finalists = best, (), assessed
                            selected_training, selected_validation = relation_training, relation_check
                            method_trial = True
                        actions.append({"action": "compare_nested_relation_revision",
                            "supported": method_trial, "training_rows": len(relation_training),
                            "validation_rows": len(relation_check or []), "compute_cost": local.spent,
                            "validation_loss": best["validation_loss"] if best else None})
                    budget.spend(local.spent)
            if (not method_trial and command_batches and laboratory.spec.risk_quantile == .5
                    and not _object_schema(self._spec(laboratory, ()))
                    and self.config.observation_budget - spent_observations >= 7 * length):
                from .control_revisions import RetainedControlRevisions
                local = SearchBudget(min(12000, budget.remaining // 3))
                revisions = RetainedControlRevisions(command_calls, self._spec(laboratory, ()),
                    local, max_history=self.config.max_history)
                control_training = [Observation(row["features"], row["target"], tuple(row["history"]))
                    for batch in command_batches for row in batch["observations"]]
                expanded_evidence = False
                for temporal in (False, True):
                    if not any(bool(item["lag_mapping"]) == temporal for item in revisions.coordinates):
                        continue
                    # A failed short fit justifies one richer control-history
                    # trajectory before paying for a new instrument mapping.
                    if temporal and not expanded_evidence:
                        extra = acquire((), "distinguish_control_history_revision")
                        if extra is None:
                            break
                        control_training.extend(extra)
                    start = local.spent
                    claims = revisions.propose(control_training, local, temporal=temporal)
                    promising = [claim for claim in claims
                        if claim["training_loss"] <= self.config.minimum_improvement]
                    checked = acquire((), "compare_control_revision") if promising else None
                    assessed = (self._assess(promising, checked, local,
                        initial_prediction(control_training, laboratory.spec.risk_quantile),
                        laboratory.spec.risk_quantile) if checked else [])
                    best = self._best(assessed)
                    supported = best is not None and best["validation_loss"] <= self.config.minimum_improvement
                    actions.append({"action": "compare_control_revision_before_measurement",
                        "direction": "history" if temporal else "mechanism", "supported": supported,
                        "compute_cost": local.spent - start, "training_rows": len(control_training),
                        "validation_rows": len(checked or []),
                        "alternatives": [{"training_loss": claim["training_loss"],
                            "control_revision": claim["control_revision"]} for claim in claims],
                        "validation_loss": best["validation_loss"] if best else None})
                    if supported:
                        selected, selected_channels, finalists = best, (), assessed
                        selected_training, selected_validation = list(control_training), checked
                        method_trial = True
                        break
                    if checked:
                        control_training.extend(checked)
                        expanded_evidence = True
                budget.spend(local.spent)
            if not method_trial:
                first, second = prefetched_inputs or (acquire(offered, "diagnose_input_mapping"),
                    acquire(offered, "compare_input_mapping"))
                if first is not None and second is not None:
                    if prefetched_inputs is None:
                        self._fit_instruments(self._spec(laboratory, ()), self._spec(laboratory, offered),
                                              paired_for(offered), budget, actions)
                    prefetched_inputs = (first, second)
                    candidates = []
                    for call in self._recall(self._spec(laboratory, offered)):
                        if not budget.spend(len(first)):
                            break
                        candidates.append({"candidate": Candidate(f"input_repair:{len(candidates)}", call,
                            "revise_input_dependency", parents=tuple(_sources(call))),
                            "kind": "input_dependency_revision", "family_id": "T03",
                            "method_source_ids": _sources(call),
                            "training_loss": loss(call, first, 0., laboratory.spec.risk_quantile)})
                    assessed = self._assess(candidates, second, budget, 0., laboratory.spec.risk_quantile)
                    intact = [item for item in assessed if max(item["training_loss"], item["validation_loss"])
                              <= self.config.minimum_improvement]
                    best = self._best(intact)
                    actions.append({"action": "distinguish_input_and_mechanism_revision",
                        "intact_mechanism_found": best is not None,
                        "measured_loss": min((item["validation_loss"] for item in assessed), default=None),
                        "rows": len(first) + len(second)})
                    if best is not None:
                        binding = self._instrument_binding(self._spec(laboratory, offered), self._spec(laboratory, ()))
                        converted = bind_measurement_arguments(best["candidate"].program, binding)
                        if {name for name, _ in converted.observed_variables()} <= set(laboratory.spec.feature_bounds):
                            selected = {**best, "candidate": replace(best["candidate"], program=converted)}
                            selected_channels, finalists, method_trial = (), [selected], True
                            selected_training, selected_validation = [[pair[0] for pair in batch]
                                                                      for batch in paired_for(offered)[-2:]]
        stages = [offered] if self.config.strategy == "fixed" or prefetched_inputs else [(), offered]
        instrument_views_tested = not offered
        for stage, channels in enumerate(dict.fromkeys(stages) if not confirmed and not method_trial else ()):
            if prefetched_inputs is not None and channels == offered:
                first, second = prefetched_inputs
                prefetched_inputs = None
            else:
                first, second = acquire(channels, "fit"), acquire(channels, "fit")
            if first is None or second is None:
                break
            validation = acquire(channels, "compare_on_new_trajectory", finalists)
            if validation is None:
                break
            instrument_views_tested |= tuple(channels) == offered
            training = [*first, *second]
            if transfer_observations and tuple(channels) in views_from(tuple(trial_channels)):
                training.extend(view_rows(transfer_observations, trial_channels, channels))
            views = [channels] if self.config.strategy == "fixed" else views_from(channels)
            views.sort(key=lambda names: (sum(rates[name] for name in names), len(names)))
            available = []
            for names in views:
                spec = self._spec(laboratory, names)
                train_view = view_rows(training, channels, names)
                validation_view = view_rows(validation, channels, names)
                view_budget = SearchBudget(min(90000, budget.remaining // 2)) if alternative_views else budget
                claims = self._research_view(spec, train_view, validation_view, view_budget, actions,
                                             transfer, graph, parent=parent,
                                             transitions=transitions_for(names, train_view))
                if alternative_views:
                    budget.spend(view_budget.spent)
                # A new random trajectory cannot overrule a paid transfer
                # counterexample. Prefer explanations of both sets of data.
                compatible = [item for item in claims
                    if item["training_loss"] <= self.config.minimum_improvement] if transfer_observations else []
                best = self._best(compatible or claims)
                if best is not None:
                    available.append((best, names, claims, train_view, validation_view))
                    if best["validation_loss"] <= self.config.minimum_improvement:
                        break
            if not available:
                break
            lowest = min(item[0]["validation_loss"] for item in available)
            candidates = [item for item in available
                          if item[0]["validation_loss"] <= lowest + self.config.minimum_improvement]
            selected, selected_channels, finalists, selected_training, selected_validation = min(
                candidates, key=lambda item: (sum(rates[name] for name in item[1]),
                                              item[0]["candidate"].program.size))
            if selected["validation_loss"] <= self.config.minimum_improvement or channels:
                break
            parent = selected
            if command_calls:
                actions.append({"action": "reopen_instrument_question",
                    "reason": "command_prediction_conflict_requires_measured_evidence",
                    "validation_loss": selected["validation_loss"]})
            actions.append({"action": "open_joint_measurement_history_question",
                "reason": "later_trajectory_residual_unexplained_by_paid_coordinates",
                "validation_loss": selected["validation_loss"],
                "parent_program": selected["candidate"].program.to_dict(),
                "offered_channels": list(offered)})

        def research_selected_structure():
            nonlocal selected, selected_training, finalists
            if selected is not None and selected.get("structure") and not confirmed:
                from .structure_lab import ObjectSchema, StructureBatch
                from .unified_structure import (propose_structures, structural_probe,
                    structural_experiments, assess_structural_experiments)
                spec = self._spec(laboratory, selected_channels)
                partial_objects = bool(spec.context.get("object_model") and not _object_schema(spec))
                if partial_objects:
                    propose_structures = partial(self._object_explanations, actions=actions)
                method_sources = selected.get("method_source_ids", [])
                method_family = selected["structure"]["fits"][0]["family"]
                # A radius consistent with one trajectory may fail the other. All
                # already paid evidence constrains the next intervention; only a
                # later untouched trajectory can confirm the resulting program.
                training_ids = {id(row) for row in selected_training}
                selected_training = [*selected_training,
                    *(row for row in selected_validation if id(row) not in training_ids)]

                def fitted_best(entries):
                    return self._best([{**entry, "validation_loss": entry["training_loss"],
                                       "selection_basis": "all_acquired_evidence"}
                                       for entry in entries])

                from .competing_object_experiments import (
                    plan_competing_object_probe, assess_competing_object_probe)
                rivals = []
                if selected.get("object_competition"):
                    local = SearchBudget(min(90000, budget.remaining // 2))
                    rivals = propose_structures(spec, selected_training, local, parent=selected,
                        tolerance=self.config.minimum_improvement, transitions=transitions_for(selected_channels),
                        new_observations=selected_validation)
                    budget.spend(local.spent)
                    if rivals:
                        selected = fitted_best(rivals)
                experiments = []
                schema = ObjectSchema.from_dict(selected["structure"]["schema"])
                plans = structural_experiments(selected, spec, serial, length=4,
                    paired_observations=[pair for batch in paired_for(selected_channels) for pair in batch]
                    ) if self.config.active_probes else ()
                for plan in plans:
                    required_rows = plan["baseline"].cost + plan["probe"].cost + length
                    if (spent_observations + required_rows > self.config.observation_budget or
                            spent_measurements + required_rows * sum(rates[name] for name in selected_channels)
                            > self.measurement_budget):
                        break
                    baseline = acquire(selected_channels, f"test_{plan['kind']}_baseline", probe=plan["baseline"])
                    baseline_batch = StructureBatch(baseline, batches[-1]["transitions"], schema, 0, 0)
                    changed = acquire(selected_channels, f"test_{plan['kind']}_intervention", probe=plan["probe"])
                    changed_batch = StructureBatch(changed, batches[-1]["transitions"], schema, 0, 0)
                    selected_training.extend([*baseline, *changed])
                    experiments.append({"kind": plan["kind"], "baseline": baseline_batch, "batch": changed_batch})
                tested_hypotheses = assess_structural_experiments(selected, experiments, budget)
                if experiments:
                    actions.append({"action": "test_structural_hypotheses", "hypotheses": tested_hypotheses,
                        "experiments": [{"kind": item["kind"], "rows": len(item["baseline"].observations)
                            + len(item["batch"].observations)} for item in experiments]})
                local = SearchBudget(min(90000, budget.remaining // 2))
                revised = propose_structures(spec, selected_training, local,
                    parent=selected, tolerance=self.config.minimum_improvement,
                    transitions=transitions_for(selected_channels),
                    **({"new_observations": [row for item in experiments
                        for batch in (item["baseline"], item["batch"]) for row in batch.observations]}
                       if partial_objects else {}))
                budget.spend(local.spent)

                if revised:
                    selected = fitted_best(revised)
                rivals = revised
                for index in range(2 * self.config.max_rounds if self.config.active_probes else 0):
                    if selected["training_loss"] > self.config.minimum_improvement:
                        break
                    if budget.remaining < (len(selected_training) + 5) * 60:
                        break
                    plan, probe = None, None
                    if selected.get("object_competition"):
                        command_spec = self._spec(laboratory, ())
                        planning_binding = self._instrument_binding(spec, command_spec)
                        plan = plan_competing_object_probe(command_spec, rivals, budget,
                            binding=planning_binding, serial=serial,
                            length=5, tolerance=self.config.minimum_improvement,
                            methods=self.library.program_experiment_methods if self.config.use_memory else None,
                            task_id=f"{laboratory.spec.task_id}:{self.config.seed}")
                        actions.append({"action": "plan_competing_object_experiment",
                            **{key: value for key, value in plan.items()
                               if key not in ("probe", "hypotheses", "predicted_fields")},
                            "probe_id": plan["probe"].probe_id if plan["probe"] else None,
                            "hypotheses": len(rivals)})
                        probe = plan["probe"]
                    if probe is None:
                        probe = structural_probe(selected, spec, serial, 5,
                            paired_observations=[pair for batch in paired_for(selected_channels) for pair in batch])
                    if probe is None:
                        break
                    # Keep the next refit and a fresh confirmation affordable.
                    required_rows = probe.cost + length
                    if (budget.remaining < (len(selected_training) + probe.cost) * 60 or
                            spent_observations + required_rows > self.config.observation_budget or
                            spent_measurements + required_rows * sum(rates[name] for name in selected_channels)
                            > self.measurement_budget):
                        break
                    planned = plan is not None and plan["probe"] is not None
                    interval = selected["structure"].get("radius_interval")
                    measured = acquire(selected_channels,
                        "distinguish_competing_object_programs" if planned else "resolve_structure_boundary",
                        probe=probe)
                    if measured is None:
                        break
                    if planned:
                        evidence = assess_competing_object_probe(plan, paired_for(selected_channels)[-1], budget,
                            tolerance=self.config.minimum_improvement)
                        actions.append({"action": "test_competing_object_predictions", "probe_id": probe.probe_id,
                            "plan": {key: value for key, value in plan.items() if key != "probe"},
                            "evidence": evidence})
                        if self.config.use_memory:
                            method_feedback = self.library.program_experiment_methods.observe(plan, evidence,
                                task_id=f"{laboratory.spec.task_id}:{self.config.seed}",
                                command_spec=command_spec, binding=planning_binding,
                                tolerance=self.config.minimum_improvement)
                            if method_feedback is not None:
                                actions.append(method_feedback)
                        if (plan["forecast_status"] == "requires_paired_measurements"
                                or evidence["calibration_status"] == "counterexample"):
                            self._fit_instruments(command_spec, spec, paired_for(selected_channels), budget, actions)
                    selected_training.extend(measured)
                    local = SearchBudget(min(30000, budget.remaining // 2))
                    revised = propose_structures(spec, selected_training, local,
                        parent=selected, tolerance=self.config.minimum_improvement,
                        transitions=transitions_for(selected_channels),
                        **({"new_observations": measured} if partial_objects else {}))
                    budget.spend(local.spent)
                    if planned:
                        actions.append({"action": "revise_competing_object_programs", "probe_id": probe.probe_id,
                            "before": len(rivals), "after": len(revised), "compute_cost": local.spent})
                    rivals = revised
                    if not revised:
                        break
                    selected = fitted_best(revised)
                    actions.append({"action": "revise_structure_boundary", "probe_id": probe.probe_id,
                        "before_interval": interval,
                        "after_interval": selected["structure"].get("radius_interval"),
                        "training_loss": selected["training_loss"], "compute_cost": local.spent})
                selected["structure"]["hypotheses"] = tested_hypotheses
                if method_sources and selected["structure"]["fits"][0]["family"] == method_family:
                    selected["method_source_ids"] = method_sources
                    selected["kind"] = f"structural_method_transfer:{method_family}"
                    selected["candidate"] = replace(selected["candidate"], parents=tuple(dict.fromkeys(
                        (*selected["candidate"].parents, *method_sources))))
                finalists = [selected]

        relation_inputs_unresolved = False

        def inspect_relation_inputs(origin, plan, evidence):
            nonlocal selected, finalists, confirmed, command_calls, relation_inputs_unresolved
            from .relation_input_diagnosis import diagnose_relation_inputs
            diagnosis = diagnose_relation_inputs(self, laboratory, selected, origin, plan, evidence,
                acquire, paired_for, budget, actions, channels=selected_channels)
            feedback_plan, feedback_evidence = plan, evidence
            feedback_binding = self._instrument_binding(self._spec(laboratory, selected_channels),
                                                        self._spec(laboratory, ()))
            if diagnosis is not None:
                new_rows = diagnosis["command_rows"]
                selected_training.extend(new_rows)
                inherited_relation_evidence.extend(new_rows)
                directed_command_rows.extend(new_rows)
                confirmed = False
                relation_inputs_unresolved = diagnosis["status"] == "unresolved_inputs"
                feedback_plan, feedback_evidence = diagnosis["plan"], diagnosis["evidence"]
                feedback_binding = diagnosis["binding"]
                if diagnosis["status"] == "inputs_revised":
                    selected, finalists = diagnosis["entry"], [diagnosis["entry"]]
                    command_calls = self._recall(self._spec(laboratory, ()))
            # A command response cannot support a relation method until the
            # same experiment has resolved its suspected input calibration.
            if self.config.use_memory and (diagnosis is None or
                    feedback_evidence["calibration_status"] != "not_forecast"):
                feedback = self.library.program_experiment_methods.observe(feedback_plan, feedback_evidence,
                    task_id=f"{laboratory.spec.task_id}:{self.config.seed}",
                    command_spec=self._spec(laboratory, ()), binding=feedback_binding,
                    tolerance=self.config.minimum_improvement)
                if feedback is not None:
                    actions.append(feedback)
            return diagnosis

        def research_selected_program_conditions():
            nonlocal selected, selected_training, finalists
            if (selected is None or selected.get("structure") or confirmed or
                    not self.config.active_probes or self.config.strategy == "fixed"):
                return
            origin = deepcopy(selected)
            steps = (selected.get("research_program") or {}).get("selection_steps", [])
            type_steps = [step for step in steps if step["operation"] in
                          {"revise_relation_type", "revise_relation_condition"}]
            if type_steps:
                from .competing_object_experiments import plan_competing_object_probe, assess_competing_object_probe
                retained = [Program.from_dict(step["program"]) for step in steps
                            if step["operation"] == "retain_composed_call"]
                comparisons = [selected, *[{"candidate": Candidate(f"retained_relation:{index}", call,
                    "retained_relation")} for index, call in enumerate(retained)]]
                step = type_steps[-1]
                condition_revision = step["operation"] == "revise_relation_condition"
                command_spec = self._spec(laboratory, ())
                spec = self._spec(laboratory, selected_channels)
                binding = self._instrument_binding(spec, command_spec)
                depth = selected["candidate"].program.history_depth
                for direction in (1., -1.):
                    questions = []
                    for index, expression in enumerate(step["relation_expressions"]):
                        distance = Program("mul", children=(Program("const", value=direction),
                                                            Program.from_dict(expression)))
                        predicate = Program("and", children=(
                            Program("lt", children=(Program("const", value=0.), distance)),
                            Program("lt", children=(distance, Program("const", value=step["radius"])))))
                        questions.append({"question_id": f"relation_direction:{direction}:{index}",
                            "predicate": predicate, "history_depth": depth,
                            "uncovered_phases": ["enter", "maintain", "exit"] if depth else ["active"],
                            "dependent_hypothesis_ids": [item["candidate"].candidate_id for item in comparisons]})
                    if condition_revision:
                        questions = []
                        for index, encoded in enumerate(step["condition_questions"]):
                            edge, condition = Program.from_dict(encoded).children
                            predicate = Program("and", children=(edge, condition if direction > 0. else
                                Program("not", children=(condition,))))
                            questions.append({"question_id": f"relation_condition:{direction}:{index}",
                                "predicate": predicate, "history_depth": depth,
                                "uncovered_phases": ["enter", "maintain", "exit"] if depth else ["active"],
                                "dependent_hypothesis_ids": [item["candidate"].candidate_id for item in comparisons]})
                    probe_reserve = max(5, 2 * depth + 3)
                    if (spent_observations + probe_reserve + length > self.config.observation_budget or
                        spent_measurements + (probe_reserve + length) * sum(rates[name] for name in selected_channels)
                            > self.measurement_budget):
                        break
                    plan = plan_competing_object_probe(command_spec, comparisons, budget, binding=binding,
                        serial=serial, length=5, tolerance=self.config.minimum_improvement,
                        methods=self.library.program_experiment_methods if self.config.use_memory else None,
                        task_id=f"{laboratory.spec.task_id}:{self.config.seed}", questions=questions)
                    if plan["probe"] is None:
                        continue
                    observed = acquire(selected_channels, "distinguish_relation_conditions" if condition_revision else
                        "distinguish_relation_types", probe=plan["probe"])
                    if observed is None:
                        break
                    actual = paired_for(selected_channels)[-1]
                    evidence = assess_competing_object_probe(plan, actual, budget,
                        tolerance=self.config.minimum_improvement)
                    selected_training.extend(observed)
                    inherited_relation_evidence.extend(observed)
                    if not selected_channels:
                        directed_command_rows.extend(pair[0] for pair in actual)
                    actions.append({"action": "test_relation_condition" if condition_revision else "test_relation_type",
                        "relation": step["relation"],
                        **({"predicate": step["predicate"], "previous_predicate": step["previous_predicate"]}
                           if condition_revision else {}),
                        "previous_relation": step["previous_relation"], "direction": direction,
                        "observation_rows": len(observed), "evidence": evidence,
                        "plan": {key: value for key, value in plan.items() if key != "probe"}})
                    diagnosis = inspect_relation_inputs(origin, plan, evidence)
                    if diagnosis is not None and diagnosis["status"] != "inputs_supported":
                        # The inherited-interval entry below restarts from the
                        # original relation claim on its newly measured inputs.
                        return
                    if evidence["calibration_status"] == "counterexample":
                        self._fit_instruments(command_spec, spec, paired_for(selected_channels), budget, actions)
                        binding = self._instrument_binding(spec, command_spec)
            relation_calls = [Program.from_dict(step["program"]) for step in steps
                if step["operation"] == "retain_composed_call"] if any(
                step["operation"] == "revise_nested_relation" for step in steps) else []

            def refine_relation_space():
                nonlocal selected, finalists
                from .relation_revisions import propose_relation_revisions
                previous = [(step.get("radius_interval")) for step in
                    (selected.get("research_program") or {}).get("selection_steps", [])
                    if step["operation"] == "revise_nested_relation"]
                paid = [*selected_training, *selected_validation]
                local = SearchBudget(min(12000, budget.remaining // 2))
                proposed = propose_relation_revisions(self._spec(laboratory, selected_channels),
                    paid, local, relation_calls, tolerance=self.config.minimum_improvement)
                budget.spend(local.spent)
                # These rows now select the radius cell as well as its linear
                # coefficients. A separate final trajectory still confirms it.
                candidates = [{**item, "validation_loss": item["training_loss"],
                               "selection_basis": "all_acquired_evidence"} for item in proposed]
                if candidates:
                    selected, finalists = self._best(candidates), candidates
                actions.append({"action": "refine_relation_parameter_space",
                    "before_intervals": previous, "evidence_rows": len(paid),
                    "after_intervals": [step["radius_interval"] for step in
                        (selected.get("research_program") or {}).get("selection_steps", [])
                        if step["operation"] == "revise_nested_relation"] if candidates else [],
                    "candidate_count": len(candidates), "compute_cost": local.spent,
                    "selection_basis": "all_acquired_evidence"})
                return bool(candidates)

            if relation_calls and not refine_relation_space():
                relation_calls = []
            for _ in range((2 if relation_calls else 1) * self.config.max_rounds):
                intervals = [step["radius_interval"] for step in
                    (selected.get("research_program") or {}).get("selection_steps", [])
                    if step["operation"] == "revise_nested_relation"]
                if relation_calls and intervals and all(high - low < 1e-6 for low, high in intervals):
                    break
                rivals = [item for item in finalists
                    if max(item.get("training_loss", float("inf")), item.get("validation_loss", float("inf")))
                    <= self.config.minimum_improvement]
                probe_reserve = max(5, min(laboratory.spec.max_probe_length,
                    2 * selected["candidate"].program.history_depth + 3)) if relation_calls else 5
                if len(rivals) < 2 or budget.remaining < (len(selected_training) + length + probe_reserve) * 60:
                    break
                if (spent_observations + length + probe_reserve > self.config.observation_budget or
                        spent_measurements + (length + probe_reserve) * sum(rates[name] for name in selected_channels)
                        > self.measurement_budget):
                    break
                questions = []
                if relation_calls:
                    relation_steps = [step for item in rivals for step in
                        item["research_program"]["selection_steps"] if step["operation"] == "revise_nested_relation"]
                    low = min(step["radius"] for step in relation_steps)
                    high = max(step["radius"] for step in relation_steps)
                    for index, expression in enumerate(relation_steps[0]["relation_expressions"]):
                        distance = Program.from_dict(expression)
                        predicate = Program("and", children=(
                            Program("ge", children=(distance, Program("const", value=low))),
                            Program("lt", children=(distance, Program("const", value=high)))))
                        depth = max(item["candidate"].program.history_depth for item in rivals)
                        questions.append({"question_id": f"relation_interval:{index}", "predicate": predicate,
                            "history_depth": depth,
                            "uncovered_phases": ["enter", "maintain", "exit"] if depth else ["active"],
                            "dependent_hypothesis_ids": [item["candidate"].candidate_id for item in rivals]})
                observed = acquire(selected_channels, "distinguish_program_conditions", rivals,
                    rows_count=5, questions=questions)
                if observed is None:
                    break
                actual_probe = observation_probes[id(observed[0])]
                tested = next(action for action in reversed(actions)
                    if action["action"] == "test_competing_program_predictions" and
                       action["probe_id"] == actual_probe.probe_id)
                plan = {**tested["plan"], "probe": actual_probe}
                diagnosis = inspect_relation_inputs(origin, plan, tested["evidence"])
                selected_training.extend(observed)
                inherited_relation_evidence.extend(observed)
                if diagnosis is not None and diagnosis["status"] != "inputs_supported":
                    return
                if relation_calls:
                    if refine_relation_space():
                        continue
                    relation_calls = []
                before = budget.spent
                revised = self._research_view(self._spec(laboratory, selected_channels), selected_training,
                    selected_validation, budget, actions, transfer, graph, parent=selected,
                    transitions=transitions_for(selected_channels))
                if not revised:
                    break
                # The newly purchased counterexample participates in choice
                # even when the previous random validation missed that region.
                best_training = min(item["training_loss"] for item in revised)
                selected = self._best([item for item in revised
                    if item["training_loss"] <= best_training + self.config.minimum_improvement])
                finalists = revised
                actions.append({"action": "revise_from_program_experiment",
                    "observation_rows": len(observed), "compute_cost": budget.spent - before,
                    "training_loss": selected["training_loss"], "candidate_id": selected["candidate"].candidate_id})

        research_selected_structure()
        research_selected_program_conditions()

        def research_inherited_relation_intervals():
            nonlocal selected, selected_training, finalists, confirmed
            if (selected is None or relation_inputs_unresolved or not self.config.active_probes or
                    self.config.strategy == "fixed"):
                return
            from .relation_uncertainty import RelationIntervalResearch
            from .competing_object_experiments import plan_competing_object_probe, assess_competing_object_probe
            origin = deepcopy(selected)
            start = budget.spent
            research = RelationIntervalResearch(selected["candidate"].program, self.library.meta_priors,
                budget, structure=selected.get("structure"), research_program=selected.get("research_program"))
            if not research.parameters:
                return
            actions.append({"action": "inherit_relation_intervals", "intervals": research.intervals,
                            "evidence_contexts": len(research.evidence_contexts),
                            "inherited_trajectories": sum(len(context["trajectories"])
                                for context in research.evidence_contexts),
                            "compute_cost": budget.spent - start})
            paid, identities = [], set()
            for row in [*selected_training, *selected_validation]:
                if id(row) not in identities:
                    identities.add(id(row))
                    paid.append(row)
            # A short recall confirmation becomes selection evidence if an
            # inherited parameter is subsequently investigated.
            if confirmed:
                paid.extend(pair[1] for pair in paired_for(selected_channels)[-1])
            spec, command_spec = self._spec(laboratory, selected_channels), self._spec(laboratory, ())
            binding = self._instrument_binding(spec, command_spec)
            distinguishing_trajectories = []
            while budget.remaining > (len(paid) + length) * 10:
                before = budget.spent
                updates_before = len(research.evidence_updates)
                candidates = research.alternatives(paid, budget, tolerance=self.config.minimum_improvement,
                                                   experiments=distinguishing_trajectories)
                for update in research.evidence_updates[updates_before:]:
                    actions.append({"action": "reopen_inherited_relation_evidence", **update})
                actions.append({"action": "condition_inherited_relation_intervals",
                    "intervals": research.intervals, "status": research.status,
                    "evidence_rows": len(paid), "candidate_count": len(candidates),
                    "distinguishing_trajectories": len(distinguishing_trajectories),
                    "compute_cost": budget.spent - before})
                if not candidates:
                    if research.status == "refuted":
                        confirmed = False
                        repaired = self._research_view(spec, paid, [], budget, actions, transfer,
                            graph, parent=selected, transitions=transitions_for(selected_channels))
                        if repaired:
                            selected, finalists = self._best(repaired), repaired
                            selected_training = list(paid)
                    break
                candidate = self._best(candidates)
                if candidate["candidate"].program != selected["candidate"].program:
                    confirmed = False
                selected, finalists = candidate, candidates
                research.adopt(selected)
                validation_ids = {id(row) for row in selected_validation}
                selected_training = [row for row in paid if id(row) not in validation_ids]
                questions = research.questions(candidates)
                if not questions or all(high - low <= 1e-6
                    for item in research.intervals for low, high in [item["radius_interval"]]):
                    break
                probe_rows = min(command_spec.max_probe_length,
                    max(5, 2 * selected["candidate"].program.history_depth + 3))
                if (spent_observations + probe_rows + length > self.config.observation_budget or
                    spent_measurements + (probe_rows + length) * sum(rates[name] for name in selected_channels)
                        > self.measurement_budget):
                    break
                plan = plan_competing_object_probe(command_spec, candidates, budget, binding=binding,
                    serial=serial, length=5, tolerance=self.config.minimum_improvement,
                    methods=self.library.program_experiment_methods if self.config.use_memory else None,
                    task_id=f"{laboratory.spec.task_id}:{self.config.seed}", questions=questions)
                actions.append({"action": "plan_inherited_relation_experiment",
                    **{key: value for key, value in plan.items()
                       if key not in ("probe", "hypotheses", "question_hypotheses", "predicted_fields")},
                    "probe_id": plan["probe"].probe_id if plan["probe"] else None})
                if plan["probe"] is None:
                    break
                if (spent_observations + plan["probe"].cost + length > self.config.observation_budget or
                    spent_measurements + (plan["probe"].cost + length) * sum(rates[name] for name in selected_channels)
                        > self.measurement_budget):
                    break
                observed = acquire(selected_channels, "distinguish_inherited_relation", probe=plan["probe"])
                if observed is None:
                    break
                actual = paired_for(selected_channels)[-1]
                evidence = assess_competing_object_probe(plan, actual, budget,
                    tolerance=self.config.minimum_improvement)
                actions.append({"action": "test_inherited_relation_predictions", "probe_id": plan["probe"].probe_id,
                    "plan": {key: value for key, value in plan.items() if key != "probe"}, "evidence": evidence})
                diagnosis = inspect_relation_inputs(origin, plan, evidence)
                if not selected_channels:
                    directed_command_rows.extend(pair[0] for pair in actual)
                if (plan["forecast_status"] == "requires_paired_measurements" or
                        evidence["calibration_status"] == "counterexample"):
                    self._fit_instruments(command_spec, spec, paired_for(selected_channels), budget, actions)
                    binding = self._instrument_binding(spec, command_spec)
                paid.extend(observed)
                selected_training.extend(observed)
                inherited_relation_evidence.extend(observed)
                if evidence["status"] == "distinguished":
                    # Keep the paid experimental trajectory's own loss test.
                    # Other fitting rows cannot restore a refuted alternative.
                    distinguishing_trajectories.append(tuple(observed))
                confirmed = False
                if diagnosis is not None:
                    paid.extend(diagnosis["command_rows"])
                    if diagnosis["status"] == "inputs_revised":
                        # Discard inferences made with the refuted input map,
                        # then consume every paid row under the corrected map.
                        research = RelationIntervalResearch(selected["candidate"].program,
                            self.library.meta_priors, budget, structure=selected.get("structure"),
                            research_program=selected.get("research_program"))
                        if research.reopen("observed_input_revision"):
                            actions.append({"action": "reopen_inherited_relation_evidence",
                                            **research.evidence_updates[-1]})
                        origin = deepcopy(selected)
                        binding = self._instrument_binding(spec, command_spec)
                        actions.append({"action": "restart_relation_parameters_after_input_revision",
                            "intervals": research.intervals, "evidence_rows": len(paid)})
                    elif diagnosis["status"] == "unresolved_inputs":
                        break

        research_inherited_relation_intervals()

        fallback = initial_prediction(selected_training, laboratory.spec.risk_quantile)
        if selected is None:
            selected = {"candidate": Candidate("unified:initial", Program("const", value=fallback), "initial"),
                        "kind": "initial", "family_id": "T02", "validation_loss": None}
        program = selected["candidate"].program
        rounds = self.config.max_rounds if laboratory.spec.risk_quantile != .5 else min(2, self.config.max_rounds)
        for attempt in range(rounds if not confirmed else 0):
            calibrated = self._tail_calibration(selected, selected_validation,
                self._spec(laboratory, selected_channels), budget, actions)
            confirmation = acquire(selected_channels, "confirm_frozen_program", finalists,
                rows_count=min(laboratory.spec.max_probe_length, length * (2 ** attempt)))
            if confirmation is None:
                break
            selected, confirmation_loss = self._compare_calibration(selected, calibrated,
                confirmation, self._spec(laboratory, selected_channels), budget, fallback, actions)
            if confirmation_loss is None:
                break
            program = selected["candidate"].program
            confirmation_rows = len(confirmation)
            confirmed = confirmation_loss <= self.config.minimum_improvement
            known_counterexample = self._object_program_refuted(selected,
                self._spec(laboratory, selected_channels))
            if known_counterexample:
                confirmed = False
                actions.append({"action": "retain_object_counterexample",
                    "reason": "later_random_agreement_does_not_resolve_observed_failure",
                    "hypothesis_id": selected.get("hypothesis_id"),
                    "confirmation_loss": confirmation_loss})
            if directed_command_rows and {name for name, _ in program.observed_variables()} <= (
                    set(laboratory.spec.feature_bounds) | {"__target__"}):
                checked = budget.spend(len(directed_command_rows))
                directed_loss = loss(program, directed_command_rows, fallback, laboratory.spec.risk_quantile) if checked else None
                if directed_loss is None or directed_loss > self.config.minimum_improvement:
                    confirmed = False
                    known_counterexample = True
                    actions.append({"action": "retain_program_experiment_counterexample",
                        "loss": directed_loss, "observation_rows": len(directed_command_rows),
                        "confirmation_loss": confirmation_loss})
            required = {name for name, _ in program.observed_variables()} - {"__target__"}
            relevant = [row for row in inherited_relation_evidence if required <= set(row.features)]
            if relevant:
                checked = budget.spend(len(relevant))
                inherited_loss = loss(program, relevant, fallback, laboratory.spec.risk_quantile) if checked else None
                if inherited_loss is None or inherited_loss > self.config.minimum_improvement:
                    confirmed = False
                    known_counterexample = True
                    actions.append({"action": "retain_inherited_relation_counterexample",
                        "loss": inherited_loss, "observation_rows": len(relevant),
                        "confirmation_loss": confirmation_loss})
            noisy_support = None
            if not confirmed and not known_counterexample and self.config.risk_learning and instrument_views_tested:
                noisy_support = self._support(program, selected_training, confirmation,
                    self._spec(laboratory, selected_channels), budget)
                noisy_support = challenge_noise(noisy_support, program, confirmation,
                    selected_channels, selected_channels, fallback)
                confirmed = noisy_support["adequate"]
            actions.append({"action": "validate_frozen_program", "attempt": attempt,
                "program": program.to_dict(), "loss": confirmation_loss, "accepted": confirmed,
                "noise_assessment": noisy_support,
                "observation_rows": len(confirmation)})
            if confirmed or self.config.strategy == "fixed":
                break
            # Preserve the actual failure as training evidence. Any changed
            # candidate needs another unqueried trajectory before adoption.
            selected_training.extend(confirmation)
            if attempt + 1 < rounds:
                if self._revise_inputs and not selected_channels and command_calls:
                    # A new response counterexample can distinguish retained
                    # relation cells while the measured input map stays valid.
                    # Keep this full-composition revision in competition before
                    # purchasing the same instruments again.
                    from .relation_revisions import propose_relation_revisions
                    local = SearchBudget(min(20000, budget.remaining // 3))
                    fit_budget = SearchBudget(local.remaining * 3 // 4)
                    repaired = propose_relation_revisions(self._spec(laboratory, ()),
                        selected_training, fit_budget, command_calls,
                        tolerance=self.config.minimum_improvement)
                    local.spend(fit_budget.spent)
                    assessed = self._assess(repaired, selected_validation, local,
                        fallback, laboratory.spec.risk_quantile)
                    best = self._best(assessed)
                    supported = best is not None and best["validation_loss"] <= self.config.minimum_improvement
                    budget.spend(local.spent)
                    if repaired:
                        actions.append({"action": "revise_relation_from_confirmation",
                            "training_rows": len(selected_training), "compute_cost": local.spent,
                            "validation_loss": best["validation_loss"] if best else None,
                            "requires_fresh_confirmation": supported})
                    if supported:
                        selected, finalists = best, assessed
                        research_selected_program_conditions()
                        program = selected["candidate"].program
                        continue
                if self._learn_instruments and not selected_channels and offered:
                    # A cheap surrogate may pass its small pilot yet fail a
                    # later trajectory. Acquire instruments again so the
                    # changed input map and response law can be distinguished.
                    actions.append({"action": "reopen_instrument_question",
                        "reason": "fresh_command_confirmation_failed", "loss": confirmation_loss})
                    first = acquire(offered, "reopen_instrument_fit")
                    second = acquire(offered, "reopen_instrument_fit")
                    checked = acquire(offered, "compare_on_new_trajectory")
                    if first is not None and second is not None and checked is not None:
                        instrument_views_tested = True
                        for names in views_from(offered):
                            train_view = view_rows([*first, *second], offered, names)
                            check_view = view_rows(checked, offered, names)
                            local = SearchBudget(min(90000, budget.remaining // 2))
                            revised = self._research_view(self._spec(laboratory, names),
                                train_view, check_view, local, actions, transfer, graph,
                                transitions=transitions_for(names, train_view))
                            budget.spend(local.spent)
                            best = self._best(revised)
                            if best is not None:
                                selected, selected_channels, finalists = best, names, revised
                                selected_training, selected_validation = train_view, check_view
                                program = best["candidate"].program
                                if best["validation_loss"] <= self.config.minimum_improvement:
                                    break
                        if selected_channels:
                            research_selected_structure()
                            program = selected["candidate"].program
                            continue
                answered = []
                counterexample = (noisy_support or {}).get("structural_counterexample")
                if counterexample is not None and self.config.max_question_depth > 0:
                    question_id = f"noise_residual_{len(graph.nodes)}"
                    node = graph.add_residual_question(question_id, program, selected_training)
                    node.expression = Program.from_dict(counterexample["program"])
                    answers = graph.execute_ready(selected_training, budget, max_nodes=2,
                                                   risk_quantile=laboratory.spec.risk_quantile)
                    answered = [{"candidate": candidate, "kind": "noise_residual_answer",
                        "family_id": "T09", "training_loss": graph.nodes[f"{question_id}:combined"].loss}
                        for candidate in answers if candidate.candidate_id == f"answer:{question_id}:combined"]
                    actions.append({"action": "revise_from_residual_answer", "question_id": question_id,
                                    "answered": bool(answered), "gain": counterexample["gain"]})
                finalists = self._research_view(self._spec(laboratory, selected_channels),
                    selected_training, selected_validation, budget, actions, transfer, graph,
                    parent=selected, transitions=transitions_for(selected_channels))
                finalists.extend(self._assess(answered, selected_validation, budget, fallback,
                                              laboratory.spec.risk_quantile))
                revised = self._best(finalists)
                if revised is not None:
                    selected = revised
                    research_selected_structure()
                    research_selected_program_conditions()
                    program = selected["candidate"].program

        used_sources = _sources(program) if confirmed else []
        if selected.get("research_program") is not None:
            selected["research_program"]["status"] = ("supported_on_fresh_trajectory"
                                                       if confirmed else "unresolved")
            selected["research_program"]["confirmation_loss"] = confirmation_loss
            selected["research_program"]["confirmation_rows"] = confirmation_rows
        for event in transfer:
            event["accepted"] = bool(confirmed and set(event["source_ids"]) & set(used_sources)
                                     and event["status"] == "supported")
        concepts_added = []
        if self.config.use_memory and self.config.strategy != "fixed":
            for event in transfer:
                if event["status"] != "counterexample":
                    continue
                for prior in self.library.meta_priors:
                    if prior["source_concept_id"] in event["source_ids"]:
                        prior.setdefault("counterexamples", []).append({
                            "task_id": laboratory.spec.task_id,
                            "loss": event.get("confirmation_loss", event["validation_loss"]),
                            "kind": "failed_role_binding",
                            "program": event["program"], "replacement_concept_id": None})
                        prior["status"] = "conditional"
            if confirmed:
                spec = self._knowledge_spec(self._spec(laboratory, selected_channels))
                concept_id, added = self.library.learn(selected["candidate"], spec,
                    [*selected_training, *selected_validation])
                concepts_added = [concept_id] if added else []
                for step in (selected.get("research_program") or {}).get("selection_steps", []):
                    if step["operation"] == "resolve_relation_intervals":
                        for parameter in step["parameters"]:
                            if parameter.get("source_concept_id") is None:
                                parameter["source_concept_id"] = concept_id
                for source_id in used_sources:
                    self.library.record_use(source_id, spec.task_id)
                for prior in self.library.meta_priors:
                    if prior["source_concept_id"] in used_sources:
                        prior["uses"] += 1
                    if prior["source_concept_id"] in selected.get("method_source_ids", []):
                        prior["method_uses"] = prior.get("method_uses", 0) + 1
                    for event in prior["counterexamples"]:
                        if event["task_id"] == spec.task_id and event["replacement_concept_id"] is None:
                            event["replacement_concept_id"] = concept_id
                existing_prior = next((prior for prior in self.library.meta_priors
                    if prior["source_concept_id"] == concept_id), None)
                if existing_prior is None:
                    self.library.meta_priors.append({
                        "prior_id": f"meta_prior_{len(self.library.meta_priors) + 1:04d}",
                        "source_concept_id": concept_id,
                        "family_id": selected["family_id"], "kind": selected["kind"],
                        "claim": "The selected measured coordinates and history suffice on a new trajectory",
                        "status": "supported", "task_id": spec.task_id,
                        "source_features": list(spec.feature_names),
                        "control_features": list(spec.feature_bounds),
                        "object_schema": _object_schema(spec),
                        "structure": selected.get("structure"),
                        "object_equivalence": selected.get("object_equivalence"),
                        "object_competition": selected.get("object_competition"),
                        "research_program": selected.get("research_program"),
                        "support_rows": len(selected_training) + len(selected_validation),
                        "validation_loss": confirmation_loss,
                        "probe_ids": [batch["probe"]["probe_id"] for batch in batches
                                      if set(selected_channels) <= set(batch["channels"])],
                        "counterexamples": [], "uses": 0, "parent_concept_ids": list(dict.fromkeys(
                            [*used_sources, *selected.get("method_source_ids", [])])),
                        "research_method": {"fit_rows": len(selected_training),
                            "validation_rows": len(selected_validation),
                            "confirmation_rows": confirmation_rows,
                            "noise_supported": bool(noisy_support and noisy_support["adequate"]),
                            "risk_quantile": spec.risk_quantile},
                        "counter_prediction": {"type": "future_trajectory",
                            "condition": "The rebound computation predicts a new measured trajectory"}})
                else:
                    self._remember_relation_research(existing_prior, selected,
                        self._spec(laboratory, offered), actions)
        measurement_program, measurement_channels, instrument_model_id = None, None, None
        if confirmed and selected_channels and self._learn_instruments:
            from .instrument_models import rebind_instrument_model
            command_spec = self._spec(laboratory, ())
            measured_spec = self._spec(laboratory, selected_channels)
            model = self._fit_instruments(command_spec, measured_spec, paired_for(selected_channels), budget, actions)
            if model is not None:
                binding = rebind_instrument_model(model, measured_spec)
                command_program = bind_measurement_arguments(program, binding)
                required = {name for name, _ in command_program.observed_variables()}
                if required <= set(command_spec.feature_names):
                    command_rows = acquire((), "confirm_instrument_replacement")
                    if command_rows is not None and budget.spend(len(command_rows)):
                        command_loss = loss(command_program, command_rows, fallback, command_spec.risk_quantile)
                        accepted = command_loss <= self.config.minimum_improvement
                        counterexample_loss = None
                        if directed_command_rows:
                            if budget.spend(len(directed_command_rows)):
                                counterexample_loss = loss(command_program, directed_command_rows,
                                    fallback, command_spec.risk_quantile)
                            accepted = accepted and counterexample_loss is not None and (
                                counterexample_loss <= self.config.minimum_improvement)
                        actions.append({"action": "compare_instrument_replacement",
                            "model_id": model["model_id"], "loss": command_loss,
                            "accepted": accepted, "observation_rows": len(command_rows),
                            "paid_counterexample_loss": counterexample_loss,
                            "replaced_channels": list(selected_channels),
                            "measurement_units_saved_per_row": sum(rates[name] for name in selected_channels)})
                        if accepted:
                            measurement_program, measurement_channels = program.to_dict(), list(selected_channels)
                            program, selected_channels = command_program, ()
                            instrument_model_id = model["model_id"]
                            confirmation_loss = command_loss
        if confirmed and self._revise_inputs:
            prior = next((item for item in self.library.meta_priors if item["source_concept_id"] == concept_id), None)
            if prior is not None and not prior.get("object_schema"):
                self._remember_inputs(prior, selected["candidate"].program, self._spec(laboratory, offered),
                    self._spec(laboratory, ()), [*selected_training, *selected_validation], budget, actions)
        if object_research and object_research.get("groups"):
            competitions = list(self._object_competitions.values())
            if competitions:
                object_research["program_competitions"] = deepcopy(competitions)
                saved = next((item for item in self.library.object_models
                              if item["model_id"] == object_research["model_id"]), None)
                if saved is not None:
                    saved["program_competitions"] = deepcopy(competitions)
            equivalence = selected.get("object_equivalence") if confirmed else None
            if equivalence is not None:
                object_research["object_equivalence"] = deepcopy(equivalence)
                saved = next((item for item in self.library.object_models
                              if item["model_id"] == object_research["model_id"]), None)
                if saved is not None:
                    saved["object_equivalence"] = deepcopy(equivalence)
                actions.append({"action": "adopt_equivalent_object_program",
                    "object_model_id": object_research["model_id"],
                    "alternatives": len(equivalence["schemas"]),
                    "unresolved": list(object_research.get("unresolved", [])),
                    "confirmation_loss": confirmation_loss})
            schema = object_research.get("schema")
            learned = next((item for item in reversed(self.library.instrument_models)
                if item.get("status") != "counterexample"
                and ((schema and item.get("source_schema")
                      and item["source_schema"]["fields"] == schema["fields"]
                      and item["source_schema"]["controls"] == schema["controls"])
                     or (not schema and item.get("source_object_model", {}).get("model_id") == object_research["model_id"]))), None)
            if learned is not None:
                object_research["sensor_model"] = deepcopy(learned)
                saved = next((item for item in self.library.object_models
                              if item["model_id"] == object_research["model_id"]), None)
                if saved is not None:
                    saved["sensor_model"] = deepcopy(learned)
        costs = {"observation_cost": spent_observations,
                 "measurement_cost": spent_measurements, "compute_cost": budget.spent,
                 "queries": len(batches)}
        return {"task_id": laboratory.spec.task_id,
                "status": "completed" if confirmed else "unresolved",
                "program": program.to_dict(), "expression": program.describe(),
                "fallback": fallback, "selected_kind": selected["kind"],
                "selected_structure": selected.get("structure"),
                "object_equivalence": selected.get("object_equivalence"),
                "object_competition": selected.get("object_competition"),
                "program_competitions": list(self._object_competitions.values()),
                "research_program": selected.get("research_program"),
                "selected_channels": list(selected_channels),
                "measurement_program": measurement_program, "measurement_channels": measurement_channels,
                "instrument_model_id": instrument_model_id,
                "learned_object_schema": deepcopy(self._object_schemas.get(tuple(offered))),
                "object_model_id": object_research.get("model_id") if object_research else None,
                "object_research": deepcopy(object_research),
                "validation_loss": (None if selected.get("selection_basis") == "all_acquired_evidence"
                                    else selected["validation_loss"]),
                "selection_loss": selected["validation_loss"],
                "selection_basis": selected.get("selection_basis", "later_trajectory"),
                "confirmation_loss": confirmation_loss, "costs": costs,
                "loss_kind": "mse" if laboratory.spec.risk_quantile == .5 else "pinball",
                "risk_quantile": laboratory.spec.risk_quantile,
                "noise_assessment": noisy_support,
                "actions": actions, "transfer": transfer, "used_sources": used_sources,
                "method_source_ids": selected.get("method_source_ids", []) if confirmed else [],
                "concepts_added": concepts_added, "research_graph": graph.to_dict(),
                "batches": batches}
