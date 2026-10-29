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
from math import ceil
from random import Random

from .action_policy import ActionOption, ResearchActionPolicy
from .continuous_state import (_linear_fit, _sum_terms, continuous_state_candidates,
                               state_identification_cost)
from .evaluation import initial_prediction, loss, predict
from .language import Definition, Program
from .knowledge_returns import return_priority, transfer_condition
from .memory import KnowledgeLibrary, _object_schema
from .meta_prior_claims import compile_claims
from .meta_prior_research import _sources
from .models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from .research_graph import ResearchGraph
from .risk_search import trajectory_blocks
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


def _ambiguous_instruments(model):
    return [name for name, field in model.get("fields", {}).items()
            if sum(item.get("status") not in {"refuted", "counterexample"}
                   for item in field.get("hypotheses", [])) > 1]


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


def _conditional_experiment_entry_quote(spec, fitting_rows, validation_rows):
    # The existing pack covers every public control and shares one reference.
    # Its caller also retains twelve rows for the measured boundary step.
    rows = max(48, (len(spec.feature_names) + 1) * min(12, spec.max_probe_length) + 12)
    return {"observation_rows": rows,
            "compute_cost": (fitting_rows + rows) * 44 + validation_rows * 8}


def compare_resumed_programs(incumbent, current, *, before_loss, after_loss,
        incumbent_fallback, current_fallback, directed_command_rows,
        inherited_relation_evidence, feature_names, budget, minimum_improvement,
        risk_quantile=.5, external_experiment=False, incumbent_refuted=False):
    """Decide retention using paid experiments as well as random confirmation.

    Confirmation scores were already paid by the caller. This consumes only
    the recorded experiment rows; it neither fits nor acquires observations.
    Returned scores belong to these two exact programs and this comparison.
    """
    evidence = {}
    for name, program in (("incumbent", incumbent), ("current", current)):
        required = {key for key, _ in program.observed_variables()} - {"__target__"}
        evidence[name] = {
            "program_experiment": list(directed_command_rows)
                if required <= set(feature_names) else [],
            "inherited_relation": [row for row in inherited_relation_evidence
                if required <= set(row.features)],
        }
    cost = sum(len(rows) for checks in evidence.values() for rows in checks.values())
    result = {"status": "compute_unresolved", "compute_required": cost,
              "compute_cost": 0, "checks": {}, "retain_incumbent": None,
              "incumbent_refuted": bool(incumbent_refuted)}
    if not budget.spend(cost):
        return result
    result.update(status="compared", compute_cost=cost)
    for name, program, fallback in (("incumbent", incumbent, incumbent_fallback),
                                    ("current", current, current_fallback)):
        result["checks"][name] = {kind: {
            "observation_rows": len(rows),
            "loss": loss(program, rows, fallback, risk_quantile),
        } for kind, rows in evidence[name].items() if rows}
    result["incumbent_refuted"] |= any(
        check["loss"] > minimum_improvement
        and evidence["incumbent"][kind] == evidence["current"][kind]
        and result["checks"]["current"][kind]["loss"] <= minimum_improvement
        for kind, check in result["checks"]["incumbent"].items())
    # Keep the established external-experiment tie rule. Internal experiments
    # additionally disqualify an incumbent when the current program resolves
    # its measured counterexample on the same rows. If both fail, this is not
    # evidence for replacing the incumbent; the later counterexample check
    # still prevents either program from borrowing a random probe's success.
    retain = (before_loss < after_loss - minimum_improvement if external_experiment
              else before_loss <= after_loss + minimum_improvement)
    result["retain_incumbent"] = bool(retain and not result["incumbent_refuted"])
    return result


class UnifiedResearchAgent:
    """A bounded common research loop, including paid joint revision and reuse."""

    def __init__(self, config=None, *, measurement_budget=768, library=None, downstream_demands=None):
        self.config = config or ResearchConfig(observation_budget=192,
            compute_budget=120000, minimum_improvement=1e-7, active_probes=False)
        self.measurement_budget = measurement_budget
        self.library = library if library is not None else KnowledgeLibrary()
        self.downstream_demands = deepcopy(downstream_demands)
        self._learn_instruments = False
        self._revise_inputs = False
        self._object_schemas = {}
        self._object_models = {}
        self._object_competitions = {}
        self._object_candidate_frontiers = {}
        self._workflow_episode = None
        self._callable_fit_serial = 0

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

    def _object_probe(self, spec, model, records, serial, *, methods=None, remaining_rows=None):
        from .object_interventions import propose_object_probe
        return propose_object_probe(spec, model, records, serial, methods=methods, remaining_rows=remaining_rows)

    def _dependency_methods(self, sources, *, budget, actions):
        from .object_dependency_methods import ObjectDependencyMethods
        return ObjectDependencyMethods(sources, budget=budget, actions=actions)

    def _research_objects(self, laboratory, channels, acquire, batches, budget, actions,
                          *, previous_model=None, remaining_rows=None, source_object_model_ids=()):
        """Discover a usable grouping from paid anonymous state transitions."""
        if previous_model is not None:
            from .object_induction_continuation import resume_object_induction
            return resume_object_induction(self, laboratory, channels, acquire, batches, budget, actions,
                previous_model=previous_model, remaining_rows=remaining_rows,
                source_object_model_ids=source_object_model_ids)
        from .object_induction import infer_object_model
        from .object_dependency_methods import ObjectDependencyMethods
        from .instrument_models import rebind_instrument_model
        from .models import Probe
        raw_spec = laboratory.spec_for(channels)
        command_spec = laboratory.spec_for(())
        start = budget.spent
        initial_batch = len(batches)
        # The purchased object bank also fits the response. Reserve a fresh
        # response comparison and confirmation, plus the existing role check,
        # rather than spending only a fixed fraction on its prerequisite.
        response_length = min(24, raw_spec.max_probe_length,
                              max(1, self.config.observation_budget // 8))
        available_rows = self.config.observation_budget if remaining_rows is None else remaining_rows
        row_limit = min(192, max(0, available_rows - 2 * response_length - 12))
        methods = self._dependency_methods([item for item in self.library.object_models
            if item.get("status") != "counterexample"], budget=budget, actions=actions)

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
                    membership_probes = {item["probe_id"] for item in previous.get("dependency_questions", [])
                        if item.get("question", {}).get("purpose") == "distinguish_membership_dependency"}
                    for probe in previous.get("dependency_probes", []):
                        if probe["probe_id"] in membership_probes:
                            continue
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
            probe, question = self._object_probe(raw_spec, model, evidence, serial, methods=methods,
                remaining_rows=row_limit - len(evidence))
            if probe is None or len(evidence) + probe.cost > row_limit:
                break
            if acquire(channels, "distinguish_object_dependencies", probe=probe) is None:
                break
            before = deepcopy(model.get("unresolved", []))
            interpret = question.get("interpret_dependencies", True)
            if interpret:
                model = infer_object_model(raw_spec, records(), budget)
            if question.get("research_program"):
                methods.observe(records())
            actions.append({"action": "test_object_dependency_question", "question": question,
                "probe_id": probe.probe_id, "observation_rows": probe.cost,
                "inference_deferred_until_paired_pilot": not interpret,
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
        model["dependency_questions"] = [deepcopy(action) for action in actions
            if action.get("action") == "test_object_dependency_question"
            and action.get("probe_id") in {item["probe_id"] for item in model["dependency_probes"]}]
        model["dependency_methods"] = methods.snapshot()
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
        bound_inputs = prior.get("input_program") or prior.get("input_research_program")
        if bound_inputs or _object_schema(measured_spec):
            from .research_program_inputs import abstract_research_inputs
            inputs = abstract_research_inputs(procedure, measured_spec, self.library)
            if inputs is None and bound_inputs:
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

    def remember_confirmed_selection(self, selected, spec, selected_training, selected_validation,
                                     *, confirmed, confirmation_loss, confirmation_rows,
                                     selected_channels=(), batches=(), actions=None,
                                     used_sources=None, noisy_support=None, offered_spec=None,
                                     confirmation_ref=None, paid_support_matches=None):
        """Save a confirmed native answer through the ordinary knowledge path.

        Paid-source callers supply their retained confirmation and per-row support
        mask; this method does not repeat research or confirmation. A retained
        confirmation reference is consumed once, preserving prior failures.
        """
        result = {"stored": False, "concept_id": None, "concepts_added": [],
                  "new_fit_compute_cost": 0, "new_confirmation": False}
        if not confirmed:
            return {**result, "reason": "confirmation_not_passed"}
        if not self.config.use_memory or self.config.strategy == "fixed":
            return {**result, "reason": "memory_disabled"}
        if (confirmation_loss is None or confirmation_rows <= 0
                or (confirmation_loss > self.config.minimum_improvement
                    and not (noisy_support or {}).get("adequate"))):
            return {**result, "reason": "confirmation_not_supported"}
        if paid_support_matches is not None and len(paid_support_matches) != (
                len(selected_training) + len(selected_validation)):
            raise ValueError("Paid support mask must cover the retained training and validation rows")
        actions = [] if actions is None else actions
        if confirmation_ref is not None:
            equivalent = selected["candidate"].program.expanded().simplified()
            concepts = {item.concept_id: item for item in self.library.concepts}
            prior = next((item for item in self.library.meta_priors
                if confirmation_ref in item.get("confirmation_refs", [])
                and concepts[item["source_concept_id"]].program.expanded().simplified() == equivalent), None)
            if prior is not None:
                return {**result, "stored": True, "concept_id": prior["source_concept_id"],
                        "reason": "paid_confirmation_already_saved", "reused": True}
        used_sources = (list(dict.fromkeys([*_sources(selected["candidate"].program),
                                          *selected.get("method_source_ids", [])]))
                        if used_sources is None else used_sources)
        if selected.get("research_program") is not None:
            selected["research_program"].update(status="supported_on_fresh_trajectory",
                confirmation_loss=confirmation_loss, confirmation_rows=confirmation_rows)
            if confirmation_ref is not None:
                selected["research_program"]["confirmation_ref"] = deepcopy(confirmation_ref)
        if selected.get("mechanism_revision"):
            selected["mechanism_revision"] = {**selected["mechanism_revision"],
                "status": "supported_on_fresh_trajectory", "confirmation_loss": confirmation_loss,
                "confirmation_rows": confirmation_rows}
        concept_id, added = self.library.learn(selected["candidate"], spec,
            [*selected_training, *selected_validation], paid_support_matches=paid_support_matches)
        concepts_added = [concept_id] if added else []
        if selected.get("research_draft_id") is not None:
            draft = next(item for item in self.library.research_drafts
                         if item["draft_id"] == selected["research_draft_id"])
            if spec.task_id == draft["source_task_id"]:
                draft["resolved_concept_id"] = concept_id
                actions.append({"action": "resolve_research_draft", "draft_id": draft["draft_id"],
                    "concept_id": concept_id, "confirmation_loss": confirmation_loss,
                    "confirmation_rows": confirmation_rows})
            else:
                draft["outcomes"].append({"task_id": spec.task_id, "concept_id": concept_id,
                    "status": "supported_on_fresh_trajectory", "confirmation_loss": confirmation_loss,
                    "confirmation_rows": confirmation_rows})
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
                "mechanism_revision": deepcopy(selected.get("mechanism_revision")),
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
                offered_spec or spec, actions)
            if selected.get("research_program") and not existing_prior.get("research_program"):
                existing_prior["research_program"] = deepcopy(selected["research_program"])
            if selected.get("mechanism_revision"):
                existing_prior["mechanism_revision"] = deepcopy(selected["mechanism_revision"])
        prior = next(item for item in self.library.meta_priors
                     if item["source_concept_id"] == concept_id)
        if existing_prior is None:
            self._remember_relation_research(prior, selected, offered_spec or spec, actions)
        if confirmation_ref is not None:
            prior.setdefault("confirmation_refs", []).append(deepcopy(confirmation_ref))
        return {**result, "stored": True, "concept_id": concept_id,
                "concepts_added": concepts_added, "reused": False}

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

    def _relation_input_bindings(self, command_spec):
        """Project learned measured relation leaves onto current commands.

        Relation priors keep their original measured units.  This helper only
        changes their input leaves when a currently executable, unique
        instrument model can map the prior's schema onto ``command_spec``;
        unresolved fields simply remain absent and the relation composer will
        decline that kernel.  No relation source record or radius is changed.
        """
        if not self._learn_instruments or not self.config.use_memory:
            return {}
        bindings = {}
        for prior in reversed(self.library.meta_priors):
            if prior.get("status") != "supported":
                continue
            source = self._input_prior(prior)
            schema = source.get("object_schema")
            features = tuple(source.get("source_features", ()))
            if schema:
                features = tuple(name for group in schema["objects"]
                                 for name in schema["fields"][group].values())
            if not features:
                continue
            # Bind source roles to the current object representation first.
            # Asking an instrument model to reconstruct the source schema also
            # reconstructs its old command names, which may no longer exist.
            current_schema = _object_schema(command_spec)
            schemas = [current_schema] if current_schema else list(self._object_schemas.values())
            current_schema = next((item for item in schemas if schema
                and len(item["objects"]) == len(schema["objects"])), None)
            if current_schema is not None:
                current_features = tuple(name for group in current_schema["objects"]
                    for name in current_schema["fields"][group].values())
                current_spec = replace(command_spec, feature_names=current_features,
                    context={**command_spec.context, "object_schema": json.dumps(current_schema)})
                instruments = self._instrument_binding(current_spec, command_spec)
                available = set(command_spec.feature_names)
                for old_group, current_group in zip(schema["objects"], current_schema["objects"]):
                    for role, old_name in schema["fields"][old_group].items():
                        current_name = current_schema["fields"][current_group].get(role)
                        value = (Program("feature", name=current_name) if current_name in available
                                 else instruments.get(current_name))
                        if value is not None:
                            bindings.setdefault(old_name, value)
                continue
            context = dict(command_spec.context)
            if schema:
                context["object_schema"] = json.dumps(schema)
            measured_spec = replace(command_spec, feature_names=features, context=context)
            mapped = self._instrument_binding(measured_spec, command_spec)
            for name, program in mapped.items():
                bindings.setdefault(name, program)
        return bindings

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

    def _record_instrument_assessment(self, model, assessment, measured_spec, rows, actions):
        """A measured failure withdraws that field, leaving other laws available."""
        from .instrument_hypotheses import instrument_model_status
        model["status"] = instrument_model_status(model)
        failed = {name: assessment["field_losses"][name] for name in assessment["refuted_fields"]
                  if name in assessment["field_losses"]}
        if failed:
            event = {"task_id": measured_spec.task_id, "field_losses": failed, "rows": rows}
            model.setdefault("counterexamples", []).append(event)
            actions.append({"action": "instrument_mapping_counterexample",
                "model_id": model["model_id"], **event,
                "retained_fields": [name for name in model["fields"]
                                    if name not in assessment["refuted_fields"]]})

    def _distinguish_instruments(self, laboratory, model, channels, acquire, paired_for,
                                budget, actions, *, max_length=None, observation_allowance=None,
                                required_fields=None):
        from .instrument_hypotheses import (
            assess_instrument_hypotheses, plan_instrument_disagreement)
        from .instrument_experiment_methods import observe_instrument_method
        command_spec = self._spec(laboratory, ())
        measured_spec = self._spec(laboratory, channels)
        methods = self.library.program_experiment_methods if self.config.use_memory else None
        while model is not None and _ambiguous_instruments(model):
            if observation_allowance is not None and observation_allowance <= 0:
                break
            local = SearchBudget(min(6000, budget.remaining // 2))
            allowed_length = (min(max_length or laboratory.spec.max_probe_length, observation_allowance)
                              if observation_allowance is not None else max_length)
            planning_model = model
            if required_fields is not None:
                from .instrument_models import rebind_instrument_fields
                needed = set(required_fields)
                projected = rebind_instrument_fields(model, measured_spec)
                planning_model = {**model, "source_schema": _object_schema(measured_spec),
                    "fields": {name: field for name, field in projected.items() if name in needed}}
            plan = plan_instrument_disagreement(planning_model, command_spec, measured_spec, local,
                max_length=allowed_length, serial=len(actions),
                tolerance=self.config.minimum_improvement, methods=methods)
            budget.spend(local.spent)
            actions.append({"action": "plan_instrument_disagreement", "model_id": model["model_id"],
                **{key: value for key, value in plan.items() if key != "probe"},
                "probe_id": plan["probe"].probe_id if plan["probe"] else None})
            if plan["probe"] is None:
                break
            measured = acquire(channels, "distinguish_instrument_hypotheses", probe=plan["probe"])
            if measured is None:
                break
            if observation_allowance is not None:
                observation_allowance -= len(measured)
            assessment = assess_instrument_hypotheses(model, command_spec, measured_spec,
                paired_for(channels)[-1], budget, tolerance=self.config.minimum_improvement)
            actions.append({"action": "test_instrument_hypotheses", "model_id": model["model_id"],
                "probe_id": plan["probe"].probe_id, "predictions": plan["predictions"],
                "assessment": assessment})
            if methods is not None:
                feedback = observe_instrument_method(methods, plan, assessment, model, command_spec,
                    task_id=f"{measured_spec.task_id}:{self.config.seed}", observation_rows=len(measured),
                    measurement_cost=len(measured) * sum(channel.cost_per_row for channel in laboratory.channels
                                                        if channel.name in channels),
                    tolerance=self.config.minimum_improvement)
                if feedback is not None:
                    actions.append(feedback)
            self._record_instrument_assessment(model, assessment, measured_spec, len(measured), actions)
            if any(name in assessment["field_losses"] for name in assessment["refuted_fields"]):
                model = self._fit_instruments(command_spec, measured_spec,
                    paired_for(channels), budget, actions, required_fields=required_fields,
                    previous_model=model)
            if assessment["unassessed_fields"]:
                break
        return model

    def _fit_instruments(self, command_spec, measured_spec, paired_batches, budget, actions, *,
                         required_fields=None, previous_model=None):
        """Fit purchased readings, then compare on a later purchased trajectory."""
        from .instrument_models import fit_instrument_model, rebind_instrument_fields
        from .instrument_hypotheses import assess_instrument_hypotheses
        if len(paired_batches) < 2:
            return None
        training = [pair for batch in paired_batches[:-1] for pair in batch]
        validation = paired_batches[-1]

        def compare(model, observations):
            assessment = assess_instrument_hypotheses(model, command_spec, measured_spec,
                observations, budget, tolerance=self.config.minimum_improvement)
            errors = {name: min(values.values()) for name, values in assessment["field_losses"].items()
                      if values}
            return assessment, errors

        # Retain independently supported fields, including partial models.
        # Coverage is measured after rebinding into this apparatus's coordinates.
        requested = [name for name in measured_spec.feature_names
                     if required_fields is None or name in required_fields]
        retained = {}
        old_models = list(self.library.instrument_models) if self.config.use_memory else []
        if previous_model is not None and all(old is not previous_model for old in old_models):
            old_models.append(previous_model)
        for old in reversed(old_models):
            if old.get("status") == "counterexample":
                continue
            mapped = rebind_instrument_fields(old, measured_spec)
            if not set(mapped) - set(retained):
                continue
            aggregates = {name: field for name, field in mapped.items()
                          if field.get("kind") == "aggregate_instrument_fields"
                          and old.get("source_schema") != _object_schema(measured_spec)}
            if aggregates:
                # A coarse reading tests the whole aggregation, not an
                # individual fine sensor whose reading was never purchased.
                aggregate = {**old, "source_schema": _object_schema(measured_spec), "fields": aggregates}
                aggregate_assessment, _ = compare(aggregate, [*training, *validation])
                for name in aggregate_assessment["refuted_fields"]:
                    projection = {"field": name, "program": aggregates[name]["program"]}
                    old.setdefault("refuted_projections", []).append(projection)
                    actions.append({"action": "instrument_aggregation_counterexample",
                        "model_id": old["model_id"], "field": name,
                        "field_losses": aggregate_assessment["field_losses"].get(name, {}),
                        "rows": len(training) + len(validation)})
                for name in aggregate_assessment["supported_fields"]:
                    retained.setdefault(name, aggregate["fields"][name])
            # Consume all paid counterexamples; a later easy trajectory cannot
            # rehabilitate an earlier refuted field.
            assessment, errors = compare(old, [*training, *validation])
            self._record_instrument_assessment(old, assessment, measured_spec,
                len(training) + len(validation), actions)
            eligible = (set(assessment["supported_fields"]) | set(assessment["ambiguous_fields"]))
            eligible -= set(assessment["unassessed_fields"])
            projected = rebind_instrument_fields({**old, "fields": {
                name: field for name, field in old["fields"].items() if name in eligible}}, measured_spec)
            for name, field in projected.items():
                retained.setdefault(name, field)
            if set(requested) <= set(projected):
                actions.append({"action": "reuse_instrument_mapping", "model_id": old["model_id"],
                    "field_losses": errors, "rows": len(training) + len(validation),
                    "retained_fields": list(projected), "ambiguous_fields": assessment["ambiguous_fields"]})
                return old
        missing = [name for name in requested if name not in retained]
        before = budget.spent
        local = SearchBudget(min(24000, budget.remaining // 2))
        model = fit_instrument_model(command_spec, measured_spec, training, local,
                                     max_history=self.config.max_history, field_names=missing)
        budget.spend(local.spent)
        assessment, errors = compare(model, validation)
        # Only fields supported by an unseen measured trajectory become
        # executable substitutes. Partial answers can remain useful knowledge.
        model["fields"] = {name: field for name, field in model["fields"].items()
                           if name not in assessment["unassessed_fields"] and (
                               name in assessment["supported_fields"] or name in assessment["ambiguous_fields"])}
        fitted = list(model["fields"])
        model["fields"] = {**retained, **model["fields"]}
        model.update(model_id=f"instrument_{len(self.library.instrument_models) + 1:04d}",
                     status=("ambiguous_on_measured_trajectory" if _ambiguous_instruments(model)
                             else "supported_on_measured_trajectory"), validation_losses=errors,
                     validation_rows=len(validation), counterexamples=[])
        actions.append({"action": "research_instrument_mapping", "model_id": model["model_id"],
            "fields": list(model["fields"]), "field_losses": errors,
            "fitted_fields": fitted, "retained_fields": list(retained), "requested_fields": requested,
            "hypothesis_assessment": assessment,
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
            predictable = [item for item in candidates
                        if {name for name, _ in item["candidate"].program.observed_variables()}
                        <= controllable][:4]
            programs = [item["candidate"].program for item in predictable]
            pool = [random_points]
            pool.append(tuple({name: limits[(step + axis) % 2]
                               for axis, (name, limits) in enumerate(bounds)}
                              for step in range(length)))
            pool.append(tuple({name: limits[(step // 3 + axis) % 2]
                               for axis, (name, limits) in enumerate(bounds)}
                              for step in range(length)))
            conditional = any(any(step["operation"] in ("discover_conditional_coordinates", "discover_composed_conditions")
                for step in item.get("research_program", {}).get("selection_steps", []))
                for item in predictable)
            if conditional and len(programs) >= 2:
                from .temporal_control_probes import propose_temporal_control_probes
                local = SearchBudget(min(6000, budget.remaining // 4))
                proposals = propose_temporal_control_probes(spec, programs, local.spend,
                    serial=serial, length=length)
                budget.spend(local.spent)
                # Acquisition is already quoted at this length. Longer prefixes
                # need a separately funded experiment, not a hidden extra cost.
                usable = [plan for plan in proposals if len(plan["probe"].sequence) <= length]
                for plan in usable:
                    points = plan["probe"].sequence
                    pool.append((*points, *(dict(points[-1]) for _ in range(length - len(points)))))
                actions.append({"action": "propose_conditional_experiments",
                    "compute_cost": local.spent, "alternatives": len(usable),
                    "boundary_intervals": [step.get("threshold_intervals", step.get("threshold_interval")) for item in predictable
                        for step in item.get("research_program", {}).get("selection_steps", [])
                        if step["operation"] in ("discover_conditional_coordinates", "discover_composed_conditions")]})
            if len(programs) >= 2 and budget.spend(len(pool) * length * len(programs)):
                scores = []
                for points in pool:
                    history, disagreement = [], 0.
                    for point in points:
                        row = Observation(point, 0., tuple(history))
                        values = [predict(item["candidate"].program, row, item.get("fallback", 0.))
                                  for item in predictable]
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

    def _recall(self, spec, observations=None, *, origins=None):
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
                    or spec.task_id in source.excluded_tasks
                    or source.risk_quantile != spec.risk_quantile):
                continue
            scope = self.library._scope_for(source, spec)
            allowed = (binding["template_id"].startswith(
                f"{source.concept_id}:condition:{scope.task_id}:supported:")
                if scope and scope.predicate else
                binding["template_id"] == f"{source.concept_id}:abstract:0")
            if not allowed:
                continue
            prior = next((item for item in reversed(self.library.meta_priors)
                          if item["source_concept_id"] == source.concept_id), None)
            payload = (prior.get("input_program") if self._revise_inputs and prior
                       and not (scope and scope.predicate) else None)
            if payload is not None:
                from .abstractions import AbstractTemplate
                template = AbstractTemplate.from_dict(payload["template"])
                definition = Definition.from_dict(payload["definition"])
                binding = {"arguments": tuple(Program.from_dict(argument) for argument in payload["arguments"])}
            records.append((source, template, definition, binding))
        # Give complete learned mechanisms a turn before multiple role bindings
        # of the same mechanism. Successful bindings are already remembered by
        # AbstractTemplate.mappings; names are not used as mechanism labels.
        condition = transfer_condition(spec, observations) if observations else None
        records.sort(key=lambda item: (
            return_priority(item[0].reuse_history, condition) if condition is not None else (0., 0.),
            -len(item[0].successful_tasks), -item[0].generation, item[0].program.size))

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

        queues = [(source, template, definition, binding, iter(role_bindings(source, template)))
                  for source, template, definition, binding in records]
        calls = []
        while queues and len(calls) < self.config.max_candidates:
            next_queues = []
            for source, template, definition, binding, mappings in queues:
                mapping = next(mappings, None)
                if mapping is None:
                    continue
                next_queues.append((source, template, definition, binding, mappings))
                call = Program("call", name=definition.definition_id, definition=definition,
                    children=tuple(bind_measurement_arguments(argument, mapping)
                                   for argument in binding["arguments"]))
                if self.library._calls_available(call, spec) and call not in calls:
                    calls.append(call)
                    if origins is not None:
                        coordinates = {name: f"coordinate:{index}"
                                       for index, name in enumerate(spec.feature_names)}
                        origins[call] = {"source_id": definition.source_concept_id,
                            "template_id": template.template_id,
                            "mapping": {role: value.to_dict() if isinstance(value, Program) else value
                                        for role, value in mapping.items()},
                            "mapping_roles": {role: json.dumps(
                                bind_measurement_arguments(value, coordinates).to_dict(), sort_keys=True)
                                if isinstance(value, Program) else spec.feature_names.index(value)
                                for role, value in mapping.items()},
                            "condition": deepcopy(condition), "adaptation_work": []}
                if len(calls) >= self.config.max_candidates:
                    break
            queues = next_queues
        return calls

    def _screen_risk_knowledge(self, spec, training, validation, budget, calls, actions, *, baseline=None):
        if spec.risk_quantile == .5 or not self.config.risk_learning or not calls:
            return calls, []
        from .knowledge_adoption import screen_risk_knowledge
        local = SearchBudget(min(16000, budget.remaining // 4))
        if baseline is None:
            baseline = _fit(_terms(spec.feature_names), training, local,
                            "knowledge_comparison", q=spec.risk_quantile)
        selected, claims, events = screen_risk_knowledge(spec, training, validation,
            baseline, calls, local, tolerance=self.config.minimum_improvement,
            priors=self.library.meta_priors if self.config.use_memory else ())
        budget.spend(local.spent)
        actions.append({"action": "screen_risk_knowledge", "compute_cost": local.spent,
            "available_calls": len(calls), "research_calls": len(selected),
            "comparisons": events})
        return selected, claims

    def _replay_program_queue(self, spec, calls, observations=None):
        """Prepare the procedures and bindings actually available for replay."""
        from .abstractions import AbstractTemplate
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
                queues.append((source, payload, template, mappings[:2]))
                queued_sources.add(source)
        if observations:
            condition = transfer_condition(spec, observations)
            sources = {source.concept_id: source for source in self.library.concepts}
            queues.sort(key=lambda item: return_priority(sources[item[0]].reuse_history, condition))
        return queues

    @staticmethod
    def _replay_knowledge(spec, plans):
        """Describe executable replay inputs, omitting old scores and task labels."""
        knowledge = []
        coordinates = {name: f"coordinate:{index}" for index, name in enumerate(spec.feature_names)}
        for _, payload, template, mappings in plans:
            graph = ResearchGraph.from_dict(payload["graph"])
            terms = [graph.nodes[key].answer for key in payload["term_nodes"]]
            saved = next((step for step in reversed(payload.get("selection_steps", []))
                          if step["operation"] == "resolve_relation_intervals"), None)
            for mapping in mappings:
                features = {original: bind_measurement_arguments(mapping[role], coordinates)
                            if isinstance(mapping[role], Program) else coordinates[mapping[role]]
                            for role, original in zip(template.roles, template.source_features)}
                description = {"terms": [bind_measurement_arguments(term, features).to_dict()
                    if term is not None else None for term in terms]}
                if saved:
                    description["relation"] = {key: deepcopy(saved[key]) for key in
                        ("parameters", "evidence_contexts") if key in saved}
                    for key in ("program", "parameter_template"):
                        if key in saved:
                            description["relation"][key] = bind_measurement_arguments(
                                Program.from_dict(saved[key]), features).to_dict()
                    description["relation"]["parameter_terms"] = [bind_measurement_arguments(
                        Program.from_dict(term), features).to_dict()
                        for term in saved.get("parameter_terms", [])]
                knowledge.append(description)
        return knowledge

    def _replay_program_claims(self, spec, training, budget, calls, *, prepared=None):
        """Refit recalled procedures without starting a different grammar search."""
        from .research_programs import replay_research_program
        plans = self._replay_program_queue(spec, calls) if prepared is None else prepared
        queues = [(source, payload, template, iter(mappings))
                  for source, payload, template, mappings in plans]
        claims = []
        # Give distinct learned procedures an initial evidence comparison before
        # another role binding of a recent procedure can consume the next slot.
        replay_count = 0
        while queues and replay_count < 4:
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
                    replay_count += 1
                if replay_count >= 4:
                    break
            queues = remaining
        return claims

    def _coordinate_claims(self, spec, training, budget, actions):
        from .research_programs import propose_coordinate_programs
        before = budget.spent
        claims = []
        if max((len(row.history) for row in training), default=0) >= 2:
            from .temporal_coordinates import propose_temporal_coordinates
            # A small joint response/window trial must survive the larger
            # current-coordinate grammar expansion. It also offers a simpler
            # explanation before any conditional branch is considered.
            local = SearchBudget(min(8000, budget.remaining // 4))
            claims = propose_temporal_coordinates(spec, training, local,
                basis=_terms(spec.feature_names, quadratic=True),
                max_candidates=self.config.max_candidates)
            budget.spend(local.spent)
        if not any(item["training_loss"] <= 1e-10 for item in claims):
            claims.extend(propose_coordinate_programs(spec, training, budget,
                max_candidates=self.config.max_candidates))
        claims = sorted(claims, key=lambda item: (
            0. if item["training_loss"] < 1e-12 else item["training_loss"],
            item["candidate"].program.size))[:self.config.max_candidates]
        actions.append({"action": "propose_coordinate_reconstruction",
            "compute_cost": budget.spent - before,
            "alternatives": [{"training_loss": item["training_loss"],
                "program": item["candidate"].program.to_dict()} for item in claims]})
        return claims

    def _conditional_claims(self, spec, training, budget, actions):
        from .conditional_coordinates import propose_conditional_coordinates
        before = budget.spent
        claims = propose_conditional_coordinates(spec, training, budget,
            basis=_terms(spec.feature_names, quadratic=True),
            max_candidates=self.config.max_candidates)
        actions.append({"action": "propose_conditional_reconstruction",
            "compute_cost": budget.spent - before,
            "alternatives": [{"training_loss": item["training_loss"],
                "program": item["candidate"].program.to_dict()} for item in claims]})
        return claims

    def _object_scope_claims(self, spec, training, budget, actions):
        """Test a remembered whole-apparatus computation at the current arity."""
        from .object_scope_revision import propose_object_scope_revisions

        if not self.config.use_memory:
            return []
        before = budget.spent
        claims = propose_object_scope_revisions(spec, training, budget,
            self.library.meta_priors, max_candidates=self.config.max_candidates)
        if claims or budget.spent != before:
            actions.append({"action": "revise_object_scope", "compute_cost": budget.spent - before,
                "alternatives": [{"source_ids": item["method_source_ids"],
                    "training_loss": item["training_loss"],
                    "scope": item["object_scope_revision"]} for item in claims]})
        return claims

    def _program_claims(self, spec, training, budget, calls, actions, *, relation_kernels=None,
                        replayed=None):
        """Execute remembered research procedures, then compose unexplained parts."""
        if self.config.max_question_depth <= 0:
            return []
        drafts = []
        if self.config.use_memory and self.library.research_drafts:
            from .research_drafts import replay_research_drafts
            drafts = replay_research_drafts(self, spec, training, budget, actions)
            if any(item["training_loss"] <= self.config.minimum_improvement for item in drafts):
                return drafts
        from .research_programs import propose_research_programs
        from .relation_kernel_composition import collect_relation_kernels, compose_relation_kernels
        before = budget.spent
        if relation_kernels is None:
            input_bindings = self._relation_input_bindings(spec)
            relation_kernels = collect_relation_kernels(spec,
                self.library.meta_priors if self.config.use_memory else [], budget,
                input_bindings=input_bindings or None)
        # When full learned calls also compete, fund the single-kernel and
        # eight temporal fits first; joint expansion must not consume their
        # entire replay/search allowance before they have been tried.
        kernel_budget = (SearchBudget(min(budget.remaining,
            27 * len(training) * min(len(relation_kernels), self.config.max_candidates)))
            if calls else budget)
        kernel_claims = compose_relation_kernels(spec, training, kernel_budget, relation_kernels,
            max_candidates=self.config.max_candidates, tolerance=self.config.minimum_improvement)
        if calls:
            budget.spend(kernel_budget.spent)
        if kernel_claims:
            actions.append({"action": "compose_relation_kernels", "compute_cost": budget.spent - before,
                "alternatives": [{"training_loss": item["training_loss"],
                    "source_ids": item["method_source_ids"]} for item in kernel_claims]})
        claims = drafts + kernel_claims
        if any(item["training_loss"] <= self.config.minimum_improvement for item in claims):
            return claims
        claims.extend(self._replay_program_claims(spec, training, budget, calls)
                      if replayed is None else replayed)
        if not any(item["training_loss"] <= self.config.minimum_improvement for item in claims):
            claims.extend(self._object_scope_claims(spec, training, budget, actions))
        if not any(item["training_loss"] <= self.config.minimum_improvement for item in claims):
            from .relation_revisions import propose_relation_revisions
            local = SearchBudget(min(12000, budget.remaining // 2))
            claims.extend(propose_relation_revisions(spec, training, local, calls,
                tolerance=self.config.minimum_improvement))
            budget.spend(local.spent)
        if not any(item["training_loss"] <= self.config.minimum_improvement for item in claims):
            claims.extend(propose_research_programs(spec, training, budget, calls,
                                                    max_candidates=self.config.max_candidates))
        if (relation_kernels and budget.remaining and not any(
                item["training_loss"] <= self.config.minimum_improvement for item in claims)):
            claims.extend(compose_relation_kernels(spec, training, budget, relation_kernels,
                max_candidates=self.config.max_candidates, tolerance=self.config.minimum_improvement))
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

    def _research_conditional_experiments(self, spec, selected, training, validation,
                                         finalists, budget, actions, *, acquire,
                                         observation_allowance, serial=0):
        """Purchase matched interventions before refitting a conditional response."""
        from .conditional_experiments import (propose_conditional_experiments,
                                             assess_conditional_experiment)
        from .conditional_coordinates import propose_experimental_conditional_coordinates

        controls = set(spec.feature_bounds)
        entry_quote = _conditional_experiment_entry_quote(spec, len(training), len(validation))
        if (selected is None or self.config.strategy == "fixed" or spec.risk_quantile != .5
                or spec.decision_costs or not set(spec.feature_names) <= controls
                or _object_schema(spec) or spec.context.get("object_model")
                or selected.get("structure") or observation_allowance < entry_quote["observation_rows"]):
            return selected, training, finalists
        plan = selected.get("research_program", {})
        conditional = any(step["operation"] in ("discover_conditional_coordinates", "discover_composed_conditions")
                          for step in plan.get("selection_steps", []))
        if (max(selected.get("training_loss", float("inf")), selected.get("validation_loss", float("inf")))
                <= self.config.minimum_improvement
                and any(step.get("experiment_evidence") for step in plan.get("selection_steps", []))):
            return selected, training, finalists
        if (not conditional and max(selected.get("training_loss", float("inf")),
                                    selected.get("validation_loss", float("inf")))
                <= self.config.minimum_improvement):
            return selected, training, finalists
        # Do not buy experiments whose joint refit and later confirmation are
        # already unaffordable. The existing model stays available throughout.
        if budget.remaining < entry_quote["compute_cost"]:
            return selected, training, finalists
        programs = list(dict.fromkeys(entry["candidate"].program
                                      for entry in [selected, *finalists]))[:4]
        composed_rows = (2 * (len(spec.feature_names) + 1) + 4) * min(12, spec.max_probe_length) + 28
        if len(spec.feature_names) >= 4 and observation_allowance >= composed_rows:
            return self._research_composed_experiments(spec, selected, training, validation,
                finalists, budget, actions, acquire=acquire, programs=programs,
                observation_allowance=observation_allowance, serial=serial)
        before = budget.spent
        plans = propose_conditional_experiments(spec, training, programs, budget,
            serial=serial, observation_budget=min(96, observation_allowance - 12))
        evidence, purchased, probe_ids = [], [], []
        for pack in plans:
            reference = acquire((), "conditional_matched_reference", probe=pack["reference"])
            if reference is None:
                break
            purchased.extend(reference)
            probe_ids.append(pack["reference"].probe_id)
            for intervention in pack["interventions"]:
                measured = acquire((), "conditional_matched_intervention", probe=intervention["probe"])
                if measured is None:
                    break
                purchased.extend(measured)
                probe_ids.append(intervention["probe"].probe_id)
                outcome = assess_conditional_experiment(spec, intervention, reference, measured, budget)
                evidence.append(outcome)
                actions.append({"action": "assess_conditional_matched_intervention",
                    "reference_probe_id": pack["reference"].probe_id,
                    "probe_id": intervention["probe"].probe_id, "evidence": outcome})
        training = [*training, *purchased]
        local = SearchBudget(max(0, min(90000, budget.remaining - len(validation) * 10)))
        proposals = propose_experimental_conditional_coordinates(spec, training, local,
            basis=_terms(spec.feature_names, quadratic=True), evidence=evidence,
            max_candidates=self.config.max_candidates)
        budget.spend(local.spent)
        fallback = initial_prediction(training, spec.risk_quantile)
        assessed = self._assess([selected, *proposals], validation, budget, fallback, spec.risk_quantile)
        if budget.spend(len(training)):
            selected = {**selected, "training_loss": loss(selected["candidate"].program,
                training, fallback, spec.risk_quantile)}
            if assessed:
                assessed[0]["training_loss"] = selected["training_loss"]
        if assessed:
            lowest = min(max(entry["training_loss"], entry["validation_loss"]) for entry in assessed)
            adequate = [entry for entry in assessed if max(entry["training_loss"],
                entry["validation_loss"]) <= lowest + self.config.minimum_improvement]
            selected = min(adequate, key=lambda entry: entry["candidate"].program.size)
            finalists = assessed
        # A good response fit can still use a broad, untested gate interval.
        # Purchase local levels before freezing it, then jointly refit on those
        # observations. Ordinary confirmation remains a separate acquisition.
        fitted = min(proposals, key=lambda entry: entry["training_loss"]) if proposals else None
        if (fitted is not None and fitted["training_loss"] <= self.config.minimum_improvement
                and observation_allowance - len(purchased) >= 12
                and budget.remaining >= (len(training) + 12) * 88 + len(validation) * 10):
            from .conditional_coordinates import revise_conditional_threshold
            step = fitted["research_program"]["selection_steps"][-1]
            control = step["gate_coordinate"]["name"]
            source = next(item for item in evidence if item.get("control") == control
                          and item.get("gain", 0.) > 1e-10)
            refined_evidence = {**source, "threshold_interval": step["threshold_interval"],
                "width": step["width"], "width_candidates": [step["width"]],
                "placement": step["placement"], "placement_candidates": [step["placement"]],
                "boundary_source": "joint_paid_response_fit"}
            left, right = step["threshold_interval"]
            measured, boundary_trials = [], []
            response = fitted["candidate"].program
            for index in range(12):
                middle = (left + right) / 2.
                point = {**source["background"], control: middle}
                if not budget.spend(2 * response.size + 2):
                    break
                alternatives = [revise_conditional_threshold(response, control, threshold)
                                for threshold in (left, right)]
                predictions = [program.predict(Observation(point, 0.)) for program in alternatives]
                if (predictions[0] - predictions[1]) ** 2 <= self.config.minimum_improvement:
                    break
                probe = Probe(f"conditional_boundary:{serial}:{index}:{control}", (point,))
                actual = acquire((), "conditional_local_boundary_scan", probe=probe)
                if actual is None or not budget.spend(2):
                    break
                measured.extend(actual)
                probe_ids.append(probe.probe_id)
                errors = [(prediction - actual[0].target) ** 2 for prediction in predictions]
                side = min(range(2), key=errors.__getitem__)
                supported = errors[side] <= self.config.minimum_improvement
                boundary_trials.append({"probe_id": probe.probe_id, "level": middle,
                    "branch_predictions": predictions, "target": actual[0].target,
                    "supported": supported, "branch": side if supported else None})
                if not supported:
                    break
                if side == 0:
                    right = middle
                else:
                    left = middle
            if measured:
                refined_evidence["threshold_interval"] = [left, right]
                refined_evidence["boundary_trials"] = boundary_trials
                training.extend(measured)
                purchased.extend(measured)
                local = SearchBudget(max(0, min(90000, budget.remaining - len(validation) * 10)))
                revised = propose_experimental_conditional_coordinates(spec, training, local,
                    basis=_terms(spec.feature_names, quadratic=True), evidence=[refined_evidence],
                    max_candidates=self.config.max_candidates)
                budget.spend(local.spent)
                fallback = initial_prediction(training, spec.risk_quantile)
                compared = self._assess([selected, *revised], validation, budget, fallback, spec.risk_quantile)
                if budget.spend(len(training)) and compared:
                    compared[0]["training_loss"] = loss(selected["candidate"].program,
                        training, fallback, spec.risk_quantile)
                if compared:
                    lowest = min(max(entry["training_loss"], entry["validation_loss"]) for entry in compared)
                    adequate = [entry for entry in compared if max(entry["training_loss"],
                        entry["validation_loss"]) <= lowest + self.config.minimum_improvement]
                    selected = min(adequate, key=lambda entry: entry["candidate"].program.size)
                    finalists = compared
                actions.append({"action": "refine_conditional_measured_boundary",
                    "before_interval": step["threshold_interval"], "after_interval": [left, right],
                    "boundary_trials": boundary_trials, "compute_cost": local.spent,
                    "joint_candidates": len(revised), "observation_cost": len(measured)})
                proposals.extend(revised)
        actions.append({"action": "research_conditional_matched_experiments",
            "probe_ids": probe_ids, "observation_cost": len(purchased),
            "compute_cost": budget.spent - before, "experiment_evidence": evidence,
            "joint_candidates": len(proposals), "training_loss": selected.get("training_loss"),
            "validation_loss": selected.get("validation_loss")})
        return selected, training, finalists

    def _research_composed_experiments(self, spec, selected, training, validation,
                                      finalists, budget, actions, *, acquire, programs,
                                      observation_allowance, serial):
        """Purchase contexts and crosses before discovering two temporal gates."""
        from .conditional_experiments import assess_conditional_experiment
        from .conditional_coordinates import propose_experimental_conditional_coordinates
        from .composed_condition_experiments import (propose_composed_condition_experiments,
            propose_composed_cross_experiment, assess_composed_cross_experiment,
            propose_composed_response_probes, assess_composed_response_probe)
        from .composed_conditions import propose_composed_conditions, propose_measured_window_response

        before = budget.spent
        plans = propose_composed_condition_experiments(spec, training, programs, budget,
            serial=serial, observation_budget=observation_allowance - 76)
        evidence, purchased, probe_ids = [], [], []
        for pack in plans:
            reference = acquire((), "composed_matched_reference", probe=pack["reference"])
            if reference is None:
                break
            purchased.extend(reference)
            probe_ids.append(pack["reference"].probe_id)
            for intervention in pack["interventions"]:
                measured = acquire((), "composed_matched_intervention", probe=intervention["probe"])
                if measured is None:
                    break
                purchased.extend(measured)
                probe_ids.append(intervention["probe"].probe_id)
                outcome = assess_conditional_experiment(spec, intervention, reference, measured, budget)
                evidence.append(outcome)
                actions.append({"action": "assess_composed_matched_intervention",
                    "reference_probe_id": pack["reference"].probe_id,
                    "probe_id": intervention["probe"].probe_id, "evidence": outcome})
        if not purchased:
            return selected, training, finalists
        cross = propose_composed_cross_experiment(spec, evidence, budget, serial=serial,
            observation_budget=observation_allowance - len(purchased) - 28)
        joint = None
        if cross is not None:
            rows_by_probe = {}
            for probe in [cross["reference"], *(item["probe"] for item in cross["interventions"])]:
                measured = acquire((), "composed_cross_intervention", probe=probe)
                if measured is None:
                    break
                rows_by_probe[probe.probe_id] = measured
                purchased.extend(measured)
                probe_ids.append(probe.probe_id)
            if len(rows_by_probe) == 4:
                joint = assess_composed_cross_experiment(spec, cross, rows_by_probe, budget)
                actions.append({"action": "assess_composed_cross_intervention", "evidence": joint})
        training = [*training, *purchased]

        def fit(actual_evidence):
            local = SearchBudget(max(0, budget.remaining - len(validation) * 10 - len(training)))
            proposals = propose_measured_window_response(spec, training, local,
                basis=_terms(spec.feature_names, quadratic=True), evidence=actual_evidence,
                max_candidates=self.config.max_candidates)
            proposer = propose_composed_conditions if joint is not None else propose_experimental_conditional_coordinates
            proposals.extend(proposer(spec, training, local,
                basis=_terms(spec.feature_names, quadratic=True), evidence=actual_evidence,
                max_candidates=max(0, self.config.max_candidates - len(proposals))))
            budget.spend(local.spent)
            return proposals

        def compare(incumbent, proposals):
            fallback = initial_prediction(training, spec.risk_quantile)
            assessed = self._assess([incumbent, *proposals], validation, budget, fallback, spec.risk_quantile)
            if assessed and budget.spend(len(training)):
                assessed[0]["training_loss"] = loss(incumbent["candidate"].program,
                    training, fallback, spec.risk_quantile)
            if not assessed:
                return incumbent, finalists
            lowest = min(max(entry["training_loss"], entry["validation_loss"]) for entry in assessed)
            adequate = [entry for entry in assessed if max(entry["training_loss"],
                entry["validation_loss"]) <= lowest + self.config.minimum_improvement]
            return min(adequate, key=lambda entry: entry["candidate"].program.size), assessed

        refined = joint
        if (joint is not None and joint.get("status") == "conditional"
                and observation_allowance - len(purchased) >= 28):
            refined = deepcopy(joint)
            boundary_trials, measured_boundaries = [], []
            for axis, context in enumerate(joint["active_contexts"]):
                control, background = context["control"], context["background"]
                left, right = refined["threshold_intervals"][axis]
                prototypes, curves = [], None

                def research_curve(reason):
                    probes = propose_composed_response_probes(spec, context, serial=serial)
                    if sum(probe.cost for probe in probes) > observation_allowance - len(purchased) - len(measured_boundaries):
                        return None
                    rows = []
                    for probe in probes:
                        actual = acquire((), "distinguish_condition_response_curve", probe=probe)
                        if actual is None:
                            break
                        rows.extend(actual)
                        measured_boundaries.extend(actual)
                        probe_ids.append(probe.probe_id)
                    outcome = assess_composed_response_probe(spec, context, rows, budget,
                        minimum_improvement=self.config.minimum_improvement)
                    actions.append({"action": "revise_condition_response_experiment",
                        "reason": reason, "control": control,
                        "observation_cost": len(rows), "evidence": outcome})
                    if outcome["status"] != "supported_response_curve":
                        return None
                    refined["active_contexts"][axis] = outcome
                    refined["threshold_intervals"][axis] = outcome["threshold_interval"]
                    return outcome

                if context.get("response_model") == "varying_response":
                    context = research_curve("crossed_constant_response_counterexample")
                    if context is None:
                        continue
                    curves = [Program.from_dict(context["branch_responses"][side]) for side in ("low", "high")]
                    left, right = context["threshold_interval"]
                else:
                    for side, endpoint in enumerate(spec.feature_bounds[control]):
                        probe = Probe(f"composed_endpoint:{serial}:{axis}:{side}",
                                      ({**background, control: endpoint},))
                        actual = acquire((), "composed_boundary_endpoint", probe=probe)
                        if actual is None:
                            break
                        measured_boundaries.extend(actual)
                        probe_ids.append(probe.probe_id)
                        prototypes.append(actual[0].target)
                    if len(prototypes) != 2:
                        break
                index = 0
                while index < 12:
                    if observation_allowance - len(purchased) - len(measured_boundaries) < 1:
                        break
                    middle = (left + right) / 2.
                    point = {**background, control: middle}
                    if not budget.spend(2 + sum(program.size for program in curves or ())):
                        break
                    predictions = [program.predict(Observation(point, 0.)) for program in curves] if curves else prototypes
                    if (predictions[0] - predictions[1]) ** 2 <= self.config.minimum_improvement:
                        break
                    probe = Probe(f"composed_boundary:{serial}:{axis}:{index}:{control}", (point,))
                    actual = acquire((), "composed_local_boundary_scan", probe=probe)
                    if actual is None:
                        break
                    measured_boundaries.extend(actual)
                    probe_ids.append(probe.probe_id)
                    errors = [(value - actual[0].target) ** 2 for value in predictions]
                    side = min(range(2), key=errors.__getitem__)
                    supported = errors[side] <= self.config.minimum_improvement
                    boundary_trials.append({"control": control, "probe_id": probe.probe_id,
                        "level": middle, "branch_predictions": predictions, "target": actual[0].target,
                        "supported": supported, "branch": side if supported else None})
                    if not supported:
                        if curves is None:
                            context = research_curve("local_constant_response_counterexample")
                            if context is not None:
                                curves = [Program.from_dict(context["branch_responses"][name]) for name in ("low", "high")]
                                left, right = context["threshold_interval"]
                                continue
                        break
                    if side == 0:
                        left = middle
                    else:
                        right = middle
                    index += 1
                refined["threshold_intervals"][axis] = [left, right]
            if measured_boundaries:
                refined["boundary_trials"] = boundary_trials
                training.extend(measured_boundaries)
                purchased.extend(measured_boundaries)
                actions.append({"action": "refine_composed_measured_boundaries",
                    "before_intervals": joint["threshold_intervals"],
                    "after_intervals": refined["threshold_intervals"],
                    "boundary_trials": boundary_trials, "observation_cost": len(measured_boundaries)})
        proposals = fit(refined if refined is not None else evidence)
        selected, finalists = compare(selected, proposals)
        actions.append({"action": "research_composed_matched_experiments",
            "probe_ids": probe_ids, "observation_cost": len(purchased),
            "compute_cost": budget.spent - before, "experiment_evidence": evidence,
            "cross_evidence": joint, "joint_candidates": len(proposals),
            "training_loss": selected.get("training_loss"), "validation_loss": selected.get("validation_loss")})
        return selected, training, finalists

    def _revise_paid_structure(self, spec, parent, paid_rows, budget, actions,
                               *, finalists=(), transitions=(), paid_scores=None,
                               research_action="structural_research"):
        """Execute a declared existing research segment on the complete paid pool."""
        from .unified_structure import propose_structures, propose_response_revisions
        from .evaluation import point_loss
        from statistics import mean
        proposals_by_action = {
            "structural_research": (propose_structures, 90000, 16),
            "response_basis_revision": (propose_response_revisions, 32000, 8),
        }
        if research_action not in proposals_by_action:
            raise ValueError(f"Unsupported paid structure research action: {research_action}")
        proposer, search_limit, maximum_new = proposals_by_action[research_action]
        pool = list(finalists)
        if parent is not None and not any(item is parent for item in pool):
            pool.append(parent)
        start = budget.spent
        event = {"action": "revise_paid_structure", "observation_rows": len(paid_rows),
                 "research_action": research_action,
                 "new_observations": 0, "retained_candidates": len(pool),
                 "comparison_basis": "all_paid_rows", "compute_cost": 0}
        result = {"status": "unavailable", "selected": parent, "claims": pool,
                  "new_claims": [], "compared_claims": [], "compute_cost": 0,
                  "full_comparison": False, "action": event}

        predictions = {}

        def finish(status):
            result["status"] = event["status"] = status
            result["compute_cost"] = event["compute_cost"] = budget.spent - start
            event["full_comparison"] = result["full_comparison"]
            chosen = result["selected"]
            if chosen is not None and chosen["candidate"].program in predictions:
                result["selected_predictions"] = predictions[chosen["candidate"].program]
            actions.append(event)
            return result

        if parent is None or not paid_rows or not _object_schema(spec):
            return finish("unavailable")
        if not parent.get("structure"):
            # Ordinary program replay keeps the fitted graph and executable
            # relation binding, rather than a StructuralDiscovery snapshot.
            # The existing structural proposer can consume that same parent.
            procedure = parent.get("research_program") or {}
            relation = next((step for step in reversed(procedure.get("selection_steps", []))
                if step["operation"] == "resolve_relation_intervals"), None)
            if relation is None or not relation.get("parameters"):
                return finish("unavailable")
            from .structural_research_program import research_program_fit_contract
            fitted = research_program_fit_contract(procedure)
            if fitted["program"] != parent["candidate"].program:
                return finish("evidence_unresolved")
        readable = [item for item in pool if all(
            item["candidate"].program.required_features() <= row.features.keys() for row in paid_rows)]
        unavailable = [item for item in pool if not any(item is entry for entry in readable)]
        event["uncompared_candidate_ids"] = [item["candidate"].candidate_id for item in unavailable]
        if not any(item is parent for item in readable):
            return finish("evidence_unresolved")
        fallback = parent.get("fallback", initial_prediction(paid_rows, spec.risk_quantile))
        programs = list(dict.fromkeys(item["candidate"].program for item in readable))
        scores = {program: paid_scores[program] for program in programs
                  if paid_scores is not None and program in paid_scores}
        missing = [program for program in programs if program not in scores]
        old_cost = len(paid_rows) * len(missing)
        event.update(reused_scores=len(scores), old_scoring_compute=old_cost)
        if not budget.can_spend(old_cost):
            return finish("compute_unresolved")
        def score(program):
            budget.spend(len(paid_rows))
            predictions[program] = [predict(program, row, fallback) for row in paid_rows]
            scores[program] = mean(point_loss(value, row.target, spec.risk_quantile)
                                  for value, row in zip(predictions[program], paid_rows))

        for program in missing:
            score(program)

        def compared(entries):
            return [{**item, "training_loss": scores[item["candidate"].program],
                     "validation_loss": scores[item["candidate"].program], "fallback": fallback,
                     "selection_basis": "all_acquired_evidence"} for item in entries]

        checked = compared(readable)
        prior = next(item for item in checked if item["candidate"] == parent["candidate"])
        best = self._best(checked)
        if prior["validation_loss"] <= best["validation_loss"] + self.config.minimum_improvement:
            best = prior
        result.update(selected=best, claims=[*checked, *unavailable], compared_claims=checked,
                      full_comparison=True)
        event.update(prior_loss=prior["validation_loss"], selected_loss=best["validation_loss"],
                     changed_program=best["candidate"].program != parent["candidate"].program)
        if best["validation_loss"] <= self.config.minimum_improvement:
            event["requires_fresh_confirmation"] = event["changed_program"]
            return finish("improved" if event["changed_program"] else "already_explained")

        # Structure research may retain eight drafts before searching another
        # family; the existing response revision returns at most eight total.
        # Both consume their complete return without shrinking the old pool.
        comparison_reserve = maximum_new * len(paid_rows)
        local_limit = min(search_limit, max(0, budget.remaining - comparison_reserve))
        event.update(new_scoring_reserve=comparison_reserve, search_allowance=local_limit)
        if local_limit <= 0:
            result["full_comparison"] = False
            return finish("compute_unresolved")
        local = SearchBudget(local_limit)
        proposals = proposer(spec, paid_rows, local, parent=parent,
            tolerance=self.config.minimum_improvement, transitions=transitions)
        budget.spend(local.spent)
        result["new_claims"] = proposals
        event.update(search_compute=local.spent, new_candidates=len(proposals))
        new_programs = list(dict.fromkeys(item["candidate"].program for item in proposals
                                         if item["candidate"].program not in scores))
        if not budget.can_spend(len(new_programs) * len(paid_rows)):
            result.update(claims=[*result["claims"], *proposals], full_comparison=False)
            return finish("compute_unresolved")
        for program in new_programs:
            score(program)
        checked.extend(compared(proposals))
        best = self._best(checked)
        if prior["validation_loss"] <= best["validation_loss"] + self.config.minimum_improvement:
            best = prior
        changed = best["candidate"].program != parent["candidate"].program
        result.update(selected=best, claims=[*checked, *unavailable], compared_claims=checked,
                      full_comparison=True)
        event.update(selected_candidate_id=best["candidate"].candidate_id,
                     selected_loss=best["validation_loss"], changed_program=changed,
                     requires_fresh_confirmation=changed)
        return finish("improved" if changed else "no_improvement")

    def _consume_residual_answer(self, spec, support, paid_rows, budget, actions, graph,
                                 *, parent=None, finalists=()):
        """Consume an already computed correction and compare the whole paid pool.

        The saved baseline owns the correction even if another candidate has
        since become the parent. These rows select a working response; only a
        subsequent frozen comparison can independently confirm it.
        """
        pool = list(finalists)
        if parent is not None and not any(item is parent for item in pool):
            pool.append(parent)
        event = {"action": "consume_paid_residual_answer", "status": "unavailable",
                 "observation_rows": len(paid_rows), "compute_cost": 0,
                 "new_observations": 0, "new_fits": 0}
        result = {"status": "unavailable", "claims": pool, "selected": parent,
                  "compute_cost": 0, "action": event}

        def finish(status):
            event["status"] = result["status"] = status
            actions.append(event)
            return result

        if self.config.max_question_depth <= 0:
            return finish("recursion_disabled")
        counterexample = (support or {}).get("structural_counterexample") or {}
        baseline_payload = (support or {}).get("counterexample_baseline_program")
        if not counterexample.get("program") or baseline_payload is None or not paid_rows:
            return finish("unavailable")
        baseline = Program.from_dict(baseline_payload)
        correction = Program.from_dict(counterexample["program"])
        required = baseline.required_features() | correction.required_features()
        if any(not required <= row.features.keys() for row in paid_rows):
            return finish("evidence_unresolved")
        readable = [item for item in pool if all(
            item["candidate"].program.required_features() <= row.features.keys() for row in paid_rows)]
        unavailable = [item for item in pool if not any(item is entry for entry in readable)]
        programs = list(dict.fromkeys(item["candidate"].program for item in readable))
        event["uncompared_candidate_ids"] = [item["candidate"].candidate_id for item in unavailable]
        if parent is not None and not any(item is parent for item in readable):
            return finish("evidence_unresolved")
        # The residual node costs two row passes; its dependent parent costs
        # one. The new response and every readable old program are then scored
        # with the parent's common fallback, rather than mixing graph scores.
        required_compute = len(paid_rows) * (4 + len(programs))
        event.update(compute_required=required_compute,
                     comparison_basis="all_paid_rows", retained_candidates=len(pool))
        if not budget.can_spend(required_compute):
            return finish("compute_unresolved")

        start = budget.spent
        question_id = f"paid_residual_{len(graph.nodes) if graph is not None else 0}"
        # Execute just this dependency pair. Other ready graph questions must
        # not launch another search while consuming an already paid answer.
        research = ResearchGraph()
        node = research.add_residual_question(question_id, baseline, paid_rows)
        node.expression = correction
        answers = research.execute_ready(paid_rows, budget, max_nodes=2,
                                         risk_quantile=spec.risk_quantile)
        if graph is not None:
            graph.nodes.update(research.nodes)
            graph.events.extend(research.events)
        combined_id = f"{question_id}:combined"
        combined = next((item for item in answers
                         if item.candidate_id == f"answer:{combined_id}"), None)
        if combined is None:
            event["compute_cost"] = result["compute_cost"] = budget.spent - start
            return finish("answer_unresolved")

        fallback = (parent.get("fallback", initial_prediction(paid_rows, spec.risk_quantile))
                    if parent is not None else initial_prediction(paid_rows, spec.risk_quantile))
        scores = {}
        for program in programs:
            budget.spend(len(paid_rows))
            scores[program] = loss(program, paid_rows, fallback, spec.risk_quantile)
        checked = [{**item, "training_loss": scores[item["candidate"].program],
                    "validation_loss": scores[item["candidate"].program],
                    "fallback": fallback, "selection_basis": "all_acquired_evidence"}
                   for item in readable]
        budget.spend(len(paid_rows))
        value = loss(combined.program, paid_rows, fallback, spec.risk_quantile)
        new = {"candidate": combined, "kind": "noise_residual_answer", "family_id": "T09",
               "training_loss": value, "validation_loss": value, "fallback": fallback,
               "selection_basis": "all_acquired_evidence"}
        checked.append(new)
        best = self._best(checked)
        prior = next((item for item in checked[:-1]
                      if parent is not None and item["candidate"] == parent["candidate"]), None)
        if prior is not None and prior["validation_loss"] <= best["validation_loss"] + self.config.minimum_improvement:
            best = prior
        changed = (parent is None or best["candidate"].program != parent["candidate"].program)
        event.update(question_id=question_id, source_kind=counterexample.get("kind"),
                     baseline_matches_parent=parent is not None and baseline == parent["candidate"].program,
                     prior_loss=prior["validation_loss"] if prior else None,
                     answer_loss=value, selected_loss=best["validation_loss"],
                     selected_candidate_id=best["candidate"].candidate_id,
                     changed_program=changed, requires_fresh_confirmation=changed,
                     compute_cost=budget.spent - start)
        result.update(claims=[*checked, *unavailable], compared_claims=checked,
                      selected=best, compute_cost=event["compute_cost"])
        return finish("improved" if changed else "no_improvement")

    def _support(self, program, training, validation, spec, budget):
        """Keep the external loss fixed while testing residual explanations."""
        from .unified_uncertainty import assess_noisy_adequacy
        support = assess_noisy_adequacy(spec, program, training, validation, budget,
            minimum_improvement=self.config.minimum_improvement)
        if support.get("residual_comparisons"):
            support.update(counterexample_baseline_program=program.to_dict(),
                           counterexample_fallback=0.)
        return support

    def _confirmation_compute_quote(self, spec, rows, *, training_rows,
                                    evidence_rows=0, rivals=0, noise_check=False,
                                    calibration_rows=0):
        """Upper quote for the existing numeric confirmation operations."""
        quantile = spec.risk_quantile != .5
        scoring = rows * (1 + int(quantile) + rivals) + evidence_rows + calibration_rows
        if not noise_check:
            return scoring
        # compile_claims returns affine, additive and coupled fits; two sparse
        # fits, kink and projection for multiple coordinates; and <=4 histories.
        families = 3 + (4 if len(spec.feature_names) > 1 else 0)
        families += min(4, max(0, spec.max_probe_length - 1))
        residual_checks = families * (rows + (2 * training_rows if quantile else 0))
        return (scoring + 2 * (training_rows + rows) + 18000 + residual_checks
                + 4 * rows + 4 * evidence_rows)

    def _refit_paid_response(self, spec, selected, training, validation, budget, actions):
        """Improve an unresolved response with paid rows, without buying confirmation.

        Only the existing response basis is refitted. Shared old evidence can
        select a working prediction; it cannot establish a new supported law.
        """
        from .mechanism_revision import propose_response_refits

        if not training or not validation:
            return None
        scoring_cost = 2 * (len(training) + len(validation))
        if budget.remaining <= scoring_cost:
            return None
        before = budget.spent
        local = SearchBudget(budget.remaining - scoring_cost)
        proposals = propose_response_refits(spec, training, local, parent=selected,
            max_candidates=1, priors=self.library.meta_priors if self.config.use_memory else ())
        budget.spend(local.spent)
        if not proposals:
            return None
        proposal = proposals[0]
        fallback = selected.get("fallback", initial_prediction(training, spec.risk_quantile))
        budget.spend(scoring_cost)
        losses = []
        for entry in (selected, proposal):
            response = entry["candidate"].program
            losses.append((loss(response, training, fallback, spec.risk_quantile),
                           loss(response, validation, fallback, spec.risk_quantile)))
        old, new = losses
        improved = (new[1] + self.config.minimum_improvement < old[1]
            and new[0] <= old[0] + self.config.minimum_improvement)
        actions.append({"action": "refit_paid_response_pending_confirmation",
            "training_rows": len(training), "validation_rows": len(validation),
            "before_training_loss": old[0], "after_training_loss": new[0],
            "before_validation_loss": old[1], "after_validation_loss": new[1],
            "accepted_as_working_prediction": improved,
            "observation_cost": 0, "compute_cost": budget.spent - before,
            "requires_fresh_confirmation": True})
        if not improved:
            return None
        proposal.update(training_loss=new[0], validation_loss=new[1], fallback=fallback,
            selection_basis="all_acquired_evidence")
        if selected.get("research_action_origin") is not None:
            proposal["research_action_origin"] = selected["research_action_origin"]
        return proposal

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
        old_loss = loss(original["candidate"].program, rows,
                        original.get("fallback", fallback), spec.risk_quantile)
        if calibrated is None or not budget.spend(len(rows)):
            return original, old_loss
        new_loss = loss(calibrated["candidate"].program, rows,
                        calibrated.get("fallback", fallback), spec.risk_quantile)
        chosen = calibrated if new_loss + self.config.minimum_improvement < old_loss else original
        actions.append({"action": "compare_tail_calibration", "rows": len(rows),
            "original_loss": old_loss, "calibrated_loss": new_loss,
            "accepted": chosen is calibrated, "fixed_quantile": spec.risk_quantile})
        if chosen is calibrated:
            chosen["calibration_parent_validation_loss"] = chosen["validation_loss"]
            chosen["validation_loss"] = new_loss
        return chosen, new_loss if chosen is calibrated else old_loss

    def _retain_confirmation_alternative(self, selected, finalists, rows, spec, budget,
                                         fallback, actions, *, incumbent_loss):
        """Use a paid counterexample to retain the better already fitted answer."""
        best, best_loss, start, compared = selected, incumbent_loss, budget.spent, 0
        seen = {selected["candidate"].program}
        for entry in finalists:
            program = entry["candidate"].program
            required = {name for name, _ in program.observed_variables()} - {"__target__"}
            if program in seen or not required <= set(spec.feature_names):
                continue
            seen.add(program)
            if not budget.spend(len(rows)):
                break
            measured = loss(program, rows, entry.get("fallback", fallback), spec.risk_quantile)
            compared += 1
            if measured + self.config.minimum_improvement < best_loss:
                best, best_loss = entry, measured
        if best is not selected:
            best = {**best, "validation_loss": best_loss,
                    "selection_basis": "all_acquired_evidence"}
        actions.append({"action": "compare_confirmation_alternatives", "rows": len(rows),
            "candidates_compared": compared, "before_loss": incumbent_loss, "after_loss": best_loss,
            "selected_kind": best["kind"], "changed": best is not selected,
            "compute_cost": budget.spent - start, "evidence_basis": "paid_confirmation_counterexample"})
        return best, best_loss

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

    def _charge_action_costs(self, origin, costs, actions, stage):
        if origin is None or not any(costs.values()):
            return
        if origin["trial"] is None:
            for name, value in costs.items():
                origin["pending_costs"][name] += value
            actions.append({"action": "research_action_pending_cost", "research_action": origin["action"],
                "context": origin["context"], "pending_id": origin["pending_id"],
                "trial": None, "stage": stage, **costs})
        else:
            feedback = ResearchActionPolicy.charge_followup(self.library.action_experience, origin, **costs)
            actions.append({"action": "research_action_followup", "research_action": origin["action"],
                "context": origin["context"], "trial": origin["trial"], "stage": stage,
                **costs, "feedback": feedback})

    def _dependent_residual_claims(self, spec, training, budget, parent, graph, actions):
        if not budget.spend(len(training)):
            return []
        base = parent["candidate"].program
        residuals = [Observation(row.features, row.target - base.predict(row), row.history)
                     for row in training]
        # Leave the child's residual transform and both graph answers funded.
        fit_budget = SearchBudget(max(0, budget.remaining - 3 * len(training)))
        children = [_fit(_terms(spec.feature_names, depth, True), residuals, fit_budget,
                         "residual_child", family="T09", q=spec.risk_quantile)
                    for depth in range(1, min(self.config.max_history, 2) + 1)]
        budget.spend(fit_budget.spent)
        children = [child for child in children if child is not None]
        if not children:
            return []
        child = min(children, key=lambda item: item["training_loss"])
        sources = _sources(base)

        def execute_child(expression, retained_base, coefficient, mode):
            child_id = f"residual_{len(graph.nodes)}"
            node = graph.add_residual_question(child_id, retained_base, training)
            node.required_measurements = sorted(set(spec.feature_names) - base.required_features())
            node.expression = expression
            answers = graph.execute_ready(training, budget, max_nodes=2, risk_quantile=spec.risk_quantile)
            combined = next((answer for answer in answers
                             if answer.candidate_id == f"answer:{child_id}:combined"), None)
            actions.append({"action": "execute_dependent_research", "question_id": child_id,
                "answered": combined is not None, "required_measurements": node.required_measurements,
                "parent_revision": mode, "parent_coefficient": coefficient,
                "parent_program": base.to_dict(), "parent_candidate_id": parent["candidate"].candidate_id,
                "parent_source_ids": sources})
            if combined is None:
                return None
            # The graph retains the exact proposed parent scaling, including a
            # zero coefficient. The executable answer drops unused computation.
            combined = replace(combined, program=combined.program.simplified(),
                parents=tuple(dict.fromkeys((*combined.parents, *sources))))
            graph.nodes[f"{child_id}:combined"].answer = combined.program
            return {"candidate": combined, "kind": "dependent_residual_revision", "family_id": "T09",
                "training_loss": graph.nodes[f"{child_id}:combined"].loss}

        fixed = execute_child(child["candidate"].program, base, 1., "fixed_parent")
        claims = [fixed] if fixed is not None else []
        if child["training_loss"] <= self.config.minimum_improvement:
            return claims
        # An incomplete-coordinate parent can contain spurious terms that a
        # short residual child cannot cancel. Re-estimate its contribution on
        # original targets together with the new child, rather than forcing its
        # coefficient to remain one. The fixed-parent answer still competes.
        for depth in range(1, min(self.config.max_history, 2) + 1):
            terms = _terms(spec.feature_names, depth, True)
            fit_cost = len(training) * (len(terms) + 3)
            if budget.remaining < fit_cost + 3 * len(training):
                break
            budget.spend(fit_cost)
            columns = [[term.predict(row) for row in training] for term in [base, *terms]]
            coefficients = _linear_fit(columns, [row.target for row in training])
            coefficient = coefficients[1] if abs(coefficients[1]) >= 1e-8 else 0.
            retained_base = Program("mul", children=(Program("const", value=coefficient), base))
            expression = _sum_terms(terms, coefficients[2:], coefficients[0]).simplified()
            revised = execute_child(expression, retained_base, coefficient, "reestimated_parent")
            if revised is not None:
                claims.append(revised)
                if revised["training_loss"] <= self.config.minimum_improvement:
                    break
        return claims

    def _select_structure_response_parent(self, spec, parent, assessed,
            retained_on_current, usable_parent, actions, comparison_rows):
        response_parent = parent
        if _object_schema(spec):
            best_response = self._best(assessed)
            if (not usable_parent or (best_response is not None
                    and retained_on_current is not None
                    and best_response["validation_loss"] + self.config.minimum_improvement
                        < retained_on_current["validation_loss"])):
                response_parent = best_response or parent
            actions.append({"action": "select_structure_response_parent",
                "prior_candidate_id": parent["candidate"].candidate_id if parent else None,
                "selected_candidate_id": response_parent["candidate"].candidate_id if response_parent else None,
                "selected_kind": response_parent.get("kind") if response_parent else None,
                "prior_loss": retained_on_current["validation_loss"] if retained_on_current else None,
                "selected_loss": (best_response["validation_loss"] if response_parent is best_response else
                    retained_on_current["validation_loss"] if retained_on_current else None),
                "evidence_basis": "current_working_comparison", "observation_rows": comparison_rows,
                "retained_candidates": len(assessed), "compute_cost": 0})
        return response_parent

    def _research_pool_explained(self, prepared):
        """Stop generation on a fully compared answer, leaving confirmation outside."""
        if not prepared["comparison_complete"]:
            return False
        selected = self._best(prepared["assessed"])
        return bool(selected and max(selected["training_loss"], selected["validation_loss"])
                    <= self.config.minimum_improvement)

    def prepare_research_decision(self, spec, training, validation, budget, actions,
                                  *, parent=None, transitions=(), observation_remaining=0,
                                  measurement_remaining=None, execution=None):
        """Run the ordinary paid prelude once and retain its actual decision state.

        The returned frame contains the live policy/library relationship. Copy
        ``(agent, frame)`` together for independent matched continuations; do
        not rebuild this state from a later winner or rerun the prelude per arm.
        """
        before = budget.spent
        if execution is not None:
            execution.update(search_compute_cost=0, compared_candidates=0, executed_directions=[])

        def executed(name, cost):
            if execution is not None and cost:
                execution["search_compute_cost"] += cost
                execution["executed_directions"].append(name)
        fallback = initial_prediction(training, spec.risk_quantile)
        independent_validation = bool(validation) and all(row not in training for row in validation)
        comparison = validation or training
        uncompared = []

        def assess(claims, *, fitted_comparison=False):
            claims = list(claims)
            checked = self._assess(claims, comparison, budget, fallback, spec.risk_quantile)
            uncompared.extend([item for item in claims if item is not None][len(checked):])
            if self._workflow_episode is not None:
                self._workflow_episode.fitted(checked, training, comparison, spec)
            if execution is not None:
                execution["compared_candidates"] += sum(parent is None or
                    item["candidate"] is not parent["candidate"] for item in checked)
            if fitted_comparison or not independent_validation:
                for claim in checked:
                    claim["selection_basis"] = "all_acquired_evidence"
            return checked

        def finish(terminal=False):
            frame = {"spec": spec, "training": training, "validation": validation,
                "parent": parent, "transitions": transitions,
                "observation_remaining": observation_remaining,
                "measurement_remaining": measurement_remaining,
                "fallback": fallback, "independent_validation": independent_validation,
                "comparison": comparison, "assessed": assessed,
                "comparison_complete": not uncompared, "uncompared": uncompared,
                "terminal": terminal, "prepare_compute_cost": budget.spent - before,
                "remaining_compute": budget.remaining}
            frame["terminal"] = terminal or self._research_pool_explained(frame)
            if not frame["terminal"]:
                frame.update(baseline=baseline,
                    call_origins=call_origins,
                    calls=calls,
                    relation_kernels=relation_kernels,
                    structured=structured,
                    usable_parent=usable_parent,
                    parent_fitted_comparison=parent_fitted_comparison,
                    retained_on_current=retained_on_current,
                    directions=directions,
                    state_identification=state_identification,
                    replay_plans=replay_plans,
                    experiment_compute=experiment_compute,
                    learn_policy=learn_policy,
                    policy=policy)
            return frame

        if self.config.strategy == "fixed":
            claims = [_fit(_terms(spec.feature_names, 2, True), training, budget,
                           "fixed_quadratic_two_step_history", family="T08", q=spec.risk_quantile)]
            claims.extend(self._risk_revisions(spec, claims, training, budget))
            assessed = assess(claims)
            return finish(True)

        # Freeze a cheap current-coordinate alternative before paying for any
        # direction. Availability, not an uncharged reuse fit, defines context.
        fit_start = budget.spent
        affine = _fit(_terms(spec.feature_names), training, budget, "current_affine", q=spec.risk_quantile)
        executed("current_affine", budget.spent - fit_start)
        assessed = assess([affine])
        baseline = self._best(assessed)
        if baseline and max(baseline["training_loss"], baseline["validation_loss"]) <= self.config.minimum_improvement:
            return finish(True)
        call_origins = {}
        calls = self._recall(spec, training, origins=call_origins)
        calls, screened = self._screen_risk_knowledge(spec, training, validation, budget,
                                                     calls, actions, baseline=baseline)
        assessed.extend(screened)
        from .relation_kernel_composition import collect_relation_kernels
        input_bindings = self._relation_input_bindings(spec)
        relation_kernels = (collect_relation_kernels(spec, self.library.meta_priors, budget,
            input_bindings=input_bindings or None)
            if self.config.use_memory and self.config.max_question_depth > 0 else {})
        if spec.context.get("object_model"):
            from .partial_object_knowledge import propose_partial_object_claims
            partial_claims = propose_partial_object_claims(spec, training, budget)
            checked = assess(partial_claims)
            assessed.extend(checked)
            actions.append({"action": "compare_partial_object_claims",
                "alternatives": [{"kind": item["kind"], "validation_loss": item["validation_loss"]} for item in checked],
                "unresolved": list(spec.context["object_model"].get("unresolved", []))})
        structured = _object_schema(spec) is not None or bool(spec.context.get("object_model"))
        usable_parent = (parent is not None and self.config.max_question_depth > 0
            and self.config.max_history > 0
            and parent["candidate"].program.required_features() <= set(spec.feature_names))
        parent_fitted_comparison = bool(usable_parent and (
            parent.get("selection_basis") == "all_acquired_evidence" or parent.get("calibration_parent_kind")))
        retained_on_current = None
        if usable_parent and budget.spend(len(training)):
            # A retained answer competes on this view's paid rows. Its old
            # loss may describe another trajectory or another instrument.
            retained = {**parent, "training_loss": loss(parent["candidate"].program,
                training, fallback, spec.risk_quantile)}
            retained_claims = assess([retained], fitted_comparison=parent_fitted_comparison)
            retained_on_current = retained_claims[0] if retained_claims else None
            assessed.extend(retained_claims)
        directions = []
        state_identification = (state_identification_cost(training)
            if self.config.state_search and not structured and spec.risk_quantile == .5
            and spec.decision_costs is None else 0)
        if state_identification:
            directions.append("state_identification")
        drafts_available = self.config.use_memory and any(
            not item.get("resolved_concept_id") for item in self.library.research_drafts)
        if calls:
            directions.append("callable_transfer")
        replay_plans = self._replay_program_queue(spec, calls, training) if self.config.max_question_depth > 0 else []
        if replay_plans:
            directions.append("program_replay")
        if ((calls or usable_parent) and self.config.max_question_depth > 0
                and spec.risk_quantile == .5 and spec.decision_costs is None):
            directions.append("mechanism_revision")
        coordinate_research = (not structured and spec.risk_quantile == .5
                               and spec.decision_costs is None and len(training) >= 6)
        composed_rows = (2 * (len(spec.feature_names) + 1) + 4) * min(12, spec.max_probe_length) + 28
        composed_experiments = len(spec.feature_names) >= 4 and observation_remaining >= composed_rows + 24
        experiment_rows = min(composed_rows if composed_experiments else 96,
                              max(0, observation_remaining - (24 if composed_experiments else 48)))
        experiment_compute = 0
        if (coordinate_research and set(spec.feature_names) <= set(spec.feature_bounds)
                and experiment_rows >= (len(spec.feature_names) + 1) * min(12, spec.max_probe_length)):
            branch_terms = 2 * (len(_terms(spec.feature_names, quadratic=True)) + 1)
            experiment_compute = min(180000 if composed_experiments else 80000,
                budget.remaining * 2 // 3 if composed_experiments else budget.remaining // 2,
                (len(training) + experiment_rows + 12) * (branch_terms + 2) * 20)
        if coordinate_research and self.config.max_question_depth > 0:
            directions.append("coordinate_reconstruction")
            if len(training) >= 8:
                directions.append("conditional_reconstruction")
        if (calls or relation_kernels or drafts_available) and self.config.max_question_depth > 0:
            directions.append("program_revision")
        if structured:
            directions.append("structural_research")
        if usable_parent:
            directions.append("dependent_residual_revision")

        learn_policy = bool(self.config.learn_research_policy and self.config.use_memory)
        policy = ResearchActionPolicy(training, spec,
            self.library.action_experience if learn_policy else {})
        policy.set_condition("unified:" + ("structured" if structured else "coordinates")
            + (":calls" if calls else ":no_calls") + (":parent" if usable_parent else ":no_parent"))
        if replay_plans:
            policy.set_action_knowledge("program_replay", self._replay_knowledge(spec, replay_plans))
        return finish()

    def _execute_prepared_research_segment(self, prepared, research_action, budget, actions,
                                           *, search_limit, assessed=None):
        """Invoke an existing native segment; its caller owns the common comparison."""
        spec, training = prepared["spec"], prepared["training"]
        local = SearchBudget(search_limit)
        if research_action == "program_replay":
            claims = self._replay_program_claims(spec, training, local, prepared["calls"],
                prepared=prepared["replay_plans"])[:self.config.max_candidates]
        elif research_action == "structural_research":
            pool = prepared["assessed"] if assessed is None else assessed
            response_parent = self._select_structure_response_parent(spec, prepared["parent"], pool,
                prepared["retained_on_current"], prepared["usable_parent"], actions,
                len(prepared["comparison"]))
            fit_features = [row.features for row in training]
            transitions = [row for row in prepared["transitions"] if row["before"] in fit_features]
            if _object_schema(spec) is not None:
                from .unified_structure import propose_structures
                claims = propose_structures(spec, training, local, parent=response_parent,
                    tolerance=self.config.minimum_improvement, transitions=transitions)
            else:
                claims = self._object_explanations(spec, training, local, parent=response_parent,
                    tolerance=self.config.minimum_improvement, transitions=transitions, actions=actions)
        else:
            raise ValueError("Only the declared native replay and structure segments are supported")
        budget.spend(local.spent)
        return claims, local

    def execute_research_decision(self, prepared, research_action, budget, actions):
        """Complete one recorded native segment and its full same-pool comparison.

        No next direction or observation is dispatched. All prior comparison
        scores belong to this prepared state and remain available to both arms.
        The caller decides whether its exposed development return supports any
        method preference; this function does not credit a confirmation.
        """
        if research_action not in {"program_replay", "structural_research"}:
            raise ValueError("Unsupported prepared native research action")
        pool = list(prepared["assessed"])
        incumbent = self._best(pool)
        result = {"status": "unresolved", "execution_complete": False,
            "completion_scope": "bounded_native_segment_and_full_pool_comparison",
            "research_action": research_action, "claims": pool, "new_claims": [],
            "selected": incumbent, "full_comparison": True, "compute_cost": 0,
            "search_compute_cost": 0, "comparison_compute_cost": 0, "new_observations": 0,
            "fallback": prepared["fallback"], "independent_confirmation": False}
        if not prepared["comparison_complete"]:
            return {**result, "status": "preparation_comparison_unresolved",
                "claims": [*pool, *prepared["uncompared"]], "full_comparison": False}
        if prepared["terminal"] or self._research_pool_explained(prepared):
            return {**result, "status": "already_explained", "execution_complete": True}
        if research_action not in prepared["directions"]:
            raise ValueError("The requested segment was not available at this decision")
        spec, comparison = prepared["spec"], prepared["comparison"]
        exports = min(4, self.config.max_candidates) if research_action == "program_replay" else 16
        reserve = exports * len(comparison)
        if research_action == "program_replay":
            conditional = any(any(step["operation"] in (
                "discover_conditional_coordinates", "discover_composed_conditions")
                for step in payload.get("selection_steps", []))
                for _, payload, _, _ in prepared["replay_plans"])
            native_limit = min(64000 if conditional else 4000, budget.remaining // 2)
        else:
            native_limit = min(90000, budget.remaining // 2)
            if spec.context.get("object_model") and not _object_schema(spec):
                native_limit = min(90000, max(0, budget.remaining - reserve))
        search_limit = min(native_limit, max(0, budget.remaining - reserve))
        result["quote"] = {"search_compute_limit": search_limit,
            "new_comparison_reserve": reserve, "maximum_new_candidates": exports,
            "old_comparison_compute": 0, "compute_limit": budget.remaining,
            "observation_cost": 0}
        if search_limit <= 0:
            return {**result, "status": "complete_segment_unfunded"}
        new, local = self._execute_prepared_research_segment(prepared, research_action,
            budget, actions, search_limit=search_limit)
        compared = self.compare_research_decision_return(prepared, research_action, new,
            budget, actions, search_compute_cost=local.spent,
            search_metadata={"search_budget_denied": bool(local.denied_requests),
                "last_denied_request": deepcopy(local.last_denied_request)})
        return {**compared, "quote": result["quote"]}

    def compare_research_decision_return(self, prepared, research_action, new_claims,
                                         budget, actions, *, paid_scores=None,
                                         reused_paid_return=False, search_compute_cost=0,
                                         search_metadata=None):
        """Consume a completed native return against the entire prepared pool.

        ``paid_scores`` explicitly binds executable Programs to scores on this
        frame's comparison rows. The caller resolves their historical evidence
        before supplying them; this entry does not load or match other states.
        Search cost is newly spent work from the current call, never old cost.
        """
        if research_action not in {"program_replay", "structural_research"}:
            raise ValueError("Unsupported prepared native research action")
        if reused_paid_return and search_compute_cost:
            raise ValueError("An existing paid return has no new search cost")
        pool, new = list(prepared["assessed"]), list(new_claims)
        incumbent = self._best(pool)
        result = {"status": "unresolved", "execution_complete": False,
            "completion_scope": ("paid_native_return_and_full_pool_comparison" if reused_paid_return
                else "bounded_native_segment_and_full_pool_comparison"),
            "research_action": research_action, "claims": pool, "new_claims": new,
            "selected": incumbent, "full_comparison": True,
            "compute_cost": search_compute_cost, "search_compute_cost": search_compute_cost,
            "comparison_compute_cost": 0, "new_comparison_compute_cost": 0,
            "new_observations": 0, "fallback": prepared["fallback"],
            "independent_confirmation": False, "reused_paid_return": reused_paid_return,
            **(search_metadata or {})}
        if reused_paid_return:
            result["new_fit_compute_cost"] = 0
        result.update(
            structure_search_complete=[{"candidate_id": item["candidate"].candidate_id,
                "search_complete": item["structure"]["search_complete"]}
                for item in new if "search_complete" in (item.get("structure") or {})])
        if not prepared["comparison_complete"]:
            return {**result, "status": "preparation_comparison_unresolved",
                "claims": [*pool, *prepared["uncompared"], *new], "full_comparison": False}
        spec, comparison = prepared["spec"], prepared["comparison"]
        start = budget.spent
        scores = {item["candidate"].program: item["validation_loss"] for item in pool}
        for program, measured in (paid_scores or {}).items():
            scores.setdefault(program, measured)
        checked = []
        for item in new:
            program = item["candidate"].program
            if program not in scores:
                if not budget.spend(len(comparison)):
                    result.update(claims=[*pool, *new], full_comparison=False,
                        compute_cost=search_compute_cost + budget.spent - start,
                        comparison_compute_cost=budget.spent - start,
                        new_comparison_compute_cost=budget.spent - start)
                    return result
                scores[program] = loss(program, comparison, prepared["fallback"], spec.risk_quantile)
            checked.append({**item, "validation_loss": scores[program],
                "selection_basis": "unfitted_validation" if prepared["independent_validation"]
                    else "all_acquired_evidence"})
        pool.extend(checked)
        selected = self._best(pool)
        if incumbent and selected and incumbent["validation_loss"] <= (
                selected["validation_loss"] + self.config.minimum_improvement):
            selected = incumbent
        result.update(status="completed_reused_paid_return" if reused_paid_return else "completed",
            execution_complete=True, claims=pool, new_claims=checked, selected=selected,
            compute_cost=search_compute_cost + budget.spent - start,
            comparison_compute_cost=budget.spent - start,
            new_comparison_compute_cost=budget.spent - start)
        actions.append({"action": "consume_paid_prepared_research_return" if reused_paid_return
                else "complete_prepared_research_segment",
            "research_action": research_action, "old_candidates": len(prepared["assessed"]),
            "new_candidates": len(new), "full_comparison": True,
            "selected_candidate_id": selected["candidate"].candidate_id if selected else None,
            "compute_cost": result["compute_cost"], "new_observations": 0,
            "reused_paid_return": reused_paid_return})
        return result

    def _research_view(self, spec, training, validation, budget, actions, transfer,
                       graph, *, parent=None, transitions=(), observation_remaining=0,
                       measurement_remaining=None, include_risk_revisions=True, test_response_noise=None,
                       execution=None):
        before = budget.spent
        prepared = self.prepare_research_decision(spec, training, validation, budget, actions,
            parent=parent, transitions=transitions, observation_remaining=observation_remaining,
            measurement_remaining=measurement_remaining, execution=execution)
        if prepared["terminal"] or self._research_pool_explained(prepared):
            return prepared["assessed"]
        fallback = prepared["fallback"]
        independent_validation = prepared["independent_validation"]
        comparison = prepared["comparison"]
        assessed = prepared["assessed"]
        baseline = prepared["baseline"]
        call_origins = prepared["call_origins"]
        calls = prepared["calls"]
        relation_kernels = prepared["relation_kernels"]
        structured = prepared["structured"]
        usable_parent = prepared["usable_parent"]
        parent_fitted_comparison = prepared["parent_fitted_comparison"]
        retained_on_current = prepared["retained_on_current"]
        directions = prepared["directions"]
        state_identification = prepared["state_identification"]
        replay_plans = prepared["replay_plans"]
        experiment_compute = prepared["experiment_compute"]
        learn_policy = prepared["learn_policy"]
        policy = prepared["policy"]
        if structured:
            if _object_schema(spec) is not None:
                from .unified_structure import propose_structures, propose_response_revisions
            else:
                propose_structures = partial(self._object_explanations, actions=actions)

        def executed(name, cost):
            if execution is not None and cost:
                execution["search_compute_cost"] += cost
                execution["executed_directions"].append(name)
        def assess(claims, *, fitted_comparison=False):
            checked = self._assess(claims, comparison, budget, fallback, spec.risk_quantile)
            if self._workflow_episode is not None:
                self._workflow_episode.fitted(checked, training, comparison, spec)
            if execution is not None:
                execution["compared_candidates"] += sum(parent is None or
                    item["candidate"] is not parent["candidate"] for item in checked)
            if fitted_comparison or not independent_validation:
                for claim in checked:
                    claim["selection_basis"] = "all_acquired_evidence"
            return checked

        leading, explained_families, response_parent_for_revision = None, set(), None
        replayed = None
        while directions:
            from .research_programs import coordinate_search_allowance
            from .conditional_coordinates import conditional_search_allowance
            conditional_replay = any(any(step["operation"] in ("discover_conditional_coordinates", "discover_composed_conditions")
                for step in payload.get("selection_steps", []))
                for _, payload, _, _ in replay_plans)
            entry_cost = len(training) * (len(spec.feature_names) + 2) + len(comparison)
            callable_fit_cost = len(training) * (len(spec.feature_names) + 4)
            search_remaining = max(0, budget.remaining - experiment_compute)
            search_limits = {
                "callable_transfer": min(16000,
                    max(budget.remaining // 2, callable_fit_cost),
                    len(calls) * callable_fit_cost),
                "coordinate_reconstruction": min(32000, budget.remaining // 2,
                    coordinate_search_allowance(len(spec.feature_names), len(training),
                        max((len(row.history) for row in training), default=0))),
                "conditional_reconstruction": min(64000, budget.remaining // 2,
                    conditional_search_allowance(len(spec.feature_names), len(training),
                        max((len(row.history) for row in training), default=0))),
                "program_replay": min(64000 if conditional_replay else 4000, budget.remaining // 2),
                "mechanism_revision": min(16000, budget.remaining // 2),
                "program_revision": min(32000, budget.remaining // 2),
            }
            if experiment_compute:
                search_limits = {name: min(allowance, max(0,
                    search_remaining - len(comparison) * self.config.max_candidates))
                    for name, allowance in search_limits.items()}
            if state_identification:
                search_limits["state_identification"] = state_identification
            quotes = {
                "dependent_residual_revision": len(training) * (len(_terms(spec.feature_names, 1, True)) + 6)
                    + len(comparison),
                **{name: allowance + len(comparison) * (
                    min(4, self.config.max_candidates) if name == "program_replay" else self.config.max_candidates)
                    for name, allowance in search_limits.items()},
            }
            quotes["callable_transfer"] = search_limits["callable_transfer"] + len(comparison) * min(
                len(calls), search_limits["callable_transfer"] // max(1, callable_fit_cost))
            if state_identification:
                quotes["state_identification"] = state_identification + 6 * (len(training) + len(comparison))
            execution_caps = {name: quotes[name] for name in search_limits}
            # A new route quotes its funded trial. Once measured, its expected
            # price comes from actual work; the execution cap remains separate.
            for name in search_limits:
                measured = policy._estimated_costs(policy.action_record(name))["compute_cost"]
                if measured > 0 and name != "state_identification":
                    quotes[name] = max(len(training) + len(comparison), ceil(measured))
            limits = {"dependent_residual_revision": 24000}
            options = [ActionOption(name, 0, quotes.get(name, entry_cost),
                payload=(None, {"compute_limit": execution_caps[name]})
                    if name in search_limits and name != "program_revision" else None)
                for name in directions
                if quotes.get(name, entry_cost) - len(comparison) <= limits.get(name, search_remaining)]
            if learn_policy:
                selected_action = policy.choose(options, observation_remaining=observation_remaining,
                    measurement_remaining=measurement_remaining, compute_remaining=search_remaining,
                    minimum_improvement=self.config.minimum_improvement)
                decision = dict(policy.last_decision)
                if selected_action is None:
                    # A changed task can reopen every still executable direction.
                    selected_action = next((option for option in options
                        if option.compute_cost <= search_remaining), None)
                    if selected_action is not None:
                        decision.update(selected_action=selected_action.name,
                            reason="retry_remaining_direction_on_current_evidence")
                actions.append({"action": "choose_research_action", **decision})
            else:
                selected_action = next((option for option in options
                    if option.compute_cost <= search_remaining), None)
            if selected_action is None:
                break
            direction = selected_action.name
            directions.remove(direction)
            alternative = self._best(assessed)
            action_start = budget.spent
            if direction == "state_identification":
                local = SearchBudget(state_identification)
                states = continuous_state_candidates(training, budget=local)
                budget.spend(local.spent)
                claims = []
                for candidate in states:
                    if budget.spend(len(training)):
                        claims.append({"candidate": candidate, "kind": "continuous_latent_state",
                            "family_id": "T08", "training_loss": loss(candidate.program,
                                training, fallback, spec.risk_quantile)})
                directional_claims = assess(claims)
                actions.append({"action": "identify_state_before_expansion",
                    "compute_allowance": state_identification,
                    "compute_cost": budget.spent - action_start,
                    "candidate_count": len(directional_claims),
                    "validation_losses": [item["validation_loss"] for item in directional_claims]})
            elif direction == "callable_transfer":
                local = SearchBudget(search_limits[direction])
                reused = []
                for call in calls:
                    started = local.spent
                    claim = _fit([call, Program("mul", children=(call, call)), *_terms(spec.feature_names)],
                        training, local, "callable_transfer", family="T03", q=spec.risk_quantile)
                    if claim is None:
                        continue
                    self._callable_fit_serial += 1
                    # The actual recalled definition can differ from the
                    # successor offer that opened this visit. Record its bound
                    # template only when the fitted computation still calls it.
                    retained_call = call.simplified()
                    def contains_call(node):
                        return node == retained_call or any(contains_call(child) for child in node.children)
                    if call in call_origins and contains_call(claim["candidate"].program):
                        origin = deepcopy(call_origins[call])
                        origin["adaptation_work"].append({
                            "work_id": f"callable:{spec.task_id}:{self._callable_fit_serial}",
                            "compute_cost": local.spent - started,
                            "source_ids": [origin["source_id"]],
                            "template_ids": [origin["template_id"]], "preserved_work": []})
                        claim["transfer_evidence"] = [origin]
                    reused.append(claim)
                budget.spend(local.spent)
                directional_claims = assess(reused)
                for item in directional_claims:
                    for origin in item.get("transfer_evidence", []):
                        origin["adaptation_work"][-1]["compute_cost"] += len(comparison)
                    transfer.append({"source_ids": _sources(item["candidate"].program),
                        "program": item["candidate"].program.to_dict(), "validation_loss": item["validation_loss"],
                        "status": "supported" if independent_validation and item["validation_loss"] <=
                            self.config.minimum_improvement else "counterexample" if independent_validation else "pending",
                        "validation_rows": len(validation), "accepted": False})
            elif direction == "mechanism_revision":
                from .mechanism_revision import propose_mechanism_revisions
                local = SearchBudget(search_limits[direction])
                seeds = ([parent] if usable_parent else []) + [
                    {"candidate": Candidate(f"mechanism_source:{index}", call,
                        "retained_mechanism", parents=tuple(_sources(call))),
                     "method_source_ids": _sources(call)} for index, call in enumerate(calls)]
                revisions, seen = [], set()
                for seed in seeds:
                    program = seed["candidate"].program
                    if program in seen or len(revisions) >= self.config.max_candidates:
                        continue
                    seen.add(program)
                    revisions.extend(propose_mechanism_revisions(spec, training, local,
                        parent=seed, priors=self.library.meta_priors,
                        max_candidates=self.config.max_candidates - len(revisions)))
                budget.spend(local.spent)
                # Equal executable fits can have two derivations. Retain the
                # explicit deletion as the question to test against its source.
                revisions.sort(key=lambda item: item["kind"] != "mechanism_deletion")
                directional_claims = assess(revisions, fitted_comparison=parent_fitted_comparison)
                actions.append({"action": "propose_mechanism_revisions", "compute_cost": local.spent,
                    "alternatives": [{"kind": item["kind"],
                        "candidate_id": item["candidate"].candidate_id,
                        "training_loss": item["training_loss"], "validation_loss": item["validation_loss"],
                        "revision": item["mechanism_revision"]} for item in directional_claims]})
            elif direction == "coordinate_reconstruction":
                local = SearchBudget(search_limits[direction])
                claims = self._coordinate_claims(spec, training, local, actions)
                budget.spend(local.spent)
                directional_claims = assess(claims)
            elif direction == "conditional_reconstruction":
                local = SearchBudget(search_limits[direction])
                claims = self._conditional_claims(spec, training, local, actions)
                budget.spend(local.spent)
                directional_claims = assess(claims)
            elif direction == "program_replay":
                replayed, local = self._execute_prepared_research_segment(prepared,
                    direction, budget, actions, search_limit=search_limits[direction])
                directional_claims = assess(replayed)
                actions.append({"action": "replay_research_programs", "compute_cost": local.spent,
                    "alternatives": [{"training_loss": item["training_loss"],
                        "validation_loss": item["validation_loss"], "source_ids": item["method_source_ids"]}
                        for item in directional_claims]})
            elif direction == "program_revision":
                local = SearchBudget(search_limits[direction])
                programs = self._program_claims(spec, training, local, calls, actions,
                    relation_kernels=relation_kernels, replayed=[] if replayed is not None else None)
                budget.spend(local.spent)
                directional_claims = assess(programs)
                for claim in directional_claims:
                    identifier = claim.get("research_draft_id")
                    if identifier is None:
                        continue
                    draft = next(item for item in self.library.research_drafts if item["draft_id"] == identifier)
                    outcome = {"task_id": spec.task_id, "training_loss": claim["training_loss"],
                        "validation_loss": claim["validation_loss"], "rows": len(comparison),
                        "status": ("supported_on_current_validation" if independent_validation
                            and claim["validation_loss"] <= self.config.minimum_improvement else
                            "current_counterexample" if independent_validation else "pending")}
                    draft["outcomes"].append(outcome)
                    actions.append({"action": "assess_research_draft", "draft_id": identifier, **outcome})
            elif direction == "dependent_residual_revision":
                local = SearchBudget(min(24000, max(0, search_remaining - len(comparison)), max(budget.remaining // 2,
                    selected_action.compute_cost - len(comparison))))
                revised = self._dependent_residual_claims(spec, training, local, parent, graph, actions)
                budget.spend(local.spent)
                directional_claims = assess(revised, fitted_comparison=parent_fitted_comparison)
            elif direction == "response_basis_revision":
                local = SearchBudget(min(32000, budget.remaining // 2))
                revised = propose_response_revisions(spec, training, local,
                    parent=response_parent_for_revision,
                    tolerance=self.config.minimum_improvement)
                budget.spend(local.spent)
                directional_claims = assess(revised)
                leading = self._best(directional_claims) or leading
                explained_families.update(item["kind"] for item in directional_claims
                    if item["training_loss"] <= self.config.minimum_improvement)
                actions.append({"action": "compare_response_basis_revisions",
                    "compute_cost": local.spent,
                    "alternatives": [{"kind": item["kind"], "training_loss": item["training_loss"],
                        "validation_loss": item["validation_loss"]} for item in directional_claims]})
            else:
                structural_allowance = (max(0, budget.remaining - 16 * len(comparison))
                    if spec.context.get("object_model") and not _object_schema(spec)
                    else budget.remaining // 2)
                structures, local = self._execute_prepared_research_segment(prepared,
                    direction, budget, actions, search_limit=min(90000, structural_allowance),
                    assessed=assessed)
                directional_claims = assess(structures)
                leading = self._best(directional_claims)
                explained_families = {item["kind"] for item in directional_claims
                    if item["training_loss"] <= self.config.minimum_improvement}
                if (_object_schema(spec) is not None and leading and not explained_families):
                    # Finish the existing relation question first. Only an
                    # unexplained measured response opens this different
                    # question against the shared remaining research budget.
                    response_parent_for_revision = leading
                    directions.append("response_basis_revision")
                actions.append({"action": "compare_structural_explanations",
                    "features": list(spec.feature_names), "compute_cost": local.spent,
                    "alternatives": [{"kind": item["kind"], "training_loss": item["training_loss"],
                                      "validation_loss": item["validation_loss"]}
                                     for item in directional_claims]})
                if any(item.get("object_equivalence") for item in directional_claims):
                    actions.append({"action": "compare_equivalent_object_programs",
                        "alternatives": [{"schemas": len(item["object_equivalence"]["schemas"]),
                            "validation_loss": item["validation_loss"], "kind": item["kind"]}
                            for item in directional_claims if item.get("object_equivalence")]})
            executed(direction, local.spent)
            assessed.extend(directional_claims)
            proposed = self._best(directional_claims)
            if learn_policy and alternative is not None:
                paid_compute = budget.spent - action_start
                before_loss = alternative["validation_loss"]
                fitted_reference = alternative.get("selection_basis") == "all_acquired_evidence"
                action_context = policy.action_context(direction)
                origin = {"context": action_context, "action": direction, "trial": None,
                    "baseline_program": alternative["candidate"].program.to_dict(),
                    "fallback": fallback, "risk_quantile": spec.risk_quantile}
                if (independent_validation and not fitted_reference
                        and not (direction in {"dependent_residual_revision", "mechanism_revision"}
                                 and parent_fitted_comparison)):
                    # Reward the fraction of the remaining error repaired,
                    # not repeated improvements over an obsolete affine fit.
                    after_loss = proposed["validation_loss"] if proposed is not None else before_loss
                    feedback = policy.update(direction, before_loss=before_loss, after_loss=after_loss,
                        observation_cost=0, compute_cost=paid_compute)
                    origin.update(trial=feedback["trials"], credited_gain=feedback["last_gain"])
                    actions.append({"action": "research_action_feedback", "research_action": direction,
                        "context": action_context, "before_loss": before_loss, "after_loss": after_loss,
                        "baseline_kind": alternative["kind"], "baseline_program": origin["baseline_program"],
                        "observation_cost": 0, "measurement_cost": 0, "compute_cost": paid_compute,
                        "evidence_rows": len(validation), "fitting_rows": len(training),
                        "evidence_basis": "common_unfitted_validation", "feedback": feedback})
                else:
                    origin.update(pending_id=len(actions), pending_status="unresolved",
                        pending_costs={"observation_cost": 0, "measurement_cost": 0, "compute_cost": paid_compute})
                    actions.append({"action": "research_action_pending", "research_action": direction,
                        "context": action_context, "pending_id": origin["pending_id"], "trial": None,
                        "evidence_basis": "reference_fitted_comparison" if fitted_reference else
                            "parent_fitted_comparison" if parent_fitted_comparison and
                            direction in {"dependent_residual_revision", "mechanism_revision"}
                            else "no_independent_validation",
                        **origin["pending_costs"]})
                for claim in directional_claims:
                    claim["research_action_origin"] = origin
            best = self._best(assessed)
            if best and best["validation_loss"] <= self.config.minimum_improvement:
                return assessed
            # Test the cheaper observation-variation hypothesis before expanding
            # another search language. This only pauses candidate generation;
            # adoption still requires a separate frozen-program confirmation.
            if (test_response_noise is not None and independent_validation and best is proposed
                    and best.get("selection_basis") != "all_acquired_evidence"
                    and alternative is not None and best["validation_loss"]
                        + self.config.minimum_improvement < alternative["validation_loss"]):
                repeat_start = len(actions)
                noise_supported = test_response_noise(best, validation)
                observation_remaining -= sum(action["observation_cost"]
                    for action in actions[repeat_start:] if action["action"] == "acquire")
                if noise_supported:
                    actions.append({"action": "defer_research_expansion_after_repeat",
                        "selected_kind": best["kind"], "remaining_directions": list(directions),
                        "validation_loss": best["validation_loss"],
                        "reason": "paid_repeat_explains_remaining_error"})
                    return assessed
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
            executed("combined_evidence_revision", revision_budget.spent)
            consistent = [{**item, "validation_loss": item["training_loss"],
                           "selection_basis": "all_acquired_evidence"} for item in revised
                          if item["training_loss"] <= self.config.minimum_improvement]
            if leading.get("research_action_origin"):
                origin = leading["research_action_origin"]
                for claim in consistent:
                    claim["research_action_origin"] = origin
                self._charge_action_costs(origin, {"observation_cost": 0, "measurement_cost": 0,
                    "compute_cost": revision_budget.spent}, actions, "combined_evidence_revision")
            assessed.extend(consistent)
            actions.append({"action": "revise_structure_from_combined_evidence",
                "family": leading["kind"], "rows": len(training) + len(validation),
                "compute_cost": revision_budget.spent, "answered": bool(consistent),
                "selection_basis": "all_acquired_evidence"})
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
        executed("compile_claims", local.spent)
        assessed.extend(assess(claims))
        joint = []
        fit_start = budget.spent
        for depth in range(1, self.config.max_history + 1):
            for quadratic in (False, True):
                if quadratic and not self.config.joint_state_coordinates:
                    continue
                joint.append(_fit(_terms(spec.feature_names, depth, quadratic), training, budget,
                    "joint_quadratic_history" if quadratic else "reset_history",
                    family="T08", q=spec.risk_quantile))
        executed("joint_history", budget.spent - fit_start)
        assessed.extend(assess(joint))

        best = self._best(assessed)
        if (self.config.state_search and best is not None
                and best["validation_loss"] > self.config.minimum_improvement):
            local = SearchBudget(min(48000, budget.remaining // 2))
            states = continuous_state_candidates(training, budget=local,
                                                   seed_programs=calls)
            budget.spend(local.spent)
            executed("continuous_latent_state", local.spent)
            state_claims = []
            for candidate in states:
                if budget.spend(len(training)):
                    state_claims.append({"candidate": candidate, "kind": "continuous_latent_state",
                        "family_id": "T08", "training_loss": loss(candidate.program,
                            training, fallback, spec.risk_quantile)})
            assessed.extend(assess(state_claims))

        fit_start = budget.spent
        risks = self._risk_revisions(spec, assessed, training, budget) if include_risk_revisions else []
        executed("risk_revision", budget.spent - fit_start)
        assessed.extend(assess(risks))
        best = self._best(assessed)
        actions.append({"action": "compare_research_directions", "features": list(spec.feature_names),
            "alternatives": [{"kind": item["kind"], "family_id": item["family_id"],
                               "validation_loss": item["validation_loss"],
                               "history_depth": item["candidate"].program.history_depth}
                              for item in assessed],
            "selected_kind": best["kind"] if best else None,
            "compute_cost": budget.spent - before})
        if _object_schema(spec) or spec.context.get("object_model"):
            from .object_continuation import encode_entry
            retained = [entry for entry in assessed if entry.get("structure") or entry.get("research_program")
                        or entry.get("object_competition")]
            if retained:
                self._object_candidate_frontiers[tuple(spec.feature_names)] = [encode_entry(entry)
                    for entry in retained[:self.config.max_candidates]]
        return assessed

    def run(self, laboratory, *, previous=None, continuation_candidate=None):
        self._workflow_episode = None
        if previous is not None and (continuation_candidate or {}).get("response_question"):
            from .response_question_continuation import run_response_question
            return run_response_question(self, laboratory, previous, continuation_candidate)
        object_continuation = previous is not None and bool(getattr(laboratory, "measurement_views", None))
        if previous is not None and not object_continuation:
            from .unified_continuation import run_continuation
            return run_continuation(self, laboratory, previous, continuation_candidate)
        if object_continuation and previous["status"] != "unresolved":
            raise ValueError("object continuation currently resumes unresolved research")
        self._object_schemas = {}
        self._object_models = {}
        self._object_competitions = {}
        self._object_candidate_frontiers = {}
        self._learn_instruments = self._instrument_knowledge(laboratory)
        self._revise_inputs = self._revisable_inputs(laboratory)
        old_costs = dict(previous["costs"]) if object_continuation else {
            "observation_cost": 0, "measurement_cost": 0, "compute_cost": 0, "queries": 0}
        if (self.config.observation_budget < old_costs["observation_cost"]
                or self.measurement_budget < old_costs["measurement_cost"]
                or self.config.compute_budget < old_costs["compute_cost"]):
            raise ValueError("continuation limits cannot be smaller than already purchased work")
        # Keep a small amount of the task cap outside the search while it is
        # still possible to need a fresh frozen-program comparison.  Without
        # this, broad fitting can spend the last units before the comparison
        # gate and leave a real selected candidate unconfirmed.  The reserve
        # is released at that gate; fixed and no-memory controls retain their
        # original budget semantics.
        confirmation_reserve = 0
        if self.config.use_memory and self.config.strategy != "fixed":
            confirmation_length = max(1, min(24, laboratory.spec.max_probe_length,
                                              self.config.observation_budget // 8))
            confirmation_reserve = 2 * confirmation_length
        budget = SearchBudget(self.config.compute_budget, old_costs["compute_cost"],
                              reserve=confirmation_reserve)
        observation_limit = self.config.observation_budget
        measurement_limit = self.measurement_budget
        actions, transfer, batches = (deepcopy(previous["actions"]), deepcopy(previous.get("transfer", [])),
                                     deepcopy(previous["batches"])) if object_continuation else ([], [], [])
        command_observations = {}
        observation_probes, noise_replications = {}, {}
        graph = ResearchGraph.from_dict(previous.get("research_graph")) if object_continuation else ResearchGraph()
        rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        alternative_views = getattr(laboratory, "measurement_views", None)
        offered = tuple(alternative_views[-1]) if alternative_views else tuple(rates)
        restored_frontier = {}
        if object_continuation:
            from .object_continuation import restore_object_state
            restored_frontier = restore_object_state(self, previous, offered)
            self._learn_instruments = self._instrument_knowledge(laboratory)
            self._revise_inputs = self._revisable_inputs(laboratory)
        batch_observations = {}
        workflow = None
        if (self.config.use_memory and self.config.strategy != "fixed"
                and self.config.learn_research_policy):
            from .workflow_execution import WorkflowEpisode
            workflow = self._workflow_episode = WorkflowEpisode(
                self.library.research_workflows, self._spec(laboratory, ()),
                laboratory.channels, attempt=self.config.seed)

        def restore_batch(index):
            if index not in batch_observations:
                batch = batches[index]
                probe = Probe(batch["probe"]["probe_id"], tuple(batch["probe"]["sequence"]))
                history, restored = [], []
                for point, payload in zip(probe.sequence, batch["observations"]):
                    row = Observation(dict(payload["features"]), payload["target"],
                        tuple(dict(past) for past in payload["history"]))
                    command = Observation(dict(point), row.target, tuple(history))
                    command_observations[id(row)] = command
                    observation_probes[id(row)] = observation_probes[id(command)] = probe
                    history.append({**point, "__target__": row.target})
                    restored.append(row)
                batch_observations[index] = restored
                if workflow is not None:
                    workflow.acquired(probe, tuple(batch["channels"]), restored,
                        purpose="restore_paid_evidence",
                        command_rows=[command_observations[id(row)] for row in restored])
            return batch_observations[index]

        def views_from(channels):
            if alternative_views:
                return list(laboratory.projectable_views(channels))
            return [subset for width in range(1, len(channels) + 1)
                    for subset in combinations(channels, width)] if channels else [()]

        def view_rows(rows, source, target):
            if source and not target:
                return [command_observations[id(row)] for row in rows]
            projected = (laboratory.project_rows(rows, source, target) if alternative_views else
                         _view(rows, self._spec(laboratory, target).feature_names))
            for original, row in zip(rows, projected):
                if id(original) in command_observations:
                    command_observations[id(row)] = command_observations[id(original)]
                if id(original) in observation_probes:
                    observation_probes[id(row)] = observation_probes[id(original)]
            if workflow is not None:
                workflow.projected(rows, projected)
            return projected

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

        def challenge_noise(support, program, validation, source, evidence_channels, fallback,
                            *, reserve_rows=None):
            uncertain_shift = (laboratory.spec.risk_quantile == .5
                               and support["reason"] == "residual_loss_shift")
            if not uncertain_shift and (not support["adequate"] or support["reason"] != "noise_compatible"):
                return support
            if risk_actions is not None:
                from .risk_experiments import distinguish_risk_explanations
                reserve = length if reserve_rows is None else reserve_rows
                research = distinguish_risk_explanations(support,
                    spec=self._spec(laboratory, evidence_channels), acquire=acquire,
                    channels=evidence_channels, budget=budget,
                    remaining=lambda: {"observation_remaining": observation_limit - spent_observations,
                        "measurement_remaining": measurement_limit - spent_measurements,
                        "compute_remaining": budget.remaining},
                    minimum_improvement=self.config.minimum_improvement,
                    measurement_rate=sum(rates[name] for name in evidence_channels),
                    reserve_rows=reserve, reserve_compute=support["training_rows"] * 6 + reserve * 16)
                actions.extend(research["events"])
                support = research["support"]
                if research["observation_cost"]:
                    # These acquired full-coordinate responses already test
                    # the competing explanations. A blind repeat would not
                    # match independently drawn instrument histories.
                    return support
            from .noise_replication import assess_noise_replication
            probe = observation_probes.get(id(validation[0])) if validation else None
            if probe is None:
                return {**support, "adequate": False, "reason": "noise_replication_inconclusive"}
            # A previous confirmation can already be part of a revised fit.
            # Its repeat cannot explain the error on this new trajectory.
            key = (tuple(evidence_channels), probe.probe_id)
            required_rows = len(noise_replications[key][0]) if key in noise_replications else len(validation)
            repeat_cost = 4 if laboratory.spec.risk_quantile == .5 else 3
            if budget.remaining < repeat_cost * required_rows:
                return {**support, "adequate": False, "reason": "noise_experiment_unfunded"}
            if key not in noise_replications:
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
            if evidence["repeatable_error"]:
                support = {**support, "adequate": False, "reason": "repeatable_unexplained_response"}
            elif not evidence["noise_supported"]:
                support = {**support, "adequate": False,
                    "reason": "noise_experiment_unfunded" if evidence["reason"] == "insufficient_budget"
                        else "noise_replication_inconclusive"}
            elif uncertain_shift and max(evidence["original_model_loss"],
                    evidence["repeated_model_loss"]) <= evidence["repeatable_loss_limit"]:
                # Fitting can understate the observation variance. The two
                # independent responses estimate it without the fitted mean;
                # both frozen-program errors must fit that measured allowance.
                # The original adequacy search already ruled out its supported
                # residual alternatives before reporting only a scale shift.
                actions.append({"action": "revise_noise_scale_from_repeat",
                    "training_limit": support["noise_loss_limit"],
                    "measured_limit": evidence["repeatable_loss_limit"],
                    "original_loss": evidence["original_model_loss"],
                    "repeated_loss": evidence["repeated_model_loss"]})
                support = {**support, "adequate": True,
                    "reason": "replicated_noise_explains_loss_shift",
                    "repeat_noise_loss_limit": evidence["repeatable_loss_limit"]}
            return {**support, "replication": evidence}

        def test_response_noise(entry, validation):
            """Pay for a repeat before buying additional ordinary response search."""
            spec = laboratory.spec
            if (offered or alternative_views or spec.risk_quantile != .5
                    or spec.decision_costs is not None or not self.config.risk_learning
                    or _object_schema(spec) or not validation):
                return False
            probe = observation_probes.get(id(validation[0]))
            if probe is None:
                return False
            key = ((), probe.probe_id)
            required = 0 if key in noise_replications else len(validation)
            # A full repeat can spend the rows needed to distinguish hidden
            # interacting conditions. Prefer the funded matched experiment;
            # noise still faces fresh confirmation after that response fit.
            available = observation_limit - spent_observations
            composed_rows = (2 * (len(spec.feature_names) + 1) + 4) * min(12, spec.max_probe_length) + 28
            if (len(spec.feature_names) >= 4 and set(spec.feature_names) <= set(spec.feature_bounds)
                    and available >= composed_rows + length
                    and available < required + composed_rows + 2 * length):
                return False
            # Leave an untouched confirmation and its own repeat affordable.
            if (observation_limit - spent_observations < required + 2 * length
                    or budget.remaining < 4 * len(validation) + 32 * length):
                return False
            before = (spent_observations, spent_measurements, budget.spent)
            if key not in noise_replications:
                repeated = acquire((), "test_response_variation_before_search",
                    probe=Probe(f"repeat_search_{serial}", probe.sequence))
                if repeated is None:
                    return False
                noise_replications[key] = (validation, repeated)
            from .noise_replication import assess_noise_replication
            original, repeated = noise_replications[key]
            evidence = assess_noise_replication(entry["candidate"].program,
                original, repeated, budget, minimum_improvement=self.config.minimum_improvement,
                fallback=initial_prediction(original, .5))
            supported = (evidence["noise_supported"] and max(evidence["original_model_loss"],
                evidence["repeated_model_loss"]) <= evidence["repeatable_loss_limit"])
            # This paid explanation belongs to this predictor and evidence,
            # not to every later model or observation in the task.
            entry["response_noise_evidence"] = {
                "program": entry["candidate"].program.to_dict(), "channels": [],
                "probe_id": probe.probe_id, "paid_observation_rows": spent_observations,
                "supported": supported, "replication": evidence}
            actions.append({"action": "test_response_noise_before_expansion",
                "candidate_id": entry["candidate"].candidate_id,
                "program": entry["candidate"].program.to_dict(), "probe_id": probe.probe_id,
                "supported": supported, **evidence})
            self._charge_action_costs(entry.get("research_action_origin"),
                {"observation_cost": spent_observations - before[0],
                 "measurement_cost": spent_measurements - before[1],
                 "compute_cost": budget.spent - before[2]}, actions, "noise_test_before_expansion")
            return supported

        def reassess_noisy_counterexamples(program, counterexamples, fallback):
            """Repeat the actual disputed trajectories after noise is established."""
            grouped = {}
            for reason, rows, channels in counterexamples:
                for row in rows:
                    probe = observation_probes.get(id(row))
                    if probe is None:
                        return False
                    key = (tuple(channels), probe.probe_id)
                    group = grouped.setdefault(key, {"rows": [], "reasons": []})
                    if row not in group["rows"]:
                        group["rows"].append(row)
                    if reason not in group["reasons"]:
                        group["reasons"].append(reason)
            supported = True
            for (channels, probe_id), group in grouped.items():
                evidence = challenge_noise({"adequate": True, "reason": "noise_compatible"},
                    program, group["rows"], channels, channels, fallback)
                supported &= evidence["adequate"]
                actions.append({"action": "reassess_noisy_counterexample",
                    "reasons": group["reasons"], "probe_id": probe_id,
                    "adequate": evidence["adequate"], "replication": evidence.get("replication")})
            return supported

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
            for index, batch in enumerate(batches):
                if tuple(channels) not in views_from(tuple(batch["channels"])):
                    continue
                measured = view_rows(restore_batch(index), tuple(batch["channels"]), channels)
                history, pairs = [], []
                for point, row in zip(batch["probe"]["sequence"], measured):
                    pairs.append((Observation(dict(point), row.target, tuple(history)), row))
                    history.append({**point, "__target__": row.target})
                if workflow is not None:
                    workflow.projected(measured, [command for command, _ in pairs])
                result.append(pairs)
            return result

        def channels_for(required):
            if alternative_views:
                return next((tuple(view) for view in alternative_views
                    if required <= set(self._spec(laboratory, view).feature_names)), offered)
            return tuple(name for name in offered if name in required)
        spent_observations, spent_measurements, serial = (old_costs["observation_cost"],
            old_costs["measurement_cost"], len(batches))
        from .object_continuation import observations as restore_observations
        directed_command_rows = restore_observations(restored_frontier.get("directed_command_rows", []))
        length = min(24, laboratory.spec.max_probe_length,
                     max(1, observation_limit // 8))

        def acquire(channels, purpose, candidates=(), *, rows_count=None, probe=None, questions=()):
            nonlocal spent_observations, spent_measurements, serial
            count = probe.cost if probe is not None else length if rows_count is None else rows_count
            rate = sum(rates[name] for name in channels)
            if (spent_observations + count > observation_limit
                    or spent_measurements + count * rate > measurement_limit):
                return None
            # Endpoint history interventions can make every event response
            # zero. A later adoption comparison must also cover varied setups;
            # targeted structural interventions are acquired explicitly below.
            comparison = purpose in {"compare_on_new_trajectory", "validate_short_learned_trial",
                                     "confirm_short_frozen_program", "confirm_frozen_program",
                                     "confirm_instrument_replacement"}
            program_plan = None
            if (probe is None and not comparison and candidates and self.config.active_probes
                    and self.config.strategy != "fixed" and purpose != "distinguish_risk_explanations"):
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
                            spent_observations + probe.cost > observation_limit
                            or spent_measurements + probe.cost * rate > measurement_limit):
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
            batch_observations[len(batches) - 1] = batch.observations
            if workflow is not None:
                workflow.acquired(probe, channels, batch.observations, purpose=purpose,
                    command_rows=[command_observations[id(row)] for row in batch.observations])
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

        def distinguish_instruments(model, channels, *, reserve_rows=0, required_fields=None):
            affordable = observation_limit - spent_observations
            rate = sum(rates[name] for name in channels)
            if rate:
                affordable = min(affordable, (measurement_limit - spent_measurements) // rate)
            affordable = max(0, affordable - reserve_rows)
            return self._distinguish_instruments(laboratory, model, channels, acquire, paired_for,
                budget, actions, max_length=min(laboratory.spec.max_probe_length, affordable),
                observation_allowance=affordable, required_fields=required_fields)

        def research_instruments(channels, *, reserve_rows=0, required_fields=None):
            model = self._fit_instruments(self._spec(laboratory, ()), self._spec(laboratory, channels),
                paired_for(channels), budget, actions, required_fields=required_fields)
            model = distinguish_instruments(model, channels, reserve_rows=reserve_rows,
                                           required_fields=required_fields)
            if model is not None and object_research and tuple(channels) == offered:
                object_research["sensor_model"] = deepcopy(model)
                saved = next((item for item in self.library.object_models
                              if item["model_id"] == object_research["model_id"]), None)
                if saved is not None:
                    saved["sensor_model"] = deepcopy(model)
            return model

        object_research = deepcopy(previous.get("object_research")) if object_continuation else None
        object_representation_changed = False
        if self._object_induction(laboratory) and not (
                object_continuation and object_research and object_research.get("complete")):
            remaining = observation_limit - spent_observations
            rate = sum(rates[name] for name in offered)
            if rate:
                remaining = min(remaining, (measurement_limit - spent_measurements) // rate)
            object_research = self._research_objects(laboratory, offered, acquire, batches, budget, actions,
                previous_model=object_research if object_continuation else None,
                remaining_rows=remaining,
                source_object_model_ids=(continuation_candidate or {}).get("source_object_model_ids", ()))
            self._learn_instruments = self._instrument_knowledge(laboratory)
            self._revise_inputs = self._revisable_inputs(laboratory)
            previous_objects = (previous or {}).get("object_research") or {}
            object_evidence_changed = any(object_research.get(key) != previous_objects.get(key)
                for key in ("model_id", "support_rows", "acquisition_rows", "groups"))
            object_representation_changed = any(object_research.get(key) != previous_objects.get(key)
                for key in ("groups", "schema", "focus_control")) or (
                self._object_schemas.get(offered) != (previous or {}).get("learned_object_schema"))
            if object_research.get("groups") and (not object_continuation or object_evidence_changed):
                # The observed representation can already explain responses.
                # Learn a command substitute when input repair or deployment
                # actually requests one, after funding the response question.
                actions.append({"action": "defer_object_instrument_mapping",
                    "object_model_id": object_research["model_id"],
                    "reason": "paid_measurements_available_for_response_research",
                    "compute_remaining": budget.remaining})

        selected_channels, finalists, parent, selected = (), [], None, None
        selected_training, selected_validation = [], []
        inherited_relation_evidence = restore_observations(restored_frontier.get("inherited_relation_evidence", []))
        object_refutations = deepcopy(restored_frontier.get("refutations", []))
        confirmation_loss, confirmed = None, False
        confirmation_rows, noisy_support = 0, None
        response_research_after_inputs = False
        observation_alternatives = []
        method_trial = False
        instrument_views_tested = not offered
        # These interfaces expose ordinary numeric fields. Object grouping and
        # risk questions retain their own joint inference paths.
        numeric_observation_research = (self._learn_instruments and bool(offered)
            and not alternative_views and laboratory.spec.risk_quantile == .5
            and not _object_schema(self._spec(laboratory, offered)))
        object_observation_research = (self._revise_inputs and bool(alternative_views)
            and laboratory.spec.risk_quantile == .5
            and bool(_object_schema(self._spec(laboratory, offered))))

        def current_command_batches():
            return [[command_observations[id(row)] for row in restore_batch(index)]
                    for index in range(len(batches))]

        def recover_observation_actions(current=None, failed_probe=None):
            if not numeric_observation_research:
                return None
            from .observation_research import research_observation_actions
            return research_observation_actions(self, laboratory, acquire=acquire, paired_for=paired_for,
                command_batches=current_command_batches, budget=budget, actions=actions,
                transfer=transfer, graph=graph, length=length, parent=current, failed_probe=failed_probe,
                alternatives=observation_alternatives,
                remaining=lambda: {"observation_remaining": observation_limit - spent_observations,
                    "measurement_remaining": measurement_limit - spent_measurements,
                    "compute_remaining": budget.remaining})

        def adopt_observation_recovery(result):
            nonlocal selected, selected_channels, selected_training, selected_validation, finalists
            nonlocal confirmed, confirmation_loss, confirmation_rows, method_trial, instrument_views_tested
            nonlocal response_research_after_inputs
            inherited_relation_evidence.extend((result or {}).get("counterexamples", ()))
            if result and result.get("alternatives"):
                observation_alternatives[:] = deepcopy(result["alternatives"])
            if not result or result.get("selected") is None:
                return False
            selected, selected_channels = result["selected"], tuple(result["channels"])
            response_research_after_inputs = selected.pop("input_recovery_response_status", None) == "counterexample"
            instrument_views_tested |= selected_channels == offered
            selected_training, selected_validation = result["training"], result["validation"]
            finalists = [selected]
            confirmed = result["confirmed"]
            confirmation_loss = result.get("confirmation_loss")
            confirmation_rows = result.get("confirmation_rows", 0)
            method_trial = True
            return True

        object_actions = None
        if object_observation_research:
            from .object_research_actions import ObjectResearchActions
            object_actions = ObjectResearchActions(self, laboratory, acquire=acquire,
                paired_for=paired_for, command_batches=current_command_batches,
                view_rows=view_rows, transitions_for=transitions_for, budget=budget,
                actions=actions, transfer=transfer, graph=graph, length=length,
                remaining=lambda: {"observation_remaining": observation_limit - spent_observations,
                    "measurement_remaining": measurement_limit - spent_measurements,
                    "compute_remaining": budget.remaining})

        risk_actions = None
        if (laboratory.spec.risk_quantile != .5 and self.config.risk_learning
                and self.config.strategy != "fixed" and not alternative_views
                and not _object_schema(self._spec(laboratory, offered))):
            from .risk_research_actions import RiskResearchActions
            risk_actions = RiskResearchActions(self, laboratory, acquire=acquire,
                paired_for=paired_for, command_batches=current_command_batches,
                view_rows=view_rows, transitions_for=transitions_for, budget=budget,
                actions=actions, transfer=transfer, graph=graph, length=length,
                remaining=lambda: {"observation_remaining": observation_limit - spent_observations,
                    "measurement_remaining": measurement_limit - spent_measurements,
                    "compute_remaining": budget.remaining})
        failed_risk_trial = None
        resumed_incumbent = None
        successor_refuted_incumbent = False
        if object_continuation:
            from .object_continuation import resume_entries
            recovered = resume_entries(self, laboratory, previous, continuation_candidate,
                paired_for=paired_for, command_batches=current_command_batches, budget=budget, actions=actions)
            selected, selected_channels = recovered["selected"], recovered["channels"]
            selected_training, selected_validation = recovered["training"], recovered["validation"]
            finalists = recovered["finalists"]
            resumed_incumbent = (recovered["incumbent"], recovered["incumbent_channels"])
            method_trial = True
            instrument_views_tested = any(tuple(offered) in views_from(tuple(batch["channels"])) for batch in batches)
            if object_actions is not None:
                object_actions.field_status = restored_frontier.get("field_status", "unassessed_inputs")

            if (continuation_candidate or {}).get("experiment") is not None:
                from .knowledge_revisits import execute_object_experiment
                experiment = execute_object_experiment(self, laboratory, continuation_candidate,
                    acquire=acquire, paired_for=paired_for, budget=budget, actions=actions)
                if experiment is not None:
                    new_rows, channels = experiment["rows"], experiment["channels"]
                    from .equivalent_object_programs import canonical_object_program
                    successor_refuted_incumbent = (channels == resumed_incumbent[1]
                        and canonical_object_program(resumed_incumbent[0]["candidate"].program.expanded())
                        in {canonical_object_program(program.expanded())
                            for program in experiment["refuted_programs"]})
                    inherited_relation_evidence.extend(new_rows)
                    if channels == selected_channels:
                        selected_training.extend(new_rows)
                        selected_validation = list(selected_training)
                    paid = [row for batch in paired_for(channels) for _, row in batch]
                    checked = self._assess(experiment["entries"], paid, budget,
                        initial_prediction(paid, laboratory.spec.risk_quantile), laboratory.spec.risk_quantile)
                    for entry in checked:
                        entry["selection_basis"] = "all_acquired_evidence"
                    best = self._best(checked)
                    if best is not None:
                        selected, selected_channels, finalists = best, channels, checked
                        selected_training = list(paid)
                        selected_validation = list(paid)
                    confirmed = False

            # A newly identified representation has not yet had a response
            # proposal. Use its purchased bank before buying another trajectory
            # merely to rediscover an old predictor's already measured error.
            if (object_representation_changed and selected.get("validation_loss") is not None
                    and selected["validation_loss"] > self.config.minimum_improvement
                    and selected_channels in ((), offered)):
                paired = paired_for(offered)
                training = [row for batch in paired[:-1] for _, row in batch]
                validation = [row for _, row in paired[-1]] if paired else []
                paid = [*training, *validation]
                spec = self._spec(laboratory, offered)
                reserve = 2 * length + len(inherited_relation_evidence) + len(directed_command_rows)
                allowance = min(90000, max(0, (budget.remaining - reserve) // 2))
                entry_cost = len(training) * (len(spec.feature_names) + 2) + len(validation)
                enough_confirmation = (observation_limit - spent_observations >= length
                    and measurement_limit - spent_measurements >= length * sum(rates[name] for name in offered))
                if training and validation and enough_confirmation and allowance >= entry_cost:
                    start = budget.spent
                    local = SearchBudget(allowance)
                    claims = self._research_view(spec, training, validation, local, actions, transfer, graph,
                        parent=selected if selected_channels == offered else None,
                        transitions=transitions_for(offered, training),
                        observation_remaining=observation_limit - spent_observations,
                        measurement_remaining=measurement_limit - spent_measurements)
                    budget.spend(local.spent)
                    comparison = SearchBudget(max(0, budget.remaining - reserve))
                    old_rows = (paid if selected_channels else
                                [command for batch in paired for command, _ in batch])
                    before_loss, best = None, None
                    checked = []
                    if comparison.spend(len(paid)):
                        before_loss = loss(selected["candidate"].program, old_rows,
                            selected.get("fallback", initial_prediction(old_rows, spec.risk_quantile)),
                            spec.risk_quantile)
                        fallback = initial_prediction(training, spec.risk_quantile)
                        checked = self._assess(claims, paid, comparison, fallback, spec.risk_quantile)
                        for entry in checked:
                            entry.update(fallback=fallback, selection_basis="all_acquired_evidence")
                        best = self._best(checked)
                    budget.spend(comparison.spent)
                    improved = best is not None and best["validation_loss"] + self.config.minimum_improvement < before_loss
                    actions.append({"action": "research_response_after_object_induction",
                        "object_model_id": object_research.get("model_id"), "channels": list(offered),
                        "previous_object_model_id": previous_objects.get("model_id"),
                        "fitting_rows": len(training), "selection_validation_rows": len(validation),
                        "paid_comparison_rows": len(paid), "candidate_count": len(checked),
                        "before_loss": before_loss, "after_loss": best["validation_loss"] if best else None,
                        "selected_kind": best["kind"] if improved else selected["kind"],
                        "status": "requires_fresh_confirmation" if improved else "no_paid_improvement",
                        "compute_cost": budget.spent - start, "compute_allowance": allowance,
                        "reserved_confirmation_compute": reserve})
                    if improved:
                        selected, selected_channels = best, offered
                        selected_training, selected_validation, finalists = training, validation, checked

        workflow_inputs = {}
        workflow_attempted = False

        def try_workflow():
            nonlocal workflow_attempted, selected, selected_channels, finalists
            nonlocal selected_training, selected_validation, method_trial, instrument_views_tested
            if workflow is None or workflow_attempted or confirmed or method_trial or object_continuation:
                return
            workflow_attempted = True
            continuation = {}
            command_spec = self._spec(laboratory, ())
            if (not offered and command_spec.risk_quantile == .5 and not command_spec.decision_costs
                    and set(command_spec.feature_names) <= set(command_spec.feature_bounds)
                    and command_spec.max_probe_length >= 8
                    and not _object_schema(command_spec) and not command_spec.context.get("object_model")):
                # A failed workflow's paid trajectories remain the fitting bank.
                # Leave one complete public-control pack, its boundary step and
                # the existing two confirmation passes executable after replay.
                entry = _conditional_experiment_entry_quote(command_spec,
                    observation_limit, length)
                next_confirmation = min(command_spec.max_probe_length, 2 * length)
                confirmation_compute = self._confirmation_compute_quote(command_spec,
                    next_confirmation,
                    training_rows=observation_limit,
                    noise_check=self.config.risk_learning)
                compute = max(entry["compute_cost"] + confirmation_compute,
                    min(90000, max(0, (budget.remaining - 2 * length) // 2)))
                observation = entry["observation_rows"] + next_confirmation * (
                    2 if self.config.risk_learning else 1)
                if (observation_limit - spent_observations >= observation
                        and budget.remaining >= compute + 2 * length):
                    continuation = {"observation_rows": observation,
                        "measurement_cost": 0, "compute_cost": compute,
                        "confirmation_compute": confirmation_compute,
                        "confirmation_rows": next_confirmation,
                        "research": "matched_public_controls_and_response_revision"}
            # Replaying a procedure is one research attempt. Its several fits
            # share one allowance; none may consume the fallback investigation.
            # Keep an equally funded alternative when a new measured view and
            # its confirmation can actually be bought with the current resources.
            replay_budget, measured_alternative = None, False
            def workflow_fit(channels, training, validation):
                nonlocal replay_budget, measured_alternative
                if replay_budget is None:
                    measured_alternative = bool(offered and
                        observation_limit - spent_observations >= 3 * length and
                        measurement_limit - spent_measurements >=
                            3 * length * sum(rates[name] for name in offered))
                    available_compute = max(0, budget.remaining - 2 * length)
                    allowance = (min(90000, available_compute // 2) if alternative_views else
                                 available_compute // 2 if measured_alternative else available_compute)
                    allowance = min(allowance, max(0,
                        available_compute - continuation.get("compute_cost", 0)))
                    replay_budget = SearchBudget(allowance)
                local = SearchBudget(min(replay_budget.remaining,
                    max(0, budget.remaining - 2 * length)))
                claims = self._research_view(self._spec(laboratory, channels), training,
                    validation, local, actions, transfer, graph, parent=parent,
                    transitions=transitions_for(channels, training),
                    observation_remaining=observation_limit - spent_observations,
                    measurement_remaining=measurement_limit - spent_measurements)
                budget.spend(local.spent)
                replay_budget.spend(local.spent)
                actions.append({"action": "budget_workflow_fit", "channels": list(channels),
                    "compute_allowance": local.limit, "compute_cost": local.spent,
                    "workflow_compute_limit": replay_budget.limit,
                    "workflow_compute_spent": replay_budget.spent,
                    "continuation_reserve": deepcopy(continuation),
                    "measured_alternative_funded": measured_alternative})
                return claims
            paid = [row for batch in current_command_batches() for row in batch]
            diagnostics = {}
            if self.downstream_demands is not None:
                diagnostics["downstream_demands"] = deepcopy(self.downstream_demands)
            if paid:
                diagnostics["transfer_condition"] = transfer_condition(self._spec(laboratory, ()), paid)
            replay = workflow.replay(acquire=acquire, project=view_rows, fit=workflow_fit,
                legal_views=[(), *(alternative_views or views_from(offered))],
                projectable=lambda channels: [(), *views_from(channels)],
                remaining=lambda: {"observation_remaining": observation_limit - spent_observations,
                    "measurement_remaining": measurement_limit - spent_measurements,
                    "compute_remaining": budget.remaining}, confirmation_length=length,
                threshold=self.config.minimum_improvement, diagnostics=diagnostics, actions=actions,
                continuation_reserve=continuation)
            if replay is not None:
                trial = replay["selected"]
                if trial is not None and replay["adequate"]:
                    selected, selected_channels, finalists, selected_training, selected_validation = trial
                    method_trial = True
                    instrument_views_tested |= tuple(selected_channels) == offered
                else:
                    # Failed experiments remain usable evidence. Keep complete
                    # acquisition trajectories together for the next native fit.
                    for rows, channels in replay["purchased"]:
                        workflow_inputs.setdefault(tuple(channels), []).append(rows)

        # A newly available experiment competes with response research under
        # the same resources; uncertainty alone does not authorize acquisition.
        startup_recovery = None
        if self._learn_instruments and self.config.use_memory and not object_continuation:
            ambiguous = [model for model in self.library.instrument_models
                         if model.get("status") != "counterexample" and _ambiguous_instruments(model)]
            if ambiguous and numeric_observation_research:
                try_workflow()
                if not method_trial:
                    startup_recovery = recover_observation_actions()
                    adopt_observation_recovery(startup_recovery)
            else:
                for model in reversed(ambiguous):
                    distinguish_instruments(model, offered)
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
        paid_object_bank = bool(object_research and object_research.get("groups")
            and sum(len(batch) for batch in paired_for(offered)) >= 12)
        response_reserve = (2 if paid_object_bank else 7) * length
        measurement_reserve = (2 if paid_object_bank else 4) * length
        trial_funded = (observation_limit - spent_observations >= response_reserve + 3 * trial_length
            and measurement_limit - spent_measurements >= measurement_reserve * sum(rates[name] for name in offered)
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
                if not self.library.transfer_refuted(spec, call):
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
                    if (item["kind"] in {"retained_calibrated_call", "retained_callable_transfer"}
                            and not noisy_methods and laboratory.spec.risk_quantile == .5
                            and value > self.config.minimum_improvement):
                        # Every retained call faced these paid validation rows,
                        # including calls that lose the candidate comparison.
                        # Keep their measured failure scoped to this execution;
                        # an earlier source can still apply on a later task.
                        transfer.append({"source_ids": _sources(item["candidate"].program),
                            "program": item["candidate"].program.to_dict(),
                            "validation_loss": value, "validation_rows": len(rows),
                            "status": "counterexample", "accepted": False,
                            "kind": "failed_retained_call"})
            transfer_work("shared_transfer_validation", before, start, observations, compared)
            return compared

        transfer_observations = []
        if (calls and trial_funded and not confirmed and not method_trial
                and not (startup_recovery or {}).get("handled")):
            if noisy_methods:
                first = acquire(trial_channels, "short_learned_coordinate_trial", rows_count=trial_length // 2)
                second = acquire(trial_channels, "short_learned_coordinate_trial",
                                 rows_count=trial_length - trial_length // 2)
                pilot = [*first, *second]
            else:
                pilot = list(acquire(trial_channels, "short_learned_coordinate_trial", rows_count=trial_length))
            transfer_observations = list(pilot)
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
                    if risk_actions is not None:
                        failed_risk_trial = (trial, tuple(channels))
                    supported = trial["validation_loss"] <= self.config.minimum_improvement
                    trial_support = None
                    if not supported and noisy_methods and self.config.risk_learning:
                        trial_support = assess_trial_noise(trial["candidate"].program, pilot,
                            validation, channels, check_channels)
                        supported = trial_support["adequate"]
                    condition_ready, condition_status = True, "not_requested"
                    if supported and self.config.active_probes and self.config.strategy != "fixed":
                        from .transfer_condition_research import (
                            TransferConditionResearch, covered_call_ids,
                            record_condition_coverages,
                        )
                        from .competing_object_experiments import assess_competing_object_probe
                        command_spec = self._spec(laboratory, ())
                        measured_spec = self._spec(laboratory, trial_channels)
                        condition_binding = self._instrument_binding(measured_spec, command_spec)
                        # A callable learned and freshly confirmed on the
                        # same command contract already carries its condition
                        # evidence. Rebinding only the measurement names must
                        # not reopen the same predicate questions; a changed
                        # contract still uses paid probes. A counterexample on
                        # another task limits the prior without erasing its
                        # earlier coverage; the planner checks every current
                        # paid row before accepting that coverage again.
                        condition_calls = calls
                        selected_sources = set(trial.get("method_source_ids", ()))
                        if selected_sources:
                            # Condition coverage belongs to the selected
                            # executable transfer. Competing recalled calls
                            # are not paid hypotheses once this call has won
                            # the shared validation comparison.
                            condition_calls = [call for call in calls
                                if selected_sources.intersection(_sources(call))]
                        command_features = set(command_spec.feature_names)
                        prevalidated = set()
                        if self.config.use_memory:
                            prevalidated.update(covered_call_ids(condition_calls,
                                getattr(self.library, "condition_coverages", ())))
                            for index, call in enumerate(condition_calls):
                                sources = set(_sources(call))
                                if not sources:
                                    continue
                                if any(
                                    prior.get("source_concept_id") in sources
                                    and prior.get("status") in {"supported", "conditional"}
                                    and (prior.get("research_program") or {}).get("status")
                                        == "supported_on_fresh_trajectory"
                                    and set(prior.get("source_features", ())) == command_features
                                    and set(prior.get("control_features", ())) == command_features
                                    for prior in self.library.meta_priors
                                ):
                                    prevalidated.add(index)
                        condition_research = TransferConditionResearch(
                            condition_calls, prevalidated_call_ids=prevalidated)
                        for step in range(self.config.max_rounds + 1):
                            paid_pairs = [pair for batch in paired_for(trial_channels) for pair in batch]
                            plan = condition_research.plan(command_spec, trial["candidate"].program,
                                budget, paid_rows=[pair[1] for pair in paid_pairs],
                                paired_rows=paid_pairs, binding=condition_binding,
                                methods=self.library.program_experiment_methods if self.config.use_memory else None,
                                serial=serial, length=4, tolerance=self.config.minimum_improvement,
                                task_id=f"{laboratory.spec.task_id}:{self.config.seed}",
                                prior_coverages=getattr(self.library, "condition_coverages", ()))
                            condition_status = plan["forecast_status"]
                            scope = plan["transfer_scope"]
                            if condition_status == "covered_by_paid_evidence":
                                record_condition_coverages(scope, condition_research.calls,
                                    self.library.condition_coverages,
                                    task_id=f"{laboratory.spec.task_id}:{self.config.seed}")
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
                            if (spent_observations + query_rows + ordinary_rows > observation_limit
                                    or spent_measurements + query_rows * sum(rates[name] for name in trial_channels)
                                    + 4 * length * sum(rates[name] for name in offered) > measurement_limit
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
                                model = research_instruments(trial_channels)
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
                        "ordinary_observation_reserve": response_reserve})
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
                        if workflow is not None:
                            workflow.frozen([trial, calibrated], training, validation_view,
                                self._spec(laboratory, channels))
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
                            if workflow is not None:
                                workflow.tested(trial["candidate"].program, confirmation_view,
                                    confirmation_loss, accepted=confirmed)
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
                                if risk_actions is not None:
                                    failed_risk_trial = (trial, tuple(channels))
        if not (startup_recovery or {}).get("handled"):
            try_workflow()
        prefetched_inputs = (startup_recovery or {}).get("pending_inputs")
        observation_recovery = startup_recovery
        if (risk_actions is not None and failed_risk_trial is not None
                and not confirmed and not method_trial):
            trial, channels = failed_risk_trial
            last = batches[-1]["probe"]
            observation_recovery = risk_actions.propose(trial, channels=channels,
                failed_probe=Probe(last["probe_id"], tuple(last["sequence"])),
                noise_assessment=noisy_support or trial_support)
            adopt_observation_recovery(observation_recovery)
        if (self._revise_inputs and (command_calls or calls) and not confirmed and not method_trial
                and numeric_observation_research and batches
                and not (observation_recovery or {}).get("handled")):
            retained = None
            if command_calls:
                retained = {"candidate": Candidate("retained:recovery", command_calls[0], "retained_response"),
                            "kind": "retained_response", "family_id": "T03", "validation_loss": None}
            last = batches[-1]["probe"]
            observation_recovery = recover_observation_actions(retained,
                Probe(last["probe_id"], tuple(last["sequence"])))
            if not adopt_observation_recovery(observation_recovery) and observation_recovery:
                prefetched_inputs = observation_recovery.get("pending_inputs")
        if (self._revise_inputs and command_calls and batches and not confirmed and not method_trial
                and not (observation_recovery or {}).get("handled")):
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
                    research_instruments(offered)
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
                research_calls, screened = self._screen_risk_knowledge(self._spec(laboratory, ()),
                    pilot, checked, local, command_calls, actions)
                proposed = [*screened, *self._program_claims(self._spec(laboratory, ()), pilot,
                                                           local, research_calls, actions)]
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
                    and observation_limit - spent_observations >= 7 * length):
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
            if object_actions is not None and command_calls and batches and not method_trial:
                # A failed individual call can be a changed composition of
                # intact dependencies. Try those on the paid command rows
                # before purchasing a new measurement/structure investigation.
                retained = {"candidate": Candidate("retained:object_recovery", command_calls[0], "retained_response"),
                            "kind": "retained_response", "family_id": "T03", "validation_loss": None}
                last = batches[-1]["probe"]
                observation_recovery = object_actions.propose(retained,
                    failed_probe=Probe(last["probe_id"], tuple(last["sequence"])))
                adopt_observation_recovery(observation_recovery)
                command_calls = self._recall(self._spec(laboratory, ()))
            if not method_trial:
                first, second = prefetched_inputs or (acquire(offered, "diagnose_input_mapping"),
                    acquire(offered, "compare_input_mapping"))
                if first is not None and second is not None:
                    if prefetched_inputs is None:
                        research_instruments(offered)
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
        reused_object_bank = False
        if (object_research and object_research.get("groups") and not object_continuation
                and prefetched_inputs is None and not confirmed and not method_trial):
            purchased = [row for batch in paired_for(offered) for _, row in batch]
            if len(purchased) >= 12:
                middle = len(purchased) // 2
                prefetched_inputs = (purchased[:middle], purchased[middle:])
                reused_object_bank = True
                actions.append({"action": "reuse_object_evidence_for_response",
                    "object_model_id": object_research.get("model_id"),
                    "channels": list(offered), "training_rows": len(purchased),
                    "reserved_fresh_response_rows": 2 * length})
        stages = ([offered] if self.config.strategy == "fixed" or prefetched_inputs
                  else [*workflow_inputs, (), offered])
        from .condition_response_actions import research_condition_response_actions
        prepared_response_inputs = False
        if (transfer_observations and not offered and not confirmed and not method_trial
                and prefetched_inputs is None and () not in workflow_inputs):
            paid_batches = current_command_batches()
            if len(paid_batches) >= 2 and sum(len(batch) for batch in paid_batches[:-1]) >= 8:
                selected_training = [row for batch in paid_batches for row in batch]
                missing = max(0, 2 * length - len(selected_training))
                if missing:
                    added = acquire((), "complete_paid_response_support", rows_count=missing)
                    if added is not None:
                        selected_training.extend(added)
                selected_validation = acquire((), "compare_on_new_trajectory", rows_count=length)
                if selected_validation is None:
                    selected_validation = paid_batches[-1]
                selected, selected_training, finalists = research_condition_response_actions(
                    self, self._spec(laboratory, ()), None, selected_training,
                    selected_validation, [], budget, actions, acquire=acquire, serial=serial,
                    remaining=lambda: {"observation_remaining": observation_limit - spent_observations,
                        "measurement_remaining": measurement_limit - spent_measurements})
                if selected is not None:
                    method_trial = max(selected["training_loss"], selected["validation_loss"]) <= self.config.minimum_improvement
                    parent = selected
                if not method_trial:
                    # Preparation has already bought complete fitting and
                    # validation trajectories even if its narrow researcher
                    # returned no candidate. Pass those exact rows onward.
                    workflow_inputs[()] = [selected_training, selected_validation]
                    prepared_response_inputs = True
        risk_response_bank = None
        for stage, channels in enumerate(dict.fromkeys(stages) if not confirmed and not method_trial else ()):
            validation = None
            if prefetched_inputs is not None and channels == offered:
                first, second = prefetched_inputs
                prefetched_inputs = None
            elif tuple(channels) in workflow_inputs:
                paid = workflow_inputs.pop(tuple(channels))
                first = [row for batch in paid[:-1] for row in batch] if len(paid) > 1 else paid[0]
                second = []
                validation = paid[-1] if len(paid) > 1 else None
                actions.append({"action": ("reuse_paid_response_evidence" if prepared_response_inputs
                    else "reuse_workflow_evidence"), "channels": list(channels),
                    "training_rows": len(first), "validation_rows": len(validation or []),
                    "new_observations": 0})
            elif risk_response_bank is not None and channels == offered:
                first, second, validation = risk_response_bank
                actions.append({"action": "reuse_paid_risk_response_views", "channels": list(channels),
                    "fitting_rows": len(first) + len(second), "validation_rows": len(validation),
                    "new_observations": 0, "comparison_basis": "same_paid_response_trajectories"})
            elif (risk_actions is not None and not channels and offered
                    and observation_limit - spent_observations >= 5 * length
                    and measurement_limit - spent_measurements >= 5 * length * sum(rates.values())):
                first, second = acquire(offered, "fit"), acquire(offered, "fit")
                if first is not None and second is not None:
                    validation = acquire(offered, "compare_on_new_trajectory", finalists)
                    if validation is not None:
                        risk_response_bank = first, second, validation
                        first, second = (view_rows(rows, offered, channels) for rows in (first, second))
                        validation = view_rows(validation, offered, channels)
            else:
                retained_rows = (len(transfer_observations) if not channels
                    and not trial_channels and len(laboratory.spec.feature_names) >= 4 else 0)
                if retained_rows:
                    # Refuted transfer experiments are still measured response
                    # evidence. Purchase the missing fit rows, preserving funds
                    # for the complete interacting-condition experiment.
                    missing = max(2, 2 * length - retained_rows)
                    first_count = min(length, missing - 1)
                    first = acquire(channels, "fit", rows_count=first_count)
                    second = acquire(channels, "fit", rows_count=missing - first_count)
                else:
                    first, second = acquire(channels, "fit"), acquire(channels, "fit")
            if first is None or second is None:
                break
            if validation is None:
                validation = acquire(channels, "compare_on_new_trajectory", finalists)
            if validation is None:
                break
            instrument_views_tested |= tuple(channels) == offered
            training = [*first, *second]
            if transfer_observations and tuple(channels) in views_from(tuple(trial_channels)):
                training.extend(view_rows(transfer_observations, trial_channels, channels))
            if numeric_observation_research and not channels:
                # A failed current-only fit does not refute a remembered
                # history law. Test the actual calls on these paid trajectories
                # before buying instruments to revise their input mapping.
                retained = []
                spec = self._spec(laboratory, ())
                fallback = initial_prediction(training, spec.risk_quantile)
                from .transfer_condition_research import _conditional
                for call in self._recall(spec, training):
                    if _conditional(call.expanded()):
                        # Random trajectory accuracy cannot establish coverage
                        # of an input-dependent branch. Keep its usual research.
                        continue
                    if not budget.spend(len(training)):
                        break
                    retained.append({"candidate": Candidate(f"retained_response:{len(retained)}",
                        call, "retain_learned_call", parents=tuple(_sources(call))),
                        "kind": "retained_callable_transfer", "family_id": "T03",
                        "method_source_ids": _sources(call),
                        "training_loss": loss(call, training, fallback, spec.risk_quantile)})
                checked = self._assess(retained, validation, budget, fallback, spec.risk_quantile)
                for claim in checked:
                    adequate = max(claim["training_loss"], claim["validation_loss"]) <= self.config.minimum_improvement
                    transfer.append({"source_ids": _sources(claim["candidate"].program),
                        "program": claim["candidate"].program.to_dict(),
                        "validation_loss": claim["validation_loss"], "validation_rows": len(validation),
                        "status": "supported" if adequate else "counterexample", "accepted": False})
                supported = [claim for claim in checked
                    if max(claim["training_loss"], claim["validation_loss"]) <= self.config.minimum_improvement]
                if supported:
                    selected, selected_channels, finalists = self._best(supported), (), checked
                    selected_training, selected_validation = training, validation
                    break
                cheap = _fit(_terms(laboratory.spec.feature_names), training, budget,
                             "observation_counterexample", q=.5)
                checked = self._assess([cheap], validation, budget, initial_prediction(training, .5), .5)
                cheap = self._best(checked)
                if cheap is not None and max(cheap["training_loss"], cheap["validation_loss"]) > self.config.minimum_improvement:
                    if adopt_observation_recovery(recover_observation_actions(cheap)):
                        break
            if not alternative_views:
                # Compare calibrated calls and ordinary history responses on
                # paid instrument rows before opening condition/duration
                # research. A failed inherited law cannot consume the chance
                # to discover a simpler replacement from those same readings.
                reused = []
                recall_views = (sorted(views_from(channels),
                    key=lambda names: (sum(rates[name] for name in names), len(names)))
                    if channels and numeric_observation_research else ())
                for names in recall_views:
                    spec = self._spec(laboratory, names)
                    train_view = view_rows(training, channels, names)
                    validation_view = view_rows(validation, channels, names)
                    fallback = initial_prediction(train_view, spec.risk_quantile)
                    for call in self._recall(spec, train_view):
                        terms = [call, Program("mul", children=(call, call)), *_terms(spec.feature_names)]
                        quote = len(train_view) * (len(set(terms)) + 2) + len(validation_view)
                        if budget.remaining < quote:
                            break
                        fitted = _fit(terms, train_view, budget, "callable_transfer",
                                      family="T03", q=spec.risk_quantile)
                        checked = self._assess([fitted], validation_view, budget, fallback, spec.risk_quantile)
                        for claim in checked:
                            adequate = max(claim["training_loss"], claim["validation_loss"]) <= self.config.minimum_improvement
                            claim["method_source_ids"] = _sources(claim["candidate"].program)
                            transfer.append({"source_ids": claim["method_source_ids"],
                                "program": claim["candidate"].program.to_dict(),
                                "validation_loss": claim["validation_loss"], "validation_rows": len(validation_view),
                                "status": "supported" if adequate else "counterexample", "accepted": False})
                            if adequate:
                                reused.append((claim, names, checked, train_view, validation_view))
                    if not reused:
                        depth_limit = min(self.config.max_history,
                            max((len(row.history) for row in train_view), default=0))
                        for depth in range(depth_limit + 1):
                            terms = _terms(spec.feature_names, depth)
                            quote = len(train_view) * (len(terms) + 2) + len(validation_view)
                            if budget.remaining < quote:
                                break
                            ordinary = _fit(terms, train_view, budget, "measured_history_response",
                                            family="T08", q=spec.risk_quantile)
                            checked = self._assess([ordinary], validation_view, budget,
                                                  fallback, spec.risk_quantile)
                            for claim in checked:
                                if max(claim["training_loss"], claim["validation_loss"]) <= self.config.minimum_improvement:
                                    reused.append((claim, names, checked, train_view, validation_view))
                            if reused:
                                break
                    if reused:
                        break
                if reused:
                    selected, selected_channels, finalists, selected_training, selected_validation = min(
                        reused, key=lambda item: item[0]["candidate"].program.size)
                    break
                early, prepared_training, prepared_claims = research_condition_response_actions(
                    self, self._spec(laboratory, channels), parent, training, validation,
                    finalists, budget, actions, acquire=acquire, serial=serial,
                    channels=channels, measurement_rate=sum(rates[name] for name in channels),
                    remaining=lambda: {"observation_remaining": observation_limit - spent_observations,
                        "measurement_remaining": measurement_limit - spent_measurements})
                if early is not None:
                    training, parent = prepared_training, early
                    if max(early["training_loss"], early["validation_loss"]) <= self.config.minimum_improvement:
                        selected, selected_channels, finalists = early, channels, prepared_claims
                        selected_training, selected_validation = training, validation
                        break
            views = [channels] if self.config.strategy == "fixed" else views_from(channels)
            views.sort(key=lambda names: (sum(rates[name] for name in names), len(names)))
            if reused_object_bank and channels == offered:
                # First use the representation whose prerequisites were just
                # measured. Other views remain fallback explanations; a solved
                # response can subsequently be compiled to calibrated controls.
                views.sort(key=lambda names: tuple(names) != tuple(offered))
            available = []
            for names in views:
                spec = self._spec(laboratory, names)
                train_view = view_rows(training, channels, names)
                validation_view = view_rows(validation, channels, names)
                if risk_actions is not None:
                    options = risk_actions._options(names)
                    followup_cost = min((option.compute_cost for option in options), default=0)
                    view_budget = SearchBudget(max(0, budget.remaining - followup_cost - 24000))
                else:
                    view_budget = SearchBudget(min(90000, budget.remaining // 2)) if alternative_views else budget
                claims = self._research_view(spec, train_view, validation_view, view_budget, actions,
                                             transfer, graph, parent=parent,
                                             test_response_noise=test_response_noise,
                                             transitions=transitions_for(names, train_view),
                                observation_remaining=observation_limit - spent_observations,
                                measurement_remaining=measurement_limit - spent_measurements)
                if alternative_views or risk_actions is not None:
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

        if (risk_actions is not None and selected is not None and not confirmed
                and risk_actions.pending is None and selected_validation):
            complete = ([row for batch in paired_for(offered) for _, row in batch] if offered
                        else [row for batch in current_command_batches() for row in batch])
            blocks = trajectory_blocks(complete)
            if len(blocks) >= 2:
                options = risk_actions._options(selected_channels)
                followup_cost = min((option.compute_cost for option in options), default=0)
                diagnosis = SearchBudget(min(24000, max(0, budget.remaining - followup_cost)))
                support = self._support(selected["candidate"].program,
                    noise_coordinates([row for block in blocks[:-1] for row in block], selected_channels, offered),
                    noise_coordinates(blocks[-1], selected_channels, offered),
                    self._spec(laboratory, offered), diagnosis)
                budget.spend(diagnosis.spent)
                actions.append({"action": "assess_paid_risk_agenda", "reason": support["reason"],
                    "adequate": support["adequate"], "training_rows": sum(map(len, blocks[:-1])),
                    "validation_rows": len(blocks[-1]), "fixed_quantile": laboratory.spec.risk_quantile,
                    "compute_cost": diagnosis.spent, "reserved_research_compute": followup_cost})
                if (support["reason"] in {"quantile_miscalibration", "residual_loss_shift",
                        "predictable_residual", "repeatable_unexplained_response"}
                        or support.get("structural_counterexample")):
                    recovered = risk_actions.propose(selected, channels=selected_channels,
                        noise_assessment=support)
                    adopt_observation_recovery(recovered)

        def charge_research_followup(origin, stage, before_costs):
            if origin is None:
                return
            obs, meter, compute = before_costs
            added = {"observation_cost": spent_observations - obs,
                     "measurement_cost": spent_measurements - meter,
                     "compute_cost": budget.spent - compute}
            if not any(added.values()):
                return
            self._charge_action_costs(origin, added, actions, stage)

        settled_direction_trials = set()

        def direction_key(origin):
            return (origin["context"], origin["action"], origin["trial"] if origin["trial"] is not None
                    else ("pending", origin["pending_id"]))

        def revise_direction_outcome(entry, *, status, basis, rows=(), measured_loss=None,
                                     program_fallback=None, probe_id=None, counterexample_loss=None,
                                     counterexample_rows=0):
            origin = entry.get("research_action_origin") if entry is not None else None
            if origin is None or not self.config.use_memory:
                return
            before_loss, after_loss, compute_cost = None, None, 0
            if rows and measured_loss is not None:
                baseline_program = Program.from_dict(origin["baseline_program"])
                # The response is already scored by the frozen confirmation.
                # Spend separately to compare its pre-direction alternative on
                # exactly the same observations, never against an older loss.
                available = set.intersection(*(set(row.features) for row in rows))
                if not baseline_program.required_features() <= available:
                    if status == "supported":
                        status = "unresolved"
                    basis = "baseline_measurements_unavailable"
                elif budget.spend(len(rows)):
                    before_loss = loss(baseline_program, rows, origin["fallback"], origin["risk_quantile"])
                    after_loss = measured_loss
                    compute_cost = len(rows)
                elif status == "supported":
                    status = "unresolved"
            evidence_rows = len(rows) or counterexample_rows
            if origin["trial"] is None:
                if before_loss is None or after_loss is None:
                    origin["pending_status"] = status
                    actions.append({"action": "research_action_pending_outcome",
                        "research_action": origin["action"], "context": origin["context"],
                        "pending_id": origin["pending_id"], "trial": None, "status": status,
                        "evidence_basis": basis, "probe_id": probe_id,
                        "counterexample_loss": counterexample_loss, "evidence_rows": evidence_rows})
                    settled_direction_trials.add(direction_key(origin))
                    return
                # An already fitted parent or an empty holdout cannot award
                # early credit. The first fresh pair creates its single trial,
                # carrying all paid work accumulated while it was pending.
                policy = ResearchActionPolicy(rows, self._spec(laboratory, selected_channels),
                                               self.library.action_experience)
                paid_costs = dict(origin["pending_costs"])
                feedback = policy.update(origin["action"], context=origin["context"],
                    before_loss=before_loss, after_loss=after_loss, **paid_costs)
                origin.update(trial=feedback["trials"], credited_gain=feedback["last_gain"])
                actions.append({"action": "research_action_feedback", "research_action": origin["action"],
                    "context": origin["context"], "pending_id": origin["pending_id"],
                    "before_loss": before_loss, "after_loss": after_loss, **paid_costs,
                    "evidence_rows": len(rows), "evidence_basis": "fresh_frozen_confirmation",
                    "cost_basis": "deferred_trial_total", "feedback": feedback})
            feedback = ResearchActionPolicy.revise_feedback(self.library.action_experience, origin,
                before_loss=before_loss, after_loss=after_loss, status=status,
                compute_cost=compute_cost, evidence_rows=evidence_rows)
            actions.append({"action": "research_action_confirmation", "research_action": origin["action"],
                "context": origin["context"], "trial": origin["trial"], "status": status,
                "before_loss": before_loss, "after_loss": after_loss,
                "baseline_program": origin["baseline_program"],
                "program": entry["candidate"].program.to_dict(),
                "baseline_fallback": origin["fallback"], "program_fallback": program_fallback,
                "risk_quantile": origin["risk_quantile"], "probe_id": probe_id,
                "evidence_rows": evidence_rows, "compute_cost": compute_cost,
                "counterexample_loss": counterexample_loss, "evidence_basis": basis,
                "feedback": feedback})
            settled_direction_trials.add(direction_key(origin))

        def direction_counterexample(entry, plan, evidence):
            if entry.get("research_action_origin") is None:
                return
            encoded = entry["candidate"].program.to_dict()
            for hypothesis, outcome in zip(plan.get("hypotheses", ()), evidence.get("hypotheses", ())):
                if hypothesis["program"] == encoded and outcome["status"] == "counterexample":
                    probe = plan.get("probe")
                    revise_direction_outcome(entry, status="refuted", basis="paid_counterexample",
                        probe_id=probe.probe_id if probe is not None else None,
                        counterexample_loss=outcome["measured_loss"], counterexample_rows=outcome["rows"])
                    break

        structure_comparison_pending = deepcopy((previous or {}).get(
            "object_continuation_state", {}).get("structure_comparison_pending"))

        def research_selected_structure():
            nonlocal selected, selected_training, finalists, structure_comparison_pending, confirmed
            if selected is not None and selected.get("structure") and not confirmed:
                action_origin = selected.get("research_action_origin")
                before_costs = (spent_observations, spent_measurements, budget.spent)
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

                from .object_continuation import decode_entry, encode_entry
                retained_pool = list(finalists)
                for payload in self._object_candidate_frontiers.get(tuple(spec.feature_names), []):
                    entry = decode_entry(payload)
                    if entry not in retained_pool:
                        retained_pool.append(entry)
                comparison_complete = True

                def preserve_pool(entries=()):
                    nonlocal finalists
                    for entry in [*entries, selected]:
                        if entry not in retained_pool:
                            retained_pool.append(entry)
                    finalists = list(retained_pool)
                    self._object_candidate_frontiers[tuple(spec.feature_names)] = [
                        encode_entry(entry) for entry in retained_pool]

                def revision_allowance(limit):
                    programs = {entry["candidate"].program for entry in [selected, *retained_pool]}
                    reserve = len(selected_training) * (len(programs) + 16)
                    return min(limit, max(0, budget.remaining - reserve))

                # Saved counterexamples already carry complete trajectories.
                # Resolve their local structure before paying for another
                # symmetry/locality experiment or replaying the same fit.
                paid_revision = None
                if not partial_objects and _object_schema(spec):
                    paid = ([row for batch in paired_for(selected_channels) for _, row in batch]
                            if selected_channels else [row for batch in current_command_batches() for row in batch])
                    paid = paid or list(selected_training)
                    paid_revision = self._revise_paid_structure(spec, selected, paid, budget, actions,
                        finalists=retained_pool, transitions=transitions_for(selected_channels))
                    if paid_revision["full_comparison"]:
                        structure_comparison_pending = None
                        selected = paid_revision["selected"]
                        selected_training = list(paid)
                        retained_pool = list(paid_revision["claims"])
                        if action_origin is not None:
                            selected["research_action_origin"] = action_origin
                    if not paid_revision["full_comparison"]:
                        confirmed = False
                        structure_comparison_pending = {"status": paid_revision["status"],
                            "rows": len(paid), "candidate_id": selected["candidate"].candidate_id,
                            "compute_cost": paid_revision["compute_cost"]}
                    preserve_pool(paid_revision["new_claims"])
                    if (not paid_revision["full_comparison"]
                            or selected["validation_loss"] <= self.config.minimum_improvement
                            or not selected.get("structure")):
                        actions.append({"action": "return_paid_structure_research",
                            "status": paid_revision["status"], "new_observations": 0,
                            "requires_fresh_confirmation": bool(paid_revision["action"].get("changed_program")),
                            "full_comparison": paid_revision["full_comparison"]})
                        charge_research_followup(action_origin, "paid_structure_revision", before_costs)
                        return

                def fitted_best(entries):
                    nonlocal comparison_complete, retained_pool, structure_comparison_pending, confirmed
                    pool = [selected, *retained_pool, *entries]
                    readable = [entry for entry in pool if all(
                        entry["candidate"].program.required_features() <= row.features.keys()
                        for row in selected_training)]
                    programs = list(dict.fromkeys(entry["candidate"].program for entry in readable))
                    required = len(selected_training) * len(programs)
                    if not budget.can_spend(required):
                        comparison_complete = False
                        confirmed = False
                        structure_comparison_pending = {"status": "compute_unresolved",
                            "rows": len(selected_training), "compute_required": required,
                            "candidate_id": selected["candidate"].candidate_id, "compute_cost": 0}
                        preserve_pool(entries)
                        actions.append({"action": "compare_structure_revision", "rows": len(selected_training),
                            "status": "compute_unresolved", "full_comparison": False,
                            "compute_required": required, "compute_cost": 0})
                        return selected
                    fallback = selected.get("fallback", initial_prediction(selected_training, spec.risk_quantile))
                    scores = {}
                    for program in programs:
                        budget.spend(len(selected_training))
                        scores[program] = loss(program, selected_training, fallback, spec.risk_quantile)
                    checked = [{**entry, "training_loss": scores[entry["candidate"].program],
                        "validation_loss": scores[entry["candidate"].program], "fallback": fallback,
                        "selection_basis": "all_acquired_evidence"} for entry in readable]
                    incumbent = checked[0]
                    best = self._best(checked)
                    if incumbent["validation_loss"] <= best["validation_loss"] + self.config.minimum_improvement:
                        best = incumbent
                    retained_pool = [*checked, *(entry for entry in pool if entry not in readable)]
                    structure_comparison_pending = None
                    actions.append({"action": "compare_structure_revision",
                        "rows": len(selected_training), "incumbent_loss": incumbent["validation_loss"],
                        "alternatives": len(entries), "selected_kind": best["kind"],
                        "selected_loss": best["validation_loss"], "full_comparison": True,
                        "compute_cost": required})
                    if action_origin is not None:
                        best["research_action_origin"] = action_origin
                    return best

                from .competing_object_experiments import (
                    plan_competing_object_probe, assess_competing_object_probe)
                rivals = []
                if selected.get("object_competition"):
                    local = SearchBudget(revision_allowance(min(90000, budget.remaining // 2)))
                    rivals = propose_structures(spec, selected_training, local, parent=selected,
                        tolerance=self.config.minimum_improvement, transitions=transitions_for(selected_channels),
                        new_observations=selected_validation)
                    budget.spend(local.spent)
                    if rivals:
                        selected = fitted_best(rivals)
                        if not comparison_complete:
                            preserve_pool(rivals)
                            charge_research_followup(action_origin, "structural_revision_comparison", before_costs)
                            return
                experiments = []
                schema = ObjectSchema.from_dict(selected["structure"]["schema"])
                plans = structural_experiments(selected, spec, serial, length=4,
                    paired_observations=[pair for batch in paired_for(selected_channels) for pair in batch]
                    ) if self.config.active_probes else ()
                for plan in plans:
                    required_rows = plan["baseline"].cost + plan["probe"].cost + length
                    if (spent_observations + required_rows > observation_limit or
                            spent_measurements + required_rows * sum(rates[name] for name in selected_channels)
                            > measurement_limit):
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
                revised = paid_revision["new_claims"] if paid_revision is not None else []
                if experiments or paid_revision is None:
                    local = SearchBudget(revision_allowance(min(90000, budget.remaining // 2)))
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
                if not comparison_complete or not selected.get("structure"):
                    preserve_pool(rivals)
                    charge_research_followup(action_origin, "structural_revision_comparison", before_costs)
                    return
                # A measured range is meaningful inside its complete response.
                # The relation-interval stage below plans experiments through
                # the learned instruments and history operators, and preserves
                # new range evidence even when the old program remains right.
                if selected["structure"].get("radius_interval") and not selected.get("object_competition"):
                    actions.append({"action": "route_structural_boundary_research",
                        "reason": "compare_complete_response_predictions",
                        "radius_interval": selected["structure"]["radius_interval"],
                        "history_depth": selected["candidate"].program.history_depth})
                for index in range(2 * self.config.max_rounds if self.config.active_probes
                                   and selected.get("object_competition") else 0):
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
                            spent_observations + required_rows > observation_limit or
                            spent_measurements + required_rows * sum(rates[name] for name in selected_channels)
                            > measurement_limit):
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
                        direction_counterexample(selected, plan, evidence)
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
                            research_instruments(selected_channels)
                    if not planned and selected.get("research_action_origin") is not None:
                        if budget.spend(len(measured)):
                            measured_error = loss(selected["candidate"].program, measured, 0., spec.risk_quantile)
                            if measured_error > self.config.minimum_improvement:
                                revise_direction_outcome(selected, status="refuted", basis="paid_counterexample",
                                    probe_id=probe.probe_id, counterexample_loss=measured_error,
                                    counterexample_rows=len(measured))
                    selected_training.extend(measured)
                    local = SearchBudget(revision_allowance(min(30000, budget.remaining // 2)))
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
                    if not comparison_complete or not selected.get("structure"):
                        break
                    actions.append({"action": "revise_structure_boundary", "probe_id": probe.probe_id,
                        "before_interval": interval,
                        "after_interval": selected["structure"].get("radius_interval"),
                        "training_loss": selected["training_loss"], "compute_cost": local.spent})
                if experiments and selected.get("structure"):
                    selected["structure"]["hypotheses"] = tested_hypotheses
                if method_sources and selected.get("structure") and selected["structure"]["fits"][0]["family"] == method_family:
                    selected["method_source_ids"] = method_sources
                    selected["kind"] = f"structural_method_transfer:{method_family}"
                    selected["candidate"] = replace(selected["candidate"], parents=tuple(dict.fromkeys(
                        (*selected["candidate"].parents, *method_sources))))
                if action_origin is not None:
                    selected["research_action_origin"] = action_origin
                preserve_pool(rivals)
                charge_research_followup(action_origin, "structural_experiments_and_revision", before_costs)

        relation_inputs_unresolved = False

        def inspect_relation_inputs(origin, plan, evidence):
            nonlocal selected, finalists, confirmed, command_calls, relation_inputs_unresolved
            nonlocal response_research_after_inputs
            from .relation_input_diagnosis import diagnose_relation_inputs

            def acquire_inputs(channels, purpose, *, probe=None):
                # Repairing the planning instruments still leaves the current
                # response needing an independent confirmation trajectory.
                allowance = observation_limit - spent_observations - length
                rate = sum(rates[name] for name in channels)
                confirmation_cost = length * sum(rates[name] for name in selected_channels)
                if rate:
                    allowance = min(allowance,
                        (measurement_limit - spent_measurements - confirmation_cost) // rate)
                if probe is not None:
                    return acquire(channels, purpose, probe=probe) if probe.cost <= allowance else None
                # Keep room for a distinguishing instrument trajectory after
                # the varied fitting/comparison readings, when funds are tight.
                count = min(length, allowance // 2)
                return acquire(channels, purpose, rows_count=count) if count > 0 else None

            alternatives = list(finalists)
            diagnosis = diagnose_relation_inputs(self, laboratory, selected, origin, plan, evidence,
                acquire_inputs, paired_for, budget, actions, channels=selected_channels)
            feedback_plan, feedback_evidence = plan, evidence
            feedback_binding = self._instrument_binding(self._spec(laboratory, selected_channels),
                                                        self._spec(laboratory, ()))
            if diagnosis is not None:
                new_rows = diagnosis["response_rows"]
                selected_training.extend(new_rows)
                inherited_relation_evidence.extend(new_rows)
                directed_command_rows.extend(diagnosis["command_rows"])
                confirmed = False
                relation_inputs_unresolved = diagnosis["status"] == "unresolved_inputs"
                feedback_plan = diagnosis.get("repaired_plan", diagnosis["plan"])
                feedback_evidence = diagnosis.get("repaired_evidence", diagnosis["evidence"])
                feedback_binding = diagnosis["binding"]
                if diagnosis["status"] == "inputs_revised":
                    selected, finalists = diagnosis["entry"], [diagnosis["entry"]]
                    command_calls = self._recall(self._spec(laboratory, ()))
                    response_research_after_inputs = diagnosis["response_status"] == "counterexample"
                    if selected_channels and response_research_after_inputs:
                        # The experiment may already support another retained
                        # radius. Compare those explanations on the paid bank
                        # before spending on a new response search.
                        rows = list({id(row): row for row in [*selected_training,
                            *selected_validation, *diagnosis["response_evidence"]]}.values())
                        before, compared = budget.spent, []
                        for item in alternatives:
                            if not budget.spend(len(rows) + len(diagnosis["response_evidence"])):
                                break
                            program = item["candidate"].program
                            measured = loss(program, rows, 0., laboratory.spec.risk_quantile)
                            current = loss(program, diagnosis["response_evidence"],
                                0., laboratory.spec.risk_quantile)
                            compared.append({**item, "training_loss": measured,
                                "validation_loss": max(measured, current),
                                "selection_basis": "all_acquired_evidence"})
                        best = self._best(compared)
                        actions.append({"action": "compare_input_repair_candidates",
                            "before_loss": diagnosis["response_loss"],
                            "after_loss": best["validation_loss"] if best else None,
                            "candidate_id": best["candidate"].candidate_id if best else None,
                            "evidence_rows": len(rows), "compute_cost": budget.spent - before})
                        if best is not None and best["validation_loss"] < diagnosis["response_loss"]:
                            selected, finalists = best, compared
                            diagnosis.update(entry=best, response_loss=best["validation_loss"],
                                response_status="supported_on_paid_rows" if best["validation_loss"] <=
                                    self.config.minimum_improvement else "counterexample")
                            response_research_after_inputs = diagnosis["response_status"] == "counterexample"
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

        def research_selected_mechanisms():
            nonlocal selected, selected_training, finalists, confirmed
            revision = (selected or {}).get("mechanism_revision")
            if (not revision or revision["status"] != "proposed" or
                    not self.config.active_probes or self.config.strategy == "fixed"):
                return
            from .competing_object_experiments import plan_competing_object_probe, assess_competing_object_probe
            original = Program.from_dict(revision["original_program"])
            source = {"candidate": Candidate("mechanism_original", original, "retain_mechanism",
                        parents=tuple(revision["source_ids"])),
                      "kind": "retained_mechanism", "family_id": selected["family_id"],
                      "method_source_ids": revision["source_ids"]}
            for key in ("research_program", "structure"):
                if revision.get("original_" + key) is not None:
                    source[key] = deepcopy(revision["original_" + key])
            alternatives = [selected, source, *[item for item in finalists
                if item is not selected and item.get("mechanism_revision", {}).get("original_program")
                == revision["original_program"]]]
            count = min(laboratory.spec.max_probe_length,
                max(5, 2 * max(item["candidate"].program.history_depth for item in alternatives) + 3))
            rate = sum(rates[name] for name in selected_channels)
            previous_noise = selected.get("response_noise_evidence", {})
            repeat_response = bool(not selected_channels and laboratory.spec.risk_quantile == .5
                and previous_noise.get("channels") == []
                and (previous_noise.get("replication", {}).get("repeat_mse") or 0.)
                    > self.config.minimum_improvement)
            experiment_rows = count * (2 if repeat_response else 1)
            confirmation_rows = length * (2 if repeat_response else 1)
            if (spent_observations + experiment_rows + confirmation_rows > observation_limit or
                    spent_measurements + (experiment_rows + confirmation_rows) * rate > measurement_limit):
                return
            action_origin = selected.get("research_action_origin")
            before_costs = spent_observations, spent_measurements, budget.spent
            spec, command_spec = self._spec(laboratory, selected_channels), self._spec(laboratory, ())
            binding = self._instrument_binding(spec, command_spec)
            local = SearchBudget(min(6000, budget.remaining // 4))
            plan = plan_competing_object_probe(command_spec, alternatives, local, binding=binding,
                serial=serial, length=count, tolerance=self.config.minimum_improvement,
                methods=self.library.program_experiment_methods if self.config.use_memory else None,
                task_id=f"{laboratory.spec.task_id}:{self.config.seed}")
            budget.spend(local.spent)
            actions.append({"action": "plan_mechanism_experiment",
                **{key: value for key, value in plan.items() if key not in ("probe", "hypotheses", "predicted_fields")},
                "probe_id": plan["probe"].probe_id if plan["probe"] else None})
            if plan["probe"] is not None:
                if repeat_response:
                    rows = plan["probe"].cost
                    # Fund both acquisitions, all frozen comparisons and new
                    # confirmation scoring before buying the noisy experiment.
                    comparison_cost = len(alternatives) * (
                        5 * rows + len(selected_training) + len(selected_validation) + 2 * rows)
                    required_compute = comparison_cost + 4 * len(selected_validation) + confirmation_rows
                    if budget.remaining < required_compute:
                        actions.append({"action": "defer_mechanism_experiment",
                            "reason": "unfunded_repeat_comparison_and_confirmation",
                            "compute_required": required_compute, "compute_remaining": budget.remaining})
                        charge_research_followup(action_origin, "mechanism_experiment", before_costs)
                        return
                observed = acquire(selected_channels, "distinguish_mechanisms", probe=plan["probe"])
                if observed is not None:
                    actual = paired_for(selected_channels)[-1]
                    repeated = None
                    if repeat_response:
                        repeated = acquire(selected_channels, "repeat_mechanism_predictions",
                            probe=Probe(f"repeat_mechanism_{serial}", plan["probe"].sequence))
                        noise_replications[(tuple(selected_channels), plan["probe"].probe_id)] = (observed, repeated)
                        repeated_probe = observation_probes[id(repeated[0])]
                        noise_replications[(tuple(selected_channels), repeated_probe.probe_id)] = (repeated, observed)
                    evidence = assess_competing_object_probe(plan, actual, budget,
                        tolerance=self.config.minimum_improvement,
                        **({"repeated_rows": repeated} if repeat_response else {}))
                    actions.append({"action": "test_mechanism_predictions", "probe_id": plan["probe"].probe_id,
                        "plan": {key: value for key, value in plan.items() if key != "probe"},
                        "evidence": evidence,
                        "alternatives": [{"candidate_id": item["candidate"].candidate_id,
                            "kind": item["kind"], "revision": item.get("mechanism_revision")}
                            for item in alternatives]})
                    if repeated is not None:
                        actions[-1]["repeat_probe_id"] = observation_probes[id(repeated[0])].probe_id
                    if self.config.use_memory:
                        tested_entries = {item["candidate"].candidate_id: item for item in alternatives}
                        for hypothesis, outcome in zip(plan["hypotheses"], evidence["hypotheses"]):
                            tested = Program.from_dict(hypothesis["program"])
                            entry = tested_entries[hypothesis["candidate_id"]]
                            candidate = replace(entry["candidate"], program=tested)
                            sources = list(dict.fromkeys([*_sources(tested), *candidate.parents,
                                *entry.get("method_source_ids", [])]))
                            record = self.library.record_transfer_counterexample(spec, candidate,
                                sources, evidence=outcome, probe_id=plan["probe"].probe_id)
                            if record is not None:
                                actions.append({"action": "record_transfer_counterexample", **record})
                    original_outcome = next((item for item in evidence["hypotheses"]
                        if item["candidate_id"] == source["candidate"].candidate_id), None)
                    if original_outcome and original_outcome["status"] == "counterexample":
                        transfer.append({"source_ids": revision["source_ids"], "program": original.to_dict(),
                            "validation_loss": original_outcome["measured_loss"],
                            "validation_rows": len(observed), "status": "counterexample", "accepted": False,
                            "kind": "mechanism_counterexample", "probe_id": plan["probe"].probe_id})
                    direction_counterexample(selected, plan, evidence)
                    measured_rows = [*observed, *(repeated or [])]
                    inherited_relation_evidence.extend(measured_rows)
                    if not selected_channels:
                        directed_command_rows.extend(measured_rows)
                    paid = [*selected_training, *selected_validation, *measured_rows]
                    scored = []
                    for item, outcome in zip(alternatives, evidence["hypotheses"]):
                        if outcome["measured_loss"] is None or not budget.spend(len(paid)):
                            continue
                        measured = loss(item["candidate"].program, paid, 0., spec.risk_quantile)
                        # A fresh targeted failure must not disappear inside
                        # the larger bank of easier, previously fitted rows.
                        compared = max(measured, outcome["measured_loss"])
                        checked = {**item, "training_loss": measured, "validation_loss": compared,
                            "selection_basis": "all_acquired_evidence"}
                        if item.get("mechanism_revision"):
                            checked["mechanism_revision"] = {**item["mechanism_revision"],
                                "status": outcome["status"], "probe_id": plan["probe"].probe_id,
                                "measured_loss": outcome["measured_loss"]}
                        scored.append(checked)
                    best = self._best(scored)
                    if best is not None:
                        selected, finalists = best, scored
                        selected_training = paid
                        confirmed = False
                        if action_origin is not None:
                            selected["research_action_origin"] = action_origin
                        actions.append({"action": "select_mechanism_revision", "selected_kind": selected["kind"],
                            "candidate_id": selected["candidate"].candidate_id,
                            "selection_loss": selected["validation_loss"],
                            "requires_fresh_confirmation": True})
                        outcome = next(item for item in evidence["hypotheses"]
                            if item["candidate_id"] == selected["candidate"].candidate_id)
                        replication = outcome.get("noise_replication") or {}
                        if repeated is not None and replication.get("noise_supported"):
                            # A noise-compatible, unresolved short experiment
                            # is not a structural refutation. Reassess the older
                            # paid repeat for this predictor and leave adoption
                            # to an independent confirmation trajectory.
                            test_response_noise(selected, selected_validation)
                    if evidence["calibration_status"] == "counterexample":
                        inspect_relation_inputs(selected, plan, evidence)
                    elif self.config.use_memory:
                        feedback = self.library.program_experiment_methods.observe(plan, evidence,
                            task_id=f"{laboratory.spec.task_id}:{self.config.seed}", command_spec=command_spec,
                            binding=binding, tolerance=self.config.minimum_improvement)
                        if feedback is not None:
                            actions.append(feedback)
            charge_research_followup(action_origin, "mechanism_experiment", before_costs)

        def research_selected_program_conditions():
            nonlocal selected, selected_training, finalists
            research_selected_mechanisms()
            if (selected is None or selected.get("structure") or confirmed or
                    not self.config.active_probes or self.config.strategy == "fixed"):
                return
            origin = deepcopy(selected)
            action_origin = selected.get("research_action_origin")
            before_costs = (spent_observations, spent_measurements, budget.spent)

            def settle_program_costs():
                if action_origin is not None:
                    selected["research_action_origin"] = action_origin
                charge_research_followup(action_origin, "program_condition_research", before_costs)

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
                    if (spent_observations + probe_reserve + length > observation_limit or
                        spent_measurements + (probe_reserve + length) * sum(rates[name] for name in selected_channels)
                            > measurement_limit):
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
                    direction_counterexample(selected, plan, evidence)
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
                        settle_program_costs()
                        return
                    if evidence["calibration_status"] == "counterexample":
                        research_instruments(selected_channels)
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
                    if action_origin is not None:
                        selected["research_action_origin"] = action_origin
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
                if (spent_observations + length + probe_reserve > observation_limit or
                        spent_measurements + (length + probe_reserve) * sum(rates[name] for name in selected_channels)
                        > measurement_limit):
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
                direction_counterexample(selected, plan, tested["evidence"])
                diagnosis = inspect_relation_inputs(origin, plan, tested["evidence"])
                selected_training.extend(observed)
                inherited_relation_evidence.extend(observed)
                if diagnosis is not None and diagnosis["status"] != "inputs_supported":
                    settle_program_costs()
                    return
                if relation_calls:
                    if refine_relation_space():
                        continue
                    relation_calls = []
                settle_program_costs()
                action_origin = None
                before = budget.spent
                revised = self._research_view(self._spec(laboratory, selected_channels), selected_training,
                    selected_validation, budget, actions, transfer, graph, parent=selected,
                    transitions=transitions_for(selected_channels),
                    observation_remaining=observation_limit - spent_observations,
                    measurement_remaining=measurement_limit - spent_measurements)
                if not revised:
                    break
                # The newly purchased counterexample participates in choice
                # even when the previous random validation missed that region.
                best_training = min(item["training_loss"] for item in revised)
                selected = self._best([item for item in revised
                    if item["training_loss"] <= best_training + self.config.minimum_improvement])
                finalists = revised
                action_origin = selected.get("research_action_origin")
                before_costs = (spent_observations, spent_measurements, budget.spent)
                actions.append({"action": "revise_from_program_experiment",
                    "observation_rows": len(observed), "compute_cost": budget.spent - before,
                    "training_loss": selected["training_loss"], "candidate_id": selected["candidate"].candidate_id})
            settle_program_costs()

        research_selected_structure()
        research_selected_program_conditions()

        pending_relation_candidates = []

        def research_inherited_relation_intervals():
            nonlocal selected, selected_training, finalists, confirmed, pending_relation_candidates
            nonlocal response_research_after_inputs, relation_inputs_unresolved, command_calls
            pending_relation_candidates = []
            if (selected is None or relation_inputs_unresolved or not self.config.active_probes or
                    self.config.strategy == "fixed"):
                return
            if response_research_after_inputs and object_actions is not None:
                response_research_after_inputs = False
                # Calibration has already been purchased and tested. Let the
                # existing action policy compare response/range repair with
                # measured structure research before fixing this failed law
                # and spending the remaining budget on all of its radii.
                object_actions.field_status = "inputs_revised"
                object_actions.attempted.add("remeasure_object_inputs")
                before = budget.spent
                recovered = object_actions.propose(selected, channels=selected_channels)
                adopted = adopt_observation_recovery(recovered)
                actions.append({"action": "research_response_after_input_revision",
                    "status": "requires_fresh_confirmation" if adopted else "no_improving_proposal",
                    "compute_cost": budget.spent - before,
                    "selected_channels": list(selected_channels),
                    "selected_kind": selected["kind"]})
                if not adopted:
                    return
                relation_inputs_unresolved = False
                command_calls = self._recall(self._spec(laboratory, ()))
                research_selected_structure()
                research_selected_program_conditions()
            from .relation_uncertainty import RelationIntervalResearch
            from .competing_object_experiments import (plan_competing_object_probe, assess_competing_object_probe,
                shorten_discriminating_probe)
            origin = deepcopy(selected)
            action_origin = selected.get("research_action_origin")
            before_costs = (spent_observations, spent_measurements, budget.spent)
            start = budget.spent
            research = RelationIntervalResearch(selected["candidate"].program, self.library.meta_priors,
                budget, structure=selected.get("structure"), research_program=selected.get("research_program"))
            if not research.parameters:
                charge_research_followup(action_origin, "inherited_relation_research", before_costs)
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

            def calibrate_relation_inputs(required):
                from .instrument_models import rebind_instrument_model
                model = research_instruments(selected_channels, reserve_rows=length, required_fields=required)
                current = self._instrument_binding(spec, command_spec)
                if model is not None and model.get("status") != "counterexample":
                    # Task-local learning remains usable in the no-memory arm.
                    current.update(rebind_instrument_model(model, spec))
                return current

            # A successor-funded experiment remains its own test after a
            # checkpoint reload; pooling it with older rows can hide a refutation.
            successor_probes = {event["probe_id"] for event in actions
                if event["action"] == "test_successor_object_experiment"
                and event["evidence"]["status"] == "distinguished"}
            distinguishing_trajectories = [tuple(view_rows(restore_batch(index),
                tuple(batch["channels"]), selected_channels))
                for index, batch in enumerate(batches)
                if batch["probe"]["probe_id"] in successor_probes
                and selected_channels in {(), *views_from(tuple(batch["channels"]))}]
            conditioning_cost_per_row = 0
            while budget.remaining > (len(paid) + length) * 10:
                before = budget.spent
                updates_before = len(research.evidence_updates)
                candidates = research.alternatives(paid, budget, tolerance=self.config.minimum_improvement,
                                                   experiments=distinguishing_trajectories)
                # Refining a retained radius continues the same mechanism
                # revision; it must keep that question and its source lineage.
                if origin.get("mechanism_revision"):
                    for item in candidates:
                        item["mechanism_revision"] = deepcopy(origin["mechanism_revision"])
                        sources = list(dict.fromkeys([*item.get("method_source_ids", []),
                            *origin.get("method_source_ids", [])]))
                        item["method_source_ids"] = sources
                        item["candidate"] = replace(item["candidate"], parents=tuple(dict.fromkeys(
                            [*item["candidate"].parents, *sources])))
                conditioning_cost_per_row = max(conditioning_cost_per_row,
                    (budget.spent - before + len(paid) - 1) // max(1, len(paid)))
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
                        revise_direction_outcome(selected, status="refuted", basis="paid_relation_space_refuted",
                            counterexample_rows=len(paid))
                        charge_research_followup(action_origin, "inherited_relation_research", before_costs)
                        action_origin = None
                        repaired = self._research_view(spec, paid, [], budget, actions, transfer,
                            graph, parent=selected, transitions=transitions_for(selected_channels),
                            observation_remaining=observation_limit - spent_observations,
                            measurement_remaining=measurement_limit - spent_measurements)
                        if repaired:
                            selected, finalists = self._best(repaired), repaired
                            selected_training = list(paid)
                            # This replacement can introduce new relation cells
                            # after the earlier program experiments have ended.
                            # Test those questions before random confirmation.
                            research_selected_structure()
                            research_selected_program_conditions()
                    break
                candidate = self._best(candidates)
                if candidate["candidate"].program != selected["candidate"].program:
                    confirmed = False
                selected, finalists = candidate, candidates
                if origin.get("research_draft_id") is not None:
                    selected["research_draft_id"] = origin["research_draft_id"]
                pending_relation_candidates = list(candidates)
                if action_origin is not None:
                    selected["research_action_origin"] = action_origin
                research.adopt(selected)
                validation_ids = {id(row) for row in selected_validation}
                selected_training = [row for row in paid if id(row) not in validation_ids]
                questions = research.questions(candidates)
                current_domains = research.parameters[candidate["parameter_index"]]["radius_intervals"]
                if (len(current_domains) == 1 and current_domains[0][1] - current_domains[0][0] <= 1e-6
                        and any(item.get("kernel_prior_status") == "pending" for item in research.parameters)):
                    continue
                if not questions or all(len(item["radius_intervals"]) == 1 and
                    item["radius_intervals"][0][1] - item["radius_intervals"][0][0] <= 1e-6
                    for item in research.parameters):
                    pending_relation_candidates = []
                    break
                # A learned procedure may distinguish these laws with a short
                # prefix. Price that actual plan below, not a generic history
                # length that can prevent method retrieval altogether.
                minimum_rows = max(item["candidate"].program.minimum_history_depth + 1 for item in candidates)
                if (spent_observations + minimum_rows + length > observation_limit or
                    spent_measurements + (minimum_rows + length) * sum(rates[name] for name in selected_channels)
                        > measurement_limit):
                    break
                required = {name for item in candidates
                            for name, _ in item["candidate"].program.observed_variables()
                            if name != "__target__"} - set(command_spec.feature_names)
                if required - set(binding) and selected_channels:
                    # Already purchased command/measurement pairs can answer
                    # this calibration question before another trial is bought.
                    binding = calibrate_relation_inputs(required)
                plan = plan_competing_object_probe(command_spec, candidates, budget, binding=binding,
                    serial=serial, length=5, tolerance=self.config.minimum_improvement,
                    methods=self.library.program_experiment_methods if self.config.use_memory else None,
                    task_id=f"{laboratory.spec.task_id}:{self.config.seed}", questions=questions)
                plan = shorten_discriminating_probe(plan, budget, tolerance=self.config.minimum_improvement)
                actions.append({"action": "plan_inherited_relation_experiment",
                    **{key: value for key, value in plan.items()
                       if key not in ("probe", "hypotheses", "question_hypotheses", "predicted_fields")},
                    "probe_id": plan["probe"].probe_id if plan["probe"] else None})
                if plan["probe"] is None:
                    break
                if (spent_observations + plan["probe"].cost + length > observation_limit or
                    spent_measurements + (plan["probe"].cost + length) * sum(rates[name] for name in selected_channels)
                        > measurement_limit):
                    break
                # The last actual conditioning pass quotes the work required
                # to consume this purchase. Funding the probe alone can leave
                # its evidence unused and spend the final confirmation budget.
                update_cost = conditioning_cost_per_row * (len(paid) + plan["probe"].cost)
                assessment_cost = plan["probe"].cost * (
                    len(plan["hypotheses"]) + len(plan["predicted_fields"]) + 1)
                confirmation_cost = 3 * (len(selected_validation) + length)
                if budget.remaining < update_cost + assessment_cost + confirmation_cost:
                    actions.append({"action": "defer_relation_experiment",
                        "reason": "reserve_evidence_update_and_confirmation",
                        "update_cost": update_cost, "assessment_cost": assessment_cost,
                        "confirmation_cost": confirmation_cost, "compute_remaining": budget.remaining})
                    break
                observed = acquire(selected_channels, "distinguish_inherited_relation", probe=plan["probe"])
                if observed is None:
                    break
                actual = paired_for(selected_channels)[-1]
                evidence = assess_competing_object_probe(plan, actual, budget,
                    tolerance=self.config.minimum_improvement)
                pending_relation_candidates = list(candidates)
                actions.append({"action": "test_inherited_relation_predictions", "probe_id": plan["probe"].probe_id,
                    "plan": {key: value for key, value in plan.items() if key != "probe"}, "evidence": evidence})
                direction_counterexample(selected, plan, evidence)
                diagnosis = inspect_relation_inputs(origin, plan, evidence)
                if not selected_channels:
                    directed_command_rows.extend(pair[0] for pair in actual)
                if diagnosis is None and (plan["forecast_status"] == "requires_paired_measurements" or
                        evidence["calibration_status"] == "counterexample"):
                    binding = calibrate_relation_inputs(required)
                paid.extend(observed)
                selected_training.extend(observed)
                inherited_relation_evidence.extend(observed)
                if evidence["status"] == "distinguished":
                    # Keep the paid experimental trajectory's own loss test.
                    # Other fitting rows cannot restore a refuted alternative.
                    distinguishing_trajectories.append(tuple(observed))
                confirmed = False
                if diagnosis is not None:
                    paid.extend(diagnosis["response_rows"])
                    if diagnosis["status"] == "inputs_revised":
                        if response_research_after_inputs and object_actions is not None:
                            charge_research_followup(action_origin, "inherited_relation_research", before_costs)
                            research_inherited_relation_intervals()
                            return
                        # Discard inferences made with the refuted input map,
                        # then consume every paid row under the corrected map.
                        research = RelationIntervalResearch(selected["candidate"].program,
                            self.library.meta_priors, budget, structure=selected.get("structure"),
                            research_program=selected.get("research_program"))
                        # Measured-coordinate range evidence survives a command
                        # calibration change. Only command-derived exclusions
                        # must be withdrawn before consuming the repaired inputs.
                        if not selected_channels and research.reopen("observed_input_revision"):
                            actions.append({"action": "reopen_inherited_relation_evidence",
                                            **research.evidence_updates[-1]})
                        origin = deepcopy(selected)
                        binding = self._instrument_binding(spec, command_spec)
                        actions.append({"action": "restart_relation_parameters_after_input_revision",
                            "intervals": research.intervals, "evidence_rows": len(paid)})
                    elif diagnosis["status"] == "unresolved_inputs":
                        break

            if action_origin is not None:
                selected["research_action_origin"] = action_origin
            charge_research_followup(action_origin, "inherited_relation_research", before_costs)

        research_inherited_relation_intervals()

        response_noise = (selected or {}).get("response_noise_evidence", {})
        validation_probe = (observation_probes.get(id(selected_validation[0]))
                            if selected_validation else None)
        conditional_steps = {"discover_conditional_coordinates", "discover_composed_conditions"}
        noise_explains_response = bool(
            response_noise.get("supported")
            and response_noise["program"] == selected["candidate"].program.to_dict()
            and response_noise["channels"] == list(selected_channels)
            and validation_probe is not None
            and response_noise["probe_id"] == validation_probe.probe_id
            and response_noise["paid_observation_rows"] == spent_observations
            and max(selected.get("training_loss", float("inf")),
                    selected.get("validation_loss", float("inf")))
                <= response_noise["replication"]["repeatable_loss_limit"]
            and not any(step["operation"] in conditional_steps
                for step in selected.get("research_program", {}).get("selection_steps", [])))
        if noise_explains_response:
            # Keep the remaining observations for fresh confirmation and
            # response refitting. Unresolved conditional boundaries still
            # receive experiments; new evidence or a new predictor reopens
            # residual research rather than inheriting this local pause.
            actions.append({"action": "defer_conditional_experiments",
                "reason": "paid_repeat_explains_current_response_error",
                "probe_id": response_noise["probe_id"],
                "paid_observation_rows": spent_observations})
        if not selected_channels and not noise_explains_response:
            old_selected, old_rows = selected, len(selected_training)
            experiment_costs = (spent_observations, spent_measurements, budget.spent)
            selected, selected_training, finalists = self._research_conditional_experiments(
                self._spec(laboratory, ()), selected, selected_training, selected_validation,
                finalists, budget, actions, acquire=acquire,
                observation_allowance=observation_limit - spent_observations
                    - (length if len(laboratory.spec.feature_names) >= 4 else 2 * length),
                serial=serial)
            if len(selected_training) > old_rows:
                confirmed = False
                if old_selected is not None and old_selected.get("research_action_origin") is not None:
                    selected.setdefault("research_action_origin", old_selected["research_action_origin"])
                    charge_research_followup(old_selected["research_action_origin"],
                        "conditional_matched_experiments_and_boundary", experiment_costs)

        fallback = initial_prediction(selected_training, laboratory.spec.risk_quantile)
        if selected is None:
            selected = {"candidate": Candidate("unified:initial", Program("const", value=fallback), "initial"),
                        "kind": "initial", "family_id": "T02", "validation_loss": None}
        fallback = selected.get("fallback", fallback)
        from .condition_response_actions import (observed_condition_followup,
                                                research_observed_condition_competition)
        from .observed_condition_uncertainty import assess_observed_condition_uncertainty
        condition_spec = self._spec(laboratory, selected_channels)
        condition_rows = [row for batch in paired_for(selected_channels) for _, row in batch]
        before_condition_program = selected["candidate"].program
        before_condition_rows = len(condition_rows)
        if (laboratory.spec.risk_quantile == .5 and selected_channels
                and not _object_schema(condition_spec) and not condition_spec.context.get("object_model")):
            selected, selected_training, finalists = research_observed_condition_competition(
                self, condition_spec, selected, condition_rows or selected_training,
                selected_validation, finalists, budget, actions, acquire=acquire,
                remaining=lambda: {"observation_remaining": observation_limit - spent_observations,
                    "measurement_remaining": measurement_limit - spent_measurements},
                serial=serial, channels=selected_channels,
                measurement_rate=sum(rates[name] for name in selected_channels))
        if (selected["candidate"].program != before_condition_program
                or len(selected_training) > before_condition_rows
                or (selected.get("condition_uncertainty") or {}).get("status") in ("unresolved", "contradicted")):
            confirmed = False
        program = selected["candidate"].program
        rounds = (self.config.max_rounds if laboratory.spec.risk_quantile != .5
                  or object_actions is not None or selected_channels
                  else min(2, self.config.max_rounds))
        resumed_measured_choice = (deepcopy(resumed_incumbent[0]), resumed_incumbent[1],
            previous.get("confirmation_loss"), deepcopy(previous.get("noise_assessment"))) if resumed_incumbent else None
        response_refit_parent, response_confirmation_length = None, None
        response_questions = []

        def retain_response_question(parent, proposal, event, probe_ids):
            if (selected_channels or laboratory.spec.risk_quantile != .5
                    or event.get("gain") is None or event["gain"] <= self.config.minimum_improvement):
                return
            controls = set(laboratory.spec.feature_bounds)
            if any(name not in controls for entry in (parent, proposal)
                   for name, _ in entry["candidate"].program.observed_variables()):
                return
            from .object_continuation import encode_entry
            parent = encode_entry({**parent, "fallback": parent.get("fallback", fallback)})
            proposal = encode_entry({**proposal, "fallback": proposal.get("fallback", fallback)})
            if parent["candidate"]["program"] == proposal["candidate"]["program"]:
                return
            if any(item["parent"]["candidate"]["program"] == parent["candidate"]["program"]
                   and item["candidate"]["candidate"]["program"] == proposal["candidate"]["program"]
                   for item in response_questions):
                return
            question = {"question_id": f"response_question_{len(response_questions) + 1:04d}",
                "parent": parent, "candidate": proposal, "observed_gain": event["gain"],
                "uncertainty_margin": event["margin"],
                "reference_loss": event.get("reference_loss", event.get("before_loss")),
                "candidate_loss": event.get("candidate_loss", event.get("after_loss")),
                "rows": event.get("paired_rows", event.get("heldout_rows")),
                "probe_ids": list(probe_ids), "status": "pending", "pilot_only": True,
                "gains": [], "reference_losses": [], "candidate_losses": []}
            response_questions.append(question)
            actions.append({"action": "retain_response_question", "question_id": question["question_id"],
                "candidate_id": proposal["candidate"]["candidate_id"], "pilot_gain": event["gain"],
                "pilot_margin": event["margin"], "probe_ids": list(probe_ids)})

        def object_confirmation_quote(rows):
            evidence = len(directed_command_rows) + len(inherited_relation_evidence)
            # The policy settles both programs after the ordinary prediction
            # gate. These evaluations and the retained counterexamples belong
            # to the same confirmation, not to the next search allowance.
            return self._confirmation_compute_quote(
                self._spec(laboratory, selected_channels), rows,
                training_rows=len(selected_training), evidence_rows=evidence,
                rivals=len(pending_relation_candidates),
                calibration_rows=len(selected_validation) if laboratory.spec.risk_quantile != .5 else 0
            ) + 2 * rows + ((rows + evidence) if resumed_incumbent is not None else 0)

        workflow_native_recovery = False
        if structure_comparison_pending is not None:
            confirmed, rounds = False, 0
            actions.append({"action": "defer_frozen_confirmation",
                "reason": "paid_structure_comparison_pending",
                "pending": deepcopy(structure_comparison_pending)})
        for attempt in range(rounds if not confirmed else 0):
            if structure_comparison_pending is not None:
                confirmed = False
                actions.append({"action": "defer_frozen_confirmation",
                    "reason": "paid_structure_comparison_pending",
                    "pending": deepcopy(structure_comparison_pending)})
                break
            rejected_response_refit = None
            confirmation_origin = selected.get("research_action_origin")
            confirmation_costs = (spent_observations, spent_measurements, budget.spent)
            comparison_parent = None
            if risk_actions is not None and risk_actions.pending is not None:
                comparison_parent = risk_actions.comparison_parent
                calibrated = selected if comparison_parent is not None else None
            else:
                calibrated = self._tail_calibration(selected, selected_validation,
                    self._spec(laboratory, selected_channels), budget, actions)
            confirmation_channels = (object_actions.confirmation_channels(selected_channels)
                                     if object_actions is not None else selected_channels)
            if risk_actions is not None:
                confirmation_channels = risk_actions.confirmation_channels(selected_channels)
            if resumed_incumbent is not None:
                needed = {tuple(confirmation_channels), tuple(resumed_incumbent[1])}
                joint_roles = (object_research or {}).get("continuation", {}).get("confirmation") or {}
                if joint_roles.get("status") == "pending":
                    needed.add(offered)
                confirmation_channels = next(view for view in sorted(map(tuple, alternative_views),
                    key=lambda view: sum(rates[name] for name in view))
                    if needed <= {(), *map(tuple, laboratory.projectable_views(view))})
            next_rows = (response_confirmation_length or
                min(laboratory.spec.max_probe_length, length * (2 ** attempt)))
            scoring_cost = next_rows * (1 + (calibrated is not None)
                + (response_refit_parent is not None) + (resumed_incumbent is not None)
                + len(pending_relation_candidates))
            scoring_cost += len(directed_command_rows) + len(inherited_relation_evidence)
            if resumed_incumbent is not None:
                scoring_cost += len(directed_command_rows) + len(inherited_relation_evidence)
            held_confirmation_reserve = budget.release_reserve()
            if held_confirmation_reserve:
                actions.append({"action": "release_confirmation_reserve",
                    "compute_cost": 0, "reserved_compute": held_confirmation_reserve,
                    "reason": "fund_frozen_confirmation_before_search_exhaustion"})
            if budget.remaining < scoring_cost:
                actions.append({"action": "defer_frozen_confirmation",
                    "reason": "unfunded_frozen_comparisons", "compute_required": scoring_cost,
                    "compute_remaining": budget.remaining, "observation_rows": next_rows})
                charge_research_followup(confirmation_origin, "frozen_confirmation", confirmation_costs)
                break
            if workflow is not None:
                workflow.frozen([selected, calibrated, comparison_parent, response_refit_parent],
                    selected_training, selected_validation, self._spec(laboratory, selected_channels))
            acquired_confirmation = acquire(confirmation_channels, "confirm_frozen_program", finalists,
                rows_count=next_rows)
            response_confirmation_length = None
            if acquired_confirmation is None:
                charge_research_followup(confirmation_origin, "frozen_confirmation", confirmation_costs)
                break
            confirmation = (view_rows(acquired_confirmation, confirmation_channels, selected_channels)
                            if confirmation_channels != selected_channels else acquired_confirmation)
            selected, confirmation_loss = self._compare_calibration(comparison_parent or selected, calibrated,
                confirmation, self._spec(laboratory, selected_channels), budget, fallback, actions)
            if confirmation_loss is None:
                charge_research_followup(confirmation_origin, "frozen_confirmation", confirmation_costs)
                break
            if response_refit_parent is not None and budget.spend(len(confirmation)):
                old_loss = loss(response_refit_parent["candidate"].program, confirmation,
                    response_refit_parent.get("fallback", fallback), laboratory.spec.risk_quantile)
                revised = selected
                retained = old_loss <= confirmation_loss + self.config.minimum_improvement
                actions.append({"action": "confirm_noisy_response_refit",
                    "before_loss": old_loss, "after_loss": confirmation_loss,
                    "accepted": not retained, "observation_rows": len(confirmation),
                    "probe_id": observation_probes[id(confirmation[0])].probe_id})
                if retained:
                    # Both responses were frozen before this new trajectory.
                    # Retain the stronger response and assess its noise anew.
                    revised_loss = confirmation_loss
                    selected, confirmation_loss = response_refit_parent, old_loss
                    rejected_response_refit = (revised, revised_loss, fallback)
                response_refit_parent = None
            joint_roles_supported = None
            retained_evidence_scores = {}
            if object_continuation:
                from .object_induction_continuation import confirm_joint_object_roles
                joint_roles_supported = confirm_joint_object_roles(self, laboratory, offered,
                    object_research, batches, budget, actions)
            if resumed_incumbent is not None:
                old_entry, old_channels = resumed_incumbent
                old_rows = view_rows(acquired_confirmation, confirmation_channels, old_channels)
                old_fallback = old_entry.get("fallback", previous["fallback"])
                if budget.spend(len(old_rows)):
                    old_loss = loss(old_entry["candidate"].program, old_rows, old_fallback,
                                    laboratory.spec.risk_quantile)
                    actions.append({"action": "compare_object_continuation_candidates",
                        "probe_id": observation_probes[id(acquired_confirmation[0])].probe_id,
                        "baseline_program": old_entry["candidate"].program.to_dict(),
                        "baseline_channels": list(old_channels), "baseline_fallback": old_fallback,
                        "program": selected["candidate"].program.to_dict(), "channels": list(selected_channels),
                        "fallback": selected.get("fallback", fallback), "before_loss": old_loss,
                        "after_loss": confirmation_loss, "risk_quantile": laboratory.spec.risk_quantile,
                        "observation_rows": len(old_rows)})
                    retention = compare_resumed_programs(old_entry["candidate"].program,
                        selected["candidate"].program, before_loss=old_loss,
                        after_loss=confirmation_loss, incumbent_fallback=old_fallback,
                        current_fallback=selected.get("fallback", fallback),
                        directed_command_rows=directed_command_rows,
                        inherited_relation_evidence=inherited_relation_evidence,
                        feature_names=laboratory.spec.feature_bounds, budget=budget,
                        minimum_improvement=self.config.minimum_improvement,
                        risk_quantile=laboratory.spec.risk_quantile,
                        external_experiment=(continuation_candidate or {}).get("experiment") is not None,
                        incumbent_refuted=successor_refuted_incumbent)
                    actions.append({"action": "compare_continuation_paid_evidence", **retention,
                        "joint_roles_supported": joint_roles_supported})
                    if retention["status"] != "compared":
                        confirmed = False
                        charge_research_followup(confirmation_origin, "frozen_confirmation", confirmation_costs)
                        break
                    retained_evidence_scores = retention["checks"]["current"]
                    if joint_roles_supported is False or retention["retain_incumbent"]:
                        selected, selected_channels = deepcopy(old_entry), old_channels
                        confirmation, confirmation_loss, fallback = old_rows, old_loss, old_fallback
                        retained_evidence_scores = retention["checks"]["incumbent"]
                        # The retained model may use a different measurement
                        # view. Reuse its paid rows, keeping this new trajectory
                        # out of residual fitting and future recalibration.
                        old_batches = ([[row for _, row in batch] for batch in paired_for(old_channels)]
                                       if old_channels else current_command_batches())
                        selected_training = [row for batch in old_batches[:-1] for row in batch]
                        selected_validation = list(selected_training)
                        finalists = [selected]
                else:
                    charge_research_followup(confirmation_origin, "frozen_confirmation", confirmation_costs)
                    break
            program = selected["candidate"].program
            fallback = selected.get("fallback", fallback)
            confirmation_rows = len(confirmation)
            confirmed = confirmation_loss <= self.config.minimum_improvement
            known_counterexample = self._object_program_refuted(selected,
                self._spec(laboratory, selected_channels))
            response_counterexamples = []
            if known_counterexample:
                confirmed = False
                actions.append({"action": "retain_object_counterexample",
                    "reason": "later_random_agreement_does_not_resolve_observed_failure",
                    "hypothesis_id": selected.get("hypothesis_id"),
                    "confirmation_loss": confirmation_loss})
            if directed_command_rows and {name for name, _ in program.observed_variables()} <= (
                    set(laboratory.spec.feature_bounds) | {"__target__"}):
                saved = retained_evidence_scores.get("program_experiment")
                checked = saved is not None or budget.spend(len(directed_command_rows))
                directed_loss = (saved["loss"] if saved is not None else
                    loss(program, directed_command_rows, fallback, laboratory.spec.risk_quantile) if checked else None)
                if directed_loss is None or directed_loss > self.config.minimum_improvement:
                    confirmed = False
                    response_counterexamples.append(("program_experiment", directed_command_rows, ()))
                    actions.append({"action": "retain_program_experiment_counterexample",
                        "loss": directed_loss, "observation_rows": len(directed_command_rows),
                        "confirmation_loss": confirmation_loss})
            required = {name for name, _ in program.observed_variables()} - {"__target__"}
            relevant = [row for row in inherited_relation_evidence if required <= set(row.features)]
            if relevant:
                saved = retained_evidence_scores.get("inherited_relation")
                checked = saved is not None or budget.spend(len(relevant))
                inherited_loss = (saved["loss"] if saved is not None else
                    loss(program, relevant, fallback, laboratory.spec.risk_quantile) if checked else None)
                if inherited_loss is None or inherited_loss > self.config.minimum_improvement:
                    confirmed = False
                    response_counterexamples.append(("inherited_relation", relevant, selected_channels))
                    actions.append({"action": "retain_inherited_relation_counterexample",
                        "loss": inherited_loss, "observation_rows": len(relevant),
                        "confirmation_loss": confirmation_loss})
            noisy_support = None
            relation_ambiguous = False
            if response_counterexamples and laboratory.spec.risk_quantile != .5:
                known_counterexample = True
            if confirmed and pending_relation_candidates:
                from .equivalent_object_programs import canonical_object_program
                chosen = canonical_object_program(program)
                survivors = []
                for entry in pending_relation_candidates:
                    alternative = entry["candidate"].program
                    if canonical_object_program(alternative) == chosen:
                        continue
                    if not budget.spend(len(confirmation)):
                        relation_ambiguous = True
                        break
                    if loss(alternative, confirmation, fallback, laboratory.spec.risk_quantile) <= self.config.minimum_improvement:
                        survivors.append(entry["candidate"].candidate_id)
                relation_ambiguous |= bool(survivors)
                if relation_ambiguous:
                    confirmed = False
                    actions.append({"action": "retain_relation_competition",
                        "reason": "relation_experiment_left_candidates_unresolved",
                        "surviving_candidates": survivors, "observation_rows": len(confirmation)})
            instrument_views_tested |= tuple(confirmation_channels) == offered
            if (not confirmed and not relation_ambiguous and not known_counterexample
                    and self.config.risk_learning and instrument_views_tested):
                if risk_actions is not None and tuple(confirmation_channels) == offered:
                    # Test unexplained output with all purchased coordinates.
                    # A predictor's cheaper deployment view cannot hide a
                    # measurable residual from the noise assessment.
                    complete_training = [row for batch in paired_for(confirmation_channels)[:-1]
                                         for _, row in batch] if confirmation_channels else [
                                             row for batch in current_command_batches()[:-1] for row in batch]
                    noisy_support = self._support(program, complete_training, acquired_confirmation,
                        self._spec(laboratory, confirmation_channels), budget)
                    noisy_support = challenge_noise(noisy_support, program, acquired_confirmation,
                        selected_channels, confirmation_channels, fallback,
                        reserve_rows=min(laboratory.spec.max_probe_length, length * (2 ** (attempt + 1))))
                else:
                    support_budget = budget
                    following_rows = min(laboratory.spec.max_probe_length, length * (2 ** (attempt + 1)))
                    if object_actions is not None:
                        paid_count = (sum(len(batch) for batch in paired_for(selected_channels))
                            if selected_channels else sum(map(len, current_command_batches())))
                        # Consume any returned answer on the paid pool and
                        # independently confirm it before another wide search.
                        newly_retained = len(confirmation) * (1 + int(resumed_incumbent is not None))
                        finish = (2 * len(confirmation) + object_confirmation_quote(following_rows)
                            + newly_retained
                            + paid_count * (len(finalists) + 5))
                        support_budget = SearchBudget(min(90000, max(0, budget.remaining - finish)))
                        actions.append({"action": "allocate_object_residual_question",
                            "compute_allowance": support_budget.limit, "reserved_finish_compute": finish,
                            "next_confirmation_rows": following_rows})
                    noisy_support = self._support(program, selected_training, confirmation,
                        self._spec(laboratory, selected_channels), support_budget)
                    if support_budget is not budget:
                        budget.spend(support_budget.spent)
                    noisy_support = challenge_noise(noisy_support, program, confirmation,
                        selected_channels, selected_channels, fallback, reserve_rows=following_rows)
                confirmed = noisy_support["adequate"]
            if response_counterexamples:
                replicated_noise = (laboratory.spec.risk_quantile == .5 and confirmed
                    and noisy_support and noisy_support.get("replication", {}).get("noise_supported"))
                confirmed = bool(replicated_noise and reassess_noisy_counterexamples(
                    program, response_counterexamples, fallback))
                if not confirmed and noisy_support:
                    noisy_support = {**noisy_support, "adequate": False,
                                     "reason": "retained_response_counterexample"}
            actions.append({"action": "validate_frozen_program", "attempt": attempt,
                "program": program.to_dict(), "loss": confirmation_loss, "accepted": confirmed,
                "noise_assessment": noisy_support,
                "observation_rows": len(confirmation)})
            if alternative_views and not confirmed:
                object_refutations.append({"program": program.to_dict(), "channels": list(selected_channels),
                    "probe_id": observation_probes[id(acquired_confirmation[0])].probe_id,
                    "loss": confirmation_loss, "observation_rows": len(confirmation)})
                inherited_relation_evidence.extend(confirmation)
            if resumed_incumbent is not None:
                resumed_measured_choice = (deepcopy(selected), selected_channels, confirmation_loss,
                                           deepcopy(noisy_support))
            charge_research_followup(confirmation_origin, "frozen_confirmation", confirmation_costs)
            if rejected_response_refit is not None:
                rejected, rejected_loss, rejected_fallback = rejected_response_refit
                revise_direction_outcome(rejected, status="refuted", basis="fresh_frozen_confirmation",
                    rows=confirmation, measured_loss=rejected_loss, program_fallback=rejected_fallback,
                    probe_id=observation_probes[id(confirmation[0])].probe_id)
            revise_direction_outcome(selected, status=("supported" if confirmed else
                "unresolved" if relation_ambiguous else "refuted"),
                basis="fresh_frozen_confirmation", rows=confirmation, measured_loss=confirmation_loss,
                program_fallback=fallback,
                probe_id=observation_probes[id(confirmation[0])].probe_id if confirmation else None)
            if object_actions is not None:
                object_actions.settle(selected, acquired_confirmation, confirmation_channels,
                    selected_channels=selected_channels, confirmed=confirmed, fallback=fallback,
                    probe_id=observation_probes[id(confirmation[0])].probe_id if confirmation else None)
            if risk_actions is not None:
                risk_actions.settle(selected, acquired_confirmation, confirmation_channels,
                    selected_channels=selected_channels, confirmed=confirmed, fallback=fallback,
                    probe_id=observation_probes[id(confirmation[0])].probe_id if confirmation else None)
            # A noisy observation does not establish that the acquired response
            # coefficients are optimal, especially inside a newly chosen window.
            # Fit only the old training rows; choose on the paid paired evidence;
            # reserve another complete trajectory and repeat before adopting.
            replication = (noisy_support or {}).get("replication", {})
            if (attempt + 1 < rounds and laboratory.spec.risk_quantile == .5
                    and self.config.strategy != "fixed" and not known_counterexample
                    and not relation_ambiguous and (replication.get("repeat_mse") or 0.)
                        > self.config.minimum_improvement):
                next_length = min(laboratory.spec.max_probe_length, length * (2 ** (attempt + 1)),
                    (observation_limit - spent_observations) // 2)
                rate = sum(rates[name] for name in selected_channels)
                if rate:
                    next_length = min(next_length, int((measurement_limit - spent_measurements) // (2 * rate)))
                confirmation_reserve = 18000 + 100 * next_length
                probe = observation_probes[id(confirmation[0])]
                pair = noise_replications.get((tuple(selected_channels), probe.probe_id))
                if next_length >= 16 and budget.remaining > confirmation_reserve and pair:
                    from .mechanism_revision import propose_response_refits
                    from .noise_replication import compare_response_refits
                    local = SearchBudget(min(8000, (budget.remaining - confirmation_reserve) // 2))
                    fit_start = budget.spent
                    proposals = propose_response_refits(self._spec(laboratory, selected_channels),
                        selected_training, local, parent=selected,
                        priors=self.library.meta_priors if self.config.use_memory else ())
                    budget.spend(local.spent)
                    actions.append({"action": "propose_noisy_response_refits", "candidates": len(proposals),
                        "training_rows": len(selected_training), "compute_cost": local.spent})
                    compared = compare_response_refits(program, proposals, *pair, budget,
                        fallback=fallback, minimum_improvement=self.config.minimum_improvement)
                    actions.extend({"action": "compare_noisy_response_refit", **event}
                                   for event in compared["comparisons"])
                    for event in compared["comparisons"]:
                        proposal = next(entry for entry in proposals
                            if entry["candidate"].candidate_id == event["candidate_id"])
                        retain_response_question(selected, proposal, event,
                            [probe.probe_id, observation_probes[id(pair[1][0])].probe_id])
                    proposed = compared["selected"]
                    refit_training = None
                    update_origin = None
                    update_selected = False
                    if proposed is None and confirmed and replication.get("noise_supported"):
                        from .mechanism_revision import propose_response_update
                        local = SearchBudget(min(8000, max(0, budget.remaining - confirmation_reserve)))
                        update = propose_response_update(self._spec(laboratory, selected_channels),
                            selected_training, *pair, local, parent=selected, fallback=fallback,
                            priors=self.library.meta_priors if self.config.use_memory else (),
                            minimum_improvement=self.config.minimum_improvement)
                        budget.spend(local.spent)
                        actions.append({"action": "propose_noisy_response_update", **update["event"]})
                        if update["selected"] is not None:
                            retain_response_question(selected, update["selected"], update["event"],
                                [probe.probe_id, observation_probes[id(pair[1][0])].probe_id])
                            policy = ResearchActionPolicy(selected_training,
                                self._spec(laboratory, selected_channels),
                                self.library.action_experience if self.config.use_memory else {})
                            policy.set_condition("noisy_response")

                            def operators(node):
                                return [node.op, *(op for child in node.children for op in operators(child))]

                            policy.set_action_knowledge("response_update", {
                                "kind": selected["kind"], "history_depth": program.history_depth,
                                "operators": sorted(set(operators(program.expanded())))})
                            option = policy.choose([ActionOption("response_update", 2 * next_length,
                                confirmation_reserve, measurement_cost=2 * next_length * rate)],
                                observation_remaining=observation_limit - spent_observations,
                                measurement_remaining=measurement_limit - spent_measurements,
                                compute_remaining=budget.remaining,
                                minimum_improvement=self.config.minimum_improvement)
                            actions.append({"action": "choose_noisy_response_update", **policy.last_decision,
                                "pilot_gain": update["event"]["gain"],
                                "pilot_margin": update["event"]["margin"]})
                            if option is not None:
                                proposed, update_selected = update["selected"], True
                                update_origin = {"context": policy.action_context("response_update"),
                                    "action": "response_update"}
                    if (proposed is None and replication.get("repeatable_error")
                            and budget.remaining > confirmation_reserve):
                        # A measured failure can justify a fresh coefficient
                        # trial even when refitting the old data changes nothing.
                        # Use the newly purchased responses for fitting; none
                        # of those rows now count as independent validation.
                        refit_training = list(selected_training)
                        refit_training.extend(row for rows in pair for row in rows
                                              if row not in refit_training)
                        local = SearchBudget(min(8000, budget.remaining - confirmation_reserve))
                        updated = propose_response_refits(self._spec(laboratory, selected_channels),
                            refit_training, local, parent=selected, max_candidates=1,
                            priors=self.library.meta_priors if self.config.use_memory else ())
                        old_fit_loss = (loss(program, refit_training, fallback, .5)
                            if local.spend(len(refit_training)) else None)
                        budget.spend(local.spent)
                        proposed = (updated[0] if updated and old_fit_loss is not None
                            and updated[0]["training_loss"] + self.config.minimum_improvement < old_fit_loss else None)
                        actions.append({"action": "propose_noise_measurement_refit",
                            "training_rows": len(refit_training), "before_loss": old_fit_loss,
                            "after_loss": updated[0]["training_loss"] if updated else None,
                            "compute_cost": local.spent, "requires_fresh_confirmation": proposed is not None})
                        if proposed is not None:
                            proposed.update(validation_loss=proposed["training_loss"],
                                            selection_basis="all_acquired_evidence")
                    if proposed is not None:
                        if refit_training is None and not update_selected:
                            proposed["validation_loss"] = next(event["candidate_loss"]
                                for event in compared["comparisons"]
                                if event["candidate_id"] == proposed["candidate"].candidate_id)
                        if self.config.use_memory:
                            if update_origin is None:
                                policy = ResearchActionPolicy(selected_training,
                                    self._spec(laboratory, selected_channels), self.library.action_experience)
                                context = policy.set_condition("noisy_response")
                                update_origin = {"context": context, "action": "response_refit"}
                            origin = {**update_origin, "trial": None,
                                "baseline_program": program.to_dict(), "fallback": fallback,
                                "risk_quantile": .5, "pending_id": len(actions), "pending_status": "unresolved",
                                "pending_costs": {"observation_cost": 0, "measurement_cost": 0,
                                                  "compute_cost": budget.spent - fit_start}}
                            proposed["research_action_origin"] = origin
                            actions.append({"action": "research_action_pending", "research_action": origin["action"],
                                "context": origin["context"], "pending_id": origin["pending_id"], "trial": None,
                                "evidence_basis": "repeated_response_pilot" if update_selected else
                                    "paired_response_selection" if refit_training is None
                                    else "new_response_fit_requires_validation", **origin["pending_costs"]})
                        actions.append({"action": "select_noisy_response_refit",
                            "program": proposed["candidate"].program.to_dict(), "parent_program": program.to_dict(),
                            "comparison_probe_id": probe.probe_id, "requires_fresh_confirmation": True})
                        response_refit_parent = selected
                        selected, program = proposed, proposed["candidate"].program
                        if refit_training is not None:
                            selected_training = refit_training
                        else:
                            selected_training.extend(confirmation)
                        response_confirmation_length = next_length
                        confirmed, noisy_support, confirmation_loss = False, None, None
                        continue
            condition_rows = [row for batch in paired_for(selected_channels) for _, row in batch]
            condition_state = assess_observed_condition_uncertainty(
                self._spec(laboratory, selected_channels), condition_rows, selected, budget,
                previous=selected.get("condition_uncertainty"), tolerance=self.config.minimum_improvement
                ) if (laboratory.spec.risk_quantile == .5 and selected_channels and object_actions is None
                      and (confirmed or selected.get("condition_uncertainty"))) else None
            selected["condition_uncertainty"] = condition_state
            condition_pending = condition_state is not None and condition_state["status"] != "supported"
            if condition_pending:
                confirmed = False
            if workflow is not None:
                workflow.tested(program, confirmation, confirmation_loss, accepted=confirmed)
            if (not confirmed and numeric_observation_research and workflow is not None
                    and workflow.replayed is not None and not workflow_native_recovery):
                # A transferred procedure proposes work, not a replacement for
                # observation research. Its first counterexample reopens the
                # existing controller with all paid evidence still available.
                workflow_native_recovery = True
                actions.append({"action": "reopen_workflow_observation_research",
                    "method_id": workflow.replayed["method_id"],
                    "confirmation_loss": confirmation_loss,
                    "condition_pending": condition_pending})
                recovered = recover_observation_actions(selected,
                    observation_probes[id(confirmation[0])])
                if adopt_observation_recovery(recovered):
                    program = selected["candidate"].program
                    fallback = selected.get("fallback",
                        initial_prediction(selected_training, laboratory.spec.risk_quantile))
                    if confirmed:
                        break
                    continue
            condition_followup = (observed_condition_followup(condition_state,
                self.config.minimum_improvement) if self.config.strategy != "fixed" else None)
            if condition_followup is not None:
                actions.append({"action": "choose_observed_response_followup",
                    "research_action": condition_followup,
                    "evidence_basis": "all_paid_original_response",
                    "paid_rows": len(condition_rows),
                    "paid_response_loss": condition_state["original_target_loss"],
                    "current_response_loss": confirmation_loss,
                    "observation_remaining": observation_limit - spent_observations,
                    "measurement_remaining": measurement_limit - spent_measurements,
                    "compute_remaining": budget.remaining})
            if condition_followup in ("revise_paid_condition", "distinguish_observed_condition"):
                actions.append({"action": "validate_observed_condition_scope",
                    "current_response_loss": confirmation_loss,
                    "status": condition_state["status"],
                    "expected_prediction_ambiguity": condition_state["expected_prediction_ambiguity"],
                    "compute_cost": condition_state["compute_cost"]})
                if attempt + 1 < rounds:
                    previous_condition_rows = len(condition_rows)
                    previous_condition_program = selected["candidate"].program
                    selected, selected_training, finalists = research_observed_condition_competition(
                        self, self._spec(laboratory, selected_channels), selected, condition_rows,
                        selected_validation, finalists, budget, actions, acquire=acquire,
                        remaining=lambda: {"observation_remaining": observation_limit - spent_observations,
                            "measurement_remaining": measurement_limit - spent_measurements},
                        serial=serial, channels=selected_channels,
                        measurement_rate=sum(rates[name] for name in selected_channels))
                    program = selected["candidate"].program
                    if (len(selected_training) > previous_condition_rows
                            or program != previous_condition_program):
                        response_confirmation_length = length
                        confirmation_loss, confirmation_rows, noisy_support = None, 0, None
                        continue
                break
            if condition_followup == "reconstruct_observed_response":
                # A fresh agreement cannot remove the earlier response
                # conflict. Fit the complete paid view before another replay.
                from .condition_response_actions import _observed_response_claims
                selected_training = condition_rows
                selected = {**selected, "training_loss": condition_state["original_target_loss"]}
                if attempt + 1 < rounds:
                    previous_condition_program = program
                    local = SearchBudget(max(0, budget.remaining - 4 * length))
                    proposals = _observed_response_claims(self,
                        self._spec(laboratory, selected_channels), selected_training,
                        selected, local, actions, selected_validation)
                    budget.spend(local.spent)
                    best = min([selected, *proposals], key=lambda item: (
                        max(item["training_loss"], item["validation_loss"]),
                        item["candidate"].program.size))
                    if (max(best["training_loss"], best["validation_loss"])
                            + self.config.minimum_improvement
                            < max(selected["training_loss"], selected["validation_loss"])):
                        selected, finalists = best, [best, selected, *finalists]
                    program = selected["candidate"].program
                    actions.append({"action": "reconstruct_paid_observed_response",
                        "evidence_rows": len(condition_rows), "measured_rows": 0,
                        "changed_program": program != previous_condition_program,
                        "compute_cost": local.spent,
                        "requires_fresh_confirmation": program != previous_condition_program})
                    if program != previous_condition_program:
                        response_confirmation_length = length
                        confirmation_loss, confirmation_rows, noisy_support = None, 0, None
                        continue
                break
            if confirmed or relation_ambiguous or self.config.strategy == "fixed" or joint_roles_supported is False:
                break
            # Preserve the actual failure as training evidence. Any changed
            # candidate needs another unqueried trajectory before adoption.
            selected_training.extend(confirmation)
            paired_evidence = noise_replications.get((tuple(selected_channels),
                observation_probes[id(confirmation[0])].probe_id)) if confirmation else None
            if paired_evidence is not None:
                # A failed noise explanation bought real additional responses.
                # Use them when refitting, rather than paying for a repeat only
                # to discard its independent response measurements.
                selected_training.extend(row for row in paired_evidence[1]
                                         if row not in selected_training)
            if attempt + 1 < rounds:
                if risk_actions is not None:
                    risk_actions.length = min(laboratory.spec.max_probe_length, length * (2 ** (attempt + 1)))
                    recovered = risk_actions.propose(selected, channels=selected_channels,
                        failed_probe=observation_probes[id(confirmation[0])], noise_assessment=noisy_support)
                    if adopt_observation_recovery(recovered):
                        program = selected["candidate"].program
                        fallback = selected.get("fallback",
                            initial_prediction(selected_training, laboratory.spec.risk_quantile))
                        continue
                    # Exhausted directions cannot silently restart the former
                    # fixed risk/reacquisition sequence outside the policy.
                    break
                if object_actions is not None:
                    next_confirmation = min(laboratory.spec.max_probe_length, length * (2 ** (attempt + 1)))
                    object_actions.confirmation_length = next_confirmation
                    quote = object_confirmation_quote(next_confirmation)
                    cover = object_actions.confirmation_channels(selected_channels)
                    # A later structure action may change instruments. Reserve
                    # a legal cover for every offered view before entering it.
                    rate = max(sum(rates[name] for name in view) for view in (cover, *alternative_views))
                    measurement_reserve = next_confirmation * rate
                    funded = (budget.remaining > quote
                        and observation_limit - spent_observations >= next_confirmation
                        and measurement_limit - spent_measurements >= measurement_reserve)
                    if not funded:
                        actions.append({"action": "defer_confirmation_revision",
                            "reason": "unfunded_object_revision_and_confirmation", "compute_required": quote,
                            "compute_remaining": budget.remaining, "observation_rows": next_confirmation,
                            "measurement_cost": measurement_reserve})
                    prior_reserve = budget.reserve
                    limits = observation_limit, measurement_limit
                    if funded:
                        budget.reserve += quote
                        actions.append({"action": "reserve_revision_confirmation",
                            "compute_cost": quote, "observation_rows": next_confirmation,
                            "measurement_cost": measurement_reserve, "confirmation_rows": next_confirmation})
                    recovered = None
                    try:
                        paid = ([row for batch in paired_for(selected_channels) for _, row in batch]
                            if selected_channels else [row for batch in current_command_batches() for row in batch])
                        answer = self._consume_residual_answer(
                            self._spec(laboratory, selected_channels), noisy_support, paid,
                            budget, actions, graph, parent=selected, finalists=finalists)
                        if answer["status"] == "improved":
                            selected = answer["selected"]
                            finalists = answer["claims"]
                            selected_training, selected_validation = list(paid), list(paid)
                            program, fallback = selected["candidate"].program, selected.get("fallback", fallback)
                            confirmation_loss, confirmation_rows, noisy_support = None, 0, None
                            if funded:
                                continue
                            break
                        if funded:
                            # ObjectResearchActions already reserves its own
                            # next_rows during acquisition; the ordinary nested
                            # callbacks below need the same outer reservation.
                            recovered = object_actions.propose(selected, channels=selected_channels,
                                failed_probe=observation_probes[id(confirmation[0])])
                            if adopt_observation_recovery(recovered):
                                if selected_channels or (recovered["object_action_origin"]["action"] == "remeasure_object_inputs"
                                                          and object_actions.field_status == "inputs_revised"):
                                    relation_inputs_unresolved = False
                                observation_limit -= next_confirmation
                                measurement_limit -= measurement_reserve
                                command_calls = self._recall(self._spec(laboratory, ()))
                                research_selected_structure()
                                research_selected_program_conditions()
                                research_inherited_relation_intervals()
                                program = selected["candidate"].program
                                fallback = initial_prediction(selected_training, laboratory.spec.risk_quantile)
                                continue
                    finally:
                        budget.reserve = prior_reserve
                        observation_limit, measurement_limit = limits
                    # Keep paid alternatives even when no complete new visit
                    # fits; using them to select still requires a fresh test.
                    retained, _ = self._retain_confirmation_alternative(selected, finalists,
                        confirmation, self._spec(laboratory, selected_channels), budget,
                        fallback, actions, incumbent_loss=confirmation_loss)
                    if retained is not selected:
                        selected = retained
                        program = selected["candidate"].program
                        fallback = selected.get("fallback", fallback)
                        confirmation_loss, confirmation_rows, noisy_support = None, 0, None
                    actions.append({"action": "stop_object_research",
                        "reason": "no_improving_affordable_proposal"})
                    break
                if numeric_observation_research and not selected_channels:
                    recovered = recover_observation_actions(selected,
                        observation_probes[id(confirmation[0])])
                    if adopt_observation_recovery(recovered):
                        program = selected["candidate"].program
                        fallback = initial_prediction(selected_training, laboratory.spec.risk_quantile)
                        if confirmed:
                            break
                        continue
                if object_actions is None and self._revise_inputs and not selected_channels and command_calls:
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
                        previous_origin = selected.get("research_action_origin")
                        if previous_origin and previous_origin["action"] == "program_revision":
                            best["research_action_origin"] = previous_origin
                        selected, finalists = best, assessed
                        research_selected_program_conditions()
                        program = selected["candidate"].program
                        continue
                if object_actions is None and self._learn_instruments and not selected_channels and offered:
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
                                transitions=transitions_for(names, train_view),
                                             observation_remaining=observation_limit - spent_observations,
                                             measurement_remaining=measurement_limit - spent_measurements)
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
                # A revision is useful only if its frozen replacement can be
                # independently scored. Keep that whole action funded while
                # every nested research callback shares the smaller limits.
                next_rows = min(laboratory.spec.max_probe_length, length * (2 ** (attempt + 1)))
                noise_check = (self.config.risk_learning and instrument_views_tested
                    and not _object_schema(self._spec(laboratory, selected_channels)))
                evidence_rows = len(directed_command_rows) + len(inherited_relation_evidence)
                pending_repeats = {}
                if noise_check:
                    for rows, channels in ((directed_command_rows, ()),
                            (inherited_relation_evidence, selected_channels)):
                        for row in rows:
                            probe = observation_probes.get(id(row))
                            if probe is not None:
                                key = (tuple(channels), probe.probe_id)
                                if key not in noise_replications:
                                    pending_repeats[key] = probe.cost
                repeat_rows = sum(pending_repeats.values())
                reserved_rows = next_rows * (2 if noise_check else 1) + repeat_rows
                # Research may add fitted rows and fresh disputed experiments.
                # Price their future checks too, using the available paid-row
                # ceiling rather than a percentage of the compute allowance.
                research_rows = max(0, observation_limit - spent_observations - reserved_rows)
                if noise_check:
                    # Every further disputed row may need its independent
                    # repeat before the replacement can be accepted.
                    repeat_views = 2 if selected_channels else 1
                    research_rows //= 1 + repeat_views
                    reserved_rows += repeat_views * research_rows
                training_ceiling = len(selected_training) + len(selected_validation) + research_rows
                evidence_ceiling = evidence_rows + 2 * research_rows
                quote = self._confirmation_compute_quote(
                    self._spec(laboratory, selected_channels), next_rows,
                    training_rows=training_ceiling, evidence_rows=evidence_ceiling,
                    rivals=len(pending_relation_candidates), noise_check=noise_check,
                    calibration_rows=len(selected_validation) if laboratory.spec.risk_quantile != .5 else 0)
                rate = sum(rates[name] for name in selected_channels)
                reserved_measurements = reserved_rows * rate
                if (observation_limit - spent_observations < reserved_rows
                        or measurement_limit - spent_measurements < reserved_measurements
                        or budget.remaining <= quote):
                    actions.append({"action": "defer_confirmation_revision",
                        "reason": "unfunded_revision_and_confirmation", "compute_required": quote,
                        "compute_remaining": budget.remaining, "observation_rows": reserved_rows,
                        "measurement_cost": reserved_measurements})
                    revised = self._refit_paid_response(
                        self._spec(laboratory, selected_channels), selected,
                        selected_training, selected_validation, budget, actions)
                    if revised is not None:
                        finalists = [revised, selected, *finalists]
                        selected = revised
                        program, fallback = revised["candidate"].program, revised["fallback"]
                        confirmation_loss, confirmation_rows, noisy_support = None, 0, None
                    break
                limits = budget.limit, observation_limit, measurement_limit
                budget.limit -= quote
                observation_limit -= reserved_rows
                measurement_limit -= reserved_measurements
                actions.append({"action": "reserve_revision_confirmation",
                    "compute_cost": quote, "observation_rows": reserved_rows,
                    "measurement_cost": reserved_measurements, "confirmation_rows": next_rows})
                try:
                    trained_ids = {id(row) for row in selected_training}
                    paid = [*selected_training, *(row for row in selected_validation if id(row) not in trained_ids)]
                    answer = self._consume_residual_answer(
                        self._spec(laboratory, selected_channels), noisy_support, paid,
                        budget, actions, graph, parent=selected, finalists=finalists)
                    if answer["status"] == "improved":
                        selected, finalists = answer["selected"], answer["claims"]
                        selected_training, selected_validation = list(paid), list(paid)
                        program, fallback = selected["candidate"].program, selected.get("fallback", fallback)
                        confirmation_loss, confirmation_rows, noisy_support = None, 0, None
                        continue
                    retained_pool = list(finalists)
                    finalists = self._research_view(self._spec(laboratory, selected_channels),
                        selected_training, selected_validation, budget, actions, transfer, graph,
                        parent=selected, transitions=transitions_for(selected_channels),
                        observation_remaining=observation_limit - spent_observations,
                        measurement_remaining=measurement_limit - spent_measurements)
                    revised = self._best(finalists)
                    if revised is not None:
                        selected = revised
                        research_selected_structure()
                        research_selected_program_conditions()
                        program = selected["candidate"].program
                    finalists.extend(item for item in retained_pool if item not in finalists)

                finally:
                    budget.limit, observation_limit, measurement_limit = limits

        unconfirmed_object_proposal = None
        if resumed_measured_choice is not None:
            unconfirmed_object_proposal = deepcopy(selected)
            unconfirmed_object_proposal["continuation_channels"] = list(selected_channels)
            selected, selected_channels, confirmation_loss, noisy_support = resumed_measured_choice
            program = selected["candidate"].program
            fallback = selected.get("fallback", previous["fallback"])
        direction_origin = selected.get("research_action_origin")
        if object_actions is not None:
            object_actions.settle(selected, None, selected_channels,
                selected_channels=selected_channels, confirmed=confirmed, fallback=fallback)
        if risk_actions is not None:
            risk_actions.settle(selected, None, selected_channels,
                selected_channels=selected_channels, confirmed=confirmed, fallback=fallback)
        if direction_origin is not None and not confirmed and direction_key(direction_origin) not in settled_direction_trials:
            revise_direction_outcome(selected, status="unresolved", basis="confirmation_unavailable")

        # The schema was available for constructing a joint proposal only.
        # If no fresh joint comparison ran, keep it pending for a later visit.
        joint_roles = (object_research or {}).get("continuation", {}).get("confirmation") or {}
        if joint_roles.get("status") == "pending":
            self._object_schemas.pop(offered, None)

        used_sources = (list(dict.fromkeys([*_sources(program), *selected.get("method_source_ids", [])]))
                        if confirmed else [])
        if selected.get("research_program") is not None:
            selected["research_program"]["status"] = ("supported_on_fresh_trajectory"
                                                       if confirmed else "unresolved")
            selected["research_program"]["confirmation_loss"] = confirmation_loss
            selected["research_program"]["confirmation_rows"] = confirmation_rows
        if confirmed and selected.get("mechanism_revision"):
            selected["mechanism_revision"] = {**selected["mechanism_revision"],
                "status": "supported_on_fresh_trajectory", "confirmation_loss": confirmation_loss,
                "confirmation_rows": confirmation_rows}
        for event in transfer:
            event["accepted"] = bool(confirmed and set(event["source_ids"]) & set(used_sources)
                                     and event["status"] == "supported")
        concepts_added = []
        if self.config.use_memory and self.config.strategy != "fixed":
            if not confirmed and selected.get("research_program"):
                from .research_drafts import remember_research_draft
                remember_research_draft(self.library, selected,
                    self._spec(laboratory, selected_channels), actions)
            for event in transfer:
                if event["status"] != "counterexample":
                    continue
                for prior in self.library.meta_priors:
                    if prior["source_concept_id"] in event["source_ids"]:
                        prior.setdefault("counterexamples", []).append({
                            "task_id": laboratory.spec.task_id,
                            "loss": event.get("confirmation_loss", event["validation_loss"]),
                            "kind": event.get("kind", "failed_role_binding"),
                            "program": event["program"], "replacement_concept_id": None})
                        prior["status"] = "conditional"
            if confirmed:
                spec = self._knowledge_spec(self._spec(laboratory, selected_channels))
                remembered = self.remember_confirmed_selection(selected, spec,
                    selected_training, selected_validation, confirmed=confirmed,
                    confirmation_loss=confirmation_loss, confirmation_rows=confirmation_rows,
                    selected_channels=selected_channels, batches=batches, actions=actions,
                    used_sources=used_sources, noisy_support=noisy_support,
                    offered_spec=self._spec(laboratory, offered))
                concept_id, concepts_added = remembered["concept_id"], remembered["concepts_added"]
        measurement_program, measurement_channels, instrument_model_id = None, None, None
        if confirmed and selected_channels and self._learn_instruments:
            from .instrument_models import rebind_instrument_model
            command_spec = self._spec(laboratory, ())
            measured_spec = self._spec(laboratory, selected_channels)
            model = research_instruments(selected_channels)
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
        if object_actions is not None:
            object_actions.finish()
        if risk_actions is not None:
            risk_actions.finish()
        if workflow is not None:
            workflow.diagnostics["transfer_condition"] = transfer_condition(
                self._spec(laboratory, ()),
                [row for batch in current_command_batches() for row in batch])
            concept = (next((item for item in self.library.concepts if item.concept_id == concept_id), None)
                       if confirmed else None)
            workflow.finish(selected["candidate"].program, confirmed=confirmed,
                threshold=self.config.minimum_improvement, actions=actions, concept=concept)
        costs = {"observation_cost": spent_observations,
                 "measurement_cost": spent_measurements, "compute_cost": budget.spent,
                 "queries": len(batches)}
        for question in response_questions:
            if confirmed and question["candidate"]["candidate"]["program"] == program.to_dict():
                question.update(status="settled", outcome="selected_on_first_visit")
        result = {"task_id": laboratory.spec.task_id,
                "research_workflow_id": workflow.method_id if workflow is not None else None,
                "status": "completed" if confirmed else "unresolved",
                "program": program.to_dict(), "expression": program.describe(),
                "fallback": fallback, "selected_kind": selected["kind"],
                "selected_structure": selected.get("structure"),
                "object_equivalence": selected.get("object_equivalence"),
                "object_competition": selected.get("object_competition"),
                "program_competitions": list(self._object_competitions.values()),
                "research_program": selected.get("research_program"),
                "condition_uncertainty": deepcopy(selected.get("condition_uncertainty")),
                "condition_research_returns": deepcopy(selected.get("condition_research_returns")),
                "research_draft_id": selected.get("research_draft_id"),
                "risk_method": deepcopy(selected.get("risk_method")),
                "mechanism_revision": deepcopy(selected.get("mechanism_revision")),
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
                "batches": batches, "response_questions": response_questions}
        if observation_alternatives or selected.get("condition_uncertainty"):
            result["continuation_state"] = {"alternatives": deepcopy(observation_alternatives),
                                             "condition_uncertainty": deepcopy(selected.get("condition_uncertainty")),
                                             "noise_assessment": deepcopy(noisy_support)}
        if alternative_views:
            from .object_continuation import encode_entry
            result["object_continuation_state"] = {
                "structure_comparison_pending": deepcopy(structure_comparison_pending),
                "selected": encode_entry({**selected, "candidate": replace(selected["candidate"], program=program)}),
                "finalists": [encode_entry({**entry, "continuation_channels": entry.get("continuation_channels",
                    list(channels_for(entry["candidate"].program.required_features())))}) for entry in [
                    *([unconfirmed_object_proposal] if unconfirmed_object_proposal is not None else []), *finalists]
                    if entry is not None][:self.config.max_candidates],
                "schemas": [{"channels": list(channels), "schema": deepcopy(schema)}
                            for channels, schema in self._object_schemas.items()],
                "models": [{"channels": list(channels), "model": deepcopy(model)}
                           for channels, model in self._object_models.items()],
                "competitions": [{"features": list(features), "competition": deepcopy(competition)}
                                 for features, competition in self._object_competitions.items()],
                "candidate_frontiers": [{"features": list(features), "entries": deepcopy(entries)}
                                        for features, entries in self._object_candidate_frontiers.items()],
                "directed_command_rows": [asdict(row) for row in directed_command_rows],
                "inherited_relation_evidence": [asdict(row) for row in inherited_relation_evidence],
                "refutations": object_refutations,
                "field_status": object_actions.field_status if object_actions else "unassessed_inputs"}
        if object_continuation:
            result["visit_costs"] = {key: value - old_costs.get(key, 0) for key, value in costs.items()}
            result["continuation"] = {"previous_status": previous["status"], "previous_costs": old_costs,
                "source_concept_ids": (continuation_candidate or {}).get("source_concept_ids", []),
                "source_object_model_ids": (continuation_candidate or {}).get("source_object_model_ids", []),
                "paid_rows_reused": old_costs["observation_cost"], "resumed_object_frontier": True}
        return result
