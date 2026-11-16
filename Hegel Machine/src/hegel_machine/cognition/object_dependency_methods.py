"""Revise executable membership experiments from their measured contrasts.

Recipes bind excitation roles to the current inferred controls. A flat pilot
changes the next experiment; it does not refute an object assignment or prove
that every other excitation would fail. Only the ordinary inducer assigns roles.
"""

from copy import deepcopy

from .models import Probe


class ObjectDependencyMethods:
    def __init__(self, sources=(), *, budget=None, actions=None):
        self.sources = sources
        self.budget = budget
        self.actions = actions if actions is not None else []
        self.evidence = []
        self.index = {}
        self.reset_by_probe = {}
        self.indexed_rows = 0
        self.pending = None
        self.last_recipe = None
        self.announced = set()

    @staticmethod
    def _key(sequence):
        return tuple(tuple(sorted(point.items())) for point in sequence)

    def _index(self, records):
        fresh = records[self.indexed_rows:]
        units = sum(len(row["controls"]) + 1 for row in fresh)
        if self.budget is not None and not self.budget.spend(units):
            return False
        for row in fresh:
            self.index[self._key([*row["history"], row["controls"]])] = row
            if row["position"] == 0 and not row["history"]:
                self.reset_by_probe[row["probe_id"]] = row
        self.indexed_rows = len(records)
        return True

    @staticmethod
    def _scope(records):
        resets = [row["target"] for row in records if row["position"] == 0]
        later = [row["target"] for row in records if row["position"] > 0]
        return {"silent_resets_with_history_signal": len(resets) >= 2
                and max(map(abs, resets), default=0.) <= 1e-8
                and max(map(abs, later), default=0.) > 1e-6}

    def _recipes(self, scope):
        retained = []
        # A contrast measured in this task outranks a recipe remembered from
        # another apparatus. Keep the other recipes available if this one
        # cannot distinguish the remaining memberships.
        for method in reversed(self.evidence):
            if method.get("status") != "discriminating" or method.get("scope") != scope:
                continue
            recipe = (method["excitation_role"], method["matched_prefix_rows"])
            if recipe not in [item[0] for item in retained]:
                retained.append((recipe, None))
        for source in reversed(self.sources):
            for method in source.get("dependency_methods", []):
                if method.get("status") != "discriminating" or method.get("scope") != scope:
                    continue
                recipe = (method["excitation_role"], method["matched_prefix_rows"])
                if recipe not in [item[0] for item in retained]:
                    retained.append((recipe, source.get("model_id")))
        # These are edits to the experiment, not predictions of its outcome.
        # Paid evidence of silent resets gives a common history priority over
        # another reset excitation when resets already have no response.
        defaults = [("drive", 0), ("quantity", 0), ("quantity", 1), ("drive", 1)]
        if scope["silent_resets_with_history_signal"]:
            defaults = [("quantity", 1), ("drive", 1), ("drive", 0), ("quantity", 0)]
        for recipe in defaults:
            if recipe not in [item[0] for item in retained]:
                retained.append((recipe, None))
        return retained

    def _assessment(self, recipe, quantity, pairs, scope):
        rows = [(self.index.get(self._key(a)), self.index.get(self._key(b))) for a, b in pairs]
        if not rows or any(a is None or b is None for a, b in rows):
            return None
        effects = [b["target"] - a["target"] for a, b in rows]
        moved = all(any(abs(b[phase].get(quantity, 0.) - a[phase].get(quantity, 0.)) > 1e-8
                        for phase in ("before", "after")) for a, b in rows)
        spread = max(effects) - min(effects)
        useful = moved and spread > 1e-7 * max(1., *map(abs, effects))
        result = {"excitation_role": recipe[0], "matched_prefix_rows": recipe[1],
            "quantity_candidate": quantity, "scope": scope,
            "status": "discriminating" if useful else "uninformative_pilot",
            "intervention_observed": moved, "response_effects": effects,
            "effect_spread": spread,
            "paired_probe_ids": [[a["probe_id"], b["probe_id"]] for a, b in rows]}
        existing = next((item for item in self.evidence
            if (item["excitation_role"], item["matched_prefix_rows"], item["quantity_candidate"])
               == (*recipe, quantity) and item["scope"] == scope
               and item["paired_probe_ids"] == result["paired_probe_ids"]), None)
        if existing is None:
            self.evidence.append(result)
            self.actions.append({"action": "assess_object_dependency_method", **deepcopy(result)})
        return result

    def observe(self, records):
        """Include the last purchased pair even when it completed induction."""
        if not self._index(records):
            return False
        if self.pending is not None:
            recipe, quantity, pairs, scope = self.pending
            self._assessment(recipe, quantity, pairs, self._scope(records))
        return True

    def snapshot(self):
        return deepcopy(self.evidence)

    def propose(self, spec, model, records, serial, selected_values, *, remaining_rows=None):
        if not self.observe(records):
            return None, {"purpose": "dependency_method_compute_exhausted"}
        bounds = spec.feature_bounds
        center = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
        selector = model["focus_control"]
        scope = self._scope(records)
        drives = {link["quantity"]: control for control, link in model.get("drive_links", {}).items()
                  if link.get("quantity") is not None}
        unresolved = {field: candidate for field, candidate in model["focus_candidates"].items()
                      if len(candidate["quantities"]) > 1}
        selected = {field: value for field, value in selected_values.items() if field in unresolved}
        # One unresolved indicator still needs an observed comparison state.
        if len(selected) == 1:
            selected.update(list((field, value) for field, value in selected_values.items()
                                 if field not in selected)[:1])
        needed = {quantity for candidate in unresolved.values() for quantity in candidate["quantities"]}
        deferred = []
        for recipe, source_id in self._recipes(scope):
            role, depth = recipe
            controls = drives if role == "drive" else model.get("quantity_controls", {})
            useful = any(item["excitation_role"] == role and item["matched_prefix_rows"] == depth
                         and item["status"] == "discriminating" and item["scope"] == scope
                         for item in self.evidence)
            # Once the recipe has an observed contrast, resolve the tightest
            # remaining pair first. If it stays ambiguous, continue with the
            # other states instead of repeatedly buying a full selector sweep.
            comparisons = [selected]
            if useful and len(selected) > 2:
                ordered = sorted(selected, key=lambda field: len(unresolved[field]["quantities"]))
                comparisons = []
                for offset in range(0, len(ordered), 2):
                    fields = ordered[offset:offset + 2]
                    if len(fields) == 1:
                        fields.append(ordered[0])
                    comparisons.append({field: selected[field] for field in fields})
            for comparison in comparisons:
                # A resolved quantity can still distinguish other memberships via
                # coupling. Prune selector comparisons, not those interventions.
                for quantity in sorted(controls, key=lambda q: q not in needed):
                    control = controls[quantity]
                    pairs = []
                    for value in comparison.values():
                        baseline = {**center, selector: value}
                        changed = {**baseline, control: bounds[control][1]}
                        prefix = [baseline] * depth
                        pairs.append(([*prefix, baseline], [*prefix, changed]))
                    result = self._assessment(recipe, quantity, pairs, scope)
                    if result is not None:
                        useful = useful or result["status"] == "discriminating"
                        if not useful:
                            break  # Revise a flat pilot before covering more excitations.
                        continue
                    missing = {self._key(step): step for contrast in pairs for step in contrast
                               if self._key(step) not in self.index}
                    pilot_rows = sum(len(step) for step in missing.values())
                    if remaining_rows is not None and pilot_rows > remaining_rows:
                        deferred.append({"excitation_role": role, "matched_prefix_rows": depth,
                            "quantity_candidate": quantity, "remaining_pilot_rows": pilot_rows})
                        continue
                    for field, pair in zip(comparison, pairs):
                        for sequence in pair:
                            if self._key(sequence) in self.index:
                                continue
                            if len(sequence) > spec.max_probe_length:
                                continue
                            if recipe != self.last_recipe:
                                if self.last_recipe is not None:
                                    self.actions.append({"action": "rewrite_object_dependency_method",
                                        "previous": {"excitation_role": self.last_recipe[0],
                                                     "matched_prefix_rows": self.last_recipe[1]},
                                        "revised": {"excitation_role": role, "matched_prefix_rows": depth},
                                        "reason": "measured_pilot_lacks_membership_contrast", "scope": scope})
                                self.last_recipe = recipe
                            if source_id and (recipe, source_id) not in self.announced:
                                self.announced.add((recipe, source_id))
                                self.actions.append({"action": "test_retained_dependency_method",
                                    "source_object_model_id": source_id, "excitation_role": role,
                                    "matched_prefix_rows": depth, "scope": scope})
                            self.pending = recipe, quantity, pairs, scope
                            question = {"purpose": "distinguish_membership_dependency",
                                "observation_rows": len(sequence), "selector_control": selector,
                                "selected_field": field, "intervention_control": control,
                                "quantity_candidate": quantity, "matched_prefix_rows": depth,
                                "response_question": "direct_response" if depth == 0 else "transition_response",
                                "research_program": {"excitation_role": role, "matched_prefix_rows": depth},
                                "remaining_pilot_rows": pilot_rows,
                                # A single missing corner cannot answer this pilot.
                                # Interpret all roles once its complete contrasts exist.
                                "interpret_dependencies": len(missing) == 1,
                                "competing_fields": list(unresolved)}
                            if source_id:
                                question["source_object_model_id"] = source_id
                            return Probe(f"object_intervention:{serial}", tuple(sequence)), question
        return None, {"purpose": "dependency_pilot_unaffordable" if deferred else
                                 "measured_dependency_methods_lack_contrast",
                      "unresolved_fields": list(unresolved), "deferred_pilots": deferred}


def form_position_owner_method(history, budget):
    """Abstract a two-stage position experiment from its frozen forecasts and measured returns."""
    start = budget.spent

    def result(status, **fields):
        return {'status': status, 'compute_cost': budget.spent - start, **fields}

    if len(history) != 2:
        return result('position_method_inapplicable', reason='two_measured_stages_required')
    plans = [stage['position_experiment'] for stage in history]
    answers = [stage['completed_relation'] for stage in history]
    if not budget.spend(sum(len(plan['candidate_owners']) + len(answer.get('corner_refs', ()))
                            for plan, answer in zip(plans, answers))):
        return result('position_method_compute_exhausted')
    first, second = plans
    before, after = answers
    if any(answer['status'] != 'position_domain_narrowed' for answer in answers):
        return result('position_method_inapplicable', reason='two_measured_narrowings_required')
    owners = set(before['retained_owners'])
    if (first['field'] != second['field'] or first['position_control'] != second['position_control']
            or before['source_model_id'] != after['source_model_id']
            or any(plan['field'] != answer['field']
                   or set(plan['candidate_owners']) != set(answer['candidate_owners'])
                   or plan['excited_quantity'] != answer['comparison']['excited_quantity']
                   for plan, answer in zip(plans, answers))
            or owners != set(second['candidate_owners']) or owners != set(second['alternative_owners'])
            or not owners < set(first['candidate_owners']) or len(owners) != 2):
        return result('position_method_inapplicable', reason='stages_do_not_continue_same_owner_pair')

    # Only the measured backgrounds common to both experiments can be reused.
    reused = [i for i, ref in enumerate(before['corner_refs'])
              if ref == after['corner_refs'][i] and ref['position'] == 0
              and first['corners'][i] == second['corners'][i]]
    excitation = second['excited_quantity']
    if reused != [0, 2] or excitation in owners or first['excited_quantity'] == excitation:
        return result('position_method_inapplicable', reason='different_excitation_on_shared_resets_required')
    forecasts = {q: set(second['owner_predictions'][q]['allowed_foreign_quantities']) for q in owners}
    left, right = forecasts.values()
    observed = set(after['comparison']['affected_quantities']) - {excitation}
    explained = {owner for owner, allowed in forecasts.items() if observed <= allowed}
    if (not left or not right or left & right or len(explained) != 1
            or explained != set(after['retained_owners'])):
        return result('position_method_inapplicable', reason='distinct_forecasts_with_measured_resolution_required')
    return result('formed_position_owner_method', method_object={
        'kind': 'partial_role_distinguishing_experiment_relation',
        'status': 'measured_position_followup',
        'position_method': {'operator': 'plan_position_owner_followup',
                            'remaining_owner_count': len(owners), 'reuse_reset_corners': reused,
                            'excitation_role': 'outside_remaining_owners',
                            'prediction_relation': 'disjoint_foreign_effects'},
        'object_assignment_transferred': False, 'target_independent_support': False})


def _position_followup(plan, result, model, records, methods, remaining_rows, *, assess):
    """Pass a frozen position question through the existing Common purchase callback."""
    budget = methods.budget
    missing = plan["missing_corner_indices"]
    if assess or not missing:
        if not methods._index(records) or not budget.spend(len(plan["corners"])):
            return {**result, "status": "scoped_unknown", "reason": "position_index_compute_exhausted",
                    "compute_cost": budget.spent, "position_experiment": plan}
        refs = plan.get("paid_corner_refs", (plan["baseline_ref"], None, None, None))
        corners = [methods.reset_by_probe.get(ref["probe_id"]) if ref is not None
                   else methods.index.get(methods._key([point]))
                   for point, ref in zip(plan["corners"], refs, strict=True)]
        missing = [i for i, row in enumerate(corners) if row is None]
        if not missing:
            from .object_induction import assess_position_experiment
            measured = assess_position_experiment(plan, model, corners, budget)
            complete = measured["status"] in {"position_domain_narrowed", "position_domain_unchanged",
                                              "position_domain_conflict", "unverified_position_intervention"}
            returned = {**result, "status": measured["status"], "position_experiment": plan,
                        "position_assessment": measured, **({"completed_relation": measured} if complete else {})}
            if complete and result.get("completed_relation"):
                # Both experiments completed in this callback. Its earlier
                # selector child still owns the selector answer.
                returned["selector_assessment"] = result["completed_relation"]
            if complete and result.get("position_first_return_ref") and measured["status"] == "position_domain_narrowed":
                from .method_action_research import read_method_reference
                first = read_method_reference(result["position_first_return_ref"], ".")
                formed = form_position_owner_method([first, returned], budget)
                if "method_object" in formed:
                    returned["method_object"] = formed["method_object"]
                returned["method_formation_status"] = formed["status"]
            return {**returned, "compute_cost": budget.spent}
    corners = plan["corners"]
    controls = len(corners[0])
    # Each missing corner is a reset. The normal controller adds one indexed
    # row and four lookups per callback, then checks four measured edges once.
    readout_cost = 4 * (controls + 1) + 4 * len(model["raw_sensor_features"]) + 3 * len(model["dynamic_fields"])
    required_compute = ((len(records) - methods.indexed_rows + len(missing)) * (controls + 1)
                        + 4 * len(missing) + readout_cost)
    affordable = len(missing) <= remaining_rows and required_compute <= budget.remaining
    return {**result, "position_experiment": plan, "compute_cost": budget.spent,
            "required_followup_compute": required_compute,
            "status": "proposed_complete_partial_role_experiment" if affordable else "scoped_unknown",
            "reason": "position_corners_outstanding" if affordable else "complete_position_experiment_unfunded",
            "required_observation_rows": len(missing), "full_pair_required": True,
            "question": {"purpose": "distinguish_position_dependency",
                         **({"source_object_model_id": result["position_method_application"]["model_id"]}
                            if result.get("position_method_application") else {})},
            "relation": {"kind": "position_membership_contrast", "excitation_role": "quantity",
                         "matched_prefix_rows": 0, "position_field": plan["field"],
                         "excited_quantity": plan["excited_quantity"]},
            "paired_contract": [[[corners[0]], [corners[1]]], [[corners[2]], [corners[3]]]],
            "actions": ([{"kind": "trajectory", "channels": ["fine"], "sequence": [corners[i]]}
                         for i in missing] if affordable else []),
            "analysis_consumer": "partial_role_experiment"}


def position_method_source(sources):
    return next((source for source in reversed(sources)
                 if source.get("position_method", {}).get("operator") == "plan_position_owner_followup"), None)


def completed_position_relation(returned):
    """Read the terminal answer for this position square, even on an older paid return."""
    measured = (returned or {}).get("completed_relation") or {}
    plan = (returned or {}).get("position_experiment") or {}
    if (measured.get("status") in {"position_domain_narrowed", "position_domain_unchanged",
                                  "position_domain_conflict", "unverified_position_intervention"}
            and plan and measured.get("field") == plan["field"]
            and measured == returned.get("position_assessment", measured)):
        return measured
    return None


def new_position_method_decision(methods, prior_model, model):
    """One further decision after a previously paid square actually narrows its parent."""
    returned = getattr(methods, "result", None) or {}
    source = position_method_source(getattr(methods, "sources", ()))
    measured = completed_position_relation(returned) or {}
    return (source is not None and not returned.get("position_method_application")
            and measured.get("status") == "position_domain_narrowed"
            and len(measured["retained_owners"]) == source["position_method"]["remaining_owner_count"]
            and model.get("position_experiment_ref") != prior_model.get("position_experiment_ref"))


def _recalled_position_followup(spec, model, records, previous, source, result, methods, remaining_rows):
    from .object_alternatives import enumerate_object_schemas
    from .object_interventions import plan_position_owner_followup

    first, measured = previous["position_experiment"], previous["completed_relation"]
    recipe = source["position_method"]
    position = model.get("position_candidates", {}).get(first["field"], {})
    owners = position.get("quantities", ())
    if (measured["status"] != "position_domain_narrowed" or measured["source_model_id"] != model.get("model_id")
            or len(owners) != recipe["remaining_owner_count"] or set(owners) != set(measured["retained_owners"])):
        return {**result, "status": "scoped_unknown", "reason": "no_unresolved_measured_owner_pair"}
    schemas = enumerate_object_schemas(model, methods.budget, max_schemas=len(owners))["schemas"]
    plan = plan_position_owner_followup(spec, model, records, methods.budget, previous_plan=first,
        previous_result=measured, schemas=schemas, remaining_rows=remaining_rows)
    if "corners" not in plan:
        return {**result, "status": "scoped_unknown", "reason": plan["status"],
                "position_planning": plan, "compute_cost": methods.budget.spent}
    reused = recipe["reuse_reset_corners"]
    if (recipe["excitation_role"] != "outside_remaining_owners" or plan["excited_quantity"] in owners
            or recipe["prediction_relation"] != "disjoint_foreign_effects"
            or any(plan["paid_corner_refs"][i] != measured["corner_refs"][i] for i in reused)):
        return {**result, "status": "scoped_unknown", "reason": "position_method_binding_unavailable",
                "compute_cost": methods.budget.spent}
    result["position_method_application"] = {"operator": recipe["operator"], "model_id": source["model_id"],
                                             "source_ref": deepcopy(source["source_ref"])}
    result["position_first_return_ref"] = deepcopy(model["position_experiment_ref"])
    return _position_followup(plan, result, model, records, methods, remaining_rows, assess=False)


def propose_partial_role_experiment(spec, model, records, *, selected_values=None,
                                    sources=(), compute_limit=0, remaining_rows=0, serial=0,
                                    previous=None, methods=None):
    """Thin Common proposal adapter over the existing measured-role method.

    This reads paid transitions and quotes every missing paired corner. It
    never acquires, fits, infers hidden roles, or transfers an object assignment
    from a source apparatus. Anonymous scalar controls do not supply this
    sensor/role contract merely by being renamed.
    """
    from .models import SearchBudget

    model, selected_values = model or {}, selected_values or {}
    missing = []
    for name in ('focus_control', 'focus_candidates', 'quantity_controls', 'drive_links'):
        if not model.get(name):
            missing.append(f'model.{name}')
    if not selected_values:
        missing.append('measured_selector_values')
    if not records:
        missing.append('paid_before_after_transition_records')
    bounds = getattr(spec, 'feature_bounds', {}) or {}
    maximum = getattr(spec, 'max_probe_length', None)
    if not isinstance(maximum, int) or maximum < 1:
        missing.append('public_max_probe_length')
    controls = [model.get('focus_control'), *model.get('quantity_controls', {}).values(),
                *model.get('drive_links', {})]
    if any(control not in bounds for control in controls if control is not None):
        missing.append('public_control_bounds_for_declared_roles')
    quantities = set(model.get('quantity_controls', {}))
    for index, record in enumerate(records):
        if (any(key not in record for key in ('controls', 'before', 'after', 'target',
                                              'history', 'position', 'probe_id'))
                or not quantities <= set(record.get('before', {}))
                or not quantities <= set(record.get('after', {}))):
            missing.append(f'paid_transition_record:{index}')
    if any(field not in model.get('focus_candidates', {}) for field in selected_values):
        missing.append('selector_values_bound_to_focus_candidates')
    result = {'status': 'UNKNOWN', 'required': missing, 'compute_cost': 0,
        'new_observations': 0, 'world_support_updated': False,
        'object_assignment_transferred': False, 'target_independent_support': False,
        'source_success_threshold': 1e-6,
        'source_threshold_scope': 'historical_object_methods_not_current_world_acceptance',
        'sources': [{key: deepcopy(source[key]) for key in ('model_id', 'source_ref') if key in source}
                    for source in sources]}
    if missing:
        return {**result, 'reason': 'explicit_measured_object_role_contract_required'}
    budget, actions = SearchBudget(max(0, compute_limit)), []
    if methods is None:
        methods = ObjectDependencyMethods(sources, budget=budget, actions=actions)
    else:
        # The caller retains only the paid transition index. Reconstruct the
        # current research decision from its original proposal and new model;
        # a previous recipe or pending question must not silently survive it.
        methods.sources, methods.budget, methods.actions = sources, budget, actions
        methods.evidence, methods.pending, methods.last_recipe = [], None, None
        methods.announced = set()
    if previous and previous.get("position_experiment"):
        result["measured_methods"] = previous.get("measured_methods", [])
        source = position_method_source(sources)
        if completed_position_relation(previous):
            if source and not previous.get("position_method_application") and model.get("position_experiment_ref"):
                return _recalled_position_followup(spec, model, records, previous, source, result, methods, remaining_rows)
            return {**previous, "compute_cost": 0}
        for key in ("position_method_application", "position_first_return_ref"):
            if key in previous:
                result[key] = previous[key]
        return _position_followup(previous["position_experiment"], result, model, records,
                                  methods, remaining_rows, assess=True)
    if previous and previous.get('paired_contract'):
        # Resume the actual proposed pair before choosing another experiment.
        # Reconstructing just the current candidate set can otherwise skip a
        # completed pilot once its observations have resolved those candidates.
        contract = previous['paired_contract']
        relation = previous['relation']
        recipe = (relation['excitation_role'], relation['matched_prefix_rows'])
        methods.evidence = deepcopy(previous.get('measured_methods', []))
        methods.last_recipe = recipe
        if not methods._index(records):
            return {**result, 'status': 'scoped_unknown',
                'reason': 'dependency_method_compute_exhausted', 'compute_cost': budget.spent}
        assessment = methods._assessment(recipe, relation['quantity_candidate'],
            contract, methods._scope(records))
        if assessment is None:
            missing_corners = {methods._key(sequence): sequence
                for pair in contract for sequence in pair
                if methods._key(sequence) not in methods.index}
            needed = sum(len(sequence) for sequence in missing_corners.values())
            affordable = needed <= remaining_rows
            return {**result, 'compute_cost': budget.spent,
                'status': ('proposed_complete_partial_role_experiment' if affordable else 'scoped_unknown'),
                'reason': ('paired_measurements_incomplete' if affordable else 'complete_paired_experiment_unfunded'),
                'relation': deepcopy(relation), 'paired_contract': deepcopy(contract),
                'actions': ([{'kind': 'trajectory', 'channels': ['fine'], 'sequence': deepcopy(sequence)}
                    for sequence in missing_corners.values()] if affordable else []),
                'required_observation_rows': needed, 'full_pair_required': True,
                'measured_methods': methods.snapshot(), 'method_actions': actions}
        result['completed_relation'] = deepcopy(assessment)
        result['method_object'] = {
            'kind': 'partial_role_distinguishing_experiment_relation',
            'dependency_methods': methods.snapshot(),
            'source_role_bindings': {key: deepcopy(model[key]) for key in
                ('focus_control', 'quantity_controls', 'drive_links')},
            'status': 'measured_complete_pair',
            'object_assignment_transferred': False,
            'target_independent_support': False}
    probe, question = methods.propose(spec, model, records, serial, selected_values,
        remaining_rows=remaining_rows)
    result.update(compute_cost=budget.spent, question=question,
        measured_methods=methods.snapshot(), method_actions=actions)
    if probe is None:
        if (question['purpose'] == 'measured_dependency_methods_lack_contrast'
                and remaining_rows > 0 and len(model.get('position_candidates', {})) >= 3
                and methods.evidence and all(item['status'] == 'uninformative_pilot' for item in methods.evidence)):
            from .object_interventions import plan_paid_position_contrast
            plan = plan_paid_position_contrast(spec, model, records, budget, remaining_rows=remaining_rows)
            if 'corners' in plan:
                return _position_followup(plan, result, model, records, methods, remaining_rows, assess=False)
            result.update(compute_cost=budget.spent, position_planning=plan)
        return {**result, 'status': 'scoped_unknown', 'reason': question['purpose']}
    recipe, quantity, pairs, scope = methods.pending
    missing_corners = {}
    for pair in pairs:
        for sequence in pair:
            key = methods._key(sequence)
            if key not in methods.index:
                missing_corners.setdefault(key, sequence)
    needed = sum(len(sequence) for sequence in missing_corners.values())
    if needed > remaining_rows:
        return {**result, 'status': 'scoped_unknown', 'reason': 'complete_paired_experiment_unfunded',
            'required_observation_rows': needed}
    # A consumer purchases this whole current contract, then calls the same
    # adapter on its paid transitions. No single corner receives pair credit.
    return {**result, 'status': 'proposed_complete_partial_role_experiment',
        'relation': {'excitation_role': recipe[0], 'matched_prefix_rows': recipe[1],
            'quantity_candidate': quantity, 'scope': scope,
            'readout': question['response_question'], 'status': 'target_pair_unmeasured'},
        'paired_contract': deepcopy(pairs),
        'actions': [{'kind': 'trajectory', 'channels': ['fine'], 'sequence': deepcopy(sequence)}
                    for sequence in missing_corners.values()],
        'required_observation_rows': needed, 'full_pair_required': True,
        'analysis_consumer': 'partial_role_experiment',
        'feedback_contract': {'previous_ref': 'saved complete proposal',
                              'records_ref': 'paid before/after records including every paired corner'}}
