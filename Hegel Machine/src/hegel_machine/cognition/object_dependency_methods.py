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
               == (*recipe, quantity) and item["scope"] == scope), None)
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
