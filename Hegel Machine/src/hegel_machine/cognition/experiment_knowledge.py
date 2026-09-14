"""Remember experiments that actually separated competing explanations."""

from copy import deepcopy
from itertools import combinations, islice, permutations
from statistics import mean

from .evaluation import predict, point_loss
from .models import Probe
from .probing import (_measurement_scenarios, _sample_rows, _trajectory_predictions,
                      transition_probe)


def research_context(spec, observations):
    response = "binary" if len({row.target for row in observations}) == 2 else "numeric"
    return {"objective": spec.risk_quantile, "response": response,
            "controls": len(spec.feature_bounds or spec.feature_names),
            "measurements": len(set(spec.feature_names) - set(spec.feature_bounds or spec.feature_names))}


class ExperimentKnowledge:
    def __init__(self, records=()):
        self.records = deepcopy(list(records))
        self.last_recall = []

    def record(self, spec, candidates, earlier, probe, acquired, *, fallback, budget, design=None):
        """Learn only from paid outcomes, with each prediction charged once."""
        if probe is None or not earlier or not acquired or not spec.feature_bounds:
            return None
        old, new = _sample_rows(earlier, 8), _sample_rows(acquired, 8)
        measured = []
        for candidate in candidates[:6]:
            if not budget.spend(len(old) + len(new)):
                break
            before = [predict(candidate.program, row, fallback) for row in old]
            after = [predict(candidate.program, row, fallback) for row in new]
            error = mean(point_loss(value, row.target, spec.risk_quantile) for value, row in zip(after, new))
            measured.append((candidate, before, after, error))
        splits = []
        for left, right in combinations(measured, 2):
            previous = mean((a - b) ** 2 for a, b in zip(left[1], right[1]))
            subsequent = mean((a - b) ** 2 for a, b in zip(left[2], right[2]))
            gain = abs(left[3] - right[3])
            if subsequent > max(1e-8, 4 * previous) and gain > 1e-8:
                splits.append((gain, previous, subsequent, left, right))
        if not splits:
            return None
        gain, previous, subsequent, left, right = max(splits, key=lambda item: item[0])
        names = tuple(spec.feature_bounds)
        normalized = []
        for point in probe.sequence:
            normalized.append([(point[name] - spec.feature_bounds[name][0]) /
                               (spec.feature_bounds[name][1] - spec.feature_bounds[name][0] or 1.)
                               for name in names])
        context = research_context(spec, earlier)
        existing = next((item for item in self.records if item["context"] == context
                         and item["normalized_sequence"] == normalized), None)
        sources = {c.memory_id for c in (left[0], right[0]) if c.memory_id}
        for candidate in (left[0], right[0]):
            sources.update(definition.get("source_concept_id") for definition in
                           candidate.program.to_dict().get("definitions", {}).values())
        sources.discard(None)
        next_id = 1 + max((int(item["experiment_id"].removeprefix("experiment_")) for item in self.records), default=0)
        result = {"experiment_id": existing["experiment_id"] if existing else f"experiment_{next_id}",
                  "context": context, "source_task": spec.task_id, "roles": list(names),
                  "normalized_sequence": normalized, "source_concepts": sorted(sources),
                  "explanations": [left[0].program.to_dict(), right[0].program.to_dict()],
                  "winner": 0 if left[3] < right[3] else 1,
                  "before_distance": previous, "after_distance": subsequent,
                  "observed_loss_gap": gain, "observations": probe.cost,
                  "uses": (existing or {}).get("uses", 0),
                  "confirmed": (existing or {}).get("confirmed", 0) + 1,
                  "failures": (existing or {}).get("failures", 0)}
        if design and design.get("policy") in {"conditional_transition_interval",
                                               "joint_conditional_transition_interval"}:
            # The successful procedure is a question about the current
            # boundary, not a commitment to its source-task coordinate.
            support = design.get("boundary_support", [])
            if design.get("boundary_targets"):
                from .probing import transition_outcome
                outcome = design.get("outcome") or transition_outcome(design, acquired)
                support = [target for target in outcome["targets"] if target["condition_reached"]]
            if support:
                result["procedure"] = {"kind": "conditional_boundary",
                    "joint_controls": design["policy"] == "joint_conditional_transition_interval",
                    "source_support": deepcopy(support)}
        if existing:
            if design and design.get("boundary_targets") and "procedure" not in result:
                existing.pop("procedure", None)
            existing.update(result)
        else:
            self.records.append(result)
        return deepcopy(result)

    def propose(self, spec, candidates, observations, *, fallback, budget, remaining, used=(),
                evidence_losses=None, boundary_proposal=None):
        """Rebind learned interventions and test their relevance to today's models."""
        self.last_recall = []
        if not candidates or not spec.feature_bounds:
            return []
        context = research_context(spec, observations)
        controls = tuple(spec.feature_bounds)
        scenarios = _measurement_scenarios(_sample_rows(observations, 12), controls)
        used_sequences = {tuple(tuple(sorted(point.items())) for point in probe.sequence) for probe in used}
        offered = []
        records = sorted((r for r in self.records if r["context"] == context),
                         key=lambda r: (r["failures"] - r["confirmed"], -r["observed_loss_gap"]))
        if not records:
            return []
        unique = list({candidate.program: candidate for candidate in candidates}.values())
        if len(unique) == 1 and not any(record.get("procedure") for record in records):
            return []
        exact_response = spec.decision_costs is None and spec.risk_quantile == .5 and context["response"] == "binary"
        measured = dict(evidence_losses or {})
        missing = [candidate for candidate in unique if candidate.program not in measured]
        price = len(missing) * len(observations)
        if exact_response and len(unique) > 1 and observations and missing:
            if not budget.spend(price):
                return []
            for candidate in missing:
                measured[candidate.program] = mean(point_loss(predict(candidate.program, row, fallback),
                    row.target, spec.risk_quantile) for row in observations)
        if exact_response and len(unique) > 1 and observations:
            best_loss = min(measured[candidate.program] for candidate in unique)
            unique = [candidate for candidate in unique if measured[candidate.program] <= best_loss + 1e-12]
        # Disagreement with an explanation already refuted on today's data is
        # not a reason to repeat an old experiment. A saved boundary procedure
        # can still refine one surviving model's unresolved response brackets.
        # Numeric/risk losses alone do not establish that the runner-up has
        # been refuted; their outcome noise is not an exact response mismatch.
        candidates = unique
        procedures = set()
        for record in records:
            procedure = record.get("procedure", {})
            if procedure.get("kind") == "conditional_boundary":
                kind = procedure["joint_controls"]
                if kind in procedures:
                    continue
                procedures.add(kind)
                started = budget.spent
                if boundary_proposal is not None:
                    rebound, details = boundary_proposal
                else:
                    rebound, details = transition_probe(spec, candidates, observations, budget=budget,
                        remaining=remaining, round_number=len(used), used=used, joint_controls=kind,
                        evidence_losses=measured)
                if rebound is not None:
                    probe = Probe(f"recalled_{record['experiment_id']}_{len(offered)}", rebound.sequence)
                    offered.append((max(details.get("candidate_disagreement", 0.),
                                        details.get("unresolved_boundary_width", 0.)) / probe.cost, probe))
                    used_sequences.add(tuple(tuple(sorted(point.items())) for point in probe.sequence))
                    self.last_recall.append({"experiment_id": record["experiment_id"],
                        "probe_id": probe.probe_id, "procedure": "conditional_boundary",
                        "binding": "current_observed_response_intervals",
                        "supported_candidates": len(candidates), "design": details,
                        "compute_cost": budget.spent - started})
                continue
            if len(candidates) < 2:
                continue
            length = len(record["normalized_sequence"])
            if length > min(remaining, spec.max_probe_length):
                continue
            for role_order in islice(permutations(controls), 6):
                sequence = tuple({name: spec.feature_bounds[name][0] + value *
                                  (spec.feature_bounds[name][1] - spec.feature_bounds[name][0])
                                  for name, value in zip(role_order, point)}
                                 for point in record["normalized_sequence"])
                key = tuple(tuple(sorted(point.items())) for point in sequence)
                if key in used_sequences:
                    continue
                price = length * min(4, len(candidates)) * len(scenarios)
                if not budget.spend(price):
                    return [item[1] for item in sorted(offered, key=lambda item: -item[0])[:4]]
                probe = Probe(f"recalled_{record['experiment_id']}_{len(offered)}", sequence)
                separations = []
                for scenario in scenarios:
                    outputs = [_trajectory_predictions(candidate, probe, fallback, scenario)
                               for candidate in candidates[:4]]
                    separations.extend(mean((a-b)**2 for a,b in zip(left,right))
                                       for left,right in combinations(outputs, 2))
                separation = max(separations, default=0.)
                if separation > 1e-8:
                    offered.append((separation / length, probe))
                    used_sequences.add(key)
                    self.last_recall.append({"experiment_id": record["experiment_id"],
                        "probe_id": probe.probe_id, "role_mapping": dict(zip(record["roles"], role_order)),
                        "supported_candidates": len(candidates),
                        "predicted_separation": separation, "compute_cost": price})
        return [item[1] for item in sorted(offered, key=lambda item: -item[0])[:4]]

    def feedback(self, probe, *, supported):
        for record in self.records:
            if probe.probe_id.startswith(f"recalled_{record['experiment_id']}_"):
                record["uses"] += 1
                record["confirmed" if supported else "failures"] += 1

    def to_dict(self):
        return deepcopy(self.records)
