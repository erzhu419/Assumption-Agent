"""Recall the instruments and exposure needed by previous measured research."""

from copy import deepcopy
from itertools import combinations

from .language import Program


def _interface(laboratory):
    """Describe the available experiment, without using task identities."""
    spec = laboratory.spec
    base = set(laboratory.spec_for(()).feature_names)
    return {"controls": {name: list(bounds) for name, bounds in spec.feature_bounds.items()},
            "features": sorted(base), "risk_quantile": spec.risk_quantile,
            "decision_costs": list(spec.decision_costs) if spec.decision_costs else None,
            "instruments": {channel.name: sorted(set(laboratory.spec_for((channel.name,)).feature_names) - base)
                            for channel in laboratory.channels}}


def _operating_rows(batches, paid_limit=None):
    """Map acquired rows to operation; another instrument can replay a probe."""
    seen, paid, operating = [], 0, 0
    for batch in batches:
        rows = len(batch["observations"])
        if paid_limit is not None:
            rows = min(rows, max(0, paid_limit - paid))
        if not rows:
            break
        if batch["probe"] not in seen:
            seen.append(batch["probe"])
            operating += rows
        paid += rows
    return operating


class MeasurementAcquisitionMemory:
    """Acquisition plans remain hypotheses until the new apparatus is measured."""

    def __init__(self, library, records=()):
        self.library = library
        self.records = deepcopy(list(records))

    def plan(self, laboratory):
        interface = _interface(laboratory)
        records = [record for record in self.records if record["interface"] == interface]
        if not records:
            return None
        rates = {channel.name: channel.cost_per_row for channel in laboratory.channels}
        eligible = {candidate.memory_id: candidate for candidate in
                    self.library.retrieve(laboratory.spec_for(tuple(rates)))}
        base = set(interface["features"])
        choices = []
        for record in records:
            if not record["successful"]:
                continue
            for source_id in record["source_ids"]:
                candidate = eligible.get(source_id)
                if candidate is None:
                    continue
                needed = candidate.program.required_features() - base
                subsets = [subset for size in range(len(rates) + 1)
                           for subset in combinations(rates, size)
                           if needed <= set().union(*(set(interface["instruments"][name]) for name in subset))]
                if not subsets:
                    continue
                channels = min(subsets, key=lambda subset: (sum(rates[name] for name in subset), len(subset)))
                length = max(record["pilot_length"], candidate.program.minimum_history_depth + 2)
                if length > laboratory.spec.max_probe_length:
                    continue
                choices.append({"channels": channels, "source_id": source_id, "pilot_length": length,
                                "rate": sum(rates[name] for name in channels), "size": candidate.program.size})
        if not choices:
            return None
        chosen = min(choices, key=lambda choice: (choice["rate"], choice["pilot_length"], choice["size"]))
        return {"initial_channels": list(chosen["channels"]),
                "pilot_length": chosen["pilot_length"],
                "coverage_rows": max(record["coverage_rows"] for record in records),
                "source_ids": sorted({choice["source_id"] for choice in choices
                                      if choice["channels"] == chosen["channels"]})}

    def record(self, laboratory, state, *, minimum_improvement=.002):
        """Store acquired coverage and preserve late conflicts even if work stops."""
        batches = state.get("batches", [])
        observed_rows = _operating_rows(batches)
        pilots = state.get("pilot_observations", [])
        if len(pilots) < 2:
            return None
        episodes = state.get("episodes", [])
        conflicts = []
        for event in [*state.get("measurement_events", []), state.get("measurement_question", {})]:
            if event.get("reason") not in {"new_trajectory_residual", "persistent_prediction_residual"}:
                continue
            position = min(observed_rows, event.get("observation_end",
                           _operating_rows(batches, event.get("evidence_rows", 0))))
            if position:
                conflicts.append({"observation_end": position, "reason": event["reason"],
                                  "channels": list(event.get("before_channels", [])),
                                  "validation_loss": event.get("current_validation_loss")})
        episode = episodes[-1] if episodes else None
        successful = bool(state.get("stage") == "completed" and episode
                          and episode.get("validation_loss") is not None
                          and episode["validation_loss"] <= minimum_improvement)
        source_ids = []
        if successful:
            program = Program.from_dict(episode["program"]).expanded().simplified()
            source_ids = [concept.concept_id for concept in self.library.concepts
                          if concept.status != "revoked" and concept.program.expanded().simplified() == program]
        if not source_ids and not conflicts:
            return None
        first_visit_end = _operating_rows(batches, sum(len(rows) for rows in pilots)
                                         + (episodes[0]["observation_cost"] if episodes else 0))
        coverage = max(first_visit_end,
                       (state.get("acquisition_plan") or {}).get("coverage_rows", 0),
                       *(event.get("observation_end", 0) for event in state.get("coverage_events", [])),
                       *(event["observation_end"] for event in conflicts))
        result = {"interface": _interface(laboratory), "source_task": state["task_id"],
                  "successful": bool(successful and source_ids), "source_ids": source_ids,
                  "channels": list(state["selected_channels"]),
                  "pilot_length": min(len(rows) for rows in pilots),
                  "coverage_rows": min(observed_rows, coverage), "conflicts": conflicts,
                  "validation_loss": episode.get("validation_loss") if episode else None}
        self.records.append(result)
        return deepcopy(result)

    def to_dict(self):
        return deepcopy(self.records)
