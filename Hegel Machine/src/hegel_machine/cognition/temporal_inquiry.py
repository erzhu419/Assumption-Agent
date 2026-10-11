"""Expose the existing temporal apparatus to the shared Python inquiry loop."""

from copy import deepcopy
from dataclasses import asdict
from random import Random

from .ambiguous_instrument_lab import AmbiguousInstrumentLab
from .models import Observation, Probe


LAB_INTERFACE = """Apply bounded control trajectories to an unknown apparatus.
query {actions:[{kind:'trajectory',sequence:[{u:number},...],channels:[]},...]}
buys responses for a complete trajectory. Select channels:['meter'] to also buy
the auxiliary readings. Every trajectory resets the apparatus; steps within it
share its evolving state. Each step costs one observation, and meter readings
add three measurement units per step. Respect the public maximum probe length.
Returned rows contain control features, the observed target, and control history.
Optional measurements are separate readings indexed by observation_index; their
row_indices identify the corresponding response rows. No mechanism formula or
finite hypothesis list is supplied. Learn and revise your own Python models.
model.py defines predict(row), using row['features'] and row['history']. During
confirmation and terminal evaluation these contain only current and previous
controls, with an empty history at each reset. Predict the entire response from
those controls; auxiliary readings and past response targets are unavailable to
the predictor. You may use purchased meter readings to learn an internal model.
Fresh confirmation buys random trajectories covering the current public horizon,
with a shorter final trajectory when needed, using the configured row budget.
"""


def _control_observation(row):
    return Observation({"u": row.features["u"]}, row.target,
                       tuple({"u": point["u"]} for point in row.history))


class TemporalInquiryLab:
    """Keep apparatus physics private and preserve its paid trajectory interface."""

    task_interface = LAB_INTERFACE

    def __init__(self, task_index=0, seed=113, apparatus_seed=None,
                 instrument_mode="recurrent", response_seed=None,
                 calibration_change=None):
        self._settings = dict(task_index=task_index, seed=seed,
                              apparatus_seed=apparatus_seed, instrument_mode=instrument_mode,
                              response_seed=response_seed,
                              calibration_change=deepcopy(calibration_change))
        self._past_costs = {"observation_cost": 0, "measurement_cost": 0, "queries": 0}
        self._settled = calibration_change is None
        self._laboratory = AmbiguousInstrumentLab(**self._apparatus_settings())

    def _apparatus_settings(self, *, settled=False):
        settings = {key: value for key, value in self._settings.items()
                    if key != "calibration_change"}
        change = self._settings["calibration_change"]
        if settled and change is not None:
            settings.update({key: change[key] for key in ("apparatus_seed", "response_seed", "instrument_mode")
                             if key in change})
        return settings

    def public_spec(self):
        spec = {"controls": ["u"], "feature_bounds": {"u": [-1., 1.]},
                "action_kinds": ["trajectory"],
                "max_probe_length": self._laboratory.spec.max_probe_length,
                "channels": [{"name": "meter", "cost_per_row": 3,
                              "description": "Read the auxiliary instrument."}],
                "observation_cost_per_row": 1,
                "reset": "Every trajectory starts from the apparatus reset state.",
                "prediction_contract": "Current controls and control history only; no meter or past targets."}
        if self._settings["calibration_change"] is not None:
            spec["settled_after_observation_rows"] = self._settings["calibration_change"]["after_rows"]
            spec["prediction_objective"] = (
                "Predict the settled apparatus. Calibration may be updated before the first "
                "trajectory starting at or after the observation checkpoint, and is fixed "
                "thereafter. No trajectory is interrupted by an update.")
        return spec

    @property
    def costs(self):
        return {key: self._past_costs[key] + value
                for key, value in self._laboratory.costs.items()}

    def action_cost(self, action):
        """Validate a proposed purchase before the runtime charges its budget."""
        if action.get("kind") != "trajectory":
            raise ValueError("the available action kind is trajectory")
        sequence = action.get("sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError("trajectory requires a nonempty sequence")
        if len(sequence) > self._laboratory.spec.max_probe_length:
            raise ValueError("trajectory exceeds the public maximum probe length")
        if action.get("channels", []) not in ([], ["meter"]):
            raise ValueError("channels must be [] or ['meter']")
        for point in sequence:
            if not isinstance(point, dict) or set(point) != {"u"}:
                raise ValueError("each trajectory point must contain only control u")
            if not isinstance(point["u"], (int, float)) or not -1. <= point["u"] <= 1.:
                raise ValueError("control u must be a number in [-1, 1]")
        return len(sequence)

    def prediction_rows(self, action):
        """Construct prospective inputs from the public action, without observing."""
        self.action_cost(action)
        history, rows = [], []
        for point in action["sequence"]:
            control = {"u": point["u"]}
            rows.append(Observation(dict(control), 0., tuple(dict(past) for past in history)))
            history.append(control)
        return rows

    def confirmation_precondition(self, score, *, mse_threshold):
        """A fixed apparatus cannot confirm a model already refuted by paid data."""
        # Calibration-change protocols deliberately retain pre-change evidence:
        # a settled model need not explain the earlier physical calibration.
        if self._settings["calibration_change"] is not None:
            return None
        mse = score.get("mse")
        if mse is not None and mse > mse_threshold:
            return (f"paid evidence contradicts this model (mse={mse}, "
                    f"threshold={mse_threshold}); revise it before fresh confirmation")
        return None

    def execute(self, action):
        count = self.action_cost(action)
        first = self.costs["observation_cost"]
        if (not self._settled
                and first >= self._settings["calibration_change"]["after_rows"]):
            self._past_costs = self.costs
            self._laboratory = AmbiguousInstrumentLab(**self._apparatus_settings(settled=True))
            self._settled = True
        probe = Probe(f"inquiry_{self.costs['queries']}", tuple(deepcopy(action["sequence"])))
        batch = self._laboratory.acquire(probe, action.get("channels", []))
        indices = list(range(first, first + count))
        measurements = [{"observation_index": index, "values": {"meter": row.features["meter"]}}
                        for index, row in zip(indices, batch.observations) if "meter" in row.features]
        return {"observation_index": first, "row_indices": indices,
                "action": deepcopy(action),
                "rows": [asdict(_control_observation(row)) for row in batch.observations],
                "measurements": measurements, "observation_cost": batch.observation_cost,
                "measurement_cost": batch.measurement_cost}

    def observation_rows(self, record):
        return [Observation(row["features"], row["target"], tuple(row["history"]))
                for row in record["rows"]]

    def target_rows(self, records, target="response"):
        """Select an observed response or purchased channel as the prediction target."""
        if target == "response":
            return [row for record in records for row in self.observation_rows(record)]
        if target not in {channel["name"] for channel in self.public_spec()["channels"]}:
            raise ValueError(f"unknown public observation target: {target}")
        rows = []
        for record in records:
            readings = {reading["observation_index"]: reading["values"][target]
                        for reading in record["measurements"] if target in reading["values"]}
            for index, row in zip(record["row_indices"], record["rows"]):
                if index in readings:
                    rows.append(Observation(dict(row["features"]), readings[index],
                                tuple(dict(point) for point in row["history"])))
        return rows

    def confirmation_actions(self, n, rng):
        """Spend exactly n rows on full legal horizons and one optional remainder."""
        actions = []
        while n > 0:
            count = min(n, self._laboratory.spec.max_probe_length)
            actions.append({"kind": "trajectory",
                            "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(count)],
                            "channels": []})
            n -= count
        return actions

    def evaluation_rows(self, *, length=256, seed=1000003):
        # Terminal evaluation concerns the settled apparatus even when called on
        # an unused clone. It does not buy a hidden prefix to cross the checkpoint.
        separate = TemporalInquiryLab(**self._apparatus_settings(settled=True))
        rows = []
        for action in separate.confirmation_actions(length, Random(seed)):
            record = separate.execute(action)
            rows.extend(separate.observation_rows(record))
        return rows
