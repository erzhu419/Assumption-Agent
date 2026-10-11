"""Expose historical simulator mechanisms through one anonymous inquiry interface.

This is a finite-control development setting, not a new simulator or a claim of
continuous-domain identification. The existing simulator computes every label;
this adapter only binds public controls, resets, records and evaluation horizons.
"""

from copy import deepcopy
from dataclasses import asdict
from math import isclose
from random import Random

from .environments import TemporalEnvironment, _Mechanism
from .models import Observation, Probe


CONTROL_LEVELS = tuple(index / 10 for index in range(-10, 11))
MAX_PROBE_LENGTH = 24

LAB_INTERFACE = """Investigate an unknown apparatus through its public controls.
query {actions:[{kind:'trajectory',sequence:[{u:number},...],channels:[]},...]}
buys the observed response at every step of each supplied control sequence.
Each trajectory starts independently at the apparatus reset state. Each step
costs one observation. Use only the listed control_levels and at most the public
max_probe_length steps per trajectory; there are no auxiliary channels.
All purchased records remain available. You may write and revise ordinary Python
predictors and experiment programs using these records and public capabilities.
The apparatus is fixed throughout an episode. A model contradicted by purchased
records must be revised before another fresh confirmation can be bought.
The predictor defines predict(row) using row['features']['u'] and row['history'],
whose entries contain only preceding controls {'u':number} since the reset.
Past response targets are not inputs during confirmation or terminal prediction.
Fresh confirmation uses new trajectories on the same public control levels.
Terminal evaluation also tests longer trajectories, up to 48 steps per reset.
"""


class AnonymousHistoryLab:
    """Use identical public inputs for static, delayed and stateful apparatuses."""

    task_interface = LAB_INTERFACE

    def __init__(self, kind="memoryless", seed=113):
        rng = Random(seed)
        if kind in {"memoryless", "delay"}:
            mechanism = _Mechanism(
                kind="affine",
                slope=rng.uniform(0.4, 1.4),
                intercept=rng.uniform(-0.4, 0.4),
                lag_weight=rng.uniform(0.5, 1.2) if kind == "delay" else 0.0,
                history_lag=rng.randint(1, 3) if kind == "delay" else 1,
            )
        elif kind == "hysteresis":
            mechanism = _Mechanism(
                kind="hysteresis",
                lower_threshold=rng.uniform(-0.75, -0.15),
                upper_threshold=rng.uniform(0.15, 0.75),
            )
        else:
            raise ValueError("unsupported development apparatus kind")
        self._seed, self._mechanism = seed, mechanism
        self._environment = self._new_environment(MAX_PROBE_LENGTH)
        self._costs = {"observation_cost": 0, "measurement_cost": 0, "queries": 0}

    def _new_environment(self, horizon):
        return TemporalEnvironment(
            task_id="apparatus", seed=self._seed, query_seed=self._seed,
            mechanism=self._mechanism, phase="development", feature_limit=1.0,
            max_probe_length=horizon, query_catalog=((), ()),
        )

    def public_spec(self):
        return {
            "controls": ["u"], "feature_bounds": {"u": [-1.0, 1.0]},
            "control_levels": {"u": list(CONTROL_LEVELS)},
            "action_kinds": ["trajectory"], "max_probe_length": MAX_PROBE_LENGTH,
            "channels": [], "observation_cost_per_row": 1,
            "reset": "Every trajectory starts from the apparatus reset state.",
            "prediction_contract": "Current and preceding controls only; no past targets.",
            "evaluation_max_trajectory_length": 48,
        }

    @property
    def costs(self):
        return dict(self._costs)

    @staticmethod
    def _sequence(action):
        if action.get("kind") != "trajectory":
            raise ValueError("the available action kind is trajectory")
        sequence = action.get("sequence")
        if not isinstance(sequence, list) or not 1 <= len(sequence) <= MAX_PROBE_LENGTH:
            raise ValueError("trajectory requires 1 to 24 control steps")
        if action.get("channels", []) != []:
            raise ValueError("the apparatus has no auxiliary channels")
        controls = []
        for point in sequence:
            if not isinstance(point, dict) or set(point) != {"u"}:
                raise ValueError("each step must contain only control u")
            value = point["u"]
            if not isinstance(value, (int, float)):
                raise ValueError("control u must be a listed numeric level")
            level = next((item for item in CONTROL_LEVELS
                          if isclose(value, item, rel_tol=0.0, abs_tol=1e-9)), None)
            if level is None:
                raise ValueError("control u must be a listed numeric level")
            controls.append({"u": level})
        return controls

    def action_cost(self, action):
        return len(self._sequence(action))

    def prediction_rows(self, action):
        """Construct public inputs before purchase, without consulting physics."""
        history, rows = [], []
        for point in self._sequence(action):
            rows.append(Observation(dict(point), 0.0, tuple(dict(past) for past in history)))
            history.append(point)
        return rows

    @staticmethod
    def _observed_rows(environment, sequence, name):
        probe = Probe(name, tuple({"x": point["u"], "z": 0.0} for point in sequence))
        return [Observation({"u": row.features["x"]}, row.target,
                            tuple({"u": past["x"]} for past in row.history))
                for row in environment.observe(probe)]

    def execute(self, action):
        sequence = self._sequence(action)
        rows = self._observed_rows(self._environment, sequence, f"query_{self._costs['queries']}")
        first, count = self._costs["observation_cost"], len(rows)
        self._costs["observation_cost"] += count
        self._costs["queries"] += 1
        return {
            "observation_index": first, "row_indices": list(range(first, first + count)),
            "action": {**deepcopy(action), "sequence": sequence},
            "rows": [asdict(row) for row in rows], "measurements": [],
            "observation_cost": count, "measurement_cost": 0,
        }

    def observation_rows(self, record):
        return [Observation(dict(row["features"]), row["target"],
                            tuple(dict(past) for past in row["history"]))
                for row in record["rows"]]

    @staticmethod
    def confirmation_precondition(score, *, mse_threshold):
        """Keep paid refutations in this fixed apparatus, using the existing score."""
        mse = score.get("mse")
        if mse is not None and mse > mse_threshold:
            return (f"paid evidence contradicts this model (mse={mse}, "
                    f"threshold={mse_threshold}); revise it before fresh confirmation")
        return None

    def confirmation_actions(self, n, rng):
        actions = []
        while n > 0:
            count = min(n, MAX_PROBE_LENGTH)
            actions.append({"kind": "trajectory", "channels": [],
                            "sequence": [{"u": rng.choice(CONTROL_LEVELS)} for _ in range(count)]})
            n -= count
        return actions

    def evaluation_rows(self, *, length=256, seed=1000003):
        """Evaluate frozen models on independent inputs and longer reset segments."""
        environment, rng, rows = self._new_environment(48), Random(seed), []
        while len(rows) < length:
            count = min(48, length - len(rows))
            sequence = [{"u": rng.choice(CONTROL_LEVELS)} for _ in range(count)]
            rows.extend(self._observed_rows(environment, sequence, f"evaluation_{len(rows)}"))
        return rows
