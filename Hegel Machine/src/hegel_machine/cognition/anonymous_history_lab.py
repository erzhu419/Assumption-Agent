"""Expose historical simulator mechanisms through one anonymous inquiry interface.

This is a finite-control development setting, not a claim of continuous-domain
identification. Historical mechanisms retain their original simulator. The
factorized hysteresis family varies only its reset rule independently of the
existing mature transition; the public inquiry interface stays the same.
"""

from copy import deepcopy
from dataclasses import asdict
from math import isclose
from random import Random

from .environments import TemporalEnvironment, _Mechanism
from .models import Observation, Probe


CONTROL_LEVELS = tuple(index / 10 for index in range(-10, 11))
MAX_PROBE_LENGTH = 24
MAX_EVALUATION_LENGTH = 48

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


class _IndependentStartupEnvironment(TemporalEnvironment):
    """Factor one reset rule out of the historical hysteresis transition."""

    def __init__(self, *, startup_threshold, **kwargs):
        super().__init__(**kwargs)
        self._startup_threshold = startup_threshold

    def _target(self, features, history, rng):
        if not history:
            return float(features[self._feature_names[0]] >= self._startup_threshold)
        return super()._target(features, history, rng)


class AnonymousHistoryLab:
    """Use identical public inputs for static, delayed and stateful apparatuses."""

    task_interface = LAB_INTERFACE

    def __init__(self, kind="memoryless", seed=113, *, startup_seed=None):
        if kind == "factorized_hysteresis" and startup_seed is None:
            raise ValueError("factorized hysteresis requires its independent startup seed")
        if startup_seed is not None and kind != "factorized_hysteresis":
            raise ValueError("startup_seed belongs to the factorized hysteresis family")
        self._startup_threshold = None
        rng = Random(seed)
        if kind in {"memoryless", "delay"}:
            mechanism = _Mechanism(
                kind="affine",
                slope=rng.uniform(0.4, 1.4),
                intercept=rng.uniform(-0.4, 0.4),
                lag_weight=rng.uniform(0.5, 1.2) if kind == "delay" else 0.0,
                history_lag=rng.randint(1, 3) if kind == "delay" else 1,
            )
        elif kind in {"hysteresis", "factorized_hysteresis"}:
            mechanism = _Mechanism(
                kind="hysteresis",
                lower_threshold=rng.uniform(-0.75, -0.15),
                upper_threshold=rng.uniform(0.15, 0.75),
            )
            if kind == "factorized_hysteresis":
                # Reuse the historical upper-threshold draw, so equal factor
                # seeds reproduce the old apparatus exactly. Independent
                # seeds change startup without changing the mature mechanism.
                startup_rng = Random(startup_seed)
                startup_rng.uniform(-0.75, -0.15)
                self._startup_threshold = startup_rng.uniform(0.15, 0.75)
        else:
            raise ValueError("unsupported development apparatus kind")
        self._seed, self._mechanism = seed, mechanism
        self._environment = self._new_environment(MAX_PROBE_LENGTH)
        self._costs = {"observation_cost": 0, "measurement_cost": 0, "queries": 0}
        self._next_row_index = 0

    def _new_environment(self, horizon):
        environment = (_IndependentStartupEnvironment
                       if self._startup_threshold is not None else TemporalEnvironment)
        startup = ({"startup_threshold": self._startup_threshold}
                   if self._startup_threshold is not None else {})
        return environment(
            task_id="apparatus", seed=self._seed, query_seed=self._seed,
            mechanism=self._mechanism, phase="development", feature_limit=1.0,
            max_probe_length=horizon, query_catalog=((), ()), **startup,
        )

    def public_spec(self):
        return {
            "controls": ["u"], "feature_bounds": {"u": [-1.0, 1.0]},
            "control_levels": {"u": list(CONTROL_LEVELS)},
            "action_kinds": ["trajectory"], "max_probe_length": MAX_PROBE_LENGTH,
            "channels": [], "observation_cost_per_row": 1,
            "reset": "Every trajectory starts from the apparatus reset state.",
            "prediction_contract": "Current and preceding controls only; no past targets.",
            "evaluation_max_trajectory_length": MAX_EVALUATION_LENGTH,
        }

    @property
    def costs(self):
        return dict(self._costs)

    def restore_paid_account(self, observations, costs):
        """Retain paid fees independently of imported development row addresses."""
        self._costs = {"observation_cost": costs["observation_cost"],
            "measurement_cost": costs["measurement_cost"], "queries": len(observations)}
        self._next_row_index = sum(len(record['rows']) for record in observations)

    @staticmethod
    def _sequence(action, *, max_length=MAX_PROBE_LENGTH):
        if action.get("kind") != "trajectory":
            raise ValueError("the available action kind is trajectory")
        sequence = action.get("sequence")
        if not isinstance(sequence, list) or not 1 <= len(sequence) <= max_length:
            raise ValueError(f"trajectory requires 1 to {max_length} control steps")
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
        first, count = self._next_row_index, len(rows)
        self._next_row_index += count
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
        environment, rng, rows = self._new_environment(MAX_EVALUATION_LENGTH), Random(seed), []
        while len(rows) < length:
            count = min(MAX_EVALUATION_LENGTH, length - len(rows))
            sequence = [{"u": rng.choice(CONTROL_LEVELS)} for _ in range(count)]
            rows.extend(self._observed_rows(environment, sequence, f"evaluation_{len(rows)}"))
        return rows

    def evaluation_sequences(self, sequences):
        """Read frozen evaluator-only trajectories on the published control grid.

        Each sequence resets independently, with the same 48-step horizon as
        ``evaluation_rows``. Ordinary queries retain their 24-step limit and
        budget; evaluator rows and their costs belong to the caller's ledger.
        """
        controls = [self._sequence({'kind': 'trajectory', 'sequence': sequence},
                                   max_length=MAX_EVALUATION_LENGTH)
                    for sequence in sequences]
        environment, rows = self._new_environment(MAX_EVALUATION_LENGTH), []
        for index, sequence in enumerate(controls):
            rows.extend(self._observed_rows(environment, sequence, f"evaluation_{index}"))
        return rows
