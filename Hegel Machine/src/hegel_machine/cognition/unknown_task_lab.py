"""A common measurement interface over an unknown sequence of mechanisms.

The apparatus configuration is private. Researchers receive controls, instrument
prices and purchased observations. Every probe starts a fresh trajectory; a
probe id and control sequence identify a repeatable physical realization.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .measurement_lab import MeasurementBatch, MeasurementChannel
from .models import Observation, Probe, TaskSpec


@dataclass(frozen=True)
class _Apparatus:
    controls: tuple[str, ...]
    instruments: tuple[str, str]
    response: str
    delay: int = 0
    relevant_instrument: int = 0
    gain: float = 1.0
    decay: float = .83
    risk_quantile: float = .5


_APPARATUS = (
    _Apparatus(("u",), ("sensor_a", "sensor_b"), "current"),
    _Apparatus(("u",), ("sensor_a", "sensor_b"), "control_history", 1),
    _Apparatus(("u",), ("sensor_a", "sensor_b"), "joint", 1),
    _Apparatus(("x",), ("channel_p", "channel_q"), "joint", 1, gain=.85),
    _Apparatus(("x",), ("channel_p", "channel_q"), "joint", 2, gain=.85),
    _Apparatus(("x",), ("channel_p", "channel_q"), "nonlinear_sensor"),
    _Apparatus(("drive",), ("reading_m", "reading_n"), "joint", 1, 1),
    _Apparatus(("p", "q"), ("meter_r", "meter_s"), "low_rank", 1),
    _Apparatus(("u",), ("sensor_a", "sensor_b"), "control_state"),
    _Apparatus(("u",), ("sensor_a", "sensor_b"), "instrument_state"),
    _Apparatus(("x",), ("channel_p", "channel_q"), "instrument_state", gain=.85),
    _Apparatus(("x",), ("channel_p", "channel_q"), "instrument_state", gain=.85, decay=.94),
    _Apparatus(("u",), ("sensor_a", "sensor_b"), "control_risk", risk_quantile=.9),
    _Apparatus(("u",), ("sensor_a", "sensor_b"), "joint_risk", 1, risk_quantile=.9),
)

TASK_COUNT = len(_APPARATUS)


class UnknownTaskLab:
    """One task interface for finite, continuous-state and noisy mechanisms."""

    def __init__(self, task_index=0, seed=113, profile="development"):
        if task_index not in range(TASK_COUNT):
            raise ValueError(f"unknown task index must be between 0 and {TASK_COUNT - 1}")
        if profile not in {"development", "confirmation"}:
            raise ValueError("profile must be development or confirmation")
        self.task_index, self.seed = task_index, seed
        self._profile = profile
        self._apparatus = _APPARATUS[task_index]
        self.channels = tuple(MeasurementChannel(name, price, "Read an auxiliary instrument.")
                              for name, price in zip(self._apparatus.instruments, (1, 3)))
        self.spec = self.spec_for()
        self.observation_cost = 0
        self.measurement_cost = 0
        self.acquisitions = []
        self._experiments = {}
        self._experiment_oracles = {}

    def _selected(self, channels):
        selected = tuple(dict.fromkeys(channels))
        unknown = set(selected) - {channel.name for channel in self.channels}
        if unknown:
            raise ValueError(f"unknown measurement channels: {sorted(unknown)}")
        return selected

    def spec_for(self, channels=()):
        selected = self._selected(channels)
        controls = self._apparatus.controls
        initial = tuple({name: (-.85, -.3, .55, .9, -.6, .15, .7, -.05)[(i + 3 * j) % 8]
                         for j, name in enumerate(controls)} for i in range(8))
        return TaskSpec(
            f"unknown_{self.task_index + 1:02d}",
            "Apply control sequences and predict the observed response using available instruments.",
            (*controls, *selected), (Probe("initial", initial),), (),
            risk_quantile=self._apparatus.risk_quantile,
            feature_bounds={name: (-1., 1.) for name in controls}, max_probe_length=96)

    @property
    def costs(self):
        return {"observation_cost": self.observation_cost,
                "measurement_cost": self.measurement_cost, "queries": len(self.acquisitions)}

    def _experiment(self, probe):
        if probe.cost > self.spec.max_probe_length:
            raise ValueError("probe exceeds the process's maximum experiment length")
        controls = self._apparatus.controls
        for point in probe.sequence:
            if set(point) != set(controls) or any(not -1. <= point[name] <= 1. for name in controls):
                raise ValueError("probe must contain only controllable inputs within their bounds")
        values = tuple(tuple(float(point[name]) for name in controls) for point in probe.sequence)
        key = probe.probe_id, values
        if key in self._experiments:
            return self._experiments[key]
        random = Random(f"{self.seed}:{self.task_index}:{self._profile}:{key!r}")
        perturbations, rows, oracle_rows = [], [], []
        apparatus = self._apparatus
        gain = apparatus.gain * (1.0 if self._profile == "development" else 1.12)
        slope = .55 if self._profile == "development" else -.7
        state = 0.
        for index, current in enumerate(values):
            # Before startup all drives are zero. Later drives are independent
            # innovations, so their values cannot be recovered from old outputs.
            readings = (0., 0.) if index == 0 else (random.uniform(-1., 1.), random.uniform(-1., 1.))
            perturbations.append(readings)
            delayed = (perturbations[index - apparatus.delay][apparatus.relevant_instrument]
                       if index >= apparatus.delay else 0.)
            target = .15 + slope * current[0]
            if apparatus.response == "control_history":
                target += 1.1 * gain * (values[index - 1][0] if index else 0.)
            elif apparatus.response == "joint":
                target += 1.2 * gain * delayed
            elif apparatus.response == "nonlinear_sensor":
                target += .95 * gain * readings[apparatus.relevant_instrument] ** 2
            elif apparatus.response == "low_rank":
                target = .15 + gain * (.8 * current[0] - .45 * current[1]) ** 2 + .7 * delayed
            elif apparatus.response in {"control_state", "instrument_state"}:
                drive = current[0] if apparatus.response == "control_state" else readings[apparatus.relevant_instrument]
                state = apparatus.decay * state + drive
                target += 1.2 * gain * state
            elif apparatus.response == "joint_risk":
                target += 1.2 * gain * delayed
            conditional_mean, noise_scale = target, 0.
            if index and apparatus.response in {"control_risk", "joint_risk"}:
                exposure = current[0] if apparatus.response == "control_risk" else readings[apparatus.relevant_instrument]
                noise_scale = .08 + .20 * exposure ** 2
                target += noise_scale * random.uniform(-1., 1.)
            features = dict(zip(controls, current))
            features.update(zip(apparatus.instruments, readings))
            rows.append((features, target))
            oracle_rows.append((conditional_mean,
                conditional_mean + (2 * apparatus.risk_quantile - 1) * noise_scale, noise_scale))
        self._experiments[key] = rows
        self._experiment_oracles[key] = oracle_rows
        return rows

    def acquire(self, probe, channels=()):
        selected = self._selected(channels)
        history, observations = [], []
        for readings, target in self._experiment(probe):
            features = {name: readings[name] for name in (*self._apparatus.controls, *selected)}
            observations.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
        measurement_cost = len(observations) * sum(channel.cost_per_row for channel in self.channels
                                                  if channel.name in selected)
        self.observation_cost += len(observations)
        self.measurement_cost += measurement_cost
        self.acquisitions.append({"probe_id": probe.probe_id, "sequence": list(probe.sequence),
                                  "channels": list(selected), "observation_cost": len(observations),
                                  "measurement_cost": measurement_cost})
        return MeasurementBatch(observations, len(observations), measurement_cost, selected)

    def observe(self, probe, channels=()):
        return self.acquire(probe, channels).observations

    def evaluation_rows(self, channels=(), *, seed=1009, length=64, trajectories=4):
        """Return independent fixed trajectories without changing learning costs."""
        return self.evaluation_oracle(channels, seed=seed, length=length,
                                      trajectories=trajectories)["observations"]

    def evaluation_oracle(self, channels=(), *, seed=1009, length=64, trajectories=4):
        """Score-only observations and truth conditional on the full apparatus state.

        This truth is for the experiment driver after learning. It includes the
        hidden noiseless state even when the evaluated predictor did not buy its
        instruments, and is not a conditional expectation over that predictor's
        restricted feature set. Researchers receive only ``acquire`` readings.
        """
        separate = UnknownTaskLab(self.task_index, seed + 1000003, self._profile)
        random = Random(seed)
        rows, means, quantiles, scales = [], [], [], []
        for index in range(trajectories):
            sequence = tuple({name: random.uniform(*bounds)
                              for name, bounds in self.spec.feature_bounds.items()} for _ in range(length))
            probe = Probe(f"evaluation_{index}", sequence)
            rows.extend(separate.observe(probe, channels))
            key = probe.probe_id, tuple(tuple(point[name] for name in self._apparatus.controls)
                                       for point in sequence)
            for mean, quantile, scale in separate._experiment_oracles[key]:
                means.append(mean)
                quantiles.append(quantile)
                scales.append(scale)
        return {"observations": rows, "conditional_means": means, "conditional_quantiles": quantiles,
                "noise_scales": scales, "risk_quantile": self.spec.risk_quantile, "costs": separate.costs}

    def restore(self, acquisitions):
        self.__init__(self.task_index, self.seed, self._profile)
        for item in acquisitions:
            self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item["channels"])

    def reset(self):
        """Each probe already starts from the same zero physical state."""

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
