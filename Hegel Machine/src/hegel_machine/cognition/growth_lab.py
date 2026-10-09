"""One response interface with changing laws, surfaces and observability.

The stage selects simulator truth for the study runner. The researcher receives
only anonymous controls, equally priced instruments and purchased responses.
Every acquisition draws a fresh disturbance, even when the probe is repeated.
"""

from dataclasses import replace
from random import Random

from .measurement_lab import MeasurementBatch, MeasurementChannel
from .models import Observation, Probe, TaskSpec


STAGES = (
    "nonlinear_source", "renamed_transfer", "local_scope_change",
    "hidden_input", "renamed_hidden_input", "stochastic_response",
    "linear_simplification", "original_return",
)


class GrowthLab:
    def __init__(self, *, seed=8601, stage=0, realization_seed=None):
        if stage not in range(len(STAGES)):
            raise ValueError("stage must identify one of the eight study tasks")
        self._stage = stage
        self._settings = {"seed": seed, "stage": stage}
        surface = (0, 1, 1, 1, 2, 2, 2, 0)[stage]
        shape = Random(seed)
        self._square = shape.uniform(.8, 1.2)
        self._cross = shape.uniform(.3, .7)
        names = Random(seed + 1009 * surface)
        controls = [f"c{value:03d}" for value in names.sample(range(100, 900), 2)]
        instruments = [f"m{value:03d}" for value in names.sample(range(100, 900), 2)]
        self._roles = tuple(controls)
        self._signal, self._decoy = instruments
        # Both names and published ordering change. Neither order encodes roles.
        names.shuffle(controls)
        names.shuffle(instruments)
        self.channels = tuple(MeasurementChannel(name, 2,
            "Read an auxiliary process instrument.") for name in instruments)
        self.spec = TaskSpec("growth_process",
            "Predict the response of a process using purchased experiments.",
            tuple(controls), (), (), feature_bounds={name: (-1., 1.) for name in controls},
            max_probe_length=64)
        self._random = Random(seed + 7919 * stage if realization_seed is None else realization_seed)
        self.observation_cost = self.measurement_cost = 0
        self.acquisitions = []
        self._last_means = []

    def spec_for(self, channels=()):
        selected = tuple(dict.fromkeys(channels))
        if not set(selected) <= {channel.name for channel in self.channels}:
            raise ValueError("unknown instrument")
        return replace(self.spec, feature_names=(*self.spec.feature_names, *selected))

    @property
    def costs(self):
        return {"observation_cost": self.observation_cost,
                "measurement_cost": self.measurement_cost, "queries": len(self.acquisitions)}

    def random_probe(self, seed, *, length=24, name="sample"):
        random = Random(seed)
        return Probe(name, tuple({key: random.uniform(*bounds)
            for key, bounds in self.spec.feature_bounds.items()} for _ in range(length)))

    def acquire(self, probe, channels=()):
        selected = tuple(dict.fromkeys(channels))
        self.spec_for(selected)
        if not 1 <= probe.cost <= self.spec.max_probe_length:
            raise ValueError("probe length must be between one and 64")
        rows, means, history = [], [], []
        for point in probe.sequence:
            if set(point) != set(self.spec.feature_bounds) or any(
                    not lower <= point[name] <= upper
                    for name, (lower, upper) in self.spec.feature_bounds.items()):
                raise ValueError("supply the public controls within their bounds")
            x, y = (point[name] for name in self._roles)
            mean = self._square * x * x + self._cross * x * y
            if self._stage == 2:
                mean += .8 * max(0., x)
            elif self._stage == 6:
                mean = .8 * x - .3 * y
            latent = self._random.uniform(-.6, .6)
            decoy = self._random.uniform(-.6, .6)
            if self._stage in (3, 4):
                mean += latent
            target = mean + (self._random.gauss(0., .12) if self._stage == 5 else 0.)
            signals = {self._signal: latent, self._decoy: decoy}
            features = {**point, **{name: signals[name] for name in selected}}
            rows.append(Observation(features, target, tuple(history)))
            means.append(Observation(features, mean, tuple(history)))
            history.append({**features, "__target__": target})
        measurement_cost = probe.cost * sum(channel.cost_per_row for channel in self.channels
                                             if channel.name in selected)
        self.observation_cost += probe.cost
        self.measurement_cost += measurement_cost
        self._last_means = means
        self.acquisitions.append({"probe_id": probe.probe_id,
            "sequence": [dict(point) for point in probe.sequence], "channels": list(selected),
            "observation_cost": probe.cost, "measurement_cost": measurement_cost})
        return MeasurementBatch(rows, probe.cost, measurement_cost, selected)

    def observe(self, probe, channels=()):
        return self.acquire(probe, channels).observations

    def evaluation_rows(self, channels=(), *, seed=1000003, length=64, trajectories=4):
        """Evaluator-only independent responses and their latent conditional means."""
        separate = GrowthLab(**self._settings, realization_seed=seed + 2147483)
        actual, means = [], []
        for trajectory in range(trajectories):
            probe = separate.random_probe(seed + 10007 * trajectory,
                length=length, name=f"terminal_{trajectory}")
            actual.extend(separate.observe(probe, channels))
            means.extend(separate._last_means)
        return {"observations": actual, "mean_observations": means,
                "irreducible_noise_variance": .12 ** 2 if self._stage == 5 else 0.,
                "costs": separate.costs}

    def reset(self):
        """Each row is a new setup; resetting does not replay random noise."""

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
