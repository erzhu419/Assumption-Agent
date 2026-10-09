"""Finite response analogs for controlled studies of research mechanisms.

Response families and intervention labels belong to the simulator protocol.
The public interface contains only anonymous controls, their units and optional
equally priced instruments. Alternative surfaces implement the same physical
response with different algebra and measurement units.
"""

from dataclasses import replace

from .growth_lab import GrowthLab
from .measurement_lab import MeasurementBatch
from .models import Observation


FAMILIES = ("quadratic_exchange", "cubic_exchange", "rectified_exchange")


class GrowthPaperLab(GrowthLab):
    def __init__(self, *, seed=9301, stage=0, family=None, new_expression=False,
                 controls_only=False, realization_seed=None):
        super().__init__(seed=seed, stage=stage, realization_seed=realization_seed)
        self._family = seed % len(FAMILIES) if family is None else family
        if self._family not in range(len(FAMILIES)):
            raise ValueError("unknown response family")
        self._new_expression = new_expression
        self._settings = {"seed": seed, "stage": stage, "family": self._family,
            "new_expression": new_expression, "controls_only": controls_only}
        self._surface = (0, 1, 1, 1, 2, 2, 2, 0)[stage]
        self._units = (((0., 1.), (0., 1.)), ((0., 2.), (3., 2.)),
                       ((-2., 1.), (.5, 1.)))[self._surface]
        bounds = {name: (offset - scale, offset + scale)
            for name, (offset, scale) in zip(self._roles, self._units)}
        self.spec = replace(self.spec, feature_bounds=bounds)
        self._instrument_unit = ((0., 1.), (.3, 1.2), (-.2, .8))[self._surface]
        if stage == 4:
            original = GrowthLab(seed=seed, stage=3)
            old_rank = [channel.name for channel in original.channels].index(original._signal)
            order = (self._signal, self._decoy) if old_rank == 1 else (self._decoy, self._signal)
            by_name = {channel.name: channel for channel in self.channels}
            self.channels = tuple(by_name[name] for name in order)
        if controls_only:
            self.channels = ()
        self.paid_observations = []
        self._last_variances = []

    def _response(self, x, y):
        if self._surface == 0:
            main = (x * x, x * x * x, abs(x))[self._family]
            return self._square * main + self._cross * x * y
        # Independently written algebraic analog: the mixed term comes from
        # subtracting isolated responses from a joint quadratic response.
        main = (x ** 2, x ** 3, max(x, -x))[self._family]
        interaction = ((x + y) ** 2 - x ** 2 - y ** 2) / 2.
        return self._square * main + self._cross * interaction

    def acquire(self, probe, channels=()):
        selected = tuple(dict.fromkeys(channels))
        self.spec_for(selected)
        if not 1 <= probe.cost <= self.spec.max_probe_length:
            raise ValueError("probe length must be between one and 64")
        rows, means, variances, history = [], [], [], []
        for point in probe.sequence:
            if set(point) != set(self.spec.feature_bounds) or any(
                    not lower <= point[name] <= upper
                    for name, (lower, upper) in self.spec.feature_bounds.items()):
                raise ValueError("supply the public controls within their bounds")
            x, y = ((point[name] - offset) / scale
                    for name, (offset, scale) in zip(self._roles, self._units))
            mean = self._response(x, y)
            if self._new_expression:
                mean += .35 * y ** 3
            if self._stage == 2:
                mean += .8 * max(0., x)
            elif self._stage == 6:
                mean = .8 * x - .3 * y
            latent = self._random.uniform(-.6, .6)
            decoy = self._random.uniform(-.6, .6)
            if self._stage in (3, 4):
                mean += latent
            sigma = .08 + .16 * abs(x) if self._stage == 5 else 0.
            target = mean + (self._random.gauss(0., sigma) if sigma else 0.)
            offset, scale = self._instrument_unit
            signals = {self._signal: offset + scale * latent,
                       self._decoy: offset + scale * decoy}
            features = {**point, **{name: signals[name] for name in selected}}
            rows.append(Observation(features, target, tuple(history)))
            means.append(Observation(features, mean, tuple(history)))
            variances.append(sigma ** 2)
            history.append({**features, "__target__": target})
        price = probe.cost * sum(channel.cost_per_row for channel in self.channels
                                 if channel.name in selected)
        self.observation_cost += probe.cost
        self.measurement_cost += price
        self._last_means, self._last_variances = means, variances
        self.paid_observations.extend(rows)
        self.acquisitions.append({"probe_id": probe.probe_id,
            "sequence": [dict(point) for point in probe.sequence], "channels": list(selected),
            "observation_cost": probe.cost, "measurement_cost": price})
        return MeasurementBatch(rows, probe.cost, price, selected)

    def evaluation_rows(self, channels=(), *, seed=1000003, length=64, trajectories=4):
        """Independent evaluation; latent means never enter the research stream."""
        separate = GrowthPaperLab(**self._settings, realization_seed=seed + 2147483)
        actual, means, variances = [], [], []
        for index in range(trajectories):
            probe = separate.random_probe(seed + 10007 * index,
                length=length, name=f"terminal_{index}")
            actual.extend(separate.observe(probe, channels))
            means.extend(separate._last_means)
            variances.extend(separate._last_variances)
        return {"observations": actual, "mean_observations": means,
                "noise_variances": variances,
                "irreducible_noise_variance": sum(variances) / len(variances),
                "costs": separate.costs}
