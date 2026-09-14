"""Scalar apparatus sequences for research through a public observation interface."""

from math import hypot
from random import Random

from .models import Observation, TaskSpec


class ScalarApparatus:
    """An unknown scalar response reached only by paid point measurements."""

    def __init__(self, response, task_id):
        self._response = response
        self.spec = TaskSpec(task_id, "Predict the measured response under controllable inputs",
            ("x", "z"), (), (), feature_bounds={"x": (-1.5, 1.5), "z": (-1.5, 1.5)},
            max_probe_length=16)
        self.observation_count = 0
        self.probes = []

    def observe(self, probe):
        if probe.cost > self.spec.max_probe_length:
            raise ValueError("the requested trajectory exceeds the public apparatus limit")
        for point in probe.sequence:
            if set(point) != set(self.spec.feature_names):
                raise ValueError("each measurement must set both public controls")
            if any(not lower <= point[name] <= upper
                   for name, (lower, upper) in self.spec.feature_bounds.items()):
                raise ValueError("the requested control lies outside the public range")
        self.observation_count += probe.cost
        self.probes.append(probe)
        return [Observation(dict(point), self._response(point["x"], point["z"]))
                for point in probe.sequence]


def make_meta_prior_sequence(seed, profile="quadratic"):
    """Private apparatus laws; neither the stage nor coefficients enter TaskSpec."""
    rng = Random(seed)
    a, b = rng.uniform(.5, 1.2), rng.uniform(-1.5, -.7)
    norm = hypot(a, b)
    a, b = a/norm, b/norm
    scale, linear, offset = rng.uniform(.7, 1.4), rng.uniform(-.3, .3), rng.uniform(-.2, .2)
    readout_scale, readout_offset = rng.uniform(1.4, 2.0), rng.uniform(.2, .5)
    orthogonal = rng.uniform(.5, .9)
    slopes = rng.uniform(.7, 1.3)
    source = lambda x, z: offset + linear*(a*x+b*z) + scale*(a*x+b*z)**2
    if profile == "quadratic":
        readout = lambda x, z: readout_offset + readout_scale*source(x, z)
    elif profile == "nested_readout":
        readout = lambda x, z: readout_offset + .25*source(x, z) + readout_scale*source(x, z)**2
    else:
        raise ValueError(f"unknown study profile: {profile}")
    changed = lambda x, z: readout(x, z) + orthogonal*(-b*x+a*z)**2
    simple = lambda x, z: .1 + slopes*x
    return [ScalarApparatus(response, f"response_{index}")
            for index, response in enumerate((source, readout, changed, simple))]
