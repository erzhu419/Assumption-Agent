"""Controlled temporal predicates measured through an unknown delayed instrument."""

from random import Random
from math import sqrt

from .measurement_lab import MeasurementBatch, MeasurementChannel
from .models import Observation, Probe, TaskSpec


class TemporalMethodLab:
    """Supply the same paid physical interface to both experimental planners."""

    def __init__(self, case="window", stage=0, seed=151, *, center=.21, half_width=1e-4,
                 response_gain=2., calibration_rows=48):
        if case not in {"window", "event"} or stage not in range(5):
            raise ValueError("expected a window/event case and one of five stages")
        self.case, self.stage, self.seed = case, stage, seed
        self._center, self._half_width = center, half_width
        self._gain, self._calibration_rows = response_gain, calibration_rows
        self.control_name, self.sensor_name = (("u", "meter") if stage in (0, 4) else ("drive", "reading"))
        self.channels = (MeasurementChannel(self.sensor_name, 3, "Read the auxiliary instrument."),)
        self.observation_cost, self.measurement_cost = 0, 0
        self.acquisitions = []
        self._drifted = False
        self.spec = self.spec_for()

    def spec_for(self, channels=()):
        selected = self._selected(channels)
        return TaskSpec("temporal_method_apparatus", "Apply public control sequences and predict the measured response.",
            (self.control_name, *selected), (), (), feature_bounds={self.control_name: (-1., 1.)}, max_probe_length=48)

    def _selected(self, channels):
        selected = tuple(dict.fromkeys(channels))
        if set(selected) - {self.sensor_name}:
            raise ValueError("unknown instrument")
        return selected

    @property
    def costs(self):
        return {"observation_cost": self.observation_cost, "measurement_cost": self.measurement_cost,
                "queries": len(self.acquisitions)}

    def _parameters(self):
        jitter = Random(self.seed).uniform(-.025, .025)
        if self.stage == 3 and self._drifted:
            return 3, -.7 + jitter
        if self.stage in (2, 3):
            return 2, .6 + jitter
        return 1, .8 + jitter

    def acquire(self, probe, channels=()):
        selected = self._selected(channels)
        if probe.cost > self.spec.max_probe_length or any(
                set(point) != {self.control_name} or not -1. <= point[self.control_name] <= 1.
                for point in probe.sequence):
            raise ValueError("probe must use public bounded controls within maximum length")
        if self.stage == 3 and self.observation_cost >= self._calibration_rows:
            self._drifted = True
        delay, gain = self._parameters()
        controls, history, observations, previous_active = [], [], [], 0.
        for point in probe.sequence:
            control = point[self.control_name]
            delayed = controls[-delay] if len(controls) >= delay else 0.
            reading = .4 * control + gain * delayed + .15 * delayed ** 2
            active = float(self._center - self._half_width <= reading < self._center + self._half_width)
            event = active - previous_active if controls else 0.
            target = self._gain * (event if self.case == "event" else active)
            features = {self.control_name: control, **({self.sensor_name: reading} if selected else {})}
            observations.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
            controls.append(control)
            previous_active = active
        measurement_cost = 3 * len(observations) if selected else 0
        self.observation_cost += len(observations)
        self.measurement_cost += measurement_cost
        self.acquisitions.append({"probe_id": probe.probe_id, "sequence": list(probe.sequence),
            "channels": list(selected), "observation_cost": len(observations), "measurement_cost": measurement_cost})
        return MeasurementBatch(observations, len(observations), measurement_cost, selected)

    def evaluation_rows(self, *, seed=1009, trajectories=4, length=32):
        """Independent random and activation/exit trajectories, never learner feedback."""
        separate = TemporalMethodLab(self.case, self.stage, self.seed, center=self._center,
            half_width=self._half_width, response_gain=self._gain, calibration_rows=self._calibration_rows)
        separate._drifted = self._drifted or (self.stage == 3 and self.observation_cost >= self._calibration_rows)
        random, rows = Random(seed), []
        delay, gain = separate._parameters()
        for trajectory in range(trajectories):
            controls = [random.uniform(-1., 1.) for _ in range(length)]
            if trajectory % 2:
                controls = [random.uniform(-.1, .1) for _ in range(length)]
                index = delay + 1
                past = controls[index - delay]
                controls[index] = (self._center - gain * past - .15 * past ** 2) / .4
                if not -1. <= controls[index] <= 1.:
                    # A legal random prefix need not allow this activation.
                    # Change its delayed control and solve the same reading;
                    # clipping the final control would erase the test event.
                    for current in (0., -1., 1.):
                        discriminant = gain * gain + .6 * (self._center - .4 * current)
                        if discriminant < 0.:
                            continue
                        roots = [(-gain + sign * sqrt(discriminant)) / .3 for sign in (-1., 1.)]
                        feasible = [value for value in roots if -1. <= value <= 1.]
                        if feasible:
                            controls[index - delay] = min(feasible, key=lambda value: abs(value - past))
                            controls[index] = current
                            break
                    else:
                        raise ValueError("activation center is unreachable with public bounded controls")
                controls[index + 1] = 0.
            sequence = tuple({self.control_name: value} for value in controls)
            rows.extend(separate.acquire(Probe(f"terminal:{trajectory}", sequence), (self.sensor_name,)).observations)
        return rows
