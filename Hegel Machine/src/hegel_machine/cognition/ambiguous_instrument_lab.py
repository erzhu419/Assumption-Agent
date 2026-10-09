"""Instruments indistinguishable under short resets, then exposed to longer probes."""

from random import Random

from .measurement_lab import MeasurementBatch
from .models import Observation, Probe, TaskSpec
from .stateful_instrument_lab import StatefulInstrumentLab


class AmbiguousInstrumentLab(StatefulInstrumentLab):
    """Expand a public experimental horizon without changing the source physics.

    A finite instrument uses the first three impulse coefficients of a recurrent
    instrument. Every legal source probe therefore agrees across those physical
    families. Longer probes can distinguish them. Stage two changes both the
    instrument parameters and the response; stage three returns to the source.
    """

    def __init__(self, task_index=0, seed=113, apparatus_seed=None, instrument_mode="recurrent",
                 response_seed=None):
        if task_index not in range(4):
            raise ValueError("instrument competition task index must be between 0 and 3")
        if instrument_mode not in ("finite2", "recurrent"):
            raise ValueError("instrument mode must be finite2 or recurrent")
        self._stage, self._instrument_mode = task_index, instrument_mode
        super().__init__((0, 0, 2, 3)[task_index], seed, apparatus_seed, states=1,
                         response_seed=response_seed)
        self.task_index = task_index

    def spec_for(self, channels=()):
        selected = self._selected(channels)
        length = 3 if self._stage == 0 else 24
        initial = tuple({"u": value} for value in (.8, 0., 0., -.7, 0., .4, -.3, 0.)[:length])
        return TaskSpec("instrument_competition", "Apply control sequences and predict the observed response using available instruments.",
            ("u", *selected), (Probe("initial", initial),), (),
            feature_bounds={"u": (-1., 1.)}, max_probe_length=length)

    def acquire(self, probe, channels=()):
        if self._instrument_mode == "recurrent":
            return super().acquire(probe, channels)
        selected = self._selected(channels)
        if probe.cost > self.spec.max_probe_length:
            raise ValueError("probe exceeds the maximum experiment length")
        if any(set(point) != {"u"} or not -1. <= point["u"] <= 1. for point in probe.sequence):
            raise ValueError("probe must contain control u in [-1, 1]")
        controls, history, observations, previous_reading = [], [], [], 0.
        for point in probe.sequence:
            control = float(point["u"])
            reading = (self._current_gain + self._state_gain) * control
            reading += sum(self._state_gain * self._decay ** lag * controls[-lag]
                           for lag in (1, 2) if len(controls) >= lag)
            target = (self._response_offset + self._response_control_gain * control
                      + self._response_square_gain * reading * reading
                      + self._response_history_gain * previous_reading)
            features = {"u": control, **({"meter": reading} if selected else {})}
            observations.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
            controls.append(control)
            previous_reading = reading
        measurement_cost = len(observations) * sum(channel.cost_per_row for channel in self.channels
                                                  if channel.name in selected)
        self.observation_cost += len(observations)
        self.measurement_cost += measurement_cost
        self.acquisitions.append({"probe_id": probe.probe_id, "sequence": list(probe.sequence),
            "channels": list(selected), "observation_cost": len(observations),
            "measurement_cost": measurement_cost})
        return MeasurementBatch(observations, len(observations), measurement_cost, selected)

    def evaluation_rows(self, channels=(), *, seed=1009, length=24, trajectories=4):
        separate = AmbiguousInstrumentLab(self.task_index, seed, self._apparatus_seed,
                                          self._instrument_mode, self._response_seed)
        random, rows = Random(seed), []
        for trajectory in range(trajectories):
            sequence = tuple({"u": random.uniform(-1., 1.)}
                             for _ in range(min(length, separate.spec.max_probe_length)))
            rows.extend(separate.observe(Probe(f"evaluation_{trajectory}", sequence), channels))
        return rows

    def restore(self, acquisitions):
        self.__init__(self.task_index, self.seed, self._apparatus_seed,
                      self._instrument_mode, self._response_seed)
        for item in acquisitions:
            self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item["channels"])
