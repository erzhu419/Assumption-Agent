"""A stream of apparatuses requiring a distinguishing procedure to grow and transfer."""

from random import Random

from .measurement_lab import MeasurementBatch
from .models import Observation, Probe, TaskSpec
from .stateful_instrument_lab import StatefulInstrumentLab


class InstrumentMethodLab(StatefulInstrumentLab):
    """Three new apparatuses, each first restricted to a short reset prefix.

    The first apparatus admits three-step probes before its interface expands;
    the second and third admit four-step probes. Finite instruments truncate
    their recurrent counterparts exactly at these short-prefix boundaries.
    The family and truncation depth are private physical parameters.
    """

    def __init__(self, task_index=0, seed=113, apparatus_seed=None, instrument_mode="recurrent"):
        if task_index not in range(6):
            raise ValueError("instrument method task index must be between 0 and 5")
        if instrument_mode not in ("finite", "recurrent"):
            raise ValueError("instrument mode must be finite or recurrent")
        self._task_index, self._instrument_mode = task_index, instrument_mode
        self._root_apparatus_seed = seed if apparatus_seed is None else apparatus_seed
        self._memory_depth = (2, 3, 3)[task_index // 2]
        self._public_max_length = 24 if task_index % 2 else self._memory_depth + 1
        physical_seed = self._root_apparatus_seed + 100003 * (task_index // 2)
        super().__init__(0, seed, physical_seed, states=1)
        self.task_index = task_index

    def spec_for(self, channels=()):
        selected = self._selected(channels)
        initial = tuple({"u": value} for value in
                        (.8, 0., 0., -.7, 0., .4, -.3, 0.)[:self._public_max_length])
        return TaskSpec(f"instrument_method_{self._task_index}_{self.seed}",
            "Apply control sequences and predict the observed response using available instruments.",
            ("u", *selected), (Probe("initial", initial),), (),
            feature_bounds={"u": (-1., 1.)}, max_probe_length=self._public_max_length)

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
                           for lag in range(1, self._memory_depth + 1) if len(controls) >= lag)
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
        separate = InstrumentMethodLab(self.task_index, seed, self._root_apparatus_seed, self._instrument_mode)
        random, rows = Random(seed), []
        for trajectory in range(trajectories):
            sequence = tuple({"u": random.uniform(-1., 1.)}
                             for _ in range(min(length, separate.spec.max_probe_length)))
            rows.extend(separate.observe(Probe(f"evaluation_{trajectory}", sequence), channels))
        return rows

    def restore(self, acquisitions):
        self.__init__(self.task_index, self.seed, self._root_apparatus_seed, self._instrument_mode)
        for item in acquisitions:
            self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item["channels"])
