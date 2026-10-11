"""A paid instrument with recurrent internal state and separately changing readout."""

from random import Random

from .measurement_lab import MeasurementBatch, MeasurementChannel
from .models import Observation, Probe, TaskSpec


class StatefulInstrumentLab:
    """Apply controls to an apparatus whose complete state resets with each probe.

    Stage zero learns a source, one changes only its instrument, two changes
    both the current instrument and the response, and three returns to source.
    The public controls and measurement contract remain identical throughout.
    """

    channels = (MeasurementChannel("meter", 3, "Read the auxiliary instrument."),)

    def __init__(self, task_index=0, seed=113, apparatus_seed=None, states=1):
        if task_index not in range(4):
            raise ValueError("stateful instrument task index must be between 0 and 3")
        if states not in (1, 2):
            raise ValueError("the instrument supports one or two internal states")
        self.task_index, self.seed = task_index, seed
        self._states = states
        self._apparatus_seed = seed if apparatus_seed is None else apparatus_seed
        random = Random(self._apparatus_seed)
        instruments = [
            (random.uniform(.62, .82), random.uniform(-.7, -.42), random.uniform(.25, .55),
             random.uniform(.55, .9), random.uniform(.25, .55)),
            (random.uniform(-.82, -.58), random.uniform(.35, .57), random.uniform(.2, .6),
             random.uniform(.5, .95), random.uniform(.3, .6)),
            (random.uniform(.86, .94), random.uniform(-.5, -.25), random.uniform(.2, .6),
             random.uniform(.45, .85), random.uniform(.25, .6)),
        ]
        responses = [(random.uniform(-.2, .2), random.uniform(-.45, .45),
                      random.uniform(.65, 1.35), random.choice((-1., 1.)) * random.uniform(.25, .6))
                     for _ in range(2)]
        (self._decay, self._second_decay, self._current_gain,
         self._state_gain, self._second_state_gain) = instruments[(0, 1, 2, 0)[task_index]]
        (self._response_offset, self._response_control_gain,
         self._response_square_gain, self._response_history_gain) = responses[int(task_index == 2)]
        self.observation_cost = self.measurement_cost = 0
        self.acquisitions = []
        self.spec = self.spec_for()

    def _selected(self, channels):
        selected = tuple(dict.fromkeys(channels))
        if set(selected) - {"meter"}:
            raise ValueError("the available auxiliary instrument is meter")
        return selected

    def spec_for(self, channels=()):
        selected = self._selected(channels)
        initial = tuple({"u": value} for value in (.8, 0., 0., -.7, 0., .4, -.3, 0.))
        return TaskSpec("stateful_instrument", "Apply control sequences and predict the observed response using available instruments.",
            ("u", *selected), (Probe("initial", initial),), (),
            feature_bounds={"u": (-1., 1.)}, max_probe_length=96)

    @property
    def costs(self):
        return {"observation_cost": self.observation_cost,
                "measurement_cost": self.measurement_cost, "queries": len(self.acquisitions)}

    def acquire(self, probe, channels=()):
        selected = self._selected(channels)
        if probe.cost > self.spec.max_probe_length:
            raise ValueError("probe exceeds the maximum experiment length")
        if any(set(point) != {"u"} or not -1. <= point["u"] <= 1. for point in probe.sequence):
            raise ValueError("probe must contain control u in [-1, 1]")
        state, second_state, previous_reading, history, observations = 0., 0., 0., [], []
        for point in probe.sequence:
            control = float(point["u"])
            state = self._decay * state + control
            second_state = self._second_decay * second_state + control
            reading = self._current_gain * control + self._state_gain * state
            if self._states == 2:
                reading += self._second_state_gain * second_state
            target = (self._response_offset + self._response_control_gain * control
                      + self._response_square_gain * reading * reading
                      + self._response_history_gain * previous_reading)
            features = {"u": control, **({"meter": reading} if selected else {})}
            observations.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
            previous_reading = reading
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
        separate = StatefulInstrumentLab(self.task_index, seed, self._apparatus_seed, self._states)
        random, rows = Random(seed), []
        for trajectory in range(trajectories):
            sequence = tuple({"u": random.uniform(-1., 1.)} for _ in range(length))
            rows.extend(separate.observe(Probe(f"evaluation_{trajectory}", sequence), channels))
        return rows

    def restore(self, acquisitions):
        self.__init__(self.task_index, self.seed, self._apparatus_seed, self._states)
        for item in acquisitions:
            self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item["channels"])

    def reset(self):
        """Each acquired probe already resets the physical state and readings."""

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
