"""A reset apparatus whose paid instrument has a finite temporal response."""

from random import Random

from .measurement_lab import MeasurementBatch, MeasurementChannel
from .models import Observation, Probe, TaskSpec


class DynamicInstrumentLab:
    """Offer controls and a paid reading without describing the hidden sensor law.

    The five tasks revisit one apparatus, change its instrument response, change
    only its output mechanism, then return to the original apparatus. Every
    experiment starts with zero prehistory, including previous sensor values.
    """

    channels = (MeasurementChannel("meter", 3, "Read the auxiliary instrument."),)

    def __init__(self, task_index=0, seed=113, apparatus_seed=None, profile="development"):
        if task_index not in range(5):
            raise ValueError("dynamic instrument task index must be between 0 and 4")
        if profile not in {"development", "confirmation"}:
            raise ValueError("profile must be development or confirmation")
        self.task_index, self.seed = task_index, seed
        self._apparatus_seed = seed if apparatus_seed is None else apparatus_seed
        self._profile = profile
        jitter = Random(self._apparatus_seed).uniform(-.04, .04)
        self._delay = 2 if task_index in (2, 3) else 1
        self._lag_gain = (-.65 if task_index in (2, 3) else .9) + jitter
        self._current_gain, self._delayed_square_gain = .4, .2
        self._response_offset, self._response_control_gain = .12, .35
        self._square_gain = 1.4 if task_index == 3 else 1.
        self._response_history_gain = -.3 if task_index == 3 else .45
        if profile == "confirmation":
            random = Random(self._apparatus_seed)
            delays = random.sample((1, 2, 3), 2)
            instruments = [(random.uniform(.25, .65),
                random.choice((-1., 1.)) * random.uniform(.45, 1.05),
                random.uniform(.08, .28)) for _ in range(2)]
            responses = [(random.uniform(-.25, .25), random.uniform(-.6, .6),
                random.uniform(.6, 1.5), random.choice((-1., 1.)) * random.uniform(.2, .65))
                for _ in range(2)]
            instrument = int(task_index in (2, 3))
            self._delay = delays[instrument]
            self._current_gain, self._lag_gain, self._delayed_square_gain = instruments[instrument]
            (self._response_offset, self._response_control_gain,
             self._square_gain, self._response_history_gain) = responses[int(task_index == 3)]
        self.observation_cost, self.measurement_cost = 0, 0
        self.acquisitions = []
        self.spec = self.spec_for()

    def _selected(self, channels):
        selected = tuple(dict.fromkeys(channels))
        if set(selected) - {channel.name for channel in self.channels}:
            raise ValueError("the available auxiliary instrument is meter")
        return selected

    def spec_for(self, channels=()):
        selected = self._selected(channels)
        initial = tuple({"u": value} for value in (.8, 0., 0., -.7, 0., .4, -.3, 0.))
        return TaskSpec("dynamic_instrument", "Apply control sequences and predict the observed response using available instruments.",
            ("u", *selected), (Probe("initial", initial),), (),
            feature_bounds={"u": (-1., 1.)}, max_probe_length=96)

    @property
    def costs(self):
        return {"observation_cost": self.observation_cost, "measurement_cost": self.measurement_cost,
                "queries": len(self.acquisitions)}

    def acquire(self, probe, channels=()):
        selected = self._selected(channels)
        if probe.cost > self.spec.max_probe_length:
            raise ValueError("probe exceeds the maximum experiment length")
        if any(set(point) != {"u"} or not -1. <= point["u"] <= 1. for point in probe.sequence):
            raise ValueError("probe must contain control u in [-1, 1]")
        controls, readings, history, observations = [], [], [], []
        for point in probe.sequence:
            control = float(point["u"])
            delayed = controls[-self._delay] if len(controls) >= self._delay else 0.
            reading = (self._current_gain * control + self._lag_gain * delayed
                       + self._delayed_square_gain * delayed ** 2)
            previous_reading = readings[-1] if readings else 0.
            target = (self._response_offset + self._response_control_gain * control
                      + self._square_gain * reading ** 2 + self._response_history_gain * previous_reading)
            features = {"u": control, **({"meter": reading} if selected else {})}
            observations.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
            controls.append(control)
            readings.append(reading)
        measurement_cost = len(observations) * sum(channel.cost_per_row for channel in self.channels
                                                  if channel.name in selected)
        self.observation_cost += len(observations)
        self.measurement_cost += measurement_cost
        self.acquisitions.append({"probe_id": probe.probe_id, "sequence": list(probe.sequence),
            "channels": list(selected), "observation_cost": len(observations), "measurement_cost": measurement_cost})
        return MeasurementBatch(observations, len(observations), measurement_cost, selected)

    def observe(self, probe, channels=()):
        return self.acquire(probe, channels).observations

    def evaluation_rows(self, channels=(), *, seed=1009, length=64, trajectories=4):
        separate = DynamicInstrumentLab(self.task_index, seed, self._apparatus_seed, self._profile)
        random, rows = Random(seed), []
        for trajectory in range(trajectories):
            sequence = tuple({"u": random.uniform(-1., 1.)} for _ in range(length))
            rows.extend(separate.observe(Probe(f"evaluation_{trajectory}", sequence), channels))
        return rows

    def restore(self, acquisitions):
        self.__init__(self.task_index, self.seed, self._apparatus_seed, self._profile)
        for item in acquisitions:
            self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item["channels"])

    def reset(self):
        """Each acquired probe already starts with zero physical prehistory."""

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
