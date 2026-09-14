"""Small vector apparatuses with paid readouts and controllable initial states.

The learner receives commands, measured one-step responses and a later task
readout. It never receives the response matrix used inside the apparatus.
"""

from dataclasses import asdict, dataclass
from random import Random

from .models import Observation, Probe, TaskSpec


@dataclass
class MechanismBatch:
    observations: list[Observation]
    responses: list[list[float]]
    trajectories: list[list[list[float]]]
    observation_cost: int
    measurement_cost: int

    def to_dict(self):
        return {**asdict(self), "observations": [asdict(row) for row in self.observations]}


class MechanismLab:
    """Four finite mechanisms; ``broken`` changes a structural assumption."""

    def __init__(self, family="low_rank", variant="regular", seed=71):
        if family not in {"low_rank", "modules", "dynamics", "topology", "order"}:
            raise ValueError("unknown mechanism family")
        if variant not in {"regular", "broken"}:
            raise ValueError("variant must be regular or broken")
        self.family, self.variant, self.seed = family, variant, seed
        self.horizon = 3 if family in {"dynamics", "topology"} else 1
        self.target_index = 3 if family == "topology" else 0
        self.feature_names = tuple(f"x{i}" for i in range(4))
        # Parameters vary between installations, while the learner only sees
        # public controls and measured responses.
        scale = 1.0 + Random(seed).uniform(-.03, .03)
        if family == "low_rank":
            left = ((1., .4), (.3, 1.), (-.8, .2), (.6, -.7))
            right = ((.4, .3, -.5, .7), (.6, -.4, .2, .5))
            matrix = [[scale * sum(a * b for a, b in zip(row, column))
                       for column in zip(*right)] for row in left]
            if variant == "broken":
                for i in range(4):
                    matrix[i][3] += .35 * (i + 1)
        elif family == "modules":
            matrix = [[.7, .25, 0., 0.], [.3, -.4, 0., 0.],
                      [0., 0., .6, -.2], [0., 0., .4, .5]]
            matrix = [[scale * value for value in row] for row in matrix]
            if variant == "broken":
                matrix[0][2] = .55
        elif family == "dynamics":
            basis = ((.5, .5, .5, .5), (.5, -.5, .5, -.5),
                     (.5, .5, -.5, -.5), (.5, -.5, -.5, .5))
            rates = (1.12 if variant == "broken" else .82, .68, .12, .04)
            matrix = [[sum(rates[k] * basis[k][i] * basis[k][j] for k in range(4))
                       for j in range(4)] for i in range(4)]
        elif family == "topology":
            matrix = [[0., 0., 0., 0.], [.6 * scale, 0., 0., 0.],
                      [0., .7 * scale, 0., 0.], [0., 0., .8 * scale, 0.]]
            if variant == "broken":
                matrix[2][1] = 0.
                matrix[3][0] = .3
        else:
            matrix = None
        self._response = matrix
        self.acquisitions = []
        names = self.feature_names + (("gain",) if family == "topology" else ())
        self.spec = TaskSpec(f"mechanism:{family}:{variant}:{seed}",
            "Predict a measured response from controlled initial coordinates", names, (), (),
            context={"mechanism_family": family, "horizon": str(self.horizon),
                     "system_id": f"mechanism:{family}:{seed}"},
            feature_bounds={name: ((.8, 1.2) if name == "gain" else (0., 1.)
                                  if family == "order" and name == "x0" else (-1., 1.)) for name in names},
            max_probe_length=32)

    @property
    def measurement_cost_per_row(self):
        if self.family == "order":
            return 1
        return 4 * self.horizon + 1

    @property
    def costs(self):
        return {"observation_cost": sum(row["observation_cost"] for row in self.acquisitions),
                "measurement_cost": sum(row["measurement_cost"] for row in self.acquisitions),
                "queries": len(self.acquisitions)}

    def point(self, values, *, gain=1.):
        row = dict(zip(self.feature_names, map(float, values)))
        if self.family == "topology":
            row["gain"] = float(gain)
        return row

    def random_probe(self, seed, length=12, name="sample"):
        random = Random(seed)
        return Probe(f"{name}:{seed}", tuple(self.point([random.uniform(0, 1) if self.family == "order" and i == 0
                                                        else random.uniform(-1, 1) for i in range(4)])
                                             for _ in range(length)))

    def acquire(self, probe):
        observations, responses, trajectories = [], [], []
        for commands in probe.sequence:
            features = self.point([commands[name] for name in self.feature_names],
                                  gain=commands.get("gain", 1.))
            state = [features[name] for name in self.feature_names]
            path = [state]
            if self.family == "order":
                x = features["x0"]
                value = 1.2 * x - .35 * x * x
                if self.variant == "broken" and x >= .85:
                    value += -.8 + 2. * (x - .85) ** 2
                state = [value]
                path.append(state)
            else:
                for _ in range(self.horizon):
                    state = [features.get("gain", 1.) * sum(value * x for value, x in zip(row, state))
                             for row in self._response]
                    path.append(state)
            observations.append(Observation(features, state[self.target_index]))
            responses.append(path[1])
            trajectories.append(path)
        costs = {"observation_cost": probe.cost,
                 "measurement_cost": probe.cost * self.measurement_cost_per_row}
        self.acquisitions.append({"probe_id": probe.probe_id, **costs})
        return MechanismBatch(observations, responses, trajectories, **costs)

    def observe(self, probe):
        return self.acquire(probe).observations

    def evaluation_rows(self, *, seed=815, length=32):
        return self.acquire(self.random_probe(seed, length, "evaluation")).observations
