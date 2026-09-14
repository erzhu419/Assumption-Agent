"""An apparatus whose channels disclose values without object or role labels."""

from random import Random

from .measurement_lab import MeasurementChannel
from .models import Observation, Probe, TaskSpec
from .unified_object_lab import ObjectMeasurementBatch, UnifiedObjectLab


def _anonymous_names(names, prefix, seed):
    rng = Random(seed)
    names = list(names)
    rng.shuffle(names)
    labels = [f"{prefix}{value}" for value in rng.sample(range(100000, 1000000), len(names))]
    mapping = dict(zip(names, labels))
    rng.shuffle(labels)
    return mapping, tuple(labels)


def _rewire(mapping, groups, seed):
    rng = Random(seed)
    result = dict(mapping)
    for group in groups:
        labels = [mapping[name] for name in group]
        rng.shuffle(labels)
        result.update(zip(group, labels))
    return result


class AnonymousObjectLab:
    """Acquire anonymous sensor banks and intervene on anonymous controls.

    The public interface contains no object partition or role assignments.
    Measurement transitions and responses are the evidence for inferring them.
    """

    measurement_views = ((), ("coarse",), ("fine",))

    def __init__(self, *, seed=71, objects=4, readout="focus", variant="exchange",
                 profile="development", apparatus_seed=None, actuator_curvature=None,
                 name_seed=1729, wiring_seed=0, relation_radius=None, relation_direction="symmetric",
                 relation_condition="always"):
        self.seed = seed
        self._settings = dict(seed=seed, objects=objects, readout=readout, variant=variant,
            profile=profile, apparatus_seed=apparatus_seed, actuator_curvature=actuator_curvature,
            name_seed=name_seed, wiring_seed=wiring_seed, relation_radius=relation_radius,
            relation_direction=relation_direction, relation_condition=relation_condition)
        self._lab = UnifiedObjectLab(seed=seed, objects=objects, readout=readout, variant=variant,
            profile=profile, apparatus_seed=apparatus_seed, actuator_curvature=actuator_curvature,
            relation_radius=relation_radius, relation_direction=relation_direction,
            relation_condition=relation_condition)
        bounds = self._lab.spec.feature_bounds
        names, self._control_order = _anonymous_names(bounds, "c", name_seed + 1109)
        groups = {}
        for name, interval in bounds.items():
            groups.setdefault(tuple(interval), []).append(name)
        self._control_names = _rewire(names, groups.values(), wiring_seed + 81031)
        self._physical_controls = {name: physical for physical, name in self._control_names.items()}
        self._bounds = {name: bounds[self._physical_controls[name]] for name in self._control_order}
        self._measurement_names, self._measurement_order = {}, {}
        for index, view in enumerate((("coarse",), ("fine",))):
            fields = self._lab.spec_for(view).feature_names
            names, order = _anonymous_names(fields, f"m{index}_", name_seed + 3001 + 10007 * index)
            self._measurement_names[view] = _rewire(names, [list(fields)], wiring_seed + 90031 + 30011 * index)
            self._measurement_order[view] = order
        self.channels = tuple(MeasurementChannel(channel.name, channel.cost_per_row,
            f"Read auxiliary sensor bank {index + 1}.") for index, channel in enumerate(self._lab.channels))
        self.acquisitions = []
        self.spec = self.spec_for()

    def _selected(self, channels):
        return self._lab._selected(channels)

    def spec_for(self, channels=()):
        selected = self._selected(channels)
        return TaskSpec("anonymous_process", "Predict the response of a controlled process.",
            self._measurement_order[selected] if selected else self._control_order, (), (),
            feature_bounds=dict(self._bounds), max_probe_length=self._lab.spec.max_probe_length)

    @property
    def observation_cost(self):
        return self._lab.observation_cost

    @property
    def measurement_cost(self):
        return self._lab.measurement_cost

    @property
    def costs(self):
        return self._lab.costs

    def acquire(self, probe, channels=()):
        selected = self._selected(channels)
        physical = Probe(probe.probe_id, tuple({self._physical_controls[name]: value
            for name, value in point.items()} for point in probe.sequence))
        batch = self._lab.acquire(physical, selected)
        mapping = self._measurement_names[selected] if selected else self._control_names
        order = self._measurement_order[selected] if selected else self._control_order

        def rename(values):
            mapped = {mapping[name]: value for name, value in values.items()}
            return {name: mapped[name] for name in order}

        history, observations = [], []
        for row in batch.observations:
            features = rename(row.features)
            observations.append(Observation(features, row.target, tuple(history)))
            history.append({**features, "__target__": row.target})
        transitions = [{side: rename(values) for side, values in transition.items()}
                       for transition in batch.transitions]
        self.acquisitions.append({"probe_id": probe.probe_id, "sequence": list(probe.sequence),
            "channels": list(selected), "observation_cost": batch.observation_cost,
            "measurement_cost": batch.measurement_cost})
        return ObjectMeasurementBatch(observations, batch.observation_cost, batch.measurement_cost,
                                      selected, transitions)

    def observe(self, probe, channels=()):
        return self.acquire(probe, channels).observations

    def random_probe(self, seed, *, length=24, name="sample"):
        rng = Random(seed)
        return Probe(name, tuple({field: rng.uniform(*bounds) for field, bounds in self._bounds.items()}
                                for _ in range(length)))

    def projectable_views(self, channels):
        return (self._selected(channels),)

    def project_rows(self, rows, source_channels, target_channels):
        if self._selected(source_channels) != self._selected(target_channels):
            raise ValueError("anonymous banks have no supplied projection between their fields")
        return rows

    def project_transitions(self, transitions, source_channels, target_channels):
        if self._selected(source_channels) != self._selected(target_channels):
            raise ValueError("anonymous banks have no supplied projection between their fields")
        return transitions

    def evaluation_rows(self, channels=(), *, seed=1009, length=64, trajectories=4):
        selected = self._selected(channels)
        with AnonymousObjectLab(**self._settings) as separate:
            return [row for index in range(trajectories)
                for row in separate.observe(separate.random_probe(seed + 10007 * index,
                    length=length, name=f"evaluation_{index}"), selected)]

    def _evaluation_schema(self, channels=("fine",)):
        """Private evaluator truth; research receives only spec and acquisitions."""
        selected = self._selected(channels)
        schema = self._lab._schema(selected)
        return schema.remap_features({**self._control_names, **self._measurement_names[selected]})

    def restore(self, acquisitions):
        self.__init__(**self._settings)
        for item in acquisitions:
            self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item["channels"])

    def reset(self):
        self._lab.reset()

    def close(self):
        self._lab.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
