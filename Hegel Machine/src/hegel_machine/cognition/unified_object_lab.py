"""Object measurements on the same acquisition interface as scalar tasks."""

from dataclasses import dataclass, field, replace
import json

from .measurement_lab import MeasurementBatch, MeasurementChannel
from .models import Observation, Probe
from .structure_lab import ObjectSchema, StructureBatch, StructureLab


@dataclass
class ObjectMeasurementBatch(MeasurementBatch):
    transitions: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload):
        batch = MeasurementBatch.from_dict(payload)
        return cls(batch.observations, batch.observation_cost, batch.measurement_cost,
                   batch.channels, payload["transitions"])


class UnifiedObjectLab:
    """Offer commands, group instruments, or individual object instruments.

    Instrument names select alternative views. The public object schema supplies
    measurable roles, while the response law is inferred from acquired rows.
    """

    measurement_views = ((), ("coarse",), ("fine",))

    def __init__(self, *, seed=71, objects=4, readout="focus", variant="exchange",
                 profile="development", rename_prefix="", apparatus_seed=None,
                 actuator_curvature=None, relation_radius=None, relation_direction="symmetric",
                 relation_condition="always", total_weight=1.0):
        if profile not in {"development", "confirmation"}:
            raise ValueError("profile must be development or confirmation")
        self.seed = seed
        self._settings = dict(seed=seed, objects=objects, readout=readout,
                              variant=variant, profile=profile, rename_prefix=rename_prefix,
                              apparatus_seed=apparatus_seed, actuator_curvature=actuator_curvature,
                              relation_radius=relation_radius, relation_direction=relation_direction,
                              relation_condition=relation_condition, total_weight=total_weight)
        plant_seed = seed if apparatus_seed is None else apparatus_seed
        self._lab = StructureLab(objects=objects, readout=readout, variant=variant,
                                 seed=plant_seed + (7919 if profile == "confirmation" else 0),
                                 relation_direction=relation_direction, relation_condition=relation_condition,
                                 total_weight=total_weight)
        if actuator_curvature is not None:
            self._lab._actuator_curvature = float(actuator_curvature)
        if relation_radius is not None:
            self._lab._radius = float(relation_radius)
        self._prefix = "measured_" + rename_prefix
        self.channels = (
            MeasurementChannel("coarse", 3,
                "Measure total quantity and impulse, mean position, and focus membership for one object group."),
            MeasurementChannel("fine", 3 * objects,
                "Measure quantity, position, impulse, and focus membership for each object separately."),
        )
        self.spec = self.spec_for()
        self.observation_cost = self.measurement_cost = 0
        self.acquisitions = []

    def _selected(self, channels):
        selected = tuple(dict.fromkeys(channels))
        if selected not in self.measurement_views:
            raise ValueError("choose the command view, coarse instruments, or fine instruments")
        return selected

    def _schema(self, channels):
        scale = len(self._lab.object_ids) if channels == ("coarse",) else 1
        schema = self._lab.schema_for(scale=scale)
        # Commands and actual instrument readings have separate identities when
        # instruments are renamed, even where their original labels coincide.
        return ObjectSchema(schema.objects,
            {item: {role: self._prefix + name for role, name in fields.items()}
             for item, fields in schema.fields.items()},
            schema.members, schema.controls, schema.focus_control)

    def spec_for(self, channels=()):
        selected = self._selected(channels)
        base = self._lab.spec
        context = {"measurement_view": selected[0] if selected else "commands"}
        if selected:
            schema = self._schema(selected)
            context["object_schema"] = json.dumps(schema.to_dict())
            context["measurement_scale"] = str(len(self._lab.object_ids) if selected == ("coarse",) else 1)
            features = schema.feature_names
        else:
            features = tuple(base.feature_bounds)
        return replace(base, feature_names=features, context=context)

    @property
    def costs(self):
        return {"observation_cost": self.observation_cost,
                "measurement_cost": self.measurement_cost, "queries": len(self.acquisitions)}

    def acquire(self, probe, channels=()):
        selected = self._selected(channels)
        scale = len(self._lab.object_ids) if selected == ("coarse",) else 1
        # The apparatus can acquire the response without auxiliary instruments.
        batch = self._lab.acquire(probe, objects=None if selected else (), scale=scale)
        history, observations = [], []
        for point, row in zip(probe.sequence, batch.observations):
            features = ({self._prefix + name: value for name, value in row.features.items()}
                        if selected else dict(point))
            observations.append(Observation(features, row.target, tuple(history)))
            history.append({**features, "__target__": row.target})
        self.observation_cost += batch.observation_cost
        self.measurement_cost += batch.measurement_cost
        self.acquisitions.append({"probe_id": probe.probe_id, "sequence": list(probe.sequence),
                                  "channels": list(selected),
                                  "observation_cost": batch.observation_cost,
                                  "measurement_cost": batch.measurement_cost})
        transitions = [{side: {self._prefix + name: value for name, value in states.items()}
                        for side, states in transition.items()} for transition in batch.transitions] if selected else []
        return ObjectMeasurementBatch(observations, batch.observation_cost,
                                      batch.measurement_cost, selected, transitions)

    def observe(self, probe, channels=()):
        return self.acquire(probe, channels).observations

    def projectable_views(self, channels):
        selected = self._selected(channels)
        return (("coarse",), ("fine",)) if selected == ("fine",) else (selected,)

    def project_rows(self, rows, source_channels, target_channels):
        """Aggregate already purchased readings; acquisition costs stay fixed."""
        source, target = self._selected(source_channels), self._selected(target_channels)
        if source == target:
            return rows
        if source != ("fine",) or target != ("coarse",):
            raise ValueError("only fine measurements can be projected into coarse measurements")
        from .structural_discovery import project_batch
        batch = StructureBatch(rows, [], self._schema(source), 0, 0)
        return project_batch(batch, self._schema(target)).observations

    def project_transitions(self, transitions, source_channels, target_channels):
        source, target = self._selected(source_channels), self._selected(target_channels)
        if source == target:
            return transitions
        if source != ("fine",) or target != ("coarse",):
            raise ValueError("only fine measurements can be projected into coarse measurements")
        from .structural_discovery import project_batch
        batch = StructureBatch([], transitions, self._schema(source), 0, 0)
        return project_batch(batch, self._schema(target)).transitions

    def random_probe(self, seed, *, length=24, name="sample"):
        return self._lab.random_probe(seed, length=length, name=name)

    def evaluation_rows(self, channels=(), *, seed=1009, length=64, trajectories=4):
        """Use new controls on the same law without modifying learning costs."""
        selected = self._selected(channels)
        with UnifiedObjectLab(**self._settings) as separate:
            return [row for index in range(trajectories)
                    for row in separate.observe(separate.random_probe(seed + 10007 * index,
                        length=length, name=f"evaluation_{index}"), selected)]

    def restore(self, acquisitions):
        self.__init__(**self._settings)
        for item in acquisitions:
            self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item["channels"])

    def reset(self):
        """Every probe already starts a new event sequence."""

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
