"""A public multi-object exchange apparatus with selectable measurement scale."""

from dataclasses import asdict, dataclass
import json
from random import Random

from .models import Observation, Probe, TaskSpec


@dataclass(frozen=True)
class ObjectSchema:
    """Observable roles and control names; no law or parameter of the apparatus."""

    objects: tuple[str, ...]
    fields: dict[str, dict[str, str]]
    members: dict[str, tuple[str, ...]]
    controls: dict[str, dict[str, str]]
    focus_control: str = "focus"

    @property
    def feature_names(self):
        return tuple(name for item in self.objects for name in self.fields[item].values())

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(tuple(payload["objects"]), payload["fields"],
                   {key: tuple(value) for key, value in payload["members"].items()},
                   payload["controls"], payload.get("focus_control", "focus"))

    def remap_features(self, mapping):
        return ObjectSchema(self.objects,
            {item: {role: mapping.get(name, name) for role, name in fields.items()}
             for item, fields in self.fields.items()}, self.members,
            {item: {role: mapping.get(name, name) for role, name in fields.items()}
             for item, fields in self.controls.items()}, mapping.get(self.focus_control, self.focus_control))


@dataclass
class StructureBatch:
    observations: list[Observation]
    transitions: list[dict]
    schema: ObjectSchema
    observation_cost: int
    measurement_cost: int

    def to_dict(self):
        return {**asdict(self), "schema": self.schema.to_dict()}

    @classmethod
    def from_dict(cls, payload):
        return cls([Observation(row["features"], row["target"], tuple(row.get("history", ())))
                    for row in payload["observations"]], payload["transitions"],
                   ObjectSchema.from_dict(payload["schema"]), payload["observation_cost"], payload["measurement_cost"])


class StructureLab:
    """Set object quantities, positions and impulses, then measure a response.

    A row is one controlled setup and one exchange step. A probe groups setups
    into an observed event sequence. The fixed readout is independent of the
    selected instruments. Coarse instruments report group totals and mean
    positions; they really acquire fewer values and can lose local information.
    """

    def __init__(self, *, objects=4, readout="focus", variant="exchange", seed=71,
                 relation_direction="symmetric", relation_condition="always", total_weight=1.0):
        if objects < 2:
            raise ValueError("the exchange apparatus needs at least two objects")
        if readout not in {"focus", "position_focus", "total", "window", "event", "total_event",
                           "focus_window", "filtered_total", "window_total_event"}:
            raise ValueError("unknown structure readout")
        if variant not in {"exchange", "sink", "anchored", "nonlocal"}:
            raise ValueError("unknown apparatus variant")
        if relation_direction not in {"symmetric", "right", "left"}:
            raise ValueError("unknown relation direction")
        if relation_condition not in {"always", "source_greater", "source_less"}:
            raise ValueError("unknown relation condition")
        self.object_ids = tuple(f"o{index}" for index in range(objects))
        self._readout, self._variant = readout, variant
        self._total_weight = float(total_weight)
        self._relation_direction = relation_direction
        self._relation_condition = relation_condition
        rng = Random(seed)
        self._coupling = rng.uniform(.18, .28)
        self._radius = rng.uniform(.7, 1.15)
        self._actuator_curvature = rng.uniform(.08, .16)
        self._window = 3
        self.observation_cost = self.measurement_cost = 0
        self.acquisitions = []
        self.object_schema = self.schema_for()
        self.spec = self.spec_for()

    def schema_for(self, objects=None, scale=1):
        selected = self.object_ids if objects is None else tuple(objects)
        if scale < 1 or any(item not in self.object_ids for item in selected):
            raise ValueError("measurement objects and group scale must exist")
        groups = [selected[index:index + scale] for index in range(0, len(selected), scale)]
        members = {"_".join(group): group for group in groups}
        fields = {key: {role: f"{role}_{key}" for role in ("q", "x", "u", "f")} for key in members}
        controls = {item: {role: f"{role}_{item}" for role in ("q", "x", "u")} for item in self.object_ids}
        return ObjectSchema(tuple(members), fields, members, controls)

    def spec_for(self, objects=None, scale=1):
        schema = self.schema_for(objects, scale)
        bounds = {name: ((-2., 2.) if role == "x" else (-1., 1.) if role == "q" else (-.25, .25))
                  for fields in schema.controls.values() for role, name in fields.items()}
        bounds[schema.focus_control] = (0., float(len(self.object_ids) - 1))
        return TaskSpec("object_process", "Predict the fixed observed response of a controlled object process.",
                        schema.feature_names, (), (), feature_bounds=bounds, max_probe_length=64,
                        context={"measurement_scale": str(scale), "measurement_objects": ",".join(schema.objects),
                                 "object_schema": json.dumps(schema.to_dict())})

    @property
    def costs(self):
        return {"observation_cost": self.observation_cost, "measurement_cost": self.measurement_cost,
                "queries": len(self.acquisitions)}

    def random_probe(self, seed, *, length=24, name="sample"):
        rng = Random(seed)
        return Probe(name, tuple({key: rng.uniform(*bounds) for key, bounds in self.spec.feature_bounds.items()}
                                 for _ in range(length)))

    def _step(self, point):
        # Quantity controls are actuator commands. Instruments measure the
        # resulting quantities; omitting individual measurements therefore does
        # not secretly leave the same values available as known controls.
        quantity = [float(point[f"q_{item}"]) + self._actuator_curvature * float(point[f"q_{item}"]) ** 3
                    for item in self.object_ids]
        position = [float(point[f"x_{item}"]) for item in self.object_ids]
        drive = [float(point[f"u_{item}"]) for item in self.object_ids]
        focus = round(point["focus"])
        directional = [0.] * len(quantity)
        if self._relation_direction != "symmetric":
            sign = 1. if self._relation_direction == "right" else -1.
            # Each directed edge transports a signed quantity from its source
            # to its target. Focus selects a readout, never the physical edges.
            for source, value in enumerate(quantity):
                for target in range(len(quantity)):
                    if 0. < sign * (position[target] - position[source]) < self._radius:
                        if self._relation_condition == "source_greater" and value <= quantity[target]:
                            continue
                        if self._relation_condition == "source_less" and value >= quantity[target]:
                            continue
                        transfer = self._coupling * value
                        directional[source] -= transfer
                        directional[target] += transfer
        after = []
        for index, value in enumerate(quantity):
            change = (self._coupling * sum(other - value for other_index, other in enumerate(quantity)
                if other_index != index and abs(position[index] - position[other_index]) < self._radius)
                if self._relation_direction == "symmetric" else directional[index])
            if self._variant == "sink":
                change -= .13 * value
            elif self._variant == "anchored":
                change += .17 * position[index] + .07 * (index + 1) * value
            elif self._variant == "nonlocal":
                change += .09 * sum(quantity)
            after.append(value + drive[index] + change)
        return quantity, position, drive, focus, after

    def _view(self, schema, quantity, position, drive, focus):
        indices = {item: index for index, item in enumerate(self.object_ids)}
        features = {}
        for group in schema.objects:
            members = [indices[item] for item in schema.members[group]]
            fields = schema.fields[group]
            features.update({fields["q"]: sum(quantity[index] for index in members),
                             fields["x"]: sum(position[index] for index in members) / len(members),
                             fields["u"]: sum(drive[index] for index in members),
                             fields["f"]: float(focus in members)})
        return features

    def acquire(self, probe, *, objects=None, scale=1):
        if not 1 <= probe.cost <= self.spec.max_probe_length:
            raise ValueError("probe length is outside the apparatus interface")
        schema = self.schema_for(objects, scale)
        history, flows, combined_events, rows, transitions = [], [], [], [], []
        filtered_total = 0.0
        for point in probe.sequence:
            if set(point) != set(self.spec.feature_bounds) or any(
                    not low <= point[name] <= high for name, (low, high) in self.spec.feature_bounds.items()):
                raise ValueError("a setup must supply the public controls within their bounds")
            quantity, position, drive, focus, after = self._step(point)
            before_features = self._view(schema, quantity, position, drive, focus)
            after_features = self._view(schema, after, position, drive, focus)
            flows.append(after[focus] - quantity[focus] - drive[focus])
            event = flows[-1] - flows[-2] if len(flows) > 1 else 0.0
            total = self._total_weight * sum(after)
            if self._readout == "total":
                target = total
            elif self._readout == "position_focus":
                # A local observation can depend on position even when the
                # physical exchange does not identify position membership.
                target = after[focus] + .23 * position[focus]
            elif self._readout == "window":
                target = sum(flows[-self._window:])
            elif self._readout == "event":
                target = event
            elif self._readout == "total_event":
                target = total + .7 * event
            elif self._readout == "focus_window":
                target = after[focus] + .6 * sum(flows[-self._window:])
            elif self._readout == "filtered_total":
                filtered_total = .8 * filtered_total + total
                target = filtered_total
            elif self._readout == "window_total_event":
                combined_events.append(total + .7 * event)
                target = sum(combined_events[-self._window:])
            else:
                target = after[focus]
            rows.append(Observation(before_features, target, tuple(history)))
            transitions.append({"before": before_features, "after": after_features})
            history.append({**before_features, "__target__": target})
        measured = 3 * len(schema.objects) * len(rows)
        self.observation_cost += len(rows)
        self.measurement_cost += measured
        self.acquisitions.append({"probe_id": probe.probe_id, "objects": list(schema.objects), "scale": scale,
                                  "observation_cost": len(rows), "measurement_cost": measured})
        return StructureBatch(rows, transitions, schema, len(rows), measured)

    def observe(self, probe, *, objects=None, scale=1):
        return self.acquire(probe, objects=objects, scale=scale).observations

    def evaluation_rows(self, *, objects=None, scale=1, seed=815, length=48):
        # The measurement protocol is identical. Keep evaluation costs visible
        # on this apparatus so callers can report their separate evaluation bill.
        return self.observe(self.random_probe(seed, length=length, name=f"evaluation_{seed}"),
                            objects=objects, scale=scale)
