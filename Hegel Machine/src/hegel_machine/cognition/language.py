"""Executable expressions shared by discovery and ordinary search baselines.

Expressions inspect only the observation passed to them. In particular, a lag
reads recorded history; it never asks the environment for an unobserved value.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Observation


class MissingHistory(ValueError):
    """The branch being executed requires an unavailable past observation."""


@dataclass(frozen=True)
class Definition:
    """A reusable body with lexical arguments and previously learned calls."""

    definition_id: str
    arguments: tuple[str, ...]
    body: "Program"
    source_concept_id: str | None = None

    def __post_init__(self):
        if not self.definition_id:
            raise ValueError("a definition needs a name")
        if not isinstance(self.arguments, tuple) or len(set(self.arguments)) != len(self.arguments):
            raise ValueError("definition arguments must be a tuple of distinct names")

    def __hash__(self):
        # Identifiers make dictionary lookup cheap; dataclass equality still
        # compares the actual body when independently built definitions meet.
        return hash(self.definition_id)

    def to_dict(self):
        body = self.body.to_dict()
        dependencies = body.pop("definitions", {})
        result = {"definition_id": self.definition_id, "arguments": list(self.arguments),
                  "body": body, "source_concept_id": self.source_concept_id}
        if dependencies:
            result["definitions"] = dependencies
        return result

    @classmethod
    def from_dict(cls, payload):
        body = dict(payload["body"])
        if payload.get("definitions"):
            body["definitions"] = payload["definitions"]
        return cls(payload["definition_id"], tuple(payload["arguments"]),
                   Program.from_dict(body), payload.get("source_concept_id"))


class _Argument:
    """A caller expression evaluated at most once, only if its branch uses it."""

    def __init__(self, program, bindings, key):
        self.program, self.bindings, self.key = program, bindings, key
        self.ready = False
        self.value = None
        self.context = None

    def read(self, context):
        if not self.ready or self.context is not context:
            self.value = context._evaluate(self.program, self.bindings)
            self.ready = True
            self.context = context
        return self.value


class EvaluationContext:
    """Share actual definition results for one observation.

    primitive_steps counts evaluated non-call expression nodes, including
    argument reads. call_evaluations counts bodies actually entered; cache_hits
    counts calls served without executing their body. A context can be shared
    by several predictions of the same observation.
    """

    def __init__(self, observation: "Observation"):
        self.observation = observation
        self.primitive_steps = 0
        self.call_evaluations = 0
        self.cache_hits = 0
        self._calls = {}

    def evaluate(self, program: "Program") -> float:
        return self._evaluate(program, {})

    def _key(self, program, bindings, local_names=()):
        if program.op == "arg":
            if program.name in local_names:
                return ("edge_argument", program.name)
            return bindings[program.name].key
        if program.op == "relation_sum":
            from .relation_language import EDGE_ARGUMENTS
            children = tuple(self._key(child, bindings,
                (*local_names, *EDGE_ARGUMENTS) if index == 1 else local_names)
                for index, child in enumerate(program.children))
            return (program.op, program.value, program.name, program.lag, program.definition, children)
        return (program.op, program.value, program.name, program.lag, program.definition,
                tuple(self._key(child, bindings, local_names) for child in program.children))

    def _at_history(self, program, bindings, index):
        from .models import Observation
        observation = self.observation
        past = Observation(observation.history[index], 0.0, tuple(observation.history[:index]))
        local = EvaluationContext(past)
        result = local._evaluate(program, bindings)
        self.primitive_steps += local.primitive_steps
        self.call_evaluations += local.call_evaluations
        self.cache_hits += local.cache_hits
        return result

    def _evaluate(self, program, bindings):
        observation = self.observation
        if program.op == "call":
            arguments = tuple(self._key(child, bindings) for child in program.children)
            key = (program.definition, arguments)
            if key in self._calls:
                self.cache_hits += 1
                return self._calls[key]
            self.call_evaluations += 1
            local = {name: _Argument(child, bindings, argument)
                     for name, child, argument in zip(program.definition.arguments, program.children, arguments)}
            result = self._evaluate(program.definition.body, local)
            self._calls[key] = result
            return result
        self.primitive_steps += 1
        if program.op == "arg":
            if program.name not in bindings:
                raise ValueError(f"unbound definition argument: {program.name}")
            return bindings[program.name].read(self)
        if program.op == "available":
            return float(len(observation.history) >= program.lag)
        if program.op == "const":
            return float(program.value)
        if program.op == "feature":
            return float(observation.features[program.name])
        if program.op == "lag":
            if len(observation.history) < program.lag:
                raise MissingHistory(f"expression needs {program.lag} historical observations")
            return float(observation.history[-program.lag][program.name])
        if program.op == "object_sum":
            return sum(self._evaluate(child, bindings) for child in program.children)
        if program.op == "neighbor_sum":
            radius = self._evaluate(program.children[0], bindings)
            values = [self._evaluate(child, bindings) for child in program.children[1:]]
            objects = [values[index:index + 3] for index in range(0, len(values), 3)]
            if program.name in {"right", "left"}:
                direction = 1. if program.name == "right" else -1.
                return sum((target_selector - source_selector) * quantity
                    for source_selector, position, quantity in objects
                    for target_selector, target_position, _ in objects
                    if 0. < direction * (target_position - position) < radius)
            return sum(selector * (other_value - value)
                       for index, (selector, position, value) in enumerate(objects)
                       for other, (_, other_position, other_value) in enumerate(objects)
                       if index != other and abs(position - other_position) < radius)
        if program.op == "relation_sum":
            from .relation_language import edge_bindings, relation_objects
            radius = self._evaluate(program.children[0], bindings)
            objects = relation_objects(program)
            values = [tuple(self._evaluate(part, bindings) for part in item) for item in objects]
            result = 0.
            for index, (selector, position, quantity) in enumerate(values):
                for other, (other_selector, other_position, other_quantity) in enumerate(values):
                    if index == other:
                        continue
                    if program.name in {"right", "left"}:
                        direction = 1. if program.name == "right" else -1.
                        active = 0. < direction * (other_position - position) < radius
                        contribution = (other_selector - selector) * quantity
                    else:
                        active = abs(position - other_position) < radius
                        contribution = selector * (other_quantity - quantity)
                    if not active:
                        continue
                    local = dict(bindings)
                    for name, expression in edge_bindings(objects[index], objects[other]).items():
                        local[name] = _Argument(expression, bindings, self._key(expression, bindings))
                    if self._evaluate(program.children[1], local):
                        result += contribution
            return result
        if program.op == "window_sum":
            start = max(0, len(observation.history) - program.lag + 1)
            return self._evaluate(program.children[0], bindings) + sum(
                self._at_history(program.children[0], bindings, index)
                for index in range(start, len(observation.history)))
        if program.op == "event_delta":
            if len(observation.history) < program.lag:
                return 0.0
            return self._evaluate(program.children[0], bindings) - self._at_history(
                program.children[0], bindings, len(observation.history) - program.lag)
        if program.op == "state_replay":
            # The learned transition keeps its startup and state conditions.
            # Close its response history using predictions, not measured labels.
            from .models import Observation
            predicted_history = []
            for point in (*observation.history, observation.features):
                features = {name: value for name, value in point.items() if name != "__target__"}
                local = EvaluationContext(Observation(features, 0.0, tuple(predicted_history)))
                value = local._evaluate(program.children[0], bindings)
                self.primitive_steps += local.primitive_steps
                self.call_evaluations += local.call_evaluations
                self.cache_hits += local.cache_hits
                predicted_history.append({**features, "__target__": value})
            return value
        if program.op == "accumulate":
            # Replay only the recorded prefix. Each drive sees that prefix's
            # current features and earlier history, never a later observation.
            from .models import Observation
            drive, initial = program.children
            state = self._evaluate(initial, bindings)
            for index, features in enumerate(observation.history):
                past = Observation(features, 0.0, tuple(observation.history[:index]))
                local = EvaluationContext(past)
                state = program.value * state + local._evaluate(drive, bindings)
                self.primitive_steps += local.primitive_steps
                self.call_evaluations += local.call_evaluations
                self.cache_hits += local.cache_hits
            return program.value * state + self._evaluate(drive, bindings)
        if program.op == "if":
            condition = self._evaluate(program.children[0], bindings)
            return self._evaluate(program.children[1 if condition else 2], bindings)
        left = self._evaluate(program.children[0], bindings)
        if program.op == "abs":
            return abs(left)
        if program.op == "not":
            return float(not left)
        right = self._evaluate(program.children[1], bindings)
        if program.op == "add":
            return left + right
        if program.op == "sub":
            return left - right
        if program.op == "mul":
            return left * right
        if program.op == "min":
            return min(left, right)
        if program.op == "max":
            return max(left, right)
        if program.op == "ge":
            return float(left >= right)
        if program.op == "lt":
            return float(left < right)
        if program.op == "and":
            return float(bool(left) and bool(right))
        return float(bool(left) or bool(right))


@dataclass(frozen=True)
class Program:
    op: str
    value: float | None = None
    name: str | None = None
    lag: int = 0
    children: tuple["Program", ...] = ()
    definition: Definition | None = None

    def __post_init__(self) -> None:
        arities = {
            "const": 0, "feature": 0, "lag": 0, "available": 0, "arg": 0, "abs": 1, "not": 1,
            "add": 2, "sub": 2, "mul": 2, "min": 2, "max": 2,
            "ge": 2, "lt": 2, "and": 2, "or": 2, "if": 3, "accumulate": 2,
            "window_sum": 1, "event_delta": 1, "state_replay": 1,
            "object_sum": len(self.children),
            "neighbor_sum": len(self.children), "relation_sum": len(self.children),
        }
        if self.op == "call":
            if self.definition is None or self.name != self.definition.definition_id:
                raise ValueError("a call must name its executable definition")
            arities["call"] = len(self.definition.arguments)
        if self.op not in arities or len(self.children) != arities[self.op]:
            raise ValueError(f"invalid expression operator or arity: {self.op}")
        if not isinstance(self.children, tuple):
            raise TypeError("program children must be a tuple")
        if self.op == "neighbor_sum" and (len(self.children) < 4 or (len(self.children) - 1) % 3):
            raise ValueError("a neighborhood reduction needs a radius and selector/position/value triples")
        if self.op == "neighbor_sum" and self.name not in {None, "right", "left"}:
            raise ValueError("a neighborhood relation is symmetric, right, or left")
        if self.op == "relation_sum" and (len(self.children) < 5 or (len(self.children) - 2) % 3):
            raise ValueError("a conditional relation needs radius, predicate, and selector/position/value triples")
        if self.op == "relation_sum" and self.name not in {None, "right", "left"}:
            raise ValueError("a conditional relation is symmetric, right, or left")
        if self.op == "const" and (self.value is None or not isfinite(self.value)):
            raise ValueError("a constant must have a finite value")
        if self.op == "accumulate" and (self.value is None or not isfinite(self.value)):
            raise ValueError("an accumulator needs a finite decay")
        if self.op in {"feature", "lag", "arg"} and not self.name:
            raise ValueError("an observed variable needs a name")
        if self.op in {"lag", "available", "window_sum", "event_delta"} and self.lag < 1:
            raise ValueError("lag 1 denotes the most recent historical observation")
        if self.op == "state_replay" and self.children[0].minimum_history_depth:
            raise ValueError("state replay needs a transition that executes from reset")

    @cached_property
    def size(self) -> int:
        # A short call name does not make its semantic computation smaller.
        if self.op == "call":
            return self.expanded().size
        return 1 + sum(child.size for child in self.children)

    @cached_property
    def history_depth(self) -> int:
        if self.op == "call":
            return self.expanded().history_depth
        if self.op in {"accumulate", "state_replay"}:
            # One indicates historical dependence, not a truncated memory:
            # execution folds the entire recorded trajectory at every length.
            return max(1, *(child.history_depth for child in self.children))
        if self.op in {"window_sum", "event_delta"}:
            return self.children[0].history_depth + self.lag - (self.op == "window_sum")
        return max(
            (self.lag if self.op == "lag" else 0),
            *(child.history_depth for child in self.children),
        ) if self.children else (self.lag if self.op == "lag" else 0)

    def required_features(self) -> set[str]:
        """Current features only; historical requirements use history_depth."""
        return set(self._required_features)

    @cached_property
    def _required_features(self):
        if self.op == "call":
            return self.expanded()._required_features
        names = {self.name} if self.op == "feature" else set()
        for child in self.children:
            names.update(child._required_features)
        return frozenset(names)

    @cached_property
    def minimum_history_depth(self) -> int:
        """History needed to execute, accounting for guarded startup branches."""
        expanded = self.expanded()
        if expanded is not self:
            return expanded.minimum_history_depth
        if self.op == "lag":
            return self.lag
        if self.op in {"window_sum", "event_delta"}:
            return 0
        if self.op == "if" and self.children[0].op == "available":
            guard, mature, startup = self.children
            if guard.lag >= mature.minimum_history_depth:
                return min(guard.lag, startup.minimum_history_depth)
            return mature.minimum_history_depth
        return max((child.minimum_history_depth for child in self.children), default=0)

    def observed_variables(self) -> set[tuple[str, int]]:
        """Names and lags read by this expression, including historical targets."""
        return set(self._observed_variables)

    @cached_property
    def _observed_variables(self):
        if self.op == "call":
            return self.expanded()._observed_variables
        variables = {(str(self.name), self.lag if self.op == "lag" else 0)} if self.op in {"feature", "lag"} else set()
        for child in self.children:
            variables.update(child._observed_variables)
        if self.op == "state_replay":
            variables = {(name, lag) for name, lag in variables
                         if name != "__target__" or lag == 0}
        return frozenset(variables)

    def remap_features(self, mapping: dict[str, str]) -> "Program":
        """Bind learned feature roles to current names without changing history."""
        name = self.name
        if self.op in {"feature", "lag"} and name != "__target__":
            name = mapping.get(str(name), name)
        return Program(self.op, self.value, name, self.lag,
                       tuple(child.remap_features(mapping) for child in self.children), self.definition)

    def predict(self, observation: "Observation", context: EvaluationContext | None = None) -> float:
        if context is None:
            context = EvaluationContext(observation)
        elif context.observation is not observation:
            raise ValueError("an evaluation context belongs to one observation")
        return context.evaluate(self)

    def _expand(self, bindings):
        if not self._contains_bindings:
            return self
        if self.op == "arg":
            return bindings.get(self.name, self)
        if self.op == "relation_sum":
            from .relation_language import EDGE_ARGUMENTS, expand_relation
            outside = {name: value for name, value in bindings.items() if name not in EDGE_ARGUMENTS}
            children = tuple(child._expand(outside if index == 1 else bindings)
                             for index, child in enumerate(self.children))
            return expand_relation(Program(self.op, self.value, self.name, self.lag, children))
        children = tuple(child._expand(bindings) for child in self.children)
        if self.op == "call":
            return self.definition.body._expand(dict(zip(self.definition.arguments, children)))
        if all(left is right for left, right in zip(children, self.children)):
            return self
        return Program(self.op, self.value, self.name, self.lag, children, self.definition)

    @cached_property
    def _contains_bindings(self):
        return self.op in {"call", "arg", "relation_sum"} or any(child._contains_bindings for child in self.children)

    @cached_property
    def _expanded_program(self):
        return self._expand({})

    def expanded(self) -> "Program":
        """Return equivalent old-language structure for analysis, not execution."""
        return self._expanded_program

    @cached_property
    def _simplified_program(self):
        from .simplification import simplify
        return simplify(self)

    def simplified(self) -> "Program":
        """Remove exact redundant computation for finite task computations.

        Calls keep their executable definitions and lexical argument behavior.
        Use ``expanded().simplified()`` to compare the bodies of learned calls.
        This does not refit or round learned parameters.
        """
        return self._simplified_program

    def to_dict(self) -> dict[str, Any]:
        definitions, seen = {}, {}

        def encode(program):
            result = {"op": program.op, "value": program.value, "name": program.name,
                      "lag": program.lag, "children": [encode(child) for child in program.children]}
            if program.op == "call":
                definition = program.definition
                key = definition.definition_id
                if key in seen and seen[key] != definition:
                    raise ValueError(f"different definitions use the same name: {key}")
                if key not in seen:
                    seen[key] = definition
                    definitions[key] = {"definition_id": key, "arguments": list(definition.arguments),
                                        "body": encode(definition.body),
                                        "source_concept_id": definition.source_concept_id}
            return result

        result = encode(self)
        if definitions:
            result["definitions"] = definitions
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Program":
        records = value.get("definitions", {})
        definitions = {}

        def resolve(name):
            if name not in definitions:
                record = records[name]
                definitions[name] = Definition(name, tuple(record["arguments"]), decode(record["body"]),
                                               record.get("source_concept_id"))
            return definitions[name]

        def decode(payload):
            definition = resolve(payload["name"]) if payload["op"] == "call" else None
            return cls(op=payload["op"], value=payload.get("value"), name=payload.get("name"),
                       lag=payload.get("lag", 0), children=tuple(decode(child) for child in payload.get("children", [])),
                       definition=definition)

        return decode(value)

    def describe(self) -> str:
        if self.op == "const":
            return format(float(self.value), ".8g")
        if self.op == "feature":
            return str(self.name)
        if self.op == "lag":
            return f"lag({self.name}, {self.lag})"
        if self.op == "available":
            return f"available({self.lag})"
        if self.op == "arg":
            return f"arg({self.name})"
        if self.op == "call":
            return f"{self.name}({', '.join(child.describe() for child in self.children)})"
        if self.op == "accumulate":
            return f"accumulate({self.value:.8g}, {', '.join(child.describe() for child in self.children)})"
        if self.op in {"window_sum", "event_delta"}:
            return f"{self.op}({self.lag}, {self.children[0].describe()})"
        return f"{self.op}({', '.join(child.describe() for child in self.children)})"
