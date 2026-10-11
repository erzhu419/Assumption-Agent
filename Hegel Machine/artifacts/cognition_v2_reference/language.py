"""Executable expressions shared by discovery and ordinary search baselines.

Expressions inspect only the observation passed to them. In particular, a lag
reads recorded history; it never asks the environment for an unobserved value.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Observation


@dataclass(frozen=True)
class Program:
    op: str
    value: float | None = None
    name: str | None = None
    lag: int = 0
    children: tuple["Program", ...] = ()

    def __post_init__(self) -> None:
        arities = {
            "const": 0, "feature": 0, "lag": 0, "abs": 1, "not": 1,
            "add": 2, "sub": 2, "mul": 2, "min": 2, "max": 2,
            "ge": 2, "lt": 2, "and": 2, "or": 2, "if": 3,
        }
        if self.op not in arities or len(self.children) != arities[self.op]:
            raise ValueError(f"invalid expression operator or arity: {self.op}")
        if not isinstance(self.children, tuple):
            raise TypeError("program children must be a tuple")
        if self.op == "const" and (self.value is None or not isfinite(self.value)):
            raise ValueError("a constant must have a finite value")
        if self.op in {"feature", "lag"} and not self.name:
            raise ValueError("an observed variable needs a name")
        if self.op == "lag" and self.lag < 1:
            raise ValueError("lag 1 denotes the most recent historical observation")

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)

    @property
    def history_depth(self) -> int:
        return max(
            (self.lag if self.op == "lag" else 0),
            *(child.history_depth for child in self.children),
        ) if self.children else (self.lag if self.op == "lag" else 0)

    def required_features(self) -> set[str]:
        """Current features only; historical requirements use history_depth."""
        names = {self.name} if self.op == "feature" else set()
        for child in self.children:
            names.update(child.required_features())
        return names

    def observed_variables(self) -> set[tuple[str, int]]:
        """Names and lags read by this expression, including historical targets."""
        variables = {(str(self.name), self.lag if self.op == "lag" else 0)} if self.op in {"feature", "lag"} else set()
        for child in self.children:
            variables.update(child.observed_variables())
        return variables

    def remap_features(self, mapping: dict[str, str]) -> "Program":
        """Bind learned feature roles to current names without changing history."""
        name = self.name
        if self.op in {"feature", "lag"} and name != "__target__":
            name = mapping.get(str(name), name)
        return Program(self.op, self.value, name, self.lag,
                       tuple(child.remap_features(mapping) for child in self.children))

    def predict(self, observation: "Observation") -> float:
        if self.op == "const":
            return float(self.value)
        if self.op == "feature":
            return float(observation.features[self.name])
        if self.op == "lag":
            if len(observation.history) < self.lag:
                raise ValueError(f"expression needs {self.lag} historical observations")
            return float(observation.history[-self.lag][self.name])
        if self.op == "if":
            condition = self.children[0].predict(observation)
            return self.children[1 if condition else 2].predict(observation)
        left = self.children[0].predict(observation)
        if self.op == "abs":
            return abs(left)
        if self.op == "not":
            return float(not left)
        right = self.children[1].predict(observation)
        if self.op == "add":
            return left + right
        if self.op == "sub":
            return left - right
        if self.op == "mul":
            return left * right
        if self.op == "min":
            return min(left, right)
        if self.op == "max":
            return max(left, right)
        if self.op == "ge":
            return float(left >= right)
        if self.op == "lt":
            return float(left < right)
        if self.op == "and":
            return float(bool(left) and bool(right))
        return float(bool(left) or bool(right))

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op, "value": self.value, "name": self.name,
            "lag": self.lag, "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Program":
        return cls(
            op=value["op"], value=value.get("value"), name=value.get("name"),
            lag=value.get("lag", 0),
            children=tuple(cls.from_dict(child) for child in value.get("children", [])),
        )

    def describe(self) -> str:
        if self.op == "const":
            return format(float(self.value), ".8g")
        if self.op == "feature":
            return str(self.name)
        if self.op == "lag":
            return f"lag({self.name}, {self.lag})"
        return f"{self.op}({', '.join(child.describe() for child in self.children)})"
