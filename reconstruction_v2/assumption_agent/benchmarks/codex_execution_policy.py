from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..models import stable_hash


CATALOG_DEFAULT = "catalog_default"
LEGACY_CODEX_AGENT_EXECUTION_POLICY_VERSION = (
    "codex_catalog_defaults_remote_compaction_v1"
)
LOW_REASONING_LOCAL_COMPACTION_POLICY_VERSION = (
    "codex_low_reasoning_early_local_compaction_v1"
)


@dataclass(frozen=True)
class CodexAgentExecutionPolicy:
    """Frozen Codex transport/context treatment for one paper protocol."""

    version: str
    model_reasoning_effort: str | None
    model_verbosity: str | None
    model_auto_compact_token_limit: int | None
    model_auto_compact_token_limit_scope: str | None
    tool_output_token_limit: int | None
    enable_request_compression: bool
    remote_compaction_v2: bool

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("Codex agent execution policy version is missing")
        if self.model_reasoning_effort not in {
            None,
            "low",
            "medium",
            "high",
            "xhigh",
        }:
            raise ValueError("invalid Codex model reasoning effort")
        if self.model_verbosity not in {None, "low", "medium", "high"}:
            raise ValueError("invalid Codex model verbosity")
        if (
            self.model_auto_compact_token_limit is not None
            and self.model_auto_compact_token_limit <= 0
        ):
            raise ValueError("invalid Codex auto-compaction token limit")
        if self.model_auto_compact_token_limit_scope not in {
            None,
            "total",
            "body_after_prefix",
        }:
            raise ValueError("invalid Codex auto-compaction token-limit scope")
        if (self.model_auto_compact_token_limit is None) != (
            self.model_auto_compact_token_limit_scope is None
        ):
            raise ValueError(
                "Codex auto-compaction limit and scope must be frozen together"
            )
        if (
            self.tool_output_token_limit is not None
            and self.tool_output_token_limit <= 0
        ):
            raise ValueError("invalid Codex tool-output token limit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_reasoning_effort": self.model_reasoning_effort or CATALOG_DEFAULT,
            "model_verbosity": self.model_verbosity or CATALOG_DEFAULT,
            "model_auto_compact_token_limit": (
                self.model_auto_compact_token_limit
                if self.model_auto_compact_token_limit is not None
                else CATALOG_DEFAULT
            ),
            "model_auto_compact_token_limit_scope": (
                self.model_auto_compact_token_limit_scope or CATALOG_DEFAULT
            ),
            "tool_output_token_limit": (
                self.tool_output_token_limit
                if self.tool_output_token_limit is not None
                else CATALOG_DEFAULT
            ),
            "enable_request_compression": self.enable_request_compression,
            "remote_compaction_v2": self.remote_compaction_v2,
        }

    @property
    def policy_hash(self) -> str:
        return stable_hash(self.to_dict())

    def codex_cli_values(self) -> tuple[str, ...]:
        """Return argv fragments inserted after ``codex exec``."""

        values: list[str] = []
        for key, value in (
            ("model_reasoning_effort", self.model_reasoning_effort),
            ("model_verbosity", self.model_verbosity),
            (
                "model_auto_compact_token_limit",
                self.model_auto_compact_token_limit,
            ),
            (
                "model_auto_compact_token_limit_scope",
                self.model_auto_compact_token_limit_scope,
            ),
            ("tool_output_token_limit", self.tool_output_token_limit),
        ):
            if value is None:
                continue
            rendered = f'"{value}"' if isinstance(value, str) else str(value)
            values.extend(("--config", f"{key}={rendered}"))
        values.extend(
            (
                "--enable" if self.enable_request_compression else "--disable",
                "enable_request_compression",
                "--enable" if self.remote_compaction_v2 else "--disable",
                "remote_compaction_v2",
            )
        )
        return tuple(values)


LEGACY_CODEX_AGENT_EXECUTION_POLICY = CodexAgentExecutionPolicy(
    version=LEGACY_CODEX_AGENT_EXECUTION_POLICY_VERSION,
    model_reasoning_effort=None,
    model_verbosity=None,
    model_auto_compact_token_limit=None,
    model_auto_compact_token_limit_scope=None,
    tool_output_token_limit=None,
    enable_request_compression=True,
    remote_compaction_v2=True,
)

LOW_REASONING_LOCAL_COMPACTION_POLICY = CodexAgentExecutionPolicy(
    version=LOW_REASONING_LOCAL_COMPACTION_POLICY_VERSION,
    model_reasoning_effort="low",
    model_verbosity="low",
    model_auto_compact_token_limit=32_768,
    model_auto_compact_token_limit_scope="body_after_prefix",
    tool_output_token_limit=10_000,
    enable_request_compression=True,
    remote_compaction_v2=False,
)

CODEX_AGENT_EXECUTION_POLICY_BY_PROTOCOL_VERSION = {
    "3.1.0": LEGACY_CODEX_AGENT_EXECUTION_POLICY,
    "3.2.0": LEGACY_CODEX_AGENT_EXECUTION_POLICY,
    "3.3.0": LOW_REASONING_LOCAL_COMPACTION_POLICY,
}


def codex_agent_execution_policy_for_protocol_version(
    protocol_version: object,
) -> CodexAgentExecutionPolicy | None:
    return CODEX_AGENT_EXECUTION_POLICY_BY_PROTOCOL_VERSION.get(
        str(protocol_version or "")
    )


def declared_policy_matches(
    policy: CodexAgentExecutionPolicy,
    declared: object,
) -> bool:
    return isinstance(declared, Mapping) and dict(declared) == policy.to_dict()
