from __future__ import annotations

import concurrent.futures
import importlib.util
import copy
import hashlib
import json
import math
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence, TypeVar
from urllib.parse import urlsplit, urlunsplit

from ..archive import PolicyArchive
from ..evaluation import PromotionGate
from ..events import Event, EventSink, NullEventSink
from ..evolution import (
    CANDIDATE_BUNDLE_POLICY_VERSION,
    COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
    COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION,
    CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
    CounterfactualEvidenceReplayCache,
    EvolutionKernel,
    EvolutionRunResult,
    PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
    TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
)
from ..models import (
    CounterfactualPair,
    ExternalOutcome,
    HypothesisKind,
    HypothesisProgram,
    HypothesisStatus,
    LaneResult,
    ResidualExample,
    RuntimeExecution,
    SplitName,
    TaskInput,
    stable_hash,
)
from ..proposer import (
    HypothesisProposalCallError,
    REPAIR_REQUEST_SCOPE_POLICY_VERSION,
    StructuredHypothesisProposer,
    TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY,
    TRAIN_ACTION_DESIGN_POLICY_VERSION,
    TRAIN_ACTION_DESIGN_POLICY_VERSIONS,
)
from ..secure_env import configured_skilllearn_provider_mode
from ..splits import AccessPhase, BenchmarkItem, SplitAccessGuard, SplitManifest
from ..validation import (
    RecursiveValidationEngine,
    ValidationContext,
    build_runtime_feature_catalog,
)
from .skilllearn_compiler import (
    NO_SKILL_TREATMENT_HASH,
    SKILL_ACTION_LOWERING_VERSION,
    SKILL_FALLBACK_SEMANTICS_VERSION,
    SKILL_ROUTING_VERSION,
    SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
    SkillLearnProgramCompiler,
    skilllearn_program_set_treatment_hash,
    skilllearn_program_treatment_hash,
)
from .skilllearnbench import (
    TRAIN_ACTION_ENVIRONMENT_PROFILE_VERSION,
    SkillLearnBenchAdapter,
)
from .codex_action_budget import (
    CODEX_ACTION_BUDGET_COST_ACCOUNTING_POLICY,
    CODEX_ACTION_BUDGET_POLICY_VERSION,
    CODEX_ACTION_BUDGET_UNIT,
    audit_codex_action_budget,
)
from .codex_execution_policy import (
    CodexAgentExecutionPolicy,
    LEGACY_CODEX_AGENT_EXECUTION_POLICY,
)
from .docker_egress import (
    DEFAULT_TRIAL_NETWORK_BYTE_LIMIT,
    DEPENDENCY_CACHE_POLICY_VERSION,
    DOCKER_EGRESS_POLICY_VERSION,
    PROVIDER_DNS_POLICY_VERSION,
    TRIAL_NETWORK_BUDGET_POLICY_VERSION,
    DockerEgressPolicy,
    configured_trial_network_byte_limit,
)
from .offline_verifier import (
    OFFLINE_VERIFIER_MOUNT,
    OFFLINE_VERIFIER_POLICY_VERSION,
    OfflineVerifierProfile,
    OfflineVerifierRuntime,
    SkillLearnOfflineVerifierRuntimeCache,
    inspect_semantic_prelude_receipt,
    offline_verifier_activation_blocker_for_family,
    offline_verifier_catalog_profile_for_family,
    offline_verifier_profile_for_family,
    test_script_requires_offline_profile,
)


BASELINE_LANE = "skilllearn_incumbent"
CANDIDATE_LANE = "skilllearn_challenger"
VERIFIER_ISOLATION_VERSION = "post_agent_verifier_copy_v1"
VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION = (
    "pytest_ctrf_reward_and_semantic_prelude_receipt_v2"
)
RUNNER_AGENT_REGISTRY_ISOLATION_VERSION = "runner_local_agent_registry_v1"
TRIAL_TIMEOUT_POLICY_VERSION = "no_fixed_trial_wall_or_container_timeout_v2"
PROVIDER_FAILURE_POLICY_VERSION = "codex_jsonl_terminal_error_circuit_v1"
MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION = (
    "process_shared_agent_stage_semaphore_v1"
)
TRAINING_EVIDENCE_POLICY_VERSION = "all_valid_before_proposal_v1"
CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION = (
    "valid_train_failures_and_success_controls_v1"
)
ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION = (
    "valid_train_failures_actionable_feedback_and_success_controls_v2"
)
TRAIN_ACTION_TRACE_PROFILE_VERSION = (
    "train_policy_off_allowlisted_tool_facts_v2"
)
CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSIONS = frozenset(
    {
        CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
        ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
    }
)
TRAINING_EVIDENCE_REPLAY_POLICY_VERSION = (
    "behavior_identical_training_replay_v1"
)
LEGACY_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION = (
    "behavior_identical_validation_baseline_arm_replay_v1"
)
SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION = (
    "behavior_identical_shared_validation_baseline_arm_replay_v2"
)
TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION = (
    "behavior_identical_shared_validation_baseline_terminal_outcome_replay_v3"
)
BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION = (
    LEGACY_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
)
SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS = frozenset(
    {
        SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
        TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
    }
)
TERMINAL_INVALID_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS = frozenset(
    {TERMINAL_SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION}
)
BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS = frozenset(
    {
        LEGACY_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
        *SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS,
    }
)
INVALID_TRIAL_RETRY_POLICY_VERSION = (
    "same_request_transient_invalid_clean_replacement_v2"
)
LOCAL_EVIDENCE_TRANSPORT_VERSION = (
    "local_content_addressed_task_and_post_agent_verifier_v1"
)
NETWORK_SCOPE_AUDIT_VERSION = "provider_only_hard_egress_and_local_evidence_v2"
PROPOSAL_FAILURE_ISOLATION_POLICY_VERSION = (
    "candidate_local_repair_and_generation_terminal_root_failure_v1"
)
PROVIDER_ROUTE_POLICY_VERSION = "single_model_single_provider_all_arms_v1"
OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION = "codex_custom_responses_provider_v1"
LEGACY_CODEX_NETWORK_MINIMIZATION_VERSION = "model_only_no_remote_tools_v2"
CODEX_NETWORK_MINIMIZATION_VERSION = "model_only_no_remote_tools_v3"
MODEL_ONLY_TOOL_POLICY_VERSION = "no_web_image_or_runtime_install_v1"
OPENAI_COMPATIBLE_CODEX_PROVIDER_ID = "assumption_v2_openai_compatible"
PREBUILT_IMAGE_POLICY_VERSION = "per_item_base_shared_agent_runtime_v3"
SHARED_AGENT_RUNTIME_MOUNT = "/opt/assumption-v2-agent"
SHARED_AGENT_RUNTIME_BUILDER_IMAGE = (
    "node@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0"
)
SHARED_CODEX_CLI_PACKAGE = "@openai/codex@0.144.1"
SHARED_CODEX_CLI_VERSION = "codex-cli 0.144.1"
CODEX_ACTION_SUPERVISOR_FILENAME = "codex-action-supervisor.mjs"
CODEX_ACTION_SUPERVISOR_PATH = Path(__file__).with_name(
    "codex_action_supervisor.mjs"
)
_InputT = TypeVar("_InputT")
_OutputT = TypeVar("_OutputT")
_FATAL_PROVIDER_ERROR_TYPES = frozenset(
    {
        "provider_rate_limit",
        "provider_authentication_failed",
        "provider_model_unavailable",
        "provider_usage_limit",
    }
)
_FORBIDDEN_CODEX_TOOL_TYPES = frozenset(
    {
        "web_search",
        "web_search_call",
        "image_generation",
        "image_generation_call",
    }
)
_ACTION_TRACE_ALLOWED_EXECUTABLES = frozenset(
    {
        "awk",
        "cat",
        "convert",
        "cp",
        "csvkit",
        "cut",
        "file",
        "find",
        "ffmpeg",
        "grep",
        "gs",
        "jq",
        "libreoffice",
        "ls",
        "mkdir",
        "mv",
        "node",
        "pandoc",
        "pdfinfo",
        "pdftoppm",
        "pdftotext",
        "python",
        "python3",
        "qpdf",
        "ruby",
        "sed",
        "sort",
        "sqlite3",
        "tar",
        "trivy",
        "unzip",
        "wc",
        "xlsx2csv",
        "xmlstarlet",
        "zip",
    }
)
_ACTION_TRACE_ALLOWED_FLAGS = frozenset(
    {
        "--cache-dir",
        "--csv",
        "--format",
        "--ignore-unfixed",
        "--input",
        "--json",
        "--no-progress",
        "--offline-scan",
        "--output",
        "--quiet",
        "--scanners",
        "--severity",
        "--skip-db-update",
        "--text",
        "--version",
        "-f",
        "-m",
        "-o",
        "-q",
    }
)
_ACTION_TRACE_SHELL_WRAPPERS = frozenset({"bash", "dash", "sh", "zsh"})
_ACTION_TRACE_COMMAND_WRAPPERS = frozenset({"env", "sudo", "timeout"})
_ACTION_TRACE_FORBIDDEN_REFERENCE_PATTERN = re.compile(
    r"(?:^|[\s/_.-])(?:tests?|verifier|oracle|solutions?)(?:$|[\s/_.-])",
    re.IGNORECASE,
)
_ACTION_TRACE_SENSITIVE_PATH_COMPONENT_PATTERN = re.compile(
    r"(?:^|[/_.-])(?:api[-_]?keys?|access[-_]?tokens?|auth|credentials?|"
    r"env|id[-_]?rsa|netrc|npmrc|passwords?|passwds?|private|pypirc|"
    r"secrets?|sk-[A-Za-z0-9_-]{8,}|tokens?|keys?)"
    r"(?:$|[/_.-])",
    re.IGNORECASE,
)
_ACTION_TRACE_SENSITIVE_PATTERN = re.compile(
    r"(?ix)(?:"
    r"\b(?:https?|ftp)://|"
    r"\bwww\.|"
    r"(?:^|\s)(?:-H|--header|--api[-_]?key|--access[-_]?token|--auth|"
    r"--password|--passwd|--secret|--token)(?:[=\s]|$)|"
    r"\bBearer\s+|"
    r"\bBasic\s+[A-Za-z0-9+/=]+|"
    r"\b[A-Z0-9_]*(?:API[-_]?KEY|ACCESS[-_]?TOKEN|AUTH[-_]?TOKEN|"
    r"PASSWORD|PASSWD|SECRET|TOKEN)\s*=|"
    r"\bsk-[A-Za-z0-9_-]{8,}\b|"
    r"(?:\?|&)[A-Za-z0-9_.~-]+=[^\s&]+"
    r")"
)
_ACTION_TRACE_ROOT_PATH_PATTERN = re.compile(
    r"/root(?:/[A-Za-z0-9._+-]+)+"
)
_FORBIDDEN_RUNTIME_COMMAND_PATTERNS = (
    re.compile(
        r"(?:^|[;&|]\s*|-(?:l)?c\s+['\"]?)(?:sudo\s+)?(?:pip|pip3)\s+install\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[;&|]\s*|-(?:l)?c\s+['\"]?)python(?:3(?:\.\d+)?)?\s+-m\s+pip\s+install\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[;&|]\s*|-(?:l)?c\s+['\"]?)uv\s+pip\s+install\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[;&|]\s*|-(?:l)?c\s+['\"]?)uvx(?:\s|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[;&|]\s*|-(?:l)?c\s+['\"]?)(?:sudo\s+)?apt(?:-get)?\s+(?:update|install)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[;&|]\s*|-(?:l)?c\s+['\"]?)(?:npm|pnpm|yarn)\s+(?:install|add)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[;&|]\s*|-(?:l)?c\s+['\"]?)npx(?:\s|$)",
        re.IGNORECASE,
    ),
)


class SkillLearnAgentTerminalError(RuntimeError):
    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


class SkillLearnTrainingEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class _VerifierExecutionReceipt:
    valid: bool
    error_type: str | None
    evidence_kind: str
    reward: int | None
    test_count: int
    semantic_prelude_required: bool
    semantic_prelude_valid: bool
    semantic_prelude_succeeded: bool
    semantic_prelude_id: str | None
    semantic_prelude_exit_code: int | None
    semantic_prelude_details: Mapping[str, str]
    semantic_prelude_receipt_hash: str
    receipt_hash: str


@dataclass(frozen=True)
class _CodexToolPolicyAudit:
    valid: bool
    error_type: str | None
    remote_tool_call_count: int
    runtime_install_command_count: int
    trace_hash: str


class SkillLearnProviderCircuit:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._error_type: str | None = None

    @property
    def error_type(self) -> str | None:
        with self._lock:
            return self._error_type

    def open(self, error_type: str) -> bool:
        if error_type not in _FATAL_PROVIDER_ERROR_TYPES:
            return False
        with self._lock:
            if self._error_type is not None:
                return False
            self._error_type = error_type
            return True


class SkillLearnModelInferenceLimiter:
    """Share a bounded online-agent slot pool across item workers."""

    policy = MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION

    def __init__(self, slots: int) -> None:
        if isinstance(slots, bool) or not isinstance(slots, int) or slots <= 0:
            raise ValueError("model inference slots must be positive")
        self.slots = slots
        self._semaphore = threading.BoundedSemaphore(slots)
        self._lock = threading.Lock()
        self._active = 0
        self._maximum_active = 0

    @property
    def maximum_active(self) -> int:
        with self._lock:
            return self._maximum_active

    @contextmanager
    def acquire(
        self,
        *,
        event_sink: EventSink,
        trace_id: str,
    ) -> Iterator[None]:
        queued_at = time.monotonic()
        event_sink.emit(
            Event(
                event="skilllearn_agent_slot_wait_started",
                stage="benchmark.skilllearn.model_concurrency",
                trace_id=trace_id,
                payload={
                    "policy": self.policy,
                    "slot_limit": self.slots,
                },
            )
        )
        self._semaphore.acquire()
        with self._lock:
            self._active += 1
            self._maximum_active = max(self._maximum_active, self._active)
            active_count = self._active
        try:
            event_sink.emit(
                Event(
                    event="skilllearn_agent_slot_acquired",
                    stage="benchmark.skilllearn.model_concurrency",
                    trace_id=trace_id,
                    payload={
                        "policy": self.policy,
                        "slot_limit": self.slots,
                        "active_count": active_count,
                        "wait_seconds": time.monotonic() - queued_at,
                    },
                )
            )
            yield
        finally:
            with self._lock:
                self._active -= 1
                active_count = self._active
            try:
                event_sink.emit(
                    Event(
                        event="skilllearn_agent_slot_released",
                        stage="benchmark.skilllearn.model_concurrency",
                        trace_id=trace_id,
                        payload={
                            "policy": self.policy,
                            "slot_limit": self.slots,
                            "active_count": active_count,
                        },
                    )
                )
            finally:
                self._semaphore.release()


class TrialVariant(str, Enum):
    POLICY_OFF = "policy_off"
    POLICY_ON = "policy_on"


@dataclass(frozen=True)
class SkillLearnTrialRequest:
    item_id: str
    family: str
    split: SplitName
    variant: TrialVariant
    evaluator_epoch: str
    pair_id: str
    repeat: int
    agent_id: str
    model: str
    max_steps: int
    manifest_hash: str
    codex_agent_execution_policy_hash: str = (
        LEGACY_CODEX_AGENT_EXECUTION_POLICY.policy_hash
    )
    program_id: str | None = None
    program_set_hash: str = ""
    treatment_hash: str = ""
    candidate_delta_program_set_hash: str = ""
    candidate_full_program_set_hash: str = ""
    matched_candidate_program_set_hash: str = ""
    selected_candidate_hypothesis_ids: tuple[str, ...] = ()
    matched_candidate_hypothesis_ids: tuple[str, ...] = ()

    @property
    def request_hash(self) -> str:
        return stable_hash(self.to_dict())

    @property
    def trial_id(self) -> str:
        return f"v2_{self.variant.value}_{self.request_hash[:18]}"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "item_id_hash": stable_hash({"item_id": self.item_id}),
            "family_hash": stable_hash({"family": self.family}),
            "split": self.split.value,
            "variant": self.variant.value,
            "evaluator_epoch": self.evaluator_epoch,
            "pair_id": self.pair_id,
            "repeat": self.repeat,
            "agent_id": self.agent_id,
            "model": self.model,
            "max_steps": self.max_steps,
            "manifest_hash": self.manifest_hash,
            "codex_agent_execution_policy_hash": (
                self.codex_agent_execution_policy_hash
            ),
            "program_id": self.program_id,
            "program_set_hash": self.program_set_hash,
            "treatment_hash": self.treatment_hash,
        }
        if self.candidate_delta_program_set_hash:
            payload.update(
                {
                    "candidate_delta_program_set_hash": (
                        self.candidate_delta_program_set_hash
                    ),
                    "candidate_full_program_set_hash": (
                        self.candidate_full_program_set_hash
                    ),
                    "matched_candidate_program_set_hash": (
                        self.matched_candidate_program_set_hash
                    ),
                    "selected_candidate_hypothesis_ids": list(
                        self.selected_candidate_hypothesis_ids
                    ),
                    "matched_candidate_hypothesis_ids": list(
                        self.matched_candidate_hypothesis_ids
                    ),
                }
            )
        return payload


@dataclass(frozen=True)
class SkillLearnTrialObservation:
    request: SkillLearnTrialRequest
    success: bool
    score: float
    metrics: Mapping[str, float]
    total_tokens: int
    steps: int
    duration_seconds: float
    provider_fingerprint: str
    fairness_fingerprint: str
    error_type: str | None = None
    upstream_result_hash: str = ""
    raw_trial_artifacts_persisted: bool = False
    prebuilt_image_key: str = ""
    prebuilt_image_id: str = ""
    prebuilt_cache_reused: bool = False
    agent_runtime_key: str = ""
    agent_runtime_version: str = ""
    offline_verifier_profile_id: str = ""
    offline_verifier_runtime_key: str = ""
    step_budget_policy: str = ""
    step_budget_unit: str = ""
    step_budget_limit: int = 0
    step_budget_truncated: bool = False
    step_budget_token_usage_complete: bool = False
    step_budget_receipt_hash: str = ""
    proposal_action_trace: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    @property
    def valid(self) -> bool:
        return self.error_type is None

    @property
    def cost_units(self) -> float:
        if self.step_budget_policy == CODEX_ACTION_BUDGET_POLICY_VERSION:
            return float(max(1, self.steps))
        if self.total_tokens > 0:
            return float(self.total_tokens)
        if self.steps > 0:
            return float(self.steps)
        return 1.0

    @property
    def observation_hash(self) -> str:
        return stable_hash(self.to_dict())

    def as_variant(self, request: SkillLearnTrialRequest) -> "SkillLearnTrialObservation":
        return replace(self, request=request)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "request": self.request.to_dict(),
            "success": self.success,
            "score": self.score,
            "metrics": dict(sorted(self.metrics.items())),
            "total_tokens": self.total_tokens,
            "steps": self.steps,
            "duration_seconds": self.duration_seconds,
            "provider_fingerprint": self.provider_fingerprint,
            "fairness_fingerprint": self.fairness_fingerprint,
            "error_type": self.error_type,
            "upstream_result_hash": self.upstream_result_hash,
            "raw_trial_artifacts_persisted": self.raw_trial_artifacts_persisted,
            "prebuilt_image_key": self.prebuilt_image_key,
            "prebuilt_image_id": self.prebuilt_image_id,
            "prebuilt_cache_reused": self.prebuilt_cache_reused,
            "agent_runtime_key": self.agent_runtime_key,
            "agent_runtime_version": self.agent_runtime_version,
            "offline_verifier_profile_id": self.offline_verifier_profile_id,
            "offline_verifier_runtime_key": self.offline_verifier_runtime_key,
            "step_budget_policy": self.step_budget_policy,
            "step_budget_unit": self.step_budget_unit,
            "step_budget_limit": self.step_budget_limit,
            "step_budget_truncated": self.step_budget_truncated,
            "step_budget_token_usage_complete": (
                self.step_budget_token_usage_complete
            ),
            "step_budget_receipt_hash": self.step_budget_receipt_hash,
            "secret_value_persisted": False,
        }
        if self.proposal_action_trace:
            payload["proposal_action_trace_hash"] = stable_hash(
                dict(self.proposal_action_trace)
            )
        return payload


def _baseline_arm_evidence_hash(
    observation: SkillLearnTrialObservation,
) -> str:
    """Hash one policy-off result independently of its challenger pairing."""

    payload = observation.to_dict()
    request = dict(payload["request"])
    for key in (
        "pair_id",
        "candidate_delta_program_set_hash",
        "candidate_full_program_set_hash",
        "matched_candidate_program_set_hash",
        "selected_candidate_hypothesis_ids",
        "matched_candidate_hypothesis_ids",
    ):
        request.pop(key, None)
    payload["request"] = request
    payload.pop("raw_trial_artifacts_persisted", None)
    return stable_hash(payload)


@dataclass(frozen=True)
class BaselineArmEvidenceRecord:
    """Immutable in-memory source record for one validation baseline arm."""

    observation: SkillLearnTrialObservation
    source_trace_id: str
    evidence_hash: str


@dataclass(frozen=True)
class BaselineArmTerminalInvalidMemo:
    """Immutable non-evidence tombstone for one terminal invalid baseline arm."""

    observation: SkillLearnTrialObservation
    source_trace_id: str
    terminal_outcome_hash: str
    error_type: str


BaselineArmReplayEntry = (
    BaselineArmEvidenceRecord | BaselineArmTerminalInvalidMemo
)


class BaselineArmEvidenceReplayCache:
    """Thread-safe policy-off outcome store, optionally shared by paired runners."""

    def __init__(
        self,
        *,
        policy: str = BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
    ) -> None:
        if policy not in BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS:
            raise ValueError(f"unsupported baseline arm replay policy: {policy}")
        self.policy = policy
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._entries: dict[str, BaselineArmReplayEntry] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @contextmanager
    def locked(self, replay_key: str) -> Iterator[None]:
        """Serialize producers for one item without serializing other items."""

        if not replay_key:
            raise ValueError("baseline arm replay key must not be empty")
        with self._lock:
            key_lock = self._key_locks.setdefault(replay_key, threading.Lock())
        with key_lock:
            yield

    def get(self, replay_key: str) -> BaselineArmReplayEntry | None:
        with self._lock:
            return self._entries.get(replay_key)

    def record(
        self,
        replay_key: str,
        *,
        observation: SkillLearnTrialObservation,
        source_trace_id: str,
    ) -> BaselineArmEvidenceRecord | None:
        """Insert once; reject invalid or conflicting evidence without mutation."""

        if not observation.valid:
            return None
        if (
            observation.request.split is not SplitName.VALIDATION
            or observation.request.variant is not TrialVariant.POLICY_OFF
        ):
            return None
        snapshot = replace(
            observation,
            metrics=MappingProxyType(dict(observation.metrics)),
        )
        candidate = BaselineArmEvidenceRecord(
            observation=snapshot,
            source_trace_id=source_trace_id,
            evidence_hash=_baseline_arm_evidence_hash(snapshot),
        )
        with self._lock:
            existing = self._entries.get(replay_key)
            if existing is None:
                self._entries[replay_key] = candidate
                return candidate
            if (
                isinstance(existing, BaselineArmEvidenceRecord)
                and existing.evidence_hash == candidate.evidence_hash
            ):
                return existing
            return None

    def memoize_terminal_invalid(
        self,
        replay_key: str,
        *,
        observation: SkillLearnTrialObservation,
        source_trace_id: str,
    ) -> BaselineArmTerminalInvalidMemo | None:
        """Memoize a terminal invalid outcome without admitting it as evidence."""

        if (
            self.policy
            not in TERMINAL_INVALID_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS
            or observation.valid
        ):
            return None
        if (
            observation.request.split is not SplitName.VALIDATION
            or observation.request.variant is not TrialVariant.POLICY_OFF
        ):
            return None
        snapshot = replace(
            observation,
            metrics=MappingProxyType(dict(observation.metrics)),
        )
        candidate = BaselineArmTerminalInvalidMemo(
            observation=snapshot,
            source_trace_id=source_trace_id,
            terminal_outcome_hash=_baseline_arm_evidence_hash(snapshot),
            error_type=str(snapshot.error_type),
        )
        with self._lock:
            existing = self._entries.get(replay_key)
            if existing is None:
                self._entries[replay_key] = candidate
                return candidate
            if (
                isinstance(existing, BaselineArmTerminalInvalidMemo)
                and existing.terminal_outcome_hash
                == candidate.terminal_outcome_hash
            ):
                return existing
            return None


@dataclass(frozen=True)
class _TrainingEvidenceRecord:
    observations: tuple[SkillLearnTrialObservation, ...]
    source_trace_id: str
    observation_set_hash: str


class TrainingEvidenceReplayCache:
    """Reuse train outcomes only while the executable incumbent is unchanged."""

    def __init__(self, *, event_sink: EventSink | None = None) -> None:
        self.event_sink = event_sink or NullEventSink()
        self._records: dict[str, _TrainingEvidenceRecord] = {}

    def run_or_replay(
        self,
        *,
        descriptor: Mapping[str, Any],
        train_item_ids: Sequence[str],
        producer: Callable[[], tuple[SkillLearnTrialObservation, ...]],
        trace_id: str,
    ) -> tuple[SkillLearnTrialObservation, ...]:
        if descriptor.get("split") != SplitName.TRAIN.value:
            raise PermissionError("training replay is restricted to train evidence")
        if descriptor.get("policy") != TRAINING_EVIDENCE_REPLAY_POLICY_VERSION:
            raise ValueError("training replay descriptor policy mismatch")
        replay_key = stable_hash(dict(descriptor))
        record = self._records.get(replay_key)
        if record is not None:
            _validate_training_observations(
                record.observations,
                train_item_ids=train_item_ids,
                descriptor=descriptor,
            )
            self.event_sink.emit(
                Event(
                    event="training_evidence_replayed",
                    stage="benchmark.skilllearn.training_replay",
                    trace_id=trace_id,
                    payload={
                        "policy": TRAINING_EVIDENCE_REPLAY_POLICY_VERSION,
                        "replay_key": replay_key,
                        "source_trace_id": record.source_trace_id,
                        "target_trace_id": trace_id,
                        "observation_set_hash": record.observation_set_hash,
                        "observation_count": len(record.observations),
                        "incumbent_behavior_set_hash": descriptor[
                            "incumbent_behavior_set_hash"
                        ],
                        "task_set_hash": descriptor["task_set_hash"],
                        "behavior_identical": True,
                        "new_training_executions": 0,
                        "sealed_test_accessed": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            return record.observations

        observations = tuple(producer())
        if any(not row.valid for row in observations):
            self.event_sink.emit(
                Event(
                    event="training_evidence_not_recorded_invalid",
                    stage="benchmark.skilllearn.training_replay",
                    trace_id=trace_id,
                    payload={
                        "policy": TRAINING_EVIDENCE_REPLAY_POLICY_VERSION,
                        "replay_key": replay_key,
                        "observation_count": len(observations),
                        "invalid_observation_count": sum(
                            not row.valid for row in observations
                        ),
                        "new_training_executions": len(observations),
                        "sealed_test_accessed": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            return observations
        _validate_training_observations(
            observations,
            train_item_ids=train_item_ids,
            descriptor=descriptor,
        )
        observation_set_hash = stable_hash(
            {"hashes": [row.observation_hash for row in observations]}
        )
        self._records[replay_key] = _TrainingEvidenceRecord(
            observations=observations,
            source_trace_id=trace_id,
            observation_set_hash=observation_set_hash,
        )
        self.event_sink.emit(
            Event(
                event="training_evidence_recorded",
                stage="benchmark.skilllearn.training_replay",
                trace_id=trace_id,
                payload={
                    "policy": TRAINING_EVIDENCE_REPLAY_POLICY_VERSION,
                    "replay_key": replay_key,
                    "source_trace_id": trace_id,
                    "observation_set_hash": observation_set_hash,
                    "observation_count": len(observations),
                    "incumbent_behavior_set_hash": descriptor[
                        "incumbent_behavior_set_hash"
                    ],
                    "task_set_hash": descriptor["task_set_hash"],
                    "new_training_executions": len(observations),
                    "sealed_test_accessed": False,
                    "raw_content_persisted": False,
                },
            )
        )
        return observations


class SkillLearnTrialBackend(Protocol):
    agent_id: str
    model: str
    max_steps: int
    codex_agent_execution_policy: CodexAgentExecutionPolicy
    codex_agent_execution_policy_hash: str

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation: ...


def codex_agent_execution_policy_for_backend(
    backend: object,
) -> CodexAgentExecutionPolicy:
    policy = getattr(backend, "codex_agent_execution_policy", None)
    return (
        policy
        if isinstance(policy, CodexAgentExecutionPolicy)
        else LEGACY_CODEX_AGENT_EXECUTION_POLICY
    )


def codex_network_minimization_for_policy(
    policy: CodexAgentExecutionPolicy,
) -> str:
    return (
        CODEX_NETWORK_MINIMIZATION_VERSION
        if policy.web_search_mode == "disabled"
        else LEGACY_CODEX_NETWORK_MINIMIZATION_VERSION
    )


def codex_action_supervisor_hash() -> str:
    return _file_content_hash(CODEX_ACTION_SUPERVISOR_PATH)


def shared_codex_agent_runtime_key() -> str:
    return stable_hash(
        {
            "policy": PREBUILT_IMAGE_POLICY_VERSION,
            "builder_image": SHARED_AGENT_RUNTIME_BUILDER_IMAGE,
            "codex_cli_package": SHARED_CODEX_CLI_PACKAGE,
            "codex_action_supervisor_policy": (
                CODEX_ACTION_BUDGET_POLICY_VERSION
            ),
            "codex_action_supervisor_sha256": (
                codex_action_supervisor_hash()
            ),
        }
    )


@dataclass(frozen=True)
class SkillLearnPrebuiltImage:
    tag: str
    cache_key: str
    environment_hash: str
    image_id: str
    agent_runtime_key: str
    agent_runtime_volume: str
    agent_runtime_version: str
    reused: bool


class SkillLearnPrebuiltImageCache:
    """Reuse exact task images; online construction requires an explicit opt-in."""

    def __init__(
        self,
        benchmark_root: str | Path,
        *,
        cache_only: bool = True,
        event_sink: EventSink | None = None,
    ) -> None:
        self.benchmark_root = Path(benchmark_root).expanduser().resolve()
        self.cache_only = cache_only
        self.event_sink = event_sink or NullEventSink()
        self._metadata: dict[tuple[str, str, str], SkillLearnPrebuiltImage] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._state_lock = threading.Lock()
        self._runtime_lock = threading.Lock()
        self._runtime_metadata: dict[str, tuple[str, str, str]] = {}

    def ensure(
        self,
        *,
        family: str,
        item_id: str,
        agent_id: str,
        runner: ModuleType,
        trace_id: str,
    ) -> SkillLearnPrebuiltImage:
        task_root = (self.benchmark_root / "tasks").resolve()
        task_path = (task_root / family / item_id).resolve()
        if task_root not in task_path.parents:
            raise PermissionError("prebuilt task path escaped the benchmark task root")
        environment = task_path / "environment"
        if not (environment / "Dockerfile").is_file():
            raise FileNotFoundError("prebuilt task environment has no Dockerfile")
        agent = runner.get_agent(agent_id)
        if not isinstance(agent, Mapping):
            raise RuntimeError("prebuilt image requires a valid upstream agent definition")
        environment_hash = _directory_content_hash(environment, excluded_top_level={"skills"})
        runner_hash = _file_content_hash(self.benchmark_root / "core" / "eval_runner.py")
        agent_image_hash = stable_hash(
            {
                "agent_id": agent_id,
                "runtime_deps": str(agent.get("runtime_deps") or ""),
                "install": str(agent.get("install") or ""),
            }
        )
        cache_key = stable_hash(
            {
                "policy": PREBUILT_IMAGE_POLICY_VERSION,
                "verifier_isolation": VERIFIER_ISOLATION_VERSION,
                "environment_hash": environment_hash,
                "runner_hash": runner_hash,
                "agent_image_hash": agent_image_hash,
            }
        )
        runtime = self._ensure_agent_runtime(
            runner=runner,
            agent=agent,
            agent_id=agent_id,
            trace_id=f"{trace_id}:agent-runtime",
        )
        metadata_key = (family, item_id, cache_key)
        with self._state_lock:
            cached = self._metadata.get(metadata_key)
            lock = self._locks.setdefault(cache_key, threading.Lock())
        if cached is not None:
            return replace(cached, reused=True)
        with lock:
            with self._state_lock:
                cached = self._metadata.get(metadata_key)
            if cached is not None:
                return replace(cached, reused=True)
            image = self._ensure_image(
                environment=environment,
                agent=agent,
                runner=runner,
                cache_key=cache_key,
                environment_hash=environment_hash,
                agent_runtime_key=runtime[0],
                agent_runtime_volume=runtime[1],
                agent_runtime_version=runtime[2],
                trace_id=trace_id,
            )
            with self._state_lock:
                self._metadata[metadata_key] = image
            return image

    def _ensure_image(
        self,
        *,
        environment: Path,
        agent: Mapping[str, Any],
        runner: ModuleType,
        cache_key: str,
        environment_hash: str,
        agent_runtime_key: str,
        agent_runtime_volume: str,
        agent_runtime_version: str,
        trace_id: str,
    ) -> SkillLearnPrebuiltImage:
        tag = f"assumption-v2-item:{cache_key[:24]}"
        existing_id = _inspect_prebuilt_image(runner, tag, cache_key)
        if existing_id:
            image = SkillLearnPrebuiltImage(
                tag=tag,
                cache_key=cache_key,
                environment_hash=environment_hash,
                image_id=existing_id,
                agent_runtime_key=agent_runtime_key,
                agent_runtime_volume=agent_runtime_volume,
                agent_runtime_version=agent_runtime_version,
                reused=True,
            )
            self._emit_image_event("skilllearn_prebuilt_image_reused", image, trace_id)
            return image

        if self.cache_only:
            self.event_sink.emit(
                Event(
                    event="skilllearn_prebuilt_image_missing_cache_only",
                    stage="benchmark.skilllearn.prebuild",
                    trace_id=trace_id,
                    payload={
                        "cache_key": cache_key,
                        "environment_hash": environment_hash,
                        "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
                        "online_build_attempted": False,
                        "secret_value_persisted": False,
                    },
                )
            )
            raise RuntimeError("prebuilt_image_cache_missing_offline")

        self.event_sink.emit(
            Event(
                event="skilllearn_prebuilt_image_build_started",
                stage="benchmark.skilllearn.prebuild",
                trace_id=trace_id,
                payload={
                    "cache_key": cache_key,
                    "environment_hash": environment_hash,
                    "policy": PREBUILT_IMAGE_POLICY_VERSION,
                    "verifier_isolation": VERIFIER_ISOLATION_VERSION,
                },
            )
        )
        prepare = getattr(runner, "_prepare_build_env", None)
        parse_copies = getattr(runner, "_parse_skill_copies", None)
        if not callable(prepare) or not callable(parse_copies):
            raise RuntimeError("upstream runner lacks required prebuild helpers")
        build_env = prepare(environment, "no_skill", None)
        build_root = build_env.parent
        try:
            for source_pattern, _ in parse_copies(build_env / "Dockerfile"):
                if source_pattern.startswith("skills/"):
                    (build_env / source_pattern).mkdir(parents=True, exist_ok=True)
            base = runner.subprocess.run(
                [
                    "docker",
                    "build",
                    "--label",
                    f"org.assumption-agent.prebuild.key={cache_key}",
                    "--label",
                    f"org.assumption-agent.prebuild.environment={environment_hash}",
                    "--label",
                    f"org.assumption-agent.prebuild.policy={PREBUILT_IMAGE_POLICY_VERSION}",
                    "-t",
                    tag,
                    str(build_env),
                ],
                capture_output=True,
                text=True,
            )
            if int(getattr(base, "returncode", 1)) != 0:
                raise RuntimeError(
                    "prebuilt_base_build_failed:"
                    + _safe_subprocess_snippet(base)
                )
            image_id = _inspect_prebuilt_image(runner, tag, cache_key)
            if not image_id:
                raise RuntimeError("prebuilt_image_inspection_failed")
            image = SkillLearnPrebuiltImage(
                tag=tag,
                cache_key=cache_key,
                environment_hash=environment_hash,
                image_id=image_id,
                agent_runtime_key=agent_runtime_key,
                agent_runtime_volume=agent_runtime_volume,
                agent_runtime_version=agent_runtime_version,
                reused=False,
            )
            self._emit_image_event("skilllearn_prebuilt_image_built", image, trace_id)
            return image
        finally:
            shutil.rmtree(build_root, ignore_errors=True)

    def _ensure_agent_runtime(
        self,
        *,
        runner: ModuleType,
        agent: Mapping[str, Any],
        agent_id: str,
        trace_id: str,
    ) -> tuple[str, str, str]:
        if agent_id != "codex":
            raise ValueError("shared agent runtime currently supports the frozen codex agent only")
        install = str(agent.get("install") or "").strip()
        if not install.startswith("npm ") or "@openai/codex" not in install:
            raise ValueError("shared codex runtime requires an npm agent install command")
        runtime_key = shared_codex_agent_runtime_key()
        volume = f"assumption-v2-agent-{runtime_key[:24]}"
        with self._runtime_lock:
            cached = self._runtime_metadata.get(runtime_key)
            if cached is not None:
                return cached
            inspected = runner.subprocess.run(
                ["docker", "volume", "inspect", volume],
                capture_output=True,
                text=True,
            )
            if int(getattr(inspected, "returncode", 1)) != 0:
                if self.cache_only:
                    self.event_sink.emit(
                        Event(
                            event="skilllearn_agent_runtime_missing_cache_only",
                            stage="benchmark.skilllearn.prebuild",
                            trace_id=trace_id,
                            payload={
                                "runtime_key": runtime_key,
                                "runtime_volume_hash": stable_hash({"volume": volume}),
                                "dependency_cache_policy": (
                                    DEPENDENCY_CACHE_POLICY_VERSION
                                ),
                                "online_install_attempted": False,
                                "secret_value_persisted": False,
                            },
                        )
                    )
                    raise RuntimeError("shared_agent_runtime_cache_missing_offline")
                created = runner.subprocess.run(
                    [
                        "docker",
                        "volume",
                        "create",
                        "--label",
                        f"org.assumption-agent.runtime.key={runtime_key}",
                        "--label",
                        f"org.assumption-agent.runtime.policy={PREBUILT_IMAGE_POLICY_VERSION}",
                        volume,
                    ],
                    capture_output=True,
                    text=True,
                )
                if int(getattr(created, "returncode", 1)) != 0:
                    raise RuntimeError(
                        "shared_agent_runtime_volume_create_failed:"
                        + _safe_subprocess_snippet(created)
                    )
                populate_command = (
                    "set -eu; mkdir -p /runtime/bin; "
                    "cp -L \"$(command -v node)\" /runtime/bin/node; "
                    f"cp /supervisor/{CODEX_ACTION_SUPERVISOR_FILENAME} "
                    f"/runtime/{CODEX_ACTION_SUPERVISOR_FILENAME}; "
                    f"chmod 0444 /runtime/{CODEX_ACTION_SUPERVISOR_FILENAME}; "
                    f"npm_config_prefix=/runtime npm install -g {SHARED_CODEX_CLI_PACKAGE}"
                )
                populated = runner.subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--pull",
                        "never",
                        "-v",
                        f"{volume}:/runtime",
                        "-v",
                        (
                            f"{CODEX_ACTION_SUPERVISOR_PATH}:"
                            f"/supervisor/{CODEX_ACTION_SUPERVISOR_FILENAME}:ro"
                        ),
                        SHARED_AGENT_RUNTIME_BUILDER_IMAGE,
                        "sh",
                        "-lc",
                        populate_command,
                    ],
                    capture_output=True,
                    text=True,
                )
                if int(getattr(populated, "returncode", 1)) != 0:
                    runner.subprocess.run(
                        ["docker", "volume", "rm", "-f", volume],
                        capture_output=True,
                    )
                    raise RuntimeError(
                        "shared_agent_runtime_population_failed:"
                        + _safe_subprocess_snippet(populated)
                    )
            else:
                try:
                    volume_payload = json.loads(
                        str(getattr(inspected, "stdout", "") or "")
                    )[0]
                    labels = volume_payload.get("Labels") or {}
                    if labels.get("org.assumption-agent.runtime.key") != runtime_key:
                        raise PermissionError(
                            "shared agent runtime volume label does not match its key"
                        )
                except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError("shared agent runtime metadata is malformed") from exc
            verified_supervisor = runner.subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--pull",
                    "never",
                    "--network",
                    "none",
                    "-e",
                    f"PATH={SHARED_AGENT_RUNTIME_MOUNT}/bin:/usr/local/bin:/usr/bin:/bin",
                    "-v",
                    f"{volume}:{SHARED_AGENT_RUNTIME_MOUNT}:ro",
                    SHARED_AGENT_RUNTIME_BUILDER_IMAGE,
                    "sh",
                    "-lc",
                    (
                        "test \"$(sha256sum "
                        f"{SHARED_AGENT_RUNTIME_MOUNT}/"
                        f"{CODEX_ACTION_SUPERVISOR_FILENAME} | cut -d' ' -f1)\" "
                        f"= {codex_action_supervisor_hash()}"
                    ),
                ],
                capture_output=True,
                text=True,
            )
            if int(getattr(verified_supervisor, "returncode", 1)) != 0:
                raise RuntimeError(
                    "shared_agent_runtime_supervisor_verification_failed:"
                    + _safe_subprocess_snippet(verified_supervisor)
                )
            verified = runner.subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--pull",
                    "never",
                    "--network",
                    "none",
                    "-e",
                    f"PATH={SHARED_AGENT_RUNTIME_MOUNT}/bin:/usr/local/bin:/usr/bin:/bin",
                    "-v",
                    f"{volume}:{SHARED_AGENT_RUNTIME_MOUNT}:ro",
                    SHARED_AGENT_RUNTIME_BUILDER_IMAGE,
                    "sh",
                    "-c",
                    "codex --version",
                ],
                capture_output=True,
                text=True,
            )
            if int(getattr(verified, "returncode", 1)) != 0:
                raise RuntimeError(
                    "shared_agent_runtime_verification_failed:"
                    + _safe_subprocess_snippet(verified)
                )
            version = re.sub(
                r"[^a-zA-Z0-9._ -]+",
                "",
                str(getattr(verified, "stdout", "") or "").strip(),
            )[:96]
            if not version:
                raise RuntimeError("shared_agent_runtime_version_missing")
            if version != SHARED_CODEX_CLI_VERSION:
                raise RuntimeError("shared_agent_runtime_version_mismatch")
            self.event_sink.emit(
                Event(
                    event="skilllearn_shared_agent_runtime_ready",
                    stage="benchmark.skilllearn.prebuild",
                    trace_id=trace_id,
                    payload={
                        "runtime_key": runtime_key,
                        "runtime_volume_hash": stable_hash({"volume": volume}),
                        "runtime_version": version,
                        "builder_image": SHARED_AGENT_RUNTIME_BUILDER_IMAGE,
                        "codex_cli_package": SHARED_CODEX_CLI_PACKAGE,
                        "codex_action_supervisor_policy": (
                            CODEX_ACTION_BUDGET_POLICY_VERSION
                        ),
                        "codex_action_supervisor_sha256": (
                            codex_action_supervisor_hash()
                        ),
                        "policy": PREBUILT_IMAGE_POLICY_VERSION,
                        "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
                        "cache_only": self.cache_only,
                    },
                )
            )
            metadata = (runtime_key, volume, version)
            self._runtime_metadata[runtime_key] = metadata
            return metadata

    def _emit_image_event(
        self,
        event: str,
        image: SkillLearnPrebuiltImage,
        trace_id: str,
    ) -> None:
        self.event_sink.emit(
            Event(
                event=event,
                stage="benchmark.skilllearn.prebuild",
                trace_id=trace_id,
                payload={
                    "cache_key": image.cache_key,
                    "environment_hash": image.environment_hash,
                    "image_id": image.image_id,
                    "agent_runtime_key": image.agent_runtime_key,
                    "agent_runtime_volume_hash": stable_hash(
                        {"volume": image.agent_runtime_volume}
                    ),
                    "agent_runtime_version": image.agent_runtime_version,
                    "reused": image.reused,
                    "policy": PREBUILT_IMAGE_POLICY_VERSION,
                    "raw_content_persisted": False,
                },
            )
        )


class SkillLearnBackendPool:
    """Bound concurrent calls to independent upstream runner instances."""

    def __init__(self, backends: Sequence[SkillLearnTrialBackend]) -> None:
        if not backends:
            raise ValueError("backend pool cannot be empty")
        first = backends[0]
        first_policy = codex_agent_execution_policy_for_backend(first)
        if any(
            (
                row.agent_id,
                row.model,
                row.max_steps,
                codex_agent_execution_policy_for_backend(row).policy_hash,
            )
            != (
                first.agent_id,
                first.model,
                first.max_steps,
                first_policy.policy_hash,
            )
            for row in backends
        ):
            raise ValueError("all pooled backends must share one frozen configuration")
        self.agent_id = first.agent_id
        self.model = first.model
        self.max_steps = first.max_steps
        self.codex_agent_execution_policy = first_policy
        self.codex_agent_execution_policy_hash = first_policy.policy_hash
        self._available: queue.Queue[SkillLearnTrialBackend] = queue.Queue()
        for backend in backends:
            self._available.put(backend)

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        backend = self._available.get()
        try:
            return backend.run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
        finally:
            self._available.put(backend)


class SkillLearnSubprocessBackend:
    """Thin, sanitized adapter around SkillLearnBench's Docker trial runner."""

    def __init__(
        self,
        benchmark_root: str | Path,
        *,
        agent_id: str = "codex",
        model: str = "gpt-5.4-mini",
        max_steps: int = 100,
        provider_mode: str | None = None,
        trials_dir: str | Path | None = None,
        record_upstream: bool = True,
        prebuilt_cache: SkillLearnPrebuiltImageCache | None = None,
        offline_verifier_cache: SkillLearnOfflineVerifierRuntimeCache | None = None,
        provider_circuit: SkillLearnProviderCircuit | None = None,
        model_inference_limiter: SkillLearnModelInferenceLimiter | None = None,
        train_action_design_policy: str | None = None,
        codex_agent_execution_policy: CodexAgentExecutionPolicy = (
            LEGACY_CODEX_AGENT_EXECUTION_POLICY
        ),
        event_sink: EventSink | None = None,
    ) -> None:
        if train_action_design_policy not in {
            None,
            *TRAIN_ACTION_DESIGN_POLICY_VERSIONS,
        }:
            raise ValueError(
                "unsupported TRAIN action design policy: "
                f"{train_action_design_policy}"
            )
        self.benchmark_root = Path(benchmark_root).expanduser().resolve()
        self.agent_id = agent_id
        self.model = model
        self.max_steps = max_steps
        self.codex_agent_execution_policy = codex_agent_execution_policy
        self.codex_agent_execution_policy_hash = codex_agent_execution_policy.policy_hash
        self.provider_mode = (
            provider_mode
            or (
                configured_skilllearn_provider_mode()
                if agent_id == "codex"
                else "openai_compatible"
            )
        )
        if self.provider_mode != "openai_compatible":
            raise ValueError("unsupported SkillLearn trial provider mode")
        self.trials_dir = Path(trials_dir).expanduser().resolve() if trials_dir else None
        self.record_upstream = record_upstream
        self.prebuilt_cache = prebuilt_cache
        self.provider_circuit = provider_circuit or SkillLearnProviderCircuit()
        self.model_inference_limiter = model_inference_limiter
        self.train_action_design_policy = train_action_design_policy
        self.trial_network_byte_limit = configured_trial_network_byte_limit()
        self.event_sink = event_sink or NullEventSink()
        self.offline_verifier_cache = (
            offline_verifier_cache
            or SkillLearnOfflineVerifierRuntimeCache(event_sink=self.event_sink)
        )
        self._runner_module: ModuleType | None = None
        self._runner_instance_token = stable_hash(
            {"benchmark_root": str(self.benchmark_root), "instance": id(self)}
        )[:12]

    def prewarm_environment(
        self,
        *,
        family: str,
        item_id: str,
        trace_id: str,
    ) -> SkillLearnPrebuiltImage:
        if self.prebuilt_cache is None:
            raise RuntimeError("environment prewarm requires the prebuilt image cache")
        return self.prebuilt_cache.ensure(
            family=family,
            item_id=item_id,
            agent_id=self.agent_id,
            runner=self._load_runner(),
            trace_id=trace_id,
        )

    def prewarm_trial_environment(
        self,
        *,
        family: str,
        item_id: str,
        trace_id: str,
    ) -> tuple[SkillLearnPrebuiltImage, OfflineVerifierRuntime | None]:
        image = self.prewarm_environment(
            family=family,
            item_id=item_id,
            trace_id=trace_id,
        )
        profile = offline_verifier_profile_for_family(family)
        test_script = (
            self.benchmark_root
            / "tasks"
            / family
            / item_id
            / "tests"
            / "test.sh"
        )
        if profile is None:
            if test_script_requires_offline_profile(test_script):
                activation_blocker = (
                    offline_verifier_activation_blocker_for_family(family)
                )
                if activation_blocker is not None:
                    self.event_sink.emit(
                        Event(
                            event=(
                                "skilllearn_trial_prewarm_blocked_inactive_"
                                "offline_verifier_profile"
                            ),
                            stage="benchmark.skilllearn.offline_verifier",
                            trace_id=trace_id,
                            payload={
                                "policy": OFFLINE_VERIFIER_POLICY_VERSION,
                                "family_hash": stable_hash({"family": family}),
                                "activation_blocker": activation_blocker,
                                "model_container_started": False,
                                "runtime_network_attempted": False,
                                "raw_content_persisted": False,
                            },
                        )
                    )
                    raise RuntimeError("offline_verifier_profile_inactive")
                raise RuntimeError("offline_verifier_profile_missing")
            return image, None
        runtime = self.offline_verifier_cache.ensure(
            profile=profile,
            base_image_tag=image.tag,
            base_image_id=image.image_id,
            delegate=self._load_runner().subprocess,
            trace_id=f"{trace_id}:offline-verifier",
        )
        return image, runtime

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        if request.agent_id != self.agent_id or request.model != self.model:
            raise ValueError("trial request does not match the frozen backend model configuration")
        if request.max_steps != self.max_steps:
            raise ValueError("trial request does not match the frozen backend step budget")
        if (
            request.codex_agent_execution_policy_hash
            != self.codex_agent_execution_policy_hash
        ):
            raise ValueError(
                "trial request does not match the frozen Codex agent execution policy"
            )
        circuit_error = self.provider_circuit.error_type
        if circuit_error:
            self.event_sink.emit(
                Event(
                    event="skilllearn_trial_skipped_provider_circuit_open",
                    stage="benchmark.skilllearn.provider_failure",
                    trace_id=trace_id,
                    payload={
                        "request_hash": request.request_hash,
                        "provider_error_type": circuit_error,
                        "policy": PROVIDER_FAILURE_POLICY_VERSION,
                    },
                )
            )
            return self._local_error(
                request,
                f"provider_circuit_open_{circuit_error}",
            )
        if request.variant is TrialVariant.POLICY_ON and skill_source_dir is None:
            return self._local_error(request, "candidate_skill_source_missing")
        offline_verifier_profile = offline_verifier_profile_for_family(request.family)
        verifier_script = (
            self.benchmark_root
            / "tasks"
            / request.family
            / request.item_id
            / "tests"
            / "test.sh"
        )
        if (
            offline_verifier_profile is None
            and test_script_requires_offline_profile(verifier_script)
        ):
            catalog_profile = offline_verifier_catalog_profile_for_family(
                request.family
            )
            activation_blocker = (
                offline_verifier_activation_blocker_for_family(request.family)
            )
            inactive = activation_blocker is not None
            self.event_sink.emit(
                Event(
                    event=(
                        "skilllearn_trial_blocked_inactive_offline_verifier_profile"
                        if inactive
                        else "skilllearn_trial_blocked_missing_offline_verifier_profile"
                    ),
                    stage="benchmark.skilllearn.offline_verifier",
                    trace_id=trace_id,
                    payload={
                        "policy": OFFLINE_VERIFIER_POLICY_VERSION,
                        "request_hash": request.request_hash,
                        "family_hash": stable_hash({"family": request.family}),
                        "catalog_profile_id": (
                            catalog_profile.profile_id if catalog_profile else None
                        ),
                        "catalog_profile_hash": (
                            catalog_profile.profile_hash if catalog_profile else None
                        ),
                        "activation_blocker": activation_blocker,
                        "model_container_started": False,
                        "runtime_network_attempted": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            return self._local_error(
                request,
                (
                    "offline_verifier_profile_inactive"
                    if inactive
                    else "offline_verifier_profile_missing"
                ),
            )

        self.event_sink.emit(
            Event(
                event="skilllearn_trial_started",
                stage="benchmark.skilllearn.trial",
                trace_id=trace_id,
                payload={
                    "request_hash": request.request_hash,
                    "variant": request.variant.value,
                    "split": request.split.value,
                    "model": request.model,
                    "max_steps": request.max_steps,
                    "provider_mode": self.provider_mode,
                    "verifier_isolation": VERIFIER_ISOLATION_VERSION,
                    "verifier_execution_receipt_policy": (
                        VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION
                    ),
                    "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
                    "offline_verifier_profile_id": (
                        offline_verifier_profile.profile_id
                        if offline_verifier_profile
                        else None
                    ),
                    "runner_agent_registry_isolation": (
                        RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
                    ),
                    "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
                    "provider_failure_policy": PROVIDER_FAILURE_POLICY_VERSION,
                    "model_inference_concurrency_policy": (
                        self.model_inference_limiter.policy
                        if self.model_inference_limiter is not None
                        else None
                    ),
                    "model_inference_slots": (
                        self.model_inference_limiter.slots
                        if self.model_inference_limiter is not None
                        else None
                    ),
                    "openai_compatible_codex_config": (
                        OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION
                        if self.provider_mode == "openai_compatible"
                        else None
                    ),
                    "codex_network_minimization": (
                        codex_network_minimization_for_policy(
                            self.codex_agent_execution_policy
                        )
                    ),
                    "codex_agent_execution_policy": (
                        self.codex_agent_execution_policy.to_dict()
                    ),
                    "codex_agent_execution_policy_hash": (
                        self.codex_agent_execution_policy_hash
                    ),
                    "model_only_tool_policy": MODEL_ONLY_TOOL_POLICY_VERSION,
                    "trial_network_budget_policy": (
                        TRIAL_NETWORK_BUDGET_POLICY_VERSION
                    ),
                    "trial_network_byte_limit": self.trial_network_byte_limit,
                    "prebuilt_policy": (
                        PREBUILT_IMAGE_POLICY_VERSION if self.prebuilt_cache else "disabled"
                    ),
                    "skill_source_hash": stable_hash({"path": str(skill_source_dir)}) if skill_source_dir else None,
                },
            )
        )
        started = time.monotonic()
        result: Mapping[str, Any]
        return_code: int
        prebuilt_image: SkillLearnPrebuiltImage | None = None
        offline_verifier_runtime: OfflineVerifierRuntime | None = None
        egress_policy: DockerEgressPolicy | None = None
        try:
            runner = self._load_runner()
            if self.prebuilt_cache is None or not self.prebuilt_cache.cache_only:
                raise RuntimeError("skilllearn_trial_requires_cache_only_prebuilt_images")
            prebuilt_image = self.prebuilt_cache.ensure(
                family=request.family,
                item_id=request.item_id,
                agent_id=self.agent_id,
                runner=runner,
                trace_id=f"{trace_id}:prebuild",
            )
            if offline_verifier_profile is not None:
                offline_verifier_runtime = self.offline_verifier_cache.ensure(
                    profile=offline_verifier_profile,
                    base_image_tag=prebuilt_image.tag,
                    base_image_id=prebuilt_image.image_id,
                    delegate=runner.subprocess,
                    trace_id=f"{trace_id}:offline-verifier",
                )
            egress_policy = DockerEgressPolicy.from_env()
            egress_policy.ensure(
                event_sink=self.event_sink,
                trace_id=f"{trace_id}:egress",
            )
            self.event_sink.emit(
                Event(
                    event="skilllearn_local_evidence_and_network_scope_declared",
                    stage="benchmark.skilllearn.evidence_transport",
                    trace_id=trace_id,
                    payload={
                        "local_evidence_policy": LOCAL_EVIDENCE_TRANSPORT_VERSION,
                        "network_audit_policy": NETWORK_SCOPE_AUDIT_VERSION,
                        "benchmark_task_transport": (
                            "local_content_addressed_prebuilt_image"
                            if prebuilt_image is not None
                            else "local_task_directory"
                        ),
                        "task_environment_hash": (
                            prebuilt_image.environment_hash if prebuilt_image else None
                        ),
                        "prebuilt_image_id": (
                            prebuilt_image.image_id if prebuilt_image else None
                        ),
                        "verifier_transport": "local_docker_copy_after_agent_exit",
                        "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
                        "offline_verifier_profile_id": (
                            offline_verifier_runtime.profile.profile_id
                            if offline_verifier_runtime
                            else None
                        ),
                        "offline_verifier_runtime_key": (
                            offline_verifier_runtime.runtime_key
                            if offline_verifier_runtime
                            else None
                        ),
                        "verifier_dependency_transport": (
                            "local_readonly_content_addressed_volume"
                            if offline_verifier_runtime
                            else "unsupported_family_fails_receipt_closed"
                        ),
                        "model_transport": "online_openai_compatible_responses",
                        "model_endpoint_origin": _configured_openai_compatible_origin(),
                        "huggingface_dataset_access_required": False,
                        "online_benchmark_dataset_access_required": False,
                        "container_egress_isolation_enforced": True,
                        "dependency_cache_only_enforced": True,
                        "trial_network_budget_policy": (
                            TRIAL_NETWORK_BUDGET_POLICY_VERSION
                        ),
                        "trial_network_byte_limit": self.trial_network_byte_limit,
                        **egress_policy.provenance(),
                        "secret_value_persisted": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            if skill_source_dir is None:
                skill_config = "no_skill"
            elif request.variant is TrialVariant.POLICY_OFF:
                skill_config = "assumption-agent-v2-incumbent"
            else:
                skill_config = "assumption-agent-v2-challenger"
            kwargs: dict[str, Any] = {
                "task_root": self.benchmark_root / "tasks",
                "agent_id": self.agent_id,
                "model": self.model,
                "record": self.record_upstream,
                "skill_config": skill_config,
                "skill_source_dir": skill_source_dir,
                "trial_id": request.trial_id,
                "max_steps": self.max_steps,
            }
            if prebuilt_image is not None:
                kwargs["prebuilt_image_tag"] = prebuilt_image.tag
                kwargs["prebuilt_has_agent"] = True
            if self.trials_dir is not None:
                kwargs["trials_dir"] = self.trials_dir
            with self._provider_runtime(
                runner,
                agent_runtime_volume=(
                    prebuilt_image.agent_runtime_volume if prebuilt_image else None
                ),
                egress_policy=egress_policy,
                offline_verifier_runtime=offline_verifier_runtime,
                trace_id=trace_id,
            ):
                return_code, result = runner.run_task(
                    f"{request.family}/{request.item_id}",
                    **kwargs,
                )
                result = self._audit_trial_artifacts(
                    runner=runner,
                    request=request,
                    skill_config=skill_config,
                    result=result,
                    offline_verifier_profile=(
                        offline_verifier_runtime.profile
                        if offline_verifier_runtime is not None
                        else None
                    ),
                    trace_id=trace_id,
                )
        except Exception as exc:
            caught_error_type = (
                exc.error_type
                if isinstance(exc, SkillLearnAgentTerminalError)
                else type(exc).__name__
            )
            self.event_sink.emit(
                Event(
                    event="skilllearn_trial_infrastructure_failed",
                    stage="benchmark.skilllearn.trial",
                    trace_id=trace_id,
                    payload={
                        "request_hash": request.request_hash,
                        "error_type": caught_error_type,
                        "error_message_hash": stable_hash({"message": str(exc)}),
                        "prebuilt_stage": bool(
                            self.prebuilt_cache is not None and prebuilt_image is None
                        ),
                        "raw_content_persisted": False,
                    },
                )
            )
            result = {"error": caught_error_type}
            return_code = 2
        observation = self._sanitize_result(
            request,
            result=result,
            return_code=return_code,
            duration_seconds=time.monotonic() - started,
            prebuilt_image=prebuilt_image,
            offline_verifier_runtime=offline_verifier_runtime,
        )
        if observation.error_type in _FATAL_PROVIDER_ERROR_TYPES:
            opened = self.provider_circuit.open(observation.error_type)
            self.event_sink.emit(
                Event(
                    event=(
                        "skilllearn_provider_circuit_opened"
                        if opened
                        else "skilllearn_provider_circuit_already_open"
                    ),
                    stage="benchmark.skilllearn.provider_failure",
                    trace_id=trace_id,
                    payload={
                        "request_hash": request.request_hash,
                        "provider_error_type": observation.error_type,
                        "policy": PROVIDER_FAILURE_POLICY_VERSION,
                    },
                )
            )
        self.event_sink.emit(
            Event(
                event="skilllearn_trial_completed",
                stage="benchmark.skilllearn.trial",
                trace_id=trace_id,
                payload={
                    "request_hash": request.request_hash,
                    "observation_hash": observation.observation_hash,
                    "variant": request.variant.value,
                    "success": observation.success,
                    "valid": observation.valid,
                    "error_type": observation.error_type,
                    "duration_seconds": observation.duration_seconds,
                    "total_tokens": observation.total_tokens,
                    "steps": observation.steps,
                    "metrics": dict(sorted(observation.metrics.items())),
                    "upstream_result_hash": observation.upstream_result_hash,
                    "provider_fingerprint": observation.provider_fingerprint,
                    "fairness_fingerprint": observation.fairness_fingerprint,
                    "raw_trial_artifacts_persisted": observation.raw_trial_artifacts_persisted,
                    "prebuilt_image_key": observation.prebuilt_image_key,
                    "prebuilt_image_id": observation.prebuilt_image_id,
                    "prebuilt_cache_reused": observation.prebuilt_cache_reused,
                    "agent_runtime_key": observation.agent_runtime_key,
                    "agent_runtime_version": observation.agent_runtime_version,
                    "offline_verifier_profile_id": (
                        observation.offline_verifier_profile_id
                    ),
                    "offline_verifier_runtime_key": (
                        observation.offline_verifier_runtime_key
                    ),
                    "step_budget_policy": observation.step_budget_policy,
                    "step_budget_unit": observation.step_budget_unit,
                    "step_budget_limit": observation.step_budget_limit,
                    "step_budget_truncated": (
                        observation.step_budget_truncated
                    ),
                    "step_budget_token_usage_complete": (
                        observation.step_budget_token_usage_complete
                    ),
                    "step_budget_receipt_hash": (
                        observation.step_budget_receipt_hash
                    ),
                },
            )
        )
        return observation

    def _load_runner(self) -> ModuleType:
        if self._runner_module is not None:
            return self._runner_module
        path = self.benchmark_root / "core" / "eval_runner.py"
        if not path.is_file():
            raise FileNotFoundError(f"SkillLearnBench eval runner not found: {path}")
        module_name = (
            f"_assumption_v2_skilllearn_eval_"
            f"{stable_hash({'path': str(path)})[:12]}_{self._runner_instance_token}"
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load SkillLearnBench eval runner")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        shared_get_agent = module.get_agent
        shared_list_agents = module.list_agents
        local_agents = {
            str(agent_id): copy.deepcopy(shared_get_agent(agent_id))
            for agent_id in shared_list_agents()
            if shared_get_agent(agent_id) is not None
        }
        module.get_agent = lambda agent_id: local_agents.get(str(agent_id))
        module.list_agents = lambda: sorted(local_agents)
        module._assumption_v2_local_agents = local_agents
        self._runner_module = module
        return module

    @contextmanager
    def _provider_runtime(
        self,
        runner: ModuleType,
        *,
        agent_runtime_volume: str | None = None,
        egress_policy: DockerEgressPolicy | None = None,
        offline_verifier_runtime: OfflineVerifierRuntime | None = None,
        trace_id: str = "skilllearn-provider-runtime",
    ) -> Iterator[None]:
        active_egress_policy = egress_policy or DockerEgressPolicy.from_env()
        if self.agent_id != "codex":
            with self._verifier_isolation(
                runner,
                agent_runtime_volume=agent_runtime_volume,
                egress_policy=active_egress_policy,
                offline_verifier_runtime=offline_verifier_runtime,
                trace_id=trace_id,
            ):
                yield
            return
        agent = runner.get_agent(self.agent_id)
        original_agent = copy.deepcopy(agent) if isinstance(agent, dict) else None
        try:
            self._prepare_openai_compatible_provider(
                runner,
                trace_id=trace_id,
            )
            if isinstance(agent, dict) and agent_runtime_volume:
                trajectory_env = dict(agent.get("trajectory_env") or {})
                trajectory_env["PATH"] = f"{SHARED_AGENT_RUNTIME_MOUNT}/bin:$PATH"
                if offline_verifier_runtime is not None:
                    trajectory_env["PYTHONPATH"] = (
                        f"{OFFLINE_VERIFIER_MOUNT}/site"
                    )
                agent["trajectory_env"] = trajectory_env
            with self._verifier_isolation(
                runner,
                agent_runtime_volume=agent_runtime_volume,
                egress_policy=active_egress_policy,
                offline_verifier_runtime=offline_verifier_runtime,
                trace_id=trace_id,
            ):
                yield
        finally:
            if isinstance(agent, dict) and original_agent is not None:
                agent.clear()
                agent.update(original_agent)

    @contextmanager
    def _verifier_isolation(
        self,
        runner: ModuleType,
        *,
        agent_runtime_volume: str | None = None,
        egress_policy: DockerEgressPolicy,
        offline_verifier_runtime: OfflineVerifierRuntime | None = None,
        trace_id: str = "skilllearn-verifier-isolation",
    ) -> Iterator[None]:
        original_subprocess = runner.subprocess
        proxy = _DockerVerifierIsolationSubprocessProxy(
            original_subprocess,
            agent_runtime_volume=agent_runtime_volume,
            offline_verifier_runtime=offline_verifier_runtime,
            egress_policy=egress_policy,
            network_byte_limit=self.trial_network_byte_limit,
            provider_mode=self.provider_mode,
            model_inference_limiter=self.model_inference_limiter,
            event_sink=self.event_sink,
            trace_id=trace_id,
        )
        runner.subprocess = proxy
        try:
            yield
        finally:
            proxy.finalize_network_monitors()
            runner.subprocess = original_subprocess

    def _prepare_openai_compatible_provider(
        self,
        runner: ModuleType,
        *,
        trace_id: str,
    ) -> None:
        api_key = (
            os.environ.get("ASSUMPTION_V2_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        base_url = (
            os.environ.get("ASSUMPTION_V2_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        ).strip()
        if not api_key:
            raise RuntimeError("openai-compatible provider API key is required")
        if not base_url:
            raise RuntimeError("openai-compatible provider base URL is required")
        codex_base_url = _normalize_openai_compatible_codex_base_url(base_url)
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = codex_base_url
        agent = runner.get_agent(self.agent_id)
        if not isinstance(agent, dict):
            raise RuntimeError("SkillLearnBench codex agent definition is unavailable")
        env_names = list(agent.get("env") or [])
        for env_name in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            if env_name not in env_names:
                env_names.append(env_name)
        run_template = str(agent.get("run") or "")
        if "codex exec" not in run_template:
            raise RuntimeError("SkillLearnBench codex run template is unavailable")
        config_values = _openai_compatible_codex_config_values(
            policy=self.codex_agent_execution_policy,
            codex_base_url=codex_base_url,
        )
        codex_config = " ".join(shlex.quote(value) for value in config_values)
        agent["env"] = env_names
        agent["setup"] = None
        codex_binary = "codex"
        if self.codex_agent_execution_policy.action_budget_enforced:
            codex_binary = f"{SHARED_AGENT_RUNTIME_MOUNT}/bin/codex"
        codex_command = f"{shlex.quote(codex_binary)} exec {codex_config}"
        if self.codex_agent_execution_policy.action_budget_enforced:
            trajectory_env = dict(agent.get("trajectory_env") or {})
            codex_home = str(trajectory_env.get("CODEX_HOME") or "/logs/agent")
            receipt_path = (
                f"{codex_home.rstrip('/')}/codex_action_budget_receipt.json"
            )
            trace_path = f"{codex_home.rstrip('/')}/codex.txt"
            codex_command = " ".join(
                (
                    "rm",
                    "-f",
                    shlex.quote(receipt_path),
                    shlex.quote(trace_path),
                    "&&",
                    "env",
                    shlex.quote(
                        "PATH="
                        f"{SHARED_AGENT_RUNTIME_MOUNT}/bin:"
                        "/usr/local/bin:/usr/bin:/bin"
                    ),
                    shlex.quote(
                        f"{SHARED_AGENT_RUNTIME_MOUNT}/bin/node"
                    ),
                    shlex.quote(
                        f"{SHARED_AGENT_RUNTIME_MOUNT}/"
                        f"{CODEX_ACTION_SUPERVISOR_FILENAME}"
                    ),
                    "--limit",
                    str(self.max_steps),
                    "--receipt",
                    shlex.quote(receipt_path),
                    "--trace",
                    shlex.quote(trace_path),
                    "--process-scope",
                    str(
                        self.codex_agent_execution_policy
                        .action_budget_process_scope
                    ),
                    "--",
                    codex_command,
                )
            )
            agent["trajectory_tee"] = None
        agent["run"] = run_template.replace("codex exec", codex_command, 1)
        endpoint = urlsplit(codex_base_url)
        endpoint_origin = f"{endpoint.scheme}://{endpoint.hostname}"
        if endpoint.port is not None:
            endpoint_origin = f"{endpoint_origin}:{endpoint.port}"
        self.event_sink.emit(
            Event(
                event="skilllearn_openai_compatible_provider_prepared",
                stage="benchmark.skilllearn.provider_config",
                trace_id=trace_id,
                payload={
                    "config_version": OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION,
                    "model": self.model,
                    "provider_mode": self.provider_mode,
                    "provider_id": OPENAI_COMPATIBLE_CODEX_PROVIDER_ID,
                    "endpoint_origin": endpoint_origin,
                    "endpoint_base_hash": stable_hash(
                        {"base_url": codex_base_url}
                    ),
                    "api_key_env_name": "OPENAI_API_KEY",
                    "wire_api": "responses",
                    "supports_websockets": False,
                    "requires_openai_auth": False,
                    "ignore_user_config": True,
                    "ephemeral": True,
                    "analytics_enabled": False,
                    "otel_exporter": "none",
                    "otel_metrics_exporter": "none",
                    "otel_trace_exporter": "none",
                    "strict_config": True,
                    "web_search_mode": (
                        self.codex_agent_execution_policy.web_search_mode
                        or "catalog_default"
                    ),
                    "web_search_enabled": (
                        self.codex_agent_execution_policy.web_search_mode
                        != "disabled"
                    ),
                    "image_generation_enabled": False,
                    "runtime_package_install_allowed": False,
                    "model_only_tool_policy": MODEL_ONLY_TOOL_POLICY_VERSION,
                    "network_minimization_version": (
                        codex_network_minimization_for_policy(
                            self.codex_agent_execution_policy
                        )
                    ),
                    "action_budget_policy": (
                        self.codex_agent_execution_policy.action_budget_policy
                    ),
                    "action_budget_unit": (
                        self.codex_agent_execution_policy.action_budget_unit
                    ),
                    "action_budget_cost_accounting_policy": (
                        self.codex_agent_execution_policy
                        .action_budget_cost_accounting_policy
                    ),
                    "action_budget_process_scope": (
                        self.codex_agent_execution_policy
                        .action_budget_process_scope
                    ),
                    "action_budget_limit": (
                        self.max_steps
                        if self.codex_agent_execution_policy.action_budget_enforced
                        else None
                    ),
                    "codex_agent_execution_policy": (
                        self.codex_agent_execution_policy.to_dict()
                    ),
                    "codex_agent_execution_policy_hash": (
                        self.codex_agent_execution_policy_hash
                    ),
                    "agent_setup": None,
                    "run_template_hash": stable_hash(
                        {"run_template": agent["run"]}
                    ),
                    "secret_value_persisted": False,
                },
            )
        )

    def _audit_trial_artifacts(
        self,
        *,
        runner: ModuleType,
        request: SkillLearnTrialRequest,
        skill_config: str,
        result: Mapping[str, Any],
        offline_verifier_profile: OfflineVerifierProfile | None,
        trace_id: str,
    ) -> Mapping[str, Any]:
        trials_root = self.trials_dir or Path(runner.TRIALS_DIR).expanduser().resolve()
        trial_path = (
            trials_root
            / skill_config
            / request.family
            / request.item_id
            / request.trial_id
        )
        test_script = (
            self.benchmark_root
            / "tasks"
            / request.family
            / request.item_id
            / "tests"
            / "test.sh"
        )
        verifier_receipt = _inspect_verifier_execution_receipt(
            test_script=test_script,
            verifier_dir=trial_path / "verifier",
            result=result,
            offline_verifier_profile=offline_verifier_profile,
        )
        codex_trace_path = trial_path / "agent" / "codex.txt"
        tool_audit = _inspect_codex_tool_policy(codex_trace_path)
        proposal_action_trace = (
            _extract_train_action_trace_profile(
                codex_trace_path,
                containment_root=trials_root,
            )
            if self.train_action_design_policy
            == TRAIN_ACTION_DESIGN_POLICY_VERSION
            and request.split is SplitName.TRAIN
            and request.variant is TrialVariant.POLICY_OFF
            else {}
        )
        trace_terminal_error = _provider_scoped_terminal_error(
            _codex_terminal_error_label(
                codex_trace_path.read_text(encoding="utf-8", errors="replace")
                if codex_trace_path.is_file()
                else None
            ),
            self.provider_mode,
        )
        action_budget_audit = (
            audit_codex_action_budget(
                trace_path=codex_trace_path,
                receipt_path=(
                    trial_path / "agent" / "codex_action_budget_receipt.json"
                ),
                supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
                expected_limit=self.max_steps,
                expected_process_scope=(
                    str(
                        self.codex_agent_execution_policy
                        .action_budget_process_scope
                    )
                ),
            )
            if self.codex_agent_execution_policy.action_budget_enforced
            else None
        )
        audited = dict(result)
        if (
            action_budget_audit is not None
            and action_budget_audit.valid
            and action_budget_audit.token_usage_complete
        ):
            audited["token_usage"] = dict(action_budget_audit.token_usage)
            audited["token_usage_source"] = "codex_action_budget_receipt"
        audited.update(
            {
                "verifier_receipt_policy": (
                    VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION
                ),
                "verifier_receipt_valid": verifier_receipt.valid,
                "verifier_receipt_kind": verifier_receipt.evidence_kind,
                "verifier_receipt_hash": verifier_receipt.receipt_hash,
                "verifier_receipt_test_count": verifier_receipt.test_count,
                "verifier_semantic_prelude_required": (
                    verifier_receipt.semantic_prelude_required
                ),
                "verifier_semantic_prelude_valid": (
                    verifier_receipt.semantic_prelude_valid
                ),
                "verifier_semantic_prelude_succeeded": (
                    verifier_receipt.semantic_prelude_succeeded
                ),
                "verifier_semantic_prelude_id": (
                    verifier_receipt.semantic_prelude_id
                ),
                "verifier_semantic_prelude_exit_code": (
                    verifier_receipt.semantic_prelude_exit_code
                ),
                "verifier_semantic_prelude_details": dict(
                    verifier_receipt.semantic_prelude_details
                ),
                "verifier_semantic_prelude_receipt_hash": (
                    verifier_receipt.semantic_prelude_receipt_hash
                ),
                "model_tool_policy": MODEL_ONLY_TOOL_POLICY_VERSION,
                "model_tool_policy_valid": tool_audit.valid,
                "model_remote_tool_call_count": (
                    tool_audit.remote_tool_call_count
                ),
                "model_runtime_install_command_count": (
                    tool_audit.runtime_install_command_count
                ),
                "model_tool_trace_hash": tool_audit.trace_hash,
                "model_terminal_error": trace_terminal_error,
                "model_terminal_trace_hash": (
                    tool_audit.trace_hash if trace_terminal_error else None
                ),
                "proposal_action_trace": proposal_action_trace,
                "steps_used": (
                    action_budget_audit.observed_steps
                    if action_budget_audit is not None
                    else result.get("steps_used")
                ),
                "step_budget_policy": (
                    CODEX_ACTION_BUDGET_POLICY_VERSION
                    if action_budget_audit is not None
                    else None
                ),
                "step_budget_unit": (
                    CODEX_ACTION_BUDGET_UNIT
                    if action_budget_audit is not None
                    else None
                ),
                "step_budget_limit": (
                    self.max_steps if action_budget_audit is not None else None
                ),
                "step_budget_truncated": (
                    action_budget_audit.budget_truncated
                    if action_budget_audit is not None
                    else False
                ),
                "step_budget_receipt_valid": (
                    action_budget_audit.valid
                    if action_budget_audit is not None
                    else None
                ),
                "step_budget_receipt_hash": (
                    action_budget_audit.receipt_hash
                    if action_budget_audit is not None
                    else None
                ),
                "step_budget_action_event_hash": (
                    action_budget_audit.action_event_hash
                    if action_budget_audit is not None
                    else None
                ),
                "step_budget_token_usage_complete": (
                    action_budget_audit.token_usage_complete
                    if action_budget_audit is not None
                    else None
                ),
            }
        )
        existing_error = _safe_error_label(audited.get("error"))
        if trace_terminal_error in _FATAL_PROVIDER_ERROR_TYPES:
            audited["error"] = trace_terminal_error
        elif existing_error is None:
            if trace_terminal_error:
                audited["error"] = trace_terminal_error
            elif not tool_audit.valid:
                audited["error"] = tool_audit.error_type
            elif action_budget_audit is not None and not action_budget_audit.valid:
                audited["error"] = action_budget_audit.error_type
            elif not verifier_receipt.valid:
                audited["error"] = verifier_receipt.error_type
        self.event_sink.emit(
            Event(
                event=(
                    "skilllearn_verifier_execution_receipt_validated"
                    if verifier_receipt.valid
                    else "skilllearn_verifier_execution_receipt_invalid"
                ),
                stage="benchmark.skilllearn.verifier_receipt",
                trace_id=trace_id,
                payload={
                    "policy": VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION,
                    "request_hash": request.request_hash,
                    "valid": verifier_receipt.valid,
                    "error_type": verifier_receipt.error_type,
                    "evidence_kind": verifier_receipt.evidence_kind,
                    "reward": verifier_receipt.reward,
                    "test_count": verifier_receipt.test_count,
                    "semantic_prelude_required": (
                        verifier_receipt.semantic_prelude_required
                    ),
                    "semantic_prelude_valid": (
                        verifier_receipt.semantic_prelude_valid
                    ),
                    "semantic_prelude_succeeded": (
                        verifier_receipt.semantic_prelude_succeeded
                    ),
                    "semantic_prelude_id": verifier_receipt.semantic_prelude_id,
                    "semantic_prelude_exit_code": (
                        verifier_receipt.semantic_prelude_exit_code
                    ),
                    "semantic_prelude_details": dict(
                        verifier_receipt.semantic_prelude_details
                    ),
                    "semantic_prelude_receipt_hash": (
                        verifier_receipt.semantic_prelude_receipt_hash
                    ),
                    "receipt_hash": verifier_receipt.receipt_hash,
                    "raw_content_persisted": False,
                },
            )
        )
        self.event_sink.emit(
            Event(
                event=(
                    "skilllearn_model_only_tool_policy_validated"
                    if tool_audit.valid
                    else "skilllearn_model_only_tool_policy_violated"
                ),
                stage="benchmark.skilllearn.model_tool_policy",
                trace_id=trace_id,
                payload={
                    "policy": MODEL_ONLY_TOOL_POLICY_VERSION,
                    "request_hash": request.request_hash,
                    "valid": tool_audit.valid,
                    "error_type": tool_audit.error_type,
                    "remote_tool_call_count": tool_audit.remote_tool_call_count,
                    "runtime_install_command_count": (
                        tool_audit.runtime_install_command_count
                    ),
                    "trace_hash": tool_audit.trace_hash,
                    "raw_content_persisted": False,
                },
            )
        )
        if trace_terminal_error:
            self.event_sink.emit(
                Event(
                    event="skilllearn_agent_terminal_error_detected",
                    stage="benchmark.skilllearn.provider_failure",
                    trace_id=trace_id,
                    payload={
                        "error_type": trace_terminal_error,
                        "policy": PROVIDER_FAILURE_POLICY_VERSION,
                        "trace_hash": tool_audit.trace_hash,
                        "source": "complete_codex_trace",
                        "raw_content_persisted": False,
                    },
                )
            )
        if action_budget_audit is not None:
            self.event_sink.emit(
                Event(
                    event=(
                        "skilllearn_agent_action_budget_validated"
                        if action_budget_audit.valid
                        else "skilllearn_agent_action_budget_invalid"
                    ),
                    stage="benchmark.skilllearn.action_budget",
                    trace_id=trace_id,
                    payload={
                        "policy": CODEX_ACTION_BUDGET_POLICY_VERSION,
                        "unit": CODEX_ACTION_BUDGET_UNIT,
                        "request_hash": request.request_hash,
                        "valid": action_budget_audit.valid,
                        "error_type": action_budget_audit.error_type,
                        "limit": self.max_steps,
                        "observed_steps": action_budget_audit.observed_steps,
                        "budget_reached": action_budget_audit.budget_reached,
                        "budget_truncated": (
                            action_budget_audit.budget_truncated
                        ),
                        "cost_accounting_policy": (
                            CODEX_ACTION_BUDGET_COST_ACCOUNTING_POLICY
                        ),
                        "token_usage_complete": (
                            action_budget_audit.token_usage_complete
                        ),
                        "turn_completed_observed": (
                            action_budget_audit.turn_completed_observed
                        ),
                        "agent_processes_exit_confirmed": (
                            action_budget_audit.agent_processes_exit_confirmed
                        ),
                        "receipt_hash": action_budget_audit.receipt_hash,
                        "action_event_hash": action_budget_audit.action_event_hash,
                        "verifier_started_after_agent_exit": (
                            action_budget_audit.process_group_exit_confirmed
                            and action_budget_audit.agent_processes_exit_confirmed
                            and result.get("verifier_exit") is not None
                        ),
                        "raw_content_persisted": False,
                    },
                )
            )
        return audited

    def _sanitize_result(
        self,
        request: SkillLearnTrialRequest,
        *,
        result: Mapping[str, Any],
        return_code: int,
        duration_seconds: float,
        prebuilt_image: SkillLearnPrebuiltImage | None = None,
        offline_verifier_runtime: OfflineVerifierRuntime | None = None,
    ) -> SkillLearnTrialObservation:
        usage = result.get("token_usage") if isinstance(result.get("token_usage"), Mapping) else {}
        total_tokens = _as_nonnegative_int(usage.get("total_tokens"))
        if not total_tokens:
            total_tokens = _as_nonnegative_int(usage.get("input_tokens")) + _as_nonnegative_int(
                usage.get("output_tokens")
            )
        steps = _as_nonnegative_int(result.get("steps_used"))
        error_type = _safe_error_label(result.get("error"))
        terminal_error = _codex_terminal_error_label(
            result.get("agent_stdout"),
            result.get("agent_stderr"),
        )
        terminal_error = _provider_scoped_terminal_error(
            terminal_error,
            self.provider_mode,
        )
        if terminal_error:
            error_type = error_type or terminal_error
        if result.get("agent_timed_out") is True:
            error_type = error_type or "agent_timeout"
        agent_exit = result.get("agent_exit")
        if (
            agent_exit is not None
            and str(agent_exit).strip() not in {"", "0"}
        ):
            error_type = error_type or "codex_agent_exit_nonzero"
        if str(result.get("verifier_exit")).strip() == "-1":
            error_type = error_type or "verifier_timeout"
        if return_code not in {0, 1} and not error_type:
            error_type = f"upstream_return_code_{return_code}"
        success = bool(result.get("passed") is True) and error_type is None
        reward = result.get("reward")
        score = float(reward) if isinstance(reward, (int, float)) else float(success)
        score = max(0.0, min(1.0, score))
        upstream_metrics = (
            result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
        )
        metrics = {
            str(key): float(value)
            for key, value in upstream_metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        metrics["task_success"] = float(success)
        metrics["evaluation_valid"] = float(error_type is None)
        provider_fingerprint = _provider_fingerprint(
            self.agent_id,
            self.model,
            self.provider_mode,
            self.codex_agent_execution_policy,
        )
        fairness_fingerprint = _fairness_fingerprint(
            agent_id=self.agent_id,
            model=self.model,
            provider_mode=self.provider_mode,
            max_steps=self.max_steps,
            provider_fingerprint=provider_fingerprint,
            prebuilt_enabled=self.prebuilt_cache is not None,
            agent_runtime_key=(
                prebuilt_image.agent_runtime_key if prebuilt_image else ""
            ),
            prebuilt_image_key=prebuilt_image.cache_key if prebuilt_image else "",
            prebuilt_image_id=prebuilt_image.image_id if prebuilt_image else "",
            offline_verifier_runtime_key=(
                offline_verifier_runtime.runtime_key
                if offline_verifier_runtime
                else ""
            ),
            codex_agent_execution_policy=self.codex_agent_execution_policy,
            model_inference_concurrency_policy=(
                self.model_inference_limiter.policy
                if self.model_inference_limiter is not None
                else None
            ),
            model_inference_slots=(
                self.model_inference_limiter.slots
                if self.model_inference_limiter is not None
                else 0
            ),
        )
        sanitized_upstream = {
            "task_id_hash": stable_hash({"task_id": result.get("task_id")}),
            "trial_id_hash": stable_hash({"trial_id": result.get("trial_id") or result.get("trial_name")}),
            "agent": result.get("agent"),
            "model": result.get("model"),
            "provider_mode": self.provider_mode,
            "verifier_isolation": VERIFIER_ISOLATION_VERSION,
            "verifier_execution_receipt_policy": (
                VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION
            ),
            "runner_agent_registry_isolation": (
                RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
            ),
            "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
            "provider_failure_policy": PROVIDER_FAILURE_POLICY_VERSION,
            "openai_compatible_codex_config": (
                OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION
                if self.provider_mode == "openai_compatible"
                else None
            ),
            "codex_network_minimization": (
                codex_network_minimization_for_policy(
                    self.codex_agent_execution_policy
                )
            ),
            "codex_agent_execution_policy": (
                self.codex_agent_execution_policy.to_dict()
            ),
            "codex_agent_execution_policy_hash": (
                self.codex_agent_execution_policy_hash
            ),
            "model_only_tool_policy": MODEL_ONLY_TOOL_POLICY_VERSION,
            "container_egress_policy": DOCKER_EGRESS_POLICY_VERSION,
            "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
            "provider_dns_policy": PROVIDER_DNS_POLICY_VERSION,
            "trial_network_budget_policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
            "trial_network_byte_limit": self.trial_network_byte_limit,
            "prebuilt_policy": (
                PREBUILT_IMAGE_POLICY_VERSION if self.prebuilt_cache else "disabled"
            ),
            "prebuilt_image_key": prebuilt_image.cache_key if prebuilt_image else None,
            "prebuilt_image_id": prebuilt_image.image_id if prebuilt_image else None,
            "agent_runtime_key": (
                prebuilt_image.agent_runtime_key if prebuilt_image else None
            ),
            "agent_runtime_version": (
                prebuilt_image.agent_runtime_version if prebuilt_image else None
            ),
            "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
            "offline_verifier_profile_id": (
                offline_verifier_runtime.profile.profile_id
                if offline_verifier_runtime
                else None
            ),
            "offline_verifier_runtime_key": (
                offline_verifier_runtime.runtime_key
                if offline_verifier_runtime
                else None
            ),
            "skill_config": result.get("skill_config"),
            "passed": result.get("passed"),
            "reward": result.get("reward"),
            "agent_exit": result.get("agent_exit"),
            "agent_timed_out": result.get("agent_timed_out"),
            "verifier_exit": result.get("verifier_exit"),
            "verifier_receipt_valid": result.get("verifier_receipt_valid"),
            "verifier_receipt_kind": result.get("verifier_receipt_kind"),
            "verifier_receipt_hash": result.get("verifier_receipt_hash"),
            "verifier_receipt_test_count": _as_nonnegative_int(
                result.get("verifier_receipt_test_count")
            ),
            "verifier_semantic_prelude_required": result.get(
                "verifier_semantic_prelude_required"
            ),
            "verifier_semantic_prelude_valid": result.get(
                "verifier_semantic_prelude_valid"
            ),
            "verifier_semantic_prelude_succeeded": result.get(
                "verifier_semantic_prelude_succeeded"
            ),
            "verifier_semantic_prelude_id": result.get(
                "verifier_semantic_prelude_id"
            ),
            "verifier_semantic_prelude_exit_code": result.get(
                "verifier_semantic_prelude_exit_code"
            ),
            "verifier_semantic_prelude_details_hash": stable_hash(
                result.get("verifier_semantic_prelude_details")
                if isinstance(
                    result.get("verifier_semantic_prelude_details"), Mapping
                )
                else {}
            ),
            "verifier_semantic_prelude_receipt_hash": result.get(
                "verifier_semantic_prelude_receipt_hash"
            ),
            "model_tool_policy_valid": result.get("model_tool_policy_valid"),
            "model_remote_tool_call_count": _as_nonnegative_int(
                result.get("model_remote_tool_call_count")
            ),
            "model_runtime_install_command_count": _as_nonnegative_int(
                result.get("model_runtime_install_command_count")
            ),
            "model_tool_trace_hash": result.get("model_tool_trace_hash"),
            "model_terminal_error": result.get("model_terminal_error"),
            "model_terminal_trace_hash": result.get(
                "model_terminal_trace_hash"
            ),
            "step_budget_policy": result.get("step_budget_policy"),
            "step_budget_unit": result.get("step_budget_unit"),
            "step_budget_limit": _as_nonnegative_int(
                result.get("step_budget_limit")
            ),
            "step_budget_truncated": result.get("step_budget_truncated"),
            "step_budget_token_usage_complete": result.get(
                "step_budget_token_usage_complete"
            ),
            "step_budget_receipt_valid": result.get(
                "step_budget_receipt_valid"
            ),
            "step_budget_receipt_hash": result.get(
                "step_budget_receipt_hash"
            ),
            "step_budget_action_event_hash": result.get(
                "step_budget_action_event_hash"
            ),
            "error_type": error_type,
            "token_usage": {str(key): _as_nonnegative_int(value) for key, value in usage.items()},
        }
        return SkillLearnTrialObservation(
            request=request,
            success=success,
            score=score,
            metrics=metrics,
            total_tokens=total_tokens,
            steps=steps,
            duration_seconds=round(duration_seconds, 6),
            provider_fingerprint=provider_fingerprint,
            fairness_fingerprint=fairness_fingerprint,
            error_type=error_type,
            upstream_result_hash=stable_hash(sanitized_upstream),
            raw_trial_artifacts_persisted=self.record_upstream,
            prebuilt_image_key=prebuilt_image.cache_key if prebuilt_image else "",
            prebuilt_image_id=prebuilt_image.image_id if prebuilt_image else "",
            prebuilt_cache_reused=bool(prebuilt_image and prebuilt_image.reused),
            agent_runtime_key=prebuilt_image.agent_runtime_key if prebuilt_image else "",
            agent_runtime_version=(
                prebuilt_image.agent_runtime_version if prebuilt_image else ""
            ),
            offline_verifier_profile_id=(
                offline_verifier_runtime.profile.profile_id
                if offline_verifier_runtime
                else ""
            ),
            offline_verifier_runtime_key=(
                offline_verifier_runtime.runtime_key
                if offline_verifier_runtime
                else ""
            ),
            step_budget_policy=str(result.get("step_budget_policy") or ""),
            step_budget_unit=str(result.get("step_budget_unit") or ""),
            step_budget_limit=_as_nonnegative_int(
                result.get("step_budget_limit")
            ),
            step_budget_truncated=bool(result.get("step_budget_truncated")),
            step_budget_token_usage_complete=bool(
                result.get("step_budget_token_usage_complete")
            ),
            step_budget_receipt_hash=str(
                result.get("step_budget_receipt_hash") or ""
            ),
            proposal_action_trace=(
                MappingProxyType(dict(result["proposal_action_trace"]))
                if isinstance(result.get("proposal_action_trace"), Mapping)
                else MappingProxyType({})
            ),
        )

    def _local_error(self, request: SkillLearnTrialRequest, error_type: str) -> SkillLearnTrialObservation:
        provider = _provider_fingerprint(
            self.agent_id,
            self.model,
            self.provider_mode,
            self.codex_agent_execution_policy,
        )
        fairness = _fairness_fingerprint(
            agent_id=self.agent_id,
            model=self.model,
            provider_mode=self.provider_mode,
            max_steps=self.max_steps,
            provider_fingerprint=provider,
            prebuilt_enabled=self.prebuilt_cache is not None,
            agent_runtime_key="",
            prebuilt_image_key="",
            prebuilt_image_id="",
            offline_verifier_runtime_key="",
            codex_agent_execution_policy=self.codex_agent_execution_policy,
            model_inference_concurrency_policy=(
                self.model_inference_limiter.policy
                if self.model_inference_limiter is not None
                else None
            ),
            model_inference_slots=(
                self.model_inference_limiter.slots
                if self.model_inference_limiter is not None
                else 0
            ),
        )
        return SkillLearnTrialObservation(
            request=request,
            success=False,
            score=0.0,
            metrics={"task_success": 0.0, "evaluation_valid": 0.0},
            total_tokens=0,
            steps=0,
            duration_seconds=0.0,
            provider_fingerprint=provider,
            fairness_fingerprint=fairness,
            error_type=error_type,
        )


class SkillLearnResidualMiner:
    """Turn failed training trials into no-gold semantic residuals."""

    def __init__(
        self,
        *,
        adapter: SkillLearnBenchAdapter,
        manifest: SplitManifest,
        guard: SplitAccessGuard,
        contrastive_training_evidence_policy: str | None = None,
        train_action_design_policy: str | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        if contrastive_training_evidence_policy not in {
            None,
            *CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSIONS,
        }:
            raise ValueError(
                "unsupported contrastive training evidence policy: "
                f"{contrastive_training_evidence_policy}"
            )
        if train_action_design_policy not in {
            None,
            *TRAIN_ACTION_DESIGN_POLICY_VERSIONS,
        }:
            raise ValueError(
                "unsupported TRAIN action design policy: "
                f"{train_action_design_policy}"
            )
        self.adapter = adapter
        self.manifest = manifest
        self.guard = guard
        self.contrastive_training_evidence_policy = (
            contrastive_training_evidence_policy
        )
        self.train_action_design_policy = train_action_design_policy
        self.event_sink = event_sink or NullEventSink()
        self.items = {item.id: item for item in adapter.discover()}

    def mine(
        self,
        observations: Sequence[SkillLearnTrialObservation],
        *,
        trace_id: str = "skilllearn_residual_mining",
    ) -> tuple[ResidualExample, ...]:
        residuals: list[ResidualExample] = []
        skipped_infrastructure = 0
        for observation in observations:
            request = observation.request
            if request.variant is not TrialVariant.POLICY_OFF or request.split is not SplitName.TRAIN:
                raise PermissionError("residual mining accepts policy-off training observations only")
            self.guard.authorize(request.item_id, AccessPhase.PROPOSAL)
            if not observation.valid:
                skipped_infrastructure += 1
                continue
            if (
                observation.success
                and self.contrastive_training_evidence_policy is None
            ):
                continue
            item = self.items[request.item_id]
            if observation.success:
                failure_type = "baseline_success_control"
                feedback: tuple[str, ...] = ()
                context: Mapping[str, Any] = {}
            else:
                instruction = self.adapter.load_instruction(
                    request.item_id,
                    phase=AccessPhase.PROPOSAL,
                    guard=self.guard,
                ).strip()
                failure_type, feedback = _classify_training_failure(
                    observation,
                    actionable_feedback=(
                        self.contrastive_training_evidence_policy
                        == ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
                    ),
                )
                context = {
                    "task_instruction": instruction,
                    "observed_metrics": dict(sorted(observation.metrics.items())),
                    "execution_signals": {
                        "total_tokens": observation.total_tokens,
                        "steps": observation.steps,
                        "duration_seconds": observation.duration_seconds,
                    },
                }
                if (
                    self.train_action_design_policy
                    == TRAIN_ACTION_DESIGN_POLICY_VERSION
                ):
                    action_profile = {
                        "policy": self.train_action_design_policy,
                        "runtime_environment": (
                            self.adapter.load_action_design_context(
                                request.item_id,
                                phase=AccessPhase.PROPOSAL,
                                guard=self.guard,
                            )
                        ),
                        "baseline_action_trace": dict(
                            observation.proposal_action_trace
                        ),
                        "evidence_scope": "train_policy_off_nonoracle_only",
                        "validation_outcomes_used": False,
                        "verifier_content_used": False,
                        "test_content_used": False,
                    }
                    action_profile_hash = stable_hash(action_profile)
                    context = {
                        **context,
                        "action_context_profile_hash": action_profile_hash,
                        TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY: action_profile,
                    }
            residual = ResidualExample(
                transition_id=f"transition_{stable_hash({'request': request.request_hash, 'outcome': observation.observation_hash})[:18]}",
                task_id=request.item_id,
                family=request.family,
                split=SplitName.TRAIN,
                features={**dict(item.features), "family": item.family},
                failure_type=failure_type,
                evaluator_feedback=feedback,
                baseline_success=observation.success,
                context=context,
            )
            issues = residual.validate()
            if issues:
                raise PermissionError(f"training residual failed isolation checks: {issues}")
            residuals.append(residual)
        failure_rows = tuple(row for row in residuals if not row.baseline_success)
        success_controls = tuple(row for row in residuals if row.baseline_success)
        self.event_sink.emit(
            Event(
                event="skilllearn_training_residuals_mined",
                stage="benchmark.skilllearn.residuals",
                trace_id=trace_id,
                payload={
                    "observation_count": len(observations),
                    "residual_count": len(failure_rows),
                    "success_control_count": len(success_controls),
                    "example_count": len(residuals),
                    "contrastive_training_evidence_policy": (
                        self.contrastive_training_evidence_policy
                    ),
                    "train_action_design_policy": self.train_action_design_policy,
                    "infrastructure_rows_skipped": skipped_infrastructure,
                    "transition_set_hash": stable_hash(
                        {"transition_ids": sorted(row.transition_id for row in residuals)}
                    ),
                    "failure_type_counts": _count_values(row.failure_type for row in residuals),
                    "family_count": len({row.family for row in residuals}),
                    "residual_manifest": [
                        {
                            "transition_id": row.transition_id,
                            "task_id_hash": stable_hash({"task_id": row.task_id}),
                            "family_hash": stable_hash({"family": row.family}),
                            "failure_type": row.failure_type,
                            "baseline_success": row.baseline_success,
                            "feature_hash": stable_hash(dict(row.features)),
                            "context_hash": stable_hash(dict(row.context)),
                        }
                        for row in residuals
                    ],
                    "source_split": "train",
                    "test_content_accessed": self.guard.test_accessed,
                    "raw_content_persisted": False,
                },
            )
        )
        return tuple(residuals)


class SkillLearnExternalEvaluator:
    id = "skilllearn_external_task_verifier"

    def __init__(self, epoch: str) -> None:
        self.epoch = epoch

    def evaluate(self, task: TaskInput, execution: RuntimeExecution) -> ExternalOutcome:
        metadata = execution.selected_result.metadata
        valid = bool(metadata.get("evaluation_valid", False))
        success = bool(metadata.get("success", False)) and valid
        score = float(metadata.get("score", float(success))) if valid else 0.0
        raw_metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), Mapping) else {}
        metrics = {
            str(key): float(value)
            for key, value in raw_metrics.items()
            if isinstance(value, (int, float))
        }
        metrics["evaluation_valid"] = float(valid)
        metrics["task_success"] = float(success)
        return ExternalOutcome(
            task_id=task.id,
            success=success,
            score=score,
            evaluator_id=self.id,
            evaluator_epoch=self.epoch,
            metrics=metrics,
        )


@dataclass(frozen=True)
class _ExternalRuntimeDescriptor:
    runtime_version: str = "skilllearn_external_runtime_v2"


def _canonical_program_bundle(
    programs: Sequence[HypothesisProgram | Mapping[str, Any]],
    *,
    label: str,
) -> tuple[HypothesisProgram, ...]:
    """Normalize a program bundle to a unique, ID-sorted immutable sequence."""

    by_id: dict[str, HypothesisProgram] = {}
    for raw_program in programs:
        program = (
            raw_program
            if isinstance(raw_program, HypothesisProgram)
            else HypothesisProgram.from_dict(raw_program)
        )
        previous = by_id.get(program.id)
        if previous is not None:
            qualifier = (
                "conflicting "
                if previous.payload_hash != program.payload_hash
                else "duplicate "
            )
            raise ValueError(f"{label} contains {qualifier}program ID: {program.id}")
        by_id[program.id] = program
    return tuple(by_id[program_id] for program_id in sorted(by_id))


class SkillLearnCounterfactualRunner:
    """Run matched policy-off/on SkillLearnBench trials under one frozen epoch."""

    def __init__(
        self,
        *,
        adapter: SkillLearnBenchAdapter,
        manifest: SplitManifest,
        guard: SplitAccessGuard,
        backend: SkillLearnTrialBackend,
        evaluator: SkillLearnExternalEvaluator,
        compiler: SkillLearnProgramCompiler,
        output_root: str | Path,
        parallel_workers: int = 1,
        invalid_trial_max_attempts: int = 1,
        invalid_trial_retry_backoff_seconds: float = 0.0,
        invalid_trial_retry_workers: int = 1,
        baseline_arm_replay_cache: BaselineArmEvidenceReplayCache | None = None,
        baseline_arm_evidence_replay_policy: str = (
            BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
        event_sink: EventSink | None = None,
    ) -> None:
        if parallel_workers <= 0:
            raise ValueError("counterfactual worker count must be positive")
        if invalid_trial_max_attempts <= 0:
            raise ValueError("invalid trial maximum attempts must be positive")
        if invalid_trial_retry_backoff_seconds < 0:
            raise ValueError("invalid trial retry backoff cannot be negative")
        if invalid_trial_retry_workers <= 0:
            raise ValueError("invalid trial retry worker count must be positive")
        if (
            baseline_arm_evidence_replay_policy
            not in BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS
        ):
            raise ValueError(
                "unsupported baseline arm replay policy: "
                f"{baseline_arm_evidence_replay_policy}"
            )
        if (
            baseline_arm_replay_cache is not None
            and baseline_arm_replay_cache.policy
            != baseline_arm_evidence_replay_policy
        ):
            raise ValueError(
                "baseline arm replay cache and runner policies must match"
            )
        self.adapter = adapter
        self.manifest = manifest
        self.guard = guard
        self.backend = backend
        backend_policy = codex_agent_execution_policy_for_backend(backend)
        self.evidence_execution_policy_hash = backend_policy.policy_hash
        self.evaluator = evaluator
        self.compiler = compiler
        self.output_root = Path(output_root)
        self.parallel_workers = parallel_workers
        self.invalid_trial_max_attempts = invalid_trial_max_attempts
        self.invalid_trial_retry_backoff_seconds = (
            invalid_trial_retry_backoff_seconds
        )
        self.invalid_trial_retry_workers = invalid_trial_retry_workers
        self._invalid_retry_semaphore = threading.Semaphore(
            invalid_trial_retry_workers
        )
        self.event_sink = event_sink or NullEventSink()
        self.items = {item.id: item for item in adapter.discover()}
        self.runtime = _ExternalRuntimeDescriptor()
        self.baseline_arm_evidence_replay_policy = (
            baseline_arm_evidence_replay_policy
        )
        self.baseline_arm_replay_cache = (
            baseline_arm_replay_cache
            if baseline_arm_replay_cache is not None
            else BaselineArmEvidenceReplayCache(
                policy=baseline_arm_evidence_replay_policy
            )
        )

    def bind_baseline_arm_replay_cache(
        self,
        cache: BaselineArmEvidenceReplayCache,
    ) -> None:
        """Bind an explicitly shared cache before a paired validation run."""

        if cache.policy != self.baseline_arm_evidence_replay_policy:
            raise ValueError(
                "baseline arm replay cache and runner policies must match"
            )
        self.baseline_arm_replay_cache = cache

    def invalid_trial_retry_descriptor(self) -> dict[str, Any]:
        """Return the execution identity of one terminal-invalid retry cohort."""

        return {
            "policy": INVALID_TRIAL_RETRY_POLICY_VERSION,
            "invalid_trial_max_attempts": self.invalid_trial_max_attempts,
            "invalid_trial_retry_backoff_seconds": (
                self.invalid_trial_retry_backoff_seconds
            ),
            "invalid_trial_retry_workers": self.invalid_trial_retry_workers,
        }

    @staticmethod
    def behavior_hash(program: HypothesisProgram) -> str:
        return skilllearn_program_treatment_hash(program)

    @staticmethod
    def behavior_set_hash(
        programs: Sequence[HypothesisProgram | Mapping[str, Any]],
    ) -> str:
        """Return the order-independent executable identity of a delta bundle."""

        return skilllearn_program_set_treatment_hash(
            _canonical_program_bundle(programs, label="candidate delta")
        )

    # Compatibility aliases for callers that name the program-set hash explicitly.
    bundle_behavior_hash = behavior_set_hash
    behavior_hash_bundle = behavior_set_hash

    def run(
        self,
        tasks: Sequence[TaskInput],
        *,
        program: HypothesisProgram,
        baseline_programs: Sequence[HypothesisProgram] = (),
        split: SplitName,
        trace_id: str = "skilllearn_counterfactual",
    ) -> tuple[CounterfactualPair, ...]:
        """Backward-compatible singleton challenger wrapper."""

        return self._run_bundle(
            tasks,
            programs=(program,),
            baseline_programs=baseline_programs,
            split=split,
            trace_id=trace_id,
            legacy_singleton=True,
        )

    def run_bundle(
        self,
        tasks: Sequence[TaskInput],
        *,
        programs: Sequence[HypothesisProgram | Mapping[str, Any]] | None = None,
        candidate_programs: Sequence[
            HypothesisProgram | Mapping[str, Any]
        ] | None = None,
        baseline_programs: Sequence[HypothesisProgram | Mapping[str, Any]] = (),
        split: SplitName,
        trace_id: str = "skilllearn_counterfactual_bundle",
    ) -> tuple[CounterfactualPair, ...]:
        """Evaluate one canonical candidate delta bundle in one paired run.

        ``candidate_programs`` is accepted as an API alias while the v3.13 core
        uses ``programs``.  The bundle is the delta relative to
        ``baseline_programs``; the policy-on treatment compiles their full union.
        """

        if programs is not None and candidate_programs is not None:
            raise TypeError("provide either programs or candidate_programs, not both")
        selected = programs if programs is not None else candidate_programs
        if selected is None:
            raise TypeError("run_bundle requires programs")
        return self._run_bundle(
            tasks,
            programs=selected,
            baseline_programs=baseline_programs,
            split=split,
            trace_id=trace_id,
            legacy_singleton=False,
        )

    def _run_bundle(
        self,
        tasks: Sequence[TaskInput],
        *,
        programs: Sequence[HypothesisProgram | Mapping[str, Any]],
        baseline_programs: Sequence[HypothesisProgram | Mapping[str, Any]],
        split: SplitName,
        trace_id: str,
        legacy_singleton: bool,
    ) -> tuple[CounterfactualPair, ...]:
        candidate_delta_programs = _canonical_program_bundle(
            programs,
            label="candidate delta",
        )
        canonical_baseline_programs = _canonical_program_bundle(
            baseline_programs,
            label="baseline",
        )
        if not candidate_delta_programs:
            raise ValueError("candidate delta bundle must not be empty")
        if any(
            row.status is not HypothesisStatus.PROMOTED
            for row in canonical_baseline_programs
        ):
            raise ValueError("baseline program bundle must contain promoted programs only")
        baseline_ids = {row.id for row in canonical_baseline_programs}
        overlapping_ids = sorted(
            row.id for row in candidate_delta_programs if row.id in baseline_ids
        )
        if overlapping_ids:
            raise ValueError(
                "candidate delta must not repeat baseline program IDs: "
                + ",".join(overlapping_ids)
            )
        if split not in {SplitName.VALIDATION, SplitName.TEST}:
            raise ValueError("external counterfactuals are restricted to validation or sealed test")
        all_programs = (*canonical_baseline_programs, *candidate_delta_programs)
        if any(row.evaluator_epoch != self.evaluator.epoch for row in all_programs):
            raise ValueError("program and SkillLearnBench evaluator epochs differ")
        if split is SplitName.TEST and any(
            row.status is not HypothesisStatus.PROMOTED
            for row in candidate_delta_programs
        ):
            raise PermissionError("sealed test requires a promoted, frozen hypothesis program")
        phase = AccessPhase.PROMOTION if split is SplitName.VALIDATION else AccessPhase.FINAL_REPORT
        for task in tasks:
            authorized = self.guard.authorize(task.id, phase)
            if authorized is not split:
                raise PermissionError("counterfactual task is in the wrong split")

        target_ids = tuple(task.id for task in tasks)
        target_hash = stable_hash({"item_ids": sorted(target_ids)})[:10]
        baseline_compile_result = None
        if canonical_baseline_programs:
            baseline_hash = skilllearn_program_set_treatment_hash(
                canonical_baseline_programs
            )[:12]
            baseline_compile_result = self.compiler.compile(
                programs=canonical_baseline_programs,
                items=tuple(self.items.values()),
                split_manifest=self.manifest,
                output_root=self.output_root,
                method_name=f"assumption-agent-v2-incumbent-{baseline_hash}-{split.value}-{target_hash}",
                allowed_statuses={HypothesisStatus.PROMOTED},
                target_item_ids=target_ids,
                target_split=split.value,
                trace_id=trace_id,
            )
        full_candidate_programs = _canonical_program_bundle(
            (*canonical_baseline_programs, *candidate_delta_programs),
            label="full candidate",
        )
        candidate_delta_program_set_hash = self.behavior_set_hash(
            candidate_delta_programs
        )
        candidate_program_set_hash = skilllearn_program_set_treatment_hash(
            full_candidate_programs
        )
        candidate_compile_result = self.compiler.compile(
            programs=full_candidate_programs,
            items=tuple(self.items.values()),
            split_manifest=self.manifest,
            output_root=self.output_root,
            method_name=f"assumption-agent-v2-challenger-{candidate_program_set_hash[:12]}-{split.value}-{target_hash}",
            allowed_statuses={
                HypothesisStatus.CANDIDATE,
                HypothesisStatus.SHADOW,
                HypothesisStatus.PROMOTED,
            },
            target_item_ids=target_ids,
            target_split=split.value,
            trace_id=trace_id,
        )

        def run_one(task: TaskInput) -> CounterfactualPair:
            return self._run_pair(
                task,
                candidate_delta_programs=candidate_delta_programs,
                baseline_programs=canonical_baseline_programs,
                full_candidate_programs=full_candidate_programs,
                candidate_delta_program_set_hash=(
                    candidate_delta_program_set_hash
                ),
                baseline_compile_result=baseline_compile_result,
                candidate_compile_result=candidate_compile_result,
                split=split,
                trace_id=trace_id,
                legacy_singleton=legacy_singleton,
            )

        return _ordered_parallel_map(tasks, run_one, self.parallel_workers)

    def _run_pair(
        self,
        task: TaskInput,
        *,
        candidate_delta_programs: Sequence[HypothesisProgram],
        baseline_programs: Sequence[HypothesisProgram],
        full_candidate_programs: Sequence[HypothesisProgram],
        candidate_delta_program_set_hash: str,
        baseline_compile_result: Any,
        candidate_compile_result: Any,
        split: SplitName,
        trace_id: str,
        legacy_singleton: bool,
    ) -> CounterfactualPair:
        task_features = {**dict(task.features), "family": task.family}
        matched_candidate_delta_programs = tuple(
            row for row in candidate_delta_programs if row.matches(task_features)
        )
        matched_candidate_hypothesis_ids = tuple(
            row.id for row in matched_candidate_delta_programs
        )
        selected_candidate_hypothesis_ids = tuple(
            row.id for row in candidate_delta_programs
        )
        trigger_matched = bool(matched_candidate_delta_programs)
        active_baseline_programs = tuple(
            row for row in baseline_programs if row.matches(task_features)
        )
        active_full_candidate_programs = (
            (*active_baseline_programs, *matched_candidate_delta_programs)
            if legacy_singleton
            else tuple(
                row
                for row in full_candidate_programs
                if row.matches(task_features)
            )
        )
        candidate_full_program_set_hash = candidate_compile_result.program_set_hash
        matched_candidate_program_set_hash = skilllearn_program_set_treatment_hash(
            matched_candidate_delta_programs
        )
        if legacy_singleton:
            pair_id_payload = {
                "trace_id": trace_id,
                "task_id": task.id,
                "challenger_treatment_hash": self.behavior_hash(
                    candidate_delta_programs[0]
                ),
                "split": split.value,
            }
        else:
            pair_id_payload = {
                "trace_id": trace_id,
                "task_id": task.id,
                "candidate_delta_program_set_hash": (
                    candidate_delta_program_set_hash
                ),
                "candidate_full_program_set_hash": (
                    candidate_full_program_set_hash
                ),
                "matched_candidate_hypothesis_ids": list(
                    matched_candidate_hypothesis_ids
                ),
                "matched_candidate_program_set_hash": (
                    matched_candidate_program_set_hash
                ),
                "split": split.value,
            }
        pair_id = stable_hash(pair_id_payload)[:20]
        baseline_skill_source = (
            baseline_compile_result.source_for(task.id)
            if baseline_compile_result
            else None
        )
        candidate_skill_source = (
            candidate_compile_result.source_for(task.id)
            if trigger_matched
            else None
        )
        baseline_program_set_hash = (
            baseline_compile_result.program_set_hash
            if baseline_compile_result
            else skilllearn_program_set_treatment_hash(())
        )
        baseline_treatment_hash = (
            baseline_compile_result.treatment_hash_for(task.id)
            if baseline_compile_result
            else NO_SKILL_TREATMENT_HASH
        )
        off_request = self._request(
            task,
            split,
            TrialVariant.POLICY_OFF,
            pair_id,
            None,
            program_set_hash=baseline_program_set_hash,
            treatment_hash=baseline_treatment_hash,
            candidate_delta_program_set_hash=(
                "" if legacy_singleton else candidate_delta_program_set_hash
            ),
            candidate_full_program_set_hash=(
                "" if legacy_singleton else candidate_full_program_set_hash
            ),
            matched_candidate_program_set_hash=(
                "" if legacy_singleton else matched_candidate_program_set_hash
            ),
            selected_candidate_hypothesis_ids=(
                () if legacy_singleton else selected_candidate_hypothesis_ids
            ),
            matched_candidate_hypothesis_ids=(
                () if legacy_singleton else matched_candidate_hypothesis_ids
            ),
        )
        on_request = self._request(
            task,
            split,
            TrialVariant.POLICY_ON,
            pair_id,
            (
                candidate_delta_programs[0].id
                if len(candidate_delta_programs) == 1
                else None
            ),
            program_set_hash=candidate_compile_result.program_set_hash,
            treatment_hash=candidate_compile_result.treatment_hash_for(task.id),
            candidate_delta_program_set_hash=(
                "" if legacy_singleton else candidate_delta_program_set_hash
            ),
            candidate_full_program_set_hash=(
                "" if legacy_singleton else candidate_full_program_set_hash
            ),
            matched_candidate_program_set_hash=(
                "" if legacy_singleton else matched_candidate_program_set_hash
            ),
            selected_candidate_hypothesis_ids=(
                () if legacy_singleton else selected_candidate_hypothesis_ids
            ),
            matched_candidate_hypothesis_ids=(
                () if legacy_singleton else matched_candidate_hypothesis_ids
            ),
        )
        def run_trial(
            request: SkillLearnTrialRequest,
            *,
            skill_source_dir: Path | None,
            arm: str,
        ) -> SkillLearnTrialObservation:
            return _run_invalid_only_trial(
                request=request,
                run_once=lambda attempt: self.backend.run(
                    request,
                    skill_source_dir=skill_source_dir,
                    trace_id=(
                        f"{trace_id}:{pair_id}:{arm}:attempt-{attempt}"
                    ),
                ),
                maximum_attempts=self.invalid_trial_max_attempts,
                backoff_seconds=self.invalid_trial_retry_backoff_seconds,
                retry_semaphore=self._invalid_retry_semaphore,
                event_sink=self.event_sink,
                trace_id=f"{trace_id}:{pair_id}:{arm}",
            )

        def run_baseline_trial() -> tuple[
            SkillLearnTrialObservation,
            str,
            bool,
            bool,
        ]:
            replay_key = self._baseline_arm_replay_key(
                task,
                split=split,
                baseline_programs=baseline_programs,
                baseline_treatment_hash=baseline_treatment_hash,
            )

            def replay_valid(
                record: BaselineArmEvidenceRecord,
                *,
                baseline_trial_executed: bool = False,
            ) -> tuple[SkillLearnTrialObservation, str, bool, bool]:
                source = record.observation
                replayed = replace(
                    source.as_variant(off_request),
                    raw_trial_artifacts_persisted=False,
                )
                self.event_sink.emit(
                    Event(
                        event="skilllearn_baseline_arm_evidence_replayed",
                        stage="benchmark.skilllearn.counterfactual",
                        trace_id=f"{trace_id}:{pair_id}:off",
                        payload={
                            "policy": self.baseline_arm_evidence_replay_policy,
                            "replay_key": replay_key,
                            "source_trace_id": record.source_trace_id,
                            "target_trace_id": f"{trace_id}:{pair_id}:off",
                            "source_request_hash": source.request.request_hash,
                            "target_request_hash": off_request.request_hash,
                            "source_observation_hash": source.observation_hash,
                            "source_evidence_hash": record.evidence_hash,
                            "behavior_identical": True,
                            "new_baseline_executions": 0,
                            "sealed_test_accessed": False,
                            "raw_content_persisted": False,
                        },
                    )
                )
                return (
                    replayed,
                    record.evidence_hash,
                    True,
                    baseline_trial_executed,
                )

            def replay_terminal_invalid(
                memo: BaselineArmTerminalInvalidMemo,
                *,
                baseline_trial_executed: bool = False,
            ) -> tuple[SkillLearnTrialObservation, str, bool, bool]:
                source = memo.observation
                replayed = replace(
                    source.as_variant(off_request),
                    raw_trial_artifacts_persisted=False,
                )
                self.event_sink.emit(
                    Event(
                        event=(
                            "skilllearn_baseline_arm_terminal_invalid_replayed"
                        ),
                        stage="benchmark.skilllearn.counterfactual",
                        trace_id=f"{trace_id}:{pair_id}:off",
                        payload={
                            "policy": self.baseline_arm_evidence_replay_policy,
                            "replay_key": replay_key,
                            "source_trace_id": memo.source_trace_id,
                            "target_trace_id": f"{trace_id}:{pair_id}:off",
                            "source_request_hash": source.request.request_hash,
                            "target_request_hash": off_request.request_hash,
                            "source_observation_hash": source.observation_hash,
                            "source_terminal_outcome_hash": (
                                memo.terminal_outcome_hash
                            ),
                            "error_type": memo.error_type,
                            "behavior_identical": True,
                            "terminal_for_replay_key": True,
                            "promotion_evidence": False,
                            "new_baseline_executions": 0,
                            "sealed_test_accessed": False,
                            "raw_content_persisted": False,
                        },
                    )
                )
                return (
                    replayed,
                    memo.terminal_outcome_hash,
                    True,
                    baseline_trial_executed,
                )

            def replay_entry(
                entry: BaselineArmReplayEntry,
                *,
                baseline_trial_executed: bool = False,
            ) -> tuple[SkillLearnTrialObservation, str, bool, bool]:
                if isinstance(entry, BaselineArmTerminalInvalidMemo):
                    return replay_terminal_invalid(
                        entry,
                        baseline_trial_executed=baseline_trial_executed,
                    )
                return replay_valid(
                    entry,
                    baseline_trial_executed=baseline_trial_executed,
                )

            if split is not SplitName.VALIDATION:
                observation = run_trial(
                    off_request,
                    skill_source_dir=baseline_skill_source,
                    arm="off",
                )
                return (
                    observation,
                    _baseline_arm_evidence_hash(observation),
                    False,
                    True,
                )

            cache = self.baseline_arm_replay_cache
            with cache.locked(replay_key):
                cached = cache.get(replay_key)
                if cached is not None:
                    return replay_entry(cached)
                observation = run_trial(
                    off_request,
                    skill_source_dir=baseline_skill_source,
                    arm="off",
                )
                source_trace_id = f"{trace_id}:{pair_id}:off"
                if not observation.valid:
                    if (
                        self.baseline_arm_evidence_replay_policy
                        in TERMINAL_INVALID_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS
                    ):
                        memoized = cache.memoize_terminal_invalid(
                            replay_key,
                            observation=observation,
                            source_trace_id=source_trace_id,
                        )
                        if memoized is None:
                            conflict = cache.get(replay_key)
                            self.event_sink.emit(
                                Event(
                                    event=(
                                        "skilllearn_baseline_arm_terminal_invalid_conflict_rejected"
                                    ),
                                    stage="benchmark.skilllearn.counterfactual",
                                    trace_id=source_trace_id,
                                    payload={
                                        "policy": (
                                            self.baseline_arm_evidence_replay_policy
                                        ),
                                        "replay_key": replay_key,
                                        "source_request_hash": (
                                            off_request.request_hash
                                        ),
                                        "source_observation_hash": (
                                            observation.observation_hash
                                        ),
                                        "error_type": observation.error_type,
                                        "terminal_for_replay_key": True,
                                        "promotion_evidence": False,
                                        "cache_mutated": False,
                                        "sealed_test_accessed": False,
                                        "raw_content_persisted": False,
                                    },
                                )
                            )
                            if conflict is not None:
                                return replay_entry(
                                    conflict,
                                    baseline_trial_executed=True,
                                )
                            return (
                                observation,
                                _baseline_arm_evidence_hash(observation),
                                False,
                                True,
                            )
                        self.event_sink.emit(
                            Event(
                                event=(
                                    "skilllearn_baseline_arm_terminal_invalid_memoized"
                                ),
                                stage="benchmark.skilllearn.counterfactual",
                                trace_id=source_trace_id,
                                payload={
                                    "policy": (
                                        self.baseline_arm_evidence_replay_policy
                                    ),
                                    "replay_key": replay_key,
                                    "source_trace_id": source_trace_id,
                                    "source_request_hash": (
                                        off_request.request_hash
                                    ),
                                    "source_observation_hash": (
                                        memoized.observation.observation_hash
                                    ),
                                    "source_terminal_outcome_hash": (
                                        memoized.terminal_outcome_hash
                                    ),
                                    "error_type": memoized.error_type,
                                    "terminal_for_replay_key": True,
                                    "same_request_retry_policy_complete": True,
                                    "promotion_evidence": False,
                                    "new_baseline_executions": 1,
                                    "sealed_test_accessed": False,
                                    "raw_content_persisted": False,
                                },
                            )
                        )
                        return (
                            memoized.observation,
                            memoized.terminal_outcome_hash,
                            False,
                            True,
                        )
                    self.event_sink.emit(
                        Event(
                            event=(
                                "skilllearn_baseline_arm_evidence_not_recorded_invalid"
                            ),
                            stage="benchmark.skilllearn.counterfactual",
                            trace_id=source_trace_id,
                            payload={
                                "policy": self.baseline_arm_evidence_replay_policy,
                                "replay_key": replay_key,
                                "source_request_hash": off_request.request_hash,
                                "source_observation_hash": (
                                    observation.observation_hash
                                ),
                                "new_baseline_executions": 1,
                                "sealed_test_accessed": False,
                                "raw_content_persisted": False,
                            },
                        )
                    )
                    return (
                        observation,
                        _baseline_arm_evidence_hash(observation),
                        False,
                        True,
                    )
                recorded = cache.record(
                    replay_key,
                    observation=observation,
                    source_trace_id=source_trace_id,
                )
                if recorded is None:
                    conflict = cache.get(replay_key)
                    self.event_sink.emit(
                        Event(
                            event=(
                                "skilllearn_baseline_arm_evidence_conflict_rejected"
                            ),
                            stage="benchmark.skilllearn.counterfactual",
                            trace_id=source_trace_id,
                            payload={
                                "policy": self.baseline_arm_evidence_replay_policy,
                                "replay_key": replay_key,
                                "source_request_hash": off_request.request_hash,
                                "source_observation_hash": (
                                    observation.observation_hash
                                ),
                                "cache_mutated": False,
                                "sealed_test_accessed": False,
                                "raw_content_persisted": False,
                            },
                        )
                    )
                    if conflict is not None:
                        return replay_entry(
                            conflict,
                            baseline_trial_executed=True,
                        )
                    return (
                        observation,
                        _baseline_arm_evidence_hash(observation),
                        False,
                        True,
                    )
                self.event_sink.emit(
                    Event(
                        event="skilllearn_baseline_arm_evidence_recorded",
                        stage="benchmark.skilllearn.counterfactual",
                        trace_id=source_trace_id,
                        payload={
                            "policy": self.baseline_arm_evidence_replay_policy,
                            "replay_key": replay_key,
                            "source_request_hash": off_request.request_hash,
                            "source_observation_hash": (
                                recorded.observation.observation_hash
                            ),
                            "source_evidence_hash": recorded.evidence_hash,
                            "new_baseline_executions": 1,
                            "sealed_test_accessed": False,
                            "raw_content_persisted": False,
                        },
                    )
                )
                return recorded.observation, recorded.evidence_hash, False, True

        run_on_first = False
        treatment_applied = False
        if trigger_matched and (
            candidate_skill_source is None or not candidate_skill_source.is_dir()
        ):
            (
                baseline_observation,
                baseline_evidence_hash,
                baseline_replayed,
                baseline_trial_executed,
            ) = run_baseline_trial()
            candidate_observation = _invalid_observation_like(
                on_request,
                baseline_observation,
                "compiled_candidate_skill_missing",
            )
        elif not trigger_matched:
            (
                baseline_observation,
                baseline_evidence_hash,
                baseline_replayed,
                baseline_trial_executed,
            ) = run_baseline_trial()
            candidate_observation = baseline_observation.as_variant(on_request)
        else:
            treatment_applied = True
            run_on_first = (
                int(stable_hash({"pair_id": pair_id, "order": "balanced"})[:8], 16) % 2 == 1
            )
            if run_on_first:
                candidate_observation = run_trial(
                    on_request,
                    skill_source_dir=candidate_skill_source,
                    arm="on",
                )
                (
                    baseline_observation,
                    baseline_evidence_hash,
                    baseline_replayed,
                    baseline_trial_executed,
                ) = run_baseline_trial()
            else:
                (
                    baseline_observation,
                    baseline_evidence_hash,
                    baseline_replayed,
                    baseline_trial_executed,
                ) = run_baseline_trial()
                candidate_observation = run_trial(
                    on_request,
                    skill_source_dir=candidate_skill_source,
                    arm="on",
                )

        baseline_treatment_applied = bool(
            active_baseline_programs
            and baseline_skill_source is not None
            and baseline_skill_source.is_dir()
        )
        baseline_execution = _execution_from_observation(
            baseline_observation,
            lane=BASELINE_LANE,
            active_programs=(
                active_baseline_programs if baseline_treatment_applied else ()
            ),
            action_activated=baseline_treatment_applied,
            baseline_preserved=True,
        )
        candidate_active_programs = (
            active_full_candidate_programs
            if treatment_applied
            else (
                active_baseline_programs
                if not trigger_matched and baseline_treatment_applied
                else ()
            )
        )
        candidate_execution = _execution_from_observation(
            candidate_observation,
            lane=CANDIDATE_LANE if trigger_matched else BASELINE_LANE,
            active_programs=candidate_active_programs,
            action_activated=treatment_applied,
            baseline_preserved=not trigger_matched,
        )
        pair = CounterfactualPair(
            task_id=task.id,
            split=split,
            evaluator_epoch=self.evaluator.epoch,
            baseline=baseline_execution,
            candidate=candidate_execution,
            baseline_outcome=self.evaluator.evaluate(task, baseline_execution),
            candidate_outcome=self.evaluator.evaluate(task, candidate_execution),
        )
        shared_baseline_replay = (
            baseline_replayed
            and self.baseline_arm_evidence_replay_policy
            in SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS
        )
        terminal_invalid_baseline_replay = (
            shared_baseline_replay
            and not baseline_observation.valid
            and self.baseline_arm_evidence_replay_policy
            in TERMINAL_INVALID_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS
        )
        if terminal_invalid_baseline_replay and not baseline_trial_executed:
            run_order = (
                "on_only_shared_baseline_terminal_invalid_replay"
                if treatment_applied
                else "baseline_alias_shared_terminal_invalid_replay"
            )
        elif shared_baseline_replay and not baseline_trial_executed:
            run_order = (
                "on_only_shared_baseline_replay"
                if treatment_applied
                else "baseline_alias_shared_replay"
            )
        elif shared_baseline_replay:
            if not treatment_applied:
                run_order = "baseline_only_with_shared_baseline_conflict_replay"
            else:
                run_order = (
                    "on_off_with_shared_baseline_conflict_replay"
                    if run_on_first
                    else "off_on_with_shared_baseline_conflict_replay"
                )
        else:
            run_order = (
                "on_off"
                if treatment_applied and run_on_first
                else "off_on"
            )
        self.event_sink.emit(
            Event(
                event="skilllearn_counterfactual_pair_completed",
                stage="benchmark.skilllearn.counterfactual",
                trace_id=f"{trace_id}:{pair_id}",
                payload={
                    "pair_id": pair_id,
                    "task_id_hash": stable_hash({"task_id": task.id}),
                    "split": split.value,
                    "hypothesis_id": selected_candidate_hypothesis_ids[0],
                    **(
                        {
                            "selected_candidate_hypothesis_ids": list(
                                selected_candidate_hypothesis_ids
                            ),
                            "matched_candidate_hypothesis_ids": list(
                                matched_candidate_hypothesis_ids
                            ),
                        }
                        if not legacy_singleton
                        else {}
                    ),
                    "baseline_hypothesis_ids": [
                        row.id
                        for row in (
                            active_baseline_programs
                            if baseline_treatment_applied
                            else ()
                        )
                    ],
                    "candidate_hypothesis_ids": [
                        row.id for row in candidate_active_programs
                    ],
                    "trigger_matched": trigger_matched,
                    "action_activated": treatment_applied,
                    "treatment_applied": treatment_applied,
                    "candidate_trial_executed": treatment_applied,
                    "fine_grained_action_receipt_available": False,
                    "skill_routing_version": SKILL_ROUTING_VERSION,
                    "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
                    "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
                    "baseline_program_set_hash": baseline_program_set_hash,
                    "candidate_program_set_hash": (
                        candidate_compile_result.program_set_hash
                    ),
                    **(
                        {
                            "candidate_full_program_set_hash": (
                                candidate_full_program_set_hash
                            ),
                            "candidate_delta_program_set_hash": (
                                candidate_delta_program_set_hash
                            ),
                            "matched_candidate_program_set_hash": (
                                matched_candidate_program_set_hash
                            ),
                        }
                        if not legacy_singleton
                        else {}
                    ),
                    "baseline_treatment_hash": baseline_treatment_hash,
                    "candidate_treatment_hash": (
                        candidate_compile_result.treatment_hash_for(task.id)
                    ),
                    "baseline_success": pair.baseline_outcome.success,
                    "candidate_success": pair.candidate_outcome.success,
                    "baseline_score": pair.baseline_outcome.score,
                    "candidate_score": pair.candidate_outcome.score,
                    "baseline_valid": bool(pair.baseline_outcome.metrics.get("evaluation_valid")),
                    "candidate_valid": bool(pair.candidate_outcome.metrics.get("evaluation_valid")),
                    "baseline_observation_hash": baseline_observation.observation_hash,
                    "baseline_evidence_hash": baseline_evidence_hash,
                    "baseline_replayed": baseline_replayed,
                    "baseline_trial_executed": baseline_trial_executed,
                    "baseline_terminal_invalid_memoized": bool(
                        split is SplitName.VALIDATION
                        and not baseline_observation.valid
                        and baseline_trial_executed
                        and not baseline_replayed
                        and self.baseline_arm_evidence_replay_policy
                        in TERMINAL_INVALID_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS
                    ),
                    "baseline_terminal_invalid_replayed": (
                        terminal_invalid_baseline_replay
                    ),
                    "baseline_promotion_evidence_eligible": (
                        baseline_observation.valid
                    ),
                    "candidate_observation_hash": candidate_observation.observation_hash,
                    "baseline_cost": baseline_observation.cost_units,
                    "candidate_cost": candidate_observation.cost_units,
                    "provider_matched": (
                        baseline_observation.provider_fingerprint
                        == candidate_observation.provider_fingerprint
                    ),
                    "budget_matched": (
                        baseline_observation.fairness_fingerprint
                        == candidate_observation.fairness_fingerprint
                    ),
                    "run_order": run_order,
                    "parallel_workers": self.parallel_workers,
                },
            )
        )
        return pair

    def _baseline_arm_replay_key(
        self,
        task: TaskInput,
        *,
        split: SplitName,
        baseline_programs: Sequence[HypothesisProgram],
        baseline_treatment_hash: str,
    ) -> str:
        provider_mode = str(
            getattr(self.backend, "provider_mode", "openai_compatible")
        )
        descriptor = {
            "policy": self.baseline_arm_evidence_replay_policy,
            "task_id": task.id,
            "family": task.family,
            "split": split.value,
            "evaluator_epoch": self.evaluator.epoch,
            "manifest_hash": self.manifest.manifest_hash,
            "agent_id": self.backend.agent_id,
            "model": self.backend.model,
            "max_steps": self.backend.max_steps,
            "provider_fingerprint": _provider_fingerprint(
                self.backend.agent_id,
                self.backend.model,
                provider_mode,
                codex_agent_execution_policy_for_backend(self.backend),
            ),
            "codex_agent_execution_policy_hash": (
                codex_agent_execution_policy_for_backend(
                    self.backend
                ).policy_hash
            ),
            "baseline_behavior_set_hash": (
                skilllearn_program_set_treatment_hash(baseline_programs)
            ),
            "runtime_version": self.runtime.runtime_version,
            "skill_routing_version": SKILL_ROUTING_VERSION,
            "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
            "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
            "agent_runtime_version": SHARED_CODEX_CLI_VERSION,
            "verifier_isolation": VERIFIER_ISOLATION_VERSION,
            "container_egress_policy": DOCKER_EGRESS_POLICY_VERSION,
            "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
            "trial_network_budget_policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
            "trial_network_byte_limit": configured_trial_network_byte_limit(),
        }
        if (
            self.baseline_arm_evidence_replay_policy
            in SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS
        ):
            descriptor["baseline_treatment_hash"] = baseline_treatment_hash
        if (
            self.baseline_arm_evidence_replay_policy
            in TERMINAL_INVALID_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSIONS
        ):
            descriptor["invalid_trial_retry"] = (
                self.invalid_trial_retry_descriptor()
            )
        return stable_hash(descriptor)

    def _request(
        self,
        task: TaskInput,
        split: SplitName,
        variant: TrialVariant,
        pair_id: str,
        program_id: str | None,
        *,
        program_set_hash: str,
        treatment_hash: str,
        candidate_delta_program_set_hash: str = "",
        candidate_full_program_set_hash: str = "",
        matched_candidate_program_set_hash: str = "",
        selected_candidate_hypothesis_ids: tuple[str, ...] = (),
        matched_candidate_hypothesis_ids: tuple[str, ...] = (),
    ) -> SkillLearnTrialRequest:
        return SkillLearnTrialRequest(
            item_id=task.id,
            family=task.family,
            split=split,
            variant=variant,
            evaluator_epoch=self.evaluator.epoch,
            pair_id=pair_id,
            repeat=1,
            agent_id=self.backend.agent_id,
            model=self.backend.model,
            max_steps=self.backend.max_steps,
            manifest_hash=self.manifest.manifest_hash,
            codex_agent_execution_policy_hash=(
                codex_agent_execution_policy_for_backend(self.backend).policy_hash
            ),
            program_id=program_id,
            program_set_hash=program_set_hash,
            treatment_hash=treatment_hash,
            candidate_delta_program_set_hash=(
                candidate_delta_program_set_hash
            ),
            candidate_full_program_set_hash=candidate_full_program_set_hash,
            matched_candidate_program_set_hash=(
                matched_candidate_program_set_hash
            ),
            selected_candidate_hypothesis_ids=(
                selected_candidate_hypothesis_ids
            ),
            matched_candidate_hypothesis_ids=(
                matched_candidate_hypothesis_ids
            ),
        )


@dataclass(frozen=True)
class SkillLearnGenerationResult:
    train_observations: tuple[SkillLearnTrialObservation, ...]
    residuals: tuple[ResidualExample, ...]
    evolution: EvolutionRunResult | None
    reason: str
    baseline_hypothesis_ids: tuple[str, ...] = ()
    proposal_model_failure_count: int = 0
    contrastive_training_evidence_policy: str | None = None
    candidate_bundle_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        decision = self.evolution.promotion_decision if self.evolution else None
        failure_count = sum(not row.baseline_success for row in self.residuals)
        success_control_count = sum(row.baseline_success for row in self.residuals)
        action_context_profile_hashes = sorted(
            {
                str(row.context.get("action_context_profile_hash") or "")
                for row in self.residuals
                if row.context.get("action_context_profile_hash")
            }
        )
        selected_candidate_hypothesis_ids: tuple[str, ...] = ()
        if self.evolution and decision:
            selected_candidate_hypothesis_ids = tuple(
                sorted(
                    set(
                        getattr(
                            self.evolution,
                            "selected_candidate_hypothesis_ids",
                            (),
                        )
                        or ()
                    )
                )
            )
            if (
                not selected_candidate_hypothesis_ids
                and self.evolution.accepted_hypothesis_id
            ):
                selected_candidate_hypothesis_ids = (
                    self.evolution.accepted_hypothesis_id,
                )
        return {
            "train_observation_count": len(self.train_observations),
            "valid_train_observation_count": sum(row.valid for row in self.train_observations),
            "training_residual_count": failure_count,
            "success_control_count": success_control_count,
            "example_count": len(self.residuals),
            **(
                {
                    "action_context_profile_count": len(
                        action_context_profile_hashes
                    ),
                    "action_context_profile_set_hash": stable_hash(
                        {"profile_hashes": action_context_profile_hashes}
                    ),
                }
                if action_context_profile_hashes
                else {}
            ),
            "contrastive_training_evidence_policy": (
                self.contrastive_training_evidence_policy
            ),
            "baseline_hypothesis_ids": list(self.baseline_hypothesis_ids),
            "evolution_trace_id": self.evolution.trace_id if self.evolution else None,
            "promoted": bool(self.evolution and self.evolution.promoted),
            "accepted_hypothesis_id": self.evolution.accepted_hypothesis_id if self.evolution else None,
            **(
                {
                    "selected_candidate_hypothesis_ids": list(
                        selected_candidate_hypothesis_ids
                    )
                }
                if self.candidate_bundle_policy == CANDIDATE_BUNDLE_POLICY_VERSION
                else {}
            ),
            "proposal_candidate_count": (
                self.evolution.proposal_candidate_count if self.evolution else 0
            ),
            "static_accepted_candidate_count": (
                self.evolution.static_accepted_candidate_count if self.evolution else 0
            ),
            "recursive_node_count": (
                self.evolution.static_validation_node_count if self.evolution else 0
            ),
            "recursive_depth": (
                self.evolution.static_validation_max_recursion_depth
                if self.evolution
                else 0
            ),
            "repaired_candidate_count": (
                self.evolution.repaired_candidate_count if self.evolution else 0
            ),
            "repair_model_failure_count": (
                self.evolution.repair_model_failure_count if self.evolution else 0
            ),
            "proposal_model_failure_count": (
                self.proposal_model_failure_count
                + (
                    self.evolution.repair_model_failure_count
                    if self.evolution
                    else 0
                )
            ),
            "selected_candidate_node_count": (
                len(self.evolution.validation_tree.nodes) if self.evolution else 0
            ),
            "selected_candidate_recursion_depth": (
                self.evolution.validation_tree.recursion_depth if self.evolution else 0
            ),
            "promotion_blockers": list(decision.blockers) if decision else [],
            "promotion_summary": decision.summary.to_dict(confidence=decision.confidence) if decision else None,
            "promotion_decision": decision.to_dict() if decision else None,
            "evaluated_candidate_treatment_hash": (
                self.evolution.evaluated_candidate_behavior_hash
                if self.evolution and decision
                else None
            ),
            "archive_node_hash": (
                self.evolution.archive_node.payload_hash
                if self.evolution and self.evolution.archive_node
                else None
            ),
            "reason": self.reason,
            "raw_content_persisted": False,
        }


class SkillLearnEvolutionHarness:
    """End-to-end train residual -> proposal -> shadow -> promotion benchmark harness."""

    def __init__(
        self,
        *,
        adapter: SkillLearnBenchAdapter,
        manifest: SplitManifest,
        guard: SplitAccessGuard,
        backend: SkillLearnTrialBackend,
        proposer: StructuredHypothesisProposer,
        validator: RecursiveValidationEngine,
        promotion_gate: PromotionGate,
        archive: PolicyArchive,
        evaluator_epoch: str,
        output_root: str | Path,
        proposal_candidates_per_generation: int = 3,
        candidate_selection_policy: str = TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
        candidate_bundle_policy: str | None = None,
        contrastive_training_evidence_policy: str | None = None,
        train_action_design_policy: str | None = None,
        repair_request_scope_policy: str | None = None,
        parallel_workers: int = 1,
        invalid_trial_max_attempts: int = 1,
        invalid_trial_retry_backoff_seconds: float = 0.0,
        invalid_trial_retry_workers: int = 1,
        baseline_arm_replay_cache: BaselineArmEvidenceReplayCache | None = None,
        baseline_arm_evidence_replay_policy: str = (
            BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
        event_sink: EventSink | None = None,
    ) -> None:
        if parallel_workers <= 0:
            raise ValueError("evolution worker count must be positive")
        if invalid_trial_max_attempts <= 0:
            raise ValueError("invalid trial maximum attempts must be positive")
        if invalid_trial_retry_backoff_seconds < 0:
            raise ValueError("invalid trial retry backoff cannot be negative")
        if invalid_trial_retry_workers <= 0:
            raise ValueError("invalid trial retry worker count must be positive")
        if contrastive_training_evidence_policy not in {
            None,
            *CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSIONS,
        }:
            raise ValueError(
                "unsupported contrastive training evidence policy: "
                f"{contrastive_training_evidence_policy}"
            )
        if train_action_design_policy not in {
            None,
            *TRAIN_ACTION_DESIGN_POLICY_VERSIONS,
        }:
            raise ValueError(
                "unsupported TRAIN action design policy: "
                f"{train_action_design_policy}"
            )
        if repair_request_scope_policy not in {
            None,
            REPAIR_REQUEST_SCOPE_POLICY_VERSION,
        }:
            raise ValueError(
                f"unsupported repair request scope policy: {repair_request_scope_policy}"
            )
        if candidate_selection_policy not in {
            TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
            CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
            PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
            *COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
        }:
            raise ValueError(
                f"unsupported candidate selection policy: {candidate_selection_policy}"
            )
        bundle_selection_enabled = (
            candidate_selection_policy
            in COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS
        )
        if bundle_selection_enabled:
            if candidate_bundle_policy != CANDIDATE_BUNDLE_POLICY_VERSION:
                raise ValueError(
                    "bundle candidate selection requires the v3.13 candidate bundle policy"
                )
        elif candidate_bundle_policy is not None:
            raise ValueError(
                "candidate bundle policy is only valid with bundle candidate selection"
            )
        if (
            contrastive_training_evidence_policy
            in CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSIONS
        ) != (
            candidate_selection_policy
            in {
                CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
                PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
                *COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
            }
        ):
            raise ValueError(
                "contrastive evidence and candidate selection policies must be paired"
            )
        self.adapter = adapter
        self.manifest = manifest
        self.guard = guard
        self.backend = backend
        self.proposer = proposer
        self.validator = validator
        self.promotion_gate = promotion_gate
        self.archive = archive
        self.evaluator_epoch = evaluator_epoch
        self.output_root = Path(output_root)
        self.parallel_workers = parallel_workers
        self.invalid_trial_max_attempts = invalid_trial_max_attempts
        self.invalid_trial_retry_backoff_seconds = (
            invalid_trial_retry_backoff_seconds
        )
        self.candidate_selection_policy = candidate_selection_policy
        self.candidate_bundle_policy = candidate_bundle_policy
        self.contrastive_training_evidence_policy = (
            contrastive_training_evidence_policy
        )
        self.train_action_design_policy = train_action_design_policy
        self.repair_request_scope_policy = repair_request_scope_policy
        self._invalid_retry_semaphore = threading.Semaphore(
            invalid_trial_retry_workers
        )
        self.event_sink = event_sink or NullEventSink()
        self.items = {item.id: item for item in adapter.discover()}
        self.residual_miner = SkillLearnResidualMiner(
            adapter=adapter,
            manifest=manifest,
            guard=guard,
            contrastive_training_evidence_policy=(
                contrastive_training_evidence_policy
            ),
            train_action_design_policy=train_action_design_policy,
            event_sink=self.event_sink,
        )
        self.compiler = SkillLearnProgramCompiler(event_sink=self.event_sink)
        self.counterfactual_runner = SkillLearnCounterfactualRunner(
            adapter=adapter,
            manifest=manifest,
            guard=guard,
            backend=backend,
            evaluator=SkillLearnExternalEvaluator(evaluator_epoch),
            compiler=self.compiler,
            output_root=self.output_root,
            parallel_workers=parallel_workers,
            invalid_trial_max_attempts=invalid_trial_max_attempts,
            invalid_trial_retry_backoff_seconds=(
                invalid_trial_retry_backoff_seconds
            ),
            invalid_trial_retry_workers=invalid_trial_retry_workers,
            baseline_arm_replay_cache=baseline_arm_replay_cache,
            baseline_arm_evidence_replay_policy=(
                baseline_arm_evidence_replay_policy
            ),
            event_sink=self.event_sink,
        )
        self.kernel = EvolutionKernel(
            proposer=proposer,
            validator=validator,
            counterfactual_runner=self.counterfactual_runner,
            promotion_gate=promotion_gate,
            archive=archive,
            split_guard=guard,
            proposal_candidates_per_generation=proposal_candidates_per_generation,
            candidate_selection_policy=candidate_selection_policy,
            candidate_bundle_policy=candidate_bundle_policy,
            contrastive_training_evidence_policy=(
                contrastive_training_evidence_policy
            ),
            train_action_design_policy=train_action_design_policy,
            repair_request_scope_policy=repair_request_scope_policy,
            event_sink=self.event_sink,
        )

    def run_generation(
        self,
        *,
        train_item_ids: Sequence[str] | None = None,
        validation_item_ids: Sequence[str] | None = None,
        training_replay_cache: TrainingEvidenceReplayCache | None = None,
        counterfactual_replay_cache: CounterfactualEvidenceReplayCache | None = None,
        trace_id: str = "skilllearn_evolution_generation",
    ) -> SkillLearnGenerationResult:
        train_ids = tuple(train_item_ids or self.manifest.train_ids)
        validation_ids = tuple(validation_item_ids or self.manifest.validation_ids)
        _require_subset(train_ids, self.manifest.train_ids, "training")
        _require_subset(validation_ids, self.manifest.validation_ids, "validation")
        observations = self.collect_training_observations(
            train_item_ids=train_ids,
            training_replay_cache=training_replay_cache,
            trace_id=trace_id,
        )
        residuals = self.residual_miner.mine(observations, trace_id=trace_id)
        return self.run_generation_from_evidence(
            observations=observations,
            residuals=residuals,
            validation_item_ids=validation_ids,
            counterfactual_replay_cache=counterfactual_replay_cache,
            trace_id=trace_id,
        )

    def collect_training_observations(
        self,
        *,
        train_item_ids: Sequence[str] | None = None,
        training_replay_cache: TrainingEvidenceReplayCache | None = None,
        trace_id: str = "skilllearn_training_observations",
    ) -> tuple[SkillLearnTrialObservation, ...]:
        train_ids = tuple(train_item_ids or self.manifest.train_ids)
        _require_subset(train_ids, self.manifest.train_ids, "training")
        incumbent_programs = self.incumbent_programs()
        incumbent_compile = self._compile_training_incumbent(
            incumbent_programs,
            train_ids=train_ids,
            trace_id=trace_id,
        )
        def run_one(item_id: str) -> SkillLearnTrialObservation:
            return self._run_training_baseline(
                item_id,
                skill_source_dir=(
                    incumbent_compile.source_for(item_id)
                    if incumbent_compile
                    else None
                ),
                program_set_hash=(
                    incumbent_compile.program_set_hash
                    if incumbent_compile
                    else skilllearn_program_set_treatment_hash(())
                ),
                treatment_hash=(
                    incumbent_compile.treatment_hash_for(item_id)
                    if incumbent_compile
                    else NO_SKILL_TREATMENT_HASH
                ),
                trace_id=trace_id,
            )

        def produce() -> tuple[SkillLearnTrialObservation, ...]:
            return _ordered_parallel_map(train_ids, run_one, self.parallel_workers)

        descriptor = self._training_replay_descriptor(
            train_ids=train_ids,
            incumbent_programs=incumbent_programs,
        )
        observations = (
            training_replay_cache.run_or_replay(
                descriptor=descriptor,
                train_item_ids=train_ids,
                producer=produce,
                trace_id=trace_id,
            )
            if training_replay_cache is not None
            else produce()
        )
        invalid = tuple(row for row in observations if not row.valid)
        if invalid:
            error_counts = _count_values(
                row.error_type or "unknown_infrastructure_error" for row in invalid
            )
            self.event_sink.emit(
                Event(
                    event="skilllearn_training_evidence_blocked",
                    stage="benchmark.skilllearn.training_evidence",
                    trace_id=trace_id,
                    payload={
                        "policy": TRAINING_EVIDENCE_POLICY_VERSION,
                        "observation_count": len(observations),
                        "invalid_observation_count": len(invalid),
                        "error_type_counts": error_counts,
                        "invalid_observation_set_hash": stable_hash(
                            {
                                "hashes": sorted(
                                    row.observation_hash for row in invalid
                                )
                            }
                        ),
                        "proposal_blocked": True,
                        "test_content_accessed": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            raise SkillLearnTrainingEvidenceError(
                "training evidence contains invalid observations: "
                + ",".join(f"{key}={value}" for key, value in error_counts.items())
            )
        return observations

    def run_generation_from_evidence(
        self,
        *,
        observations: Sequence[SkillLearnTrialObservation],
        residuals: Sequence[ResidualExample],
        validation_item_ids: Sequence[str] | None = None,
        proposal_candidates: Sequence[HypothesisProgram] | None = None,
        counterfactual_replay_cache: CounterfactualEvidenceReplayCache | None = None,
        trace_id: str = "skilllearn_evolution_from_evidence",
    ) -> SkillLearnGenerationResult:
        observations = tuple(observations)
        residuals = tuple(residuals)
        validation_ids = tuple(validation_item_ids or self.manifest.validation_ids)
        _require_subset(validation_ids, self.manifest.validation_ids, "validation")
        observation_ids = {row.request.item_id for row in observations}
        _require_subset(tuple(observation_ids), self.manifest.train_ids, "shared training evidence")
        if any(
            row.request.split is not SplitName.TRAIN
            or row.request.manifest_hash != self.manifest.manifest_hash
            or row.request.evaluator_epoch != self.evaluator_epoch
            for row in observations
        ):
            raise PermissionError("shared training observation identity mismatch")
        if any(not row.valid for row in observations):
            raise SkillLearnTrainingEvidenceError(
                "shared training evidence contains invalid observations"
            )
        if any(row.task_id not in observation_ids or row.split is not SplitName.TRAIN for row in residuals):
            raise PermissionError("shared residual is outside the training observation checkpoint")
        if self.contrastive_training_evidence_policy:
            observation_by_id = {
                row.request.item_id: row for row in observations
            }
            residual_by_id = {row.task_id: row for row in residuals}
            if (
                len(observation_by_id) != len(observations)
                or len(residual_by_id) != len(residuals)
                or set(residual_by_id) != set(observation_by_id)
            ):
                raise PermissionError(
                    "contrastive shared evidence must bind every valid training observation"
                )
            for item_id, observation in observation_by_id.items():
                example = residual_by_id[item_id]
                expected_transition_id = (
                    "transition_"
                    + stable_hash(
                        {
                            "request": observation.request.request_hash,
                            "outcome": observation.observation_hash,
                        }
                    )[:18]
                )
                if example.transition_id != expected_transition_id:
                    raise PermissionError(
                        "contrastive shared evidence transition identity mismatch"
                    )
                if example.baseline_success != observation.success:
                    raise PermissionError(
                        "contrastive shared evidence label does not match training outcome"
                    )
                if example.baseline_success and (
                    example.failure_type != "baseline_success_control"
                    or example.evaluator_feedback
                    or example.context
                ):
                    raise PermissionError(
                        "contrastive success control violates the no-context contract"
                    )
        for row in residuals:
            issues = row.validate()
            if issues:
                raise PermissionError(f"shared residual failed isolation checks: {issues}")
            self.guard.authorize(row.task_id, AccessPhase.PROPOSAL)
        incumbent_programs = self.incumbent_programs()
        if not any(not row.baseline_success for row in residuals):
            result = SkillLearnGenerationResult(
                train_observations=observations,
                residuals=residuals,
                evolution=None,
                reason="no_valid_failed_training_rows",
                baseline_hypothesis_ids=tuple(row.id for row in incumbent_programs),
                contrastive_training_evidence_policy=(
                    self.contrastive_training_evidence_policy
                ),
                candidate_bundle_policy=self.candidate_bundle_policy,
            )
            self._emit_generation_result(result, trace_id)
            return result
        validation_tasks = self.tasks(validation_ids)
        validation_context = self.validation_context(residuals)
        try:
            evolution = self.kernel.evolve_once(
                residuals=residuals,
                validation_tasks=validation_tasks,
                validation_context=validation_context,
                proposal_candidates=proposal_candidates,
                counterfactual_replay_cache=counterfactual_replay_cache,
                trace_id=trace_id,
            )
        except HypothesisProposalCallError as exc:
            return self.record_proposal_failure(
                observations=observations,
                residuals=residuals,
                error=exc,
                trace_id=trace_id,
            )
        result = SkillLearnGenerationResult(
            train_observations=observations,
            residuals=residuals,
            evolution=evolution,
            reason=evolution.reason,
            baseline_hypothesis_ids=tuple(row.id for row in incumbent_programs),
            contrastive_training_evidence_policy=(
                self.contrastive_training_evidence_policy
            ),
            candidate_bundle_policy=self.candidate_bundle_policy,
        )
        self._emit_generation_result(result, trace_id)
        return result

    def record_proposal_failure(
        self,
        *,
        observations: Sequence[SkillLearnTrialObservation],
        residuals: Sequence[ResidualExample],
        error: HypothesisProposalCallError,
        trace_id: str,
    ) -> SkillLearnGenerationResult:
        incumbent_programs = self.incumbent_programs()
        self.event_sink.emit(
            Event(
                event="skilllearn_generation_stopped_after_proposal_model_failure",
                stage="benchmark.skilllearn.evolution",
                trace_id=trace_id,
                payload={
                    "policy": PROPOSAL_FAILURE_ISOLATION_POLICY_VERSION,
                    "request_kind": error.request_kind,
                    "request_hash": error.request_hash,
                    "response_hash": error.response_hash,
                    "failure_phase": error.failure_phase,
                    "error_type": error.error_type,
                    "generation_terminal": True,
                    "report_preserved": True,
                    "performance_claim_eligible": False,
                    "raw_error_persisted": False,
                    "sealed_test_accessed": False,
                },
            )
        )
        result = SkillLearnGenerationResult(
            train_observations=tuple(observations),
            residuals=tuple(residuals),
            evolution=None,
            reason="proposal_model_failed",
            baseline_hypothesis_ids=tuple(row.id for row in incumbent_programs),
            proposal_model_failure_count=1,
            contrastive_training_evidence_policy=(
                self.contrastive_training_evidence_policy
            ),
            candidate_bundle_policy=self.candidate_bundle_policy,
        )
        self._emit_generation_result(result, trace_id)
        return result

    def propose_candidates(
        self,
        residuals: Sequence[ResidualExample],
        *,
        trace_id: str,
    ) -> tuple[HypothesisProgram, ...]:
        return self.kernel.propose_candidates(
            residuals,
            validation_context=self.validation_context(residuals),
            trace_id=trace_id,
        )

    def validation_context(
        self,
        residuals: Sequence[ResidualExample],
    ) -> ValidationContext:
        return ValidationContext(
            evaluator_epoch=self.evaluator_epoch,
            residuals=tuple(residuals),
            available_lanes=frozenset({BASELINE_LANE, CANDIDATE_LANE}),
            baseline_lane=BASELINE_LANE,
            allowed_runtime_kinds=frozenset(
                {HypothesisKind.TASK, HypothesisKind.POLICY}
            ),
            allowed_action_operations=SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
            action_semantics=SKILL_ACTION_LOWERING_VERSION,
            external_evidence_is_hidden=True,
            contrastive_training_evidence_policy=(
                self.contrastive_training_evidence_policy
            ),
            train_action_design_policy=self.train_action_design_policy,
            action_design_profiles={
                str(row.context["action_context_profile_hash"]): dict(
                    row.context[TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY]
                )
                for row in residuals
                if not row.baseline_success
                and isinstance(
                    row.context.get(TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY),
                    Mapping,
                )
                and row.context.get("action_context_profile_hash")
            },
            repair_request_scope_policy=self.repair_request_scope_policy,
            train_coverage_objective=(
                {
                    "policy": self.candidate_selection_policy,
                    **(
                        {
                            "candidate_bundle_policy": (
                                self.candidate_bundle_policy
                            ),
                            **(
                                {
                                    "actual_family_count_precedes_failure_support": True,
                                    "failure_support_precedes_bundle_size": True,
                                }
                                if self.candidate_selection_policy
                                == COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
                                else {}
                            ),
                        }
                        if self.candidate_bundle_policy
                        else {}
                    ),
                    "evidence_scope": "train_only",
                    "coverage_unit": "distinct_failure_family",
                    "minimum_activation_rate": (
                        self.promotion_gate.spec.minimum_activation_rate
                    ),
                    "train_family_count": len(
                        {
                            row.family
                            for row in residuals
                            if row.split is SplitName.TRAIN
                        }
                    ),
                    "failure_activation_family_target": (
                        0
                        if self.promotion_gate.spec.minimum_activation_rate <= 0.0
                        else max(
                            1,
                            math.ceil(
                                self.promotion_gate.spec.minimum_activation_rate
                                * len(
                                    {
                                        row.family
                                        for row in residuals
                                        if row.split is SplitName.TRAIN
                                    }
                                )
                            ),
                        )
                    ),
                    **(
                        {
                            "family_target_deficit_capped_at_target": True,
                            "post_target_actual_family_count_tiebreak": True,
                        }
                        if self.candidate_selection_policy
                        == COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
                        else {"coverage_reward_capped_at_target": True}
                    ),
                    "validation_features_used": False,
                    "validation_outcomes_used": False,
                }
                if self.candidate_selection_policy
                in {
                    PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
                    *COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
                }
                else None
            ),
            trigger_feature_catalog=build_runtime_feature_catalog(
                [
                    {
                        **dict(self.items[item_id].features),
                        "family": self.items[item_id].family,
                    }
                    for item_id in self.manifest.train_ids
                ]
            ),
        )

    def run_sealed_test(
        self,
        program: HypothesisProgram,
        *,
        test_item_ids: Sequence[str] | None = None,
        trace_id: str = "skilllearn_sealed_test",
    ) -> tuple[CounterfactualPair, ...]:
        test_ids = tuple(test_item_ids or self.manifest.test_ids)
        _require_subset(test_ids, self.manifest.test_ids, "sealed test")
        return self.counterfactual_runner.run(
            self.tasks(test_ids),
            program=program,
            baseline_programs=tuple(
                row for row in self.incumbent_programs() if row.id != program.id
            ),
            split=SplitName.TEST,
            trace_id=trace_id,
        )

    def tasks(self, item_ids: Sequence[str]) -> tuple[TaskInput, ...]:
        return tuple(
            TaskInput(
                id=item_id,
                family=self.items[item_id].family,
                features={**dict(self.items[item_id].features), "family": self.items[item_id].family},
                payload=None,
            )
            for item_id in item_ids
        )

    def incumbent_programs(self) -> tuple[HypothesisProgram, ...]:
        if not self.archive.incumbent_id:
            return ()
        node = self.archive.nodes[self.archive.incumbent_id]
        return tuple(self.archive.hypotheses[hypothesis_id] for hypothesis_id in node.active_hypothesis_ids)

    def _compile_training_incumbent(
        self,
        programs: Sequence[HypothesisProgram],
        *,
        train_ids: Sequence[str],
        trace_id: str,
    ):
        for item_id in train_ids:
            self.guard.authorize(item_id, AccessPhase.PROPOSAL)
        if not programs:
            return None
        incumbent_hash = skilllearn_program_set_treatment_hash(programs)[:12]
        return self.compiler.compile(
            programs=programs,
            items=tuple(self.items.values()),
            split_manifest=self.manifest,
            output_root=self.output_root,
            method_name=f"assumption-agent-v2-incumbent-{incumbent_hash}-train",
            allowed_statuses={HypothesisStatus.PROMOTED},
            target_item_ids=train_ids,
            target_split=SplitName.TRAIN.value,
            trace_id=trace_id,
        )

    def _run_training_baseline(
        self,
        item_id: str,
        *,
        skill_source_dir: Path | None,
        program_set_hash: str,
        treatment_hash: str,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        self.guard.authorize(item_id, AccessPhase.PROPOSAL)
        item = self.items[item_id]
        pair_id = stable_hash(
            {
                "trace_id": trace_id,
                "item_id": item_id,
                "stage": "training_baseline",
                "program_set_hash": program_set_hash,
                "treatment_hash": treatment_hash,
            }
        )[:20]
        request = SkillLearnTrialRequest(
            item_id=item_id,
            family=item.family,
            split=SplitName.TRAIN,
            variant=TrialVariant.POLICY_OFF,
            evaluator_epoch=self.evaluator_epoch,
            pair_id=pair_id,
            repeat=1,
            agent_id=self.backend.agent_id,
            model=self.backend.model,
            max_steps=self.backend.max_steps,
            manifest_hash=self.manifest.manifest_hash,
            codex_agent_execution_policy_hash=(
                codex_agent_execution_policy_for_backend(self.backend).policy_hash
            ),
            program_set_hash=program_set_hash,
            treatment_hash=treatment_hash,
        )
        return _run_invalid_only_trial(
            request=request,
            run_once=lambda attempt: self.backend.run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=(
                    f"{trace_id}:{pair_id}:train:attempt-{attempt}"
                ),
            ),
            maximum_attempts=self.invalid_trial_max_attempts,
            backoff_seconds=self.invalid_trial_retry_backoff_seconds,
            retry_semaphore=self._invalid_retry_semaphore,
            event_sink=self.event_sink,
            trace_id=f"{trace_id}:{pair_id}:train",
        )

    def _training_replay_descriptor(
        self,
        *,
        train_ids: Sequence[str],
        incumbent_programs: Sequence[HypothesisProgram],
    ) -> dict[str, Any]:
        task_rows = [
            {
                "task_id": item_id,
                "family": self.items[item_id].family,
                "feature_hash": stable_hash(
                    {
                        **dict(self.items[item_id].features),
                        "family": self.items[item_id].family,
                    }
                ),
            }
            for item_id in train_ids
        ]
        incumbent_behavior_hashes = sorted(
            skilllearn_program_treatment_hash(row) for row in incumbent_programs
        )
        descriptor = {
            "policy": TRAINING_EVIDENCE_REPLAY_POLICY_VERSION,
            "split": SplitName.TRAIN.value,
            "manifest_hash": self.manifest.manifest_hash,
            "evaluator_epoch": self.evaluator_epoch,
            "runtime_version": self.counterfactual_runner.runtime.runtime_version,
            "skill_routing_version": SKILL_ROUTING_VERSION,
            "action_lowering_version": SKILL_ACTION_LOWERING_VERSION,
            "fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
            "provider_route_policy": PROVIDER_ROUTE_POLICY_VERSION,
            "agent_id": self.backend.agent_id,
            "model": self.backend.model,
            "max_steps": self.backend.max_steps,
            "codex_agent_execution_policy_hash": (
                codex_agent_execution_policy_for_backend(self.backend).policy_hash
            ),
            "incumbent_behavior_set_hash": stable_hash(
                incumbent_behavior_hashes
            ),
            "task_set_hash": stable_hash(task_rows),
        }
        if self.train_action_design_policy:
            descriptor.update(
                {
                    "train_action_design_policy": self.train_action_design_policy,
                    "train_action_environment_profile_version": (
                        TRAIN_ACTION_ENVIRONMENT_PROFILE_VERSION
                    ),
                    "train_action_trace_profile_version": (
                        TRAIN_ACTION_TRACE_PROFILE_VERSION
                    ),
                }
            )
        return descriptor

    def _emit_generation_result(self, result: SkillLearnGenerationResult, trace_id: str) -> None:
        self.event_sink.emit(
            Event(
                event="skilllearn_evolution_generation_completed",
                stage="benchmark.skilllearn.evolution",
                trace_id=trace_id,
                payload=result.to_dict(),
            )
        )


def _execution_from_observation(
    observation: SkillLearnTrialObservation,
    *,
    lane: str,
    active_programs: Sequence[HypothesisProgram],
    action_activated: bool,
    baseline_preserved: bool,
) -> RuntimeExecution:
    metadata = {
        "success": observation.success,
        "score": observation.score,
        "metrics": dict(observation.metrics),
        "evaluation_valid": observation.valid,
        "error_type": observation.error_type,
        "observation_hash": observation.observation_hash,
        "provider_fingerprint": observation.provider_fingerprint,
        "fairness_fingerprint": observation.fairness_fingerprint,
        "prebuilt_image_key": observation.prebuilt_image_key,
        "prebuilt_image_id": observation.prebuilt_image_id,
        "prebuilt_cache_reused": observation.prebuilt_cache_reused,
        "agent_runtime_key": observation.agent_runtime_key,
        "agent_runtime_version": observation.agent_runtime_version,
        "raw_trial_artifacts_persisted": observation.raw_trial_artifacts_persisted,
    }
    selected = LaneResult(
        lane=lane,
        answer=observation.success if observation.valid else None,
        confidence=1.0 if observation.valid else 0.0,
        cost=observation.cost_units,
        metadata=metadata,
    )
    return RuntimeExecution(
        task_id=observation.request.item_id,
        selected_result=selected,
        lane_results=(selected,),
        activated_hypothesis_ids=tuple(row.id for row in active_programs),
        plan_hash=stable_hash(
            {
                "request_hash": observation.request.request_hash,
                "lane": lane,
                "active_program_hashes": [row.payload_hash for row in active_programs],
            }
        ),
        action_activated=action_activated,
        baseline_preserved=baseline_preserved,
    )


def _invalid_observation_like(
    request: SkillLearnTrialRequest,
    baseline: SkillLearnTrialObservation,
    error_type: str,
) -> SkillLearnTrialObservation:
    return SkillLearnTrialObservation(
        request=request,
        success=False,
        score=0.0,
        metrics={"task_success": 0.0, "evaluation_valid": 0.0},
        total_tokens=0,
        steps=0,
        duration_seconds=0.0,
        provider_fingerprint=baseline.provider_fingerprint,
        fairness_fingerprint=baseline.fairness_fingerprint,
        error_type=error_type,
    )


def _classify_training_failure(
    observation: SkillLearnTrialObservation,
    *,
    actionable_feedback: bool = False,
) -> tuple[str, tuple[str, ...]]:
    recall = observation.metrics.get("trajectory_key_point_recall")
    if recall is not None and recall < 0.5:
        return (
            "trajectory_keypoints_missing",
            (
                "The external verifier rejected the baseline outcome.",
                "The observed trajectory key-point recall was below the frozen evaluator threshold.",
            ),
        )
    if actionable_feedback:
        return (
            "external_task_verifier_failed",
            (
                "The offline TRAIN verifier rejected the baseline outcome.",
                "Infer a concrete reusable corrective operator from the TRAIN task instruction; use complete imperative task-local steps and do not default to a generic completeness check.",
            ),
        )
    return (
        "external_task_verifier_failed",
        (
            "The external task verifier rejected the baseline outcome.",
            "Propose a reusable procedure with an explicit completion check and preserve-baseline fallback.",
        ),
    )


class _ContainerNetworkBudgetMonitor:
    def __init__(
        self,
        delegate: Any,
        *,
        container_name: str,
        byte_limit: int,
        event_sink: EventSink,
        trace_id: str,
        poll_seconds: float = 2.0,
    ) -> None:
        self.delegate = delegate
        self.container_name = container_name
        self.byte_limit = byte_limit
        self.event_sink = event_sink
        self.trace_id = trace_id
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._rx_bytes = 0
        self._tx_bytes = 0
        self._exceeded = False
        self._finalized = False

    @property
    def exceeded(self) -> bool:
        with self._lock:
            return self._exceeded

    def start(self) -> None:
        self.event_sink.emit(
            Event(
                event="skilllearn_trial_network_budget_monitor_started",
                stage="benchmark.skilllearn.network_budget",
                trace_id=self.trace_id,
                payload={
                    "policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
                    "container_name_hash": stable_hash(
                        {"container_name": self.container_name}
                    ),
                    "byte_limit": self.byte_limit,
                    "poll_seconds": self.poll_seconds,
                },
            )
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"network-budget-{stable_hash({'name': self.container_name})[:10]}",
            daemon=True,
        )
        self._thread.start()

    def finalize(self) -> None:
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(6.0, self.poll_seconds + 4.0))
        self._sample()
        with self._lock:
            rx_bytes = self._rx_bytes
            tx_bytes = self._tx_bytes
            exceeded = self._exceeded
        self.event_sink.emit(
            Event(
                event="skilllearn_trial_network_usage_finalized",
                stage="benchmark.skilllearn.network_budget",
                trace_id=self.trace_id,
                payload={
                    "policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
                    "container_name_hash": stable_hash(
                        {"container_name": self.container_name}
                    ),
                    "rx_bytes": rx_bytes,
                    "tx_bytes": tx_bytes,
                    "total_bytes": rx_bytes + tx_bytes,
                    "byte_limit": self.byte_limit,
                    "limit_exceeded": exceeded,
                    "raw_content_persisted": False,
                },
            )
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            if self.exceeded:
                return
            self._stop.wait(self.poll_seconds)

    def _sample(self) -> None:
        try:
            completed = self.delegate.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{json .}}",
                    self.container_name,
                ],
                capture_output=True,
                text=True,
                timeout=4,
            )
            if int(getattr(completed, "returncode", 1)) != 0:
                return
            rows = [
                row
                for row in str(getattr(completed, "stdout", "") or "").splitlines()
                if row.strip()
            ]
            if not rows:
                return
            payload = json.loads(rows[-1])
            rx_bytes, tx_bytes = _parse_docker_net_io(str(payload.get("NetIO") or ""))
        except (
            OSError,
            subprocess.SubprocessError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return
        should_kill = False
        with self._lock:
            self._rx_bytes = max(self._rx_bytes, rx_bytes)
            self._tx_bytes = max(self._tx_bytes, tx_bytes)
            total_bytes = self._rx_bytes + self._tx_bytes
            if total_bytes > self.byte_limit and not self._exceeded:
                self._exceeded = True
                should_kill = True
        if not should_kill:
            return
        self.delegate.run(
            ["docker", "kill", self.container_name],
            capture_output=True,
            text=True,
        )
        self.event_sink.emit(
            Event(
                event="skilllearn_trial_network_budget_exceeded",
                stage="benchmark.skilllearn.network_budget",
                trace_id=self.trace_id,
                payload={
                    "policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
                    "container_name_hash": stable_hash(
                        {"container_name": self.container_name}
                    ),
                    "observed_bytes": total_bytes,
                    "byte_limit": self.byte_limit,
                    "container_kill_attempted": True,
                    "raw_content_persisted": False,
                },
            )
        )


def _parse_docker_net_io(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split("/")]
    if len(parts) != 2:
        raise ValueError("Docker NetIO must contain received and transmitted values")
    return _parse_docker_byte_size(parts[0]), _parse_docker_byte_size(parts[1])


def _parse_docker_byte_size(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b)", value.strip(), re.I)
    if match is None:
        raise ValueError("unsupported Docker byte size")
    amount = float(match.group(1))
    unit = match.group(2).lower()
    factors = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    return int(amount * factors[unit])


def _docker_removed_container(command: Any) -> str | None:
    if not isinstance(command, list) or command[:2] != ["docker", "rm"]:
        return None
    values = [str(value) for value in command[2:] if not str(value).startswith("-")]
    return values[-1] if values else None


class _DockerVerifierIsolationSubprocessProxy:
    """Hide verifier files and enforce provider-only container networking."""

    def __init__(
        self,
        delegate: Any,
        *,
        agent_runtime_volume: str | None = None,
        offline_verifier_runtime: OfflineVerifierRuntime | None = None,
        egress_policy: DockerEgressPolicy,
        network_byte_limit: int = DEFAULT_TRIAL_NETWORK_BYTE_LIMIT,
        provider_mode: str = "openai_compatible",
        model_inference_limiter: SkillLearnModelInferenceLimiter | None = None,
        event_sink: EventSink | None = None,
        trace_id: str = "skilllearn-docker-isolation",
    ) -> None:
        self.delegate = delegate
        self.agent_runtime_volume = agent_runtime_volume
        self.offline_verifier_runtime = offline_verifier_runtime
        self.egress_policy = egress_policy
        self.network_byte_limit = network_byte_limit
        self.provider_mode = provider_mode
        self.model_inference_limiter = model_inference_limiter
        self.event_sink = event_sink or NullEventSink()
        self.trace_id = trace_id
        self._verifier_sources: dict[str, Path] = {}
        self._network_monitors: dict[str, _ContainerNetworkBudgetMonitor] = {}

    def finalize_network_monitors(self) -> None:
        for container_name, monitor in tuple(self._network_monitors.items()):
            monitor.finalize()
            self._network_monitors.pop(container_name, None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def run(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        command = list(args) if isinstance(args, (list, tuple)) else args
        started_container_name: str | None = None
        removed_container_name = _docker_removed_container(command)
        if removed_container_name in self._network_monitors:
            self._network_monitors.pop(removed_container_name).finalize()
        if (
            isinstance(command, list)
            and len(command) >= 3
            and command[0] == "docker"
            and command[1] == "run"
            and "--name" in command
        ):
            container_name = str(command[command.index("--name") + 1])
            started_container_name = container_name
            if command[-2:] == ["sleep", "3600"]:
                command[-2:] = [
                    "sh",
                    "-c",
                    "trap 'exit 0' TERM INT; while :; do sleep 86400; done",
                ]
                self.event_sink.emit(
                    Event(
                        event="skilllearn_fixed_container_lifetime_removed",
                        stage="benchmark.skilllearn.timeout_policy",
                        trace_id=self.trace_id,
                        payload={
                            "policy": TRIAL_TIMEOUT_POLICY_VERSION,
                            "container_name_hash": stable_hash(
                                {"container_name": container_name}
                            ),
                            "original_lifetime_seconds": 3600,
                            "replacement": "signal_terminable_keepalive_loop",
                        },
                    )
                )
            index = 0
            while index < len(command) - 1:
                if command[index] == "-v" and ":/tests" in str(command[index + 1]):
                    raw_mount = str(command[index + 1])
                    source = raw_mount.split(":/tests", 1)[0]
                    self._verifier_sources[container_name] = Path(source).resolve()
                    del command[index : index + 2]
                    continue
                index += 1
            self.event_sink.emit(
                Event(
                    event="skilllearn_verifier_mount_withheld",
                    stage="benchmark.skilllearn.isolation",
                    trace_id=self.trace_id,
                    payload={
                        "container_name_hash": stable_hash(
                            {"container_name": container_name}
                        ),
                        "verifier_source_hash": stable_hash(
                            {
                                "source": str(
                                    self._verifier_sources.get(container_name) or ""
                                )
                            }
                        ),
                        "agent_runtime_mounted": bool(self.agent_runtime_volume),
                        "offline_verifier_runtime_mounted": bool(
                            self.offline_verifier_runtime
                        ),
                        "codex_home_mounted": False,
                        "tests_mount_present_during_agent": False,
                    },
                )
            )
            if self.agent_runtime_volume:
                mount = (
                    f"{self.agent_runtime_volume}:{SHARED_AGENT_RUNTIME_MOUNT}:ro"
                )
                if mount not in command:
                    command[3:3] = ["-v", mount]
            if self.offline_verifier_runtime is not None:
                verifier_mount = (
                    f"{self.offline_verifier_runtime.volume_name}:"
                    f"{OFFLINE_VERIFIER_MOUNT}:ro"
                )
                if verifier_mount not in command:
                    command[3:3] = ["-v", verifier_mount]
            network_args = self.egress_policy.docker_run_args()
            command[2:2] = network_args
            self.event_sink.emit(
                Event(
                    event="skilllearn_trial_container_network_restricted",
                    stage="benchmark.skilllearn.network_isolation",
                    trace_id=self.trace_id,
                    payload={
                        **self.egress_policy.provenance(),
                        "container_name_hash": stable_hash(
                            {"container_name": container_name}
                        ),
                        "docker_run_network_args_injected": True,
                        "trial_network_budget_policy": (
                            TRIAL_NETWORK_BUDGET_POLICY_VERSION
                        ),
                        "trial_network_byte_limit": self.network_byte_limit,
                        "secret_value_persisted": False,
                    },
                )
            )
        if (
            isinstance(command, list)
            and len(command) >= 4
            and command[:2] == ["docker", "exec"]
            and str(command[2]) in self._verifier_sources
            and "/tests/test.sh" in {str(value) for value in command[3:]}
        ):
            container_name = str(command[2])
            source = self._verifier_sources[container_name]
            if not source.is_dir():
                raise FileNotFoundError(
                    "verifier source disappeared before post-agent injection"
                )
            self.delegate.run(
                ["docker", "exec", container_name, "mkdir", "-p", "/tests"],
                check=True,
                capture_output=True,
            )
            self.delegate.run(
                ["docker", "cp", f"{source}/.", f"{container_name}:/tests"],
                check=True,
                capture_output=True,
            )
            self.event_sink.emit(
                Event(
                    event="skilllearn_verifier_materialized_post_agent",
                    stage="benchmark.skilllearn.isolation",
                    trace_id=self.trace_id,
                    payload={
                        "container_name_hash": stable_hash(
                            {"container_name": container_name}
                        ),
                        "verifier_content_hash": _directory_content_hash(source),
                        "materialization": "docker_cp_after_agent_exit",
                        "tests_mount_present_during_agent": False,
                    },
                )
            )
            if self.offline_verifier_runtime is not None:
                command = [
                    "docker",
                    "exec",
                    container_name,
                    "sh",
                    "-lc",
                    self.offline_verifier_runtime.profile.verifier_command,
                ]
                self.event_sink.emit(
                    Event(
                        event="skilllearn_offline_verifier_command_selected",
                        stage="benchmark.skilllearn.offline_verifier",
                        trace_id=self.trace_id,
                        payload={
                            "policy": OFFLINE_VERIFIER_POLICY_VERSION,
                            "profile_id": (
                                self.offline_verifier_runtime.profile.profile_id
                            ),
                            "profile_hash": (
                                self.offline_verifier_runtime.profile.profile_hash
                            ),
                            "runtime_key": (
                                self.offline_verifier_runtime.runtime_key
                            ),
                            "semantic_prelude_id": (
                                self.offline_verifier_runtime.profile.semantic_prelude_id
                            ),
                            "model_secret_env_unset_before_verifier": True,
                            "container_name_hash": stable_hash(
                                {"container_name": container_name}
                            ),
                            "original_online_test_script_executed": False,
                            "verifier_network": "provider_restricted_no_dependency_access",
                            "runtime_mount_read_only": True,
                            "raw_content_persisted": False,
                        },
                    )
                )
        timeout_stage = _trial_timeout_stage(command)
        if timeout_stage and "timeout" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("timeout", None)
            self.event_sink.emit(
                Event(
                    event="skilllearn_fixed_wall_timeout_removed",
                    stage="benchmark.skilllearn.timeout_policy",
                    trace_id=self.trace_id,
                    payload={
                        "policy": TRIAL_TIMEOUT_POLICY_VERSION,
                        "trial_stage": timeout_stage,
                        "timeout_argument_removed": True,
                    },
                )
            )
        if timeout_stage == "agent" and self.model_inference_limiter is not None:
            with self.model_inference_limiter.acquire(
                event_sink=self.event_sink,
                trace_id=self.trace_id,
            ):
                completed = self.delegate.run(command, *positional, **kwargs)
        else:
            completed = self.delegate.run(command, *positional, **kwargs)
        if (
            started_container_name
            and int(getattr(completed, "returncode", 1)) == 0
            and re.fullmatch(
                r"[0-9a-f]{12,64}",
                str(getattr(completed, "stdout", "") or "").strip().lower(),
            )
        ):
            prior = self._network_monitors.pop(started_container_name, None)
            if prior is not None:
                prior.finalize()
            monitor = _ContainerNetworkBudgetMonitor(
                self.delegate,
                container_name=started_container_name,
                byte_limit=self.network_byte_limit,
                event_sink=self.event_sink,
                trace_id=self.trace_id,
            )
            self._network_monitors[started_container_name] = monitor
            monitor.start()
        if timeout_stage == "agent":
            container_name = str(command[2]) if len(command) > 2 else ""
            monitor = self._network_monitors.get(container_name)
            if monitor is not None and monitor.exceeded:
                raise SkillLearnAgentTerminalError(
                    "trial_network_byte_limit_exceeded"
                )
            terminal_error = _codex_terminal_error_label(
                getattr(completed, "stdout", None),
                getattr(completed, "stderr", None),
            )
            terminal_error = _provider_scoped_terminal_error(
                terminal_error,
                self.provider_mode,
            )
            if terminal_error:
                self.event_sink.emit(
                    Event(
                        event="skilllearn_agent_terminal_error_detected",
                        stage="benchmark.skilllearn.provider_failure",
                        trace_id=self.trace_id,
                        payload={
                            "error_type": terminal_error,
                            "policy": PROVIDER_FAILURE_POLICY_VERSION,
                            "stdout_hash": stable_hash(
                                {"stdout": str(getattr(completed, "stdout", "") or "")}
                            ),
                            "stderr_hash": stable_hash(
                                {"stderr": str(getattr(completed, "stderr", "") or "")}
                            ),
                            "raw_content_persisted": False,
                        },
                    )
                )
                raise SkillLearnAgentTerminalError(terminal_error)
        return completed


def _directory_content_hash(
    root: Path,
    *,
    excluded_top_level: set[str] | None = None,
) -> str:
    excluded = excluded_top_level or set()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in excluded:
            continue
        if path.is_symlink():
            raise PermissionError("prebuilt environments may not contain symbolic links")
        mode = path.stat().st_mode & 0o777
        if path.is_dir():
            rows.append({"path": relative.as_posix(), "kind": "dir", "mode": mode})
        elif path.is_file():
            rows.append(
                {
                    "path": relative.as_posix(),
                    "kind": "file",
                    "mode": mode,
                    "size": path.stat().st_size,
                    "sha256": _file_content_hash(path),
                }
            )
    return stable_hash(rows)


def _file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_prebuilt_image(
    runner: ModuleType,
    tag: str,
    expected_cache_key: str,
) -> str | None:
    inspected = runner.subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
        text=True,
    )
    if int(getattr(inspected, "returncode", 1)) != 0:
        return None
    try:
        payload = json.loads(str(getattr(inspected, "stdout", "") or ""))
        row = payload[0]
        labels = row.get("Config", {}).get("Labels") or {}
        if labels.get("org.assumption-agent.prebuild.key") != expected_cache_key:
            raise PermissionError("prebuilt image label does not match its cache key")
        image_id = str(row.get("Id") or "")
        if not image_id.startswith("sha256:"):
            raise ValueError("prebuilt image has no immutable Docker image ID")
        return image_id
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("prebuilt image metadata is malformed") from exc


def _safe_subprocess_snippet(result: Any) -> str:
    raw = str(getattr(result, "stderr", "") or getattr(result, "stdout", "") or "")
    cleaned = re.sub(r"\s+", " ", raw).strip()
    return cleaned[:400] or "no_diagnostic_output"


def _fairness_fingerprint(
    *,
    agent_id: str,
    model: str,
    provider_mode: str,
    max_steps: int,
    provider_fingerprint: str,
    prebuilt_enabled: bool,
    agent_runtime_key: str,
    prebuilt_image_key: str,
    prebuilt_image_id: str,
    offline_verifier_runtime_key: str,
    codex_agent_execution_policy: CodexAgentExecutionPolicy = (
        LEGACY_CODEX_AGENT_EXECUTION_POLICY
    ),
    model_inference_concurrency_policy: str | None = None,
    model_inference_slots: int = 0,
) -> str:
    payload = {
            "backend": "skilllearnbench_upstream_v1",
            "agent_id": agent_id,
            "model": model,
            "max_steps": max_steps,
            "provider_fingerprint": provider_fingerprint,
            "prebuilt_policy": (
                PREBUILT_IMAGE_POLICY_VERSION if prebuilt_enabled else "disabled"
            ),
            "agent_runtime_key": agent_runtime_key,
            "prebuilt_image_key": prebuilt_image_key,
            "prebuilt_image_id": prebuilt_image_id,
            "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
            "offline_verifier_runtime_key": offline_verifier_runtime_key,
            "verifier_isolation": VERIFIER_ISOLATION_VERSION,
            "verifier_execution_receipt_policy": (
                VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION
            ),
            "runner_agent_registry_isolation": (
                RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
            ),
            "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
            "provider_failure_policy": PROVIDER_FAILURE_POLICY_VERSION,
            "codex_network_minimization": (
                codex_network_minimization_for_policy(codex_agent_execution_policy)
            ),
            "codex_agent_execution_policy_hash": (
                codex_agent_execution_policy.policy_hash
            ),
            "model_only_tool_policy": MODEL_ONLY_TOOL_POLICY_VERSION,
            "container_egress_policy": DOCKER_EGRESS_POLICY_VERSION,
            "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
            "provider_dns_policy": PROVIDER_DNS_POLICY_VERSION,
            "trial_network_budget_policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
            "trial_network_byte_limit": configured_trial_network_byte_limit(),
            "provider_route_policy": PROVIDER_ROUTE_POLICY_VERSION,
            "openai_compatible_codex_config": (
                OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION
                if agent_id == "codex" and provider_mode == "openai_compatible"
                else None
            ),
        }
    if model_inference_concurrency_policy is not None:
        payload["model_inference_concurrency_policy"] = (
            model_inference_concurrency_policy
        )
        payload["model_inference_slots"] = model_inference_slots
    return stable_hash(payload)


def _provider_fingerprint(
    agent_id: str,
    model: str,
    provider_mode: str,
    codex_agent_execution_policy: CodexAgentExecutionPolicy = (
        LEGACY_CODEX_AGENT_EXECUTION_POLICY
    ),
) -> str:
    base_url = (
        os.environ.get("ASSUMPTION_V2_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("GPT5_BASE_URL")
        or os.environ.get("RUOLI_BASE_URL")
        or "provider-default"
    )
    if base_url != "provider-default":
        base_url = _normalize_openai_compatible_codex_base_url(base_url)
    return stable_hash(
        {
            "agent_id": agent_id,
            "model": model,
            "provider_mode": provider_mode,
            "base_url": base_url,
            "verifier_isolation": VERIFIER_ISOLATION_VERSION,
            "verifier_execution_receipt_policy": (
                VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION
            ),
            "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
            "runner_agent_registry_isolation": (
                RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
            ),
            "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
            "provider_failure_policy": PROVIDER_FAILURE_POLICY_VERSION,
            "codex_network_minimization": (
                codex_network_minimization_for_policy(codex_agent_execution_policy)
            ),
            "codex_agent_execution_policy_hash": (
                codex_agent_execution_policy.policy_hash
            ),
            "model_only_tool_policy": MODEL_ONLY_TOOL_POLICY_VERSION,
            "container_egress_policy": DOCKER_EGRESS_POLICY_VERSION,
            "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
            "provider_dns_policy": PROVIDER_DNS_POLICY_VERSION,
            "allowed_ipv4s_hash": stable_hash(
                {
                    "allowed_ipv4s": os.environ.get(
                        "ASSUMPTION_V2_API_ALLOWED_IPV4S", ""
                    )
                }
            ),
            "provider_route_policy": PROVIDER_ROUTE_POLICY_VERSION,
            "openai_compatible_codex_config": (
                OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION
                if agent_id == "codex"
                else None
            ),
        }
    )


def _openai_compatible_codex_config_values(
    *,
    policy: CodexAgentExecutionPolicy,
    codex_base_url: str,
) -> tuple[str, ...]:
    return (
        "--ignore-user-config",
        "--ephemeral",
        "--strict-config",
        "--disable",
        "image_generation",
        "--disable",
        "standalone_web_search",
        "--config",
        "analytics.enabled=false",
        "--config",
        'otel.exporter="none"',
        "--config",
        'otel.metrics_exporter="none"',
        "--config",
        'otel.trace_exporter="none"',
        *policy.codex_cli_values(),
        "--config",
        (
            "developer_instructions="
            + json.dumps(
                "This evaluation environment is offline except for model inference. "
                "Do not use web search, image generation services, package installation, "
                "or commands that fetch network content. Use only preinstalled local tools "
                "and files."
            )
        ),
        "--config",
        f"model_provider={json.dumps(OPENAI_COMPATIBLE_CODEX_PROVIDER_ID)}",
        "--config",
        (
            f'model_providers.{OPENAI_COMPATIBLE_CODEX_PROVIDER_ID}.name='
            f'{json.dumps("OpenAI-compatible paper route")}'
        ),
        "--config",
        (
            f'model_providers.{OPENAI_COMPATIBLE_CODEX_PROVIDER_ID}.base_url='
            f"{json.dumps(codex_base_url)}"
        ),
        "--config",
        (
            f'model_providers.{OPENAI_COMPATIBLE_CODEX_PROVIDER_ID}.env_key='
            f'{json.dumps("OPENAI_API_KEY")}'
        ),
        "--config",
        (
            f'model_providers.{OPENAI_COMPATIBLE_CODEX_PROVIDER_ID}.wire_api='
            f'{json.dumps("responses")}'
        ),
        "--config",
        (
            f"model_providers.{OPENAI_COMPATIBLE_CODEX_PROVIDER_ID}."
            "supports_websockets=false"
        ),
        "--config",
        (
            f"model_providers.{OPENAI_COMPATIBLE_CODEX_PROVIDER_ID}."
            "requires_openai_auth=false"
        ),
    )


def _normalize_openai_compatible_codex_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("openai-compatible provider base URL must be HTTP(S)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("openai-compatible provider base URL contains unsupported components")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _configured_openai_compatible_origin() -> str | None:
    base_url = (
        os.environ.get("ASSUMPTION_V2_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("GPT5_BASE_URL")
        or os.environ.get("RUOLI_BASE_URL")
    )
    if not base_url:
        return None
    parsed = urlsplit(_normalize_openai_compatible_codex_base_url(base_url))
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _provider_scoped_terminal_error(
    error_type: str | None,
    provider_mode: str,
) -> str | None:
    return {
        "subscription_authentication_failed": "provider_authentication_failed",
        "subscription_model_unavailable": "provider_model_unavailable",
        "subscription_usage_limit": "provider_usage_limit",
    }.get(error_type, error_type)


def _trial_timeout_stage(command: Any) -> str | None:
    if not isinstance(command, list) or len(command) < 4:
        return None
    if command[:2] != ["docker", "exec"]:
        return None
    command_text = " ".join(str(value) for value in command[3:])
    if "/tests/test.sh" in command_text:
        return "verifier"
    if "codex exec" in command_text:
        return "agent"
    return None


def _codex_terminal_error_label(*streams: Any) -> str | None:
    generic_failure_observed = False
    for stream in streams:
        if stream is None:
            continue
        if isinstance(stream, bytes):
            text = stream.decode(errors="replace")
        else:
            text = str(stream)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping):
                continue
            event_type = str(row.get("type") or "")
            if event_type not in {"error", "turn.failed"}:
                continue
            nested = row.get("error") if isinstance(row.get("error"), Mapping) else {}
            message = str(row.get("message") or nested.get("message") or "").lower()
            if "usage limit" in message or "quota" in message:
                return "provider_usage_limit"
            if "rate limit" in message or "too many requests" in message:
                return "provider_rate_limit"
            if any(
                value in message
                for value in (
                    "not logged in",
                    "unauthorized",
                    "authentication",
                    "login required",
                    "api key",
                    "invalid key",
                )
            ):
                return "provider_authentication_failed"
            if any(
                value in message
                for value in (
                    "model is not available",
                    "model unavailable",
                    "unsupported model",
                    "503 service unavailable",
                    "distributor",
                    "无可用渠道",
                )
            ):
                return "provider_model_unavailable"
            generic_failure_observed = True
    return "codex_turn_failed" if generic_failure_observed else None


def _safe_error_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    label = str(value).strip().lower().replace(" ", "_")
    label = re.sub(r"[^a-z0-9_.-]+", "_", label).strip("_")
    return label[:96] or "upstream_error"


def _inspect_verifier_execution_receipt(
    *,
    test_script: Path,
    verifier_dir: Path,
    result: Mapping[str, Any],
    offline_verifier_profile: OfflineVerifierProfile | None = None,
) -> _VerifierExecutionReceipt:
    if offline_verifier_profile is not None:
        executed_verifier_command = offline_verifier_profile.verifier_command
        evidence_kind = (
            "pytest_ctrf"
            if "--ctrf" in executed_verifier_command
            else "unsupported"
        )
    else:
        executed_verifier_command = None
        evidence_kind = (
            "pytest_ctrf" if _test_script_uses_ctrf(test_script) else "unsupported"
        )
    reward: int | None = None
    test_count = 0
    evidence_hashes: dict[str, Any] = {
        "policy": VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION,
        "test_script_hash": _file_content_hash(test_script) if test_script.is_file() else None,
        "evidence_kind": evidence_kind,
        "executed_verifier_source": (
            "offline_verifier_profile"
            if offline_verifier_profile is not None
            else "benchmark_test_script"
        ),
    }
    if offline_verifier_profile is not None:
        evidence_hashes.update(
            {
                "offline_verifier_profile_id": offline_verifier_profile.profile_id,
                "offline_verifier_profile_hash": offline_verifier_profile.profile_hash,
                "executed_verifier_command_hash": stable_hash(
                    {"command": executed_verifier_command}
                ),
            }
        )
    semantic_prelude = inspect_semantic_prelude_receipt(
        profile=offline_verifier_profile,
        verifier_dir=verifier_dir,
    )
    evidence_hashes["semantic_prelude"] = {
        "required": semantic_prelude.required,
        "valid": semantic_prelude.valid,
        "succeeded": semantic_prelude.succeeded,
        "prelude_id": semantic_prelude.prelude_id,
        "exit_code": semantic_prelude.exit_code,
        "details": dict(semantic_prelude.details),
        "receipt_hash": semantic_prelude.receipt_hash,
    }

    error_type: str | None = None
    verifier_exit = result.get("verifier_exit")
    try:
        verifier_exit_code = int(verifier_exit)
    except (TypeError, ValueError):
        verifier_exit_code = None
    if verifier_exit_code is None:
        error_type = "verifier_execution_exit_missing"
    elif verifier_exit_code != 0:
        error_type = (
            "verifier_timeout"
            if verifier_exit_code == -1
            else "verifier_execution_nonzero_exit"
        )

    reward_file = verifier_dir / "reward.txt"
    if reward_file.is_file():
        raw_reward = reward_file.read_text(encoding="utf-8", errors="replace").strip()
        if raw_reward in {"0", "1"}:
            reward = int(raw_reward)
            evidence_hashes["reward_sha256"] = _file_content_hash(reward_file)
        elif error_type is None:
            error_type = "verifier_execution_reward_malformed"
    elif error_type is None:
        error_type = "verifier_execution_reward_missing"

    result_reward = result.get("reward")
    if (
        reward is not None
        and isinstance(result_reward, (int, float))
        and float(result_reward) != float(reward)
        and error_type is None
    ):
        error_type = "verifier_execution_reward_mismatch"

    if not test_script.is_file() and error_type is None:
        error_type = "verifier_execution_test_script_missing"
    elif evidence_kind != "pytest_ctrf" and error_type is None:
        error_type = "verifier_execution_receipt_unsupported"
    elif evidence_kind == "pytest_ctrf":
        ctrf_file = verifier_dir / "ctrf.json"
        if not ctrf_file.is_file():
            if error_type is None:
                error_type = "verifier_execution_ctrf_missing"
        else:
            evidence_hashes["ctrf_sha256"] = _file_content_hash(ctrf_file)
            try:
                payload = json.loads(ctrf_file.read_text(encoding="utf-8"))
                results = payload.get("results") if isinstance(payload, Mapping) else None
                summary = results.get("summary") if isinstance(results, Mapping) else None
                test_rows = results.get("tests") if isinstance(results, Mapping) else None
                if not isinstance(summary, Mapping) or not isinstance(test_rows, list):
                    raise ValueError("CTRF results are incomplete")
                test_count = _as_nonnegative_int(summary.get("tests"))
                if test_count <= 0 or len(test_rows) != test_count:
                    raise ValueError("CTRF contains no complete test execution")
                evidence_hashes["test_count"] = test_count
                evidence_hashes["summary_hash"] = stable_hash(
                    {
                        key: summary.get(key)
                        for key in (
                            "tests",
                            "passed",
                            "failed",
                            "skipped",
                            "pending",
                            "other",
                        )
                    }
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                if error_type is None:
                    error_type = "verifier_execution_ctrf_malformed"

    if not semantic_prelude.valid and error_type is None:
        error_type = semantic_prelude.error_type

    receipt_hash = stable_hash(evidence_hashes)
    return _VerifierExecutionReceipt(
        valid=error_type is None,
        error_type=error_type,
        evidence_kind=evidence_kind,
        reward=reward,
        test_count=test_count,
        semantic_prelude_required=semantic_prelude.required,
        semantic_prelude_valid=semantic_prelude.valid,
        semantic_prelude_succeeded=semantic_prelude.succeeded,
        semantic_prelude_id=semantic_prelude.prelude_id,
        semantic_prelude_exit_code=semantic_prelude.exit_code,
        semantic_prelude_details=semantic_prelude.details,
        semantic_prelude_receipt_hash=semantic_prelude.receipt_hash,
        receipt_hash=receipt_hash,
    )


def _test_script_uses_ctrf(test_script: Path) -> bool:
    if not test_script.is_file():
        return False
    return "--ctrf" in test_script.read_text(encoding="utf-8", errors="replace")


def _inspect_codex_tool_policy(trace_path: Path) -> _CodexToolPolicyAudit:
    if not trace_path.is_file():
        return _CodexToolPolicyAudit(
            valid=False,
            error_type="model_tool_audit_trace_missing",
            remote_tool_call_count=0,
            runtime_install_command_count=0,
            trace_hash="",
        )
    remote_tool_ids: set[str] = set()
    runtime_install_ids: set[str] = set()
    for raw_line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw_line)
        except (TypeError, json.JSONDecodeError):
            continue
        item = row.get("item") if isinstance(row, Mapping) else None
        if not isinstance(item, Mapping):
            continue
        item_type = str(item.get("type") or "")
        item_id = str(item.get("id") or stable_hash(item))
        if item_type in _FORBIDDEN_CODEX_TOOL_TYPES:
            remote_tool_ids.add(item_id)
        if item_type == "command_execution":
            command = str(item.get("command") or "")
            if any(pattern.search(command) for pattern in _FORBIDDEN_RUNTIME_COMMAND_PATTERNS):
                runtime_install_ids.add(item_id)
    remote_tool_count = len(remote_tool_ids)
    runtime_install_count = len(runtime_install_ids)
    error_type = None
    if remote_tool_count:
        error_type = "model_remote_tool_policy_violation"
    elif runtime_install_count:
        error_type = "model_runtime_install_policy_violation"
    return _CodexToolPolicyAudit(
        valid=error_type is None,
        error_type=error_type,
        remote_tool_call_count=remote_tool_count,
        runtime_install_command_count=runtime_install_count,
        trace_hash=_file_content_hash(trace_path),
    )


def _extract_train_action_trace_profile(
    trace_path: Path,
    *,
    containment_root: Path | None = None,
) -> dict[str, Any]:
    """Extract bounded, allowlisted action facts without raw command text.

    The resulting mapping is safe to place in a proposal request: it contains only
    a fixed executable label, fixed flag labels, normalized task-local paths,
    outcome metadata, and a one-way hash of the original command.  Any reference
    to verifier/oracle material or credential-bearing/network syntax is discarded
    as a whole rather than partially redacted.
    """

    if trace_path.is_symlink():
        return {}
    if containment_root is not None:
        try:
            root = containment_root.resolve(strict=True)
        except FileNotFoundError:
            return {}
        try:
            relative = trace_path.relative_to(root)
        except ValueError:
            return {}
        current = root
        for part in relative.parts:
            if part in {"", ".", ".."}:
                return {}
            current = current / part
            if current.is_symlink():
                return {}
        try:
            resolved_trace = trace_path.resolve(strict=True)
            resolved_trace.relative_to(root)
        except (FileNotFoundError, ValueError):
            return {}
    if not trace_path.is_file():
        return {}
    command_rows: list[dict[str, Any]] = []
    changed_paths: set[str] = set()
    for raw_line in trace_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        try:
            row = json.loads(raw_line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(row, Mapping) or row.get("type") != "item.completed":
            continue
        item = row.get("item")
        if not isinstance(item, Mapping):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "command_execution":
            command = str(item.get("command") or "")
            signature = _allowlisted_action_trace_command(command)
            if signature is None:
                continue
            exit_code = item.get("exit_code")
            normalized_exit_code = (
                0
                if isinstance(exit_code, int)
                and not isinstance(exit_code, bool)
                and exit_code == 0
                else (
                    1
                    if isinstance(exit_code, int)
                    and not isinstance(exit_code, bool)
                    else None
                )
            )
            command_rows.append(
                {
                    **signature,
                    "status": _normalized_action_trace_status(
                        item.get("status")
                    ),
                    "exit_code": normalized_exit_code,
                }
            )
        elif item_type == "file_change":
            changes = item.get("changes")
            if not isinstance(changes, list):
                continue
            for change in changes:
                if not isinstance(change, Mapping):
                    continue
                path = str(change.get("path") or "").strip()
                safe_path = _allowlisted_action_trace_root_path(path)
                if safe_path is not None:
                    changed_paths.add(safe_path)
    unique_by_hash: dict[str, dict[str, Any]] = {}
    for row in command_rows:
        unique_by_hash.setdefault(str(row["original_command_hash"]), row)
    unique = list(unique_by_hash.values())
    failed = [
        row
        for row in unique
        if row["status"] == "failed"
        or (isinstance(row["exit_code"], int) and row["exit_code"] != 0)
    ]
    successful = [row for row in unique if row not in failed]
    selected = [*failed, *successful]
    selected = selected[:12]
    profile = {
        "policy": TRAIN_ACTION_TRACE_PROFILE_VERSION,
        "command_signatures": selected,
        "commands_observed": len(command_rows),
        "unique_commands_observed": len(unique),
        "failed_commands_observed": len(failed),
        "commands_returned": len(selected),
        "changed_task_paths": sorted(changed_paths)[:12],
        "model_messages_used": False,
        "command_output_used": False,
        "verifier_trace_used": False,
        "raw_trace_content_persisted": False,
    }
    return profile


def _allowlisted_action_trace_command(command: str) -> dict[str, Any] | None:
    """Return non-free-text command facts, or fail closed for unsafe input."""

    value = command.strip()
    if not value or len(value) > 100_000:
        return None
    decoded_for_checks = value.replace("%2f", "/").replace("%2F", "/")
    decoded_for_checks = decoded_for_checks.replace("%2e", ".").replace("%2E", ".")
    if (
        _ACTION_TRACE_FORBIDDEN_REFERENCE_PATTERN.search(decoded_for_checks)
        or _ACTION_TRACE_SENSITIVE_PATTERN.search(value)
        or "\x00" in value
    ):
        return None
    tokens = _action_trace_command_tokens(value)
    if not tokens:
        return None
    executable = Path(tokens[0]).name.lower()
    if re.fullmatch(r"python3(?:\.\d+)?", executable):
        executable = "python3"
    if executable not in _ACTION_TRACE_ALLOWED_EXECUTABLES:
        return None
    safe_flags = sorted(
        {
            flag
            for token in tokens[1:]
            if (flag := token.split("=", 1)[0])
            in _ACTION_TRACE_ALLOWED_FLAGS
        }
    )
    observed_task_local_paths = [
        match.group(0)
        for match in _ACTION_TRACE_ROOT_PATH_PATTERN.finditer(value)
    ]
    normalized_task_local_paths = [
        _allowlisted_action_trace_root_path(path)
        for path in observed_task_local_paths
    ]
    if any(path is None for path in normalized_task_local_paths):
        return None
    task_local_paths = sorted(
        {str(path) for path in normalized_task_local_paths if path is not None}
    )[:12]
    return {
        "executable_basename": executable,
        "safe_flags": safe_flags,
        "task_local_paths": task_local_paths,
        "original_command_hash": stable_hash({"command": value}),
    }


def _action_trace_command_tokens(command: str) -> list[str]:
    """Parse the invoked local executable through a small wrapper allowlist."""

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return []
    if not tokens:
        return []
    for _ in range(4):
        while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
            tokens.pop(0)
        if not tokens:
            return []
        executable = Path(tokens[0]).name.lower()
        if executable in _ACTION_TRACE_SHELL_WRAPPERS:
            script_index = next(
                (
                    index + 1
                    for index, token in enumerate(tokens[:-1])
                    if token in {"-c", "-lc"}
                ),
                None,
            )
            if script_index is None:
                return []
            try:
                tokens = shlex.split(tokens[script_index], posix=True)
            except ValueError:
                return []
            continue
        if executable in _ACTION_TRACE_COMMAND_WRAPPERS:
            tokens = tokens[1:]
            while tokens and (
                tokens[0].startswith("-")
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0])
            ):
                tokens.pop(0)
            continue
        break
    if not tokens or any(token in {"&&", "||", ";", "|"} for token in tokens):
        return []
    return tokens


def _allowlisted_action_trace_root_path(value: str) -> str | None:
    candidate = value.strip().rstrip(".,;:)]}\"'")
    if not candidate.startswith("/root/") or len(candidate) > 300:
        return None
    if not _ACTION_TRACE_ROOT_PATH_PATTERN.fullmatch(candidate):
        return None
    if any(part in {"", ".", ".."} for part in candidate.split("/")[2:]):
        return None
    if _ACTION_TRACE_FORBIDDEN_REFERENCE_PATTERN.search(candidate):
        return None
    if _ACTION_TRACE_SENSITIVE_PATH_COMPONENT_PATTERN.search(candidate):
        return None
    return candidate


def _normalized_action_trace_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in {"completed", "complete", "succeeded", "success"}:
        return "completed"
    if status in {"failed", "failure", "error"}:
        return "failed"
    return "unknown"


def _as_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _run_invalid_only_trial(
    *,
    request: SkillLearnTrialRequest,
    run_once: Callable[[int], SkillLearnTrialObservation],
    maximum_attempts: int,
    backoff_seconds: float,
    retry_semaphore: threading.Semaphore,
    event_sink: EventSink,
    trace_id: str,
) -> SkillLearnTrialObservation:
    if maximum_attempts <= 0:
        raise ValueError("invalid trial maximum attempts must be positive")
    if backoff_seconds < 0:
        raise ValueError("invalid trial retry backoff cannot be negative")
    observation = run_once(1)
    if observation.request.request_hash != request.request_hash:
        raise PermissionError("trial execution changed the frozen request key")
    for attempt in range(2, maximum_attempts + 1):
        if observation.valid:
            break
        suppression_reason = _invalid_retry_suppression_reason(observation)
        if suppression_reason is not None:
            event_sink.emit(
                Event(
                    event="skilllearn_invalid_trial_retry_suppressed",
                    stage="benchmark.skilllearn.invalid_retry",
                    trace_id=trace_id,
                    payload={
                        "policy": INVALID_TRIAL_RETRY_POLICY_VERSION,
                        "request_hash": request.request_hash,
                        "attempts_executed": attempt - 1,
                        "maximum_attempts": maximum_attempts,
                        "error_type": observation.error_type,
                        "suppression_reason": suppression_reason,
                        "hard_budget_violation": (
                            observation.error_type
                            == "trial_network_byte_limit_exceeded"
                        ),
                        "same_request_key": True,
                        "raw_content_persisted": False,
                    },
                )
            )
            break
        previous = observation
        delay = backoff_seconds * (attempt - 1)
        event_sink.emit(
            Event(
                event="skilllearn_invalid_trial_retry_scheduled",
                stage="benchmark.skilllearn.invalid_retry",
                trace_id=trace_id,
                payload={
                    "policy": INVALID_TRIAL_RETRY_POLICY_VERSION,
                    "request_hash": request.request_hash,
                    "attempt": attempt,
                    "maximum_attempts": maximum_attempts,
                    "backoff_seconds": delay,
                    "previous_error_type": previous.error_type,
                    "previous_observation_hash": previous.observation_hash,
                    "same_request_key_required": True,
                    "raw_content_persisted": False,
                },
            )
        )
        if delay:
            time.sleep(delay)
        with retry_semaphore:
            replacement = run_once(attempt)
        if replacement.request.request_hash != request.request_hash:
            raise PermissionError("invalid retry changed the frozen request key")
        observation = replacement
        event_sink.emit(
            Event(
                event=(
                    "skilllearn_invalid_trial_clean_replacement"
                    if replacement.valid
                    else "skilllearn_invalid_trial_retry_failed"
                ),
                stage="benchmark.skilllearn.invalid_retry",
                trace_id=trace_id,
                payload={
                    "policy": INVALID_TRIAL_RETRY_POLICY_VERSION,
                    "request_hash": request.request_hash,
                    "attempt": attempt,
                    "maximum_attempts": maximum_attempts,
                    "previous_error_type": previous.error_type,
                    "replacement_error_type": replacement.error_type,
                    "previous_observation_hash": previous.observation_hash,
                    "replacement_observation_hash": replacement.observation_hash,
                    "replacement_valid": replacement.valid,
                    "same_request_key": True,
                    "raw_content_persisted": False,
                },
            )
        )
    return observation


def _retryable_invalid_observation(
    observation: SkillLearnTrialObservation,
) -> bool:
    return _invalid_retry_suppression_reason(observation) is None


def _invalid_retry_suppression_reason(
    observation: SkillLearnTrialObservation,
) -> str | None:
    error_type = observation.error_type or ""
    if not error_type:
        return "missing_error_type"
    if error_type in _FATAL_PROVIDER_ERROR_TYPES:
        return "fatal_provider_error"
    if error_type.startswith("provider_circuit_open_"):
        return "provider_circuit_open"
    if error_type == "trial_network_byte_limit_exceeded":
        return "hard_network_budget_exceeded"
    if error_type.startswith("verifier_execution_"):
        return "nontransient_verifier_infrastructure_error"
    if error_type.startswith("offline_verifier_"):
        return "nontransient_offline_verifier_configuration_error"
    if error_type.startswith("model_") and error_type.endswith("_policy_violation"):
        return "nontransient_model_tool_policy_violation"
    if error_type in {
        "candidate_skill_source_missing",
        "compiled_candidate_skill_missing",
        "runner_configuration_error",
    }:
        return "nontransient_configuration_error"
    return None


def _validate_training_observations(
    observations: Sequence[SkillLearnTrialObservation],
    *,
    train_item_ids: Sequence[str],
    descriptor: Mapping[str, Any],
) -> None:
    if tuple(row.request.item_id for row in observations) != tuple(train_item_ids):
        raise PermissionError("training replay task identity mismatch")
    if any(not row.valid for row in observations):
        raise SkillLearnTrainingEvidenceError(
            "invalid training evidence may not enter the replay cache"
        )
    if any(
        row.request.split is not SplitName.TRAIN
        or row.request.manifest_hash != descriptor.get("manifest_hash")
        or row.request.evaluator_epoch != descriptor.get("evaluator_epoch")
        or row.request.agent_id != descriptor.get("agent_id")
        or row.request.model != descriptor.get("model")
        or row.request.max_steps != descriptor.get("max_steps")
        or row.request.codex_agent_execution_policy_hash
        != descriptor.get("codex_agent_execution_policy_hash")
        for row in observations
    ):
        raise PermissionError("training replay crossed a frozen execution boundary")


def _ordered_parallel_map(
    values: Sequence[_InputT],
    function: Callable[[_InputT], _OutputT],
    workers: int,
) -> tuple[_OutputT, ...]:
    if workers <= 1 or len(values) <= 1:
        return tuple(function(value) for value in values)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return tuple(executor.map(function, values))


def _require_subset(selected: Sequence[str], allowed: Sequence[str], label: str) -> None:
    unexpected = sorted(set(selected) - set(allowed))
    if unexpected:
        raise PermissionError(f"{label} selection contains IDs outside its frozen split")


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))
