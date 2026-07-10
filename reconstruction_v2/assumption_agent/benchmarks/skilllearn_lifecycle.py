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
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence, TypeVar

from ..archive import PolicyArchive
from ..evaluation import PromotionGate
from ..events import Event, EventSink, NullEventSink
from ..evolution import EvolutionKernel, EvolutionRunResult
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
from ..proposer import StructuredHypothesisProposer
from ..secure_env import configured_skilllearn_provider_mode, resolve_codex_auth_path
from ..splits import AccessPhase, BenchmarkItem, SplitAccessGuard, SplitManifest
from ..validation import (
    RecursiveValidationEngine,
    ValidationContext,
    build_runtime_feature_catalog,
)
from .skilllearn_compiler import SkillLearnProgramCompiler
from .skilllearnbench import SkillLearnBenchAdapter


BASELINE_LANE = "skilllearn_incumbent"
CANDIDATE_LANE = "skilllearn_challenger"
VERIFIER_ISOLATION_VERSION = "post_agent_verifier_copy_v1"
RUNNER_AGENT_REGISTRY_ISOLATION_VERSION = "runner_local_agent_registry_v1"
TRIAL_TIMEOUT_POLICY_VERSION = "no_fixed_trial_stage_wall_timeout_v1"
PREBUILT_IMAGE_POLICY_VERSION = "per_item_base_shared_agent_runtime_v3"
SHARED_AGENT_RUNTIME_MOUNT = "/opt/assumption-v2-agent"
SHARED_AGENT_RUNTIME_BUILDER_IMAGE = (
    "node@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0"
)
SHARED_CODEX_CLI_PACKAGE = "@openai/codex@0.144.1"
SHARED_CODEX_CLI_VERSION = "codex-cli 0.144.1"
_InputT = TypeVar("_InputT")
_OutputT = TypeVar("_OutputT")


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
    program_id: str | None = None

    @property
    def request_hash(self) -> str:
        return stable_hash(self.to_dict())

    @property
    def trial_id(self) -> str:
        return f"v2_{self.variant.value}_{self.request_hash[:18]}"

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "program_id": self.program_id,
        }


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

    @property
    def valid(self) -> bool:
        return self.error_type is None

    @property
    def cost_units(self) -> float:
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
        return {
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
            "secret_value_persisted": False,
        }


class SkillLearnTrialBackend(Protocol):
    agent_id: str
    model: str
    max_steps: int

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation: ...


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
    """Build and reuse agent-ready images for one exact task environment."""

    def __init__(
        self,
        benchmark_root: str | Path,
        *,
        event_sink: EventSink | None = None,
    ) -> None:
        self.benchmark_root = Path(benchmark_root).expanduser().resolve()
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
            agent_image_hash=agent_image_hash,
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
        agent_image_hash: str,
        trace_id: str,
    ) -> tuple[str, str, str]:
        if agent_id != "codex":
            raise ValueError("shared agent runtime currently supports the frozen codex agent only")
        install = str(agent.get("install") or "").strip()
        if not install.startswith("npm ") or "@openai/codex" not in install:
            raise ValueError("shared codex runtime requires an npm agent install command")
        runtime_key = stable_hash(
            {
                "policy": PREBUILT_IMAGE_POLICY_VERSION,
                "builder_image": SHARED_AGENT_RUNTIME_BUILDER_IMAGE,
                "agent_image_hash": agent_image_hash,
                "codex_cli_package": SHARED_CODEX_CLI_PACKAGE,
            }
        )
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
                    f"npm_config_prefix=/runtime npm install -g {SHARED_CODEX_CLI_PACKAGE}"
                )
                populated = runner.subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "-v",
                        f"{volume}:/runtime",
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
            verified = runner.subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
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
                        "policy": PREBUILT_IMAGE_POLICY_VERSION,
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
        if any(
            (row.agent_id, row.model, row.max_steps)
            != (first.agent_id, first.model, first.max_steps)
            for row in backends
        ):
            raise ValueError("all pooled backends must share one frozen configuration")
        self.agent_id = first.agent_id
        self.model = first.model
        self.max_steps = first.max_steps
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
        model: str = "gpt-5.3-codex-spark",
        max_steps: int = 100,
        provider_mode: str | None = None,
        trials_dir: str | Path | None = None,
        record_upstream: bool = True,
        prebuilt_cache: SkillLearnPrebuiltImageCache | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.benchmark_root = Path(benchmark_root).expanduser().resolve()
        self.agent_id = agent_id
        self.model = model
        self.max_steps = max_steps
        self.provider_mode = (
            provider_mode
            or (
                configured_skilllearn_provider_mode()
                if agent_id == "codex"
                else "openai_compatible"
            )
        )
        if self.provider_mode not in {"codex_subscription", "openai_compatible"}:
            raise ValueError("unsupported SkillLearn trial provider mode")
        if self.provider_mode == "codex_subscription" and self.agent_id != "codex":
            raise ValueError("codex_subscription trial mode requires the codex agent")
        self.trials_dir = Path(trials_dir).expanduser().resolve() if trials_dir else None
        self.record_upstream = record_upstream
        self.prebuilt_cache = prebuilt_cache
        self.event_sink = event_sink or NullEventSink()
        self._runner_module: ModuleType | None = None
        self._runner_instance_token = stable_hash(
            {"benchmark_root": str(self.benchmark_root), "instance": id(self)}
        )[:12]
        self._provider_lock = threading.RLock()

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
        if request.variant is TrialVariant.POLICY_ON and skill_source_dir is None:
            return self._local_error(request, "candidate_skill_source_missing")

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
                    "runner_agent_registry_isolation": (
                        RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
                    ),
                    "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
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
        try:
            runner = self._load_runner()
            if self.prebuilt_cache is not None:
                prebuilt_image = self.prebuilt_cache.ensure(
                    family=request.family,
                    item_id=request.item_id,
                    agent_id=self.agent_id,
                    runner=runner,
                    trace_id=f"{trace_id}:prebuild",
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
                trace_id=trace_id,
            ):
                return_code, result = runner.run_task(
                    f"{request.family}/{request.item_id}",
                    **kwargs,
                )
        except Exception as exc:
            self.event_sink.emit(
                Event(
                    event="skilllearn_trial_infrastructure_failed",
                    stage="benchmark.skilllearn.trial",
                    trace_id=trace_id,
                    payload={
                        "request_hash": request.request_hash,
                        "error_type": type(exc).__name__,
                        "error_message_hash": stable_hash({"message": str(exc)}),
                        "prebuilt_stage": bool(
                            self.prebuilt_cache is not None and prebuilt_image is None
                        ),
                        "raw_content_persisted": False,
                    },
                )
            )
            result = {"error": type(exc).__name__}
            return_code = 2
        observation = self._sanitize_result(
            request,
            result=result,
            return_code=return_code,
            duration_seconds=time.monotonic() - started,
            prebuilt_image=prebuilt_image,
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
        trace_id: str = "skilllearn-provider-runtime",
    ) -> Iterator[None]:
        if self.agent_id != "codex":
            with self._verifier_isolation(
                runner,
                agent_runtime_volume=agent_runtime_volume,
                trace_id=trace_id,
            ):
                yield
            return
        if self.provider_mode == "openai_compatible":
            agent = runner.get_agent(self.agent_id)
            original_agent = copy.deepcopy(agent) if isinstance(agent, dict) else None
            try:
                self._prepare_openai_compatible_provider(runner)
                if isinstance(agent, dict) and agent_runtime_volume:
                    trajectory_env = dict(agent.get("trajectory_env") or {})
                    trajectory_env["PATH"] = f"{SHARED_AGENT_RUNTIME_MOUNT}/bin:$PATH"
                    agent["trajectory_env"] = trajectory_env
                with self._verifier_isolation(
                    runner,
                    agent_runtime_volume=agent_runtime_volume,
                    trace_id=trace_id,
                ):
                    yield
            finally:
                if isinstance(agent, dict) and original_agent is not None:
                    agent.clear()
                    agent.update(original_agent)
            return
        auth_path = resolve_codex_auth_path()
        if auth_path is None:
            raise RuntimeError("local Codex subscription auth.json is required")
        agent = runner.get_agent(self.agent_id)
        if not isinstance(agent, dict):
            raise RuntimeError("SkillLearnBench codex agent definition is unavailable")
        secret_tmp_root = os.environ.get("ASSUMPTION_V2_SECRET_TMPDIR", "").strip() or None
        if secret_tmp_root and not Path(secret_tmp_root).expanduser().is_dir():
            raise RuntimeError("ASSUMPTION_V2_SECRET_TMPDIR is not a directory")
        with self._provider_lock:
            original_agent = copy.deepcopy(agent)
            original_subprocess = runner.subprocess
            with tempfile.TemporaryDirectory(
                prefix="assumption-v2-codex-auth-",
                dir=str(Path(secret_tmp_root).expanduser()) if secret_tmp_root else None,
            ) as secret_dir:
                ephemeral_auth = Path(secret_dir) / "auth.json"
                shutil.copyfile(auth_path, ephemeral_auth)
                ephemeral_auth.chmod(0o600)
                agent["env"] = []
                agent["setup"] = None
                trajectory_env = dict(agent.get("trajectory_env") or {})
                trajectory_env["CODEX_HOME"] = "/root/.codex"
                if agent_runtime_volume:
                    trajectory_env["PATH"] = f"{SHARED_AGENT_RUNTIME_MOUNT}/bin:$PATH"
                agent["trajectory_env"] = trajectory_env
                runner.subprocess = _DockerCodexHomeSubprocessProxy(
                    original_subprocess,
                    host_codex_home=Path(secret_dir),
                    agent_runtime_volume=agent_runtime_volume,
                    event_sink=self.event_sink,
                    trace_id=trace_id,
                )
                try:
                    yield
                finally:
                    runner.subprocess = original_subprocess
                    agent.clear()
                    agent.update(original_agent)

    @contextmanager
    def _verifier_isolation(
        self,
        runner: ModuleType,
        *,
        agent_runtime_volume: str | None = None,
        trace_id: str = "skilllearn-verifier-isolation",
    ) -> Iterator[None]:
        original_subprocess = runner.subprocess
        runner.subprocess = _DockerCodexHomeSubprocessProxy(
            original_subprocess,
            host_codex_home=None,
            agent_runtime_volume=agent_runtime_volume,
            event_sink=self.event_sink,
            trace_id=trace_id,
        )
        try:
            yield
        finally:
            runner.subprocess = original_subprocess

    def _prepare_openai_compatible_provider(self, runner: ModuleType) -> None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ASSUMPTION_V2_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("ASSUMPTION_V2_API_BASE")
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        if base_url:
            os.environ.setdefault("OPENAI_BASE_URL", base_url.rstrip("/"))
        agent = runner.get_agent(self.agent_id)
        if isinstance(agent, dict) and os.environ.get("OPENAI_BASE_URL"):
            env_names = list(agent.get("env") or [])
            if "OPENAI_BASE_URL" not in env_names:
                agent["env"] = [*env_names, "OPENAI_BASE_URL"]

    def _sanitize_result(
        self,
        request: SkillLearnTrialRequest,
        *,
        result: Mapping[str, Any],
        return_code: int,
        duration_seconds: float,
        prebuilt_image: SkillLearnPrebuiltImage | None = None,
    ) -> SkillLearnTrialObservation:
        usage = result.get("token_usage") if isinstance(result.get("token_usage"), Mapping) else {}
        total_tokens = _as_nonnegative_int(usage.get("total_tokens"))
        if not total_tokens:
            total_tokens = _as_nonnegative_int(usage.get("input_tokens")) + _as_nonnegative_int(
                usage.get("output_tokens")
            )
        steps = _as_nonnegative_int(result.get("steps_used"))
        error_type = _safe_error_label(result.get("error"))
        if result.get("agent_timed_out") is True:
            error_type = error_type or "agent_timeout"
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
        )
        fairness_fingerprint = _fairness_fingerprint(
            agent_id=self.agent_id,
            model=self.model,
            max_steps=self.max_steps,
            provider_fingerprint=provider_fingerprint,
            prebuilt_enabled=self.prebuilt_cache is not None,
            agent_runtime_key=(
                prebuilt_image.agent_runtime_key if prebuilt_image else ""
            ),
            prebuilt_image_key=prebuilt_image.cache_key if prebuilt_image else "",
            prebuilt_image_id=prebuilt_image.image_id if prebuilt_image else "",
        )
        sanitized_upstream = {
            "task_id_hash": stable_hash({"task_id": result.get("task_id")}),
            "trial_id_hash": stable_hash({"trial_id": result.get("trial_id") or result.get("trial_name")}),
            "agent": result.get("agent"),
            "model": result.get("model"),
            "provider_mode": self.provider_mode,
            "verifier_isolation": VERIFIER_ISOLATION_VERSION,
            "runner_agent_registry_isolation": (
                RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
            ),
            "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
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
            "skill_config": result.get("skill_config"),
            "passed": result.get("passed"),
            "reward": result.get("reward"),
            "agent_exit": result.get("agent_exit"),
            "agent_timed_out": result.get("agent_timed_out"),
            "verifier_exit": result.get("verifier_exit"),
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
        )

    def _local_error(self, request: SkillLearnTrialRequest, error_type: str) -> SkillLearnTrialObservation:
        provider = _provider_fingerprint(
            self.agent_id,
            self.model,
            self.provider_mode,
        )
        fairness = _fairness_fingerprint(
            agent_id=self.agent_id,
            model=self.model,
            max_steps=self.max_steps,
            provider_fingerprint=provider,
            prebuilt_enabled=self.prebuilt_cache is not None,
            agent_runtime_key="",
            prebuilt_image_key="",
            prebuilt_image_id="",
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
        event_sink: EventSink | None = None,
    ) -> None:
        self.adapter = adapter
        self.manifest = manifest
        self.guard = guard
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
            if observation.success:
                continue
            item = self.items[request.item_id]
            instruction = self.adapter.load_instruction(
                request.item_id,
                phase=AccessPhase.PROPOSAL,
                guard=self.guard,
            ).strip()
            failure_type, feedback = _classify_training_failure(observation)
            residual = ResidualExample(
                transition_id=f"transition_{stable_hash({'request': request.request_hash, 'outcome': observation.observation_hash})[:18]}",
                task_id=request.item_id,
                family=request.family,
                split=SplitName.TRAIN,
                features={**dict(item.features), "family": item.family},
                failure_type=failure_type,
                evaluator_feedback=feedback,
                baseline_success=False,
                context={
                    "task_instruction": instruction,
                    "observed_metrics": dict(sorted(observation.metrics.items())),
                    "execution_signals": {
                        "total_tokens": observation.total_tokens,
                        "steps": observation.steps,
                        "duration_seconds": observation.duration_seconds,
                    },
                },
            )
            issues = residual.validate()
            if issues:
                raise PermissionError(f"training residual failed isolation checks: {issues}")
            residuals.append(residual)
        self.event_sink.emit(
            Event(
                event="skilllearn_training_residuals_mined",
                stage="benchmark.skilllearn.residuals",
                trace_id=trace_id,
                payload={
                    "observation_count": len(observations),
                    "residual_count": len(residuals),
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
    runtime_version: str = "skilllearn_external_runtime_v1"


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
        event_sink: EventSink | None = None,
    ) -> None:
        if parallel_workers <= 0:
            raise ValueError("counterfactual worker count must be positive")
        self.adapter = adapter
        self.manifest = manifest
        self.guard = guard
        self.backend = backend
        self.evaluator = evaluator
        self.compiler = compiler
        self.output_root = Path(output_root)
        self.parallel_workers = parallel_workers
        self.event_sink = event_sink or NullEventSink()
        self.items = {item.id: item for item in adapter.discover()}
        self.runtime = _ExternalRuntimeDescriptor()

    def run(
        self,
        tasks: Sequence[TaskInput],
        *,
        program: HypothesisProgram,
        baseline_programs: Sequence[HypothesisProgram] = (),
        split: SplitName,
        trace_id: str = "skilllearn_counterfactual",
    ) -> tuple[CounterfactualPair, ...]:
        if split not in {SplitName.VALIDATION, SplitName.TEST}:
            raise ValueError("external counterfactuals are restricted to validation or sealed test")
        if program.evaluator_epoch != self.evaluator.epoch:
            raise ValueError("program and SkillLearnBench evaluator epochs differ")
        if split is SplitName.TEST and program.status is not HypothesisStatus.PROMOTED:
            raise PermissionError("sealed test requires a promoted, frozen hypothesis program")
        phase = AccessPhase.PROMOTION if split is SplitName.VALIDATION else AccessPhase.FINAL_REPORT
        for task in tasks:
            authorized = self.guard.authorize(task.id, phase)
            if authorized is not split:
                raise PermissionError("counterfactual task is in the wrong split")

        target_ids = tuple(task.id for task in tasks)
        target_hash = stable_hash({"item_ids": sorted(target_ids)})[:10]
        baseline_compile_result = None
        if baseline_programs:
            baseline_hash = stable_hash(
                {"program_hashes": [row.payload_hash for row in baseline_programs]}
            )[:12]
            baseline_compile_result = self.compiler.compile(
                programs=baseline_programs,
                items=tuple(self.items.values()),
                split_manifest=self.manifest,
                output_root=self.output_root,
                method_name=f"assumption-agent-v2-incumbent-{baseline_hash}-{split.value}-{target_hash}",
                allowed_statuses={HypothesisStatus.PROMOTED},
                target_item_ids=target_ids,
                target_split=split.value,
                trace_id=trace_id,
            )
        candidate_programs = tuple(
            {row.id: row for row in (*baseline_programs, program)}.values()
        )
        candidate_compile_result = self.compiler.compile(
            programs=candidate_programs,
            items=tuple(self.items.values()),
            split_manifest=self.manifest,
            output_root=self.output_root,
            method_name=f"assumption-agent-v2-challenger-{program.payload_hash[:12]}-{split.value}-{target_hash}",
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
                program=program,
                baseline_programs=baseline_programs,
                candidate_programs=candidate_programs,
                baseline_compile_result=baseline_compile_result,
                candidate_compile_result=candidate_compile_result,
                split=split,
                trace_id=trace_id,
            )

        return _ordered_parallel_map(tasks, run_one, self.parallel_workers)

    def _run_pair(
        self,
        task: TaskInput,
        *,
        program: HypothesisProgram,
        baseline_programs: Sequence[HypothesisProgram],
        candidate_programs: Sequence[HypothesisProgram],
        baseline_compile_result: Any,
        candidate_compile_result: Any,
        split: SplitName,
        trace_id: str,
    ) -> CounterfactualPair:
        pair_id = stable_hash(
            {
                "trace_id": trace_id,
                "task_id": task.id,
                "program_id": program.id,
                "split": split.value,
            }
        )[:20]
        off_request = self._request(task, split, TrialVariant.POLICY_OFF, pair_id, None)
        on_request = self._request(task, split, TrialVariant.POLICY_ON, pair_id, program.id)
        activated = program.matches(task.features)
        baseline_skill_source = (
            baseline_compile_result.source_for(task.id)
            if baseline_compile_result
            else None
        )
        candidate_skill_source = (
            candidate_compile_result.source_for(task.id) if activated else None
        )
        run_on_first = False
        if activated and (
            candidate_skill_source is None or not candidate_skill_source.is_dir()
        ):
            baseline_observation = self.backend.run(
                off_request,
                skill_source_dir=baseline_skill_source,
                trace_id=f"{trace_id}:{pair_id}:off",
            )
            candidate_observation = _invalid_observation_like(
                on_request,
                baseline_observation,
                "compiled_candidate_skill_missing",
            )
        elif not activated:
            baseline_observation = self.backend.run(
                off_request,
                skill_source_dir=baseline_skill_source,
                trace_id=f"{trace_id}:{pair_id}:off",
            )
            candidate_observation = baseline_observation.as_variant(on_request)
        else:
            run_on_first = (
                int(stable_hash({"pair_id": pair_id, "order": "balanced"})[:8], 16) % 2 == 1
            )
            if run_on_first:
                candidate_observation = self.backend.run(
                    on_request,
                    skill_source_dir=candidate_skill_source,
                    trace_id=f"{trace_id}:{pair_id}:on",
                )
                baseline_observation = self.backend.run(
                    off_request,
                    skill_source_dir=baseline_skill_source,
                    trace_id=f"{trace_id}:{pair_id}:off",
                )
            else:
                baseline_observation = self.backend.run(
                    off_request,
                    skill_source_dir=baseline_skill_source,
                    trace_id=f"{trace_id}:{pair_id}:off",
                )
                candidate_observation = self.backend.run(
                    on_request,
                    skill_source_dir=candidate_skill_source,
                    trace_id=f"{trace_id}:{pair_id}:on",
                )

        baseline_execution = _execution_from_observation(
            baseline_observation,
            lane=BASELINE_LANE,
            active_programs=baseline_programs,
            action_activated=False,
        )
        candidate_execution = _execution_from_observation(
            candidate_observation,
            lane=CANDIDATE_LANE if activated else BASELINE_LANE,
            active_programs=candidate_programs if activated else baseline_programs,
            action_activated=activated,
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
        self.event_sink.emit(
            Event(
                event="skilllearn_counterfactual_pair_completed",
                stage="benchmark.skilllearn.counterfactual",
                trace_id=f"{trace_id}:{pair_id}",
                payload={
                    "pair_id": pair_id,
                    "task_id_hash": stable_hash({"task_id": task.id}),
                    "split": split.value,
                    "hypothesis_id": program.id,
                    "baseline_hypothesis_ids": [row.id for row in baseline_programs],
                    "candidate_hypothesis_ids": [
                        row.id for row in (candidate_programs if activated else baseline_programs)
                    ],
                    "action_activated": activated,
                    "baseline_success": pair.baseline_outcome.success,
                    "candidate_success": pair.candidate_outcome.success,
                    "baseline_score": pair.baseline_outcome.score,
                    "candidate_score": pair.candidate_outcome.score,
                    "baseline_valid": bool(pair.baseline_outcome.metrics.get("evaluation_valid")),
                    "candidate_valid": bool(pair.candidate_outcome.metrics.get("evaluation_valid")),
                    "baseline_observation_hash": baseline_observation.observation_hash,
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
                    "run_order": "on_off" if activated and run_on_first else "off_on",
                    "parallel_workers": self.parallel_workers,
                },
            )
        )
        return pair

    def _request(
        self,
        task: TaskInput,
        split: SplitName,
        variant: TrialVariant,
        pair_id: str,
        program_id: str | None,
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
            program_id=program_id,
        )


@dataclass(frozen=True)
class SkillLearnGenerationResult:
    train_observations: tuple[SkillLearnTrialObservation, ...]
    residuals: tuple[ResidualExample, ...]
    evolution: EvolutionRunResult | None
    reason: str
    baseline_hypothesis_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        decision = self.evolution.promotion_decision if self.evolution else None
        return {
            "train_observation_count": len(self.train_observations),
            "valid_train_observation_count": sum(row.valid for row in self.train_observations),
            "training_residual_count": len(self.residuals),
            "baseline_hypothesis_ids": list(self.baseline_hypothesis_ids),
            "evolution_trace_id": self.evolution.trace_id if self.evolution else None,
            "promoted": bool(self.evolution and self.evolution.promoted),
            "accepted_hypothesis_id": self.evolution.accepted_hypothesis_id if self.evolution else None,
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
            "selected_candidate_node_count": (
                len(self.evolution.validation_tree.nodes) if self.evolution else 0
            ),
            "selected_candidate_recursion_depth": (
                self.evolution.validation_tree.recursion_depth if self.evolution else 0
            ),
            "promotion_blockers": list(decision.blockers) if decision else [],
            "promotion_summary": decision.summary.to_dict(confidence=decision.confidence) if decision else None,
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
        parallel_workers: int = 1,
        event_sink: EventSink | None = None,
    ) -> None:
        if parallel_workers <= 0:
            raise ValueError("evolution worker count must be positive")
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
        self.event_sink = event_sink or NullEventSink()
        self.items = {item.id: item for item in adapter.discover()}
        self.residual_miner = SkillLearnResidualMiner(
            adapter=adapter,
            manifest=manifest,
            guard=guard,
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
            event_sink=self.event_sink,
        )

    def run_generation(
        self,
        *,
        train_item_ids: Sequence[str] | None = None,
        validation_item_ids: Sequence[str] | None = None,
        trace_id: str = "skilllearn_evolution_generation",
    ) -> SkillLearnGenerationResult:
        train_ids = tuple(train_item_ids or self.manifest.train_ids)
        validation_ids = tuple(validation_item_ids or self.manifest.validation_ids)
        _require_subset(train_ids, self.manifest.train_ids, "training")
        _require_subset(validation_ids, self.manifest.validation_ids, "validation")
        observations = self.collect_training_observations(
            train_item_ids=train_ids,
            trace_id=trace_id,
        )
        residuals = self.residual_miner.mine(observations, trace_id=trace_id)
        return self.run_generation_from_evidence(
            observations=observations,
            residuals=residuals,
            validation_item_ids=validation_ids,
            trace_id=trace_id,
        )

    def collect_training_observations(
        self,
        *,
        train_item_ids: Sequence[str] | None = None,
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
                trace_id=trace_id,
            )

        return _ordered_parallel_map(train_ids, run_one, self.parallel_workers)

    def run_generation_from_evidence(
        self,
        *,
        observations: Sequence[SkillLearnTrialObservation],
        residuals: Sequence[ResidualExample],
        validation_item_ids: Sequence[str] | None = None,
        proposal_candidates: Sequence[HypothesisProgram] | None = None,
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
        if any(row.task_id not in observation_ids or row.split is not SplitName.TRAIN for row in residuals):
            raise PermissionError("shared residual is outside the training observation checkpoint")
        for row in residuals:
            issues = row.validate()
            if issues:
                raise PermissionError(f"shared residual failed isolation checks: {issues}")
            self.guard.authorize(row.task_id, AccessPhase.PROPOSAL)
        incumbent_programs = self.incumbent_programs()
        if not residuals:
            result = SkillLearnGenerationResult(
                train_observations=observations,
                residuals=(),
                evolution=None,
                reason="no_valid_failed_training_rows",
                baseline_hypothesis_ids=tuple(row.id for row in incumbent_programs),
            )
            self._emit_generation_result(result, trace_id)
            return result
        validation_tasks = self.tasks(validation_ids)
        validation_context = self.validation_context(residuals)
        evolution = self.kernel.evolve_once(
            residuals=residuals,
            validation_tasks=validation_tasks,
            validation_context=validation_context,
            proposal_candidates=proposal_candidates,
            trace_id=trace_id,
        )
        result = SkillLearnGenerationResult(
            train_observations=observations,
            residuals=residuals,
            evolution=evolution,
            reason=evolution.reason,
            baseline_hypothesis_ids=tuple(row.id for row in incumbent_programs),
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
        incumbent_hash = stable_hash({"program_hashes": [row.payload_hash for row in programs]})[:12]
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
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        self.guard.authorize(item_id, AccessPhase.PROPOSAL)
        item = self.items[item_id]
        pair_id = stable_hash({"trace_id": trace_id, "item_id": item_id, "stage": "training_baseline"})[:20]
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
        )
        return self.backend.run(
            request,
            skill_source_dir=skill_source_dir,
            trace_id=f"{trace_id}:{pair_id}:train",
        )

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
        baseline_preserved=all(row.fallback == "preserve_baseline" for row in active_programs),
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
    return (
        "external_task_verifier_failed",
        (
            "The external task verifier rejected the baseline outcome.",
            "Propose a reusable procedure with an explicit completion check and preserve-baseline fallback.",
        ),
    )


class _DockerCodexHomeSubprocessProxy:
    """Hide verifier files until agent exit and optionally bind a Codex home."""

    def __init__(
        self,
        delegate: Any,
        *,
        host_codex_home: Path | None,
        agent_runtime_volume: str | None = None,
        event_sink: EventSink | None = None,
        trace_id: str = "skilllearn-docker-isolation",
    ) -> None:
        self.delegate = delegate
        self.host_codex_home = host_codex_home.resolve() if host_codex_home else None
        self.agent_runtime_volume = agent_runtime_volume
        self.event_sink = event_sink or NullEventSink()
        self.trace_id = trace_id
        self._verifier_sources: dict[str, Path] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def run(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        command = list(args) if isinstance(args, (list, tuple)) else args
        if (
            isinstance(command, list)
            and len(command) >= 3
            and command[0] == "docker"
            and command[1] == "run"
            and "--name" in command
        ):
            container_name = str(command[command.index("--name") + 1])
            index = 0
            while index < len(command) - 1:
                if command[index] == "-v" and ":/tests" in str(command[index + 1]):
                    raw_mount = str(command[index + 1])
                    source = raw_mount.split(":/tests", 1)[0]
                    self._verifier_sources[container_name] = Path(source).resolve()
                    del command[index : index + 2]
                    continue
                index += 1
            if self.host_codex_home is not None:
                mount = f"{self.host_codex_home}:/root/.codex"
                if mount not in command:
                    command[3:3] = ["-v", mount]
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
                        "codex_home_mounted": self.host_codex_home is not None,
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
        return self.delegate.run(command, *positional, **kwargs)


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
    max_steps: int,
    provider_fingerprint: str,
    prebuilt_enabled: bool,
    agent_runtime_key: str,
    prebuilt_image_key: str,
    prebuilt_image_id: str,
) -> str:
    return stable_hash(
        {
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
            "verifier_isolation": VERIFIER_ISOLATION_VERSION,
            "runner_agent_registry_isolation": (
                RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
            ),
            "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
        }
    )


def _provider_fingerprint(agent_id: str, model: str, provider_mode: str) -> str:
    if provider_mode == "codex_subscription":
        return stable_hash(
            {
                "agent_id": agent_id,
                "model": model,
                "provider_mode": provider_mode,
                "auth_materialization": "ephemeral_codex_home_bind",
                "verifier_isolation": VERIFIER_ISOLATION_VERSION,
                "runner_agent_registry_isolation": (
                    RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
                ),
                "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
            }
        )
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("ASSUMPTION_V2_API_BASE")
        or os.environ.get("GPT5_BASE_URL")
        or os.environ.get("RUOLI_BASE_URL")
        or "provider-default"
    )
    return stable_hash(
        {
            "agent_id": agent_id,
            "model": model,
            "provider_mode": provider_mode,
            "base_url": base_url,
            "verifier_isolation": VERIFIER_ISOLATION_VERSION,
            "runner_agent_registry_isolation": (
                RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
            ),
            "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
        }
    )


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


def _safe_error_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    label = str(value).strip().lower().replace(" ", "_")
    label = re.sub(r"[^a-z0-9_.-]+", "_", label).strip("_")
    return label[:96] or "upstream_error"


def _as_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


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
