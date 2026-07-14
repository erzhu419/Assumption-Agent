from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from assumption_agent.benchmarks import prewarm as prewarm_module
from assumption_agent.benchmarks import skilllearn_lifecycle
from assumption_agent.benchmarks.codex_action_budget import (
    CODEX_ACTION_BUDGET_POLICY_VERSION,
)
from assumption_agent.benchmarks.docker_egress import DEPENDENCY_CACHE_POLICY_VERSION
from assumption_agent.benchmarks.offline_verifier import (
    OFFLINE_VERIFIER_POLICY_VERSION,
    offline_verifier_profile_for_family,
    offline_verifier_runtime_key,
)
from assumption_agent.benchmarks.prewarm import (
    FrozenTaskInputPrebuiltImageCache,
    TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION,
    _selected_item_set_hash,
    validate_development_prewarm_receipt,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    PREBUILT_IMAGE_POLICY_VERSION,
    SHARED_CODEX_CLI_VERSION,
    SkillLearnPrebuiltImageCache,
    SkillLearnSubprocessBackend,
    codex_action_supervisor_hash,
    shared_codex_agent_runtime_key,
)
from assumption_agent.benchmarks.task_input_closure import (
    TASK_INPUT_BUILD_CONTEXT_POLICY_VERSION,
    TASK_INPUT_CLOSURE_POLICY_VERSION,
    TaskInputClosureError,
    source_environment_hash,
)
from assumption_agent.benchmarks.task_input_freeze import (
    FrozenTaskInputClosure,
    load_frozen_task_input_closure,
)
from assumption_agent.models import stable_hash
from assumption_agent.splits import SplitManifest


class _ImageDocker:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.images: dict[str, dict[str, Any]] = {}

    def run(self, args, *positional, **kwargs):
        command = list(args)
        if command[:3] == ["docker", "image", "inspect"]:
            image = self.images.get(command[3])
            if image is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="missing")
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([image]),
                stderr="",
            )
        if command[:2] == ["docker", "build"]:
            self.order.append("build")
            labels = {
                command[index + 1].split("=", 1)[0]: command[index + 1].split("=", 1)[1]
                for index, value in enumerate(command)
                if value == "--label"
            }
            tag = command[command.index("-t") + 1]
            self.images[tag] = {
                "Id": "sha256:" + labels["org.assumption-agent.prebuild.key"],
                "Config": {"Labels": labels},
            }
            return SimpleNamespace(returncode=0, stdout="built", stderr="")
        raise AssertionError(f"unexpected subprocess command: {command}")


def _closure_benchmark(tmp_path: Path) -> tuple[Path, ModuleType, _ImageDocker]:
    benchmark = tmp_path / "benchmark"
    environment = (
        benchmark
        / "tasks"
        / "organize-messy-files"
        / "organize-messy-files-1"
        / "environment"
    )
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text(
        "FROM scratch\nWORKDIR /root/papers/all\n",
        encoding="utf-8",
    )
    (benchmark / "core").mkdir()
    (benchmark / "core" / "eval_runner.py").write_text(
        "# frozen runner\n",
        encoding="utf-8",
    )
    order: list[str] = []
    docker = _ImageDocker(order)
    runner = ModuleType("task_input_closure_runner")
    runner.subprocess = docker
    runner.get_agent = lambda agent_id: {
        "runtime_deps": "RUN-DEPS",
        "install": "npm install -g @openai/codex",
    }
    build_index = 0

    def prepare(source: Path, skill_mode: str, skill_source_dir) -> Path:
        nonlocal build_index
        build_index += 1
        build_env = tmp_path / f"build-{build_index}" / "environment"
        shutil.copytree(source, build_env)
        return build_env

    runner._prepare_build_env = prepare
    runner._parse_skill_copies = lambda dockerfile: []
    return benchmark, runner, docker


def test_prebuilt_cache_materializes_and_inspects_opt_in_task_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark, runner, docker = _closure_benchmark(tmp_path)
    order = docker.order
    closure_hash = "a" * 64
    environment = (
        benchmark
        / "tasks"
        / "organize-messy-files"
        / "organize-messy-files-1"
        / "environment"
    )
    manifest = {
        "closure_hash": closure_hash,
        "manifest_hash": closure_hash,
        "source_environment_hash": source_environment_hash(environment),
    }
    monkeypatch.setattr(
        skilllearn_lifecycle,
        "load_task_input_closure",
        lambda cache_root, family, item_id: manifest,
    )

    def materialize(**kwargs):
        order.append("materialize")
        return {"receipt_hash": "b" * 64}

    def inspect(**kwargs):
        order.append("inspect")
        assert kwargs["run"] == runner.subprocess.run
        assert kwargs["image"].startswith("sha256:")
        assert kwargs["image_id"] == kwargs["image"]
        return {
            "passed": True,
            "container_network": "none",
            "manifest_hash": closure_hash,
            "receipt_hash": "c" * 64,
        }

    monkeypatch.setattr(skilllearn_lifecycle, "materialize_build_context", materialize)
    monkeypatch.setattr(skilllearn_lifecycle, "inspect_image_inputs", inspect)
    monkeypatch.setattr(
        SkillLearnPrebuiltImageCache,
        "_ensure_agent_runtime",
        lambda self, **kwargs: ("runtime-key", "runtime-volume", "runtime-version"),
    )

    cache = SkillLearnPrebuiltImageCache(
        benchmark,
        cache_only=False,
        task_input_closure_policy=TASK_INPUT_CLOSURE_POLICY_VERSION,
        task_input_cache_root=tmp_path / "closure-cache",
    )
    built = cache.ensure(
        family="organize-messy-files",
        item_id="organize-messy-files-1",
        agent_id="codex",
        runner=runner,
        trace_id="build-closure",
    )

    assert order == ["materialize", "build", "inspect"]
    assert built.task_input_closure_required is True
    assert built.task_input_closure_policy == TASK_INPUT_CLOSURE_POLICY_VERSION
    assert built.task_input_closure_hash == closure_hash
    assert built.task_input_build_context_receipt_hash == "b" * 64
    assert built.task_input_integrity_receipt_hash == "c" * 64
    assert built.task_input_integrity_container_network == "none"
    assert built.environment_hash != built.source_environment_hash

    order.clear()
    reused = SkillLearnPrebuiltImageCache(
        benchmark,
        task_input_closure_policy=TASK_INPUT_CLOSURE_POLICY_VERSION,
        task_input_cache_root=tmp_path / "closure-cache",
    ).ensure(
        family="organize-messy-files",
        item_id="organize-messy-files-1",
        agent_id="codex",
        runner=runner,
        trace_id="reuse-closure",
    )
    assert order == ["inspect"]
    assert reused.reused is True
    assert reused.task_input_integrity_receipt_hash == "c" * 64

    docker.images[built.tag]["Config"]["Labels"][
        "org.assumption-agent.prebuild.task-input-closure"
    ] = "d" * 64
    with pytest.raises(PermissionError, match="closure label"):
        SkillLearnPrebuiltImageCache(
            benchmark,
            task_input_closure_policy=TASK_INPUT_CLOSURE_POLICY_VERSION,
            task_input_cache_root=tmp_path / "closure-cache",
        ).ensure(
            family="organize-messy-files",
            item_id="organize-messy-files-1",
            agent_id="codex",
            runner=runner,
            trace_id="reject-mislabeled-closure",
        )


def test_prebuilt_cache_rejects_closure_after_source_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark, runner, docker = _closure_benchmark(tmp_path)
    environment = (
        benchmark
        / "tasks"
        / "organize-messy-files"
        / "organize-messy-files-1"
        / "environment"
    )
    manifest = {
        "closure_hash": "a" * 64,
        "manifest_hash": "a" * 64,
        "source_environment_hash": source_environment_hash(environment),
    }
    monkeypatch.setattr(
        skilllearn_lifecycle,
        "load_task_input_closure",
        lambda cache_root, family, item_id: manifest,
    )
    dockerfile = environment / "Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text(encoding="utf-8")
        + "# https://arxiv.org/pdf/changed-input.pdf\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskInputClosureError, match="source environment hash mismatch"):
        SkillLearnPrebuiltImageCache(
            benchmark,
            cache_only=False,
            task_input_closure_policy=TASK_INPUT_CLOSURE_POLICY_VERSION,
            task_input_cache_root=tmp_path / "closure-cache",
        ).ensure(
            family="organize-messy-files",
            item_id="organize-messy-files-1",
            agent_id="codex",
            runner=runner,
            trace_id="reject-source-drift",
        )
    assert docker.order == []


def test_closure_source_hash_survives_real_upstream_build_env_staging() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    benchmark = (
        repository_root
        / "reference"
        / "self_evo_continual_20260707"
        / "repos"
        / "SkillLearnBench"
    )
    source = (
        benchmark
        / "tasks"
        / "organize-messy-files"
        / "organize-messy-files-3"
        / "environment"
    )
    runner = SkillLearnSubprocessBackend(benchmark)._load_runner()
    staged = runner._prepare_build_env(source, "no_skill", None)
    try:
        assert (staged / "skills").is_dir()
        assert source_environment_hash(staged) == source_environment_hash(source)
        if str(staged).startswith("/mnt/c/"):
            assert (
                (staged / "Dockerfile").stat().st_mode & 0o777
            ) != ((source / "Dockerfile").stat().st_mode & 0o777)
    finally:
        shutil.rmtree(staged.parent, ignore_errors=True)


def _v5_receipt(
    manifest: SplitManifest,
) -> tuple[dict[str, Any], FrozenTaskInputClosure]:
    rows: list[dict[str, Any]] = []
    selected_ids = (
        *manifest.train_ids,
        *manifest.validation_ids,
        *manifest.test_ids,
    )
    for item_id in selected_ids:
        family = manifest.family_by_id[item_id]
        profile = offline_verifier_profile_for_family(family)
        closure_required = family in {
            "organize-messy-files",
            "stock-data-visualization",
        }
        rows.append(
            {
                "item_id_hash": stable_hash({"item_id": item_id}),
                "family_hash": stable_hash({"family": family}),
                "attempt_count": 1,
                "passed": True,
                "prebuilt_image_key": stable_hash({"image": item_id}),
                "prebuilt_image_id": "sha256:" + stable_hash({"image_id": item_id}),
                "agent_runtime_key": shared_codex_agent_runtime_key(),
                "agent_runtime_version": SHARED_CODEX_CLI_VERSION,
                "verifier_runtime_mode": (
                    "local_profile" if profile is not None else "native_image"
                ),
                "offline_verifier_profile_id": (
                    profile.profile_id if profile is not None else None
                ),
                "offline_verifier_profile_hash": (
                    profile.profile_hash if profile is not None else None
                ),
                "offline_verifier_runtime_key": (
                    offline_verifier_runtime_key(profile=profile)
                    if profile is not None
                    else None
                ),
                "verifier_runtime_network": "none",
                "error_type": None,
                "error_message_hash": None,
                "task_input_closure_required": closure_required,
                "task_input_closure_policy": (
                    TASK_INPUT_CLOSURE_POLICY_VERSION if closure_required else None
                ),
                "task_input_closure_hash": (
                    stable_hash({"closure": item_id}) if closure_required else None
                ),
                "task_input_integrity_receipt_hash": (
                    stable_hash({"integrity": item_id}) if closure_required else None
                ),
                "task_input_integrity_container_network": (
                    "none" if closure_required else None
                ),
            }
        )
    closure_rows = [row for row in rows if row["task_input_closure_required"]]
    profile_hashes = {
        row["offline_verifier_profile_hash"]
        for row in rows
        if row["offline_verifier_profile_hash"]
    }
    runtime_keys = {
        row["offline_verifier_runtime_key"]
        for row in rows
        if row["offline_verifier_runtime_key"]
    }
    receipt: dict[str, Any] = {
        "prewarm_version": TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION,
        "manifest_hash": manifest.manifest_hash,
        "split_names": ["train", "validation", "test"],
        "selected_item_set_hash": _selected_item_set_hash(manifest),
        "selected_item_count": len(rows),
        "completed_item_count": len(rows),
        "passed_item_count": len(rows),
        "failed_item_count": 0,
        "unique_image_count": len(rows),
        "parallel_workers": 3,
        "maximum_attempts": 1,
        "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
        "dependency_cache_only_enforced": True,
        "agent_runtime_policy": PREBUILT_IMAGE_POLICY_VERSION,
        "agent_runtime_key": shared_codex_agent_runtime_key(),
        "agent_runtime_version": SHARED_CODEX_CLI_VERSION,
        "codex_action_supervisor_policy": CODEX_ACTION_BUDGET_POLICY_VERSION,
        "codex_action_supervisor_sha256": codex_action_supervisor_hash(),
        "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
        "offline_verifier_runtime_network": "none",
        "offline_verifier_runtime_network_fallback_allowed": False,
        "local_profile_item_count": sum(
            row["verifier_runtime_mode"] == "local_profile" for row in rows
        ),
        "native_image_verifier_item_count": sum(
            row["verifier_runtime_mode"] == "native_image" for row in rows
        ),
        "unique_offline_verifier_profile_count": len(profile_hashes),
        "unique_offline_verifier_runtime_count": len(runtime_keys),
        "offline_verifier_profile_set_hash": stable_hash(sorted(profile_hashes)),
        "offline_verifier_runtime_set_hash": stable_hash(sorted(runtime_keys)),
        "online_build_attempted": False,
        "passed": True,
        "items": rows,
        "test_infrastructure_inspected": True,
        "sealed_test_scoring_performed": False,
        "sealed_test_bytes_exposed_to_model": False,
        "task_input_closure_policy": TASK_INPUT_CLOSURE_POLICY_VERSION,
        "task_input_build_context_policy": TASK_INPUT_BUILD_CONTEXT_POLICY_VERSION,
        "task_input_integrity_container_network": "none",
        "task_input_runtime_network_fallback_allowed": False,
        "task_input_closure_required_item_count": len(closure_rows),
        "task_input_closure_verified_item_count": len(closure_rows),
        "task_input_closure_set_hash": stable_hash(
            sorted(row["task_input_closure_hash"] for row in closure_rows)
        ),
        "task_input_integrity_receipt_set_hash": stable_hash(
            sorted(row["task_input_integrity_receipt_hash"] for row in closure_rows)
        ),
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    source = {
        "preparation_receipt_file_sha256": "1" * 64,
        "preparation_receipt_hash": "2" * 64,
        "closure_ledger_item_count": len(closure_rows),
        "closure_ledger_hash": "3" * 64,
        "content_object_count": len(closure_rows),
        "object_set_hash": "4" * 64,
    }
    frozen = FrozenTaskInputClosure(
        source=source,
        receipt={},
        receipt_path=Path("synthetic-preparation-receipt.json"),
        ledger_by_item_hash={
            str(row["item_id_hash"]): {
                "family_hash": row["family_hash"],
                "closure_hash": row["task_input_closure_hash"],
            }
            for row in closure_rows
        },
    )
    receipt.update(
        {
            "task_input_preparation_receipt_file_sha256": source[
                "preparation_receipt_file_sha256"
            ],
            "task_input_preparation_receipt_hash": source[
                "preparation_receipt_hash"
            ],
            "task_input_closure_ledger_item_count": source[
                "closure_ledger_item_count"
            ],
            "task_input_closure_ledger_hash": source["closure_ledger_hash"],
            "task_input_content_object_count": source[
                "content_object_count"
            ],
            "task_input_object_set_hash": source["object_set_hash"],
            "task_input_freeze_hash": frozen.freeze_hash,
        }
    )
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt, frozen


def test_v5_prewarm_receipt_requires_exact_task_input_integrity_evidence() -> None:
    manifest = SplitManifest(
        benchmark="synthetic",
        protocol="instance_holdout",
        seed="task-input-closure-v5",
        train_ids=("organize-messy-files-1",),
        validation_ids=("stock-data-visualization-1",),
        test_ids=("unaffected-1",),
        family_by_id={
            "organize-messy-files-1": "organize-messy-files",
            "stock-data-visualization-1": "stock-data-visualization",
            "unaffected-1": "unaffected-family",
        },
    )
    receipt, frozen = _v5_receipt(manifest)

    assert validate_development_prewarm_receipt(
        receipt,
        manifest=manifest,
        expected_version=TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION,
        frozen_task_inputs=frozen,
    ) == receipt["receipt_hash"]

    tampered_rows = [dict(row) for row in receipt["items"]]
    tampered_rows[0]["task_input_integrity_container_network"] = "default"
    tampered = {**receipt, "items": tampered_rows}
    tampered["receipt_hash"] = stable_hash(
        {key: value for key, value in tampered.items() if key != "receipt_hash"}
    )
    with pytest.raises(ValueError, match="integrity provenance"):
        validate_development_prewarm_receipt(
            tampered,
            manifest=manifest,
            expected_version=TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION,
            frozen_task_inputs=frozen,
        )


def test_runtime_cache_is_bound_to_the_validated_v5_closure_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            root
            / "manifests"
            / "skilllearn_paper_protocol_v3_19_ruoli_gpt54mini.json"
        ).read_text(encoding="utf-8")
    )
    frozen = load_frozen_task_input_closure(protocol, project_root=root)
    assert frozen is not None
    monkeypatch.setattr(
        prewarm_module,
        "verify_current_task_input_closure",
        lambda frozen, cache_root=None: tmp_path,
    )
    item_id = "organize-messy-files-1"
    family = "organize-messy-files"
    item_hash = stable_hash({"item_id": item_id})
    closure_hash = frozen.ledger_by_item_hash[item_hash]["closure_hash"]
    image = SimpleNamespace(
        task_input_closure_policy=TASK_INPUT_CLOSURE_POLICY_VERSION,
        task_input_closure_hash=closure_hash,
        cache_key="image-key",
        image_id="sha256:image-id",
        task_input_integrity_receipt_hash="a" * 64,
        task_input_integrity_container_network="none",
    )
    monkeypatch.setattr(
        SkillLearnPrebuiltImageCache,
        "ensure",
        lambda self, **kwargs: image,
    )
    expected_row = {
        "item_id_hash": item_hash,
        "family_hash": stable_hash({"family": family}),
        "task_input_closure_hash": closure_hash,
        "prebuilt_image_key": image.cache_key,
        "prebuilt_image_id": image.image_id,
        "task_input_integrity_receipt_hash": (
            image.task_input_integrity_receipt_hash
        ),
    }
    cache = FrozenTaskInputPrebuiltImageCache(
        tmp_path,
        frozen_task_inputs=frozen,
        expected_prewarm_rows={item_hash: expected_row},
    )
    assert cache.ensure(
        family=family,
        item_id=item_id,
        agent_id="codex",
        runner=SimpleNamespace(),
        trace_id="bound-runtime",
    ) is image

    drifted = dict(expected_row)
    drifted["prebuilt_image_id"] = "sha256:different"
    cache = FrozenTaskInputPrebuiltImageCache(
        tmp_path,
        frozen_task_inputs=frozen,
        expected_prewarm_rows={item_hash: drifted},
    )
    with pytest.raises(PermissionError, match="validated v5 prewarm row"):
        cache.ensure(
            family=family,
            item_id=item_id,
            agent_id="codex",
            runner=SimpleNamespace(),
            trace_id="drifted-runtime",
        )
