from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace

import pytest

from assumption_agent.benchmarks.skilllearn_lifecycle import (
    PortableTaskCapabilityRuntimeContext,
    SkillLearnAgentTerminalError,
    SkillLearnSubprocessBackend,
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.benchmarks.runtime_profile_injection import (
    RUNTIME_PROFILE_PROMPT_CONTAINER_PATH,
    RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION,
)
from assumption_agent.benchmarks.typed_task_capability import (
    PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
    build_compiled_portable_task_capability,
    portable_role_spec_for_recipe,
)
from assumption_agent.events import MemoryEventSink
from assumption_agent.models import SplitName, stable_hash
from tests.test_portable_capability_compiler import (
    STOCK_FAMILY,
    _compile,
)
from tests.test_typed_task_capability import (
    ORGANIZE_FAMILY,
    _portable_graph_for_family,
)


BENCHMARK_ROOT = Path(
    "reference/self_evo_continual_20260707/repos/SkillLearnBench"
).resolve()
ITEM_ID = f"{STOCK_FAMILY}-3"
INPUT_LOCATOR = "/root/data/stock-descriptions.tsv"


def _copy_skills_to_dest(source: Path, destination: Path) -> bool:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    skills = sorted(
        path
        for path in source.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    )
    if not skills:
        return False
    for skill in skills:
        shutil.copytree(skill, destination / skill.name)
    return True


class FakeDockerSubprocess:
    """Small filesystem-backed Docker cp/exec seam; never starts a process."""

    def __init__(self, container_root: Path, *, tamper: str | None = None) -> None:
        self.container_root = container_root
        self.tamper = tamper
        self.commands: list[tuple[str, ...]] = []
        self.input_read_count = 0

    def container_path(self, value: str) -> Path:
        raw = value.split(":", 1)[1] if ":" in value else value
        raw = raw.removesuffix("/.")
        path = PurePosixPath(raw)
        assert path.is_absolute()
        return self.container_root.joinpath(*path.parts[1:])

    @staticmethod
    def _result(returncode: int = 0, *, stdout: str = ""):
        return SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    def _copy_from_container(self, source: str, destination: str) -> None:
        source_path = self.container_path(source)
        destination_path = Path(destination)
        if source.endswith("/."):
            destination_path.mkdir(parents=True, exist_ok=True)
            for child in source_path.iterdir():
                target = destination_path / child.name
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)
            return

        if source.split(":", 1)[1] == INPUT_LOCATOR:
            self.input_read_count += 1
            if self.tamper == "input" and self.input_read_count == 2:
                source_path.write_bytes(source_path.read_bytes() + b"tampered\n")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)
        if (
            self.tamper == "output_readback"
            and "/.assumption-v2/capabilities/" in source
        ):
            destination_path.write_bytes(
                destination_path.read_bytes() + b"tampered\n"
            )
        if (
            self.tamper == "prompt_readback"
            and RUNTIME_PROFILE_PROMPT_CONTAINER_PATH in source
        ):
            destination_path.write_bytes(
                destination_path.read_bytes() + b"tampered\n"
            )

    def _copy_to_container(self, source: str, destination: str) -> None:
        destination_path = self.container_path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(source), destination_path)

    def run(self, args, **_kwargs):
        command = tuple(str(value) for value in args)
        self.commands.append(command)
        if command[:2] == ("docker", "cp"):
            source, destination = command[2], command[3]
            if ":" in source:
                self._copy_from_container(source, destination)
            else:
                self._copy_to_container(source, destination)
            return self._result()
        if command[:3] != ("docker", "exec", "trial"):
            return self._result(1)
        operation = command[3:]
        if operation[:2] == ("mkdir", "-p"):
            self.container_path(operation[2]).mkdir(
                parents=True,
                exist_ok=True,
            )
            return self._result()
        if operation[:2] == ("chmod", "0444"):
            self.container_path(operation[2]).chmod(0o444)
            return self._result()
        if operation[:2] == ("test", "-f"):
            return self._result(
                0 if self.container_path(operation[2]).is_file() else 1
            )
        if operation[:2] == ("test", "-d"):
            return self._result(
                0 if self.container_path(operation[2]).is_dir() else 1
            )
        if operation[:3] == ("test", "!", "-L"):
            return self._result(
                0 if not self.container_path(operation[3]).is_symlink() else 1
            )
        if operation[:3] == ("test", "!", "-e"):
            return self._result(
                0 if not self.container_path(operation[3]).exists() else 1
            )
        if operation[:2] == ("find", "/root"):
            entries = []
            root = self.container_path("/root")
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(self.container_root)
                locator = str(PurePosixPath("/").joinpath(*relative.parts))
                kind = "l" if path.is_symlink() else (
                    "d" if path.is_dir() else "f"
                )
                entries.append(f"{kind}\t{locator}\x00")
            return self._result(stdout="".join(entries))
        return self._result(1)


def _runtime_fixture(
    tmp_path: Path,
    *,
    tamper: str | None = None,
    delivery_mode: str = "",
    run_template: str = (
        'codex exec --json --model {model} -- "$(cat {instruction_file})"'
    ),
):
    compiled, _, _ = _compile(tmp_path / "compile")
    source = compiled.source_for(ITEM_ID)
    assert source is not None
    source_receipt = compiled.source_receipt_for(ITEM_ID)
    role_hashes = compiled.item_portable_capability_role_spec_hashes[
        stable_hash({"item_id": ITEM_ID})
    ]
    request = SkillLearnTrialRequest(
        item_id=ITEM_ID,
        family=STOCK_FAMILY,
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_ON,
        evaluator_epoch="portable-runtime-fixture-v1",
        pair_id="portable-runtime-pair",
        repeat=1,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash=stable_hash({"manifest": "portable-runtime"}),
        program_id=compiled.hypothesis_ids[0],
        program_set_hash=compiled.program_set_hash,
        treatment_hash=compiled.treatment_hash_for(ITEM_ID),
        compile_manifest_hash=compiled.manifest_hash,
        skill_source_receipt_hash=source_receipt.receipt_hash,
        compile_root=compiled.output_root,
        typed_binding_set_hash=compiled.typed_binding_set_hash,
        typed_snapshot_hashes=compiled.typed_snapshot_hashes,
        typed_snapshot_ledger_hash=compiled.typed_snapshot_ledger_hash,
        portable_capability_compiler_mode=(
            compiled.portable_capability_compiler_mode
        ),
        portable_capability_role_spec_set_hash=(
            compiled.portable_capability_role_spec_set_hash
        ),
        portable_capability_role_spec_hashes=role_hashes,
        portable_capability_delivery_mode=delivery_mode,
    )
    sink = MemoryEventSink()
    backend = SkillLearnSubprocessBackend(
        BENCHMARK_ROOT,
        model="gpt-5.4-mini",
        provider_mode="openai_compatible",
        event_sink=sink,
    )
    context = backend._load_portable_task_capability_context(
        request=request,
        source_receipt=source_receipt,
        compile_root=compiled.output_root,
    )
    assert isinstance(context, PortableTaskCapabilityRuntimeContext)

    container_root = tmp_path / "container"
    input_path = container_root / "root/data/stock-descriptions.tsv"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(
        "ticker\tname\tsector\tmarketCap\n"
        "AAA\tAlpha\tTechnology\t1000\n"
        "BBB\tBeta\tFinance\t2000\n",
        encoding="utf-8",
    )
    delegate = FakeDockerSubprocess(container_root, tamper=tamper)
    runner = ModuleType("fake_portable_runtime_runner")
    runner.subprocess = delegate
    runner._copy_skills_to_dest = _copy_skills_to_dest
    agent = {"run": run_template}
    runner.get_agent = lambda agent_id: agent if agent_id == "codex" else None

    def inject(container_name, skill_source_dir, copies):
        assert container_name == "trial"
        for source_pattern, destination in copies:
            assert source_pattern == "skills"
            assert _copy_skills_to_dest(
                Path(skill_source_dir),
                delegate.container_path(destination),
            )

    runner._inject_skills_runtime = inject
    backend._install_treatment_receipt_adapter(runner)
    runner._assumption_v2_task_capability_context = context
    return backend, runner, delegate, source, context, sink, input_path


def test_trial_request_requires_complete_portable_provenance(
    tmp_path: Path,
) -> None:
    common = {
        "item_id": ITEM_ID,
        "family": STOCK_FAMILY,
        "split": SplitName.TRAIN,
        "variant": TrialVariant.POLICY_ON,
        "evaluator_epoch": "portable-runtime-fixture-v1",
        "pair_id": "portable-runtime-pair",
        "repeat": 1,
        "agent_id": "codex",
        "model": "gpt-5.4-mini",
        "max_steps": 100,
        "manifest_hash": "manifest",
        "compile_manifest_hash": "compile",
        "skill_source_receipt_hash": "source-receipt",
        "compile_root": tmp_path,
        "typed_binding_set_hash": "binding-set",
        "typed_snapshot_hashes": ("snapshot",),
        "typed_snapshot_ledger_hash": "ledger",
    }
    with pytest.raises(
        ValueError,
        match="portable capability compile provenance must be complete",
    ):
        SkillLearnTrialRequest(
            **common,
            portable_capability_compiler_mode=(
                PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
            ),
        )

    request = SkillLearnTrialRequest(
        **common,
        portable_capability_compiler_mode=(
            PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
        ),
        portable_capability_role_spec_set_hash="role-set",
        portable_capability_role_spec_hashes=(),
    )
    payload = request.to_dict()
    assert payload["portable_capability_compiler_mode"] == (
        PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
    )
    assert payload["portable_capability_role_spec_set_hash"] == "role-set"
    assert payload["portable_capability_role_spec_hashes"] == []
    assert "compile_root" not in payload

    with pytest.raises(ValueError, match="delivery mode is unsupported"):
        SkillLearnTrialRequest(
            **common,
            portable_capability_compiler_mode=(
                PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
            ),
            portable_capability_role_spec_set_hash="role-set",
            portable_capability_role_spec_hashes=(),
            portable_capability_delivery_mode="future-delivery-mode",
        )
    with pytest.raises(
        ValueError,
        match="requires a compiled policy-on treatment",
    ):
        SkillLearnTrialRequest(
            **{**common, "variant": TrialVariant.POLICY_OFF},
            portable_capability_compiler_mode=(
                PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
            ),
            portable_capability_role_spec_set_hash="role-set",
            portable_capability_role_spec_hashes=(),
            portable_capability_delivery_mode=(
                RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION
            ),
        )


def test_compiled_portable_capability_executes_before_agent_start(
    tmp_path: Path,
) -> None:
    (
        _backend,
        runner,
        delegate,
        source,
        context,
        sink,
        input_path,
    ) = _runtime_fixture(tmp_path)
    input_before = input_path.read_bytes()

    runner._inject_skills_runtime(
        "trial",
        source,
        [("skills", "/root/.codex/skills")],
    )

    # Raw item identity is available only in the ephemeral context.
    assert context.item_id == ITEM_ID
    assert context.family == STOCK_FAMILY
    assert "item_id" not in context.safe_payload()
    assert "family" not in context.safe_payload()
    assert ITEM_ID not in json.dumps(context.safe_payload(), sort_keys=True)
    assert context.item_id_hash == stable_hash({"item_id": ITEM_ID})

    effects = runner._assumption_v2_task_capability_effects
    assert isinstance(effects, tuple) and len(effects) == 1
    output_locator = context.metadata[0].output_container_locator
    output_path = delegate.container_path(output_locator)
    assert output_path.is_file()
    assert effects[0].agent_payload["profile_locator"] == output_locator
    assert input_path.read_bytes() == input_before

    allowed_agent_payload_keys = {
        "policy",
        "role",
        "artifact_format",
        "profile_locator",
        "effect_receipt_hash",
        "profile_created_before_agent_start",
        "source_artifact_locator_disclosed",
    }
    payload = dict(effects[0].agent_payload)
    assert set(payload) == allowed_agent_payload_keys
    assert payload["source_artifact_locator_disclosed"] is False
    payload_text = json.dumps(payload, sort_keys=True)
    assert INPUT_LOCATOR not in payload_text
    assert ITEM_ID not in payload_text

    installed = runner._assumption_v2_installed_skill_receipt
    assert installed["agent_started"] is False
    verified = [
        row
        for row in sink.events
        if row["event"] == "skilllearn_pre_agent_task_capability_verified"
    ]
    assert len(verified) == 1
    assert verified[0]["payload"]["agent_started"] is False
    assert verified[0]["payload"]["agent_payloads"] == [payload]
    assert INPUT_LOCATOR not in json.dumps(verified[0], sort_keys=True)


def test_runtime_profile_prompt_delivery_is_explicit_and_receipted(
    tmp_path: Path,
) -> None:
    (
        _backend,
        runner,
        delegate,
        source,
        context,
        sink,
        _input_path,
    ) = _runtime_fixture(
        tmp_path,
        delivery_mode=RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION,
    )
    original_template = runner.get_agent("codex")["run"]

    runner._inject_skills_runtime(
        "trial",
        source,
        [("skills", "/root/.codex/skills")],
    )

    receipt = runner._assumption_v2_runtime_profile_injection_receipt
    assert receipt.request_hash == context.request_hash
    assert receipt.context_hash == context.context_hash
    assert receipt.profile_count == 1
    assert receipt.safe_payload()["profile_present_in_effective_launch_prompt"] is True
    assert receipt.safe_payload()["semantic_consumption_claimed"] is False
    assert receipt.safe_payload()["task_effect_attributed"] is False
    assert runner.get_agent("codex")["run"] != original_template
    assert runner.get_agent("codex")["run"].count(
        RUNTIME_PROFILE_PROMPT_CONTAINER_PATH
    ) == 1

    fragment = delegate.container_path(
        RUNTIME_PROFILE_PROMPT_CONTAINER_PATH
    ).read_bytes()
    assert hashlib.sha256(fragment).hexdigest() == receipt.fragment_sha256
    assert b"ASSUMPTION_V2_VERIFIED_RUNTIME_CONTEXT" in fragment
    assert b'"record_count":2' in fragment
    assert INPUT_LOCATOR.encode("utf-8") not in fragment
    effects = runner._assumption_v2_task_capability_effects
    assert hashlib.sha256(effects[0].profile_bytes).hexdigest() == (
        effects[0].output_sha256
    )
    assert "profile_bytes" not in effects[0].safe_payload()

    injected = [
        row
        for row in sink.events
        if row["event"]
        == "skilllearn_pre_agent_runtime_profile_prompt_injected"
    ]
    assert len(injected) == 1
    assert injected[0]["payload"]["receipt_hash"] == receipt.receipt_hash
    assert injected[0]["payload"]["agent_started_at_receipt_time"] is False
    assert INPUT_LOCATOR not in json.dumps(injected[0], sort_keys=True)


def test_default_portable_runtime_does_not_change_agent_prompt(
    tmp_path: Path,
) -> None:
    (
        _backend,
        runner,
        delegate,
        source,
        _context,
        sink,
        _input_path,
    ) = _runtime_fixture(tmp_path)
    original_template = runner.get_agent("codex")["run"]

    runner._inject_skills_runtime(
        "trial",
        source,
        [("skills", "/root/.codex/skills")],
    )

    assert runner.get_agent("codex")["run"] == original_template
    assert runner._assumption_v2_runtime_profile_injection_receipt is None
    assert not delegate.container_path(
        RUNTIME_PROFILE_PROMPT_CONTAINER_PATH
    ).exists()
    assert not any(
        row["event"]
        == "skilllearn_pre_agent_runtime_profile_prompt_injected"
        for row in sink.events
    )


@pytest.mark.parametrize(
    ("tamper", "run_template"),
    (
        ("prompt_readback", 'codex exec -- "$(cat {instruction_file})"'),
        (None, "codex exec -- no-instruction-expansion"),
    ),
)
def test_runtime_profile_prompt_delivery_tamper_fails_before_agent(
    tmp_path: Path,
    tamper: str | None,
    run_template: str,
) -> None:
    (
        _backend,
        runner,
        _delegate,
        source,
        _context,
        sink,
        _input_path,
    ) = _runtime_fixture(
        tmp_path,
        tamper=tamper,
        delivery_mode=RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION,
        run_template=run_template,
    )

    with pytest.raises(
        SkillLearnAgentTerminalError,
        match="task_capability_prompt_delivery_invalid",
    ):
        runner._inject_skills_runtime(
            "trial",
            source,
            [("skills", "/root/.codex/skills")],
        )

    assert runner._assumption_v2_runtime_profile_injection_receipt is None
    blocked = [
        row
        for row in sink.events
        if row["event"]
        == "skilllearn_trial_blocked_invalid_runtime_profile_prompt"
    ]
    assert len(blocked) == 1
    assert blocked[0]["payload"]["agent_started"] is False
    assert blocked[0]["payload"]["model_invoked"] is False


@pytest.mark.parametrize("tamper", ("output_readback", "input"))
def test_pre_agent_capability_tamper_fails_closed(
    tmp_path: Path,
    tamper: str,
) -> None:
    (
        _backend,
        runner,
        _delegate,
        source,
        _context,
        sink,
        _input_path,
    ) = _runtime_fixture(tmp_path, tamper=tamper)

    with pytest.raises(
        SkillLearnAgentTerminalError,
        match="task_capability_pre_agent_invalid",
    ):
        runner._inject_skills_runtime(
            "trial",
            source,
            [("skills", "/root/.codex/skills")],
        )

    assert runner._assumption_v2_installed_skill_receipt is None
    assert runner._assumption_v2_task_capability_effects is None
    assert runner._assumption_v2_task_capability_agent_payloads is None
    blocked = [
        row
        for row in sink.events
        if row["event"]
        == "skilllearn_trial_blocked_invalid_pre_agent_task_capability"
    ]
    assert len(blocked) == 1
    assert blocked[0]["payload"]["agent_started"] is False
    assert blocked[0]["payload"]["model_invoked"] is False
    assert not any(
        row["event"] == "skilllearn_pre_agent_task_capability_verified"
        for row in sink.events
    )


@pytest.mark.parametrize("linked_ancestor", ("input", "output"))
def test_pre_agent_capability_rejects_linked_path_ancestor(
    tmp_path: Path,
    linked_ancestor: str,
) -> None:
    (
        _backend,
        runner,
        delegate,
        source,
        _context,
        sink,
        _input_path,
    ) = _runtime_fixture(tmp_path)
    if linked_ancestor == "input":
        data = delegate.container_root / "root/data"
        real_data = delegate.container_root / "root/data-real"
        data.rename(real_data)
        data.symlink_to(real_data, target_is_directory=True)
    else:
        assumption_root = delegate.container_root / "root/.assumption-v2"
        real_assumption_root = (
            delegate.container_root / "root/.assumption-v2-real"
        )
        real_assumption_root.mkdir(parents=True)
        assumption_root.symlink_to(
            real_assumption_root,
            target_is_directory=True,
        )

    with pytest.raises(
        SkillLearnAgentTerminalError,
        match="task_capability_pre_agent_invalid",
    ):
        runner._inject_skills_runtime(
            "trial",
            source,
            [("skills", "/root/.codex/skills")],
        )

    assert runner._assumption_v2_installed_skill_receipt is None
    assert runner._assumption_v2_task_capability_effects is None
    assert runner._assumption_v2_task_capability_agent_payloads is None
    assert any(
        row["event"]
        == "skilllearn_trial_blocked_invalid_pre_agent_task_capability"
        and row["payload"]["model_invoked"] is False
        for row in sink.events
    )


def test_pre_agent_capability_rechecks_public_instruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        backend,
        runner,
        _delegate,
        source,
        context,
        sink,
        _input_path,
    ) = _runtime_fixture(tmp_path)
    monkeypatch.setattr(
        backend,
        "_load_portable_public_instruction",
        lambda **_kwargs: context.public_instruction + "\ndrifted",
    )

    with pytest.raises(
        SkillLearnAgentTerminalError,
        match="task_capability_pre_agent_invalid",
    ):
        runner._inject_skills_runtime(
            "trial",
            source,
            [("skills", "/root/.codex/skills")],
        )

    assert runner._assumption_v2_installed_skill_receipt is None
    assert any(
        row["event"]
        == "skilllearn_trial_blocked_invalid_pre_agent_task_capability"
        and row["payload"]["agent_started"] is False
        for row in sink.events
    )


def test_pre_agent_office_collection_discovery_and_readback_are_read_only(
    tmp_path: Path,
) -> None:
    item_id = "organize-messy-files-3"
    graph = _portable_graph_for_family(ORGANIZE_FAMILY)
    recipe = next(
        row
        for row in graph.recipes
        if row.workflow.value == "organize_collection"
    )
    role_spec = portable_role_spec_for_recipe(graph, recipe.recipe_id)
    metadata = build_compiled_portable_task_capability(
        role_spec,
        item_id=item_id,
        program_id="portable-organize-fixture",
        typed_binding_hash="a" * 64,
        bound_recipe_hash="b" * 64,
    )
    instruction = (
        BENCHMARK_ROOT
        / "tasks"
        / ORGANIZE_FAMILY
        / item_id
        / "instruction.md"
    ).read_text(encoding="utf-8").strip()
    context = PortableTaskCapabilityRuntimeContext(
        request_hash="c" * 64,
        item_id_hash=stable_hash({"item_id": item_id}),
        source_receipt_hash="d" * 64,
        typed_binding_set_hash="e" * 64,
        role_spec_hashes=(role_spec.role_spec_hash,),
        metadata_file_hashes=(),
        metadata=(metadata,),
        public_instruction_hash=stable_hash(
            {"public_instruction": instruction}
        ),
        item_id=item_id,
        family=ORGANIZE_FAMILY,
        public_instruction=instruction,
    )
    container_root = tmp_path / "organize-container"
    collection = container_root / "root/papers/all"
    collection.mkdir(parents=True)
    (collection / "paper.pdf").write_bytes(b"%PDF fixture")
    (collection / "notes.docx").write_bytes(b"docx fixture")
    (collection / "slides.pptx").write_bytes(b"pptx fixture")
    before = {
        path.name: path.read_bytes()
        for path in sorted(collection.iterdir())
    }
    delegate = FakeDockerSubprocess(container_root)
    runner = ModuleType("fake_office_collection_runtime_runner")
    runner.subprocess = delegate
    backend = SkillLearnSubprocessBackend(
        BENCHMARK_ROOT,
        model="gpt-5.4-mini",
        provider_mode="openai_compatible",
    )

    effects = backend._execute_portable_task_capabilities_in_container(
        runner=runner,
        container_name="trial",
        context=context,
    )

    assert len(effects) == 1
    assert effects[0].role_spec_hash == role_spec.role_spec_hash
    assert metadata.output_container_locator == effects[0].agent_payload[
        "profile_locator"
    ]
    assert {
        path.name: path.read_bytes()
        for path in sorted(collection.iterdir())
    } == before
    assert delegate.container_path(
        metadata.output_container_locator
    ).is_file()
    assert any(
        command[3:5] == ("find", "/root")
        for command in delegate.commands
    )
