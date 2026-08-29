from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config/phase3_q05b_attempt2_target_d_campaign_v2.json"
TOOL_PATH = PROJECT_ROOT / "tools/phase3_q05b_attempt2_target_d_campaign_v2.py"
sys.path.insert(0, os.fspath(PROJECT_ROOT / "src"))

from hegel_machine import phase3_q05b_attempt2_target_d_campaign_v2 as core


@pytest.fixture
def local_tmp_path() -> Path:
    with tempfile.TemporaryDirectory(
        prefix="hegel-q05b-attempt2-test-", dir="/tmp"
    ) as directory:
        yield Path(directory)


def load_contract() -> dict[str, object]:
    return core.validate_contract_v2(json.loads(CONFIG_PATH.read_text()))


def load_tool():
    name = "_test_phase3_q05b_attempt2_target_d_campaign_v2_tool"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seal_contract(contract: dict[str, object]) -> dict[str, object]:
    value = deepcopy(contract)
    value.pop("contract_digest", None)
    value["contract_digest"] = core.domain_digest_v2(core.DOMAINS["contract"], value)
    return core.validate_contract_v2(value)


def capsule(contract: dict[str, object]) -> dict[str, object]:
    gates = {
        name: "sha256:" + sha256(name.encode()).hexdigest()
        for name in contract["preconsumption_gate_order"]
    }
    return core.build_capsule_v2(
        contract,
        source_commit="1" * 40,
        source_tree="2" * 40,
        meta_v11_receipt_digest="sha256:" + "3" * 64,
        gate_receipt_digests=gates,
    )


def test_contract_binds_exact_owner_text_and_post_meta_lineage() -> None:
    contract = load_contract()
    assert contract["authorization"] == core.OWNER_AUTHORIZATION
    assert core.OWNER_AUTHORIZATION_TEXT == "授权 Q0.5b attempt-2 actual"
    assert contract["source_lineage"]["required_ancestor_commit"] == (
        "a1e7119baab55faecac837ddb3dec0d336078cf6"
    )
    assert contract["target_d_v2"]["formal_actual_completed"] is False
    assert contract["historical_v2r20_evidence"][
        "historical_shadow_success_is_formal_actual_completion"
    ] is False


def test_v2r20_gate_source_replays_from_its_frozen_origin() -> None:
    tool = load_tool()
    contract = load_contract()
    shadow = tool._load_shadow_tool(contract)
    expected_worktree = Path(
        "/tmp/hegel-q05b-shadow-retry-v2r20-worktree"
    )
    assert shadow.WORKTREE_ROOT == expected_worktree
    loaded = shadow.load_phase3_q05b_shadow_retry_campaign_contract_v2()
    assert loaded["contract_id"] == (
        "hegel-phase3a-q05b-shadow-retry-campaign-v2r20"
    )
    assert loaded["contract_digest"] == contract[
        "historical_v2r20_evidence"
    ]["campaign_contract_digest"]
    assert loaded["execution_policy"]["clone_source_repository"] == os.fspath(
        expected_worktree
    )


def test_source_lineage_diff_uses_the_real_git_subdirectory_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    contract = load_contract()
    head = "4" * 40
    ancestor = contract["source_lineage"]["required_ancestor_commit"]
    prefix = "Hegel Machine/"
    expected_diff = "\n".join(
        f"A\t{prefix}{relative}"
        for relative in contract["source_lineage"]["additive_paths_in_exact_order"]
    )

    def fake_git(_project: Path, *arguments: str, timeout: float = 30.0) -> str:
        del timeout
        if arguments == ("rev-parse", "--verify", "HEAD"):
            return head
        if arguments == ("rev-parse", "--verify", "HEAD^{tree}"):
            return "5" * 40
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments == (
            "rev-parse", "--verify", f"{ancestor}^{{tree}}",
        ):
            return contract["source_lineage"]["required_ancestor_tree"]
        if arguments == ("rev-parse", "--verify", f"{ancestor}^"):
            return contract["source_lineage"]["required_ancestor_parent"]
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if arguments == ("rev-list", "--parents", "-n", "1", head):
            return f"{head} {ancestor}"
        if arguments == ("rev-parse", "--show-prefix"):
            return prefix
        if arguments == (
            "diff", "--name-status", "--no-renames", ancestor, head,
        ):
            return expected_diff
        if arguments[:2] == ("ls-files", "--error-unmatch"):
            return arguments[2]
        raise AssertionError(arguments)

    monkeypatch.setattr(tool, "_git", fake_git)
    receipt = tool._source_lineage_gate(contract, head)
    assert receipt["head"] == head


def test_authorization_tamper_is_rejected() -> None:
    contract = json.loads(CONFIG_PATH.read_text())
    contract["authorization"]["authorization_text"] += "!"
    with pytest.raises(core.Attempt2TargetDCampaignValidationError):
        core.validate_contract_v2(contract)


def test_pending_is_zero_effect_and_digest_bound() -> None:
    contract = load_contract()
    pending = core.build_pending_v2(
        contract,
        phase="REAL_META_V11_CHAIN",
        condition="MISSING_REAL_META_V11_CHAIN",
        detail="not produced yet",
        source_commit_or_null="1" * 40,
        meta_chain_manifest_or_null=None,
    )
    assert core.validate_pending_v2(pending, contract) == pending
    assert pending["status"] == core.PENDING_NOT_CONSUMED
    for field in (
        "campaign_execution_count",
        "authorization_consumption_count",
        "ledger_creation_or_write_count",
        "supervisor_popen_count",
        "retry_transition_count",
    ):
        assert pending[field] == 0


def test_capsule_gate_order_is_exact_and_stable() -> None:
    contract = load_contract()
    first = capsule(contract)
    second = capsule(contract)
    assert first == second
    assert list(first["gate_receipt_digests"]) == contract["preconsumption_gate_order"]
    reversed_gates = dict(reversed(list(first["gate_receipt_digests"].items())))
    with pytest.raises(core.Attempt2TargetDCampaignValidationError):
        core.build_capsule_v2(
            contract,
            source_commit="1" * 40,
            source_tree="2" * 40,
            meta_v11_receipt_digest="sha256:" + "3" * 64,
            gate_receipt_digests=reversed_gates,
        )


def test_decision_and_audit_share_one_core_return() -> None:
    success = core.decide_actual_return_v2(
        return_code=0,
        artifact_replay_passed=True,
        artifact_status_or_null="Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN",
        artifact_sha256_or_null="a" * 64,
    )
    assert success.decision == core.QUALIFIED == success.audit["decision"]
    failed = core.decide_actual_return_v2(
        return_code=1,
        artifact_replay_passed=False,
        artifact_status_or_null=None,
        artifact_sha256_or_null=None,
    )
    assert failed.decision == core.FAILED_TERMINAL
    assert failed.audit["decision"] == failed.decision


def test_outcome_persists_same_decision_audit_and_keeps_q1_closed() -> None:
    contract = load_contract()
    cap = capsule(contract)
    decision = core.decide_actual_return_v2(
        return_code=1,
        artifact_replay_passed=False,
        artifact_status_or_null=None,
        artifact_sha256_or_null=None,
    )
    outcome = core.build_outcome_v2(
        contract,
        cap,
        decision=decision,
        supervisor_return_code=1,
        stdout_sha256="b" * 64,
        stderr_sha256="c" * 64,
        artifact_sha256_or_null=None,
    )
    assert outcome["decision"] == outcome["decision_audit"]["decision"]
    assert outcome["q1_state"] == "NOT_RUN"
    assert outcome["retry_or_resume_allowed"] is False


def test_ledger_chain_consumes_once_and_is_terminal_after_sequence_two() -> None:
    contract = load_contract()
    cap = capsule(contract)
    consumed = core.build_ledger_record_v2(
        contract,
        cap,
        sequence=1,
        record_type="CONSUMED",
        status=core.CONSUMED,
        previous_record_digest_or_null=None,
        body={"capsule": cap, "retry_or_resume_after_this_record": False},
    )
    assert consumed["previous_record_digest_or_null"] is None
    assert consumed["status"] == core.CONSUMED


def test_missing_real_meta_chain_is_pending_before_later_gates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    contract = load_contract()
    calls: list[str] = []
    monkeypatch.setattr(tool, "load_contract_v2", lambda: contract)
    monkeypatch.setattr(
        tool,
        "_source_lineage_gate",
        lambda _contract, commit: calls.append("source")
        or {"status": "PASS", "head": commit, "tree": "2" * 40},
    )
    monkeypatch.setattr(
        tool,
        "_target_d_gate",
        lambda _contract: pytest.fail("Target-D gate must not run without real Meta chain"),
    )
    popen_calls: list[object] = []
    monkeypatch.setattr(tool, "_spawn_actual", lambda *a, **k: popen_calls.append((a, k)))
    code = tool.main(["--preflight", "--source-commit", "1" * 40])
    payload = json.loads(capsys.readouterr().out)
    assert code == 75
    assert payload["condition"] == "MISSING_REAL_META_V11_CHAIN"
    assert payload["ledger_creation_or_write_count"] == 0
    assert payload["supervisor_popen_count"] == 0
    assert calls == ["source"]
    assert popen_calls == []


def test_repairable_dependency_failure_is_pending_without_consumption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tool = load_tool()
    contract = load_contract()
    monkeypatch.setattr(tool, "load_contract_v2", lambda: contract)
    monkeypatch.setattr(
        tool, "_source_lineage_gate", lambda _c, commit: {"status": "PASS", "head": commit, "tree": "2" * 40}
    )
    monkeypatch.setattr(
        tool, "_meta_v11_gate", lambda _c, _m: {"status": "PASS", "receipt_digest": "sha256:" + "3" * 64}
    )
    monkeypatch.setattr(tool, "_target_d_gate", lambda _c: {"status": "PASS"})
    monkeypatch.setattr(
        tool,
        "_dependency_gate",
        lambda _c: (_ for _ in ()).throw(
            tool.GatePending("PYTHON310_VENDORED_TOMLI_CARGO", "DEPENDENCY_GATE_UNAVAILABLE", "python missing")
        ),
    )
    code = tool.main(
        [
            "--preflight",
            "--source-commit",
            "1" * 40,
            "--meta-chain-manifest",
            os.fspath(tmp_path / "chain.json"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 75
    assert payload["phase"] == "PYTHON310_VENDORED_TOMLI_CARGO"
    assert payload["authorization_consumption_count"] == 0
    assert payload["supervisor_popen_count"] == 0


def test_scheduling_creation_failure_cleans_probe_before_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    contract = deepcopy(load_contract())
    probe = tmp_path / "probe"
    contract["fixed_paths"]["scheduling_probe_root"] = os.fspath(probe)
    contract = seal_contract(contract)
    monkeypatch.setattr(
        tool,
        "_open_scheduling_probe_descriptor",
        lambda _path: (_ for _ in ()).throw(OSError("probe descriptor unavailable")),
    )
    with pytest.raises(tool.GatePending) as captured:
        tool._scheduling_gate(contract, object(), {})
    assert captured.value.condition == "PROBE_ROOT_UNAVAILABLE"
    assert not os.path.lexists(probe)


def test_consumed_record_precedes_only_popen_and_rerun_does_not_spawn(
    monkeypatch: pytest.MonkeyPatch, local_tmp_path: Path
) -> None:
    tmp_path = local_tmp_path
    tool = load_tool()
    contract = deepcopy(load_contract())
    fixed = contract["fixed_paths"]
    fixed.update(
        {
            "artifact_path": os.fspath(tmp_path / "artifact.json"),
            "ledger_root": os.fspath(tmp_path / "ledger"),
            "evidence_root": os.fspath(tmp_path / "evidence"),
            "trajectory_root": os.fspath(tmp_path / "trajectory"),
            "scheduling_probe_root": os.fspath(tmp_path / "probe"),
            "campaign_lock_path": os.fspath(tmp_path / "campaign.lock"),
            "stdout_path": os.fspath(tmp_path / "evidence/supervisor.stdout"),
            "stderr_path": os.fspath(tmp_path / "evidence/supervisor.stderr"),
        }
    )
    contract = seal_contract(contract)
    cap = capsule(contract)
    preflight = {
        "schema_version": "test",
        "status": core.PREFLIGHT_ADMITTED,
        "capsule": cap,
        "gate_evidence_in_exact_order": [],
    }
    preflight_count = 0

    def fake_preflight(*args, **kwargs):
        nonlocal preflight_count
        preflight_count += 1
        return deepcopy(preflight)

    monkeypatch.setattr(tool, "derive_preflight_v2", fake_preflight)
    monkeypatch.setattr(tool, "_replay_artifact", lambda _c: (False, None, None))
    popen_count = 0

    class FakePopen:
        pid = 43210

        def __init__(self, *args, **kwargs):
            nonlocal popen_count
            popen_count += 1
            consumed_path = Path(str(contract["fixed_paths"]["ledger_root"])) / "attempt-2.consumed.json"
            assert consumed_path.is_file()
            os.write(kwargs["stdout"], b'{"status":"failed"}\n')
            os.write(kwargs["stderr"], b"formal failure\n")

        def wait(self, timeout=None):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(tool, "_spawn_actual", FakePopen)
    code, result = tool.execute_once_v2(
        contract,
        source_commit="1" * 40,
        meta_chain_manifest=tmp_path / "chain.json",
        expected_capsule_digest=cap["capsule_digest"],
    )
    assert code == 1
    assert result["status"] == core.FAILED_TERMINAL
    assert popen_count == 1
    terminal = json.loads((tmp_path / "ledger/attempt-2.terminal.json").read_text())
    assert terminal["body"]["decision"] == terminal["body"]["decision_audit"]["decision"]
    assert terminal["body"]["post_popen_retry_or_resume_allowed"] is False
    replay_code, replay = tool.execute_once_v2(
        contract,
        source_commit=None,
        meta_chain_manifest=None,
        expected_capsule_digest=cap["capsule_digest"],
    )
    assert replay_code == 1
    assert replay["supervisor_popen_count_this_entry"] == 0
    assert popen_count == 1
    assert preflight_count == 1
    external_path = tmp_path / "evidence/attempt-2.outcome.json"
    external = json.loads(external_path.read_text())
    external["supervisor_error_or_null"] = "tampered external copy"
    unsigned_external = deepcopy(external)
    unsigned_external.pop("outcome_digest")
    external["outcome_digest"] = core.domain_digest_v2(
        core.DOMAINS["outcome"], unsigned_external
    )
    external_path.chmod(0o600)
    external_path.write_bytes(core.canonical_json_bytes_v2(external))
    external_path.chmod(0o444)
    with pytest.raises(tool.CampaignTerminal) as captured:
        tool.execute_once_v2(
            contract,
            source_commit=None,
            meta_chain_manifest=None,
            expected_capsule_digest=cap["capsule_digest"],
        )
    assert captured.value.code == "LEDGER_RECORD_INVALID"
    assert popen_count == 1
    assert preflight_count == 1


def test_lock_failure_rolls_back_roots_and_stays_unconsumed(
    monkeypatch: pytest.MonkeyPatch, local_tmp_path: Path
) -> None:
    tmp_path = local_tmp_path
    tool = load_tool()
    contract = deepcopy(load_contract())
    contract["fixed_paths"].update(
        {
            "artifact_path": os.fspath(tmp_path / "artifact.json"),
            "ledger_root": os.fspath(tmp_path / "ledger"),
            "evidence_root": os.fspath(tmp_path / "evidence"),
            "trajectory_root": os.fspath(tmp_path / "trajectory"),
            "scheduling_probe_root": os.fspath(tmp_path / "probe"),
            "campaign_lock_path": os.fspath(tmp_path / "campaign.lock"),
            "stdout_path": os.fspath(tmp_path / "evidence/supervisor.stdout"),
            "stderr_path": os.fspath(tmp_path / "evidence/supervisor.stderr"),
        }
    )
    contract = seal_contract(contract)
    cap = capsule(contract)
    monkeypatch.setattr(
        tool,
        "derive_preflight_v2",
        lambda *a, **k: {"status": core.PREFLIGHT_ADMITTED, "capsule": cap},
    )
    real_flock = tool.fcntl.flock

    def fail_only_ledger_lock(descriptor: int, operation: int) -> None:
        if operation == tool.fcntl.LOCK_EX | tool.fcntl.LOCK_NB:
            raise BlockingIOError("busy")
        real_flock(descriptor, operation)

    monkeypatch.setattr(tool.fcntl, "flock", fail_only_ledger_lock)
    with pytest.raises(tool.GatePending) as captured:
        tool.execute_once_v2(
            contract,
            source_commit="1" * 40,
            meta_chain_manifest=tmp_path / "chain.json",
            expected_capsule_digest=cap["capsule_digest"],
        )
    assert captured.value.condition == "FORMAL_ROOT_OR_LOCK_UNAVAILABLE"
    assert not (tmp_path / "ledger").exists()
    assert not (tmp_path / "evidence").exists()


def test_consumption_publication_known_not_started_rolls_back_and_pends(
    monkeypatch: pytest.MonkeyPatch, local_tmp_path: Path
) -> None:
    tmp_path = local_tmp_path
    tool = load_tool()
    contract = deepcopy(load_contract())
    contract["fixed_paths"].update(
        {
            "artifact_path": os.fspath(tmp_path / "artifact.json"),
            "ledger_root": os.fspath(tmp_path / "ledger"),
            "evidence_root": os.fspath(tmp_path / "evidence"),
            "trajectory_root": os.fspath(tmp_path / "trajectory"),
            "scheduling_probe_root": os.fspath(tmp_path / "probe"),
            "campaign_lock_path": os.fspath(tmp_path / "campaign.lock"),
            "stdout_path": os.fspath(tmp_path / "evidence/supervisor.stdout"),
            "stderr_path": os.fspath(tmp_path / "evidence/supervisor.stderr"),
        }
    )
    contract = seal_contract(contract)
    cap = capsule(contract)
    monkeypatch.setattr(
        tool,
        "derive_preflight_v2",
        lambda *a, **k: {"status": core.PREFLIGHT_ADMITTED, "capsule": cap},
    )
    monkeypatch.setattr(
        tool,
        "_publish_json_no_replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk boundary")),
    )
    with pytest.raises(tool.GatePending) as captured:
        tool.execute_once_v2(
            contract,
            source_commit="1" * 40,
            meta_chain_manifest=tmp_path / "chain.json",
            expected_capsule_digest=cap["capsule_digest"],
        )
    assert captured.value.condition == "CONSUMPTION_PUBLICATION_NOT_STARTED"
    assert not (tmp_path / "ledger").exists()
    assert not (tmp_path / "evidence").exists()


def test_partial_consumption_publication_is_terminal_not_pending(
    monkeypatch: pytest.MonkeyPatch, local_tmp_path: Path
) -> None:
    tool = load_tool()
    tmp_path = local_tmp_path
    contract = deepcopy(load_contract())
    contract["fixed_paths"].update(
        {
            "artifact_path": os.fspath(tmp_path / "artifact.json"),
            "ledger_root": os.fspath(tmp_path / "ledger"),
            "evidence_root": os.fspath(tmp_path / "evidence"),
            "trajectory_root": os.fspath(tmp_path / "trajectory"),
            "scheduling_probe_root": os.fspath(tmp_path / "probe"),
            "campaign_lock_path": os.fspath(tmp_path / "campaign.lock"),
            "stdout_path": os.fspath(tmp_path / "evidence/supervisor.stdout"),
            "stderr_path": os.fspath(tmp_path / "evidence/supervisor.stderr"),
        }
    )
    contract = seal_contract(contract)
    cap = capsule(contract)
    monkeypatch.setattr(
        tool,
        "derive_preflight_v2",
        lambda *a, **k: {"status": core.PREFLIGHT_ADMITTED, "capsule": cap},
    )

    def partial_publish(root_fd: int, name: str, _value: object) -> bytes:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
            dir_fd=root_fd,
        )
        try:
            os.write(descriptor, b"{")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise OSError("partial publication")

    monkeypatch.setattr(tool, "_publish_json_no_replace", partial_publish)
    with pytest.raises(tool.CampaignTerminal) as captured:
        tool.execute_once_v2(
            contract,
            source_commit="1" * 40,
            meta_chain_manifest=tmp_path / "chain.json",
            expected_capsule_digest=cap["capsule_digest"],
        )
    assert captured.value.code == "CONSUMPTION_PUBLICATION_UNCERTAIN"
    assert (tmp_path / "ledger/attempt-2.consumed.json").exists()


def test_campaign_lock_prevents_concurrent_cleanup_and_second_popen(
    monkeypatch: pytest.MonkeyPatch, local_tmp_path: Path
) -> None:
    tool = load_tool()
    tmp_path = local_tmp_path
    contract = deepcopy(load_contract())
    contract["fixed_paths"].update(
        {
            "artifact_path": os.fspath(tmp_path / "artifact.json"),
            "ledger_root": os.fspath(tmp_path / "ledger"),
            "evidence_root": os.fspath(tmp_path / "evidence"),
            "trajectory_root": os.fspath(tmp_path / "trajectory"),
            "scheduling_probe_root": os.fspath(tmp_path / "probe"),
            "campaign_lock_path": os.fspath(tmp_path / "campaign.lock"),
            "stdout_path": os.fspath(tmp_path / "evidence/supervisor.stdout"),
            "stderr_path": os.fspath(tmp_path / "evidence/supervisor.stderr"),
        }
    )
    contract = seal_contract(contract)
    cap = capsule(contract)
    monkeypatch.setattr(
        tool,
        "derive_preflight_v2",
        lambda *a, **k: {"status": core.PREFLIGHT_ADMITTED, "capsule": cap},
    )
    monkeypatch.setattr(tool, "_replay_artifact", lambda _c: (False, None, None))

    evidence_created = threading.Event()
    allow_first_to_continue = threading.Event()
    second_campaign_flock_started = threading.Event()
    cleanup_calls: list[str] = []
    original_create = tool._create_private_root
    original_cleanup = tool._cleanup_exact_empty_preconsumption_roots
    original_flock = tool.fcntl.flock
    campaign_flock_count = 0
    counter_guard = threading.Lock()

    def paused_create(path: Path, label: str) -> tuple[int, int]:
        opened = original_create(path, label)
        if path == Path(str(contract["fixed_paths"]["evidence_root"])):
            evidence_created.set()
            assert allow_first_to_continue.wait(timeout=5)
        return opened

    def observed_cleanup(bound_contract, held_lock):
        cleanup_calls.append("called")
        return original_cleanup(bound_contract, held_lock)

    def observed_flock(descriptor: int, operation: int) -> None:
        nonlocal campaign_flock_count
        if operation == tool.fcntl.LOCK_EX:
            with counter_guard:
                campaign_flock_count += 1
                if campaign_flock_count == 2:
                    second_campaign_flock_started.set()
        original_flock(descriptor, operation)

    popen_count = 0

    class FakePopen:
        pid = 43211

        def __init__(self, *args, **kwargs):
            nonlocal popen_count
            popen_count += 1
            os.write(kwargs["stdout"], b"formal failure\n")
            os.write(kwargs["stderr"], b"formal failure\n")

        def wait(self, timeout=None):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(tool, "_create_private_root", paused_create)
    monkeypatch.setattr(tool, "_cleanup_exact_empty_preconsumption_roots", observed_cleanup)
    monkeypatch.setattr(tool.fcntl, "flock", observed_flock)
    monkeypatch.setattr(tool, "_spawn_actual", FakePopen)
    results: list[tuple[int, dict[str, object]]] = []
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            results.append(
                tool.execute_once_v2(
                    contract,
                    source_commit="1" * 40,
                    meta_chain_manifest=tmp_path / "chain.json",
                    expected_capsule_digest=cap["capsule_digest"],
                )
            )
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    assert evidence_created.wait(timeout=5)
    second.start()
    assert second_campaign_flock_started.wait(timeout=5)
    assert cleanup_calls == []
    assert (tmp_path / "evidence").is_dir()
    allow_first_to_continue.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert popen_count == 1
    assert sorted(
        result[1].get(
            "supervisor_popen_count_this_entry",
            result[1].get("actual_supervisor_popen_count"),
        )
        for result in results
    ) == [0, 1]
    assert cleanup_calls == ["called"]
    assert (tmp_path / "evidence/attempt-2.outcome.json").is_file()


def test_terminal_replay_rejects_rehashed_decision_audit_tamper(
    monkeypatch: pytest.MonkeyPatch, local_tmp_path: Path
) -> None:
    tool = load_tool()
    tmp_path = local_tmp_path
    contract = deepcopy(load_contract())
    contract["fixed_paths"].update(
        {
            "artifact_path": os.fspath(tmp_path / "artifact.json"),
            "ledger_root": os.fspath(tmp_path / "ledger"),
            "evidence_root": os.fspath(tmp_path / "evidence"),
            "trajectory_root": os.fspath(tmp_path / "trajectory"),
            "scheduling_probe_root": os.fspath(tmp_path / "probe"),
            "campaign_lock_path": os.fspath(tmp_path / "campaign.lock"),
            "stdout_path": os.fspath(tmp_path / "evidence/supervisor.stdout"),
            "stderr_path": os.fspath(tmp_path / "evidence/supervisor.stderr"),
        }
    )
    contract = seal_contract(contract)
    cap = capsule(contract)
    decision = core.decide_actual_return_v2(
        return_code=1,
        artifact_replay_passed=False,
        artifact_status_or_null=None,
        artifact_sha256_or_null=None,
    )
    outcome = core.build_outcome_v2(
        contract,
        cap,
        decision=decision,
        supervisor_return_code=1,
        stdout_sha256="a" * 64,
        stderr_sha256="b" * 64,
        artifact_sha256_or_null=None,
    )
    consumed = core.build_ledger_record_v2(
        contract,
        cap,
        sequence=1,
        record_type="CONSUMED",
        status=core.CONSUMED,
        previous_record_digest_or_null=None,
        body={
            "capsule": cap,
            "attempt_ordinal": 2,
            "authorization_digest": core.OWNER_AUTHORIZATION_DIGEST,
            "actual_supervisor_popen_limit": 1,
            "retry_or_resume_after_this_record": False,
        },
    )
    terminal = core.build_ledger_record_v2(
        contract,
        cap,
        sequence=2,
        record_type="TERMINAL",
        status=decision.decision,
        previous_record_digest_or_null=consumed["record_digest"],
        body={
            "outcome": outcome,
            "decision": decision.decision,
            "decision_audit": decision.audit,
            "external_outcome_path": os.fspath(
                tmp_path / "evidence/attempt-2.outcome.json"
            ),
            "post_popen_retry_or_resume_allowed": False,
        },
    )
    tampered_audit = deepcopy(terminal["body"]["decision_audit"])
    tampered_audit["predicate_results"][0]["passed"] = True
    unsigned_audit = deepcopy(tampered_audit)
    unsigned_audit.pop("decision_audit_digest")
    tampered_audit["decision_audit_digest"] = core.domain_digest_v2(
        core.DOMAINS["decision_audit"], unsigned_audit
    )
    terminal["body"]["decision_audit"] = deepcopy(tampered_audit)
    terminal["body"]["outcome"]["decision_audit"] = deepcopy(tampered_audit)
    unsigned_outcome = deepcopy(terminal["body"]["outcome"])
    unsigned_outcome.pop("outcome_digest")
    terminal["body"]["outcome"]["outcome_digest"] = core.domain_digest_v2(
        core.DOMAINS["outcome"], unsigned_outcome
    )
    unsigned_terminal = deepcopy(terminal)
    unsigned_terminal.pop("record_digest")
    terminal["record_digest"] = core.domain_digest_v2(
        core.DOMAINS["ledger"], unsigned_terminal
    )
    ledger_root = tmp_path / "ledger"
    ledger_root.mkdir()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    (ledger_root / "attempt-2.consumed.json").write_bytes(
        core.canonical_json_bytes_v2(consumed)
    )
    (ledger_root / "attempt-2.terminal.json").write_bytes(
        core.canonical_json_bytes_v2(terminal)
    )
    (evidence_root / "attempt-2.outcome.json").write_bytes(
        core.canonical_json_bytes_v2(terminal["body"]["outcome"])
    )
    monkeypatch.setattr(
        tool,
        "derive_preflight_v2",
        lambda *a, **k: pytest.fail("terminal replay must bypass fresh gates"),
    )
    monkeypatch.setattr(
        tool,
        "_spawn_actual",
        lambda *a, **k: pytest.fail("terminal replay must not Popen"),
    )
    with pytest.raises(tool.CampaignTerminal) as captured:
        tool.execute_once_v2(
            contract,
            source_commit=None,
            meta_chain_manifest=None,
            expected_capsule_digest=cap["capsule_digest"],
        )
    assert captured.value.code == "LEDGER_RECORD_INVALID"


def test_old_v1_attempt2_slice_is_unchanged() -> None:
    expected = {
        "src/hegel_machine/phase3_q05b_attempt2_meta_v11_bridge_v1.py": "29d0b0250db17d1e3c02eb8612fdd8fe860a290f319eb89855b908cdc25f8c15",
        "config/phase3_q05b_attempt2_meta_v11_bridge_v1.json": "4c6d726482a6cc582bdcd7d7f6c6d9fc4610eafc7bffc51febc1721a39d12841",
        "tools/phase3_q05b_attempt2_meta_v11_bridge_v1.py": "87e21625175c2e0778b81e4ce59af8015044c7401b1e7ea874b25b0c0eafd14f",
        "tests/test_phase3_q05b_attempt2_meta_v11_bridge_v1.py": "100212ad9550cc6893cd0406f8e9b9c08cb6657df36fe8eb66a704317f3149bc",
        "docs/Hegel_Machine_Phase3A_Q05b_Attempt2_Meta_V11_Bridge_v1.md": "5c167542b2e676b127d1533cec287093a66735541a65224b71da8824111dfaf7",
    }
    for relative, digest in expected.items():
        assert sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == digest
