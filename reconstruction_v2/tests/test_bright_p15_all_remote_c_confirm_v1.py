from pathlib import Path
import sys

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_direct_c_confirm_v1 as p14,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p15_all_remote_c_confirm_v1 as runtime,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p15_extension_acquisition_v1 as acquisition,
)
from reconstruction_v2.replication_runtime.bright_p15_all_remote_v1 import (
    runner,
)


def _items() -> tuple[acquisition.RuntimeItem, ...]:
    rows = []
    ordinal = 0
    for family in runtime.FAMILIES:
        for attempt in range(acquisition.ATTEMPTS_PER_FAMILY):
            rows.append(
                acquisition.RuntimeItem(
                    ordinal=ordinal,
                    family=family,
                    attempt_ordinal=attempt,
                    family_hmac_position=72 + attempt,
                    item_key=f"{family}-{attempt}",
                    query="query",
                    source_query_id=f"{family}-q-{attempt}",
                    excluded_ids=(),
                )
            )
            ordinal += 1
    return tuple(rows)


def test_complete_case_selection_remains_first_ten_terminal_per_family() -> None:
    items = _items()
    converted = tuple(
        p14.RuntimeItem(
            ordinal=item.ordinal,
            family=item.family,
            attempt_ordinal=item.attempt_ordinal,
            family_hmac_position=item.family_hmac_position,
            item_key=item.item_key,
            query=item.query,
            source_query_id=item.source_query_id,
            excluded_ids=item.excluded_ids,
        )
        for item in items
    )
    terminal = tuple(
        item.ordinal for item in items if item.attempt_ordinal not in {0, 3}
    )
    capacity, selected, counts = p14.select_complete_cases(
        converted, terminal
    )
    assert capacity is True
    assert counts == {family: 18 for family in runtime.FAMILIES}
    assert {
        family: [
            item.attempt_ordinal for item in selected if item.family == family
        ]
        for family in runtime.FAMILIES
    } == {
        family: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
        for family in runtime.FAMILIES
    }


def test_minimal_worker_environment_does_not_forward_API_secrets(
    tmp_path: Path,
) -> None:
    environment = runner._minimal_environment(
        root=tmp_path, visible_gpu="0"
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert all(
        token not in key.upper()
        for key in environment
        for token in ("API", "OPENAI", "RUOLI", "TOKEN")
        if key != "TOKENIZERS_PARALLELISM"
    )


def test_network_audit_rejects_external_denied_attempts(
    tmp_path: Path,
) -> None:
    (tmp_path / "network.trace.1").write_text(
        'connect(3, {sa_family=AF_UNIX, sun_path="/run/x"}, 16) = -1 EPERM\n',
        encoding="ascii",
    )
    local = runner._network_audit(tmp_path, "network.trace")
    assert local["denied_external_network_syscall_count"] == 0
    (tmp_path / "network.trace.2").write_text(
        "connect(3, {sa_family=AF_INET}, 16) = -1 EPERM\n",
        encoding="ascii",
    )
    external = runner._network_audit(tmp_path, "network.trace")
    assert external["denied_external_network_syscall_count"] == 1


def test_strace_injection_does_not_require_newer_seccomp_acceleration(
    tmp_path: Path,
) -> None:
    command = runner._strace_command(["/bin/true"], tmp_path / "trace")
    assert "--seccomp-bpf" not in command
    assert "inject=connect,sendto,sendmsg,sendmmsg:error=EPERM" in command


def test_finalizer_does_not_open_labels_before_action_archive_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "reconstruction_v2"
    base.mkdir()
    monkeypatch.setattr(
        runtime,
        "load_acquisition",
        lambda _base: ({"self_sha256": "a" * 64}, _items()),
    )
    monkeypatch.setattr(
        runtime,
        "load_freeze",
        lambda _base, _root: {"self_sha256": "b" * 64},
    )
    opened = False

    def forbidden(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("label loader must not run")

    monkeypatch.setattr(runtime, "_load_selected_gold_ids", forbidden)
    with pytest.raises(runtime.P15AllRemoteError, match="remote action result"):
        runtime.finalize(tmp_path)
    assert opened is False


def test_remote_work_root_is_one_shot(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="ascii")
    with pytest.raises(Exception):
        runner.run(plan)


def test_remote_timeout_covers_the_frozen_cpu_openie_canary() -> None:
    assert runner.TIMEOUT_SECONDS == 3600


def test_p15_minilm_binding_verifies_model_and_fingerprints_live_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "asset.json"
    model = tmp_path / "model"
    manifest.write_text("{}", encoding="ascii")
    model.mkdir()
    monkeypatch.setattr(
        runner.minilm_asset,
        "_load_asset_manifest",
        lambda path: (path, {"asset": "verified"}),
    )
    monkeypatch.setattr(
        runner.minilm_asset, "_verify_manifest_contract", lambda _asset: None
    )
    monkeypatch.setattr(
        runner.minilm_asset,
        "_verify_model_tree",
        lambda _asset, path: path,
    )
    monkeypatch.setattr(
        runner.minilm_asset,
        "_PACKAGE_TO_MODULE",
        {"numpy": ("numpy", "numpy")},
    )
    receipt = runner._verify_p15_minilm_binding(
        asset_manifest_path=manifest, model_root=model
    )
    assert receipt["runtime_versions"] == {
        "numpy": runner.np.__version__,
        "python": ".".join(map(str, sys.version_info[:3])),
    }
    assert receipt["status"] == (
        "verified_P15_fingerprinted_runtime_with_immutable_model"
    )


def test_tree_receipt_binds_content_and_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="ascii")
    first = runner._tree_receipt(root)
    (root / "a.txt").write_text("beta", encoding="ascii")
    second = runner._tree_receipt(root)
    assert first["tree_sha256"] != second["tree_sha256"]
    (root / "link").symlink_to(root / "a.txt")
    with pytest.raises(runner.P15RemoteRuntimeError, match="symlink"):
        runner._tree_receipt(root)


def test_runtime_inventory_receipt_is_repeat_stable() -> None:
    assert runner._runtime_inventory_receipt() == runner._runtime_inventory_receipt()


def test_remote_freeze_binds_implementation_and_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementation = tmp_path / "implementation.py"
    dependency = tmp_path / "dependency.py"
    implementation.write_text("implementation", encoding="ascii")
    dependency.write_text("dependency", encoding="ascii")
    body = {
        "dependency_bindings": [
            {
                "relative_path": dependency.name,
                "sha256": runner._file_sha256(dependency),
            }
        ],
        "implementation_bindings": [
            {
                "relative_path": implementation.name,
                "sha256": runner._file_sha256(implementation),
            }
        ],
        "schema": "test_freeze",
    }
    freeze = {**body, "self_sha256": runner._stable_hash(body)}
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_bytes(runner._canonical_json_bytes(freeze))
    monkeypatch.setattr(runner.contract, "FREEZE_RELATIVE", Path("freeze.json"))
    monkeypatch.setattr(runner.contract, "FREEZE_SCHEMA", "test_freeze")
    runner._verify_remote_freeze(
        tmp_path, {"implementation_freeze_self_sha256": freeze["self_sha256"]}
    )
    dependency.write_text("drifted", encoding="ascii")
    with pytest.raises(runner.P15RemoteRuntimeError, match="file drifted"):
        runner._verify_remote_freeze(
            tmp_path,
            {"implementation_freeze_self_sha256": freeze["self_sha256"]},
        )
