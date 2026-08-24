from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.machinery
import importlib.util
import inspect
import io
import itertools
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT.parent
RUNNER_PATH = ROOT / "tools/phase3_q05b_attempt2_meta_v11_bridge_v1.py"
CORE_PATH = ROOT / "src/hegel_machine/phase3_q05b_attempt2_meta_v11_bridge_v1.py"
CONTRACT_PATH = ROOT / "config/phase3_q05b_attempt2_meta_v11_bridge_v1.json"
DOC_PATH = ROOT / "docs/Hegel_Machine_Phase3A_Q05b_Attempt2_Meta_V11_Bridge_v1.md"
FROZEN_CONTRACT_DIGEST = (
    "sha256:e36ecb243a7f36d24c5a29f4fd6beb913c195848350f845ce615d8b0304535ed"
)


def _load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_old_dont = sys.dont_write_bytecode
_old_prefix = sys.pycache_prefix
RUNNER = _load_source("_q05b_attempt2_bridge_runner_kat", RUNNER_PATH)
# Do not leak the CLI's process-wide setting into unrelated pytest modules.
sys.dont_write_bytecode = _old_dont
sys.pycache_prefix = _old_prefix
CORE = _load_source("_q05b_attempt2_bridge_core_kat", CORE_PATH)
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _digest(value: object) -> str:
    return "sha256:" + sha256(_canonical(value)).hexdigest()


def _namespace() -> argparse.Namespace:
    values: dict[str, object] = {
        name: Path(f"/tmp/q05b-attempt2-{index:02d}.json")
        for index, (name, _flag) in enumerate(RUNNER.BRIDGE_INPUT_ROWS)
    }
    values.update(
        {
            "preflight": True,
            "run": False,
            "meta_project_root": RUNNER.META_ROOT,
            "hegel_project_root": RUNNER.FROZEN_PROJECT_ROOT,
            "source_commit": "a" * 40,
            "artifact": RUNNER.FROZEN_ARTIFACT,
            "cargo_cache_source": RUNNER.FROZEN_CARGO_CACHE,
            "ledger_root": RUNNER.FROZEN_LEDGER_ROOT,
        }
    )
    values.update(
        {
            f"expected_{name}_digest": "sha256:" + f"{index + 1:064x}"
            for index, name in enumerate(RUNNER.EXPECTED_DIGEST_NAMES)
        }
    )
    return argparse.Namespace(**values)


def _synthetic_actual_receipt() -> dict[str, object]:
    rows = {}
    for index, (relative, digest) in enumerate(
        CORE.HEGEL_UNCHANGED_RAW_SHA256.items(), start=1
    ):
        rows[relative] = {
            "path": CORE.PROJECT_ROOT + "/" + relative,
            "device": 7,
            "inode": index,
            "nlink": 1,
            "mode": stat.S_IFREG | 0o444,
            "size": index,
            "sha256": digest,
        }
    return {
        "policy": "SOURCE_ONLY_HEGEL_ACTUAL_IMPORT_AND_CALL",
        "preloaded_hegel_machine_modules": [
            "hegel_machine",
            "hegel_machine.phase3_q05b_attempt2_meta_v11_bridge_v1",
        ],
        "preloaded_modules_within_allowlist": True,
        "dont_write_bytecode_active": True,
        "pycache_prefix": RUNNER.HEGEL_PYCACHE_PREFIX,
        "standard_import_machinery_only": True,
        "source_file_loader_only_for_project": True,
        "actual_tool_source_origin": CORE.PROJECT_ROOT
        + "/tools/phase3_q05b_dual_qualification_v1.py",
        "actual_tool_raw_sha256": CORE.HEGEL_UNCHANGED_RAW_SHA256[
            "tools/phase3_q05b_dual_qualification_v1.py"
        ],
        "unchanged_source_file_evidence": rows,
        "run_actual_callable_identity_verified": True,
        "loaded_hegel_source_module_origins": {
            "hegel_machine.phase3_q05b_attempt2_meta_v11_bridge_v1": (
                CORE.PROJECT_ROOT
                + "/src/hegel_machine/phase3_q05b_attempt2_meta_v11_bridge_v1.py"
            ),
            "hegel_machine.phase3_q05b_actual_admission_v1": (
                CORE.PROJECT_ROOT
                + "/src/hegel_machine/phase3_q05b_actual_admission_v1.py"
            ),
        },
        "in_tree_bytecode_loaded": False,
        "path_hooks_and_importer_cache_restored": True,
        "interpreter_import_state_restored": True,
        "live_source_commit_tree_clean_raw_hash_checks_performed": True,
        "unchanged_source_path_hash_checks_performed": True,
    }


def _synthetic_meta_receipt() -> dict[str, object]:
    rows = {}
    for index, (relative, digest) in enumerate(
        CORE.META_EXACT_FIVE_RAW_SHA256.items(), start=101
    ):
        rows[relative] = {
            "path": CORE.META_SOURCE_BINDING_ROOT + "/" + relative,
            "device": 8,
            "inode": index,
            "nlink": 1,
            "mode": stat.S_IFREG | 0o444,
            "size": index,
            "sha256": digest,
        }
    return {
        "repo_root": CORE.META_SOURCE_BINDING_ROOT,
        "git_commit": CORE.META_SOURCE_BINDING["git_commit"],
        "git_tree": CORE.META_SOURCE_BINDING["git_tree"],
        "git_clean": True,
        "v11_contract_digest": CORE.META_V11_CONTRACT_DIGEST,
        "exact_five_file_evidence": rows,
        "commit_tree_clean_checks_performed": True,
        "exact_five_file_checks_performed": True,
        "v11_contract_digest_check_performed": True,
    }


def _synthetic_hegel_live_receipt() -> dict[str, object]:
    return {
        "project_root": CORE.PROJECT_ROOT,
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "clean": True,
        "baseline_commit": RUNNER.BASELINE_COMMIT,
        "baseline_tree": RUNNER.BASELINE_TREE,
        "source_parent_commit": RUNNER.BASELINE_COMMIT,
        "source_commit_count_after_baseline": 1,
        "source_direct_child_of_baseline": True,
        "attempt1_source_commit": RUNNER.ATTEMPT1_COMMIT,
        "repair_commit": RUNNER.REPAIR_COMMIT,
        "repair_is_ancestor": True,
        "exact_five_diff_paths": list(CORE.EXACT_FIVE_PATHS),
        "exact_five_file_evidence": {},
        "commit_tree_clean_checks_performed": True,
        "direct_child_and_commit_count_checks_performed": True,
        "attempt1_repair_lineage_checks_performed": True,
        "exact_five_diff_check_performed": True,
        "exact_five_file_checks_performed": True,
        "no_deletions_renames_or_other_edits": True,
    }


@pytest.fixture(scope="module")
def real_bridge_chain():
    """Build a real qualified V11 chain with only the frozen pure Meta runners."""

    meta_root = Path(CORE.META_SOURCE_BINDING_ROOT)
    script = r'''
import json, os, subprocess, sys
from pathlib import Path
root = Path(os.environ["META_ROOT"])
target = Path(os.environ["TARGET"])
sys.path[:0] = [str(root), str(root / "tests")]
import test_shadow_interval_multi_q_post_transition_qualification as q
paths, values = q._qualification_cli_inputs(target)
completed = subprocess.run(
    [sys.executable, str(q.POST_TRANSITION_QUALIFICATION_RUNNER),
     *q._input_flags(paths), *q._digest_flags(values)],
    cwd=root, stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
)
if completed.returncode != 0:
    raise SystemExit(completed.stderr.decode("utf-8", "replace"))
report_path = target / "input-29-post-transition-qualification-report.json"
report_path.write_bytes(completed.stdout)
report = json.loads(completed.stdout)
print(json.dumps({"paths": [str(path) for path in (*paths, report_path)],
                  "digests": values,
                  "report_digest": report["report_digest"]},
                 sort_keys=True, separators=(",", ":")))
'''
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        target = Path(directory)
        environment = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "META_ROOT": str(meta_root),
            "TARGET": str(target),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=meta_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
        generated = json.loads(completed.stdout)
        paths = [Path(value) for value in generated["paths"]]
        assert len(paths) == 30
        objects = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        anchor_map = {}
        for name in CORE.EXPECTED_DIGEST_NAMES:
            short = name.removeprefix("expected_").removesuffix("_digest")
            anchor_map[name] = (
                generated["report_digest"]
                if short == "post_transition_qualification_report"
                else generated["digests"][short]
            )
        exact_hashes = {
            relative: sha256((WORKTREE / relative).read_bytes()).hexdigest()
            for relative in CORE.EXACT_FIVE_PATHS
        }
        hegel_binding = {
            "project_root": CORE.PROJECT_ROOT,
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "clean": True,
            "baseline_commit": RUNNER.BASELINE_COMMIT,
            "baseline_tree": RUNNER.BASELINE_TREE,
            "repair_is_ancestor": True,
            "exact_five_diff_paths": list(CORE.EXACT_FIVE_PATHS),
            "exact_five_raw_sha256": exact_hashes,
            "existing_actual_tool_raw_sha256": CORE.HEGEL_UNCHANGED_RAW_SHA256[
                "tools/phase3_q05b_dual_qualification_v1.py"
            ],
            "existing_admission_core_raw_sha256": CORE.HEGEL_UNCHANGED_RAW_SHA256[
                "src/hegel_machine/phase3_q05b_actual_admission_v1.py"
            ],
            "existing_artifact_core_raw_sha256": CORE.HEGEL_UNCHANGED_RAW_SHA256[
                "src/hegel_machine/phase3_q05b_actual_artifact_v1.py"
            ],
            "isolation_config_raw_sha256": CORE.HEGEL_UNCHANGED_RAW_SHA256[
                "config/phase3_q05b_dual_isolation_v1.json"
            ],
            "qualification_config_raw_sha256": CORE.HEGEL_UNCHANGED_RAW_SHA256[
                "config/phase3_q05b_node3_dual_projection_qualification_v1.json"
            ],
            "existing_unchanged_raw_sha256": deepcopy(
                CORE.HEGEL_UNCHANGED_RAW_SHA256
            ),
        }
        bridge_input = {
            "schema_version": CORE.INPUT_SCHEMA_VERSION,
            "owner_authorization": deepcopy(CORE.OWNER_AUTHORIZATION),
            "meta_source_binding": deepcopy(CORE.META_SOURCE_BINDING),
            "meta_v11_expected_digests": anchor_map,
            "attempt_history_binding": deepcopy(CORE.ATTEMPT_HISTORY_BINDING),
            "hegel_source_binding": hegel_binding,
            "durable_ledger_binding": {
                "root": CORE.LEDGER_ROOT,
                "expected_initial_state": "OPEN_OR_EXACT_BRIDGE_STAGED_NOT_CONSUMED",
            },
            "execution_request": {
                "mode": "HEGEL_Q05B_ACTUAL_ATTEMPT2",
                "attempt_ordinal": 2,
                "maximum_actual_invocations": 1,
                "source_commit": "a" * 40,
                "artifact_path": CORE.Q05B_ARTIFACT_PATH,
                "cargo_cache_source": CORE.CARGO_CACHE_SOURCE,
                "run_actual_entrypoint": CORE.RUN_ACTUAL_ENTRYPOINT,
            },
        }
        bridge_input_path = target / "input-30-attempt2-bridge-input.json"
        bridge_input_path.write_bytes(_canonical(bridge_input))
        contract_path = target / "input-31-attempt2-bridge-contract.json"
        contract_path.write_bytes(CONTRACT_PATH.read_bytes())
        all_paths = [*paths, bridge_input_path, contract_path]
        all_objects = [*objects, bridge_input, CONTRACT]
        artifacts = {}
        for name, path in zip(CORE.BRIDGE_INPUT_NAMES, all_paths, strict=True):
            raw = path.read_bytes()
            artifacts[name] = {
                "path": str(path),
                "bytes": len(raw),
                "sha256": sha256(raw).hexdigest(),
            }
        result = CORE.build_phase3_q05b_attempt2_meta_v11_bridge_v1(
            *all_objects, input_artifacts=artifacts
        )
        report = result.to_dict()
        receipt = CORE.verify_phase3_q05b_attempt2_meta_v11_bridge_v1(
            *all_objects,
            report,
            expected_attempt2_bridge_report_digest=report["report_digest"],
            expected_input_artifacts=artifacts,
        )
        yield {
            "objects": all_objects,
            "paths": all_paths,
            "artifacts": artifacts,
            "bridge_input": bridge_input,
            "result": result,
            "report": report,
            "receipt": receipt,
        }


def test_real_30_object_meta_replay_admits_only_qualified_v11(real_bridge_chain) -> None:
    result = real_bridge_chain["result"]
    report = real_bridge_chain["report"]
    receipt = real_bridge_chain["receipt"]
    assert result.disposition == CORE.BRIDGE_ADMITTED
    assert result.report_digest == report["report_digest"]
    assert result.admitted is True
    assert receipt == {
        "status": "VERIFIED_" + CORE.BRIDGE_ADMITTED,
        "disposition": CORE.BRIDGE_ADMITTED,
        "report_digest": report["report_digest"],
        "contract_digest": FROZEN_CONTRACT_DIGEST,
        "bridge_input_digest": report["bridge_input_digest"],
        "meta_v11_report_digest": report["meta_v11_replay_receipt"]["report_digest"],
        "destination_adoption_receipt_digest": report["destination_adoption_receipt"]["receipt_digest"],
        "admitted": True,
    }
    assert report["meta_v11_replay_receipt"]["qualified"] is True
    assert report["authority_boundary"]["ledger_consumed"] is False
    assert report["authority_boundary"]["actual_called"] is False


def test_real_public_verifier_rejects_all_object_report_and_anchor_tampering(
    real_bridge_chain,
) -> None:
    objects = real_bridge_chain["objects"]
    report = real_bridge_chain["report"]
    artifacts = real_bridge_chain["artifacts"]
    for index in range(30):
        changed = deepcopy(objects)
        changed[index] = {**changed[index], "unregistered_surface": index}
        with pytest.raises(ValueError):
            CORE.verify_phase3_q05b_attempt2_meta_v11_bridge_v1(
                *changed,
                report,
                expected_attempt2_bridge_report_digest=report["report_digest"],
                expected_input_artifacts=artifacts,
            )
    changed_report = deepcopy(report)
    changed_report["authority_boundary"]["meta_global_adoption"] = True
    body = {key: value for key, value in changed_report.items() if key != "report_digest"}
    changed_report["report_digest"] = "sha256:" + sha256(
        CORE.DOMAINS["report"] + _canonical(body)
    ).hexdigest()
    with pytest.raises(ValueError):
        CORE.verify_phase3_q05b_attempt2_meta_v11_bridge_v1(
            *objects,
            changed_report,
            expected_attempt2_bridge_report_digest=changed_report["report_digest"],
            expected_input_artifacts=artifacts,
        )
    for name in CORE.EXPECTED_DIGEST_NAMES:
        changed = deepcopy(objects)
        changed[30]["meta_v11_expected_digests"][name] = "sha256:" + "0" * 64
        with pytest.raises(ValueError):
            CORE.build_phase3_q05b_attempt2_meta_v11_bridge_v1(
                *changed, input_artifacts=artifacts
            )



def test_contract_manifest_and_public_signatures_are_exact() -> None:
    assert CORE.FROZEN_CONTRACT_DIGEST == FROZEN_CONTRACT_DIGEST
    assert CORE.validate_phase3_q05b_attempt2_meta_v11_bridge_contract_v1(
        CONTRACT
    ) == CONTRACT
    assert len(RUNNER.V11_INPUT_ROWS) == 30
    assert len(RUNNER.BRIDGE_INPUT_ROWS) == 32
    assert len(RUNNER.EXPECTED_DIGEST_NAMES) == 29
    assert len(tuple(itertools.combinations(RUNNER.BRIDGE_INPUT_ROWS, 2))) == 496
    builder = inspect.signature(CORE.build_phase3_q05b_attempt2_meta_v11_bridge_v1)
    verifier = inspect.signature(CORE.verify_phase3_q05b_attempt2_meta_v11_bridge_v1)
    assert list(builder.parameters)[:32] == [name for name, _ in RUNNER.BRIDGE_INPUT_ROWS]
    assert list(verifier.parameters)[:33] == [
        *(name for name, _ in RUNNER.BRIDGE_INPUT_ROWS),
        "attempt2_bridge_report",
    ]
    assert list(builder.parameters)[32:] == ["input_artifacts"]
    assert list(verifier.parameters)[33:] == [
        "expected_attempt2_bridge_report_digest",
        "expected_input_artifacts",
    ]
    for function in (
        CORE.build_attempt2_consumed_record_v1,
        CORE.build_attempt2_execution_link_v1,
        CORE.build_attempt2_failure_link_v1,
        CORE.validate_attempt2_execution_link_record_v1,
        CORE.validate_attempt2_failure_link_record_v1,
    ):
        assert callable(function)


def test_cli_surface_has_no_caller_selected_output() -> None:
    parser = RUNNER._parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {flag for _name, flag in RUNNER.V11_INPUT_ROWS}.issubset(options)
    assert "--bridge-input" in options and "--bridge-contract" in options
    assert "--preflight" in options and "--run" in options
    assert not any(option == "--out" or option.endswith("-out") for option in options)
    for name in RUNNER.EXPECTED_DIGEST_NAMES:
        assert "--expected-" + name.replace("_", "-") + "-digest" in options


def test_cli_help_exits_zero_without_supervisor_error_wire(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert RUNNER.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "--preflight" in captured.out
    assert captured.err == ""
    assert "hegel-phase3a-q05b-attempt2-meta-v11-bridge-error/1" not in captured.out


def test_owner_authorization_every_authoritative_surface_is_frozen() -> None:
    owner = deepcopy(CORE.OWNER_AUTHORIZATION)
    assert CORE._validate_owner_authorization(owner) == owner
    assert owner["selected_antecedent_text"] == (
        "下一步若要进入 Hegel Q0.5b actual，需要你另行明确授权 attempt-2，并先补 "
        "digest-bound external authority/adoption bridge"
    )
    mutations = []
    for key in (
        "schema_version",
        "authority_role",
        "selected_antecedent_text",
        "confirmation_text",
        "confirmation_semantics",
        "authorized_attempt_ordinal",
        "maximum_actual_attempts",
        "authorization_is_execution_evidence",
        "authorization_is_external_signature",
        "authorization_digest",
    ):
        changed = deepcopy(owner)
        value = changed[key]
        changed[key] = (not value) if type(value) is bool else (value + 1 if type(value) is int else str(value) + "-drift")
        mutations.append(changed)
    for key in ("authorized_actions", "forbidden_actions"):
        changed = deepcopy(owner)
        changed[key] = list(reversed(changed[key]))
        mutations.append(changed)
        changed = deepcopy(owner)
        changed[key][0] += "-drift"
        mutations.append(changed)
    for changed in mutations:
        with pytest.raises(ValueError):
            CORE._validate_owner_authorization(changed)


def test_missing_real_v11_wire_and_strict_json_edges(tmp_path: Path) -> None:
    expected = CORE.build_missing_real_v11_report_error_v1()
    assert expected["status"] == "MISSING_REAL_V11_REPORT"
    assert all(
        expected[key] is False
        for key in (
            "bridge_report_created",
            "ledger_created",
            "ledger_consumed",
            "actual_called",
            "docker_or_cargo_called",
            "q05b_artifact_written",
        )
    )
    for raw in (b"", b"null", b"[]", b"0", b'"x"'):
        with pytest.raises(RUNNER.MissingV11):
            RUNNER._strict_object(raw, "V11", missing=True)
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
        with pytest.raises(RUNNER.RunnerError):
            RUNNER._strict_object(raw, "JSON")
    target = tmp_path / "report.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "report-link.json"
    link.symlink_to(target)
    with pytest.raises(RUNNER.MissingV11):
        RUNNER._read_file(link, "V11", missing=True)


def test_missing_report_precedes_every_other_gate_and_has_zero_effects(tmp_path: Path) -> None:
    missing = tmp_path / "missing-v11.json"
    argv = ["--preflight"]
    for name, flag in RUNNER.V11_INPUT_ROWS:
        path = missing if name == "post_transition_qualification_report" else tmp_path / f"{name}.json"
        argv.extend([flag, str(path)])
    argv += [
        "--bridge-input", str(tmp_path / "bridge-input.json"),
        "--bridge-contract", str(tmp_path / "bridge-contract.json"),
        "--meta-project-root", "/invalid/meta",
        "--hegel-project-root", "/invalid/hegel",
        "--source-commit", "invalid",
        "--artifact", "/invalid/artifact",
        "--cargo-cache-source", "/invalid/cache",
        "--ledger-root", "/invalid/ledger",
    ]
    for name in RUNNER.EXPECTED_DIGEST_NAMES:
        argv += ["--expected-" + name.replace("_", "-") + "-digest", "bad"]
    stdout, stderr = io.BytesIO(), io.BytesIO()
    assert RUNNER.main(argv, stdout=stdout, stderr=stderr) == 2
    assert stdout.getvalue() == b""
    assert json.loads(stderr.getvalue())["status"] == "MISSING_REAL_V11_REPORT"
    assert not missing.exists()


def test_all_496_path_and_inode_aliases_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _namespace()
    paths = [getattr(args, name) for name, _ in RUNNER.BRIDGE_INPUT_ROWS]
    inode_by_path = {path: (1, index + 1) for index, path in enumerate(paths)}

    def fake_load(path, label, *, missing=False):
        return {}, {"path": str(path), "bytes": 2, "sha256": "0" * 64}, inode_by_path[path], b"{}"

    monkeypatch.setattr(RUNNER, "_load_json", fake_load)
    for left, right in itertools.combinations(range(32), 2):
        right_name = RUNNER.BRIDGE_INPUT_ROWS[right][0]
        original = getattr(args, right_name)
        setattr(args, right_name, paths[left])
        with pytest.raises(RUNNER.RunnerError, match="paths must be distinct"):
            RUNNER._load_all_inputs(args)
        setattr(args, right_name, original)

        old_inode = inode_by_path[original]
        inode_by_path[original] = inode_by_path[paths[left]]
        with pytest.raises(RUNNER.RunnerError, match="inodes must be distinct"):
            RUNNER._load_all_inputs(args)
        inode_by_path[original] = old_inode


def test_source_file_evidence_requires_ordinary_unique_exact_bytes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    expected = {}
    for index in range(3):
        relative = f"p{index}.py"
        raw = f"value={index}\n".encode()
        (root / relative).write_bytes(raw)
        expected[relative] = sha256(raw).hexdigest()
    receipt = RUNNER._source_file_evidence(root, expected, "source")
    assert set(receipt) == set(expected)
    assert len({(row["device"], row["inode"]) for row in receipt.values()}) == 3
    assert all(stat.S_ISREG(row["mode"]) and row["nlink"] == 1 for row in receipt.values())
    (root / "p2.py").unlink()
    os.link(root / "p1.py", root / "p2.py")
    with pytest.raises(RUNNER.RunnerError):
        RUNNER._source_file_evidence(root, expected, "source")


def test_git_is_read_only_noninteractive_and_network_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(RUNNER.subprocess, "run", fake_run)
    assert RUNNER._git(Path("/tmp"), "status") == "ok"
    env = captured["env"]
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_PROTOCOL_FROM_USER"] == "0"
    assert env["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert "core.fsmonitor=false" in captured["command"]
    assert "core.untrackedCache=false" in captured["command"]


def test_source_only_import_context_replaces_and_restores_all_import_authority(tmp_path: Path) -> None:
    class MaliciousFinder:
        pass

    old_dont = sys.dont_write_bytecode
    old_prefix = sys.pycache_prefix
    old_path = list(sys.path)
    old_meta = list(sys.meta_path)
    old_hooks = list(sys.path_hooks)
    old_cache = sys.path_importer_cache
    malicious = MaliciousFinder()
    sys.meta_path.insert(0, malicious)
    sys.path_importer_cache[str(tmp_path)] = malicious
    expected_meta = list(sys.meta_path)
    expected_cache = sys.path_importer_cache
    try:
        with RUNNER._standard_source_imports("/dev/null/test", tmp_path):
            assert sys.dont_write_bytecode is True
            assert sys.pycache_prefix == "/dev/null/test"
            assert malicious not in sys.meta_path
            assert sys.meta_path == [
                importlib.machinery.BuiltinImporter,
                importlib.machinery.FrozenImporter,
                importlib.machinery.PathFinder,
            ]
            assert sys.path_importer_cache == {}
            assert all("SourcelessFileLoader" not in repr(hook) for hook in sys.path_hooks)
        assert sys.meta_path == expected_meta
        assert sys.path_importer_cache is expected_cache
    finally:
        sys.dont_write_bytecode = old_dont
        sys.pycache_prefix = old_prefix
        sys.path[:] = old_path
        sys.meta_path[:] = old_meta
        sys.path_hooks[:] = old_hooks
        sys.path_importer_cache = old_cache


def test_ambient_hegel_module_blocks_source_only_bridge_import() -> None:
    marker = "hegel_machine.ambient_attack"
    sys.modules[marker] = SimpleNamespace()
    try:
        with pytest.raises(RUNNER.RunnerError, match="ambient Hegel modules"):
            RUNNER._load_bridge_core()
    finally:
        sys.modules.pop(marker, None)


def test_copied_runner_root_is_rejected_before_any_project_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RUNNER, "PROJECT_ROOT", Path("/tmp/copied-hegel"))
    with pytest.raises(RUNNER.RunnerError, match="fixed self-location"):
        RUNNER._verify_fixed_self_location()


def test_actual_and_meta_live_receipts_reject_every_self_asserted_drift() -> None:
    actual = _synthetic_actual_receipt()
    meta = _synthetic_meta_receipt()
    assert CORE.validate_hegel_actual_source_only_import_receipt_v1(actual) == actual
    assert CORE.validate_meta_live_source_validation_receipt_v1(meta) == meta
    for key in CORE.ACTUAL_SOURCE_ONLY_RECEIPT_FIELDS - {
        "preloaded_hegel_machine_modules",
        "loaded_hegel_source_module_origins",
        "unchanged_source_file_evidence",
    }:
        changed = deepcopy(actual)
        value = changed[key]
        changed[key] = (not value) if type(value) is bool else str(value) + "-drift"
        with pytest.raises(ValueError):
            CORE.validate_hegel_actual_source_only_import_receipt_v1(changed)
    relative = next(iter(CORE.HEGEL_UNCHANGED_RAW_SHA256))
    invalid_values = {
        "path": "drift",
        "device": -1,
        "inode": 0,
        "nlink": 2,
        "mode": stat.S_IFDIR | 0o755,
        "size": -1,
        "sha256": "0" * 64,
    }
    for key, invalid in invalid_values.items():
        changed = deepcopy(actual)
        changed["unchanged_source_file_evidence"][relative][key] = invalid
        with pytest.raises(ValueError):
            CORE.validate_hegel_actual_source_only_import_receipt_v1(changed)
    changed = deepcopy(actual)
    changed["loaded_hegel_source_module_origins"] = {
        "hegel_machine.evil": CORE.PROJECT_ROOT + "/src/hegel_machine/../../evil.py"
    }
    with pytest.raises(ValueError):
        CORE.validate_hegel_actual_source_only_import_receipt_v1(changed)
    for key in CORE.META_LIVE_SOURCE_RECEIPT_FIELDS - {"exact_five_file_evidence"}:
        changed = deepcopy(meta)
        value = changed[key]
        changed[key] = (not value) if type(value) is bool else str(value) + "-drift"
        with pytest.raises(ValueError):
            CORE.validate_meta_live_source_validation_receipt_v1(changed)


def test_ledger_root_creation_is_anchored_durable_and_close_on_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory) / "attempt2-ledger"
        monkeypatch.setattr(RUNNER, "FROZEN_LEDGER_ROOT", root)
        fsync_inodes = []
        real_fsync = os.fsync

        def recording_fsync(fd):
            fsync_inodes.append(os.fstat(fd).st_ino)
            return real_fsync(fd)

        monkeypatch.setattr(RUNNER.os, "fsync", recording_fsync)
        fd = RUNNER._open_ledger(create=True)
        assert fd is not None
        try:
            child = os.fstat(fd)
            parent = root.parent.stat()
            assert stat.S_IMODE(child.st_mode) == 0o700
            assert fsync_inodes[:2] == [child.st_ino, parent.st_ino]
            assert RUNNER.fcntl.fcntl(fd, RUNNER.fcntl.F_GETFD) & RUNNER.fcntl.FD_CLOEXEC
        finally:
            RUNNER._close_ledger(fd)


def test_ledger_records_are_no_replace_immutable_and_state_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory) / "ledger"
        monkeypatch.setattr(RUNNER, "FROZEN_LEDGER_ROOT", root)
        fd = RUNNER._open_ledger(create=True)
        assert fd is not None
        try:
            assert RUNNER._ledger_snapshot(fd)[0] == "OPEN_ALL_FOUR_RECORDS_ABSENT"
            payload = _canonical({"report_digest": "sha256:" + "1" * 64})
            status = RUNNER._publish(fd, RUNNER.LEDGER_NAMES["bridge"], payload)
            assert stat.S_IMODE(status.st_mode) == 0o444 and status.st_nlink == 1
            assert RUNNER._ledger_snapshot(fd)[0] == "BRIDGE_STAGED_NOT_CONSUMED"
            with pytest.raises(FileExistsError):
                RUNNER._publish(fd, RUNNER.LEDGER_NAMES["bridge"], payload)
        finally:
            RUNNER._close_ledger(fd)


def test_consumed_o_excl_race_has_one_winner_and_crash_state_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory) / "ledger"
        monkeypatch.setattr(RUNNER, "FROZEN_LEDGER_ROOT", root)
        fd = RUNNER._open_ledger(create=True)
        assert fd is not None
        bridge = _canonical({"kind": "exact-bridge-stage"})
        consumed = _canonical({"kind": "durable-consumed-before-actual"})
        RUNNER._publish(fd, RUNNER.LEDGER_NAMES["bridge"], bridge)
        barrier = threading.Barrier(2)
        outcomes = []

        def contender() -> None:
            barrier.wait()
            try:
                RUNNER._publish(fd, RUNNER.LEDGER_NAMES["consumed"], consumed)
            except FileExistsError:
                outcomes.append("lost")
            else:
                outcomes.append("consumed")

        threads = [threading.Thread(target=contender) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert sorted(outcomes) == ["consumed", "lost"]
        assert RUNNER._ledger_snapshot(fd)[0] == "CONSUMED_OUTCOME_UNKNOWN"
        RUNNER._close_ledger(fd)
        reopened = RUNNER._open_ledger(create=False)
        assert reopened is not None
        try:
            assert RUNNER._ledger_snapshot(reopened)[0] == "CONSUMED_OUTCOME_UNKNOWN"
        finally:
            RUNNER._close_ledger(reopened)


def test_terminal_ledger_validation_uses_full_standalone_validators() -> None:
    calls = []

    class FakeCore:
        def validate_phase3_q05b_attempt2_meta_v11_bridge_report_v1(self, *args):
            calls.append("report")

        def validate_attempt2_consumed_record_v1(self, *args):
            calls.append("consumed")

        def validate_attempt2_execution_link_record_v1(self, *args):
            calls.append("execution")

        def validate_attempt2_failure_link_record_v1(self, *args):
            calls.append("failure")

    values = {
        "bridge": ({}, b"", None),
        "consumed": ({}, b"", None),
        "execution": ({}, b"", None),
        "failure": ({}, b"", None),
    }
    RUNNER._validate_consumed_state(FakeCore(), values, {})
    assert calls == ["report", "consumed", "execution", "failure"]


def test_existing_consumed_authorization_precedes_replacement_input_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _namespace()

    class Parser:
        def parse_args(self, argv):
            return args

    fake_core = SimpleNamespace(ALREADY_CONSUMED="ATTEMPT2_AUTHORIZATION_ALREADY_CONSUMED")
    monkeypatch.setattr(RUNNER, "_parser", lambda: Parser())
    monkeypatch.setattr(RUNNER, "_load_json", lambda *a, **k: ({}, {}, (1, 1), b"{}"))
    monkeypatch.setattr(RUNNER, "_verify_ledger_argument", lambda a: None)
    monkeypatch.setattr(RUNNER, "_load_bridge_core", lambda: fake_core)
    monkeypatch.setattr(RUNNER, "_open_ledger", lambda **k: 91)
    monkeypatch.setattr(
        RUNNER,
        "_ledger_snapshot",
        lambda fd: ("CONSUMED_OUTCOME_UNKNOWN", {"bridge": ({}, b"", None), "consumed": ({}, b"", None)}),
    )
    monkeypatch.setattr(RUNNER, "_validate_consumed_state", lambda *a: None)
    monkeypatch.setattr(RUNNER, "_close_ledger", lambda fd: None)
    monkeypatch.setattr(
        RUNNER,
        "_verify_runtime_arguments",
        lambda a: (_ for _ in ()).throw(AssertionError("replacement replay reached")),
    )
    stdout, stderr = io.BytesIO(), io.BytesIO()
    assert RUNNER.main([], stdout=stdout, stderr=stderr) == 2
    wire = json.loads(stderr.getvalue())
    assert stdout.getvalue() == b""
    assert wire["status"] == "ATTEMPT2_AUTHORIZATION_ALREADY_CONSUMED"
    assert wire["ledger_created"] is True and wire["ledger_consumed"] is True
    assert wire["actual_called"] is False


def test_staged_bridge_fact_survives_later_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _namespace()

    class Parser:
        def parse_args(self, argv):
            return args

    class FakeCore:
        @staticmethod
        def build_attempt2_supervisor_error_v1(status, detail, **facts):
            return {"schema_version": "error/1", "status": status, "detail": detail, **facts}

        @staticmethod
        def validate_attempt2_supervisor_error_v1(value):
            return value

    monkeypatch.setattr(RUNNER, "_parser", lambda: Parser())
    monkeypatch.setattr(RUNNER, "_load_json", lambda *a, **k: ({}, {}, (1, 1), b"{}"))
    monkeypatch.setattr(RUNNER, "_verify_fixed_self_location", lambda: None)
    monkeypatch.setattr(RUNNER, "_verify_ledger_argument", lambda a: None)
    monkeypatch.setattr(RUNNER, "_load_bridge_core", lambda: FakeCore())
    monkeypatch.setattr(RUNNER, "_open_ledger", lambda **k: 93)
    monkeypatch.setattr(
        RUNNER,
        "_ledger_snapshot",
        lambda fd: ("BRIDGE_STAGED_NOT_CONSUMED", {"bridge": ({}, b"staged", None)}),
    )
    monkeypatch.setattr(RUNNER, "_close_ledger", lambda fd: None)
    monkeypatch.setattr(
        RUNNER,
        "_verify_runtime_arguments",
        lambda a: (_ for _ in ()).throw(RUNNER.RunnerError("LATER_GATE_FAILED", "later")),
    )
    stdout, stderr = io.BytesIO(), io.BytesIO()
    assert RUNNER.main([], stdout=stdout, stderr=stderr) == 2
    wire = json.loads(stderr.getvalue())
    assert wire["bridge_report_created"] is True
    assert wire["ledger_created"] is True
    assert wire["ledger_consumed"] is False
    assert wire["actual_called"] is False


class _FakeResult:
    def __init__(self, report):
        self.report = report

    def to_dict(self):
        return deepcopy(self.report)


def _install_main_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    actual_behavior: str = "return",
    execution_behavior: str = "return",
):
    args = _namespace()
    args.preflight = mode == "preflight"
    args.run = mode == "run"
    anchors = {
        f"expected_{name}_digest": getattr(args, f"expected_{name}_digest")
        for name in RUNNER.EXPECTED_DIGEST_NAMES
    }
    loaded = {name: {} for name, _ in RUNNER.BRIDGE_INPUT_ROWS}
    loaded["attempt2_bridge_input"] = {
        "meta_v11_expected_digests": anchors,
        "meta_source_binding": {},
        "hegel_source_binding": {},
    }
    loaded["attempt2_bridge_contract"] = {}
    report = {
        "disposition": "META_V11_QUALIFIED_BRIDGE_ADMITTED_ATTEMPT2_PENDING_CONSUMPTION",
        "report_digest": "sha256:" + "9" * 64,
        "contract_digest": FROZEN_CONTRACT_DIGEST,
        "bridge_input_digest": "sha256:" + "7" * 64,
        "meta_v11_replay_receipt": {"report_digest": "sha256:" + "6" * 64},
        "destination_adoption_receipt": {"receipt_digest": "sha256:" + "5" * 64},
    }
    events = []
    captured = {}

    class Parser:
        def parse_args(self, argv):
            return args

    class FakeCore:
        BRIDGE_ADMITTED = report["disposition"]
        BRIDGE_REFUSED = "META_V11_REPORT_NOT_QUALIFIED_BRIDGE_REFUSED"
        ALREADY_CONSUMED = "ATTEMPT2_AUTHORIZATION_ALREADY_CONSUMED"

        canonical_json_bytes_v1 = staticmethod(_canonical)

        def validate_meta_live_source_validation_receipt_v1(self, receipt):
            events.append("meta-receipt-validated")

        def validate_hegel_live_source_validation_receipt_v1(self, receipt, binding):
            events.append("hegel-live-receipt-validated")

        def build_attempt2_supervisor_error_v1(self, status, detail, **facts):
            return {
                "schema_version": "error/1",
                "status": status,
                "detail": detail,
                **facts,
            }

        def validate_attempt2_supervisor_error_v1(self, value):
            return value

        def build_phase3_q05b_attempt2_meta_v11_bridge_v1(self, *values, **kwargs):
            assert len(values) == 32 and kwargs == {"input_artifacts": {}}
            events.append("full-32-builder")
            return _FakeResult(report)

        def verify_phase3_q05b_attempt2_meta_v11_bridge_v1(self, *values, **kwargs):
            assert len(values) == 33
            assert values[-1] == report
            assert kwargs == {
                "expected_attempt2_bridge_report_digest": report["report_digest"],
                "expected_input_artifacts": {},
            }
            events.append("full-33-verifier")
            return {
                "status": "VERIFIED_" + report["disposition"],
                "disposition": report["disposition"],
                "report_digest": report["report_digest"],
                "contract_digest": report["contract_digest"],
                "bridge_input_digest": report["bridge_input_digest"],
                "meta_v11_report_digest": report["meta_v11_replay_receipt"]["report_digest"],
                "destination_adoption_receipt_digest": report["destination_adoption_receipt"]["receipt_digest"],
                "admitted": True,
            }

        def build_attempt2_consumed_record_v1(self, *values):
            events.append("build-consumed")
            return {"record_digest": "sha256:" + "8" * 64}

        def validate_consumed_file_evidence_v1(self, *values):
            events.append("consumed-evidence-validated")

        def build_attempt2_execution_link_v1(self, *values, **kwargs):
            events.append("build-execution")
            assert kwargs["meta_live_source_validation_receipt"]["git_clean"] is True
            assert kwargs["hegel_live_source_validation_receipt"]["clean"] is True
            if execution_behavior == "raise":
                raise RUNNER.RunnerError("POSTRETURN_INVALID", "strict replay failed")
            return {"status": "ATTEMPT2_QUALIFIED_LINKED"}

        def build_attempt2_failure_link_v1(self, *values, **kwargs):
            events.append("build-failure")
            captured.update(kwargs)
            return {"status": "ATTEMPT2_FAILED_CLOSED"}

    class Actual:
        def __init__(self, core):
            events.append("actual-session-constructed")

        def __enter__(self):
            events.append("actual-source-loaded")
            return self

        def call(self, source_commit):
            events.append("actual-called")
            if actual_behavior == "raise":
                raise RuntimeError("actual raised")
            return {"returned": True}

        def __exit__(self, *args):
            events.append("actual-session-restored")

        def receipt(self):
            return _synthetic_actual_receipt()

    ledger_open_count = 0

    def open_ledger(*, create):
        nonlocal ledger_open_count
        ledger_open_count += 1
        events.append("ledger-open-create" if create else "ledger-open-existing")
        return None if ledger_open_count == 1 else 80 + ledger_open_count

    snapshots = iter(
        [
            ("OPEN_ALL_FOUR_RECORDS_ABSENT", {}),
            ("OPEN_ALL_FOUR_RECORDS_ABSENT", {}),
            ("CONSUMED_OUTCOME_UNKNOWN", {}),
        ]
    )

    def publish(fd, name, payload):
        events.append("publish-" + name)
        return SimpleNamespace(st_dev=2, st_ino=3, st_nlink=1, st_mode=stat.S_IFREG | 0o444)

    monkeypatch.setattr(RUNNER, "_parser", lambda: Parser())
    monkeypatch.setattr(RUNNER, "_load_json", lambda *a, **k: ({}, {}, (1, 1), b"{}"))
    monkeypatch.setattr(RUNNER, "_verify_ledger_argument", lambda a: None)
    monkeypatch.setattr(RUNNER, "_verify_runtime_arguments", lambda a: None)
    monkeypatch.setattr(RUNNER, "_load_bridge_core", lambda: FakeCore())
    monkeypatch.setattr(RUNNER, "_open_ledger", open_ledger)
    monkeypatch.setattr(RUNNER, "_ledger_snapshot", lambda fd: next(snapshots))
    monkeypatch.setattr(RUNNER, "_close_ledger", lambda fd: events.append("ledger-closed") if fd is not None else None)
    monkeypatch.setattr(RUNNER, "_load_all_inputs", lambda a: (loaded, {}, ()))
    monkeypatch.setattr(RUNNER, "_verify_meta_source", lambda b: _synthetic_meta_receipt())
    monkeypatch.setattr(
        RUNNER, "_verify_hegel_source", lambda *a: _synthetic_hegel_live_receipt()
    )
    monkeypatch.setattr(
        RUNNER.os.path,
        "lexists",
        lambda path: execution_behavior == "raise" and "actual-called" in events,
    )
    monkeypatch.setattr(RUNNER, "_ActualSourceSession", Actual)
    monkeypatch.setattr(RUNNER, "_publish", publish)
    monkeypatch.setattr(
        RUNNER,
        "_consumed_evidence",
        lambda status, payload: {"path": "consumed", "sha256": sha256(payload).hexdigest()},
    )
    monkeypatch.setattr(
        RUNNER,
        "_read_file",
        lambda *a, **k: (b'{"artifact":true}', SimpleNamespace()),
    )
    return events, captured, report


def test_preflight_emits_report_and_performs_zero_write_or_actual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _captured, report = _install_main_fakes(monkeypatch, mode="preflight")
    stdout, stderr = io.BytesIO(), io.BytesIO()
    assert RUNNER.main([], stdout=stdout, stderr=stderr) == 0
    assert stdout.getvalue() == _canonical(report) and stderr.getvalue() == b""
    assert "full-32-builder" in events
    assert "full-33-verifier" in events
    assert not any(item.startswith("publish-") for item in events)
    assert "actual-session-constructed" not in events and "actual-called" not in events


def test_public_33_object_verifier_refusal_precedes_ledger_and_actual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _captured, _report = _install_main_fakes(monkeypatch, mode="run")
    original_factory = RUNNER._load_bridge_core
    core = original_factory()

    def reject(*args, **kwargs):
        events.append("full-33-verifier-refused")
        raise RUNNER.RunnerError("INVALID_ATTEMPT2_META_V11_BRIDGE", "verifier refused")

    core.verify_phase3_q05b_attempt2_meta_v11_bridge_v1 = reject
    monkeypatch.setattr(RUNNER, "_load_bridge_core", lambda: core)
    stdout, stderr = io.BytesIO(), io.BytesIO()
    assert RUNNER.main([], stdout=stdout, stderr=stderr) == 2
    assert stdout.getvalue() == b""
    assert json.loads(stderr.getvalue())["status"] == "INVALID_ATTEMPT2_META_V11_BRIDGE"
    assert "full-33-verifier-refused" in events
    assert not any(item.startswith("publish-") for item in events)
    assert "actual-session-constructed" not in events


def test_run_consumes_and_closes_ledger_before_exactly_one_actual_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _captured, _report = _install_main_fakes(monkeypatch, mode="run")
    stdout, stderr = io.BytesIO(), io.BytesIO()
    assert RUNNER.main([], stdout=stdout, stderr=stderr) == 0
    assert json.loads(stdout.getvalue())["status"] == "ATTEMPT2_QUALIFIED_LINKED"
    assert stderr.getvalue() == b""
    consumed = events.index("publish-attempt-2.consumed.json")
    called = events.index("actual-called")
    assert consumed < events.index("ledger-closed", consumed) < called
    assert events.count("actual-called") == 1
    assert events.index("publish-attempt-2.execution-link.json") > called


def test_call_raised_and_postreturn_failures_bind_actual_returned_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _events, captured, _ = _install_main_fakes(
        monkeypatch, mode="run", actual_behavior="raise"
    )
    stdout, stderr = io.BytesIO(), io.BytesIO()
    assert RUNNER.main([], stdout=stdout, stderr=stderr) == 1
    assert stderr.getvalue() == b"" and captured["actual_returned"] is False
    assert json.loads(stdout.getvalue())["status"] == "ATTEMPT2_FAILED_CLOSED"


def test_postreturn_strict_replay_failure_binds_true_and_residual_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _events, captured, _ = _install_main_fakes(
        monkeypatch, mode="run", execution_behavior="raise"
    )
    stdout, stderr = io.BytesIO(), io.BytesIO()
    assert RUNNER.main([], stdout=stdout, stderr=stderr) == 1
    assert stderr.getvalue() == b"" and captured["actual_returned"] is True
    assert json.loads(stdout.getvalue())["status"] == "ATTEMPT2_FAILED_CLOSED"


def test_dynamic_error_wire_never_denies_durable_consumption_or_actual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _events, _captured, _ = _install_main_fakes(
        monkeypatch, mode="run", execution_behavior="raise"
    )

    def failure_builder_raises(self, *args, **kwargs):
        raise RuntimeError("failure-link publication unavailable")

    # Replace the fake core factory with one whose failure builder itself fails.
    original_factory = RUNNER._load_bridge_core
    core = original_factory()
    core.build_attempt2_failure_link_v1 = failure_builder_raises.__get__(core)
    monkeypatch.setattr(RUNNER, "_load_bridge_core", lambda: core)
    stdout, stderr = io.BytesIO(), io.BytesIO()
    assert RUNNER.main([], stdout=stdout, stderr=stderr) == 2
    wire = json.loads(stderr.getvalue())
    assert stdout.getvalue() == b""
    assert wire["ledger_created"] is True
    assert wire["ledger_consumed"] is True
    assert wire["actual_called"] is True
    assert wire["docker_or_cargo_called"] is True
    assert wire["q05b_artifact_written"] is True


def test_pure_builders_and_validators_are_not_execution_authority() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    doc = DOC_PATH.read_text(encoding="utf-8")
    assert "untrusted structural constructors" in doc
    assert "full replay" in doc and "live Git/raw checks" in doc
    assert "run_actual_v1" in source
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    subprocess_calls = [
        node for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(subprocess_calls) == 1
    assert "docker" not in source.lower() or "docker_or_cargo_called" in source
    assert "requests" not in source and "socket." not in source


def test_docs_freeze_attempt_history_authority_ledger_and_next_gate() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    for literal in (
        FROZEN_CONTRACT_DIGEST,
        "MISSING_REAL_V11_REPORT",
        "all 496 unordered alias pairs",
        "full replay",
        "GIT_OPTIONAL_LOCKS=0",
        "BRIDGE_STAGED_NOT_CONSUMED",
        "CONSUMED_OUTCOME_UNKNOWN",
        "actual_returned=false",
        "actual_returned=true",
        "Q1",
        "NOT_RUN",
        "no Meta-global adoption",
        "no caller-selected output path",
    ):
        assert literal in text
    assert "IMPLEMENTED_NOT_EXECUTED" in text
    assert "No actual attempt" in text


def test_exact_five_additive_isolation_and_unchanged_old_hashes() -> None:
    completed = subprocess.run(
        ["/usr/bin/git", "-c", "core.quotePath=false", "-C", str(WORKTREE), "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        check=True,
        stdout=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "GIT_OPTIONAL_LOCKS": "0"},
    )
    rows = [
        row.decode("utf-8")
        for row in completed.stdout.split(b"\0")
        if row
    ]
    if rows:
        assert {row[3:] for row in rows} == set(RUNNER.EXACT_FIVE_PATHS)
    else:
        diff = subprocess.check_output(
            [
                "/usr/bin/git", "-C", str(WORKTREE), "diff-tree", "--no-commit-id",
                "--name-status", "-r", "--no-renames", RUNNER.BASELINE_COMMIT, "HEAD",
            ],
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "GIT_OPTIONAL_LOCKS": "0"},
        ).decode("ascii").splitlines()
        assert len(diff) == 5
        assert all(row.startswith("A\t") for row in diff)
        assert {row.split("\t", 1)[1] for row in diff} == set(RUNNER.EXACT_FIVE_PATHS)
    for relative, digest in RUNNER.UNCHANGED_HEGEL_SHA256.items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == digest
