from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from assumption_agent.benchmarks.codex_action_budget import (
    CODEX_ACTION_BUDGET_POLICY_VERSION,
    CODEX_ACTION_BUDGET_UNIT,
    CodexActionBudgetCounter,
    audit_codex_action_budget,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    CODEX_ACTION_SUPERVISOR_PATH,
    _inspect_codex_tool_policy,
)


def _started(item_id: str, item_type: str = "command_execution") -> str:
    return json.dumps(
        {
            "type": "item.started",
            "item": {"id": item_id, "type": item_type},
        },
        separators=(",", ":"),
    )


def _completed() -> str:
    return json.dumps(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_output_tokens": 0,
            },
        },
        separators=(",", ":"),
    )


def _run_supervisor(
    tmp_path: Path,
    *,
    rows: list[str],
    limit: int,
    linger_seconds: int = 0,
    exit_code: int = 0,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the frozen Codex supervisor")
    receipt = tmp_path / "receipt.json"
    trace = tmp_path / "codex.txt"
    child_code = (
        "import sys,time\n"
        f"rows={rows!r}\n"
        "for row in rows:\n"
        " print(row, flush=True)\n"
        f"time.sleep({linger_seconds})\n"
        f"raise SystemExit({exit_code})\n"
    )
    completed = subprocess.run(
        [
            node,
            str(CODEX_ACTION_SUPERVISOR_PATH),
            "--limit",
            str(limit),
            "--receipt",
            str(receipt),
            "--trace",
            str(trace),
            "--process-scope",
            "process_group",
            "--",
            sys.executable,
            "-u",
            "-c",
            child_code,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return completed, trace, receipt


def test_counter_counts_only_well_formed_item_started_rows() -> None:
    counter = CodexActionBudgetCounter(limit=5)

    assert counter.observe_line("not-json") is False
    assert counter.observe_line(json.dumps({"type": "turn.started"})) is False
    assert counter.observe_line(_started("one")) is False
    assert counter.observe_line(_started("one")) is False
    assert counter.observe_line(_started("search", "web_search")) is False
    assert counter.observe_line(json.dumps({"type": "item.started"})) is False
    assert counter.observe_line(_started("four", "file_change")) is True

    assert counter.observed_action_starts == 5
    assert counter.invalid_action_event_count == 1
    assert counter.limit_reached is True


def test_supervisor_natural_completion_produces_valid_receipt(tmp_path: Path) -> None:
    completed, trace, receipt = _run_supervisor(
        tmp_path,
        rows=[
            json.dumps({"type": "turn.started"}),
            _started("one"),
            _completed(),
        ],
        limit=2,
    )

    assert completed.returncode == 0
    assert stat.S_IMODE(trace.stat().st_mode) == 0o644
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o644
    audit = audit_codex_action_budget(
        trace_path=trace,
        receipt_path=receipt,
        supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
        expected_limit=2,
    )
    assert audit.valid is True
    assert audit.observed_steps == 1
    assert audit.budget_reached is False
    assert audit.token_usage_complete is True


def test_supervisor_budget_hit_is_valid_truncation(tmp_path: Path) -> None:
    completed, trace, receipt = _run_supervisor(
        tmp_path,
        rows=[_started("one"), _started("two")],
        limit=2,
        linger_seconds=60,
    )

    assert completed.returncode == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["policy"] == CODEX_ACTION_BUDGET_POLICY_VERSION
    assert payload["unit"] == CODEX_ACTION_BUDGET_UNIT
    assert payload["sigterm_attempted"] is True
    assert payload["sigterm_delivered"] is True
    assert payload["agent_exit_confirmed"] is True
    assert payload["process_group_exit_confirmed"] is True
    assert payload["post_trigger_started_count"] == 0
    assert payload["budget_truncated"] is True
    audit = audit_codex_action_budget(
        trace_path=trace,
        receipt_path=receipt,
        supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
        expected_limit=2,
    )
    assert audit.valid is True
    assert audit.observed_steps == 2
    assert audit.budget_reached is True
    assert audit.budget_truncated is True
    assert audit.token_usage_complete is False


@pytest.mark.parametrize(
    "rows,exit_code",
    (
        ([_started("one")], 0),
        ([_started("one"), json.dumps({"type": "turn.completed", "usage": {}})], 0),
        ([_started("one"), json.dumps({"type": "turn.completed", "usage": {}})], 42),
    ),
)
def test_natural_incomplete_or_nonzero_exit_is_invalid(
    tmp_path: Path,
    rows: list[str],
    exit_code: int,
) -> None:
    completed, trace, receipt = _run_supervisor(
        tmp_path,
        rows=rows,
        limit=2,
        exit_code=exit_code,
    )

    assert completed.returncode == 70
    audit = audit_codex_action_budget(
        trace_path=trace,
        receipt_path=receipt,
        supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
        expected_limit=2,
    )
    assert audit.valid is False
    assert audit.error_type == "codex_action_budget_receipt_invalid"


def test_spawn_failure_is_invalid(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the frozen Codex supervisor")
    receipt = tmp_path / "receipt.json"
    trace = tmp_path / "codex.txt"
    completed = subprocess.run(
        [
            node,
            str(CODEX_ACTION_SUPERVISOR_PATH),
            "--limit",
            "2",
            "--receipt",
            str(receipt),
            "--trace",
            str(trace),
            "--process-scope",
            "process_group",
            "--",
            str(tmp_path / "missing-codex"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 70
    audit = audit_codex_action_budget(
        trace_path=trace,
        receipt_path=receipt,
        supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
        expected_limit=2,
    )
    assert audit.valid is False
    assert audit.error_type == "codex_action_budget_receipt_invalid"


def test_post_trigger_action_in_same_stdout_chunk_fails_closed(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the frozen Codex supervisor")
    receipt = tmp_path / "receipt.json"
    trace = tmp_path / "codex.txt"
    rows = [_started("one"), _started("two"), _started("overflow")]
    child_code = (
        "import sys,time\n"
        f"sys.stdout.write({(''.join(row + chr(10) for row in rows))!r})\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    completed = subprocess.run(
        [
            node,
            str(CODEX_ACTION_SUPERVISOR_PATH),
            "--limit",
            "2",
            "--receipt",
            str(receipt),
            "--trace",
            str(trace),
            "--process-scope",
            "process_group",
            "--",
            sys.executable,
            "-u",
            "-c",
            child_code,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))

    assert completed.returncode == 70
    assert payload["post_trigger_started_count"] == 1
    audit = audit_codex_action_budget(
        trace_path=trace,
        receipt_path=receipt,
        supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
        expected_limit=2,
    )
    assert audit.valid is False


def test_natural_exit_with_background_descendant_is_cleaned_and_invalid(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the frozen Codex supervisor")
    receipt = tmp_path / "receipt.json"
    trace = tmp_path / "codex.txt"
    child_code = (
        "import subprocess,sys\n"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
        "stderr=subprocess.DEVNULL)\n"
        f"print({_started('one')!r}, flush=True)\n"
        f"print({_completed()!r}, flush=True)\n"
    )
    completed = subprocess.run(
        [
            node,
            str(CODEX_ACTION_SUPERVISOR_PATH),
            "--limit",
            "2",
            "--receipt",
            str(receipt),
            "--trace",
            str(trace),
            "--process-scope",
            "process_group",
            "--",
            sys.executable,
            "-u",
            "-c",
            child_code,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))

    assert completed.returncode == 70
    assert payload["sigkill_delivered"] is True
    assert payload["process_group_exit_confirmed"] is True
    audit = audit_codex_action_budget(
        trace_path=trace,
        receipt_path=receipt,
        supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
        expected_limit=2,
    )
    assert audit.valid is False


def test_budget_receipt_trace_tamper_fails_closed(tmp_path: Path) -> None:
    _, trace, receipt = _run_supervisor(
        tmp_path,
        rows=[
            _started("one"),
            _completed(),
        ],
        limit=2,
    )
    trace.write_text(trace.read_text(encoding="utf-8") + _started("extra") + "\n")

    audit = audit_codex_action_budget(
        trace_path=trace,
        receipt_path=receipt,
        supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
        expected_limit=2,
    )
    assert audit.valid is False
    assert audit.error_type == "codex_action_budget_trace_mismatch"


def test_receipt_from_previous_attempt_cannot_validate_current_trace(
    tmp_path: Path,
) -> None:
    _, trace, receipt = _run_supervisor(
        tmp_path,
        rows=[_started("one"), _completed()],
        limit=2,
    )
    stale_receipt = receipt.read_text(encoding="utf-8")
    _run_supervisor(
        tmp_path,
        rows=[_started("one"), _completed()],
        limit=2,
    )
    receipt.write_text(stale_receipt, encoding="utf-8")

    audit = audit_codex_action_budget(
        trace_path=trace,
        receipt_path=receipt,
        supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
        expected_limit=2,
    )
    assert audit.valid is False
    assert audit.error_type == "codex_action_budget_trace_mismatch"


def test_web_search_counts_as_step_and_remains_policy_invalid(tmp_path: Path) -> None:
    trace = tmp_path / "codex.txt"
    trace.write_text(_started("search", "web_search") + "\n", encoding="utf-8")
    counter = CodexActionBudgetCounter(limit=100)
    counter.observe_line(trace.read_text(encoding="utf-8"))

    assert counter.observed_action_starts == 1
    audit = _inspect_codex_tool_policy(trace)
    assert audit.valid is False
    assert audit.error_type == "model_remote_tool_policy_violation"


def test_dedicated_container_task_scan_cleans_zombie_leader_worker_thread(
    tmp_path: Path,
) -> None:
    node_image = (
        "node@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0"
    )
    gcc = shutil.which("gcc")
    docker = shutil.which("docker")
    if gcc is None or docker is None:
        pytest.skip("gcc and Docker are required for the thread-group containment probe")
    if subprocess.run(
        [docker, "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0:
        pytest.skip("Docker daemon is unavailable")
    if subprocess.run(
        [docker, "image", "inspect", node_image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0:
        pytest.skip("the pinned offline Node image is not cached")

    source = tmp_path / "thread_escape.c"
    helper = tmp_path / "thread_escape"
    source.write_text(
        r"""
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

static const char *marker_path;
static const char *tid_path;
static const char *ready_path;
static volatile sig_atomic_t worker_ready = 0;

static void write_number(const char *path, long value) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) _exit(20);
    dprintf(fd, "%ld\n", value);
    close(fd);
}

static void *worker(void *unused) {
    (void)unused;
    write_number(tid_path, syscall(SYS_gettid));
    write_number(ready_path, 1);
    worker_ready = 1;
    sleep(3);
    write_number(marker_path, 1);
    sleep(30);
    return NULL;
}

int main(int argc, char **argv) {
    pthread_t thread;
    if (argc != 5) return 2;
    marker_path = argv[1];
    tid_path = argv[2];
    ready_path = argv[3];
    if (setsid() < 0) return 3;
    write_number(argv[4], getpid());
    if (pthread_create(&thread, NULL, worker, NULL) != 0) return 4;
    while (!worker_ready) sched_yield();
    syscall(SYS_exit, 0);
    return 5;
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [gcc, "-O2", "-pthread", str(source), "-o", str(helper)],
        check=True,
        capture_output=True,
        text=True,
    )
    output_root = tmp_path / "out"
    output_root.mkdir()
    container = f"codex-action-thread-{os.getpid()}-{tmp_path.name}"
    started = subprocess.run(
        [
            docker,
            "run",
            "-d",
            "--name",
            container,
            "--network",
            "none",
            "-v",
            f"{CODEX_ACTION_SUPERVISOR_PATH}:/probe/supervisor.mjs:ro",
            "-v",
            f"{helper}:/probe/thread_escape:ro",
            "-v",
            f"{output_root}:/out",
            node_image,
            "sleep",
            "infinity",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if started.returncode != 0:
        pytest.fail(f"failed to start containment probe container: {started.stderr}")

    def run_case(name: str, *, limit: int, completed: bool) -> tuple[dict, object]:
        case_dir = output_root / name
        case_dir.mkdir()
        action = _started("one")
        terminal = f"printf '%s\\n' {_completed()!r}; exit 0" if completed else "sleep 30"
        script = (
            f"rm -f /out/{name}/marker /out/{name}/tid /out/{name}/ready "
            f"/out/{name}/tgid; "
            f"/probe/thread_escape /out/{name}/marker /out/{name}/tid "
            f"/out/{name}/ready /out/{name}/tgid >/dev/null 2>&1 & "
            f"i=0; while [ ! -s /out/{name}/ready ] && [ $i -lt 200 ]; do "
            "i=$((i+1)); sleep 0.01; done; "
            f"test -s /out/{name}/ready || exit 91; sleep 0.1; "
            f"printf '%s\\n' {action!r}; {terminal}"
        )
        receipt = case_dir / "receipt.json"
        trace = case_dir / "codex.txt"
        process = subprocess.run(
            [
                docker,
                "exec",
                container,
                "node",
                "/probe/supervisor.mjs",
                "--limit",
                str(limit),
                "--receipt",
                f"/out/{name}/receipt.json",
                "--trace",
                f"/out/{name}/codex.txt",
                "--process-scope",
                "dedicated_container",
                "--",
                "sh",
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert stat.S_IMODE(trace.stat().st_mode) == 0o644
        assert stat.S_IMODE(receipt.stat().st_mode) == 0o644
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        audit = audit_codex_action_budget(
            trace_path=trace,
            receipt_path=receipt,
            supervisor_path=CODEX_ACTION_SUPERVISOR_PATH,
            expected_limit=limit,
            expected_process_scope="dedicated_container",
        )
        tgid = int((case_dir / "tgid").read_text(encoding="utf-8").strip())
        tid = int((case_dir / "tid").read_text(encoding="utf-8").strip())
        task_gone = subprocess.run(
            [
                docker,
                "exec",
                container,
                "sh",
                "-c",
                f"test ! -e /proc/{tgid}/task/{tid}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert task_gone.returncode == 0
        assert payload["process_task_scan_complete"] is True
        assert payload["residual_process_count"] >= 1
        assert payload["residual_tid_count"] >= 1
        assert payload["agent_processes_exit_confirmed"] is True
        return {"process": process, "payload": payload, "marker": case_dir / "marker"}, audit

    try:
        natural, natural_audit = run_case("natural", limit=2, completed=True)
        assert natural["process"].returncode == 70
        assert natural_audit.valid is False
        assert natural_audit.error_type == "codex_action_budget_receipt_invalid"

        budget, budget_audit = run_case("budget", limit=1, completed=False)
        assert budget["process"].returncode == 0
        assert budget["payload"]["budget_truncated"] is True
        assert budget_audit.valid is True

        time.sleep(3.2)
        assert not natural["marker"].exists()
        assert not budget["marker"].exists()
    finally:
        subprocess.run(
            [docker, "rm", "-f", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
