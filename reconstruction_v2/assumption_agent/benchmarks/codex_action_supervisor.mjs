#!/usr/bin/env node

import { createHash, randomBytes } from "node:crypto";
import { spawn } from "node:child_process";
import {
  chmodSync,
  closeSync,
  fchmodSync,
  openSync,
  readdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
  writeSync,
} from "node:fs";

const POLICY = "codex_jsonl_action_start_budget_v1";
const UNIT = "codex_action_start_v1";
const OVERFLOW_POLICY = "terminate_on_limit_action_start_v1";
const SIGKILL_GRACE_UPPER_BOUND_SECONDS = 15;
const SUPERVISOR_START_EVENT = "assumption.action_budget.started";

function parseArgs(argv) {
  const separator = argv.indexOf("--");
  if (separator < 0) throw new Error("missing command separator");
  const options = argv.slice(0, separator);
  const command = argv.slice(separator + 1);
  let limit = null;
  let receiptPath = null;
  let tracePath = null;
  let processScope = null;
  for (let index = 0; index < options.length; index += 2) {
    if (options[index] === "--limit") limit = Number(options[index + 1]);
    else if (options[index] === "--receipt") receiptPath = options[index + 1];
    else if (options[index] === "--trace") tracePath = options[index + 1];
    else if (options[index] === "--process-scope") processScope = options[index + 1];
    else throw new Error(`unsupported option: ${options[index]}`);
  }
  if (!Number.isInteger(limit) || limit <= 0) throw new Error("invalid action limit");
  if (!receiptPath) throw new Error("missing receipt path");
  if (!tracePath) throw new Error("missing trace path");
  if (!["process_group", "dedicated_container"].includes(processScope)) {
    throw new Error("invalid process scope");
  }
  if (command.length === 0) throw new Error("missing Codex command");
  return { limit, receiptPath, tracePath, processScope, command };
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]),
    );
  }
  return value;
}

function stableHash(value) {
  return createHash("sha256")
    .update(JSON.stringify(canonicalize(value)))
    .digest("hex");
}

function writeReceipt(path, payload) {
  const complete = { ...payload, receipt_hash: stableHash(payload) };
  const temporary = `${path}.tmp-${process.pid}`;
  writeFileSync(temporary, `${JSON.stringify(complete, null, 2)}\n`, { mode: 0o644 });
  chmodSync(temporary, 0o644);
  renameSync(temporary, path);
}

function isTransientProcError(error) {
  return Boolean(error && ["ENOENT", "ESRCH"].includes(error.code));
}

function readTaskRecord(tgid, tid) {
  try {
    const stat = readFileSync(`/proc/${tgid}/task/${tid}/stat`, "utf8");
    const close = stat.lastIndexOf(")");
    if (close < 0) return { complete: false, record: null };
    const fields = stat.slice(close + 2).trim().split(/\s+/);
    const state = fields[0];
    const processGroup = Number(fields[2]);
    const startTime = fields[19];
    if (!state || !Number.isInteger(processGroup) || !startTime) {
      return { complete: false, record: null };
    }
    return {
      complete: true,
      record: {
        identity: `${tid}:${startTime}`,
        process_group: processGroup,
        state,
        tgid,
        tid,
      },
    };
  } catch (error) {
    return { complete: isTransientProcError(error), record: null };
  }
}

function taskSnapshot() {
  const rows = new Map();
  let complete = true;
  let processNames;
  try {
    processNames = readdirSync("/proc");
  } catch {
    return { complete: false, rows };
  }
  for (const name of processNames) {
    if (!/^\d+$/.test(name)) continue;
    const tgid = Number(name);
    let taskNames;
    try {
      taskNames = readdirSync(`/proc/${tgid}/task`);
    } catch (error) {
      if (!isTransientProcError(error)) complete = false;
      continue;
    }
    for (const taskName of taskNames) {
      if (!/^\d+$/.test(taskName)) continue;
      const tid = Number(taskName);
      const inspected = readTaskRecord(tgid, tid);
      if (!inspected.complete) complete = false;
      if (inspected.record) rows.set(inspected.record.identity, inspected.record);
    }
  }
  return { complete, rows };
}

function taskIsLive(record) {
  return !["Z", "X", "x"].includes(record.state);
}

function validUsage(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const inputTokens = value.input_tokens;
  const outputTokens = value.output_tokens;
  if (
    !Number.isInteger(inputTokens)
      || inputTokens < 0
      || !Number.isInteger(outputTokens)
      || outputTokens < 0
  ) return null;
  const optional = {};
  for (const key of ["cached_input_tokens", "reasoning_output_tokens"]) {
    const item = value[key];
    if (item === undefined || item === null) continue;
    if (!Number.isInteger(item) || item < 0) return null;
    optional[key] = item;
  }
  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
    ...optional,
  };
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

const {
  limit,
  receiptPath,
  tracePath,
  processScope,
  command,
} = parseArgs(process.argv.slice(2));
const supervisorHash = createHash("sha256")
  .update(readFileSync(new URL(import.meta.url)))
  .digest("hex");
const runNonce = randomBytes(16).toString("hex");
const traceFd = openSync(tracePath, "w", 0o644);
fchmodSync(traceFd, 0o644);
const traceHasher = createHash("sha256");

function writeTrace(value) {
  const chunk = Buffer.isBuffer(value) ? value : Buffer.from(value, "utf8");
  process.stdout.write(chunk);
  writeSync(traceFd, chunk);
  traceHasher.update(chunk);
}

const startRow = {
  type: SUPERVISOR_START_EVENT,
  policy: POLICY,
  unit: UNIT,
  limit,
  run_nonce: runNonce,
  supervisor_hash: supervisorHash,
};
writeTrace(`${JSON.stringify(startRow)}\n`);
const baselineSnapshot = processScope === "dedicated_container"
  ? taskSnapshot()
  : { complete: true, rows: new Map() };
const processBaseline = baselineSnapshot.rows;
const processBaselineHash = stableHash(Array.from(processBaseline.keys()).sort());
let processTaskScanComplete = baselineSnapshot.complete;

let buffer = "";
let observedSteps = 0;
let invalidActionEvents = 0;
let budgetReached = false;
let spawnError = false;
let sigtermAttempted = false;
let sigtermDelivered = false;
let sigkillAttempted = false;
let sigkillDelivered = false;
let postTriggerStartedCount = 0;
let turnCompletedCount = 0;
let turnFailedCount = 0;
let invalidTerminalUsageCount = 0;
let tokenUsage = null;
const actionEvents = [];
let killTimer = null;
let child = null;

function processGroupExists() {
  if (!child || !Number.isInteger(child.pid)) return false;
  const snapshot = taskSnapshot();
  if (!snapshot.complete) processTaskScanComplete = false;
  const exists = Array.from(snapshot.rows.values()).some((record) => (
    record.process_group === child.pid && taskIsLive(record)
  ));
  return exists || !snapshot.complete;
}

function deliverGroupSignal(signalName) {
  if (!child || !Number.isInteger(child.pid)) return false;
  try {
    process.kill(-child.pid, signalName);
    return true;
  } catch {
    return false;
  }
}

function terminateForBudget() {
  if (sigtermAttempted) return;
  budgetReached = true;
  sigtermAttempted = true;
  sigtermDelivered = deliverGroupSignal("SIGTERM");
  killTimer = setTimeout(() => {
    if (!processGroupExists()) return;
    sigkillAttempted = true;
    sigkillDelivered = deliverGroupSignal("SIGKILL");
  }, SIGKILL_GRACE_UPPER_BOUND_SECONDS * 1000);
  killTimer.unref();
}

function inspectLine(line) {
  let row;
  try {
    row = JSON.parse(line);
  } catch {
    return;
  }
  if (!row || typeof row !== "object" || Array.isArray(row)) return;
  if (row.type === "turn.completed") {
    turnCompletedCount += 1;
    const parsed = validUsage(row.usage);
    if (parsed === null) invalidTerminalUsageCount += 1;
    else tokenUsage = parsed;
  } else if (row.type === "turn.failed") {
    turnFailedCount += 1;
  }
  if (row.type !== "item.started") return;
  const itemValid = row.item && typeof row.item === "object" && !Array.isArray(row.item);
  const itemId = itemValid && typeof row.item.id === "string" ? row.item.id : "";
  const itemType = itemValid && typeof row.item.type === "string" ? row.item.type : "";
  const malformed = !itemValid || !itemId || !itemType;
  if (malformed) invalidActionEvents += 1;
  observedSteps += 1;
  actionEvents.push({
    event_index: observedSteps,
    item_id: itemId,
    item_type: itemType,
    malformed,
  });
  if (budgetReached) postTriggerStartedCount += 1;
  if (observedSteps === limit) terminateForBudget();
}

async function confirmProcessGroupExit() {
  if (!processGroupExists()) return true;
  if (!sigkillAttempted) {
    sigkillAttempted = true;
    sigkillDelivered = deliverGroupSignal("SIGKILL");
  }
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (!processGroupExists()) return true;
    await delay(100);
  }
  return !processGroupExists();
}

function residualTaskGroups() {
  if (processScope !== "dedicated_container") {
    return { complete: true, groups: new Map() };
  }
  const snapshot = taskSnapshot();
  if (!snapshot.complete) processTaskScanComplete = false;
  const groups = new Map();
  for (const [identity, record] of snapshot.rows.entries()) {
    if (
      processBaseline.has(identity)
        || record.tgid === process.pid
        || !taskIsLive(record)
    ) continue;
    if (!groups.has(record.tgid)) groups.set(record.tgid, new Map());
    groups.get(record.tgid).set(record.identity, record);
  }
  return { complete: snapshot.complete, groups };
}

function deliverTaskGroupKill(tgid, records) {
  const targets = [tgid, ...Array.from(records.values()).map((record) => record.tid)];
  for (const target of new Set(targets)) {
    try {
      process.kill(target, "SIGKILL");
      return true;
    } catch {
      // A zombie leader may reject the signal; try a live non-leader TID.
    }
  }
  return false;
}

async function cleanupResidualProcesses() {
  const observedGroups = new Set();
  const observedTasks = new Set();
  const attemptedGroups = new Set();
  const deliveredGroups = new Set();
  for (let attempt = 0; attempt <= 50; attempt += 1) {
    const residual = residualTaskGroups();
    for (const [tgid, records] of residual.groups.entries()) {
      observedGroups.add(tgid);
      for (const identity of records.keys()) observedTasks.add(identity);
    }
    if (residual.groups.size === 0) {
      return {
        process_count: observedGroups.size,
        tid_count: observedTasks.size,
        sigkill_attempted_count: attemptedGroups.size,
        sigkill_delivered_count: deliveredGroups.size,
        exit_confirmed: residual.complete && processTaskScanComplete,
      };
    }
    for (const [tgid, records] of residual.groups.entries()) {
      attemptedGroups.add(tgid);
      if (deliverTaskGroupKill(tgid, records)) deliveredGroups.add(tgid);
    }
    if (attempt === 50) break;
    await delay(100);
  }
  const remaining = residualTaskGroups();
  return {
    process_count: observedGroups.size,
    tid_count: observedTasks.size,
    sigkill_attempted_count: attemptedGroups.size,
    sigkill_delivered_count: deliveredGroups.size,
    exit_confirmed: (
      remaining.groups.size === 0
        && remaining.complete
        && processTaskScanComplete
    ),
  };
}

async function containUnkillableResiduals() {
  if (processScope === "dedicated_container" && process.pid !== 1) {
    try {
      process.kill(1, "SIGKILL");
    } catch {
      // Fall through to a non-returning cleanup loop.
    }
  }
  while (true) {
    const residual = residualTaskGroups();
    for (const [tgid, records] of residual.groups.entries()) {
      deliverTaskGroupKill(tgid, records);
    }
    await delay(1000);
  }
}

child = spawn(command[0], command.slice(1), {
  detached: true,
  env: process.env,
  stdio: ["inherit", "pipe", "pipe"],
});

child.stdout.on("data", (chunk) => {
  writeTrace(chunk);
  buffer += chunk.toString("utf8");
  while (buffer.includes("\n")) {
    const index = buffer.indexOf("\n");
    const line = buffer.slice(0, index).trim();
    buffer = buffer.slice(index + 1);
    if (line) inspectLine(line);
  }
});
child.stderr.pipe(process.stderr);

child.on("error", () => {
  spawnError = true;
  process.stderr.write("codex action supervisor failed to spawn child\n");
});

child.on("close", async (code, signalName) => {
  const finalLine = buffer.trim();
  if (finalLine) inspectLine(finalLine);
  if (killTimer) clearTimeout(killTimer);
  const rawProcessGroupExitConfirmed = await confirmProcessGroupExit();
  const residual = await cleanupResidualProcesses();
  if (!residual.exit_confirmed) await containUnkillableResiduals();
  const processGroupExitConfirmed = (
    rawProcessGroupExitConfirmed && processTaskScanComplete
  );
  const agentProcessesExitConfirmed = (
    residual.exit_confirmed && processTaskScanComplete
  );
  closeSync(traceFd);
  const tokenUsageComplete = Boolean(
    turnCompletedCount === 1
      && turnFailedCount === 0
      && invalidTerminalUsageCount === 0
      && tokenUsage !== null,
  );
  const budgetTruncated = Boolean(
    budgetReached
      && !tokenUsageComplete
      && (sigtermDelivered || sigkillDelivered),
  );
  const payload = {
    policy: POLICY,
    unit: UNIT,
    overflow_policy: OVERFLOW_POLICY,
    limit,
    observed_steps: observedSteps,
    budget_reached: budgetReached,
    trigger_event_index: budgetReached ? limit : null,
    action_event_hash: stableHash(actionEvents),
    invalid_action_event_count: invalidActionEvents,
    run_nonce: runNonce,
    trace_sha256: traceHasher.digest("hex"),
    turn_completed_observed: turnCompletedCount > 0,
    turn_completed_count: turnCompletedCount,
    turn_failed_count: turnFailedCount,
    invalid_terminal_usage_count: invalidTerminalUsageCount,
    token_usage_complete: tokenUsageComplete,
    token_usage: tokenUsage || {},
    spawn_error: spawnError,
    sigterm_attempted: sigtermAttempted,
    sigterm_delivered: sigtermDelivered,
    sigkill_attempted: sigkillAttempted,
    sigkill_delivered: sigkillDelivered,
    sigkill_grace_upper_bound_seconds: SIGKILL_GRACE_UPPER_BOUND_SECONDS,
    agent_exit_code: code,
    agent_exit_signal: signalName,
    agent_exit_confirmed: true,
    process_group_exit_confirmed: processGroupExitConfirmed,
    process_scope: processScope,
    process_baseline_hash: processBaselineHash,
    process_task_scan_complete: processTaskScanComplete,
    residual_process_count: residual.process_count,
    residual_tid_count: residual.tid_count,
    residual_sigkill_attempted_count: residual.sigkill_attempted_count,
    residual_sigkill_delivered_count: residual.sigkill_delivered_count,
    agent_processes_exit_confirmed: agentProcessesExitConfirmed,
    post_trigger_started_count: postTriggerStartedCount,
    budget_truncated: budgetTruncated,
    supervisor_hash: supervisorHash,
    raw_content_persisted: false,
  };
  try {
    writeReceipt(receiptPath, payload);
  } catch (error) {
    process.stderr.write(`failed to write Codex action-budget receipt: ${error.message}\n`);
    process.exitCode = 70;
    return;
  }
  const normalCompletion = (
    !spawnError && code === 0 && signalName === null && tokenUsageComplete
  );
  const controlledBudgetTermination = (
    budgetReached && (sigtermDelivered || sigkillDelivered)
  );
  const normalCompletionSignalState = budgetReached
    ? sigtermAttempted
    : !sigtermAttempted && !sigkillAttempted && residual.process_count === 0;
  const evidenceValid = (
    processGroupExitConfirmed
      && processTaskScanComplete
      && agentProcessesExitConfirmed
      && invalidActionEvents === 0
      && postTriggerStartedCount === 0
      && observedSteps <= limit
      && (
        controlledBudgetTermination
          || (normalCompletion && normalCompletionSignalState)
      )
  );
  process.exitCode = evidenceValid ? 0 : 70;
});
