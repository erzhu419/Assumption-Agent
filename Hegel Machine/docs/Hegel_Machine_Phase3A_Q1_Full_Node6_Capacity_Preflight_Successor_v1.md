# Hegel Machine Phase-3A-Q1 Full Node-6 Capacity-Preflight Successor v4

## 1. Position and claim

This exact source-closed contract defines a distinct outer successor-v4 after successor-v3
consumed its authorization and terminated with a known endpoint-sequence
failure. It retains action id `full-node6-capacity-preflight-v1`; it does not
resume v3 or reuse any v3 measurement work product.

The only permitted successful claim remains:

```text
PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
```

Q1 remains `NOT_RUN`, `0/20`, mask zero, with no receipt and eight null output
slots. Target truth, split and role inputs remain inaccessible. This contract
does not authorize adoption, promotion, production, a current-pointer write or
an `ACTIVE` transition.

## 2. Bound predecessor and terminal successor-v3 evidence

The Q0.5b attempt-2 predecessor remains terminally qualified at `20/20`, while
Q1 is not run. Successor-v4 additionally binds the complete terminal v3 outer
campaign:

- source commit `ea667c9b7ecdb22368a9ed61982424497e02bd6d`;
- capsule `sha256:2b75870fe25903afa129e0f277a38fa263f5ddc65d02c37f4d6c35080d766e37`;
- consumed file SHA-256
  `21f7c892cfb9be8af76642e13b2491104b9aa8a95c576c378c56b11b4d858e01`
  and record
  `sha256:63dd4ae1bbcb58d8b733dc0a85c61df869f71d89ba45825f696cd3ac95e07124`;
- terminal file SHA-256
  `a4e28d73f3542efa7d296974584635b4fc3a99c5b8aa6eae8691a8926d6010a8`
  and record
  `sha256:3bee33a58091eb41ffd56ea1d1ee0b0e41bb1baec45e7af5dbab9b9ee85a9ecf`;
- outcome
  `sha256:3a948c3c4a51d464e4e2cb46ab53402f09e9b6896fe0382a69f0344c494b0944`
  and external outcome-file SHA-256
  `4c10dc7307bac862980000be2bce1cb6c00f8679e30032ed5321b33a66ed4f3a`;
- trajectory file SHA-256
  `7177b757a29f533c56eccbb6e574b8605bb9741b2ee052609357def165a3b234`
  and terminal-trajectory file SHA-256
  `b05e0d6aed098e94231b29da005df13566c560a1a8c3e665e49cab36b6f91e8d`;
- terminal error `FULL_NODE6_ENDPOINT_SEQUENCE_FAILED`, detail
  `an ordered endpoint did not exit cleanly`.

The trajectory contains exactly 31 events. Python Odd and Rust Odd each started
and completed once, each returned one with
`FAIL_Q1_PREFLIGHT_LATE_LOWER_DEPTH_BANK_MUTATION`; Sink/trusted-host replay did
not start. Both Odd measurements conservatively count. The outcome is a known
terminal failure, not a recovery point or reusable output.

## 3. Conservative additional-measurement boundary

Both v3 Odd measurements count. Any v4 measurement is explicitly additional,
scoped only to one exact future v4 capsule, and is not a resume or continuation
of v3. No v3 work product may be reused. The v3 capsule, ledger, evidence,
trajectory, run, probe and artifact
roots are read-only evidence and are never cleanup, absence or resume targets.

This document and the implemented contract do not themselves authorize v4
actual execution. Only a future user's exact authorization of the distinct
outer successor-v4 actual campaign may authorize one fresh logical campaign.
The exact authorization behavior text is:

```text
授权 Q1 full-node6 capacity-preflight diagnostic successor-v4 distinct-outer actual；确认 successor-v3 known terminal failure、terminal no-resume，且 Python Odd measurement 与 Rust Odd measurement 各一次 conservatively count，Sink/trusted host replay 0；另行授权 exact v4 capsule-only additional fresh remeasurement；禁止恢复 successor-v3 或复用 successor-v3 work product
```

That future authorization remains bounded to at most one supervisor Popen and
at most one quotient measurement for each endpoint/signature pair.

An exact pre-consumption `PENDING` result may be replayed only for the same v4
capsule digest. It is not a campaign execution, consumes no authorization and
invokes no supervisor. After consumption, any failure or unknown measurement
outcome is terminal no-resume.

## 4. Fresh successor-v4 roots and admission order

The contract uses only fresh `q1-full-node6-capacity-v4-*` lock, ledger,
evidence, trajectory, run and probe roots, plus
`artifacts/phase3_q1_full_node6_capacity_preflight_v4.json`. Required source
parent is `ea667c9b7ecdb22368a9ed61982424497e02bd6d`. The exact-addition registry
includes `Hegel_Machine_Current_State_Reconciliation_v5.md`; the exact-modification
registry contains only the Python technical-action and partition-snapshot cores
and their focused tests.

All static and live admission gates precede authorization consumption and
supervisor Popen. The order reserves `RUNTIME_ISOLATION_LIVE_GAUNTLET` after
CPU topology/affinity and before memory/storage reservation. Its required
probe is `/bin/true`; the receipt must state zero quotient-engine invocations,
zero signature selections, zero production endpoint starts and zero supervisor
Popen. During an actual run, that exact receipt is the first durable semantic
evidence in the consumed ledger and is fsynced before supervisor Popen; a
pre-consumption `PENDING` writes no ledger. The gauntlet is implemented, but it
has not yet been executed, preflighted or authorized under successor-v4.

After the outer child returns on every path, including normal return codes zero
and one, timeout and exception, the wrapper records one cleanup receipt. It
requires process-group absence, exact absence of all three production and all
three control containers, probe absence and exact absence of the three scratch
roots. Run evidence is retained. Any cleanup item that cannot be proved changes
the result to `FULL_NODE6_CONSUMED_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME`.

The technical execution order remains strict: Python must exit completely
before Rust starts, and Rust must exit completely before trusted-host replay
starts. Endpoints remain target-blind and network-disabled.

## 5. Terminal routing

Only `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY` can support a later byte-exact
formal-resource amendment. A capacity guard is inconclusive; disagreement,
isolation failure, replay failure or an unknown outcome is terminal failure.
None starts Q1 or establishes a scientific result.

Formal Q1 execution remains a separate later `phase3-q1-start` action after a
saturated diagnostic and its mechanical amendment. Successor-v4 provides no
Q1 start action.
