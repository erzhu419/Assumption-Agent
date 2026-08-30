# Hegel Machine Current-State Reconciliation v5

## Status

This dashboard follows the consumed successor-v3 campaign. It supersedes v4
only as current narration; earlier reconciliations and all bound evidence remain
immutable history. This document is not execution authorization.

Q0.5b attempt 2 remains terminal success at 20/20 with Q1 `NOT_RUN`.
Successor-v3 capsule
`sha256:2b75870fe25903afa129e0f277a38fa263f5ddc65d02c37f4d6c35080d766e37`
consumed one authorization and one supervisor Popen, then terminated
`FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME`. Python Odd and Rust Odd each
started and completed once and each returned
`FAIL_Q1_PREFLIGHT_LATE_LOWER_DEPTH_BANK_MUTATION`; Sink/trusted-host replay did
not start. Q1 remains `NOT_RUN`, 0/20, mask zero, null receipt and eight null
output slots.

## Bound successor-v3 evidence

- source commit: `ea667c9b7ecdb22368a9ed61982424497e02bd6d`;
- consumed file/record:
  `21f7c892cfb9be8af76642e13b2491104b9aa8a95c576c378c56b11b4d858e01` /
  `sha256:63dd4ae1bbcb58d8b733dc0a85c61df869f71d89ba45825f696cd3ac95e07124`;
- terminal file/record:
  `a4e28d73f3542efa7d296974584635b4fc3a99c5b8aa6eae8691a8926d6010a8` /
  `sha256:3bee33a58091eb41ffd56ea1d1ee0b0e41bb1baec45e7af5dbab9b9ee85a9ecf`;
- outcome digest:
  `sha256:3a948c3c4a51d464e4e2cb46ab53402f09e9b6896fe0382a69f0344c494b0944`;
- external outcome-file SHA-256:
  `4c10dc7307bac862980000be2bce1cb6c00f8679e30032ed5321b33a66ed4f3a`;
- 31-event trajectory/terminal files:
  `7177b757a29f533c56eccbb6e574b8605bb9741b2ee052609357def165a3b234` /
  `b05e0d6aed098e94231b29da005df13566c560a1a8c3e665e49cab36b6f91e8d`.

Successor-v3 is permanently no-resume. Its run snapshots, binaries, endpoint
outputs, resource evidence and other measurement work products cannot seed or
continue another campaign. A later contract may read only the exact consumed,
terminal, outcome and trajectory evidence to bind this terminal boundary.

## Distinct successor-v4

Successor-v4 is an implemented outer contract only. It uses fresh
`q1-full-node6-capacity-v4-*` lock, ledger, evidence, trajectory, run and probe
roots and fresh artifact
`artifacts/phase3_q1_full_node6_capacity_preflight_v4.json`. It is not a v3
retry. Python Odd and Rust Odd from v3 each conservatively count as one prior
measurement; any v4 remeasurement is explicitly additional and limited to one
future exact v4 capsule under new exact user authorization.

After every outer child return path, including normal return codes zero and one,
timeout and exception, v4 records process-group absence, exact production and
control-container absence, probe absence and three-scratch-root absence. The
cleanup retains run evidence. Any cleanup uncertainty forces
`FULL_NODE6_CONSUMED_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME`.

The only permitted successful claim remains
`PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY`. Network, Q1 start, target/split/role
access, adoption, promotion, production, `ACTIVE` and current-pointer writes
remain forbidden.

The exact future authorization behavior text is:

```text
授权 Q1 full-node6 capacity-preflight diagnostic successor-v4 distinct-outer actual；确认 successor-v3 known terminal failure、terminal no-resume，且 Python Odd measurement 与 Rust Odd measurement 各一次 conservatively count，Sink/trusted host replay 0；另行授权 exact v4 capsule-only additional fresh remeasurement；禁止恢复 successor-v3 或复用 successor-v3 work product
```
