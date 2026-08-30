# Hegel Machine Current-State Reconciliation v6

## Status

This dashboard follows the consumed successor-v4 campaign and supersedes v5
only as current narration. Earlier reconciliations and all bound evidence are
immutable history. This document is not execution authorization.

Q0.5b attempt 2 remains terminal success at 20/20. Q1 remains `NOT_RUN`, 0/20, mask zero, with null receipt and eight null output slots. Network, Q1 start,
target/split/role access, adoption, promotion, production, `ACTIVE` and the
current pointer remain forbidden.

## Bound successor-v4 terminal evidence

Successor-v4 capsule
`sha256:73a07996d7acb21ae3c6d8b35a096850d7fe5f854269f2b13da67da6c595b6a0`
at source commit `5626e1d025e604fc4df5824f40a337a4e756c3dc` consumed exactly one
authorization and launched exactly one supervisor Popen. The supervisor
returned one and the campaign terminated
`FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME`; replay did not pass and no
artifact was published.

The durable bindings are:

- consumed record/file:
  `sha256:fc719de068b9fa8db653359b9048be80b0f4e1140a2858ea32d1ce2a92afbb8e` /
  `sha256:aa7fc01c6b87578fedd5d8f96e1ab00d8cdaec73f1a5dd5c0f3a10101930ac66`;
- terminal record/file:
  `sha256:ddf92134dc2b036084091baf7bc6063c8ce1bf41060f8ce2ae6d26a097d60bfe` /
  `sha256:926e4943846ccc9d7fbe6fb27eed9bd49bcc238ddd42620dc86008ad3b80a5e4`;
- outcome digest/file:
  `sha256:3e2d6780a75a8da0e91b4eca5cbdda6cdff1c2132b47e8ac6db2da217c824c6c` /
  `sha256:ed860e4a7a4128b14a10dc68e41909afeae2db43998318065359fe9558328795`;
- 31-event trajectory/terminal files:
  `sha256:aedeaa0d089f57219609087fc01990dce90803e26cb9ea2a5e879e4c0a7c70b4` /
  `sha256:c451a18be5542ebed13ebd51f576567b55135510d052fc8decbed08260f86d1f`.

Python started and completed once with rc=1 and stdout SHA-256
`sha256:f10025c57b41a12268743d0c13cea17f7fa0660ff22d7586bf35315ec746dd3b`.
It reported `FAIL_Q1_PREFLIGHT_BEHAVIOR`: `one AST has two behaviors`. Only
after Python exited and its evidence persisted did Rust start. Rust completed
once with rc=1 and stdout SHA-256
`sha256:4051a7222ad27e1d4aace73d53831fe63ed28f4b38b2d57d9e1dac9ef6347585`.
It reported the same code: `one canonical AST has two behaviors`.
Sink/trusted-host replay did not start. Python Odd and Rust Odd each
conservatively count once; Sink counts zero.

The bounded counterexample is `difference(x,x)`: bottom propagation conflicts
with canonical zero. This locates the v4 behavior conflict but is not a resumable
state or reusable work product.

The Python/Rust/aggregate resource receipt hashes are
`sha256:4b88c0c37d3aec1cf7c653ada1a02b49abfd4ea90e577576b2817fe41f7b994d`,
`sha256:f6db20b7920ce26e409fb666a6b3deaa2a109e18462c6faf162eeae59c296002`
and
`sha256:dc455f467008798f20c02b1223da840854dddd4024a941793c121dcb6dde0a2a`.
They record one row per endpoint, strict Python-then-Rust order, maximum one
live measurement container and no overlap. Scratch cleanup hash
`sha256:edfa3752d74c47fa079ca020096ad83fcb32d8b66542e27f3c940c0503cefc99`
has no failure rows and proves both roots absent. The complete outer
post-consumption cleanup receipt is `PASS` for process-group, production and
control-container, probe-root and production-scratch absence.

Successor-v4 is permanently no-resume. Its run snapshots, binaries, endpoint
outputs, resource receipts and all other measurement work products may not seed
or continue another campaign.

## Distinct successor-v5

Successor-v5 is an implemented outer contract only. It uses fresh
`q1-full-node6-capacity-v5-*` lock, ledger, evidence, trajectory, run and probe
roots and
`artifacts/phase3_q1_full_node6_capacity_preflight_v5.json`. It is not a v4
retry. Any v5 remeasurement is explicitly additional and limited to one exact
future v5 capsule under new exact user authorization.

An exact pre-consumption `PENDING` replay for that same capsule does not count
as a campaign, consume authorization, invoke the supervisor, or start a
production endpoint or quotient engine. Any post-consumption failure, unknown
measurement outcome or cleanup absence that cannot be proved is terminal
no-resume. The only permitted successful claim remains
`PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY`.

The exact future authorization behavior text is:

```text
授权 Q1 full-node6 capacity-preflight diagnostic successor-v5 distinct-outer actual；确认 successor-v4 known terminal failure、terminal no-resume，且 Python Odd measurement 与 Rust Odd measurement 各一次 conservatively count，Sink/trusted host replay 0；另行授权 exact v5 capsule-only additional fresh remeasurement；禁止恢复 successor-v4 或复用 successor-v4 work product
```
