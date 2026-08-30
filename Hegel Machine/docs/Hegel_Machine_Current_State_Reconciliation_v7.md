# Hegel Machine Current-State Reconciliation v7

## Status

This dashboard follows the consumed successor-v5 campaign and supersedes v6
only as current narration. Earlier reconciliations and all bound evidence are
immutable history. This document is not execution authorization.

Q0.5b attempt 2 remains terminal success at 20/20. Q1 remains `NOT_RUN`, 0/20, mask zero, with null receipt and eight null output slots. Network, Q1 start,
target/split/role access, adoption, promotion, production, `ACTIVE`, and the
current pointer remain forbidden. The only permitted successful diagnostic
claim remains `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY`.

## Bound successor-v5 terminal evidence

Successor-v5 capsule
`sha256:6819416eed0a41573f21b4280a6a39e97481b87f8dd1c9e0054529e478bc7d9d`
at commit/tree/closure
`3086adf389b8d9fa1a93c0c36fd300ed8bca19db`,
`9cbdbee241e709296d40f9cf7b66fe26d038c67e`, and
`sha256:bd87e0af65e864fabd07af58e4ba552def3325d0a4493a1cade16f7a3a7c252d`
consumed exactly one authorization and launched exactly one supervisor Popen.
Its contract/authorization digests are
`sha256:23e8631c3633c24c230a32c92bb02b067553f35c0afef1ab19b4448f9a435008`
and
`sha256:d55d5cbde77f58385ed6decbf5756f2042a4cf3819281819ded853554356da8d`.

The durable bindings are:

- consumed record/file:
  `sha256:98945a148bec0a3a356f997891d625990be40f16539990e4abac77daab395f4d` /
  `sha256:43ee028f57b93b9b166005bfe640b4aa46090ddb5e0a1240a7162b2b6e1e7d4d`;
- terminal record/file:
  `sha256:d2cb047773d716f310991d9ead0580058ffd778c7ce3183bf2610867e8de2aa6` /
  `sha256:2ee9dbe129bfa2e1f0ddeff1bfb95e97d81b3c093c64a3d7280aa563f7d9f5b9`;
- outcome digest/file:
  `sha256:108f4dc6971499b9b736f899ffeb1d8b3003fd9b9d56c85c185da60f5206b43d` /
  `sha256:3f08fa61d3901ee970f8b901300c7bc870d3863004bb2747d24ea92e947168d0`;
- supervisor stdout/stderr:
  `sha256:474d16bf26e2576d76db979bbb926916467b3e2c8e64364b74af7274d7a85ebe` /
  `sha256:8d898ef4222871214aa4141702e612aae7656ad7272c5eaa4c66d863a662f07c`;
- 34-event trajectory/terminal:
  `sha256:fcf6940289491851e6675d8854a7aa99938e4e0a501f2feef83dfddca9251e3f` /
  `sha256:30daf6dd4efd56bff9cdff560d04a85629acb1bb95c34d882bf031f9200bdeb9`.

Python and Rust each started once, completed rc=0, and emitted
`PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY`. Their stdout hashes are
`sha256:f200c5a466954c5f12d8182c28a4187b9031cddca45be42adcb839ed63451d9b`
and
`sha256:b667b13b281b832dc78c2e13411a7f5c615b634ffb0fc4f88792bd0ca6a68b31`;
both stderr files are empty with SHA-256
`sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Python evidence persisted before Rust started. Rust exited before trusted-host
replay started. Trusted-host replay started once, returned rc=1, and its
stdout/stderr hashes are
`sha256:11c1ebed96bd610584294cd175978c4a44b0c710d0ec08d0801b64898ef93573` /
`sha256:f7111d670e7d0d1d0c1421a10fabceb9a9308bb0d5c7d567190455c4829dc29a`.
It reported `FAIL_FULL_NODE6_HOST_RESOURCE`:
`RUST_ENDPOINT actual scratch accounting differs`.

The Python/Rust/aggregate/host resource receipt hashes are
`sha256:9d4facca1c85f2cc5a27bc2225527bd4b29cddf26821bc648252b1171686c6e2`,
`sha256:b82085715b23ce5c64bfc3d344a319018fb773ec9f62584a069dedfd7fbbabc2`,
`sha256:259730b31c525b6d58d0ea8390a87a9736704e4045932fe496e8f5912d3cf263`,
and
`sha256:705350f854c526fa7f2e2e73906e674b9bf79ea51d7d29b0ebf8e17913ccc3c5`.
The endpoint/host scratch cleanup hashes are
`sha256:505e39652b74c45353ee48c665d16dba35b552a8ae72cfa2bba703e34001882f`
and
`sha256:5830397ea43e8d65897e30d0c0075b759cfd9f33995f5c7f9a46f04532b98792`.
Both prove terminal absence with no failure rows. Complete outer
post-consumption cleanup is `PASS` for process-group, production/control
containers, probe root, and production scratch.

The supervisor returned one, replay did not pass, no artifact was published,
and successor-v5 terminated
`FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME`. All outputs retain Q1
`NOT_RUN`, count zero, mask zero, and eight null output slots. Python Odd/Sink
and Rust Odd/Sink each conservatively count once; trusted-host replay
counts once. Successor-v5 is permanently no-resume and none of its work product
may seed another campaign.

## Distinct successor-v6

Successor-v6 is an implemented outer contract only. It uses fresh
`q1-full-node6-capacity-v6-*` lock, ledger, evidence, trajectory, run and probe
roots and
`artifacts/phase3_q1_full_node6_capacity_preflight_v6.json`. It is not a v5
retry. Any v6 remeasurement is explicitly additional and limited to one exact
future v6 capsule under new exact user authorization.

An exact pre-consumption `PENDING` replay for that same capsule does not count
as a campaign, consume authorization, invoke the supervisor, or start a
production endpoint or quotient engine. Any post-consumption failure, unknown
measurement outcome, or cleanup absence that cannot be proved is terminal
no-resume.

The exact future authorization behavior text is:

```text
授权 Q1 full-node6 capacity-preflight diagnostic successor-v6 distinct-outer actual；确认 successor-v5 known terminal failure、terminal no-resume，且 Python Odd/Sink 与 Rust Odd/Sink measurement 各一次 conservatively count，trusted host replay 一次 conservatively count；另行授权 exact v6 capsule-only additional fresh remeasurement；禁止恢复 successor-v5 或复用 successor-v5 work product
```
