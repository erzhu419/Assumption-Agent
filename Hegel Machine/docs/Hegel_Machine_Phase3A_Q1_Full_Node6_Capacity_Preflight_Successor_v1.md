# Hegel Machine Phase-3A-Q1 Full Node-6 Capacity-Preflight Successor v6

## 1. Position and claim

This exact source-closed contract defines distinct outer successor-v6 after
successor-v5 consumed its authorization and ended in a known terminal failure.
It retains action id `full-node6-capacity-preflight-v1`; it does not resume v5
or reuse any v5 work product.

The only permitted successful claim remains:

```text
PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
```

Q1 remains `NOT_RUN`, `0/20`, mask zero, with no receipt and eight null output
slots. Target truth, split and role inputs remain inaccessible. Network, Q1
start, adoption, promotion, production, a current-pointer write and `ACTIVE`
remain forbidden.

## 2. Exact successor-v5 terminal boundary

Successor-v6 binds the successor-v5 capsule
`sha256:6819416eed0a41573f21b4280a6a39e97481b87f8dd1c9e0054529e478bc7d9d`
to source commit/tree/closure
`3086adf389b8d9fa1a93c0c36fd300ed8bca19db`,
`9cbdbee241e709296d40f9cf7b66fe26d038c67e`, and
`sha256:bd87e0af65e864fabd07af58e4ba552def3325d0a4493a1cade16f7a3a7c252d`.
The v5 contract and authorization digests are
`sha256:23e8631c3633c24c230a32c92bb02b067553f35c0afef1ab19b4448f9a435008`
and
`sha256:d55d5cbde77f58385ed6decbf5756f2042a4cf3819281819ded853554356da8d`.

The consumed record/file are
`sha256:98945a148bec0a3a356f997891d625990be40f16539990e4abac77daab395f4d` /
`sha256:43ee028f57b93b9b166005bfe640b4aa46090ddb5e0a1240a7162b2b6e1e7d4d`;
the terminal record/file are
`sha256:d2cb047773d716f310991d9ead0580058ffd778c7ce3183bf2610867e8de2aa6` /
`sha256:2ee9dbe129bfa2e1f0ddeff1bfb95e97d81b3c093c64a3d7280aa563f7d9f5b9`;
and the outcome digest/file are
`sha256:108f4dc6971499b9b736f899ffeb1d8b3003fd9b9d56c85c185da60f5206b43d` /
`sha256:3f08fa61d3901ee970f8b901300c7bc870d3863004bb2747d24ea92e947168d0`.
Exactly one campaign, one authorization consumption, and one supervisor Popen
occurred. The supervisor returned one, replay did not pass, no artifact was
published, and the campaign ended
`FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME` with
`FULL_NODE6_HOST_REPLAY_FAILED: trusted replay did not exit cleanly`.

The supervisor stdout/stderr SHA-256 values are
`sha256:474d16bf26e2576d76db979bbb926916467b3e2c8e64364b74af7274d7a85ebe`
and
`sha256:8d898ef4222871214aa4141702e612aae7656ad7272c5eaa4c66d863a662f07c`.
The complete 34-event trajectory and terminal receipt are bound by
`sha256:fcf6940289491851e6675d8854a7aa99938e4e0a501f2feef83dfddca9251e3f`
and
`sha256:30daf6dd4efd56bff9cdff560d04a85629acb1bb95c34d882bf031f9200bdeb9`.

Python exited rc=0 and persisted a saturated diagnostic before Rust started.
Its stdout/stderr hashes are
`sha256:f200c5a466954c5f12d8182c28a4187b9031cddca45be42adcb839ed63451d9b` /
`sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
with actual peak/terminal-live scratch `40452166/0`. Rust then exited rc=0 with
the same saturated status; its stdout/stderr hashes are
`sha256:b667b13b281b832dc78c2e13411a7f5c615b634ffb0fc4f88792bd0ca6a68b31` /
the same empty-stderr hash, with actual peak/terminal-live scratch `0/0`.
Only after Rust exited did trusted-host replay start. It exited rc=1; its
stdout/stderr hashes are
`sha256:11c1ebed96bd610584294cd175978c4a44b0c710d0ec08d0801b64898ef93573` /
`sha256:f7111d670e7d0d1d0c1421a10fabceb9a9308bb0d5c7d567190455c4829dc29a`
and it reported `FAIL_FULL_NODE6_HOST_RESOURCE` with detail
`RUST_ENDPOINT actual scratch accounting differs`.

The Python, Rust, aggregate and host resource receipts are respectively bound
by
`sha256:9d4facca1c85f2cc5a27bc2225527bd4b29cddf26821bc648252b1171686c6e2`,
`sha256:b82085715b23ce5c64bfc3d344a319018fb773ec9f62584a069dedfd7fbbabc2`,
`sha256:259730b31c525b6d58d0ea8390a87a9736704e4045932fe496e8f5912d3cf263`,
and
`sha256:705350f854c526fa7f2e2e73906e674b9bf79ea51d7d29b0ebf8e17913ccc3c5`.
They record strict Python-then-Rust-then-host ordering. Endpoint and host scratch
cleanup receipts are bound by
`sha256:505e39652b74c45353ee48c665d16dba35b552a8ae72cfa2bba703e34001882f`
and
`sha256:5830397ea43e8d65897e30d0c0075b759cfd9f33995f5c7f9a46f04532b98792`;
both have no failure rows and prove their roots absent. The complete outer
post-consumption cleanup receipt is `PASS` for the supervisor process group,
production and control containers, probe root, and production scratch.

All endpoint and host outputs retain Q1 `NOT_RUN`, count zero, mask zero, and
eight null output slots. Python Odd/Sink and Rust Odd/Sink each conservatively
count once; trusted-host replay conservatively counts once. Successor-v5 is
terminal no-resume, and none of its snapshots, binaries, outputs, receipts, or
other work product may seed successor-v6.

## 3. Successor-v6 repair delta

Successor-v6 changes the implementation only where v5 evidence exposed a real
failure or where an independent replay check was incomplete:

- The Rust spill sorter now accounts for and removes its non-empty final
  in-memory buffer through the same tracked scratch path as earlier buffers.
  A bounded KAT proves a positive scratch peak, identical root/count/distinct
  output, zero terminal live bytes, and scratch-path absence.
- Rust candidate visitation now matches the frozen Python schedule: global
  program order for unary and non-commutative applications, node-hash order for
  commutative applications, node-CBOR order for Boolean applications, and
  pair-major operator interleaving. Bounded node-3/node-4 and explicit event
  sequence KATs cover this alignment.
- Trusted-host replay validates the exact frozen structural limits and every
  persisted barrier transition. It validates each endpoint's mutation history
  locally but excludes that diagnostic-only intermediate history from
  cross-endpoint identity; visible-frontier high-water remains exact. Legal
  scheduler differences are preserved as per-signature work-queue and round
  high-water vectors and combined by exact `MAX`, with both values bounded by
  the frozen limits.

These changes do not reinterpret v5. They apply only to fresh v6 source and
roots; v5 remains the terminal result recorded above.

## 4. Conservative successor-v6 boundary

Any v6 measurement is explicitly additional, limited to one exact future v6
capsule, and is not a resume or continuation of v5. This implemented contract
does not itself execute or preflight successor-v6. Only a future user's exact
authorization text may authorize one fresh logical campaign:

```text
授权 Q1 full-node6 capacity-preflight diagnostic successor-v6 distinct-outer actual；确认 successor-v5 known terminal failure、terminal no-resume，且 Python Odd/Sink 与 Rust Odd/Sink measurement 各一次 conservatively count，trusted host replay 一次 conservatively count；另行授权 exact v6 capsule-only additional fresh remeasurement；禁止恢复 successor-v5 或复用 successor-v5 work product
```

That authorization is bounded to at most one supervisor Popen and one new
quotient measurement for each endpoint/signature pair. An exact
pre-consumption `PENDING` may be replayed only for the same v6 capsule digest;
it is not a campaign, consumes no authorization, invokes no supervisor, and
starts no production endpoint or quotient engine. After consumption, any
failure, unknown measurement outcome, or cleanup absence that cannot be proved
is terminal no-resume.

## 5. Fresh roots and admission

Successor-v6 uses only fresh `q1-full-node6-capacity-v6-*` lock, ledger,
evidence, trajectory, run and probe roots and
`artifacts/phase3_q1_full_node6_capacity_preflight_v6.json`. Required source
parent is `3086adf389b8d9fa1a93c0c36fd300ed8bca19db`. The additive source registry
includes `Hegel_Machine_Current_State_Reconciliation_v7.md`; the ancestor-based
modified registry remains exactly the capacity core, snapshot core, and their
two focused tests.

All gates precede authorization consumption and supervisor Popen. The gate
re-reads and validates the v5 capsule identity, source identity, contract and
authorization bindings, ledgers, outcome, all 34 trajectory events, every
stdout/stderr and resource/scratch receipt, complete post-consumption cleanup,
and the Q1 null nonclaims. The existing post-consumption cleanup mechanism
remains mandatory on every child-return path and preserves run evidence.

The runtime-isolation gauntlet uses `/bin/true` only and records zero quotient
engine invocations, signature selections, production endpoint starts and
supervisor Popen. The gauntlet is implemented. Successor-v6 has not yet been executed, preflighted or authorized.

## 6. Terminal routing

Only `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY` can support a later byte-exact
formal-resource amendment. A guard, disagreement, isolation or replay failure,
unknown outcome, or unproved cleanup is terminal and cannot resume. Formal Q1
execution remains a separate later `phase3-q1-start` action after a saturated
diagnostic and its mechanical amendment. Successor-v6 provides no Q1 start.
