# Hegel Machine Phase-3A-Q1 Full Node-6 Capacity-Preflight Successor v5

## 1. Position and claim

This exact source-closed contract defines distinct outer successor-v5 after
successor-v4 consumed its authorization and ended in a known terminal failure.
It retains action id `full-node6-capacity-preflight-v1`; it does not resume v4
or reuse any v4 work product.

The only permitted successful claim remains:

```text
PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
```

Q1 remains `NOT_RUN`, `0/20`, mask zero, with no receipt and eight null output
slots. Target truth, split and role inputs remain inaccessible. Network, Q1
start, adoption, promotion, production, a current-pointer write and `ACTIVE`
remain forbidden.

## 2. Exact successor-v4 terminal boundary

The Q0.5b attempt-2 predecessor remains terminally qualified at `20/20`; Q1
is not run. Successor-v5 binds successor-v4 as follows:

- source commit/tree/closure:
  `5626e1d025e604fc4df5824f40a337a4e756c3dc`,
  `d5527a765dd82677417525d8250da761affc9ad1`, and
  `sha256:703acfe82dbb098c1be8324fa458bd6e6c063c30382d21f0f4a37def06f0eb6b`;
- capsule:
  `sha256:73a07996d7acb21ae3c6d8b35a096850d7fe5f854269f2b13da67da6c595b6a0`;
- v4 contract/authorization:
  `sha256:1a93d654963bcf05369c2e5ea3c63af8e713151893c32ea8bbdc86e9d00ca222`
  and
  `sha256:6f1018a9ed53d9f20a051bf12b506d8a2c65134332199e16e6e3d3830853360a`;
- consumed record/file:
  `sha256:fc719de068b9fa8db653359b9048be80b0f4e1140a2858ea32d1ce2a92afbb8e`
  and
  `sha256:aa7fc01c6b87578fedd5d8f96e1ab00d8cdaec73f1a5dd5c0f3a10101930ac66`;
- terminal record/file:
  `sha256:ddf92134dc2b036084091baf7bc6063c8ce1bf41060f8ce2ae6d26a097d60bfe`
  and
  `sha256:926e4943846ccc9d7fbe6fb27eed9bd49bcc238ddd42620dc86008ad3b80a5e4`;
- outcome digest/file:
  `sha256:3e2d6780a75a8da0e91b4eca5cbdda6cdff1c2132b47e8ac6db2da217c824c6c`
  and
  `sha256:ed860e4a7a4128b14a10dc68e41909afeae2db43998318065359fe9558328795`;
- supervisor stdout/stderr files:
  `sha256:7a4c092db7a13d2b98a51d2338f5ff924a8987a833c33db27e5257eb2c2ec12f`
  and
  `sha256:4296c332ff1ba38a0f54754715c9cdaba664fae7192a2ff1062be56ca1ce3fba`;
- 31-event trajectory/terminal files:
  `sha256:aedeaa0d089f57219609087fc01990dce90803e26cb9ea2a5e879e4c0a7c70b4`
  and
  `sha256:c451a18be5542ebed13ebd51f576567b55135510d052fc8decbed08260f86d1f`.

Exactly one campaign execution, one authorization consumption and one
supervisor Popen occurred. The supervisor returned one, replay did not pass,
and no artifact was published. The terminal error was
`FULL_NODE6_ENDPOINT_SEQUENCE_FAILED` with detail
`an ordered endpoint did not exit cleanly`.

Python completed before Rust started; both returned one. Python stdout is bound
by
`sha256:f10025c57b41a12268743d0c13cea17f7fa0660ff22d7586bf35315ec746dd3b`
and reports `FAIL_Q1_PREFLIGHT_BEHAVIOR`, detail
`one AST has two behaviors`. Rust stdout is bound by
`sha256:4051a7222ad27e1d4aace73d53831fe63ed28f4b38b2d57d9e1dac9ef6347585`
and reports the same code, detail `one canonical AST has two behaviors`. Both
stderr files are empty with SHA-256
`sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Each Odd measurement conservatively counts once; Sink/trusted-host replay did
not start and counts zero.

The Python, Rust and aggregate resource receipt file hashes are respectively
`sha256:4b88c0c37d3aec1cf7c653ada1a02b49abfd4ea90e577576b2817fe41f7b994d`,
`sha256:f6db20b7920ce26e409fb666a6b3deaa2a109e18462c6faf162eeae59c296002`
and
`sha256:dc455f467008798f20c02b1223da840854dddd4024a941793c121dcb6dde0a2a`.
They prove one Python row, then one Rust row, maximum one live endpoint
container and no measurement-interval overlap. The scratch cleanup receipt is
bound by
`sha256:edfa3752d74c47fa079ca020096ad83fcb32d8b66542e27f3c940c0503cefc99`;
it has no failure rows and proves both scratch roots absent. The outer
post-consumption cleanup receipt is fully `PASS` for the supervisor process
group, production and control containers, probe root and production scratch.

The bounded counterexample is `difference(x,x)`: bottom propagation conflicts
with canonical zero. This localizes the known failure; it is not reusable v4
measurement work product.

## 3. Conservative successor-v5 boundary

Both v4 Odd measurements count. Any v5 measurement is explicitly additional,
limited to one exact future v5 capsule, and is not a resume or continuation of
v4. No v4 snapshot, binary, endpoint output, resource receipt or other work
product may seed v5 execution. The v4 paths above are read-only terminal
evidence, never cleanup or resume targets.

This implemented contract does not itself authorize or execute successor-v5.
Only a future user's exact authorization text may authorize one fresh logical
campaign:

```text
授权 Q1 full-node6 capacity-preflight diagnostic successor-v5 distinct-outer actual；确认 successor-v4 known terminal failure、terminal no-resume，且 Python Odd measurement 与 Rust Odd measurement 各一次 conservatively count，Sink/trusted host replay 0；另行授权 exact v5 capsule-only additional fresh remeasurement；禁止恢复 successor-v4 或复用 successor-v4 work product
```

That authorization is bounded to at most one supervisor Popen and one new
quotient measurement for each endpoint/signature pair. An exact
pre-consumption `PENDING` may be replayed only for the same v5 capsule digest;
it is not a campaign, consumes no authorization, invokes no supervisor and
starts no production endpoint or quotient engine. After consumption, any
failure, unknown measurement outcome or cleanup absence that cannot be proved
is terminal no-resume.

## 4. Fresh roots and admission

Successor-v5 uses only fresh `q1-full-node6-capacity-v5-*` lock, ledger,
evidence, trajectory, run and probe roots and
`artifacts/phase3_q1_full_node6_capacity_preflight_v5.json`. Required source
parent is `5626e1d025e604fc4df5824f40a337a4e756c3dc`. The additive source registry
includes `Hegel_Machine_Current_State_Reconciliation_v6.md`; the ancestor-based
modified registry contains exactly the capacity core, snapshot core and their
two focused tests.

All gates precede authorization consumption and supervisor Popen. The live
isolation gauntlet uses `/bin/true` only and records zero quotient-engine
invocations, signature selections, production endpoint starts and supervisor
Popen. The gauntlet is implemented. Successor-v5 has not yet been executed, preflighted or authorized.

The existing post-consumption cleanup mechanism remains mandatory on every
child-return path. Run evidence is retained. Python must exit and persist its
evidence before Rust starts; Rust must exit before trusted-host replay starts.

## 5. Terminal routing

Only `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY` can support a later byte-exact
formal-resource amendment. A guard, disagreement, isolation or replay failure,
unknown outcome, or unproved cleanup is terminal and cannot resume. Formal Q1
execution remains a separate later `phase3-q1-start` action after a saturated
diagnostic and its mechanical amendment. Successor-v5 provides no Q1 start.
