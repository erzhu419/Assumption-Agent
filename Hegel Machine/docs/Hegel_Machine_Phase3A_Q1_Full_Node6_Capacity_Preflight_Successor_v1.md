# Hegel Machine Phase-3A-Q1 Full Node-6 Capacity-Preflight Successor v3

## 1. Position and claim

This additive contract defines a distinct outer successor-v3 after the
successor-v2 campaign consumed its authorization and terminated with an unknown
measurement outcome. It retains the frozen scientific/technical action
`full-node6-capacity-preflight-v1`; it does not rewrite or continue v2.

The only permitted successful claim remains:

```text
PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
```

Q1 remains `NOT_RUN`, `0/20`, mask zero, with no receipt and eight null output
slots. Target truth, split and role inputs remain inaccessible. This contract
does not authorize adoption, promotion, production, a current-pointer write or
an `ACTIVE` transition.

## 2. Bound predecessor and terminal successor-v2 evidence

The Q0.5b attempt-2 predecessor remains terminally qualified at `20/20`, while
Q1 is not run. Successor-v3 additionally binds the complete terminal v2 outer
campaign:

- source commit `1321eb3be3c28f823fba04b947bb96ee16d35f6c`;
- capsule `sha256:8eec9c3afaaea443553cd886c84879f465e0b15f406c7f88a6bfef512867cdde`;
- consumed file SHA-256
  `d43522bc1df643dc1e56cf0a30826b44e5f758e14dfeab9d58cdf07d59e724a2`
  and record
  `sha256:4643ee4c0740f5f374eb32d6dfaf9c24e0338d47979666a246c1be920e7dc6e4`;
- terminal file SHA-256
  `f85afc4790dd25da050ba91f36044429030cbae5aaab5f4275ca96603fa661bd`
  and record
  `sha256:fabe961525bfc101907b099ca11a8c77b404e2e27253f8f6a615af68dc7b5e81`;
- outcome
  `sha256:b5bfd879546b15bfe9a19646ab87e8fd5a886dcd4ed99122d4d69644dc43f7cd`;
- trajectory file SHA-256
  `bd2959d92d39c4c983c6797f3e0dfdf1f1e23be256a0d61a6cfaf967aafe40cd`
  and terminal-trajectory file SHA-256
  `d5f9a867dbc65fb673e41195498334f96551bc78ecd4a2665b413b4ad089fbf8`;
- terminal error `FAIL_FULL_NODE6_SUPERVISOR_RESOURCE`, detail
  `PYTHON_ENDPOINT isolation differs`.

The trajectory contains exactly 27 events. It records one
`PYTHON_ENDPOINT STARTED`, no `RUST_ENDPOINT STARTED`, and no trusted-host replay
start. Python actor readiness or a `STARTED` event is not a quotient result,
successful measurement, reusable work product or recovery point. Because the
Python endpoint failed before a complete endpoint result and replay chain, the
measurement outcome is `UNKNOWN`. It is terminal and no-resume.

## 3. Conservative additional-measurement boundary

The prior unknown conservatively counts as a measurement. Any v3 measurement
would therefore be explicitly additional, scoped only to one exact future v3
capsule, and not a resume or continuation of v2. No v2 work product may be
reused. The v2 capsule, ledger, evidence, trajectory, run, probe and artifact
roots are read-only evidence and are never cleanup, absence or resume targets.

This document and the implemented contract do not themselves authorize v3
actual execution. Only a future user's exact authorization of the distinct
outer successor-v3 actual campaign may authorize one fresh logical campaign.
The exact authorization behavior text is:

```text
授权 Q1 full-node6 capacity-preflight diagnostic successor-v3 distinct-outer actual；确认 successor-v2 measurement outcome UNKNOWN、terminal no-resume，且 prior unknown conservatively counts as measurement；另行授权 exact v3 capsule-only additional fresh remeasurement；禁止复用 successor-v2 work product
```

That future authorization remains bounded to at most one supervisor Popen and
at most one quotient measurement for each endpoint/signature pair.

An exact pre-consumption `PENDING` result may be replayed only for the same v3
capsule digest. It is not a campaign execution, consumes no authorization and
invokes no supervisor. After consumption, any failure or unknown measurement
outcome is terminal no-resume.

## 4. Fresh successor-v3 roots and admission order

The contract uses only fresh `q1-full-node6-capacity-v3-*` lock, ledger,
evidence, trajectory, run and probe roots, plus
`artifacts/phase3_q1_full_node6_capacity_preflight_v3.json`. Required source
parent is `1321eb3be3c28f823fba04b947bb96ee16d35f6c`; the additive registry includes
`Hegel_Machine_Current_State_Reconciliation_v4.md`.

All static and live admission gates precede authorization consumption and
supervisor Popen. The order reserves `RUNTIME_ISOLATION_LIVE_GAUNTLET` after
CPU topology/affinity and before memory/storage reservation. Its required
probe is `/bin/true`; the receipt must state zero quotient-engine invocations,
zero signature selections, zero production endpoint starts and zero supervisor
Popen. During an actual run, that exact receipt is the first durable semantic
evidence in the consumed ledger and is fsynced before supervisor Popen; a
pre-consumption `PENDING` writes no ledger. The gauntlet is implemented, but it
has not yet been executed, preflighted or authorized under successor-v3.

The technical execution order remains strict: Python must exit completely
before Rust starts, and Rust must exit completely before trusted-host replay
starts. Endpoints remain target-blind and network-disabled.

## 5. Terminal routing

Only `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY` can support a later byte-exact
formal-resource amendment. A capacity guard is inconclusive; disagreement,
isolation failure, replay failure or an unknown outcome is terminal failure.
None starts Q1 or establishes a scientific result.

Formal Q1 execution remains a separate later `phase3-q1-start` action after a
saturated diagnostic and its mechanical amendment. Successor-v3 provides no
Q1 start action.
