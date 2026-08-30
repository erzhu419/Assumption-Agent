# Hegel Machine Current-State Reconciliation v4

## 1. Status and precedence

This is the repository-level current dashboard after full-node6 successor-v2
reached terminal failure with an unknown measurement outcome. It supersedes v3
only as narration. V1--v3 remain immutable historical snapshots, and the bound
ledgers, outcome and trajectory remain authoritative for their narrow claims.

This document is not execution authorization, retry authorization, a gate
receipt, adoption, promotion, production, a current-pointer write, an `ACTIVE`
transition or a scientific result.

## 2. Reconciled state

| Track | Current state | Exact claim boundary |
|---|---|---|
| Q0.5b attempt 2 | **TERMINAL SUCCESS; 20/20; Q1 not run** | Unchanged terminal predecessor; no retry or resume. |
| Full-node6 transport v1 | **TERMINAL FAILURE; no endpoint start** | Historical v1 transport remains consumed and no-resume. |
| Full-node6 successor v2 | **CONSUMED; TERMINAL FAILURE; measurement outcome `UNKNOWN`** | One supervisor Popen; Python `STARTED` once, Rust and trusted host zero; no complete quotient result and no reusable v2 work product. |
| Full-node6 successor v3 | **IMPLEMENTED; not executed; awaiting future exact user authorization** | Distinct outer campaign and authorization identity, fresh v3 roots, frozen technical action v1. |
| Formal Q1 | **`NOT_RUN`; 0/20** | Count and mask zero, receipt null, eight formal output slots null. |

The current ordering is:

```text
Q0.5b attempt 2 = terminal success / 20/20 / Q1 NOT_RUN
  -> full-node6 v1 = terminal transport failure / no endpoint start
  -> full-node6 v2 = consumed / Python STARTED once
       / Rust and trusted host not started / outcome UNKNOWN / no-resume
  -> successor-v3 = distinct outer contract / fresh roots / not authorized
  -> Q1 = 0/20 / NOT_RUN
```

## 3. Exact successor-v2 evidence

Successor-v2 is bound to source
`1321eb3be3c28f823fba04b947bb96ee16d35f6c` and capsule
`sha256:8eec9c3afaaea443553cd886c84879f465e0b15f406c7f88a6bfef512867cdde`.
Its consumed file/record are respectively
`d43522bc1df643dc1e56cf0a30826b44e5f758e14dfeab9d58cdf07d59e724a2`
and `sha256:4643ee4c0740f5f374eb32d6dfaf9c24e0338d47979666a246c1be920e7dc6e4`.
Its terminal file/record are respectively
`f85afc4790dd25da050ba91f36044429030cbae5aaab5f4275ca96603fa661bd`
and `sha256:fabe961525bfc101907b099ca11a8c77b404e2e27253f8f6a615af68dc7b5e81`.
The outcome digest is
`sha256:b5bfd879546b15bfe9a19646ab87e8fd5a886dcd4ed99122d4d69644dc43f7cd`.

The 27-event trajectory file SHA-256 is
`bd2959d92d39c4c983c6797f3e0dfdf1f1e23be256a0d61a6cfaf967aafe40cd`;
the terminal-trajectory file SHA-256 is
`d5f9a867dbc65fb673e41195498334f96551bc78ecd4a2665b413b4ad089fbf8`.
Terminal error is `FAIL_FULL_NODE6_SUPERVISOR_RESOURCE` with detail
`PYTHON_ENDPOINT isolation differs`.

The Python actor becoming ready enough to emit `PYTHON_ENDPOINT STARTED` is not
a successful measurement and cannot be treated as a recovery checkpoint. Rust
and trusted-host replay never started. The measurement outcome is therefore
conservatively `UNKNOWN`; v2 is permanently no-resume.

## 4. Successor-v3 boundary

The prior unknown conservatively counts as a measurement. A future v3 run is
an explicitly additional remeasurement, scoped to one exact v3 capsule only,
not a v2 resume or continuation. V2 work-product reuse is forbidden.

Fresh v3 lock, ledger, evidence, trajectory, run, probe and artifact roots are
the only writable campaign roots. V1 and v2 roots are read-only evidence.
Pre-consumption admission reserves a `/bin/true` runtime-isolation live gauntlet
after CPU admission and before resource reservation. Its receipt must show
zero quotient-engine invocations, signature selections, production endpoint
starts and supervisor Popen. For an actual run, the exact receipt is bound into
the consumed ledger and fsynced before supervisor Popen; `PENDING` creates no
ledger. The gauntlet is implemented but has not yet been executed, preflighted
or authorized under successor-v3.

Only a future user's exact authorization may permit one distinct fresh
successor-v3 logical campaign. Its exact behavior text is:

```text
授权 Q1 full-node6 capacity-preflight diagnostic successor-v3 distinct-outer actual；确认 successor-v2 measurement outcome UNKNOWN、terminal no-resume，且 prior unknown conservatively counts as measurement；另行授权 exact v3 capsule-only additional fresh remeasurement；禁止复用 successor-v2 work product
```

Same-capsule pre-consumption `PENDING` replay
does not execute a campaign, consume authorization or call the supervisor.
Post-consumption failure or unknown outcome remains terminal no-resume.

Success remains limited to `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY`. Q1 remains
`NOT_RUN`, `0/20`, with null formal roots. Network, target/split/role access,
adoption, promotion, production, `ACTIVE` and current-pointer writes remain
forbidden.
