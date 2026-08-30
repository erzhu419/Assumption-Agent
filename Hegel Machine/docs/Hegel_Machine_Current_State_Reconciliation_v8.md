# Hegel Machine Current-State Reconciliation v8

## Status

This dashboard follows the successful successor-v6 full-node6 capacity
diagnostic and supersedes v7 only as current narration. Earlier reconciliations,
the v6 source, and all consumed evidence remain immutable history. This document
is not execution authorization.

The current state is:

| Boundary | Current result |
|---|---|
| Q0.5b attempt 2 | terminal success, 20/20, no retry or resume |
| Full-node6 successor-v6 | `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY`; host replay PASS |
| Formal-resource amendment | mechanically derived candidate; not owner-authorized |
| Formal Q1 | `NOT_RUN`, 0/20, mask zero, null genesis and receipt, eight null outputs |

Network, Q1 start, target/split/role access, production, adoption, promotion,
the current pointer, and `ACTIVE` remain forbidden.

## Successful successor-v6 terminal evidence

Successor-v6 consumed capsule
`sha256:a22cc6660e01c56b5670cca770d29d06c40bd06851d18868cbff07b61e918a29`
at source commit/tree/closure
`86cedd4a43f6aa769a52ec9b246cfc1fa678a2a0`,
`8bc7767bad36745d9c69e8ea8b5b25e9e0e54eec`, and
`sha256:66eeb948463f1934d324db0d45ead1206297642f5460d3de6bf31b5ce8974fe3`.
Its contract and authorization digests are
`sha256:1addc9ac8fe99ef4bed62146d41668911ff04a002b5a421389e1f4eb041c67ac`
and
`sha256:6da8ac001e4213960daf5733b64e081834589063a715f7d63cef5d33a7a1ced1`.

Exactly one campaign, one authorization consumption, and one supervisor Popen
occurred. Python completed and persisted its result before Rust started; Rust
completed before trusted-host replay started. Both signatures saturated in
both endpoints, replay passed, the supervisor returned zero, and the terminal
status is `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY`.

The 11,869-byte artifact
`artifacts/phase3_q1_full_node6_capacity_preflight_v6.json` has SHA-256
`73c319b54c53af75c3dbe01d64cd74aeb903c7d2edf1491cea7a8b7eb38194f4`.
The dual semantic agreement value is
`ce8f77867675413d58b5e446f87a16c661e2cfcf27c27aef545e23ae63352671`.
Post-consumption cleanup passed; endpoint and host scratch roots are proved
absent. The campaign is terminal and cannot be resumed.

## Mechanically derived candidate envelope

The exact candidate count guards are:

| Signature | raw | class | frontier total/per class | bank total/per class | queue |
|---:|---:|---:|---:|---:|---:|
| 1 | 32,768 | 1,024 | 4,096 / 32 | 8,192 / 4,096 | 16,384 |
| 2 | 32,768 | 256 | 1,024 / 256 | 8,192 / 4,096 | 16,384 |

The formal saturation-round guard is `7`. Endpoint resource guards are
2 GiB memory, 16 MiB output, 4 MiB scratch, and 3,600 seconds wall time. Host
replay guards are 64 MiB memory, 2 MiB output, 1 byte scratch, and 3,600 seconds
wall time. Every value is below the preregistered hard ceiling.

These values are recorded in
`config/phase3_q1_formal_resource_amendment_v1.json`. They are mechanical
candidate values, not a Q1 state transition.

## Exact next boundary

The only next normative boundary is an owner-approved, byte-exact formal
resource/genesis amendment. The repository currently lacks an approved Q1
run-genesis tag/schema/domain/field order and an approved Gates 1–14 evidence or
receipt wire. The candidate documents do not invent them.

A future exact capsule must bind the approved bytes, the successful v6 evidence,
the mechanical envelope, final implementation identities, and the continuing
eight-null-output condition. Only a new exact owner authorization may publish
the genesis and establish:

```text
Q1 = NOT_RUN / 14/20 / mask 0x003fff
run genesis = non-null
completion receipt = null
eight formal output slots = null
```

Non-core pre-consumption problems may use same-capsule `PENDING` without
publication, authorization consumption, endpoint invocation, or another
measurement. Q1 execution remains a later, separately authorized
`phase3-q1-start` action. Q2 remains downstream of a sealed Q1 `COMPLETE`
result.

## Continuing prohibitions

No successor action may retry or resume v6, remeasure the v6 technical action,
change the frozen archive/projection mechanics, reinterpret diagnostic roots as
formal outputs, open target/split/role inputs, enter Path B or D, use network,
or perform production, adoption, promotion, current-pointer, or `ACTIVE`
transitions.
