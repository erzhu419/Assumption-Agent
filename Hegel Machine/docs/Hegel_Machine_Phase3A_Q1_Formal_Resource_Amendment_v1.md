# Hegel Machine Phase-3A-Q1 Formal Resource Amendment v1 — Candidate

## 1. Status and position

**Status:** `CANDIDATE_MECHANICALLY_DERIVED_NOT_OWNER_AUTHORIZED`

This document records the only successor permitted after the successful
full-node6 capacity diagnostic. It freezes the successful v6 predecessor and
the resource values mechanically determined by the preregistered formulas. It
does not approve a Q1 run genesis, qualify any Q1 gate, start Q1, or define new
formal wire bytes.

The state remains:

```text
q1_state        = NOT_RUN
q1_gate_count   = 0
q1_gate_mask    = 0x000000
q1_run_genesis  = null
q1_receipt      = null
q1_outputs      = [null, null, null, null, null, null, null, null]
```

The successful diagnostic is not itself Q1 evidence. The preregistration says
that `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY` may only be followed by a byte-exact
formal-resource amendment which binds the diagnostic, applies the frozen
formulas without discretion, freezes a separately approved run genesis, and
qualifies Gates 1–14 without starting Q1. This candidate completes only the
first two of those tasks.

## 2. Exact v6 predecessor

The predecessor is the single consumed successor-v6 campaign:

```text
capsule          sha256:a22cc6660e01c56b5670cca770d29d06c40bd06851d18868cbff07b61e918a29
source commit    86cedd4a43f6aa769a52ec9b246cfc1fa678a2a0
source tree      8bc7767bad36745d9c69e8ea8b5b25e9e0e54eec
source closure   sha256:66eeb948463f1934d324db0d45ead1206297642f5460d3de6bf31b5ce8974fe3
contract         sha256:1addc9ac8fe99ef4bed62146d41668911ff04a002b5a421389e1f4eb041c67ac
authorization    sha256:6da8ac001e4213960daf5733b64e081834589063a715f7d63cef5d33a7a1ced1
```

It executed one logical campaign, consumed one authorization, and invoked one
supervisor Popen. The supervisor returned zero, trusted-host replay passed, and
the terminal result was:

```text
PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
```

The committed artifact is
`artifacts/phase3_q1_full_node6_capacity_preflight_v6.json`, 11,869 bytes, with
SHA-256
`73c319b54c53af75c3dbe01d64cd74aeb903c7d2edf1491cea7a8b7eb38194f4`.
The dual semantic agreement value is
`ce8f77867675413d58b5e446f87a16c661e2cfcf27c27aef545e23ae63352671`.

The durable outer bindings are:

- consumed ledger record/file:
  `sha256:0928ba91ff57886c93c5d85003b1518c730868db53c9166ec63290ff459edb27` /
  `sha256:665c97e418f7cafcc0ce6e25812e147d763414d096de53f2e8d0a2d6f01d391f`;
- terminal ledger record/file:
  `sha256:6385c52fb63b9745ffa414fc90eef3362b42f2cb8601c64ee79c508011e20fa9` /
  `sha256:d0c11ef939df13e70502d2f623a44fc42fefef0beaffeb3a0d1ba77e36521624`;
- outcome object/file:
  `sha256:4a29e82d50cb29bfa9b61c980568885296511820b7ced1b91571dd06a13985ab` /
  `sha256:56e7e7cccaa95c3ffee4c8df6cec993404373fa4076babeae488c74dd49886de`;
- supervisor stdout/stderr:
  `sha256:5253f71c325694c9ad0d1c83bbfc39c67851955b69d827d1a3ed6ac402f65224` /
  `sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- 40-event trajectory/terminal:
  `sha256:8b942618f910305ff1c277d9b94bec996ef03510ebbf85763506f06468d93971` /
  `sha256:bcad1cb65ee3204c9b1037dd47e1aaea2ed701d6a4c418e0c8b3c16f088a8bbd`.

Endpoint stdout and trusted-host replay stdout are bound respectively by
`sha256:9c71a2ba7e2bc41e5cc0390c8f53eab0dbb000ea916ccb5be4a75cc56bf85442`,
`sha256:c5e641f18ab42982e92035b89131b81c0c61d73dcd139fad015bc75ef21cb779`,
and
`sha256:33cc056f1c9ddc3a12920f5651aa0d32a0d9e17bd81f804d2409ae143dc194b4`.
Post-consumption cleanup passed. Endpoint and host scratch cleanup both prove
terminal absence. Retry or resume is forbidden.

## 3. Mechanical count envelope

For every accepted-state high-water value `x`, the preregistered rule is:

```text
formal_count_guard = pow2ceil(max(1, 2 * x))
```

Python and Rust produced identical semantic high-water values. Work-queue
values use the preregistered per-signature maximum across legal endpoint
schedules.

| Signature | Quantity | Observed | Formal guard |
|---:|---|---:|---:|
| 1 | raw operator applications | 11,846 | 32,768 |
| 1 | behavior classes | 418 | 1,024 |
| 1 | visible frontier total | 1,098 | 4,096 |
| 1 | visible frontier per class | 14 | 32 |
| 1 | continuation bank total | 3,611 | 8,192 |
| 1 | continuation bank per class | 1,175 | 4,096 |
| 1 | work queue points | 8,150 | 16,384 |
| 2 | raw operator applications | 12,908 | 32,768 |
| 2 | behavior classes | 89 | 256 |
| 2 | visible frontier total | 372 | 1,024 |
| 2 | visible frontier per class | 124 | 256 |
| 2 | continuation bank total | 3,277 | 8,192 |
| 2 | continuation bank per class | 1,667 | 4,096 |
| 2 | work queue points | 7,774 | 16,384 |

The observed maximum saturation-round count is `5`; the frozen formula
`observed + 2` gives a formal round guard of `7`.

## 4. Mechanical endpoint and host envelope

The formal archive projection is identical across Python and Rust:

```text
projected archive payload             4,742,060 bytes
run metadata reservation              1,048,576 bytes
projected total endpoint output       5,790,636 bytes
projected peak endpoint scratch       1,646,592 bytes
```

Python measured peak RSS `112,295,936` bytes and wall time
`729,612,002,970` ns. Rust measured peak RSS `540,782,592` bytes and wall time
`7,702,460,263` ns. Applying the frozen maximum, power-of-two, and whole-hour
formulas gives:

| Endpoint guard | Formal value | Preregistered hard ceiling |
|---|---:|---:|
| memory | 2,147,483,648 bytes | 15,032,385,536 bytes |
| output | 16,777,216 bytes | 34,359,738,368 bytes |
| scratch | 4,194,304 bytes | 51,539,607,552 bytes |
| wall | 3,600 seconds | 172,800 seconds |

Trusted-host replay measured peak RSS `17,690,624` bytes and wall time
`1,469,117,302` ns. Its frozen projections are `1,048,576` output bytes and
zero scratch bytes. The mechanical host envelope is:

| Host guard | Formal value | Preregistered hard ceiling |
|---|---:|---:|
| memory | 67,108,864 bytes | 15,032,385,536 bytes |
| output | 2,097,152 bytes | 34,359,738,368 bytes |
| scratch | 1 byte | 51,539,607,552 bytes |
| wall | 3,600 seconds | 172,800 seconds |

Every derived value is within its preregistered hard ceiling. No manual budget
choice, favorable-endpoint selection, multiplier change, or post-result resource
redesign was used.

## 5. Candidate boundary and missing normative bytes

This candidate deliberately does not invent bytes that the current repository
never froze. In particular, the existing Q1 formal archive registry has no
approved Q1 run-genesis tag, schema, domain, or field order, and no approved
wire for a Gates 1–14 admission evidence object or receipt.

Those bytes require a new owner-approved normative amendment. Until that
approval is bound to a new exact capsule and the resulting evidence is
qualified, all fourteen Q1 gates remain absent. Q0.5b's 20/20 qualification
cannot be copied into the distinct Q1 gate registry, and the diagnostic
projection or its sidecars cannot be reinterpreted as formal Q1 outputs.

The future authoritative pre-start state, if and only if separately approved
and qualified, is:

```text
q1_state                 = NOT_RUN
q1_gate_count            = 14
q1_gate_mask             = 0x003fff
q1_run_genesis           = non-null approved exact object
q1_completion_receipt    = null
q1_outputs               = [null, null, null, null, null, null, null, null]
```

This text does not create that state.

## 6. Authorization and continuation

Engineering, synthetic tests, and side-effect-free replay may prepare an exact
candidate capsule. Publishing a non-null Q1 run genesis or any Gates 1–14
evidence requires a new exact owner authorization bound to that capsule. The
consumed v6 diagnostic authorization does not cover that publication.

Non-core path, permission, format, dependency, memory, or storage failures may
enter an exact pre-consumption `PENDING` state only under the same future
capsule. Such a PENDING replay may not publish genesis or gate evidence, consume
the future authorization, invoke an endpoint, or measure the quotient again.

Even after a valid `14/20 + NOT_RUN` pre-start state exists, formal execution
requires another separate explicit `phase3-q1-start` authorization. The
resource amendment authorization cannot be interpreted as Q1 start authority.

## 7. Forbidden actions

The following remain forbidden:

- any v6 retry, resume, or additional endpoint/signature measurement;
- changing the frozen wire, chunking, allocator, or merge schedule;
- inventing a run-genesis or fourteen-gate receipt wire without owner normative
  approval;
- treating diagnostic roots as formal Q1 outputs;
- Q1 start, target truth, split, or role access;
- Path B or Path D transition;
- network, production, adoption, promotion, current-pointer write, or `ACTIVE`.
