# Hegel Machine Phase-3 shrink-5 sealed dual strict qualification result v1

Status: **PASS — NON-FORMAL DUAL STRICT QUALIFICATION; M3 NOT RUN**

This record publishes the commit-bound shrink-5 admission qualification. It
establishes exact Python/Rust agreement at the six-node recognizer boundary and
on the two frozen replay sets. It is not a closure run, formal-root
publication, target evaluation, outside-language certificate, or
ACTIVE-governance transition.

## 1. Immutable basis and evidence

```text
Source Commit S  320b0a3458901090cb738023a4398220fb1d9277
source subject   hegel: freeze shrink5 six-node admission
parent commit    1bbdae8f3131625621c0bc1cfdfe5d7da6035e13
source rows      71
source-set root  sha256:4a7ae37381f7ec77a362d0cb945f2ddaf0649353777b911e32f696e747ebfeaf
Git archive      3362e19a39940276c3628ddea5de5c8df93679750e55383a53395c444e14720e
```

The parent is the shrink-4 dual complete diagnostic evidence commit. Its
bound record admits exactly `reduce max_total_node_count from 7 to 6` and does
not permit a formal status promotion.

The host supervisor verified its own bytes against Source S. Both recognizers
and both capacity endpoints executed from the extracted Source-S archive. The
canonical qualification report is:

```text
artifacts/phase3_m3_runtime/phase3_shrink5_sealed_dual_strict_qualification_v1.json
file SHA-256      75761fc536d96d5d0bc91c5c0ba30dbc7c9ee21aac8d3f1dc5c96f6aca919b76
diagnostic hash   sha256:5ee04b21477fd9f09271272fd6ecbf876b885b7831b37a868343a93996a187db
```

The published file is the exact canonical one-line JSON emitted by the run.
The portable evidence validator decodes ASCII JSON with duplicate-key and
non-finite-number rejection, requires canonical bytes, invokes the frozen
Source-S exact-schema validator, and independently checks Git rows, object
types, commitments, authority guards, and forbidden future-closure fields.

## 2. Dual strict outcome

The Python and Rust implementations agreed on every sealed wire:

```text
status              SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS
claim level         NON_FORMAL_DUAL_STRICT_QUALIFICATION
sealed vectors      22 / 22 on each implementation
manifest root       sha256:156f7e20407437bb753b097a87932f469701d1de6d1d577b0fa1b7a98f47e52e
dual outcome root   sha256:8f82178c0f33d5295601d2e112b0b6e25ef18d73e5fc35d8d601024c1f0ddf94
```

Accepted outcomes bind both frozen structural maxima:

```text
maximum AST nodes          6
maximum top-level clauses  2
```

The exact endpoint schemas remain intentionally asymmetric: Python contributes
80 combined fields and Rust contributes 78. Only their frozen comparable
fields are normalized, and those fields agree type-strictly.

## 3. Survivor and removed-boundary replay

Both implementations replayed the complete frozen 175-program inherited
target-free survivor set:

```text
source / accepted / unique       175 / 175 / 175
parent identity matches          175
rejections                       0
accepted-set commitment          sha256:f5ab7f079ad943d65a74881eb59c7bb46385e1c437ca8ab036bb071dfa3874ac
```

They separately replayed all 2,160 source programs in the inherited shrink-4
AND2 capacity set. Each parent program was accepted under v1.4.0 with exactly
seven AST nodes, then rejected under v1.5.0 at both boundaries:

```text
parent-only source candidates    2,160
parent accepted / node count     2,160 / 7
child source rejections          2,160 REJECT_STRUCTURAL_LIMIT
child formal rejections          2,160 REJECT_STRUCTURAL_LIMIT
parent-only set commitment       sha256:7e0e8780149f03ce85723408f7e3eff2cd684e8938896125cf8e34be9ac70b5e
source rejection commitment      sha256:8617b56bdfa347f11f2c68b6a41f0992652f1e23e6d651017b17eb50169a9f39
formal rejection commitment      sha256:9a6b489ed90960008aebbecdbcf0bc5cf1595b7a8206d179bbe898540dabf617
```

These two replay sets prove recognizer identity preservation and the exact
six-node boundary. They do not enumerate the shrink-5 closure and do not imply
its cardinality.

## 4. Isolation and reproducibility boundary

The run used two digest-pinned local images, `--pull=never`,
`--network=none`, a read-only container root filesystem, a read-only source
snapshot, all Linux capabilities dropped, `no-new-privileges`, a fresh
temporary Rust target volume, and eight replay workers. The temporary target
volume was removed after the run.

The offline Cargo transport was committed under a separate domain before the
build:

```text
regular files    43
total bytes      3,907,160
manifest root    sha256:60e5cad5134fc5aeac81185e73597469356d51da59e8fe72379ecdd402b38b59
Rust binary      f1c7b5295e7e42a2d2ca92054c7ae37f41cacb02a4ad20a265ba8fd8ad6413a5
```

The isolation claim is technical role/process isolation under one accepted
administrative controller. The report explicitly records that the actors are
not organizationally independent and are not independent humans.

## 5. Authority boundary and next admission

The result preserves every closed guard:

```text
execution_state                 NOT_RUN
closure_executed                false
formal_roots_generated          false
formal_roots                    null
seed/signature/certificate      absent
target roles evaluated          false
ACTIVE governance changed       false
formal state transition allowed false
```

Accordingly, this evidence admits development of the independent shrink-5
complete-enumeration diagnostic. It does not establish `COMPLETE`,
`DSL_TOO_LARGE`, an odd-target result, a hidden-sink result, MDL success, or
`OUTSIDE_FROZEN_CLOSURE(...)`. Any complete diagnostic must be built in a new
source commit and run only from that immutable child basis. No remote push is
part of this evidence step.
