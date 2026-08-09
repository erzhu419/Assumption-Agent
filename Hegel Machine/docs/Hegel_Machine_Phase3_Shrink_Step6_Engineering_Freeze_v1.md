# Hegel Machine Phase-3 shrink step 6 engineering freeze v1

Status: **SEALED NORMATIVE ENGINEERING FREEZE; NON-FORMAL AND NOT RUN**

This document freezes the final preregistered shrink operation as a child
language construction. It does not execute closure, publish a witness, create
formal roots, evaluate a target role, instantiate a seed or key, sign a
certificate, or move M3 out of `NOT_RUN`.

## 1. Sole admission authority

The sole routing authority is shrink-5 Evidence V:

```text
parent evidence commit  5bfe8474ca63abbadb1d3484a51ce3012081dfb3
parent artifact path    Hegel Machine/artifacts/phase3_shrink5_dual_complete_enumeration_diagnostic_v1.json
parent artifact SHA-256 99a799e34876754a8f938f8e25f756992d0784b03bae398b1434e57320b80c82
parent record ID        phase3_shrink5_dual_complete_enumeration_diagnostic_f33b86f3fbab70acb7d8e61fa47f59568a0d56c884c4cf75dfef961cc73dd34b
parent status           DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS
parent claim level      NON_FORMAL_DUAL_CHILD_DIAGNOSTIC
parent DSL/freeze       hegel-old-dsl-v1.5.0 / hegel-freeze-p2b-p3-v1.5.0
parent execution state  NOT_RUN
parent formal roots     null
```

Evidence V authorizes exactly:

```text
operation                         reduce max_total_ast_depth from 4 to 3
preregistered_shrink_order_step   6
maximum_ast_node_count_remains    6
maximum_top_level_clauses_remains 2
only_open_route                   true
authority                         ENGINEERING_ONLY
formal_status_promotion_allowed   false
```

The shrink-6 source commit must be a single-parent direct child of Evidence V.
An ancestor-only binding, modified artifact, or prefix-substituted record ID is
rejected.

## 2. Child identity and sole delta

```text
parent DSL     hegel-old-dsl-v1.5.0
parent freeze  hegel-freeze-p2b-p3-v1.5.0
child DSL      hegel-old-dsl-v1.6.0
child freeze   hegel-freeze-p2b-p3-v1.6.0
amendment      hegel-freeze-p2b-p3-v1.6.0-shrink-step6
step           SHRINK_STEP_6_REDUCE_MAX_TOTAL_AST_DEPTH_4_TO_3
```

The only language change is:

```text
maximum_ast_depth / max_total_ast_depth: 4 -> 3
maximum_ast_node_count:                    remains 6
maximum_top_level_clauses:                 remains 2
```

The source recognizer must complete the entire v1.5 parse, typing, registry,
tombstone, rewrite, normalization, six-node, and two-clause pipeline before
enforcing depth three on the normalized tree. The formal decoder must first
establish an exact v1.5 canonical AST. A surviving AST keeps identical CBOR,
`HEGEL/AST/V1` hash, registry IDs, and Q32 MDL length. An otherwise valid
canonical depth-four parent returns `REJECT_STRUCTURAL_LIMIT`; it is never
truncated, rewritten across the boundary, or migrated.

Everything else is inherited unchanged, including active/tombstoned/reserved
IDs, explicit coercions, operator typing and meaning, bottom semantics, scope
rules, canonical rewrites, deterministic CBOR, AST identity, MDL, budgets, and
target-independent traversal order.

## 3. Frozen qualification surface

The strict qualification uses an ordered 25-vector manifest. It must cover:

- parent-identical survivors through canonical depth three;
- raw sources whose frozen normalization reduces apparent depth;
- source and formal depth-four structural rejection for each declared
  challenge-template family; and
- inherited malformed, type, tombstone, reserved-ID, alias, and noncanonical
  priorities.

The capacity replay is explicitly named:

```text
FROZEN_DEPTH4_CHALLENGE_LATTICE_ONLY_NOT_COMPLETE
```

It combines the inherited atom survivor controls with a finite, deterministic
set of depth-four challenge templates. Python and Rust must independently
generate the declared sources, parent-canonicalize them, partition normalized
survivors from parent-only depth-four programs, deduplicate by canonical CBOR,
and commit source/formal outcomes. This lattice is complete only for its
declared templates. It is not the complete depth-four parent language, not a
closure sample, and not evidence for `COMPLETE`.

Independent Python and Rust implementations generated the same golden and
capacity values. The resulting engineering commitments are therefore sealed
as follows; sealing them is not execution of closure.

```text
golden vector count     25
golden manifest root    sha256:2690413926d15db52dbd5a502ebe3fdfb1dc74d5ee3c82b2ed868cd16ab34a42
golden outcome root     sha256:e5fd0885f95669dc6d369d0d3274778425fabb7e8c6286a27237a1b2bc8d3960
ordered IDs             S01,S02,S03,N01,N02,L01,L02,L03,P01,P02,P03,P04,P05,F01,F02,F03,F04,F05,F06,F07,F08,F09,F10,F11,F12
```

```text
child DSL root          da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae
operator root           922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03
registry root           64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1
AST schema root         5de72fc51e27e5501561ffda6b05522f4941d1a13c4b324f5edcc15fa584a0bd
CBOR profile root       ef0008912962de9da322eaeea6e421e1e58d16be152f968298774af0fd3249ab
```

The frozen challenge lattice has 1,266 source rows (`A=486`, `B_abs=390`,
`B_sign=390`), all accepted by the parent, and 1,249 unique parent canonical
programs. Its exact partition is:

```text
inherited survivors             175 source / 175 unique
normalization survivors          67 source /  50 unique
parent-only depth-four rows    1,199 source / 1,199 unique
combined child survivors         242 source / 225 unique
parent-only family counts        A=453, B_abs=373, B_sign=373
normalization family counts      A=33,  B_abs=17,  B_sign=17
source structural rejections     1,199 REJECT_STRUCTURAL_LIMIT
formal structural rejections     1,199 REJECT_STRUCTURAL_LIMIT
```

```text
challenge source lattice  sha256:a8cfb37278000933c2c51a2797e5bc0f4e7aad6970b37e178fc681f9358574d0
parent canonical set      sha256:8f125763d3098d087dd7e9eb484b93097295ebd765b6f079795e8009623fb13e
inherited survivor set    sha256:477a5abe659a7a7e7d2d50b2a5bda61b0dae1019c44fe84950c4a05036258619
normalized survivor set   sha256:dcbb5562fc754fdef932188b189dbcdc0f7c500d3fc49651ee4dbb0f271afd29
combined survivor set     sha256:6787cd6c0782fda149e1ee93b37ca8d425f5ac78850c610e21cebf9da13a16d1
parent-only set           sha256:d3eb2b2d9caf1eece5a709d8113540e4709d579cdfbe3194f1cf176c9100b20d
source rejection outcome sha256:9b0b766a4139db6297aea8b6032ad49147c1a26bf9b56291444a83681428cb0e
formal rejection outcome sha256:97d50c34f51683a2502157961acc79d3b4e108b28bdaa266cf3721ffda8b3a96
```

The sorted survivor boundary identities are CBOR
`820183010283010083000100` / hash
`sha256:0f319bb95ea24abc9b4c62d03274a20cefe5dbb92fcfffbce0f0e9449aab04a6`
and CBOR `820186000305030180` / hash
`sha256:e35153f2bdd1a6e25d629ed3ab9afb178bb45ecd163efba4960a2a69db40ce2c`.

Exact top-level endpoint schemas contain 17/13 fields for Python strict
accept/reject; Rust source strict contains 22/17, while Rust formal strict
contains 23/18 because it adds `generic_cbor_parse`. Python/Rust capacity
replay contains 63/62 and golden replay contains 35/34. No shrink-5 commitment
is copied forward as a shrink-6 value.

## 4. Enumerator preregistration without execution

The later child enumerator is preregistered with depths `0..3`, node counts
`1..6`, and five output sorts. Its formal bucket lattice therefore contains:

```text
5 output sorts * 4 depth values * 6 node values = 120 buckets
```

Evidence V observed its shrink-5 rank-50,001 witness at depth two and node
four. Because the child removes only depth four, it is a non-authoritative
engineering prediction that the AST-CBOR/hash prefix, witness wire/ordinal,
and raw count through that boundary remain unchanged. This is not an observed
shrink-6 result. Program records bind new v1.6 diagnostic roots, so program
archive, stream, chunk, and bucket roots must be regenerated and must not be
predicted equal across versions.

The source freeze retains:

```text
execution_state                  NOT_RUN
closure_executed                 false
formal_roots_generated           false
formal_roots                     null
formal_state_transition_allowed  false
target_roles_evaluated           false
certificate/signature/seed       absent
```

## 5. Isolation, parallelism, and two-commit rule

The supervisor executes only a committed Git archive. Python and Rust use
distinct digest-pinned local images with `--pull=never`, `--network=none`, a
read-only root filesystem, dropped capabilities, no-new-privileges, fixed
memory/PID/nofile limits, and fresh tmpfs. Rust uses
`cargo --release --locked --offline` with read-only cache/index seeds and a
fresh commit-named target volume. Once the images and Cargo cache exist, this
qualification performs no network access.

Python and Rust replays run concurrently. `--workers` controls bounded host
orchestration only; it never changes vector order, source order, commitments,
or report identity.

The source and observation remain separate local commits:

1. Source W freezes implementation, vectors, profiles, tests, and protocol,
   with no observed qualification result.
2. Qualification executes only the immutable Source-W archive.
3. A later Evidence commit may record the Source-W-bound successful report.

A strict pass authorizes only construction of the independent shrink-6
complete-diagnostic source. It cannot start formal M3 or promote governance.
