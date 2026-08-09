# Hegel Machine Phase-3 shrink-6 sealed dual strict qualification protocol v1

Status: **SEALED ENGINEERING QUALIFICATION PROTOCOL; NOT RUN**

This protocol defines the commit-bound dual-recognizer qualification for
`hegel-old-dsl-v1.6.0`. It is not a closure run and has no authority to create
formal roots, evaluate target roles, issue a certificate, or move M3 out of
`NOT_RUN`.

## 1. Immutable parent basis

```text
Evidence V commit  5bfe8474ca63abbadb1d3484a51ce3012081dfb3
artifact path      Hegel Machine/artifacts/phase3_shrink5_dual_complete_enumeration_diagnostic_v1.json
artifact SHA-256   99a799e34876754a8f938f8e25f756992d0784b03bae398b1434e57320b80c82
record ID          phase3_shrink5_dual_complete_enumeration_diagnostic_f33b86f3fbab70acb7d8e61fa47f59568a0d56c884c4cf75dfef961cc73dd34b
status             DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS
route              max_total_ast_depth 4 -> 3
authority          ENGINEERING_ONLY
```

The supervisor accepts only a full 40-hex Source-W commit whose sole parent is
Evidence V. It compares the parent artifact with the Evidence-V Git blob and
the exact SHA above, then validates its record ID, v1.5 identity, status,
claim level, route, `NOT_RUN`, null roots, and no-promotion guards.

## 2. Child contract

```text
DSL                      hegel-old-dsl-v1.6.0
freeze                   hegel-freeze-p2b-p3-v1.6.0
amendment                hegel-freeze-p2b-p3-v1.6.0-shrink-step6
step                     SHRINK_STEP_6_REDUCE_MAX_TOTAL_AST_DEPTH_4_TO_3
maximum_ast_depth        3
maximum_ast_node_count   6
maximum_top_level_clauses 2
formal bucket count      120
```

The child first executes the complete parent admission path and then enforces
the normalized depth limit. Accepted parent/child wires must have identical
canonical CBOR, AST hash, root operator, output sort, depth, node count, and
MDL length.

## 3. Sealed commitments and schemas

Independent Python and Rust generation agreed bit-for-bit before these values
were inserted. The supervisor validates against these constants and never
derives an expected value from the endpoint report under test.

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

The exact counts are 1,266 challenge sources and parent accepts, 1,249 unique
parent programs, 175/175 inherited survivors, 67/50 normalization survivors,
1,199/1,199 parent-only depth-four rows, and 242/225 combined child survivors
(source/unique). Source and formal decoding each reject all 1,199 parent-only
rows with `REJECT_STRUCTURAL_LIMIT`.

The first sorted survivor is CBOR `820183010283010083000100`, hash
`sha256:0f319bb95ea24abc9b4c62d03274a20cefe5dbb92fcfffbce0f0e9449aab04a6`;
the last is CBOR `820186000305030180`, hash
`sha256:e35153f2bdd1a6e25d629ed3ab9afb178bb45ecd163efba4960a2a69db40ce2c`.

Exact top-level endpoint schemas contain 17/13 fields for Python strict
accept/reject; Rust source strict contains 22/17, while Rust formal strict
contains 23/18 because it adds `generic_cbor_parse`. Python/Rust capacity
replay contains 63/62 and golden replay contains 35/34. The supervisor retains
`FAIL_SHRINK6_DUAL_STRICT_UNSEALED_COMMITMENTS` as a fail-closed source guard,
but every required Source-W sentinel is now non-null.

## 4. Ordered vectors and capacity semantics

The manifest contains exactly 25 ordered vectors. Per-vector input bytes,
expected disposition, normalized acceptance/rejection bytes, and the ordered
outcome root must agree across Python and Rust. JSON parsing rejects duplicate
keys and non-finite numbers; equality is canonical-byte and type-strict, so
`true`, `1`, and `1.0` are distinct.

The capacity status is exactly:

```text
FROZEN_DEPTH4_CHALLENGE_LATTICE_ONLY_NOT_COMPLETE
```

The replay must report separately:

- inherited survivor source and unique-program counts;
- normalization-survivor source and unique-program counts;
- parent-only depth-four source and unique-program counts;
- source/formal `REJECT_STRUCTURAL_LIMIT` counts;
- ordered input, accepted-set, and rejection commitments; and
- declared template-family counts.

The challenge lattice is finite and target-free but not a complete language or
closure enumeration. `first_out_of_budget_ordinal`, closure cardinality, and
all formal roots remain null in this phase.

## 5. Commit-bound hard isolation

The supervisor archives only selected Git blobs from Source W, checks their
mode, blob OID, byte SHA, framed source-set root, and archive SHA, safely
extracts them, and never executes mutable worktree bytes.

```text
Python image python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3
Rust image   rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89
pull policy  never
network      none
```

Every Docker invocation uses the explicit Unix socket, a private empty client
configuration, read-only root filesystem, dropped capabilities,
no-new-privileges, purpose-specific seccomp, fixed resources, and fresh tmpfs.
Runtime recognizers use UID/GID 65534. Python runs `-I -S -B`. Rust builds
`--release --locked --offline`; only committed read-only Cargo cache/index
seed bytes are mounted, and the fresh target volume is removed after the run.

Python and Rust vector/capacity work is dispatched concurrently. The
`--workers` argument is recorded in the runtime receipt and bounded to a safe
positive range, but cannot alter semantic ordering or identity.

## 6. Authority guard and invocation

A sealed pass may claim only:

```text
status       SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS
claim level  NON_FORMAL_DUAL_STRICT_QUALIFICATION
```

It must retain `NOT_RUN`, null formal roots, no target/split/secret access, no
seed/key/signature, no certificate, no ACTIVE change, and no formal state
transition. Source W contains no observed shrink-6 closure output.

After Source W exists, invocation is:

```bash
python3 'Hegel Machine/tools/phase3_shrink6_dual_strict_qualification_v1.py' \
  --basis-commit FULL_40_HEX_SOURCE_W_COMMIT \
  --workers 8
```

Success emits one canonical diagnostic JSON object; failure emits one stable
fail-closed object and no evidence artifact.
