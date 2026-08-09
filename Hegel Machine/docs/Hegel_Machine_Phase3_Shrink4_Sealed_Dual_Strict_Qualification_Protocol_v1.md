# Hegel Machine Phase-3 shrink-4 sealed dual strict qualification protocol v1

Status: **ENGINEERING QUALIFICATION PROTOCOL; NON-FORMAL AND NOT RUN**

This protocol qualifies the v1.4.0 admission boundary before independent
complete-enumerator work. It does not run closure, publish an enumeration
count or witness, create formal roots, evaluate a target, instantiate a seed,
sign a certificate, or move M3 out of `NOT_RUN`.

## 1. Frozen parent and child basis

The only routing authority is Evidence Commit N:

```text
commit       c286732c140bd9adcfd3eef2b1788b3eac0eb3e9
record ID    phase3_shrink3_dual_complete_enumeration_diagnostic_3030ad10f2cd4f767a8397597be1ab3ed6cac7cd71975d69f59cc5abec6a4f5a
route        maximum_top_level_clauses 3 -> 2
authority    ENGINEERING_ONLY
```

The supervisor requires this commit to be an ancestor of its full 40-hex
basis, requires the evidence artifact bytes to equal the artifact at that
commit, and revalidates its record ID, diagnostic status, routing row,
`NOT_RUN`, null formal roots, and no-formal-promotion guard.

The child identity is:

```text
DSL          hegel-old-dsl-v1.4.0
freeze       hegel-freeze-p2b-p3-v1.4.0
amendment    hegel-freeze-p2b-p3-v1.4.0-shrink-step4
step         SHRINK_STEP_4_REDUCE_MAX_TOP_LEVEL_CLAUSES_3_TO_2
```

Every serialized strict, golden, and capacity admission report must contain
the exact field `maximum_top_level_clauses: 2`. The shorter key
`max_top_level_clauses` is not an alias and is rejected.

## 2. Five non-formal roots

The supervisor hard-pins and recomputes the diagnostic profile's five roots:

```text
child_dsl_spec_root       736c9cf98749d9a9d2d98596d15a5b09329e1d6eb74d4bee172837fdd34e876f
operator_semantics_root   45fe7c575759b6955eb6b52ad954a9ca6561083dbdb67155f9731e795c6fe050
identifier_registry_root  1f9b886480ace19440469267abd06f24c65ef61fb9c734c5ef5ff8ae7e981fd3
canonical_ast_schema_root f9b02ddad69f04f1f9137501dccfdcefa111d0402570197b68b98c11ebcb4eda
canonical_cbor_profile_root b7fd10722f31d780d53b2f490c92491872ffc749b4cb5cdfccc3eebd5f18837f
```

They are domain-separated synthetic engineering bindings. They are not formal
CBOR/RFC6962 roots and have no state-transition authority.

## 3. Sealed 22-vector evidence generator

The exact ordered vector IDs are:

```text
S01 S02 S03 N01 N02 L01 L02 P01 P02 P03 P04 P05
F01 F02 F03 F04 F05 F06 F07 F08 F09 F10
```

Their frozen commitments are:

```text
manifest root  sha256:f84035e632bf5a655a9ebd636a0cafe7ab1097c45be87d4db944a0012f52aa90
outcome root   sha256:c19341f08ac5f5759c2cdcb3681a37d958de362b81d02c184f7e2413dca18d7c
vector count  22
```

The exact category counts are:

| report field | count |
|---|---:|
| `surviving_identity_checks` | 3 |
| `source_normalization_before_limit_checks` | 2 |
| `source_structural_limit_checks` | 2 |
| `source_priority_checks` | 5 |
| `formal_surviving_identity_checks` | 1 |
| `formal_structural_limit_checks` | 1 |
| `formal_priority_checks` | 8 |

This matrix covers direct survivors, one-clause collapse, duplicate-clause
collapse before the new limit, source/formal three-clause rejection, and the
priority of malformed/type errors, three inherited tombstones, source-only
alias/noncanonicality, reserved IDs, ordering, and four-clause noncanonical
shape relative to the new limit.

Each endpoint receives the exact source-JSON or deterministic-CBOR wire. For
an accepted vector the supervisor recomputes the AST hash and normalizes:

```text
"ACCEPT" || 0x00 || u64be(cbor_length) || canonical_ast_cbor || raw_ast_hash
```

For rejection it normalizes:

```text
"REJECT" || 0x00 || ASCII(error_code)
```

Non-normative Python/Rust error prose is excluded. Acceptance compares exact
CBOR, AST hash, root operator, output sort, depth, node count, and
`maximum_top_level_clauses`. Category totals without per-wire equality are
insufficient.

## 4. Full 2,160-survivor replay

Both implementations must independently replay the full inherited 2,160
source set. Every source is a normalized AND2 survivor, not a sample from that
set. Required common fields include:

```text
source_candidate_count       2160
normalized_and2_count        2160
accepted_source_count        2160
accepted_unique_count        2160
parent_identity_match_count  2160
rejected_count               0
rewrite_collapsed_count      0
accepted_set_commitment      sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e
subset_status                FULL_AND2_SURVIVOR_SET_ONLY_NOT_COMPLETE
executed_closure_status      NOT_RUN
formal_roots                 null
```

The Python implementation may internally recheck MDL identity, but this
strict report does not invent a separate MDL-match count unless both
implementations independently compute it. The accepted CBOR/hash commitment
plus the unchanged frozen MDL table is the current claim boundary.

Exact schemas are also frozen. The capacity reports have 37 Python fields and
36 Rust fields; the only Python-only field is `loaded_hegel_modules`. Golden
reports have 33 Python fields and 32 Rust fields; again the only Python-only
field is `loaded_hegel_modules`. Both golden reports bind the exact ordered
IDs, manifest root, and outcome root. Unknown, missing, aliased, or extra keys
fail closed.

Even all 2,160 survivors remain a constructive subset of the language. This
replay is never `COMPLETE` and supplies no closure cardinality.

## 5. Commit-bound hard isolation

The qualification basis must be a committed source snapshot containing the
supervisor, profiles, exact Python dependency closure, all inherited Rust
crates, and the shrink-4 Rust crate. Worktree bytes must equal every selected
Git blob. The supervisor records a framed source-file-set root, creates a Git
archive, safely extracts it, and rehashes every executed file. Recognizers
never execute from the mutable worktree.

The two endpoints use distinct locally cached digest-pinned images:

```text
Python  python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3
Rust    rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89
```

Every Docker call uses the explicit local Unix socket, a fresh empty private
client configuration, exact environment allowlist, `--pull never`,
`--network none`, read-only root, dropped capabilities, no-new-privileges,
purpose-specific seccomp, fixed memory/PID/nofile limits, and private tmpfs.
Runtime recognizers use UID/GID 65534. Python uses `-I -S -B` and a read-only
snapshot mount.

Rust builds from the committed archive with `--release --locked --offline`,
read-only `cache` and `index` seed subtrees, and a fresh commit-named local
Docker volume. No pre-unpacked registry `src` tree is mounted; Cargo unpacks
the locked crates into a fresh build-container tmpfs home. Before building,
the supervisor records every regular seed file's canonical relative path,
mode, size and SHA-256 under the domain-separated
`HEGEL/SHRINK4/CARGO_SEED_MANIFEST/V1` root, then requires the same exact
manifest again after the build.
The volume is runtime read-only and removed after qualification. No registry
access, image pull, dependency download, target/split input, seed, key,
signature, or formal-root material is allowed.

This is technical process/container independence under one administrative
controller. It is not organizational or independent-human custody, and the
report says so.

## 6. Two-commit rule and authority guard

The source/protocol commit and observed qualification evidence must be
separate local commits:

1. Source Commit O freezes code, vectors, profiles, tests, and this protocol.
2. Qualification executes only from the immutable Source-O archive.
3. Evidence Commit P may record the successful Source-O-bound report.

Source O contains no observed enumeration outputs. A passing strict report is
limited to:

```text
status       SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS
claim level  NON_FORMAL_DUAL_STRICT_QUALIFICATION
```

It must preserve:

```text
execution_state                  NOT_RUN
closure_executed                 false
formal_roots_generated           false
formal_roots                     null
certificate/signature/seed       absent
target_roles_evaluated           false
ACTIVE governance changed        false
formal state transition allowed  false
```

Every object and row in the qualification report has an exact key set.
Missing or additional keys fail closed at every nesting level; capacity and
runtime receipts therefore cannot carry future enumeration values under a
recomputed diagnostic hash.

Success authorizes only implementation and later qualification of independent
shrink-4 complete enumerators. Failure is fail-closed and has no routing
authority. No observed shrink-4 program count, witness, archive root, chunk
root, bucket root, or closure verdict belongs in Source O.

## 7. Invocation after Source O exists

```bash
python3 'Hegel Machine/tools/phase3_shrink4_dual_strict_qualification_v1.py' \
  --basis-commit FULL_40_HEX_SOURCE_O_COMMIT \
  --workers 8
```

Success emits one canonical diagnostic JSON object. Failure emits one stable
fail-closed object to stderr and no evidence artifact.
