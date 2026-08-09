# Hegel Machine Phase-3 shrink-5 sealed dual strict qualification protocol v1

Status: **ENGINEERING QUALIFICATION PROTOCOL; NON-FORMAL AND NOT RUN**

This protocol qualifies the v1.5.0 admission boundary before any independent
complete-enumerator diagnostic. It does not run closure, publish a program
count or witness, create a formal root, evaluate a target, instantiate a seed,
sign a certificate, or move M3 out of `NOT_RUN`.

## 1. Immutable basis

The only parent authority is:

```text
Evidence R commit  1bbdae8f3131625621c0bc1cfdfe5d7da6035e13
record ID          phase3_shrink4_dual_complete_enumeration_diagnostic_5693b38315689969a1a525b75bec2917f95af1aa54951267797a0319afc60521
artifact SHA-256   2d653f667d8d43e0e8e68c54d6f0a939aab57bf6ba3add9b334809ca17745058
route              max_total_node_count 7 -> 6; top clauses remain 2
authority          ENGINEERING_ONLY
```

The supervisor requires its full 40-hex basis commit to be a single-parent
direct child of Evidence R. It requires the current parent artifact bytes to
equal the Git blob at Evidence R and its hard-pinned SHA-256, then revalidates
the record ID, status, claim level, exact routing object, `NOT_RUN`, null formal
roots, and no-formal-promotion guard.

The child identity is:

```text
DSL        hegel-old-dsl-v1.5.0
freeze     hegel-freeze-p2b-p3-v1.5.0
amendment  hegel-freeze-p2b-p3-v1.5.0-shrink-step5
step       SHRINK_STEP_5_REDUCE_MAX_TOTAL_NODE_COUNT_7_TO_6
```

Every strict, golden and capacity report binds both:

```text
maximum_ast_node_count    6
maximum_top_level_clauses 2
```

## 2. Sealed non-formal roots

The supervisor recomputes and hard-pins these five engineering bindings:

```text
child_dsl_spec_root         3340b3278caf562b560cc30cd14d3cd5f1d628e222b43d29d9d1e41b379f5675
operator_semantics_root     5d2700884ae7125b9712a2bd06aa929feaf2fad1d4bfcd4fa5953c157a720ee1
identifier_registry_root    1b0c141126b278778009d3ebbbf49f5de231ad0166a88a8a9caf367b35bff8ef
canonical_ast_schema_root   828fdcc9f16ebd590702ff4297cac6f6ffa19b01299ea7a93753a4fced0961c5
canonical_cbor_profile_root 0ccbd740c0b1f6a39fb8151ea56e114561093ee4fccb228bf83a9294e0bae783
```

They are domain-separated SHA-256 diagnostics, not formal CBOR/RFC6962
commitments and not state-transition authority.

## 3. Sealed 22-vector replay

The exact ordered vector IDs are:

```text
S01 S02 S03 N01 N02 L01 L02 P01 P02 P03 P04 P05
F01 F02 F03 F04 F05 F06 F07 F08 F09 F10
```

```text
manifest root  sha256:156f7e20407437bb753b097a87932f469701d1de6d1d577b0fa1b7a98f47e52e
outcome root   sha256:8f82178c0f33d5295601d2e112b0b6e25ef18d73e5fc35d8d601024c1f0ddf94
vector count  22
```

The vectors cover:

- byte-identical parent survivors at node counts below or equal to six;
- source normalization and deduplication before the child node check;
- source and formal seven-node structural rejection;
- inherited type, registry, tombstone, reserved-ID and noncanonical priority;
- exact parent/child CBOR and AST-hash identity for every accepted wire.

Each endpoint receives the same exact source JSON or deterministic CBOR bytes.
For acceptance, the supervisor recomputes `HEGEL/AST/V1` and normalizes:

```text
"ACCEPT" || 0x00 || u64be(cbor_length) || canonical_ast_cbor || raw_ast_hash
```

For rejection it normalizes:

```text
"REJECT" || 0x00 || ASCII(error_code)
```

Non-normative Python/Rust error prose is excluded. Acceptance comparison binds
exact CBOR, AST hash, root operator, output sort, depth, node count, maximum
AST nodes and maximum top-level clauses. Category totals alone cannot pass.

## 4. Complete survivor and boundary replays

The capacity control combines two disjoint, exact inherited sets:

- 175 atom survivors: 15 constant comparisons, 16 rational aggregate leaves
  and 144 mixed comparisons; and
- the 2,160-source AND2 parent-only boundary set.

Both implementations must establish byte/hash/MDL-stable parent identity for
all 175 survivors. Independent parent replay establishes that all 2,160 AND2
programs have seven nodes, so both child source and child formal boundaries
must reject all 2,160. Required counts include:

```text
survivor_source_candidate_count                       175
survivor_accepted_count                               175
survivor_unique_count                                 175
survivor_parent_identity_match_count                  175
parent_only_source_candidate_count                   2160
parent_only_parent_accepted_count                    2160
parent_only_node_count                                  7
parent_only_source_child_rejected_count              2160
parent_only_source_child_rejection_counts.REJECT_STRUCTURAL_LIMIT 2160
parent_only_formal_child_rejected_count              2160
parent_only_formal_child_rejection_counts.REJECT_STRUCTURAL_LIMIT 2160
executed_closure_status                           NOT_RUN
formal_roots                                      null
```

```text
survivor accepted-set commitment       sha256:f5ab7f079ad943d65a74881eb59c7bb46385e1c437ca8ab036bb071dfa3874ac
parent-only input-set commitment       sha256:7e0e8780149f03ce85723408f7e3eff2cd684e8938896125cf8e34be9ac70b5e
source rejection outcome commitment   sha256:8617b56bdfa347f11f2c68b6a41f0992652f1e23e6d651017b17eb50169a9f39
formal rejection outcome commitment   sha256:9a6b489ed90960008aebbecdbcf0bc5cf1595b7a8206d179bbe898540dabf617
capacity fields                       Python 46 / Rust 45
golden fields                         Python 34 / Rust 33
```

The report must not use the former
`FULL_AND2_SURVIVOR_SET_ONLY_NOT_COMPLETE` label: the AND2 inputs do not
survive. The combined sets are complete for their declared survivor/boundary
roles but remain only constructive subsets of the child language; they are not
a closure enumeration and never mean `COMPLETE`.

All Python/Rust shared fields must be exactly equal. Python may add only its
sealed loaded-module receipt. Unknown, missing, aliased, substituted, or extra
keys fail closed at every report nesting level. Diagnostic JSON rejects
duplicate keys and non-finite numbers; equality is canonical-byte and
type-strict, so `true`, `1`, and `1.0` are never aliases.

## 5. Commit-bound hard isolation

The qualification executes only the committed Source-S snapshot. The
supervisor records every selected Git blob, a framed source-file-set root and
the Git archive hash; it safely extracts the archive and rehashes every file.
No recognizer executes mutable worktree bytes.

The endpoints use distinct locally cached digest-pinned images:

```text
Python  python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3
Rust    rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89
```

Every Docker call uses the explicit Unix socket and private empty client
configuration, `--pull=never`, `--network=none`, a read-only root filesystem,
dropped capabilities, no-new-privileges, purpose-specific seccomp, fixed
memory/PID/nofile limits and fresh tmpfs. Runtime recognizers use UID/GID
65534. Python runs `-I -S -B` from a read-only snapshot mount.

Rust builds from the committed archive with `--release --locked --offline`.
Only read-only Cargo `cache` and `index` seed subtrees are mounted; the build
uses a fresh tmpfs Cargo home and fresh commit-named Docker target volume. The
supervisor commits every seed file's relative path, mode, size and SHA-256
under `HEGEL/SHRINK5/CARGO_SEED_MANIFEST/V1`, requires the same manifest after
the build, freezes produced files, mounts the runtime volume read-only, and
removes it after qualification. No image pull, registry access, target/split
data, seed, key, signature, or formal-root material is allowed.

This supplies technical process/container independence under one
administrative controller. It does not claim organizational independence or
independent-human custody.

## 6. Two-commit rule and authority guard

1. Source Commit S freezes code, vectors, profiles, tests and this protocol.
2. Qualification executes only the immutable Source-S archive.
3. Evidence Commit T may record the successful Source-S-bound report.

Source S contains no observed closure output. A strict pass is limited to:

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

The final report uses exact schemas recursively for repository, vector,
capacity, runtime, Docker, source and Cargo receipts. Missing or additional
keys fail closed so later closure outputs cannot be smuggled into a strict
qualification artifact.

## 7. Invocation after Source S exists

```bash
python3 'Hegel Machine/tools/phase3_shrink5_dual_strict_qualification_v1.py' \
  --basis-commit FULL_40_HEX_SOURCE_S_COMMIT \
  --workers 8
```

Success emits one canonical diagnostic JSON object. Failure emits one stable
fail-closed object to stderr and no evidence artifact. A pass authorizes only
the next independent shrink-5 complete-diagnostic source phase; it does not
start formal M3.
