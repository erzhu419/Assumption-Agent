# Hegel Machine Phase-3 shrink-3 dual complete-enumeration diagnostic profile v1

Status: **FROZEN FOR A NON-FORMAL DUAL CHILD DIAGNOSTIC**

Machine profile ID: `hegel-m3-shrink3-dual-diagnostic-profile-v1`
Claim level: `NON_FORMAL_DUAL_CHILD_DIAGNOSTIC`

## 1. Purpose and authority boundary

This profile preregisters a target-free Python/Rust replay for
`hegel-old-dsl-v1.3.0`. Its sole language delta from v1.2.0 is that
`BinaryOperatorId/v1=0` (`add`) is a permanent tombstone while ID 1
(`difference`) remains active. It may route later engineering, but it cannot
instantiate formal roots or execute formal M3.

Every endpoint and host receipt must carry:

```text
execution_state = NOT_RUN
formal_roots_generated = false
formal_roots = null
formal_state_transition_allowed = false
authoritative_claim_allowed = false
target_roles_evaluated = false
split_material_accessed = false
secrets_accessed = false
```

The protocol/implementation commit is Source Commit M. Python, Rust, and host
replay must run only from immutable `git archive` snapshots of Commit M.
Observed counts, witnesses, roots, runtime identities, and external archive
digests belong only to a later Evidence Commit N.

The engineering lineage is:

```text
parent diagnostic result commit = d9334589343554841d9f9fd30456a7402bcc7d33
parent implementation basis     = f94cf1fb27c6734f24d4510efba0ca3726132706
parent DSL/freeze                = hegel-old-dsl-v1.2.0 / hegel-freeze-p2b-p3-v1.2.0
child DSL/freeze                 = hegel-old-dsl-v1.3.0 / hegel-freeze-p2b-p3-v1.3.0
human amendment                  = hegel-freeze-p2b-p3-v1.3.0-shrink-step3
shrink step                      = SHRINK_STEP_3_REMOVE_ADD_RETAIN_DIFFERENCE
```

The sealed 36-vector dual strict qualification is prior engineering evidence;
it is not closure execution and does not promote `NOT_RUN`.

## 2. Frozen non-formal bindings

Each root below is plain SHA-256 of the displayed UTF-8 byte string. `\x00`
denotes one NUL byte. The first three roots are embedded in every diagnostic
program record.

| field | exact preimage | SHA-256 |
|---|---|---|
| `child_dsl_spec_root` | `HEGEL/M3/SHRINK3/DIAGNOSTIC_BINDING/V1\x00CHILD_DSL_SPEC\x00hegel-old-dsl-v1.3.0\x00hegel-freeze-p2b-p3-v1.3.0\x00shrink-step3` | `64aaf01392ca89a1ade3a3766d756b53e9b0e7ec6ab4ca2b4fb74ec658490677` |
| `operator_semantics_root` | `HEGEL/M3/SHRINK3/DIAGNOSTIC_BINDING/V1\x00OPERATOR_SEMANTICS\x00hegel-old-dsl-v1.3.0\x00hegel-canonical-ast-v1\x00hegel-mdl-prefix-v1.0.0\x00binary-active-formal:1,2,3,5,6\x00binary-tombstones:0\x00binary-source-alias:4` | `e3337cc67974c8fbbfa6d8f89301184c1658a98b80c0a1fac11251ede9aa15f1` |
| `identifier_registry_root` | `HEGEL/M3/SHRINK3/DIAGNOSTIC_BINDING/V1\x00IDENTIFIER_REGISTRY\x00hegel-old-dsl-v1.3.0\x00aggregate-active:0,1,5\x00aggregate-tombstones:2,3,4\x00rational-active:1,3,5\x00rational-tombstones:0,2,4,6\x00rational-reserved:7\x00binary-source-active:1,2,3,4,5,6\x00binary-formal-active:1,2,3,5,6\x00binary-source-alias:4\x00binary-tombstones:0\x00binary-reserved:7` | `9dd80c452334db8afd9fbb56f1c74f365f63db61ec4c5667bddbb88e57ec05c8` |
| `canonical_ast_schema_root` | `HEGEL/M3/SHRINK3/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_AST_SCHEMA\x00hegel-canonical-ast-v1\x00strict-numeric-tag-cbor-array` | `bc426bb2dbb519533cfe79a33e5d677d23a56b6f73a4e781f327fc303946f92c` |
| `canonical_cbor_profile_root` | `HEGEL/M3/SHRINK3/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_CBOR_PROFILE\x00hegel-cbor-det-v1\x00RFC8949-deterministic-no-map-text-float-tag-indefinite` | `072d40b7db3b91c283ff401b77766149f9b3ae7283cf8fc865090052100ccc96` |

These are synthetic diagnostic identities, not content roots in a formal
child root DAG. They must never be promoted, converted, or signed as formal
roots.

## 3. Frozen surface, budgets, and report schema

The inherited limits remain depth `0..4`, node count `1..7`, output sort IDs
`1..5`, 50,000 canonical programs, 5,000,000 raw attempts, 4,096 records per
chunk, and 175 sort-major accounting buckets.

- Aggregate active IDs: `[0,1,5]`; tombstones: `[2,3,4]`.
- Rational active IDs: `[1,3,5]`; tombstones: `[0,2,4,6]`; reserved: `[7]`.
- Binary source-active IDs: `[1,2,3,4,5,6]`.
- Binary formal-canonical IDs: `[1,2,3,5,6]`.
- Binary source alias: `[4]`; tombstone: `[0]`; reserved: `[7]`.
- `add` is never generated, folded, migrated, or archived; `difference`
  remains ordered and active.
- Pre-count aliases `greater_equal` and `approx_equal:tolerance=0` remain
  excluded.

The traversal key is exactly:

```text
(ast_depth, ast_node_count, output_sort_id,
 root_operator_id, canonical_ast_cbor_bytes)
```

Buckets are fully generated and sorted before a threshold decision. Program
records are traversal-major; bucket rows are sort-major with index
`(sort-1)*35 + depth*7 + (nodes-1)` and include all zero buckets.

The Rust report has exactly the inherited 59 common fields plus these seven:

```text
active_source_binary_operator_ids
active_formal_canonical_binary_operator_ids
source_alias_binary_operator_ids
tombstoned_binary_operator_ids
reserved_binary_operator_ids
operator_id_compaction_performed
automatic_operator_migration_performed
```

Thus Rust has exactly 66 fields. Python has those fields plus exactly
`loaded_hegel_modules`, `target_free_isolation_verified`, and
`target_or_split_modules_loaded`, for 69 fields. Unknown and missing fields,
duplicate JSON keys, and non-finite constants are fatal.

`DSL_TOO_LARGE` is reserved for both exact frozen budgets. A reduced canonical
or raw budget may report only `DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED` and has no
routing authority. Reaching the raw cap before a closed decision is
`INCONCLUSIVE_BUDGET` and must publish no output directory.

## 4. Isolated endpoints

Python is invoked directly under `-I -S -B`:

```text
python3 -I -S -B \
  src/hegel_machine/phase3_m3_shrink3_isolated_entrypoint_v1.py \
  --enumerate-diagnostic \
  --child-dsl-spec-root 64aaf01392ca89a1ade3a3766d756b53e9b0e7ec6ab4ca2b4fb74ec658490677 \
  --operator-semantics-root e3337cc67974c8fbbfa6d8f89301184c1658a98b80c0a1fac11251ede9aa15f1 \
  --identifier-registry-root 9dd80c452334db8afd9fbb56f1c74f365f63db61ec4c5667bddbb88e57ec05c8 \
  --output-directory OUTPUT
```

It installs a minimal package shell and checks an exact 14-module target-free
allowlist before enumeration, after enumeration, and before publication. The
host is also direct-only:

```text
python3 -I -S -B \
  src/hegel_machine/phase3_m3_shrink3_dual_diagnostic_entrypoint_v1.py \
  --validate-dual \
  --python-output-directory PYTHON_OUTPUT \
  --rust-output-directory RUST_OUTPUT
```

The host freezes the same 14 modules plus its validator. Any package
initializer, target, split, seed, role, sink, odd, or evaluator module is
fatal.

All containers use the already-cached pinned image, `--pull never`,
`--network none`, a read-only source snapshot and root filesystem, dropped
capabilities, no mounted secrets, and only fresh dedicated output/target
mounts. Rust uses `cargo --release --locked --offline`.

The Rust build does not trust or mount a pre-unpacked Cargo `src` cache. The
supervisor hashes the exact read-only `cache`/`index` seed, verifies every
`.crate` SHA-256 against the committed `Cargo.lock`, copies only those two
subtrees into a fresh tmpfs `CARGO_HOME`, and lets Cargo unpack them again
under `--locked --offline`. The dependency seed file-set root is recorded in
the supervisor receipt.

The exact cached images are Python
`sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3`
and Rust
`sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`.

## 5. Dual replay obligations

The host must:

1. require the exact four-file output set and immutable regular files;
2. normalize only endpoint implementation identity and Python isolation
   fields, then compare every common report field;
3. compare all three framed streams byte for byte;
4. decode every program and witness with the shrink-3 decoder, thereby proving
   absence of removed `add`, and replay index, AST/hash, metadata, MDL Q32,
   three bindings, uniqueness, and global order;
5. derive the terminal boundary from an empty shrink-3 generator before
   consulting either endpoint witness;
6. replay all reported program and chunk frames plus all 175 bucket frames;
   the `DSL_TOO_LARGE` path requires exactly 50,000 programs and 13 chunks,
   while `COMPLETE` uses its exact closed cardinality and ceiling chunk count;
   replay includes RFC6962 roots, chunk blob hashes/ranges, raw counter
   partitions, and all zero buckets; and
7. for `DSL_TOO_LARGE`, prove the reported witness is the absent adjacent
   rank-50,001 program in the fully closed boundary bucket, with no later
   traversal bucket touched.

The receipt must state that the boundary derivation is
`INDEPENDENT_OF_ENDPOINT_REPORTED_WITNESS_NOT_A_THIRD_IMPLEMENTATION`: it
reuses the committed Python generator and is not a third enumerator. For
`DSL_TOO_LARGE`, the raw-count scope is exactly
`THROUGH_FULLY_CLOSED_BOUNDARY_BUCKET`; for `COMPLETE`, it is
`THROUGH_FULLY_CLOSED_FRONTIER`.

`COMPLETE` instead requires the entire typed frontier to close, no witness,
and no residual canonical program. Any mismatch is
`INCONCLUSIVE_DIAGNOSTIC`; one endpoint may not be selected over the other.

## 6. Probe quarantine and routing

Earlier Python-only observations, including raw count `3,120,739`, witness
hash prefix `040eb...`, and bucket-root prefix `085c...`, are not expected
values in this source freeze and carry no routing authority. They may be
mentioned only after the dual run as historical cross-checks. Earlier program
and chunk roots bind different synthetic inputs and are not reusable.

- Dual `DSL_TOO_LARGE`: only after Evidence Commit N binds a passing dual run,
  route to the already-preregistered shrink-order step 4,
  `reduce max_top_level_clauses from 3 to 2`; do not invent another shrink and
  do not assert a formal child terminal.
- Dual `COMPLETE`: begin formal child root and implementation
  requalification; do not evaluate roles directly.
- Any mismatch, raw-cap exit, partial archive, module drift, overwrite, or
  host failure: remain inconclusive and do not shrink.

Nothing in this profile establishes full closure cardinality beyond a bounded
rank-50,001 result, evaluates odd/sink roles, issues
`OUTSIDE_FROZEN_CLOSURE(...)` or an MDL certificate, signs a formal object,
starts M3, authorizes M4, or changes ACTIVE governance.
