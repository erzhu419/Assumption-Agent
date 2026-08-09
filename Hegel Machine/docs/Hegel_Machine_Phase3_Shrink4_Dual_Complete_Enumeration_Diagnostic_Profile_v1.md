# Hegel Machine Phase-3 shrink-4 dual complete-enumeration diagnostic profile v1

Status: **SOURCE Q ENGINEERING FREEZE; NON-FORMAL AND NOT RUN**

Machine profile ID: `hegel-m3-shrink4-dual-diagnostic-profile-v1`

Claim level: `NON_FORMAL_DUAL_CHILD_DIAGNOSTIC`

## 1. Purpose and authority boundary

This profile preregisters a target-free Python/Rust complete-enumeration
diagnostic for `hegel-old-dsl-v1.4.0`. The only child-language delta from
v1.3.0 is the normalized top-level conjunction limit:

```text
maximum_top_level_clauses: 3 -> 2
```

No registry ID, operator meaning, type rule, AST/CBOR identity, MDL rule,
budget, traversal key, or equivalence relation changes. In particular, the
enumerator constructs normalized AND2 candidates only. It never generates an
AND3 candidate and therefore records:

```text
maximum_top_level_clauses          = 2
and3_generator_attempts_allowed    = false
and3_raw_operator_application_count = 0
```

Every endpoint remains:

```text
execution_state                  = NOT_RUN
formal_roots_generated           = false
formal_roots                     = null
authoritative_claim_allowed      = false
target_roles_evaluated           = false
split_material_accessed          = false
secrets_accessed                 = false
```

The host receipt carries the same fields and additionally freezes:

```text
formal_state_transition_allowed = false
```

The supervisor summary expresses the same isolation boundary with its own
exact schema:

```text
execution_state                  = NOT_RUN
formal_roots_generated           = false
formal_roots                     = null
formal_state_transition_allowed = false
authoritative_claim_allowed      = false
target_roles_evaluated           = false
split_material_accessed          = false
seeds_accessed                   = false
keys_or_signatures_generated     = false
```

Source Q contains protocol and implementation only. It contains no observed
shrink-4 enumeration count, raw count, witness, program/archive root, chunk
root, bucket root, output archive digest, runtime result, or terminal verdict.
Python, Rust and the host replay may execute only from an immutable
`git archive` of the later committed Source Q. A result belongs only to the
separate Evidence R commit.

## 2. Unique engineering admission

Source Q has exactly one engineering admission: Evidence P and its canonical
qualification report.

```text
strict_qualification_source_commit
  = cd2c32bd3a27004b40f4550229f33afd73647433
strict_qualification_evidence_commit
  = c78e19b44ca85645d20790d7aefe1d8137b4e2bb
strict_qualification_artifact_path
  = Hegel Machine/artifacts/phase3_m3_runtime/phase3_shrink4_sealed_dual_strict_qualification_v1.json
strict_qualification_artifact_sha256
  = 41fdea5fd9b16ab436386ef7794412ffa46e17e68efc6b8448deed17c7f99aae
strict_qualification_diagnostic_report_hash
  = sha256:44b4e0c0a2b79f6afb67ace348c1b3726e0ba64058c97c4c61be0c111ef6acec
strict_qualification_status
  = SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS
```

The Source Q commit must be a single-parent child of Evidence P. Its source
manifest must include the byte-identical artifact, and the supervisor replays
the Source-O validator over those bytes before creating any container. The six
fields above are required unchanged in both endpoint reports and the host
receipt. Evidence P remains non-formal and leaves M3 at `NOT_RUN`.

## 3. Frozen non-formal bindings

Each value is plain SHA-256 of the corresponding frozen UTF-8 preimage in
`phase3_m3_shrink4_diagnostic_profile_v1.py`. They are diagnostic bindings,
not formal roots.

| field | SHA-256 hex |
|---|---|
| `child_dsl_spec_root` | `736c9cf98749d9a9d2d98596d15a5b09329e1d6eb74d4bee172837fdd34e876f` |
| `operator_semantics_root` | `45fe7c575759b6955eb6b52ad954a9ca6561083dbdb67155f9731e795c6fe050` |
| `identifier_registry_root` | `1f9b886480ace19440469267abd06f24c65ef61fb9c734c5ef5ff8ae7e981fd3` |
| `canonical_ast_schema_root` | `f9b02ddad69f04f1f9137501dccfdcefa111d0402570197b68b98c11ebcb4eda` |
| `canonical_cbor_profile_root` | `b7fd10722f31d780d53b2f490c92491872ffc749b4cb5cdfccc3eebd5f18837f` |

The first three values bind every diagnostic program record. All five must
remain byte-stable with the Evidence P qualification. They may not be
converted, signed, or reused as formal child roots.

## 4. Budgets, traversal and exact report schemas

The frozen budgets remain 50,000 canonical programs before extensional
quotient, 5,000,000 raw operator applications, 4,096 program records per
chunk, depth `0..4`, node count `1..7`, output sort IDs `1..5`, and 175
sort-major accounting buckets. These are preregistered limits, not observed
results.

The traversal key is exactly:

```text
(ast_depth, ast_node_count, output_sort_id,
 root_operator_id, canonical_ast_cbor_bytes)
```

Buckets are completely generated and sorted before a threshold decision.
Program records are traversal-major. Bucket rows are sort-major, include all
zero buckets, and use index `(sort-1)*35 + depth*7 + (nodes-1)`.

The Rust report has exactly 75 common fields. The Python report has those 75
plus exactly `loaded_hegel_modules`, `target_free_isolation_verified`, and
`target_or_split_modules_loaded`, for 78 fields. Relative to shrink-3, the
common schema adds the three shrink-4 structural fields and the six Evidence P
authority fields listed above. All nested report and receipt objects use exact
key sets. Duplicate JSON names, unknown or missing fields, non-finite numbers,
wrong nullability, and legacy aliases are fatal.

`DSL_TOO_LARGE` is reserved for both exact frozen budgets. A reduced diagnostic
budget may emit only `DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED` and has no routing
authority. Reaching the raw cap before a closed decision is
`INCONCLUSIVE_BUDGET`; it must publish no output directory.

## 5. Isolated dual execution

The Python and Rust endpoints start concurrently in disjoint digest-pinned
containers. The host replay starts only after both immutable four-file output
sets exist. All three actors receive the same read-only Source Q archive and
no target, role, split, seed, key, signature, or formal-root input.

The fixed controls are:

- `--pull never` and `--network none`;
- a read-only root filesystem and source snapshot;
- all capabilities dropped and `no-new-privileges` enabled;
- pinned Python and Rust image digests;
- Python `-I -S -B` direct-file entrypoints with exact target-free module
  allowlists;
- one fresh local-driver Rust target volume, build read-write and runtime
  read-only, removed after every success or failure; and
- an external, initially absent result directory.

The Rust build mounts only the exact Cargo `cache` and `index` file set. Each
regular file is bound by ordered path, mode, size and SHA-256. Every `.crate`
is checked against the committed `Cargo.lock`; pre-unpacked `src` is excluded.
The two subtrees are copied into a fresh 64 MiB tmpfs `CARGO_HOME`, Cargo runs
with `--release --locked --offline`, and the exact manifest is replayed before
and after the build. The cap-dropped build explicitly does not copy host UID/GID
ownership; bytes, paths, modes and the frozen manifest remain authoritative. A
changed dependency byte or mode is fail-closed.

## 6. Dual replay obligations

The host must:

1. reject any output set other than the exact report plus three framed streams,
   and require each captured endpoint stdout object to equal its retained
   `report.json` object;
2. validate recursive exact schemas, raw-count lower bounds and all fixed
   Evidence P/structural guards;
3. normalize only implementation identity and the three Python-only isolation
   fields, then compare every common report field;
4. compare all framed stream bytes before decoding them;
5. decode every program and possible boundary witness through the shrink-4
   decoder, replaying indices, AST/hash identity, metadata, Q32 MDL, the three
   diagnostic bindings, uniqueness and global traversal order;
6. independently derive the terminal boundary from an empty committed
   shrink-4 generator, without treating either endpoint's reported boundary
   values or witness as authoritative;
7. replay program, chunk and all 175 bucket frames, including raw-counter
   partitions, zero buckets, blob hashes and RFC6962 roots; and
8. fail as `INCONCLUSIVE_DIAGNOSTIC` on any mismatch, never selecting one
   endpoint over the other.

The boundary derivation is
`INDEPENDENT_OF_ENDPOINT_REPORTED_WITNESS_NOT_A_THIRD_IMPLEMENTATION`: it
reuses the committed Python generator and is not a third enumerator.

## 7. Evidence R state machine

Only Evidence R may record an observed terminal result and route later work:

- dual `DSL_TOO_LARGE` may open the already-frozen shrink-order step 5,
  `reduce max_total_node_count from 7 to 6`;
- dual `COMPLETE` may open formal child-root and implementation
  requalification; and
- mismatch, raw-cap exit, partial output, schema/module drift, overwrite, or
  isolation failure remains inconclusive and opens no shrink or formal route.

These are conditional transitions, not claims that any outcome has occurred.
Nothing in Source Q or Evidence R alone starts M3, evaluates odd/sink roles,
issues `OUTSIDE_FROZEN_CLOSURE(...)` or an MDL certificate, creates a split
seed, signs a formal object, authorizes M4, or changes ACTIVE governance.
