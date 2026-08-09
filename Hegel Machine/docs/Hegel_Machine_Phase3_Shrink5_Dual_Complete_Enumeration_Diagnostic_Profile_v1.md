# Hegel Machine Phase-3 shrink-5 dual complete-enumeration diagnostic profile v1

Status: **SOURCE U ENGINEERING FREEZE; NON-FORMAL AND NOT RUN**

Machine profile ID: `hegel-m3-shrink5-dual-diagnostic-profile-v1`

Claim level: `NON_FORMAL_DUAL_CHILD_DIAGNOSTIC`

## 1. Purpose and authority boundary

This profile preregisters a target-free Python/Rust complete-enumeration
diagnostic for `hegel-old-dsl-v1.5.0`. The only child-language delta from
v1.4.0 is the total canonical-AST node limit:

```text
maximum_ast_node_count: 7 -> 6
```

No registry ID, operator meaning, type rule, AST/CBOR identity, MDL rule,
budget, traversal key, or equivalence relation changes. The inherited AND2
limit remains in force, so the existing AND3 exclusion fields remain part of
the byte-stable 75-field report schema. The child enumerator admits no
seven-node program and records:

```text
maximum_top_level_clauses          = 2
maximum_ast_node_count             = 6
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

Source U contains protocol and implementation only. It contains no observed
shrink-5 enumeration count, raw count, witness, program/archive root, chunk
root, bucket root, output archive digest, runtime result, or terminal verdict.
Python, Rust and the host replay may execute only from an immutable
`git archive` of the later committed Source U. A result belongs only to the
separate Evidence V commit.

## 2. Unique engineering admission

Source U has exactly one engineering admission: Evidence T and its canonical
qualification report.

```text
strict_qualification_source_commit
  = 320b0a3458901090cb738023a4398220fb1d9277
strict_qualification_evidence_commit
  = 01b66cd8effeab258797998f594b250188d823da
strict_qualification_artifact_path
  = Hegel Machine/artifacts/phase3_m3_runtime/phase3_shrink5_sealed_dual_strict_qualification_v1.json
strict_qualification_artifact_sha256
  = 75761fc536d96d5d0bc91c5c0ba30dbc7c9ee21aac8d3f1dc5c96f6aca919b76
strict_qualification_diagnostic_report_hash
  = sha256:5ee04b21477fd9f09271272fd6ecbf876b885b7831b37a868343a93996a187db
strict_qualification_status
  = SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS
```

The Source U commit must be a single-parent child of Evidence T. Its source
manifest must include the byte-identical artifact, and the supervisor replays
the Source-S validator over those bytes before creating any container. The six
fields above are required unchanged in both endpoint reports and the host
receipt. Evidence T remains non-formal and leaves M3 at `NOT_RUN`.

## 3. Frozen non-formal bindings

Each value is plain SHA-256 of the corresponding frozen UTF-8 preimage in
`phase3_m3_shrink5_diagnostic_profile_v1.py`. They are diagnostic bindings,
not formal roots.

| field | SHA-256 hex |
|---|---|
| `child_dsl_spec_root` | `3340b3278caf562b560cc30cd14d3cd5f1d628e222b43d29d9d1e41b379f5675` |
| `operator_semantics_root` | `5d2700884ae7125b9712a2bd06aa929feaf2fad1d4bfcd4fa5953c157a720ee1` |
| `identifier_registry_root` | `1b0c141126b278778009d3ebbbf49f5de231ad0166a88a8a9caf367b35bff8ef` |
| `canonical_ast_schema_root` | `828fdcc9f16ebd590702ff4297cac6f6ffa19b01299ea7a93753a4fced0961c5` |
| `canonical_cbor_profile_root` | `0ccbd740c0b1f6a39fb8151ea56e114561093ee4fccb228bf83a9294e0bae783` |

The first three values bind every diagnostic program record. All five must
remain byte-stable with the Evidence T qualification. They may not be
converted, signed, or reused as formal child roots.

## 4. Budgets, traversal and exact report schemas

The frozen budgets remain 50,000 canonical programs before extensional
quotient, 5,000,000 raw operator applications, 4,096 program records per
chunk, depth `0..4`, node count `1..6`, output sort IDs `1..5`, and 150
sort-major accounting buckets. These are preregistered limits, not observed
results.

The traversal key is exactly:

```text
(ast_depth, ast_node_count, output_sort_id,
 root_operator_id, canonical_ast_cbor_bytes)
```

Buckets are completely generated and sorted before a threshold decision.
Program records are traversal-major. Bucket rows are sort-major, include all
zero buckets, and use index `(sort-1)*30 + depth*6 + (nodes-1)`. Seven-node
programs are outside this child lattice; no new node-7 counter field is added.

The Rust report has exactly 75 common fields. The Python report has those 75
plus exactly `loaded_hegel_modules`, `target_free_isolation_verified`, and
`target_or_split_modules_loaded`, for 78 fields. Relative to shrink-4, the
schema keys are unchanged; the node limit, bucket count and Evidence T binding
values change. All nested report and receipt objects use exact
key sets. Duplicate JSON names, unknown or missing fields, non-finite numbers,
wrong nullability, and legacy aliases are fatal.

`DSL_TOO_LARGE` is reserved for both exact frozen budgets. A reduced diagnostic
budget may emit only `DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED` and has no routing
authority. Reaching the raw cap before a closed decision is
`INCONCLUSIVE_BUDGET`; it must publish no output directory.

## 5. Isolated dual execution

The Python and Rust endpoints start concurrently in disjoint digest-pinned
containers. The host replay starts only after both immutable four-file output
sets exist. All three actors receive the same read-only Source U archive and
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
   Evidence T/structural guards;
3. normalize only implementation identity and the three Python-only isolation
   fields, then compare every common report field;
4. compare all framed stream bytes before decoding them;
5. decode every program and possible boundary witness through the shrink-5
   decoder, replaying indices, AST/hash identity, metadata, Q32 MDL, the three
   diagnostic bindings, uniqueness and global traversal order;
6. independently derive the terminal boundary from an empty committed
   shrink-5 generator, without treating either endpoint's reported boundary
   values or witness as authoritative;
7. replay program, chunk and all 150 bucket frames, including raw-counter
   partitions, zero buckets, blob hashes and RFC6962 roots; and
8. fail as `INCONCLUSIVE_DIAGNOSTIC` on any mismatch, never selecting one
   endpoint over the other.

The boundary derivation is
`INDEPENDENT_OF_ENDPOINT_REPORTED_WITNESS_NOT_A_THIRD_IMPLEMENTATION`: it
reuses the committed Python generator and is not a third enumerator.

## 7. Evidence V state machine

Only Evidence V may record an observed terminal result and route later work:

- dual `DSL_TOO_LARGE` may open the already-frozen shrink-order step 6,
  `reduce max_total_ast_depth from 4 to 3`;
- dual `COMPLETE` may open formal child-root and implementation
  requalification; and
- mismatch, raw-cap exit, partial output, schema/module drift, overwrite, or
  isolation failure remains inconclusive and opens no shrink or formal route.

These are conditional transitions, not claims that any outcome has occurred.
Nothing in Source U or Evidence V alone starts M3, evaluates odd/sink roles,
issues `OUTSIDE_FROZEN_CLOSURE(...)` or an MDL certificate, creates a split
seed, signs a formal object, authorizes M4, or changes ACTIVE governance.
