# Hegel Machine Phase-3 shrink-6 dual complete-enumeration diagnostic profile v1

Status: **SOURCE Y ENGINEERING FREEZE; NON-FORMAL AND NOT RUN**

Machine profile ID: `hegel-m3-shrink6-dual-diagnostic-profile-v1`

Claim level: `NON_FORMAL_DUAL_CHILD_DIAGNOSTIC`

## 1. Purpose and authority boundary

This profile preregisters a target-free Python/Rust complete-enumeration
diagnostic for `hegel-old-dsl-v1.6.0`. The only child-language delta from
v1.5.0 is the total canonical-AST depth limit:

```text
maximum_ast_depth: 4 -> 3
```

No registry ID, operator meaning, type rule, AST/CBOR identity, MDL rule,
budget, traversal key, or equivalence relation changes. The inherited AND2
limit remains in force, so the existing AND3 exclusion fields remain part of
the byte-stable 75-field report schema. The child enumerator admits no
depth-four program and records:

```text
maximum_top_level_clauses          = 2
maximum_ast_node_count             = 6
maximum_ast_depth                  = 3
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

Source Y contains protocol and implementation only. It contains no observed
shrink-6 enumeration count, raw count, witness, program/archive root, chunk
root, bucket root, output archive digest, runtime result, or terminal verdict.
Python, Rust and the host replay may execute only from an immutable
`git archive` of the later committed Source Y. A result belongs only to the
separate Evidence Z commit.

## 2. Unique engineering admission

Source Y has exactly one engineering admission: Evidence X and its canonical
qualification report.

```text
strict_qualification_source_commit
  = a69bf6d9746e302a07019f122047ac0bc74aa1c1
strict_qualification_evidence_commit
  = f9218e28740953c9ac15a2ada70a8616e92c378b
strict_qualification_artifact_path
  = Hegel Machine/artifacts/phase3_m3_runtime/phase3_shrink6_sealed_dual_strict_qualification_v1.json
strict_qualification_artifact_sha256
  = d5417639c651ea5d8dfbc224c79b0af56f1eb9d8705ee244f19dc9d95e6f2d08
strict_qualification_diagnostic_report_hash
  = sha256:3d2a6f06daa47b34aa56ae0d318cc818ba211859063d7a6b81271bc6bf1f8287
strict_qualification_status
  = SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS
```

The Source Y commit must be a single-parent child of Evidence X. Its source
manifest must include the byte-identical artifact, and the supervisor replays
the Source-W validator over those bytes before creating any container. The six
fields above are required unchanged in both endpoint reports and the host
receipt. Evidence X remains non-formal and leaves M3 at `NOT_RUN`.

## 3. Frozen non-formal bindings

Each value is plain SHA-256 of the corresponding frozen UTF-8 preimage in
`phase3_m3_shrink6_diagnostic_profile_v1.py`. They are diagnostic bindings,
not formal roots.

| field | SHA-256 hex |
|---|---|
| `child_dsl_spec_root` | `da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae` |
| `operator_semantics_root` | `922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03` |
| `identifier_registry_root` | `64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1` |
| `canonical_ast_schema_root` | `5de72fc51e27e5501561ffda6b05522f4941d1a13c4b324f5edcc15fa584a0bd` |
| `canonical_cbor_profile_root` | `ef0008912962de9da322eaeea6e421e1e58d16be152f968298774af0fd3249ab` |

The first three values bind every diagnostic program record. All five must
remain byte-stable with the Evidence X qualification. They may not be
converted, signed, or reused as formal child roots.

## 4. Budgets, traversal and exact report schemas

The frozen budgets remain 50,000 canonical programs before extensional
quotient, 5,000,000 raw operator applications, 4,096 program records per
chunk, depth `0..3`, node count `1..6`, output sort IDs `1..5`, and 120
sort-major accounting buckets. These are preregistered limits, not observed
results.

The traversal key is exactly:

```text
(ast_depth, ast_node_count, output_sort_id,
 root_operator_id, canonical_ast_cbor_bytes)
```

Buckets are completely generated and sorted before a threshold decision.
Program records are traversal-major. Bucket rows are sort-major, include all
zero buckets, and use index `(sort-1)*24 + depth*6 + (nodes-1)`. Depth-four
programs are outside this child lattice; no depth-four counter field is added.

The Rust report has exactly 75 common fields. The Python report has those 75
plus exactly `loaded_hegel_modules`, `target_free_isolation_verified`, and
`target_or_split_modules_loaded`, for 78 fields. Relative to shrink-5, the
schema keys are unchanged; the depth limit, bucket count and Evidence X binding
values change. All nested report and receipt objects use exact
key sets. Duplicate JSON names, unknown or missing fields, non-finite numbers,
wrong nullability, and legacy aliases are fatal.

`DSL_TOO_LARGE` is reserved for both exact frozen budgets. A reduced diagnostic
budget may emit only `DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED` and has no routing
authority. Reaching the raw cap before a closed decision is
`INCONCLUSIVE_BUDGET`; it must publish no output directory.

### 4.1 Preregistered prefix-preservation expectation

Shrink step 6 removes only depth-four programs. The previous rank-50,001
boundary lies at depth two, so Source Y preregisters the following expectation
before any shrink-6 complete diagnostic is executed:

```text
expectation_id = SHRINK6_PRESERVE_SHRINK5_PREFIX_THROUGH_CLOSED_BOUNDARY_BUCKET_V1
canonical_program_count = 50000
first_out_of_budget_program_ordinal = 50001
raw_operator_application_count = 3120719
residual_out_of_budget_canonical_programs = 2237
witness_output_sort_id = 3
witness_ast_depth = 2
witness_ast_node_count = 4
witness_bucket_index = 63
witness_cbor = 820183010384020183000001860003050200818203f5
witness_hash = 31320fc9f8926792aaf1416a4963df46a2300d87db8096f42e574a62272a68ee
```

These values are **expected**, not observed shrink-6 results. A dual execution
must derive them independently. Any difference terminates as
`INCONCLUSIVE_PRESERVATION_MISMATCH`; it may not be rewritten as
`DSL_TOO_LARGE`, `COMPLETE`, or a passing Evidence-Z result.

The expected AST prefix and witness bytes are stable because strict AST
identity is unchanged below depth four. Diagnostic archive identity is not:

- every program record binds the new v1.6 child, operator, and registry roots;
- every chunk manifest recursively binds the changed program records; and
- the bucket lattice has 120 rows, with sort-major indices based on 24 rows per
  sort rather than 30.

Therefore the program archive root, chunk-manifest root, bucket-accounting
root, all three stream hashes, endpoint report hashes, and retained archive
hash must be regenerated. Source Y freezes none of those future values and
must not copy any shrink-5 root into an output guard.

## 5. Isolated dual execution

The Python and Rust endpoints start concurrently in disjoint digest-pinned
containers. The host replay starts only after both immutable four-file output
sets exist. All three actors receive the same read-only Source Y archive and
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
- a build-only `--pids-limit=256`, permitting at most eight Cargo workers with
  one release codegen unit, while every Python, Rust and host-replay runtime
  actor retains `--pids-limit=64`; and
- an external, initially absent result directory.

The Rust build mounts only the exact Cargo `cache` and `index` file set. Each
regular file is bound by ordered path, mode, size and SHA-256. Every `.crate`
is checked against the committed `Cargo.lock`; pre-unpacked `src` is excluded.
The two subtrees are copied into a fresh 64 MiB tmpfs `CARGO_HOME`, Cargo runs
with `--release --locked --offline`, and the exact manifest is replayed before
and after the build. The cap-dropped build explicitly does not copy host UID/GID
ownership; bytes, paths, modes and the frozen manifest remain authoritative. A
changed dependency byte or mode is fail-closed. The build-only PID increase is
an engineering resource allowance, not a relaxation of runtime isolation or an
authority to contact the network.

## 6. Dual replay obligations

The host must:

1. reject any output set other than the exact report plus three framed streams,
   and require each captured endpoint stdout object to equal its retained
   `report.json` object;
2. validate recursive exact schemas, raw-count lower bounds and all fixed
   Evidence X/structural guards;
3. normalize only implementation identity and the three Python-only isolation
   fields, then compare every common report field;
4. compare all framed stream bytes before decoding them;
5. decode every program and possible boundary witness through the shrink-6
   decoder, replaying indices, AST/hash identity, metadata, Q32 MDL, the three
   diagnostic bindings, uniqueness and global traversal order;
6. independently derive the terminal boundary from an empty committed
   shrink-6 generator, without treating either endpoint's reported boundary
   values or witness as authoritative;
7. replay program, chunk and all 120 bucket frames, including raw-counter
   partitions, zero buckets, blob hashes and RFC6962 roots; and
8. check the preregistered rank-50,001 preservation tuple only after deriving
   it from the empty shrink-6 traversal, and fail as
   `INCONCLUSIVE_PRESERVATION_MISMATCH` on any difference; and
9. fail as `INCONCLUSIVE_DIAGNOSTIC` on every other mismatch, never selecting
   one endpoint over the other.

The boundary derivation is
`INDEPENDENT_OF_ENDPOINT_REPORTED_WITNESS_NOT_A_THIRD_IMPLEMENTATION`: it
reuses the committed Python generator and is not a third enumerator.

## 7. Evidence Z state machine and exhausted shrink order

Source Y preregisters exactly six shrink-order steps and has consumed through
step 6:

```text
preregistered_shrink_order_total_steps = 6
preregistered_shrink_order_consumed_through_step = 6
next_preregistered_shrink_step_or_null = null
```

Only Evidence Z may record an observed terminal result. Its routes are exact:

- dual `COMPLETE` uses
  `FORMAL_CHILD_ROOT_AND_IMPLEMENTATION_REQUALIFICATION_ELIGIBLE_NOT_STARTED`;
  this is eligibility only and does not start M3;
- dual `DSL_TOO_LARGE` uses
  `HALT_NO_PREREGISTERED_SHRINK_REMAINING_NEEDS_NEW_NORMATIVE_DECISION`;
  it authorizes no budget change, additional shrink, `shrink7`, v1.7 DSL, or
  formal transition; and
- preservation mismatch, raw-cap exit, partial output, schema/module drift,
  overwrite, or isolation failure remains inconclusive and opens no route.

The valid `DSL_TOO_LARGE` result remains a terminal engineering diagnostic;
the fail-closed condition applies to continuation authority. Any later budget,
language, closure representation, or certificate change requires a new
explicit normative decision and freeze.

These are conditional transitions, not claims that any outcome has occurred.
Nothing in Source Y or Evidence Z alone starts M3, evaluates odd/sink roles,
issues `OUTSIDE_FROZEN_CLOSURE(...)` or an MDL certificate, creates a split
seed, signs a formal object, authorizes M4, or changes ACTIVE governance.
