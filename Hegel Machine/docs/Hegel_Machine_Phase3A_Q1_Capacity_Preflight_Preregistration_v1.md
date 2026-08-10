# Hegel Machine Phase-3A-Q1 Capacity Preflight Preregistration v1

## 0. Authority and current state

This document preregisters a **target-blind, non-formal capacity preflight** for
the full production quotient engine.  It is an engineering diagnostic protocol,
not the Q1 formal execution freeze and not an execution result.

Its phase position is **Q0.5 / Q1-admission preflight**: it measures whether the
Q1 construction is feasible while Q1 itself remains `NOT_RUN`.  Calling the
protocol “Q1 capacity preflight” does not mean that Q1 execution has started.

```yaml
preflight_id: hegel-phase3a-q1-capacity-preflight-v1
schema_version: hegel-phase3a-q1-capacity-preflight-preregistration/1
dsl_version: hegel-old-dsl-v1.6.0
dsl_freeze_version: hegel-freeze-p2b-p3-v1.6.0
closure_semantics_version: hegel-quotient-closure-v1.0.1

preflight_status: PREREGISTERED_NOT_RUN
q1_state: NOT_RUN
q1_execution_started: false
q1_gate_count: 0
q1_gate_mask: 0x000000
q1_formal_roots: null
q1_receipt: null
q2_state: NOT_RUN
role_evaluation_performed: false
m3_formal_roots: null
outside_certificate_issued: false
active_transition_allowed: false
```

The machine-readable companion is:

```text
Hegel Machine/config/phase3_q1_capacity_preflight_v1.json
```

The post-shrink-6 direction makes C3 exact quotient closure primary and permits
Path B only after a target-blind quotient-capacity failure under a separately
frozen resource contract.  Q0 has now qualified the micro mechanics at 14/14,
but its result explicitly leaves Q1 `NOT_RUN`.  This preregistration is the next
capacity-measurement step; it inherits no Q1 gate pass and creates no formal
closure object.

The historical Source-Y syntactic-capacity negative is preserved unchanged:
execution-source commit `5217568303d5c7f902682c092750f637c64f080a`,
Evidence-Z repository commit `ea98157f5d6eb2930ab28dda8f3a6839b343673c`, evidence record
`artifacts/phase3_shrink6_dual_complete_enumeration_diagnostic_v1.json`, artifact
SHA-256 `cb7e2d003382e4cfeb69ac69b7a3316b970510077c646ce63a9a8718e157d30b`.
Its authority remains `HISTORICAL_SYNTACTIC_CAPACITY_NEGATIVE_ONLY`; it passes
no Q1 gate and cannot be reinterpreted as quotient cardinality or closure.

---

## 1. Exact claim boundary

A successful preflight may say only:

```text
PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
```

This means that both target-blind production-signature quotient states reached
their exact fixed points within the provisional hard envelope in the particular
source-bound diagnostic run.  Even then, the following remain false:

```text
Q1 started
Q1 COMPLETE
Q1 formal roots generated
Q1 gate evidence issued
Q1 receipt issued
Q2 role evaluation performed
target truth or split opened
OUTSIDE_FROZEN_QUOTIENT_CLOSURE issued
M3 or ACTIVE advanced
```

The preflight may emit canonical diagnostic JSON and target-free growth curves.
It must not emit an object using a formal Q1 receipt tag, populate a Q1 output
slot, or call a diagnostic archive root a formal root.

---

## 2. Two independent production quotients

The preflight computes two separate quotient states:

```text
Q_odd  = P_G / ~_(OddInputV1, U_odd_480)
Q_sink = P_G / ~_(SinkInputV1, U_sink_85)
```

It does **not** concatenate the universes into one 565-row behavior vector and
does not merge a class across input signatures.  The input-only bindings are:

| Input signature | Rows | Frozen universe root | Truth root in preflight |
|---|---:|---|---|
| `OddInputV1` (`1`) | 480 | `sha256:b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05` | `null` |
| `SinkInputV1` (`2`) | 85 | `sha256:1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5` | `null` |

These are reused historical payload-universe identities, not roots generated
by this preflight.  Universe rows and their canonical ordering are allowed
target-independent inputs.  Target truth, target IDs, role bindings, and split
material are not.

---

## 3. Full v1.6 construction surface

The preflight is not the 15-leaf Q0 projection.  It must construct the complete
v1.6 leaf surface before quotient saturation:

| Leaf family | Count |
|---|---:|
| active rational constants (`RationalParameterId` `1,3,5`) | 3 |
| `bit_at(0..7)` | 8 |
| `set_size()` | 1 |
| aggregates | 792 |
| `context_flag(c0..c3)` | 4 |
| `task_flag(t0..t1)` | 2 |
| **Total** | **810** |

The 792 aggregate leaves are exactly:

```text
3 active maps       {0,1,5}
x 4 scopes          {0,1,2,3}
x 2 quantities      {0,1}
x 33 extensions     {1 empty, 8 one-clause, 24 two-clause}
= 792
```

Aggregate IDs `2,3,4` remain tombstones.  The canonical operator surface is:

```text
unary IDs:                 0,1,2,3
canonical binary IDs:      1,2,3,5,6
approx_equal tolerance:    1,2
top-level conjunction:     AND2 only
```

Source aliases, removed operators, tolerance zero, AND1, AND3, implicit
coercions, target-aware rewrites, and new symbolic reducers are excluded.

The strict structural limits are:

```yaml
maximum_ast_depth: 3
maximum_ast_node_count: 6
maximum_top_level_clauses: 2
maximum_distinct_bit_slots: 4
maximum_aggregate_leaves: 1
maximum_scope_clauses: 2
maximum_old_law_composition_depth: 2
maximum_scalar_parameter_occurrences: 3
```

The v1.6 program grammar contains no old-law-composition AST node, so every
realized value of that metric is exactly zero even though the historical bound
remains recorded as two.

---

## 4. Quotient construction rules

Behavior identity remains:

```text
(
  input_signature_id,
  frozen_universe_root,
  output_sort_id,
  exact ordered output cells with typed bottom
)
```

Bool, Bit, Sign, BoundedInt, RationalValue, and typed bottom remain distinct.
Symbolic normalization is a construction aid, never semantic class identity.

Each behavior class retains:

1. its complete exact behavior bytes;
2. its globally recomputed visible MDL-aware Pareto frontier;
3. its complete target-blind continuation bank, including dominated cohorts;
4. its exact minimum admitted Q32 MDL;
5. real admitted AST representatives and hashes.

The normalization-witness capacity is:

| Output sort | Representatives per exact-signature cohort |
|---|---:|
| Bool | 2 |
| RationalValue | 2 |
| Bit | 1 |
| Sign | 1 |
| BoundedInt | 1 |

Every new bank representative is expanded exactly once even if its cohort is
currently dominated.  Visible frontiers are recomputed from the entire bank.
Intermediate round mutation history is validated independently per endpoint but
is excluded from cross-endpoint semantic identity.

The preflight does not build or claim a complete syntactic program archive.
Its syntax-facing counts must be named diagnostic quotient-expansion candidate
counts, never full canonical syntax cardinality.

---

## 5. Local prototype observations are guidance only

A local, non-formal prototype exercised all 810 leaves, the two independent
universes, and full-bank expansion at partial node limits.  Its ordinary
package import path was not yet process-isolated from historical target
modules, so the observations are recorded only to make the provisional
resource decision auditable and are not target-blind qualification evidence:

| Node limit | Signature | Classes | Visible frontier | Bank | Raw applications |
|---:|---|---:|---:|---:|---:|
| 3 | Odd | 40 | 59 | 110 | 1,048 |
| 3 | Sink | 28 | 84 | 144 | 1,101 |
| 4 | Odd | 107 | 154 | 478 | 1,844 |
| 4 | Sink | 47 | 138 | 466 | 2,108 |

These rows are explicitly:

```text
LOCAL_UNISOLATED_NON_FORMAL_PROTOTYPE_GUIDANCE_ONLY
```

They are not full node-six saturation, are not source-bound formal evidence,
do not establish a production closure cardinality, and do not pass a Q1 gate.
No extrapolated value from this table is a frozen formal budget.

---

## 6. Provisional hard preflight envelope

Python and Rust endpoints run concurrently.  Within each endpoint, Odd and Sink
run sequentially in ascending input-signature order so that two signature states
cannot silently double one endpoint's memory envelope.

The exact provisional hard envelope is:

```yaml
per_endpoint:
  cpus: 12
  memory_bytes: 15032385536       # 14 GiB
  scratch_bytes: 51539607552      # 48 GiB
  output_bytes: 34359738368       # 32 GiB
  wall_time_seconds: 172800       # 48 hours

trusted_host_replay_after_endpoints_exit:
  memory_bytes: 15032385536       # 14 GiB
  scratch_bytes: 51539607552      # 48 GiB
  output_bytes: 34359738368       # 32 GiB
  wall_time_seconds: 172800       # 48 hours

per_input_signature:
  maximum_raw_operator_applications: 4294967295
  maximum_behavior_classes: 16777216
  maximum_visible_frontier_points: 33554432
  maximum_visible_frontier_points_per_class: 65536
  maximum_continuation_bank_points: 67108864
  maximum_continuation_bank_points_per_class: 65536
  maximum_work_queue_points: 67108864
  maximum_saturation_rounds: 16
```

These values are provisional ceilings for capacity observation, not Q1 formal
budgets.  A value equal to a ceiling is admitted; the next event that would
exceed it terminates before acceptance.

The container profile is:

```text
two endpoint containers in parallel
--pull=never
--network=none
read-only root and source snapshot
UID/GID 65534:65534
all capabilities dropped
no-new-privileges
sealed offline Cargo dependency closure
no endpoint-output exchange
no dependency download or image pull
--memory-swap=15032385536 (equal to --memory; no extra swap allowance)
Python cpuset 0-11; Rust cpuset 12-23
--pids-limit=128
--ulimit nofile=256:256
--ipc=private
no Docker socket mount
runtime seccomp config/phase3_internal_actor_seccomp_v1.json
runtime seccomp sha256:d1284e4731683b73352ecdd1577704ea87aa0b5c582b7b00757c3db4d2c950ca
/tmp tmpfs: rw,noexec,nosuid,nodev,size=1g,mode=1777
```

The pinned local images are:

```text
Python  python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3
Rust    rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89
```

Rust uses `cargo build --release --locked --offline` against a sealed Cargo
home during offline build qualification.  Rust compilation is excluded from
runtime capacity measurements: the runtime binary must be built offline first,
then its source, dependencies, toolchain/image, flags, and exact binary bytes
must be identity-bound before it is executed.  Before execution the diagnostic
report must bind a non-null clean Git
source commit, distinct read-only endpoint source snapshots, and each endpoint's
complete implementation root.  The preregistration cannot pre-populate those
run-produced identities, so `preflight_source_commit` remains `null` until a
candidate implementation has been committed.

The host must replay target-blind state after both endpoints terminate.  It is
a trusted replay process under the same administrative controller, not a third
independent endpoint and not an external signer.  It nevertheless runs in a
fresh third target-blind, no-network, read-only Docker container after the two
endpoints exit, with the same 14-GiB memory/swap cap and cpuset `0-11`; this is
hard filesystem/resource isolation, not organizational independence.
Gate 17 depends on that replay, so Q0.5a must freeze its streaming/replay
algorithm, report wire, allocator and merge schedule.  The authoritative
preflight must execute and measure it using cgroup-v2 `memory.peak`, including
descendants.  Its elapsed interval begins immediately before Docker start and
ends at container exit, including language import, replay and projection
encoding.  It inherits the endpoint hardening profile (no network or pull,
read-only root and exact source snapshot, unprivileged UID/GID, dropped
capabilities, no-new-privileges, seccomp, private IPC, fixed pids/nofile/tmpfs,
no Docker socket); both endpoint artifact mounts are read-only and only its
frozen scratch mount is writable.  Host wall, projected output and projected
scratch receive their own mechanically derived envelope under the ceilings
above; they may not be granted an unrecorded post-result budget.

At this source-freeze stage, only the import-isolated local subset endpoint is
admitted.  A no-argument node-six run must fail closed: the Q1 archive/resource
projection wire, Rust endpoint, dual supervisor, commit-bound execution
manifest, and filesystem-isolated snapshots are not yet qualified.

The machine-readable config's `diagnostic_wire_contract` freezes the exact
engine, partition, limit, and depth-barrier JSON field sets.  Engine schema is
`hegel-phase3a-q1-capacity-preflight/1`; the current Python wrapper schema is
`hegel-phase3a-q1-python-capacity-preflight-endpoint/1`.  JSON is ASCII-escaped
UTF-8 with sorted keys, separators `,` and `:`, no duplicate keys, no NaN or
Infinity, and exactly one trailing LF whose byte is included in the output
guard.  This remains a diagnostic JSON identity, never a formal CBOR tag/root.
Q0.5a archive-projection or supervisor fields must use a new schema version;
they may not be appended silently to either current `/1` diagnostic schema.

---

## 7. Mechanical derivation of the later formal envelope

The formal Q1 envelope is intentionally **not** chosen in this document.  Before
an authoritative full preflight, Q0.5a must freeze the target-blind Q1 archive
schemas/tags, field order, CBOR profile, chunk boundaries, manifest/root DAG,
streaming or external-sort algorithm, scratch allocator/merge schedule, and
resource-projection golden vectors.  Seeing node-six capacity first and then
choosing a more convenient wire is forbidden.

If and only if a later source-bound dual preflight saturates both signatures
without a guard hit under that already-frozen projection profile, the formal
envelope amendment computes its values mechanically.

Let `pow2ceil(x)` be the least positive power of two greater than or equal to
`x`.  For every per-signature deterministic count:

```text
formal_count_guard =
  pow2ceil(max(1, 2 * aggregated_accepted_state_saturated_high_water_value))
```

This applies separately to:

- the accepted-state high-water of raw operator applications;
- the accepted-state high-water of behavior classes;
- accepted-state high-water values for total and per-class visible frontier
  points;
- accepted-state high-water values for total and per-class continuation-bank
  points;
- accepted-state maximum work-queue points.

The partition diagnostic records these guard observations explicitly as
`peak_raw_operator_application_count`, `peak_behavior_class_count`,
`peak_visible_frontier_point_count`,
`peak_visible_frontier_points_per_class`,
`peak_continuation_bank_point_count`,
`peak_continuation_bank_points_per_class`, `peak_work_queue_points`, and
`peak_saturation_round_count`.  They update only after an event or barrier is
accepted; a rolled-back event cannot raise a reported peak.

Class/frontier/bank and per-class values must first be byte-exactly equal across
Python and Rust and then use the common per-signature value.  The exact eligible
operator-application set is part of the frozen traversal and therefore its raw
count must also be identical.  A work-queue high-water may differ under
separately frozen legal schedules; it uses the larger endpoint value per
signature.  The round observation is the maximum across both endpoints and
both input signatures.  A final frontier or cache count may be lower than its
run-time high-water and must never be substituted for the corresponding guard
observation.

Other guards are:

```text
formal_round_guard = observed_saturation_rounds + 2

formal_wall_guard =
  3600 * ceil_div(
    3 * slower_endpoint_elapsed_monotonic_ns_including_projection_encoder,
    3600 * 1000000000)

formal_memory_guard =
  pow2ceil(max(1, 2 * maximum_endpoint_peak_rss_bytes))

projected_formal_total_output_bytes_per_endpoint =
  projected_formal_archive_payload_bytes_per_endpoint
  + deterministic_worst_case_run_metadata_overhead_bytes_per_endpoint

formal_output_guard =
  pow2ceil(max(1, 2 * projected_formal_total_output_bytes_per_endpoint))

formal_scratch_guard =
  pow2ceil(max(1, 2 * projected_peak_formal_scratch_bytes_per_endpoint))

host_replay_wall_guard =
  3600 * ceil_div(
    3 * host_replay_elapsed_monotonic_ns,
    3600 * 1000000000)

host_replay_memory_guard =
  pow2ceil(max(1, 2 * host_replay_peak_rss_bytes))

host_replay_output_guard =
  pow2ceil(max(1, 2 * projected_host_replay_output_bytes))

host_replay_scratch_guard =
  pow2ceil(max(1, 2 * projected_peak_host_replay_scratch_bytes))
```

The Python and Rust formal archive payload bytes must be identical.  The common
endpoint output guard uses that payload plus the larger role-specific bounded
metadata overhead; endpoint wall, RSS and projected scratch use the worse of
the two endpoints.  Host replay is measured and derived separately because it
uses a different trusted algorithm and produces replay evidence rather than a
third closure archive.

Four byte quantities are deliberately distinct:

1. `actual_preflight_diagnostic_output_bytes`, including the exact trailing LF
   and declared sidecars, guards the diagnostic run only;
2. `actual_preflight_peak_scratch_bytes`, the atomic logical allocated-byte
   high-water under its frozen allocator, guards the diagnostic run only;
3. `projected_formal_archive_payload_bytes`, obtained by exact counting/discard
   encoding under the pre-run frozen formal wire, excludes run metadata;
4. `projected_peak_formal_scratch_bytes`, obtained from the frozen allocator and
   merge schedule, budgets the later formal run.

Only items 3 and 4, plus deterministic worst-case overhead for run IDs,
timestamps, signatures, receipts, and other run-produced fixed-width/ranged
metadata, derive the endpoint formal output/scratch guards.  The analogous
frozen counting/discard and scratch-high-water measurements for trusted host
replay derive its separate envelope.  A small diagnostic JSON or zero
diagnostic scratch can never be substituted for them.  Any wire,
chunking, allocator, or merge-schedule change creates a new version, invalidates
the projection, and requires the whole preflight to be rerun.

Every derived value must remain within the provisional ceiling in Section 6.
If any derived value exceeds its ceiling, the result is:

```text
PREFLIGHT_RESOURCE_ENVELOPE_INFEASIBLE
```

Manual rounding, selecting a more favorable endpoint, changing the multiplier,
raising only the guard that failed, or redrawing a resource profile after seeing
the result is forbidden.  A successful preflight therefore determines the next
formal envelope without target-conditioned or operator-specific discretion.

---

## 8. Fixed-point condition for a saturated diagnostic

Each signature must independently satisfy:

```text
work_queue_empty == true
zero_delta_full_round == true
all_eligible_operator_and_bank_tuple_expansions_covered == true
final_new_behavior_class_delta == 0
final_visible_frontier_delta == 0
final_continuation_bank_delta == 0
no_resource_guard_hit == true
```

The last admitted construction depth may legitimately add classes or bank
points.  Completion therefore appends a separate
`STRUCTURAL_BOUNDARY` record at depth four: because the frozen grammar admits
only depth at most three, its eligible queue and every delta are exactly zero.
The zero-delta requirement refers to this terminal boundary witness, while
coverage requires every eligible depth-zero through depth-three leaf/operator/
bank tuple to have been processed exactly once.

Python and Rust must independently construct the states.  A common diagnostic
status additionally requires byte-identical final class, visible-frontier,
continuation-bank, coverage, and fixed-point preimages after canonical ordering.
Intermediate round histories may differ legally and do not enter that equality.

Reaching these conditions produces only
`PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY`.  The formal run must later replay the
work from a frozen Q1 genesis and a separately frozen resource envelope.

---

## 9. Resource guard registry

| ID | Guard |
|---:|---|
| 1 | `RAW_OPERATOR_APPLICATIONS` |
| 2 | `BEHAVIOR_CLASSES` |
| 3 | `VISIBLE_FRONTIER_TOTAL` |
| 4 | `VISIBLE_FRONTIER_PER_CLASS` |
| 5 | `CONTINUATION_BANK_TOTAL` |
| 6 | `CONTINUATION_BANK_PER_CLASS` |
| 7 | `WORK_QUEUE_POINTS` |
| 8 | `SATURATION_ROUNDS` |
| 9 | `OUTPUT_BYTES` |
| 10 | `SCRATCH_BYTES` |
| 11 | `RESIDENT_MEMORY` |
| 12 | `WALL_TIME` |

A clean guard hit must name exactly one first guard ID and produce no fabricated
fixed-point claim.  Partial diagnostic material may be retained but cannot be
inserted into a formal Q1 archive or receipt.

Both numeric `resource_guard_id` and its registered `resource_guard_name` are
mandatory (or both are `null`).  Internal events are prospective and atomic:
the event that would exceed a ceiling is rolled back.  If one prospective event
would violate more than one guard, the smallest numeric guard ID wins.  Runtime
RSS/wall/output/scratch guards use the exact measurement contract in the config;
their violation is `INCONCLUSIVE` and cannot authorize Path B.

---

## 10. Terminal routes

```text
PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
  -> mechanically derive and freeze formal Q1 resource envelope
  -> Q1 remains NOT_RUN and 0/20 until separate admission

PREFLIGHT_CAPACITY_GUARD_HIT
  -> INCONCLUSIVE; halt without Q1 start
  -> repair or reimplement the target-blind preflight and rerun
  -> Path B is not authorized by an ordinary guard hit

PREFLIGHT_RESOURCE_ENVELOPE_INFEASIBLE
  -> halt without Q1 start
  -> Path-B normative review becomes eligible only after the complete
     exactness/isolation/dual-agreement adjudication stated below

PREFLIGHT_SEMANTICS_FAILURE
  -> return to Q0 / closure-semantics review
  -> Path B is not authorized

PREFLIGHT_IMPLEMENTATION_DISAGREEMENT
  -> repair and requalify implementations
  -> Path B is not authorized

PREFLIGHT_ISOLATION_FAILURE
  -> repair isolation and rerun
  -> Path B is not authorized

PREFLIGHT_UNCONTROLLED_EXECUTION_FAILURE
  -> repair execution and rerun
  -> Path B is not authorized
```

No terminal route automatically changes the DSL, enters Path B, opens target
truth, or issues a certificate.  Any Path-B action still requires a new
normative amendment.

Path-B review becomes eligible only if all of the following are established
together: the C3 exactness obligations have passed; truth/split/role inputs
remained unopened; Python, Rust, and host replay agree on the target-blind
diagnostic preimages; implementation inefficiency and uncontrolled execution
failure have been excluded; and the mechanically derived formal envelope
exceeds the preregistered hard ceiling.  A single guard hit, timeout, memory
spike, queue implementation choice, or incomplete prefix is only
`INCONCLUSIVE` and provides no Path-B authority.

This preflight cannot select Path D.  D is the fallback if C3 exactness or
completeness cannot be certified, or—only where the capacity preconditions made
Path B separately applicable and a new amendment authorized it—if Path B still
cannot certify them.  Path B is not a mandatory detour when its preconditions
do not hold.  Entering D always requires another normative amendment that
narrows the claim to a publishable negative result.

---

## 11. Planned 20-gate Q1 registry

The following registry is preregistered so the capacity run cannot invent a
more convenient readiness definition afterward:

| # | Exact gate name |
|---:|---|
| 1 | `POST_SHRINK6_NORMATIVE_AND_Q1_FREEZE_BYTES_BOUND` |
| 2 | `Q0_14_OF_14_RECEIPT_AND_SOURCE_REPLAYED` |
| 3 | `V16_FULL_DSL_STRICT_ROOT_FAMILY_REQUALIFIED` |
| 4 | `PRODUCTION_UNIVERSE_ROWS_AND_ADAPTERS_DUAL_QUALIFIED` |
| 5 | `Q1_BEHAVIOR_BOTTOM_AND_COLLISION_CODEC_QUALIFIED` |
| 6 | `Q1_CONSTRUCTION_SIGNATURE_AND_MULTIPLICITY_POLICY_QUALIFIED` |
| 7 | `Q1_PARETO_DOMINANCE_AND_CLASS_MDL_QUALIFIED` |
| 8 | `FULL_OPERATOR_CONGRUENCE_AND_REWRITE_PROFILES_QUALIFIED` |
| 9 | `STRUCTURAL_INDUCTION_AND_FIXED_POINT_CONTRACT_QUALIFIED` |
| 10 | `Q1_ARCHIVE_WIRE_ROOT_DAG_RESOURCE_PROJECTION_AND_GOLDENS_PASS` |
| 11 | `TARGET_BLIND_CAPACITY_PREFLIGHT_AND_RESOURCE_ENVELOPE_FROZEN` |
| 12 | `PYTHON_RUST_IMPLEMENTATION_AND_OFFLINE_ISOLATION_BOUND` |
| 13 | `TARGET_TRUTH_SPLIT_ROLE_INPUTS_ABSENT` |
| 14 | `Q1_RUN_GENESIS_NON_NULL_AND_EIGHT_OUTPUT_SLOTS_NULL` |
| 15 | `BOTH_SIGNATURE_FIXED_POINTS_COMPLETE_NO_GUARD_HIT` |
| 16 | `PYTHON_RUST_SIGNATURE_ARCHIVES_AND_STATE_PREIMAGES_BYTE_EQUAL` |
| 17 | `HOST_REPLAY_DUAL_SIGNATURE_CLOSURES_PASS` |
| 18 | `TARGET_BLIND_ACCESS_LEDGER_AND_ISOLATION_POSTCHECK_PASS` |
| 19 | `Q1_CLOSURE_BUNDLE_AND_COMPLETION_RECEIPT_SEALED` |
| 20 | `Q1_COMPLETE_Q2_M3_ROLE_CERTIFICATE_OUTPUTS_NULL_NOT_RUN` |

This preflight passes **none** of them and emits no gate rows:

```text
q1_gate_count = 0
q1_gate_mask = 0x000000
```

After a successful preflight, the formal resource amendment and remaining Q1
engineering qualification must still establish Gates 1–14.  Only 14/20 may
authorize an explicit `phase3-q1-start`; only 20/20 may produce Q1 `COMPLETE`.

---

## 12. Strictly forbidden inputs and actions

The source snapshots, mounts, imports, and branch predicates must exclude:

- Odd and Sink truth rows or truth roots;
- split seed, commitments, assignments, partition roots, or membership;
- target/control role-binding manifests;
- `BOOL_BIT_EXACT_PREDICATE_MATCH_V1`;
- match sets and role receipts;
- synthesis traces and target-conditioned scores;
- hidden-outcome-dependent budget, grammar, or operator selection.

The preflight also cannot:

- generate a new split seed;
- alter either universe membership or ordering;
- concatenate the two signatures into one quotient identity;
- use the old 50,000 syntactic budget as quotient closure cardinality;
- truncate on a guard and call the prefix complete;
- write formal roots into an M3/Q1 slot;
- sign or issue an outside/MDL certificate;
- promote governance or ACTIVE state.

---

## 13. Authorized successors and ordering

The immediate successor to this preregistration is **Q0.5a archive/resource
projection freeze**, not the full preflight.  It must freeze and dual-qualify
the archive wire/root DAG/chunking/streaming/scratch profile described in
Section 7, freeze the trusted host replay algorithm and its separate resource
projection, implement the independent Rust endpoint and dual supervisor, bind
filesystem-isolated source snapshots, and keep Q1 at `0/20 / NOT_RUN`.

Only then may Q0.5b run the full source-bound dual capacity preflight.  The only
successor to its `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY` result is a byte-exact
formal-resource amendment that:

1. binds the preflight source and diagnostic result;
2. applies the formulas in Section 7 without discretion;
3. binds the already-frozen Q1 wire/projection identity and freezes the run
   genesis and final implementation identities;
4. leaves every run-produced output root null;
5. qualifies the first fourteen Q1 gates without starting Q1.

Formal Q1 execution then requires a separate explicit `phase3-q1-start` action.
Q2 remains downstream of a sealed Q1 `COMPLETE` result.
