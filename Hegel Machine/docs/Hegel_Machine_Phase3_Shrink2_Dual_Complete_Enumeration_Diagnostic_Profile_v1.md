# Hegel Machine Phase-3 shrink-2 dual complete-enumeration diagnostic profile v1

Status: **FROZEN FOR A NON-FORMAL DUAL CHILD DIAGNOSTIC**

Machine profile ID: `hegel-m3-shrink2-dual-diagnostic-profile-v1`
Claim level: `NON_FORMAL_DUAL_CHILD_DIAGNOSTIC`

## 1. Purpose and state boundary

This profile preregisters a target-free Python/Rust replay for child DSL
`hegel-old-dsl-v1.2.0`. It may determine which shrink branch to engineer next.
It does not instantiate child formal roots, start a child M3 run, or amend the
already-published parent terminal.

Every endpoint and the host receipt must therefore carry:

```text
execution_state = NOT_RUN
formal_roots_generated = false
formal_roots = null
authoritative_claim_allowed = false
target_roles_evaluated = false
split_material_accessed = false
secrets_accessed = false
```

The implementation/protocol commit is Commit H. Complete Python and Rust runs
must start only from immutable snapshots of Commit H. Observed results, source
and binary digests, environment receipts, archive comparisons, and the Commit H
hash belong to the subsequent evidence Commit I.

## 2. Parent and child bindings

```text
parent terminal commit = db612e403bb46e6a295fed01e85649f8af0924b4
parent formal run ID   = e4af9f57c38fb298462ec628c4ed8a03
parent terminal status = DSL_TOO_LARGE
parent DSL             = hegel-old-dsl-v1.1.0
parent freeze          = hegel-freeze-p2b-p3-v1.1.2

child DSL              = hegel-old-dsl-v1.2.0
child freeze           = hegel-freeze-p2b-p3-v1.2.0
human amendment        = hegel-freeze-p2b-p3-v1.2.0-shrink-step2
shrink step            = SHRINK_STEP_2_REDUCE_RATIONAL_PARAMETER_TO_NEG1_ZERO_POS1
```

The tuple below is named `NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1`. Each value is
plain SHA-256 of the displayed UTF-8 byte string, where `\x00` is one NUL byte.
The first three roots are embedded in every diagnostic program record and are
the only bindings permitted by the two final-run CLIs.

| field | exact preimage | SHA-256 |
|---|---|---|
| `child_dsl_spec_root` | `HEGEL/M3/SHRINK2/DIAGNOSTIC_BINDING/V1\x00CHILD_DSL_SPEC\x00hegel-old-dsl-v1.2.0\x00hegel-freeze-p2b-p3-v1.2.0\x00shrink-step2` | `281c2f8adc41fdc467613b88c3c0caf3648efa883186bac61aabc2f8b575b3be` |
| `operator_semantics_root` | `HEGEL/M3/SHRINK2/DIAGNOSTIC_BINDING/V1\x00OPERATOR_SEMANTICS\x00hegel-old-dsl-v1.2.0\x00hegel-canonical-ast-v1\x00hegel-mdl-prefix-v1.0.0` | `7e09babc9dd91cfb8cab305f623957dab3f181cc70dacac9035cacf6d019d4bd` |
| `identifier_registry_root` | `HEGEL/M3/SHRINK2/DIAGNOSTIC_BINDING/V1\x00IDENTIFIER_REGISTRY\x00hegel-old-dsl-v1.2.0\x00aggregate-active:0,1,5\x00aggregate-tombstones:2,3,4\x00rational-active:1,3,5\x00rational-tombstones:0,2,4,6\x00rational-reserved:7` | `e6620b5f29151dda2a552425d19f53d5378d91641320d280701871eb5639e699` |
| `canonical_ast_schema_root` | `HEGEL/M3/SHRINK2/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_AST_SCHEMA\x00hegel-canonical-ast-v1\x00strict-numeric-tag-cbor-array` | `892bd6c958dd0dc300d30c13b5c7a2eaeedd36a0853a775097cc99aa3c2b544e` |
| `canonical_cbor_profile_root` | `HEGEL/M3/SHRINK2/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_CBOR_PROFILE\x00hegel-cbor-det-v1\x00RFC8949-deterministic-no-map-text-float-tag-indefinite` | `ab03d374143db9e64520a6f055439108e59ad13912544026789bb0d9558ebd32` |

These are diagnostic identity values, not content roots in the formal child
root DAG. Old v1.1 roots and these synthetic roots must not be inherited,
promoted, converted, or republished as v1.2 formal roots. Formal child
requalification requires newly constructed objects and dual root replay.

## 3. Frozen language surface and enumeration order

The inherited limits are depth `0..4`, node count `1..7`, output sort ID `1..5`,
50,000 canonical programs, 5,000,000 raw source attempts, 4,096 records per
chunk, and 175 sort-major accounting buckets.

- Active aggregate IDs: `0,1,5`; tombstones: `2,3,4`.
- Active rational parameter IDs: `1,3,5`; tombstones: `0,2,4,6`; reserved: `7`.
- Scope extensions: empty, then singles in context/boolean order, then pairs
  with increasing context IDs and boolean products.
- Pre-count aliases remain excluded: `greater_equal` and
  `approx_equal:tolerance=0`.
- One raw attempt is counted immediately before strict child canonicalization.
  Rewrite, duplicate, type, structural, and accepted outcomes are disjoint.
- Every accepted candidate is re-decoded with the shrink-2 formal decoder.

The program traversal key is exactly:

```text
(ast_depth, ast_node_count, output_sort_id,
 root_operator_id, canonical_ast_cbor_bytes)
```

Traversal is depth-major, node-major, sort-major. A bucket is generated and
sorted completely before a threshold decision. Commutative binary/ternary
children use `(SHA256(node_cbor), node_cbor)`; `AND` children use `node_cbor`.
Bucket records are instead sort-major with index
`(sort-1)*35 + depth*7 + (nodes-1)` and include all zero buckets.

## 4. Terminal rules

`DSL_TOO_LARGE` is diagnostic-only and valid only if:

1. archive indices are exactly `0..49,999`;
2. the containing traversal bucket is fully generated and sorted;
3. ordinal 50,001 is the adjacent next item under the frozen global key;
4. the witness is absent from the archive and binds both strict AST CBOR and
   `SHA256(AST_CBOR)`;
5. raw count includes the complete witness bucket; and
6. all three archive roots and all accounting checks replay.

`COMPLETE` requires every typed frontier bucket to close and carries no
witness. If the raw cap is reached before either condition, both endpoints
must exit non-zero with `INCONCLUSIVE_BUDGET` and must publish no report or
archive directory. No inconclusive path permits another shrink.

## 5. Isolated endpoints

Python is invoked directly, never with `-m` and never through the package
initializer:

```text
python3 -I -S -B \
  Hegel Machine/src/hegel_machine/phase3_m3_shrink2_isolated_entrypoint_v1.py \
  --enumerate-diagnostic \
  --child-dsl-spec-root 281c2f8adc41fdc467613b88c3c0caf3648efa883186bac61aabc2f8b575b3be \
  --operator-semantics-root 7e09babc9dd91cfb8cab305f623957dab3f181cc70dacac9035cacf6d019d4bd \
  --identifier-registry-root e6620b5f29151dda2a552425d19f53d5378d91641320d280701871eb5639e699 \
  --output-directory OUTPUT
```

It requires `-I -S -B`, creates a minimal package shell, and checks the same
exact eleven-module project allowlist before enumeration, after enumeration,
and before report publication. Any target, split, seed, role, sink, odd, or
evaluator module is fatal. The Commit-I run uses cached image
`python:3.10-slim@sha256:e5300dc020a26a34a19337a57602955a2510e22abeb176edd6de6cd2cc927dd4`
with `--network none`, `--pull never`, a read-only source snapshot, a read-only
root filesystem, dropped capabilities, and a fresh writable output mount.

Rust is built and tested with `cargo --locked --offline` inside the pinned
image `rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`,
with `--network none`, `--pull never`, a read-only source snapshot, and a fresh
target volume. Its CLI accepts the same three roots and no target-bearing
input. Both output directories and every file are create-new; replay against
an existing path must fail without changing existing bytes.

Host replay is also direct-only and must never import the package initializer:

```text
python3 -I -S -B \
  Hegel Machine/src/hegel_machine/phase3_m3_shrink2_dual_diagnostic_entrypoint_v1.py \
  --validate-dual \
  --python-output-directory PYTHON_OUTPUT \
  --rust-output-directory RUST_OUTPUT
```

The host freezes a twelve-module project allowlist: the endpoint's eleven
target-free modules plus the dual validator itself. Both the public validator
and its direct entrypoint check that exact closure before and after replay.
Calling the public validator from a process contaminated by
`hegel_machine.__init__`, a target, split, seed, role, sink, odd, or evaluator
module is fatal and cannot emit a replay receipt.

## 6. Dual comparison and host replay

Endpoint schema and implementation identity may differ. All other frozen
identity, limits, registry, status, count, root, witness, and claim-boundary
fields must agree. The Rust report has exactly the 59 canonical fields frozen in
the host validator. The Python report has those same fields plus exactly
`loaded_hegel_modules`, `target_free_isolation_verified`, and
`target_or_split_modules_loaded`. Missing or unknown fields are fatal. The nine
legacy Python aliases for budgets, roots, ordinal, and diagnostic binding roots
are not accepted, and non-finite JSON constants are forbidden.

The host derives the terminal boundary from an empty generator state before it
consults either reported witness. It traverses every frozen typed bucket in
depth/node/sort order, closes and sorts each bucket, and independently derives
the first cumulative rank 50,001. The observed archive must equal the exact
derived prefix and the observed witness must equal the derived rank-50,001
program. This rule also covers a witness that is the first item of a new bucket;
no same-bucket or byte-last-position shortcut is permitted.

In addition, Commit I must:

1. compare all three `.cborframed` streams byte for byte;
2. decode all 50,000 program records and every witness with the shrink-2
   decoder;
3. replay MDL Q32 lengths, program indices, three embedded roots, RFC6962
   roots, chunk ranges/subtree roots/blob hashes, and all 175 bucket rows;
4. prove witness rank using the independently reconstructed closed typed bucket,
   not merely `witness_key > last_key` or a CBOR byte-successor pattern;
5. bind Commit H, source/dependency manifests, Python identity, Rust binary
   digest, OCI image digest, commands, exit codes, stdout/report hashes, and
   archive file hashes; and
6. emit a receipt that remains `NOT_RUN`, `formal_roots=null`, unsigned, and
   non-authoritative.

## 7. Routing after Commit I

- Dual `DSL_TOO_LARGE`: the preregistered shrink-3 engineering branch may
  start; no formal child terminal is asserted.
- Dual `COMPLETE`: prepare child formal-root and implementation
  requalification; do not begin role evaluation directly.
- Any mismatch, raw-cap exit, partial archive, module drift, overwrite attempt,
  or host replay failure: `INCONCLUSIVE_DIAGNOSTIC`; do not shrink.

Nothing in this profile authorizes target evaluation, outside-language or sink
claims, a certificate, signatures, M4, ACTIVE governance, or Phase-3 exit.
