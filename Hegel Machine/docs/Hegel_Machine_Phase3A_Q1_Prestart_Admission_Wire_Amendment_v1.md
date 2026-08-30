# Hegel Machine Phase-3A Q1 prestart admission wire amendment v1

Status: **candidate engineering freeze; not yet owner-authorized**.

This amendment defines the control-plane objects needed to move Q1 from
`NOT_RUN / 0 of 20` to `NOT_RUN / 14 of 20`. It does not start Q1, invoke an
endpoint, read target/split/role inputs, populate a formal output slot, or
authorize an `ACTIVE` transition. A later `phase3-q1-start` action requires a
separate exact capsule and owner authorization.

## Separate control-plane namespace

The frozen formal archive registry remains exactly `0x3700..0x370C`. The
prestart control plane uses a disjoint registry:

| Tag | Object | Schema | Root domain |
|---:|---|---|---|
| `0x3B00` | `Q1RunGenesisV1` | `hegel-q1-run-genesis/1` | `HEGEL/Q1/RUN_GENESIS/V1` |
| `0x3B01` | `Q1PrestartAdmissionReceiptV1` | `hegel-q1-prestart-admission-receipt/1` | `HEGEL/Q1/PRESTART_ADMISSION_RECEIPT/V1` |

The tag registry root uses `HEGEL/Q1/PRESTART_TAG_REGISTRY/V1`. The complete
20-gate registry root uses `HEGEL/Q1/GATE_REGISTRY/V1` and binds both the gate
names and the exact positional subject schema for Gates 1 through 14.

## Final actor identities and archive layout

Gate 12 accepts only these final actor roles, in order:

1. `hegel-python-q1-formal-archive-materializer-v1`;
2. `hegel-rust-q1-formal-archive-materializer-v1`;
3. `hegel-host-q1-formal-archive-replay-v1`.

Historical `diagnostic_only` actor identities are not aliases for these roles.
The final implementation roots are frozen only after the corresponding source,
runtime, entrypoint, and offline-isolation closures exist.

Each actor root uses `HEGEL/Q1/FORMAL_ACTOR_IMPLEMENTATION/V1` over its exact
role, implementation ID, action ID, source-manifest root, entrypoint raw SHA-256,
dependency-closure root, runtime-identity root, offline-isolation root, the
common output-layout root, an empty forbidden-input-key registry, and exact
false target/split/role flags. Those digests change Gate 12 admission and are
therefore operational identities, not decorative checksums.

Both materializers produce the same deterministic relative layout:

```text
signature-01/
  stream-01/chunk-000000.framed ...
  stream-02/chunk-000000.framed ...
  stream-03/chunk-000000.framed ...
  stream-04/chunk-000000.framed ...
  chunk-manifests.framed
  fixed-point.framed
  signature-manifest.framed
signature-02/
  ...same registry...
closure-bundle.framed
endpoint-result.json
```

Every `.framed` file is the existing four-byte big-endian length followed by
the existing strict-CBOR payload. Chunk names and counts are derived from the
frozen stream descriptors and manifests. No new archive tag, record type,
chunking rule, compression rule, allocator, or merge schedule is introduced.

## `Q1RunGenesisV1`

The strict array field order is:

```text
[
  1, 0x3B00, b"hegel-q1-run-genesis/1",
  run_id_16_bytes,
  source_identity,
  version_binding_rows,
  prestart_tag_registry_root,
  complete_q1_20_gate_registry_root,
  q1_semantic_binding_root,
  q1_projection_profile_root,
  q0_14_of_14_receipt_root,
  q05b_qualification_receipt_root,
  full_v16_leaf_manifest_root,
  formal_resource_envelope,
  final_implementation_root_rows,
  prestart_authority_object,
  created_at_unix_seconds
]
```

`source_identity` is
`[1, b"git-sha1-commit-tree-sha256-closure/1", commit20, tree20,
closure_sha256_32]`. The run ID and timestamp are explicit capsule-bound input
bytes; replay never reads a clock or generates randomness.

The resource envelope is the exact mechanically derived successor of the v6
saturated diagnostic: Odd guards `32768/1024/4096/32/8192/4096/16384/7`, Sink
guards `32768/256/1024/256/8192/4096/16384/7`, Python and Rust endpoint bounds
`16 MiB output / 4 MiB scratch / 2 GiB RSS / 3600 s`, and host bounds `2 MiB
output / 1 byte scratch / 64 MiB RSS / 3600 s`.

## Gate evidence

Each passed row is `(gate_id, gate_name, true, evidence_root32)`. Its root is:

```text
content_hash(
  "HEGEL/Q1/PRESTART_GATE_EVIDENCE/V1",
  (source_identity, complete_gate_registry_root,
   gate_id, gate_name, exact_gate_subjects)
)
```

The exact subject shapes are frozen in
`phase3_q1_prestart_wire_v1.Q1_PRESTART_GATE_SUBJECT_REGISTRY`. Important
boundaries are:

- Gate 10 replays the Q0.5b qualification against the final archive-wire source
  closure; the historical receipt alone is insufficient.
- Gate 11 binds the v6 capsule, terminal record, successful artifact/outcome,
  and the exact formal resource envelope.
- Gate 12 binds the three final actor roots, final source closure, offline
  runtime isolation, and output layout. Diagnostic actors cannot pass it.
- Gate 13 binds the final source closure, mount manifest, input-key-registry
  absence, and three exact `false` values for target truth, split, and role
  evaluation.
- Gate 14 binds the non-null genesis root, the complete embedded genesis
  object, all eight named null output slots, and `q1_start_authorized=false`.

The compact receipt carries evidence roots. Its enclosing capsule and final
artifact carry the exact subject rows; admission recomputes all fourteen roots
with `verify_prestart_gate_evidence_v1`.

## `Q1PrestartAdmissionReceiptV1`

The strict array field order is:

```text
[
  1, 0x3B01, b"hegel-q1-prestart-admission-receipt/1",
  source_identity,
  version_binding_rows,
  prestart_tag_registry_root,
  complete_q1_20_gate_registry_root,
  complete_q1_20_gate_registry,
  run_genesis_root,
  run_genesis_object,
  14,
  gate_rows_1_through_14,
  14,
  0x003fff,
  true,
  prestart_authority_object
]
```

Strict replay re-encodes each CBOR object, recomputes tag/gate/genesis/evidence
roots, checks ordinal/name/pass fields, and requires count `14`, mask `0x003fff`,
and popcount `14`.

## Authority state

The authority object remains:

```text
Q1 state                 = NOT_RUN
Q1 gate count / mask     = 14 / 0x003fff
Q1 gate total            = 20
Q1 output slots          = eight named nulls
Q1 completion receipt    = null
Q2 state                 = NOT_RUN
M3 formal roots          = null
formal fixed point       = false / null
target/split/role access = false / false / false
outside certificate      = false
Q1 start authorized      = false
ACTIVE transition        = false
```

Publishing this receipt therefore admits prestart state only. It is not a Q1
measurement, execution, completion, adoption, promotion, production, or
current-pointer action.
