# Rust M3 bounded closure enumerator

This crate is the independent Rust implementation of the Phase-3 M3
canonical-program enumeration path.  It is not the historical 25,872-source
capacity subset and it does not evaluate either target role.

The formal entrypoint requires the three committed semantic roots embedded in
every `CanonicalProgramRecordV2`:

```bash
cargo run --release --locked --offline -- \
  --enumerate-prefix \
  --child-dsl-spec-root HEX64 \
  --operator-semantics-root HEX64 \
  --identifier-registry-root HEX64 \
  --output-directory /tmp/hegel-rust-m3
```

The output directory contains length-framed canonical program records, chunk
manifests, bucket-accounting records, and a JSON summary.  No network access,
seed, target truth table, or role evaluation is used by this executable.

Stdout is one strict JSON object with schema
`hegel-m3-rust-closure-enumerator-report/1`.  It supplies the deterministic
enumeration fields consumed when constructing
`M3ImplementationEnumerationReceiptV1`; it is not itself a signed M3 receipt,
because the run ID, execution-manifest root, committed implementation binding,
environment root, and timestamps are ceremony inputs.  The executable has no
target/split/seed arguments and does not import or execute role evaluation.

`--binding-material` prints deterministic public metadata for the enclosing
`ImplementationBindingV1`.  The final binding must additionally be constructed
from the exact committed source/dependency roots, compiled binary digest,
offline OCI environment root, golden-vector root, and repository commit.

## Frozen diagnostic replay

The committed machine-readable diagnostic is
[`golden/diagnostic_prefix_v1.json`](golden/diagnostic_prefix_v1.json).  It uses
the visibly synthetic roots `11…11`, `22…22`, and `33…33`; its roots are **not
formal M3 roots** and may never be copied into a formal execution manifest.

It was originally reproduced in the already-local pinned image.  The current
admissible replay path is the commit-bound implementation qualification CLI:

```bash
python -m hegel_machine.phase3_m3_implementation_qualification_cli_v1 \
  --basis-commit COMMITTED_SHA1 \
  --output /tmp/hegel-m3-implementation-qualification.json
```

Qualification reads each crates.io archive named by the committed
`Cargo.lock`, verifies its locked SHA-256 on the host, and extracts only those
archives into a run-private mode-0700 vendor snapshot.  The Rust build
container mounts that snapshot read-only.  It never mounts
`~/.cargo/registry`, uses `cargo --locked --offline`, and runs with
`--pull=never --network=none`.  The one-time cache bootstrap that supplied the
single previously absent locked crate is recorded in
`artifacts/phase3_m3_cargo_offline_bootstrap_record_v1.json`; it is not part of
qualification and may not recur during later builds or replays.

The frozen diagnostic result is `DSL_TOO_LARGE`: 3,292,439 raw applications,
50,000 archived records, 13 chunks, 175 accounting buckets, and an independent
program-50,001 witness.  The warmed release replay took 2.77 seconds on the
recording machine.  The development-tree binary digest is intentionally not
frozen here.  `ImplementationBindingV1` qualification rebuilds from exact
Commit-A blobs and binds that resulting binary digest; a pre-commit digest is
never accepted as a substitute.

Bucket accounting uses mutually exclusive admission classes: a source whose
frozen normalization changes its AST increments `rewrite_collapses` and exits
that attempt; it is not also counted as a `syntactic_duplicate`.  The normal
form is reached through its own canonical source-token construction.  This
rule is part of the Python/Rust bit-exact agreement and fixes the discarded
pre-agreement diagnostic bucket root that double-counted rewritten aliases.

The current diagnostic agreement values are:

```text
canonical_program_archive_root = a23151e07f77edcbebe5b7e2e382e1a81b36c6b15c8997899f7f43dcbda874d1
program_chunk_manifest_root    = 98c8deb02a62630f5813717a28c3b9deb5a3845e663b9af5a78fe7f9427f453d
bucket_accounting_root          = 5dd13e5d284785dab7fbe3c16fbb1f1bcba3a44466ab0fb258f75f36ee9661ec
program_50001_ast_hash          = 96200a6a131204315ffcd1efd0aa2dcfe2ce665a2c06516461772c9812f0ec71
```

`cargo test --release --locked --offline` includes both a small ten-program
test-mode prefix check and the full 50,001 diagnostic golden replay.  Test mode
is library-internal and cannot weaken the fixed budgets exposed by the CLI.
Even the fixed-budget report is
`FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY` with
`authoritative_claim_allowed=false` until later formal receipt assembly.
