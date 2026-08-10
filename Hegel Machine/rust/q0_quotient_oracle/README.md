# Phase-3A-Q0 independent Rust quotient oracle

This crate is the target-blind Rust endpoint for the frozen
`hegel-q0-micro-projection-v1`. It independently implements:

- the four-row Odd/Sink observation adapter and exact-bottom evaluator;
- strict rational-grid behavior over all 15 frozen leaves and active v1.6
  canonical operators;
- exhaustive canonical syntax enumeration;
- exact behavior classes, MDL-aware visible Pareto frontiers, and the
  capacity-bounded continuation cohort bank;
- direct quotient fixed-point saturation, including expansion of every real
  admitted bank representative exactly once;
- canonical CBOR, ContentHash, RFC6962 program/class/coverage archives, and
  the implementation-neutral endpoint state.

The existing shrink-6 crate is used only as the strict admission boundary.
The evaluator, input adapter, quotient state, bank, fixed-point engine, and
wire construction in this crate do not import Python results, target truth,
split assignments, role evaluation, or formal M3 roots.

The single-endpoint PASS status is
`SINGLE_IMPLEMENTATION_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS`. It is a Q0
qualification result only. It is not the host-only DUAL status, a Q1/Q2
result, an M3 formal root, role membership, or an outside-language
certificate.

## Offline build and replay

The crate has no crates.io dependency beyond dependencies already present in
the sealed Cargo cache. Use the pinned Rust image and disable the network:

```bash
docker run --rm --pull=never --network=none \
  --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --memory=512m --memory-swap=512m --pids-limit=64 \
  --ulimit=nofile=128:128 \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=512m,mode=1777 \
  --user="$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e CARGO_HOME=/cargo-home \
  -e CARGO_TARGET_DIR=/tmp/cargo-target \
  -e CARGO_NET_OFFLINE=true \
  -e CARGO_BUILD_JOBS=1 \
  -v /path/to/sealed-cargo-home:/cargo-home:ro \
  -v "/path/to/Hegel Machine:/input:ro" \
  -w /input/rust/q0_quotient_oracle \
  rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89 \
  cargo test --locked --offline -j 1 --target x86_64-unknown-linux-gnu
```

Replace `cargo test` with `cargo run` to emit the deterministic diagnostic
JSON endpoint. The supervisor deterministically constructs the sealed Cargo
home from the 21 `Cargo.lock`-selected, checksum-verified crate archives; the
container never mounts the mutable provisioning cache. No network access is
required after the pinned image and crate archives have been provisioned once.

The golden tests bind the probe, projection manifest, semantic binding,
program archive, visible class archive, syntax/direct coverage, complete
saturation-state preimages, saturation roots, endpoint state,
target-isolation flags, resource guards, and the multiplicity-resurrection
counterexample that invalidated the earlier single-representative draft.
