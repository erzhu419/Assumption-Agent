# Phase-3A M2.5 Rust Bridge-DAG Offline Binary Qualification v1

## 1. Status and claim boundary

This engineering path qualifies the independent Rust
`m25_bridge_dag_replay` implementation from one exact deterministic
implementation basis commit (Commit A).  Its result is
`IMPLEMENTATION_BINARY_QUALIFICATION_ONLY_NOT_FORMAL_EVIDENCE`.

It does not instantiate a split seed, generate a private key or signature,
publish an authoritative/formal root, advance any M3 gate, or change
`NOT_RUN`.  The replay fixture is a checked, public, unsigned purpose-1
package.  It contains no private key, seed, or signature.  All roots inside
that fixture are explicitly synthetic and non-authoritative test values.

The technical-role disclosure remains:

- `same_admin_controller=true`
- `organizational_independence=false`
- `independent_human_actors=false`
- `technical_role_independence=true`
- `owner_accepted_threat_model=true`
- `remote_attestation=false`
- `hardware_key_nonexportability=false`

## 2. Frozen output identities

The stable release binary is:

`rust/m25_bridge_dag_replay/target/commit_a_qualified/hegel-m25-bridge-dag-replay`

The stable canonical qualification report is:

`artifacts/phase3_m25_external/phase3_m25_bridge_dag_rust_binary_qualification_v1.json`

The binary is published only by a mode-`0755`, fsync-backed atomic replace
after all fresh replay checks pass.  The report is one canonical ASCII JSON
line with a domain-separated SHA-256 self-binding.

## 3. Exact Commit-A and dependency closure

The qualifier first compares every declared qualification/build/test blob in
the worktree byte-for-byte with Commit A.  It then invokes `/usr/bin/git`
with system config and replacement objects disabled, creates an exact
`git archive` containing only those paths, safely extracts it on a private
Linux-local `/tmp` filesystem, and compares every extracted file again with
`git show <commit>:<path>`.

The Rust build includes both crates:

- `rust/m25_bridge_dag_replay`
- its path dependency `rust/formal_bridge_m25`

The root `Cargo.lock` is parsed as an exact registry package set.  Every
`.crate` archive is selected only if its SHA-256 equals the lock checksum,
safely extracted into a run-private vendor snapshot, and represented by the
same typed dependency-snapshot root used by the approved M2.5 OCI policy.
The host Cargo cache is never mounted into Docker.

## 4. Frozen offline OCI execution

The only Rust image is:

`rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`

The control plane is the exact `/usr/bin/docker` client and local
`unix:///var/run/docker.sock`, with an empty private Docker config and a
sanitized environment.  Every build and replay uses:

- `--pull=never`
- `--network=none`
- a read-only container root
- all capabilities dropped
- `no-new-privileges`
- the committed build or runtime seccomp profile
- `env -i` plus the exact frozen whitelist

Compilation uses `cargo --locked --offline`, a read-only Git archive, a
read-only vendor snapshot, and a fresh Linux-local target.  No Cargo home,
registry directory, or cache is mounted.

## 5. Qualification replay set

Before publication, the fresh release binary must pass, inside the same
network-disabled OCI boundary:

1. unsigned public purpose-1 full-DAG replay;
2. public-preimage substitution rejection with
   `FAIL_M25_BRIDGE_REPLAY_ROOT_BINDING`;
3. node-omission rejection with `FAIL_M25_BRIDGE_REPLAY_NODE_SET`;
4. authoritative flag without runtime opt-in rejection with
   `FAIL_M25_BRIDGE_REPLAY_AUTHORITY_GUARD`.

After atomic publication, the stable binary repeats the positive replay and
must emit byte-identical canonical receipt output.  The report retains only
test IDs, return codes, exact negative codes, and stdout/stderr hashes; it
does not publish the synthetic candidate/bridge roots from the fixture.

## 6. Executor binding

The executor-facing API is:

```python
report, binary_sha256 = load_qualified_rust_bridge_dag_binary_binding_v1(
    expected_basis_commit=commit_a,
)
```

That call strictly loads the canonical stable report, rechecks its Commit-A
source bindings, validates the Docker/toolchain/dependency/replay boundary,
and verifies the mode and SHA-256 of
`DEFAULT_RUST_BRIDGE_DAG_BINARY`.  The executor should copy exactly that
binary into its private runtime snapshot and bind both `binary_sha256` and
the report's `diagnostic_report_sha256`.  A caller-supplied binary path,
report path, digest, Cargo executable, image, or source tree is not an
equivalent qualification.

## 7. Commit sequencing

Commit A includes the qualifier module, tool, tests, this document, the
public unsigned fixture, both Rust crate sources/locks, and the frozen OCI and
seccomp policy files.  It does not include the generated release binary
(`rust/**/target/` is ignored) or the generated qualification report.

Only after Commit A is committed and pushed may the operator run:

```text
python3 -I tools/phase3_m25_bridge_dag_binary_qualification_v1.py \
  --implementation-basis-commit <COMMIT_A>
```

The generated stable report may be archived in the later evidence commit;
it must never be retrospectively inserted into Commit A, whose exact bytes it
qualifies.
