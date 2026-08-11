# Hegel Machine Phase-3A Q0.5b Actual Attempt 1 — Failure Errata v1

## Status and claim boundary

This document records the only actual Q0.5b attempt made from Commit A:

- source commit: `07d91847ceff1dc1692b05084b17718409155392`;
- branch: `codex/hegel-m3-start-runtime`;
- exactly-once attempt consumed: `true`;
- attempt result: `FAILED_CLOSED_AT_STAGE_07`;
- last completed stage: `6`;
- failed stage: `7`;
- process exit code: `1`;
- canonical error wire emitted: `false` (stdout was empty);
- Q0.5b qualification predicates: `0/20`;
- qualification candidate receipt: `null`;
- qualification final receipt: `null`;
- published artifact: absent;
- intended artifact path:
  `/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/artifacts/phase3_q05b_dual_qualification_v1.json`;
- Q1 state: `NOT_RUN`;
- Q1 formal output roots: eight `null` values.

The attempt therefore produced no Q0.5b qualification and no Q1 result.  This
errata is an engineering failure record, not a qualification receipt.  The
attempt is not retried by this change.

## Authorized invocation

The production entry point was invoked exactly once with:

```text
PYTHONPATH='/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/src' PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B '/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q05b_dual_qualification_v1.py' --run --project-root '/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine' --source-commit 07d91847ceff1dc1692b05084b17718409155392 --artifact '/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/artifacts/phase3_q05b_dual_qualification_v1.json' --cargo-cache-source /home/erzhu419/.local/state/hegel-machine/rust-cargo-cache
```

Observed resource summary:

| Field | Value |
|---|---:|
| elapsed seconds | `1938.34` |
| user CPU seconds | `1080.19` |
| system CPU seconds | `116.97` |
| maximum RSS, KiB | `1108388` |
| stdout bytes | `0` |

## First cause

The attempt completed the two offline Rust builds, binary detach and sealing,
the Python and Rust endpoint runs, and the trusted-host run.  It then failed
while Stage 7 assembled its outer-replay evidence.

The real return type of `dual_actor_host_replay_v1` is
`DualHostReplayV1`.  It exposes the already-computed 32-byte field
`shadow_assembler_root`; it does not expose a top-level `shadow_assembler`
object.  Commit A read the nonexistent nested field:

```python
replay.shadow_assembler.root.hex()
```

The exact terminal exception was:

```text
Traceback (most recent call last):
  File "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q05b_dual_qualification_v1.py", line 16081, in <module>
    raise SystemExit(main())
  File "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q05b_dual_qualification_v1.py", line 16032, in main
    value = run_actual_v1(
  File "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q05b_dual_qualification_v1.py", line 15711, in run_actual_v1
    result = orchestrate_actual_with_backend_v1(
  File "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q05b_dual_qualification_v1.py", line 15350, in orchestrate_actual_with_backend_v1
    execute_stage(stage_id, stage_name)
  File "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q05b_dual_qualification_v1.py", line 15289, in execute_stage
    result = method(context_snapshot())
  File "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q05b_dual_qualification_v1.py", line 14157, in stage_07_v1
    "shadow_assembler_root": replay.shadow_assembler.root.hex(),
AttributeError: 'DualHostReplayV1' object has no attribute 'shadow_assembler'. Did you mean: 'shadow_assembler_root'?
```

This was a production blocker, but it was fail-closed: Stage 8 candidate
construction, Stage 9 final-receipt construction, Stage 10 publication
binding, and atomic artifact publication were never reached.

## Cleanup closure

Failure cleanup completed without cleanup errors.

- The fixed artifact path and its temporary publication siblings were absent.
- The private work root `/tmp/hegel-q05b-actual-u8chjhwl` was absent.
- The repository was clean before and after the attempt.
- The Docker inventory was a byte-identical, sorted set of 18 full 64-hex
  container IDs before and after the attempt.  Both snapshots have SHA-256
  `8d6bffea616001f85404321046a82c8914507aa920b017039c2fc2f733d8be3f`.
- No Q0.5b process, container name, or ownership label remained.

The attempt execution namespace was:

```text
50388cf1854c56b348f25ec3e35f1d00fa7d56b293d4649f6f55548d8223f212
```

For each slot below, two consecutive production-format authoritative-absence
samples were validated.  Each sample returned exit code `1`, stdout `[]\n`
with SHA-256
`37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`,
and the exact name-bound “no such object” stderr represented by the final
column.

| Slot | Name suffix | stderr SHA-256 |
|---|---|---|
| `RUST_TEST` | `rust-test` | `05714f8e08fcdb030712afa4d1fcc694a39a2ed514de1f0e429d7748ed59ad84` |
| `RUST_RELEASE` | `rust-release` | `b178892d8cad354d347b33eb4065c6d26e19cc7f90e6d0e7ee6aa57d5b738b22` |
| `PYTHON_ENDPOINT` | `python` | `dd7424e1913255e2b96b4c48eb036a91848ae1146f52bcc9cca6f2a6cf61946f` |
| `RUST_ENDPOINT` | `rust` | `1a8f1c171b02759177b1ae9ff13889811d4c50ae9f7ff0fcbea61129becaa41a` |
| `TRUSTED_HOST_REPLAY` | `host` | `986235e27054060aea55d4734e0328ee6fb136487840859f8de167aae891d58b` |

Each full name is
`hegel-q05b-<execution-namespace>-<name-suffix>`.

## Raw failure-evidence manifest

The one-attempt log root was mode `0700`, owned by uid/gid `1000/1000`.  Its
nine regular files were mode `0400` and had the following frozen metadata:

| File | Bytes | SHA-256 |
|---|---:|---|
| `docker-after.ids` | 1170 | `8d6bffea616001f85404321046a82c8914507aa920b017039c2fc2f733d8be3f` |
| `docker-before.ids` | 1170 | `8d6bffea616001f85404321046a82c8914507aa920b017039c2fc2f733d8be3f` |
| `git-status-after.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `git-status-before.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `head-before.txt` | 41 | `3db580f19ce1715894f2ddcbabc3910f05302d81310edeaa04f7fd9c86d1434e` |
| `name-absence.json` | 7320 | `6aaecd75430240769710844facd1ad48f3e3d729deabc3372dd6ac80071cfaa7` |
| `stderr.log` | 1392 | `474dfc007dcaf51aaf3e23c3d9088b0eddc10fcfb23cd3869b823def52033647` |
| `stdout.json` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `time.txt` | 103 | `37cf58f8f4970c4c8a585afdf4042a9b1e1cd06b89e7b36d5390a939d7a4a58b` |

The manifest digest is
`805aca208885717710061ac8fa5bd4f4605c8da6fcfde60b7bce22c1915bd0bc`.
It is SHA-256 over canonical ASCII JSON plus LF for the filename-sorted rows
`[basename, mode_decimal, uid, gid, byte_length, payload_sha256]`.

The raw `/tmp/q05b-actual-once.MzfOWl3Z` directory is deliberately not a
durable repository artifact.  It may be removed only after this record and the
fix have been locally committed.

## Repair and regression boundary

The production repair is intentionally schema-neutral:

```python
replay.shadow_assembler_root.hex()
```

No serialized field, schema version, root domain, or root preimage changes.

The prior Stage 4-through-7 regression used a `SimpleNamespace` that invented
the same nonexistent top-level `shadow_assembler` attribute, so the mock
validated the bug instead of the production interface.  The regression now
constructs the real `DualHostReplayV1`, uses its exact predicate and pending
registries, asserts that no top-level `shadow_assembler` exists, and checks the
Stage 7 evidence against `dual.shadow_assembler_root.hex()`.

The repaired bytes passed:

- Python bytecode compilation for the production tool and dual test module;
- the real-interface Concrete Stage 4-through-7 regression;
- synthetic Stage 8-through-final-delivery orchestration;
- Stage 8/9 receipt-registry tamper rejection; and
- Stage 10 receipt/root/artifact-byte adapter cross-checks.

These are non-actual tests.  No second actual attempt, Docker run, Cargo run,
artifact publication, or Q1 execution was performed after the failure.

## Disposition

Commit A remains the only source commit on which an actual attempt was made,
and that attempt failed.  The follow-up repair commit is local-only and remains
actual-unexecuted.  Any second actual attempt requires a new explicit operator
authorization; it must never be inferred from this repair or this errata.
