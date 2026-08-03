# Phase-3A M2.5 Offline Technical-Actor Isolation Qualification v1

**status:** `IMPLEMENTATION_QUALIFIED_ZERO_SECRET`
**formal gate count:** unchanged at `14/24`
**M3 child state:** unchanged at `NOT_RUN`
**real seed/key/signature/formal root generated:** `false`
**authority class:** `OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1`

## 1. Qualified claim

The implementation can create four simultaneously live, purpose-private technical actors
using only locally present digest-pinned Docker images, with `--pull=never` and
`--network=none`. Before key generation and before every sensitive operation, the same
worker process—or, for purpose 3, the parent shell that subsequently invokes the signing
tool—produces a fresh operation-bound receipt. The host independently strict-parses the
receipt and replays the container, volume, filesystem, cgroup, environment, namespace,
capability and seccomp evidence.

This qualification adopts the owner-authorized disclosure without exaggeration:

```yaml
same_admin_controller: true
organizational_independence: false
independent_human_actors: false
technical_role_independence: true
owner_accepted_threat_model: true
remote_attestation: false
hardware_key_nonexportability: false
```

It establishes reproducible technical role separation. It does not establish different
people, organizations, host administrators, remote attestation or protection against a
malicious Docker daemon/root user.

## 2. Frozen host boundaries

- Every Docker command starts with exact `/usr/bin/docker
  --host=unix:///var/run/docker.sock`.
- The client uses an empty private `config.json` and an exact sanitized environment; no
  ambient Docker context or proxy variable is inherited.
- A live receipt binds the local Linux daemon, Unix socket, Docker root mount and storage
  driver. The receipt digest is bound into actor and volume labels.
- Ephemeral client state, committed input snapshots, Rust build inputs/outputs and seccomp
  snapshots are created under a validated, caller-owned Linux-local `/tmp` directory.
- Persistent custody is deliberately not temporary: it must be repository-external,
  non-symlink, caller-owned `0700`, outside DrvFS/9p/NFS/CIFS/FUSE/cloud paths, and on an
  allowlisted durable Linux-local filesystem. Qualification includes file fsync, atomic
  rename and directory fsync probes.
- Rust probe/split compilation uses the separately frozen offline-build seccomp profile,
  `--pull=never` and `--network=none`.
- Each key volume must inspect as daemon-managed `Driver=local`, `Scope=local`, empty
  options and the exact Docker-root mountpoint. Bind-, NFS- and plugin-backed substitutes
  fail closed.

## 3. Operation-bound evidence

The four long-lived actors first execute `qualify-only`. A sensitive operation is rejected
until all four receipts pass and container IDs, host PIDs and PID/mount/network/IPC/UTS
namespace identities are pairwise distinct.

Every later operation receives an exact `env -i` allowlist and binds:

1. purpose ID and operation ID;
2. a purpose-local monotonically increasing sequence;
3. a fresh 16-byte nonce;
4. a SHA-256 digest of the exact operation request and complete purpose input snapshot;
5. run, basis, profile, image and live-daemon identity;
6. the same container identity before and after execution.

The long-lived PID 1 actors receive only the report-safe base environment. The Python
actors produce canonical, self-hashed JSON from the same process that continues into the
requested operation, and remove the raw host-repository path from that process's actual
environment before any key, replay, signing or seed action. The Rust parent validates its
exact 19-key launch environment, immediately moves the raw path into a non-exported shell
value, and injects it only into the live absence-probe child; every other descendant
inherits the exact 18-key safe environment. The independent dependency-free Rust probe
also removes the raw key from its actual environment before serialization or lingering,
then the parent hashes that canonical receipt and
produces a strict parent-binding JSON receipt before continuing. All Rust shell utilities
and cryptographic commands use absolute paths.

Purpose 1's custody write probe is enabled only for `seed-split-real`,
`seed-split-resume`, `seed-split-synthetic` and `complete-marker`. `qualify-only`, keygen,
ordinary purpose-1 signing and bridge signing do not write-probe custody. Purposes 2–4 do
not receive the custody mount.

## 4. Live zero-secret result

The opt-in test
`test_four_long_lived_signers_qualify_offline_without_secret_generation` executed against
the already-local pinned Python/Rust images. It deliberately used a non-authoritative
working-tree snapshot and a harmless replay-binary placeholder, so its receipts are
implementation evidence only and cannot enter formal Gate 15–24 evidence.

Observed assertions:

```text
qualified purposes = 1,2,3,4
initial operation receipts = 4
purpose-1 repeated qualify-only sequence = previous + 1
purpose-1 repeated nonce = fresh
purpose-1 repeated request digest = fresh
public/private key count = 0
custody entry count = 0
real seed/signature/formal-root count = 0
remaining labelled containers after cleanup = 0
remaining labelled actor-key volumes after cleanup = 0
```

The relevant local-runtime, actor-runtime, purpose-4 detached-audit and formal-executor
test groups pass together with this live probe enabled. Other opt-in integration probes
remain skipped unless separately requested.

## 5. Cleanup semantics

The canonical completion APIs name the exact target: four actor key volumes. A successful
COMPLETE cleanup removes and verifies those volumes and their containers. It retains the
raw purpose-1 split seed, generation intent, completion receipt and COMPLETE marker in
durable custody under the frozen seed-continuity policy. No implementation receipt or
documentation may describe that state as “all private state destroyed.”

## 6. Formal non-effect and next gates

This result removes the actor-live-probe implementation blocker only. It neither advances a
formal gate nor authorizes external genesis. The authoritative state remains:

```text
14/24 / NOT_RUN
```

The remaining formal executor blockers are the full detached purpose-4 Git replay and the
complete purpose-1/2/3 bridge-DAG replay. Only after those implementations, their tests and
a real Commit-A-bound admission all pass may the separately explicit external-genesis path
generate real keys/seed/signatures/roots and attempt `24/24 / NOT_RUN`. M3 execution still
requires a later, separate `phase3-m3-start` action.
