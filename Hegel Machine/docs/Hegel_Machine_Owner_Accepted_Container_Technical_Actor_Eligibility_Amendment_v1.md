# Hegel Machine Owner-Accepted Container Technical-Actor Eligibility Amendment v1

**Document ID**: `hegel-owner-accepted-container-technical-actor-eligibility-amendment-v1`
**Policy profile ID**: `hegel-owner-accepted-container-technical-actors-v1`
**Applies to**: Phase-3A M2.5 purpose actors 1–4
**Wire effect**: none
**DSL/freeze effect**: none
**Status**: `OWNER_AUTHORIZED_TECHNICAL_ACTOR_ELIGIBILITY`

## 1. Decision

For this project, actor independence is judged by reproducible technical controls and
observable role separation, not by the number, nationality, employer or informal promise
of human participants. A process does not qualify merely because a different person runs
it; conversely, a containerized actor is not rejected merely because the same project owner
administers the host.

The human-only eligibility language in the M2.5 external-genesis runbook and earlier
readiness discussions is therefore replaced, for actor purposes 1–4, by the exact technical
profile in this amendment. Existing formal CBOR tags, schemas, hash domains, purpose IDs,
signature preimages, Gate 15–24 predicates, DSL ID and freeze ID remain byte-for-byte
unchanged.

This is a threat-model decision, not a claim that containers create different people or
organizations.

## 2. Accepted authority class

The accepted class is:

```text
OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1
```

It may satisfy the existing formal actor/custody predicates only when every required live
probe and artifact check passes. Every publication must disclose:

```yaml
same_admin_controller: true
organizational_independence: false
independent_human_actors: false
technical_role_independence: true
owner_accepted_threat_model: true
remote_attestation: false
hardware_key_nonexportability: false
```

Within this profile, the existing field `external_actor_evidence` means evidence produced
outside the repository-building/orchestrator process by the frozen, isolated purpose
runtime. It does not mean evidence from an external company, jurisdiction or human
committee.

## 3. Exact offline container profile

The machine-readable profile is
`config/phase3_container_actor_profile_v1.json`; the runtime syscall filter is
`config/phase3_internal_actor_seccomp_v1.json` and the separately frozen compiler filter is
`config/phase3_m3_offline_build_seccomp_v1.json`. Their Git blobs and SHA-256 digests must
be bound by the execution basis commit and replay receipt.

Every Docker call uses the exact `/usr/bin/docker
--host=unix:///var/run/docker.sock` control plane, a private empty client config and a
sanitized six-key host environment. Ambient Docker contexts, proxy variables and remote
endpoints are forbidden. A live receipt binds the local Linux daemon, Unix socket,
Docker-root mount and storage driver. All ephemeral input snapshots, builds and client
state are created below a validated Linux-local `/tmp` private directory; persistent
custody is instead a repository-external, non-`/tmp`, non-DrvFS/network/FUSE, caller-owned
`0700` directory on a durable local filesystem, qualified by file/directory fsync and
atomic-rename probes.

Each purpose runs in a different container with all of the following controls:

- image selected by immutable content digest;
- `--pull=never` and no implicit build or registry operation;
- `--network=none` for every probe and execution container;
- read-only root filesystem and read-only, commit-bound inputs;
- non-root UID/GID `65534:65534`;
- all Linux capabilities removed;
- `no-new-privileges` enabled;
- the project seccomp profile loaded;
- one purpose-private, run-labelled daemon-managed Docker volume with exact
  `Driver=local`, `Scope=local`, empty options and local mountpoint, mounted at `/state`,
  with no shared writable role mount;
- bounded memory and PID count;
- exact environment, file-descriptor and output allowlists;
- no Docker socket, live repository, host home or `/mnt/c` mount;
- container removal and descendant absence checked after exit.

Missing local images are a fail-closed condition. The runner must never pull an image,
contact a registry or enable networking to repair the condition. One-time future image
replacement requires a new owner-approved profile version and fresh qualification; it is
not an automatic update path.

Each newly created purpose volume starts root-owned under Docker. Before an actor starts,
a one-shot administrative initializer from the same pinned local image must run with
`--pull=never`, `--network=none`, read-only root, the frozen seccomp profile, root UID and
exactly `CAP_CHOWN`; it sets `/state` to `65534:65534` mode `0700`. A second one-shot process
must run as `65534:65534` with all capabilities removed and successfully stat, create and
remove a probe file. The resulting receipt binds run, purpose, basis, profile digest and
image digest. The administrative initializer is not a formal actor and receives no seed,
key, signing request or Docker socket.

## 4. Live evidence, not requested configuration

Requested Docker flags are not evidence. Before keygen and before every sensitive
operation, the same Python worker process or the purpose-3 parent Rust worker must create a
fresh operation-bound receipt. It binds a purpose-local monotonic sequence, a fresh
16-byte nonce, exact request/input digest, run/profile/image/daemon identity and exact
`env -i` allowlist. The host strict-JSON parses it, independently replays its fields, and
checks the same live container with Docker inspect both before and after the operation.
At startup all four long-lived workers first perform `qualify-only`; no sensitive operation
is admitted until their container IDs, host PIDs and PID/mount/network/IPC/UTS namespaces
are pairwise distinct. Each purpose worker must report and the supervisor must independently
validate live state:

```yaml
CapInh: 0
CapPrm: 0
CapEff: 0
CapBnd: 0
CapAmb: 0
NoNewPrivs: 1
Seccomp: 2
network_interfaces_exactly: [lo]
```

The worker must actually invoke all six probes below. Each must return `-1` with
`errno=EPERM`:

1. `socket(AF_INET, SOCK_STREAM)`;
2. `socket(AF_INET6, SOCK_STREAM)`;
3. `mount`;
4. `ptrace(PTRACE_TRACEME)`;
5. `bpf(BPF_MAP_CREATE)`;
6. `perf_event_open`.

The default Docker seccomp profile is explicitly insufficient: local qualification showed
that it allowed both socket probes and `PTRACE_TRACEME`. Only the committed custom profile,
or a later owner-approved profile with equivalent or stronger passing probes, can qualify.

Additional required attacks include write attempts against the root and bound inputs,
visibility checks for the live worktree, Docker socket and host paths, inherited FD/env
checks, cross-purpose path checks, output framing/size/secret lint, signature replay across
purposes, and Python/Rust result mismatch injection.

Purpose 1 may exercise a `/custody` write probe only for `seed-split-real`,
`seed-split-resume`, `seed-split-synthetic` and `complete-marker`. Qualification, keygen,
ordinary signing and bridge signing see custody but may not use a write probe. Purposes 2–4
never receive the custody mount.

## 5. Purpose separation

The existing purpose registry remains authoritative:

| Purpose | Container responsibility |
|---:|---|
| 1 | custodian identity, formal split-seed first instantiation and constrained Python/Rust calculators |
| 2 | Python bridge replay and attestation |
| 3 | Rust bridge replay and attestation |
| 4 | parent-absence and policy audit |

Four containers must have distinct container IDs, PID/mount/network/IPC/UTS namespace
identities, ceremony-generated Ed25519 keys, public keys and key IDs. Purpose 1 alone may
receive the formal raw split seed. Purposes 2–4, the orchestrator and synthesis code must
never receive it. During execution, private-key material exists only in the corresponding
purpose-private persistent Docker volume. Those volumes exist solely to make the frozen
no-redraw recovery path possible; their names and labels are public metadata, while their
contents never cross purpose mounts. The raw split seed may additionally persist in the
runbook's repository-external `0700` secret-state directory as one `0600` regular file,
mounted only into purpose 1 when a later authorized stage requires it. Neither private keys nor the raw
seed may appear in any declared public output, log, argv, environment, stdin, repository
blob or cross-purpose mount.

Volume lifecycle is marker-sensitive and fail closed: marker absent means no irreversible
seed choice and requires destroy-and-verify; `PENDING` retains all four volumes for an
explicit same-run recovery; malformed or unreadable marker state retains them and fails;
`COMPLETE` permits destruction only after the durable staged evidence has been reloaded,
prospectively replayed and shown equal to the real marker. Ordinary `execute` never chooses
the recovery path implicitly. COMPLETE cleanup destroys and verifies exactly the four actor
key volumes; it retains the raw seed plus intent/completion/marker in durable custody under
the frozen continuity policy. It must never be described as destruction of all private
state.

Python and Rust calculators/attesters must be built from separately bound source paths and
must agree bit-for-bit on the frozen formal objects and roots. Different containers running
the same implementation are not implementation diversity.

## 6. Formal admission effect

When this profile passes, its four purpose actors are eligible to create the existing formal
M2.5 objects and signatures. It does not itself pass any gate. Gate 15–24 advance only after
fresh formal-domain replay establishes every existing predicate, including real first seed
instantiation, genesis-only ledger, parent-absence audit, dual roots, split quotas,
signatures, execution manifest and fifteen null output slots.

The state sequence remains:

```text
14/24 / NOT_RUN
-> fresh container technical-actor ceremony
-> 24/24 / NOT_RUN
-> separate explicit phase3-m3-start
-> RUNNING/CANONICAL_ENUMERATION
```

The shadow tag/domain/state artifacts remain permanently non-authoritative. They may
qualify mechanics and reveal defects, but they cannot be renamed, rehashed or signed into a
formal object. The formal container ceremony must generate fresh formal seed, keys,
objects, roots and envelopes.

## 7. Claim boundary

Allowed after a successful formal ceremony:

> Four owner-accepted, digest-pinned, offline container technical actors executed the
> frozen purpose protocol with measured namespace, capability, filesystem, seccomp,
> channel and implementation-separation controls. The actors share one administrative
> controller and do not constitute independent people or organizations.

Not allowed without additional evidence:

- third-party or organizational independence;
- protection against a malicious host root or Docker daemon;
- remote or hardware-backed attestation;
- absence of all covert channels;
- formal Gate 15–24 completion based only on container launch flags or shadow evidence;
- outside-language, MDL, invention or ACTIVE claims before their separate gates.

This disclosed threat model is the project owner's chosen formal actor standard. It is
stronger and more reproducible than unmeasured human role-playing for the risks in scope,
while remaining explicit about the risks it does not address.
