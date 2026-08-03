# Phase-3A M2.5 live Docker actor protocol qualification v1

Status: engineering qualification contract

Formal status before and after: `14/24 / NOT_RUN`

Authority class: `NON_FORMAL_PUBLIC_SYNTHETIC_PROTOCOL_QUALIFICATION`

## Purpose

This qualification closes the remaining implementation-risk question before
external genesis: can the four hard-isolated technical roles execute the exact
Commit-A protocol together, using actual ephemeral purpose-private Ed25519
keys, without instantiating the real split seed?

The run exercises:

1. four simultaneously live, digest-pinned Docker actors;
2. offline `--network=none` and `--pull=never` operation;
3. distinct purpose-private key volumes and operation-bound live receipts;
4. purpose 4 rebuilding and auditing the detached full parent history inside
   its own key-bearing actor;
5. purpose 1 replaying and signing the full bridge DAG;
6. purpose 2 independently replaying the full DAG and verifying purpose 1;
7. purpose 3 doing the same through the exact post-Commit-A-qualified Rust
   binary;
8. verified removal of all four containers and all four ephemeral key volumes.

It uses one frozen public synthetic split response and an in-memory synthetic
prospective marker. The wrapper never delegates `seed_split`, seed resume,
post-stage recovery, or `complete_marker` to the Docker custodian. Therefore no
custody marker, seed intent, seed completion receipt, or raw seed can be
created.

## Independence claim

Docker isolation is accepted as the formal technical-role model for this
project. The diagnostic report always discloses the exact limitation:

```text
same_admin_controller=true
organizational_independence=false
independent_human_actors=false
technical_role_independence=true
owner_accepted_threat_model=true
remote_attestation=false
hardware_key_nonexportability=false
```

This is a technical reproducibility and role-separation claim. It is not a
claim that independent organizations or independent human custodians took
part. Human nationality, employer, or headcount is not an engineering control
and is not part of qualification eligibility.

## Commit-A binding

The requested basis must be exactly the current `HEAD`, not merely an ancestor.
Every input in `REQUIRED_COMMIT_A_INPUTS`, together with this module, CLI, test,
and document, is read both from Git and the worktree and must be byte-identical.
The report contains only a domain-separated digest and count of that source
set. Archive replay recomputes both values from the exact Git commit. It also
replays the committed actor profile and its four digest-pinned image
assignments rather than trusting report-supplied image strings.

Every non-formal live-admission Commit-A read—qualification, actor runtime,
container ceremony, errata replay, secret-absence replay, parent audit,
purpose-4 detached audit default Git, and M3 shadow admission—invokes absolute
`/usr/bin/git` in the exact repository root with a closed environment. System
and global config are disabled and pointed at `/dev/null`; replace objects and
lazy fetch are disabled; protocol-from-user, SSH and terminal prompting are
disabled; `HOME`, locale and `PATH` are fixed; no `GIT_DIR`, work-tree,
object-directory, alternate-object, replace-base or config-injection variable
is inherited. A replace ref, partial-repository lazy fetch, fake `PATH` Git or
hostile parent environment therefore cannot make source checking read one
tree while actor use reads another. Purpose-4 may still replay through an
explicitly supplied, digest-bound detached Git runtime; its default is the
same `/usr/bin/git` and its environment has the same offline/no-replace rules.

Both Rust executables are fixed:

- the formal static replayer is the executable bound by the qualified M3
  implementation basis;
- the bridge-DAG replayer is the stable binary and strict diagnostic report
  accepted by `phase3_m25_bridge_dag_binary_qualification_v1`.

No caller-selected executable or qualification report is accepted.
The complete M3 implementation-qualification receipt is archived and replayed
against the committed golden vector. The formal Rust and bridge Rust binary
digests and the bridge qualification report are checked against the fixed
local artifacts. Static Rust replay uses the actor backend's sanitized local
Docker control plane and its validated daemon receipt; ambient `PATH`, Docker
configuration and proxy variables are not inherited.

## Custody reservation and cleanup

The operator supplies an existing, empty, caller-owned mode-`0700` directory
on a durable Linux-local filesystem. It must be outside the repository,
outside `/tmp`, outside `/mnt/c`, and outside detected cloud-sync paths.

Before inspecting emptiness or making the first file visible, the supervisor
opens the already existing custody directory with `O_NOFOLLOW|O_DIRECTORY` and
takes a non-blocking exclusive directory `flock`. It then creates exactly three
mode-`0600` files with exclusive writes:

```text
phase3_m25_ceremony.lock
opaque-run-<16-byte-id>.reserved
opaque-ledger-<16-byte-id>.reserved
```

They provide the minimum identity set expected by the existing actor backend.
The file lock is opened, inode-matched and locked immediately after its durable
creation, while the directory lock already excludes recovery. Both locks
remain held throughout the run. Cleanup never recursively deletes a path.
Each reservation's device, inode, mode, and payload digest is rechecked before
unlink. The three exact files are removed only after the backend has verified
that all actor containers and key volumes are absent. Opaque unlinks receive a
directory-fsync barrier; the final lock-path unlink and its directory fsync
also complete before either flock is released. The supplied directory itself
is never removed.

If an unrecognized custody entry appears, cleanup refuses to proceed. If actor
absence cannot be established, reservation files are retained rather than
claiming a clean qualification.

If the supervisor is terminated, a later invocation first acquires the stale
directory lock and then the stale file lock non-blockingly before listing or
changing entries. It strictly replays the Commit-A/run/ledger reservation and
invokes the existing exact run-labelled pre-seed recovery path even for a
reservation prefix. Only after containers and key volumes are verified absent
may it remove the exact recorded inodes. Qualification never invokes the
backend seed handoff, so the directory owner must remain the supervisor UID;
an actor-UID ownership transition is not a legal qualifier state. A live
directory/file lock, another commit, a seed/marker path, an unknown file, or
an invalid identity fails closed. A write/fsync exception after inode creation
removes only the exact device/inode created by that call. A truncated first
lock with no opaque files is removable without Docker because actor start was
unreachable; a truncated opaque body is removed only after lock-bound exact
run-labelled Docker absence recovery.

Raw descriptors are never left to object lifetime or garbage collection. If
fingerprint validation, unlink, either fsync barrier, backend absence checking,
or orphan parsing raises, both advisory locks are explicitly unlocked/closed
while the remaining disk evidence is preserved. A later same-process recovery
can therefore reacquire the directory and file locks instead of wedging until
process termination.

## Publication boundary

The process constructs synthetic protocol objects in memory because the
signers must receive the same typed DAG that formal genesis will later use.
Those objects are never evaluated or promoted as `GateEvidenceInputsV1`.
The replay archive deliberately serializes the public qualification material
needed to detect a digest-only forgery:

- four `LIVE_PROTOCOL_QUALIFICATION_ONLY` public-key manifests;
- every operation request body, actor receipt, Rust raw receipt where
  applicable, and host receipt;
- the complete purpose-4 request/response and all three bridge packages,
  replay receipts and signatures;
- the exact frozen public synthetic split fixture;
- four purpose-separated pre-cleanup statements signed by a dedicated
  in-container `qualification-finalize` worker;
- the exact destruction plan, container/volume identities, and the separate
  post-cleanup host/daemon absence receipt.

The finalizer independently reconstructs the statement hash domain, purpose
octet, signature preimage and canonical request before using its fixed
purpose-private key. The four actors sign the evidence and destruction plan;
they do not falsely sign a post-destruction observation. The bundle authority
ID binds those statements, the plan and the later cleanup receipt without a
signature cycle.

The archive never contains any actor private key, real or latent split seed,
split assignment, formal role key, authoritative formal root, gate/promotion
evidence or M3 execution identity. Its diagnostic self-hash is not a trust
anchor. Strict replay independently checks all nested bodies, signatures,
content IDs and the final bundle ID, and rejects extra fields, duplicate JSON
keys, floats, non-canonical JSON, changed operation order or authority
escalation. Actor key-volume names must equal the exact run/purpose-derived
name, Docker return codes must be JSON integers rather than booleans, and all
Ed25519 signature transports must be exactly 128 lowercase hexadecimal
characters; equivalent but non-unique text encodings are rejected.

## Live capability versus archive replay

Loading the JSON returns only
`ReplayedActorProtocolQualificationV1`. It can support consistency checks and
recovery, but it can never unblock formal execution.

A successful same-process live run instead returns an opaque
`LiveActorProtocolAdmissionV1`. Only after actor/key-volume destruction and a
complete strict archive replay does the live qualifier's terminal branch
lazily create a process-random HMAC secret and bind a fresh nonce, issuer PID,
Commit A, daemon receipt, bundle ID, four qualification key IDs and immutable
canonical archive bytes. There is no module-level replay-object-to-token
issuer: archive validation and loading cannot reach capability issuance. The
token is non-serializable, rejected after `fork`, and consumable once via
`consume_live_actor_protocol_admission_v1`. Direct construction, copied disk
fields, cross-commit use, HMAC mutation and a second consumption fail closed.

This is a same-admin Python-process engineering boundary, not protection
against arbitrary monkeypatching by an administrator in that process. That
limitation is part of the frozen seven-field disclosure.

Actual ephemeral signing is acknowledged, but it cannot change the formal
state:

```text
m3_gates_before=14
m3_gates_after=14
m3_gate_delta=0
m3_state=NOT_RUN
m3_run_started=false
```

The report also states the more precise construction boundary:

```text
authoritative_formal_roots_generated=false
synthetic_formal_shaped_roots_computed_in_memory=true
formal_roots_published=false
```

## Operator command

Run only after deterministic Commit A has been committed and pushed, the Rust
bridge binary qualification report exists for that exact commit, and a durable
empty custody directory has been created:

```bash
TMPDIR=/tmp PYTHONPATH='Hegel Machine/src' \
python3 'Hegel Machine/tools/phase3_m25_actor_protocol_qualification_v1.py' \
  --basis-commit "$(git rev-parse HEAD)" \
  --custody-directory "${HEGEL_PROTOCOL_QUALIFICATION_CUSTODY:?set repo-external mode-0700 path}" \
  --output 'Hegel Machine/artifacts/phase3_m25_external/phase3_m25_live_actor_protocol_qualification_v1.json'
```

The output path is exclusive and is never overwritten. The custody directory
must already exist, be empty, and have mode `0700`. This command requires no
network access after the pinned OCI images and checksum-verified Cargo vendor
snapshot have been bootstrapped once.

The standalone command consumes its one-shot live token solely to write the
diagnostic archive. Formal execution does not load that file as authority: it
runs and consumes a new live admission in the same process, durably embeds the
exact canonical bundle in the prestage transaction before any formal entropy,
and uses the embedded bundle for crash recovery rather than rerunning the
qualifier.

Passing this qualification permits the separate external-genesis action to be
considered. It does not execute that action and does not itself advance any
gate.
