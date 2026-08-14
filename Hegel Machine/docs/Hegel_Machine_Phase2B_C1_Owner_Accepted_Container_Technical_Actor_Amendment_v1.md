# Hegel Machine Phase-2B C1 Owner-Accepted Container Technical-Actor Amendment v1

**Document ID**: `hegel-phase2b-c1-owner-accepted-container-technical-actor-amendment-v1`

**Policy profile ID**: `hegel-owner-accepted-phase2b-c1-container-technical-actors-v1`

**Authority class**: `OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1`

**Applies to**: Phase-2B C1 technical-actor eligibility and a future actual-unsealed-960 ceremony

**Wire effect**: none

**Statistical-contract effect**: none
**Status**: `OWNER_AUTHORIZED_PHASE2B_TECHNICAL_ACTOR_ELIGIBILITY_POLICY_ONLY`

## 1. Scope extension

The owner-accepted authority class previously adopted for Phase-3A M2.5 is
extended to the technical actors needed by Phase-2B C1. In this scope,
“independent” means measured technical role separation outside the repository
builder/orchestrator process. It does not mean different people, organizations,
administrators, jurisdictions, hosts, or Docker daemons.

This amendment replaces only a human- or organization-only eligibility reading
of Phase-2B custodian, store, time, recognizer, and evaluator roles. It does not
change any Phase-2B wire, hash domain, case allocation, threshold, confidence
bound, preservation requirement, baseline requirement, reveal order, or C1 gate.
It does not turn existing candidate mechanics, synthetic fixtures, or Phase-3
qualification reports into Phase-2B evidence.

The machine-readable policy is
`config/phase2b_c1_container_actor_profile_v1.json`. Loading and validating that
profile is a static operation. It performs no Docker call, key generation,
signature, ledger mutation, recognizer execution, scoring, gate evaluation, or
C1 promotion.

## 2. Exact six-role topology

The future live Phase-2B ceremony requires six simultaneously qualified,
purpose-separated containers:

| Purpose | Role | Authority effect | Exclusive writable mounts |
|---:|---|---|---|
| 1 | `GOVERNANCE_ANCHOR` | signs the owner-accepted registry and terminal checkpoint | `/state` |
| 2 | `CUSTODIAN` | generates, seals, holds, reveals, and signs custody events | `/state`, `/custody` |
| 3 | `STORE_RECEIPT` | sole writer for durable `IF_ABSENT` ledger/CAS objects and receipts | `/state`, `/ledger`, `/cas` |
| 4 | `TIME_ATTESTOR` | signs monotonic logical-order tokens | `/state` |
| 5 | `RECOGNIZER` | performs the one blinded 960-case prediction operation | `/state`, `/prediction` |
| 6 | `SCORER_EVALUATOR` | scores once after reveal and signs report roots | `/state`, `/score` |

Purposes 1–4 are the exact four graph-signature roles represented by the
Phase-2B candidate checkpoint mechanics, in this registry order:

```text
GOVERNANCE_ANCHOR / GOVERNANCE_CHECKPOINT
CUSTODIAN        / CUSTODIAN_CHECKPOINT
STORE_RECEIPT    / STORE_IF_ABSENT_RECEIPT
TIME_ATTESTOR    / DECLARED_TIME_TOKEN
```

The four signatures do not all live in the checkpoint tuple. Their frozen
graph-wide locations are governance and custodian at
`checkpoint.checkpoint_signatures[0:2]`, store at
`store_receipt.detached_signature`, and time at
`time_token.detached_signature`. Purposes 5 and 6 are execution actors, not
extra checkpoint signers. Their fresh purpose keys bind respectively a
`RECOGNIZER_OPERATION_RECEIPT` and `SCORER_EVALUATION_REPORT`, using the exact
domains and signed-field orders in the machine-readable profile. They must not
be inserted into the frozen four-role candidate DTO.

The existing external-candidate checkpoint verifier remains valid mechanics,
but its frozen authority requirement demands organizational independence,
external CAS/rollback witness, and independently authoritative time. It cannot
by itself establish this owner-accepted Docker authority class. A future
Phase-2B execution registry must instead sign and bind both the versioned
owner-accepted authority-requirement ID and the content ID of this six-role
profile, together with the campaign and attempt, before holdout generation.
This is an overlay/version boundary; the old signed preimage is not silently
reinterpreted.

Every role requires a fresh ceremony Ed25519 operation key. Keys, qualification
keys, private volumes, and Phase-3 artifacts may not be reused across purposes
or ceremonies. Public keys may enter a registry only before holdout generation.
The static profile loader never creates a key or registry.

## 3. Dataflow and mount separation

The governance actor receives profile, owner-accepted requirement, registry,
public commitment roots, execution-operation receipts, the terminal report
store receipt, and the signed logical terminal token—not plaintext answers or
recognizer workspaces. The custodian alone receives
the durable secret custody mount. The store alone receives the durable ledger
and CAS mounts; it stores opaque bytes and content-address requests and must not
receive the raw custodian seed or plaintext answer manifest. The time actor sees
event roots, not answer contents.

Before reveal, the custodian must receive and verify the prediction
`IF_ABSENT` receipt and signed `PREDICTIONS_COMMITTED` token. The recognizer
receives only the profile, execution-freeze manifest, signed `RUN_STARTED`
token, and blinded recognizer-input archive as read-only input, plus its
purpose-private prediction output. It receives no answer or gate manifest,
generator source, custody, ledger, live repository, Docker socket, or other
role state. The scorer receives the prediction `IF_ABSENT` receipt, signed
reveal token, committed prediction archive, and answer/gate manifests only
after the reveal gate. It receives no uncommitted prediction workspace,
custodian state, ledger state, or live repository. Thus the actor contracts
carry the evidence needed to enforce the three chronology gates instead of
trusting an unrecorded host decision.

No two roles share a writable mount. The host orchestrator may transport only
the exact public or opaque objects permitted by the profile and may never act as
a generic signing oracle.

## 4. Docker and durable-state controls

All roles use the local `/usr/bin/docker
--host=unix:///var/run/docker.sock` control plane with a private empty client
configuration and sanitized environment. Images are selected only by future
execution-basis content digests; missing images fail closed. Pulling, registry
access, and runtime networking are forbidden.

Every role runs non-root as `65534:65534`, with a read-only root filesystem, all
capabilities dropped, `no-new-privileges`, the bound custom seccomp filter,
bounded resources, private namespaces, and no Docker socket, live repository,
host home, `/mnt/c`, or other-role private mount. Requested flags are not
evidence: a Phase-2B-specific six-role qualifier must observe the live container
IDs, host PIDs, PID/mount/network/IPC/UTS namespaces, capability sets, seccomp
mode, network interfaces, inherited descriptors/environment, filesystem
visibility, blocked syscalls, and post-exit removal.

Persistent custody, ledger, CAS, and logical-order state must be outside the
repository, `/tmp`, `/mnt/c`, cloud sync, network filesystems, and FUSE. The
directory must be a real mode-`0700` Linux-local durable directory with no
symlink traversal. Records use mode `0600`; file and directory `fsync`, atomic
rename probes, hard-link no-replace publication, a cooperating-process lock,
and explicit crash recovery are required.

These controls support one-shot enforcement against cooperating processes and
ordinary restarts. They do not resist a malicious administrator who deletes or
rolls back all local state, and they do not establish cross-host global
one-shot behavior.

## 5. Signed logical chronology

The time actor provides a persistent, signed, strictly monotonic logical order.
It does not claim UTC accuracy, an external timestamp authority, or an
independent wall clock. The frozen attempt order remains:

```text
PREREGISTERED
-> RAW_INPUT_COMMITTED
-> PACKAGE_COMMITMENTS_FROZEN
-> RUN_STARTED
-> RUN_FINISHED
-> PREDICTIONS_COMMITTED
-> REVEALED
-> SCORING_STARTED
-> REPORTS_COMMITTED
-> TERMINAL_CONSUMED
```

Reveal requires the prediction `IF_ABSENT` store receipt. Scoring requires the
reveal token. Terminal consumption requires the report store receipt. A later
operational verifier must jointly replay custodian signatures, store receipts,
logical-order tokens, content addresses, and actor-runtime receipts before it
may make a chronology or one-shot claim.

## 6. Reuse boundary

The Phase-3 container profile, runtime seccomp filter, offline-build seccomp
filter, Linux-local durable-path checks, sanitized Docker control-plane binding,
daemon identity receipt, and live-probe logic are frozen security-control
sources for the new implementation. Their exact paths and SHA-256 values are
bound in the machine-readable profile.

The existing Phase-3 qualifier and formal executor are not Phase-2B evidence.
They hard-code four M2.5 purposes and M2.5 operations. A fresh Phase-2B
six-role qualifier, adapted probe, operation allowlist, ceremony executor, and
receipt verifier are required before any live claim can advance.

## 7. Mandatory disclosure and claim boundary

Every profile snapshot and future publication must retain:

```yaml
same_admin_controller: true
organizational_independence: false
independent_human_actors: false
technical_role_independence: true
owner_accepted_threat_model: true
remote_attestation: false
hardware_key_nonexportability: false
```

Here `technical_role_independence: true` is a required profile property, not a
live observation. The distinct-container qualification claim remains false
until the six-role probe has actually run and been verified.

Static validation may claim only that the profile schema, six-role contract,
base-control bindings, and disclosure are frozen and content-addressed. All
actual-run claims in the profile are exactly `false`, including Docker use,
live actor qualification, key/signature generation, ledger/one-shot and
timeline verification, recognizer/runtime execution, actual-960, scoring,
formal gates, effect, and C1 exit.

Even after a successful future run, the following remain false without a new
threat-model amendment and new evidence:

- organizational or independent-human authority;
- protection against malicious host root, Docker daemon, or administrator;
- owner deletion or rollback resistance and cross-host global one-shot;
- remote or hardware-backed attestation;
- authoritative UTC or external timestamping;
- absence of all covert channels.

A successful profile load is therefore neither execution authorization nor
evidence that Phase-2B or C1 is effective.
