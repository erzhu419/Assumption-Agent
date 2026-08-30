# Hegel Machine Phase 3A Q1 start v1

Status: **implemented/preflight-only until a fresh exact-capsule-digest
authorization is recorded**.

## Authority boundary

The only admissible predecessor is the authoritative Q1 prestart receipt with
this exact state:

- `q1_state = NOT_RUN`;
- `q1_gate_count = 14`, `q1_gate_total = 20`, and
  `q1_gate_mask = 0x003fff`;
- all eight named Q1 output slots are null;
- Q2 is `NOT_RUN`;
- M3 formal roots and role/certificate outputs are null or false;
- target, split, and role access are false;
- no `ACTIVE`, adoption, promotion, production, or current-pointer state exists.

That predecessor does not authorize Q1 start. A start requires a fresh capsule
that binds the exact predecessor, source/runtime/actor identities, run roots,
resource contract, and output locations. The project owner must authorize that
capsule's exact digest separately. Authorization of a different digest, an
earlier preflight, or the predecessor amendment is not interchangeable with
this authorization. This document itself records no authorization and consumes
none.

## One sequential campaign

One authorized logical campaign has exactly three actors, in this order:

1. the Python formal archive materializer;
2. the Rust formal archive materializer;
3. the trusted host replay.

Each actor may receive at most one intent and may start at most once. Actors do
not overlap. Python must have an exact persisted completion before Rust intent;
Rust must have an exact persisted completion before host intent; the host must
have an exact persisted completion before final completion is sealed.

Completed work is never measured or invoked again. Resume from the Python
checkpoint starts with Rust. Resume from the Python-plus-Rust checkpoint starts
with the host. A checkpoint cannot be used to repeat, replace, or reinterpret a
completed actor.

## PENDING, checkpoints, and terminal uncertainty

Before authorization consumption and before Python intent, a non-core local
problem may produce pre-consumption `PENDING`. Re-entry must use the same run
root and exact capsule digest. It performs full validation again but does not
consume authorization, invoke an actor, or count as a campaign execution.

After consumption, clean-checkpoint `PENDING` is allowed only at these three
boundaries:

- authorization is exactly consumed, no Python intent exists, and the actor
  prefix is empty;
- Python is exactly complete and no Rust intent exists;
- Python and Rust are exactly complete and no host intent exists.

The completed prefix and its immutable artifacts must replay exactly before the
next actor can receive intent. No clean checkpoint exists after host intent.
The in-memory consumption candidate is not treated as consumed until the final
consumption path replays byte-exactly. A failure before publication therefore
remains unconsumed; a tail failure after the exact final path exists replays as
the consumed empty-prefix checkpoint.

Once an actor intent exists, an absent, partial, inconsistent, or unknown exact
completion for that actor is terminal no-resume. The same rule applies to an
interruption whose post-intent state cannot be proven exact. Such a state cannot
return to PENDING, cannot resume the actor, and cannot launch a replacement
actor under the consumed authorization.

After all three actors complete, a partial final publication never invokes the
supervisor or an actor again. Exact persisted artifact/outcome windows are
replayed and only the missing outcome or terminal seal is completed. Recovery
still replays the capsule-bound controller HEAD, tree, and source closure; the
only permitted worktree dirtiness is the exact fixed completion artifact. An
artifact that cannot be verified is retained as unbound failure evidence and
produces a dedicated terminal `NOT_RUN` outcome, never a Q1 output.

## Gates 15--20

Gates 1--14 are inherited byte-exactly from the predecessor receipt. The start
campaign may add only the following ordered gates:

| Gate | Frozen name | Exact successful evidence |
|---:|---|---|
| 15 | `BOTH_SIGNATURE_FIXED_POINTS_COMPLETE_NO_GUARD_HIT` | The run-genesis root, predecessor receipt root, and ordered Python/Rust endpoint completion evidence. Each endpoint binds the exact successful canonical `endpoint-result.json`, the frozen actor implementation/source identity, and the replayed fixed-point records. The frozen materializer path rejects every non-`SATURATED` partition before an endpoint result can exist; actor resource measurements are also below their frozen ceilings. The eight count rows are the exact frozen guard ceilings, not claimed observed peak counts. |
| 16 | `PYTHON_RUST_SIGNATURE_ARCHIVES_AND_STATE_PREIMAGES_BYTE_EQUAL` | The two endpoint evidence roots, common formal-file closure root, ordered Odd/Sink completion rows, common closure-bundle root, exact archive-byte equality, and exact state-preimage equality. |
| 17 | `HOST_REPLAY_DUAL_SIGNATURE_CLOSURES_PASS` | One dual-replay agreement object and its root. It binds both endpoint roots, the common signature rows and closure bundle, the host implementation and replay result, and successful byte/state/host replay agreement. |
| 18 | `TARGET_BLIND_ACCESS_LEDGER_AND_ISOLATION_POSTCHECK_PASS` | One target-blind access ledger object and its root. It binds ordered Python/Rust/host isolation postchecks, an empty forbidden-access event ledger, absent target-input keys, clean scratch termination, and false network/target/split/role access. |
| 19 | `Q1_CLOSURE_BUNDLE_AND_COMPLETION_RECEIPT_SEALED` | The closure-bundle root, the complete completion-core object and root, and the fixed completion-seal policy. The core already binds the predecessor, Gates 15--18, and output slots 1--7. |
| 20 | `Q1_COMPLETE_Q2_M3_ROLE_CERTIFICATE_OUTPUTS_NULL_NOT_RUN` | The completion-core root, the exact Q1 `COMPLETE` authority object, and the fixed slot-8 `SELF_CONTENT_ROOT` projection policy. |

Every passed gate row is `(gate_id, gate_name, true, evidence_root32)`. Gate
order, name, subject shape, registry roots, and evidence roots are replayed
strictly; Python boolean/integer aliases are not wire-equivalent.

Endpoint-result and host-result roots are computed only from strict canonical
JSON objects that satisfy their successful frozen result boundaries. The
root adapters require the current capsule digest, source commit, run-genesis
root, predecessor receipt root, closure-bundle root, and ordered Odd/Sink
signature roots as expected bindings and reject any stale-result splice.
The formal-file and output-tree closure roots are computed from ordered relative
paths and exact file bytes using the supervisor's frozen streaming recipe.
Gate 18 precheck/postcheck, source-manifest, mount-manifest, output-closure,
and scratch-absence roots are likewise derived from typed evidence preimages;
the completion assembler does not accept an outer caller's ungrounded root as
a substitute for those derivations. Each persisted isolation row carries its
actor-local mount-manifest root, and its precheck and postcheck roots are
recomputed from the other exact row fields during construction and strict
decode. The runtime mount registry is then derived only from the ordered
`python`, `rust`, `host` pairs of actor ID and actor-local mount-manifest root,
with network, target, split, and role access all false. The ledger also requires
the unchanged prestart formal-input-key absence root; neither aggregate root is
an opaque caller assertion.

Formal endpoint replay is imported only from the materialized, manifest-bound
host actor source snapshot under a capsule-specific private module namespace.
It does not reuse a `hegel_machine` package already loaded from the mutable
controller worktree.

## Completion core, receipt, and output slots

The completion core is constructed before Gates 19 and 20. It binds:

- the run-genesis and predecessor prestart-receipt roots;
- the exact Gate 15--18 subject rows and recomputed gate rows;
- the seven non-recursive output roots listed below;
- Q2 `NOT_RUN` and null/false M3, role, certificate, target, and split state.

The final completion receipt then embeds that core, adds the recomputed Gate 19
and Gate 20 rows, and embeds the `COMPLETE` authority. The receipt deliberately
does not embed its own root. Its internal authority puts the fixed
`SELF_CONTENT_ROOT` policy in slot 8. Only after the receipt bytes are complete
is the receipt content root computed and projected into external output slot 8.
This is the sole slot-8 self-reference resolution; no provisional root or
recursive hash is accepted.

The final external Q1 output registry is exactly:

| Slot | Canonical name | Value |
|---:|---|---|
| 1 | `odd_signature_archive_manifest_root` | Odd signature archive-manifest root |
| 2 | `odd_signature_saturation_state_root` | Odd signature saturation-state root |
| 3 | `sink_signature_archive_manifest_root` | Sink signature archive-manifest root |
| 4 | `sink_signature_saturation_state_root` | Sink signature saturation-state root |
| 5 | `q1_closure_bundle_root` | Common Q1 closure-bundle root |
| 6 | `q1_dual_replay_agreement_root` | Dual Python/Rust/host replay-agreement root |
| 7 | `q1_target_blind_access_ledger_root` | Target-blind access-ledger root |
| 8 | `q1_completion_receipt_root` | Final receipt content root, projected externally |

Successful strict replay yields exactly:

```text
Q1 state                 = COMPLETE
Q1 gate count / mask     = 20 / 0x0fffff
Q1 gate total            = 20
Q1 output slots          = eight non-null named roots
Q2 state                 = NOT_RUN
M3 formal roots          = null
role / certificate       = null / false
target/split/role access = false / false / false
network access           = false
adoption / promotion     = false / false
current / ACTIVE         = false / false
```

## Claim boundary

The implementation and its synthetic validation are an implemented,
preflight-only control plane until a fresh exact capsule is published and its
exact digest is explicitly authorized. Before that event Q1 remains the
predecessor `NOT_RUN / 14 of 20` state. No code or document publication starts
Q1, completes Q1, adopts a result, or updates authority state.

The campaign is target-blind, split-blind, role-blind, offline, non-adopting,
and non-promoting. It permits no network access and no current-pointer or
`ACTIVE` transition. Its eventual local result, whether success or terminal
failure, is internal engineering evidence only and makes no external
scientific claim.
