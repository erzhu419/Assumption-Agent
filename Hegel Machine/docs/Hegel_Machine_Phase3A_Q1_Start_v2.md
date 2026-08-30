# Hegel Machine Phase 3A — Q1 target-blind start successor v2

## Purpose and authority

This protocol replaces, but never resumes, the terminal Q1-start-v1 campaign. One exact v2 owner authorization covers one logical campaign: it binds the pure, unconsumed successor-prestart candidate bundle and authorizes one fresh `Python -> Rust -> trusted host replay` actor sequence. The candidate bundle has no independent execution, adoption, or publication authority.

The authorized scope is exactly `TARGET_BLIND_Q1_START_V2_ONCE_WITH_BOUND_SUCCESSOR_PRESTART_BUNDLE`. Each actor may receive at most one durable intent and one Popen. Completed actors are never measured again.

## Bound lineage and resource correction

The capsule binds three distinct inputs:

1. the authoritative Q1 prestart state `NOT_RUN`, 14/20, mask `16383`, with eight null output slots;
2. the immutable Q1-start-v1 terminal record, whose Python actor started once while Rust and host did not; and
3. the complete successor-prestart candidate bundle, its canonical bytes, raw SHA-256, semantic bundle root, fresh source manifests, derived actor implementation roots, run genesis, and admission receipt.

The v1 Python action is conservatively counted. V1 is terminal no-resume, and neither its output nor scratch work product may enter v2. The new authorization explicitly permits one fresh v2 actor sequence from new roots. The corrected endpoint scratch envelope is exactly `134217728` bytes (128 MiB).

This is a post-v1 successor correction, not a reinterpretation of the historical preregistration or 14/20 receipt. Those predecessor bytes continue to record the original 4 MiB guard. V2 derives 128 MiB from the already-persisted v6 values `max(1646592, 40452166, 3194598)`, applies the same twice-and-power-of-two rule, and binds the result in new Gate 11--14 evidence and fresh actor implementation roots.

## Fresh local execution boundary

V2 uses independent run, ledger, build, artifact, and candidate-bundle paths. The candidate bundle is outside both the run and ledger roots. The controller closure binds the v2 bundle/campaign/controller/supervisor code, the exact v1 replay and formal-actor modules, pinned local container images, Rust sources and lockfile, and the read-only seccomp profile. Target truth, split, and role inputs are absent. Network access is prohibited.

## PENDING and terminal rules

Before consumption, a local path, dependency, or capacity problem may produce exact same-capsule PENDING. It consumes no authorization, starts no campaign, creates no actor intent, and invokes no supervisor.

After consumption, PENDING is allowed only at a clean completed prefix `[]`, `[python]`, or `[python, rust]`, before the next actor intent. Reopening replays all bindings and never repeats a completed actor. Once host intent exists, no PENDING route remains. Any post-intent unknown outcome, unverifiable actor artifact, unproved cleanup/absence, inconsistent ledger state, or other non-replayable condition is terminal no-resume.

Partial finalization may only complete a missing outcome or terminal record from exact durable evidence. It cannot call the supervisor or any actor again.

## Success and claim boundary

Success alone publishes Q1 as `COMPLETE`, 20/20, mask `1048575`, with eight non-null output roots. Q2 remains `NOT_RUN`; M3 roots, role evaluation, and outside certificates remain null or false. This campaign is target-blind, local-only, non-adopting, and non-promoting. It does not update a current pointer, enter `ACTIVE` or production, or make an external scientific claim.
