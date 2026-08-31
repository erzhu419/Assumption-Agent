# Hegel Machine Phase 3A — Q1 target-blind start successor v5

## Purpose and authority

This protocol replaces, but never resumes, the terminal Q1-start-v4 campaign. One exact v5 owner authorization covers one logical campaign: it binds the pure, unconsumed successor-prestart candidate bundle and authorizes one fresh `Python -> Rust -> trusted host replay` actor sequence. The candidate bundle has no independent execution, adoption, or publication authority.

The authorized scope is exactly `TARGET_BLIND_Q1_START_V5_ONCE_WITH_BOUND_SUCCESSOR_PRESTART_BUNDLE`. Each actor may receive at most one durable intent and one Popen. Completed actors are never measured again.

## Bound lineage and resource correction

The capsule binds five distinct inputs:

1. the authoritative Q1 prestart state `NOT_RUN`, 14/20, mask `16383`, with eight null output slots;
2. the immutable Q1-start-v2 `OUTCOME_UNKNOWN` terminal lineage, with all actor counts zero;
3. the immutable Q1-start-v3 `INCONCLUSIVE` terminal record with failure `REJECT_Q1_START_SUCCESSOR_BUNDLE` / `validator import failed: AttributeError: module 'importlib' has no attribute 'util'`, before which none of the three actors started; and
4. the immutable Q1-start-v4 `INCONCLUSIVE` terminal record with failure `Q1_START_ENDPOINT_REPLAY_MISMATCH` / `same_run_record_set_root differs from replay`, after exactly one Python actor action and before any Rust or host action; and
5. the complete successor-prestart candidate bundle, its canonical bytes, raw SHA-256, semantic bundle root, fresh source manifests, derived actor implementation roots, run genesis, and admission receipt.

V4 is terminal no-resume. Its completed prefix is empty, its Python action is conservatively counted once, and none of its output or scratch work product may enter v5. The new authorization explicitly permits one fresh v5 actor sequence from new roots. The corrected endpoint scratch envelope remains exactly `134217728` bytes (128 MiB).

This is a distinct-outer successor to v4, not a reinterpretation of the historical preregistration or 14/20 receipt. V5 corrects the reached replay bug by retaining the old three-stream snapshot diagnostic root for structural validation while comparing endpoint results against the full-node6 four-stream `PROGRAM -> COHORT -> CLASS -> COVERAGE` same-run root. V5 carries the corrected 128 MiB resource envelope and binds it again in new V5 Gate 11--14 evidence and fresh V5 roots.

## Fresh local execution boundary

V5 uses independent run, ledger, build, artifact, and candidate-bundle paths. The candidate bundle is outside both the run and ledger roots. The controller closure binds the new v5 bundle/campaign/controller/supervisor code, the exact v2/v3/v4 terminal lineages, the existing v1 formal-actor modules, pinned local container images, Rust sources and lockfile, and the read-only seccomp profile. Target truth, split, and role inputs are absent. Network access is prohibited.

## PENDING and terminal rules

Before consumption, each controller entry may invoke exactly one bound independent `/usr/bin/python3.10 -I -S -B tools/phase3_q1_start_supervisor_v5.py --startup-smoke --run-spec ...` Popen. This smoke is neither a campaign-supervisor invocation nor an actor invocation. Its strict receipt must pass before consumption. A smoke failure, local path, dependency, or capacity problem produces exact same-capsule PENDING: it consumes no authorization, starts no campaign, creates no actor intent, and invokes no campaign supervisor or actor.

After consumption, PENDING is allowed only at a clean completed prefix `[]`, `[python]`, or `[python, rust]`, before the next actor intent. Reopening replays all bindings and never repeats a completed actor. Once host intent exists, no PENDING route remains. Any post-intent unknown outcome, unverifiable actor artifact, unproved cleanup/absence, inconsistent ledger state, or other non-replayable condition is terminal no-resume.

Partial finalization may only complete a missing outcome or terminal record from exact durable evidence. It cannot call the supervisor or any actor again.

## Success and claim boundary

Success alone publishes Q1 as `COMPLETE`, 20/20, mask `1048575`, with eight non-null output roots. Q2 remains `NOT_RUN`; M3 roots, role evaluation, and outside certificates remain null or false. This campaign is target-blind, local-only, non-adopting, and non-promoting. It does not update a current pointer, enter `ACTIVE` or production, or make an external scientific claim.
