# Hegel Machine Phase-3 shrink-4 sealed dual strict qualification result v1

Status: **PASS — NON-FORMAL DUAL STRICT QUALIFICATION; M3 NOT RUN**

This record publishes the result of the commit-bound shrink-4 admission
qualification. It is evidence for the Python/Rust recognizer boundary only.
It is not a closure run, a formal root publication, an outside-language
certificate, a target evaluation, or an ACTIVE-governance transition.

## 1. Immutable basis and evidence

```text
Source Commit O  cd2c32bd3a27004b40f4550229f33afd73647433
source subject   hegel: freeze shrink4 two-clause admission
parent commit    c286732c140bd9adcfd3eef2b1788b3eac0eb3e9
source rows      61
source-set root  sha256:03d5ab95e02f5fa6bb48db11ccb3682e0250985cb2ea17ad4372f4b2969c1a8e
Git archive      48aaf482708a00ab3cc84710007b13aa6975f71155d906722ece356b5c05d96e
```

The host supervisor verified its own bytes against Source O. Both recognizers
and both replay endpoints executed only from the extracted Source-O archive.
The canonical qualification report is:

```text
artifacts/phase3_m3_runtime/phase3_shrink4_sealed_dual_strict_qualification_v1.json
file SHA-256      41fdea5fd9b16ab436386ef7794412ffa46e17e68efc6b8448deed17c7f99aae
diagnostic hash   sha256:44b4e0c0a2b79f6afb67ace348c1b3726e0ba64058c97c4c61be0c111ef6acec
```

The report is canonical one-line JSON with recursively exact object schemas.
Unknown or additional fields fail closed even if the diagnostic hash is
recomputed.

## 2. Dual qualification outcome

The Python and Rust implementations agreed on every sealed wire:

```text
status              SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS
claim level         NON_FORMAL_DUAL_STRICT_QUALIFICATION
sealed vectors      22 / 22 on each implementation
manifest root       sha256:f84035e632bf5a655a9ebd636a0cafe7ab1097c45be87d4db944a0012f52aa90
dual outcome root   sha256:c19341f08ac5f5759c2cdcb3681a37d958de362b81d02c184f7e2413dca18d7c
```

Both implementations also replayed the entire inherited constructive AND2
survivor set:

```text
source candidates              2,160
accepted / unique              2,160 / 2,160
parent identity matches        2,160
rejections / rewrite collapse  0 / 0
accepted-set commitment        sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e
subset status                  FULL_AND2_SURVIVOR_SET_ONLY_NOT_COMPLETE
```

This 2,160-program set is a survivor replay, not a closure cardinality.

## 3. Isolation and reproducibility

The run used two digest-pinned local images, `--pull=never`,
`--network=none`, read-only runtime mounts, dropped capabilities, and a fresh
temporary Rust target volume. Two replay workers were used. The volume was
removed after the run.

The offline Cargo transport was committed under a separate domain-separated
manifest before the build and checked again after it:

```text
regular files    43
total bytes      3,907,160
manifest root    sha256:a280e5a05d54c2904c19b5ad296650acd90de853ce5260deb93cdade595cef80
Rust binary      d82f329b354bc4722cb3eaedac501018ccde7c02f3116926f62c2406bb985adc
```

An independently replayed offline rebuild from the Source-O archive reproduced the same
Rust binary hash, 22/22 golden result, 2,160/2,160 capacity replay, and dual
outcome root. Two separate audits found no P0, P1, or P2 finding and
confirmed that the dedicated qualification and audit volumes were absent
after cleanup.

The isolation claim remains technical process/container independence under
one administrative controller. It is not organizational or independent-human
custody.

## 4. Authority boundary and next admission

The evidence preserves every closed guard:

```text
execution_state                 NOT_RUN
closure_executed                false
formal_roots_generated          false
formal_roots                    null
seed/signature/certificate      absent
target roles evaluated          false
ACTIVE governance changed       false
formal state transition allowed false
```

Accordingly, this result admits development of the independent shrink-4
complete enumerators. It does not establish `COMPLETE`, `DSL_TOO_LARGE`, an
odd-target result, a hidden-sink result, MDL success, or
`OUTSIDE_FROZEN_CLOSURE(...)`.

The next two-commit step is a source-only complete-enumeration freeze
(Source Commit Q), followed by an immutable dual run and a separate Evidence
Commit R. No remote push was performed for Source O or this evidence work.
