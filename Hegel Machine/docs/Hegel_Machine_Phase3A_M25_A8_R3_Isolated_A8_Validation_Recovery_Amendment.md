# Phase-3A M2.5 A8 R3 Isolated-A8 Validation Recovery Amendment

Status: implementation amendment for one fixed transaction. It is not a new
formal basis, not an ordinary retry mechanism, and not an M3 start action.

## Scope

R3 is recovery attempt ordinal 3 for only:

- formal basis `0af65964235390ce2bebefea7379eaa9c50eda24`;
- run `e4af9f57c38fb298462ec628c4ed8a03`;
- ledger `ec849e2f1e2e1163cfc450370b25b484`;
- the existing `PENDING / RESERVED` complete-seed prefix;
- `REAL_PENDING_RESUME`, with zero formal-identity entropy draws.

R1 ended before source admission. R2 was separately authorized and consumed,
then ended at `SOURCE_ADMISSION` with no admission receipt. Its exact failure
is bound by raw SHA-256
`bd64cfa99885dd60750615fcb23abd960aed78ef676a0d2d4d8ed942e5395d56`
and receipt SHA-256
`87b400cf0070efdb3e2f9d7b37dc09675258c5b0341ce629b7c7b6c5431f3f58`.
R3 never reuses attempt 2.

## Root cause and representation boundary

The persisted intent is diagnostic JSON. Its decoder historically restored
every JSON array as a Python tuple. The actor report contains 37 JSON arrays,
the errata report 19, and the transaction-local protocol bundle 208. Their
canonical JSON bytes and stored hashes are unchanged, but live report
validators require JSON lists. R2 therefore failed first at the actor image
binding. Blindly applying transport normalization is insufficient: a side
worktree also changes the actor report's expected host-repository path digest,
and the amendment source differs from the original A8 errata source set.

R3 consequently keeps two domains distinct:

1. formal typed arrays remain tuples;
2. the three embedded diagnostic JSON documents retain JSON list identity;
3. canonical bytes, hashes and round trips must remain exact;
4. report admission runs in an isolated interpreter bound to the unchanged
   main A8 context;
5. ordinary live validation remains the default everywhere outside the exact
   R3 admission.

## Isolated A8 child

The validator is launched by the exact command prefix
`/usr/bin/python3.10 -I -S -B -X
pycache_prefix=/nonexistent/hegel-r3-pycache`, using the fixed executable
SHA-256 and a sanitized environment. `-I` excludes caller path and Python
environment influence, `-S` disables automatic site initialization, and the
nonexistent cache prefix prevents reads from or writes to the main A8 package's
existing `__pycache__`. Before import it verifies the main worktree HEAD and
every tracked package, configuration, tool and Rust input used by admission
against Git A8 blobs. It rejects an untracked top-level package module,
verifies imported `__file__` paths are under the main A8 source root, and
repeats the closure verification after validation.

Because Python 3.10 with `-S` does not expose the required TOML and
cryptographic dependencies, R3 also freezes a source-vendored `tomli` 2.4.1
tree and the installed Ubuntu `cryptography`/`bcrypt` trees plus the exact
native `_cffi_backend`, `six`, OpenSSL, libc, libffi and loader leaves. The
validator verifies those bytes and metadata before exposing their three
explicit import roots, checks loaded module locations and `/proc/self/maps`,
then verifies the closures again. Their combined dependency-closure root is
`f39b2f922af5723ee50374b4f04be5c6525a58a87e19de9376d2525a108d1dc7`.

The child performs full `validate_ceremony_admission_v1` and transaction-local
actor-protocol bundle replay. Its deterministic canonical receipt contains
only public identities, the complete 98-input A8 binding, zero-entropy and
secret-absence flags. Its frozen raw SHA-256 is
`ef18694aa41a78389cef2265eb121174f2e68548928f89f7fcad3f55fb261ee4`.
Preparation records it; execution must independently reproduce the same exact
bytes.

## Provenance and authorization

R3 must be one clean committed sole child of R2 commit
`ec7c04cf62190558c72448639d7e3cd13a5b6903`. Its manifest binds every changed
blob. The fresh audit path is fixed to attempt 3 and binds the exact R1 chain,
all six R2 terminal records, the R2 chain root, unchanged custody/stage/public
reservations, four Docker key volumes, zero run-labelled containers, fixed
runtime binaries and the isolated A8 receipt.

Pre-attempt preparation is crash-resumable only as an exact immutable prefix.
Visible records are installed after full fsync by no-replace hard link. A
fresh project-owner authorization must contain the exact phrase:

`AUTHORIZE_A8_R3_ATTEMPT_3_COMPLETE_ONLY_REAL_PENDING_RESUME`

`attempt-start.json` is the consumption edge. It is published only by linking
a complete, file-fsynced hidden inode into the visible name without replace,
then fsyncing the audit directory; a partial record can therefore never become
the consumption edge. A caught post-link durability fault is stabilized and
terminalized as failure. An uncatchable process/power loss, or a persistent
storage failure that also prevents the terminal failure receipt from being
installed, may leave an exact complete `attempt-start.json` as the
authoritative fail-closed consumed tombstone, never as authority to retry
attempt 3. Unpublished admission, finalize and failure `.next` files are
non-authoritative and are discarded with a directory fsync on caught failure.
Once the visible record exists, attempt 3 may never be invoked again. Success
creates `admission.json` and `finalize.json`; caught failure creates a terminal
`failure.json` when storage permits. A later continuation would require a new
ordinal, new code amendment, new audit directory and new authorization.

## Prohibited operations

R3 does not authorize ordinary execute, abort, redraw, post-stage recovery,
raw-seed read, raw-seed hashing, identity regeneration, network access, or
`phase3-m3-start`. The only formal mutation path is the complete-seed-only
pending recovery core after durable source admission. Success must replay
24/24 gates, retain child state `NOT_RUN`, and keep all 15 M3 output roots
null. M3 remains a separate explicit future action.
