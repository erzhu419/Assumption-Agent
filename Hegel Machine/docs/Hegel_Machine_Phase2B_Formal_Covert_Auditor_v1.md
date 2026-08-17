# Phase-2B Formal Covert Auditor v1

Status: production candidate for synthetic/development qualification only. It is
not an actual-corpus audit, a custody claim, an execution authorization, or an
actual-960 result.

The implementation is synchronized to the exact candidate Wire V3 source,
test, config, and documentation SHA-256 values
`c8120920f0e31b959043f63a0f672471fcbea50d05ca2cd8eaf7fd0c29089765`,
`253001d11d7603456b80cf43837ed71de567bfd2cd6613cc78b21ca23f7db0f0`,
`4f82ab060e2d53912cd17c8966a681a53d11e96ce3c6bb02ea3a42c5d2e85784`,
and `704a937967441ebbb9b2a77a865f83890e241d37031e2b0a87f1da2877b3456a`.
Git commit identity is not treated as runtime authority; these exact candidate
bytes are not yet an R03/R04/R05 qualification. The frozen rich-graph
1,024-row external supervised run remains pending.

Frozen candidate IDs are
`phase2b_formal_covert_semantics_39c7e1fddbaa05232b425da271b2403d30327971e4ddc7fae7c46012435fa4a0`
and
`phase2b_formal_covert_policy_binding_cb421f2b625865f10b6c82b21b6131bbf2433a85a3ac4d6ea4608f2c22ef8240`.

## Closed entry point and trust boundary

The only audit entry point is
`run_formal_covert_audit_v1(envelopes, authority, wire_adapter,
recognizer_adapter)`. It does not accept a caller-provided alignment,
caller-produced
invariance receipts, recognizer outputs, statistical thresholds, permutation
seeds, or feature lists.

Before any execution the module atomically reserves one of eight live
process-local receipt slots. Issued receipts are retained until the caller uses
`release_formal_covert_audit_receipt_v1`; live receipts are never silently
evicted. A full registry rejects before decode or recognizer execution. This
underscore-token/factory/object-identity mechanism is cooperative same-process
provenance only, not unforgeability or a security boundary.

`FormalAuditAuthorityV1` contains private source-order labels and strata. Its
transcript root proves only the bytes used by this audit; it does not prove
origin or independence. `FormalWireV3AdapterV1(build_success)` accepts only an
exact wire-issued `FormalTrustedWireBuildSuccessV3`; the public batch contains
neither its private alignment nor its private variant context. The adapter
retains the full build success and replays its alignment and variant context;
neither private object is accepted as a standalone caller surface.
Private semantic inputs are canonical RFC-8785 label bytes. Wire projects any
source UUID references through the exact per-row source-to-public mapping,
issues sealed public-order private rows, and binds each row's envelope digest,
source index, source-case commitment, label-binding ID, projected bytes, and
row-binding digest. The auditor compares every field and commits the public
label-projection root. Swapped, stale, mapping-mutated, or spliced labels reject
before recognizer or statistics. These checks prove linkage, not label truth or
origin authentication.

Before calling any adapter, the auditor captures every authority primitive and
reconstructs an isolated canonical authority copy. It replays that copy after
the adapter returns and rejects any mutation. All downstream alignment,
statistics, and receipt roots use a second fresh reconstruction that was never
exposed to the adapter. A frozen dataclass alone is not treated as protection
against `object.__setattr__`.

Parent-side adapter properties, private-authority binding, and public decode
are a cooperative trusted computing base. This module hash-pins the Wire V3
STOP bytes, but a recognizer or other adapter's caller-supplied implementation
ID is only recorded and required to remain stable; it is not a verified content
hash. External qualification must bind the exact recognizer and any other
adapter bytes through its signed or CAS artifact. The raw fork isolates the 81
suite/recognizer callbacks for liveness, resource control, and copy-on-write
Python state; it is not an integrity sandbox against malicious callback code,
which could alter child-local comparison or artifact-writing logic. Neither
the fork nor the process-local receipt registry proves origin or
unforgeability.

The adapter reconciles every Wire V3 `CHANNEL_ONLY` concrete leaf with exactly
one owner among UUID, missingness, provenance, record order, component-slot
order, eligibility mask, and control metadata. Zero or multiple owners reject.
Control metadata includes batch context, schema/policy/field-manifest IDs,
transform-catalog root, collision count, run commitment, graph schema, and
candidate capacity. Every UUID-shaped public leaf must be canonical UUIDv4 and
must belong to exactly one of the frozen ten namespaces. An unmanifested UUID
rejects.

Each channel projection separately binds the raw public path for integrity and
the closed statistical value. Record-order statistics use only ordered record
identity tokens, component-slot statistics use only ordered axis tokens, and
eligibility statistics use the wire's typed mask tokens; they never hash an
entire evidence-bearing record array and relabel it as an order channel.

## Auditor-generated variants

One baseline plus 32 global namespace-wise bijective UUID renamings, 32 global
case-order permutations, and 16 within-case observation-order permutations are
constructed and executed inside the auditor: 81 recognizer batch calls in
total. The 32 case-order schedules are pairwise distinct and non-identity.
Every mutation is an explicit `FormalVariantScheduleV3` executed only through
the full `FormalTrustedWireBuildSuccessV3`; its private
`_FormalVariantContextV3` remains build-success-owned and is never accepted as
a standalone input. Raw payload/root-only re-encoding is not an adapter
surface. Wire validates inverse semantic and
catalog replay and recomputes all content-derived provenance. After inverse UUID/order mapping,
decision, family, binding, scale set, and candidate residual bytes must be
exactly identical. Any structural or invariance failure is atomic.

All 80 variant schedules are submitted in one bounded Wire V3 bulk-suite call
inside a raw `setsid` child. The adapter replays the shared BuildSuccess,
variant-context, private-alignment, and bound-envelope state exactly once for
that atomic call; every one of the 80 plans and resulting schedule DTOs is
still replayed independently. The unchanged single-batch API performs the
same bound-state replay on every call. Every returned envelope is publicly
decoded and passed through the auditor's decoded-shape validator before
recognizer use. The pinned public decoder accepts only raw envelope bytes;
all decoded fields and false claim flags are freshly derived in one union
validation with exactly one canonical JSON parse. This removes a duplicate
DTO cache replay without caching, deduplicating, or bypassing any of the
`rows * 80` public decoder or auditor-validator calls. The child must produce
80 batches, `rows * 80` consumer
decodes and decoded-shape validations, 81 recognizer runs, and `rows * 80 * 5`
comparisons. The parent owns
the PID through pidfd/waitid/wait4. At 7,200 seconds the parent begins
termination, permits at most five seconds of TERM grace, then KILLs and reaps;
this nominal phase deadline is not an end-to-end 7,200-second termination
guarantee. The parent replays one closed RFC-8785 artifact from a retained
no-follow directory FD.
The parent-computed SHA-256 is a transcript content binding after canonical
replay, not an independently authenticated child commitment.

## Field and feature closure

The unique-ID family contains all 128 individual bits; first and last 8, 16,
and 32 bit decompositions as explicit semantic aliases of those same bits;
Hamming weight; integer residues modulo 3, 5, 7, 11, and 13; and all 16
hex-character histogram counts. Only those explicit prefix/suffix bit aliases
are deduplicated. Observed-equal but semantically distinct fields remain
distinct hypotheses in the global family.

Every union UUID path has a separate presence hypothesis. Absence uses an
internal all-zero 128-bit block, which is outside the valid UUIDv4 domain, while
the full ID feature family still executes. Provenance union paths likewise have
a presence hypothesis and an internal all-zero 32-byte absent value. Other
variable channel paths use separate presence plus tagged present/absent value
hypotheses.

Padding, envelope, and provenance prefix/suffix bytes retain raw-byte
categorical hypotheses. Each audited byte also has eight individual-bit, 28
within-byte pair-XOR, high/low-nibble categorical, and byte-Hamming hypotheses.
These supplements are not claimed to be mathematically equivalent to the raw
256-category test. Padding additionally retains length, total Hamming weight,
and all 256 byte-histogram counts.

Semantic descriptor text remains evidence-bearing; the auditor does not hash
the whole text or treat lexical identity as a channel category. It registers a
separate descriptor-metadata family covering descriptor kind, segment count,
segment presence and order position, UTF-8/code-point lengths, leading and
trailing ASCII spaces, first/last boundary shapes, and separator runs. It also
records the complete ordered ASCII-class and Unicode general-category
sequences; the exact ordered `(position, codepoint)` sequence for allowed
whitespace; whitespace run start, length, and code-point sequences; whitespace
and run counts; ordered lexical-token code-point/UTF-8-length and token-shape
vectors; all class/category/shape and token-length-bucket counts; and whether
NFKC changes the segment together with its code-point and UTF-8 length deltas.
No normalized text is retained.

Each descriptor additionally contributes one ordered `representation_composite`
categorical hypothesis made from the canonical metadata projection of its kind
and segments. This pre-registered joint feature closes the two-marginal/XOR
blind spot. It contains no descriptor text, whole-text hash, lexical identity,
or exact punctuation/symbol identity; exact code points occur only for allowed
whitespace. Consequently `alpha` and `omega` have the same representation
projection, as do punctuation/symbol substitutions with the same frozen
classes and categories. Descriptor feature counts are derived from the closed
alias manifest and are never asserted as a fixture-specific constant such as
137.

The validator is pinned to Unicode 13.0.0. It binds UCD 13
`DerivedCoreProperties.txt` SHA-256
`a5d45f59b39deaab3c72ce8c1a2e212a5e086dff11b1f9d5bb0e352642e82248`,
`PropList.txt` SHA-256
`485b5a3ed25dbf1f94dfa5a9b69d8b4550ffd0c33045ccc55ccfd7c80b2a40cf`,
4,173 Default_Ignorable_Code_Point values, and canonical range-table SHA-256
`629f6c5f9f20e8322371ff59b61ce34246a9c4afb8fd299756540058dc8a553e`.
The generic adapter mirrors Wire's exact closed object, nonempty segment,
RFC-8785, NFC, and UTF-8 bounds; it rejects categories `Cc`, `Cf`, `Cs`, `Co`,
`Cn`, `Zl`, and `Zp`, all frozen default ignorables, Unicode noncharacters,
UUID tokens, `sha256:` tokens, and standalone 64-hex tokens. Under UCD 13,
U+2000 and U+2001 normalize to U+2002 and U+2003 respectively, so the former
pair is an atomic non-NFC rejection KAT. The NFC-stable U+2002/U+2003 pair is
the positive exact-whitespace covert-leak FAIL oracle. Trailing-space and
two-marginal XOR-composite probes are also required FAIL oracles.

The public payload recursively proves the frozen MUST-BE-ABSENT fields,
including filename/source-order artifacts, private answers, per-case transform
catalogs, challenge/order/overlap/private-truth labels, and unused-transform
lists. The batch-global transform-catalog root
is required control metadata and is not forbidden.

## Eligibility-mask quota gate

For each row the mask signature uses only presence and count for observation,
role-candidate, and scale namespaces; member order and ID bytes are excluded.
For family, scale, and answerability separately, every signature-by-class cell
must satisfy the exact integer equality

`count(signature, class) * N = count(signature) * count(class)`.

No floating tolerance or adaptive balancing is permitted. A quota failure
rejects atomically before recognizer execution. The quota receipt commits the
public signatures and private contingency tables without asserting label
origin. ID bytes remain covered by the UUID feature audit, and mask features
remain registered hypotheses for targets such as binding. The preregistered
synthetic binding-only mask-leak oracle is required and observed to return
FAIL. This is not a universal detection claim: a smaller effect or finite-
sample realization that does not meet the E1 materiality, target-Holm, and
localization conjunction may return PASS.

The accepted-corpus selection event is also part of the randomization design.
For family, scale, and answerability, the exact structural signature is
appended to the target's pre-existing permutation stratum. Consequently every
permutation preserves every signature-by-class quota cell. Sufficiency is
recomputed on these conditioned strata. Binding and joint-decision
permutations do not alter any label read by the quota gate, so they retain
their pre-existing strata; this also preserves the binding-only eligibility
leak negative control.

## Frozen statistics

There are five private targets: family, binding, scale,
answerable-versus-abstain, and joint decision class. Each target uses its
pre-registered private strata. Insufficient classes, leave-one-out support, or
mutable strata return an atomic insufficient receipt without recognizer or
statistical execution.

For every target-feature pair, the auditor computes raw normalized mutual
information and exact categorical leave-one-out balanced-accuracy advantage.
The caller cannot supply a random seed. Before reading authority or envelope
content, the auditor draws 32 bytes from the operating-system CSPRNG, commits
the seed, derives target/stratum shuffles through the frozen SHA-256 PRF
schedule, and later reveals the seed in the receipt. For each target the same
within-stratum label permutation is shared across every feature.

This schedule is a computational randomization construction. Neither the
content commitment nor the public receipt authenticates seed independence, and
the implementation does not claim unconditional exact Monte Carlo inference.
Its target-family error statement is conditional on label exchangeability
within every pre-registered private stratum (including the quota-conditioned
signature strata above) and on the declared CSPRNG/PRF randomization
assumption.

For a target-feature pair, let `Z_0` be the observed NMI and `Z_1,...,Z_10000`
the permuted NMIs. The bias center is completely symmetric:

`C = (Z_0 + Z_1 + ... + Z_10000) / 10001`.

Every nonnegative binary64 NMI is decoded to its exact integer multiple of
`2^-1074`. The implementation adds those integers on that lattice in a
17-word/1,088-bit accumulator (including carry capacity for 10,001 terms),
divides the exact rational by 10,001, and performs one
pinned-runtime correctly-rounded binary64 conversion. Thus the center is
bit-identical when any member is rotated into the observed position;
observed-first floating-point accumulation is forbidden. The same `C` is used
for the observed and all 10,000 permuted omnibus statistics. The omnibus is

`max(bias_corrected_NMI / 0.02, max(0, BA_advantage) / 0.05)`.

For each target and exchangeable index `b`, define

`M_b = max_feature S_(b, feature)`.

The target-family maxT value is

`p_target = (1 + count_{b=1..10000}(M_b >= M_0)) / 10001`.

Ties are always in the tail. Exactly five target-family p-values enter one
exact-rational Holm procedure at family-wise alpha 0.01; cross-target
dependence is allowed. The five inferential hypotheses are complete-target
nulls: for a target, all registered features are null under its declared
quota-conditioned strata. Holm controls family-wise error across those five
complete-target nulls; it does not establish partial-null or per-feature
strong FWER. For localization only, a feature records

`p_loc = (1 + count_{b=1..10000}(M_b >= S_(0, feature))) / 10001`.

`p_loc` is a non-inferential witness, not a per-feature partial-null p-value or
a strong-FWER claim. An executed audit returns FAIL if and only if one and the
same feature has `S_observed > 1`, its target family is rejected by the
five-target Holm procedure, and `p_loc` is strictly below that target's Holm
critical value. Every equality is PASS. Otherwise the sole permitted positive
decision text is
`NO_MATERIAL_COVERT_CHANNEL_DETECTED_AT_TARGET_FAMILY_FWER_0.01`. It does not
mean that no leak of any size exists.

The original 10,000-permutation target-feature p-values and their Holm replay
over all target-feature pairs remain in the transcript solely as a
resolution-limited diagnostic. They are explicitly `decision_eligible=false`.
For example, with 84,315 hypotheses their smallest possible raw value
`1/10001` cannot approach the first Holm critical value, so that replay cannot
serve as a decision gate.

Metric DTOs reject non-finite or derived-value drift. All p-values, adjusted
p-values, and critical values are reduced integer fractions. Receipt IDs encode
binary64 metrics with `float.hex()`. This is pinned-runtime replay evidence,
not a cross-runtime binary64 equivalence claim.

For each unique statistical partition, the observed `(NMI, BA)` pair and the
ordered 10,000 permuted pairs are computed once. The permuted pairs occupy one
bounded ephemeral tuple; the symmetric center and pairwise diagnostics are
computed in frozen order, while a length-10,000 vector accumulates the shared
target maximum. The tuple is released before the next partition. This is one
intersection pass plus scalar scans; work accounting uses `P+1=10001`, never
`P`. It does not reduce permutations, concrete features, raw-byte categorical
hypotheses, or targets.

CPU accounting separately records canonical-center work as
`unique_partitions * 5 * (10001 * 17 + 17^2)` 64-bit-word operations, covering
all 10,001 additions plus a quadratic exact final division/rounding allowance rather
than hiding them inside one scalar scan. SHA-256 rejection sampling is also
bounded: each `randbelow` permits at most 4,096 64-bit draws and then rejects
atomically. The resource receipt binds the worst-case label-shuffle call count
`10000 * 5 * max(0, rows - 1)` and its corresponding rejection-draw upper
bound.

Executed receipts must carry nonempty invariant and permutation roots, exactly
81 recognizer batch calls, and exactly `rows * 80 * 5` bit-exact field
comparisons. Receipt DTO construction is not execution issuance: validation
requires identity in the bounded cooperative process-local registry. That
check is not designed to resist same-process code with module access. The
receipt content hash is neither a signature nor cross-process authority.
Durable qualification must run in the pinned isolated executor and bind the
artifact through a separately authenticated signed or CAS record.

## Resource and failure boundary

The hard input boundary is 1,024 rows; 1,025 rejects before decode. The current
caps are 32,768 concrete features, 20,000,000,000 bitset intersections,
50,000,000,000 total statistical work units, and 12 GiB conservative aggregate
SELF-plus-worker RSS. Total work includes `P+1` contingency intersections,
17-word center accumulation/finalization, scalar maxT scans, and bounded label
shuffle calls. Aggregate statistics RSS is

`max(parent current RSS, label masks + feature matrix + envelopes/decodes +
partition matrix + retained adapter graph + 64 MiB) * (1 + workers)
+ metric cache * workers + ephemeral permutation metric tuples
+ target-max vectors + worker artifacts + parent result working graph`.

This deliberately counts fork-shared pages once for the parent and once per
worker. Feature, decode, partition, adapter, and result terms are explicitly
bound. Recursive shared-object-aware `sys.getsizeof` observations are
multiplied by four where the profile says so; this is pinned-runtime accounting,
not a portable Python heap theorem. Feature/path projection enforces the
32,768 cap incrementally before column materialization. Per-column cardinality
and partition allocation are projected before the single materialization pass,
which has a separate 600-second cooperative deadline.

The cgroup parser reads `/proc/self/cgroup` and `/proc/self/mountinfo`, applies
mountinfo's filesystem-root-to-visible-mount projection (including subtree and
combined v1 mounts), reads each limit/usage or quota/period pair from the same
hierarchy level, walks every ancestor, and takes the tightest finite value.
Leaf `max` never hides a finite ancestor. Module preflight uses the minimum of
that value and Linux `MemAvailable`, plus a 2 GiB margin. A durable
qualification requires a fresh finite cgroup-v2 memory/pids scope supplied by
the external supervisor; a host-only `None` cgroup observation is not R05
execution evidence.

Statistics use exactly one or five raw `fork` children. Five are selected only
when OS CPU count, scheduler affinity, the tightest cgroup quota, projected
work, and the full five-worker RSS model all permit them; otherwise one child
executes targets 0 through 4 sequentially. Failure to read scheduler affinity
conservatively forces the one-child path. There is no direct unbounded path.
Python `SIGCHLD=SIG_IGN` is rejected before either raw fork because automatic
reaping would destroy PID ownership. After the selected lifecycle and
cancellation signals are blocked, the parent temporarily sets SIGCHLD to its
default disposition through the auditor's own wait4, then restores the caller's
original non-ignored disposition before unblocking. Other parent signal
handlers are cooperative TCB. External qualification requires a fresh process
with default signal dispositions in addition to its cgroup/pids supervisor.
Before the first fork, ownership slots and reap markers are preallocated and
the selected lifecycle/cancellation signals are blocked; each successful PID is stored immediately,
so a later fork or pidfd failure still terminates and wait4-reaps every child.
Each child uses `setsid`, closes every non-stdio FD except the retained artifact
directory FD, sets a hard address-space limit, and emits fixed-name closed
RFC-8785 artifacts. The parent accepts exactly five files through retained
directory-FD, no-follow, regular-file, single-link, size, canonical-schema,
finite `float.hex()`, and decimal-range checks. No child result object or
pickle crosses into the parent. Parent-computed SHA-256 values bind canonical
replayed content into the receipt; they are not transport authentication.
Ordinary success and writer-failure paths remove their private artifacts. If a
same-UID actor replaces the path or inserts an unexpected subdirectory,
cleanup fails closed without following the replacement and without recursive
deletion; it may leave one bounded forensic temporary directory. A fresh
external qualification namespace owns final cleanup of such leftovers.

At 1,800 seconds the raw statistics worker supervisor begins termination,
permits at most five seconds of TERM grace, then KILLs and reaps. This is a
nominal phase termination trigger, not a promise that every child is gone at
exactly 1,800 seconds. pidfd/waitid-with-`WNOWAIT` keeps PID/process-group
ownership until descendant cleanup and exactly-once wait4, whose rusage
supplies full direct-child lifetime peak RSS. Parent result construction,
receipt replay/hash, and RSS stabilization perform absolute monotonic checks
against the same deadline, but those checks are cooperative and explicitly
`parent_postprocess_deadline_preemptive=false`.
Finalization permits at most three complete construct/replay/provisional-
register/validate passes. If validation raises historical RSS, the provisional
registry entry is removed, the same reserved slot is restored atomically, and
the next receipt binds the new peak; failure to stabilize rejects without a
live receipt.
The implementation does not replace or consume the caller's SIGALRM state.
End-to-end preemption, descendant containment, aggregate memory, and pids
limits therefore require a fresh isolated outer supervisor/cgroup; the receipt
sets `external_supervisor_required=true`. The conservative runtime sum uses
the parent's historical `ru_maxrss` plus each direct raw child's wait4 peak and
rejects above 12 GiB. It is not observed physical PSS.
`process_group_empty_at_exit=true` covers the direct child's setsid process
group at the final check only; it is not continuous descendant evidence and
cannot detect a helper that already exited or escaped into a new session.

Wire's logical suite output is `rows * 80 * 65,536` bytes, while the pinned
bulk-suite reuse contract bounds resident identity-aware content by at most 49
unique-envelope equivalents per row plus 80 reference slots and a fixed
allowance. The receipt records both logical coverage and observed unique-object
graph size. This reuse bound makes the 1,024-row resource formula reachable;
the unit KAT that demonstrates arithmetic admissibility uses synthetic
partitions and injected bounds and is not an executed R05 capacity result.

Any future 1,024-row R05 qualification is valid only for the qualification
profile's frozen rich-graph shape/path bound. The wire schema may admit substantially
larger record graphs (up to its own bounded record limits); this auditor does
not claim that worst-case Wire V3 graphs fit the qualified capacity. Every
actual R15 input must be freshly decoded and preflighted, and may fail closed
on feature, work, host-memory, or envelope limits.

## Claim boundary and next hard gate

Synthetic known-null PASS, known UUID/padding-XOR/eligibility/presence leak
FAIL, exact quota rejection, malformed alignment rejection, and resource-plan
formula KATs can test this implementation. No rich-V3 1,024-row statistics
capacity qualification is claimed by this document. These checks do not
authenticate authority origin, establish independent key sources, load or
audit the actual corpus, generate actual 720+240 or 496+76 material, invoke
Docker, run the actual recognizer/scorer ceremony, or authorize reveal.

R04/R05 may be marked only by a separate integrated qualification artifact
that binds the final source bytes and completes the exact rich-V3 1,024-row
run under a fresh finite cgroup-v2 memory/pids scope and external hard-deadline
supervisor. R15 and the later recognizer commitment/reveal/scorer sequence
remain separate hard gates.
