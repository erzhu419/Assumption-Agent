# Phase-2B formal trusted-wire/covert qualification scaffold v1

**Status:** `SYNTHETIC_IMPLEMENTATION_QUALIFICATION_SCAFFOLD_FROZEN`
**Claim level:** `SYNTHETIC_DEV_KAT_ONLY_NOT_FORMAL_WIRE_NOT_FORMAL_CORPUS_AUDIT_NOT_EXECUTION`
**Profile ID:** `phase2b_formal_trusted_wire_covert_qualification_profile_v1_de3f7edfacc8115c565b30a60bd285500cb0027cf4e7640749c8c1b954ae5c63`

## 1. Exact result

This slice freezes a closed profile and a deterministic two-authority
development known-answer test. It exercises the existing public V2 trusted
wire builder, public typed replay, fixed 65,536-byte envelope decoder,
namespace field-manifest parser, and frozen covert-audit structural mechanics.
It also requires complete caller-supplied receipt coverage for 32 global
renamings, 32 case-order permutations, and 16 observation-order permutations.

The expected covert result is `ABSTAIN`: two development rows cannot satisfy
the frozen metadata-predictivity sufficiency requirements. No permutations are
executed, no metadata-predictivity gate passes, and no formal audit evidence is
issued. This is a qualification scaffold for already existing mechanics, not
closure of the exact-freeze wire/auditor prerequisite.

The canonical profile is
`config/phase2b_formal_trusted_wire_covert_qualification_v1.json`. Its source
manifest binds the critical transitive Python inputs used by the KAT to
expected SHA-256 values and records the pre-slice loaded-mechanics basis commit
`0cdf729e80f828cf2fc1cf2fd1ca67319103d829`. The pure qualification entrypoint
does not read or rehash files. Independent tests perform the live file rehash;
therefore the receipt permanently keeps `source_files_rehashed=false` and
`git_commit_live_verified=false`.

## 2. Qualified synthetic surfaces

The KAT may claim only that these development mechanics completed:

1. closed canonical profile parsing, duplicate/float/type rejection, and a
   content-addressed profile;
2. the schema-closed ASCII/safe-integer accepted-JCS subset roundtrip;
3. the fixed envelope and closed eight-field wrapper;
4. three pairwise-distinct deterministic KAT IKM inputs accepted by the
   existing purpose-separated key schedule;
5. post-shuffle UUIDv4 allocation, namespace-manifest parsing, public typed
   replay, and exact-transform replay for the two frozen synthetic authorities;
6. structural equality over all caller-supplied 32/32/16 invariance receipts;
7. fail-closed `ABSTAIN` for insufficient metadata-predictivity rows; and
8. empty-root rejection for ordinary malformed inputs or post-build failures.

Pairwise-distinct KAT IKM does not establish independent key provenance. The
32/32/16 receipts do not establish that a recognizer was really rerun under
generated variants.

## 3. Formal blockers deliberately left open

The profile and every success receipt fix the following claims to false:

```yaml
full_rfc8785_domain_qualified: false
formal_wire_builder_implemented: false
formal_corpus_auditor_implemented: false
source_files_rehashed: false
git_commit_live_verified: false
key_source_independence_verified: false
metadata_predictivity_statistics_executed: false
metadata_predictivity_gate_passed: false
global_consistent_renaming_execution_verified: false
capacity_1024_authority_qualified: false
actual_corpus_loaded: false
actual_corpus_audited: false
formal_uuid_audit: false
formal_covert_audit: false
origin_authenticated: false
formal_auditor_authority_verified: false
recognizer_capacity_evidence: false
sealed_holdout_eligible: false
actual_execution_claimed: false
execution_authorized: false
formal_holdout_generation_authorized: false
exact_freeze_blocker_closed: false
c1_exit_evidence: false
```

The accepted-JCS implementation is a restricted RFC 8785-derived profile, not
a complete RFC 8785 implementation: it rejects non-ASCII strings and all
floating-point values. The existing covert statistics are capped at 64 rows,
so they cannot audit an actual 960-row corpus and do not qualify the 1,024-row
authority capacity boundary.

The formal auditor still needs closed evidence-bearing/channel-only leaf
classification over all ten namespaces; all UUID-field, missingness,
filename/archive-order, provenance, eligibility-mask, payload-length, padding,
unused-transform/slot, and derived-private-answer feature groups; and actual
recognizer reruns for every frozen 32/32/16 variant. The current missingness
wire shape and per-case transform catalog also conflict with the exact freeze.
ID-message framing, the whole-batch `STANDARD_ERROR_UNSUPPORTED` path, and a
cross-runtime replacement for the current Python binary64 statistics remain
unclosed.

## 4. Side-effect and authority boundary

`filesystem_accessed=false`, `docker_invoked=false`, `network_accessed=false`,
and `keys_generated=false` describe only the qualification entrypoint. The
test harness necessarily reads the profile and source files and constructs
deterministic fixtures outside that entrypoint. These fields do not make claims
about the caller, test runner, operating system, repository, or any future
campaign.

No campaign ID, attempt ID, holdout row, answer, prediction, score, signature,
Docker receipt, authority receipt, seal, execution gate, or Phase-2B/C1 exit
evidence is created by this slice.

## 5. Next implementation boundary

The next slice must define and test the missing formal leaf-classification and
feature-group contract, reconcile missingness and transform-catalog wire
representations with the exact freeze, close ID framing and standard-error
atomicity, and implement actual variant execution plus 960/1,024-scale
capacity. Only a later independent corpus qualification may change the formal
wire/auditor blocker; this scaffold cannot be promoted by relabelling.
