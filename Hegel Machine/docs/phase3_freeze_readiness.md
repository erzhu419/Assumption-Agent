# Phase-3 old-language freeze readiness

## Decided and machine-frozen

The repository now freezes the decided surface of the old DSL:

- the primitive sorts, leaves, operators and forbidden symbol IDs;
- maximum relation arity 3 and `EntitySet` size 8;
- AST depth 4, at most 3 top-level conjunction clauses;
- old-law composition depth 2;
- at most 3 fitted parameters and 2 scope clauses;
- at most 50,000 syntactically canonical old-language programs;
- four possible adequacy verdicts:
  `IN_LANGUAGE`, `OUTSIDE_FROZEN_CLOSURE`, `INCONCLUSIVE_BUDGET` and
  `INCONCLUSIVE_SEMANTICS`;
- exact `Fraction`-based MDL arithmetic precheck over caller-supplied lengths;
- shadow-only operation with ACTIVE promotion disabled.

All closure receipts are currently untrusted wire records. Even after every
content binding is filled, no outside verdict is possible until a sealed
evaluator independently replays the archive and recomputes closure/target
roots. That verifier is a machine-readable blocker and certificate issuance is
hard-disabled. The four verdict values freeze the future result vocabulary;
they do not mean the current receipt parser can issue semantic verdicts.

## Parity finding that changes the target specification

The allowed old DSL contains `difference`, `absolute` and `approx_equal`.
Consequently binary XOR is already expressible:

```text
abs(x - y)
```

with truth table `0,1,1,0`. Three- and four-input parity also have shallow
nested absolute-difference constructions. Therefore banning tokens named
`XOR`, `modulo` and `parity` does not by itself put a parity-like target outside
the old language.

The implementation records the XOR2 truth table only as a target-design sanity
under intended standard numeric semantics. It does not parse or execute a
frozen old-DSL AST, so it is not a formal `IN_LANGUAGE` closure verdict. A
genuine outside target must be specified over a larger bounded
`EntitySet` (the proposed range is 5–8 elements) and then checked against the
complete extensional closure. Target naming or intuition cannot substitute for
that enumeration.

## Hidden-sink null control

The hidden-sink control is valid only if the sink flow is already present as an
opaque typed measurement but omitted by the initial scope or aggregation. The
old conservation law must recover it by a preregistered scope/aggregation
refinement.

If the sink is genuinely latent and unobserved, the current old DSL has no
latent-variable leaf, so the case cannot be preregistered as an in-language
null. A null-control receipt with no exact old-language match is fail-closed as
`INCONCLUSIVE_SEMANTICS`; it may never be promoted to an outside target after
seeing the result.

## Remaining blockers

The default contract exposes the following unresolved content IDs:

- exact rational grid;
- complete bounded sort universe and vocabulary;
- total executable operator semantics;
- extensional/algebraic equivalence contract;
- canonicalizer implementation;
- deterministic closure enumerator and traversal order;
- complete MDL token/identifier/new-symbol code table;
- exact higher-arity parity task;
- observed-but-omitted hidden-sink task and old-language witness;
- independent hidden generator specification.
- sealed closure archive schema, independent replay/root recomputation verifier
  and trusted evaluator attestation;
- sealed MDL scorer that binds the scoring partition and recomputes lengths
  from the frozen program/data/code table.

Until these are frozen, Phase-3 implementation infrastructure may be developed,
but Phase-3A has not started its formal experiment and no language-inadequacy
claim is available. `MdlGainReceipt.numeric_threshold_passed` remains a pure
arithmetic diagnostic and the formal `mdl_gain_gate` remains false.
