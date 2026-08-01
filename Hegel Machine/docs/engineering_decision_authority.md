# Engineering decision authority

Effective date: 2026-08-02

This file records the project owner's standing instruction for Hegel Machine
work. Web-chat GPT output is advisory: it may propose goals, plans, design
directions, or reviews, but it does not become an exact wire or implementation
decision merely because it is written in answer form.

## Precedence

When inputs conflict, use this order:

1. the project owner's explicit objective, claim boundary, and authorization;
2. already published formal identities and externally signed evidence that the
   owner has not explicitly superseded;
3. executable cross-language behavior, strict parsers, golden vectors, tests,
   and replayable repository evidence;
4. Codex's documented engineering decision, including any minimal closure
   amendment needed to remove an implementation ambiguity;
5. webpage GPT plans, proposed field layouts, estimates, and review comments.

The higher item controls. A lower item may expose a defect in a higher item,
but it cannot silently rewrite it.

## Operational rule

- Treat webpage GPT material as design input or review, not as automatically
  normative byte-level truth.
- Resolve local implementation details from the actual repository, both
  language implementations, strict negative cases, and reproducible tests.
- If an advisory answer leaves two executable interpretations, select the one
  with the stronger identity, trust, fail-closed, and replay properties; record
  the decision in a versioned addendum and a regression test.
- If a proposed detail cannot pass the frozen engineering checks, reject or
  amend that detail instead of weakening the checks to preserve the prose.
- Do not use this policy to broaden authority. External custody, signatures,
  sealed data, formal roots, gate transitions, ACTIVE governance, and other
  owner- or actor-controlled actions still require their specified evidence.
- Never convert a diagnostic or candidate value into a formal claim because a
  plan says the stage is complete. Status follows produced evidence.

## Required record for a material engineering override

A material override of advisory detail must leave all of the following:

1. the conflict and the selected interpretation;
2. the identity/security reason for the selection;
3. exact code or schema changes;
4. positive and negative regression vectors;
5. the resulting claim and authority boundary;
6. a local commit, and a push when the owner has authorized publication.

For the Phase-3A M2.5 E1-E12 work, the first such record is
[`Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md`](Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md).
