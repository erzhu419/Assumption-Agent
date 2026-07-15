---
name: financial-sec13f-contract-operator-v2
description: Solve public SEC Form 13F comparison tasks with exact typed contract semantics.
---

# SEC Form 13F contract operator

For a task that compares two official SEC Form 13F periods, follow the task's
declared data roots and semantics exactly.

1. Select the globally latest `REPORTCALENDARORQUARTER` in each period, then
   exclude filings whose `REPORTTYPE` contains `NOTICE`; do not fall back to an
   older date.
2. Normalize manager and issuer identities with Unicode NFKC, case folding, and
   ASCII alphanumeric token boundaries. Match identities exactly. A selected
   manager must have exactly one eligible accession in every required period.
3. Keep `VALUE` exact during aggregation. Normalize CUSIPs by uppercasing and
   removing whitespace. Apply only the exact stock-title set declared in the
   instruction.
4. Aggregate before ranking. Use descending value and ascending canonical
   CUSIP or normalized manager as the deterministic tie-break.
5. Write only the requested ordered keys to `/root/answers.json`, and verify the
   file locally without inspecting benchmark verifier or expected-answer data.

The runtime may apply the same typed public-contract computation after the
agent finishes and before offline verification. It uses no network service,
model, hidden verifier content, gold answer, or sealed benchmark content.
