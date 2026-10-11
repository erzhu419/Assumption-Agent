# Same-budget source-feedback execution

This is the next step after the action-divergence receipt.  It uses the real
`failed_successor` and `source_competition` fixtures.  Both arms receive the
same two already-screened offers, the same paid predecessor, the same target
laboratory, and the same continuation limits.  The candidate arm retains the
measured source returns; the matched arm removes only those `research_return`
records.

The candidate arm withdrew `concept_0001` and executed the repaired
`concept_0002`.  It completed with paid-revisit loss `6.28e-20` and held-out
later loss `9.73e-20`.  The matched arm executed `concept_0001`, remained
unresolved, and had paid-revisit loss `1.33e-3` and held-out later loss
`2.75e-3`.  Both arms paid exactly 24 observations, zero measurement cost,
120 continuation compute, and five queries.

The execution goes through `UnifiedResearchSession._perform_revisit`; no
rows, gains, or returns are injected.  Both arms used zero external model
calls.  This establishes same-budget action-to-outcome causality for the
fixture.  It is still one deterministic conflict fixture, so it does not yet
establish multi-task long-run net benefit or average performance.

Run:

```bash
PYTHONPATH=src:tests python3 scripts/run_source_feedback_same_budget.py
```

