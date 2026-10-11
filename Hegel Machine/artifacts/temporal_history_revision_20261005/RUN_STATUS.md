# Temporal inquiry: completed pilots and one bounded repair

The corrected-interface gpt-6-luna pilots use identical task settings and limits
in separate development batches. Code succeeds on 1/2 sources and 0/2 targets;
Hegel succeeds on 0/2 sources and 0/2 targets. Unfinished tasks retain their token
budget outcomes. There is no demonstrated Hegel completion advantage.

The Hegel finite target fits its 72 short rows at MSE 4.15e-30, but after longer
experiments its training error is 0.00704 and fresh-confirmation error 0.01037.
Its subsequent search drops cross-history interactions and does not replace the
predictor. The recurrent target rewrites a retained experiment under a child
question, but buys no meter readings and fails to produce a revised predictor.

A developer-authored history revision now preserves all quadratic interactions,
adds one history coordinate, selects on complete paid trajectories, and exports
a predictor whose fit updates the actual coefficients. In the finite replay,
240 paid rows fit candidates and the final 48 paid counterexamples select the
extension. Its separately drawn 256-point diagnostic MSE is 6.81e-30. No new
model calls or training observations are used. Four operator tests pass.
The same operator leaves the recurrent case unsupported: its validation design
has rank12 for15 terms, and the retained model remains inaccurate.

Limitations: this repair was designed after inspecting failures. It is not an
LLM-discovered method, a fresh-confirmed agent result, or a change to pilot
scores. The Code and Hegel candidate_diagnostic.json files separately evaluate
explicit unsubmitted candidates after all training stopped; those scores remain
outside registered outcomes. Source implementation errors and procedural
noncompletion are not recast as prediction failures.

Next: invoke the repaired procedure in a prospective research loop; when the
available evidence cannot support it or residuals persist, investigate hidden
instrument state through distinguishing measurements. Keep this task fixed
while establishing that loop before expanding the three-arm comparison.
