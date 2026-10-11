# Fixed portfolio extension — completed

Roots04–12:18 continuations and252 terminal scores completed, using the unchanged0.176.0/admission-ablation runtime versus0.173. All runs were offline; learning finished before terminal evaluation.
Together with roots01–03 development, the selected comparison contains24 continuations and336 scores. The rejected unrestricted arm remains separate (3 continuations/42 scores).
Combined:139/168 versus137/168 passed; MSE0.022013 versus0.043306; observations+4695 and compute+1300649. Extension alone:100/126 in both; six recoveries and six new failures, chain W/T/L5/1/3.
`combined_summary.json` contains exact paired results, failures, costs, CPU and historical protocol differences. This study uses real0.168 visit14 paid prefixes and terminal offset1000003; the older0.173 study used offset920404 and a different learning start, so cross-study MSE is not a matched comparison.
Production adoption as0.176.1 is maintained by the parent task. No additional experiments are pending.
