# Narrow independent confirmation — rejected

Frozen 0.179.0 and 0.178.0 completed four matched continuations and 56 terminal scores, with zero external API calls. Candidate passed 21/28 versus 23/28; three root07 failures and one root09 recovery. It saved 48 rows and 99,470 compute but reduced quality. The planned ten-root extension was not started.

`compact_results.json` contains all costs and quality. `paid_regression_diagnosis.json` traces the lost targeted confirmation at root07 p7, followed by missed repairs at p2 and p8. Candidate runtime and negative evidence remain archived; active production was restored to 0.178.0. Candidate tests: 43; restoration tests are reported separately by the parent.

The separately authorized budget-reserve correction is recorded in `correction/`; it does not overwrite this result.
