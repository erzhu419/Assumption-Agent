# Representation ambiguity: initial cohort and continuation complete

Eight episodes cover two parameter pairs, finite-history/recurrent physics and Code/Hegel, using gpt-6-luna and the same six-source archive. All five token-truncated branches resumed; three initial successes were retained. Only the cumulative token allowance rose from 1.2M to 4M; 48 calls, 576 observation rows, 300 Python seconds and 96-row confirmation remained cumulative.

| Arm | Initial finite / recurrent passes | Final finite / recurrent passes |
| --- | --- | --- |
| Code | 2/2 / 0/2 | 2/2 / 0/2 |
| Hegel | 1/2 / 0/2 | 2/2 / 0/2 |

Hegel case003 needed one additional finish call: its existing three-lag predictor passed confirmation and terminal scoring (MSE 7.5566e-31). Both arms' finite-history successes include actual representation expansion.
Recurrent case001: Hegel found a precise meter-history submodel but did not compose it into a response predictor, then exhausted calls. Code revised from paid counterexamples; its last fresh MSE was 5.7585e-10, above the fixed 1e-10 threshold.
Recurrent case002: Hegel's full-history quadratic reached fresh MSE 2.7593e-8 and exhausted confirmation observations. Code's MLP had paid MSE 7.1629e-5; broad cross-validation exhausted Python time, so subsequent finish attempts did not execute.
No child research, experiment-comparison or formal revision-proposal calls occurred. Ordinary Python models were revised; Code case001 also executed one unchanged archive scratch script. Neither arm completed a verified compact recurrent-state model.

| Cost | Initial eight | Added continuation | Cumulative eight |
| --- | ---: | ---: | ---: |
| Model calls / reported tokens | 135 / 8,532,327 | 78 / 6,279,197 | 213 / 14,811,524 |
| Observation rows / meter cost units | 2,112 / 216 | 960 / 360 | 3,072 / 576 |
| Cached input tokens / calls with hits | 2,554,112 / 108 | 1,207,040 / 73 | 3,761,152 / 181 |

Costs cover these inquiry episodes and exclude prior learning of the six-source archive. Full costs, phase distinctions and per-episode evidence are in `analysis.json`; original summaries remain unchanged.

## Limitations
Two parameter pairs and a development continuation cannot establish Hegel superiority, recursive advantage or transfer. Full-history success within the 24-step horizon does not identify an unbounded recurrence. Failed fresh scores are not terminal scores; unsubmitted recurrent models have no terminal score. Precise meter fitting and purposeful probing did not establish a composed response model or scored prospective candidate discrimination.

Next: reuse historical target-specific scoring for actor-authored free-Python submodels and connect child evidence to parent decisions; this is planned, not implemented, and supplies no host-written scientific submodel. The post-batch Python-budget early-stop change does not alter these frozen results.
