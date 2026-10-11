# Portfolio local-admission ablation

Complete3 new continuations and42 terminal scores. Baseline and unrestricted portfolio results are reused; no prior arm reruns.
The experimental0.176.0/admission-ablation copy changes exactly one predicate from eligible to buy. Production source was not modified by this study.
Admission39/42, MSE0.00560209; baseline37/42,0.02251209; unrestricted38/42,0.01419372. Last7:19/21 versus18/21 in both controls.
Both baseline failures root01p9/root02p1 recover; unrestricted's extra failure root01p10 also recovers. No new failure versus baseline.
Actual visits33 versus50/69; independent positive fresh comparisons2 versus0/2. No locally declined purchase is admitted.
Total compute10768335 versus10674455/12295680; suffix rows3936 versus3072/5280. Recorded CPU109.80 versus129.32/203.52seconds.
Exact changes and all matched outcomes are in admission_ablation.diff, freeze.json and development_summary.json.
The unchanged admission runtime now feeds the separate fixed roots04–12 extension.

Limitations: this mechanism comparison was chosen after seeing the three-root development result. It establishes a local distinction, not broad generalization. Terminal scores and fees use the same endpoint; CPU is a single measurement per arm.
