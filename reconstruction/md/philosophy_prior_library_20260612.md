# Philosophy / Methodology Prior Library

- pass: `True`
- principles: `30`
- min success cases: `2`
- min negative cases: `1`
- conservative gate ready coverage: `1.0`
- top-3 expert agreement: `1.0`
- graph roundtrip exact: `True`

## Principle IDs

- `control_variables`: When causal attribution is uncertain, vary one factor while holding other factors fixed.
- `divide_and_conquer`: Split a complex task into separable subproblems with explicit interfaces, then compose the results.
- `proof_by_contradiction`: To test a claim, assume its negation and derive an impossible or inconsistent consequence.
- `reductio_ad_absurdum`: Stress an assumption by extending it to an absurd consequence that exposes its boundary.
- `occams_razor`: Prefer the simpler model when competing explanations have similar evidence and predictive power.
- `bayesian_update`: Update belief strength as new evidence arrives, weighting prior confidence by observed likelihood.
- `analogical_reasoning`: Map a new problem to a structurally similar solved problem, then test which relations transfer.
- `boundary_condition_analysis`: Test the edges of the scope because assumptions often fail at extremes or transitions.
- `negative_control`: Use a condition where the effect should not appear to detect leakage, confounding, or prompt artifacts.
- `minimum_viable_prototype`: Build the smallest working version that can falsify the core uncertainty before scaling.
- `incremental_replacement`: Replace one bounded component behind an interface while preserving rollback to the working baseline.
- `model_comparison`: Compare alternative models on matched evidence, heldout performance, and failure boundaries.
- `error_decomposition`: Decompose aggregate error into attributable components before choosing a repair.
- `invariant_search`: Identify quantities, roles, or relations that should remain unchanged across transformations.
- `causal_intervention`: Actively intervene on a suspected cause to distinguish causation from correlation.
- `local_linearization`: Approximate a nonlinear system locally when changes are small and verify the approximation boundary.
- `feedback_stability`: When a disturbance grows, look for feedback that amplifies, opposes, or stabilizes the change.
- `special_to_general`: Generalize from concrete successful cases only after identifying the invariant that explains them.
- `general_to_special`: Apply a general principle to a specific case by checking scope, assumptions, and local constraints.
- `prior_estimate_then_update`: Start with a calibrated prior estimate, then revise after observing task specific evidence.
- `duality`: Solve a problem by switching to a dual representation where constraints or objectives are easier.
- `conservation_law`: Track conserved quantities through a transformation and reject explanations that leak or create mass, budget, or probability.
- `dimensional_analysis`: Check whether quantities and formulas are meaningful by preserving units, scale, and dimension.
- `limiting_case_analysis`: Verify that a proposed framework reduces to a known result under a limiting scope condition.
- `falsifiability`: State what observation would make the claim fail before treating it as useful knowledge.
- `robustness_testing`: Test whether behavior survives noise, perturbation, distribution shift, and adversarial cases.
- `ablation`: Remove or disable one component to measure whether it is necessary for the observed effect.
- `placebo_control`: Use a sham or inert intervention to estimate expectation, style, or measurement artifacts.
- `cross_domain_transfer`: Transfer a method across domains only after matching structure, invariants, and failure boundaries.
- `scope_narrowing`: When a broad claim fails, narrow its scope to the conditions where it remains true and useful.

## Claim Boundary

- `complete_cyc_style_common_sense_library`
- `human_priors_as_unquestioned_axioms`
- `automatic_core_prior_promotion`
- `retrieval_agreement_as_expert_proof`
