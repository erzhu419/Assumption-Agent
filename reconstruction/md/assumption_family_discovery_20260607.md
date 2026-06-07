# Open-set assumption family discovery

Date: 2026-06-07

## Why this replaces the narrow context-edge result

The previous `structural_context_edges_20260607` validation was useful but too narrow. It proved that one context pattern, "growth/perturbation plus opposing response plus constraint", can be expanded from local HippoRAG-style context edges into a generalized assumption context such as Lenz law / Le Chatelier / homeostasis.

That is not the full target. The intended mechanism is broader:

- input: theory cards from philosophy, science, mathematics, algorithms, and engineering;
- extraction: roles, morphisms, invariants, negative controls, and reusable assumption kernels;
- induction: discover how many assumption families exist in the supplied corpus;
- integration: decide whether a new theory belongs to an existing family or starts a new family;
- graph form: theory -> primitive role -> assumption family, plus theory-theory structural morphism edges.

The system should not require us to know beforehand how many "philosophies" exist or which theorem belongs to which family.

## Implemented module

`assumption_os.assumption_family_discovery`

Core functions:

- `extract_theory_signature(card)`
- `build_assumption_family_discovery_payload(...)`
- `classify_new_theory_card(card, discovered_payload, ...)`

The current implementation is deterministic and auditable. It is a category-inspired assumption-kernel induction layer, not a complete category-theory theorem prover.

## Validation fixture

The performance validation uses 30 theory cards across 10 families:

- residual correction: ResNet, Kalman innovation update, Newton residual update;
- negative feedback: Lenz law, Le Chatelier, homeostasis;
- signal/nuisance separation: seismic autocorrelation, JEPA latent prediction, PCA;
- controlled intervention: randomized controlled trial, A/B test, controlled-variable experiment;
- decomposition/composition: divide-and-conquer, MapReduce, modular proof;
- bottleneck/capacity: Amdahl law, rate-limited queue, Liebig law of the minimum;
- counterexample refinement: Popper falsifiability, proof by counterexample, CEGIS;
- conservation/balance: energy conservation, Kirchhoff current law, probability mass normalization;
- monotone progress: Lyapunov descent, policy iteration, coordinate ascent;
- representation transform: Fourier transform, Laplace transform, log transform for products.

The final family is intentionally not mapped to the old `DEFAULT_STRUCTURAL_PATTERNS` catalog, so the validation checks open-set new-family discovery rather than only matching old patterns.

## Result

Command:

```bash
python3 -m assumption_os.assumption_family_discovery --out 'phase four/assumption_graph/paper_readiness_20260604/assumption_family_discovery_20260607.json'
```

Metrics:

| metric | value |
| --- | ---: |
| input cards | 30 |
| discovered families | 10 |
| gold families | 10 |
| cluster purity | 1.0 |
| same-family pair recall | 1.0 |
| cross-family block rate | 1.0 |
| word-context pair recall | 0.3 |
| nonlexical positive pair count | 21 |
| nonlexical positive pair recall | 1.0 |
| new open-set family count | 1 |

Interpretation:

- The structural induction layer clusters cross-domain theories that share an assumption kernel even when word-level context retrieval is weak.
- It discovers at least one new family not present in the old structural pattern catalog.
- The output is graph-shaped: theory cards realize families through primitive roles, and same-family theory pairs receive structural morphism edges.

## Boundary

This does not mean the system has enumerated all philosophical, scientific, or mathematical assumptions. It means the architecture no longer depends on hand-writing one context pattern at a time. A larger corpus can now be passed in as theory cards, and the system will produce candidate families, new-family seeds, morphism edges, and integration decisions that can be recursively validated downstream.
