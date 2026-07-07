# Math Verification Literature And Module Audit 2026-06-16

## Why This Audit

Fresh HLE Math exactMatch runs showed that Assumption Agent modules are activating correctly, but the Agent still ties GPT-5.5 / HippoRAG on retained runs. The failure mode is not missing graph/morphism/world-model/recursive modules. The failure mode is that recursive child prompts often converge to the same wrong answer, and the current restricted SymPy tool abstains on most non-trivial HLE math questions.

## Recent Literature Signal

Primary sources checked:

- Safe, `arXiv:2506.04592`: retrospective step-aware formal verification uses Lean 4 to formalize and verify LLM reasoning steps. Key idea: replace arbitrary confidence scores with proof-backed step states. URL: https://arxiv.org/html/2506.04592v1
- NL-FL HybridReasoning, `arXiv:2505.23703`: converts natural-language QA math problems into formal/existence-style problems, uses a formal-language reasoner, then extracts an NL answer. This directly matches our HLE exactMatch gap. URL: https://arxiv.org/html/2505.23703v3
- Formally Solving Answer-Construction Problems in Lean, `arXiv:2505.18492`: emphasizes that many contest problems require constructing an answer/witness, not merely proving a stated theorem. Introduces Enumerate-Conjecture-Prove style neuro-symbolic answer construction. URL: https://arxiv.org/html/2505.18492v5
- ThinkPRM, `arXiv:2504.16828`: process reward models that verify each step with a generated verification chain, improving best-of-N / guided search. URL: https://arxiv.org/html/2504.16828v3
- R-PRM, `arXiv:2503.21295`: reasoning-driven PRM reports large gains when used for guided math reasoning. URL: https://arxiv.org/html/2503.21295v1
- AlphaProof, Nature 2025: treats Lean as a verifiable RL environment; proof search gets hard rewards from formal state transitions rather than self-report. URL: https://www.nature.com/articles/s41586-025-09833-y
- APOLLO, NeurIPS 2025: compiler-guided Lean repair decomposes failing proofs/subgoals and uses targeted repair, rather than repeatedly sampling whole proofs. URL: https://papers.neurips.cc/paper_files/paper/2025/file/3b77109ad4dd4ba82d07cacd4b24207e-Paper-Conference.pdf
- PROVE, `arXiv:2410.12608`: programs-as-verifiers filter self-consistency votes; not all votes should count. URL: https://arxiv.org/abs/2410.12608
- ProcessBench, `arXiv:2412.06559`: hard math process-error detection remains challenging; answer-only matching is not enough. URL: https://huggingface.co/papers/2412.06559

Conclusion: the next useful improvement should be a verifier backend, not another prompt child. The relevant pattern is `generate candidate -> translate to executable/formal claim -> check -> only then vote/retain`.

## Local Module Audit

Artifacts audited:

- `hle_triad_fresh_math_canonicalizer_gpt55_n3_scan50000_20260616`
- `hle_triad_fresh_math_no_early_quorum_gpt55_n3_scan50000_20260616`
- `hle_triad_fresh_math_consensus_challenge_gpt55_n3_scan50000_20260616`

Current module status:

| Module | Status | Interpretation |
| --- | --- | --- |
| raw GPT-5.5 control | correct | Only `answer_type_router` and `answer_format_verifier` activate. No graph/morphism/world-model/recursive modules leak into raw. |
| HippoRAG baseline | correct | `hipporag_context_retrieval`, `hipporag_associative_rerank`, and prompt builder activate. Assumption Graph, morphism, world model, and recursive runner are `not_applicable`. |
| Assumption Graph retrieval | correct | Activates 3/3 in each retained fresh run. On HLE math it mostly retrieves generic harness nodes, so activation is correct but evidence quality is weak. |
| structural morphism transfer | correct | Activates 3/3; formal/structural hits are often empty on HLE math. This is a coverage problem, not a missing-module problem. |
| world model router | correct | Activates 3/3 and chooses `abstain_to_raw_prompt` in the audited math runs. This is the right behavior because retrieved graph context is generic and should not be injected. |
| recursive assumption runner | correct | Activates 3/3 and builds applicability frames. In `no_early_quorum`, reflective child paths actually run instead of being skipped. |
| recursive child validation | correct | Activates 3/3, logs child counts, early-stop status, prompt kinds, hashes, and timeouts. |
| multi-candidate verifier | mechanically correct | Selects majority or verified tool candidates as designed. It cannot fix same-model consensus hallucination without an external verifier. |
| HLE evidence bridge | correct | Activates 3/3, logs only hashes/counts/chars. It helps HippoRAG-style retrieval but does not solve math exactMatch. |
| restricted math tool | safe but low coverage | SymPy path is safe and logged; it abstains on most non-trivial HLE math. |
| residual writeback | correct for HLE smoke | Marked `not_applicable`; HLE evaluation artifacts are not written back into the main graph by design. |
| raw content persistence | correct | `raw_content_persisted=false`; question/gold answer/prediction text are not stored in scored rows. |

The script `hle_module_activation_audit.py` reports `module_gap_identified` for this input because its pass gate is designed to diagnose old `assumption_wrapper` artifacts. For the new artifacts, `expected_but_missing_modules={}` and module traces are present. Treat the audit failure as a historical-diagnostic gate, not as a current module failure.

Validation run:

- `python3 -m unittest discover tests -k hle`: 25 tests OK
- `python3 -m unittest tests.test_assumption_os`: 167 tests OK

Local tool availability:

- `SymPy`: available
- `Lean 4`: available, `Lean 4.31.0`
- `lake`: available
- `Z3`: not installed
- `Sage`: not installed
- `lean_dojo`: not installed

## Engineering Recommendation

Do not add more HLE prompt children by default. The negative `consensus_challenge` run showed that more child prompting can add cost without improving accuracy.

Build a verifier backend in four bounded layers:

1. `MathProblemRouter`
   Classify HLE math exactMatch into `arithmetic`, `symbolic_expression`, `equation_system`, `integer/combinatorics`, `geometry`, `proof/theorem`, `answer_construction`, or `unknown`.

2. `ExecutableClaimExtractor`
   From each candidate answer, extract a machine-checkable claim:
   - SymPy expression/equation/modular arithmetic for algebraic cases.
   - Lean theorem skeleton for proof/existence/answer-construction cases.
   - Later optional Z3/Sage adapters for constraints, combinatorics, and number theory.

3. `VerifierGate`
   Return structured states inspired by Safe / ProcessBench:
   - `verified`
   - `refuted`
   - `inconclusive`
   - `unsafe_or_unformalizable`
   Only `verified` candidates may override prompt majority. `inconclusive` cannot beat raw/direct fallback.

4. `VerifierAwareSelection`
   Replace `majority-first` with:
   - verified candidate wins
   - otherwise verified-refuted candidates are removed
   - otherwise fall back to direct / non-regressing majority
   - log verifier state into residual taxonomy for future self-evolution

## How This Fits Assumption Agent

- Graph retrieval should retrieve verifier policies and reusable formalization templates, not generic harness text.
- Morphism transfer should map a new math problem to a known verifier template, e.g. recurrence, invariant, modular arithmetic, extremal argument, construction witness.
- World model should decide when verifier cost is worth paying.
- Recursive runner should propose formal subclaims and counterclaims, not merely more natural-language answers.
- Residual writeback should store `unformalizable`, `verified_wrong_majority`, `tool_abstain`, and `template_missing` as first-class residuals.

## Promotion Gate

Before push/promotion, run same-batch fresh HLE triad:

- `raw`
- `hipporag_baseline`
- `assumption_agent_recursive_verify`

Required:

- no raw-content persistence
- no control contamination
- no expected modules missing
- Agent accuracy >= both controls
- at least one Agent-only correct case, or a larger batch with positive bootstrap CI

Current status: modules are working correctly; performance bottleneck is verifier capability.
