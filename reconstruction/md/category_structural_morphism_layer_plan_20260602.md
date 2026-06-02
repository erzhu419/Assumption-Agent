# Structural Morphism Layer Plan - 2026-06-02

## 0. Thesis

这个项目当前已经有 Assumption Graph、formal mapping audit、finite kernel metrics、recursive runner、world model、verifier stack。范畴论方向下一步不应该做成“替代知识图谱”的存储结构，也不应该第一版就包装成完整范畴论推理引擎。工程上应收窄为：

> 在 Assumption Graph 之上的 bounded structural morphism layer：把新问题/新假设抽象成 typed diagram，检索旧 pattern，构造 candidate structural mapping，验证核心不变量是否被保持，再把通过验证的结构迁移作为 recursive assumption runner 的候选假设。

简化成一句话：

```text
KG/Assumption Graph stores memory.
Structural Morphism Layer recognizes structural lineage.
Recursive runner tests whether the lineage is useful.
```

理论上可以说它受 category-theoretic diagrams 启发；代码和实验里先称为 structural morphism / structural pattern layer，避免把第一版 typed graph matching 夸大成 category theory solver。

## 0.1 Construction Update - 2026-06-02

已按收窄版开始施工：

- 新增 `assumption_os.structural_patterns`。
- 默认 seed patterns：residual correction、controlled intervention、incremental replacement、negative feedback、signal-vs-nuisance separation。
- 新增 deterministic diagram extraction audit，避免把 LLM extraction 的不稳定性藏到后续 matcher。
- 新增 structural pair suite、non-lexical retrieval probe、offline behavior probe。
- 新增 structural morphism promotion gate，并接入 `verifier_stack` 为 `V2b structural_morphism_gate`。
- `retrieval_policy` 现在会以 shadow mode 注入 `Structural Morphism Reasoning`，不自动写图。

验证结果：

```text
python3 -m unittest tests.test_assumption_os
72 tests OK

structural extraction audit: pass
structural pair suite: positive_top1_rate=1.0, negative_rejection_rate=1.0, pass
non-lexical retrieval probe: top1_hit_rate=1.0, pass
behavior probe: guided_mean_score=0.9458, mean_delta=0.4875, guided_win_rate=1.0, pass
```

## 1. Why This Is Not Just a Knowledge Graph

HippoRAG 2 / From RAG to Memory 的关键点是：普通向量检索缺少长期记忆里的 sense-making 和 associativity，所以它用 LLM 抽取 open KG triples，把 phrase nodes、passage nodes、relation edges、synonym edges、context edges 组合起来，再用 query-to-triple、recognition memory 和 PPR 做多跳激活。

这解决的是：

```text
Which facts/passages/concepts should be activated together?
```

但当前问题需要的是：

```text
Which old abstract mechanism is this new hypothesis a realization of?
```

两者不同。KG 可以发现“Le Chatelier”和“equilibrium”相关，也可以发现“Lenz”和“induction”相关；但它不会自然发现二者共享同一个 abstract negative feedback diagram。要发现这种跨领域结构相似性，需要 diagram-level matching，而不是 triple-level spreading。

因此保留 Assumption Graph，新增 Categorical Structural Morphism Layer。

## 2. Correct Mathematical Framing

不要说：

```text
新假设 = morphism
老假设 = morphism
二者相似 = 两条边相似
```

更准确的说法是：

```text
一个假设 = 一个小 diagram / morphism family
一个具体算法/定律/策略 = 该 diagram 在某个 domain 里的 realization
新旧假设之间的关系 = candidate functor / natural transformation / meta-morphism
相似性 = object roles, morphism roles, compositions, invariants, predicted effects 是否被保持
```

这让“ResNet 思想被外延到很多网络”变成可计算问题：

```text
AbstractResidualCorrection
  object: state x
  morphism: residual_update F
  composition: output = identity(x) + F(x)
  invariants:
    - identity path is preserved
    - learned module only needs to model deviation
    - optimization can fall back to near-identity
```

不同模型只是 realization：

```text
ResNet block
Transformer residual stream
Diffusion U-Net residual/skip structure
Adapter/LoRA delta update
Neural ODE residual view
```

这些不是因为文本相似才相似，而是因为它们保持同一个 structural invariant set。

## 3. Literature Anchors

Use these as grounding, not as slogans.

- From RAG to Memory / HippoRAG 2: graph memory is useful for associative retrieval, especially query-to-triple, recognition memory, passage integration, and PPR. This supports the storage/retrieval substrate, not the full structural analogy layer.
- Markov categories and comparison of statistical experiments: Markov categories give a categorical language for stochastic kernels and statistical experiment comparison; Blackwell-style comparison is a useful formal model for “one hypothesis/experiment contains enough information to simulate another.” Current repo metrics are only finite executable diagnostics and an entropy-based Blackwell-style proxy, not a full Blackwell order computation.
- Markov categories + entropy/divergence: enriched Markov categories with divergence/metric structures support the idea that categorical morphisms can have quantitative distance, which matches the current finite kernel metrics in `assumption_os.formal_mapping`.
- ResNet: residual learning can be abstracted as learning a residual function relative to an identity mapping. This is a high-confidence seed pattern.
- JEPA / LeJEPA / LeWorldModel / Sub-JEPA: latent prediction plus Gaussian or subspace-Gaussian regularization is a current example of structural inductive bias in world models. Its relationship to older signal/noise separation methods should be treated as a candidate morphism and verified, not assumed.
- Seismic random-noise suppression / blind-spot denoising / correlation-based processing: these provide possible old realizations of a “predictable structure vs stochastic nuisance” pattern.

Primary source links used for this plan:

- From RAG to Memory: https://arxiv.org/abs/2502.14802
- Representable Markov Categories and Comparison of Statistical Experiments: https://arxiv.org/abs/2010.07416
- Markov Categories and Entropy: https://arxiv.org/abs/2212.11719
- Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385
- Identity Mappings in Deep Residual Networks: https://arxiv.org/abs/1603.05027
- LeJEPA: https://arxiv.org/abs/2511.08544
- LeWorldModel: https://arxiv.org/abs/2603.19312
- Sub-JEPA: https://huggingface.co/papers/2605.09241
- V-JEPA feature prediction: https://arxiv.org/abs/2404.08471
- Seismic self-supervised random noise suppression: https://arxiv.org/abs/2109.07344
- Seismic interferometry from correlated noise sources: https://arxiv.org/abs/2105.07250

## 4. Current Repo Baseline

Relevant existing pieces:

- `AssumptionNode.formal_form` and `payload` can already store typed formal objects without schema changes.
- `EdgeType.IS_FORMAL_ISOMORPHISM_OF` and `EdgeType.IS_ANALOGY_OF` already exist.
- `assumption_os.formal_mapping` already groups feature / constraint / decomposition / verification / hp_change nodes, audits complete mappings, computes finite stochastic kernel metrics, and runs formal search.
- `retrieval_policy.format_policy_context` already injects formal mapping hits into the solver context.
- `recursive_assumption_runner` already has recursive proposal/evidence/resume flow.
- `verifier_stack` already blocks unsafe promotion and keeps graph mutation gated.

Therefore this should be an additive layer, not a rewrite.

## 5. Data Model

### 5.1 Categorical Pattern

Store as an `AssumptionNode`:

```json
{
  "type": "alignment",
  "kind": "formal_mapping",
  "tags": ["categorical_pattern", "structural_morphism"],
  "formal_form": {
    "formal_kind": "categorical_pattern",
    "pattern_id": "pat_residual_correction",
    "objects": [
      {"id": "input_state", "role": "state"},
      {"id": "residual_update", "role": "learned_transform"},
      {"id": "output_state", "role": "state"}
    ],
    "morphisms": [
      {
        "id": "identity_path",
        "source": "input_state",
        "target": "output_state",
        "role": "identity_preserving_map"
      },
      {
        "id": "learn_delta",
        "source": "input_state",
        "target": "residual_update",
        "role": "deviation_estimator"
      },
      {
        "id": "compose_add",
        "source": ["identity_path", "learn_delta"],
        "target": "output_state",
        "role": "residual_composition"
      }
    ],
    "composition_laws": [
      "output_state = identity_path(input_state) + learn_delta(input_state)"
    ],
    "invariants": [
      "identity_path_preserved",
      "learned_part_models_deviation",
      "zero_delta_recovers_input",
      "optimization_has_near_identity_fallback"
    ],
    "negative_controls": [
      "plain_stack_without_identity_path",
      "arbitrary_ensemble_without_residual_composition"
    ]
  }
}
```

### 5.2 Realization

A realization binds an abstract pattern to a domain:

```json
{
  "formal_kind": "categorical_realization",
  "pattern_id": "pat_residual_correction",
  "domain": "deep_learning",
  "object_bindings": {
    "input_state": "block input activation x",
    "residual_update": "convolutional residual branch F(x)",
    "output_state": "block output y"
  },
  "morphism_bindings": {
    "identity_path": "skip connection",
    "learn_delta": "residual branch",
    "compose_add": "elementwise addition"
  },
  "evidence_refs": ["ResNet 2015", "Identity Mappings 2016"]
}
```

### 5.3 Structural Morphism Candidate

This is the central object:

```json
{
  "formal_kind": "structural_morphism_candidate",
  "source_pattern_id": "pat_signal_noise_separation",
  "target_pattern_id": "pat_latent_predictive_world_model",
  "object_map": {
    "stable_signal": "predictable_latent_state",
    "random_noise": "unpredictable_pixel_or_detail_variation",
    "correlation_operator": "latent_prediction_regularization"
  },
  "morphism_map": {
    "suppress_uncorrelated_noise": "avoid_reconstructing_unpredictable_detail",
    "recover_invariant_signal": "learn_stable_world_state_features"
  },
  "preserved_invariants": [
    "separate_predictable_structure_from_stochastic_nuisance",
    "use_distributional_assumption_to_stabilize_estimation"
  ],
  "broken_or_uncertain_invariants": [
    "classical autocorrelation is explicit second-order statistic; JEPA regularization is learned latent distribution matching"
  ],
  "status": "candidate",
  "required_validation": [
    "negative-control pair comparison",
    "downstream transfer probe",
    "human/LLM adjudication of preserved invariants"
  ]
}
```

## 6. Seed Pattern Library

Start with a small curated library. Do not let the system invent arbitrary categories before the gates work.

### Pattern A: Negative Feedback / Equilibrium Restoration

Positive realizations:

- Le Chatelier principle
- Lenz law
- basic control-system negative feedback

Core invariants:

- external perturbation changes a state variable
- system response is induced by the perturbation
- response opposes or compensates the imposed change
- a constraint/potential/conservation condition explains the opposition

Negative controls:

- positive feedback
- random change without induced response
- compensation that does not preserve any constraint

### Pattern B: Residual Correction / Identity-Preserving Update

Positive realizations:

- ResNet residual block
- Transformer residual stream
- adapter / LoRA delta update
- iterative refinement loops in recursive assumption runner

Core invariants:

- identity or baseline path remains available
- learned component models deviation
- zero update recovers baseline
- optimization or reasoning can proceed incrementally

Negative controls:

- plain stack with no identity path
- unrelated ensemble
- overwrite-style update with no fallback

### Pattern C: Signal vs Stochastic Nuisance Separation

Positive/candidate realizations:

- seismic random-noise suppression
- correlation/stacking/matched-filter style signal processing
- blind-spot denoising
- JEPA/LeJEPA/LeWorldModel latent prediction with Gaussian regularization
- Sub-JEPA subspace Gaussian regularization

Core invariants:

- useful structure is predictable/correlated/constrained
- nuisance component is random, weakly correlated, or not worth reconstructing
- operation biases model toward invariant signal
- validation should show better downstream prediction or denoising

Important caution:

The JEPA-to-seismic mapping is only a candidate. It must pass invariant-preservation checks and negative controls before being treated as accepted lineage.

### Pattern D: Controlled Intervention / A-B Falsification

Positive realizations:

- scientific control variables
- software ablation
- POPPER-style sequential falsification
- current verifier stack / trigger-control acceptance

Core invariants:

- one change at a time or explicitly modeled coupled changes
- matched control exists
- predicted effect is declared before outcome readback
- promotion depends on falsification, not only plausibility

## 7. Matching Algorithm

### Step 1: Extract Candidate Diagram

Input sources:

- user problem
- failed trace residual
- proposed new hypothesis
- paper abstract / reference note
- accepted/rejected proposal manifest

Output:

```text
candidate objects
candidate morphisms
candidate invariants
candidate predicted effects
uncertainties
```

Implementation first pass:

- use deterministic templates where possible
- use LLM only to fill typed roles
- store all extraction calls as TrialManifest

### Step 2: Retrieve Similar Patterns

Use a hybrid score:

```text
retrieval_score =
  lexical_trigger_score
  + role_overlap_score
  + invariant_overlap_score
  + graph_activation_score
  + metaproductivity_prior
```

This extends current `search_formal_mappings`. It should not replace PPR-style graph retrieval; it should add structural hits as another policy context section.

### Step 3: Generate Candidate Functor

For each retrieved old pattern, propose:

```text
object_map
morphism_map
invariant_map
composition_map
broken_invariants
transfer_prediction
```

The transfer prediction is required. Example:

```text
If this mapping is valid, then injecting residual-correction guidance should improve tasks where the current plan overwrites a baseline instead of preserving a fallback path.
```

### Step 4: Gate the Morphism

Gate criteria:

```text
object_role_coverage >= 0.75
morphism_role_coverage >= 0.70
composition_preservation >= 0.60
invariant_preservation >= 0.70
broken_invariant_count <= allowed threshold
negative_control_margin > 0
downstream_transfer_prediction is testable
```

No graph mutation from this gate alone. Passing creates a candidate proposal with a TrialManifest.

### Step 5: Validate Transfer

Use three validation layers:

1. Structural validation: does the diagram map preserve invariants better than negative controls?
2. Retrieval validation: does the pattern improve retrieval on non-lexical cross-domain probes?
3. Behavior validation: when injected into runner context, does it improve answer/task quality on heldout tasks?

Only behavior validation can promote a morphism from candidate to active.

## 8. Integration With Recursive Assumption Runner

Current recursive runner should gain one new child type:

```text
structural_transfer_hypothesis
```

Flow:

```text
parent problem/residual
-> extract diagram
-> retrieve old patterns
-> propose structural morphism
-> run structural gate
-> if pass, create child proposal
-> child proposal runs fresh ablation / judge / control
-> child returns accepted/rejected/revise payload
-> parent uses accepted morphism as reasoning context
```

This makes the user’s original idea operational:

```text
The agent does not merely propose a new hypothesis.
It asks whether the new hypothesis is a structure-preserving extension of an old one,
then recursively argues and tests that claim.
```

## 9. Evaluations

### 9.1 Structural Pair Suite

Create labeled pairs:

Positive:

- Le Chatelier ↔ Lenz
- ResNet ↔ Transformer residual stream
- control variable ↔ software ablation
- signal averaging / random noise suppression ↔ blind-spot denoising

Candidate / uncertain:

- seismic autocorrelation denoising ↔ LeJEPA/LeWorldModel Gaussian latent regularization
- Blackwell experiment comparison ↔ evaluator sufficiency gate

Negative:

- Le Chatelier ↔ arbitrary graph synonym match with no feedback
- ResNet ↔ plain feedforward stack
- JEPA Gaussian regularization ↔ any Gaussian noise assumption with no signal/nuisance separation

Metrics:

```text
positive_top1_rate
negative_rejection_rate
invariant_precision
invariant_recall
broken_invariant_detection_rate
```

### 9.2 Retrieval Probe

Ask queries with low lexical overlap:

```text
"A system changes in a way that cancels the disturbance that caused it."
"A model keeps a default path and learns only the correction."
"The useful component is stable but nuisance variation should not be reconstructed."
```

Expected:

- first retrieves negative feedback pattern and Le Chatelier/Lenz
- second retrieves residual correction
- third retrieves signal-noise separation and JEPA candidates, with uncertainty

### 9.3 Downstream Answer Probe

Compare:

```text
baseline graph context
vs
graph context + structural morphism context
```

Scoring must use existing heldout/judge/control machinery. Do not accept if the win is only style.

### 9.4 Recursive Runner Probe

Test whether runner can:

```text
residual -> diagram -> old pattern -> morphism candidate -> gated child -> validation -> return update
```

This is the key behavioral proof that the layer is part of recursive self-argument, not just a static analyzer.

## 10. Implementation Milestones

### M0: Documentation and Fixtures

- Save this plan.
- Add 10-20 hand-written pattern fixtures.
- Add positive/negative labeled pair fixture.

Exit condition:

```text
fixtures parse and can be loaded into AssumptionNode formal_form without schema changes
```

### M0.5: Diagram Extraction Audit

Before trusting structural matching, evaluate deterministic or LLM extraction on short labeled cases:

```text
object_role_precision
object_role_recall
morphism_role_precision
morphism_role_recall
invariant_precision
invariant_recall
broken_invariant_detection
```

Exit condition:

```text
extraction precision/recall is high enough before extraction feeds runner proposals
```

### M1: `structural_patterns.py`

Functions:

```text
load_structural_patterns(store)
extract_structural_signature(formal_form)
score_pattern_match(query_diagram, pattern)
propose_structural_morphism(query_diagram, pattern)
score_structural_morphism(candidate)
```

Exit condition:

```text
unit tests pass for positive/negative pairs
```

### M2: Formal Mapping Integration

Extend `formal_mapping.py` or add an adjacent module:

```text
build_categorical_pattern_payload
search_categorical_patterns
format_categorical_applications
build_structural_morphism_eval_payload
```

Exit condition:

```text
retrieval_policy can inject a "Structural Morphism Reasoning" section
```

### M3: Structural Gate

Add a verifier stage or pre-verifier payload:

```text
V2b structural_morphism_gate
```

It blocks promotion when:

- invariant preservation is weak
- negative control is closer than the proposed mapping
- transfer prediction is missing
- source/target diagrams are under-specified

Exit condition:

```text
unsafe morphisms cannot produce accepted graph proposals
```

### M4: Recursive Runner Child Type

Add child type:

```text
structural_transfer_hypothesis
```

First implementation should not change the runner enum. Use:

```json
{
  "frame_type": "candidate_hypothesis",
  "proposal_type": "structural_transfer_hypothesis",
  "formal_kind": "structural_morphism_candidate"
}
```

Promote to a dedicated enum only after behavior validation passes.

Exit condition:

```text
runner can generate, gate, execute, and resume at least one structural transfer child
```

### M5: Performance Validation

Run:

- structural pair suite
- retrieval probe
- downstream answer probe
- recursive runner probe

Exit condition:

```text
overall pass requires both structural accuracy and downstream benefit
```

First proof-of-concept should prioritize:

```text
Residual Correction prevents destructive overwrite.
```

Le Chatelier/Lenz and JEPA/seismic remain motivating or candidate-lineage examples, not the first main proof. The first main proof should be agent-behavioral and low-dispute: when a plan risks overwriting a working baseline, structural context should preserve baseline/fallback and apply a local delta instead of rewriting the whole system.

## 11. What Not To Do

Do not:

- claim category theory can represent all hypotheses directly
- mutate the whole graph schema before a bounded layer works
- accept a morphism because an LLM gives a beautiful analogy
- treat the JEPA ↔ seismic link as established before validation
- optimize only retrieval hit rate; behavior has to improve
- let formalism hide broken invariants

## 12. Success Criteria

This direction is working if the system can do the following:

```text
1. Given a new method, abstract its diagram.
2. Retrieve an old structurally similar pattern without relying on shared terms.
3. Explain the object/morphism/invariant mapping.
4. Identify what is not preserved.
5. Make a falsifiable transfer prediction.
6. Run a controlled validation.
7. Promote only if downstream evidence supports the structural transfer.
```

The strongest proof-of-concept would be:

```text
The runner sees a new residual where a plan overwrites useful baseline behavior.
It retrieves AbstractResidualCorrection.
It proposes a residual-style update instead of a replacement-style update.
It validates on heldout tasks.
It passes trigger/control gates.
It writes a gated, evidence-backed structural morphism into Assumption Graph.
```

That would make the category-theory direction a real part of recursive self-evolution, not just a theoretical appendix.
