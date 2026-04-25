# On World Models, Reasoning Layers, and Where Scenario-Branching Should Actually Live

Written 2026-04-22 after baseline_long revealed scaffold-ceiling collapse. Captures the deeper architectural question raised during v13 design.

## The core question

Our v13-scenario proposal says: *explicitly prompt the LLM to generate hypothetical scenarios and evaluate each against the current problem's signals.*

But maybe the real observation is:

> "递归式的循环提出假设到论证假设，也许本来就该发生在 reasoning/thinking 层"
>
> (The recursive loop of proposing hypotheses and arguing them — maybe this should happen INSIDE the reasoning layer itself.)

## Why this question matters

Current LLMs (o1, Gemini-3-flash's thinking mode, Claude extended thinking) already expose a "reasoning/thinking" layer — hidden chain-of-thought tokens that execute BEFORE the user-visible output. These tokens are where the model does its "deliberate" work.

Our empirical observations:
- gemini-3-flash via proxy consistently generates ~1000 output tokens for visibly short answers
- Meaning: the model spends hundreds of tokens thinking internally before writing anything
- Our scaffold gets loaded **as prompt input**, not as guidance TO the thinking layer

So there's a layering question:

```
┌─────────────────────────────────────────┐
│  User prompt (problem + our scaffold)   │
│  ↓                                      │
│  Thinking layer (hidden CoT tokens)     │  ← where hypothesis/argument
│  ↓                                      │    SHOULD happen ideally
│  User-visible output                    │
└─────────────────────────────────────────┘
```

**If the thinking layer already does hypothesis-proposal-and-argument well, our external scenario scaffolding is at best redundant, at worst interference.**

---

## What we don't know (but matters)

### 1. Does 3-flash's thinking layer do scenario-branching natively?

Evidence suggestive of YES:
- baseline_long > v11 on most domains (+20pp) → bare prompt gives the thinking layer MORE room
- v12c's gains are mostly length artifact → added scaffolding doesn't qualitatively improve reasoning

Evidence suggestive of NO:
- If thinking layer did full scenario-branching, we'd expect much higher math scores (current math wr ~60% at best; strong reasoning should be 80%+)
- Failure modes like "over-confident wrong answers on hard problems" suggest the thinking doesn't stop to branch-evaluate

### 2. If we could see and edit the thinking layer, would we?

With o1-like models or thinking-mode APIs, you can see the hidden CoT. Our pipeline can't — we use a proxy to gemini-3-flash via OpenAI-compatible endpoint that doesn't expose thinking content. So even if we wanted to inject scenario construction INTO the thinking layer, we can't with our current setup.

### 3. What's the cost of replicating thinking-layer logic externally?

SMR via explicit 2-turn prompting **re-derives** what the thinking layer might already do. Each external turn costs:
- Full prompt re-read (the thinking layer's work gets DISCARDED between turns)
- Explicit tokens for user-visible output
- Round-trip API latency (~8s × 2 turns = 16s per problem)

An internally-native implementation would be ~1x cost. Our external version is ~2x.

---

## Three possible stances

### Stance A: "Ride the thinking layer, stop scaffolding"

If 3-flash's thinking already handles scenario-branching well, then:
- Our whole scaffold is redundant
- Just write good problem statements, give adequate budget
- baseline_long is the right architecture; walk away

**Evidence for**: baseline_long's sweep on 3-flash.
**Evidence against**: Certain pockets (science falsifiability, business/engineering priors) still benefit from external scaffolds → thinking layer doesn't handle everything.

### Stance B: "Thinking layer handles local reasoning; we provide global structure"

Maybe the LLM's thinking does fine on LOCAL scenario-branching (given a context window of relevant facts) but struggles with META-structure like "which principle applies here". External scaffolding then:
- Selects the right meta-principle (we're currently doing this via priors)
- Provides scenario scaffolds (our v13-scenario direction)
- Leaves local reasoning to thinking layer

This is the current v13 MVP hypothesis. Testing right now.

### Stance C: "Thinking layer is still impoverished; rebuild reasoning externally"

If the thinking layer is NOT doing robust scenario-branching:
- External multi-turn is valid (our v13-v14 direction)
- But we're essentially building a poor replica of what a native reasoning-trained model would do
- Better long-term: wait for/use models with better native reasoning (Opus 5, GPT-6-reasoning)

---

## The world-model dependency

Whichever stance is correct, **the scenario generation quality is bounded by the LLM's world model**. This breaks down by domain:

| Domain | World-model richness | External SMR viability |
|---|---|---|
| business, daily_life | Very high (vast text about human situations) | Strong — scenarios will be plausible and diverse |
| software engineering | High | Moderate — LLM already has strong native heuristics |
| science | Medium | Lower for novel questions; higher for textbook-type |
| mathematics | Low (no "scenario" concept for pure proof) | **Doesn't apply** — keep v12c hygiene route |
| engineering | Medium-high | Moderate |

This validates our decision to route math/science to hygiene in v13-scenario/reflect. **Where there's no valid world model to simulate, don't try.**

---

## The "no logic/reasoning world model exists" gap

Per the user's observation:

> "好像目前没有合适逻辑/语言/推理类问题使用的世界模型"

This is correct and important. World models in robotics/games/physics:
- Have explicit state representations
- Forward simulation with causal constraints
- Well-defined branching points (action choices)

For reasoning/language problems, there's no equivalent:
- State is unstructured (what are the "variables" of a strategic advice problem?)
- Forward simulation is unconstrained (infinite "next sentences")
- Branching is not enumerable

The LLM IS the de facto world model for these problems — its learned distribution over text continuations. **There is no cleaner substrate to bolt onto.**

Implication: SMR-style scenario construction MUST use the LLM itself as the simulator. We can't factor out the world-model dependency; we can only structure HOW we query it.

---

## Practical decision for now

Given we can't modify 3-flash's thinking layer:

1. **Run v13-scenario and v13-reflect MVPs** to see if external structuring adds value
2. **If v13-scenario clearly beats baseline_long by ≥+5pp**: Stance B is right, continue building SMR
3. **If v13-scenario ≤ baseline_long or only marginally beats**: Stance A is likely right on 3-flash. Shift effort to:
   - Test scaffold on stronger generators (Claude Opus, GPT-5.4) where thinking quality is different
   - Investigate thinking-mode APIs (o1, gemini-thinking) for inside-the-layer hypothesis injection

## The longer-range question

If internal reasoning models improve fast enough (2026 trajectory: o1 → o3/o4 → o5), external scaffolding becomes lower and lower value. The architectural bet is:

- **Short-term (2026)**: External SMR can still add value on 3-flash-class models
- **Medium-term (2027+)**: Native reasoning does most of what SMR does; external scaffold is mostly routing + meta-principle selection
- **Long-term**: Reasoning models internalize scenario-branching fully; external intervention becomes fine-grained goal specification only

We should build v13-scenario as a SHORT-TERM contribution, not expect it to be durable. The intellectual value is the demonstration: *with explicit scenario-branching, you can close part of the gap between weak LLMs and native reasoning models.*
