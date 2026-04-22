# Phase 2 Final State Report — v1 through v12c + baseline_long

Written 2026-04-22, immediately after baseline_long verdict (+20pp over v11).

## TL;DR

**Our phase2 scaffold (priors + triggers + wisdom) adds net negative value on 3-flash when baseline is given equal token budget.** The entire 14-iteration improvement chain (v1 → v12c) dissolves once baseline's 400-character constraint is lifted.

The ONLY robust wins:
1. **Stage-1 attention priors on business/engineering**: real +8-10pp, non-artifact
2. **Science falsifiability hygiene (v12c's "可证伪性" self-check)**: only domain where v12c still beats baseline_long (93-7 on science)

Everything else in the scaffold is either neutral or negative-valued attention noise.

---

## Full variant trajectory

### Scaffold variants (all built on Stage-1 `orient_hybrid` priors)

| Variant | What it added | vs baseline | vs prior best | Verdict |
|---|---|---|---|---|
| phase2_triggers (v1) | 161 triggers mined from 55 losses | 53% | +3pp vs ours_27 | First winner |
| phase2_v2 | 6 GPT-5.4 aphorism triggers | 42% | -11pp | REVERT (Flash can't unpack aphorisms) |
| phase2_v3 | 75-entry wisdom library (dropped triggers) | 40% | -12pp | REVERT (confound: dropped Stage-1 too) |
| phase2_v4 | wisdom + Stage-1 (no 161 triggers) | 48% | -5pp | wisdom alone not enough |
| **phase2_v5** | Stage-1 + 161 triggers + wisdom (stacked) | **52-54%** | — | **Old ceiling** |
| phase2_v6 | v5 + 140 new triggers from v5 losses (301 total) | 52% | 50-50 vs v5 | WASH |
| phase2_v7a | embedding signal retrieval | 50% | 41-59 vs v5 | REVERT |
| phase2_v7b | thin mode on math/science | 52% | 48-52 vs v5 | REVERT |
| phase2_v7c | wisdom 75→105 | 48% | 48-52 vs v5 | REVERT |
| phase2_v8 | math hygiene prototype (proto-v12c) | 52% | 48-52 vs v5 | WASH overall but math 60% vs baseline |
| phase2_v9 | L1 index (90 compact labels replacing 161 unpacked) | 38% | 45-55 vs v5 | REVERT (decoder needs full content) |
| phase2_v10 | verified-only (161→72 via GPT-5.4 rating) | 52% | 51-49 vs v5 | WASH but hard 63-37 |
| **phase2_v11** | compressed triggers (40-80 → 22-30 chars) | **54%** | 53% vs v5 | KEEP (+1pp real) |
| phase2_v12a | compressed ∩ verified triggers | — | 41-58 vs v11 | REVERT (over-subtraction) |
| phase2_v12b | v11 + compressed wisdom | — | 49-50 vs v11 | WASH |
| **phase2_v12c** | v11 + math/sci hygiene routing | **57%** | **57-43 vs v11** | APPARENT winner |
| phase2_v12 | v12a + v12b + v12c stacked | — | 55-45 vs v11 | < v12c alone |

### Reverse / control variants

| Variant | What | vs v11 | vs v12c | What it showed |
|---|---|---|---|---|
| baseline (400 char) | plain prompt | — | — | reference |
| phase2_v12c_trunc | v12c truncated to v11 length | — | — (vs v11: **34-66**) | Length is ~60% of v12c's advantage |
| **baseline_long** | plain prompt + v12c's token budget on math/sci | **60-40** (baseline wins!) | 51-49 (tied) | **Scaffold is net negative with equal budget** |

### Held-out replication (seed=7, 50 problems disjoint from seed=42)

| Comparison | seed=42 (100) | seed=7 (50) | |
|---|---|---|---|
| v11 vs baseline | 54% | 54% | ✅ stable |
| v12c vs baseline | 57% | 54% | -3pp (scaffold reverts to v11 level) |
| v12c vs v11 | 57% | 58% | ✅ stable (but confounded by length) |

### Self-Discover comparison (CONFOUNDED: ours_27 on 2.5-flash, v11/v12c on 3-flash)

| Comparison | Result | Notes |
|---|---|---|
| v11 vs ours_27 | 73-27 (+46pp) | generator upgrade ~10-15pp, scaffold+format ~30pp residual |
| v12c vs ours_27 | 77-23 (+54pp) | Same confound |

Fair rerun (ours_27 on 3-flash) **in progress as of report writing**.

---

## Length audit (key diagnostic)

Why baseline_long's discovery is so damning:

| Variant | avg chars | math chars | token budget | prompt constraint |
|---|---|---|---|---|
| baseline | 682 | 725 | 800 | "不超过 400 字" (ignored) |
| phase2_v5 | 599 | 659 | 900 | "不超过 500 字" |
| phase2_v11 | 578 | 627 | 900 | "不超过 500 字" |
| phase2_v12c | **687** | **1133** | 900/1100 | "可适当放宽" for math |
| **baseline_long** | **737** | **1046** | 800/1100/900 | baseline wording, relaxed for math/sci |
| ours_27 (2.5-flash) | 1232 | 1166 | 800 | no length constraint |

Key observations:
1. baseline obeys "400 字" soft constraint only loosely (goes to 682) but max_tokens=800 still caps effective length
2. v11 gets more tokens (900) but scaffold eats some
3. v12c math gets both more tokens AND relaxed wording → double unlock → 1133 chars
4. **baseline_long proves the budget matters more than the scaffold content**

---

## Why v12c "wins" — decomposition

Based on side-by-side analysis of math/science problems where v12c beat v11:

| Factor | Contribution to +14pp | Is this real reasoning? |
|---|---|---|
| Removes wisdom W-ID citations | ~+3pp | **Yes** — judge calls W0XX refs "指代不明" and penalizes |
| Token budget unlock (900→1100 math) | ~+4-6pp | **Partly** — more room for formal derivations, but also pure length bias |
| Hygiene prompt's "可证伪性" checklist | ~+3-4pp | **Half-artifact** — LLM dutifully adds "可证伪性验证" section that rubric-matches judge |
| Longer answer → judge bias | ~+3pp | **Artifact** |

After controlling for length (v12c_trunc vs v11 = 34-66), v12c drops **23pp**. Even accounting for truncation mid-derivation penalty, the real reasoning improvement is maybe +3-5pp at best.

---

## Domain variance — the architectural lesson

On 3-flash (baseline_long vs v11), where does scaffolding still help vs hurt?

| Domain | v11 (scaffolded) wr | Interpretation |
|---|---|---|
| business | 60% | ✅ Stage-1 domain priors encode useful knowledge |
| engineering | 60% | ✅ Same |
| science | 47% | 🟡 Baseline catches up with just budget |
| daily_life | 40% | ❌ Scaffold hurts; LLM has strong native knowledge |
| mathematics | 20% | ❌ Scaffold severely hurts; pure computation needs no meta |
| **software_engineering** | **24%** | ❌ Catastrophic — our sw_eng priors/triggers are net noise |

The "scaffold helps soft problems" intuition is PARTIALLY wrong. It only holds for business/engineering — domains where category-level process knowledge actually encodes something the LLM doesn't have on demand. For daily_life and sw_eng, the LLM's training data already contains the relevant heuristics, and scaffold just adds distraction.

---

## Big picture

**Modern LLMs (gemini-3-flash era) have rich enough world models that the old scaffold story — "inject meta-cognition to prevent LLM from being shallow" — is largely solved internally.** Our scaffold was fighting a battle against 2.5-flash that 3-flash has already won on its own.

Residual pockets where scaffold still matters:
1. **Domain-specific process knowledge** (business/engineering strategic framings)
2. **Rubric-matching hygiene prompts** (science falsifiability)
3. **Token budget allocation** (route hard-compute domains to long budget)

These are sparse enough that the right architecture is not "one stack for everything" but **condition-routed minimal interventions**.

Beyond this, the next ceiling (v13+) requires structural change — explicit multi-turn reasoning with scenario construction and branch evaluation. See `v13_architecture_proposal.md`.

---

## What to discard

Based on 14 iterations of evidence, the following ideas are **dead** for 3-flash:

1. ❌ Embedding-based signal retrieval (v7a REVERT, v9 REVERT)
2. ❌ Wisdom library size expansion (v7c, v12b both WASH/REVERT)
3. ❌ Trigger library size expansion (v6, v12a WASH/REVERT)
4. ❌ Aphorism-level compression (v2 REVERT, v11 is borderline)
5. ❌ Thin-mode scaffolding (v7b REVERT)
6. ❌ Stacking all tricks (v12 < v12c alone)

## What to keep

1. ✅ Stage-1 attention priors for business/engineering (+8-10pp real)
2. ✅ Math/science hygiene routing (v12c's EXECUTE_MATH/SCIENCE)
3. ✅ Soft "可选引用" wisdom phrasing (NOT forced citation)
4. ✅ Content-hash judge cache (infrastructure)

## What's missing (next phase)

See `v13_architecture_proposal.md` for scenario-branch architecture, and `world_model_thinking_layer.md` for the deeper question of whether recursive hypothesis-and-evaluation belongs in the reasoning layer itself rather than external prompt scaffolding.
