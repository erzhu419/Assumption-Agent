# Experiment Roadmap — Post-v12c Direction

Written 2026-04-22, consolidating all proposed experiments into priority order.

## Currently running (dispatched 2026-04-22)

1. **ours_27 regen on 3-flash** (`bvkb1vcjw`) — clean Self-Discover comparison
2. **v13-reflect gen** (`bfh3udgly`) — 2-turn reflection MVP
3. **v13-scenario gen** (`bm1dx5s29`) — SMR 2-turn MVP

After these finish (~20 min each), dispatch:
- ours_27_g3 vs baseline / v11 / v12c
- v13-reflect vs baseline_long / v11 / v12c
- v13-scenario vs baseline_long / v11 / v12c / v13-reflect

---

## Priority-ordered experiment queue

### P0 — Critical validations (already partly done)

| Status | Experiment | Purpose |
|---|---|---|
| ✅ done | Held-out 50 (seed=7) | Validate v11/v12c replicate on unseen | 
| ✅ done | Verbosity control (v12c_trunc vs v11) | Isolate length artifact from reasoning |
| ✅ done | baseline_long vs v11/v12c | Control for token budget |
| 🏃 running | ours_27 regen on 3-flash | Remove generator confound in Self-Discover compare |

### P1 — Architecture exploration

| Status | Experiment | Purpose |
|---|---|---|
| 🏃 running | v13-reflect MVP | Does generic 2-turn reflection help? |
| 🏃 running | v13-scenario MVP | Does scenario-branching specifically help? |
| ⏳ queued | **v14-branchplan (4-turn)** | Only if v13-scenario shows signal ≥+5pp |
| ⏳ queued | Scenario quality human audit | Inspect 30 v13-scenario Turn 1 outputs manually |

### P2 — Cross-model portability

| Status | Experiment | Purpose |
|---|---|---|
| ⏳ queued | v11 / v12c with Claude Opus 4 as generator | Does scaffold help stronger models? |
| ⏳ queued | v11 / v12c with GPT-5.4 as generator | Same check, different family |
| ⏳ queued | Thinking-mode API test (o1/gemini-thinking) | Native reasoning vs external scaffold |
| ⏳ queued | Rerun v13-scenario on Claude Opus | Is SMR portable upmarket? |

### P3 — Judge robustness

| Status | Experiment | Purpose |
|---|---|---|
| ⏳ queued | Human blind judgment on 20 problems | Calibrate LLM judge reliability |
| ⏳ queued | Re-judge with GPT-5.4 as judge | Cross-judge stability test |
| ⏳ queued | Length-controlled regen (not truncate) | Cleaner length-control than v12c_trunc |

### P4 — Scale

| Status | Experiment | Purpose |
|---|---|---|
| ⏳ queued | n=300 sample run | Is 6pp difference stat-sig at larger n? |
| ⏳ queued | Harder-problems subset | Test on only hard problems (v11/v12c both struggle) |

---

## Decision tree for next work

```
wait for ours_27_g3 + v13-reflect + v13-scenario judges
│
├── v13-scenario >> baseline_long (+5pp or more)
│   ├── Go deep on SMR: build v14-branchplan (4-turn)
│   └── Write preliminary paper: "Scenario-Branch Reasoning"
│
├── v13-scenario ≈ v13-reflect > baseline_long (both marginal)
│   ├── Mechanism is generic 2-pass reflection, not scenarios
│   ├── Simpler paper framing: "Self-Refine on meta-principles"
│   └── Skip v14
│
├── v13-scenario ≈ v13-reflect ≈ baseline_long
│   ├── External scaffolding is dead on 3-flash
│   ├── P2 tier becomes critical: test on Claude/GPT-5.4
│   └── If scaffold still helps stronger models, paper is about
│       "When scaffolding stops helping = world-model threshold"
│
└── All worse than baseline_long
    ├── Our setup is the issue (not the ideas)
    ├── Reset: test scaffold with thinking-mode APIs
    └── Or pivot entirely
```

---

## What we're NOT doing (deliberate cuts)

1. **More trigger/wisdom mining** — v6 through v12 all showed this path is dead
2. **More embedding retrieval experiments** — v7a, v9 both failed
3. **More stacking** — v12 showed stacking dilutes
4. **More domain-specific prompt tuning** — v7b, v8 showed it's not generalizable
5. **More category-level ablations** — orient vs technique vs hybrid is settled

## Budget note

Each 100-problem run costs:
- Generation: ~17 min wall, ~80K tokens input + ~40K output
- Judge (vs one opponent): ~14 min wall, ~50K tokens input + ~10K output

Full 4-turn v14 would be ~70 min per run (4x gen + 2x judge). Reserve for confirmed signals only.

---

## Files to watch

After background gens finish, the judges will populate:
- `phase two/analysis/cache/answers/ours_27_answers.json` (overwritten, 3-flash)
- `phase two/analysis/cache/answers/phase2_v13_reflect_answers.json`
- `phase two/analysis/cache/answers/phase2_v13_scenario_answers.json`
- `phase two/analysis/cache/answers/phase2_v13_scenario_scenarios.json` (Turn 1 outputs — inspect for scenario quality)

For the scenario quality audit, read that second file and spot-check:
- Do scenarios actually differ? (not 3 paraphrases)
- Do they match the problem's real structure?
- Does signal_hit self-rating track with apparent scenario fit?

If many scenarios are generic/hallucinated, v13-scenario signal will be weak irrespective of the architecture's theoretical merit.
