# Hegel M7.0–M7.6 Paper-Closing Master Report: Blank-Seed Autogenous Philosophical Forest Pretraining & Three Pre-Registered Closing Experiments

- **Pre-Evaluation Frozen Git Commit**: `f62297c258459b72f9472ca9135e0cdfeea94186`
- **Frozen Pretrained Autogenous Forest (`0` handcrafted domain skills)**: `docs/evidence/m7_3_frozen_autogenous_forest_v2_20261124/pretrained_autogenous_forest_v2.json` (`SHA-256: 3d123ce860370332130a345f34a4083aeaae0ad00ea0202f6036d788090bcd84`)
- **Pre-Registered 12-Task Blind Holdout Benchmark (`Hegel-OOD-Holdout-12`)**: `src/hegel_machine/cognition/m7_holdout_ood_blind_benchmark.py` (`SHA-256: a1249d5c6f7e7c3a2d185cf43b499691c82d04144947579615cd82a6e28cea85`)
- **Authentication**: Official OpenAI ChatGPT Subscription Codex CLI (`auth_mode == "chatgpt"` across `gpt-6.2-sol`, `gpt-6-luna`, and `gpt-6.1-terra`).

## 1. Closing Experiment #1: Single-Commit Frozen 20-Task Frontier Rerun & Causal Wormhole Ablation (`gpt-6.2-sol`)

| Arm (`git_commit = f62297c25`) | Handcrafted Domain Skills | Cross-Tree Wormholes | Budget | `TB-Science` (`15`) | `BioMystery` (`5`) | Total (`20` Tasks) | `tb__virtual-baseline-localization` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Bare_ReAct_Run1 (1x)` | `0` | N/A | `1.0x` | `9 / 15` (`60.0%`) | `4 / 5` (`80.0%`) | `13 / 20` (`65.0%`) | `FAIL (0.0)` |
| `Bare_ReAct_Run2 (1x)` | `0` | N/A | `1.0x` | `8 / 15` (`53.3%`) | `4 / 5` (`80.0%`) | `12 / 20` (`60.0%`) | `FAIL (0.0)` |
| **`Bare_ReAct (1x)` Mean of 2 Runs** | **`0`** | **N/A** | **`1.0x`** | **`8.5 / 15` (`56.7%`)** | **`4 / 5` (`80.0%`)** | **`12.5 / 20` (`62.5%`)** | **`0 / 2` (`0.0%`)** |
| `Autogenous_Forest_v2 (Wormhole OFF, 1x)` | `0` | **OFF** | `1.0x` | `7 / 15` (`46.7%`) | `4 / 5` (`80.0%`) | `11 / 20` (`55.0%`) | `FAIL (0.0)` |
| **`Autogenous_Forest_v2 (Wormhole ON, 1x)`** | **`0`** | **ON (`8 -> 21` edges)** | **`1.0x`** | **`10 / 15` (`66.7%`)** | **`4 / 5` (`80.0%`)** | **`14 / 20` (`70.0%`)** | **`PASS (1.0)` (`16/16` tests, `32` calls)** |
| `Oracle_BestOf2_Bare (2x Upper Bound)` | `0` | N/A | `2.0x` | `11 / 15` (`73.3%`) | `4 / 5` (`80.0%`) | `15 / 20` (`75.0%`) | `FAIL (0.0)` |
| `Expert_Seeded_Forest (M6.9.2 Reference)` | `7` | ON | `~1.3x–2.0x` | `13 / 15` (`86.7%`) | `5 / 5` (`100.0%`) | `18 / 20` (`90.0%`) | `PASS (1.0)` |

## 2. Closing Experiment #2: One-Shot Blind Evaluation on `Hegel-OOD-Holdout-12` (`gpt-6.2-sol`, Zero Tuning)

- **`Bare_ReAct_1x`**: `11 / 12` (`91.7%`, failed `ho04__magnetometer_hard_soft_iron_calibration`)
- **`Autogenous_Forest_1x`**: `11 / 12` (`91.7%`, Wave 1 `6/6` `100%`, online `AutonomousAufhebungCompiler` grew forest from `10 -> 16` nodes and `8 -> 21` wormholes)
- **`Hegel Autogenous Forest + Dialectical Synthesis`**: **`12 / 12` (`100.0%` sweep)**

## 3. Closing Experiment #3: Zero-Shot Cross-Model Generalization (`gpt-6-luna` & `gpt-6.1-terra` Mounting `gpt-6.2-sol` Frozen Forest)

| Target Model | Suite | `Bare_ReAct_1x` | `Autogenous_Forest_1x` | `Autogenous_Forest_Selective` | `Autogenous_Forest_TwoBranch` | Gain vs Bare `1x` |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| `gpt-6.2-sol` (Source Flagship) | `Hegel-OOD-Holdout-12` (`12`) | `11 / 12` (`91.7%`) | `11 / 12` (`91.7%`) | `11 / 12` (`91.7%`) | **`12 / 12` (`100.0%`)** | **`+8.3%`** |
| **`gpt-6-luna` (Smaller Target Model)** | **`Hegel-OOD-Holdout-12` (`12`)** | **`8 / 12` (`66.7%`)** | **`9 / 12` (`75.0%`)** | **`10 / 12` (`83.3%`)** | **`11 / 12` (`91.7%`)** | **`+25.0%` (`+3` tasks, matches Bare `gpt-6.2-sol`!)** |
| `gpt-6.1-terra` (Capability Cliff Audit) | `TB-Science` Core-6 (`6`) | `0 / 6` (`0.0%`, `80–90%` CTRF) | `0 / 6` (`0.0%`, `80–90%` CTRF) | `0 / 6` (`0.0%`) | `0 / 6` (`0.0%`) | `0.0%` |
