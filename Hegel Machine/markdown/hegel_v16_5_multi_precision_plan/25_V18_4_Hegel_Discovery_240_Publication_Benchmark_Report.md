# 25. Hegel V18.4 (`Hegel-Discovery-240`): Publication-Scale 240-Task × 3-Model × 3-Arm Live Continual Discovery Benchmark & Amortization Ledger

- **Status**: `COMPLETED & CRYPTOGRAPHICALLY VERIFIED` (`2,160 / 2,160` Live Multi-Turn Official ChatGPT Subscription Episodes, `11,644 / 11,644` SHA-256 Verified Evidence Files)
- **Date**: `2026-11-21`
- **Provider & Compliance**: Official OpenAI ChatGPT Subscription (`CodexSubscriptionClient`, `auth_mode == "chatgpt"`, ZERO `ruoli.dev` traffic, ZERO credentials persisted)
- **Evidence Root**: `docs/evidence/v18_4_hegel_discovery_240_20261121/`

---

## 1. Benchmark Design & Publication Alignment (`Hegel-Discovery-240`)

To match ICML/NeurIPS/ICLR scientific discovery benchmark scale (`LLM-SRBench` 239 tasks, `DiscoveryBench` 264 tasks), `V18.4` constructs **`240` strictly disjoint unseen tasks** (`216` in-scope quantitative mechanism tasks across `12` families + `24` non-stationary chaotic out-of-scope honesty tasks), organized into **`6` sequential continual waves (`wave_1`–`wave_6`, `40` tasks/wave)** over an online-evolving source forest (`K = 45 -> 48` nodes across 10 scientific disciplines):

1. **Cluster I — Cross-Temporal & Multi-Scale Folding (`F01`–`F04`, `72` tasks)**:
   - `F01_IrregularMemoryKernel_ODE` (`18` tasks): Irregular shifted prefix trajectories (`mem_conv = None`) requiring origin-shift-invariant memory state extraction + 2-state RK4 integration.
   - `F02_AdiabaticCoarse_Fold` (`18` tasks): Certified fast-damped macro-scale slow queries (`M` unit cost halved via coarse-channel closure certificate).
   - `F03_DualHorizon_EndogenousReversal` (`18` tasks): Short-window positive gain (`H <= 0.48`) masking long-horizon reversal (`H in [19, 25]`), requiring `probing.py` task-risk VOI probe acquisition.
   - `F04_BeforeAfter_NormalizedHysteresis` (`18` tasks): Dimensionless `tau_scale` perturbation contrast across 3 orders of magnitude (`0.004 .. 5.0`).
2. **Cluster II — Progressive Shared Base Subtrees, Gray-Zone Sublation & 2-Hop Wormholes (`F05`–`F08`, `72` tasks)**:
   - `F05_TwoLayer_ConjunctiveSiphon_Subtree` (`18` tasks): 2-layer conjunctive `C1 ∧ C2` siphon subtree across normalized scales.
   - `F06_NarrowPilot_HeteroParam_WideEscape` (`18` tasks): Collinear narrow-`s_grad` pilot with out-of-prior target `rho in [0.11, 0.23] U [0.85, 0.92]` testing soft warm-start + wide escape.
   - `F07_CriticalThreshold_GrayZone_Avalanche` (`18` tasks): Supercritical cavitation breakdown (`s_crit in [2.52, 2.64]`); Wave-1 refutation & sublation teaches the `[2.36, 2.68]` gray zone used proactively in Waves `2..6` without perturbing subcritical tasks (`s_grad <= 2.25`).
   - `F08_TwoHop_WormholeBridge_Transfer` (`18` tasks): Cross-disciplinary 2-hop wormhole bridges (`epidemic_mobility`, `thermal_fin`, `supply_bullwhip` -> `node_C_source_inland_hub`) under `33` polysemous scientific distractors (`K = 45`).
3. **Cluster III — `LLM-SRBench`-Style Transformed Equations & 5-Parameter Multi-Tree Hybrids (`F09`–`F12`, `72` tasks)**:
   - `F09_LSR_Transformed_HillInhibition` (`18` tasks): Cooperative Hill-2 dimer flux with competitive sink inhibition & linear bleed (`4` parameters).
   - `F10_LSR_Transformed_BarrierSiphon` (`18` tasks): Eyring/Arrhenius barrier-activated flux coupled with 2-layer sink inhibition (`4` parameters).
   - `F11_SpatioTemporal_SiphonMemory_5P` (`18` tasks): 5-parameter spatio-temporal siphon-driven 2-state RK4 memory ODE (`K = 45 -> 46` online consolidation).
   - `F12_SinkCoupled_DualHorizon_5P` (`18` tasks): 5-parameter sink-coupled dual-horizon adaptation (`K = 47 -> 48` online consolidation).
4. **Adversarial OOS Honesty Suite (`OOS_Adversarial_NonStationary`, `24` tasks)**: High-frequency non-stationary chaotic dynamics paired with seductive lure signatures.

---

## 2. Master `2,160`-Episode Results (`240` Tasks × `3` Subscription Models × `3` Arms)

| Model | Controlled Arm | In-Scope Strict Pass (`216`) | OOS Honesty (`24`) | False Confirms | API Calls | Effective Tokens | Total Composite Cost `C` | Mean `C` / Task |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gpt-6-luna`** | **`Hegel_M5_LiveGPT`** | **`199/216` (`92.13%`)** | **`24/24`** | `6` | `605` | **`5,753,022`** | **`3,738.71`** | **`15.58`** |
| `gpt-6-luna` | `EqualAuth_Flat_LiveGPT` | `191/216` (`88.43%`) | `24/24` | `7` | `662` | `6,145,365` | `3,842.73` | `16.01` |
| `gpt-6-luna` | `Strong_Ordinary_LiveGPT` | `201/216` (`93.06%`) | `24/24` | `0` | `596` | `7,451,866` | `4,680.69` | `19.50` |
| **`gpt-6.1-terra`** | **`Hegel_M5_LiveGPT`** | **`214/216` (`99.07%`)** | **`24/24`** | **`0`** | **`455`** | **`2,953,788`** | **`3,603.73`** | **`15.02`** |
| `gpt-6.1-terra` | `EqualAuth_Flat_LiveGPT` | `203/216` (`93.98%`) | `24/24` | `4` | `488` | `3,275,907` | `3,642.66` | `15.18` |
| `gpt-6.1-terra` | `Strong_Ordinary_LiveGPT` | `211/216` (`97.69%`) | `24/24` | `0` | `442` | `3,955,888` | `4,452.14` | `18.55` |
| **`gpt-6.2-sol`** | **`Hegel_M5_LiveGPT`** | **`208/216` (`96.30%`)** | **`24/24`** | **`0`** | `431` | **`4,507,917`** | **`3,549.16`** | **`14.79`** |
| `gpt-6.2-sol` | `EqualAuth_Flat_LiveGPT` | `199/216` (`92.13%`) | `24/24` | `1` | `503` | `5,459,838` | `3,707.09` | `15.45` |
| `gpt-6.2-sol` | `Strong_Ordinary_LiveGPT` | `214/216` (`99.07%`) | `24/24` | `0` | `382` | `4,991,227` | `4,433.01` | `18.47` |

---

## 3. Six-Wave Continual Evolution Trajectory (`K = 45 -> 48`)

| Wave (`40` Tasks/Wave) | `luna + Hegel` | `luna + Flat` | `luna + Strong` | `terra + Hegel` | `terra + Flat` | `terra + Strong` | `sol + Hegel` | `sol + Flat` | `sol + Strong` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`wave_1` (`K=45` Cold)** | **`33/36`** (`624.2C`) | `26/36` (`660.4C`) | `27/36` (`814.8C`) | **`36/36`** (`589.5C`) | `29/36` (`644.7C`) | `33/36` (`780.1C`) | **`36/36`** (`590.3C`) | `27/36` (`626.9C`) | `36/36` (`742.1C`) |
| **`wave_2` (`K=48` Live)** | `31/36` (`603.0C`) | `31/36` (`628.3C`) | `33/36` (`765.2C`) | **`34/36`** (`592.2C`) | `33/36` (`595.6C`) | `34/36` (`721.8C`) | `33/36` (`591.0C`) | `33/36` (`617.3C`) | `34/36` (`731.4C`) |
| **`wave_3` (`K=48` Live)** | **`36/36`** (`616.9C`) | `35/36` (`638.2C`) | `35/36` (`773.0C`) | **`36/36`** (`603.8C`) | `36/36` (`602.6C`) | `36/36` (`741.0C`) | **`36/36`** (`600.3C`) | `36/36` (`631.1C`) | `36/36` (`737.7C`) |
| **`wave_4` (`K=48` Live)** | **`35/36`** (`627.4C`) | `34/36` (`648.0C`) | `36/36` (`776.3C`) | **`36/36`** (`610.2C`) | `35/36` (`594.7C`) | `36/36` (`738.3C`) | **`35/36`** (`596.3C`) | `33/36` (`599.4C`) | `36/36` (`739.8C`) |
| **`wave_5` (`K=48` Live)** | **`33/36`** (`634.6C`) | `32/36` (`641.3C`) | `35/36` (`771.8C`) | **`36/36`** (`605.1C`) | `35/36` (`601.3C`) | `36/36` (`735.8C`) | **`35/36`** (`588.9C`) | `35/36` (`618.7C`) | `36/36` (`739.9C`) |
| **`wave_6` (`K=48` Live)** | `31/36` (`632.4C`) | `33/36` (`626.4C`) | `35/36` (`779.6C`) | **`36/36`** (`602.9C`) | `35/36` (`603.7C`) | `36/36` (`735.1C`) | `33/36` (`582.4C`) | `35/36` (`613.6C`) | `36/36` (`742.1C`) |

---

## 4. Offline-to-Online Zero-GPU Amortization Over `N = 240` Tasks

- **Offline Seed Forest Formation (`K=45`)**: `27.608 C` (`0.0 GB` GPU VRAM, `< 10 ms` single-core CPU).
- **Online Continual Writeback (`K = 45 -> 48`: `node_O`, `node_P`, `node_S`)**: `5.40 C` (`3 × 1.80 C`).
- **Total Cold-Start + Continual Lifecycle Overhead**: `33.008 C`.

| Model | Baseline Compared | Break-Even Task `N*` (Full Cold-Start Lifecycle) | Net Online Savings (`N=240`) | Net Full Cold-Start Lifecycle Savings (`N=240`) |
| :--- | :--- | :---: | :---: | :---: |
| **`gpt-6-luna`** | vs `EqualAuth_Flat_LiveGPT` | **`N* = 34`** | **`+104.02 C`** (`+8` tasks) | **`+71.02 C`** |
| **`gpt-6-luna`** | vs `Strong_Ordinary_LiveGPT` | **`N* = 5`** | **`+941.98 C`** (`-20.1%` cost) | **`+908.98 C`** |
| **`gpt-6.1-terra`** | vs `EqualAuth_Flat_LiveGPT` | **`N* = 32`** | **`+38.93 C`** (`+11` tasks, `0` vs `4` false confirms) | **`+5.93 C`** |
| **`gpt-6.1-terra`** | vs `Strong_Ordinary_LiveGPT` | **`N* = 5`** | **`+848.42 C`** (`+3` tasks, `-19.1%` cost) | **`+815.41 C`** |
| **`gpt-6.2-sol`** | vs `EqualAuth_Flat_LiveGPT` | **`N* = 24`** | **`+157.93 C`** (`+9` tasks, `0` vs `1` false confirm) | **`+124.93 C`** |
| **`gpt-6.2-sol`** | vs `Strong_Ordinary_LiveGPT` | **`N* = 5`** | **`+883.85 C`** (`-19.9%` cost) | **`+850.85 C`** |
