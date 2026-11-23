# External Independent Audit Guide — Hegel M6.9.1-A (`18/20 = 90.0%` Native Two-Branch & `17/20 = 85.0%` Native Single-Branch)

This package provides a complete, self-contained audit dossier for external reviewers (including GPT-6 web review and human peer audit) verifying closure of **100% of findings in `markdown/Hegel_M63_M67_Audit/`** and documenting the clean end-to-end **`Hegel M6.9.1-A`** Frontier empirical results on `exp/m6-causal-wormhole-sandbox`.

---

## 1. Headline Numbers (Zero Post-Hoc Regrade Patches on Hegel)

- **Model & Auth**: `gpt-6.2-sol` via official OpenAI ChatGPT Subscription Codex CLI (`auth_mode == "chatgpt"`, zero third-party proxies).
- **3-Run `Bare_ReAct_LiveGPT` (`1x = 28` Turns) Baseline**:
  - `Run 1` (`m6_1`): `12/20 (60.0%)` native (`13/20` w/ `05_regrade_receipt.json`)
  - `Run 2` (`m6_9_a` Thesis): `8/20 (40.0%)` native (`9/20` w/ post-hoc `AMR` `WORKDIR` fix)
  - `Run 3` (`m6_9_1_a` Thesis): `10/20 (50.0%)` native
  - **3-Run `Mean_Bare_1x`**: **`10.0/20 (50.0%)` native** (`10.67/20 = 53.3%` receipt-adjusted)
  - **Matched `2x` Clairvoyant Bare Oracle (`Oracle_BestOf2_Bare_2x = Run1 ∪ Run3`)**: **`15/20 (75.0%)`**
  - **3-Run Clairvoyant Bare Oracle (`Oracle_BestOf3_Bare_3x = Run1 ∪ Run2 ∪ Run3`)**: **`16/20 (80.0%)`**
- **Clean End-to-End `Hegel M6.9.1-A` (`docs/evidence/m6_9_1_a_two_branch_frontier_40ep_20261123`)**:
  - **Matched `1x` (`28`-Turn) Single-Branch `Hegel_M6_9_1_Antithesis_1x`**: **`17/20 (85.0%)` NATIVE** (`13/15 = 86.7%` on `Terminal-Bench-Science`! Beats simultaneous `Bare_Run3` by **`+7` tasks / `+35.0%`**, beats `Mean_Bare_1x` by **`+7.0` tasks / `+35.0%`**, and beats even `Oracle_BestOf3_Bare_3x` (`16/20`) at `1/3` the turn budget!).
  - **Declared `2x` (`56`-Turn) Two-Branch `Hegel_M6_9_1_A_Synthesized_2x`**: **`18/20 (90.0%)` NATIVE** (`14/15 = 93.3%` on `Terminal-Bench-Science`, `10/10 = 100.0%` on Wave 1, `8/10 = 80.0%` on Wave 2, **`0` synthesis selection regret**, **`0` false promotions vs `Thesis`**, beats `Oracle_BestOf2_Bare_2x` `15/20` by **`+3` tasks / `+15.0%`**!).

---

## 2. One-Command Independent Verification Commands

From `/home/erzhu419/mine_code/Asumption Agent/.worktrees/m6-causal-wormhole-sandbox/Hegel Machine`:

```bash
# 1. Run all 31 unit & regression tests covering P0a-P1b, S1-S3, and R1-R3
PYTHONPATH=src python3 -m pytest -q tests/test_m6_*.py

# 2. Run the 8-check cryptographic SHA-256, provenance DAG, zero-leakage, Skill∞ & container audit
python3 scripts/audit_m6_4_executable_wormhole_bundle.py \
  --evidence-dir docs/evidence/m6_9_1_a_two_branch_frontier_40ep_20261123 \
  --arm Hegel_M6_9_A_TwoBranch_LiveGPT
```

---

## 3. Contents of This Audit Package

1. `00_READ_ME_EXTERNAL_AUDITOR.md` — Executive summary & reproduction commands.
2. `01_Audit_Defect_to_Fix_and_Test_Traceability_Matrix.md` — Complete line-level traceability matrix closing every defect in `Hegel_M63_M67_Audit` plus `M6.9-A` -> `M6.9.1-A` scaffolding fixes (`S1–S3`, `R1–R3`, `P1–P4`).
3. `02_M6_9_1_A_Complete_20Task_Empirical_Ledger.json` — Machine-readable 20-task verdict, branch outcome, Skill∞ count, and SHA-256 terminal hash ledger across all 3 Bare runs and `M6.9`/`M6.9.1-A`.
4. `03_Architecture_and_Key_Implementation_Summary.md` — Architectural overview of `Stage 1 (B′)` and `Stage 2 (A)` preserving Plan-20 `Pi_fold`, Mori-Zwanzig memory kernels, `SharedBaseSubtreePrior`, role/unit/time alignment, Simpson's-paradox deconfounding, collinear PZT obstruction tomography, and zero-leakage public container synthesis.
