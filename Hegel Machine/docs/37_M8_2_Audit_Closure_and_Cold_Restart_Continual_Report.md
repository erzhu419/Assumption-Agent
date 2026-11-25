# Hegel M8.2 Incremental Audit Closure, Pre-Payment Persistent Planner & 3-Process Cold-Restart Report

- **Branch**: `exp/m6-causal-wormhole-sandbox` (isolated worktree; mainline untouched)
- **Base Commit Audited by GPT-6**: `42d1ea688ebdccb35e6d893cc98ae0034f83df26`
- **M8.2 Code Tree SHA-256 (`src/` + `scripts/`)**: `ef68dad67d9ea26d10fa14b256a9e5013ade58b43d0844f0f19f2797f2628fe1`
- **Model Backend**: Official ChatGPT CLI (`codex`, `auth_mode == "chatgpt"`, `gpt-6.2-sol`)

## 1. Sovereign Triage of `Hegel_M8_1_Audit`

| Audit Finding | Decision | Rationale & Code Closure |
| :--- | :--- | :--- |
| **1.5 Boundary / Scope Controls** (`atexit` stdout spoof, `@shift` decorator, legacy cache fp, new-format stale wormhole, `fine` node `CANDIDATE_SYNTAX_ONLY`, dual `UNKNOWN` sparse gap) | **Adopted 100%** | Genuine security/epistemic boundary leaks. Closed in `m7_holdout_ood_blind_benchmark.py` (single-use POSIX pipe `w_fd`), `m6_8_b_prime_capability_bridge.py`, `m7_autogenous_forest_pretraining.py`, `m6_9_a_dialectical_synthesis_engine.py`, and `run_m7_closing_experiments.py`. |
| **2. Atomic `TaskModel` Commit, Authoritative Persistence, Role-Mapped Auto-Recall, Pre-Payment Budgeting, Reachable `AcquireObservation`, & Failed-Call Token Ledger** | **Adopted 100%** | Closed all 10 `planner_controls.py` counterexamples in `m8_persistent_variational_planner.py`: frozen `committed_nig_state` (`0.0` re-eval drift), fixed `gate_scale` (`0.0` batch-std drift), `terminal_memory_states` carry-over (`0.0` split error), exact `y -> 1e-6 * y` Jacobian (`< 1e-9` nats), `PhilosophicalForest` + `save_checkpoint` sync, autonomous `ReuseMechanism` across renamed columns, pre-payment `budget_remaining` gating, and full failed-call token retention. |
| **3. Four-State Container Verification (`SUPPORTED_IN_PARTIAL_SCOPE != VERIFIED`), Symmetric Minimax Tie-Break, Raw Replay Default, & Non-Trivial NIG-Beta Governance** | **Adopted 100%** | Closed in `m8_offline_container_planner.py`: `skill`/`nogood` activity can never forge `VERIFIED` or flip worse minimax error (`150mm > 100mm`); `load_historical_suite_episode_records` defaults to raw unpatched archive (`0` private-label reads); NIG-Beta posterior governs partial/unknown states (`12/12` on Holdout-12 vs `11/12` rule-only, `11/12` on Luna Holdout-12 vs `10/12` rule-only, `14/20` on raw unpatched Frontier-20 LOTO). |
| **4. Bridge Persistent Graph -> Real Docker `LocalContainerRepair` -> Checkpoint Writeback** | **Adopted & Streamlined** | Built `m8_container_mechanism_adapter.py` (`probe_public_cohort_contract_and_invariance` + `choose_and_execute_container_mechanism_action`), replacing expensive `2x` full-branch reruns on localized contract/invariance defects (`ho12`) with a `0.35x` 1-turn `GPT CLI` repair + persistent `container_invariance_guard` writeback. |
| **5. Multi-Hour Re-Sweeps / Heavy Infrastructure Bureaucracy** | **Streamlined / Rejected** | Replaced slow sweeps with a fast (`128.6 s`), SHA-256-locked **3-OS-Process Cold-Restart Continual Litmus** (`run_m8_2_cold_restart_continual_litmus.py`) + 15 deterministic adversarial control suites (`15/15` green). |

## 2. Calibrated Empirical Summary (`m8_2_cold_restart_autonomous_reuse_and_repair_20261125`)

- **3 Distinct OS Processes**: `PID 4061294 -> exit -> PID 4061654 -> exit -> PID 4063663 -> exit` (`128.57 s` total wall time, `8/8` files SHA-256 verified).
- **OS Process 1 (`PID 4061294`)**: `AncestorReweight` updated `shared_sensor_premise` (`v1 -> v2`); live `GPT CLI` (`gpt-6.2-sol`) synthesized `ConceptLift` `"5.0 * np.tanh(x['x1'] / 5.0) * np.tanh(1.4 * x['x2'])"` (`val_rmse: 1.41233 -> 0.09025`); saved `checkpoint_after_p1.json`.
- **OS Process 2 (`PID 4061654`, Cold Reload)**:
  - **Autonomous Role-Mapped Recall**: Without passing `lift_ids` (`codex_client=None`), `M8VariationalPlanner` autonomously selected `ReuseMechanism` on renamed columns `('chan_u', 'chan_v')` (`0` LLM calls, `val_rmse: 1.80033 -> 0.07513`) while rejecting on an unrelated linear task (`NoChange`).
  - **Real Docker `LocalContainerRepair` (`ho12`)**: Under tight budget `0.50` (`< 1.0`, blocking second-branch spawning), our label-free public probe caught both `amino_acid_substitution == None` and `CHR_SEG_` renaming crash (`ValueError`) on M8.1's broken `plasmid_audit.py`; 1 targeted `GPT CLI` repair turn cleared both public invariants, flipped the air-gapped Holdout verifier from `False -> True`, and persisted `mech_guard_structuraldepthandstrand_s5` to `checkpoint_after_p2.json`.
- **OS Process 3 (`PID 4063663`, Second Cold Reload & Prospective Verification)**:
  - Zero-shot passed a brand-new prospective OOD genomic cohort (`CHROM_BACKBONE_*` tags + `K3E` substitution, `0` extra LLM calls, `CommitPrimary1x`);
  - Autonomously recalled the nonlinear conduit on a prospective unseen variant `('probe_a', 'probe_b')` (`0` LLM calls, `val_rmse: 1.73244 -> 0.09539`);
  - Cross-process prediction drift on `task_sentinel_acoustics`: **`0.0`**.
