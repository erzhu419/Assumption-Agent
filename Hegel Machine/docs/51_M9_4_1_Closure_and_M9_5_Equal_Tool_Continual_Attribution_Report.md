# Hegel M9.4.1 Audit Closure & M9.5 Equal-Tool Three-Arm Multi-World Continual Attribution Report

> **Machine-Generated Provenance Guarantee (`A10`)**: Every numerical figure and table in this report is programmatically generated from sealed JSON receipts by `scripts/generate_m9_5_report_from_sealed_json.py` after verifying episode-level conservation and SHA-256 manifests (`0` manual copy-paste numbers).

## 1. Executive Summary

Following the `M9.4` external audit (`Hegel_M9_4_Audit`), we executed a two-stage closure:
1. **`M9.4.1` Deterministic Gates `A–D` (`Commit 1`: `348a27d23`, `0` LLM calls, `22/22` regression tests passing)**:
   - Closed `C01 + C02` via isolated-worker process separation (`hegel_host_isolated_worker_scientific_probe_v4`) and the shared full-7-coefficient + canonical KKT kernel (`m9_4_1_shared_verification_kernel.py`), rejecting both `hereditary_weights = zeros(7)` and `hereditary_weights[2:7] = 1_000_000` in both public and private gates while verifying `34/34` sealed `M9.4` episodes (`0` discrepancies).
   - Closed `A04 + A05 + A07` + `F05–F09` via strict fail-closed 3-valued guards, `MechanismExecutableBinding` SHA/qualification/foreign-key enforcement, `Program("domain_guard", ...)` AST runtime domain checks, per-public-function stratified `F08` determinism capture, and `exact AND reachable` `F05` provenance funneling.
   - Closed `A01 + A10 + E04–E09` via atomic per-attempt raw ledger mirroring (`immediate_archive_dir`), non-destructive DAG verification append (`append_post_episode_verification_to_dag`), JSON-generated errata (`docs/50b_*`), and real `Stage S` counterfactual attribution.
2. **`M9.5` Equal-Tool 3-Arm Multi-World Continual Benchmark (`Frozen Commit`: `3607aed14e8b`, `32`-Way Concurrent `gpt-6.2-sol` CLI, `8` Truly Independent Worlds, `200` Live Episodes)**:
   - Every arm (`Archive_AutoReplay`, `Flat_ParameterizedCache`, `Full_Mechanism_Graph`) shares the **exact same 1-probe pre-LLM container execution service**, **exact same shared public scientific gate**, **exact same `/tmp/hegel_continual_archive/` (`solvers + calibration history`)**, **100% unified non-`T` input/output schemas**, and **identical non-terminal (`step < 8`) consolidation rules**.
   - Across **`n = 8` truly independent worlds** (`alpha_w01..04`, `beta_w01..04`, each with its own independent blind `Stage S` run and randomized anonymous `batch_01..08` schedule):
     - **`Full_Mechanism_Graph` vs `Archive_AutoReplay`**: Strict pass rate rises from **`43/64` (`67.19%`) -> `63/64` (`98.44%`)**, solver billed token proxy drops **`-64.26%` (`-8,168,004`)**, lifecycle token proxy drops **`-63.05%` (`-9,610,349`)**, winning **`8/8` independent worlds** (**exact two-sided sign-flip permutation $p = 0.007812$**).
     - **`Full_Mechanism_Graph` vs `Flat_ParameterizedCache` (Strong Flat Parameter + Recency Cache Baseline)**: Strict pass rate rises from **`56/64` (`87.50%`) -> `63/64` (`98.44%`)**, solver billed token proxy drops **`-61.49%` (`-7,254,215`)**, lifecycle token proxy drops **`-60.5%` (`-8,629,214`)**, winning **`8/8` independent worlds** (**exact two-sided sign-flip permutation $p = 0.007812$**).

---

## 2. Overall Three-Arm Continual Comparison (`8` Independent Worlds, `64` Stream Episodes / Arm)

| Metric | `Archive_AutoReplay` (Stateless Raw Replay) | `Flat_ParameterizedCache` (Flat `lstsq` + Recency Cache) | `Full_Mechanism_Graph` (Wing-Scoped `M8` + 3-Valued Guard) |
| :--- | :---: | :---: | :---: |
| **Strict Pass Rate (`64` eps)** | `43 / 64` (`67.19%`) | `56 / 64` (`87.50%`) | **`63 / 64` (`98.44%`)** |
| **Zero-Turn Pre-LLM Closures** | `18 / 64` | `21 / 64` | **`48 / 64`** |
| **Total LLM Turns (`Mean / Ep`)** | `585` (`9.14`) | `545` (`8.52`) | **`207` (`3.23`)** |
| **Pure Solver Token Cost Proxy** | `12,711,096` | `11,797,307` | **`4,543,092`** |
| **Consolidation Token Cost Proxy** | `2,532,086` | `2,464,740` | **`1,089,741`** |
| **Total Lifecycle Token Cost Proxy** | `15,243,182` | `14,262,047` | **`5,632,833`** |
| **Sum of Stream Wall-Clock Time (incl. Probes)** | `23,817.6 s` | `22,424.0 s` | **`9,830.1 s`** |

---

## 3. Causal Attribution by Continual Event Type (`8` Worlds per Event)

| Event | `Archive` Passes (`0`-Turn) | `Archive` Solver Proxy | `Flat` Passes (`0`-Turn) | `Flat` Solver Proxy | `Full` Passes (`0`-Turn) | `Full` Solver Proxy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `N` | `8/8` (`8` 0-turn) | `0` | `8/8` (`6` 0-turn) | `485,282` | `8/8` (`8` 0-turn) | `0` |
| `G1` | `8/8` (`8` 0-turn) | `0` | `8/8` (`8` 0-turn) | `0` | `8/8` (`8` 0-turn) | `0` |
| `T` | `7/8` (`0` 0-turn) | `2,276,902` | `7/8` (`0` 0-turn) | `2,355,350` | `7/8` (`0` 0-turn) | `2,333,584` |
| `C` | `8/8` (`0` 0-turn) | `2,320,376` | `8/8` (`1` 0-turn) | `1,982,924` | `8/8` (`0` 0-turn) | `2,209,508` |
| `R` | `2/8` (`0` 0-turn) | `2,272,424` | `6/8` (`3` 0-turn) | `1,279,823` | `8/8` (`8` 0-turn) | `0` |
| `C_prime` | `1/8` (`0` 0-turn) | `2,008,550` | `7/8` (`3` 0-turn) | `1,317,964` | `8/8` (`8` 0-turn) | `0` |
| `G2` | `5/8` (`2` 0-turn) | `1,582,212` | `6/8` (`0` 0-turn) | `2,398,098` | `8/8` (`8` 0-turn) | `0` |
| `M` | `4/8` (`0` 0-turn) | `2,250,632` | `6/8` (`0` 0-turn) | `1,977,866` | `8/8` (`8` 0-turn) | `0` |

### Key Causal Findings Across Events:
- **Negative Controls (`N`, `G1` — Full In-Batch Calibration Present)**: `Archive_AutoReplay` achieves `16/16` (`16` zero-turn closures, `0` tokens), `Flat_ParameterizedCache` achieves `16/16` (`14` zero-turn closures; `2` worlds had `T` immediately precede `N`, placing `T`'s multi-budget/secant solver in the recency slot!), and `Full_Mechanism_Graph` achieves `16/16` (`16` zero-turn closures, `0` tokens). Zero artificial `M8` credit is claimed when immediate calibration suffices (`A01` closed).
- **Catastrophic Recency Thrashing in `Flat_ParameterizedCache` on `R <-> C'` (`A02` & `E03` Closed)**: Under 100% unified input/output schemas, `Flat_ParameterizedCache` solves `3/8` `C'` episodes in `0` turns when `C'` immediately follows `C`, and `3/8` `R` episodes on Alpha, but suffers recency slot overwriting across `C -> R -> C'` transitions on Beta (`0` physical guards). `Full_Mechanism_Graph` evaluates shuffled 3-valued `ObservablePreconditionGuard`s to achieve **`8/8` zero-turn closures on `R` (`C0_base`) AND `8/8` zero-turn closures on unseen `C'` (`C1_extension`)**!
- **Multi-Branch Fanout (`fanout = 2`) & Sibling Wing Isolation on `G2 -> M` (`A06` Closed)**: On all `8/8` `G2` episodes, `Full_Mechanism_Graph`'s `M8VariationalPlanner` selects `AncestorReweight` on recalibrated Wing A (`v2 -> v3`, propagating simultaneously to **both `C0` and `C1`, `fanout = 2`**) while selecting `NoChange` (`0.0` weight drift) on EMI-spiked sibling Wing B and `0.0` prediction drift on `sentinel_acoustic_calibration`, closing **`8/8` `G2` (`C0@v3`) AND `8/8` `M` (`C1@v3`) in `0` LLM turns** (`100%` pass rate vs `75.0%` on `Flat` and `56.25%` on `Archive`).

---

## 4. Per-World Breakdown & World-Clustered Paired Inference (`n = 8` Independent Worlds)

| World (`Family`) | Anonymous Causal Schedule (`batch_01..08`) | `Archive` Pass (`Solver Proxy`) | `Flat` Pass (`Solver Proxy`) | `Full` Pass (`Solver Proxy`) |
| :--- | :--- | :---: | :---: | :---: |
| `alpha_w01` (`alpha`) | `T->N->G1->C->R->C_prime->G2->M` | `4/8` (`1,854,719`) | `8/8` (`1,436,882`) | `8/8` (`587,009`) |
| `alpha_w02` (`alpha`) | `N->T->G1->C->C_prime->R->G2->M` | `4/8` (`1,817,550`) | `5/8` (`1,649,906`) | `8/8` (`604,936`) |
| `alpha_w03` (`alpha`) | `N->G1->T->C->R->C_prime->G2->M` | `4/8` (`1,902,014`) | `7/8` (`1,485,955`) | `7/8` (`695,197`) |
| `alpha_w04` (`alpha`) | `T->G1->N->C->C_prime->R->G2->M` | `7/8` (`1,617,396`) | `5/8` (`1,442,331`) | `8/8` (`527,326`) |
| `beta_w01` (`beta`) | `T->N->G1->C->R->C_prime->G2->M` | `6/8` (`1,541,603`) | `8/8` (`1,537,073`) | `8/8` (`597,895`) |
| `beta_w02` (`beta`) | `N->T->G1->C->C_prime->R->G2->M` | `4/8` (`1,282,423`) | `8/8` (`1,134,391`) | `8/8` (`390,116`) |
| `beta_w03` (`beta`) | `N->G1->T->C->R->C_prime->G2->M` | `6/8` (`1,491,004`) | `8/8` (`1,627,160`) | `8/8` (`553,090`) |
| `beta_w04` (`beta`) | `T->G1->N->C->C_prime->R->G2->M` | `8/8` (`1,204,387`) | `7/8` (`1,483,609`) | `8/8` (`587,523`) |

- **`Full_Mechanism_Graph` vs `Flat_ParameterizedCache` (`n = 8` worlds)**:
  - Solver proxy mean delta per world: **`-906,776.88`** (95% bootstrap CI `[-983358.95, -833809.51]`, wins **`8/8` worlds**, exact two-sided sign-flip **$p = 0.007812$**).
  - Lifecycle proxy mean delta per world: **`-1,078,651.75`** (95% bootstrap CI `[-1166612.43, -989536.88]`, wins **`8/8` worlds**, exact two-sided sign-flip **$p = 0.007812$**).
- **`Full_Mechanism_Graph` vs `Archive_AutoReplay` (`n = 8` worlds)**:
  - Solver proxy mean delta per world: **`-1,021,000.50`** (95% bootstrap CI `[-1152486.36, -873511.88]`, wins **`8/8` worlds**, exact two-sided sign-flip **$p = 0.007812$**).
- **`Flat_ParameterizedCache` vs `Archive_AutoReplay` (`n = 8` worlds)**:
  - Strict passes improve `+13` (`43/64 -> 56/64`); solver proxy delta `-913,789` (`-7.19%`, lower in `6/8` worlds, exact two-sided $p = 0.195312$).

---

## 5. Complete Forensic & Campaign Cost Accounting (`E04–E09` Closed)

- **Main Confirmation Bundle (`docs/evidence/m9_5_equal_tool_continual_attribution_run2_20261127`)**:
  - Frozen Commit Start & End: `3607aed14e8b223d62779b2414d726d50e84870b` (`dirty_end = False`).
  - Python Source-Tree SHA-256 Start & End: `e5cf7b4c68e9dd996fe672da4b0e9697277354ec5970720523b85478e2a251af` (`100%` invariant).
  - Appended Provenance DAGs Verified: **`200 / 200`**.
  - Raw Provider Attempt Mirror Coverage: **`1544 / 1544`** (`100.0%` episode + compiler raw JSONs archived, `usage_unknown_attempt_count = 0`).
  - Shared `Stage S` 8-World Source Proxy: `2,139,800`; All-Stages Main Run Billed Proxy (incl. `Stage S` + `Stage S` compilation + 3 arms): `39,888,625` (`wall_clock_sec = 4,097.16 s` under `32`-way CLI concurrency).
- **Pre-Fix Aborted Phase-1 Ledger (`docs/evidence/m9_5_aborted_phase1_dag_hydration_bug_20261127`, Sealed on Commit `8a793fd46fdc`)**:
  - Preserved intact on disk with verified `SHA256SUMS.txt` (`105` raw attempt files, `0` stream episodes started, aborted on non-empty DAG `_cum_*` hydration check before `Commit 2b`): `2,876,595` billed proxy (`0` unknown usage).
  - Combined Campaign Total (`Main Confirmation Run + Preserved Aborted Phase-1 Infrastructure Run`): **`42,765,220`** billed proxy tokens (`100%` accounted on disk).
