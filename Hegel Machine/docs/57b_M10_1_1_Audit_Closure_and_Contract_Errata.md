# `M10.1.1` Audit Closure, Strict Semantic Receipt Contracts, and Forensic Errata (`U01–U12`, `0` Paid Calls)

> **Sealed Directory**: `docs/evidence/m10_1_1_audit_closure_and_contract_reverification_20261128/` (`0` mutations to prior archives or `60/60/61` scores).

- **`U02` Guard Collision Closure**: `86` historical unadopted post-overlap Hegel guard calls (`1,663,837 v2`) caught in `0` LLM calls across `43` steps.
- **`U03 + U04` Strict Semantic 4-Way FK & Transaction Separation**: `3` code-consistent vs `5` code-mismatched historical tickets classified; `8/8` negative controls (incl. missing index & concurrent `os.link` race) rejected; `a03/S1` (`193,777 v2` fallback) decoupled.
- **`U05 + U06` Compiler Anti-Wrapper/Dead-Branch & Docker Replays**: `3/3` semantic shell / unused-branch / identity negative controls rejected; `a04/S4` diamond accepted; isolated Docker replay (`exact_container_public_sha_match = False`) selects `dag_06` on `a02/S7` (`public_cv_y_mae = 0.027161`, diagnostic private `0.028308`) and keeps `dag_07` on `S8`.
- **`U09 + U11 + U12`**: `T05/T06` wired into Layer 2 & `M10.0` `validate_public_schema.py`; full online prefix hydration verified; `32/32` AST lifting sample scope documented.

| World ID | Arm | Transition | Same Old Node Code SHAs | Strict v2 Verdict |
|---|---|---|---|---|
| `conf_m101_a02` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_01 -> dag_03` | `True` | **`strict_semantic_valid`** |
| `conf_m101_a02` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_02 -> dag_06` | `False` | **`strict_semantic_rejected_historical_mismatch`** |
| `conf_m101_a02` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_05 -> dag_07` | `False` | **`strict_semantic_rejected_historical_mismatch`** |
| `conf_m101_a03` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_05 -> dag_07` | `False` | **`strict_semantic_rejected_historical_mismatch`** |
| `conf_m101_a04` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_06 -> dag_07` | `True` | **`strict_semantic_valid`** |
| `conf_m101_b01` | `Fixed_TwoStage_Modular_Online` | `dag_06 -> dag_07` | `True` | **`strict_semantic_valid`** |
| `conf_m101_b01` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_01 -> dag_05` | `False` | **`strict_semantic_rejected_historical_mismatch`** |
| `conf_m101_b02` | `Fixed_TwoStage_Modular_Online` | `dag_07 -> dag_10` | `False` | **`strict_semantic_rejected_historical_mismatch`** |
