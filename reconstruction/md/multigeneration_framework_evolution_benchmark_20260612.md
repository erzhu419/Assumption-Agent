# Multigeneration Framework Evolution Benchmark

- pass: True
- failed_gates: []
- generations: 5
- candidates: 40
- accepted/rejected: 33/7
- framework growth score: 0.7122
- margin vs local patch: 0.26
- margin vs raw wisdom: 0.39
- margin vs best ablation: 0.1
- cross-generation active survival count: 23
- old success preservation: 0.9707
- residual explanation: 0.8256
- prompt trick retained: 0
- core prior promotions: 0

| Variant | Growth | Old Success | Residual | Simulator Reduction | Regression |
| --- | --- | --- | --- | --- | --- |
| `no_framework_evolution` | `0.2922` | `0.91` | `0.5` | `0.0` | `0.1` |
| `local_patch_only` | `0.4522` | `0.92` | `0.62` | `0.0` | `0.07` |
| `raw_wisdom_generation` | `0.3222` | `0.86` | `0.55` | `0.0` | `0.12` |
| `simulator_without_conservative_gate` | `0.5822` | `0.89` | `0.72` | `0.4` | `0.08` |
| `conservative_gate_without_simulator` | `0.6122` | `0.965` | `0.78` | `0.0` | `0.018` |
| `full_framework_evolution_agent` | `0.7122` | `0.968` | `0.83` | `0.7422` | `0.011` |
