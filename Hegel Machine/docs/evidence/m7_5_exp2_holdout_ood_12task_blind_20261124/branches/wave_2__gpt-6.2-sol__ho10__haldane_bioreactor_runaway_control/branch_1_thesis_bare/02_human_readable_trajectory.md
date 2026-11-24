# Forensic Post-Mortem Trajectory: `gpt-6.2-sol` | `Bare_ReAct_LiveGPT` | `ho10__haldane_bioreactor_runaway_control`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `Hegel-OOD-Holdout-12` |
| **Domain** | `Process Control / Substrate-Inhibited Exothermic Bioreactor` |
| **Wave / Wormhole Family** | `wave_2` / `fam_robust_control_and_safety` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6.2-sol` (`gpt-6.2-sol`) |
| **Arm** | `Bare_ReAct_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `DEFERRED_FOR_BLIND_SYNTHESIS` |
| **Turns Used / Max Turns** | `5 / 28` |
| **ATMS Nogoods Recorded** | `0` |
| **Staged Executable Wormholes / Runtime Calls** | `0` / `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `92,037` (`89,800` / `12,544` / `2,237`) |
| **Total Episode Wall Time (s)** | `93.26s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`9951 ms` LLM)
- **Rationale**: Inspect the simulator for the actual model, actuator delay, and observation interface before choosing and testing a controller.

```bash
cd /holdout_task && ls -la && cat plant_sim.py
```
- **Container Exit Code**: `0` (`318 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 12
drwxr-xr-x 2 root root 4096 Nov 24 09:12 .
drwxr-xr-x 1 root root 4096 Nov 24 09:12 ..
-rw-r--r-- 1 root root 2058 Nov 24 09:12 plant_sim.py
import numpy as np

NOMINAL_SCENARIOS = [
    {"name": "nominal", "mu0": 0.42, "Ks": 0.8, "Ki": 15.0, "ua_scale": 1.0, "heat_pulse": 0.0},
    {"name": "high_mu_low_Ki", "mu0": 0.48, "Ks": 0.8, "Ki": 12.0, "ua_scale": 0.82, "heat_pulse": 2.2},
    {"name": "low_mu_fouled", "mu0": 0.36, "Ks": 0.9, "Ki": 16.0, "ua_scale": 0.75, "heat_pulse": 1.5},
]

def run_closed_loop(compute_control_step, scen, seed=10):
    rng = np.random.default_rng(seed)
    dt = 0.05
    X, S, V, T = 1.5, 2.2, 4.0, 30.0
    Sf = 45.0; Yxs = 0.50; k_heat = 2.8; T_cool = 18.0; T_amb = 29.0
    mem = {}
    prev_u = {"feed_rate_L_per_h": 0.2, "coolant_flow_frac": 0.3}
    max_T, max_V = T, V
    for step in range(400):
        t = step * dt
        obs = {
            "t_h": t, "dt_h": dt, "V_L": V,
            "X_meas_g_L": max(0.0, X + rng.normal(0.0, 0.04)),
            "S_meas_g_L": max(0.0, S + rng.normal(0.0, 0.08)),
            "T_meas_C": T + rng.normal(0.0, 0.05),
        }
        u_cmd, mem = compute_control_step(obs, mem)
        F = float(np.clip(prev_u["feed_rate_L_per_h"], 0.0, 2.5))
        uc = float(np.clip(prev_u["coolant_flow_frac"], 0.0, 1.0))
        prev_u = u_cmd
        # Sub-step RK2
        for _ in range(5):
            hs = dt / 5.0
            mu = scen["mu0"] * np.exp(0.065 * (T - 30.0)) * max(S, 0.0) / (scen["Ks"] + max(S, 0.0) + (max(S, 0.0) ** 2) / scen["Ki"])
            dV = F
            dX = mu * X - (F / V) * X
            dS = -(mu / Yxs) * X + (F / V) * (Sf - S)
            q_rxn = k_heat * mu * X
            q_cool = 3.6 * scen["ua_scale"] * uc * (T - T_cool) / (V / 4.0)
            q_dist = scen["heat_pulse"] if (9.0 <= t <= 12.5) else 0.0
            dT = q_rxn - q_cool + 0.15 * (T_amb - T) + (F / V) * (28.0 - T) + q_dist
            V += hs * dV; X = max(1e-4, X + hs * dX); S = max(0.0, S + hs * dS); T += hs * dT
            max_T = max(max_T, T); max_V = max(max_V, V)
    return {"final_biomass_g": float(X * V), "max_T_C": float(max_T), "max_V_L": float(max_V), "safe": bool(max_T <= 34.5 and max_V <= 12.0)}
```

### Turn 2 — `execute_in_remote_docker` (`33156 ms` LLM)
- **Rationale**: Implement substrate and thermal feedback with delay-aware feed budgeting. Run the supplied scenarios to detect tracking, temperature, or biomass shortfalls before tuning.
- **Alternatives Weighed**: `Operate at the full volume and maximize biomass` (rejected: The requirement is a biomass floor; limiting carbon inventory leaves more cooling headroom under uncertainty.)

```bash
cat > /holdout_task/bioreactor_ctrl.py <<'PY'
"""Feedback control for the supplied delayed fed-batch simulator."""

from math import exp


def compute_control_step(obs: dict, state_memory: dict) -> tuple[dict, dict]:
    """Return the next feed and coolant commands and one-step memory."""
    dt = obs['dt_h']
    volume = obs['V_L']
    biomass = obs['X_meas_g_L']
    substrate = obs['S_meas_g_L']
    temp = obs['T_meas_C']
    pending_feed = state_memory.get('feed', 0.2)
    actuation_volume = volume + dt * pending_feed

    # Use the growth model to replace substrate as it is consumed, with
    # feedback on S kept below the peak over the stated inhibition range.
    mu = 0.42 * exp(0.065 * (temp - 30.0)) * substrate / (
        0.8 + substrate + substrate * substrate / 15.0
    )
    growth = mu * biomass
    substrate_slope = pending_feed / volume * (45.0 - substrate) - 2.0 * growth
    actuation_substrate = substrate + dt * substrate_slope
    feed = actuation_volume * (2.0 * growth + 1.8 * (2.35 - actuation_substrate)) / (45.0 - 2.35)

    # Limit carbon inventory as well as volume: after feeding 3.5 L,
    # complete conversion would give about 89 g of biomass in this model.
    feed_capacity = (7.5 - actuation_volume) / dt
    feed = max(0.0, min(2.5, feed, feed_capacity))

    # Cancel estimated heat production, then use temperature feedback
    # to handle plant mismatch and the unmeasured heat pulse.
    cooling_gain = 3.6 * 0.85 * (temp - 18.0) / (volume / 4.0)
    heat_load = 2.8 * growth + 0.15 * (29.0 - temp) + feed / volume * (28.0 - temp)
    coolant = max(0.0, min(1.0, (heat_load + 13.0 * (temp - 33.05)) / cooling_gain))
    if temp >= 33.85:
        coolant = 1.0

    return {'feed_rate_L_per_h': feed, 'coolant_flow_frac': coolant}, {'feed': feed}
PY
cd /holdout_task && python - <<'PY'
from plant_sim import NOMINAL_SCENARIOS, run_closed_loop
from bioreactor_ctrl import compute_control_step
for s in NOMINAL_SCENARIOS:
    for seed in (10, 123):
        print(s['name'], seed, run_closed_loop(compute_control_step, s, seed))
PY
```
- **Container Exit Code**: `127` (`376 ms`, timed_out=`False`)
- **Filesystem Touched**: `['/holdout_task/bioreactor_ctrl.py']`
- **Stderr (tail)**:
```text
bash: line 42: python: command not found
```

### Turn 3 — `execute_in_remote_docker` (`10215 ms` LLM)
- **Rationale**: Retry with python3 after the missing python executable. If a supplied scenario breaches a limit or misses the biomass floor, use the reported maxima and final mass to identify which part of the controller needs adjustment.

```bash
cd /holdout_task && python3 - <<'PY'
from plant_sim import NOMINAL_SCENARIOS, run_closed_loop
from bioreactor_ctrl import compute_control_step
for scenario in NOMINAL_SCENARIOS:
    for seed in (10, 123):
        print(scenario['name'], seed, run_closed_loop(compute_control_step, scenario, seed))
PY
```
- **Container Exit Code**: `0` (`645 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
nominal 10 {'final_biomass_g': 89.20210873357685, 'max_T_C': 33.194044453786525, 'max_V_L': 7.500000000000001, 'safe': True}
nominal 123 {'final_biomass_g': 89.20210649067099, 'max_T_C': 33.164539335638494, 'max_V_L': 7.500000000000001, 'safe': True}
high_mu_low_Ki 10 {'final_biomass_g': 89.20738132640896, 'max_T_C': 33.31796195331711, 'max_V_L': 7.500000000000001, 'safe': True}
high_mu_low_Ki 123 {'final_biomass_g': 89.20736924063463, 'max_T_C': 33.35908986031595, 'max_V_L': 7.4999999999999964, 'safe': True}
low_mu_fouled 10 {'final_biomass_g': 89.19468288559244, 'max_T_C': 33.23942383339883, 'max_V_L': 7.4999999999999964, 'safe': True}
low_mu_fouled 123 {'final_biomass_g': 89.19473401922309, 'max_T_C': 33.26111718290237, 'max_V_L': 7.4999999999999964, 'safe': True}
```

### Turn 4 — `execute_in_remote_docker` (`20569 ms` LLM)
- **Rationale**: The supplied scenarios passed. Test combined perturbation endpoints and multiple noise seeds to detect coupled safety or productivity failures that would require retuning. Check raw command bounds because simulator clipping could otherwise hide an interface bug.

```bash
cd /holdout_task && ls -l bioreactor_ctrl.py && PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from itertools import product
from math import isfinite
from plant_sim import NOMINAL_SCENARIOS, run_closed_loop
from bioreactor_ctrl import compute_control_step

# Check the command contract as well as the simulator results; the simulator clips commands.
def checked_step(obs, memory):
    command, memory = compute_control_step(obs, memory)
    assert isinstance(memory, dict)
    assert set(command) == {'feed_rate_L_per_h', 'coolant_flow_frac'}
    for key, upper in [('feed_rate_L_per_h', 2.5), ('coolant_flow_frac', 1.0)]:
        assert isinstance(command[key], float) and isfinite(command[key]) and 0 <= command[key] <= upper, (obs, command)
    return command, memory

scenarios = list(NOMINAL_SCENARIOS)
for mu, ki, ks, ua, pulse in product((0.42*.85, 0.42*1.15), (12., 18.), (0.8, 0.9), (0.75, 1.), (0., 2.2)):
    scenarios.append(dict(name=(mu,ki,ks,ua,pulse), mu0=mu, Ki=ki, Ks=ks, ua_scale=ua, heat_pulse=pulse))
results = []
for scenario, seed in product(scenarios, (0, 1, 10, 42, 123, 1000)):
    result = run_closed_loop(checked_step, scenario, seed)
    assert result['safe'] and result['final_biomass_g'] >= 62, (scenario, seed, result)
    results.append((result, scenario['name'], seed))
print(f'{len(results)} runs passed')
for key, select in [('final_biomass_g', min), ('max_T_C', max), ('max_V_L', max)]:
    result, name, seed = select(results, key=lambda entry: entry[0][key])
    print(f'{key}: {result[key]:.6f}; scenario={name}; seed={seed}')
PY
```
- **Container Exit Code**: `0` (`2559 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
-rw-r--r-- 1 root root 1751 Nov 24 09:13 bioreactor_ctrl.py
210 runs passed
final_biomass_g: 89.192472; scenario=(0.357, 12.0, 0.9, 1.0, 0.0); seed=42
max_T_C: 33.422781; scenario=(0.48299999999999993, 18.0, 0.8, 0.75, 0.0); seed=123
max_V_L: 7.500000; scenario=(0.357, 12.0, 0.9, 0.75, 0.0); seed=1000
```

### Turn 5 — `submit_completed_task` (`9122 ms` LLM)
- **Rationale**: The artifact is saved and verified. Testing covered the supplied scenarios, combined perturbation endpoints, and raw actuator command bounds.
- **Submitted Final Answer / Summary**: `Implemented `compute_control_step` in `/holdout_task/bioreactor_ctrl.py`. All 210 tested runs met the safety and productivity requirements. The lowest final biomass was 89.19 g, the highest temperature was 33.43 °C (rounded up), and the highest volume was 7.50 L. These results cover the tested scenarios and noise seeds, not every unseen case.`

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "Hegel-OOD-Holdout-12",
  "reward": 0.0,
  "strict_pass": false,
  "deferred_until_after_blind_synthesis": true,
  "verifier_wall_ms": 0.0
}
```
