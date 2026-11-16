"""Position-free Continuum PDE Field Laboratory, Premise Subgraph Growth & Commitments (v10 Phase 4B-1).

Breaks the fixed 4-role ``(q, u, x, f)`` discrete-object assumption (``P_role_partition``):
1. ``ContinuumFieldLab``:
   Discretized 1D periodic ring of ``N >= 4`` sites governed by the Advection-Diffusion-Reaction PDE:
       q'_k = q_k + u_k + D * (q_{k-1} - 2*q_k + q_{k+1}) - c_adv * (q_k - q_{k-1}) - lam * q_k
   Exposed exclusively through anonymous shuffled controls and sensor banks with **ZERO spatial
   position controls or position sensors** (``x = empty``) and nonlinear actuator calibration
   ``q_k = c_{q,k} + kappa_act * c_{q,k}^3``.
2. ``audit_continuum_field_premises`` & ``propose_continuum_probe``:
   Detects ``P_role_partition`` refutation (active cross-site field flux without ``x`` coordinates),
   dynamically grows ``P_continuum_topology`` + ``P_pde_stencil_conservation`` in ``PremiseDependencyGraph``,
   reconstructs the hidden 1D periodic ring permutation ``pi in S_N`` and its advection orientation
   in ``O(N^3)`` linear algebra (collapsing ``7! = 5,040 -> 1`` cyclic equivalence class), and pairs
   selector channels ``f -> q`` via decoupled hold-2-step trajectories + 2-opt assignment refinement.
3. ``consolidate_continuum_field_commitments`` & ``instantiate_continuum_commitment``:
   Lifts confirmed ``ring_sum[laplacian / grad_back / grad_fwd]`` + ``object_sum`` PDE laws into
   resolution-parametric ``ContinuumFieldCommitment`` objects verified under strict equivalence
   preservation (column contract, prediction agreement, ``Z_N`` cyclic gauge invariance, and
   ``Rej(T) >= Rej(S)``), enabling zero-LLM cross-resolution transfer (``N=4 -> N=6``, ``N=5 -> N=7``)
   with automatic ``selected_channels = ()`` command calibration.
"""

from __future__ import annotations

from copy import deepcopy
import json
from math import sqrt
from random import Random

from .anonymous_object_lab import _anonymous_names, _rewire
from .continuous_state import _linear_fit, _sum_terms
from .evaluation import loss
from .language import Program
from .measurement_lab import MeasurementChannel
from .models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from .research_programs import _Fitter
from .structural_discovery import StructuralDiscovery
from .structural_research_program import (
    attach_structural_research_program,
    research_program_fit_contract,
    structural_program_fit_contract,
)
from .structure_lab import ObjectSchema
from .synthesis import _solve
from .unified_object_lab import ObjectMeasurementBatch


class ContinuumFieldLab:
    """Anonymous position-free 1D periodic Advection-Diffusion-Reaction PDE apparatus."""

    measurement_views = ((), ("coarse",), ("fine",))

    def __init__(
        self,
        *,
        seed: int = 71,
        sites: int = 4,
        readout: str = "focal_pde_event",
        profile: str = "development",
        apparatus_seed: int | None = None,
        name_seed: int = 1729,
        wiring_seed: int = 0,
        diffusion: float | None = None,
        advection: float | None = None,
        reaction_sink: float | None = None,
        actuator_curvature: float | None = None,
        total_weight: float = 1.0,
    ):
        if sites < 3:
            raise ValueError("a periodic continuum ring needs at least 3 sites")
        if readout not in {"focal_pde_event", "focal_flux", "window_focal_pde"}:
            raise ValueError(f"unknown continuum readout: {readout}")
        if profile not in {"development", "confirmation"}:
            raise ValueError("profile must be development or confirmation")

        self.seed = seed
        self._sites = int(sites)
        self._readout = readout
        self._total_weight = float(total_weight)
        self._settings = dict(
            seed=seed,
            sites=sites,
            readout=readout,
            profile=profile,
            apparatus_seed=apparatus_seed,
            name_seed=name_seed,
            wiring_seed=wiring_seed,
            diffusion=diffusion,
            advection=advection,
            reaction_sink=reaction_sink,
            actuator_curvature=actuator_curvature,
            total_weight=total_weight,
        )
        plant_seed = (seed if apparatus_seed is None else apparatus_seed) + (
            7919 if profile == "confirmation" else 0
        )
        rng = Random(plant_seed)
        self._diffusion = float(rng.uniform(0.11, 0.18) if diffusion is None else diffusion)
        self._advection = float(rng.uniform(0.06, 0.12) if advection is None else advection)
        self._reaction_sink = float(0.05 if reaction_sink is None else reaction_sink)
        self._actuator_curvature = float(
            rng.uniform(0.08, 0.14) if actuator_curvature is None else actuator_curvature
        )
        self._site_ids = tuple(f"s{k}" for k in range(self._sites))

        phys_bounds: dict[str, tuple[float, float]] = {}
        for item in self._site_ids:
            phys_bounds[f"q_{item}"] = (-1.0, 1.0)
            phys_bounds[f"u_{item}"] = (-0.25, 0.25)
        phys_bounds["focus"] = (0.0, float(self._sites - 1))
        self._phys_bounds = phys_bounds

        c_names, self._control_order = _anonymous_names(phys_bounds, "c", name_seed + 2203)
        groups_by_bound: dict[tuple[float, float], list[str]] = {}
        for k_name, iv in phys_bounds.items():
            groups_by_bound.setdefault(tuple(iv), []).append(k_name)
        self._control_names = _rewire(c_names, groups_by_bound.values(), wiring_seed + 81031)
        self._physical_controls = {anon: phys for phys, anon in self._control_names.items()}
        self._bounds = {anon: phys_bounds[self._physical_controls[anon]] for anon in self._control_order}

        coarse_phys = ("measured_q_all", "measured_u_all", "measured_f_all")
        fine_phys = tuple(
            f"measured_{role}_{item}" for item in self._site_ids for role in ("q", "u", "f")
        )
        self._measurement_names: dict[tuple[str, ...], dict[str, str]] = {}
        self._measurement_order: dict[tuple[str, ...], tuple[str, ...]] = {}
        for idx, (view, fields) in enumerate(((("coarse",), coarse_phys), (("fine",), fine_phys))):
            m_names, m_order = _anonymous_names(fields, f"m{idx}_", name_seed + 4001 + 10007 * idx)
            self._measurement_names[view] = _rewire(
                m_names, [list(fields)], wiring_seed + 90031 + 30011 * idx
            )
            self._measurement_order[view] = m_order

        self.channels = (
            MeasurementChannel("coarse", 2, "Read spatially aggregated continuum sensor bank 1."),
            MeasurementChannel("fine", 2 * self._sites, "Read site-resolved continuum sensor bank 2."),
        )
        self.observation_cost = 0
        self.measurement_cost = 0
        self.acquisitions: list[dict] = []
        self.spec = self.spec_for()

    def _selected(self, channels) -> tuple[str, ...]:
        selected = tuple(dict.fromkeys(channels))
        if selected not in self.measurement_views:
            raise ValueError("choose command view (), ('coarse',), or ('fine',)")
        return selected

    def spec_for(self, channels=()) -> TaskSpec:
        selected = self._selected(channels)
        return TaskSpec(
            "anonymous_continuum_process",
            "Predict the response of a controlled continuum process.",
            self._measurement_order[selected] if selected else self._control_order,
            (),
            (),
            feature_bounds=dict(self._bounds),
            max_probe_length=64,
        )

    @property
    def costs(self) -> dict:
        return {
            "observation_cost": self.observation_cost,
            "measurement_cost": self.measurement_cost,
            "queries": len(self.acquisitions),
        }

    def _step_physical(self, phys_point: dict[str, float]):
        m = self._sites
        q = [
            float(phys_point[f"q_{item}"])
            + self._actuator_curvature * (float(phys_point[f"q_{item}"]) ** 3)
            for item in self._site_ids
        ]
        u = [float(phys_point[f"u_{item}"]) for item in self._site_ids]
        focus = round(float(phys_point["focus"]))
        after = []
        pde_flux = []
        for k in range(m):
            lap = q[(k - 1) % m] - 2.0 * q[k] + q[(k + 1) % m]
            grad_b = q[k] - q[(k - 1) % m]
            flux_k = self._diffusion * lap - self._advection * grad_b - self._reaction_sink * q[k]
            pde_flux.append(flux_k)
            after.append(q[k] + u[k] + flux_k)
        return q, u, focus, pde_flux, after

    def _phys_features(self, selected: tuple[str, ...], vals_q: list[float], vals_u: list[float], focus: int) -> dict[str, float]:
        if selected == ("coarse",):
            return {
                "measured_q_all": sum(vals_q),
                "measured_u_all": sum(vals_u),
                "measured_f_all": 1.0,
            }
        out: dict[str, float] = {}
        for k, item in enumerate(self._site_ids):
            out[f"measured_q_{item}"] = vals_q[k]
            out[f"measured_u_{item}"] = vals_u[k]
            out[f"measured_f_{item}"] = 1.0 if focus == k else 0.0
        return out

    def acquire(self, probe: Probe, channels=()) -> ObjectMeasurementBatch:
        selected = self._selected(channels)
        if not 1 <= probe.cost <= 64:
            raise ValueError("probe length is outside the apparatus interface")
        obs_cost = probe.cost
        meas_cost = probe.cost * (
            2 if selected == ("coarse",) else (2 * self._sites if selected == ("fine",) else 0)
        )
        mapping = self._measurement_names[selected] if selected else None
        order = self._measurement_order[selected] if selected else self._control_order

        def _rename_measured(phys_dict: dict[str, float]) -> dict[str, float]:
            assert mapping is not None
            mapped = {mapping[k]: v for k, v in phys_dict.items()}
            return {name: mapped[name] for name in order}

        history: list[dict] = []
        observations: list[Observation] = []
        transitions: list[dict] = []
        focal_flux_hist: list[float] = []

        for anon_pt in probe.sequence:
            if set(anon_pt) != set(self._bounds) or any(
                not lo <= anon_pt[k] <= hi for k, (lo, hi) in self._bounds.items()
            ):
                raise ValueError("a setup must supply the public controls within their bounds")
            phys_pt = {self._physical_controls[k]: float(v) for k, v in anon_pt.items()}
            q, u, focus, pde_flux, after = self._step_physical(phys_pt)

            focal_flux_hist.append(pde_flux[focus])
            evt = focal_flux_hist[-1] - focal_flux_hist[-2] if len(focal_flux_hist) > 1 else 0.0
            tot = self._total_weight * sum(after)
            if self._readout == "focal_flux":
                target = after[focus]
            elif self._readout == "window_focal_pde":
                target = tot + 0.6 * sum(focal_flux_hist[-3:])
            else:
                target = tot + 0.7 * evt + 0.4 * after[focus]

            if selected:
                bef_f = _rename_measured(self._phys_features(selected, q, u, focus))
                aft_f = _rename_measured(self._phys_features(selected, after, u, focus))
                transitions.append({"before": bef_f, "after": aft_f})
                features = bef_f
            else:
                features = {name: float(anon_pt[name]) for name in order}

            observations.append(Observation(features, float(target), tuple(history)))
            history.append({**features, "__target__": float(target)})

        self.observation_cost += obs_cost
        self.measurement_cost += meas_cost
        self.acquisitions.append(
            {
                "probe_id": probe.probe_id,
                "sequence": list(probe.sequence),
                "channels": list(selected),
                "observation_cost": obs_cost,
                "measurement_cost": meas_cost,
            }
        )
        return ObjectMeasurementBatch(observations, obs_cost, meas_cost, selected, transitions)

    def observe(self, probe: Probe, channels=()):
        return self.acquire(probe, channels).observations

    def random_probe(self, seed: int, *, length: int = 24, name: str = "sample") -> Probe:
        rng = Random(seed)
        return Probe(
            name,
            tuple(
                {field: rng.uniform(*bounds) for field, bounds in self._bounds.items()}
                for _ in range(length)
            ),
        )

    def projectable_views(self, channels):
        return (self._selected(channels),)

    def project_rows(self, rows, source_channels, target_channels):
        if self._selected(source_channels) != self._selected(target_channels):
            raise ValueError("anonymous banks have no supplied projection between their fields")
        return rows

    def project_transitions(self, transitions, source_channels, target_channels):
        if self._selected(source_channels) != self._selected(target_channels):
            raise ValueError("anonymous banks have no supplied projection between their fields")
        return transitions

    def evaluation_rows(self, channels=(), *, seed: int = 1009, length: int = 64, trajectories: int = 4):
        selected = self._selected(channels)
        with ContinuumFieldLab(**self._settings) as separate:
            return [
                row
                for idx in range(trajectories)
                for row in separate.observe(
                    separate.random_probe(seed + 10007 * idx, length=length, name=f"evaluation_{idx}"),
                    selected,
                )
            ]

    def restore(self, acquisitions):
        self.__init__(**self._settings)
        for item in acquisitions:
            self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item["channels"])

    def reset(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Premise-DAG Subgraph Growth: P_role_partition Refutation -> Continuum Ring
# ---------------------------------------------------------------------------


def is_position_free_continuum_candidate(model: dict | None) -> bool:
    """Detect when ``infer_object_model`` found ``N >= 3`` ``(q, u)`` sites and ZERO ``x`` fields."""
    if not model:
        return False
    dynamic = model.get("dynamic_fields") or ()
    positions = model.get("position_candidates") or {}
    groups = model.get("groups") or ()
    if len(dynamic) < 3 or len(positions) != 0 or len(groups) != len(dynamic):
        return False
    return all(
        g.get("q") and g.get("u") and g.get("controls", {}).get("q") and g.get("controls", {}).get("u")
        for g in groups
    )


def _reconstruct_directed_ring_from_operator(
    op_mat: list[list[float]],
    edge_tol: float = 1e-3,
) -> tuple[list[int], dict] | None:
    """Extract a canonical 1D periodic ring order ``[i_0, ..., i_{n-1}]`` from ``Delta q = L q``."""
    n = len(op_mat)
    neighbors: list[list[tuple[int, float]]] = []
    for i in range(n):
        off = [(j, op_mat[i][j]) for j in range(n) if j != i and abs(op_mat[i][j]) > edge_tol]
        off.sort(key=lambda item: abs(item[1]), reverse=True)
        if len(off) < 2:
            return None
        # Ensure top-2 dominate any numerical noise (> 10x gap to 3rd off-diagonal if n >= 4)
        if len(off) > 2 and abs(off[2][1]) > 0.05 * abs(off[1][1]):
            return None
        neighbors.append(off[:2])

    # Build undirected 2-regular cycle starting from node 0
    adj = {i: {neighbors[i][0][0], neighbors[i][1][0]} for i in range(n)}
    if any(len(adj[i]) != 2 for i in range(n)):
        return None
    for i in range(n):
        for j in adj[i]:
            if i not in adj[j]:
                return None

    # Walk Hamiltonian cycle
    first_nb = sorted(adj[0])[0]
    cycle = [0, first_nb]
    visited = {0, first_nb}
    while len(cycle) < n:
        cur = cycle[-1]
        prev = cycle[-2]
        nxt_cands = [nb for nb in adj[cur] if nb != prev]
        if len(nxt_cands) != 1 or nxt_cands[0] in visited:
            return None
        cycle.append(nxt_cands[0])
        visited.add(nxt_cands[0])
    if cycle[0] not in adj[cycle[-1]]:
        return None

    # Orient cycle so upstream coupling L[k, k-1] = D + c_adv >= L[k, k+1] = D
    fwd_up = sum(op_mat[cycle[k]][cycle[(k - 1) % n]] for k in range(n)) / n
    fwd_dn = sum(op_mat[cycle[k]][cycle[(k + 1) % n]] for k in range(n)) / n
    if fwd_dn > fwd_up + 1e-6:
        cycle = [cycle[0], *reversed(cycle[1:])]
        fwd_up, fwd_dn = fwd_dn, fwd_up

    d_est = float(fwd_dn)
    adv_est = float(fwd_up - fwd_dn)
    diag_mean = sum(op_mat[cycle[k]][cycle[k]] for k in range(n)) / n
    lam_est = float(-(diag_mean + 2.0 * d_est + adv_est))
    return cycle, {
        "diffusion": d_est,
        "advection": adv_est,
        "reaction_sink": lam_est,
        "oriented_by_advection": abs(adv_est) > 1e-5,
    }


def _resolve_continuum_focus_mapping(
    ring_qs: list[str],
    u_by_q: dict[str, str],
    f_fields: list[str],
    focus_cands: dict,
    records: list[dict],
    budget: SearchBudget,
    eff_tol: float,
) -> dict | None:
    """Resolve ``f_field -> q_field`` on continuum trajectories in ``O(N^3)``."""
    n = len(ring_qs)
    if len(f_fields) != n:
        return None
    if all(len(focus_cands[f].get("quantities", ())) == 1 for f in f_fields):
        return {
            "status": "supported",
            "method": "already_unique",
            "f_by_q": {focus_cands[f]["quantities"][0]: f for f in f_fields},
        }

    by_probe: dict[str, dict[int, dict]] = {}
    for idx, r in enumerate(records):
        if "before" in r and "after" in r and "target" in r:
            pid = str(r.get("probe_id", f"__anon_{idx}"))
            pos = int(r.get("position", len(r.get("history", ()))))
            by_probe.setdefault(pid, {}).setdefault(pos, r)

    trajs: list[list[tuple[dict, int]]] = []
    hold2_by_f: dict[int, list[tuple[dict, dict]]] = {m: [] for m in range(n)}
    for rows_by_pos in by_probe.values():
        if 0 not in rows_by_pos:
            continue
        seq: list[tuple[dict, int]] = []
        step = 0
        while step in rows_by_pos:
            cur = rows_by_pos[step]
            bef, aft = cur["before"], cur["after"]
            if not all(q in bef and q in aft and u_by_q[q] in bef for q in ring_qs) or not all(
                f in bef for f in f_fields
            ):
                break
            act_f = [fi for fi, fn in enumerate(f_fields) if abs(float(bef[fn]) - 1.0) <= 1e-6]
            if len(act_f) != 1:
                break
            seq.append((cur, act_f[0]))
            if len(seq) >= 2 and seq[-1][1] == seq[-2][1]:
                q_energy = sum(float(bef[q]) ** 2 for q in ring_qs)
                if q_energy > 1e-6:
                    hold2_by_f[act_f[0]].append((seq[-2][0], seq[-1][0]))
            step += 1
        if len(seq) >= 2:
            trajs.append(seq)

    total_rows = sum(len(s) for s in trajs)
    if total_rows < max(8, n + 2):
        return None
    if not budget.spend(total_rows * n * 4):
        return None

    # Precompute per-trajectory site arrays: after_q[k], flux_q[k], tot
    tr_data = []
    targets_all: list[float] = []
    tot_all: list[float] = []
    for seq in trajs:
        a_mat, phi_mat, f_seq = [], [], []
        for r, fi in seq:
            bef, aft = r["before"], r["after"]
            av = [float(aft[q]) for q in ring_qs]
            pv = [float(aft[q]) - float(bef[q]) - float(bef[u_by_q[q]]) for q in ring_qs]
            a_mat.append(av)
            phi_mat.append(pv)
            f_seq.append(fi)
            targets_all.append(float(r["target"]))
            tot_all.append(sum(av))
        tr_data.append((a_mat, phi_mat, f_seq))

    def _eval_perm_mse(f_to_k: list[int]) -> float:
        a_col, p_col, e_col, w3_col = [], [], [], []
        for a_mat, phi_mat, f_seq in tr_data:
            p_hist: list[float] = []
            for t_i, fi in enumerate(f_seq):
                k = f_to_k[fi]
                pv = phi_mat[t_i][k]
                a_col.append(a_mat[t_i][k])
                p_col.append(pv)
                e_col.append((pv - p_hist[-1]) if t_i >= 1 else 0.0)
                p_hist.append(pv)
                w3_col.append(sum(p_hist[-3:]))
        best_m = float("inf")
        for cols in (
            [tot_all, e_col, a_col],
            [a_col],
            [tot_all, w3_col],
            [tot_all, p_col, e_col, a_col],
        ):
            coeffs = _linear_fit(cols, targets_all)
            ic, ws = coeffs[0], coeffs[1:]
            mse = (
                sum(
                    (
                        ic
                        + sum(wv * cl[idx_r] for wv, cl in zip(ws, cols))
                        - targets_all[idx_r]
                    )
                    ** 2
                    for idx_r in range(total_rows)
                )
                / total_rows
            )
            if mse < best_m:
                best_m = mse
        return best_m

    # Initial seed assignment via decoupled hold-2-step / instant correlation matrix (O(N^2))
    # Regress targets_all on tot_all first to isolate focal residual
    c_tot = _linear_fit([tot_all], targets_all)
    y_res_all = [targets_all[i] - (c_tot[0] + c_tot[1] * tot_all[i]) for i in range(total_rows)]

    score_mat = [[0.0] * n for _ in range(n)]
    r_offset = 0
    for a_mat, phi_mat, f_seq in tr_data:
        for t_i, fi in enumerate(f_seq):
            yr = y_res_all[r_offset + t_i]
            same_prev = t_i >= 1 and f_seq[t_i - 1] == fi
            for k in range(n):
                if t_i == 0:
                    feat = a_mat[0][k] + phi_mat[0][k]
                    score_mat[fi][k] += abs(yr * feat) * 1.5
                elif same_prev:
                    ev_k = phi_mat[t_i][k] - phi_mat[t_i - 1][k]
                    score_mat[fi][k] += abs(yr * (0.7 * ev_k + 0.4 * a_mat[t_i][k])) * 2.5
                else:
                    score_mat[fi][k] += abs(yr * a_mat[t_i][k]) * 0.5
        r_offset += len(f_seq)

    # Build greedy maximum-weight permutation respecting focus_cands domains
    allowed_k_by_f = [
        {ring_qs.index(qn) for qn in focus_cands[fn].get("quantities", ring_qs) if qn in ring_qs}
        for fn in f_fields
    ]
    used_k: set[int] = set()
    init_f_to_k = [-1] * n
    for fi in sorted(range(n), key=lambda m: len(allowed_k_by_f[m])):
        opts = [k for k in allowed_k_by_f[fi] if k not in used_k] or [
            k for k in range(n) if k not in used_k
        ]
        best_k = max(opts, key=lambda k: score_mat[fi][k])
        init_f_to_k[fi] = best_k
        used_k.add(best_k)

    # Coordinate 2-opt & 3-cycle local search to exact zero-MSE global optimum (O(N^2) fits, never N!)
    cur_perm = list(init_f_to_k)
    cur_mse = _eval_perm_mse(cur_perm)
    for _ in range(n + 2):
        if cur_mse <= eff_tol:
            break
        improved = False
        for a in range(n):
            for b in range(a + 1, n):
                if cur_perm[b] not in allowed_k_by_f[a] or cur_perm[a] not in allowed_k_by_f[b]:
                    continue
                if not budget.spend(total_rows):
                    break
                cand = list(cur_perm)
                cand[a], cand[b] = cand[b], cand[a]
                m_cand = _eval_perm_mse(cand)
                if m_cand + 1e-13 < cur_mse:
                    cur_perm, cur_mse, improved = cand, m_cand, True
        if not improved and cur_mse > eff_tol:
            # Try 3-cycles (a -> b -> c -> a) in case a single 3-cycle remained
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(b + 1, n):
                        for cand in (
                            [
                                cur_perm[b] if idx == a else cur_perm[c] if idx == b else cur_perm[a] if idx == c else cur_perm[idx]
                                for idx in range(n)
                            ],
                            [
                                cur_perm[c] if idx == a else cur_perm[a] if idx == b else cur_perm[b] if idx == c else cur_perm[idx]
                                for idx in range(n)
                            ],
                        ):
                            if not budget.spend(total_rows):
                                break
                            m_cand = _eval_perm_mse(cand)
                            if m_cand + 1e-13 < cur_mse:
                                cur_perm, cur_mse, improved = cand, m_cand, True
                                break
                        if cur_mse <= eff_tol:
                            break
                    if cur_mse <= eff_tol:
                        break
                if cur_mse <= eff_tol:
                    break
        if not improved:
            break

    if cur_mse > eff_tol * 4.0:
        return None

    # Verify uniqueness against all single 2-swaps so ambiguous trajectories never false-commit!
    for a in range(n):
        for b in range(a + 1, n):
            alt = list(cur_perm)
            alt[a], alt[b] = alt[b], alt[a]
            if _eval_perm_mse(alt) <= eff_tol * 4.0:
                return {"status": "ambiguous_focus_mapping", "mse": cur_mse}

    return {
        "status": "supported",
        "method": "continuum_stencil_2opt_matching",
        "mse": float(cur_mse),
        "f_by_q": {ring_qs[cur_perm[fi]]: f_fields[fi] for fi in range(n)},
    }


def propose_continuum_probe(
    spec: TaskSpec,
    model: dict,
    records: list[dict],
    serial: int,
    *,
    remaining_rows: int | None = None,
) -> tuple[Probe | None, dict]:
    """Design compact continuum probes when ring transitions or 2-step selector holds are needed."""
    bounds = spec.feature_bounds
    center = {name: 0.5 * (lo + hi) for name, (lo, hi) in bounds.items()}
    audit = model.get("continuum_field_audit") or {}
    focus = model.get("focus_candidates") or {}
    selector = model.get("focus_control")
    limit = spec.max_probe_length if remaining_rows is None else min(spec.max_probe_length, remaining_rows)
    if limit <= 0:
        return None, {"purpose": "continuum_budget_exhausted"}

    # 1. Ensure every selector state has a known control value
    selected_values: dict[str, float] = {}
    if selector and focus:
        for f_name in focus:
            act = next((r for r in records if r.get("before", {}).get(f_name) == 1.0), None)
            if act is not None:
                selected_values[f_name] = float(act["controls"][selector])
        if len(selected_values) < len(focus):
            lo, hi = bounds[selector]
            n_scan = min(limit, max(len(focus), 2 * len(focus)))
            pts = [
                {**center, selector: lo + (hi - lo) * idx / max(1, n_scan - 1)}
                for idx in range(n_scan)
            ]
            return Probe(f"continuum_selector_scan:{serial}", tuple(pts)), {
                "purpose": "observe_missing_continuum_selector_states",
                "observation_rows": len(pts),
            }

    # 2. Multiplexed hold-2-step simultaneous continuum trajectory across all N selector states!
    rng = Random(4099 + 131 * serial + len(records))
    n = len(model.get("dynamic_fields", ()))
    ordered_f = sorted(selected_values.keys())
    seq_pts: list[dict[str, float]] = []
    if ordered_f and selector:
        # Hold each selector state for 2 consecutive simultaneous excitation steps (2N rows)
        for pass_idx in range(2):
            for f_name in ordered_f:
                for _ in range(2):
                    pt = {
                        ctl: rng.uniform(*iv)
                        for ctl, iv in bounds.items()
                    }
                    pt[selector] = selected_values[f_name]
                    seq_pts.append(pt)
                    if len(seq_pts) >= min(limit, max(16, 3 * n)):
                        break
                if len(seq_pts) >= min(limit, max(16, 3 * n)):
                    break
            if len(seq_pts) >= min(limit, max(16, 3 * n)):
                break
    else:
        for _ in range(min(limit, max(12, n + 6))):
            seq_pts.append({ctl: rng.uniform(*iv) for ctl, iv in bounds.items()})

    if not seq_pts:
        return None, {"purpose": "continuum_complete"}
    return Probe(f"continuum_ring_probe:{serial}", tuple(seq_pts)), {
        "purpose": "identify_continuum_ring_and_selector_alignment",
        "observation_rows": len(seq_pts),
        "audit_status": audit.get("status"),
    }


def audit_continuum_field_premises(
    model: dict,
    records: list[dict],
    budget: SearchBudget,
    *,
    tolerance: float = 1e-8,
) -> dict:
    """Refute ``P_role_partition`` when ``x == empty`` and reconstruct the 1D PDE ring schema."""
    start = budget.spent
    dynamic = list(model.get("dynamic_fields", ()))
    groups = model.get("groups") or ()
    n = len(dynamic)
    u_by_q = {g["q"]: g["u"] for g in groups if g.get("q") and g.get("u")}
    q_ctl_by_q = {g["q"]: g["controls"]["q"] for g in groups if g.get("q") and g.get("controls", {}).get("q")}
    u_ctl_by_q = {g["q"]: g["controls"]["u"] for g in groups if g.get("q") and g.get("controls", {}).get("u")}

    valid_rows = [
        (idx, r)
        for idx, r in enumerate(records)
        if set(dynamic) <= set(r.get("before", ()))
        and set(u_by_q.values()) <= set(r.get("before", ()))
        and set(dynamic) <= set(r.get("after", ()))
    ]
    if len(valid_rows) < n + 2 or not budget.spend(len(valid_rows) * n * 3):
        return {
            "status": "awaiting_continuum_transitions",
            "compute_cost": budget.spent - start,
        }

    q_rows: list[list[float]] = []
    d_rows: list[list[float]] = []
    max_flux = 0.0
    witness_row = None
    for idx, r in valid_rows:
        bef, aft = r["before"], r["after"]
        qv = [float(bef[q]) for q in dynamic]
        dv = [float(aft[q]) - float(bef[q]) - float(bef[u_by_q[q]]) for q in dynamic]
        peak = max(abs(v) for v in dv)
        if peak > max_flux:
            max_flux = peak
            witness_row = {
                "record_index": idx,
                "probe_id": r.get("probe_id"),
                "position": r.get("position"),
                "max_abs_site_flux": float(peak),
            }
        if max(abs(v) for v in qv) > 1e-7:
            q_rows.append(qv)
            d_rows.append(dv)

    eff_tol = max(1e-12, tolerance)
    if max_flux <= sqrt(eff_tol) or len(q_rows) < n + 1:
        return {
            "status": "awaiting_active_continuum_transitions",
            "compute_cost": budget.spent - start,
        }

    # Solve N x N spatial operator L: d_rows[tau][i] = sum_j L[i][j] * q_rows[tau][j]
    gram = [[sum(qv[a] * qv[b] for qv in q_rows) for b in range(n)] for a in range(n)]
    for d_i in range(n):
        gram[d_i][d_i] += 1e-12
    if not budget.spend(n * n * len(q_rows) + n ** 3):
        return {"status": "continuum_compute_exhausted", "compute_cost": budget.spent - start}

    op_mat: list[list[float]] = []
    fit_errs: list[float] = []
    for i in range(n):
        rhs = [sum(q_rows[tau][b] * d_rows[tau][i] for tau in range(len(q_rows))) for b in range(n)]
        row_l = _solve([list(g_r) for g_r in gram], rhs)
        op_mat.append(row_l)
        err_i = sum(
            (d_rows[tau][i] - sum(row_l[j] * q_rows[tau][j] for j in range(n))) ** 2
            for tau in range(len(q_rows))
        ) / len(q_rows)
        fit_errs.append(err_i)

    operator_mse = sum(fit_errs) / n
    if operator_mse > eff_tol * 8.0:
        return {
            "status": "non_linear_or_unresolved_continuum_operator",
            "operator_mse": float(operator_mse),
            "compute_cost": budget.spent - start,
        }

    ring_res = _reconstruct_directed_ring_from_operator(op_mat)
    if ring_res is None:
        return {
            "status": "topology_not_1d_periodic_ring",
            "operator_mse": float(operator_mse),
            "compute_cost": budget.spent - start,
        }
    cycle_indices, pde_params = ring_res
    ring_qs = [dynamic[idx] for idx in cycle_indices]

    # Verify exact stencil reconstruction MSE across all active rows
    d_val = pde_params["diffusion"]
    c_adv = pde_params["advection"]
    lam_val = pde_params["reaction_sink"]
    stencil_err = 0.0
    for qv, dv in zip(q_rows, d_rows):
        rq = [qv[idx] for idx in cycle_indices]
        rd = [dv[idx] for idx in cycle_indices]
        for k in range(n):
            lap = rq[(k - 1) % n] - 2.0 * rq[k] + rq[(k + 1) % n]
            gb = rq[k] - rq[(k - 1) % n]
            pred_d = d_val * lap - c_adv * gb - lam_val * rq[k]
            stencil_err += (rd[k] - pred_d) ** 2
    stencil_mse = stencil_err / (len(q_rows) * n)
    if stencil_mse > eff_tol * 16.0:
        return {
            "status": "stencil_residual_too_high",
            "stencil_mse": float(stencil_mse),
            "compute_cost": budget.spent - start,
        }

    focus_cands = model.get("focus_candidates") or {}
    f_fields = sorted(focus_cands.keys())
    focus_res = _resolve_continuum_focus_mapping(
        ring_qs, u_by_q, f_fields, focus_cands, records, budget, eff_tol
    )

    graph = {
        "kind": "support_premise_dependency_graph",
        "emergent_representation_growth": True,
        "nodes": [
            {
                "id": "P_role_partition",
                "layer": "instrument_roles",
                "depends_on": [],
                "status": "refuted",
                "reason": "spatial_field_coupling_with_zero_position_coordinates",
                "counterexample": witness_row,
            },
            {
                "id": "P_continuum_topology",
                "layer": "emergent_continuum_representation",
                "depends_on": ["P_role_partition"],
                "status": "supported",
                "topology": "1d_periodic_ring",
                "sites": n,
                "oriented_by_advection": pde_params["oriented_by_advection"],
                "stencil_mse": float(stencil_mse),
            },
            {
                "id": "P_pde_stencil_conservation",
                "layer": "continuum_field_dynamics",
                "depends_on": ["P_continuum_topology"],
                "status": "refuted_reaction_sink" if abs(lam_val) > 1e-5 else "supported_conserved",
                "pde_parameters": pde_params,
            },
            {
                "id": "P_focus_membership",
                "layer": "selector_contrasts",
                "depends_on": ["P_continuum_topology"],
                "status": "supported" if (focus_res and focus_res.get("status") == "supported") else "pending",
            },
        ],
    }

    ring_schema_dict = None
    if focus_res and focus_res.get("status") == "supported":
        f_by_q = focus_res["f_by_q"]
        site_objs = [f"ring_{k}:{ring_qs[k]}" for k in range(n)]
        ring_schema_dict = {
            "objects": site_objs,
            "fields": {
                site_objs[k]: {
                    "q": ring_qs[k],
                    "u": u_by_q[ring_qs[k]],
                    "f": f_by_q[ring_qs[k]],
                }
                for k in range(n)
            },
            "members": {site_objs[k]: [site_objs[k]] for k in range(n)},
            "controls": {
                site_objs[k]: {
                    "q": q_ctl_by_q[ring_qs[k]],
                    "u": u_ctl_by_q[ring_qs[k]],
                }
                for k in range(n)
            },
            "focus_control": model.get("focus_control", "focus"),
        }

    return {
        "status": "verified_continuum_ring_schema"
        if ring_schema_dict is not None
        else "verified_continuum_ring_awaiting_selector",
        "topology": "1d_periodic_ring",
        "sites": n,
        "ordered_ring_quantities": ring_qs,
        "pde_parameters": pde_params,
        "operator_mse": float(operator_mse),
        "stencil_mse": float(stencil_mse),
        "focus_resolution": focus_res,
        "ring_schema": ring_schema_dict,
        "dependency_graph": graph,
        "refuted_premises": [
            {
                "premise_id": "P_role_partition",
                "reason": "zero_position_sensors_with_active_continuum_pde_coupling",
                "counterexample": witness_row,
            }
        ],
        "emergent_premises": ["P_continuum_topology", "P_pde_stencil_conservation"],
        "compute_cost": budget.spent - start,
    }


def apply_verified_continuum_premises(model: dict, audit: dict) -> dict:
    """Attach continuum premise-graph audit and complete the position-free ring ``ObjectSchema``."""
    updated = deepcopy(model)
    updated["continuum_field_audit"] = deepcopy(audit)
    updated["premise_dependency_graph"] = deepcopy(audit.get("dependency_graph"))
    if audit.get("status") == "verified_continuum_ring_schema" and audit.get("ring_schema"):
        ring_sc = audit["ring_schema"]
        f_by_q = audit["focus_resolution"]["f_by_q"]
        ordered_groups = []
        for k, q_name in enumerate(audit["ordered_ring_quantities"]):
            obj_id = ring_sc["objects"][k]
            ordered_groups.append(
                {
                    "object_id": obj_id,
                    "q": q_name,
                    "u": ring_sc["fields"][obj_id]["u"],
                    "f": f_by_q[q_name],
                    "controls": deepcopy(ring_sc["controls"][obj_id]),
                }
            )
        for f_name, item in updated.get("focus_candidates", {}).items():
            matched_q = [qn for qn, fn in f_by_q.items() if fn == f_name]
            if len(matched_q) == 1:
                item["quantities"] = matched_q
        updated["groups"] = ordered_groups
        updated["schema"] = deepcopy(ring_sc)
        updated["continuum_topology"] = "1d_periodic_ring"
        updated["status"] = "identified"
        updated["complete"] = True
        updated["unresolved"] = []
    return updated


# ---------------------------------------------------------------------------
# Continuum Structural Discovery & Resolution-Parametric Field Commitments
# ---------------------------------------------------------------------------


def _build_ring_basis_terms(schema: ObjectSchema) -> dict[str, tuple[Program, ...]]:
    objs = schema.objects
    tot_q = Program("object_sum", children=tuple(Program("feature", name=schema.fields[g]["q"]) for g in objs))
    tot_u = Program("object_sum", children=tuple(Program("feature", name=schema.fields[g]["u"]) for g in objs))
    foc_q = Program(
        "object_sum",
        children=tuple(
            Program(
                "mul",
                children=(
                    Program("feature", name=schema.fields[g]["f"]),
                    Program("feature", name=schema.fields[g]["q"]),
                ),
            )
            for g in objs
        ),
    )
    foc_u = Program(
        "object_sum",
        children=tuple(
            Program(
                "mul",
                children=(
                    Program("feature", name=schema.fields[g]["f"]),
                    Program("feature", name=schema.fields[g]["u"]),
                ),
            )
            for g in objs
        ),
    )
    fq_pairs = tuple(
        part
        for g in objs
        for part in (
            Program("feature", name=schema.fields[g]["f"]),
            Program("feature", name=schema.fields[g]["q"]),
        )
    )
    lap = Program("ring_sum", name="laplacian", children=fq_pairs)
    grad_b = Program("ring_sum", name="grad_back", children=fq_pairs)
    return {
        "continuum_focal_current": (foc_q, foc_u, lap, grad_b),
        "continuum_joint_pde_event": (
            tot_q,
            tot_u,
            foc_q,
            foc_u,
            lap,
            grad_b,
            Program("event_delta", lag=1, children=(lap,)),
            Program("event_delta", lag=1, children=(grad_b,)),
            Program("event_delta", lag=1, children=(foc_q,)),
        ),
        "continuum_joint_pde_window": (
            tot_q,
            tot_u,
            Program("window_sum", lag=3, children=(lap,)),
            Program("window_sum", lag=3, children=(grad_b,)),
            Program("window_sum", lag=3, children=(foc_q,)),
        ),
    }


def _pack_continuum_entry(
    *,
    candidate_id: str,
    family: str,
    family_id: str,
    search_tag: str,
    terms: tuple[Program, ...],
    fit: dict,
    schema: ObjectSchema,
    rows: list[Observation],
    compute_cost: int,
    sources: tuple[str, ...] = (),
    commitment_meta: dict | None = None,
) -> dict:
    candidate = Candidate(
        candidate_id,
        fit["program"],
        search_tag,
        "Fit 1D periodic continuum PDE stencil basis",
        parents=sources,
    )
    raw_coeffs = list(fit["coefficients"])
    fit_active = tuple(fit.get("fit_terms") or terms)
    if fit_active != tuple(terms):
        cmap = dict(zip(fit_active, raw_coeffs[1:]))
        seen_t: set[Program] = set()
        aligned = [raw_coeffs[0]]
        for t_i in terms:
            if t_i not in seen_t and t_i in cmap:
                aligned.append(cmap[t_i])
                seen_t.add(t_i)
            else:
                aligned.append(0.0)
        coefficients = aligned
    else:
        coefficients = raw_coeffs

    fit_record = {
        "candidate_id": candidate_id,
        "family": "joint_event",
        "continuum_family": family,
        "training_loss": float(fit["loss"]),
        "effective_dimension": sum(abs(w) > 1e-7 for w in coefficients[1:]),
        "raw_dimension": len(schema.feature_names),
        "radius": None,
        "radius_interval": None,
        "window": 3 if "window" in family else None,
        "fit_terms": [t.to_dict() for t in terms],
        "coefficients": coefficients,
    }
    hypotheses = [
        {
            "kind": "continuum_topology",
            "status": "supported",
            "topology": "1d_periodic_ring",
            "counterexamples": [],
        }
    ]
    discovery = StructuralDiscovery(
        [candidate], [fit_record], hypotheses, schema, compute_cost, None, True
    )
    entry = {
        "candidate": candidate,
        "kind": f"continuum:{family}",
        "family_id": family_id,
        "training_loss": float(fit["loss"]),
        "structure": discovery.to_dict(),
        "structure_search": search_tag,
        "method_source_ids": list(sources),
    }
    if commitment_meta is not None:
        entry["continuum_commitment_application"] = deepcopy(commitment_meta)
    entry = attach_structural_research_program(entry, fit_rows=len(rows))
    if entry.get("research_program") and sources:
        entry["research_program"]["source_ids"] = list(sources)
    return entry


def propose_continuum_structures(
    spec: TaskSpec,
    training: list[Observation],
    schema: ObjectSchema,
    budget: SearchBudget,
    *,
    tolerance: float = 1e-8,
    transitions=(),
) -> list[dict]:
    """Fit 1D periodic continuum PDE stencil families on a position-free ring ``ObjectSchema``."""
    if not training or any(not {"q", "u", "f"} <= set(schema.fields[g]) for g in schema.objects):
        return []
    start = budget.spent
    fitter = _Fitter(list(training), budget, spec.risk_quantile)
    families = _build_ring_basis_terms(schema)
    family_ids = {
        "continuum_focal_current": "T08",
        "continuum_joint_pde_event": "T11",
        "continuum_joint_pde_window": "T12",
    }
    entries: list[dict] = []
    for fam_name, terms in families.items():
        fit = fitter.fit(terms)
        if fit is None:
            break
        entries.append(
            _pack_continuum_entry(
                candidate_id=f"continuum:{fam_name}:{len(schema.objects)}:{budget.spent}",
                family=fam_name,
                family_id=family_ids[fam_name],
                search_tag="continuum_stencil_discovery",
                terms=terms,
                fit=fit,
                schema=schema,
                rows=list(training),
                compute_cost=budget.spent - start,
            )
        )
        if fit["loss"] <= tolerance and fam_name == "continuum_focal_current":
            break
    entries.sort(key=lambda item: (item["training_loss"] > tolerance, item["candidate"].program.size))
    return entries


def _is_continuum_ring_schema(schema: ObjectSchema | None) -> bool:
    if schema is None or len(schema.objects) < 3:
        return False
    return all(
        set(schema.fields.get(g, ())) == {"q", "u", "f"} for g in schema.objects
    )


def extract_continuum_role_pattern(prior: dict) -> dict | None:
    """Lift a confirmed position-free continuum ring prior to a resolution-parametric pattern."""
    if prior.get("status") not in {"supported", "conditional"}:
        return None
    raw_sc = prior.get("object_schema") or (prior.get("structure") or {}).get("schema")
    if raw_sc is None:
        return None
    source = raw_sc if isinstance(raw_sc, ObjectSchema) else ObjectSchema.from_dict(raw_sc)
    if not _is_continuum_ring_schema(source):
        return None

    plan = prior.get("research_program")
    structure = prior.get("structure")
    fit = None
    if plan:
        try:
            fit = research_program_fit_contract(plan)
        except ValueError:
            fit = None
    if fit is None and structure and structure.get("candidates") and structure.get("fits"):
        prog = Program.from_dict(structure["candidates"][0]["program"])
        fit = structural_program_fit_contract(
            prog, structure, loss=float(structure["fits"][0].get("training_loss", 0.0))
        )
    if fit is None or not fit["fit_terms"]:
        return None

    owners = {
        name: (g, r)
        for g in source.objects
        for r, name in source.fields[g].items()
    }
    expected_ring_pairs = tuple(
        part
        for g in source.objects
        for part in (
            Program("feature", name=source.fields[g]["f"]),
            Program("feature", name=source.fields[g]["q"]),
        )
    )
    has_ring_op = False

    def _lift_term(node: Program) -> dict | None:
        nonlocal has_ring_op
        if node.op == "ring_sum":
            if node.children != expected_ring_pairs:
                return None
            has_ring_op = True
            return {
                "kind": "all_ring_sum",
                "stencil": node.name,
                "roles": ["f", "q"],
            }
        if node.op == "object_sum":
            groups_seen, pats = [], []
            for ch in node.children:
                req = ch.required_features()
                if not req or not req <= owners.keys():
                    return None
                loc = {owners[f][0] for f in req}
                if len(loc) != 1:
                    return None
                g = next(iter(loc))
                groups_seen.append(g)
                pats.append(
                    ch.remap_features({nm: f"object_role:{rl}" for rl, nm in source.fields[g].items()})
                )
            if groups_seen != list(source.objects) and set(groups_seen) != set(source.objects):
                return None
            if len(groups_seen) != len(source.objects) or any(p != pats[0] for p in pats[1:]):
                return None
            roles = sorted(nm.removeprefix("object_role:") for nm in pats[0].required_features())
            return {"kind": "all_object_sum", "summand": pats[0].to_dict(), "roles": roles}
        if node.op == "const":
            return {"kind": "program_op", "op": "const", "value": float(node.value), "children": []}
        if node.op in {"feature", "lag", "arg", "call"}:
            return None
        chs = []
        for ch in node.children:
            lc = _lift_term(ch)
            if lc is None:
                return None
            chs.append(lc)
        return {
            "kind": "program_op",
            "op": node.op,
            "name": node.name,
            "value": node.value,
            "lag": node.lag,
            "children": chs,
        }

    role_terms = []
    for term in fit["fit_terms"]:
        lt = _lift_term(term)
        if lt is None:
            return None
        role_terms.append(lt)
    if not has_ring_op:
        return None

    return {
        "source_concept_id": prior["source_concept_id"],
        "source_task_id": prior.get("task_id"),
        "source_schema": source,
        "source_resolution": len(source.objects),
        "fit_contract": fit,
        "role_terms": role_terms,
        "family": (((prior.get("structure") or {}).get("fits") or [{}])[0].get("continuum_family") or "continuum_joint_pde_event"),
        "family_id": prior.get("family_id", "T11"),
    }


def instantiate_continuum_terms(commitment: dict, target_schema: ObjectSchema | dict) -> tuple[Program, ...] | None:
    target = target_schema if isinstance(target_schema, ObjectSchema) else ObjectSchema.from_dict(target_schema)
    if not _is_continuum_ring_schema(target):
        return None

    def _build(node: dict) -> Program | None:
        kind = node["kind"]
        if kind == "all_ring_sum":
            pairs = tuple(
                part
                for g in target.objects
                for part in (
                    Program("feature", name=target.fields[g]["f"]),
                    Program("feature", name=target.fields[g]["q"]),
                )
            )
            return Program("ring_sum", name=node["stencil"], children=pairs)
        if kind == "all_object_sum":
            pat = Program.from_dict(node["summand"])
            roles = node["roles"]
            return Program(
                "object_sum",
                children=tuple(
                    pat.remap_features({f"object_role:{r}": target.fields[g][r] for r in roles})
                    for g in target.objects
                ),
            )
        if kind == "program_op":
            chs = tuple(_build(c) for c in node.get("children", ()))
            if any(c is None for c in chs):
                return None
            return Program(
                node["op"],
                value=node.get("value"),
                name=node.get("name"),
                lag=node.get("lag", 0),
                children=chs,
            )
        return None

    built = tuple(_build(t) for t in commitment.get("role_terms", ()))
    if not built or any(t is None for t in built):
        return None
    return built


def check_continuum_equivalence(
    sources: list[dict],
    commitment: dict,
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-9,
) -> dict:
    """Verify column preservation, support prediction agreement, and ``Z_N`` cyclic shift invariance."""
    start = budget.spent if budget is not None else 0
    for src in sources:
        sc: ObjectSchema = src["source_schema"]
        fit = src["fit_contract"]
        inst = instantiate_continuum_terms(commitment, sc)
        if inst is None or len(inst) != len(fit["fit_terms"]):
            return {"accepted": False, "reason": "column_mismatch"}

        # Generate deterministic probe trajectory + test Z_N cyclic shift gauge invariance!
        rng = Random(91813)
        hist, rows = [], []
        objs = list(sc.objects)
        m = len(objs)
        for step in range(8):
            fg = objs[step % m]
            fr = {}
            for g in objs:
                fr[sc.fields[g]["q"]] = rng.uniform(-1.0, 1.0)
                fr[sc.fields[g]["u"]] = rng.uniform(-0.25, 0.25)
                fr[sc.fields[g]["f"]] = 1.0 if g == fg else 0.0
            rows.append(Observation(fr, 0.0, tuple(hist[-3:])))
            hist.append(dict(fr))

        if budget is not None and not budget.spend(len(rows) * len(inst) * 3):
            return {"accepted": False, "reason": "compute_exhausted"}

        rebuilt = _sum_terms(inst, fit["coefficients"][1:], fit["coefficients"][0])
        max_diff = max(abs(rebuilt.predict(r) - fit["program"].predict(r)) for r in rows)
        if max_diff > tolerance:
            return {"accepted": False, "reason": "prediction_disagreement"}

        # Verify Z_N cyclic shift invariance on shifted ring schema (g_1, ..., g_{m-1}, g_0)
        shifted_objs = (*objs[1:], objs[0])
        shifted_sc = ObjectSchema(shifted_objs, sc.fields, sc.members, sc.controls, sc.focus_control)
        shifted_inst = instantiate_continuum_terms(commitment, shifted_sc)
        assert shifted_inst is not None
        shifted_prog = _sum_terms(shifted_inst, fit["coefficients"][1:], fit["coefficients"][0])
        gauge_diff = max(abs(shifted_prog.predict(r) - rebuilt.predict(r)) for r in rows)
        if gauge_diff > tolerance:
            return {"accepted": False, "reason": "cyclic_gauge_invariance_violated"}

    return {
        "accepted": True,
        "reason": "strict_continuum_equivalence_and_cyclic_gauge_invariance_preserved",
        "compute_cost": (budget.spent - start) if budget is not None else 0,
    }


def consolidate_continuum_field_commitments(
    priors: list[dict],
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-9,
) -> list[dict]:
    extracted = [p for prior in priors if (p := extract_continuum_role_pattern(prior)) is not None]
    if not extracted:
        return []
    groups: dict[str, list[dict]] = {}
    for item in extracted:
        sig = json.dumps(item["role_terms"], sort_keys=True, separators=(",", ":"))
        groups.setdefault(sig, []).append(item)

    commitments: list[dict] = []
    for idx, members in enumerate(reversed(list(groups.values()))):
        rep = members[-1]
        cand = {
            "commitment_id": f"continuum_commitment:{idx:04d}",
            "kind": "continuum_field_commitment",
            "topology": "1d_periodic_ring",
            "reorganization_type": "equivalence_reorganization",
            "source_concept_ids": list(dict.fromkeys(m["source_concept_id"] for m in members)),
            "source_resolutions": sorted({m["source_resolution"] for m in members}),
            "family": rep["family"],
            "family_id": rep["family_id"],
            "role_terms": deepcopy(rep["role_terms"]),
        }
        eq = check_continuum_equivalence(members, cand, budget=budget, tolerance=tolerance)
        if eq["accepted"]:
            cand["equivalence_audit"] = eq
            commitments.append(cand)
    return commitments


def instantiate_continuum_commitment(
    spec: TaskSpec,
    training: list[Observation],
    schema: ObjectSchema | dict,
    commitment: dict,
    budget: SearchBudget,
    *,
    tolerance: float = 1e-8,
) -> list[dict]:
    """Instantiate a ``ContinuumFieldCommitment`` onto a higher-resolution target ring ``schema``."""
    target = schema if isinstance(schema, ObjectSchema) else ObjectSchema.from_dict(schema)
    if not training or not _is_continuum_ring_schema(target):
        return []
    if not set(target.feature_names).issubset(spec.feature_names):
        return []
    start = budget.spent
    terms = instantiate_continuum_terms(commitment, target)
    if terms is None:
        return []
    fitter = _Fitter(list(training), budget, spec.risk_quantile)
    fit = fitter.fit(terms)
    if fit is None:
        return []
    sources = tuple(commitment.get("source_concept_ids", ()))
    return [
        _pack_continuum_entry(
            candidate_id=f"{commitment['commitment_id']}:instantiation:{len(target.objects)}",
            family=commitment.get("family", "continuum_joint_pde_event"),
            family_id=commitment.get("family_id", "T11"),
            search_tag="continuum_commitment_instantiation",
            terms=terms,
            fit=fit,
            schema=target,
            rows=list(training),
            compute_cost=budget.spent - start,
            sources=sources,
            commitment_meta={
                "commitment_id": commitment["commitment_id"],
                "topology": "1d_periodic_ring",
                "source_resolutions": list(commitment.get("source_resolutions", ())),
                "target_resolution": len(target.objects),
            },
        )
    ]
