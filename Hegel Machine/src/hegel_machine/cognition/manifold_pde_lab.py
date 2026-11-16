"""2D Manifold, Non-Periodic Boundary (Neumann/Dirichlet), & General Graph-Laplacian PDE Lab (v10 Phase 5A).

Extends position-free continuum PDE discovery (Phase 4B-1) beyond 1D periodic rings to:
1. ``ManifoldPDELab``:
   Anonymous position-free apparatus (``x = empty``) supporting three manifold PDE classes:
     - ``topology="torus_2d"``: 2D periodic torus ``Z_{Nx} x Z_{Ny}`` (e.g. ``3x3 = 9 -> 4x4 = 16``,
       where ``16! = 2.09e13`` site permutations!) with anisotropic diffusion & advection:
         q'_{r,c} = q_{r,c} + u_{r,c} + D_x Lap_x(q) + D_y Lap_y(q) - c_x Grad_bx(q) - c_y Grad_by(q) - lam q_{r,c}
     - ``topology="bounded_chain"``: Non-periodic 1D domain with either reflecting **Neumann**
       (``d_n q = 0``, mass-conserving walls) or grounded absorbing **Dirichlet** (``q|_{dOmega} = 0``,
       wall leakage ``-D * (q_0 + q_{N-1})``) boundary conditions + zero-wall-flux upwind advection.
     - ``topology="general_graph"``: Reaction-diffusion PDE on an arbitrary anonymous non-regular
       connected graph ``G = (V, E, w)``:
         q'_u = q_u + u_u - D * sum_{v ~ u} w_{uv} (q_u - q_v) - lam * q_u
2. ``audit_manifold_pde_premises`` & ``propose_manifold_probe``:
   Recovers the ``N x N`` discrete PDE generator ``Delta q = L q`` in ``O(N^3)``, automatically
   classifies topology (``2d_periodic_torus``, ``1d_bounded_neumann``, ``1d_bounded_dirichlet``,
   ``general_graph_laplacian``), grows ``P_continuum_topology`` + ``P_boundary_condition`` +
   ``P_pde_stencil_conservation`` in ``PremiseDependencyGraph``, reconstructs canonical coordinates
   up to ``Aut(M)``, and resolves ``f -> q`` via bracketed hold-2-step probes + 2-opt/3-cycle search.
3. ``consolidate_manifold_pde_commitments`` & ``instantiate_manifold_commitment``:
   Lifts confirmed ``manifold_sum`` programs into ``ManifoldPDECommitment`` objects verified under
   ``Aut(M)`` gauge invariance, supporting cross-resolution transfer (``3x3 -> 4x4`` torus,
   ``N=5 -> N=8`` bounded chains, ``N=6 -> N=9`` irregular graphs) AND conservative boundary
   premise revision (``P_boundary_condition: neumann <-> dirichlet``).
"""

from __future__ import annotations

from copy import deepcopy
import json
from math import sqrt
from random import Random

from .anonymous_object_lab import _anonymous_names, _rewire
from .continuum_field_lab import (
    _resolve_continuum_focus_mapping,
    audit_continuum_field_premises,
    extract_continuum_u_links,
    is_position_free_continuum_candidate,
)
from .continuous_state import _linear_fit, _sum_terms
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


def canonical_irregular_graph_edges(sites: int, family: str = "branched_wheel") -> tuple[tuple[int, int, float], ...]:
    """Generate a deterministic connected non-regular graph (not cycle, path, or 4-regular torus)."""
    if sites < 4:
        raise ValueError("irregular graph needs at least 4 vertices")
    edges: dict[tuple[int, int], float] = {}

    def _add(u: int, v: int, w: float = 1.0) -> None:
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        edges[(a, b)] = float(w)

    if family == "branched_wheel":
        # Hub 0 connected to core spokes, outer chords, plus pendant/bridge branches so degrees are mixed (1,2,3,4+)
        ring_m = max(3, sites - 2)
        for k in range(1, ring_m):
            _add(k, (k % (ring_m - 1)) + 1, 1.0)
        _add(0, 1, 1.0)
        _add(0, 2, 1.0)
        if ring_m >= 4:
            _add(0, 3, 1.0)
        # Attach remaining vertices as asymmetric branches/bridges
        for extra in range(ring_m, sites):
            anchor = (extra - ring_m + 1) % ring_m
            _add(anchor, extra, 1.0)
            if extra + 1 < sites and extra % 2 == 1:
                _add(extra, (anchor + 2) % ring_m, 1.0)
    else:
        # Ladder + asymmetric pendants
        spine = max(3, (sites + 1) // 2)
        for k in range(spine - 1):
            _add(k, k + 1, 1.0)
        for idx in range(spine, sites):
            base = idx - spine
            _add(base, idx, 1.0)
            if idx > spine and idx < sites - 1:
                _add(idx - 1, idx, 1.0)

    return tuple((u, v, w) for (u, v), w in sorted(edges.items()))


class ManifoldPDELab:
    """Anonymous position-free 2D Torus, Bounded Neumann/Dirichlet, & Graph-Laplacian PDE apparatus."""

    measurement_views = ((), ("coarse",), ("fine",))

    def __init__(
        self,
        *,
        seed: int = 71,
        topology: str = "torus_2d",
        grid_shape: tuple[int, int] = (3, 3),
        sites: int | None = None,
        boundary_condition: str = "neumann",
        graph_family: str = "branched_wheel",
        graph_edges: tuple[tuple[int, int, float], ...] | None = None,
        readout: str = "focal_pde_event",
        profile: str = "development",
        apparatus_seed: int | None = None,
        name_seed: int = 1729,
        wiring_seed: int = 0,
        diffusion: float | None = None,
        diffusion_x: float | None = None,
        diffusion_y: float | None = None,
        advection: float | None = None,
        advection_x: float | None = None,
        advection_y: float | None = None,
        reaction_sink: float | None = None,
        actuator_curvature: float | None = None,
        total_weight: float = 1.0,
    ):
        if topology not in {"torus_2d", "bounded_chain", "general_graph"}:
            raise ValueError(f"unknown manifold PDE topology: {topology}")
        if readout not in {"focal_pde_event", "focal_flux", "window_focal_pde"}:
            raise ValueError(f"unknown manifold readout: {readout}")
        if profile not in {"development", "confirmation"}:
            raise ValueError("profile must be development or confirmation")
        if boundary_condition not in {"neumann", "dirichlet"}:
            raise ValueError("boundary_condition must be neumann or dirichlet")

        self.seed = seed
        self._topology = topology
        self._boundary_condition = boundary_condition
        self._readout = readout
        self._total_weight = float(total_weight)

        if topology == "torus_2d":
            nx, ny = int(grid_shape[0]), int(grid_shape[1])
            if nx < 3 or ny < 3:
                raise ValueError("2D periodic torus requires Nx >= 3 and Ny >= 3")
            self._grid_shape = (nx, ny)
            self._sites = nx * ny
            self._graph_edges: tuple[tuple[int, int, float], ...] = ()
        elif topology == "bounded_chain":
            m_sites = 5 if sites is None else int(sites)
            if m_sites < 4:
                raise ValueError("bounded chain PDE requires at least 4 sites")
            self._grid_shape = (m_sites, 1)
            self._sites = m_sites
            self._graph_edges = ()
        else:
            m_sites = 6 if sites is None else int(sites)
            if m_sites < 4:
                raise ValueError("general graph PDE requires at least 4 sites")
            self._grid_shape = (m_sites, 1)
            self._sites = m_sites
            self._graph_edges = (
                tuple((int(u), int(v), float(w)) for u, v, w in graph_edges)
                if graph_edges is not None
                else canonical_irregular_graph_edges(m_sites, graph_family)
            )

        self._settings = dict(
            seed=seed,
            topology=topology,
            grid_shape=grid_shape,
            sites=sites,
            boundary_condition=boundary_condition,
            graph_family=graph_family,
            graph_edges=graph_edges,
            readout=readout,
            profile=profile,
            apparatus_seed=apparatus_seed,
            name_seed=name_seed,
            wiring_seed=wiring_seed,
            diffusion=diffusion,
            diffusion_x=diffusion_x,
            diffusion_y=diffusion_y,
            advection=advection,
            advection_x=advection_x,
            advection_y=advection_y,
            reaction_sink=reaction_sink,
            actuator_curvature=actuator_curvature,
            total_weight=total_weight,
        )

        plant_seed = (seed if apparatus_seed is None else apparatus_seed) + (
            7919 if profile == "confirmation" else 0
        )
        rng = Random(plant_seed)
        self._diffusion = float(rng.uniform(0.10, 0.16) if diffusion is None else diffusion)
        self._diffusion_x = float(
            (rng.uniform(0.13, 0.17) if diffusion is None else diffusion)
            if diffusion_x is None
            else diffusion_x
        )
        self._diffusion_y = float(
            (rng.uniform(0.08, 0.11) if diffusion is None else 0.75 * float(diffusion))
            if diffusion_y is None
            else diffusion_y
        )
        self._advection = float(rng.uniform(0.05, 0.10) if advection is None else advection)
        self._advection_x = float(rng.uniform(0.06, 0.10) if advection_x is None else advection_x)
        self._advection_y = float(rng.uniform(0.03, 0.06) if advection_y is None else advection_y)
        self._reaction_sink = float(0.04 if reaction_sink is None else reaction_sink)
        self._actuator_curvature = float(
            rng.uniform(0.08, 0.13) if actuator_curvature is None else actuator_curvature
        )

        self._site_ids = tuple(f"s{k}" for k in range(self._sites))
        phys_bounds: dict[str, tuple[float, float]] = {}
        for item in self._site_ids:
            phys_bounds[f"q_{item}"] = (-1.0, 1.0)
            phys_bounds[f"u_{item}"] = (-0.25, 0.25)
        phys_bounds["focus"] = (0.0, float(self._sites - 1))
        self._phys_bounds = phys_bounds

        c_names, self._control_order = _anonymous_names(phys_bounds, "c", name_seed + 3307)
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
            m_names, m_order = _anonymous_names(fields, f"m{idx}_", name_seed + 5003 + 10007 * idx)
            self._measurement_names[view] = _rewire(
                m_names, [list(fields)], wiring_seed + 90031 + 30011 * idx
            )
            self._measurement_order[view] = m_order

        self._max_probe_len = max(64, 4 * self._sites)
        self.channels = (
            MeasurementChannel("coarse", 2, "Read spatially aggregated manifold sensor bank 1."),
            MeasurementChannel("fine", 2 * self._sites, "Read site-resolved manifold sensor bank 2."),
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
            "anonymous_manifold_pde_process",
            "Predict the response of a controlled spatial manifold process.",
            self._measurement_order[selected] if selected else self._control_order,
            (),
            (),
            feature_bounds=dict(self._bounds),
            max_probe_length=self._max_probe_len,
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
        pde_flux = [0.0] * m

        if self._topology == "torus_2d":
            nx, ny = self._grid_shape
            for r in range(nx):
                for c in range(ny):
                    idx = r * ny + c
                    n_up = ((r - 1) % nx) * ny + c
                    n_dn = ((r + 1) % nx) * ny + c
                    w_up = r * ny + ((c - 1) % ny)
                    w_dn = r * ny + ((c + 1) % ny)
                    lap_x = q[n_up] - 2.0 * q[idx] + q[n_dn]
                    lap_y = q[w_up] - 2.0 * q[idx] + q[w_dn]
                    gb_x = q[idx] - q[n_up]
                    gb_y = q[idx] - q[w_up]
                    pde_flux[idx] = (
                        self._diffusion_x * lap_x
                        + self._diffusion_y * lap_y
                        - self._advection_x * gb_x
                        - self._advection_y * gb_y
                        - self._reaction_sink * q[idx]
                    )
        elif self._topology == "bounded_chain":
            for k in range(m):
                if k == 0:
                    lap = (
                        (q[1] - q[0])
                        if self._boundary_condition == "neumann"
                        else (q[1] - 2.0 * q[0])
                    )
                    adv_term = q[0]
                elif k == m - 1:
                    lap = (
                        (q[m - 2] - q[m - 1])
                        if self._boundary_condition == "neumann"
                        else (q[m - 2] - 2.0 * q[m - 1])
                    )
                    adv_term = -q[m - 2]
                else:
                    lap = q[k - 1] - 2.0 * q[k] + q[k + 1]
                    adv_term = q[k] - q[k - 1]
                pde_flux[k] = (
                    self._diffusion * lap
                    - self._advection * adv_term
                    - self._reaction_sink * q[k]
                )
        else:
            for k in range(m):
                pde_flux[k] = -self._reaction_sink * q[k]
            for u_v, v_v, w_uv in self._graph_edges:
                diff_uv = self._diffusion * w_uv * (q[v_v] - q[u_v])
                pde_flux[u_v] += diff_uv
                pde_flux[v_v] -= diff_uv

        after = [q[k] + u[k] + pde_flux[k] for k in range(m)]
        return q, u, focus, pde_flux, after

    def _phys_features(
        self, selected: tuple[str, ...], vals_q: list[float], vals_u: list[float], focus: int
    ) -> dict[str, float]:
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
        if not 1 <= probe.cost <= self._max_probe_len:
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
        with ManifoldPDELab(**self._settings) as separate:
            return [
                row
                for idx in range(trajectories)
                for row in separate.observe(
                    separate.random_probe(
                        seed + 10007 * idx,
                        length=min(self._max_probe_len, length),
                        name=f"evaluation_{idx}",
                    ),
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
# Operator Topology Classification & Coordinate Reconstruction (O(N^3))
# ---------------------------------------------------------------------------


def _reconstruct_bounded_chain_from_operator(
    op_mat: list[list[float]],
    edge_tol: float = 1e-3,
) -> tuple[list[int], str, dict] | None:
    """Detect a 1D non-periodic path operator and classify Neumann vs Dirichlet walls."""
    n = len(op_mat)
    if n < 4:
        return None
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and (abs(op_mat[i][j]) > edge_tol or abs(op_mat[j][i]) > edge_tol):
                adj[i].add(j)
                adj[j].add(i)

    endpoints = [i for i in range(n) if len(adj[i]) == 1]
    interiors = [i for i in range(n) if len(adj[i]) == 2]
    if len(endpoints) != 2 or len(interiors) != n - 2:
        return None

    # Walk unique path from endpoints[0] to endpoints[1]
    path = [endpoints[0]]
    visited = {endpoints[0]}
    while len(path) < n:
        nxt = [nb for nb in adj[path[-1]] if nb not in visited]
        if len(nxt) != 1:
            return None
        path.append(nxt[0])
        visited.add(nxt[0])
    if path[-1] != endpoints[1]:
        return None

    # Orient chain so interior upstream coupling L[k, k-1] = D + c_adv >= L[k, k+1] = D
    up_mean = sum(op_mat[path[k]][path[k - 1]] for k in range(1, n - 1)) / max(1, n - 2)
    dn_mean = sum(op_mat[path[k]][path[k + 1]] for k in range(1, n - 1)) / max(1, n - 2)
    if dn_mean > up_mean + 1e-6:
        path = list(reversed(path))
        up_mean, dn_mean = dn_mean, up_mean

    d_est = float(dn_mean)
    adv_est = float(up_mean - dn_mean)
    if d_est <= edge_tol:
        return None

    # Column sums distinguish Neumann (zero wall leakage: col_sum == -lam everywhere)
    # from Dirichlet (grounded absorbing wall: boundary col_sum == -lam - D)!
    col_sums = [sum(op_mat[path[r]][path[c]] for r in range(n)) for c in range(n)]
    int_col_mean = sum(col_sums[1 : n - 1]) / (n - 2)
    bnd_col_mean = 0.5 * (col_sums[0] + col_sums[n - 1])
    lam_est = float(-int_col_mean)
    wall_leak = float(int_col_mean - bnd_col_mean)

    if abs(wall_leak) <= 0.15 * d_est:
        bc_type = "neumann"
    elif abs(wall_leak - d_est) <= 0.15 * d_est:
        bc_type = "dirichlet"
    else:
        return None

    # Verify exact stencil match against op_mat
    max_op_err = 0.0
    for r in range(n):
        for c in range(n):
            expected = 0.0
            if r == c:
                if r == 0:
                    expected = (
                        (-d_est - adv_est - lam_est)
                        if bc_type == "neumann"
                        else (-2.0 * d_est - adv_est - lam_est)
                    )
                elif r == n - 1:
                    expected = (
                        (-d_est - lam_est)
                        if bc_type == "neumann"
                        else (-2.0 * d_est - lam_est)
                    )
                else:
                    expected = -2.0 * d_est - adv_est - lam_est
            elif c == r - 1:
                expected = d_est + adv_est
            elif c == r + 1:
                expected = d_est
            max_op_err = max(max_op_err, abs(op_mat[path[r]][path[c]] - expected))
    if max_op_err > 5e-3:
        return None

    return path, bc_type, {
        "diffusion": d_est,
        "advection": adv_est,
        "reaction_sink": lam_est,
        "boundary_condition": bc_type,
        "wall_leakage_per_boundary_site": float(d_est if bc_type == "dirichlet" else 0.0),
        "oriented_by_advection": abs(adv_est) > 1e-5,
    }


def _reconstruct_2d_torus_from_operator(
    op_mat: list[list[float]],
    edge_tol: float = 1e-3,
) -> tuple[list[int], tuple[int, int], dict] | None:
    """Reconstruct canonical 2D periodic torus ``(Nx, Ny)`` row-major ordering from ``Delta q = L q``."""
    n = len(op_mat)
    if n < 9:
        return None

    adj: dict[int, set[int]] = {}
    for i in range(n):
        off = [(j, op_mat[i][j]) for j in range(n) if j != i and abs(op_mat[i][j]) > edge_tol]
        off.sort(key=lambda item: abs(item[1]), reverse=True)
        if len(off) < 4:
            return None
        if len(off) > 4 and abs(off[4][1]) > 0.05 * abs(off[3][1]):
            return None
        adj[i] = {j for j, _ in off[:4]}

    if any(len(adj[i]) != 4 for i in range(n)):
        return None
    for i in range(n):
        for j in adj[i]:
            if i not in adj[j]:
                return None

    factor_pairs = [
        (nx, n // nx)
        for nx in range(3, n // 3 + 1)
        if n % nx == 0 and (n // nx) >= 3
    ]
    if not factor_pairs:
        return None

    nbs0 = sorted(adj[0])
    w_tol = 3e-3
    valid_embeddings: list[tuple[tuple[float, float, float, float], tuple[int, int], list[int], dict]] = []

    for nx, ny in factor_pairs:
        for south in nbs0:
            w_up_x = op_mat[south][0]
            w_dn_x = op_mat[0][south]
            if w_dn_x <= edge_tol or w_up_x < w_dn_x - w_tol:
                continue
            for east in nbs0:
                if east == south:
                    continue
                # Orthogonal neighbors on C_Nx [] C_Ny (Nx, Ny >= 3) share exactly 2 common neighbors: 0 and (1, 1)
                if len(adj[south] & adj[east]) != 2:
                    continue
                w_up_y = op_mat[east][0]
                w_dn_y = op_mat[0][east]
                if w_dn_y <= edge_tol or w_up_y < w_dn_y - w_tol:
                    continue

                # Enumerate simple cycle along row 0 (0, east, ..., length Ny) and col 0 (0, south, ..., length Nx)
                def _axis_cycles(first_v: int, length: int, w_up: float, w_dn: float) -> list[list[int]]:
                    found: list[list[int]] = []

                    def _dfs(chain: list[int], used: set[int]) -> None:
                        if len(chain) == length:
                            last_v = chain[-1]
                            if (
                                0 in adj[last_v]
                                and abs(op_mat[0][last_v] - w_up) <= w_tol
                                and abs(op_mat[last_v][0] - w_dn) <= w_tol
                            ):
                                found.append(list(chain))
                            return
                        prev_v = chain[-1]
                        for cand in sorted(adj[prev_v] - used):
                            if (
                                abs(op_mat[cand][prev_v] - w_up) <= w_tol
                                and abs(op_mat[prev_v][cand] - w_dn) <= w_tol
                            ):
                                _dfs([*chain, cand], used | {cand})

                    _dfs([0, first_v], {0, first_v})
                    return found

                row0_cands = _axis_cycles(east, ny, w_up_y, w_dn_y)
                col0_cands = _axis_cycles(south, nx, w_up_x, w_dn_x)
                for row0 in row0_cands:
                    for col0 in col0_cands:
                        if set(row0[1:]) & set(col0[1:]):
                            continue
                        grid = [[-1] * ny for _ in range(nx)]
                        for c_i in range(ny):
                            grid[0][c_i] = row0[c_i]
                        for r_i in range(nx):
                            grid[r_i][0] = col0[r_i]
                        used_all = set(row0) | set(col0)
                        ok = True
                        for r_i in range(1, nx):
                            for c_i in range(1, ny):
                                common = (adj[grid[r_i - 1][c_i]] & adj[grid[r_i][c_i - 1]]) - {
                                    grid[r_i - 1][c_i - 1]
                                }
                                unused_common = [v for v in common if v not in used_all]
                                if len(unused_common) != 1:
                                    ok = False
                                    break
                                v_rc = unused_common[0]
                                grid[r_i][c_i] = v_rc
                                used_all.add(v_rc)
                            if not ok:
                                break
                        if not ok or len(used_all) != n:
                            continue

                        dx_est = float(w_dn_x)
                        cx_est = float(max(0.0, w_up_x - w_dn_x))
                        dy_est = float(w_dn_y)
                        cy_est = float(max(0.0, w_up_y - w_dn_y))
                        diag_mean = sum(op_mat[i][i] for i in range(n)) / n
                        lam_est = float(-(diag_mean + 2.0 * dx_est + 2.0 * dy_est + cx_est + cy_est))

                        # Verify full N x N operator agreement across all wrap-around torus edges
                        max_err = 0.0
                        for r_i in range(nx):
                            for c_i in range(ny):
                                u_id = grid[r_i][c_i]
                                n_up = grid[(r_i - 1) % nx][c_i]
                                n_dn = grid[(r_i + 1) % nx][c_i]
                                w_up_nb = grid[r_i][(c_i - 1) % ny]
                                w_dn_nb = grid[r_i][(c_i + 1) % ny]
                                exp_row = {
                                    u_id: -(2.0 * dx_est + 2.0 * dy_est + cx_est + cy_est + lam_est),
                                    n_up: dx_est + cx_est,
                                    n_dn: dx_est,
                                    w_up_nb: dy_est + cy_est,
                                    w_dn_nb: dy_est,
                                }
                                for v_id in range(n):
                                    max_err = max(max_err, abs(op_mat[u_id][v_id] - exp_row.get(v_id, 0.0)))
                        if max_err <= 5e-3:
                            flat_order = [grid[r_i][c_i] for r_i in range(nx) for c_i in range(ny)]
                            rank_key = (
                                round(float(nx), 2),
                                round(dx_est, 5),
                                round(cx_est, 5),
                                -max_err,
                            )
                            valid_embeddings.append(
                                (
                                    rank_key,
                                    (nx, ny),
                                    flat_order,
                                    {
                                        "diffusion_x": dx_est,
                                        "diffusion_y": dy_est,
                                        "advection_x": cx_est,
                                        "advection_y": cy_est,
                                        "reaction_sink": lam_est,
                                        "grid_shape": [nx, ny],
                                    },
                                )
                            )

    if not valid_embeddings:
        return None
    valid_embeddings.sort(key=lambda item: item[0], reverse=True)
    _, shape_best, order_best, params_best = valid_embeddings[0]
    return order_best, shape_best, params_best


def _reconstruct_general_graph_laplacian(
    op_mat: list[list[float]],
    edge_tol: float = 1e-3,
) -> tuple[list[int], list[tuple[int, int, float]], dict] | None:
    """Verify ``L = -D * L_G - lam * I`` for an arbitrary symmetric weighted connected graph ``G``."""
    n = len(op_mat)
    if n < 4:
        return None

    # Check symmetry of off-diagonal entries: |L_ij - L_ji| <= tol and L_ij >= -tol
    raw_edges: list[tuple[int, int, float]] = []
    deg_sum = [0.0] * n
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    max_off = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            avg_w = 0.5 * (op_mat[i][j] + op_mat[j][i])
            if abs(op_mat[i][j] - op_mat[j][i]) > 5e-3:
                return None
            if avg_w < -5e-3:
                return None
            if avg_w > edge_tol:
                raw_edges.append((i, j, avg_w))
                deg_sum[i] += avg_w
                deg_sum[j] += avg_w
                adj[i].add(j)
                adj[j].add(i)
                if avg_w > max_off:
                    max_off = avg_w

    if not raw_edges or max_off <= edge_tol:
        return None

    # Check graph connectivity via BFS
    visited = {0}
    queue = [0]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    if len(visited) != n:
        return None

    # Check uniform bulk reaction sink: -L_ii - sum_{j!=i} L_ij == lam for all i
    lam_per_site = [-(op_mat[i][i] + deg_sum[i]) for i in range(n)]
    lam_est = sum(lam_per_site) / n
    if max(abs(lv - lam_est) for lv in lam_per_site) > 5e-3:
        return None

    # Canonical vertex ordering by degree + weighted degree (ties broken by original index)
    order = sorted(range(n), key=lambda i: (-len(adj[i]), -round(deg_sum[i], 5), i))
    inv_order = {orig: new_idx for new_idx, orig in enumerate(order)}
    norm_edges = sorted(
        (
            min(inv_order[u], inv_order[v]),
            max(inv_order[u], inv_order[v]),
            round(float(w / max_off), 6),
        )
        for u, v, w in raw_edges
    )
    degrees = sorted([len(adj[i]) for i in range(n)], reverse=True)
    return order, norm_edges, {
        "diffusion_scale": float(max_off),
        "reaction_sink": float(lam_est),
        "edge_count": len(norm_edges),
        "degree_sequence": degrees,
    }


def propose_manifold_probe(
    spec: TaskSpec,
    model: dict,
    records: list[dict],
    serial: int,
    *,
    remaining_rows: int | None = None,
) -> tuple[Probe | None, dict]:
    """Design compact manifold PDE probes with half-integer cut bracketing for ``N <= 16+``."""
    bounds = spec.feature_bounds
    center = {name: 0.5 * (lo + hi) for name, (lo, hi) in bounds.items()}
    audit = model.get("manifold_pde_audit") or model.get("continuum_field_audit") or {}
    focus = model.get("focus_candidates") or {}
    selector = model.get("focus_control")
    limit = spec.max_probe_length if remaining_rows is None else min(spec.max_probe_length, remaining_rows)
    if limit <= 0:
        return None, {"purpose": "manifold_budget_exhausted"}

    # 1. Ensure every selector state has a known control integer/bin center
    selected_bins: dict[str, float] = {}
    if selector and focus:
        lo, hi = bounds[selector]
        for f_name in focus:
            act_vals = [
                float(r["controls"][selector])
                for r in records
                if r.get("before", {}).get(f_name) == 1.0 and selector in r.get("controls", {})
            ]
            if act_vals:
                selected_bins[f_name] = round(sum(act_vals) / len(act_vals))
        if len(selected_bins) < len(focus):
            n_f = len(focus)
            n_scan = min(limit, n_f)
            pts = [
                {**center, selector: max(lo, min(hi, lo + (hi - lo) * idx / max(1, n_scan - 1)))}
                for idx in range(n_scan)
            ]
            return Probe(f"manifold_selector_scan:{serial}", tuple(pts)), {
                "purpose": "observe_missing_manifold_selector_states",
                "observation_rows": len(pts),
            }

    # 2. Multiplexed hold-2-step simultaneous excitation WITH tight half-integer cut bracketing!
    # Step 1 of bin k uses max(lo, k - 0.44) and Step 2 uses min(hi, k + 0.44):
    #   * Both round to k (giving an exact 2-step hold for event_delta alignment), AND
    #   * Every half-integer boundary k + 0.5 is tightly bracketed inside [k + 0.44, k + 0.56]
    #     so instrument_models._fit_membership recovers exact rounding cuts for selected_channels=()!
    rng = Random(5107 + 137 * serial + len(records))
    n = len(model.get("dynamic_fields", ()))
    seq_pts: list[dict[str, float]] = []
    if selected_bins and selector:
        lo, hi = bounds[selector]
        ordered_f = sorted(selected_bins.keys(), key=lambda fn: selected_bins[fn])
        target_len = min(limit, max(24, 3 * n))
        for _ in range(2):
            for f_name in ordered_f:
                k_bin = selected_bins[f_name]
                for offset in (-0.44, 0.44):
                    pt = {ctl: rng.uniform(*iv) for ctl, iv in bounds.items()}
                    pt[selector] = max(lo, min(hi, k_bin + offset))
                    seq_pts.append(pt)
                    if len(seq_pts) >= target_len:
                        break
                if len(seq_pts) >= target_len:
                    break
            if len(seq_pts) >= target_len:
                break
    else:
        for _ in range(min(limit, max(14, n + 6))):
            seq_pts.append({ctl: rng.uniform(*iv) for ctl, iv in bounds.items()})

    if not seq_pts:
        return None, {"purpose": "manifold_complete"}
    return Probe(f"manifold_pde_probe:{serial}", tuple(seq_pts)), {
        "purpose": "identify_manifold_operator_and_selector_alignment",
        "observation_rows": len(seq_pts),
        "audit_status": audit.get("status"),
    }


def audit_manifold_pde_premises(
    model: dict,
    records: list[dict],
    budget: SearchBudget,
    *,
    tolerance: float = 1e-8,
) -> dict:
    """Unified position-free PDE audit covering 1D Ring, 2D Torus, Neumann/Dirichlet Chains, & General Graphs."""
    # First check 1D periodic ring so Phase 4B tasks remain byte-identical!
    ring_audit = audit_continuum_field_premises(model, records, budget, tolerance=tolerance)
    if ring_audit.get("status") != "topology_not_1d_periodic_ring":
        return ring_audit

    start = budget.spent
    dynamic = list(model.get("dynamic_fields", ()))
    n = len(dynamic)
    links = extract_continuum_u_links(model)
    if links is None:
        return {
            "status": "awaiting_manifold_transitions",
            "compute_cost": ring_audit["compute_cost"],
        }
    u_by_q, q_ctl_by_q, u_ctl_by_q = links

    valid_rows = [
        (idx, r)
        for idx, r in enumerate(records)
        if set(dynamic) <= set(r.get("before", ()))
        and set(u_by_q.values()) <= set(r.get("before", ()))
        and set(dynamic) <= set(r.get("after", ()))
    ]
    if len(valid_rows) < n + 2 or not budget.spend(len(valid_rows) * n * 2):
        return {
            "status": "awaiting_manifold_transitions",
            "compute_cost": ring_audit["compute_cost"] + (budget.spent - start),
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
    gram = [[sum(qv[a] * qv[b] for qv in q_rows) for b in range(n)] for a in range(n)]
    for d_i in range(n):
        gram[d_i][d_i] += 1e-12
    if not budget.spend(n * n * len(q_rows) + n ** 3):
        return {
            "status": "manifold_compute_exhausted",
            "compute_cost": ring_audit["compute_cost"] + (budget.spent - start),
        }

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

    topology = None
    boundary_condition = "periodic"
    ordered_indices: list[int] | None = None
    pde_params: dict = {}
    grid_shape: tuple[int, int] | None = None
    graph_edges: list[tuple[int, int, float]] = []

    # 1. Test 2D Periodic Torus (4-regular Cartesian cycle product)
    torus_res = _reconstruct_2d_torus_from_operator(op_mat)
    if torus_res is not None:
        ordered_indices, grid_shape, pde_params = torus_res
        topology = "2d_periodic_torus"
        boundary_condition = "periodic"

    # 2. Test 1D Bounded Non-Periodic Chain (Neumann vs Dirichlet)
    if topology is None:
        chain_res = _reconstruct_bounded_chain_from_operator(op_mat)
        if chain_res is not None:
            ordered_indices, bc_type, pde_params = chain_res
            topology = f"1d_bounded_{bc_type}"
            boundary_condition = bc_type

    # 3. Test General Unknown Graph-Laplacian Reaction-Diffusion
    if topology is None:
        graph_res = _reconstruct_general_graph_laplacian(op_mat)
        if graph_res is not None:
            ordered_indices, graph_edges, pde_params = graph_res
            topology = "general_graph_laplacian"
            boundary_condition = "conservation_closed_graph"

    if topology is None or ordered_indices is None:
        return {
            "status": "unclassified_manifold_operator",
            "operator_mse": float(operator_mse),
            "compute_cost": ring_audit["compute_cost"] + (budget.spent - start),
        }

    ordered_qs = [dynamic[idx] for idx in ordered_indices]
    focus_cands = model.get("focus_candidates") or {}
    f_fields = sorted(focus_cands.keys())
    focus_res = _resolve_continuum_focus_mapping(
        ordered_qs, u_by_q, f_fields, focus_cands, records, budget, eff_tol
    )

    lam_val = float(pde_params.get("reaction_sink", 0.0))
    emergent_premises = [
        "P_continuum_topology",
        "P_boundary_condition",
        "P_pde_stencil_conservation",
    ]
    graph_nodes = [
        {
            "id": "P_role_partition",
            "layer": "instrument_roles",
            "depends_on": [],
            "status": "refuted",
            "reason": "spatial_manifold_coupling_with_zero_position_coordinates",
            "counterexample": witness_row,
        },
        {
            "id": "P_continuum_topology",
            "layer": "emergent_continuum_representation",
            "depends_on": ["P_role_partition"],
            "status": "supported",
            "topology": topology,
            "sites": n,
            "grid_shape": list(grid_shape) if grid_shape else None,
            "operator_mse": float(operator_mse),
        },
        {
            "id": "P_boundary_condition",
            "layer": "emergent_continuum_representation",
            "depends_on": ["P_continuum_topology"],
            "status": "supported",
            "boundary_condition": boundary_condition,
        },
        {
            "id": "P_pde_stencil_conservation",
            "layer": "continuum_field_dynamics",
            "depends_on": ["P_continuum_topology", "P_boundary_condition"],
            "status": (
                "refuted_dirichlet_wall_and_reaction_sink"
                if boundary_condition == "dirichlet" and abs(lam_val) > 1e-5
                else "refuted_dirichlet_wall_sink"
                if boundary_condition == "dirichlet"
                else "refuted_reaction_sink"
                if abs(lam_val) > 1e-5
                else "supported_conserved"
            ),
            "pde_parameters": pde_params,
        },
        {
            "id": "P_focus_membership",
            "layer": "selector_contrasts",
            "depends_on": ["P_continuum_topology"],
            "status": "supported" if (focus_res and focus_res.get("status") == "supported") else "pending",
        },
    ]
    dep_graph = {
        "kind": "support_premise_dependency_graph",
        "emergent_representation_growth": True,
        "nodes": graph_nodes,
    }

    manifold_schema_dict = None
    if focus_res and focus_res.get("status") == "supported":
        f_by_q = focus_res["f_by_q"]
        if topology == "2d_periodic_torus":
            assert grid_shape is not None
            nx, ny = grid_shape
            site_objs = [
                f"torus2d_{nx}x{ny}_{k // ny}_{k % ny}:{ordered_qs[k]}" for k in range(n)
            ]
        elif topology == "1d_bounded_neumann":
            site_objs = [f"chain_neumann_{k}:{ordered_qs[k]}" for k in range(n)]
        elif topology == "1d_bounded_dirichlet":
            site_objs = [f"chain_dirichlet_{k}:{ordered_qs[k]}" for k in range(n)]
        else:
            edge_code = "_".join(f"{u}-{v}" for u, v, _ in graph_edges)
            site_objs = [f"graph_{edge_code}_{k}:{ordered_qs[k]}" for k in range(n)]

        manifold_schema_dict = {
            "objects": site_objs,
            "fields": {
                site_objs[k]: {
                    "q": ordered_qs[k],
                    "u": u_by_q[ordered_qs[k]],
                    "f": f_by_q[ordered_qs[k]],
                }
                for k in range(n)
            },
            "members": {site_objs[k]: [site_objs[k]] for k in range(n)},
            "controls": {
                site_objs[k]: {
                    "q": q_ctl_by_q[ordered_qs[k]],
                    "u": u_ctl_by_q[ordered_qs[k]],
                }
                for k in range(n)
            },
            "focus_control": model.get("focus_control", "focus"),
        }

    return {
        "status": (
            "verified_manifold_pde_schema"
            if manifold_schema_dict is not None
            else "verified_manifold_pde_awaiting_selector"
        ),
        "topology": topology,
        "boundary_condition": boundary_condition,
        "sites": n,
        "grid_shape": list(grid_shape) if grid_shape else None,
        "graph_edges": [list(e) for e in graph_edges],
        "ordered_manifold_quantities": ordered_qs,
        "pde_parameters": pde_params,
        "operator_mse": float(operator_mse),
        "focus_resolution": focus_res,
        "manifold_schema": manifold_schema_dict,
        "dependency_graph": dep_graph,
        "refuted_premises": [
            {
                "premise_id": "P_role_partition",
                "reason": f"zero_position_sensors_with_active_{topology}_pde",
                "counterexample": witness_row,
            }
        ],
        "emergent_premises": emergent_premises,
        "compute_cost": ring_audit["compute_cost"] + (budget.spent - start),
    }


def apply_verified_manifold_premises(model: dict, audit: dict) -> dict:
    """Attach manifold or 1D ring premise audit and finalize ``ObjectSchema``."""
    if audit.get("topology") == "1d_periodic_ring" or "ring_schema" in audit:
        from .continuum_field_lab import apply_verified_continuum_premises

        return apply_verified_continuum_premises(model, audit)

    updated = deepcopy(model)
    updated["manifold_pde_audit"] = deepcopy(audit)
    updated["continuum_field_audit"] = deepcopy(audit)
    updated["premise_dependency_graph"] = deepcopy(audit.get("dependency_graph"))
    if audit.get("status") == "verified_manifold_pde_schema" and audit.get("manifold_schema"):
        m_sc = audit["manifold_schema"]
        f_by_q = audit["focus_resolution"]["f_by_q"]
        ordered_groups = []
        for k, q_name in enumerate(audit["ordered_manifold_quantities"]):
            obj_id = m_sc["objects"][k]
            ordered_groups.append(
                {
                    "object_id": obj_id,
                    "q": q_name,
                    "u": m_sc["fields"][obj_id]["u"],
                    "f": f_by_q[q_name],
                    "controls": deepcopy(m_sc["controls"][obj_id]),
                }
            )
        for f_name, item in updated.get("focus_candidates", {}).items():
            matched_q = [qn for qn, fn in f_by_q.items() if fn == f_name]
            if len(matched_q) == 1:
                item["quantities"] = matched_q
        updated["groups"] = ordered_groups
        updated["schema"] = deepcopy(m_sc)
        updated["continuum_topology"] = audit["topology"]
        updated["boundary_condition"] = audit.get("boundary_condition")
        updated["status"] = "identified"
        updated["complete"] = True
        updated["unresolved"] = []
    return updated


# ---------------------------------------------------------------------------
# Manifold Schema Metadata, Basis Synthesis, & Gauge-Invariant Commitments
# ---------------------------------------------------------------------------


def parse_manifold_schema_metadata(schema: ObjectSchema | dict | None, spec: TaskSpec | None = None) -> dict | None:
    """Extract manifold topology, dimensions, boundary type, or graph edges from ``ObjectSchema``."""
    if schema is None:
        return None
    sc = schema if isinstance(schema, ObjectSchema) else ObjectSchema.from_dict(schema)
    if len(sc.objects) < 3 or any(set(sc.fields.get(g, ())) != {"q", "u", "f"} for g in sc.objects):
        return None
    first = sc.objects[0].split(":", 1)[0]
    if first.startswith("torus2d_"):
        parts = first.split("_")
        nx_s, ny_s = parts[1].split("x", 1)
        nx, ny = int(nx_s), int(ny_s)
        if nx * ny != len(sc.objects):
            return None
        return {
            "topology": "2d_periodic_torus",
            "boundary_condition": "periodic",
            "grid_shape": (nx, ny),
            "sites": len(sc.objects),
        }
    if first.startswith("chain_neumann_"):
        return {
            "topology": "1d_bounded_neumann",
            "boundary_condition": "neumann",
            "sites": len(sc.objects),
        }
    if first.startswith("chain_dirichlet_"):
        return {
            "topology": "1d_bounded_dirichlet",
            "boundary_condition": "dirichlet",
            "sites": len(sc.objects),
        }
    if first.startswith("graph_"):
        edges: list[tuple[int, int, float]] = []
        if spec is not None and spec.context.get("object_model"):
            om = spec.context["object_model"]
            om = json.loads(om) if isinstance(om, str) else om
            raw_e = ((om or {}).get("manifold_pde_audit") or {}).get("graph_edges") or ()
            edges = [(int(e[0]), int(e[1]), float(e[2])) for e in raw_e]
        if not edges:
            # Parse fallback edge code embedded in object_id: graph_{u0-v0_u1-v1}_{k}:{q}
            body = first.removeprefix("graph_")
            edge_part = body.rsplit("_", 1)[0]
            for tok in edge_part.split("_"):
                if "-" in tok:
                    u_s, v_s = tok.split("-", 1)
                    edges.append((int(u_s), int(v_s), 1.0))
        if not edges:
            return None
        return {
            "topology": "general_graph_laplacian",
            "boundary_condition": "conservation_closed_graph",
            "graph_edges": edges,
            "sites": len(sc.objects),
        }
    return None


def _build_manifold_basis_terms(
    schema: ObjectSchema,
    meta: dict,
) -> dict[str, tuple[Program, ...]]:
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
    topo = meta["topology"]
    if topo == "2d_periodic_torus":
        nx, ny = meta["grid_shape"]
        dim_tag = f"{nx}x{ny}"
        lx = Program("manifold_sum", name=f"torus2d_lap_x:{dim_tag}", children=fq_pairs)
        ly = Program("manifold_sum", name=f"torus2d_lap_y:{dim_tag}", children=fq_pairs)
        gx = Program("manifold_sum", name=f"torus2d_grad_bx:{dim_tag}", children=fq_pairs)
        gy = Program("manifold_sum", name=f"torus2d_grad_by:{dim_tag}", children=fq_pairs)
        stencils = (lx, ly, gx, gy)
        extra_global: tuple[Program, ...] = ()
    elif topo == "1d_bounded_neumann":
        lap = Program("manifold_sum", name="chain_lap_neumann", children=fq_pairs)
        adv = Program("manifold_sum", name="chain_adv_zeroflux", children=fq_pairs)
        stencils = (lap, adv)
        extra_global = ()
    elif topo == "1d_bounded_dirichlet":
        lap = Program("manifold_sum", name="chain_lap_dirichlet", children=fq_pairs)
        adv = Program("manifold_sum", name="chain_adv_zeroflux", children=fq_pairs)
        bnd_tot = Program("manifold_sum", name="chain_boundary_sum", children=fq_pairs)
        stencils = (lap, adv)
        extra_global = (bnd_tot,)
    else:
        edge_children: list[Program] = []
        for u_i, v_i, w_uv in meta["graph_edges"]:
            gu, gv = objs[u_i], objs[v_i]
            edge_children.extend(
                (
                    Program("const", value=float(w_uv)),
                    Program("feature", name=schema.fields[gu]["f"]),
                    Program("feature", name=schema.fields[gu]["q"]),
                    Program("feature", name=schema.fields[gv]["f"]),
                    Program("feature", name=schema.fields[gv]["q"]),
                )
            )
        glap = Program("manifold_sum", name="graph_laplacian", children=tuple(edge_children))
        stencils = (glap,)
        extra_global = ()

    return {
        "manifold_focal_current": (foc_q, foc_u, *stencils),
        "manifold_joint_pde_event": (
            tot_q,
            tot_u,
            *extra_global,
            foc_q,
            foc_u,
            *stencils,
            *(Program("event_delta", lag=1, children=(st,)) for st in stencils),
            Program("event_delta", lag=1, children=(foc_q,)),
        ),
        "manifold_joint_pde_window": (
            tot_q,
            tot_u,
            *extra_global,
            *(Program("window_sum", lag=3, children=(st,)) for st in stencils),
            Program("window_sum", lag=3, children=(foc_q,)),
        ),
    }


def _pack_manifold_entry(
    *,
    candidate_id: str,
    family: str,
    family_id: str,
    search_tag: str,
    topology: str,
    boundary_condition: str,
    terms: tuple[Program, ...],
    fit: dict,
    schema: ObjectSchema,
    rows: list[Observation],
    compute_cost: int,
    sources: tuple[str, ...] = (),
    commitment_meta: dict | None = None,
    revision_meta: dict | None = None,
) -> dict:
    candidate = Candidate(
        candidate_id,
        fit["program"],
        search_tag,
        f"Fit {topology} ({boundary_condition}) manifold PDE stencil basis",
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
        "manifold_family": family,
        "manifold_topology": topology,
        "boundary_condition": boundary_condition,
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
            "kind": "manifold_topology",
            "status": "supported",
            "topology": topology,
            "boundary_condition": boundary_condition,
            "counterexamples": [],
        }
    ]
    discovery = StructuralDiscovery(
        [candidate], [fit_record], hypotheses, schema, compute_cost, None, True
    )
    entry = {
        "candidate": candidate,
        "kind": f"manifold:{family}",
        "family_id": family_id,
        "training_loss": float(fit["loss"]),
        "structure": discovery.to_dict(),
        "structure_search": search_tag,
        "method_source_ids": list(sources),
    }
    if commitment_meta is not None:
        entry["manifold_commitment_application"] = deepcopy(commitment_meta)
    if revision_meta is not None:
        entry["scientific_revision"] = deepcopy(revision_meta)
    entry = attach_structural_research_program(entry, fit_rows=len(rows))
    if entry.get("research_program") and sources:
        entry["research_program"]["source_ids"] = list(sources)
    return entry


def propose_manifold_structures(
    spec: TaskSpec,
    training: list[Observation],
    schema: ObjectSchema,
    budget: SearchBudget,
    *,
    tolerance: float = 1e-8,
    transitions=(),
) -> list[dict]:
    """Fit manifold PDE stencil families on a position-free 2D torus, bounded chain, or graph ``ObjectSchema``."""
    meta = parse_manifold_schema_metadata(schema, spec)
    if meta is None:
        from .continuum_field_lab import propose_continuum_structures

        return propose_continuum_structures(
            spec, training, schema, budget, tolerance=tolerance, transitions=transitions
        )
    if not training:
        return []
    start = budget.spent
    fitter = _Fitter(list(training), budget, spec.risk_quantile)
    families = _build_manifold_basis_terms(schema, meta)
    family_ids = {
        "manifold_focal_current": "T08",
        "manifold_joint_pde_event": "T11",
        "manifold_joint_pde_window": "T12",
    }
    entries: list[dict] = []
    for fam_name, terms in families.items():
        fit = fitter.fit(terms)
        if fit is None:
            break
        entries.append(
            _pack_manifold_entry(
                candidate_id=f"manifold:{meta['topology']}:{fam_name}:{len(schema.objects)}:{budget.spent}",
                family=fam_name,
                family_id=family_ids[fam_name],
                search_tag="manifold_stencil_discovery",
                topology=meta["topology"],
                boundary_condition=meta["boundary_condition"],
                terms=terms,
                fit=fit,
                schema=schema,
                rows=list(training),
                compute_cost=budget.spent - start,
            )
        )
        if fit["loss"] <= tolerance and fam_name == "manifold_focal_current":
            break
    entries.sort(key=lambda item: (item["training_loss"] > tolerance, item["candidate"].program.size))
    return entries


def extract_manifold_role_pattern(prior: dict) -> dict | None:
    """Lift a confirmed manifold PDE prior into a resolution/graph-parametric pattern."""
    if prior.get("status") not in {"supported", "conditional"}:
        return None
    raw_sc = prior.get("object_schema") or (prior.get("structure") or {}).get("schema")
    if raw_sc is None:
        return None
    source = raw_sc if isinstance(raw_sc, ObjectSchema) else ObjectSchema.from_dict(raw_sc)
    meta = parse_manifold_schema_metadata(source)
    if meta is None:
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
    expected_pairs = tuple(
        part
        for g in source.objects
        for part in (
            Program("feature", name=source.fields[g]["f"]),
            Program("feature", name=source.fields[g]["q"]),
        )
    )
    has_manifold_op = False

    def _lift_term(node: Program) -> dict | None:
        nonlocal has_manifold_op
        if node.op == "manifold_sum":
            m_name = str(node.name or "")
            if m_name == "graph_laplacian":
                has_manifold_op = True
                return {
                    "kind": "all_manifold_sum",
                    "stencil": "graph_laplacian",
                    "roles": ["f", "q"],
                }
            if node.children != expected_pairs:
                return None
            has_manifold_op = True
            base_stencil = m_name.split(":", 1)[0]
            return {
                "kind": "all_manifold_sum",
                "stencil": base_stencil,
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
            if len(groups_seen) != len(source.objects) or set(groups_seen) != set(source.objects):
                return None
            if any(p != pats[0] for p in pats[1:]):
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
    if not has_manifold_op:
        return None

    return {
        "source_concept_id": prior["source_concept_id"],
        "source_task_id": prior.get("task_id"),
        "source_schema": source,
        "source_meta": meta,
        "source_resolution": len(source.objects),
        "fit_contract": fit,
        "role_terms": role_terms,
        "family": (
            ((prior.get("structure") or {}).get("fits") or [{}])[0].get("manifold_family")
            or "manifold_joint_pde_event"
        ),
        "family_id": prior.get("family_id", "T11"),
    }


def instantiate_manifold_terms(
    role_terms: list[dict] | tuple[dict, ...],
    target_schema: ObjectSchema | dict,
    *,
    spec: TaskSpec | None = None,
) -> tuple[Program, ...] | None:
    target = target_schema if isinstance(target_schema, ObjectSchema) else ObjectSchema.from_dict(target_schema)
    meta = parse_manifold_schema_metadata(target, spec)
    if meta is None:
        return None
    objs = target.objects
    fq_pairs = tuple(
        part
        for g in objs
        for part in (
            Program("feature", name=target.fields[g]["f"]),
            Program("feature", name=target.fields[g]["q"]),
        )
    )

    def _build(node: dict) -> Program | None:
        kind = node["kind"]
        if kind == "all_manifold_sum":
            st = node["stencil"]
            if st == "graph_laplacian":
                if meta["topology"] != "general_graph_laplacian":
                    return None
                edge_children: list[Program] = []
                for u_i, v_i, w_uv in meta["graph_edges"]:
                    gu, gv = objs[u_i], objs[v_i]
                    edge_children.extend(
                        (
                            Program("const", value=float(w_uv)),
                            Program("feature", name=target.fields[gu]["f"]),
                            Program("feature", name=target.fields[gu]["q"]),
                            Program("feature", name=target.fields[gv]["f"]),
                            Program("feature", name=target.fields[gv]["q"]),
                        )
                    )
                return Program("manifold_sum", name="graph_laplacian", children=tuple(edge_children))
            if st.startswith("torus2d_"):
                if meta["topology"] != "2d_periodic_torus":
                    return None
                nx, ny = meta["grid_shape"]
                return Program("manifold_sum", name=f"{st}:{nx}x{ny}", children=fq_pairs)
            if st in {
                "chain_lap_neumann",
                "chain_lap_dirichlet",
                "chain_adv_zeroflux",
                "chain_boundary_sum",
                "chain_focal_boundary",
            }:
                if not meta["topology"].startswith("1d_bounded_"):
                    return None
                return Program("manifold_sum", name=st, children=fq_pairs)
            return None
        if kind == "all_object_sum":
            pat = Program.from_dict(node["summand"])
            roles = node["roles"]
            return Program(
                "object_sum",
                children=tuple(
                    pat.remap_features({f"object_role:{r}": target.fields[g][r] for r in roles})
                    for g in objs
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

    built = tuple(_build(t) for t in role_terms)
    if not built or any(t is None for t in built):
        return None
    return built


def check_manifold_equivalence(
    sources: list[dict],
    commitment: dict,
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-9,
) -> dict:
    """Verify column preservation, support prediction agreement, and ``Aut(M)`` gauge invariance."""
    start = budget.spent if budget is not None else 0
    for src in sources:
        sc: ObjectSchema = src["source_schema"]
        meta = src["source_meta"]
        fit = src["fit_contract"]
        inst = instantiate_manifold_terms(commitment["role_terms"], sc)
        if inst is None or len(inst) != len(fit["fit_terms"]):
            return {"accepted": False, "reason": "column_mismatch"}

        rng = Random(93179)
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

        # Check Aut(M) gauge invariance where nontrivial shift group acts (2D Torus Z_Nx x Z_Ny!)
        if meta["topology"] == "2d_periodic_torus":
            nx, ny = meta["grid_shape"]
            shifted_objs = tuple(
                objs[((r + 1) % nx) * ny + ((c + 1) % ny)]
                for r in range(nx)
                for c in range(ny)
            )
            shifted_sc = ObjectSchema(
                shifted_objs, sc.fields, sc.members, sc.controls, sc.focus_control
            )
            shifted_inst = instantiate_manifold_terms(commitment["role_terms"], shifted_sc)
            assert shifted_inst is not None
            shifted_prog = _sum_terms(shifted_inst, fit["coefficients"][1:], fit["coefficients"][0])
            gauge_diff = max(abs(shifted_prog.predict(r) - rebuilt.predict(r)) for r in rows)
            if gauge_diff > tolerance:
                return {"accepted": False, "reason": "torus_translation_gauge_invariance_violated"}

    return {
        "accepted": True,
        "reason": "strict_manifold_equivalence_and_automorphism_invariance_preserved",
        "compute_cost": (budget.spent - start) if budget is not None else 0,
    }


def consolidate_manifold_pde_commitments(
    priors: list[dict],
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-9,
) -> list[dict]:
    extracted = [p for prior in priors if (p := extract_manifold_role_pattern(prior)) is not None]
    if not extracted:
        return []
    groups: dict[str, list[dict]] = {}
    for item in extracted:
        sig = json.dumps(
            {
                "topology": item["source_meta"]["topology"],
                "boundary_condition": item["source_meta"]["boundary_condition"],
                "role_terms": item["role_terms"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        groups.setdefault(sig, []).append(item)

    commitments: list[dict] = []
    for idx, members in enumerate(reversed(list(groups.values()))):
        rep = members[-1]
        cand = {
            "commitment_id": f"manifold_commitment:{idx:04d}",
            "kind": "manifold_pde_commitment",
            "topology": rep["source_meta"]["topology"],
            "boundary_condition": rep["source_meta"]["boundary_condition"],
            "reorganization_type": "equivalence_reorganization",
            "source_concept_ids": list(dict.fromkeys(m["source_concept_id"] for m in members)),
            "source_resolutions": sorted({m["source_resolution"] for m in members}),
            "family": rep["family"],
            "family_id": rep["family_id"],
            "role_terms": deepcopy(rep["role_terms"]),
        }
        eq = check_manifold_equivalence(members, cand, budget=budget, tolerance=tolerance)
        if eq["accepted"]:
            cand["equivalence_audit"] = eq
            commitments.append(cand)
    return commitments


def _revise_bounded_chain_role_terms(
    role_terms: list[dict],
    source_bc: str,
    target_bc: str,
) -> list[dict] | None:
    """Conservatively revise ONLY ``P_boundary_condition`` between Neumann and Dirichlet chains."""
    if {source_bc, target_bc} != {"neumann", "dirichlet"}:
        return None
    src_lap = f"chain_lap_{source_bc}"
    dst_lap = f"chain_lap_{target_bc}"

    def _swap(node: dict) -> dict:
        if node.get("kind") == "all_manifold_sum" and node.get("stencil") == src_lap:
            return {**node, "stencil": dst_lap}
        if node.get("kind") == "program_op":
            return {**node, "children": [_swap(c) for c in node.get("children", ())]}
        return deepcopy(node)

    revised = [_swap(t) for t in role_terms]
    if target_bc == "dirichlet":
        bnd_term = {
            "kind": "all_manifold_sum",
            "stencil": "chain_boundary_sum",
            "roles": ["f", "q"],
        }
        if bnd_term not in revised:
            revised.insert(min(2, len(revised)), bnd_term)
    else:
        revised = [
            t
            for t in revised
            if not (t.get("kind") == "all_manifold_sum" and t.get("stencil") == "chain_boundary_sum")
        ]
    return revised


def instantiate_manifold_commitment(
    spec: TaskSpec,
    training: list[Observation],
    schema: ObjectSchema | dict,
    commitment: dict,
    budget: SearchBudget,
    *,
    tolerance: float = 1e-8,
) -> list[dict]:
    """Instantiate a ``ManifoldPDECommitment`` (or conservative boundary revision) onto ``schema``."""
    target = schema if isinstance(schema, ObjectSchema) else ObjectSchema.from_dict(schema)
    meta = parse_manifold_schema_metadata(target, spec)
    if not training or meta is None:
        return []
    if not set(target.feature_names).issubset(spec.feature_names):
        return []

    start = budget.spent
    src_topo = commitment.get("topology")
    src_bc = commitment.get("boundary_condition")
    dst_topo = meta["topology"]
    dst_bc = meta["boundary_condition"]
    revision_meta = None
    role_terms = commitment.get("role_terms", ())
    search_tag = "manifold_commitment_instantiation"

    if src_topo != dst_topo:
        if (
            src_topo in {"1d_bounded_neumann", "1d_bounded_dirichlet"}
            and dst_topo in {"1d_bounded_neumann", "1d_bounded_dirichlet"}
            and src_bc != dst_bc
        ):
            revised = _revise_bounded_chain_role_terms(list(role_terms), str(src_bc), str(dst_bc))
            if revised is None:
                return []
            role_terms = revised
            search_tag = "manifold_boundary_premise_revision"
            revision_meta = {
                "kind": "boundary_condition_revision",
                "revised_premise": "P_boundary_condition",
                "retracted_boundary_condition": src_bc,
                "replacement_boundary_condition": dst_bc,
                "source_commitment_id": commitment["commitment_id"],
            }
        else:
            return []

    terms = instantiate_manifold_terms(role_terms, target, spec=spec)
    if terms is None:
        return []
    fitter = _Fitter(list(training), budget, spec.risk_quantile)
    fit = fitter.fit(terms)
    if fit is None:
        return []
    sources = tuple(commitment.get("source_concept_ids", ()))
    return [
        _pack_manifold_entry(
            candidate_id=f"{commitment['commitment_id']}:{search_tag}:{len(target.objects)}",
            family=commitment.get("family", "manifold_joint_pde_event"),
            family_id=commitment.get("family_id", "T11"),
            search_tag=search_tag,
            topology=dst_topo,
            boundary_condition=dst_bc,
            terms=terms,
            fit=fit,
            schema=target,
            rows=list(training),
            compute_cost=budget.spent - start,
            sources=sources,
            commitment_meta={
                "commitment_id": commitment["commitment_id"],
                "source_topology": src_topo,
                "target_topology": dst_topo,
                "source_boundary_condition": src_bc,
                "target_boundary_condition": dst_bc,
                "source_resolutions": list(commitment.get("source_resolutions", ())),
                "target_resolution": len(target.objects),
                "reorganization_type": (
                    "scientific_premise_revision"
                    if revision_meta is not None
                    else "equivalence_reorganization"
                ),
            },
            revision_meta=revision_meta,
        )
    ]
