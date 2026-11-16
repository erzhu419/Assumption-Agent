"""Support-premise dependency graph and transition premise auditing (v10).

Separates scientific support premises from ``ResearchGraph`` compute caches:

1. ``PremiseDependencyGraph`` models the causal/logical dependencies among
   empirically falsifiable premises:
     - ``P_role_partition`` -> ``(Q, U, X, F)`` sensor role separation
     - ``P_conservation`` -> ``sum_i (q'_i - q_i - u_i) = 0``
     - ``P_exchange_form`` -> unconditional vs state-conditional pairwise transport
     - ``P_interaction_direction`` -> ``symmetric`` / ``right`` / ``left``
     - ``P_position_ownership`` -> bijection ``pi_X: Q -> X``
     - ``P_locality_radius`` -> consistent interval ``(r_low, r_high)``, ``gamma > 0``
     - ``P_focus_membership`` -> bijection ``pi_F: Q -> F``
     - ``P_temporal_readout`` -> linear readout over role-uniform temporal basis
2. ``audit_transition_premises`` tests ``P_conservation``, ``P_exchange_form``,
   ``P_interaction_direction``, ``P_position_ownership``, and ``P_locality_radius``
   directly on already-purchased ``before -> after`` sensor transitions
   (completely decoupled from unresolved ``focus`` ``f``), charging every step
   to ``SearchBudget``:
     - When ``P_conservation`` holds and unconditional pairwise exchange is
       refuted, audits state-dependent edge predicates from ``relation_conditions``
       (``source_greater``, ``source_less``) and emits ``exchange_condition_revision``.
     - When ``P_conservation`` is refuted with ``sum_i delta_i = kappa * sum_i q_i``,
       de-couples ``sink`` (local proportional dissipation ``kappa * q_i``) vs
       ``nonlocal`` (uniform global coupling ``(kappa / N) * sum_j q_j``) to verify
       downstream geometry/roles and emit ``nonconservation_mechanism_revision``.
3. ``apply_verified_transition_premises`` prunes falsified ``x -> q`` and ``f -> q``
   owners whenever exhaustive transition/trajectory premises achieve machine-level
   consistency across ALL paid transitions.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite, sqrt

from .language import EvaluationContext, Program
from .models import Observation, SearchBudget
from .object_induction import _finish_object_schema, _propagate
from .relation_conditions import TRUE, atomic_conditions


VERIFIED_EXCHANGE_STATUSES = frozenset(
    {
        "verified_unconditional_exchange",
        "verified_conditional_exchange",
        "verified_nonconservative_exchange",
        "verified_latent_transition_exchange",
    }
)


def edge_condition_predicate(condition: str | dict | Program | None) -> Program:
    """Return the ``relation_conditions`` predicate AST for ``condition``."""
    if condition is None or condition == "unconditional":
        return TRUE
    if isinstance(condition, Program):
        return condition
    if isinstance(condition, dict):
        return Program.from_dict(condition)
    atoms = atomic_conditions("quantity")
    src_arg = Program("arg", name="edge_source_quantity")
    dst_arg = Program("arg", name="edge_target_quantity")
    if condition == "source_greater":
        target_lt_source = Program("lt", children=(dst_arg, src_arg))
        return next((p for p in atoms if p == target_lt_source), target_lt_source)
    if condition == "source_less":
        source_lt_target = Program("lt", children=(src_arg, dst_arg))
        return next((p for p in atoms if p == source_lt_target), source_lt_target)
    raise ValueError(f"unknown symbolic edge condition: {condition}")


def describe_edge_condition(predicate: str | dict | Program | None) -> str | dict:
    """Map an edge predicate AST or tag to a canonical condition identifier."""
    if predicate is None or predicate == "unconditional":
        return "unconditional"
    if isinstance(predicate, str):
        return predicate
    prog = predicate if isinstance(predicate, Program) else Program.from_dict(predicate)
    if prog == TRUE:
        return "unconditional"
    if prog == edge_condition_predicate("source_greater"):
        return "source_greater"
    if prog == edge_condition_predicate("source_less"):
        return "source_less"
    return prog.to_dict()


def _edge_condition_holds(
    edge_condition: str | Program | dict,
    x_src: float,
    q_src: float,
    x_dst: float,
    q_dst: float,
) -> bool:
    if edge_condition == "unconditional":
        return True
    if edge_condition == "source_greater":
        return q_src > q_dst
    if edge_condition == "source_less":
        return q_src < q_dst
    prog = edge_condition if isinstance(edge_condition, Program) else Program.from_dict(edge_condition)
    bound = prog._expand(
        {
            "edge_source_position": Program("const", value=float(x_src)),
            "edge_source_quantity": Program("const", value=float(q_src)),
            "edge_target_position": Program("const", value=float(x_dst)),
            "edge_target_quantity": Program("const", value=float(q_dst)),
        }
    )
    return bool(EvaluationContext(Observation({}, 0.0)).evaluate(bound))


def build_premise_dependency_graph(commitment: dict | None = None, model: dict | None = None) -> dict:
    """Construct the explicit DAG of support premises and observable tests."""
    committed_dir = None
    committed_cond: str | dict = "unconditional"
    committed_symbol = None
    if commitment and commitment.get("radius_parameters"):
        primary = commitment["radius_parameters"][0]
        committed_symbol = primary["symbol"]
        committed_dir = primary["direction"]
        committed_cond = describe_edge_condition(primary.get("edge_condition"))
    committed_noncons = (commitment or {}).get("nonconservation_mechanism")
    committed_latent = (commitment or {}).get("latent_state_mechanism")

    nodes = [
        {
            "id": "P_role_partition",
            "layer": "instrument_roles",
            "depends_on": [],
            "observable_test": "single_control_and_additive_sweeps",
            "status": "supported"
            if (
                model
                and model.get("dynamic_fields")
                and len(model.get("position_candidates", {})) == len(model.get("dynamic_fields", ()))
                and all(g.get("q") and g.get("u") for g in model.get("groups", ()))
            )
            else "pending",
        },
        {
            "id": "P_markov_state",
            "layer": "state_sufficiency",
            "depends_on": ["P_role_partition"],
            "committed_value": committed_latent or "markov_sufficient",
            "observable_test": "reset_vs_history_recurrence_and_readout_residuals",
            "status": "pending",
        },
        {
            "id": "P_latent_order",
            "layer": "state_sufficiency",
            "depends_on": ["P_markov_state"],
            "committed_value": 2 if committed_latent and "oscillatory" in str(committed_latent) else 1,
            "observable_test": "first_order_exponential_vs_second_order_ar2_recurrence",
            "status": "pending",
        },
        {
            "id": "P_latent_coupling",
            "layer": "state_sufficiency",
            "depends_on": ["P_markov_state"],
            "committed_value": "coupled" if committed_latent and "coupled" in str(committed_latent) else "independent",
            "observable_test": "independent_per_object_vs_spatial_neighbor_latent_transport",
            "status": "pending",
        },
        {
            "id": "P_conservation",
            "layer": "state_transitions",
            "depends_on": ["P_role_partition"],
            "committed_value": committed_noncons or "conserved",
            "observable_test": "sum_i(after[q_i] - before[q_i] - before[u_i]) == 0",
            "status": "pending",
        },
        {
            "id": "P_exchange_form",
            "layer": "state_transitions",
            "depends_on": ["P_role_partition", "P_conservation"],
            "committed_value": committed_cond,
            "observable_test": "distance_gated_pairwise_transport_and_edge_condition",
            "status": "pending",
        },
        {
            "id": "P_interaction_direction",
            "layer": "state_transitions",
            "depends_on": ["P_exchange_form"],
            "committed_value": committed_dir,
            "symbol": committed_symbol,
            "observable_test": "directional_asymmetry_on_paid_transitions",
            "status": "pending",
        },
        {
            "id": "P_position_ownership",
            "layer": "state_transitions",
            "depends_on": ["P_interaction_direction"],
            "observable_test": "bijection_pi_X_compatibility_across_paid_transitions",
            "status": "pending",
        },
        {
            "id": "P_locality_radius",
            "layer": "state_transitions_and_response",
            "depends_on": ["P_interaction_direction", "P_position_ownership"],
            "observable_test": "nonempty_consistent_distance_interval_with_positive_coupling",
            "status": "pending",
        },
        {
            "id": "P_focus_membership",
            "layer": "selector_contrasts",
            "depends_on": ["P_role_partition"],
            "observable_test": "matched_prefix_selector_excitation_contrasts",
            "status": "supported"
            if (model and model.get("groups") and all(g.get("f") for g in model["groups"]))
            else "pending",
        },
        {
            "id": "P_temporal_readout",
            "layer": "trajectory_response",
            "depends_on": [
                "P_position_ownership",
                "P_locality_radius",
                "P_focus_membership",
            ],
            "observable_test": "linear_readout_loss_on_paid_trajectories",
            "status": "pending",
        },
    ]
    return {
        "kind": "support_premise_dependency_graph",
        "commitment_id": commitment.get("commitment_id") if commitment else None,
        "nodes": nodes,
    }


def _enumerate_position_bijections(
    q_list: list[str],
    x_fields: list[str],
    positions: dict,
    *,
    max_bijections: int = 720,
) -> tuple[list[tuple[int, ...]], bool]:
    """Enumerate bijections ``q_to_x`` (mapping q_idx -> x_idx) allowed by ``positions``."""
    count = len(q_list)
    q_index = {name: idx for idx, name in enumerate(q_list)}
    allowed_q_for_x = [
        [q_index[name] for name in positions[x_field].get("quantities", ()) if name in q_index]
        for x_field in x_fields
    ]
    if any(not opts for opts in allowed_q_for_x):
        return [], True

    x_order = sorted(range(count), key=lambda x_idx: (len(allowed_q_for_x[x_idx]), x_idx))
    results: list[tuple[int, ...]] = []
    assigned_x_for_q = [-1] * count
    used_q = [False] * count
    exhaustive = True

    def backtrack(pos: int):
        nonlocal exhaustive
        if len(results) > max_bijections:
            exhaustive = False
            return
        if pos == count:
            results.append(tuple(assigned_x_for_q))
            return
        x_idx = x_order[pos]
        for q_idx in allowed_q_for_x[x_idx]:
            if not used_q[q_idx]:
                used_q[q_idx] = True
                assigned_x_for_q[q_idx] = x_idx
                backtrack(pos + 1)
                assigned_x_for_q[q_idx] = -1
                used_q[q_idx] = False
                if not exhaustive:
                    return

    backtrack(0)
    if not exhaustive:
        return results[:max_bijections], False
    return results, True


def _exact_interval_across_transitions(
    q_rows: list[list[float]],
    x_rows: list[list[float]],
    delta_rows: list[list[float]],
    direction: str,
    q_to_x: tuple[int, ...],
    eff_tol: float,
    *,
    edge_condition: str | Program | dict = "unconditional",
) -> dict | None:
    """Find the exact consistent radius interval across ALL supplied transitions."""
    t_count = len(q_rows)
    n = len(q_to_x)
    total_entries = t_count * n
    g_mat = [[0.0] * n for _ in range(t_count)]
    events: list[tuple[float, int, int, int, float, float]] = []

    for tau in range(t_count):
        q_row = q_rows[tau]
        x_row = x_rows[tau]
        if direction == "symmetric":
            if edge_condition != "unconditional":
                return None
            for i in range(n):
                xi = x_row[q_to_x[i]]
                for j in range(i + 1, n):
                    xj = x_row[q_to_x[j]]
                    dist = abs(xi - xj)
                    dq = q_row[j] - q_row[i]
                    if dq == 0.0:
                        continue
                    if dist == 0.0:
                        g_mat[tau][i] += dq
                        g_mat[tau][j] -= dq
                    else:
                        events.append((dist, tau, i, j, dq, -dq))
        else:
            sign = 1.0 if direction == "right" else -1.0
            for src in range(n):
                q_src = q_row[src]
                if q_src == 0.0:
                    continue
                x_src = x_row[q_to_x[src]]
                for dst in range(n):
                    if src == dst:
                        continue
                    x_dst = x_row[q_to_x[dst]]
                    dist = sign * (x_dst - x_src)
                    if dist > 0.0 and _edge_condition_holds(
                        edge_condition, x_src, q_src, x_dst, q_row[dst]
                    ):
                        events.append((dist, tau, src, dst, -q_src, q_src))

    events.sort(key=lambda item: item[0])

    grouped_thresholds: list[float] = []
    grouped_batches: list[list[tuple[int, int, int, float, float]]] = []
    for dist, tau, i, j, di, dj in events:
        if grouped_thresholds and abs(dist - grouped_thresholds[-1]) <= 1e-14:
            grouped_batches[-1].append((tau, i, j, di, dj))
        else:
            grouped_thresholds.append(dist)
            grouped_batches.append([(tau, i, j, di, dj)])

    s_gg = sum(g_mat[tau][i] ** 2 for tau in range(t_count) for i in range(n))
    s_gd = sum(g_mat[tau][i] * delta_rows[tau][i] for tau in range(t_count) for i in range(n))

    supported_cells: list[tuple[float, float, float, float]] = []

    def check_current_cell(low_bound: float, high_bound: float):
        if s_gg <= 1e-18:
            return
        gamma = s_gd / s_gg
        if gamma <= 1e-9 or not isfinite(gamma):
            return
        mse = (
            sum(
                (delta_rows[tau][i] - gamma * g_mat[tau][i]) ** 2
                for tau in range(t_count)
                for i in range(n)
            )
            / total_entries
        )
        if mse <= eff_tol:
            max_row_mse = max(
                sum((delta_rows[tau][i] - gamma * g_mat[tau][i]) ** 2 for i in range(n)) / n
                for tau in range(t_count)
            )
            if max_row_mse <= max(eff_tol * 4.0, 1e-12):
                supported_cells.append((low_bound, high_bound, gamma, mse))

    upper_cap = (grouped_thresholds[-1] + 1.0) if grouped_thresholds else 4.0
    first_high = grouped_thresholds[0] if grouped_thresholds else upper_cap
    check_current_cell(0.0, first_high)

    for idx, batch in enumerate(grouped_batches):
        for tau, i, j, di, dj in batch:
            old_gi = g_mat[tau][i]
            new_gi = old_gi + di
            g_mat[tau][i] = new_gi
            s_gg += new_gi * new_gi - old_gi * old_gi
            s_gd += di * delta_rows[tau][i]

            old_gj = g_mat[tau][j]
            new_gj = old_gj + dj
            g_mat[tau][j] = new_gj
            s_gg += new_gj * new_gj - old_gj * old_gj
            s_gd += dj * delta_rows[tau][j]

        low_b = grouped_thresholds[idx]
        high_b = grouped_thresholds[idx + 1] if idx + 1 < len(grouped_thresholds) else upper_cap
        check_current_cell(low_b, high_b)

    if not supported_cells:
        return None

    best_cell = min(supported_cells, key=lambda item: item[3])
    low_interval = min(cell[0] for cell in supported_cells)
    high_interval = max(cell[1] for cell in supported_cells)
    return {
        "direction": direction,
        "edge_condition": describe_edge_condition(edge_condition),
        "q_to_x": list(q_to_x),
        "radius_interval": [float(low_interval), float(high_interval)],
        "coupling_estimate": float(best_cell[2]),
        "mse": float(best_cell[3]),
        "supported_cell_count": len(supported_cells),
    }


def _screen_on_single_transition(
    q_row: list[float],
    x_row: list[float],
    delta_row: list[float],
    direction: str,
    bijections: list[tuple[int, ...]],
    eff_tol: float,
    *,
    edge_condition: str | Program | dict = "unconditional",
) -> list[tuple[int, ...]]:
    """Losslessly filter ``bijections`` on one active transition ``delta_row``."""
    if edge_condition != "unconditional":
        return [
            q_to_x
            for q_to_x in bijections
            if _exact_interval_across_transitions(
                [q_row],
                [x_row],
                [delta_row],
                direction,
                q_to_x,
                eff_tol,
                edge_condition=edge_condition,
            )
            is not None
        ]

    n = len(q_row)
    pos_dists = sorted(
        {
            abs(x_row[a] - x_row[b])
            for a in range(n)
            for b in range(a + 1, n)
            if abs(x_row[a] - x_row[b]) > 0.0
        }
    )
    test_radii = []
    if pos_dists:
        if direction == "symmetric" and any(
            x_row[a] == x_row[b] for a in range(n) for b in range(a + 1, n)
        ):
            test_radii.append(pos_dists[0] * 0.5)
        for left, right in zip(pos_dists, pos_dists[1:]):
            test_radii.append(0.5 * (left + right))
        test_radii.append(pos_dists[-1] + 1.0)
    elif direction == "symmetric":
        test_radii.append(1.0)
    if not test_radii:
        return []

    adjacencies = []
    for r_val in test_radii:
        if direction == "symmetric":
            adj = [
                [other for other in range(n) if other != idx and abs(x_row[idx] - x_row[other]) < r_val]
                for idx in range(n)
            ]
            adjacencies.append((adj, None))
        else:
            sign = 1.0 if direction == "right" else -1.0
            incoming = [
                [src for src in range(n) if src != dst and 0.0 < sign * (x_row[dst] - x_row[src]) < r_val]
                for dst in range(n)
            ]
            outdeg = [
                sum(
                    1
                    for dst in range(n)
                    if dst != src and 0.0 < sign * (x_row[dst] - x_row[src]) < r_val
                )
                for src in range(n)
            ]
            adjacencies.append((incoming, outdeg))

    survivors = []
    for q_to_x in bijections:
        x_to_q = [0] * n
        for q_idx, x_idx in enumerate(q_to_x):
            x_to_q[x_idx] = q_idx
        matched = False
        for first_part, second_part in adjacencies:
            s_gg = 0.0
            s_gd = 0.0
            g_vec = [0.0] * n
            if direction == "symmetric":
                for i in range(n):
                    xi = q_to_x[i]
                    qi = q_row[i]
                    val = 0.0
                    for xj in first_part[xi]:
                        val += q_row[x_to_q[xj]] - qi
                    g_vec[i] = val
                    s_gg += val * val
                    s_gd += val * delta_row[i]
            else:
                for i in range(n):
                    xi = q_to_x[i]
                    val = -second_part[xi] * q_row[i]
                    for x_src in first_part[xi]:
                        val += q_row[x_to_q[x_src]]
                    g_vec[i] = val
                    s_gg += val * val
                    s_gd += val * delta_row[i]
            if s_gg <= 1e-18:
                continue
            gamma = s_gd / s_gg
            if gamma <= 1e-9:
                continue
            mse = sum((delta_row[i] - gamma * g_vec[i]) ** 2 for i in range(n)) / n
            if mse <= eff_tol:
                matched = True
                break
        if matched:
            survivors.append(q_to_x)
    return survivors


def extract_necessary_active_transfer_geometry(
    q_rows: list[list[float]],
    x_rows: list[list[float]],
    delta_rows: list[list[float]],
    bijections: list[tuple[int, ...]],
    budget: SearchBudget,
    eff_tol: float,
) -> dict | None:
    """Infer predicate-agnostic geometric constraints from conservative active transfers.

    Whenever ``P_conservation`` holds on public ``(before, after)`` transitions,
    any active edge ``(src -> dst)`` transfers ``gamma * q_src`` and MUST lie inside
    ``0 < sign(d) * (x_dst - x_src) <= r_min < r`` regardless of ``edge_predicate``.
    Charges ``SearchBudget`` for every transition and bijection inspected.
    """
    from collections import Counter

    if not q_rows or not bijections:
        return None
    n = len(q_rows[0])
    if not budget.spend(max(1, len(q_rows) * n)):
        return None

    pair_tol = max(1e-7, sqrt(eff_tol))
    single_pairs: list[tuple[int, int, int, float]] = []
    for tau, d_vec in enumerate(delta_rows):
        nz = [i for i, v in enumerate(d_vec) if abs(v) > pair_tol]
        if len(nz) == 2:
            i, j = nz
            if abs(d_vec[i] + d_vec[j]) <= pair_tol * 4.0:
                for src, dst in ((i, j), (j, i)):
                    if abs(q_rows[tau][src]) > 1e-5:
                        g_est = -d_vec[src] / q_rows[tau][src]
                        if 0.01 < g_est < 1.0:
                            single_pairs.append((tau, src, dst, g_est))
    if not single_pairs:
        return None

    def _decode_for_gamma(gamma_val: float) -> list[tuple[int, int, int]]:
        decoded: list[tuple[int, int, int]] = []
        for tau, d_vec in enumerate(delta_rows):
            donors = []
            for src in range(n):
                unit = gamma_val * q_rows[tau][src]
                if abs(unit) > 1e-6:
                    ratio = -d_vec[src] / unit
                    if ratio > 0.5 and abs(ratio - round(ratio)) <= 2e-3:
                        donors.append((src, int(round(ratio)), unit))
            if len(donors) == 1:
                src, k, unit = donors[0]
                recips = [
                    dst
                    for dst in range(n)
                    if dst != src and abs(d_vec[dst] - unit) <= max(1e-5, abs(unit) * 3e-3)
                ]
                if len(recips) == k:
                    for dst in recips:
                        decoded.append((tau, src, dst))
        return decoded

    gamma_bins = Counter(round(item[3], 4) for item in single_pairs)
    mode_bin, count = gamma_bins.most_common(1)[0]
    if count >= 2:
        gamma_est = sum(item[3] for item in single_pairs if abs(item[3] - mode_bin) <= 5e-4) / sum(
            1 for item in single_pairs if abs(item[3] - mode_bin) <= 5e-4
        )
        decoded_edges = _decode_for_gamma(gamma_est)
    else:
        best_dec: list[tuple[int, int, int]] = []
        gamma_est = single_pairs[0][3]
        for _, _, _, g_cand in single_pairs:
            cand_dec = _decode_for_gamma(g_cand)
            if len(cand_dec) > len(best_dec):
                best_dec = cand_dec
                gamma_est = g_cand
        decoded_edges = best_dec if len({t for t, _, _ in best_dec}) >= 2 else []
    if not decoded_edges:
        return None

    by_direction: dict[str, list[tuple[tuple[int, ...], float]]] = {}
    for direction in ("right", "left"):
        if not budget.spend(max(1, len(bijections))):
            break
        sign = 1.0 if direction == "right" else -1.0
        valid_bijs: list[tuple[tuple[int, ...], float]] = []
        for q_to_x in bijections:
            dists = [
                sign * (x_rows[tau][q_to_x[dst]] - x_rows[tau][q_to_x[src]])
                for tau, src, dst in decoded_edges
            ]
            if dists and all(d > 0.0 for d in dists):
                valid_bijs.append((q_to_x, float(max(dists))))
        if valid_bijs:
            by_direction[direction] = valid_bijs

    if not by_direction:
        return None

    fired_by_tau: dict[int, set[tuple[int, int]]] = {}
    for tau, src, dst in decoded_edges:
        fired_by_tau.setdefault(tau, set()).add((src, dst))

    certified_samples: dict[str, dict] = {}
    for direction, valid_bijs in by_direction.items():
        if len(valid_bijs) > 2:
            continue
        q_to_x, r_min = valid_bijs[0]
        sign = 1.0 if direction == "right" else -1.0
        fired_rows = []
        blocked_rows = []
        for tau, d_vec in enumerate(delta_rows):
            is_zero = max(abs(v) for v in d_vec) <= pair_tol
            t_fired = fired_by_tau.get(tau, set())
            if not is_zero and not t_fired:
                continue
            for src in range(n):
                if abs(q_rows[tau][src]) <= 1e-5:
                    continue
                for dst in range(n):
                    if src == dst:
                        continue
                    dist = sign * (x_rows[tau][q_to_x[dst]] - x_rows[tau][q_to_x[src]])
                    if 0.0 < dist <= r_min:
                        entry = {
                            "tau": tau,
                            "q_src": round(float(q_rows[tau][src]), 5),
                            "q_dst": round(float(q_rows[tau][dst]), 5),
                            "x_src": round(float(x_rows[tau][q_to_x[src]]), 5),
                            "x_dst": round(float(x_rows[tau][q_to_x[dst]]), 5),
                            "fired": (src, dst) in t_fired,
                        }
                        if entry["fired"]:
                            fired_rows.append(entry)
                        elif is_zero or all(s != src for s, _ in t_fired):
                            blocked_rows.append(entry)
        # Preserve dialectical counterexamples to built-in source_greater (q_src > q_dst yet blocked)
        # across the full trajectory rather than truncating chronologically at tau < 20.
        sg_falsifying_pos = [
            r for r in blocked_rows if r["q_src"] > 0.0 and r["q_src"] > r["q_dst"]
        ]
        sg_falsifying_neg = [
            r for r in blocked_rows if r["q_src"] <= 0.0 and r["q_src"] > r["q_dst"]
        ]
        other_blocked = [r for r in blocked_rows if r["q_src"] <= r["q_dst"]]
        curated_blocked = [
            *sg_falsifying_pos[:14],
            *sg_falsifying_neg[:10],
            *other_blocked[:8],
        ]
        certified_samples[direction] = {
            "surviving_bijection_count": len(valid_bijs),
            "representative_q_to_x": list(q_to_x),
            "minimum_active_radius": round(r_min, 5),
            "fired_in_radius_edges": fired_rows[:24],
            "blocked_in_radius_edges": curated_blocked,
        }

    return {
        "coupling_estimate": round(float(gamma_est), 6),
        "decoded_active_edge_count": len(decoded_edges),
        "decoded_edges": list(decoded_edges),
        "surviving_by_direction": {
            d: [(list(b), round(rm, 6)) for b, rm in items] for d, items in by_direction.items()
        },
        "certified_in_radius_samples": certified_samples,
    }


def _find_nonzero_comparator_const_paths(
    node: Program, path: tuple[int, ...] = ()
) -> list[tuple[tuple[int, ...], Program]]:
    """Locate ``const(c != 0)`` leaves directly compared against an expression in ``lt``/``ge``."""
    found: list[tuple[tuple[int, ...], Program]] = []
    if node.op in {"lt", "ge"} and len(node.children) == 2:
        left, right = node.children
        if (
            left.op == "const"
            and left.value is not None
            and abs(float(left.value)) > 1e-9
            and right.op != "const"
        ):
            found.append(((*path, 0), right))
        elif (
            right.op == "const"
            and right.value is not None
            and abs(float(right.value)) > 1e-9
            and left.op != "const"
        ):
            found.append(((*path, 1), left))
    for idx, child in enumerate(node.children):
        found.extend(_find_nonzero_comparator_const_paths(child, (*path, idx)))
    return found


def _replace_const_at_path(node: Program, path: tuple[int, ...], new_val: float) -> Program:
    return _replace_node_at_path(node, path, Program("const", value=float(new_val)))


def _replace_node_at_path(node: Program, path: tuple[int, ...], replacement: Program) -> Program:
    if not path:
        return replacement
    head, *rest = path
    new_children = list(node.children)
    new_children[head] = _replace_node_at_path(new_children[head], tuple(rest), replacement)
    return Program(node.op, value=node.value, name=node.name, lag=node.lag, children=tuple(new_children))


def describe_comparator_threshold_parameter(predicate: Program) -> dict | None:
    """Describe a unique 1D continuous comparator constant inside ``predicate``."""
    const_paths = _find_nonzero_comparator_const_paths(predicate)
    if len(const_paths) != 1:
        return None
    c_path, c_expr = const_paths[0]
    cur = predicate
    negations = 0
    for step in c_path[:-1]:
        if cur.op == "not":
            negations += 1
        cur = cur.children[step]
    const_side = c_path[-1]
    init_c = float(cur.children[const_side].value)
    decreasing = (cur.op == "lt" and const_side == 0) or (cur.op == "ge" and const_side == 1)
    if negations % 2 == 1:
        decreasing = not decreasing
    return {
        "path": list(c_path),
        "comparator_lhs": c_expr.to_dict(),
        "comparator_monotonicity": "decreasing" if decreasing else "increasing",
        "initial_value": init_c,
    }


def extract_comparator_threshold_interval(
    predicate: Program,
    q_rows: list[list[float]],
    x_rows: list[list[float]],
    delta_rows: list[list[float]],
    direction: str,
    q_to_x: tuple[int, ...],
    budget: SearchBudget,
    eff_tol: float,
    *,
    anchor_value: float | None = None,
) -> list[float] | None:
    """Compute the exact empirical supported interval ``[c_low, c_high]`` on paid transitions."""
    desc = describe_comparator_threshold_parameter(predicate)
    if desc is None or not q_rows:
        return None
    c_path = tuple(desc["path"])
    c_expr = Program.from_dict(desc["comparator_lhs"])
    center_c = float(desc["initial_value"] if anchor_value is None else anchor_value)
    n = len(q_rows[0])
    if not budget.spend(max(1, len(q_rows) * n)):
        return None
    sign = 1.0 if direction == "right" else -1.0
    vals: set[float] = set()
    ctx = EvaluationContext(Observation({}, 0.0))
    for tau in range(min(len(q_rows), 96)):
        for src in range(n):
            if q_rows[tau][src] == 0.0:
                continue
            x_src = x_rows[tau][q_to_x[src]]
            for dst in range(n):
                if src == dst:
                    continue
                x_dst = x_rows[tau][q_to_x[dst]]
                if direction in {"right", "left"} and sign * (x_dst - x_src) <= 0.0:
                    continue
                bound = c_expr._expand(
                    {
                        "edge_source_position": Program("const", value=float(x_src)),
                        "edge_source_quantity": Program("const", value=float(q_rows[tau][src])),
                        "edge_target_position": Program("const", value=float(x_dst)),
                        "edge_target_quantity": Program("const", value=float(q_rows[tau][dst])),
                    }
                )
                v = float(ctx.evaluate(bound))
                if isfinite(v):
                    vals.add(round(v, 9))
    sorted_vals = sorted(vals)
    span = max(1.0, (sorted_vals[-1] - sorted_vals[0]) if len(sorted_vals) >= 2 else abs(center_c))
    low_bound = (min(sorted_vals[0], center_c) if sorted_vals else center_c) - 0.5 * span
    high_bound = (max(sorted_vals[-1], center_c) if sorted_vals else center_c) + 0.5 * span
    raw_cuts = [low_bound, *sorted_vals, high_bound]
    cuts = [raw_cuts[0]]
    for val in raw_cuts[1:]:
        if val - cuts[-1] > 1e-9:
            cuts.append(val)
    if len(cuts) < 2:
        return None

    def _cell_valid(cell_idx: int) -> bool:
        if not budget.spend(len(q_rows) * n):
            return False
        mid = 0.5 * (cuts[cell_idx] + cuts[cell_idx + 1])
        return (
            _exact_interval_across_transitions(
                q_rows,
                x_rows,
                delta_rows,
                direction,
                q_to_x,
                eff_tol,
                edge_condition=_replace_const_at_path(predicate, c_path, mid),
            )
            is not None
        )

    start_idx = next(
        (idx for idx in range(len(cuts) - 1) if cuts[idx] <= center_c <= cuts[idx + 1]),
        0,
    )
    if not _cell_valid(start_idx):
        # Check immediate neighbor if center_c sat on a cut boundary
        alt_idx = next(
            (
                cand
                for cand in (start_idx - 1, start_idx + 1)
                if 0 <= cand < len(cuts) - 1 and _cell_valid(cand)
            ),
            None,
        )
        if alt_idx is None:
            return None
        start_idx = alt_idx

    left_idx = start_idx
    while left_idx > 0 and _cell_valid(left_idx - 1):
        left_idx -= 1
    right_idx = start_idx
    while right_idx + 1 < len(cuts) - 1 and _cell_valid(right_idx + 1):
        right_idx += 1
    return [float(cuts[left_idx]), float(cuts[right_idx + 1])]


def _expr_uses_position_args(node: Program) -> bool:
    if node.op == "arg" and node.name in {"edge_source_position", "edge_target_position"}:
        return True
    return any(_expr_uses_position_args(child) for child in node.children)


def _candidate_comparator_thresholds(
    expr: Program,
    q_rows: list[list[float]],
    x_rows: list[list[float]],
    q_to_x: tuple[int, ...] | None,
    direction: str,
    initial_c: float,
    budget: SearchBudget,
    max_candidates: int = 64,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> list[float]:
    """Compute empirical boundary midpoints for a 1D comparator expression ``expr``."""
    n = len(q_rows[0])
    if not budget.spend(max(1, len(q_rows) * n)):
        return []
    sign = 1.0 if direction == "right" else -1.0
    uses_pos = _expr_uses_position_args(expr)
    if q_to_x is None and uses_pos:
        q_to_x = tuple(range(n))
    vals: set[float] = set()
    ctx = EvaluationContext(Observation({}, 0.0))
    for tau in range(min(len(q_rows), 96)):
        for src in range(n):
            if q_rows[tau][src] == 0.0:
                continue
            x_src = x_rows[tau][q_to_x[src]] if q_to_x is not None else 0.0
            for dst in range(n):
                if src == dst:
                    continue
                x_dst = x_rows[tau][q_to_x[dst]] if q_to_x is not None else 0.0
                if q_to_x is not None and sign * (x_dst - x_src) <= 0.0:
                    continue
                bound = expr._expand(
                    {
                        "edge_source_position": Program("const", value=float(x_src)),
                        "edge_source_quantity": Program("const", value=float(q_rows[tau][src])),
                        "edge_target_position": Program("const", value=float(x_dst)),
                        "edge_target_quantity": Program("const", value=float(q_rows[tau][dst])),
                    }
                )
                v = float(ctx.evaluate(bound))
                if isfinite(v):
                    vals.add(round(v, 8))
    sorted_vals = sorted(vals)
    if len(sorted_vals) < 2:
        return []
    midpoints = [
        0.5 * (a + b)
        for a, b in zip(sorted_vals, sorted_vals[1:])
        if abs(b - a) > 1e-7
        and (lower_bound is None or 0.5 * (a + b) >= lower_bound - 1e-7)
        and (upper_bound is None or 0.5 * (a + b) <= upper_bound + 1e-7)
    ]
    if not midpoints:
        return []
    near = sorted(midpoints, key=lambda m: abs(m - initial_c))[: max(16, max_candidates // 2)]
    q_count = max(48, max_candidates)
    quantiles = (
        [midpoints[round(i * (len(midpoints) - 1) / (q_count - 1))] for i in range(q_count)]
        if len(midpoints) > q_count
        else list(midpoints)
    )
    combined: list[float] = []
    for cand in [*near, *quantiles]:
        if all(abs(cand - prev) > 1e-9 for prev in combined):
            combined.append(cand)
    return combined[: max(max_candidates, 64)]


def screen_custom_edge_predicate_on_transitions(
    q_rows: list[list[float]],
    x_rows: list[list[float]],
    delta_rows: list[list[float]],
    dynamic: list[str],
    x_fields: list[str],
    bijections: list[tuple[int, ...]],
    predicate: Program,
    directions: tuple[str, ...] = ("right", "left"),
    *,
    budget: SearchBudget,
    eff_tol: float,
) -> dict:
    """Verify an open-grammar edge ``predicate`` across all paid transitions."""
    if not q_rows or not bijections:
        return {"supported": False, "matches": []}
    active_threshold = max(1e-7, sqrt(eff_tol))
    active_taus = sorted(
        (
            tau
            for tau, d_vec in enumerate(delta_rows)
            if max(abs(val) for val in d_vec) > active_threshold
        ),
        key=lambda tau: sum(val * val for val in delta_rows[tau]),
        reverse=True,
    )
    if not active_taus:
        return {"supported": False, "matches": [], "reason": "inactive_transitions"}

    nec_geom = extract_necessary_active_transfer_geometry(
        q_rows, x_rows, delta_rows, bijections, budget, eff_tol
    )
    tau0 = active_taus[0]
    tau1 = active_taus[1] if len(active_taus) > 1 else None
    n = len(dynamic)

    def _run_exact(pred_ast: Program) -> tuple[list[dict], tuple[str, tuple[int, ...], dict] | None, bool]:
        found_matches = []
        witness_cand = None
        for direction in directions:
            dir_bijs = bijections
            if nec_geom and direction in nec_geom["surviving_by_direction"]:
                dir_bijs = [
                    tuple(b) for b, _ in nec_geom["surviving_by_direction"][direction]
                ]
            elif nec_geom and nec_geom["surviving_by_direction"]:
                continue
            if not budget.spend(max(1, len(dir_bijs) * n)):
                return [], witness_cand, True
            survivors = _screen_on_single_transition(
                q_rows[tau0],
                x_rows[tau0],
                delta_rows[tau0],
                direction,
                dir_bijs,
                eff_tol,
                edge_condition=pred_ast,
            )
            if survivors and witness_cand is None:
                single_fit = _exact_interval_across_transitions(
                    [q_rows[tau0]],
                    [x_rows[tau0]],
                    [delta_rows[tau0]],
                    direction,
                    survivors[0],
                    eff_tol,
                    edge_condition=pred_ast,
                )
                if single_fit is not None:
                    witness_cand = (direction, survivors[0], single_fit)
            if len(survivors) > 8 and tau1 is not None:
                if not budget.spend(max(1, len(survivors) * n)):
                    return [], witness_cand, True
                survivors = _screen_on_single_transition(
                    q_rows[tau1],
                    x_rows[tau1],
                    delta_rows[tau1],
                    direction,
                    survivors,
                    eff_tol,
                    edge_condition=pred_ast,
                )
            for q_to_x in survivors:
                if not budget.spend(len(q_rows) * n):
                    return [], witness_cand, True
                solved = _exact_interval_across_transitions(
                    q_rows,
                    x_rows,
                    delta_rows,
                    direction,
                    q_to_x,
                    eff_tol,
                    edge_condition=pred_ast,
                )
                if solved is not None:
                    solved["mapping"] = {
                        x_fields[x_idx]: dynamic[q_idx] for q_idx, x_idx in enumerate(q_to_x)
                    }
                    found_matches.append(solved)
        return found_matches, witness_cand, False

    def _attach_threshold_meta(
        matched_list: list[dict], cal_prog: Program, *, recenter: bool = False
    ) -> dict:
        desc = describe_comparator_threshold_parameter(cal_prog)
        t_interval = None
        if desc is not None and matched_list:
            rep = matched_list[0]
            q_idx_by_name = {name: i for i, name in enumerate(dynamic)}
            x_idx_by_name = {name: i for i, name in enumerate(x_fields)}
            rep_q_to_x = [0] * n
            for x_f, q_f in rep["mapping"].items():
                rep_q_to_x[q_idx_by_name[q_f]] = x_idx_by_name[x_f]
            t_interval = extract_comparator_threshold_interval(
                cal_prog,
                q_rows,
                x_rows,
                delta_rows,
                rep["direction"],
                tuple(rep_q_to_x),
                budget,
                eff_tol,
                anchor_value=desc["initial_value"],
            )
            if t_interval is not None:
                if recenter and not (float(t_interval[0]) < float(desc["initial_value"]) < float(t_interval[1])):
                    c_center = 0.5 * (float(t_interval[0]) + float(t_interval[1]))
                    cal_prog = _replace_const_at_path(cal_prog, tuple(desc["path"]), c_center)
                    desc = {**desc, "initial_value": c_center}
                elif recenter:
                    c_center = 0.5 * (float(t_interval[0]) + float(t_interval[1]))
                    cal_prog = _replace_const_at_path(cal_prog, tuple(desc["path"]), c_center)
                    desc = {**desc, "initial_value": c_center}
                for m in matched_list:
                    m["threshold_interval"] = list(t_interval)
        return {
            "supported": True,
            "matches": matched_list,
            "calibrated_predicate": cal_prog,
            "threshold_interval": t_interval,
            "threshold_parameter": desc,
        }

    matches, tau0_witness_candidate, exhausted = _run_exact(predicate)
    if exhausted:
        return {"supported": False, "matches": [], "reason": "compute_exhausted"}
    if matches:
        return _attach_threshold_meta(matches, predicate, recenter=False)

    # If literal threshold missed on finite transitions (or target apparatus has shifted threshold),
    # deduce necessary [lower, upper] bounds from decoded active & blocked edges, then scan 1D cells.
    desc_param = describe_comparator_threshold_parameter(predicate)
    if desc_param is not None:
        c_path = tuple(desc_param["path"])
        c_expr = Program.from_dict(desc_param["comparator_lhs"])
        decreasing = desc_param["comparator_monotonicity"] == "decreasing"
        init_c = float(desc_param["initial_value"])
        pred_minus = _replace_const_at_path(predicate, c_path, -1e9)
        pred_plus = _replace_const_at_path(predicate, c_path, 1e9)
        eval_ctx = EvaluationContext(Observation({}, 0.0))
        pair_tol = max(1e-7, sqrt(eff_tol))

        def _lhs_val(tau: int, src: int, dst: int, x_src: float, x_dst: float) -> float:
            bound = c_expr._expand(
                {
                    "edge_source_position": Program("const", value=float(x_src)),
                    "edge_source_quantity": Program("const", value=float(q_rows[tau][src])),
                    "edge_target_position": Program("const", value=float(x_dst)),
                    "edge_target_quantity": Program("const", value=float(q_rows[tau][dst])),
                }
            )
            return float(eval_ctx.evaluate(bound))

        if nec_geom and nec_geom["surviving_by_direction"]:
            decoded_edges = nec_geom.get("decoded_edges", ())
            for d_name, bij_items in nec_geom["surviving_by_direction"].items():
                if d_name not in directions or not bij_items:
                    continue
                sign = 1.0 if d_name == "right" else -1.0
                for bij_list, r_min in bij_items:
                    rep_bij = tuple(bij_list)
                    lower_b = -float("inf")
                    upper_b = float("inf")
                    if budget.spend(max(1, len(q_rows) * n)):
                        for tau, src, dst in decoded_edges:
                            xs, xd = x_rows[tau][rep_bij[src]], x_rows[tau][rep_bij[dst]]
                            qs, qd = q_rows[tau][src], q_rows[tau][dst]
                            if _edge_condition_holds(pred_minus, xs, qs, xd, qd) != _edge_condition_holds(
                                pred_plus, xs, qs, xd, qd
                            ):
                                v = _lhs_val(tau, src, dst, xs, xd)
                                if decreasing:
                                    upper_b = min(upper_b, v)
                                else:
                                    lower_b = max(lower_b, v)
                        for tau, d_vec in enumerate(delta_rows):
                            for src in range(n):
                                if q_rows[tau][src] == 0.0 or abs(d_vec[src]) > pair_tol:
                                    continue
                                xs = x_rows[tau][rep_bij[src]]
                                qs = q_rows[tau][src]
                                for dst in range(n):
                                    if src == dst:
                                        continue
                                    xd = x_rows[tau][rep_bij[dst]]
                                    if 0.0 < sign * (xd - xs) <= r_min + 1e-9:
                                        qd = q_rows[tau][dst]
                                        if _edge_condition_holds(
                                            pred_minus, xs, qs, xd, qd
                                        ) != _edge_condition_holds(pred_plus, xs, qs, xd, qd):
                                            v = _lhs_val(tau, src, dst, xs, xd)
                                            if decreasing:
                                                lower_b = max(lower_b, v)
                                            else:
                                                upper_b = min(upper_b, v)
                    if lower_b < upper_b - 1e-12:
                        if isfinite(lower_b) and isfinite(upper_b):
                            direct_cs = [0.5 * (lower_b + upper_b)]
                        elif isfinite(upper_b):
                            direct_cs = [upper_b - 0.05, upper_b - 0.2]
                        elif isfinite(lower_b):
                            direct_cs = [lower_b + 0.05, lower_b + 0.2]
                        else:
                            direct_cs = []
                        for c_direct in direct_cs:
                            alt_direct = _replace_const_at_path(predicate, c_path, c_direct)
                            if not budget.spend(len(q_rows) * n):
                                break
                            if (
                                _exact_interval_across_transitions(
                                    q_rows,
                                    x_rows,
                                    delta_rows,
                                    d_name,
                                    rep_bij,
                                    eff_tol,
                                    edge_condition=alt_direct,
                                )
                                is not None
                            ):
                                cal_matches, _, _ = _run_exact(alt_direct)
                                if cal_matches:
                                    return _attach_threshold_meta(cal_matches, alt_direct, recenter=True)
                    if lower_b > upper_b + 1e-9:
                        continue
                    mids = _candidate_comparator_thresholds(
                        c_expr,
                        q_rows,
                        x_rows,
                        rep_bij,
                        d_name,
                        init_c,
                        budget,
                        max_candidates=64,
                        lower_bound=lower_b if isfinite(lower_b) else None,
                        upper_bound=upper_b if isfinite(upper_b) else None,
                    )
                    valid_cs = []
                    for cand_c in mids:
                        alt_pred = _replace_const_at_path(predicate, c_path, cand_c)
                        if not budget.spend(len(q_rows) * n):
                            break
                        if (
                            _exact_interval_across_transitions(
                                q_rows,
                                x_rows,
                                delta_rows,
                                d_name,
                                rep_bij,
                                eff_tol,
                                edge_condition=alt_pred,
                            )
                            is not None
                        ):
                            valid_cs.append(cand_c)
                    if valid_cs:
                        c_mid = 0.5 * (min(valid_cs) + max(valid_cs))
                        cal_pred = _replace_const_at_path(predicate, c_path, c_mid)
                        cal_matches, _, _ = _run_exact(cal_pred)
                        if cal_matches:
                            return _attach_threshold_meta(cal_matches, cal_pred, recenter=True)
        else:
            mids_any = _candidate_comparator_thresholds(
                c_expr,
                q_rows,
                x_rows,
                None,
                directions[0] if directions else "right",
                init_c,
                budget,
                max_candidates=96,
            )
            for cand_c in mids_any[:96]:
                alt_pred = _replace_const_at_path(predicate, c_path, cand_c)
                cal_matches, _, ex_c = _run_exact(alt_pred)
                if ex_c:
                    break
                if cal_matches:
                    return _attach_threshold_meta(cal_matches, alt_pred, recenter=True)

    # Construct truthful counterexamples using necessary-geometry bijection when available
    counterexamples = []
    witness_setup = None
    if nec_geom and nec_geom["surviving_by_direction"]:
        for d_name, bij_items in nec_geom["surviving_by_direction"].items():
            if d_name in directions and bij_items:
                witness_setup = (
                    d_name,
                    tuple(bij_items[0][0]),
                    float(nec_geom["coupling_estimate"]),
                    float(bij_items[0][1]) + 0.02,
                )
                break
    if witness_setup is None and tau0_witness_candidate is not None:
        w_dir, w_bij, w_fit = tau0_witness_candidate
        witness_setup = (
            w_dir,
            w_bij,
            float(w_fit["coupling_estimate"]),
            0.5 * (w_fit["radius_interval"][0] + min(w_fit["radius_interval"][1], 2.0)),
        )
    if witness_setup is not None and budget.spend(len(q_rows) * n):
        w_dir, w_bij, w_gamma, w_rad = witness_setup
        sign = 1.0 if w_dir == "right" else -1.0
        scored_ce = []
        for tau in range(len(q_rows)):
            pred_d = [0.0] * n
            for src in range(n):
                q_src = q_rows[tau][src]
                if q_src == 0.0:
                    continue
                x_src = x_rows[tau][w_bij[src]]
                for dst in range(n):
                    if src == dst:
                        continue
                    x_dst = x_rows[tau][w_bij[dst]]
                    if 0.0 < sign * (x_dst - x_src) < w_rad and _edge_condition_holds(
                        predicate, x_src, q_src, x_dst, q_rows[tau][dst]
                    ):
                        pred_d[src] -= w_gamma * q_src
                        pred_d[dst] += w_gamma * q_src
            row_mse = sum((delta_rows[tau][i] - pred_d[i]) ** 2 for i in range(n)) / n
            if row_mse > max(eff_tol * 4.0, 1e-10):
                scored_ce.append(
                    (
                        row_mse,
                        {
                            "tau": tau,
                            "direction": w_dir,
                            "q": [round(q_rows[tau][i], 5) for i in range(n)],
                            "paired_x_estimate": [round(x_rows[tau][w_bij[i]], 5) for i in range(n)],
                            "observed_delta": [round(delta_rows[tau][i], 6) for i in range(n)],
                            "candidate_predicted_delta": [round(pred_d[i], 6) for i in range(n)],
                        },
                    )
                )
        scored_ce.sort(key=lambda item: item[0], reverse=True)
        counterexamples = [item[1] for item in scored_ce[:4]]
    return {"supported": False, "matches": [], "counterexamples": counterexamples}


def _solve_exchange_for_condition(
    q_rows: list[list[float]],
    x_rows: list[list[float]],
    delta_rows: list[list[float]],
    dynamic: list[str],
    x_fields: list[str],
    bijections: list[tuple[int, ...]],
    ordered_directions: list[str],
    active_taus: list[int],
    eff_tol: float,
    budget: SearchBudget,
    *,
    edge_condition: str | Program | dict = "unconditional",
) -> tuple[dict[str, list[dict]], bool]:
    """Screen and verify ``bijections`` across ``ordered_directions`` for ``edge_condition``."""
    n = len(dynamic)
    tau0 = active_taus[0]
    tau1 = active_taus[1] if len(active_taus) > 1 else None
    distinct_dists_tau0 = max(
        1,
        len(
            {
                abs(x_rows[tau0][a] - x_rows[tau0][b])
                for a in range(n)
                for b in range(a + 1, n)
            }
        ),
    )
    supported_by_direction: dict[str, list[dict]] = {}
    for direction in ordered_directions:
        if direction == "symmetric" and edge_condition != "unconditional":
            continue
        if not budget.spend(max(1, len(bijections) * n)):
            return supported_by_direction, True
        survivors = _screen_on_single_transition(
            q_rows[tau0],
            x_rows[tau0],
            delta_rows[tau0],
            direction,
            bijections,
            eff_tol,
            edge_condition=edge_condition,
        )
        if len(survivors) > 8 and tau1 is not None:
            if not budget.spend(max(1, len(survivors) * distinct_dists_tau0)):
                return supported_by_direction, True
            survivors = _screen_on_single_transition(
                q_rows[tau1],
                x_rows[tau1],
                delta_rows[tau1],
                direction,
                survivors,
                eff_tol,
                edge_condition=edge_condition,
            )
        dir_matches = []
        for q_to_x in survivors:
            if not budget.spend(len(q_rows) * n):
                return supported_by_direction, True
            solved = _exact_interval_across_transitions(
                q_rows,
                x_rows,
                delta_rows,
                direction,
                q_to_x,
                eff_tol,
                edge_condition=edge_condition,
            )
            if solved is not None:
                solved["mapping"] = {
                    x_fields[x_idx]: dynamic[q_idx] for q_idx, x_idx in enumerate(q_to_x)
                }
                dir_matches.append(solved)
        if dir_matches:
            supported_by_direction[direction] = dir_matches
    return supported_by_direction, False


def _audit_focus_membership_from_trajectories(
    model: dict,
    records: list[dict],
    narrowed_position_domains: dict[str, list[str]] | None,
    budget: SearchBudget,
    eff_tol: float,
    max_bijections: int = 720,
    *,
    drive_by_q_override: dict[str, str] | None = None,
    allow_higher_order_latent: bool = True,
    verified_direction: str | None = None,
    radius_interval: list[float] | None = None,
    commitments: list[dict] | tuple[dict, ...] = (),
) -> dict | None:
    """Audit ``P_focus_membership`` (``f -> q``) on paid multi-step trajectories.

    Uses ONLY public ``before``, ``after``, ``target``, ``probe_id``, and ``position``
    fields from already-purchased transitions. Never narrows ``focus_candidates``
    when the readout is focus-invariant (e.g. ``total``) or outside the observable
    flow/state basis, and takes the sound union over all exact-fitting bijections.
    """
    from .continuous_state import _linear_fit

    focus_cands = model.get("focus_candidates") or {}
    dynamic = list(model.get("dynamic_fields", ()))
    groups = model.get("groups") or ()
    n = len(dynamic)
    if n == 0 or len(focus_cands) != n or not groups:
        return None
    f_fields = list(focus_cands.keys())
    if all(len(focus_cands[f].get("quantities", ())) == 1 for f in f_fields):
        return {
            "status": "already_resolved",
            "exhaustive": True,
            "surviving_count": 1,
            "narrowed_focus_domains": {
                f: list(focus_cands[f]["quantities"]) for f in f_fields
            },
        }

    u_by_q = (
        dict(drive_by_q_override)
        if drive_by_q_override and len(drive_by_q_override) == n
        else {
            g["q"]: g["u"]
            for g in groups
            if isinstance(g, dict) and "q" in g and "u" in g
        }
    )
    if len(u_by_q) != n:
        return None

    by_probe: dict[str, dict[int, dict]] = {}
    for idx, row in enumerate(records):
        if "target" not in row or "before" not in row or "after" not in row:
            continue
        pid = str(row.get("probe_id", f"__anon_{idx}"))
        pos = row.get("position")
        if pos is None:
            pos = len(row.get("history", ()))
        pos = int(pos)
        if pos == 0 and row.get("history"):
            continue
        by_probe.setdefault(pid, {}).setdefault(pos, row)

    # Model-agnostic S_{N-1} symmetry-breaking selector contrast on single-object
    # reset / matched-prefix excitations (zero functional-form assumptions):
    if n >= 3 and budget.spend(max(1, len(by_probe) * n)):
        x_all = sorted((model.get("position_candidates") or {}).keys())
        grouped_excitations: dict[tuple, dict[int, float]] = {}
        for rows_by_pos in by_probe.values():
            for step_pos in (0, 1):
                if step_pos not in rows_by_pos:
                    continue
                if step_pos == 1:
                    if 0 not in rows_by_pos:
                        continue
                    b0 = rows_by_pos[0]["before"]
                    if not all(
                        abs(float(b0.get(q, 0.0))) <= 1e-9
                        and abs(float(b0.get(u_by_q[q], 0.0))) <= 1e-9
                        for q in dynamic
                    ):
                        continue
                r_cur = rows_by_pos[step_pos]
                bef = r_cur["before"]
                if not all(
                    q in bef and u_by_q[q] in bef for q in dynamic
                ) or not all(f in bef for f in f_fields):
                    continue
                if x_all and any(abs(float(bef.get(xf, 0.0))) > 1e-9 for xf in x_all):
                    continue
                active_f = [
                    f_idx for f_idx, f in enumerate(f_fields) if abs(float(bef[f]) - 1.0) <= 1e-6
                ]
                if len(active_f) != 1:
                    continue
                excited_qs = [
                    q
                    for q in dynamic
                    if abs(float(bef[q])) > 1e-6 or abs(float(bef[u_by_q[q]])) > 1e-6
                ]
                if len(excited_qs) != 1:
                    continue
                q_exc = excited_qs[0]
                key = (
                    step_pos,
                    q_exc,
                    round(float(bef[q_exc]), 8),
                    round(float(bef[u_by_q[q_exc]]), 8),
                )
                grouped_excitations.setdefault(key, {})[active_f[0]] = float(r_cur["target"])

        inferred_f_for_q: dict[str, int] = {}
        for (_, q_exc, _, _), targets_by_f in grouped_excitations.items():
            if len(targets_by_f) < n or q_exc in inferred_f_for_q:
                continue
            items_f = sorted(targets_by_f.items())
            outliers = []
            for idx_cand, (f_cand, y_cand) in enumerate(items_f):
                others = [y_other for j, (_, y_other) in enumerate(items_f) if j != idx_cand]
                if max(others) - min(others) <= 1e-9 and abs(y_cand - others[0]) > 1e-5:
                    outliers.append(f_cand)
            if len(outliers) == 1:
                inferred_f_for_q[q_exc] = outliers[0]

        if len(inferred_f_for_q) >= n - 1:
            used_fs = set(inferred_f_for_q.values())
            if len(used_fs) == len(inferred_f_for_q):
                if len(inferred_f_for_q) == n - 1:
                    rem_q = next(q for q in dynamic if q not in inferred_f_for_q)
                    rem_f = next(fi for fi in range(n) if fi not in used_fs)
                    inferred_f_for_q[rem_q] = rem_f
                narrowed_direct = {
                    f_fields[f_idx]: [q_name]
                    for q_name, f_idx in inferred_f_for_q.items()
                }
                if all(
                    narrowed_direct[f][0] in focus_cands[f].get("quantities", ())
                    for f in f_fields
                ):
                    return {
                        "status": "supported",
                        "exhaustive": True,
                        "method": "differential_selector_symmetry_contrast",
                        "trajectory_rows": len(records),
                        "surviving_count": 1,
                        "narrowed_focus_domains": narrowed_direct,
                    }

    trajectories: list[list[dict]] = []
    for rows_by_pos in by_probe.values():
        if 0 not in rows_by_pos:
            continue
        seq = []
        step = 0
        while step in rows_by_pos:
            r = rows_by_pos[step]
            before, after = r["before"], r["after"]
            if not all(
                q in before and q in after and u_by_q[q] in before for q in dynamic
            ) or not all(f in before for f in f_fields):
                break
            active_f = [
                f_idx for f_idx, f in enumerate(f_fields) if abs(before[f] - 1.0) <= 1e-6
            ]
            if len(active_f) != 1:
                break
            seq.append((r, active_f[0]))
            step += 1
        if len(seq) >= 4:
            trajectories.append(seq)

    diverse_trajectories = [
        seq for seq in trajectories if len({f_idx for _, f_idx in seq}) >= 2
    ]
    if diverse_trajectories:
        diverse_trajectories.sort(
            key=lambda seq: (-len({f_idx for _, f_idx in seq}), -len(seq))
        )
        selected_trajectories = []
        acc_rows = 0
        for seq in diverse_trajectories:
            selected_trajectories.append(seq)
            acc_rows += len(seq)
            if acc_rows >= 32:
                break
        trajectories = selected_trajectories

    total_rows = sum(len(tr) for tr in trajectories)
    if total_rows < 8:
        return None

    unique_x_for_q: dict[str, str] = {}
    if narrowed_position_domains and all(
        len(v) == 1 for v in narrowed_position_domains.values()
    ):
        unique_x_for_q = {v[0]: x_f for x_f, v in narrowed_position_domains.items()}

    targets: list[float] = []
    totals_per_tr: list[list[float]] = []
    before_q_per_tr: list[list[list[float]]] = []
    before_u_per_tr: list[list[list[float]]] = []
    after_q_per_tr: list[list[list[float]]] = []
    flow_q_per_tr: list[list[list[float]]] = []
    pos_q_per_tr: list[list[list[float]]] | None = [] if len(unique_x_for_q) == n else None
    active_f_per_tr: list[list[int]] = []

    for tr in trajectories:
        tr_tot, tr_b, tr_u, tr_a, tr_phi, tr_f = [], [], [], [], [], []
        tr_x = [] if pos_q_per_tr is not None else None
        for r, f_idx in tr:
            before, after = r["before"], r["after"]
            targets.append(float(r["target"]))
            b_vec = [float(before[q]) for q in dynamic]
            u_vec = [float(before[u_by_q[q]]) for q in dynamic]
            a_vec = [float(after[q]) for q in dynamic]
            phi_vec = [
                float(after[q]) - float(before[q]) - float(before[u_by_q[q]])
                for q in dynamic
            ]
            tr_tot.append(sum(a_vec))
            tr_b.append(b_vec)
            tr_u.append(u_vec)
            tr_a.append(a_vec)
            tr_phi.append(phi_vec)
            tr_f.append(f_idx)
            if tr_x is not None:
                tr_x.append([float(before[unique_x_for_q[q]]) for q in dynamic])
        totals_per_tr.append(tr_tot)
        before_q_per_tr.append(tr_b)
        before_u_per_tr.append(tr_u)
        after_q_per_tr.append(tr_a)
        flow_q_per_tr.append(tr_phi)
        active_f_per_tr.append(tr_f)
        if pos_q_per_tr is not None and tr_x is not None:
            pos_q_per_tr.append(tr_x)

    def _fit_mse(columns: list[list[float]]) -> float:
        coeffs = _linear_fit(columns, targets)
        intercept = coeffs[0]
        weights = coeffs[1:]
        err = 0.0
        for r_idx, y_val in enumerate(targets):
            pred = intercept
            for w_val, col in zip(weights, columns):
                pred += w_val * col[r_idx]
            diff = pred - y_val
            err += diff * diff
        return err / total_rows

    flat_tot = [v for tr_tot in totals_per_tr for v in tr_tot]
    win_tots: dict[int, list[float]] = {}
    for w in (2, 3, 4):
        col = []
        for tr_tot in totals_per_tr:
            for t in range(len(tr_tot)):
                col.append(sum(tr_tot[max(0, t - w + 1) : t + 1]))
        win_tots[w] = col

    if not budget.spend(total_rows * 6):
        return None
    for inv_cols in ([flat_tot], [win_tots[2]], [win_tots[3]], [win_tots[4]]):
        if _fit_mse(inv_cols) <= eff_tol:
            return {"status": "focus_invariant_readout", "exhaustive": True, "surviving_count": 0}

    bijections, exhaustive = _enumerate_position_bijections(
        dynamic, f_fields, focus_cands, max_bijections=max_bijections
    )
    if not exhaustive or not bijections:
        return None

    surviving_f_to_q: list[list[int]] = []
    for q_to_f in bijections:
        if not budget.spend(max(1, total_rows * 2)):
            return None
        f_to_q = [0] * n
        for q_idx, f_idx in enumerate(q_to_f):
            f_to_q[f_idx] = q_idx

        alpha_col: list[float] = []
        q2_col: list[float] = []
        psi_col: list[float] = []
        evt_col: list[float] = []
        xf_col: list[float] | None = [] if pos_q_per_tr is not None else None
        win_psi: dict[int, list[float]] = {2: [], 3: [], 4: []}
        win_evt: dict[int, list[float]] = {2: [], 3: [], 4: []}

        for tr_idx in range(len(trajectories)):
            tr_b = before_q_per_tr[tr_idx]
            tr_a = after_q_per_tr[tr_idx]
            tr_phi = flow_q_per_tr[tr_idx]
            tr_f = active_f_per_tr[tr_idx]
            tr_x = pos_q_per_tr[tr_idx] if pos_q_per_tr is not None else None
            psi_seq: list[float] = []
            evt_seq: list[float] = []
            for t in range(len(tr_f)):
                q_idx = f_to_q[tr_f[t]]
                b_val = tr_b[t][q_idx]
                a_val = tr_a[t][q_idx]
                p_val = tr_phi[t][q_idx]
                e_val = (p_val - psi_seq[-1]) if t >= 1 else 0.0
                psi_seq.append(p_val)
                evt_seq.append(e_val)
                alpha_col.append(a_val)
                q2_col.append(b_val * b_val)
                psi_col.append(p_val)
                evt_col.append(e_val)
                if xf_col is not None and tr_x is not None:
                    xf_col.append(tr_x[t][q_idx])
                for w in (2, 3, 4):
                    s = max(0, t - w + 1)
                    win_psi[w].append(sum(psi_seq[s : t + 1]))
                    win_evt[w].append(sum(evt_seq[s : t + 1]))

        if total_rows >= 12:
            inst_superset = [flat_tot, alpha_col, psi_col, evt_col, q2_col]
            if xf_col is not None:
                inst_superset.append(xf_col)
            passes_screen = _fit_mse(inst_superset) <= eff_tol * 4.0 or any(
                _fit_mse([win_tots[w], alpha_col, win_psi[w], win_evt[w]]) <= eff_tol * 4.0
                for w in (2, 3, 4)
            )
            if not passes_screen:
                continue

        families: list[list[list[float]]] = [
            [alpha_col],
            [psi_col],
            [evt_col],
            [flat_tot, evt_col],
            [flat_tot, evt_col, q2_col],
        ]
        if xf_col is not None:
            families.append([alpha_col, xf_col])
        for w in (2, 3, 4):
            families.append([win_psi[w]])
            families.append([alpha_col, win_psi[w]])
            families.append([win_tots[w], win_evt[w]])

        if any(_fit_mse(cols) <= eff_tol for cols in families):
            surviving_f_to_q.append(f_to_q)

    # Fallback for POMDP per-object focal reservoir readouts (P_markov_state refuted on readout)
    if not surviving_f_to_q and total_rows >= 14:
        best_pole_by_bij: list[tuple[float, list[int]]] = []
        for q_to_f in bijections:
            if not budget.spend(max(1, total_rows * 3)):
                break
            f_to_q = [0] * n
            for q_idx, f_idx in enumerate(q_to_f):
                f_to_q[f_idx] = q_idx
            alpha_col, evt_col = [], []
            for tr_idx in range(len(trajectories)):
                tr_a = after_q_per_tr[tr_idx]
                tr_phi = flow_q_per_tr[tr_idx]
                tr_f = active_f_per_tr[tr_idx]
                psi_seq = []
                for t in range(len(tr_f)):
                    q_idx = f_to_q[tr_f[t]]
                    p_val = tr_phi[t][q_idx]
                    evt_col.append((p_val - psi_seq[-1]) if t >= 1 else 0.0)
                    psi_seq.append(p_val)
                    alpha_col.append(tr_a[t][q_idx])

            def _latent_mse(pole_val: float) -> float:
                zq_col, zu_col = [], []
                for tr_idx in range(len(trajectories)):
                    tr_b = before_q_per_tr[tr_idx]
                    tr_u = before_u_per_tr[tr_idx]
                    tr_f = active_f_per_tr[tr_idx]
                    zq = [0.0] * n
                    zu = [0.0] * n
                    for t in range(len(tr_f)):
                        for obj_i in range(n):
                            zq[obj_i] = pole_val * zq[obj_i] + tr_b[t][obj_i]
                            zu[obj_i] = pole_val * zu[obj_i] + tr_u[t][obj_i]
                        q_idx = f_to_q[tr_f[t]]
                        zq_col.append(zq[q_idx])
                        zu_col.append(zu[q_idx])
                return _fit_mse([flat_tot, alpha_col, evt_col, zq_col, zu_col])

            coarse = [(p, _latent_mse(p)) for p in (0.5, 0.65, 0.8)]
            best_p, best_err = min(coarse, key=lambda item: item[1])
            if best_err <= 5e-3:
                lo_p, hi_p = max(0.2, best_p - 0.18), min(0.95, best_p + 0.18)
                gr = (sqrt(5.0) - 1.0) / 2.0
                c1 = hi_p - gr * (hi_p - lo_p)
                c2 = lo_p + gr * (hi_p - lo_p)
                f1, f2 = _latent_mse(c1), _latent_mse(c2)
                for _ in range(16):
                    if f1 < f2:
                        hi_p, c2, f2 = c2, c1, f1
                        c1 = hi_p - gr * (hi_p - lo_p)
                        f1 = _latent_mse(c1)
                    else:
                        lo_p, c1, f1 = c1, c2, f2
                        c2 = lo_p + gr * (hi_p - lo_p)
                        f2 = _latent_mse(c2)
                best_err = min(f1, f2)
            if best_err <= max(eff_tol * 4.0, 1e-10):
                best_pole_by_bij.append((best_err, f_to_q))
        if best_pole_by_bij:
            surviving_f_to_q = [item[1] for item in best_pole_by_bij]

    # Higher-order AR(2), spatially coupled, and consolidated open-grammar commitment fallbacks:
    if not surviving_f_to_q and total_rows >= 14 and allow_higher_order_latent:
        # 2A. Spatially coupled 1st-order focal memory when x -> q and radius are already resolved
        if pos_q_per_tr is not None and radius_interval is not None:
            r_mid = 0.5 * (float(radius_interval[0]) + float(radius_interval[1]))
            dir_eff = verified_direction or "right"
            sign_eff = 1.0 if dir_eff == "right" else -1.0
            coupled_survivors: list[list[int]] = []
            for q_to_f in bijections:
                if not budget.spend(max(1, total_rows * 4)):
                    break
                f_to_q = [0] * n
                for q_idx, f_idx in enumerate(q_to_f):
                    f_to_q[f_idx] = q_idx
                alpha_col, evt_col = [], []
                for tr_idx in range(len(trajectories)):
                    tr_a = after_q_per_tr[tr_idx]
                    tr_phi = flow_q_per_tr[tr_idx]
                    tr_f = active_f_per_tr[tr_idx]
                    psi_seq = []
                    for t in range(len(tr_f)):
                        q_idx = f_to_q[tr_f[t]]
                        p_val = tr_phi[t][q_idx]
                        evt_col.append((p_val - psi_seq[-1]) if t >= 1 else 0.0)
                        psi_seq.append(p_val)
                        alpha_col.append(tr_a[t][q_idx])

                def _coupled_mse(pole_val: float) -> float:
                    zq_col, zu_col, nb_zq_col, nb_zu_col = [], [], [], []
                    for tr_idx in range(len(trajectories)):
                        tr_b = before_q_per_tr[tr_idx]
                        tr_u = before_u_per_tr[tr_idx]
                        tr_f = active_f_per_tr[tr_idx]
                        tr_x = pos_q_per_tr[tr_idx]
                        zq = [0.0] * n
                        zu = [0.0] * n
                        for t in range(len(tr_f)):
                            for obj_i in range(n):
                                zq[obj_i] = pole_val * zq[obj_i] + tr_b[t][obj_i]
                                zu[obj_i] = pole_val * zu[obj_i] + tr_u[t][obj_i]
                            q_idx = f_to_q[tr_f[t]]
                            zq_col.append(zq[q_idx])
                            zu_col.append(zu[q_idx])
                            x_cur = tr_x[t]
                            if dir_eff in {"right", "left"}:
                                fq_flux = 0.0
                                fu_flux = 0.0
                                for src_i in range(n):
                                    for dst_i in range(n):
                                        if src_i != dst_i and 0.0 < sign_eff * (x_cur[dst_i] - x_cur[src_i]) < r_mid:
                                            if dst_i == q_idx:
                                                fq_flux += zq[src_i]
                                                fu_flux += zu[src_i]
                                            elif src_i == q_idx:
                                                fq_flux -= zq[src_i]
                                                fu_flux -= zu[src_i]
                            else:
                                fq_flux = sum(
                                    zq[oth] - zq[q_idx]
                                    for oth in range(n)
                                    if oth != q_idx and abs(x_cur[oth] - x_cur[q_idx]) < r_mid
                                )
                                fu_flux = sum(
                                    zu[oth] - zu[q_idx]
                                    for oth in range(n)
                                    if oth != q_idx and abs(x_cur[oth] - x_cur[q_idx]) < r_mid
                                )
                            nb_zq_col.append(fq_flux)
                            nb_zu_col.append(fu_flux)
                    return _fit_mse([flat_tot, alpha_col, evt_col, zq_col, zu_col, nb_zq_col, nb_zu_col])

                coarse_c = [(p, _coupled_mse(p)) for p in (0.5, 0.65, 0.8)]
                best_pc, best_errc = min(coarse_c, key=lambda item: item[1])
                if best_errc <= 5e-3:
                    lo_p, hi_p = max(0.2, best_pc - 0.18), min(0.95, best_pc + 0.18)
                    gr = (sqrt(5.0) - 1.0) / 2.0
                    c1 = hi_p - gr * (hi_p - lo_p)
                    c2 = lo_p + gr * (hi_p - lo_p)
                    f1, f2 = _coupled_mse(c1), _coupled_mse(c2)
                    for _ in range(16):
                        if f1 < f2:
                            hi_p, c2, f2 = c2, c1, f1
                            c1 = hi_p - gr * (hi_p - lo_p)
                            f1 = _coupled_mse(c1)
                        else:
                            lo_p, c1, f1 = c1, c2, f2
                            c2 = lo_p + gr * (hi_p - lo_p)
                            f2 = _coupled_mse(c2)
                    best_errc = min(f1, f2)
                if best_errc <= max(eff_tol * 4.0, 1e-10):
                    coupled_survivors.append(f_to_q)
            if coupled_survivors:
                surviving_f_to_q = coupled_survivors

        # 2B. 2nd-order oscillatory AR(2) focal memory fallback
        if not surviving_f_to_q and total_rows >= 16:
            ar2_survivors: list[list[int]] = []
            gr = (sqrt(5.0) - 1.0) / 2.0
            for q_to_f in bijections:
                if not budget.spend(max(1, total_rows * 6)):
                    break
                f_to_q = [0] * n
                for q_idx, f_idx in enumerate(q_to_f):
                    f_to_q[f_idx] = q_idx
                alpha_col, evt_col = [], []
                for tr_idx in range(len(trajectories)):
                    tr_a = after_q_per_tr[tr_idx]
                    tr_phi = flow_q_per_tr[tr_idx]
                    tr_f = active_f_per_tr[tr_idx]
                    psi_seq = []
                    for t in range(len(tr_f)):
                        q_idx = f_to_q[tr_f[t]]
                        p_val = tr_phi[t][q_idx]
                        evt_col.append((p_val - psi_seq[-1]) if t >= 1 else 0.0)
                        psi_seq.append(p_val)
                        alpha_col.append(tr_a[t][q_idx])

                def _ar2_focal_mse(a1_val: float, a2_val: float) -> float:
                    zq0_col, zq1_col, zu0_col, zu1_col = [], [], [], []
                    for tr_idx in range(len(trajectories)):
                        tr_b = before_q_per_tr[tr_idx]
                        tr_u = before_u_per_tr[tr_idx]
                        tr_f = active_f_per_tr[tr_idx]
                        zq_cur, zq_prev = [0.0] * n, [0.0] * n
                        zu_cur, zu_prev = [0.0] * n, [0.0] * n
                        for t in range(len(tr_f)):
                            q_idx = f_to_q[tr_f[t]]
                            zq1_col.append(zq_cur[q_idx])
                            zu1_col.append(zu_cur[q_idx])
                            for obj_i in range(n):
                                nq = a1_val * zq_cur[obj_i] + a2_val * zq_prev[obj_i] + tr_b[t][obj_i]
                                nu = a1_val * zu_cur[obj_i] + a2_val * zu_prev[obj_i] + tr_u[t][obj_i]
                                zq_prev[obj_i], zq_cur[obj_i] = zq_cur[obj_i], nq
                                zu_prev[obj_i], zu_cur[obj_i] = zu_cur[obj_i], nu
                            zq0_col.append(zq_cur[q_idx])
                            zu0_col.append(zu_cur[q_idx])
                    return _fit_mse([flat_tot, alpha_col, evt_col, zq0_col, zq1_col, zu0_col, zu1_col])

                seeds_ar2 = [
                    (2.0 * rho * cw, -(rho * rho))
                    for rho in (0.58, 0.72, 0.84)
                    for cw in (0.25, 0.62, 0.85)
                ]
                best_a1, best_a2, best_err2 = min(
                    ((a1, a2, _ar2_focal_mse(a1, a2)) for a1, a2 in seeds_ar2),
                    key=lambda item: item[2],
                )
                if best_err2 <= 2e-2:
                    span1, span2 = 0.28, 0.20
                    for _ in range(5):
                        lo1, hi1 = max(-1.9, best_a1 - span1), min(1.9, best_a1 + span1)
                        c1 = hi1 - gr * (hi1 - lo1)
                        c2 = lo1 + gr * (hi1 - lo1)
                        f1, f2 = _ar2_focal_mse(c1, best_a2), _ar2_focal_mse(c2, best_a2)
                        for _ in range(10):
                            if f1 < f2:
                                hi1, c2, f2 = c2, c1, f1
                                c1 = hi1 - gr * (hi1 - lo1)
                                f1 = _ar2_focal_mse(c1, best_a2)
                            else:
                                lo1, c1, f1 = c1, c2, f2
                                c2 = lo1 + gr * (hi1 - lo1)
                                f2 = _ar2_focal_mse(c2, best_a2)
                        if f1 < f2 and f1 < best_err2:
                            best_a1, best_err2 = c1, f1
                        elif f2 <= f1 and f2 < best_err2:
                            best_a1, best_err2 = c2, f2

                        lo2, hi2 = max(-0.96, best_a2 - span2), min(-0.02, best_a2 + span2)
                        d1 = hi2 - gr * (hi2 - lo2)
                        d2 = lo2 + gr * (hi2 - lo2)
                        g1, g2 = _ar2_focal_mse(best_a1, d1), _ar2_focal_mse(best_a1, d2)
                        for _ in range(10):
                            if g1 < g2:
                                hi2, d2, g2 = d2, d1, g1
                                d1 = hi2 - gr * (hi2 - lo2)
                                g1 = _ar2_focal_mse(best_a1, d1)
                            else:
                                lo2, d1, g1 = d1, d2, g2
                                d2 = lo2 + gr * (hi2 - lo2)
                                g2 = _ar2_focal_mse(best_a1, d2)
                        if g1 < g2 and g1 < best_err2:
                            best_a2, best_err2 = d1, g1
                        elif g2 <= g1 and g2 < best_err2:
                            best_a2, best_err2 = d2, g2
                        span1 *= 0.38
                        span2 *= 0.38
                if best_err2 <= max(eff_tol * 8.0, 1e-9):
                    ar2_survivors.append(f_to_q)
            if ar2_survivors:
                surviving_f_to_q = ar2_survivors

        # 2C. Consolidated open-grammar / POMDP commitment role-term screening on cross-scale transfer
        if (
            not surviving_f_to_q
            and commitments
            and len(unique_x_for_q) == n
            and radius_interval is not None
        ):
            from .structure_lab import ObjectSchema
            from .world_structure_commitments import (
                _rebind_latent_pole_in_role_terms,
                instantiate_commitment_terms,
            )

            obs_trajs: list[Observation] = []
            for tr in trajectories:
                hist_cur: list[dict] = []
                for r_row, _ in tr:
                    bef_f = dict(r_row["before"])
                    tgt_f = float(r_row["target"])
                    obs_trajs.append(Observation(bef_f, tgt_f, tuple(hist_cur)))
                    hist_cur.append({**bef_f, "__target__": tgt_f})
            r_mid_comm = 0.5 * (float(radius_interval[0]) + float(radius_interval[1]))
            comm0 = commitments[0]
            r_bindings = {
                p["symbol"]: r_mid_comm for p in comm0.get("radius_parameters", ())
            }
            d_overrides = (
                {p["symbol"]: verified_direction for p in comm0.get("radius_parameters", ())}
                if verified_direction
                else None
            )
            has_poles = bool(comm0.get("source_latent_poles"))
            comm_survivors: list[list[int]] = []
            gr = (sqrt(5.0) - 1.0) / 2.0
            for q_to_f in bijections:
                if not budget.spend(max(1, total_rows * 4)):
                    break
                f_by_q = {dynamic[q_idx]: f_fields[f_idx] for q_idx, f_idx in enumerate(q_to_f)}
                objs_tmp = tuple(f"obj_{i}" for i in range(n))
                sc_tmp = ObjectSchema(
                    objs_tmp,
                    {
                        f"obj_{i}": {
                            "q": dynamic[i],
                            "u": u_by_q[dynamic[i]],
                            "x": unique_x_for_q[dynamic[i]],
                            "f": f_by_q[dynamic[i]],
                        }
                        for i in range(n)
                    },
                    {f"obj_{i}": (f"obj_{i}",) for i in range(n)},
                    {f"obj_{i}": {"q": dynamic[i], "u": u_by_q[dynamic[i]], "x": unique_x_for_q[dynamic[i]]} for i in range(n)},
                )

                def _comm_mse(p_val: float | None = None) -> float:
                    rt_ov = (
                        _rebind_latent_pole_in_role_terms(comm0["role_terms"], float(p_val))
                        if p_val is not None
                        else None
                    )
                    built = instantiate_commitment_terms(
                        comm0,
                        sc_tmp,
                        radius_bindings=r_bindings,
                        direction_overrides=d_overrides,
                        role_terms_override=rt_ov,
                    )
                    if not built:
                        return float("inf")
                    cols_b = [[t_p.predict(ob) for ob in obs_trajs] for t_p in built]
                    return _fit_mse(cols_b)

                if not has_poles:
                    err_c = _comm_mse(None)
                else:
                    coarse_cp = [(pv, _comm_mse(pv)) for pv in (0.5, 0.68, 0.82)]
                    bp_c, err_c = min(coarse_cp, key=lambda item: item[1])
                    if err_c <= 1e-2:
                        lo_p, hi_p = max(0.2, bp_c - 0.18), min(0.95, bp_c + 0.18)
                        c1 = hi_p - gr * (hi_p - lo_p)
                        c2 = lo_p + gr * (hi_p - lo_p)
                        f1, f2 = _comm_mse(c1), _comm_mse(c2)
                        for _ in range(14):
                            if f1 < f2:
                                hi_p, c2, f2 = c2, c1, f1
                                c1 = hi_p - gr * (hi_p - lo_p)
                                f1 = _comm_mse(c1)
                            else:
                                lo_p, c1, f1 = c1, c2, f2
                                c2 = lo_p + gr * (hi_p - lo_p)
                                f2 = _comm_mse(c2)
                        err_c = min(err_c, f1, f2)
                if err_c <= max(eff_tol * 4.0, 1e-10):
                    f_to_q = [0] * n
                    for q_idx, f_idx in enumerate(q_to_f):
                        f_to_q[f_idx] = q_idx
                    comm_survivors.append(f_to_q)
            if comm_survivors:
                surviving_f_to_q = comm_survivors

    if not surviving_f_to_q or len(surviving_f_to_q) >= len(bijections):
        return {
            "status": "unresolved_focus_membership",
            "exhaustive": exhaustive,
            "surviving_count": len(surviving_f_to_q),
        }

    narrowed_focus = {
        f_fields[f_idx]: sorted({dynamic[f_to_q[f_idx]] for f_to_q in surviving_f_to_q})
        for f_idx in range(n)
    }
    return {
        "status": "supported"
        if all(len(v) == 1 for v in narrowed_focus.values())
        else "narrowed",
        "exhaustive": True,
        "trajectory_rows": total_rows,
        "surviving_count": len(surviving_f_to_q),
        "narrowed_focus_domains": narrowed_focus,
    }


def _solve_coupled_latent_exchange(
    step_tuples: list[tuple[list[float], list[float], list[float], list[float]]],
    reset_q_rows: list[list[float]],
    reset_x_rows: list[list[float]],
    reset_d_rows: list[list[float]],
    dynamic: list[str],
    x_fields: list[str],
    bijections: list[tuple[int, ...]],
    ordered_directions: list[str],
    eff_tol: float,
    budget: SearchBudget,
) -> tuple[dict[str, list[dict]], float | None]:
    """Identify spatial exchange accompanied by cross-object latent neighbor transport.

    ``step_tuples`` holds ``(q_v, x_v, z_prev_unit, td_v)`` where ``td_v`` has already
    been deflated by the local per-object latent term ``z_state[i]``.
    """
    from .synthesis import _solve

    n = len(dynamic)
    if not step_tuples or not bijections:
        return {}, None

    # Use reset rows (where z_prev_unit == 0) to prune (direction, q_to_x) fast if active!
    reset_active = any(max(abs(v) for v in dv) > max(1e-7, sqrt(eff_tol)) for dv in reset_d_rows)
    by_dir_out: dict[str, list[dict]] = {}
    eta_coupled_found: float | None = None

    for direction in ordered_directions:
        sign = 1.0 if direction == "right" else -1.0
        dir_matches: list[dict] = []
        for q_to_x in bijections:
            if not budget.spend(max(1, len(step_tuples) * 2)):
                return by_dir_out, eta_coupled_found
            r_bounds_init = None
            if reset_active and reset_q_rows:
                res_solve = _exact_interval_across_transitions(
                    reset_q_rows,
                    reset_x_rows,
                    reset_d_rows,
                    direction,
                    q_to_x,
                    eff_tol * 4.0,
                    edge_condition="unconditional",
                )
                if res_solve is None:
                    continue
                r_bounds_init = res_solve["radius_interval"]

            dists_set: set[float] = set()
            for q_v, x_v, _, _ in step_tuples:
                for s_i in range(n):
                    for t_i in range(n):
                        if s_i == t_i:
                            continue
                        dx = (
                            sign * (x_v[q_to_x[t_i]] - x_v[q_to_x[s_i]])
                            if direction in {"right", "left"}
                            else abs(x_v[q_to_x[t_i]] - x_v[q_to_x[s_i]])
                        )
                        if dx > 1e-9:
                            dists_set.add(round(dx, 9))
            sorted_d = sorted(dists_set)
            if not sorted_d:
                continue
            cuts = [0.0, *sorted_d, sorted_d[-1] + 0.5]
            valid_cells: list[tuple[float, float, float, float, float]] = []
            for c_idx in range(len(cuts) - 1):
                lo_c, hi_c = cuts[c_idx], cuts[c_idx + 1]
                if hi_c - lo_c <= 1e-9:
                    continue
                if r_bounds_init is not None and (hi_c <= r_bounds_init[0] + 1e-9 or lo_c >= r_bounds_init[1] - 1e-9):
                    continue
                r_mid = 0.5 * (lo_c + hi_c)
                if not budget.spend(len(step_tuples)):
                    break
                g00 = g01 = g11 = b0 = b1 = 0.0
                gq_vecs = []
                gz_vecs = []
                for q_v, x_v, z_v, td_v in step_tuples:
                    gq = [0.0] * n
                    gz = [0.0] * n
                    if direction in {"right", "left"}:
                        for s_i in range(n):
                            for t_i in range(n):
                                if s_i != t_i and 0.0 < sign * (x_v[q_to_x[t_i]] - x_v[q_to_x[s_i]]) < r_mid:
                                    gq[s_i] -= q_v[s_i]
                                    gq[t_i] += q_v[s_i]
                                    gz[s_i] -= z_v[s_i]
                                    gz[t_i] += z_v[s_i]
                    else:
                        for s_i in range(n):
                            for t_i in range(n):
                                if s_i != t_i and abs(x_v[q_to_x[t_i]] - x_v[q_to_x[s_i]]) < r_mid:
                                    gq[s_i] += q_v[t_i] - q_v[s_i]
                                    gz[s_i] += z_v[t_i] - z_v[s_i]
                    gq_vecs.append(gq)
                    gz_vecs.append(gz)
                    for i in range(n):
                        g00 += gq[i] * gq[i]
                        g01 += gq[i] * gz[i]
                        g11 += gz[i] * gz[i]
                        b0 += gq[i] * td_v[i]
                        b1 += gz[i] * td_v[i]
                if g00 <= 1e-14 or g11 <= 1e-14:
                    continue
                gamma_c, eta_c = _solve([[g00 + 1e-14, g01], [g01, g11 + 1e-14]], [b0, b1])
                if gamma_c <= 1e-4 or abs(eta_c) <= 1e-5:
                    continue
                err = 0.0
                for (q_v, x_v, z_v, td_v), gq, gz in zip(step_tuples, gq_vecs, gz_vecs):
                    for i in range(n):
                        diff = td_v[i] - (gamma_c * gq[i] + eta_c * gz[i])
                        err += diff * diff
                mse_cell = err / (len(step_tuples) * n)
                if mse_cell <= eff_tol * 4.0:
                    valid_cells.append((lo_c, hi_c, float(gamma_c), float(eta_c), float(mse_cell)))

            if valid_cells:
                r_lo = valid_cells[0][0]
                r_hi = valid_cells[-1][1]
                best_c = min(valid_cells, key=lambda item: item[4])
                if eta_coupled_found is None:
                    eta_coupled_found = best_c[3]
                dir_matches.append(
                    {
                        "direction": direction,
                        "edge_condition": "unconditional",
                        "mapping": {x_fields[x_idx]: dynamic[q_idx] for q_idx, x_idx in enumerate(q_to_x)},
                        "radius_interval": [float(r_lo), float(r_hi)],
                        "coupling_estimate": best_c[2],
                        "latent_coupling_estimate": best_c[3],
                        "mse": best_c[4],
                    }
                )
        if dir_matches:
            by_dir_out[direction] = dir_matches
    return by_dir_out, eta_coupled_found


def _extract_latent_trajectories_and_resets(
    valid_rows: list[tuple[int, dict]],
    observations: list | tuple = (),
    min_len: int = 3,
) -> tuple[list[list[tuple[int, dict]]], list[tuple[int, dict]]]:
    trajectories: list[list[tuple[int, dict]]] = []
    reset_rows: list[tuple[int, dict]] = []
    has_probe_meta = any(
        row.get("probe_id") is not None and row.get("position") is not None for _, row in valid_rows
    )
    if has_probe_meta:
        by_probe: dict[str, dict[int, tuple[int, dict]]] = {}
        for rec_idx, row in valid_rows:
            pid = str(row.get("probe_id", f"__anon_{rec_idx}"))
            pos = int(row.get("position", len(row.get("history", ()))))
            if pos == 0 and not row.get("history"):
                reset_rows.append((rec_idx, row))
            by_probe.setdefault(pid, {}).setdefault(pos, (rec_idx, row))
        for rows_by_pos in by_probe.values():
            if 0 not in rows_by_pos:
                continue
            seq = []
            step = 0
            while step in rows_by_pos:
                seq.append(rows_by_pos[step])
                step += 1
            if len(seq) >= min_len:
                trajectories.append(seq)
    elif len(observations) == len(valid_rows):
        cur_seq: list[tuple[int, dict]] = []
        for (rec_idx, row), obs in zip(valid_rows, observations):
            hlen = len(getattr(obs, "history", ()))
            if hlen == 0:
                reset_rows.append((rec_idx, row))
                if len(cur_seq) >= min_len:
                    trajectories.append(cur_seq)
                cur_seq = [(rec_idx, row)]
            elif len(cur_seq) == hlen:
                cur_seq.append((rec_idx, row))
            else:
                if len(cur_seq) >= min_len:
                    trajectories.append(cur_seq)
                cur_seq = []
        if len(cur_seq) >= min_len:
            trajectories.append(cur_seq)
    return trajectories, reset_rows


def _audit_latent_transition_state(
    valid_rows: list[tuple[int, dict]],
    dynamic: list[str],
    drive_by_q: dict[str, str],
    x_fields: list[str],
    bijections: list[tuple[int, ...]],
    ordered_directions: list[str],
    eff_tol: float,
    budget: SearchBudget,
    *,
    observations: list | tuple = (),
    allow_higher_order_latent: bool = True,
) -> dict | None:
    """Audit unobserved per-object latent transition states across order & spatial coupling.

    Tests hierarchically:
    1. 1st-order independent exponential reservoir:
       ``z_{i,t} = alpha * z_{i,t-1} + c_u * u_{i,t} + c_q * q_{i,t}``
    2. 1st-order spatially coupled latent reservoir (``P_latent_coupling``):
       ``after[i]`` receives both local ``z_{i,t-1}`` and neighbor flux ``Delta_z(t-1)_i``
    3. 2nd-order oscillatory / AR(2) internal resonator (``P_latent_order``):
       ``z_{i,t} = a_1 * z_{i,t-1} + a_2 * z_{i,t-2} + c_u * u_{i,t} + c_q * q_{i,t}``
    4. Certified ``unexplained_latent_transition_state`` (``P_latent_transition_drive``)
       when probe resets conserve ``sum_i delta_{i,0} = 0`` and satisfy spatial exchange,
       yet ``t >= 1`` violates conservation and refutes all linear reservoirs.
    """
    from .synthesis import _solve

    n = len(dynamic)
    if not valid_rows or not bijections:
        return None
    if not budget.spend(len(valid_rows) * (n + 4)):
        return None

    trajectories, reset_rows = _extract_latent_trajectories_and_resets(
        valid_rows, observations=observations, min_len=3
    )
    if not reset_rows or not trajectories:
        return None

    def _vecs(row: dict) -> tuple[list[float], list[float], list[float], list[float]]:
        before, after = row["before"], row["after"]
        q_v = [float(before[q]) for q in dynamic]
        u_v = [float(before[drive_by_q[q]]) for q in dynamic]
        x_v = [float(before[x]) for x in x_fields]
        d_v = [float(after[q]) - float(before[q]) - float(before[drive_by_q[q]]) for q in dynamic]
        return q_v, u_v, x_v, d_v

    max_reset_abs = max(abs(sum(_vecs(r)[3])) for _, r in reset_rows)
    if max_reset_abs > max(1e-6, sqrt(eff_tol) * n * 4.0):
        return None

    eq_rows: list[tuple[float, float, float, float]] = []
    eq2_rows: list[tuple[float, float, float, float, float]] = []
    for seq in trajectories:
        prev_q, prev_u, _, prev_d = _vecs(seq[0][1])
        prev_r = sum(prev_d)
        prev2_r = 0.0
        for t in range(1, len(seq)):
            cur_q, cur_u, _, cur_d = _vecs(seq[t][1])
            cur_r = sum(cur_d)
            eq_rows.append((prev_r, sum(prev_u), sum(prev_q), cur_r))
            eq2_rows.append((prev_r, prev2_r, sum(prev_u), sum(prev_q), cur_r))
            prev2_r = prev_r
            prev_q, prev_u, prev_r = cur_q, cur_u, cur_r

    if len(eq_rows) < 5:
        return None

    def _fit_homogeneous(rows_mat: list[tuple], col_indices: tuple[int, ...], rhs_idx: int) -> list[float]:
        k = len(col_indices)
        gram = [
            [sum(eq[i] * eq[j] for eq in rows_mat) for j in col_indices]
            for i in col_indices
        ]
        for idx_d in range(k):
            gram[idx_d][idx_d] += 1e-12
        rhs = [sum(eq[i] * eq[rhs_idx] for eq in rows_mat) for i in col_indices]
        return _solve(gram, rhs)

    # --- Attempt 1: 1st-Order Global Recurrence (Independent & Spatially Coupled) ---
    alpha, c_u, c_q = _fit_homogeneous(eq_rows, (0, 1, 2), 3)
    if abs(c_q) < 1e-6 and abs(c_u) >= 1e-6:
        alpha, c_u = _fit_homogeneous(eq_rows, (0, 1), 3)
        c_q = 0.0
    elif abs(c_u) < 1e-6 and abs(c_q) >= 1e-6:
        alpha, c_q = _fit_homogeneous(eq_rows, (0, 2), 3)
        c_u = 0.0

    rec_mse = (
        sum((eq[3] - (alpha * eq[0] + c_u * eq[1] + c_q * eq[2])) ** 2 for eq in eq_rows)
        / len(eq_rows)
        if (1e-4 < abs(alpha) < 0.9995 and max(abs(c_u), abs(c_q)) > 1e-5)
        else float("inf")
    )
    if rec_mse <= eff_tol * n * 8.0:
        def_q_rows: list[list[float]] = []
        def_x_rows: list[list[float]] = []
        def_d_rows: list[list[float]] = []
        seen_sigs: set[tuple] = set()
        covered_rec_ids: set[int] = set()
        step_tuples_for_coupled: list[tuple[list[float], list[float], list[float], list[float]]] = []
        drive_norm = c_u if abs(c_u) >= abs(c_q) and abs(c_u) > 1e-9 else (c_q if abs(c_q) > 1e-9 else 1.0)

        def _add_deflated(q_v: list[float], x_v: list[float], td_v: list[float]):
            sig = (
                tuple(round(v, 10) for v in q_v),
                tuple(round(v, 10) for v in x_v),
                tuple(round(v, 10) for v in td_v),
            )
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                def_q_rows.append(q_v)
                def_x_rows.append(x_v)
                def_d_rows.append(td_v)

        for seq in trajectories:
            z_state = [0.0] * n
            for rec_idx, row in seq:
                covered_rec_ids.add(rec_idx)
                q_v, u_v, x_v, d_v = _vecs(row)
                td_v = [d_v[i] - z_state[i] for i in range(n)]
                _add_deflated(q_v, x_v, td_v)
                step_tuples_for_coupled.append(
                    (q_v, x_v, [z_state[i] / drive_norm for i in range(n)], td_v)
                )
                z_state = [alpha * z_state[i] + c_u * u_v[i] + c_q * q_v[i] for i in range(n)]

        reset_q_only, reset_x_only, reset_d_only = [], [], []
        for rec_idx, row in reset_rows:
            q_v, _, x_v, d_v = _vecs(row)
            reset_q_only.append(q_v)
            reset_x_only.append(x_v)
            reset_d_only.append(d_v)
            if rec_idx not in covered_rec_ids:
                _add_deflated(q_v, x_v, d_v)

        if def_q_rows:
            deflated_cons_mse = sum(sum(td) ** 2 for td in def_d_rows) / len(def_d_rows)
            if deflated_cons_mse <= eff_tol * n * 8.0:
                active_threshold = max(1e-7, sqrt(eff_tol))
                active_taus = sorted(
                    (
                        tau
                        for tau, td in enumerate(def_d_rows)
                        if max(abs(v) for v in td) > active_threshold
                    ),
                    key=lambda tau: sum(v * v for v in def_d_rows[tau]),
                    reverse=True,
                )
                if active_taus:
                    matches_by_cond: dict[str, dict[str, list[dict]]] = {}
                    for cond_name in ("unconditional", "source_greater", "source_less"):
                        by_dir, exhausted = _solve_exchange_for_condition(
                            def_q_rows,
                            def_x_rows,
                            def_d_rows,
                            dynamic,
                            x_fields,
                            bijections,
                            ordered_directions,
                            active_taus,
                            eff_tol * 4.0,
                            budget,
                            edge_condition=cond_name,
                        )
                        if exhausted:
                            break
                        if by_dir:
                            matches_by_cond[cond_name] = by_dir
                            if cond_name == "unconditional":
                                break
                    if matches_by_cond:
                        return {
                            "mechanism": "transition_memory",
                            "latent_order": 1,
                            "latent_coupling": "independent",
                            "pole": float(alpha),
                            "drive_weights": {"u": float(c_u), "q": float(c_q)},
                            "recurrence_mse": float(rec_mse),
                            "deflated_conservation_mse": float(deflated_cons_mse),
                            "deflated_unique_transitions": len(def_q_rows),
                            "matches_by_cond": matches_by_cond,
                        }
                    if allow_higher_order_latent:
                        coupled_by_dir, eta_coupled = _solve_coupled_latent_exchange(
                            step_tuples_for_coupled,
                            reset_q_only,
                            reset_x_only,
                            reset_d_only,
                            dynamic,
                            x_fields,
                            bijections,
                            ordered_directions,
                            eff_tol,
                            budget,
                        )
                        if coupled_by_dir:
                            return {
                                "mechanism": "coupled_transition_memory",
                                "latent_order": 1,
                                "latent_coupling": "spatial_neighbor_coupled",
                                "pole": float(alpha),
                                "latent_coupling_rate": float(eta_coupled or 0.0),
                                "drive_weights": {"u": float(c_u), "q": float(c_q)},
                                "recurrence_mse": float(rec_mse),
                                "deflated_conservation_mse": float(deflated_cons_mse),
                                "deflated_unique_transitions": len(def_q_rows),
                                "matches_by_cond": {"unconditional": coupled_by_dir},
                            }

    # --- Attempt 2: 2nd-Order Oscillatory / AR(2) Internal Resonator ---
    if allow_higher_order_latent and len(eq2_rows) >= 6:
        a1, a2, c2_u, c2_q = _fit_homogeneous(eq2_rows, (0, 1, 2, 3), 4)
        if abs(c2_q) < 1e-6 and abs(c2_u) >= 1e-6:
            a1, a2, c2_u = _fit_homogeneous(eq2_rows, (0, 1, 2), 4)
            c2_q = 0.0
        elif abs(c2_u) < 1e-6 and abs(c2_q) >= 1e-6:
            a1, a2, c2_q = _fit_homogeneous(eq2_rows, (0, 1, 3), 4)
            c2_u = 0.0

        schur_stable = (
            abs(a2) < 0.999
            and abs(a2) > 1e-4
            and (1.0 - a1 - a2) > 1e-4
            and (1.0 + a1 - a2) > 1e-4
            and max(abs(c2_u), abs(c2_q)) > 1e-5
        )
        if schur_stable:
            rec2_mse = sum(
                (eq[4] - (a1 * eq[0] + a2 * eq[1] + c2_u * eq[2] + c2_q * eq[3])) ** 2
                for eq in eq2_rows
            ) / len(eq2_rows)
            if rec2_mse <= eff_tol * n * 12.0:
                def2_q_rows: list[list[float]] = []
                def2_x_rows: list[list[float]] = []
                def2_d_rows: list[list[float]] = []
                seen2_sigs: set[tuple] = set()
                covered2_ids: set[int] = set()

                def _add_def2(q_v: list[float], x_v: list[float], td_v: list[float]):
                    sig = (
                        tuple(round(v, 10) for v in q_v),
                        tuple(round(v, 10) for v in x_v),
                        tuple(round(v, 10) for v in td_v),
                    )
                    if sig not in seen2_sigs:
                        seen2_sigs.add(sig)
                        def2_q_rows.append(q_v)
                        def2_x_rows.append(x_v)
                        def2_d_rows.append(td_v)

                for seq in trajectories:
                    z_cur = [0.0] * n
                    z_prev = [0.0] * n
                    for rec_idx, row in seq:
                        covered2_ids.add(rec_idx)
                        q_v, u_v, x_v, d_v = _vecs(row)
                        td_v = [d_v[i] - z_cur[i] for i in range(n)]
                        _add_def2(q_v, x_v, td_v)
                        z_nxt = [
                            a1 * z_cur[i] + a2 * z_prev[i] + c2_u * u_v[i] + c2_q * q_v[i]
                            for i in range(n)
                        ]
                        z_prev, z_cur = z_cur, z_nxt

                for rec_idx, row in reset_rows:
                    if rec_idx not in covered2_ids:
                        q_v, _, x_v, d_v = _vecs(row)
                        _add_def2(q_v, x_v, d_v)

                if def2_q_rows:
                    def2_cons_mse = sum(sum(td) ** 2 for td in def2_d_rows) / len(def2_d_rows)
                    if def2_cons_mse <= eff_tol * n * 8.0:
                        act_thresh2 = max(1e-7, sqrt(eff_tol))
                        act_taus2 = sorted(
                            (
                                tau
                                for tau, td in enumerate(def2_d_rows)
                                if max(abs(v) for v in td) > act_thresh2
                            ),
                            key=lambda tau: sum(v * v for v in def2_d_rows[tau]),
                            reverse=True,
                        )
                        if act_taus2:
                            matches2_by_cond: dict[str, dict[str, list[dict]]] = {}
                            for cond_name in ("unconditional", "source_greater", "source_less"):
                                by_dir2, exh2 = _solve_exchange_for_condition(
                                    def2_q_rows,
                                    def2_x_rows,
                                    def2_d_rows,
                                    dynamic,
                                    x_fields,
                                    bijections,
                                    ordered_directions,
                                    act_taus2,
                                    eff_tol * 4.0,
                                    budget,
                                    edge_condition=cond_name,
                                )
                                if exh2:
                                    break
                                if by_dir2:
                                    matches2_by_cond[cond_name] = by_dir2
                                    if cond_name == "unconditional":
                                        break
                            if matches2_by_cond:
                                return {
                                    "mechanism": "oscillatory_transition_memory",
                                    "latent_order": 2,
                                    "latent_coupling": "independent",
                                    "ar2_params": [float(a1), float(a2)],
                                    "discriminant": float(a1 * a1 + 4.0 * a2),
                                    "pole": float(sqrt(abs(a2))),
                                    "drive_weights": {"u": float(c2_u), "q": float(c2_q)},
                                    "recurrence_mse": float(rec2_mse),
                                    "deflated_conservation_mse": float(def2_cons_mse),
                                    "deflated_unique_transitions": len(def2_q_rows),
                                    "matches_by_cond": matches2_by_cond,
                                }

    # --- Attempt 3: Certified Unexplained Nonlinear Latent Transition State ---
    # Reset transitions (t == 0) satisfy conservation (max_reset_abs ~ 0), while t >= 1
    # violates conservation AND refutes static/1st-order/coupled/2nd-order linear drives!
    reset_q_r, reset_x_r, reset_d_r = [], [], []
    for _, row in reset_rows:
        q_v, _, x_v, d_v = _vecs(row)
        reset_q_r.append(q_v)
        reset_x_r.append(x_v)
        reset_d_r.append(d_v)
    act_reset_taus = sorted(
        (
            tau
            for tau, dv in enumerate(reset_d_r)
            if max(abs(v) for v in dv) > max(1e-7, sqrt(eff_tol))
        ),
        key=lambda tau: sum(v * v for v in reset_d_r[tau]),
        reverse=True,
    )
    reset_by_dir: dict[str, list[dict]] = {}
    if act_reset_taus:
        reset_by_dir, _ = _solve_exchange_for_condition(
            reset_q_r,
            reset_x_r,
            reset_d_r,
            dynamic,
            x_fields,
            bijections,
            ordered_directions,
            act_reset_taus,
            eff_tol * 4.0,
            budget,
            edge_condition="unconditional",
        )
    if reset_by_dir:
        rep_dir = next(iter(reset_by_dir))
        rep_match = reset_by_dir[rep_dir][0]
        q_idx_map = {q_n: idx_q for idx_q, q_n in enumerate(dynamic)}
        x_idx_map = {x_n: idx_x for idx_x, x_n in enumerate(x_fields)}
        rep_q_to_x = [0] * n
        for xf, qf in rep_match["mapping"].items():
            rep_q_to_x[q_idx_map[qf]] = x_idx_map[xf]
        r_mid_res = 0.5 * (rep_match["radius_interval"][0] + rep_match["radius_interval"][1])
        gamma_res = float(rep_match["coupling_estimate"])
        sign_res = 1.0 if rep_dir == "right" else -1.0

        trace_samples = []
        for tr_idx, seq in enumerate(trajectories[:3]):
            steps_info = []
            for step_t, (_, row) in enumerate(seq[:6]):
                q_v, u_v, x_v, d_v = _vecs(row)
                ex_v = [0.0] * n
                if rep_dir in {"right", "left"}:
                    for s_i in range(n):
                        for t_i in range(n):
                            if s_i != t_i and 0.0 < sign_res * (x_v[rep_q_to_x[t_i]] - x_v[rep_q_to_x[s_i]]) < r_mid_res:
                                ex_v[s_i] -= gamma_res * q_v[s_i]
                                ex_v[t_i] += gamma_res * q_v[s_i]
                else:
                    for s_i in range(n):
                        for t_i in range(n):
                            if s_i != t_i and abs(x_v[rep_q_to_x[t_i]] - x_v[rep_q_to_x[s_i]]) < r_mid_res:
                                ex_v[s_i] += gamma_res * (q_v[t_i] - q_v[s_i])
                lat_inj = [round(d_v[i] - ex_v[i], 6) for i in range(n)]
                steps_info.append(
                    {
                        "step": step_t,
                        "q": [round(v, 5) for v in q_v],
                        "u": [round(v, 5) for v in u_v],
                        "x": [round(x_v[rep_q_to_x[i]], 5) for i in range(n)],
                        "sum_delta_total": round(sum(d_v), 6),
                        "deflated_latent_injection_per_object": lat_inj,
                    }
                )
            trace_samples.append({"trajectory_index": tr_idx, "steps": steps_info})

        return {
            "unexplained_nonlinear_latent": True,
            "reset_exchange_by_direction": reset_by_dir,
            "representative_direction": rep_dir,
            "representative_mapping": rep_match["mapping"],
            "representative_radius_interval": rep_match["radius_interval"],
            "representative_coupling": gamma_res,
            "deflated_trajectory_samples": trace_samples,
        }

    return None


def screen_custom_latent_transition_drive(
    valid_rows: list[tuple[int, dict]],
    dynamic: list[str],
    drive_by_q: dict[str, str],
    x_fields: list[str],
    bijections: list[tuple[int, ...]],
    drive_program: Program,
    ordered_directions: tuple[str, ...] | list[str] = ("right", "left", "symmetric"),
    *,
    budget: SearchBudget,
    eff_tol: float,
    observations: list | tuple = (),
) -> dict:
    """Verify an open-grammar per-object nonlinear latent transition drive AST on paid transitions."""
    from .synthesis import _solve

    n = len(dynamic)
    if not valid_rows or not bijections:
        return {"supported": False, "matches": []}
    trajectories, reset_rows = _extract_latent_trajectories_and_resets(
        valid_rows, observations=observations, min_len=3
    )
    if not trajectories or not reset_rows:
        return {"supported": False, "matches": []}
    if not budget.spend(len(valid_rows) * (n + 4)):
        return {"supported": False, "matches": [], "reason": "compute_exhausted"}

    def _vecs(row: dict) -> tuple[list[float], list[float], list[float], list[float]]:
        before, after = row["before"], row["after"]
        q_v = [float(before[q]) for q in dynamic]
        u_v = [float(before[drive_by_q[q]]) for q in dynamic]
        x_v = [float(before[x]) for x in x_fields]
        d_v = [float(after[q]) - float(before[q]) - float(before[drive_by_q[q]]) for q in dynamic]
        return q_v, u_v, x_v, d_v

    uses_x = "object_role:x" in drive_program.required_features()
    bij_candidates = bijections if uses_x else [bijections[0]]

    for q_to_x_seed in bij_candidates:
        eq_custom: list[tuple[float, float, float]] = []
        seq_drives: list[list[list[float]]] = []
        valid_eval = True
        for seq in trajectories:
            d_seq: list[list[float]] = []
            for _, row in seq:
                q_v, u_v, x_v, _ = _vecs(row)
                step_d = []
                for i in range(n):
                    feat_i = {
                        "object_role:q": q_v[i],
                        "object_role:u": u_v[i],
                        "object_role:x": x_v[q_to_x_seed[i]],
                    }
                    val = float(drive_program.predict(Observation(feat_i, 0.0)))
                    if not isfinite(val):
                        valid_eval = False
                        break
                    step_d.append(val)
                if not valid_eval:
                    break
                d_seq.append(step_d)
            if not valid_eval:
                break
            seq_drives.append(d_seq)
            prev_r = sum(_vecs(seq[0][1])[3])
            for t in range(1, len(seq)):
                cur_r = sum(_vecs(seq[t][1])[3])
                eq_custom.append((prev_r, sum(d_seq[t - 1]), cur_r))
                prev_r = cur_r
        if not valid_eval or len(eq_custom) < 4:
            continue

        g00 = sum(r[0] * r[0] for r in eq_custom) + 1e-14
        g01 = sum(r[0] * r[1] for r in eq_custom)
        g11 = sum(r[1] * r[1] for r in eq_custom) + 1e-14
        b0 = sum(r[0] * r[2] for r in eq_custom)
        b1 = sum(r[1] * r[2] for r in eq_custom)
        alpha_est, eta_est = _solve([[g00, g01], [g01, g11]], [b0, b1])
        if not (1e-4 < abs(alpha_est) < 0.999) or abs(eta_est) <= 1e-6:
            continue
        rec_mse = sum(
            (r[2] - (alpha_est * r[0] + eta_est * r[1])) ** 2 for r in eq_custom
        ) / len(eq_custom)
        if rec_mse > eff_tol * n * 12.0:
            continue

        def_q, def_x, def_d = [], [], []
        seen_s: set[tuple] = set()
        cov_ids: set[int] = set()

        def _push(qv, xv, tdv):
            sg = (
                tuple(round(v, 10) for v in qv),
                tuple(round(v, 10) for v in xv),
                tuple(round(v, 10) for v in tdv),
            )
            if sg not in seen_s:
                seen_s.add(sg)
                def_q.append(qv)
                def_x.append(xv)
                def_d.append(tdv)

        for seq, d_seq in zip(trajectories, seq_drives):
            z_st = [0.0] * n
            for (rec_idx, row), step_d in zip(seq, d_seq):
                cov_ids.add(rec_idx)
                qv, _, xv, dv = _vecs(row)
                tdv = [dv[i] - z_st[i] for i in range(n)]
                _push(qv, xv, tdv)
                z_st = [alpha_est * z_st[i] + eta_est * step_d[i] for i in range(n)]
        for rec_idx, row in reset_rows:
            if rec_idx not in cov_ids:
                qv, _, xv, dv = _vecs(row)
                _push(qv, xv, dv)

        def_cons_mse = sum(sum(td) ** 2 for td in def_d) / max(1, len(def_d))
        if def_cons_mse > eff_tol * n * 8.0:
            continue

        act_taus = sorted(
            (
                tau
                for tau, td in enumerate(def_d)
                if max(abs(v) for v in td) > max(1e-7, sqrt(eff_tol))
            ),
            key=lambda tau: sum(v * v for v in def_d[tau]),
            reverse=True,
        )
        if not act_taus:
            continue
        test_bijs = [q_to_x_seed] if uses_x else bijections
        by_dir, _ = _solve_exchange_for_condition(
            def_q,
            def_x,
            def_d,
            dynamic,
            x_fields,
            test_bijs,
            list(ordered_directions),
            act_taus,
            eff_tol * 4.0,
            budget,
            edge_condition="unconditional",
        )
        if by_dir:
            flat_m = [m for items in by_dir.values() for m in items]
            return {
                "supported": True,
                "verified_pole": float(alpha_est),
                "latent_weight_estimate": float(eta_est),
                "recurrence_mse": float(rec_mse),
                "deflated_conservation_mse": float(def_cons_mse),
                "matches": flat_m,
                "matches_by_direction": by_dir,
            }

    return {
        "supported": False,
        "matches": [],
        "reason": "failed_latent_transition_deflation",
    }


def _recover_latent_drive_and_position_roles(
    model: dict,
    records: list[dict],
    budget: SearchBudget,
    *,
    allow_higher_order_latent: bool = True,
) -> tuple[dict[str, str], dict[str, dict]] | None:
    """Identify ``drive_by_q`` and ``position_candidates`` when latent memory carries across steps.

    Supports both 1st-order exponential reservoirs and 2nd-order oscillatory AR(2) resonators
    on ``q == 0`` sweeps.
    """
    dynamic = list(model.get("dynamic_fields", ()))
    role_cands = model.get("role_candidates", {})
    n = len(dynamic)
    if n < 2 or len(role_cands) != 2 * n or not records:
        return None
    stable_fields = sorted(role_cands.keys())
    if not budget.spend(len(records) * len(stable_fields) * 2):
        return None

    by_probe: dict[str, dict[int, dict]] = {}
    for idx, row in enumerate(records):
        if "before" not in row or "after" not in row:
            continue
        pid = str(row.get("probe_id", f"__anon_{idx}"))
        pos = int(row.get("position", len(row.get("history", ()))))
        by_probe.setdefault(pid, {}).setdefault(pos, row)

    q_zero_eval: list[tuple[list[dict], int]] = []
    probe_trajectories: list[list[dict]] = []
    for rows_by_pos in by_probe.values():
        if 0 not in rows_by_pos:
            continue
        seq: list[dict] = []
        step = 0
        while step in rows_by_pos:
            cur_r = rows_by_pos[step]
            bef = cur_r["before"]
            if not all(f in bef for f in (*dynamic, *stable_fields)):
                break
            seq.append(cur_r)
            if all(abs(float(bef[q])) <= 1e-9 for q in dynamic):
                q_zero_eval.append((seq, step))
            step += 1
        if seq:
            probe_trajectories.append(seq)

    if len(q_zero_eval) < 2 * n:
        return None

    drive_by_q: dict[str, str] = {}
    gr = (sqrt(5.0) - 1.0) / 2.0
    for q_name in dynamic:
        y_after = [float(seq[t]["after"][q_name]) for seq, t in q_zero_eval]
        if max(abs(v) for v in y_after) <= 1e-8:
            return None
        matches_s: list[str] = []
        for s_name in stable_fields:
            s_cur = [float(seq[t]["before"][s_name]) for seq, t in q_zero_eval]
            if max(abs(v) for v in s_cur) <= 1e-8:
                continue
            residuals = [y_after[idx] - s_cur[idx] for idx in range(len(y_after))]
            if sum(r * r for r in residuals) / len(residuals) <= 1e-12:
                matches_s.append(s_name)
                continue

            def _pole_mse(alpha_val: float) -> float:
                zs_by_id: dict[int, list[float]] = {}
                zq_by_id: dict[int, list[float]] = {}
                for seq in probe_trajectories:
                    zs_acc = 0.0
                    zq_acc = 0.0
                    zs_list = []
                    zq_list = []
                    for r_step in seq:
                        zs_list.append(zs_acc)
                        zq_list.append(zq_acc)
                        zs_acc = alpha_val * zs_acc + float(r_step["before"][s_name])
                        zq_acc = alpha_val * zq_acc + float(r_step["before"][q_name])
                    zs_by_id[id(seq)] = zs_list
                    zq_by_id[id(seq)] = zq_list
                c_zs = [zs_by_id[id(seq)][t] for seq, t in q_zero_eval]
                c_zq = [zq_by_id[id(seq)][t] for seq, t in q_zero_eval]
                g00 = sum(v * v for v in c_zs) + 1e-15
                g01 = sum(a * b for a, b in zip(c_zs, c_zq))
                g11 = sum(v * v for v in c_zq) + 1e-15
                b0 = sum(a * r for a, r in zip(c_zs, residuals))
                b1 = sum(a * r for a, r in zip(c_zq, residuals))
                det = g00 * g11 - g01 * g01
                if abs(det) <= 1e-24:
                    w0 = b0 / g00
                    w1 = 0.0
                else:
                    w0 = (g11 * b0 - g01 * b1) / det
                    w1 = (g00 * b1 - g01 * b0) / det
                return sum(
                    (residuals[i] - (w0 * c_zs[i] + w1 * c_zq[i])) ** 2
                    for i in range(len(residuals))
                ) / len(residuals)

            coarse = [(p, _pole_mse(p)) for p in (-0.6, -0.25, 0.25, 0.5, 0.68, 0.82, 0.93)]
            best_p, best_err = min(coarse, key=lambda item: item[1])
            if best_err <= 5e-2:
                lo_p = max(-0.96, best_p - 0.22)
                hi_p = min(0.985, best_p + 0.20)
                c1 = hi_p - gr * (hi_p - lo_p)
                c2 = lo_p + gr * (hi_p - lo_p)
                f1, f2 = _pole_mse(c1), _pole_mse(c2)
                for _ in range(22):
                    if min(f1, f2) <= 1e-16:
                        break
                    if f1 < f2:
                        hi_p, c2, f2 = c2, c1, f1
                        c1 = hi_p - gr * (hi_p - lo_p)
                        f1 = _pole_mse(c1)
                    else:
                        lo_p, c1, f1 = c1, c2, f2
                        c2 = lo_p + gr * (hi_p - lo_p)
                        f2 = _pole_mse(c2)
                best_err = min(best_err, f1, f2)
            if best_err <= 1e-11:
                matches_s.append(s_name)
                continue

            # 2nd-order oscillatory AR(2) role recovery fallback
            if allow_higher_order_latent:
                def _ar2_role_mse(a1_v: float, a2_v: float) -> float:
                    zs_by_id: dict[int, list[float]] = {}
                    for seq in probe_trajectories:
                        zs_cur, zs_prev = 0.0, 0.0
                        zs_list = []
                        for r_step in seq:
                            zs_list.append(zs_cur)
                            zs_nxt = a1_v * zs_cur + a2_v * zs_prev + float(r_step["before"][s_name])
                            zs_prev, zs_cur = zs_cur, zs_nxt
                        zs_by_id[id(seq)] = zs_list
                    c_zs = [zs_by_id[id(seq)][t] for seq, t in q_zero_eval]
                    g00 = sum(v * v for v in c_zs) + 1e-15
                    b0 = sum(a * r for a, r in zip(c_zs, residuals))
                    w0 = b0 / g00
                    return sum((residuals[i] - w0 * c_zs[i]) ** 2 for i in range(len(residuals))) / len(residuals)

                seeds2_polar = [
                    (rho, cw)
                    for rho in (0.50, 0.64, 0.76, 0.87)
                    for cw in (-0.25, 0.18, 0.56, 0.84)
                ]
                brho, bcw, berr2 = min(
                    (
                        (r_c, cw_c, _ar2_role_mse(2.0 * r_c * cw_c, -(r_c * r_c)))
                        for r_c, cw_c in seeds2_polar
                    ),
                    key=lambda item: item[2],
                )
                if berr2 <= 5e-2:
                    sp_r, sp_w = 0.16, 0.28
                    for _ in range(7):
                        if berr2 <= 1e-16:
                            break
                        lo_r, hi_r = max(0.08, brho - sp_r), min(0.985, brho + sp_r)
                        c1 = hi_r - gr * (hi_r - lo_r)
                        c2 = lo_r + gr * (hi_r - lo_r)
                        f1 = _ar2_role_mse(2.0 * c1 * bcw, -(c1 * c1))
                        f2 = _ar2_role_mse(2.0 * c2 * bcw, -(c2 * c2))
                        for _ in range(12):
                            if f1 < f2:
                                hi_r, c2, f2 = c2, c1, f1
                                c1 = hi_r - gr * (hi_r - lo_r)
                                f1 = _ar2_role_mse(2.0 * c1 * bcw, -(c1 * c1))
                            else:
                                lo_r, c1, f1 = c1, c2, f2
                                c2 = lo_r + gr * (hi_r - lo_r)
                                f2 = _ar2_role_mse(2.0 * c2 * bcw, -(c2 * c2))
                        if f1 < f2 and f1 < berr2:
                            brho, berr2 = c1, f1
                        elif f2 <= f1 and f2 < berr2:
                            brho, berr2 = c2, f2

                        lo_w, hi_w = max(-0.985, bcw - sp_w), min(0.985, bcw + sp_w)
                        d1 = hi_w - gr * (hi_w - lo_w)
                        d2 = lo_w + gr * (hi_w - lo_w)
                        a2_f = -(brho * brho)
                        g1 = _ar2_role_mse(2.0 * brho * d1, a2_f)
                        g2 = _ar2_role_mse(2.0 * brho * d2, a2_f)
                        for _ in range(12):
                            if g1 < g2:
                                hi_w, d2, g2 = d2, d1, g1
                                d1 = hi_w - gr * (hi_w - lo_w)
                                g1 = _ar2_role_mse(2.0 * brho * d1, a2_f)
                            else:
                                lo_w, d1, g1 = d1, d2, g2
                                d2 = lo_w + gr * (hi_w - lo_w)
                                g2 = _ar2_role_mse(2.0 * brho * d2, a2_f)
                        if g1 < g2 and g1 < berr2:
                            bcw, berr2 = d1, g1
                        elif g2 <= g1 and g2 < berr2:
                            bcw, berr2 = d2, g2
                        sp_r *= 0.52
                        sp_w *= 0.52
                if berr2 <= 1e-8:
                    matches_s.append(s_name)
        if len(matches_s) != 1:
            return None
        drive_by_q[q_name] = matches_s[0]

    if len(set(drive_by_q.values())) != n:
        return None

    used_drives = set(drive_by_q.values())
    pos_cands = {
        s_name: {
            "control": role_cands[s_name]["control"],
            "quantities": list(dynamic),
            "support_pairs": role_cands[s_name].get("support_pairs", 0),
        }
        for s_name in stable_fields
        if s_name not in used_drives
    }
    if len(pos_cands) != n:
        return None
    return drive_by_q, pos_cands


def audit_transition_premises(
    model: dict | None,
    records: list[dict],
    budget: SearchBudget,
    *,
    commitments: list[dict] | tuple[dict, ...] = (),
    tolerance: float = 1e-8,
    max_bijections: int = 720,
    allow_expanded_revisions: bool = True,
    allow_latent_state: bool = True,
    allow_higher_order_latent: bool = True,
    observations: list | tuple = (),
) -> dict:
    """Audit support premises on paid ``before -> after`` state transitions."""
    start = budget.spent
    primary_commitment = commitments[0] if commitments else None
    graph = build_premise_dependency_graph(primary_commitment, model)

    def _set_node(node_id: str, status: str, **extra):
        for node in graph["nodes"]:
            if node["id"] == node_id:
                node["status"] = status
                node.update(extra)
                break

    if not model or not records:
        return {
            "status": "awaiting_role_identification",
            "dependency_graph": graph,
            "supported_premises": [],
            "refuted_premises": [],
            "scientific_revision_triggers": [],
            "compute_cost": 0,
        }

    dynamic = list(model.get("dynamic_fields", ()))
    positions = model.get("position_candidates", {})
    drive_by_q = {}
    for group in model.get("groups", ()):
        if group.get("q") and group.get("u"):
            drive_by_q[group["q"]] = group["u"]
    if len(drive_by_q) < len(dynamic):
        for link in model.get("drive_links", {}).values():
            if link.get("quantity") and link.get("field"):
                drive_by_q.setdefault(link["quantity"], link["field"])

    n = len(dynamic)
    recovered_roles: tuple[dict[str, str], dict[str, dict]] | None = None
    if (
        allow_latent_state
        and n >= 2
        and (len(drive_by_q) != n or len(set(drive_by_q.values())) != n or len(positions) != n)
    ):
        recovered_roles = _recover_latent_drive_and_position_roles(
            model, records, budget, allow_higher_order_latent=allow_higher_order_latent
        )
        if recovered_roles is not None:
            drive_by_q, positions = recovered_roles
    x_fields = sorted(positions.keys())
    if n < 2 or len(drive_by_q) != n or len(set(drive_by_q.values())) != n or len(x_fields) != n:
        return {
            "status": "awaiting_role_identification",
            "dependency_graph": graph,
            "supported_premises": [],
            "refuted_premises": [],
            "scientific_revision_triggers": [],
            "compute_cost": budget.spent - start,
        }

    _set_node("P_role_partition", "supported")
    required_before = {*dynamic, *drive_by_q.values(), *x_fields}
    valid_rows = [
        (idx, row)
        for idx, row in enumerate(records)
        if required_before <= set(row.get("before", ()))
        and set(dynamic) <= set(row.get("after", ()))
    ]
    if not valid_rows:
        return {
            "status": "no_transition_rows",
            "dependency_graph": graph,
            "supported_premises": ["P_role_partition"],
            "refuted_premises": [],
            "scientific_revision_triggers": [],
            "compute_cost": 0,
        }

    if not budget.spend(len(valid_rows) * (3 * n + 2)):
        return {
            "status": "premise_audit_compute_exhausted",
            "dependency_graph": graph,
            "supported_premises": ["P_role_partition"],
            "refuted_premises": [],
            "scientific_revision_triggers": [],
            "compute_cost": budget.spent - start,
        }

    all_deltas = []
    unique_map = {}
    q_rows, x_rows, delta_rows, row_refs = [], [], [], []
    for rec_idx, row in valid_rows:
        before, after = row["before"], row["after"]
        q_vec = [float(before[q]) for q in dynamic]
        x_vec = [float(before[x]) for x in x_fields]
        d_vec = [
            float(after[q]) - float(before[q]) - float(before[drive_by_q[q]])
            for q in dynamic
        ]
        all_deltas.append((rec_idx, row, d_vec))
        sig = (
            tuple(round(v, 12) for v in q_vec),
            tuple(round(v, 12) for v in x_vec),
            tuple(round(v, 12) for v in d_vec),
        )
        if sig not in unique_map:
            unique_map[sig] = len(q_rows)
            q_rows.append(q_vec)
            x_rows.append(x_vec)
            delta_rows.append(d_vec)
            row_refs.append(
                {
                    "record_index": rec_idx,
                    "probe_id": row.get("probe_id"),
                    "position": row.get("position"),
                }
            )

    scale = max(
        1.0,
        sum(val * val for _, _, d_vec in all_deltas for val in d_vec) / (len(all_deltas) * n),
    )
    eff_tol = max(1e-12, tolerance * scale)

    # Enumerate admissible position bijections once for downstream checks.
    bijections, exhaustive = _enumerate_position_bijections(
        dynamic, x_fields, positions, max_bijections=max_bijections
    )

    committed_specs = []
    for comm in commitments:
        noncons = comm.get("nonconservation_mechanism")
        for param in comm.get("radius_parameters", ()):
            cond_id = describe_edge_condition(param.get("edge_condition"))
            spec_tuple = (param["direction"], cond_id, noncons)
            if spec_tuple not in committed_specs:
                committed_specs.append(spec_tuple)

    committed_directions = []
    for d_val, _, _ in committed_specs:
        if d_val not in committed_directions:
            committed_directions.append(d_val)
    ordered_directions = [
        *committed_directions,
        *(d for d in ("symmetric", "right", "left") if d not in committed_directions),
    ]

    # 1. Test Conservation Premise: sum_i delta_{i,tau} == 0
    sums = [(rec_idx, row, sum(d_vec)) for rec_idx, row, d_vec in all_deltas]
    conservation_mse = sum(val * val for _, _, val in sums) / len(sums)
    supported_premises = ["P_role_partition"]
    refuted_premises = []
    scientific_triggers = []

    if conservation_mse > eff_tol:
        worst_idx, worst_row, worst_sum = max(sums, key=lambda item: abs(item[2]))
        witness = {
            "record_index": worst_idx,
            "probe_id": worst_row.get("probe_id"),
            "position": worst_row.get("position"),
            "sum_delta": float(worst_sum),
            "conservation_mse": float(conservation_mse),
        }
        _set_node("P_conservation", "refuted", counterexample=witness)
        refuted_premises.append(
            {
                "premise_id": "P_conservation",
                "reason": "non_zero_total_state_residual",
                "counterexample": witness,
            }
        )

        # Phase 2A Non-Conservative De-Coupling Audit:
        # Test whether SumDelta(tau) = kappa * sum_j q_{j,tau} holds across all transitions,
        # and distinguish local dissipation ('sink': kappa * q_i) vs global coupling
        # ('nonlocal': (kappa / N) * sum_j q_j).
        s_qq = 0.0
        s_qd = 0.0
        q_sums = [sum(q_vec) for q_vec in q_rows]
        d_sums = [sum(d_vec) for d_vec in delta_rows]
        for sq, sd in zip(q_sums, d_sums):
            s_qq += sq * sq
            s_qd += sq * sd

        noncons_matches_by_mech: dict[str, dict[str, dict[str, list[dict]]]] = {}
        kappa = (s_qd / s_qq) if s_qq > 1e-18 else 0.0
        total_linear_mse = (
            sum((sd - kappa * sq) ** 2 for sq, sd in zip(q_sums, d_sums)) / len(q_rows)
            if s_qq > 1e-18
            else float("inf")
        )
        if (
            allow_expanded_revisions
            and abs(kappa) > 1e-9
            and total_linear_mse <= eff_tol * n
            and bijections
        ):
            candidate_mechs = [
                (
                    "sink",
                    float(kappa),
                    [
                        [d_vec[i] - kappa * q_vec[i] for i in range(n)]
                        for q_vec, d_vec in zip(q_rows, delta_rows)
                    ],
                ),
                (
                    "nonlocal",
                    float(kappa / n),
                    [
                        [d_vec[i] - (kappa / n) * sq for i in range(n)]
                        for sq, d_vec in zip(q_sums, delta_rows)
                    ],
                ),
            ]
            active_threshold = max(1e-7, sqrt(eff_tol))
            for mech_name, mech_rate, residual_deltas in candidate_mechs:
                active_taus_nc = sorted(
                    (
                        tau
                        for tau, d_vec in enumerate(residual_deltas)
                        if max(abs(val) for val in d_vec) > active_threshold
                    ),
                    key=lambda tau: sum(val * val for val in residual_deltas[tau]),
                    reverse=True,
                )
                if not active_taus_nc:
                    continue
                for cond_name in ("unconditional", "source_greater", "source_less"):
                    by_dir, exhausted = _solve_exchange_for_condition(
                        q_rows,
                        x_rows,
                        residual_deltas,
                        dynamic,
                        x_fields,
                        bijections,
                        ordered_directions,
                        active_taus_nc,
                        eff_tol,
                        budget,
                        edge_condition=cond_name,
                    )
                    if exhausted:
                        break
                    if by_dir:
                        for items in by_dir.values():
                            for item in items:
                                item["nonconservation_mechanism"] = mech_name
                                item["nonconservation_rate"] = mech_rate
                        noncons_matches_by_mech.setdefault(mech_name, {})[cond_name] = by_dir
                        if cond_name == "unconditional":
                            break

        flat_nc_matches = [
            m
            for by_cond in noncons_matches_by_mech.values()
            for by_dir in by_cond.values()
            for items in by_dir.values()
            for m in items
        ]
        if not flat_nc_matches:
            latent_res = (
                _audit_latent_transition_state(
                    valid_rows,
                    dynamic,
                    drive_by_q,
                    x_fields,
                    bijections,
                    ordered_directions,
                    eff_tol,
                    budget,
                    observations=observations,
                    allow_higher_order_latent=allow_higher_order_latent,
                )
                if allow_expanded_revisions and allow_latent_state
                else None
            )
            if latent_res is not None and latent_res.get("unexplained_nonlinear_latent"):
                reset_by_dir = latent_res["reset_exchange_by_direction"]
                flat_reset = [m for items in reset_by_dir.values() for m in items]
                narrowed_pos_nl = (
                    {
                        xf: sorted({m["mapping"][xf] for m in flat_reset})
                        for xf in x_fields
                    }
                    if exhaustive and flat_reset
                    else None
                )
                rep_dir_nl = latent_res["representative_direction"]
                rep_r_iv = latent_res["representative_radius_interval"]
                focus_nl = _audit_focus_membership_from_trajectories(
                    model,
                    records,
                    narrowed_pos_nl,
                    budget,
                    eff_tol=eff_tol,
                    max_bijections=max_bijections,
                    drive_by_q_override=drive_by_q,
                    allow_higher_order_latent=allow_higher_order_latent,
                    verified_direction=rep_dir_nl,
                    radius_interval=rep_r_iv,
                    commitments=commitments,
                )
                narrowed_focus_nl = (
                    focus_nl["narrowed_focus_domains"]
                    if focus_nl is not None and focus_nl.get("narrowed_focus_domains")
                    else None
                )
                _set_node(
                    "P_markov_state",
                    "refuted",
                    reason="reset_conserved_but_history_transitions_violate_linear_reservoirs",
                    counterexample=witness,
                )
                _set_node(
                    "P_latent_order",
                    "refuted_linear_reservoirs",
                    reason="nonlinear_per_object_latent_drive_required",
                )
                refuted_premises.insert(
                    0,
                    {
                        "premise_id": "P_latent_transition_drive",
                        "reason": "nonlinear_per_object_latent_transition_drive_required",
                        "counterexample": witness,
                    },
                )
                refuted_premises.insert(
                    0,
                    {
                        "premise_id": "P_markov_state",
                        "reason": "unexplained_nonlinear_latent_transition_state",
                        "counterexample": witness,
                    },
                )
                return {
                    "status": "unexplained_latent_transition_state",
                    "conservation_supported": False,
                    "reset_conservation_supported": True,
                    "conservation_mse": float(conservation_mse),
                    "exhaustive_bijections": exhaustive,
                    "recovered_drive_by_q": dict(drive_by_q) if recovered_roles is not None else None,
                    "recovered_position_candidates": deepcopy(positions) if recovered_roles is not None else None,
                    "narrowed_position_domains": narrowed_pos_nl,
                    "narrowed_focus_domains": narrowed_focus_nl,
                    "verified_direction": rep_dir_nl,
                    "radius_interval": rep_r_iv,
                    "matches_by_direction": reset_by_dir,
                    "transition_latent_payload": {
                        "valid_rows": valid_rows,
                        "dynamic": dynamic,
                        "drive_by_q": dict(drive_by_q),
                        "x_fields": x_fields,
                        "eff_tol": eff_tol,
                        "representative_direction": rep_dir_nl,
                        "representative_mapping": latent_res["representative_mapping"],
                        "representative_radius_interval": rep_r_iv,
                        "representative_coupling": latent_res["representative_coupling"],
                        "deflated_trajectory_samples": latent_res["deflated_trajectory_samples"],
                    },
                    "dependency_graph": graph,
                    "supported_premises": supported_premises,
                    "refuted_premises": refuted_premises,
                    "scientific_revision_triggers": scientific_triggers,
                    "compute_cost": budget.spent - start,
                }
            if latent_res is not None:
                lat_mech = latent_res.get("mechanism", "transition_memory")
                lat_order = int(latent_res.get("latent_order", 1))
                lat_coupling = latent_res.get("latent_coupling", "independent")
                ar2_params_lat = latent_res.get("ar2_params")
                lat_coupling_rate = latent_res.get("latent_coupling_rate")
                flat_lat_matches = [
                    m
                    for by_dir in latent_res["matches_by_cond"].values()
                    for items in by_dir.values()
                    for m in items
                ]
                supported_conds_lat = sorted(latent_res["matches_by_cond"].keys())
                supported_dirs_lat = list(
                    dict.fromkeys(
                        d_name
                        for by_dir in latent_res["matches_by_cond"].values()
                        for d_name in by_dir.keys()
                    )
                )
                matches_by_dir_lat: dict[str, list[dict]] = {}
                for m in flat_lat_matches:
                    matches_by_dir_lat.setdefault(m["direction"], []).append(m)

                _set_node(
                    "P_markov_state",
                    "refuted",
                    latent_mechanism=lat_mech,
                    latent_order=lat_order,
                    latent_coupling=lat_coupling,
                    pole=latent_res["pole"],
                    ar2_params=ar2_params_lat,
                    drive_weights=latent_res["drive_weights"],
                    recurrence_mse=latent_res["recurrence_mse"],
                    counterexample=witness,
                )
                _set_node(
                    "P_latent_order",
                    "revised_to_order_2" if lat_order == 2 else "supported",
                    verified_order=lat_order,
                    ar2_params=ar2_params_lat,
                )
                _set_node(
                    "P_latent_coupling",
                    "revised_to_spatial_coupled" if lat_coupling != "independent" else "supported",
                    verified_coupling=lat_coupling,
                    latent_coupling_rate=lat_coupling_rate,
                )
                refuted_premises.insert(
                    0,
                    {
                        "premise_id": "P_markov_state",
                        "reason": "unobserved_per_object_latent_transition_state_detected",
                        "latent_mechanism": lat_mech,
                        "latent_order": lat_order,
                        "latent_coupling": lat_coupling,
                        "pole": latent_res["pole"],
                        "ar2_params": ar2_params_lat,
                        "drive_weights": latent_res["drive_weights"],
                        "counterexample": witness,
                    },
                )
                _set_node(
                    "P_conservation",
                    "supported_after_latent_deflation",
                    deflated_conservation_mse=latent_res["deflated_conservation_mse"],
                )
                _set_node(
                    "P_exchange_form",
                    "supported" if supported_conds_lat == ["unconditional"] else "revised_to_conditional",
                    verified_conditions=supported_conds_lat,
                )
                if len(supported_dirs_lat) == 1:
                    _set_node("P_interaction_direction", "supported", verified_direction=supported_dirs_lat[0])
                    supported_premises.append("P_interaction_direction")
                else:
                    _set_node("P_interaction_direction", "narrowed", candidate_directions=supported_dirs_lat)

                narrowed_domains_lat: dict[str, list[str]] | None = None
                if exhaustive:
                    narrowed_domains_lat = {
                        x_field: sorted({m["mapping"][x_field] for m in flat_lat_matches})
                        for x_field in x_fields
                    }
                    unique_owners_lat = all(len(owners) == 1 for owners in narrowed_domains_lat.values())
                    _set_node(
                        "P_position_ownership",
                        "supported" if unique_owners_lat else "narrowed",
                        surviving_bijections=len(flat_lat_matches),
                        narrowed_domain_sizes={k: len(v) for k, v in narrowed_domains_lat.items()},
                    )
                    if unique_owners_lat:
                        supported_premises.append("P_position_ownership")

                overall_low_lat = min(m["radius_interval"][0] for m in flat_lat_matches)
                overall_high_lat = max(m["radius_interval"][1] for m in flat_lat_matches)
                radius_interval_lat = [float(overall_low_lat), float(overall_high_lat)]
                _set_node("P_locality_radius", "supported", radius_interval=radius_interval_lat)
                supported_premises.append("P_locality_radius")

                focus_audit_lat = _audit_focus_membership_from_trajectories(
                    model,
                    records,
                    narrowed_domains_lat,
                    budget,
                    eff_tol=eff_tol,
                    max_bijections=max_bijections,
                    drive_by_q_override=drive_by_q,
                    allow_higher_order_latent=allow_higher_order_latent,
                    verified_direction=supported_dirs_lat[0] if len(supported_dirs_lat) == 1 else None,
                    radius_interval=radius_interval_lat,
                    commitments=commitments,
                )
                narrowed_focus_lat = (
                    focus_audit_lat["narrowed_focus_domains"]
                    if focus_audit_lat is not None and focus_audit_lat.get("narrowed_focus_domains")
                    else None
                )
                if narrowed_focus_lat:
                    unique_focus_lat = all(len(v) == 1 for v in narrowed_focus_lat.values())
                    _set_node(
                        "P_focus_membership",
                        "supported" if unique_focus_lat else "narrowed",
                        surviving_bijections=focus_audit_lat.get("surviving_count"),
                        narrowed_domain_sizes={k: len(v) for k, v in narrowed_focus_lat.items()},
                    )
                    if unique_focus_lat and "P_focus_membership" not in supported_premises:
                        supported_premises.append("P_focus_membership")

                if exhaustive:
                    for cond_name, by_dir in latent_res["matches_by_cond"].items():
                        pred_prog = edge_condition_predicate(cond_name)
                        for alt_dir, alt_matches in by_dir.items():
                            low = min(m["radius_interval"][0] for m in alt_matches)
                            high = max(m["radius_interval"][1] for m in alt_matches)
                            scientific_triggers.append(
                                {
                                    "kind": "latent_transition_state_revision",
                                    "retracted_markov": "markov_sufficient",
                                    "latent_mechanism": lat_mech,
                                    "latent_order": lat_order,
                                    "latent_coupling": lat_coupling,
                                    "latent_pole": latent_res["pole"],
                                    "ar2_params": deepcopy(ar2_params_lat),
                                    "latent_coupling_rate": lat_coupling_rate,
                                    "latent_drive_weights": deepcopy(latent_res["drive_weights"]),
                                    "replacement_direction": alt_dir,
                                    "replacement_condition": cond_name,
                                    "predicate_program": None
                                    if cond_name == "unconditional"
                                    else pred_prog.to_dict(),
                                    "radius_interval": [float(low), float(high)],
                                    "supporting_transitions": latent_res["deflated_unique_transitions"],
                                    "counterexample": witness,
                                }
                            )

                return {
                    "status": "verified_latent_transition_exchange",
                    "conservation_supported": False,
                    "conservation_mse": float(conservation_mse),
                    "latent_transition_state": {
                        "mechanism": lat_mech,
                        "latent_order": lat_order,
                        "latent_coupling": lat_coupling,
                        "pole": latent_res["pole"],
                        "ar2_params": deepcopy(ar2_params_lat),
                        "latent_coupling_rate": lat_coupling_rate,
                        "drive_weights": deepcopy(latent_res["drive_weights"]),
                        "recurrence_mse": latent_res["recurrence_mse"],
                        "deflated_conservation_mse": latent_res["deflated_conservation_mse"],
                    },
                    "recovered_drive_by_q": dict(drive_by_q) if recovered_roles is not None else None,
                    "recovered_position_candidates": deepcopy(positions) if recovered_roles is not None else None,
                    "exhaustive_bijections": exhaustive,
                    "tested_transitions": len(valid_rows),
                    "unique_transitions": len(q_rows),
                    "supported_directions": supported_dirs_lat,
                    "verified_direction": supported_dirs_lat[0] if len(supported_dirs_lat) == 1 else None,
                    "verified_edge_condition": supported_conds_lat[0] if len(supported_conds_lat) == 1 else None,
                    "surviving_bijection_count": len(flat_lat_matches),
                    "narrowed_position_domains": narrowed_domains_lat,
                    "narrowed_focus_domains": narrowed_focus_lat,
                    "focus_membership_audit": focus_audit_lat,
                    "radius_interval": radius_interval_lat,
                    "verified_radius_interval": radius_interval_lat,
                    "matches_by_direction": {
                        d: [
                            {
                                "mapping": m["mapping"],
                                "radius_interval": m["radius_interval"],
                                "coupling_estimate": m["coupling_estimate"],
                                "edge_condition": m["edge_condition"],
                                "mse": m["mse"],
                            }
                            for m in items
                        ]
                        for d, items in matches_by_dir_lat.items()
                    },
                    "dependency_graph": graph,
                    "supported_premises": supported_premises,
                    "refuted_premises": refuted_premises,
                    "scientific_revision_triggers": scientific_triggers,
                    "compute_cost": budget.spent - start,
                }

            _set_node("P_exchange_form", "blocked_by_upstream_refutation")
            return {
                "status": "conservation_refuted",
                "conservation_supported": False,
                "conservation_mse": float(conservation_mse),
                "tested_transitions": len(valid_rows),
                "unique_transitions": len(q_rows),
                "dependency_graph": graph,
                "supported_premises": supported_premises,
                "refuted_premises": refuted_premises,
                "scientific_revision_triggers": scientific_triggers,
                "compute_cost": budget.spent - start,
            }

        # Non-conservative exchange verified!
        supported_mechs = list(noncons_matches_by_mech.keys())
        supported_conds = sorted(
            {
                cond_name
                for by_cond in noncons_matches_by_mech.values()
                for cond_name in by_cond.keys()
            }
        )
        supported_dirs = list(
            dict.fromkeys(
                d_name
                for by_cond in noncons_matches_by_mech.values()
                for by_dir in by_cond.values()
                for d_name in by_dir.keys()
            )
        )
        matches_by_direction: dict[str, list[dict]] = {}
        for m in flat_nc_matches:
            matches_by_direction.setdefault(m["direction"], []).append(m)

        _set_node(
            "P_conservation",
            "refuted",
            counterexample=witness,
            identified_mechanisms=supported_mechs,
            total_rate_kappa=float(kappa),
        )
        _set_node(
            "P_exchange_form",
            "supported" if supported_conds == ["unconditional"] else "revised_to_conditional",
            verified_conditions=supported_conds,
        )
        if len(supported_dirs) == 1:
            _set_node("P_interaction_direction", "supported", verified_direction=supported_dirs[0])
            supported_premises.append("P_interaction_direction")
        else:
            _set_node("P_interaction_direction", "narrowed", candidate_directions=supported_dirs)

        narrowed_domains: dict[str, list[str]] | None = None
        if exhaustive:
            narrowed_domains = {
                x_field: sorted({m["mapping"][x_field] for m in flat_nc_matches})
                for x_field in x_fields
            }
            unique_owners = all(len(owners) == 1 for owners in narrowed_domains.values())
            _set_node(
                "P_position_ownership",
                "supported" if unique_owners else "narrowed",
                surviving_bijections=len(flat_nc_matches),
                narrowed_domain_sizes={k: len(v) for k, v in narrowed_domains.items()},
            )
            if unique_owners:
                supported_premises.append("P_position_ownership")

        overall_low = min(m["radius_interval"][0] for m in flat_nc_matches)
        overall_high = max(m["radius_interval"][1] for m in flat_nc_matches)
        radius_interval = [float(overall_low), float(overall_high)]
        _set_node("P_locality_radius", "supported", radius_interval=radius_interval)
        supported_premises.append("P_locality_radius")

        focus_audit = _audit_focus_membership_from_trajectories(
            model,
            records,
            narrowed_domains,
            budget,
            eff_tol=eff_tol,
            max_bijections=max_bijections,
        )
        narrowed_focus_domains = (
            focus_audit["narrowed_focus_domains"]
            if focus_audit is not None and focus_audit.get("narrowed_focus_domains")
            else None
        )
        if narrowed_focus_domains:
            unique_focus = all(len(v) == 1 for v in narrowed_focus_domains.values())
            _set_node(
                "P_focus_membership",
                "supported" if unique_focus else "narrowed",
                surviving_bijections=focus_audit.get("surviving_count"),
                narrowed_domain_sizes={k: len(v) for k, v in narrowed_focus_domains.items()},
            )
            if unique_focus and "P_focus_membership" not in supported_premises:
                supported_premises.append("P_focus_membership")

        if exhaustive:
            for mech_name, by_cond in noncons_matches_by_mech.items():
                for cond_name, by_dir in by_cond.items():
                    pred_prog = edge_condition_predicate(cond_name)
                    for alt_dir, alt_matches in by_dir.items():
                        low = min(m["radius_interval"][0] for m in alt_matches)
                        high = max(m["radius_interval"][1] for m in alt_matches)
                        scientific_triggers.append(
                            {
                                "kind": "nonconservation_mechanism_revision",
                                "retracted_conservation": "conserved",
                                "replacement_mechanism": mech_name,
                                "estimated_rate": float(alt_matches[0]["nonconservation_rate"]),
                                "replacement_direction": alt_dir,
                                "replacement_condition": cond_name,
                                "predicate_program": None
                                if cond_name == "unconditional"
                                else pred_prog.to_dict(),
                                "radius_interval": [float(low), float(high)],
                                "supporting_transitions": len(q_rows),
                                "counterexample": witness,
                            }
                        )

        return {
            "status": "verified_nonconservative_exchange",
            "conservation_supported": False,
            "conservation_mse": float(conservation_mse),
            "nonconservation_mechanisms": supported_mechs,
            "exhaustive_bijections": exhaustive,
            "tested_transitions": len(valid_rows),
            "unique_transitions": len(q_rows),
            "supported_directions": supported_dirs,
            "verified_direction": supported_dirs[0] if len(supported_dirs) == 1 else None,
            "verified_edge_condition": supported_conds[0] if len(supported_conds) == 1 else None,
            "surviving_bijection_count": len(flat_nc_matches),
            "narrowed_position_domains": narrowed_domains,
            "narrowed_focus_domains": narrowed_focus_domains,
            "focus_membership_audit": focus_audit,
            "radius_interval": radius_interval,
            "verified_radius_interval": radius_interval,
            "matches_by_direction": {
                d: [
                    {
                        "mapping": m["mapping"],
                        "radius_interval": m["radius_interval"],
                        "coupling_estimate": m["coupling_estimate"],
                        "edge_condition": m["edge_condition"],
                        "nonconservation_mechanism": m["nonconservation_mechanism"],
                        "mse": m["mse"],
                    }
                    for m in items
                ]
                for d, items in matches_by_direction.items()
            },
            "dependency_graph": graph,
            "supported_premises": supported_premises,
            "refuted_premises": refuted_premises,
            "scientific_revision_triggers": scientific_triggers,
            "compute_cost": budget.spent - start,
        }

    _set_node("P_markov_state", "supported")
    supported_premises.append("P_markov_state")
    _set_node("P_conservation", "supported", conservation_mse=float(conservation_mse))
    supported_premises.append("P_conservation")

    # 2. Identify active exchange transitions
    active_threshold = max(1e-7, sqrt(eff_tol))
    active_taus = sorted(
        (
            tau
            for tau, d_vec in enumerate(delta_rows)
            if max(abs(val) for val in d_vec) > active_threshold
        ),
        key=lambda tau: sum(val * val for val in delta_rows[tau]),
        reverse=True,
    )
    if not active_taus:
        _set_node("P_exchange_form", "underdetermined_inactive_transitions")
        return {
            "status": "underdetermined_inactive_transitions",
            "conservation_supported": True,
            "conservation_mse": float(conservation_mse),
            "tested_transitions": len(valid_rows),
            "unique_transitions": len(q_rows),
            "active_transitions": 0,
            "dependency_graph": graph,
            "supported_premises": supported_premises,
            "refuted_premises": refuted_premises,
            "scientific_revision_triggers": scientific_triggers,
            "compute_cost": budget.spent - start,
        }

    if not bijections:
        _set_node("P_position_ownership", "refuted", reason="empty_position_domain")
        return {
            "status": "inconsistent_position_domains",
            "conservation_supported": True,
            "tested_transitions": len(valid_rows),
            "unique_transitions": len(q_rows),
            "active_transitions": len(active_taus),
            "dependency_graph": graph,
            "supported_premises": supported_premises,
            "refuted_premises": [{"premise_id": "P_position_ownership", "reason": "empty_domain"}],
            "scientific_revision_triggers": [],
            "compute_cost": budget.spent - start,
        }

    # 3. Screen committed custom predicates first (if any), then unconditional exchange.
    custom_committed_preds = []
    for _, cond_id, _ in committed_specs:
        if isinstance(cond_id, dict) and cond_id not in custom_committed_preds:
            custom_committed_preds.append(cond_id)

    supported_by_direction: dict[str, list[dict]] = {}
    verified_cond_label: str | dict = "unconditional"
    verified_threshold_interval: list[float] | None = None
    verified_threshold_parameter: dict | None = None
    budget_exhausted = False

    if custom_committed_preds:
        for custom_pred_dict in custom_committed_preds:
            custom_prog = Program.from_dict(custom_pred_dict)
            by_dir, budget_exhausted = _solve_exchange_for_condition(
                q_rows,
                x_rows,
                delta_rows,
                dynamic,
                x_fields,
                bijections,
                ordered_directions,
                active_taus,
                eff_tol,
                budget,
                edge_condition=custom_prog,
            )
            if budget_exhausted:
                break
            if by_dir:
                supported_by_direction = by_dir
                verified_cond_label = custom_pred_dict
                desc_cur = describe_comparator_threshold_parameter(custom_prog)
                if desc_cur is not None:
                    verified_threshold_parameter = desc_cur
                    rep_m = next(iter(next(iter(by_dir.values()))))
                    q_idx_by_name = {name: i for i, name in enumerate(dynamic)}
                    x_idx_by_name = {name: i for i, name in enumerate(x_fields)}
                    rep_q_to_x = [0] * n
                    for x_f, q_f in rep_m["mapping"].items():
                        rep_q_to_x[q_idx_by_name[q_f]] = x_idx_by_name[x_f]
                    verified_threshold_interval = extract_comparator_threshold_interval(
                        custom_prog,
                        q_rows,
                        x_rows,
                        delta_rows,
                        rep_m["direction"],
                        tuple(rep_q_to_x),
                        budget,
                        eff_tol,
                        anchor_value=desc_cur["initial_value"],
                    )
                break
            cal_res = screen_custom_edge_predicate_on_transitions(
                q_rows,
                x_rows,
                delta_rows,
                dynamic,
                x_fields,
                bijections,
                custom_prog,
                tuple(d for d in ordered_directions if d in {"right", "left"}),
                budget=budget,
                eff_tol=eff_tol,
            )
            if cal_res.get("supported") and cal_res.get("matches"):
                by_dir_cal: dict[str, list[dict]] = {}
                for m in cal_res["matches"]:
                    by_dir_cal.setdefault(m["direction"], []).append(m)
                supported_by_direction = by_dir_cal
                verified_cond_label = cal_res["calibrated_predicate"].to_dict()
                verified_threshold_interval = cal_res.get("threshold_interval")
                verified_threshold_parameter = cal_res.get("threshold_parameter")
                break

    if not supported_by_direction and not budget_exhausted:
        supported_by_direction, budget_exhausted = _solve_exchange_for_condition(
            q_rows,
            x_rows,
            delta_rows,
            dynamic,
            x_fields,
            bijections,
            ordered_directions,
            active_taus,
            eff_tol,
            budget,
            edge_condition="unconditional",
        )
        verified_cond_label = "unconditional"

    if budget_exhausted:
        return {
            "status": "premise_audit_compute_exhausted",
            "conservation_supported": True,
            "tested_transitions": len(valid_rows),
            "unique_transitions": len(q_rows),
            "active_transitions": len(active_taus),
            "dependency_graph": graph,
            "supported_premises": supported_premises,
            "refuted_premises": refuted_premises,
            "scientific_revision_triggers": scientific_triggers,
            "compute_cost": budget.spent - start,
        }

    all_matches = [
        item for matches in supported_by_direction.values() for item in matches
    ]

    # 4. Phase 2A Conditional Exchange Audit when unconditional exchange fails:
    if not all_matches:
        tau0 = active_taus[0]
        witness = {
            **row_refs[tau0],
            "max_abs_delta": float(max(abs(v) for v in delta_rows[tau0])),
        }
        cond_matches_by_type: dict[str, dict[str, list[dict]]] = {}
        for cond_name in (("source_greater", "source_less") if allow_expanded_revisions else ()):
            by_dir, budget_exhausted = _solve_exchange_for_condition(
                q_rows,
                x_rows,
                delta_rows,
                dynamic,
                x_fields,
                bijections,
                ordered_directions,
                active_taus,
                eff_tol,
                budget,
                edge_condition=cond_name,
            )
            if budget_exhausted:
                break
            if by_dir:
                cond_matches_by_type[cond_name] = by_dir

        flat_cond_matches = [
            m
            for by_dir in cond_matches_by_type.values()
            for items in by_dir.values()
            for m in items
        ]
        if not flat_cond_matches:
            _set_node(
                "P_exchange_form",
                "refuted" if exhaustive else "unverified",
                reason="no_unconditional_or_atomic_conditional_exchange_fits_transitions",
                counterexample=witness,
            )
            if exhaustive:
                refuted_premises.append(
                    {
                        "premise_id": "P_exchange_form",
                        "reason": "conditional_or_non_pairwise_exchange_required",
                        "counterexample": witness,
                    }
                )
            return {
                "status": "conditional_or_nonstandard_exchange",
                "conservation_supported": True,
                "conservation_mse": float(conservation_mse),
                "exhaustive_bijections": exhaustive,
                "tested_transitions": len(valid_rows),
                "unique_transitions": len(q_rows),
                "active_transitions": len(active_taus),
                "transition_payload": {
                    "q_rows": q_rows,
                    "x_rows": x_rows,
                    "delta_rows": delta_rows,
                    "dynamic": dynamic,
                    "x_fields": x_fields,
                    "eff_tol": eff_tol,
                },
                "dependency_graph": graph,
                "supported_premises": supported_premises,
                "refuted_premises": refuted_premises,
                "scientific_revision_triggers": scientific_triggers,
                "compute_cost": budget.spent - start,
            }

        # State-conditional exchange (source_greater / source_less) verified!
        supported_conds = list(cond_matches_by_type.keys())
        supported_directions = list(
            dict.fromkeys(
                d_name
                for by_dir in cond_matches_by_type.values()
                for d_name in by_dir.keys()
            )
        )
        matches_by_direction = {}
        for m in flat_cond_matches:
            matches_by_direction.setdefault(m["direction"], []).append(m)

        # Record refutation if no commitment already expected this exact condition.
        committed_cond_ids = [cond_id for _, cond_id, _ in committed_specs]
        unconditional_expected = (not committed_cond_ids) or any(
            c not in supported_conds for c in committed_cond_ids
        )
        if exhaustive and unconditional_expected:
            _set_node(
                "P_exchange_form",
                "refuted",
                verified_conditions=supported_conds,
                counterexample=witness,
            )
            refuted_premises.append(
                {
                    "premise_id": "P_exchange_form",
                    "committed_condition": committed_cond_ids[0]
                    if committed_cond_ids
                    else "unconditional",
                    "verified_conditions": supported_conds,
                    "reason": "state_dependent_edge_condition_required",
                    "counterexample": witness,
                }
            )
            for cond_name, by_dir in cond_matches_by_type.items():
                pred_prog = edge_condition_predicate(cond_name)
                for alt_dir, alt_matches in by_dir.items():
                    low = min(m["radius_interval"][0] for m in alt_matches)
                    high = max(m["radius_interval"][1] for m in alt_matches)
                    scientific_triggers.append(
                        {
                            "kind": "exchange_condition_revision",
                            "retracted_condition": committed_cond_ids[0]
                            if committed_cond_ids
                            else "unconditional",
                            "replacement_condition": cond_name,
                            "predicate_program": pred_prog.to_dict(),
                            "replacement_direction": alt_dir,
                            "radius_interval": [float(low), float(high)],
                            "supporting_transitions": len(q_rows),
                            "counterexample": witness,
                        }
                    )
        else:
            _set_node("P_exchange_form", "supported", verified_conditions=supported_conds)
            supported_premises.append("P_exchange_form")

        if exhaustive:
            for comm_dir, comm_cond, _ in committed_specs:
                if (
                    comm_cond in cond_matches_by_type
                    and comm_dir not in cond_matches_by_type[comm_cond]
                ):
                    dir_witness = {
                        **row_refs[tau0],
                        "refuted_direction": comm_dir,
                        "verified_condition": comm_cond,
                    }
                    refuted_premises.append(
                        {
                            "premise_id": "P_interaction_direction",
                            "committed_direction": comm_dir,
                            "reason": "committed_direction_inconsistent_with_paid_transitions",
                            "counterexample": dir_witness,
                        }
                    )
                    for alt_dir, alt_matches in cond_matches_by_type[comm_cond].items():
                        low = min(m["radius_interval"][0] for m in alt_matches)
                        high = max(m["radius_interval"][1] for m in alt_matches)
                        scientific_triggers.append(
                            {
                                "kind": "interaction_direction_revision",
                                "retracted_direction": comm_dir,
                                "replacement_direction": alt_dir,
                                "radius_interval": [float(low), float(high)],
                                "supporting_transitions": len(q_rows),
                                "counterexample": dir_witness,
                            }
                        )

        if len(supported_directions) == 1:
            _set_node(
                "P_interaction_direction",
                "supported",
                verified_direction=supported_directions[0],
            )
            supported_premises.append("P_interaction_direction")
        else:
            _set_node(
                "P_interaction_direction",
                "narrowed",
                candidate_directions=supported_directions,
            )

        narrowed_domains = None
        if exhaustive:
            narrowed_domains = {
                x_field: sorted({m["mapping"][x_field] for m in flat_cond_matches})
                for x_field in x_fields
            }
            unique_owners = all(len(owners) == 1 for owners in narrowed_domains.values())
            _set_node(
                "P_position_ownership",
                "supported" if unique_owners else "narrowed",
                surviving_bijections=len(flat_cond_matches),
                narrowed_domain_sizes={k: len(v) for k, v in narrowed_domains.items()},
            )
            if unique_owners:
                supported_premises.append("P_position_ownership")

        overall_low = min(m["radius_interval"][0] for m in flat_cond_matches)
        overall_high = max(m["radius_interval"][1] for m in flat_cond_matches)
        radius_interval = [float(overall_low), float(overall_high)]
        _set_node("P_locality_radius", "supported", radius_interval=radius_interval)
        supported_premises.append("P_locality_radius")

        focus_audit = _audit_focus_membership_from_trajectories(
            model,
            records,
            narrowed_domains,
            budget,
            eff_tol=eff_tol,
            max_bijections=max_bijections,
        )
        narrowed_focus_domains = (
            focus_audit["narrowed_focus_domains"]
            if focus_audit is not None and focus_audit.get("narrowed_focus_domains")
            else None
        )
        if narrowed_focus_domains:
            unique_focus = all(len(v) == 1 for v in narrowed_focus_domains.values())
            _set_node(
                "P_focus_membership",
                "supported" if unique_focus else "narrowed",
                surviving_bijections=focus_audit.get("surviving_count"),
                narrowed_domain_sizes={k: len(v) for k, v in narrowed_focus_domains.items()},
            )
            if unique_focus and "P_focus_membership" not in supported_premises:
                supported_premises.append("P_focus_membership")

        return {
            "status": "verified_conditional_exchange",
            "conservation_supported": True,
            "conservation_mse": float(conservation_mse),
            "exhaustive_bijections": exhaustive,
            "tested_transitions": len(valid_rows),
            "unique_transitions": len(q_rows),
            "active_transitions": len(active_taus),
            "supported_directions": supported_directions,
            "verified_direction": supported_directions[0] if len(supported_directions) == 1 else None,
            "supported_edge_conditions": supported_conds,
            "verified_edge_condition": supported_conds[0] if len(supported_conds) == 1 else None,
            "surviving_bijection_count": len(flat_cond_matches),
            "narrowed_position_domains": narrowed_domains,
            "narrowed_focus_domains": narrowed_focus_domains,
            "focus_membership_audit": focus_audit,
            "radius_interval": radius_interval,
            "verified_radius_interval": radius_interval,
            "matches_by_direction": {
                d: [
                    {
                        "mapping": m["mapping"],
                        "radius_interval": m["radius_interval"],
                        "coupling_estimate": m["coupling_estimate"],
                        "edge_condition": m["edge_condition"],
                        "mse": m["mse"],
                    }
                    for m in items
                ]
                for d, items in matches_by_direction.items()
            },
            "dependency_graph": graph,
            "supported_premises": supported_premises,
            "refuted_premises": refuted_premises,
            "scientific_revision_triggers": scientific_triggers,
            "compute_cost": budget.spent - start,
        }

    # Unconditional (or already-committed custom predicate) pairwise exchange succeeded!
    _set_node("P_exchange_form", "supported", verified_condition=verified_cond_label)
    supported_premises.append("P_exchange_form")

    tau0 = active_taus[0]
    # Check if a commitment expected conditional exchange or nonconservation that was refuted by unconditional conservative data
    for comm_dir, comm_cond, comm_noncons in committed_specs:
        if exhaustive and comm_cond != verified_cond_label:
            witness = {
                **row_refs[tau0],
                "refuted_condition": comm_cond,
                "verified_condition": verified_cond_label,
            }
            refuted_premises.append(
                {
                    "premise_id": "P_exchange_form",
                    "committed_condition": comm_cond,
                    "reason": "committed_edge_condition_refuted_by_unconditional_transitions",
                    "counterexample": witness,
                }
            )
            for alt_dir, alt_matches in supported_by_direction.items():
                low = min(m["radius_interval"][0] for m in alt_matches)
                high = max(m["radius_interval"][1] for m in alt_matches)
                scientific_triggers.append(
                    {
                        "kind": "exchange_condition_revision",
                        "retracted_condition": comm_cond,
                        "replacement_condition": verified_cond_label,
                        "predicate_program": verified_cond_label
                        if isinstance(verified_cond_label, dict)
                        else None,
                        "threshold_interval": verified_threshold_interval,
                        "threshold_parameter": verified_threshold_parameter,
                        "replacement_direction": alt_dir,
                        "radius_interval": [float(low), float(high)],
                        "supporting_transitions": len(q_rows),
                        "counterexample": witness,
                    }
                )

    supported_directions = list(supported_by_direction.keys())
    for comm_dir, comm_cond, _ in committed_specs:
        if exhaustive and comm_cond == verified_cond_label and comm_dir not in supported_by_direction:
            witness = {
                **row_refs[tau0],
                "refuted_direction": comm_dir,
                "active_transition_energy": float(sum(v * v for v in delta_rows[tau0])),
            }
            refuted_premises.append(
                {
                    "premise_id": "P_interaction_direction",
                    "committed_direction": comm_dir,
                    "reason": "committed_direction_inconsistent_with_paid_transitions",
                    "counterexample": witness,
                }
            )
            for alt_dir, alt_matches in supported_by_direction.items():
                low = min(m["radius_interval"][0] for m in alt_matches)
                high = max(m["radius_interval"][1] for m in alt_matches)
                scientific_triggers.append(
                    {
                        "kind": "interaction_direction_revision",
                        "retracted_direction": comm_dir,
                        "replacement_direction": alt_dir,
                        "radius_interval": [float(low), float(high)],
                        "supporting_transitions": len(q_rows),
                        "counterexample": witness,
                    }
                )

    if len(supported_directions) == 1:
        _set_node(
            "P_interaction_direction",
            "supported",
            verified_direction=supported_directions[0],
        )
        supported_premises.append("P_interaction_direction")
    else:
        _set_node(
            "P_interaction_direction",
            "narrowed",
            candidate_directions=supported_directions,
        )

    narrowed_domains = None
    if exhaustive:
        narrowed_domains = {
            x_field: sorted({match["mapping"][x_field] for match in all_matches})
            for x_field in x_fields
        }
        unique_owners = all(len(owners) == 1 for owners in narrowed_domains.values())
        _set_node(
            "P_position_ownership",
            "supported" if unique_owners else "narrowed",
            surviving_bijections=len(all_matches),
            narrowed_domain_sizes={k: len(v) for k, v in narrowed_domains.items()},
        )
        if unique_owners:
            supported_premises.append("P_position_ownership")

    overall_low = min(m["radius_interval"][0] for m in all_matches)
    overall_high = max(m["radius_interval"][1] for m in all_matches)
    radius_interval = [float(overall_low), float(overall_high)]
    _set_node(
        "P_locality_radius",
        "supported",
        radius_interval=radius_interval,
        coupling_estimates=[round(m["coupling_estimate"], 8) for m in all_matches[:4]],
    )
    supported_premises.append("P_locality_radius")

    verified_direction = (
        supported_directions[0] if len(supported_directions) == 1 else None
    )

    focus_audit = _audit_focus_membership_from_trajectories(
        model,
        records,
        narrowed_domains,
        budget,
        eff_tol=eff_tol,
        max_bijections=max_bijections,
        allow_higher_order_latent=allow_higher_order_latent,
        verified_direction=verified_direction,
        radius_interval=radius_interval,
        commitments=commitments,
    )
    narrowed_focus_domains = None
    if focus_audit is not None and focus_audit.get("narrowed_focus_domains"):
        narrowed_focus_domains = focus_audit["narrowed_focus_domains"]
        unique_focus = all(len(v) == 1 for v in narrowed_focus_domains.values())
        _set_node(
            "P_focus_membership",
            "supported" if unique_focus else "narrowed",
            surviving_bijections=focus_audit.get("surviving_count"),
            narrowed_domain_sizes={k: len(v) for k, v in narrowed_focus_domains.items()},
        )
        if unique_focus and "P_focus_membership" not in supported_premises:
            supported_premises.append("P_focus_membership")

    return {
        "status": "verified_unconditional_exchange",
        "conservation_supported": True,
        "conservation_mse": float(conservation_mse),
        "exhaustive_bijections": exhaustive,
        "tested_transitions": len(valid_rows),
        "unique_transitions": len(q_rows),
        "active_transitions": len(active_taus),
        "supported_directions": supported_directions,
        "verified_direction": verified_direction,
        "verified_edge_condition": verified_cond_label,
        "verified_threshold_interval": verified_threshold_interval,
        "verified_threshold_parameter": verified_threshold_parameter,
        "surviving_bijection_count": len(all_matches),
        "narrowed_position_domains": narrowed_domains,
        "narrowed_focus_domains": narrowed_focus_domains,
        "focus_membership_audit": focus_audit,
        "radius_interval": radius_interval,
        "verified_radius_interval": radius_interval,
        "matches_by_direction": {
            d: [
                {
                    "mapping": m["mapping"],
                    "radius_interval": m["radius_interval"],
                    "coupling_estimate": m["coupling_estimate"],
                    "edge_condition": m["edge_condition"],
                    "mse": m["mse"],
                }
                for m in matches
            ]
            for d, matches in supported_by_direction.items()
        },
        "dependency_graph": graph,
        "supported_premises": supported_premises,
        "refuted_premises": refuted_premises,
        "scientific_revision_triggers": scientific_triggers,
        "compute_cost": budget.spent - start,
    }


def apply_verified_transition_premises(model: dict, audit_result: dict) -> dict:
    """Apply sound transition-level position and focus narrowings to ``model``."""
    updated = deepcopy(model)
    updated["premise_audit"] = deepcopy(audit_result)
    updated["premise_dependency_graph"] = deepcopy(audit_result.get("dependency_graph"))

    if (
        audit_result.get("status") not in (*VERIFIED_EXCHANGE_STATUSES, "unexplained_latent_transition_state")
        or not audit_result.get("exhaustive_bijections")
    ):
        return updated

    roles_recovered = False
    rec_drives = audit_result.get("recovered_drive_by_q")
    rec_pos = audit_result.get("recovered_position_candidates")
    role_cands = updated.get("role_candidates", {})
    if rec_drives and rec_pos:
        roles_recovered = True
        updated["position_candidates"] = deepcopy(rec_pos)
        drive_links = deepcopy(updated.get("drive_links", {}))
        for q_name, u_field in rec_drives.items():
            u_ctl = role_cands.get(u_field, {}).get("control")
            if u_ctl:
                drive_links[u_ctl] = {
                    "field": u_field,
                    "quantity_candidates": [q_name],
                    "quantity": q_name,
                    "support_pairs": role_cands.get(u_field, {}).get("support_pairs", 1),
                }
        updated["drive_links"] = drive_links
        for group in updated.get("groups", ()):
            q_name = group.get("q")
            if q_name in rec_drives:
                u_field = rec_drives[q_name]
                group["u"] = u_field
                u_ctl = role_cands.get(u_field, {}).get("control")
                if u_ctl:
                    group.setdefault("controls", {})["u"] = u_ctl
        if updated.get("focus_control") is None:
            f_set = set(updated.get("focus_candidates", {}).keys())
            f_ctls = [
                ctl
                for ctl, eff in updated.get("control_effects", {}).items()
                if eff.get("before_fields") and set(eff["before_fields"]) <= f_set
            ]
            if len(f_ctls) == 1:
                updated["focus_control"] = f_ctls[0]

    positions = updated.get("position_candidates", {})
    focus_cands = updated.get("focus_candidates", {})
    dynamic = list(updated.get("dynamic_fields", ()))
    quantities = set(dynamic)
    narrowed = audit_result.get("narrowed_position_domains") or {}
    narrowed_focus = audit_result.get("narrowed_focus_domains") or {}

    pos_changed = False
    for x_field, allowed in narrowed.items():
        if x_field not in positions:
            continue
        intersected = sorted(set(positions[x_field]["quantities"]) & set(allowed))
        if not intersected:
            updated.setdefault("contradictions", []).append(
                {
                    "field": x_field,
                    "role": "x",
                    "reason": "transition_premise_conflict_with_position_domain",
                    "previous_candidates": list(positions[x_field]["quantities"]),
                    "transition_candidates": list(allowed),
                }
            )
            return updated
        if intersected != positions[x_field]["quantities"]:
            pos_changed = True
            positions[x_field]["quantities"] = intersected

    if pos_changed:
        propagated = _propagate(
            {name: item["quantities"] for name, item in positions.items()}, quantities
        )
        for name, values in propagated.items():
            positions[name]["quantities"] = sorted(values)

        for x_field, item in positions.items():
            if len(item["quantities"]) == 1:
                updated.setdefault("membership_hypotheses", []).append(
                    {
                        "field": x_field,
                        "role": "x",
                        "kind": "transition_premise_dependency_audit",
                        "status": "supported",
                        "quantities": list(item["quantities"]),
                        "supported_directions": list(audit_result.get("supported_directions", ())),
                        "tested_transitions": audit_result.get("tested_transitions", 0),
                    }
                )

        for group in updated.get("groups", ()):
            q_name = group.get("q")
            matched_x = [
                field for field, item in positions.items() if item["quantities"] == [q_name]
            ]
            if len(matched_x) == 1:
                x_field = matched_x[0]
                group["x"] = x_field
                group.setdefault("controls", {})["x"] = positions[x_field]["control"]
            else:
                group.pop("x", None)
                if "controls" in group:
                    group["controls"].pop("x", None)

    focus_changed = False
    for f_field, allowed in narrowed_focus.items():
        if f_field not in focus_cands:
            continue
        intersected = sorted(set(focus_cands[f_field]["quantities"]) & set(allowed))
        if not intersected:
            updated.setdefault("contradictions", []).append(
                {
                    "field": f_field,
                    "role": "f",
                    "reason": "transition_premise_conflict_with_focus_domain",
                    "previous_candidates": list(focus_cands[f_field]["quantities"]),
                    "transition_candidates": list(allowed),
                }
            )
            return updated
        if intersected != focus_cands[f_field]["quantities"]:
            focus_changed = True
            focus_cands[f_field]["quantities"] = intersected

    if focus_changed:
        propagated_f = _propagate(
            {name: item["quantities"] for name, item in focus_cands.items()}, quantities
        )
        for name, values in propagated_f.items():
            focus_cands[name]["quantities"] = sorted(values)

        for f_field, item in focus_cands.items():
            if len(item["quantities"]) == 1:
                updated.setdefault("membership_hypotheses", []).append(
                    {
                        "field": f_field,
                        "role": "f",
                        "kind": "trajectory_premise_dependency_audit",
                        "status": "supported",
                        "quantities": list(item["quantities"]),
                        "trajectory_rows": (audit_result.get("focus_membership_audit") or {}).get(
                            "trajectory_rows", 0
                        ),
                    }
                )

        for group in updated.get("groups", ()):
            q_name = group.get("q")
            matched_f = [
                field for field, item in focus_cands.items() if item["quantities"] == [q_name]
            ]
            if len(matched_f) == 1:
                group["f"] = matched_f[0]
            else:
                group.pop("f", None)

    if not pos_changed and not focus_changed and not roles_recovered:
        return updated

    binary_count = len(focus_cands)
    _finish_object_schema(updated, binary_count)
    return updated


def compare_flat_error_vs_premise_graph(
    commitment: dict,
    model: dict,
    records: list[dict],
    trajectory_rows: list,
    budget: SearchBudget,
    *,
    tolerance: float = 1e-8,
) -> dict:
    """Diagnostic comparison (AWM C.6 analogue): flat trajectory error vs premise DAG."""
    from .continuous_state import _linear_fit, _sum_terms
    from .evaluation import loss
    from .object_alternatives import enumerate_object_schemas
    from .world_structure_commitments import instantiate_commitment_terms

    flat_start = budget.spent
    enum = enumerate_object_schemas(model, budget, max_schemas=8)
    flat_losses = []
    for schema_dict in enum["schemas"]:
        terms = instantiate_commitment_terms(commitment, schema_dict)
        if terms is None or not trajectory_rows:
            continue
        cost = len(trajectory_rows) * (len(terms) + 2)
        if not budget.spend(cost):
            break
        cols = [[term.predict(row) for row in trajectory_rows] for term in terms]
        coeffs = _linear_fit(cols, [row.target for row in trajectory_rows])
        prog = _sum_terms(terms, coeffs[1:], coeffs[0])
        flat_losses.append(loss(prog, trajectory_rows, 0.0, 0.5))
    flat_best = min(flat_losses) if flat_losses else None
    flat_cost = budget.spent - flat_start
    flat_diagnosis = (
        "supported"
        if (flat_best is not None and flat_best <= tolerance)
        else "ambiguous_trajectory_residual_failure"
    )

    graph_start = budget.spent
    audit = audit_transition_premises(
        model, records, budget, commitments=[commitment], tolerance=tolerance
    )
    graph_cost = budget.spent - graph_start
    if audit["refuted_premises"]:
        graph_diagnosis = audit["refuted_premises"][0]["premise_id"]
    elif audit["status"] in VERIFIED_EXCHANGE_STATUSES:
        graph_diagnosis = "transitions_verified_check_readout_or_radius"
    else:
        graph_diagnosis = audit["status"]

    return {
        "flat_item_error": {
            "best_loss": flat_best,
            "searched_schemas": len(flat_losses),
            "diagnosis": flat_diagnosis,
            "isolated_violated_premise": None,
            "compute_cost": flat_cost,
        },
        "premise_dependency_graph": {
            "audit_status": audit["status"],
            "diagnosis": graph_diagnosis,
            "isolated_violated_premise": audit["refuted_premises"][0]["premise_id"]
            if audit["refuted_premises"]
            else None,
            "scientific_revision_triggers": audit["scientific_revision_triggers"],
            "surviving_bijection_count": audit.get("surviving_bijection_count"),
            "compute_cost": graph_cost,
        },
    }
