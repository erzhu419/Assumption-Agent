"""Coupled multi-parameter nonlinear hypersurface calibration & active probing (v10 Phase 4A).

Extends v10 Phase 2C's 1D threshold interval bisection to coupled multi-parameter
curved hypersurface inequalities in 4D edge state space ``(q_src, q_dst, x_src, x_dst)``:

    G(e)  and  (phi_0(e) - sum_{m=1}^M c_m * psi_m(e) > c_0)

Key capabilities:
1. ``describe_linear_in_params_hypersurface(predicate)``:
   Parses open-grammar edge predicates with ``M >= 1`` continuous parameters
   ``(c_0, c_1, ..., c_M)`` into ``(guard_ast, phi_0, [(path_m, c_m, psi_m)], sign)``
   without hardcoding mechanism labels.
2. ``extract_hypersurface_polytope_2d(...)``:
   Computes the exact 2D convex uncertainty polygon ``P subset R^2`` for ``(c_0, c_1)``
   via Sutherland-Hodgman half-plane clipping over decoded fired edges and certified
   in-radius guard-active blocked edges, together with its Chebyshev / analytic
   interior center, marginal intervals, and Euclidean vertex diameter ``diam(P)``.
3. ``calibrate_hypersurface_on_transitions(...)``:
   Deterministically calibrates shifted multi-parameter curved predicates on paid
   ``before -> after`` transitions (used during ``N=4 -> N=5`` cross-scale transfer).
4. ``plan_coupled_hypersurface_probe(...)``:
   Synthesizes 1-step active interventions that align ``(1, psi_1(e))`` with the
   diametral vertex pair ``argmax_{u,v in ext(P)} ||c^(u) - c^(v)||_2`` and place
   ``phi_0(e)`` at the midpoint ``0.5 * (L_A(e) + L_B(e))`` inside public control
   bounds, slicing oblique parameter valleys exponentially faster than passive
   sampling or axis-aligned 1D coordinate bisection.
"""

from __future__ import annotations

from math import hypot, isfinite, sqrt
from typing import Callable

from .language import EvaluationContext, Program
from .models import Observation, SearchBudget
from .relation_conditions import TRUE


def _has_nonzero_const(node: Program) -> bool:
    if node.op == "const" and node.value is not None and abs(float(node.value)) > 1e-9:
        return True
    return any(_has_nonzero_const(ch) for ch in node.children)


def _replace_node_at_path(node: Program, path: tuple[int, ...], replacement: Program) -> Program:
    if not path:
        return replacement
    head, *rest = path
    children = list(node.children)
    children[head] = _replace_node_at_path(children[head], tuple(rest), replacement)
    return Program(
        node.op,
        value=node.value,
        name=node.name,
        lag=node.lag,
        children=tuple(children),
        definition=node.definition,
    )


def rebind_hypersurface_constants(
    predicate: Program,
    desc: dict,
    values: list[float] | tuple[float, ...],
) -> Program:
    """Return ``predicate`` with ``(c_0, c_1, ...)`` replaced by ``values``."""
    updated = predicate
    updated = _replace_node_at_path(
        updated, tuple(desc["c0_path"]), Program("const", value=float(values[0]))
    )
    for term_info, val in zip(desc["slope_terms"], values[1:]):
        updated = _replace_node_at_path(
            updated, tuple(term_info["path"]), Program("const", value=float(val))
        )
    return updated


def normalize_hypersurface_for_grouping(predicate: Program) -> Program | None:
    """Replace all continuous hypersurface constants with canonical symbolic args."""
    desc = describe_linear_in_params_hypersurface(predicate)
    if desc is None:
        return None
    updated = _replace_node_at_path(
        predicate, tuple(desc["c0_path"]), Program("arg", name="__HS_C0__")
    )
    for idx, term_info in enumerate(desc["slope_terms"], start=1):
        updated = _replace_node_at_path(
            updated, tuple(term_info["path"]), Program("arg", name=f"__HS_C{idx}__")
        )
    return updated


def _decompose_linear_expr(
    node: Program,
    path: tuple[int, ...],
    outer_sign: float,
) -> tuple[list[tuple[float, Program]], list[dict]] | None:
    """Decompose ``outer_sign * node`` into base terms and ``c_m * psi_m`` slope terms."""
    if node.op == "add" and len(node.children) == 2:
        left_res = _decompose_linear_expr(node.children[0], (*path, 0), outer_sign)
        right_res = _decompose_linear_expr(node.children[1], (*path, 1), outer_sign)
        if left_res is None or right_res is None:
            return None
        return [*left_res[0], *right_res[0]], [*left_res[1], *right_res[1]]
    if node.op == "sub" and len(node.children) == 2:
        left_res = _decompose_linear_expr(node.children[0], (*path, 0), outer_sign)
        right_res = _decompose_linear_expr(node.children[1], (*path, 1), -outer_sign)
        if left_res is None or right_res is None:
            return None
        return [*left_res[0], *right_res[0]], [*left_res[1], *right_res[1]]
    if node.op == "mul" and len(node.children) == 2:
        left, right = node.children
        if (
            left.op == "const"
            and left.value is not None
            and abs(float(left.value)) > 1e-9
            and not _has_nonzero_const(right)
        ):
            return [], [
                {
                    "path": list((*path, 0)),
                    "initial_value": float(left.value),
                    "raw_psi": right,
                    "outer_sign": float(outer_sign),
                }
            ]
        if (
            right.op == "const"
            and right.value is not None
            and abs(float(right.value)) > 1e-9
            and not _has_nonzero_const(left)
        ):
            return [], [
                {
                    "path": list((*path, 1)),
                    "initial_value": float(right.value),
                    "raw_psi": left,
                    "outer_sign": float(outer_sign),
                }
            ]
    if _has_nonzero_const(node):
        return None
    return [(float(outer_sign), node)], []


def _assemble_signed_sum(items: list[tuple[float, Program]]) -> Program:
    if not items:
        return Program("const", value=0.0)
    first_sign, first_prog = items[0]
    acc = (
        first_prog
        if first_sign > 0.0
        else Program("sub", children=(Program("const", value=0.0), first_prog))
    )
    for sign_val, prog in items[1:]:
        if sign_val > 0.0:
            acc = Program("add", children=(acc, prog))
        else:
            acc = Program("sub", children=(acc, prog))
    return acc


def describe_linear_in_params_hypersurface(predicate: Program) -> dict | None:
    """Parse a coupled multi-parameter curved predicate ``G(e) and (phi_0 - sum c_m psi_m > c_0)``.

    Returns ``None`` if ``predicate`` does not have ``>= 2`` continuous parameters
    (an intercept ``c_0`` plus at least one curvature/slope coefficient ``c_1``) so 1D
    predicates continue using ``describe_comparator_threshold_parameter`` unchanged.
    """
    guard_ast = TRUE
    cmp_node = predicate
    cmp_path: tuple[int, ...] = ()

    if predicate.op == "and" and len(predicate.children) == 2:
        c0_has = _has_nonzero_const(predicate.children[0])
        c1_has = _has_nonzero_const(predicate.children[1])
        if not c0_has and c1_has:
            guard_ast = predicate.children[0]
            cmp_node = predicate.children[1]
            cmp_path = (1,)
        elif c0_has and not c1_has:
            guard_ast = predicate.children[1]
            cmp_node = predicate.children[0]
            cmp_path = (0,)
        else:
            return None

    if cmp_node.op not in {"lt", "ge"} or len(cmp_node.children) != 2:
        return None

    left, right = cmp_node.children
    if left.op == "const" and left.value is not None and right.op != "const":
        c0_path = (*cmp_path, 0)
        c0_init = float(left.value)
        expr_node = right
        expr_path = (*cmp_path, 1)
        # lt(c_0, expr): fires iff c_0 < expr; ge(c_0, expr): fires iff c_0 >= expr
        fires_when_c0_lt_expr = cmp_node.op == "lt"
    elif right.op == "const" and right.value is not None and left.op != "const":
        c0_path = (*cmp_path, 1)
        c0_init = float(right.value)
        expr_node = left
        expr_path = (*cmp_path, 0)
        # lt(expr, c_0): fires iff expr < c_0; ge(expr, c_0): fires iff expr >= c_0
        fires_when_c0_lt_expr = cmp_node.op == "ge"
    else:
        return None

    decomp = _decompose_linear_expr(expr_node, expr_path, 1.0)
    if decomp is None:
        return None
    base_items, slope_raw = decomp
    if not slope_raw or not base_items:
        return None

    phi0_prog = _assemble_signed_sum(base_items)
    # expr = phi_0 + sum_m outer_sign_m * c_m * raw_psi_m = phi_0 - sum_m c_m * psi_m
    # where psi_m = -outer_sign_m * raw_psi_m.
    slope_terms = []
    for item in slope_raw:
        raw_psi = item["raw_psi"]
        psi_prog = (
            raw_psi
            if item["outer_sign"] < 0.0
            else Program("sub", children=(Program("const", value=0.0), raw_psi))
        )
        slope_terms.append(
            {
                "path": item["path"],
                "initial_value": float(item["initial_value"]),
                "psi_expr": psi_prog.to_dict(),
            }
        )

    return {
        "dimension": 1 + len(slope_terms),
        "guard": guard_ast.to_dict(),
        "c0_path": list(c0_path),
        "c0_initial": c0_init,
        "phi0_expr": phi0_prog.to_dict(),
        "slope_terms": slope_terms,
        "fires_when_c0_lt_expr": bool(fires_when_c0_lt_expr),
        "initial_vector": [c0_init, *(t["initial_value"] for t in slope_terms)],
    }


def evaluate_edge_hypersurface_features(
    desc: dict,
    x_src: float,
    q_src: float,
    x_dst: float,
    q_dst: float,
) -> tuple[bool, float, list[float]]:
    """Return ``(guard_active, phi0_val, [psi_1_val, ...])`` on one directed edge."""
    bindings = {
        "edge_source_position": Program("const", value=float(x_src)),
        "edge_source_quantity": Program("const", value=float(q_src)),
        "edge_target_position": Program("const", value=float(x_dst)),
        "edge_target_quantity": Program("const", value=float(q_dst)),
    }
    ctx = EvaluationContext(Observation({}, 0.0))
    guard_prog = Program.from_dict(desc["guard"])
    guard_ok = True if guard_prog == TRUE else bool(ctx.evaluate(guard_prog._expand(bindings)))
    phi0_val = float(ctx.evaluate(Program.from_dict(desc["phi0_expr"])._expand(bindings)))
    psi_vals = [
        float(ctx.evaluate(Program.from_dict(t["psi_expr"])._expand(bindings)))
        for t in desc["slope_terms"]
    ]
    return guard_ok, phi0_val, psi_vals


def _clip_polygon_halfplane(
    poly: list[tuple[float, float]],
    a0: float,
    a1: float,
    rhs: float,
) -> list[tuple[float, float]]:
    """Clip 2D convex polygon ``poly`` to ``a0 * c0 + a1 * c1 <= rhs`` (Sutherland-Hodgman)."""
    if not poly:
        return []
    out: list[tuple[float, float]] = []
    m = len(poly)
    for idx in range(m):
        cur = poly[idx]
        nxt = poly[(idx + 1) % m]
        v_cur = a0 * cur[0] + a1 * cur[1] - rhs
        v_nxt = a0 * nxt[0] + a1 * nxt[1] - rhs
        in_cur = v_cur <= 1e-12
        in_nxt = v_nxt <= 1e-12
        if in_cur and in_nxt:
            out.append(nxt)
        elif in_cur and not in_nxt:
            denom = v_cur - v_nxt
            t = v_cur / denom if abs(denom) > 1e-18 else 0.5
            t = max(0.0, min(1.0, t))
            out.append((cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])))
        elif not in_cur and in_nxt:
            denom = v_cur - v_nxt
            t = v_cur / denom if abs(denom) > 1e-18 else 0.5
            t = max(0.0, min(1.0, t))
            out.append((cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])))
            out.append(nxt)
    # Deduplicate near-coincident consecutive vertices
    cleaned: list[tuple[float, float]] = []
    for pt in out:
        if not cleaned or hypot(pt[0] - cleaned[-1][0], pt[1] - cleaned[-1][1]) > 1e-11:
            cleaned.append(pt)
    if len(cleaned) >= 2 and hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) <= 1e-11:
        cleaned.pop()
    return cleaned


def _polygon_area_and_centroid(poly: list[tuple[float, float]]) -> tuple[float, tuple[float, float]]:
    if not poly:
        return 0.0, (0.0, 0.0)
    if len(poly) == 1:
        return 0.0, poly[0]
    if len(poly) == 2:
        return 0.0, (0.5 * (poly[0][0] + poly[1][0]), 0.5 * (poly[0][1] + poly[1][1]))
    twice_area = 0.0
    cx = 0.0
    cy = 0.0
    m = len(poly)
    for i in range(m):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % m]
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(twice_area) <= 1e-14:
        return (
            0.0,
            (
                sum(p[0] for p in poly) / len(poly),
                sum(p[1] for p in poly) / len(poly),
            ),
        )
    inv = 1.0 / (3.0 * twice_area)
    return 0.5 * abs(twice_area), (cx * inv, cy * inv)


def _chebyshev_like_interior_point(poly: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute a deep interior point maximizing minimum Euclidean distance to polygon edges."""
    area, centroid = _polygon_area_and_centroid(poly)
    if len(poly) < 3 or area <= 1e-14:
        return centroid
    edges: list[tuple[float, float, float]] = []
    m = len(poly)
    # Orient CCW if needed
    signed_sum = sum(
        poly[i][0] * poly[(i + 1) % m][1] - poly[(i + 1) % m][0] * poly[i][1] for i in range(m)
    )
    ordered = poly if signed_sum >= 0.0 else list(reversed(poly))
    for i in range(m):
        x0, y0 = ordered[i]
        x1, y1 = ordered[(i + 1) % m]
        dx, dy = x1 - x0, y1 - y0
        length = hypot(dx, dy)
        if length <= 1e-13:
            continue
        # Outward unit normal for CCW polygon is (dy / length, -dx / length)
        nx, ny = dy / length, -dx / length
        rhs = nx * x0 + ny * y0
        edges.append((nx, ny, rhs))
    if not edges:
        return centroid

    def min_slack(cx: float, cy: float) -> float:
        return min(rhs - (nx * cx + ny * cy) for nx, ny, rhs in edges)

    best_pt = centroid
    best_slack = min_slack(centroid[0], centroid[1])
    v_avg = (
        sum(p[0] for p in ordered) / len(ordered),
        sum(p[1] for p in ordered) / len(ordered),
    )
    if min_slack(*v_avg) > best_slack:
        best_pt, best_slack = v_avg, min_slack(*v_avg)

    # Subgradient ascent on concave min_slack inside bounding box of polygon
    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]
    span = max(1e-6, hypot(max(xs) - min(xs), max(ys) - min(ys)))
    cur_x, cur_y = best_pt
    step = 0.18 * span
    for _ in range(48):
        worst_edge = min(edges, key=lambda e: e[2] - (e[0] * cur_x + e[1] * cur_y))
        cand_x = cur_x - step * worst_edge[0]
        cand_y = cur_y - step * worst_edge[1]
        s_cand = min_slack(cand_x, cand_y)
        if s_cand > best_slack:
            best_pt, best_slack = (cand_x, cand_y), s_cand
            cur_x, cur_y = cand_x, cand_y
        else:
            step *= 0.65
            cur_x = 0.5 * (cur_x + best_pt[0])
            cur_y = 0.5 * (cur_y + best_pt[1])
        if step < 1e-11:
            break
    return best_pt


def extract_hypersurface_polytope_2d(
    predicate: Program,
    q_rows: list[list[float]],
    x_rows: list[list[float]],
    delta_rows: list[list[float]],
    direction: str,
    q_to_x: tuple[int, ...],
    *,
    decoded_edges: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    certified_in_radius: float,
    prior_box: tuple[tuple[float, float], tuple[float, float]] = ((-2.0, 2.0), (-2.0, 2.0)),
    eff_tol: float = 1e-10,
) -> dict | None:
    """Intersect exact half-planes from decoded fired and certified in-radius blocked edges."""
    desc = describe_linear_in_params_hypersurface(predicate)
    if desc is None or desc["dimension"] != 2 or not q_rows:
        return None

    (c0_lo, c0_hi), (c1_lo, c1_hi) = prior_box
    poly: list[tuple[float, float]] = [
        (float(c0_lo), float(c1_lo)),
        (float(c0_hi), float(c1_lo)),
        (float(c0_hi), float(c1_hi)),
        (float(c0_lo), float(c1_hi)),
    ]
    n = len(q_rows[0])
    sign = 1.0 if direction == "right" else -1.0
    pair_tol = max(1e-7, sqrt(eff_tol))
    fires_lt = bool(desc["fires_when_c0_lt_expr"])
    margin = 1e-8

    fired_by_tau: dict[int, set[tuple[int, int]]] = {}
    for tau, src, dst in decoded_edges:
        fired_by_tau.setdefault(tau, set()).add((src, dst))

    # Estimate coupling gamma from 1-edge decoded transitions to decode >=3-edge DAG rows soundly
    gamma_samples: list[float] = []
    for tau_g, edges_g in fired_by_tau.items():
        if len(edges_g) == 1:
            s_g, d_g = next(iter(edges_g))
            if abs(q_rows[tau_g][s_g]) > 1e-5:
                gamma_samples.append(abs(delta_rows[tau_g][d_g]) / abs(q_rows[tau_g][s_g]))
    gamma_est = sorted(gamma_samples)[len(gamma_samples) // 2] if gamma_samples else None

    uncertain_by_tau: dict[int, set[tuple[int, int]]] = {}
    fully_decoded_taus: set[int] = set()
    if gamma_est is not None and gamma_est > 1e-6:
        for tau, d_vec in enumerate(delta_rows):
            if max(abs(v) for v in d_vec) <= pair_tol:
                fully_decoded_taus.add(tau)
                continue
            existing = fired_by_tau.get(tau)
            if existing:
                resid = list(d_vec)
                for s_e, d_e in existing:
                    flow = gamma_est * q_rows[tau][s_e]
                    resid[s_e] += flow
                    resid[d_e] -= flow
                if max(abs(v) for v in resid) <= pair_tol * 6.0:
                    fully_decoded_taus.add(tau)
                    continue
            # Backtrack over directed guard-active edges (at most C(N, 2) <= 15 for N <= 6)
            cand_edges: list[tuple[int, int, float]] = []
            for s_c in range(n):
                qs_c = q_rows[tau][s_c]
                if abs(qs_c) <= 1e-6:
                    continue
                xs_c = x_rows[tau][q_to_x[s_c]]
                for d_c in range(n):
                    if s_c == d_c:
                        continue
                    xd_c = x_rows[tau][q_to_x[d_c]]
                    if sign * (xd_c - xs_c) <= 0.0:
                        continue
                    g_ok_c, _, _ = evaluate_edge_hypersurface_features(
                        desc, xs_c, qs_c, xd_c, q_rows[tau][d_c]
                    )
                    if g_ok_c:
                        cand_edges.append((s_c, d_c, gamma_est * qs_c))
            if 0 < len(cand_edges) <= 15:
                # Group by donor s_c so out-degree pruning keeps search tiny
                by_donor: dict[int, list[tuple[int, int, float]]] = {}
                for item_c in cand_edges:
                    by_donor.setdefault(item_c[0], []).append(item_c)
                donors_ord = sorted(
                    range(n), key=lambda idx_o: sign * x_rows[tau][q_to_x[idx_o]]
                )
                valid_sets: list[set[tuple[int, int]]] = []

                def _bt(d_pos: int, acc_d: list[float], chosen: list[tuple[int, int]]):
                    if len(valid_sets) >= 8:
                        return
                    if d_pos == n:
                        if max(abs(acc_d[k] - d_vec[k]) for k in range(n)) <= pair_tol * 6.0:
                            valid_sets.append(set(chosen))
                        return
                    u_node = donors_ord[d_pos]
                    outs = by_donor.get(u_node, [])
                    # Because donors_ord is topologically sorted along direction, u_node receives NO future incoming edges!
                    need_out = acc_d[u_node] - d_vec[u_node]
                    if not outs:
                        if abs(need_out) <= pair_tol * 6.0:
                            _bt(d_pos + 1, acc_d, chosen)
                        return
                    unit_flow = gamma_est * q_rows[tau][u_node]
                    k_Float = need_out / unit_flow if abs(unit_flow) > 1e-9 else 0.0
                    k_int = round(k_Float)
                    if k_int < 0 or k_int > len(outs) or abs(need_out - k_int * unit_flow) > pair_tol * 6.0:
                        return
                    from itertools import combinations as _comb

                    for sub in _comb(outs, k_int):
                        nxt_d = list(acc_d)
                        for s_u, t_u, fl_u in sub:
                            nxt_d[s_u] -= fl_u
                            nxt_d[t_u] += fl_u
                        _bt(d_pos + 1, nxt_d, [*chosen, *((s_u, t_u) for s_u, t_u, _ in sub)])

                _bt(0, [0.0] * n, [])
                if len(valid_sets) == 1:
                    fired_by_tau[tau] = valid_sets[0]
                    fully_decoded_taus.add(tau)
                elif len(valid_sets) > 1:
                    common_f = set.intersection(*valid_sets)
                    union_f = set.union(*valid_sets)
                    if common_f:
                        fired_by_tau.setdefault(tau, set()).update(common_f)
                    uncertain_by_tau[tau] = union_f - fired_by_tau.get(tau, set())
                    fully_decoded_taus.add(tau)

    all_certified_fired = [
        (tau_f, s_f, d_f)
        for tau_f, fset in sorted(fired_by_tau.items())
        for s_f, d_f in sorted(fset)
    ]

    fired_halfspaces = 0
    blocked_halfspaces = 0

    # 1. Fired edges: every decoded active transfer MUST satisfy both guard and inequality
    for tau, src, dst in all_certified_fired:
        xs, xd = x_rows[tau][q_to_x[src]], x_rows[tau][q_to_x[dst]]
        qs, qd = q_rows[tau][src], q_rows[tau][dst]
        guard_ok, phi0, psi_list = evaluate_edge_hypersurface_features(desc, xs, qs, xd, qd)
        if not guard_ok:
            return None
        psi1 = psi_list[0]
        if fires_lt:
            # c0 + psi1 * c1 <= phi0 - margin
            poly = _clip_polygon_halfplane(poly, 1.0, psi1, phi0 - margin)
        else:
            # -c0 - psi1 * c1 <= -phi0 - margin
            poly = _clip_polygon_halfplane(poly, -1.0, -psi1, -phi0 - margin)
        fired_halfspaces += 1
        if not poly:
            return None

    # 2. Certified in-radius guard-active blocked edges (dist <= certified_in_radius)
    for tau, d_vec in enumerate(delta_rows):
        t_fired = fired_by_tau.get(tau, set())
        t_uncertain = uncertain_by_tau.get(tau, set())
        is_zero_row = max(abs(v) for v in d_vec) <= pair_tol
        row_known = tau in fully_decoded_taus or is_zero_row
        for src in range(n):
            qs = q_rows[tau][src]
            if abs(qs) <= 1e-6:
                continue
            donor_silent = is_zero_row or (
                abs(d_vec[src]) <= pair_tol and all(s != src for s, _ in t_fired)
            )
            edges_from_src_known = row_known or donor_silent or any(s == src for s, _ in t_fired)
            if not edges_from_src_known:
                continue
            xs = x_rows[tau][q_to_x[src]]
            for dst in range(n):
                if src == dst or (src, dst) in t_fired or (src, dst) in t_uncertain:
                    continue
                xd = x_rows[tau][q_to_x[dst]]
                dist = sign * (xd - xs)
                if 0.0 < dist <= certified_in_radius + 1e-11:
                    qd = q_rows[tau][dst]
                    guard_ok, phi0, psi_list = evaluate_edge_hypersurface_features(
                        desc, xs, qs, xd, qd
                    )
                    if not guard_ok:
                        continue
                    psi1 = psi_list[0]
                    if fires_lt:
                        # Blocked => c0 + psi1 * c1 >= phi0 <=> -c0 - psi1 * c1 <= -phi0
                        poly = _clip_polygon_halfplane(poly, -1.0, -psi1, -phi0)
                    else:
                        poly = _clip_polygon_halfplane(poly, 1.0, psi1, phi0)
                    blocked_halfspaces += 1
                    if not poly:
                        return None

    area, centroid = _polygon_area_and_centroid(poly)
    center = _chebyshev_like_interior_point(poly)
    c0_vals = [p[0] for p in poly]
    c1_vals = [p[1] for p in poly]
    diam = 0.0
    diam_pair = (poly[0], poly[-1])
    for i in range(len(poly)):
        for j in range(i + 1, len(poly)):
            d = hypot(poly[i][0] - poly[j][0], poly[i][1] - poly[j][1])
            if d > diam:
                diam = d
                diam_pair = (poly[i], poly[j])

    return {
        "vertices": [[float(p[0]), float(p[1])] for p in poly],
        "vertex_count": len(poly),
        "area": float(area),
        "centroid": [float(centroid[0]), float(centroid[1])],
        "chebyshev_center": [float(center[0]), float(center[1])],
        "c0_interval": [float(min(c0_vals)), float(max(c0_vals))],
        "c1_interval": [float(min(c1_vals)), float(max(c1_vals))],
        "diameter": float(diam),
        "diametral_vertices": [
            [float(diam_pair[0][0]), float(diam_pair[0][1])],
            [float(diam_pair[1][0]), float(diam_pair[1][1])],
        ],
        "fired_halfspaces": fired_halfspaces,
        "blocked_halfspaces": blocked_halfspaces,
    }


def calibrate_hypersurface_on_transitions(
    predicate: Program,
    q_rows: list[list[float]],
    x_rows: list[list[float]],
    delta_rows: list[list[float]],
    dynamic: list[str],
    x_fields: list[str],
    nec_geom: dict | None,
    directions: tuple[str, ...],
    *,
    budget: SearchBudget,
    eff_tol: float,
    prior_box: tuple[tuple[float, float], tuple[float, float]] = ((-2.0, 2.0), (-2.0, 2.0)),
) -> dict | None:
    """Calibrate a 2-parameter curved edge predicate on paid transitions via polytope clipping."""
    from .premise_dependency_audit import _exact_interval_across_transitions

    desc = describe_linear_in_params_hypersurface(predicate)
    if desc is None or desc["dimension"] != 2 or not nec_geom:
        return None
    surv_by_dir = nec_geom.get("surviving_by_direction") or {}
    decoded_edges = nec_geom.get("decoded_edges") or ()
    if not surv_by_dir or not decoded_edges:
        return None
    n = len(dynamic)

    for d_name, bij_items in surv_by_dir.items():
        if d_name not in directions or not bij_items:
            continue
        sign = 1.0 if d_name == "right" else -1.0
        for bij_list, r_min in bij_items:
            rep_bij = tuple(bij_list)
            if not budget.spend(max(1, len(q_rows) * n * 2)):
                return None

            # Candidate certified radii: start with r_min (strictly necessary!), then
            # test larger observed directional distances if r_min leaves outer ambiguity.
            obs_dists = sorted(
                {
                    round(sign * (x_rows[tau][rep_bij[dst]] - x_rows[tau][rep_bij[src]]), 9)
                    for tau in range(len(q_rows))
                    for src in range(n)
                    if abs(q_rows[tau][src]) > 1e-6
                    for dst in range(n)
                    if src != dst and sign * (x_rows[tau][rep_bij[dst]] - x_rows[tau][rep_bij[src]]) > r_min + 1e-8
                }
            )
            all_r_certs = [float(r_min), *(float(od) + 1e-8 for od in obs_dists)]
            # Monotone binary search for largest r_cert whose half-space polytope is non-empty:
            lo_idx, hi_idx = 0, len(all_r_certs) - 1
            max_valid_idx = 0
            poly_cache: dict[int, dict | None] = {}

            def _poly_at(idx_c: int) -> dict | None:
                if idx_c not in poly_cache:
                    poly_cache[idx_c] = extract_hypersurface_polytope_2d(
                        predicate,
                        q_rows,
                        x_rows,
                        delta_rows,
                        d_name,
                        rep_bij,
                        decoded_edges=decoded_edges,
                        certified_in_radius=all_r_certs[idx_c],
                        prior_box=prior_box,
                        eff_tol=eff_tol,
                    )
                return poly_cache[idx_c]

            if _poly_at(0) is None:
                continue
            while lo_idx <= hi_idx:
                mid_idx = (lo_idx + hi_idx) // 2
                if _poly_at(mid_idx) is not None:
                    max_valid_idx = mid_idx
                    lo_idx = mid_idx + 1
                else:
                    hi_idx = mid_idx - 1

            # Test witness candidates from max_valid_idx downward, plus index 0:
            order_indices = list(range(max_valid_idx, -1, -1))
            for idx_r in order_indices:
                poly_info = _poly_at(idx_r)
                if poly_info is None:
                    continue
                for pt_cand in (poly_info["chebyshev_center"], poly_info["centroid"]):
                    cal_pred = rebind_hypersurface_constants(predicate, desc, pt_cand)
                    if not budget.spend(len(q_rows) * n):
                        return None
                    solved = _exact_interval_across_transitions(
                        q_rows,
                        x_rows,
                        delta_rows,
                        d_name,
                        rep_bij,
                        eff_tol,
                        edge_condition=cal_pred,
                    )
                    if solved is not None:
                        r_lo_verified = float(solved["radius_interval"][0])
                        # Sound uncertainty polytope uses ONLY strictly certified in-radius edges!
                        refined_poly = (
                            extract_hypersurface_polytope_2d(
                                predicate,
                                q_rows,
                                x_rows,
                                delta_rows,
                                d_name,
                                rep_bij,
                                decoded_edges=decoded_edges,
                                certified_in_radius=max(float(r_min), r_lo_verified - 1e-8),
                                prior_box=prior_box,
                                eff_tol=eff_tol,
                            )
                            or poly_info
                        )
                        best_vec = refined_poly["chebyshev_center"]
                        best_pred = rebind_hypersurface_constants(predicate, desc, best_vec)
                        recheck = _exact_interval_across_transitions(
                            q_rows,
                            x_rows,
                            delta_rows,
                            d_name,
                            rep_bij,
                            eff_tol,
                            edge_condition=best_pred,
                        )
                        if recheck is not None:
                            solved = recheck
                        else:
                            best_pred = cal_pred
                        solved["mapping"] = {
                            x_fields[x_idx]: dynamic[q_idx] for q_idx, x_idx in enumerate(rep_bij)
                        }
                        return {
                            "supported": True,
                            "matches": [solved],
                            "calibrated_predicate": best_pred,
                            "hypersurface_polytope": refined_poly,
                            "hypersurface_descriptor": desc,
                            "threshold_interval": refined_poly["c0_interval"],
                        }
    return None


def invert_monotone_sensor_fn(
    forward_fn: Callable[[float], float],
    bounds: tuple[float, float],
    target_sensor_val: float,
    *,
    steps: int = 34,
) -> float:
    """Invert a monotone 1D control->sensor map ``forward_fn`` on ``bounds`` via bisection."""
    lo, hi = float(bounds[0]), float(bounds[1])
    v_lo, v_hi = forward_fn(lo), forward_fn(hi)
    increasing = v_hi >= v_lo
    if increasing:
        if target_sensor_val <= v_lo:
            return lo
        if target_sensor_val >= v_hi:
            return hi
    else:
        if target_sensor_val >= v_lo:
            return lo
        if target_sensor_val <= v_hi:
            return hi
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        v_mid = forward_fn(mid)
        if (v_mid < target_sensor_val) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def plan_coupled_hypersurface_probe(
    predicate: Program,
    polytope: dict,
    radius_interval: list[float] | tuple[float, float],
    direction: str,
    control_map: dict[str, dict[str, str]],
    feature_bounds: dict[str, tuple[float, float]],
    *,
    focus_control: str = "focus",
    q_forward_fns: dict[str, Callable[[float], float]] | None = None,
    mode: str = "coupled_diametral",
    axis_toggle: int = 0,
    probe_id: str = "hypersurface_probe_0",
    allow_fine_radius_polish: bool = False,
) -> dict | None:
    """Design a 1-row intervention to shrink coupled ``(r_0, c_0, c_1)`` uncertainty.

    Modes:
    - ``coupled_diametral`` (Phase 4A):
      1. If ``r_high - r_low > 0.35 * diam(P)`` and ``r_high - r_low > 5e-4``, probes
         ``r_mid = 0.5 * (r_low + r_high)`` deep inside the certified firing half-space
         (``phi_0(e) > max_{v in ext(P)} L_v(e) + margin``) so radius and hypersurface
         never confound each other.
      2. Otherwise picks the diametral extreme vertex pair ``(c^(A), c^(B)) = argmax ||u - v||_2``
         of ``P``, searches feasible single-edge setups ``(q_src, q_dst, dx)`` inside
         ``0 < dx < r_low`` maximizing ``|(c_0^(B) - c_0^(A)) + (c_1^(B) - c_1^(A)) * psi_1(e)|``
         subject to ``phi_0(e) == 0.5 * (L_A(e) + L_B(e))`` and ``G(e) == True``.
    - ``axis_aligned_1d`` (Baseline arm):
      Fixes ``c_1`` at its current marginal midpoint (when ``axis_toggle % 2 == 0``) or
      ``c_0`` at its current marginal midpoint (when ``axis_toggle % 2 == 1``) at a single
      nominal interior background ``dx = 0.55 * r_low`` without diametral leverage selection.
    """
    desc = describe_linear_in_params_hypersurface(predicate)
    if desc is None or desc["dimension"] != 2:
        return None
    obj_keys = list(control_map.keys())
    if len(obj_keys) < 2:
        return None
    src_obj, dst_obj = obj_keys[0], obj_keys[1]
    other_objs = obj_keys[2:]

    r_lo, r_hi = float(radius_interval[0]), float(radius_interval[1])
    diam = float(polytope["diameter"])
    verts = [tuple(v) for v in polytope["vertices"]]
    sign = 1.0 if direction == "right" else -1.0

    q_fns = q_forward_fns or {}
    f_src = q_fns.get(src_obj, lambda u: float(u))
    f_dst = q_fns.get(dst_obj, lambda u: float(u))
    b_qs = feature_bounds[control_map[src_obj]["q"]]
    b_qd = feature_bounds[control_map[dst_obj]["q"]]
    b_xs = feature_bounds[control_map[src_obj]["x"]]
    b_xd = feature_bounds[control_map[dst_obj]["x"]]

    qs_min, qs_max = min(f_src(b_qs[0]), f_src(b_qs[1])), max(f_src(b_qs[0]), f_src(b_qs[1]))
    qd_min, qd_max = min(f_dst(b_qd[0]), f_dst(b_qd[1])), max(f_dst(b_qd[0]), f_dst(b_qd[1]))
    max_dx_feasible = min(3.8, (b_xd[1] - b_xs[0]) if sign > 0 else (b_xs[1] - b_xd[0]))

    def _pack_point(qs_phys: float, qd_phys: float, dx_pos: float) -> dict[str, float]:
        pt: dict[str, float] = {}
        # Place src_obj and dst_obj near -1.2 so other objects can sit safely at +1.95 (or -1.95)
        if sign > 0:
            xs_val = max(b_xs[0], -1.6)
            xd_val = min(b_xd[1], xs_val + dx_pos)
            park_x = min(b_xs[1], 1.98)
            # Notice: for right-directed transfer (x_t > x_s), parking spectators at
            # x = -1.98 < xs_val guarantees spectators cannot be recipients from src/dst,
            # and setting q_spectator = 0.0 guarantees spectators donate 0!
            park_x = max(b_xs[0], -1.98)
            xs_val = max(b_xs[0], min(b_xs[1] - dx_pos, 0.2))
            xd_val = xs_val + dx_pos
        else:
            xs_val = min(b_xs[1], max(b_xs[0] + dx_pos, -0.2))
            xd_val = xs_val - dx_pos
            park_x = min(b_xs[1], 1.98)

        pt[control_map[src_obj]["q"]] = invert_monotone_sensor_fn(f_src, b_qs, qs_phys)
        pt[control_map[src_obj]["x"]] = float(xs_val)
        pt[control_map[src_obj]["u"]] = 0.0

        pt[control_map[dst_obj]["q"]] = invert_monotone_sensor_fn(f_dst, b_qd, qd_phys)
        pt[control_map[dst_obj]["x"]] = float(xd_val)
        pt[control_map[dst_obj]["u"]] = 0.0

        for idx_o, oth in enumerate(other_objs):
            b_qo = feature_bounds[control_map[oth]["q"]]
            f_oth = q_fns.get(oth, lambda u: float(u))
            pt[control_map[oth]["q"]] = invert_monotone_sensor_fn(f_oth, b_qo, 0.0)
            b_xo = feature_bounds[control_map[oth]["x"]]
            pt[control_map[oth]["x"]] = max(b_xo[0], min(b_xo[1], park_x + 0.01 * idx_o * (-sign)))
            pt[control_map[oth]["u"]] = 0.0

        if focus_control in feature_bounds:
            lo_f, hi_f = feature_bounds[focus_control]
            # Focus on dst_obj (index 1) or src_obj (index 0) within bounds
            pt[focus_control] = max(lo_f, min(hi_f, 0.0))
        for k_fb, (lo_fb, hi_fb) in feature_bounds.items():
            if k_fb not in pt:
                pt[k_fb] = float(0.5 * (lo_fb + hi_fb))
            else:
                pt[k_fb] = max(float(lo_fb), min(float(hi_fb), float(pt[k_fb])))
        return pt

    # Step A: Decoupled radius bisection first to unlock full in-radius leverage cone [0, r_lo],
    # plus optional sub-micron radius polish once the 2D hypersurface polytope is tightly localized.
    r_span = r_hi - r_lo
    need_radius_step = (r_span > max(0.015, 0.08 * diam)) or (
        allow_fine_radius_polish and diam <= 0.024 and r_span > 5e-7
    )
    if (
        mode == "coupled_diametral"
        and need_radius_step
        and 0.02 < 0.5 * (r_lo + r_hi) < max_dx_feasible - 0.02
    ):
        r_mid = 0.5 * (r_lo + r_hi)
        # Pick (qs, qd) at dx = r_mid guaranteed to satisfy the hypersurface across ALL vertices of P!
        best_rad_cand = None
        for qs_try in (0.98 * qs_max, 0.90 * qs_max, 0.78 * qs_max):
            for qd_try in (0.95 * qd_min, 0.5 * qd_min, 0.0, 0.20 * qd_max):
                guard_ok, phi0, psi_list = evaluate_edge_hypersurface_features(
                    desc, 0.0, qs_try, sign * r_mid, qd_try
                )
                if not guard_ok or abs(qs_try) < 0.1:
                    continue
                worst_margin = min(
                    (phi0 - (v0 + v1 * psi_list[0]))
                    if desc["fires_when_c0_lt_expr"]
                    else ((v0 + v1 * psi_list[0]) - phi0)
                    for v0, v1 in verts
                )
                if best_rad_cand is None or worst_margin > best_rad_cand[0]:
                    best_rad_cand = (worst_margin, qs_try, qd_try)
        if best_rad_cand is not None and best_rad_cand[0] > 1e-3:
            _, qs_r, qd_r = best_rad_cand
            return {
                "probe_id": probe_id,
                "target_kind": "radius_bisection",
                "point": _pack_point(qs_r, qd_r, r_mid),
                "target_radius": r_mid,
                "safety_margin": float(best_rad_cand[0]),
            }

    # Step B: Hypersurface cut safely inside verified radius r_lo
    safe_r_max = min(max_dx_feasible - 0.02, max(0.05, r_lo - 5e-4))
    if mode == "axis_aligned_1d":
        # 1D boundary bisection along primary control qs at a single fixed nominal background
        # (dx_nom, qd_nom) without coupled diametral leverage optimization across psi_1.
        c0_mid = 0.5 * (polytope["c0_interval"][0] + polytope["c0_interval"][1])
        c1_mid = 0.5 * (polytope["c1_interval"][0] + polytope["c1_interval"][1])
        dx_nom = 0.55 * safe_r_max
        qd_nom = 0.25 * qs_max
        qs_grid = [qs_min + (qs_max - qs_min) * k / 48.0 for k in range(25, 48)]
        best_1d = None
        for qs_try in qs_grid:
            guard_ok, phi0, psi_list = evaluate_edge_hypersurface_features(
                desc, 0.0, qs_try, sign * dx_nom, qd_nom
            )
            if not guard_ok or abs(qs_try) < 0.08:
                continue
            target_phi = c0_mid + c1_mid * psi_list[0]
            resid = abs(phi0 - target_phi)
            if best_1d is None or resid < best_1d[0]:
                best_1d = (resid, qs_try, qd_nom, dx_nom, psi_list[0], target_phi)
        if best_1d is None:
            return None
        _, qs_b, qd_b, dx_b, psi_b, tphi_b = best_1d
        lo_q, hi_q = max(0.05, qs_b - 0.08), min(qs_max, qs_b + 0.08)
        def _f_1d(q_v: float) -> float:
            _, ph, ps = evaluate_edge_hypersurface_features(desc, 0.0, q_v, sign * dx_b, qd_b)
            return ph - (c0_mid + c1_mid * ps[0])
        if _f_1d(lo_q) * _f_1d(hi_q) <= 0.0:
            inc = _f_1d(hi_q) >= _f_1d(lo_q)
            for _ in range(30):
                mq = 0.5 * (lo_q + hi_q)
                if (_f_1d(mq) < 0.0) == inc:
                    lo_q = mq
                else:
                    hi_q = mq
            qs_b = 0.5 * (lo_q + hi_q)
        return {
            "probe_id": probe_id,
            "target_kind": "axis_aligned_1d_slice",
            "point": _pack_point(qs_b, qd_b, dx_b),
            "psi1": float(psi_b),
            "target_phi0": float(tphi_b),
        }

    # Phase 4A Coupled Diametral-Vertex Max-Leverage Synthesis:
    va, vb = polytope["diametral_vertices"]
    dc0 = float(vb[0] - va[0])
    dc1 = float(vb[1] - va[1])

    dx_candidates = [
        0.01,
        max(0.02, safe_r_max * 0.15),
        max(0.03, safe_r_max * 0.45),
        max(0.04, safe_r_max * 0.78),
        max(0.05, safe_r_max * 0.992),
    ]
    qd_candidates = [
        qd_min + (qd_max - qd_min) * k / 24.0 for k in range(25)
    ]
    qs_coarse = [
        qs_min + (qs_max - qs_min) * k / 32.0 for k in range(1, 33)
    ]

    best_coupled = None
    for dx_c in dx_candidates:
        for qd_c in qd_candidates:
            # Scan qs_coarse for sign change of g(qs) = phi0(e) - 0.5 * (L_A(e) + L_B(e))
            prev_q = None
            prev_g = None
            for qs_c in qs_coarse:
                if abs(qs_c) < 0.08:
                    continue
                guard_ok, phi0, psi_list = evaluate_edge_hypersurface_features(
                    desc, 0.0, qs_c, sign * dx_c, qd_c
                )
                if not guard_ok:
                    prev_q, prev_g = None, None
                    continue
                psi1 = psi_list[0]
                mid_lv = 0.5 * ((va[0] + va[1] * psi1) + (vb[0] + vb[1] * psi1))
                g_val = phi0 - mid_lv
                if prev_q is not None and prev_g is not None and prev_g * g_val <= 0.0:
                    # Exact root via 28-step bisection along qs!
                    lo_q, hi_q = prev_q, qs_c
                    inc = g_val >= prev_g
                    valid_root = True
                    for _ in range(28):
                        mq = 0.5 * (lo_q + hi_q)
                        g_ok_m, ph_m, ps_m = evaluate_edge_hypersurface_features(
                            desc, 0.0, mq, sign * dx_c, qd_c
                        )
                        if not g_ok_m:
                            valid_root = False
                            break
                        gv_m = ph_m - 0.5 * ((va[0] + va[1] * ps_m[0]) + (vb[0] + vb[1] * ps_m[0]))
                        if (gv_m < 0.0) == inc:
                            lo_q = mq
                        else:
                            hi_q = mq
                    if valid_root:
                        qs_star = 0.5 * (lo_q + hi_q)
                        g_ok_s, ph_s, ps_s = evaluate_edge_hypersurface_features(
                            desc, 0.0, qs_star, sign * dx_c, qd_c
                        )
                        if g_ok_s and abs(qs_star) >= 0.08:
                            leverage = abs(dc0 + dc1 * ps_s[0])
                            if best_coupled is None or leverage > best_coupled[0]:
                                best_coupled = (leverage, qs_star, qd_c, dx_c, ps_s[0], ph_s)
                prev_q, prev_g = qs_c, g_val

    if best_coupled is None:
        # Fallback if midpoint lies slightly outside achievable control box: minimize |g_val| weighted by leverage
        for dx_c in dx_candidates:
            for qd_c in qd_candidates[::2]:
                for qs_c in qs_coarse:
                    if abs(qs_c) < 0.08:
                        continue
                    g_ok, phi0, psi_list = evaluate_edge_hypersurface_features(
                        desc, 0.0, qs_c, sign * dx_c, qd_c
                    )
                    if not g_ok:
                        continue
                    psi1 = psi_list[0]
                    mid_lv = 0.5 * ((va[0] + va[1] * psi1) + (vb[0] + vb[1] * psi1))
                    score = abs(dc0 + dc1 * psi1) - 4.0 * abs(phi0 - mid_lv)
                    if best_coupled is None or score > best_coupled[0]:
                        best_coupled = (score, qs_c, qd_c, dx_c, psi1, phi0)

    if best_coupled is None:
        return None
    lev, qs_opt, qd_opt, dx_opt, psi_opt, phi_opt = best_coupled
    return {
        "probe_id": probe_id,
        "target_kind": "coupled_diametral_hypersurface_slice",
        "point": _pack_point(qs_opt, qd_opt, dx_opt),
        "diametral_vertices": [list(va), list(vb)],
        "leverage": float(lev),
        "psi1": float(psi_opt),
        "target_phi0": float(phi_opt),
        "dx": float(dx_opt),
    }
