import json
import math
from pathlib import Path

cfg = json.loads(Path("synth_config.json").read_text(encoding="utf-8"))
raw_rows = json.loads(Path("observations.json").read_text(encoding="utf-8"))
train_idx = cfg["train_indices"]
val_idx = cfg["val_indices"]
lens = cfg["observation_lens"]
allowed_schemas = cfg.get("allowed_schemas")

sign = float(lens.get("sign", 1.0))
scale = max(1e-9, float(lens.get("scale", 1.0)))
offset = float(lens.get("offset", 0.0))

def to_latent(y_raw: float) -> float:
    return (float(y_raw) - offset) / (sign * scale)

def to_raw(y_lat: float) -> float:
    return offset + sign * scale * float(y_lat)

rows_lat = [
    {"features": r["features"], "y_raw": float(r["target"]), "y_lat": to_latent(r["target"])}
    for r in raw_rows
]
train_rows = [rows_lat[i] for i in train_idx]
val_rows = [rows_lat[i] for i in val_idx] if val_idx else train_rows

fnames = list(cfg["feature_names"])
bounds = cfg["feature_bounds"]
active_f = [f for f in fnames if (bounds[f][1] - bounds[f][0]) > 1e-9]

def solve_linear_ls(basis_fn, rows):
    if not rows:
        return [0.0]
    m = len(basis_fn(rows[0]["features"]))
    ata = [[1e-8 if i == j else 0.0 for j in range(m)] for i in range(m)]
    aty = [0.0 for _ in range(m)]
    for r in rows:
        b = basis_fn(r["features"])
        y = r["y_lat"]
        for i in range(m):
            aty[i] += b[i] * y
            for j in range(m):
                ata[i][j] += b[i] * b[j]
    aug = [ata[i] + [aty[i]] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r_i: abs(aug[r_i][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) > 1e-12:
            for r_i in range(col + 1, m):
                fac = aug[r_i][col] / aug[col][col]
                for c_i in range(col, m + 1):
                    aug[r_i][c_i] -= fac * aug[col][c_i]
    w = [0.0] * m
    for r_i in range(m - 1, -1, -1):
        if abs(aug[r_i][r_i]) > 1e-12:
            w[r_i] = (aug[r_i][m] - sum(aug[r_i][c_i] * w[c_i] for c_i in range(r_i + 1, m))) / aug[r_i][r_i]
    return w

def rmse_raw(pred_lat_fn, rows):
    if not rows:
        return float("inf")
    return math.sqrt(sum((to_raw(pred_lat_fn(r["features"])) - r["y_raw"]) ** 2 for r in rows) / len(rows))

def allow(tag):
    return allowed_schemas is None or tag in allowed_schemas

candidates = []

# ==================== 1D MECHANISMS (6) ====================
if len(active_f) == 1 or (allowed_schemas is not None and len(allowed_schemas) == 1 and allowed_schemas[0].endswith("_1d")):
    for u_k in (active_f if len(active_f) > 1 else [active_f[0]]):
        if allow("linear_proportional_1d"):
            w = solve_linear_ls(lambda f, k=u_k: [1.0, float(f[k])], train_rows)
            w0, w1 = max(1e-5, w[0]) if w[0] >= 0 else w[0], max(1e-5, w[1])
            fn = lambda f, a=w0, b=w1, k=u_k: a + b * float(f[k])
            expr = f"({abs(w0):.10g} + {w1:.10g} * f[{u_k!r}])" if w0 >= 0 else f"(-{abs(w0):.10g} + {w1:.10g} * f[{u_k!r}])"
            if w0 >= 0:
                fn = lambda f, a=abs(w0), b=w1, k=u_k: a + b * float(f[k])
            candidates.append(("linear_proportional_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("rational_monod_1d"):
            pts = [(float(r["features"][u_k]), r["y_lat"]) for r in train_rows if float(r["features"][u_k]) > 0 and r["y_lat"] > 1e-6]
            if len(pts) >= 2:
                xs = [1.0 / u for u, _ in pts]
                zs = [1.0 / y for _, y in pts]
                ws = [max(1e-8, y ** 4) for _, y in pts]
                wsum = sum(ws)
                mx = sum(w * x for w, x in zip(ws, xs)) / wsum
                mz = sum(w * z for w, z in zip(ws, zs)) / wsum
                den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
                b_p = max(1e-5, sum(w * (x - mx) * (z - mz) for w, x, z in zip(ws, xs, zs)) / den if den > 1e-12 else 0.2)
                a_p = max(1e-5, mz - b_p * mx)
                for _ in range(30):
                    ga = gb = haa = hab = hbb = 0.0
                    cur = sum(((u / (a_p * u + b_p)) - y) ** 2 for u, y in pts)
                    for u, y in pts:
                        d = a_p * u + b_p
                        e = (u / d) - y
                        ja, jb = -(u * u) / (d * d), -u / (d * d)
                        ga += e * ja; gb += e * jb; haa += ja * ja; hab += ja * jb; hbb += jb * jb
                    det = (haa + 1e-6) * (hbb + 1e-6) - hab * hab
                    if det <= 1e-15:
                        break
                    na = max(1e-6, a_p - 0.8 * (((hbb + 1e-6) * ga - hab * gb) / det))
                    nb = max(1e-6, b_p - 0.8 * (((haa + 1e-6) * gb - hab * ga) / det))
                    if sum(((u / (na * u + nb)) - y) ** 2 for u, y in pts) <= cur:
                        a_p, b_p = na, nb
                fn = lambda f, a=a_p, b=b_p, k=u_k: float(f[k]) / (a * float(f[k]) + b)
                expr = f"(f[{u_k!r}] / ({a_p:.10g} * f[{u_k!r}] + {b_p:.10g}))"
                candidates.append(("rational_monod_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("congestion_rollover_1d") and len(train_rows) >= 3:
            pts = [(float(r["features"][u_k]), r["y_lat"]) for r in train_rows if float(r["features"][u_k]) > 0 and r["y_lat"] > 1e-6]
            if len(pts) >= 3:
                pseudo = [{"features": {u_k: u}, "y_lat": u / y} for u, y in pts]
                w = solve_linear_ls(lambda f, k=u_k: [1.0, float(f[k]), float(f[k]) ** 2], pseudo)
                b_r, a_r, g_r = max(1e-5, w[0]), max(1e-5, w[1]), max(1e-6, w[2])
                fn = lambda f, a=a_r, b=b_r, g=g_r, k=u_k: float(f[k]) / (a * float(f[k]) + b + g * (float(f[k]) ** 2))
                expr = f"(f[{u_k!r}] / ({a_r:.10g} * f[{u_k!r}] + {b_r:.10g} + {g_r:.10g} * (f[{u_k!r}] ** 2)))"
                candidates.append(("congestion_rollover_1d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("hill_cooperative_n2_1d"):
            pts = [(float(r["features"][u_k]), r["y_lat"]) for r in train_rows if float(r["features"][u_k]) > 0 and r["y_lat"] > 1e-6]
            if len(pts) >= 2:
                xs = [1.0 / (u * u) for u, _ in pts]
                zs = [1.0 / y for _, y in pts]
                ws = [max(1e-8, y ** 4) for _, y in pts]
                wsum = sum(ws)
                mx = sum(w * x for w, x in zip(ws, xs)) / wsum
                mz = sum(w * z for w, z in zip(ws, zs)) / wsum
                den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
                b_p = max(1e-5, sum(w * (x - mx) * (z - mz) for w, x, z in zip(ws, xs, zs)) / den if den > 1e-12 else 0.5)
                a_p = max(1e-5, mz - b_p * mx)
                for _ in range(30):
                    ga = gb = haa = hab = hbb = 0.0
                    cur = sum((((u * u) / (a_p * u * u + b_p)) - y) ** 2 for u, y in pts)
                    for u, y in pts:
                        u2 = u * u
                        d = a_p * u2 + b_p
                        e = (u2 / d) - y
                        ja, jb = -(u2 * u2) / (d * d), -u2 / (d * d)
                        ga += e * ja; gb += e * jb; haa += ja * ja; hab += ja * jb; hbb += jb * jb
                    det = (haa + 1e-6) * (hbb + 1e-6) - hab * hab
                    if det <= 1e-15:
                        break
                    na = max(1e-6, a_p - 0.8 * (((hbb + 1e-6) * ga - hab * gb) / det))
                    nb = max(1e-6, b_p - 0.8 * (((haa + 1e-6) * gb - hab * ga) / det))
                    if sum((((u * u) / (na * u * u + nb)) - y) ** 2 for u, y in pts) <= cur:
                        a_p, b_p = na, nb
                fn = lambda f, a=a_p, b=b_p, k=u_k: (float(f[k]) ** 2) / (a * (float(f[k]) ** 2) + b)
                expr = f"((f[{u_k!r}] ** 2) / ({a_p:.10g} * (f[{u_k!r}] ** 2) + {b_p:.10g}))"
                candidates.append(("hill_cooperative_n2_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("inv_quad_crest_1d") and len(train_rows) >= 3:
            w = solve_linear_ls(lambda f, k=u_k: [1.0, 1.0 / float(f[k]), 1.0 / (float(f[k]) ** 2)], train_rows)
            w0, w1, w2 = max(1e-5, w[0]), max(1e-5, w[1]), max(1e-5, w[2])
            fn = lambda f, a=w0, b=w1, c=w2, k=u_k: a + b / float(f[k]) + c / (float(f[k]) ** 2)
            expr = f"({w0:.10g} + {w1:.10g} / f[{u_k!r}] + {w2:.10g} / (f[{u_k!r}] ** 2))"
            candidates.append(("inv_quad_crest_1d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("exp_washout_decay_1d") and len(train_rows) >= 3:
            best_ew = None
            for tau in [1.5 + 0.35 * i for i in range(32)]:
                w = solve_linear_ls(lambda f, k=u_k, tv=tau: [1.0, math.exp(-float(f[k]) / tv)], train_rows)
                if w[0] > 0 and w[1] > 0:
                    fn = lambda f, a=w[0], b=w[1], tv=tau, k=u_k: a + b * math.exp(-float(f[k]) / tv)
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.45 * tr_e + 0.55 * va_e
                    if best_ew is None or sc < best_ew[0]:
                        best_ew = (sc, tr_e, va_e, tau, w[0], w[1])
            if best_ew is not None:
                tau_b = best_ew[3]
                for step in (0.12, 0.03, 0.008):
                    for dtau in (-step, step):
                        nt = max(0.2, tau_b + dtau)
                        w = solve_linear_ls(lambda f, k=u_k, tv=nt: [1.0, math.exp(-float(f[k]) / tv)], train_rows)
                        if w[0] > 0 and w[1] > 0:
                            fn = lambda f, a=w[0], b=w[1], tv=nt, k=u_k: a + b * math.exp(-float(f[k]) / tv)
                            tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                            sc = 0.45 * tr_e + 0.55 * va_e
                            if sc < best_ew[0]:
                                tau_b = nt
                                best_ew = (sc, tr_e, va_e, nt, w[0], w[1])
                expr = f"({best_ew[4]:.10g} + {best_ew[5]:.10g} * math.exp(-f[{u_k!r}] / {best_ew[3]:.10g}))"
                candidates.append(("exp_washout_decay_1d", 3, best_ew[1], best_ew[2], expr, [u_k]))

        if allow("inv_linear_decay_1d"):
            w = solve_linear_ls(lambda f, k=u_k: [1.0, 1.0 / float(f[k])], train_rows)
            w0, w1 = max(1e-5, w[0]), max(1e-5, w[1])
            fn = lambda f, a=w0, b=w1, k=u_k: a + b / float(f[k])
            expr = f"({w0:.10g} + {w1:.10g} / f[{u_k!r}])"
            candidates.append(("inv_linear_decay_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("sqrt_orifice_1d"):
            w = solve_linear_ls(lambda f, k=u_k: [1.0, math.sqrt(max(0.0, float(f[k])))], train_rows)
            w0, w1 = max(1e-5, w[0]), max(1e-5, w[1])
            fn = lambda f, a=w0, b=w1, k=u_k: a + b * math.sqrt(max(0.0, float(f[k])))
            expr = f"({w0:.10g} + {w1:.10g} * math.sqrt(f[{u_k!r}]))"
            candidates.append(("sqrt_orifice_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

        if allow("log_saturation_1d"):
            w = solve_linear_ls(lambda f, k=u_k: [1.0, math.log(1.0 + max(0.0, float(f[k])))], train_rows)
            w0, w1 = max(1e-5, w[0]), max(1e-5, w[1])
            fn = lambda f, a=w0, b=w1, k=u_k: a + b * math.log(1.0 + max(0.0, float(f[k])))
            expr = f"({w0:.10g} + {w1:.10g} * math.log(1.0 + f[{u_k!r}]))"
            candidates.append(("log_saturation_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

# ==================== 2D & 3D MECHANISMS (12) ====================
if len(active_f) >= 2:
    role_pairs = cfg.get("forced_role_order")
    pairs_to_try = [tuple(role_pairs[:2])] if (role_pairs and len(role_pairs) >= 2) else [
        (f_a, f_b) for f_a in active_f for f_b in active_f if f_a != f_b
    ]
    tau_grid = [0.5 + 0.10 * i for i in range(38)]
    allow_2d_subspace = (len(active_f) == 2) or (allowed_schemas is not None and len(allowed_schemas) == 1)

    if allow_2d_subspace and allow("conjugate_relaxation_reset"):
        best_rel = None
        for d_k, t_k in pairs_to_try:
            for tau in tau_grid:
                w = solve_linear_ls(
                    lambda f, dk=d_k, tk=t_k, tv=tau: [float(f[dk]), float(f[dk]) * math.exp(-float(f[tk]) / tv)],
                    train_rows,
                )
                if w[0] > 0 and w[1] > 0.05:
                    fn = lambda f, w=w, dk=d_k, tk=t_k, tv=tau: w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / tv)
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.45 * tr_e + 0.55 * va_e
                    if best_rel is None or sc < best_rel[0]:
                        best_rel = (sc, tr_e, va_e, d_k, t_k, tau, w)
        if best_rel is not None:
            _, tr_e, va_e, d_k, t_k, tau_b, w_b = best_rel
            for step in (0.04, 0.01):
                for dtau in (-step, step):
                    nt = max(0.15, tau_b + dtau)
                    w = solve_linear_ls(lambda f, dk=d_k, tk=t_k, tv=nt: [float(f[dk]), float(f[dk]) * math.exp(-float(f[tk]) / tv)], train_rows)
                    if w[0] > 0 and w[1] > 0.05:
                        fn = lambda f, w=w, dk=d_k, tk=t_k, tv=nt: w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / tv)
                        ntr, nva = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        nsc = 0.45 * ntr + 0.55 * nva
                        if nsc < best_rel[0]:
                            tau_b, w_b = nt, w
                            best_rel = (nsc, ntr, nva, d_k, t_k, nt, w)
            expr = f"({w_b[0]:.10g} * f[{d_k!r}] + {w_b[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tau_b:.10g}))"
            candidates.append(("conjugate_relaxation_reset", 3, best_rel[1], best_rel[2], expr, [d_k, t_k]))

    if len(active_f) == 2 and allow("overdamped_charging_2d"):
        best_od = None
        for d_k, t_k in pairs_to_try:
            for tau in tau_grid:
                w = solve_linear_ls(
                    lambda f, dk=d_k, tk=t_k, tv=tau: [float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv))],
                    train_rows,
                )
                if w[0] > 0.05:
                    fn = lambda f, k0=w[0], dk=d_k, tk=t_k, tv=tau: k0 * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv))
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.45 * tr_e + 0.55 * va_e
                    if best_od is None or sc < best_od[0]:
                        best_od = (sc, tr_e, va_e, d_k, t_k, tau, w[0])
        if best_od is not None:
            _, _, _, d_k, t_k, tau_b, w0_b = best_od
            for step in (0.04, 0.01, 0.0025):
                for dtau in (-step, step):
                    nt = max(0.15, tau_b + dtau)
                    w = solve_linear_ls(lambda f, dk=d_k, tk=t_k, tv=nt: [float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv))], train_rows)
                    if w[0] > 0.05:
                        fn = lambda f, k0=w[0], dk=d_k, tk=t_k, tv=nt: k0 * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv))
                        ntr, nva = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        nsc = 0.45 * ntr + 0.55 * nva
                        if nsc < best_od[0]:
                            tau_b, w0_b = nt, w[0]
                            best_od = (nsc, ntr, nva, d_k, t_k, nt, w[0])
            expr = f"({w0_b:.10g} * f[{d_k!r}] * (1.0 - math.exp(-f[{t_k!r}] / {tau_b:.10g})))"
            candidates.append(("overdamped_charging_2d", 2, best_od[1], best_od[2], expr, [d_k, t_k]))

    if len(active_f) == 2 and allow("underdamped_ringing_2d") and len(train_rows) >= 4:
        best_ud = None
        for d_k, t_k in pairs_to_try:
            for tau in [1.2 + 0.2 * j for j in range(11)]:
                for om in [1.2 + 0.08 * j for j in range(14)]:
                    w = solve_linear_ls(
                        lambda f, dk=d_k, tk=t_k, tv=tau, wv=om: [
                            float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv) * math.cos(wv * float(f[tk])))
                        ],
                        train_rows,
                    )
                    if w[0] > 0.05:
                        fn = lambda f, k0=w[0], dk=d_k, tk=t_k, tv=tau, wv=om: (
                            k0 * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv) * math.cos(wv * float(f[tk])))
                        )
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.45 * tr_e + 0.55 * va_e
                        if best_ud is None or sc < best_ud[0]:
                            best_ud = (sc, tr_e, va_e, d_k, t_k, tau, om, w[0])
        if best_ud is not None:
            _, _, _, d_k, t_k, tau_b, om_b, w0_b = best_ud
            for st_t, st_o in [(0.06, 0.03), (0.015, 0.008), (0.004, 0.002)]:
                for _ in range(6):
                    for dt, do in [(st_t,0),(-st_t,0),(0,st_o),(0,-st_o)]:
                        nt, no = max(0.3, tau_b + dt), max(0.4, om_b + do)
                        w = solve_linear_ls(
                            lambda f, dk=d_k, tk=t_k, tv=nt, wv=no: [
                                float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv) * math.cos(wv * float(f[tk])))
                            ],
                            train_rows,
                        )
                        if w[0] > 0.05:
                            fn = lambda f, k0=w[0], dk=d_k, tk=t_k, tv=nt, wv=no: (
                                k0 * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv) * math.cos(wv * float(f[tk])))
                            )
                            ntr, nva = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                            nsc = 0.45 * ntr + 0.55 * nva
                            if nsc + 1e-9 < best_ud[0]:
                                tau_b, om_b, w0_b = nt, no, w[0]
                                best_ud = (nsc, ntr, nva, d_k, t_k, nt, no, w[0])
            expr = f"({w0_b:.10g} * f[{d_k!r}] * (1.0 - math.exp(-f[{t_k!r}] / {tau_b:.10g}) * math.cos({om_b:.10g} * f[{t_k!r}])))"
            candidates.append(("underdamped_ringing_2d", 3, best_ud[1], best_ud[2], expr, [d_k, t_k]))

    if len(active_f) == 2 and allow("bimodal_relaxation_2mode") and (len(train_rows) >= 7 or allowed_schemas == ["bimodal_relaxation_2mode"]):
        best_bi = None
        for d_k, t_k in pairs_to_try:
            for tf in (0.35, 0.45, 0.55):
                for ts in (4.5, 5.5, 6.0):
                    w = solve_linear_ls(
                        lambda f, dk=d_k, tk=t_k, t1=tf, t2=ts: [
                            float(f[dk]),
                            float(f[dk]) * math.exp(-float(f[tk]) / t1),
                            float(f[dk]) * math.exp(-float(f[tk]) / t2),
                        ],
                        train_rows,
                    )
                    if w[0] > 0 and w[1] > 0.2 and w[2] > 0.2:
                        fn = lambda f, w=w, dk=d_k, tk=t_k, t1=tf, t2=ts: (
                            w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / t1) + w[2] * float(f[dk]) * math.exp(-float(f[tk]) / t2)
                        )
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.45 * tr_e + 0.55 * va_e
                        if best_bi is None or sc < best_bi[0]:
                            expr = (
                                f"({w[0]:.10g} * f[{d_k!r}] + {w[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tf:.10g}) "
                                f"+ {w[2]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {ts:.10g}))"
                            )
                            best_bi = (sc, tr_e, va_e, expr, [d_k, t_k])
        if best_bi is not None:
            candidates.append(("bimodal_relaxation_2mode", 5, best_bi[1], best_bi[2], best_bi[3], best_bi[4]))

    if len(active_f) == 2 and allow("gaussian_weibull_relax_2d") and len(train_rows) >= 4:
        best_gw = None
        for d_k, t_k in pairs_to_try:
            for t2 in [1.0 + 0.35 * i for i in range(16)]:
                w = solve_linear_ls(
                    lambda f, dk=d_k, tk=t_k, tv2=t2: [float(f[dk]), float(f[dk]) * math.exp(-(float(f[tk]) ** 2) / tv2)],
                    train_rows,
                )
                if w[0] > 0 and w[1] > 0.05:
                    fn = lambda f, w=w, dk=d_k, tk=t_k, tv2=t2: w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-(float(f[tk]) ** 2) / tv2)
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.45 * tr_e + 0.55 * va_e
                    if best_gw is None or sc < best_gw[0]:
                        best_gw = (sc, tr_e, va_e, d_k, t_k, t2, w)
        if best_gw is not None:
            _, _, _, d_k, t_k, t2_b, w_b = best_gw
            for step in (0.08, 0.02):
                for dt2 in (-step, step):
                    nt2 = max(0.2, t2_b + dt2)
                    w = solve_linear_ls(lambda f, dk=d_k, tk=t_k, tv2=nt2: [float(f[dk]), float(f[dk]) * math.exp(-(float(f[tk]) ** 2) / tv2)], train_rows)
                    if w[0] > 0 and w[1] > 0.05:
                        fn = lambda f, w=w, dk=d_k, tk=t_k, tv2=nt2: w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-(float(f[tk]) ** 2) / tv2)
                        ntr, nva = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        nsc = 0.45 * ntr + 0.55 * nva
                        if nsc < best_gw[0]:
                            t2_b, w_b = nt2, w
                            best_gw = (nsc, ntr, nva, d_k, t_k, nt2, w)
            expr = f"({w_b[0]:.10g} * f[{d_k!r}] + {w_b[1]:.10g} * f[{d_k!r}] * math.exp(-(f[{t_k!r}] ** 2) / {t2_b:.10g}))"
            candidates.append(("gaussian_weibull_relax_2d", 3, best_gw[1], best_gw[2], expr, [d_k, t_k]))

    if len(active_f) == 2:
        k1, k2 = active_f[0], active_f[1]
        if allow("liebig_min_2d"):
            r1 = max(r["y_lat"] / max(1e-6, float(r["features"][k1])) for r in train_rows)
            r2 = max(r["y_lat"] / max(1e-6, float(r["features"][k2])) for r in train_rows)
            best_m = None
            for s1 in [0.75 + 0.025 * i for i in range(20)]:
                for s2 in [0.75 + 0.025 * j for j in range(20)]:
                    c1, c2 = max(1e-4, r1 * s1), max(1e-4, r2 * s2)
                    fn = lambda f, a=c1, b=c2, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]))
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.5 * tr_e + 0.5 * va_e
                    if best_m is None or sc < best_m[0]:
                        best_m = (sc, tr_e, va_e, c1, c2)
            c1, c2 = best_m[3], best_m[4]
            for step in (0.04, 0.008):
                for _ in range(12):
                    for dc1, dc2 in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                        nc1, nc2 = max(1e-4, c1 + dc1), max(1e-4, c2 + dc2)
                        fn = lambda f, a=nc1, b=nc2, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]))
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if sc + 1e-9 < best_m[0]:
                            c1, c2 = nc1, nc2
                            best_m = (sc, tr_e, va_e, c1, c2)
            expr = f"min({c1:.10g} * f[{k1!r}], {c2:.10g} * f[{k2!r}])"
            candidates.append(("liebig_min_2d", 2, best_m[1], best_m[2], expr, [k1, k2]))

        if allow("liebig_capped_min_2d") and len(train_rows) >= 4:
            max_obs = max(r["y_lat"] for r in train_rows)
            low_sub = [r for r in train_rows if r["y_lat"] < 0.96 * max_obs]
            base_rows = low_sub if len(low_sub) >= 2 else train_rows
            r1 = max(r["y_lat"] / max(1e-6, float(r["features"][k1])) for r in base_rows)
            r2 = max(r["y_lat"] / max(1e-6, float(r["features"][k2])) for r in base_rows)
            best_cm = None
            for cap_mul in [0.96, 0.99, 1.0, 1.02, 1.06]:
                cap_v = max(1e-4, max_obs * cap_mul)
                for s1 in [0.85 + 0.03 * i for i in range(10)]:
                    for s2 in [0.85 + 0.03 * j for j in range(10)]:
                        c1, c2 = max(1e-4, r1 * s1), max(1e-4, r2 * s2)
                        fn = lambda f, a=c1, b=c2, cp=cap_v, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]), cp)
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if best_cm is None or sc < best_cm[0]:
                            best_cm = (sc, tr_e, va_e, c1, c2, cap_v)
            c1, c2, cap_v = best_cm[3], best_cm[4], best_cm[5]
            for step in (0.03, 0.006):
                for _ in range(12):
                    for dc1, dc2, dcp in [(step,0,0),(-step,0,0),(0,step,0),(0,-step,0),(0,0,step*3),(0,0,-step*3)]:
                        nc1, nc2, ncp = max(1e-4, c1+dc1), max(1e-4, c2+dc2), max(1e-4, cap_v+dcp)
                        fn = lambda f, a=nc1, b=nc2, cp=ncp, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]), cp)
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if sc + 1e-9 < best_cm[0]:
                            c1, c2, cap_v = nc1, nc2, ncp
                            best_cm = (sc, tr_e, va_e, c1, c2, cap_v)
            expr = f"min({c1:.10g} * f[{k1!r}], {c2:.10g} * f[{k2!r}], {cap_v:.10g})"
            candidates.append(("liebig_capped_min_2d", 3, best_cm[1], best_cm[2], expr, [k1, k2]))

        if allow("cobb_douglas_sqrt_2d"):
            w = solve_linear_ls(lambda f, x1=k1, x2=k2: [math.sqrt(max(0.0, float(f[x1]) * float(f[x2])))], train_rows)
            c_cd = max(1e-5, w[0])
            fn = lambda f, c=c_cd, x1=k1, x2=k2: c * math.sqrt(max(0.0, float(f[x1]) * float(f[x2])))
            expr = f"({c_cd:.10g} * math.sqrt(f[{k1!r}] * f[{k2!r}]))"
            candidates.append(("cobb_douglas_sqrt_2d", 1, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [k1, k2]))

        if allow("parallel_harmonic_flux_2d"):
            valid_h = [r for r in train_rows if r["y_lat"] > 1e-5 and float(r["features"][k1]) > 0 and float(r["features"][k2]) > 0]
            if len(valid_h) >= 2:
                pseudo = [{"features": r["features"], "y_lat": (float(r["features"][k1]) * float(r["features"][k2])) / r["y_lat"]} for r in valid_h]
                w = solve_linear_ls(lambda f, x1=k1, x2=k2: [float(f[x1]), float(f[x2])], pseudo)
                a_h, b_h = max(1e-5, w[0]), max(1e-5, w[1])
                fn = lambda f, a=a_h, b=b_h, x1=k1, x2=k2: (float(f[x1]) * float(f[x2])) / (a * float(f[x1]) + b * float(f[x2]))
                expr = f"((f[{k1!r}] * f[{k2!r}]) / ({a_h:.10g} * f[{k1!r}] + {b_h:.10g} * f[{k2!r}]))"
                candidates.append(("parallel_harmonic_flux_2d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [k1, k2]))

        if allow("langmuir_product_2d") and len(train_rows) >= 4:
            valid_lp = [r for r in train_rows if r["y_lat"] > 1e-5 and float(r["features"][k1]) > 0 and float(r["features"][k2]) > 0]
            if len(valid_lp) >= 3:
                pseudo = [{"features": r["features"], "y_lat": (float(r["features"][k1]) * float(r["features"][k2])) / r["y_lat"]} for r in valid_lp]
                w = solve_linear_ls(lambda f, x1=k1, x2=k2: [1.0, float(f[x1]), float(f[x2])], pseudo)
                c0, a_l, b_l = max(1e-4, w[0]), max(1e-5, w[1]), max(1e-5, w[2])
                fn = lambda f, c0=c0, a=a_l, b=b_l, x1=k1, x2=k2: (float(f[x1]) * float(f[x2])) / (c0 + a * float(f[x1]) + b * float(f[x2]))
                expr = f"((f[{k1!r}] * f[{k2!r}]) / ({c0:.10g} + {a_l:.10g} * f[{k1!r}] + {b_l:.10g} * f[{k2!r}]))"
                candidates.append(("langmuir_product_2d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [k1, k2]))

        if allow("monod_plus_linear_2d") and len(train_rows) >= 4:
            best_ml = None
            for s_k, b_k in pairs_to_try:
                for km in [1.0 + 0.35 * i for i in range(14)]:
                    w = solve_linear_ls(
                        lambda f, sk=s_k, bk=b_k, kmv=km: [float(f[sk]) / (kmv + float(f[sk])), float(f[bk])],
                        train_rows,
                    )
                    if w[0] > 0.1 and w[1] > 0.02:
                        fn = lambda f, vm=w[0], bp=w[1], sk=s_k, bk=b_k, kmv=km: (vm * float(f[sk])) / (kmv + float(f[sk])) + bp * float(f[bk])
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.45 * tr_e + 0.55 * va_e
                        if best_ml is None or sc < best_ml[0]:
                            expr = f"((({w[0]:.10g} * f[{s_k!r}]) / ({km:.10g} + f[{s_k!r}])) + {w[1]:.10g} * f[{b_k!r}])"
                            best_ml = (sc, tr_e, va_e, expr, [s_k, b_k])
            if best_ml is not None:
                candidates.append(("monod_plus_linear_2d", 3, best_ml[1], best_ml[2], best_ml[3], best_ml[4]))

        if allow("haldane_shunt_2d") and len(train_rows) >= 5:
            best_hs = None
            for s_k, b_k in pairs_to_try:
                for km in (1.6, 2.0, 2.2, 2.4, 2.8):
                    for ci in (0.045, 0.058, 0.065, 0.072, 0.085):
                        w = solve_linear_ls(
                            lambda f, sk=s_k, bk=b_k, kmv=km, civ=ci: [
                                float(f[sk]) / (kmv + float(f[sk]) + civ * (float(f[sk]) ** 2)),
                                float(f[bk]),
                            ],
                            train_rows,
                        )
                        if w[0] > 0.1 and w[1] > 0.02:
                            fn = lambda f, vm=w[0], bp=w[1], sk=s_k, bk=b_k, kmv=km, civ=ci: (
                                (vm * float(f[sk])) / (kmv + float(f[sk]) + civ * (float(f[sk]) ** 2)) + bp * float(f[bk])
                            )
                            tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                            sc = 0.45 * tr_e + 0.55 * va_e
                            if best_hs is None or sc < best_hs[0]:
                                expr = (
                                    f"((({w[0]:.10g} * f[{s_k!r}]) / ({km:.10g} + f[{s_k!r}] + {ci:.10g} * (f[{s_k!r}] ** 2))) "
                                    f"+ {w[1]:.10g} * f[{b_k!r}])"
                                )
                                best_hs = (sc, tr_e, va_e, expr, [s_k, b_k])
            if best_hs is not None:
                candidates.append(("haldane_shunt_2d", 4, best_hs[1], best_hs[2], best_hs[3], best_hs[4]))

    # ==================== 3D MECHANISMS (3) ====================
    if len(active_f) == 3:
        triples = [tuple(role_pairs[:3])] if (role_pairs and len(role_pairs) >= 3) else [
            (f_a, f_b, f_c) for f_a in active_f for f_b in active_f for f_c in active_f
            if len({f_a, f_b, f_c}) == 3
        ]
        if allow("conjugate_relaxation_state_memory"):
            best_mem = None
            for d_k, t_k, m_k in triples:
                for tau in tau_grid:
                    w = solve_linear_ls(
                        lambda f, dk=d_k, tk=t_k, mk=m_k, tv=tau: [
                            float(f[dk]),
                            float(f[dk]) * math.exp(-float(f[tk]) / tv),
                            float(f[mk]) * math.exp(-float(f[tk]) / tv),
                        ],
                        train_rows,
                    )
                    if w[0] > 0 and w[1] > 0.05 and w[2] > 0.05:
                        fn = lambda f, w=w, dk=d_k, tk=t_k, mk=m_k, tv=tau: (
                            w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / tv) + w[2] * float(f[mk]) * math.exp(-float(f[tk]) / tv)
                        )
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.45 * tr_e + 0.55 * va_e
                        if best_mem is None or sc < best_mem[0]:
                            expr = (
                                f"({w[0]:.10g} * f[{d_k!r}] + {w[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}) "
                                f"+ {w[2]:.10g} * f[{m_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}))"
                            )
                            best_mem = (sc, tr_e, va_e, expr, [d_k, t_k, m_k])
            if best_mem is not None:
                candidates.append(("conjugate_relaxation_state_memory", 4, best_mem[1], best_mem[2], best_mem[3], best_mem[4]))

        if allow("tri_resource_liebig_min_3d"):
            x1, x2, x3 = active_f[0], active_f[1], active_f[2]
            r1 = max(r["y_lat"] / max(1e-6, float(r["features"][x1])) for r in train_rows)
            r2 = max(r["y_lat"] / max(1e-6, float(r["features"][x2])) for r in train_rows)
            r3 = max(r["y_lat"] / max(1e-6, float(r["features"][x3])) for r in train_rows)
            best_t = None
            for s1 in [0.85 + 0.04 * i for i in range(8)]:
                for s2 in [0.85 + 0.04 * j for j in range(8)]:
                    for s3 in [0.85 + 0.04 * k for k in range(8)]:
                        c1, c2, c3 = max(1e-4, r1*s1), max(1e-4, r2*s2), max(1e-4, r3*s3)
                        fn = lambda f, a=c1, b=c2, c=c3, k1=x1, k2=x2, k3=x3: min(a*float(f[k1]), b*float(f[k2]), c*float(f[k3]))
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if best_t is None or sc < best_t[0]:
                            best_t = (sc, tr_e, va_e, c1, c2, c3)
            c1, c2, c3 = best_t[3], best_t[4], best_t[5]
            for step in (0.03, 0.006):
                for _ in range(12):
                    for d1, d2, d3 in [(step,0,0),(-step,0,0),(0,step,0),(0,-step,0),(0,0,step),(0,0,-step)]:
                        nc1, nc2, nc3 = max(1e-4, c1+d1), max(1e-4, c2+d2), max(1e-4, c3+d3)
                        fn = lambda f, a=nc1, b=nc2, c=nc3, k1=x1, k2=x2, k3=x3: min(a*float(f[k1]), b*float(f[k2]), c*float(f[k3]))
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if sc + 1e-9 < best_t[0]:
                            c1, c2, c3 = nc1, nc2, nc3
                            best_t = (sc, tr_e, va_e, c1, c2, c3)
            expr = f"min({c1:.10g} * f[{x1!r}], {c2:.10g} * f[{x2!r}], {c3:.10g} * f[{x3!r}])"
            candidates.append(("tri_resource_liebig_min_3d", 3, best_t[1], best_t[2], expr, [x1, x2, x3]))

        if allow("damped_driven_with_leak_3d"):
            best_dl = None
            for d_k, t_k, l_k in triples:
                for tau in tau_grid:
                    w = solve_linear_ls(
                        lambda f, dk=d_k, tk=t_k, lk=l_k, tv=tau: [
                            float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv)),
                            float(f[lk]),
                        ],
                        train_rows,
                    )
                    if w[0] > 0.05 and w[1] > 0.05:
                        fn = lambda f, a=w[0], b=w[1], dk=d_k, tk=t_k, lk=l_k, tv=tau: (
                            a * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv)) + b * float(f[lk])
                        )
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.45 * tr_e + 0.55 * va_e
                        if best_dl is None or sc < best_dl[0]:
                            best_dl = (sc, tr_e, va_e, d_k, t_k, l_k, tau, w)
            if best_dl is not None:
                _, _, _, d_k, t_k, l_k, tau_b, w_b = best_dl
                for step in (0.04, 0.01):
                    for dtau in (-step, step):
                        nt = max(0.2, tau_b + dtau)
                        w = solve_linear_ls(
                            lambda f, dk=d_k, tk=t_k, lk=l_k, tv=nt: [
                                float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv)),
                                float(f[lk]),
                            ],
                            train_rows,
                        )
                        if w[0] > 0.05 and w[1] > 0.05:
                            fn = lambda f, a=w[0], b=w[1], dk=d_k, tk=t_k, lk=l_k, tv=nt: (
                                a * float(f[dk]) * (1.0 - math.exp(-float(f[tk]) / tv)) + b * float(f[lk])
                            )
                            ntr, nva = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                            nsc = 0.45 * ntr + 0.55 * nva
                            if nsc < best_dl[0]:
                                tau_b, w_b = nt, w
                                best_dl = (nsc, ntr, nva, d_k, t_k, l_k, nt, w)
                expr = f"({w_b[0]:.10g} * f[{d_k!r}] * (1.0 - math.exp(-f[{t_k!r}] / {tau_b:.10g})) + {w_b[1]:.10g} * f[{l_k!r}])"
                candidates.append(("damped_driven_with_leak_3d", 3, best_dl[1], best_dl[2], expr, [d_k, t_k, l_k]))

if not candidates:
    raise RuntimeError("No candidate expression could be synthesized")

candidates.sort(key=lambda item: (round(0.65 * item[3] + 0.35 * item[2] + 0.002 * item[1], 9), item[1], item[0]))
best = candidates[0]
schema_tag, n_params, tr_rmse, va_rmse, core_expr_str, role_order = best

model_py = (
    "import math\n\n"
    f"_SCHEMA_FAMILY_TAG = {schema_tag!r}\n"
    f"_CORE_LAW_EXPR = {core_expr_str!r}\n"
    f"_LENS = {lens!r}\n\n"
    "def fit(observations):\n"
    "    return None\n\n"
    "def predict(row):\n"
    "    f = row['features']\n"
    f"    latent = {core_expr_str}\n"
    f"    return {offset:.10g} + ({sign:.10g}) * ({scale:.10g}) * latent\n"
)
Path(cfg.get("output_model_path", "model.py")).write_text(model_py, encoding="utf-8")

print(json.dumps({
    "selected_schema": schema_tag,
    "num_params": n_params,
    "train_rmse": round(tr_rmse, 6),
    "val_rmse": round(va_rmse, 6),
    "role_order": role_order,
    "evaluated_candidate_count": len(candidates),
}))
