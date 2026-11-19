import json
import math
from pathlib import Path
import sys

cfg = json.loads(Path("synth_config.json").read_text(encoding="utf-8"))
raw_rows = json.loads(Path("observations.json").read_text(encoding="utf-8"))
train_idx = cfg["train_indices"]
val_idx = cfg["val_indices"]
lens = cfg["observation_lens"]  # {"sign": +-1, "scale": s, "offset": b}
allowed_schemas = cfg.get("allowed_schemas")  # None = open compositional search; or specific list

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

candidates = []

def allow(tag):
    return allowed_schemas is None or tag in allowed_schemas

# --- 1D Compositional Candidates ---
if len(active_f) == 1:
    u_k = active_f[0]
    if allow("linear_1d"):
        w = solve_linear_ls(lambda f, k=u_k: [1.0, float(f[k])], train_rows)
        fn = lambda f, w=w, k=u_k: w[0] + w[1] * float(f[k])
        expr = f"({w[0]:.10g} + {w[1]:.10g} * f[{u_k!r}])"
        candidates.append(("linear_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

    if allow("rational_1d"):
        # Gauss-Newton in original raw scale on y_lat = u / (a*u + b)
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
            for _ in range(35):
                ga = gb = haa = hab = hbb = 0.0
                cur = sum(((u / (a_p * u + b_p)) - y) ** 2 for u, y in pts)
                for u, y in pts:
                    d = a_p * u + b_p
                    e = (u / d) - y
                    ja, jb = -(u * u) / (d * d), -u / (d * d)
                    ga += e * ja; gb += e * jb
                    haa += ja * ja; hab += ja * jb; hbb += jb * jb
                det = (haa + 1e-6) * (hbb + 1e-6) - hab * hab
                if det <= 1e-15:
                    break
                da = ((hbb + 1e-6) * ga - hab * gb) / det
                db = ((haa + 1e-6) * gb - hab * ga) / det
                na, nb = max(1e-6, a_p - 0.8 * da), max(1e-6, b_p - 0.8 * db)
                if sum(((u / (na * u + nb)) - y) ** 2 for u, y in pts) <= cur:
                    a_p, b_p = na, nb
            fn = lambda f, a=a_p, b=b_p, k=u_k: float(f[k]) / (a * float(f[k]) + b)
            expr = f"(f[{u_k!r}] / ({a_p:.10g} * f[{u_k!r}] + {b_p:.10g}))"
            candidates.append(("rational_1d", 2, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

    if allow("inv_quad_crest_1d") and len(train_rows) >= 3:
        w = solve_linear_ls(lambda f, k=u_k: [1.0, 1.0 / float(f[k]), 1.0 / (float(f[k]) ** 2)], train_rows)
        fn = lambda f, w=w, k=u_k: w[0] + w[1] / float(f[k]) + w[2] / (float(f[k]) ** 2)
        expr = f"({w[0]:.10g} + {w[1]:.10g} / f[{u_k!r}] + {w[2]:.10g} / (f[{u_k!r}] ** 2))"
        candidates.append(("inv_quad_crest_1d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

    if allow("congestion_rollover_1d") and len(train_rows) >= 4:
        pts = [(float(r["features"][u_k]), r["y_lat"]) for r in train_rows if float(r["features"][u_k]) > 0 and r["y_lat"] > 1e-6]
        if len(pts) >= 4:
            # u / y = b + a*u + g*u^2
            pseudo = [{"features": {u_k: u}, "y_lat": u / y} for u, y in pts]
            w = solve_linear_ls(lambda f, k=u_k: [1.0, float(f[k]), float(f[k]) ** 2], pseudo)
            b_r, a_r, g_r = max(1e-5, w[0]), max(1e-5, w[1]), max(0.0, w[2])
            fn = lambda f, a=a_r, b=b_r, g=g_r, k=u_k: float(f[k]) / (a * float(f[k]) + b + g * (float(f[k]) ** 2))
            expr = f"(f[{u_k!r}] / ({a_r:.10g} * f[{u_k!r}] + {b_r:.10g} + {g_r:.10g} * (f[{u_k!r}] ** 2)))"
            candidates.append(("congestion_rollover_1d", 3, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [u_k]))

# --- Multi-Dimensional Compositional Candidates (2D / 3D Role-Agnostic!) ---
if len(active_f) >= 2:
    # 1. Single-mode relaxation over every ordered pair (drive_k, time_k)
    role_pairs = cfg.get("forced_role_order")
    pairs_to_try = [tuple(role_pairs[:2])] if (role_pairs and len(role_pairs) >= 2) else [
        (f_a, f_b) for f_a in active_f for f_b in active_f if f_a != f_b
    ]
    tau_grid = [0.4 + 0.12 * i for i in range(32)]
    if allow("conjugate_relaxation_reset"):
        best_rel = None
        for d_k, t_k in pairs_to_try:
            for tau in tau_grid:
                w = solve_linear_ls(
                    lambda f, dk=d_k, tk=t_k, tv=tau: [float(f[dk]), float(f[dk]) * math.exp(-float(f[tk]) / tv)],
                    train_rows,
                )
                if w[0] > 0 and w[1] > -0.2:
                    fn = lambda f, w=w, dk=d_k, tk=t_k, tv=tau: w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / tv)
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    score = 0.45 * tr_e + 0.55 * va_e
                    if best_rel is None or score < best_rel[0]:
                        expr = f"({w[0]:.10g} * f[{d_k!r}] + {w[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}))"
                        best_rel = (score, tr_e, va_e, expr, [d_k, t_k])
        if best_rel is not None:
            candidates.append(("conjugate_relaxation_reset", 3, best_rel[1], best_rel[2], best_rel[3], best_rel[4]))

    # 2. State-augmented relaxation with initial memory decay (3 active dims!)
    if len(active_f) >= 3 and allow("conjugate_relaxation_state_memory"):
        best_mem = None
        triples = [tuple(role_pairs[:3])] if (role_pairs and len(role_pairs) >= 3) else [
            (f_a, f_b, f_c) for f_a in active_f for f_b in active_f for f_c in active_f
            if len({f_a, f_b, f_c}) == 3
        ]
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
                fn = lambda f, w=w, dk=d_k, tk=t_k, mk=m_k, tv=tau: (
                    w[0] * float(f[dk]) + w[1] * float(f[dk]) * math.exp(-float(f[tk]) / tv) + w[2] * float(f[mk]) * math.exp(-float(f[tk]) / tv)
                )
                tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                score = 0.45 * tr_e + 0.55 * va_e
                if best_mem is None or score < best_mem[0]:
                    expr = (
                        f"({w[0]:.10g} * f[{d_k!r}] + {w[1]:.10g} * f[{d_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}) "
                        f"+ {w[2]:.10g} * f[{m_k!r}] * math.exp(-f[{t_k!r}] / {tau:.10g}))"
                    )
                    best_mem = (score, tr_e, va_e, expr, [d_k, t_k, m_k])
        if best_mem is not None:
            candidates.append(("conjugate_relaxation_state_memory", 4, best_mem[1], best_mem[2], best_mem[3], best_mem[4]))

    # 3. Bimodal two-timescale viscoelastic relaxation
    if allow("bimodal_relaxation_2mode") and len(train_rows) >= 5:
        best_bi = None
        for d_k, t_k in pairs_to_try:
            for tf in (0.25, 0.35, 0.5):
                for ts in (3.0, 4.5, 6.0):
                    w = solve_linear_ls(
                        lambda f, dk=d_k, tk=t_k, t1=tf, t2=ts: [
                            float(f[dk]),
                            float(f[dk]) * math.exp(-float(f[tk]) / t1),
                            float(f[dk]) * math.exp(-float(f[tk]) / t2),
                        ],
                        train_rows,
                    )
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

    # 4. Dual-resource Liebig piecewise min(c1*x1, c2*x2) & Capped min(c1*x1, c2*x2, cap) & Cobb-Douglas
    if len(active_f) == 2:
        k1, k2 = active_f[0], active_f[1]
        if allow("liebig_min_2d"):
            # Coordinate grid + refinement on c1, c2
            ratios_1 = [r["y_lat"] / max(1e-6, float(r["features"][k1])) for r in train_rows]
            ratios_2 = [r["y_lat"] / max(1e-6, float(r["features"][k2])) for r in train_rows]
            c1_init = max(ratios_1)
            c2_init = max(ratios_2)
            best_m = None
            for s1 in [0.75 + 0.025 * i for i in range(22)]:
                for s2 in [0.75 + 0.025 * j for j in range(22)]:
                    c1, c2 = c1_init * s1, c2_init * s2
                    fn = lambda f, a=c1, b=c2, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]))
                    tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                    sc = 0.5 * tr_e + 0.5 * va_e
                    if best_m is None or sc < best_m[0]:
                        best_m = (sc, tr_e, va_e, c1, c2)
            c1, c2 = best_m[3], best_m[4]
            for step in (0.04, 0.01, 0.002):
                improved = True
                while improved:
                    improved = False
                    for dc1, dc2 in [(step, 0), (-step, 0), (0, step), (0, -step), (step, step), (-step, -step)]:
                        nc1, nc2 = max(1e-4, c1 + dc1), max(1e-4, c2 + dc2)
                        fn = lambda f, a=nc1, b=nc2, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]))
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if sc + 1e-9 < best_m[0]:
                            c1, c2 = nc1, nc2
                            best_m = (sc, tr_e, va_e, c1, c2)
                            improved = True
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
                cap_v = max_obs * cap_mul
                for s1 in [0.85 + 0.03 * i for i in range(12)]:
                    for s2 in [0.85 + 0.03 * j for j in range(12)]:
                        c1, c2 = r1 * s1, r2 * s2
                        fn = lambda f, a=c1, b=c2, cp=cap_v, x1=k1, x2=k2: min(a * float(f[x1]), b * float(f[x2]), cp)
                        tr_e, va_e = rmse_raw(fn, train_rows), rmse_raw(fn, val_rows)
                        sc = 0.5 * tr_e + 0.5 * va_e
                        if best_cm is None or sc < best_cm[0]:
                            best_cm = (sc, tr_e, va_e, c1, c2, cap_v)
            c1, c2, cap_v = best_cm[3], best_cm[4], best_cm[5]
            for step in (0.03, 0.008):
                for _ in range(15):
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
            fn = lambda f, c=w[0], x1=k1, x2=k2: c * math.sqrt(max(0.0, float(f[x1]) * float(f[x2])))
            expr = f"({w[0]:.10g} * math.sqrt(f[{k1!r}] * f[{k2!r}]))"
            candidates.append(("cobb_douglas_sqrt_2d", 1, rmse_raw(fn, train_rows), rmse_raw(fn, val_rows), expr, [k1, k2]))

if not candidates:
    raise RuntimeError("No candidate expression could be synthesized")

# Occam-penalized validation selection
candidates.sort(key=lambda item: 0.65 * item[3] + 0.35 * item[2] + 0.004 * item[1])
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

summary = {
    "selected_schema": schema_tag,
    "num_params": n_params,
    "train_rmse": round(tr_rmse, 6),
    "val_rmse": round(va_rmse, 6),
    "role_order": role_order,
    "competing_candidates": [
        {"schema": c[0], "num_params": c[1], "train_rmse": round(c[2], 6), "val_rmse": round(c[3], 6)}
        for c in candidates
    ],
}
print(json.dumps(summary))
