open('model.py', 'w', encoding='utf-8').write('import math\n\n_SCHEMA_FAMILY_TAG = \'atomic_michaelis_1d\'\n_CORE_LAW_EXPR = "((_PARAMS[\'p0\'] * f[\'drive_load\']) / (_PARAMS[\'p1\'] + f[\'drive_load\']))"\n_PARAMS = {\'p0\': 6, \'p1\': 2}\n\ndef fit(observations):\n    if not observations:\n        return\n    rows_f = [r[\'features\'] for r in observations]\n    ys = [(float(r[\'target\']) - (14)) / ((-1) * (1.25)) for r in observations]\n    lbs = [0.0001, 0.0001]\n    ubs = [500.0, 500.0]\n    m_p = len(lbs)\n    def _res(p):\n        out = []\n        for f, y in zip(rows_f, ys):\n            out.append((((p[0] * f[\'drive_load\']) / (p[1] + f[\'drive_load\']))) - y)\n        return out\n    def _loss(p):\n        rv = _res(p)\n        return sum(e * e for e in rv)\n    def _lm(x_start):\n        p = [max(lbs[i], min(ubs[i], float(x_start[i]))) for i in range(m_p)]\n        cur = _loss(p)\n        lam = 1e-3\n        for _ in range(28):\n            if cur <= 1e-13:\n                break\n            r0 = _res(p)\n            jac = []\n            for j in range(m_p):\n                h = max(1e-6, 1e-5 * abs(p[j]))\n                pj = min(ubs[j], p[j] + h)\n                actual_h = pj - p[j]\n                if abs(actual_h) < 1e-9:\n                    pj = max(lbs[j], p[j] - h)\n                    actual_h = pj - p[j]\n                p_pert = list(p)\n                p_pert[j] = pj\n                rp = _res(p_pert)\n                jac.append([(rp[k] - r0[k]) / actual_h for k in range(len(r0))])\n            jtj = [[(lam if a == b else 0.0) + sum(jac[a][k] * jac[b][k] for k in range(len(r0))) for b in range(m_p)] for a in range(m_p)]\n            jtr = [-sum(jac[a][k] * r0[k] for k in range(len(r0))) for a in range(m_p)]\n            aug = [jtj[a] + [jtr[a]] for a in range(m_p)]\n            ok = True\n            for col in range(m_p):\n                piv = max(range(col, m_p), key=lambda r_i: abs(aug[r_i][col]))\n                aug[col], aug[piv] = aug[piv], aug[col]\n                if abs(aug[col][col]) <= 1e-15:\n                    ok = False\n                    break\n                inv_d = 1.0 / aug[col][col]\n                for c2 in range(col, m_p + 1):\n                    aug[col][c2] *= inv_d\n                for r_i in range(m_p):\n                    if r_i != col and abs(aug[r_i][col]) > 0.0:\n                        fac = aug[r_i][col]\n                        for c2 in range(col, m_p + 1):\n                            aug[r_i][c2] -= fac * aug[col][c2]\n            if not ok:\n                lam *= 10.0\n                continue\n            dp = [aug[a][m_p] for a in range(m_p)]\n            cand = [max(lbs[i], min(ubs[i], p[i] + dp[i])) for i in range(m_p)]\n            c_loss = _loss(cand)\n            if c_loss < cur:\n                impr = cur - c_loss\n                p, cur = cand, c_loss\n                lam = max(1e-7, lam * 0.35)\n                if impr < 1e-12:\n                    break\n            else:\n                lam = min(1e6, lam * 6.0)\n        for step in (0.15, 0.03, 0.005):\n            for j in range(m_p):\n                for sgn in (-1.0, 1.0):\n                    cand = list(p)\n                    cand[j] = max(lbs[j], min(ubs[j], p[j] + sgn * step * max(0.5, abs(p[j]))))\n                    cl = _loss(cand)\n                    if cl < cur:\n                        p, cur = cand, cl\n        return cur, p\n    starts = [[6.0, 2.0]]\n    if ys:\n        if False and m_p >= 2:\n            s_emp = list(starts[0])\n            s_emp[-1] = max(lbs[-1], min(ubs[-1], max(ys)))\n            starts.insert(0, s_emp)\n    best_cost, best_p = None, list(starts[0])\n    for st in starts:\n        c_val, c_p = _lm(st)\n        if best_cost is None or c_val < best_cost:\n            best_cost, best_p = c_val, c_p\n            if best_cost <= 1e-10:\n                break\n    _PARAMS[\'p0\'] = float(best_p[0])\n    _PARAMS[\'p1\'] = float(best_p[1])\n\ndef predict(row):\n    f = row[\'features\']\n    latent = ((_PARAMS[\'p0\'] * f[\'drive_load\']) / (_PARAMS[\'p1\'] + f[\'drive_load\']))\n    return 14 + (-1) * (1.25) * latent\n')
import json, math, statistics
import model as current_model

rows = json.loads(open('model_observations.json', encoding='utf-8').read())
if hasattr(current_model, 'fit'):
    current_model.fit(rows)

unlensed = []
resids = []
for r in rows:
    f = r['features']
    x0 = float(f['drive_load'])
    x1 = float(f['bypass_flow'])
    y_raw = float(r['target'])
    p_raw = float(current_model.predict(r))
    y_u = (y_raw - (14)) / ((-1) * (1.25))
    p_u = (p_raw - (14)) / ((-1) * (1.25))
    unlensed.append((x0, x1, y_u, p_u))
    resids.append(y_u - p_u)

by_x0 = sorted(unlensed, key=lambda t: t[0])
n = len(by_x0)
ys_x0 = [t[2] for t in by_x0]
peak_y = max(ys_x0) if ys_x0 else 0.0
peak_idx = ys_x0.index(peak_y) if ys_x0 else 0
end_y = ys_x0[-1] if ys_x0 else 0.0

# 1. High-drive turnover drop (Haldane / pore-collapse signature)
high_drive_turnover_drop = bool(
    n >= 5 and peak_idx < n - 1 and (peak_y - end_y) > 0.10 * max(abs(peak_y), 1e-6)
    and (by_x0[-1][2] - by_x0[-1][3]) < -0.06 * max(abs(peak_y), 1e-6)
)

# 2. Sharp variational ceiling plateau (top >= 2 points nearly constant while base model rises)
top_k = by_x0[max(0, n - 3):]
top_ys = [t[2] for t in top_k]
top_ps = [t[3] for t in top_k]
top_span = (max(top_ys) - min(top_ys)) if len(top_ys) >= 2 else 999.0
pred_rise = (top_ps[-1] - top_ps[0]) if len(top_ps) >= 2 else 0.0
kink_plateau_ceiling = bool(
    not high_drive_turnover_drop
    and len(top_ys) >= 3
    and top_span <= 0.035 * max(abs(peak_y), 1e-6)
    and (
        pred_rise > 0.015 * max(abs(peak_y), 1e-6)
        or abs(top_ps[-1] - top_ys[-1]) > 0.03 * max(abs(peak_y), 1e-6)
    )
)

# 3. Secondary channel partial residual correlation (after detrending primary curve in x0!)
secondary_channel_corr = 0.0
secondary_active = False
if True and len(unlensed) >= 5:
    xs0_v = [t[0] for t in unlensed]
    xs1_v = [t[1] for t in unlensed]
    ys_v = [t[2] for t in unlensed]
    if max(xs1_v) - min(xs1_v) > 1e-6:
        # 3x3 OLS quadratic detrend on x0 so primary curvature never masks secondary channel x1
        b_mat = [[1.0, x, x * x] for x in xs0_v]
        ata = [[(1e-8 if i == j else 0.0) + sum(b[i] * b[j] for b in b_mat) for j in range(3)] for i in range(3)]
        aty = [sum(b[i] * y for b, y in zip(b_mat, ys_v)) for i in range(3)]
        aug = [ata[i] + [aty[i]] for i in range(3)]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r_i: abs(aug[r_i][col]))
            aug[col], aug[piv] = aug[piv], aug[col]
            if abs(aug[col][col]) > 1e-12:
                inv_d = 1.0 / aug[col][col]
                for c2 in range(col, 4):
                    aug[col][c2] *= inv_d
                for r_i in range(3):
                    if r_i != col:
                        fac = aug[r_i][col]
                        for c2 in range(col, 4):
                            aug[r_i][c2] -= fac * aug[col][c2]
        w_q = [aug[i][3] for i in range(3)]
        det_res = [y - (w_q[0] + w_q[1] * x + w_q[2] * x * x) for x, y in zip(xs0_v, ys_v)]
        # Use whichever residual stream has smaller variance (best base model vs quadratic detrend)
        cand_res = resids if sum(r * r for r in resids) < sum(r * r for r in det_res) else det_res
        m1 = sum(xs1_v) / len(xs1_v)
        mr = sum(cand_res) / len(cand_res)
        cov = sum((a - m1) * (b - mr) for a, b in zip(xs1_v, cand_res))
        v1 = sum((a - m1) ** 2 for a in xs1_v)
        vr = sum((b - mr) ** 2 for b in cand_res)
        if v1 > 1e-12 and vr > 1e-12:
            secondary_channel_corr = cov / math.sqrt(v1 * vr)
            secondary_active = bool(abs(secondary_channel_corr) > 0.62 and (vr / len(cand_res)) > 0.003)

# 4. Oscillatory overshoot vs Fast-mode short-time adiabatic decay along time axis
oscillatory_overshoot = False
fast_transient_decay = False
if True:
    by_t = sorted(unlensed, key=lambda t: t[1])
    t_vals = [t[1] for t in by_t]
    norm_yt = [t[2] / max(t[0], 1e-6) for t in by_t if t[0] > 1e-4]
    if len(norm_yt) >= 5 and (max(t_vals) - min(t_vals)) > 2.5:
        late_asym = sum(norm_yt[-2:]) / 2.0
        interior_max = max(norm_yt[1:])
        early_nt = norm_yt[0]
        if early_nt < late_asym * 0.90 and interior_max > late_asym * 1.06:
            oscillatory_overshoot = True
        elif early_nt > late_asym * 1.15 and early_nt > norm_yt[-1] + 0.12:
            fast_transient_decay = True

report = {
    'high_drive_turnover_drop': high_drive_turnover_drop,
    'kink_plateau_ceiling': kink_plateau_ceiling,
    'secondary_channel_corr': round(secondary_channel_corr, 4),
    'secondary_active': secondary_active,
    'oscillatory_overshoot': oscillatory_overshoot,
    'fast_transient_decay': fast_transient_decay,
    'residual_rmse': round(math.sqrt(sum(r * r for r in resids) / max(1, len(resids))), 6),
}
print(json.dumps(report, sort_keys=True))
