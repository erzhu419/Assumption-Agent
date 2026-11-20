import math

_SCHEMA_FAMILY_TAG = 'atomic_murray_2_3__comb_shunt'
_CORE_LAW_EXPR = "((_PARAMS['p0'] * (f['feed_conc'] ** (2.0 / 3.0))) + (_PARAMS['p1'] * f['recycle_bleed']))"
_PARAMS = {'p0': 1.8, 'p1': 0.6}

def fit(observations):
    if not observations:
        return
    rows_f = [r['features'] for r in observations]
    ys = [(float(r['target']) - (3.310159548)) / ((1) * (1)) for r in observations]
    lbs = [0.0001, -80.0]
    ubs = [300.0, 80.0]
    m_p = len(lbs)
    def _res(p):
        out = []
        for f, y in zip(rows_f, ys):
            out.append((((p[0] * (f['feed_conc'] ** (2.0 / 3.0))) + (p[1] * f['recycle_bleed']))) - y)
        return out
    def _loss(p):
        rv = _res(p)
        return sum(e * e for e in rv)
    def _lm(x_start):
        p = [max(lbs[i], min(ubs[i], float(x_start[i]))) for i in range(m_p)]
        cur = _loss(p)
        lam = 1e-3
        for _ in range(28):
            if cur <= 1e-13:
                break
            r0 = _res(p)
            jac = []
            for j in range(m_p):
                h = max(1e-6, 1e-5 * abs(p[j]))
                pj = min(ubs[j], p[j] + h)
                actual_h = pj - p[j]
                if abs(actual_h) < 1e-9:
                    pj = max(lbs[j], p[j] - h)
                    actual_h = pj - p[j]
                p_pert = list(p)
                p_pert[j] = pj
                rp = _res(p_pert)
                jac.append([(rp[k] - r0[k]) / actual_h for k in range(len(r0))])
            jtj = [[(lam if a == b else 0.0) + sum(jac[a][k] * jac[b][k] for k in range(len(r0))) for b in range(m_p)] for a in range(m_p)]
            jtr = [-sum(jac[a][k] * r0[k] for k in range(len(r0))) for a in range(m_p)]
            aug = [jtj[a] + [jtr[a]] for a in range(m_p)]
            ok = True
            for col in range(m_p):
                piv = max(range(col, m_p), key=lambda r_i: abs(aug[r_i][col]))
                aug[col], aug[piv] = aug[piv], aug[col]
                if abs(aug[col][col]) <= 1e-15:
                    ok = False
                    break
                inv_d = 1.0 / aug[col][col]
                for c2 in range(col, m_p + 1):
                    aug[col][c2] *= inv_d
                for r_i in range(m_p):
                    if r_i != col and abs(aug[r_i][col]) > 0.0:
                        fac = aug[r_i][col]
                        for c2 in range(col, m_p + 1):
                            aug[r_i][c2] -= fac * aug[col][c2]
            if not ok:
                lam *= 10.0
                continue
            dp = [aug[a][m_p] for a in range(m_p)]
            cand = [max(lbs[i], min(ubs[i], p[i] + dp[i])) for i in range(m_p)]
            c_loss = _loss(cand)
            if c_loss < cur:
                impr = cur - c_loss
                p, cur = cand, c_loss
                lam = max(1e-7, lam * 0.35)
                if impr < 1e-12:
                    break
            else:
                lam = min(1e6, lam * 6.0)
        for step in (0.15, 0.03, 0.005):
            for j in range(m_p):
                for sgn in (-1.0, 1.0):
                    cand = list(p)
                    cand[j] = max(lbs[j], min(ubs[j], p[j] + sgn * step * max(0.5, abs(p[j]))))
                    cl = _loss(cand)
                    if cl < cur:
                        p, cur = cand, cl
        return cur, p
    starts = [[1.8, 0.6]]
    if ys:
        if False and m_p >= 2:
            s_emp = list(starts[0])
            s_emp[-1] = max(lbs[-1], min(ubs[-1], max(ys)))
            starts.insert(0, s_emp)
    best_cost, best_p = None, list(starts[0])
    for st in starts:
        c_val, c_p = _lm(st)
        if best_cost is None or c_val < best_cost:
            best_cost, best_p = c_val, c_p
            if best_cost <= 1e-10:
                break
    _PARAMS['p0'] = float(best_p[0])
    _PARAMS['p1'] = float(best_p[1])

def predict(row):
    f = row['features']
    latent = ((_PARAMS['p0'] * (f['feed_conc'] ** (2.0 / 3.0))) + (_PARAMS['p1'] * f['recycle_bleed']))
    return 3.310159548 + (1) * (1) * latent
