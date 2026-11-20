import math
_PARAMS = {'intercept': 0.0, 'slope': 1.0}
def fit(observations):
    if not observations:
        return
    xs, ys = [], []
    for r in observations:
        u = float(r['features']['u_drive'])
        t = float(r['features']['T_horizon'])
        ym = u * (2.26724920557144 + -1.54313533827954 * math.exp(-0.766096974033998 * t) + 0.355886506689266 * math.exp(-9.01446728928889 * t))
        xs.append(ym)
        ys.append(float(r['target']))
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sum((x - mx) ** 2 for x in xs) + 1e-12)
    _PARAMS['slope'] = sl
    _PARAMS['intercept'] = my - sl * mx
def predict(row):
    f = row.get('features', row)
    u = float(f['u_drive'])
    t = float(f['T_horizon'])
    ym = u * (2.26724920557144 + -1.54313533827954 * math.exp(-0.766096974033998 * t) + 0.355886506689266 * math.exp(-9.01446728928889 * t))
    return _PARAMS['intercept'] + _PARAMS['slope'] * ym
