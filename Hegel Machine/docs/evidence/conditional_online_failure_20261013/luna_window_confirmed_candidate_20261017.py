import numpy as np

_COEF = None
_THRESHOLD = 0.0
_DEGREE = 4

def _terms(c):
    x, z = float(c['x']), float(c['z'])
    return np.asarray([x**i * z**j for d in range(_DEGREE + 1) for i in range(d + 1) for j in [d - i]], dtype=float)

def _basis(c, t):
    v = _terms(c)
    low = float(float(c['x']) < t)
    # Separate polynomial surfaces on either side of the x threshold.
    return np.concatenate([v, low * v, (1.0 - low) * v])

def _design(row, t):
    controls = list(row.get('history', [])[-2:]) + [row['features']]
    return sum((_basis(c, t) for c in controls), np.zeros(3 * len(_terms({'x': 0.0, 'z': 0.0}))))

def fit(observations):
    global _COEF, _THRESHOLD
    y = np.asarray([float(r['target']) for r in observations])
    xs = np.asarray([float(r['features']['x']) for r in observations])
    thresholds = np.unique(np.quantile(xs, np.linspace(0.02, 0.98, 121)))
    best = None
    for t in thresholds:
        A = np.vstack([_design(r, float(t)) for r in observations])
        coef = np.linalg.lstsq(A, y, rcond=1e-10)[0]
        loss = float(np.mean((A @ coef - y) ** 2))
        if best is None or loss < best[0]:
            best = (loss, float(t), coef)
    _, _THRESHOLD, _COEF = best

def predict(row):
    if _COEF is None:
        raise RuntimeError('fit must be called before predict')
    return float(_design(row, _THRESHOLD) @ _COEF)
