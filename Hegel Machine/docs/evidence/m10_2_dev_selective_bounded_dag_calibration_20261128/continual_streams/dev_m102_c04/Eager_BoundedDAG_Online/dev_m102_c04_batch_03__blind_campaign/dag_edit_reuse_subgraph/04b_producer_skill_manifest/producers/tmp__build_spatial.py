from pathlib import Path
import ast
src = Path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py').read_text()
parts = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
header = '"""Spatial calibration and prediction for the frozen trajectory DAG."""\nimport numpy as np\nfrom scipy.optimize import least_squares\n'
body = '''

def _calibrate(panel):
    """Fit offsets, separate gains, and shared cyclic shear coefficients."""
    u, v = np.asarray(panel['u'], float), np.asarray(panel['v'], float)
    n, d = u.shape
    j = np.arange(d)
    nxt = np.roll(u, -1, axis=1)
    h = np.zeros((n, d, 2*d + 3))
    h[:, j, j] = 1
    h[:, j, d+j] = u
    h[:, :, -3] = nxt
    h[:, :, -2] = u*nxt
    h[:, :, -1] = u**3
    keep = np.ones(n, dtype=bool)
    for _ in range(12):
        w = _ls(h[keep].reshape(-1, h.shape[-1]), v[keep].ravel())
        err = np.linalg.norm(h @ w - v, axis=1)
        updated = err <= max(5*np.median(err), 1e-8)
        if np.array_equal(keep, updated) or updated.sum() < d+4:
            break
        keep = updated
    return _ls(h[keep].reshape(-1, h.shape[-1]), v[keep].ravel())


def _inverse(x, w):
    d = x.shape[1]
    j = np.arange(d)
    k = (j+1) % d
    offset, gain = w[:d], w[d:2*d]
    linear, shear, cubic = w[-3:]
    def forward(z):
        return offset + gain*z + (linear + shear*z)*z[k] + cubic*z**3
    def jac(z):
        g = np.zeros((d, d))
        g[j, j] = gain + shear*z[k] + 3*cubic*z**2
        g[j, k] = linear + shear*z
        return g
    initial = np.linalg.solve(jac(np.zeros(d)), (x-offset).T).T
    return np.array([
        least_squares(lambda z: forward(z)-row, guess, jac=jac,
                      gtol=1e-12, max_nfev=60).x
        for row, guess in zip(x, initial)
    ]).reshape(x.shape)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    z = _inverse(x, _calibrate(study['panel_a']))
    pop, rop = _panel_operators(study['panel_c'])
    common, offdiag = _features(pop(z), rop(z))
    n = len(train['x'])
    y_hat = _ridge(common[:n], train['y'], common[n:], 1e-6)
    return {'common': common, 'offdiag': offdiag, 'y_hat': y_hat}
'''
Path('/holdout_task/node_00.py').write_text(header + '\n\n'.join(parts[k] for k in ('_affine', '_ls', '_panel_operators', '_features', '_ridge')) + '\n' + body)
