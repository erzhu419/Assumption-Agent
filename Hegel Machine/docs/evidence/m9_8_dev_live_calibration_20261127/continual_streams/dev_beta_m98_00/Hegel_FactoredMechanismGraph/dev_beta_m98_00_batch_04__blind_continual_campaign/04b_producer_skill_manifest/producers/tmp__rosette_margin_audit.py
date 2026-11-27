import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up

item = pickle.load(open('public_batch.pkl', 'rb'))[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
clean = np.max(np.abs(y), axis=1) < 8  # Public EMI rows are visibly separated.
axial = np.count_nonzero(x, axis=1) == 1
edge = np.max(np.abs(x), axis=1) >= np.quantile(np.max(np.abs(x), axis=1), .8)
print('study regime fields:', [k for k in s if any(t in k for t in ('batch', 'relay', 'meter', 'cohort', 'bank'))])
print('diagnostics present:', item['campaign_diagnostics'] is not None)

def rms(a):
    return np.sqrt(np.mean(a**2))

# Compare prespecified nonlinear extensions on the same held-out rows. The
# baseline support and training outlier mask are selected inside each fold.
for name, hold in [('all axial', axial), ('outer radius', edge), ('positive/negative difference tail', np.abs(x[:, 0]-x[:, 1]) >= np.quantile(np.abs(x[:, 0]-x[:, 1]), .8))]:
    train, val = ~hold, hold & clean
    coef = up._fit(x[train], y[train])
    a = up._features(x)
    err = np.linalg.norm(a @ coef - y, axis=1)
    use = train & (err <= 6 * np.median(err[train]))
    inv = np.linalg.norm(up._invert(y[val], coef) - x[val], axis=1).max()
    scores = {}
    for kind in ('coordinate rational', 'pair difference rational', 'three-way'):
        pred = np.empty_like(y[val])
        for j in range(4):
            base = a[:, coef[:, j] != 0]
            q = x[:, j] - x[:, (j+1)%4]
            if kind == 'coordinate rational':
                z = x[:, j]**3 / (1 + .5*x[:, j]**2)
            elif kind == 'pair difference rational':
                z = q / (1 + .5*q**2)
            else:
                z = x[:, j] * x[:, (j+1)%4] * x[:, (j+2)%4]
            aug = np.column_stack((base, z))
            pred[:, j] = aug[val] @ np.linalg.lstsq(aug[use], y[use, j], rcond=None)[0]
        scores[kind] = rms(pred - y[val])
    print(name, 'n:', val.sum(), 'baseline forward RMS:', rms((a @ coef - y)[val]), 'inverse worst L2:', inv, 'extension forward RMS:', scores)
    assert inv < 2e-4

# Detect actual observation rows with a poorly conditioned or stalled inverse.
coef = up._fit(x, y)
raw = np.asarray(s['y_obs_raw']).reshape(-1, 4)
z = up._invert(raw, coef)
h = 1e-5
jac = np.stack([(up._features(z+d) - up._features(z-d)) @ coef / (2*h) for d in h*np.eye(4)], axis=-1)
min_sigma = np.linalg.svd(jac, compute_uv=False)[:, -1].min()
replay = np.abs(up._features(z) @ coef - raw).max()
print('observation inverse: min Jacobian singular value:', min_sigma, 'max replay residual:', replay)
assert min_sigma > .1 and replay < 1e-10
