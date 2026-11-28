import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
sw = study['sensor_calibration_sweeps']
x, y = sw['ref_rosette'], sw['raw_rosette']
clean = np.max(np.abs(y), axis=1) < 5
obs = study['y_obs_raw'].reshape(-1, 4)
base = up._fit_forward(x, y)
base_obs = up._invert(obs, base)

def compact(ref, raw):
    a = up._features(ref)
    c = np.zeros((a.shape[1], 4))
    for j in range(4):
        k = (j + 1) % 4
        cols = [0, 1+j, 1+k, 1+(j-1)%4, 5+j, 9+up._PAIRS.index(tuple(sorted((j,k))))]
        b = a[:, cols]
        w = np.ones(len(b))
        for _ in range(24):
            v = np.linalg.lstsq(b*w[:, None], raw[:, j]*w, rcond=None)[0]
            e = np.abs(b @ v - raw[:, j])
            scale = max(np.median(e), 1e-12)
            w = 1/np.hypot(1, e/(2*scale))
        keep = e <= 6*scale
        if np.linalg.matrix_rank(b[keep]) == len(cols):
            v = np.linalg.lstsq(b[keep], raw[keep, j], rcond=None)[0]
        c[cols, j] = v
    return c

rows = np.arange(len(x))
print('EMI indices:', rows[~clean].tolist())
for name, fit in [('delivered', up._fit_forward), ('compact cyclic alternative', compact)]:
    maxima = []
    for block in np.array_split(rows, 8):
        train = ~np.isin(rows, block)
        test = clean & ~train
        c = fit(x[train], y[train])
        worst = np.max(np.abs(up._invert(y[test], c) - x[test]))
        shift = np.max(np.abs(up._invert(obs, c) - base_obs))
        maxima.append((worst, shift, int(block[0])))
    c = fit(x, y)
    no_emi = fit(x[clean], y[clean])
    emi_shift = np.max(np.abs(up._invert(obs, c) - up._invert(obs, no_emi)))
    axes = [np.linspace(min(a,b),max(d,e),9) for a,b,d,e in zip(x.min(0),base_obs.min(0),x.max(0),base_obs.max(0))]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'),axis=-1).reshape(-1, 4)
    jacs = np.array([up._jacobian(p,c) for p in grid])
    min_sv = np.linalg.svd(jacs,compute_uv=False)[:,-1].min()
    grid_err = np.max(np.abs(up._invert(up._features(grid) @ c,c)-grid))
    print(name, 'worst contiguous holdout (inverse error, obs shift, block start):', max(maxima), 'max obs shift:', max(m[1] for m in maxima), 'EMI-induced obs shift:', emi_shift, 'grid min singular value:', min_sv, 'grid inverse error:', grid_err)
    if name == 'delivered':
        assert max(maxima)[0] < 1e-4 and emi_shift < 1e-5 and grid_err < 1e-9
