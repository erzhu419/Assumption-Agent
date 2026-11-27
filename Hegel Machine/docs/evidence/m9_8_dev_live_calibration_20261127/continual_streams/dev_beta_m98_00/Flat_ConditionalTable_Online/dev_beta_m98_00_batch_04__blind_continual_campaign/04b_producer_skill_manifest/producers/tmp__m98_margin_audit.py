import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
seed = np.c_[np.ones(len(x)), x]
err = np.linalg.norm(raw - seed @ np.linalg.lstsq(seed, raw, rcond=None)[0], axis=1)
clean = err <= 6 * np.median(err)
for ch in range(4):
    a = m._rosette_terms(x[clean], ch)
    y = raw[clean, ch]
    options = [(0., a)]
    for q in (a[:,1], a[:,1]+a[:,2], a[:,1]-a[:,2], a[:,1]+a[:,3], a[:,1]-a[:,3]):
        for strength in (1e-5, 1e-4, 1e-3, 1e-2):
            options.append((strength, a / (1 + strength * q[:,None]**2)))
    baseline, selected, strengths = [], [], []
    for j in range(len(y)):
        train = np.arange(len(y)) != j
        scores, predictions = [], []
        for strength, design in options:
            fit = design[train]
            residual = y[train] - fit @ np.linalg.lstsq(fit, y[train], rcond=None)[0]
            leverage = np.sum(np.linalg.qr(fit, mode='reduced')[0]**2, axis=1)
            scores.append(np.mean((residual/(1-leverage))**2))
            predictions.append(design[j] @ np.linalg.lstsq(fit, y[train], rcond=None)[0])
        k = int(np.argmin(scores))
        baseline.append(predictions[0]-y[j]); selected.append(predictions[k]-y[j]); strengths.append(options[k][0])
    print(f'channel {ch}: clean sweep max LOO cubic={max(np.abs(baseline)):.3g}, nested rational={max(np.abs(selected)):.3g}, nonzero rational selections={np.count_nonzero(strengths)}/{len(y)}')

bank = c['single_bank_sweeps']
coef = m._fit_rosette(x, raw)
for ch in range(4):
    mask = bank['bank_id'] == ch
    refs = np.zeros((mask.sum(),4)); refs[:,ch] = bank['ref_signal'][mask]
    e = np.sort(np.abs(m._rosette_terms(refs,ch) @ coef[:,ch] - bank['raw_readout'][mask]))
    print(f'bank {ch}: sorted independent sweep absolute errors {np.array2string(e, precision=4)}')

D, _ = m._design(s); y = m._calibrate(s); Z = s['Z_nuisance']
ns = len(s['mu_shifts']); groups = np.arange(len(y)).reshape(ns,-1,4).transpose(1,0,2)
for profile in (False, True):
    errors = []
    for block in groups:
        test = block.ravel(); keep = np.ones(len(y), bool); keep[test] = False
        a, b, z = D[keep], y[keep], Z[keep]
        if profile:
            p = np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
            T, r = a-z@(p@a), b-z@(p@b)
        else:
            T, r = a, b
        w, _ = m._box_qp(T.T@T+s['source_ridge']*np.eye(T.shape[1]), -T.T@r, s['u_box'], s['total_budget'])
        pred = D[test]@w
        if profile:
            pred += Z[test]@(p@(b-a@w))
        errors.append((pred-y[test]).reshape(ns,4)**2)
    e = np.array(errors)
    print(f'profile={profile}: shift RMSE={np.sqrt(e.mean(axis=(0,2)))}; worst shift/location RMSE={np.sqrt(e.mean(axis=2)).max():.6g}')
