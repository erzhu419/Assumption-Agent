import importlib.util
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as base
spec = importlib.util.spec_from_file_location('challenger', '/tmp/quintic_trial/upstream_transducer.py')
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)
for item in pickle.load(open('/holdout_task/public_batch.pkl', 'rb')):
    s = item['study']; c = s['sensor_calibration_sweeps']; b = c['single_bank_sweeps']
    x, y = c['ref_rosette'], c['raw_rosette']; j, u, v = (b[k] for k in ('bank_id', 'ref_signal', 'raw_readout'))
    good = base._clean(y); bg = np.zeros(len(v), bool)
    for h in range(4): bg[j == h] = base._clean(v[j == h, None])
    amp = np.max(abs(x), axis=1)
    for m in (base, trial):
        err = np.zeros(len(x)); bank = np.zeros(len(v)); closure = 0.; conditioning = []; box_margin = []
        for fold in range(7):
            tr = np.arange(len(x)) % 6 != fold; bt = np.arange(len(v)) % 6 != fold
            sh, cf = m._fit(dict(ref_rosette=x[tr], raw_rosette=y[tr], single_bank_sweeps={k: a[bt] for k, a in b.items()}))
            a, d = sh
            z = np.linspace(-1, 1, 801)[:, None] * np.max(abs(x[good]), axis=0)
            diag = a*cf[:,1]*(1-np.tanh(a*z)**2) + cf[:,2] + 3*cf[:,3]*z*z
            if cf.shape[1] == 7: diag += 5*cf[:,4]*z**4
            off = cf[:,-2] + 2*d*cf[:,-1]*z*(1-np.tanh(d*z*z)**2)
            # Each 2x2 pair is nonsingular over the sweep box if the diagonal
            # product lower bound exceeds the off-diagonal product upper bound.
            lo = diag.min(0); hi = np.abs(off).max(0)
            box_margin.append(float(np.min((lo*lo[m._PAIR]-hi*hi[m._PAIR])[:2])))
            conditioning.append(float(lo.min()))
            if fold == 6: break
            te = ~tr & good; be = ~bt & bg
            pred = m._inverse(y[te], sh, cf)
            err[te] = np.max(abs(pred-x[te]), axis=1)
            closure = max(closure, float(np.max(abs(m._forward(pred, sh, cf)-y[te]))))
            bx = np.zeros((be.sum(), 4)); idx = np.arange(len(bx)); bx[idx,j[be]] = u[be]
            bank[be] = abs(m._forward(bx, sh, cf)[idx,j[be]]-v[be])
        print(m.__name__, 'central/mid/edge inverse max', [float(err[q & good].max()) for q in (amp <= .75, (amp > .75) & (amp <= 1.4), amp > 1.4)], 'edge p95', np.quantile(err[(amp > 1.4) & good], .95), 'bank max', bank.max(), 'closure', closure, 'sampled box minimum diagonal/pair margin', min(conditioning), min(box_margin))
