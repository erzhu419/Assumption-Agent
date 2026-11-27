import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)

for item in items:
    s = item['study']
    R = s['sensor_calibration_sweeps']['ref_rosette']
    Y = s['sensor_calibration_sweeps']['raw_rosette']
    S, _ = m._forward(s)
    folds = np.random.default_rng(2401).permutation(len(R)) % 4
    def fit(x,y,diagonal):
        if diagonal:
            c = np.array([np.linalg.lstsq(np.c_[np.ones(len(x)),a], b, rcond=None)[0] for a,b in zip(x.T,y.T)])
            return c[:,0],np.diag(c[:,1])
        c = np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]
        return c[0],c[1:]
    for diagonal in (False,True):
        held = []
        for fold in range(4):
            test = folds == fold
            b,A = fit(R[~test],Y[~test],diagonal)
            held.extend(np.sqrt(np.mean((R[test]@A+b-Y[test])**2,axis=0)))
        b,A = fit(R,Y,diagonal)
        pulse = []
        for p in item.get('campaign_diagnostics') or []:
            raw = p['y_pulse_raw']
            cal = np.linalg.solve(A.T,(raw.reshape(-1,4)-b).T).T.ravel() if diagonal else m._calibrate(dict(s,y_obs_raw=raw))
            err = (S@p['w_pulse']-cal).reshape(len(s['mu_shifts']),-1)
            labels = np.asarray(s['row_bank_id']).reshape(err.shape)
            pulse.extend(np.sqrt(np.mean(e[lab==j]**2)) for e,lab in zip(err,labels) for j in np.unique(lab))
        print('diagonal' if diagonal else 'full affine', 'sweep worst/RMS',max(held),np.linalg.norm(held)/np.sqrt(len(held)), 'pulse worst/RMS',max(pulse,default=float('nan')),np.linalg.norm(pulse)/np.sqrt(len(pulse)) if pulse else float('nan'))
    b,A = fit(R,Y,False)
    print('largest absolute off-diagonal mixing coefficient',np.max(np.abs(A-np.diag(np.diag(A)))))
