import importlib.util, json, pickle
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
root = Path('/holdout_task')
s = importlib.util.spec_from_file_location('solver', root/'batch_solver.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = pickle.loads((root/'public_batch.pkl').read_bytes())[0]['dataset']
c = d['calibration_load_cells']; raw, true = c['bridge_raw'], c['true_channel_loads']; basis = m._bridge_basis(true)
coef = m._fit_bridge(raw, true); resid = raw - (basis*coef).sum(-1)
clean = np.linalg.norm(resid, axis=1) < 0.001
print('Rejected sweep count:', (~clean).sum(), 'max absolute raw sum deviation on rejected sweeps:', np.max(np.abs(resid[~clean].sum(1))))
rng = np.random.default_rng(17); folds = np.array_split(rng.permutation(len(raw)), 8)
e_aff, e_cyc, e_rat = [np.zeros_like(raw) for _ in range(3)]
for h in folds:
    keep = np.ones(len(raw), bool); keep[h] = False
    b = m._fit_bridge(raw[keep], true[keep]); fitted = m._channel_loads(raw[h[clean[h]]], b)
    e_cyc[h[clean[h]]] = fitted - true[h[clean[h]]]
    affine = np.linalg.lstsq(np.c_[raw[keep], np.ones(sum(keep))], true[keep], rcond=None)[0]
    e_aff[h] = np.c_[raw[h], np.ones(len(h))] @ affine - true[h]
    good = keep & (np.linalg.norm(raw - (basis*b).sum(-1), axis=1) < 0.001)
    # Test saturating adjacent-channel difference, nested around the cyclic cubic model.
    def forward(v, b, sat):
        B = m._bridge_basis(v)
        out = (B*b).sum(-1)
        return out + b[:,4]*B[...,4]*(1/(1+sat*(B[...,1]-B[...,3])**2)-1)
    p = least_squares(lambda p: (forward(true[good],p[:24].reshape(4,6),p[24:])-raw[good]).ravel(), np.r_[b.ravel(), np.zeros(4)], bounds=(np.r_[np.full(24,-np.inf),np.zeros(4)], np.full(28,np.inf)), gtol=1e-10).x
    for j,v in zip(h[clean[h]], fitted):
        e_rat[j] = least_squares(lambda v: forward(v,p[:24].reshape(4,6),p[24:])-raw[j],v,gtol=1e-12).x - true[j]
for mask,name in [(clean & (np.count_nonzero(true,axis=1)==1),'single'),(clean & (np.count_nonzero(true,axis=1)==4),'mixed'),(clean & (np.max(abs(true),axis=1)>1.25),'extreme')]:
    print(name,int(mask.sum()),'axial RMSE affine/cyclic/rational',*[f'{np.sqrt(np.mean(e[mask].sum(1)**2)):.8g}' for e in (e_aff,e_cyc,e_rat)],'cyclic max channel',np.max(np.abs(e_cyc[mask])))
entries=json.loads(Path('/tmp/hegel_continual_archive/archive_apparatus_log.json').read_text()); traces=np.array([e['facility_telemetry_trace'] for e in entries]); labels=[e['completed_short_name'] for e in entries[:-1]]
for cols in ([1,4,7],[2],[5],[1,2,4,5,7]):
    levels=np.median(traces[:,:,cols],axis=1); dist=np.linalg.norm(levels[:-1]-levels[-1],axis=1); print('lineage channels',cols,dict(zip(labels,dist.round(4))))
