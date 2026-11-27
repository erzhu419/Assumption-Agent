import glob, importlib.util, pickle
import numpy as np
p='/tmp/hegel_continual_archive/'
def load(pattern):
    spec=importlib.util.spec_from_file_location('m',glob.glob(p+pattern)[0]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
mods={k:load(s) for k,s in [('robin','*s00_solver.py'),('sealed','*01_solver.py'),('periodic','*03_solver.py'),('nonlinear','*06_solver.py')]}
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; sweeps=s['sensor_calibration_sweeps']; x,y=sweeps['ref_rosette'],sweeps['raw_rosette']; Z=s['Z_nuisance']; n=len(x); idx=np.arange(n)
def fit(x,y,nonlinear):
    X=np.c_[np.ones(len(x)),x]; b=np.linalg.lstsq(X,y,rcond=None)[0]; err=np.linalg.norm(y-X@b,axis=1); keep=err<=max(6*np.median(err),1e-12)
    if nonlinear: return mods['nonlinear']._fit_rosette(x,y)
    return np.linalg.lstsq(X[keep],y[keep],rcond=None)[0]
for nonlinear in [False,True]:
    errs=[]
    for fold in range(5):
        test=idx%5==fold; b=fit(x[~test],y[~test],nonlinear)
        pred=np.column_stack([mods['nonlinear']._rosette_terms(x[test],i)@b[:,i] for i in range(4)]) if nonlinear else np.c_[np.ones(test.sum()),x[test]]@b
        errs.extend(np.linalg.norm(y[test]-pred,axis=1))
    print('cal nonlinear=',nonlinear,'error q50/80/90/max',np.quantile(errs,[.5,.8,.9,1])); print('coefficients',fit(x,y,nonlinear).round(6))
    for boundary in ('robin','sealed','periodic'):
        D,_=mods[boundary]._design(s); stats=[]
        for pulse in a['campaign_diagnostics']:
            if nonlinear: yc=mods['nonlinear']._calibrate(dict(s,y_obs_raw=pulse['y_pulse_raw']))
            else:
                b=fit(x,y,False); yc=np.linalg.solve(b[1:].T,(pulse['y_pulse_raw'].reshape(-1,4)-b[0]).T).T.ravel()
            d=yc-D@pulse['w_pulse']; err=[]
            for group in np.array_split(np.arange(len(d)),len(d)//4):
                keep=~np.isin(np.arange(len(d)),group); gamma=np.linalg.lstsq(Z[keep],d[keep],rcond=None)[0]; err.extend(d[group]-Z[group]@gamma)
            err=np.asarray(err).reshape(-1,4); stats.append((np.sqrt(np.mean(err**2)),np.max(np.sqrt(np.mean(err**2,axis=1))),np.sqrt(np.mean(d*d))))
        print(boundary,stats)
print('single-bank sweep layout', {k:np.shape(v) for k,v in sweeps['single_bank_sweeps'].items()})
