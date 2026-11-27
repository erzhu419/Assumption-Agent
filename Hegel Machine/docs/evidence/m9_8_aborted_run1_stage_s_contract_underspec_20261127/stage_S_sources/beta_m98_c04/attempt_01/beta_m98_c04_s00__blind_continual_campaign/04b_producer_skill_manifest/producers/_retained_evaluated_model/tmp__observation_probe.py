import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 import transport_refine as t
from wall_grammar import axes_for
import numpy as np
from scipy.optimize import least_squares
s=t.s; c=s['sensor_calibration_sweeps']; print('study keys',list(s)); print('calibration keys',list(c)); print('metadata',[(k,v) for k,v in s.items() if isinstance(v,(str,int,float))]); print('bank sweep details');
for k,v in c['single_bank_sweeps'].items():
 a=np.asarray(v); print(k,a.shape,a[:3],a[-3:]);
x=c['ref_rosette']; y=c['raw_rosette']; print('sweep ranges',x.min(0),x.max(0)); print('cross coefficients',np.linalg.lstsq(np.c_[np.ones(len(x)),x],y-x,rcond=None)[0]);
init=np.r_[t.r0,np.zeros(4)]; p=least_squares(lambda q:(t.pred(q[:4],axes_for(q,'wall-row'))-t.Y).ravel(),init,gtol=1e-13).x; print('identified walls',p)
F=t.F; t.F=s['source_modes'].transpose(0,2,1).reshape(-1,s['nx']*s['ny']).T
S=t.pred(p[:4],axes_for(p,'wall-row')); y=s['y_obs_raw']; X=np.c_[S,t.Z]; w=np.linalg.lstsq(X,y,rcond=None)[0]; print('all-mode design:',S.shape,'OLS weights + nuisance',w,'weight sum',w[:S.shape[1]].sum(),'residual rms',np.std(X@w-y)); print('shift RMS',np.sqrt(np.mean((X@w-y).reshape(len(s['mu_shifts']),-1)**2,axis=1))); print('design singular values',np.linalg.svd(X,compute_uv=False)); print('terminal candidate magnitude',S.min(),S.max()); print('pulse Gram singular values',np.linalg.svd(F,compute_uv=False));
