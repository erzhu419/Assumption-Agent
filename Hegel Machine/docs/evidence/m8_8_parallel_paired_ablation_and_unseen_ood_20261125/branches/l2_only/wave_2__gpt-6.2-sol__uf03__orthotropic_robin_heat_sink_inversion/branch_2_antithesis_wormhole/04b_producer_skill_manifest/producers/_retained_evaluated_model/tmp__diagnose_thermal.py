import sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy.optimize import least_squares
import thermal_sink_solver as solver
# Load the independent forward generator without rerunning the completed sweeps.
text = open('/tmp/check_thermal.py').read()
exec(text[text.index('def synthetic('):text.index('\nfor label, args in')])
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from thermal_sink_solver import _plate_operator

def check(m, gt):
    y = m['T_obs']; bank = solver._response_bank(m); n = m['N']
    c = (np.arange(n)+.5)/n; s = m['sigma_m']
    def residual(p):
        x,z,a,b = p
        q = np.exp(-.5*((c[:,None]-x)**2+(c[None,:]-z)**2)/s**2)
        return a*np.einsum('mij,ij->m',bank,q)+b-y
    def params(d): return np.r_[d['source_xy'],d['amplitude'],d['junction_offset']]
    truth = params(gt); est = params(solver.localize_thermal_source(m))
    gt_r = residual(truth); bad = abs(gt_r) > .02
    keep = len(y)-int(np.ceil(.10*len(y)))
    def summary(name,p):
        r=residual(p); order=np.argsort(abs(r));
        print(name, np.round(p,7), 'trim SSE', np.sum(r[order[:keep]]**2), 'known-good SSE', np.sum(r[~bad]**2), 'stuck among trimmed', np.sum(bad[order[keep:]]))
    summary('truth',truth); summary('solver',est)
    bounds=([0,0,0,-np.inf],[1,1,np.inf,np.inf])
    for origin,p0 in [('solver',est),('truth',truth)]:
        for loss in ('huber','soft_l1','cauchy'):
            fit=least_squares(residual,p0,bounds=bounds,loss=loss,f_scale=.0015,x_scale='jac',max_nfev=150)
            summary(origin+' '+loss,fit.x)
        fit=least_squares(lambda p: residual(p)[~bad],p0,bounds=bounds,x_scale='jac')
        summary(origin+' oracle inliers',fit.x)
    print('stuck sensors', m['sensor_ij'][bad].tolist(), 'truth residuals', np.round(gt_r[bad],4).tolist())
    print('nearest sensors [i,j,obs,truth residual,solver residual]'); d=np.linalg.norm(c[m['sensor_ij']]-truth[:2],axis=1); ix=np.argsort(d)[:12]
    print(np.round(np.column_stack([m['sensor_ij'],y,gt_r,residual(est)])[ix],6))

synthetic(35,.09,7.,(.96,.045),64,1.1,.08,0)
