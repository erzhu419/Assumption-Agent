import importlib.util, json, pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_beta_m99_00_s00_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; d=i['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; loc=ij[:,0]+nx*ij[:,1]; rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T @ np.array([a['w_pulse'] for a in d]).T
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; a,b=np.array([np.linalg.lstsq(np.c_[np.ones(len(x)),x[:,k]],y[:,k],rcond=None)[0] for k in range(4)]).T; obs=np.array([((p['y_pulse_raw'].reshape(-1,4)-a)/b).ravel() for p in d]).T
base=[m._axis(s['kappa_'+k],s['vel_'+k],s['L'+k]/(n-1),s['robin_'+k]).tolil() for k,n in zip('xy',(nx,ny))]; rx=diags(s['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny,format='csc'); Z=s['Z_nuisance']; q=np.linalg.qr(Z)[0]
def model(p):
    axes=[]
    for A,pars,k in zip(base,p.reshape(2,2,2),'xy'):
        A=A.copy(); n=A.shape[0]; h=s['L'+k]/(n-1); r=s['robin_'+k]/h
        for (j,l),(f,g) in zip(((0,1),(n-1,n-2)),pars):
            A[j,l]*=f; A[j,j]=-A[j,l]+g*r
        axes.append(A.tocsc())
    ax,ay=axes; L=(kron(eye(ny),ax)+kron(ay,eye(nx))+rx).tocsc()
    return np.concatenate([splu(L+v*I).solve(rhs)[loc] for v in s['mu_shifts']])
def rms(a): return np.sqrt(np.mean(a*a,axis=0))
full=np.ones(8); e=model(full)-obs; print('baseline pulse RMSE',rms(e),'after exact nuisance removal',rms(e-q@(q.T@e)))
for projected in (False,True):
    def error(p):
        e=model(p)-obs
        return e-q@(q.T@e) if projected else e
    for hold in (-1,0,1,2):
        ids=np.arange(len(d))!=hold
        out=least_squares(lambda p:error(p)[:,ids].ravel(),full,diff_step=1e-5,gtol=1e-12,ftol=1e-12,xtol=1e-12)
        print('projected',projected,'held',hold,'p',out.x.round(6),'pulse RMSE',rms(error(out.x)),'raw RMSE',rms(model(out.x)-obs),'min sv',np.linalg.svd(out.jac,compute_uv=False)[-1])
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for j in log:
    x=np.array(j['facility_telemetry_trace']); print(j['completed_campaign_short_name'], 'trace mean',x.mean(axis=0).round(4),'std',x.std(axis=0).round(4))
