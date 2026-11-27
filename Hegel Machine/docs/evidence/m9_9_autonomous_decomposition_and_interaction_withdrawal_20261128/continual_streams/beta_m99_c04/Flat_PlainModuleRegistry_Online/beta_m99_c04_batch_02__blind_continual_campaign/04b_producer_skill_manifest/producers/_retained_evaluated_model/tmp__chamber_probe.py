import sys,pickle,json,itertools
import numpy as np
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
D='/holdout_task/'
item=pickle.load(open(D+'public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open(D+'_upstream_param_payload.json'))
pulses=item['campaign_diagnostics']; W=np.array([p['w_pulse'] for p in pulses]).T
Y=np.column_stack([calibrate_upstream({**s,'y_obs_raw':p['y_pulse_raw']},pay)['y_obs_raw'] for p in pulses])
Z=s['Z_nuisance']; yr=Y-Z@np.linalg.lstsq(Z,Y,rcond=None)[0]

def axis(k,v,h,r,left,right):
    f=(k[1:]+k[:-1])/2/h**2
    lo=-f-np.maximum(v[1:],0)/h; hi=-f-np.maximum(-v[:-1],0)/h
    lo[-1]=-k[-1]/h**2; hi[0]=-k[0]/h**2
    d=-np.r_[0,lo]-np.r_[hi,0]; d[[0,-1]]+=r/h*np.array([left,right]); return diags([lo,d,hi],[-1,0,1],format='csc')

def design(p,s=s):
    nx,ny=s['nx'],s['ny']; n=nx*ny
    x=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],*p[:2]); y=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],*p[2:]); L=kron(eye(ny),x)+kron(y,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))
    Q=np.stack([m.ravel(order='F') for m in s['source_modes']],axis=1); ii,jj=s['sensor_ij'].T; ids=ii+nx*jj
    U=[splu((L+shift*eye(n)).tocsc()).solve(Q) for shift in s['mu_shifts']]; return np.vstack([u[ids] for u in U])

def residual(p):
    R=design(p)@W-Y
    return R-Z@np.linalg.lstsq(Z,R,rcond=None)[0]

if __name__=='__main__':
    from scipy.optimize import least_squares
    for i,e in enumerate(json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))):
        A=np.array(e['facility_telemetry_trace']); print('archive',i,'mean',A.mean(0).round(5),'std',A.std(0).round(5),'last',A[-1])
    print('study telemetry mean/std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0))
    sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; y=sw['raw_rosette']; ok=np.all(np.abs(y)<=8.5,axis=1); print('sweep inliers',ok.sum(),'raw range',y.min(0),y.max(0))
    for k in range(4):
        d=np.column_stack((x,y[:,0]*0+1)); co=np.linalg.lstsq(d[ok],y[ok,k],rcond=None)[0]; print('sweep channel',k,co,'rmse',np.std(d[ok]@co-y[ok,k]))
    print('pulse RMS projected',np.sqrt(np.mean(yr**2,axis=0)))
    scores=sorted((np.sqrt(np.mean(residual(p)**2)),p) for p in itertools.product((0,1),repeat=4)); print('Robin switches',scores)
    fit=least_squares(lambda p:residual(p).ravel(),np.ones(4),bounds=(0,10)); print('continuous Robin',fit.x,'pulse RMSE',np.sqrt(np.mean(fit.fun.reshape(-1,3)**2,axis=0))); print('raw RMSE',np.sqrt(np.mean((design(fit.x)@W-Y)**2,axis=0)))
