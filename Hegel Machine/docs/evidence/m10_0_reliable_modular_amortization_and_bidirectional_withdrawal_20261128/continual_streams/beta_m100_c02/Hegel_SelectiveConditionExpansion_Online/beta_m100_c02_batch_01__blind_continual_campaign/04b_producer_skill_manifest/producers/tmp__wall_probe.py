import sys,pickle,json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m100_c02_s00_solver as old
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; nx,ny=s['nx'],s['ny']; nxny=nx*ny
p=i['campaign_diagnostics']; w=np.array([e['w_pulse'] for e in p]).T
def cal(y): return old.calibrate(dict(s,y_obs_raw=y))
y=np.column_stack([cal(e['y_pulse_raw']) for e in p]); base,U=old.chamber_response(s)
print('baseline max and rms per pulse and shift', np.max(np.abs((base@w-y).reshape(3,32,3)),axis=1),np.mean((base@w-y).reshape(3,32,3)**2,axis=1)**.5)
print('sensor range',s['sensor_ij'].min(axis=0),s['sensor_ij'].max(axis=0)); print('aux',s['aux_manifold_telemetry'][:6]); print('log means and std',[(np.mean(t['facility_telemetry_trace'],0),np.std(t['facility_telemetry_trace'],0)) for t in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))])
axes=[old.axis_operator(s['kappa_'+j],s['vel_'+j],s['L'+j]/(s['n'+j]-1),s['robin_'+j]).toarray() for j in 'xy']
print('base wall rows',[(a[0],a[-1]) for a in axes]); R=np.asarray(s['source_modes']).reshape(-1,nxny,order='F').T@w
ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; eye=sparse.eye(nxny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def pred(x, offsets, r=None):
    aa=[a.copy() for a in axes]; j=0
    for a in aa:
        for row,columns in [(0,offsets),(-1,[-1-k for k in offsets])]:
            for col in columns:
                a[row,col]+=x[j]; j+=1
    A=sparse.kron(sparse.eye(ny),aa[0],format='csc')+sparse.kron(aa[1],sparse.eye(nx),format='csc')+V
    return np.concatenate([splu(A+mu*eye).solve(R if r is None else r)[idx] for mu in s['mu_shifts']])
if __name__=='__main__':
 for off in ([0],[0,1],[0,1,-1],[0,1,2,-1,-2]):
    for leave in (None,0,1,2):
        t=np.array([j for j in range(3) if j!=leave]); f=lambda a: (pred(a,off)[:,t]-y[:,t]).ravel()
        z=least_squares(f,np.zeros(4*len(off)),max_nfev=250,gtol=1e-12,ftol=1e-11,xtol=1e-11)
        print('offsets',off,'leave',leave,'max rms',np.max(np.mean((pred(z.x,off)-y)**2,axis=0)**.5),'params',np.round(z.x.reshape(4,-1),6).tolist(),flush=True)
