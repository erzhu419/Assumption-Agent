import sys,pickle
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream

# Each wall has an inward coupling multiplier and an independent row sum,
# expressed in units of the default Robin loss for that axis.
def axis(k,v,h,r,p):
    a=(k[:-1]+k[1:])/(2*h*h)
    lo=np.r_[-a[:-1]-np.maximum(v[1:-1],0)/h,-p[2]*k[-1]/h**2]
    hi=np.r_[-p[0]*k[0]/h**2,-a[1:]-np.maximum(-v[1:-1],0)/h]
    mid=np.r_[-hi[0]+p[1]*r/h,-lo[:-1]-hi[1:],-lo[-1]+p[3]*r/h]
    return diags([lo,mid,hi],[-1,0,1],format='csc')

def forward(s,p,modes=None):
    nx,ny=s['nx'],s['ny']; I=eye(nx*ny,format='csc')
    x=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],p[:4]); y=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],p[4:])
    L=(kron(eye(ny),x)+kron(y,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    ii,jj=np.asarray(s['sensor_ij']).T; ids=ii+nx*jj
    if modes is None: modes=s['source_modes']
    rhs=modes.reshape(len(modes),-1,order='F').T
    values=[]
    for mu in s['mu_shifts']:
        u=splu(L+mu*I).solve(rhs); values.append(u[ids])
    return np.vstack(values), u.T.reshape(len(modes),nx,ny,order='F')

def load():
    d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
    s=calibrate_upstream(dict(d['study'],reference_tracer_pulses=d['campaign_diagnostics']))
    pulses=s['reference_tracer_pulses']; W=np.stack([q['w_pulse'] for q in pulses]); Y=np.stack([q['y_pulse_raw'] for q in pulses],axis=1)
    return s,np.einsum('pm,mxy->pxy',W,s['source_modes']),Y

if __name__=='__main__':
    s,M,Y=load()
    def summary(p):
        e=forward(s,p,M)[0]-Y
        return np.round(np.sqrt(np.mean(e**2,axis=0)),9),np.round(np.max(np.abs(e.reshape(3,-1)),axis=1),9)
    for p in (np.ones(8), np.tile([1,0],4), np.tile([2,0],4)):
        print('baseline',p,summary(p))
    masks=[np.ones(Y.shape,bool)]
    for i in range(Y.shape[1]):
        mask=masks[0].copy(); mask[:,i]=False; masks.append(mask)
    for i in range(3):
        mask=masks[0].copy(); mask.reshape(3,-1)[i]=False; masks.append(mask)
    for mask in masks:
        def residual(p): return (forward(s,p,M)[0]-Y)[mask]
        sol=least_squares(residual,np.ones(8),bounds=(np.tile([0.,0.],4),np.tile([8.,15.],4)),xtol=1e-12,gtol=1e-13,ftol=1e-12)
        err=forward(s,sol.x,M)[0]-Y; test=err[~mask] if np.any(~mask) else err.ravel()
        print('fit',np.round(sol.x,8),'train',np.sqrt(np.mean(err[mask]**2)),'test RMSE/max',np.sqrt(np.mean(test**2)),max(abs(test)),'singular',np.round(np.linalg.svd(sol.jac,compute_uv=False),5))
