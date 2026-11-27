import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

def axis(k,v,h,r, p):
    f=(k[1:]+k[:-1])/2/h**2
    lo=-np.r_[f[:-1]+np.maximum(v[1:-1],0)/h,k[-1]/h**2*p[3]]
    hi=-np.r_[k[0]/h**2*p[2], f[1:]+np.maximum(-v[1:-1],0)/h]
    d=np.r_[-hi[0]+r/h*p[0], f[:-1]+f[1:]+abs(v[1:-1])/h, -lo[-1]+r/h*p[1]]
    return diags([lo,d,hi],[-1,0,1],format='csc')

def response(s,p,modes=None):
    nx,ny=s['nx'],s['ny']; ix,iy=eye(nx),eye(ny)
    ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],p[:4]); ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],p[4:]); L=(kron(iy,ax)+kron(ay,ix)+diags(s['V_reaction_2d'].ravel('F'))).tocsc(); I=eye(nx*ny,format='csc')
    modes=s['source_modes'] if modes is None else modes
    rhs=modes.transpose(0,2,1).reshape(len(modes),-1).T
    obs=[]; ij=np.asarray(s['sensor_ij']); ns=len(s['y_obs_raw'])//len(s['mu_shifts'])
    for a,mu in enumerate(s['mu_shifts']):
        u=splu(L+mu*I).solve(rhs)
        pts=ij if len(ij)==ns else ij[a*ns:(a+1)*ns]
        obs.append(u[pts[:,0]+nx*pts[:,1]])
    return np.vstack(obs),u.T.reshape(-1,ny,nx).transpose(0,2,1)

if __name__=='__main__':
 import pickle
 from scipy.optimize import least_squares
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; print('identity sweep RMSE/max',np.sqrt(np.mean((x-y)**2)),np.max(abs(x-y)))
 print('pulse shapes',[(p['w_pulse'].shape,p['y_pulse_raw'].shape) for p in s['reference_tracer_pulses']]); print('ij bounds',s['sensor_ij'].min(0),s['sensor_ij'].max(0)); print('telemetry mean/std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0)); print('item keys',pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0].keys())
 P=s['reference_tracer_pulses']; W=np.stack([p['w_pulse'] for p in P]); Y=np.stack([p['y_pulse_raw'] for p in P],axis=1); M=np.einsum('pm,mxy->pxy',W,s['source_modes']); base=np.ones(8); D=response(s,base,M)[0]-Y; print('nominal pulse RMSE/max by pulse',np.sqrt(np.mean(D**2,axis=0)),np.max(abs(D),axis=0)); print('nominal residual by shift',np.sqrt(np.mean(D.reshape(3,-1,2)**2,axis=1)))
 for groups in [[[0,1],[4,5]],[[0],[1],[4],[5]],[[0],[1],[4],[5],[2],[3],[6],[7]]]:
  def params(t):
   p=base.copy()
   for z,g in zip(t,groups): p[g]=z
   return p
  for train in [[0],[1],[0,1]]:
   def fun(t): return (response(s,params(t),M)[0]-Y)[:,train].ravel()
   opt=least_squares(fun,np.ones(len(groups)),bounds=(0,10),gtol=1e-13,ftol=1e-13,xtol=1e-13); D=response(s,params(opt.x),M)[0]-Y; print('groups',groups,'train',train,'p',np.round(params(opt.x),6),'RMSE',np.sqrt(np.mean(D**2,axis=0)),'worst',np.max(abs(D)))
