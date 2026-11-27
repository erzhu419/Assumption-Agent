import pickle
import numpy as np
from scipy import sparse as sp
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
from transport_probe import axis
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; V=s['V_reaction_2d'].ravel(order='F'); Z=s['Z_nuisance']; pulses=s['reference_tracer_pulses']; W=np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([p['y_pulse_raw'] for p in pulses]); F=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@W; I=sp.eye(nx*ny,format='csc'); g=(-1,'ar','cell','none'); bases=[axis(s['kappa_'+d],s['vel_'+d],s['L'+d],0,g) for d in ('x','y')]
def pred(r,axes=bases,extra=None):
 Ax,Ay=[B+sp.diags(np.r_[r[2*i],np.zeros(B.shape[0]-2),r[2*i+1]]) for i,B in enumerate(axes)]; A=sp.kron(sp.eye(ny),Ax)+sp.kron(Ay,sp.eye(nx))+sp.diags(V if extra is None else V+extra); return np.vstack([splu((A+m*I).tocsc()).solve(F)[idx] for m in s['mu_shifts']])
r0=np.array([3.66,4.45,6.51,5.37]); rnom=np.repeat([s['robin_x'],s['robin_y']],2); h=np.repeat([s['Lx']/(nx-1),s['Ly']/(ny-1)],2); k=np.r_[s['kappa_x'][[0,-1]],s['kappa_y'][[0,-1]]]; v=np.r_[s['vel_x'][[0,-1]],s['vel_y'][[0,-1]]]; print('wall nominal/r_over_h/series/series_half/k_r_over_h/velocity',rnom,rnom/h,rnom/h/(1+rnom*h/k),rnom/h/(1+.5*rnom*h/k),k*rnom/h,v)
for project in (False,True):
 def fun(r):
  e=pred(r)-Y
  return e-Z@np.linalg.lstsq(Z,e,rcond=None)[0] if project else e
 out=least_squares(lambda r:fun(r).ravel(),r0,gtol=1e-13); e=fun(out.x); print('project',project,'walls',out.x,'RMS',np.sqrt(np.mean(e**2)),'nuisance fit',np.linalg.lstsq(Z,pred(out.x)-Y,rcond=None)[0].T); print('shift RMS',np.sqrt(np.mean(e.reshape(3,32,2)**2,axis=(1,2))),'bank RMS',np.sqrt(np.mean(e.reshape(-1,4,2)**2,axis=(0,2)))); print('top',[(int(i//32),ij[i%32].tolist(),np.round(e[i],7).tolist()) for i in np.argsort(np.max(np.abs(e),axis=1))[-9:]])
print('Y extrema',Y.min(axis=0),Y.max(axis=0),'pulses',W.T); print('source ranges',[(round(a.min(),4),round(a.max(),4)) for a in s['source_modes']]); print('telemetry col correlation',np.round(np.corrcoef(s['aux_manifold_telemetry'].T),2)); print('telemetry blocks means',np.round(s['aux_manifold_telemetry'].reshape(3,-1,4).mean(axis=1),3))
