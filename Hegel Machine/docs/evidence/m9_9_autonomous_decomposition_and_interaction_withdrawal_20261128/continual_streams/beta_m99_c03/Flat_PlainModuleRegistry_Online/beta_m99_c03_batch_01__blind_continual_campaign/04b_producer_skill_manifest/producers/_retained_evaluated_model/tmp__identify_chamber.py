import sys,pickle,itertools
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m99_c03_s00_solver as old
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; cal=s['sensor_calibration_sweeps']; x,y=cal['ref_rosette'],cal['raw_rosette']
powers=[p for d in range(4) for p in itertools.combinations_with_replacement(range(4),d)]
def features(x,ps): return np.stack([np.prod(x[...,p],axis=-1) if p else np.ones(x.shape[:-1]) for p in ps],axis=-1)
coef=np.linalg.lstsq(features(x,powers),y,rcond=None)[0]
for p,c in zip(powers,coef):
 if max(abs(c))>1e-4: print('cal',p,np.round(c,6))
active=[p for p,c in zip(powers,coef) if max(abs(c))>1e-4]; aff=powers[:5]
for ps in [aff,active]:
 f=features(x,ps); b=np.linalg.lstsq(f,y,rcond=None)[0]; r=f@b-y; print('cal RMSE/max',np.sqrt(np.mean(r*r)),np.max(abs(r)))
 ps_err=[]
 for j in range(len(x)):
  k=np.arange(len(x))!=j; ps_err.append(f[j]@np.linalg.lstsq(f[k],y[k],rcond=None)[0]-y[j])
 print('cal held-out max',np.max(np.abs(ps_err)))
nx,ny=s['nx'],s['ny']; axes=[old._axis_operator(s['kappa_'+a],s['vel_'+a],s['robin_'+a],s['L'+a]).toarray() for a in 'xy']; I=sparse.eye(nx*ny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); ix,iy=s['sensor_ij'].T; ix=ix+nx*iy
W=np.stack([p['w_pulse'] for p in item['campaign_diagnostics']],axis=1); rhs=np.column_stack([m.ravel(order='F') for m in s['source_modes']])@W; Y=np.stack([p['y_pulse_raw'].reshape(-1,4) for p in item['campaign_diagnostics']]); B=np.linalg.lstsq(features(x,active),y,rcond=None)[0]
def predict(p):
 aa=[a.copy() for a in axes]
 for ai,a in enumerate(aa):
  for ei,(i,j) in enumerate([(0,1),(-1,-2)]):
   d,r=p.reshape(2,2,2)[ai,ei]; oldd=-a[i,j]; oldr=a[i,i]-oldd; a[i,j]=-d*oldd; a[i,i]=d*oldd+r*oldr
 L=sparse.kron(sparse.eye(ny),sparse.csc_matrix(aa[0]),format='csc')+sparse.kron(sparse.csc_matrix(aa[1]),sparse.eye(nx),format='csc')+V
 z=np.vstack([splu(L+mu*I).solve(rhs)[ix] for mu in s['mu_shifts']]).T.reshape(Y.shape)
 return features(z,active)@B
p=np.ones(8); print('baseline pulse RMSE',np.sqrt(np.mean((predict(p)-Y)**2,axis=(1,2))))
for j in [-1,0,1,2]:
 keep=np.arange(len(Y))!=j; fit=least_squares(lambda q:(predict(q)[keep]-Y[keep]).ravel(),p,bounds=(0,8),gtol=1e-13,xtol=1e-13,ftol=1e-13)
 print('omit',j,'fit',fit.x,'pulse RMSE',np.sqrt(np.mean((predict(fit.x)-Y)**2,axis=(1,2))))
