import importlib.util,pickle
from itertools import product
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
path='/tmp/hegel_continual_archive/beta_m99_c04_batch_01_solver.py'
spec=importlib.util.spec_from_file_location('prior',path); prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; nx,ny=s['nx'],s['ny']
P=np.array([q['w_pulse'] for q in pulses]); Y=np.array([prior._calibrate(dict(s,y_obs_raw=q['y_pulse_raw'])) for q in pulses]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
axes=[]
for d,n in [('x',nx),('y',ny)]:
 k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(n-1); r=s['robin_'+d]/h
 axes.append((prior._axis(k,v,h,0).tolil(),k[[0,-1]]/h**2,r,(k[[0,-1]]+k[[1,-2]])/(2*h**2),v[[0,-1]]/h))
I=eye(nx*ny,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); rhs=np.array([m.ravel(order='F') for m in s['source_modes']]).T@P.T
ij=s['sensor_ij']@np.array([1,nx]);
def pred(p):
 aa=[]
 for j,(base,c,r,face,vel) in enumerate(axes):
  a=base.copy(); n=a.shape[0]; c0,c1,l0,l1=p[4*j:4*j+4]; a[0,1]=-c0*c[0]; a[-1,-2]=-c1*c[1]; a[0,0]=c0*c[0]+l0*r; a[-1,-1]=c1*c[1]+l1*r; aa.append(a.tocsc())
 L=kron(eye(ny),aa[0],format='csc')+kron(aa[1],eye(nx),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[ij] for mu in s['mu_shifts']]).T

def project(e): return e-(e@Q)@Q.T

def rms(e): return np.sqrt(np.mean(e**2,axis=1))
print('wall coefficients baseline coupling, leak, face ratio, velocity/h:',[(c,r,face/c,v) for _,c,r,face,v in axes]); print('calibrated pulse RMS',rms(Y))
rank=[]
for walls in product([0,1],repeat=4):
 for c in [1,2]:
  p=np.r_[c,c,walls[:2],c,c,walls[2:]]; e=pred(p)-Y; rank.append((max(rms(e)),max(rms(project(e))),p.tolist()))
print('best discrete by direct RMS', sorted(rank)[:6]); print('best discrete by nuisance-projected RMS',sorted(rank,key=lambda x:x[1])[:3])
start=np.ones(8)
for name,transform in [('direct',lambda x:x),('projected',project)]:
 for j in [None,0,1,2]:
  train=np.array([i for i in range(len(Y)) if i!=j]); fit=least_squares(lambda p:transform(pred(p)-Y)[train].ravel(),start,bounds=(0,10),xtol=1e-12,ftol=1e-12,gtol=1e-12); e=pred(fit.x)-Y
  print(name,'held out',j,'params',np.round(fit.x,6),'direct RMS',rms(e),'projected RMS',rms(project(e)),'min jac sv',np.linalg.svd(fit.jac,compute_uv=False)[-1])
