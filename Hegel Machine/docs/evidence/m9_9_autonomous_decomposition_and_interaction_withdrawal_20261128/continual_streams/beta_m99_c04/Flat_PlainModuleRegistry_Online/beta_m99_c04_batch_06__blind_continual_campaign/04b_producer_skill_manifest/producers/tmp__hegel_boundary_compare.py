import sys, json, pickle, itertools
sys.path[:0] = ['/holdout_task', '/tmp/hegel_continual_archive']
import numpy as np
from scipy.sparse import eye, kron, diags
from scipy.sparse.linalg import splu
import beta_m99_c04_batch_05_solver as prev
from _verified_upstream import calibrate_upstream
p = '/holdout_task/'
s = calibrate_upstream(pickle.load(open(p+'public_batch.pkl','rb'))[0]['study'], json.load(open(p+'_upstream_param_payload.json')))
nx,ny = s['nx'],s['ny']
ns = len(s['sensor_ij']); shifts=s['mu_shifts']; ids=s['sensor_ij'] @ np.array([1,nx]); m=len(s['source_modes']); size=nx*ny
rhs=np.stack([u.ravel(order='F') for u in s['source_modes']],axis=1)
z=s['Z_nuisance']; y=s['y_obs_raw']; n=len(y); nr=s['nuisance_ridge']; sr=s['source_ridge']; cap=s['u_box']; bud=s['total_budget']
def axis(d,k):
 h=s['L'+d]/(s['n'+d]-1); a=s['kappa_'+d]; v=s['vel_'+d]
 if k=='P': return prev._periodic_axis(a,v,h)
 out=prev._reflecting_axis(a,v,h)
 if k=='O':
  diag=np.zeros(len(a)); diag[[0,-1]]=s['robin_'+d]/h; out=out+diags(diag)
 return out

def design(k):
 l=kron(eye(ny),axis('x',k[0]))+kron(axis('y',k[1]),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))
 return np.vstack([splu((l+mu*eye(size)).tocsc()).solve(rhs)[ids] for mu in shifts])
def fit(S,mask):
 a,b,c=S[mask],z[mask],y[mask]; pz=np.linalg.solve(b.T@b+nr*np.eye(b.shape[1]),b.T)
 at=a-b@(pz@a); ct=c-b@(pz@c)
 w,nu=prev._box_simplex(at.T@at+sr*np.eye(m),at.T@ct,cap,bud)
 return w,pz@(c-a@w),np.linalg.norm(at@w-ct)/np.sqrt(len(c))
for k in itertools.product('ORP',repeat=2):
 S=design(k); w,g,train=fit(S,np.ones(n,bool)); scores=[]
 for folds in (np.tile(np.arange(ns),len(shifts)),np.repeat(np.arange(len(shifts)),ns),s['row_bank_id']):
  scores.append([])
  for i in np.unique(folds):
   test=folds==i; w1,g1,_=fit(S,~test); scores[-1].append(np.sqrt(np.mean((S[test]@w1+z[test]@g1-y[test])**2)))
 print(''.join(k),'train %.6g'%train,'sensor/shift/bank pooled,max',[(round(np.sqrt(np.mean(np.square(x))),6),round(max(x),6)) for x in scores],'w',np.round(w,4))
print('observation RMS',np.sqrt(np.mean(y*y)), 'sensor range',s['sensor_ij'].min(0),s['sensor_ij'].max(0))
