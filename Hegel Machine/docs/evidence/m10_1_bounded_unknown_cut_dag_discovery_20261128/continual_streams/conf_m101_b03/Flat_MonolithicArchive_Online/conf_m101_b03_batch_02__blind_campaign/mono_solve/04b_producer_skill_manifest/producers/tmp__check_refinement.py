from check_batch import m, s, tr, idx
import numpy as np
from scipy.optimize import least_squares
fit=m._fit
z=m._calibrate(s['panel_a'],tr['x']); c,w=m._operators(s['panel_c'],z)
B=m._aff(np.c_[m._quadratic(z),c*w]); t=s['eval_times']; rad=np.linalg.norm(z,axis=1)

def pred(i,j,k):
 D=m._basis(t,m._rates(t,tr['traj'][i])); A=fit(D,tr['traj'].T).T; ba=fit(B[i],A[i]); p=B[j]@ba
 if k=='base': return p@D.T
 R=A[i,0]-B[i]@ba[:,0]
 def residualize(f): return f-B@fit(B[i],f[i])
 if k.startswith('sqrt') or k.startswith('cos'):
  q=z+c; q=(q-q[i].mean(0))/q[i].std(0)
  f=np.sqrt(.25+q*q) if k.startswith('sqrt') else np.cos(q)
  f=residualize(f); f=f/np.maximum(f[i].std(0),1e-8)
  a=np.linalg.solve(f[i].T@f[i]+float(k.split(':')[1])*np.eye(4),f[i].T@R)
 else:
  def features(v):
   q=z+v[0]*c+v[1]; return np.sqrt(v[2]**2+q*q) if k=='profile' else np.log1p((q/v[2])**2)
  def obj(v):
   f=residualize(features(v))[i]; return np.r_[f@fit(f,R)-R,.008*(v-np.array([1.,0.,.5]))]
  sol=min((least_squares(obj,x,bounds=([.1,-1,.08],[3,1,2])) for x in ([1,0,.5],[.4,0,1.2],[1.5,0,.2])),key=lambda x:x.cost)
  f=residualize(features(sol.x)); a=fit(f[i],R)
  if np.array_equal(i,idx[:52]): print(k,'params',sol.x.round(4),'coeff',a.round(4),'train residual',np.mean(abs(f[i]@a-R)))
 p[:,0]+=f[j]@a; return p@D.T

rng=np.random.default_rng(22467)
splits=[(idx[:52],idx[52:]),(idx[:56],idx[56:])]
r=np.argsort(rad); splits.append((r[:52],r[52:]));
for e in [z[:,j] for j in range(4)]+[-z[:,j] for j in range(4)]:
 q=np.argsort(e); splits.append((q[:56],q[56:]))
for _ in range(20):
 q=rng.permutation(idx); splits.append((q[:52],q[52:]))
for k in ('base','sqrt:1','sqrt:4','sqrt:8','cos:1','profile','logprofile'):
 a=[]
 for i,j in splits:
  e=np.abs(pred(i,j,k)-tr['traj'][j]); a.append([e.mean(),e.mean(0).max()])
 a=np.array(a)
 print(k,'ordered/radial',a[:3].round(5).tolist(),'coord max',a[3:11].max(0).round(5),'random mean/max',a[11:].mean(0).round(5),a[11:].max(0).round(5))
