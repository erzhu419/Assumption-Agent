import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; n=len(d['x']); c=s['panel_c']; u=c['u']; v=c['v2']; f=m._fit
print('study keys', sorted(s), 'train keys',sorted(d),'telemetry shape',np.shape(s['telemetry_vec']))
# Compare a shared cosine frequency to a rational denominator, profiling on each retained panel fold.
for name,fun,bnd in [('cosine',lambda x,a:np.cos(a*x),(1.5,3.2)),('rational',lambda x,a:1/(1+a*x*x),(.01,8))]:
 err=[]
 for i in range(len(u)):
  tr=np.arange(len(u))!=i
  def design(x,a): return m._aff(fun(x,a))
  def loss(a):
   A=design(u[tr],a); return np.linalg.norm(A@f(A,v[tr])-v[tr])**2
  a=minimize_scalar(loss,bounds=bnd,method='bounded').x
  err.append(np.abs(design(u[i:i+1],a)@f(design(u[tr],a),v[tr])-v[i]))
 print('panel C2 LOO',name,np.mean(err))
z,c1,c2,h=m._features(s,d['x']); norms=np.linalg.norm(z,axis=1)
# Calibration reverse-map residuals; remove each reference row independently.
a=s['panel_a']; er=[]
for j in range(len(a['u'])):
 k=np.arange(len(a['u']))!=j
 pred=m._calibrate({key:w[k] for key,w in a.items()},a['v'][j:j+1]); er.append(np.linalg.norm(pred-a['u'][j]))
print('calibration LOO latent error quantiles',np.quantile(er,[.5,.75,.9,.95,1]))
splits={'fixed':np.arange(n)[52:],'largest_radius':np.argsort(norms)[-20:]}
for j in range(z.shape[1]):
 idx=np.argsort(z[:,j]); splits[f'coord{j}_low']=idx[:20]; splits[f'coord{j}_high']=idx[-20:]
for j in range(z.shape[1]):
 for k in range(j+1,z.shape[1]):
  for sign in [-1,1]:
   val=np.abs(z[:,j]+sign*z[:,k]); splits[f'pair{j},{k},{sign}']=np.argsort(val)[-20:]
truth=np.c_[d['y'],d['traj']]; worst=np.zeros(4)
for name,te in splits.items():
 tr=np.setdiff1d(np.arange(n),te); p=m.solve_batch(dict(s,train={**d,**{k:d[k][tr] for k in ('x','y','traj')}},test={'x':d['x'][te]})); y=p['y_hat']; state=p['state_pred']; err=abs(np.c_[y,state]-truth[te]); last=state[:,-1]; np.testing.assert_allclose(p['diagnostics_vec'],[y.mean(),y.std(),last.mean(),last.std()]); ms=np.array([err[:,0].mean(),err[:,1:].mean(),err[:,1:].mean(0).max(),err.max()]); worst=np.maximum(worst,ms); print(name,np.round(ms,6))
print('worst across fixed and directional slices',worst); assert max(worst[:3])<.038
