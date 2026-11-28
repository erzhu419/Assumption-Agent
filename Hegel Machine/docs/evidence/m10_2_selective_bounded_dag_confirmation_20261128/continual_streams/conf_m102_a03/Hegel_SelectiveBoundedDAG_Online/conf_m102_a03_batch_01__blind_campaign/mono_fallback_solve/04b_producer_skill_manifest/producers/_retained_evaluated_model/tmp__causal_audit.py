import importlib.util, pickle, numpy as np
from scipy.optimize import minimize_scalar
sp=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=(tr[k] for k in ('x','y','traj')); t=s['eval_times']; A,B=(s[k] for k in ('panel_a','panel_b')); fit=m._fit; aug=m._affine
u,v=A['u'],A['v']; c=m._calibration(A); w=np.roll(u,-1,axis=1); residual=np.linalg.norm(c[0]+c[1]*u+c[2]*w+c[3]*u*w+c[4]*u**3-v,axis=1); good=residual<6*np.median(residual); print('Calibration inliers/outliers',good.sum(),(~good).sum())
for name,F in [('affine',lambda r:aug(r)),('quadratic',lambda r:np.c_[aug(r),*[r[:,i]*r[:,j] for i in range(4) for j in range(i,4)]])]:
 e=[]
 for k in np.flatnonzero(good):
  mask=good.copy(); mask[k]=False; e.append(F(u[k:k+1])@fit(F(u[mask]),v[mask])-v[k]);
 print('Calibration LOO',name,'MAE',np.mean(np.abs(e)))
e=[]
for k in np.flatnonzero(good):
 mask=np.arange(len(u))!=k; cc=m._calibration({'u':u[mask],'v':v[mask]}); r=u[k]; n=np.roll(r,-1); e.append(cc[0]+cc[1]*r+cc[2]*n+cc[3]*r*n+cc[4]*r**3-v[k]);
print('Calibration LOO cyclic cubic MAE/max',np.mean(np.abs(e)),np.max(np.abs(e)))
q=m._inverse(x,c); F=aug(B['u']); b=fit(F,B['v']); z=aug(q)@b; print('Panel B affine residual MAE',np.mean(abs(F@b-B['v'])))
p=z[:,0]*z[:,2]; r=z[:,1]*z[:,3]; design={'sum':np.c_[z,p+r],'difference':np.c_[z,p-r],'separate':np.c_[z,p,r],'all_quadratic':np.c_[aug(z),*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]]}; errors={k:[] for k in (*design,'rational')}; coeff=[]
for order in [np.arange(len(x))]+[np.random.default_rng(j).permutation(len(x)) for j in range(32)]:
 a,j=order[:52],order[52:];
 for name,D in design.items(): errors[name].append(np.mean(abs(D[j]@fit(D[a],y[a])-y[j])))
 Ds=design['sum']; terms=[z[:,i]+sg*z[:,j] for i in range(4) for j in range(i+1,4) for sg in (-1,1)]+[p+r,p-r]
 best=None
 for term in terms:
  h=term**2/max(np.mean(term[a]**2),1e-12)
  def err(alpha):
   d=Ds[a]/(1+alpha*h[a,None]); rr=d@fit(d,y[a])-y[a]; return rr@rr
  o=minimize_scalar(err,bounds=(0,1),method='bounded'); rec=(o.fun,o.x,h)
  if best is None or rec[0]<best[0]: best=rec
 _,alpha,h=best; d=Ds/(1+alpha*h[:,None]); errors['rational'].append(np.mean(abs(d[j]@fit(d[a],y[a])-y[j]))); coeff.append(alpha)
for k,e in errors.items(): print('Readout',k,'prescribed/mean/worst',e[0],np.mean(e),max(e))
print('Rational fitted alpha range',min(coeff),max(coeff)); print('Study fields',sorted(s),'Train fields',sorted(tr),'telemetry shape',np.shape(s['telemetry_vec']))
# Remove one independent calibration or operator reference row at a time.
a,j=np.arange(52),np.arange(52,72); fold=dict(s,train={k:tr[k][a] for k in ('x','y','traj')},test={'x':x[j]}); err=[]
for name in ('panel_a','panel_b'):
 for k in range(len(s[name]['u'])):
  f=dict(fold); f[name]={field:np.delete(arr,k,axis=0) for field,arr in s[name].items()}; o=m.solve_batch(f); err.append([np.mean(abs(o['y_hat']-y[j])),np.mean(abs(o['state_pred']-T[j]))]);
print('Reference-row jackknife worst y/traj MAE',np.max(err,axis=0)); assert np.max(err)<.038
