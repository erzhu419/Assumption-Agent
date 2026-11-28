import importlib.util, pickle, numpy as np
from scipy.optimize import minimize_scalar
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']; x=T['x']; z=T['traj']; y=T['y']; ix=np.arange(len(x)); rng=np.random.default_rng(2851)
folds=[(ix[:52],ix[52:])]
for _ in range(12):
 j=rng.permutation(ix); folds.append((j[:52],j[52:]))
for v in [*x.T,np.linalg.norm(x,axis=1),y,z[:,-1]]:
 j=np.argsort(v); folds.extend([(j[:52],j[52:]),(j[20:],j[:20])])
a=s['panel_a']; u,v=a['u'],a['v']; d=np.linalg.norm(v-u,axis=1); u,v=u[d<max(4.,4.*np.median(d))],v[d<max(4.,4.*np.median(d))]; W=m._fit(m._affine(u),v)
panel_errors=[]
for j in range(len(u)):
 tr=np.arange(len(u))!=j
 baseline=(m._affine(u[j:j+1])@m._fit(m._affine(u[tr]),v[tr]))[0]
 challenger=m._operator(u[tr],v[tr],lambda q,k:np.sin(k*q),(1.5,3.))(u[j:j+1])[0]
 panel_errors.append([np.mean(abs(baseline-v[j])),np.mean(abs(challenger-v[j]))])
q=m._coordinates(a,x); p,r=m._operators(s['panel_c'],q); f=m._affine(np.c_[p,r,np.sum(p*r,axis=1)])
qa=np.linalg.solve(W[1:].T,(x-W[0]).T).T; pa,ra=m._operators(s['panel_c'],qa); fa=m._affine(np.c_[pa,ra,np.sum(pa*ra,axis=1)])
res=[]
for tr,te in folds:
 st=dict(s,train={**T,**{k:T[k][tr] for k in ('x','y','traj')}},test={'x':x[te]}); o=m.solve_batch(st); pred=o['state_pred']; yp=o['y_hat']; ed=abs(pred-z[te]); B=np.linalg.svd(z[tr],full_matrices=False)[2][:2]
 dot=f[te]@m._fit(f[tr],z[tr]@B.T)@B
 ya=fa[te]@m._fit(fa[tr],y[tr]); end=pred[:,-1]
 assert np.allclose(o['diagnostics_vec'],[yp.mean(),yp.std(),end.mean(),end.std()],rtol=0,atol=1e-12)
 res.append([np.mean(abs(yp-y[te])),ed.mean(),ed.mean(0).max(),np.mean(abs(ya-y[te])),np.mean(abs(dot-z[te]))])
res=np.asarray(res); panel_errors=np.asarray(panel_errors)
print('ng_t04 calibration LOO affine/nonlinear mean:',panel_errors.mean(0).tolist(),'worst:',panel_errors.max(0).tolist())
print('columns: candidate y, candidate state, worst time, affine calibration y, dot-only state')
print('ordered',res[0].tolist(),'random worst',res[1:13].max(0).tolist(),'edge worst',res[13:].max(0).tolist())
assert res[:,:3].max()<.038 and panel_errors.mean(0)[1]<.1*panel_errors.mean(0)[0] and res[:,1].max()<res[:,4].max()
