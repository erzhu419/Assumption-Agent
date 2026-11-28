import pickle, numpy as np, importlib.util
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']; x=T['x']; y=T['y']; z=T['traj']; t=s['eval_times']; ix=np.arange(len(x)); rng=np.random.default_rng(1729)
folds=[(ix[:52],ix[52:])];
for _ in range(20):
 j=rng.permutation(ix); folds.append((j[:52],j[52:]))
for v in [*x.T,np.linalg.norm(x,axis=1),y,z[:,-1]]:
 j=np.argsort(v); folds.extend([(j[:52],j[52:]),(j[20:],j[:20])])
u,v=s['panel_a']['u'],s['panel_a']['v']; ids=np.flatnonzero(np.linalg.norm(v-u,axis=1)<4)
for name,fn,lim in [('affine',None,None),('sin',lambda q,k:np.sin(k*q),(1.5,3)),('rational',lambda q,k:q/(1+k*q*q),(.1,1))]:
 errors=[]
 for i in range(5):
  te=ids[i::5]; tr=np.setdiff1d(ids,te)
  pred=m._affine(u[te])@m._fit(m._affine(u[tr]),v[tr]) if fn is None else m._operator(u[tr],v[tr],fn,lim)(u[te]); errors.append(np.mean(abs(pred-v[te])))
 print('panel A held-out mean/worst',name,np.mean(errors),np.max(errors))
q=m._coordinates(s['panel_a'],x); p,r=m._operators(s['panel_c'],q); J=(p[:,:,None]*r[:,None,:]).reshape(len(x),16); d=np.trace(J.reshape(-1,4,4),axis1=1,axis2=2); off=np.array([i for i in range(16) if i%5]); a=m._affine(np.c_[p,r,d]); full=m._affine(np.c_[p,r,J]); print('study keys/shapes',[(k,np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in s.items()]); print('telemetry and summary',s['telemetry_vec'],T['summary'])
def sparse(tr,target,k):
 selected=[]; remaining=list(off)
 for _ in range(k):
  best=min(remaining,key=lambda j: np.linalg.norm((A:=np.c_[a[tr],J[tr][:,selected+[j]]])@m._fit(A,target)-target)); selected.append(best); remaining.remove(best)
 return np.c_[a,J[:,selected]],tuple(selected)
for tag in ['full','dot','sparse1','sparse2','sparse3','sparse1-exp','sparse2-exp','sparse1-diff','sparse1-rational']:
 err=[]; selections=[]
 for tr,te in folds:
  if tag.endswith('-exp'):
   def basis(k):return -np.expm1(-t[:,None]*k)
   def resid(k):
    f=basis(k);return (f@m._fit(f,z[tr].T)-z[tr].T).ravel()
   B=np.linalg.qr(basis(least_squares(resid,[1.8,.4],bounds=([.8,.03],[8.,.79])).x))[0].T
  else: B=np.linalg.svd(z[tr],full_matrices=False)[2][:2]
  target=z[tr]@B.T
  if tag=='full': f=full
  elif tag=='dot': f=a
  else:
   f,sel=sparse(tr,target,int(tag[6]));selections.append(sel)
   if tag.endswith('-diff'):f=np.c_[f,(p[:,0]-p[:,1])*(r[:,0]-r[:,1])]
   if tag.endswith('-rational'):
    f=min([f]+[np.c_[f,f[:,9:]/(1+k*g[:,None]**2)] for k in (.1,.5,1.) for g in (p[:,0]-p[:,1],r[:,0]-r[:,1],d)],key=lambda F: np.linalg.norm(F[tr]@m._fit(F[tr],target)-target)+(.02 if F.shape[1]>f.shape[1] else 0))
  e=np.abs(f[te]@m._fit(f[tr],target)@B-z[te]);err.append([e.mean(),e.mean(0).max()])
 err=np.asarray(err);print(tag,'ordered',err[0],'random worst',err[1:21].max(0),'edge worst',err[21:].max(0),'selections',[(g,selections.count(g)) for g in sorted(set(selections)) if selections.count(g)>1])
