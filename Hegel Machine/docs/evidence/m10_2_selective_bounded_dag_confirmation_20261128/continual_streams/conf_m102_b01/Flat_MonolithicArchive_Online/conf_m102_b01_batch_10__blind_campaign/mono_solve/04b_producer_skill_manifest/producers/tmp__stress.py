import pickle,runpy,numpy as np,itertools as it
m=runpy.run_path('/holdout_task/batch_solver.py');s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];q=m['_coordinates'](s,np.r_[tr['x'],s['test']['x']]);d=m['_quadratic'](q); y=tr['y']; T=tr['traj']; LS=m['_fit']; ix=np.arange(len(y)); rng=np.random.default_rng(574)
splits=[np.arange(52,72)]+[rng.choice(ix,20,False) for _ in range(20)]+[np.argsort(a[:72])[-k:] for a in (np.linalg.norm(q,axis=1),np.max(abs(tr['x']),axis=1),*abs(q.T)) for k in (8,12)]
for name,k in [('linear',[]),('full',list(range(5,15))),('sparse nested',None)]:
 e=[]
 for te in splits:
  ti=np.setdiff1d(ix,te)
  if k is None:
   best=None
   for n in range(5):
    for kk in it.combinations(range(5,15),n):
     a=d[ti][:,[*range(5),*kk]];p=np.linalg.pinv(a);r=(a@p@y[ti]-y[ti])/np.maximum(1e-6,1-(a*p.T).sum(-1));score=np.mean(r*r)*(1+.035*n)
     if best is None or score<best[0]:best=score,kk
   kk=best[1]
  else:kk=k
  a=d[:,[*range(5),*kk]];e.append(np.mean(abs(a[te]@LS(a[ti],y[ti])-y[te])))
  if len(e)==1: print(name,'canonical columns',kk,'coef',LS(a[ti],y[ti]))
 print(name,'canonical, random worst, boundary worst',e[0],max(e[1:21]),max(e[21:]))
e=[]
for te in splits:
 ti=np.setdiff1d(ix,te); ss=dict(s,train={k:v[ti] if len(v)==72 else v for k,v in tr.items()},test={'x':tr['x'][te]});p=m['solve_batch'](ss);e.append(np.mean(abs(p['state_pred']-T[te]),axis=0))
e=np.array(e);print('state per time canonical',e[0]);print('trajectory canonical, random and boundary worst',e[0].mean(),max(e[1:21].mean(1)),max(e[21:].mean(1))); print('worst individual horizon',e.max());print('train/test q maxima',np.max(abs(q[:72]),axis=0),np.max(abs(q[72:]),axis=0))
