import numpy as np,pickle,runpy,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); d=f['run_node']({'__input__':s}); x=s['train']['x']; y=np.c_[s['train']['y'],d['traj']]; z=d['z']; a=f['_panel'](s['panel_a'],3); b=f['_panel'](s['panel_b']); w=f['_affine'](f['_affine'](x)@a)@b
print('affine',a.round(3),b.round(3),sep='\n'); print('std targets',y.std(0).round(4)); print('singular trajectory',np.linalg.svd(y[:,1:],compute_uv=False).round(4)); print('w mean/std',w.mean(0),w.std(0));
def poly(t,deg):
 p=[np.ones(len(t))]; names=['1'];
 for k in range(1,deg+1):
  for i in itertools.combinations_with_replacement(range(t.shape[1]),k): p.append(np.prod(t[:,i],axis=1)); names.append(''.join(map(str,i)))
 return np.array(p).T,names
spl=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),v),v) for v in np.array_split(np.random.default_rng(123).permutation(72),4)]
def test(K,lams=(1e-6,1e-4,1e-3,.01,.1,1)):
 results=[]
 for l in lams:
  err=[]
  for t,v in spl:
   p=K[np.ix_(v,t)]@np.linalg.solve(K[np.ix_(t,t)]+l*np.eye(len(t)),y[t]); r=np.abs(p-y[v]); err.append([r[:,0].mean(),r[:,1:].mean()])
  results.append(np.array(err))
 return np.array(results)
for name,t in [('x',x),('w',w),('z',z),('a',f['_affine'](x)@a)]:
 print(name)
 for deg in range(1,5):
  p,n=poly(t,deg); q=np.linalg.lstsq(p,y,rcond=None)[0]; g=test(p@p.T); print('poly',deg,'fit',np.mean(np.abs(p@q-y),axis=0)[[0,-1]].round(4),'official',g[0,0].round(4),'worst',g.max(1).min(0).round(4));
  if deg in (2,3) and name in ('x','w'): print('tops', [[(n[i],round(q[i,j],4)) for i in np.argsort(-np.abs(q[:,j]))[:10]] for j in (0,-1)])
 dist=((t[:,None,:]-t[None,:,:])**2).sum(2); p,_=poly(t,2)
 for e in (.04,.1,.25,.5,1):
  g=test(np.exp(-e*dist)+p@p.T); print('rbf',e,'official',g[:,0].min(0).round(4),'worst',g.max(1).min(0).round(4))
 u=s['panel_c']['u']; out=np.c_[s['panel_c']['v1'],s['panel_c']['v2']]; p,n=poly(u,2); q=np.linalg.lstsq(p,out,rcond=None)[0]; print('panel_c quad rmse',np.sqrt(((out-p@q)**2).mean(0)).round(4)) if name=='x' else None
