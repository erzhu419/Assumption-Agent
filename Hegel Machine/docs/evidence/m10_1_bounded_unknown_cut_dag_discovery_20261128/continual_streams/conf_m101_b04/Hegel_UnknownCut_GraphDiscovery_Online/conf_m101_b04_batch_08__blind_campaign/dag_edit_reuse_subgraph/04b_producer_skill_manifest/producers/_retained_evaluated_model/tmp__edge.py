import pickle,runpy,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; N=np.arange(len(x)); f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); w=f['_affine'](f['_affine'](x)@f['_panel'](s['panel_a'],3))@f['_panel'](s['panel_b']); c=s['panel_c']; u=c['u']
for k in ('v1','v2'):
 for basis in ([u,u*u],[u,u*u,u**3],[u,u*u,u**3,u**4],[u,u*u,u**3,np.sin(3*u),np.cos(3*u)]):
  P=np.column_stack([np.ones(len(u)),*basis]); q=np.linalg.lstsq(P,c[k],rcond=None)[0]; print(k,P.shape[1],np.sqrt(np.mean((P@q-c[k])**2)),np.round(q[1:],2).tolist())
splits=[('public',N[52:]),('edge',np.argsort(np.linalg.norm(x,axis=1))[-12:])]+[(str(i),v) for i,v in enumerate(np.array_split(np.random.default_rng(427).permutation(len(x)),4))]
splits += [('edge-'+str(j)+str(d),np.argsort(x[:,j]*d)[-12:]) for j in range(4) for d in [-1,1]]
models=[]
x1=x-x.mean(0); w1=w-w.mean(0)
dot=x@x.T; quad=1+dot+(dot**2+(x*x)@(x*x).T)/2
for name,b in [('x',x1),('w',w1),('white',(x1@np.linalg.inv(np.linalg.cholesky(np.cov(x1.T))).T)*.55)]:
 D=(b[:,None,:]-b[None,:,:])**2; g=b@b.T
 for e in (.2,.4,.6,1.):
  for a in (0,.1,.4,1):
   for label,k in [('rbf',np.exp(-e*D.sum(2))),('add',np.exp(-e*D).sum(2)),('pair',sum(np.exp(-e*(D[:,:,i]+D[:,:,j])) for i,j in itertools.combinations(range(4),2))),('matern',(1+np.sqrt(e*D.sum(2)))*np.exp(-np.sqrt(e*D.sum(2))))]:
    for reg in (.0003,.002,.01): models.append(((name,label,e,a,reg),quad+k+a*g**3,reg))
for label,K,reg in models:
 errors=[]
 for _,v in splits:
  t=np.setdiff1d(N,v); p=K[np.ix_(v,t)]@np.linalg.solve(K[np.ix_(t,t)]+reg*np.eye(len(t)),y[t]); errors.append(np.abs(y[v]-p).mean())
 scores=np.array(errors)
 if 'results' not in globals():results=[]
 results.append((label,scores))
for label,e in sorted(results,key=lambda a:max(a[1][:6]))[:12]: print(label,e.round(4))
print('all-slice robustness')
for label,e in sorted(results,key=lambda a:max(a[1]))[:10]: print(label,e.round(4))
