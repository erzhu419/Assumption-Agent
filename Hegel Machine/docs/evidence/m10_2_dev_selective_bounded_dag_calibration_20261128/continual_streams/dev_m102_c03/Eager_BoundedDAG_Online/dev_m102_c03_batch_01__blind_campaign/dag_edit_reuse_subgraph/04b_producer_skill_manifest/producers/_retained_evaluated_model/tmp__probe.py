import numpy as np,pickle,runpy
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=np.c_[s['train']['y'],s['train']['traj']]; n=len(x)
sp=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; old=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c03_batch_00_solver.py')['_panel_pred']
def aff(u): return np.c_[np.ones(len(u)),u]
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def rob(u,v):
 h=aff(u); k=np.ones(len(u),bool)
 for _ in range(8):
  r=np.linalg.norm(v-h@fit(h[k],v[k]),axis=1); k=r<max(.4,2.5*np.median(r))
 return fit(h[k],v[k]),r,k
coords={'raw':x}; p=s['panel_a']; a,r,k=rob(p['u'],p['v']); print('A coef',a.round(4),'residuals',np.sort(r).round(3),'archived keep',sum(np.linalg.norm(p['v']-aff(p['u'])@fit(aff(p['u']),p['v']),axis=1)<3))
u,v=p['u'][k],p['v'][k]; d=np.array([fit(aff(u)[:,[0,j+1]],v[:,j]) for j in range(4)]); print('A diagonal',d.round(5)); z=(x-d[:,0])/d[:,1]; coords['diag']=z; coords['full']=(x-a[0])@np.linalg.inv(a[1:]);
for pn in ['panel_b','panel_c']:
 p=s[pn]; u=p['u']; h=aff(u)
 for name in p.keys()-{'u'}:
  v=p[name]; b=fit(h,v); print(pn,name,'linear',b.round(3),'sv',np.linalg.svd(b[1:],compute_uv=False).round(3))
  for tag,func in [('quad',lambda z:z*z),('cube',lambda z:z**3),('sat',lambda z:np.expm1(-.2962*z*z)),('tanh',np.tanh),('sin',np.sin)]:
   g=np.c_[h,func(u)]; loo=(v-g@fit(g,v))/(1-np.sum(np.linalg.svd(g,full_matrices=False)[0]**2,axis=1))[:,None]; print(tag,round(np.mean(abs(loo)),5),end='; ')
  print()
  w=fit(np.c_[h,np.expm1(-.2962*u*u)],v); coords[pn+name]=z+(np.c_[aff(z),np.expm1(-.2962*z*z)]@w-aff(z)@b)@np.linalg.inv(b[1:]);
def quad(z): return np.c_[z,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]]
def poly(z,d):
 from itertools import combinations_with_replacement
 return np.column_stack([np.prod(z[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(4),k)])
folds=[np.arange(52,72),*np.array_split(np.random.default_rng(12).permutation(n),4)]; idx=[np.setdiff1d(np.arange(n),j) for j in folds]; sparse=[]; archive=[]
for i,j in zip(idx,folds):
 st=dict(s,train=dict(x=x[i],y=y[i,0],traj=y[i,1:]),test={'x':x[j]}); o=sp({'__input__':st}); sparse.append(np.c_[o['y_hat'],o['state_pred']]); archive.append(old(st,1.,.3,'sat'))
wt=np.r_[.5,np.repeat(.65,8)];
def score(p):
 out=[]
 for j,a,b in zip(folds,sparse,p):
  e=np.abs((1-wt)*a+wt*b-y[j]); out.append([e[:,0].mean(),e[:,1:].mean()])
 return np.array(out)
print('sparse',score(sparse).round(4)); print('archive',score(archive).round(4)); scores=[]
for cn,z in coords.items():
 for fn,f in [('quad',quad(z)),('tanh',np.c_[quad(z),quad(np.tanh(z))]),('poly3',poly(z,3)),('fourier',np.c_[quad(z),np.sin(z),np.cos(z),np.sin(2*z),np.cos(2*z)]),('raw+tanh',np.c_[quad(x),quad(np.tanh(z))])]:
  for alpha in [.003,.03,.3,1,3,10]:
   pred=[]
   for i,j in zip(idx,folds):
    zz=(f-f[i].mean(0))/np.maximum(f[i].std(0),1e-7); q=zz[i]; m=y[i].mean(0); pred.append(zz[j]@np.linalg.solve(q.T@q+alpha*np.eye(q.shape[1]),q.T@(y[i]-m))+m)
   e=score(pred); scores.append((e.max(0).max(),cn,fn,alpha,e))
for _,cn,fn,al,e in sorted(scores,key=lambda a:a[0])[:22]: print(cn,fn,al,e.round(4).tolist())
