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
from scipy.optimize import least_squares
from scipy.special import expit
from itertools import combinations_with_replacement
p=s['panel_c']; u=p['u']; h=aff(u)
for typ in ['sin','cos','tanh','cos2','rational','sin+lin','cos+lin','wave']:
 def fun(t,xx):
  a=xx@t[:4]+t[4]; c=t[5:]
  if typ=='sin': return c[0]+c[1]*np.sin(a)
  if typ=='cos': return c[0]+c[1]*np.cos(a)
  if typ=='tanh': return c[0]+c[1]*np.tanh(a)
  if typ=='cos2': return c[0]+c[1]*np.cos(a)**2
  if typ=='rational': return c[0]+c[1]/(1+a*a)
  if typ=='sin+lin': return c[0]+c[1]*np.sin(a)+xx@c[2:6]
  if typ=='cos+lin': return c[0]+c[1]*np.cos(a)+xx@c[2:6]
  return c[0]+c[1]*np.sin(a)+c[2]*np.cos(2*a)
 rr=np.random.default_rng(23); out=[]
 for v in p['v1'].T:
  opts=[least_squares(lambda t:fun(t,u)-v,rr.normal(size={'sin+lin':11,'cos+lin':11,'wave':8}.get(typ,7))*rr.choice([1.,1.8,2.7]),max_nfev=75) for _ in range(75)]
  o=min(opts,key=lambda q:q.cost); out.append(o.x); print(typ,round(np.sqrt(2*o.cost/len(u)),5),o.x.round(3))
 if typ=='sin': coords['c1sin']=np.column_stack([fun(t,z) for t in out]); coords['c1arg']=np.column_stack([z@t[:4]+t[4] for t in out])
bp=s['panel_b']; b=fit(aff(bp['u']),bp['v']); coords['bforward']=aff(z)@b; coords['binverse']=(z-b[0])@np.linalg.inv(b[1:]);
for sgn in [-1,1]:
 for name in ['diag','bforward','binverse','c1arg','c1sin']:
  zz=coords[name]; coords[name+str(sgn)]=zz+sgn*.5*np.expm1(-.2962*zz*zz)
q=-np.expm1(-s['eval_times'][:,None]*np.array([.392,1.857])); a=fit(q,y[:,1:].T).T; print('time residual',np.mean(abs(a@q.T-y[:,1:]),axis=0)); print('official first/errors',np.c_[x[52:],y[52:,0],sparse[0][:,0]].round(3))
features={}
for name,zz in coords.items():
 for typ,f in [('quad',quad(zz)),('poly3',poly(zz,3)),('fourier',np.c_[quad(zz),np.sin(zz),np.cos(zz),np.sin(2*zz),np.cos(2*zz)]),('pair',np.c_[quad(zz),*[np.sin(k*(zz[:,i]+sg*zz[:,j])) for i,j in combinations(range(4),2) for sg in [-1,1] for k in [1,2]],*[np.cos(k*(zz[:,i]+sg*zz[:,j])) for i,j in combinations(range(4),2) for sg in [-1,1] for k in [1,2]]])]: features[(name,typ)]=f
out=[]
for (cn,fn),f in features.items():
 for alpha in [.003,.03,.3,1,3]:
  er=[]
  for i,j in zip(idx,folds):
   zz=(f-f[i].mean(0))/np.maximum(f[i].std(0),1e-7); f0=zz[i]; m=y[i].mean(0); yh=zz[j]@np.linalg.solve(f0.T@f0+alpha*np.eye(f0.shape[1]),f0.T@(y[i]-m))+m; e=np.abs(yh-y[j]); er.append([e[:,0].mean(),e[:,1:].mean()]);
  er=np.array(er); out.append((er[:,0].mean(),er[:,1].mean(),cn,fn,alpha,er))
for k0 in [0,1]:
 for a0 in sorted(out,key=lambda a:a[k0])[:12]: print(a0[:5],a0[-1].round(4).tolist())
