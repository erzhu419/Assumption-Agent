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
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
q=-np.expm1(-s['eval_times'][:,None]*np.array([.392,1.857])); amp=fit(q,y[:,1:].T).T; print('mode noise',np.mean(abs(amp@q.T-y[:,1:]),axis=0))
for col in range(3):
 t=np.c_[y[:,0],amp][:,col]; print('target',col,np.std(t));
 for deg in [1,2,3,4]:
  f=aff(poly(z,deg)); u=np.linalg.svd(f,full_matrices=False)[0]; r=t-f@fit(f,t); print(deg,np.std(r),np.mean(abs(r/np.maximum(1e-5,1-(u*u).sum(1)))))

def err(pred):
 return np.array([[abs(a[:,0]-y[j,0]).mean(),abs(a[:,1:]-y[j,1:]).mean()] for a,j in zip(pred,folds)])

def ridge(f,i,j,a):
 f=(f-f[i].mean(0))/np.maximum(f[i].std(0),1e-7); h=f[i]; m=y[i].mean(0); return f[j]@h.T@np.linalg.solve(h@h.T+a*np.eye(len(i)),y[i]-m)+m
out=[]
for nm,u in [('raw',x),('diag',z),('fwd',aff(z)@b),('inv',(z-b[0])@np.linalg.inv(b[1:]))]:
 for v in [-1,-.5,-.25,0,.25,.5,1]:
  w=u+v*np.expm1(-.2962*u*u)
  for kernel in ['thin_plate_spline','cubic','quintic','gaussian','multiquadric','inverse_quadratic']:
   for degree in [1,2,3]:
    for eps in [.25,.5,1.,2.]:
     if kernel in ['thin_plate_spline','cubic','quintic'] and eps!=1: continue
     for reg in [.0001,.01,.1]:
      pred=[RBFInterpolator(w[i],y[i],kernel=kernel,degree=degree,epsilon=eps,smoothing=reg)(w[j]) for i,j in zip(idx,folds)]; e=err(pred); out.append((e,(nm,v,kernel,degree,eps,reg)))
  for a in [.005,.015,.05,.15,.5]:
   p3=poly(w,3)
   for nm2,f in [('poly',p3),('tanh',np.c_[p3,quad(np.tanh(w))]),('sin',np.c_[p3,quad(np.sin(w)),np.cos(w)]),('rational',np.c_[p3,quad(w/(1+.5*w*w))])]:
    e=err([ridge(f,i,j,a) for i,j in zip(idx,folds)]); out.append((e,(nm,v,nm2,a)))
for j in range(2):
 for e,k in sorted(out,key=lambda t: .45*t[0][:,j].max()+.55*t[0][:,j].mean())[:12]: print(j,k,e.round(4).tolist())
