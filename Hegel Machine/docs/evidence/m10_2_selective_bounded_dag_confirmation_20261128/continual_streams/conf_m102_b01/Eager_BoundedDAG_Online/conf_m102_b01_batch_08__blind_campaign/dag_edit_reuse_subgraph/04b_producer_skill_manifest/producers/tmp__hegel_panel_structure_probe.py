import numpy as np,pickle,runpy,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); w=a['run_node']({'__input__':s})['w']; ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
def pol(x,d):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ix],axis=1) for m in range(1,d+1) for ix in itertools.combinations_with_replacement(range(x.shape[1]),m)])
def fits(d,v):
 h=np.linalg.pinv(d); c=h@v;r=d@c-v; return np.mean(r*r)**.5,np.mean((r/np.maximum(1-(d*h.T).sum(1),1e-8)[:,None])**2)**.5,c
b=s['panel_b']; c=s['panel_c']; u=(w-pol(b['u'],1).__class__([]) if False else w-ls(pol(b['u'],1),b['v'])[0])@np.linalg.inv(ls(pol(b['u'],1),b['v'])[1:]); print('u range',u.min(0),u.max(0));
for name,x,v in [('b',b['u'],b['v']),('binv',b['v'],b['u']),('c2',c['u'],c['v2']),('c2from1',c['v1'],c['v2']),('c2inv',c['v2'],c['u'])]:
 print(name)
 for deg in [1,2]:
  d=pol(x,deg)
  for k,fn,lim in [('id',lambda x,g:x,(1,2)),('sinh',lambda x,g:np.sinh(g*x)/g,(0.001,7)),('tan',lambda x,g:np.tan(g*x)/g,(0.001,1.55/abs(v).max())),('atanh',lambda x,g:np.arctanh(g*x)/g,(0.001,.999/abs(v).max())),('cube',lambda x,g:x+g*x**3,(0,50)),('asinh',lambda x,g:np.arcsinh(g*x)/g,(.001,10)),('atan',lambda x,g:np.arctan(g*x)/g,(.001,10))]:
   ans=minimize_scalar(lambda g:fits(d,fn(v,g))[1],bounds=lim,method='bounded');print(deg,k,round(ans.x,5),np.round(fits(d,fn(v,ans.x))[:2],6))
train=s['train'];n=len(train['y']);t=np.arange(n);splits=[(t[:52],t[52:])]+[(np.delete(t,i),i) for i in np.array_split(np.random.default_rng(57).permutation(t),4)]
def evaluate(fs,ys=train['y']):
 out=[]
 for tr,te in splits:
  sc=fs[tr].std(0);sc[sc<1e-10]=1; ff=fs/sc; out.append([np.mean(abs(ff[te]@np.linalg.solve(ff[tr].T@ff[tr]+np.eye(ff.shape[1])*r,ff[tr].T@ys[tr])-ys[te])) for r in [.0001,.03,.3,2.]])
 return np.array(out)
x1=pol(u,2)@ls(pol(c['u'],2),c['v1']); x2=pol(u,2)@ls(pol(c['u'],2),c['v2']);
rows=[]
for name,z in [('u',u),('w',w),('c1',x1),('c2',x2),('c12',np.c_[x1,x2])]:
 for k,fn in [('id',lambda x:x),('atan',np.arctan),('asinh',np.arcsinh),('tanh',np.tanh),('inv3',lambda x:np.sinh(np.arcsinh(x)/3))]:
  for gain in [.4,1,2,4,7]:
   f=fn(gain*z);
   for d in [1,2,3] if z.shape[1]==4 else [1,2]:
    for ext in [False,True]:
     v=pol(f,d);v=np.c_[v,u] if ext else v
     e=evaluate(v);r=np.argmin(e[1:].mean(0));rows.append((e[1:,r].mean(),e[:,r].round(4),name,k,gain,d,ext,r))
for row in sorted(rows,key=lambda x:x[0])[:24]: print(row)
print('traj');
for name,z in [('u',u),('c1',x1),('c2',x2),('c12',np.c_[x1,x2])]:
 for d in [1,2]:
  e=evaluate(pol(z,d),train['traj']);print(name,d,e[:,e[1:].mean(0).argmin()].round(4))
