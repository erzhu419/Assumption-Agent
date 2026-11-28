import pickle,runpy,numpy as np,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; old=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py')
def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def poly(x,d,lo=0):
 return np.column_stack([np.prod(x[:,c],axis=1) for j in range(lo,d+1) for c in itertools.combinations_with_replacement(range(x.shape[1]),j)])
def loo(a,v):
 u,g,w=np.linalg.svd(a,full_matrices=False); u=u[:,g>g[0]*1e-10]; e=(v-u@(u.T@v))/np.maximum(1-(u*u).sum(1)[:,None],1e-10); return np.linalg.norm(e)/np.sqrt(e.size)
p=s['panel_a']; x,v=p['u'],p['v']; a=poly(x,1); w0,w=old['_calibration'](p); print('archive A',w0,np.round(w,4),np.round(np.sort(np.linalg.norm(x@w+w0-v,axis=1)),4)); rng=np.random.default_rng(521); best=(np.inf,None)
for _ in range(3000):
 i=rng.choice(len(x),5,False); w=ls(a[i],v[i]); e=np.linalg.norm(a@w-v,axis=1); score=np.median(e);
 if score<best[0]: best=score,w
print('robust A',np.round(best[1],4),np.round(np.sort(np.linalg.norm(a@best[1]-v,axis=1)),5))
p=s['panel_c']; x=p['u']; a=poly(x,1); q=poly(x,2,2); c=poly(x,3,3); pairs=list(itertools.combinations(range(4),2)); xx=np.column_stack([op(x[:,i],x[:,j]) for i,j in pairs for op in [np.add,np.subtract]]); z=np.c_[x,xx];
for name in ('v1','v2'):
 v=p[name]; print(name,'affine',loo(a,v),'quadratic',loo(np.c_[a,q],v),'even',loo(np.c_[a,q],v),'cubic',loo(np.c_[a,c],v)); print('quadratic coefficients',np.round(ls(np.c_[a,q],v)[5:],3));
 for typ in ('odd','odd2','even','abs','tanh','sin'):
  def g(y,k): return {'odd':lambda:y/(1+k*abs(y)),'odd2':lambda:y/(1+k*y*y),'even':lambda:y*y/(1+k*y*y),'abs':lambda:abs(y)/(1+k*abs(y)),'tanh':lambda:np.tanh(k*y),'sin':lambda:np.sin(k*y)}[typ]()
  def opt(f):
   grid=np.geomspace(.02,5,35); i=np.argmin([f(k) for k in grid]); return minimize_scalar(f,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded')
  for bname,b in [('a',a),('q',np.c_[a,q])]:
   res=opt(lambda k:loo(np.c_[b,g(x,k)],v)); print(typ,bname,round(res.x,4),round(res.fun,6),end='; ')
  print()
  if typ in ('odd','odd2','even','tanh'):
   for j in range(4):
    vals=[]
    for i in range(z.shape[1]):
     res=opt(lambda k:loo(np.c_[a,g(z[:,i],k)],v[:,j:j+1])); vals.append((round(res.fun,5),i,round(res.x,4)))
    print(j,sorted(vals)[:3],end='; ')
   print()
t=s['train']; x=t['x']; y=np.c_[t['y'],t['traj']]; tr=np.arange(52); te=np.arange(52,72)
for j in (1,2,3):
 f=poly(x,j); w=ls(f[tr],y[tr]); e=abs(f[te]@w-y[te]); print('direct polynomial',j,e[:,0].mean(),e[:,1:].mean())
